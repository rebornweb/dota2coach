
import json
import httpx
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from pydantic import BaseModel, Field
from ollama import Client, ResponseError
import joblib

app = FastAPI()
client = Client(host="http://localhost:11434", timeout=120)
ml_model = joblib.load(Path(__file__).resolve().parent / "dota_ml.joblib")

# Allow local React development server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)



SYSTEM_PROMPT = """
You are a Dota 2 strategy coach.

Important:
- Turbo is a Dota 2 game mode, NOT a hero.
- Turbo has accelerated gold and experience.
- Item timings and unconventional builds differ from
  normal or ranked Dota.
- Dazzle can be experimented with as a right-click carry.
- Distinguish experimental builds from established ones.
- Ask about the game mode and enemy lineup when relevant.
- Never invent hero abilities or item effects.
- If uncertain about a patch, say so.


When recommending Dota 2 items:
- Only recommend items whose existence and effects
  you can establish from provided reference data.
- Do not invent item names, abilities or attributes.
- The ML model only predicts an item category.
- The ML prediction does not validate specific item choices.
- If current item data is unavailable, explain the
  limitation rather than inventing a complete build.


Give concise, practical strategy advice.
"""


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1)


class ItemRequest(ChatRequest):
    hp: float = Field(ge=0, le=100)
    enemy_burst: float = Field(ge=0, le=10)


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=20000)


class CoachRequest(ChatRequest):
    hp: float | None = Field(default=None, ge=0, le=100)
    enemy_burst: float | None = Field(default=None, ge=0, le=10)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=40)


PREDICTION_TOOL = {
    "type": "function",
    "function": {
        "name": "predict_build_priority",
        "description": (
            "Get an experimental Defense/Damage prediction for Dazzle in Turbo "
            "using the HP and enemy burst values provided by the player. "
            "Use only when the question needs this build-priority prediction."
        ),
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
}


def predict_priority(hp: float, enemy_burst: float) -> str:
    prediction = ml_model.predict([[hp, enemy_burst]])[0]
    return "Defense" if prediction == 0 else "Damage"


def ollama_chat(**kwargs):
    try:
        return client.chat(model="llama3.2:3b", options={"temperature": 0.3}, **kwargs)
    except ResponseError as exc:
        raise HTTPException(status_code=502, detail=(
            "Ollama could not complete the request. Check that llama3.2:3b is installed "
            "and your Ollama version supports tool calling."
        )) from exc
    except (ConnectionError, TimeoutError, httpx.TransportError) as exc:
        raise HTTPException(status_code=503, detail="Ollama is unavailable or timed out. Check that it is running.") from exc


@app.post("/chat")
def chat(request: CoachRequest):
    has_context = (
        request.hp is not None
        and request.enemy_burst is not None
    )

    messages = [{
        "role": "system",
        "content": SYSTEM_PROMPT + "\n" + (
            "The prediction tool is an optional experiment for Dazzle in Turbo only. "
            "Use it for relevant damage-versus-defense questions, not general questions. "
            "It uses synthetic training examples, not validated strategy. "
            "Never claim to have run it unless you receive a tool result. "
            "If HP or burst values are missing, ask for them when a prediction is needed. "
            "Explain tool results briefly and acknowledge the model's limitations."
        )
    }]

    # Preserve conversation history
    messages.extend(
        message.model_dump()
        for message in request.history
    )

    # Add current user question and game context
    messages.append({
        "role": "user",
        "content": request.prompt + (
            f"\n\nCurrent form values for Dazzle in Turbo: HP={request.hp}%, "
            f"enemy burst threat={request.enemy_burst}/10. "
            "These values apply only to the experimental prediction."
            if has_context else ""
        )
    })

    # First LLM call: Decide whether to use the ML tool
    response = ollama_chat(
        messages=messages,
        tools=[PREDICTION_TOOL] if has_context else []
    )

    category = None

    if response.message.tool_calls:
        messages.append(response.message)

        for call in response.message.tool_calls:

            # Reject unknown or unavailable tools
            if (
                call.function.name != "predict_build_priority"
                or not has_context
            ):
                result = {
                    "error": "Unknown or unavailable tool."
                }

            # Prevent duplicate ML predictions
            elif category is not None:
                result = {
                    "error": "Prediction already performed."
                }

            else:
                # Run specialised ML model once
                # Use validated form values, not LLM-generated values
                category = predict_priority(
                    request.hp,
                    request.enemy_burst
                )

                result = {
                    "prediction": category,
                    "hp": request.hp,
                    "enemy_burst": request.enemy_burst,
                    "limitation": (
                        "Synthetic training data; "
                        "not validated Dota strategy."
                    )
                }

            # Return tool result to Ollama
            messages.append({
                "role": "tool",
                "tool_name": call.function.name,
                "content": json.dumps(result)
            })

        # Second LLM call: Explain the ML prediction
        # Only one tool round is allowed
        response = ollama_chat(messages=messages)

    # Validate the final response
    if not response.message.content:
        raise HTTPException(
            status_code=502,
            detail="Ollama returned an empty answer. Please try again."
        )

    return {
        "answer": response.message.content,
        "ml_prediction": category
    }


@app.post("/recommend")
def recommend(request: ItemRequest):

    # STEP 1: Specialised ML prediction
    category = predict_priority(request.hp, request.enemy_burst)

    # STEP 2: Pass ML output to Ollama
    response = ollama_chat(
        messages=[
            {
                "role": "system",
                "content": (
                    SYSTEM_PROMPT + "\n"
                    "Explain experimental ML predictions clearly. "
                    "The ML model uses synthetic data, so do not "
                    "present its predictions as proven strategy. "
                    "Do not invent item mechanics."
                )
            },
            {
                "role": "user",
                "content": f"""
                Hero: Dazzle
                Game mode: Turbo
                Player question: {request.prompt}

                HP: {request.hp}%
                Enemy burst threat: {request.enemy_burst}/10

                Specialised ML prediction: {category}

                Explain the prediction briefly.
                """
            }
        ]
    )
    
    # STEP 3: Return both results to React
    return {
        "ml_prediction": category,
        "llm_explanation": response.message.content
    }

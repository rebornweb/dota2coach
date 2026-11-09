
import json
import httpx
import joblib
import pandas as pd

from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from ollama import Client, ResponseError

# Initialise FastAPI and Ollama
app = FastAPI()
client = Client(host="http://localhost:11434", timeout=120)

# Get the directory containing main.py
BASE_DIR = Path(__file__).resolve().parent

# MODEL 1: Item Priority Classifier
# Predicts Damage or Defense
ml_model = joblib.load(
    BASE_DIR / "dota_ml.joblib"
)

# MODEL 2: Hero Performance Regressor
# Predicts estimated Turbo win rate
hero_bundle = joblib.load(
    BASE_DIR / "ml" / "hero_model.joblib"
)

hero_model = hero_bundle["model"]
hero_features = hero_bundle["features"]

# Allow React to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)

# Your SYSTEM_PROMPT goes below this
# Followed by your request models, tools and endpoints


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
    hero: str = Field(default="Dazzle", min_length=1)
    game_mode: Literal["Turbo", "All Pick"] = "Turbo"
    hp: float | None = Field(default=None, ge=0, le=100)
    enemy_burst: float | None = Field(default=None, ge=0, le=10)
    history: list[HistoryMessage] = Field(
        default_factory=list, max_length=40
    )

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

    # Item ML only supports Dazzle in Turbo
    has_context = (
        request.hero == "Dazzle"
        and request.game_mode == "Turbo"
        and request.hp is not None
        and request.enemy_burst is not None
    )

    # System instructions
    messages = [{
        "role": "system",
        "content": SYSTEM_PROMPT + "\n" + (
            "You are assisting a player with Dota 2 strategy. "
            "The player's selected hero and game mode are supplied "
            "by the application and are authoritative. "
            "Always use the current selection rather than assuming "
            "the player is using a hero from earlier messages. "
            "Distinguish standard hero roles from experimental builds. "
            "The item prediction tool is experimental and supports "
            "Dazzle in Turbo only. "
            "Its training data is synthetic, not validated strategy. "
            "Never claim to have run a tool unless a result is provided. "
            "Do not invent hero abilities, items or statistics."
        )
    }]

    # Preserve conversation history
    messages.extend(
        message.model_dump()
        for message in request.history
    )

    # Always provide CURRENT React selection
    messages.append({
        "role": "system",
        "content": (
            "CURRENT GAME CONTEXT:\n"
            f"Selected hero: {request.hero}\n"
            f"Selected game mode: {request.game_mode}\n"
            "These values override previous selections in history. "
            "When asked 'What hero am I?', answer using the selected hero."
        )
    })

    # Add the current user question
    user_content = request.prompt

    # Add experimental ML inputs only when applicable
    if has_context:
        user_content += (
            f"\n\nExperimental Dazzle ML context: "
            f"HP={request.hp}%, "
            f"enemy burst threat={request.enemy_burst}/10."
        )

    messages.append({
        "role": "user",
        "content": user_content
    })

    # First LLM call: decide whether ML is needed
    response = ollama_chat(
        messages=messages,
        tools=[PREDICTION_TOOL] if has_context else []
    )

    category = None

    if response.message.tool_calls:
        messages.append(response.message)

        for call in response.message.tool_calls:

            if (
                call.function.name != "predict_build_priority"
                or not has_context
            ):
                result = {
                    "error": "Unknown or unavailable tool."
                }

            elif category is not None:
                # Prevent duplicate ML predictions
                result = {
                    "error": "Prediction already performed."
                }

            else:
                # Use validated React form values
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

        # Second LLM call: explain prediction
        response = ollama_chat(messages=messages)

    # Validate response
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

@lru_cache(maxsize=1)
def get_heroes():
    response = httpx.get(
        "https://api.opendota.com/api/heroStats",
        timeout=20
    )
    response.raise_for_status()
    return response.json()


def predict_hero_performance(hero_name: str):
    heroes = get_heroes()

    hero = next(
        (
            h for h in heroes
            if h["localized_name"].casefold() ==
               hero_name.strip().casefold()
        ),
        None
    )

    if hero is None:
        return {"error": f"Hero '{hero_name}' not found."}

    # Use the same features and order as training
    X = pd.DataFrame([{
        feature: hero[feature]
        for feature in hero_features
    }])

    predicted_rate = hero_model.predict(X)[0]

    actual_rate = (
        hero["turbo_wins"] / hero["turbo_picks"] * 100
        if hero["turbo_picks"] else None
    )

    return {
        "hero": hero["localized_name"],
        "estimated_turbo_win_rate": round(
            float(predicted_rate), 2
        ),
        "observed_turbo_win_rate": (
            round(actual_rate, 2)
            if actual_rate is not None else None
        ),
        "limitation": (
            "Experimental regression using hero attributes. "
            "Not a reliable forecast of match outcomes."
        )
    }
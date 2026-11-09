
import json
import re
import httpx
import joblib
import pandas as pd

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from ollama import Client, ResponseError
from dota_data import get_heroes

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
    allow_methods=["GET", "POST"],
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


HERO_PREDICTION_TOOL = {
    "type": "function",
    "function": {
        "name": "predict_hero_performance",
        "description": (
            "Estimate the currently selected hero's Turbo win rate using an experimental "
            "regression model and report observed Turbo wins/picks. Use for questions "
            "about this hero's Turbo performance, not identity or basic attributes. "
            "This is not a forecast of the player's match outcome."
        ),
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
}


def find_hero(hero_name: str):
    return next((hero for hero in get_heroes()
                 if hero["localized_name"].casefold() == hero_name.strip().casefold()), None)


def predict_priority(hp: float, enemy_burst: float) -> str:
    prediction = ml_model.predict([[hp, enemy_burst]])[0]
    return "Defense" if prediction == 0 else "Damage"


def is_tool_payload(content: str | None) -> bool:
    text = (content or "").strip()
    if text.startswith("```") and text.endswith("```"):
        text = "\n".join(text.splitlines()[1:-1]).strip()
    try:
        payload = json.loads(text)
    except ValueError:
        # A model can emit unescaped nested JSON or truncate the payload.
        # Recognise its tool envelope without parsing or executing the text.
        return bool(
            re.search(r'["\']name["\']\s*:\s*["\'][^"\']+["\']', text)
            and re.search(r'["\'](?:parameters|arguments)["\']\s*:', text)
            or re.search(r'["\']tool_calls["\']\s*:', text)
        )
    entries = payload if isinstance(payload, list) else [payload]
    return any(isinstance(entry, dict) and (
        "tool_calls" in entry or "function" in entry or
        ("name" in entry and ("parameters" in entry or "arguments" in entry))
    ) for entry in entries)


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



@app.get("/heroes")
def list_heroes():
    try:
        heroes = get_heroes()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return sorted({hero["localized_name"] for hero in heroes}, key=str.casefold)


@app.post("/chat")
def chat(request: CoachRequest):

    try:
        selected_hero = find_hero(request.hero)
        data_error = None if selected_hero else "Selected hero was not found in the hero data."
    except RuntimeError:
        selected_hero = None
        data_error = "Hero reference data is unavailable; do not invent statistics."

    # Keep the original field names: base stats are not total in-game stats.
    reference = ({key: selected_hero[key] for key in dict.fromkeys([
        "localized_name", "primary_attr", "attack_type", "roles", *hero_features,
        "base_health_regen", "base_mana", "base_mana_regen", "base_mr",
        "attack_rate", "attack_point", "day_vision", "night_vision",
    ]) if key in selected_hero} if selected_hero else {"error": data_error})
    hero_tool_available = request.game_mode == "Turbo" and selected_hero is not None

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
            "Do not invent hero abilities, items or statistics. "
            "Use the supplied hero reference fields for attribute questions. "
            "Base attributes are already provided; there is no get_base_attributes tool. "
            "Answer attribute questions directly from the reference data in plain language. "
            "Never print tool-call JSON as an answer or invent tool names. "
            "Base stats are not total in-game stats; do not equate base_health with total HP. "
            "Hero reference data may be cached and does not include ability or item mechanics. "
            "The hero performance tool is a separate experimental regression for Turbo only. "
            "Distinguish its estimated rate from the observed rate and from a personal win chance."
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
            "When asked 'What hero am I?', answer using the selected hero.\n"
            "Hero reference data (JSON): " + json.dumps(reference)
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
    tools = ([PREDICTION_TOOL] if has_context else [])
    if hero_tool_available:
        tools.append(HERO_PREDICTION_TOOL)
    response = ollama_chat(
        messages=messages,
        tools=tools
    )

    category = None
    hero_prediction = None

    if response.message.tool_calls:
        messages.append(response.message)

        for call in response.message.tool_calls:

            if call.function.name == "predict_hero_performance" and hero_tool_available:
                if hero_prediction is None:
                    hero_prediction = predict_hero_performance(request.hero, selected_hero)
                result = hero_prediction

            elif (
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

    # Some small models print a function request as content instead of tool_calls.
    # Do not execute this text; ask once for an answer using the existing data.
    if is_tool_payload(response.message.content):
        messages.append(response.message)
        messages.append({"role": "user", "content": (
            "Please answer my previous question in plain language using the supplied "
            "hero reference data and any actual tool results. Do not output a function "
            "request or tool JSON. If information is missing, say so."
        )})
        response = ollama_chat(messages=messages)
    if response.message.tool_calls or is_tool_payload(response.message.content):
        raise HTTPException(status_code=502, detail=(
            "The AI returned a tool request instead of an answer. Please try again."
        ))

    # Validate response
    if not response.message.content:
        raise HTTPException(
            status_code=502,
            detail="Ollama returned an empty answer. Please try again."
        )

    return {
        "answer": response.message.content,
        "ml_prediction": category,
        "hero_prediction": hero_prediction,
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

def predict_hero_performance(hero_name: str, hero=None):
    if hero is None:
        try:
            hero = find_hero(hero_name)
        except RuntimeError:
            return {"error": "Hero data is unavailable. Please try again later."}

    if hero is None:
        return {"error": f"Hero '{hero_name}' not found."}

    # Use the same features and order as training
    required = [*hero_features, "turbo_wins", "turbo_picks"]
    if any(not isinstance(hero.get(field), (int, float))
           or not pd.notna(hero[field]) for field in required):
        return {"error": "Hero data is incomplete; a performance prediction is unavailable."}
    if hero["turbo_picks"] < 0 or not 0 <= hero["turbo_wins"] <= hero["turbo_picks"]:
        return {"error": "Hero win/pick counts are invalid."}
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
        "game_mode": "Turbo",
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

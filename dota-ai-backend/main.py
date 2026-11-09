
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ollama import Client
import joblib
from pydantic import BaseModel

app = FastAPI()
client = Client(host="http://localhost:11434")

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

Give concise, practical strategy advice.
"""


class ChatRequest(BaseModel):
    prompt: str

@app.post("/chat")
def chat(request: ChatRequest):
    response = client.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "system",
                "content":SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": request.prompt
            }
        ],
        options={"temperature": 0.3}
    )

    return {"answer": response.message.content}

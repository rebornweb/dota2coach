# dota2coach
Dota 2 Coach

Run Ollama locally with `llama3.2:3b` installed (`ollama pull llama3.2:3b`).
Use an up-to-date Ollama installation with tool calling support.

In `dota-ai-backend`, install the Python dependencies:

```powershell
py -m pip install fastapi uvicorn ollama scikit-learn joblib httpx
py train_ml.py
py -m uvicorn main:app --reload
```

In a separate terminal, in `dota-ai-coach`:

```powershell
npm install
npm run dev
```

Open http://localhost:5173. Ask general Dota questions, or fill in the optional
Dazzle/Turbo HP and enemy burst values and ask whether to prioritise damage or
defense. Ollama can request the decision-tree prediction when relevant. React
displays the prediction only when the tool actually runs. The model is trained
on fictional examples and is not validated strategy advice.

`POST /chat` accepts `prompt`, optional `hp` and `enemy_burst`, and optional
`history` (up to 40 user/assistant messages). It returns `answer` and a nullable
`ml_prediction`. The original `POST /recommend` remains available for explicitly
running the prediction every time. API documentation: http://localhost:8000/docs.

Run backend checks from `dota-ai-backend` with `py -m unittest test_main`.
These use the real trained decision tree and mock Ollama responses.

# Dota 2 AI Coach

A local-first Dota 2 strategy chatbot and machine-learning proof of concept. A React/TypeScript interface sends questions to a Python FastAPI backend, which uses a locally running Llama 3.2 model through Ollama. Experimental scikit-learn models can supply structured predictions for the LLM to explain.

> **Project status: experimental PoC.** Model integration is the primary milestone. The current predictions are not validated item-buying or win-probability advice. LLM responses may invent hero abilities, items, statistics, or patch details.

## Architecture

```text
React + TypeScript + Vite (localhost:5173)
                  |
                  v
          FastAPI /chat (localhost:8000)
                  |
                  v
           Ollama / Llama 3.2 (localhost:11434)
                  |
         optional tool requests
                  |
           Python tool dispatcher
               /      \
              v        v
    Item-priority ML   Hero-performance ML
    (decision tree)    (random-forest regressor)
                            |
                            v
                  OpenDota /heroStats data
                  (if retrieval is configured)
                  |
           Results returned to Ollama
                  |
           Response returned to React
```

The hero-performance model and its OpenDota tool are extensions under development; check your current `main.py` to confirm they are loaded and registered. A downloaded or trained `.joblib` file alone does **not** make a tool available to Ollama.

## Features

- React chat UI with conversation history and a Clear chat control.
- Hero and game-mode selectors; the current UI selection should be sent with each request and take precedence over earlier chat context.
- Optional HP (%) and enemy-burst (0–10) fields for the Dazzle-in-Turbo item-priority experiment.
- FastAPI request validation and CORS for local Vite development.
- Locally hosted Ollama model (`llama3.2:3b`) for conversation and explanations.
- Ollama tool calling to request specialised predictions when relevant.
- Duplicate item-ML prediction protection within a request.
- API tests using FastAPI TestClient and mocked Ollama responses.

### Experimental models

| Component | Input | Output | Limitation |
| --- | --- | --- | --- |
| Item-priority decision tree (`dota_ml.joblib`) | HP and enemy-burst rating | `Defense` or `Damage` | Trained on a tiny, synthetic dataset; only an educational Dazzle/Turbo PoC. |
| Hero-performance regressor (`ml/hero_model.joblib`) | Hero attributes from OpenDota | Estimated aggregate Turbo win rate | Limited hero-level sample; not an individual match forecast. Requires training and tool wiring. |
| Llama 3.2 through Ollama | Conversation and tool outputs | Human-readable strategy discussion | May hallucinate or misinterpret facts; it is not retrained by chat or tool responses. |

## Requirements

- Windows 11 (or another OS supported by the tools)
- Node.js and npm
- Python 3.10+ (compatible with installed dependencies)
- Ollama and the `llama3.2:3b` model

### 1. Install and start Ollama

Install from <https://ollama.com/download> and download the model:

```powershell
ollama pull llama3.2:3b
ollama run llama3.2:3b
```

Check that the local server is available:

```powershell
curl.exe http://localhost:11434/api/tags
```

On Windows, the Ollama application commonly runs its server automatically. If it is not running, start Ollama or use `ollama serve` in a separate terminal.

### 2. Install Python dependencies

From the directory containing `main.py`:

```powershell
py -m pip install fastapi uvicorn ollama httpx pydantic scikit-learn pandas joblib
```

For a more reproducible project, pin the tested package versions in a `requirements.txt` and install with `py -m pip install -r requirements.txt`.

### 3. Train/load the ML model files

If `train_ml.py` exists at the repository root:

```powershell
py train_ml.py
```

This should generate `dota_ml.joblib`. If you have added `ml/train_hero_model.py`:

```powershell
py ml/train_hero_model.py
```

This should generate `ml/hero_model.joblib`. Your backend will fail at startup if it tries to load a model file that has not been created.

**Security:** Only load `.joblib` files you generated or trust. Joblib deserialisation can execute arbitrary code.

### 4. Start FastAPI

From the directory containing `main.py`:

```powershell
py -m uvicorn main:app --reload
```

API documentation: <http://localhost:8000/docs>

The `--reload` option detects Python source edits. If you retrain a `.joblib` model, restart FastAPI so it loads the updated model.

### 5. Start React

If the Vite project already exists:

```powershell
npm.cmd install
npm.cmd run dev
```

Open <http://localhost:5173> (or the URL Vite prints). `npm.cmd` is useful in Windows PowerShell when script execution policy blocks `npm.ps1`.

If starting from scratch:

```powershell
npm.cmd create vite@latest dota2coach -- --template react-ts
cd dota2coach
npm.cmd install
npm.cmd run dev
```

## API overview

### `POST /chat`

Sends a question, the selected game context, optional ML inputs, and recent conversation history.

```json
{
  "prompt": "What hero am I, and should I prioritise defense?",
  "hero": "Dazzle",
  "game_mode": "Turbo",
  "hp": 25,
  "enemy_burst": 9,
  "history": []
}
```

A response from the item-only implementation has this shape:

```json
{
  "answer": "The experimental classifier suggests considering defense...",
  "ml_prediction": "Defense"
}
```

If the hero-performance tool has been implemented, the API may additionally return `hero_prediction`. Tool calls are optional: general conversation should not automatically run ML. The precise response depends on the implementation and which tools Ollama requested.

### `POST /recommend`

A direct educational item-priority demo that runs the classifier and asks Ollama to explain the result:

```json
{
  "prompt": "Damage or defense for Dazzle?",
  "hp": 25,
  "enemy_burst": 9
}
```

## OpenDota integration and data freshness

OpenDota's public endpoint <https://api.opendota.com/api/heroStats> provides hero attributes, roles, and aggregate pick/win counts, including `turbo_picks` and `turbo_wins`. It is useful for grounding factual responses and as a data source for educational ML experiments.

- The endpoint is updated as OpenDota processes source data; do **not** assume every response is real-time or immediately reflects a new patch.
- The Python sample `@lru_cache(maxsize=1)` caches indefinitely for the lifetime of the process. Restart the backend or clear the cache to refresh it.
- A future improvement is a time-based cache (for example, 6 hours) with an explicit `fetched_at` timestamp in the API response.
- OpenDota hero-level attributes and aggregate Turbo win rates do **not** contain enough evidence to train a reliable next-item recommendation model.
- API retrieval is different from RAG with embeddings and a vector index. A full document-based RAG pipeline has not yet been established in this PoC.

OpenDota documentation: <https://docs.opendota.com/>.

## Running tests

If the existing tests are in `tests/`:

```powershell
py -m unittest discover -s tests -v
```

Current test coverage includes request validation, history handling, tool routing, duplicate/unknown tool handling, Ollama availability errors, and the direct recommendation endpoint. Mocked Ollama tests verify application behaviour; they do **not** measure real LLM accuracy or whether the live model consistently chooses the right tool.

## Known limitations

1. **Unreliable game knowledge:** Llama 3.2 may confidently invent item names, item effects, hero roles, and numeric stats. Fetch relevant facts from a validated data source instead of trusting unsupported claims.
2. **Synthetic labels:** The item-priority classifier learns hand-authored examples, not observed optimal build choices.
3. **Hero-performance prediction:** A model trained on hero-level aggregate stats is limited and should not be treated as match-specific win probability.
4. **Changing game state:** The selected hero and mode must be included in every request. FastAPI should validate them rather than silently fall back to Dazzle.
5. **Caching:** An in-process unlimited-lifetime OpenDota cache can become stale.
6. **Conversation history:** React-held history is not permanent and must be bounded by the model's actual context capacity.
7. **Local development:** CORS is configured for `http://localhost:5173`; server authentication and deployment security are not part of the current PoC.

## Roadmap

- [x] React chat UI + local FastAPI/Ollama communication
- [x] Experimental item-priority classifier and LLM handoff
- [x] Hero/game-mode selection and basic chat context
- [x] API-level tests with mocked Ollama responses
- [ ] Verify the hero-performance ML tool is registered and tested end-to-end (if not already done)
- [ ] Add a read-only `get_hero_stats` tool using OpenDota data
- [ ] Add a TTL cache and show the source fetch timestamp
- [ ] Validate hero names against OpenDota rather than relying only on a hardcoded dropdown
- [ ] Compare live LLM answers against sourced Dota facts
- [ ] Collect real match-level item and outcome data for more meaningful ML
- [ ] Evaluate predictions with held-out examples and clear baselines
- [ ] Consider patch-aware document retrieval / RAG
- [ ] Add optional streaming and persistent conversation storage

## Learning goal

The purpose of this project is to understand how a local LLM, specialised classical ML models, external data and tool-calling orchestration work together. **Successful model handoff is not the same as accurate Dota strategy advice.**

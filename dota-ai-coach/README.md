feat: implement Dota 2 AI Coach with local ML-to-LLM integration

- Built React and TypeScript chat interface using Vite.
- Integrated Python FastAPI with local Ollama (Llama 3.2).
- Trained a basic scikit-learn decision tree to predict Damage or Defense item priorities.
- Implemented LLM tool calling to invoke the specialised ML model when relevant.
- Added conversation history and optional gameplay inputs.
- Added input validation, error handling and duplicate prediction protection.
- Added API tests covering model handoff and validation.
- Successfully demonstrated the ML → LLM → React response pipeline.

**Current limitations:**
- ML classifier uses synthetic training data.
- LLM can generate inaccurate Dota item and strategy recommendations.
- No verified Dota knowledge source or RAG integration yet.

**Next steps:**
- Integrate validated Dota hero and item data.
- Improve recommendation accuracy and model evaluation.
- Explore RAG for patch-aware strategy recommendations.
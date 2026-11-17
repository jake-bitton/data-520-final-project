# IJC Policy Assistant

FastAPI and Streamlit provide a conversational policy assistant, with optional retrieval from the private IJC Qdrant corpus. Individual runs, prompts, source excerpts and usage remain in SQLite; conversation memory lasts only for the current Streamlit session.

## Setup (Git Bash on Windows)

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
cp .env.example .env  # only if .env does not already exist
```

The existing `.venv` may point at a removed Windows Store Python. If it cannot launch, create a new environment with a working Python installation. Keep existing `.env` credentials private.

Start with `LLM_MODE=fake`, which needs no API key. For OpenAI, set `LLM_MODE=real`, `OPENAI_API_KEY`, and `OPENAI_MODEL` to a text model available to your API account that supports streaming Chat Completions. Restart the backend after editing `.env`. API usage may incur charges. The application sends `max_completion_tokens` but does not send temperature, seed, or reasoning-effort overrides, since support differs by model. An unsupported or inaccessible model produces an error rather than being substituted.

In separate Git Bash terminals, with the environment activated:

```bash
uvicorn app.main:app --reload --port 8000
streamlit run ui/app.py --server.address 127.0.0.1
```

## Private retrieval

Connect WireGuard, set `RAG_ENABLED=true`, and set `QDRANT_READ_ONLY_API_KEY` privately in `.env`. Defaults are `QDRANT_URL=http://10.66.66.1:6333` and `QDRANT_COLLECTION=ijc_policy_with_pdfs_20261116`. The collection must already exist with 384-dimensional cosine vectors. No corpus ingestion occurs. The backend checks collection access on startup and health checks. If unreachable, inspect the active WireGuard route and deployed private server binding; the checked-in Qdrant Compose file maps only loopback. Never expose Qdrant publicly to solve this.

Queries use the pinned `intfloat/multilingual-e5-small` revision `614241f622f53c4eeff9890bdc4f31cfecc418b3` locally. The model may need an initial download. A follow-up in RAG mode makes an additional model call to rewrite the search query. Missing or failed retrieval is reported; it does not silently fall back to baseline. Empty search results are labeled.

When using OpenAI, retained conversation messages and selected document excerpts are sent to OpenAI. Query embeddings are computed locally and the vector database stays on the VPS. SQLite run records also contain rendered prompts and source excerpts: protect `data/app.db` accordingly.

## Chat and API

The chat retains the most recent 10 complete exchanges, up to 24,000 characters, as model context. Older exchanges remain visible in the session but may be omitted from model context. Failed and truncated answers are excluded. Start a new chat to change between RAG and baseline. Each response has its own citations and source list; run records remain accessible in History and via `/runs`.

`POST /generate` accepts the original `query` and `use_rag` fields, plus optional `history`: an alternating list of `{"role":"user"|"assistant", "content":"..."}` complete exchanges. It streams NDJSON `meta`, `sources`, `reasoning` (when available), `token`, and `done` or `error` events. `GET /health`, `GET /config`, `GET /runs`, and `GET /runs/{run_id}` provide status, non-secret configuration, and run history.

```bash
pytest -q
```

Tests use fakes/mocks and do not require an OpenAI key or the VPS. A real end-to-end check requires valid API credentials, model access, WireGuard, and the read-only Qdrant key.

# IJC Policy Assistant — baseline LLM pipeline + front-end (Phases 0–2)

Implements the *LLM Pipeline and Front-End Design (Phases 0–2)* doc for
**Strategies to Avoid Spilling Water**: a FastAPI backend that sends policy requests to a
locally served gpt-oss model through an OpenAI-compatible API, logs every run to SQLite,
and a Streamlit UI for asking questions and browsing run history. Retrieval (Phase 3) and
evaluation (Phase 5) plug into the seams already here.

```
Streamlit UI (ui/) --HTTP--> FastAPI (app/)
                               Orchestrator -> Retriever (NullRetriever now, ChromaRetriever in Phase 3)
                                            -> PromptBuilder (config/prompts/<version>.yaml)
                                            -> LLMClient (OpenAICompatClient | FakeLLMClient)
                                            -> RunsRepo (SQLite: runs, run_sources)
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env          # macOS/Linux: cp .env.example .env
```

`.env.example` starts in **fake mode** (`LLM_MODE=fake`), so everything works with no model.

## Run

Terminal 1 (backend):

```bash
uvicorn app.main:app --reload --port 8000
```

Terminal 2 (UI):

```bash
streamlit run ui/app.py
```

## Using the real model (gpt-oss-20b via Ollama)

```bash
ollama pull gpt-oss:20b
```

Then set `LLM_MODE=real` in `.env` and restart the backend. The 20B model needs roughly
16 GB of memory. For vLLM: `vllm serve openai/gpt-oss-20b`, set
`LLM_BASE_URL=http://localhost:8001/v1`, `LLM_MODEL=openai/gpt-oss-20b`, `LLM_SERVER=vllm`
(run vLLM on a port other than the API's 8000).

### Phase 0 smoke test (do this once per server)

```bash
curl http://localhost:11434/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"gpt-oss:20b\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"Say hi\"}],\"reasoning_effort\":\"low\"}"
```

Record where the reasoning text appears (`reasoning` vs `reasoning_content`), which usage
fields come back, and the finish reasons. If the field name differs, set `REASONING_FIELDS`.
If reasoning token counts don't change between `low` and `high`, set
`REASONING_IN_SYSTEM_PROMPT=true`.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | status, LLM reachable, RAG available |
| GET | `/config` | model, server, prompt version, defaults + limits |
| POST | `/generate` | NDJSON stream: `meta`, `sources`, `reasoning`*, `token`*, `done` or `error` |
| GET | `/runs?limit&offset` | run summaries |
| GET | `/runs/{run_id}` | full run record with sources |

Example: `curl -N -X POST localhost:8000/generate -H "Content-Type: application/json" -d "{\"query\":\"Propose a flood governance structure\"}"`

`use_rag=true` returns **400** until a corpus is connected; there is no silent fallback to the baseline.

## Tests

```bash
pytest -q
```

No test needs a model or GPU. Regenerate prompt golden files after an intentional template
change with `UPDATE_GOLDEN=1 pytest tests/test_prompt_builder.py`.

## Prompts

`config/prompts/v0_placeholder.yaml` is a **placeholder**. The prompt author adds new
immutable versions (e.g. `v1.yaml`) and the team switches with `PROMPT_VERSION`. Every run
stores the prompt version and the exact rendered messages. The with-RAG and without-RAG
prompts must differ only by the `{context_block}`.

## Phase 3 handoff

Implement `ChromaRetriever(Retriever)` in `app/retrieval/`, pass it to `create_app` /
the lifespan in `app/main.py`, and `rag_available` becomes true. The orchestrator,
`run_sources` logging, the `sources` event and the UI Sources expander already handle chunks.

import json


def stream_events(client, payload):
    r = client.post("/generate", json=payload)
    assert r.status_code == 200, r.text
    return [json.loads(line) for line in r.text.splitlines() if line.strip()]


def test_health_and_config(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["llm_reachable"] and h["rag_available"] is False
    c = client.get("/config").json()
    assert c["prompt_version"] == "v1_conversation"
    assert {"temperature", "max_tokens", "reasoning_effort", "k", "seed"} <= set(c["defaults"])
    assert c["limits"]["max_tokens"]["max"] > c["defaults"]["max_tokens"]


def test_generate_event_order_and_run_matches(client):
    events = stream_events(client, {"query": "Recommend a data sharing policy"})
    types = [e["type"] for e in events]
    assert types[:2] == ["meta", "sources"] and types[-1] == "done"
    order = {"meta": 0, "sources": 1, "reasoning": 2, "token": 3, "done": 4}
    assert [order[t] for t in types] == sorted(order[t] for t in types)

    run_id = events[0]["run_id"]
    run = client.get(f"/runs/{run_id}").json()
    assert run["output"] == "".join(e["text"] for e in events if e["type"] == "token")
    assert run["reasoning"] == "".join(e["text"] for e in events if e["type"] == "reasoning")
    assert run["status"] == "completed"
    assert run["sources"] == []

    runs = client.get("/runs").json()
    assert runs[0]["run_id"] == run_id


def test_use_rag_without_corpus_is_400(client):
    r = client.post("/generate", json={"query": "q", "use_rag": True})
    assert r.status_code == 400
    assert "RAG is not available" in r.json()["detail"]
    assert client.get("/runs").json() == []  # no run created


def test_invalid_params_422(client):
    assert client.post("/generate", json={"query": "q", "temperature": 5}).status_code == 422
    assert client.post("/generate", json={"query": ""}).status_code == 422
    assert client.post("/generate", json={"query": "x" * 4001}).status_code == 422
    assert client.get("/runs").json() == []


def test_unknown_run_404(client):
    assert client.get("/runs/nope").status_code == 404


def test_delete_run_removes_sources_and_preserves_other_runs(client):
    from app.db.connection import connect
    from app.schemas import Chunk

    first = stream_events(client, {"query": "first"})[0]["run_id"]
    other = stream_events(client, {"query": "other"})[0]["run_id"]
    repo = client.app.state.repo
    repo.insert_sources(first, [Chunk(chunk_id="a", doc_id="d", text="source")])
    repo.insert_sources(other, [Chunk(chunk_id="b", doc_id="d", text="other source")])

    assert client.delete(f"/runs/{first}").status_code == 204
    assert client.get(f"/runs/{first}").status_code == 404
    assert client.delete(f"/runs/{first}").status_code == 404
    assert client.get(f"/runs/{other}").json()["sources"][0]["text"] == "other source"
    with connect(repo.db_path) as conn:
        assert conn.execute("SELECT count(*) FROM run_sources WHERE run_id = ?", (first,)).fetchone()[0] == 0
    assert [run["run_id"] for run in client.get("/runs").json()] == [other]


def test_delete_running_run_returns_conflict(client):
    from app.schemas import Chunk
    repo = client.app.state.repo
    repo.create_run({"run_id": "active", "created_at": "2026-01-01", "status": "running", "query": "q", "use_rag": 0})
    repo.insert_sources("active", [Chunk(chunk_id="a", doc_id="d", text="source")])
    assert client.delete("/runs/active").status_code == 409
    run = client.get("/runs/active").json()
    assert run["status"] == "running" and len(run["sources"]) == 1

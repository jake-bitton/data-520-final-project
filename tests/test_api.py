import json


def stream_events(client, payload):
    r = client.post("/generate", json=payload)
    assert r.status_code == 200, r.text
    return [json.loads(line) for line in r.text.splitlines() if line.strip()]


def test_health_and_config(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["llm_reachable"] and h["rag_available"] is False
    c = client.get("/config").json()
    assert c["prompt_version"] == "v0_placeholder"
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

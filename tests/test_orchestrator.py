import asyncio

import pytest

from app.pipeline.orchestrator import RagUnavailableError
from app.schemas import GenerateRequest


def collect(orch, req):
    async def _go():
        return [ev async for ev in orch.run(req)]

    return asyncio.run(_go())


def test_success(make_orchestrator):
    orch = make_orchestrator()
    events = collect(orch, GenerateRequest(query="Propose a flood protocol"))
    types = [e["type"] for e in events]
    assert types[0] == "meta" and types[1] == "sources" and types[-1] == "done"
    assert "reasoning" in types and "token" in types
    # all reasoning events come before the first token
    assert max(i for i, t in enumerate(types) if t == "reasoning") < types.index("token")

    run = orch.repo.get_run(events[0]["run_id"])
    assert run["status"] == "completed"
    assert run["output"] == "".join(e["text"] for e in events if e["type"] == "token")
    assert run["reasoning"] and run["reasoning"] not in run["output"]
    assert run["rendered_prompt"][0]["role"] == "system"
    assert run["prompt_version"] == "v0_placeholder"
    assert run["app_git_sha"] == "test"
    assert run["finish_reason"] == "stop"


def test_llm_failure(make_orchestrator):
    orch = make_orchestrator("fail")
    events = collect(orch, GenerateRequest(query="q"))
    assert events[-1]["type"] == "error"
    run = orch.repo.get_run(events[0]["run_id"])
    assert run["status"] == "failed"
    assert "configured to fail" in run["error"]
    assert run["reasoning"]  # partial reasoning kept


def test_truncation_via_low_max_tokens(make_orchestrator):
    orch = make_orchestrator()
    events = collect(orch, GenerateRequest(query="q", max_tokens=20))
    done = events[-1]
    assert done["type"] == "done" and done["status"] == "truncated" and done["finish_reason"] == "length"
    run = orch.repo.get_run(done["run_id"])
    assert run["status"] == "truncated"
    assert run["output"] == ""  # whole budget spent reasoning


def test_truncate_behavior(make_orchestrator):
    orch = make_orchestrator("truncate")
    done = collect(orch, GenerateRequest(query="q"))[-1]
    assert done["status"] == "truncated"


def test_empty_content(make_orchestrator):
    orch = make_orchestrator("empty")
    events = collect(orch, GenerateRequest(query="q"))
    assert events[-1]["type"] == "done"
    assert not [e for e in events if e["type"] == "token"]
    assert orch.repo.get_run(events[0]["run_id"])["output"] == ""


def test_client_disconnect_keeps_partial_output(make_orchestrator):
    orch = make_orchestrator()

    async def _go():
        gen = orch.run(GenerateRequest(query="q"))
        run_id, tokens = None, 0
        async for ev in gen:
            if ev["type"] == "meta":
                run_id = ev["run_id"]
            if ev["type"] == "token":
                tokens += 1
                if tokens == 3:
                    break
        await gen.aclose()
        return run_id

    run = orch.repo.get_run(asyncio.run(_go()))
    assert run["status"] == "failed"
    assert run["error"] == "client_disconnected"
    assert run["output"]  # partial answer kept


def test_rag_unavailable_raises(make_orchestrator):
    orch = make_orchestrator()
    with pytest.raises(RagUnavailableError):
        orch.check_request(GenerateRequest(query="q", use_rag=True))


def test_overrides_logged(make_orchestrator):
    orch = make_orchestrator()
    events = collect(orch, GenerateRequest(query="q", temperature=0.3, seed=7, reasoning_effort="high"))
    run = orch.repo.get_run(events[0]["run_id"])
    assert (run["temperature"], run["seed"], run["reasoning_effort"]) == (0.3, 7, "high")

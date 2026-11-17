import asyncio
from types import SimpleNamespace as NS

import pytest
import httpx

from app.pipeline.orchestrator import retained_history
from app.retrieval.qdrant import QdrantRetriever, REVISION
from app.schemas import ChatMessage, GenerateRequest
from app.settings import Settings


def test_history_validation_and_limits():
    with pytest.raises(ValueError):
        GenerateRequest(query="q", history=[{"role": "system", "content": "bad"}])
    history = [ChatMessage(role=role, content=str(i)) for i in range(12) for role in ("user", "assistant")]
    kept, trimmed = retained_history(history)
    assert trimmed and len(kept) == 20 and kept[0]["content"] == "2"
    history = [ChatMessage(role=role, content="x" * 13000) for role in ("user", "assistant")]
    assert retained_history(history) == ([], True)


def test_prompt_context_is_only_difference():
    from app.pipeline.prompt_builder import PromptBuilder
    from app.schemas import Chunk
    s = Settings(_env_file=None)
    b = PromptBuilder(s.prompts_dir, "v1_conversation")
    history = [{"role": "user", "content": "first"}, {"role": "assistant", "content": "answer"}]
    baseline = b.render("next", [], history=history)
    rag = b.render("next", [Chunk(chunk_id="1", doc_id="d", text="text")], history=history)
    assert baseline[:-1] == rag[:-1] and rag[-1]["content"].endswith(baseline[-1]["content"])


def test_qdrant_encoder_and_mapping():
    class Model:
        def tokenizer(self, text, **kwargs):
            assert text == "query: flood" and not kwargs["truncation"]
            return {"input_ids": [1, 2]}
        def encode(self, text, normalize_embeddings):
            assert normalize_embeddings
            return NS(tolist=lambda: [0.1] * 384)
    class Client:
        def get_collection(self, name):
            return NS(config=NS(params=NS(vectors=NS(size=384, distance="Cosine"))))
        def query_points(self, **kwargs):
            assert len(kwargs["query"]) == 384 and kwargs["limit"] == 3
            return NS(points=[NS(id="point", score=.8, payload={"artifact_sha256": "doc", "text": "evidence", "page_start": 2, "page_end": 3, "heading_path": ["A", "B"], "release_id": "r1"})])
    s = Settings(_env_file=None, qdrant_read_only_api_key="test")
    r = QdrantRetriever(s, Client(), lambda: Model())
    assert r.check()
    chunk = asyncio.run(r.retrieve("flood", 3))[0]
    assert (chunk.doc_id, chunk.location, chunk.section, r.release) == ("doc", "pp. 2–3", "A > B", "r1")
    assert len(REVISION) == 40


@pytest.mark.parametrize("error, expected", [
    (httpx.ConnectError("secret-key"), "connection failed or was refused"),
    (httpx.ConnectTimeout("secret-key"), "connection timed out"),
    (type("AuthError", (Exception,), {"status_code": 401})("secret-key"), "HTTP 401"),
    (type("MissingError", (Exception,), {"status_code": 404})("secret-key"), "HTTP 404"),
])
def test_qdrant_readiness_logs_safe_failure_and_recovers(caplog, error, expected):
    class Client:
        def __init__(self): self.failure = error
        def get_collection(self, name):
            if self.failure: raise self.failure
            return NS(config=NS(params=NS(vectors=NS(size=384, distance="Cosine"))))
    client = Client()
    settings = Settings(_env_file=None, qdrant_url="http://user:secret-key@127.0.0.1:6333/path?key=secret-key",
                        qdrant_read_only_api_key="secret-key")
    retriever = QdrantRetriever(settings, client)
    with caplog.at_level("INFO", logger="app.retrieval.qdrant"):
        assert not retriever.check()
        assert not retriever.check()
        assert len(caplog.records) == 1
        assert expected in caplog.text and "127.0.0.1:6333" in caplog.text
        assert "secret-key" not in caplog.text
        client.failure = None
        assert retriever.check()
        assert "collection and vector configuration verified" in caplog.text


def test_qdrant_check_logs_missing_key_and_incompatible_vectors(caplog):
    settings = Settings(_env_file=None)
    assert not QdrantRetriever(settings).check()
    assert "QDRANT_READ_ONLY_API_KEY is not configured" in caplog.text
    client = NS(get_collection=lambda name: NS(config=NS(params=NS(vectors=NS(size=5, distance="Cosine")))))
    settings = Settings(_env_file=None, qdrant_read_only_api_key="test")
    assert not QdrantRetriever(settings, client).check()
    assert "384-dimensional cosine" in caplog.text


def test_qdrant_search_logs_failure_without_query_or_secret(caplog):
    class Model:
        def tokenizer(self, text, **kwargs): return {"input_ids": [1]}
        def encode(self, text, normalize_embeddings): return NS(tolist=lambda: [0.1] * 384)
    class Client:
        def query_points(self, **kwargs): raise httpx.ConnectError("secret-key")
    settings = Settings(_env_file=None, qdrant_read_only_api_key="secret-key")
    retriever = QdrantRetriever(settings, Client(), lambda: Model())
    with pytest.raises(RuntimeError):
        asyncio.run(retriever.retrieve("private query text", 3))
    assert "Qdrant search failed" in caplog.text
    assert "secret-key" not in caplog.text and "private query text" not in caplog.text


def test_followup_records_rewrite_and_sources(make_orchestrator):
    class Retriever:
        available = True
        release = "release"
        async def retrieve(self, query, k):
            assert "flood" in query and "response" in query
            return []
    orch = make_orchestrator(retriever=Retriever())
    req = GenerateRequest(query="How about response?", use_rag=True, history=[
        {"role": "user", "content": "flood"}, {"role": "assistant", "content": "Prior answer"}])
    async def run():
        return [event async for event in orch.run(req)]
    events = asyncio.run(run())
    assert events[-1]["type"] == "done"
    assert any("No matching sources" in warning for warning in events[-1]["warnings"])
    saved = orch.repo.get_run(events[0]["run_id"])
    assert saved["retrieval_query"] == "flood How about response?"
    assert saved["retrieval_release"] == "release" and saved["rewrite_prompt"]
    assert saved["rendered_prompt"][1]["content"] == "flood"

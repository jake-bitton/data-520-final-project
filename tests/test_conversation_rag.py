import asyncio
from types import SimpleNamespace as NS

import pytest

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

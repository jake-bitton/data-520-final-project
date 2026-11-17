import asyncio
from types import SimpleNamespace

from openai.types.chat import ChatCompletionChunk

from app.llm.base import GenParams, LLMChunk, LLMFinal
from app.llm.openai_client import OpenAICompatClient
from app.settings import Settings


def _chunk(delta=None, finish=None, usage=None):
    choices = [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish}]
    return ChatCompletionChunk.model_validate(
        {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m", "choices": choices, "usage": usage}
    )


class _FakeStream:
    def __init__(self, chunks):
        self.chunks = chunks

    def __aiter__(self):
        async def gen():
            for c in self.chunks:
                yield c
        return gen()

    async def close(self):
        pass


def test_parses_reasoning_content_usage_and_finish(monkeypatch):
    chunks = [
        _chunk({"role": "assistant", "reasoning": "think "}),       # Ollama field name
        _chunk({"reasoning_content": "more"}),                      # vLLM field name
        _chunk({"content": "Hello "}),
        _chunk({"content": "world"}, finish="stop"),
        _chunk(usage={"prompt_tokens": 10, "completion_tokens": 7, "total_tokens": 17,
                      "completion_tokens_details": {"reasoning_tokens": 3}}),
    ]
    client = OpenAICompatClient(Settings(_env_file=None))

    async def fake_create(messages, params):
        return _FakeStream(chunks)

    monkeypatch.setattr(client, "_create", fake_create)

    async def run():
        params = GenParams(model="m", temperature=1.0, max_tokens=100, reasoning_effort="low")
        return [item async for item in client.stream([], params)]

    items = asyncio.run(run())
    reasoning = "".join(i.text for i in items if isinstance(i, LLMChunk) and i.kind == "reasoning")
    content = "".join(i.text for i in items if isinstance(i, LLMChunk) and i.kind == "content")
    final = items[-1]
    assert reasoning == "think more"
    assert content == "Hello world"
    assert isinstance(final, LLMFinal) and final.finish_reason == "stop"
    assert final.usage == {"prompt_tokens": 10, "completion_tokens": 7, "reasoning_tokens": 3}


def test_request_uses_portable_openai_parameters():
    settings = Settings(_env_file=None, llm_mode="real", openai_api_key="test", openai_model="model")
    client = OpenAICompatClient(settings)
    captured = {}
    async def create(**kwargs):
        captured.update(kwargs)
        return "stream"
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    params = GenParams(model="model", temperature=.5, max_tokens=100, reasoning_effort="high", seed=3)
    assert asyncio.run(client._create([{"role": "user", "content": "hello"}], params)) == "stream"
    assert captured["max_completion_tokens"] == 100
    assert not {"max_tokens", "temperature", "seed", "reasoning_effort", "extra_body"} & captured.keys()


def test_real_mode_requires_key_and_model():
    import pytest
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAICompatClient(Settings(_env_file=None, llm_mode="real"))


def test_ping_rejects_authentication_error(monkeypatch):
    class FakeHTTPClient:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def get(self, *args, **kwargs): return SimpleNamespace(is_success=False, status_code=401)
    monkeypatch.setattr("app.llm.openai_client.httpx.AsyncClient", FakeHTTPClient)
    client = OpenAICompatClient(Settings(_env_file=None, llm_mode="real", openai_api_key="test", openai_model="model"))
    assert not asyncio.run(client.ping())

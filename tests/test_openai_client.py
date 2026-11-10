import asyncio

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

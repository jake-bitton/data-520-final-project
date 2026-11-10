"""Fake LLM so UI work and tests never need a model or GPU (LLM_MODE=fake)."""
from __future__ import annotations

import asyncio
import re
from typing import AsyncIterator

from app.llm.base import GenParams, LLMChunk, LLMClient, LLMError, LLMFinal, StreamItem

_REASONING = (
    "The user wants a policy recommendation for transboundary water governance. "
    "I should consider the Boundary Waters Treaty of 1909, the role of the International Joint "
    "Commission, existing watershed boards, and the interests of Indigenous nations along the border. "
    "If sources are provided I should ground the answer in them and cite them by number."
)

_ANSWER = """**[FAKE MODE — not a real model response]**

### Recommendation
1. **Establish a watershed-level advisory board** with equal U.S. and Canadian membership and dedicated seats for Indigenous nations.
2. **Adopt shared monitoring standards** for water levels and quality so both countries report comparable data.
3. **Create an early-warning protocol** for floods and droughts with pre-agreed escalation steps to the IJC.

### Rationale
These measures build on existing IJC practice of binational boards and reference studies, and address gaps in data sharing and stakeholder inclusion.

*Request received:* "{query}" · *sources supplied:* {n_sources}
"""


class FakeLLMClient(LLMClient):
    """Streams a reasoning phase, then an answer. Each word counts as one token against max_tokens,
    so a low max_tokens naturally produces a truncated (finish_reason=length) run."""

    def __init__(self, behavior: str = "normal", delay_s: float = 0.02):
        self.behavior = behavior
        self.delay_s = delay_s

    async def _pause(self):
        if self.delay_s:
            await asyncio.sleep(self.delay_s)

    async def stream(self, messages: list[dict], params: GenParams) -> AsyncIterator[StreamItem]:
        user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        n_sources = len(re.findall(r"^\[\d+\]", user_msg, flags=re.M))
        query = user_msg.strip().splitlines()[-1][:120] if user_msg.strip() else ""

        reasoning_words = [w + " " for w in _REASONING.split(" ")]
        answer = _ANSWER.format(query=query, n_sources=n_sources)
        answer_words = [w + " " for w in answer.split(" ")]
        if self.behavior == "empty":
            answer_words = []

        budget = params.max_tokens
        if self.behavior == "truncate":
            budget = min(budget, len(reasoning_words) + 15)
        used = 0
        prompt_tokens = sum(len(m["content"].split()) for m in messages)

        for i, w in enumerate(reasoning_words):
            if self.behavior == "fail" and i == 5:
                raise LLMError("Fake LLM configured to fail (FAKE_BEHAVIOR=fail).")
            if used >= budget:
                break
            await self._pause()
            used += 1
            yield LLMChunk("reasoning", w)
        reasoning_used = used

        for w in answer_words:
            if used >= budget:
                break
            await self._pause()
            used += 1
            yield LLMChunk("content", w)

        truncated = used >= budget and used < len(reasoning_words) + len(answer_words)
        yield LLMFinal(
            finish_reason="length" if truncated else "stop",
            usage={"prompt_tokens": prompt_tokens, "completion_tokens": used, "reasoning_tokens": reasoning_used},
        )

    async def ping(self) -> bool:
        return True

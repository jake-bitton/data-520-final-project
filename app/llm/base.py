"""LLM client interface (design doc, Section 9.2)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Literal, Optional, Union


@dataclass
class GenParams:
    model: str
    temperature: float
    max_tokens: int
    reasoning_effort: str
    seed: Optional[int] = None


@dataclass
class LLMChunk:
    kind: Literal["reasoning", "content"]
    text: str


@dataclass
class LLMFinal:
    finish_reason: Optional[str]
    usage: dict = field(default_factory=dict)  # prompt_tokens, completion_tokens, reasoning_tokens


StreamItem = Union[LLMChunk, LLMFinal]


class LLMError(Exception):
    """Readable error raised by a client; its message is shown to the user."""


class LLMClient(ABC):
    @abstractmethod
    def stream(self, messages: list[dict], params: GenParams) -> AsyncIterator[StreamItem]:
        """Yield LLMChunk items (reasoning or content), then exactly one LLMFinal."""

    @abstractmethod
    async def ping(self) -> bool:
        """True if the model endpoint responds."""

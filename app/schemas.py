"""Data contracts (design doc, Section 5)."""
from __future__ import annotations

from typing import Optional, Literal

from pydantic import BaseModel, Field, model_validator

from app.settings import LIMITS, ReasoningEffort


class Chunk(BaseModel):
    """The unit retrieval returns. Populated in Phase 3; defined now so UI and logging handle it."""

    chunk_id: str
    doc_id: str
    text: str
    title: Optional[str] = None
    section: Optional[str] = None
    source_url: Optional[str] = None
    location: Optional[str] = None  # page number or heading path
    date: Optional[str] = None
    score: Optional[float] = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=24000)


class GenerateRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    use_rag: bool = False
    history: list[ChatMessage] = Field(default_factory=list, max_length=200)
    temperature: Optional[float] = Field(None, ge=LIMITS["temperature"]["min"], le=LIMITS["temperature"]["max"])
    max_tokens: Optional[int] = Field(None, ge=LIMITS["max_tokens"]["min"], le=LIMITS["max_tokens"]["max"])
    reasoning_effort: Optional[ReasoningEffort] = None
    k: Optional[int] = Field(None, ge=LIMITS["k"]["min"], le=LIMITS["k"]["max"])
    seed: Optional[int] = Field(None, ge=LIMITS["seed"]["min"], le=LIMITS["seed"]["max"])

    @model_validator(mode="after")
    def validate_history(self):
        if len(self.history) % 2 or any(m.role != ("user" if i % 2 == 0 else "assistant") for i, m in enumerate(self.history)):
            raise ValueError("history must contain complete user/assistant exchanges")
        return self

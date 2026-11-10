"""Data contracts (design doc, Section 5)."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

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


class GenerateRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    use_rag: bool = False
    temperature: Optional[float] = Field(None, ge=LIMITS["temperature"]["min"], le=LIMITS["temperature"]["max"])
    max_tokens: Optional[int] = Field(None, ge=LIMITS["max_tokens"]["min"], le=LIMITS["max_tokens"]["max"])
    reasoning_effort: Optional[ReasoningEffort] = None
    k: Optional[int] = Field(None, ge=LIMITS["k"]["min"], le=LIMITS["k"]["max"])
    seed: Optional[int] = Field(None, ge=LIMITS["seed"]["min"], le=LIMITS["seed"]["max"])


class RunSummary(BaseModel):
    run_id: str
    created_at: str
    status: str
    query: str
    use_rag: bool
    model: Optional[str] = None
    latency_ms: Optional[int] = None


class RunDetail(BaseModel):
    run_id: str
    created_at: str
    status: str
    batch_id: Optional[str] = None
    app_git_sha: Optional[str] = None
    query: str
    use_rag: bool
    model: Optional[str] = None
    llm_server: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    reasoning_effort: Optional[str] = None
    seed: Optional[int] = None
    k: Optional[int] = None
    prompt_version: Optional[str] = None
    rendered_prompt: Optional[Any] = None
    output: Optional[str] = None
    reasoning: Optional[str] = None
    error: Optional[str] = None
    finish_reason: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    retrieval_ms: Optional[int] = None
    latency_ms: Optional[int] = None
    sources: list[Chunk] = []

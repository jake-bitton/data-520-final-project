"""Retriever interface. Phase 3 adds ChromaRetriever (ChromaDB + Qwen3-Embedding-8B) behind it."""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.schemas import Chunk


class Retriever(ABC):
    @property
    @abstractmethod
    def available(self) -> bool:
        """Whether this retriever is backed by a real corpus (drives rag_available)."""

    @abstractmethod
    async def retrieve(self, query: str, k: int) -> list[Chunk]:
        ...

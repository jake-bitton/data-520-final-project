from __future__ import annotations

from app.retrieval.base import Retriever
from app.schemas import Chunk


class NullRetriever(Retriever):
    """Baseline condition: no retrieved context. Phases 1–2 ship only this."""

    @property
    def available(self) -> bool:
        return False

    async def retrieve(self, query: str, k: int) -> list[Chunk]:
        return []

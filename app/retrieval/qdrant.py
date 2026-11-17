"""Read-only retrieval against the existing private Qdrant collection."""
from __future__ import annotations

import asyncio
from functools import lru_cache

from app.retrieval.base import Retriever
from app.schemas import Chunk

MODEL = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"


@lru_cache(maxsize=1)
def encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL, revision=REVISION, device="cpu")


class QdrantRetriever(Retriever):
    def __init__(self, settings, client=None, model_loader=encoder):
        self.settings = settings
        self.client = client
        self.model_loader = model_loader
        self._available = False
        self.release = None

    @property
    def available(self):
        return self._available

    def check(self):
        self._available = False
        if not self.settings.qdrant_read_only_api_key:
            return False
        try:
            if self.client is None:
                from qdrant_client import QdrantClient
                self.client = QdrantClient(url=self.settings.qdrant_url,
                    api_key=self.settings.qdrant_read_only_api_key, timeout=5)
            collection = self.client.get_collection(self.settings.qdrant_collection)
            vectors = collection.config.params.vectors
            self._available = vectors.size == 384 and str(vectors.distance).lower().endswith("cosine")
        except Exception:
            self._available = False
        return self._available

    def _retrieve(self, query, k):
        model = self.model_loader()
        text = "query: " + query
        if len(model.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]) > 512:
            raise ValueError("Retrieval query exceeds the embedding model's 512-token limit.")
        vector = model.encode(text, normalize_embeddings=True).tolist()
        points = self.client.query_points(collection_name=self.settings.qdrant_collection,
                                           query=vector, limit=k, with_payload=True).points
        chunks = []
        releases = set()
        for point in points:
            p = point.payload or {}
            if p.get("release_id"):
                releases.add(str(p["release_id"]))
            start, end = p.get("page_start"), p.get("page_end")
            location = (f"p. {start}" if start == end or end is None else f"pp. {start}–{end}") if start else None
            heading = p.get("heading_path")
            if isinstance(heading, list):
                heading = " > ".join(str(x) for x in heading)
            chunks.append(Chunk(chunk_id=str(point.id), doc_id=str(p.get("artifact_sha256") or point.id),
                text=p.get("text") or "", title=p.get("title"), section=heading,
                source_url=p.get("source_url"), location=location, score=point.score))
        self.release = ", ".join(sorted(releases)) or None
        return chunks

    async def retrieve(self, query, k):
        try:
            return await asyncio.to_thread(self._retrieve, query, k)
        except ValueError:
            raise
        except Exception as exc:
            self._available = False
            raise RuntimeError("Qdrant retrieval failed; check the private connection and read-only key.") from exc

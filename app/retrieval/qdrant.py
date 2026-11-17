"""Read-only retrieval against the existing private Qdrant collection."""
from __future__ import annotations

import asyncio
import logging
from functools import lru_cache
from urllib.parse import urlsplit

import httpx

from app.retrieval.base import Retriever
from app.schemas import Chunk

MODEL = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
log = logging.getLogger(__name__)


def _endpoint(url):
    """Report only scheme, host and port, never credentials, paths or query strings."""
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        if not parsed.scheme or not host:
            return "<invalid URL>"
        if ":" in host:
            host = f"[{host}]"
        return f"{parsed.scheme}://{host}" + (f":{parsed.port}" if parsed.port else "")
    except ValueError:
        return "<invalid URL>"


def _failure(exc):
    """Classify failures without logging exception messages or response bodies."""
    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        status = getattr(current, "status_code", None)
        if isinstance(status, int):
            if status in (401, 403):
                return f"authentication or authorization failed (HTTP {status})"
            if status == 404:
                return "collection or endpoint not found (HTTP 404)"
            return f"server returned HTTP {status}"
        if isinstance(current, (httpx.TimeoutException, TimeoutError)):
            return "connection timed out"
        if isinstance(current, (httpx.ConnectError, ConnectionRefusedError)):
            return "connection failed or was refused"
        current = getattr(current, "source", None) or current.__cause__ or current.__context__
    return f"unexpected error ({type(exc).__name__})"


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
        self._last_check_result = None

    def _report_check(self, available, reason):
        result = (available, reason)
        if result != self._last_check_result:
            log.log(logging.INFO if available else logging.WARNING,
                    "Qdrant readiness: %s; endpoint=%s collection=%s; %s",
                    "available" if available else "unavailable", _endpoint(self.settings.qdrant_url),
                    self.settings.qdrant_collection, reason)
            self._last_check_result = result
        self._available = available
        return available

    @property
    def available(self):
        return self._available

    def check(self):
        self._available = False
        if not self.settings.qdrant_read_only_api_key:
            return self._report_check(False, "QDRANT_READ_ONLY_API_KEY is not configured")
        try:
            if self.client is None:
                from qdrant_client import QdrantClient
                self.client = QdrantClient(url=self.settings.qdrant_url,
                    api_key=self.settings.qdrant_read_only_api_key, timeout=5)
            collection = self.client.get_collection(self.settings.qdrant_collection)
            vectors = collection.config.params.vectors
            size = getattr(vectors, "size", None)
            distance = getattr(vectors, "distance", None)
            if size != 384 or not str(distance).lower().endswith("cosine"):
                return self._report_check(False, "collection must have a single 384-dimensional cosine vector")
            return self._report_check(True, "collection and vector configuration verified")
        except Exception as exc:
            return self._report_check(False, _failure(exc))

    def _retrieve(self, query, k):
        try:
            model = self.model_loader()
        except Exception as exc:
            log.error("Qdrant retrieval: query embedding model failed to load (%s)", type(exc).__name__)
            raise
        text = "query: " + query
        if len(model.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]) > 512:
            raise ValueError("Retrieval query exceeds the embedding model's 512-token limit.")
        vector = model.encode(text, normalize_embeddings=True).tolist()
        try:
            points = self.client.query_points(collection_name=self.settings.qdrant_collection,
                                               query=vector, limit=k, with_payload=True).points
        except Exception as exc:
            log.warning("Qdrant search failed: endpoint=%s collection=%s; %s",
                        _endpoint(self.settings.qdrant_url), self.settings.qdrant_collection, _failure(exc))
            raise
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
        log.info("Qdrant retrieval completed: collection=%s returned=%d", self.settings.qdrant_collection, len(chunks))
        return chunks

    async def retrieve(self, query, k):
        try:
            return await asyncio.to_thread(self._retrieve, query, k)
        except ValueError:
            raise
        except Exception as exc:
            self._available = False
            raise RuntimeError("Qdrant retrieval failed; check the private connection and read-only key.") from exc

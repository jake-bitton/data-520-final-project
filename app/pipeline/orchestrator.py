"""Pipeline orchestrator (design doc, Section 9.1).

Resolves config, creates the run record, retrieves (or not), builds the prompt,
streams from the LLM, and finalizes the record. Yields event dicts that the API
serializes as newline-delimited JSON:

  meta -> sources -> reasoning* -> token* -> done | error
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

from app.db.runs_repo import RunsRepo
from app.llm.base import GenParams, LLMChunk, LLMClient, LLMError, LLMFinal
from app.pipeline.prompt_builder import PromptBuilder
from app.retrieval.base import Retriever
from app.schemas import GenerateRequest
from app.settings import Settings

log = logging.getLogger(__name__)


class RagUnavailableError(Exception):
    pass


class Orchestrator:
    def __init__(
        self,
        settings: Settings,
        llm: LLMClient,
        retriever: Retriever,
        prompt_builder: PromptBuilder,
        repo: RunsRepo,
    ):
        self.settings = settings
        self.llm = llm
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.repo = repo
        self.semaphore = asyncio.Semaphore(settings.max_concurrent_generations)
        self.git_sha = settings.resolved_git_sha()

    @property
    def rag_available(self) -> bool:
        return self.retriever.available

    def check_request(self, req: GenerateRequest) -> None:
        # Never silently fall back to baseline: that would corrupt study data.
        if req.use_rag and not self.rag_available:
            raise RagUnavailableError(
                "use_rag=true was requested but RAG is not available (no corpus connected yet)."
            )

    def _db(self, fn, *args, warnings: list[str]) -> None:
        """Run a DB write; on failure log it and keep going so the user still gets the answer."""
        try:
            fn(*args)
        except Exception as e:  # noqa: BLE001
            log.exception("Database write failed")
            warnings.append(f"Run logging failed: {e}")

    async def run(self, req: GenerateRequest, batch_id: Optional[str] = None) -> AsyncIterator[dict]:
        self.check_request(req)
        s = self.settings
        t0 = time.perf_counter()
        warnings: list[str] = []

        # 1. Resolve config: request overrides, then server defaults (bounds enforced by Pydantic).
        params = GenParams(
            model=s.llm_model,
            temperature=req.temperature if req.temperature is not None else s.default_temperature,
            max_tokens=req.max_tokens if req.max_tokens is not None else s.default_max_tokens,
            reasoning_effort=req.reasoning_effort or s.reasoning_effort,
            seed=req.seed if req.seed is not None else s.default_seed,
        )
        k = req.k if req.k is not None else s.default_k
        run_id = str(uuid.uuid4())

        # 2. Insert the run row with status=running.
        self._db(
            self.repo.create_run,
            {
                "run_id": run_id,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "status": "running",
                "batch_id": batch_id,
                "app_git_sha": self.git_sha,
                "query": req.query,
                "use_rag": int(req.use_rag),
                "model": params.model,
                "llm_server": s.llm_server,
                "temperature": params.temperature,
                "max_tokens": params.max_tokens,
                "reasoning_effort": params.reasoning_effort,
                "seed": params.seed,
                "k": k if req.use_rag else None,
                "prompt_version": self.prompt_builder.version,
            },
            warnings=warnings,
        )
        yield {"type": "meta", "run_id": run_id}

        output_parts: list[str] = []
        reasoning_parts: list[str] = []
        finished = False
        try:
            # 3. Retrieve (Phase 1-2: NullRetriever only, so this branch is gated by check_request).
            sources = []
            retrieval_ms = None
            if req.use_rag:
                tr = time.perf_counter()
                sources = await self.retriever.retrieve(req.query, k)
                retrieval_ms = int((time.perf_counter() - tr) * 1000)
                self._db(self.repo.insert_sources, run_id, sources, warnings=warnings)
            yield {"type": "sources", "chunks": [c.model_dump() for c in sources]}

            # 4. Render the prompt and save it exactly as sent.
            messages = self.prompt_builder.render(req.query, sources, params.reasoning_effort)
            self._db(
                self.repo.update_run,
                run_id,
                {"rendered_prompt": json.dumps(messages, ensure_ascii=False), "retrieval_ms": retrieval_ms},
                warnings=warnings,
            )

            # 5. Stream from the LLM, accumulating reasoning and answer separately.
            final = LLMFinal(finish_reason=None)
            async with self.semaphore:
                async for item in self.llm.stream(messages, params):
                    if isinstance(item, LLMChunk):
                        if item.kind == "reasoning":
                            reasoning_parts.append(item.text)
                            yield {"type": "reasoning", "text": item.text}
                        else:
                            output_parts.append(item.text)
                            yield {"type": "token", "text": item.text}
                    elif isinstance(item, LLMFinal):
                        final = item

            # 6. Finalize.
            latency_ms = int((time.perf_counter() - t0) * 1000)
            status = "truncated" if final.finish_reason == "length" else "completed"
            self._db(
                self.repo.update_run,
                run_id,
                {
                    "status": status,
                    "output": "".join(output_parts),
                    "reasoning": "".join(reasoning_parts),
                    "finish_reason": final.finish_reason,
                    "prompt_tokens": final.usage.get("prompt_tokens"),
                    "completion_tokens": final.usage.get("completion_tokens"),
                    "reasoning_tokens": final.usage.get("reasoning_tokens"),
                    "latency_ms": latency_ms,
                },
                warnings=warnings,
            )
            finished = True
            yield {
                "type": "done",
                "run_id": run_id,
                "status": status,
                "usage": final.usage,
                "finish_reason": final.finish_reason,
                "latency_ms": latency_ms,
                "warnings": warnings,
            }

        except (asyncio.CancelledError, GeneratorExit):
            # 7a. Client disconnected mid-stream: keep partial output, mark failed.
            if not finished:
                self._fail(run_id, "client_disconnected", output_parts, reasoning_parts, t0, warnings)
            raise
        except Exception as e:  # noqa: BLE001
            # 7b. Any other failure: mark failed, store the error, emit an error event.
            msg = str(e) if isinstance(e, LLMError) else f"{type(e).__name__}: {e}"
            if not isinstance(e, LLMError):
                log.exception("Run %s failed", run_id)
            self._fail(run_id, msg, output_parts, reasoning_parts, t0, warnings)
            yield {"type": "error", "run_id": run_id, "message": msg, "warnings": warnings}

    def _fail(self, run_id, error, output_parts, reasoning_parts, t0, warnings) -> None:
        self._db(
            self.repo.update_run,
            run_id,
            {
                "status": "failed",
                "error": error,
                "output": "".join(output_parts),
                "reasoning": "".join(reasoning_parts),
                "latency_ms": int((time.perf_counter() - t0) * 1000),
            },
            warnings=warnings,
        )

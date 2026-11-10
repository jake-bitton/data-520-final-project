"""OpenAI-compatible streaming client for gpt-oss served by Ollama / vLLM / llama.cpp."""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

from app.llm.base import GenParams, LLMChunk, LLMClient, LLMError, LLMFinal, StreamItem
from app.settings import Settings

log = logging.getLogger(__name__)


class OpenAICompatClient(LLMClient):
    def __init__(self, settings: Settings):
        self.settings = settings
        # SDK retries disabled: we retry only connection errors and 5xx, never 4xx.
        self.client = AsyncOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=httpx.Timeout(settings.llm_timeout_s, connect=10.0),
            max_retries=0,
        )
        self.reasoning_fields = settings.reasoning_field_list

    async def _create(self, messages: list[dict], params: GenParams):
        kwargs = dict(
            model=params.model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            temperature=params.temperature,
            max_tokens=params.max_tokens,
            extra_body={"reasoning_effort": params.reasoning_effort},
        )
        if params.seed is not None:
            kwargs["seed"] = params.seed

        attempts = self.settings.llm_max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                return await self.client.chat.completions.create(**kwargs)
            except APITimeoutError as e:
                raise LLMError(
                    f"The LLM server at {self.settings.llm_base_url} timed out after "
                    f"{self.settings.llm_timeout_s:.0f}s."
                ) from e
            except APIConnectionError as e:
                if attempt == attempts:
                    raise LLMError(
                        f"Could not reach the LLM server at {self.settings.llm_base_url}. "
                        f"Is {self.settings.llm_server} running and is '{params.model}' pulled?"
                    ) from e
            except APIStatusError as e:
                if e.status_code < 500 or attempt == attempts:
                    raise LLMError(f"LLM server returned {e.status_code}: {e.message}") from e
            await asyncio.sleep(min(2**attempt, 8))
            log.warning("Retrying LLM request (attempt %d/%d)", attempt + 1, attempts)

    def _reasoning_text(self, delta) -> str | None:
        extra = getattr(delta, "model_extra", None) or {}
        for name in self.reasoning_fields:
            val = extra.get(name) if name in extra else getattr(delta, name, None)
            if val:
                return val
        return None

    async def stream(self, messages: list[dict], params: GenParams) -> AsyncIterator[StreamItem]:
        resp = await self._create(messages, params)
        finish_reason = None
        usage: dict = {}
        try:
            async for chunk in resp:
                if chunk.usage:
                    u = chunk.usage
                    usage = {"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens}
                    details = getattr(u, "completion_tokens_details", None)
                    if details is not None and getattr(details, "reasoning_tokens", None) is not None:
                        usage["reasoning_tokens"] = details.reasoning_tokens
                for choice in chunk.choices or []:
                    delta = choice.delta
                    if delta is not None:
                        r = self._reasoning_text(delta)
                        if r:
                            yield LLMChunk("reasoning", r)
                        if delta.content:
                            yield LLMChunk("content", delta.content)
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason
        except (APIConnectionError, httpx.HTTPError) as e:
            raise LLMError(f"Lost connection to the LLM server mid-stream: {e}") from e
        finally:
            await resp.close()
        yield LLMFinal(finish_reason=finish_reason, usage=usage)

    async def ping(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                r = await c.get(
                    self.settings.llm_base_url.rstrip("/") + "/models",
                    headers={"Authorization": f"Bearer {self.settings.llm_api_key}"},
                )
                return r.status_code < 500
        except Exception:
            return False

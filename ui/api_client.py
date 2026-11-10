"""Thin httpx wrapper around the backend API. The UI never imports pipeline code."""
from __future__ import annotations

import json
import os
from typing import Iterator

import httpx

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")


# One shared client: building a new one per call (SSL context, proxy lookup) costs ~0.5 s on Windows.
_client = httpx.Client(base_url=API_URL)


class BackendError(Exception):
    """A readable error for the UI."""


def _detail(resp: httpx.Response) -> str:
    try:
        body = resp.json()
    except ValueError:
        return resp.text
    detail = body.get("detail", body)
    if isinstance(detail, list):  # Pydantic 422
        return "; ".join(f"{'.'.join(str(p) for p in d.get('loc', [])[1:])}: {d.get('msg')}" for d in detail)
    return str(detail)


def _get(path: str, timeout: float = 10.0, **params):
    try:
        r = _client.get(path, params=params or None, timeout=timeout)
    except httpx.ConnectError as e:
        raise BackendError(f"Backend is not reachable at {API_URL}. Start it with `uvicorn app.main:app`.") from e
    except httpx.TimeoutException as e:
        raise BackendError(f"Backend at {API_URL} timed out.") from e
    if r.status_code >= 400:
        raise BackendError(f"{r.status_code}: {_detail(r)}")
    return r.json()


def health() -> dict:
    return _get("/health", timeout=8.0)


def config() -> dict:
    return _get("/config")


def list_runs(limit: int = 100, offset: int = 0) -> list[dict]:
    return _get("/runs", limit=limit, offset=offset)


def get_run(run_id: str) -> dict:
    return _get(f"/runs/{run_id}")


def generate_stream(payload: dict) -> Iterator[dict]:
    """Yield parsed NDJSON events from POST /generate."""
    timeout = httpx.Timeout(connect=5.0, read=600.0, write=30.0, pool=5.0)
    try:
        with _client.stream("POST", "/generate", json=payload, timeout=timeout) as r:
            if r.status_code >= 400:
                r.read()
                raise BackendError(f"{r.status_code}: {_detail(r)}")
            for line in r.iter_lines():
                if line.strip():
                    yield json.loads(line)
    except httpx.ConnectError as e:
        raise BackendError(f"Backend is not reachable at {API_URL}. Start it with `uvicorn app.main:app`.") from e
    except httpx.ReadTimeout as e:
        raise BackendError("Timed out waiting for the model. It may still be loading; try again.") from e

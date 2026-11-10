from __future__ import annotations

from fastapi import APIRouter, Request

from app.settings import LIMITS

router = APIRouter()


@router.get("/health")
async def health(request: Request):
    orch = request.app.state.orchestrator
    llm_ok = await orch.llm.ping()  # real ping: also warms a cold model server
    return {
        "status": "ok",
        "llm_reachable": llm_ok,
        "llm_mode": orch.settings.llm_mode,
        "rag_available": orch.rag_available,
    }


@router.get("/config")
def config(request: Request):
    orch = request.app.state.orchestrator
    s = orch.settings
    return {
        "model": s.llm_model,
        "llm_server": s.llm_server,
        "llm_mode": s.llm_mode,
        "prompt_version": orch.prompt_builder.version,
        "rag_available": orch.rag_available,
        "defaults": {
            "temperature": s.default_temperature,
            "max_tokens": s.default_max_tokens,
            "reasoning_effort": s.reasoning_effort,
            "k": s.default_k,
            "seed": s.default_seed,
        },
        "limits": LIMITS,
    }

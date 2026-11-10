from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.pipeline.orchestrator import RagUnavailableError
from app.schemas import GenerateRequest

router = APIRouter()


@router.post("/generate")
async def generate(req: GenerateRequest, request: Request):
    """Stream newline-delimited JSON events: meta, sources, reasoning*, token*, done|error."""
    orch = request.app.state.orchestrator
    try:
        orch.check_request(req)
    except RagUnavailableError as e:
        raise HTTPException(status_code=400, detail=str(e))

    async def event_stream():
        async for event in orch.run(req):
            yield json.dumps(event, ensure_ascii=False) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")

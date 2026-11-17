from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request, Response
from app.db.runs_repo import RunStillRunningError

router = APIRouter()


@router.get("/runs")
def list_runs(request: Request, limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
    return request.app.state.repo.list_runs(limit=limit, offset=offset)


@router.get("/runs/{run_id}")
def get_run(run_id: str, request: Request):
    run = request.app.state.repo.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return run


@router.delete("/runs/{run_id}", status_code=204)
def delete_run(run_id: str, request: Request):
    try:
        deleted = request.app.state.repo.delete_run(run_id)
    except RunStillRunningError:
        raise HTTPException(status_code=409, detail="Cannot delete a run while it is generating.")
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return Response(status_code=204)

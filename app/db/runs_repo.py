"""All SQLite reads and writes for runs and run_sources."""
from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path
from typing import Any, Optional

from app.db.connection import connect
from app.schemas import Chunk

_CHUNK_FIELDS = ("chunk_id", "doc_id", "text", "title", "section", "source_url", "location", "date", "score")


class RunsRepo:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def _conn(self):
        return closing(connect(self.db_path))

    def create_run(self, record: dict[str, Any]) -> None:
        cols = ", ".join(record)
        marks = ", ".join("?" for _ in record)
        with self._conn() as c, c:
            c.execute(f"INSERT INTO runs ({cols}) VALUES ({marks})", list(record.values()))

    def update_run(self, run_id: str, fields: dict[str, Any]) -> None:
        sets = ", ".join(f"{k} = ?" for k in fields)
        with self._conn() as c, c:
            c.execute(f"UPDATE runs SET {sets} WHERE run_id = ?", [*fields.values(), run_id])

    def insert_sources(self, run_id: str, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        rows = [(run_id, rank, *(getattr(ch, f) for f in _CHUNK_FIELDS)) for rank, ch in enumerate(chunks, 1)]
        with self._conn() as c, c:
            c.executemany(
                f"INSERT INTO run_sources (run_id, rank, {', '.join(_CHUNK_FIELDS)}) "
                f"VALUES (?, ?, {', '.join('?' for _ in _CHUNK_FIELDS)})",
                rows,
            )

    def list_runs(self, limit: int = 50, offset: int = 0) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT run_id, created_at, status, query, use_rag, model, latency_ms "
                "FROM runs ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [{**dict(r), "use_rag": bool(r["use_rag"])} for r in rows]

    def get_run(self, run_id: str) -> Optional[dict]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if row is None:
                return None
            srcs = c.execute(
                "SELECT * FROM run_sources WHERE run_id = ? ORDER BY rank", (run_id,)
            ).fetchall()
        run = dict(row)
        run["use_rag"] = bool(run["use_rag"])
        if run.get("rendered_prompt"):
            try:
                run["rendered_prompt"] = json.loads(run["rendered_prompt"])
            except json.JSONDecodeError:
                pass
        run["sources"] = [{f: s[f] for f in _CHUNK_FIELDS} for s in srcs]
        return run

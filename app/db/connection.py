from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Path) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        columns = {row[1] for row in conn.execute("PRAGMA table_info(runs)")}
        for name in ("retrieval_query", "retrieval_collection", "retrieval_release", "embedding_revision", "rewrite_prompt", "rewrite_output", "rewrite_usage"):
            if name not in columns:
                conn.execute(f"ALTER TABLE runs ADD COLUMN {name} TEXT")

CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  status TEXT NOT NULL,              -- running | completed | truncated | failed
  batch_id TEXT,                     -- used by the Phase 5 batch runner
  app_git_sha TEXT,
  query TEXT NOT NULL,
  use_rag INTEGER NOT NULL,
  model TEXT, llm_server TEXT,
  temperature REAL, max_tokens INTEGER, reasoning_effort TEXT, seed INTEGER, k INTEGER,
  prompt_version TEXT,
  rendered_prompt TEXT,              -- exact messages sent, as JSON
  output TEXT, reasoning TEXT, error TEXT, finish_reason TEXT,
  prompt_tokens INTEGER, completion_tokens INTEGER, reasoning_tokens INTEGER,
  retrieval_ms INTEGER, latency_ms INTEGER
);

CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at);

CREATE TABLE IF NOT EXISTS run_sources (
  run_id TEXT NOT NULL REFERENCES runs(run_id),
  rank INTEGER NOT NULL,
  chunk_id TEXT, doc_id TEXT, text TEXT, title TEXT, section TEXT,
  source_url TEXT, location TEXT, date TEXT, score REAL,
  PRIMARY KEY (run_id, rank)
);

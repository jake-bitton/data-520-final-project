"""Environment-driven configuration (design doc, Section 4)."""
from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent

# Server-side bounds for request overrides. /config exposes these so the UI can
# build its widgets from them.
LIMITS = {
    "temperature": {"min": 0.0, "max": 2.0},
    "max_tokens": {"min": 16, "max": 32768},
    "k": {"min": 1, "max": 50},
    "seed": {"min": 0, "max": 2**31 - 1},
    "reasoning_effort": ["low", "medium", "high"],
}

ReasoningEffort = Literal["low", "medium", "high"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # LLM endpoint (any OpenAI-compatible server: Ollama, vLLM, llama.cpp)
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    llm_model: str = "gpt-oss:20b"
    llm_server: str = "ollama"  # ollama | vllm | llamacpp — logged per run
    llm_mode: Literal["real", "fake"] = "real"

    # Where the server puts reasoning text in a streamed delta. Field names vary
    # by server/version; confirm in the Phase 0 smoke test and edit if needed.
    reasoning_fields: str = "reasoning,reasoning_content"
    # Fallback if the server ignores reasoning_effort: add "Reasoning: <effort>"
    # to the system message instead.
    reasoning_in_system_prompt: bool = False

    # Generation defaults (pinned and logged per run)
    reasoning_effort: ReasoningEffort = "medium"
    default_temperature: float = 1.0  # gpt-oss model card recommends temperature=1.0, top_p=1.0
    default_max_tokens: int = 4000
    default_k: int = 5
    default_seed: int | None = None
    max_concurrent_generations: int = 1

    # Timeouts / retries
    llm_timeout_s: float = 300.0  # generous: first request after idle loads the model
    llm_max_retries: int = 2      # connection errors and 5xx only, never 4xx

    # Fake client behaviour (LLM_MODE=fake): normal | fail | truncate | empty
    fake_behavior: Literal["normal", "fail", "truncate", "empty"] = "normal"
    fake_delay_s: float = 0.02

    # Prompts and storage
    prompt_version: str = "v0_placeholder"
    prompts_dir: Path = ROOT / "config" / "prompts"
    db_path: Path = ROOT / "data" / "app.db"

    app_git_sha: str | None = None

    @property
    def reasoning_field_list(self) -> list[str]:
        return [f.strip() for f in self.reasoning_fields.split(",") if f.strip()]

    def resolved_git_sha(self) -> str | None:
        if self.app_git_sha:
            return self.app_git_sha
        try:
            out = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=3
            )
            return out.stdout.strip() or None if out.returncode == 0 else None
        except Exception:
            return None


@lru_cache
def get_settings() -> Settings:
    return Settings()

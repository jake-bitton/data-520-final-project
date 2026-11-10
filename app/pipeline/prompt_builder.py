"""Loads versioned prompt templates and renders chat messages (design doc, Section 7).

Prompt wording is authored externally; this module only defines how templates are
loaded, rendered, and versioned.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from app.schemas import Chunk

REQUIRED_KEYS = ("system", "user", "context_block", "source")


def _fill(template: str, **values: object) -> str:
    # Plain replacement (not str.format) so braces in user queries or document text are safe.
    out = template
    for key, val in values.items():
        out = out.replace("{" + key + "}", "" if val is None else str(val))
    return out


class PromptBuilder:
    def __init__(self, prompts_dir: Path, version: str, reasoning_in_system: bool = False):
        path = Path(prompts_dir) / f"{version}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Prompt version '{version}' not found at {path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        missing = [k for k in REQUIRED_KEYS if k not in data]
        if missing:
            raise ValueError(f"Prompt file {path.name} is missing keys: {missing}")
        self.version = version
        self.template = data
        self.reasoning_in_system = reasoning_in_system

    def render(self, query: str, sources: list[Chunk], reasoning_effort: str = "medium") -> list[dict]:
        t = self.template
        reasoning_line = f"\nReasoning: {reasoning_effort}" if self.reasoning_in_system else ""
        system = _fill(t["system"], reasoning_line=reasoning_line).strip()

        if sources:
            rendered_sources = "\n".join(
                _fill(
                    t["source"],
                    rank=i,
                    title=c.title or "Untitled",
                    section=c.section or "—",
                    source_url=c.source_url or "no URL",
                    location=c.location or "—",
                    date=c.date or "undated",
                    text=c.text,
                ).strip()
                + "\n"
                for i, c in enumerate(sources, start=1)
            )
            context_block = _fill(t["context_block"], sources=rendered_sources.strip()).rstrip() + "\n\n"
        else:
            context_block = ""

        user = _fill(t["user"], context_block=context_block, query=query.strip()).strip()
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

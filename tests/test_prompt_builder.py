import os
from pathlib import Path

from app.pipeline.prompt_builder import PromptBuilder
from app.schemas import Chunk
from app.settings import Settings

GOLDEN = Path(__file__).parent / "golden"
QUERY = "How should the IJC coordinate {flood} response in the Souris River basin?"
SOURCES = [
    Chunk(chunk_id="c1", doc_id="d1", text="The Board shall report annually.", title="Souris River Directive",
          section="Duties", source_url="https://ijc.org/a", location="p. 3", date="2015-01-01", score=0.91),
    Chunk(chunk_id="c2", doc_id="d2", text="Flood forecasting is shared.", title="Flood Study", score=0.80),
]


def _builder():
    s = Settings(_env_file=None)
    return PromptBuilder(s.prompts_dir, "v0_placeholder")


def _check_golden(name: str, messages: list[dict]):
    rendered = "\n\n=====\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)
    path = GOLDEN / name
    if os.environ.get("UPDATE_GOLDEN") or not path.exists():
        path.parent.mkdir(exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
    assert rendered == path.read_text(encoding="utf-8")


def test_without_sources_golden():
    msgs = _builder().render(QUERY, [])
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "Context from IJC documents" not in msgs[1]["content"]
    assert "{flood}" in msgs[1]["content"]  # braces in user text are left alone
    _check_golden("prompt_no_sources.txt", msgs)


def test_with_sources_golden():
    msgs = _builder().render(QUERY, SOURCES)
    assert "[1] Souris River Directive" in msgs[1]["content"]
    assert "[2] Flood Study" in msgs[1]["content"]
    _check_golden("prompt_with_sources.txt", msgs)


def test_variants_differ_only_by_context_block():
    b = _builder()
    no_rag, rag = b.render(QUERY, []), b.render(QUERY, SOURCES)
    assert no_rag[0] == rag[0]  # identical system message
    assert rag[1]["content"].endswith(no_rag[1]["content"])  # user msg = context block + identical tail


def test_reasoning_line_fallback():
    s = Settings(_env_file=None)
    msgs = PromptBuilder(s.prompts_dir, "v0_placeholder", reasoning_in_system=True).render(QUERY, [], "high")
    assert msgs[0]["content"].endswith("Reasoning: high")

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.db.connection import init_db  # noqa: E402
from app.db.runs_repo import RunsRepo  # noqa: E402
from app.llm.fake_client import FakeLLMClient  # noqa: E402
from app.main import create_app  # noqa: E402
from app.pipeline.orchestrator import Orchestrator  # noqa: E402
from app.pipeline.prompt_builder import PromptBuilder  # noqa: E402
from app.retrieval.null import NullRetriever  # noqa: E402
from app.settings import Settings  # noqa: E402


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, llm_mode="fake", fake_delay_s=0, db_path=tmp_path / "test.db", app_git_sha="test")


@pytest.fixture
def make_orchestrator(settings):
    def _make(behavior="normal", retriever=None):
        init_db(settings.db_path)
        repo = RunsRepo(settings.db_path)
        return Orchestrator(
            settings=settings,
            llm=FakeLLMClient(behavior=behavior, delay_s=0),
            retriever=retriever or NullRetriever(),
            prompt_builder=PromptBuilder(settings.prompts_dir, settings.prompt_version),
            repo=repo,
        )

    return _make


@pytest.fixture
def client(settings):
    app = create_app(settings=settings, llm=FakeLLMClient(delay_s=0))
    with TestClient(app) as c:
        yield c

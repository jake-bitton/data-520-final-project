"""FastAPI entry point.  Run:  uvicorn app.main:app --reload"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI

from app.api import routes_generate, routes_meta, routes_runs
from app.db.connection import init_db
from app.db.runs_repo import RunsRepo
from app.llm.base import LLMClient
from app.llm.fake_client import FakeLLMClient
from app.llm.openai_client import OpenAICompatClient
from app.pipeline.orchestrator import Orchestrator
from app.pipeline.prompt_builder import PromptBuilder
from app.retrieval.base import Retriever
from app.retrieval.null import NullRetriever
from app.settings import Settings, get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def build_llm(settings: Settings) -> LLMClient:
    if settings.llm_mode == "fake":
        return FakeLLMClient(behavior=settings.fake_behavior, delay_s=settings.fake_delay_s)
    return OpenAICompatClient(settings)


def create_app(
    settings: Optional[Settings] = None,
    llm: Optional[LLMClient] = None,
    retriever: Optional[Retriever] = None,
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        init_db(settings.db_path)
        repo = RunsRepo(settings.db_path)
        app.state.repo = repo
        app.state.orchestrator = Orchestrator(
            settings=settings,
            llm=llm or build_llm(settings),
            retriever=retriever or NullRetriever(),  # Phase 3: ChromaRetriever
            prompt_builder=PromptBuilder(
                settings.prompts_dir, settings.prompt_version, settings.reasoning_in_system_prompt
            ),
            repo=repo,
        )
        logging.getLogger(__name__).info(
            "Backend ready: mode=%s model=%s server=%s prompt=%s db=%s",
            settings.llm_mode, settings.llm_model, settings.llm_server, settings.prompt_version, settings.db_path,
        )
        yield

    app = FastAPI(title="IJC Policy Assistant API", version="0.1.0", lifespan=lifespan)
    app.include_router(routes_meta.router)
    app.include_router(routes_generate.router)
    app.include_router(routes_runs.router)
    return app


app = create_app()

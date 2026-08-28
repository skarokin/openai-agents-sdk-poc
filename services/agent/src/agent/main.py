"""FastAPI application entry point."""

import os
from contextlib import asynccontextmanager

import uvicorn
from agent_common import HealthResponse
from fastapi import FastAPI

from agent.core import AgentFactory, SessionManager
from agent.core.config import load_service_config
from agent.core.observability import (
    setup_observability,
    shutdown_observability,
)
from agent.formatters import CompleteFormatter, EventsFormatter
from agent.protocols import rest_router, sse_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_service_config()
    sessions = SessionManager.from_environment()
    factory = AgentFactory(config)
    app.state.session_manager = sessions
    app.state.complete_formatter = CompleteFormatter(factory, sessions)
    app.state.events_formatter = EventsFormatter(factory, sessions)
    yield
    shutdown_observability()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Multi-Protocol Agent POC",
        version="0.1.0",
        lifespan=lifespan,
    )
    setup_observability(app)
    app.include_router(rest_router)
    app.include_router(sse_router)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse()

    return app


app = create_app()


def main() -> None:
    uvicorn.run(
        "agent.main:app",
        host=os.getenv("AGENT_HOST", "127.0.0.1"),
        port=int(os.getenv("AGENT_PORT", "8000")),
        reload=False,
    )

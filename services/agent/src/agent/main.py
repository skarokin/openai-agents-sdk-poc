"""FastAPI application entry point."""

import os
from contextlib import asynccontextmanager
from typing import Annotated

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request

from agent.common.http_dependencies import trusted_identity
from agent.config import load_service_config
from agent.core import TokenVault
from agent.core.agent_factory import AgentFactory
from agent.core.models import IdentityContext
from agent.core.observability import (
    setup_observability,
    shutdown_observability,
)
from agent.endpoints import rest_router, sse_router
from agent.runners import AgentRunner
from agent_common import HealthResponse, VaultTokenRequest, VaultTokenResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_service_config()
    token_vault = TokenVault.from_environment()
    factory = AgentFactory(config)
    app.state.token_vault = token_vault
    app.state.agent_runner = AgentRunner(factory, token_vault)
    try:
        yield
    finally:
        await factory.close()
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

    @app.post("/v1/agent/vault/tokens", response_model=VaultTokenResponse)
    async def store_vault_token(
        body: VaultTokenRequest,
        identity: Annotated[IdentityContext, Depends(trusted_identity)],
        request: Request,
    ) -> VaultTokenResponse:
        try:
            await request.app.state.token_vault.put(
                identity,
                body.service,
                body.access_token,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return VaultTokenResponse(service=body.service)

    return app


app = create_app()


def main() -> None:
    uvicorn.run(
        "agent.main:app",
        host=os.getenv("AGENT_HOST", "127.0.0.1"),
        port=int(os.getenv("AGENT_PORT", "8000")),
        reload=False,
    )

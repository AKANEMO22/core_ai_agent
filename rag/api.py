"""Backend-only API. A single authenticated principal owns each instance."""

import hmac
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from .engine import RAGEngine
from .indexing import build_index
from .schemas import RAGFollowUpRequest, RAGInterviewRequest, RAGInterviewResponse
from .storage import RAGStore

ROOT = Path(__file__).resolve().parents[1]


class IndexCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    activate_version: str | None = None


def create_app(
    *,
    store_path=None,
    data_dir=None,
    engine=None,
    api_token=None,
    tenant_id=None,
    allow_local=None,
):
    store_path = store_path or os.getenv("RAG_DB_PATH", str(ROOT / "data/rag/index.db"))
    data_dir = data_dir or ROOT / "data"
    token = api_token if api_token is not None else os.getenv("RAG_API_TOKEN", "")
    owner = tenant_id or os.getenv("RAG_TENANT_ID", "local")
    local = (
        allow_local
        if allow_local is not None
        else os.getenv("RAG_ALLOW_LOCAL", "0") == "1"
    )
    guard = threading.RLock()

    @asynccontextmanager
    async def lifespan(app):
        if not token and not local:
            raise RuntimeError(
                "Set RAG_API_TOKEN or explicitly enable RAG_ALLOW_LOCAL=1 for loopback development"
            )
        app.state.engine = engine or RAGEngine(RAGStore(store_path), data_dir)
        app.state.engine.store.purge()
        yield

    app = FastAPI(title="RoleUp RAG + GraphRAG", version="1.0.0", lifespan=lifespan)

    @app.middleware("http")
    async def body_limit(request: Request, call_next):
        # Bound both Content-Length and streamed/chunked input before model validation.
        size, parts = 0, []
        async for part in request.stream():
            size += len(part)
            if size > 262144:
                return JSONResponse(
                    status_code=413, content={"detail": "Request exceeds 256 KiB"}
                )
            parts.append(part)
        request._body = b"".join(parts)
        return await call_next(request)

    def authorize(request: Request, authorization: str | None = Header(default=None)):
        if token:
            supplied = (
                authorization.removeprefix("Bearer ")
                if authorization and authorization.startswith("Bearer ")
                else ""
            )
            if not hmac.compare_digest(supplied, token):
                raise HTTPException(401, "Invalid backend credentials")
        elif (
            not local
            or not request.client
            or request.client.host not in {"127.0.0.1", "::1", "localhost"}
        ):
            raise HTTPException(
                403, "Unauthenticated access is restricted to loopback development"
            )
        return owner

    def tenant_check(value, principal):
        if value != principal:
            raise HTTPException(403, "Tenant is not authorized for this instance")

    @app.exception_handler(KeyError)
    async def missing_handler(request, exc):
        return JSONResponse(status_code=404, content={"detail": str(exc.args[0])})

    @app.exception_handler(ValueError)
    async def invalid_handler(request, exc):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(PermissionError)
    async def permission_handler(request, exc):
        return JSONResponse(
            status_code=403, content={"detail": "Resource is not authorized"}
        )

    @app.get("/health")
    def health():
        current = app.state.engine
        return {
            "status": "ready",
            "schema_version": "1.0",
            "index_version": current.version,
            "embedding_provider": current.provider.signature,
        }

    @app.post(
        "/rag/context",
        response_model=RAGInterviewResponse,
    )
    def context(body: RAGInterviewRequest, principal: Annotated[str, Depends(authorize)]):
        tenant_check(body.tenant_id, principal)
        with guard:
            return app.state.engine.retrieve(body)

    @app.post("/rag/follow-up", response_model=RAGInterviewResponse)
    def follow_up(body: RAGFollowUpRequest, principal: Annotated[str, Depends(authorize)]):
        tenant_check(body.tenant_id, principal)
        with guard:
            return app.state.engine.follow_up(body)

    @app.get("/rag/trace/{request_id}")
    def trace(request_id: str, principal: Annotated[str, Depends(authorize)]):
        return app.state.engine.store.trace(request_id, principal)

    @app.delete("/rag/session/{session_id}")
    def delete_session(session_id: str, principal: Annotated[str, Depends(authorize)]):
        with guard:
            store = app.state.engine.store
            store.session(session_id, principal)
            store.delete_session(session_id, principal)
        return {"deleted": True}

    @app.post("/rag/index")
    def index(body: IndexCommand, principal: Annotated[str, Depends(authorize)]):
        # Trusted backend/admin operation, serialized with serving. No arbitrary paths/models.
        with guard:
            old = app.state.engine
            if body.activate_version:
                replacement = RAGEngine(
                    old.store, data_dir, version=body.activate_version
                )
                old.store.activate(replacement.version)
                result = {"index_version": replacement.version, "activated": True}
            else:
                result = build_index(old.store, data_dir, old.provider)
                replacement = RAGEngine(old.store, data_dir, provider=old.provider)
            app.state.engine = replacement
        return result

    return app


app = create_app()

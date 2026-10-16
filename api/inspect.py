"""Read-only, authenticated inspection of persisted core execution data."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Optional

import jwt
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from redis.exceptions import RedisError

from api.auth import require_service_token
from core.workspace.session import SessionWorkspace, normalize_session_id
from core.workspace.security import WorkspacePathError
from fe_inspect.store import sanitize_payload

router = APIRouter(prefix="/api/v1/inspect", tags=["fe-inspect"])


def _recorder(request: Request):
    recorder = request.app.state.inspect_trace_recorder
    if recorder is None:
        raise HTTPException(503, "Persistent tracing is not enabled")
    return recorder


def _storage(request: Request) -> str:
    """The durable trace backend actually in use ("postgres" or "sqlite")."""
    recorder = request.app.state.inspect_trace_recorder
    if recorder is not None:
        return recorder.store.backend
    return getattr(request.app.state, "storage_backend", "sqlite")


def _read_json(workspace: SessionWorkspace, relative: str) -> Optional[dict]:
    try:
        data = json.loads(workspace.resolve_path(relative).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data.get("payload", data)
    except (OSError, ValueError, TypeError, WorkspacePathError):
        pass
    return None


def _load_id(path: Path) -> Optional[str]:
    try:
        data = json.loads((path / "inputs/form_input.json").read_text(encoding="utf-8"))
        return data.get("session_id") if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


async def _workspace(request: Request, session_id: str) -> SessionWorkspace:
    root = Path(request.app.state.workspace_mgr.root_base).resolve()
    path = root / normalize_session_id(session_id)
    try:
        resolved = path.resolve(strict=True)
        if path.is_symlink() or resolved.parent != root or not resolved.is_dir():
            raise ValueError
        stored_id = await asyncio.to_thread(_load_id, resolved)
        if stored_id != session_id:
            raise ValueError
    except (OSError, RuntimeError, ValueError):
        raise HTTPException(404, "Run workspace not found")
    return SessionWorkspace(session_id, resolved, path.name, fsync=False)


def _artifacts(workspace: SessionWorkspace) -> list[dict]:
    rows = []
    for path in workspace.list_artifacts():
        try:
            stat = path.stat()
            rows.append({"path": path.relative_to(workspace.root_path).as_posix(), "size_bytes": stat.st_size,
                         "modified_at": stat.st_mtime, "extension": path.suffix.lower().lstrip(".")})
        except OSError:
            continue
    return sorted(rows, key=lambda row: row["path"])


async def _state(request: Request, session_id: str) -> tuple[Optional[dict], str]:
    store = _recorder(request).store
    archived = await asyncio.to_thread(store.read_state, session_id)
    outage = getattr(request.state, "inspect_redis_error", None)
    try:
        if outage:
            raise ConnectionError(outage)
        live = await asyncio.wait_for(request.app.state.state_mgr.aget(session_id), timeout=1.0)
    except (RedisError, TimeoutError, OSError) as exc:
        request.state.inspect_redis_error = str(exc) or type(exc).__name__
        snapshot = dict(archived or {"session_id": session_id, "status": "ARCHIVED"})
        snapshot["state_refresh_error"] = sanitize_payload(str(exc) or type(exc).__name__)
        return snapshot, store.backend if archived else "not_recorded"
    if live is not None:
        snapshot = live.to_dict()
        if archived and archived.get("_inspect_provider"):
            snapshot["_inspect_provider"] = archived["_inspect_provider"]
        if archived is None or archived.get("state_version") != snapshot.get("state_version"):
            # Archive only observed authoritative state, including legacy sessions.
            await asyncio.to_thread(store.save_state, snapshot)
        return sanitize_payload(snapshot), "redis"
    return archived, store.backend if archived else "not_recorded"


async def _events(request: Request, session_id: str, after_sequence: int = 0, limit: int = 5000):
    store = _recorder(request).store
    # Import surviving pre-instrumentation Redis events verbatim; never invent old spans.
    imported = await asyncio.to_thread(store.get_metadata, session_id, "redis_history_imported")
    if not imported:
        try:
            if getattr(request.state, "inspect_redis_error", None):
                raise ConnectionError(request.state.inspect_redis_error)
            surviving = await asyncio.wait_for(request.app.state.event_store.get_events_async(session_id), timeout=2.0)
        except (RedisError, TimeoutError, OSError):
            surviving = None
        if surviving is not None:
            for event in surviving:
                await asyncio.to_thread(store.save_event, event.model_dump(mode="json"))
            await asyncio.to_thread(store.set_metadata, session_id, "redis_history_imported", {"observed_at": time.time()})
    return await asyncio.to_thread(store.read_events, session_id, after_sequence, limit)


def _current_provider(request: Request) -> dict:
    app = request.app
    return {"client": type(app.state.llm_client).__name__,
            "model": app.state.settings.llm_model if app.state.studio_real_provider else None,
            "real_provider": bool(app.state.studio_real_provider),
            "retrieval_backend": app.state.knowledge_service.backend_name}


@router.get("/config")
async def config(request: Request):
    settings = request.app.state.settings
    provider = _current_provider(request)
    return {"local_token_enabled": bool(settings.INSPECTOR_LOCAL_AUTH and settings.APP_ENV.lower() != "production"),
            "provider": provider["client"], "provider_model": provider["model"],
            "real_provider": provider["real_provider"] and provider["retrieval_backend"] != "mock",
            "retrieval_backend": provider["retrieval_backend"], "storage": _storage(request),
            "test_ui_enabled": settings.TEST_UI_ENABLED,
            "execution_enabled": provider["real_provider"] and provider["retrieval_backend"] != "mock"}


@router.post("/token")
async def local_token(request: Request):
    settings = request.app.state.settings
    if not settings.INSPECTOR_LOCAL_AUTH or settings.APP_ENV.lower() == "production":
        raise HTTPException(404, "Local token issuing disabled")
    now = int(time.time())
    token = jwt.encode({"iss": "core-studio", "aud": "ai-runtime", "iat": now, "exp": now + 3600},
                       settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)
    return {"token": token, "expires_in": 3600}


@router.get("/system", dependencies=[Depends(require_service_token)])
async def system(request: Request):
    app = request.app
    provider = _current_provider(request)
    return {"service": "AI Interview Core", "environment": app.state.settings.APP_ENV,
            "provider": provider["client"], "provider_model": provider["model"],
            "real_provider": provider["real_provider"], "retrieval_backend": provider["retrieval_backend"],
            "storage": _storage(request), "trace_database": _recorder(request).store.location,
            "active_connections": app.state.connection_mgr.active_connection_count(),
            "assessor_pool": app.state.worker_pool.status()}


def _candidates(root: Path, query: str) -> list[dict]:
    rows = []
    for path in root.iterdir():
        if path.is_symlink() or not path.is_dir() or path.resolve().parent != root:
            continue
        session_id = _load_id(path)
        if not session_id or normalize_session_id(session_id) != path.name:
            continue
        workspace = SessionWorkspace(session_id, path, path.name, fsync=False)
        form = _read_json(workspace, "inputs/form_input.json") or {}
        if query.casefold() not in f"{session_id} {form.get('candidate_name', '')} {form.get('target_role', '')}".casefold():
            continue
        rows.append({"session_id": session_id, "candidate_name": form.get("candidate_name"),
                     "target_role": form.get("target_role"), "updated_at": path.stat().st_mtime,
                     "workspace": workspace})
    return sorted(rows, key=lambda row: row["updated_at"], reverse=True)


@router.get("/runs", dependencies=[Depends(require_service_token)])
async def runs(request: Request, q: str = Query("", max_length=160), limit: int = Query(100, ge=1, le=250),
               offset: int = Query(0, ge=0)):
    root = Path(request.app.state.workspace_mgr.root_base).resolve()
    candidates = await asyncio.to_thread(_candidates, root, q)
    rows = []
    store = _recorder(request).store
    for row in candidates[offset:offset + limit]:
        workspace = row.pop("workspace")
        snapshot, source = await _state(request, row["session_id"])
        trace = await asyncio.to_thread(store.span_summary, row["session_id"])
        artifact_list = await asyncio.to_thread(_artifacts, workspace)
        review = await asyncio.to_thread(_read_json, workspace, "reports/review.json") or {}
        decision = await asyncio.to_thread(_read_json, workspace, "reports/decision.json") or {}
        row.update({"status": (snapshot or {}).get("status", "ARCHIVED"), "state_source": source,
                    "execution_origin": "legacy_simulation" if workspace.exists("inspect/simulation_result.json") else "core",
                    "current_turn": (snapshot or {}).get("current_turn"),
                    "created_at": (snapshot or {}).get("created_at"),
                    "updated_at": (snapshot or {}).get("updated_at", row["updated_at"]),
                    "event_count": await asyncio.to_thread(store.count_events, row["session_id"]),
                    "span_count": trace["span_count"], "trace_coverage": trace["coverage"],
                    "artifact_count": len(artifact_list),
                    "verdict": review.get("final_verdict") or decision.get("verdict")})
        rows.append(row)
    return {"runs": sorted(rows, key=lambda row: row.get("updated_at") or 0, reverse=True),
            "total": len(candidates), "offset": offset, "limit": limit, "storage": store.backend}


@router.get("/runs/{session_id}/trace", dependencies=[Depends(require_service_token)])
async def trace(session_id: str, request: Request):
    await _workspace(request, session_id)
    return await asyncio.to_thread(_recorder(request).store.read_spans, session_id)


@router.get("/runs/{session_id}/events", dependencies=[Depends(require_service_token)])
async def events(session_id: str, request: Request, after_sequence: int = Query(0, ge=0),
                 limit: int = Query(1000, ge=1, le=5000)):
    await _workspace(request, session_id)
    rows = await _events(request, session_id, after_sequence, limit)
    return {"events": rows, "storage": _storage(request), "next_sequence": rows[-1]["sequence"] if rows else after_sequence,
            "has_more": len(rows) == limit}


@router.get("/runs/{session_id}", dependencies=[Depends(require_service_token)])
async def run(session_id: str, request: Request):
    workspace = await _workspace(request, session_id)
    snapshot, source = await _state(request, session_id)
    trace_data = await asyncio.to_thread(_recorder(request).store.read_spans, session_id)
    artifact_list = await asyncio.to_thread(_artifacts, workspace)
    event_rows = await _events(request, session_id)
    span_rows = trace_data["spans"]
    provider = (snapshot or {}).get("_inspect_provider") or {"client": None, "model": None, "real_provider": None, "retrieval_backend": None}
    token_usage = [(span.get("outputs") or {}).get("usage", {}).get("total_tokens")
                   for span in span_rows if span["kind"] == "llm" and span.get("attributes", {}).get("method") == "stream_chat"]
    measured_tokens = [count for count in token_usage if isinstance(count, (int, float))]
    total_tokens = sum(measured_tokens) if measured_tokens else None
    decision = await asyncio.to_thread(_read_json, workspace, "reports/decision.json") or {}
    status = (snapshot or {}).get("status", "ARCHIVED")
    started, ended = (snapshot or {}).get("created_at"), (snapshot or {}).get("updated_at")
    terminal = status in {"REPORT_READY", "FAILED", "ASSESSMENT_FAILED", "CANCELLED", "EXPIRED"}
    return {"session": dict(snapshot or {"session_id": session_id, "status": status}, state_source=source),
            "execution_origin": "legacy_simulation" if workspace.exists("inspect/simulation_result.json") else "core",
            "summary": {"status": status, "span_count": len(span_rows), "event_count": await asyncio.to_thread(_recorder(request).store.count_events, session_id),
                        "artifact_count": len(artifact_list), "turn_count": (snapshot or {}).get("current_turn"),
                        "token_count": total_tokens, "verdict": decision.get("verdict"),
                        "duration_seconds": max(0, ended - started) if terminal and started and ended else None},
            "spans": span_rows, "trace": trace_data, "provider": provider, "events": event_rows,
            "events_truncated": len(event_rows) == 5000, "artifacts": artifact_list}


@router.get("/runs/{session_id}/export", dependencies=[Depends(require_service_token)])
async def export_run(session_id: str, request: Request):
    return await run(session_id, request)


@router.get("/runs/{session_id}/artifacts/{artifact_path:path}", dependencies=[Depends(require_service_token)])
async def artifact(session_id: str, artifact_path: str, request: Request):
    workspace = await _workspace(request, session_id)
    try:
        target = workspace.resolve_path(artifact_path)
        if not target.is_file():
            raise ValueError
        if target.stat().st_size > 8 * 1024 * 1024:
            raise HTTPException(413, "Artifact exceeds the 8 MiB preview limit")
        raw = await asyncio.to_thread(target.read_text, encoding="utf-8")
        try:
            content = json.loads(raw) if target.suffix.lower() == ".json" else raw
        except ValueError:
            content = raw
    except (OSError, UnicodeDecodeError, ValueError, WorkspacePathError):
        raise HTTPException(404, "Artifact not found")
    return {"session_id": session_id, "path": artifact_path, "content": sanitize_payload(content)}

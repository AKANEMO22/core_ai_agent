"""FastAPI REST routes under /api/v1/sessions for session lifecycle management."""

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse

from api.auth import create_session_token, require_service_token, require_session_or_service_token
from api.schemas import (
    CancelSessionResponse,
    CreateSessionRequest,
    CreateSessionResponse,
    DeleteSessionResponse,
    RequeueAssessmentsResponse,
    ReviewRequest,
    SessionReportResponse,
    SessionStatusResponse,
    StartInterviewResponse,
)
from config.settings import get_settings
from core.assessor.queue import session_assessment_keys
from core.observability import session_ref
from core.profiler.agent import ProfilerAgent
from core.runtime.agent import sanitize_secrets
from core.runtime.types import LoopEvent, LoopEventType
from core.session.idempotency import IdempotencyConflictError, IdempotencyInFlightError
from core.session.state import (
    InvalidStateTransitionError,
    LifecycleStatus,
    SessionAlreadyExistsError,
    SessionState,
    TERMINAL_STATUSES,
)
from core.workspace.session import SessionWorkspace

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/sessions", tags=["sessions"])

SAFE_FAILURE_MESSAGES = {
    "PROFILING_FAILED": "Interview preparation failed. Please create a new session.",
    "PROFILING_TIMEOUT": "Interview preparation timed out. Please create a new session.",
    "ASSESSMENT_EXHAUSTED": "Assessment could not be completed. An operator can retry it.",
}


def _event(session_id: str, event_type: LoopEventType, payload: Dict[str, Any], turn_id: str = "lifecycle") -> LoopEvent:
    return LoopEvent(
        schema_version="1.0",
        event_id=f"evt_{uuid.uuid4().hex[:16]}",
        session_id=session_id,
        turn_id=turn_id,
        sequence=0,
        timestamp=time.time(),
        type=event_type,
        payload=payload,
    )


async def _fail_session(app: Any, session_id: str, failure_code: str) -> None:
    """Move a session to FAILED with a safe client message and publish session_failed."""
    state_store = app.state.state_mgr
    state = await state_store.aget(session_id)
    if state is None or state.status in TERMINAL_STATUSES:
        return
    correlation_id = uuid.uuid4().hex[:16]
    try:
        await state_store.atransition(
            state,
            expected_version=state.state_version,
            new_status=LifecycleStatus.FAILED,
            metadata={
                "failure_code": failure_code,
                "failure_message": SAFE_FAILURE_MESSAGES.get(failure_code, "Session failed."),
                "correlation_id": correlation_id,
            },
        )
    except InvalidStateTransitionError:
        return
    await app.state.connection_mgr.publish(
        session_id,
        _event(session_id, LoopEventType.SESSION_FAILED, {"failure_code": failure_code, "correlation_id": correlation_id}),
    )
    logger.error(
        "Session failed",
        extra={"session": session_ref(session_id), "failure_code": failure_code, "correlation_id": correlation_id},
    )


async def _run_profiling_task(app: Any, session_id: str, workspace: SessionWorkspace) -> None:
    """Run pre-flight profiling under a deadline; READY on success, FAILED otherwise."""
    profiler: Optional[ProfilerAgent] = getattr(app.state, "profiler_agent", None)
    settings = app.state.settings
    state_store = app.state.state_mgr
    if profiler is None:
        await _fail_session(app, session_id, "PROFILING_FAILED")
        return

    semaphore = getattr(app.state, "profiler_semaphore", None)
    try:
        if semaphore is not None:
            async with semaphore:
                result = await asyncio.wait_for(
                    profiler.run_profiling(workspace), timeout=settings.PROFILING_DEADLINE_SECONDS
                )
        else:
            result = await asyncio.wait_for(
                profiler.run_profiling(workspace), timeout=settings.PROFILING_DEADLINE_SECONDS
            )
    except asyncio.TimeoutError:
        await _fail_session(app, session_id, "PROFILING_TIMEOUT")
        return
    except Exception as e:
        logger.error("Profiling crashed for session %s: %s", session_ref(session_id), sanitize_secrets(str(e)))
        await _fail_session(app, session_id, "PROFILING_FAILED")
        return

    if result.get("status") != "SUCCESS":
        await _fail_session(app, session_id, "PROFILING_FAILED")
        return

    ready_metadata: Dict[str, Any] = {"retrieval_status": result.get("retrieval_status", "disabled")}
    for hook in getattr(app.state, "post_profiling_hooks", []):
        try:
            extra = await hook(session_id, workspace)
            if isinstance(extra, dict):
                ready_metadata.update(extra)
        except Exception as e:
            logger.warning("Post-profiling hook failed: %s", sanitize_secrets(str(e)))

    current = await state_store.aget(session_id)
    if current is not None and current.status == LifecycleStatus.PROFILING:
        try:
            await state_store.atransition(
                current,
                expected_version=current.state_version,
                new_status=LifecycleStatus.READY,
                metadata=ready_metadata,
            )
            logger.info("Session ready", extra={"session": session_ref(session_id)})
        except InvalidStateTransitionError:
            pass


async def _validate_session_access(request: Request, claims: Dict[str, Any], session_id: str) -> None:
    """Allow service tokens for any session; session tokens only for their own, non-revoked session."""
    if claims.get("aud") == "ai-runtime":
        return
    token_session = claims.get("session_id") or claims.get("sub")
    if token_session != session_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Token is not authorized for this session",
        )
    revocations = getattr(request.app.state, "revocations", None)
    if revocations is not None and await revocations.is_revoked(session_id):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session token revoked")


async def _require_state(request: Request, session_id: str) -> SessionState:
    state = await request.app.state.state_mgr.aget(session_id)
    if state is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return state


@router.post(
    "",
    response_model=CreateSessionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create or initialize an interview session",
)
async def create_session(
    request: Request,
    request_body: CreateSessionRequest,
    background_tasks: BackgroundTasks,
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    claims: Dict[str, Any] = Depends(require_service_token),
) -> Any:
    """Initialize a candidate interview session and schedule pre-flight profiling.

    - Repeating an Idempotency-Key with the same payload returns the original response (202).
    - Reusing it with a different payload returns 409; a concurrent duplicate returns 409 too.
    - Exceeding the per-caller creation rate returns 429.
    """
    app = request.app
    settings = getattr(app.state, "settings", get_settings())
    state_store = app.state.state_mgr
    idempotency = app.state.idempotency_mgr

    limiter = getattr(app.state, "session_create_limiter", None)
    caller = str(claims.get("iss") or "service")
    if limiter is not None and not limiter.acquire(caller):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Session creation rate limit exceeded",
            headers={"Retry-After": "1"},
        )

    req_payload = request_body.model_dump()
    if idempotency_key:
        try:
            claimed, cached = await idempotency.abegin(f"create_session:{idempotency_key}", payload=req_payload)
        except IdempotencyConflictError as e:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) from e
        except IdempotencyInFlightError as e:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) from e
        if not claimed:
            return JSONResponse(status_code=status.HTTP_202_ACCEPTED, content=cached)

    try:
        session_id = request_body.session_id or f"sess_{uuid.uuid4().hex[:12]}"
        try:
            state = await state_store.acreate(
                session_id=session_id,
                initial_status=LifecycleStatus.INITIALIZED,
                metadata=request_body.metadata,
            )
        except SessionAlreadyExistsError as e:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Session already exists") from e

        workspace = app.state.workspace_mgr.get_or_create_workspace(session_id)
        workspace.initialize()
        form_data = {
            "candidate_name": request_body.candidate_name,
            "experience_years": request_body.experience_years,
            "target_company": request_body.target_company,
            "target_role": request_body.target_role,
            "skills": request_body.skills,
            "job_description": request_body.job_description,
            "interview_format": request_body.interview_format,
        }
        await asyncio.to_thread(
            workspace.write_artifact, "inputs/form_input.json", artifact_type="input", payload=form_data
        )
        if request_body.cv_text:
            await asyncio.to_thread(workspace.write_text, "inputs/cv_input.txt", request_body.cv_text)

        state = await state_store.atransition(
            state, expected_version=state.state_version, new_status=LifecycleStatus.PROFILING
        )
        session_token = create_session_token(session_id=session_id, secret=settings.JWT_SECRET)
        background_tasks.add_task(_run_profiling_task, app=app, session_id=session_id, workspace=workspace)

        response_data = CreateSessionResponse(
            session_id=session_id,
            status=LifecycleStatus.PROFILING,
            state_version=state.state_version,
            session_token=session_token,
            websocket_stream_url=f"/ws/sessions/{session_id}/stream",
            created_at=state.created_at,
        ).model_dump(mode="json")
    except Exception:
        if idempotency_key:
            await idempotency.arelease(f"create_session:{idempotency_key}")
        raise

    if idempotency_key:
        await idempotency.arecord(f"create_session:{idempotency_key}", result=response_data, payload=req_payload)
    return JSONResponse(status_code=status.HTTP_202_ACCEPTED, content=response_data)


@router.get(
    "/{session_id}",
    response_model=SessionStatusResponse,
    summary="Retrieve current session state, stage info and counters",
)
async def get_session_status(
    session_id: str,
    request: Request,
    claims: Dict[str, Any] = Depends(require_session_or_service_token),
) -> SessionStatusResponse:
    """Lifecycle status, version, stage, turn counter, safe failure details and readiness flags."""
    await _validate_session_access(request, claims, session_id)
    state = await _require_state(request, session_id)
    metadata = dict(state.metadata)
    coverage = None
    blackboard = getattr(request.app.state, "blackboard", None)
    if blackboard is not None:
        full = await blackboard.get(session_id, "coverage")
        if full:
            coverage = {
                "uncovered": full.get("uncovered", []),
                "must_have_uncovered": full.get("must_have_uncovered", []),
                "criteria": {
                    cid: {"level": c.get("level"), "confidence": c.get("confidence"), "covered": c.get("covered")}
                    for cid, c in (full.get("criteria") or {}).items()
                },
            }
    return SessionStatusResponse(
        session_id=state.session_id,
        status=state.status,
        state_version=state.state_version,
        current_stage_id=state.current_stage_id,
        is_stage_complete=state.is_stage_complete,
        current_turn=state.current_turn,
        metadata={k: v for k, v in metadata.items() if k not in ("failure_message", "correlation_id")},
        failure_code=metadata.get("failure_code"),
        failure_message=metadata.get("failure_message"),
        retrieval_status=metadata.get("retrieval_status"),
        brief_status=metadata.get("brief_status"),
        coverage=coverage,
        created_at=state.created_at,
        updated_at=state.updated_at,
    )


@router.post(
    "/{session_id}/start",
    response_model=StartInterviewResponse,
    summary="Start the interview and generate the opening question",
)
async def start_interview(
    session_id: str,
    request: Request,
    claims: Dict[str, Any] = Depends(require_session_or_service_token),
) -> Any:
    """Backend-driven start: READY → STAGE_ACTIVE plus the opening question (turn 0).

    Events are also published to the session stream. Idempotent: repeated calls return the
    recorded opening.
    """
    await _validate_session_access(request, claims, session_id)
    await _require_state(request, session_id)
    result = await request.app.state.controller.start_interview(session_id)

    if result.status == "busy":
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content=StartInterviewResponse(session_id=session_id, status="busy", error_code=result.error_code).model_dump(),
            headers={"Retry-After": str(max(1, (result.retry_after_ms or 1000) // 1000))},
        )
    if result.status == "rejected":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=result.error_code)
    if result.status == "failed":
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Opening generation failed; retry")

    workspace = request.app.state.workspace_mgr.get_or_create_workspace(session_id)
    text = None
    if workspace.exists("turns/turn_0_interviewer.json"):
        text = workspace.read_json("turns/turn_0_interviewer.json").get("content")
    return StartInterviewResponse(
        session_id=session_id, status=result.status, turn_id=result.turn_id, interviewer_response=text
    )


@router.get(
    "/{session_id}/report",
    response_model=SessionReportResponse,
    summary="Retrieve final interview report and assessment",
)
async def get_session_report(
    session_id: str,
    request: Request,
    claims: Dict[str, Any] = Depends(require_session_or_service_token),
) -> Any:
    """200 in REPORT_READY, 202 while processing, 409 for failed/cancelled/expired, 404 if absent."""
    await _validate_session_access(request, claims, session_id)
    state = await _require_state(request, session_id)

    if state.status == LifecycleStatus.REPORT_READY:
        workspace = request.app.state.workspace_mgr.get_or_create_workspace(session_id)
        report_md: Optional[str] = None
        assessment_data: Optional[Dict[str, Any]] = None
        decision: Optional[Dict[str, Any]] = None
        if workspace.exists("reports/interview_report.md"):
            report_md = workspace.read_text("reports/interview_report.md")
        if workspace.exists("reports/final_assessment.json"):
            assessment_data = workspace.read_json("reports/final_assessment.json")
        if workspace.exists("reports/decision.json"):
            decision = json.loads(workspace.read_text("reports/decision.json"))
            if workspace.exists("reports/review.json"):
                review = workspace.read_json("reports/review.json")
                decision["review"] = review.get("payload", review)

        verdict = None
        if decision is not None:
            verdict = (decision.get("review") or {}).get("final_verdict") or decision.get("verdict")
        resp = SessionReportResponse(
            session_id=session_id,
            status=state.status,
            report_markdown=report_md,
            final_assessment=assessment_data,
            decision=decision,
            verdict=verdict,
            created_at=state.created_at,
            completed_at=state.updated_at,
            message="Final assessment and report ready",
        )
        return JSONResponse(status_code=status.HTTP_200_OK, content=resp.model_dump(mode="json"))

    if state.status in (
        LifecycleStatus.FAILED,
        LifecycleStatus.ASSESSMENT_FAILED,
        LifecycleStatus.CANCELLED,
        LifecycleStatus.EXPIRED,
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Session is in terminal non-reportable status: {state.status.value}",
        )

    resp = SessionReportResponse(
        session_id=session_id,
        status=state.status,
        created_at=state.created_at,
        message="Assessment evaluation and report generation are currently processing.",
    )
    return JSONResponse(status_code=status.HTTP_202_ACCEPTED, content=resp.model_dump(mode="json"))


@router.post(
    "/{session_id}/cancel",
    response_model=CancelSessionResponse,
    summary="Cancel an active interview session",
)
async def cancel_session(
    session_id: str,
    request: Request,
    claims: Dict[str, Any] = Depends(require_session_or_service_token),
) -> CancelSessionResponse:
    """Idempotently cancel a session: stops generation, delegated work and pending synthesis."""
    await _validate_session_access(request, claims, session_id)
    state = await _require_state(request, session_id)
    app = request.app

    if state.status == LifecycleStatus.CANCELLED:
        return CancelSessionResponse(session_id=session_id, status=LifecycleStatus.CANCELLED, message="Session already cancelled")
    if state.status in (LifecycleStatus.REPORT_READY, LifecycleStatus.FAILED, LifecycleStatus.EXPIRED):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot cancel session in terminal status '{state.status.value}'",
        )

    app.state.controller.cancel_session(session_id)
    try:
        updated = await app.state.state_mgr.atransition(
            state, expected_version=state.state_version, new_status=LifecycleStatus.CANCELLED
        )
    except InvalidStateTransitionError:
        updated = await _require_state(request, session_id)
        if updated.status != LifecycleStatus.CANCELLED:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Concurrent state change; retry")
    await app.state.connection_mgr.publish(session_id, _event(session_id, LoopEventType.SESSION_CANCELLED, {}))
    return CancelSessionResponse(session_id=session_id, status=updated.status, message="Session cancelled successfully")


@router.post(
    "/{session_id}/assessments/requeue",
    response_model=RequeueAssessmentsResponse,
    summary="Operator: requeue dead-lettered assessments",
)
async def requeue_assessments(
    session_id: str,
    request: Request,
    claims: Dict[str, Any] = Depends(require_service_token),
) -> RequeueAssessmentsResponse:
    """Re-enqueue a session's dead-lettered jobs and move ASSESSMENT_FAILED → ASSESSING (idempotent)."""
    await _require_state(request, session_id)
    app = request.app
    count = await app.state.pipeline.requeue_failed(session_id, app.state.assessment_queue)
    state = await _require_state(request, session_id)
    return RequeueAssessmentsResponse(session_id=session_id, requeued_jobs=count, status=state.status)


@router.post(
    "/{session_id}/review",
    summary="Record a human reviewer's final verdict",
)
async def review_report(
    session_id: str,
    body: ReviewRequest,
    request: Request,
    claims: Dict[str, Any] = Depends(require_service_token),
) -> Any:
    """Attach a human review to decision.json (required when the verdict is NEEDS_REVIEW in hiring_assist mode)."""
    state = await _require_state(request, session_id)
    if state.status != LifecycleStatus.REPORT_READY:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Report not ready")
    reviewer = getattr(request.app.state, "review_service", None)
    if reviewer is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Review service unavailable")
    decision = await reviewer.record_review(session_id, body.reviewer_id, body.final_verdict, body.notes)
    return JSONResponse(status_code=status.HTTP_200_OK, content={"session_id": session_id, "decision": decision})


@router.delete(
    "/{session_id}",
    response_model=DeleteSessionResponse,
    summary="Delete session workspace and artifacts under retention policy",
)
async def delete_session(
    session_id: str,
    request: Request,
    claims: Dict[str, Any] = Depends(require_service_token),
) -> DeleteSessionResponse:
    """Cancel active work, purge every Redis key and the workspace, revoke tokens, audit (content-free)."""
    app = request.app
    state = await app.state.state_mgr.aget(session_id)
    workspace_exists = app.state.cleanup_mgr.get_workspace_size_bytes(session_id) > 0
    if state is None and not workspace_exists:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    await purge_session(app, session_id)
    return DeleteSessionResponse(
        session_id=session_id,
        status="DELETED",
        message="Session and associated workspace artifacts deleted successfully",
    )


async def purge_session(app: Any, session_id: str) -> None:
    """Retention deletion shared by the DELETE endpoint and the retention sweeper."""
    controller = getattr(app.state, "controller", None)
    if controller is not None:
        controller.runners.evict(session_id)
        # Background work (orchestrator, delegations, summaries) must stop before files go away.
        await controller.cancellations.cancel_session_and_wait(session_id)
    await app.state.connection_mgr.close_session(session_id, reason="Session deleted")

    redis = app.state.redis_client
    revocations = getattr(app.state, "revocations", None)
    if revocations is not None:
        await revocations.revoke(session_id)
    await app.state.state_mgr.adelete(session_id)
    await app.state.connection_mgr.event_store.purge(session_id)
    keys = session_assessment_keys(session_id) + [
        f"session:{session_id}:messages",
        f"session:{session_id}:turn_lock",
        f"session:{session_id}:synth_lock",
        f"session:{session_id}:blackboard",
        f"session:{session_id}:summary_lock",
    ]
    await redis.delete(*keys)
    try:
        await asyncio.to_thread(app.state.cleanup_mgr.delete_workspace, session_id)
    except Exception as e:
        logger.warning("Workspace deletion error for session %s: %s", session_ref(session_id), sanitize_secrets(str(e)))
    app.state.workspace_mgr.evict_workspace(session_id)
    if controller is not None:
        controller.cancellations.forget(session_id)
        # Keep refusing work for the deleted id within this process.
        controller.cancellations.cancel_session(session_id)
    try:
        await redis.xadd(
            "audit:session_deletions",
            {"session_ref": session_ref(session_id), "deleted_at": str(time.time())},
            maxlen=100000,
            approximate=True,
        )
    except Exception as e:
        logger.warning("Audit append failed: %s", sanitize_secrets(str(e)))
    logger.info("Session deleted", extra={"session": session_ref(session_id)})

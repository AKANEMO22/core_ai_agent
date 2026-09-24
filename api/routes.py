"""FastAPI REST routes under /api/v1/sessions for session lifecycle management."""

import asyncio
import logging
import uuid
from typing import Any, Dict, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    Header,
    HTTPException,
    Request,
    Security,
    status,
)
from fastapi.responses import JSONResponse

from api.auth import (
    AuthError,
    create_session_token,
    require_service_token,
    require_session_or_service_token,
    security_scheme,
    verify_service_token,
    verify_session_token,
)
from api.schemas import (
    CancelSessionResponse,
    CreateSessionRequest,
    CreateSessionResponse,
    DeleteSessionResponse,
    SessionReportResponse,
    SessionStatusResponse,
)
from config.settings import get_settings
from core.profiler.agent import ProfilerAgent
from core.runtime.agent import sanitize_secrets
from core.session.idempotency import IdempotencyConflictError, IdempotencyManager
from core.session.state import LifecycleStatus, SessionStateManager
from core.workspace.cleanup import WorkspaceCleanupManager
from core.workspace.session import SessionWorkspace, SessionWorkspaceManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/sessions", tags=["sessions"])


async def _run_profiling_task(
    app: Any,
    session_id: str,
    workspace: SessionWorkspace,
    state_mgr: SessionStateManager,
) -> None:
    """Execute pre-flight profiling recon agent in the background and transition to READY."""
    profiler: Optional[ProfilerAgent] = getattr(app.state, "profiler_agent", None)
    if profiler is None:
        # If no profiler configured, simulate transition to READY if needed
        return

    try:
        res = await profiler.run_profiling(workspace)
        if res.get("status") == "SUCCESS":
            current = state_mgr.get_state(session_id)
            if current and current.status == LifecycleStatus.PROFILING:
                state_mgr.transition(
                    current,
                    expected_version=current.state_version,
                    new_status=LifecycleStatus.READY,
                )
                logger.info("Session '%s' transitioned to READY after profiling", session_id)
    except Exception as e:
        logger.error(
            "Background profiling encountered error for session '%s': %s",
            session_id,
            sanitize_secrets(str(e)),
        )


def _validate_session_access(claims: Dict[str, Any], session_id: str) -> None:
    """Ensure token is either a backend service token or scoped to the targeted session_id."""
    aud = claims.get("aud")
    if aud == "ai-runtime":
        # Service token has universal session access
        return

    token_session = claims.get("session_id") or claims.get("sub")
    if token_session != session_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Token is scoped to '{token_session}', cannot access session '{session_id}'",
        )


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
    """Initialize a new candidate interview session with pre-flight profiling.

    Supports:
    - Idempotency-Key: repeated calls with identical payload return the cached response (202).
    - Idempotency-Key with conflicting payload returns HTTP 409 Conflict.
    - Transitions session to PROFILING status and launches background recon.
    """
    app = request.app
    state_mgr: SessionStateManager = app.state.state_mgr
    workspace_mgr: SessionWorkspaceManager = app.state.workspace_mgr
    idempotency_mgr: IdempotencyManager = app.state.idempotency_mgr
    settings = getattr(app.state, "settings", get_settings())

    req_payload = request_body.model_dump()

    # 1. Idempotency Check
    if idempotency_key:
        try:
            cached = idempotency_mgr.get(idempotency_key, payload=req_payload)
            if cached is not None:
                logger.info("Returning cached idempotent response for key '%s'", idempotency_key)
                return JSONResponse(status_code=status.HTTP_202_ACCEPTED, content=cached)
        except IdempotencyConflictError as e:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(e),
            ) from e

    # 2. Assign Session ID and Workspace
    session_id = request_body.session_id or f"sess_{uuid.uuid4().hex[:12]}"
    workspace = workspace_mgr.get_or_create_workspace(session_id)
    workspace.initialize()

    # 3. Persist Initial Inputs
    form_data = {
        "candidate_name": request_body.candidate_name,
        "experience_years": request_body.experience_years,
        "target_company": request_body.target_company,
        "target_role": request_body.target_role,
        "skills": request_body.skills,
        "job_description": request_body.job_description,
        "interview_format": request_body.interview_format,
    }
    workspace.write_artifact(
        "inputs/form_input.json",
        artifact_type="input",
        payload=form_data,
    )
    if request_body.cv_text:
        (workspace.inputs_dir / "cv_input.txt").write_text(
            request_body.cv_text,
            encoding="utf-8",
        )

    # 4. FSM Lifecycle Transitions: INITIALIZED -> PROFILING
    state = state_mgr.create_session(
        session_id=session_id,
        initial_status=LifecycleStatus.INITIALIZED,
        metadata=request_body.metadata,
    )
    state = state_mgr.transition(
        state,
        expected_version=1,
        new_status=LifecycleStatus.PROFILING,
    )

    # 5. Generate Candidate Session Token
    session_token = create_session_token(
        session_id=session_id,
        secret=settings.JWT_SECRET,
    )

    # 6. Schedule Background Profiling Recon
    background_tasks.add_task(
        _run_profiling_task,
        app=app,
        session_id=session_id,
        workspace=workspace,
        state_mgr=state_mgr,
    )

    response_data = CreateSessionResponse(
        session_id=session_id,
        status=LifecycleStatus.PROFILING,
        state_version=state.state_version,
        session_token=session_token,
        websocket_stream_url=f"/ws/sessions/{session_id}/stream",
        created_at=state.created_at,
    )

    # 7. Record Idempotency Cache
    if idempotency_key:
        idempotency_mgr.record(
            idempotency_key,
            result=response_data.model_dump(mode="json"),
            payload=req_payload,
        )

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content=response_data.model_dump(mode="json"),
    )


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
    """Retrieve session lifecycle status, state version, current stage, and turn counters."""
    _validate_session_access(claims, session_id)
    state_mgr: SessionStateManager = request.app.state.state_mgr

    state = state_mgr.get_state(session_id)
    if state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found",
        )

    return SessionStatusResponse(
        session_id=state.session_id,
        status=state.status,
        state_version=state.state_version,
        current_stage_id=state.current_stage_id,
        is_stage_complete=state.is_stage_complete,
        current_turn=state.current_turn,
        metadata=state.metadata,
        created_at=state.created_at,
        updated_at=state.updated_at,
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
    """Retrieve final coaching report and numerical assessment.

    - Returns 200 OK with report markdown and final assessment JSON if in REPORT_READY.
    - Returns 202 Accepted while assessment is still processing.
    - Returns 409 Conflict if session failed or was cancelled.
    - Returns 404 Not Found if session is absent.
    """
    _validate_session_access(claims, session_id)
    state_mgr: SessionStateManager = request.app.state.state_mgr
    workspace_mgr: SessionWorkspaceManager = request.app.state.workspace_mgr

    state = state_mgr.get_state(session_id)
    if state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found",
        )

    # 1. Report Ready: Return HTTP 200 with generated artifacts
    if state.status == LifecycleStatus.REPORT_READY:
        workspace = workspace_mgr.get_or_create_workspace(session_id)
        report_md: Optional[str] = None
        assessment_data: Optional[Dict[str, Any]] = None

        report_file = workspace.root_path / "reports" / "interview_report.md"
        if report_file.exists():
            report_md = report_file.read_text(encoding="utf-8", errors="ignore")

        eval_file = workspace.root_path / "reports" / "final_assessment.json"
        if eval_file.exists():
            try:
                assessment_data = workspace.read_json("reports/final_assessment.json")
            except Exception:
                pass

        resp = SessionReportResponse(
            session_id=session_id,
            status=state.status,
            report_markdown=report_md,
            final_assessment=assessment_data,
            created_at=state.created_at,
            completed_at=state.updated_at,
            message="Final assessment and report ready",
        )
        return JSONResponse(status_code=status.HTTP_200_OK, content=resp.model_dump(mode="json"))

    # 2. Terminal failure or cancellation states: Return HTTP 409 Conflict
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

    # 3. In-flight processing states: Return HTTP 202 Accepted
    resp = SessionReportResponse(
        session_id=session_id,
        status=state.status,
        created_at=state.created_at,
        message="Assessment evaluation and report generation are currently processing.",
    )
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content=resp.model_dump(mode="json"),
    )


@router.post(
    "/{session_id}/cancel",
    response_model=CancelSessionResponse,
    summary="Cancel an active interview session",
)
async def cancel_session(
    session_id: str,
    request: Request,
    claims: Dict[str, Any] = Depends(require_service_token),
) -> CancelSessionResponse:
    """Idempotently cancel an interview session."""
    state_mgr: SessionStateManager = request.app.state.state_mgr

    state = state_mgr.get_state(session_id)
    if state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found",
        )

    # If already cancelled, return existing state idempotently
    if state.status == LifecycleStatus.CANCELLED:
        return CancelSessionResponse(
            session_id=session_id,
            status=LifecycleStatus.CANCELLED,
            message="Session already cancelled",
        )

    # Terminal state check
    if state.status in (LifecycleStatus.REPORT_READY, LifecycleStatus.FAILED, LifecycleStatus.EXPIRED):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot cancel session in terminal status '{state.status.value}'",
        )

    # Transition to CANCELLED
    updated = state_mgr.transition(
        state,
        expected_version=state.state_version,
        new_status=LifecycleStatus.CANCELLED,
    )

    return CancelSessionResponse(
        session_id=session_id,
        status=updated.status,
        message="Session cancelled successfully",
    )


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
    """Delete session workspace, artifacts, and registry tracking under retention flow."""
    state_mgr: SessionStateManager = request.app.state.state_mgr
    cleanup_mgr: WorkspaceCleanupManager = request.app.state.cleanup_mgr

    state = state_mgr.get_state(session_id)
    if state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found",
        )

    # Run cleanup deletion
    try:
        cleanup_mgr.delete_workspace(session_id)
    except Exception as e:
        logger.warning(
            "Encountered error during workspace deletion for '%s': %s",
            session_id,
            sanitize_secrets(str(e)),
        )

    return DeleteSessionResponse(
        session_id=session_id,
        status="DELETED",
        message="Session and associated workspace artifacts deleted successfully",
    )

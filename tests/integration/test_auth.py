"""Integration tests for JWT authentication, service tokens, and session token scopes."""

import pytest
import time
from api.auth import (
    create_service_token,
    verify_service_token,
    create_session_token,
    verify_session_token,
    AuthError,
)


def test_service_token_validation():
    """Verify service token generation and expected audience/issuer validation."""
    secret = "test_jwt_secret_32_chars_long_1234"
    token = create_service_token(issuer="backend", audience="ai-runtime", secret=secret)
    claims = verify_service_token(token, secret=secret, expected_audience="ai-runtime")
    assert claims["iss"] == "backend"
    assert claims["aud"] == "ai-runtime"

    with pytest.raises(AuthError):
        verify_service_token(token, secret=secret, expected_audience="wrong-aud")


def test_service_token_issuer_check():
    """Verify service token verification rejects mismatched issuer if specified."""
    secret = "test_jwt_secret_32_chars_long_1234"
    token = create_service_token(issuer="backend", audience="ai-runtime", secret=secret)
    
    # Matching issuer passes
    claims = verify_service_token(token, secret=secret, expected_audience="ai-runtime", expected_issuer="backend")
    assert claims["iss"] == "backend"

    # Mismatched issuer fails
    with pytest.raises(AuthError):
        verify_service_token(token, secret=secret, expected_audience="ai-runtime", expected_issuer="other-service")


def test_service_token_expiration():
    """Verify that expired service tokens raise AuthError."""
    secret = "test_jwt_secret_32_chars_long_1234"
    token = create_service_token(
        issuer="backend",
        audience="ai-runtime",
        secret=secret,
        expires_minutes=-5,
    )
    with pytest.raises(AuthError):
        verify_service_token(token, secret=secret, expected_audience="ai-runtime")


def test_service_token_tampered_secret():
    """Verify that verifying with a different secret raises AuthError."""
    secret1 = "test_jwt_secret_32_chars_long_1234"
    secret2 = "wrong_jwt_secret_32_chars_long_5678"
    token = create_service_token(issuer="backend", audience="ai-runtime", secret=secret1)

    with pytest.raises(AuthError):
        verify_service_token(token, secret=secret2, expected_audience="ai-runtime")


def test_session_token_validation():
    """Verify session token creation and scoped session_id validation."""
    secret = "test_jwt_secret_32_chars_long_1234"
    token = create_session_token(session_id="sess_123", secret=secret)
    claims = verify_session_token(token, secret=secret, expected_session_id="sess_123")
    assert claims["session_id"] == "sess_123"
    assert claims["aud"] == "ai-interview-session"

    # Verifying with wrong expected session_id must fail
    with pytest.raises(AuthError):
        verify_session_token(token, secret=secret, expected_session_id="sess_other_456")


def test_session_token_expiration():
    """Verify expired session token raises AuthError."""
    secret = "test_jwt_secret_32_chars_long_1234"
    token = create_session_token(session_id="sess_123", secret=secret, expires_minutes=-10)
    with pytest.raises(AuthError):
        verify_session_token(token, secret=secret, expected_session_id="sess_123")


def test_auth_error_sanitizes_token_content():
    """Verify AuthError never leaks raw token content in error representations."""
    secret = "test_jwt_secret_32_chars_long_1234"
    token = create_service_token(issuer="backend", audience="ai-runtime", secret=secret)
    
    with pytest.raises(AuthError) as exc_info:
        verify_service_token(token, secret="wrong_secret_12345678901234567890", expected_audience="ai-runtime")
    
    # Raw token or secret must not appear in the exception string
    err_str = str(exc_info.value)
    assert token not in err_str
    assert "wrong_secret_12345678901234567890" not in err_str


@pytest.mark.asyncio
async def test_rest_health_probes(tmp_path):
    """Verify /health/live and /health/ready endpoints."""
    from httpx import ASGITransport, AsyncClient
    from api.main import create_app

    app = create_app(workspace_base=str(tmp_path / "workspaces"), fake_redis=True)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        live_res = await ac.get("/health/live")
        assert live_res.status_code == 200
        assert live_res.json()["live"] is True

        ready_res = await ac.get("/health/ready")
        assert ready_res.status_code == 200
        assert ready_res.json()["status"] == "ready"
        assert ready_res.json()["details"]["redis"] == "connected"
        assert ready_res.json()["details"]["workspace"] == "writable"


@pytest.mark.asyncio
async def test_rest_create_session_and_idempotency(tmp_path):
    """Verify session creation, idempotency replay, and payload mismatch conflict (409)."""
    from httpx import ASGITransport, AsyncClient
    from api.main import create_app
    from config.settings import get_settings

    settings = get_settings()
    app = create_app(workspace_base=str(tmp_path / "workspaces"), fake_redis=True)

    token = create_service_token(issuer="backend", audience="ai-runtime", secret=settings.JWT_SECRET)
    headers = {"Authorization": f"Bearer {token}", "Idempotency-Key": "test_key_001"}

    payload1 = {
        "candidate_name": "Nguyen Van A",
        "experience_years": 5,
        "target_company": "Tech Corp",
        "target_role": "Backend Engineer",
        "skills": ["Python", "FastAPI"],
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # 1. Initial creation -> 202 PROFILING
        res1 = await ac.post("/api/v1/sessions", json=payload1, headers=headers)
        assert res1.status_code == 202
        data1 = res1.json()
        assert data1["status"] == "PROFILING"
        assert "session_id" in data1
        assert "session_token" in data1
        sess_id = data1["session_id"]

        # 2. Repeated request with identical key + payload -> 202 with cached response
        res2 = await ac.post("/api/v1/sessions", json=payload1, headers=headers)
        assert res2.status_code == 202
        data2 = res2.json()
        assert data2["session_id"] == sess_id

        # 3. Repeated request with same key but differing payload -> 409 Conflict
        payload_conflict = dict(payload1, candidate_name="Different Name")
        res3 = await ac.post("/api/v1/sessions", json=payload_conflict, headers=headers)
        assert res3.status_code == 409

        # 4. Status retrieval with service token
        status_res = await ac.get(f"/api/v1/sessions/{sess_id}", headers=headers)
        assert status_res.status_code == 200
        assert status_res.json()["session_id"] == sess_id

        # 5. Status retrieval with candidate session token
        sess_token = data1["session_token"]
        cand_headers = {"Authorization": f"Bearer {sess_token}"}
        cand_status_res = await ac.get(f"/api/v1/sessions/{sess_id}", headers=cand_headers)
        assert cand_status_res.status_code == 200
        assert cand_status_res.json()["session_id"] == sess_id

        # 6. Session token accessing different session -> 403 Forbidden
        other_res = await ac.get("/api/v1/sessions/other_session_id", headers=cand_headers)
        assert other_res.status_code == 403

        # 7. Unauthenticated request -> 401 Unauthorized
        unauth_res = await ac.get(f"/api/v1/sessions/{sess_id}")
        assert unauth_res.status_code == 401


@pytest.mark.asyncio
async def test_rest_cancel_and_delete_session(tmp_path):
    """Verify session cancellation and retention deletion."""
    from httpx import ASGITransport, AsyncClient
    from api.main import create_app
    from config.settings import get_settings

    settings = get_settings()
    app = create_app(workspace_base=str(tmp_path / "workspaces"), fake_redis=True)
    token = create_service_token(issuer="backend", audience="ai-runtime", secret=settings.JWT_SECRET)
    headers = {"Authorization": f"Bearer {token}"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Create session
        c_res = await ac.post("/api/v1/sessions", json={"candidate_name": "Test Candidate"}, headers=headers)
        sess_id = c_res.json()["session_id"]

        # Cancel session -> 200 CANCELLED
        cancel_res = await ac.post(f"/api/v1/sessions/{sess_id}/cancel", headers=headers)
        assert cancel_res.status_code == 200
        assert cancel_res.json()["status"] == "CANCELLED"

        # Idempotent cancel
        repeat_cancel = await ac.post(f"/api/v1/sessions/{sess_id}/cancel", headers=headers)
        assert repeat_cancel.status_code == 200
        assert repeat_cancel.json()["status"] == "CANCELLED"

        # Check status is CANCELLED
        stat_res = await ac.get(f"/api/v1/sessions/{sess_id}", headers=headers)
        assert stat_res.json()["status"] == "CANCELLED"

        # Delete session
        del_res = await ac.delete(f"/api/v1/sessions/{sess_id}", headers=headers)
        assert del_res.status_code == 200
        assert del_res.json()["status"] == "DELETED"


@pytest.mark.asyncio
async def test_rest_report_flow(tmp_path):
    """Verify report status endpoints (202 during processing, 200 on REPORT_READY, 409 on cancel)."""
    from httpx import ASGITransport, AsyncClient
    from api.main import create_app
    from core.session.state import LifecycleStatus
    from config.settings import get_settings

    settings = get_settings()
    app = create_app(workspace_base=str(tmp_path / "workspaces"), fake_redis=True)
    token = create_service_token(issuer="backend", audience="ai-runtime", secret=settings.JWT_SECRET)
    headers = {"Authorization": f"Bearer {token}"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        c_res = await ac.post("/api/v1/sessions", json={"candidate_name": "Report Candidate"}, headers=headers)
        sess_id = c_res.json()["session_id"]

        # In PROFILING -> report returns 202 Accepted
        rep_res = await ac.get(f"/api/v1/sessions/{sess_id}/report", headers=headers)
        assert rep_res.status_code == 202

        # Simulate reaching REPORT_READY with generated report files
        ws = app.state.workspace_mgr.get_or_create_workspace(sess_id)
        ws.write_artifact("reports/interview_report.md", artifact_type="report_markdown", payload="### Interview Report\nStrong skills.")
        ws.write_artifact("reports/final_assessment.json", artifact_type="final_assessment", payload={"overall_score": 8.5})

        store = app.state.state_mgr
        curr_state = await store.aget(sess_id)
        # Advance through valid transitions to REPORT_READY
        s2 = await store.atransition(curr_state, expected_version=curr_state.state_version, new_status=LifecycleStatus.READY)
        s3 = await store.atransition(s2, expected_version=s2.state_version, new_status=LifecycleStatus.STAGE_ACTIVE)
        s4 = await store.atransition(s3, expected_version=s3.state_version, new_status=LifecycleStatus.DIALOGUE_COMPLETED)
        s5 = await store.atransition(s4, expected_version=s4.state_version, new_status=LifecycleStatus.ASSESSING)
        await store.atransition(s5, expected_version=s5.state_version, new_status=LifecycleStatus.REPORT_READY)

        # Now report should return 200 OK
        ready_rep = await ac.get(f"/api/v1/sessions/{sess_id}/report", headers=headers)
        assert ready_rep.status_code == 200
        rep_data = ready_rep.json()
        assert rep_data["status"] == "REPORT_READY"
        assert "Interview Report" in rep_data["report_markdown"]
        assert rep_data["final_assessment"]["overall_score"] == 8.5


@pytest.mark.asyncio
async def test_candidate_self_cancel_session(tmp_path):
    """Verify that a candidate using their session token can cancel their own session and not others."""
    from httpx import ASGITransport, AsyncClient
    from api.main import create_app
    from config.settings import get_settings

    settings = get_settings()
    app = create_app(workspace_base=str(tmp_path / "workspaces"), fake_redis=True)
    service_token = create_service_token(issuer="backend", audience="ai-runtime", secret=settings.JWT_SECRET)
    service_headers = {"Authorization": f"Bearer {service_token}"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Create session
        c_res = await ac.post("/api/v1/sessions", json={"candidate_name": "Self Cancel Candidate"}, headers=service_headers)
        assert c_res.status_code == 202
        sess_id = c_res.json()["session_id"]
        sess_token = c_res.json()["session_token"]
        candidate_headers = {"Authorization": f"Bearer {sess_token}"}

        # Candidate cancels their own session -> 200 CANCELLED
        cancel_res = await ac.post(f"/api/v1/sessions/{sess_id}/cancel", headers=candidate_headers)
        assert cancel_res.status_code == 200
        assert cancel_res.json()["status"] == "CANCELLED"

        # Verify candidate cannot cancel another session -> 403 Forbidden
        other_cancel = await ac.post("/api/v1/sessions/other_session_999/cancel", headers=candidate_headers)
        assert other_cancel.status_code == 403



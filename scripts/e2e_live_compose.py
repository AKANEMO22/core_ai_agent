"""Live Compose acceptance for Brave search, CV planning, interview, report, and MinIO.

Run from the repository root with:
    Get-Content scripts/e2e_live_compose.py -Raw | docker compose exec -T app python -

Uses a synthetic CV and the configured providers. The generated session is deleted after
the archive has been restored and verified; no provider keys or candidate text are printed.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
import websockets

from api.auth import create_service_token
from config.settings import get_settings
from core.storage import ArtifactArchive, S3ObjectStore
from core.workspace.session import SessionWorkspaceManager


BASE_URL = "http://127.0.0.1:8000"
WS_BASE_URL = "ws://127.0.0.1:8000"

SYNTHETIC_CV = """Candidate: Synthetic Candidate
Career objective: Build reliable backend services and grow into technical ownership.
Education: Example Institute of Technology, BSc Computer Science, 2022-2026; GPA 3.7/4.0.
Experience: Software Engineering Intern at Example Labs, 2025-2026.
Responsibilities: Implemented Python APIs and added Redis caching with tests.
Project: Campus Events Platform. Problem: event searches slowed during registration.
Technologies: Python, FastAPI, PostgreSQL, Redis, Docker.
Contribution: Designed the cache key and invalidation path; added load tests.
Outcome: p95 search latency fell from 800 ms to 250 ms in a local benchmark.
Skills: Python, SQL, Redis, REST APIs, Git, Docker, teamwork.
Certification: AWS Certified Cloud Practitioner, Example Academy, 2025.
Award: First place, Example Institute Software Project Showcase, 2025.
"""

SYNTHETIC_JD = """Software Engineer Intern
Build and test backend APIs in Python, review SQL queries, use Redis where appropriate,
write automated tests, explain project decisions, and work with product and engineering teams.
"""

ANSWERS = [
    "I am a computer science student focused on backend engineering, and I would like to introduce the campus events project.",
    "The search endpoint slowed during registration. I measured it, added a Redis cache with explicit invalidation, and tested the change.",
    "I chose a cache key based on the event filter and organization. The benchmark reduced p95 latency from 800 to 250 milliseconds locally.",
    "I would first clarify freshness and traffic requirements, then compare database indexes, caching, and read replicas with measurements.",
    "When the first cache invalidation approach returned stale results, I reproduced it, added a regression test, and changed the invalidation path.",
    "I coordinated with the registration team to agree which event changes must invalidate the cache, then documented the tradeoff and rollout.",
    "I tested cache hits, misses, invalidation, and concurrent updates, and added an integration check for stale event data.",
    "What does success look like for an intern on this team, and how are code reviews handled?",
]


def payload(path: Path) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    value = artifact.get("payload", artifact)
    return value if isinstance(value, dict) else {}


async def receive_turn(socket: Any) -> dict[str, Any]:
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        raw = await asyncio.wait_for(socket.recv(), timeout=max(1, deadline - time.monotonic()))
        frame = json.loads(raw)
        frame_type = frame.get("type")
        if frame_type in {"error", "busy"}:
            code = (frame.get("payload") or {}).get("code", "UNKNOWN")
            if code in {"TURN_IN_PROGRESS", "GENERATION_BUSY"} or frame_type == "busy":
                return frame
            raise RuntimeError(f"Interview websocket error: {code}")
        if frame_type == "turn_complete":
            return frame
    raise TimeoutError("Interview turn did not finish within 180 seconds")


async def run() -> dict[str, Any]:
    settings = get_settings()
    if not settings.BRAVE_SEARCH_API_KEY or not settings.llm_api_key:
        raise RuntimeError("Live Compose acceptance requires Brave and LLM provider keys")

    service_token = create_service_token(
        issuer="backend", audience="ai-runtime", secret=settings.JWT_SECRET
    )
    service_headers = {"Authorization": f"Bearer {service_token}"}
    session_id: str | None = None
    client = httpx.AsyncClient(base_url=BASE_URL, timeout=httpx.Timeout(300.0))
    result: dict[str, Any] = {}
    try:
        form_payload = {
            "candidate_name": "Synthetic Candidate",
            "interview_language": "vi",
            "target_company": "Microsoft",
            "target_role": "Software Engineer Intern",
        }
        created = await client.post(
            "/api/v1/sessions/upload",
            headers=service_headers,
            files=[
                ("payload", (None, json.dumps(form_payload), "application/json")),
                ("cv_file", ("synthetic-cv.txt", SYNTHETIC_CV.encode("utf-8"), "text/plain")),
                ("job_description_file", ("synthetic-jd.txt", SYNTHETIC_JD.encode("utf-8"), "text/plain")),
            ],
        )
        created.raise_for_status()
        creation = created.json()
        session_id = creation["session_id"]
        session_token = creation["session_token"]

        deadline = time.monotonic() + 300
        status: dict[str, Any] = {}
        while time.monotonic() < deadline:
            response = await client.get(f"/api/v1/sessions/{session_id}", headers=service_headers)
            response.raise_for_status()
            status = response.json()
            if status["status"] == "READY":
                break
            if status["status"] in {"FAILED", "ASSESSMENT_FAILED"}:
                raise RuntimeError(f"Profiling failed: {status.get('failure_code', 'unknown')}")
            await asyncio.sleep(1)
        if status.get("status") != "READY":
            raise TimeoutError("Session profiling did not become ready within 300 seconds")
        if status.get("retrieval_status") != "ok":
            raise RuntimeError("RAG retrieval did not produce a usable knowledge pack")

        workspace = SessionWorkspaceManager(root_base=settings.WORKSPACE_ROOT_BASE).get_or_create_workspace(session_id)
        profile = payload(workspace.resolve_path("context/candidate_profile.json"))
        initial_brief = payload(workspace.resolve_path("context/interview_brief.json"))
        initial_plan = payload(workspace.resolve_path("context/stage_plan.json"))
        recon = payload(workspace.resolve_path("context/company_recon.json"))
        rubric = payload(workspace.resolve_path("context/rubric.json"))
        form_input = payload(workspace.resolve_path("inputs/form_input.json"))

        profile_fields = (
            "career_goals", "experience", "projects", "skills", "education", "awards", "certifications"
        )
        extracted_fields = [field for field in profile_fields if profile.get(field)]
        if len(extracted_fields) != len(profile_fields):
            missing = sorted(set(profile_fields) - set(extracted_fields))
            raise AssertionError(f"CV extraction missed standard sections: {missing}")
        documents = {row.get("kind"): row for row in form_input.get("documents", []) if isinstance(row, dict)}
        for kind, original_text in (("cv", SYNTHETIC_CV), ("job_description", SYNTHETIC_JD)):
            record = documents.get(kind) or {}
            relative_path = record.get("path")
            if not relative_path or not workspace.exists(relative_path):
                raise AssertionError(f"Original {kind} upload was not retained")
            original_bytes = workspace.resolve_path(relative_path).read_bytes()
            if hashlib.sha256(original_bytes).hexdigest() != record.get("sha256"):
                raise AssertionError(f"Original {kind} upload hash does not match its retained metadata")
            if original_bytes.decode("utf-8") != original_text:
                raise AssertionError(f"Original {kind} upload bytes changed")
        if not rubric.get("competencies"):
            raise AssertionError("RAG rubric is empty")
        web_sources = recon.get("web_recon") or []
        if recon.get("web_recon_status") != "available" or not web_sources:
            raise AssertionError("Brave company research returned no source cards")
        if any(not source.get("url") or not source.get("trust_tier") for source in web_sources):
            raise AssertionError("Company research source cards are missing URL or trust provenance")
        initial_duration = initial_brief.get("duration") or {}
        if not initial_duration.get("planned_minutes") or not initial_duration.get("source"):
            raise AssertionError("Omitted duration was not resolved with provenance")
        if initial_brief.get("seniority", {}).get("value") != "intern":
            raise AssertionError("Intern seniority was not resolved from the target role")
        initial_stages = initial_plan.get("stages", [])
        if not initial_stages or initial_stages[0].get("stage_id") != "opening":
            raise AssertionError("Interview plan does not begin with the introduction stage")
        if initial_stages[-1].get("stage_id") != "candidate_questions":
            raise AssertionError("Interview plan does not finish with candidate Q&A")
        if not any(
            stage.get("criteria")
            for stage in initial_stages
            if stage.get("stage_id") not in {"opening", "candidate_questions"}
        ):
            raise AssertionError("Interview plan has no scored core stage")

        # Exercise the review-before-start flow and keep the live run short and bounded.
        reviewed = await client.patch(
            f"/api/v1/sessions/{session_id}/plan",
            headers=service_headers,
            json={"duration_minutes": 10},
        )
        reviewed.raise_for_status()
        reviewed_plan = reviewed.json()["stage_plan"]
        if reviewed_plan.get("planned_minutes") != 10:
            raise AssertionError("Reviewed duration was not applied to the stage plan")

        transient_busy_retries = 0
        interview_answer_turns = 0
        observed_stages: list[str] = []
        body_answers = ANSWERS[1:-1]
        planned_turns = sum(int(stage.get("target_turns") or 1) for stage in reviewed_plan.get("stages", []))
        max_answer_turns = min(20, max(10, planned_turns + 8))
        async with websockets.connect(
            f"{WS_BASE_URL}/ws/sessions/{session_id}/stream",
            additional_headers={"Authorization": f"Bearer {session_token}"},
            open_timeout=20,
            close_timeout=10,
        ) as socket:
            await socket.send(json.dumps({"type": "start_interview"}))
            opening = await receive_turn(socket)
            opening_payload = opening.get("payload", {})
            if opening_payload.get("turn_number") != 0 or opening_payload.get("stage_id") != "opening":
                raise AssertionError("Opening interview turn was not generated")
            opening_text = str(opening_payload.get("interviewer_response") or "").lower()
            planned_round = str((initial_brief.get("interview_round") or {}).get("value") or "")
            round_terms = {
                "combined": ("combined", "kết hợp", "tổng hợp"),
                "technical": ("technical", "kỹ thuật"),
                "hr": (" hr ", "nhân sự", "hành chính"),
                "hiring_manager": ("hiring manager", "quản lý tuyển dụng", "quản lý"),
            }.get(planned_round.lower(), tuple(re.split(r"[_\s-]+", planned_round.lower())))
            duration = int(reviewed_plan.get("planned_minutes") or initial_duration["planned_minutes"])
            if "synthetic" not in opening_text:
                raise AssertionError("Opening did not address the candidate by the submitted name")
            english_role = "software" in opening_text and "intern" in opening_text
            vietnamese_role = "thực tập" in opening_text and any(
                term in opening_text for term in ("phần mềm", "software", "kỹ sư")
            )
            if not (english_role or vietnamese_role):
                raise AssertionError("Opening did not name the target role")
            if round_terms and not any(term in opening_text for term in round_terms):
                raise AssertionError("Opening did not name the planned interview round")
            has_duration = bool(re.search(rf"\b{duration}[\s-]*minutes?\b", opening_text)) or f"{duration} phút" in opening_text
            if not has_duration:
                raise AssertionError("Opening did not state the planned duration")
            vietnamese_intro = "giới thiệu" in opening_text and "bản thân" in opening_text
            english_intro = any(phrase in opening_text for phrase in (
                "introduce yourself", "tell me about yourself", "introduce your",
            ))
            if not (vietnamese_intro or english_intro):
                raise AssertionError("Opening did not invite a candidate introduction")
            if " tuổi" in opening_text or "age" in opening_text:
                raise AssertionError("Opening asked an unrelated age question")
            observed_stages = [str(opening_payload.get("stage_id") or "")]

            candidate_questions_answered = False
            for index in range(1, max_answer_turns + 1):
                live_state_response = await client.get(
                    f"/api/v1/sessions/{session_id}", headers=service_headers
                )
                live_state_response.raise_for_status()
                live_state = live_state_response.json()
                current_stage = str(live_state.get("current_stage_id") or "")
                if current_stage == "candidate_questions":
                    answer = ANSWERS[-1]
                    candidate_questions_answered = True
                elif index == 1:
                    answer = ANSWERS[0]
                else:
                    answer = body_answers[(index - 2) % len(body_answers)]
                expected_turn = int(live_state.get("current_turn") or 0) + 1
                retry = 0
                while True:
                    await socket.send(json.dumps({
                        "type": "candidate_answer",
                        "message_id": f"live-e2e-{index}",
                        "candidate_answer": answer,
                        "expected_turn": expected_turn,
                    }))
                    turn = await receive_turn(socket)
                    if turn.get("type") not in {"error", "busy"}:
                        break
                    retry += 1
                    transient_busy_retries += 1
                    if retry > 5:
                        raise RuntimeError("Interview turn remained temporarily busy after bounded retries")
                    delay_ms = int((turn.get("payload") or {}).get("retry_after_ms") or 250)
                    await asyncio.sleep(max(0.15, min(2.0, delay_ms / 1000)))
                if turn.get("payload", {}).get("turn_number") != expected_turn:
                    raise AssertionError(f"Interview turn {expected_turn} did not complete in order")
                interview_answer_turns += 1
                observed_stages.append(current_stage)
                if candidate_questions_answered:
                    break

        if not candidate_questions_answered or "candidate_questions" not in observed_stages:
            raise AssertionError("The complete interview did not reach the candidate Q&A and closing stage")

        report_deadline = time.monotonic() + 240
        report: dict[str, Any] = {}
        while time.monotonic() < report_deadline:
            response = await client.get(f"/api/v1/sessions/{session_id}/report", headers=service_headers)
            if response.status_code == 200:
                report = response.json()
                break
            if response.status_code != 202:
                raise RuntimeError(f"Report endpoint returned HTTP {response.status_code}")
            await asyncio.sleep(1)
        if report.get("status") != "REPORT_READY" or not report.get("report_markdown"):
            raise TimeoutError("Live interview did not produce a final report")

        object_store = S3ObjectStore(
            endpoint_url=settings.OBJECT_STORE_ENDPOINT,
            access_key=settings.OBJECT_STORE_ACCESS_KEY,
            secret_key=settings.OBJECT_STORE_SECRET_KEY,
            region=settings.OBJECT_STORE_REGION,
            verify_tls=settings.OBJECT_STORE_VERIFY_TLS,
            timeout_seconds=settings.OBJECT_STORE_TIMEOUT_SECONDS,
        )
        archive = ArtifactArchive(
            object_store=object_store,
            bucket=settings.ARTIFACT_ARCHIVE_BUCKET,
            catalog_path=settings.ARTIFACT_ARCHIVE_CATALOG_PATH,
            max_restore_bytes=settings.ARTIFACT_ARCHIVE_MAX_RESTORE_BYTES,
        )
        archive_status = archive.status(session_id)
        if not archive_status or archive_status.get("state") != "archived":
            raise AssertionError("MinIO archive was not finalized")
        with tempfile.TemporaryDirectory(prefix="rag-live-e2e-") as temporary:
            restored = Path(temporary) / "restored-workspace"
            await asyncio.to_thread(archive.restore_session, session_id, restored, owner_id=session_id)
            if not (restored / "reports" / "decision.json").is_file():
                raise AssertionError("MinIO restore did not include the final decision")
            if not (restored / "context" / "candidate_profile.json").is_file():
                raise AssertionError("MinIO restore did not include the extracted CV profile")
            for filename in ("cv_original.txt", "job_description_original.txt"):
                if not (restored / "inputs" / "documents" / filename).is_file():
                    raise AssertionError(f"MinIO restore did not include {filename}")

        metrics = (report.get("final_assessment") or {}).get("metrics") or {}
        result = {
            "result": "PASS",
            "rag_status": status.get("retrieval_status"),
            "company_sources": len(web_sources),
            "company_source_cards_with_provenance": len(web_sources),
            "source_trust": sorted({str(source.get("trust_tier", "unreviewed")) for source in web_sources}),
            "cv_sections_extracted": extracted_fields,
            "original_uploads_retained": sorted(documents),
            "seniority": initial_brief.get("seniority", {}).get("value"),
            "duration_inferred_minutes": initial_duration.get("planned_minutes"),
            "duration_inference_source": initial_duration.get("source"),
            "reviewed_duration_minutes": reviewed_plan.get("planned_minutes"),
            "interview_answer_turns": interview_answer_turns,
            "transient_busy_retries": transient_busy_retries,
            "interview_stages_observed": list(dict.fromkeys(observed_stages)),
            "report_status": report.get("status"),
            "evaluated_turns": metrics.get("total_turns_evaluated"),
            "minio_archive_revision": archive_status.get("revision"),
            "minio_restore": "verified",
        }
    finally:
        if session_id is not None:
            deleted = await client.delete(f"/api/v1/sessions/{session_id}", headers=service_headers)
            if deleted.status_code not in (200, 404):
                raise RuntimeError(f"Synthetic E2E cleanup returned HTTP {deleted.status_code}")
            if result:
                result["synthetic_session_cleanup"] = deleted.json().get("status", "not_found")
        await client.aclose()
    return result


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run()), ensure_ascii=True, sort_keys=True))

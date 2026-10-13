"""Live end-to-end run against the configured real LLM (staging smoke test).

Every agent (profiler, brief, speaker, orchestrator, evaluator, planners, narrative) uses the real
provider; a second model instance plays the candidate at a target level. Results (transcript,
report, decision, timings) are written to data/live_runs/<timestamp>-<persona>/.

    RUN_LIVE_LLM=1 uv run python -m core.simulation.live --persona strong --persona weak
"""

import argparse
import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import fakeredis.aioredis
from httpx import ASGITransport, AsyncClient

from api.auth import create_service_token
from api.main import create_app
from config.settings import get_settings
from core.runtime.providers.openai import OpenAIRuntimeLLMClient
from core.session.state import LifecycleStatus

PERSONAS: Dict[str, Dict[str, str]] = {
    "strong": {
        "level": "a strong senior backend engineer",
        "style": "Give concrete, structured answers with real trade-offs, numbers and failure modes.",
    },
    "weak": {
        "level": "a junior engineer applying above their level",
        "style": "Answer vaguely, stay at buzzword level, skip trade-offs and failure handling, sometimes misunderstand.",
    },
    "average": {
        "level": "a mid-level backend engineer",
        "style": "Give reasonable answers with some detail but miss deeper trade-offs and edge cases.",
    },
}


@dataclass
class LiveCandidate:
    """LLM-played candidate."""

    name: str
    llm: OpenAIRuntimeLLMClient

    async def answer(self, question: str, history: List[Dict[str, str]]) -> str:
        persona = PERSONAS[self.name]
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are {persona['level']} in a spoken job interview for a Senior Backend Engineer role "
                    f"at Acme Systems. {persona['style']} Answer only the interviewer's latest question in 3-6 "
                    "spoken sentences. No lists, no markdown. Stay in character."
                ),
            },
            *history[-8:],
            {"role": "user", "content": question},
        ]
        for _ in range(3):  # some gateways occasionally return an empty completion
            tokens: List[str] = []
            async for chunk in self.llm.stream_chat(messages=messages, temperature=0.7):
                if chunk.token:
                    tokens.append(chunk.token)
            text = "".join(tokens).strip()
            if text:
                return text
        return "Could you repeat the question, please?"


RELAXED_PROFILES = json.dumps({
    "speaker": {"turn_deadline_ms": 45000},
    "evaluator": {"turn_deadline_ms": 120000},
    "summary": {"turn_deadline_ms": 45000},
    "followup": {"turn_deadline_ms": 45000},
    "guard": {"turn_deadline_ms": 30000},
    "claims": {"turn_deadline_ms": 60000},
    "brief": {"turn_deadline_ms": 120000},
    "narrative": {"turn_deadline_ms": 180000},
})


async def run_persona(
    name: str,
    out_root: Path,
    max_turns: int = 20,
    relaxed: bool = False,
    model: Optional[str] = None,
    speaker_model: Optional[str] = None,
) -> Dict[str, Any]:
    update: Dict[str, Any] = {
        "ASSESSOR_BLOCK_MS": 50, "ASSESSOR_RECOVERY_INTERVAL_SECONDS": 0.5, "REPORT_SWEEP_INTERVAL_SECONDS": 1.0,
    }
    if relaxed:
        # Slow (buffering / reasoning) gateways: validate behavior, not latency.
        update.update({
            "AGENT_PROFILES_OVERRIDES": RELAXED_PROFILES,
            "TURN_DEADLINE_SECONDS": 60.0,
            "ORCHESTRATOR_DEADLINE_MS": 120000,
            "ORCHESTRATOR_JOIN_MS": 90000,
            "PROFILING_DEADLINE_SECONDS": 300.0,
        })
    if model:
        update["LLM_MODEL"] = model
    if speaker_model:
        overrides = json.loads(update.get("AGENT_PROFILES_OVERRIDES") or "{}")
        overrides.setdefault("speaker", {})["model"] = speaker_model
        update["AGENT_PROFILES_OVERRIDES"] = json.dumps(overrides)
    settings = get_settings().model_copy(update=update)
    out_dir = out_root / f"{time.strftime('%Y%m%d-%H%M%S')}-{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    app = create_app(
        workspace_base=str(out_dir / "workspaces"),
        redis_client=fakeredis.aioredis.FakeRedis(decode_responses=True),
        settings=settings,
    )
    candidate = LiveCandidate(name, OpenAIRuntimeLLMClient(model=settings.llm_model))
    token = create_service_token(issuer="live", audience="ai-runtime", secret=settings.JWT_SECRET)
    timings: Dict[str, Any] = {"turns": []}

    started = time.perf_counter()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://live", timeout=300) as client:
        res = await client.post(
            "/api/v1/sessions",
            json={
                "candidate_name": f"Live {name}",
                "experience_years": 6 if name != "weak" else 2,
                "target_company": "Acme Systems",
                "target_role": "Senior Backend Engineer",
                "skills": ["Python", "PostgreSQL", "Redis", "Kafka"],
                "cv_text": "Backend engineer. Built payment APIs, led a ledger migration, scaled a service to 20k rps.",
            },
            headers={"Authorization": f"Bearer {token}", "Idempotency-Key": uuid.uuid4().hex},
        )
        res.raise_for_status()
        session_id = res.json()["session_id"]
    timings["profiling_seconds"] = round(time.perf_counter() - started, 2)

    await app.state.worker_pool.start()
    controller = app.state.controller
    workspace = app.state.workspace_mgr.get_or_create_workspace(session_id)
    state = await app.state.state_mgr.aget(session_id)
    transcript: List[Dict[str, Any]] = []
    history: List[Dict[str, str]] = []
    try:
        if state.status != LifecycleStatus.READY:
            raise RuntimeError(f"profiling did not reach READY: {state.status} {state.metadata}")

        t0 = time.perf_counter()
        opening = await controller.start_interview(session_id)
        timings["opening_seconds"] = round(time.perf_counter() - t0, 2)
        if opening.status not in ("completed", "interrupted"):
            raise RuntimeError(f"opening failed: {opening.error_code}")
        question = str(workspace.read_json("turns/turn_0_interviewer.json").get("content", ""))
        transcript.append({"turn": 0, "interviewer": question})

        for turn in range(1, max_turns + 1):
            state = await app.state.state_mgr.aget(session_id)
            if state.status != LifecycleStatus.STAGE_ACTIVE:
                break
            # The candidate "speaks" while the control plane finishes the previous cycle.
            answer_task = asyncio.create_task(candidate.answer(question, history))
            drain_start = time.perf_counter()
            await controller.drain(timeout=60)
            control_plane_seconds = time.perf_counter() - drain_start
            answer = await answer_task
            history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]

            before = len(app.state.metrics._histograms.get("ttft_seconds", {}).get((), {}).get("values", []))
            t_turn = time.perf_counter()
            result = await controller.submit_answer(session_id, f"live-{turn}", answer)
            retries = 0
            while result.status == "failed" and retries < 2:  # a client retries the same message_id
                retries += 1
                result = await controller.submit_answer(session_id, f"live-{turn}", answer)
            timings.setdefault("turn_retries", 0)
            timings["turn_retries"] += retries
            turn_seconds = time.perf_counter() - t_turn
            values = app.state.metrics._histograms.get("ttft_seconds", {}).get((), {}).get("values", [])
            ttft = values[-1] if len(values) > before else None
            if result.status not in ("completed", "interrupted"):
                raise RuntimeError(f"turn {turn} failed: {result.error_code}")
            artifact = workspace.read_json(f"turns/turn_{turn}_interviewer.json")
            question = str(artifact.get("content", ""))
            transcript.append({
                "turn": turn,
                "candidate": answer,
                "interviewer": question,
                "stage": artifact.get("stage_id"),
                "directive_mode": artifact.get("directive_mode"),
                "directive_source": artifact.get("directive_source"),
                "target_criterion": artifact.get("target_criterion"),
            })
            timings["turns"].append({
                "turn": turn,
                "ttft_seconds": round(ttft, 3) if ttft is not None else None,
                "turn_seconds": round(turn_seconds, 3),
                "control_plane_wait_seconds": round(control_plane_seconds, 3),
            })
            print(f"[{name}] turn {turn}: ttft={timings['turns'][-1]['ttft_seconds']}s "
                  f"stage={artifact.get('stage_id')} directive={artifact.get('directive_source')}/{artifact.get('directive_mode')}")

        await controller.drain(timeout=60)
        deadline = time.perf_counter() + 300
        report_started = time.perf_counter()
        while time.perf_counter() < deadline:
            state = await app.state.state_mgr.aget(session_id)
            if state.status in (LifecycleStatus.REPORT_READY, LifecycleStatus.ASSESSMENT_FAILED, LifecycleStatus.FAILED):
                break
            await asyncio.sleep(0.5)
        timings["report_seconds_after_dialogue"] = round(time.perf_counter() - report_started, 2)
    finally:
        await app.state.worker_pool.stop(grace_seconds=10)

    metrics = app.state.metrics
    decision = json.loads(workspace.read_text("reports/decision.json")) if workspace.exists("reports/decision.json") else {}
    evaluations = []
    for path in sorted(workspace.list_artifacts("evaluations")):
        data = workspace.read_json(path.relative_to(workspace.root_path).as_posix())
        evaluations.append(data.get("payload", data))
    ttfts = sorted(t["ttft_seconds"] for t in timings["turns"] if t["ttft_seconds"] is not None)
    summary = {
        "persona": name,
        "session_id": session_id,
        "final_status": state.status.value,
        "verdict": decision.get("verdict"),
        "weighted_score": decision.get("weighted_score"),
        "verdict_reasons": decision.get("verdict_reasons"),
        "turns": len(timings["turns"]),
        "ttft_p50": ttfts[len(ttfts) // 2] if ttfts else None,
        "ttft_max": ttfts[-1] if ttfts else None,
        "orchestrator_directives": sum(1 for t in transcript if t.get("directive_source") == "orchestrator"),
        "fallback_directives": sum(1 for t in transcript if t.get("directive_source") == "fallback"),
        "degraded": {str(k): v for k, v in metrics._counters.get("degraded_total", {}).items()},
        "provider_errors": {str(k): v for k, v in metrics._counters.get("provider_errors_total", {}).items()},
        "speaker_hedges": {str(k): v for k, v in metrics._counters.get("speaker_hedge_total", {}).items()},
        "stage_extensions": sum(
            1
            for path in workspace.list_artifacts("orchestration")
            if path.name.startswith("directive_")
            and (workspace.read_json(path.relative_to(workspace.root_path).as_posix()).get("payload") or {}).get("extend_stage")
        ),
        "must_have_borderline": decision.get("must_have_borderline"),
        "evaluation_modes": sorted({e.get("evaluation_mode", "?") for e in evaluations}),
        "unverified_quotes_dropped": sum(int((e.get("flags") or {}).get("unverified_quotes_dropped", 0)) for e in evaluations),
        "brief_status": state.metadata.get("brief_status"),
        "models": {"default": settings.llm_model, "speaker": speaker_model or settings.llm_model},
        "retrieval_status": state.metadata.get("retrieval_status"),
        "timings": timings,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "transcript.json").write_text(json.dumps(transcript, indent=2, ensure_ascii=False), encoding="utf-8")
    if workspace.exists("reports/interview_report.md"):
        (out_dir / "interview_report.md").write_text(workspace.read_text("reports/interview_report.md"), encoding="utf-8")
    summary["out_dir"] = str(out_dir)
    return summary


async def main_async(personas: List[str], out_root: Path, relaxed: bool = False,
                     model: Optional[str] = None, speaker_model: Optional[str] = None) -> None:
    for name in personas:
        summary = await run_persona(name, out_root, relaxed=relaxed, model=model, speaker_model=speaker_model)
        printable = {k: v for k, v in summary.items() if k != "timings"}
        print(json.dumps(printable, indent=2, ensure_ascii=True))


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Live end-to-end interview with the configured LLM")
    parser.add_argument("--persona", action="append", choices=sorted(PERSONAS), default=None)
    parser.add_argument("--out", type=Path, default=Path("data/live_runs"))
    parser.add_argument("--relaxed", action="store_true", help="long deadlines for slow gateways")
    parser.add_argument("--model", help="model for all agents (overrides LLM_MODEL)")
    parser.add_argument("--speaker-model", help="model for the realtime speaker only")
    args = parser.parse_args(argv)
    asyncio.run(main_async(args.persona or ["strong"], args.out, args.relaxed, args.model, args.speaker_model))


if __name__ == "__main__":
    main()

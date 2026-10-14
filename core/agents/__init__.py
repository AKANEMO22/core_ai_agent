"""Specialized subagents and their registration with the delegation runtime.

Điểm đăng ký các agent phụ vào delegation runtime.
Hàm register_subagents nhận client LLM, workspace và cấu hình profile; tạo wrapper
đổi task/session_id thành lời gọi planner, claims hoặc guard rồi trả các instance.
CriterionEvaluator được xuất để assessor dùng riêng, không đăng ký trong hàm này.
"""

import asyncio
from typing import Any, Dict, Mapping, Optional

from core.agents.claims import ClaimChecker
from core.agents.evaluator import CriterionEvaluator, verify_quote
from core.agents.followup import FollowUpPlanner
from core.agents.guard import InputGuard, heuristic_flags
from core.runtime.client import RuntimeLLMClient
from core.runtime.profiles import AgentProfile

__all__ = [
    "ClaimChecker",
    "CriterionEvaluator",
    "FollowUpPlanner",
    "InputGuard",
    "heuristic_flags",
    "register_subagents",
    "verify_quote",
]


def register_subagents(
    delegation: Any,
    llm: Any,
    workspace_mgr: Any,
    retrieval_client: Optional[Any] = None,
    profiles: Optional[Mapping[str, AgentProfile]] = None,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> Dict[str, Any]:
    """Register follow-up planner, claim checker and input guard as delegable agents."""
    profiles = profiles or {}

    def deadline(name: str, default_ms: int) -> int:
        profile = profiles.get(name)
        return profile.turn_deadline_ms if profile else default_ms

    def client(name: str) -> RuntimeLLMClient:
        # `llm` is either one client or a factory returning the client for a profile name
        return llm(name) if callable(llm) and not hasattr(llm, "stream_chat") else llm

    planner = FollowUpPlanner(client("followup"), retrieval_client, deadline("followup", 6000) / 1000, semaphore)
    claims = ClaimChecker(client("claims"), deadline_seconds=deadline("claims", 15000) / 1000, semaphore=semaphore)
    guard = InputGuard(client("guard"), deadline("guard", 4000) / 1000, semaphore)

    # Wrapper giữ giao diện task của runtime, còn agent nhận workspace và dữ liệu cụ thể.
    async def run_followup(session_id: str, task: Dict[str, Any]) -> Dict[str, Any]:
        ws = workspace_mgr.get_or_create_workspace(session_id)
        return await planner.plan(ws, int(task["turn_number"]), task.get("question", ""), task.get("answer", ""))

    async def run_claims(session_id: str, task: Dict[str, Any]) -> Dict[str, Any]:
        ws = workspace_mgr.get_or_create_workspace(session_id)
        return await claims.check(ws, int(task["turn_number"]))

    async def run_guard(session_id: str, task: Dict[str, Any]) -> Dict[str, Any]:
        return await guard.check(task.get("question", ""), task.get("answer", ""))

    # Giới hạn ở đây là concurrency theo agent; semaphore chung còn giới hạn các cuộc gọi LLM.
    delegation.register("followup", run_followup, concurrency=64, default_deadline_ms=deadline("followup", 6000))
    delegation.register("claims", run_claims, concurrency=16, default_deadline_ms=deadline("claims", 15000))
    delegation.register("guard", run_guard, concurrency=64, default_deadline_ms=deadline("guard", 4000))
    return {"followup": planner, "claims": claims, "guard": guard}

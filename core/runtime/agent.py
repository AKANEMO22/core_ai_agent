"""Core streaming agent loop with tool execution, soft schema repair, and budget control.

Luồng sử dụng: Vòng điều khiển một lượt agent: nhận lịch sử chat, gọi provider, phát LoopEvent cho tầng session.
API lắp RuntimeAgent với profile và tool; run_loop_streaming nối kết quả tool vào lịch sử
rồi gọi model tiếp, đến khi có câu trả lời cuối hoặc chạm giới hạn vòng/thời gian/hủy.
Policy, hook và schema được kiểm tra trước khi chạy tool; workspace giữ đầu ra lớn ngoài prompt.
"""

import asyncio
import json
import logging
import re
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

from core.runtime.client import RuntimeLLMClient
from core.runtime.compaction import ContextCompactor, strip_private_keys
from core.runtime.hooks import HookBus, HookContext, HookEvent
from core.runtime.policy import ToolPolicy
from core.runtime.profiles import AgentProfile
from core.runtime.sanitize import sanitize_secrets  # noqa: F401  (re-exported)
from core.runtime.tooling import RuntimeTool
from core.runtime.types import IterationChunk, LoopEvent, LoopEventType, LoopStatus
from core.workspace.session import SessionWorkspace
from core.workspace.spill import WorkspaceSpillManager

logger = logging.getLogger(__name__)


class RuntimeAgent:
    """Core autonomous streaming agent coordinating the LLM execution loop.

    Features:
        - Fast-path token streaming without blocking on full turn completion.
        - Tool execution with auto schema validation and self-correcting repair loops.
        - Payload spillover integration for tool outputs exceeding 32 KB.
        - Turn budget enforcement with automatic final response nudging.
        - Safe cancellation handling and strict credential sanitization in events.

    Attributes:
        client: The underlying LLM provider adapter.
        tools: Registered executable tools.
        workspace: Optional active session workspace for artifact storage and spillover.
        max_iterations: Hard upper bound on tool execution iterations per turn.
        schema_repair_budget: Max retry attempts allowed for model schema violations.
        transient_retry_budget: Max transient retry attempts for failing tools.
        profile: Optional agent profile (model, tool allowlist, deadline, compaction).
        hooks: Optional hook bus receiving lifecycle events.
        policy: Tool policy enforcing the profile allowlist.
        compactor: Optional context compactor applied before model calls.
        metrics: Optional metrics registry.
        turn_deadline_seconds: Wall-clock deadline of one loop run (overrides the profile).
    """

    def __init__(
        self,
        client: RuntimeLLMClient,
        tools: Optional[List[RuntimeTool]] = None,
        workspace: Optional[SessionWorkspace] = None,
        max_iterations: Optional[int] = None,
        schema_repair_budget: int = 1,
        transient_retry_budget: int = 1,
        profile: Optional[AgentProfile] = None,
        hooks: Optional[HookBus] = None,
        policy: Optional[ToolPolicy] = None,
        compactor: Optional[ContextCompactor] = None,
        metrics: Optional[Any] = None,
        turn_deadline_seconds: Optional[float] = None,
    ) -> None:
        self.client = client
        self.tools = tools or []
        self.tools_map: Dict[str, RuntimeTool] = {t.name: t for t in self.tools}
        self.workspace = workspace
        self.profile = profile
        self.max_iterations = (
            max_iterations if max_iterations is not None else (profile.max_iterations if profile else 5)
        )
        self.schema_repair_budget = schema_repair_budget
        self.transient_retry_budget = transient_retry_budget
        self.hooks = hooks
        self.policy = policy or ToolPolicy()
        self.compactor = compactor
        self.metrics = metrics
        if turn_deadline_seconds is not None:
            self.turn_deadline_seconds: Optional[float] = turn_deadline_seconds
        elif profile is not None:
            self.turn_deadline_seconds = profile.turn_deadline_ms / 1000
        else:
            self.turn_deadline_seconds = None
        self.status = LoopStatus.IDLE

    @property
    def agent_name(self) -> str:
        """Profile name used for hooks and metrics."""
        return self.profile.name if self.profile else "agent"

    async def _emit_hook(self, event: HookEvent, session_id: str, turn_id: str, **data: Any) -> Any:
        if self.hooks is None:
            return None
        return await self.hooks.emit(
            event,
            HookContext(event=event, session_id=session_id, turn_id=turn_id, agent=self.agent_name, data=data),
        )

    def _record_usage(self, usage: Dict[str, Any]) -> None:
        if self.metrics is None or not usage:
            return
        for kind in ("prompt_tokens", "completion_tokens"):
            value = usage.get(kind)
            if isinstance(value, (int, float)):
                self.metrics.inc("llm_tokens_total", float(value), {"agent": self.agent_name, "kind": kind})

    async def run_loop_streaming(
        self,
        session_id: str,
        turn_id: str,
        messages: List[Dict[str, Any]],
        cancellation_token: Optional[asyncio.Event] = None,
        **kwargs: Any,
    ) -> AsyncIterator[LoopEvent]:
        """Execute the agent streaming loop for a single dialogue turn.

        Emits an ordered stream of LoopEvents containing progressive generation tokens,
        tool invocations, intermediate results, and completion or error statuses.

        Args:
            session_id: Session identifier.
            turn_id: Turn identifier.
            messages: Input chat messages conversation history.
            cancellation_token: Optional asyncio event signaling external cancellation.
            **kwargs: Extra parameters passed to the LLM client.

        Yields:
            LoopEvent: Standardized lifecycle event.
        """
        sequence = 0
        self.status = LoopStatus.RUNNING

        def create_event(event_type: LoopEventType, payload: Dict[str, Any]) -> LoopEvent:
            nonlocal sequence
            sequence += 1
            return LoopEvent(
                schema_version="1.0",
                event_id=f"evt_{uuid.uuid4().hex[:16]}",
                session_id=session_id,
                turn_id=turn_id,
                sequence=sequence,
                timestamp=time.time(),
                type=event_type,
                payload=payload,
            )

        # Sao chép danh sách để lịch sử làm việc của lượt không nối thêm phần tử vào input của caller.
        working_messages = list(messages)
        visible_tools = self.policy.visible_tools(self.profile, self.tools)
        formatted_tools = [t.to_openai_format() for t in visible_tools] if visible_tools else None
        schema_repairs_used = 0
        iteration = 0
        full_response_text = ""
        accumulated_usage: Dict[str, Any] = {}
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.turn_deadline_seconds if self.turn_deadline_seconds else None
        if self.profile is not None:
            if self.profile.model and "model" not in kwargs:
                kwargs["model"] = self.profile.model
            kwargs.setdefault("temperature", self.profile.temperature)

        await self._emit_hook(HookEvent.TURN_START, session_id, turn_id, message_count=len(working_messages))

        try:
            # Mỗi vòng gọi model có thể tạo tool call; kết quả tool được nối lại vào lịch sử rồi model chạy tiếp.
            while iteration < self.max_iterations:
                iteration += 1

                # Compact the working context before each model call when the profile asks for it
                if (
                    self.compactor is not None
                    and self.profile is not None
                    and self.compactor.needs_compaction(
                        working_messages, self.profile.context_budget_tokens, self.profile.compaction
                    )
                ):
                    await self._emit_hook(HookEvent.BEFORE_COMPACTION, session_id, turn_id)
                    compaction = await self.compactor.maybe_compact(
                        working_messages,
                        self.profile.context_budget_tokens,
                        self.profile.compaction,
                        agent=self.agent_name,
                    )
                    if compaction.compacted:
                        working_messages = compaction.messages
                        if self.metrics is not None:
                            self.metrics.inc("compactions_total", 1.0, {"agent": self.agent_name})
                        yield create_event(
                            LoopEventType.COMPACTION,
                            {
                                "steps": compaction.steps,
                                "tokens_before": compaction.tokens_before,
                                "tokens_after": compaction.tokens_after,
                            },
                        )
                    await self._emit_hook(HookEvent.AFTER_COMPACTION, session_id, turn_id, steps=compaction.steps)

                # Check cancellation token at iteration start
                if cancellation_token and cancellation_token.is_set():
                    self.status = LoopStatus.CANCELLED
                    yield create_event(
                        LoopEventType.ERROR,
                        {"error": "Execution was cancelled", "code": "CANCELLED"},
                    )
                    return

                # Emit progress event at iteration start
                yield create_event(
                    LoopEventType.PROGRESS,
                    {"iteration": iteration, "status": "generating"},
                )

                turn_tokens: List[str] = []
                turn_tool_calls: List[Dict[str, Any]] = []
                await self._emit_hook(HookEvent.BEFORE_MODEL, session_id, turn_id, iteration=iteration)

                # Stream from LLM provider under the turn deadline
                stream = self.client.stream_chat(
                    messages=strip_private_keys(working_messages),
                    tools=formatted_tools,
                    **kwargs,
                )
                # Deadline dùng đồng hồ monotonic và áp lên từng lần chờ chunk, không chỉ lúc mở request.
                iterator = stream.__aiter__()
                try:
                    while True:
                        remaining = None if deadline is None else deadline - loop.time()
                        if remaining is not None and remaining <= 0:
                            raise asyncio.TimeoutError()
                        try:
                            if remaining is None:
                                chunk = await iterator.__anext__()
                            else:
                                chunk = await asyncio.wait_for(iterator.__anext__(), timeout=remaining)
                        except StopAsyncIteration:
                            break
                        if cancellation_token and cancellation_token.is_set():
                            self.status = LoopStatus.CANCELLED
                            yield create_event(
                                LoopEventType.ERROR,
                                {"error": "Execution was cancelled", "code": "CANCELLED"},
                            )
                            return

                        # Emit reasoning preview summary if present
                        if chunk.reasoning_summary:
                            yield create_event(
                                LoopEventType.REASONING_SUMMARY,
                                {"summary": chunk.reasoning_summary},
                            )

                        # Emit fast-path token event
                        if chunk.token:
                            turn_tokens.append(chunk.token)
                            yield create_event(
                                LoopEventType.RESPONSE_TOKEN,
                                {"token": chunk.token},
                            )

                        # Collect tool calls and usage
                        if chunk.tool_calls:
                            turn_tool_calls.extend(chunk.tool_calls)

                        if chunk.usage:
                            accumulated_usage.update(chunk.usage)

                except asyncio.TimeoutError:
                    self.status = LoopStatus.FAILED
                    full_response_text += "".join(turn_tokens)
                    yield create_event(
                        LoopEventType.ERROR,
                        {
                            "error": f"Turn deadline of {self.turn_deadline_seconds}s exceeded",
                            "code": "TURN_DEADLINE",
                            "final_response": full_response_text,
                        },
                    )
                    return
                except asyncio.CancelledError:
                    self.status = LoopStatus.CANCELLED
                    yield create_event(
                        LoopEventType.ERROR,
                        {"error": "Execution cancelled by task termination", "code": "CANCELLED"},
                    )
                    return
                except Exception as provider_err:
                    self.status = LoopStatus.FAILED
                    if self.metrics is not None:
                        self.metrics.inc("provider_errors_total", 1.0, {"agent": self.agent_name})
                    clean_msg = sanitize_secrets(str(provider_err))
                    yield create_event(
                        LoopEventType.ERROR,
                        {
                            "error": f"Provider error: {clean_msg}",
                            "code": "PROVIDER_ERROR",
                        },
                    )
                    return
                finally:
                    aclose = getattr(iterator, "aclose", None)
                    if aclose is not None:
                        try:
                            await aclose()
                        except Exception:
                            pass

                turn_text = "".join(turn_tokens)
                full_response_text += turn_text
                self._record_usage(accumulated_usage)
                await self._emit_hook(
                    HookEvent.AFTER_MODEL,
                    session_id,
                    turn_id,
                    iteration=iteration,
                    tool_calls=[tc.get("name") for tc in turn_tool_calls],
                    usage=dict(accumulated_usage),
                )

                # Case 1: No tool calls emitted -> Generation completed
                if not turn_tool_calls:
                    self.status = LoopStatus.COMPLETED
                    await self._emit_hook(HookEvent.TURN_END, session_id, turn_id, status="completed")
                    yield create_event(
                        LoopEventType.COMPLETE,
                        {
                            "final_response": full_response_text,
                            "total_iterations": iteration,
                            "usage": accumulated_usage,
                        },
                    )
                    return

                # Case 2: Tool calls emitted -> Execute tools and feed results back
                self.status = LoopStatus.WAITING_TOOL

                # Normalize IDs across all tool calls once prior to building assistant_record
                for tc in turn_tool_calls:
                    if not tc.get("id"):
                        tc["id"] = f"call_{uuid.uuid4().hex[:8]}"

                assistant_record: Dict[str, Any] = {
                    "role": "assistant",
                    "content": turn_text if turn_text else None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc.get("name"),
                                "arguments": (
                                    json.dumps(tc.get("arguments"))
                                    if isinstance(tc.get("arguments"), dict)
                                    else str(tc.get("arguments") if tc.get("arguments") is not None else "{}")
                                ),
                            },
                        }
                        for tc in turn_tool_calls
                    ],
                }
                working_messages.append(assistant_record)

                schema_budget_exceeded = False

                for tc in turn_tool_calls:
                    call_id = tc["id"]
                    tool_name = tc.get("name", "")
                    raw_args = tc.get("arguments")

                    if raw_args is None:
                        parsed_args = {}
                    elif isinstance(raw_args, str):
                        try:
                            parsed = json.loads(raw_args)
                            parsed_args = parsed if isinstance(parsed, dict) else {"input": parsed}
                        except json.JSONDecodeError:
                            parsed_args = {"input": raw_args}
                    elif isinstance(raw_args, dict):
                        parsed_args = dict(raw_args)
                    else:
                        parsed_args = {}

                    # Emit TOOL_CALL_START event
                    yield create_event(
                        LoopEventType.TOOL_CALL_START,
                        {
                            "tool_name": tool_name,
                            "call_id": call_id,
                            "arguments": parsed_args,
                        },
                    )

                    tool = self.tools_map.get(tool_name)
                    if not tool:
                        error_res = f"Error: Tool '{tool_name}' not found."
                        yield create_event(
                            LoopEventType.TOOL_CALL_RESULT,
                            {
                                "tool_name": tool_name,
                                "call_id": call_id,
                                "result": error_res,
                            },
                        )
                        working_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call_id,
                                "name": tool_name,
                                "content": error_res,
                            }
                        )
                        continue

                    # Policy (profile allowlist) and blocking BEFORE_TOOL hooks may deny execution
                    decision = self.policy.check(self.profile, tool_name)
                    deny_reason = None if decision.allowed else decision.reason
                    if deny_reason is None:
                        hook_outcome = await self._emit_hook(
                            HookEvent.BEFORE_TOOL, session_id, turn_id, tool_name=tool_name, arguments=parsed_args
                        )
                        if hook_outcome is not None and hook_outcome.denied:
                            deny_reason = "; ".join(hook_outcome.reasons) or "denied by hook"
                    if deny_reason is not None:
                        denied_msg = f"Error: tool call denied ({deny_reason})."
                        yield create_event(
                            LoopEventType.TOOL_CALL_RESULT,
                            {"tool_name": tool_name, "call_id": call_id, "result": denied_msg, "code": "TOOL_DENIED"},
                        )
                        working_messages.append(
                            {"role": "tool", "tool_call_id": call_id, "name": tool_name, "content": denied_msg}
                        )
                        continue

                    # Validate schema
                    # Lỗi schema thành tool result để model tự sửa; ngân sách sửa dùng chung cho cả lượt.
                    is_valid, violations = tool.validate_arguments(parsed_args)
                    if not is_valid:
                        if schema_repairs_used < self.schema_repair_budget:
                            schema_repairs_used += 1
                            repair_feedback = (
                                f"Tool argument validation failed: {'; '.join(violations)}. "
                                "Please correct the parameters according to schema and retry."
                            )
                        else:
                            schema_budget_exceeded = True
                            repair_feedback = (
                                f"Schema validation error: {'; '.join(violations)}. "
                                "Schema repair budget exceeded. Do not attempt further tool calls; "
                                "synthesize your final response to the candidate now."
                            )

                        yield create_event(
                            LoopEventType.TOOL_CALL_RESULT,
                            {
                                "tool_name": tool_name,
                                "call_id": call_id,
                                "result": repair_feedback,
                                "schema_violations": violations,
                            },
                        )
                        working_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call_id,
                                "name": tool_name,
                                "content": repair_feedback,
                            }
                        )
                        continue

                    # Execute tool with retry budget
                    execution_result: Any = None
                    last_exec_error: Optional[str] = None
                    for attempt in range(1 + self.transient_retry_budget):
                        try:
                            execution_result = await tool.execute(
                                parsed_args, workspace=self.workspace
                            )
                            last_exec_error = None
                            break
                        except Exception as tool_err:
                            last_exec_error = sanitize_secrets(str(tool_err))
                            if attempt < self.transient_retry_budget:
                                await asyncio.sleep(0.05)

                    if last_exec_error is not None:
                        exec_msg = f"Error executing tool '{tool_name}': {last_exec_error}"
                        yield create_event(
                            LoopEventType.TOOL_CALL_RESULT,
                            {
                                "tool_name": tool_name,
                                "call_id": call_id,
                                "result": exec_msg,
                            },
                        )
                        working_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call_id,
                                "name": tool_name,
                                "content": exec_msg,
                            }
                        )
                        continue

                    # Check spillover threshold if session workspace is attached
                    if self.workspace:
                        spill_mgr = WorkspaceSpillManager(self.workspace)
                        # Event giữ đường dẫn/preview; prompt chỉ nhận phần gọn để đầu ra lớn không làm phình context.
                        spill_info = spill_mgr.maybe_spill(tool_name, execution_result)
                        if spill_info["is_spilled"]:
                            preview_text = spill_info["preview"]
                            rel_path = spill_info["rel_spill_path"]
                            event_payload = {
                                "tool_name": tool_name,
                                "call_id": call_id,
                                "result": preview_text,
                                "preview": preview_text,
                                "is_spilled": True,
                                "spill_path": spill_info["spill_path"],
                                "original_size_bytes": spill_info["original_size_bytes"],
                            }
                            tool_content = (
                                f"<Large output spilled to {rel_path} "
                                f"({spill_info['original_size_bytes']} bytes)>\n"
                                f"Preview: {preview_text}"
                            )
                        else:
                            event_payload = {
                                "tool_name": tool_name,
                                "call_id": call_id,
                                "result": execution_result,
                                "is_spilled": False,
                            }
                            tool_content = (
                                json.dumps(execution_result)
                                if isinstance(execution_result, (dict, list))
                                else str(execution_result)
                            )
                    else:
                        event_payload = {
                            "tool_name": tool_name,
                            "call_id": call_id,
                            "result": execution_result,
                            "is_spilled": False,
                        }
                        tool_content = (
                            json.dumps(execution_result)
                            if isinstance(execution_result, (dict, list))
                            else str(execution_result)
                        )

                    yield create_event(LoopEventType.TOOL_CALL_RESULT, event_payload)
                    working_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": tool_name,
                            "content": tool_content,
                        }
                    )
                    await self._emit_hook(
                        HookEvent.AFTER_TOOL,
                        session_id,
                        turn_id,
                        tool_name=tool_name,
                        is_spilled=bool(event_payload.get("is_spilled")),
                    )

                # Schema repair budget nudge: if budget exhausted, instruct model to answer directly
                if schema_budget_exceeded:
                    working_messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Schema repair budget exhausted. Please synthesize your final "
                                "response to the candidate now without initiating further tool calls."
                            ),
                        }
                    )

                # Tool nudge: if approaching budget limit, instruct the model to finish
                if iteration == self.max_iterations - 1:
                    working_messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Tool iteration budget reached. Please synthesize your final "
                                "response to the candidate now without initiating further tool calls."
                            ),
                        }
                    )

            # Loop budget exhausted
            self.status = LoopStatus.FAILED
            yield create_event(
                LoopEventType.ERROR,
                {
                    "error": f"Agent exceeded maximum iterations budget ({self.max_iterations})",
                    "code": "MAX_ITERATIONS_EXCEEDED",
                    "final_response": full_response_text,
                },
            )

        except Exception as unhandled_err:
            self.status = LoopStatus.FAILED
            clean_err = sanitize_secrets(str(unhandled_err))
            yield create_event(
                LoopEventType.ERROR,
                {
                    "error": f"Unhandled runtime error: {clean_err}",
                    "code": "INTERNAL_ERROR",
                },
            )

"""Tool policy: which tools an agent profile may see and execute.

Luồng sử dụng: RuntimeAgent dùng policy để lọc tool gửi cho model và kiểm tra lại khi model yêu cầu chạy.
Đầu vào là AgentProfile và tên tool; PolicyDecision trả quyền cùng lý do từ chối.
Không có profile thì cho phép mọi tool đã đăng ký; có profile thì dùng allowed_tools.
"""

from dataclasses import dataclass
from typing import Iterable, List, Optional

from core.runtime.profiles import AgentProfile
from core.runtime.tooling import RuntimeTool


@dataclass(frozen=True)
class PolicyDecision:
    """Outcome of a policy check."""

    allowed: bool
    reason: str = ""


class ToolPolicy:
    """Allowlist enforcement for agent tools.

    Defense in depth: tools outside the profile are not advertised to the model, and a call to
    one anyway (hallucinated name, prompt injection) is refused at execution time.
    """

    def check(self, profile: Optional[AgentProfile], tool_name: str) -> PolicyDecision:
        """Decide whether `profile` may execute `tool_name`. No profile means unrestricted."""
        if profile is None:
            return PolicyDecision(True)
        if tool_name in profile.allowed_tools:
            return PolicyDecision(True)
        return PolicyDecision(False, f"tool '{tool_name}' is not allowed for agent '{profile.name}'")

    def visible_tools(self, profile: Optional[AgentProfile], tools: Iterable[RuntimeTool]) -> List[RuntimeTool]:
        """Tools to advertise to the model for this profile."""
        # Dùng cùng một phép kiểm tra cho quảng bá và thực thi để hai ranh giới không lệch nhau.
        return [t for t in tools if self.check(profile, t.name).allowed]

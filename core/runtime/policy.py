"""Tool policy: which tools an agent profile may see and execute."""

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
        return [t for t in tools if self.check(profile, t.name).allowed]

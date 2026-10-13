"""Agent profiles: per-role model, tools, budgets, deadlines and compaction policy."""

import json
import logging
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Mapping, Optional

logger = logging.getLogger(__name__)

DEV_ONLY_TOOLS: FrozenSet[str] = frozenset({"bash", "read_file", "write_file", "edit_file"})
DEFAULT_PROFILES_PATH = Path(__file__).resolve().parents[2] / "config" / "agent_profiles.json"


@dataclass(frozen=True)
class CompactionPolicy:
    """When and how the runtime compacts a growing message list.

    Attributes:
        enabled: Whether compaction runs for this profile.
        threshold_ratio: Fraction of the context budget that triggers compaction.
        keep_last_messages: Most recent messages never compacted.
        summary_max_tokens: Target size of the generated summary.
    """

    enabled: bool = False
    threshold_ratio: float = 0.7
    keep_last_messages: int = 6
    summary_max_tokens: int = 600


@dataclass(frozen=True)
class AgentProfile:
    """Configuration of one agent role.

    Attributes:
        name: Profile name (speaker, orchestrator, evaluator, ...).
        model: Provider model id; empty uses the client's default.
        temperature: Sampling temperature.
        allowed_tools: Tool names the agent may see and call. The speaker has none.
        max_iterations: Model calls per turn (1 = no tool loop).
        turn_deadline_ms: Hard deadline for one turn of this agent.
        context_budget_tokens: Prompt budget used by context assembly and compaction.
        compaction: Compaction policy.
        output_schema_name: Name of the JSON schema for structured agents, if any.
    """

    name: str
    model: str = ""
    temperature: float = 0.3
    allowed_tools: FrozenSet[str] = frozenset()
    max_iterations: int = 1
    turn_deadline_ms: int = 30000
    context_budget_tokens: int = 6000
    compaction: CompactionPolicy = field(default_factory=CompactionPolicy)
    output_schema_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["allowed_tools"] = sorted(self.allowed_tools)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AgentProfile":
        values = dict(data)
        values["allowed_tools"] = frozenset(values.get("allowed_tools", []))
        compaction = values.get("compaction") or {}
        values["compaction"] = compaction if isinstance(compaction, CompactionPolicy) else CompactionPolicy(**compaction)
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in values.items() if k in known})


DEFAULT_PROFILES: Dict[str, AgentProfile] = {
    "speaker": AgentProfile(
        name="speaker",
        temperature=0.6,
        allowed_tools=frozenset(),
        max_iterations=1,
        turn_deadline_ms=8000,
        context_budget_tokens=6000,
    ),
    "orchestrator": AgentProfile(
        name="orchestrator",
        temperature=0.2,
        allowed_tools=frozenset({"set_directive", "complete_stage", "retrieve", "read_artifact", "update_notes"}),
        max_iterations=4,
        turn_deadline_ms=8000,
        context_budget_tokens=12000,
        compaction=CompactionPolicy(enabled=True, threshold_ratio=0.9, keep_last_messages=6),
    ),
    "evaluator": AgentProfile(
        name="evaluator", temperature=0.0, max_iterations=1, turn_deadline_ms=60000,
        context_budget_tokens=8000, output_schema_name="criterion_evaluation",
    ),
    "followup": AgentProfile(
        name="followup", temperature=0.3, allowed_tools=frozenset({"retrieve"}), max_iterations=2,
        turn_deadline_ms=10000, context_budget_tokens=4000, output_schema_name="followup_plan",
    ),
    "claims": AgentProfile(
        name="claims", temperature=0.0, max_iterations=1, turn_deadline_ms=15000,
        context_budget_tokens=6000, output_schema_name="claim_check",
    ),
    "guard": AgentProfile(
        name="guard", temperature=0.0, max_iterations=1, turn_deadline_ms=8000,
        context_budget_tokens=2000, output_schema_name="guard_verdict",
    ),
    "summary": AgentProfile(
        name="summary", temperature=0.0, max_iterations=1, turn_deadline_ms=20000,
        context_budget_tokens=6000, output_schema_name="rolling_summary",
    ),
    "brief": AgentProfile(
        name="brief", temperature=0.2, allowed_tools=frozenset({"read_artifact"}), max_iterations=3,
        turn_deadline_ms=60000, context_budget_tokens=16000,
        compaction=CompactionPolicy(enabled=True, threshold_ratio=0.7),
    ),
    "profiler": AgentProfile(
        name="profiler", temperature=0.2, allowed_tools=frozenset({"retrieve", "safe_web_search", "market_intel"}),
        max_iterations=6, turn_deadline_ms=90000, context_budget_tokens=16000,
        compaction=CompactionPolicy(enabled=True, threshold_ratio=0.7),
    ),
    "narrative": AgentProfile(
        name="narrative", temperature=0.2, allowed_tools=frozenset({"read_artifact"}), max_iterations=3,
        turn_deadline_ms=90000, context_budget_tokens=16000,
        compaction=CompactionPolicy(enabled=True, threshold_ratio=0.7),
    ),
}


class ProfileValidationError(ValueError):
    """Raised when profiles violate production policy."""


def validate_profiles(profiles: Mapping[str, AgentProfile], app_env: str = "development") -> List[str]:
    """Return policy violations; development-only tools are forbidden in production."""
    errors: List[str] = []
    production = app_env.strip().lower() == "production"
    for name, profile in profiles.items():
        if production:
            dev_tools = sorted(profile.allowed_tools & DEV_ONLY_TOOLS)
            if dev_tools:
                errors.append(f"profile '{name}' enables development-only tools {dev_tools}")
        if name == "speaker" and profile.allowed_tools:
            errors.append("the speaker profile must not have tools (hot-path latency contract)")
        if profile.max_iterations < 1:
            errors.append(f"profile '{name}' must allow at least one iteration")
    return errors


def load_profiles(
    path: Optional[Path] = None,
    overrides_json: str = "",
    app_env: str = "development",
) -> Dict[str, AgentProfile]:
    """Load profiles: defaults, then `config/agent_profiles.json`, then a JSON override string.

    Raises:
        ProfileValidationError: If the merged profiles violate policy.
    """
    profiles = dict(DEFAULT_PROFILES)
    sources: List[Dict[str, Any]] = []
    file_path = path or DEFAULT_PROFILES_PATH
    if file_path.exists():
        sources.append(json.loads(file_path.read_text(encoding="utf-8")))
    if overrides_json.strip():
        sources.append(json.loads(overrides_json))

    for source in sources:
        for name, values in source.get("profiles", source).items():
            base = profiles.get(name, AgentProfile(name=name)).to_dict()
            merged = {**base, **values, "name": name}
            if "compaction" in values:
                merged["compaction"] = {**base["compaction"], **values["compaction"]}
            profiles[name] = AgentProfile.from_dict(merged)

    errors = validate_profiles(profiles, app_env)
    if errors:
        raise ProfileValidationError("; ".join(errors))
    return profiles


def with_tools(profile: AgentProfile, *tools: str) -> AgentProfile:
    """Return a copy of a profile with extra allowed tools (tests and dev tooling)."""
    return replace(profile, allowed_tools=profile.allowed_tools | frozenset(tools))

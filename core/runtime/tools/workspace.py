"""Read-only and typed-write workspace tools for agents.

Agents never choose filesystem paths for writes: the path is derived from the artifact type,
and the payload is validated against that type's JSON schema.
"""

import json
from typing import Any, Callable, Dict, Optional, Tuple

import jsonschema

from core.runtime.tooling import RuntimeTool
from core.workspace.security import WorkspacePathError
from core.workspace.session import SessionWorkspace

READABLE_PREFIXES: Tuple[str, ...] = ("context/", "turns/", "evaluations/", "orchestration/")
MAX_READ_CHARS = 8000

DIRECTIVE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "mode": {"type": "string", "enum": ["follow_up", "next_question", "transition", "wrap_up", "redirect"]},
        "target_criterion": {"type": ["string", "null"]},
        "question_id": {"type": ["string", "null"]},
        "question_text": {"type": "string", "maxLength": 600},
        "probe_hint": {"type": ["string", "null"], "maxLength": 400},
        "reason": {"type": "string", "maxLength": 400},
    },
    "required": ["mode", "question_text", "reason"],
    "additionalProperties": False,
}

DELEGATION_RESULT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "agent": {"type": "string"},
        "summary": {"type": "string", "maxLength": 2000},
        "data": {"type": "object"},
    },
    "required": ["agent", "summary"],
    "additionalProperties": False,
}

# artifact_type -> (schema, relative path factory)
ARTIFACT_TYPES: Dict[str, Tuple[Dict[str, Any], Callable[[Dict[str, Any], Dict[str, Any]], str]]] = {
    "directive": (DIRECTIVE_SCHEMA, lambda payload, ctx: f"orchestration/directive_{int(ctx['turn_number'])}.json"),
    "delegation_result": (
        DELEGATION_RESULT_SCHEMA,
        lambda payload, ctx: f"orchestration/delegation_{ctx.get('delegation_id', 'adhoc')}.json",
    ),
}


class ReadArtifactTool(RuntimeTool):
    """Read a session artifact under an allowlisted prefix (JSON or text), size-capped."""

    def __init__(self, workspace: SessionWorkspace, prefixes: Tuple[str, ...] = READABLE_PREFIXES) -> None:
        self.workspace = workspace
        self.prefixes = prefixes
        super().__init__(
            name="read_artifact",
            description="Read a session artifact, e.g. 'context/interview_brief.md' or 'evaluations/turn_002_eval.json'.",
            func=self.read,
            param_schema={
                "type": "object",
                "properties": {"path": {"type": "string", "minLength": 3, "maxLength": 200}},
                "required": ["path"],
                "additionalProperties": False,
            },
        )

    async def read(self, path: str, **_: Any) -> Dict[str, Any]:
        normalized = path.replace("\\", "/").lstrip("/")
        if ".." in normalized.split("/") or not normalized.startswith(self.prefixes):
            return {"status": "denied", "error": f"path must start with one of {list(self.prefixes)}"}
        try:
            if not self.workspace.exists(normalized):
                return {"status": "not_found", "path": normalized}
            if normalized.endswith(".json"):
                data = self.workspace.read_json(normalized)
                data = data.get("payload", data) if isinstance(data, dict) else data
                text = json.dumps(data, ensure_ascii=False, default=str)
            else:
                text = self.workspace.read_text(normalized)
        except WorkspacePathError:
            return {"status": "denied", "error": "path escapes the workspace"}
        truncated = len(text) > MAX_READ_CHARS
        return {
            "status": "ok",
            "path": normalized,
            "content": text[:MAX_READ_CHARS],
            "truncated": truncated,
        }


class WriteArtifactTool(RuntimeTool):
    """Write a schema-validated artifact whose path is derived from its type."""

    def __init__(self, workspace: SessionWorkspace, context: Optional[Dict[str, Any]] = None) -> None:
        self.workspace = workspace
        self.context = dict(context or {})
        super().__init__(
            name="write_artifact",
            description="Write a typed artifact (directive, delegation_result). The path is chosen by the runtime.",
            func=self.write,
            param_schema={
                "type": "object",
                "properties": {
                    "artifact_type": {"type": "string", "enum": sorted(ARTIFACT_TYPES)},
                    "payload": {"type": "object"},
                },
                "required": ["artifact_type", "payload"],
                "additionalProperties": False,
            },
        )

    async def write(self, artifact_type: str, payload: Dict[str, Any], **_: Any) -> Dict[str, Any]:
        if artifact_type not in ARTIFACT_TYPES:
            return {"status": "error", "error": f"unknown artifact_type '{artifact_type}'"}
        schema, path_factory = ARTIFACT_TYPES[artifact_type]
        errors = [e.message for e in jsonschema.Draft7Validator(schema).iter_errors(payload)]
        if errors:
            return {"status": "invalid", "errors": errors[:5]}
        try:
            rel_path = path_factory(payload, self.context)
        except (KeyError, ValueError, TypeError):
            return {"status": "error", "error": "runtime context missing for this artifact type"}
        self.workspace.write_artifact(rel_path, artifact_type=artifact_type, payload=payload)
        return {"status": "ok", "path": rel_path}

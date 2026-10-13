"""Structured (JSON-schema validated) model output with bounded repair."""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional

import jsonschema

from core.runtime.client import RuntimeLLMClient
from core.runtime.sanitize import sanitize_secrets


class StructuredOutputError(Exception):
    """Raised when the model cannot produce schema-valid JSON within the repair budget."""

    def __init__(self, message: str, violations: Optional[List[str]] = None) -> None:
        super().__init__(sanitize_secrets(message))
        self.violations = violations or []


def extract_json_payload(text: str) -> Dict[str, Any]:
    """Parse a JSON object from model output (plain, fenced, or embedded in prose).

    Raises:
        ValueError: If no JSON object can be parsed. The message never echoes the text.
    """
    clean_text = (text or "").strip()
    try:
        parsed = json.loads(clean_text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_text)
    if match:
        try:
            parsed = json.loads(match.group(1).strip())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    start, end = clean_text.find("{"), clean_text.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(clean_text[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    raise ValueError("Could not parse valid JSON object from model output")


def schema_violations(schema: Dict[str, Any], data: Any) -> List[str]:
    """Human-readable JSON-schema violations (empty if valid)."""
    violations = []
    for error in jsonschema.Draft7Validator(schema).iter_errors(data):
        path = "/".join(str(p) for p in error.absolute_path)
        violations.append(f"{path or '<root>'}: {error.message}")
    return violations


async def _complete(llm: RuntimeLLMClient, messages: List[Dict[str, Any]], **kwargs: Any) -> str:
    tokens: List[str] = []
    async for chunk in llm.stream_chat(messages=messages, **kwargs):
        if chunk.token:
            tokens.append(chunk.token)
    return "".join(tokens)


async def run_structured(
    llm: RuntimeLLMClient,
    messages: List[Dict[str, Any]],
    schema: Dict[str, Any],
    *,
    max_repairs: int = 1,
    deadline_seconds: Optional[float] = None,
    semaphore: Optional[asyncio.Semaphore] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Call the model and return a JSON object valid against `schema`.

    Uses the provider's JSON mode when its capabilities declare `json_mode`. Invalid output is
    fed back with the violation list for at most `max_repairs` additional attempts.

    Raises:
        StructuredOutputError: When output stays invalid or the deadline is exceeded.
    """
    capabilities = getattr(llm, "capabilities", None)
    if getattr(capabilities, "json_mode", False):
        kwargs.setdefault("response_format", {"type": "json_object"})

    schema_text = json.dumps(schema, ensure_ascii=False)
    contract = (
        "\n\nOutput contract: return exactly one JSON object that validates against this JSON Schema. "
        "Use these property names at the top level; do not wrap the object in another key.\n"
        f"JSON Schema: {schema_text}"
    )

    def _with_contract(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        updated = [dict(m) for m in msgs]
        for message in reversed(updated):
            if message.get("role") == "user":
                message["content"] = f"{message.get('content', '')}{contract}"
                return updated
        return updated + [{"role": "user", "content": contract.strip()}]

    def _unwrap(data: Dict[str, Any]) -> Dict[str, Any]:
        # Models often wrap the payload as {"<kind>": {...}}; accept the inner object if it is valid.
        if len(data) == 1:
            inner = next(iter(data.values()))
            if isinstance(inner, dict) and not schema_violations(schema, inner):
                return inner
        return data

    async def _attempts() -> Dict[str, Any]:
        working = _with_contract(messages)
        last_violations: List[str] = []
        for attempt in range(max_repairs + 1):
            if semaphore is not None:
                async with semaphore:
                    text = await _complete(llm, working, **kwargs)
            else:
                text = await _complete(llm, working, **kwargs)
            try:
                data = _unwrap(extract_json_payload(text))
                last_violations = schema_violations(schema, data)
            except ValueError:
                data, last_violations = None, ["output is not a JSON object"]
            if not last_violations and data is not None:
                return data
            if attempt < max_repairs:
                working = working + [
                    {"role": "assistant", "content": text[:4000]},
                    {
                        "role": "user",
                        "content": "Your JSON did not match the required schema: "
                        + "; ".join(last_violations[:8])
                        + ". Respond again with one corrected JSON object only, matching exactly: "
                        + schema_text[:6000],
                    },
                ]
        raise StructuredOutputError("structured output invalid after repair budget", last_violations)

    try:
        if deadline_seconds is not None:
            return await asyncio.wait_for(_attempts(), timeout=deadline_seconds)
        return await _attempts()
    except asyncio.TimeoutError as e:
        raise StructuredOutputError(f"structured output exceeded {deadline_seconds}s deadline") from e

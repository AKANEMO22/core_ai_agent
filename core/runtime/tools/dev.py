"""Development-only generic tools (read_file, write_file, edit_file, bash) in a sandbox directory.

These mirror a minimal coding-agent harness for offline tooling and experiments. They are refused
when APP_ENV=production, and agent profiles that enable them fail validation in production.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, List

from core.runtime.tooling import RuntimeTool
from core.workspace.security import WorkspacePathError, validate_path_safety

MAX_OUTPUT_CHARS = 8000


class DevToolsDisabledError(RuntimeError):
    """Raised when development tools are requested in production."""


def build_dev_tools(sandbox_root: Path, app_env: str, bash_timeout_seconds: float = 10.0) -> List[RuntimeTool]:
    """Create sandboxed generic tools. Raises DevToolsDisabledError in production."""
    if app_env.strip().lower() == "production":
        raise DevToolsDisabledError("development tools are disabled when APP_ENV=production")
    root = Path(sandbox_root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    def _resolve(path: str) -> Path:
        return validate_path_safety(root, path)

    async def read_file(path: str, **_: Any) -> Dict[str, Any]:
        try:
            target = _resolve(path)
        except WorkspacePathError:
            return {"status": "denied"}
        if not target.is_file():
            return {"status": "not_found"}
        text = target.read_text(encoding="utf-8", errors="replace")
        return {"status": "ok", "content": text[:MAX_OUTPUT_CHARS], "truncated": len(text) > MAX_OUTPUT_CHARS}

    async def write_file(path: str, content: str, **_: Any) -> Dict[str, Any]:
        try:
            target = _resolve(path)
        except WorkspacePathError:
            return {"status": "denied"}
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"status": "ok", "bytes": len(content.encode("utf-8"))}

    async def edit_file(path: str, old_text: str, new_text: str, **_: Any) -> Dict[str, Any]:
        try:
            target = _resolve(path)
        except WorkspacePathError:
            return {"status": "denied"}
        if not target.is_file():
            return {"status": "not_found"}
        text = target.read_text(encoding="utf-8")
        count = text.count(old_text)
        if count != 1:
            return {"status": "invalid", "error": f"old_text must match exactly once (matched {count})"}
        target.write_text(text.replace(old_text, new_text), encoding="utf-8")
        return {"status": "ok"}

    async def bash(command: str, **_: Any) -> Dict[str, Any]:
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=str(root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            output, _ = await asyncio.wait_for(process.communicate(), timeout=bash_timeout_seconds)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return {"status": "timeout"}
        text = output.decode("utf-8", errors="replace")
        return {"status": "ok", "exit_code": process.returncode, "output": text[:MAX_OUTPUT_CHARS]}

    def _schema(props: Dict[str, Any]) -> Dict[str, Any]:
        return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}

    string = {"type": "string"}
    return [
        RuntimeTool("read_file", "Read a file in the sandbox.", read_file, _schema({"path": string})),
        RuntimeTool("write_file", "Write a file in the sandbox.", write_file, _schema({"path": string, "content": string})),
        RuntimeTool(
            "edit_file",
            "Replace exactly one occurrence of old_text in a sandbox file.",
            edit_file,
            _schema({"path": string, "old_text": string, "new_text": string}),
        ),
        RuntimeTool("bash", "Run a shell command in the sandbox (dev only).", bash, _schema({"command": string})),
    ]

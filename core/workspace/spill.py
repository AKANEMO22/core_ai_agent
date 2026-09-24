"""Payload spillover management to disk when tool execution outputs exceed threshold."""

import base64
import json
import re
import uuid
from pathlib import Path
from typing import Any, Optional, Union

from config.settings import get_settings
from core.workspace.session import SessionWorkspace


class WorkspaceSpillManager:
    """Manages spilling large tool output payloads (> 32 KB) to workspace disk.

    Avoids exploding LLM context buffers or WebSocket frame sizes by offloading
    large tool execution outputs into the workspace `spill/` folder and returning
    a truncated preview along with a filesystem reference.

    Attributes:
        workspace: The active SessionWorkspace.
        max_bytes: Byte size threshold beyond which payload is spilled.
    """

    def __init__(
        self,
        workspace: SessionWorkspace,
        max_bytes: Optional[int] = None,
    ) -> None:
        """Initialize spill manager for a workspace.

        Args:
            workspace: The session workspace where spills are stored.
            max_bytes: Maximum allowed payload size before spilling (defaults to MAX_TOOL_RESULT_BYTES).
        """
        self.workspace = workspace
        settings = get_settings()
        self.max_bytes = max_bytes if max_bytes is not None else settings.MAX_TOOL_RESULT_BYTES

    def maybe_spill(self, tool_name: str, payload: Any) -> dict[str, Any]:
        """Check payload size and spill to disk if it exceeds the configured byte threshold.

        Args:
            tool_name: Name of the tool generating the payload.
            payload: Tool execution output (string, dict, list, or bytes).

        Returns:
            dict: Metadata containing either unspilled payload or spill reference with preview.
        """
        is_binary = False
        if isinstance(payload, str):
            payload_str = payload
            payload_bytes = payload.encode("utf-8")
            content_for_artifact = payload
        elif isinstance(payload, (bytes, bytearray)):
            payload_bytes = bytes(payload)
            try:
                payload_str = payload_bytes.decode("utf-8")
                content_for_artifact = payload_str
            except UnicodeDecodeError:
                is_binary = True
                content_for_artifact = base64.b64encode(payload_bytes).decode("ascii")
                payload_str = f"<binary data: {len(payload_bytes)} bytes>"
        elif isinstance(payload, (dict, list)):
            payload_str = json.dumps(payload)
            payload_bytes = payload_str.encode("utf-8")
            content_for_artifact = payload
        else:
            payload_str = str(payload)
            payload_bytes = payload_str.encode("utf-8")
            content_for_artifact = payload_str

        size_bytes = len(payload_bytes)

        if size_bytes <= self.max_bytes:
            return {
                "is_spilled": False,
                "payload": payload,
                "size_bytes": size_bytes,
                "tool_name": tool_name,
            }

        # Payload exceeds threshold: spill to spill/ directory
        clean_tool = re.sub(r"[^a-zA-Z0-9_-]", "_", tool_name)
        spill_filename = f"spill_tool_{clean_tool}_{uuid.uuid4().hex[:12]}.json"
        rel_spill_path = f"spill/{spill_filename}"

        written_path = self.workspace.write_artifact(
            rel_path=rel_spill_path,
            artifact_type="tool_spill",
            payload={
                "tool_name": tool_name,
                "content": content_for_artifact,
                "is_binary": is_binary,
                "size_bytes": size_bytes,
            },
        )

        preview = payload_str[: self.max_bytes]

        return {
            "is_spilled": True,
            "spill_path": str(written_path),
            "rel_spill_path": rel_spill_path,
            "preview": preview,
            "original_size_bytes": size_bytes,
            "tool_name": tool_name,
        }

    def read_spilled(self, spill_path: Union[str, Path]) -> Any:
        """Retrieve original spilled payload from disk.

        Args:
            spill_path: Relative or absolute path to the spilled artifact.

        Returns:
            Any: Original payload stored in the artifact.
        """
        resolved = self.workspace.resolve_path(spill_path)
        with open(resolved, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "content" in data:
            if data.get("is_binary"):
                return base64.b64decode(data["content"].encode("ascii"))
            return data["content"]
        return data

"""Isolated session workspace management with atomic writes, artifact envelopes, and path security."""

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Union

from config.settings import get_settings
from core.workspace.security import validate_path_safety


def normalize_session_id(session_id: str) -> str:
    """Derive a deterministic, safe directory name from a session identifier.

    Format: `session_{clean_id}-{sha256(session_id)[:12]}`.

    Args:
        session_id: Raw session identifier.

    Returns:
        str: Sanitized directory name.
    """
    clean_id = re.sub(r"[^a-zA-Z0-9]+", "_", session_id).strip("_")
    hash_suffix = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:12]
    return f"session_{clean_id}-{hash_suffix}"


class SessionWorkspace:
    """Represents a sandboxed, isolated workspace for an interview session.

    Directory Layout:
        inputs/       - Untrusted candidate resume, JD, initial forms
        context/      - Profiler intelligence artifacts, candidate profile, stage plan
        turns/        - Individual turn states, transcripts, subagent outputs
        evaluations/  - Per-turn rubric scores, assessor results
        reports/      - Synthesized final assessment and candidate report
        spill/        - Spillover storage for payloads exceeding 32 KB

    Attributes:
        session_id: Raw original session identifier.
        root_path: Absolute canonical Path to this workspace's root folder.
        normalized_session_id: Sanitized safe directory name with SHA-256 fingerprint.
    """

    SUBDIRECTORIES: tuple[str, ...] = (
        "inputs",
        "context",
        "turns",
        "evaluations",
        "reports",
        "spill",
    )

    def __init__(
        self,
        session_id: str,
        root_path: Union[str, Path],
        normalized_session_id: Optional[str] = None,
        fsync: bool = True,
    ) -> None:
        """Initialize session workspace reference.

        Args:
            session_id: Raw session identifier.
            root_path: Base directory for the session sandbox.
            normalized_session_id: Optional precomputed normalized session name.
            fsync: Whether to enforce os.fsync on atomic writes.
        """
        self.session_id = session_id
        self.root_path = Path(root_path).resolve()
        self.normalized_session_id = normalized_session_id or normalize_session_id(session_id)
        self.fsync = fsync
        self._json_cache: Dict[str, Any] = {}

    def initialize(self) -> None:
        """Create workspace directory structure on disk if not already existing."""
        self.root_path.mkdir(parents=True, exist_ok=True)
        for subdir in self.SUBDIRECTORIES:
            (self.root_path / subdir).mkdir(parents=True, exist_ok=True)

    @property
    def inputs_dir(self) -> Path:
        """Path to inputs directory."""
        return self.root_path / "inputs"

    @property
    def context_dir(self) -> Path:
        """Path to context directory."""
        return self.root_path / "context"

    @property
    def turns_dir(self) -> Path:
        """Path to turns directory."""
        return self.root_path / "turns"

    @property
    def evaluations_dir(self) -> Path:
        """Path to evaluations directory."""
        return self.root_path / "evaluations"

    @property
    def reports_dir(self) -> Path:
        """Path to reports directory."""
        return self.root_path / "reports"

    @property
    def spill_dir(self) -> Path:
        """Path to spill directory."""
        return self.root_path / "spill"

    def resolve_path(self, rel_path: Union[str, Path]) -> Path:
        """Resolve a path relative to workspace root with strict traversal defense.

        Args:
            rel_path: Target relative or absolute subpath.

        Returns:
            Path: Canonical resolved path within workspace.

        Raises:
            WorkspacePathError: If path attempts to escape the workspace root.
        """
        return validate_path_safety(self.root_path, rel_path)

    def write_artifact(
        self,
        rel_path: Union[str, Path],
        artifact_type: str,
        payload: Any,
        schema_version: str = "1.0",
    ) -> Path:
        """Atomically write an artifact with standard envelope and SHA-256 checksum.

        Enforces atomic creation via temporary file, fsync, and replace to prevent
        corrupted or half-written artifacts.

        Args:
            rel_path: Relative destination within workspace (e.g. 'context/intel.json').
            artifact_type: Classifier tag for the artifact (e.g. 'company_intel').
            payload: JSON-serializable artifact payload.
            schema_version: Schema version string.

        Returns:
            Path: Absolute resolved path to the written artifact file.
        """
        target_path = self.resolve_path(rel_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        payload_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
        checksum = hashlib.sha256(payload_bytes).hexdigest()

        # Build envelope: preserve top-level keys for dicts for easy access
        if isinstance(payload, dict):
            envelope = dict(payload)
            envelope["payload"] = payload
        else:
            envelope = {"payload": payload}

        envelope["schema_version"] = schema_version
        envelope["session_id"] = self.session_id
        envelope["artifact_type"] = artifact_type
        envelope["created_at"] = datetime.now(timezone.utc).isoformat()
        envelope["checksum_sha256"] = checksum

        # Write to temporary file in the same directory, fsync, and replace atomically
        temp_file = tempfile.NamedTemporaryFile(
            dir=target_path.parent,
            delete=False,
            mode="w",
            encoding="utf-8",
            prefix=f".tmp_{target_path.name}_",
        )
        try:
            json.dump(envelope, temp_file, indent=2)
            temp_file.flush()
            if self.fsync:
                os.fsync(temp_file.fileno())
            temp_file.close()
            os.replace(temp_file.name, target_path)
        except Exception:
            if not temp_file.closed:
                temp_file.close()
            if os.path.exists(temp_file.name):
                os.remove(temp_file.name)
            raise

        try:
            os.utime(self.root_path, None)
        except OSError:
            pass

        key = str(rel_path).replace("\\", "/")
        self._json_cache[key] = envelope

        return target_path

    def read_json(self, rel_path: Union[str, Path]) -> dict[str, Any]:
        """Read and parse a JSON artifact from the workspace.

        Args:
            rel_path: Relative path to JSON artifact.

        Returns:
            dict: Parsed JSON data.
        """
        key = str(rel_path).replace("\\", "/")
        if key in self._json_cache:
            cached = self._json_cache[key]
            return dict(cached) if isinstance(cached, dict) else cached

        target_path = self.resolve_path(rel_path)
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self._json_cache[key] = data
            return data

    def write_text(self, rel_path: Union[str, Path], text: str) -> Path:
        """Atomically write text content (e.g. Markdown reports) to workspace.

        Args:
            rel_path: Relative destination path.
            text: Text content to write.

        Returns:
            Path: Absolute resolved path to the written file.
        """
        target_path = self.resolve_path(rel_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        temp_file = tempfile.NamedTemporaryFile(
            dir=target_path.parent,
            delete=False,
            mode="w",
            encoding="utf-8",
            prefix=f".tmp_{target_path.name}_",
        )
        try:
            temp_file.write(text)
            temp_file.flush()
            if self.fsync:
                os.fsync(temp_file.fileno())
            temp_file.close()
            os.replace(temp_file.name, target_path)
        except Exception:
            if not temp_file.closed:
                temp_file.close()
            if os.path.exists(temp_file.name):
                os.remove(temp_file.name)
            raise

        try:
            os.utime(self.root_path, None)
        except OSError:
            pass

        return target_path

    def read_text(self, rel_path: Union[str, Path]) -> str:
        """Read text content from a file in the workspace.

        Args:
            rel_path: Relative path to text file.

        Returns:
            str: File contents.
        """
        target_path = self.resolve_path(rel_path)
        return target_path.read_text(encoding="utf-8")

    def exists(self, rel_path: Union[str, Path]) -> bool:
        """Check if a relative path exists inside the workspace.

        Args:
            rel_path: Relative path to check.

        Returns:
            bool: True if the path exists on disk, False otherwise.
        """
        key = str(rel_path).replace("\\", "/")
        if key in self._json_cache:
            return True
        return self.resolve_path(rel_path).exists()

    def list_artifacts(self, subdir: str = "") -> List[Path]:
        """List all files in the specified workspace subdirectory.

        Args:
            subdir: Subdirectory name (e.g. 'context', 'turns').

        Returns:
            List[Path]: List of resolved file paths.
        """
        dir_path = self.resolve_path(subdir)
        if not dir_path.exists():
            return []
        return [p for p in dir_path.glob("**/*") if p.is_file()]


class SessionWorkspaceManager:
    """Manages creation, resolution, and isolation of session workspaces."""

    def __init__(
        self,
        root_base: Optional[Union[str, Path]] = None,
        fsync: bool = True,
    ) -> None:
        """Initialize workspace manager with configurable root base.

        Args:
            root_base: Base directory for session workspaces (defaults to settings.WORKSPACE_ROOT_BASE).
            fsync: Whether to enforce os.fsync on managed workspaces.
        """
        if root_base is not None:
            self.root_base = Path(root_base).resolve()
        else:
            self.root_base = Path(get_settings().WORKSPACE_ROOT_BASE).resolve()
        self.fsync = fsync
        self._workspaces: dict[str, SessionWorkspace] = {}

    def get_or_create_workspace(self, session_id: str) -> SessionWorkspace:
        """Get or initialize an isolated workspace for the given session ID.

        Normalizes session identifier into safe directory name:
        `session_{clean_id}-{sha256(session_id)[:12]}`.

        Args:
            session_id: Raw session identifier.

        Returns:
            SessionWorkspace: Initialized session workspace object.
        """
        if session_id in self._workspaces:
            cached_ws = self._workspaces[session_id]
            if cached_ws.root_path.exists():
                return cached_ws
            del self._workspaces[session_id]

        normalized_id = normalize_session_id(session_id)
        workspace_path = self.root_base / normalized_id
        workspace = SessionWorkspace(
            session_id=session_id,
            root_path=workspace_path,
            normalized_session_id=normalized_id,
            fsync=self.fsync,
        )
        workspace.initialize()
        self._workspaces[session_id] = workspace
        return workspace

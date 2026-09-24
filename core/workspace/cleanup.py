"""Workspace retention policy and idempotent cleanup manager."""

import os
import shutil
import time
from pathlib import Path
from typing import List, Optional, Union

from config.settings import get_settings
from core.workspace.security import WorkspacePathError, validate_path_safety
from core.workspace.session import normalize_session_id


class WorkspaceCleanupManager:
    """Manages retention, disk space audits, and idempotent deletion of session workspaces.

    Attributes:
        root_base: Canonical Path to the root directory containing session workspaces.
        retention_hours: Maximum inactivity duration before a workspace is considered expired.
    """

    def __init__(
        self,
        root_base: Optional[Union[str, Path]] = None,
        retention_hours: Optional[int] = None,
    ) -> None:
        """Initialize workspace cleanup manager.

        Args:
            root_base: Base directory for workspaces (defaults to settings.WORKSPACE_ROOT_BASE).
            retention_hours: Session retention limit in hours (defaults to settings.SESSION_RETENTION_HOURS).
        """
        settings = get_settings()
        self.root_base = (
            Path(root_base).resolve()
            if root_base is not None
            else Path(settings.WORKSPACE_ROOT_BASE).resolve()
        )
        self.retention_hours = (
            retention_hours if retention_hours is not None else settings.SESSION_RETENTION_HOURS
        )

    def _resolve_workspace_dir(self, session_id_or_dir: str) -> Optional[Path]:
        """Resolve a workspace directory path with strict traversal and ancestor checks.

        Args:
            session_id_or_dir: Directory name or raw session identifier.

        Returns:
            Optional[Path]: Absolute resolved directory if found and valid, else None.

        Raises:
            WorkspacePathError: If path escapes root_base or targets root_base itself.
        """
        if not session_id_or_dir or not str(session_id_or_dir).strip():
            return None

        # Verify path safety to prevent directory traversal
        target_path = validate_path_safety(self.root_base, session_id_or_dir)

        # Enforce that target is strictly a descendant of root_base (not root_base itself)
        if target_path == self.root_base:
            raise WorkspacePathError("Target directory cannot be the workspace root base itself")

        if target_path.exists() and target_path.is_dir():
            return target_path

        # If not found directly, check by normalized session ID
        normalized_name = normalize_session_id(session_id_or_dir)
        normalized_path = validate_path_safety(self.root_base, normalized_name)
        if normalized_path.exists() and normalized_path.is_dir():
            return normalized_path

        return None

    def delete_workspace(self, session_id_or_dir: str) -> bool:
        """Idempotently delete a specific session workspace directory.

        Args:
            session_id_or_dir: Directory name or session identifier.

        Returns:
            bool: True if workspace existed and was deleted, False if already absent.
        """
        ws_dir = self._resolve_workspace_dir(session_id_or_dir)
        if ws_dir is None or not ws_dir.exists():
            return False

        try:
            shutil.rmtree(ws_dir, ignore_errors=False)
            return True
        except FileNotFoundError:
            return False

    def cleanup_expired_workspaces(self, max_age_hours: Optional[int] = None) -> List[str]:
        """Scan workspace root and remove workspaces older than retention threshold.

        Considers latest modification time across workspace root and all contained files.

        Args:
            max_age_hours: Override retention period in hours.

        Returns:
            List[str]: List of deleted workspace directory names.
        """
        if not self.root_base.exists():
            return []

        effective_hours = max_age_hours if max_age_hours is not None else self.retention_hours
        cutoff_time = time.time() - (effective_hours * 3600)
        deleted: List[str] = []

        for entry in self.root_base.iterdir():
            if not entry.is_dir():
                continue

            try:
                # Find latest modification time across directory and all contained files
                last_active = entry.stat().st_mtime
                for root, _, files in os.walk(entry):
                    for f in files:
                        try:
                            fp = os.path.join(root, f)
                            fmtime = os.path.getmtime(fp)
                            if fmtime > last_active:
                                last_active = fmtime
                        except OSError:
                            continue

                if last_active < cutoff_time:
                    shutil.rmtree(entry, ignore_errors=True)
                    deleted.append(entry.name)
            except (OSError, FileNotFoundError):
                continue

        return deleted

    def get_workspace_size_bytes(self, session_id_or_dir: str) -> int:
        """Calculate total disk usage of a workspace in bytes.

        Args:
            session_id_or_dir: Directory name or session identifier.

        Returns:
            int: Total size in bytes, or 0 if workspace does not exist.
        """
        ws_dir = self._resolve_workspace_dir(session_id_or_dir)
        if ws_dir is None or not ws_dir.exists():
            return 0

        total_bytes = 0
        for root, _, files in os.walk(ws_dir):
            for f in files:
                fp = os.path.join(root, f)
                try:
                    total_bytes += os.path.getsize(fp)
                except OSError:
                    continue
        return total_bytes

    def list_workspaces(self) -> List[str]:
        """List all active workspace directory names.

        Returns:
            List[str]: Names of workspace directories found under root_base.
        """
        if not self.root_base.exists():
            return []
        return [entry.name for entry in self.root_base.iterdir() if entry.is_dir()]

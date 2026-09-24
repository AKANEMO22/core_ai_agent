"""Workspace retention policy and idempotent cleanup manager."""

import os
import shutil
import time
from pathlib import Path
from typing import List, Optional, Union

from config.settings import get_settings


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
        """Resolve a workspace directory path by exact name, normalized name, or prefix."""
        target = self.root_base / session_id_or_dir
        if target.exists() and target.is_dir():
            return target

        # Search for directory starting with session_id_or_dir
        if self.root_base.exists():
            for entry in self.root_base.iterdir():
                if entry.is_dir() and (entry.name == session_id_or_dir or entry.name.startswith(session_id_or_dir)):
                    return entry
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
                stat = entry.stat()
                last_active = stat.st_mtime
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

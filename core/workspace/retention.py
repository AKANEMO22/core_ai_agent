"""Retention sweeper: raw-upload deletion after reporting and expiry of old workspaces.

Ứng dụng gọi ``RetentionService.sweep()`` định kỳ: trước hết xóa riêng raw inputs của
phiên đã phát hành report đủ lâu, sau đó chuyển việc quét workspace cũ sang thread để
không block event loop. Redis state có TTL riêng và không bị module này xóa trực tiếp.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict

from core.observability import session_ref
from core.runtime.agent import sanitize_secrets
from core.session.state import LifecycleStatus, SessionStateStore
from core.storage.archive import ArtifactArchive
from core.workspace.cleanup import WorkspaceCleanupManager
from core.workspace.session import SessionWorkspace, SessionWorkspaceManager, normalize_session_id

logger = logging.getLogger(__name__)


class RetentionService:
    """Applies the retention policy of spec Phase 1.

    - Raw uploads (`inputs/`) are deleted `raw_upload_retention_hours` after the report was
      synthesized; derived artifacts stay.
    - Whole workspaces inactive for `artifact_retention_days` are deleted. Redis keys expire
      through their own TTLs.
    """

    def __init__(
        self,
        state_store: SessionStateStore,
        workspace_mgr: SessionWorkspaceManager,
        cleanup_mgr: WorkspaceCleanupManager,
        raw_upload_retention_hours: int = 24,
        artifact_retention_days: int = 30,
        artifact_archive: ArtifactArchive | None = None,
    ) -> None:
        self.state_store = state_store
        self.workspace_mgr = workspace_mgr
        self.cleanup_mgr = cleanup_mgr
        self.raw_upload_retention_hours = raw_upload_retention_hours
        self.artifact_retention_days = artifact_retention_days
        self.artifact_archive = artifact_archive

    def _completed_local_sessions(self) -> Dict[str, str]:
        """Find finished report workspaces even after their Redis state TTL expires."""
        found: Dict[str, str] = {}
        root = self.cleanup_mgr.root_base
        for directory_name in self.cleanup_mgr.list_workspaces():
            directory = root / directory_name
            if directory.is_symlink() or not (directory / "reports" / "final_assessment.json").is_file():
                continue
            try:
                form = json.loads((directory / "inputs" / "form_input.json").read_text(encoding="utf-8"))
                session_id = form.get("session_id") if isinstance(form, dict) else None
                if session_id and normalize_session_id(session_id) == directory_name:
                    found[session_id] = str(directory)
            except (OSError, ValueError, TypeError):
                continue
        return found

    async def sweep(self) -> Dict[str, Any]:
        """Run one retention pass. Returns counts for logging and tests."""
        raw_deleted = 0
        cutoff = time.time() - self.raw_upload_retention_hours * 3600
        protected_sessions: set[str] = set()
        if self.artifact_archive is not None:
            report_ready = set(await self.state_store.alist_by_status(LifecycleStatus.REPORT_READY))
            local_completed = await asyncio.to_thread(self._completed_local_sessions)
            for session_id in report_ready | set(local_completed):
                workspace_path = Path(
                    local_completed.get(session_id)
                    or self.workspace_mgr.root_base / normalize_session_id(session_id)
                )
                if not workspace_path.is_dir() or workspace_path.is_symlink():
                    if not self.artifact_archive.is_archived(session_id, owner_id=session_id):
                        protected_sessions.add(session_id)
                    continue
                workspace = SessionWorkspace(
                    session_id,
                    workspace_path,
                    normalize_session_id(session_id),
                    fsync=False,
                )
                try:
                    # Hash every completed workspace; this also captures later review artifacts.
                    await asyncio.to_thread(
                        self.artifact_archive.archive_session,
                        session_id,
                        workspace.root_path,
                        owner_id=session_id,
                    )
                except Exception as e:
                    logger.warning(
                        "Artifact archive pending for session %s: %s",
                        session_ref(session_id),
                        sanitize_secrets(str(e)),
                    )
                archived = self.artifact_archive.is_archived(session_id, owner_id=session_id)
                if not archived:
                    protected_sessions.add(session_id)
                    continue
                state = await self.state_store.aget(session_id)
                synthesized_at = (state.metadata if state else {}).get("report_synthesized_at")
                report_path = workspace.reports_dir / "final_assessment.json"
                if not synthesized_at and report_path.is_file():
                    synthesized_at = report_path.stat().st_mtime
                if not synthesized_at or synthesized_at > cutoff:
                    continue
                for path in list(workspace.inputs_dir.glob("*")):
                    try:
                        if path.is_file():
                            await asyncio.to_thread(path.unlink)
                            raw_deleted += 1
                    except OSError as e:
                        logger.warning(
                            "Raw upload deletion failed for session %s: %s",
                            session_ref(session_id),
                            sanitize_secrets(str(e)),
                        )
        else:
            # Keep the existing local-only policy when object archiving is disabled.
            for session_id in await self.state_store.alist_by_status(LifecycleStatus.REPORT_READY):
                state = await self.state_store.aget(session_id)
                synthesized_at = (state.metadata if state else {}).get("report_synthesized_at")
                if not synthesized_at or synthesized_at > cutoff:
                    continue
                ws = self.workspace_mgr.get_or_create_workspace(session_id)
                for path in list(ws.inputs_dir.glob("*")):
                    try:
                        if path.is_file():
                            await asyncio.to_thread(path.unlink)
                            raw_deleted += 1
                    except OSError as e:
                        logger.warning(
                            "Raw upload deletion failed for session %s: %s",
                            session_ref(session_id),
                            sanitize_secrets(str(e)),
                        )

        # Bước hai mới áp thời hạn artifact toàn workspace; I/O đệ quy chạy ở thread để không chặn event loop.
        expired = await asyncio.to_thread(
            self.cleanup_mgr.cleanup_expired_workspaces,
            self.artifact_retention_days * 24,
            list(protected_sessions),
        )
        if expired:
            self.workspace_mgr.clear_cache()
        return {"raw_uploads_deleted": raw_deleted, "workspaces_expired": len(expired)}

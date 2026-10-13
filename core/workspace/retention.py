"""Retention sweeper: raw-upload deletion after reporting and expiry of old workspaces."""

import asyncio
import logging
import time
from typing import Any, Dict

from core.observability import session_ref
from core.runtime.agent import sanitize_secrets
from core.session.state import LifecycleStatus, SessionStateStore
from core.workspace.cleanup import WorkspaceCleanupManager
from core.workspace.session import SessionWorkspaceManager

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
    ) -> None:
        self.state_store = state_store
        self.workspace_mgr = workspace_mgr
        self.cleanup_mgr = cleanup_mgr
        self.raw_upload_retention_hours = raw_upload_retention_hours
        self.artifact_retention_days = artifact_retention_days

    async def sweep(self) -> Dict[str, Any]:
        """Run one retention pass. Returns counts for logging and tests."""
        raw_deleted = 0
        cutoff = time.time() - self.raw_upload_retention_hours * 3600
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

        expired = await asyncio.to_thread(
            self.cleanup_mgr.cleanup_expired_workspaces, self.artifact_retention_days * 24
        )
        if expired:
            self.workspace_mgr.clear_cache()
        return {"raw_uploads_deleted": raw_deleted, "workspaces_expired": len(expired)}

"""Lớp lưu trữ và quản lý file riêng cho từng phiên.

``SessionWorkspaceManager`` tạo sandbox; ``SessionWorkspace`` đọc/ghi artifact bên trong
sandbox đó; ``security`` kiểm tra đường dẫn/upload; ``spill`` chuyển output lớn ra đĩa;
``cleanup`` và ``retention`` xóa dữ liệu theo chính sách vòng đời. Các agent không nên
tự ghép đường dẫn workspace hoặc tự ghi artifact ngoài các cổng này.
"""

from core.workspace.security import (
    WorkspaceSecurityError,
    WorkspacePathError,
    WorkspaceUploadError,
    validate_path_safety,
    validate_upload_content,
)
from core.workspace.session import SessionWorkspace, SessionWorkspaceManager, normalize_session_id
from core.workspace.spill import WorkspaceSpillManager
from core.workspace.cleanup import WorkspaceCleanupManager

__all__ = [
    "WorkspaceSecurityError",
    "WorkspacePathError",
    "WorkspaceUploadError",
    "validate_path_safety",
    "validate_upload_content",
    "SessionWorkspace",
    "SessionWorkspaceManager",
    "normalize_session_id",
    "WorkspaceSpillManager",
    "WorkspaceCleanupManager",
]

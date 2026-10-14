"""Security protections for isolated workspaces, path resolution, and file uploads.

Workspace gọi ``validate_path_safety`` trước khi đọc/ghi/xóa file; API gọi
``validate_upload_content`` trước khi chấp nhận CV/JD. Kiểm tra đường dẫn xử lý traversal
và symlink, còn upload đi tuần tự qua giới hạn kích thước, chữ ký nhị phân, cấu trúc JSON
và chính sách MIME.
"""

import json
import os
from pathlib import Path
from typing import Any, Optional, Set, Union

from config.settings import get_settings


class WorkspaceSecurityError(Exception):
    """Base exception for workspace security violations."""


class WorkspacePathError(WorkspaceSecurityError):
    """Raised when an unsafe path resolution or directory traversal is detected."""


class WorkspaceUploadError(WorkspaceSecurityError, ValueError):
    """Raised when uploaded content fails security, format, or size validation."""


# Known executable binary magic byte headers
EXECUTABLE_MAGIC_HEADERS: tuple[bytes, ...] = (
    b"MZ",  # Windows PE executable / DLL
    b"\x7fELF",  # Linux ELF executable / library
    b"\xca\xfe\xba\xbe",  # Mach-O universal binary or Java class
    b"\xce\xfa\xed\xfe",  # Mach-O 32-bit
    b"\xcf\xfa\xed\xfe",  # Mach-O 64-bit
    b"\xfe\xed\xfa\xce",  # Mach-O 32-bit reverse endian
    b"\xfe\xed\xfa\xcf",  # Mach-O 64-bit reverse endian
)


def validate_path_safety(root_path: Union[str, Path], target_path: Union[str, Path]) -> Path:
    """Validate that a target path strictly resolves within the specified workspace root.

    Prevents directory traversal, relative escape (e.g. '../'), absolute path escape
    (e.g. '/etc/passwd' or 'C:\\Windows'), and symlink escapes.

    Args:
        root_path: Workspace root directory.
        target_path: Relative or absolute path within the workspace.

    Returns:
        Path: The canonical resolved absolute Path inside root_path.

    Raises:
        WorkspacePathError: If the target path resolves outside root_path or is malformed.
    """
    def _inside(path: Path, root: Path) -> bool:
        path_s = os.path.normcase(str(path))
        root_s = os.path.normcase(str(root))
        return path_s == root_s or path_s.startswith(root_s.rstrip("\\/") + os.sep)

    try:
        root_resolved = Path(root_path).resolve()
        target = Path(target_path)

        # Kiểm tra đường dẫn thuần trước, tránh cho ../ hoặc đường dẫn tuyệt đối chạm ra ngoài workspace.
        # 1. Lexical check: rejects '../' escapes and absolute paths without touching the filesystem.
        lexical = Path(os.path.normpath(root_resolved / target))
        if not _inside(lexical, root_resolved):
            raise ValueError("lexical escape")

        # Kiểm tra thêm symlink vì đường dẫn hợp lệ về mặt chữ vẫn có thể trỏ ra ngoài qua liên kết.
        # 2. Symlink check. The parent directory is resolved (directories are never atomically
        #    replaced); the leaf is resolved only if it is a symlink. Resolving a regular file
        #    while another task atomically replaces it is unreliable on Windows.
        resolved = lexical.parent.resolve() / lexical.name if lexical != root_resolved else root_resolved
        if resolved.is_symlink():
            resolved = resolved.resolve()
        if not _inside(resolved, root_resolved):
            raise ValueError("symlink escape")
        return resolved
    except (ValueError, TypeError, RuntimeError, OSError) as e:
        # Never echo absolute host paths (they embed session identifiers) into errors or logs.
        raise WorkspacePathError(f"Path traversal detected: '{target_path}' resolves outside the workspace root") from e


def _calculate_json_depth(obj: Any, current_depth: int = 1) -> int:
    """Compute the maximum nesting depth of a parsed JSON object.

    Args:
        obj: The object to inspect (dict, list, or primitive).
        current_depth: Current recursion depth.

    Returns:
        int: Maximum nesting depth.
    """
    if isinstance(obj, dict):
        if not obj:
            return current_depth
        return max(_calculate_json_depth(v, current_depth + 1) for v in obj.values())
    elif isinstance(obj, list):
        if not obj:
            return current_depth
        return max(_calculate_json_depth(item, current_depth + 1) for item in obj)
    return current_depth


def validate_upload_content(
    content: Union[str, bytes],
    filename: Optional[str] = None,
    content_type: Optional[str] = None,
    max_size_bytes: Optional[int] = None,
    max_json_depth: Optional[int] = None,
    allowed_mime_types: Optional[Set[str]] = None,
) -> bytes:
    """Validate uploaded content against size, executable magic bytes, and JSON recursion depth.

    Args:
        content: Uploaded payload as bytes or string.
        filename: Optional name of the uploaded file for extension sniffing.
        content_type: Optional declared MIME type.
        max_size_bytes: Maximum allowed size in bytes (defaults to settings.MAX_UPLOAD_SIZE_BYTES).
        max_json_depth: Maximum JSON nesting depth (defaults to settings.MAX_JSON_DEPTH).
        allowed_mime_types: Optional set of allowed MIME types.

    Returns:
        bytes: Validated raw bytes.

    Raises:
        WorkspaceUploadError: If validation fails.
    """
    # Hàm chạy trước khi upload được lưu: giới hạn byte/depth và loại executable trước khi kiểm MIME.
    settings = get_settings()
    effective_max_size = max_size_bytes if max_size_bytes is not None else settings.MAX_UPLOAD_SIZE_BYTES
    effective_max_depth = max_json_depth if max_json_depth is not None else settings.MAX_JSON_DEPTH

    # Convert to bytes if string
    if isinstance(content, str):
        raw_bytes = content.encode("utf-8")
    elif isinstance(content, (bytes, bytearray)):
        raw_bytes = bytes(content)
    else:
        raise WorkspaceUploadError(f"Unsupported content type: {type(content).__name__}")

    # Check size constraint
    if len(raw_bytes) > effective_max_size:
        raise WorkspaceUploadError(
            f"Upload size {len(raw_bytes)} bytes exceeds maximum allowed {effective_max_size} bytes"
        )

    # Check executable magic bytes
    for header in EXECUTABLE_MAGIC_HEADERS:
        if raw_bytes.startswith(header):
            raise WorkspaceUploadError(
                f"Malicious or executable binary signature detected: {filename or 'unknown'}"
            )

    # Sniff and inspect JSON payloads
    is_json = False
    if filename and filename.lower().endswith(".json"):
        is_json = True
    elif content_type and "json" in content_type.lower():
        is_json = True
    elif raw_bytes.lstrip().startswith((b"{", b"[")):
        is_json = True

    if is_json:
        try:
            # Chỉ parse JSON khi nội dung có dấu hiệu là JSON; độ sâu chặn payload lồng quá mức.
            decoded = raw_bytes.decode("utf-8")
            parsed = json.loads(decoded)
            depth = _calculate_json_depth(parsed)
            if depth > effective_max_depth:
                raise WorkspaceUploadError(
                    f"JSON nesting depth {depth} exceeds maximum allowed limit {effective_max_depth}"
                )
        except UnicodeDecodeError as e:
            raise WorkspaceUploadError(f"JSON upload contains invalid UTF-8 encoding: {e}") from e
        except json.JSONDecodeError as e:
            if (filename and filename.lower().endswith(".json")) or (content_type and "json" in content_type.lower()):
                raise WorkspaceUploadError(f"Invalid JSON syntax in upload: {e}") from e

    # MIME khai báo từ client không đủ tin cậy; so với chữ ký file và cách giải mã thực tế.
    detected_mime: Optional[str] = None
    if raw_bytes.startswith(b"%PDF-"):
        detected_mime = "application/pdf"
    elif raw_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        detected_mime = "image/png"
    elif raw_bytes.startswith(b"\xff\xd8\xff"):
        detected_mime = "image/jpeg"
    elif is_json:
        detected_mime = "application/json"
    else:
        try:
            raw_bytes.decode("utf-8")
            if filename and filename.lower().endswith(".md"):
                detected_mime = "text/markdown"
            else:
                detected_mime = "text/plain"
        except UnicodeDecodeError:
            detected_mime = "application/octet-stream"

    # Enforce allowed_mime_types if specified
    if allowed_mime_types is not None:
        effective_mime = (
            content_type.lower().split(";")[0].strip()
            if content_type
            else detected_mime
        )
        if not effective_mime or effective_mime not in allowed_mime_types:
            raise WorkspaceUploadError(
                f"MIME type '{effective_mime}' is not permitted in allowed MIME types: {sorted(allowed_mime_types)}"
            )

    return raw_bytes

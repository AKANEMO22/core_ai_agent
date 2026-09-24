"""Unit tests for configuration, isolated session workspace, path security, spillover, and cleanup."""

import json
import os
import time
from pathlib import Path
import pytest

from config.settings import Settings, get_settings
from core.workspace.session import SessionWorkspace, SessionWorkspaceManager
from core.workspace.security import (
    WorkspacePathError,
    WorkspaceUploadError,
    validate_path_safety,
    validate_upload_content,
)
from core.workspace.spill import WorkspaceSpillManager
from core.workspace.cleanup import WorkspaceCleanupManager


def test_normalized_session_id():
    """Verify session ID normalization produces isolated directory name with 12-char SHA-256 suffix."""
    manager = SessionWorkspaceManager(root_base="./test_data/workspaces")
    ws = manager.get_or_create_workspace("user/session@123")
    assert "user_session_123" in ws.normalized_session_id
    assert len(ws.normalized_session_id.split("-")[-1]) == 12


def test_path_traversal_detection(tmp_path: Path):
    """Verify relative and absolute traversal attempts outside workspace root are rejected."""
    ws = SessionWorkspace(session_id="sess_1", root_path=tmp_path / "sess_1")
    ws.initialize()
    with pytest.raises(WorkspacePathError):
        ws.resolve_path("../outside.txt")
    with pytest.raises(WorkspacePathError):
        ws.resolve_path("/etc/passwd")


def test_atomic_artifact_write(tmp_path: Path):
    """Verify atomic artifact writes produce valid envelopes with checksums and data integrity."""
    ws = SessionWorkspace(session_id="sess_1", root_path=tmp_path / "sess_1")
    ws.initialize()
    data = {"status": "ok", "value": 42}
    rel_path = "context/test_artifact.json"
    written_path = ws.write_artifact(rel_path=rel_path, artifact_type="test", payload=data)
    assert written_path.exists()
    read_data = ws.read_json(rel_path)
    assert read_data["value"] == 42
    assert "checksum_sha256" in read_data
    assert read_data["schema_version"] == "1.0"
    assert read_data["session_id"] == "sess_1"
    assert read_data["artifact_type"] == "test"
    assert "created_at" in read_data


def test_spillover_exceeding_threshold(tmp_path: Path):
    """Verify payloads exceeding max bytes are spilled to disk with a truncated preview."""
    ws = SessionWorkspace(session_id="sess_1", root_path=tmp_path / "sess_1")
    ws.initialize()
    spill_mgr = WorkspaceSpillManager(ws, max_bytes=100)
    large_text = "A" * 200
    res = spill_mgr.maybe_spill(tool_name="test_tool", payload=large_text)
    assert res["is_spilled"] is True
    assert "spill_path" in res
    assert len(res["preview"]) <= 100
    assert Path(res["spill_path"]).exists()


def test_spillover_within_threshold(tmp_path: Path):
    """Verify payloads within threshold are not spilled."""
    ws = SessionWorkspace(session_id="sess_1", root_path=tmp_path / "sess_1")
    ws.initialize()
    spill_mgr = WorkspaceSpillManager(ws, max_bytes=100)
    small_text = "Hello world"
    res = spill_mgr.maybe_spill(tool_name="test_tool", payload=small_text)
    assert res["is_spilled"] is False
    assert res["payload"] == small_text
    assert "spill_path" not in res


def test_workspace_directory_layout(tmp_path: Path):
    """Verify initialization creates all required subdirectories."""
    ws = SessionWorkspace(session_id="sess_layout", root_path=tmp_path / "sess_layout")
    ws.initialize()
    expected_subdirs = ["inputs", "context", "turns", "evaluations", "reports", "spill"]
    for subdir in expected_subdirs:
        sub_path = ws.root_path / subdir
        assert sub_path.exists()
        assert sub_path.is_dir()


def test_validate_upload_content_valid_pdf_and_json():
    """Verify valid content passes validation."""
    valid_json = b'{"name": "Candidate", "experience": 5}'
    res_json = validate_upload_content(valid_json, filename="resume.json")
    assert res_json == valid_json

    pdf_bytes = b"%PDF-1.4 mock content for pdf"
    res_pdf = validate_upload_content(pdf_bytes, filename="cv.pdf")
    assert res_pdf == pdf_bytes


def test_validate_upload_content_size_limit():
    """Verify oversized uploads are rejected."""
    oversized = b"x" * 200
    with pytest.raises(WorkspaceUploadError):
        validate_upload_content(oversized, filename="data.txt", max_size_bytes=100)


def test_validate_upload_content_executable_magic_bytes_rejected():
    """Verify executable binaries (Windows PE MZ, ELF) are rejected."""
    pe_bytes = b"MZ\x90\x00\x03\x00\x00\x00"
    with pytest.raises(WorkspaceUploadError):
        validate_upload_content(pe_bytes, filename="doc.txt")

    elf_bytes = b"\x7fELF\x02\x01\x01\x00"
    with pytest.raises(WorkspaceUploadError):
        validate_upload_content(elf_bytes, filename="doc.txt")


def test_validate_upload_content_deep_json_rejected():
    """Verify nested JSON exceeding max depth is rejected."""
    nested = {"level": 1}
    curr = nested
    for i in range(2, 15):
        curr["child"] = {"level": i}
        curr = curr["child"]
    deep_json_bytes = json.dumps(nested).encode("utf-8")

    with pytest.raises(WorkspaceUploadError):
        validate_upload_content(deep_json_bytes, filename="nested.json", max_json_depth=5)


def test_cleanup_manager_retention_and_idempotence(tmp_path: Path):
    """Verify cleanup manager purges expired workspaces and is idempotent."""
    root_base = tmp_path / "workspaces"
    root_base.mkdir()
    cleanup_mgr = WorkspaceCleanupManager(root_base=root_base, retention_hours=1)

    # Create two workspaces
    ws1 = root_base / "session_old-123456789012"
    ws1.mkdir()
    (ws1 / "context").mkdir()
    ws2 = root_base / "session_new-123456789012"
    ws2.mkdir()
    (ws2 / "context").mkdir()

    # Modify mtime of ws1 to 2 hours ago
    two_hours_ago = time.time() - (2 * 3600)
    os.utime(ws1, (two_hours_ago, two_hours_ago))

    deleted = cleanup_mgr.cleanup_expired_workspaces(max_age_hours=1)
    assert ws1.name in deleted
    assert not ws1.exists()
    assert ws2.exists()

    # Run cleanup again to verify idempotence
    deleted_again = cleanup_mgr.cleanup_expired_workspaces(max_age_hours=1)
    assert len(deleted_again) == 0

    # Test explicit delete
    assert cleanup_mgr.delete_workspace(ws2.name) is True
    assert not ws2.exists()
    # Idempotent delete on non-existent
    assert cleanup_mgr.delete_workspace(ws2.name) is False


def test_settings_singleton_and_defaults():
    """Verify settings singleton loads expected defaults and behaves consistently."""
    settings = get_settings()
    assert isinstance(settings, Settings)
    assert settings.MAX_TOOL_RESULT_BYTES == 32768
    assert settings.WORKSPACE_ROOT_BASE == Path("./data/workspaces")
    assert settings.REDIS_URL.startswith("redis://")
    assert settings.SESSION_RETENTION_HOURS == 24
    assert get_settings() is settings


def test_path_traversal_subtle_cases(tmp_path: Path):
    """Verify tricky path traversal vectors like inner relative breakout are caught."""
    ws = SessionWorkspace(session_id="sess_subtle", root_path=tmp_path / "sess_subtle")
    ws.initialize()

    # Breakout from nested path
    with pytest.raises(WorkspacePathError):
        ws.resolve_path("context/../../outside.txt")

    # Safe relative navigation within workspace
    safe_path = ws.resolve_path("context/../inputs/file.txt")
    assert safe_path == (ws.root_path / "inputs" / "file.txt").resolve()


def test_write_and_read_text(tmp_path: Path):
    """Verify atomic text writing and reading for non-JSON content (e.g. Markdown reports)."""
    ws = SessionWorkspace(session_id="sess_text", root_path=tmp_path / "sess_text")
    ws.initialize()
    report_content = "# Candidate Interview Report\n\nScore: 9/10"
    rel_path = "reports/final_report.md"
    written_path = ws.write_text(rel_path, report_content)
    assert written_path.exists()
    assert ws.read_text(rel_path) == report_content


def test_read_spilled_payload(tmp_path: Path):
    """Verify spilled payloads can be retrieved and decoded accurately."""
    ws = SessionWorkspace(session_id="sess_spill_read", root_path=tmp_path / "sess_spill_read")
    ws.initialize()
    spill_mgr = WorkspaceSpillManager(ws, max_bytes=50)
    data = {"long_key": "val" * 50}
    res = spill_mgr.maybe_spill(tool_name="json_tool", payload=data)
    assert res["is_spilled"] is True
    retrieved = spill_mgr.read_spilled(res["spill_path"])
    assert retrieved == data


def test_cleanup_manager_size_and_listing(tmp_path: Path):
    """Verify workspace size calculation and listing."""
    root_base = tmp_path / "workspaces"
    root_base.mkdir()
    ws_dir = root_base / "session_size_test-123456789012"
    ws = SessionWorkspace(session_id="test", root_path=ws_dir)
    ws.initialize()
    ws.write_text("inputs/resume.txt", "1234567890")

    cleanup_mgr = WorkspaceCleanupManager(root_base=root_base)
    workspaces = cleanup_mgr.list_workspaces()
    assert ws_dir.name in workspaces

    size = cleanup_mgr.get_workspace_size_bytes(ws_dir.name)
    assert size >= 10


def test_cleanup_manager_path_traversal_defense(tmp_path: Path):
    """Verify cleanup manager rejects path traversal attempts and root base deletion."""
    root_base = tmp_path / "workspaces"
    root_base.mkdir()
    cleanup_mgr = WorkspaceCleanupManager(root_base=root_base)

    with pytest.raises(WorkspacePathError):
        cleanup_mgr.delete_workspace("..")

    with pytest.raises(WorkspacePathError):
        cleanup_mgr.delete_workspace("../../outside")

    with pytest.raises(WorkspacePathError):
        cleanup_mgr.delete_workspace(".")


def test_cleanup_manager_no_prefix_hazard_and_normalized_delete(tmp_path: Path):
    """Verify cleanup manager disallows empty/generic prefixes and resolves raw session ID."""
    root_base = tmp_path / "workspaces"
    root_base.mkdir()
    ws_dir = root_base / "session_cand_456-123456789012"
    ws_dir.mkdir()

    cleanup_mgr = WorkspaceCleanupManager(root_base=root_base)

    # Empty string should not match or delete
    assert cleanup_mgr.delete_workspace("") is False
    assert ws_dir.exists()

    # Short prefix should not match or delete
    assert cleanup_mgr.delete_workspace("session_") is False
    assert ws_dir.exists()

    # Deleting by raw session ID works via normalization
    from core.workspace.session import SessionWorkspaceManager
    mgr = SessionWorkspaceManager(root_base=root_base)
    raw_session_id = "cand@456"
    ws = mgr.get_or_create_workspace(raw_session_id)
    assert ws.root_path.exists()
    assert cleanup_mgr.delete_workspace(raw_session_id) is True
    assert not ws.root_path.exists()


def test_spillover_binary_payload(tmp_path: Path):
    """Verify raw binary bytes exceeding threshold are base64-serialized without crashing."""
    ws = SessionWorkspace(session_id="sess_bin", root_path=tmp_path / "sess_bin")
    ws.initialize()
    spill_mgr = WorkspaceSpillManager(ws, max_bytes=50)

    binary_data = b"\x00\x01\x02\xff\xfe\xfd" * 20  # Non-UTF8 binary data > 50 bytes
    res = spill_mgr.maybe_spill(tool_name="audio_tool", payload=binary_data)
    assert res["is_spilled"] is True
    assert Path(res["spill_path"]).exists()

    # Read back and verify exact binary recovery
    recovered = spill_mgr.read_spilled(res["spill_path"])
    assert recovered == binary_data


def test_validate_upload_content_allowed_mime_types():
    """Verify allowed_mime_types enforces MIME restrictions and detects violations."""
    pdf_bytes = b"%PDF-1.5 test document"
    allowed = {"application/pdf"}

    # Permitted PDF
    assert validate_upload_content(pdf_bytes, filename="cv.pdf", allowed_mime_types=allowed) == pdf_bytes

    # Disallowed plain text
    with pytest.raises(WorkspaceUploadError):
        validate_upload_content(b"Hello plain text", filename="cv.txt", allowed_mime_types=allowed)

    # Disallowed explicit content_type
    with pytest.raises(WorkspaceUploadError):
        validate_upload_content(
            pdf_bytes,
            filename="cv.pdf",
            content_type="text/plain",
            allowed_mime_types=allowed,
        )


def test_workspace_root_mtime_touch_on_write(tmp_path: Path):
    """Verify write_artifact and write_text touch workspace root directory mtime."""
    ws = SessionWorkspace(session_id="sess_touch", root_path=tmp_path / "sess_touch")
    ws.initialize()

    # Backdate root_path mtime
    past_time = time.time() - 5000
    os.utime(ws.root_path, (past_time, past_time))
    assert abs(ws.root_path.stat().st_mtime - past_time) < 2

    # write_artifact touches root_path
    ws.write_artifact("turns/turn_1.json", artifact_type="turn", payload={"step": 1})
    assert time.time() - ws.root_path.stat().st_mtime < 5

    # Backdate again and test write_text
    os.utime(ws.root_path, (past_time, past_time))
    ws.write_text("turns/transcript.txt", "Turn 1 transcript")
    assert time.time() - ws.root_path.stat().st_mtime < 5



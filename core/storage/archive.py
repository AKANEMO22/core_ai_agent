"""Durable, checksummed snapshots of completed session workspaces."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sqlite3
import stat
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from core.storage.artifacts import ObjectStore

logger = logging.getLogger(__name__)


class ArchiveError(RuntimeError):
    """An archive operation could not be completed or verified."""


class ArchiveNotFound(ArchiveError):
    """No restorable archive exists for the requested session."""


@dataclass(frozen=True)
class ArchiveRecord:
    session_id: str
    owner_id: str
    revision: int
    object_key: str
    sha256: str
    size_bytes: int
    status: str


class ArtifactArchive:
    """Coordinate local workspace snapshots, durable metadata and S3 objects."""

    def __init__(
        self,
        *,
        object_store: ObjectStore,
        bucket: str,
        catalog_path: str | Path,
        max_restore_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        self.object_store = object_store
        self.bucket = bucket
        self.catalog_path = Path(catalog_path).resolve()
        self.max_restore_bytes = max_restore_bytes
        self._locks_guard = threading.Lock()
        self._session_locks: dict[str, threading.Lock] = {}
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_catalog()

    def initialize(self) -> None:
        """Create the private bucket and enable object versioning."""
        self.object_store.ensure_bucket(self.bucket, versioning=True)

    def check(self) -> None:
        self.object_store.check_bucket(self.bucket)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.catalog_path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def _initialize_catalog(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS archive_sessions (
                    session_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    current_revision INTEGER,
                    updated_at REAL NOT NULL,
                    error TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS archive_revisions (
                    session_id TEXT NOT NULL REFERENCES archive_sessions(session_id),
                    revision INTEGER NOT NULL,
                    object_key TEXT NOT NULL UNIQUE,
                    sha256 TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    version_id TEXT,
                    status TEXT NOT NULL,
                    archived_at REAL,
                    error TEXT,
                    PRIMARY KEY (session_id, revision)
                );
                """
            )
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(archive_sessions)")}
            if "metadata_json" not in columns:
                connection.execute(
                    "ALTER TABLE archive_sessions ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'"
                )
            revision_columns = {row["name"] for row in connection.execute("PRAGMA table_info(archive_revisions)")}
            if "version_id" not in revision_columns:
                connection.execute("ALTER TABLE archive_revisions ADD COLUMN version_id TEXT")

    def _lock_for(self, session_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._session_locks.setdefault(session_id, threading.Lock())

    @staticmethod
    def _session_digest(session_id: str) -> str:
        return hashlib.sha256(session_id.encode("utf-8")).hexdigest()

    def _current(self, session_id: str) -> sqlite3.Row | None:
        with self._connect() as connection:
            return connection.execute(
                """SELECT s.owner_id, s.state, s.current_revision, r.object_key, r.sha256,
                          r.size_bytes, r.status
                   FROM archive_sessions s LEFT JOIN archive_revisions r
                     ON r.session_id=s.session_id AND r.revision=s.current_revision
                   WHERE s.session_id=?""",
                (session_id,),
            ).fetchone()

    def status(self, session_id: str) -> dict[str, Any] | None:
        row = self._current(session_id)
        if row is None:
            return None
        return {
            "owner_id": row["owner_id"],
            "state": row["state"],
            "revision": row["current_revision"],
            "object_key": row["object_key"],
            "sha256": row["sha256"],
            "size_bytes": row["size_bytes"],
            "revision_status": row["status"],
        }

    def has_session(self, session_id: str) -> bool:
        row = self._current(session_id)
        return bool(row and row["state"] != "deleted")

    def list_archived_sessions(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT session_id, owner_id, updated_at, metadata_json FROM archive_sessions
                   WHERE state IN ('archived', 'stale') ORDER BY updated_at DESC"""
            ).fetchall()
        result = []
        for row in rows:
            try:
                metadata = json.loads(row["metadata_json"])
            except (TypeError, ValueError):
                metadata = {}
            result.append(
                {
                    "session_id": row["session_id"],
                    "owner_id": row["owner_id"],
                    "updated_at": row["updated_at"],
                    **metadata,
                }
            )
        return result

    def is_archived(self, session_id: str, owner_id: str | None = None) -> bool:
        row = self._current(session_id)
        if row is None or row["state"] != "archived" or row["status"] != "archived":
            return False
        return owner_id is None or row["owner_id"] == owner_id

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _zip_workspace(workspace_path: Path, archive_path: Path) -> None:
        root = workspace_path.resolve(strict=True)
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for current, directories, filenames in os.walk(root, followlinks=False):
                current_path = Path(current)
                directories[:] = [name for name in directories if not (current_path / name).is_symlink()]
                for filename in filenames:
                    path = current_path / filename
                    if path.is_symlink() or not path.is_file():
                        continue
                    relative = path.relative_to(root).as_posix()
                    archive.write(path, arcname=relative)

    def archive_session(
        self,
        session_id: str,
        workspace_path: str | Path,
        *,
        owner_id: str | None = None,
    ) -> dict[str, Any]:
        """Upload a stable workspace revision and verify the stored metadata."""
        owner_id = owner_id or session_id
        workspace = Path(workspace_path).resolve(strict=True)
        lock = self._lock_for(session_id)
        with lock:
            with tempfile.TemporaryDirectory(prefix="artifact-archive-") as temporary_directory:
                archive_path = Path(temporary_directory) / "workspace.zip"
                self._zip_workspace(workspace, archive_path)
                digest = self._hash_file(archive_path)
                size_bytes = archive_path.stat().st_size
                metadata = self._workspace_metadata(workspace)
                existing = self._current(session_id)
                if existing:
                    if existing["owner_id"] != owner_id:
                        raise ArchiveError("Session owner does not match archive owner")
                    if existing["state"] in {"deleting", "deleted"}:
                        raise ArchiveError("Session archive is being deleted")
                    if (
                        existing["state"] in {"archived", "stale"}
                        and existing["status"] == "archived"
                        and existing["sha256"] == digest
                        and existing["size_bytes"] == size_bytes
                    ):
                        if existing["state"] == "stale":
                            with self._connect() as connection:
                                connection.execute(
                                    "UPDATE archive_sessions SET state='archived', updated_at=?, error=NULL WHERE session_id=?",
                                    (time.time(), session_id),
                                )
                        return self.status(session_id) or {"state": "archived"}

                with self._connect() as connection:
                    row = connection.execute(
                        "SELECT owner_id, state, current_revision FROM archive_sessions WHERE session_id=?",
                        (session_id,),
                    ).fetchone()
                    if row and row["owner_id"] != owner_id:
                        raise ArchiveError("Session owner does not match archive owner")
                    if row and row["state"] in {"deleting", "deleted"}:
                        raise ArchiveError("Session archive is being deleted")
                    revision_row = connection.execute(
                        "SELECT COALESCE(MAX(revision), 0) AS revision FROM archive_revisions WHERE session_id=?",
                        (session_id,),
                    ).fetchone()
                    revision = int(revision_row["revision"]) + 1
                    object_key = (
                        f"sessions/{self._session_digest(session_id)}/revisions/{revision:08d}.zip"
                    )
                    now = time.time()
                    connection.execute(
                        """INSERT INTO archive_sessions(session_id, owner_id, state, current_revision, updated_at, metadata_json)
                           VALUES (?, ?, 'uploading', NULL, ?, ?)
                           ON CONFLICT(session_id) DO UPDATE SET state='uploading', updated_at=excluded.updated_at,
                             error=NULL, metadata_json=excluded.metadata_json""",
                        (session_id, owner_id, now, json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
                    )
                    connection.execute(
                        """INSERT INTO archive_revisions(session_id, revision, object_key, sha256, size_bytes, status)
                           VALUES (?, ?, ?, ?, ?, 'uploading')""",
                        (session_id, revision, object_key, digest, size_bytes),
                    )
                try:
                    stored = self.object_store.put_file(self.bucket, object_key, archive_path, sha256=digest)
                    if stored.size_bytes != size_bytes or stored.sha256 != digest:
                        raise ArchiveError("Object store upload metadata did not match the archive")
                    if not self._verify_head(object_key, digest, size_bytes):
                        raise ArchiveError("Object store verification failed after upload")
                except Exception as exc:
                    self._mark_failed(session_id, revision, str(exc))
                    raise ArchiveError("Workspace archive upload failed") from exc
                now = time.time()
                with self._connect() as connection:
                    connection.execute(
                        "UPDATE archive_revisions SET status='archived', archived_at=?, version_id=?, error=NULL WHERE session_id=? AND revision=?",
                        (now, stored.version_id, session_id, revision),
                    )
                    connection.execute(
                        "UPDATE archive_sessions SET state='archived', current_revision=?, updated_at=?, error=NULL WHERE session_id=?",
                        (revision, now, session_id),
                    )
            return self.status(session_id) or {"state": "archived"}

    def _verify_head(self, key: str, digest: str, size_bytes: int) -> bool:
        head = self.object_store.head_object(self.bucket, key)
        return head.size_bytes == size_bytes and head.sha256 == digest

    @staticmethod
    def _workspace_metadata(workspace: Path) -> dict[str, Any]:
        try:
            form = json.loads((workspace / "inputs" / "form_input.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(form, dict):
            return {}
        payload = form.get("payload") if isinstance(form.get("payload"), dict) else form
        return {
            "candidate_name": payload.get("candidate_name"),
            "target_role": payload.get("target_role"),
        }

    def _mark_failed(self, session_id: str, revision: int, error: str) -> None:
        clean_error = error[:500]
        with self._connect() as connection:
            connection.execute(
                "UPDATE archive_revisions SET status='failed', error=? WHERE session_id=? AND revision=?",
                (clean_error, session_id, revision),
            )
            connection.execute(
                """UPDATE archive_sessions SET state=CASE WHEN current_revision IS NOT NULL THEN 'stale' ELSE 'failed' END,
                   updated_at=?, error=? WHERE session_id=?""",
                (time.time(), clean_error, session_id),
            )

    def restore_session(
        self,
        session_id: str,
        destination: str | Path,
        *,
        owner_id: str | None = None,
    ) -> Path:
        """Verify and safely materialize the current archive when local workspace expired."""
        owner_id = owner_id or session_id
        with self._lock_for(session_id):
            return self._restore_session_locked(session_id, destination, owner_id)

    def _restore_session_locked(
        self,
        session_id: str,
        destination: str | Path,
        owner_id: str,
    ) -> Path:
        row = self._current(session_id)
        if row is None or row["state"] in {"deleting", "deleted"}:
            raise ArchiveNotFound("Session archive not found")
        if row["owner_id"] != owner_id:
            raise ArchiveError("Session owner does not match archive owner")
        if row["state"] not in {"archived", "stale"} or row["status"] != "archived":
            raise ArchiveError("Session archive is not ready to restore")

        destination_path = Path(destination).resolve()
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".restore-", dir=destination_path.parent))
        archive_path = stage / "snapshot.zip"
        extracted = stage / "workspace"
        try:
            head = self.object_store.head_object(self.bucket, row["object_key"])
            if head.size_bytes != row["size_bytes"] or head.sha256 != row["sha256"]:
                raise ArchiveError("Stored archive metadata does not match its catalog")
            if head.size_bytes > self.max_restore_bytes:
                raise ArchiveError("Archive exceeds the restore size limit")
            try:
                stored = self.object_store.get_file(self.bucket, row["object_key"], archive_path)
            except Exception as exc:
                raise ArchiveError("Archived workspace could not be downloaded") from exc
            if stored.size_bytes != row["size_bytes"] or stored.sha256 != row["sha256"]:
                raise ArchiveError("Stored archive metadata does not match its catalog")
            if self._hash_file(archive_path) != row["sha256"]:
                raise ArchiveError("Stored archive checksum verification failed")
            extracted.mkdir()
            total_bytes = 0
            with zipfile.ZipFile(archive_path, "r") as archive:
                for info in archive.infolist():
                    relative = PurePosixPath(info.filename)
                    mode = info.external_attr >> 16
                    if (
                        relative.is_absolute()
                        or not relative.parts
                        or any(part in {"", ".", ".."} for part in relative.parts)
                        or "\\" in info.filename
                        or stat.S_ISLNK(mode)
                    ):
                        raise ArchiveError("Archive contains an unsafe path")
                    total_bytes += info.file_size
                    if total_bytes > self.max_restore_bytes:
                        raise ArchiveError("Archive exceeds the restore size limit")
                    target = extracted.joinpath(*relative.parts)
                    resolved = target.resolve()
                    if not resolved.is_relative_to(extracted.resolve()):
                        raise ArchiveError("Archive path escaped the workspace")
                    if info.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(info, "r") as source, target.open("wb") as output:
                            shutil.copyfileobj(source, output)
            try:
                form = json.loads((extracted / "inputs" / "form_input.json").read_text(encoding="utf-8"))
                stored_session_id = form.get("session_id") if isinstance(form, dict) else None
            except (OSError, ValueError):
                stored_session_id = None
            if stored_session_id != session_id:
                raise ArchiveError("Restored workspace belongs to a different session")
            if destination_path.exists():
                raise ArchiveError("Workspace appeared during archive restore")
            os.replace(extracted, destination_path)
            return destination_path
        except ArchiveError:
            raise
        except Exception as exc:
            raise ArchiveError("Archived workspace could not be restored") from exc
        finally:
            shutil.rmtree(stage, ignore_errors=True)

    def delete_session(self, session_id: str, *, owner_id: str | None = None) -> None:
        """Tombstone the session, remove every object revision, and retain the tombstone."""
        owner_id = owner_id or session_id
        with self._lock_for(session_id):
            self._delete_session_locked(session_id, owner_id)

    def _delete_session_locked(self, session_id: str, owner_id: str) -> None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT owner_id, state FROM archive_sessions WHERE session_id=?", (session_id,)
            ).fetchone()
            if row and row["owner_id"] != owner_id:
                raise ArchiveError("Session owner does not match archive owner")
            now = time.time()
            if row is None:
                connection.execute(
                    "INSERT INTO archive_sessions(session_id, owner_id, state, updated_at) VALUES (?, ?, 'deleting', ?)",
                    (session_id, owner_id, now),
                )
            elif row["state"] != "deleted":
                connection.execute(
                    "UPDATE archive_sessions SET state='deleting', updated_at=?, error=NULL WHERE session_id=?",
                    (now, session_id),
                )
        prefix = f"sessions/{self._session_digest(session_id)}/"
        self.object_store.delete_prefix(self.bucket, prefix)
        with self._connect() as connection:
            connection.execute(
                "UPDATE archive_revisions SET status='deleted' WHERE session_id=?", (session_id,)
            )
            connection.execute(
                "UPDATE archive_sessions SET state='deleted', updated_at=?, error=NULL WHERE session_id=?",
                (time.time(), session_id),
            )

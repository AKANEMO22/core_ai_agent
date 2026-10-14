"""Ports and metadata contracts for durable artifact storage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class StoredObject:
    key: str
    size_bytes: int
    sha256: str
    version_id: str | None = None


class ObjectStore(Protocol):
    """S3-compatible object operations required by the archive layer."""

    def ensure_bucket(self, bucket: str, *, versioning: bool = True) -> None: ...

    def check_bucket(self, bucket: str) -> None: ...

    def put_file(self, bucket: str, key: str, path: Path, *, sha256: str) -> StoredObject: ...

    def get_file(self, bucket: str, key: str, destination: Path) -> StoredObject: ...

    def head_object(self, bucket: str, key: str) -> StoredObject: ...

    def delete_prefix(self, bucket: str, prefix: str) -> None: ...

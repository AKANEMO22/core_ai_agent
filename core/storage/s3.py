"""Boto3 adapter for an S3-compatible object store such as MinIO."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import boto3
from botocore.client import Config

from core.storage.artifacts import StoredObject


class S3ObjectStore:
    """Keep S3 SDK details behind the storage port."""

    def __init__(
        self,
        *,
        endpoint_url: str,
        access_key: str,
        secret_key: str,
        region: str = "us-east-1",
        verify_tls: bool = True,
        timeout_seconds: int = 30,
    ) -> None:
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
            verify=verify_tls,
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
                connect_timeout=min(10, timeout_seconds),
                read_timeout=timeout_seconds,
                retries={"max_attempts": 3, "mode": "standard"},
            ),
        )

    def ensure_bucket(self, bucket: str, *, versioning: bool = True) -> None:
        try:
            self.client.head_bucket(Bucket=bucket)
        except self.client.exceptions.NoSuchBucket:
            self.client.create_bucket(Bucket=bucket)
        except Exception as exc:
            code = str(getattr(exc, "response", {}).get("Error", {}).get("Code", ""))
            if code in {"404", "NoSuchBucket", "NotFound"}:
                self.client.create_bucket(Bucket=bucket)
            else:
                raise
        if versioning:
            self.client.put_bucket_versioning(
                Bucket=bucket,
                VersioningConfiguration={"Status": "Enabled"},
            )

    def check_bucket(self, bucket: str) -> None:
        self.client.head_bucket(Bucket=bucket)

    def put_file(self, bucket: str, key: str, path: Path, *, sha256: str) -> StoredObject:
        size_bytes = path.stat().st_size
        with path.open("rb") as body:
            result = self.client.put_object(
                Bucket=bucket,
                Key=key,
                Body=body,
                ContentLength=size_bytes,
                ContentType="application/zip",
                Metadata={"sha256": sha256},
            )
        return StoredObject(
            key=key,
            size_bytes=size_bytes,
            sha256=sha256,
            version_id=result.get("VersionId"),
        )

    def get_file(self, bucket: str, key: str, destination: Path) -> StoredObject:
        destination.parent.mkdir(parents=True, exist_ok=True)
        response = self.client.get_object(Bucket=bucket, Key=key)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".part", dir=destination.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as output:
                body = response["Body"]
                while chunk := body.read(1024 * 1024):
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return StoredObject(
            key=key,
            size_bytes=destination.stat().st_size,
            sha256=response.get("Metadata", {}).get("sha256", ""),
            version_id=response.get("VersionId"),
        )

    def head_object(self, bucket: str, key: str) -> StoredObject:
        response = self.client.head_object(Bucket=bucket, Key=key)
        return StoredObject(
            key=key,
            size_bytes=response["ContentLength"],
            sha256=response.get("Metadata", {}).get("sha256", ""),
            version_id=response.get("VersionId"),
        )

    def delete_prefix(self, bucket: str, prefix: str) -> None:
        paginator = self.client.get_paginator("list_object_versions")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            entries = [
                {"Key": item["Key"], "VersionId": item.get("VersionId", "null")}
                for field in ("Versions", "DeleteMarkers")
                for item in page.get(field, [])
            ]
            for offset in range(0, len(entries), 1000):
                batch = entries[offset : offset + 1000]
                if batch:
                    result = self.client.delete_objects(
                        Bucket=bucket,
                        Delete={"Objects": batch, "Quiet": True},
                    )
                    if result.get("Errors"):
                        raise RuntimeError("Object-store version deletion was incomplete")

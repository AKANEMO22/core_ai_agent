# RAG Artifact Storage Implementation Plan

> **For agentic workers:** Execute this plan inline in this worktree. Preserve the carried user changes. Keep each slice small and report its validation evidence before moving to the next stage.

**Goal:** Archive completed session workspaces durably in a MinIO S3-compatible object store, restore them for inspection, and ensure session deletion removes archived artifacts.

**Architecture:** Keep local `SessionWorkspace` as the runtime filesystem contract. Add a storage port, an S3 adapter, and a SQLite archive catalog/outbox. The existing retention sweep archives completed sessions before raw-input or workspace cleanup; inspection restores a verified archive when local files have expired. Compose runs a source-pinned MinIO Community build for local development only; production distribution and credentials remain deployment configuration.

**Tech Stack:** Python 3.11, FastAPI lifespan, SQLite, boto3 S3 client, Docker Compose, MinIO Community source tag `RELEASE.2025-10-15T17-29-55Z` for local development.

---

### Task 1: Add object-store port, S3 adapter, and durable archive catalog

**Files:**
- Create: `core/storage/artifacts.py`
- Create: `core/storage/s3.py`
- Create: `core/storage/archive.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `config/settings.py`

- [ ] Define a small S3 object port used by archive code: put file, get object to file, head metadata, delete key, create bucket, and enable bucket versioning.
- [ ] Implement the S3 adapter with a boto3 client configured by endpoint, region, credentials, TLS, and bounded request timeout; verify SHA-256 from application metadata instead of interpreting ETag as a digest.
- [ ] Implement a SQLite catalog with WAL/FULL durability and rows for session owner, archive state, archive revision, object key, byte size, SHA-256, timestamps, and deletion tombstone.
- [ ] Create a deterministic workspace ZIP without following symlinks; key it by opaque session digest and immutable revision; verify hash/size after upload.
- [ ] Add verified, traversal-safe restore and idempotent all-revision deletion; reject restore for deleting/deleted tombstones.
- [ ] Add validated settings with archive disabled by default outside the explicit local Compose profile; fail fast when enabled without endpoint, credentials, bucket, or durable catalog path.

### Task 2: Add a local MinIO development service and wire startup

**Files:**
- Create: `docker/minio/Dockerfile`
- Modify: `compose.yaml`
- Modify: `.env.docker.example`
- Modify: `api/main.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

- [ ] Build the upstream Community source tag into a local development image because the upstream no longer publishes current Community images; use the fixed release tag and label the service development-only.
- [ ] Add loopback-only S3 API and object browser ports, persistent object/catalog volumes, readiness check, restart policy, and explicit development credentials sourced through Compose variables.
- [ ] On app lifespan startup, create the bucket idempotently, enable versioning where supported, and verify the S3 endpoint; if object storage is disabled, create no network client.
- [ ] Do not use the MinIO root password in production; production configuration must use externally provisioned least-privilege S3 credentials and a supported MinIO distribution.

### Task 3: Archive report-ready sessions before retention cleanup

**Files:**
- Modify: `core/workspace/retention.py`
- Modify: `core/workspace/cleanup.py`
- Modify: `api/main.py`
- Modify: `config/settings.py`

- [ ] Inject an archive service into the retention service and scan only `REPORT_READY` sessions for archival.
- [ ] Archive before deleting raw inputs; if upload fails, keep originals and return a failed archive state for retry.
- [ ] Protect report-ready workspaces from expiry cleanup until the catalog confirms a verified archive; preserve old behavior for archive-disabled local/test environments.
- [ ] Make archive jobs idempotent so periodic sweeps skip the current archived revision and retry failed writes safely.

### Task 4: Restore archived workspaces and delete archived sessions

**Files:**
- Modify: `api/inspect.py`
- Modify: `api/routes.py`
- Modify: `core/workspace/archive.py` or `core/storage/archive.py`

- [ ] When inspection finds no local workspace, check the durable catalog, verify the session scope, restore the ZIP into a temporary directory, verify every path and checksum, then atomically materialize the workspace.
- [ ] Treat missing catalog entries as not found; surface storage outage as retryable service unavailable without claiming data was absent.
- [ ] During explicit session deletion, write a deletion tombstone, revoke the session, cancel work, delete archived object revisions, and only then remove Redis/local data; failed object deletion remains retryable and is not reported as complete.
- [ ] Deny restore/list/download for tombstoned sessions so late archive jobs cannot recreate user data.

### Task 5: Document operations and validate the local stack

**Files:**
- Modify: `README.md`
- Modify: `.env.docker.example`
- Modify: `compose.yaml`

- [ ] Document local MinIO UI/API ports, the development-only credentials, archive state behavior, restore/delete behavior, and upstream Community maintenance limitation.
- [ ] Document backup as a separate requirement from a named volume and give the exact restore-check workflow.
- [ ] Validate Compose configuration, compile modified Python modules, build the local MinIO image, and verify S3 create-bucket/put/head/get/delete through the running stack.

## Completion criteria

- A completed workspace is recorded as archived only after object size and SHA-256 metadata match.
- Retention cannot remove the only remaining copy of a report-ready session.
- An expired local workspace can be restored from object storage with its original artifact paths and bytes.
- Session deletion removes all stored revisions and retries safely after object-store failure.
- Redis remains the coordination/cache layer; session artifact durability does not depend on Redis TTL.
- Production is not represented as ready until a maintained MinIO distribution, least-privilege credentials, owner authorization, and independent backup/restore are configured.

## Validation boundary

The existing worktree has extensive uncommitted user changes. They are carried into this isolated worktree unchanged. Do not run or add tests unless the user requests test/verification; use static/build/runtime evidence that does not invoke the test suite, and list any gap explicitly.

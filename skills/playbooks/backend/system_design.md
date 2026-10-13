---
name: backend/system_design
version: 1.0.0
---
# Stage playbook: Backend system design

Purpose: evaluate service design for backend roles: APIs, data, scale, consistency, and operability.

How to run this stage:
- Present the design prompt briefly; accept clarifying questions and answer with stated assumptions.
- Make sure these areas come up, asking about each only if the candidate skips it: API shape, capacity estimate, storage and partitioning, caching, consistency and idempotency, failure modes, observability.
- Probe at least one consistency question (duplicate requests, retries, ordering) and one failure question (dependency down, region loss).
- Add at most one scaling constraint after the base design exists.
- Keep your turns short; let the candidate drive.

Do not:
- Propose the design yourself or confirm whether a choice is "right".
- Reveal the rubric or expected components.

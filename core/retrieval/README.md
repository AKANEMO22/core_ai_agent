# Retrieval integration

The profiler and control-plane agents use one shared contract: `RetrievalQuery` and
`RetrievedChunk` in `port.py`. The default `rag` backend implements that contract in-process
through `rag.adapter.RAGRetrievalClient`; `InterviewKnowledgeService` and `RetrieveTool` share the
same client instance.

## Data and indexing

- The agent RAG indexes `data/interview_frameworks.json` and loads the separate
  `data/it_seniority_rubrics.json` level rubric.
- Index snapshots are stored at `RAG_INDEX_PATH` (default `data/rag/index.db`). If the corpus is
  new or changed, startup builds and activates a versioned index using offline LSA, BM25, and the
  bounded concept graph.
- Set `RAG_DATA_DIR`, `RAG_INDEX_PATH`, or `RAG_EMBEDDING_PROVIDER` to override the local paths or
  embedding provider. The default needs no external embedding key.
- Candidate and job profile JSON files are not loaded by the agent adapter. Company passages are
  generic framework context; the corpus does not contain verified company-specific intelligence.

## Source-level support

The bundled question corpus covers `Intern`, `Fresher`, and `Junior`. Mid and Senior receive a
project-maintained generic cross-role rubric, combined with target-role competencies, but no
retrieved question-bank items; the core generates probes from that rubric, the candidate profile
and JD. This material is not company-specific or human-validated. Staff/Principal is outside the
supported Intern–Senior scope. The adapter never relabels junior questions as senior evidence. The
source evaluation matrix is projected to core rubric chunks. Question-to-rubric matching is
query-time retrieval against each rubric criterion and is marked in returned metadata.

Company-specific context is unavailable unless live web search is configured with
`DOCKER_BRAVE_SEARCH_API_KEY`. Search results and bounded fetched excerpts are stored with
provenance but remain unreviewed leads; they must not be presented as official company policy.
Plans retain source limitations and the user reviews the proposed round, duration and evidence
before starting.

Duration-derived turn counts and per-stage minute allocations are estimates. The live runtime
enforces turn counts and does not stop automatically at a wall-clock deadline.

## Explicit mock backend

The deterministic mock remains useful for tests and fixture-driven development:

```powershell
$env:RETRIEVAL_BACKEND = "mock"
uv run pytest tests/unit/test_retrieval.py
```

Run the selected backend against the shared contract with:

```powershell
uv run pytest tests/contract/test_retrieval_contract.py
```

The contract checks collections, metadata, filter semantics, ordering, `top_k`, and knowledge-pack
shape. The bundled retrieval evaluation cases are synthetic and unreviewed; they are not evidence
of human-validated answer quality.

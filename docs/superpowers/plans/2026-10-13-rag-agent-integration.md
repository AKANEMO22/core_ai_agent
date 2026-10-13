# RAG to Agent Integration Plan

## Goal

Replace the application flow's default mock retrieval backend with the newly added local RAG corpus while keeping one stable contract between retrieval and the profiler/agent tools.

## Decisions

- `core.retrieval.port.RetrievalQuery` and `RetrievedChunk` remain the canonical shared schema.
- `rag` implements that contract through one in-process `RAGRetrievalClient`; the same client is supplied to the profiler knowledge service and agent retrieval tools.
- The local index is built from `data/interview_frameworks.json` with offline LSA embeddings, BM25, and the existing graph retriever. It never loads candidate or job profile files in the agent adapter.
- RAG's session-level API models remain separate and receive distinct names so they cannot be mistaken for the core retrieval contract.
- Only `Intern`, `Fresher`, and `Junior` source questions are available. Unsupported core levels return no chunks and are not silently mapped to junior content.
- The mock backend remains available for explicit tests and fixtures; application defaults and Compose use RAG.
- Docker includes only the interview framework corpus and stores generated indexes in a dedicated writable volume.

## Implementation steps

1. Add integration tests for schema fields, filters, relevance, provenance, unsupported levels, and PII isolation using a small local corpus.
2. Remove imports that point into the sibling `src.*` project; define the minimal validated RAG corpus and request models inside this package.
3. Implement the core retrieval adapter over the existing RAG index, with role-family projection, question/rubric/profile/company collections, and explicit source-level metadata.
4. Register/configure the adapter and make both profiler and control-plane tools receive the same client.
5. Update package dependencies, Docker, Compose, and retrieval documentation for an offline RAG default and safe corpus inclusion.
6. Run targeted retrieval tests, the full test suite, lock/build checks, and a container configuration/startup smoke check where the local environment permits.

## Acceptance criteria

- The registered default backend is `rag`, and its adapter passes the shared retrieval contract.
- All four collections produce correctly shaped chunks from the framework corpus.
- Filters, ordering, `top_k`, provenance, and error handling match the core port contract.
- Agent retrieval does not read candidate/contact datasets or put them into the RAG image/index.
- A missing index is built from the checked-in corpus without external API keys; a changed corpus creates a fresh version.
- No unsupported source level is represented as supported evidence.
- Existing mock-based unit and integration flows remain available by explicitly selecting `mock`.

## Verification and known limits

- The bundled evaluation labels are synthetic and unreviewed, so retrieval quality must not be described as human-validated.
- Local LSA is an offline baseline; BM25 and graph retrieval remain available, while higher-quality remote/neural embeddings can be configured separately.
- The bundled corpus does not cover mid/senior/staff or verified company-specific knowledge.

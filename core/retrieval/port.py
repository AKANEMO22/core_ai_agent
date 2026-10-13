"""Retrieval port: the shared contract between the interview engine and its knowledge source.

The engine depends only on `RetrievalClient`. The default `RAGRetrievalClient` adapts the local
framework index; `MockRetrievalClient` remains available for tests and explicit fixtures.

Collections and the metadata each chunk must carry:

- `question_bank`: content is the question text.
    question_id (str), competency (str; the rubric scope used for query-time matching),
    role_family (str: "backend" | "frontend" | other domain families | "any"),
    levels (list[str]: "junior" | "mid" | "senior" | "staff"),
    stage_type (str: "warmup" | "technical" | "system_design" | "behavioral" | "wrapup"),
    difficulty (int 1-5), expected_points (list[str]), follow_ups (list[str]).
- `rubric`: content is the competency description.
    competency (str), name (str), role_family (str), levels (list[str]), weight (float),
    must_have (bool), pass_level (int 1-4), anchors (dict[str, str] keyed "1".."4").
- `role_profile`: content is the role expectation summary.
    role_family (str), level (str), competencies (list[str]), pass_threshold (float on the 1-4 scale),
    rubric_version (str).
- `company_knowledge`: content is a knowledge passage.
    company (str, lowercase; "generic" for fallback passages), topic (str), confidence (float 0-1).

Filter semantics (`RetrievalQuery.filters`), applied per key against chunk metadata:
- scalar filter value: matches when metadata equals it, or metadata is a list containing it;
- list filter value: matches when any of its elements matches by the scalar rule;
- a key missing from the chunk metadata never matches;
- string comparison is case-insensitive.

The bundled RAG corpus currently supplies Intern/Fresher/Junior evidence only. Its adapter maps
those source levels to core `junior`; core `mid`, `senior`, and `staff` queries return no chunks.
The adapter does not invent higher-level evidence.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Protocol

from core.runtime.agent import sanitize_secrets


class RetrievalCollection(str, Enum):
    """Logical knowledge collections the interview engine queries."""

    QUESTION_BANK = "question_bank"
    RUBRIC = "rubric"
    ROLE_PROFILE = "role_profile"
    COMPANY_KNOWLEDGE = "company_knowledge"


@dataclass
class RetrievalQuery:
    """A single retrieval request against one collection.

    Attributes:
        collection: Target collection name (see RetrievalCollection).
        text: Free-text query used for relevance ranking.
        filters: Metadata filters applied before ranking (see module docstring).
        top_k: Maximum number of chunks to return.
    """

    collection: str
    text: str
    filters: Dict[str, Any] = field(default_factory=dict)
    top_k: int = 5


@dataclass
class RetrievedChunk:
    """A ranked knowledge chunk returned by the retrieval backend.

    Attributes:
        chunk_id: Stable identifier of the chunk in the backend index.
        collection: Collection the chunk belongs to.
        content: Chunk text.
        score: Relevance score; higher is more relevant. Scale is backend-specific.
        metadata: Structured fields defined per collection (see module docstring).
        source: Provenance reference (URL or URI) for auditability.
    """

    chunk_id: str
    collection: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to a JSON-serializable dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "collection": self.collection,
            "content": self.content,
            "score": self.score,
            "metadata": dict(self.metadata),
            "source": self.source,
        }


class RetrievalError(Exception):
    """Raised when the retrieval backend fails, times out, or returns an unusable response."""

    def __init__(self, message: str) -> None:
        super().__init__(sanitize_secrets(message))


class RetrievalClient(Protocol):
    """Protocol every retrieval backend adapter must implement."""

    backend_name: str

    async def search(self, query: RetrievalQuery) -> List[RetrievedChunk]:
        """Return up to `query.top_k` chunks from `query.collection`, most relevant first.

        Raises:
            RetrievalError: On backend failure. Adapters must not leak credentials in messages.
        """
        ...

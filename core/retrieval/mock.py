"""Deterministic in-memory retrieval adapter backed by a JSON fixture corpus."""

import asyncio
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.retrieval.port import RetrievalError, RetrievalQuery, RetrievedChunk

DEFAULT_CORPUS_PATH = Path(__file__).parent / "fixtures" / "mock_corpus.json"

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "for", "from", "how", "i", "in",
    "is", "it", "me", "of", "on", "or", "that", "the", "to", "what", "when", "with", "you", "your",
}


def _tokenize(text: str) -> List[str]:
    """Lowercase, split on non-alphanumerics, and drop stopwords."""
    return [t for t in _TOKEN_PATTERN.findall(text.lower()) if t not in _STOPWORDS]


def _normalize(value: Any) -> Any:
    """Normalize strings for case-insensitive filter comparison."""
    return value.strip().lower() if isinstance(value, str) else value


def _scalar_matches(metadata_value: Any, filter_value: Any) -> bool:
    """Apply the scalar filter rule from the retrieval port contract."""
    expected = _normalize(filter_value)
    if isinstance(metadata_value, list):
        return any(_normalize(item) == expected for item in metadata_value)
    return _normalize(metadata_value) == expected


def matches_filters(metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    """Return True if chunk metadata satisfies every filter (see port contract)."""
    for key, filter_value in filters.items():
        if key not in metadata:
            return False
        metadata_value = metadata[key]
        if isinstance(filter_value, (list, tuple, set)):
            if not any(_scalar_matches(metadata_value, fv) for fv in filter_value):
                return False
        elif not _scalar_matches(metadata_value, filter_value):
            return False
    return True


class MockRetrievalClient:
    """In-memory RetrievalClient for development and tests.

    Ranks chunks by query-token overlap against chunk content and string metadata.
    Results are deterministic: ties are broken by chunk_id.

    Attributes:
        backend_name: Identifier recorded in retrieved artifacts.
        latency_seconds: Simulated per-call latency.
        fail_with: Optional exception raised on every call (failure injection).
        queries: Log of every query received, for test assertions.
    """

    backend_name = "mock"

    def __init__(
        self,
        corpus: Optional[List[Dict[str, Any]]] = None,
        corpus_path: Optional[Path] = None,
        latency_seconds: float = 0.0,
        fail_with: Optional[Exception] = None,
    ) -> None:
        if corpus is None:
            path = corpus_path or DEFAULT_CORPUS_PATH
            corpus = json.loads(path.read_text(encoding="utf-8"))["chunks"]
        self._chunks: List[Dict[str, Any]] = list(corpus)
        self._index: Dict[str, set[str]] = {
            chunk["chunk_id"]: set(_tokenize(self._searchable_text(chunk))) for chunk in self._chunks
        }
        self.latency_seconds = latency_seconds
        self.fail_with = fail_with
        self.queries: List[RetrievalQuery] = []

    @staticmethod
    def _searchable_text(chunk: Dict[str, Any]) -> str:
        """Concatenate content with string and string-list metadata values."""
        parts = [str(chunk.get("content", ""))]
        for value in chunk.get("metadata", {}).values():
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, list):
                parts.extend(str(v) for v in value if isinstance(v, str))
        return " ".join(parts)

    def _score(self, chunk_id: str, query_tokens: List[str]) -> float:
        """Fraction of unique query tokens present in the chunk, damped by chunk length."""
        if not query_tokens:
            return 0.0
        chunk_tokens = self._index.get(chunk_id, set())
        unique_query = set(query_tokens)
        overlap = len(unique_query & chunk_tokens)
        if overlap == 0:
            return 0.0
        length_damping = 1.0 / math.log2(2 + len(chunk_tokens) / 50.0)
        return round((overlap / len(unique_query)) * length_damping, 4)

    async def search(self, query: RetrievalQuery) -> List[RetrievedChunk]:
        """Return the top_k filtered chunks of a collection ranked by token overlap."""
        self.queries.append(query)
        if self.latency_seconds > 0:
            await asyncio.sleep(self.latency_seconds)
        if self.fail_with is not None:
            raise RetrievalError(f"Mock retrieval failure: {self.fail_with}")

        collection = str(getattr(query.collection, "value", query.collection))
        query_tokens = _tokenize(query.text or "")
        scored: List[RetrievedChunk] = []
        for chunk in self._chunks:
            if chunk.get("collection") != collection:
                continue
            metadata = chunk.get("metadata", {})
            if not matches_filters(metadata, query.filters):
                continue
            scored.append(
                RetrievedChunk(
                    chunk_id=chunk["chunk_id"],
                    collection=collection,
                    content=str(chunk.get("content", "")),
                    score=self._score(chunk["chunk_id"], query_tokens),
                    metadata=dict(metadata),
                    source=str(chunk.get("source", f"mock://{collection}/{chunk['chunk_id']}")),
                )
            )

        scored.sort(key=lambda c: (-c.score, c.chunk_id))
        return scored[: max(0, query.top_k)]

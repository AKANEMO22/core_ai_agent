"""Retrieval backend selection from application settings."""

from typing import Optional

from config.settings import Settings
from core.retrieval.knowledge import InterviewKnowledgeService
from core.retrieval.mock import MockRetrievalClient
from core.retrieval.port import RetrievalClient


def build_retrieval_client(settings: Settings) -> RetrievalClient:
    """Instantiate the retrieval adapter named by `RETRIEVAL_BACKEND`.

    `rag` is the in-process framework-backed retriever; `mock` remains available for
    isolated tests and explicitly selected development fixtures.

    Raises:
        ValueError: If the configured backend is unknown.
    """
    backend = settings.RETRIEVAL_BACKEND.strip().lower()
    if backend == "rag":
        from rag.adapter import RAGRetrievalClient

        return RAGRetrievalClient(
            data_dir=settings.RAG_DATA_DIR,
            index_path=settings.RAG_INDEX_PATH,
            embedding_provider=settings.RAG_EMBEDDING_PROVIDER,
        )
    if backend == "mock":
        return MockRetrievalClient()
    raise ValueError(
        f"Unsupported RETRIEVAL_BACKEND '{backend}'. Choose 'rag' or 'mock'."
    )


def build_knowledge_service(
    settings: Settings,
    client: Optional[RetrievalClient] = None,
) -> InterviewKnowledgeService:
    """Build the interview knowledge service around the configured (or injected) adapter."""
    return InterviewKnowledgeService(
        client=client or build_retrieval_client(settings),
        timeout_seconds=settings.RETRIEVAL_TIMEOUT_SECONDS,
        questions_per_competency=settings.RETRIEVAL_QUESTIONS_PER_COMPETENCY,
        company_top_k=settings.RETRIEVAL_COMPANY_TOP_K,
    )

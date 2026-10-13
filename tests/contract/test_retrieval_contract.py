"""Retrieval backend contract suite.

Every adapter registered in `core/retrieval/factory.py` must pass this suite. It runs against the
backend named by RETRIEVAL_BACKEND (default "rag"):

    uv run pytest tests/contract/test_retrieval_contract.py

The suite checks behavior the interview engine relies on: collections, metadata fields, filter
semantics, top_k, ordering, error type, and that knowledge packs are usable for grounding.
"""

import asyncio
from typing import List

import pytest

from config.settings import get_settings
from core.retrieval import (
    InterviewKnowledgeService,
    RetrievalCollection,
    RetrievalError,
    RetrievalQuery,
    RetrievedChunk,
    ground_stage_plan,
)
from core.retrieval.factory import build_retrieval_client

REQUIRED_METADATA = {
    RetrievalCollection.QUESTION_BANK.value: {
        "question_id", "competency", "role_family", "levels", "stage_type", "difficulty", "expected_points", "follow_ups",
    },
    RetrievalCollection.RUBRIC.value: {
        "competency", "name", "role_family", "levels", "weight", "must_have", "pass_level", "anchors",
    },
    RetrievalCollection.ROLE_PROFILE.value: {
        "role_family", "level", "competencies", "pass_threshold", "rubric_version",
    },
    RetrievalCollection.COMPANY_KNOWLEDGE.value: {"company", "topic", "confidence"},
}


@pytest.fixture(scope="module")
def client():
    return build_retrieval_client(get_settings())


def _search(client, query: RetrievalQuery) -> List[RetrievedChunk]:
    return asyncio.run(client.search(query))


@pytest.mark.parametrize("collection", sorted(REQUIRED_METADATA))
def test_collection_returns_documented_metadata(client, collection):
    chunks = _search(client, RetrievalQuery(collection=collection, text="backend engineer", top_k=5))
    assert chunks, f"collection {collection} returned nothing"
    for chunk in chunks:
        assert chunk.collection == collection
        assert chunk.chunk_id and chunk.content
        missing = REQUIRED_METADATA[collection] - set(chunk.metadata)
        assert not missing, f"{chunk.chunk_id} missing {missing}"


def test_rubric_metadata_types(client):
    for chunk in _search(client, RetrievalQuery(collection="rubric", text="system design", top_k=20)):
        meta = chunk.metadata
        assert isinstance(meta["weight"], (int, float)) and meta["weight"] > 0
        assert isinstance(meta["must_have"], bool)
        assert 1 <= int(meta["pass_level"]) <= 4
        assert set(meta["anchors"]) == {"1", "2", "3", "4"}


def test_top_k_and_descending_scores(client):
    chunks = _search(client, RetrievalQuery(collection="question_bank", text="cache distributed lock", top_k=3))
    assert 0 < len(chunks) <= 3
    scores = [c.score for c in chunks]
    assert scores == sorted(scores, reverse=True)


def test_scalar_filter_matches_list_membership_case_insensitively(client):
    sample = _search(client, RetrievalQuery(collection="question_bank", text="backend engineer", top_k=1))
    assert sample
    level = sample[0].metadata["levels"][0]
    chunks = _search(
        client,
        RetrievalQuery(collection="question_bank", text="design", filters={"levels": level.upper()}, top_k=20),
    )
    assert chunks
    assert all(level in [str(v).lower() for v in c.metadata["levels"]] for c in chunks)


def test_list_filter_matches_any_element(client):
    chunks = _search(
        client,
        RetrievalQuery(collection="rubric", text="", filters={"role_family": ["backend", "any"]}, top_k=50),
    )
    assert chunks
    assert {c.metadata["role_family"] for c in chunks} <= {"backend", "any"}


def test_unknown_filter_key_matches_nothing(client):
    chunks = _search(
        client,
        RetrievalQuery(collection="rubric", text="design", filters={"no_such_field": "x"}, top_k=5),
    )
    assert chunks == []


def test_knowledge_pack_is_usable_for_grounding(client):
    service = InterviewKnowledgeService(client=client)
    pack = asyncio.run(service.build_pack(target_company="Acme Systems", target_role="Junior Backend Engineer"))
    competencies = pack["rubric"]["competencies"]
    assert competencies
    assert abs(sum(c["weight"] for c in competencies) - 1.0) < 1e-2
    assert all(pack["question_bank"].get(c["competency_id"]) for c in competencies)
    plan = ground_stage_plan(
        {"stages": [{"stage_id": "s1", "name": "Technical"}, {"stage_id": "s2", "name": "Wrap-up"}]}, pack
    )
    covered = {cid for s in plan["stages"] for cid in s["criteria"]}
    assert covered == {c["competency_id"] for c in competencies}


def test_backend_errors_surface_as_retrieval_error():
    class Broken:
        backend_name = "broken"

        async def search(self, query):
            raise ConnectionError("down")

    service = InterviewKnowledgeService(client=Broken(), timeout_seconds=0.5)
    with pytest.raises(RetrievalError):
        asyncio.run(service.build_pack(target_company="Acme", target_role="Backend Engineer"))

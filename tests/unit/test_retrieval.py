"""Unit tests for the retrieval port, mock adapter, knowledge service and stage plan grounding."""

import asyncio
from typing import List

import pytest

from config.settings import Settings
from core.retrieval import (
    InterviewKnowledgeService,
    MockRetrievalClient,
    RetrievalError,
    RetrievalQuery,
    RetrievedChunk,
    classify_stage_type,
    ground_stage_plan,
    infer_level,
    infer_role_family,
)
from core.retrieval.factory import build_knowledge_service, build_retrieval_client

TINY_CORPUS = [
    {
        "chunk_id": "a",
        "collection": "question_bank",
        "content": "How do you prevent a cache stampede on a hot key?",
        "metadata": {"competency": "dist", "role_family": "backend", "levels": ["mid", "senior"]},
    },
    {
        "chunk_id": "b",
        "collection": "question_bank",
        "content": "Explain database indexing strategies.",
        "metadata": {"competency": "data", "role_family": "any", "levels": ["junior", "mid"]},
    },
    {
        "chunk_id": "c",
        "collection": "rubric",
        "content": "Cache design competency.",
        "metadata": {"competency": "dist", "role_family": "backend"},
    },
]


@pytest.mark.asyncio
async def test_mock_search_scopes_collection_and_ranks_by_overlap():
    client = MockRetrievalClient(corpus=TINY_CORPUS)
    results = await client.search(
        RetrievalQuery(collection="question_bank", text="cache stampede", top_k=5)
    )
    assert [r.chunk_id for r in results] == ["a", "b"]
    assert results[0].score > results[1].score == 0.0
    assert all(r.collection == "question_bank" for r in results)


@pytest.mark.asyncio
async def test_mock_search_filter_semantics():
    client = MockRetrievalClient(corpus=TINY_CORPUS)

    # Scalar filter matches list metadata by membership, case-insensitively
    senior = await client.search(
        RetrievalQuery(collection="question_bank", text="", filters={"levels": "SENIOR"})
    )
    assert [r.chunk_id for r in senior] == ["a"]

    # List filter matches if any element matches
    family = await client.search(
        RetrievalQuery(
            collection="question_bank", text="", filters={"role_family": ["frontend", "any"]}
        )
    )
    assert [r.chunk_id for r in family] == ["b"]

    # Missing metadata key never matches
    missing = await client.search(
        RetrievalQuery(collection="rubric", text="", filters={"levels": "mid"})
    )
    assert missing == []


@pytest.mark.asyncio
async def test_mock_search_respects_top_k_and_records_queries():
    client = MockRetrievalClient(corpus=TINY_CORPUS)
    query = RetrievalQuery(collection="question_bank", text="cache", top_k=1)
    results = await client.search(query)
    assert len(results) == 1
    assert client.queries == [query]


@pytest.mark.asyncio
async def test_mock_failure_injection_raises_retrieval_error():
    client = MockRetrievalClient(corpus=TINY_CORPUS, fail_with=RuntimeError("index offline"))
    with pytest.raises(RetrievalError, match="index offline"):
        await client.search(RetrievalQuery(collection="rubric", text="x"))


def test_default_corpus_loads_every_collection():
    client = MockRetrievalClient()
    collections = {chunk["collection"] for chunk in client._chunks}
    assert collections == {"question_bank", "rubric", "role_profile", "company_knowledge"}


@pytest.mark.parametrize(
    "role, years, expected",
    [
        ("Senior Backend Engineer", None, "senior"),
        ("Staff Engineer", 3, "staff"),
        ("Junior Frontend Developer", 10, "junior"),
        ("Software Engineer", 1, "junior"),
        ("Software Engineer", 6, "senior"),
        ("Software Engineer", None, "mid"),
    ],
)
def test_infer_level(role, years, expected):
    assert infer_level(role, years) == expected


@pytest.mark.parametrize(
    "role, skills, expected",
    [
        ("Senior Frontend Engineer", ["Go"], "frontend"),
        ("Backend Architect", ["React"], "backend"),
        ("Software Engineer", ["React", "TypeScript"], "frontend"),
        ("Software Engineer", [], "backend"),
    ],
)
def test_infer_role_family(role, skills, expected):
    assert infer_role_family(role, skills) == expected


@pytest.mark.parametrize(
    "stage, expected",
    [
        ({"stage_id": "warmup", "name": "Warmup"}, "warmup"),
        ({"stage_id": "s2", "name": "Technical Depth"}, "technical"),
        ({"stage_id": "system_design", "name": "System Design"}, "system_design"),
        ({"stage_id": "s3", "name": "Behavioral", "goals": ["STAR conflict scenarios"]}, "behavioral"),
        ({"stage_id": "qa_close", "name": "Q&A and Close"}, "wrapup"),
    ],
)
def test_classify_stage_type(stage, expected):
    assert classify_stage_type(stage) == expected


@pytest.mark.asyncio
async def test_build_pack_backend_senior():
    service = InterviewKnowledgeService(client=MockRetrievalClient())
    pack = await service.build_pack(
        target_company="Acme Systems",
        target_role="Senior Backend Engineer",
        skills=["Python", "Redis"],
    )

    assert pack["retrieval_backend"] == "mock"
    assert (pack["role_family"], pack["level"]) == ("backend", "senior")

    rubric = pack["rubric"]
    assert rubric["rubric_version"] == "backend_senior_v1"
    assert rubric["pass_threshold"] == 3.0
    ids = [c["competency_id"] for c in rubric["competencies"]]
    assert ids == pack["role_profile"]["competencies"]
    assert abs(sum(c["weight"] for c in rubric["competencies"]) - 1.0) < 1e-3
    assert all(set(c["anchors"]) == {"1", "2", "3", "4"} for c in rubric["competencies"])

    for cid in ids:
        questions = pack["question_bank"][cid]
        assert questions, cid
        assert all(q["competency_id"] == cid and q["expected_points"] for q in questions)

    topics = {(c["topic"], c["is_generic"]) for c in pack["company_context"]}
    assert ("tech_stack", False) in topics
    assert pack["sources"]


@pytest.mark.asyncio
async def test_build_pack_filters_by_level():
    service = InterviewKnowledgeService(client=MockRetrievalClient())
    pack = await service.build_pack(target_company="TechCorp", target_role="Junior Backend Engineer")

    ids = [c["competency_id"] for c in pack["rubric"]["competencies"]]
    assert "be_system_design" not in ids
    all_questions = [q["question_id"] for qs in pack["question_bank"].values() for q in qs]
    assert "q_be_dist_locks" not in all_questions  # senior/staff only


@pytest.mark.asyncio
async def test_build_pack_times_out_as_retrieval_error():
    service = InterviewKnowledgeService(
        client=MockRetrievalClient(latency_seconds=0.2), timeout_seconds=0.01
    )
    with pytest.raises(RetrievalError, match="timed out"):
        await service.build_pack(target_company="Acme Systems", target_role="Backend Engineer")


@pytest.mark.asyncio
async def test_build_pack_wraps_unexpected_adapter_errors():
    class BrokenClient:
        backend_name = "broken"

        async def search(self, query: RetrievalQuery) -> List[RetrievedChunk]:
            raise ConnectionError("connection refused with api_key=supersecretvalue123")

    service = InterviewKnowledgeService(client=BrokenClient())
    with pytest.raises(RetrievalError) as exc_info:
        await service.build_pack(target_company="Acme", target_role="Backend Engineer")
    assert "supersecretvalue123" not in str(exc_info.value)


@pytest.fixture
def senior_backend_pack():
    service = InterviewKnowledgeService(client=MockRetrievalClient())
    return asyncio.run(
        service.build_pack(target_company="Acme Systems", target_role="Senior Backend Engineer")
    )


def test_ground_stage_plan_assigns_criteria_and_covers_every_competency(senior_backend_pack):
    plan = {
        "stages": [
            {"stage_id": "warmup", "name": "Warmup", "target_turns": 2},
            {"stage_id": "technical", "name": "Technical Depth", "target_turns": 4},
            {"stage_id": "qa_close", "name": "Q&A and Close", "target_turns": 1},
        ]
    }
    grounded = ground_stage_plan(plan, senior_backend_pack)

    stages = {s["stage_id"]: s for s in grounded["stages"]}
    assert stages["warmup"]["stage_type"] == "warmup"
    assert stages["warmup"]["criteria"] == ["beh_communication"]
    assert stages["qa_close"]["stage_type"] == "wrapup"

    rubric_ids = {c["competency_id"] for c in senior_backend_pack["rubric"]["competencies"]}
    covered = {cid for s in grounded["stages"] for cid in s["criteria"]}
    assert covered == rubric_ids

    # Turn budgets and ids are untouched; expected answers never reach the interviewer plan
    assert [s["target_turns"] for s in grounded["stages"]] == [2, 4, 1]
    question_ids = []
    for stage in grounded["stages"]:
        assert len(stage["criteria_labels"]) == len(stage["criteria"])
        assert 0 < len(stage["question_candidates"]) <= 3
        for question in stage["question_candidates"]:
            assert "expected_points" not in question
            question_ids.append(question["question_id"])
    assert len(question_ids) == len(set(question_ids))

    assert all(q["question_id"].startswith("q_warmup") for q in stages["warmup"]["question_candidates"])
    assert grounded["rubric_version"] == "backend_senior_v1"


def test_ground_stage_plan_keeps_valid_llm_criteria(senior_backend_pack):
    plan = {
        "stages": [
            {
                "stage_id": "s1",
                "name": "Deep Dive",
                "stage_type": "system_design",
                "criteria": ["be_system_design", "made_up_competency"],
            }
        ]
    }
    grounded = ground_stage_plan(plan, senior_backend_pack)
    stage = grounded["stages"][0]
    assert stage["stage_type"] == "system_design"
    assert stage["criteria"][0] == "be_system_design"
    assert "made_up_competency" not in stage["criteria"]


def test_ground_stage_plan_returns_nonconforming_input_unchanged(senior_backend_pack):
    assert ground_stage_plan(["not", "a", "plan"], senior_backend_pack) == ["not", "a", "plan"]
    no_stages = {"phases": []}
    assert ground_stage_plan(no_stages, senior_backend_pack) is no_stages
    plan = {"stages": [{"stage_id": "s1"}]}
    assert ground_stage_plan(plan, {"rubric": {"competencies": []}}) is plan


def test_factory_builds_mock_and_rejects_unknown_backend():
    service = build_knowledge_service(
        Settings(RETRIEVAL_BACKEND="mock", RETRIEVAL_TIMEOUT_SECONDS=1.5)
    )
    assert service.backend_name == "mock"
    assert service.timeout_seconds == 1.5
    with pytest.raises(ValueError, match="Unsupported RETRIEVAL_BACKEND"):
        build_retrieval_client(Settings(RETRIEVAL_BACKEND="vector_db"))


@pytest.mark.asyncio
async def test_retrieve_tool_enforces_allowlist_bounds_and_hides_expected_answers():
    from core.retrieval.tool import RetrieveTool

    tool = RetrieveTool(MockRetrievalClient(), max_top_k=2, fixed_filters={"levels": "senior"})
    denied = await tool.execute({"collection": "role_profile", "query": "backend"})
    assert denied["status"] == "denied"

    valid, violations = tool.validate_arguments({"collection": "question_bank", "query": "cache", "top_k": 9})
    assert not valid and violations

    result = await tool.execute({"collection": "question_bank", "query": "cache stampede", "top_k": 2})
    assert result["status"] == "ok" and 0 < len(result["results"]) <= 2
    for item in result["results"]:
        assert "expected_points" not in item["metadata"]

    slow = RetrieveTool(MockRetrievalClient(latency_seconds=0.2), deadline_seconds=0.01)
    assert (await slow.execute({"collection": "rubric", "query": "design"}))["status"] == "error"

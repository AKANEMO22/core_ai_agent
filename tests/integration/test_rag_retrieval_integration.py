"""Exercise the in-process RAG adapter against a small, offline corpus."""

import asyncio
import json

from config.settings import Settings
from core.retrieval.factory import build_knowledge_service
from core.retrieval.port import RetrievalError, RetrievalQuery
from rag.adapter import RAGRetrievalClient
from rag.ontology import resolve_role
from rag.schemas import RAGInterviewRequest


def _framework():
    return {
        "position_id": "IF_BE_GEN",
        "role_title": "Backend Engineer",
        "major_category": "Software Engineering",
        "role_family": "Software Engineering",
        "target_levels": ["Intern", "Fresher", "Junior"],
        "target_competencies": ["Python API design", "Database indexing"],
        "vn_recruitment_process": "Technical interview covering APIs and databases.",
        "frequently_asked_by": ["Example Corp"],
        "interview_stages": [
            {"stage_name": "Project screening", "weight_percent": 20},
            {"stage_name": "Technical interview", "weight_percent": 60},
            {"stage_name": "Behavioral interview", "weight_percent": 20},
        ],
        "evaluation_matrix": [
            {
                "pillar": "Technical Competency",
                "pillar_en": "Technical Competency & CS Fundamentals",
                "weight_percent": 60,
                "criteria": "Explain API design, Python, and database fundamentals.",
            },
            {
                "pillar": "Communication",
                "pillar_en": "Communication & Collaboration",
                "weight_percent": 40,
                "criteria": "Explain decisions clearly and work collaboratively.",
            },
        ],
        "scoring_anchors": {
            "1": "Missing fundamentals",
            "2": "Needs frequent help",
            "3": "Meets the expected standard",
            "4": "Exceeds the expected standard",
            "5": "Exceptional",
        },
        "passing_thresholds": {"Junior": "3.6 / 5.0"},
        "technical_questions": [
            {
                "id": "BE_INT_01",
                "level": "Intern",
                "question": "How would you test a small Python API change?",
                "key_concepts": ["Python", "API", "testing"],
                "expected_answer": "Describe a focused test and the behavior it protects.",
                "scoring_rubric": {
                    "poor": "Names no test or behavior.",
                    "acceptable": "Describes a relevant test.",
                    "excellent": "Connects the test to a failure mode and validation result.",
                },
                "citation_url": "https://example.test/intern-api-test",
            },
            {
                "id": "BE_Q01",
                "level": "Junior",
                "question": "How would you design a Python REST API for a high-volume service?",
                "key_concepts": ["Python", "REST", "API"],
                "expected_answer": "Discuss resource design, validation, status codes, and pagination.",
                "scoring_rubric": {
                    "poor": "Names an endpoint without explaining its behavior.",
                    "acceptable": "Explains resource design and response codes.",
                    "excellent": "Also addresses validation, pagination, and operational tradeoffs.",
                },
                "citation_url": "https://example.test/backend-api",
            },
            {
                "id": "BE_Q02",
                "level": "Fresher",
                "question": "What does a database index improve, and what tradeoff does it add?",
                "key_concepts": ["database", "index"],
                "expected_answer": "An index speeds lookups while adding storage and write overhead.",
                "scoring_rubric": {
                    "poor": "Says an index always makes a database faster.",
                    "acceptable": "Explains lookup improvement and storage cost.",
                    "excellent": "Also discusses write cost and index selection.",
                },
                "citation_url": "https://example.test/database-index",
            },
        ],
        "behavioral_questions": [
            {
                "id": "STAR_01",
                "question": "Tell me about a time you resolved a disagreement with a teammate.",
                "evaluation_focus": "Clear communication, listening, and collaborative resolution.",
                "star_criteria": {
                    "situation_task": "Sets the context and responsibility.",
                    "action": "Describes constructive communication and actions.",
                    "result": "Explains the outcome and learning.",
                },
            }
        ],
    }


def _client(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "interview_frameworks.json").write_text(
        json.dumps([_framework()], ensure_ascii=False), encoding="utf-8"
    )
    # These local files must not be loaded into the agent RAG adapter.
    (corpus / "candidates.json").write_text(
        json.dumps([{"candidate_id": "private-1", "email": "private@example.test"}]),
        encoding="utf-8",
    )
    (corpus / "jobs_below_mid.json").write_text(
        json.dumps([{"job_id": "private-job", "title": "Private employer"}]),
        encoding="utf-8",
    )
    return RAGRetrievalClient(
        data_dir=corpus,
        index_path=tmp_path / "index" / "rag.db",
        embedding_provider="lsa",
    )


def _search(client, query):
    return asyncio.run(client.search(query))


def test_role_resolution_prefers_general_swe_title_before_language_skill_for_intern():
    assert resolve_role(
        "Software Engineer Intern Python FastAPI Redis backend APIs",
        {"IF_SWE", "IF_BE_GEN"},
    ) == "IF_SWE"


def test_rag_adapter_serves_the_core_collections_and_builds_an_offline_index(tmp_path):
    client = _client(tmp_path)

    profile = _search(
        client,
        RetrievalQuery(
            collection="role_profile",
            text="Junior Backend Engineer Python",
            filters={"role_family": "backend", "level": "junior"},
            top_k=1,
        ),
    )
    assert len(profile) == 1
    assert profile[0].metadata["role_family"] == "backend"
    assert profile[0].metadata["level"] == "junior"
    assert profile[0].metadata["competencies"]
    assert profile[0].source.startswith("interview_frameworks.json#IF_BE_GEN")

    rubrics = _search(
        client,
        RetrievalQuery(
            collection="rubric",
            text="Python API communication",
            filters={"role_family": ["backend", "any"], "levels": "junior"},
            top_k=10,
        ),
    )
    assert len(rubrics) == 2
    assert all(
        {"competency", "name", "role_family", "levels", "weight", "must_have", "pass_level", "anchors"}
        <= chunk.metadata.keys()
        for chunk in rubrics
    )
    assert all(set(chunk.metadata["anchors"]) == {"1", "2", "3", "4"} for chunk in rubrics)

    questions = _search(
        client,
        RetrievalQuery(
            collection="question_bank",
            text="Python REST API design",
            filters={"role_family": "backend", "levels": "junior"},
            top_k=2,
        ),
    )
    assert questions
    assert questions[0].metadata["question_id"] == "IF_BE_GEN_BE_Q01"
    assert questions[0].metadata["expected_points"]
    assert questions[0].metadata["levels"] == ["junior"]
    assert all(questions[i].score >= questions[i + 1].score for i in range(len(questions) - 1))

    company = _search(
        client,
        RetrievalQuery(
            collection="company_knowledge",
            text="Example Corp backend interview",
            filters={"company": ["example corp", "generic"]},
            top_k=5,
        ),
    )
    assert company
    assert all(chunk.metadata["company"] == "generic" for chunk in company)
    assert all("Example Corp" not in chunk.content for chunk in company)

    assert client.backend_name == "rag"
    assert client.engine.candidates == {}
    assert client.engine.jobs == {}
    assert client.index_audit["provider"].startswith("lsa:")

    context = client.engine.retrieve(
        RAGInterviewRequest(
            role_id="IF_BE_GEN",
            target_level="Junior",
            query="Python REST API design",
            required_skills=["Python"],
            top_k=2,
            candidate_pool=10,
        ),
        persist=False,
    )
    assert context.status == "ok"
    assert context.selected_questions


def test_rag_knowledge_pack_maps_intern_source_level_to_junior_core_rubric(tmp_path):
    service = build_knowledge_service(
        Settings(RETRIEVAL_BACKEND="mock"),
        client=_client(tmp_path),
    )
    pack = asyncio.run(service.build_pack(
        target_company="Example Corp",
        target_role="Software Engineer Intern",
        skills=["Python"],
    ))

    assert pack["level"] == "junior"
    assert pack["source_level"] == "intern"
    assert pack["rubric"]["competencies"]
    assert any(pack["question_bank"].values())


def test_rag_adapter_does_not_map_unsupported_levels_to_junior(tmp_path):
    client = _client(tmp_path)
    chunks = _search(
        client,
        RetrievalQuery(
            collection="question_bank",
            text="Senior Backend Engineer Python API",
            filters={"role_family": "backend", "levels": "senior"},
            top_k=5,
        ),
    )
    assert chunks == []


def test_unknown_filters_do_not_leak_framework_chunks(tmp_path):
    client = _client(tmp_path)
    chunks = _search(
        client,
        RetrievalQuery(
            collection="rubric",
            text="Python",
            filters={"tenant_id": "unknown"},
            top_k=5,
        ),
    )
    assert chunks == []


def test_factory_builds_the_agent_knowledge_pack_from_rag(tmp_path):
    client = _client(tmp_path)
    service = build_knowledge_service(
        Settings(
            RETRIEVAL_BACKEND="rag",
            RAG_DATA_DIR=client.data_dir,
            RAG_INDEX_PATH=client.store.path,
            RAG_EMBEDDING_PROVIDER="lsa",
        )
    )

    pack = asyncio.run(
        service.build_pack(
            target_company="Example Corp",
            target_role="Junior Backend Engineer",
            skills=["Python", "REST API"],
        )
    )

    assert service.backend_name == "rag"
    assert pack["retrieval_backend"] == "rag"
    assert pack["rubric"]["competencies"]
    assert all(
        pack["question_bank"].get(competency["competency_id"])
        for competency in pack["rubric"]["competencies"]
    )
    assert pack["company_context"]
    assert all(item["is_generic"] for item in pack["company_context"])

    try:
        asyncio.run(
            service.build_pack(
                target_company="Example Corp",
                target_role="Senior Backend Engineer",
            )
        )
    except RetrievalError as error:
        assert "level='senior'" in str(error)
    else:
        raise AssertionError("Unsupported senior evidence must be reported explicitly")

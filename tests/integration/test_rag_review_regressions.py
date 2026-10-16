"""Semantic acceptance against the shipped corpus, without external providers."""
import asyncio
from pathlib import Path

import pytest

from core.retrieval.knowledge import InterviewKnowledgeService
from core.retrieval.port import RetrievalQuery
from rag.adapter import RAGRetrievalClient


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    return RAGRetrievalClient(Path("data"), tmp_path_factory.mktemp("semantic-rag") / "index.db")


def test_technical_questions_do_not_inherit_shared_screening_stage(corpus):
    framework = corpus.frameworks["IF_SWE"]
    technical = [q for q in corpus.questions.values() if q["role_id"] == "IF_SWE" and q["question_type"] != "behavioral"]
    assert technical
    assert all(corpus._stage_type(q, framework) == "technical" for q in technical)


def test_competency_query_cannot_relabel_hashing_or_recursion(corpus):
    comp = next(c for c in corpus._competencies_by_role["IF_SWE"] if "Communication" in c.get("pillar_en", ""))
    chunks = asyncio.run(corpus.search(RetrievalQuery(
        collection="question_bank", text="Software Engineer Intern",
        filters={"competency": comp["competency_id"], "levels": "intern", "role_family": "backend"}, top_k=50,
    )))
    assert not {"IF_SWE_SWE_Q03", "IF_SWE_SWE_Q05"}.intersection(c.chunk_id for c in chunks)
    assert all(c.metadata["competency_match"] != "query_relevance" for c in chunks)


@pytest.mark.parametrize("level", ["intern", "fresher", "junior", "mid", "senior"])
def test_all_framework_titles_have_truthfully_calibrated_probes(corpus, level):
    service = InterviewKnowledgeService(corpus, timeout_seconds=20)
    for role_id, framework in corpus.frameworks.items():
        pack = asyncio.run(service.build_pack("Example", f"{level.title()} {framework['role_title']}", seniority=level))
        assert pack["rubric"]["competencies"], (role_id, level)
        assert role_id in pack["role_profile"]["source"], (role_id, level, pack["role_profile"])
        for competency in pack["rubric"]["competencies"]:
            questions = pack["question_bank"][competency["competency_id"]]
            assert questions, (role_id, level, competency)
            for question in questions:
                assert question["calibration_level"] == level
                assert question["selected_role"] == framework["role_title"]
                assert question["expected_points"]
                if question["provenance"] == "project_authored":
                    assert question["trust"] == "project_authored_unreviewed"
                    assert "authored" in question["source"]
                else:
                    assert level not in {"mid", "senior"}
                    assert level.title() in question["source_levels"]

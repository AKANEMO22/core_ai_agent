"""Regression tests for RAG adapter level filtering, provenance and event-loop behavior."""
import asyncio
import json
import time
from pathlib import Path

import pytest

import rag.adapter as adapter_module
from core.retrieval.port import RetrievalQuery
from rag.adapter import RAGRetrievalClient
from rag.indexing import build_index
from rag.scaffolds import RUBRIC_VERSION, VERSION, authored_probe
from rag.storage import RAGStore


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    return RAGRetrievalClient(Path("data"), tmp_path_factory.mktemp("adapter-rag") / "index.db")


def _search(client, **kwargs):
    return asyncio.run(client.search(RetrievalQuery(**kwargs)))


def _android_text(corpus):
    return corpus.frameworks["IF_MOBILE_ANDROID"]["role_title"]


def test_search_runs_off_the_event_loop_so_deadlines_fire(corpus, monkeypatch):
    monkeypatch.setattr(corpus, "_search_sync", lambda query: time.sleep(0.5) or [])

    async def run():
        started = time.perf_counter()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                corpus.search(RetrievalQuery(collection="rubric", text="x", top_k=1)), timeout=0.05
            )
        return time.perf_counter() - started

    assert asyncio.run(run()) < 0.4


def test_rubric_without_source_questions_is_not_labelled_as_source_level(corpus):
    chunks = _search(
        corpus, collection="rubric", text=_android_text(corpus),
        filters={"role_family": "mobile", "levels": "intern"}, top_k=20,
    )
    android = [c for c in chunks if "IF_MOBILE_ANDROID" in c.source]
    assert android
    for chunk in android:
        assert chunk.metadata["source_levels"] == []
        assert chunk.metadata["provenance"] == "project_authored"
        assert chunk.metadata["calibration_level"] == "intern"


def test_rubric_with_source_questions_reports_indexed_provenance(corpus):
    chunks = _search(
        corpus, collection="rubric", text=_android_text(corpus),
        filters={"role_family": "mobile", "levels": "fresher"}, top_k=20,
    )
    assert chunks
    assert all(c.metadata["source_levels"] == ["Fresher"] for c in chunks)
    assert all(c.metadata["provenance"] == "indexed_source" for c in chunks)


@pytest.mark.parametrize("collection", ["rubric", "question_bank", "role_profile"])
@pytest.mark.parametrize("level", ["principal", "staff"])
def test_unknown_or_unsupported_levels_return_nothing(corpus, collection, level):
    assert _search(
        corpus, collection=collection, text=_android_text(corpus),
        filters={"role_family": "mobile", "levels": level}, top_k=10,
    ) == []
    assert _search(
        corpus, collection=collection, text=_android_text(corpus),
        filters={"role_family": "mobile", "level": level}, top_k=10,
    ) == []


def test_singular_level_filter_matches_like_levels(corpus):
    singular = _search(
        corpus, collection="question_bank", text="Software Engineer Python",
        filters={"role_family": "backend", "level": "junior"}, top_k=5,
    )
    plural = _search(
        corpus, collection="question_bank", text="Software Engineer Python",
        filters={"role_family": "backend", "levels": "junior"}, top_k=5,
    )
    assert singular
    assert [c.chunk_id for c in singular] == [c.chunk_id for c in plural]


def test_multi_level_question_filter_serves_each_requested_level(corpus):
    chunks = _search(
        corpus, collection="question_bank", text=_android_text(corpus),
        filters={"role_family": "mobile", "levels": ["mid", "senior"]}, top_k=200,
    )
    assert {c.metadata["calibration_level"] for c in chunks} == {"mid", "senior"}
    assert all(c.metadata["provenance"] == "project_authored" for c in chunks)


def test_competency_filter_is_case_insensitive_for_authored_questions(corpus):
    pillar = corpus._competencies_by_role["IF_MOBILE_ANDROID"][0]
    competency = f"{pillar['competency_id']}:mid".upper()
    chunks = _search(
        corpus, collection="question_bank", text=_android_text(corpus),
        filters={"role_family": "mobile", "levels": "mid", "competency": competency}, top_k=5,
    )
    assert chunks
    assert all(c.metadata["competency"].casefold() == competency.casefold() for c in chunks)


def test_staff_role_profile_is_not_relabelled_junior(corpus):
    assert _search(
        corpus, collection="role_profile", text=_android_text(corpus),
        filters={"role_family": "mobile", "level": "staff"}, top_k=1,
    ) == []


def test_empty_query_text_still_returns_eligible_questions(corpus):
    chunks = _search(
        corpus, collection="question_bank", text="",
        filters={"role_family": "backend", "levels": "junior"}, top_k=3,
    )
    assert len(chunks) == 3
    again = _search(
        corpus, collection="question_bank", text="",
        filters={"role_family": "backend", "levels": "junior"}, top_k=3,
    )
    assert [c.chunk_id for c in chunks] == [c.chunk_id for c in again]


def test_dense_failure_is_logged_and_flagged(corpus, monkeypatch, caplog):
    def broken(*args, **kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(adapter_module, "dense_retrieve", broken)
    with caplog.at_level("WARNING", logger="rag.adapter"):
        chunks = _search(
            corpus, collection="question_bank", text="Python REST API",
            filters={"role_family": "backend", "levels": "junior"}, top_k=2,
        )
    assert chunks
    assert all("dense_unavailable:RuntimeError" in c.metadata["retrieval_degraded"] for c in chunks)
    assert "Dense retrieval unavailable" in caplog.text


def test_authored_role_profile_version_tracks_scaffold_content(corpus):
    profile = _search(
        corpus, collection="role_profile", text=_android_text(corpus),
        filters={"role_family": "mobile", "level": "senior"}, top_k=1,
    )
    assert profile[0].metadata["rubric_version"] == RUBRIC_VERSION
    assert RUBRIC_VERSION.startswith(VERSION + ":") and len(RUBRIC_VERSION) > len(VERSION) + 1


def test_authored_probe_falls_back_for_unmapped_family():
    framework = {"role_title": "Generalist", "target_competencies": []}
    probe = authored_probe("X", framework, {"pillar": "Problem solving", "competency_id": "c1"}, "any", "junior")
    assert "Generalist" in probe["text"]


def test_engine_is_pinned_to_the_index_it_just_built(tmp_path, monkeypatch):
    frameworks = json.loads(Path("data/interview_frameworks.json").read_text(encoding="utf-8"))
    other = tmp_path / "other"
    other.mkdir()
    (other / "interview_frameworks.json").write_text(json.dumps(frameworks[:1]), encoding="utf-8")
    store_path = tmp_path / "index.db"
    other_version = build_index(RAGStore(store_path), other)["index_version"]

    real_build = adapter_module.build_index

    def build_then_switch(store, *args, **kwargs):
        audit = real_build(store, *args, **kwargs)
        store.activate(other_version)  # A concurrent writer activates a different index.
        return audit

    monkeypatch.setattr(adapter_module, "build_index", build_then_switch)
    client = RAGRetrievalClient(Path("data"), store_path)
    assert client.engine.version == client.index_audit["index_version"] != other_version

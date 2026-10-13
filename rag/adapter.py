"""Adapter from the interview RAG index to the agent's retrieval port."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from core.retrieval.port import (
    RetrievalCollection,
    RetrievalError,
    RetrievalQuery,
    RetrievedChunk,
)

from .engine import RAGEngine
from .indexing import build_index, digest
from .ontology import resolve_role, tokens
from .retrievers import bm25_retrieve, dense_retrieve, fuse
from .storage import RAGStore

CORE_LEVELS = {"junior", "mid", "senior", "staff"}
SOURCE_LEVELS = {"intern", "fresher", "junior"}

ROLE_ID_FAMILIES = {
    "IF_FE": "frontend",
    "IF_FULLSTACK": "backend",
    "IF_AI": "data",
    "IF_AI_SWE": "data",
    "IF_AGENTIC_AI": "data",
    "IF_DA": "data",
    "IF_DE": "data",
    "IF_DS": "data",
    "IF_QA_QC": "qa",
    "IF_AUTO_QA": "qa",
    "IF_DEVOPS": "devops",
    "IF_MOBILE_ANDROID": "mobile",
    "IF_MOBILE_IOS": "mobile",
    "IF_EMBEDDED": "embedded",
    "IF_NETWORK": "infrastructure",
    "IF_UNITY": "game",
    "IF_BA": "business_analyst",
    "IF_ERP": "enterprise",
    "IF_PRODUCT_ENG": "backend",
    "IF_BE_JAVA": "backend",
    "IF_BE_NET": "backend",
    "IF_BE_PHP": "backend",
    "IF_BE_GEN": "backend",
    "IF_SWE": "backend",
    "IF_SWD": "backend",
}

CATEGORY_FAMILIES = {
    "software engineering": "backend",
    "data & ai": "data",
    "quality assurance": "qa",
    "infrastructure & cloud": "devops",
    "mobile development": "mobile",
    "hardware & embedded": "embedded",
    "business analysis & product": "business_analyst",
}

ANCHOR_DEFAULTS = {
    "1": "Insufficient evidence for the stated criterion.",
    "2": "Partial evidence with important gaps.",
    "3": "Meets the expected standard.",
    "4": "Exceeds the expected standard with clear evidence.",
}


class RAGRetrievalClient:
    """Serve core retrieval collections from the local framework RAG index.

    Candidate and job lookup files are deliberately disabled here. The agent path
    receives role/skill/JD text through core queries and indexes only the shared
    interview framework corpus.
    """

    backend_name = "rag"

    def __init__(
        self,
        data_dir: str | Path,
        index_path: str | Path,
        embedding_provider: str = "lsa",
        *,
        engine: RAGEngine | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.store = RAGStore(index_path)
        if engine is None:
            self.index_audit = build_index(
                self.store,
                self.data_dir,
                provider_kind=embedding_provider,
            )
            self.engine = RAGEngine(
                self.store,
                self.data_dir,
                include_local_profiles=False,
            )
        else:
            self.engine = engine
            self.index_audit = dict(engine.payload.get("audit", {}))
        self.frameworks = self.engine.payload["frameworks"]
        self.questions = self.engine.questions
        self.texts = self.engine.texts
        self._competencies: dict[str, tuple[str, dict[str, Any]]] = {}
        self._competencies_by_role: dict[str, list[dict[str, Any]]] = {}
        for role_id, framework in self.frameworks.items():
            entries = []
            for pillar in framework.get("evaluation_matrix", []):
                competency_id = "competency:" + digest(
                    [role_id, pillar["pillar"]]
                )[:20]
                entry = {**pillar, "competency_id": competency_id}
                entries.append(entry)
                self._competencies[competency_id] = (role_id, entry)
            self._competencies_by_role[role_id] = entries

    @property
    def version(self) -> str:
        return self.engine.version

    async def search(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        collection = str(getattr(query.collection, "value", query.collection))
        if collection not in {item.value for item in RetrievalCollection}:
            raise RetrievalError(f"Unsupported retrieval collection '{collection}'")
        if query.top_k <= 0:
            return []
        roles = self._select_roles(query)
        if not roles:
            return []

        if collection == RetrievalCollection.QUESTION_BANK.value:
            return self._search_questions(query, roles)
        if collection == RetrievalCollection.RUBRIC.value:
            return self._search_rubric(query, roles)
        if collection == RetrievalCollection.ROLE_PROFILE.value:
            return self._search_role_profile(query, roles)
        return self._search_company_knowledge(query, roles)

    def _select_roles(self, query: RetrievalQuery) -> list[str]:
        family_filter = self._as_values(query.filters.get("role_family"))
        normalized_families = {
            self._normalize_family(value) for value in family_filter
        }
        allowed = [
            role_id
            for role_id, framework in self.frameworks.items()
            if not family_filter
            or self._role_family(role_id, framework) in normalized_families
        ]
        if not allowed:
            return []

        role_hint = resolve_role(query.text or "", set(self.frameworks))
        if role_hint in allowed:
            return [role_hint]
        query_tokens = set(tokens(query.text or ""))
        if not query_tokens:
            return allowed
        ranked = []
        for role_id in allowed:
            framework = self.frameworks[role_id]
            profile_text = " ".join(
                [
                    framework.get("role_title", ""),
                    framework.get("major_category", ""),
                    *framework.get("target_competencies", []),
                ]
            )
            overlap = len(query_tokens & set(tokens(profile_text)))
            ranked.append((overlap, role_id))
        best_score = max(score for score, _ in ranked)
        if best_score == 0:
            return [min(allowed)]
        return [min(role_id for score, role_id in ranked if score == best_score)]

    @staticmethod
    def _as_values(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return list(value)
        return [value]

    @staticmethod
    def _normalize_family(value: Any) -> str:
        text = str(value).strip().casefold()
        return CATEGORY_FAMILIES.get(text, text)

    @staticmethod
    def _role_family(role_id: str, framework: dict[str, Any]) -> str:
        if role_id in ROLE_ID_FAMILIES:
            return ROLE_ID_FAMILIES[role_id]
        category = str(framework.get("role_family", "")).casefold()
        return CATEGORY_FAMILIES.get(category, "any")

    def _source_levels(self, query: RetrievalQuery, framework: dict[str, Any]) -> list[str]:
        raw = query.filters.get("levels", query.filters.get("level"))
        available = {
            str(question.get("level", "")).casefold()
            for question in self.questions.values()
            if question.get("role_id") == framework["position_id"]
        }
        if raw is None:
            return sorted(level.title() for level in available & SOURCE_LEVELS)

        values = [str(value).strip() for value in self._as_values(raw)]
        supported = [value.casefold() for value in values if value.casefold() in SOURCE_LEVELS]
        if not supported:
            # Core levels beyond Junior have no source evidence in this corpus.
            return []
        exact = next(
            (
                value
                for value in values
                if value in {"Intern", "Fresher", "Junior"}
            ),
            None,
        )
        if exact:
            supported = [exact.casefold()]
        else:
            explicit = re.search(r"\b(intern|fresher|junior)\b", query.text or "", re.I)
            if explicit:
                marker = explicit.group(1).casefold()
                if marker in supported:
                    supported = [marker]
        return sorted(
            level.title()
            for level in available & set(supported)
        )

    @staticmethod
    def _core_level(values: list[Any]) -> str | None:
        if not values:
            return None
        normalized = [str(value).strip().casefold() for value in values]
        if any(value not in CORE_LEVELS for value in normalized):
            return None
        if "junior" not in normalized:
            return None
        return "junior"

    @staticmethod
    def _matches_filters(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
        for key, expected in filters.items():
            if key not in metadata:
                return False
            actual = metadata[key]
            expected_values = RAGRetrievalClient._as_values(expected)
            actual_values = RAGRetrievalClient._as_values(actual)
            if not any(
                str(left).casefold() == str(right).casefold()
                for left in actual_values
                for right in expected_values
            ):
                return False
        return True

    def _search_role_profile(
        self, query: RetrievalQuery, roles: list[str]
    ) -> list[RetrievedChunk]:
        output = []
        requested_level = self._core_level(
            self._as_values(query.filters.get("level"))
        )
        if query.filters.get("level") is not None and requested_level is None:
            return []
        for role_id in roles:
            framework = self.frameworks[role_id]
            family = self._role_family(role_id, framework)
            competencies = self._competencies_by_role[role_id]
            profile_level = requested_level or "junior"
            metadata = {
                "role_family": family,
                "level": profile_level,
                "competencies": [item["competency_id"] for item in competencies],
                "pass_threshold": self._pass_threshold(framework, query.text),
                "rubric_version": f"rag:{self.version}",
            }
            if not self._matches_filters(metadata, query.filters):
                continue
            content = "\n".join(
                [
                    str(framework.get("role_title", "")),
                    str(framework.get("major_category", "")),
                    *framework.get("target_competencies", []),
                ]
            ).strip()
            output.append(
                RetrievedChunk(
                    chunk_id=f"role:{role_id}:{profile_level}",
                    collection=RetrievalCollection.ROLE_PROFILE.value,
                    content=content,
                    score=self._score(query.text, content),
                    metadata=metadata,
                    source=f"interview_frameworks.json#{role_id}/target_competencies",
                )
            )
        return self._rank_chunks(output, query)[: query.top_k]

    def _search_rubric(
        self, query: RetrievalQuery, roles: list[str]
    ) -> list[RetrievedChunk]:
        output = []
        for role_id in roles:
            framework = self.frameworks[role_id]
            source_levels = self._source_levels(query, framework)
            if not source_levels:
                continue
            family = self._role_family(role_id, framework)
            for index, pillar in enumerate(self._competencies_by_role[role_id]):
                metadata = {
                    "competency": pillar["competency_id"],
                    "name": pillar.get("pillar_en") or pillar["pillar"],
                    "role_family": family,
                    "levels": ["junior"],
                    "source_levels": source_levels,
                    "weight": float(pillar.get("weight_percent", 1)),
                    "must_have": False,
                    "pass_level": 3,
                    "anchors": self._anchors(framework),
                }
                if not self._matches_filters(metadata, query.filters):
                    continue
                content = str(pillar.get("criteria", "")).strip()
                if not content:
                    content = str(pillar.get("pillar_en") or pillar["pillar"])
                output.append(
                    RetrievedChunk(
                        chunk_id=pillar["competency_id"],
                        collection=RetrievalCollection.RUBRIC.value,
                        content=content,
                        score=self._score(query.text, content + " " + metadata["name"]),
                        metadata=metadata,
                        source=f"interview_frameworks.json#{role_id}/evaluation_matrix/{index}",
                    )
                )
        return self._rank_chunks(output, query)[: query.top_k]

    def _search_questions(
        self, query: RetrievalQuery, roles: list[str]
    ) -> list[RetrievedChunk]:
        eligible: set[str] = set()
        metadata_by_id: dict[str, dict[str, Any]] = {}
        extra_query = query.text or ""
        requested_competency = query.filters.get("competency")
        if requested_competency is not None:
            competency_id = str(requested_competency)
            competency = self._competencies.get(competency_id)
            if competency is None:
                return []
            role_id, pillar = competency
            if role_id not in roles:
                return []
            extra_query = " ".join(
                [
                    extra_query,
                    str(pillar.get("pillar_en", "")),
                    str(pillar.get("pillar", "")),
                    str(pillar.get("criteria", "")),
                ]
            )

        for question_id, question in self.questions.items():
            role_id = question.get("role_id")
            if role_id not in roles:
                continue
            framework = self.frameworks[role_id]
            source_levels = self._source_levels(query, framework)
            if question.get("level") not in source_levels and not set(
                question.get("applicable_levels", [])
            ).intersection(source_levels):
                continue
            family = self._role_family(role_id, framework)
            target_competency = requested_competency
            if target_competency is None:
                target_competency = self._best_competency(role_id, question)
            metadata = self._question_metadata(
                question_id,
                question,
                framework,
                family,
                target_competency,
                source_levels,
            )
            filters = {
                key: value
                for key, value in query.filters.items()
                if key not in {"competency", "role_family", "levels", "level"}
            }
            if not self._matches_filters(metadata, filters):
                continue
            if requested_competency is not None and not self._matches_filters(
                {"competency": metadata["competency"]},
                {"competency": requested_competency},
            ):
                continue
            eligible.add(question_id)
            metadata_by_id[question_id] = metadata

        if not eligible:
            return []
        pool_size = min(len(eligible), max(query.top_k * 5, 10))
        channels = {
            "bm25": bm25_retrieve(
                extra_query,
                self.questions,
                self.texts,
                eligible,
                pool_size,
            )
        }
        try:
            channels["dense"] = dense_retrieve(
                extra_query,
                self.engine.provider,
                self.engine.vectors,
                self.engine.ids,
                eligible,
                pool_size,
                getattr(self.engine, "dense_min_score", 0.05),
            )
        except (RuntimeError, ValueError, ImportError):
            # BM25 and the graph remain useful if a configured dense provider is unavailable.
            channels["dense"] = []
        try:
            graph_hits, _ = self.engine.graph.retrieve(
                extra_query,
                eligible,
                pool_size,
                max_hops=2,
                timeout=0.05,
            )
            channels["graph"] = graph_hits
        except (RuntimeError, ValueError, KeyError):
            channels["graph"] = []
        ranked = fuse(channels)
        results = []
        for question_id, score in ranked[: query.top_k]:
            question = self.questions[question_id]
            source = (question.get("sources") or [{}])[0]
            results.append(
                RetrievedChunk(
                    chunk_id=question_id,
                    collection=RetrievalCollection.QUESTION_BANK.value,
                    content=str(question.get("question", "")),
                    score=float(score),
                    metadata=metadata_by_id[question_id],
                    source=str(source.get("url") or source.get("locator") or ""),
                )
            )
        return results

    def _question_metadata(
        self,
        question_id: str,
        question: dict[str, Any],
        framework: dict[str, Any],
        family: str,
        competency: str,
        source_levels: list[str],
    ) -> dict[str, Any]:
        level = str(question.get("level", "Fresher"))
        expected = str(question.get("expected_answer", "")).strip()
        return {
            "question_id": question_id,
            "competency": competency,
            "role_family": family,
            "levels": ["junior"],
            "source_levels": list(question.get("applicable_levels") or [level]),
            "stage_type": self._stage_type(question, framework),
            "difficulty": {"Intern": 1, "Fresher": 2, "Junior": 3}.get(level, 2),
            "expected_points": [expected] if expected else [],
            "follow_ups": [],
            "source_concepts": [
                self.engine.payload["concepts"].get(concept, concept)
                for concept in question.get("concepts", [])
            ],
            "competency_match": "query_relevance" if competency else "unmapped",
            "selected_role": framework.get("role_title", ""),
            "source_level_set": source_levels,
        }

    def _best_competency(self, role_id: str, question: dict[str, Any]) -> str:
        competencies = self._competencies_by_role.get(role_id, [])
        if not competencies:
            return "general"
        question_text = " ".join(
            [
                question.get("question", ""),
                question.get("expected_answer", ""),
                *[
                    self.engine.payload["concepts"].get(concept, "")
                    for concept in question.get("concepts", [])
                ],
            ]
        )
        scored = [
            (
                self._score(
                    " ".join(
                        [
                            str(pillar.get("pillar_en", "")),
                            str(pillar.get("pillar", "")),
                            str(pillar.get("criteria", "")),
                        ]
                    ),
                    question_text,
                ),
                pillar["competency_id"],
            )
            for pillar in competencies
        ]
        best_score = max(score for score, _ in scored)
        return min(item for score, item in scored if score == best_score)

    def _search_company_knowledge(
        self, query: RetrievalQuery, roles: list[str]
    ) -> list[RetrievedChunk]:
        output = []
        requested_companies = {
            str(value).strip().casefold()
            for value in self._as_values(query.filters.get("company"))
        }
        if requested_companies and "generic" not in requested_companies:
            return []
        for role_id in roles:
            framework = self.frameworks[role_id]
            content = str(framework.get("vn_recruitment_process") or "").strip()
            if not content:
                continue
            metadata = {
                "company": "generic",
                "topic": "vn_recruitment_process",
                "confidence": 0.5,
                "scope": "generic_framework_context",
            }
            filters = {
                key: value for key, value in query.filters.items() if key != "company"
            }
            if not self._matches_filters(metadata, filters):
                continue
            output.append(
                RetrievedChunk(
                    chunk_id=f"company:generic:{role_id}:recruitment_process",
                    collection=RetrievalCollection.COMPANY_KNOWLEDGE.value,
                    content=content,
                    score=self._score(query.text, content),
                    metadata=metadata,
                    source=f"interview_frameworks.json#{role_id}/vn_recruitment_process",
                )
            )
        return self._rank_chunks(output, query)[: query.top_k]

    def _rank_chunks(
        self, chunks: list[RetrievedChunk], query: RetrievalQuery
    ) -> list[RetrievedChunk]:
        chunks.sort(key=lambda chunk: (-chunk.score, chunk.chunk_id))
        return chunks

    def _score(self, query: str, text: str) -> float:
        query_terms = set(tokens(query or ""))
        if not query_terms:
            return 0.0
        document_terms = set(tokens(text or ""))
        return len(query_terms & document_terms) / len(query_terms)

    def _anchors(self, framework: dict[str, Any]) -> dict[str, str]:
        source = framework.get("scoring_anchors") or {}
        return {
            key: str(source.get(key, ANCHOR_DEFAULTS[key]))
            for key in ("1", "2", "3", "4")
        }

    def _pass_threshold(self, framework: dict[str, Any], query_text: str) -> float:
        source_thresholds = framework.get("passing_thresholds") or {}
        requested = re.search(r"\b(intern|fresher|junior)\b", query_text or "", re.I)
        level = requested.group(1).title() if requested else "Junior"
        raw = str(source_thresholds.get(level, ""))
        match = re.search(r"(\d+(?:\.\d+)?)\s*/\s*5(?:\.0)?", raw)
        if not match:
            return 3.0
        source_score = float(match.group(1))
        return round(1 + (source_score - 1) * 3 / 4, 2)

    def _stage_type(self, question: dict[str, Any], framework: dict[str, Any]) -> str:
        if question.get("question_type") == "behavioral":
            return "behavioral"
        stage_ids = set(question.get("stages", []))
        for stage in framework.get("interview_stages", []):
            if stage.get("stage") in stage_ids:
                name = str(stage.get("stage_name", "")).casefold()
                if re.search(r"system|architecture|design", name):
                    return "system_design"
                if re.search(r"warm|intro|screen|background|cv", name):
                    return "warmup"
                if re.search(r"wrap|closing|candidate question", name):
                    return "wrapup"
        return "technical"

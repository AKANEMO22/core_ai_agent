"""Interview knowledge service composing retrieval results into interview-ready artifacts."""

import asyncio
import re
from typing import Any, Dict, List, Optional, Sequence, Set

from core.retrieval.port import (
    RetrievalClient,
    RetrievalCollection,
    RetrievalError,
    RetrievalQuery,
    RetrievedChunk,
)

STAGE_TYPES = ("warmup", "technical", "system_design", "behavioral", "wrapup")

_LEVEL_PATTERNS = (
    ("staff", re.compile(r"\b(staff|principal|distinguished)\b")),
    ("senior", re.compile(r"\b(senior|sr|lead|architect)\b")),
    ("junior", re.compile(r"\b(junior|jr|intern|fresher|entry|graduate)\b")),
    ("mid", re.compile(r"\b(mid|intermediate)\b")),
)

_ROLE_FAMILY_PATTERNS = (
    ("frontend", re.compile(r"\b(front[\s-]?end|react|vue|angular|ui engineer|web developer)\b")),
    ("backend", re.compile(r"\b(back[\s-]?end|server|api|platform|distributed|microservices?)\b")),
    ("data", re.compile(r"\b(data engineer|data analyst|data scientist|machine learning|\bml\b|\bai\b|analytics|\bdba\b)\b")),
    ("qa", re.compile(r"\b(qa|qc|quality assurance|test engineer|software tester|automation tester)\b")),
    ("devops", re.compile(r"\b(devops|sre|cloud engineer|platform operations)\b")),
    ("mobile", re.compile(r"\b(mobile|android|ios|flutter|kotlin|swift)\b")),
    ("embedded", re.compile(r"\b(embedded|firmware|autosar|microcontroller)\b")),
    ("infrastructure", re.compile(r"\b(network engineer|networking|sysadmin|infrastructure engineer)\b")),
    ("game", re.compile(r"\b(game developer|game engineer|unity developer)\b")),
    ("enterprise", re.compile(r"\b(erp|sap|odoo)\b")),
    ("business_analyst", re.compile(r"\b(business analyst|product analyst|business analysis|business intelligence)\b")),
)

# Ordered: the first matching stage type wins.
_STAGE_TYPE_PATTERNS = (
    ("wrapup", re.compile(r"(wrap|q\s*&\s*a|\bqa\b|\bclos(e|ing)\b|candidate questions|reverse)")),
    ("warmup", re.compile(r"(warm|\bintro|background|\bcv\b|resume|walkthrough|ice[\s-]?break)")),
    ("behavioral", re.compile(r"(behavio|\bstar\b|conflict|culture|leadership|teamwork|collaborat)")),
    ("system_design", re.compile(r"(system[\s_-]?design|architect|\bdesign\b|scalab)")),
)

DEFAULT_ROLE_FAMILY = "backend"
DEFAULT_PASS_THRESHOLD = 3.0
DEFAULT_RUBRIC_VERSION = "rubric_v1"


def infer_level(target_role: Optional[str], experience_years: Optional[int] = None) -> str:
    """Map a free-text role title (and experience) onto the junior/mid/senior/staff ladder."""
    role_text = (target_role or "").lower()
    for level, pattern in _LEVEL_PATTERNS:
        if pattern.search(role_text):
            return level
    if experience_years is None:
        return "mid"
    if experience_years < 2:
        return "junior"
    if experience_years < 5:
        return "mid"
    if experience_years < 9:
        return "senior"
    return "staff"


def infer_role_family(target_role: Optional[str], skills: Optional[Sequence[str]] = None) -> str:
    """Map a role title and skill list onto a role family; the title takes precedence over skills."""
    for text in ((target_role or "").lower(), " ".join(skills or []).lower()):
        for family, pattern in _ROLE_FAMILY_PATTERNS:
            if pattern.search(text):
                return family
    return DEFAULT_ROLE_FAMILY


def classify_stage_type(stage: Dict[str, Any]) -> str:
    """Classify a stage into one of STAGE_TYPES from its id, name and goals."""
    goals = stage.get("goals") or []
    goals_text = " ".join(str(g) for g in goals) if isinstance(goals, list) else str(goals)
    text = f"{stage.get('stage_id', '')} {stage.get('name', '')} {goals_text}".lower()
    for stage_type, pattern in _STAGE_TYPE_PATTERNS:
        if pattern.search(text):
            return stage_type
    return "technical"


class InterviewKnowledgeService:
    """Builds the interview knowledge pack (rubric, question bank, company context) via retrieval.

    Queries carry only role, level, skills, company and job description text; candidate identity
    and raw CV text are never sent to the retrieval backend.

    Attributes:
        client: Retrieval backend adapter.
        timeout_seconds: Deadline applied to each retrieval call.
        questions_per_competency: Questions retrieved per competency.
        company_top_k: Company knowledge passages retrieved.
        max_competencies: Upper bound on rubric competencies retrieved.
    """

    def __init__(
        self,
        client: RetrievalClient,
        timeout_seconds: float = 3.0,
        questions_per_competency: int = 4,
        company_top_k: int = 5,
        max_competencies: int = 12,
    ) -> None:
        self.client = client
        self.timeout_seconds = timeout_seconds
        self.questions_per_competency = questions_per_competency
        self.company_top_k = company_top_k
        self.max_competencies = max_competencies

    @property
    def backend_name(self) -> str:
        """Name of the underlying retrieval backend."""
        return str(getattr(self.client, "backend_name", "unknown"))

    async def _search(self, query: RetrievalQuery) -> List[RetrievedChunk]:
        """Run one retrieval call under the per-call deadline, normalizing failures."""
        try:
            return await asyncio.wait_for(self.client.search(query), timeout=self.timeout_seconds)
        except RetrievalError:
            raise
        except asyncio.TimeoutError as e:
            raise RetrievalError(
                f"Retrieval timed out after {self.timeout_seconds}s on collection '{query.collection}'"
            ) from e
        except Exception as e:
            raise RetrievalError(f"Retrieval failed on collection '{query.collection}': {e}") from e

    async def build_pack(
        self,
        target_company: str,
        target_role: str,
        skills: Optional[Sequence[str]] = None,
        experience_years: Optional[int] = None,
        job_description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retrieve and assemble the knowledge pack for one interview.

        Args:
            target_company: Company the candidate is preparing for.
            target_role: Target job title.
            skills: Candidate-declared skills used to focus retrieval.
            experience_years: Years of experience, used when the title carries no level.
            job_description: Optional job description text (first 500 characters are used).

        Returns:
            Dict[str, Any]: Pack with role_family, level, role_profile, rubric, question_bank,
                company_context, sources and retrieval_backend.

        Raises:
            RetrievalError: If any retrieval call fails or times out.
        """
        role_family = infer_role_family(target_role, skills)
        level = infer_level(target_role, experience_years)
        family_filter = [role_family, "any"]
        focus_text = " ".join(
            part
            for part in (target_role, " ".join(skills or []), (job_description or "")[:500])
            if part
        )
        company_key = (target_company or "").strip().lower()

        role_chunks, rubric_chunks, company_chunks = await asyncio.gather(
            self._search(
                RetrievalQuery(
                    collection=RetrievalCollection.ROLE_PROFILE.value,
                    text=focus_text,
                    filters={"role_family": role_family, "level": level},
                    top_k=1,
                )
            ),
            self._search(
                RetrievalQuery(
                    collection=RetrievalCollection.RUBRIC.value,
                    text=focus_text,
                    filters={"role_family": family_filter, "levels": level},
                    top_k=self.max_competencies,
                )
            ),
            self._search(
                RetrievalQuery(
                    collection=RetrievalCollection.COMPANY_KNOWLEDGE.value,
                    text=f"{target_company} {target_role} engineering culture tech stack interview",
                    filters={"company": [company_key, "generic"]},
                    top_k=self.company_top_k,
                )
            ),
        )

        if self.backend_name == "rag" and (not role_chunks or not rubric_chunks):
            raise RetrievalError(
                "RAG has no role and rubric evidence for "
                f"role_family='{role_family}', level='{level}'"
            )

        role_profile = self._build_role_profile(role_chunks, role_family, level)
        rubric = self._build_rubric(rubric_chunks, role_profile, role_family, level)

        question_lists = await asyncio.gather(
            *[
                self._search(
                    RetrievalQuery(
                        collection=RetrievalCollection.QUESTION_BANK.value,
                        text=f"{focus_text} {comp['name']}",
                        filters={
                            "competency": comp["competency_id"],
                            "levels": level,
                            "role_family": family_filter,
                        },
                        top_k=self.questions_per_competency,
                    )
                )
                for comp in rubric["competencies"]
            ]
        )
        if self.backend_name == "rag" and not any(question_lists):
            raise RetrievalError(
                "RAG has no question evidence for "
                f"role_family='{role_family}', level='{level}'"
            )
        question_bank = {
            comp["competency_id"]: [self._to_question(chunk) for chunk in chunks]
            for comp, chunks in zip(rubric["competencies"], question_lists)
        }

        company_context = [
            {
                "topic": chunk.metadata.get("topic", ""),
                "content": chunk.content,
                "confidence": chunk.metadata.get("confidence"),
                "is_generic": chunk.metadata.get("company") == "generic",
                "source": chunk.source,
            }
            for chunk in company_chunks
        ]

        sources = [c.chunk_id for c in role_chunks + rubric_chunks + company_chunks]
        for chunks in question_lists:
            sources.extend(c.chunk_id for c in chunks)

        return {
            "retrieval_backend": self.backend_name,
            "role_family": role_family,
            "level": level,
            "role_profile": role_profile,
            "rubric": rubric,
            "question_bank": question_bank,
            "company_context": company_context,
            "sources": sources,
        }

    @staticmethod
    def _build_role_profile(
        chunks: List[RetrievedChunk], role_family: str, level: str
    ) -> Dict[str, Any]:
        """Extract the role profile from the top role_profile chunk, with safe defaults."""
        if not chunks:
            return {
                "role_family": role_family,
                "level": level,
                "expectations": "",
                "competencies": [],
                "pass_threshold": DEFAULT_PASS_THRESHOLD,
                "rubric_version": DEFAULT_RUBRIC_VERSION,
                "source": "",
            }
        top = chunks[0]
        return {
            "role_family": role_family,
            "level": level,
            "expectations": top.content,
            "competencies": list(top.metadata.get("competencies", [])),
            "pass_threshold": float(top.metadata.get("pass_threshold", DEFAULT_PASS_THRESHOLD)),
            "rubric_version": str(top.metadata.get("rubric_version", DEFAULT_RUBRIC_VERSION)),
            "source": top.source,
        }

    @staticmethod
    def _build_rubric(
        chunks: List[RetrievedChunk],
        role_profile: Dict[str, Any],
        role_family: str,
        level: str,
    ) -> Dict[str, Any]:
        """Build the weighted competency rubric, ordered and scoped by the role profile."""
        by_id: Dict[str, RetrievedChunk] = {}
        for chunk in chunks:
            cid = chunk.metadata.get("competency")
            if cid and cid not in by_id:
                by_id[cid] = chunk

        profile_ids = role_profile.get("competencies") or []
        ordered_ids = [cid for cid in profile_ids if cid in by_id] if profile_ids else list(by_id)

        total_weight = sum(float(by_id[cid].metadata.get("weight", 1.0)) for cid in ordered_ids)
        competencies = []
        for cid in ordered_ids:
            chunk = by_id[cid]
            raw_weight = float(chunk.metadata.get("weight", 1.0))
            competencies.append(
                {
                    "competency_id": cid,
                    "name": chunk.metadata.get("name", cid),
                    "description": chunk.content,
                    "weight": round(raw_weight / total_weight, 4) if total_weight else 0.0,
                    "must_have": bool(chunk.metadata.get("must_have", False)),
                    "pass_level": int(chunk.metadata.get("pass_level", 3)),
                    "anchors": dict(chunk.metadata.get("anchors", {})),
                    "source": chunk.source,
                }
            )

        return {
            "rubric_version": role_profile.get("rubric_version", DEFAULT_RUBRIC_VERSION),
            "role_family": role_family,
            "level": level,
            "scale": {"min": 1, "max": 4},
            "pass_threshold": role_profile.get("pass_threshold", DEFAULT_PASS_THRESHOLD),
            "competencies": competencies,
        }

    @staticmethod
    def _to_question(chunk: RetrievedChunk) -> Dict[str, Any]:
        """Convert a question_bank chunk into a question record."""
        meta = chunk.metadata
        return {
            "question_id": meta.get("question_id", chunk.chunk_id),
            "competency_id": meta.get("competency", ""),
            "text": chunk.content,
            "stage_type": meta.get("stage_type", "technical"),
            "difficulty": meta.get("difficulty"),
            "expected_points": list(meta.get("expected_points", [])),
            "follow_ups": list(meta.get("follow_ups", [])),
            "score": chunk.score,
            "source": chunk.source,
        }


def ground_stage_plan(
    stage_plan: Any,
    knowledge_pack: Dict[str, Any],
    max_questions_per_stage: int = 3,
) -> Any:
    """Attach rubric criteria and retrieved question candidates to each stage of a plan.

    - Keeps LLM-proposed `criteria` that exist in the rubric; otherwise derives criteria from the
      stage type and the stage types of retrieved questions.
    - Guarantees every rubric competency is assigned to at least one stage.
    - Adds `stage_type`, `criteria_labels` and `question_candidates` (without expected answers,
      which stay in the evaluator-only knowledge pack).

    Stage ids and turn budgets are never modified. Non-conforming plans are returned unchanged.
    """
    if not isinstance(stage_plan, dict) or not isinstance(stage_plan.get("stages"), list):
        return stage_plan
    competencies = knowledge_pack.get("rubric", {}).get("competencies") or []
    if not competencies:
        return stage_plan

    comp_by_id = {c["competency_id"]: c for c in competencies}
    question_bank: Dict[str, List[Dict[str, Any]]] = knowledge_pack.get("question_bank", {})
    stage_types_by_comp: Dict[str, Set[str]] = {
        cid: {q.get("stage_type", "technical") for q in question_bank.get(cid, [])}
        for cid in comp_by_id
    }

    grounded_stages: List[Any] = []
    for raw_stage in stage_plan["stages"]:
        if not isinstance(raw_stage, dict):
            grounded_stages.append(raw_stage)
            continue
        stage = dict(raw_stage)
        stage_type = stage.get("stage_type")
        if stage_type not in STAGE_TYPES:
            stage_type = classify_stage_type(stage)
        stage["stage_type"] = stage_type

        proposed = stage.get("criteria")
        criteria = [c for c in proposed if c in comp_by_id] if isinstance(proposed, list) else []
        if not criteria:
            criteria = [cid for cid in comp_by_id if stage_type in stage_types_by_comp[cid]]
        stage["criteria"] = list(dict.fromkeys(criteria))
        grounded_stages.append(stage)

    dict_stages = [s for s in grounded_stages if isinstance(s, dict)]
    if not dict_stages:
        return stage_plan

    covered = {cid for s in dict_stages for cid in s["criteria"]}
    for cid in comp_by_id:
        if cid not in covered:
            target = _pick_stage_for_competency(dict_stages, stage_types_by_comp[cid])
            target["criteria"].append(cid)

    used_question_ids: Set[str] = set()
    for stage in dict_stages:
        stage["criteria_labels"] = [comp_by_id[cid]["name"] for cid in stage["criteria"]]
        stage["question_candidates"] = _select_questions(
            stage, question_bank, max_questions_per_stage, used_question_ids
        )

    grounded = dict(stage_plan)
    grounded["stages"] = grounded_stages
    grounded["rubric_version"] = knowledge_pack.get("rubric", {}).get(
        "rubric_version", DEFAULT_RUBRIC_VERSION
    )
    return grounded


def _pick_stage_for_competency(
    stages: List[Dict[str, Any]], competency_stage_types: Set[str]
) -> Dict[str, Any]:
    """Choose the stage that should gather evidence for an otherwise uncovered competency."""
    for stage in stages:
        if stage["stage_type"] != "wrapup" and stage["stage_type"] in competency_stage_types:
            return stage
    for stage in stages:
        if stage["stage_type"] == "technical":
            return stage
    for stage in reversed(stages):
        if stage["stage_type"] != "wrapup":
            return stage
    return stages[-1]


def _select_questions(
    stage: Dict[str, Any],
    question_bank: Dict[str, List[Dict[str, Any]]],
    limit: int,
    used_question_ids: Set[str],
) -> List[Dict[str, Any]]:
    """Pick up to `limit` unused questions, round-robin across criteria, matching stage type first.

    Warmup and wrapup stages only take questions of their own type.
    """
    strict_type = stage["stage_type"] in ("warmup", "wrapup")
    queues: List[List[Dict[str, Any]]] = []
    for cid in stage["criteria"]:
        questions = [q for q in question_bank.get(cid, []) if q["question_id"] not in used_question_ids]
        matching = [q for q in questions if q.get("stage_type") == stage["stage_type"]]
        others = [] if strict_type else [q for q in questions if q.get("stage_type") != stage["stage_type"]]
        queues.append(matching + others)

    selected: List[Dict[str, Any]] = []
    while len(selected) < limit and any(queues):
        for queue in queues:
            while queue and queue[0]["question_id"] in used_question_ids:
                queue.pop(0)
            if not queue or len(selected) >= limit:
                continue
            question = queue.pop(0)
            used_question_ids.add(question["question_id"])
            selected.append(
                {
                    "question_id": question["question_id"],
                    "competency_id": question["competency_id"],
                    "text": question["text"],
                    "follow_ups": question.get("follow_ups", [])[:2],
                }
            )
    return selected

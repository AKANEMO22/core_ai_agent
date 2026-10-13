"""RAG's session-level API schemas.

The agent integration contract lives in :mod:`core.retrieval.port`. These schemas
describe the richer standalone interview-context API and deliberately use distinct
names so the two interfaces cannot be confused.
"""

from typing import Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

RAGLevel = Literal["Intern", "Fresher", "Junior"]
RetrievalMode = Literal["bm25", "dense", "text", "graph", "hybrid", "auto"]


class CandidateProject(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str | None = None
    role: str | None = None
    tech_stack: list[str] = Field(default_factory=list)
    description: str | None = None


class CandidateExperience(BaseModel):
    model_config = ConfigDict(extra="ignore")

    company: str | None = None
    position: str | None = None
    responsibilities: list[str] = Field(default_factory=list)


class CandidateProfile(BaseModel):
    """Minimal local shape needed by RAG; unrelated profile data is ignored."""

    model_config = ConfigDict(extra="ignore")

    candidate_id: str | None = None
    current_level: str = "Fresher"
    years_of_experience: float = 0
    technical_skills: list[str] = Field(default_factory=list)
    projects: list[CandidateProject] = Field(default_factory=list)
    work_experience: list[CandidateExperience] = Field(default_factory=list)


class JobDescription(BaseModel):
    """Minimal local shape needed by RAG, including legacy Vietnamese data aliases."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    job_id: str | None = None
    title: str = Field(
        default="", validation_alias=AliasChoices("title", "tên job")
    )
    normalized_title: str = Field(
        default="",
        validation_alias=AliasChoices("normalized_title", "normalized_job_title"),
    )
    job_family: str = ""
    levels: str = ""
    must_have_requirements: list[str] = Field(default_factory=list)
    preferred_requirements: list[str] = Field(default_factory=list)
    skill_requirements: list[str] = Field(default_factory=list)


class SourceRef(BaseModel):
    locator: str
    content_hash: str
    url: str | None = None
    review_status: str = "unreviewed"


class EvidencePath(BaseModel):
    nodes: list[str]
    relations: list[str]
    sources: list[SourceRef]
    confidence: float = Field(ge=0, le=1)


class RetrievalHit(BaseModel):
    question_id: str
    channel: str
    rank: int
    score: float
    paths: list[EvidencePath] = Field(default_factory=list)


class QuestionBundle(BaseModel):
    question_id: str
    original_id: str
    role_id: str
    level: RAGLevel
    question_type: Literal["technical", "behavioral"] = "technical"
    applicable_levels: list[RAGLevel] = Field(default_factory=list)
    stages: list[int] = Field(default_factory=lambda: [2])
    question: str
    expected_answer: str
    rubric: dict[str, str]
    concepts: list[str]
    sources: list[SourceRef]
    reasons: list[str] = Field(default_factory=list)
    graph_paths: list[EvidencePath] = Field(default_factory=list)
    score: float = 0


class SkillEvidence(BaseModel):
    skill: str
    status: Literal[
        "claimed",
        "project_evidence",
        "not_evidenced",
        "verified_in_interview",
        "failed_verification",
    ]
    evidence: list[str] = Field(default_factory=list)


class RAGInterviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str | None = None
    candidate: CandidateProfile | None = None
    job_id: str | None = None
    job: JobDescription | None = None
    role_id: str | None = None
    target_level: RAGLevel | None = None
    query: str = Field(default="", max_length=16000)
    required_skills: list[str] = Field(default_factory=list, max_length=100)
    mode: RetrievalMode = "hybrid"
    stage: int = Field(default=2, ge=1, le=10)
    top_k: int = Field(default=5, ge=1, le=25)
    candidate_pool: int = Field(default=30, ge=1, le=100)
    token_budget: int = Field(default=24000, ge=256, le=24000)
    asked_question_ids: list[str] = Field(default_factory=list, max_length=500)
    session_id: str | None = None
    tenant_id: str = Field(default="local", min_length=1, max_length=100)
    include_prerequisites: bool = True
    graph_max_hops: int = Field(default=3, ge=1, le=4)
    rerank: bool = True
    plan_coverage: bool = True

    @model_validator(mode="after")
    def validate_input(self):
        if self.candidate and self.candidate_id:
            raise ValueError("Use candidate or candidate_id, not both")
        if self.job and self.job_id:
            raise ValueError("Use job or job_id, not both")
        if not self.job and not self.job_id and not self.role_id:
            raise ValueError("A job or explicit role_id is required")
        if self.candidate_pool < self.top_k:
            raise ValueError("candidate_pool must be >= top_k")
        return self


class RAGInterviewResponse(BaseModel):
    schema_version: str = "1.0"
    request_id: str
    session_id: str | None = None
    index_version: str
    embedding_provider: str
    status: Literal[
        "ok", "empty", "unsupported_role", "unsupported_level", "unsupported_stage"
    ]
    role_id: str | None = None
    candidate_level: str | None = None
    target_level: RAGLevel | None = None
    interviewer_persona: dict[str, Any] = Field(default_factory=dict)
    candidate_summary: dict[str, Any] = Field(default_factory=dict)
    job_requirements: dict[str, Any] = Field(default_factory=dict)
    interview_agenda: list[dict[str, Any]] = Field(default_factory=list)
    skill_evidence: list[SkillEvidence] = Field(default_factory=list)
    selected_questions: list[QuestionBundle] = Field(default_factory=list)
    interview_framework: dict[str, Any] = Field(default_factory=dict)
    covered_requirements: list[str] = Field(default_factory=list)
    uncovered_requirements: list[str] = Field(default_factory=list)
    graph_paths: list[EvidencePath] = Field(default_factory=list)
    supporting_knowledge: list[dict[str, Any]] = Field(default_factory=list)
    prompt_context: str = ""
    degraded_mode: list[str] = Field(default_factory=list)
    trace: dict[str, Any] = Field(default_factory=dict)


class RAGFollowUpRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    tenant_id: str = "local"
    answer: str = Field(default="", max_length=16000)
    answered_question_id: str | None = None
    # Evidence is supplied by a trusted interviewer/grader, never inferred from a mention.
    verified_skills: list[str] = Field(default_factory=list)
    failed_skills: list[str] = Field(default_factory=list)
    top_k: int = Field(default=3, ge=1, le=25)


class GoldCase(BaseModel):
    id: str
    split: Literal["dev", "test"]
    group_id: str
    query_type: str
    request: RAGInterviewRequest
    relevance: dict[str, int]
    answerable: bool = True
    label_origin: str = "synthetic_unreviewed"

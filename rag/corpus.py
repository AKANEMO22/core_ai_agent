"""Validated, local schema for the checked-in interview framework corpus."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class InterviewStage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    stage_name: str
    stage: int | None = None
    weight_percent: float | None = None
    description: str | None = None


class EvaluationPillar(BaseModel):
    model_config = ConfigDict(extra="ignore")

    pillar: str
    pillar_en: str
    weight_percent: float = 1
    criteria: str = ""


class TechnicalQuestion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    question: str
    level: str
    expected_answer: str
    key_concepts: list[str] = Field(default_factory=list)
    scoring_rubric: dict[str, str] = Field(default_factory=dict)
    citation_url: str | None = None


class BehavioralQuestion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    question: str
    evaluation_focus: str
    star_criteria: dict[str, str] = Field(default_factory=dict)


class InterviewFramework(BaseModel):
    """Fields used by the indexer; unrelated source fields remain preserved."""

    model_config = ConfigDict(extra="allow")

    position_id: str
    role_title: str
    major_category: str = ""
    role_family: str = ""
    target_levels: list[str] = Field(default_factory=list)
    target_competencies: list[str] = Field(default_factory=list)
    vn_recruitment_process: str | None = None
    frequently_asked_by: list[str] = Field(default_factory=list)
    interview_stages: list[InterviewStage] = Field(default_factory=list)
    evaluation_matrix: list[EvaluationPillar] = Field(default_factory=list)
    scoring_anchors: dict[str, str] = Field(default_factory=dict)
    passing_thresholds: dict[str, str] = Field(default_factory=dict)
    technical_questions: list[TechnicalQuestion] = Field(default_factory=list)
    behavioral_questions: list[BehavioralQuestion] = Field(default_factory=list)

    @model_validator(mode="after")
    def number_stages(self):
        for number, stage in enumerate(self.interview_stages, start=1):
            if stage.stage is None:
                stage.stage = number
        return self


def validate_frameworks(value: Any) -> list[InterviewFramework]:
    if not isinstance(value, list):
        raise ValueError("interview_frameworks.json must contain a JSON array")
    frameworks = [InterviewFramework.model_validate(item) for item in value]
    if not frameworks:
        raise ValueError("No frameworks to index")
    ids = [framework.position_id for framework in frameworks]
    if len(ids) != len(set(ids)):
        raise ValueError("Framework position_id values must be unique")
    return frameworks

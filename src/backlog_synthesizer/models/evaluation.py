"""Evaluation framework data models for the Backlog Synthesizer system."""

from pydantic import BaseModel, Field

from backlog_synthesizer.models.output import UserStory


class GoldenEntry(BaseModel):
    """A single entry in the golden dataset for evaluation."""

    transcript: str
    expected_stories: list[UserStory]


class JudgeScores(BaseModel):
    """LLM-as-judge scores for a generated user story."""

    relevance: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    clarity: int = Field(ge=1, le=5)


class EvaluationCaseResult(BaseModel):
    """Result of evaluating a single golden dataset entry."""

    case_index: int
    keyword_overlap_score: float = Field(ge=0.0, le=1.0)
    judge_scores: JudgeScores | None = None
    failure_reason: str | None = None


class EvaluationReport(BaseModel):
    """Complete evaluation report with per-case and aggregate metrics."""

    results: list[EvaluationCaseResult]
    aggregate_keyword_overlap_mean: float
    aggregate_keyword_overlap_min: float
    aggregate_relevance_mean: float | None = None
    aggregate_completeness_mean: float | None = None
    aggregate_clarity_mean: float | None = None

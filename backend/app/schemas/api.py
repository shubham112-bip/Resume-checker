"""Request/response schemas of the REST API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from app.schemas.base import CamelModel
from app.schemas.candidate import CandidateProfile
from app.schemas.evaluation import CandidateEvaluation, Recommendation
from app.schemas.job import JobProfile
from app.schemas.quality import ResumeQualityEvaluation


class JobCreateRequest(CamelModel):
    text: str = Field(min_length=30, max_length=20000)
    title: str | None = Field(default=None, max_length=200)


class JobSummary(CamelModel):
    job_id: str
    title: str
    required_skills: list[str]
    preferred_skills: list[str]
    created_at: datetime


class UploadedResume(CamelModel):
    candidate_id: str
    file_name: str
    candidate_name: str
    status: str  # PARSED | PARSING_FAILED
    warnings: list[str] = Field(default_factory=list)


class UploadError(CamelModel):
    file_name: str
    error: str


class UploadResponse(CamelModel):
    candidates: list[UploadedResume]
    errors: list[UploadError] = Field(default_factory=list)


class EvaluateRequest(CamelModel):
    job_id: str
    candidate_ids: list[str] = Field(min_length=1, max_length=50)


class RankingEntry(CamelModel):
    rank: int
    evaluation_id: str
    candidate_id: str
    candidate_name: str
    file_name: str
    status: str
    overall_score: float
    recommendation: Recommendation
    recommendation_label: str
    needs_human_review: bool
    review_flags: list[str]
    keyword_ats_score: float
    keyword_rank: int
    ml_label: str | None = None
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)


class EvaluationRunResponse(CamelModel):
    run_id: str
    job_id: str
    job_title: str
    created_at: datetime
    duration_ms: float
    ranking: list[RankingEntry]
    evaluations: list[CandidateEvaluation]


class RunSummary(CamelModel):
    run_id: str
    job_id: str
    job_title: str
    candidate_count: int
    created_at: datetime


class CandidateSummary(CamelModel):
    candidate_id: str
    candidate_name: str
    file_name: str
    created_at: datetime
    latest_evaluation_id: str | None = None
    latest_score: float | None = None
    latest_recommendation: str | None = None


class ModelMetrics(CamelModel):
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    shortlist_recall: float
    cv_f1_mean: float
    cv_f1_std: float
    confusion_matrix: list[list[int]]
    feature_importance: dict[str, float]
    importance_method: str


class DatasetInfo(CamelModel):
    source: str
    records: int
    train_records: int
    test_records: int
    features: list[str]
    target: str
    labels: list[str]
    class_distribution: dict[str, int]
    preprocessing: str
    split: str
    random_seed: int


class CalibrationMethodMetrics(CamelModel):
    log_loss: float
    brier_score: float
    macro_f1: float


class ModelMetricsResponse(CamelModel):
    available: bool
    selected_model: str | None = None
    selection_criterion: str = ""
    dataset: DatasetInfo | None = None
    models: list[ModelMetrics] = Field(default_factory=list)
    disclaimer: str = ""
    trained_at: str | None = None
    calibration_comparison: dict[str, CalibrationMethodMetrics] | None = None


class ComparisonCase(CamelModel):
    case_id: str
    title: str
    description: str
    requirement: str
    resume_text: str
    keyword_ats_found: bool
    keyword_mentions: int
    ai_status: str
    ai_similarity: float
    ai_evidence: str | None = None
    stuffing_suspected: bool
    verdict: str


class ComparisonResponse(CamelModel):
    cases: list[ComparisonCase]
    embedding_backend: str


class ChatRequest(CamelModel):
    question: str = Field(min_length=3, max_length=1000)
    run_id: str | None = None


class ToolCallRecord(CamelModel):
    tool: str
    arguments: dict[str, Any]
    result_summary: str


class ChatResponse(CamelModel):
    answer: str
    generated_by: str  # gemini | rule-based
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


class HealthResponse(CamelModel):
    status: str
    gemini_enabled: bool
    gemini_model: str | None
    embedding_backend: str
    embedding_model: str
    ml_model_available: bool
    selected_model: str | None


class EvaluationBrief(CamelModel):
    evaluation_id: str
    run_id: str
    job_title: str
    overall_score: float
    recommendation: Recommendation
    created_at: datetime


class CandidateDetail(CamelModel):
    candidate_id: str
    candidate_name: str
    file_name: str
    created_at: datetime
    profile: CandidateProfile | None
    evaluations: list[EvaluationBrief]


class EpisodeResponse(CamelModel):
    episode_id: int
    candidate_id: str
    candidate_name: str
    job_id: str
    job_title: str
    event_type: str
    summary: str
    overall_score: float | None
    created_at: datetime


class GraphResponse(CamelModel):
    mermaid: str
    nodes: list[str]
    conditional_edges: list[str]


class DemoJob(CamelModel):
    demo_id: str
    title: str
    text: str


class DemoLoadResponse(CamelModel):
    job: JobProfile
    upload: UploadResponse


class DatasetResumeBrief(CamelModel):
    resume_id: str
    candidate_name: str
    file_type: str
    word_count: int
    quality_score: float
    quality_grade: str
    action_verbs_count: int
    quantified_bullets_count: int


class DatasetJdBrief(CamelModel):
    jd_id: str
    title: str
    experience_level: str
    required_skills_count: int
    preferred_skills_count: int


class DatasetSummaryResponse(CamelModel):
    total_resumes: int
    total_job_descriptions: int
    total_pairs: int
    class_distribution: dict[str, int]
    labels_distribution: dict[str, int]
    label_sources: dict[str, int]
    resumes: list[DatasetResumeBrief] = Field(default_factory=list)
    job_descriptions: list[DatasetJdBrief] = Field(default_factory=list)
    notice: str


class DatasetPairSummary(CamelModel):
    resume_id: str
    jd_id: str
    candidate_name: str
    job_title: str
    overall_candidate_fit_score: float
    resume_quality_score: float
    label: str
    classification_label: str
    label_source: str
    required_skill_match: float
    experience_match: float
    semantic_similarity: float
    education_match: float
    project_match: float
    ats_score: float
    resume_quality: float


class ResumeQualityUploadResponse(CamelModel):
    candidate_name: str
    file_name: str
    evaluation: ResumeQualityEvaluation


class HumanValidationPair(CamelModel):
    resume_id: str
    jd_id: str
    candidate_name: str
    job_title: str
    human_label: str
    rule_label: str
    ml_prediction: str
    overall_fit_score: float
    resume_quality_score: float
    notes: str = ""
    evidence: str = ""


class HumanValidationResponse(CamelModel):
    available: bool = True
    total_pairs: int = 64
    total_resumes: int = 45
    total_jds: int = 12
    reviewer_count: int = 1
    reviewer_id: str = "human_reviewer_01"
    class_distribution_4class: dict[str, int] = Field(default_factory=dict)
    class_distribution_3class: dict[str, int] = Field(default_factory=dict)
    ml_vs_human: dict[str, Any] = Field(default_factory=dict)
    rule_vs_human: dict[str, Any] = Field(default_factory=dict)
    calibrated_metrics: dict[str, Any] | None = None
    pairs: list[HumanValidationPair] = Field(default_factory=list)
    notice: str = ""


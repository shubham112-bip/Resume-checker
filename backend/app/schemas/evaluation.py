"""Evaluation contracts exchanged between agents and returned by the API."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import Field

from app.schemas.base import CamelModel
from app.schemas.candidate import CandidateProfile
from app.schemas.job import SkillTier
from app.schemas.quality import ResumeQualityEvaluation


def utc_now() -> datetime:
    return datetime.now(UTC)


class EvidenceStatus(StrEnum):
    MATCH = "MATCH"  # explicit mention supported by experience/project evidence
    SEMANTIC_MATCH = "SEMANTIC_MATCH"  # different wording, strong semantic evidence
    CLAIMED_ONLY = "CLAIMED_ONLY"  # listed (e.g. skills section) without supporting evidence
    INFERRED = "INFERRED"  # implied by a related skill; needs human verification
    UNCERTAIN = "UNCERTAIN"  # weak semantic evidence
    NOT_FOUND = "NOT_FOUND"


class Recommendation(StrEnum):
    STRONG_SHORTLIST = "STRONG_SHORTLIST"
    SHORTLIST = "SHORTLIST"
    REVIEW = "REVIEW"
    WEAK_MATCH = "WEAK_MATCH"


RECOMMENDATION_LABELS: dict[Recommendation, str] = {
    Recommendation.STRONG_SHORTLIST: "Strong Match",
    Recommendation.SHORTLIST: "Match",
    Recommendation.REVIEW: "Needs Review",
    Recommendation.WEAK_MATCH: "Weak Match",
}


class EvidencePassage(CamelModel):
    text: str
    section: str
    similarity: float = 0.0
    match_type: str = "semantic"  # "keyword" | "semantic" | "reflection"


class RequirementMatch(CamelModel):
    requirement: str
    tier: SkillTier
    status: EvidenceStatus
    confidence: float = 0.0
    credit: float = 0.0
    keyword_found: bool = False
    reflected: bool = False
    evidence: list[EvidencePassage] = Field(default_factory=list)
    note: str = ""


class ATSCheck(CamelModel):
    name: str
    category: str  # contact | sections | formatting | keywords
    passed: bool
    detail: str = ""


class ATSResult(CamelModel):
    score: float  # 0-100
    contact_score: float
    section_score: float
    keyword_coverage: float
    formatting_score: float
    checks: list[ATSCheck] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    matched_keywords: list[str] = Field(default_factory=list)
    missing_keywords: list[str] = Field(default_factory=list)


class StuffedTerm(CamelModel):
    term: str
    mentions: int
    evidence_mentions: int


class StuffingResult(CamelModel):
    suspected: bool = False
    keyword_density: float = 0.0
    flagged_terms: list[StuffedTerm] = Field(default_factory=list)
    detail: str = ""


class FeatureVector(CamelModel):
    """Normalised (0-1) job-relevant features. Contains no protected attributes."""

    required_skill_match: float
    preferred_skill_match: float
    experience_match: float
    education_match: float
    project_match: float
    semantic_similarity: float
    ats_score: float
    resume_quality: float


class ScoreComponent(CamelModel):
    key: str
    label: str
    weight: float
    value: float  # 0-100
    contribution: float  # weight * value


class ScoreBreakdown(CamelModel):
    components: list[ScoreComponent]
    overall_score: float
    formula: str


class MLPrediction(CamelModel):
    available: bool = False
    model_name: str = ""
    predicted_label: str = ""
    probabilities: dict[str, float] = Field(default_factory=dict)
    agrees_with_rules: bool = True
    detail: str = ""
    model_version: str = "ml_v2"
    feature_version: str = "canonical_v2"
    feature_contributions: dict[str, float] = Field(default_factory=dict)


class ReviewFlag(CamelModel):
    code: str  # BORDERLINE | ML_DISAGREEMENT | KEYWORD_STUFFING | PARSING_ISSUE | LOW_CONFIDENCE | BORDERLINE_ML_SECOND_OPINION | BORDERLINE_ML_CAUTION | DIVERGENCE_REJECTED
    message: str


class HybridDecision(CamelModel):
    """Deterministic resolution combining Rule v2 transparent score and ML v2 calibrated second-opinion."""

    rule_score: float
    rule_decision: str  # "STRONG_SHORTLIST" | "SHORTLIST" | "REVIEW" | "WEAK_MATCH"
    ml_decision: str  # "shortlist" | "review" | "weak_match" | "unavailable"
    ml_probability: float
    final_decision: str  # "STRONG_SHORTLIST" | "SHORTLIST" | "REVIEW" | "WEAK_MATCH"
    agreement: bool
    is_borderline: bool
    policy_reason: str
    review_recommendation_flag: str | None = None
    feature_contributions: dict[str, float] = Field(default_factory=dict)


class AgentTraceEvent(CamelModel):
    """Safe, user-visible action trace (no hidden chain-of-thought)."""

    step: int = 0
    agent: str
    task: str
    tool: str | None = None
    tool_input: str | None = None
    observation: str = ""
    status: str = "success"  # success | warning | error | skipped
    duration_ms: float = 0.0
    details: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=utc_now)


class Explanation(CamelModel):
    summary: str
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    rationale: str = ""
    generated_by: str = "template"  # "gemini" | "template"
    model: str | None = None
    fallback_reason: str | None = None


class MemoryEpisode(CamelModel):
    candidate_name: str
    job_title: str
    overall_score: float
    recommendation: str
    created_at: datetime


class CandidateEvaluation(CamelModel):
    evaluation_id: str
    run_id: str
    candidate_id: str
    job_id: str
    job_title: str
    candidate_name: str
    file_name: str
    status: str = "COMPLETED"  # COMPLETED | PARSING_FAILED
    error: str | None = None

    overall_score: float = 0.0
    ats_score: float = 0.0
    required_skill_match: float = 0.0
    preferred_skill_match: float = 0.0
    experience_match: float = 0.0
    semantic_similarity: float = 0.0
    project_match: float = 0.0
    education_match: float = 0.0
    resume_quality: float = 0.0
    quality_evaluation: ResumeQualityEvaluation | None = None
    keyword_ats_score: float = 0.0  # traditional keyword-only baseline

    score_breakdown: ScoreBreakdown | None = None
    features: FeatureVector | None = None
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    uncertain_skills: list[str] = Field(default_factory=list)
    requirement_matches: list[RequirementMatch] = Field(default_factory=list)
    ats: ATSResult | None = None
    stuffing: StuffingResult | None = None
    ml_prediction: MLPrediction | None = None
    hybrid_decision: HybridDecision | None = None

    recommendation: Recommendation = Recommendation.WEAK_MATCH
    recommendation_label: str = "Weak Match"
    needs_human_review: bool = False
    review_flags: list[ReviewFlag] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    explanation: Explanation | None = None

    recruiter_decision: str | None = None
    recruiter_notes: str | None = None
    recruiter_decided_at: str | None = None

    experience_years: float = 0.0
    profile: CandidateProfile | None = None
    memory_recall: list[MemoryEpisode] = Field(default_factory=list)
    trace: list[AgentTraceEvent] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class RecruiterDecisionRequest(CamelModel):
    decision: str = Field(..., description="Recruiter determination: SHORTLIST, REVIEW, or REJECT")
    notes: str = Field(default="", description="Recruiter feedback or interview notes")

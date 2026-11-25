"""LangGraph state representation for candidate evaluation."""

from __future__ import annotations

from typing import Annotated, Any
from pathlib import Path

from pydantic import BaseModel, Field
from app.schemas.candidate import CandidateProfile
from app.schemas.evaluation import (
    ATSResult,
    AgentTraceEvent,
    Explanation,
    FeatureVector,
    HybridDecision,
    MLPrediction,
    MemoryEpisode,
    Recommendation,
    RequirementMatch,
    ReviewFlag,
    ScoreBreakdown,
    StuffingResult,
)
from app.schemas.job import JobProfile


def append_traces(existing: list[AgentTraceEvent], new_traces: list[AgentTraceEvent]) -> list[AgentTraceEvent]:
    return existing + new_traces


class CandidateAgentState(BaseModel):
    # Pipeline execution parameters
    candidate_id: str
    run_id: str
    file_path: str
    job: JobProfile

    # Parsing artifacts
    profile: CandidateProfile | None = None
    parsing_failed: bool = False
    parsing_strategy: str = "primary"
    parsing_error: str | None = None

    # ATS and Semantic evaluation
    ats: ATSResult | None = None
    stuffing: StuffingResult | None = None
    matches: list[RequirementMatch] = Field(default_factory=list)
    mean_semantic_sim: float = 0.0
    needs_reflection: bool = False

    # Scoring and Features
    features: FeatureVector | None = None
    score_breakdown: ScoreBreakdown | None = None
    overall_score: float = 0.0
    keyword_ats_score: float = 0.0

    # ML & Human Review
    ml_prediction: MLPrediction | None = None
    hybrid_decision: HybridDecision | None = None
    recommendation: Recommendation = Recommendation.WEAK_MATCH
    recommendation_label: str = "Weak Match"
    needs_human_review: bool = False
    review_flags: list[ReviewFlag] = Field(default_factory=list)

    # Final outputs
    explanation: Explanation | None = None
    memory_episodes: list[MemoryEpisode] = Field(default_factory=list)

    # Safe Agent Action Trace
    trace: Annotated[list[AgentTraceEvent], append_traces] = Field(default_factory=list)

"""Pydantic schemas for the standalone Resume Quality Model (separate from Candidate Fit)."""

from __future__ import annotations

from pydantic import Field
from app.schemas.base import CamelModel


class QualityDimension(CamelModel):
    name: str
    label: str
    weight: float
    score: float  # 0.0 - 1.0
    value_points: float  # score * 100
    details: list[str] = Field(default_factory=list)


class ResumeQualityEvaluation(CamelModel):
    overall_quality_score: float  # 0.0 - 100.0
    quality_grade: str  # Excellent (>=85), Good (>=70), Needs Improvement (>=50), Low (<50)
    completeness_score: float  # 0.0 - 1.0
    ats_readability_score: float  # 0.0 - 1.0
    structure_score: float  # 0.0 - 1.0
    action_verb_score: float  # 0.0 - 1.0
    quantification_score: float  # 0.0 - 1.0
    clarity_score: float  # 0.0 - 1.0
    formatting_consistency_score: float  # 0.0 - 1.0

    dimensions: list[QualityDimension] = Field(default_factory=list)
    action_verbs_found: list[str] = Field(default_factory=list)
    quantified_bullets_count: int = 0
    total_bullets_count: int = 0
    strengths: list[str] = Field(default_factory=list)
    improvement_areas: list[str] = Field(default_factory=list)
    verdict: str = ""

"""Deterministic, evidence-grounded candidate explanation generator.

Computes explainable summaries, strengths, weaknesses, and scoring rationales
strictly from verified feature engineering, skill taxonomy matches, and ML calibration.
Does not use external Generative AI or LLMs.
"""

from __future__ import annotations

import logging
from typing import Any

from app.schemas.candidate import CandidateProfile
from app.schemas.evaluation import Explanation, Recommendation, RequirementMatch
from app.schemas.job import JobProfile

logger = logging.getLogger(__name__)


def generate_candidate_explanation(
    profile: CandidateProfile,
    job: JobProfile,
    overall_score: float,
    recommendation: Recommendation,
    matches: list[RequirementMatch],
    features: Any,
    ats_score: float,
) -> Explanation:
    """Generate a fully deterministic, mathematically grounded explanation for candidate fit.

    Every statement is backed strictly by verified skills, experience thresholds,
    semantic similarity metrics, and ATS compatibility checks.
    """
    matched = [m.requirement for m in matches if m.status.value in {"MATCH", "SEMANTIC_MATCH"}]
    missing = [m.requirement for m in matches if m.status.value == "NOT_FOUND"]
    uncertain = [m.requirement for m in matches if m.status.value in {"UNCERTAIN", "CLAIMED_ONLY", "INFERRED"}]

    exp_years = profile.total_experience_months / 12.0
    sem_sim = features.semantic_similarity if features else 0.0
    min_exp_years = job.min_experience_years

    strengths: list[str] = []
    weaknesses: list[str] = []

    if matched:
        strengths.append(
            f"Demonstrates verified alignment on core skills: {', '.join(matched[:5])}."
        )
    if exp_years >= min_exp_years and min_exp_years > 0:
        strengths.append(
            f"Meets experience requirement with {exp_years:.1f} years of relevant industry experience."
        )
    if sem_sim >= 0.70:
        strengths.append(
            "Strong contextual and conceptual overlap with job responsibilities identified via semantic embeddings."
        )
    if ats_score >= 80.0:
        strengths.append(
            f"High ATS compatibility ({ats_score:.1f}/100) with well-structured resume sections and contact information."
        )
    if uncertain:
        strengths.append(
            f"Mentions relevant domain competencies ({', '.join(uncertain[:4])}) requiring recruiter experience verification."
        )
    if not strengths:
        strengths.append("Candidate profile submitted with initial domain qualifications for role evaluation.")

    if missing:
        weaknesses.append(
            f"Skill gap identified in required/preferred competencies: {', '.join(missing[:4])}."
        )
    if exp_years < min_exp_years and min_exp_years > 0:
        weaknesses.append(
            f"Experience level ({exp_years:.1f} yrs) is below the stated requirement ({min_exp_years:.1f} yrs)."
        )
    if uncertain:
        weaknesses.append(
            f"Partial evidence detected for: {', '.join(uncertain[:3])} (requires recruiter verification)."
        )

    matched_txt = f"{len(matched)} key skills ({', '.join(matched[:3])})" if matched else "limited matching skills"
    missing_txt = f"Missing requirements include {', '.join(missing[:3])}." if missing else "No major skill gaps identified."

    summary = (
        f"Candidate {profile.name} achieved an overall fit score of {overall_score}/100 for the {job.title} role. "
        f"The evaluation verified {matched_txt}. {missing_txt} "
        f"Based on the objective scoring framework, the recommendation is {recommendation.value}."
    )

    rationale = (
        f"Score of {overall_score} reflects technical match ({len(matched)} matched, {len(missing)} missing), "
        f"experience relevance ({exp_years:.1f} yrs vs {min_exp_years:.1f} req), semantic embedding similarity "
        f"({sem_sim*100:.1f}%), and ATS formatting score ({ats_score:.1f}%)."
    )

    return Explanation(
        summary=summary,
        strengths=strengths,
        weaknesses=weaknesses,
        rationale=rationale,
        generated_by="deterministic",
        model=None,
        fallback_reason=None,
    )

"""Google Gemini integration with grounded template fallbacks for structured explanation."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from app.config import get_settings
from app.schemas.candidate import CandidateProfile
from app.schemas.evaluation import Explanation, Recommendation, RequirementMatch
from app.schemas.job import JobProfile

logger = logging.getLogger(__name__)


def generate_grounded_explanation_template(
    candidate_name: str,
    job_title: str,
    overall_score: float,
    recommendation: Recommendation,
    matched_skills: list[str],
    missing_skills: list[str],
    uncertain_skills: list[str],
    experience_years: float,
    min_exp_years: float,
    semantic_sim: float,
    ats_score: float,
    fallback_reason: str = "GEMINI_API_KEY not configured",
) -> Explanation:
    """Deterministic, grounded template fallback that never invents skills or experience."""
    strengths: list[str] = []
    weaknesses: list[str] = []

    if matched_skills:
        strengths.append(
            f"Demonstrates verified alignment on core skills: {', '.join(matched_skills[:5])}."
        )
    if experience_years >= min_exp_years and min_exp_years > 0:
        strengths.append(
            f"Meets experience requirement with {experience_years:.1f} years of relevant industry experience."
        )
    if semantic_sim >= 0.70:
        strengths.append(
            "Strong contextual and conceptual overlap with job responsibilities identified via semantic embeddings."
        )
    if ats_score >= 80.0:
        strengths.append(
            f"High ATS compatibility ({ats_score:.1f}/100) with well-structured resume sections and contact information."
        )
    if uncertain_skills:
        strengths.append(
            f"Mentions relevant domain competencies ({', '.join(uncertain_skills[:4])}) requiring recruiter experience verification."
        )
    if not strengths:
        strengths.append("Candidate profile submitted with initial domain qualifications for role evaluation.")

    if missing_skills:
        weaknesses.append(
            f"Skill gap identified in required/preferred competencies: {', '.join(missing_skills[:4])}."
        )
    if experience_years < min_exp_years and min_exp_years > 0:
        weaknesses.append(
            f"Experience level ({experience_years:.1f} yrs) is below the stated requirement ({min_exp_years:.1f} yrs)."
        )
    if uncertain_skills:
        weaknesses.append(
            f"Partial evidence detected for: {', '.join(uncertain_skills[:3])} (requires recruiter verification)."
        )

    matched_txt = f"{len(matched_skills)} key skills ({', '.join(matched_skills[:3])})" if matched_skills else "limited matching skills"
    missing_txt = f"Missing requirements include {', '.join(missing_skills[:3])}." if missing_skills else "No major skill gaps identified."

    summary = (
        f"Candidate {candidate_name} achieved an overall fit score of {overall_score}/100 for the {job_title} role. "
        f"The evaluation verified {matched_txt}. {missing_txt} "
        f"Based on the objective scoring framework, the recommendation is {recommendation.value}."
    )

    rationale = (
        f"Score of {overall_score} reflects technical match ({len(matched_skills)} matched, {len(missing_skills)} missing), "
        f"experience relevance ({experience_years:.1f} yrs vs {min_exp_years:.1f} req), semantic embedding similarity "
        f"({semantic_sim*100:.1f}%), and ATS formatting score ({ats_score:.1f}%)."
    )

    return Explanation(
        summary=summary,
        strengths=strengths,
        weaknesses=weaknesses,
        rationale=rationale,
        generated_by="template",
        model=None,
        fallback_reason=fallback_reason,
    )


def generate_candidate_explanation(
    profile: CandidateProfile,
    job: JobProfile,
    overall_score: float,
    recommendation: Recommendation,
    matches: list[RequirementMatch],
    features: Any,
    ats_score: float,
) -> Explanation:
    matched = [m.requirement for m in matches if m.status.value in {"MATCH", "SEMANTIC_MATCH"}]
    missing = [m.requirement for m in matches if m.status.value == "NOT_FOUND"]
    uncertain = [m.requirement for m in matches if m.status.value in {"UNCERTAIN", "CLAIMED_ONLY", "INFERRED"}]

    exp_years = profile.total_experience_months / 12.0
    sem_sim = features.semantic_similarity if features else 0.0

    settings = get_settings()
    api_key = settings.gemini_api_key or os.environ.get("GEMINI_API_KEY")

    if not api_key:
        return generate_grounded_explanation_template(
            candidate_name=profile.name,
            job_title=job.title,
            overall_score=overall_score,
            recommendation=recommendation,
            matched_skills=matched,
            missing_skills=missing,
            uncertain_skills=uncertain,
            experience_years=exp_years,
            min_exp_years=job.min_experience_years,
            semantic_sim=sem_sim,
            ats_score=ats_score,
            fallback_reason="GEMINI_API_KEY not configured in environment (deterministic grounded fallback active)",
        )

    # Use Google GenAI SDK
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        prompt = f"""You are an objective academic AI recruitment evaluator.
Explain the evaluation result of candidate '{profile.name}' for the position '{job.title}'.

GROUND TRUTH EVALUATION METRICS (Do NOT hallucinate or invent new experience):
- Overall Candidate Fit Score: {overall_score}/100
- Recommendation: {recommendation.value}
- ATS Score: {ats_score}/100
- Experience: {exp_years:.1f} years (Job required: {job.min_experience_years} years)
- Matched Skills: {json.dumps(matched)}
- Missing Skills: {json.dumps(missing)}
- Uncertain/Claimed Skills: {json.dumps(uncertain)}

Respond ONLY with a valid JSON object matching this schema:
{{
  "summary": "Professional 2-3 sentence overview explaining why the candidate received this score and recommendation.",
  "strengths": ["List of 2-4 factual strengths grounded strictly in the matched skills and experience"],
  "weaknesses": ["List of 1-3 factual gaps or missing skills"],
  "rationale": "Clear mathematical and evidence-based explanation of the score components."
}}
"""

        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )

        data = json.loads(response.text)
        return Explanation(
            summary=data.get("summary", ""),
            strengths=data.get("strengths", []),
            weaknesses=data.get("weaknesses", []),
            rationale=data.get("rationale", ""),
            generated_by="gemini",
            model=settings.gemini_model,
            fallback_reason=None,
        )
    except Exception as e:
        logger.warning("Gemini explanation generation failed: %s. Using grounded template.", e)
        return generate_grounded_explanation_template(
            candidate_name=profile.name,
            job_title=job.title,
            overall_score=overall_score,
            recommendation=recommendation,
            matched_skills=matched,
            missing_skills=missing,
            uncertain_skills=uncertain,
            experience_years=exp_years,
            min_exp_years=job.min_experience_years,
            semantic_sim=sem_sim,
            ats_score=ats_score,
            fallback_reason=f"Gemini API call failed ({type(e).__name__}): {e}",
        )

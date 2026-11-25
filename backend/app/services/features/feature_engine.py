"""Feature engineering, transparent scoring framework, and recommendation logic."""

from __future__ import annotations

import logging

from app.config import get_settings
from app.schemas.candidate import CandidateProfile
from app.schemas.evaluation import (
    ATSResult,
    FeatureVector,
    Recommendation,
    RequirementMatch,
    ReviewFlag,
    ScoreBreakdown,
    ScoreComponent,
    StuffingResult,
)
from app.schemas.job import JobProfile, SkillTier

logger = logging.getLogger(__name__)


def compute_features(
    profile: CandidateProfile,
    job: JobProfile,
    matches: list[RequirementMatch],
    ats: ATSResult,
    mean_semantic_sim: float,
) -> tuple[FeatureVector, ScoreBreakdown, float, Recommendation, list[ReviewFlag]]:
    settings = get_settings()
    w = settings.weights
    th = settings.thresholds

    # 1. Required Skill Match (0 - 1)
    req_matches = [m for m in matches if m.tier == SkillTier.REQUIRED]
    if req_matches:
        req_score = sum(m.credit for m in req_matches) / len(req_matches)
    else:
        has_req_in_job = any(s.tier == SkillTier.REQUIRED for s in job.skills)
        req_score = 0.0 if has_req_in_job else (1.0 if not job.skills else 0.5)

    # 2. Preferred Skill Match (0 - 1)
    pref_matches = [m for m in matches if m.tier in {SkillTier.PREFERRED, SkillTier.OPTIONAL}]
    if pref_matches:
        pref_score = sum(m.credit for m in pref_matches) / len(pref_matches)
    else:
        pref_score = req_score  # fallback to required skill score

    # 3. Experience Match (0 - 1)
    cand_years = profile.total_experience_months / 12.0
    req_min_years = job.min_experience_years
    if req_min_years <= 0.0:
        exp_score = min(1.0, 0.7 + 0.15 * min(cand_years, 2.0))
    elif cand_years >= req_min_years:
        exp_score = min(1.0, 0.85 + 0.15 * ((cand_years - req_min_years) / 2.0))
    else:
        # Graceful linear drop for partial experience
        exp_score = max(0.2, (cand_years / req_min_years) * 0.85)

    # 4. Education Match (0 - 1)
    cand_deg_level = max((e.degree_level for e in profile.education), default=0)
    req_deg_level = job.required_degree_level or 2  # default bachelor
    has_cs_field = any(e.is_relevant_field for e in profile.education)

    if cand_deg_level >= req_deg_level:
        edu_score = 0.9 if not has_cs_field else 1.0
    elif cand_deg_level == req_deg_level - 1:
        edu_score = 0.7 if not has_cs_field else 0.8
    elif cand_deg_level > 0:
        edu_score = 0.5
    else:
        edu_score = 0.4 if len(profile.education) > 0 else 0.2

    # 5. Project Match (0 - 1)
    proj_count = len(profile.projects)
    if proj_count == 0:
        proj_score = 0.4 if cand_years > 2.0 else 0.2
    else:
        # Check how many projects mention JD skills
        jd_skills_lower = {s.name.lower() for s in job.skills}
        relevant_projs = 0
        for p in profile.projects:
            if any(t.lower() in jd_skills_lower for t in p.technologies):
                relevant_projs += 1
        relevance_ratio = relevant_projs / proj_count
        proj_score = min(1.0, 0.5 * min(proj_count / 3.0, 1.0) + 0.5 * relevance_ratio)

    # 6. Semantic Similarity (0 - 1)
    sem_score = max(0.0, min(1.0, mean_semantic_sim))

    # 7. ATS Score (0 - 1)
    ats_score = max(0.0, min(1.0, ats.score / 100.0))

    # 8. Resume Quality (0 - 1) evaluated objectively using multi-dimensional quality engine
    from app.services.quality.resume_quality import evaluate_resume_quality

    qual_eval = evaluate_resume_quality(profile)
    resume_quality = max(0.1, min(1.0, qual_eval.overall_quality_score / 100.0))

    features = FeatureVector(
        required_skill_match=round(req_score, 3),
        preferred_skill_match=round(pref_score, 3),
        experience_match=round(exp_score, 3),
        education_match=round(edu_score, 3),
        project_match=round(proj_score, 3),
        semantic_similarity=round(sem_score, 3),
        ats_score=round(ats_score, 3),
        resume_quality=round(resume_quality, 3),
    )

    # Compute overall score:
    # 0.30 * Required + 0.20 * Exp + 0.15 * Semantic + 0.10 * Proj + 0.10 * ATS + 0.10 * Edu + 0.05 * Quality
    components = [
        ScoreComponent(
            key="requiredSkillMatch",
            label="Required Technical Skill Match",
            weight=w.requiredSkillMatch,
            value=round(req_score * 100.0, 1),
            contribution=round(w.requiredSkillMatch * req_score * 100.0, 1),
        ),
        ScoreComponent(
            key="experienceMatch",
            label="Experience Relevance",
            weight=w.experienceMatch,
            value=round(exp_score * 100.0, 1),
            contribution=round(w.experienceMatch * exp_score * 100.0, 1),
        ),
        ScoreComponent(
            key="semanticSimilarity",
            label="Semantic Similarity",
            weight=w.semanticSimilarity,
            value=round(sem_score * 100.0, 1),
            contribution=round(w.semanticSimilarity * sem_score * 100.0, 1),
        ),
        ScoreComponent(
            key="projectMatch",
            label="Project Relevance",
            weight=w.projectMatch,
            value=round(proj_score * 100.0, 1),
            contribution=round(w.projectMatch * proj_score * 100.0, 1),
        ),
        ScoreComponent(
            key="atsScore",
            label="ATS Compatibility",
            weight=w.atsScore,
            value=round(ats_score * 100.0, 1),
            contribution=round(w.atsScore * ats_score * 100.0, 1),
        ),
        ScoreComponent(
            key="educationMatch",
            label="Education Match",
            weight=w.educationMatch,
            value=round(edu_score * 100.0, 1),
            contribution=round(w.educationMatch * edu_score * 100.0, 1),
        ),
        ScoreComponent(
            key="resumeQuality",
            label="Resume Quality",
            weight=w.resumeQuality,
            value=round(resume_quality * 100.0, 1),
            contribution=round(w.resumeQuality * resume_quality * 100.0, 1),
        ),
    ]

    overall_score = round(sum(c.contribution for c in components), 1)

    formula = (
        "Overall Score = 0.30 × Required Skill + 0.20 × Experience + "
        "0.15 × Semantic Similarity + 0.10 × Project + 0.10 × ATS + "
        "0.10 × Education + 0.05 × Quality"
    )

    score_breakdown = ScoreBreakdown(
        components=components,
        overall_score=overall_score,
        formula=formula,
    )

    # Recommendation determination
    if overall_score >= th.strongShortlist:
        recommendation = Recommendation.STRONG_SHORTLIST
    elif overall_score >= th.shortlist:
        recommendation = Recommendation.SHORTLIST
    elif overall_score >= th.review:
        recommendation = Recommendation.REVIEW
    else:
        recommendation = Recommendation.WEAK_MATCH

    # Check for borderline margin
    review_flags: list[ReviewFlag] = []
    margin = th.borderlineMargin
    for threshold_val in [th.strongShortlist, th.shortlist, th.review]:
        if abs(overall_score - threshold_val) <= margin:
            review_flags.append(
                ReviewFlag(
                    code="BORDERLINE",
                    message=f"Candidate overall score ({overall_score}) is within {margin} pts of threshold ({threshold_val}). Recommended for human review.",
                )
            )
            break

    return features, score_breakdown, overall_score, recommendation, review_flags

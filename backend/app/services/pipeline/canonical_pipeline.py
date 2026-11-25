"""Canonical Resume-JD Feature Extraction & Evaluation Pipeline.

Architectural Rule (Section 5):
Create ONE canonical feature pipeline that is used both during:
1. Model Development / Training:
   Resume + JD -> Canonical Pipeline -> Feature Vector -> ML Training
2. Production Inference:
   NEW Unseen Resume + NEW Unseen JD -> EXACT SAME Canonical Pipeline -> Feature Vector -> ML Predictor

Guarantees 100% mathematical and feature consistency between training and inference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.schemas.candidate import CandidateProfile
from app.schemas.evaluation import (
    ATSResult,
    CandidateEvaluation,
    FeatureVector,
    MLPrediction,
    Recommendation,
    RequirementMatch,
    ReviewFlag,
    ScoreBreakdown,
    StuffingResult,
)
from app.schemas.job import JobProfile, SkillTier
from app.schemas.quality import ResumeQualityEvaluation
from app.services.ats.ats_engine import detect_keyword_stuffing, evaluate_ats
from app.services.embeddings.vector_store import FAISSVectorStore
from app.services.fairness import audit_feature_names, redact_sensitive_attributes
from app.services.features.feature_engine import compute_features
from app.services.ml.dataset import FEATURE_NAMES
from app.services.ml.predictor import predict_candidate_fit
from app.services.parsing.jd_parser import parse_job_description, parse_job_file
from app.services.parsing.resume_parser import parse_resume
from app.services.quality.resume_quality import evaluate_resume_quality
from app.services.semantic.matcher import match_requirements

logger = logging.getLogger(__name__)


@dataclass
class CanonicalFeatures:
    """The standard evaluation and feature representation for any Resume-JD pair."""

    profile: CandidateProfile
    job: JobProfile
    features: FeatureVector
    feature_vector_7d: list[float]
    sub_features: dict[str, Any]
    score_breakdown: ScoreBreakdown
    overall_score: float
    recommendation: Recommendation
    recommendation_label: str
    review_flags: list[ReviewFlag]
    matches: list[RequirementMatch]
    ats_result: ATSResult
    stuffing_result: StuffingResult
    quality_evaluation: ResumeQualityEvaluation
    mean_semantic_sim: float


def extract_canonical_features(
    profile: CandidateProfile,
    job: JobProfile,
    prebuilt_store: FAISSVectorStore | None = None,
) -> CanonicalFeatures:
    """Canonical feature pipeline used identically in development and production."""
    # 1. Protected attribute sanitization (Section 3)
    img_cnt = profile.diagnostics.image_count if profile.diagnostics else 0
    sanitized_text, findings = redact_sensitive_attributes(profile.text, image_count=img_cnt)
    profile.text = sanitized_text
    if findings:
        for f in findings:
            if not any(s.attribute == f.attribute for s in profile.sensitive_attributes_removed):
                profile.sensitive_attributes_removed.append(f)

    # 2. ATS evaluation & keyword stuffing audit
    ats_res = evaluate_ats(profile, job)
    stuffing_res = detect_keyword_stuffing(profile, job)

    # 3. Dense semantic vector retrieval via FAISS (BGE-small embeddings)
    matches, mean_semantic_sim = match_requirements(profile, job, prebuilt_store=prebuilt_store)

    # 4. Objective standalone resume quality evaluation
    quality = evaluate_resume_quality(profile)

    # 5. Compute primary 7 measurable features and formula score
    features, breakdown, overall_score, recommendation, review_flags = compute_features(
        profile=profile,
        job=job,
        matches=matches,
        ats=ats_res,
        mean_semantic_sim=mean_semantic_sim,
    )

    # 6. Verify fairness audit
    violating = audit_feature_names(FEATURE_NAMES)
    if violating:
        raise ValueError(f"Fairness audit failed! Sensitive features detected: {violating}")

    # 7. Compute granular sub-features (Section 6)
    req_matches = [m for m in matches if m.tier == SkillTier.REQUIRED]
    pref_matches = [m for m in matches if m.tier in {SkillTier.PREFERRED, SkillTier.OPTIONAL}]

    matched_req_count = sum(1 for m in req_matches if m.status.value in {"MATCH", "SEMANTIC_MATCH"})
    missing_req_count = sum(1 for m in req_matches if m.status.value == "NOT_FOUND")
    matched_pref_count = sum(1 for m in pref_matches if m.status.value in {"MATCH", "SEMANTIC_MATCH"})

    keyword_matched_count = sum(1 for m in matches if m.keyword_found)
    semantic_matched_count = sum(1 for m in matches if m.status.value == "SEMANTIC_MATCH")
    total_evidence_count = sum(len(m.evidence) for m in matches)

    keyword_match_ratio = (keyword_matched_count / len(matches)) if matches else 1.0
    semantic_match_ratio = (semantic_matched_count / len(matches)) if matches else 0.0

    sub_features: dict[str, Any] = {
        "preferred_skill_match": features.preferred_skill_match,
        "required_skill_count": len(req_matches),
        "matched_required_skill_count": matched_req_count,
        "missing_required_skill_count": missing_req_count,
        "preferred_skill_count": len(pref_matches),
        "matched_preferred_skill_count": matched_pref_count,
        "keyword_match_ratio": round(keyword_match_ratio, 3),
        "semantic_match_ratio": round(semantic_match_ratio, 3),
        "evidence_count": total_evidence_count,
        "keyword_stuffing_flag": 1 if stuffing_res.suspected else 0,
        "achievement_quantification_score": round(quality.quantification_score, 3),
        "action_verb_score": round(quality.action_verb_score, 3),
        "grammar_score": round(quality.clarity_score, 3),
        "section_completeness": round(quality.completeness_score, 3),
        "formatting_score": round(quality.formatting_consistency_score, 3),
    }

    feature_vector_7d = [
        features.required_skill_match,
        features.experience_match,
        features.semantic_similarity,
        features.education_match,
        features.project_match,
        features.ats_score,
        features.resume_quality,
    ]

    rec_label_map = {
        Recommendation.STRONG_SHORTLIST: "Strong Shortlist",
        Recommendation.SHORTLIST: "Shortlist",
        Recommendation.REVIEW: "Needs Review",
        Recommendation.WEAK_MATCH: "Weak Match",
    }

    return CanonicalFeatures(
        profile=profile,
        job=job,
        features=features,
        feature_vector_7d=feature_vector_7d,
        sub_features=sub_features,
        score_breakdown=breakdown,
        overall_score=overall_score,
        recommendation=recommendation,
        recommendation_label=rec_label_map.get(recommendation, "Weak Match"),
        review_flags=review_flags,
        matches=matches,
        ats_result=ats_res,
        stuffing_result=stuffing_res,
        quality_evaluation=quality,
        mean_semantic_sim=mean_semantic_sim,
    )


def evaluate_unseen_resume_and_jd(
    resume: CandidateProfile | Path | str,
    job: JobProfile | Path | str,
    candidate_id: str | None = None,
    job_title: str | None = None,
) -> CandidateEvaluation:
    """Production evaluation for ANY unseen resume + ANY unseen JD."""
    # 1. Resolve Candidate Profile
    if isinstance(resume, CandidateProfile):
        profile = resume
    elif isinstance(resume, Path):
        profile = parse_resume(resume, candidate_id=candidate_id or f"unseen_cand_{resume.stem}")
    elif isinstance(resume, str):
        is_path = ("\n" not in resume) and (len(resume) < 255) and Path(resume).exists()
        if is_path:
            profile = parse_resume(Path(resume), candidate_id=candidate_id or f"unseen_cand_{Path(resume).stem}")
        else:
            from app.services.parsing.resume_parser import parse_resume_text

            profile = parse_resume_text(resume, candidate_id=candidate_id or "unseen_cand")
    else:
        raise TypeError(f"Unsupported resume input type: {type(resume)}")

    # 2. Resolve Job Profile
    if isinstance(job, JobProfile):
        job_prof = job
    elif isinstance(job, Path):
        job_prof = parse_job_file(job)
    elif isinstance(job, str):
        is_path = ("\n" not in job) and (len(job) < 255) and Path(job).exists()
        if is_path:
            job_prof = parse_job_file(Path(job))
        else:
            job_prof = parse_job_description(job, title=job_title or "Target Role")
    else:
        raise TypeError(f"Unsupported job input type: {type(job)}")

    # 3. Execute Canonical Feature Pipeline
    canonical = extract_canonical_features(profile, job_prof)

    # 4. Predict via trained ML model and resolve hybrid decision
    ml_pred = predict_candidate_fit(canonical.features, canonical.recommendation)
    from app.services.evaluation.hybrid_engine import resolve_hybrid_decision
    hybrid_decision = resolve_hybrid_decision(canonical.overall_score, canonical.recommendation, ml_pred)

    review_flags = list(canonical.review_flags)
    if hybrid_decision.review_recommendation_flag:
        review_flags.append(
            ReviewFlag(
                code=hybrid_decision.review_recommendation_flag,
                message=hybrid_decision.policy_reason,
            )
        )

    # 5. Extract matched, missing, uncertain skills
    matched_skills = [
        m.requirement for m in canonical.matches if m.status.value in {"MATCH", "SEMANTIC_MATCH"}
    ]
    missing_skills = [
        m.requirement for m in canonical.matches if m.status.value == "NOT_FOUND"
    ]
    uncertain_skills = [
        m.requirement for m in canonical.matches if m.status.value in {"UNCERTAIN", "CLAIMED_ONLY", "INFERRED"}
    ]

    # 6. Formulate evidence-based strengths and gaps (Section 15)
    strengths: list[str] = []
    weaknesses: list[str] = []

    req_matches = [m for m in canonical.matches if m.tier == SkillTier.REQUIRED]
    matched_reqs = [m for m in req_matches if m.status.value in {"MATCH", "SEMANTIC_MATCH"}]
    strengths.append(f"✓ {len(matched_reqs)}/{len(req_matches)} required technical skills supported by evidence.")

    cand_years = profile.total_experience_months / 12.0
    if cand_years >= job_prof.min_experience_years:
        strengths.append(f"✓ {cand_years:.1f} years relevant experience (minimum required: {job_prof.min_experience_years:g} years).")
    else:
        weaknesses.append(f"✗ Experience gap: resume shows {cand_years:.1f} years, role requires {job_prof.min_experience_years:g}+ years.")

    if canonical.mean_semantic_sim >= 0.70:
        strengths.append(f"✓ Strong semantic alignment (cosine similarity: {canonical.mean_semantic_sim:.2f}) across technical domains.")

    if missing_skills:
        weaknesses.append(f"✗ Missing key requirements: {', '.join(missing_skills[:4])}.")

    if canonical.stuffing_result.suspected:
        weaknesses.append("⚠ Keyword stuffing suspected: repetitive mentions without backing evidence.")

    from app.services.explanation.explainer import generate_candidate_explanation
    explanation = generate_candidate_explanation(
        profile=profile,
        job=job_prof,
        overall_score=canonical.overall_score,
        recommendation=canonical.recommendation,
        matches=canonical.matches,
        features=canonical.features,
        ats_score=canonical.ats_result.score,
    )

    # 7. Formulate CandidateEvaluation
    evaluation = CandidateEvaluation(
        evaluation_id=f"eval_canon_{profile.candidate_id}",
        run_id="run_inference",
        candidate_id=profile.candidate_id,
        job_id=job_prof.job_id or "job_unseen",
        job_title=job_prof.title,
        candidate_name=profile.name,
        file_name=profile.candidate_id,
        status="COMPLETED",
        overall_score=canonical.overall_score,
        ats_score=canonical.ats_result.score,
        required_skill_match=round(canonical.features.required_skill_match * 100.0, 1),
        preferred_skill_match=round(canonical.features.preferred_skill_match * 100.0, 1),
        experience_match=round(canonical.features.experience_match * 100.0, 1),
        semantic_similarity=round(canonical.features.semantic_similarity * 100.0, 1),
        project_match=round(canonical.features.project_match * 100.0, 1),
        education_match=round(canonical.features.education_match * 100.0, 1),
        resume_quality=round(canonical.features.resume_quality * 100.0, 1),
        quality_evaluation=canonical.quality_evaluation,
        keyword_ats_score=canonical.ats_result.score,
        score_breakdown=canonical.score_breakdown,
        features=canonical.features,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        uncertain_skills=uncertain_skills,
        requirement_matches=canonical.matches,
        ats=canonical.ats_result,
        stuffing=canonical.stuffing_result,
        ml_prediction=ml_pred,
        hybrid_decision=hybrid_decision,
        recommendation=canonical.recommendation,
        recommendation_label=canonical.recommendation_label,
        needs_human_review=len(review_flags) > 0,
        review_flags=review_flags,
        strengths=strengths,
        weaknesses=weaknesses,
        explanation=explanation,
        experience_years=cand_years,
        profile=profile,
    )

    return evaluation

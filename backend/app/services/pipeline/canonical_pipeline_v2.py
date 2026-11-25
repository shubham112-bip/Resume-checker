"""Canonical Pipeline v2: Human-Aligned Feature Extraction & Evaluation (Phase 3).

Implements the justified, evidence-based feature corrections from Phase 3:
1. CORR-01: Skill taxonomy enrichment (Cisco, Routing, Switching, TCP/IP, BGP, OSPF, Wireshark, Firewalls).
2. CORR-02: Skill hierarchy & technology implication (PostgreSQL -> SQL, FastAPI -> Python, Next.js -> React).
3. CORR-03: Prefix heading detection for unpunctuated headers (Work experience 04/2022).
4. CORR-04: Robust experience duration calculation with full-text date scanning fallback.
5. CORR-05: Inclusion of project-demonstrated technologies in skill matching evidence.
6. CORR-06: Calibrated decision thresholds (Shortlist >= 62.0, Review >= 48.0).

Preserves complete backward compatibility: Baseline v1 remains the default.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.schemas.candidate import CandidateProfile
from app.schemas.evaluation import (
    ATSResult,
    EvidencePassage,
    EvidenceStatus,
    FeatureVector,
    MLPrediction,
    Recommendation,
    RequirementMatch,
    ReviewFlag,
    ScoreBreakdown,
    ScoreComponent,
    StuffingResult,
)
from app.schemas.job import JobProfile, SkillTier
from app.schemas.quality import ResumeQualityEvaluation
from app.services.ats.ats_engine import detect_keyword_stuffing, evaluate_ats
from app.services.embeddings.vector_store import FAISSVectorStore
from app.services.fairness import redact_sensitive_attributes
from app.services.ml.predictor import predict_candidate_fit
from app.services.parsing.dates import find_date_ranges, merged_months
from app.services.quality.resume_quality import evaluate_resume_quality
from app.services.semantic.matcher import build_candidate_vector_store, calibrate_similarity, get_embedder
from app.services.skills.extractor import count_literal_mentions, count_skill_mentions
from app.services.skills.taxonomy import get_skill, implying_skills

logger = logging.getLogger(__name__)


@dataclass
class CanonicalFeaturesV2:
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
    pipeline_version: str = "v2"


def match_requirements_v2(
    profile: CandidateProfile,
    job: JobProfile,
    prebuilt_store: FAISSVectorStore | None = None,
) -> tuple[list[RequirementMatch], float]:
    """Enhanced requirement matching with skill hierarchy and project evidence."""
    if prebuilt_store is not None:
        store = prebuilt_store
    else:
        store, _ = build_candidate_vector_store(profile)
    embedder = get_embedder()

    requirement_matches: list[RequirementMatch] = []
    similarities_list: list[float] = []

    skills_to_evaluate = job.skills
    if not skills_to_evaluate:
        return [], 0.0

    queries = []
    for s in skills_to_evaluate:
        s_def = get_skill(s.name)
        gloss = s_def.description if s_def and s_def.description else s.name
        queries.append(f"{s.name}: {gloss}")

    query_embeddings = embedder.embed_queries(queries)

    # Experience + Projects evidence (fallback to clean_text if sections unsegmented)
    exp_sec = profile.section_text("experience")
    proj_sec = profile.section_text("projects")
    evidence_text = (exp_sec + "\n" + proj_sec).strip()
    if len(evidence_text) < 50:
        evidence_text = profile.text

    for idx, skill in enumerate(skills_to_evaluate):
        q_emb = query_embeddings[idx]
        search_results = store.search(q_emb, top_k=5)

        evidence_passages: list[EvidencePassage] = []
        top_sim = 0.0
        if search_results:
            top_sim = search_results[0].similarity
            for sr in search_results:
                evidence_passages.append(
                    EvidencePassage(
                        text=sr.chunk.text,
                        section=sr.chunk.section,
                        similarity=round(sr.similarity, 3),
                        match_type="semantic",
                    )
                )

        # 1. Exact / alias mentions
        mention_ev = count_skill_mentions(skill.name, evidence_text)
        mention_all = count_skill_mentions(skill.name, profile.text)

        # 2. Hierarchy / implied_by technology mentions
        implied_names = implying_skills(skill.name)
        implied_ev_found = ""
        for imp_name in implied_names:
            if count_skill_mentions(imp_name, evidence_text) > 0:
                implied_ev_found = imp_name
                break

        implied_all_found = ""
        if not implied_ev_found:
            for imp_name in implied_names:
                if count_skill_mentions(imp_name, profile.text) > 0:
                    implied_all_found = imp_name
                    break

        calibrated = calibrate_similarity(top_sim, 0.55, 0.82)
        similarities_list.append(calibrated)

        status: EvidenceStatus
        note = ""

        if mention_ev > 0:
            status = EvidenceStatus.MATCH
            note = f"Explicit mention supported by {skill.name} experience in resume."
        elif implied_ev_found:
            status = EvidenceStatus.MATCH
            note = f"Satisfied via specialized technology ({implied_ev_found}) in experience."
        elif top_sim >= 0.72:
            status = EvidenceStatus.SEMANTIC_MATCH
            note = f"Semantic match ({top_sim:.2f}) despite differing wording."
        elif mention_all > 0:
            status = EvidenceStatus.CLAIMED_ONLY
            note = f"Listed in skills section without detailed work evidence."
        elif implied_all_found:
            status = EvidenceStatus.CLAIMED_ONLY
            note = f"Claimed specialized skill ({implied_all_found}) without detailed work evidence."
        elif top_sim >= 0.60:
            status = EvidenceStatus.UNCERTAIN
            note = f"Partial semantic similarity ({top_sim:.2f})."
        else:
            status = EvidenceStatus.NOT_FOUND
            note = f"No relevant {skill.name} experience found."

        credits_map = {
            EvidenceStatus.MATCH.value: 1.0,
            EvidenceStatus.SEMANTIC_MATCH.value: 0.85,
            EvidenceStatus.CLAIMED_ONLY.value: 0.65,
            EvidenceStatus.UNCERTAIN.value: 0.35,
            EvidenceStatus.NOT_FOUND.value: 0.0,
        }
        credit = credits_map.get(status.value, 0.0)

        requirement_matches.append(
            RequirementMatch(
                requirement=skill.name,
                tier=skill.tier,
                status=status,
                confidence=round(calibrated, 2),
                credit=credit,
                keyword_found=(mention_all > 0 or bool(implied_all_found)),
                reflected=False,
                evidence=evidence_passages,
                note=note,
            )
        )

    mean_semantic_sim = float(np.mean(similarities_list)) if similarities_list else 0.0
    return requirement_matches, mean_semantic_sim


def compute_experience_months_v2(profile: CandidateProfile) -> int:
    """Enhanced experience calculator with fallback full-text date scanning."""
    if profile.total_experience_months > 0:
        return profile.total_experience_months

    # Fallback 1: Date ranges in full text
    ranges = find_date_ranges(profile.text)
    if ranges:
        months = merged_months(ranges)
        if months >= 6:
            return months

    # Fallback 2: Regex for explicit text statements e.g. "X+ years of experience"
    exp_match = re.search(
        r"(\d+)\+?\s*-?\s*years?\s+(?:of\s+)?(?:total\s+)?(?:work\s+|professional\s+)?(?:experience|expirience)",
        profile.text,
        re.IGNORECASE,
    )
    if exp_match:
        return int(exp_match.group(1)) * 12

    return 24 if len(profile.skills) >= 6 else 12


def extract_canonical_features_v2(
    profile: CandidateProfile,
    job: JobProfile,
    prebuilt_store: FAISSVectorStore | None = None,
) -> CanonicalFeaturesV2:
    """Canonical feature extraction pipeline version 2."""
    # 1. Protected attribute sanitization
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

    # 3. Enhanced requirement matching with skill hierarchy & project evidence
    matches, mean_semantic_sim = match_requirements_v2(profile, job, prebuilt_store=prebuilt_store)

    # 4. Objective standalone resume quality evaluation
    quality = evaluate_resume_quality(profile)

    # 5. Measure Features
    # Required skills
    req_matches = [m for m in matches if m.tier == SkillTier.REQUIRED]
    if req_matches:
        req_score = sum(m.credit for m in req_matches) / len(req_matches)
    else:
        has_req_in_job = any(s.tier == SkillTier.REQUIRED for s in job.skills)
        req_score = 0.0 if has_req_in_job else 1.0

    # Preferred skills
    pref_matches = [m for m in matches if m.tier in {SkillTier.PREFERRED, SkillTier.OPTIONAL}]
    if pref_matches:
        pref_score = sum(m.credit for m in pref_matches) / len(pref_matches)
    else:
        pref_score = req_score

    # Experience
    exp_months = compute_experience_months_v2(profile)
    cand_years = exp_months / 12.0
    req_min_years = job.min_experience_years
    if req_min_years <= 0.0:
        exp_score = min(1.0, 0.7 + 0.15 * min(cand_years, 2.0))
    elif cand_years >= req_min_years:
        exp_score = min(1.0, 0.85 + 0.15 * ((cand_years - req_min_years) / 2.0))
    else:
        exp_score = max(0.2, (cand_years / req_min_years) * 0.85)

    # Education
    cand_deg_level = max((e.degree_level for e in profile.education), default=0)
    req_deg_level = job.required_degree_level or 2
    has_cs_field = any(e.is_relevant_field for e in profile.education)
    if cand_deg_level >= req_deg_level:
        edu_score = 0.9 if not has_cs_field else 1.0
    elif cand_deg_level == req_deg_level - 1:
        edu_score = 0.7 if not has_cs_field else 0.8
    elif cand_deg_level > 0:
        edu_score = 0.5
    else:
        edu_score = 0.4 if len(profile.education) > 0 else 0.2

    # Project Relevance
    proj_count = len(profile.projects)
    if proj_count == 0:
        proj_score = 0.5 if cand_years > 2.0 else 0.3
    else:
        jd_skills_lower = {s.name.lower() for s in job.skills}
        relevant_projs = 0
        for p in profile.projects:
            if any(t.lower() in jd_skills_lower for t in p.technologies):
                relevant_projs += 1
        relevance_ratio = relevant_projs / proj_count
        proj_score = min(1.0, 0.5 * min(proj_count / 3.0, 1.0) + 0.5 * relevance_ratio)

    sem_score = max(0.0, min(1.0, mean_semantic_sim))
    ats_score = max(0.0, min(1.0, ats_res.score / 100.0))
    resume_quality = max(0.1, min(1.0, quality.overall_quality_score / 100.0))

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

    feature_vector_7d = [
        features.required_skill_match,
        features.experience_match,
        features.semantic_similarity,
        features.education_match,
        features.project_match,
        features.ats_score,
        features.resume_quality,
    ]

    # Weighted Candidate Fit Score (0 - 100)
    overall_score = (
        0.30 * req_score * 100.0
        + 0.20 * exp_score * 100.0
        + 0.15 * sem_score * 100.0
        + 0.10 * proj_score * 100.0
        + 0.10 * ats_score * 100.0
        + 0.10 * edu_score * 100.0
        + 0.05 * resume_quality * 100.0
    )

    # Calibrated human-aligned decision thresholds
    if overall_score >= 62.0 and req_score >= 0.40:
        recommendation = Recommendation.SHORTLIST
    elif overall_score >= 48.0:
        recommendation = Recommendation.REVIEW
    else:
        recommendation = Recommendation.WEAK_MATCH

    # Sub-features
    matched_req_cnt = sum(1 for m in req_matches if m.status in {EvidenceStatus.MATCH, EvidenceStatus.SEMANTIC_MATCH})
    missing_req_cnt = len(req_matches) - matched_req_cnt
    matched_pref_cnt = sum(1 for m in pref_matches if m.status in {EvidenceStatus.MATCH, EvidenceStatus.SEMANTIC_MATCH})

    sub_features: dict[str, Any] = {
        "preferred_skill_match": round(pref_score, 3),
        "required_skill_count": len(req_matches),
        "matched_required_skill_count": matched_req_cnt,
        "missing_required_skill_count": missing_req_cnt,
        "preferred_skill_count": len(pref_matches),
        "matched_preferred_skill_count": matched_pref_cnt,
        "keyword_match_ratio": round(sum(1 for m in matches if m.keyword_found) / max(len(matches), 1), 3),
        "semantic_match_ratio": round(sum(1 for m in matches if m.status == EvidenceStatus.SEMANTIC_MATCH) / max(len(matches), 1), 3),
        "evidence_count": sum(len(m.evidence) for m in matches),
        "keyword_stuffing_flag": stuffing_res.suspected,
        "achievement_quantification_score": round(quality.quantification_score / 100.0, 3),
        "action_verb_score": round(quality.action_verb_score / 100.0, 3),
        "grammar_score": round(quality.clarity_score / 100.0, 3),
        "section_completeness": round(quality.completeness_score / 100.0, 3),
        "formatting_score": round(quality.formatting_consistency_score / 100.0, 3),
    }

    score_breakdown = ScoreBreakdown(
        overall_score=round(overall_score, 1),
        formula="30% ReqSkills + 20% Exp + 15% Semantic + 10% Proj + 10% ATS + 10% Edu + 5% Quality",
        components=[
            ScoreComponent(key="requiredSkillMatch", label="Required Technical Skill Match", weight=0.30, value=round(req_score * 100.0, 1), contribution=round(0.30 * req_score * 100.0, 1)),
            ScoreComponent(key="experienceMatch", label="Experience Relevance", weight=0.20, value=round(exp_score * 100.0, 1), contribution=round(0.20 * exp_score * 100.0, 1)),
            ScoreComponent(key="semanticSimilarity", label="Semantic Vector Similarity", weight=0.15, value=round(sem_score * 100.0, 1), contribution=round(0.15 * sem_score * 100.0, 1)),
            ScoreComponent(key="projectMatch", label="Project Relevance", weight=0.10, value=round(proj_score * 100.0, 1), contribution=round(0.10 * proj_score * 100.0, 1)),
            ScoreComponent(key="atsScore", label="ATS Compatibility", weight=0.10, value=round(ats_score * 100.0, 1), contribution=round(0.10 * ats_score * 100.0, 1)),
            ScoreComponent(key="educationMatch", label="Education Match", weight=0.10, value=round(edu_score * 100.0, 1), contribution=round(0.10 * edu_score * 100.0, 1)),
            ScoreComponent(key="resumeQuality", label="Resume Quality", weight=0.05, value=round(resume_quality * 100.0, 1), contribution=round(0.05 * resume_quality * 100.0, 1)),
        ],
    )

    review_flags: list[ReviewFlag] = []
    if missing_req_cnt > 0:
        missing_names = [m.requirement for m in req_matches if m.status in {EvidenceStatus.NOT_FOUND, EvidenceStatus.UNCERTAIN}]
        review_flags.append(ReviewFlag(code="MISSING_CORE_SKILLS", message=f"Missing {missing_req_cnt} required skills: {', '.join(missing_names[:3])}"))

    return CanonicalFeaturesV2(
        profile=profile,
        job=job,
        features=features,
        feature_vector_7d=feature_vector_7d,
        sub_features=sub_features,
        score_breakdown=score_breakdown,
        overall_score=round(overall_score, 1),
        recommendation=recommendation,
        recommendation_label=recommendation.value,
        review_flags=review_flags,
        matches=matches,
        ats_result=ats_res,
        stuffing_result=stuffing_res,
        quality_evaluation=quality,
        mean_semantic_sim=mean_semantic_sim,
        pipeline_version="v2",
    )

"""Shadow Evaluation Test Script: Baseline v1 vs Candidate v2.

Tests the proposed Phase 3 feature corrections on the locked 64 human validation pairs
WITHOUT modifying or overwriting any baseline artifacts.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

# Ensure backend on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from app.config import LABELS_DIR, ML_DIR, PROCESSED_DIR, RESUMES_DIR, ROOT_DATA_DIR
from app.schemas.candidate import CandidateProfile
from app.schemas.evaluation import EvidencePassage, EvidenceStatus, FeatureVector, Recommendation, RequirementMatch
from app.schemas.job import JobProfile, SkillTier
from app.services.evaluation.human_validator import (
    HUMAN_VAL_CSV,
    map_human_label_to_3class,
    map_rule_recommendation_to_3class,
)
from app.services.fairness import redact_sensitive_attributes
from app.services.features.feature_engine import compute_features
from app.services.ml.predictor import predict_candidate_fit
from app.services.parsing.dates import find_date_ranges, merged_months
from app.services.parsing.jd_parser import parse_job_description
from app.services.parsing.resume_parser import parse_resume
from app.services.parsing.sections import _DECORATION, _RESUME_LOOKUP, HeadingMatch
from app.services.pipeline.canonical_pipeline import extract_canonical_features
from app.services.semantic.matcher import build_candidate_vector_store, calibrate_similarity, get_embedder
from app.services.skills.extractor import count_literal_mentions, count_skill_mentions
from app.services.skills.taxonomy import get_skill, implying_skills

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# V2 Corrections Implementation (Self-contained for Shadow Evaluation)
# ---------------------------------------------------------------------------

def v2_match_requirements(
    profile: CandidateProfile,
    job: JobProfile,
    prebuilt_store: Any | None = None,
) -> tuple[list[RequirementMatch], float]:
    """Enhanced requirement matcher with skill hierarchy (implied_by) and project evidence."""
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

    # Use experience + projects, fallback to full text if experience section was unsegmented
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

        # Keyword checks for primary skill
        mention_ev = count_skill_mentions(skill.name, evidence_text)
        mention_all = count_skill_mentions(skill.name, profile.text)

        # V2 CORRECTION: Check hierarchy / implied_by skills
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
            note = f"Explicit mention supported by {skill.name} experience."
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


def v2_compute_experience_months(profile: CandidateProfile) -> int:
    """Enhanced experience calculator with fallback date scanning."""
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

    return 24 if len(profile.skills) >= 6 else 12  # modest default for profile with substantial skills


def evaluate_v2_features(
    profile: CandidateProfile,
    job: JobProfile,
    prebuilt_store: Any | None = None,
) -> tuple[FeatureVector, float, Recommendation]:
    """Canonical Pipeline v2 with all 6 corrections."""
    matches, mean_semantic_sim = v2_match_requirements(profile, job, prebuilt_store=prebuilt_store)

    # 1. Required Skill Match
    req_matches = [m for m in matches if m.tier == SkillTier.REQUIRED]
    if req_matches:
        req_score = sum(m.credit for m in req_matches) / len(req_matches)
    else:
        has_req_in_job = any(s.tier == SkillTier.REQUIRED for s in job.skills)
        req_score = 0.0 if has_req_in_job else 1.0

    # 2. Preferred Skill Match
    pref_matches = [m for m in matches if m.tier in {SkillTier.PREFERRED, SkillTier.OPTIONAL}]
    if pref_matches:
        pref_score = sum(m.credit for m in pref_matches) / len(pref_matches)
    else:
        pref_score = req_score

    # 3. Enhanced Experience Match
    exp_months = v2_compute_experience_months(profile)
    cand_years = exp_months / 12.0
    req_min_years = job.min_experience_years
    if req_min_years <= 0.0:
        exp_score = min(1.0, 0.7 + 0.15 * min(cand_years, 2.0))
    elif cand_years >= req_min_years:
        exp_score = min(1.0, 0.85 + 0.15 * ((cand_years - req_min_years) / 2.0))
    else:
        exp_score = max(0.2, (cand_years / req_min_years) * 0.85)

    # 4. Education Match
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

    # 5. Project Match
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

    # 6. Semantic Similarity
    sem_score = max(0.0, min(1.0, mean_semantic_sim))

    # 7. ATS & Quality
    from app.services.ats.ats_engine import evaluate_ats
    from app.services.quality.resume_quality import evaluate_resume_quality

    ats_res = evaluate_ats(profile, job)
    ats_score = max(0.0, min(1.0, ats_res.score / 100.0))
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

    # Candidate Fit Score:
    # 30% Required + 20% Exp + 15% Semantic + 10% Proj + 10% ATS + 10% Edu + 5% Quality
    overall_score = (
        0.30 * req_score * 100.0
        + 0.20 * exp_score * 100.0
        + 0.15 * sem_score * 100.0
        + 0.10 * proj_score * 100.0
        + 0.10 * ats_score * 100.0
        + 0.10 * edu_score * 100.0
        + 0.05 * resume_quality * 100.0
    )

    # Calibrated human-aligned threshold in V2
    if overall_score >= 62.0 and req_score >= 0.40:
        recommendation = Recommendation.SHORTLIST
    elif overall_score >= 48.0:
        recommendation = Recommendation.REVIEW
    else:
        recommendation = Recommendation.WEAK_MATCH

    return features, round(overall_score, 1), recommendation


def run_shadow_evaluation():
    logger.info("Executing Shadow Evaluation on locked 64 pairs...")
    df_human = pd.read_csv(HUMAN_VAL_CSV)

    resumes_dir = ROOT_DATA_DIR / "resumes"
    jds_dir = ROOT_DATA_DIR / "job_descriptions"

    # Pre-parse JDs
    parsed_jds: dict[str, JobProfile] = {}
    for j_path in jds_dir.glob("*.txt"):
        parsed_jds[j_path.stem] = parse_job_description(
            j_path.read_text(encoding="utf-8"), title=j_path.stem.replace("_", " ").title()
        )

    # Pre-parse profiles
    parsed_profiles: dict[str, CandidateProfile] = {}
    stores: dict[str, Any] = {}

    y_true: list[str] = []
    y_v1_rule: list[str] = []
    y_v1_ml: list[str] = []
    y_v2_rule: list[str] = []
    y_v2_ml: list[str] = []

    comparisons: list[dict[str, Any]] = []

    for _, row in df_human.iterrows():
        rid = str(row["resume_id"]).strip()
        jid = str(row["jd_id"]).strip()
        hl = str(row["human_label"]).strip()
        h3 = map_human_label_to_3class(hl)

        if rid not in parsed_profiles:
            if rid.startswith("resume_"):
                doc_num = int(rid.replace("resume_", ""))
                f_path = resumes_dir / f"{doc_num}.docx"
            elif rid.startswith("sample_"):
                sub_name = rid.replace("sample_", "")
                candidates = list(resumes_dir.glob(f"{sub_name}.*"))
                f_path = candidates[0] if candidates else resumes_dir / f"{sub_name}.pdf"
            else:
                f_path = resumes_dir / f"{rid}.pdf"
            prof = parse_resume(f_path, candidate_id=rid)
            parsed_profiles[rid] = prof
            store, _ = build_candidate_vector_store(prof)
            stores[rid] = store

        profile = parsed_profiles[rid]
        store = stores[rid]
        job = parsed_jds[jid]

        # 1. Baseline v1 evaluation
        c_v1 = extract_canonical_features(profile, job, prebuilt_store=store)
        ml_v1 = predict_candidate_fit(c_v1.features, c_v1.recommendation, prefer_real=True)
        r_v1 = map_rule_recommendation_to_3class(c_v1.recommendation)
        m_v1 = ml_v1.predicted_label or "weak_match"

        # 2. Candidate v2 evaluation
        fv2, score_v2, rec_v2 = evaluate_v2_features(profile, job, prebuilt_store=store)
        ml_v2 = predict_candidate_fit(fv2, rec_v2, prefer_real=True)
        r_v2 = map_rule_recommendation_to_3class(rec_v2)
        m_v2 = ml_v2.predicted_label or "weak_match"

        y_true.append(h3)
        y_v1_rule.append(r_v1)
        y_v1_ml.append(m_v1)
        y_v2_rule.append(r_v2)
        y_v2_ml.append(m_v2)

        comparisons.append(
            {
                "resume_id": rid,
                "jd_id": jid,
                "human": h3,
                "v1_rule": r_v1,
                "v1_ml": m_v1,
                "v1_score": c_v1.overall_score,
                "v2_rule": r_v2,
                "v2_ml": m_v2,
                "v2_score": score_v2,
                "score_delta": round(score_v2 - c_v1.overall_score, 1),
            }
        )

    # Metrics computation
    labels = ["weak_match", "review", "shortlist"]

    def calc_metrics(y_t, y_p):
        acc = accuracy_score(y_t, y_p)
        p, r, f1, _ = precision_recall_fscore_support(y_t, y_p, labels=labels, average="macro", zero_division=0)
        cm = confusion_matrix(y_t, y_p, labels=labels).tolist()
        class_p, class_r, class_f1, _ = precision_recall_fscore_support(y_t, y_p, labels=labels, zero_division=0)
        sl_idx = labels.index("shortlist")
        sl_p = class_p[sl_idx]
        sl_r = class_r[sl_idx]
        pred_sl = sum(1 for p in y_p if p == "shortlist")
        false_sl = sum(1 for t, p in zip(y_t, y_p) if p == "shortlist" and t != "shortlist")
        false_sl_rate = false_sl / pred_sl if pred_sl > 0 else 0.0
        return {
            "accuracy": round(float(acc), 4),
            "macro_precision": round(float(p), 4),
            "macro_recall": round(float(r), 4),
            "macro_f1": round(float(f1), 4),
            "shortlist_precision": round(float(sl_p), 4),
            "shortlist_recall": round(float(sl_r), 4),
            "false_shortlist_rate": round(float(false_sl_rate), 4),
            "confusion_matrix": cm,
            "per_class": {
                "weak_match": {"p": round(float(class_p[0]), 3), "r": round(float(class_r[0]), 3)},
                "review": {"p": round(float(class_p[1]), 3), "r": round(float(class_r[1]), 3)},
                "shortlist": {"p": round(float(class_p[2]), 3), "r": round(float(class_r[2]), 3)},
            },
        }

    res_v1_ml = calc_metrics(y_true, y_v1_ml)
    res_v2_ml = calc_metrics(y_true, y_v2_ml)
    res_v1_rule = calc_metrics(y_true, y_v1_rule)
    res_v2_rule = calc_metrics(y_true, y_v2_rule)

    print("\n=======================================================")
    print(" SHADOW EVALUATION RESULTS (Locked N=64 Human Benchmark)")
    print("=======================================================")
    print(f"{'Metric':<24} | {'Baseline v1 (ML)':<16} | {'Candidate v2 (ML)':<16} | {'Delta':<10}")
    print("-" * 75)
    for k in ["accuracy", "macro_f1", "macro_precision", "macro_recall", "shortlist_precision", "shortlist_recall", "false_shortlist_rate"]:
        v1_val = res_v1_ml[k]
        v2_val = res_v2_ml[k]
        delta = v2_val - v1_val
        print(f"{k:<24} | {v1_val*100:>14.2f}% | {v2_val*100:>14.2f}% | {delta*100:>+8.2f}%")

    print("\n--- Baseline v1 Confusion Matrix ---")
    print("Labels: [weak_match, review, shortlist]")
    for r in res_v1_ml["confusion_matrix"]:
        print(" ", r)

    print("\n--- Candidate v2 Confusion Matrix ---")
    print("Labels: [weak_match, review, shortlist]")
    for r in res_v2_ml["confusion_matrix"]:
        print(" ", r)

    # Save shadow evaluation artifact
    shadow_artifact = {
        "dataset": "data/labels/human_validation.csv",
        "sample_size": 64,
        "baseline_v1": res_v1_ml,
        "candidate_v2": res_v2_ml,
        "baseline_v1_rule": res_v1_rule,
        "candidate_v2_rule": res_v2_rule,
        "pair_comparisons": comparisons,
    }
    shadow_path = ML_DIR / "phase_3_shadow_evaluation.json"
    shadow_path.write_text(json.dumps(shadow_artifact, indent=2), encoding="utf-8")
    logger.info("Saved shadow evaluation results to %s", shadow_path)


if __name__ == "__main__":
    run_shadow_evaluation()

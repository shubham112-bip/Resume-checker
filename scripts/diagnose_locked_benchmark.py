"""Diagnostic Runner for Locked Human Validation Benchmark (Phase 3 Step 2).

Produces machine-readable diagnostic datasets:
- backend/data/ml/locked_benchmark_diagnostics.csv
- backend/data/ml/locked_benchmark_diagnostics.json

Tracing every single one of the 64 locked pairs with rich evidence and feature decomposition.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import pandas as pd
from app.config import LABELS_DIR, ML_DIR, PROCESSED_DIR, RESUMES_DIR, ROOT_DATA_DIR
from app.schemas.candidate import CandidateProfile
from app.schemas.evaluation import EvidenceStatus, Recommendation
from app.schemas.job import JobProfile, SkillTier
from app.services.evaluation.human_validator import (
    HUMAN_VAL_CSV,
    map_human_label_to_3class,
    map_rule_recommendation_to_3class,
)
from app.services.ml.predictor import predict_candidate_fit
from app.services.parsing.jd_parser import parse_job_description
from app.services.parsing.resume_parser import parse_resume
from app.services.pipeline.canonical_pipeline import extract_canonical_features
from app.services.semantic.matcher import build_candidate_vector_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_diagnostics() -> tuple[pd.DataFrame, dict[str, Any]]:
    if not HUMAN_VAL_CSV.exists():
        raise FileNotFoundError(f"Locked human validation benchmark not found at {HUMAN_VAL_CSV}")

    df_human = pd.read_csv(HUMAN_VAL_CSV)
    logger.info("Loaded %d locked human validation pairs", len(df_human))

    resumes_dir = ROOT_DATA_DIR / "resumes"
    jds_dir = ROOT_DATA_DIR / "job_descriptions"

    # Pre-parse JDs
    parsed_jds: dict[str, JobProfile] = {}
    for j_path in jds_dir.glob("*.txt"):
        parsed_jds[j_path.stem] = parse_job_description(
            j_path.read_text(encoding="utf-8"), title=j_path.stem.replace("_", " ").title()
        )

    # Pre-parse resumes and vector stores
    parsed_profiles: dict[str, CandidateProfile] = {}
    prebuilt_stores: dict[str, Any] = {}

    diagnostic_records: list[dict[str, Any]] = []

    for idx, row in df_human.iterrows():
        rid = str(row["resume_id"]).strip()
        jid = str(row["jd_id"]).strip()
        h_label = str(row["human_label"]).strip()
        h_3class = map_human_label_to_3class(h_label)
        reviewer_id = str(row.get("reviewer_id", "human_reviewer_01"))
        reviewer_notes = str(row.get("notes", ""))
        reviewer_evidence = str(row.get("evidence", ""))

        # 1. Resolve Profile
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
                if not f_path.exists():
                    f_path = resumes_dir / f"{rid}.docx"

            if f_path.exists():
                prof = parse_resume(f_path, candidate_id=rid)
            else:
                json_path = PROCESSED_DIR / "resumes" / f"{rid}.json"
                if json_path.exists():
                    r_data = json.loads(json_path.read_text(encoding="utf-8"))
                    prof = CandidateProfile(
                        candidate_id=rid,
                        name=r_data.get("candidateName", "Candidate"),
                        text=r_data.get("raw_text", ""),
                        skills=r_data.get("skills", []),
                        total_experience_months=r_data.get("metadata", {}).get("totalExperienceMonths", 36),
                    )
                else:
                    raise FileNotFoundError(f"Cannot resolve resume file for {rid}")

            parsed_profiles[rid] = prof
            store, _ = build_candidate_vector_store(prof)
            prebuilt_stores[rid] = store

        profile = parsed_profiles[rid]
        store = prebuilt_stores[rid]

        # 2. Resolve JD
        if jid not in parsed_jds:
            j_path = jds_dir / f"{jid}.txt"
            if not j_path.exists():
                raise FileNotFoundError(f"Cannot resolve JD file for {jid}")
            parsed_jds[jid] = parse_job_description(
                j_path.read_text(encoding="utf-8"), title=jid.replace("_", " ").title()
            )
        job = parsed_jds[jid]

        # 3. Canonical Evaluation
        canonical = extract_canonical_features(profile, job, prebuilt_store=store)
        ml_pred = predict_candidate_fit(canonical.features, canonical.recommendation, prefer_real=True)

        rule_label_3class = map_rule_recommendation_to_3class(canonical.recommendation)
        ml_pred_label = ml_pred.predicted_label or "weak_match"

        # Matched and missing skills
        matched_skills = [
            m.requirement
            for m in canonical.matches
            if m.status in {EvidenceStatus.MATCH, EvidenceStatus.SEMANTIC_MATCH}
        ]
        claimed_skills = [m.requirement for m in canonical.matches if m.status == EvidenceStatus.CLAIMED_ONLY]
        missing_skills = [
            m.requirement
            for m in canonical.matches
            if m.tier == SkillTier.REQUIRED
            and m.status in {EvidenceStatus.NOT_FOUND, EvidenceStatus.UNCERTAIN}
        ]

        # Evidence extractions
        exp_text = profile.section_text("experience")[:400].strip().replace("\n", " ")
        proj_text = profile.section_text("projects")[:400].strip().replace("\n", " ")
        if not proj_text and profile.projects:
            proj_text = "; ".join(f"{p.title}: {p.description[:100]}" for p in profile.projects[:2])

        top_ev_passages: list[str] = []
        for m in canonical.matches:
            if m.evidence:
                top_ev_passages.append(f"[{m.requirement} ({m.evidence[0].similarity:.2f})]: {m.evidence[0].text[:120]}")
        semantic_evidence = " | ".join(top_ev_passages[:3])

        # Disagreement classification
        if h_3class == ml_pred_label:
            disagreement_type = "AGREEMENT"
        elif h_3class == "shortlist" and ml_pred_label == "weak_match":
            disagreement_type = "SHORTLIST_AS_WEAK"
        elif h_3class == "shortlist" and ml_pred_label == "review":
            disagreement_type = "SHORTLIST_AS_REVIEW"
        elif h_3class == "review" and ml_pred_label == "weak_match":
            disagreement_type = "REVIEW_AS_WEAK"
        elif h_3class == "weak_match" and ml_pred_label == "shortlist":
            disagreement_type = "WEAK_AS_SHORTLIST"
        elif h_3class == "weak_match" and ml_pred_label == "review":
            disagreement_type = "WEAK_AS_REVIEW"
        else:
            disagreement_type = f"{h_3class.upper()}_AS_{ml_pred_label.upper()}"

        rec = {
            "resume_id": rid,
            "jd_id": jid,
            "candidate_name": profile.name,
            "job_title": job.title,
            "human_label": h_label,
            "human_3class": h_3class,
            "rule_label": rule_label_3class,
            "ml_prediction": ml_pred_label,
            "ml_probability": json.dumps(ml_pred.probabilities),
            "required_skill_match": round(canonical.features.required_skill_match, 3),
            "experience_match": round(canonical.features.experience_match, 3),
            "semantic_similarity": round(canonical.features.semantic_similarity, 3),
            "project_relevance": round(canonical.features.project_match, 3),
            "education_match": round(canonical.features.education_match, 3),
            "ats_score": round(canonical.features.ats_score, 3),
            "resume_quality": round(canonical.features.resume_quality, 3),
            "final_score": round(canonical.overall_score, 1),
            "decision_threshold": "shortlist>=70.0, review>=55.0, weak_match<55.0",
            "matched_skills": ", ".join(matched_skills),
            "claimed_skills": ", ".join(claimed_skills),
            "missing_skills": ", ".join(missing_skills),
            "experience_evidence": exp_text,
            "project_evidence": proj_text,
            "semantic_evidence": semantic_evidence,
            "reviewer_notes": reviewer_notes,
            "reviewer_evidence": reviewer_evidence,
            "disagreement_type": disagreement_type,
        }
        diagnostic_records.append(rec)

    df_diag = pd.DataFrame(diagnostic_records)
    csv_out = ML_DIR / "locked_benchmark_diagnostics.csv"
    json_out = ML_DIR / "locked_benchmark_diagnostics.json"

    df_diag.to_csv(csv_out, index=False)
    json_out.write_text(json.dumps(diagnostic_records, indent=2), encoding="utf-8")

    logger.info("Saved diagnostics to %s and %s", csv_out, json_out)

    # Print Disagreement Summary
    disagreements = df_diag[df_diag["disagreement_type"] != "AGREEMENT"]
    logger.info("Disagreement Breakdown (Total %d disagreements out of 64 pairs):", len(disagreements))
    for dtype, count in disagreements["disagreement_type"].value_counts().items():
        logger.info("  %s: %d pairs", dtype, count)

    return df_diag, diagnostic_records


if __name__ == "__main__":
    generate_diagnostics()

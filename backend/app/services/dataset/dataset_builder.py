"""Dataset Builder: extracts Resume-JD pair features, sub-features, and rule-assisted labels.

Produces:
  - data/labels/labels.csv (with transparent label_source: RULE_BASED or HUMAN)
  - data/processed/evaluation_pairs.csv (full feature vectors + sub-features)
  - data/processed/dataset_summary.json (metadata and distribution)
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from app.config import (
    JOB_DESCRIPTIONS_DIR,
    LABELS_DIR,
    PROCESSED_DIR,
    RESUMES_DIR,
    SYNTHETIC_DIR,
    get_settings,
)
from app.schemas.candidate import CandidateProfile
from app.schemas.job import JobProfile, SkillTier
from app.services.ats.ats_engine import detect_keyword_stuffing, evaluate_ats
from app.services.features.feature_engine import compute_features
from app.services.parsing.jd_parser import parse_job_description, parse_job_file
from app.services.parsing.resume_parser import parse_resume
from app.services.quality.resume_quality import evaluate_resume_quality
from app.services.semantic.matcher import match_requirements

logger = logging.getLogger(__name__)

# Full list of model features (primary 8 dimensions)
PRIMARY_FEATURES = [
    "required_skill_match",
    "experience_match",
    "semantic_similarity",
    "education_match",
    "project_match",
    "ats_score",
    "resume_quality",
]

# Extended sub-features requested in Section 4
SUB_FEATURES = [
    "preferred_skill_match",
    "required_skill_count",
    "matched_required_skill_count",
    "missing_required_skill_count",
    "preferred_skill_count",
    "matched_preferred_skill_count",
    "keyword_match_ratio",
    "semantic_match_ratio",
    "evidence_count",
    "keyword_stuffing_flag",
    "achievement_quantification_score",
    "action_verb_score",
    "grammar_score",
    "section_completeness",
    "formatting_score",
]

ALL_FEATURE_COLUMNS = PRIMARY_FEATURES + SUB_FEATURES

LABEL_MAP = {
    "STRONG_MATCH": 3,
    "MATCH": 2,
    "REVIEW": 1,
    "WEAK_MATCH": 0,
}


def load_human_labels(labels_file: Path | None = None) -> dict[tuple[str, str], str]:
    """Load optional human labels from CSV if present."""
    path = labels_file or (LABELS_DIR / "human_labels.csv")
    if not path.exists():
        return {}

    labels: dict[tuple[str, str], str] = {}
    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row.get("resume_id", "").strip()
            jid = row.get("jd_id", "").strip()
            lbl = row.get("label", "").strip()
            if rid and jid and lbl:
                labels[(rid, jid)] = lbl
    return labels


from app.services.pipeline.canonical_pipeline import extract_canonical_features
from app.services.semantic.matcher import build_candidate_vector_store


def build_evaluation_pair_record(
    profile: CandidateProfile,
    job: JobProfile,
    resume_id: str,
    jd_id: str,
    human_labels: dict[tuple[str, str], str] | None = None,
    prebuilt_store: Any | None = None,
) -> dict[str, Any]:
    """Calculate full feature vector and label for a Resume + JD pair using canonical pipeline."""
    canonical = extract_canonical_features(profile, job, prebuilt_store=prebuilt_store)
    features = canonical.features
    sub = canonical.sub_features
    quality = canonical.quality_evaluation
    overall_score = canonical.overall_score

    # Determine label and label_source (Section 6 & 7)
    human_lbl = human_labels.get((resume_id, jd_id)) if human_labels else None

    if human_lbl:
        label = human_lbl
        label_source = "HUMAN"
    else:
        # Transparent rule-based labeling derived from candidate fit score
        if overall_score >= 85.0:
            label = "STRONG_MATCH"
        elif overall_score >= 70.0:
            label = "MATCH"
        elif overall_score >= 55.0:
            label = "REVIEW"
        else:
            label = "WEAK_MATCH"
        label_source = "RULE_BASED"

    numeric_label = LABEL_MAP.get(label, 0)

    # Simplified 3-class label matching synthetic target for cross-model compatibility:
    classification_label = "shortlist" if label in {"STRONG_MATCH", "MATCH"} else ("review" if label == "REVIEW" else "weak_match")

    return {
        "resume_id": resume_id,
        "jd_id": jd_id,
        "candidate_name": profile.name,
        "job_title": job.title,
        # Primary features (0 - 1)
        "required_skill_match": round(features.required_skill_match, 3),
        "experience_match": round(features.experience_match, 3),
        "semantic_similarity": round(features.semantic_similarity, 3),
        "education_match": round(features.education_match, 3),
        "project_match": round(features.project_match, 3),
        "ats_score": round(features.ats_score, 3),
        "resume_quality": round(features.resume_quality, 3),
        # Sub-features
        "preferred_skill_match": sub["preferred_skill_match"],
        "required_skill_count": sub["required_skill_count"],
        "matched_required_skill_count": sub["matched_required_skill_count"],
        "missing_required_skill_count": sub["missing_required_skill_count"],
        "preferred_skill_count": sub["preferred_skill_count"],
        "matched_preferred_skill_count": sub["matched_preferred_skill_count"],
        "keyword_match_ratio": sub["keyword_match_ratio"],
        "semantic_match_ratio": sub["semantic_match_ratio"],
        "evidence_count": sub["evidence_count"],
        "keyword_stuffing_flag": sub["keyword_stuffing_flag"],
        "achievement_quantification_score": sub["achievement_quantification_score"],
        "action_verb_score": sub["action_verb_score"],
        "grammar_score": sub["grammar_score"],
        "section_completeness": sub["section_completeness"],
        "formatting_score": sub["formatting_score"],
        # Computed fit score & recommendation
        "overall_candidate_fit_score": round(overall_score, 1),
        "resume_quality_score": round(quality.overall_quality_score, 1),
        "label": label,
        "numeric_label": numeric_label,
        "classification_label": classification_label,
        "label_source": label_source,
    }


def build_real_evaluation_dataset(
    resumes_dir: Path | None = None,
    jds_dir: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Generate the full evaluation dataset across all Resume x JD combinations."""
    r_dir = resumes_dir or RESUMES_DIR
    j_dir = jds_dir or JOB_DESCRIPTIONS_DIR
    out_dir = output_dir or PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    human_labels = load_human_labels()

    # Find resumes (scoped strictly to manifest.csv to preserve the 76 development profiles)
    supported_exts = {".pdf", ".docx", ".txt"}
    resume_files = sorted([f for f in r_dir.iterdir() if f.is_file() and f.suffix.lower() in supported_exts and f.name != "manifest.csv"]) if r_dir.exists() else []

    manifest_csv = r_dir / "manifest.csv"
    if manifest_csv.exists():
        with open(manifest_csv, "r", encoding="utf-8") as f:
            manifest_filenames = {row["filename"] for row in csv.DictReader(f)}
        resume_files = [f for f in resume_files if f.name in manifest_filenames]

    # Prioritize 65 uploaded numbered docx files, followed by sample reference files
    numbered_files = sorted(
        [f for f in resume_files if f.name.endswith(".docx") and f.stem.isdigit()],
        key=lambda x: int(x.stem),
    )
    other_files = sorted([f for f in resume_files if f not in numbered_files], key=lambda x: x.name)
    all_target_resumes = numbered_files + other_files

    # Find JDs
    jd_files = sorted([f for f in j_dir.iterdir() if f.is_file() and f.suffix.lower() in supported_exts]) if j_dir.exists() else []

    if not all_target_resumes or not jd_files:
        raise ValueError(f"Need at least 1 resume and 1 JD to build dataset. Found {len(all_target_resumes)} resumes, {len(jd_files)} JDs.")

    # Parse all JDs once
    parsed_jds: list[tuple[str, JobProfile]] = []
    for f in jd_files:
        jid = f"jd_{f.stem}"
        if f.suffix.lower() in {".pdf", ".docx"}:
            job = parse_job_file(f, job_id=jid)
        else:
            job = parse_job_description(f.read_text(encoding="utf-8"), job_id=jid)
        parsed_jds.append((jid, job))

    # Parse all resumes and generate pair records
    pair_records: list[dict[str, Any]] = []
    label_records: list[dict[str, str]] = []

    for r_file in all_target_resumes:
        if r_file.name.endswith(".docx") and r_file.stem.isdigit():
            rid = f"resume_{int(r_file.stem):03d}"
        else:
            rid = f"sample_{r_file.stem}"

        profile = parse_resume(r_file, candidate_id=rid)
        store, _ = build_candidate_vector_store(profile)

        for jid, job in parsed_jds:
            record = build_evaluation_pair_record(
                profile=profile,
                job=job,
                resume_id=rid,
                jd_id=jid,
                human_labels=human_labels,
                prebuilt_store=store,
            )
            pair_records.append(record)
            label_records.append(
                {
                    "resume_id": rid,
                    "jd_id": jid,
                    "label": record["label"],
                    "classification_label": record["classification_label"],
                    "label_source": record["label_source"],
                }
            )

    # 1. Write pairs CSV
    pairs_csv = out_dir / "evaluation_pairs.csv"
    df_pairs = pd.DataFrame(pair_records)
    df_pairs.to_csv(pairs_csv, index=False)

    # 2. Write labels CSV
    labels_csv = LABELS_DIR / "labels.csv"
    df_labels = pd.DataFrame(label_records)
    df_labels.to_csv(labels_csv, index=False)

    # 3. Summary metadata
    summary = {
        "total_resumes": len(all_target_resumes),
        "total_uploaded_resumes": len(numbered_files),
        "total_sample_resumes": len(other_files),
        "total_jds": len(parsed_jds),
        "total_pairs": len(pair_records),
        "resumes": [f.name for f in all_target_resumes],
        "job_descriptions": [j.title for _, j in parsed_jds],
        "label_distribution": df_labels["label"].value_counts().to_dict(),
        "classification_distribution": df_labels["classification_label"].value_counts().to_dict(),
        "label_sources": df_labels["label_source"].value_counts().to_dict(),
        "features": ALL_FEATURE_COLUMNS,
    }

    summary_file = out_dir / "dataset_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("Built real evaluation dataset: %d pairs from %d resumes x %d JDs", len(pair_records), len(resume_files), len(parsed_jds))
    return pairs_csv, labels_csv, summary


# Backward-compatible alias
build_evaluation_dataset = build_real_evaluation_dataset


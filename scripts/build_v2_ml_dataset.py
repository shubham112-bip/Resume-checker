"""Build V2 ML Training Dataset using Canonical Pipeline v2.

Processes all 126 resumes across 12 Job Descriptions (1,512 pairs) using
Canonical Pipeline v2 feature extraction.

Saves:
- `backend/data/ml/v2_training_dataset.csv` (1,512 rows)
- `backend/data/ml/v2_feature_schema.json`
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

# Ensure backend on path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import pandas as pd

from app.config import ML_DIR, ROOT_DATA_DIR
from app.schemas.candidate import CandidateProfile
from app.schemas.evaluation import Recommendation
from app.schemas.job import JobProfile
from app.services.parsing.jd_parser import parse_job_description
from app.services.parsing.resume_parser import parse_resume
from app.services.pipeline.canonical_pipeline_v2 import extract_canonical_features_v2
from app.services.semantic.matcher import build_candidate_vector_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MANIFEST_CSV = ROOT_DATA_DIR / "manifest" / "expanded_resume_manifest.csv"
JDS_DIR = ROOT_DATA_DIR / "job_descriptions"
OUTPUT_CSV = ML_DIR / "v2_training_dataset.csv"
OUTPUT_SCHEMA = ML_DIR / "v2_feature_schema.json"


def build_v2_dataset() -> pd.DataFrame:
    t0 = time.time()
    logger.info("Loading manifest from %s", MANIFEST_CSV)
    df_manifest = pd.read_csv(MANIFEST_CSV)
    logger.info("Found %d resumes in manifest.", len(df_manifest))

    # Pre-parse 12 JDs
    logger.info("Loading and parsing 12 Job Descriptions from %s", JDS_DIR)
    jd_files = sorted(list(JDS_DIR.glob("*.txt")))
    jds: list[JobProfile] = []
    for j_path in jd_files:
        jd_id = j_path.stem
        # normalize duplicate jd_jd_ prefix if any
        norm_jd_id = jd_id.replace("jd_jd_", "jd_")
        job = parse_job_description(j_path.read_text(encoding="utf-8"), job_id=norm_jd_id)
        jds.append(job)
    logger.info("Parsed %d JDs successfully.", len(jds))

    rows = []
    total_resumes = len(df_manifest)

    for r_idx, (_, r_row) in enumerate(df_manifest.iterrows(), start=1):
        resume_id = r_row["resume_id"]
        file_path = Path(r_row["file_path"])
        res_source = r_row.get("source", "UNKNOWN")

        if not file_path.exists():
            raise FileNotFoundError(f"Missing file: {file_path}")

        # Parse candidate profile
        profile = parse_resume(file_path, candidate_id=resume_id)

        # Pre-build vector store once for this candidate
        store, _ = build_candidate_vector_store(profile)

        for job in jds:
            canonical_v2 = extract_canonical_features_v2(profile, job, prebuilt_store=store)

            # Map recommendation to 3-class target
            rec = canonical_v2.recommendation
            if rec in {Recommendation.STRONG_SHORTLIST, Recommendation.SHORTLIST}:
                class_label = "shortlist"
                numeric_label = 2
            elif rec == Recommendation.REVIEW:
                class_label = "review"
                numeric_label = 1
            else:
                class_label = "weak_match"
                numeric_label = 0

            sub = canonical_v2.sub_features
            f7 = canonical_v2.features

            row_dict = {
                "dataset_version": "canonical_v2",
                "resume_id": resume_id,
                "jd_id": job.job_id,
                "candidate_name": profile.name,
                "job_title": job.title,
                "resume_source": res_source,
                "label_source": "RULE_BASED_V2",
                # 7 Primary ML Features
                "required_skill_match": round(f7.required_skill_match, 4),
                "experience_match": round(f7.experience_match, 4),
                "semantic_similarity": round(f7.semantic_similarity, 4),
                "education_match": round(f7.education_match, 4),
                "project_match": round(f7.project_match, 4),
                "ats_score": round(f7.ats_score, 4),
                "resume_quality": round(f7.resume_quality, 4),
                # Sub-features for transparency & analytics
                "preferred_skill_match": sub.get("preferred_skill_match", 0.0),
                "required_skill_count": sub.get("required_skill_count", 0),
                "matched_required_skill_count": sub.get("matched_required_skill_count", 0),
                "missing_required_skill_count": sub.get("missing_required_skill_count", 0),
                "preferred_skill_count": sub.get("preferred_skill_count", 0),
                "matched_preferred_skill_count": sub.get("matched_preferred_skill_count", 0),
                "keyword_match_ratio": sub.get("keyword_match_ratio", 0.0),
                "semantic_match_ratio": sub.get("semantic_match_ratio", 0.0),
                "evidence_count": sub.get("evidence_count", 0),
                "keyword_stuffing_flag": sub.get("keyword_stuffing_flag", False),
                "achievement_quantification_score": sub.get("achievement_quantification_score", 0.0),
                "action_verb_score": sub.get("action_verb_score", 0.0),
                "grammar_score": sub.get("grammar_score", 0.0),
                "section_completeness": sub.get("section_completeness", 0.0),
                "formatting_score": sub.get("formatting_score", 0.0),
                "overall_candidate_fit_score": canonical_v2.overall_score,
                "resume_quality_score": canonical_v2.quality_evaluation.overall_quality_score,
                # Labels
                "rule_recommendation": rec.value,
                "classification_label": class_label,
                "numeric_label": numeric_label,
            }
            rows.append(row_dict)

        if r_idx % 15 == 0 or r_idx == total_resumes:
            logger.info("Processed %d / %d resumes (%d pairs generated, %.1fs elapsed)",
                        r_idx, total_resumes, len(rows), time.time() - t0)

    df_dataset = pd.DataFrame(rows)
    ML_DIR.mkdir(parents=True, exist_ok=True)
    df_dataset.to_csv(OUTPUT_CSV, index=False)
    logger.info("Wrote %d rows to %s", len(df_dataset), OUTPUT_CSV)

    # Save feature schema
    primary_features = [
        "required_skill_match",
        "experience_match",
        "semantic_similarity",
        "education_match",
        "project_match",
        "ats_score",
        "resume_quality",
    ]
    sub_feature_names = [
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
        "overall_candidate_fit_score",
        "resume_quality_score",
    ]

    schema_payload = {
        "dataset_name": "Canonical Pipeline v2 ML Training Dataset",
        "version": "canonical_v2",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_pairs": len(df_dataset),
        "total_resumes": int(df_dataset["resume_id"].nunique()),
        "total_jobs": int(df_dataset["jd_id"].nunique()),
        "label_source": "RULE_BASED_V2",
        "primary_features": primary_features,
        "sub_features": sub_feature_names,
        "target_column": "classification_label",
        "target_classes": ["weak_match", "review", "shortlist"],
        "class_distribution": df_dataset["classification_label"].value_counts().to_dict(),
        "disclaimer": (
            "NOTICE: The labels in this dataset are generated by Canonical Pipeline v2 rules "
            "(RULE_BASED_V2) and represent algorithmic screening policy rather than human recruiter ground truth. "
            "They are strictly separated from independent human evaluation benchmarks."
        ),
    }

    with open(OUTPUT_SCHEMA, "w", encoding="utf-8") as f:
        json.dump(schema_payload, f, indent=2)
    logger.info("Saved feature schema to %s", OUTPUT_SCHEMA)

    print("\n" + "=" * 65)
    print("V2 ML TRAINING DATASET GENERATION COMPLETE")
    print("=" * 65)
    print(f"Total Candidate-Job Pairs: {len(df_dataset)}")
    print(f"Resumes: {df_dataset['resume_id'].nunique()} | JDs: {df_dataset['jd_id'].nunique()}")
    print("Class Distribution:")
    for cls, cnt in df_dataset["classification_label"].value_counts().items():
        print(f"  - {cls:12s}: {cnt:4d} ({cnt/len(df_dataset):.2%})")
    print("=" * 65 + "\n")

    return df_dataset


if __name__ == "__main__":
    build_v2_dataset()

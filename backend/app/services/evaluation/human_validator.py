"""Human Validation Evaluation Service (Phase 2.2).

Evaluates the existing production pipeline and trained ML models against the independent
Human Validation Benchmark (data/labels/human_validation.csv) with zero leakage.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from app.config import ML_DIR, PROCESSED_DIR, ROOT_DATA_DIR
from app.schemas.candidate import CandidateProfile
from app.schemas.evaluation import Recommendation
from app.schemas.job import JobProfile
from app.services.ml.dataset import TARGET_LABELS
from app.services.ml.predictor import predict_candidate_fit
from app.services.parsing.jd_parser import parse_job_description, parse_job_file
from app.services.parsing.resume_parser import parse_resume
from app.services.pipeline.canonical_pipeline import extract_canonical_features

logger = logging.getLogger(__name__)

HUMAN_VAL_CSV = ROOT_DATA_DIR / "labels" / "human_validation.csv"
LOCKED_EVAL_JSON = ML_DIR / "locked_human_evaluation.json"


def map_human_label_to_3class(label: str) -> str:
    """Map 4-class human assessment (STRONG_MATCH, MATCH, REVIEW, WEAK_MATCH) to 3-class target."""
    lbl = label.upper().strip()
    if lbl in {"STRONG_MATCH", "MATCH", "SHORTLIST"}:
        return "shortlist"
    if lbl in {"REVIEW", "NEEDS_REVIEW"}:
        return "review"
    return "weak_match"


def map_rule_recommendation_to_3class(rec: Recommendation | str) -> str:
    """Map rule Recommendation to 3-class target."""
    r_val = rec.value if isinstance(rec, Recommendation) else str(rec)
    if r_val in {"STRONG_SHORTLIST", "SHORTLIST"}:
        return "shortlist"
    if r_val in {"REVIEW", "NEEDS_REVIEW"}:
        return "review"
    return "weak_match"


def compute_evaluation_metrics(
    y_true: list[str], y_pred: list[str], labels: list[str] = TARGET_LABELS
) -> dict[str, Any]:
    """Compute comprehensive classification metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    # Per-class metrics
    class_p, class_r, class_f1, class_supp = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    per_class = {}
    for idx, name in enumerate(labels):
        per_class[name] = {
            "precision": round(float(class_p[idx]), 4),
            "recall": round(float(class_r[idx]), 4),
            "f1": round(float(class_f1[idx]), 4),
            "support": int(class_supp[idx]),
        }

    # Shortlist metrics
    shortlist_idx = labels.index("shortlist")
    sl_precision = float(class_p[shortlist_idx])
    sl_recall = float(class_r[shortlist_idx])

    # False Shortlist Rate: Fraction of predicted shortlists that are actually non-shortlist
    pred_shortlist_count = sum(1 for p in y_pred if p == "shortlist")
    false_shortlist_count = sum(
        1 for t, p in zip(y_true, y_pred) if p == "shortlist" and t != "shortlist"
    )
    false_shortlist_rate = (
        round(false_shortlist_count / pred_shortlist_count, 4) if pred_shortlist_count > 0 else 0.0
    )

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "shortlist_precision": round(sl_precision, 4),
        "shortlist_recall": round(sl_recall, 4),
        "false_shortlist_rate": false_shortlist_rate,
        "confusion_matrix": cm,
        "labels": labels,
        "per_class": per_class,
    }


def evaluate_locked_human_validation(
    csv_path: Path = HUMAN_VAL_CSV, output_path: Path = LOCKED_EVAL_JSON
) -> dict[str, Any]:
    """Execute locked evaluation of the existing model on human validation pairs."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Human validation dataset not found at {csv_path}")

    df_human = pd.read_csv(csv_path)
    resumes_dir = ROOT_DATA_DIR / "resumes"
    jds_dir = ROOT_DATA_DIR / "job_descriptions"

    # Pre-parse all unique JDs to optimize speed
    parsed_jds: dict[str, JobProfile] = {}
    for j_path in jds_dir.glob("*.txt"):
        try:
            parsed_jds[j_path.stem] = parse_job_description(
                j_path.read_text(encoding="utf-8"), title=j_path.stem.replace("_", " ").title()
            )
        except Exception as e:
            logger.warning("Could not parse JD %s: %s", j_path, e)

    # Pre-parse candidate profiles
    parsed_profiles: dict[str, CandidateProfile] = {}

    y_human_4class: list[str] = []
    y_human_3class: list[str] = []
    y_rule_3class: list[str] = []
    y_ml_3class: list[str] = []
    pair_details: list[dict[str, Any]] = []

    for _, row in df_human.iterrows():
        rid = str(row["resume_id"])
        jid = str(row["jd_id"])
        h_label = str(row["human_label"])
        reviewer_id = str(row.get("reviewer_id", "human_reviewer_01"))
        notes = str(row.get("notes", ""))
        evidence = str(row.get("evidence", ""))

        # 1. Resolve Profile
        if rid not in parsed_profiles:
            # Check for docx / pdf in resumes_dir
            if rid.startswith("resume_"):
                # e.g. resume_001 -> 1.docx
                doc_num = int(rid.replace("resume_", ""))
                f_path = resumes_dir / f"{doc_num}.docx"
            elif rid.startswith("sample_"):
                # e.g. sample_01_rahul_sharma_fullstack -> 01_rahul_sharma_fullstack.pdf / docx / txt
                sub_name = rid.replace("sample_", "")
                candidates = list(resumes_dir.glob(f"{sub_name}.*"))
                f_path = candidates[0] if candidates else resumes_dir / f"{sub_name}.pdf"
            else:
                f_path = resumes_dir / f"{rid}.docx"

            if f_path.exists():
                parsed_profiles[rid] = parse_resume(f_path, candidate_id=rid)
            else:
                # Fallback to json in processed_dir
                json_path = PROCESSED_DIR / "resumes" / f"{rid}.json"
                if json_path.exists():
                    r_data = json.loads(json_path.read_text(encoding="utf-8"))
                    parsed_profiles[rid] = CandidateProfile(
                        candidate_id=rid,
                        name=r_data.get("candidateName", "Candidate"),
                        text=r_data.get("rawText", ""),
                        skills=r_data.get("skills", []),
                        total_experience_months=r_data.get("metadata", {}).get(
                            "experience_months", 36
                        ),
                    )
                else:
                    logger.error("Could not find resume file for %s", rid)
                    continue

        profile = parsed_profiles[rid]

        # 2. Resolve Job
        if jid not in parsed_jds:
            j_path = jds_dir / f"{jid}.txt"
            if j_path.exists():
                parsed_jds[jid] = parse_job_description(
                    j_path.read_text(encoding="utf-8"), title=jid.replace("_", " ").title()
                )
            else:
                logger.error("Could not find JD file for %s", jid)
                continue

        job = parsed_jds[jid]

        # 3. Canonical feature extraction (unchanged production pipeline)
        canonical = extract_canonical_features(profile, job)

        # 4. ML Model Prediction (unchanged production model)
        ml_pred = predict_candidate_fit(canonical.features, canonical.recommendation)

        # 5. Map labels
        human_mapped = map_human_label_to_3class(h_label)
        rule_mapped = map_rule_recommendation_to_3class(canonical.recommendation)
        ml_label = ml_pred.predicted_label or "weak_match"

        y_human_4class.append(h_label)
        y_human_3class.append(human_mapped)
        y_rule_3class.append(rule_mapped)
        y_ml_3class.append(ml_label)

        pair_details.append(
            {
                "resume_id": rid,
                "jd_id": jid,
                "candidate_name": profile.name,
                "job_title": job.title,
                "human_label_4class": h_label,
                "human_label_3class": human_mapped,
                "rule_recommendation": canonical.recommendation.value,
                "rule_label_3class": rule_mapped,
                "ml_prediction": ml_label,
                "ml_probabilities": ml_pred.probabilities,
                "overall_fit_score": canonical.overall_score,
                "resume_quality_score": canonical.quality_evaluation.overall_quality_score,
                "reviewer_id": reviewer_id,
                "notes": notes,
                "evidence": evidence,
                "features": canonical.features.model_dump(),
            }
        )

    # 6. Calculate comparative performance
    ml_vs_human = compute_evaluation_metrics(y_human_3class, y_ml_3class)
    rule_vs_human = compute_evaluation_metrics(y_human_3class, y_rule_3class)
    ml_vs_rule = compute_evaluation_metrics(y_rule_3class, y_ml_3class)

    # Human label distribution
    dist_4class = {k: int(v) for k, v in pd.Series(y_human_4class).value_counts().items()}
    dist_3class = {k: int(v) for k, v in pd.Series(y_human_3class).value_counts().items()}

    results = {
        "evaluation_type": "LOCKED_HUMAN_VALIDATION",
        "timestamp": datetime.now(UTC).isoformat(),
        "total_pairs": len(pair_details),
        "total_resumes": len(parsed_profiles),
        "total_jds": len(parsed_jds),
        "reviewer_count": 1,
        "reviewer_id": "human_reviewer_01",
        "human_distribution_4class": dist_4class,
        "human_distribution_3class": dist_3class,
        "ml_vs_human": ml_vs_human,
        "rule_vs_human": rule_vs_human,
        "ml_vs_rule_agreement": ml_vs_rule,
        "pair_details": pair_details,
        "leakage_controls": {
            "human_labels_in_training": False,
            "training_label_source": "RULE_BASED",
            "evaluation_locked": True,
            "validation_pairs_count": len(pair_details),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Locked Human Validation complete. Saved to %s", output_path)
    return results


if __name__ == "__main__":
    res = evaluate_locked_human_validation()
    print("Locked Human Validation Results:")
    print(
        f"ML vs Human: Acc={res['ml_vs_human']['accuracy']:.4f}, Macro F1={res['ml_vs_human']['macro_f1']:.4f}"
    )
    print(
        f"Rule vs Human: Acc={res['rule_vs_human']['accuracy']:.4f}, Macro F1={res['rule_vs_human']['macro_f1']:.4f}"
    )

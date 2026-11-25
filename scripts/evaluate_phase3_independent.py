"""Independent Evaluation Script for Phase 3: Baseline v1 vs Candidate v2.

Evaluates both Baseline v1 and Candidate v2 feature pipelines against the fresh,
independent human validation dataset `data/labels/human_validation_phase3.csv` (28 pairs).

Saves comprehensive metrics to `backend/data/ml/phase_3_independent_evaluation.json`.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Ensure backend on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from app.config import LABELS_DIR, ML_DIR, PROCESSED_DIR, ROOT_DATA_DIR
from app.schemas.candidate import CandidateProfile
from app.schemas.job import JobProfile
from app.services.evaluation.human_validator import (
    map_human_label_to_3class,
    map_rule_recommendation_to_3class,
)
from app.services.ml.predictor import predict_candidate_fit
from app.services.parsing.jd_parser import parse_job_description
from app.services.parsing.resume_parser import parse_resume
from app.services.pipeline.canonical_pipeline import extract_canonical_features
from app.services.pipeline.canonical_pipeline_v2 import extract_canonical_features_v2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

INDEPENDENT_CSV = LABELS_DIR / "human_validation_phase3.csv"
OUTPUT_JSON = ML_DIR / "phase_3_independent_evaluation.json"


# Load manifest map for resume file paths
MANIFEST_CSV = ROOT_DATA_DIR / "manifest" / "expanded_resume_manifest.csv"
manifest_df = pd.read_csv(MANIFEST_CSV)
MANIFEST_MAP = dict(zip(manifest_df["resume_id"], manifest_df["file_path"]))


def load_candidate_profile(resume_id: str) -> CandidateProfile:
    file_path = MANIFEST_MAP.get(resume_id)
    if not file_path or not Path(file_path).exists():
        raise FileNotFoundError(f"Missing resume file for {resume_id}: {file_path}")
    return parse_resume(Path(file_path), candidate_id=resume_id)


def load_job_profile(jd_id: str) -> JobProfile:
    norm_id = jd_id.replace("jd_jd_", "jd_")
    txt_path = ROOT_DATA_DIR / "job_descriptions" / f"{norm_id}.txt"
    if not txt_path.exists():
        raise FileNotFoundError(f"Missing job description: {txt_path}")
    with open(txt_path, encoding="utf-8") as f:
        text = f.read()
    return parse_job_description(text, job_id=norm_id)


def compute_metrics(y_true: list[str], y_pred: list[str], labels: list[str] = ["shortlist", "review", "weak_match"]) -> dict[str, Any]:
    acc = float(accuracy_score(y_true, y_pred))
    p, r, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    per_class = {}
    for idx, label in enumerate(labels):
        per_class[label] = {
            "precision": round(float(p[idx]), 4),
            "recall": round(float(r[idx]), 4),
            "f1": round(float(f1[idx]), 4),
            "support": int(sup[idx]),
        }

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Shortlist specific metrics
    # shortlist is index 0
    shortlist_idx = labels.index("shortlist")
    sl_precision = float(p[shortlist_idx])
    sl_recall = float(r[shortlist_idx])
    pred_sl_count = sum(1 for yp in y_pred if yp == "shortlist")
    false_sl_count = sum(1 for yt, yp in zip(y_true, y_pred) if yp == "shortlist" and yt != "shortlist")
    false_sl_rate = (false_sl_count / pred_sl_count) if pred_sl_count > 0 else 0.0

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "shortlist_precision": round(sl_precision, 4),
        "shortlist_recall": round(sl_recall, 4),
        "false_shortlist_rate": round(false_sl_rate, 4),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm.tolist(),
        },
    }


def main():
    logger.info(f"Loading independent human validation dataset: {INDEPENDENT_CSV}")
    if not INDEPENDENT_CSV.exists():
        raise FileNotFoundError(f"Missing {INDEPENDENT_CSV}")

    df_human = pd.read_csv(INDEPENDENT_CSV)
    logger.info(f"Loaded {len(df_human)} pairs for independent validation.")

    # Cache profiles
    unique_resumes = df_human["resume_id"].unique()
    unique_jds = df_human["jd_id"].unique()

    profiles: dict[str, CandidateProfile] = {}
    for rid in unique_resumes:
        profiles[rid] = load_candidate_profile(rid)

    jobs: dict[str, JobProfile] = {}
    for jid in unique_jds:
        jobs[jid] = load_job_profile(jid)

    records = []

    y_true_3class = []
    y_v1_rule_3class = []
    y_v1_ml_3class = []
    y_v2_rule_3class = []
    y_v2_ml_3class = []

    for idx, row in df_human.iterrows():
        rid = row["resume_id"]
        jid = row["jd_id"]
        human_raw = row["human_label"]
        human_3class = map_human_label_to_3class(human_raw)
        y_true_3class.append(human_3class)

        prof = profiles[rid]
        job = jobs[jid]

        # 1. Baseline v1 Pipeline
        v1_feat = extract_canonical_features(prof, job)
        v1_rule_rec = v1_feat.recommendation.value
        v1_rule_3class = map_rule_recommendation_to_3class(v1_rule_rec)
        v1_ml_pred = predict_candidate_fit(v1_feat.features, v1_feat.recommendation)
        v1_ml_cat = v1_ml_pred.predicted_label.lower() if v1_ml_pred.predicted_label else "weak_match"
        v1_ml_3class = "weak_match" if "weak" in v1_ml_cat else v1_ml_cat

        y_v1_rule_3class.append(v1_rule_3class)
        y_v1_ml_3class.append(v1_ml_3class)

        # 2. Candidate v2 Pipeline
        v2_feat = extract_canonical_features_v2(prof, job)
        v2_rule_rec = v2_feat.recommendation.value
        v2_rule_3class = map_rule_recommendation_to_3class(v2_rule_rec)
        v2_ml_pred = predict_candidate_fit(v2_feat.features, v2_feat.recommendation)
        v2_ml_cat = v2_ml_pred.predicted_label.lower() if v2_ml_pred.predicted_label else "weak_match"
        v2_ml_3class = "weak_match" if "weak" in v2_ml_cat else v2_ml_cat

        y_v2_rule_3class.append(v2_rule_3class)
        y_v2_ml_3class.append(v2_ml_3class)

        record = {
            "pair_index": idx + 1,
            "resume_id": rid,
            "jd_id": jid,
            "human_label": human_raw,
            "human_3class": human_3class,
            "reviewer_notes": row.get("notes", ""),
            "reviewer_evidence": row.get("evidence", ""),
            "v1": {
                "overall_fit_score": v1_feat.overall_score,
                "rule_recommendation": v1_rule_rec,
                "rule_3class": v1_rule_3class,
                "ml_recommendation": v1_ml_pred.predicted_label,
                "ml_probability": v1_ml_pred.probabilities.get(v1_ml_pred.predicted_label, 0.0) if v1_ml_pred.probabilities else 0.0,
                "rule_agrees_with_human": (v1_rule_3class == human_3class),
                "ml_agrees_with_human": (v1_ml_3class == human_3class),
                "features_7d": v1_feat.feature_vector_7d,
            },
            "v2": {
                "overall_fit_score": v2_feat.overall_score,
                "rule_recommendation": v2_rule_rec,
                "rule_3class": v2_rule_3class,
                "ml_recommendation": v2_ml_pred.predicted_label,
                "ml_probability": v2_ml_pred.probabilities.get(v2_ml_pred.predicted_label, 0.0) if v2_ml_pred.probabilities else 0.0,
                "rule_agrees_with_human": (v2_rule_3class == human_3class),
                "ml_agrees_with_human": (v2_ml_3class == human_3class),
                "features_7d": v2_feat.feature_vector_7d,
            },
            "score_delta_v2_minus_v1": round(v2_feat.overall_score - v1_feat.overall_score, 1),
            "rule_improvement": (v2_rule_3class == human_3class) and (v1_rule_3class != human_3class),
            "ml_improvement": (v2_ml_3class == human_3class) and (v1_ml_3class != human_3class),
        }
        records.append(record)

    # Compute comprehensive evaluation metrics
    v1_rule_metrics = compute_metrics(y_true_3class, y_v1_rule_3class)
    v1_ml_metrics = compute_metrics(y_true_3class, y_v1_ml_3class)
    v2_rule_metrics = compute_metrics(y_true_3class, y_v2_rule_3class)
    v2_ml_metrics = compute_metrics(y_true_3class, y_v2_ml_3class)

    # Summary comparisons
    comparison = {
        "rule_based": {
            "v1_accuracy": v1_rule_metrics["accuracy"],
            "v2_accuracy": v2_rule_metrics["accuracy"],
            "accuracy_delta": round(v2_rule_metrics["accuracy"] - v1_rule_metrics["accuracy"], 4),
            "v1_macro_f1": v1_rule_metrics["macro_f1"],
            "v2_macro_f1": v2_rule_metrics["macro_f1"],
            "macro_f1_delta": round(v2_rule_metrics["macro_f1"] - v1_rule_metrics["macro_f1"], 4),
            "v1_shortlist_recall": v1_rule_metrics["shortlist_recall"],
            "v2_shortlist_recall": v2_rule_metrics["shortlist_recall"],
            "shortlist_recall_delta": round(v2_rule_metrics["shortlist_recall"] - v1_rule_metrics["shortlist_recall"], 4),
            "v1_shortlist_precision": v1_rule_metrics["shortlist_precision"],
            "v2_shortlist_precision": v2_rule_metrics["shortlist_precision"],
            "v1_false_shortlist_rate": v1_rule_metrics["false_shortlist_rate"],
            "v2_false_shortlist_rate": v2_rule_metrics["false_shortlist_rate"],
        },
        "ml_predictor": {
            "v1_accuracy": v1_ml_metrics["accuracy"],
            "v2_accuracy": v2_ml_metrics["accuracy"],
            "accuracy_delta": round(v2_ml_metrics["accuracy"] - v1_ml_metrics["accuracy"], 4),
            "v1_macro_f1": v1_ml_metrics["macro_f1"],
            "v2_macro_f1": v2_ml_metrics["macro_f1"],
            "macro_f1_delta": round(v2_ml_metrics["macro_f1"] - v1_ml_metrics["macro_f1"], 4),
            "v1_shortlist_recall": v1_ml_metrics["shortlist_recall"],
            "v2_shortlist_recall": v2_ml_metrics["shortlist_recall"],
            "shortlist_recall_delta": round(v2_ml_metrics["shortlist_recall"] - v1_ml_metrics["shortlist_recall"], 4),
            "v1_shortlist_precision": v1_ml_metrics["shortlist_precision"],
            "v2_shortlist_precision": v2_ml_metrics["shortlist_precision"],
            "v1_false_shortlist_rate": v1_ml_metrics["false_shortlist_rate"],
            "v2_false_shortlist_rate": v2_ml_metrics["false_shortlist_rate"],
        },
    }

    output_payload = {
        "dataset_metadata": {
            "name": "Phase 3 Fresh Independent Human Validation Dataset",
            "file": "data/labels/human_validation_phase3.csv",
            "total_pairs": len(df_human),
            "label_distribution": df_human["human_label"].value_counts().to_dict(),
            "three_class_distribution": pd.Series(y_true_3class).value_counts().to_dict(),
            "reviewer_id": "human_reviewer_02",
            "protocol": "Single-blind (independent human evaluation without exposure to model predictions)",
        },
        "metrics": {
            "baseline_v1_rule": v1_rule_metrics,
            "candidate_v2_rule": v2_rule_metrics,
            "baseline_v1_ml": v1_ml_metrics,
            "candidate_v2_ml": v2_ml_metrics,
        },
        "comparison_summary": comparison,
        "pair_evaluations": records,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    logger.info(f"Evaluation complete! Results saved to {OUTPUT_JSON}")
    print("\n" + "=" * 70)
    print("PHASE 3 INDEPENDENT EVALUATION SUMMARY (28 FRESH UNSEEN PAIRS)")
    print("=" * 70)
    print(f"Rule v1 vs Human: Acc = {v1_rule_metrics['accuracy']:.2%}, Macro F1 = {v1_rule_metrics['macro_f1']:.2%}, Shortlist Recall = {v1_rule_metrics['shortlist_recall']:.2%}")
    print(f"Rule v2 vs Human: Acc = {v2_rule_metrics['accuracy']:.2%}, Macro F1 = {v2_rule_metrics['macro_f1']:.2%}, Shortlist Recall = {v2_rule_metrics['shortlist_recall']:.2%}")
    print(f"  -> Delta Rule:  Acc {comparison['rule_based']['accuracy_delta']:+.2%}, F1 {comparison['rule_based']['macro_f1_delta']:+.2%}, Shortlist Recall {comparison['rule_based']['shortlist_recall_delta']:+.2%}")
    print("-" * 70)
    print(f"ML v1 vs Human:   Acc = {v1_ml_metrics['accuracy']:.2%}, Macro F1 = {v1_ml_metrics['macro_f1']:.2%}, Shortlist Recall = {v1_ml_metrics['shortlist_recall']:.2%}")
    print(f"ML v2 vs Human:   Acc = {v2_ml_metrics['accuracy']:.2%}, Macro F1 = {v2_ml_metrics['macro_f1']:.2%}, Shortlist Recall = {v2_ml_metrics['shortlist_recall']:.2%}")
    print(f"  -> Delta ML:    Acc {comparison['ml_predictor']['accuracy_delta']:+.2%}, F1 {comparison['ml_predictor']['macro_f1_delta']:+.2%}, Shortlist Recall {comparison['ml_predictor']['shortlist_recall_delta']:+.2%}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

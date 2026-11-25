"""Evaluate Retrained V2 ML Model against Human Validation Benchmarks.

Evaluates against:
1. Locked Historical Human Validation (64 pairs in `data/labels/human_validation.csv`)
2. Fresh Independent Human Validation (28 pairs in `data/labels/human_validation_phase3.csv`)

Compares:
- Rule v1
- Rule v2
- ML v1 (original v1 weights)
- ML v2 (retrained v2 weights on Canonical v2 feature space)
- ML v2 Calibrated (Isotonic calibrated)

Saves results to `backend/data/ml/v2_ml_human_evaluations.json`.
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any

# Ensure backend on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from app.config import LABELS_DIR, ML_DIR, ROOT_DATA_DIR
from app.schemas.candidate import CandidateProfile
from app.schemas.evaluation import Recommendation
from app.schemas.job import JobProfile
from app.services.evaluation.human_validator import (
    map_human_label_to_3class,
    map_rule_recommendation_to_3class,
)
from app.services.ml.dataset import FEATURE_NAMES, TARGET_LABELS
from app.services.ml.predictor import predict_candidate_fit
from app.services.parsing.jd_parser import parse_job_description
from app.services.parsing.resume_parser import parse_resume
from app.services.pipeline.canonical_pipeline import extract_canonical_features
from app.services.pipeline.canonical_pipeline_v2 import extract_canonical_features_v2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LOCKED_CSV = LABELS_DIR / "human_validation.csv"
INDEPENDENT_CSV = LABELS_DIR / "human_validation_phase3.csv"
OUTPUT_JSON = ML_DIR / "v2_ml_human_evaluations.json"
MANIFEST_CSV = ROOT_DATA_DIR / "manifest" / "expanded_resume_manifest.csv"

manifest_df = pd.read_csv(MANIFEST_CSV)
MANIFEST_MAP = dict(zip(manifest_df["resume_id"], manifest_df["file_path"]))


def load_candidate(resume_id: str) -> CandidateProfile:
    file_path = MANIFEST_MAP.get(resume_id)
    if not file_path or not Path(file_path).exists():
        raise FileNotFoundError(f"Missing resume file for {resume_id}: {file_path}")
    return parse_resume(Path(file_path), candidate_id=resume_id)


def load_job(jd_id: str) -> JobProfile:
    norm_id = jd_id.replace("jd_jd_", "jd_")
    txt_path = ROOT_DATA_DIR / "job_descriptions" / f"{norm_id}.txt"
    if not txt_path.exists():
        raise FileNotFoundError(f"Missing job description: {txt_path}")
    return parse_job_description(txt_path.read_text(encoding="utf-8"), job_id=norm_id)


def compute_metrics(y_true: list[str], y_pred: list[str], labels: list[str] = TARGET_LABELS) -> dict[str, Any]:
    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_class, r_class, f1_class, sup_class = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    sl_idx = labels.index("shortlist")
    sl_prec = float(p_class[sl_idx])
    sl_rec = float(r_class[sl_idx])

    pred_sl_count = sum(1 for p in y_pred if p == "shortlist")
    false_sl_count = sum(1 for t, p in zip(y_true, y_pred) if p == "shortlist" and t != "shortlist")
    false_sl_rate = (false_sl_count / pred_sl_count) if pred_sl_count > 0 else 0.0

    per_class = {}
    for idx, label in enumerate(labels):
        per_class[label] = {
            "precision": round(float(p_class[idx]), 4),
            "recall": round(float(r_class[idx]), 4),
            "f1": round(float(f1_class[idx]), 4),
            "support": int(sup_class[idx]),
        }

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "shortlist_precision": round(sl_prec, 4),
        "shortlist_recall": round(sl_rec, 4),
        "false_shortlist_rate": round(false_sl_rate, 4),
        "false_shortlist_count": false_sl_count,
        "predicted_shortlist_count": pred_sl_count,
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def evaluate_dataset(csv_path: Path, v2_scaler: Any, v2_model: Any, v2_calibrated: Any) -> dict[str, Any]:
    df = pd.read_csv(csv_path)
    logger.info("Evaluating on %s (%d pairs)", csv_path.name, len(df))

    # Cache loaded profiles & jobs
    res_ids = df["resume_id"].unique()
    jd_ids = df["jd_id"].unique()

    profiles = {rid: load_candidate(rid) for rid in res_ids}
    jobs = {jid: load_job(jid) for jid in jd_ids}

    y_true: list[str] = []
    y_rule_v1: list[str] = []
    y_rule_v2: list[str] = []
    y_ml_v1: list[str] = []
    y_ml_v2: list[str] = []
    y_ml_v2_cal: list[str] = []

    pairs_detail = []

    for idx, row in df.iterrows():
        rid = row["resume_id"]
        jid = row["jd_id"]
        human_raw = row["human_label"]
        human_3class = map_human_label_to_3class(human_raw)
        y_true.append(human_3class)

        prof = profiles[rid]
        job = jobs[jid]

        # 1. Pipeline v1 (Baseline)
        feat_v1 = extract_canonical_features(prof, job)
        rec_v1 = feat_v1.recommendation.value
        c3_rule_v1 = map_rule_recommendation_to_3class(rec_v1)
        # ML v1 (original predictor with v1 model & scaler)
        ml_v1_pred = predict_candidate_fit(feat_v1.features, feat_v1.recommendation, prefer_real=True)
        c3_ml_v1 = ml_v1_pred.predicted_label.lower() if ml_v1_pred.predicted_label else "weak_match"
        if "weak" in c3_ml_v1:
            c3_ml_v1 = "weak_match"

        y_rule_v1.append(c3_rule_v1)
        y_ml_v1.append(c3_ml_v1)

        # 2. Pipeline v2 (Corrected Feature Space)
        feat_v2 = extract_canonical_features_v2(prof, job)
        rec_v2 = feat_v2.recommendation.value
        c3_rule_v2 = map_rule_recommendation_to_3class(rec_v2)
        y_rule_v2.append(c3_rule_v2)

        # ML v2 on v2 feature vector
        raw_v2_vec = np.array(feat_v2.feature_vector_7d).reshape(1, -1)
        scaled_v2_vec = v2_scaler.transform(raw_v2_vec)

        pred_v2_raw = str(v2_model.predict(scaled_v2_vec)[0])
        pred_v2_cal = str(v2_calibrated.predict(scaled_v2_vec)[0])

        y_ml_v2.append(pred_v2_raw)
        y_ml_v2_cal.append(pred_v2_cal)

        pairs_detail.append({
            "pair_index": idx + 1,
            "resume_id": rid,
            "jd_id": jid,
            "human_label": human_raw,
            "human_3class": human_3class,
            "rule_v1": c3_rule_v1,
            "rule_v2": c3_rule_v2,
            "ml_v1": c3_ml_v1,
            "ml_v2": pred_v2_raw,
            "ml_v2_calibrated": pred_v2_cal,
            "score_v1": feat_v1.overall_score,
            "score_v2": feat_v2.overall_score,
        })

    metrics = {
        "dataset_name": csv_path.stem,
        "sample_size": len(df),
        "rule_v1": compute_metrics(y_true, y_rule_v1),
        "rule_v2": compute_metrics(y_true, y_rule_v2),
        "ml_v1": compute_metrics(y_true, y_ml_v1),
        "ml_v2": compute_metrics(y_true, y_ml_v2),
        "ml_v2_calibrated": compute_metrics(y_true, y_ml_v2_cal),
        "pairs_detail": pairs_detail,
    }

    return metrics


def main():
    # Load v2 model artifacts
    scaler_path = ML_DIR / "v2_scaler.pkl"
    model_path = ML_DIR / "v2_logistic_regression.pkl"
    calibrated_path = ML_DIR / "v2_calibrated_model.pkl"

    if not (scaler_path.exists() and model_path.exists() and calibrated_path.exists()):
        raise FileNotFoundError("Missing v2 model artifacts. Run scripts/train_v2_ml_models.py first.")

    with open(scaler_path, "rb") as f:
        v2_scaler = pickle.load(f)
    with open(model_path, "rb") as f:
        v2_model = pickle.load(f)
    with open(calibrated_path, "rb") as f:
        v2_calibrated = pickle.load(f)

    # 1. Evaluate Locked Historical Human Validation (64 pairs)
    locked_eval = evaluate_dataset(LOCKED_CSV, v2_scaler, v2_model, v2_calibrated)

    # 2. Evaluate Independent Human Validation (28 pairs)
    independent_eval = evaluate_dataset(INDEPENDENT_CSV, v2_scaler, v2_model, v2_calibrated)

    payload = {
        "evaluation_timestamp": pd.Timestamp.now().isoformat(),
        "model_evaluated": "Logistic Regression (Balanced, v2)",
        "feature_version": "canonical_v2",
        "locked_benchmark_64": locked_eval,
        "independent_benchmark_28": independent_eval,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info("Evaluation complete! Results saved to %s", OUTPUT_JSON)

    # Print summary comparison
    print("\n" + "=" * 88)
    print("PHASE 4 HUMAN VALIDATION BENCHMARKS COMPARISON: RULE vs ML (v1 vs v2)")
    print("=" * 88)
    print(f"{'Benchmark / Model':32s} | {'Acc':6s} | {'Macro F1':8s} | {'SL Rec':6s} | {'SL Prec':7s} | {'False SL%':9s}")
    print("-" * 88)

    for bench_name, res in [("Locked 64-Pair Benchmark", locked_eval), ("Independent 28-Pair Pool", independent_eval)]:
        print(f"[{bench_name}]")
        for m_name in ["rule_v1", "rule_v2", "ml_v1", "ml_v2", "ml_v2_calibrated"]:
            m = res[m_name]
            print(
                f"  {m_name:28s} | {m['accuracy']:.2%} | {m['macro_f1']:.2%}   | "
                f"{m['shortlist_recall']:.2%} | {m['shortlist_precision']:.2%}  | {m['false_shortlist_rate']:.2%}"
            )
        print("-" * 88)
    print("=" * 88 + "\n")


if __name__ == "__main__":
    main()

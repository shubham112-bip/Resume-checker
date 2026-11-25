"""Model Calibration & Class Imbalance Investigation (Phase 2.2).

Investigates:
1. Class-weighted Logistic Regression (handling 14:2.5:1 imbalance)
2. CalibratedClassifierCV (Platt Scaling via 5-fold cross-validation)
3. Evaluates calibrated models on both Dev Test Partition and Held-out Human Validation Set.
"""

from __future__ import annotations

import json
import logging
import pickle
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from app.config import ML_DIR, PROCESSED_DIR, ROOT_DATA_DIR
from app.schemas.evaluation import FeatureVector
from app.services.evaluation.human_validator import compute_evaluation_metrics, map_human_label_to_3class
from app.services.ml.dataset import FEATURE_NAMES, TARGET_LABELS

logger = logging.getLogger(__name__)

CALIBRATED_METRICS_JSON = ML_DIR / "calibrated_metrics.json"
CALIBRATED_MODEL_PKL = ML_DIR / "calibrated_logistic_regression.pkl"
CALIBRATED_SCALER_PKL = ML_DIR / "calibrated_scaler.pkl"


def train_and_evaluate_calibrated_model() -> dict[str, Any]:
    """Train class-weighted calibrated model on development dataset with zero human leakage."""
    dev_csv = PROCESSED_DIR / "evaluation_pairs.csv"
    if not dev_csv.exists():
        raise FileNotFoundError(f"Dev dataset not found at {dev_csv}")

    df = pd.read_csv(dev_csv)
    X = df[FEATURE_NAMES].values
    y = df["classification_label"].values
    groups = df["resume_id"].values

    # Same grouped split by resume_id (zero candidate leakage)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(sgkf.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. Baseline Logistic Regression (Unweighted, as currently in production)
    lr_baseline = LogisticRegression(max_iter=1000, random_state=42)
    lr_baseline.fit(X_train_scaled, y_train)
    y_pred_dev_base = lr_baseline.predict(X_test_scaled)
    dev_base_metrics = compute_evaluation_metrics(list(y_test), list(y_pred_dev_base))

    # 2. Class-Weighted Balanced Logistic Regression (handling 729:131:52 skew)
    lr_balanced = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr_balanced.fit(X_train_scaled, y_train)
    y_pred_dev_bal = lr_balanced.predict(X_test_scaled)
    dev_bal_metrics = compute_evaluation_metrics(list(y_test), list(y_pred_dev_bal))

    # 3. Platt-Calibrated Model (CalibratedClassifierCV with 5-fold CV)
    calibrated_model = CalibratedClassifierCV(
        estimator=LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        method="sigmoid",
        cv=5,
    )
    calibrated_model.fit(X_train_scaled, y_train)
    y_pred_dev_cal = calibrated_model.predict(X_test_scaled)
    dev_cal_metrics = compute_evaluation_metrics(list(y_test), list(y_pred_dev_cal))

    # Save calibrated model artifacts separately (Never overwrite original models)
    with open(CALIBRATED_MODEL_PKL, "wb") as f:
        pickle.dump(calibrated_model, f)
    with open(CALIBRATED_SCALER_PKL, "wb") as f:
        pickle.dump(scaler, f)

    # 4. Evaluate on Held-out Human Validation Set
    human_val_csv = ROOT_DATA_DIR / "labels" / "human_validation.csv"
    locked_json = ML_DIR / "locked_human_evaluation.json"

    human_eval_results = {}
    if locked_json.exists():
        locked_data = json.loads(locked_json.read_text(encoding="utf-8"))
        pairs = locked_data["pair_details"]

        y_human: list[str] = []
        y_pred_base_human: list[str] = []
        y_pred_bal_human: list[str] = []
        y_pred_cal_human: list[str] = []

        for p in pairs:
            feat_dict = p["features"]
            raw_feats = [
                feat_dict["requiredSkillMatch"],
                feat_dict["experienceMatch"],
                feat_dict["semanticSimilarity"],
                feat_dict["educationMatch"],
                feat_dict["projectMatch"],
                feat_dict["atsScore"],
                feat_dict["resumeQuality"],
            ]
            x_raw = np.array(raw_feats).reshape(1, -1)
            x_scaled = scaler.transform(x_raw)

            pred_base = lr_baseline.predict(x_scaled)[0]
            pred_bal = lr_balanced.predict(x_scaled)[0]
            pred_cal = calibrated_model.predict(x_scaled)[0]

            y_human.append(p["human_label_3class"])
            y_pred_base_human.append(pred_base)
            y_pred_bal_human.append(pred_bal)
            y_pred_cal_human.append(pred_cal)

        human_base_metrics = compute_evaluation_metrics(y_human, y_pred_base_human)
        human_bal_metrics = compute_evaluation_metrics(y_human, y_pred_bal_human)
        human_cal_metrics = compute_evaluation_metrics(y_human, y_pred_cal_human)

        human_eval_results = {
            "baseline_unweighted_on_human": human_base_metrics,
            "class_weighted_on_human": human_bal_metrics,
            "calibrated_model_on_human": human_cal_metrics,
        }

    results = {
        "timestamp": datetime.now(UTC).isoformat(),
        "methodology": (
            "Model calibration conducted strictly on the development dataset using class_weight='balanced' "
            "and 5-fold CalibratedClassifierCV (Platt Scaling). Evaluated on both the development test partition "
            "and the held-out Human Validation set without human validation data leaking into training."
        ),
        "development_test_metrics": {
            "baseline_unweighted": dev_base_metrics,
            "class_weighted": dev_bal_metrics,
            "calibrated_model": dev_cal_metrics,
        },
        "human_validation_metrics": human_eval_results,
    }

    with open(CALIBRATED_METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Calibration investigation complete. Saved to %s", CALIBRATED_METRICS_JSON)
    return results


if __name__ == "__main__":
    res = train_and_evaluate_calibrated_model()
    print("Dev Test Partition:")
    print("  Baseline Unweighted Acc:", res["development_test_metrics"]["baseline_unweighted"]["accuracy"])
    print("  Class-Weighted Acc:     ", res["development_test_metrics"]["class_weighted"]["accuracy"])
    print("  Calibrated Acc:         ", res["development_test_metrics"]["calibrated_model"]["accuracy"])

    if "human_validation_metrics" in res and res["human_validation_metrics"]:
        h = res["human_validation_metrics"]
        print("\nHuman Validation Benchmark (N=64):")
        print(f"  Baseline (Locked):  Acc={h['baseline_unweighted_on_human']['accuracy']:.4f}, Macro F1={h['baseline_unweighted_on_human']['macro_f1']:.4f}, Shortlist Recall={h['baseline_unweighted_on_human']['shortlist_recall']:.4f}")
        print(f"  Class-Weighted:     Acc={h['class_weighted_on_human']['accuracy']:.4f}, Macro F1={h['class_weighted_on_human']['macro_f1']:.4f}, Shortlist Recall={h['class_weighted_on_human']['shortlist_recall']:.4f}")
        print(f"  Platt Calibrated:   Acc={h['calibrated_model_on_human']['accuracy']:.4f}, Macro F1={h['calibrated_model_on_human']['macro_f1']:.4f}, Shortlist Recall={h['calibrated_model_on_human']['shortlist_recall']:.4f}")

"""ML Training & Evaluation Script for Phase 4 (Canonical Pipeline v2 Features).

Trains and compares:
1. Logistic Regression (Standard vs Balanced)
2. Random Forest (Standard vs Balanced)
3. KNN (Uniform vs Distance-weighted)

Uses group-aware splitting (StratifiedGroupKFold) by resume_id so zero resumes
leak between train, validation, and test sets.

Evaluates shortlist safety (False Shortlist Count, False Shortlist Rate, Shortlist Precision/Recall).
Evaluates probability calibration (Platt/Sigmoid vs Isotonic) on held-out test data.

Saves versioned v2 artifacts:
- `backend/data/ml/v2_scaler.pkl`
- `backend/data/ml/v2_logistic_regression.pkl`
- `backend/data/ml/v2_random_forest.pkl`
- `backend/data/ml/v2_knn.pkl`
- `backend/data/ml/v2_best_model.txt`
- `backend/data/ml/v2_calibrated_model.pkl`
- `backend/data/ml/v2_model_metrics.json`
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
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from app.config import ML_DIR
from app.services.fairness import audit_feature_names
from app.services.ml.dataset import FEATURE_NAMES, TARGET_LABELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATASET_CSV = ML_DIR / "v2_training_dataset.csv"
METRICS_JSON = ML_DIR / "v2_model_metrics.json"


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str] = TARGET_LABELS) -> dict[str, Any]:
    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    p_class, r_class, f1_class, sup_class = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)

    per_class = {}
    for idx, label in enumerate(labels):
        per_class[label] = {
            "precision": round(float(p_class[idx]), 4),
            "recall": round(float(r_class[idx]), 4),
            "f1": round(float(f1_class[idx]), 4),
            "support": int(sup_class[idx]),
        }

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    # Shortlist Safety Metrics
    # labels = ["weak_match", "review", "shortlist"]
    sl_idx = labels.index("shortlist")
    weak_idx = labels.index("weak_match")

    sl_prec = float(p_class[sl_idx])
    sl_rec = float(r_class[sl_idx])

    pred_sl_count = int(np.sum(y_pred == "shortlist"))
    # Dangerous failure: True is weak_match, but predicted as shortlist
    false_sl_from_weak = int(np.sum((y_true == "weak_match") & (y_pred == "shortlist")))
    # All false shortlists (true review or true weak_match)
    total_false_sl = int(np.sum((y_true != "shortlist") & (y_pred == "shortlist")))
    false_sl_rate = float(total_false_sl / pred_sl_count) if pred_sl_count > 0 else 0.0

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_wt), 4),
        "shortlist_precision": round(sl_prec, 4),
        "shortlist_recall": round(sl_rec, 4),
        "predicted_shortlist_count": pred_sl_count,
        "false_shortlist_count": total_false_sl,
        "false_shortlist_from_weak_count": false_sl_from_weak,
        "false_shortlist_rate": round(false_sl_rate, 4),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def run_training_and_comparison():
    logger.info("Loading v2 ML dataset: %s", DATASET_CSV)
    if not DATASET_CSV.exists():
        raise FileNotFoundError(f"Missing {DATASET_CSV}. Run scripts/build_v2_ml_dataset.py first.")

    df = pd.read_csv(DATASET_CSV)
    logger.info("Loaded %d rows across %d resumes and %d JDs.", len(df), df["resume_id"].nunique(), df["jd_id"].nunique())

    # Audit features
    violating = audit_feature_names(FEATURE_NAMES)
    if violating:
        raise ValueError(f"Fairness audit failed! Sensitive features detected: {violating}")

    X = df[FEATURE_NAMES].values
    y = df["classification_label"].values
    groups = df["resume_id"].values

    # 1. Group-aware train / test split
    # 5-fold StratifiedGroupKFold gives 80% train resumes, 20% test resumes
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(sgkf.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    train_groups = groups[train_idx]
    test_groups = groups[test_idx]

    # Verify zero group leakage
    train_resume_ids = set(train_groups)
    test_resume_ids = set(test_groups)
    overlap = train_resume_ids.intersection(test_resume_ids)
    assert len(overlap) == 0, f"DATA LEAKAGE DETECTED! Resumes in both train and test: {overlap}"
    logger.info("Grouped split verified: 0 resume overlap! (Train resumes: %d, Test resumes: %d)",
                len(train_resume_ids), len(test_resume_ids))

    # Scaler fit only on train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save v2 scaler
    with open(ML_DIR / "v2_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    logger.info("Saved v2_scaler.pkl")

    # 2. Define Candidate Models
    candidate_configs: dict[str, dict[str, Any]] = {
        "Logistic Regression (Standard)": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "use_scaled": True,
        },
        "Logistic Regression (Balanced)": {
            "model": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            "use_scaled": True,
        },
        "Random Forest (Standard)": {
            "model": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
            "use_scaled": False,
        },
        "Random Forest (Balanced)": {
            "model": RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=42),
            "use_scaled": False,
        },
        "KNN (Uniform, k=5)": {
            "model": KNeighborsClassifier(n_neighbors=5, weights="uniform"),
            "use_scaled": True,
        },
        "KNN (Distance, k=5)": {
            "model": KNeighborsClassifier(n_neighbors=5, weights="distance"),
            "use_scaled": True,
        },
    }

    # Cross-validation setup (inner StratifiedGroupKFold on train)
    inner_cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)

    model_evaluations: dict[str, dict[str, Any]] = {}

    for name, config in candidate_configs.items():
        clf = config["model"]
        use_scaled = config["use_scaled"]
        cur_X_tr = X_train_scaled if use_scaled else X_train
        cur_X_te = X_test_scaled if use_scaled else X_test

        # Cross-validation macro F1 across grouped folds
        cv_macro_f1s = []
        for inner_tr_idx, inner_val_idx in inner_cv.split(cur_X_tr, y_train, groups=train_groups):
            # Verify no inner fold leakage
            assert len(set(train_groups[inner_tr_idx]).intersection(set(train_groups[inner_val_idx]))) == 0
            inner_clf = type(clf)(**clf.get_params())
            inner_clf.fit(cur_X_tr[inner_tr_idx], y_train[inner_tr_idx])
            inner_pred = inner_clf.predict(cur_X_tr[inner_val_idx])
            _, _, f1_val, _ = precision_recall_fscore_support(
                y_train[inner_val_idx], inner_pred, average="macro", zero_division=0
            )
            cv_macro_f1s.append(f1_val)

        cv_mean = float(np.mean(cv_macro_f1s))
        cv_std = float(np.std(cv_macro_f1s))

        # Train on full training set
        clf.fit(cur_X_tr, y_train)
        y_test_pred = clf.predict(cur_X_te)

        test_metrics = evaluate_predictions(y_test, y_test_pred)
        test_metrics["cv_macro_f1_mean"] = round(cv_mean, 4)
        test_metrics["cv_macro_f1_std"] = round(cv_std, 4)

        # Feature importance / coefficients
        if isinstance(clf, LogisticRegression):
            coef_mean = np.mean(np.abs(clf.coef_), axis=0)
            test_metrics["feature_importance"] = {
                feat: round(float(v), 4) for feat, v in zip(FEATURE_NAMES, coef_mean)
            }
            test_metrics["importance_method"] = "Mean Absolute Coefficient magnitude"
            test_metrics["coefficients_by_class"] = {
                cls: [round(float(v), 4) for v in row] for cls, row in zip(clf.classes_, clf.coef_)
            }
        elif isinstance(clf, RandomForestClassifier):
            test_metrics["feature_importance"] = {
                feat: round(float(v), 4) for feat, v in zip(FEATURE_NAMES, clf.feature_importances_)
            }
            test_metrics["importance_method"] = "MDI (Gini Impurity reduction)"
        else:
            test_metrics["feature_importance"] = {}
            test_metrics["importance_method"] = "N/A (Instance-based KNN)"

        model_evaluations[name] = test_metrics

    # Save artifacts for primary models
    with open(ML_DIR / "v2_logistic_regression.pkl", "wb") as f:
        pickle.dump(candidate_configs["Logistic Regression (Balanced)"]["model"], f)
    with open(ML_DIR / "v2_random_forest.pkl", "wb") as f:
        pickle.dump(candidate_configs["Random Forest (Balanced)"]["model"], f)
    with open(ML_DIR / "v2_knn.pkl", "wb") as f:
        pickle.dump(candidate_configs["KNN (Distance, k=5)"]["model"], f)
    logger.info("Saved v2_logistic_regression.pkl, v2_random_forest.pkl, and v2_knn.pkl")

    # 3. Model Selection
    # Balance between: Macro F1, Shortlist Recall, Shortlist Precision, False Shortlist Rate
    logger.info("Comparing candidate models on held-out test cohort...")
    print("\n" + "=" * 80)
    print("PHASE 4 MODEL EVALUATION & COMPARISON TABLE (HELD-OUT GROUPED TEST SET)")
    print("=" * 80)
    header = f"{'Model':32s} | {'Acc':6s} | {'Macro F1':8s} | {'SL Rec':6s} | {'SL Prec':7s} | {'False SL%':9s} | {'CV F1':13s}"
    print(header)
    print("-" * 80)

    best_model_name = ""
    best_score = -1.0

    for name, m in model_evaluations.items():
        line = (
            f"{name:32s} | {m['accuracy']:.2%} | {m['macro_f1']:.2%}   | "
            f"{m['shortlist_recall']:.2%} | {m['shortlist_precision']:.2%}  | "
            f"{m['false_shortlist_rate']:.2%}   | {m['cv_macro_f1_mean']:.4f}±{m['cv_macro_f1_std']:.3f}"
        )
        print(line)

        # Composite score prioritizing shortlist safety + recall + macro F1
        composite = m["macro_f1"] * 0.4 + m["shortlist_recall"] * 0.4 + m["shortlist_precision"] * 0.2
        if composite > best_score:
            best_score = composite
            best_model_name = name

    print("=" * 80)
    print(f"Selected Best Model: {best_model_name} (Composite score: {best_score:.4f})")
    print("=" * 80 + "\n")

    # Record best model text
    (ML_DIR / "v2_best_model.txt").write_text(best_model_name)

    # 4. Probability Calibration on Selected Model
    logger.info("Evaluating Probability Calibration for %s...", best_model_name)
    best_base_clf = candidate_configs[best_model_name]["model"]
    best_is_scaled = candidate_configs[best_model_name]["use_scaled"]
    cur_X_tr = X_train_scaled if best_is_scaled else X_train
    cur_X_te = X_test_scaled if best_is_scaled else X_test

    # Calibrated models fit via inner group cross-validation splits
    cal_cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
    cv_splits = list(cal_cv.split(cur_X_tr, y_train, groups=train_groups))

    # A. Platt / Sigmoid
    cal_sigmoid = CalibratedClassifierCV(
        estimator=type(best_base_clf)(**best_base_clf.get_params()),
        method="sigmoid",
        cv=cv_splits,
    )
    cal_sigmoid.fit(cur_X_tr, y_train)
    sig_probs = cal_sigmoid.predict_proba(cur_X_te)
    sig_pred = cal_sigmoid.predict(cur_X_te)

    # B. Isotonic
    cal_isotonic = CalibratedClassifierCV(
        estimator=type(best_base_clf)(**best_base_clf.get_params()),
        method="isotonic",
        cv=cv_splits,
    )
    cal_isotonic.fit(cur_X_tr, y_train)
    iso_probs = cal_isotonic.predict_proba(cur_X_te)
    iso_pred = cal_isotonic.predict(cur_X_te)

    # Raw probabilities from base model
    raw_probs = best_base_clf.predict_proba(cur_X_te)

    # Compute Log Loss and Brier Score
    classes = list(best_base_clf.classes_)
    y_test_bin = np.zeros((len(y_test), len(classes)))
    for c_idx, c in enumerate(classes):
        y_test_bin[:, c_idx] = (y_test == c).astype(int)

    def calc_brier_multiclass(y_bin: np.ndarray, probs: np.ndarray) -> float:
        return float(np.mean(np.sum((probs - y_bin) ** 2, axis=1)))

    raw_loss = float(log_loss(y_test, raw_probs))
    raw_brier = calc_brier_multiclass(y_test_bin, raw_probs)

    sig_loss = float(log_loss(y_test, sig_probs))
    sig_brier = calc_brier_multiclass(y_test_bin, sig_probs)

    iso_loss = float(log_loss(y_test, iso_probs))
    iso_brier = calc_brier_multiclass(y_test_bin, iso_probs)

    calibration_results = {
        "uncalibrated": {
            "log_loss": round(raw_loss, 4),
            "brier_score": round(raw_brier, 4),
            "macro_f1": model_evaluations[best_model_name]["macro_f1"],
        },
        "platt_sigmoid": {
            "log_loss": round(sig_loss, 4),
            "brier_score": round(sig_brier, 4),
            "macro_f1": round(float(precision_recall_fscore_support(y_test, sig_pred, average='macro', zero_division=0)[2]), 4),
        },
        "isotonic": {
            "log_loss": round(iso_loss, 4),
            "brier_score": round(iso_brier, 4),
            "macro_f1": round(float(precision_recall_fscore_support(y_test, iso_pred, average='macro', zero_division=0)[2]), 4),
        },
    }

    print("\n" + "=" * 65)
    print("PROBABILITY CALIBRATION COMPARISON (HELD-OUT TEST SET)")
    print("=" * 65)
    print(f"Uncalibrated Base : Log Loss = {raw_loss:.4f} | Brier Score = {raw_brier:.4f}")
    print(f"Platt / Sigmoid   : Log Loss = {sig_loss:.4f} | Brier Score = {sig_brier:.4f} (Macro F1 = {calibration_results['platt_sigmoid']['macro_f1']:.2%})")
    print(f"Isotonic          : Log Loss = {iso_loss:.4f} | Brier Score = {iso_brier:.4f} (Macro F1 = {calibration_results['isotonic']['macro_f1']:.2%})")
    print("=" * 65 + "\n")

    # Select calibration method: Sigmoid is safer and smoother on moderate sample sizes without step artifacts
    winning_calibrator = cal_sigmoid if sig_loss <= iso_loss else cal_isotonic
    winning_cal_method = "sigmoid" if sig_loss <= iso_loss else "isotonic"

    with open(ML_DIR / "v2_calibrated_model.pkl", "wb") as f:
        pickle.dump(winning_calibrator, f)
    logger.info("Saved v2_calibrated_model.pkl using %s calibration", winning_cal_method)

    # 5. Output Summary Payload
    summary_payload = {
        "dataset_metadata": {
            "dataset_version": "canonical_v2",
            "source_file": "backend/data/ml/v2_training_dataset.csv",
            "total_pairs": len(df),
            "total_resumes": int(df["resume_id"].nunique()),
            "total_jds": int(df["jd_id"].nunique()),
            "class_distribution": df["classification_label"].value_counts().to_dict(),
            "train_pairs": len(X_train),
            "test_pairs": len(X_test),
            "train_resumes": len(train_resume_ids),
            "test_resumes": len(test_resume_ids),
            "resume_overlap": len(overlap),
            "features": FEATURE_NAMES,
            "target_classes": TARGET_LABELS,
        },
        "model_evaluations": model_evaluations,
        "selected_best_model": best_model_name,
        "calibration_comparison": calibration_results,
        "selected_calibration_method": winning_cal_method,
    }

    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    logger.info("Saved v2 model metrics to %s", METRICS_JSON)

    return summary_payload


if __name__ == "__main__":
    run_training_and_comparison()

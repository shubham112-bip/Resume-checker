"""ML Training & Evaluation Service for Canonical Pipeline v2 Features.

Trains and compares:
1. Logistic Regression (Standard vs Balanced)
2. Random Forest (Standard vs Balanced)
3. KNN (Uniform vs Distance-weighted)

Uses group-aware splitting (StratifiedGroupKFold) by resume_id so zero resumes
leak between train, validation, and test sets.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
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

    sl_idx = labels.index("shortlist")
    sl_prec = float(p_class[sl_idx])
    sl_rec = float(r_class[sl_idx])

    pred_sl_count = int(np.sum(y_pred == "shortlist"))
    false_sl_from_weak = int(np.sum((y_true == "weak_match") & (y_pred == "shortlist")))
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


def run_training_and_comparison() -> dict[str, Any]:
    logger.info("Loading v2 ML dataset: %s", DATASET_CSV)
    if not DATASET_CSV.exists():
        raise FileNotFoundError(f"Missing {DATASET_CSV}.")

    df = pd.read_csv(DATASET_CSV)
    logger.info("Loaded %d rows across %d resumes and %d JDs.", len(df), df["resume_id"].nunique(), df["jd_id"].nunique())

    violating = audit_feature_names(FEATURE_NAMES)
    if violating:
        raise ValueError(f"Fairness audit failed! Sensitive features detected: {violating}")

    X = df[FEATURE_NAMES].values
    y = df["classification_label"].values
    groups = df["resume_id"].values

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(sgkf.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    train_groups = groups[train_idx]
    test_groups = groups[test_idx]

    train_resume_ids = set(train_groups)
    test_resume_ids = set(test_groups)
    overlap = train_resume_ids.intersection(test_resume_ids)
    assert len(overlap) == 0, f"DATA LEAKAGE DETECTED! Resumes in both train and test: {overlap}"

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    with open(ML_DIR / "v2_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

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

    inner_cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
    model_evaluations: dict[str, dict[str, Any]] = {}

    for name, config in candidate_configs.items():
        clf = config["model"]
        use_scaled = config["use_scaled"]
        cur_X_tr = X_train_scaled if use_scaled else X_train
        cur_X_te = X_test_scaled if use_scaled else X_test

        cv_macro_f1s = []
        for inner_tr_idx, inner_val_idx in inner_cv.split(cur_X_tr, y_train, groups=train_groups):
            inner_clf = type(clf)(**clf.get_params())
            inner_clf.fit(cur_X_tr[inner_tr_idx], y_train[inner_tr_idx])
            inner_pred = inner_clf.predict(cur_X_tr[inner_val_idx])
            _, _, f1_val, _ = precision_recall_fscore_support(
                y_train[inner_val_idx], inner_pred, average="macro", zero_division=0
            )
            cv_macro_f1s.append(f1_val)

        cv_mean = float(np.mean(cv_macro_f1s))
        cv_std = float(np.std(cv_macro_f1s))

        clf.fit(cur_X_tr, y_train)
        y_test_pred = clf.predict(cur_X_te)

        test_metrics = evaluate_predictions(y_test, y_test_pred)
        test_metrics["cv_macro_f1_mean"] = round(cv_mean, 4)
        test_metrics["cv_macro_f1_std"] = round(cv_std, 4)

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

    with open(ML_DIR / "v2_logistic_regression.pkl", "wb") as f:
        pickle.dump(candidate_configs["Logistic Regression (Balanced)"]["model"], f)
    with open(ML_DIR / "v2_random_forest.pkl", "wb") as f:
        pickle.dump(candidate_configs["Random Forest (Balanced)"]["model"], f)
    with open(ML_DIR / "v2_knn.pkl", "wb") as f:
        pickle.dump(candidate_configs["KNN (Distance, k=5)"]["model"], f)

    best_model_name = "Logistic Regression (Balanced)"
    (ML_DIR / "v2_best_model.txt").write_text(best_model_name)

    best_base_clf = candidate_configs[best_model_name]["model"]
    best_is_scaled = candidate_configs[best_model_name]["use_scaled"]
    cur_X_tr = X_train_scaled if best_is_scaled else X_train
    cur_X_te = X_test_scaled if best_is_scaled else X_test

    cal_cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
    cv_splits = list(cal_cv.split(cur_X_tr, y_train, groups=train_groups))

    cal_sigmoid = CalibratedClassifierCV(
        estimator=type(best_base_clf)(**best_base_clf.get_params()),
        method="sigmoid",
        cv=cv_splits,
    )
    cal_sigmoid.fit(cur_X_tr, y_train)
    sig_probs = cal_sigmoid.predict_proba(cur_X_te)
    sig_pred = cal_sigmoid.predict(cur_X_te)

    cal_isotonic = CalibratedClassifierCV(
        estimator=type(best_base_clf)(**best_base_clf.get_params()),
        method="isotonic",
        cv=cv_splits,
    )
    cal_isotonic.fit(cur_X_tr, y_train)
    iso_probs = cal_isotonic.predict_proba(cur_X_te)
    iso_pred = cal_isotonic.predict(cur_X_te)

    raw_probs = best_base_clf.predict_proba(cur_X_te)

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
            "macro_f1": round(float(precision_recall_fscore_support(y_test, sig_pred, average="macro", zero_division=0)[2]), 4),
        },
        "isotonic": {
            "log_loss": round(iso_loss, 4),
            "brier_score": round(iso_brier, 4),
            "macro_f1": round(float(precision_recall_fscore_support(y_test, iso_pred, average="macro", zero_division=0)[2]), 4),
        },
    }

    winning_calibrator = cal_sigmoid if sig_loss <= iso_loss else cal_isotonic
    winning_cal_method = "sigmoid" if sig_loss <= iso_loss else "isotonic"

    with open(ML_DIR / "v2_calibrated_model.pkl", "wb") as f:
        pickle.dump(winning_calibrator, f)

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

    return summary_payload

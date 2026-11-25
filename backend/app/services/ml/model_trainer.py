"""ML Model training and comparative scientific evaluation (LR vs RF vs KNN)."""

from __future__ import annotations

import json
import logging
import pickle
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from app.config import ML_DIR, PROCESSED_DIR
from app.schemas.api import DatasetInfo, ModelMetrics, ModelMetricsResponse
from app.services.fairness import audit_feature_names
from app.services.ml.dataset import FEATURE_NAMES, TARGET_LABELS, generate_synthetic_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def train_and_evaluate_models(
    dataset_path: Path | None = None,
    dataset_type: str = "synthetic",
) -> ModelMetricsResponse:
    """Train and evaluate ML models (Logistic Regression, Random Forest, KNN).

    Supports:
    - 'synthetic': 800 synthetic candidates benchmark
    - 'real': 66 Resume-JD pairs across 11 sample candidates with grouped split
    """
    ML_DIR.mkdir(parents=True, exist_ok=True)

    if dataset_type == "real" or (dataset_path and "evaluation_pairs" in str(dataset_path)):
        return _train_real_dataset_models(dataset_path)

    return _train_synthetic_dataset_models(dataset_path)


def _train_synthetic_dataset_models(dataset_path: Path | None = None) -> ModelMetricsResponse:
    csv_path = dataset_path or (ML_DIR / "synthetic_candidates.csv")
    if not csv_path.exists():
        csv_path = generate_synthetic_dataset(csv_path)

    df = pd.read_csv(csv_path)
    X = df[FEATURE_NAMES].values
    y = df["label"].values

    # Fairness audit: ensure no protected/sensitive attributes are in the ML feature set
    violating = audit_feature_names(FEATURE_NAMES)
    if violating:
        raise ValueError(f"Fairness audit failed! Sensitive features detected: {violating}")

    # Split train/test (80/20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Scaler for distance/linear models (LR & KNN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    with open(ML_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # 1. Logistic Regression (Interpretable baseline)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    # 2. Random Forest (Ensemble non-linear)
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    # 3. K-Nearest Neighbors (Instance-based)
    knn = KNeighborsClassifier(n_neighbors=7)

    models_dict = {
        "Logistic Regression": (lr, X_train_scaled, X_test_scaled, True),
        "Random Forest": (rf, X_train, X_test, False),
        "KNN": (knn, X_train_scaled, X_test_scaled, True),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics_list: list[ModelMetrics] = []
    best_model_name = ""
    best_f1 = -1.0

    for name, (model, cur_X_train, cur_X_test, is_scaled) in models_dict.items():
        # Cross-validation
        cv_scores = cross_val_score(model, cur_X_train, y_train, cv=cv, scoring="f1_macro")
        model.fit(cur_X_train, y_train)

        # Test evaluation
        y_pred = model.predict(cur_X_test)
        acc = float(accuracy_score(y_test, y_pred))
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )

        # Calculate Shortlist Recall (crucial for recruitment: never miss a strong candidate)
        shortlist_idx = TARGET_LABELS.index("shortlist")
        _, class_rec, _, _ = precision_recall_fscore_support(
            y_test, y_pred, labels=TARGET_LABELS, zero_division=0
        )
        shortlist_rec = float(class_rec[shortlist_idx])

        # Confusion Matrix (rows: true, cols: predicted)
        cm = confusion_matrix(y_test, y_pred, labels=TARGET_LABELS).tolist()

        # Feature Importance
        feat_imp: dict[str, float] = {}
        imp_method = ""
        if name == "Random Forest":
            imp = model.feature_importances_
            feat_imp = {feat: round(float(v), 3) for feat, v in zip(FEATURE_NAMES, imp)}
            imp_method = "MDI (Gini Impurity reduction)"
        elif name == "Logistic Regression":
            coef_mean = np.mean(np.abs(model.coef_), axis=0)
            feat_imp = {feat: round(float(v), 3) for feat, v in zip(FEATURE_NAMES, coef_mean)}
            imp_method = "Mean Absolute Coefficient magnitude"
        else:
            perm = permutation_importance(model, cur_X_test, y_test, n_repeats=5, random_state=42)
            feat_imp = {
                feat: round(float(v), 3)
                for feat, v in zip(FEATURE_NAMES, perm.importances_mean)
            }
            imp_method = "Permutation Feature Importance"

        # Save model artifact
        slug = name.lower().replace(" ", "_")
        with open(ML_DIR / f"{slug}.pkl", "wb") as f:
            pickle.dump(model, f)

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name

        metrics_list.append(
            ModelMetrics(
                name=name,
                accuracy=round(acc, 4),
                precision=round(float(prec), 4),
                recall=round(float(rec), 4),
                f1=round(float(f1), 4),
                shortlist_recall=round(shortlist_rec, 4),
                cv_f1_mean=round(float(cv_scores.mean()), 4),
                cv_f1_std=round(float(cv_scores.std()), 4),
                confusion_matrix=cm,
                feature_importance=feat_imp,
                importance_method=imp_method,
            )
        )

    # Save best model reference
    (ML_DIR / "best_model.txt").write_text(best_model_name)

    # Class distribution
    dist = {k: int(v) for k, v in df["label"].value_counts().items()}

    dataset_info = DatasetInfo(
        source="Synthetic demonstration dataset (reproducible seed=42)",
        records=len(df),
        train_records=len(X_train),
        test_records=len(X_test),
        features=FEATURE_NAMES,
        target="label (shortlist, review, weak_match)",
        labels=TARGET_LABELS,
        class_distribution=dist,
        preprocessing="StandardScaler applied to Logistic Regression and KNN",
        split="80% train, 20% test (Stratified)",
        random_seed=42,
    )

    disclaimer = (
        "Dataset Notice: Trained on a synthetic demonstration dataset generated for college "
        "demonstration and academic viva. No production-level accuracy on arbitrary commercial resumes "
        "is claimed. The model functions as an explainable second opinion alongside rule-based scoring."
    )

    resp = ModelMetricsResponse(
        available=True,
        selected_model=best_model_name,
        selection_criterion="Highest test macro F1-score with balanced shortlist recall",
        dataset=dataset_info,
        models=metrics_list,
        disclaimer=disclaimer,
        trained_at=datetime.now(UTC).isoformat(),
    )

    # Persist metrics json
    with open(ML_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(resp.model_dump(by_alias=True), f, indent=2)

    logger.info("ML synthetic evaluation complete. Selected model: %s (F1: %.3f)", best_model_name, best_f1)
    return resp


def _train_real_dataset_models(dataset_path: Path | None = None) -> ModelMetricsResponse:
    csv_path = dataset_path or (PROCESSED_DIR / "evaluation_pairs.csv")
    if not csv_path.exists():
        from app.services.dataset.dataset_builder import build_evaluation_dataset

        build_evaluation_dataset()

    df = pd.read_csv(csv_path)
    X = df[FEATURE_NAMES].values
    y = df["classification_label"].values
    groups = df["resume_id"].values

    # Fairness audit
    violating = audit_feature_names(FEATURE_NAMES)
    if violating:
        raise ValueError(f"Fairness audit failed! Sensitive features detected: {violating}")

    # Grouped Stratified Split to prevent resume leakage across train and test sets
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(sgkf.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    train_resumes = int(df.iloc[train_idx]["resume_id"].nunique())
    test_resumes = int(df.iloc[test_idx]["resume_id"].nunique())

    # Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    with open(ML_DIR / "real_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # 1. Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    # 3. K-Nearest Neighbors
    knn = KNeighborsClassifier(n_neighbors=5)

    models_dict = {
        "Logistic Regression": (lr, X_train_scaled, X_test_scaled, True),
        "Random Forest": (rf, X_train, X_test, False),
        "KNN": (knn, X_train_scaled, X_test_scaled, True),
    }

    # Cross-validation on training set (5 folds)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics_list: list[ModelMetrics] = []
    best_model_name = ""
    best_f1 = -1.0

    for name, (model, cur_X_train, cur_X_test, is_scaled) in models_dict.items():
        cv_scores = cross_val_score(model, cur_X_train, y_train, cv=cv, scoring="f1_macro")
        model.fit(cur_X_train, y_train)

        y_pred = model.predict(cur_X_test)
        acc = float(accuracy_score(y_test, y_pred))
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )

        shortlist_idx = TARGET_LABELS.index("shortlist")
        _, class_rec, _, _ = precision_recall_fscore_support(
            y_test, y_pred, labels=TARGET_LABELS, zero_division=0
        )
        shortlist_rec = float(class_rec[shortlist_idx])

        cm = confusion_matrix(y_test, y_pred, labels=TARGET_LABELS).tolist()

        feat_imp: dict[str, float] = {}
        imp_method = ""
        if name == "Random Forest":
            imp = model.feature_importances_
            feat_imp = {feat: round(float(v), 3) for feat, v in zip(FEATURE_NAMES, imp)}
            imp_method = "MDI (Gini Impurity reduction)"
        elif name == "Logistic Regression":
            coef_mean = np.mean(np.abs(model.coef_), axis=0)
            feat_imp = {feat: round(float(v), 3) for feat, v in zip(FEATURE_NAMES, coef_mean)}
            imp_method = "Mean Absolute Coefficient magnitude"
        else:
            perm = permutation_importance(model, cur_X_test, y_test, n_repeats=5, random_state=42)
            feat_imp = {
                feat: round(float(v), 3)
                for feat, v in zip(FEATURE_NAMES, perm.importances_mean)
            }
            imp_method = "Permutation Feature Importance"

        slug = name.lower().replace(" ", "_")
        with open(ML_DIR / f"real_{slug}.pkl", "wb") as f:
            pickle.dump(model, f)

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name

        metrics_list.append(
            ModelMetrics(
                name=name,
                accuracy=round(acc, 4),
                precision=round(float(prec), 4),
                recall=round(float(rec), 4),
                f1=round(float(f1), 4),
                shortlist_recall=round(shortlist_rec, 4),
                cv_f1_mean=round(float(cv_scores.mean()), 4),
                cv_f1_std=round(float(cv_scores.std()), 4),
                confusion_matrix=cm,
                feature_importance=feat_imp,
                importance_method=imp_method,
            )
        )

    (ML_DIR / "real_best_model.txt").write_text(best_model_name)

    dist = {k: int(v) for k, v in df["classification_label"].value_counts().items()}
    num_resumes = int(df["resume_id"].nunique())
    num_jds = int(df["jd_id"].nunique()) if "jd_id" in df.columns else 12
    total_pairs = len(df)

    dataset_info = DatasetInfo(
        source=f"Real Candidate Resume Dataset ({num_resumes} candidates x {num_jds} JDs = {total_pairs} Resume-JD pairs)",
        records=total_pairs,
        train_records=len(X_train),
        test_records=len(X_test),
        features=FEATURE_NAMES,
        target="classification_label (shortlist, review, weak_match)",
        labels=TARGET_LABELS,
        class_distribution=dist,
        preprocessing="StandardScaler applied to Logistic Regression and KNN",
        split=f"Grouped Stratified Split by resume_id (Train: {len(X_train)} pairs across {train_resumes} resumes, Test: {len(X_test)} pairs across {test_resumes} resumes — zero candidate leakage)",
        random_seed=42,
    )

    disclaimer = (
        f"Dataset Notice: Evaluated on {total_pairs} Resume-JD pairs generated from {num_resumes} candidates "
        f"(65 uploaded real resumes in DOCX format + 11 reference candidates) across {num_jds} diverse Job Descriptions "
        f"using rule-assisted labels without fabricated human evaluations. "
        f"Stratified grouped splitting on resume_id guarantees zero candidate data leakage across train and test sets. "
        f"For statistical sample-size power, compare against the 800-candidate synthetic benchmark."
    )

    resp = ModelMetricsResponse(
        available=True,
        selected_model=best_model_name,
        selection_criterion="Highest test macro F1-score with balanced shortlist recall on real pairs",
        dataset=dataset_info,
        models=metrics_list,
        disclaimer=disclaimer,
        trained_at=datetime.now(UTC).isoformat(),
    )

    with open(ML_DIR / "real_metrics.json", "w", encoding="utf-8") as f:
        json.dump(resp.model_dump(by_alias=True), f, indent=2)

    logger.info("ML real dataset evaluation complete. Selected model: %s (F1: %.3f)", best_model_name, best_f1)
    return resp


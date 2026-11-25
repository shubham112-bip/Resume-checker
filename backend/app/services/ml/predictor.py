"""Inference service for ML classification, probability calibration, and explainability.

Supports:
- Version 2 (Canonical Pipeline v2, retrained on 1,512 Resume-JD pairs with class-balanced weighting)
- Version 1 (Historical Real dataset)
- Synthetic Demonstration Benchmark
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from app.config import ML_DIR
from app.schemas.api import ModelMetricsResponse
from app.schemas.evaluation import FeatureVector, MLPrediction, Recommendation
from app.services.ml.dataset import FEATURE_NAMES, TARGET_LABELS

logger = logging.getLogger(__name__)

_cached_models: dict[str, tuple[Any, Any, str, Any, str, str]] = {}


def _load_best_model(
    prefer_v2: bool = True,
    prefer_real: bool = True,
) -> tuple[Any, Any, str, Any, str, str]:
    """Load model, scaler, name, base_linear_model (for feature contributions), model_version, feature_version."""
    cache_key = "v2" if prefer_v2 else ("real" if prefer_real else "synthetic")
    if cache_key in _cached_models:
        return _cached_models[cache_key]

    # 1. Prefer v2 Retrained Models (Phase 4)
    v2_calibrated = ML_DIR / "v2_calibrated_model.pkl"
    v2_raw = ML_DIR / "v2_logistic_regression.pkl"
    v2_scaler_path = ML_DIR / "v2_scaler.pkl"
    v2_best_file = ML_DIR / "v2_best_model.txt"

    if prefer_v2 and v2_calibrated.exists() and v2_scaler_path.exists():
        with open(v2_calibrated, "rb") as f:
            model = pickle.load(f)
        with open(v2_scaler_path, "rb") as f:
            scaler = pickle.load(f)
        best_name = v2_best_file.read_text(encoding="utf-8").strip() if v2_best_file.exists() else "Logistic Regression (Balanced)"

        base_lr = None
        if v2_raw.exists():
            with open(v2_raw, "rb") as f:
                base_lr = pickle.load(f)

        res = (model, scaler, best_name, base_lr, "ml_v2", "canonical_v2")
        _cached_models[cache_key] = res
        return res

    # 2. Historical Real Model (Phase 2)
    real_best_file = ML_DIR / "real_best_model.txt"
    synth_best_file = ML_DIR / "best_model.txt"

    use_real = prefer_real and real_best_file.exists()
    best_file = real_best_file if use_real else synth_best_file

    if not best_file.exists():
        from app.services.ml.model_trainer import train_and_evaluate_models

        train_and_evaluate_models(dataset_type="real" if prefer_real else "synthetic")
        best_file = real_best_file if use_real and real_best_file.exists() else synth_best_file

    best_name = best_file.read_text().strip()
    slug = best_name.lower().replace(" ", "_")
    prefix = "real_" if use_real else ""
    model_path = ML_DIR / f"{prefix}{slug}.pkl"
    scaler_path = ML_DIR / f"{prefix}scaler.pkl"

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    scaler = None
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

    res = (model, scaler, best_name, model, "ml_v1" if use_real else "synthetic", "canonical_v1")
    _cached_models[cache_key] = res
    return res


def predict_candidate_fit(
    features: FeatureVector,
    rule_recommendation: Recommendation,
    prefer_v2: bool = True,
    prefer_real: bool = True,
) -> MLPrediction:
    try:
        model, scaler, model_name, base_lr, model_ver, feat_ver = _load_best_model(
            prefer_v2=prefer_v2, prefer_real=prefer_real
        )
    except Exception as e:
        logger.warning("ML model could not be loaded: %s", e)
        return MLPrediction(available=False, detail=str(e))

    raw_features = [
        features.required_skill_match,
        features.experience_match,
        features.semantic_similarity,
        features.education_match,
        features.project_match,
        features.ats_score,
        features.resume_quality,
    ]

    arr = np.array(raw_features).reshape(1, -1)
    arr_scaled = arr
    if scaler is not None:
        arr_scaled = scaler.transform(arr)

    # Prediction
    pred_label = str(model.predict(arr_scaled)[0])
    probs: dict[str, float] = {}

    if hasattr(model, "predict_proba"):
        p = model.predict_proba(arr_scaled)[0]
        classes = list(model.classes_)
        for c, prob in zip(classes, p):
            probs[c] = round(float(prob), 4)

    # Feature Contributions (Explainability)
    contributions: dict[str, float] = {}
    if base_lr is not None and hasattr(base_lr, "coef_") and hasattr(base_lr, "classes_"):
        try:
            pred_idx = list(base_lr.classes_).index(pred_label)
            coefs = base_lr.coef_[pred_idx]
            for feat_name, c_val, f_val in zip(FEATURE_NAMES, coefs, arr_scaled[0]):
                contributions[feat_name] = round(float(c_val * f_val), 4)
        except Exception as e:
            logger.debug("Could not compute linear feature contributions: %s", e)

    # Check agreement with rule recommendation
    # Rule recommendation: STRONG_SHORTLIST / SHORTLIST -> "shortlist"
    # REVIEW -> "review"
    # WEAK_MATCH -> "weak_match"
    rule_mapped: str
    if rule_recommendation in {Recommendation.STRONG_SHORTLIST, Recommendation.SHORTLIST}:
        rule_mapped = "shortlist"
    elif rule_recommendation == Recommendation.REVIEW:
        rule_mapped = "review"
    else:
        rule_mapped = "weak_match"

    agrees = (pred_label == rule_mapped)
    detail = (
        f"ML model [{model_ver}] ({model_name}) predicts '{pred_label}' (calibrated confidence: {probs.get(pred_label, 0.0):.2f}). "
        f"{'Agrees with' if agrees else 'Disagrees with'} rule-based recommendation '{rule_mapped}'."
    )

    return MLPrediction(
        available=True,
        model_name=model_name,
        predicted_label=pred_label,
        probabilities=probs,
        agrees_with_rules=agrees,
        detail=detail,
        model_version=model_ver,
        feature_version=feat_ver,
        feature_contributions=contributions,
    )


def get_cached_metrics(dataset_type: str = "real") -> ModelMetricsResponse:
    if dataset_type in {"v2", "canonical_v2", "real", "canonical"}:
        target_file = ML_DIR / "v2_model_metrics.json"
        if not target_file.exists():
            from scripts.train_v2_ml_models import run_training_and_comparison

            run_training_and_comparison()

        data = json.loads(target_file.read_text(encoding="utf-8"))
        if "available" not in data:
            models_list = []
            for m_name, m_dict in data.get("model_evaluations", {}).items():
                models_list.append({
                    "name": m_name,
                    "accuracy": m_dict["accuracy"],
                    "precision": m_dict["macro_precision"],
                    "recall": m_dict["macro_recall"],
                    "f1": m_dict["macro_f1"],
                    "shortlist_recall": m_dict["shortlist_recall"],
                    "cv_f1_mean": m_dict.get("cv_macro_f1_mean", 0.0),
                    "cv_f1_std": m_dict.get("cv_macro_f1_std", 0.0),
                    "confusion_matrix": m_dict["confusion_matrix"],
                    "feature_importance": m_dict.get("feature_importance", {}),
                    "importance_method": m_dict.get("importance_method", "Coefficients"),
                })
            meta = data.get("dataset_metadata", {})
            return ModelMetricsResponse(
                available=True,
                selected_model=data.get("selected_best_model", "Logistic Regression (Balanced)"),
                selection_criterion="Highest test macro F1-score with balanced shortlist recall on Canonical v2 features",
                dataset={
                    "source": "Real Corpus Canonical Feature Dataset (1,512 Resume-JD pairs)",
                    "records": meta.get("total_pairs", 1512),
                    "train_records": meta.get("train_pairs", 1212),
                    "test_records": meta.get("test_pairs", 300),
                    "features": meta.get("features", FEATURE_NAMES),
                    "target": "classification_label",
                    "labels": meta.get("target_classes", TARGET_LABELS),
                    "class_distribution": meta.get("class_distribution", {}),
                    "preprocessing": "StandardScaler applied to LR and KNN",
                    "split": "Group-aware StratifiedGroupKFold by resume_id (zero leakage)",
                    "random_seed": 42,
                },
                models=models_list,
                calibration_comparison=data.get("calibration_comparison"),
                disclaimer="Notice: Evaluated on Canonical Pipeline v2 features across 1,512 pairs. Benchmarks evaluate model agreement with rule-assisted screening policy labels.",
                trained_at=meta.get("dataset_version", "canonical_v2"),
            )
        return ModelMetricsResponse.model_validate(data)

    target_file = ML_DIR / "real_metrics.json" if dataset_type == "real" else ML_DIR / "metrics.json"
    if not target_file.exists():
        from app.services.ml.model_trainer import train_and_evaluate_models

        train_and_evaluate_models(dataset_type=dataset_type)

    data = json.loads(target_file.read_text(encoding="utf-8"))
    return ModelMetricsResponse.model_validate(data)

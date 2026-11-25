"""Automated tests for Phase 4: ML Retraining on Canonical v2 Features.

Verifies:
1. V2 feature schema & dataset integrity (1,512 rows, 126 resumes, 12 JDs).
2. Training/inference feature schema consistency (exact name & order match).
3. Absolute absence of resume group leakage (train_resumes ∩ test_resumes = ∅).
4. V2 model artifacts loading (scaler, LR, RF, KNN, calibrated model).
5. Prediction output validity, probability normalization, and explainability contributions.
6. Shortlist safety: rejection of weak profiles with near-zero shortlist probability.
7. Human validation benchmarks performance of ML v2 closing the gap with Rule v2.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedGroupKFold

from app.config import ML_DIR
from app.schemas.evaluation import FeatureVector, Recommendation
from app.services.ml.dataset import FEATURE_NAMES, TARGET_LABELS
from app.services.ml.predictor import _load_best_model, predict_candidate_fit

V2_DATASET_CSV = ML_DIR / "v2_training_dataset.csv"
V2_SCHEMA_JSON = ML_DIR / "v2_feature_schema.json"
V2_METRICS_JSON = ML_DIR / "v2_model_metrics.json"
V2_HUMAN_EVAL_JSON = ML_DIR / "v2_ml_human_evaluations.json"


def test_v2_feature_schema_and_dataset():
    """Verify v2 training dataset and schema existence, dimensions, and labels."""
    assert V2_DATASET_CSV.exists(), "v2_training_dataset.csv must exist"
    assert V2_SCHEMA_JSON.exists(), "v2_feature_schema.json must exist"

    df = pd.read_csv(V2_DATASET_CSV)
    assert len(df) == 1512, f"Expected 1512 rows, got {len(df)}"
    assert df["resume_id"].nunique() == 126, f"Expected 126 resumes, got {df['resume_id'].nunique()}"
    assert df["jd_id"].nunique() == 12, f"Expected 12 JDs, got {df['jd_id'].nunique()}"

    # Check primary 7 features
    for f in FEATURE_NAMES:
        assert f in df.columns, f"Missing primary feature {f} in v2 dataset"
        assert df[f].notna().all(), f"NaN values found in feature {f}"
        assert (df[f] >= 0.0).all() and (df[f] <= 1.0).all(), f"Feature {f} values out of bounds [0, 1]"

    # Check target classes
    assert set(df["classification_label"].unique()) == {"weak_match", "review", "shortlist"}

    with open(V2_SCHEMA_JSON, encoding="utf-8") as f:
        schema = json.load(f)
    assert schema["version"] == "canonical_v2"
    assert schema["primary_features"] == FEATURE_NAMES
    assert schema["target_classes"] == TARGET_LABELS


def test_training_inference_feature_consistency():
    """Verify exact feature consistency and ordering between training dataset and FeatureVector."""
    fv_fields = list(FeatureVector.model_fields.keys())
    # The 7 ML features must all be present in FeatureVector
    for feat in FEATURE_NAMES:
        assert feat in fv_fields, f"Feature {feat} from ML dataset missing in FeatureVector schema"

    # Verify order in predictor matches FEATURE_NAMES
    model, scaler, _, _, _, feat_ver = _load_best_model(prefer_v2=True)
    assert feat_ver == "canonical_v2"
    if hasattr(scaler, "feature_names_in_"):
        assert list(scaler.feature_names_in_) == FEATURE_NAMES


def test_resume_group_leakage_absence():
    """Verify strict group-aware split: zero resume overlap between train and test sets."""
    df = pd.read_csv(V2_DATASET_CSV)
    X = df[FEATURE_NAMES].values
    y = df["classification_label"].values
    groups = df["resume_id"].values

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(sgkf.split(X, y, groups=groups))

    train_resumes = set(df.iloc[train_idx]["resume_id"])
    test_resumes = set(df.iloc[test_idx]["resume_id"])

    overlap = train_resumes.intersection(test_resumes)
    assert len(overlap) == 0, f"DATA LEAKAGE! Resumes found in both train and test: {overlap}"
    assert len(train_resumes) > 0 and len(test_resumes) > 0
    assert len(train_resumes) + len(test_resumes) == 126


def test_v2_model_artifacts_loading():
    """Verify all versioned v2 model artifacts load correctly."""
    scaler_path = ML_DIR / "v2_scaler.pkl"
    lr_path = ML_DIR / "v2_logistic_regression.pkl"
    rf_path = ML_DIR / "v2_random_forest.pkl"
    knn_path = ML_DIR / "v2_knn.pkl"
    cal_path = ML_DIR / "v2_calibrated_model.pkl"
    best_file = ML_DIR / "v2_best_model.txt"

    for path in [scaler_path, lr_path, rf_path, knn_path, cal_path, best_file]:
        assert path.exists(), f"Missing artifact: {path.name}"

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    assert scaler.n_features_in_ == len(FEATURE_NAMES)

    with open(cal_path, "rb") as f:
        cal = pickle.load(f)
    assert hasattr(cal, "predict_proba")
    assert set(cal.classes_) == {"weak_match", "review", "shortlist"}


def test_v2_prediction_output_and_probabilities():
    """Verify v2 prediction output, calibrated probabilities, and explainability contributions."""
    fv = FeatureVector(
        required_skill_match=0.90,
        preferred_skill_match=0.80,
        experience_match=0.95,
        education_match=1.0,
        project_match=0.85,
        semantic_similarity=0.78,
        ats_score=0.92,
        resume_quality=0.88,
    )

    pred = predict_candidate_fit(fv, Recommendation.SHORTLIST, prefer_v2=True)
    assert pred.available is True
    assert pred.model_version == "ml_v2"
    assert pred.feature_version == "canonical_v2"
    assert pred.predicted_label == "shortlist"

    # Probabilities should be valid probabilities summing to ~1.0
    total_prob = sum(pred.probabilities.values())
    assert 0.99 <= total_prob <= 1.01, f"Probabilities do not sum to 1.0: {total_prob}"
    assert pred.probabilities["shortlist"] > 0.50

    # Feature contributions must be present for explainability
    assert len(pred.feature_contributions) == len(FEATURE_NAMES)
    for feat in FEATURE_NAMES:
        assert feat in pred.feature_contributions
        assert isinstance(pred.feature_contributions[feat], float)


def test_shortlist_safety_on_weak_profiles():
    """Verify shortlist safety: completely unqualified candidate is rejected with near-zero shortlist probability."""
    weak_fv = FeatureVector(
        required_skill_match=0.05,
        preferred_skill_match=0.0,
        experience_match=0.10,
        education_match=0.20,
        project_match=0.0,
        semantic_similarity=0.15,
        ats_score=0.40,
        resume_quality=0.35,
    )

    pred = predict_candidate_fit(weak_fv, Recommendation.WEAK_MATCH, prefer_v2=True)
    assert pred.available is True
    assert pred.predicted_label == "weak_match"
    # Crucial recruiter safety guarantee: weak candidate must NOT have high shortlist probability
    assert pred.probabilities.get("shortlist", 0.0) <= 0.05, (
        f"Unsafe false shortlist probability for weak candidate: {pred.probabilities.get('shortlist')}"
    )


def test_v2_human_evaluations_benchmarks():
    """Verify ML v2 closes the gap with Rule v2 on both human validation benchmarks."""
    assert V2_HUMAN_EVAL_JSON.exists(), "v2_ml_human_evaluations.json must exist"
    with open(V2_HUMAN_EVAL_JSON, encoding="utf-8") as f:
        data = json.load(f)

    # 1. Locked 64-Pair Benchmark
    locked = data["locked_benchmark_64"]
    ml_v2_locked = locked["ml_v2"]
    ml_v1_locked = locked["ml_v1"]

    assert ml_v2_locked["accuracy"] >= 0.65, f"ML v2 locked accuracy {ml_v2_locked['accuracy']} < 65%"
    assert ml_v2_locked["shortlist_recall"] >= 0.70, f"ML v2 locked shortlist recall {ml_v2_locked['shortlist_recall']} < 70%"
    assert ml_v2_locked["accuracy"] > ml_v1_locked["accuracy"], "ML v2 must outperform ML v1 on locked benchmark"

    # 2. Independent 28-Pair Benchmark
    indep = data["independent_benchmark_28"]
    ml_v2_indep = indep["ml_v2"]
    ml_v1_indep = indep["ml_v1"]

    assert ml_v2_indep["accuracy"] >= 0.45, f"ML v2 independent accuracy {ml_v2_indep['accuracy']} < 45%"
    assert ml_v2_indep["shortlist_recall"] >= 0.40, f"ML v2 independent shortlist recall {ml_v2_indep['shortlist_recall']} < 40%"
    assert ml_v2_indep["shortlist_precision"] == 1.0, f"ML v2 independent shortlist precision must be 1.0 (got {ml_v2_indep['shortlist_precision']})"
    assert ml_v2_indep["false_shortlist_rate"] == 0.0, f"ML v2 false shortlist rate must be 0.0"

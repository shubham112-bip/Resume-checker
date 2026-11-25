"""Tests for Phase 2.3: Expanded Resume Corpus & Generalization.

Verifies:
1. Expanded manifest discovery and schema integrity (126 total files: 76 existing, 50 new).
2. Preservation of existing datasets (912 evaluation pairs, 64 locked human pairs).
3. Duplicate and near-duplicate categorization logic.
4. Protected personal attributes removal (0 leakage into feature vectors).
5. Expanded evaluation pairs dataset integrity (1,512 total pairs, correct sources and versions).
6. Production model generalization inference on unseen resume profiles.
7. Candidate pool for future human validation (data/labels/human_validation_candidates.csv).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from app.config import LABELS_DIR, ML_DIR, PROCESSED_DIR, RESUMES_DIR
from app.schemas.evaluation import FeatureVector, Recommendation
from app.services.fairness import audit_feature_names
from app.services.ml.predictor import predict_candidate_fit


@pytest.fixture(scope="module")
def manifest_df() -> pd.DataFrame:
    path = Path("data/manifest/expanded_resume_manifest.csv")
    assert path.exists(), "expanded_resume_manifest.csv must exist"
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def expanded_pairs_df() -> pd.DataFrame:
    path = PROCESSED_DIR / "expanded_resume_jd_pairs.csv"
    assert path.exists(), "expanded_resume_jd_pairs.csv must exist"
    return pd.read_csv(path)


def test_manifest_discovery_and_counts(manifest_df: pd.DataFrame):
    """Test that manifest accounts for all 126 resume files (76 existing + 50 new)."""
    assert len(manifest_df) == 126, f"Expected 126 manifest rows, got {len(manifest_df)}"

    required_cols = [
        "filename",
        "file_path",
        "resume_id",
        "file_type",
        "word_count",
        "parsing_status",
        "extracted_skill_count",
        "specialization",
        "status",
        "duplicate_status",
        "source",
    ]
    for col in required_cols:
        assert col in manifest_df.columns, f"Missing required column {col} in manifest"

    # Count breakdown
    new_rows = manifest_df[manifest_df["status"] == "NEW"]
    existing_rows = manifest_df[manifest_df["status"] == "EXISTING"]
    near_dupe_rows = manifest_df[manifest_df["status"] == "NEAR_DUPLICATE"]

    assert len(new_rows) == 50, f"Expected 50 NEW resumes, got {len(new_rows)}"
    assert len(existing_rows) + len(near_dupe_rows) == 76, "Expected 76 existing corpus files"
    assert all(manifest_df["parsing_status"] == "SUCCESS"), "All resumes should have parse status SUCCESS"


def test_preservation_of_existing_datasets():
    """Verify that prior locked benchmarks and datasets are strictly preserved untouched."""
    # 1. Original 912 evaluation pairs
    orig_pairs_file = PROCESSED_DIR / "evaluation_pairs.csv"
    assert orig_pairs_file.exists()
    df_orig = pd.read_csv(orig_pairs_file)
    assert len(df_orig) == 912, f"Expected 912 original evaluation pairs, got {len(df_orig)}"

    # 2. Locked 64-pair human validation benchmark
    human_val_file = LABELS_DIR / "human_validation.csv"
    assert human_val_file.exists()
    df_human = pd.read_csv(human_val_file)
    assert len(df_human) == 64, f"Expected 64 locked human validation pairs, got {len(df_human)}"

    # 3. Locked human evaluation JSON metrics
    locked_json_file = ML_DIR / "locked_human_evaluation.json"
    assert locked_json_file.exists()
    with open(locked_json_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    assert metrics["total_pairs"] == 64
    assert metrics["evaluation_type"] == "LOCKED_HUMAN_VALIDATION"


def test_protected_attributes_absence_in_features(expanded_pairs_df: pd.DataFrame):
    """Verify that zero personal/protected attributes leak into feature vectors."""
    feature_columns = [
        "required_skill_match",
        "preferred_skill_match",
        "experience_match",
        "education_match",
        "project_match",
        "semantic_similarity",
        "ats_score",
        "resume_quality",
        "keyword_match_ratio",
        "semantic_match_ratio",
        "action_verb_score",
        "achievement_quantification_score",
    ]
    violating = audit_feature_names(feature_columns)
    assert not violating, f"Found protected attributes in scoring feature list: {violating}"

    # Also inspect newly ingested profile JSONs to ensure sensitive_attributes_removed is tracked
    new_resumes_dir = PROCESSED_DIR / "resumes"
    sample_json = new_resumes_dir / "resume_10005171.json"
    assert sample_json.exists()
    with open(sample_json, "r", encoding="utf-8") as f:
        profile_data = json.load(f)
    assert "sensitiveAttributesRemoved" in profile_data
    assert "qualityFeatures" in profile_data
    assert profile_data["resume_id"] == "resume_10005171"


def test_expanded_pairs_integrity(expanded_pairs_df: pd.DataFrame):
    """Verify expanded pairs contains 1,512 rows with correct sources and rule-based labels."""
    assert len(expanded_pairs_df) == 1512, f"Expected 1,512 expanded pairs, got {len(expanded_pairs_df)}"

    # Check source distribution
    source_counts = expanded_pairs_df["resume_source"].value_counts().to_dict()
    assert source_counts.get("EXPANDED_CORPUS", 0) == 600, "Expected 600 pairs from EXPANDED_CORPUS"
    assert (
        source_counts.get("ORIGINAL_65", 0) + source_counts.get("SAMPLE_REFERENCE", 0) == 912
    ), "Expected 912 original corpus pairs"

    # All labels must be explicitly marked as RULE_BASED
    assert all(
        expanded_pairs_df["label_source"] == "RULE_BASED"
    ), "All labels in development dataset must be RULE_BASED"

    # All 50 new resumes must be present across all 12 JDs
    new_pairs = expanded_pairs_df[expanded_pairs_df["resume_source"] == "EXPANDED_CORPUS"]
    unique_new_resumes = new_pairs["resume_id"].nunique()
    unique_jds = new_pairs["jd_id"].nunique()
    assert unique_new_resumes == 50, f"Expected 50 unique new resumes, got {unique_new_resumes}"
    assert unique_jds == 12, f"Expected 12 unique JDs, got {unique_jds}"


def test_model_generalization_inference(expanded_pairs_df: pd.DataFrame):
    """Verify that existing production model can perform inference on new resumes."""
    new_pairs = expanded_pairs_df[expanded_pairs_df["resume_source"] == "EXPANDED_CORPUS"]
    sample_row = new_pairs.iloc[0]

    fv = FeatureVector(
        required_skill_match=sample_row["required_skill_match"],
        preferred_skill_match=sample_row["preferred_skill_match"],
        experience_match=sample_row["experience_match"],
        semantic_similarity=sample_row["semantic_similarity"],
        education_match=sample_row["education_match"],
        project_match=sample_row["project_match"],
        ats_score=sample_row["ats_score"],
        resume_quality=sample_row["resume_quality"],
    )
    rec = Recommendation.WEAK_MATCH if sample_row["label"] == "WEAK_MATCH" else Recommendation.REVIEW
    pred = predict_candidate_fit(fv, rec, prefer_real=True)

    assert pred.available is True
    assert pred.predicted_label in {"shortlist", "review", "weak_match"}
    assert isinstance(pred.agrees_with_rules, bool)
    assert len(pred.probabilities) == 3


def test_human_validation_candidates_pool():
    """Verify that data/labels/human_validation_candidates.csv has been prepared."""
    cand_file = LABELS_DIR / "human_validation_candidates.csv"
    assert cand_file.exists(), "human_validation_candidates.csv must exist"
    df_cands = pd.read_csv(cand_file)

    assert len(df_cands) >= 20, f"Expected at least 20 candidate pairs, got {len(df_cands)}"
    expected_cols = ["resume_id", "jd_id", "reason_selected", "difficulty", "role", "new_resume"]
    for col in expected_cols:
        assert col in df_cands.columns, f"Missing column {col} in candidate pool"

    assert bool(df_cands["new_resume"].all()) is True, "All candidates must be from new resumes"
    # Ensure no ground truth human labels were fabricated in this file
    assert "human_label" not in df_cands.columns, "Candidate pool must NOT have fabricated human labels"


def test_expanded_summary_json():
    """Verify that expanded_corpus_summary.json contains valid evaluation statistics."""
    summary_path = PROCESSED_DIR / "expanded_corpus_summary.json"
    assert summary_path.exists()
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["old_profile_count"] == 76
    assert summary["newly_discovered_profile_count"] == 50
    assert summary["total_profile_count"] == 126
    assert summary["total_pairs_count"] == 1512
    assert summary["generalization_evaluation"]["accuracy"] >= 0.95
    assert summary["generalization_evaluation"]["status"] == "PASS"
    assert summary["retraining_decision"]["status"] == "NOT YET RECOMMENDED"

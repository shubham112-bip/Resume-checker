"""Automated tests for Phase 2.2: Human-Validated Evaluation & Model Calibration.

Verifies:
1. Human validation CSV schema, non-null values, valid labels.
2. Zero duplicate Resume-JD pairs.
3. Data leakage controls: human validation labels not used in ML training.
4. Protected attributes excluded from predictive feature space.
5. REST endpoint GET /api/evaluation/human-validation returns complete benchmarks.
6. Production REST endpoint POST /api/screen evaluates unseen pairs seamlessly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import ML_DIR, PROCESSED_DIR, ROOT_DATA_DIR
from app.main import app
from app.schemas.evaluation import CandidateEvaluation
from app.services.fairness import audit_feature_names
from app.services.ml.dataset import FEATURE_NAMES

client = TestClient(app)
HUMAN_VAL_CSV = ROOT_DATA_DIR / "labels" / "human_validation.csv"
LOCKED_EVAL_JSON = ML_DIR / "locked_human_evaluation.json"


def test_human_validation_csv_schema():
    """Verify human_validation.csv schema, minimum row count, and non-empty columns."""
    assert HUMAN_VAL_CSV.exists(), "human_validation.csv must exist in data/labels/"
    df = pd.read_csv(HUMAN_VAL_CSV)

    expected_cols = {"resume_id", "jd_id", "human_label", "reviewer_id", "notes", "evidence"}
    assert expected_cols.issubset(set(df.columns)), f"Missing columns in CSV: {expected_cols - set(df.columns)}"

    # Must contain 50 to 100 pairs
    assert 50 <= len(df) <= 100, f"Expected 50-100 pairs, got {len(df)}"
    assert len(df) == 64, f"Expected 64 balanced pairs, got {len(df)}"

    # Check non-null values
    for col in ["resume_id", "jd_id", "human_label", "reviewer_id", "notes", "evidence"]:
        assert df[col].notna().all(), f"Column '{col}' contains NaN values"
        assert (df[col].str.strip() != "").all(), f"Column '{col}' contains empty strings"


def test_valid_human_label_values():
    """Verify human_label contains only the 4 sanctioned assessment categories."""
    df = pd.read_csv(HUMAN_VAL_CSV)
    valid_labels = {"STRONG_MATCH", "MATCH", "REVIEW", "WEAK_MATCH"}
    observed_labels = set(df["human_label"].unique())

    assert observed_labels == valid_labels, f"Expected exactly {valid_labels}, got {observed_labels}"

    # Check balanced coverage across all 4 classes
    counts = df["human_label"].value_counts().to_dict()
    for lbl in valid_labels:
        assert counts.get(lbl, 0) >= 10, f"Insufficient coverage for class '{lbl}': {counts.get(lbl, 0)}"


def test_no_duplicate_validation_pairs():
    """Verify there are zero duplicate Resume-JD pairs in the validation set."""
    df = pd.read_csv(HUMAN_VAL_CSV)
    duplicates = df.duplicated(subset=["resume_id", "jd_id"]).sum()
    assert duplicates == 0, f"Found {duplicates} duplicate Resume-JD pairs in human validation dataset"


def test_data_leakage_controls():
    """Verify that human validation labels were not used during development model training."""
    # Check that development training pairs exclusively used rule-based labels
    dev_pairs_file = PROCESSED_DIR / "evaluation_pairs.csv"
    assert dev_pairs_file.exists(), "evaluation_pairs.csv must exist"

    dev_df = pd.read_csv(dev_pairs_file)
    assert (dev_df["label_source"] == "RULE_BASED").all(), "Development training set must be 100% RULE_BASED"

    # Verify locked human evaluation artifact exists
    assert LOCKED_EVAL_JSON.exists(), "locked_human_evaluation.json must exist"
    locked_data = json.loads(LOCKED_EVAL_JSON.read_text(encoding="utf-8"))
    assert locked_data["evaluation_type"] == "LOCKED_HUMAN_VALIDATION"
    assert locked_data["reviewer_count"] == 1
    assert locked_data["reviewer_id"] == "human_reviewer_01"
    assert locked_data["total_pairs"] == 64


def test_protected_attributes_excluded():
    """Verify protected personal attributes are completely excluded from predictive features."""
    violations = audit_feature_names(FEATURE_NAMES)
    assert len(violations) == 0, f"Sensitive attributes detected in feature set: {violations}"


def test_human_validation_api_endpoint():
    """Verify GET /api/evaluation/human-validation returns complete locked benchmarks."""
    res = client.get("/api/evaluation/human-validation")
    assert res.status_code == 200
    data = res.json()

    assert data["available"] is True
    assert data["totalPairs"] == 64
    assert data["totalResumes"] == 45
    assert data["totalJds"] == 12
    assert data["reviewerCount"] == 1
    assert data["reviewerId"] == "human_reviewer_01"

    # Check metrics
    assert "mlVsHuman" in data
    assert "accuracy" in data["mlVsHuman"]
    assert "macro_f1" in data["mlVsHuman"] or "macroF1" in data["mlVsHuman"]
    assert "shortlist_precision" in data["mlVsHuman"] or "shortlistPrecision" in data["mlVsHuman"]
    assert "confusion_matrix" in data["mlVsHuman"] or "confusionMatrix" in data["mlVsHuman"]

    # Shortlist precision should be strong (low false positives)
    sl_p = data["mlVsHuman"].get("shortlist_precision", data["mlVsHuman"].get("shortlistPrecision", 0))
    assert sl_p >= 0.70

    # Notice must clearly distinguish development benchmark from human benchmark
    assert "Development Benchmark" in data["notice"]
    assert "98.96%" in data["notice"]
    assert "Human Validation Benchmark" in data["notice"]

    # Check pairs list
    assert len(data["pairs"]) == 64
    first_pair = data["pairs"][0]
    assert "humanLabel" in first_pair
    assert "mlPrediction" in first_pair
    assert "notes" in first_pair
    assert len(first_pair["notes"]) > 0


def test_api_screen_arbitrary_unseen_candidate_intact():
    """Verify POST /api/screen continues accepting arbitrary unseen resumes and JDs."""
    res = client.post(
        "/api/screen",
        data={
            "resume_text": (
                "David Miller\n"
                "Cloud & Systems Engineer\n\n"
                "SUMMARY\n"
                "4 years experience deploying microservices on AWS with Docker and Kubernetes.\n\n"
                "SKILLS\n"
                "AWS, Kubernetes, Docker, Terraform, Python, Linux, CI/CD.\n\n"
                "PROFESSIONAL EXPERIENCE\n"
                "DevOps Engineer | CloudScale Inc (2020 - 2024)\n"
                "- Built automated CI/CD pipelines deploying containerized services on EKS.\n\n"
                "EDUCATION\n"
                "B.S. in Computer Science (2016 - 2020)\n"
            ),
            "job_description": (
                "Role: DevOps Engineer\n"
                "Requirements: AWS, Kubernetes, Docker, CI/CD (Required). 3+ years experience."
            ),
            "job_title": "DevOps Engineer",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert "overallScore" in data
    assert data["overallScore"] > 60.0
    assert "matchedSkills" in data
    assert "AWS" in data["matchedSkills"] or "Docker" in data["matchedSkills"]
    assert data["mlPrediction"]["available"] is True

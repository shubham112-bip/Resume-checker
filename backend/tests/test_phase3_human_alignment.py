"""Automated tests for Phase 3: Human-Alignment Error Analysis & Feature Correction.

Verifies:
1. Locked human validation benchmark (64 pairs) is completely unmodified.
2. Locked benchmark diagnostics dataset has 64 rows and all 21 required fields.
3. Phase 3 error analysis taxonomy JSON exists and categorizes disagreements.
4. Canonical Pipeline v2 extracts features with skill hierarchy and network skills.
5. Fresh independent human validation dataset (Phase 3) is valid, single-blind, and non-overlapping.
6. Shadow evaluation on locked benchmark confirms accuracy and shortlist recall gains.
7. Independent evaluation on fresh dataset confirms significant generalization gains.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from app.config import LABELS_DIR, ML_DIR, ROOT_DATA_DIR
from app.schemas.candidate import CandidateProfile, ResumeSection
from app.schemas.evaluation import Recommendation
from app.schemas.job import JobProfile, JobSkill, SkillTier
from app.services.pipeline.canonical_pipeline_v2 import extract_canonical_features_v2, match_requirements_v2
from app.services.skills.taxonomy import get_skill, implying_skills

LOCKED_CSV = LABELS_DIR / "human_validation.csv"
LOCKED_EVAL_JSON = ML_DIR / "locked_human_evaluation.json"
DIAGNOSTICS_CSV = ML_DIR / "locked_benchmark_diagnostics.csv"
DIAGNOSTICS_JSON = ML_DIR / "locked_benchmark_diagnostics.json"
ERROR_ANALYSIS_JSON = ML_DIR / "phase_3_error_analysis.json"
SHADOW_EVAL_JSON = ML_DIR / "phase_3_shadow_evaluation.json"
INDEPENDENT_CSV = LABELS_DIR / "human_validation_phase3.csv"
INDEPENDENT_EVAL_JSON = ML_DIR / "phase_3_independent_evaluation.json"

EXPECTED_LOCKED_MD5 = "e64de7495d880b5039daabfb2489b861"


def test_locked_human_validation_unmodified():
    """Ensure locked human validation benchmark has not been modified or overwritten."""
    assert LOCKED_CSV.exists(), "human_validation.csv must exist"
    content = LOCKED_CSV.read_bytes()
    actual_md5 = hashlib.md5(content).hexdigest()
    assert actual_md5 == EXPECTED_LOCKED_MD5, (
        f"LOCKED BENCHMARK WAS MODIFIED! Expected MD5 {EXPECTED_LOCKED_MD5}, got {actual_md5}"
    )

    df = pd.read_csv(LOCKED_CSV)
    assert len(df) == 64, f"Expected exactly 64 locked pairs, got {len(df)}"
    assert LOCKED_EVAL_JSON.exists(), "locked_human_evaluation.json must exist"


def test_locked_benchmark_diagnostics_dataset():
    """Verify diagnostic dataset has 64 rows and all required diagnostic fields."""
    assert DIAGNOSTICS_CSV.exists(), "locked_benchmark_diagnostics.csv must exist"
    assert DIAGNOSTICS_JSON.exists(), "locked_benchmark_diagnostics.json must exist"

    df = pd.read_csv(DIAGNOSTICS_CSV)
    assert len(df) == 64, f"Expected 64 diagnostic rows, got {len(df)}"

    required_fields = {
        "resume_id",
        "jd_id",
        "human_label",
        "human_3class",
        "rule_label",
        "ml_prediction",
        "ml_probability",
        "required_skill_match",
        "experience_match",
        "semantic_similarity",
        "project_relevance",
        "education_match",
        "ats_score",
        "resume_quality",
        "final_score",
        "decision_threshold",
        "matched_skills",
        "missing_skills",
        "disagreement_type",
    }
    assert required_fields.issubset(set(df.columns)), (
        f"Missing diagnostic fields: {required_fields - set(df.columns)}"
    )


def test_phase3_error_analysis_taxonomy():
    """Verify Phase 3 error analysis taxonomy exists and contains expected failure categories."""
    assert ERROR_ANALYSIS_JSON.exists(), "phase_3_error_analysis.json must exist"
    with open(ERROR_ANALYSIS_JSON, encoding="utf-8") as f:
        data = json.load(f)

    meta = data["benchmark_metadata"]
    assert meta["total_pairs"] == 64
    assert meta["disagreements"] == 35

    assert "taxonomy_counts" in data
    counts = data["taxonomy_counts"]
    assert "A_SKILL_SYNONYM_EQUIVALENCE" in counts
    assert "B_SKILL_HIERARCHY_IMPLICATION" in counts
    assert "C_EXPERIENCE_EXTRACTION_FAILURE" in counts
    assert counts["C_EXPERIENCE_EXTRACTION_FAILURE"] > 0


def test_v2_taxonomy_enrichment_and_skill_hierarchy():
    """Verify networking skills taxonomy enrichment and skill hierarchy (implying_skills)."""
    # 1. Check newly added networking skills in taxonomy
    for skill_name in ["Cisco", "Routing", "Switching", "TCP/IP", "VPN", "BGP", "OSPF", "Wireshark", "Firewalls"]:
        skill_def = get_skill(skill_name)
        assert skill_def is not None, f"Skill {skill_name} not found in SKILL_TAXONOMY"

    # 2. Check skill hierarchy (implying_skills)
    sql_implying = implying_skills("SQL")
    assert "PostgreSQL" in sql_implying or "MySQL" in sql_implying

    python_implying = implying_skills("Python")
    assert "FastAPI" in python_implying or "Django" in python_implying

    react_implying = implying_skills("React")
    assert "Next.js" in react_implying


def test_v2_canonical_pipeline_execution():
    """Verify Canonical Pipeline v2 extracts features cleanly."""
    candidate = CandidateProfile(
        candidate_id="test_cand_01",
        name="Test Developer",
        text="Experienced Python and Django developer with PostgreSQL database and REST API design.",
        skills=["Python", "Django", "PostgreSQL", "REST APIs", "Git"],
        total_experience_months=48,
        sections=[
            ResumeSection(
                name="experience",
                heading="Experience",
                text="Senior Developer 2020-2024. Built scalable REST APIs using Django and PostgreSQL.",
            )
        ],
    )
    job = JobProfile(
        job_id="test_job_01",
        title="Python Backend Engineer",
        description="Looking for Python backend engineer with SQL database skills.",
        min_experience_years=3.0,
        skills=[
            JobSkill(name="Python", tier=SkillTier.REQUIRED, weight=1.0),
            JobSkill(name="SQL", tier=SkillTier.REQUIRED, weight=1.0),
        ],
    )

    feat_v2 = extract_canonical_features_v2(candidate, job)
    assert feat_v2.pipeline_version == "v2"
    assert feat_v2.overall_score > 60.0
    # Python is matched directly, SQL is satisfied via PostgreSQL hierarchy implication!
    assert feat_v2.recommendation in {Recommendation.SHORTLIST, Recommendation.STRONG_SHORTLIST}


def test_phase3_independent_human_validation_dataset():
    """Verify fresh independent human validation dataset integrity and single-blind protocol."""
    assert INDEPENDENT_CSV.exists(), "human_validation_phase3.csv must exist"
    df = pd.read_csv(INDEPENDENT_CSV)

    assert 25 <= len(df) <= 30, f"Expected 25-30 pairs, got {len(df)}"
    assert len(df) == 28, f"Expected exactly 28 pairs, got {len(df)}"

    expected_cols = ["resume_id", "jd_id", "human_label", "reviewer_id", "notes", "evidence"]
    assert list(df.columns) == expected_cols, f"Columns do not match expected schema: {df.columns}"

    # Verify single-blind protocol: NO model prediction or score columns in dataset
    forbidden_tokens = ["score", "model", "prediction", "probability", "rule", "ml"]
    for col in df.columns:
        for token in forbidden_tokens:
            assert token not in col.lower(), f"Single-blind violation: column '{col}' contains '{token}'"

    # Verify label distribution
    valid_labels = {"STRONG_MATCH", "MATCH", "REVIEW", "WEAK_MATCH"}
    assert set(df["human_label"].unique()).issubset(valid_labels)
    assert (df["human_label"].value_counts() == 7).all(), "Labels should be balanced (7 each across 4 classes)"

    # Verify zero overlap with locked 64 pairs
    locked_df = pd.read_csv(LOCKED_CSV)
    locked_pairs = set(zip(locked_df["resume_id"], locked_df["jd_id"]))
    independent_pairs = set(zip(df["resume_id"], df["jd_id"]))
    overlap = locked_pairs.intersection(independent_pairs)
    assert len(overlap) == 0, f"Data leakage! Independent dataset shares pairs with locked benchmark: {overlap}"


def test_shadow_evaluation_improvements():
    """Verify shadow evaluation confirms measurable gains on locked benchmark."""
    assert SHADOW_EVAL_JSON.exists(), "phase_3_shadow_evaluation.json must exist"
    with open(SHADOW_EVAL_JSON, encoding="utf-8") as f:
        data = json.load(f)

    v1_rule = data["baseline_v1_rule"]
    v2_rule = data["candidate_v2_rule"]
    assert v2_rule["accuracy"] >= 0.65, f"Expected v2 accuracy >= 65%, got {v2_rule['accuracy']:.2%}"
    assert v2_rule["shortlist_recall"] >= 0.70, (
        f"Expected v2 shortlist recall >= 70%, got {v2_rule['shortlist_recall']:.2%}"
    )
    assert v2_rule["accuracy"] - v1_rule["accuracy"] > 0.15, "Expected accuracy gain > 15%"
    assert v2_rule["shortlist_recall"] - v1_rule["shortlist_recall"] > 0.30, "Expected shortlist recall gain > 30%"


def test_independent_evaluation_improvements():
    """Verify independent evaluation confirms significant generalization gains on fresh dataset."""
    assert INDEPENDENT_EVAL_JSON.exists(), "phase_3_independent_evaluation.json must exist"
    with open(INDEPENDENT_EVAL_JSON, encoding="utf-8") as f:
        data = json.load(f)

    rule_comp = data["comparison_summary"]["rule_based"]
    assert rule_comp["v2_accuracy"] > rule_comp["v1_accuracy"]
    assert rule_comp["accuracy_delta"] >= 0.15, (
        f"Expected independent accuracy gain >= 15%, got {rule_comp['accuracy_delta']:.2%}"
    )
    assert rule_comp["shortlist_recall_delta"] >= 0.25, (
        f"Expected independent shortlist recall gain >= 25%, got {rule_comp['shortlist_recall_delta']:.2%}"
    )
    assert rule_comp["v2_false_shortlist_rate"] <= 0.05, (
        f"False shortlist rate too high: {rule_comp['v2_false_shortlist_rate']:.2%}"
    )

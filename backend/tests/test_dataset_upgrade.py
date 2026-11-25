"""Tests for Phase 2 Real Resume Dataset & ML Training Upgrade.

Validates:
1. Multi-format real resume ingestion (PDF, DOCX, TXT).
2. Standalone Resume Quality Model (action verbs, quantification, 6 dimensions).
3. Separation of Resume Quality vs Candidate-Job Fit.
4. Dataset builder pair generation (66 pairs, RULE_BASED label source, zero human fabrication).
5. Real ML model training with Grouped Stratified Split (zero candidate data leakage).
6. REST API endpoints (/api/dataset/summary, /api/dataset/pairs, /api/models/metrics?dataset=real, /api/resumes/quality).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from app.config import ML_DIR, PROCESSED_DIR, RESUMES_DIR, ROOT_DATA_DIR
from app.main import app
from app.schemas.candidate import (
    CandidateProfile,
    ContactInfo,
    EducationEntry,
    ExperienceEntry,
    ProjectEntry,
)
from app.services.dataset.dataset_builder import build_evaluation_dataset
from app.services.ingestion.resume_ingester import ingest_single_resume
from app.services.ml.model_trainer import train_and_evaluate_models
from app.services.ml.predictor import get_cached_metrics
from app.services.parsing.resume_parser import parse_resume
from app.services.quality.resume_quality import evaluate_resume_quality
from fastapi.testclient import TestClient

client = TestClient(app)


def test_resume_quality_evaluation_standalone():
    """Verify objective quality evaluation (action verbs, quantification, 6 dimensions)."""
    # Strong profile with action verbs and metrics
    strong_profile = CandidateProfile(
        candidate_id="test_strong",
        name="Test Strong Candidate",
        contact=ContactInfo(email="test@example.com", phone="+91 9876543210"),
        text=(
            "Summary: Results-oriented engineer with 4 years experience.\n"
            "Experience: Architected and deployed microservices that reduced latency by 35%.\n"
            "Spearheaded automated testing pipeline saving $40k annually and cutting build times by 50%.\n"
            "Optimized PostgreSQL database queries improving throughput by 4x for 100000 users.\n"
            "Education: Bachelor of Technology in Computer Science.\n"
            "Skills: Python, FastAPI, Docker, Kubernetes, PostgreSQL."
        ),
        skills=["Python", "FastAPI", "Docker", "Kubernetes", "PostgreSQL"],
        education=[
            EducationEntry(
                degree="BTech",
                degree_level=2,
                field="Computer Science",
                is_relevant_field=True,
                raw="BTech in Computer Science",
            )
        ],
        experience=[
            ExperienceEntry(
                title="Senior Backend Engineer",
                organization="Tech Corp",
                duration_months=36,
                is_current=True,
                is_internship=False,
                bullets=[
                    "Architected and deployed microservices that reduced latency by 35%.",
                    "Spearheaded automated testing pipeline saving $40k annually and cutting build times by 50%.",
                    "Optimized PostgreSQL database queries improving throughput by 4x for 100000 users.",
                ],
            )
        ],
        projects=[
            ProjectEntry(
                name="Cloud Platform",
                description="Engineered distributed system deployed across AWS.",
                technologies=["Python", "Docker"],
            )
        ],
    )

    strong_eval = evaluate_resume_quality(strong_profile)
    assert strong_eval.overall_quality_score >= 75.0
    assert strong_eval.action_verb_score >= 0.8
    assert strong_eval.quantification_score >= 0.8
    assert len(strong_eval.action_verbs_found) >= 3
    assert strong_eval.quantified_bullets_count >= 2
    assert len(strong_eval.dimensions) == 6

    # Sparse profile with poor formatting and no metrics
    sparse_profile = CandidateProfile(
        candidate_id="test_sparse",
        name="Sparse Candidate",
        contact=ContactInfo(),
        text="Skills: Python, Java, SQL, C++, HTML, CSS, JavaScript, React.",
        skills=["Python", "Java", "SQL"],
    )

    sparse_eval = evaluate_resume_quality(sparse_profile)
    assert sparse_eval.overall_quality_score < strong_eval.overall_quality_score
    assert sparse_eval.quantification_score < 0.5
    assert len(sparse_eval.action_verbs_found) == 0


def test_multi_format_ingestion(tmp_path: Path):
    """Verify ingestion pipeline handles PDF, DOCX, and TXT real sample resumes."""
    pdf_path = RESUMES_DIR / "01_rahul_sharma_fullstack.pdf"
    docx_path = RESUMES_DIR / "04_sneha_rao_data_scientist.docx"
    txt_path = RESUMES_DIR / "09_rohan_mehta_mobile_developer.txt"

    assert pdf_path.exists(), "Sample PDF not found"
    assert docx_path.exists(), "Sample DOCX not found"
    assert txt_path.exists(), "Sample TXT not found"

    # Ingest PDF
    res_pdf = ingest_single_resume(pdf_path, "test_pdf_01", output_dir=tmp_path)
    assert res_pdf["candidateName"] == "Rahul Sharma"
    assert len(res_pdf["skills"]) > 5
    assert "qualityFeatures" in res_pdf

    # Ingest DOCX
    res_docx = ingest_single_resume(docx_path, "test_docx_04", output_dir=tmp_path)
    assert res_docx["candidateName"] == "Sneha Rao"
    assert len(res_docx["skills"]) > 5

    # Ingest TXT
    res_txt = ingest_single_resume(txt_path, "test_txt_09", output_dir=tmp_path)
    assert res_txt["candidateName"] == "Rohan Mehta"
    assert len(res_txt["skills"]) > 3


def test_dataset_builder_and_transparency():
    """Verify Resume-JD dataset generation, label distribution, and label source transparency."""
    _, _, summary = build_evaluation_dataset()
    assert summary["total_resumes"] >= 76
    assert summary["total_jds"] == 12
    assert summary["total_pairs"] >= 912

    # Verify no fabricated human labels
    assert summary["label_sources"] == {"RULE_BASED": summary["total_pairs"]}

    pairs_df = pd.read_csv(PROCESSED_DIR / "evaluation_pairs.csv")
    assert len(pairs_df) >= 912
    assert set(pairs_df["label_source"].unique()) == {"RULE_BASED"}

    # Check primary ML features exist and are normalized in [0, 1]
    feature_cols = [
        "required_skill_match",
        "experience_match",
        "semantic_similarity",
        "education_match",
        "project_match",
        "ats_score",
        "resume_quality",
    ]
    for col in feature_cols:
        assert col in pairs_df.columns
        assert pairs_df[col].min() >= 0.0
        assert pairs_df[col].max() <= 1.0


def test_quality_vs_fit_separation():
    """Verify that Resume Quality is distinct from Candidate-Job Fit."""
    pairs_df = pd.read_csv(PROCESSED_DIR / "evaluation_pairs.csv")

    # Rahul Sharma: High quality (~86.2) across all pairs
    rahul_pairs = pairs_df[pairs_df["candidate_name"] == "Rahul Sharma"]
    assert len(rahul_pairs) == 12

    # Quality score is identical and high regardless of target job
    quality_scores = rahul_pairs["resume_quality_score"].unique()
    assert len(quality_scores) == 1
    assert quality_scores[0] > 80.0

    # But Candidate Fit Score varies drastically depending on JD:
    # High fit for Full Stack, low fit for Data Scientist
    fullstack_pair = rahul_pairs[rahul_pairs["job_title"].str.contains("Full Stack")].iloc[0]
    ds_pair = rahul_pairs[rahul_pairs["job_title"].str.contains("Data Scientist")].iloc[0]

    assert fullstack_pair["overall_candidate_fit_score"] > 70.0
    assert fullstack_pair["classification_label"] == "shortlist"

    assert ds_pair["overall_candidate_fit_score"] < 55.0
    assert ds_pair["classification_label"] == "weak_match"


def test_real_ml_training_zero_data_leakage():
    """Train ML models on real dataset and verify zero candidate leakage across train and test."""
    metrics_resp = train_and_evaluate_models(dataset_type="real")
    assert metrics_resp.available is True
    assert len(metrics_resp.models) == 3

    # Check model names
    model_names = {m.name for m in metrics_resp.models}
    assert model_names == {"Logistic Regression", "Random Forest", "KNN"}

    # Best model should achieve strong test performance
    best = next(m for m in metrics_resp.models if m.name == metrics_resp.selected_model)
    assert best.accuracy >= 0.70
    assert best.f1 >= 0.70
    assert best.shortlist_recall >= 0.60

    # Verify split metadata states grouped split
    assert metrics_resp.dataset is not None
    assert "Grouped Stratified Split" in metrics_resp.dataset.split
    assert "zero candidate leakage" in metrics_resp.dataset.split

    # Verify model artifact files exist
    assert (ML_DIR / "real_metrics.json").exists()
    assert (ML_DIR / "real_best_model.txt").exists()


def test_api_dataset_endpoints():
    """Verify all new REST API endpoints for Phase 2."""
    # 1. Dataset summary
    res = client.get("/api/dataset/summary")
    assert res.status_code == 200
    data = res.json()
    assert data["totalResumes"] >= 76
    assert data["totalJobDescriptions"] == 12
    assert data["totalPairs"] >= 912
    assert "RULE_BASED" in data["labelSources"]
    assert len(data["resumes"]) >= 76
    assert len(data["jobDescriptions"]) == 12

    # 2. Dataset pairs (filtered)
    res = client.get("/api/dataset/pairs?limit=5")
    assert res.status_code == 200
    pairs = res.json()
    assert len(pairs) == 5
    first = pairs[0]
    assert "overallCandidateFitScore" in first
    assert "resumeQualityScore" in first
    assert first["labelSource"] == "RULE_BASED"

    # 3. ML metrics with dataset switcher query parameter
    res_real = client.get("/api/models/metrics?dataset=real")
    assert res_real.status_code == 200
    assert "Real" in res_real.json()["dataset"]["source"]

    res_syn = client.get("/api/models/metrics?dataset=synthetic")
    assert res_syn.status_code == 200
    assert "Synthetic demonstration dataset" in res_syn.json()["dataset"]["source"]

    # 4. Standalone resume quality POST endpoint
    sample_pdf = RESUMES_DIR / "01_rahul_sharma_fullstack.pdf"
    with open(sample_pdf, "rb") as f:
        res_q = client.post(
            "/api/resumes/quality",
            files={"file": ("01_rahul_sharma_fullstack.pdf", f, "application/pdf")},
        )
    assert res_q.status_code == 200
    q_data = res_q.json()
    assert q_data["candidateName"] == "Rahul Sharma"
    assert q_data["evaluation"]["overallQualityScore"] > 80.0
    assert len(q_data["evaluation"]["actionVerbsFound"]) > 0
    assert len(q_data["evaluation"]["dimensions"]) == 6

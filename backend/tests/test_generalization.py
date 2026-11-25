"""Comprehensive generalization tests for Phase 2.1:
Unseen Resumes, Unseen JDs, Protected Attribute Filtering, Manifest Integrity, and Canonical Feature Consistency.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import ML_DIR, PROCESSED_DIR, ROOT_DATA_DIR
from app.main import app
from app.schemas.evaluation import CandidateEvaluation, Recommendation
from app.schemas.job import JobProfile, SkillTier
from app.schemas.candidate import CandidateProfile, ContactInfo, EducationEntry, ExperienceEntry, ProjectEntry
from app.services.fairness import audit_feature_names, redact_sensitive_attributes
from app.services.ml.dataset import FEATURE_NAMES
from app.services.parsing.jd_parser import parse_job_description
from app.services.pipeline.canonical_pipeline import evaluate_unseen_resume_and_jd, extract_canonical_features

client = TestClient(app)
RESUMES_DIR = ROOT_DATA_DIR / "resumes"


def test_manifest_verification():
    """Verify manifest.csv contains all 65 uploaded resumes + reference files, with 0 failures."""
    manifest_path = RESUMES_DIR / "manifest.csv"
    assert manifest_path.exists(), "data/resumes/manifest.csv must exist"

    df_manifest = pd.read_csv(manifest_path)
    assert len(df_manifest) >= 65, "Manifest must track at least 65 resumes"

    # Verify the 65 uploaded docx files
    uploaded_rows = df_manifest[df_manifest["source"] == "UPLOADED_REAL_65"]
    assert len(uploaded_rows) == 65, f"Expected 65 uploaded files in manifest, got {len(uploaded_rows)}"

    # Check parse status
    assert (uploaded_rows["parse_status"] == "SUCCESS").all(), "All 65 uploaded resumes must be SUCCESS"

    # Check near duplicates identified
    duplicates = df_manifest[df_manifest["duplicate_status"].str.contains("NEAR_DUPLICATE", na=False)]
    assert len(duplicates) >= 1, "Near duplicate pairs should be flagged in manifest"

    # Check zero failed
    assert len(df_manifest[df_manifest["parse_status"] == "FAILED"]) == 0


def test_protected_attribute_removal():
    """Verify personal/protected attributes are thoroughly redacted from text and excluded from ML."""
    raw_bio = (
        "Name: Johnathan Doe\n"
        "Email: john.doe@example.com\n"
        "Phone: +1 555-234-5678 or 9876543210\n"
        "Current Address: 123 Main Street, Apt 4B, Springfield, IL 62701\n"
        "Date of Birth: 15/08/1992\n"
        "Age: 32 years\n"
        "Gender: Male\n"
        "Marital Status: Married\n"
        "Religion: Hindu\n"
        "Nationality: Indian\n"
        "Caste: General\n"
        "Skills: Python, FastAPI, Docker, PostgreSQL\n"
    )

    sanitized, findings = redact_sensitive_attributes(raw_bio)

    # Assert personal info is purged
    assert "john.doe@example.com" not in sanitized
    assert "555-234-5678" not in sanitized
    assert "123 Main Street" not in sanitized
    assert "15/08/1992" not in sanitized
    assert "Gender: Male" not in sanitized
    assert "Marital Status: Married" not in sanitized
    assert "Religion: Hindu" not in sanitized

    # Assert technical content is retained
    assert "Python" in sanitized
    assert "FastAPI" in sanitized
    assert "Docker" in sanitized
    assert "PostgreSQL" in sanitized

    # Assert findings recorded
    assert len(findings) > 0

    # Assert fairness audit on ML features passes with 0 violations
    violations = audit_feature_names(FEATURE_NAMES)
    assert len(violations) == 0, f"Sensitive attributes found in feature set: {violations}"


def test_feature_consistency_canonical():
    """Verify that canonical feature extraction produces identical values for stored dataset pairs."""
    pairs_file = PROCESSED_DIR / "evaluation_pairs.csv"
    assert pairs_file.exists(), "evaluation_pairs.csv must exist"

    df = pd.read_csv(pairs_file)
    first_row = df.iloc[0]

    resume_id = str(first_row["resume_id"])
    jd_id = str(first_row["jd_id"])

    # Load resume and JD
    from app.services.parsing.resume_parser import parse_resume
    profile = parse_resume(RESUMES_DIR / "1.docx", candidate_id=resume_id)

    clean_jd = jd_id.removeprefix("jd_")
    jd_path = ROOT_DATA_DIR / "job_descriptions" / f"{clean_jd}.txt"
    assert jd_path.exists(), f"JD text {jd_path} not found"
    job = parse_job_description(jd_path.read_text(encoding="utf-8"), title=str(first_row["job_title"]))

    # Extract features via canonical pipeline
    canonical = extract_canonical_features(profile, job)

    # Verify primary features match closely within numerical tolerance
    assert abs(canonical.features.required_skill_match - first_row["required_skill_match"]) < 0.05
    assert abs(canonical.features.education_match - first_row["education_match"]) < 0.05
    assert abs(canonical.overall_score - first_row["overall_candidate_fit_score"]) < 1.0


def test_unseen_resume_generalization():
    """Test evaluation on an UNSEEN candidate resume against an existing JD."""
    unseen_resume_text = (
        "Alice Smith\n"
        "Cloud Architect\n\n"
        "SUMMARY\n"
        "8 years building large scale distributed cloud infrastructure.\n\n"
        "SKILLS\n"
        "Python, Go, Kubernetes, Terraform, AWS, Docker, CI/CD, Prometheus, Grafana, Linux.\n\n"
        "PROFESSIONAL EXPERIENCE\n"
        "Cloud Lead | FinTech Solutions\n"
        "2018 - 2024 (6 years)\n"
        "- Architected and deployed hybrid cloud using Terraform and Kubernetes on AWS.\n"
        "- Led migration of 50 microservices to AWS EKS cutting cloud cost by 40%.\n\n"
        "PROJECTS\n"
        "Hybrid Cloud Orchestrator: Built Kubernetes multi-cluster controller with Go.\n\n"
        "EDUCATION\n"
        "Bachelor of Technology in Computer Science (2014 - 2018)\n"
    )

    jd_path = ROOT_DATA_DIR / "job_descriptions" / "jd_004_devops_cloud.txt"
    job = parse_job_description(jd_path.read_text(encoding="utf-8"), title="DevOps Platform Engineer")

    eval_result = evaluate_unseen_resume_and_jd(
        resume=unseen_resume_text,
        job=job,
        candidate_id="unseen_alice_smith",
    )

    assert isinstance(eval_result, CandidateEvaluation)
    assert eval_result.overall_score > 65.0
    assert eval_result.recommendation in {Recommendation.SHORTLIST, Recommendation.STRONG_SHORTLIST}
    assert "Kubernetes" in eval_result.matched_skills or "Terraform" in eval_result.matched_skills or "AWS" in eval_result.matched_skills
    assert eval_result.ml_prediction is not None
    assert eval_result.ml_prediction.available is True
    assert len(eval_result.strengths) > 0


def test_unseen_jd_generalization():
    """Test evaluation on an existing candidate against a completely UNSEEN Job Description."""
    novel_jd_text = (
        "Role: Embedded Rust Autonomous Robotics Engineer\n"
        "Experience: 4+ years\n"
        "Requirements:\n"
        "- Rust, C++, Embedded Systems, ROS2, Linux Kernel, Microcontrollers (Required)\n"
        "- CAN bus, ARM Cortex, Real-Time Operating Systems (RTOS) (Preferred)\n"
        "Responsibilities: Develop deterministic real-time firmware for autonomous mobile robots."
    )

    # Evaluate Rahul Sharma (Full Stack Web Developer) against novel Robotics JD
    sample_resume = RESUMES_DIR / "01_rahul_sharma_fullstack.pdf"
    eval_result = evaluate_unseen_resume_and_jd(
        resume=sample_resume,
        job=novel_jd_text,
        job_title="Robotics Engineer",
    )

    assert isinstance(eval_result, CandidateEvaluation)
    # Since Rahul Sharma does not have Rust/ROS2/RTOS, fit score should be low
    assert eval_result.overall_score < 50.0
    assert eval_result.recommendation == Recommendation.WEAK_MATCH
    assert "C++" in eval_result.missing_skills or len(eval_result.missing_skills) > 0
    assert len(eval_result.weaknesses) > 0


def test_unseen_resume_and_unseen_jd():
    """Test evaluation on BOTH a completely unseen resume AND a completely unseen Job Description."""
    unseen_resume = (
        "Elena Rostova\n"
        "Cybersecurity Incident Response Lead\n\n"
        "SUMMARY\n"
        "5 years leading enterprise threat intelligence and SOC incident response.\n\n"
        "SKILLS\n"
        "SIEM, Splunk, Wireshark, Threat Hunting, Incident Response, Python, Linux.\n\n"
        "PROFESSIONAL EXPERIENCE\n"
        "Senior SOC Analyst | CyberShield Corp\n"
        "2019 - Present (5 years)\n"
        "- Led high-severity incident response investigations utilizing Splunk and Wireshark.\n"
        "- Designed automated threat hunting playbooks with Python and SIEM correlation rules.\n\n"
        "PROJECTS\n"
        "Enterprise SOC Automation: Built Python pipelines for SIEM alert triage.\n\n"
        "EDUCATION\n"
        "Master of Science in Information Security (2017 - 2019)\n"
    )

    unseen_jd = (
        "Job Title: Senior Threat Intelligence & Incident Response Specialist\n"
        "Experience: 4+ years\n"
        "Requirements:\n"
        "- Splunk, Threat Hunting, Incident Response, SIEM (Required)\n"
        "- Python, Wireshark, Linux (Preferred)\n"
        "Responsibilities: Lead enterprise incident response and proactive threat hunts."
    )

    eval_result = evaluate_unseen_resume_and_jd(
        resume=unseen_resume,
        job=unseen_jd,
        candidate_id="unseen_elena",
        job_title="Threat Intel Specialist",
    )

    assert isinstance(eval_result, CandidateEvaluation)
    assert eval_result.overall_score >= 65.0
    assert eval_result.recommendation in {Recommendation.SHORTLIST, Recommendation.STRONG_SHORTLIST}
    assert eval_result.ml_prediction.available is True
    assert eval_result.resume_quality > 50.0


def test_production_api_screen_endpoint():
    """Verify the production REST endpoint POST /api/screen handles unseen multipart and form inputs."""
    # Test via form data (raw text resume + raw text JD)
    res = client.post(
        "/api/screen",
        data={
            "resume_text": "DevOps pro with Docker, Kubernetes, CI/CD, Python. 4 years exp. BTech CS.",
            "job_description": "Wanted: DevOps engineer with Kubernetes, Docker, CI/CD. 3+ years experience.",
            "job_title": "DevOps Engineer",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert "overallScore" in data
    assert "resumeQuality" in data or "qualityEvaluation" in data
    assert "matchedSkills" in data
    assert "mlPrediction" in data
    assert data["mlPrediction"]["available"] is True
    assert len(data["strengths"]) > 0

    # Test rejection when neither resume file nor text is provided
    bad_res = client.post("/api/screen", data={"job_description": "Some job"})
    assert bad_res.status_code == 400

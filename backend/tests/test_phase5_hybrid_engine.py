"""Phase 5 Tests: Production Hardening, Hybrid Decision Engine, Fairness, and Explainability."""

import pytest
from app.schemas.candidate import CandidateProfile
from app.schemas.evaluation import (
    FeatureVector,
    HybridDecision,
    MLPrediction,
    Recommendation,
)
from app.schemas.job import JobProfile, JobSkill, SkillTier
from app.services.evaluation.hybrid_engine import (
    BORDERLINE_LOWER,
    BORDERLINE_UPPER,
    SHORTLIST_THRESHOLD,
    resolve_hybrid_decision,
)
from app.services.fairness import PROTECTED_ATTRIBUTES, audit_feature_names, redact_sensitive_attributes
from app.services.ml.predictor import FEATURE_NAMES
from app.services.pipeline.canonical_pipeline import evaluate_unseen_resume_and_jd
from app.services.quality.resume_quality import evaluate_resume_quality


# ==============================================================================
# 1. Hybrid Decision Engine Policy Tests
# ==============================================================================

def test_hybrid_consensus_shortlist():
    """When Rule and ML both recommend shortlist, consensus is reached."""
    ml = MLPrediction(
        available=True,
        model_name="LogisticRegression",
        predicted_label="shortlist",
        probabilities={"shortlist": 0.92, "review": 0.06, "weak_match": 0.02},
    )
    res = resolve_hybrid_decision(80.0, Recommendation.SHORTLIST, ml)
    assert res.final_decision == "SHORTLIST"
    assert res.agreement is True
    assert res.is_borderline is False
    assert res.review_recommendation_flag is None
    assert "Consensus" in res.policy_reason


def test_hybrid_consensus_review():
    """When Rule and ML both recommend review, consensus is reached."""
    ml = MLPrediction(
        available=True,
        model_name="LogisticRegression",
        predicted_label="review",
        probabilities={"shortlist": 0.15, "review": 0.75, "weak_match": 0.10},
    )
    res = resolve_hybrid_decision(55.0, Recommendation.REVIEW, ml)
    assert res.final_decision == "REVIEW"
    assert res.agreement is True
    assert res.is_borderline is False
    assert res.review_recommendation_flag is None
    assert "Consensus" in res.policy_reason


def test_hybrid_consensus_weak_match():
    """When Rule and ML both recommend weak match, consensus is reached."""
    ml = MLPrediction(
        available=True,
        model_name="LogisticRegression",
        predicted_label="weak_match",
        probabilities={"shortlist": 0.02, "review": 0.08, "weak_match": 0.90},
    )
    res = resolve_hybrid_decision(30.0, Recommendation.WEAK_MATCH, ml)
    assert res.final_decision == "WEAK_MATCH"
    assert res.agreement is True
    assert res.is_borderline is False
    assert res.review_recommendation_flag is None
    assert "Consensus" in res.policy_reason


def test_borderline_review_with_strong_ml_opinion():
    """Rule REVIEW in [60.0, 61.9] with high-confidence ML Shortlist (>=0.70).

    CRITICAL INVARIANT: The system does NOT silently override Rule to SHORTLIST.
    It preserves REVIEW, marks is_borderline=True, and raises BORDERLINE_ML_SECOND_OPINION.
    """
    ml = MLPrediction(
        available=True,
        model_name="LogisticRegression",
        predicted_label="shortlist",
        probabilities={"shortlist": 0.88, "review": 0.10, "weak_match": 0.02},
    )
    res = resolve_hybrid_decision(61.2, Recommendation.REVIEW, ml)
    assert res.final_decision == "REVIEW"  # No silent promotion
    assert res.rule_decision == "REVIEW"
    assert res.ml_decision == "shortlist"
    assert res.agreement is False
    assert res.is_borderline is True
    assert res.review_recommendation_flag == "BORDERLINE_ML_SECOND_OPINION"
    assert "Borderline Review" in res.policy_reason


def test_borderline_shortlist_with_ml_caution():
    """Rule SHORTLIST in [62.0, 63.0] with ML Review or Weak.

    CRITICAL INVARIANT: Primary rule authority is preserved, but candidate receives
    BORDERLINE_ML_CAUTION so recruiter audits the boundary case.
    """
    ml = MLPrediction(
        available=True,
        model_name="LogisticRegression",
        predicted_label="review",
        probabilities={"shortlist": 0.35, "review": 0.60, "weak_match": 0.05},
    )
    res = resolve_hybrid_decision(62.4, Recommendation.SHORTLIST, ml)
    assert res.final_decision == "SHORTLIST"  # Rule authority preserved
    assert res.rule_decision == "SHORTLIST"
    assert res.ml_decision == "review"
    assert res.agreement is False
    assert res.is_borderline is True
    assert res.review_recommendation_flag == "BORDERLINE_ML_CAUTION"
    assert "Borderline Shortlist" in res.policy_reason


def test_safety_invariant_weak_match_never_promoted():
    """A WEAK_MATCH candidate must NEVER be promoted to SHORTLIST by ML alone under any circumstance."""
    ml = MLPrediction(
        available=True,
        model_name="LogisticRegression",
        predicted_label="shortlist",
        probabilities={"shortlist": 0.99, "review": 0.01, "weak_match": 0.00},
    )
    res = resolve_hybrid_decision(38.0, Recommendation.WEAK_MATCH, ml)
    assert res.final_decision == "WEAK_MATCH"  # Overridden by Rule safety floor
    assert res.agreement is False
    assert res.review_recommendation_flag == "DIVERGENCE_REJECTED"
    assert "Safety Invariant" in res.policy_reason


def test_general_disagreement_preserves_rule_authority():
    """Outside borderline window, Rule engine takes precedence and divergence is flagged."""
    ml = MLPrediction(
        available=True,
        model_name="LogisticRegression",
        predicted_label="review",
        probabilities={"shortlist": 0.20, "review": 0.70, "weak_match": 0.10},
    )
    res = resolve_hybrid_decision(85.0, Recommendation.STRONG_SHORTLIST, ml)
    assert res.final_decision == "STRONG_SHORTLIST"
    assert res.agreement is False
    assert res.is_borderline is False
    assert res.review_recommendation_flag == "ML_DISAGREEMENT"


def test_ml_unavailable_deterministic_fallback():
    """If ML model is unavailable, decision deterministically falls back to Rule score."""
    res = resolve_hybrid_decision(75.0, Recommendation.SHORTLIST, None)
    assert res.final_decision == "SHORTLIST"
    assert res.ml_decision == "unavailable"
    assert res.agreement is True
    assert res.is_borderline is False


# ==============================================================================
# 2. Strict Separation of Scores (Fit vs Quality)
# ==============================================================================

def test_score_separation_resume_quality_vs_candidate_fit():
    """Verify standalone Resume Quality is distinct from Candidate-JD Fit."""
    profile = CandidateProfile(
        candidate_id="test_sep_01",
        name="Alex River",
        text=(
            "EXPERIENCE\n"
            "Software Engineer at Acme Corp (2020 - Present)\n"
            "- Spearheaded migration of 15 microservices to AWS, reducing latency by 35%.\n"
            "- Implemented CI/CD pipelines increasing deployment frequency by 40%.\n\n"
            "EDUCATION\n"
            "B.S. in Computer Science, State University\n\n"
            "SKILLS\n"
            "Python, Docker, Kubernetes, AWS\n"
        ),
        skills=["Python", "Docker", "Kubernetes", "AWS"],
        total_experience_months=48,
    )

    # 1. Standalone Quality Evaluation (Independent of any JD)
    quality_eval = evaluate_resume_quality(profile)
    assert 0.0 <= quality_eval.overall_quality_score <= 100.0
    assert quality_eval.quantification_score > 0.0

    # 2. Candidate-JD Fit Evaluation (Relative to JD)
    # JD requiring skills NOT in the resume (Data Science / NLP)
    mismatch_jd = JobProfile(
        job_id="job_ds",
        title="Senior NLP Scientist",
        skills=[
            JobSkill(name="PyTorch", tier=SkillTier.REQUIRED),
            JobSkill(name="Transformers", tier=SkillTier.REQUIRED),
            JobSkill(name="Large Language Models", tier=SkillTier.REQUIRED),
        ],
        min_experience_years=5,
    )

    eval_result = evaluate_unseen_resume_and_jd(profile, mismatch_jd)

    # Candidate has moderate-to-good resume quality (~60%),
    # but low JD fit score (<30%) because technical skills do not match!
    assert eval_result.quality_evaluation is not None
    assert eval_result.quality_evaluation.overall_quality_score >= 50.0
    assert eval_result.overall_score < 30.0  # Mismatched fit
    assert eval_result.recommendation == Recommendation.WEAK_MATCH

    # Verify both scores are cleanly exposed as separate fields
    assert eval_result.overall_score != eval_result.quality_evaluation.overall_quality_score
    assert eval_result.hybrid_decision is not None
    assert eval_result.hybrid_decision.final_decision == "WEAK_MATCH"


# ==============================================================================
# 3. Protected Attribute Safety Audit
# ==============================================================================

def test_protected_attribute_fairness_audit():
    """Scoring feature vector must contain 0 protected or demographic attributes."""
    violations = audit_feature_names(FEATURE_NAMES)
    assert violations == [], f"Protected attributes found in scoring features: {violations}"

    # Also test redaction of sensitive lines
    sample_text = (
        "John Doe\n"
        "Date of Birth: 15/08/1992\n"
        "Gender: Male\n"
        "Marital status: Married\n"
        "Religion: None\n"
        "Email: john.doe@example.com\n"
        "Phone: +1-555-123-4567\n"
        "Skills: Python, React, SQL\n"
    )
    redacted_text, findings = redact_sensitive_attributes(sample_text)
    assert len(findings) >= 5
    assert "1992" not in redacted_text
    assert "john.doe@example.com" not in redacted_text
    assert "+1-555-123-4567" not in redacted_text
    assert "Python, React, SQL" in redacted_text  # Job-relevant skills preserved


# ==============================================================================
# 4. Canonical Pipeline End-to-End Integration
# ==============================================================================

def test_canonical_pipeline_includes_hybrid_decision():
    """Unseen evaluation returns a complete HybridDecision and feature contributions."""
    resume_text = (
        "Jane Doe\n"
        "Senior Backend Engineer with 5 years experience in Python, FastAPI, PostgreSQL, Docker.\n"
        "Built distributed REST APIs serving 10M daily requests.\n"
        "BS in Computer Science."
    )
    jd_text = (
        "Backend Developer\n"
        "Requirements:\n"
        "- 3+ years experience\n"
        "- Python, FastAPI, PostgreSQL\n"
        "- Docker and API development\n"
    )

    evaluation = evaluate_unseen_resume_and_jd(resume_text, jd_text)
    assert evaluation.status == "COMPLETED"
    assert evaluation.hybrid_decision is not None
    assert isinstance(evaluation.hybrid_decision, HybridDecision)
    assert evaluation.hybrid_decision.rule_score == evaluation.overall_score
    assert len(evaluation.hybrid_decision.feature_contributions) > 0
    assert "required_skill_match" in evaluation.hybrid_decision.feature_contributions

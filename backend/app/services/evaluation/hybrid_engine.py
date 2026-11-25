"""Hybrid Decision Engine for Candidate Evaluation.

Combines transparent Rule v2 candidate scoring with calibrated ML v2 predictions.

Deterministic Policy:
1. Rule v2 is the PRIMARY authority:
   - Evaluated from canonical features: Required Skill Match (30%), Experience Relevance (20%),
     Semantic Similarity (15%), Project Relevance (10%), ATS Compatibility (10%),
     Education Match (10%), Resume Quality (5%).
2. Calibrated ML v2 acts as an auditable SECOND OPINION:
   - Provides calibrated class probabilities, directional log-odds feature contributions,
     and divergence detection.
   - Never silently overrides Rule v2.
3. Borderline Logic [60.0, 63.0] (around the 62.0 shortlist boundary):
   - 60.0 to 61.9 (Rule REVIEW): If ML predicts SHORTLIST with >= 0.70 calibrated confidence,
     candidate remains REVIEW, but receives 'BORDERLINE_ML_SECOND_OPINION' priority review flag.
   - 62.0 to 63.0 (Rule SHORTLIST): If ML predicts REVIEW,
     candidate remains SHORTLIST, but receives 'BORDERLINE_ML_CAUTION' flag.
4. Strict Safety Invariants:
   - A candidate evaluated as WEAK_MATCH (<45.0) can NEVER be promoted to SHORTLIST by ML.
"""

from __future__ import annotations

import logging
from app.schemas.evaluation import HybridDecision, MLPrediction, Recommendation

logger = logging.getLogger(__name__)

BORDERLINE_LOWER = 60.0
BORDERLINE_UPPER = 63.0
SHORTLIST_THRESHOLD = 62.0
STRONG_ML_CONFIDENCE = 0.70


def resolve_hybrid_decision(
    rule_score: float,
    rule_recommendation: Recommendation,
    ml_prediction: MLPrediction | None,
) -> HybridDecision:
    """Deterministically resolve rule evaluation and ML second-opinion into a HybridDecision."""
    rule_decision = rule_recommendation.value

    # Map rule recommendation to 3-class target space
    if rule_recommendation in {Recommendation.STRONG_SHORTLIST, Recommendation.SHORTLIST}:
        rule_mapped = "shortlist"
    elif rule_recommendation == Recommendation.REVIEW:
        rule_mapped = "review"
    else:
        rule_mapped = "weak_match"

    is_borderline = (BORDERLINE_LOWER <= rule_score <= BORDERLINE_UPPER)

    # 1. Fallback if ML is unavailable
    if ml_prediction is None or not ml_prediction.available:
        return HybridDecision(
            rule_score=round(rule_score, 1),
            rule_decision=rule_decision,
            ml_decision="unavailable",
            ml_probability=0.0,
            final_decision=rule_decision,
            agreement=True,
            is_borderline=is_borderline,
            policy_reason="ML prediction unavailable; final decision determined solely by deterministic Rule v2 score.",
            review_recommendation_flag=None,
            feature_contributions={},
        )

    ml_decision = ml_prediction.predicted_label
    ml_prob = ml_prediction.probabilities.get(ml_decision, 0.0)
    contributions = ml_prediction.feature_contributions or {}
    agreement = (rule_mapped == ml_decision)

    # 2. Case: Consensus (Rule & ML agree)
    if agreement:
        return HybridDecision(
            rule_score=round(rule_score, 1),
            rule_decision=rule_decision,
            ml_decision=ml_decision,
            ml_probability=round(ml_prob, 4),
            final_decision=rule_decision,
            agreement=True,
            is_borderline=is_borderline,
            policy_reason=(
                f"Consensus: Rule engine ({rule_score:.1f}/100 -> {rule_decision}) and "
                f"calibrated ML v2 model ({ml_decision} with {ml_prob*100:.1f}% confidence) fully agree."
            ),
            review_recommendation_flag=None,
            feature_contributions=contributions,
        )

    # 3. Case: Safety Invariant — Strict Weak Match Protection
    if rule_recommendation == Recommendation.WEAK_MATCH and ml_decision == "shortlist":
        return HybridDecision(
            rule_score=round(rule_score, 1),
            rule_decision=rule_decision,
            ml_decision=ml_decision,
            ml_probability=round(ml_prob, 4),
            final_decision=rule_decision,
            agreement=False,
            is_borderline=is_borderline,
            policy_reason=(
                f"Safety Invariant: Candidate overall score ({rule_score:.1f}/100) is in Weak Match tier. "
                f"ML shortlist prediction ({ml_prob*100:.1f}%) is overridden by Rule safety floor to eliminate false shortlists."
            ),
            review_recommendation_flag="DIVERGENCE_REJECTED",
            feature_contributions=contributions,
        )

    # 4. Case: Borderline Window [60.0, 63.0]
    if is_borderline:
        # Subcase 4A: Borderline Review with strong ML support (60.0 <= score < 62.0)
        if rule_score < SHORTLIST_THRESHOLD and ml_decision == "shortlist" and ml_prob >= STRONG_ML_CONFIDENCE:
            return HybridDecision(
                rule_score=round(rule_score, 1),
                rule_decision=rule_decision,
                ml_decision=ml_decision,
                ml_probability=round(ml_prob, 4),
                final_decision=rule_decision,  # Stays REVIEW, not auto-promoted!
                agreement=False,
                is_borderline=True,
                policy_reason=(
                    f"Borderline Review ({rule_score:.1f}/100): Candidate is just below shortlist threshold ({SHORTLIST_THRESHOLD}), "
                    f"but calibrated ML v2 predicts Shortlist ({ml_prob*100:.1f}% confidence). "
                    f"Candidate flagged with priority second-opinion for recruiter audit."
                ),
                review_recommendation_flag="BORDERLINE_ML_SECOND_OPINION",
                feature_contributions=contributions,
            )

        # Subcase 4B: Borderline Shortlist with ML caution (62.0 <= score <= 63.0)
        if rule_score >= SHORTLIST_THRESHOLD and ml_decision in {"review", "weak_match"}:
            return HybridDecision(
                rule_score=round(rule_score, 1),
                rule_decision=rule_decision,
                ml_decision=ml_decision,
                ml_probability=round(ml_prob, 4),
                final_decision=rule_decision,  # Stays SHORTLIST, but with caution flag
                agreement=False,
                is_borderline=True,
                policy_reason=(
                    f"Borderline Shortlist ({rule_score:.1f}/100): Candidate narrowly meets shortlist threshold ({SHORTLIST_THRESHOLD}), "
                    f"but ML v2 indicates hesitation ({ml_decision} with {ml_prob*100:.1f}% confidence). "
                    f"Shortlist retained with advisory caution flag for recruiter audit."
                ),
                review_recommendation_flag="BORDERLINE_ML_CAUTION",
                feature_contributions=contributions,
            )

        # Subcase 4C: Other borderline divergence
        return HybridDecision(
            rule_score=round(rule_score, 1),
            rule_decision=rule_decision,
            ml_decision=ml_decision,
            ml_probability=round(ml_prob, 4),
            final_decision=rule_decision,
            agreement=False,
            is_borderline=True,
            policy_reason=(
                f"Borderline Disagreement ({rule_score:.1f}/100): Rule evaluates to {rule_decision} "
                f"while ML v2 predicts {ml_decision} ({ml_prob*100:.1f}% confidence). "
                f"Rule authority preserved; flagged for recruiter review."
            ),
            review_recommendation_flag="ML_DISAGREEMENT",
            feature_contributions=contributions,
        )

    # 5. Case: General Disagreement outside borderline zone
    return HybridDecision(
        rule_score=round(rule_score, 1),
        rule_decision=rule_decision,
        ml_decision=ml_decision,
        ml_probability=round(ml_prob, 4),
        final_decision=rule_decision,
        agreement=False,
        is_borderline=False,
        policy_reason=(
            f"Rule authority retained: Rule engine ({rule_score:.1f}/100 -> {rule_decision}) "
            f"takes precedence over ML v2 second opinion ({ml_decision} with {ml_prob*100:.1f}% confidence). "
            f"Flagged for human review."
        ),
        review_recommendation_flag="ML_DISAGREEMENT",
        feature_contributions=contributions,
    )

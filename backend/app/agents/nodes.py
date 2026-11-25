"""Agent node implementations for the LangGraph workflow."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from app.agents.state import CandidateAgentState
from app.agents.tools import tool_predict_ml_fit
from app.db.database import recall_candidate_memory
from app.schemas.evaluation import (
    AgentTraceEvent,
    EvidencePassage,
    EvidenceStatus,
    Recommendation,
    RequirementMatch,
    ReviewFlag,
)
from app.schemas.job import SkillTier
from app.services.ats.ats_engine import detect_keyword_stuffing, evaluate_ats
from app.services.evaluation.hybrid_engine import resolve_hybrid_decision
from app.services.features.feature_engine import compute_features
from app.services.explanation.explainer import generate_candidate_explanation
from app.services.ml.predictor import predict_candidate_fit
from app.services.parsing.resume_parser import parse_resume
from app.services.semantic.matcher import match_requirements
from app.services.skills.extractor import count_skill_mentions
from app.services.skills.taxonomy import implying_skills

logger = logging.getLogger(__name__)


def resume_parsing_node(state: CandidateAgentState) -> dict[str, Any]:
    t0 = time.time()
    file_p = Path(state.file_path)
    step = len(state.trace) + 1

    try:
        profile = parse_resume(file_p, candidate_id=state.candidate_id, strategy=state.parsing_strategy)
        duration = round((time.time() - t0) * 1000, 1)

        trace = AgentTraceEvent(
            step=step,
            agent="Resume Parsing Agent",
            task=f"Parse resume document '{file_p.name}' using {profile.diagnostics.parser_used if profile.diagnostics else 'primary parser'}",
            tool="parse_resume",
            tool_input=str(file_p),
            observation=(
                f"Successfully parsed candidate profile for '{profile.name}'. "
                f"Extracted {len(profile.skills)} skills, {len(profile.experience)} experience entries, "
                f"{len(profile.education)} education entries, {len(profile.projects)} projects."
            ),
            status="success",
            duration_ms=duration,
            details={
                "candidateName": profile.name,
                "skillsExtracted": len(profile.skills),
                "experienceMonths": profile.total_experience_months,
                "education": [e.degree for e in profile.education],
                "charCount": profile.diagnostics.char_count if profile.diagnostics else 0,
            },
        )
        return {"profile": profile, "parsing_failed": False, "trace": [trace]}
    except Exception as e:
        duration = round((time.time() - t0) * 1000, 1)
        trace = AgentTraceEvent(
            step=step,
            agent="Resume Parsing Agent",
            task=f"Parse resume document '{file_p.name}'",
            tool="parse_resume",
            tool_input=str(file_p),
            observation=f"Primary parsing failed: {e}. Handing off to Error Recovery Agent.",
            status="error",
            duration_ms=duration,
            details={"error": str(e)},
        )
        return {
            "parsing_failed": True,
            "parsing_error": str(e),
            "trace": [trace],
        }


def error_recovery_node(state: CandidateAgentState) -> dict[str, Any]:
    t0 = time.time()
    file_p = Path(state.file_path)
    step = len(state.trace) + 1

    logger.warning("Error Recovery Agent activated for candidate %s (%s)", state.candidate_id, file_p.name)

    try:
        # Attempt alternate block reconstruction strategy
        profile = parse_resume(file_p, candidate_id=state.candidate_id, strategy="alternate")
        duration = round((time.time() - t0) * 1000, 1)

        trace = AgentTraceEvent(
            step=step,
            agent="Error Recovery Agent",
            task="Recover unparsed document via alternate text-block reconstruction",
            tool="parse_resume[alternate]",
            tool_input=str(file_p),
            observation=(
                f"Error recovery succeeded! Reconstructed candidate profile for '{profile.name}' "
                f"({len(profile.skills)} skills recovered)."
            ),
            status="success",
            duration_ms=duration,
            details={"strategy": "alternate_block_reconstruction", "candidateName": profile.name},
        )
        return {"profile": profile, "parsing_failed": False, "parsing_error": None, "trace": [trace]}
    except Exception as e:
        duration = round((time.time() - t0) * 1000, 1)
        trace = AgentTraceEvent(
            step=step,
            agent="Error Recovery Agent",
            task="Recover unparsed document via alternate text-block reconstruction",
            tool="parse_resume[alternate]",
            tool_input=str(file_p),
            observation=f"Error recovery exhausted: {e}. Candidate marked as UNREADABLE.",
            status="error",
            duration_ms=duration,
            details={"error": str(e)},
        )
        return {
            "parsing_failed": True,
            "parsing_error": f"Document could not be parsed: {e}",
            "trace": [trace],
        }


def ats_evaluation_node(state: CandidateAgentState) -> dict[str, Any]:
    t0 = time.time()
    profile = state.profile
    assert profile is not None
    step = len(state.trace) + 1

    ats_res = evaluate_ats(profile, state.job)
    stuffing_res = detect_keyword_stuffing(profile, state.job)
    duration = round((time.time() - t0) * 1000, 1)

    obs = (
        f"ATS Score: {ats_res.score}/100. Contact: {ats_res.contact_score}%, "
        f"Sections: {ats_res.section_score}%, Formatting: {ats_res.formatting_score}%, "
        f"Keyword Coverage: {ats_res.keyword_coverage}% ({len(ats_res.matched_keywords)}/{len(state.job.skills)} terms). "
    )
    if stuffing_res.suspected:
        obs += f"WARNING: Keyword stuffing suspected ({len(stuffing_res.flagged_terms)} flagged terms)."

    trace = AgentTraceEvent(
        step=step,
        agent="ATS Evaluation Agent",
        task="Evaluate traditional ATS compatibility, contact info, formatting, and keyword stuffing",
        tool="evaluate_ats",
        tool_input=f"Candidate: {profile.name}, Job: {state.job.title}",
        observation=obs,
        status="warning" if stuffing_res.suspected else "success",
        duration_ms=duration,
        details={
            "atsScore": ats_res.score,
            "matchedKeywords": ats_res.matched_keywords,
            "missingKeywords": ats_res.missing_keywords,
            "keywordStuffingSuspected": stuffing_res.suspected,
            "flaggedTerms": [t.term for t in stuffing_res.flagged_terms],
        },
    )

    return {
        "ats": ats_res,
        "stuffing": stuffing_res,
        "keyword_ats_score": ats_res.keyword_coverage,
        "trace": [trace],
    }


def semantic_matching_node(state: CandidateAgentState) -> dict[str, Any]:
    t0 = time.time()
    profile = state.profile
    assert profile is not None
    step = len(state.trace) + 1

    matches, mean_sim = match_requirements(profile, state.job)
    duration = round((time.time() - t0) * 1000, 1)

    uncertain_count = sum(1 for m in matches if m.status.value in {"UNCERTAIN", "NOT_FOUND"})
    needs_refl = uncertain_count > 0

    semantic_matched = [m.requirement for m in matches if m.status.value == "SEMANTIC_MATCH"]
    explicit_matched = [m.requirement for m in matches if m.status.value == "MATCH"]

    obs = (
        f"Vector search matched {len(explicit_matched)} explicit and {len(semantic_matched)} semantic requirements "
        f"(mean similarity: {mean_sim*100:.1f}%). "
    )
    if semantic_matched:
        obs += f"Semantic matches detected without exact keywords: {', '.join(semantic_matched)}. "
    if needs_refl:
        obs += f"{uncertain_count} requirement(s) uncertain or missing; triggering Reflection Agent."

    trace = AgentTraceEvent(
        step=step,
        agent="Semantic Matching Agent",
        task="Chunk resume, build FAISS vector index, and match JD requirements semantically",
        tool="vector_search",
        tool_input=f"{len(state.job.skills)} JD requirements against FAISS index",
        observation=obs,
        status="success",
        duration_ms=duration,
        details={
            "meanSimilarity": round(mean_sim, 3),
            "explicitMatches": explicit_matched,
            "semanticMatches": semantic_matched,
            "uncertainCount": uncertain_count,
        },
    )

    return {
        "matches": matches,
        "mean_semantic_sim": mean_sim,
        "needs_reflection": needs_refl,
        "trace": [trace],
    }


def reflection_node(state: CandidateAgentState) -> dict[str, Any]:
    """Reflection Agent: re-evaluates uncertain/missing skills for ecosystem implication."""
    t0 = time.time()
    profile = state.profile
    assert profile is not None
    step = len(state.trace) + 1

    updated_matches: list[RequirementMatch] = []
    reflected_count = 0
    reflections: list[str] = []

    for m in state.matches:
        if m.status.value in {"UNCERTAIN", "NOT_FOUND"}:
            # Check if any implied skill is present in the resume
            impliers = implying_skills(m.requirement)
            detected_impliers = [s for s in impliers if count_skill_mentions(s, profile.text) > 0]

            if detected_impliers:
                # E.g. Django is present, implying Python!
                # Mark as INFERRED (never automatic MATCH; requires human verification)
                reflected_count += 1
                refl_note = (
                    f"Reflection verified: '{detected_impliers[0]}' experience implies knowledge of {m.requirement}. "
                    f"Status updated from {m.status.value} to INFERRED (requires recruiter verification)."
                )
                reflections.append(f"{m.requirement} implied by {', '.join(detected_impliers)}")

                # Copy with updated status
                m_updated = m.model_copy(
                    update={
                        "status": EvidenceStatus.INFERRED,
                        "reflected": True,
                        "credit": 0.5,
                        "note": refl_note,
                    }
                )
                updated_matches.append(m_updated)
            else:
                updated_matches.append(m)
        else:
            updated_matches.append(m)

    duration = round((time.time() - t0) * 1000, 1)
    obs = (
        f"Reflection completed. Re-examined {len(state.matches)} requirements: {reflected_count} requirement(s) "
        f"inferred via related ecosystem skills ({', '.join(reflections) if reflections else 'no changes'})."
    )

    trace = AgentTraceEvent(
        step=step,
        agent="Reflection Agent",
        task="Re-evaluate uncertain requirements against skill dependency graph and evidence passages",
        tool="skill_dependency_graph",
        tool_input=f"{len(state.matches)} candidate requirement matches",
        observation=obs,
        status="success",
        duration_ms=duration,
        details={"reflectedCount": reflected_count, "reflections": reflections},
    )

    return {"matches": updated_matches, "trace": [trace]}


def feature_engineering_node(state: CandidateAgentState) -> dict[str, Any]:
    t0 = time.time()
    profile = state.profile
    ats = state.ats
    assert profile is not None and ats is not None
    step = len(state.trace) + 1

    features, breakdown, overall_score, recommendation, review_flags = compute_features(
        profile=profile,
        job=state.job,
        matches=state.matches,
        ats=ats,
        mean_semantic_sim=state.mean_semantic_sim,
    )
    duration = round((time.time() - t0) * 1000, 1)

    trace = AgentTraceEvent(
        step=step,
        agent="Feature Engineering Agent",
        task="Compute normalized measurable candidate features and apply transparent scoring formula",
        tool="compute_features",
        tool_input="Evaluated requirements, ATS metrics, experience and project vectors",
        observation=(
            f"Overall Candidate Fit Score: {overall_score}/100. "
            f"Req Skill: {features.required_skill_match*100:.1f}%, Exp: {features.experience_match*100:.1f}%, "
            f"Semantic: {features.semantic_similarity*100:.1f}%, ATS: {features.ats_score*100:.1f}%."
        ),
        status="success",
        duration_ms=duration,
        details={
            "overallScore": overall_score,
            "features": features.model_dump(),
            "recommendation": recommendation.value,
        },
    )

    return {
        "features": features,
        "score_breakdown": breakdown,
        "overall_score": overall_score,
        "recommendation": recommendation,
        "review_flags": review_flags,
        "trace": [trace],
    }


def ml_evaluation_node(state: CandidateAgentState) -> dict[str, Any]:
    t0 = time.time()
    features = state.features
    assert features is not None
    step = len(state.trace) + 1

    pred = predict_candidate_fit(features, state.recommendation)
    hybrid_decision = resolve_hybrid_decision(state.overall_score, state.recommendation, pred)
    duration = round((time.time() - t0) * 1000, 1)

    review_flags = list(state.review_flags)
    if hybrid_decision.review_recommendation_flag:
        review_flags.append(
            ReviewFlag(
                code=hybrid_decision.review_recommendation_flag,
                message=hybrid_decision.policy_reason,
            )
        )

    obs = (
        f"ML model [{pred.model_version}] ({pred.model_name}) predicts '{pred.predicted_label}' "
        f"(confidence: {pred.probabilities.get(pred.predicted_label, 0.0):.2f}). "
        f"Hybrid Decision: {hybrid_decision.final_decision} (Agreement: {hybrid_decision.agreement}, Borderline: {hybrid_decision.is_borderline})."
    )

    trace = AgentTraceEvent(
        step=step,
        agent="ML Evaluation Agent",
        task=f"Predict candidate classification using trained {pred.model_name} model and resolve hybrid decision",
        tool="predict_ml_fit",
        tool_input=str(features.model_dump()),
        observation=obs,
        status="warning" if not pred.agrees_with_rules else "success",
        duration_ms=duration,
        details={
            "modelName": pred.model_name,
            "predictedLabel": pred.predicted_label,
            "probabilities": pred.probabilities,
            "agreesWithRules": pred.agrees_with_rules,
            "hybridDecision": hybrid_decision.model_dump(),
        },
    )

    return {
        "ml_prediction": pred,
        "hybrid_decision": hybrid_decision,
        "review_flags": review_flags,
        "trace": [trace],
    }


def review_agent_node(state: CandidateAgentState) -> dict[str, Any]:
    t0 = time.time()
    step = len(state.trace) + 1

    flags = list(state.review_flags)
    # Check if keyword stuffing was flagged
    if state.stuffing and state.stuffing.suspected:
        flags.append(
            ReviewFlag(
                code="KEYWORD_STUFFING",
                message=f"Repetitive keyword frequency detected without supporting project/experience evidence: {state.stuffing.detail}",
            )
        )

    # Deduplicate flags by code
    seen_codes: set[str] = set()
    deduped_flags: list[ReviewFlag] = []
    for f in flags:
        if f.code not in seen_codes:
            deduped_flags.append(f)
            seen_codes.add(f.code)

    duration = round((time.time() - t0) * 1000, 1)
    trace = AgentTraceEvent(
        step=step,
        agent="Review Agent",
        task="Audit evaluation results for borderline scores, ML divergences, and suspicious patterns",
        tool="audit_candidate",
        tool_input=f"{len(deduped_flags)} potential flag conditions",
        observation=f"Candidate flagged for human recruiter review: {len(deduped_flags)} issue(s) detected.",
        status="warning",
        duration_ms=duration,
        details={"flags": [f.model_dump() for f in deduped_flags]},
    )

    return {
        "needs_human_review": True,
        "review_flags": deduped_flags,
        "trace": [trace],
    }


def recommendation_node(state: CandidateAgentState) -> dict[str, Any]:
    t0 = time.time()
    step = len(state.trace) + 1

    rec_labels = {
        Recommendation.STRONG_SHORTLIST: "Strong Match",
        Recommendation.SHORTLIST: "Match",
        Recommendation.REVIEW: "Needs Review",
        Recommendation.WEAK_MATCH: "Weak Match",
    }
    label = rec_labels.get(state.recommendation, "Weak Match")
    duration = round((time.time() - t0) * 1000, 1)

    trace = AgentTraceEvent(
        step=step,
        agent="Recommendation Agent",
        task="Finalize candidate recommendation based on calibrated thresholds",
        tool="assign_recommendation",
        tool_input=f"Score: {state.overall_score}/100",
        observation=f"Candidate assigned recommendation: {state.recommendation.value} ({label}).",
        status="success",
        duration_ms=duration,
        details={"recommendation": state.recommendation.value, "label": label},
    )

    return {"recommendation_label": label, "trace": [trace]}


def explanation_node(state: CandidateAgentState) -> dict[str, Any]:
    t0 = time.time()
    profile = state.profile
    ats = state.ats
    assert profile is not None and ats is not None
    step = len(state.trace) + 1

    explanation = generate_candidate_explanation(
        profile=profile,
        job=state.job,
        overall_score=state.overall_score,
        recommendation=state.recommendation,
        matches=state.matches,
        features=state.features,
        ats_score=ats.score,
    )

    # Recall episodic memory
    episodes = recall_candidate_memory(profile.name, exclude_run_id=state.run_id)

    duration = round((time.time() - t0) * 1000, 1)
    trace = AgentTraceEvent(
        step=step,
        agent="Explanation Agent",
        task=f"Generate human-readable explainable evaluation report via {explanation.generated_by}",
        tool="generate_explanation",
        tool_input=f"Grounded metrics for {profile.name}",
        observation=(
            f"Explanation generated successfully via {explanation.generated_by}"
            + (f" ({explanation.model})" if explanation.model else "")
            + f": {len(explanation.strengths)} strengths, {len(explanation.weaknesses)} weaknesses identified."
        ),
        status="success",
        duration_ms=duration,
        details={
            "generatedBy": explanation.generated_by,
            "strengthsCount": len(explanation.strengths),
            "weaknessesCount": len(explanation.weaknesses),
        },
    )

    return {
        "explanation": explanation,
        "memory_episodes": episodes,
        "trace": [trace],
    }

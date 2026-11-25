"""Candidate evaluation orchestrator running the LangGraph state machine."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path

from app.agents.graph import get_compiled_graph
from app.agents.state import CandidateAgentState
from app.db.database import save_candidate, save_evaluation
from app.schemas.candidate import CandidateProfile
from app.schemas.evaluation import CandidateEvaluation, Recommendation
from app.schemas.job import JobProfile
from app.services.quality.resume_quality import evaluate_resume_quality

logger = logging.getLogger(__name__)


def evaluate_single_candidate(
    file_path: Path,
    job: JobProfile,
    run_id: str,
    candidate_id: str | None = None,
) -> CandidateEvaluation:
    cid = candidate_id or f"cand_{uuid.uuid4().hex[:8]}"
    eval_id = f"eval_{uuid.uuid4().hex[:8]}"

    initial_state = CandidateAgentState(
        candidate_id=cid,
        run_id=run_id,
        file_path=str(file_path),
        job=job,
    )

    graph = get_compiled_graph()
    result_state = graph.invoke(initial_state)

    # Convert state to typed CandidateEvaluation
    profile: CandidateProfile | None = result_state.get("profile")
    c_name = profile.name if profile else "Unknown Candidate"
    raw_fname = file_path.name
    file_name = raw_fname[len(cid) + 1 :] if raw_fname.startswith(f"{cid}_") else raw_fname

    if result_state.get("parsing_failed"):
        eval_obj = CandidateEvaluation(
            evaluation_id=eval_id,
            run_id=run_id,
            candidate_id=cid,
            job_id=job.job_id,
            job_title=job.title,
            candidate_name=c_name,
            file_name=file_name,
            status="PARSING_FAILED",
            error=result_state.get("parsing_error") or "Document could not be parsed.",
            trace=result_state.get("trace", []),
        )
        return eval_obj

    # Persist candidate profile in long-term memory
    if profile:
        save_candidate(profile, file_name=file_name)

    matches = result_state.get("matches", [])
    matched_skills = [
        m.requirement for m in matches if m.status.value in {"MATCH", "SEMANTIC_MATCH"}
    ]
    missing_skills = [
        m.requirement for m in matches if m.status.value == "NOT_FOUND"
    ]
    uncertain_skills = [
        m.requirement
        for m in matches
        if m.status.value in {"UNCERTAIN", "CLAIMED_ONLY", "INFERRED"}
    ]

    features = result_state.get("features")
    ats = result_state.get("ats")

    rec = result_state.get("recommendation", Recommendation.WEAK_MATCH)
    rec_label = result_state.get("recommendation_label", "Weak Match")

    exp_years = (profile.total_experience_months / 12.0) if profile else 0.0

    eval_obj = CandidateEvaluation(
        evaluation_id=eval_id,
        run_id=run_id,
        candidate_id=cid,
        job_id=job.job_id,
        job_title=job.title,
        candidate_name=c_name,
        file_name=file_name,
        status="COMPLETED",
        error=None,
        overall_score=result_state.get("overall_score", 0.0),
        ats_score=ats.score if ats else 0.0,
        required_skill_match=round(features.required_skill_match * 100.0, 1) if features else 0.0,
        preferred_skill_match=round(features.preferred_skill_match * 100.0, 1) if features else 0.0,
        experience_match=round(features.experience_match * 100.0, 1) if features else 0.0,
        semantic_similarity=round(features.semantic_similarity * 100.0, 1) if features else 0.0,
        project_match=round(features.project_match * 100.0, 1) if features else 0.0,
        education_match=round(features.education_match * 100.0, 1) if features else 0.0,
        resume_quality=round(features.resume_quality * 100.0, 1) if features else 0.0,
        quality_evaluation=evaluate_resume_quality(profile) if profile else None,
        keyword_ats_score=result_state.get("keyword_ats_score", 0.0),
        score_breakdown=result_state.get("score_breakdown"),
        features=features,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        uncertain_skills=uncertain_skills,
        requirement_matches=matches,
        ats=ats,
        stuffing=result_state.get("stuffing"),
        ml_prediction=result_state.get("ml_prediction"),
        hybrid_decision=result_state.get("hybrid_decision"),
        recommendation=rec,
        recommendation_label=rec_label,
        needs_human_review=result_state.get("needs_human_review", False),
        review_flags=result_state.get("review_flags", []),
        strengths=result_state.get("explanation").strengths if result_state.get("explanation") else [],
        weaknesses=result_state.get("explanation").weaknesses if result_state.get("explanation") else [],
        explanation=result_state.get("explanation"),
        experience_years=exp_years,
        profile=profile,
        memory_recall=result_state.get("memory_episodes", []),
        trace=result_state.get("trace", []),
        created_at=datetime.now(UTC),
    )

    # Save evaluation result in database
    save_evaluation(eval_obj)
    return eval_obj

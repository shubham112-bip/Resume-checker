"""ReAct tool definitions used by agents, recruiter assistant, and the MCP server."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.db.database import get_candidate_by_id, get_job_by_id, list_evaluations_for_run, recall_candidate_memory
from app.schemas.candidate import CandidateProfile
from app.schemas.evaluation import FeatureVector, MLPrediction, Recommendation
from app.schemas.job import JobProfile
from app.services.ats.ats_engine import evaluate_ats
from app.services.embeddings.embedder import get_embedder
from app.services.embeddings.vector_store import FAISSVectorStore
from app.services.features.feature_engine import compute_features
from app.services.ml.predictor import predict_candidate_fit
from app.services.parsing.jd_parser import parse_job_description
from app.services.parsing.resume_parser import parse_resume
from app.services.semantic.matcher import build_candidate_vector_store, calibrate_similarity


def tool_parse_resume(file_path: str, strategy: str = "primary") -> dict[str, Any]:
    """Parse resume PDF/DOCX and return structured profile."""
    profile = parse_resume(Path(file_path), strategy=strategy)
    return {
        "candidateId": profile.candidate_id,
        "name": profile.name,
        "skills": profile.skills,
        "experienceMonths": profile.total_experience_months,
        "education": [e.degree for e in profile.education],
        "diagnostics": profile.diagnostics.model_dump() if profile.diagnostics else None,
    }


def tool_analyze_job(text: str) -> dict[str, Any]:
    """Analyze job description and classify requirements."""
    job = parse_job_description(text)
    return {
        "jobId": job.job_id,
        "title": job.title,
        "requiredSkills": job.required_skills,
        "preferredSkills": job.preferred_skills,
        "minExperience": job.min_experience_years,
    }


def tool_search_resume_evidence(candidate_id: str, requirement: str, top_k: int = 3) -> dict[str, Any]:
    """Search vector index of candidate resume for passages relevant to a requirement."""
    profile, _ = get_candidate_by_id(candidate_id)
    if not profile:
        return {"error": f"Candidate {candidate_id} not found", "passages": []}

    store, _ = build_candidate_vector_store(profile)
    embedder = get_embedder()
    q_emb = embedder.embed_queries([requirement])[0]
    results = store.search(q_emb, top_k=top_k)

    passages = [
        {
            "text": r.chunk.text,
            "section": r.chunk.section,
            "similarity": round(r.similarity, 3),
        }
        for r in results
    ]
    return {"candidateId": candidate_id, "requirement": requirement, "passages": passages}


def tool_evaluate_ats_rules(candidate_id: str, job_id: str) -> dict[str, Any]:
    """Execute traditional ATS rule-based compatibility checks."""
    profile, _ = get_candidate_by_id(candidate_id)
    job = get_job_by_id(job_id)
    if not profile or not job:
        return {"error": "Profile or Job not found"}

    ats = evaluate_ats(profile, job)
    return {
        "score": ats.score,
        "contactScore": ats.contact_score,
        "sectionScore": ats.section_score,
        "keywordCoverage": ats.keyword_coverage,
        "matchedKeywords": ats.matched_keywords,
        "missingKeywords": ats.missing_keywords,
        "issues": ats.issues,
    }


def tool_predict_ml_fit(features_dict: dict[str, float], recommendation_str: str) -> dict[str, Any]:
    """Predict candidate shortlist status using the trained ML model."""
    f = FeatureVector.model_validate(features_dict)
    rec = Recommendation(recommendation_str)
    pred = predict_candidate_fit(f, rec)
    return pred.model_dump(by_alias=True)


def tool_recall_memory(candidate_name: str) -> dict[str, Any]:
    """Recall past candidate evaluation episodes from long-term memory."""
    episodes = recall_candidate_memory(candidate_name)
    return {
        "candidateName": candidate_name,
        "episodes": [e.model_dump(by_alias=True) for e in episodes],
    }

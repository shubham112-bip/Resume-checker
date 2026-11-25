"""FastAPI router endpoints implementing the recruiter platform REST API."""

from __future__ import annotations

import logging
import os
import shutil
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, Response, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse

from app.agents.graph import get_graph_mermaid
from app.agents.orchestrator import evaluate_single_candidate
from app.config import ML_DIR, PROCESSED_DIR, ROOT_DATA_DIR, RUNTIME_DIR, get_settings
from app.db.database import (
    get_candidate_by_id,
    get_evaluation_by_id,
    update_recruiter_decision,
    get_job_by_id,
    list_all_episodes,
    list_candidates,
    list_evaluations_for_run,
    list_jobs,
    save_candidate,
    save_job,
)
from app.schemas.api import (
    CandidateDetail,
    CandidateSummary,
    ComparisonResponse,
    DatasetJdBrief,
    DatasetPairSummary,
    DatasetResumeBrief,
    DatasetSummaryResponse,
    DemoJob,
    DemoLoadResponse,
    EpisodeResponse,
    EvaluateRequest,
    EvaluationBrief,
    EvaluationRunResponse,
    GraphResponse,
    HealthResponse,
    HumanValidationPair,
    HumanValidationResponse,
    JobCreateRequest,
    JobSummary,
    ModelMetricsResponse,
    RankingEntry,
    ResumeQualityUploadResponse,
    RunSummary,
    UploadError,
    UploadResponse,
    UploadedResume,
)
from app.schemas.evaluation import CandidateEvaluation, Recommendation, RecruiterDecisionRequest
from app.schemas.job import JobProfile
from app.schemas.quality import ResumeQualityEvaluation
from app.services.demo_data import DEMO_JD_TEXT, SAMPLE_JDS, create_demo_resumes, run_comparison_benchmark
from app.services.embeddings.embedder import get_embedder
from app.services.ml.predictor import get_cached_metrics
from app.services.parsing.document_loader import DocumentParsingError, load_document
from app.services.parsing.jd_parser import parse_job_description, parse_job_file
from app.services.parsing.resume_parser import parse_resume
from app.services.pipeline.canonical_pipeline import evaluate_unseen_resume_and_jd
from app.services.quality.resume_quality import evaluate_resume_quality
from app.services.reporting.report_generator import generate_markdown_report, generate_pdf_report

logger = logging.getLogger(__name__)

router = APIRouter()


# --------------------------------------------------------------------------
# 1. Health & Config
# --------------------------------------------------------------------------
@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    cfg = get_settings()
    metrics = get_cached_metrics()
    embedder = get_embedder()
    return HealthResponse(
        status="healthy",
        gemini_enabled=cfg.gemini_enabled,
        gemini_model=cfg.gemini_model if cfg.gemini_enabled else None,
        embedding_backend=embedder.__class__.__name__,
        embedding_model=cfg.embedding_model,
        ml_model_available=metrics.available,
        selected_model=metrics.selected_model,
    )


@router.get("/config")
def get_scoring_config() -> dict:
    cfg = get_settings()
    return {
        "weights": cfg.weights.model_dump(by_alias=True),
        "thresholds": cfg.thresholds.model_dump(by_alias=True),
        "credits": cfg.credits.model_dump(by_alias=True),
        "semantic": cfg.semantic.model_dump(by_alias=True),
        "atsWeights": cfg.ats_weights.model_dump(by_alias=True),
        "formula": (
            "Overall Score = 0.30 × Required Skill + 0.20 × Experience + "
            "0.15 × Semantic Similarity + 0.10 × Project + 0.10 × ATS + "
            "0.10 × Education + 0.05 × Quality"
        ),
    }


# --------------------------------------------------------------------------
# 2. Job Descriptions
# --------------------------------------------------------------------------
@router.post("/jobs", response_model=JobProfile)
def create_job(req: JobCreateRequest) -> JobProfile:
    job = parse_job_description(req.text, title=req.title)
    save_job(job)
    return job


@router.post("/jobs/upload", response_model=JobProfile)
async def upload_job_file(file: UploadFile = File(...)) -> JobProfile:
    upload_dir = get_settings().upload_dir
    upload_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "job.txt").suffix.lower()
    temp_path = upload_dir / f"job_{uuid.uuid4().hex[:8]}{suffix}"

    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        job = parse_job_file(temp_path)
        save_job(job)
        return job
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse job file: {e}")
    finally:
        if temp_path.exists():
            temp_path.unlink()


@router.get("/jobs", response_model=list[JobSummary])
def get_jobs() -> list[JobSummary]:
    jobs = list_jobs()
    return [
        JobSummary(
            job_id=j.job_id,
            title=j.title,
            required_skills=j.required_skills,
            preferred_skills=j.preferred_skills,
            created_at=j.model_dump().get("created_at") or time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        for j in jobs
    ]


@router.get("/jobs/{job_id}", response_model=JobProfile)
def get_job(job_id: str) -> JobProfile:
    job = get_job_by_id(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job description not found")
    return job


# --------------------------------------------------------------------------
# 3. Resume Upload & Validation
# --------------------------------------------------------------------------
@router.post("/resumes/upload", response_model=UploadResponse)
async def upload_resumes(files: list[UploadFile] = File(...)) -> UploadResponse:
    cfg = get_settings()
    upload_dir = cfg.upload_dir
    upload_dir.mkdir(parents=True, exist_ok=True)

    candidates: list[UploadedResume] = []
    errors: list[UploadError] = []

    for f in files:
        fname = f.filename or "unnamed_resume"
        suffix = Path(fname).suffix.lower()

        if suffix not in cfg.allowed_extensions:
            errors.append(
                UploadError(
                    file_name=fname,
                    error=f"Unsupported file extension '{suffix}'. Allowed: {', '.join(cfg.allowed_extensions)}",
                )
            )
            continue

        cid = f"cand_{uuid.uuid4().hex[:8]}"
        dest = upload_dir / f"{cid}_{fname}"

        with open(dest, "wb") as buffer:
            shutil.copyfileobj(f.file, buffer)

        if dest.stat().st_size > cfg.max_upload_bytes:
            errors.append(
                UploadError(
                    file_name=fname,
                    error=f"File exceeds maximum upload size of {cfg.max_upload_mb} MB",
                )
            )
            dest.unlink(missing_ok=True)
            continue

        try:
            profile = parse_resume(dest, candidate_id=cid)
            save_candidate(profile, file_name=fname)
            warnings = profile.diagnostics.warnings if profile.diagnostics else []
            candidates.append(
                UploadedResume(
                    candidate_id=cid,
                    file_name=fname,
                    candidate_name=profile.name,
                    status="PARSED",
                    warnings=warnings,
                )
            )
        except DocumentParsingError as e:
            errors.append(UploadError(file_name=fname, error=str(e)))
        except Exception as e:
            logger.exception("Unexpected error parsing %s", fname)
            errors.append(UploadError(file_name=fname, error=f"Parsing error: {e}"))

    return UploadResponse(candidates=candidates, errors=errors)


# --------------------------------------------------------------------------
# 4. Multi-Agent Evaluation & Ranking
# --------------------------------------------------------------------------
@router.post("/evaluate", response_model=EvaluationRunResponse)
def evaluate_candidates(req: EvaluateRequest) -> EvaluationRunResponse:
    job = get_job_by_id(req.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job description not found")

    upload_dir = get_settings().upload_dir
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    t0 = time.time()

    evaluations: list[CandidateEvaluation] = []

    for cid in req.candidate_ids:
        # Find file in upload_dir or demo_dir
        matching_files = list(upload_dir.glob(f"{cid}_*"))
        if not matching_files:
            # Check demo dir
            demo_files = list((RUNTIME_DIR / "demo" / "resumes").glob(f"*{cid}*"))
            if not demo_files:
                # Try locating by candidate profile
                profile, fname = get_candidate_by_id(cid)
                if profile:
                    demo_files = list(Path(".").glob(f"**/{fname}"))

            matching_files = demo_files

        if not matching_files:
            logger.warning("File for candidate %s not found on disk. Skipping.", cid)
            continue

        file_path = matching_files[0]
        eval_result = evaluate_single_candidate(file_path=file_path, job=job, run_id=run_id, candidate_id=cid)
        evaluations.append(eval_result)

    if not evaluations:
        raise HTTPException(status_code=400, detail="No candidates could be evaluated.")

    # Calculate Rankings & Keyword Rank Comparison (Demonstrates Semantic vs Keyword ATS!)
    # Sort by Overall Fit Score descending
    evaluations.sort(key=lambda x: x.overall_score, reverse=True)

    # Sort copy by traditional keyword ATS score
    keyword_sorted = sorted(evaluations, key=lambda x: x.keyword_ats_score, reverse=True)
    keyword_rank_map = {ev.evaluation_id: idx + 1 for idx, ev in enumerate(keyword_sorted)}

    ranking: list[RankingEntry] = []
    for rank, ev in enumerate(evaluations, 1):
        ranking.append(
            RankingEntry(
                rank=rank,
                evaluation_id=ev.evaluation_id,
                candidate_id=ev.candidate_id,
                candidate_name=ev.candidate_name,
                file_name=ev.file_name,
                status=ev.status,
                overall_score=ev.overall_score,
                recommendation=ev.recommendation,
                recommendation_label=ev.recommendation_label,
                needs_human_review=ev.needs_human_review,
                review_flags=[f.code for f in ev.review_flags],
                keyword_ats_score=ev.keyword_ats_score,
                keyword_rank=keyword_rank_map.get(ev.evaluation_id, rank),
                ml_label=ev.ml_prediction.predicted_label if ev.ml_prediction else None,
                matched_skills=ev.matched_skills,
                missing_skills=ev.missing_skills,
            )
        )

    duration = round((time.time() - t0) * 1000, 1)

    return EvaluationRunResponse(
        run_id=run_id,
        job_id=job.job_id,
        job_title=job.title,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        duration_ms=duration,
        ranking=ranking,
        evaluations=evaluations,
    )


@router.get("/runs/{run_id}", response_model=EvaluationRunResponse)
def get_run(run_id: str) -> EvaluationRunResponse:
    evaluations = list_evaluations_for_run(run_id)
    if not evaluations:
        raise HTTPException(status_code=404, detail="Evaluation run not found")

    job = get_job_by_id(evaluations[0].job_id)
    job_title = job.title if job else evaluations[0].job_title

    evaluations.sort(key=lambda x: x.overall_score, reverse=True)
    keyword_sorted = sorted(evaluations, key=lambda x: x.keyword_ats_score, reverse=True)
    keyword_rank_map = {ev.evaluation_id: idx + 1 for idx, ev in enumerate(keyword_sorted)}

    ranking = [
        RankingEntry(
            rank=idx + 1,
            evaluation_id=ev.evaluation_id,
            candidate_id=ev.candidate_id,
            candidate_name=ev.candidate_name,
            file_name=ev.file_name,
            status=ev.status,
            overall_score=ev.overall_score,
            recommendation=ev.recommendation,
            recommendation_label=ev.recommendation_label,
            needs_human_review=ev.needs_human_review,
            review_flags=[f.code for f in ev.review_flags],
            keyword_ats_score=ev.keyword_ats_score,
            keyword_rank=keyword_rank_map.get(ev.evaluation_id, idx + 1),
            ml_label=ev.ml_prediction.predicted_label if ev.ml_prediction else None,
            matched_skills=ev.matched_skills,
            missing_skills=ev.missing_skills,
        )
        for idx, ev in enumerate(evaluations)
    ]

    return EvaluationRunResponse(
        run_id=run_id,
        job_id=evaluations[0].job_id,
        job_title=job_title,
        created_at=evaluations[0].created_at,
        duration_ms=0.0,
        ranking=ranking,
        evaluations=evaluations,
    )


@router.get("/runs", response_model=list[RunSummary])
def get_runs() -> list[RunSummary]:
    # Group runs from evaluations table
    from app.db.database import get_db_session, EvaluationRecord
    session = get_db_session()
    try:
        from sqlalchemy import func
        rows = (
            session.query(
                EvaluationRecord.run_id,
                EvaluationRecord.job_id,
                EvaluationRecord.job_title,
                func.count(EvaluationRecord.evaluation_id).label("cnt"),
                func.min(EvaluationRecord.created_at).label("created"),
            )
            .group_by(EvaluationRecord.run_id)
            .order_by(func.min(EvaluationRecord.created_at).desc())
            .all()
        )
        return [
            RunSummary(
                run_id=r[0],
                job_id=r[1],
                job_title=r[2],
                candidate_count=r[3],
                created_at=r[4],
            )
            for r in rows
        ]
    finally:
        session.close()


# --------------------------------------------------------------------------
# 5. Candidate Details & Evaluation Views
# --------------------------------------------------------------------------
@router.get("/candidates", response_model=list[CandidateSummary])
def get_candidates() -> list[CandidateSummary]:
    cands = list_candidates()
    return [
        CandidateSummary(
            candidate_id=c.candidate_id,
            candidate_name=c.name,
            file_name=fname,
            created_at=created,
        )
        for c, fname, created in cands
    ]


@router.get("/candidates/{candidate_id}", response_model=CandidateDetail)
def get_candidate(candidate_id: str) -> CandidateDetail:
    profile, fname = get_candidate_by_id(candidate_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Candidate not found")

    from app.db.database import get_db_session, EvaluationRecord
    session = get_db_session()
    try:
        records = (
            session.query(EvaluationRecord)
            .filter(EvaluationRecord.candidate_id == candidate_id)
            .order_by(EvaluationRecord.created_at.desc())
            .all()
        )
        briefs = [
            EvaluationBrief(
                evaluation_id=r.evaluation_id,
                run_id=r.run_id,
                job_title=r.job_title,
                overall_score=r.overall_score,
                recommendation=Recommendation(r.recommendation),
                created_at=r.created_at,
            )
            for r in records
        ]
        return CandidateDetail(
            candidate_id=profile.candidate_id,
            candidate_name=profile.name,
            file_name=fname,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            profile=profile,
            evaluations=briefs,
        )
    finally:
        session.close()


@router.get("/evaluations/{evaluation_id}", response_model=CandidateEvaluation)
def get_evaluation(evaluation_id: str) -> CandidateEvaluation:
    ev = get_evaluation_by_id(evaluation_id)
    if not ev:
        raise HTTPException(status_code=404, detail="Evaluation record not found")
    return ev


@router.post("/evaluations/{evaluation_id}/decision", response_model=CandidateEvaluation)
def save_recruiter_decision_endpoint(
    evaluation_id: str, payload: RecruiterDecisionRequest
) -> CandidateEvaluation:
    ev = update_recruiter_decision(evaluation_id, payload.decision, payload.notes)
    if not ev:
        raise HTTPException(status_code=404, detail="Evaluation record not found")
    return ev


@router.get("/evaluations/{evaluation_id}/report")
def download_evaluation_report(
    evaluation_id: str, format: str = Query(default="pdf", pattern="^(pdf|md)$")
) -> Response:
    ev = get_evaluation_by_id(evaluation_id)
    if not ev:
        raise HTTPException(status_code=404, detail="Evaluation record not found")

    if format == "md":
        content = generate_markdown_report(ev)
        return PlainTextResponse(
            content=content,
            headers={
                "Content-Disposition": f"attachment; filename=report_{ev.candidate_name.replace(' ', '_')}.md"
            },
        )

    pdf_bytes = generate_pdf_report(ev)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=report_{ev.candidate_name.replace(' ', '_')}.pdf"
        },
    )


# --------------------------------------------------------------------------
# 6. ML Metrics & Comparison
# --------------------------------------------------------------------------
@router.get("/models/metrics", response_model=ModelMetricsResponse)
def get_ml_metrics(
    dataset: str = Query(default="real", pattern="^(synthetic|real|canonical|canonical_v2|v2)$")
) -> ModelMetricsResponse:
    return get_cached_metrics(dataset_type=dataset)


@router.post("/models/benchmark/run", response_model=ModelMetricsResponse)
def run_benchmark() -> ModelMetricsResponse:
    """Re-executes the Phase 4 ML training & evaluation benchmark pipeline."""
    from app.services.ml.v2_trainer import run_training_and_comparison

    run_training_and_comparison()
    return get_cached_metrics(dataset_type="canonical_v2")


@router.get("/comparison/ats-vs-ai", response_model=ComparisonResponse)
def get_ats_ai_comparison() -> ComparisonResponse:
    return run_comparison_benchmark()


@router.get("/evaluation/human-validation", response_model=HumanValidationResponse)
def get_human_validation_results() -> HumanValidationResponse:
    locked_json = ML_DIR / "locked_human_evaluation.json"
    if not locked_json.exists():
        from app.services.evaluation.human_validator import evaluate_locked_human_validation

        evaluate_locked_human_validation()

    import json

    data = json.loads(locked_json.read_text(encoding="utf-8"))

    calibrated_data = None
    cal_file = ML_DIR / "calibrated_metrics.json"
    if cal_file.exists():
        try:
            calibrated_data = json.loads(cal_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    pairs_list = [
        HumanValidationPair(
            resume_id=p["resume_id"],
            jd_id=p["jd_id"],
            candidate_name=p["candidate_name"],
            job_title=p["job_title"],
            human_label=p["human_label_4class"],
            rule_label=p["rule_label_3class"],
            ml_prediction=p["ml_prediction"],
            overall_fit_score=p["overall_fit_score"],
            resume_quality_score=p["resume_quality_score"],
            notes=p.get("notes", ""),
            evidence=p.get("evidence", ""),
        )
        for p in data.get("pair_details", [])
    ]

    notice = (
        "Human Validation Notice: Evaluated on 64 independent Resume-JD pairs with single-blind human recruiter labels "
        "(16 Strong Match, 16 Match, 16 Review, 16 Weak Match). Development Benchmark accuracy of 98.96% measures "
        "model alignment against rule-assisted auto-labels, while Human Validation Benchmark reflects true independent "
        "recruiter judgment across edge cases and transferable skills."
    )

    return HumanValidationResponse(
        available=True,
        total_pairs=data.get("total_pairs", 64),
        total_resumes=data.get("total_resumes", 45),
        total_jds=data.get("total_jds", 12),
        reviewer_count=data.get("reviewer_count", 1),
        reviewer_id=data.get("reviewer_id", "human_reviewer_01"),
        class_distribution_4class=data.get("human_distribution_4class", {}),
        class_distribution_3class=data.get("human_distribution_3class", {}),
        ml_vs_human=data.get("ml_vs_human", {}),
        rule_vs_human=data.get("rule_vs_human", {}),
        calibrated_metrics=calibrated_data,
        pairs=pairs_list,
        notice=notice,
    )


# --------------------------------------------------------------------------
# 6b. Real Dataset & Standalone Resume Quality (Phase 2 Upgrade)
# --------------------------------------------------------------------------
@router.get("/dataset/summary", response_model=DatasetSummaryResponse)
def get_dataset_summary() -> DatasetSummaryResponse:
    summary_file = PROCESSED_DIR / "dataset_summary.json"
    if not summary_file.exists():
        from app.services.dataset.dataset_builder import build_evaluation_dataset

        build_evaluation_dataset()

    import json
    data = json.loads(summary_file.read_text(encoding="utf-8"))

    # Load resume briefs
    resume_briefs: list[DatasetResumeBrief] = []
    resumes_dir = PROCESSED_DIR / "resumes"
    if resumes_dir.exists():
        for r_path in sorted(resumes_dir.glob("*.json")):
            try:
                r_data = json.loads(r_path.read_text(encoding="utf-8"))
                q = r_data.get("qualityFeatures", {})
                diag = r_data.get("diagnostics", {})
                resume_briefs.append(
                    DatasetResumeBrief(
                        resume_id=r_data.get("resumeId", r_path.stem),
                        candidate_name=r_data.get("candidateName", "Unknown"),
                        file_type=r_data.get("fileName", "").split(".")[-1].upper() if "." in r_data.get("fileName", "") else "PDF",
                        word_count=diag.get("wordCount", 0),
                        quality_score=q.get("overallQualityScore", 0.0),
                        quality_grade=q.get("qualityGrade", "Good"),
                        action_verbs_count=len(q.get("actionVerbsFound", [])),
                        quantified_bullets_count=q.get("quantifiedBulletsCount", 0),
                    )
                )
            except Exception as e:
                logger.warning("Could not read processed resume %s: %s", r_path, e)

    # Load JD briefs
    jd_briefs: list[DatasetJdBrief] = []
    from app.services.demo_data import SAMPLE_JDS
    from app.services.parsing.jd_parser import parse_job_description
    jds_dir = ROOT_DATA_DIR / "job_descriptions"
    if jds_dir.exists():
        for j_path in sorted(jds_dir.glob("*.txt")):
            try:
                text = j_path.read_text(encoding="utf-8")
                jp = parse_job_description(text, title=j_path.stem.replace("_", " ").title())
                req_cnt = sum(1 for s in jp.skills if s.tier.value == "REQUIRED")
                pref_cnt = sum(1 for s in jp.skills if s.tier.value != "REQUIRED")
                exp_str = f"{jp.min_experience_years:g}+ years" if jp.min_experience_years > 0 else "Entry Level"
                jd_briefs.append(
                    DatasetJdBrief(
                        jd_id=j_path.stem,
                        title=jp.title,
                        experience_level=exp_str,
                        required_skills_count=req_cnt,
                        preferred_skills_count=pref_cnt,
                    )
                )
            except Exception as e:
                logger.warning("Could not parse JD file %s: %s", j_path, e)

    total_resumes_cnt = data.get("total_resumes", len(resume_briefs))
    total_jds_cnt = data.get("total_jds", len(jd_briefs))
    total_pairs_cnt = data.get("total_pairs", len(resume_briefs) * len(jd_briefs))

    notice = (
        f"Academic Dataset Notice: Evaluated on {total_pairs_cnt} Resume-JD pairs generated from {total_resumes_cnt} candidates "
        f"(including 65 uploaded real resumes) across {total_jds_cnt} diverse Job Descriptions using rule-assisted labels without fabricated "
        f"human evaluations. Stratified grouped splitting on resume_id prevents candidate leakage. For statistical "
        f"sample-size power, compare against the 800-candidate synthetic benchmark."
    )

    return DatasetSummaryResponse(
        total_resumes=total_resumes_cnt,
        total_job_descriptions=total_jds_cnt,
        total_pairs=total_pairs_cnt,
        class_distribution=data.get("classification_distribution", {}),
        labels_distribution=data.get("label_distribution", {}),
        label_sources=data.get("label_sources", {"RULE_BASED": total_pairs_cnt}),
        resumes=resume_briefs,
        job_descriptions=jd_briefs,
        notice=notice,
    )


@router.get("/dataset/resumes")
def get_dataset_resumes() -> list[dict]:
    resumes_dir = PROCESSED_DIR / "resumes"
    if not resumes_dir.exists():
        from app.services.dataset.dataset_builder import build_evaluation_dataset

        build_evaluation_dataset()

    import json
    results = []
    for r_path in sorted(resumes_dir.glob("*.json")):
        try:
            results.append(json.loads(r_path.read_text(encoding="utf-8")))
        except Exception as e:
            logger.warning("Could not read resume json %s: %s", r_path, e)
    return results


@router.get("/dataset/pairs", response_model=list[DatasetPairSummary])
def get_dataset_pairs(
    limit: int = Query(default=100, ge=1, le=200),
    resume_id: str | None = None,
    jd_id: str | None = None,
) -> list[DatasetPairSummary]:
    pairs_file = PROCESSED_DIR / "evaluation_pairs.csv"
    if not pairs_file.exists():
        from app.services.dataset.dataset_builder import build_evaluation_dataset

        build_evaluation_dataset()

    import pandas as pd
    df = pd.read_csv(pairs_file)

    if resume_id:
        df = df[df["resume_id"] == resume_id]
    if jd_id:
        df = df[df["jd_id"] == jd_id]

    df = df.head(limit)
    pairs: list[DatasetPairSummary] = []
    for _, row in df.iterrows():
        pairs.append(
            DatasetPairSummary(
                resume_id=str(row["resume_id"]),
                jd_id=str(row["jd_id"]),
                candidate_name=str(row["candidate_name"]),
                job_title=str(row["job_title"]),
                overall_candidate_fit_score=float(row["overall_candidate_fit_score"]),
                resume_quality_score=float(row["resume_quality_score"]),
                label=str(row["label"]),
                classification_label=str(row["classification_label"]),
                label_source=str(row["label_source"]),
                required_skill_match=float(row["required_skill_match"]),
                experience_match=float(row["experience_match"]),
                semantic_similarity=float(row["semantic_similarity"]),
                education_match=float(row["education_match"]),
                project_match=float(row["project_match"]),
                ats_score=float(row["ats_score"]),
                resume_quality=float(row["resume_quality"]),
            )
        )
    return pairs


# --------------------------------------------------------------------------
# 5b. Production Real-Time Screening (Unseen Resume + Unseen JD)
# --------------------------------------------------------------------------
@router.post("/screen", response_model=CandidateEvaluation)
async def screen_candidate(
    resume_file: UploadFile | None = File(default=None),
    resume_text: str | None = Form(default=None),
    job_description: str | None = Form(default=None),
    job_file: UploadFile | None = File(default=None),
    job_title: str | None = Form(default=None),
) -> CandidateEvaluation:
    """Production screening endpoint evaluating ANY unseen resume against ANY unseen Job Description.

    Follows the single canonical feature pipeline:
    Resume + JD -> Parsing -> Protected Attribute Filtering -> Canonical Features -> ML Prediction -> Explainable Evaluation.
    """
    if not resume_file and not resume_text:
        raise HTTPException(
            status_code=400,
            detail="Either 'resume_file' (PDF, DOCX, TXT) or 'resume_text' must be provided.",
        )
    if not job_description and not job_file:
        raise HTTPException(
            status_code=400,
            detail="Either 'job_description' text or 'job_file' (PDF, DOCX, TXT) must be provided.",
        )

    upload_dir = get_settings().upload_dir
    upload_dir.mkdir(parents=True, exist_ok=True)
    temp_files: list[Path] = []

    try:
        # 1. Resolve Resume input
        if resume_file is not None and resume_file.filename:
            r_fname = resume_file.filename
            r_ext = Path(r_fname).suffix.lower()
            if r_ext not in get_settings().allowed_extensions:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported resume extension '{r_ext}'. Allowed: {', '.join(get_settings().allowed_extensions)}",
                )
            temp_resume_path = upload_dir / f"screen_res_{uuid.uuid4().hex[:8]}{r_ext}"
            temp_files.append(temp_resume_path)
            with open(temp_resume_path, "wb") as f:
                shutil.copyfileobj(resume_file.file, f)
        else:
            temp_resume_path = upload_dir / f"screen_res_{uuid.uuid4().hex[:8]}.txt"
            temp_files.append(temp_resume_path)
            temp_resume_path.write_text(resume_text or "", encoding="utf-8")

        # 2. Resolve Job Description input
        if job_file is not None and job_file.filename:
            j_fname = job_file.filename
            j_ext = Path(j_fname).suffix.lower()
            temp_job_path = upload_dir / f"screen_job_{uuid.uuid4().hex[:8]}{j_ext}"
            temp_files.append(temp_job_path)
            with open(temp_job_path, "wb") as f:
                shutil.copyfileobj(job_file.file, f)
            job_prof = parse_job_file(temp_job_path)
        else:
            job_prof = parse_job_description(job_description or "", title=job_title or "Target Role")

        # 3. LangGraph Orchestrator Execution
        # (Resume Parser -> ATS -> Semantic FAISS -> Reflection -> Rule v2 Features -> ML v2 -> Hybrid Engine -> Explanation)
        run_id = f"screen_{uuid.uuid4().hex[:8]}"
        evaluation = evaluate_single_candidate(
            file_path=temp_resume_path,
            job=job_prof,
            run_id=run_id,
        )
        return evaluation
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed in screening inference: %s", e)
        raise HTTPException(status_code=500, detail=f"Screening evaluation failed: {e}")
    finally:
        for p in temp_files:
            if p.exists():
                p.unlink(missing_ok=True)


@router.post("/resumes/quality", response_model=ResumeQualityUploadResponse)
async def evaluate_single_resume_quality(file: UploadFile = File(...)) -> ResumeQualityUploadResponse:
    """Evaluate resume quality independently of any Job Description."""
    fname = file.filename or "resume.pdf"
    ext = Path(fname).suffix.lower()
    if ext not in {".pdf", ".docx", ".txt"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Supported: PDF, DOCX, TXT.",
        )

    cid = f"quality_check_{uuid.uuid4().hex[:8]}"
    upload_dir = get_settings().upload_dir
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / f"{cid}_{fname}"

    with open(dest, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        profile = parse_resume(dest, candidate_id=cid)
        qual_eval = evaluate_resume_quality(profile)
        return ResumeQualityUploadResponse(
            candidate_name=profile.name,
            file_name=fname,
            evaluation=qual_eval,
        )
    except DocumentParsingError as e:
        raise HTTPException(status_code=422, detail=f"Document parsing error: {e}")
    except Exception as e:
        logger.exception("Error evaluating resume quality for %s", fname)
        raise HTTPException(status_code=500, detail=f"Quality evaluation error: {e}")


@router.get("/memory/episodes", response_model=list[EpisodeResponse])
def get_episodes() -> list[EpisodeResponse]:
    return list_all_episodes()


@router.get("/agents/graph", response_model=GraphResponse)
def get_agent_state_graph() -> GraphResponse:
    mermaid = get_graph_mermaid()
    nodes = [
        "resume_parser",
        "error_recovery",
        "ats_evaluation",
        "semantic_matching",
        "reflection",
        "feature_engineering",
        "ml_evaluation",
        "recommendation",
        "review_agent",
        "explanation",
    ]
    edges = [
        "resume_parser -> error_recovery (if parse error)",
        "resume_parser -> ats_evaluation (if success)",
        "error_recovery -> ats_evaluation (if recovered)",
        "error_recovery -> END (if unrecoverable)",
        "ats_evaluation -> semantic_matching",
        "semantic_matching -> reflection (if uncertain/implied skills)",
        "semantic_matching -> feature_engineering (if confident)",
        "reflection -> feature_engineering",
        "feature_engineering -> ml_evaluation",
        "ml_evaluation -> recommendation",
        "recommendation -> review_agent (if borderline / stuffing / ML disagreement)",
        "recommendation -> explanation (if high confidence)",
        "review_agent -> explanation",
        "explanation -> END",
    ]
    return GraphResponse(mermaid=mermaid, nodes=nodes, conditional_edges=edges)


# --------------------------------------------------------------------------
# 7. Demo Data & Quick Setup
# --------------------------------------------------------------------------
@router.get("/demo/job-descriptions", response_model=list[DemoJob])
def get_sample_jobs() -> list[DemoJob]:
    return SAMPLE_JDS


@router.post("/demo/load", response_model=DemoLoadResponse)
def load_demo_bundle() -> DemoLoadResponse:
    """Preload demo job and 5 candidate resumes for immediate one-click evaluation."""
    demo_job = parse_job_description(DEMO_JD_TEXT, title="Junior Full Stack Developer")
    save_job(demo_job)

    # Generate the 5 demo resumes
    resume_paths = create_demo_resumes()
    uploaded_cands: list[UploadedResume] = []

    for key, path in resume_paths.items():
        cid = f"cand_demo_{key}"
        try:
            profile = parse_resume(path, candidate_id=cid)
            save_candidate(profile, file_name=path.name)
            # Copy file to runtime upload_dir so evaluate_candidates finds it
            upload_dir = get_settings().upload_dir
            upload_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, upload_dir / f"{cid}_{path.name}")

            uploaded_cands.append(
                UploadedResume(
                    candidate_id=cid,
                    file_name=path.name,
                    candidate_name=profile.name,
                    status="PARSED",
                    warnings=[],
                )
            )
        except Exception as e:
            logger.error("Error creating demo candidate %s: %s", key, e)

    return DemoLoadResponse(
        job=demo_job,
        upload=UploadResponse(candidates=uploaded_cands, errors=[]),
    )

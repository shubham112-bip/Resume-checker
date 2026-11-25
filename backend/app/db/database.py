"""SQLite database models and repository for long-term and episodic memory."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import RUNTIME_DIR, get_settings
from app.schemas.api import EpisodeResponse
from app.schemas.candidate import CandidateProfile
from app.schemas.evaluation import CandidateEvaluation, MemoryEpisode
from app.schemas.job import JobProfile
from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class JobRecord(Base):
    __tablename__ = "jobs"
    job_id = Column(String(64), primary_key=True)
    title = Column(String(255), nullable=False)
    profile_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))


class CandidateRecord(Base):
    __tablename__ = "candidates"
    candidate_id = Column(String(64), primary_key=True)
    name = Column(String(255), nullable=False)
    file_name = Column(String(255), nullable=False)
    profile_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))


class EvaluationRecord(Base):
    __tablename__ = "evaluations"
    evaluation_id = Column(String(64), primary_key=True)
    run_id = Column(String(64), index=True, nullable=False)
    candidate_id = Column(String(64), index=True, nullable=False)
    job_id = Column(String(64), index=True, nullable=False)
    candidate_name = Column(String(255), nullable=False)
    job_title = Column(String(255), nullable=False)
    overall_score = Column(Float, nullable=False)
    recommendation = Column(String(64), nullable=False)
    recruiter_decision = Column(String(64), nullable=True)
    recruiter_notes = Column(Text, nullable=True)
    recruiter_decided_at = Column(String(64), nullable=True)
    data_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))


class EpisodicEventRecord(Base):
    __tablename__ = "episodic_events"
    id = Column(Integer, primary_key=True, autoincrement=True)
    candidate_id = Column(String(64), index=True, nullable=False)
    candidate_name = Column(String(255), nullable=False)
    job_id = Column(String(64), index=True, nullable=False)
    job_title = Column(String(255), nullable=False)
    event_type = Column(String(64), nullable=False)
    summary = Column(Text, nullable=False)
    overall_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))


_engine = None
_SessionFactory = None


def init_db() -> sessionmaker[Session]:
    global _engine, _SessionFactory
    if _SessionFactory is not None:
        return _SessionFactory

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    db_url = get_settings().database_url
    _engine = create_engine(db_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(_engine)
    with _engine.connect() as conn:
        for col, col_type in [
            ("recruiter_decision", "VARCHAR(64)"),
            ("recruiter_notes", "TEXT"),
            ("recruiter_decided_at", "VARCHAR(64)"),
        ]:
            try:
                conn.execute(text(f"ALTER TABLE evaluations ADD COLUMN {col} {col_type}"))
                conn.commit()
            except Exception:
                pass
    _SessionFactory = sessionmaker(bind=_engine, autocommit=False, autoflush=False)
    return _SessionFactory


def get_db_session() -> Session:
    factory = init_db()
    return factory()


# Database repository helpers
def save_job(job: JobProfile) -> None:
    session = get_db_session()
    try:
        rec = JobRecord(
            job_id=job.job_id,
            title=job.title,
            profile_json=job.model_dump_json(by_alias=True),
        )
        session.merge(rec)
        session.commit()
    finally:
        session.close()


def get_job_by_id(job_id: str) -> JobProfile | None:
    session = get_db_session()
    try:
        rec = session.query(JobRecord).filter(JobRecord.job_id == job_id).first()
        if not rec:
            return None
        return JobProfile.model_validate(json.loads(rec.profile_json))
    finally:
        session.close()


def list_jobs() -> list[JobProfile]:
    session = get_db_session()
    try:
        records = session.query(JobRecord).order_by(JobRecord.created_at.desc()).all()
        return [JobProfile.model_validate(json.loads(r.profile_json)) for r in records]
    finally:
        session.close()


def save_candidate(profile: CandidateProfile, file_name: str) -> None:
    session = get_db_session()
    try:
        rec = CandidateRecord(
            candidate_id=profile.candidate_id,
            name=profile.name,
            file_name=file_name,
            profile_json=profile.model_dump_json(by_alias=True),
        )
        session.merge(rec)
        session.commit()
    finally:
        session.close()


def get_candidate_by_id(candidate_id: str) -> tuple[CandidateProfile | None, str]:
    session = get_db_session()
    try:
        rec = session.query(CandidateRecord).filter(CandidateRecord.candidate_id == candidate_id).first()
        if not rec:
            return None, ""
        return CandidateProfile.model_validate(json.loads(rec.profile_json)), rec.file_name
    finally:
        session.close()


def list_candidates() -> list[tuple[CandidateProfile, str, datetime]]:
    session = get_db_session()
    try:
        records = session.query(CandidateRecord).order_by(CandidateRecord.created_at.desc()).all()
        return [
            (CandidateProfile.model_validate(json.loads(r.profile_json)), r.file_name, r.created_at)
            for r in records
        ]
    finally:
        session.close()


def save_evaluation(eval_res: CandidateEvaluation) -> None:
    session = get_db_session()
    try:
        rec = EvaluationRecord(
            evaluation_id=eval_res.evaluation_id,
            run_id=eval_res.run_id,
            candidate_id=eval_res.candidate_id,
            job_id=eval_res.job_id,
            candidate_name=eval_res.candidate_name,
            job_title=eval_res.job_title,
            overall_score=eval_res.overall_score,
            recommendation=eval_res.recommendation.value,
            recruiter_decision=eval_res.recruiter_decision,
            recruiter_notes=eval_res.recruiter_notes,
            recruiter_decided_at=eval_res.recruiter_decided_at,
            data_json=eval_res.model_dump_json(by_alias=True),
        )
        session.merge(rec)

        # Also store in episodic memory
        ep = EpisodicEventRecord(
            candidate_id=eval_res.candidate_id,
            candidate_name=eval_res.candidate_name,
            job_id=eval_res.job_id,
            job_title=eval_res.job_title,
            event_type="EVALUATION_COMPLETED",
            summary=(
                f"Candidate {eval_res.candidate_name} evaluated for '{eval_res.job_title}'. "
                f"Overall Score: {eval_res.overall_score}/100, Recommendation: {eval_res.recommendation_label}."
            ),
            overall_score=eval_res.overall_score,
        )
        session.add(ep)
        session.commit()
    finally:
        session.close()


def get_evaluation_by_id(evaluation_id: str) -> CandidateEvaluation | None:
    session = get_db_session()
    try:
        rec = session.query(EvaluationRecord).filter(EvaluationRecord.evaluation_id == evaluation_id).first()
        if not rec:
            return None
        data = json.loads(rec.data_json)
        if rec.recruiter_decision:
            data["recruiterDecision"] = rec.recruiter_decision
            data["recruiterNotes"] = rec.recruiter_notes or ""
            data["recruiterDecidedAt"] = rec.recruiter_decided_at or ""
        return CandidateEvaluation.model_validate(data)
    finally:
        session.close()


def update_recruiter_decision(evaluation_id: str, decision: str, notes: str) -> CandidateEvaluation | None:
    session = get_db_session()
    try:
        rec = session.query(EvaluationRecord).filter(EvaluationRecord.evaluation_id == evaluation_id).first()
        if not rec:
            return None
        now_str = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        rec.recruiter_decision = decision
        rec.recruiter_notes = notes
        rec.recruiter_decided_at = now_str

        data = json.loads(rec.data_json)
        data["recruiterDecision"] = decision
        data["recruiterNotes"] = notes
        data["recruiterDecidedAt"] = now_str
        rec.data_json = json.dumps(data)

        # Log to episodic events memory
        ep = EpisodicEventRecord(
            candidate_id=rec.candidate_id,
            candidate_name=rec.candidate_name,
            job_id=rec.job_id,
            job_title=rec.job_title,
            event_type="RECRUITER_DETERMINATION",
            summary=(
                f"Recruiter set determination to '{decision}' for candidate {rec.candidate_name} ({rec.job_title}). "
                f"Notes: {notes.strip() if notes.strip() else 'No notes added.'}"
            ),
            overall_score=rec.overall_score,
        )
        session.add(ep)
        session.commit()
        return CandidateEvaluation.model_validate(data)
    finally:
        session.close()


def list_evaluations_for_run(run_id: str) -> list[CandidateEvaluation]:
    session = get_db_session()
    try:
        records = (
            session.query(EvaluationRecord)
            .filter(EvaluationRecord.run_id == run_id)
            .order_by(EvaluationRecord.overall_score.desc())
            .all()
        )
        return [CandidateEvaluation.model_validate(json.loads(r.data_json)) for r in records]
    finally:
        session.close()


def recall_candidate_memory(candidate_name: str, exclude_run_id: str | None = None) -> list[MemoryEpisode]:
    """Retrieve prior evaluation memory episodes for a candidate across past jobs."""
    session = get_db_session()
    try:
        q = session.query(EvaluationRecord).filter(EvaluationRecord.candidate_name.ilike(f"%{candidate_name}%"))
        if exclude_run_id:
            q = q.filter(EvaluationRecord.run_id != exclude_run_id)
        records = q.order_by(EvaluationRecord.created_at.desc()).limit(5).all()

        episodes: list[MemoryEpisode] = []
        for r in records:
            episodes.append(
                MemoryEpisode(
                    candidate_name=r.candidate_name,
                    job_title=r.job_title,
                    overall_score=r.overall_score,
                    recommendation=r.recommendation,
                    created_at=r.created_at,
                )
            )
        return episodes
    finally:
        session.close()


def list_all_episodes() -> list[EpisodeResponse]:
    session = get_db_session()
    try:
        records = session.query(EpisodicEventRecord).order_by(EpisodicEventRecord.created_at.desc()).limit(50).all()
        return [
            EpisodeResponse(
                episode_id=r.id,
                candidate_id=r.candidate_id,
                candidate_name=r.candidate_name,
                job_id=r.job_id,
                job_title=r.job_title,
                event_type=r.event_type,
                summary=r.summary,
                overall_score=r.overall_score,
                created_at=r.created_at,
            )
            for r in records
        ]
    finally:
        session.close()

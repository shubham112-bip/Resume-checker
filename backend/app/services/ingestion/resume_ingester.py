"""Real Resume Ingestion Pipeline.

Parses PDF, DOCX, and TXT resumes into normalized structured JSON profiles
with embedded resume quality evaluations, saved to data/processed/resumes/.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from app.config import PROCESSED_DIR, RESUMES_DIR
from app.schemas.candidate import CandidateProfile
from app.services.parsing.resume_parser import parse_resume
from app.services.quality.resume_quality import evaluate_resume_quality

logger = logging.getLogger(__name__)


def ingest_single_resume(
    file_path: Path,
    resume_id: str | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Parse resume and produce normalized JSON schema specified in Section 2."""
    rid = resume_id or f"resume_{file_path.stem}"
    out_dir = output_dir or (PROCESSED_DIR / "resumes")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parse text and extract structured profile
    profile = parse_resume(file_path, candidate_id=rid)

    # 2. Compute standalone resume quality metrics
    quality = evaluate_resume_quality(profile)

    # 3. Format sections as dict: {canonical_name: section_text}
    sections_dict = {s.name: s.text for s in profile.sections}

    # Extract spoken languages if present in text
    languages: list[str] = []
    lang_match = re.search(r"Languages?[:\s]+([^\n\r.]+)", profile.text, re.IGNORECASE)
    if lang_match:
        raw_langs = re.split(r"[,;/|•\t]+", lang_match.group(1))
        languages = [l.strip() for l in raw_langs if l.strip() and len(l.strip()) < 30]

    # 4. Form normalized JSON record (supporting both snake_case and camelCase)
    metadata: dict[str, Any] = {
        "fileName": file_path.name,
        "candidateName": profile.name,
        "fileType": file_path.suffix.upper().replace(".", ""),
        "totalExperienceMonths": profile.total_experience_months,
        "experienceYears": round(profile.total_experience_months / 12.0, 1),
        "diagnostics": profile.diagnostics.model_dump(by_alias=True) if profile.diagnostics else None,
        "sectionsAvailable": list(sections_dict.keys()),
    }

    normalized: dict[str, Any] = {
        # Canonical schema per Section 2
        "resume_id": rid,
        "resumeId": rid,
        "fileName": file_path.name,
        "candidateName": profile.name,
        "skills": profile.skills,
        "education": [e.model_dump(by_alias=True) for e in profile.education],
        "experience": [e.model_dump(by_alias=True) for e in profile.experience],
        "projects": [p.model_dump(by_alias=True) for p in profile.projects],
        "certifications": profile.certifications,
        "achievements": profile.achievements,
        "languages": languages,
        "raw_text": profile.text,
        "contact": profile.contact.model_dump(by_alias=True),
        "summary": profile.summary,
        "totalExperienceMonths": profile.total_experience_months,
        "experienceYears": round(profile.total_experience_months / 12.0, 1),
        "sections": sections_dict,
        "metadata": metadata,
        "sensitiveAttributesRemoved": [
            s.model_dump(by_alias=True) for s in profile.sensitive_attributes_removed
        ],
        "sensitive_attributes_removed": [
            s.model_dump() for s in profile.sensitive_attributes_removed
        ],
        "qualityFeatures": {
            "overallQualityScore": quality.overall_quality_score,
            "qualityGrade": quality.quality_grade,
            "completenessScore": quality.completeness_score,
            "atsReadabilityScore": quality.ats_readability_score,
            "structureScore": quality.structure_score,
            "actionVerbScore": quality.action_verb_score,
            "quantificationScore": quality.quantification_score,
            "clarityScore": quality.clarity_score,
            "formattingConsistencyScore": quality.formatting_consistency_score,
            "actionVerbsFound": quality.action_verbs_found,
            "quantifiedBulletsCount": quality.quantified_bullets_count,
            "totalBulletsCount": quality.total_bullets_count,
            "strengths": quality.strengths,
            "improvementAreas": quality.improvement_areas,
            "verdict": quality.verdict,
        },
    }

    # Write normalized JSON
    dest_path = out_dir / f"{rid}.json"
    dest_path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    logger.info("Ingested resume %s -> %s", file_path.name, dest_path)
    return normalized


def ingest_resume_directory(
    source_dir: Path | None = None,
    output_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Batch ingest all 65 uploaded resumes and sample reference resumes."""
    s_dir = source_dir or RESUMES_DIR
    out_dir = output_dir or (PROCESSED_DIR / "resumes")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not s_dir.exists():
        logger.warning("Resume directory %s does not exist", s_dir)
        return []

    results: list[dict[str, Any]] = []
    supported_exts = {".pdf", ".docx", ".txt"}

    files = [f for f in s_dir.iterdir() if f.is_file() and f.suffix.lower() in supported_exts]

    # Prioritize numbered uploaded files (1.docx .. 65.docx)
    numbered_files = sorted(
        [f for f in files if f.name.endswith(".docx") and f.stem.isdigit()],
        key=lambda x: int(x.stem),
    )
    other_files = sorted([f for f in files if f not in numbered_files], key=lambda x: x.name)

    # Ingest 65 uploaded real resumes
    for f in numbered_files:
        try:
            rid = f"resume_{int(f.stem):03d}"
            record = ingest_single_resume(f, resume_id=rid, output_dir=out_dir)
            results.append(record)
        except Exception as e:
            logger.error("Failed to ingest %s: %s", f.name, e)

    # Ingest sample reference resumes
    for f in other_files:
        try:
            rid = f"sample_{f.stem}"
            record = ingest_single_resume(f, resume_id=rid, output_dir=out_dir)
            results.append(record)
        except Exception as e:
            logger.error("Failed to ingest sample %s: %s", f.name, e)

    logger.info("Ingested %d resumes from %s", len(results), s_dir)
    return results

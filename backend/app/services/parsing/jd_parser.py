"""Job description parsing service that extracts structured job profiles."""

from __future__ import annotations

import json
import logging
import re
import uuid
from pathlib import Path

from app.config import get_settings
from app.schemas.job import JobProfile, JobSkill, SkillTier
from app.services.fairness import redact_sensitive_attributes
from app.services.parsing.document_loader import load_document
from app.services.parsing.education import find_degrees
from app.services.parsing.sections import match_heading
from app.services.skills.extractor import extract_skills, skill_category

logger = logging.getLogger(__name__)

_EXP_RANGE_REGEX = re.compile(
    r"(?P<min>\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(?P<max>\d+(?:\.\d+)?)\s*(?:\+)?\s*(?:years?|yrs?)",
    re.IGNORECASE,
)
_EXP_MIN_REGEX = re.compile(
    r"(?:at\s+least|minimum|min|more\s+than|\b)?\s*(?P<min>\d+(?:\.\d+)?)\s*(?:\+)?\s*(?:years?|yrs?)(?:\s+of)?(?:\s+experience)?",
    re.IGNORECASE,
)


def extract_job_title(text: str) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return "Software Engineer"
    first = lines[0]
    # Check for "Job Title:", "Role:", "Position:"
    if ":" in first:
        head, _, rest = first.partition(":")
        if head.lower() in {"job title", "role", "position", "title"}:
            return rest.strip()
    if len(first.split()) <= 6 and not any(ch in first for ch in "@|;"):
        return first.strip("#* ")
    return "Software Engineer"


def extract_experience_requirements(text: str) -> tuple[float, float | None]:
    range_match = _EXP_RANGE_REGEX.search(text)
    if range_match:
        return float(range_match.group("min")), float(range_match.group("max"))
    min_match = _EXP_MIN_REGEX.search(text)
    if min_match:
        val = float(min_match.group("min"))
        if val <= 30:  # avoid years like 2024
            return val, None
    return 0.0, None


def split_jd_sections(text: str) -> dict[str, str]:
    lines = text.splitlines()
    sections: dict[str, list[str]] = {}
    current = "intro"
    sections[current] = []

    for line in lines:
        m = match_heading(line, kind="jd")
        if m:
            current = m.canonical
            if current not in sections:
                sections[current] = []
            if m.remainder:
                sections[current].append(m.remainder)
        else:
            if current not in sections:
                sections[current] = []
            sections[current].append(line)

    return {k: "\n".join(v).strip() for k, v in sections.items() if v}


def parse_job_description(
    text: str, job_id: str | None = None, title: str | None = None
) -> JobProfile:
    jid = job_id or f"job_{uuid.uuid4().hex[:8]}"
    clean_text, _ = redact_sensitive_attributes(text)
    j_title = title or extract_job_title(clean_text)
    min_exp, max_exp = extract_experience_requirements(clean_text)

    sections = split_jd_sections(clean_text)
    req_text = sections.get("required", "")
    pref_text = sections.get("preferred", "")
    opt_text = sections.get("optional", "")
    resp_text = sections.get("responsibilities", "")
    edu_text = sections.get("education", "")

    req_skills = extract_skills(req_text)
    pref_skills = extract_skills(pref_text)
    opt_skills = extract_skills(opt_text)

    all_skills_found = extract_skills(clean_text)
    classified: list[JobSkill] = []
    seen: set[str] = set()

    for s in req_skills:
        if s not in seen:
            classified.append(
                JobSkill(name=s, tier=SkillTier.REQUIRED, category=skill_category(s))
            )
            seen.add(s)

    for s in pref_skills:
        if s not in seen:
            classified.append(
                JobSkill(name=s, tier=SkillTier.PREFERRED, category=skill_category(s))
            )
            seen.add(s)

    for s in opt_skills:
        if s not in seen:
            classified.append(
                JobSkill(name=s, tier=SkillTier.OPTIONAL, category=skill_category(s))
            )
            seen.add(s)

    # Any remaining technical skills in the text are classified as REQUIRED if no explicit required section,
    # otherwise PREFERRED.
    for s in all_skills_found:
        if s not in seen:
            tier = SkillTier.REQUIRED if not req_skills else SkillTier.PREFERRED
            classified.append(JobSkill(name=s, tier=tier, category=skill_category(s)))
            seen.add(s)

    # Extract education requirements
    degrees = find_degrees(edu_text if edu_text else clean_text)
    edu_reqs = [d.degree for d in degrees]
    max_degree_level = max((d.level for d in degrees), default=0)

    responsibilities = [
        l.strip("•- ") for l in resp_text.splitlines() if len(l.strip()) > 10
    ]

    soft_skills = [s.name for s in classified if s.category == "soft"]

    return JobProfile(
        job_id=jid,
        title=j_title,
        skills=classified,
        min_experience_years=min_exp,
        max_experience_years=max_exp,
        education_requirements=edu_reqs,
        required_degree_level=max_degree_level,
        responsibilities=responsibilities,
        tools=[s.name for s in classified if s.category == "tool"],
        soft_skills=soft_skills,
        extraction_method="rule-based",
        raw_text=clean_text,
    )


def parse_job_file(file_path: Path, job_id: str | None = None) -> JobProfile:
    loaded = load_document(file_path)
    return parse_job_description(loaded.text, job_id=job_id)

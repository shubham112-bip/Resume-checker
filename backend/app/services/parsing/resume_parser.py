"""Resume parsing service that extracts structured candidate profiles."""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any

from app.schemas.candidate import (
    CandidateProfile,
    ContactInfo,
    EducationEntry,
    ExperienceEntry,
    ProjectEntry,
    ResumeSection,
)
from app.services.fairness import redact_sensitive_attributes
from app.services.parsing.dates import find_date_ranges, merged_months
from app.services.parsing.document_loader import load_document
from app.services.parsing.education import find_degrees
from app.services.parsing.sections import looks_like_heading, match_heading
from app.services.skills.extractor import extract_skills

_EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
_PHONE_REGEX = re.compile(
    r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\b\d{10}\b"
)
_LINKEDIN_REGEX = re.compile(
    r"(?:https?://)?(?:www\.)?linkedin\.com/in/[A-Za-z0-9_-]+/?", re.IGNORECASE
)
_GITHUB_REGEX = re.compile(
    r"(?:https?://)?(?:www\.)?github\.com/[A-Za-z0-9_-]+/?", re.IGNORECASE
)
_PORTFOLIO_REGEX = re.compile(
    r"(?:https?://)?(?:www\.)?(?:[A-Za-z0-9_-]+\.(?:dev|me|io|in|com))/[A-Za-z0-9_/-]*",
    re.IGNORECASE,
)


def extract_contact_info(text: str) -> ContactInfo:
    email_match = _EMAIL_REGEX.search(text)
    phone_match = _PHONE_REGEX.search(text)
    linkedin_match = _LINKEDIN_REGEX.search(text)
    github_match = _GITHUB_REGEX.search(text)
    portfolio_match = _PORTFOLIO_REGEX.search(text)

    # Discard github/linkedin matches if mistakenly caught as portfolio
    portfolio = None
    if portfolio_match:
        cand = portfolio_match.group(0)
        if "github.com" not in cand and "linkedin.com" not in cand:
            portfolio = cand

    return ContactInfo(
        email=email_match.group(0) if email_match else None,
        phone=phone_match.group(0) if phone_match else None,
        linkedin=linkedin_match.group(0) if linkedin_match else None,
        github=github_match.group(0) if github_match else None,
        portfolio=portfolio,
    )


def extract_name(text: str, lines: list[str]) -> str:
    """Heuristic extraction of candidate name from the top header lines."""
    for line in lines[:8]:
        clean = line.strip()
        if not clean:
            continue
        # Avoid lines with email, phone, links, or headers
        if _EMAIL_REGEX.search(clean) or _PHONE_REGEX.search(clean):
            continue
        if "linkedin.com" in clean.lower() or "github.com" in clean.lower():
            continue
        if match_heading(clean, "resume"):
            break
        # Look for 1-4 title words
        words = clean.split()
        if 1 <= len(words) <= 4:
            if all(w[0].isupper() or w in {"de", "van", "von", "la"} for w in words if w.isalpha()):
                return clean
    return "Candidate"


def split_sections(text: str) -> tuple[list[ResumeSection], list[str]]:
    lines = text.splitlines()
    sections: list[ResumeSection] = []
    unrecognized: list[str] = []

    current_canonical = "header"
    current_heading = "Header"
    current_lines: list[str] = []
    start_line = 0

    for idx, line in enumerate(lines):
        m = match_heading(line, kind="resume")
        if m:
            if current_lines:
                sections.append(
                    ResumeSection(
                        name=current_canonical,
                        heading=current_heading,
                        text="\n".join(current_lines).strip(),
                        start_line=start_line,
                    )
                )
                current_lines = []
            current_canonical = m.canonical
            current_heading = m.heading
            start_line = idx
            if m.remainder:
                current_lines.append(m.remainder)
        elif looks_like_heading(line):
            unrecognized.append(line.strip())
            current_lines.append(line)
        else:
            current_lines.append(line)

    if current_lines:
        sections.append(
            ResumeSection(
                name=current_canonical,
                heading=current_heading,
                text="\n".join(current_lines).strip(),
                start_line=start_line,
            )
        )

    return sections, unrecognized


def parse_education(sec_text: str, full_text: str) -> list[EducationEntry]:
    text_to_search = sec_text if len(sec_text) > 20 else full_text
    matches = find_degrees(text_to_search)
    entries: list[EducationEntry] = []
    lines = text_to_search.splitlines()

    for m in matches:
        raw_line = ""
        for line in lines:
            if m.degree.lower() in line.lower():
                raw_line = line.strip()
                break
        entries.append(
            EducationEntry(
                degree=m.degree,
                degree_level=m.level,
                field=m.field,
                is_relevant_field=m.is_relevant_field,
                raw=raw_line or m.degree,
            )
        )
    return entries


def parse_experience(sec_text: str) -> tuple[list[ExperienceEntry], int]:
    if not sec_text.strip():
        return [], 0
    lines = sec_text.splitlines()
    entries: list[ExperienceEntry] = []
    current_title = ""
    current_org = ""
    current_bullets: list[str] = []
    current_range: Any = None

    date_ranges = find_date_ranges(sec_text)
    total_months = merged_months(date_ranges)

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        line_ranges = find_date_ranges(stripped)
        if line_ranges:
            if current_title or current_bullets:
                entries.append(
                    ExperienceEntry(
                        title=current_title or "Software Engineer",
                        organization=current_org,
                        start=current_range.start_text if current_range else None,
                        end=current_range.end_text if current_range else None,
                        duration_months=current_range.months if current_range else 0,
                        is_current=current_range.is_current if current_range else False,
                        bullets=current_bullets,
                    )
                )
                current_bullets = []
            current_range = line_ranges[0]
            # Try to grab title/org from the rest of the line or previous line
            current_title = stripped.replace(current_range.start_text, "").replace(current_range.end_text, "").strip(" -–—|")
            current_org = ""
        elif stripped.startswith("•") or stripped.startswith("-"):
            current_bullets.append(stripped.lstrip("•- \t"))
        else:
            if not current_title:
                current_title = stripped
            elif not current_org:
                current_org = stripped
            else:
                current_bullets.append(stripped)

    if current_title or current_bullets:
        entries.append(
            ExperienceEntry(
                title=current_title or "Experience Entry",
                organization=current_org,
                start=current_range.start_text if current_range else None,
                end=current_range.end_text if current_range else None,
                duration_months=current_range.months if current_range else 0,
                is_current=current_range.is_current if current_range else False,
                bullets=current_bullets,
            )
        )

    return entries, total_months


def parse_projects(sec_text: str) -> list[ProjectEntry]:
    if not sec_text.strip():
        return []
    lines = sec_text.splitlines()
    projects: list[ProjectEntry] = []
    current_name = ""
    current_desc: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("•") or stripped.startswith("-"):
            current_desc.append(stripped.lstrip("•- \t"))
        elif ":" in stripped and len(stripped.split(":")[0].split()) <= 4:
            if current_name:
                projects.append(
                    ProjectEntry(
                        name=current_name,
                        description=" ".join(current_desc),
                        technologies=extract_skills(current_name + " " + " ".join(current_desc)),
                    )
                )
                current_desc = []
            name, _, rest = stripped.partition(":")
            current_name = name.strip()
            if rest.strip():
                current_desc.append(rest.strip())
        else:
            if not current_name:
                current_name = stripped
            else:
                current_desc.append(stripped)

    if current_name:
        projects.append(
            ProjectEntry(
                name=current_name,
                description=" ".join(current_desc),
                technologies=extract_skills(current_name + " " + " ".join(current_desc)),
            )
        )

    return projects


def parse_resume(
    file_path: Path, candidate_id: str | None = None, strategy: str = "primary"
) -> CandidateProfile:
    cid = candidate_id or f"cand_{uuid.uuid4().hex[:8]}"
    loaded = load_document(file_path, strategy=strategy)
    raw_text = loaded.text
    diagnostics = loaded.diagnostics

    # Fairness step: redact sensitive attributes before any downstream processing
    clean_text, findings = redact_sensitive_attributes(
        raw_text, image_count=diagnostics.image_count
    )

    contact = extract_contact_info(clean_text)
    sections, unrecognized = split_sections(clean_text)
    lines = clean_text.splitlines()
    name = extract_name(clean_text, lines)

    sec_dict = {s.name: s.text for s in sections}
    summary = sec_dict.get("summary", "")
    education = parse_education(sec_dict.get("education", ""), clean_text)
    experience, exp_months = parse_experience(sec_dict.get("experience", ""))
    projects = parse_projects(sec_dict.get("projects", ""))

    skills_text = sec_dict.get("skills", "")
    all_skills = extract_skills(clean_text)

    certifications: list[str] = []
    if "certifications" in sec_dict:
        certifications = [
            l.strip("•- ") for l in sec_dict["certifications"].splitlines() if l.strip()
        ]

    achievements: list[str] = []
    if "achievements" in sec_dict:
        achievements = [
            l.strip("•- ") for l in sec_dict["achievements"].splitlines() if l.strip()
        ]

    return CandidateProfile(
        candidate_id=cid,
        name=name,
        contact=contact,
        summary=summary,
        education=education,
        skills=all_skills,
        experience=experience,
        projects=projects,
        certifications=certifications,
        achievements=achievements,
        sections=sections,
        unrecognized_headings=unrecognized,
        total_experience_months=exp_months,
        diagnostics=diagnostics,
        sensitive_attributes_removed=findings,
        text=clean_text,
    )


def parse_resume_text(
    text: str, candidate_id: str | None = None, name: str | None = None
) -> CandidateProfile:
    """Parse raw resume text directly into CandidateProfile without requiring a file."""
    cid = candidate_id or f"cand_{uuid.uuid4().hex[:8]}"
    clean_text, findings = redact_sensitive_attributes(text)

    contact = extract_contact_info(clean_text)
    sections, unrecognized = split_sections(clean_text)
    lines = clean_text.splitlines()
    extracted_name = name or extract_name(clean_text, lines)

    sec_dict = {s.name: s.text for s in sections}
    summary = sec_dict.get("summary", "")
    education = parse_education(sec_dict.get("education", ""), clean_text)
    experience, exp_months = parse_experience(sec_dict.get("experience", ""))
    projects = parse_projects(sec_dict.get("projects", ""))
    all_skills = extract_skills(clean_text)

    certifications = [
        l.strip("•- ") for l in sec_dict.get("certifications", "").splitlines() if l.strip()
    ]
    achievements = [
        l.strip("•- ") for l in sec_dict.get("achievements", "").splitlines() if l.strip()
    ]

    return CandidateProfile(
        candidate_id=cid,
        name=extracted_name or "Candidate",
        contact=contact,
        summary=summary,
        education=education,
        skills=all_skills,
        experience=experience,
        projects=projects,
        certifications=certifications,
        achievements=achievements,
        sections=sections,
        unrecognized_headings=unrecognized,
        total_experience_months=exp_months,
        sensitive_attributes_removed=findings,
        text=clean_text,
    )

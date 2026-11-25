"""Traditional ATS check engine and keyword stuffing detector."""

from __future__ import annotations

import re
from collections import Counter

from app.config import get_settings
from app.schemas.candidate import CandidateProfile
from app.schemas.evaluation import ATSCheck, ATSResult, StuffedTerm, StuffingResult
from app.schemas.job import JobProfile, SkillTier
from app.services.skills.extractor import count_literal_mentions, mentions_skill


def evaluate_ats(profile: CandidateProfile, job: JobProfile) -> ATSResult:
    cfg = get_settings().ats_weights
    checks: list[ATSCheck] = []
    issues: list[str] = []

    # 1. Contact Information checks
    contact = profile.contact
    has_email = bool(contact.email)
    has_phone = bool(contact.phone)
    has_linkedin = bool(contact.linkedin)
    has_github = bool(contact.github)

    checks.append(ATSCheck(name="Email provided", category="contact", passed=has_email))
    checks.append(ATSCheck(name="Phone number provided", category="contact", passed=has_phone))
    checks.append(ATSCheck(name="LinkedIn profile included", category="contact", passed=has_linkedin))
    checks.append(ATSCheck(name="GitHub/Portfolio included", category="contact", passed=has_github or bool(contact.portfolio)))

    contact_score = (
        (0.4 if has_email else 0.0)
        + (0.3 if has_phone else 0.0)
        + (0.15 if has_linkedin else 0.0)
        + (0.15 if (has_github or contact.portfolio) else 0.0)
    ) * 100.0

    if not has_email:
        issues.append("Missing email address in contact information.")
    if not has_phone:
        issues.append("Missing phone number in contact information.")

    # 2. Resume Sections checks
    sec_names = {s.name for s in profile.sections}
    has_summary = "summary" in sec_names
    has_education = "education" in sec_names or len(profile.education) > 0
    has_experience = "experience" in sec_names or len(profile.experience) > 0
    has_skills = "skills" in sec_names or len(profile.skills) > 0
    has_projects = "projects" in sec_names or len(profile.projects) > 0
    has_certifications = "certifications" in sec_names or len(profile.certifications) > 0

    checks.append(ATSCheck(name="Summary / Objective section", category="sections", passed=has_summary))
    checks.append(ATSCheck(name="Education section", category="sections", passed=has_education))
    checks.append(ATSCheck(name="Experience section", category="sections", passed=has_experience))
    checks.append(ATSCheck(name="Skills section", category="sections", passed=has_skills))
    checks.append(ATSCheck(name="Projects section", category="sections", passed=has_projects))
    checks.append(ATSCheck(name="Certifications section", category="sections", passed=has_certifications))

    section_score = (
        (0.25 if has_experience else 0.0)
        + (0.25 if has_skills else 0.0)
        + (0.20 if has_education else 0.0)
        + (0.15 if has_projects else 0.0)
        + (0.10 if has_summary else 0.0)
        + (0.05 if has_certifications else 0.0)
    ) * 100.0

    if not has_experience:
        issues.append("No standard Experience/Work History section detected.")
    if not has_skills:
        issues.append("No dedicated Skills section detected.")

    # 3. Formatting checks
    diag = profile.diagnostics
    formatting_passed = True
    formatting_deductions = 0.0

    if diag:
        if diag.table_count > 4:
            issues.append(f"Excessive tables detected ({diag.table_count}). Many ATS parsers mishandle multi-column tables.")
            formatting_deductions += 20.0
            formatting_passed = False
        if diag.image_count > 3:
            issues.append(f"Excessive embedded images detected ({diag.image_count}). Visual elements cannot be indexed by ATS.")
            formatting_deductions += 15.0
            formatting_passed = False
        if diag.char_count < 200:
            issues.append("Resume contains very little readable text.")
            formatting_deductions += 40.0
            formatting_passed = False

    if profile.unrecognized_headings:
        issues.append(f"Non-standard headings found: {', '.join(profile.unrecognized_headings[:3])}")
        formatting_deductions += 10.0

    formatting_score = max(0.0, 100.0 - formatting_deductions)
    checks.append(
        ATSCheck(
            name="ATS-friendly formatting",
            category="formatting",
            passed=formatting_passed,
            detail=f"Deductions: {formatting_deductions} pts",
        )
    )

    # 4. Keyword Coverage (Traditional Literal Keyword Matching)
    matched_keywords: list[str] = []
    missing_keywords: list[str] = []
    weighted_matched = 0.0
    weighted_total = 0.0

    tier_weights = get_settings().tier_weights

    for skill in job.skills:
        weight = getattr(tier_weights, skill.tier.value, 1.0)
        weighted_total += weight
        # Pure keyword check: literal mentions in text
        if count_literal_mentions(skill.name, profile.text) > 0 or mentions_skill(skill.name, profile.text):
            matched_keywords.append(skill.name)
            weighted_matched += weight
        else:
            missing_keywords.append(skill.name)

    keyword_coverage = (weighted_matched / weighted_total * 100.0) if weighted_total > 0 else 100.0
    checks.append(
        ATSCheck(
            name="Job keyword coverage",
            category="keywords",
            passed=keyword_coverage >= 50.0,
            detail=f"{len(matched_keywords)} of {len(job.skills)} terms matched ({keyword_coverage:.1f}%)",
        )
    )

    # Final ATS score
    overall_ats = (
        cfg.contact * contact_score
        + cfg.sections * section_score
        + cfg.formatting * formatting_score
        + cfg.keywordCoverage * keyword_coverage
    )

    return ATSResult(
        score=round(overall_ats, 1),
        contact_score=round(contact_score, 1),
        section_score=round(section_score, 1),
        keyword_coverage=round(keyword_coverage, 1),
        formatting_score=round(formatting_score, 1),
        checks=checks,
        issues=issues,
        matched_keywords=matched_keywords,
        missing_keywords=missing_keywords,
    )


def detect_keyword_stuffing(profile: CandidateProfile, job: JobProfile) -> StuffingResult:
    """Detect keyword stuffing by comparing repeated mentions against supporting evidence."""
    cfg = get_settings().stuffing
    evidence_text = profile.section_text("experience") + "\n" + profile.section_text("projects")
    all_text = profile.text

    words = re.findall(r"\w+", all_text.lower())
    total_tokens = len(words)
    if total_tokens == 0:
        return StuffingResult()

    flagged_terms: list[StuffedTerm] = []
    total_jd_term_tokens = 0

    for skill in job.skills:
        all_count = count_literal_mentions(skill.name, all_text)
        ev_count = count_literal_mentions(skill.name, evidence_text)
        skill_token_len = len(skill.name.split())
        total_jd_term_tokens += all_count * skill_token_len

        # Flag if heavily repeated but lacking in experience/projects
        if all_count >= cfg.repeatThreshold:
            ev_ratio = ev_count / all_count if all_count > 0 else 0.0
            if ev_ratio < cfg.maxEvidenceRatio:
                flagged_terms.append(
                    StuffedTerm(
                        term=skill.name,
                        mentions=all_count,
                        evidence_mentions=ev_count,
                    )
                )

    density = total_jd_term_tokens / total_tokens if total_tokens > 0 else 0.0
    suspected = len(flagged_terms) > 0 or density >= cfg.densityThreshold

    detail = ""
    if suspected:
        detail = (
            f"Candidate exhibits repetitive keyword patterns: {len(flagged_terms)} skill(s) "
            f"repeated >= {cfg.repeatThreshold} times with < {int(cfg.maxEvidenceRatio*100)}% "
            f"evidence in experience/projects (density: {density*100:.1f}%)."
        )

    return StuffingResult(
        suspected=suspected,
        keyword_density=round(density, 3),
        flagged_terms=flagged_terms,
        detail=detail,
    )

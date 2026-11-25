"""Standalone Resume Quality Evaluation Engine.

Evaluates resume construction, readability, quantification, action verbs, and structure
INDEPENDENTLY of any specific Job Description.

Key Principle:
A high-quality resume (well-formatted, quantified) can have low Job Fit for a specific JD.
A plain-text resume can have high Job Fit if skills and experience match the JD perfectly.
"""

from __future__ import annotations

import re
from typing import Any

from app.schemas.candidate import CandidateProfile
from app.schemas.quality import QualityDimension, ResumeQualityEvaluation

# Curated list of strong resume action verbs
ACTION_VERBS = frozenset(
    {
        "achieved", "analyzed", "architected", "automated", "built", "collaborated",
        "configured", "constructed", "coordinated", "created", "debugged", "decreased",
        "delivered", "deployed", "designed", "developed", "devised", "directed",
        "engineered", "enhanced", "established", "executed", "expanded", "formulated",
        "generated", "implemented", "improved", "increased", "initiated", "integrated",
        "introduced", "investigated", "launched", "led", "managed", "maximized",
        "mentored", "migrated", "minimized", "modeled", "modernized", "monitored",
        "negotiated", "optimized", "orchestrated", "organized", "overhauled", "pioneered",
        "planned", "produced", "programmed", "reduced", "refactored", "resolved",
        "revamped", "scaled", "solved", "spearheaded", "standardized", "streamlined",
        "supervised", "tested", "trained", "transformed", "upgraded", "validated",
    }
)

# Quantification patterns (percentages, metrics, multipliers, money, quantities)
QUANTIFICATION_REGEX = re.compile(
    r"\b(?:\d+(?:\.\d+)?%|\$\d+(?:\.\d+)?(?:k|m|b)?|\d+(?:\.\d+)?(?:k|m|b)?\b|\d+x\b|"
    r"(?:reduced|increased|improved|boosted|grew|cut)\s+(?:by\s+)?\d+|(?:ms|sec|seconds|minutes|hours|days|users|queries|req/s))",
    re.IGNORECASE,
)


def evaluate_resume_quality(profile: CandidateProfile) -> ResumeQualityEvaluation:
    text = profile.text
    words = re.findall(r"\b[A-Za-z0-9+#.-]+\b", text.lower())
    word_count = len(words)

    diag = profile.diagnostics
    contact = profile.contact

    # 1. Section Completeness (Weight: 20%)
    sec_names = {s.name for s in profile.sections}
    has_contact = bool(contact.email and contact.phone)
    has_summary = "summary" in sec_names
    has_education = len(profile.education) > 0 or "education" in sec_names
    has_experience = len(profile.experience) > 0 or "experience" in sec_names
    has_skills = len(profile.skills) > 0 or "skills" in sec_names
    has_projects = len(profile.projects) > 0 or "projects" in sec_names
    has_certs = len(profile.certifications) > 0 or "certifications" in sec_names

    core_sections = [has_contact, has_education, has_experience, has_skills, has_projects]
    comp_score = (sum(1 for c in core_sections if c) / len(core_sections)) * 0.85
    if has_summary:
        comp_score += 0.08
    if has_certs:
        comp_score += 0.07
    comp_score = min(1.0, max(0.2, comp_score))

    # 2. ATS Readability (Weight: 20%)
    read_deductions = 0.0
    if diag:
        if diag.table_count > 3:
            read_deductions += 0.20
        if diag.image_count > 2:
            read_deductions += 0.15
        if diag.is_scanned:
            read_deductions += 0.50
    if not contact.email:
        read_deductions += 0.20
    if not contact.phone:
        read_deductions += 0.15
    if profile.unrecognized_headings:
        read_deductions += min(0.15, len(profile.unrecognized_headings) * 0.05)
    read_score = max(0.1, 1.0 - read_deductions)

    # 3. Professional Structure & Length (Weight: 15%)
    # Ideal single/two-page resume has 300 to 1100 words
    if 350 <= word_count <= 950:
        struct_score = 1.0
    elif 250 <= word_count <= 1300:
        struct_score = 0.85
    elif 150 <= word_count < 250:
        struct_score = 0.60
    elif word_count > 1300:
        struct_score = 0.70  # overly verbose
    else:
        struct_score = 0.35  # too sparse

    # 4. Action Verbs (Weight: 15%)
    exp_and_proj_bullets: list[str] = []
    for exp in profile.experience:
        exp_and_proj_bullets.extend(exp.bullets)
    for proj in profile.projects:
        if proj.description:
            exp_and_proj_bullets.append(proj.description)

    # If no structured experience bullets were parsed, fall back to lines from raw text
    if not exp_and_proj_bullets:
        for line in text.split("\n"):
            line = line.strip()
            if len(line.split()) >= 3:
                exp_and_proj_bullets.append(line)

    verbs_found: set[str] = set()
    for b in exp_and_proj_bullets:
        bullet_words = re.findall(r"\b[A-Za-z]+\b", b.lower())
        for w in bullet_words:
            if w in ACTION_VERBS:
                verbs_found.add(w)

    verb_count = len(verbs_found)
    if verb_count >= 8:
        verb_score = 1.0
    elif verb_count >= 5:
        verb_score = 0.85
    elif verb_count >= 3:
        verb_score = 0.70
    elif verb_count >= 1:
        verb_score = 0.50
    else:
        verb_score = 0.30

    # 5. Quantification & Impact Metrics (Weight: 15%)
    quant_bullets = 0
    total_bullets = len(exp_and_proj_bullets)
    for b in exp_and_proj_bullets:
        if QUANTIFICATION_REGEX.search(b):
            quant_bullets += 1

    if total_bullets > 0:
        quant_ratio = quant_bullets / total_bullets
        if quant_ratio >= 0.40:
            quant_score = 1.0
        elif quant_ratio >= 0.25:
            quant_score = 0.85
        elif quant_ratio >= 0.10:
            quant_score = 0.70
        elif quant_bullets >= 1:
            quant_score = 0.55
        else:
            quant_score = 0.35
    else:
        # Check full text for any numbers/percentages
        full_matches = len(QUANTIFICATION_REGEX.findall(text))
        quant_score = min(0.8, 0.3 + 0.1 * min(full_matches, 5))

    # 6. Clarity & Formatting Consistency (Weight: 15%)
    clarity_deductions = 0.0
    # Check for excessive un-bulleted blocks
    paragraphs = text.split("\n\n")
    long_blocks = [p for p in paragraphs if len(p.split()) > 140]
    if long_blocks:
        clarity_deductions += 0.15

    # Check for date consistency in experience
    starts = [exp.start for exp in profile.experience if exp.start]
    if len(starts) >= 2:
        styles = set()
        for s in starts:
            if re.search(r"[A-Za-z]", s):
                styles.add("named")
            elif "/" in s or "-" in s:
                styles.add("numeric")
        if len(styles) > 1:
            clarity_deductions += 0.10  # mixed date formatting

    clarity_score = max(0.2, 1.0 - clarity_deductions)

    # Weighted Overall Resume Quality Score
    dimensions = [
        QualityDimension(
            name="completeness",
            label="Section Completeness",
            weight=0.20,
            score=round(comp_score, 3),
            value_points=round(comp_score * 100.0, 1),
            details=[f"{'✓' if c else '✗'} {s}" for c, s in zip(
                [has_contact, has_education, has_experience, has_skills, has_projects, has_summary, has_certs],
                ["Contact Info", "Education", "Experience", "Skills", "Projects", "Summary", "Certifications"]
            )],
        ),
        QualityDimension(
            name="ats_readability",
            label="ATS Readability & Layout",
            weight=0.20,
            score=round(read_score, 3),
            value_points=round(read_score * 100.0, 1),
            details=[
                f"Contact completeness: {'100%' if has_contact else 'Partial'}",
                f"Table count: {diag.table_count if diag else 0} (penalty if >3)",
                f"Embedded images: {diag.image_count if diag else 0}",
            ],
        ),
        QualityDimension(
            name="structure",
            label="Professional Structure & Length",
            weight=0.15,
            score=round(struct_score, 3),
            value_points=round(struct_score * 100.0, 1),
            details=[f"Word count: {word_count} words (optimal range: 350-950)"],
        ),
        QualityDimension(
            name="action_verbs",
            label="Action Verbs & Direct Phrasing",
            weight=0.15,
            score=round(verb_score, 3),
            value_points=round(verb_score * 100.0, 1),
            details=[f"Distinct action verbs found: {len(verbs_found)} ({', '.join(sorted(verbs_found)[:6])})"],
        ),
        QualityDimension(
            name="quantification",
            label="Achievement Quantification & Metrics",
            weight=0.15,
            score=round(quant_score, 3),
            value_points=round(quant_score * 100.0, 1),
            details=[f"Quantified bullet points: {quant_bullets} of {total_bullets} bullets ({int(quant_score*100)}%)"],
        ),
        QualityDimension(
            name="clarity_consistency",
            label="Formatting Consistency & Clarity",
            weight=0.15,
            score=round(clarity_score, 3),
            value_points=round(clarity_score * 100.0, 1),
            details=["Clean bulleted spacing", "Standard date formatting"],
        ),
    ]

    total_quality_points = sum(d.weight * d.value_points for d in dimensions)
    overall_score = round(total_quality_points, 1)

    if overall_score >= 85.0:
        grade = "Excellent"
    elif overall_score >= 70.0:
        grade = "Good"
    elif overall_score >= 50.0:
        grade = "Needs Improvement"
    else:
        grade = "Low Quality"

    strengths: list[str] = []
    improvements: list[str] = []

    if comp_score >= 0.85:
        strengths.append("Comprehensive section structure covering experience, education, skills, and projects.")
    else:
        improvements.append("Missing core sections (consider adding explicit Experience or Projects sections).")

    if verb_score >= 0.80:
        strengths.append(f"Strong action verbs utilized across accomplishments ({', '.join(sorted(verbs_found)[:5])}).")
    else:
        improvements.append("Incorporate more active engineering verbs (e.g. engineered, deployed, architected, optimized).")

    if quant_score >= 0.80:
        strengths.append("Accomplishments effectively quantify impact with measurable metrics and KPIs.")
    else:
        improvements.append("Add measurable outcomes to project bullets (e.g., latency reduction %, users scaled, efficiency gains).")

    if read_score >= 0.90:
        strengths.append("Clean, ATS-parseable plain formatting without obstructive visual tables or columns.")

    verdict = (
        f"Resume Quality Score: {overall_score}/100 ({grade}). "
        f"Independent of candidate-job fit, the resume demonstrates {grade.lower()} professional structure, "
        f"{'strong' if quant_score >= 0.7 else 'limited'} achievement quantification, and "
        f"{'high' if read_score >= 0.8 else 'moderate'} ATS readability."
    )

    return ResumeQualityEvaluation(
        overall_quality_score=overall_score,
        quality_grade=grade,
        completeness_score=round(comp_score, 3),
        ats_readability_score=round(read_score, 3),
        structure_score=round(struct_score, 3),
        action_verb_score=round(verb_score, 3),
        quantification_score=round(quant_score, 3),
        clarity_score=round(clarity_score, 3),
        formatting_consistency_score=round(clarity_score, 3),
        dimensions=dimensions,
        action_verbs_found=sorted(verbs_found),
        quantified_bullets_count=quant_bullets,
        total_bullets_count=total_bullets,
        strengths=strengths,
        improvement_areas=improvements,
        verdict=verdict,
    )

"""Candidate report generation service in Markdown and PDF format."""

from __future__ import annotations

import io
from pathlib import Path

from app.schemas.evaluation import CandidateEvaluation
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def generate_markdown_report(ev: CandidateEvaluation) -> str:
    md = f"""# Candidate Evaluation Report: {ev.candidate_name}

**Job Title:** {ev.job_title}  
**Evaluation ID:** `{ev.evaluation_id}`  
**Run ID:** `{ev.run_id}`  
**Evaluated At:** {ev.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Document:** `{ev.file_name}`  

---

## Executive Summary

- **Overall Candidate Fit Score:** **{ev.overall_score}/100**
- **Recommendation:** **{ev.recommendation_label}** (`{ev.recommendation.value}`)
- **ATS Compatibility Score:** {ev.ats_score:.1f}/100
- **Relevant Experience:** {ev.experience_years:.1f} years
- **Traditional Keyword ATS Score:** {ev.keyword_ats_score:.1f}/100
- **Human Review Flagged:** {'YES' if ev.needs_human_review else 'NO'}

{f"**Review Flags:** " + "; ".join(f.message for f in ev.review_flags) if ev.review_flags else ""}

---

## Transparent Score Breakdown

| Evaluation Dimension | Weight | Candidate Score | Contribution |
|---|---|---|---|
"""
    if ev.score_breakdown:
        for c in ev.score_breakdown.components:
            md += f"| {c.label} | {c.weight*100:.0f}% | {c.value:.1f}/100 | +{c.contribution:.1f} pts |\n"
        md += f"\n*Formula: {ev.score_breakdown.formula}*\n\n"

    md += f"""---

## Skill & Requirement Match

- **Matched Skills ({len(ev.matched_skills)}):** {', '.join(ev.matched_skills) if ev.matched_skills else 'None'}
- **Missing Skills ({len(ev.missing_skills)}):** {', '.join(ev.missing_skills) if ev.missing_skills else 'None'}
- **Uncertain / Inferred Skills ({len(ev.uncertain_skills)}):** {', '.join(ev.uncertain_skills) if ev.uncertain_skills else 'None'}

### Detailed Evidence Audit

"""
    for req in ev.requirement_matches:
        md += f"#### {req.requirement} ({req.tier.value})\n"
        md += f"- **Status:** `{req.status.value}` (Credit: {req.credit*100:.0f}%, Confidence: {req.confidence:.2f})\n"
        md += f"- **Note:** {req.note}\n"
        if req.evidence:
            md += f"- **Supporting Evidence:** *\"{req.evidence[0].text}\"* (Section: {req.evidence[0].section}, Cosine Sim: {req.evidence[0].similarity:.2f})\n"
        md += "\n"

    if ev.explanation:
        md += f"""---

## AI-Generated Explanation ({ev.explanation.generated_by.upper()})

> {ev.explanation.summary}

### Strengths
"""
        for s in ev.explanation.strengths:
            md += f"- {s}\n"

        md += "\n### Weaknesses / Gaps\n"
        for w in ev.explanation.weaknesses:
            md += f"- {w}\n"

        if ev.explanation.rationale:
            md += f"\n**Evaluation Rationale:**\n{ev.explanation.rationale}\n"

    md += """
---

*Notice: AI provides decision support. Final hiring decisions should remain with qualified human recruiters.*
"""
    return md


def generate_pdf_report(ev: CandidateEvaluation) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontSize=20,
        leading=24,
        textColor=HexColor("#1e1b4b"),
    )
    h2_style = ParagraphStyle(
        "ReportH2",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        textColor=HexColor("#312e81"),
        spaceBefore=10,
        spaceAfter=5,
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        textColor=HexColor("#1e293b"),
    )
    score_style = ParagraphStyle(
        "ScoreBadge",
        parent=styles["Normal"],
        fontSize=12,
        leading=16,
        textColor=HexColor("#0f766e"),
        fontName="Helvetica-Bold",
    )

    story: list[Any] = []

    # Title
    story.append(Paragraph(f"Candidate Evaluation: {ev.candidate_name}", title_style))
    story.append(Paragraph(f"Target Role: {ev.job_title} | File: {ev.file_name}", body_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=HexColor("#4338ca"), spaceAfter=10))

    # Summary Table
    summary_data = [
        ["Overall Fit Score", f"{ev.overall_score}/100", "Recommendation", ev.recommendation_label],
        ["ATS Score", f"{ev.ats_score:.1f}/100", "Experience", f"{ev.experience_years:.1f} years"],
        ["Semantic Similarity", f"{ev.semantic_similarity:.1f}%", "Human Review", "FLAGGED" if ev.needs_human_review else "NO"],
    ]
    t = Table(summary_data, colWidths=[120, 140, 120, 140])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), HexColor("#f8fafc")),
                ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cbd5e1")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 12))

    # Score Breakdown Table
    story.append(Paragraph("Transparent Scoring Formula Breakdown", h2_style))
    if ev.score_breakdown:
        breakdown_rows = [["Component", "Weight", "Score", "Contribution"]]
        for c in ev.score_breakdown.components:
            breakdown_rows.append([c.label, f"{c.weight*100:.0f}%", f"{c.value:.1f}/100", f"+{c.contribution:.1f} pts"])
        breakdown_rows.append(["Total Overall Fit Score", "100%", "", f"{ev.overall_score}/100"])

        t_break = Table(breakdown_rows, colWidths=[200, 100, 100, 120])
        t_break.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), HexColor("#e0e7ff")),
                    ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#c7d2fe")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
                    ("BACKGROUND", (0, -1), (-1, -1), HexColor("#ede9fe")),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                    ("PADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(t_break)
        story.append(Spacer(1, 10))

    # Skills summary
    story.append(Paragraph("Skill Coverage & Requirements", h2_style))
    story.append(Paragraph(f"<b>Matched Skills ({len(ev.matched_skills)}):</b> {', '.join(ev.matched_skills) if ev.matched_skills else 'None'}", body_style))
    story.append(Paragraph(f"<b>Missing Skills ({len(ev.missing_skills)}):</b> {', '.join(ev.missing_skills) if ev.missing_skills else 'None'}", body_style))
    story.append(Spacer(1, 10))

    # Explanation
    if ev.explanation:
        story.append(Paragraph(f"AI-Generated Evaluation ({ev.explanation.generated_by.upper()})", h2_style))
        story.append(Paragraph(ev.explanation.summary, body_style))
        story.append(Spacer(1, 6))

        for s in ev.explanation.strengths[:3]:
            story.append(Paragraph(f"• <b>Strength:</b> {s}", body_style))
        for w in ev.explanation.weaknesses[:2]:
            story.append(Paragraph(f"• <b>Gap:</b> {w}", body_style))

    story.append(Spacer(1, 14))
    story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor("#94a3b8"), spaceAfter=6))
    story.append(
        Paragraph(
            "<i>Notice: AI provides decision support. Final hiring decisions should remain with qualified human recruiters.</i>",
            ParagraphStyle("Disclaimer", parent=styles["Italic"], fontSize=7.5, textColor=HexColor("#64748b")),
        )
    )

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

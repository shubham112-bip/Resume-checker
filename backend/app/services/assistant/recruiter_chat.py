"""Recruiter AI Assistant: answers natural language queries grounded in evaluation evidence."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from app.config import get_settings
from app.db.database import list_evaluations_for_run
from app.schemas.api import ChatResponse, ToolCallRecord
from app.schemas.evaluation import CandidateEvaluation

logger = logging.getLogger(__name__)


def answer_recruiter_query(question: str, run_id: str | None = None) -> ChatResponse:
    evaluations: list[CandidateEvaluation] = []
    if run_id:
        evaluations = list_evaluations_for_run(run_id)

    q_lower = question.lower().strip()
    tool_calls: list[ToolCallRecord] = []
    sources: list[str] = []

    # Check if Gemini is configured
    settings = get_settings()
    api_key = settings.gemini_api_key or os.environ.get("GEMINI_API_KEY")

    # Format grounded context from evaluations
    context_blocks: list[str] = []
    for ev in evaluations:
        sources.append(f"{ev.candidate_name} ({ev.file_name})")
        matched_str = ", ".join(ev.matched_skills) if ev.matched_skills else "None"
        missing_str = ", ".join(ev.missing_skills) if ev.missing_skills else "None"
        ev_str = (
            f"Candidate: {ev.candidate_name}\n"
            f"  Overall Score: {ev.overall_score}/100 | ATS Score: {ev.ats_score}/100\n"
            f"  Recommendation: {ev.recommendation_label} ({ev.recommendation.value})\n"
            f"  Experience: {ev.experience_years:.1f} years\n"
            f"  Matched Skills: {matched_str}\n"
            f"  Missing Skills: {missing_str}\n"
            f"  Review Flags: {[f.message for f in ev.review_flags]}\n"
        )
        context_blocks.append(ev_str)

    grounded_context = "\n".join(context_blocks)

    if api_key and evaluations:
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
            prompt = f"""You are the Recruiter AI Assistant for an AI-Based Resume Screening System.
Answer the recruiter's question accurately and professionally.
GROUNDING INSTRUCTIONS:
- You must ground your answer strictly in the evaluation data below.
- Do NOT invent skills, experience, or ratings.
- Cite specific scores, matched skills, or evidence passages where relevant.

EVALUATION DATA FOR CURRENT RUN ({run_id or 'latest'}):
{grounded_context}

RECRUITER QUESTION:
{question}
"""
            tool_calls.append(
                ToolCallRecord(
                    tool="query_evaluation_store",
                    arguments={"runId": run_id, "candidates": len(evaluations)},
                    result_summary=f"Retrieved structured evaluation records for {len(evaluations)} candidate(s).",
                )
            )

            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.2),
            )
            return ChatResponse(
                answer=response.text.strip(),
                generated_by="gemini",
                tool_calls=tool_calls,
                sources=sources,
            )
        except Exception as e:
            logger.warning("Gemini recruiter assistant failed: %s. Using grounded template query matcher.", e)

    # Grounded rule-based semantic query responder (works 100% offline without API key)
    tool_calls.append(
        ToolCallRecord(
            tool="query_evaluation_store",
            arguments={"runId": run_id, "query": question},
            result_summary=f"Retrieved {len(evaluations)} candidate evaluation records from memory.",
        )
    )

    if not evaluations:
        return ChatResponse(
            answer="No evaluations found for this run. Please analyze candidates on the Dashboard first.",
            generated_by="rule-based",
            tool_calls=tool_calls,
            sources=[],
        )

    # Handle common recruiter intent patterns
    # 1. "Why was [Candidate] shortlisted / ranked..."
    cand_match = None
    for ev in evaluations:
        if ev.candidate_name.lower() in q_lower:
            cand_match = ev
            break

    if cand_match:
        if "why" in q_lower or "shortlist" in q_lower or "rank" in q_lower:
            matched_txt = ", ".join(cand_match.matched_skills) or "None"
            missing_txt = ", ".join(cand_match.missing_skills) or "None"
            ans = (
                f"**{cand_match.candidate_name}** achieved an Overall Fit Score of **{cand_match.overall_score}/100** "
                f"with recommendation **{cand_match.recommendation_label}**.\n\n"
                f"**Key Drivers:**\n"
                f"- **Verified Skills ({len(cand_match.matched_skills)}):** {matched_txt}\n"
                f"- **Experience:** {cand_match.experience_years:.1f} years\n"
                f"- **ATS Compatibility:** {cand_match.ats_score:.1f}/100\n"
                f"- **Skill Gaps:** {missing_txt}\n\n"
                f"**Summary:** {cand_match.explanation.summary if cand_match.explanation else 'Evaluation completed.'}"
            )
            return ChatResponse(answer=ans, generated_by="rule-based", tool_calls=tool_calls, sources=[cand_match.candidate_name])

        if "missing" in q_lower or "gap" in q_lower:
            missing_txt = ", ".join(cand_match.missing_skills) if cand_match.missing_skills else "No critical skills missing."
            ans = f"Missing skills for **{cand_match.candidate_name}**: {missing_txt}"
            return ChatResponse(answer=ans, generated_by="rule-based", tool_calls=tool_calls, sources=[cand_match.candidate_name])

    # 2. "Compare Candidate A and Candidate B"
    found_cands = [ev for ev in evaluations if ev.candidate_name.lower() in q_lower]
    if len(found_cands) >= 2:
        c1, c2 = found_cands[0], found_cands[1]
        ans = (
            f"### Comparison: {c1.candidate_name} vs {c2.candidate_name}\n\n"
            f"| Metric | {c1.candidate_name} | {c2.candidate_name} |\n"
            f"|---|---|---|\n"
            f"| **Overall Score** | **{c1.overall_score}/100** | **{c2.overall_score}/100** |\n"
            f"| **Recommendation** | {c1.recommendation_label} | {c2.recommendation_label} |\n"
            f"| **ATS Score** | {c1.ats_score:.1f}/100 | {c2.ats_score:.1f}/100 |\n"
            f"| **Experience** | {c1.experience_years:.1f} yrs | {c2.experience_years:.1f} yrs |\n"
            f"| **Matched Skills** | {len(c1.matched_skills)} skills | {len(c2.matched_skills)} skills |\n"
            f"| **Missing Skills** | {', '.join(c1.missing_skills) or 'None'} | {', '.join(c2.missing_skills) or 'None'} |\n\n"
            f"**Conclusion:** {c1.candidate_name if c1.overall_score > c2.overall_score else c2.candidate_name} "
            f"has stronger overall alignment with this specific job description."
        )
        return ChatResponse(answer=ans, generated_by="rule-based", tool_calls=tool_calls, sources=[c1.candidate_name, c2.candidate_name])

    # 3. "Which candidate has strongest [Skill] experience?"
    skill_words = re.findall(r"\b[A-Za-z.+]+", q_lower)
    for w in skill_words:
        matching_cands = [ev for ev in evaluations if any(w == s.lower() for s in ev.matched_skills)]
        if matching_cands:
            best = max(matching_cands, key=lambda x: x.overall_score)
            ans = (
                f"Candidate **{best.candidate_name}** demonstrated verified **{w.title()}** experience "
                f"(Overall Score: {best.overall_score}/100). "
                f"Other candidates with {w.title()}: {', '.join(c.candidate_name for c in matching_cands if c != best) or 'None'}."
            )
            return ChatResponse(answer=ans, generated_by="rule-based", tool_calls=tool_calls, sources=[c.candidate_name for c in matching_cands])

    # 4. Default: Run ranking summary
    top_cand = evaluations[0]
    ans = (
        f"**Run Summary for {top_cand.job_title}:**\n"
        f"- Top Candidate: **{top_cand.candidate_name}** ({top_cand.overall_score}/100, {top_cand.recommendation_label})\n"
        f"- Total Evaluated: {len(evaluations)} candidate(s)\n\n"
        f"**Rankings:**\n"
        + "\n".join(f"{idx+1}. {ev.candidate_name}: {ev.overall_score}/100 ({ev.recommendation_label})" for idx, ev in enumerate(evaluations[:5]))
    )
    return ChatResponse(answer=ans, generated_by="rule-based", tool_calls=tool_calls, sources=sources)

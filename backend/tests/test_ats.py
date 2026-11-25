"""Tests for ATS evaluation and keyword stuffing detection."""

from app.schemas.candidate import CandidateProfile, ResumeSection
from app.schemas.job import JobProfile, JobSkill, SkillTier
from app.services.ats.ats_engine import detect_keyword_stuffing, evaluate_ats


def test_keyword_stuffing_detection():
    job = JobProfile(
        job_id="j1",
        title="Software Engineer",
        skills=[JobSkill(name="Python", tier=SkillTier.REQUIRED, category="language")],
    )

    # Resume with repeated Python in summary but no experience evidence
    cand = CandidateProfile(
        candidate_id="c_stuff",
        name="Stuffing Candidate",
        text="Python Python Python Python Python Python developer seeking job. Worked in sales.",
        sections=[
            ResumeSection(name="summary", heading="Summary", text="Python Python Python Python Python Python developer"),
            ResumeSection(name="experience", heading="Experience", text="Sales associate at Retail Store"),
        ],
    )

    stuffing = detect_keyword_stuffing(cand, job)
    assert stuffing.suspected is True
    assert len(stuffing.flagged_terms) > 0
    assert stuffing.flagged_terms[0].term == "Python"

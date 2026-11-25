"""Unit tests for document parsing, sections, and degree extraction."""

from pathlib import Path
from app.services.parsing.education import find_degrees
from app.services.parsing.jd_parser import parse_job_description
from app.services.parsing.resume_parser import extract_contact_info, split_sections
from app.services.parsing.sections import match_heading


def test_degree_extraction():
    text = "Graduated with MCA (Master of Computer Applications) in 2022 after completing BCA in 2020."
    degrees = find_degrees(text)
    names = [d.degree for d in degrees]
    assert "MCA" in names
    assert "BCA" in names


def test_contact_extraction():
    text = """
    John Doe
    Email: john.doe@example.com
    Phone: +1 (555) 234-5678
    LinkedIn: linkedin.com/in/johndoe
    GitHub: github.com/johndoe
    """
    contact = extract_contact_info(text)
    assert contact.email == "john.doe@example.com"
    assert "linkedin.com/in/johndoe" in (contact.linkedin or "")
    assert "github.com/johndoe" in (contact.github or "")


def test_jd_skill_tiers():
    jd_text = """
    Role: Frontend Engineer
    Experience: 2-4 years
    
    Required Skills:
    - React
    - TypeScript
    - HTML
    
    Preferred Skills:
    - Next.js
    - Tailwind CSS
    """
    job = parse_job_description(jd_text)
    assert "React" in job.required_skills
    assert "Next.js" in job.preferred_skills
    assert job.min_experience_years == 2.0
    assert job.max_experience_years == 4.0

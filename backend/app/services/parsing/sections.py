"""Section heading detection shared by the resume and JD parsers."""

from __future__ import annotations

import re
from dataclasses import dataclass

RESUME_SECTION_SYNONYMS: dict[str, tuple[str, ...]] = {
    "summary": ("summary", "professional summary", "profile", "profile summary", "objective",
                "career objective", "about me", "about", "professional profile", "career summary"),
    "education": ("education", "academic background", "academics", "educational qualifications",
                  "education details", "academic qualifications", "qualifications"),
    "experience": ("experience", "work experience", "professional experience", "employment history",
                   "work history", "internships", "internship", "internship experience",
                   "employment", "relevant experience", "industry experience"),
    "skills": ("skills", "technical skills", "key skills", "core competencies", "skill set",
               "tech stack", "technologies", "technical proficiency", "skills & tools", "tools"),
    "projects": ("projects", "academic projects", "personal projects", "key projects",
                 "project experience", "major projects", "selected projects"),
    "certifications": ("certifications", "certificates", "courses", "certifications & courses",
                       "licenses", "training", "trainings"),
    "achievements": ("achievements", "awards", "honors", "honours", "accomplishments",
                     "awards & achievements", "extracurricular activities", "activities"),
    "other": ("languages", "hobbies", "interests", "personal details", "personal information",
              "declaration", "references", "strengths", "soft skills"),
}

JD_SECTION_SYNONYMS: dict[str, tuple[str, ...]] = {
    "required": ("required", "requirements", "required skills", "must have", "must-have",
                 "required qualifications", "what you need", "key requirements", "mandatory skills",
                 "technical requirements", "skills required"),
    "preferred": ("preferred", "preferred skills", "nice to have", "nice-to-have", "good to have",
                  "bonus", "desirable", "preferred qualifications", "plus"),
    "optional": ("optional", "optional skills", "exposure to", "familiarity with"),
    "responsibilities": ("responsibilities", "key responsibilities", "what you'll do",
                         "what you will do", "role", "duties", "the role", "job responsibilities"),
    "experience": ("experience", "experience required", "work experience"),
    "education": ("education", "qualification", "qualifications", "educational qualification"),
    "soft_skills": ("soft skills", "personal attributes", "competencies"),
    "about": ("about us", "about the company", "company overview", "who we are", "benefits",
              "perks", "location", "job type", "salary"),
}

_DECORATION = re.compile(r"^[\s#*_=\-–•]+|[\s#*_=\-–:]+$")


@dataclass(frozen=True)
class HeadingMatch:
    canonical: str
    heading: str
    remainder: str  # inline content after "Heading:" on the same line


def _lookup(synonyms: dict[str, tuple[str, ...]]) -> dict[str, str]:
    return {syn: canonical for canonical, syns in synonyms.items() for syn in syns}


_RESUME_LOOKUP = _lookup(RESUME_SECTION_SYNONYMS)
_JD_LOOKUP = _lookup(JD_SECTION_SYNONYMS)


def match_heading(line: str, kind: str = "resume") -> HeadingMatch | None:
    lookup = _RESUME_LOOKUP if kind == "resume" else _JD_LOOKUP
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return None
    if ":" in stripped:  # "Required: React, Node.js" or "Skills:"
        head, _, rest = stripped.partition(":")
        key = _DECORATION.sub("", head).lower()
        if key in lookup and len(key.split()) <= 5:
            return HeadingMatch(lookup[key], head.strip(), rest.strip())
    key = _DECORATION.sub("", stripped).lower()
    if key in lookup:
        return HeadingMatch(lookup[key], stripped, "")

    # Prefix match on lines starting with canonical headings without a colon
    for syn in sorted(lookup.keys(), key=len, reverse=True):
        if len(syn) < 4:
            continue
        pattern = rf"^({re.escape(syn)})\b\s*(.*)$"
        m = re.match(pattern, stripped, re.IGNORECASE)
        if m:
            head = m.group(1)
            rest = m.group(2).strip()
            if rest and not rest.lower().startswith(("with ", "in ", "of ", "at ", "is ", "was ", "has ", "and ", "to ")):
                return HeadingMatch(lookup[syn], head, rest)

    return None


def looks_like_heading(line: str) -> bool:
    """Heuristic for unknown headings: short ALL-CAPS line without contact data."""
    stripped = _DECORATION.sub("", line.strip())
    if not stripped or len(stripped.split()) > 4 or any(ch.isdigit() for ch in stripped):
        return False
    if any(ch in stripped for ch in "@|,.()/"):
        return False
    letters = [c for c in stripped if c.isalpha()]
    return len(letters) >= 4 and all(c.isupper() for c in letters)

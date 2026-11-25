"""Structured candidate profile produced by the Resume Parsing Agent."""

from __future__ import annotations

from pydantic import Field

from app.schemas.base import CamelModel


class ContactInfo(CamelModel):
    email: str | None = None
    phone: str | None = None
    linkedin: str | None = None
    github: str | None = None
    portfolio: str | None = None


class ResumeSection(CamelModel):
    name: str  # canonical name: summary, education, experience, skills, ...
    heading: str  # heading text exactly as written in the resume
    text: str
    start_line: int = 0


class ExperienceEntry(CamelModel):
    title: str = ""
    organization: str = ""
    start: str | None = None
    end: str | None = None
    duration_months: int = 0
    is_current: bool = False
    is_internship: bool = False
    bullets: list[str] = Field(default_factory=list)


class EducationEntry(CamelModel):
    degree: str
    degree_level: int = 0  # 1 diploma, 2 bachelor, 3 master, 4 doctorate
    field: str | None = None
    is_relevant_field: bool = False
    institution: str | None = None
    year: str | None = None
    raw: str = ""


class ProjectEntry(CamelModel):
    name: str
    description: str = ""
    technologies: list[str] = Field(default_factory=list)


class DocumentDiagnostics(CamelModel):
    file_type: str
    parser_used: str
    page_count: int = 0
    char_count: int = 0
    word_count: int = 0
    image_count: int = 0
    table_count: int = 0
    is_scanned: bool = False
    warnings: list[str] = Field(default_factory=list)


class SensitiveAttributeFinding(CamelModel):
    attribute: str
    action: str  # "redacted" | "excluded"


class CandidateProfile(CamelModel):
    candidate_id: str
    name: str = "Unknown Candidate"
    contact: ContactInfo = Field(default_factory=ContactInfo)
    summary: str = ""
    education: list[EducationEntry] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    experience: list[ExperienceEntry] = Field(default_factory=list)
    projects: list[ProjectEntry] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    sections: list[ResumeSection] = Field(default_factory=list)
    unrecognized_headings: list[str] = Field(default_factory=list)
    total_experience_months: int = 0
    diagnostics: DocumentDiagnostics | None = None
    sensitive_attributes_removed: list[SensitiveAttributeFinding] = Field(default_factory=list)
    text: str = Field(default="", exclude=True)  # redacted full text, never serialised

    def section_text(self, name: str) -> str:
        return "\n".join(s.text for s in self.sections if s.name == name)

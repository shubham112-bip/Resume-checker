"""Structured job profile produced by the JD Analysis Agent."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from app.schemas.base import CamelModel


class SkillTier(StrEnum):
    REQUIRED = "REQUIRED"
    PREFERRED = "PREFERRED"
    OPTIONAL = "OPTIONAL"


class JobSkill(CamelModel):
    name: str
    tier: SkillTier
    category: str = "technical"
    context: str = ""  # JD sentence the skill was found in


class JobProfile(CamelModel):
    job_id: str = ""
    title: str = "Untitled Role"
    skills: list[JobSkill] = Field(default_factory=list)
    min_experience_years: float = 0.0
    max_experience_years: float | None = None
    education_requirements: list[str] = Field(default_factory=list)
    required_degree_level: int = 0
    requires_relevant_field: bool = False
    responsibilities: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    domain_requirements: list[str] = Field(default_factory=list)
    soft_skills: list[str] = Field(default_factory=list)
    extraction_method: str = "rule-based"
    raw_text: str = ""

    def skills_by_tier(self, tier: SkillTier) -> list[JobSkill]:
        return [s for s in self.skills if s.tier == tier]

    @property
    def required_skills(self) -> list[str]:
        return [s.name for s in self.skills_by_tier(SkillTier.REQUIRED)]

    @property
    def preferred_skills(self) -> list[str]:
        return [s.name for s in self.skills_by_tier(SkillTier.PREFERRED)]

    @property
    def optional_skills(self) -> list[str]:
        return [s.name for s in self.skills_by_tier(SkillTier.OPTIONAL)]

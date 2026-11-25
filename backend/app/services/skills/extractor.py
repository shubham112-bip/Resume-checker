"""Dictionary/regex based skill extraction with alias normalisation."""

from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache

from app.services.skills.taxonomy import SKILL_TAXONOMY, SkillDefinition, get_skill

_LEFT = r"(?<![A-Za-z0-9+#])"
_RIGHT = r"(?![A-Za-z0-9+#]|\.[A-Za-z0-9])"


def _alias_pattern(alias: str) -> str:
    escaped = re.escape(alias).replace(r"\ ", r"[\s\-]+")
    return f"{_LEFT}{escaped}{_RIGHT}"


@lru_cache(maxsize=512)
def _skill_regexes(name: str) -> tuple[re.Pattern[str], ...]:
    skill = get_skill(name)
    if skill is None:
        return (re.compile(_alias_pattern(name), re.IGNORECASE),)
    patterns: list[re.Pattern[str]] = []
    insensitive = [a for a in (skill.name, *skill.aliases) if a not in skill.case_sensitive]
    if insensitive:
        joined = "|".join(_alias_pattern(a) for a in sorted(insensitive, key=len, reverse=True))
        patterns.append(re.compile(joined, re.IGNORECASE))
    if skill.case_sensitive:
        joined = "|".join(_alias_pattern(a) for a in skill.case_sensitive)
        patterns.append(re.compile(joined))
    return tuple(patterns)


@lru_cache(maxsize=512)
def _literal_regex(term: str) -> re.Pattern[str]:
    forms = [term]
    if "." not in term and re.search(r"([A-Z]|[a-z]{3})s$", term):
        forms.append(term[:-1])  # simple singular form: "REST APIs" -> "REST API"
    return re.compile("|".join(_alias_pattern(f) for f in forms), re.IGNORECASE)


def count_skill_mentions(name: str, text: str) -> int:
    """Number of mentions of a skill or any of its aliases."""
    return sum(len(p.findall(text)) for p in _skill_regexes(name))


def mentions_skill(name: str, text: str) -> bool:
    return any(p.search(text) for p in _skill_regexes(name))


def count_literal_mentions(term: str, text: str) -> int:
    """Literal keyword count (no synonyms) - emulates a traditional keyword ATS."""
    return len(_literal_regex(term).findall(text))


def extract_skills(text: str, categories: set[str] | None = None) -> list[str]:
    """Return canonical skill names found in text, ordered by first appearance."""
    found: list[tuple[int, str]] = []
    for skill in SKILL_TAXONOMY:
        if categories and skill.category not in categories:
            continue
        first = _first_position(skill, text)
        if first is not None:
            found.append((first, skill.name))
    return [name for _, name in sorted(found)]


def skill_frequencies(text: str, names: list[str]) -> Counter[str]:
    return Counter({n: count_skill_mentions(n, text) for n in names})


def _first_position(skill: SkillDefinition, text: str) -> int | None:
    positions = [m.start() for p in _skill_regexes(skill.name) for m in [p.search(text)] if m]
    return min(positions) if positions else None


def skill_category(name: str) -> str:
    skill = get_skill(name)
    return skill.category if skill else "technical"

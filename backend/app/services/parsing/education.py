"""Degree recognition shared by resume and JD parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass

DIPLOMA, BACHELOR, MASTER, DOCTORATE = 1, 2, 3, 4

# (pattern, canonical degree, level, inherently computing-related)
_DEGREES: tuple[tuple[re.Pattern[str], str, int, bool], ...] = tuple(
    (re.compile(p, flags), name, level, cs)
    for p, flags, name, level, cs in (
        (r"\bph\.?\s?d\b|\bdoctorate\b", re.I, "PhD", DOCTORATE, False),
        (r"\bm\.?\s?tech\b", re.I, "MTech", MASTER, False),
        (r"\bm\.?\s?c\.?\s?a\b", re.I, "MCA", MASTER, True),
        (r"\bm\.?\s?sc\b", re.I, "MSc", MASTER, False),
        (r"\bM\.?E\.(?=[\s,(]|$)", 0, "ME", MASTER, False),
        (r"\bm\.?\s?b\.?\s?a\b", re.I, "MBA", MASTER, False),
        (r"\bmaster(?:'s|s)?\b", re.I, "Master's", MASTER, False),
        (r"\bb\.?\s?tech\b", re.I, "BTech", BACHELOR, False),
        (r"\bB\.?E\.(?=[\s,(]|$)|\bB\.?E\b(?=\s+in\b)", 0, "BE", BACHELOR, False),
        (r"\bb\.?\s?c\.?\s?a\b", re.I, "BCA", BACHELOR, True),
        (r"\bb\.?\s?sc\b", re.I, "BSc", BACHELOR, False),
        (r"\bb\.?\s?com\b", re.I, "BCom", BACHELOR, False),
        (r"\bB\.?A\.(?=[\s,(]|$)", 0, "BA", BACHELOR, False),
        (r"\bbachelor(?:'s|s)?\b", re.I, "Bachelor's", BACHELOR, False),
        (r"\bdiploma\b", re.I, "Diploma", DIPLOMA, False),
    )
)

_RELEVANT_FIELD = re.compile(
    r"computer|information\s+technology|\bIT\b|\bCSE?\b|software|data\s+science|"
    r"artificial\s+intelligence|\bAI\b|computing|computer\s+applications|informatics",
    re.IGNORECASE,
)
_FIELD = re.compile(r"\(([^)]+)\)|\bin\s+([A-Za-z &]+)|\b(?:-|–)\s*([A-Za-z &]+)")


@dataclass(frozen=True)
class DegreeMatch:
    degree: str
    level: int
    field: str | None
    is_relevant_field: bool


def find_degrees(text: str) -> list[DegreeMatch]:
    """All degrees mentioned in text (deduplicated, in order)."""
    found: list[tuple[int, DegreeMatch]] = []
    seen: set[str] = set()
    for pattern, name, level, inherently_cs in _DEGREES:
        for m in pattern.finditer(text):
            if name in seen:
                continue
            seen.add(name)
            tail = text[m.end(): m.end() + 80].split("\n")[0]
            field_match = _FIELD.search(tail)
            field = next((g for g in field_match.groups() if g), None) if field_match else None
            relevant = inherently_cs or bool(_RELEVANT_FIELD.search(tail))
            found.append((m.start(), DegreeMatch(name, level, field.strip() if field else None, relevant)))
    # Generic words ("Master's", "Bachelor's") are dropped when a specific degree matched.
    specific = [d for _, d in found if d.degree not in {"Master's", "Bachelor's"}]
    generic = [d for _, d in found if d.degree in {"Master's", "Bachelor's"}]
    levels = {d.level for d in specific}
    result = [d for _, d in sorted(found, key=lambda x: x[0]) if d in specific]
    result += [d for d in generic if d.level not in levels]
    return result


def mentions_relevant_field(text: str) -> bool:
    return bool(_RELEVANT_FIELD.search(text))

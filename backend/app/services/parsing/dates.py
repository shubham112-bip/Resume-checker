"""Date-range extraction and experience-duration arithmetic."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_DATE = rf"(?:{_MONTH}\s*,?\s*\d{{4}}|\d{{1,2}}\s*/\s*\d{{4}}|\d{{4}})"
_END = rf"(?:{_DATE}|present|current|now|till\s+date|ongoing|today)"
DATE_RANGE = re.compile(rf"(?P<start>{_DATE})(?:\s*(?:-|–|—|to|until)\s*|\s+)(?P<end>{_END})", re.IGNORECASE)
_CURRENT = {"present", "current", "now", "till date", "ongoing", "today"}


@dataclass(frozen=True)
class DateRange:
    start_text: str
    end_text: str
    start: tuple[int, int]
    end: tuple[int, int]
    is_current: bool
    format_style: str  # "month-name" | "numeric" | "year-only"

    @property
    def months(self) -> int:
        return max(0, (self.end[0] - self.start[0]) * 12 + (self.end[1] - self.start[1])) or 1

    @property
    def is_inverted(self) -> bool:
        return self.end < self.start


def _parse_point(text: str, is_end: bool, today: date) -> tuple[tuple[int, int], str]:
    t = text.strip().lower()
    if re.sub(r"\s+", " ", t) in _CURRENT:
        return (today.year, today.month), "current"
    numeric = re.match(r"(\d{1,2})\s*/\s*(\d{4})", t)
    if numeric:
        return (int(numeric.group(2)), min(max(int(numeric.group(1)), 1), 12)), "numeric"
    named = re.match(r"([a-z]+)\.?\s*,?\s*(\d{4})", t)
    if named and named.group(1)[:3] in _MONTHS:
        return (int(named.group(2)), _MONTHS[named.group(1)[:3]]), "month-name"
    year = int(re.search(r"\d{4}", t).group(0))  # type: ignore[union-attr]
    return (year, 12 if is_end else 1), "year-only"


def find_date_ranges(text: str, today: date | None = None) -> list[DateRange]:
    today = today or date.today()
    ranges: list[DateRange] = []
    for match in DATE_RANGE.finditer(text):
        start, start_style = _parse_point(match.group("start"), False, today)
        end, end_style = _parse_point(match.group("end"), True, today)
        style = start_style if end_style == "current" else (
            start_style if start_style == end_style else "mixed"
        )
        ranges.append(
            DateRange(match.group("start"), match.group("end"), start, end, end_style == "current", style)
        )
    return ranges


def merged_months(ranges: list[DateRange]) -> int:
    """Total months covered by possibly-overlapping ranges."""
    intervals = sorted(
        (r.start[0] * 12 + r.start[1], r.end[0] * 12 + r.end[1]) for r in ranges if not r.is_inverted
    )
    total, current_start, current_end = 0, None, None
    for start, end in intervals:
        if current_end is None or start > current_end:
            if current_start is not None and current_end is not None:
                total += max(current_end - current_start, 1)
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_start is not None and current_end is not None:
        total += max(current_end - current_start, 1)
    return total

"""Fairness safeguards: remove protected / sensitive attributes before any analysis.

Nothing in this module contributes to a score. It redacts sensitive lines so they never
reach feature engineering, embeddings, the ML model or the LLM, and it audits the
feature list used for scoring.
"""

from __future__ import annotations

import re

from app.schemas.candidate import SensitiveAttributeFinding

# attribute label -> regex matching a "Label: value" style line
_SENSITIVE_PATTERNS: dict[str, re.Pattern[str]] = {
    label: re.compile(rf"^\s*(?:{pattern})\s*[:\-–]\s*.+$", re.IGNORECASE | re.MULTILINE)
    for label, pattern in {
        "Date of birth": r"date\s+of\s+birth|d\.?o\.?b\.?|birth\s*date",
        "Age": r"age",
        "Gender": r"gender|sex",
        "Marital status": r"marital\s+status",
        "Religion": r"religion",
        "Caste / category": r"caste|category",
        "Nationality": r"nationality|citizenship",
        "Ethnicity / race": r"ethnicity|race",
        "Family details": r"father'?s\s+name|mother'?s\s+name|husband'?s\s+name|spouse",
        "Blood group": r"blood\s+group",
        "Height / weight": r"height|weight",
        "Photograph": r"photo(?:graph)?",
        "Physical address": r"home\s+address|street\s+address|current\s+address|residential\s+address",
        "Email contact": r"e-?mail(?:\s+address)?",
        "Phone contact": r"phone(?:\s+number)?|mobile(?:\s+number)?|cell(?:\s+number)?",
    }.items()
}

_EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
_PHONE_REGEX = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b")

PROTECTED_ATTRIBUTES: frozenset[str] = frozenset(
    {"gender", "sex", "age", "dob", "birth", "religion", "caste", "race", "ethnicity",
     "marital", "nationality", "photo", "photograph", "name", "address", "disability",
     "email", "phone", "cell", "mobile"}
)

REDACTION = "[REDACTED - not used for evaluation]"


def redact_sensitive_attributes(
    text: str, image_count: int = 0
) -> tuple[str, list[SensitiveAttributeFinding]]:
    findings: list[SensitiveAttributeFinding] = []
    for label, pattern in _SENSITIVE_PATTERNS.items():
        if pattern.search(text):
            text = pattern.sub(REDACTION, text)
            findings.append(SensitiveAttributeFinding(attribute=label, action="redacted"))

    # Also redact raw email addresses and phone numbers to prevent accidental leakage
    if _EMAIL_REGEX.search(text):
        text = _EMAIL_REGEX.sub("[EMAIL REDACTED]", text)
        if not any(f.attribute == "Email address" for f in findings):
            findings.append(SensitiveAttributeFinding(attribute="Email address", action="redacted"))

    if _PHONE_REGEX.search(text):
        text = _PHONE_REGEX.sub("[PHONE REDACTED]", text)
        if not any(f.attribute == "Phone number" for f in findings):
            findings.append(SensitiveAttributeFinding(attribute="Phone number", action="redacted"))

    if image_count:
        findings.append(
            SensitiveAttributeFinding(
                attribute=f"Embedded images/photos ({image_count})", action="excluded"
            )
        )
    return text, findings


def audit_feature_names(feature_names: list[str]) -> list[str]:
    """Return any feature names that look like protected attributes (should be empty)."""
    return [
        f for f in feature_names if any(tok in PROTECTED_ATTRIBUTES for tok in re.split(r"[_\W]+", f.lower()))
    ]

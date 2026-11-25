"""Centralised configuration.

All scoring weights, recommendation thresholds, evidence credits and semantic
thresholds live here so they can be changed without touching the pipeline.
Secrets are read only from environment variables (or backend/.env).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
ROOT_DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR = ROOT_DATA_DIR if ROOT_DATA_DIR.exists() else (BACKEND_DIR / "data")
RUNTIME_DIR = BACKEND_DIR / "data" / "runtime"  # git-ignored: uploads, sqlite db
DEMO_DIR = BACKEND_DIR / "data" / "demo"
ML_DIR = BACKEND_DIR / "data" / "ml"

# Phase 2 Dataset Directory Paths
RESUMES_DIR = ROOT_DATA_DIR / "resumes"
JOB_DESCRIPTIONS_DIR = ROOT_DATA_DIR / "job_descriptions"
LABELS_DIR = ROOT_DATA_DIR / "labels"
PROCESSED_DIR = ROOT_DATA_DIR / "processed"
SYNTHETIC_DIR = ROOT_DATA_DIR / "synthetic"


class ScoringWeights(BaseModel):
    """Weights of the transparent Overall Candidate Fit Score (must sum to 1)."""

    requiredSkillMatch: float = 0.30
    experienceMatch: float = 0.20
    semanticSimilarity: float = 0.15
    projectMatch: float = 0.10
    atsScore: float = 0.10
    educationMatch: float = 0.10
    resumeQuality: float = 0.05

    @model_validator(mode="after")
    def _sum_to_one(self) -> ScoringWeights:
        total = sum(self.model_dump().values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Scoring weights must sum to 1.0 (got {total:.3f})")
        return self


class RecommendationThresholds(BaseModel):
    """Lower bounds (0-100) of each recommendation band."""

    strongShortlist: float = 85.0
    shortlist: float = 70.0
    review: float = 55.0
    borderlineMargin: float = 3.0  # +/- points around a band edge => human review


class EvidenceCredits(BaseModel):
    """Credit (0-1) granted to a requirement for each evidence status."""

    MATCH: float = 1.0
    SEMANTIC_MATCH: float = 0.85
    CLAIMED_ONLY: float = 0.6
    INFERRED: float = 0.5
    UNCERTAIN: float = 0.25
    NOT_FOUND: float = 0.0


class SemanticThresholds(BaseModel):
    """Cosine-similarity thresholds calibrated for BAAI/bge-small-en-v1.5."""

    strong: float = 0.72  # >= strong  -> semantic match
    weak: float = 0.62  # [weak, strong) -> uncertain, triggers reflection
    calibrationLow: float = 0.55  # similarity mapped to 0
    calibrationHigh: float = 0.82  # similarity mapped to 1
    topK: int = 3


class ATSWeights(BaseModel):
    contact: float = 0.20
    sections: float = 0.30
    keywordCoverage: float = 0.30
    formatting: float = 0.20


class SkillTierWeights(BaseModel):
    """Relative weight of JD skill tiers inside keyword coverage."""

    REQUIRED: float = 3.0
    PREFERRED: float = 2.0
    OPTIONAL: float = 1.0


class StuffingConfig(BaseModel):
    repeatThreshold: int = 4  # a JD term mentioned >= N times...
    maxEvidenceRatio: float = 0.34  # ...with fewer than this share in evidence sections
    densityThreshold: float = 0.12  # JD-term tokens / all tokens


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL")
    gemini_timeout_seconds: int = Field(default=30, alias="GEMINI_TIMEOUT_SECONDS")
    use_llm_jd_extraction: bool = Field(default=True, alias="USE_LLM_JD_EXTRACTION")

    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5", alias="EMBEDDING_MODEL")
    embedding_backend: str = Field(default="sentence-transformers", alias="EMBEDDING_BACKEND")

    max_upload_mb: int = Field(default=5, alias="MAX_UPLOAD_MB")
    max_files_per_upload: int = Field(default=20, alias="MAX_FILES_PER_UPLOAD")
    frontend_origin: str = Field(default="http://localhost:3000", alias="FRONTEND_ORIGIN")

    database_url: str = Field(
        default=f"sqlite:///{RUNTIME_DIR / 'resume_screening.db'}", alias="DATABASE_URL"
    )
    upload_dir: Path = Field(default=RUNTIME_DIR / "uploads", alias="UPLOAD_DIR")
    evaluation_workers: int = Field(default=4, alias="EVALUATION_WORKERS")

    weights: ScoringWeights = ScoringWeights()
    thresholds: RecommendationThresholds = RecommendationThresholds()
    credits: EvidenceCredits = EvidenceCredits()
    semantic: SemanticThresholds = SemanticThresholds()
    ats_weights: ATSWeights = ATSWeights()
    tier_weights: SkillTierWeights = SkillTierWeights()
    stuffing: StuffingConfig = StuffingConfig()

    allowed_extensions: tuple[str, ...] = (".pdf", ".docx", ".txt")

    @property
    def gemini_enabled(self) -> bool:
        return bool(self.gemini_api_key and self.gemini_api_key.strip())

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()

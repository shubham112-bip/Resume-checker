"""Synthetic dataset generator for academic candidate evaluation ML models."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
from app.config import ML_DIR

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "required_skill_match",
    "experience_match",
    "semantic_similarity",
    "education_match",
    "project_match",
    "ats_score",
    "resume_quality",
]

TARGET_LABELS = ["weak_match", "review", "shortlist"]


def generate_synthetic_dataset(
    output_path: Path | None = None, n_samples: int = 800, random_seed: int = 42
) -> Path:
    """Generate reproducible synthetic candidate features with realistic correlations.

    Documented as synthetic demonstration data for college demo / viva.
    """
    rng = np.random.default_rng(random_seed)
    path = output_path or (ML_DIR / "synthetic_candidates.csv")
    path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, str | float]] = []

    for i in range(n_samples):
        # Latent profile competence: [0.0 - 1.0]
        latent = rng.beta(2.5, 2.5)

        # Generate correlated features with realistic noise
        req_skill = np.clip(latent + rng.normal(0, 0.12), 0.05, 1.0)
        exp_match = np.clip(latent * 0.85 + rng.normal(0, 0.15), 0.1, 1.0)
        sem_sim = np.clip(latent * 0.75 + req_skill * 0.2 + rng.normal(0, 0.1), 0.1, 1.0)
        edu_match = rng.choice([0.5, 0.7, 0.85, 1.0], p=[0.15, 0.25, 0.35, 0.25])
        proj_match = np.clip(latent * 0.7 + rng.normal(0, 0.18), 0.1, 1.0)
        ats_score = np.clip(rng.beta(4.0, 1.5) * 0.95 + rng.normal(0, 0.08), 0.2, 1.0)
        res_qual = np.clip(0.6 + rng.normal(0, 0.15), 0.2, 1.0)

        # Realistic weighted score for ground truth label assignment
        weighted = (
            0.30 * req_skill
            + 0.20 * exp_match
            + 0.15 * sem_sim
            + 0.10 * proj_match
            + 0.10 * ats_score
            + 0.10 * edu_match
            + 0.05 * res_qual
        )

        # Add realistic decision noise / human reviewer subjectivity
        noisy_score = weighted + rng.normal(0, 0.04)

        if noisy_score >= 0.70:
            label = "shortlist"
        elif noisy_score >= 0.52:
            label = "review"
        else:
            label = "weak_match"

        records.append(
            {
                "candidate_id": f"SYN_{i+1:04d}",
                "required_skill_match": round(float(req_skill), 3),
                "experience_match": round(float(exp_match), 3),
                "semantic_similarity": round(float(sem_sim), 3),
                "education_match": round(float(edu_match), 3),
                "project_match": round(float(proj_match), 3),
                "ats_score": round(float(ats_score), 3),
                "resume_quality": round(float(res_qual), 3),
                "label": label,
            }
        )

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    logger.info("Generated %d synthetic candidate records at %s", n_samples, path)
    return path

"""Section-aware chunking and semantic matching using FAISS and local embeddings."""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
from app.config import get_settings
from app.schemas.candidate import CandidateProfile
from app.schemas.evaluation import EvidencePassage, EvidenceStatus, RequirementMatch
from app.schemas.job import JobProfile, JobSkill
from app.services.embeddings.embedder import get_embedder
from app.services.embeddings.vector_store import DocumentChunk, FAISSVectorStore
from app.services.skills.extractor import count_literal_mentions, count_skill_mentions
from app.services.skills.taxonomy import get_skill

logger = logging.getLogger(__name__)


def chunk_resume(profile: CandidateProfile) -> list[DocumentChunk]:
    """Break resume into meaningful semantic passages with section metadata."""
    chunks: list[DocumentChunk] = []
    chunk_idx = 0

    # 1. Experience bullets / passages (highest value evidence)
    for exp in profile.experience:
        role_ctx = f"Role: {exp.title} at {exp.organization}"
        for b in exp.bullets:
            if len(b.strip()) > 15:
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"exp_{chunk_idx}",
                        text=f"{role_ctx}. {b.strip()}",
                        section="experience",
                        metadata={"type": "experience", "title": exp.title, "org": exp.organization},
                    )
                )
                chunk_idx += 1

    # 2. Project entries
    for proj in profile.projects:
        proj_ctx = f"Project: {proj.name}. Tech: {', '.join(proj.technologies)}"
        if proj.description:
            chunks.append(
                DocumentChunk(
                    chunk_id=f"proj_{chunk_idx}",
                    text=f"{proj_ctx}. {proj.description.strip()}",
                    section="projects",
                    metadata={"type": "project", "name": proj.name},
                )
            )
            chunk_idx += 1

    # 3. Summary section
    if profile.summary:
        for sentence in re.split(r"(?<=[.!?])\s+", profile.summary):
            if len(sentence.strip()) > 20:
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"sum_{chunk_idx}",
                        text=sentence.strip(),
                        section="summary",
                        metadata={"type": "summary"},
                    )
                )
                chunk_idx += 1

    # 4. Fallback: all section text paragraphs if no bullets were extracted
    if len(chunks) < 3:
        for sec in profile.sections:
            paragraphs = sec.text.split("\n\n")
            for p in paragraphs:
                p_clean = " ".join(p.split())
                if len(p_clean) > 25:
                    chunks.append(
                        DocumentChunk(
                            chunk_id=f"sec_{chunk_idx}",
                            text=p_clean,
                            section=sec.name,
                            metadata={"type": sec.name},
                        )
                    )
                    chunk_idx += 1

    return chunks


def build_candidate_vector_store(profile: CandidateProfile) -> tuple[FAISSVectorStore, list[DocumentChunk]]:
    embedder = get_embedder()
    chunks = chunk_resume(profile)
    store = FAISSVectorStore(dimension=embedder.dimension)

    if chunks:
        passages = [c.text for c in chunks]
        embeddings = embedder.embed_passages(passages)
        store.add_chunks(chunks, embeddings)

    return store, chunks


def _build_skill_query(skill: JobSkill) -> str:
    s_def = get_skill(skill.name)
    gloss = s_def.description if s_def and s_def.description else skill.name
    if skill.context:
        return f"{skill.name}: {gloss}. Context: {skill.context}"
    return f"{skill.name}: {gloss}"


def calibrate_similarity(sim: float, low: float = 0.55, high: float = 0.82) -> float:
    """Calibrate raw cosine similarity to a 0.0 - 1.0 confidence score."""
    if sim <= low:
        return 0.0
    if sim >= high:
        return 1.0
    return (sim - low) / (high - low)


def match_requirements(
    profile: CandidateProfile,
    job: JobProfile,
    prebuilt_store: FAISSVectorStore | None = None,
) -> tuple[list[RequirementMatch], float]:
    """Evaluate candidate against job requirements using vector search + ATS checks."""
    if prebuilt_store is not None:
        store = prebuilt_store
    else:
        store, _ = build_candidate_vector_store(profile)
    embedder = get_embedder()
    cfg = get_settings().semantic
    credits = get_settings().credits

    requirement_matches: list[RequirementMatch] = []
    similarities_list: list[float] = []

    skills_to_evaluate = job.skills
    if not skills_to_evaluate:
        return [], 0.0

    queries = [_build_skill_query(s) for s in skills_to_evaluate]
    query_embeddings = embedder.embed_queries(queries)

    evidence_text = profile.section_text("experience") + "\n" + profile.section_text("projects")

    for idx, skill in enumerate(skills_to_evaluate):
        q_emb = query_embeddings[idx]
        search_results = store.search(q_emb, top_k=cfg.topK)

        evidence_passages: list[EvidencePassage] = []
        top_sim = 0.0
        if search_results:
            top_sim = search_results[0].similarity
            for sr in search_results:
                evidence_passages.append(
                    EvidencePassage(
                        text=sr.chunk.text,
                        section=sr.chunk.section,
                        similarity=round(sr.similarity, 3),
                        match_type="semantic",
                    )
                )

        # Keyword checks
        literal_all = count_literal_mentions(skill.name, profile.text)
        literal_ev = count_literal_mentions(skill.name, evidence_text)
        mention_ev = count_skill_mentions(skill.name, evidence_text)
        mention_all = count_skill_mentions(skill.name, profile.text)

        keyword_found = (mention_all > 0)
        calibrated = calibrate_similarity(top_sim, cfg.calibrationLow, cfg.calibrationHigh)
        similarities_list.append(calibrated)

        # Determine evidence status
        status: EvidenceStatus
        note = ""

        if mention_ev > 0:
            status = EvidenceStatus.MATCH
            note = f"Explicit mention supported by {skill.name} experience in resume."
        elif top_sim >= cfg.strong:
            status = EvidenceStatus.SEMANTIC_MATCH
            note = f"Semantic match (similarity: {top_sim:.2f}) despite differing wording."
        elif mention_all > 0:
            status = EvidenceStatus.CLAIMED_ONLY
            note = f"Listed in skills section but without detailed project/work evidence."
        elif top_sim >= cfg.weak:
            status = EvidenceStatus.UNCERTAIN
            note = f"Possible partial match ({top_sim:.2f}). Flagged for reflection/review."
        else:
            status = EvidenceStatus.NOT_FOUND
            note = f"No relevant {skill.name} experience was found in the resume."

        credit = getattr(credits, status.value, 0.0)

        requirement_matches.append(
            RequirementMatch(
                requirement=skill.name,
                tier=skill.tier,
                status=status,
                confidence=round(calibrated, 2),
                credit=credit,
                keyword_found=keyword_found,
                reflected=False,
                evidence=evidence_passages,
                note=note,
            )
        )

    mean_semantic_sim = float(np.mean(similarities_list)) if similarities_list else 0.0
    return requirement_matches, mean_semantic_sim

#!/usr/bin/env python3
"""One-command CLI Demonstration Runner for Viva and Faculty Presentation.

Runs the complete multi-agent pipeline on the 5-candidate demo scenario and prints
the ranking table, evidence citations, ATS vs AI comparison, and ML model performance.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from app.api.endpoints import load_demo_bundle
from app.db.database import init_db
from app.schemas.api import EvaluateRequest
from app.api.endpoints import evaluate_candidates
from app.services.demo_data import run_comparison_benchmark
from app.services.ml.predictor import get_cached_metrics


def main():
    print("=" * 80)
    print("AI-BASED RESUME SCREENING & CANDIDATE EVALUATION SYSTEM")
    print("Academic Viva & Technical Demonstration CLI")
    print("=" * 80)

    print("\n[1/4] Initializing Database & Pre-warming Vector Embeddings (BGE-small)...")
    init_db()

    print("\n[2/4] Loading Demo Job Description & 5 Candidate Resumes...")
    demo_bundle = load_demo_bundle()
    job = demo_bundle.job
    candidates = demo_bundle.upload.candidates
    print(f"  ✓ Target Job: '{job.title}'")
    print(f"  ✓ Required Skills: {', '.join(job.required_skills)}")
    print(f"  ✓ Preferred Skills: {', '.join(job.preferred_skills)}")
    print(f"  ✓ Candidates Staged: {len(candidates)}")

    print("\n[3/4] Executing LangGraph Multi-Agent Evaluation Pipeline...")
    t0 = time.time()
    req = EvaluateRequest(
        jobId=job.job_id,
        candidateIds=[c.candidate_id for c in candidates],
    )
    run_resp = evaluate_candidates(req)
    duration = time.time() - t0
    print(f"  ✓ Multi-Agent Pipeline Completed in {duration:.2f}s (Run ID: {run_resp.run_id})\n")

    print("-" * 80)
    print(f"{'Rank':<5} {'Candidate Name':<20} {'Score':<10} {'Recommendation':<18} {'Keyword ATS':<12} {'Delta'}")
    print("-" * 80)

    for r in run_resp.ranking:
        delta = r.keyword_rank - r.rank
        delta_str = f"+{delta}" if delta > 0 else f"{delta}" if delta < 0 else "="
        flag_str = " [FLAGGED]" if r.needs_human_review else ""
        print(
            f"#{r.rank:<4} {r.candidate_name:<20} {r.overall_score:>5.1f}/100  "
            f"{r.recommendation_label + flag_str:<18} {r.keyword_ats_score:>5.1f}% (Rank #{r.keyword_rank})  {delta_str}"
        )

    print("-" * 80)

    print("\n[4/4] Controlled Benchmark: Traditional Keyword ATS vs Semantic AI (Cases A-D):")
    comp = run_comparison_benchmark()
    for c in comp.cases:
        print(f"\n  • {c.title}:")
        print(f"    - Requirement: {c.requirement}")
        print(f"    - Traditional ATS: {'Found' if c.keyword_ats_found else 'MISSING (0 pts)'} ({c.keyword_mentions} mentions)")
        print(f"    - Semantic AI:     Status: {c.ai_status} (Sim: {c.ai_similarity:.2f})")
        print(f"    - Verdict:         {c.verdict}")

    metrics = get_cached_metrics()
    print(f"\n[ML Validation] Best Model: {metrics.selected_model}")
    for m in metrics.models:
        print(f"  - {m.name:<20} Acc: {m.accuracy*100:.1f}% | Macro F1: {m.f1*100:.1f}% | CV F1: {m.cv_f1_mean*100:.1f}%")

    print("\n" + "=" * 80)
    print("VIVA DEMO READY: Start web servers with: uvicorn app.main:app & npm run dev (in frontend)")
    print("=" * 80)


if __name__ == "__main__":
    main()

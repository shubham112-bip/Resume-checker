"""Integration tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "embeddingBackend" in data
    assert data["mlModelAvailable"] is True


def test_config_endpoint():
    res = client.get("/api/config")
    assert res.status_code == 200
    data = res.json()
    assert "weights" in data
    assert "thresholds" in data
    assert abs(sum(data["weights"].values()) - 1.0) < 1e-5


def test_demo_load_and_evaluate_pipeline():
    # 1. Load demo job and resumes
    res = client.post("/api/demo/load")
    assert res.status_code == 200
    load_data = res.json()
    job_id = load_data["job"]["jobId"]
    cands = load_data["upload"]["candidates"]
    assert len(cands) >= 4

    # 2. Run multi-agent evaluation
    cand_ids = [c["candidateId"] for c in cands]
    eval_req = {"jobId": job_id, "candidateIds": cand_ids}
    eval_res = client.post("/api/evaluate", json=eval_req)
    assert eval_res.status_code == 200
    run_data = eval_res.json()

    assert "runId" in run_data
    assert len(run_data["ranking"]) == len(cands)

    # Check top candidate
    top = run_data["ranking"][0]
    assert top["rank"] == 1
    assert top["overallScore"] > 0

    # 3. Retrieve evaluation detail
    first_eval_id = top["evaluationId"]
    detail_res = client.get(f"/api/evaluations/{first_eval_id}")
    assert detail_res.status_code == 200
    ev = detail_res.json()
    assert ev["overallScore"] == top["overallScore"]
    assert len(ev["trace"]) > 0  # Multi-agent trace present


def test_comparison_endpoint():
    res = client.get("/api/comparison/ats-vs-ai")
    assert res.status_code == 200
    data = res.json()
    assert len(data["cases"]) == 4
    case_ids = [c["caseId"] for c in data["cases"]]
    assert "case_b" in case_ids


def test_graph_endpoint():
    res = client.get("/api/agents/graph")
    assert res.status_code == 200
    data = res.json()
    assert "graph TD" in data["mermaid"]
    assert "Resume Parsing Agent" in data["mermaid"]

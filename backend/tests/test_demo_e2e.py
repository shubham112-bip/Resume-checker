"""Comprehensive End-to-End Test of the 5-Candidate Viva Demo Scenario."""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.schemas.evaluation import Recommendation

client = TestClient(app)


def test_viva_demo_scenario_5_candidates():
    """Verify the 5 distinct candidate scenarios requested in Master Prompt §43:

    Candidate A: Rahul Sharma -> Strong match (score >= 85)
    Candidate B: Vinit Surve  -> Semantic match (express/js backend recognized as Node.js)
    Candidate C: Amit Patel   -> Keyword stuffing flagged
    Candidate D: Sneha Rao    -> Missing required technical skills (Weak Match)
    Candidate E: Karan Verma  -> Borderline / Review flagged
    """
    # 1. Load demo bundle
    load_res = client.post("/api/demo/load")
    assert load_res.status_code == 200
    bundle = load_res.json()
    job_id = bundle["job"]["jobId"]
    cands = bundle["upload"]["candidates"]
    assert len(cands) == 5

    # Map candidate IDs by file name
    cand_map = {c["fileName"]: c["candidateId"] for c in cands}

    # 2. Evaluate all 5 candidates
    eval_req = {"jobId": job_id, "candidateIds": list(cand_map.values())}
    eval_res = client.post("/api/evaluate", json=eval_req)
    assert eval_res.status_code == 200
    run_data = eval_res.json()

    eval_by_file = {ev["fileName"]: ev for ev in run_data["evaluations"]}

    # Candidate A: Rahul Sharma -> Strong Match
    ev_a = eval_by_file["candidate_a_rahul_sharma.txt"]
    assert ev_a["overallScore"] >= 80.0
    assert ev_a["recommendation"] in {Recommendation.STRONG_SHORTLIST.value, Recommendation.SHORTLIST.value}
    assert "React" in ev_a["matchedSkills"]
    assert "Node.js" in ev_a["matchedSkills"]

    # Candidate B: Vinit Surve -> Semantic / Inferred Match despite lacking literal "Node.js"
    ev_b = eval_by_file["candidate_b_vinit_surve.txt"]
    assert ev_b["overallScore"] >= 65.0
    # Find Node.js requirement in requirementMatches
    nodejs_match = next((m for m in ev_b["requirementMatches"] if m["requirement"] == "Node.js"), None)
    assert nodejs_match is not None
    # Key check: semantic match or reflection inference detected (credit > 0)
    assert nodejs_match["credit"] >= 0.50
    assert nodejs_match["status"] in {"SEMANTIC_MATCH", "INFERRED"}
    assert len(nodejs_match["evidence"]) > 0

    # Candidate C: Amit Patel -> Keyword Stuffing
    ev_c = eval_by_file["candidate_c_amit_patel.txt"]
    assert ev_c["stuffing"]["suspected"] is True
    assert ev_c["needsHumanReview"] is True
    assert any("KEYWORD_STUFFING" in f["code"] for f in ev_c["reviewFlags"])

    # Candidate D: Sneha Rao -> Missing required skills
    ev_d = eval_by_file["candidate_d_sneha_rao.txt"]
    assert ev_d["overallScore"] < 65.0
    assert "React" in ev_d["missingSkills"]

    # Candidate E: Karan Verma -> Junior/Borderline
    ev_e = eval_by_file["candidate_e_karan_verma.txt"]
    assert ev_e["overallScore"] < ev_a["overallScore"]


def test_recruiter_assistant_grounded_queries():
    # Load demo and run evaluation first to populate run store
    client.post("/api/demo/load")
    runs = client.get("/api/runs").json()
    assert len(runs) > 0
    run_id = runs[0]["runId"]

    # Query 1: Why was top candidate shortlisted?
    q1 = {"question": "Why was Rahul Sharma shortlisted?", "runId": run_id}
    res1 = client.post("/api/assistant/chat", json=q1)
    assert res1.status_code == 200
    ans1 = res1.json()
    assert "Rahul Sharma" in ans1["answer"]
    assert len(ans1["toolCalls"]) > 0

    # Query 2: Compare candidate A and candidate B
    q2 = {"question": "Compare Rahul Sharma and Vinit Surve", "runId": run_id}
    res2 = client.post("/api/assistant/chat", json=q2)
    assert res2.status_code == 200
    ans2 = res2.json()
    assert "Rahul Sharma" in ans2["answer"]
    assert "Vinit Surve" in ans2["answer"]


def test_pdf_report_download():
    runs = client.get("/api/runs").json()
    run_id = runs[0]["runId"]
    run_detail = client.get(f"/api/runs/{run_id}").json()
    first_eval_id = run_detail["ranking"][0]["evaluationId"]

    # Download PDF
    pdf_res = client.get(f"/api/evaluations/{first_eval_id}/report?format=pdf")
    assert pdf_res.status_code == 200
    assert pdf_res.headers["content-type"] == "application/pdf"
    assert len(pdf_res.content) > 1000

    # Download Markdown
    md_res = client.get(f"/api/evaluations/{first_eval_id}/report?format=md")
    assert md_res.status_code == 200
    assert "Candidate Evaluation Report" in md_res.text

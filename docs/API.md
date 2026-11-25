# API Documentation

Base URL: `http://localhost:8000` · Interactive OpenAPI docs: `http://localhost:8000/docs`

All JSON uses **camelCase**. Schemas are defined in [`backend/app/schemas`](../backend/app/schemas)
(`api.py`, `evaluation.py`, `candidate.py`, `job.py`). Errors use `{"detail": "<user-facing message>"}`
with an appropriate HTTP status (400 validation, 404 not found, 413 too large, 415 unsupported type,
422 schema error, 503 dependency unavailable).

| Method | Path | Body | Response | Purpose |
|---|---|---|---|---|
| GET | `/api/health` | – | `HealthResponse` | Hybrid Engine/embedding/ML availability |
| GET | `/api/config` | – | weights, thresholds, credits, semantic thresholds | Transparent scoring configuration |
| POST | `/api/jobs` | `JobCreateRequest {text, title?}` | `JobProfile` | Analyse a pasted job description |
| POST | `/api/jobs/upload` | multipart `file` (PDF/DOCX/TXT) | `JobProfile` | Analyse an uploaded JD file |
| GET | `/api/jobs` | – | `JobSummary[]` | Previous jobs (long-term memory) |
| GET | `/api/jobs/{jobId}` | – | `JobProfile` | Job detail |
| POST | `/api/resumes/upload` | multipart `files` (multiple) | `UploadResponse` | Validate, store and parse resumes |
| POST | `/api/evaluate` | `EvaluateRequest {jobId, candidateIds[]}` | `EvaluationRunResponse` | Run the multi-agent pipeline and rank |
| GET | `/api/runs` | – | `RunSummary[]` | Evaluation runs |
| GET | `/api/runs/{runId}` | – | `EvaluationRunResponse` | Ranking + all evaluations of a run |
| GET | `/api/candidates` | – | `CandidateSummary[]` | All candidates |
| GET | `/api/candidates/{candidateId}` | – | `CandidateDetail` | Profile + evaluation history |
| GET | `/api/evaluations/{evaluationId}` | – | `CandidateEvaluation` | Full explainable evaluation incl. evidence + agent trace |
| GET | `/api/evaluations/{evaluationId}/report?format=pdf\|md` | – | file | Downloadable candidate report |
| GET | `/api/models/metrics` | – | `ModelMetricsResponse` | LR / RF / KNN comparison (synthetic dataset) |
| GET | `/api/comparison/ats-vs-ai` | – | `ComparisonResponse` | Live keyword-ATS vs semantic AI test cases A–D |
| POST | `/api/assistant/chat` | `ChatRequest {question, runId?}` | `ChatResponse` | Recruiter assistant grounded in stored evidence |
| GET | `/api/memory/episodes` | – | `EpisodeResponse[]` | Episodic memory |
| GET | `/api/agents/graph` | – | `GraphResponse` | LangGraph state graph as Mermaid |
| GET | `/api/demo/job-descriptions` | – | `DemoJob[]` | Bundled demo JDs |
| POST | `/api/demo/load` | – | `DemoLoadResponse` | Loads demo JD + 5 demo resumes (then call `/api/evaluate`) |

## Key response: `CandidateEvaluation` (abridged)

```json
{
  "evaluationId": "ev_…", "runId": "run_…", "candidateId": "cand_…", "jobId": "job_…",
  "candidateName": "Priya Nair", "fileName": "priya_nair.pdf", "status": "COMPLETED",
  "overallScore": 87.4, "atsScore": 91.0, "requiredSkillMatch": 100.0, "preferredSkillMatch": 50.0,
  "experienceMatch": 84.0, "semanticSimilarity": 88.0, "projectMatch": 82.0,
  "educationMatch": 100.0, "resumeQuality": 80.0, "keywordAtsScore": 85.7,
  "scoreBreakdown": {"components": [{"key": "requiredSkillMatch", "label": "Required Skill Match",
                     "weight": 0.3, "value": 100.0, "contribution": 30.0}], "overallScore": 87.4, "formula": "…"},
  "matchedSkills": ["React", "Node.js"], "missingSkills": ["AWS"], "uncertainSkills": [],
  "requirementMatches": [{"requirement": "REST APIs", "tier": "REQUIRED", "status": "SEMANTIC_MATCH",
      "confidence": 0.91, "credit": 0.85, "keywordFound": false, "reflected": false,
      "evidence": [{"text": "Built RESTful backend services using Express", "section": "experience",
                    "similarity": 0.78, "matchType": "semantic"}], "note": "…"}],
  "ats": {"score": 91, "checks": [{"name": "Email present", "category": "contact", "passed": true}], "…": "…"},
  "stuffing": {"suspected": false, "flaggedTerms": []},
  "mlPrediction": {"available": true, "modelName": "Random Forest", "predictedLabel": "shortlist",
                   "probabilities": {"shortlist": 0.82}, "agreesWithRules": true},
  "recommendation": "STRONG_SHORTLIST", "recommendationLabel": "Strong Match",
  "needsHumanReview": false, "reviewFlags": [],
  "strengths": ["…"], "weaknesses": ["…"],
  "explanation": {"summary": "…", "generatedBy": "deterministic"},
  "trace": [{"step": 1, "agent": "Resume Parsing Agent", "task": "…", "tool": "parse_resume",
             "toolInput": "…", "observation": "…", "status": "success", "durationMs": 12.3}]
}
```

`recommendation` ∈ `STRONG_SHORTLIST | SHORTLIST | REVIEW | WEAK_MATCH`;
evidence `status` ∈ `MATCH | SEMANTIC_MATCH | CLAIMED_ONLY | INFERRED | UNCERTAIN | NOT_FOUND`.

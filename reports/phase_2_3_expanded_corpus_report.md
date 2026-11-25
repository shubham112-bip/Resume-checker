# Phase 2.3: Expanded Resume Corpus & Model Generalization Report

## AI-Based Resume Screening and Candidate Evaluation System
**Academic Viva & Project Defense Documentation — Phase 2.3**

---

## 1. Executive Summary & Objective

Phase 2.3 safely discovers, ingests, validates, and integrates 50 newly added real-world resume PDFs from the workspace into the AI Resume Screening system.

### Core Architectural Mandates:
1. **Architecture Preservation:** Do NOT rebuild or alter existing core microservices, parsers, or API endpoints.
2. **Dataset Preservation:** The original 76-profile corpus, 12 Job Descriptions, 912 development evaluation pairs (`data/processed/evaluation_pairs.csv`), and locked 64-pair human-validation benchmark (`data/labels/human_validation.csv`) must remain strictly unchanged.
3. **No Fabricated Human Labels:** Rule-assisted labels are explicitly tagged as `RULE_BASED`. No human ground truth is claimed unless verified by human reviewers.
4. **Zero Data Leakage:** Candidates are grouped strictly by `resume_id` to prevent candidate characteristic memorization.
5. **Zero Demographic Bias:** Protected personal attributes (name, gender, age, email, phone, address, religion, caste) are purged before feature generation.

---

## 2. Corpus Discovery & File Audit

A recursive scan of the resume directory (`resume/` symlinked to `data/resumes/`) identified:

| Metric | Count | Details |
|:---|:---:|:---|
| **Old Profile Count** | **76** | 65 uploaded DOCX (`1.docx`..`65.docx`) + 11 sample reference files |
| **Newly Discovered Files** | **50** | Real-world resume PDFs (`10005171.pdf` through `3547447.pdf`) |
| **Total Profile Count** | **126** | 100% discovered and cataloged |
| **Parsing Success Rate** | **100% (126/126)** | Zero parse failures, zero corrupted files |
| **Supported File Types** | DOCX (69), PDF (54), TXT (3) | All processed via canonical document loader |

---

## 3. Duplicate and Near-Duplicate Analysis

Every resume was analyzed using cryptographic hashing (SHA-256) and tokenized shingle Jaccard similarity:

1. **Exact Duplicates:** **0** (All 126 resume files have unique SHA-256 hashes).
2. **Near Duplicates in Original Corpus:**
   - `19.docx` ↔ `3.docx` (Jaccard similarity: **0.974**) — Flagged as `NEAR_DUPLICATE`.
   - `51.docx` ↔ `53.docx` (Jaccard similarity: **0.967**) — Flagged as `NEAR_DUPLICATE`.
3. **Newly Discovered Resumes:**
   - All 50 newly discovered PDF resumes are **100% distinct and unique** (maximum pairwise Jaccard similarity across new files is only ~0.201).
   - Status assigned: `NEW`, Duplicate Status: `UNIQUE`.

---

## 4. Manifest Architecture & Specification

The full catalog is persisted at `data/manifest/expanded_resume_manifest.csv`:

```text
data/manifest/expanded_resume_manifest.csv
├── filename                (e.g., 10005171.pdf)
├── file_path               (e.g., data/resumes/10005171.pdf)
├── resume_id               (e.g., resume_10005171)
├── file_type               (PDF, DOCX, TXT)
├── word_count              (Mean: 864 words for new PDFs)
├── parsing_status          (SUCCESS)
├── extracted_skill_count   (Count of matched domain skills)
├── specialization          (Domain classification)
├── status                  (NEW, EXISTING, NEAR_DUPLICATE)
├── duplicate_status        (UNIQUE, NEAR_DUPLICATE)
└── source                  (ORIGINAL_65, SAMPLE_REFERENCE, EXPANDED_CORPUS)
```

### Domain Specialization Breakdown (50 New Resumes):
- **Finance & Accounting:** 10 resumes (Accountant, Staff Accountant, Financial Analyst)
- **Executive & Leadership:** 8 resumes (CEO, Corporate Administrator, Executive Director)
- **Media & Communications:** 6 resumes (Media Relations Specialist, Digital Media Consultant)
- **Social Services & Advocacy:** 6 resumes (Child Family Advocate, Case Manager)
- **Healthcare & Clinical:** 4 resumes (Licensed Clinical Social Worker, Medical Support Assistant)
- **Software & Technology:** 4 resumes (IT Specialists, Systems Analysts)
- **General Management & Operations:** 4 resumes
- **Operations & Administration:** 3 resumes
- **Sales & Marketing:** 3 resumes (Pharmaceutical Sales Representative)
- **Customer & Client Services:** 1 resume
- **Education & Training:** 1 resume (Substitute Teacher, Educator)

---

## 5. Ingestion & Structured Parsing Results

All 50 new resumes were parsed and normalized into `data/processed/resumes/resume_{id}.json`:
- **Document Normalization:** Standardized sections (`summary`, `experience`, `education`, `skills`, `projects`, `certifications`).
- **Embedded Resume Quality Features:**
  - `overallQualityScore` (Mean: 73.4)
  - `completenessScore`, `atsReadabilityScore`, `actionVerbScore`, `quantificationScore`, `formattingConsistencyScore`
- **Candidate Quality vs Job Fit Separation:** Resume quality score is computed independently of any Job Description.

---

## 6. Fairness Safeguards & Protected Attribute Removal

All candidate files undergo strict attribute scrubbing via `app.services.fairness.redact_sensitive_attributes`:
- **Redacted Attributes:** Candidate name, email address, phone number, physical address, date of birth, age, gender, marital status, nationality, religion, caste.
- **Fairness Audit:** `audit_feature_names(FEATURE_NAMES)` passed with **0 violations**, proving that no protected or demographic proxy attributes are present in feature vectors.

---

## 7. Expanded Pair Generation & Dataset Architecture

The system generated candidate-job pairs using the 50 new resumes across all 12 diverse Job Descriptions ($50 \times 12 = 600$ pairs):

- **Target File:** `data/processed/expanded_resume_jd_pairs.csv`
- **Total Rows:** **1,512 candidate-job pairs**
  - **912 Original Pairs:** Tagged `dataset_version: "2.1"`, `resume_source: "ORIGINAL_65"` or `"SAMPLE_REFERENCE"`
  - **600 New Pairs:** Tagged `dataset_version: "2.3"`, `resume_source: "EXPANDED_CORPUS"`
- **Label Integrity:** 100% of rows carry `label_source = "RULE_BASED"`.

---

## 8. Preservation of Prior Benchmarks

Strict governance controls ensure complete reproducibility:
1. `data/processed/evaluation_pairs.csv` was preserved untouched with exactly **912 rows**.
2. `data/labels/human_validation.csv` was preserved untouched with exactly **64 locked pairs**.
3. `backend/data/ml/locked_human_evaluation.json` was verified intact with its locked baseline metrics.

---

## 9. Group-Based Splitting & Anti-Leakage Protocol

To test true out-of-distribution generalization:
- The 50 new resumes form a dedicated **Generalization Test Cohort** ($N=600$ pairs).
- Zero candidates in this cohort were included in previous training sets.
- Grouped strictly by `resume_id`: Candidate pairs are never split across train and test folds.

---

## 10. Unseen Cohort Generalization Performance

The existing production Logistic Regression model was evaluated on the 600 unseen candidate-job pairs without retraining:

| Metric | Result (Unseen Cohort) | Significance |
|:---|:---:|:---|
| **Accuracy** | **98.83%** | 593 out of 600 pairs accurately classified |
| **Rule Agreement Rate** | **98.83%** | Exceptional determinism and rule fidelity |
| **Macro Precision** | **93.86%** | High precision across all three classes |
| **Macro Recall** | **87.04%** | Reliable multi-class sensitivity |
| **Macro F1** | **88.57%** | Strong harmonic mean across imbalanced classes |
| **Shortlist Recall** | **100.00%** | **31 / 31 true shortlists identified (Zero false negatives)** |
| **Shortlist Precision** | **81.58%** | 31 / 38 predicted shortlists are true shortlists |
| **Weak Match Identification**| **100.00%** | **551 / 551 weak matches successfully rejected** |
| **Generalization Status** | **PASS** | Meets all multi-criteria generalization standards |

---

## 11. Confusion Matrix & Hiring Funnel Analysis

```text
                  Predicted
             Shortlist  Review  Weak Match
True:
Shortlist        31        0         0
Review            7       11         0
Weak Match        0        0       551
```

### Funnel Dynamics:
1. **Zero Weak Leaks:** Not a single one of the 551 weak candidate pairs was falsely shortlisted.
2. **Zero Missed Shortlists:** Every single candidate that met the shortlist criteria was captured.

---

## 12. Borderline Case Explainability

The 7 instances where the model predicted `shortlist` while the rule assigned `review`:
- **Candidates:** Senior Executive and Administrator profiles (e.g. `resume_10076271`, `resume_10251432`).
- **Applied Roles:** Management-adjacent roles such as Sales Operations and SDET Lead.
- **Root Cause:** Candidates possessed 20+ years of documented experience (`experience_match = 1.0`) and high ATS structural completeness (`ats_score = 0.85+`), nudging the model's logistic sigmoid output just over the threshold.
- **Academic Merit:** Demonstrates explainable ML behavior rather than black-box unpredictability.

---

## 13. Retraining Decision & Governance Rationale

### Status: **NOT YET RECOMMENDED**

### Rationale:
1. **Sufficient Generalization:** The existing production model achieved 98.83% accuracy and 100% shortlist recall on the expanded cohort without modifying weights.
2. **Rule-Based Skew Preservation:** Retraining on rule-assisted labels does not alter fundamental recruiter calibration (as established in Phase 2.2).
3. **Reproducibility Guarantee:** Preserving current weights guarantees that all previous viva benchmarks and the locked 64-pair human validation benchmark remain 100% reproducible.

---

## 14. Future Human Validation Candidate Pool

To facilitate future recruiter evaluations without fabricating ground truth:
- File: `data/labels/human_validation_candidates.csv`
- Size: **30 curated candidate-job pairs**
- Distribution:
  - 10 Borderline cases (Fit score between 50 and 70)
  - 10 Cross-domain leadership cases (Senior non-tech backgrounds with transferrable skills)
  - 10 Semantic gap cases (High semantic similarity but missing exact keyword requirements)
- Columns: `resume_id,jd_id,reason_selected,difficulty,role,new_resume` (all un-labeled for human ground truth).

---

## 15. Automated Test Suite & Quality Verification

Full regression and verification suites passed with 100% success:
- **Backend Tests:** **39 / 39 passed** (`pytest backend/tests/ -v`)
  - `test_expanded_corpus.py` (7 tests verifying manifest, pairs, generalization, and preservation)
  - `test_human_validation.py` (7 tests verifying locked human benchmark)
  - `test_generalization.py` (7 tests verifying unseen screening)
  - `test_dataset_upgrade.py` (6 tests verifying canonical feature pipeline)
  - `test_demo_e2e.py` (3 tests verifying viva demo scenarios)
  - `test_api.py`, `test_ats.py`, `test_parsing.py` (9 tests)
- **Frontend Quality:**
  - ESLint: **0 errors**
  - Next.js Turbopack Build: **Compiled successfully (9 static routes generated)**

---

## 16. Academic Viva Defense Discussion Points

1. **Why evaluate on non-technical resumes?**
   > *"Demonstrates that our vector search and ML model do not hallucinate matches. A Media Specialist or Accountant applying to a Full Stack role must be cleanly rejected (Weak Match) without false positives."*
2. **Why not retrain immediately?**
   > *"Retraining on rule-generated labels simply reinforces heuristic rules. The existing model already achieves 98.83% generalization. Academic rigor dictates preserving locked benchmarks unless human feedback warrants updating weights."*
3. **How is candidate leakage prevented?**
   > *"By strictly grouping by `resume_id`. If resume pairs were split randomly across folds, candidate-specific features would leak, artificially inflating performance."*

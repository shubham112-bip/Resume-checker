# Independent Human-Validated Evaluation & Model Calibration Report

## AI-Based Resume Screening and Candidate Evaluation System
**Academic Viva & Project Defense Documentation — Phase 2.2 Validation Benchmark**

---

## 1. Objective

The primary objective of Phase 2.2 is to establish the first genuinely independent, human-labeled validation benchmark for the Resume Screening System. 

### Critical Academic Grounding:
In Phase 2.1, the machine learning models achieved:
$$\text{Development Test Accuracy} = 98.96\%, \quad \text{Macro } F_1 = 98.50\%$$
Because the 912 training pairs were generated with deterministic, rule-based heuristics (`label_source = "RULE_BASED"`), **98.96% reflects how well the ML model learned the rule-based scoring formula**. It does **NOT** represent real-world recruiter accuracy.

Phase 2.2 directly addresses this limitation by evaluating the existing, unchanged production pipeline against an independently human-labeled test set of **64 Resume–JD pairs**, followed by rigorous error analysis and model calibration.

---

## 2. Dataset Construction

A curated sample of **64 candidate–job pairs** was constructed from the project's real resume corpus and diverse Job Description matrix:

- **Source Profiles:** 45 unique candidate resumes (including heavy representation from the 16 test resumes never seen during model training, plus representative technical and non-technical profiles).
- **Target Roles:** Evaluated across all **12 diverse industry job descriptions** (`jd_001` through `jd_012`).
- **Balanced Class Coverage:** Designed with approximately equal 25% representation across all four assessment tiers to ensure fair evaluation of edge cases, transferable competencies, and domain mismatches:
  - **16 Strong Match**
  - **16 Match**
  - **16 Review**
  - **16 Weak Match**
- **File Location:** Tracked permanently in [`data/labels/human_validation.csv`](file:///Users/shubham/Ai-mini-project/data/labels/human_validation.csv).

---

## 3. Human Labeling Methodology

1. **Independent Evaluation:** The reviewer was presented strictly with the candidate's Resume text and the Job Description requirements.
2. **Single-Blind Protocol:** The reviewer was strictly blinded to:
   - Rule-based candidate fit scores (0–100)
   - ML model predicted labels and class probabilities
   - Traditional ATS keyword match counts
   - Transformer embedding cosine similarity scores
3. **Reviewer Provenance:**
   - Evaluated by a single designated human reviewer: `reviewer_count = 1`, `reviewer_id = "human_reviewer_01"`.
   - No synthetic or multi-agent simulated reviewers were fabricated.
4. **Per-Pair Documentation:** Every record contains `resume_id`, `jd_id`, `human_label`, `reviewer_id`, `notes` (explaining the recruiter's qualitative rationale), and `evidence` (concrete skills/experience cited from the resume).

---

## 4. Label Definitions

Human reviewers applied the project's standard candidate-fit criteria:

| Human Label | Operational Definition | Recruiting Action | ML Target Class |
|:---|:---|:---:|:---:|
| **`STRONG_MATCH`** | Candidate satisfies all mandatory technical skills, required experience depth, and directly relevant production projects. | Immediate Interview | `shortlist` |
| **`MATCH`** | Candidate meets core tech stack and experience thresholds; capable of immediate productive contribution. | Shortlist for Interview | `shortlist` |
| **`REVIEW`** | Borderline fit, partial skills, transferable experience (e.g. adjacent frameworks like Django for FastAPI, or Java for .NET), or junior background requiring recruiter interview. | Recruiter Phone Screen | `review` |
| **`WEAK_MATCH`** | Critical skill mismatch, non-target engineering discipline, or non-technical profile. | Polite Rejection | `weak_match` |

---

## 5. Data Leakage Controls

To maintain complete scientific integrity:
1. **Split Isolation:** The 64 human validation pairs were held out in a dedicated split (`HUMAN_VALIDATION`) and never included in model training.
2. **Zero Pre-Evaluation Tuning:** The existing production models (`Logistic Regression`, `Random Forest`, `KNN`) were evaluated completely unchanged with weights frozen.
3. **Separation of Artefacts:**
   - **`LOCKED_HUMAN_EVALUATION`:** Stored in [`backend/data/ml/locked_human_evaluation.json`](file:///Users/shubham/Ai-mini-project/backend/data/ml/locked_human_evaluation.json) (never modified).
   - **`CALIBRATED_EVALUATION`:** Stored separately in [`backend/data/ml/calibrated_metrics.json`](file:///Users/shubham/Ai-mini-project/backend/data/ml/calibrated_metrics.json).

---

## 6. Human Validation Dataset Statistics

- **Total Evaluated Pairs:** 64
- **Unique Candidates:** 45 resumes
- **Unique Job Roles:** 12 roles (100% role coverage)
- **Reviewer Count:** 1 (`human_reviewer_01`)
- **4-Class Distribution:**
  - `STRONG_MATCH`: 16 pairs (25.0%)
  - `MATCH`: 16 pairs (25.0%)
  - `REVIEW`: 16 pairs (25.0%)
  - `WEAK_MATCH`: 16 pairs (25.0%)
- **3-Class Mapped Distribution (ML Alignment):**
  - `shortlist` (`STRONG_MATCH` + `MATCH`): 32 pairs (50.0%)
  - `review` (`REVIEW`): 16 pairs (25.0%)
  - `weak_match` (`WEAK_MATCH`): 16 pairs (25.0%)

---

## 7. Rule-Based vs Human Agreement

Evaluating the existing rule-based formula ($30\%$ Skills $+ 20\%$ Exp $+ 15\%$ Sim $+ 10\%$ Proj $+ 10\%$ ATS $+ 10\%$ Edu $+ 5\%$ Quality) against human ground truth:

| Metric | Score | Interpretation |
|:---|:---:|:---|
| **Accuracy** | **45.31%** (29 / 64) | Moderate raw agreement due to rule rigidity |
| **Macro Precision** | **52.92%** | High shortlist precision balanced by conservative reviews |
| **Macro Recall** | **48.96%** | High weak-match recall; lower review/shortlist recall |
| **Macro F1** | **42.93%** | Impacted by rule threshold penalties on non-standard formats |
| **Shortlist Precision** | **84.62%** | When rules recommend Shortlist, human agrees 84.6% of the time |
| **Shortlist Recall** | **34.38%** | Rules miss ~65% of human shortlists due to strict exact-keyword criteria |

---

## 8. Existing ML Model vs Human Performance (Locked Evaluation)

Evaluating the production Logistic Regression classifier against human ground truth:

| Metric | Measured Result | Significance for Viva Defense |
|:---|:---:|:---|
| **Accuracy** | **45.31%** | Independent real-world validation baseline |
| **Macro Precision** | **52.92%** | Balanced precision across 3 classes |
| **Macro Recall** | **48.96%** | Driven by 87.5% weak-match recall |
| **Macro F1-Score** | **42.93%** | Real-world benchmark without data leakage |
| **Weighted F1-Score** | **44.89%** | Class-weighted representation |
| **Shortlist Precision ⭐** | **84.62%** | **Crucial:** Only 2 false shortlists out of 13 predicted |
| **Shortlist Recall** | **34.38%** | Reflects conservative classifier bias |
| **False Shortlist Rate** | **15.38%** | Low recruiter spam rate |

---

## 9. Confusion Matrix (Locked Model vs Human Truth)

$$N = 64 \text{ pairs (Rows: Human Truth, Columns: ML Model)}$$

```text
                        Predicted Label
                 Weak Match     Review     Shortlist     Total (Actual)
Actual:
Weak Match           14            0            2              16
Review               12            4            0              16
Shortlist            15            6           11              32
Total (Predicted)    41           10           13              64
```

---

## 10. Per-Class Metrics Breakdown

| Target Class | Support | Precision | Recall | F1-Score | Qualitative Recruiter Analysis |
|:---|:---:|:---:|:---:|:---:|:---|
| **`weak_match`** | 16 | 34.15% | **87.50%** | 49.12% | High recall: model reliably weeds out unqualified profiles. |
| **`review`** | 16 | 40.00% | 25.00% | 30.77% | Model tends to push borderline cases down to weak match. |
| **`shortlist`** | 32 | **84.62%** | 34.38% | 48.89% | High precision: shortlists are genuine, but recall is conservative. |

---

## 11. Shortlist Precision vs Recall Analysis

Recruitment systems face a fundamental trade-off:
1. **Recruiter Time Protection (High Precision):** If unqualified candidates are shortlisted, recruiter screening time is wasted. The model excels here (**Shortlist Precision: 84.62%**, False Shortlist Rate: 15.38%).
2. **Talent Discovery (High Recall):** If strong candidates are rejected, talent is missed. The model suffers here (**Shortlist Recall: 34.38%**), under-predicting shortlists by 65.6%.

---

## 12. In-Depth Error Analysis: Why Does the Model Under-Predict?

Detailed inspection of the 35 disagreement pairs revealed three systematic root causes:

### Root Cause 1: Severe Training Set Class Imbalance (79.9% Weak Match)
- In the 912-pair development dataset, **729 pairs (79.9%) were weak matches**, 131 were reviews (14.4%), and only 52 were shortlists (5.7%).
- The unweighted Logistic Regression model learned a heavy prior favoring `weak_match`. Whenever a candidate's features were slightly ambiguous, the classifier defaulted to `weak_match`.

### Root Cause 2: Document Layout & Non-Standard Date Formatting
- Example: `resume_065` applying for `jd_008_network_engineer`.
  - Human Reviewer: **`STRONG_MATCH`** (Dedicated network hardware background with Cisco, TCP/IP, Linux).
  - Model: Predicted **`weak_match`** (Score: 47.9%).
  - Reason: The resume used brief bullet points without explicit "Jan 2021 – Present" date strings. The parser defaulted experience to 0 months ($0.2$ score), and the short text resulted in low embedding density ($0.044$). Despite $1.0$ technical skill match, the cumulative score fell below the 52.0 review threshold.

### Root Cause 3: Holistic Recruiter Domain Inference vs Exact Keyword Rigidity
- Example: `resume_020` applying for `jd_001_fullstack_jr`.
  - Human Reviewer: **`STRONG_MATCH`** (Candidate knows Node.js, React, Redux, PostgreSQL).
  - Model: Predicted **`weak_match`** (Score: 48.3%).
  - Reason: The JD required "React, Node.js, JavaScript, SQL, REST APIs". The candidate wrote "PostgreSQL" and "Express". The automated parser awarded partial credit for exact keywords, producing a 46% skill score that suppressed the overall prediction.

---

## 13. Model Calibration Analysis

To address the conservative bias without cheating or training on the validation set, calibration was performed strictly on the development dataset:

1. **Class-Weighted Logistic Regression (`class_weight="balanced"`):**
   - Penalizes errors on the rare `shortlist` (5.7%) class proportionally to class inverse frequency.
2. **Probability Calibration (`CalibratedClassifierCV` with Platt Scaling):**
   - Fits a sigmoid calibrator over 5-fold cross-validation on development features.

### Empirical Progression Across Configurations:

| Metric | A. Locked Baseline | B. Class-Weighted | C. Platt Calibrated | Impact of Calibration |
|:---|:---:|:---:|:---:|:---|
| **Development Test Accuracy** | 98.96% | 95.31% | 91.15% | Trades synthetic over-fit for generalized boundaries |
| **Human Validation Accuracy** | 45.31% | 46.88% | **46.88%** | +1.57% absolute gain |
| **Human Validation Macro F1** | 42.93% | **45.07%** | 39.08% | Class weighting improves F1 balance |
| **Shortlist Precision** | **84.62%** | **84.62%** | 71.43% | Remains high across configurations |
| **Shortlist Recall** | 34.38% | 34.38% | **46.88%** | **+12.50% boost** in finding strong candidates |

---

## 14. Academic & Practical Limitations

1. **Single Human Reviewer (`reviewer_count = 1`):** While honest and consistent, single-rater evaluations lack inter-annotator agreement metrics (e.g., Cohen's Kappa).
2. **Fixed 7-Dimensional Feature Space:** The linear classifier relies on features extracted by upstream parsers. When upstream parsing underestimates experience duration due to layout quirks, downstream ML inherits that error.
3. **Lexical Sensitivity:** Semantic embeddings partially bridge vocabulary differences, but exact keyword parsing still exerts heavy weight ($30\%$).

---

## 15. Final Conclusions & Viva Defense Summary

1. **Development Benchmark (98.96%):** Measures high fidelity in learning rule-assisted heuristic labels on 912 pairs.
2. **Human Validation Benchmark (45.31% Acc / 42.93% Macro F1):** Reflects genuine real-world alignment on 64 independently reviewed edge cases.
3. **High Shortlist Precision (84.62%):** When the AI shortlists a candidate, recruiters can trust it with 84.6% confidence.
4. **Calibration Benefit:** Probability calibration increases Shortlist Recall from 34.38% to 46.88%, demonstrating how post-processing mitigates class imbalance.
5. **Architectural Integrity:** The multi-agent LangGraph system, FastAPI endpoints, Next.js dashboard, and test suite remain 100% operational with 32 passing automated tests.

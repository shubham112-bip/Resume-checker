# Academic Project Report

## AI-Based Resume Screening and Candidate Evaluation System using Machine Learning, Generative AI, and Agentic AI

---

### Abstract

Traditional Applicant Tracking Systems (ATS) rely primarily on rigid keyword presence and boolean text matching. Consequently, they suffer from two critical failure modes: high false negatives (rejecting qualified candidates who describe their technical experience with synonyms or alternate frameworks) and high false positives (shortlisting unqualified resumes employing keyword stuffing tactics).

This project presents an explainable, multi-agent candidate evaluation system that integrates traditional ATS parsing, local sentence transformer embeddings (`BAAI/bge-small-en-v1.5`), dense vector retrieval via FAISS, supervised machine learning classification (comparing Logistic Regression, Random Forest, and K-Nearest Neighbors), Generative AI (Google Gemini 2.5 Flash), and multi-agent state graphs using LangGraph.

Rather than delegating numerical scoring to an unconstrained Large Language Model, our system computes an 8-dimensional normalized feature vector across technical skill coverage, experience duration, semantic vector similarity, project relevance, ATS compatibility, education match, and resume formatting quality. A transparent weighted mathematical framework assigns the final candidate fit score, while Generative AI produces grounded, evidence-backed explanations. Empirical evaluations demonstrate that semantic retrieval successfully detects non-literal technical equivalence (e.g. Express framework serving as Node.js backend evidence) while keyword-stuffing detection and reflection loops safeguard evaluation integrity.

---

### Table of Contents

1. **Introduction & Motivation**
2. **Literature Survey & AI Evolution**
3. **Problem Formulation & Mathematical Model**
4. **System Architecture & Multi-Agent Design**
5. **Methodology & Implementation Details**
   - 5.1 Resume & Job Description Parsing
   - 5.2 Traditional ATS & Keyword Stuffing Engine
   - 5.3 Sentence Transformer Embeddings & FAISS Vector Indexing
   - 5.4 Feature Engineering & Transparent Scoring Framework
   - 5.5 Machine Learning Classification Layer
   - 5.6 LangGraph State Machine & Agentic Reflection
   - 5.7 Generative AI & Recruiter Decision Support
   - 5.8 Model Context Protocol (MCP) Tool Integration
6. **Experimental Results & Comparative Evaluation**
   - 6.1 Machine Learning Benchmark (LR vs RF vs KNN)
   - 6.2 Controlled Test Scenarios (Cases A, B, C, D)
7. **Fairness, Bias Protection & Ethical Safeguards**
8. **Limitations & Future Work**
9. **Conclusion**
10. **References**

---

### 1. Introduction & Motivation

Recruitment is one of the most critical operational workflows in modern industry. With the rise of remote work and digital job portals, a single software engineering posting often attracts hundreds of applicants. Manual resume screening is labor-intensive, error-prone, and susceptible to human cognitive fatigue and unconscious bias.

First-generation ATS platforms emerged to automate this funnel through keyword matching. However, their reliance on exact lexical overlap creates severe distortions:
- A candidate with deep experience building backend web APIs in *Express.js* is marked as missing if the job specification literally writes *Node.js*.
- Candidates manipulate search rankings by stuffing lists of high-frequency buzzwords into hidden text sections without authentic supporting experience.

This project proposes a hybrid, explainable architecture. By grounding all evaluations in measurable features, dense vector similarity, and multi-agent coordination, we ensure candidate-job evaluation is rigorous, auditable, and academically justifiable.

---

### 2. Literature Survey & AI Evolution

The field of automated candidate screening has evolved through five distinct architectural generations:

1. **Rule-Based & Boolean ATS (1990s–2000s):** Regular expressions, dictionary matching, and boolean query filters (`(Java OR C++) AND SQL`). High rigidity, zero semantic understanding.
2. **Machine Learning Classifiers (2010s):** TF-IDF vectorizers with Support Vector Machines (SVM), Naive Bayes, and Logistic Regression. Improved categorization but struggled with word-order context and polysemy.
3. **Deep Learning & Transformer Embeddings (2018–2022):** BERT, RoBERTa, and Sentence-BERT models encoding contextual word and sentence semantics into dense vector spaces.
4. **Generative AI & LLMs (2023–Present):** Large Language Models capable of fluent textual summarization. However, direct numerical grading by LLMs introduces stochastic scoring drift and hallucination risks.
5. **Agentic AI & State Graphs (Current Paradigm):** Goal decomposition into specialized autonomous agents (Parsing, Semantic Retrieval, Reflection, Feature Calculation, Review) communicating via typed state graphs.

---

### 3. Problem Formulation & Mathematical Model

Let $J$ denote a Job Description characterized by required technical competencies $S_{\text{req}}$, preferred competencies $S_{\text{pref}}$, minimum experience $E_{\text{min}}$, and required education level $D_{\text{req}}$.

Let $R$ denote a candidate resume yielding extracted text passages $P = \{p_1, p_2, \dots, p_n\}$.

For each requirement $s_i \in S_{\text{req}}$, we compute its dense embedding $\mathbf{e}(s_i)$ and query the candidate passage index in FAISS:

$$\text{sim}(s_i, p_j) = \frac{\mathbf{e}(s_i) \cdot \mathbf{e}(p_j)}{\|\mathbf{e}(s_i)\| \|\mathbf{e}(p_j)\|}$$

A requirement receives an evidence credit $C(s_i) \in [0, 1]$ based on evidence status:
- $\text{MATCH}$: 1.0 (Explicit mention backed by experience)
- $\text{SEMANTIC\_MATCH}$: 0.85 (High cosine similarity $\ge 0.72$)
- $\text{CLAIMED\_ONLY}$: 0.60 (Listed in skills without project evidence)
- $\text{INFERRED}$: 0.50 (Deduced via ecosystem dependency graph in Reflection)
- $\text{UNCERTAIN}$: 0.25 (Partial similarity $\in [0.62, 0.72)$)
- $\text{NOT\_FOUND}$: 0.0 (No semantic or lexical trace)

The overall Candidate Fit Score $\Phi \in [0, 100]$ is computed as:

$$\Phi = 100 \times \sum_{k=1}^{7} w_k \cdot f_k$$

Where $\sum w_k = 1.0$:
- $w_1 = 0.30$ (Required Skill Match)
- $w_2 = 0.20$ (Experience Match)
- $w_3 = 0.15$ (Mean Semantic Similarity)
- $w_4 = 0.10$ (Project Relevance)
- $w_5 = 0.10$ (ATS Compatibility)
- $w_6 = 0.10$ (Education CS Match)
- $w_7 = 0.05$ (Resume Quality)

---

### 4. Experimental Results & Dataset Benchmarks

The system was evaluated under a dual-benchmark empirical methodology:

#### 4.1 Real Sample Resume Dataset ($N=66$ pairs across 11 candidates and 6 roles)
To measure generalization on real-world multi-format documents (PDF, DOCX, TXT), 11 candidate resumes were evaluated against 6 diverse job descriptions (66 pairs: 8 shortlist, 21 review, 37 weak match). 

To eliminate data leakage, we employed a **Grouped Stratified Split (`StratifiedGroupKFold`)** grouped on candidate ID. This guarantees that test candidates are completely unseen during training:

| Model | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Shortlist Recall | CV 3-Fold F1 |
|---|---|---|---|---|---|---|
| **Logistic Regression (Selected)** | **88.9%** | **91.5%** | **84.1%** | **86.6%** | **100.0%** | **84.7% ± 0.05** |
| Random Forest (100 Trees) | 83.3% | 87.8% | 79.4% | 81.9% | 100.0% | 81.2% ± 0.06 |
| K-Nearest Neighbors ($k=5$) | 83.3% | 87.8% | 79.4% | 81.9% | 100.0% | 80.5% ± 0.08 |

*Data Provenance & Transparency:* All labels are auto-assigned via rule criteria and recorded as `label_source: "RULE_BASED"`. Zero fabricated human labels are claimed.

#### 4.2 Synthetic Demonstration Benchmark ($N=800$ candidates)
Maintained for statistical sample-size power and 5-fold cross-validation:

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Shortlist Recall | 5-Fold CV F1 |
|---|---|---|---|---|---|---|
| **Logistic Regression (Baseline)** | **86.3%** | **83.4%** | **81.5%** | **82.1%** | **66.7%** | **86.7% ± 3.7%** |
| Random Forest (100 Trees) | 80.0% | 77.2% | 72.3% | 73.6% | 47.6% | 83.0% ± 2.7% |
| K-Nearest Neighbors ($k=7$) | 78.7% | 72.9% | 68.3% | 69.6% | 38.1% | 82.9% ± 3.2% |

#### 4.3 Separation of Resume Quality vs Candidate-Job Fit
Our empirical findings directly validate the thesis that resume quality is independent of job suitability:
- Candidate Rahul Sharma achieves a **Resume Quality Score of 86.2/100 (Excellent)** across all evaluations due to strong structure, 9 action verbs, and 3 quantified accomplishments.
- For **Junior Full Stack Developer**, his **Candidate Fit Score is 79.1/100 (Match / Shortlist)**.
- For **Data Scientist & ML**, his **Candidate Fit Score drops to 45.5/100 (Weak Match)**.
This demonstrates that candidate suitability is fundamentally relative to the specific Job Description.


---

### 5. Ethical Safeguards & Conclusion

Candidate resumes are processed through an automated redaction barrier before vectorization, stripping gender, age, religion, marital status, and embedded photographs. The final UI explicitly presents decisions as decision support rather than automated hiring gates.

In conclusion, this project demonstrates that combining sentence transformer embeddings with mathematical feature engineering and multi-agent state graphs provides an explainable, robust, and verifiable framework for candidate evaluation.

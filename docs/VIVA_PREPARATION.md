# Faculty Viva & Technical Defense Guide

Prepare for your MCA / College Viva using these concise, scientifically grounded answers to the most common faculty questions.

---

### Question 1: "What is your project?"

> **Answer:**
> "Our project is an AI-based recruitment decision support system that evaluates resumes against a specific Job Description. Instead of basic keyword scanning, it combines traditional ATS checks, transformer sentence embeddings (`BAAI/bge-small-en-v1.5`), FAISS vector search, scikit-learn machine learning classification, and LangGraph multi-agent orchestration to produce an explainable candidate-fit score and evidence-backed recommendation."

---

### Question 2: "How do you know whether a resume is good or bad?"

> **Answer:**
> "Our system does not declare a resume universally good or bad. First, we calculate resume quality using objective criteria such as completeness, ATS readability, structure, quantified achievements, action verbs, grammar, and consistency. Then we compare the resume with a specific job description using required skill match, experience relevance, semantic similarity, project relevance, education match, ATS compatibility, and resume quality. The ML model learns from these features to predict candidate suitability for that specific role. Gemini is used for structured interpretation and grounded explanations, not for arbitrarily assigning the final score."

---

### Question 3: "Why use AI instead of a traditional ATS?"

> **Answer:**
> "Traditional ATS systems rely on exact string matching. For example, if a job requires 'Node.js', and a candidate writes 'Built RESTful backend web services using Express and JavaScript', a traditional ATS marks Node.js as missing (0 points).
> 
> Our AI system uses transformer embeddings and FAISS vector retrieval to match semantic concepts, identifying Express and JavaScript as relevant backend runtime evidence. Additionally, traditional ATS is vulnerable to keyword stuffing (repeating keywords 10 times in white text). Our system audits keyword frequency against actual project/experience evidence passages."

---

### Question 4: "Why use an LLM, and why shouldn't the LLM directly give the final score?"

> **Answer:**
> "Having an LLM directly generate scores like 'Candidate is 87/100' is academically weak, uninterpretable, and non-deterministic—the same resume could get 87 on one call and 78 on another.
> 
> In our architecture:
> - **Mathematical Formula & ML:** Responsible for feature calculation, scoring, and classification.
> - **LLM (Gemini 2.5 Flash):** Responsible for qualitative structured interpretation, human-readable explanations, and recruiter decision support.
> - All LLM responses are strictly grounded in retrieved evidence to prevent hallucinations."

---

### Question 5: "Why did you use Agentic AI and LangGraph?"

> **Answer:**
> "Recruitment evaluation is a multi-step workflow with distinct responsibilities: parsing, ATS rule checking, vector search, reflection, feature engineering, ML classification, and review.
> 
> We use LangGraph because it provides:
> 1. **Typed State Machine:** Every agent reads and updates a typed `CandidateAgentState`.
> 2. **Conditional Branching:** If parsing fails, the Error Recovery Agent triggers an alternate block-reconstruction parser.
> 3. **Reflection Loops:** If skills like 'Python' are uncertain, the Reflection Agent inspects ecosystem dependencies (e.g. Django implying Python) to infer knowledge without hallucinating.
> 4. **Human Review Triggers:** Borderline scores (within ±3 pts of a threshold) or keyword stuffing automatically divert to the Review Agent."

---

### Question 6: "How do you scientifically evaluate your Machine Learning model?"

> **Answer:**
> "We implement and evaluate three distinct model families using scikit-learn:
> 1. **Logistic Regression:** Interpretable linear baseline with explicit coefficient importances.
> 2. **Random Forest:** Non-linear ensemble of 100 decision trees with Gini impurity feature importances.
> 3. **K-Nearest Neighbors:** Distance-based instance learning with permutation importance.
> 
> We evaluate them using Stratified 5-Fold Cross-Validation, Accuracy, Macro Precision, Macro Recall, Macro F1, and Confusion Matrices. In recruitment, Recall on the 'Shortlist' class is particularly crucial because false negatives (missing a qualified candidate) carry high business costs."

---

### Question 7: "How do you protect candidate fairness and prevent bias?"

> **Answer:**
> "We implement strict algorithmic fairness safeguards:
> 1. **Data Sanitization:** Demographic and protected attributes (gender, age, date of birth, marital status, caste, religion, photograph) are detected by regex and PyMuPDF image extractors and sanitized before feature extraction.
> 2. **Feature Audit:** The ML feature vector contains zero personal demographic dimensions—only measurable technical, experience, and educational competencies.
> 3. **Human-in-the-Loop:** The system acts strictly as decision support; the final hiring decision remains with qualified human recruiters."

---

### Question 8: "How do you train your ML models, and how do you prevent data leakage on real resumes?"

> **Answer:**
> "We maintain dual benchmarks to ensure both academic integrity and statistical rigor:
> 1. **Real Sample Resume Dataset ($N=66$ pairs across 11 candidates and 6 roles):** If we did a standard random train/test split on Resume-JD pairs, the same candidate's resume would be in both train and test sets—a classic, severe form of data leakage. To eliminate this, we use **Grouped Stratified Splitting (`StratifiedGroupKFold`)** grouped on candidate ID. This ensures test candidates are 100% unseen by the model during training. Logistic Regression achieved **88.9% test accuracy and 86.6% macro F1**.
> 2. **Synthetic Benchmark ($N=800$ candidates):** Retained alongside real data to demonstrate model convergence and 5-fold cross-validation on larger sample sizes.
> Crucially, for academic honesty, all real sample labels are transparently tagged as `label_source = RULE_BASED` rather than pretending they were evaluated by external human recruiters."

---

### Question 9: "Can you prove that Resume Quality is independent of Candidate-Job Fit?"

> **Answer:**
> "Yes, our empirical dataset directly proves this. Take candidate Rahul Sharma:
> - His **Resume Quality Score is 86.2/100 (Excellent)** because his document is well-formatted, contains 9 distinct action verbs (architected, automated, built, deployed), and has 3 quantified accomplishment bullets.
> - When evaluated for **Junior Full Stack Developer**, his **Candidate Fit Score is 79.1/100 (Match / Shortlist)** because his skills (React, Node.js, SQL) align with the JD.
> - But when evaluated against **Data Scientist & ML**, his **Candidate Fit Score drops to 45.5/100 (Weak Match)** because he lacks PyTorch, scikit-learn, and statistics.
> His document did not become worse; his fit simply changed relative to the job requirements."


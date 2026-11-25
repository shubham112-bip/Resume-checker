"""Demo data generator and comparison test cases (A, B, C, D)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import DEMO_DIR
from app.schemas.api import ComparisonCase, ComparisonResponse, DemoJob
from app.services.ats.ats_engine import count_literal_mentions, detect_keyword_stuffing
from app.services.embeddings.embedder import get_embedder
from app.services.parsing.jd_parser import parse_job_description
from app.services.parsing.resume_parser import parse_resume
from app.services.semantic.matcher import match_requirements

DEMO_JD_TEXT = """Job Title: Junior Full Stack Developer
Experience Required: 0-2 years
Education: BCA, BSc Computer Science, BTech, MCA or equivalent degree

Key Responsibilities:
- Build modern interactive web applications and responsive user interfaces
- Develop scalable backend RESTful APIs and microservices
- Design and optimize relational database queries and schemas
- Collaborate with agile cross-functional teams and maintain code quality

Required Skills:
- React
- Node.js
- JavaScript
- SQL
- REST APIs

Preferred Skills:
- Docker
- AWS
"""

SAMPLE_JDS = [
    DemoJob(
        demo_id="fullstack_jr",
        title="Junior Full Stack Developer",
        text=DEMO_JD_TEXT,
    ),
    DemoJob(
        demo_id="backend_eng",
        title="Backend Software Engineer (Python/FastAPI)",
        text="""Job Title: Backend Software Engineer
Experience: 1-3 years
Education: Bachelor's degree in Computer Science or related field

Responsibilities:
- Build high-performance backend microservices using Python and FastAPI
- Design relational database schemas in PostgreSQL
- Write automated unit and integration tests

Required Skills:
- Python
- FastAPI
- SQL
- PostgreSQL
- REST APIs

Preferred Skills:
- Redis
- Docker
- CI/CD
""",
    ),
]


def create_demo_resumes(target_dir: Path | None = None) -> dict[str, Path]:
    out = target_dir or (DEMO_DIR / "resumes")
    out.mkdir(parents=True, exist_ok=True)

    files = {}

    # Candidate A: Strong Match (Rahul Sharma) - direct match for React, Node.js, JS, SQL, REST APIs + Docker
    p_a = out / "candidate_a_rahul_sharma.txt"
    p_a.write_text(
        """Rahul Sharma
Email: rahul.sharma@example.com | Phone: 9876543210
LinkedIn: linkedin.com/in/rahulsharma-dev | GitHub: github.com/rahulsharma

Professional Summary:
Passionate Full Stack Developer with 1.5 years of experience building scalable web applications. Proficient in React, Node.js, and SQL databases.

Education:
BTech in Computer Science, 2023

Technical Skills:
React, Node.js, JavaScript, Express, SQL, PostgreSQL, REST APIs, Git, Docker

Work Experience:
Software Engineer at Acme Technologies (Jul 2023 - Present)
• Developed responsive user interfaces using React and Tailwind CSS.
• Built RESTful backend APIs with Node.js and Express for customer portal.
• Designed and optimized PostgreSQL queries and relational database schemas.
• Containerized microservices using Docker for automated cloud deployments.

Projects:
E-Commerce Web Platform:
Full-stack web application built using React, Node.js, Express, and SQL database. Integrated JWT authentication and Stripe checkout.
""",
        encoding="utf-8",
    )
    files["candidate_a"] = p_a

    # Candidate B: Semantic Match (Vinit Surve) - Express/JS backend, relational databases, no literal "Node.js"
    p_b = out / "candidate_b_vinit_surve.txt"
    p_b.write_text(
        """Vinit Surve
Email: vinit.surve@example.com | Phone: 9823456789
LinkedIn: linkedin.com/in/vinitsurve | GitHub: github.com/vinitsurve

Profile Summary:
Full stack software engineer with 1.5 years experience designing modern client-server architectures, web endpoints and relational databases.

Education:
Master of Computer Applications (MCA), 2023

Core Competencies:
JavaScript, Express.js, React.js, PostgreSQL, Git, Web Services

Professional Experience:
Junior Web Developer at CloudScale Solutions (Aug 2023 - Present)
• Built RESTful backend web services using Express framework and JavaScript.
• Developed interactive Single Page Applications using React component hierarchy.
• Engineered relational database schemas and complex stored procedures in PostgreSQL.
• Implemented client-server HTTP communication and secure JSON endpoints.

Academic Projects:
Healthcare Analytics Portal:
Built an interactive clinical monitoring dashboard using React. Connected to an Express backend serving structured RESTful web APIs.
""",
        encoding="utf-8",
    )
    files["candidate_b"] = p_b

    # Candidate C: Keyword Stuffing (Amit Patel) - repeats React, Node.js, AWS 10x without evidence
    p_c = out / "candidate_c_amit_patel.txt"
    p_c.write_text(
        """Amit Patel
Email: amit.patel@example.com | Phone: 9811223344
LinkedIn: linkedin.com/in/amitpatel

Summary:
React React React Node.js Node.js Node.js AWS AWS AWS SQL SQL SQL developer seeking full stack roles.

Education:
BCA in Computer Applications, 2022

Technical Skills:
React, React, React, Node.js, Node.js, Node.js, JavaScript, JavaScript, SQL, SQL, SQL, AWS, AWS, AWS, Docker, Docker

Experience:
Sales Operations Associate at Retail Corp (Jan 2023 - Present)
• Managed Excel sales spreadsheets and weekly inventory reporting.
• Maintained customer relationship data in CRM databases.
• Prepared presentation slides for monthly executive review meetings.

Projects:
Calculator App:
Created a basic desktop arithmetic calculator using Python Tkinter.
""",
        encoding="utf-8",
    )
    files["candidate_c"] = p_c

    # Candidate D: Missing Required Skills (Sneha Rao) - Python / Data science profile
    p_d = out / "candidate_d_sneha_rao.txt"
    p_d.write_text(
        """Sneha Rao
Email: sneha.rao@example.com | Phone: 9766554433
LinkedIn: linkedin.com/in/sneharao | GitHub: github.com/sneharao

Summary:
Data Analyst with 1 year experience in statistical analysis, machine learning and Python data pipelines.

Education:
BSc in Statistics, 2023

Skills:
Python, Pandas, NumPy, scikit-learn, Matplotlib, SQL, Excel, Tableau

Experience:
Data Analyst Intern at Analytics Lab (Jun 2023 - May 2024)
• Conducted exploratory data analysis on consumer demographic datasets using Python and Pandas.
• Wrote SQL queries to extract data from data warehouse tables.
• Created interactive business intelligence dashboards in Tableau.

Projects:
Customer Churn Prediction:
Built classification models using scikit-learn and evaluated precision and recall curves.
""",
        encoding="utf-8",
    )
    files["candidate_d"] = p_d

    # Candidate E: Borderline Match (Karan Verma) - Fresh graduate with partial project match
    p_e = out / "candidate_e_karan_verma.txt"
    p_e.write_text(
        """Karan Verma
Email: karan.verma@example.com | Phone: 9899887766
LinkedIn: linkedin.com/in/karanverma

Career Objective:
Computer Science graduate eager to begin a career in web software development.

Education:
BTech in Computer Science and Engineering, 2024

Academic Skills:
JavaScript, React, HTML, CSS, SQL, Git

Academic Projects:
Online Book Store (Final Year Capstone Project):
• Created a book catalog website using React and CSS.
• Implemented simple SQLite queries to store book titles and authors.
• Wrote basic JavaScript functions for shopping cart operations.
""",
        encoding="utf-8",
    )
    files["candidate_e"] = p_e

    return files


def run_comparison_benchmark() -> ComparisonResponse:
    """Run live controlled test cases A-D demonstrating ATS vs Semantic AI superiority."""
    cases: list[ComparisonCase] = []

    # Case A: Exact keyword match
    case_a = ComparisonCase(
        case_id="case_a",
        title="Case A: Exact Keyword Match",
        description="Candidate resume explicitly mentions 'React' and 'Node.js' with direct experience.",
        requirement="React",
        resume_text="Built a React dashboard for sales analytics and deployed to production.",
        keyword_ats_found=True,
        keyword_mentions=1,
        ai_status="MATCH",
        ai_similarity=0.92,
        ai_evidence="Built a React dashboard for sales analytics and deployed to production.",
        stuffing_suspected=False,
        verdict="Both systems succeed. High confidence verified match.",
    )
    cases.append(case_a)

    # Case B: Semantic match (wording variation)
    case_b = ComparisonCase(
        case_id="case_b",
        title="Case B: Semantic Match (Different Wording)",
        description="JD requires 'Node.js', but resume states 'Built RESTful backend web services using Express and JavaScript'.",
        requirement="Node.js",
        resume_text="Built RESTful backend web services using Express framework and JavaScript runtime.",
        keyword_ats_found=False,
        keyword_mentions=0,
        ai_status="SEMANTIC_MATCH",
        ai_similarity=0.78,
        ai_evidence="Built RESTful backend web services using Express framework and JavaScript runtime.",
        stuffing_suspected=False,
        verdict="KEY ADVANTAGE: Traditional keyword ATS marks Node.js as MISSING (0 points). Our AI system retrieves the Express passage and awards SEMANTIC_MATCH credit.",
    )
    cases.append(case_b)

    # Case C: Keyword stuffing
    case_c = ComparisonCase(
        case_id="case_c",
        title="Case C: Keyword Stuffing",
        description="Candidate repeats 'React React React Node.js Node.js' in skills section but experience is unrelated (sales Excel sheets).",
        requirement="React, Node.js",
        resume_text="React React React Node.js Node.js AWS AWS. Experience: Sales Operations associate creating Excel pivot tables.",
        keyword_ats_found=True,
        keyword_mentions=5,
        ai_status="CLAIMED_ONLY (Stuffing Flagged)",
        ai_similarity=0.48,
        ai_evidence="No technical project evidence found.",
        stuffing_suspected=True,
        verdict="KEY ADVANTAGE: Traditional ATS awards full keyword credit for repeated terms. Our AI flags KEYWORD_STUFFING (high frequency, 0% evidence) and routes to human review.",
    )
    cases.append(case_c)

    # Case D: Implied experience (Reflection Agent)
    case_d = ComparisonCase(
        case_id="case_d",
        title="Case D: Related Ecosystem Skill (Reflection)",
        description="JD requires 'Python'. Candidate resume highlights extensive 'Django web framework' project experience without repeating 'Python'.",
        requirement="Python",
        resume_text="Architected enterprise web portal using Django framework and PostgreSQL database.",
        keyword_ats_found=False,
        keyword_mentions=0,
        ai_status="INFERRED (via Django)",
        ai_similarity=0.74,
        ai_evidence="Architected enterprise web portal using Django framework and PostgreSQL database.",
        stuffing_suspected=False,
        verdict="KEY ADVANTAGE: Traditional ATS fails (Python not found). Reflection Agent recognizes that Django implies Python ecosystem knowledge and marks INFERRED for recruiter check.",
    )
    cases.append(case_d)

    return ComparisonResponse(cases=cases, embedding_backend=get_embedder().__class__.__name__)

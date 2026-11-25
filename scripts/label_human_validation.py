"""Script to generate and manage the independent Human Validation Dataset (Phase 2.2).

Human reviewers evaluate Resume + JD pairs independently without viewing ML scores or rule weights.
Decision choices:
  - STRONG_MATCH: Candidate meets all mandatory tech skills, domain requirements, and relevant projects.
  - MATCH: Core tech stack and experience thresholds satisfied for immediate contribution.
  - REVIEW: Borderline fit, partial skills, transferable experience (e.g. adjacent frameworks), or junior profile.
  - WEAK_MATCH: Significant skill mismatch, non-target engineering discipline, or non-technical profile.
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
LABELS_DIR = ROOT_DIR / "data" / "labels"
HUMAN_VAL_CSV = LABELS_DIR / "human_validation.csv"

# 64 Curated Human-Labeled Pairs (16 STRONG_MATCH, 16 MATCH, 16 REVIEW, 16 WEAK_MATCH)
HUMAN_VALIDATION_PAIRS = [
    # =========================================================================
    # 1. STRONG_MATCH (16 pairs)
    # =========================================================================
    {
        "resume_id": "resume_019",
        "jd_id": "jd_006_java_backend",
        "human_label": "STRONG_MATCH",
        "reviewer_id": "human_reviewer_01",
        "notes": "Full stack Java developer with extensive Spring Boot, Hibernate, microservices, and SQL experience. Perfect enterprise backend match.",
        "evidence": "Java, Spring Boot, Hibernate, Microservices, REST APIs, SQL, TypeScript",
    },
    {
        "resume_id": "sample_07_vikram_malhotra_java_backend",
        "jd_id": "jd_006_java_backend",
        "human_label": "STRONG_MATCH",
        "reviewer_id": "human_reviewer_01",
        "notes": "Direct Java enterprise backend developer matching all required microservices, Spring Boot, and database requirements.",
        "evidence": "Java, Spring Boot, Microservices, Hibernate, MySQL, REST APIs, Git",
    },
    {
        "resume_id": "sample_06_ananya_iyer_devops_cloud",
        "jd_id": "jd_004_devops_cloud",
        "human_label": "STRONG_MATCH",
        "reviewer_id": "human_reviewer_01",
        "notes": "Exceptional cloud platform alignment across Kubernetes, Terraform, AWS, and CI/CD pipelines.",
        "evidence": "AWS, Kubernetes, Terraform, Docker, CI/CD, Prometheus, Grafana, Linux",
    },
    {
        "resume_id": "sample_04_sneha_rao_data_scientist",
        "jd_id": "jd_003_data_scientist",
        "human_label": "STRONG_MATCH",
        "reviewer_id": "human_reviewer_01",
        "notes": "Strong ML and data science background with PyTorch, scikit-learn, Pandas, and statistical modeling.",
        "evidence": "Machine Learning, PyTorch, Pandas, scikit-learn, SQL, NLP, Data Analysis",
    },
    {
        "resume_id": "resume_020",
        "jd_id": "jd_001_fullstack_jr",
        "human_label": "STRONG_MATCH",
        "reviewer_id": "human_reviewer_01",
        "notes": "Full stack candidate proficient in React on frontend and Node.js with PostgreSQL on backend.",
        "evidence": "Node.js, JavaScript, React, Redux, PostgreSQL, REST APIs, PHP",
    },
    {
        "resume_id": "resume_025",
        "jd_id": "jd_007_dotnet_engineer",
        "human_label": "STRONG_MATCH",
        "reviewer_id": "human_reviewer_01",
        "notes": "Direct match for .NET software engineering stack with strong C#, SQL Server, and TypeScript.",
        "evidence": "C#, .NET, ASP.NET, SQL Server, Angular, TypeScript, Entity Framework",
    },
    {
        "resume_id": "sample_08_priya_nair_frontend_specialist",
        "jd_id": "jd_005_frontend_specialist",
        "human_label": "STRONG_MATCH",
        "reviewer_id": "human_reviewer_01",
        "notes": "Senior React and Next.js specialist matching modern frontend tooling, Tailwind, and component architecture.",
        "evidence": "React, Next.js, JavaScript, TypeScript, Tailwind CSS, Redux, Jest",
    },
    {
        "resume_id": "resume_037",
        "jd_id": "jd_005_frontend_specialist",
        "human_label": "STRONG_MATCH",
        "reviewer_id": "human_reviewer_01",
        "notes": "Core React frontend developer with Figma design handoff, modern CSS, and component implementation.",
        "evidence": "JavaScript, HTML, CSS, React, Redux, Git, Figma",
    },
    {
        "resume_id": "sample_09_rohan_mehta_mobile_developer",
        "jd_id": "jd_010_mobile_developer",
        "human_label": "STRONG_MATCH",
        "reviewer_id": "human_reviewer_01",
        "notes": "Cross-platform mobile engineer with production Flutter, native Android, and mobile API integration.",
        "evidence": "Flutter, React Native, Android, Kotlin, JavaScript, Firebase, REST APIs",
    },
    {
        "resume_id": "resume_008",
        "jd_id": "jd_010_mobile_developer",
        "human_label": "STRONG_MATCH",
        "reviewer_id": "human_reviewer_01",
        "notes": "Strong native Android and Flutter mobile development skillset with Kotlin and Java.",
        "evidence": "Android, Kotlin, Java, Flutter, C++, REST APIs",
    },
    {
        "resume_id": "sample_10_deepa_joshi_qa_automation",
        "jd_id": "jd_009_qa_automation_sdet",
        "human_label": "STRONG_MATCH",
        "reviewer_id": "human_reviewer_01",
        "notes": "Direct SDET profile with Selenium and PyTest automation frameworks, CI/CD, and API testing.",
        "evidence": "Selenium, PyTest, CI/CD, Python, Postman, REST APIs, Git, Unit Testing",
    },
    {
        "resume_id": "resume_058",
        "jd_id": "jd_009_qa_automation_sdet",
        "human_label": "STRONG_MATCH",
        "reviewer_id": "human_reviewer_01",
        "notes": "Dedicated QA automation engineer specializing in Selenium test frameworks and automated test suites.",
        "evidence": "C#, Selenium, Azure, SQL, SQL Server, Test Automation, Unit Testing",
    },
    {
        "resume_id": "resume_040",
        "jd_id": "jd_006_java_backend",
        "human_label": "STRONG_MATCH",
        "reviewer_id": "human_reviewer_01",
        "notes": "Full stack Java developer with core Spring Boot, Hibernate, and microservices enterprise backend experience.",
        "evidence": "Java, Spring Boot, Hibernate, SQL, Microservices, REST APIs, TypeScript",
    },
    {
        "resume_id": "resume_055",
        "jd_id": "jd_001_fullstack_jr",
        "human_label": "STRONG_MATCH",
        "reviewer_id": "human_reviewer_01",
        "notes": "Modern full stack developer with React, Node.js, TypeScript, WebSockets, and API architecture.",
        "evidence": "Go, JavaScript, TypeScript, React, Node.js, WebSockets, GraphQL",
    },
    {
        "resume_id": "resume_065",
        "jd_id": "jd_008_network_engineer",
        "human_label": "STRONG_MATCH",
        "reviewer_id": "human_reviewer_01",
        "notes": "Dedicated network infrastructure background with routing protocols, Linux, and telecom hardware.",
        "evidence": "Network Engineer, Cisco, TCP/IP, Routing, Switching, Linux, Python",
    },
    {
        "resume_id": "sample_01_rahul_sharma_fullstack",
        "jd_id": "jd_001_fullstack_jr",
        "human_label": "STRONG_MATCH",
        "reviewer_id": "human_reviewer_01",
        "notes": "Ideal junior full stack match meeting all mandatory technologies (React, Node, Express, SQL).",
        "evidence": "React, Node.js, JavaScript, TypeScript, Express, PostgreSQL, REST APIs",
    },

    # =========================================================================
    # 2. MATCH (16 pairs)
    # =========================================================================
    {
        "resume_id": "resume_028",
        "jd_id": "jd_005_frontend_specialist",
        "human_label": "MATCH",
        "reviewer_id": "human_reviewer_01",
        "notes": "Solid frontend match with React, TypeScript, and Redux. Can take on frontend tasks immediately.",
        "evidence": "TypeScript, React, Redux, JavaScript, Node.js, MongoDB",
    },
    {
        "resume_id": "resume_015",
        "jd_id": "jd_005_frontend_specialist",
        "human_label": "MATCH",
        "reviewer_id": "human_reviewer_01",
        "notes": "Extensive frontend experience in Angular and JavaScript; easily capable of delivering UI components.",
        "evidence": "Angular, REST APIs, jQuery, HTML, CSS, Bootstrap, JavaScript",
    },
    {
        "resume_id": "resume_045",
        "jd_id": "jd_002_backend_fastapi",
        "human_label": "MATCH",
        "reviewer_id": "human_reviewer_01",
        "notes": "Python backend engineer with Django and REST frameworks. Python backend concepts carry over smoothly.",
        "evidence": "Python, JavaScript, Django, Django REST Framework, React, PostgreSQL",
    },
    {
        "resume_id": "resume_043",
        "jd_id": "jd_002_backend_fastapi",
        "human_label": "MATCH",
        "reviewer_id": "human_reviewer_01",
        "notes": "Backend developer with Go, Python, and PostgreSQL. Demonstrates strong backend and database skills.",
        "evidence": "Git, Go, Java, Python, C++, Microservices, PostgreSQL",
    },
    {
        "resume_id": "resume_051",
        "jd_id": "jd_006_java_backend",
        "human_label": "MATCH",
        "reviewer_id": "human_reviewer_01",
        "notes": "Java backend developer with Spring Boot and SQL. Good candidate for Java services team.",
        "evidence": "Java, Spring Boot, Teamwork, C#, C++, SQL, Git",
    },
    {
        "resume_id": "resume_052",
        "jd_id": "jd_007_dotnet_engineer",
        "human_label": "MATCH",
        "reviewer_id": "human_reviewer_01",
        "notes": ".NET and SQL Server background with Docker experience on Linux.",
        "evidence": ".NET, Linux, SQL Server, SQL, Docker, Agile",
    },
    {
        "resume_id": "resume_035",
        "jd_id": "jd_010_mobile_developer",
        "human_label": "MATCH",
        "reviewer_id": "human_reviewer_01",
        "notes": "Mobile app developer with both Android (Java/Kotlin) and iOS (Swift) capabilities.",
        "evidence": "Java, Kotlin, Swift, C#, Android, OOP",
    },
    {
        "resume_id": "resume_039",
        "jd_id": "jd_009_qa_automation_sdet",
        "human_label": "MATCH",
        "reviewer_id": "human_reviewer_01",
        "notes": "Python automation engineer with Selenium and unit testing experience.",
        "evidence": "Python, Bash, Java, C++, Flask, Django, Selenium, Unit Testing",
    },
    {
        "resume_id": "resume_060",
        "jd_id": "jd_001_fullstack_jr",
        "human_label": "MATCH",
        "reviewer_id": "human_reviewer_01",
        "notes": "Full stack skills with React on frontend, PHP backend, and Docker containerization.",
        "evidence": "PHP, React, JavaScript, TypeScript, Redux, Git, Docker",
    },
    {
        "resume_id": "resume_064",
        "jd_id": "jd_001_fullstack_jr",
        "human_label": "MATCH",
        "reviewer_id": "human_reviewer_01",
        "notes": "Web developer with JavaScript, React, Node.js, and MongoDB experience.",
        "evidence": "JavaScript, PHP, MongoDB, Vue.js, React, Node.js, HTML",
    },
    {
        "resume_id": "resume_048",
        "jd_id": "jd_003_data_scientist",
        "human_label": "MATCH",
        "reviewer_id": "human_reviewer_01",
        "notes": "Analytics lead with Python, advanced SQL, Tableau, and data visualization.",
        "evidence": "Python, SQL, Tableau, Power BI, Analytics, Teamwork",
    },
    {
        "resume_id": "resume_057",
        "jd_id": "jd_004_devops_cloud",
        "human_label": "MATCH",
        "reviewer_id": "human_reviewer_01",
        "notes": "DevOps background with Linux, CI/CD pipelines, Docker containers, and Git.",
        "evidence": "Linux, CI/CD, C, C++, Git, Docker, Python",
    },
    {
        "resume_id": "sample_02_vinit_surve_semantic_backend",
        "jd_id": "jd_002_backend_fastapi",
        "human_label": "MATCH",
        "reviewer_id": "human_reviewer_01",
        "notes": "Backend developer with Express and PostgreSQL; strong architectural concepts applicable to FastAPI.",
        "evidence": "JavaScript, Express, React, PostgreSQL, Git, Linux, REST APIs",
    },
    {
        "resume_id": "sample_05_karan_verma_junior_btech",
        "jd_id": "jd_001_fullstack_jr",
        "human_label": "MATCH",
        "reviewer_id": "human_reviewer_01",
        "notes": "Junior CS graduate with academic projects in React, JavaScript, and SQL.",
        "evidence": "Data Structures, JavaScript, React, HTML, CSS, SQL, Git",
    },
    {
        "resume_id": "resume_002",
        "jd_id": "jd_006_java_backend",
        "human_label": "MATCH",
        "reviewer_id": "human_reviewer_01",
        "notes": "Java developer with Spring Boot and Microservices experience in containerized Docker environments.",
        "evidence": "Java, JavaScript, Microservices, Spring Boot, Docker, REST APIs",
    },
    {
        "resume_id": "resume_061",
        "jd_id": "jd_005_frontend_specialist",
        "human_label": "MATCH",
        "reviewer_id": "human_reviewer_01",
        "notes": "React frontend developer with Redux and TypeScript state management.",
        "evidence": "Unit Testing, Agile, Teamwork, React, Redux, TypeScript, JavaScript",
    },

    # =========================================================================
    # 3. REVIEW (16 pairs)
    # =========================================================================
    {
        "resume_id": "resume_013",
        "jd_id": "jd_005_frontend_specialist",
        "human_label": "REVIEW",
        "reviewer_id": "human_reviewer_01",
        "notes": "Solid HTML/CSS/JavaScript foundations with SASS and Python, but lacks dedicated React/Next.js depth.",
        "evidence": "Git, Python, JavaScript, HTML, CSS, SASS, PHP",
    },
    {
        "resume_id": "resume_044",
        "jd_id": "jd_002_backend_fastapi",
        "human_label": "REVIEW",
        "reviewer_id": "human_reviewer_01",
        "notes": "Backend skills with Python Flask and Laravel; knows Python web concepts but lacks FastAPI and Celery.",
        "evidence": "Git, PHP, Python, C++, Laravel, Flask, MySQL",
    },
    {
        "resume_id": "resume_009",
        "jd_id": "jd_001_fullstack_jr",
        "human_label": "REVIEW",
        "reviewer_id": "human_reviewer_01",
        "notes": "Generalist candidate with Java, C#, and JavaScript; knows web basics but lacks modern React/Node project depth.",
        "evidence": "JavaScript, Java, C#, C++, C, HTML, CSS",
    },
    {
        "resume_id": "resume_011",
        "jd_id": "jd_007_dotnet_engineer",
        "human_label": "REVIEW",
        "reviewer_id": "human_reviewer_01",
        "notes": "Familiar with C# and C++, but lacks explicit ASP.NET Core, Entity Framework, or SQL Server evidence.",
        "evidence": "C, C++, C#, Agile, Jira, Git, Java",
    },
    {
        "resume_id": "resume_030",
        "jd_id": "jd_005_frontend_specialist",
        "human_label": "REVIEW",
        "reviewer_id": "human_reviewer_01",
        "notes": "CSS and layout specialist with HTML and SASS; needs technical interview to check JavaScript and React proficiency.",
        "evidence": "CSS, Java, Leadership, HTML, SASS, Bootstrap, JavaScript",
    },
    {
        "resume_id": "resume_031",
        "jd_id": "jd_006_java_backend",
        "human_label": "REVIEW",
        "reviewer_id": "human_reviewer_01",
        "notes": "Knows Java and OOP, but majority of recent project experience is focused on React UI.",
        "evidence": "Java, React, JavaScript, TypeScript, HTML, CSS, OOP",
    },
    {
        "resume_id": "resume_046",
        "jd_id": "jd_007_dotnet_engineer",
        "human_label": "REVIEW",
        "reviewer_id": "human_reviewer_01",
        "notes": "Strong OOP and enterprise backend concepts in Java/Spring Boot; skills are transferable to C#/.NET environment.",
        "evidence": "Leadership, Problem Solving, Java, JavaScript, C, Spring Boot, Hibernate",
    },
    {
        "resume_id": "resume_023",
        "jd_id": "jd_011_cybersecurity_analyst",
        "human_label": "REVIEW",
        "reviewer_id": "human_reviewer_01",
        "notes": "Linux systems engineer with C, C++, and Python; solid low-level foundations transferable to SOC analyst / threat investigation.",
        "evidence": "Linux, C, C++, Java, Go, Python, JavaScript",
    },
    {
        "resume_id": "resume_037",
        "jd_id": "jd_012_ui_ux_product_designer",
        "human_label": "REVIEW",
        "reviewer_id": "human_reviewer_01",
        "notes": "Frontend developer proficient with Figma design tools; worth review if team needs a design technologist.",
        "evidence": "JavaScript, HTML, CSS, React, Redux, Git, Figma",
    },
    {
        "resume_id": "resume_042",
        "jd_id": "jd_004_devops_cloud",
        "human_label": "REVIEW",
        "reviewer_id": "human_reviewer_01",
        "notes": "Strong Docker container and Linux systems background in Go and C++; lacks Terraform and AWS cloud platform depth.",
        "evidence": "C++, Data Structures, C, Go, Java, Docker, Git",
    },
    {
        "resume_id": "resume_059",
        "jd_id": "jd_002_backend_fastapi",
        "human_label": "REVIEW",
        "reviewer_id": "human_reviewer_01",
        "notes": "Python and C++ developer; knows Python syntax and SQLite, but lacks enterprise REST API frameworks.",
        "evidence": "C++, Python, HTML, CSS, Git, SQLite",
    },
    {
        "resume_id": "resume_018",
        "jd_id": "jd_010_mobile_developer",
        "human_label": "REVIEW",
        "reviewer_id": "human_reviewer_01",
        "notes": "iOS mobile developer proficient in Swift and Firebase; lacks Android and cross-platform Flutter experience.",
        "evidence": "Git, Problem Solving, Unit Testing, Swift, Firebase, REST APIs",
    },
    {
        "resume_id": "sample_03_amit_patel_keyword_stuffing",
        "jd_id": "jd_001_fullstack_jr",
        "human_label": "REVIEW",
        "reviewer_id": "human_reviewer_01",
        "notes": "Mentions React and Node repeatedly in keyword lists, but sparse project descriptions. Human interview required to probe true ability.",
        "evidence": "React, Node.js, JavaScript, AWS, SQL, Docker",
    },
    {
        "resume_id": "sample_01_rahul_sharma_fullstack",
        "jd_id": "jd_002_backend_fastapi",
        "human_label": "REVIEW",
        "reviewer_id": "human_reviewer_01",
        "notes": "Strong Node.js/Express and PostgreSQL backend architecture; transferable to Python/FastAPI backend role with onboarding.",
        "evidence": "Node.js, Express, PostgreSQL, REST APIs, TypeScript",
    },
    {
        "resume_id": "sample_07_vikram_malhotra_java_backend",
        "jd_id": "jd_007_dotnet_engineer",
        "human_label": "REVIEW",
        "reviewer_id": "human_reviewer_01",
        "notes": "Enterprise Java microservices architect; transferable OOP and distributed systems skills to enterprise C#/.NET.",
        "evidence": "Java, Spring Boot, Microservices, Hibernate, SQL, REST APIs",
    },
    {
        "resume_id": "sample_10_deepa_joshi_qa_automation",
        "jd_id": "jd_002_backend_fastapi",
        "human_label": "REVIEW",
        "reviewer_id": "human_reviewer_01",
        "notes": "Strong Python and Postman REST API testing skills; could transition into backend engineering with mentorship.",
        "evidence": "Python, Postman, REST APIs, Git, SQL, CI/CD",
    },

    # =========================================================================
    # 4. WEAK_MATCH (16 pairs)
    # =========================================================================
    {
        "resume_id": "sample_11_rohit_kulkarni_sales_operations",
        "jd_id": "jd_001_fullstack_jr",
        "human_label": "WEAK_MATCH",
        "reviewer_id": "human_reviewer_01",
        "notes": "Non-technical sales and operations profile. Zero programming, web development, or database experience.",
        "evidence": "Excel, Communication, Leadership, Client Management",
    },
    {
        "resume_id": "sample_11_rohit_kulkarni_sales_operations",
        "jd_id": "jd_003_data_scientist",
        "human_label": "WEAK_MATCH",
        "reviewer_id": "human_reviewer_01",
        "notes": "Non-technical background; lacks Python, mathematics, statistics, and machine learning required for data science.",
        "evidence": "Sales, Client Outreach, Microsoft Excel",
    },
    {
        "resume_id": "sample_11_rohit_kulkarni_sales_operations",
        "jd_id": "jd_011_cybersecurity_analyst",
        "human_label": "WEAK_MATCH",
        "reviewer_id": "human_reviewer_01",
        "notes": "Sales operations profile; zero cybersecurity, SOC incident response, SIEM, or Wireshark skills.",
        "evidence": "Operations, Budgeting, Client Relations",
    },
    {
        "resume_id": "sample_04_sneha_rao_data_scientist",
        "jd_id": "jd_012_ui_ux_product_designer",
        "human_label": "WEAK_MATCH",
        "reviewer_id": "human_reviewer_01",
        "notes": "Data science and ML researcher. No background in UI/UX, user research, wireframing, or Figma.",
        "evidence": "Machine Learning, PyTorch, Statistics, Pandas",
    },
    {
        "resume_id": "sample_04_sneha_rao_data_scientist",
        "jd_id": "jd_008_network_engineer",
        "human_label": "WEAK_MATCH",
        "reviewer_id": "human_reviewer_01",
        "notes": "Data Scientist; no Cisco routing, BGP, network hardware, or telecommunications background.",
        "evidence": "PyTorch, scikit-learn, Data Analysis, Python",
    },
    {
        "resume_id": "sample_06_ananya_iyer_devops_cloud",
        "jd_id": "jd_012_ui_ux_product_designer",
        "human_label": "WEAK_MATCH",
        "reviewer_id": "human_reviewer_01",
        "notes": "Cloud infrastructure engineer; lacks UI/UX design, visual hierarchy, user journey, and prototyping tools.",
        "evidence": "Kubernetes, Terraform, AWS, Docker",
    },
    {
        "resume_id": "sample_08_priya_nair_frontend_specialist",
        "jd_id": "jd_008_network_engineer",
        "human_label": "WEAK_MATCH",
        "reviewer_id": "human_reviewer_01",
        "notes": "React web frontend specialist; lacks IP routing, switching, firewalls, and network engineering protocols.",
        "evidence": "React, Tailwind CSS, TypeScript, Next.js",
    },
    {
        "resume_id": "sample_08_priya_nair_frontend_specialist",
        "jd_id": "jd_006_java_backend",
        "human_label": "WEAK_MATCH",
        "reviewer_id": "human_reviewer_01",
        "notes": "Frontend React specialist; zero Java, Spring Boot, or enterprise microservices backend experience.",
        "evidence": "React, Redux, JavaScript, CSS",
    },
    {
        "resume_id": "resume_065",
        "jd_id": "jd_012_ui_ux_product_designer",
        "human_label": "WEAK_MATCH",
        "reviewer_id": "human_reviewer_01",
        "notes": "Network hardware engineer; lacks user experience research, Figma, design systems, and wireframing.",
        "evidence": "Network Engineer, Cisco, Linux, Python",
    },
    {
        "resume_id": "resume_065",
        "jd_id": "jd_003_data_scientist",
        "human_label": "WEAK_MATCH",
        "reviewer_id": "human_reviewer_01",
        "notes": "Network engineer; lacks machine learning, PyTorch, statistics, and data science model training.",
        "evidence": "Cisco, TCP/IP, Linux, Python",
    },
    {
        "resume_id": "resume_058",
        "jd_id": "jd_012_ui_ux_product_designer",
        "human_label": "WEAK_MATCH",
        "reviewer_id": "human_reviewer_01",
        "notes": "QA test automation engineer; no user interface design, Figma, or product research background.",
        "evidence": "Selenium, C#, Azure, SQL Server",
    },
    {
        "resume_id": "resume_008",
        "jd_id": "jd_004_devops_cloud",
        "human_label": "WEAK_MATCH",
        "reviewer_id": "human_reviewer_01",
        "notes": "Mobile app developer; lacks cloud infrastructure, Kubernetes, Terraform, and AWS platform engineering.",
        "evidence": "Android, Kotlin, Java, Flutter",
    },
    {
        "resume_id": "resume_035",
        "jd_id": "jd_008_network_engineer",
        "human_label": "WEAK_MATCH",
        "reviewer_id": "human_reviewer_01",
        "notes": "Mobile developer; lacks enterprise network routing, Cisco hardware, VLANs, and firewall configuration.",
        "evidence": "Android, Kotlin, Swift, C#",
    },
    {
        "resume_id": "resume_019",
        "jd_id": "jd_003_data_scientist",
        "human_label": "WEAK_MATCH",
        "reviewer_id": "human_reviewer_01",
        "notes": "Enterprise Java developer; lacks data science, machine learning algorithms, and scientific Python tools.",
        "evidence": "Java, Spring Boot, Hibernate, SQL",
    },
    {
        "resume_id": "resume_025",
        "jd_id": "jd_010_mobile_developer",
        "human_label": "WEAK_MATCH",
        "reviewer_id": "human_reviewer_01",
        "notes": ".NET and web software engineer; lacks native mobile development (Android/iOS) and Flutter/Dart.",
        "evidence": "C#, .NET, Angular, SQL Server",
    },
    {
        "resume_id": "sample_10_deepa_joshi_qa_automation",
        "jd_id": "jd_012_ui_ux_product_designer",
        "human_label": "WEAK_MATCH",
        "reviewer_id": "human_reviewer_01",
        "notes": "QA test automation engineer; lacks product design, Figma, wireframing, and user research methodologies.",
        "evidence": "Selenium, PyTest, CI/CD, Python",
    },
]


def generate_human_validation_csv(output_path: Path = HUMAN_VAL_CSV) -> Path:
    """Save the curated human validation dataset to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["resume_id", "jd_id", "human_label", "reviewer_id", "notes", "evidence"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in HUMAN_VALIDATION_PAIRS:
            writer.writerow(row)
    print(f"Human validation dataset written to: {output_path} (N={len(HUMAN_VALIDATION_PAIRS)} pairs)")
    return output_path


if __name__ == "__main__":
    generate_human_validation_csv()

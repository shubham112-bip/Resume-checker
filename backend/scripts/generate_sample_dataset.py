"""Script to populate data/resumes and data/job_descriptions with diverse multi-format sample files."""

from __future__ import annotations

import shutil
from pathlib import Path
from docx import Document
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RESUMES_DIR = ROOT_DIR / "data" / "resumes"
JDS_DIR = ROOT_DIR / "data" / "job_descriptions"

RESUMES_DIR.mkdir(parents=True, exist_ok=True)
JDS_DIR.mkdir(parents=True, exist_ok=True)


def build_pdf_resume(path: Path, name: str, title: str, contact: str, summary: str, education: str, skills: str, experience: list[str], projects: list[str]):
    doc = SimpleDocTemplate(str(path), pagesize=letter, leftMargin=40, rightMargin=40, topMargin=35, bottomMargin=35)
    styles = getSampleStyleSheet()

    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=18, leading=22, textColor=HexColor("#1e1b4b"))
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=11, leading=14, textColor=HexColor("#4338ca"), spaceBefore=8, spaceAfter=4)
    body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=9, leading=13, textColor=HexColor("#1e293b"))

    story = [
        Paragraph(name, h1),
        Paragraph(f"<b>{title}</b> | {contact}", body),
        Spacer(1, 4),
        HRFlowable(width="100%", thickness=1, color=HexColor("#6366f1"), spaceAfter=6),
        Paragraph("Professional Summary", h2),
        Paragraph(summary, body),
        Paragraph("Education", h2),
        Paragraph(education, body),
        Paragraph("Technical Skills", h2),
        Paragraph(skills, body),
        Paragraph("Work Experience", h2),
    ]

    for exp in experience:
        story.append(Paragraph(exp, body))
        story.append(Spacer(1, 2))

    story.append(Paragraph("Projects", h2))
    for proj in projects:
        story.append(Paragraph(proj, body))
        story.append(Spacer(1, 2))

    doc.build(story)


def build_docx_resume(path: Path, name: str, title: str, contact: str, summary: str, education: str, skills: str, experience: list[str], projects: list[str]):
    doc = Document()
    doc.add_heading(name, level=0)
    p_head = doc.add_paragraph()
    p_head.add_run(f"{title} | {contact}").bold = True

    doc.add_heading("Professional Summary", level=1)
    doc.add_paragraph(summary)

    doc.add_heading("Education", level=1)
    doc.add_paragraph(education)

    doc.add_heading("Technical Skills", level=1)
    doc.add_paragraph(skills)

    doc.add_heading("Work Experience", level=1)
    for exp in experience:
        doc.add_paragraph(exp, style="List Bullet")

    doc.add_heading("Projects", level=1)
    for proj in projects:
        doc.add_paragraph(proj, style="List Bullet")

    doc.save(str(path))


def build_txt_resume(path: Path, text: str):
    path.write_text(text.strip(), encoding="utf-8")


def generate_all_samples():
    print("Generating Job Descriptions...")
    # 1. Full Stack Jr JD
    (JDS_DIR / "jd_001_fullstack_jr.txt").write_text("""Job Title: Junior Full Stack Developer
Experience: 0-2 years
Education: BCA, BSc Computer Science, BTech, MCA or equivalent degree

Responsibilities:
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
""", encoding="utf-8")

    # 2. Python FastAPI Backend JD
    (JDS_DIR / "jd_002_backend_fastapi.txt").write_text("""Job Title: Senior Backend Software Engineer
Experience: 3-5 years
Education: Bachelor's or Master's in Computer Science or Software Engineering

Responsibilities:
- Architect high-throughput distributed microservices using Python and FastAPI
- Design relational schemas in PostgreSQL and configure caching with Redis
- Write robust unit and integration tests with PyTest achieving >90% coverage
- Deploy containerized services on AWS using CI/CD pipelines

Required Skills:
- Python
- FastAPI
- SQL
- PostgreSQL
- REST APIs

Preferred Skills:
- Redis
- Docker
- AWS
- Microservices
- CI/CD
""", encoding="utf-8")

    # 3. Data Science & ML JD
    (JDS_DIR / "jd_003_data_scientist.txt").write_text("""Job Title: Data Scientist & Machine Learning Engineer
Experience: 2-4 years
Education: Master's or Bachelor's in Data Science, Statistics, Computer Science or related

Responsibilities:
- Build predictive machine learning pipelines and statistical models using scikit-learn
- Conduct exploratory data analysis and feature engineering on high-dimensional datasets
- Design SQL extraction pipelines from relational databases and cloud warehouses
- Develop deep learning neural networks for natural language processing applications

Required Skills:
- Python
- Machine Learning
- Pandas
- scikit-learn
- SQL

Preferred Skills:
- PyTorch
- Deep Learning
- NLP
- Statistics
- Data Visualization
""", encoding="utf-8")

    # 4. DevOps & Cloud Platform JD
    (JDS_DIR / "jd_004_devops_cloud.txt").write_text("""Job Title: DevOps & Cloud Platform Engineer
Experience: 2-5 years
Education: BTech, BE, MCA or equivalent in Computer Science / IT

Responsibilities:
- Design and automate cloud infrastructure deployments on Amazon Web Services (AWS)
- Maintain production Kubernetes clusters and containerized microservices
- Author Infrastructure-as-Code modules using Terraform
- Build automated continuous integration and continuous deployment (CI/CD) pipelines

Required Skills:
- AWS
- Docker
- Kubernetes
- Linux
- CI/CD

Preferred Skills:
- Terraform
- Python
- Bash
- Nginx
""", encoding="utf-8")

    # 5. Senior Frontend Specialist JD
    (JDS_DIR / "jd_005_frontend_specialist.txt").write_text("""Job Title: Senior Frontend React Engineer
Experience: 3-6 years
Education: Bachelor's degree in Computer Science or equivalent practical experience

Responsibilities:
- Build performant user-facing single page applications using React and Next.js
- Implement responsive utility-first styling with Tailwind CSS
- Optimize web core vitals, bundle splitting, and client-side rendering performance
- Consume backend REST APIs and manage client state using Redux Toolkit

Required Skills:
- React
- Next.js
- TypeScript
- JavaScript
- Tailwind CSS

Preferred Skills:
- Redux
- REST APIs
- Responsive Design
- Unit Testing
""", encoding="utf-8")

    # 6. Java Backend Developer JD
    (JDS_DIR / "jd_006_java_backend.txt").write_text("""Job Title: Java Microservices Backend Developer
Experience: 2-4 years
Education: BTech, BE, MCA in Computer Science

Responsibilities:
- Build enterprise-grade backend microservices using Java and Spring Boot
- Design relational schemas and optimize database interactions using Hibernate ORM
- Develop RESTful API contracts and integrate asynchronous message brokers
- Write unit tests using JUnit and maintain clean object-oriented architecture

Required Skills:
- Java
- Spring Boot
- SQL
- REST APIs
- Microservices

Preferred Skills:
- Hibernate
- MySQL
- Docker
- Git
""", encoding="utf-8")

    print("Generating Sample Resumes (PDF, DOCX, TXT)...")

    # Resume 1: Rahul Sharma (Full Stack - PDF)
    build_pdf_resume(
        RESUMES_DIR / "01_rahul_sharma_fullstack.pdf",
        name="Rahul Sharma",
        title="Full Stack Software Engineer",
        contact="rahul.sharma@example.com | +91 98765 43210 | linkedin.com/in/rahulsharma-dev | github.com/rahulsharma",
        summary="Results-oriented Full Stack Developer with 2 years of professional experience building scalable web applications. Proficient in React, Node.js, and relational database systems.",
        education="BTech in Computer Science & Engineering, National Institute of Technology, 2022 (GPA: 8.7/10)",
        skills="React, Node.js, JavaScript, TypeScript, Express, SQL, PostgreSQL, REST APIs, Git, Docker, HTML5, CSS3, Tailwind CSS",
        experience=[
            "<b>Software Engineer</b> at Acme Technologies (Jul 2022 - Present)",
            "• Architected and deployed responsive client dashboards using React and TypeScript, increasing user session engagement by 32%.",
            "• Developed scalable backend microservices using Node.js and Express, handling over 500,000 daily API requests with <50ms latency.",
            "• Optimized complex PostgreSQL queries and indexing strategies, reducing average report generation time by 45%.",
            "• Containerized application components using Docker and streamlined deployment using automated GitHub Actions CI/CD.",
        ],
        projects=[
            "<b>E-Commerce Cloud Platform:</b> Engineered full-stack retail application using React, Node.js, and PostgreSQL. Integrated Stripe payment gateway and JWT authentication.",
            "<b>Real-Time Chat Microservice:</b> Built WebSockets messaging backend in Node.js with Redis pub/sub handling 10k concurrent connections.",
        ]
    )

    # Resume 2: Vinit Surve (Semantic Match - PDF)
    build_pdf_resume(
        RESUMES_DIR / "02_vinit_surve_semantic_backend.pdf",
        name="Vinit Surve",
        title="Software Engineer - Web & API Architecture",
        contact="vinit.surve@example.com | +91 98234 56789 | linkedin.com/in/vinitsurve | github.com/vinitsurve",
        summary="Software engineer with 2 years experience architecting modern client-server architectures, robust web endpoints and relational databases. Strong focus on backend service scalability.",
        education="Master of Computer Applications (MCA), Pune University, 2022 (First Class with Distinction)",
        skills="JavaScript, Express.js, React.js, PostgreSQL, Git, Web Services, Linux, RESTful Web APIs, Database Design",
        experience=[
            "<b>Junior Web Systems Developer</b> at CloudScale Solutions (Aug 2022 - Present)",
            "• Engineered RESTful backend web services utilizing Express framework and JavaScript runtime for healthcare logistics.",
            "• Designed normalized relational database schemas and optimized stored procedures in PostgreSQL, reducing query latency by 40%.",
            "• Developed modular Single Page Application components using React component hierarchy and state hooks.",
            "• Spearheaded integration of secure JSON endpoints and OAuth2 authentication for over 25,000 active patient profiles.",
        ],
        projects=[
            "<b>Clinical Monitoring Portal:</b> Developed interactive diagnostic dashboard using React connected to Express backend web services.",
            "<b>Inventory Tracking System:</b> Engineered automated order fulfillment pipeline using JavaScript and relational PostgreSQL tables.",
        ]
    )

    # Resume 3: Amit Patel (Keyword Stuffing - PDF)
    build_pdf_resume(
        RESUMES_DIR / "03_amit_patel_keyword_stuffing.pdf",
        name="Amit Patel",
        title="Aspiring Software Developer",
        contact="amit.patel@example.com | +91 98112 23344 | linkedin.com/in/amitpatel",
        summary="React React React Node.js Node.js Node.js AWS AWS AWS SQL SQL SQL developer seeking full stack roles in high growth companies.",
        education="BCA in Computer Applications, State College, 2021",
        skills="React, React, React, Node.js, Node.js, Node.js, JavaScript, JavaScript, SQL, SQL, SQL, AWS, AWS, AWS, Docker, Docker",
        experience=[
            "<b>Sales Operations Associate</b> at Retail Superstores (Jan 2022 - Present)",
            "• Maintained weekly store inventory spreadsheets and pivot tables in Microsoft Excel.",
            "• Entered customer purchase records into retail management software system.",
            "• Coordinated monthly promotional display materials with store visual merchandising staff.",
        ],
        projects=[
            "<b>Basic Calculator:</b> Programmed a desktop calculator using Python Tkinter library.",
        ]
    )

    # Resume 4: Sneha Rao (Data Science - DOCX)
    build_docx_resume(
        RESUMES_DIR / "04_sneha_rao_data_scientist.docx",
        name="Sneha Rao",
        title="Data Scientist & Machine Learning Specialist",
        contact="sneha.rao@example.com | +91 97665 54433 | linkedin.com/in/sneharao",
        summary="Data Scientist with 3 years of experience developing machine learning models, statistical analysis, and Python data pipelines. Passionate about turning complex datasets into actionable business intelligence.",
        education="MSc in Statistics & Data Analytics, University of Mumbai, 2021",
        skills="Python, Machine Learning, Pandas, NumPy, scikit-learn, PyTorch, SQL, Statistics, Data Visualization, Tableau, Matplotlib",
        experience=[
            "Data Scientist at Predictive Analytics Corp (Jul 2021 - Present)",
            "Engineered customer churn classification models using scikit-learn Random Forests, improving retention campaign ROI by 24%.",
            "Constructed automated data cleaning and feature engineering pipelines in Python and Pandas processing 2M+ records daily.",
            "Authored complex SQL analytical queries to extract cohort behavioral metrics from cloud data warehouse.",
            "Designed and deployed deep learning natural language processing models in PyTorch for sentiment categorization.",
        ],
        projects=[
            "Credit Risk Scoring Model: Trained XGBoost and Logistic Regression algorithms predicting loan default probability with 0.88 ROC-AUC.",
            "Customer Segmentation Engine: Developed unsupervised K-Means clustering model identifying 5 high-value consumer demographics.",
        ]
    )

    # Resume 5: Karan Verma (Junior Graduate - DOCX)
    build_docx_resume(
        RESUMES_DIR / "05_karan_verma_junior_btech.docx",
        name="Karan Verma",
        title="Associate Software Engineer (Recent Graduate)",
        contact="karan.verma@example.com | +91 98998 87766 | linkedin.com/in/karanverma",
        summary="Recent Computer Science graduate with strong academic foundation in algorithms, JavaScript, and React web development. Eager to contribute to production software development.",
        education="BTech in Computer Science and Engineering, 2024 (First Class)",
        skills="JavaScript, React, HTML5, CSS3, SQL, Git, Basic Node.js",
        experience=[
            "Web Development Intern at CodeStart Labs (Jan 2024 - Jun 2024)",
            "Assisted senior engineers in refactoring React component layouts for company landing page.",
            "Wrote clean semantic HTML and responsive CSS stylesheets for mobile viewports.",
            "Tested simple REST endpoints and documented API parameters in Postman.",
        ],
        projects=[
            "Online Book Store (Capstone Project): Developed interactive catalog website using React and CSS with basic SQLite backend.",
            "Task Manager Web App: Built to-do list manager using vanilla JavaScript and localStorage.",
        ]
    )

    # Resume 6: Ananya Iyer (DevOps - PDF)
    build_pdf_resume(
        RESUMES_DIR / "06_ananya_iyer_devops_cloud.pdf",
        name="Ananya Iyer",
        title="DevOps & Cloud Infrastructure Engineer",
        contact="ananya.iyer@example.com | +91 98450 12345 | linkedin.com/in/ananyaiyer",
        summary="DevOps Engineer with 4 years of experience automating AWS cloud infrastructure, orchestrating Kubernetes clusters, and building robust CI/CD pipelines for high-availability systems.",
        education="BTech in Information Technology, 2020",
        skills="AWS, Docker, Kubernetes, Terraform, Linux, CI/CD, Python, Bash, Git, Nginx",
        experience=[
            "<b>Senior DevOps Engineer</b> at FinTech Cloud Solutions (Oct 2020 - Present)",
            "• Architected multi-region AWS cloud infrastructure utilizing Terraform, achieving 99.99% service availability.",
            "• Managed production Kubernetes clusters (EKS) hosting 45+ microservices with automated Horizontal Pod Autoscaling.",
            "• Built automated CI/CD pipelines using GitHub Actions, cutting release deployment cycles from 2 hours to 14 minutes.",
            "• Hardened Linux operating system environments and implemented automated vulnerability scanning in Docker image builds.",
        ],
        projects=[
            "<b>Infrastructure-as-Code Framework:</b> Automated provisioning of VPC, RDS, and EKS clusters using reusable Terraform modules.",
            "<b>Zero-Downtime Migration:</b> Orchestrated migration of 12 legacy monoliths into Docker containers hosted on AWS.",
        ]
    )

    # Resume 7: Vikram Malhotra (Java Microservices - DOCX)
    build_docx_resume(
        RESUMES_DIR / "07_vikram_malhotra_java_backend.docx",
        name="Vikram Malhotra",
        title="Senior Java Backend Developer",
        contact="vikram.malhotra@example.com | +91 98711 22334 | linkedin.com/in/vikrammalhotra",
        summary="Backend Software Engineer with 3.5 years of expertise in enterprise Java, Spring Boot microservices, Hibernate ORM, and high-performance relational databases.",
        education="BTech in Computer Science, 2020",
        skills="Java, Spring Boot, Hibernate, SQL, MySQL, REST APIs, Microservices, Git, Docker, Unit Testing",
        experience=[
            "Java Backend Developer at Global Payments Inc (Sep 2020 - Present)",
            "Developed transactional banking microservices using Java and Spring Boot processing $2.5M in daily transactions.",
            "Optimized Hibernate object-relational mapping and MySQL index configurations, boosting throughput by 35%.",
            "Designed and implemented RESTful APIs adhering to OpenAPI 3.0 specifications with Spring Security authentication.",
            "Maintained 92% automated unit test coverage using JUnit and Mockito.",
        ],
        projects=[
            "Payment Processing Gateway: Engineered asynchronous transaction dispatcher using Spring Boot and MySQL.",
            "Account Reconciliation Service: Automated daily ledger settlement using Java scheduled batch processing.",
        ]
    )

    # Resume 8: Priya Nair (Frontend Specialist - PDF)
    build_pdf_resume(
        RESUMES_DIR / "08_priya_nair_frontend_specialist.pdf",
        name="Priya Nair",
        title="Senior Frontend React / Next.js Engineer",
        contact="priya.nair@example.com | +91 99887 66554 | linkedin.com/in/priyanair-dev",
        summary="Frontend Engineer with 4 years of experience crafting high-performance, accessible web interfaces using React, Next.js, TypeScript, and modern CSS frameworks.",
        education="BTech in Computer Science, 2020",
        skills="React, Next.js, TypeScript, JavaScript, Tailwind CSS, Redux, REST APIs, HTML, CSS, Responsive Design, Unit Testing",
        experience=[
            "<b>Senior Frontend Engineer</b> at SaaSify Media (Aug 2020 - Present)",
            "• Spearheaded migration of legacy single page application to Next.js with server-side rendering, improving First Contentful Paint by 48%.",
            "• Architected component design system in React, TypeScript, and Tailwind CSS adopted across 4 engineering teams.",
            "• Integrated Redux Toolkit for complex client-side state caching, reducing redundant network queries by 60%.",
            "• Wrote automated frontend unit and integration tests using Jest and React Testing Library.",
        ],
        projects=[
            "<b>Analytics Dashboard Portal:</b> Built interactive chart visualization suite using React, Next.js, and Recharts.",
            "<b>Design Component Library:</b> Created open-source accessible component library with 100% TypeScript type safety.",
        ]
    )

    # Resume 9: Rohan Mehta (Mobile Engineer - TXT)
    build_txt_resume(
        RESUMES_DIR / "09_rohan_mehta_mobile_developer.txt",
        """Rohan Mehta
Email: rohan.mehta@example.com | Phone: +91 98200 11223
LinkedIn: linkedin.com/in/rohanmehta-mobile | GitHub: github.com/rohanmehta

Professional Summary:
Mobile Application Developer with 2.5 years of experience building cross-platform iOS and Android apps using Flutter, Dart, and React Native.

Education:
BCA in Computer Applications, 2021

Technical Skills:
Flutter, Dart, React Native, JavaScript, Firebase, REST APIs, Git, Android Studio, Mobile App Development

Professional Experience:
Mobile Developer at AppCrafters Studio (Jul 2021 - Present)
• Built and published 3 cross-platform mobile apps on Google Play Store and Apple App Store using Flutter and Dart.
• Integrated Firebase cloud authentication, Firestore real-time database, and cloud push notifications.
• Consumed backend REST APIs and implemented offline-first SQLite caching for 80,000 active mobile users.
• Improved app launch time by 30% through widget tree optimization and state management in Riverpod.

Projects:
Fitness Tracker Mobile App:
Cross-platform Flutter application tracking workouts and health metrics with interactive charts and REST backend sync.
"""
    )

    # Resume 10: Deepa Joshi (QA Automation - TXT)
    build_txt_resume(
        RESUMES_DIR / "10_deepa_joshi_qa_automation.txt",
        """Deepa Joshi
Email: deepa.joshi@example.com | Phone: +91 97112 33445
LinkedIn: linkedin.com/in/deepajoshi-qa

Professional Summary:
QA Automation Engineer with 3 years of experience in automated test framework development, API validation, and continuous testing in CI/CD pipelines.

Education:
BTech in Information Technology, 2021

Technical Skills:
Selenium, PyTest, Unit Testing, Python, Postman, REST APIs, Git, CI/CD, SQL, Test Automation

Experience:
Quality Assurance Engineer at Enterprise Soft (Jul 2021 - Present)
• Developed automated end-to-end regression test suite in Python and Selenium, reducing manual testing effort by 70%.
• Authored comprehensive API integration tests in Postman and PyTest validating 120+ microservice endpoints.
• Integrated test execution into Jenkins and GitHub Actions CI/CD pipelines for automated pull request gating.
• Performed SQL backend database validations ensuring transaction data consistency.

Projects:
Automated Web Test Framework:
Built modular Page Object Model framework in Python and Selenium with automated HTML test reporting.
"""
    )

    # Resume 11: Rohit Kulkarni (Non-tech Sales - TXT)
    build_txt_resume(
        RESUMES_DIR / "11_rohit_kulkarni_sales_operations.txt",
        """Rohit Kulkarni
Email: rohit.kulkarni@example.com | Phone: +91 98330 44556
LinkedIn: linkedin.com/in/rohitkulkarni

Career Profile:
Sales Operations and Business Development professional with 2 years of experience in enterprise lead qualification, CRM pipeline management, and client relationship building.

Education:
Bachelor of Business Administration (BBA), 2022

Core Competencies:
Sales Operations, Lead Generation, CRM Management, Microsoft Excel, Client Communication, Contract Negotiation

Professional Experience:
Sales Associate at Apex Commercial Services (Jun 2022 - Present)
• Managed outbound sales prospecting campaigns, generating 140+ qualified enterprise leads per quarter.
• Maintained customer account records and opportunity stages in Salesforce CRM database.
• Prepared weekly sales pipeline reports and executive slide decks for regional leadership reviews.
• Negotiated pricing proposals and coordinated contract execution with enterprise clients.
"""
    )

    print("Successfully generated all sample resumes and JDs!")


if __name__ == "__main__":
    generate_all_samples()

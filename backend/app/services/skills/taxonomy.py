"""Curated skill taxonomy.

Each skill has:
  * aliases            - surface forms normalised to the canonical name (NLP normalisation)
  * case_sensitive     - aliases that are ambiguous in lower case (e.g. "Go", "Express")
  * description        - short natural-language gloss used to build semantic queries
  * implied_by         - related skills whose presence *suggests* this skill. Used only by
                         the Reflection Agent to mark a requirement INFERRED (needs human
                         verification) - never to grant a full match.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SkillDefinition:
    name: str
    category: str
    aliases: tuple[str, ...] = ()
    case_sensitive: tuple[str, ...] = ()
    description: str = ""
    implied_by: tuple[str, ...] = field(default_factory=tuple)


def _s(
    name: str,
    category: str,
    aliases: tuple[str, ...] = (),
    description: str = "",
    implied_by: tuple[str, ...] = (),
    case_sensitive: tuple[str, ...] = (),
) -> SkillDefinition:
    return SkillDefinition(name, category, aliases, case_sensitive, description, implied_by)


SKILL_TAXONOMY: tuple[SkillDefinition, ...] = (
    # ---- Programming languages -------------------------------------------------
    _s("Python", "language", ("python3",), "Python programming for scripting, backend or data work",
       ("Django", "Flask", "FastAPI", "Pandas", "NumPy", "scikit-learn", "PyTorch", "TensorFlow")),
    _s("JavaScript", "language", ("JS", "ECMAScript", "ES6", "Javascript"),
       "JavaScript programming for web front-end or back-end development",
       ("React", "Node.js", "Express", "Angular", "Vue.js", "Next.js", "TypeScript", "jQuery")),
    _s("TypeScript", "language", (), "TypeScript, statically typed JavaScript", ("Angular",),
       case_sensitive=("TS",)),
    _s("Java", "language", ("Core Java", "J2EE"), "Java programming and object-oriented development",
       ("Spring Boot", "Hibernate")),
    _s("C++", "language", ("CPP",), "C++ programming"),
    _s("C#", "language", ("C Sharp", "CSharp"), "C# programming on the .NET platform", (".NET",)),
    _s("C", "language", ("C programming", "C language", "Embedded C"), "C programming language"),
    _s("Go", "language", ("Golang",), "Go (Golang) programming", case_sensitive=("Go",)),
    _s("Kotlin", "language", (), "Kotlin programming"),
    _s("Swift", "language", (), "Swift programming for Apple platforms"),
    _s("PHP", "language", (), "PHP server-side web programming", ("Laravel",)),
    _s("Ruby", "language", (), "Ruby programming", ("Ruby on Rails",)),
    _s("Bash", "language", ("Shell Scripting", "shell script", "Bash scripting"), "Unix shell scripting"),
    # ---- Front-end -------------------------------------------------------------
    _s("React", "frontend", ("React.js", "ReactJS", "React JS"),
       "React library for building interactive front-end user interfaces and components",
       ("Next.js", "Redux", "React Native")),
    _s("Next.js", "frontend", ("NextJS", "Next JS"), "Next.js React framework with server-side rendering"),
    _s("Angular", "frontend", ("AngularJS", "Angular.js"), "Angular front-end framework"),
    _s("Vue.js", "frontend", ("Vue", "VueJS"), "Vue.js front-end framework"),
    _s("Redux", "frontend", ("Redux Toolkit",), "Redux state management"),
    _s("HTML", "frontend", ("HTML5",), "HTML markup for web pages", ("React", "Angular", "Vue.js")),
    _s("CSS", "frontend", ("CSS3",), "CSS styling of web pages", ("Tailwind CSS", "Bootstrap", "SASS")),
    _s("Tailwind CSS", "frontend", ("Tailwind", "TailwindCSS"), "Tailwind utility-first CSS framework"),
    _s("Bootstrap", "frontend", (), "Bootstrap CSS framework"),
    _s("SASS", "frontend", ("SCSS",), "SASS/SCSS CSS preprocessor"),
    _s("jQuery", "frontend", (), "jQuery JavaScript library"),
    # ---- Back-end --------------------------------------------------------------
    _s("Node.js", "backend", ("NodeJS", "Node JS", "Node"),
       "Node.js server-side JavaScript runtime for building backend services and APIs",
       ("Express", "NestJS")),
    _s("Express", "backend", ("Express.js", "ExpressJS"), "Express web framework for Node.js",
       case_sensitive=("Express",)),
    _s("NestJS", "backend", ("Nest.js",), "NestJS framework for Node.js"),
    _s("Django", "backend", (), "Django Python web framework", ("Django REST Framework",)),
    _s("Django REST Framework", "backend", ("DRF",), "Django REST Framework for building APIs"),
    _s("Flask", "backend", (), "Flask Python web micro-framework"),
    _s("FastAPI", "backend", (), "FastAPI Python framework for building APIs"),
    _s("Spring Boot", "backend", ("Spring MVC", "Spring Framework"), "Spring Boot Java backend framework"),
    _s("Hibernate", "backend", (), "Hibernate Java ORM"),
    _s(".NET", "backend", ("dotnet", "ASP.NET", ".NET Core"), ".NET backend development"),
    _s("Laravel", "backend", (), "Laravel PHP framework"),
    _s("Ruby on Rails", "backend", ("Rails",), "Ruby on Rails web framework"),
    _s("REST APIs", "backend", ("REST API", "RESTful", "RESTful API", "RESTful APIs"),
       "designing and building RESTful web APIs and HTTP endpoints for backend services",
       ("Express", "Django REST Framework", "FastAPI", "Spring Boot", "Flask"),
       case_sensitive=("REST",)),
    _s("GraphQL", "backend", (), "GraphQL API query language"),
    _s("Microservices", "backend", ("microservice", "micro-services"), "microservices architecture"),
    _s("WebSockets", "backend", ("WebSocket", "Socket.io"), "real-time communication with WebSockets"),
    # ---- Databases -------------------------------------------------------------
    _s("SQL", "database", ("Structured Query Language", "SQL queries"),
       "writing SQL queries and working with relational databases",
       ("MySQL", "PostgreSQL", "SQLite", "Oracle Database", "SQL Server")),
    _s("MySQL", "database", (), "MySQL relational database"),
    _s("PostgreSQL", "database", ("Postgres",), "PostgreSQL relational database"),
    _s("SQLite", "database", (), "SQLite embedded database"),
    _s("Oracle Database", "database", ("Oracle DB", "PL/SQL"), "Oracle relational database"),
    _s("SQL Server", "database", ("MS SQL", "MSSQL", "T-SQL"), "Microsoft SQL Server database"),
    _s("MongoDB", "database", ("Mongo", "Mongoose"), "MongoDB NoSQL document database"),
    _s("Redis", "database", (), "Redis in-memory data store and cache"),
    _s("Firebase", "database", ("Firestore",), "Firebase backend-as-a-service"),
    # ---- Cloud / DevOps ----------------------------------------------------------
    _s("AWS", "cloud", ("Amazon Web Services", "EC2", "S3", "AWS Lambda", "Lambda", "RDS", "CloudFront"),
       "Amazon Web Services cloud infrastructure such as EC2, S3 and Lambda"),
    _s("Azure", "cloud", ("Microsoft Azure",), "Microsoft Azure cloud platform"),
    _s("GCP", "cloud", ("Google Cloud", "Google Cloud Platform"), "Google Cloud Platform"),
    _s("Docker", "devops", ("Dockerfile", "containerization", "containerized"),
       "packaging and deploying applications in Docker containers", ("Kubernetes",)),
    _s("Kubernetes", "devops", ("K8s",), "Kubernetes container orchestration"),
    _s("CI/CD", "devops", ("CI CD", "continuous integration", "continuous deployment", "GitHub Actions",
                            "Jenkins", "GitLab CI"), "continuous integration and deployment pipelines"),
    _s("Linux", "devops", ("Ubuntu", "Unix"), "Linux operating system and command line"),
    _s("Terraform", "devops", (), "Terraform infrastructure as code"),
    _s("Nginx", "devops", (), "Nginx web server and reverse proxy"),
    # ---- Tools -------------------------------------------------------------------
    _s("Git", "tool", ("GitHub", "GitLab", "Bitbucket", "version control"), "Git version control"),
    _s("Postman", "tool", (), "Postman API testing tool"),
    _s("Jira", "tool", (), "Jira issue tracking"),
    _s("Figma", "tool", (), "Figma UI design tool"),
    _s("Excel", "tool", ("MS Excel", "Microsoft Excel", "pivot tables"), "spreadsheet analysis in Excel",
       case_sensitive=("Excel",)),
    _s("Tableau", "tool", (), "Tableau data visualisation"),
    _s("Power BI", "tool", ("PowerBI",), "Power BI business intelligence dashboards"),
    _s("Tkinter", "tool", (), "Tkinter Python GUI toolkit"),
    # ---- Testing -------------------------------------------------------------------
    _s("Unit Testing", "testing", ("unit tests", "Jest", "PyTest", "JUnit", "Mocha"), "writing automated unit tests"),
    _s("Selenium", "testing", (), "Selenium browser automation testing"),
    # ---- Data / AI -------------------------------------------------------------------
    _s("Machine Learning", "data", ("machine-learning",), "building machine learning models",
       ("scikit-learn", "TensorFlow", "PyTorch"), case_sensitive=("ML",)),
    _s("Deep Learning", "data", ("neural networks", "CNN", "RNN"), "deep learning with neural networks",
       ("TensorFlow", "PyTorch")),
    _s("NLP", "data", ("Natural Language Processing",), "natural language processing"),
    _s("Pandas", "data", (), "Pandas data analysis library"),
    _s("NumPy", "data", (), "NumPy numerical computing"),
    _s("scikit-learn", "data", ("sklearn", "scikit learn"), "scikit-learn machine learning library"),
    _s("TensorFlow", "data", ("Keras",), "TensorFlow deep learning framework"),
    _s("PyTorch", "data", (), "PyTorch deep learning framework"),
    _s("Data Analysis", "data", ("data analytics", "data analyst", "exploratory data analysis", "EDA"),
       "analysing datasets to derive insights"),
    _s("Data Visualization", "data", ("data visualisation", "Matplotlib", "Seaborn"),
       "visualising data in charts and dashboards", ("Tableau", "Power BI")),
    _s("Statistics", "data", ("statistical analysis",), "statistics and statistical analysis"),
    # ---- Mobile ----------------------------------------------------------------------
    _s("React Native", "mobile", (), "React Native cross-platform mobile development"),
    _s("Flutter", "mobile", ("Dart",), "Flutter cross-platform mobile development"),
    _s("Android", "mobile", ("Android Studio",), "Android app development"),
    # ---- Concepts --------------------------------------------------------------------
    _s("OOP", "concept", ("object-oriented programming", "object oriented programming", "OOPS"),
       "object-oriented programming principles"),
    _s("Data Structures", "concept", ("DSA", "data structures and algorithms", "algorithms"),
       "data structures and algorithms"),
    _s("Agile", "concept", ("Scrum", "Kanban", "sprint"), "agile / scrum software delivery"),
    _s("System Design", "concept", (), "software system design and architecture"),
    _s("Authentication", "concept", ("JWT", "OAuth", "OAuth2", "authorization"),
       "user authentication and authorization (JWT / OAuth)"),
    _s("Responsive Design", "concept", ("responsive web design", "mobile-first"),
       "responsive web design for multiple screen sizes"),
    # ---- Networking / Infrastructure -------------------------------------------------
    _s("Cisco", "cloud", ("Cisco IOS", "Cisco networking", "CCNA", "CCNP"), "Cisco networking equipment and configurations"),
    _s("Routing", "cloud", ("network routing", "IP routing", "static routing", "dynamic routing"), "network packet routing"),
    _s("Switching", "cloud", ("network switching", "L2 switching", "L3 switching", "VLANs", "VLAN"), "network switching and VLANs"),
    _s("TCP/IP", "cloud", ("TCP", "IP", "UDP", "networking protocols", "IPv4", "IPv6"), "TCP/IP networking suite"),
    _s("VPN", "cloud", ("IPsec", "OpenVPN", "WireGuard", "virtual private network"), "virtual private networks and secure tunneling"),
    _s("BGP", "cloud", ("Border Gateway Protocol",), "Border Gateway Protocol routing"),
    _s("OSPF", "cloud", ("Open Shortest Path First",), "Open Shortest Path First interior gateway protocol"),
    _s("Wireshark", "tool", ("packet analysis", "network packet capture", "pcap"), "Wireshark packet inspection and analysis"),
    _s("Firewalls", "cloud", ("Firewall", "pfSense", "Palo Alto", "Fortinet", "iptables"), "network firewalls and security rules"),
    # ---- Soft skills ------------------------------------------------------------------
    _s("Communication", "soft", ("communication skills", "verbal communication", "written communication"),
       "clear verbal and written communication"),
    _s("Teamwork", "soft", ("team player", "collaboration", "collaborative", "cross-functional"),
       "working collaboratively in a team"),
    _s("Problem Solving", "soft", ("problem-solving", "analytical skills", "analytical thinking"),
       "analytical problem solving"),
    _s("Leadership", "soft", ("led a team", "team lead", "mentored"), "leading and mentoring others"),
    _s("Time Management", "soft", ("prioritization", "meeting deadlines"), "managing time and priorities"),
)

SKILLS_BY_NAME: dict[str, SkillDefinition] = {s.name.lower(): s for s in SKILL_TAXONOMY}


def get_skill(name: str) -> SkillDefinition | None:
    return SKILLS_BY_NAME.get(name.lower())


def implying_skills(name: str) -> tuple[str, ...]:
    skill = get_skill(name)
    return skill.implied_by if skill else ()

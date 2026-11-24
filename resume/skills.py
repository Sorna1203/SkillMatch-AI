import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# ─── All Skills by Domain ───────────────────────────────────────────────────

SKILLS_BY_DOMAIN = {
    "IT & Software": [
        "python", "java", "javascript", "typescript", "c", "c++", "c#",
        "ruby", "go", "rust", "php", "swift", "kotlin", "scala",
        "html", "css", "react", "angular", "vue", "nodejs", "flask",
        "django", "spring", "sql", "mongodb", "postgresql", "mysql",
        "redis", "docker", "kubernetes", "aws", "azure", "git",
        "machine learning", "deep learning", "data science", "artificial intelligence",
        "tensorflow", "pytorch", "pandas", "numpy", "scikit-learn",
        "linux", "bash", "graphql", "rest api", "microservices",
        "devops", "ci/cd", "jenkins", "terraform", "ansible"
    ],
    "Finance & Accounting": [
        "accounting", "auditing", "taxation", "tally", "sap",
        "financial analysis", "financial modeling", "bookkeeping",
        "excel", "ms office", "gst", "payroll", "budgeting",
        "forecasting", "risk management", "banking", "investment",
        "portfolio management", "equity research", "forex"
    ],
    "Healthcare & Medical": [
        "nursing", "patient care", "clinical research", "pharmacology",
        "anatomy", "physiology", "medical coding", "ehr",
        "radiology", "surgery", "diagnosis", "pathology",
        "public health", "epidemiology", "healthcare management",
        "first aid", "cpr", "medical billing"
    ],
    "Marketing & Sales": [
        "digital marketing", "seo", "sem", "social media marketing",
        "content marketing", "email marketing", "google analytics",
        "facebook ads", "brand management", "market research",
        "crm", "salesforce", "lead generation", "copywriting",
        "advertising", "public relations", "sales strategy"
    ],
    "Design & Creative": [
        "photoshop", "illustrator", "figma", "sketch", "adobe xd",
        "ui design", "ux design", "graphic design", "video editing",
        "after effects", "premiere pro", "3d modeling", "blender",
        "typography", "branding", "motion graphics", "photography"
    ],
    "Engineering": [
        "autocad", "solidworks", "matlab", "simulink",
        "circuit design", "embedded systems", "plc", "scada",
        "mechanical design", "structural analysis", "civil engineering",
        "electrical engineering", "project management", "six sigma",
        "lean manufacturing", "quality control", "supply chain"
    ],
    "Education & Research": [
        "curriculum development", "teaching", "e-learning",
        "research methodology", "data analysis", "spss",
        "academic writing", "training", "mentoring", "coaching"
    ],
    "Soft Skills": [
        "leadership", "communication", "teamwork", "problem solving",
        "time management", "critical thinking", "adaptability",
        "project management", "negotiation", "presentation"
    ]
}

# Flat list of all skills
ALL_SKILLS = []
for domain_skills in SKILLS_BY_DOMAIN.values():
    ALL_SKILLS.extend(domain_skills)

SINGLE_SKILLS = [s for s in ALL_SKILLS if len(s.split()) == 1]
MULTI_SKILLS  = [s for s in ALL_SKILLS if len(s.split()) > 1]


def extract_skills(text):
    """Extract skills from resume text — handles both single and multi-word skills."""
    if not text:
        return []

    text_lower = text.lower()
    found = set()

    # Multi-word skills — direct substring match on raw text
    for skill in MULTI_SKILLS:
        if skill in text_lower:
            found.add(skill)

    # Single-word skills — tokenize and filter
    try:
        tokens = word_tokenize(text_lower)
    except Exception:
        tokens = text_lower.split()

    tokens = [t for t in tokens if t.isalpha()]
    stop_words = set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in stop_words]
    token_set = set(tokens)

    for skill in SINGLE_SKILLS:
        if skill in token_set:
            found.add(skill)

    return sorted(list(found))


def get_skill_domain(skill):
    """Return the domain of a skill."""
    skill_lower = skill.lower()
    for domain, skills in SKILLS_BY_DOMAIN.items():
        if skill_lower in skills:
            return domain
    return "General"


def get_domain_from_skills(skills):
    """Determine the primary domain from a list of skills."""
    domain_count = {}
    for skill in skills:
        domain = get_skill_domain(skill)
        domain_count[domain] = domain_count.get(domain, 0) + 1

    if not domain_count:
        return "General"

    return max(domain_count, key=domain_count.get)
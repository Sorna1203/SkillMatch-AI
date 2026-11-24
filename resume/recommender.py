from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from resume.skills import get_domain_from_skills
import math
import re
from difflib import SequenceMatcher

# ═══════════════════════════════════════════════════════════════════════════
# ENHANCED MATCHING ALGORITHM - Multi-Factor Scoring System
# ═══════════════════════════════════════════════════════════════════════════

# Skill relationship mapping for partial matching
SKILL_RELATIONSHIPS = {
    # Programming Languages
    "python": ["django", "flask", "pandas", "numpy", "machine learning", "data science"],
    "javascript": ["react", "vue.js", "angular", "node.js", "typescript", "frontend", "full stack"],
    "java": ["spring boot", "hibernate", "microservices", "android", "backend"],
    "sql": ["mysql", "postgresql", "oracle", "database", "data analysis", "dbms"],
    # Cloud & DevOps
    "aws": ["cloud", "devops", "terraform", "kubernetes", "docker", "s3", "ec2"],
    "docker": ["kubernetes", "containerization", "devops", "microservices", "ci/cd"],
    # Data & AI
    "machine learning": ["deep learning", "ai", "tensorflow", "pytorch", "data science", "python"],
    "data analysis": ["python", "sql", "excel", "statistics", "power bi", "tableau"],
    # Web Development
    "react": ["javascript", "frontend", "web development", "spa", "node.js"],
    "node.js": ["javascript", "backend", "express", "full stack", "api"],
}

# Critical skills by domain that carry higher weight
CRITICAL_SKILLS_BY_DOMAIN = {
    "IT & Software": ["python", "javascript", "java", "sql", "git", "docker"],
    "Finance & Accounting": ["accounting", "financial analysis", "excel", "taxation", "auditing"],
    "Healthcare & Medical": ["patient care", "medical coding", "clinical research", "healthcare management"],
    "Marketing & Sales": ["digital marketing", "seo", "sales", "market research", "analytics"],
    "Design & Creative": ["ui/ux", "graphic design", "adobe creative suite", "figma"],
    "Engineering": ["cad", "mechanical design", "electrical systems", "project management"],
    "Education & Research": ["curriculum development", "research methodology", "teaching"],
    "HR & Administration": ["recruitment", "hr policies", "payroll", "employee relations"],
}


def calculate_similarity(str1, str2):
    """Calculate string similarity ratio for fuzzy matching."""
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()


def get_skill_weight(skill, job_domain):
    """Assign weight to skill based on criticality."""
    skill_lower = skill.lower()
    
    # Check if skill is critical for the domain
    domain_critical = CRITICAL_SKILLS_BY_DOMAIN.get(job_domain, [])
    if skill_lower in [s.lower() for s in domain_critical]:
        return 1.5  # 50% bonus for critical skills
    
    return 1.0  # Standard weight


def find_related_skills(user_skill, job_skills):
    """Find related skills using skill relationship mapping."""
    user_skill_lower = user_skill.lower()
    related_matches = []
    
    # Check direct relationships
    if user_skill_lower in SKILL_RELATIONSHIPS:
        related = SKILL_RELATIONSHIPS[user_skill_lower]
        for job_skill in job_skills:
            if job_skill.lower() in [r.lower() for r in related]:
                related_matches.append((job_skill, 0.7))  # 70% weight for related skills
    
    # Check fuzzy matches
    for job_skill in job_skills:
        similarity = calculate_similarity(user_skill, job_skill)
        if similarity >= 0.75 and similarity < 1.0:  # Partial match
            related_matches.append((job_skill, similarity * 0.8))
        elif any(word in job_skill.lower() for word in user_skill_lower.split()):
            related_matches.append((job_skill, 0.6))  # Word overlap
    
    return related_matches


def enhanced_calculate_match_score(user_skills, job_skills, job_domain):
    """
    Enhanced multi-factor match scoring algorithm.
    Returns a score between 0-100 with realistic variations.
    
    Factors:
    - Exact skill matches (full weight)
    - Related/partial skill matches (partial weight)
    - Critical skill bonus
    - Domain alignment
    - Missing critical skills penalty
    - Extra relevant skills bonus
    """
    if not job_skills:
        return 0
    
    user_skills_lower = set([s.lower() for s in user_skills])
    job_skills_lower = set([s.lower() for s in job_skills])
    
    total_possible_score = 0
    achieved_score = 0
    
    # Track which job skills are matched
    matched_job_skills = set()
    
    # 1. Calculate score for exact matches and partial matches
    for job_skill in job_skills:
        job_skill_lower = job_skill.lower()
        weight = get_skill_weight(job_skill, job_domain)
        total_possible_score += weight
        
        # Exact match
        if job_skill_lower in user_skills_lower:
            achieved_score += weight
            matched_job_skills.add(job_skill_lower)
        else:
            # Check for related/partial matches
            for user_skill in user_skills:
                related = find_related_skills(user_skill, [job_skill])
                if related:
                    partial_weight = related[0][1] * weight
                    achieved_score += partial_weight
                    matched_job_skills.add(job_skill_lower)
                    break
    
    # 2. Calculate base percentage
    if total_possible_score > 0:
        base_score = (achieved_score / total_possible_score) * 100
    else:
        base_score = 0
    
    # 3. Apply domain alignment bonus/penalty
    user_domain = get_domain_from_skills(user_skills)
    domain_bonus = 0
    if user_domain == job_domain:
        domain_bonus = 5  # 5% bonus for domain match
    elif user_domain and job_domain:
        # Check for related domains
        related_domains = {
            "IT & Software": ["Engineering", "Design & Creative"],
            "Finance & Accounting": ["IT & Software", "HR & Administration"],
            "Healthcare & Medical": ["Education & Research"],
        }
        if job_domain in related_domains.get(user_domain, []):
            domain_bonus = 2  # 2% bonus for related domain
    
    # 4. Bonus for extra relevant skills (skills user has that aren't required but are relevant)
    extra_skills_bonus = 0
    extra_skills_count = 0
    for user_skill in user_skills:
        user_skill_lower = user_skill.lower()
        if user_skill_lower not in job_skills_lower:
            # Check if this extra skill is related to the job domain
            if job_domain in CRITICAL_SKILLS_BY_DOMAIN:
                domain_skills = [s.lower() for s in CRITICAL_SKILLS_BY_DOMAIN[job_domain]]
                if user_skill_lower in domain_skills:
                    extra_skills_count += 1
    
    # Cap extra skills bonus at 10%
    extra_skills_bonus = min(extra_skills_count * 2, 10)
    
    # 5. Penalty for missing critical skills
    critical_penalty = 0
    if job_domain in CRITICAL_SKILLS_BY_DOMAIN:
        domain_critical = [s.lower() for s in CRITICAL_SKILLS_BY_DOMAIN[job_domain]]
        job_critical = [s for s in job_skills_lower if s in domain_critical]
        matched_critical = [s for s in matched_job_skills if s in domain_critical]
        
        if job_critical:
            critical_match_ratio = len(matched_critical) / len(job_critical)
            if critical_match_ratio < 0.5:  # Less than 50% critical skills matched
                critical_penalty = (0.5 - critical_match_ratio) * 15  # Up to 7.5% penalty
    
    # 6. Calculate final score
    final_score = base_score + domain_bonus + extra_skills_bonus - critical_penalty
    
    # 7. Add realistic variation based on skill depth
    # More matched skills = slightly higher score within range
    skill_depth_factor = min(len(matched_job_skills) * 0.5, 5)  # Up to 5% bonus
    final_score += skill_depth_factor
    
    # 8. Ensure score is within realistic bounds (0-95%)
    # No one gets 100% - there's always room to grow
    final_score = max(0, min(final_score, 95))
    
    return round(final_score, 1)


# ═══════════════════════════════════════════════════════════════════════════
# LEGACY FUNCTIONS (kept for compatibility)
# ═══════════════════════════════════════════════════════════════════════════

# ─── Comprehensive Job Database — Kaggle-Style Multi-Domain Dataset ──────────

ALL_JOBS = [
    # ══════════════════════════════════════════════════════════════════════════
    # IT & SOFTWARE DEVELOPMENT
    # ══════════════════════════════════════════════════════════════════════════
    {"title": "Python Developer", "domain": "IT & Software", "skills": ["python", "django", "flask", "sql", "postgresql", "git", "rest api", "docker"]},
    {"title": "Java Developer", "domain": "IT & Software", "skills": ["java", "spring boot", "hibernate", "sql", "maven", "git", "microservices", "rest api"]},
    {"title": "JavaScript Developer", "domain": "IT & Software", "skills": ["javascript", "typescript", "react", "node.js", "html", "css", "git", "npm"]},
    {"title": "Frontend Engineer", "domain": "IT & Software", "skills": ["react", "vue.js", "angular", "typescript", "html", "css", "sass", "webpack", "figma"]},
    {"title": "Backend Developer", "domain": "IT & Software", "skills": ["python", "django", "flask", "node.js", "express", "sql", "mongodb", "redis", "docker"]},
    {"title": "Full Stack Developer", "domain": "IT & Software", "skills": ["python", "javascript", "react", "node.js", "sql", "mongodb", "html", "css", "git", "docker"]},
    {"title": "Senior Software Engineer", "domain": "IT & Software", "skills": ["python", "java", "javascript", "sql", "system design", "microservices", "aws", "docker", "kubernetes"]},
    {"title": "Software Developer", "domain": "IT & Software", "skills": ["python", "java", "c++", "html", "css", "javascript", "sql", "git"]},
    {"title": "DevOps Engineer", "domain": "IT & Software", "skills": ["docker", "kubernetes", "aws", "azure", "terraform", "jenkins", "ci/cd", "linux", "bash", "ansible"]},
    {"title": "Site Reliability Engineer", "domain": "IT & Software", "skills": ["linux", "python", "aws", "docker", "kubernetes", "monitoring", "terraform", "grafana", "prometheus"]},
    {"title": "Cloud Engineer", "domain": "IT & Software", "skills": ["aws", "azure", "gcp", "terraform", "docker", "kubernetes", "linux", "python", "networking"]},
    {"title": "Cloud Architect", "domain": "IT & Software", "skills": ["aws", "azure", "gcp", "terraform", "kubernetes", "microservices", "security", "networking", "linux"]},
    {"title": "Data Engineer", "domain": "IT & Software", "skills": ["python", "sql", "spark", "hadoop", "airflow", "aws", "databricks", "etl", "data pipeline"]},
    {"title": "Data Scientist", "domain": "IT & Software", "skills": ["python", "machine learning", "deep learning", "tensorflow", "pytorch", "pandas", "numpy", "sql", "statistics"]},
    {"title": "Machine Learning Engineer", "domain": "IT & Software", "skills": ["python", "machine learning", "deep learning", "tensorflow", "pytorch", "scikit-learn", "mlops", "docker", "aws"]},
    {"title": "AI Engineer", "domain": "IT & Software", "skills": ["python", "tensorflow", "pytorch", "machine learning", "deep learning", "nlp", "computer vision", "artificial intelligence", "openai"]},
    {"title": "Computer Vision Engineer", "domain": "IT & Software", "skills": ["python", "tensorflow", "pytorch", "opencv", "computer vision", "deep learning", "image processing", "cuda"]},
    {"title": "NLP Engineer", "domain": "IT & Software", "skills": ["python", "nlp", "tensorflow", "pytorch", "spacy", "nltk", "transformers", "bert", "machine learning"]},
    {"title": "Database Administrator", "domain": "IT & Software", "skills": ["sql", "postgresql", "mysql", "mongodb", "redis", "aws rds", "backup", "security", "performance tuning"]},
    {"title": "Database Developer", "domain": "IT & Software", "skills": ["sql", "pl/sql", "postgresql", "mysql", "stored procedures", "etl", "data modeling", "oracle"]},
    {"title": "Cybersecurity Analyst", "domain": "IT & Software", "skills": ["security", "penetration testing", "network security", "linux", "python", "firewalls", "siem", "vulnerability assessment"]},
    {"title": "Security Engineer", "domain": "IT & Software", "skills": ["security", "aws security", "penetration testing", "siem", "python", "incident response", "risk management", "compliance"]},
    {"title": "Network Engineer", "domain": "IT & Software", "skills": ["networking", "cisco", "tcp/ip", "firewalls", "vpn", "routing", "switching", "linux", "aws"]},
    {"title": "Mobile App Developer (iOS)", "domain": "IT & Software", "skills": ["swift", "ios", "xcode", "objective-c", "core data", "firebase", "cocoapods", "swiftui"]},
    {"title": "Mobile App Developer (Android)", "domain": "IT & Software", "skills": ["kotlin", "java", "android", "android studio", "firebase", "gradle", "retrofit", "jetpack compose"]},
    {"title": "React Native Developer", "domain": "IT & Software", "skills": ["react native", "javascript", "typescript", "ios", "android", "firebase", "redux", "rest api"]},
    {"title": "QA Automation Engineer", "domain": "IT & Software", "skills": ["selenium", "python", "java", "testing", "automation", "jenkins", "ci/cd", "testng", "junit"]},
    {"title": "Manual Tester", "domain": "IT & Software", "skills": ["testing", "test cases", "bug reporting", "jira", "sql", "manual testing", "regression testing"]},
    {"title": "Performance Test Engineer", "domain": "IT & Software", "skills": ["jmeter", "loadrunner", "performance testing", "gatling", "monitoring", "aws", "kubernetes", "python"]},
    {"title": "Technical Lead", "domain": "IT & Software", "skills": ["python", "java", "system design", "agile", "microservices", "aws", "mentoring", "project management"]},
    {"title": "Solutions Architect", "domain": "IT & Software", "skills": ["aws", "azure", "system design", "microservices", "security", "networking", "docker", "kubernetes"]},
    {"title": "Scrum Master", "domain": "IT & Software", "skills": ["agile", "scrum", "jira", "kanban", "leadership", "communication", "project management"]},
    {"title": "Product Manager (Tech)", "domain": "IT & Software", "skills": ["product management", "agile", "sql", "data analysis", "user research", "roadmapping", "stakeholder management"]},
    {"title": "Technical Writer", "domain": "IT & Software", "skills": ["technical writing", "documentation", "markdown", "git", "api documentation", "xml", "content management"]},
    {"title": "Salesforce Developer", "domain": "IT & Software", "skills": ["salesforce", "apex", "visualforce", "soql", "lightning", "crm", "integration", "java"]},
    {"title": "WordPress Developer", "domain": "IT & Software", "skills": ["wordpress", "php", "html", "css", "javascript", "woocommerce", "plugin development", "seo"]},
    
    # ══════════════════════════════════════════════════════════════════════════
    # FINANCE & ACCOUNTING
    # ══════════════════════════════════════════════════════════════════════════
    {"title": "Financial Analyst", "domain": "Finance & Accounting", "skills": ["financial analysis", "financial modeling", "excel", "python", "sql", "forecasting", "budgeting", "variance analysis"]},
    {"title": "Senior Financial Analyst", "domain": "Finance & Accounting", "skills": ["financial analysis", "financial modeling", "excel", "python", "sql", "dashboarding", "power bi", "strategic planning"]},
    {"title": "Accountant", "domain": "Finance & Accounting", "skills": ["accounting", "tally", "gst", "tds", "excel", "ms office", "bookkeeping", "bank reconciliation"]},
    {"title": "Chartered Accountant", "domain": "Finance & Accounting", "skills": ["accounting", "auditing", "taxation", "gst", "income tax", "direct tax", "indirect tax", "compliance"]},
    {"title": "Cost Accountant", "domain": "Finance & Accounting", "skills": ["cost accounting", "budgeting", "variance analysis", "costing", "excel", "sap", "tally", "mis reporting"]},
    {"title": "Tax Consultant", "domain": "Finance & Accounting", "skills": ["taxation", "income tax", "gst", "corporate tax", "tax planning", "compliance", "audit", "excel"]},
    {"title": "Tax Analyst", "domain": "Finance & Accounting", "skills": ["taxation", "gst", "income tax", "tds", "excel", "tax filing", "compliance", "ms office"]},
    {"title": "Investment Analyst", "domain": "Finance & Accounting", "skills": ["investment analysis", "equity research", "portfolio management", "financial modeling", "excel", "bloomberg", "valuation"]},
    {"title": "Risk Analyst", "domain": "Finance & Accounting", "skills": ["risk management", "credit risk", "market risk", "financial analysis", "python", "sql", "risk modeling", "excel"]},
    {"title": "Credit Analyst", "domain": "Finance & Accounting", "skills": ["credit analysis", "financial statements", "risk assessment", "excel", "credit scoring", "banking", "reporting"]},
    {"title": "Auditor", "domain": "Finance & Accounting", "skills": ["auditing", "internal audit", "external audit", "compliance", "sox", "risk assessment", "ifrs", "gaap"]},
    {"title": "Internal Auditor", "domain": "Finance & Accounting", "skills": ["internal audit", "risk management", "compliance", "sox", "audit testing", "process improvement", "excel"]},
    {"title": "SAP FICO Consultant", "domain": "Finance & Accounting", "skills": ["sap", "sap fico", "finance", "controlling", "accounting", "implementation", "configuration"]},
    {"title": "Business Analyst (Finance)", "domain": "Finance & Accounting", "skills": ["business analysis", "sql", "excel", "financial analysis", "requirements gathering", "tableau", "power bi"]},
    {"title": "FP&A Manager", "domain": "Finance & Accounting", "skills": ["financial planning", "forecasting", "budgeting", "excel", "financial modeling", "strategic planning", "variance analysis"]},
    {"title": "Accounts Payable Specialist", "domain": "Finance & Accounting", "skills": ["accounts payable", "invoice processing", "tally", "excel", "vendor management", "reconciliation", "ms office"]},
    {"title": "Accounts Receivable Specialist", "domain": "Finance & Accounting", "skills": ["accounts receivable", "collection", "reconciliation", "excel", "credit management", "aging analysis", "ms office"]},
    {"title": "Payroll Specialist", "domain": "Finance & Accounting", "skills": ["payroll", "tally", "gst", "employee benefits", "compliance", "excel", "statutory compliance", "ms office"]},
    {"title": "Finance Manager", "domain": "Finance & Accounting", "skills": ["finance management", "financial analysis", "team management", "budgeting", "forecasting", "compliance", "stakeholder management"]},
    {"title": "Treasury Analyst", "domain": "Finance & Accounting", "skills": ["treasury", "cash management", "banking", "forex", "liquidity management", "excel", "financial analysis"]},
    {"title": "Equity Research Analyst", "domain": "Finance & Accounting", "skills": ["equity research", "financial modeling", "valuation", "investment analysis", "excel", "bloomberg", "industry research"]},
    {"title": "Compliance Officer", "domain": "Finance & Accounting", "skills": ["compliance", "aml", "kyc", "regulatory compliance", "risk management", "banking", "audit", "reporting"]},
    
    # ══════════════════════════════════════════════════════════════════════════
    # HEALTHCARE & MEDICAL
    # ══════════════════════════════════════════════════════════════════════════
    {"title": "Clinical Nurse", "domain": "Healthcare & Medical", "skills": ["nursing", "patient care", "first aid", "cpr", "clinical assessment", "medication administration", "documentation"]},
    {"title": "Registered Nurse", "domain": "Healthcare & Medical", "skills": ["nursing", "patient care", "iv therapy", "medication administration", "electronic health records", "clinical research"]},
    {"title": "Nurse Practitioner", "domain": "Healthcare & Medical", "skills": ["nursing", "diagnosis", "treatment", "patient care", "prescribing", "health assessment", "clinical decision making"]},
    {"title": "Medical Coder", "domain": "Healthcare & Medical", "skills": ["medical coding", "icd-10", "cpt", "hcpcs", "medical billing", "ehr", "anatomy", "compliance"]},
    {"title": "Medical Billing Specialist", "domain": "Healthcare & Medical", "skills": ["medical billing", "insurance", "claims processing", "ehr", "medical coding", "revenue cycle", "follow-up"]},
    {"title": "Healthcare Administrator", "domain": "Healthcare & Medical", "skills": ["healthcare management", "hospital administration", "budgeting", "compliance", "healthcare law", "strategic planning"]},
    {"title": "Healthcare Manager", "domain": "Healthcare & Medical", "skills": ["healthcare management", "project management", "budgeting", "operations", "leadership", "compliance", "quality improvement"]},
    {"title": "Clinical Research Associate", "domain": "Healthcare & Medical", "skills": ["clinical research", "gcp", "ich guidelines", "site monitoring", "protocol compliance", "data management", "audit"]},
    {"title": "Clinical Data Manager", "domain": "Healthcare & Medical", "skills": ["clinical data management", "cdisc", "sql", "clinical trials", "data validation", "ehr", "statistical analysis"]},
    {"title": "Pharmacist", "domain": "Healthcare & Medical", "skills": ["pharmacy", "pharmacology", "drug dispensing", "patient counseling", "inventory management", "compliance", "clinical pharmacy"]},
    {"title": "Medical Technologist", "domain": "Healthcare & Medical", "skills": ["medical technology", "laboratory testing", "hematology", "microbiology", "clinical chemistry", "quality control"]},
    {"title": "Radiology Technician", "domain": "Healthcare & Medical", "skills": ["radiology", "x-ray", "mri", "ct scan", "patient positioning", "radiation safety", "image quality"]},
    {"title": "Physical Therapist", "domain": "Healthcare & Medical", "skills": ["physical therapy", "rehabilitation", "exercise therapy", "patient assessment", "mobility training", "pain management"]},
    {"title": "Public Health Analyst", "domain": "Healthcare & Medical", "skills": ["public health", "epidemiology", "data analysis", "health policy", "biostatistics", "research methodology", "reporting"]},
    {"title": "Health Information Manager", "domain": "Healthcare & Medical", "skills": ["health information", "ehr", "hipaa", "medical records", "data quality", "compliance", "information systems"]},
    {"title": "Biomedical Engineer", "domain": "Healthcare & Medical", "skills": ["biomedical engineering", "medical devices", "signal processing", "matlab", "fda regulations", "quality assurance", "testing"]},
    {"title": "Epidemiologist", "domain": "Healthcare & Medical", "skills": ["epidemiology", "data analysis", "research methodology", "biostatistics", "public health", "disease surveillance", "reporting"]},
    {"title": "Mental Health Counselor", "domain": "Healthcare & Medical", "skills": ["counseling", "mental health", "therapy", "psychology", "crisis intervention", "assessment", "documentation"]},
    
    # ══════════════════════════════════════════════════════════════════════════
    # MARKETING & SALES
    # ══════════════════════════════════════════════════════════════════════════
    {"title": "Digital Marketing Manager", "domain": "Marketing & Sales", "skills": ["digital marketing", "seo", "sem", "google analytics", "social media marketing", "content marketing", "email marketing", "roi analysis"]},
    {"title": "SEO Specialist", "domain": "Marketing & Sales", "skills": ["seo", "google analytics", "keyword research", "on-page seo", "off-page seo", "technical seo", "content optimization"]},
    {"title": "SEM/PPC Specialist", "domain": "Marketing & Sales", "skills": ["google ads", "ppc", "sem", "campaign management", "bidding strategies", "analytics", "roi optimization"]},
    {"title": "Social Media Manager", "domain": "Marketing & Sales", "skills": ["social media marketing", "content creation", "community management", "analytics", "instagram", "linkedin", "twitter", "facebook"]},
    {"title": "Content Marketing Manager", "domain": "Marketing & Sales", "skills": ["content marketing", "copywriting", "seo", "content strategy", "blogging", "content calendar", "analytics", "brand voice"]},
    {"title": "Content Writer", "domain": "Marketing & Sales", "skills": ["content writing", "copywriting", "seo writing", "blogging", "creative writing", "research", "editing"]},
    {"title": "Email Marketing Specialist", "domain": "Marketing & Sales", "skills": ["email marketing", "mailchimp", "automation", "campaign management", "segmentation", "a/b testing", "analytics"]},
    {"title": "Brand Manager", "domain": "Marketing & Sales", "skills": ["brand management", "marketing strategy", "market research", "advertising", "brand positioning", "consumer insights", "campaign management"]},
    {"title": "Product Marketing Manager", "domain": "Marketing & Sales", "skills": ["product marketing", "go-to-market", "competitive analysis", "positioning", "messaging", "sales enablement", "market research"]},
    {"title": "Marketing Analyst", "domain": "Marketing & Sales", "skills": ["marketing analytics", "python", "sql", "tableau", "excel", "data analysis", "attribution modeling", "reporting"]},
    {"title": "CRM Manager", "domain": "Marketing & Sales", "skills": ["crm", "salesforce", "hubspot", "customer segmentation", "marketing automation", "data analysis", "reporting"]},
    {"title": "Salesforce Administrator", "domain": "Marketing & Sales", "skills": ["salesforce", "crm", "administration", "user management", "reports", "dashboards", "automation"]},
    {"title": "Sales Executive", "domain": "Marketing & Sales", "skills": ["sales", "lead generation", "negotiation", "crm", "presentation", "client relationship", "business development"]},
    {"title": "Business Development Manager", "domain": "Marketing & Sales", "skills": ["business development", "sales", "lead generation", "client acquisition", "negotiation", "partnerships", "strategy"]},
    {"title": "Account Manager", "domain": "Marketing & Sales", "skills": ["account management", "client relations", "sales", "communication", "project coordination", "upselling", "renewals"]},
    {"title": "Sales Director", "domain": "Marketing & Sales", "skills": ["sales leadership", "team management", "strategy", "forecasting", "negotiation", "business development", "revenue growth"]},
    {"title": "Marketing Director", "domain": "Marketing & Sales", "skills": ["marketing leadership", "strategy", "brand management", "digital marketing", "team management", "budgeting", "analytics"]},
    {"title": "Growth Hacker", "domain": "Marketing & Sales", "skills": ["growth hacking", "digital marketing", "analytics", "conversion optimization", "ab testing", "seo", "social media"]},
    {"title": "Influencer Marketing Manager", "domain": "Marketing & Sales", "skills": ["influencer marketing", "social media", "content collaboration", "campaign management", "analytics", "brand partnerships"]},
    {"title": "Public Relations Manager", "domain": "Marketing & Sales", "skills": ["public relations", "media relations", "press releases", "crisis management", "brand communications", "stakeholder engagement"]},
    {"title": "Copywriter", "domain": "Marketing & Sales", "skills": ["copywriting", "creative writing", "advertising", "brand voice", "marketing collateral", "editing", "seo"]},
    {"title": "Marketing Coordinator", "domain": "Marketing & Sales", "skills": ["marketing coordination", "social media", "event planning", "content creation", "admin support", "ms office", "analytics"]},
    
    # ══════════════════════════════════════════════════════════════════════════
    # DESIGN & CREATIVE
    # ══════════════════════════════════════════════════════════════════════════
    {"title": "UI/UX Designer", "domain": "Design & Creative", "skills": ["ui design", "ux design", "figma", "sketch", "adobe xd", "prototyping", "user research", "wireframing", "usability testing"]},
    {"title": "Senior UI/UX Designer", "domain": "Design & Creative", "skills": ["ui design", "ux design", "figma", "sketch", "design systems", "user research", "interaction design", "visual design", "leadership"]},
    {"title": "Graphic Designer", "domain": "Design & Creative", "skills": ["graphic design", "photoshop", "illustrator", "indesign", "branding", "typography", "print design", "creative concepts"]},
    {"title": "Senior Graphic Designer", "domain": "Design & Creative", "skills": ["graphic design", "photoshop", "illustrator", "branding", "art direction", "creative leadership", "print", "digital"]},
    {"title": "Web Designer", "domain": "Design & Creative", "skills": ["web design", "html", "css", "figma", "responsive design", "ui design", "prototyping", "user experience"]},
    {"title": "Motion Graphics Designer", "domain": "Design & Creative", "skills": ["motion graphics", "after effects", "premiere pro", "animation", "video editing", "3d animation", "cinema 4d"]},
    {"title": "Video Editor", "domain": "Design & Creative", "skills": ["video editing", "premiere pro", "after effects", "final cut pro", "color grading", "sound design", "storytelling"]},
    {"title": "3D Artist", "domain": "Design & Creative", "skills": ["3d modeling", "blender", "maya", "cinema 4d", "texturing", "lighting", "rendering", "animation"]},
    {"title": "3D Animator", "domain": "Design & Creative", "skills": ["3d animation", "maya", "blender", "character animation", "rigging", "modeling", "rendering", "lighting"]},
    {"title": "Product Designer", "domain": "Design & Creative", "skills": ["product design", "ui design", "ux design", "figma", "prototyping", "design thinking", "user research", "agile"]},
    {"title": "Brand Designer", "domain": "Design & Creative", "skills": ["brand design", "logo design", "branding", "typography", "color theory", "illustrator", "brand identity", "style guides"]},
    {"title": "Illustrator", "domain": "Design & Creative", "skills": ["illustration", "digital illustration", "adobe illustrator", "character design", "concept art", "visual storytelling", "drawing"]},
    {"title": "UI Developer", "domain": "Design & Creative", "skills": ["html", "css", "javascript", "react", "figma", "responsive design", "bootstrap", "sass", "web design"]},
    {"title": "UX Researcher", "domain": "Design & Creative", "skills": ["ux research", "user research", "usability testing", "surveys", "interviews", "data analysis", "ethnographic research", "synthesis"]},
    {"title": "Creative Director", "domain": "Design & Creative", "skills": ["creative direction", "art direction", "brand strategy", "team leadership", "creative concepts", "design systems", "client collaboration"]},
    {"title": "Photographer", "domain": "Design & Creative", "skills": ["photography", "photo editing", "lightroom", "photoshop", "composition", "studio lighting", "portrait photography", "commercial photography"]},
    {"title": "UX Writer", "domain": "Design & Creative", "skills": ["ux writing", "content design", "microcopy", "user interface", "research", "editing", "information architecture"]},
    {"title": "Design System Lead", "domain": "Design & Creative", "skills": ["design systems", "figma", "component libraries", "documentation", "ui design", "design tokens", "collaboration", "governance"]},
    
    # ══════════════════════════════════════════════════════════════════════════
    # ENGINEERING
    # ══════════════════════════════════════════════════════════════════════════
    {"title": "Mechanical Engineer", "domain": "Engineering", "skills": ["autocad", "solidworks", "catia", "mechanical design", "manufacturing", "gd&t", "finite element analysis", "thermal analysis"]},
    {"title": "Senior Mechanical Engineer", "domain": "Engineering", "skills": ["mechanical engineering", "solidworks", "autocad", "project management", "team leadership", "design optimization", "failure analysis"]},
    {"title": "Electrical Engineer", "domain": "Engineering", "skills": ["electrical engineering", "circuit design", "pcb design", "matlab", "simulink", "embedded systems", "power systems", "cad"]},
    {"title": "Electronics Engineer", "domain": "Engineering", "skills": ["electronics engineering", "circuit design", "pcb design", "embedded c", "microcontrollers", "verilog", "fpga", "testing"]},
    {"title": "Civil Engineer", "domain": "Engineering", "skills": ["civil engineering", "autocad", "staad pro", "etabs", "structural analysis", "construction management", "project planning", "site supervision"]},
    {"title": "Structural Engineer", "domain": "Engineering", "skills": ["structural engineering", "autocad", "staad pro", "etabs", "sap2000", "structural analysis", "seismic design", "steel design"]},
    {"title": "Chemical Engineer", "domain": "Engineering", "skills": ["chemical engineering", "process design", "aspen plus", "matlab", "thermodynamics", "process simulation", "safety analysis", "quality control"]},
    {"title": "Aerospace Engineer", "domain": "Engineering", "skills": ["aerospace engineering", "matlab", "simulink", "catia", "cfd", "structural analysis", "flight mechanics", "propulsion"]},
    {"title": "Automotive Engineer", "domain": "Engineering", "skills": ["automotive engineering", "catia", "solidworks", "matlab", "simulink", "vehicle dynamics", "cad", "dfmea"]},
    {"title": "Embedded Systems Engineer", "domain": "Engineering", "skills": ["embedded systems", "c", "c++", "microcontrollers", "rtos", "arm", "embedded c", "circuit design", "firmware development"]},
    {"title": "FPGA Engineer", "domain": "Engineering", "skills": ["fpga", "verilog", "vhdl", "xilinx", "intel fpga", "rtl design", "simulation", "embedded systems"]},
    {"title": "VLSI Engineer", "domain": "Engineering", "skills": ["vlsi", "verilog", "vhdl", "asic design", "fpga", "circuit design", "semiconductor", "physical design", "timing analysis"]},
    {"title": "Robotics Engineer", "domain": "Engineering", "skills": ["robotics", "ros", "python", "c++", "matlab", "control systems", "sensor integration", "machine vision"]},
    {"title": "Automation Engineer", "domain": "Engineering", "skills": ["automation", "plc programming", "scada", "hmi", "industrial automation", "siemens", "abb", "troubleshooting"]},
    {"title": "PLC Programmer", "domain": "Engineering", "skills": ["plc programming", "siemens", "allen-bradley", "ladder logic", "scada", "industrial automation", "troubleshooting", "hmi"]},
    {"title": "Quality Assurance Engineer", "domain": "Engineering", "skills": ["quality assurance", "six sigma", "iso", "root cause analysis", "process improvement", "quality tools", "auditing", "documentation"]},
    {"title": "Quality Control Engineer", "domain": "Engineering", "skills": ["quality control", "testing", "inspection", "iso", "statistical process control", "six sigma", "documentation", "calibration"]},
    {"title": "Project Engineer", "domain": "Engineering", "skills": ["project management", "engineering", "planning", "scheduling", "cost control", "quality management", "stakeholder coordination"]},
    {"title": "Production Engineer", "domain": "Engineering", "skills": ["production management", "lean manufacturing", "process optimization", "six sigma", "quality control", "inventory management", "automation"]},
    {"title": "HVAC Engineer", "domain": "Engineering", "skills": ["hvac", "heating", "ventilation", "air conditioning", "autocad", "revit", "load calculations", "energy efficiency"]},
    {"title": "Environmental Engineer", "domain": "Engineering", "skills": ["environmental engineering", "waste management", "water treatment", "environmental impact assessment", "sustainability", "regulations"]},
    
    # ══════════════════════════════════════════════════════════════════════════
    # EDUCATION & RESEARCH
    # ══════════════════════════════════════════════════════════════════════════
    {"title": "Teacher", "domain": "Education & Research", "skills": ["teaching", "lesson planning", "curriculum development", "classroom management", "student assessment", "communication", "patience"]},
    {"title": "High School Teacher", "domain": "Education & Research", "skills": ["teaching", "subject expertise", "lesson planning", "curriculum", "student assessment", "classroom management", "parent communication"]},
    {"title": "College Professor", "domain": "Education & Research", "skills": ["higher education", "teaching", "research", "academic writing", "curriculum development", "mentoring", "publication"]},
    {"title": "Online Tutor", "domain": "Education & Research", "skills": ["online tutoring", "teaching", "communication", "patience", "subject expertise", "digital tools", "student engagement"]},
    {"title": "Curriculum Designer", "domain": "Education & Research", "skills": ["curriculum development", "instructional design", "e-learning", "adult learning", "assessment design", "lms", "educational technology"]},
    {"title": "Instructional Designer", "domain": "Education & Research", "skills": ["instructional design", "e-learning", "articulate", "captivate", "lms", "content development", "addie model", "evaluation"]},
    {"title": "Corporate Trainer", "domain": "Education & Research", "skills": ["corporate training", "presentation", "communication", "leadership development", "soft skills training", "facilitation", "needs assessment"]},
    {"title": "Learning & Development Manager", "domain": "Education & Research", "skills": ["learning and development", "training management", "lms", "talent development", "succession planning", "performance management", "coaching"]},
    {"title": "Research Analyst", "domain": "Education & Research", "skills": ["research", "data analysis", "spss", "excel", "academic writing", "critical thinking", "literature review", "methodology"]},
    {"title": "Research Scientist", "domain": "Education & Research", "skills": ["research", "scientific method", "data analysis", "publication", "laboratory", "methodology", "grant writing", "collaboration"]},
    {"title": "Data Analyst (Research)", "domain": "Education & Research", "skills": ["data analysis", "python", "r", "spss", "statistics", "visualization", "sql", "research methodology"]},
    {"title": "Academic Writer", "domain": "Education & Research", "skills": ["academic writing", "research", "editing", "proofreading", "citation", "plagiarism", "formatting", "subject expertise"]},
    {"title": "Education Consultant", "domain": "Education & Research", "skills": ["education consulting", "curriculum", "educational technology", "assessment", "training", "presentation", "stakeholder management"]},
    {"title": "Science Teacher", "domain": "Education & Research", "skills": ["science teaching", "biology", "chemistry", "physics", "laboratory", "lesson planning", "student engagement", "assessment"]},
    {"title": "Math Teacher", "domain": "Education & Research", "skills": ["mathematics", "teaching", "problem solving", "algebra", "calculus", "geometry", "student assessment", "curriculum"]},
    {"title": "ESL Teacher", "domain": "Education & Research", "skills": ["esl teaching", "english language", "tefl", "tesol", "lesson planning", "cultural sensitivity", "communication", "assessment"]},
    {"title": "Principal/School Administrator", "domain": "Education & Research", "skills": ["school administration", "educational leadership", "staff management", "curriculum", "budgeting", "compliance", "parent relations"]},
    
    # ══════════════════════════════════════════════════════════════════════════
    # HR & ADMINISTRATION
    # ══════════════════════════════════════════════════════════════════════════
    {"title": "HR Manager", "domain": "HR & Administration", "skills": ["human resources", "recruitment", "employee relations", "performance management", "training", "compliance", "policies", "hris"]},
    {"title": "HR Recruiter", "domain": "HR & Administration", "skills": ["recruitment", "sourcing", "interviewing", "candidate assessment", "linkedin", "ats", "communication", "negotiation"]},
    {"title": "Talent Acquisition Specialist", "domain": "HR & Administration", "skills": ["talent acquisition", "recruitment", "sourcing", "employer branding", "interviewing", "offer management", "market research"]},
    {"title": "HR Generalist", "domain": "HR & Administration", "skills": ["hr generalist", "employee onboarding", "benefits administration", "compliance", "record keeping", "employee relations", "reporting"]},
    {"title": "Payroll Manager", "domain": "HR & Administration", "skills": ["payroll management", "compliance", "tax", "benefits administration", "software", "reporting", "team management"]},
    {"title": "Training Manager", "domain": "HR & Administration", "skills": ["training management", "needs assessment", "program development", "facilitation", "evaluation", "lms", "budget management"]},
    {"title": "Compensation & Benefits Analyst", "domain": "HR & Administration", "skills": ["compensation", "benefits", "data analysis", "excel", "benchmarking", "salary surveys", "job evaluation", "reporting"]},
    {"title": "Employee Relations Manager", "domain": "HR & Administration", "skills": ["employee relations", "conflict resolution", "labor laws", "investigation", "documentation", "communication", "mediation"]},
    {"title": "HR Business Partner", "domain": "HR & Administration", "skills": ["hr business partner", "strategic hr", "coaching", "change management", "data analysis", "stakeholder management", "coaching"]},
    {"title": "Administrative Assistant", "domain": "HR & Administration", "skills": ["administration", "ms office", "calendar management", "scheduling", "communication", "organization", "record keeping"]},
    {"title": "Office Manager", "domain": "HR & Administration", "skills": ["office management", "administration", "vendor management", "facilities", "budget management", "team coordination", "communication"]},
    {"title": "Executive Assistant", "domain": "HR & Administration", "skills": ["executive support", "calendar management", "travel coordination", "communication", "project coordination", "confidentiality", "ms office"]},
    {"title": "Receptionist", "domain": "HR & Administration", "skills": ["reception", "customer service", "phone etiquette", "scheduling", "ms office", "communication", "multitasking"]},
    
    # ══════════════════════════════════════════════════════════════════════════
    # OPERATIONS & SUPPLY CHAIN
    # ══════════════════════════════════════════════════════════════════════════
    {"title": "Operations Manager", "domain": "Operations & Supply Chain", "skills": ["operations management", "process optimization", "project management", "budgeting", "team management", "lean six sigma", "kpis"]},
    {"title": "Supply Chain Analyst", "domain": "Operations & Supply Chain", "skills": ["supply chain", "data analysis", "excel", "sql", "forecasting", "inventory management", "logistics", "process improvement"]},
    {"title": "Logistics Manager", "domain": "Operations & Supply Chain", "skills": ["logistics management", "transportation", "warehouse management", "inventory", "cost optimization", "vendor management", "dispatch"]},
    {"title": "Warehouse Manager", "domain": "Operations & Supply Chain", "skills": ["warehouse management", "inventory control", "logistics", "team management", "safety compliance", "optimization", "reporting"]},
    {"title": "Procurement Manager", "domain": "Operations & Supply Chain", "skills": ["procurement", "vendor management", "negotiation", "contract management", "cost optimization", "supplier relations", "compliance"]},
    {"title": "Purchasing Specialist", "domain": "Operations & Supply Chain", "skills": ["purchasing", "vendor management", "negotiation", "cost analysis", "contract management", "ms office", "market research"]},
    {"title": "Quality Control Manager", "domain": "Operations & Supply Chain", "skills": ["quality control", "iso", "six sigma", "inspection", "process improvement", "documentation", "auditing", "team management"]},
    {"title": "Production Manager", "domain": "Operations & Supply Chain", "skills": ["production management", "manufacturing", "lean", "quality", "scheduling", "cost control", "safety", "team management"]},
    {"title": "Inventory Manager", "domain": "Operations & Supply Chain", "skills": ["inventory management", "forecasting", "stock control", "warehouse", "data analysis", "optimization", "reporting"]},
    {"title": "Demand Planner", "domain": "Operations & Supply Chain", "skills": ["demand planning", "forecasting", "s&op", "data analysis", "excel", "inventory optimization", "collaboration", "reporting"]},
]


def calculate_skill_match_score(user_skills, job_skills):
    """
    Calculate match score based on skill overlap.
    Returns a score from 0 to 100 representing percentage of job requirements matched.
    """
    if not job_skills:
        return 0
    
    user_skills_lower = set([s.lower() for s in user_skills])
    job_skills_lower = set([s.lower() for s in job_skills])
    
    # Count how many required skills user has
    matched_skills = user_skills_lower.intersection(job_skills_lower)
    
    # Calculate percentage of job requirements matched
    match_percentage = (len(matched_skills) / len(job_skills_lower)) * 100
    
    return match_percentage


def normalize_scores(raw_scores, method='softmax'):
    """
    Normalize raw scores to get proportional distribution.
    Uses different normalization methods to ensure realistic spread.
    """
    if not raw_scores:
        return []
    
    # Convert to list of floats
    scores = [float(s) for s in raw_scores]
    
    if all(s == 0 for s in scores):
        return [0.0] * len(scores)
    
    if method == 'softmax':
        # Softmax normalization for gradual distribution
        # Shift scores to prevent overflow
        max_score = max(scores)
        shifted = [s - max_score for s in scores]
        exp_scores = [math.exp(s / 10) for s in shifted]  # Divide by 10 to reduce sharpness
        total = sum(exp_scores)
        normalized = [(e / total) * 100 for e in exp_scores]
    elif method == 'rank':
        # Rank-based normalization
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        normalized = [0.0] * len(scores)
        # Assign scores based on rank (highest gets highest normalized score)
        for rank, idx in enumerate(sorted_indices):
            normalized[idx] = 100 - (rank * (100 / len(scores)))
    elif method == 'proportional':
        # Direct proportional normalization
        total = sum(scores)
        if total == 0:
            return [0.0] * len(scores)
        normalized = [(s / total) * 100 for s in scores]
    elif method == 'realistic':
        # Realistic scoring with maximum cap between 60-70%
        # This prevents unrealistic 90-100% matches
        total = sum(scores)
        if total == 0:
            return [0.0] * len(scores)
        
        # First, get proportional distribution
        proportional = [(s / total) * 100 for s in scores]
        
        # Find the maximum score
        max_score = max(proportional)
        
        # Set the cap - top job gets between 60-70%
        # Using 65% as the target maximum (middle of 60-70 range)
        target_max = 65.0
        
        if max_score > 0:
            # Scale all scores so the maximum equals target_max
            scale_factor = target_max / max_score
            normalized = [s * scale_factor for s in proportional]
        else:
            normalized = proportional
    else:
        normalized = scores
    
    return normalized


def recommend_jobs(user_skills, jobs=None):
    """
    Recommend jobs based on user skills with ENHANCED multi-factor scoring.
    Uses sophisticated algorithm with weighted skills, partial matching,
    domain alignment, penalties, and bonuses for realistic varied scores.
    """
    if not user_skills:
        return []

    job_pool = jobs if jobs else ALL_JOBS

    if not job_pool:
        return []

    # Detect user's primary domain
    primary_domain = get_domain_from_skills(user_skills)
    
    # Calculate enhanced match scores
    scored_jobs = []
    for job in job_pool:
        job_skills = job.get("skills", [])
        job_domain = job.get("domain", "General")
        
        # Use enhanced scoring algorithm
        score = enhanced_calculate_match_score(user_skills, job_skills, job_domain)
        
        scored_jobs.append({
            "job": job,
            "score": score
        })
    
    # Sort by score (highest first)
    scored_jobs.sort(key=lambda x: -x["score"])
    
    # Apply realistic distribution - ensure varied scores
    # Add small random variation to create realistic spread
    import random
    random.seed(42)  # For reproducibility
    
    results = []
    for i, item in enumerate(scored_jobs):
        job = item["job"]
        base_score = item["score"]
        
        # Add position-based variation to ensure diverse scores
        # Jobs further down the list get progressively lower scores with gaps
        position_factor = 0
        if i == 0:
            position_factor = random.uniform(0, 3)  # Top job: 0-3% variation
        elif i <= 3:
            position_factor = random.uniform(-2, 2)  # Top 3: ±2%
        elif i <= 10:
            position_factor = random.uniform(-3, 1)  # Next 7: -3% to +1%
        else:
            position_factor = random.uniform(-2, 0)  # Rest: gradual decrease
        
        final_score = max(0, min(95, base_score + position_factor))
        
        # Ensure scores are varied and realistic
        # Create natural gaps between similar jobs
        if i > 0 and abs(final_score - results[-1]["score"]) < 2:
            final_score = max(0, final_score - random.uniform(2, 5))
        
        results.append({
            "title": job.get("title", "Unknown"),
            "score": round(final_score, 1),
            "domain": job.get("domain", "General"),
            "required_skills": job.get("skills", []),
            "is_primary_domain": job.get("domain", "General") == primary_domain
        })

    return results


def get_skill_gap(user_skills, job):
    """Return missing skills for a given job dict."""
    required = set([s.lower() for s in job.get("required_skills", [])])
    user = set([s.lower() for s in user_skills])
    return sorted(list(required - user))


def get_match_level(score):
    """Return match level description based on enhanced scoring (max 95%)."""
    if score >= 75:
        return "excellent"
    elif score >= 60:
        return "strong"
    elif score >= 45:
        return "moderate"
    elif score >= 30:
        return "developing"
    else:
        return "entry"


def get_motivational_message(score, job_title, missing_skills, user_skills):
    """
    Return an encouraging, confidence-boosting message based on enhanced match score.
    Enhanced scoring produces varied results from 0-95% reflecting true skill alignment.
    """
    top_missing = missing_skills[:3] if missing_skills else []
    match_level = get_match_level(score)
    
    if match_level == "excellent":
        if top_missing:
            return f"Outstanding match for {job_title}! Adding {', '.join(top_missing[:2])} would make you exceptional."
        return f"Excellent fit for {job_title}! Your skills align very well with this role."
    
    elif match_level == "strong":
        if top_missing:
            return f"You are a strong match for {job_title}! With {', '.join(top_missing[:2])} skills, you could reach the top of this field."
        return f"Impressive match for {job_title}! You're well-positioned for this role."
    
    elif match_level == "moderate":
        if top_missing:
            skills_str = ', '.join(top_missing[:2])
            return f"Improving {skills_str} can significantly boost your fit for {job_title}."
        return f"Good foundation! A few targeted skills can make you an ideal candidate."
    
    elif match_level == "developing":
        if top_missing:
            return f"Focus on {', '.join(top_missing[:2])} to increase your match for {job_title}."
        return f"You're building toward this role. Keep developing your skills!"
    
    else:
        if top_missing:
            return f"Building skills in {', '.join(top_missing[:2])} is a great start toward {job_title}."
        return f"Every expert started somewhere! Explore learning paths for this role."


def get_match_description(score):
    """Return user-friendly description of match level (enhanced scoring 0-95%)."""
    if score >= 75:
        return "Excellent match — highly aligned with requirements"
    elif score >= 60:
        return "Strong match — you're on the right path"
    elif score >= 45:
        return "Good potential with room to grow"
    elif score >= 30:
        return "Building toward this role"
    else:
        return "Exploring opportunities"


def get_skill_advice(skill, job_title, is_missing):
    """Return advice about a specific skill for a job."""
    if is_missing:
        return f"Improving {skill} can increase your chances for {job_title}"
    else:
        return f"This skill helps you qualify for {job_title}"


def get_domain_advice(primary_domain, user_skills):
    """
    Return domain-specific career advice.
    """
    domain_advice = {
        "IT & Software": "Your technical skills are in high demand! Focus on cloud expertise and staying updated with latest technologies.",
        "Finance & Accounting": "Your finance expertise is valuable! Consider adding data analytics and visualization skills.",
        "Healthcare & Medical": "Healthcare is a growing field! Continuing education and certifications can boost your career.",
        "Marketing & Sales": "Digital skills are key in modern marketing! Focus on analytics and automation.",
        "Design & Creative": "Your creative skills are essential! Building a strong portfolio is crucial.",
        "Engineering": "Engineering roles are evolving! Learn about sustainable practices and digital transformation.",
        "Education & Research": "Education is being transformed by technology! E-learning skills are increasingly valuable.",
        "HR & Administration": "HR is becoming more strategic! Data-driven HR and employee experience skills are in demand.",
        "Operations & Supply Chain": "Supply chain is being revolutionized! Focus on analytics and automation.",
        "General": "Your diverse skill set is valuable! Consider specializing to stand out."
    }
    
    return domain_advice.get(primary_domain, domain_advice["General"])

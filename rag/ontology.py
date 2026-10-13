import hashlib
import re
import unicodedata

ONTOLOGY_VERSION = "roleup-ontology-4"
LEVELS = ("Intern", "Fresher", "Junior")


def normalized(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold().strip()
    return re.sub(r"\s+", " ", text)


ALIASES = {
    "embeddings": "embedding",
    "embedding vectors": "embedding",
    "vector databases": "vector database",
    "vector search": "retrieval",
    "semantic search": "retrieval",
    "rag": "retrieval augmented generation",
    "retrieval-augmented generation": "retrieval augmented generation",
    "vector db": "vector database",
    "vectordb": "vector database",
    "llm": "large language model",
    "llms": "large language model",
    "jwt": "json web token",
    "js": "javascript",
    "reactjs": "react",
    "react.js": "react",
    "spring": "spring boot",
    "postgres": "postgresql",
    "nodejs": "node.js",
    "node js": "node.js",
    "tf": "tensorflow",
    "sklearn": "scikit-learn",
    "k8s": "kubernetes",
    "ci/cd": "continuous integration",
    "cicd": "continuous integration",
}

SKILL_CONCEPTS = {
    "python", "java", "javascript", "typescript", "c", "c++", "c#", "go",
    "golang", "php", "ruby", "kotlin", "swift", "scala", "r", "sql",
    "html", "css", "node.js", "django", "fastapi", "flask", "spring boot",
    "react", "vue", "angular", "next.js", "postgresql", "mysql", "sqlite",
    "mongodb", "redis", "elasticsearch", "kafka", "spark", "airflow", "dbt",
    "docker", "kubernetes", "terraform", "aws", "azure", "gcp", "linux",
    "git", "github actions", "jenkins", "pytorch", "tensorflow", "scikit-learn",
    "pandas", "numpy", "machine learning", "deep learning", "natural language processing",
    "computer vision", "large language model", "retrieval augmented generation",
    "vector database", "postgresql", "graphql", "rest api", "grpc", "microservices",
    "system design", "unit testing", "integration testing", "continuous integration",
}


def canonical(label: str) -> str:
    label = normalized(label)
    return ALIASES.get(label, label)


def entity_id(label: str) -> str:
    return "concept:" + hashlib.sha256(canonical(label).encode()).hexdigest()[:20]


def tokens(text: str) -> list[str]:
    return re.findall(r"[\w]+(?:[.+#/-][\w+#]+)*|c\+\+|c#|\.net", normalized(text))


def mentions(text: str, label: str) -> bool:
    text = normalized(text)
    label = canonical(label)
    variants = {label} | {a for a, b in ALIASES.items() if b == label}
    return any(re.search(r"(?<!\w)" + re.escape(v) + r"(?!\w)", text) for v in variants)


# Rule-defined concept relations. They are inspectable seeds, not human-certified facts.
# Every edge is attributed to this ontology version and flagged rule_defined.
RELATIONS = [
    ("retrieval augmented generation", "HAS_COMPONENT", "retrieval"),
    ("retrieval augmented generation", "HAS_COMPONENT", "vector database"),
    ("retrieval augmented generation", "HAS_COMPONENT", "embedding"),
    ("retrieval augmented generation", "EVALUATED_BY", "faithfulness"),
    ("retrieval", "EVALUATED_BY", "recall"),
    ("retrieval", "EVALUATED_BY", "precision"),
    ("vector database", "USES_CONCEPT", "embedding"),
    ("large language model", "HAS_RISK", "hallucination"),
    ("pytorch", "USES_CONCEPT", "deep learning"),
    ("deep learning", "HAS_PREREQUISITE", "machine learning"),
    ("spring boot", "USES_CONCEPT", "dependency injection"),
    ("spring boot", "HAS_PREREQUISITE", "java"),
    ("react", "USES_CONCEPT", "virtual dom"),
    ("react", "USES_CONCEPT", "hooks"),
    ("react", "HAS_PREREQUISITE", "javascript"),
    ("javascript", "USES_CONCEPT", "event loop"),
    ("sql", "USES_CONCEPT", "index"),
    ("sql", "USES_CONCEPT", "transaction"),
    ("docker", "USES_CONCEPT", "container"),
    ("kubernetes", "USES_CONCEPT", "container"),
    ("json web token", "USES_CONCEPT", "authentication"),
]

ROLE_PATTERNS = [
    ("IF_AGENTIC_AI", r"agentic|ai agent|ai tác tử"),
    ("IF_AI_SWE", r"ai software|llm|\brag\b|genai"),
    ("IF_DS", r"data scientist|data science"),
    ("IF_DE", r"data engineer|\betl\b|database engineer|\bdba\b"),
    ("IF_DA", r"data analyst|business intelligence"),
    ("IF_AI", r"ai engineer|machine learning|deep learning|trí tuệ nhân tạo"),
    ("IF_AUTO_QA", r"automation.*(?:qa|test|qc)|(?:qa|test|qc).*automation"),
    ("IF_QA_QC", r"\bqa\b|\bqc\b|tester|testing|kiểm thử"),
    ("IF_EMBEDDED", r"embedded|firmware|autosar|vi điều khiển"),
    ("IF_UNITY", r"unity|game developer|game engineer"),
    ("IF_MOBILE_IOS", r"\bios\b|swift"),
    ("IF_MOBILE_ANDROID", r"android|kotlin|flutter|mobile developer"),
    ("IF_DEVOPS", r"devops|\bsre\b|cloud engineer"),
    ("IF_NETWORK", r"network|sysadmin|system engineer|infrastructure"),
    ("IF_ERP", r"\berp\b|odoo|\bsap\b"),
    ("IF_PRODUCT_ENG", r"product engineer|growth engineer"),
    ("IF_BA", r"business analyst|phân tích nghiệp vụ"),
    ("IF_FULLSTACK", r"full[ -]?stack"),
    ("IF_BE_JAVA", r"\bjava\b|spring"),
    ("IF_BE_NET", r"\.net|c#|dotnet"),
    ("IF_BE_PHP", r"\bphp\b|laravel"),
    ("IF_FE", r"front[ -]?end|react|vue|angular"),
    ("IF_BE_GEN", r"back[ -]?end|python|django|fastapi|node\.?js|golang"),
    ("IF_SWE", r"software engineer"),
    ("IF_SWD", r"software developer|lập trình viên phần mềm"),
]


def resolve_role(text: str, framework_ids: set[str]) -> str | None:
    for role, pattern in ROLE_PATTERNS:
        if role in framework_ids and re.search(pattern, normalized(text)):
            return role
    return None


def resolve_level(text: str) -> str | None:
    value = normalized(text)
    # Never silently turn a Senior-only JD into a Fresher interview.
    for level in reversed(LEVELS):
        if normalized(level) in value:
            return level
    if "thực tập" in value or "internship" in value:
        return "Intern"
    if "mới tốt nghiệp" in value or "entry level" in value:
        return "Fresher"
    return None

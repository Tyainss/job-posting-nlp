import re
from collections.abc import Iterable


FALLBACK_CATEGORY = "Other"

KEYWORD_RULES: dict[str, tuple[str, ...]] = {
    "Data Analyst": (
        "business intelligence",
        "dashboard",
        "reporting",
        "data visualization",
        "tableau",
        "power bi",
        "excel",
    ),
    "Data Engineer": (
        "data pipeline",
        "etl",
        "elt",
        "data warehouse",
        "data lake",
        "airflow",
        "apache spark",
        "databricks",
    ),
    "Data Scientist": (
        "data science",
        "statistics",
        "predictive",
        "a/b testing",
        "experimentation",
        "experiment",
        "forecasting",
    ),
    "Machine Learning Engineer": (
        "mlops",
        "model deployment",
        "model serving",
        "model monitoring",
        "deep learning",
        "pytorch",
        "tensorflow",
        "machine learning systems",
    ),
    "DevOps / Cloud Engineer": (
        "ci/cd",
        "infrastructure as code",
        "terraform",
        "kubernetes",
        "site reliability",
        "container orchestration",
        "cloud infrastructure",
    ),
    "Software Engineer": (
        "backend",
        "frontend",
        "full stack",
        "software development",
        "microservice",
        "object oriented",
        "rest api",
        "c++",
        ".net",
    ),
}


_PLURAL_KEYWORDS = {
    "dashboard",
    "data lake",
    "data pipeline",
    "data warehouse",
    "feature store",
    "inference service",
    "microservice",
    "rest api",
    "training pipeline",
}


def _compile_keyword(keyword: str) -> re.Pattern[str]:
    keyword_pattern = r"(?:\s+|-)".join(
        re.escape(part) for part in keyword.split()
    )
    plural_suffix = "s?" if keyword in _PLURAL_KEYWORDS else ""

    return re.compile(
        rf"(?<!\w){keyword_pattern}{plural_suffix}(?!\w)",
        flags=re.IGNORECASE,
    )


_KEYWORD_PATTERNS = {
    category: tuple(_compile_keyword(keyword) for keyword in keywords)
    for category, keywords in KEYWORD_RULES.items()
}


def predict_keyword_role(description: str) -> str:
    """Predict one role from fixed description-level keyword evidence."""
    scores = {
        category: sum(
            pattern.search(description) is not None
            for pattern in patterns
        )
        for category, patterns in _KEYWORD_PATTERNS.items()
    }

    highest_score = max(scores.values())
    if highest_score == 0:
        return FALLBACK_CATEGORY

    leading_categories = [
        category
        for category, score in scores.items()
        if score == highest_score
    ]

    if len(leading_categories) != 1:
        return FALLBACK_CATEGORY

    return leading_categories[0]


def predict_keyword_baseline(descriptions: Iterable[str]) -> list[str]:
    """Predict roles for a collection of cleaned descriptions."""
    return [predict_keyword_role(description) for description in descriptions]

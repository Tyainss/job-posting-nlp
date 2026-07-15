import pytest

from job_posting_nlp.baselines import predict_keyword_role


@pytest.mark.parametrize(
    ("description", "expected_role"),
    [
        (
            "Build ETL data pipelines with Airflow.",
            "Data Engineer",
        ),
        (
            "Improve CI/CD automation.",
            "DevOps / Cloud Engineer",
        ),
        (
            "Maintain C++ applications.",
            "Software Engineer",
        ),
    ],
)
def test_clear_role_evidence_predicts_expected_category(
    description: str,
    expected_role: str,
):
    assert predict_keyword_role(description) == expected_role


def test_no_match_and_tied_evidence_use_other_fallback():
    assert predict_keyword_role("Support customers and manage accounts.") == "Other"
    assert predict_keyword_role("Build dashboards and data pipelines.") == "Other"

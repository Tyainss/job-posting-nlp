import pandas as pd

from job_posting_nlp.preprocessing import (
    clean_description,
    prepare_modeling_rows,
)


def test_clean_description_applies_observed_transformations():
    description = (
        "Build C++ &amp; C#.\r\n"
        "See https://example.com/jobs.\t Next\ufeff"
        "We\u200Dcanâ\x80\x99t wait."
    )

    assert clean_description(description) == (
        "Build C++ & C#. See [URL]. Next We can't wait."
    )


def test_clean_description_preserves_technical_punctuation():
    description = "C++ C# .NET CI/CD Node.js"

    assert clean_description(description) == description


def test_prepare_modeling_rows_excludes_and_deduplicates_as_intended():
    postings = pd.DataFrame(
        [
            {
                "job_id": 1,
                "title": "Software Engineer",
                "role_category": "Software Engineer",
                "company_id": 10,
                "company_name": "A",
                "description": "https://example.com/job",
            },
            {
                "job_id": 2,
                "title": "Software Engineer",
                "role_category": "Software Engineer",
                "company_id": 10,
                "company_name": "A",
                "description": "Build C++\nservices",
            },
            {
                "job_id": 3,
                "title": "Senior Software Engineer",
                "role_category": "Software Engineer",
                "company_id": 11,
                "company_name": "B",
                "description": "Build C++ services",
            },
            {
                "job_id": 4,
                "title": "Data Engineer",
                "role_category": "Data Engineer",
                "company_id": 11,
                "company_name": "B",
                "description": "Build C++ services",
            },
        ]
    )

    prepared = prepare_modeling_rows(postings)

    assert prepared["job_id"].tolist() == [2, 4]
    assert prepared["cleaned_description"].tolist() == [
        "Build C++ services",
        "Build C++ services",
    ]
    assert prepared["role_category"].tolist() == [
        "Software Engineer",
        "Data Engineer",
    ]

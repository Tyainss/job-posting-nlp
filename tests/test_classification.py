from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from job_posting_nlp.classification import build_tfidf_logistic_pipeline


def test_tfidf_logistic_pipeline_structure_and_fit():
    pipeline = build_tfidf_logistic_pipeline(
        ngram_range=(1, 2),
        class_weight="balanced",
        min_df=2,
    )

    assert isinstance(pipeline, Pipeline)
    assert isinstance(pipeline.named_steps["tfidf"], TfidfVectorizer)
    assert isinstance(pipeline.named_steps["classifier"], LogisticRegression)
    assert pipeline.named_steps["tfidf"].ngram_range == (1, 2)
    assert pipeline.named_steps["tfidf"].min_df == 2
    assert pipeline.named_steps["classifier"].class_weight == "balanced"

    descriptions = [
        "build backend services",
        "maintain backend services",
        "build data pipelines",
        "maintain data pipelines",
    ]
    labels = [
        "Software Engineer",
        "Software Engineer",
        "Data Engineer",
        "Data Engineer",
    ]

    pipeline.fit(descriptions, labels)
    predictions = pipeline.predict(descriptions)

    assert len(predictions) == len(descriptions)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_tfidf_logistic_pipeline(
    *,
    ngram_range: tuple[int, int],
    class_weight: str | dict[str, float] | None,
    min_df: int,
) -> Pipeline:
    """Build a TF-IDF and logistic-regression pipeline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=ngram_range,
                    min_df=min_df,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    solver="lbfgs",
                    C=1.0,
                    class_weight=class_weight,
                    max_iter=1000,
                ),
            ),
        ]
    )

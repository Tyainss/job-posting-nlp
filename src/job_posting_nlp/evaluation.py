import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold


RANDOM_STATE = 42   # (Answer to life, universe and everything)
N_SPLITS = 5


def create_development_test_split(
    modeling_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the grouped 80/20 development-test split."""
    holdout_splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # Five folds give us an approximately 20% holdout. We use the first one as test.
    development_indices, test_indices = next(
        holdout_splitter.split(
            modeling_rows,
            y=modeling_rows["role_category"],
            groups=modeling_rows["cleaned_description"],
        )
    )

    return (
        modeling_rows.iloc[development_indices].copy(),
        modeling_rows.iloc[test_indices].copy(),
    )


def make_grouped_cv() -> StratifiedGroupKFold:
    """Create the grouped five-fold splitter used inside development data."""
    return StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def classification_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> dict[str, float]:
    """Calculate the shared classification metrics."""
    return {
        "macro F1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "weighted F1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "accuracy": accuracy_score(y_true, y_pred),
    }

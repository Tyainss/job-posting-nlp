import pandas as pd
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

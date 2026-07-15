import numpy as np
import pandas as pd

from job_posting_nlp.evaluation import (
    create_development_test_split,
    make_grouped_cv,
)


def make_split_rows() -> pd.DataFrame:
    rows = [
        {
            "role_category": role_category,
            "cleaned_description": f"{role_category} description {group_number}",
        }
        for role_category in ["Data", "Software"]
        for group_number in range(5)
    ]
    rows.extend(
        [
            {
                "role_category": "Data",
                "cleaned_description": "Conflicting description",
            },
            {
                "role_category": "Software",
                "cleaned_description": "Conflicting description",
            },
        ]
    )

    return pd.DataFrame(rows)


def test_development_test_split_is_deterministic_and_grouped():
    rows = make_split_rows()

    development_rows, test_rows = create_development_test_split(rows)
    repeated_development_rows, repeated_test_rows = (
        create_development_test_split(rows)
    )

    assert development_rows.index.equals(repeated_development_rows.index)
    assert test_rows.index.equals(repeated_test_rows.index)
    assert set(development_rows.index).isdisjoint(test_rows.index)
    assert sorted([*development_rows.index, *test_rows.index]) == list(rows.index)
    assert set(development_rows["cleaned_description"]).isdisjoint(
        test_rows["cleaned_description"]
    )

    conflicting_indices = rows.index[
        rows["cleaned_description"].eq("Conflicting description")
    ]
    assert (
        conflicting_indices.isin(development_rows.index).all()
        or conflicting_indices.isin(test_rows.index).all()
    )


def test_grouped_cv_is_deterministic_and_keeps_groups_isolated():
    rows = make_split_rows()
    y = rows["role_category"]
    groups = rows["cleaned_description"]

    folds = list(make_grouped_cv().split(rows, y=y, groups=groups))
    repeated_folds = list(
        make_grouped_cv().split(rows, y=y, groups=groups)
    )

    validation_counts = np.zeros(len(rows), dtype=int)

    for (train_indices, validation_indices), repeated_fold in zip(
        folds,
        repeated_folds,
        strict=True,
    ):
        repeated_train_indices, repeated_validation_indices = repeated_fold

        assert np.array_equal(train_indices, repeated_train_indices)
        assert np.array_equal(validation_indices, repeated_validation_indices)
        assert set(groups.iloc[train_indices]).isdisjoint(
            groups.iloc[validation_indices]
        )

        validation_counts[validation_indices] += 1

    assert np.all(validation_counts == 1)

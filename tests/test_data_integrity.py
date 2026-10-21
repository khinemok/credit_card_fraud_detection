
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"

EXPECTED_FEATURES = {
    *(f"V{i}" for i in range(1, 29)),
    "Amount",
}
REQUIRED_COLUMNS = EXPECTED_FEATURES | {"Class"}
SPLIT_FILES = {
    "train": SPLIT_DIR / "train.csv",
    "validation": SPLIT_DIR / "validation.csv",
    "test": SPLIT_DIR / "test.csv",
}


@pytest.fixture(scope="module")
def splits():
    missing = [
        str(path)
        for path in SPLIT_FILES.values()
        if not path.exists()
    ]
    if missing:
        pytest.fail(
            "Required processed split files were not found: "
            + ", ".join(missing)
            + ". Update SPLIT_DIR to your actual split location."
        )

    return {
        name: pd.read_csv(path)
        for name, path in SPLIT_FILES.items()
    }


def test_split_schema_and_predictors(splits):
    for name, df in splits.items():
        assert REQUIRED_COLUMNS.issubset(df.columns), (
            f"{name} is missing required columns: "
            f"{REQUIRED_COLUMNS - set(df.columns)}"
        )
        assert df.columns.is_unique, (
            f"{name} has duplicate column names"
        )
        assert "Time" not in EXPECTED_FEATURES
        assert "Class" not in EXPECTED_FEATURES


def test_target_labels_and_missing_values(splits):
    for name, df in splits.items():
        assert df["Class"].notna().all(), (
            f"{name} contains missing target labels"
        )
        assert set(df["Class"].unique()).issubset({0, 1}), (
            f"{name} contains labels other than 0 and 1"
        )
        assert not df[list(EXPECTED_FEATURES)].isna().any().any(), (
            f"{name} contains missing predictor values"
        )
        assert df[list(EXPECTED_FEATURES)].apply(
            lambda col: pd.to_numeric(col, errors="coerce").notna().all()
        ).all(), f"{name} contains non-numeric predictors"


def test_splits_are_nonempty(splits):
    for name, df in splits.items():
        assert len(df) > 0, f"{name} split is empty"


def test_split_rows_do_not_overlap(splits):
    # Use all shared columns for record identity, including Time.
    # Time is excluded from model predictors but can distinguish
    # otherwise identical transactions.
    for first, second in [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]:
        common_columns = sorted(
            set(splits[first].columns)
            & set(splits[second].columns)
        )

        left = splits[first][common_columns].drop_duplicates()
        right = splits[second][common_columns].drop_duplicates()

        overlap = left.merge(
            right,
            how="inner",
            on=common_columns,
        )

        assert overlap.empty, (
            f"Identical complete records occur in "
            f"{first} and {second}: {len(overlap)} unique records"
        )



def test_duplicate_predictors_do_not_have_conflicting_labels(splits):
    # This checks conflicting labels within each partition.
    # Cross-partition checks require source-row identity or a
    # dataset-level audit as well.
    for name, df in splits.items():
        grouped = df.groupby(
            sorted(EXPECTED_FEATURES),
            dropna=False,
        )["Class"].nunique()

        assert (grouped <= 1).all(), (
            f"{name} contains identical predictor rows "
            "with conflicting labels"
        )


def test_each_split_is_chronologically_ordered(splits):
    for name, df in splits.items():
        assert "Time" in df.columns, (
            f"{name} is missing the Time column"
        )

        time_values = pd.to_numeric(
            df["Time"], errors="raise"
        )

        assert time_values.notna().all(), (
            f"{name} contains missing timestamps"
        )

        assert time_values.is_monotonic_increasing, (
            f"{name} is not ordered by Time"
        )

def test_chronological_split_boundaries(splits):
    train_max = splits["train"]["Time"].max()
    validation_min = splits["validation"]["Time"].min()
    validation_max = splits["validation"]["Time"].max()
    test_min = splits["test"]["Time"].min()

    assert train_max <= validation_min, (
        f"Training ends at {train_max}, "
        f"after validation begins at {validation_min}"
    )

    assert validation_max <= test_min, (
        f"Validation ends at {validation_max}, "
        f"after test begins at {test_min}"
    )

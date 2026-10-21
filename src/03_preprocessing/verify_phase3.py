
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

EXPECTED = {
    "train": (170884, 170524, 360),
    "validation": (56961, 56904, 57),
    "test": (56962, 56887, 75),
}


def verify_split(split, expected):
    expected_rows, expected_legitimate, expected_fraud = expected

    base = ROOT / "outputs" / "preprocessing"
    tree_dir = base / "unscaled"
    lr_dir = base / "logistic_regression_scaled"

    paths = {
        "tree_X": tree_dir / f"{split}_X.csv",
        "tree_y": tree_dir / f"{split}_y.csv",
        "lr_X": lr_dir / f"{split}_X_scaled.csv",
        "lr_y": lr_dir / f"{split}_y.csv",
    }

    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {name} file for {split}: {path}"
            )

    X = pd.read_csv(paths["tree_X"])
    y = pd.read_csv(paths["tree_y"])["Class"]
    X_lr = pd.read_csv(paths["lr_X"])
    y_lr = pd.read_csv(paths["lr_y"])["Class"]

    # Verify row counts and target alignment.
    assert len(X) == expected_rows, f"{split}: incorrect X row count"
    assert len(y) == expected_rows, f"{split}: incorrect y row count"
    assert len(X_lr) == expected_rows, f"{split}: incorrect scaled row count"
    assert len(y_lr) == expected_rows, f"{split}: incorrect scaled y row count"
    assert y.equals(y_lr), f"{split}: target labels/order differ"

    # Verify class counts and valid binary labels.
    assert set(y.unique()) == {0, 1}, f"{split}: labels must be 0 and 1"
    assert int((y == 0).sum()) == expected_legitimate, (
        f"{split}: legitimate count mismatch"
    )
    assert int((y == 1).sum()) == expected_fraud, (
        f"{split}: fraud count mismatch"
    )

    # Verify the feature schema.
    assert list(X.columns) == list(X_lr.columns), (
        f"{split}: scaled and unscaled feature columns differ"
    )
    assert X.shape[1] == 29, f"{split}: expected 29 predictors"
    assert "Class" not in X.columns, f"{split}: target leaked into X"
    assert "Time" not in X.columns, f"{split}: Time unexpectedly included"

    # Verify all predictor values are finite.
    assert np.isfinite(X.to_numpy()).all(), (
        f"{split}: non-finite unscaled values"
    )
    assert np.isfinite(X_lr.to_numpy()).all(), (
        f"{split}: non-finite scaled values"
    )

    print(
        f"PASS {split}: rows={len(X)}, features={X.shape[1]}, "
        f"legitimate={int((y == 0).sum())}, "
        f"fraud={int((y == 1).sum())}"
    )


def main():
    print("Verifying Phase 3 preprocessing outputs...\n")

    for split, expected in EXPECTED.items():
        verify_split(split, expected)

    print("\nSUCCESS: All Phase 3 output checks passed.")


if __name__ == "__main__":
    main()

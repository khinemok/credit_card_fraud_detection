#!/usr/bin/env python3
"""
Phase 3: Data preprocessing for credit-card fraud detection.

Expected project layout:
  data/processed/splits/train.csv
  data/processed/splits/validation.csv
  data/processed/splits/test.csv

Run from anywhere:
  python src/03_preprocessing/prepare_phase3.py

This script does not modify the Phase 2 split CSVs.
It:
- validates each split;
- uses Time for audit/order only and excludes it from predictors;
- separates Class from predictors;
- writes unscaled feature matrices for tree models;
- fits StandardScaler on TRAIN ONLY for Logistic Regression;
- transforms validation/test with that already-fitted scaler;
- saves scaler, manifest, and validation report.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler


EXPECTED_TARGET = "Class"
ORDER_ONLY_COLUMN = "Time"
SPLITS = ("train", "validation", "test")


def project_root() -> Path:
    # Script is expected at <project>/src/03_preprocessing/prepare_phase3.py.
    return Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fail(message: str) -> None:
    raise ValueError(message)


def load_and_validate(path: Path, split_name: str) -> pd.DataFrame:
    if not path.exists():
        fail(
            f"Missing {split_name} input: {path}\n"
            "Run the completed Phase 2 validation/splitting script first."
        )

    df = pd.read_csv(path)
    if df.empty:
        fail(f"{split_name} split is empty: {path}")
    if EXPECTED_TARGET not in df.columns:
        fail(f"{split_name} split has no '{EXPECTED_TARGET}' target column.")

    if df.columns.duplicated().any():
        duplicates = df.columns[df.columns.duplicated()].tolist()
        fail(f"{split_name} split has duplicate column names: {duplicates}")

    target = df[EXPECTED_TARGET]
    if target.isna().any():
        fail(f"{split_name} contains missing target labels.")
    labels = set(pd.to_numeric(target, errors="coerce").dropna().unique().tolist())
    if target.isna().sum() or labels != {0, 1}:
        fail(
            f"{split_name} target must contain only numeric 0 and 1 labels; "
            f"observed labels: {sorted(labels)}"
        )

    # All predictors must be numeric and finite. Time is checked too if present.
    non_target = [c for c in df.columns if c != EXPECTED_TARGET]
    for col in non_target:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.isna().any():
            fail(f"{split_name} column '{col}' contains missing or non-numeric values.")
        values = converted.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            fail(f"{split_name} column '{col}' contains non-finite values.")
        df[col] = converted

    df[EXPECTED_TARGET] = pd.to_numeric(df[EXPECTED_TARGET], errors="raise").astype("int8")

    # Verify the split's chronological order if Time is available.
    if ORDER_ONLY_COLUMN in df.columns:
        if not df[ORDER_ONLY_COLUMN].is_monotonic_increasing:
            fail(f"{split_name} is not ordered by non-decreasing Time.")

    return df


def main() -> int:
    root = project_root()
    split_dir = root / "data" / "processed" / "splits"
    output_dir = root / "outputs" / "preprocessing"
    unscaled_dir = output_dir / "unscaled"
    scaled_dir = output_dir / "logistic_regression_scaled"
    artifact_dir = root / "artifacts" / "preprocessing"

    for directory in (unscaled_dir, scaled_dir, artifact_dir):
        directory.mkdir(parents=True, exist_ok=True)

    input_paths = {name: split_dir / f"{name}.csv" for name in SPLITS}
    frames = {name: load_and_validate(path, name) for name, path in input_paths.items()}

    # Require exactly the same columns in the same order in every split.
    train_columns = list(frames["train"].columns)
    for name, df in frames.items():
        if list(df.columns) != train_columns:
            fail(
                f"Column names/order differ in {name}.csv compared with train.csv.\n"
                f"Train columns: {train_columns}\n{name} columns: {list(df.columns)}"
            )

    if ORDER_ONLY_COLUMN not in train_columns:
        print(
            "WARNING: 'Time' is absent. The script can continue, but confirm "
            "chronological ordering was already enforced in Phase 2.",
            file=sys.stderr,
        )

    # Exclude target and Time from primary predictors.
    feature_columns = [
        col for col in train_columns
        if col not in (EXPECTED_TARGET, ORDER_ONLY_COLUMN)
    ]
    if not feature_columns:
        fail("No predictor columns remain after excluding Class and Time.")

    # Write consistent unscaled X and y files for tree models and general use.
    split_summary = {}
    for name, df in frames.items():
        X = df[feature_columns].copy()
        y = df[EXPECTED_TARGET].copy()

        X.to_csv(unscaled_dir / f"{name}_X.csv", index=False)
        y.to_frame(name=EXPECTED_TARGET).to_csv(unscaled_dir / f"{name}_y.csv", index=False)

        split_summary[name] = {
            "rows": int(len(df)),
            "class_0": int((y == 0).sum()),
            "class_1": int((y == 1).sum()),
            "fraud_prevalence_percent": float((y == 1).mean() * 100),
            "time_min": float(df[ORDER_ONLY_COLUMN].min()) if ORDER_ONLY_COLUMN in df else None,
            "time_max": float(df[ORDER_ONLY_COLUMN].max()) if ORDER_ONLY_COLUMN in df else None,
            "input_file": str(input_paths[name].relative_to(root)),
            "input_sha256": sha256_file(input_paths[name]),
        }

    # Scale only for Logistic Regression. Fit exclusively on training predictors.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(frames["train"][feature_columns])
    X_val_scaled = scaler.transform(frames["validation"][feature_columns])
    X_test_scaled = scaler.transform(frames["test"][feature_columns])

    for name, matrix in (
        ("train", X_train_scaled),
        ("validation", X_val_scaled),
        ("test", X_test_scaled),
    ):
        pd.DataFrame(matrix, columns=feature_columns).to_csv(
            scaled_dir / f"{name}_X_scaled.csv", index=False
        )
        # Write the corresponding labels in the same row order.
        frames[name][[EXPECTED_TARGET]].to_csv(
            scaled_dir / f"{name}_y.csv", index=False
        )

    scaler_path = artifact_dir / "logistic_regression_standard_scaler.joblib"
    joblib.dump(scaler, scaler_path)

    # Sanity check: train-scaled features should have means close to zero.
    max_abs_train_mean = float(np.abs(X_train_scaled.mean(axis=0)).max())
    if max_abs_train_mean > 1e-6:
        fail(
            "Scaled training feature means are unexpectedly far from zero: "
            f"{max_abs_train_mean}"
        )

    # Confirm original partitions were not altered: we only read them and write elsewhere.
    manifest = {
        "phase": 3,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Leakage-aware model preprocessing",
        "target_column": EXPECTED_TARGET,
        "chronological_order_only_column_excluded_from_predictors": ORDER_ONLY_COLUMN,
        "predictor_columns": feature_columns,
        "excluded_from_predictors": [EXPECTED_TARGET, ORDER_ONLY_COLUMN],
        "split_summary": split_summary,
        "preprocessing": {
            "tree_models": "Unscaled numeric predictors; no resampling in this phase.",
            "logistic_regression": (
                "StandardScaler fitted on training predictors only; the same fitted "
                "scaler transforms validation and test."
            ),
            "validation_and_test_used_to_fit_transformations": False,
            "smote_applied": False,
            "phase2_input_files_modified": False,
        },
        "artifacts": {
            "unscaled_matrices": str(unscaled_dir.relative_to(root)),
            "logistic_regression_scaled_matrices": str(scaled_dir.relative_to(root)),
            "scaler": str(scaler_path.relative_to(root)),
        },
        "software": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "checks": {
            "same_columns_and_order_across_splits": True,
            "numeric_finite_predictors": True,
            "chronological_order_checked_when_time_present": True,
            "scaled_training_max_absolute_feature_mean": max_abs_train_mean,
            "rows_and_class_counts": "Recorded in split_summary for audit.",
        },
    }

    manifest_path = output_dir / "preprocessing_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report_lines = [
        "PHASE 3 PREPROCESSING VALIDATION REPORT",
        f"Generated (UTC): {manifest['created_utc']}",
        "",
        "Predictor policy:",
        f"- Target: {EXPECTED_TARGET}",
        f"- Excluded order-only column: {ORDER_ONLY_COLUMN}",
        f"- Number of predictors: {len(feature_columns)}",
        f"- Predictor columns: {', '.join(feature_columns)}",
        "",
        "Split summary:",
    ]
    for name in SPLITS:
        s = split_summary[name]
        report_lines.append(
            f"- {name}: rows={s['rows']}, class_0={s['class_0']}, "
            f"class_1={s['class_1']}, fraud_prevalence={s['fraud_prevalence_percent']:.6f}%, "
            f"Time range={s['time_min']}..{s['time_max']}"
        )
    report_lines.extend([
        "",
        "Checks passed:",
        "- Inputs exist and are non-empty.",
        "- Target labels are binary 0/1.",
        "- Predictors are numeric and finite.",
        "- Columns and their order match across splits.",
        "- Time is excluded from primary predictors.",
        "- StandardScaler was fitted on training predictors only.",
        "- Validation and test were transformed, not fitted.",
        "- No SMOTE/resampling was performed.",
        "- Phase 2 source CSVs were not modified.",
        "",
        "Reminder: choose thresholds and tune models using validation data only; reserve test data for final evaluation.",
    ])
    report_path = output_dir / "preprocessing_validation.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("PHASE 3 PREPROCESSING COMPLETED")
    print(f"Project root: {root}")
    print(f"Predictors: {len(feature_columns)} (Time and Class excluded)")
    for name in SPLITS:
        s = split_summary[name]
        print(
            f"{name:10s} rows={s['rows']:6d} | fraud={s['class_1']:4d} | "
            f"legitimate={s['class_0']:6d} | prevalence={s['fraud_prevalence_percent']:.6f}%"
        )
    print(f"Unscaled matrices: {unscaled_dir}")
    print(f"Scaled LR matrices: {scaled_dir}")
    print(f"Scaler artifact: {scaler_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Validation report: {report_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: Phase 3 preprocessing failed: {exc}", file=sys.stderr)
        raise SystemExit(1)

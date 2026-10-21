#!/usr/bin/env python3
"""
Phase 2: data validation, duplicate audit, and deterministic chronological splits.

Run from the project root:
    python -u src/02_data_validation/validate_and_split.py

The raw CSV is read-only. Exact duplicates are audited, not automatically removed.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "creditcard_ulb.csv"
SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
REPORT_DIR = PROJECT_ROOT / "outputs" / "data_validation"

TARGET = "Class"
TIME_COL = "Time"
EXPECTED_FEATURES = [TIME_COL] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
EXPECTED_COLUMNS = EXPECTED_FEATURES + [TARGET]
TRAIN_FRAC = 0.60
VALIDATION_FRAC = 0.20
TEST_FRAC = 0.20


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Raw dataset:  {RAW_CSV}")

    if not RAW_CSV.is_file():
        print("\nERROR: Raw CSV not found at the expected path.")
        print("Check the filename and update RAW_CSV near the top of this script.")
        return 1

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Read without modifying the original file.
    df = pd.read_csv(RAW_CSV)
    original_rows, original_cols = df.shape
    input_hash = sha256_file(RAW_CSV)
    print(f"Loaded shape: {df.shape}")
    print(f"SHA-256:      {input_hash}")

    # Schema checks: report exact expected columns and any discrepancies.
    missing_columns = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    unexpected_columns = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    schema_rows = []
    for col in df.columns:
        schema_rows.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "missing_count": int(df[col].isna().sum()),
            "missing_pct": float(df[col].isna().mean() * 100),
            "non_finite_count": (
                int((~np.isfinite(pd.to_numeric(df[col], errors="coerce")) &
                     pd.to_numeric(df[col], errors="coerce").notna()).sum())
                if pd.api.types.is_numeric_dtype(df[col]) else np.nan
            ),
            "expected_column": col in EXPECTED_COLUMNS,
        })
    write_csv(pd.DataFrame(schema_rows), REPORT_DIR / "schema_missing_values.csv")

    if missing_columns:
        print(f"\nERROR: Missing required columns: {missing_columns}")
        print(f"Unexpected columns: {unexpected_columns}")
        return 1

    # The dataset is expected to contain numeric predictors and binary target labels.
    numeric_conversion_errors = []
    for col in EXPECTED_FEATURES + [TARGET]:
        converted = pd.to_numeric(df[col], errors="coerce")
        bad = int((converted.isna() & df[col].notna()).sum())
        if bad:
            numeric_conversion_errors.append({"column": col, "non_numeric_non_null_count": bad})
        df[col] = converted

    class_counts = df[TARGET].value_counts(dropna=False).sort_index()
    invalid_class_count = int((~df[TARGET].isin([0, 1])).sum())
    missing_target_count = int(df[TARGET].isna().sum())
    missing_feature_cells = int(df[EXPECTED_FEATURES].isna().sum().sum())
    nonfinite_feature_cells = int(
        (~np.isfinite(df[EXPECTED_FEATURES].to_numpy(dtype=float))).sum()
    )

    quality = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(RAW_CSV),
        "input_sha256": input_hash,
        "rows": int(original_rows),
        "columns": int(original_cols),
        "expected_columns_missing": json.dumps(missing_columns),
        "unexpected_columns": json.dumps(unexpected_columns),
        "missing_target_count": missing_target_count,
        "invalid_target_label_count": invalid_class_count,
        "missing_feature_cells": missing_feature_cells,
        "nonfinite_feature_cells": nonfinite_feature_cells,
        "numeric_conversion_errors": json.dumps(numeric_conversion_errors),
        "class_0_count": int((df[TARGET] == 0).sum()),
        "class_1_count": int((df[TARGET] == 1).sum()),
        "exact_duplicate_rows_beyond_first": int(df.duplicated(keep="first").sum()),
        "rows_in_exact_duplicate_groups_including_first": int(df.duplicated(keep=False).sum()),
        "duplicate_predictor_rows_beyond_first": int(df.duplicated(subset=EXPECTED_FEATURES, keep="first").sum()),
        "rows_in_duplicate_predictor_groups": int(df.duplicated(subset=EXPECTED_FEATURES, keep=False).sum()),
        "duplicates_removed": 0,
    }
    write_csv(pd.DataFrame([quality]), REPORT_DIR / "data_quality_summary.csv")

    # Exact duplicate diagnostics, including class labels.
    exact_duplicate_mask = df.duplicated(keep=False)
    exact_duplicate_rows = df.loc[exact_duplicate_mask].copy()
    if not exact_duplicate_rows.empty:
        exact_duplicate_rows = exact_duplicate_rows.sort_values(
            EXPECTED_COLUMNS, kind="mergesort"
        )
    write_csv(exact_duplicate_rows, REPORT_DIR / "exact_duplicate_rows.csv")

    # Group by predictor values to find identical predictors with conflicting labels.
    predictor_group = (
        df.groupby(EXPECTED_FEATURES, dropna=False, sort=False)[TARGET]
        .agg(row_count="size", distinct_class_count="nunique",
             class_0_count=lambda s: int((s == 0).sum()),
             class_1_count=lambda s: int((s == 1).sum()))
        .reset_index()
    )
    predictor_group["conflicting_class_labels"] = predictor_group["distinct_class_count"] > 1
    write_csv(
        predictor_group.loc[predictor_group["row_count"] > 1],
        REPORT_DIR / "duplicate_predictor_groups.csv",
    )
    write_csv(
        predictor_group.loc[predictor_group["conflicting_class_labels"]],
        REPORT_DIR / "conflicting_predictor_labels.csv",
    )

    duplicate_summary = pd.DataFrame([
        {"measure": "exact_duplicate_rows_beyond_first", "count": quality["exact_duplicate_rows_beyond_first"]},
        {"measure": "rows_in_exact_duplicate_groups_including_first", "count": quality["rows_in_exact_duplicate_groups_including_first"]},
        {"measure": "duplicate_predictor_rows_beyond_first", "count": quality["duplicate_predictor_rows_beyond_first"]},
        {"measure": "rows_in_duplicate_predictor_groups", "count": quality["rows_in_duplicate_predictor_groups"]},
        {"measure": "conflicting_predictor_groups", "count": int(predictor_group["conflicting_class_labels"].sum())},
        {"measure": "duplicates_removed", "count": 0},
    ])
    write_csv(duplicate_summary, REPORT_DIR / "duplicate_summary.csv")

    # Stop before splitting if core data assumptions fail; do not silently clean.
    if missing_target_count or invalid_class_count or missing_feature_cells or nonfinite_feature_cells or numeric_conversion_errors:
        print("\nERROR: Core data validation failed. Review reports before splitting.")
        print(f"See: {REPORT_DIR}")
        return 2
    if not set(class_counts.index).issubset({0, 1}):
        print("\nERROR: Target labels are not exactly 0 and 1.")
        return 2
    if df[TIME_COL].isna().any() or not np.isfinite(df[TIME_COL].to_numpy(dtype=float)).all():
        print("\nERROR: Time contains missing or non-finite values.")
        return 2

    # Stable chronological ordering. Original input row position is the deterministic
    # tie-breaker for equal Time values; source row index is recorded only in the manifest.
    df = df.copy()
    df["_source_row_position"] = np.arange(len(df), dtype=np.int64)
    df = df.sort_values([TIME_COL, "_source_row_position"], kind="mergesort").reset_index(drop=True)

    n = len(df)
    train_end = int(n * TRAIN_FRAC)
    validation_end = train_end + int(n * VALIDATION_FRAC)
    # Remaining rows go to test so partitions sum exactly to n.
    partitions = {
        "train": df.iloc[:train_end].copy(),
        "validation": df.iloc[train_end:validation_end].copy(),
        "test": df.iloc[validation_end:].copy(),
    }

    # Boundary timestamps may be equal across adjacent partitions if ties straddle a cut.
    manifest_rows = []
    for name, part in partitions.items():
        manifest_rows.append({
            "partition": name,
            "rows": int(len(part)),
            "class_0_count": int((part[TARGET] == 0).sum()),
            "class_1_count": int((part[TARGET] == 1).sum()),
            "fraud_prevalence_pct": float(part[TARGET].mean() * 100),
            "time_min": float(part[TIME_COL].min()),
            "time_max": float(part[TIME_COL].max()),
            "first_source_row_position": int(part["_source_row_position"].iloc[0]),
            "last_source_row_position": int(part["_source_row_position"].iloc[-1]),
        })
        out = part[EXPECTED_COLUMNS].copy()
        write_csv(out, SPLIT_DIR / f"{name}.csv")

    manifest = pd.DataFrame(manifest_rows)
    manifest["split_method"] = "stable chronological sort by Time; original row position breaks Time ties"
    manifest["fractions_requested"] = "60% train / 20% validation / remainder test"
    manifest["input_sha256"] = input_hash
    write_csv(manifest, REPORT_DIR / "chronological_split_manifest.csv")

    # Cross-partition duplicate audit using full feature+target rows and predictor-only rows.
    full_row_sets = {
        name: set(map(tuple, part[EXPECTED_COLUMNS].itertuples(index=False, name=None)))
        for name, part in partitions.items()
    }
    predictor_sets = {
        name: set(map(tuple, part[EXPECTED_FEATURES].itertuples(index=False, name=None)))
        for name, part in partitions.items()
    }
    cross_rows = []
    names = list(partitions)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            cross_rows.append({
                "partition_a": a,
                "partition_b": b,
                "shared_exact_feature_and_target_rows": len(full_row_sets[a] & full_row_sets[b]),
                "shared_predictor_vectors_any_label": len(predictor_sets[a] & predictor_sets[b]),
            })
    write_csv(pd.DataFrame(cross_rows), REPORT_DIR / "cross_split_duplicate_audit.csv")

    # Verify split integrity.
    partition_row_total = sum(len(p) for p in partitions.values())
    partition_class0_total = sum(int((p[TARGET] == 0).sum()) for p in partitions.values())
    partition_class1_total = sum(int((p[TARGET] == 1).sum()) for p in partitions.values())
    time_order_ok = all(
        part[TIME_COL].is_monotonic_increasing for part in partitions.values()
    )
    boundaries_ok = (
        partitions["train"][TIME_COL].max() <= partitions["validation"][TIME_COL].min()
        and partitions["validation"][TIME_COL].max() <= partitions["test"][TIME_COL].min()
    )
    audit = {
        "input_rows": int(n),
        "partition_rows_total": int(partition_row_total),
        "row_counts_reconcile": bool(partition_row_total == n),
        "input_class_0_count": int((df[TARGET] == 0).sum()),
        "partition_class_0_total": int(partition_class0_total),
        "class_0_counts_reconcile": bool(partition_class0_total == int((df[TARGET] == 0).sum())),
        "input_class_1_count": int((df[TARGET] == 1).sum()),
        "partition_class_1_total": int(partition_class1_total),
        "class_1_counts_reconcile": bool(partition_class1_total == int((df[TARGET] == 1).sum())),
        "each_partition_time_monotonic": bool(time_order_ok),
        "chronological_boundaries_non_decreasing": bool(boundaries_ok),
        "equal_Time_boundary_overlap_possible": True,
        "raw_csv_sha256": input_hash,
        "duplicates_removed": 0,
    }
    (REPORT_DIR / "split_integrity_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    readme = f"""PHASE 2 DATA VALIDATION REPORT
Generated (UTC): {datetime.now(timezone.utc).isoformat()}

Input file: {RAW_CSV}
Input SHA-256: {input_hash}
Input dimensions: {original_rows} rows x {original_cols} columns

Validation:
- Missing target values: {missing_target_count}
- Invalid target labels: {invalid_class_count}
- Missing predictor cells: {missing_feature_cells}
- Non-finite predictor cells: {nonfinite_feature_cells}
- Exact duplicate rows beyond first occurrence: {quality['exact_duplicate_rows_beyond_first']}
- Rows in exact duplicate groups including first occurrence: {quality['rows_in_exact_duplicate_groups_including_first']}
- Duplicate predictor rows beyond first occurrence: {quality['duplicate_predictor_rows_beyond_first']}
- Conflicting predictor groups: {int(predictor_group['conflicting_class_labels'].sum())}
- Duplicate records removed: 0 (audit only)

Split method:
- Stable chronological sort by Time.
- Original CSV row position is the deterministic tie-breaker for equal Time values.
- 60% train, 20% validation, remaining rows test.
- Time is used for ordering, not included as a predictor decision here; split CSVs retain Time for traceability.
- If equal Time values straddle a partition boundary, the same Time value can appear at both adjacent boundaries. This is documented rather than silently changing the split.

Important:
- Do not use the test partition to tune models, preprocessing, resampling, or thresholds.
- Fit transformations on training data only.
- Apply SMOTE only to training data or within training folds.
- Inspect all reports before proceeding to model training.
"""
    (REPORT_DIR / "PHASE_2_README.txt").write_text(readme, encoding="utf-8")

    print("\nPhase 2 run completed.")
    print("\nChronological split manifest:")
    print(manifest.to_string(index=False))
    print("\nIntegrity audit:")
    print(json.dumps(audit, indent=2))
    print(f"\nSplit files: {SPLIT_DIR}")
    print(f"Reports:     {REPORT_DIR}")

    if not all([audit["row_counts_reconcile"], audit["class_0_counts_reconcile"],
                audit["class_1_counts_reconcile"], audit["each_partition_time_monotonic"],
                audit["chronological_boundaries_non_decreasing"]]):
        print("\nWARNING: One or more split integrity checks failed.")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

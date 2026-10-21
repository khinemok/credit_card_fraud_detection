"""
Complete EDA script for the ULB/MLG credit-card fraud dataset.

Expected layout:
credit_card_fraud_detection_ulb/
├── data/
│   └── raw/
│       └── creditcard.csv
├── outputs/
└── src/
    └── 01_eda/
        └── eda_credit_card_fraud.py

Run from the project root:
    python -u src/01_eda/eda_credit_card_fraud.py

No command-line arguments are needed. Dataset and output paths are resolved
relative to this file, not the terminal's current directory.
"""

from pathlib import Path
import sys
import traceback
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Allows plots to be saved in terminal/headless environments.
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

# This script is expected to be in <project_root>/src/01_eda/.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard_ulb.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

TARGET_COL = "Class"
TIME_COL = "Time"
AMOUNT_COL = "Amount"


def log(message=""):
    """Print progress immediately, even when Python output is buffered."""
    print(message, flush=True)


def create_directories():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def resolve_dataset_path():
    """Find the dataset at the expected path, with helpful fallback checks."""
    if DATA_PATH.exists():
        return DATA_PATH

    # Common alternative locations within the project.
    candidates = [
        PROJECT_ROOT / "creditcard.csv",
        PROJECT_ROOT / "data" / "creditcard.csv",
        PROJECT_ROOT / "data" / "raw" / "creditcard.csv",
        PROJECT_ROOT / "data" / "raw" / "CreditCard.csv",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            log(f"Dataset found at an alternative location: {candidate}")
            log(f"Recommended location is: {DATA_PATH}")
            return candidate

    # Do not scan the entire computer; report likely locations to check.
    raise FileNotFoundError(
        "Could not find creditcard.csv.\n"
        f"Expected location: {DATA_PATH}\n"
        f"Project root detected as: {PROJECT_ROOT}\n\n"
        "Put the dataset in the data/raw folder. From the project root, you can run:\n"
        "  mkdir -p data/raw\n"
        "  cp /actual/path/to/creditcard.csv data/raw/creditcard.csv\n"
        "Replace /actual/path/to/creditcard.csv with the real file location."
    )


def load_data():
    log("=" * 72)
    log("CREDIT CARD FRAUD DETECTION — EXPLORATORY DATA ANALYSIS")
    log("=" * 72)
    log(f"Script path : {Path(__file__).resolve()}")
    log(f"Project root: {PROJECT_ROOT}")
    dataset_path = resolve_dataset_path()
    log(f"Dataset path: {dataset_path}")
    df = pd.read_csv(dataset_path)
    log(f"Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df, dataset_path


def validate_schema(df):
    log("\n[1/8] DATA STRUCTURE AND DATA QUALITY")
    log("-" * 72)
    required = {TARGET_COL, TIME_COL, AMOUNT_COL}
    missing_required = required.difference(df.columns)
    if missing_required:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_required)}")

    df.info()
    log("\nFirst five rows:")
    log(df.head().to_string(index=False))

    schema = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "missing_count": [int(df[c].isna().sum()) for c in df.columns],
        "missing_percent": [float(df[c].isna().mean() * 100) for c in df.columns],
        "unique_values": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    schema.to_csv(TABLES_DIR / "schema_missing_values.csv", index=False)

    summary = pd.DataFrame([{
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
        "exact_duplicate_rows_beyond_first": int(df.duplicated().sum()),
        "fraud_count": int((df[TARGET_COL] == 1).sum()),
        "legitimate_count": int((df[TARGET_COL] == 0).sum()),
        "fraud_prevalence_percent": float((df[TARGET_COL] == 1).mean() * 100),
    }])
    summary.to_csv(TABLES_DIR / "dataset_summary.csv", index=False)
    log(f"Missing cells: {df.isna().sum().sum():,}")
    log(f"Exact duplicate rows beyond first occurrence: {df.duplicated().sum():,}")


def analyze_class_distribution(df):
    log("\n[2/8] CLASS DISTRIBUTION")
    log("-" * 72)
    counts = df[TARGET_COL].value_counts(dropna=False).sort_index()
    percentages = df[TARGET_COL].value_counts(normalize=True, dropna=False).sort_index() * 100
    labels = [
        "Legitimate" if value == 0 else "Fraud" if value == 1 else str(value)
        for value in counts.index
    ]
    distribution = pd.DataFrame({
        "Class": counts.index,
        "count": counts.values,
        "percentage": percentages.values,
        "label": labels,
    })
    distribution.to_csv(TABLES_DIR / "class_distribution.csv", index=False)
    log(distribution.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, counts.values)
    ax.set_title("Transaction Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of transactions")
    ax.ticklabel_format(axis="y", style="plain")
    for bar, count in zip(bars, counts.values):
        ax.annotate(
            f"{int(count):,}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4), textcoords="offset points", ha="center", va="bottom"
        )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_distribution.png", dpi=160)
    plt.close(fig)


def analyze_duplicates(df):
    log("\n[3/8] DUPLICATE INVESTIGATION")
    log("-" * 72)
    duplicate_mask = df.duplicated(keep=False)
    duplicate_rows = df.loc[duplicate_mask]
    duplicate_count = int(df.duplicated().sum())

    pd.DataFrame([{
        "duplicate_rows_beyond_first": duplicate_count,
        "rows_in_duplicate_groups_including_first": int(duplicate_mask.sum()),
        "note": "Duplicates are reported, not automatically removed.",
    }]).to_csv(TABLES_DIR / "duplicate_summary.csv", index=False)

    if duplicate_rows.empty:
        by_class = pd.DataFrame(columns=[TARGET_COL, "rows_in_duplicate_groups"])
    else:
        by_class = (
            duplicate_rows.groupby(TARGET_COL).size()
            .rename("rows_in_duplicate_groups").reset_index()
        )
    by_class.to_csv(TABLES_DIR / "duplicate_class_counts.csv", index=False)

    log(f"Duplicate rows beyond first occurrence: {duplicate_count:,}")
    log("Rows belonging to duplicate groups, grouped by Class:")
    log(by_class.to_string(index=False) if not by_class.empty else "None found")
    log("Duplicates have NOT been removed automatically.")


def analyze_numeric_features(df):
    log("\n[4/8] NUMERIC FEATURES BY CLASS")
    log("-" * 72)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    df.groupby(TARGET_COL)[numeric_cols].describe().T.to_csv(
        TABLES_DIR / "numeric_summary_by_class.csv"
    )

    medians = df.groupby(TARGET_COL)[numeric_cols].median().T
    medians.columns = [f"Class_{col}_median" for col in medians.columns]
    medians.to_csv(TABLES_DIR / "feature_medians_by_class.csv")
    log("Saved class-wise descriptive statistics and medians.")

    feature_cols = [
        c for c in ["V1", "V2", "V3", "V4", "V5", "V10", "V12", "V14", "V17", AMOUNT_COL]
        if c in df.columns
    ]
    if feature_cols:
        ncols = 2
        nrows = int(np.ceil(len(feature_cols) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.5 * nrows))
        axes = np.atleast_1d(axes).ravel()
        for ax, col in zip(axes, feature_cols):
            values = [
                df.loc[df[TARGET_COL] == 0, col].dropna().to_numpy(),
                df.loc[df[TARGET_COL] == 1, col].dropna().to_numpy(),
            ]
            # Compatible with current Matplotlib: use tick_labels, not labels.
            ax.boxplot(
                values,
                tick_labels=["Legitimate", "Fraud"],
                showfliers=False,
            )
            ax.set_title(f"{col} by class")
            ax.set_ylabel(col)
        for ax in axes[len(feature_cols):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "feature_boxplots_by_class.png", dpi=160)
        plt.close(fig)


def analyze_amount(df):
    log("\n[5/8] TRANSACTION AMOUNTS")
    log("-" * 72)
    amount_summary = df.groupby(TARGET_COL)[AMOUNT_COL].describe()
    amount_summary.to_csv(TABLES_DIR / "amount_summary_by_class.csv")
    log(amount_summary.to_string())

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, group in df.groupby(TARGET_COL):
        ax.hist(
            np.log1p(group[AMOUNT_COL].clip(lower=0)),
            bins=60, alpha=0.55,
            label="Fraud" if label == 1 else "Legitimate",
        )
    ax.set_title("Transaction Amount Distribution (log1p scale)")
    ax.set_xlabel("log(1 + Amount)")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "amount_distribution_log1p.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    values = [
        df.loc[df[TARGET_COL] == 0, AMOUNT_COL].dropna().to_numpy(),
        df.loc[df[TARGET_COL] == 1, AMOUNT_COL].dropna().to_numpy(),
    ]
    ax.boxplot(values, tick_labels=["Legitimate", "Fraud"], showfliers=False)
    ax.set_title("Transaction Amount by Class (outliers hidden)")
    ax.set_ylabel("Amount")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "amount_boxplot.png", dpi=160)
    plt.close(fig)


def analyze_time(df):
    log("\n[6/8] TRANSACTION TIME")
    log("-" * 72)
    time_min, time_max = df[TIME_COL].min(), df[TIME_COL].max()
    log(f"Time min: {time_min}")
    log(f"Time max: {time_max}")
    log(f"Approximate span in hours: {(time_max - time_min) / 3600:.2f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df[TIME_COL].dropna(), bins=80)
    ax.set_title("Distribution of Transaction Time")
    ax.set_xlabel("Time (seconds since first recorded transaction)")
    ax.set_ylabel("Number of transactions")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "time_distribution.png", dpi=160)
    plt.close(fig)

    time_bins = pd.cut(df[TIME_COL], bins=20, duplicates="drop")
    grouped = df.groupby(time_bins, observed=False)[TARGET_COL].agg(
        transactions="size", fraud_count="sum", fraud_rate="mean"
    ).reset_index()
    grouped["time_bin"] = grouped[TIME_COL].astype(str)
    grouped = grouped.drop(columns=[TIME_COL])
    grouped.to_csv(TABLES_DIR / "fraud_by_time_bin.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(len(grouped)), grouped["fraud_rate"] * 100, marker="o")
    ax.set_title("Fraud Rate Across Chronological Time Bins")
    ax.set_xlabel("Chronological time bin (earlier to later)")
    ax.set_ylabel("Fraud rate (%)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fraud_rate_over_time.png", dpi=160)
    plt.close(fig)


def analyze_correlations(df):
    log("\n[7/8] CORRELATION ANALYSIS")
    log("-" * 72)
    numeric = df.select_dtypes(include=np.number)
    corr = numeric.corr()
    corr.to_csv(TABLES_DIR / "correlation_matrix.csv")

    if TARGET_COL in corr.columns:
        strongest = (
            corr[TARGET_COL].drop(TARGET_COL).abs()
            .sort_values(ascending=False).head(10)
        )
        log("Features with strongest absolute linear correlation with Class:")
        log(strongest.to_string())

    fig, ax = plt.subplots(figsize=(13, 10))
    image = ax.imshow(corr.to_numpy(), aspect="auto", interpolation="nearest")
    ax.set_title("Correlation Heatmap")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=7)
    fig.colorbar(image, ax=ax, shrink=0.8, label="Correlation")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=160)
    plt.close(fig)


def analyze_chronological_split(df):
    log("\n[8/8] PROPOSED CHRONOLOGICAL 60/20/20 SPLIT")
    log("-" * 72)
    sorted_df = df.sort_values(TIME_COL, kind="mergesort").reset_index(drop=False)
    n = len(sorted_df)
    train_end = int(n * 0.60)
    validation_end = int(n * 0.80)
    partitions = [
        ("train", sorted_df.iloc[:train_end]),
        ("validation", sorted_df.iloc[train_end:validation_end]),
        ("test", sorted_df.iloc[validation_end:]),
    ]

    records = []
    for name, part in partitions:
        records.append({
            "partition": name,
            "rows": int(len(part)),
            "legitimate_count": int((part[TARGET_COL] == 0).sum()),
            "fraud_count": int((part[TARGET_COL] == 1).sum()),
            "fraud_prevalence_percent": float(part[TARGET_COL].mean() * 100),
            "time_min": float(part[TIME_COL].min()),
            "time_max": float(part[TIME_COL].max()),
        })
    split_summary = pd.DataFrame(records)
    split_summary.to_csv(TABLES_DIR / "chronological_split_summary.csv", index=False)
    log(split_summary.to_string(index=False))
    log(
        "Tie note: partitions are cut by row position after a stable sort. "
        "Equal Time values may occur on both sides of a boundary; stable original "
        "row order makes assignment deterministic."
    )


def write_readme(df, dataset_path):
    fraud_count = int((df[TARGET_COL] == 1).sum())
    legitimate_count = int((df[TARGET_COL] == 0).sum())
    fraud_rate = float((df[TARGET_COL] == 1).mean() * 100)
    duplicates = int(df.duplicated().sum())
    missing = int(df.isna().sum().sum())
    time_span_hours = float((df[TIME_COL].max() - df[TIME_COL].min()) / 3600)

    content = f"""EDA SUMMARY — CREDIT CARD FRAUD DATASET

Dataset path: {dataset_path}
Rows: {len(df):,}
Columns: {len(df.columns)}
Legitimate transactions (Class 0): {legitimate_count:,}
Fraudulent transactions (Class 1): {fraud_count:,}
Fraud prevalence: {fraud_rate:.6f}%
Missing cells: {missing:,}
Exact duplicate rows beyond first occurrence: {duplicates:,}
Time span: approximately {time_span_hours:.2f} hours

Interpretation cautions:
- Accuracy alone is inadequate for this severely imbalanced dataset.
- Duplicate rows are reported, not automatically removed.
- V1-V28 are anonymized PCA-derived features; do not assign unsupported meanings.
- Time is used to describe and order transactions in the proposed chronological split.
- Final split and duplicate-handling policies must be documented.
- The dataset spans about two days, so it cannot establish long-term concept drift.
- Test labels must not be used for model selection or threshold tuning.

Tables: {TABLES_DIR}
Figures: {FIGURES_DIR}
"""
    (OUTPUT_DIR / "EDA_README.txt").write_text(content, encoding="utf-8")


def main():
    create_directories()
    df, dataset_path = load_data()
    validate_schema(df)
    analyze_class_distribution(df)
    analyze_duplicates(df)
    analyze_numeric_features(df)
    analyze_amount(df)
    analyze_time(df)
    analyze_correlations(df)
    analyze_chronological_split(df)
    write_readme(df, dataset_path)

    log("\n" + "=" * 72)
    log("EDA COMPLETE")
    log(f"Tables saved to : {TABLES_DIR}")
    log(f"Figures saved to: {FIGURES_DIR}")
    log(f"Summary saved to: {OUTPUT_DIR / 'EDA_README.txt'}")
    log("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log("\nEDA FAILED")
        log(f"{type(exc).__name__}: {exc}")
        log("Full traceback:")
        traceback.print_exc()
        sys.exit(1)

#!/usr/bin/env python3
"""
RQ5 independent-fit feature-importance stability experiment.

Purpose:
- Train the same Random Forest independently on several different subsamples
  drawn ONLY from the existing training partition.
- Exclude Class and Time from predictors.
- Save each fit's feature importances, ranking table, and pairwise Spearman
  rank correlations.
- Never loads validation or test data.

Run from the project root:
    python3 src/04_model_training/rq5_independent_fit_stability.py

Before running:
- Confirm the training CSV path in TRAIN_CSV_CANDIDATES below.
- This script assumes the training CSV contains the target column named Class.
"""

from pathlib import Path
import json
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier

# Edit this list only if your saved training partition has a different path.
TRAIN_CSV_CANDIDATES = [
    Path("data/processed/splits/train.csv"),
]

TARGET = "Class"
EXCLUDE_FROM_FEATURES = {"Class", "Time"}
N_FITS = 10
SAMPLE_SIZE = 25_000
SEEDS = [11, 23, 37, 53, 71, 89, 101, 131, 151, 181]
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "class_weight": "balanced_subsample",
    "n_jobs": -1,
}
OUT_DIR = Path("outputs/rq5_independent_fit_stability")


def find_training_csv():
    for candidate in TRAIN_CSV_CANDIDATES:
        if candidate.exists():
            return candidate
    print("ERROR: Could not find the training CSV at any expected path:")
    for candidate in TRAIN_CSV_CANDIDATES:
        print(f"  - {candidate}")
    print("\nPlease update TRAIN_CSV_CANDIDATES near the top of this script "
          "to the exact path of your existing training partition.")
    sys.exit(2)


def main():
    train_path = find_training_csv()
    df = pd.read_csv(train_path)

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' was not found in {train_path}. "
            f"Columns found: {list(df.columns)}"
        )

    feature_cols = [
        c for c in df.columns
        if c not in EXCLUDE_FROM_FEATURES
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not feature_cols:
        raise ValueError("No numeric predictor columns found after exclusions.")

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    y = df[TARGET]

    if y.isna().any():
        raise ValueError("The training target contains missing values.")
    if y.nunique() != 2:
        raise ValueError(f"Expected binary target; found {y.nunique()} classes.")

    # Handle missing predictor values using medians calculated on training data only.
    medians = X.median()
    X = X.fillna(medians)

    sample_n = min(SAMPLE_SIZE, len(df))
    if sample_n < 100:
        raise ValueError(f"Training partition appears too small: {len(df)} rows.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    importance_rows = []
    ranking_rows = []
    fit_metadata = []

    for fit_no, seed in enumerate(SEEDS[:N_FITS], start=1):
        rng = np.random.default_rng(seed)
        # Independent subsampling without replacement from training partition only.
        selected = rng.choice(len(df), size=sample_n, replace=False)
        X_fit = X.iloc[selected]
        y_fit = y.iloc[selected]

        if y_fit.nunique() != 2:
            raise ValueError(
                f"Fit {fit_no} sample did not contain both classes. "
                "Increase SAMPLE_SIZE or use a stratified sampling approach."
            )

        model = RandomForestClassifier(random_state=seed, **RF_PARAMS)
        model.fit(X_fit, y_fit)

        ordered = sorted(
            zip(feature_cols, model.feature_importances_),
            key=lambda item: item[1],
            reverse=True,
        )
        ranks = {feature: rank for rank, (feature, _) in enumerate(ordered, start=1)}

        for feature, importance in ordered:
            importance_rows.append({
                "fit": fit_no,
                "seed": seed,
                "feature": feature,
                "importance": float(importance),
                "rank": int(ranks[feature]),
            })
            ranking_rows.append({
                "fit": fit_no,
                "seed": seed,
                "feature": feature,
                "rank": int(ranks[feature]),
            })

        fit_metadata.append({
            "fit": fit_no,
            "seed": seed,
            "sample_rows": int(sample_n),
            "sample_fraud_count": int(y_fit.sum()),
            "sample_legitimate_count": int((y_fit == 0).sum()),
            "features": len(feature_cols),
        })
        print(f"Completed fit {fit_no}/{N_FITS}: seed={seed}, "
              f"rows={sample_n}, fraud={int(y_fit.sum())}")

    imp_df = pd.DataFrame(importance_rows)
    rank_df = pd.DataFrame(ranking_rows)
    imp_df.to_csv(OUT_DIR / "importance_by_fit.csv", index=False)
    rank_df.to_csv(OUT_DIR / "rankings_by_fit.csv", index=False)

    rank_wide = rank_df.pivot(index="feature", columns="fit", values="rank")
    rank_wide.columns = [f"fit_{c}_rank" for c in rank_wide.columns]
    rank_wide["mean_rank"] = rank_wide.mean(axis=1)
    rank_wide["rank_sd"] = rank_wide.std(axis=1, ddof=1)
    rank_wide = rank_wide.sort_values(["mean_rank", "rank_sd"])
    rank_wide.to_csv(OUT_DIR / "rank_stability_summary.csv")

    # Pairwise Spearman correlation across complete feature rankings.
    corr_rows = []
    fit_ids = sorted(rank_df["fit"].unique())
    for i, fit_a in enumerate(fit_ids):
        a = rank_df[rank_df["fit"] == fit_a].set_index("feature")["rank"]
        for fit_b in fit_ids[i + 1:]:
            b = rank_df[rank_df["fit"] == fit_b].set_index("feature")["rank"]
            common = a.index.intersection(b.index)
            rho = spearmanr(a.loc[common], b.loc[common]).statistic
            corr_rows.append({
                "fit_a": int(fit_a),
                "fit_b": int(fit_b),
                "spearman_rank_correlation": float(rho),
            })

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(OUT_DIR / "pairwise_spearman_correlations.csv", index=False)

    top_features = (
        rank_wide.reset_index()
        .sort_values(["mean_rank", "rank_sd"])
        .head(10)
    )
    top_features.to_csv(OUT_DIR / "top10_mean_rank_features.csv", index=False)

    metadata = {
        "experiment": "RQ5 independent-fit feature-importance stability",
        "training_csv": str(train_path),
        "rows_in_training_partition": int(len(df)),
        "sample_rows_per_fit": int(sample_n),
        "number_of_fits": int(N_FITS),
        "seeds": SEEDS[:N_FITS],
        "target": TARGET,
        "excluded_predictor_columns": sorted(EXCLUDE_FROM_FEATURES),
        "feature_columns": feature_cols,
        "random_forest_parameters": RF_PARAMS,
        "sampling": "independent subsamples without replacement from training partition",
        "validation_loaded": False,
        "test_loaded": False,
        "mean_pairwise_spearman": (
            float(corr_df["spearman_rank_correlation"].mean())
            if len(corr_df) else None
        ),
        "min_pairwise_spearman": (
            float(corr_df["spearman_rank_correlation"].min())
            if len(corr_df) else None
        ),
        "max_pairwise_spearman": (
            float(corr_df["spearman_rank_correlation"].max())
            if len(corr_df) else None
        ),
        "note": "Subsamples overlap in source rows but are independently sampled; "
                "this measures sensitivity to training-sample composition, not full population stability."
    }
    with open(OUT_DIR / "experiment_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nRQ5 experiment finished.")
    print(f"Training partition used: {train_path}")
    print(f"Output folder: {OUT_DIR}")
    print(f"Mean pairwise Spearman rank correlation: "
          f"{metadata['mean_pairwise_spearman']:.4f}")
    print("Saved: rank_stability_summary.csv, pairwise_spearman_correlations.csv, "
          "top10_mean_rank_features.csv, importance_by_fit.csv, rankings_by_fit.csv, "
          "experiment_metadata.json")


if __name__ == "__main__":
    main()

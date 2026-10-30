#!/usr/bin/env python3
"""Phase 4.3: Train fraud-detection models using SMOTE on training data only.

Run from the project root:
    python src/04_model_training/train_smote_models_traceable.py

Important methodological choices:
- Reads only train and validation partitions; never loads test data.
- Uses features standardized by the Phase 3 scaler fitted on training data.
- Applies SMOTE to the training partition only. Validation remains imbalanced and untouched.
- Uses class_weight=None so this is a SMOTE-only experiment, not SMOTE + class weighting.
- Uses the same threshold (0.5) and metrics as prior experiments for initial comparison.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # headless backend: save to file, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from imblearn.over_sampling import SMOTE
from matplotlib.ticker import FuncFormatter
from sklearn.utils import check_random_state
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
THRESHOLD = 0.5
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREPROCESSING_ROOT = PROJECT_ROOT / "outputs" / "preprocessing"
SCALED_ROOT = PREPROCESSING_ROOT / "logistic_regression_scaled"
UNSCALED_ROOT = PREPROCESSING_ROOT / "unscaled"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "smote"
MODEL_ROOT = PROJECT_ROOT / "artifacts" / "smote"
PREDICTION_ROOT = OUTPUT_ROOT / "validation_predictions"
DEBUG_ROOT = OUTPUT_ROOT / "debug"
FIGURE_ROOT = OUTPUT_ROOT / "figures"
DEBUG_MODE = True  # Set to False to reduce diagnostic console output.
SYNTHETIC_SAMPLE_SIZE = 20


class TraceableSMOTE(SMOTE):
    """SMOTE variant that records the original minority rows used for each sample."""

    def __init__(self, *, original_minority_positions, **kwargs):
        super().__init__(**kwargs)
        self.original_minority_positions = np.asarray(original_minority_positions, dtype=int)
        self.synthetic_source_trace = []

    def _make_samples(
        self, X, y_dtype, y_type, nn_data, nn_num, n_samples,
        step_size=1.0, y=None,
    ):
        # SMOTE draws a flattened neighbor index, then an interpolation value.
        # Mirror those deterministic draws (random_state is an integer) before
        # calling the parent implementation, so each generated row gets its pair.
        rng = check_random_state(self.random_state)
        sample_indices = rng.randint(low=0, high=nn_num.size, size=n_samples)
        steps = step_size * rng.uniform(size=n_samples)
        rows = np.floor_divide(sample_indices, nn_num.shape[1])
        cols = np.mod(sample_indices, nn_num.shape[1])
        neighbor_local_indices = nn_num[rows, cols]

        for i in range(n_samples):
            base_local = int(rows[i])
            neighbor_local = int(neighbor_local_indices[i])
            self.synthetic_source_trace.append({
                "source_row_A_train_index": int(self.original_minority_positions[base_local]),
                "source_row_B_train_index": int(self.original_minority_positions[neighbor_local]),
                "source_A_minority_class_index": base_local,
                "source_B_minority_class_index": neighbor_local,
                "interpolation_lambda": float(steps[i]),
            })

        return super()._make_samples(
            X, y_dtype, y_type, nn_data, nn_num, n_samples,
            step_size=step_size, y=y,
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_xy(x_path: Path, y_path: Path, label: str):
    if not x_path.exists():
        raise FileNotFoundError(f"Missing {label} feature file: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Missing {label} target file: {y_path}")
    X = pd.read_csv(x_path)
    y_frame = pd.read_csv(y_path)
    if "Class" in y_frame.columns:
        y = y_frame["Class"]
    elif y_frame.shape[1] == 1:
        y = y_frame.iloc[:, 0]
    else:
        raise ValueError(f"{label}: target file must contain 'Class' or exactly one column")
    y = pd.Series(y.to_numpy(), name="Class").astype(int)
    if len(X) != len(y):
        raise ValueError(f"{label}: X/y row mismatch ({len(X)} vs {len(y)})")
    if X.empty or X.isna().any().any() or y.isna().any():
        raise ValueError(f"{label}: empty data or missing values found")
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError(f"{label}: target must contain only 0 and 1")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError(f"{label}: non-finite features found")
    if "Time" in X.columns:
        raise ValueError("Time is present in predictors; it should be excluded in the primary experiment")
    return X, y


def calculate_metrics(y_true, probabilities):
    predictions = (probabilities >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "threshold": THRESHOLD,
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "validation_rows": int(len(y_true)),
        "validation_fraud_count": int(np.sum(y_true == 1)),
        "validation_legitimate_count": int(np.sum(y_true == 0)),
        "validation_fraud_prevalence": float(np.mean(y_true == 1)),
    }


def plot_training_class_counts(counts_before, counts_after, output_path: Path) -> None:
    """Save a before-vs-after SMOTE class-count chart for the TRAINING set only."""
    classes = ["Legitimate\n(Class 0)", "Fraud\n(Class 1)"]
    colors = ["#3A76AF", "#D9534F"]
    panels = [
        ("Training set: BEFORE SMOTE", [counts_before["0"], counts_before["1"]]),
        ("Training set: AFTER SMOTE", [counts_after["0"], counts_after["1"]]),
    ]
    synthetic_added = counts_after["1"] - counts_before["1"]
    y_max = max(max(v) for _, v in panels) * 1.18

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True, sharey=True)
    for ax, (title, vals) in zip(axes, panels):
        total = sum(vals)
        bars = ax.bar(classes, vals, width=0.55, color=colors)
        ax.set_title(f"{title}\n{total:,} rows", fontsize=13, fontweight="bold")
        ax.set_ylim(0, y_max)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for bar, v in zip(bars, vals):
            ax.annotate(
                f"{v:,}\n({v / total:.2%})",
                (bar.get_x() + bar.get_width() / 2, v),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=11, fontweight="bold",
            )
    axes[0].set_ylabel("Number of rows")

    fig.suptitle("Credit Card Fraud Detection: Training Class Counts Before vs After SMOTE",
                 fontsize=15, fontweight="bold")
    fig.supxlabel(
        f"SMOTE added {synthetic_added:,} synthetic fraud rows to the training set only. "
        "Validation data is not resampled.",
        fontsize=10,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    for folder in (OUTPUT_ROOT, MODEL_ROOT, PREDICTION_ROOT, DEBUG_ROOT, FIGURE_ROOT):
        folder.mkdir(parents=True, exist_ok=True)

    # Use train-fitted standardized predictors for SMOTE's distance calculation.
    # Targets are loaded from the unscaled files; labels must never be scaled.
    X_train, y_train = read_xy(
        SCALED_ROOT / "train_X_scaled.csv", UNSCALED_ROOT / "train_y.csv", "scaled train")
    X_val, y_val = read_xy(
        SCALED_ROOT / "validation_X_scaled.csv", UNSCALED_ROOT / "validation_y.csv", "scaled validation")

    if list(X_train.columns) != list(X_val.columns):
        raise ValueError("Training and validation feature columns/order differ")
    if y_train.nunique() != 2 or y_val.nunique() != 2:
        raise ValueError("Both train and validation must contain both target classes")
    if int((y_train == 1).sum()) <= 5:
        raise ValueError("SMOTE with k_neighbors=5 requires more than five fraud examples in training")

    # ---- TRACE 1: inspect input data before SMOTE ----
    counts_before = {
        str(int(k)): int(v)
        for k, v in y_train.value_counts().sort_index().items()
    }
    print("\n=== TRACE: BEFORE SMOTE ===")
    print(f"Training features shape: {X_train.shape}")
    print(f"Training labels shape:   {y_train.shape}")
    print(f"Feature count:            {X_train.shape[1]}")
    print(f"Class counts before:      {counts_before}")
    print(f"Missing feature values:   {int(X_train.isna().sum().sum())}")
    print(f"Validation rows:          {len(X_val)}")
    print(f"Validation class counts:  {y_val.value_counts().sort_index().to_dict()}")

    # Guard against unexpected input problems before SMOTE runs.
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("Training feature/label row counts do not match.")
    if not X_train.index.equals(y_train.index):
        # read_xy normally creates a default aligned index; make the alignment explicit.
        y_train = y_train.reset_index(drop=True)
        X_train = X_train.reset_index(drop=True)
    if not X_train.columns.is_unique:
        raise ValueError("Training data contains duplicate feature column names.")

    # Keep the original training CSV row positions for exact source-row tracing.
    original_minority_positions = np.flatnonzero(y_train.to_numpy() == 1)
    smote = TraceableSMOTE(
        original_minority_positions=original_minority_positions,
        random_state=RANDOM_STATE,
        k_neighbors=5,
    )
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Convert outputs to predictable pandas objects for diagnostics and saving.
    X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
    y_resampled = pd.Series(np.asarray(y_resampled).astype(int), name="Class")

    counts_after = {
        str(int(k)): int(v)
        for k, v in y_resampled.value_counts().sort_index().items()
    }
    synthetic_count = len(X_resampled) - len(X_train)

    # ---- TRACE 2: validate SMOTE output ----
    print("\n=== TRACE: AFTER SMOTE ===")
    print(f"Resampled features shape: {X_resampled.shape}")
    print(f"Resampled labels shape:   {y_resampled.shape}")
    print(f"Class counts after:       {counts_after}")
    print(f"Rows before SMOTE:        {len(X_train)}")
    print(f"Rows after SMOTE:         {len(X_resampled)}")
    print(f"New synthetic rows:       {synthetic_count}")
    print(f"Missing values after:     {int(X_resampled.isna().sum().sum())}")

    if len(X_resampled) != len(y_resampled):
        raise ValueError("SMOTE output feature/label row counts do not match.")
    if X_resampled.isna().any().any() or not np.isfinite(X_resampled.to_numpy(dtype=float)).all():
        raise ValueError("SMOTE output contains missing or non-finite feature values.")
    if counts_after.get("0", 0) != counts_after.get("1", 0):
        raise ValueError(f"Expected balanced classes after SMOTE, got: {counts_after}")
    if synthetic_count != counts_after["1"] - counts_before["1"]:
        raise ValueError(
            "Synthetic-row count does not match the increase in minority-class rows. "
            "Check SMOTE output ordering and class counts."
        )

    # ---- FIGURE: training class counts before vs after SMOTE ----
    figure_path = FIGURE_ROOT / "training_class_counts_before_after_smote.png"
    plot_training_class_counts(counts_before, counts_after, figure_path)
    print(f"Saved training class-count figure: {figure_path}")

    # Standard SMOTE appends generated rows after the original training rows.
    # Inspect and save a small sample so the full synthetic dataset is not dumped to disk.
    synthetic_X = X_resampled.iloc[len(X_train):].reset_index(drop=True)
    synthetic_y = y_resampled.iloc[len(y_train):].reset_index(drop=True)
    source_trace = pd.DataFrame(smote.synthetic_source_trace)
    if len(source_trace) != len(synthetic_X):
        raise ValueError(
            f"SMOTE source trace has {len(source_trace)} records for "
            f"{len(synthetic_X)} synthetic rows. Check imbalanced-learn compatibility."
        )

    sample_n = min(SYNTHETIC_SAMPLE_SIZE, len(synthetic_X))
    trace_sample = source_trace.head(sample_n).reset_index(drop=True)
    synthetic_features_sample = synthetic_X.head(sample_n).reset_index(drop=True)
    synthetic_sample = pd.DataFrame({
        "synthetic_row_id": [f"SMOTE_FRAUD_{i + 1:06d}" for i in range(sample_n)],
        "generation_method": ["SMOTE"] * sample_n,
        "Class": synthetic_y.head(sample_n).to_numpy(),
    })
    synthetic_sample = pd.concat([synthetic_sample, trace_sample], axis=1)

    # Include source A, source B, and synthetic values for every feature. Source
    # row indexes are zero-based positions in train_X_scaled.csv/train_y.csv.
    for feature in X_train.columns:
        synthetic_sample[f"source_A_{feature}"] = [
            X_train.iloc[int(pos)][feature]
            for pos in trace_sample["source_row_A_train_index"]
        ]
        synthetic_sample[f"source_B_{feature}"] = [
            X_train.iloc[int(pos)][feature]
            for pos in trace_sample["source_row_B_train_index"]
        ]
        synthetic_sample[f"synthetic_{feature}"] = synthetic_features_sample[feature].to_numpy()
    synthetic_sample["synthetic_Class"] = synthetic_y.head(sample_n).to_numpy()

    print("\n=== TRACE: SAMPLE SYNTHETIC ROWS WITH SOURCE PAIRS ===")
    if synthetic_sample.empty:
        print("No synthetic rows were generated.")
    else:
        preview_cols = [
            "synthetic_row_id", "source_row_A_train_index", "source_row_B_train_index",
            "interpolation_lambda", "Class",
        ]
        print(synthetic_sample[preview_cols].to_string(index=False))
    print(f"Total synthetic fraud rows: {int((synthetic_y == 1).sum())}")

    DEBUG_ROOT.mkdir(parents=True, exist_ok=True)
    synthetic_sample_path = DEBUG_ROOT / "synthetic_fraud_rows_sample_with_sources.csv"
    synthetic_sample.to_csv(synthetic_sample_path, index=False)
    print(f"Saved traceable synthetic-row sample: {synthetic_sample_path}")

    # Validation data must remain untouched by SMOTE.
    print(
        f"Validation unchanged: {len(X_val)} rows; "
        f"fraud={int((y_val == 1).sum())}; legitimate={int((y_val == 0).sum())}"
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    }

    input_paths = [
        SCALED_ROOT / "train_X_scaled.csv", UNSCALED_ROOT / "train_y.csv",
        SCALED_ROOT / "validation_X_scaled.csv", UNSCALED_ROOT / "validation_y.csv",
    ]
    metadata = {
        "experiment": "phase_4_3_smote_models",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "smote": {"sampling_strategy": "auto", "k_neighbors": 5, "random_state": RANDOM_STATE},
        "class_weight": None,
        "threshold": THRESHOLD,
        "test_split_used": False,
        "feature_scaling": "Phase 3 StandardScaler fitted on training data only; scaled predictors used for all models so SMOTE nearest-neighbour distances are scale-aware.",
        "training_rows_before_smote": int(len(y_train)),
        "training_class_counts_before_smote": counts_before,
        "training_rows_after_smote": int(len(y_resampled)),
        "training_class_counts_after_smote": counts_after,
        "validation_rows": int(len(y_val)),
        "validation_fraud_count": int((y_val == 1).sum()),
        "validation_fraud_prevalence": float((y_val == 1).mean()),
        "feature_count": int(X_train.shape[1]),
        "features": list(X_train.columns),
        "scikit_learn_version": sklearn.__version__,
        "input_files": {str(p.relative_to(PROJECT_ROOT)): sha256_file(p) for p in input_paths},
        "note": "SMOTE was applied only to the training set. Validation was not resampled. Threshold 0.5 is used for a directly comparable initial result; threshold tuning is a later validation-only step.",
        "models": {},
    }

    rows = []
    for name, model in models.items():
        print(f"\nTraining {name} on SMOTE-resampled training data...")
        model.fit(X_resampled, y_resampled)
        print(f"  Training rows used: {len(X_resampled)}; validation rows evaluated: {len(X_val)}")
        probabilities = model.predict_proba(X_val)[:, 1]
        result = calculate_metrics(y_val.to_numpy(), probabilities)
        result.update({"model": name, "imbalance_strategy": "SMOTE_only"})
        rows.append(result)

        model_path = MODEL_ROOT / f"{name}_smote.joblib"
        joblib.dump(model, model_path)
        prediction_path = PREDICTION_ROOT / f"{name}_validation_predictions.csv"
        pd.DataFrame({
            "y_true": y_val.to_numpy(),
            "fraud_probability": probabilities,
            "y_pred_threshold_0_5": (probabilities >= THRESHOLD).astype(int),
        }).to_csv(prediction_path, index=False)
        metadata["models"][name] = {
            "parameters": model.get_params(deep=False),
            "model_path": str(model_path.relative_to(PROJECT_ROOT)),
            "model_sha256": sha256_file(model_path),
            "prediction_path": str(prediction_path.relative_to(PROJECT_ROOT)),
            "metrics": result,
        }
        print(f"  AP={result['average_precision']:.6f} | ROC-AUC={result['roc_auc']:.6f} | "
              f"Precision={result['precision']:.4f} | Recall={result['recall']:.4f} | "
              f"F1={result['f1']:.4f} | TP={result['tp']} FP={result['fp']} "
              f"FN={result['fn']} TN={result['tn']}")

    columns = ["model", "imbalance_strategy", "average_precision", "roc_auc", "threshold",
               "precision", "recall", "f1", "tn", "fp", "fn", "tp",
               "validation_rows", "validation_fraud_count", "validation_legitimate_count",
               "validation_fraud_prevalence"]
    metrics_path = OUTPUT_ROOT / "smote_validation_metrics.csv"
    pd.DataFrame(rows)[columns].to_csv(metrics_path, index=False)
    metadata["metrics_csv"] = str(metrics_path.relative_to(PROJECT_ROOT))
    metadata["metrics_csv_sha256"] = sha256_file(metrics_path)
    metadata_path = OUTPUT_ROOT / "smote_experiment_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")

    print("\nSaved metrics:", metrics_path)
    print("Saved metadata:", metadata_path)
    print("Saved models:", MODEL_ROOT)
    print("Saved validation predictions:", PREDICTION_ROOT)
    print("Completed. The test split was not loaded or used.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise

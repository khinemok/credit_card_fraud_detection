#!/usr/bin/env python3
"""Phase 4.2: Train class-weighted fraud detection models.

Run from the project root:
    python src/04_model_training/train_class_weighted_models.py

Uses Phase 3 scaled features for Logistic Regression and unscaled features
for Decision Tree/Random Forest. Target labels are read from the shared unscaled
target files because labels must not be scaled. The test split is never loaded.

This version also prints and records the actual training/validation class counts,
class proportions, and the balanced class weights calculated from TRAINING labels.
It does not create a separate weighted dataset.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight

RANDOM_STATE = 42
THRESHOLD = 0.5
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREPROCESSING_ROOT = PROJECT_ROOT / "outputs" / "preprocessing"
UNSCALED_ROOT = PREPROCESSING_ROOT / "unscaled"
SCALED_ROOT = PREPROCESSING_ROOT / "logistic_regression_scaled"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "class_weighted"
MODEL_ROOT = PROJECT_ROOT / "artifacts" / "class_weighted"
PREDICTION_ROOT = OUTPUT_ROOT / "validation_predictions"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
        raise ValueError(
            f"{label}: target file must have a Class column or one column total"
        )

    y = pd.Series(y.to_numpy(), name="Class").astype(int)

    if len(X) != len(y):
        raise ValueError(f"{label}: X/y row count mismatch ({len(X)} vs {len(y)})")
    if X.empty or X.isna().any().any() or y.isna().any():
        raise ValueError(f"{label}: empty data or missing values found")
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError(f"{label}: target must contain only 0 and 1")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError(f"{label}: non-finite predictor values found")
    if "Time" in X.columns:
        raise ValueError(
            f"{label}: Time is present; it should be excluded from primary predictors"
        )

    return X, y


def class_distribution(y: pd.Series) -> dict:
    """Return row counts and proportions for the binary target."""
    total = int(len(y))
    fraud_count = int((y == 1).sum())
    legitimate_count = int((y == 0).sum())

    return {
        "rows": total,
        "legitimate_count_class_0": legitimate_count,
        "fraud_count_class_1": fraud_count,
        "legitimate_proportion": float(legitimate_count / total) if total else 0.0,
        "fraud_proportion": float(fraud_count / total) if total else 0.0,
        "fraud_percentage": float(100 * fraud_count / total) if total else 0.0,
    }


def calculate_balanced_weights(y: pd.Series) -> dict:
    """Calculate sklearn's balanced class weights using training labels only."""
    classes = np.sort(y.unique())
    if len(classes) != 2 or set(classes) != {0, 1}:
        raise ValueError("Class-weight calculation requires both classes 0 and 1.")

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y.to_numpy(),
    )
    weight_map = {int(cls): float(weight) for cls, weight in zip(classes, weights)}

    # For binary labels, report how many times more weight one fraud example
    # receives than one legitimate example.
    ratio = weight_map[1] / weight_map[0] if weight_map[0] else None

    return {
        "formula": "total_training_rows / (number_of_classes * rows_in_class)",
        "class_0_legitimate_weight": weight_map[0],
        "class_1_fraud_weight": weight_map[1],
        "fraud_to_legitimate_per_row_weight_ratio": float(ratio) if ratio is not None else None,
        "calculated_from": "training labels only",
    }


def metrics(y_true, probability, threshold=THRESHOLD):
    y_pred = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "average_precision": float(average_precision_score(y_true, probability)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "validation_rows": int(len(y_true)),
        "validation_fraud_count": int(np.sum(y_true == 1)),
        "validation_legitimate_count": int(np.sum(y_true == 0)),
        "validation_fraud_prevalence": float(np.mean(y_true == 1)),
    }


def print_distribution(label: str, y: pd.Series) -> None:
    stats = class_distribution(y)
    print(f"\n{label} class distribution:")
    print(f"  Total rows:               {stats['rows']:,}")
    print(f"  Legitimate (Class 0):     {stats['legitimate_count_class_0']:,}")
    print(f"  Fraud (Class 1):          {stats['fraud_count_class_1']:,}")
    print(f"  Fraud percentage:         {stats['fraud_percentage']:.6f}%")


def main():
    for folder in (OUTPUT_ROOT, MODEL_ROOT, PREDICTION_ROOT):
        folder.mkdir(parents=True, exist_ok=True)

    X_train_lr, y_train = read_xy(
        SCALED_ROOT / "train_X_scaled.csv",
        UNSCALED_ROOT / "train_y.csv",
        "scaled train",
    )
    X_val_lr, y_val = read_xy(
        SCALED_ROOT / "validation_X_scaled.csv",
        UNSCALED_ROOT / "validation_y.csv",
        "scaled validation",
    )
    X_train_tree, y_train_tree = read_xy(
        UNSCALED_ROOT / "train_X.csv",
        UNSCALED_ROOT / "train_y.csv",
        "unscaled train",
    )
    X_val_tree, y_val_tree = read_xy(
        UNSCALED_ROOT / "validation_X.csv",
        UNSCALED_ROOT / "validation_y.csv",
        "unscaled validation",
    )

    if not np.array_equal(y_train.to_numpy(), y_train_tree.to_numpy()):
        raise ValueError("Scaled and unscaled training targets do not align")
    if not np.array_equal(y_val.to_numpy(), y_val_tree.to_numpy()):
        raise ValueError("Scaled and unscaled validation targets do not align")
    if list(X_train_lr.columns) != list(X_val_lr.columns):
        raise ValueError("Logistic Regression train/validation features do not align")
    if list(X_train_tree.columns) != list(X_val_tree.columns):
        raise ValueError("Tree-model train/validation features do not align")
    if list(X_train_lr.columns) != list(X_train_tree.columns):
        raise ValueError("Scaled and unscaled feature columns/order do not align")
    if y_train.nunique() != 2 or y_val.nunique() != 2:
        raise ValueError("Both training and validation data must contain both target classes")

    # Explicitly calculate and display the weights for transparency.
    # The estimators still use class_weight="balanced", which applies the same
    # sklearn balanced-weight formula internally during fit.
    train_distribution = class_distribution(y_train)
    validation_distribution = class_distribution(y_val)
    actual_class_weights = calculate_balanced_weights(y_train)

    print_distribution("TRAINING", y_train)
    print("\nBalanced class weights (calculated from TRAINING labels):")
    print(f"  Class 0 / legitimate:     {actual_class_weights['class_0_legitimate_weight']:.8f}")
    print(f"  Class 1 / fraud:          {actual_class_weights['class_1_fraud_weight']:.8f}")
    print(
        "  Fraud-to-legitimate per-row weight ratio: "
        f"{actual_class_weights['fraud_to_legitimate_per_row_weight_ratio']:.4f}x"
    )
    print_distribution("VALIDATION", y_val)
    print("\nNote: class weighting does not add, remove, or duplicate dataset rows.")

    model_specs = {
        "logistic_regression": (
            LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE,
                class_weight="balanced",
            ),
            X_train_lr, X_val_lr, y_train, y_val,
        ),
        "decision_tree": (
            DecisionTreeClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced",
            ),
            X_train_tree, X_val_tree, y_train_tree, y_val_tree,
        ),
        "random_forest": (
            RandomForestClassifier(
                n_estimators=200,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced",
            ),
            X_train_tree, X_val_tree, y_train_tree, y_val_tree,
        ),
    }

    input_paths = [
        SCALED_ROOT / "train_X_scaled.csv",
        UNSCALED_ROOT / "train_y.csv",
        SCALED_ROOT / "validation_X_scaled.csv",
        UNSCALED_ROOT / "validation_y.csv",
        UNSCALED_ROOT / "train_X.csv",
        UNSCALED_ROOT / "train_y.csv",
        UNSCALED_ROOT / "validation_X.csv",
        UNSCALED_ROOT / "validation_y.csv",
    ]

    metadata = {
        "experiment": "phase_4_2_class_weighted_models",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "class_weight": "balanced",
        "balanced_class_weights_calculated_from_training_labels": actual_class_weights,
        "training_class_distribution": train_distribution,
        "validation_class_distribution": validation_distribution,
        "threshold": THRESHOLD,
        "test_split_used": False,
        "training_rows": int(len(y_train)),
        "training_fraud_count": int((y_train == 1).sum()),
        "training_legitimate_count": int((y_train == 0).sum()),
        "validation_rows": int(len(y_val)),
        "validation_fraud_count": int((y_val == 1).sum()),
        "validation_legitimate_count": int((y_val == 0).sum()),
        "feature_count": int(X_train_lr.shape[1]),
        "features": list(X_train_lr.columns),
        "scikit_learn_version": sklearn.__version__,
        "input_files": {
            str(p.relative_to(PROJECT_ROOT)): sha256_file(p)
            for p in input_paths if p.exists()
        },
        "models": {},
        "note": (
            "Threshold fixed at 0.5 for this experiment; no threshold tuning performed. "
            "Class weights affect model fitting and do not create a separate weighted dataset."
        ),
    }

    metric_rows = []

    for name, (model, X_tr, X_va, y_tr, y_va) in model_specs.items():
        print(f"\nTraining {name} with class_weight='balanced'...")
        model.fit(X_tr, y_tr)
        probabilities = model.predict_proba(X_va)[:, 1]
        result = metrics(y_va.to_numpy(), probabilities, threshold=THRESHOLD)
        result.update({"model": name, "class_weight": "balanced"})
        metric_rows.append(result)

        model_path = MODEL_ROOT / f"{name}_class_weighted.joblib"
        joblib.dump(model, model_path)

        prediction_path = PREDICTION_ROOT / f"{name}_validation_predictions.csv"
        pd.DataFrame({
            "y_true": y_va.to_numpy(),
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

        print(
            f"  AP={result['average_precision']:.6f} | "
            f"ROC-AUC={result['roc_auc']:.6f} | "
            f"Precision={result['precision']:.4f} | "
            f"Recall={result['recall']:.4f} | "
            f"F1={result['f1']:.4f} | "
            f"TP={result['tp']} FP={result['fp']} "
            f"FN={result['fn']} TN={result['tn']}"
        )

    columns = [
        "model", "class_weight", "average_precision", "roc_auc", "threshold",
        "precision", "recall", "f1", "tn", "fp", "fn", "tp",
        "validation_rows", "validation_fraud_count", "validation_legitimate_count",
        "validation_fraud_prevalence",
    ]
    metrics_path = OUTPUT_ROOT / "class_weighted_validation_metrics.csv"
    pd.DataFrame(metric_rows)[columns].to_csv(metrics_path, index=False)

    metadata["metrics_csv"] = str(metrics_path.relative_to(PROJECT_ROOT))
    metadata["metrics_csv_sha256"] = sha256_file(metrics_path)
    metadata_path = OUTPUT_ROOT / "class_weighted_experiment_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    print("\nSaved metrics:", metrics_path)
    print("Saved metadata:", metadata_path)
    print("Saved models:", MODEL_ROOT)
    print("Saved predictions:", PREDICTION_ROOT)
    print("Completed. The test split was not loaded or used.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise

#!/usr/bin/env python3
"""Phase 4.3: Train fraud-detection models using SMOTE on training data only.

Run from the project root:
    python src/04_model_training/train_smote_models.py

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
import numpy as np
import pandas as pd
import sklearn
from imblearn.over_sampling import SMOTE
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


def main():
    for folder in (OUTPUT_ROOT, MODEL_ROOT, PREDICTION_ROOT):
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

    counts_before = {str(int(k)): int(v) for k, v in y_train.value_counts().sort_index().items()}
    print("Training class counts before SMOTE:", counts_before)
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    counts_after = {str(int(k)): int(v) for k, v in pd.Series(y_resampled).value_counts().sort_index().items()}
    print("Training class counts after SMOTE:", counts_after)
    print(f"Validation rows remain unchanged: {len(X_val)}; validation fraud count: {(y_val == 1).sum()}")

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

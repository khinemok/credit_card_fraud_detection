
from pathlib import Path
import json
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

warnings.filterwarnings("default")

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "outputs" / "preprocessing"
MODEL_DIR = ROOT / "artifacts" / "models"
RESULT_DIR = ROOT / "outputs" / "evaluation"
PRED_DIR = RESULT_DIR / "baseline_predictions"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)


def load_split(split):
    y_path = DATA / "unscaled" / f"{split}_y.csv"

    # Logistic Regression uses training-fitted scaling.
    if split == "train":
        x_path = (
            DATA / "logistic_regression_scaled"
            / f"{split}_X_scaled.csv"
        )
    else:
        x_path = (
            DATA / "logistic_regression_scaled"
            / f"{split}_X_scaled.csv"
        )

    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path).squeeze("columns").astype(int)

    if len(X) != len(y):
        raise ValueError(f"{split}: X/y row counts do not match.")

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError(f"{split}: X contains non-finite values.")

    return X, y


def classification_metrics(y_true, probabilities, threshold=0.5):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "threshold": float(threshold),
        "average_precision": float(
            average_precision_score(y_true, probabilities)
        ),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main():
    print("Loading Phase 3 preprocessed data...")

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("validation")

    expected_features = list(X_train.columns)

    if list(X_val.columns) != expected_features:
        raise ValueError("Train and validation feature columns differ.")

    if y_train.nunique() != 2 or y_val.nunique() != 2:
        raise ValueError("Expected both classes in train and validation.")

    print(f"Training rows: {len(X_train):,}")
    print(f"Validation rows: {len(X_val):,}")
    print(f"Features: {len(expected_features)}")
    print(f"Training fraud cases: {int(y_train.sum())}")
    print(f"Validation fraud cases: {int(y_val.sum())}")

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            random_state=42,
        ),
        "decision_tree": DecisionTreeClassifier(
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")

        # Fit only on the training set.
        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_val)[:, 1]
        metrics = classification_metrics(y_val, probabilities)

        row = {"model": name, **metrics}
        results.append(row)

        joblib.dump(model, MODEL_DIR / f"{name}_baseline.joblib")

        prediction_file = PRED_DIR / f"{name}_validation_predictions.csv"
        pd.DataFrame({
            "y_true": y_val.to_numpy(),
            "fraud_probability": probabilities,
            "predicted_class_at_0_5": (
                probabilities >= 0.5
            ).astype(int),
        }).to_csv(prediction_file, index=False)

        print(
            f"  Average precision (PR-AUC summary): "
            f"{metrics['average_precision']:.6f}"
        )
        print(f"  ROC-AUC:  {metrics['roc_auc']:.6f}")
        print(f"  Precision: {metrics['precision']:.6f}")
        print(f"  Recall:    {metrics['recall']:.6f}")
        print(f"  F1:        {metrics['f1']:.6f}")
        print(
            f"  Confusion matrix counts: "
            f"TN={metrics['tn']}, FP={metrics['fp']}, "
            f"FN={metrics['fn']}, TP={metrics['tp']}"
        )

    results_df = pd.DataFrame(results).sort_values(
        "average_precision", ascending=False
    )

    results_path = RESULT_DIR / "baseline_validation_metrics.csv"
    results_df.to_csv(results_path, index=False)

    metadata = {
        "experiment": "baseline_models",
        "training_split": "train",
        "evaluation_split": "validation",
        "threshold": 0.5,
        "resampling": "none",
        "class_weighting": "none",
        "features": expected_features,
        "models": list(models.keys()),
        "note": (
            "Baseline comparison only. Threshold 0.5 is a reference; "
            "it is not assumed optimal for imbalanced fraud detection."
        ),
    }

    with open(
        RESULT_DIR / "baseline_experiment_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2)

    print("\nBaseline validation results:")
    print(results_df.to_string(index=False))
    print(f"\nSaved metrics: {results_path}")
    print(f"Saved models: {MODEL_DIR}")
    print("\nPhase 4 baseline training completed.")


if __name__ == "__main__":
    main()

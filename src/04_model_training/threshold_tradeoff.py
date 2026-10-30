"""
Phase 4.5: Threshold vs False-Positive Trade-off Analysis
Run from the project root:
    python src/04_model_training/phase_4_5_threshold_tradeoff.py

Uses saved VALIDATION prediction probabilities only.
Does not retrain models and does not load test-set predictions.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "threshold"
FIGURES = OUT / "figures"

FILES = {
    ("Logistic Regression", "Baseline"): ROOT / "outputs/evaluation/baseline_predictions/logistic_regression_validation_predictions.csv",
    ("Decision Tree", "Baseline"): ROOT / "outputs/evaluation/baseline_predictions/decision_tree_validation_predictions.csv",
    ("Random Forest", "Baseline"): ROOT / "outputs/evaluation/baseline_predictions/random_forest_validation_predictions.csv",
    ("Logistic Regression", "Class-weighted"): ROOT / "outputs/class_weighted/validation_predictions/logistic_regression_validation_predictions.csv",
    ("Decision Tree", "Class-weighted"): ROOT / "outputs/class_weighted/validation_predictions/decision_tree_validation_predictions.csv",
    ("Random Forest", "Class-weighted"): ROOT / "outputs/class_weighted/validation_predictions/random_forest_validation_predictions.csv",
    ("Logistic Regression", "SMOTE"): ROOT / "outputs/smote/validation_predictions/logistic_regression_validation_predictions.csv",
    ("Decision Tree", "SMOTE"): ROOT / "outputs/smote/validation_predictions/decision_tree_validation_predictions.csv",
    ("Random Forest", "SMOTE"): ROOT / "outputs/smote/validation_predictions/random_forest_validation_predictions.csv",
}

# Dense, consistent grid makes model/strategy comparisons easy to interpret.
# Include 0.5 explicitly; probabilities are evaluated using p >= threshold.
THRESHOLDS = np.unique(np.concatenate([np.linspace(0.0, 1.0, 201), [0.5]]))


def load_predictions(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing validation prediction file:\n{path}\n"
            "Run the relevant baseline, class-weighted, and SMOTE training scripts first."
        )

    df = pd.read_csv(path)
    if "y_true" not in df.columns:
        raise ValueError(f"{path}: expected a 'y_true' column; found {list(df.columns)}")

    probability_col = next(
        (col for col in ["fraud_probability", "y_probability", "y_prob", "probability"]
         if col in df.columns),
        None,
    )
    if probability_col is None:
        raise ValueError(
            f"{path}: no probability column found. Expected one of "
            "'fraud_probability', 'y_probability', 'y_prob', 'probability'."
        )

    y = pd.to_numeric(df["y_true"], errors="raise").to_numpy(dtype=int)
    p = pd.to_numeric(df[probability_col], errors="raise").to_numpy(dtype=float)

    if len(y) == 0:
        raise ValueError(f"{path}: prediction file is empty.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError(f"{path}: y_true must contain only 0 and 1.")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError(f"{path}: probabilities must be finite values between 0 and 1.")
    if len(np.unique(y)) < 2:
        raise ValueError(f"{path}: validation labels must contain both classes.")

    return y, p


def metrics_at_threshold(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "flagged_count": int(pred.sum()),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    rows = []
    loaded = {}

    for (model, strategy), path in FILES.items():
        y, p = load_predictions(path)
        loaded[(model, strategy)] = (y, p)

        for threshold in THRESHOLDS:
            rows.append({
                "model": model,
                "strategy": strategy,
                "n_validation": int(len(y)),
                "fraud_count": int(y.sum()),
                "threshold": float(threshold),
                **metrics_at_threshold(y, p, threshold),
                "prediction_file": str(path.relative_to(ROOT)),
            })

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "threshold_tradeoff.csv", index=False)

    # Save a compact comparison at common thresholds.
    selected_thresholds = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    compact = results[results["threshold"].isin(selected_thresholds)].copy()
    compact.to_csv(OUT / "threshold_tradeoff_selected_thresholds.csv", index=False)

    # Graph 1: Recall against number of false positives, one line per strategy.
    for model in ["Logistic Regression", "Decision Tree", "Random Forest"]:
        fig, ax = plt.subplots(figsize=(9, 6))
        for strategy in ["Baseline", "Class-weighted", "SMOTE"]:
            subset = results[
                (results["model"] == model) & (results["strategy"] == strategy)
            ].sort_values("threshold", ascending=False)
            ax.plot(subset["fp"], subset["recall"], marker=".", markersize=3, label=strategy)

        ax.set_title(f"Recall vs False Positives — {model}")
        ax.set_xlabel("False positives (legitimate transactions incorrectly flagged)")
        ax.set_ylabel("Recall (proportion of fraud detected)")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        filename = model.lower().replace(" ", "_") + "_recall_vs_fp.png"
        fig.savefig(FIGURES / filename, dpi=200)
        plt.close(fig)

    # Graph 2: threshold against FP count, one line per strategy/model.
    fig, ax = plt.subplots(figsize=(11, 7))
    for (model, strategy), subset in results.groupby(["model", "strategy"], sort=False):
        subset = subset.sort_values("threshold")
        ax.plot(
            subset["threshold"], subset["fp"],
            label=f"{model} — {strategy}", linewidth=1.5
        )
    ax.set_title("Classification Threshold vs False Positives")
    ax.set_xlabel("Classification threshold (fraud if probability >= threshold)")
    ax.set_ylabel("False positives")
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "threshold_vs_false_positives_all_models.png", dpi=200)
    plt.close(fig)

    # Graph 3: threshold against recall, one line per strategy/model.
    fig, ax = plt.subplots(figsize=(11, 7))
    for (model, strategy), subset in results.groupby(["model", "strategy"], sort=False):
        subset = subset.sort_values("threshold")
        ax.plot(
            subset["threshold"], subset["recall"],
            label=f"{model} — {strategy}", linewidth=1.5
        )
    ax.set_title("Classification Threshold vs Recall")
    ax.set_xlabel("Classification threshold (fraud if probability >= threshold)")
    ax.set_ylabel("Recall")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "threshold_vs_recall_all_models.png", dpi=200)
    plt.close(fig)

    print("\nPHASE 4.5 COMPLETE — VALIDATION ONLY")
    print(f"Rows written: {len(results):,}")
    print(f"CSV: {OUT / 'threshold_tradeoff.csv'}")
    print(f"CSV: {OUT / 'threshold_tradeoff_selected_thresholds.csv'}")
    print(f"Figures: {FIGURES}")
    print("\nAt threshold 0.50:")
    print(
        results[np.isclose(results["threshold"], 0.50)][
            ["model", "strategy", "precision", "recall", "f1", "fp", "fn"]
        ].to_string(index=False)
    )
    print("\nTest set was not loaded.")


if __name__ == "__main__":
    main()

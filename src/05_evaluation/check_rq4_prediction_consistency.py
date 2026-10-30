
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def main():
    # Project root: credit_card_fraud_detection_ulb
    root = Path(__file__).resolve().parents[2]

    checks = [
        {
            "name": "Decision Tree",
            "model": "artifacts/models/decision_tree_baseline.joblib",
            "features": "outputs/preprocessing/unscaled/validation_X.csv",
            "predictions": (
                "outputs/evaluation/baseline_predictions/"
                "decision_tree_validation_predictions.csv"
            ),
        },
        {
            "name": "Random Forest",
            "model": "artifacts/models/random_forest_baseline.joblib",
            "features": "outputs/preprocessing/unscaled/validation_X.csv",
            "predictions": (
                "outputs/evaluation/baseline_predictions/"
                "random_forest_validation_predictions.csv"
            ),
        },
    ]

    y_path = (
        root / "outputs/preprocessing/unscaled/validation_y.csv"
    )
    y = pd.read_csv(y_path).iloc[:, 0].astype(int).to_numpy()

    for item in checks:
        print("\n" + "=" * 55)
        print(item["name"])
        print("=" * 55)

        model_path = root / item["model"]
        features_path = root / item["features"]
        predictions_path = root / item["predictions"]

        for path in [model_path, features_path, predictions_path]:
            if not path.exists():
                raise FileNotFoundError(f"Required file not found: {path}")

        X = pd.read_csv(features_path)
        saved = pd.read_csv(predictions_path)
        model = joblib.load(model_path)

        current_scores = model.predict_proba(X)[:, 1]
        saved_scores = saved["fraud_probability"].to_numpy()
        saved_y = saved["y_true"].astype(int).to_numpy()

        print(
            "Saved prediction AP:",
            f"{average_precision_score(saved_y, saved_scores):.6f}",
        )
        print(
            "Current model AP:",
            f"{average_precision_score(y, current_scores):.6f}",
        )
        print(
            "Saved labels match validation labels:",
            np.array_equal(saved_y, y),
        )
        print(
            "Probabilities match:",
            np.allclose(saved_scores, current_scores),
        )
        print(
            "Maximum absolute probability difference:",
            f"{np.max(np.abs(saved_scores - current_scores)):.10f}",
        )


if __name__ == "__main__":
    main()

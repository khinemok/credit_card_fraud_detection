
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score


def main():
    root = Path(__file__).resolve().parents[2]

    X_path = (
        root / "outputs/preprocessing/"
        "logistic_regression_scaled/validation_X_scaled.csv"
    )
    y_path = (
        root / "outputs/preprocessing/"
        "unscaled/validation_y.csv"
    )

    checks = {
        "Decision Tree": (
            root / "artifacts/models/decision_tree_baseline.joblib",
            0.282910,
        ),
        "Random Forest": (
            root / "artifacts/models/random_forest_baseline.joblib",
            0.772913,
        ),
    }

    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path).iloc[:, 0].astype(int)

    print("Testing scaled validation features")
    print("Rows:", len(X))
    print("Features:", len(X.columns))

    for name, (model_path, original_ap) in checks.items():
        model = joblib.load(model_path)
        probabilities = model.predict_proba(X)[:, 1]
        ap = average_precision_score(y, probabilities)

        print(f"\n{name}")
        print(f"Original Phase 4.4 AP: {original_ap:.6f}")
        print(f"AP on scaled features: {ap:.6f}")
        print(f"Absolute difference: {abs(ap - original_ap):.6f}")


if __name__ == "__main__":
    main()

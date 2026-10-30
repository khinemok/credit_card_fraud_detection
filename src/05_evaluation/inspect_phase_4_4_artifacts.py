
from pathlib import Path
import json
import hashlib

import joblib


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    root = Path(__file__).resolve().parents[2]

    metadata_path = (
        root / "outputs/phase_4_4/phase_4_4_metadata.json"
    )

    print("\n=== PHASE 4.4 METADATA ===")
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        print(json.dumps(metadata, indent=2))
    else:
        print("Metadata file not found:", metadata_path)

    model_paths = [
        root / "artifacts/models/decision_tree_baseline.joblib",
        root / "artifacts/models/random_forest_baseline.joblib",
        root / "artifacts/models/logistic_regression_baseline.joblib",
    ]

    print("\n=== CURRENT BASELINE MODEL ARTIFACTS ===")

    for path in model_paths:
        print(f"\nModel: {path.name}")

        if not path.exists():
            print("File not found.")
            continue

        print("SHA-256:", sha256_file(path))

        try:
            model = joblib.load(path)
            print("Model class:", type(model).__name__)

            for attribute in [
                "n_features_in_",
                "n_outputs_",
                "classes_",
                "random_state",
                "max_depth",
                "n_estimators",
                "criterion",
                "class_weight",
            ]:
                if hasattr(model, attribute):
                    print(f"{attribute}: {getattr(model, attribute)}")

        except Exception as error:
            print("Could not inspect model:", repr(error))


if __name__ == "__main__":
    main()

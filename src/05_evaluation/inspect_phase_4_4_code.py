
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]

    keywords = [
        "validation_comparison_threshold_0_5.csv",
        "precision_recall_curve_points.csv",
        "baseline_predictions",
        "decision_tree_baseline.joblib",
        "random_forest_baseline.joblib",
        "average_precision_score",
        "train_test_split",
    ]

    excluded_dirs = {
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        ".ipynb_checkpoints",
    }

    print("Project root:", root)
    print("\nSearching project Python files...\n")

    found = False

    for path in (root / "src").rglob("*.py"):
        if any(part in excluded_dirs for part in path.parts):
            continue

        try:
            lines = path.read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines()
        except OSError:
            continue

        matches = [
            i for i, line in enumerate(lines)
            if any(keyword in line for keyword in keywords)
        ]

        if not matches:
            continue

        found = True
        print("=" * 75)
        print(path.relative_to(root))

        shown = set()

        for i in matches:
            start = max(0, i - 4)
            end = min(len(lines), i + 5)

            for j in range(start, end):
                if j not in shown:
                    print(f"{j + 1:4}: {lines[j]}")
                    shown.add(j)

            print()

    if not found:
        print("No matching source files found under src.")


if __name__ == "__main__":
    main()

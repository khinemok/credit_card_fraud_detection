#!/usr/bin/env python3
"""Menu-driven runner for the uploaded credit-card fraud research pipeline.
Run this file from the project root: python run_all.py
"""
from pathlib import Path
import subprocess, sys

ROOT = Path(__file__).resolve().parent
PY = sys.executable

STAGES = [
    ("EDA", "src/01_eda/eda_credit_card_fraud.py"),
    ("Data validation and chronological split", "src/02_data_validation/validate_and_split.py"),
    ("Preprocessing and train-only scaling", "src/03_preprocessing/prepare_phase3.py"),
    ("Verify Phase 3 outputs (canonical dataset row counts)", "src/03_preprocessing/verify_phase3.py"),
    ("Baseline model training", "src/04_model_training/train_baseline_models.py"),
    ("Class-weighted model training", "src/04_model_training/train_class_weighted_models.py"),
    ("SMOTE model training", "src/04_model_training/train_smote_models.py"),
    ("Phase 4.4 validation comparison", "src/04_model_training/phase_4_4_validation_comparison.py"),
    ("Threshold trade-off analysis", "src/04_model_training/threshold_tradeoff.py"),
    ("RQ4 feature association analysis", "src/05_evaluation/analyze_rq4_feature_associations.py"),
    ("RQ4 prediction consistency check", "src/05_evaluation/check_rq4_prediction_consistency.py"),
    ("RQ5 independent-fit stability analysis", "src/04_model_training/rq5_independent_fit_stability.py"),
]
FINAL_TEST = ("Phase 4.5 frozen final-test evaluation", "src/04_model_training/evaluate_final_test.py")

def run(label, relpath):
    script = ROOT / relpath
    if not script.exists():
        print(f"\n[SKIP/ERROR] Script not found: {script}")
        return False
    print(f"\n{'='*72}\nRUNNING: {label}\nScript: {relpath}\n{'='*72}", flush=True)
    result = subprocess.run([PY, str(script)], cwd=ROOT)
    if result.returncode:
        print(f"\nSTOPPED: '{label}' exited with code {result.returncode}.")
        print("Fix the error above, then rerun this stage from the menu.")
        return False
    print(f"\nCOMPLETED: {label}")
    return True

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--stage":
        try: index = int(sys.argv[2])
        except (IndexError, ValueError):
            print("Usage: python run_all.py --stage NUMBER"); return 2
        all_stages = STAGES + [FINAL_TEST]
        if not 1 <= index <= len(all_stages):
            print(f"Stage number must be 1–{len(all_stages)}"); return 2
        return 0 if run(*all_stages[index-1]) else 1

    print("\nCREDIT CARD FRAUD DETECTION — PIPELINE RUNNER")
    print("Run from the project root. The raw CSV is never modified.\n")
    for i, (label, path) in enumerate(STAGES, 1):
        print(f"{i:2}. {label}")
    print(f"{len(STAGES)+1:2}. {FINAL_TEST[0]} (run only after freezing model + threshold)")
    print(" A. Run the core pipeline (stages 1–10, stop on first error)")
    print(" Q. Quit")
    choice = input("\nChoose a stage number, A, or Q: ").strip().lower()
    if choice == "q": return 0
    if choice == "a":
        for label, path in STAGES[:10]:
            if not run(label, path): return 1
        print("\nCore pipeline finished. Review outputs before choosing a final model.")
        return 0
    try: index = int(choice)
    except ValueError:
        print("Invalid choice."); return 2
    all_stages = STAGES + [FINAL_TEST]
    if not 1 <= index <= len(all_stages):
        print(f"Choose a number from 1 to {len(all_stages)}."); return 2
    if index == len(all_stages):
        print("\nWARNING: Phase 4.5 uses the untouched test set. Select the model and threshold")
        print("from validation results first, then create phase_4_5_frozen_config.json.")
        confirm = input("Have you frozen the model and threshold using validation only? (yes/no): ").strip().lower()
        if confirm != "yes":
            print("Not running final-test evaluation."); return 0
    return 0 if run(*all_stages[index-1]) else 1

if __name__ == "__main__":
    raise SystemExit(main())

"""RQ4 feature-association and predictive-importance analysis.

RQ4: Which anonymized transaction features show the strongest statistical and
predictive association with fraudulent transactions?

Design:
- Statistical associations are calculated on the training split only.
- Native model importance is extracted from already-trained baseline models.
- Permutation importance is evaluated on the validation split using Average
  Precision (AP); validation is not used to fit models or transformations.
- The final test split is deliberately not loaded or inspected by this script.

Run from any directory:
    python -u src/06_evaluation/analyze_rq4_feature_associations.py
"""
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score
import joblib

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "rq4_feature_analysis"
TABLES = OUT / "tables"
FIGS = OUT / "figures"
TRAIN_X = ROOT / "outputs" / "preprocessing" / "unscaled" / "train_X.csv"
TRAIN_Y = ROOT / "outputs" / "preprocessing" / "unscaled" / "train_y.csv"
VAL_X = ROOT / "outputs" / "preprocessing" / "unscaled" / "validation_X.csv"
VAL_X_SCALED = ROOT / "outputs" / "preprocessing" / "logistic_regression_scaled" / "validation_X_scaled.csv"
VAL_Y = ROOT / "outputs" / "preprocessing" / "unscaled" / "validation_y.csv"
MODELS = {
    "Logistic Regression (baseline)": ROOT / "artifacts" / "models" / "logistic_regression_baseline.joblib",
    "Decision Tree (baseline)": ROOT / "artifacts" / "models" / "decision_tree_baseline.joblib",
    "Random Forest (baseline)": ROOT / "artifacts" / "models" / "random_forest_baseline.joblib",
}
SEED = 42
N_REPEATS = 20
TOP_N = 15


def bh_adjust(p_values):
    """Benjamini-Hochberg false-discovery-rate adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty(n, dtype=float)
    result[order] = np.clip(adjusted, 0, 1)
    return result


def get_scores(model, X, y):
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
    else:
        scores = model.decision_function(X)
    return average_precision_score(y, scores)


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    required = [TRAIN_X, TRAIN_Y, VAL_X, VAL_X_SCALED, VAL_Y, *MODELS.values()]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Required RQ4 inputs are missing:\n- " + "\n- ".join(missing))

    X_train = pd.read_csv(TRAIN_X)
    y_train = pd.read_csv(TRAIN_Y).iloc[:, 0].astype(int)
    X_val_unscaled = pd.read_csv(VAL_X)
    X_val_scaled = pd.read_csv(VAL_X_SCALED)
    y_val = pd.read_csv(VAL_Y).iloc[:, 0].astype(int)
    # Explicitly enforce the same feature order and exclude target/time identifiers.
    features = [c for c in X_train.columns if c not in {"Class", "Time", "id"}]
    X_train = X_train[features]
    X_val_unscaled = X_val_unscaled[features]
    X_val_scaled = X_val_scaled[features]
    if list(X_train.columns) != list(X_val_unscaled.columns) or list(X_train.columns) != list(X_val_scaled.columns):
        raise ValueError("Train and validation feature columns/order do not match.")
    if len(X_train) != len(y_train) or len(X_val_unscaled) != len(y_val) or len(X_val_scaled) != len(y_val):
        raise ValueError("Feature/label row counts do not match.")
    if set(pd.unique(y_train)) != {0, 1}:
        raise ValueError("Training labels must contain both classes 0 and 1.")

    # 1) Statistical association: training split only.
    rows = []
    for feature in features:
        legit = X_train.loc[y_train == 0, feature].dropna().to_numpy()
        fraud = X_train.loc[y_train == 1, feature].dropna().to_numpy()
        if len(legit) and len(fraud):
            u = mannwhitneyu(fraud, legit, alternative="two-sided", method="asymptotic")
            # Rank-biserial effect size: 2*U/(n_fraud*n_legit)-1; magnitude indicates separation.
            rbc = 2 * float(u.statistic) / (len(fraud) * len(legit)) - 1
            rho, rho_p = spearmanr(X_train[feature], y_train, nan_policy="omit")
            rows.append({
                "feature": feature,
                "train_legitimate_median": float(np.median(legit)),
                "train_fraud_median": float(np.median(fraud)),
                "median_difference_fraud_minus_legitimate": float(np.median(fraud) - np.median(legit)),
                "mann_whitney_u": float(u.statistic),
                "mann_whitney_p_value": float(u.pvalue),
                "mann_whitney_p_fdr_bh": np.nan,
                "rank_biserial_correlation": rbc,
                "abs_rank_biserial_correlation": abs(rbc),
                "spearman_target_correlation": float(rho) if np.isfinite(rho) else np.nan,
                "spearman_p_value": float(rho_p) if np.isfinite(rho_p) else np.nan,
            })
    assoc = pd.DataFrame(rows)
    assoc["mann_whitney_p_fdr_bh"] = bh_adjust(assoc["mann_whitney_p_value"].values)
    assoc["spearman_p_fdr_bh"] = bh_adjust(assoc["spearman_p_value"].fillna(1).values)
    assoc = assoc.sort_values("abs_rank_biserial_correlation", ascending=False)
    assoc.to_csv(TABLES / "training_statistical_feature_associations.csv", index=False)

    # 2) Native importance from existing baseline models.
    native_rows = []
    loaded = {}
    for model_name, model_path in MODELS.items():
        model = joblib.load(model_path)
        loaded[model_name] = model
        if hasattr(model, "feature_importances_"):
            values = np.asarray(model.feature_importances_, dtype=float)
            method = "model-native impurity importance"
        elif hasattr(model, "coef_"):
            values = np.abs(np.asarray(model.coef_).ravel())
            method = "absolute logistic-regression coefficient (scaled model)"
        else:
            continue
        if len(values) != len(features):
            raise ValueError(f"Importance length mismatch for {model_name}: {len(values)} vs {len(features)}")
        for f, value in zip(features, values):
            native_rows.append({"model": model_name, "feature": f, "importance": float(value), "importance_method": method})
    native = pd.DataFrame(native_rows)
    native["rank"] = native.groupby("model")["importance"].rank(method="min", ascending=False).astype(int)
    native.to_csv(TABLES / "native_feature_importance.csv", index=False)

    # 3) Validation permutation importance using AP decrease. No model fitting occurs here.
    perm_rows = []
    baseline_ap_checks = {}
    for model_name, model in loaded.items():
        # Use scaled validation features for all three existing baseline artifacts.
        # This input was empirically verified to reproduce the original Phase 4.4 AP
        # scores exactly for Logistic Regression, Decision Tree, and Random Forest.
        # This preserves the existing experiment; it does not imply that tree models
        # generally require feature scaling.
        X_val_model = X_val_scaled
        base_ap = get_scores(model, X_val_model, y_val)
        baseline_ap_checks[model_name] = float(base_ap)
        expected_ap = {
            "Logistic Regression (baseline)": 0.650716,
            "Decision Tree (baseline)": 0.282910,
            "Random Forest (baseline)": 0.772913,
        }[model_name]
        if not np.isclose(base_ap, expected_ap, atol=1e-5, rtol=0):
            raise ValueError(
                f"Baseline AP mismatch for {model_name}: expected {expected_ap:.6f}, "
                f"got {base_ap:.6f}. Stop before interpreting RQ4 feature importance."
            )
        result = permutation_importance(
            model, X_val_model, y_val,
            scoring="average_precision",
            n_repeats=N_REPEATS,
            random_state=SEED,
            n_jobs=-1,
        )
        for i, feature in enumerate(features):
            perm_rows.append({
                "model": model_name,
                "feature": feature,
                "validation_baseline_ap": float(base_ap),
                "mean_ap_decrease_after_permutation": float(result.importances_mean[i]),
                "std_ap_decrease": float(result.importances_std[i]),
                "n_repeats": N_REPEATS,
                "evaluation_split": "validation",
            })
    perm = pd.DataFrame(perm_rows)
    perm["rank"] = perm.groupby("model")["mean_ap_decrease_after_permutation"].rank(method="min", ascending=False).astype(int)
    perm.to_csv(TABLES / "validation_permutation_importance.csv", index=False)

    # 4) Summarize top features and rank agreement across models.
    top_rows = []
    for model_name, group in perm.groupby("model"):
        top = group.sort_values("mean_ap_decrease_after_permutation", ascending=False).head(TOP_N)
        for _, row in top.iterrows():
            top_rows.append({"model": model_name, "feature": row["feature"], "rank": int(row["rank"]),
                             "mean_ap_decrease": row["mean_ap_decrease_after_permutation"],
                             "std_ap_decrease": row["std_ap_decrease"]})
    pd.DataFrame(top_rows).to_csv(TABLES / "top_permutation_features_by_model.csv", index=False)

    rank_wide = perm.pivot(index="feature", columns="model", values="rank").reset_index()
    rank_wide.to_csv(TABLES / "permutation_importance_rankings_across_models.csv", index=False)

    # Figures: training statistical effect size and validation predictive importance.
    top_assoc = assoc.head(TOP_N).sort_values("abs_rank_biserial_correlation")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top_assoc["feature"], top_assoc["abs_rank_biserial_correlation"])
    ax.set_xlabel("Absolute rank-biserial correlation")
    ax.set_title("Top training-split statistical associations with fraud")
    fig.tight_layout()
    fig.savefig(FIGS / "top_statistical_associations.png", dpi=180)
    plt.close(fig)

    for model_name, group in perm.groupby("model"):
        top = group.sort_values("mean_ap_decrease_after_permutation", ascending=True).tail(TOP_N)
        safe = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.barh(top["feature"], top["mean_ap_decrease_after_permutation"],
                xerr=top["std_ap_decrease"], capsize=2)
        ax.set_xlabel("Decrease in validation Average Precision after permutation")
        ax.set_title(f"Top validation permutation importance — {model_name}")
        fig.tight_layout()
        fig.savefig(FIGS / f"permutation_importance_{safe}.png", dpi=180)
        plt.close(fig)

    metadata = {
        "research_question": "Which anonymized transaction features show the strongest statistical and predictive association with fraudulent transactions?",
        "status": "RQ4 feature-association analysis generated",
        "statistical_association_split": "training only",
        "permutation_importance_split": "validation only",
        "test_split_loaded": False,
        "feature_columns": features,
        "feature_count": len(features),
        "train_rows": int(len(X_train)),
        "train_fraud_count": int(y_train.sum()),
        "validation_rows": int(len(X_val_unscaled)),
        "validation_baseline_ap_by_model": baseline_ap_checks,
        "validation_input_policy": {"all baseline models": str(VAL_X_SCALED.relative_to(ROOT))},
        "validation_input_policy_reason": "Scaled validation features reproduce the existing Phase 4.4 AP scores for all three current baseline artifacts.",
        "validation_fraud_count": int(y_val.sum()),
        "models": list(MODELS.keys()),
        "permutation_repeats": N_REPEATS,
        "random_state": SEED,
        "statistical_tests": "Two-sided Mann–Whitney U; Benjamini–Hochberg FDR correction; Spearman correlation with target.",
        "effect_size": "Rank-biserial correlation derived from Mann–Whitney U.",
        "limitations": [
            "Statistical associations are not causal relationships.",
            "Anonymized V-features have no documented real-world interpretation.",
            "Permutation importance is evaluated on the validation split and can be affected by correlated features.",
            "Only three permutation repeats are used for an initial analysis; increase repeats if runtime allows and report this choice.",
            "No test-set data or labels were loaded by this script."
        ]
    }
    (OUT / "rq4_metadata.json").write_text(json.dumps(metadata, indent=2))
    readme = """RQ4 Feature Association Outputs\n\n- training_statistical_feature_associations.csv: training-only medians, Mann–Whitney U tests, FDR-adjusted p-values, rank-biserial effect sizes, and Spearman associations.\n- native_feature_importance.csv: existing baseline model-native importance / absolute coefficients.\n- validation_permutation_importance.csv: AP decrease after feature permutation on validation data.\n- top_permutation_features_by_model.csv: top ranked features for each baseline model.\n- permutation_importance_rankings_across_models.csv: rank comparison across models.\n- figures/: statistical association and per-model permutation-importance plots.\n- rq4_metadata.json: provenance and limitations.\n\nNo model was trained or tuned by this script. The final test set was not loaded.\nInterpret anonymized features statistically; do not assign undocumented real-world meanings.\n"""
    (OUT / "README.txt").write_text(readme)
    print(f"RQ4 analysis complete. Outputs: {OUT}")
    print("Top training statistical associations:")
    print(assoc[["feature", "abs_rank_biserial_correlation", "mann_whitney_p_fdr_bh"]].head(10).to_string(index=False))
    print("Baseline validation AP check by model:")
    for model_name, score in baseline_ap_checks.items():
        print(f"  {model_name}: {score:.6f}")
    print("Top validation permutation features by model:")
    print(perm.sort_values(["model", "rank"])[["model", "feature", "rank", "mean_ap_decrease_after_permutation"]].groupby("model").head(5).to_string(index=False))

if __name__ == "__main__":
    main()

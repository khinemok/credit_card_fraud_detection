from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

root = Path("/Users/sarahmay/credit_card_fraud_detection_ulb")
fig_dir = root / "outputs/eda/figures"
df = pd.read_csv(root / "data/raw/creditcard_ulb.csv")

BLUE, RED = "#3B6FB6", "#B8392B"

# 1) Box plots: 2x2 grid for the four features that matter most
feats = ["V14", "V12", "V10", "V4"]
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, f in zip(axes.ravel(), feats):
    data = [df.loc[df["Class"] == 0, f], df.loc[df["Class"] == 1, f]]
    bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.55,
                    medianprops=dict(color="black", linewidth=2))
    for patch, c in zip(bp["boxes"], [BLUE, RED]):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Legitimate", "Fraud"], fontsize=13)
    ax.set_title(f, fontsize=16, fontweight="bold")
    ax.tick_params(axis="y", labelsize=12)
    ax.axhline(0, color="grey", linewidth=0.6, linestyle=":")
fig.suptitle("Fraud shifts and widens the key features (outliers hidden)", fontsize=15)
fig.tight_layout()
fig.savefig(fig_dir / "boxplots_key_features.png", dpi=200)
plt.close(fig)

# 2) Heatmap: V1-V28, Amount and Class (Time dropped: not a model predictor)
cols = [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
corr = df[cols].corr()
fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(cols)))
ax.set_yticks(range(len(cols)))
ax.set_xticklabels(cols, rotation=90, fontsize=9)
ax.set_yticklabels(cols, fontsize=9)
ax.set_title("Correlation heatmap (V features are PCA components)", fontsize=14)
fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson correlation")
fig.tight_layout()
fig.savefig(fig_dir / "correlation_heatmap_clean.png", dpi=200)
plt.close(fig)

# 3) The informative view: correlation with Class, sorted
c = corr["Class"].drop("Class").sort_values(key=abs, ascending=False).head(10)
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.barh(c.index[::-1], c.values[::-1],
        color=[RED if v < 0 else BLUE for v in c.values[::-1]])
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Pearson correlation with Class", fontsize=12)
ax.set_title("Top 10 features by correlation with fraud", fontsize=14)
fig.tight_layout()
fig.savefig(fig_dir / "correlation_with_class.png", dpi=200)
plt.close(fig)

print("Saved 3 figures to", fig_dir)
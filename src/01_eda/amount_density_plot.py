from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

root = Path("/Users/sarahmay/credit_card_fraud_detection_ulb")
df = pd.read_csv(root / "data/raw/creditcard_ulb.csv", usecols=["Amount", "Class"])
df["log_amount"] = np.log1p(df["Amount"])

legit = df.loc[df["Class"] == 0]
fraud = df.loc[df["Class"] == 1]

bins = np.linspace(0, df["log_amount"].max(), 50)

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.hist(legit["log_amount"], bins=bins, density=True, alpha=0.55,
        color="#3B6FB6", label=f"Legitimate (n={len(legit):,})")
ax.hist(fraud["log_amount"], bins=bins, density=True, alpha=0.55,
        color="#B8392B", label=f"Fraud (n={len(fraud):,})")

for sub, color in [(legit, "#3B6FB6"), (fraud, "#B8392B")]:
    med = sub["Amount"].median()
    ax.axvline(np.log1p(med), color=color, linestyle="--", linewidth=1.5)
    ax.text(np.log1p(med) + 0.05, ax.get_ylim()[1] * 0.92,
            f"median ${med:,.2f}", color=color, fontsize=10)

ax.set_title("Transaction amount by class (each class normalised)")
ax.set_xlabel("log(1 + Amount)")
ax.set_ylabel("Density")
ax.legend()
fig.tight_layout()

out = root / "outputs/eda/figures/amount_density_by_class.png"
fig.savefig(out, dpi=200)
print(f"Saved: {out}")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

root = Path("/Users/sarahmay/credit_card_fraud_detection_ulb")
df = pd.read_csv(root / "data/raw/creditcard_ulb.csv", usecols=["Time", "Class"])
df["hour"] = df["Time"] / 3600

edges = np.arange(0, 50, 2)                      # 2-hour bins over 48 hours
df["bin"] = pd.cut(df["hour"], edges, right=False, include_lowest=True)
g = df.groupby("bin", observed=True)["Class"].agg(n="size", k="sum").reset_index()
g["mid"] = [iv.mid for iv in g["bin"]]
g["rate"] = g["k"] / g["n"] * 100

z = 1.96                                          # Wilson 95% interval
p = g["k"] / g["n"]
den = 1 + z**2 / g["n"]
ctr = (p + z**2 / (2 * g["n"])) / den
half = z * np.sqrt(p * (1 - p) / g["n"] + z**2 / (4 * g["n"]**2)) / den
g["lo"] = (ctr - half) * 100
g["hi"] = (ctr + half) * 100

overall = df["Class"].mean() * 100
b1, b2 = 120396 / 3600, 145248 / 3600             # train/val, val/test cuts

BLUE, RED, GREY = "#3B6FB6", "#B8392B", "#8C93A1"
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True,
                               gridspec_kw={"height_ratios": [2.2, 1]})
for ax in (ax1, ax2):
    ax.axvspan(0, b1, color="#E8EEF7", alpha=0.7, lw=0)
    ax.axvspan(b1, b2, color="#F7EBD8", alpha=0.7, lw=0)
    ax.axvspan(b2, 48, color="#E3F0E8", alpha=0.7, lw=0)
    ax.axvline(b1, color=GREY, lw=1, ls="--")
    ax.axvline(b2, color=GREY, lw=1, ls="--")

ax1.errorbar(g["mid"], g["rate"],
             yerr=[g["rate"] - g["lo"], g["hi"] - g["rate"]],
             fmt="o", color=BLUE, ecolor=BLUE, capsize=3, ms=6)
ax1.axhline(overall, color=RED, ls=":", lw=1.8)
ax1.text(47.8, overall + 0.03, f"overall {overall:.2f}%", color=RED,
         ha="right", fontsize=12)
ax1.set_ylabel("Fraud rate (%)", fontsize=13)
ax1.set_title("Fraud rate by 2-hour bin, with 95% intervals", fontsize=15)
ax1.set_ylim(bottom=0)
for x, lab in [(b1 / 2, "Train"), ((b1 + b2) / 2, "Validation"), ((b2 + 48) / 2, "Test")]:
    ax1.text(x, ax1.get_ylim()[1] * 0.96, lab, ha="center", va="top",
             fontsize=13, fontweight="bold", color="#1F2D4A")

ax2.bar(g["mid"], g["k"], width=1.7, color=RED, alpha=0.8)
ax2.set_ylabel("Frauds per bin", fontsize=13)
ax2.set_xlabel("Hours since first transaction", fontsize=13)
ax2.set_xlim(0, 48)
for ax in (ax1, ax2):
    ax.tick_params(labelsize=12)

fig.tight_layout()
out = root / "outputs/eda/figures/fraud_rate_over_time.png"
fig.savefig(out, dpi=200)
print("Saved:", out)
print(g[["mid", "n", "k", "rate", "lo", "hi"]].round(3).to_string(index=False))
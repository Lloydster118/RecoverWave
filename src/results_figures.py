"""
results_figures.py — generate Analysis-section figures from CV outputs.

Produces:
    fig_16_model_comparison.png   bar chart of MAE per model with fold error bars
    fig_17_predictions_vs_actual.png  LSTM predictions overlaid on actuals (Fold 4)
    fig_18_residuals.png          residual distributions per model
    fig_19_fold_mae.png           per-fold MAE for each model (line plot)
    fig_20_pred_scatter.png       predicted vs actual scatter (LSTM and GBM)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data/processed"
FIG = PROC / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.titleweight": "bold",
})

PALETTE = {
    "Persistence": "#999999",
    "SeasonalDoW": "#cc8c4d",
    "Ridge": "#4c72b0",
    "GBM": "#55a868",
    "LSTM": "#c44e52",
}

MODEL_ORDER = ["Persistence", "SeasonalDoW", "Ridge", "GBM", "LSTM"]

results = pd.read_csv(PROC / "cv_results.csv")
preds = pd.read_csv(PROC / "cv_predictions.csv", parse_dates=["date"])
summary = pd.read_csv(PROC / "cv_summary.csv")


# ─────────────────────────────────────────────────────────────────────
# Fig 16 — model comparison (MAE)
# ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
agg = results.groupby("model").agg(mean=("mae", "mean"), std=("mae", "std")).loc[MODEL_ORDER]
xpos = np.arange(len(agg))
bars = ax.bar(xpos, agg["mean"], yerr=agg["std"], capsize=6,
              color=[PALETTE[m] for m in agg.index],
              edgecolor="black", linewidth=0.8)
for x, v in zip(xpos, agg["mean"]):
    ax.text(x, v + 0.4, f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xticks(xpos)
ax.set_xticklabels(agg.index, rotation=15)
ax.set_ylabel("Mean Absolute Error  (recovery score points)")
ax.set_title("Figure 16. Cross-validated MAE by model (5 chronological folds)")
ax.set_ylim(0, max(agg["mean"] + agg["std"]) * 1.15)
plt.tight_layout()
plt.savefig(FIG / "fig_16_model_comparison.png", bbox_inches="tight")
plt.close()


# ─────────────────────────────────────────────────────────────────────
# Fig 17 — predictions vs actual (Fold 4, the most recent test window)
# ─────────────────────────────────────────────────────────────────────
fold4 = preds[preds.fold == 4].sort_values("date")
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(fold4["date"], fold4["actual"], color="black", lw=1.5,
        label="Actual recovery", marker="o", ms=3, alpha=0.85)
ax.plot(fold4["date"], fold4["LSTM"], color=PALETTE["LSTM"], lw=1.5,
        label="LSTM", alpha=0.9)
ax.plot(fold4["date"], fold4["GBM"], color=PALETTE["GBM"], lw=1.0,
        label="GBM", alpha=0.7, ls="--")
ax.plot(fold4["date"], fold4["SeasonalDoW"], color=PALETTE["SeasonalDoW"],
        lw=1.0, label="Seasonal DoW", alpha=0.7, ls=":")
ax.set_ylabel("Next-day recovery score (%)")
ax.set_xlabel("Date")
ax.set_ylim(0, 100)
ax.set_title("Figure 17. Predictions vs actual recovery — most recent test fold "
             f"({fold4.date.min().date()} – {fold4.date.max().date()})")
ax.legend(loc="upper right", framealpha=0.9, ncol=4)
plt.tight_layout()
plt.savefig(FIG / "fig_17_predictions_vs_actual.png", bbox_inches="tight")
plt.close()


# ─────────────────────────────────────────────────────────────────────
# Fig 18 — residual distributions
# ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for m in MODEL_ORDER:
    if m not in preds.columns:
        continue
    resid = preds[m] - preds["actual"]
    ax.hist(resid, bins=40, alpha=0.45, color=PALETTE[m], label=m,
            edgecolor="black", linewidth=0.4, density=True)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Residual  (predicted − actual)")
ax.set_ylabel("Density")
ax.set_title("Figure 18. Residual distributions across all CV folds")
ax.legend(framealpha=0.9)
plt.tight_layout()
plt.savefig(FIG / "fig_18_residuals.png", bbox_inches="tight")
plt.close()


# ─────────────────────────────────────────────────────────────────────
# Fig 19 — per-fold MAE
# ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for m in MODEL_ORDER:
    sub = results[results.model == m].sort_values("fold")
    ax.plot(sub["fold"], sub["mae"], "-o", color=PALETTE[m], label=m, lw=1.5, ms=7)
ax.set_xticks(sorted(results.fold.unique()))
ax.set_xlabel("Cross-validation fold (chronological)")
ax.set_ylabel("MAE")
ax.set_title("Figure 19. MAE by fold — performance over time")
ax.legend(framealpha=0.9)
plt.tight_layout()
plt.savefig(FIG / "fig_19_fold_mae.png", bbox_inches="tight")
plt.close()


# ─────────────────────────────────────────────────────────────────────
# Fig 20 — predicted vs actual scatter (LSTM + GBM side by side)
# ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
for ax, m in zip(axes, ["LSTM", "GBM"]):
    ax.scatter(preds["actual"], preds[m], s=14, alpha=0.45,
               color=PALETTE[m], edgecolor="black", linewidth=0.2)
    ax.plot([0, 100], [0, 100], color="black", lw=1, ls="--")
    r = np.corrcoef(preds["actual"], preds[m])[0, 1]
    mae = np.mean(np.abs(preds[m] - preds["actual"]))
    ax.set_title(f"{m}    r = {r:.2f}    MAE = {mae:.2f}")
    ax.set_xlabel("Actual recovery (%)")
    ax.set_ylabel("Predicted recovery (%)")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
fig.suptitle("Figure 20. Predicted vs actual recovery scores (all CV folds)",
             fontweight="bold")
plt.tight_layout()
plt.savefig(FIG / "fig_20_pred_scatter.png", bbox_inches="tight")
plt.close()


print("Saved figures:")
for p in sorted(FIG.glob("fig_1[6-9]*.png")) + sorted(FIG.glob("fig_20*.png")):
    print(" ", p.name)

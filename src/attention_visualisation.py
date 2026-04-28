"""attention_visualisation.py

Loads the saved attention-LSTM artefacts, runs a forward pass on the test
set, captures per-step attention weights, and produces:
    fig_21_attention_weights.png   mean attention by sequence-position day
    fig_22_attention_heatmap.png   per-sample attention heatmap
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from models import (DEVICE, FEATURES, RecoveryLSTM, SEQ_LEN, TARGET,
                    build_sequences, load_modelling_data)

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artefacts"
FIG = ROOT / "data/processed/figures"

with open(ART / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open(ART / "feature_columns.json") as f:
    feat_cols = json.load(f)
cfg = json.load(open(ART / "lstm_config.json"))

df = load_modelling_data()
X = df[feat_cols].to_numpy(dtype="float32")
y = df[TARGET].to_numpy(dtype="float32")
Xs = scaler.transform(X)
X_seq, y_seq = build_sequences(Xs, y)

model = RecoveryLSTM(n_features=cfg["n_features"], hidden=cfg["hidden"],
                     num_layers=cfg["num_layers"], dropout=cfg["dropout"])
model.load_state_dict(torch.load(ART / "lstm.pt", map_location=DEVICE))
model.to(DEVICE).eval()

# Use the most recent 159 test sequences (Fold 4 window) for visualisation
N = 159
X_test = torch.tensor(X_seq[-N:]).to(DEVICE)
with torch.no_grad():
    _, weights = model(X_test, return_weights=True)
W = weights.cpu().numpy()                      # (N, SEQ_LEN)

# ── fig 21: average attention over the 7-day window ──────────────────
mean_w = W.mean(axis=0); std_w = W.std(axis=0)
day_offsets = np.arange(-SEQ_LEN, 0) + 1       # -6 ... 0  (today = 0)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.bar(day_offsets, mean_w, yerr=std_w, capsize=5,
       color="#c44e52", edgecolor="black", linewidth=0.8, alpha=0.85)
ax.set_xticks(day_offsets)
ax.set_xticklabels([f"t−{abs(d)}" if d < 0 else "t" for d in day_offsets])
ax.set_xlabel("Sequence position relative to prediction day t+1")
ax.set_ylabel("Mean attention weight")
ax.set_title("Figure 21. Mean attention weight by lag (test window N = 159)",
             fontweight="bold")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / "fig_21_attention_weights.png", dpi=300, bbox_inches="tight")
plt.close()

# ── fig 22: heatmap of attention across samples ──────────────────────
fig, ax = plt.subplots(figsize=(11, 4.5))
im = ax.imshow(W.T, aspect="auto", cmap="magma",
               vmin=0, vmax=W.max())
ax.set_yticks(np.arange(SEQ_LEN))
ax.set_yticklabels([f"t−{SEQ_LEN-1-i}" if SEQ_LEN-1-i > 0 else "t" for i in range(SEQ_LEN)])
ax.set_xlabel("Test sample index (chronological)")
ax.set_ylabel("Day in 7-day input window")
ax.set_title("Figure 22. Per-sample attention weights across the test window",
             fontweight="bold")
fig.colorbar(im, ax=ax, label="Attention weight")
plt.tight_layout()
plt.savefig(FIG / "fig_22_attention_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()

print("Attention summary across test window:")
for d, m, s in zip(day_offsets, mean_w, std_w):
    print(f"  t{d:+d}:  mean={m:.3f}   std={s:.3f}")
print(f"\nSaved fig_21_attention_weights.png  fig_22_attention_heatmap.png")

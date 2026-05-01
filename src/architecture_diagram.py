"""architecture_diagram.py

Publication-quality system architecture diagram for the technical-depth
appendix.  Pure matplotlib (no Graphviz dependency).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "data/processed/figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

fig, ax = plt.subplots(figsize=(13, 8.5))
ax.set_xlim(0, 100); ax.set_ylim(0, 100)
ax.set_aspect("equal"); ax.axis("off")

# ── colour palette ───────────────────────────────────────────────────
C_DATA = "#dbe9f4";  E_DATA = "#3a6e9c"
C_PREP = "#fce8d4";  E_PREP = "#cc8c4d"
C_FEAT = "#dff0d8";  E_FEAT = "#55a868"
C_MOD  = "#f7d6d6";  E_MOD  = "#c44e52"
C_OUT  = "#e6dcf2";  E_OUT  = "#7a5cad"

def box(x, y, w, h, label, fc, ec, fontsize=9, bold=True):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                       linewidth=1.4, facecolor=fc, edgecolor=ec)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold" if bold else "normal", color="black")

def arrow(x1, y1, x2, y2, color="black"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", lw=1.3, color=color,
                                shrinkA=2, shrinkB=2))

# ── Layer 1  — raw data sources (left column) ────────────────────────
box(2, 78, 22, 9, "Whoop CSV exports\n(physiological_cycles,\nworkouts, sleeps, recovery)", C_DATA, E_DATA)
box(2, 65, 22, 9, "Spotify Extended\nStreaming History\n(JSON · 8.4 yr · 274 678 ev.)", C_DATA, E_DATA)
box(2, 52, 22, 9, "ReccoBeats public API\n(audio features:\ntempo · valence · energy)", C_DATA, E_DATA)

# ── Layer 2  — parsing / alignment ───────────────────────────────────
box(30, 78, 22, 9, "whoop_parser.py\n• CSV \u2192 typed schema\n• timezone fixes", C_PREP, E_PREP)
box(30, 65, 22, 9, "spotify_parser.py\n• 30s listen filter\n• Europe/London \u2192 UTC", C_PREP, E_PREP)
box(30, 52, 22, 9, "audio_features.py\n• ID lookup (cache)\n• retry + back-off", C_PREP, E_PREP)

# ── Layer 3  — feature engineering ───────────────────────────────────
box(58, 78, 22, 9, "alignment.py\n+2 h post-workout\nlistening window\n(74 % hit rate)", C_FEAT, E_FEAT)
box(58, 65, 22, 9, "post_workout_features.py\nN-tracks, tempo, valence,\nenergy, mood diversity", C_FEAT, E_FEAT)
box(58, 52, 22, 9, "sleep_features.py\nbedtime delta · onset lag ·\nduration · architecture", C_FEAT, E_FEAT)
box(58, 39, 22, 9, "build_modelling_dataset.py\n954 daily rows  \u00b7  36 features\n+ next-day recovery target", C_FEAT, E_FEAT)

# ── Layer 4  — model ─────────────────────────────────────────────────
box(58, 22, 22, 12,
    "RecoveryLSTM (PyTorch)\n2-layer LSTM (h=64)\n\u2192 additive attention\n\u2192 MLP head \u2192 \u0177 \u2208 [0,100]",
    C_MOD, E_MOD)

# ── Layer 5  — evaluation + deployment ───────────────────────────────
box(85, 78, 13, 9, "Blocked\n5-fold CV\n(no leakage)", C_OUT, E_OUT)
box(85, 60, 13, 9, "MAE / RMSE / R\u00b2\nresults tables\n+ figures", C_OUT, E_OUT)
box(85, 41, 13, 9, "Persisted artefacts:\nlstm.pt · gbm.pkl ·\nscaler.pkl", C_OUT, E_OUT)
box(85, 22, 13, 12, "Streamlit\ndashboard\n(viva demo,\nlive day)", C_OUT, E_OUT)

# ── Arrows ───────────────────────────────────────────────────────────
arrow(24, 82.5, 30, 82.5);  arrow(24, 69.5, 30, 69.5);  arrow(24, 56.5, 30, 56.5)
arrow(52, 82.5, 58, 82.5);  arrow(52, 69.5, 58, 69.5);  arrow(52, 56.5, 58, 56.5)

# Feature engineering converges into modelling dataset
arrow(69, 78, 69, 48)        # post-workout to dataset
arrow(69, 65, 69, 48)        # alignment also (visual collapse)
arrow(69, 52, 69, 48)        # sleep to dataset

# Modelling dataset feeds LSTM
arrow(69, 39, 69, 34)

# Model feeds evaluation + dashboard
arrow(80, 28, 85, 28)        # to dashboard
arrow(80, 30, 85, 45)        # to artefacts
arrow(80, 32, 85, 64)        # to results
arrow(80, 33, 85, 82)        # to CV

# ── Lane labels (above each column) ──────────────────────────────────
for x, lab in [(13, "Sources"), (41, "Parsing"),
               (69, "Feature Engineering"),
               (69, "Modelling"),
               (91.5, "Evaluation\n+ Deployment")]:
    pass  # we'll do them individually below

ax.text(13, 92, "Sources",            ha="center", fontsize=11, fontweight="bold", color=E_DATA)
ax.text(41, 92, "Parsing",            ha="center", fontsize=11, fontweight="bold", color=E_PREP)
ax.text(69, 92, "Feature engineering",ha="center", fontsize=11, fontweight="bold", color=E_FEAT)
ax.text(69, 18, "Modelling",          ha="center", fontsize=11, fontweight="bold", color=E_MOD)
ax.text(91.5, 92, "Evaluation\n+ Deployment", ha="center", fontsize=11, fontweight="bold", color=E_OUT)

# Title
ax.text(50, 98, "RecoverWave system architecture",
        ha="center", fontsize=14, fontweight="bold")
ax.text(50, 95, "Whoop CSV  +  Spotify JSON  \u2192  feature pipeline  \u2192  attention LSTM  \u2192  Streamlit",
        ha="center", fontsize=10, style="italic", color="#444")

# Bottom legend caption
ax.text(50, 5, "Figure A1.  End-to-end data flow.  Layers run left-to-right; "
               "each box maps to one Python module under recoverwave/src/.",
        ha="center", fontsize=9, style="italic", color="#444")

plt.tight_layout()
plt.savefig(FIG / "fig_A1_architecture.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved fig_A1_architecture.png")

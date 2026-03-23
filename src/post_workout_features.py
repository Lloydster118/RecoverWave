"""
post_workout_features.py — Aggregate track-level audio features into
per-workout summaries and run the first correlation pass against
next-day recovery.

For each workout with a post-workout listening window (2h after workout end):
  - mean, median, std of tempo / valence / energy / danceability / loudness / acousticness
  - mood-diversity metric: std of [valence, energy, danceability] per window
  - listening intensity: n_tracks, total_listen_minutes, unique_artists
  - tempo/energy drift: linear slope across tracks within the window
  - feature coverage: % of tracks in window with features available

Output:
  data/processed/workout_features.csv  — one row per workout
  data/processed/correlations.csv     — Pearson r vs next_day_recovery
  figures/fig_11_feature_hist.png
  figures/fig_12_feature_vs_recovery.png
  figures/fig_13_correlation_heatmap.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spotify_parser import SpotifyParser  # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────
WINDOW_HOURS = 2.0
MIN_TRACKS_IN_WINDOW = 3  # drop windows too small for meaningful aggregation

NUMERIC_FEATURES = ("tempo", "valence", "energy", "danceability",
                    "loudness", "acousticness", "instrumentalness",
                    "speechiness", "liveness")
MOOD_DIVERSITY_FIELDS = ("valence", "energy", "danceability")

# Plot styling
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.family": "DejaVu Sans",
    "font.size": 10, "axes.edgecolor": "#28251D", "axes.labelcolor": "#28251D",
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})
ACCENT = "#01696F"
ACCENT2 = "#A84B2F"
MUTED = "#7A7974"


# ─────────────────────────────────────────────────────────────────────
# 1. Load everything
# ─────────────────────────────────────────────────────────────────────


def load_data():
    parser = SpotifyParser(str(ROOT / "data/spotify/Spotify Extended Streaming History"))
    parser.load(); parser.clean()

    feat = pd.read_csv(ROOT / "data/processed/audio_features.csv")
    feat = feat.rename(columns={feat.columns[0]: "track_id"})
    feat = feat.set_index("track_id")

    wk = pd.read_csv(ROOT / "data/processed/whoop_workouts_clean.csv")
    wk["end_utc"] = (pd.to_datetime(wk["workout_end_time"], errors="coerce")
                       .dt.tz_localize("Europe/London",
                                       ambiguous="NaT", nonexistent="NaT")
                       .dt.tz_convert("UTC"))
    wk = wk.dropna(subset=["end_utc"]).reset_index(drop=True)
    return parser, feat, wk


def _slope(values: np.ndarray) -> float:
    """Linear slope across a sequence (per-track index)."""
    if len(values) < 2 or np.all(np.isnan(values)):
        return np.nan
    x = np.arange(len(values))
    y = values.astype(float)
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return np.nan
    slope, _, _, _, _ = stats.linregress(x[mask], y[mask])
    return float(slope)


if __name__ == "__main__":
    main()

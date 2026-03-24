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


def aggregate_window(window_tracks: pd.DataFrame, feat: pd.DataFrame) -> dict:
    """Join window tracks to features, compute aggregates."""
    joined = window_tracks.merge(feat, left_on="track_id", right_index=True, how="left")
    n_total = len(window_tracks)
    n_with_feat = joined[NUMERIC_FEATURES[0]].notna().sum()

    agg = {
        "n_tracks": n_total,
        "n_tracks_with_features": int(n_with_feat),
        "feature_coverage": float(n_with_feat / n_total) if n_total else 0.0,
        "total_listen_minutes": float(window_tracks["listen_seconds"].sum() / 60),
        "unique_artists": int(window_tracks["artist_name"].nunique()),
        "mean_track_len_min": float(window_tracks["listen_seconds"].mean() / 60),
        "artist_concentration": (  # HHI-like: 1 means all same artist
            float((window_tracks["artist_name"].value_counts(normalize=True) ** 2).sum())
            if n_total else np.nan
        ),
    }

    if n_with_feat == 0:
        for f in NUMERIC_FEATURES:
            agg[f"{f}_mean"] = np.nan
            agg[f"{f}_median"] = np.nan
            agg[f"{f}_std"] = np.nan
        agg["mood_diversity"] = np.nan
        agg["tempo_slope"] = np.nan
        agg["energy_slope"] = np.nan
        agg["valence_slope"] = np.nan
        return agg

    for f in NUMERIC_FEATURES:
        vals = joined[f].dropna().to_numpy()
        agg[f"{f}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
        agg[f"{f}_median"] = float(np.median(vals)) if len(vals) else np.nan
        agg[f"{f}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    # Mood diversity: average std across normalised mood dims
    mood_stds = []
    for f in MOOD_DIVERSITY_FIELDS:
        vals = joined[f].dropna().to_numpy()
        if len(vals) > 1:
            mood_stds.append(np.std(vals, ddof=1))
    agg["mood_diversity"] = float(np.mean(mood_stds)) if mood_stds else np.nan

    # Trajectory slopes (track-by-track)
    series = joined.sort_values("start_time")
    agg["tempo_slope"] = _slope(series["tempo"].to_numpy())
    agg["energy_slope"] = _slope(series["energy"].to_numpy())
    agg["valence_slope"] = _slope(series["valence"].to_numpy())

    return agg


if __name__ == "__main__":
    main()

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


# ─────────────────────────────────────────────────────────────────────
# 2. Per-window aggregation
# ─────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────
# 3. Build workout-level DataFrame
# ─────────────────────────────────────────────────────────────────────
def build_workout_features(parser, feat, wk) -> pd.DataFrame:
    rows = []
    for _, w in wk.iterrows():
        win = parser.get_listening_window(w["end_utc"], window_hours=WINDOW_HOURS)
        if len(win) == 0:
            continue
        agg = aggregate_window(win, feat)
        agg.update({
            "workout_end_utc": w["end_utc"],
            "workout_end_local": w["workout_end_time"],
            "activity": w.get("activity_name"),
            "activity_strain": w.get("activity_strain"),
            "duration": w.get("duration"),
            "next_day_recovery": w.get("next_day_recovery"),
            "day_strain": w.get("strain_score"),
        })
        rows.append(agg)
    df = pd.DataFrame(rows)
    print(f"Workouts with ≥1 post-track: {len(df)}")
    df = df[df["n_tracks"] >= MIN_TRACKS_IN_WINDOW].reset_index(drop=True)
    print(f"After filter ≥{MIN_TRACKS_IN_WINDOW} tracks: {len(df)}")
    trainable = df.dropna(subset=["next_day_recovery"])
    print(f"With next-day recovery (trainable): {len(trainable)}")
    return df


# ─────────────────────────────────────────────────────────────────────
# 4. Correlation pass
# ─────────────────────────────────────────────────────────────────────
def correlate(df: pd.DataFrame, target: str = "next_day_recovery") -> pd.DataFrame:
    trainable = df.dropna(subset=[target]).copy()
    candidates = [c for c in trainable.columns
                  if c not in {target, "workout_end_utc", "workout_end_local",
                               "activity"}
                  and trainable[c].dtype != "O"]
    results = []
    for c in candidates:
        pair = trainable[[c, target]].dropna()
        if len(pair) < 20:
            continue
        r, p = stats.pearsonr(pair[c], pair[target])
        rho, p_s = stats.spearmanr(pair[c], pair[target])
        results.append({
            "feature": c, "n": len(pair),
            "pearson_r": r, "pearson_p": p,
            "spearman_r": rho, "spearman_p": p_s,
        })
    res = pd.DataFrame(results).sort_values("pearson_r", key=abs, ascending=False)
    return res


# ─────────────────────────────────────────────────────────────────────
# 5. Figures
# ─────────────────────────────────────────────────────────────────────
def plot_feature_distributions(df, out_path):
    feats = ["tempo_mean", "valence_mean", "energy_mean",
             "danceability_mean", "loudness_mean", "mood_diversity"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for ax, f in zip(axes.flat, feats):
        vals = df[f].dropna()
        ax.hist(vals, bins=30, color=ACCENT, alpha=0.85, edgecolor="white")
        ax.axvline(vals.median(), color=ACCENT2, ls="--",
                   label=f"median = {vals.median():.2f}")
        ax.set_title(f.replace("_", " "))
        ax.legend(fontsize=8)
    fig.suptitle("Post-workout listening features — distribution across workouts",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path); plt.close()


def plot_feature_vs_recovery(df, out_path):
    trainable = df.dropna(subset=["next_day_recovery"])
    feats = [("tempo_mean", "Mean tempo (BPM)"),
             ("valence_mean", "Mean valence"),
             ("energy_mean", "Mean energy"),
             ("danceability_mean", "Mean danceability"),
             ("mood_diversity", "Mood diversity"),
             ("total_listen_minutes", "Total listen (min)")]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, (f, label) in zip(axes.flat, feats):
        pair = trainable[[f, "next_day_recovery"]].dropna()
        ax.scatter(pair[f], pair["next_day_recovery"], s=12, alpha=0.35, color=ACCENT)
        if len(pair) >= 20:
            r, p = stats.pearsonr(pair[f], pair["next_day_recovery"])
            slope, intercept, _, _, _ = stats.linregress(pair[f], pair["next_day_recovery"])
            xs = np.linspace(pair[f].min(), pair[f].max(), 50)
            ax.plot(xs, intercept + slope * xs, color=ACCENT2, lw=1.5)
            ax.set_title(f"{label}\nr = {r:+.3f}, p = {p:.3f}, n = {len(pair)}",
                         fontsize=10)
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("Next-day recovery %", fontsize=9)
    fig.suptitle("Post-workout listening features vs. next-day recovery",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path); plt.close()


def plot_correlation_heatmap(res, out_path):
    top = res.head(20).copy().reset_index(drop=True)
    n = len(top)

    # Two-panel layout: bars on the left, clean annotation table on the right
    fig, (ax_bar, ax_tbl) = plt.subplots(
        1, 2, figsize=(12, 7),
        gridspec_kw={"width_ratios": [3, 1.3], "wspace": 0.05},
    )
    y = np.arange(n)[::-1]
    colors = [ACCENT if r >= 0 else ACCENT2 for r in top["pearson_r"]]
    ax_bar.barh(y, top["pearson_r"], color=colors, alpha=0.88)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(top["feature"].str.replace("_", " "), fontsize=10)
    ax_bar.axvline(0, color="#28251D", lw=0.6)
    ax_bar.set_xlabel("Pearson r (vs next-day recovery)")
    ax_bar.set_title("Top 20 post-workout listening features ranked by |r| with next-day recovery",
                     pad=12, loc="left")

    xmax = max(abs(top["pearson_r"].max()), abs(top["pearson_r"].min())) * 1.15
    ax_bar.set_xlim(-xmax, xmax)

    # Right panel: clean annotation table (r, p, n)
    ax_tbl.set_xlim(0, 1)
    ax_tbl.set_ylim(ax_bar.get_ylim())
    ax_tbl.axis("off")
    ax_tbl.text(0.08, n - 0.5, "r", fontsize=10, fontweight="bold", color="#28251D")
    ax_tbl.text(0.45, n - 0.5, "p", fontsize=10, fontweight="bold", color="#28251D")
    ax_tbl.text(0.80, n - 0.5, "n", fontsize=10, fontweight="bold", color="#28251D")
    for i, row in top.iterrows():
        yi = n - 1 - i
        sig = "*" if row["pearson_p"] < 0.05 else " "
        ax_tbl.text(0.08, yi, f"{row['pearson_r']:+.3f}", fontsize=10,
                    va="center", color="#28251D",
                    fontweight="bold" if row["pearson_p"] < 0.05 else "normal")
        ax_tbl.text(0.45, yi, f"{row['pearson_p']:.3f}{sig}", fontsize=10,
                    va="center", color="#28251D")
        ax_tbl.text(0.80, yi, f"{int(row['n'])}", fontsize=10,
                    va="center", color=MUTED)

    fig.text(0.5, 0.02,
             "* p < 0.05    |    Teal = positive correlation, red = negative",
             ha="center", fontsize=9, color=MUTED)
    plt.savefig(out_path, bbox_inches="tight"); plt.close()


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    print("Loading data…")
    parser, feat, wk = load_data()
    print(f"  Audio features: {len(feat)} tracks")
    print(f"  Workouts: {len(wk)}")

    print("\nBuilding per-workout features…")
    df = build_workout_features(parser, feat, wk)

    out_csv = ROOT / "data/processed/workout_features.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved → {out_csv}")

    print("\nFeature coverage (mean per workout): "
          f"{df['feature_coverage'].mean():.2%}")

    print("\nRunning correlations vs next_day_recovery…")
    res = correlate(df)
    res_csv = ROOT / "data/processed/correlations.csv"
    res.to_csv(res_csv, index=False)
    print(f"Saved → {res_csv}")

    print("\n=== Top features by |Pearson r| vs next-day recovery ===")
    print(res.head(15).to_string(
        index=False,
        formatters={
            "pearson_r": "{:+.3f}".format, "pearson_p": "{:.3f}".format,
            "spearman_r": "{:+.3f}".format, "spearman_p": "{:.3f}".format,
        },
    ))

    print("\nGenerating figures…")
    FIGDIR = ROOT / "data/processed/figures"
    plot_feature_distributions(df, FIGDIR / "fig_11_feature_distributions.png")
    print("  fig_11_feature_distributions.png")
    plot_feature_vs_recovery(df, FIGDIR / "fig_12_feature_vs_recovery.png")
    print("  fig_12_feature_vs_recovery.png")
    plot_correlation_heatmap(res, FIGDIR / "fig_13_correlation_ranking.png")
    print("  fig_13_correlation_ranking.png")


if __name__ == "__main__":
    main()

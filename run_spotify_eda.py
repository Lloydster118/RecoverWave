"""Characterise the real Spotify extended streaming history and align with Whoop workouts."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from spotify_parser import SpotifyParser  # noqa: E402

SPOTIFY_DIR = ROOT / "data" / "spotify" / "Spotify Extended Streaming History"
WORKOUTS_CSV = ROOT / "data" / "processed" / "whoop_workouts_clean.csv"
OUT_FIG_DIR = ROOT / "data" / "processed" / "figures"
OUT_CSV_DIR = ROOT / "data" / "processed"
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Visual style to match existing fig_02..06 ────────────────────────────
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": "#28251D",
    "axes.labelcolor": "#28251D",
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": "#28251D",
    "ytick.color": "#28251D",
})
ACCENT = "#01696F"
ACCENT2 = "#A84B2F"
MUTED = "#7A7974"

# ── 1. Parse ──────────────────────────────────────────────────────────────
print("=" * 70)
print("REAL Spotify Extended Streaming History — EDA")
print("=" * 70)
parser = SpotifyParser(str(SPOTIFY_DIR))
raw = parser.load()
clean = parser.clean()
print()

print(f"Raw events        : {len(raw):,}")
print(f"After filter (≥30s): {len(clean):,} "
      f"({100 * len(clean) / len(raw):.1f}%)")
print(f"Date range        : {clean.date.min()} → {clean.date.max()}")
print(f"Years covered     : {(clean.end_time.max() - clean.end_time.min()).days / 365.25:.1f}")
print(f"Unique tracks     : {clean.track_id.nunique():,}")
print(f"Unique artists    : {clean.artist_name.nunique():,}")
print(f"Total listening   : {clean.listen_seconds.sum() / 3600:,.0f} hours "
      f"= {clean.listen_seconds.sum() / 3600 / 24:.0f} days")
print()

# Filter to Whoop era for alignment work
whoop_start = pd.Timestamp("2023-08-01", tz="UTC")
clean_era = clean[clean.end_time >= whoop_start].copy()
print(f"Events since Whoop started (Aug 2023): {len(clean_era):,}")
print()

# Save clean to CSV
clean_path = OUT_CSV_DIR / "spotify_clean.csv"
clean.to_csv(clean_path, index=False)
print(f"Saved clean Spotify → {clean_path}")

# ── 2. Load workouts, align ───────────────────────────────────────────────
wk = pd.read_csv(WORKOUTS_CSV)
end_col = next((c for c in wk.columns if "end" in c.lower() and
                ("time" in c.lower() or "ts" in c.lower())), None)
if end_col is None:
    # Try common names
    for cand in ("workout_end_ts", "end_time", "Workout end time"):
        if cand in wk.columns:
            end_col = cand
            break
print(f"Workout end column: {end_col}")
wk[end_col] = pd.to_datetime(wk[end_col], utc=True, errors="coerce")
wk = wk.dropna(subset=[end_col]).reset_index(drop=True)
print(f"Total workouts: {len(wk)}")

# For each workout, count tracks in 2h post-window
rows = []
for _, w in wk.iterrows():
    end_ts = w[end_col]
    win = parser.get_listening_window(end_ts, window_hours=2.0)
    rows.append({
        "workout_end": end_ts,
        "n_tracks_2h": len(win),
        "total_listen_minutes_2h": win.listen_seconds.sum() / 60 if len(win) else 0.0,
        "unique_artists_2h": win.artist_name.nunique() if len(win) else 0,
    })
align = pd.DataFrame(rows)
align_csv = OUT_CSV_DIR / "workout_listening_alignment.csv"
align.to_csv(align_csv, index=False)

with_tracks = (align.n_tracks_2h > 0).sum()
pct = 100 * with_tracks / len(align)
print(f"Workouts with ≥1 post-workout track (within 2h): {with_tracks}/{len(align)} ({pct:.1f}%)")
print(f"Median tracks per post-workout window: {int(align.n_tracks_2h.median())}")
print(f"Mean listen minutes per window: {align.total_listen_minutes_2h.mean():.1f}")

# ── 3. Figures ────────────────────────────────────────────────────────────
# Fig 07: listens per month over time
print("\nBuilding figures…")
monthly = clean.set_index("end_time").resample("MS").size()
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.fill_between(monthly.index, monthly.values, color=ACCENT, alpha=0.25)
ax.plot(monthly.index, monthly.values, color=ACCENT, lw=1.8)
ax.axvline(whoop_start, color=ACCENT2, ls="--", lw=1.2, alpha=0.8)
ax.text(whoop_start, ax.get_ylim()[1] * 0.92, "  Whoop start",
        color=ACCENT2, fontsize=9)
ax.set_title("Spotify listening volume by month, 2017–2026")
ax.set_ylabel("Tracks played per month")
ax.set_xlabel("")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.tight_layout()
p = OUT_FIG_DIR / "fig_07_listening_volume.png"
plt.savefig(p); plt.close()
print(f"  {p.name}")

# Fig 08: hour-of-day heatmap (DOW × hour)
era = clean_era.copy()
era["dow"] = era.end_time.dt.dayofweek  # 0 = Mon
era["hr"] = era.hour
grid = era.pivot_table(index="dow", columns="hr", values="track_id", aggfunc="count").fillna(0)
grid = grid.reindex(index=range(7), columns=range(24), fill_value=0)
fig, ax = plt.subplots(figsize=(10, 4))
im = ax.imshow(grid.values, aspect="auto", cmap="YlGnBu")
ax.set_yticks(range(7))
ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
ax.set_xticks(range(0, 24, 2))
ax.set_xticklabels([f"{h:02d}" for h in range(0, 24, 2)])
ax.set_title("Listening by weekday × hour (UTC), Whoop era")
ax.set_xlabel("Hour of day")
plt.colorbar(im, ax=ax, label="Tracks")
plt.tight_layout()
p = OUT_FIG_DIR / "fig_08_listening_heatmap.png"
plt.savefig(p); plt.close()
print(f"  {p.name}")

# Fig 09: post-workout track count distribution
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(align.n_tracks_2h, bins=range(0, int(align.n_tracks_2h.max()) + 2),
        color=ACCENT, alpha=0.85, edgecolor="white")
ax.axvline(align.n_tracks_2h.median(), color=ACCENT2, ls="--",
           label=f"Median = {int(align.n_tracks_2h.median())} tracks")
ax.set_title("Post-workout listening: tracks within 2 hours of workout end")
ax.set_xlabel("Tracks played in 2h window")
ax.set_ylabel("Workouts")
ax.legend()
plt.tight_layout()
p = OUT_FIG_DIR / "fig_09_post_workout_tracks.png"
plt.savefig(p); plt.close()
print(f"  {p.name}")

# Fig 10: top artists in Whoop era
top = era.artist_name.value_counts().head(15)
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(top.index[::-1], top.values[::-1], color=ACCENT)
ax.set_title("Top 15 artists by play count, Whoop era (Aug 2023 →)")
ax.set_xlabel("Plays")
plt.tight_layout()
p = OUT_FIG_DIR / "fig_10_top_artists.png"
plt.savefig(p); plt.close()
print(f"  {p.name}")

# ── 4. Summary to stdout ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("KEY NUMBERS FOR DISSERTATION")
print("=" * 70)
print(f"Spotify history spans        : {clean.date.min()} → {clean.date.max()}")
print(f"Total genuine listens        : {len(clean):,}")
print(f"Listens in Whoop era         : {len(clean_era):,}")
print(f"Unique tracks                : {clean.track_id.nunique():,}")
print(f"Unique artists               : {clean.artist_name.nunique():,}")
print(f"Total listening hours        : {clean.listen_seconds.sum() / 3600:,.0f}")
print(f"Workouts with post-listen    : {with_tracks} / {len(align)} ({pct:.1f}%)")
print(f"Median post-workout tracks   : {int(align.n_tracks_2h.median())}")
print(f"Mean post-workout minutes    : {align.total_listen_minutes_2h.mean():.1f}")

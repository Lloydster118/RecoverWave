"""
01_whoop_eda.py — Exploratory Data Analysis of Whoop exports.

Run this first once you've placed your Whoop CSVs in data/whoop/.
This notebook-style script will:
  1. Load and summarise all 4 CSV files
  2. Visualise recovery score distribution and trends
  3. Explore sleep architecture patterns
  4. Analyse workout strain patterns
  5. Check data quality and identify gaps

Can be run as a script or converted to a Jupyter notebook.

Usage:
  python notebooks/01_whoop_eda.py
"""

import sys
sys.path.insert(0, "..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.whoop_parser import WhoopParser

# ── Config ────────────────────────────────────────────────────────────

WHOOP_DIR = "../data/whoop"  # Adjust if needed
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

# ── Load Data ─────────────────────────────────────────────────────────

print("=" * 60)
print("RecoverWave — Whoop EDA")
print("=" * 60)

parser = WhoopParser(WHOOP_DIR)
data = parser.load_all()
timeline = parser.build_daily_timeline()

# ── 1. Data Summary ──────────────────────────────────────────────────

print("\n📊 Data Summary")
print("-" * 40)
for name, df in data.items():
    print(f"\n{name}:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Date range: {df.iloc[:, 0].min()} → {df.iloc[:, 0].max()}")
    print(f"  Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# ── 2. Recovery Score Analysis ───────────────────────────────────────

if "recovery_score" in timeline.columns:
    recovery = timeline["recovery_score"].dropna()

    print(f"\n🟢 Recovery Score Stats")
    print(f"  Mean: {recovery.mean():.1f}%")
    print(f"  Median: {recovery.median():.1f}%")
    print(f"  Std: {recovery.std():.1f}%")
    print(f"  Min: {recovery.min():.1f}%, Max: {recovery.max():.1f}%")
    print(f"  Days in green (≥67%): {(recovery >= 67).sum()} ({100*(recovery >= 67).mean():.0f}%)")
    print(f"  Days in yellow (34-66%): {((recovery >= 34) & (recovery < 67)).sum()}")
    print(f"  Days in red (<34%): {(recovery < 34).sum()}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Distribution
    axes[0].hist(recovery, bins=20, edgecolor="white", alpha=0.8)
    axes[0].axvline(recovery.mean(), color="red", linestyle="--", label=f"Mean: {recovery.mean():.0f}%")
    axes[0].set_xlabel("Recovery Score (%)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Recovery Score Distribution")
    axes[0].legend()

    # Time series
    if "date" in timeline.columns:
        axes[1].plot(timeline["date"], timeline["recovery_score"], alpha=0.7, linewidth=0.8)
        axes[1].axhline(67, color="green", linestyle="--", alpha=0.5, label="Green zone")
        axes[1].axhline(33, color="red", linestyle="--", alpha=0.5, label="Red zone")
        axes[1].set_xlabel("Date")
        axes[1].set_ylabel("Recovery Score (%)")
        axes[1].set_title("Recovery Over Time")
        axes[1].legend()

    plt.tight_layout()
    plt.savefig("../data/processed/recovery_eda.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved: data/processed/recovery_eda.png")

# ── 3. Workout Patterns ─────────────────────────────────────────────

workouts = data["workouts"]
if "strain_score" in workouts.columns:
    strain = workouts["strain_score"].dropna()

    print(f"\n💪 Workout Stats")
    print(f"  Total workouts: {len(workouts)}")
    print(f"  Mean strain: {strain.mean():.1f}")
    print(f"  Workouts per week: {len(workouts) / max((timeline['date'].max() - timeline['date'].min()).days / 7, 1):.1f}")

    if "workout_end_dt" in workouts.columns:
        workouts["hour_of_day"] = workouts["workout_end_dt"].dt.hour
        print(f"  Most common workout hour: {workouts['hour_of_day'].mode().iloc[0]}:00")

# ── 4. Sleep Architecture ───────────────────────────────────────────

sleeps = data["sleeps"]
sleep_cols = [c for c in sleeps.columns if "ratio" in c or "duration" in c or "efficiency" in c]
if sleep_cols:
    print(f"\n😴 Sleep Stats")
    for col in sleep_cols[:6]:
        values = sleeps[col].dropna()
        if len(values) > 0:
            print(f"  {col}: mean={values.mean():.2f}, std={values.std():.2f}")

# ── 5. Correlation Preview ───────────────────────────────────────────

print(f"\n🔗 Key Correlations with Recovery Score")
if "recovery_score" in timeline.columns:
    numeric_cols = timeline.select_dtypes(include=[np.number]).columns
    correlations = timeline[numeric_cols].corr()["recovery_score"].drop("recovery_score").sort_values()

    print("  Top positive correlations:")
    for feat, corr in correlations.tail(5).items():
        print(f"    {feat}: {corr:.3f}")

    print("  Top negative correlations:")
    for feat, corr in correlations.head(5).items():
        print(f"    {feat}: {corr:.3f}")

print("\n✅ EDA complete. Next step: run 02_spotify_eda.py when your Spotify data arrives.")

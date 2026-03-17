"""
sleep_features.py — Add post-workout sleep features to the workout-feature table
to test the indirect pathway: music → sleep timing/quality → next-day recovery.

For each workout we attach:
  - bedtime_local            (clock time of sleep onset, hours past noon — wraps cleanly)
  - bedtime_delta_personal   (offset vs the participant's median bedtime, hours)
  - sleep_onset_lag_h        (hours from workout end to sleep onset)
  - sleep_duration_h         (asleep duration of the post-workout night)
  - sleep_efficiency         (% from Whoop)
  - sleep_performance        (% from Whoop)
  - sleep_consistency        (% from Whoop)
  - rem_minutes / deep_minutes / light_minutes / awake_minutes
  - is_late_bedtime          (1 if bedtime later than personal 75th percentile)

Then re-runs correlations against next-day recovery.

Outputs:
  data/processed/workout_features_with_sleep.csv
  data/processed/correlations_with_sleep.csv
  figures/fig_14_sleep_correlations.png
  figures/fig_15_indirect_pathway.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]


# ─────────────────────────────────────────────────────────────────────
# Style (matches earlier figures)
# ─────────────────────────────────────────────────────────────────────
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
# 1. Load and clean the sleeps table
# ─────────────────────────────────────────────────────────────────────


def load_sleeps() -> pd.DataFrame:
    sl = pd.read_csv(ROOT / "data/whoop/sleeps.csv")
    # Drop naps — only main sleeps are relevant for recovery
    if "Nap" in sl.columns:
        sl = sl[sl["Nap"].astype(str).str.lower() != "true"].copy()

    # Parse timestamps — Whoop stores UTC ISO strings
    sl["sleep_onset"] = pd.to_datetime(sl["Sleep onset"], utc=True, errors="coerce")
    sl["wake_onset"] = pd.to_datetime(sl["Wake onset"], utc=True, errors="coerce")
    sl = sl.dropna(subset=["sleep_onset", "wake_onset"]).reset_index(drop=True)

    # Convert to local clock for bedtime feature
    sl["sleep_onset_local"] = sl["sleep_onset"].dt.tz_convert("Europe/London")
    # Express bedtime as decimal hours past midday (so 23:00 -> 11.0,
    # 02:00 -> 14.0). Wraps cleanly across midnight.
    h = sl["sleep_onset_local"].dt.hour + sl["sleep_onset_local"].dt.minute / 60
    sl["bedtime_hours_past_noon"] = (h - 12.0) % 24

    # Personal-norm reference
    median_bed = sl["bedtime_hours_past_noon"].median()
    p75_bed = sl["bedtime_hours_past_noon"].quantile(0.75)
    sl["bedtime_delta_personal"] = sl["bedtime_hours_past_noon"] - median_bed
    sl["is_late_bedtime"] = (sl["bedtime_hours_past_noon"] > p75_bed).astype(int)

    # Rename for cleanliness
    sl = sl.rename(columns={
        "Asleep duration (min)": "asleep_min",
        "In bed duration (min)": "in_bed_min",
        "Sleep efficiency %": "sleep_efficiency",
        "Sleep performance %": "sleep_performance",
        "Sleep consistency %": "sleep_consistency",
        "Light sleep duration (min)": "light_min",
        "Deep (SWS) duration (min)": "deep_min",
        "REM duration (min)": "rem_min",
        "Awake duration (min)": "awake_min",
        "Respiratory rate (rpm)": "respiratory_rate",
    })
    sl["sleep_duration_h"] = sl["asleep_min"] / 60
    return sl


if __name__ == "__main__":
    main()

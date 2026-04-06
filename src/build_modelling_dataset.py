"""
build_modelling_dataset.py — Construct the daily modelling dataset for RecoverWave.

One row per day, across the Whoop era (Aug 2023 → Apr 2026). Each row contains:
  * Same-day biometrics (HRV, RHR, recovery, sleep architecture, day strain)
  * Same-day Spotify listening aggregates (track count, total minutes,
    mean tempo / valence / energy / danceability / instrumentalness,
    mood diversity)
  * Workout flag and post-workout listening features (when applicable)
  * Target: next-day recovery score (recovery_score shifted by -1 day)

This is the foundation for all modelling: persistence, linear regression, GBM,
and the LSTM (which consumes a 7-day rolling sequence of these rows).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spotify_parser import SpotifyParser  # noqa: E402

AUDIO_FEATURES = ("tempo", "valence", "energy", "danceability",
                  "loudness", "acousticness", "instrumentalness", "speechiness")


def load_daily_biometrics() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data/processed/whoop_daily_timeline.csv")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date").reset_index(drop=True)
    keep = ["date", "recovery_score", "resting_heart_rate", "hrv_rmssd",
            "skin_temp", "blood_oxygen", "day_strain", "energy_burned",
            "sleep_performance", "respiratory_rate", "asleep_duration",
            "in_bed_duration", "light_sleep_duration", "deep_sws_duration",
            "rem_duration", "awake_duration", "sleep_efficiency",
            "sleep_consistency", "sleep_debt"]
    keep = [c for c in keep if c in df.columns]
    return df[keep]


if __name__ == "__main__":
    main()

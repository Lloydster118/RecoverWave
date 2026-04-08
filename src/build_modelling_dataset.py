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


def daily_listening_aggregates() -> pd.DataFrame:
    """One row per (London-local) day with Spotify aggregates."""
    parser = SpotifyParser(str(ROOT / "data/spotify/Spotify Extended Streaming History"))
    parser.load(); parser.clean()
    sp = parser.clean_df.copy()

    feat = pd.read_csv(ROOT / "data/processed/audio_features.csv")
    feat = feat.rename(columns={feat.columns[0]: "track_id"}).set_index("track_id")

    sp["local_date"] = (sp["end_time"].dt.tz_convert("Europe/London")
                                          .dt.date)
    sp = sp.merge(feat, left_on="track_id", right_index=True, how="left")

    grouped = sp.groupby("local_date")
    out = grouped.agg(
        n_tracks=("track_id", "count"),
        total_listen_minutes=("listen_seconds", lambda s: s.sum() / 60),
        unique_artists=("artist_name", "nunique"),
        artist_concentration=("artist_name",
                              lambda s: float((s.value_counts(normalize=True) ** 2).sum())),
    )
    for f in AUDIO_FEATURES:
        out[f"{f}_mean"] = grouped[f].mean()
        out[f"{f}_std"] = grouped[f].std()
    out["mood_diversity"] = grouped[["valence", "energy", "danceability"]].apply(
        lambda g: float(g.std().mean()) if len(g) > 1 else np.nan)

    out = out.reset_index().rename(columns={"local_date": "date"})
    return out


def post_workout_aggregates() -> pd.DataFrame:
    """One row per workout-day summarising the post-workout 2h window."""
    wkf = pd.read_csv(ROOT / "data/processed/workout_features_with_sleep.csv")
    wkf["workout_end_utc"] = pd.to_datetime(wkf["workout_end_utc"], utc=True,
                                            errors="coerce")
    wkf["date"] = (wkf["workout_end_utc"].dt.tz_convert("Europe/London")
                                          .dt.date)

    rename = {
        "n_tracks": "pw_n_tracks",
        "total_listen_minutes": "pw_listen_minutes",
        "unique_artists": "pw_unique_artists",
        "mood_diversity": "pw_mood_diversity",
        "artist_concentration": "pw_artist_concentration",
        "tempo_mean": "pw_tempo_mean",
        "valence_mean": "pw_valence_mean",
        "energy_mean": "pw_energy_mean",
        "danceability_mean": "pw_danceability_mean",
        "instrumentalness_mean": "pw_instrumentalness_mean",
        "loudness_mean": "pw_loudness_mean",
        "activity_strain": "pw_activity_strain",
    }
    cols = ["date"] + list(rename.keys())
    out = wkf[cols].rename(columns=rename)
    # If multiple workouts on a day, average — keeps schema 1-row-per-day
    out = out.groupby("date", as_index=False).mean(numeric_only=True)
    out["had_workout"] = 1
    return out


if __name__ == "__main__":
    main()

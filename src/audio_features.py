"""
audio_features.py — Fetch track-level audio features from ReccoBeats
(replacement for the deprecated Spotify audio-features endpoint).

Why ReccoBeats: Spotify deprecated `/v1/audio-features` on 27 Nov 2024 for new
apps. ReccoBeats exposes the same fields (danceability, energy, valence, tempo,
loudness, key, mode, acousticness, instrumentalness, liveness, speechiness)
keyed by Spotify track ID, free, no API key required.

Flow:
    1. Batch-lookup Spotify track IDs -> ReccoBeats internal IDs (up to 40/call)
    2. Fetch audio-features per ReccoBeats ID (parallelised)
    3. Persist to on-disk cache so we never re-hit the API for the same track
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


BASE = "https://api.reccobeats.com/v1"
BATCH_SIZE = 40
TIMEOUT = 15
CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "audio_features_cache.json"
FEATURE_FIELDS = ("danceability", "energy", "valence", "tempo", "loudness",
                  "key", "mode", "acousticness", "instrumentalness",
                  "liveness", "speechiness")


# ─────────────────────────────────────────────────────────────────────
# Cache
# ─────────────────────────────────────────────────────────────────────


def load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=0))


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    # Run against the post-workout track set
    sys.path.insert(0, str(Path(__file__).parent))
    from spotify_parser import SpotifyParser  # noqa: E402

    ROOT = Path(__file__).resolve().parents[1]
    parser = SpotifyParser(
        str(ROOT / "data/spotify/Spotify Extended Streaming History"))
    parser.load(); parser.clean()

    wk = pd.read_csv(ROOT / "data/processed/whoop_workouts_clean.csv")
    wk["end_utc"] = (pd.to_datetime(wk["workout_end_time"], errors="coerce")
                       .dt.tz_localize("Europe/London", ambiguous="NaT",
                                       nonexistent="NaT")
                       .dt.tz_convert("UTC"))
    wk = wk.dropna(subset=["end_utc"])

    all_tracks: set[str] = set()
    for _, w in wk.iterrows():
        win = parser.get_listening_window(w["end_utc"], window_hours=2.0)
        all_tracks.update(win["track_id"].dropna().unique())

    print(f"Unique tracks in post-workout windows: {len(all_tracks)}")
    features = fetch_all_features(all_tracks)

    # Quick diagnostic
    feat_df = pd.DataFrame({
        sp: f for sp, f in features.items() if f
    }).T
    feat_df.index.name = "track_id"
    out = ROOT / "data/processed/audio_features.csv"
    feat_df.to_csv(out)
    print(f"\nSaved {len(feat_df)} tracks with features → {out}")
    print(feat_df.describe().round(3).T)

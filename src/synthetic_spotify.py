"""
synthetic_spotify.py — Generate a synthetic Spotify extended-streaming-history
placeholder matching the exact schema of the real GDPR export.

Purpose
-------
The real Spotify export has been requested but not yet delivered. This module
fabricates a drop-in replacement so the downstream pipeline (parser, alignment,
audio-features join, LSTM training, Streamlit dashboard) can be developed and
tested end-to-end. When the real export arrives, `data/spotify/` is simply
overwritten and the pipeline runs unchanged.

Design choices
--------------
* Matches the real GDPR JSON schema field-for-field (see spotify_parser.py).
* Timestamps align with the investigator's real Whoop workouts so that a
  believable post-workout listening window exists for every session.
* Audio features (tempo, valence, energy, loudness, danceability) are sampled
  from a small library of archetype "playlist moods" rather than random noise,
  so the dataset carries testable structure.
* A deterministic seed is used so results are reproducible.
* Every record is tagged with `__synthetic__ = True` in a sidecar CSV so no
  synthetic row can ever be silently mistaken for real data downstream.

Ethics
------
No third-party data is used. All records are fabricated. This avoids the
GDPR / Data Protection Act 2018 complications that would follow from ingesting
public Whoop or Spotify exports belonging to other individuals.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SEED = 20260421
random.seed(SEED)
rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# Mood archetypes — each post-workout session is sampled from one of these
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Mood:
    name: str
    tempo: tuple[float, float]      # BPM mean, std
    valence: tuple[float, float]    # 0-1
    energy: tuple[float, float]     # 0-1
    danceability: tuple[float, float]
    loudness: tuple[float, float]   # dB
    track_pool: tuple[str, ...]


MOODS: tuple[Mood, ...] = (
    Mood("cooldown_ambient",
         tempo=(78.0, 6.0), valence=(0.35, 0.10), energy=(0.30, 0.08),
         danceability=(0.40, 0.08), loudness=(-14.0, 2.0),
         track_pool=("Weightless", "An Ending (Ascent)", "Spiegel im Spiegel",
                     "Avril 14th", "Nuvole Bianche", "On the Nature of Daylight")),
    Mood("lofi_focus",
         tempo=(85.0, 5.0), valence=(0.55, 0.10), energy=(0.40, 0.08),
         danceability=(0.55, 0.07), loudness=(-11.0, 1.5),
         track_pool=("Snowman", "Coffee", "Affection", "Midnight Walk",
                     "Daydream", "Slow Train")),
    Mood("indie_chill",
         tempo=(104.0, 8.0), valence=(0.60, 0.12), energy=(0.55, 0.10),
         danceability=(0.62, 0.08), loudness=(-9.0, 1.8),
         track_pool=("Naive", "Fluorescent Adolescent", "Dog Days Are Over",
                     "Electric Feel", "Little Dark Age", "Two Weeks")),
    Mood("hype_gym",
         tempo=(128.0, 6.0), valence=(0.65, 0.12), energy=(0.85, 0.06),
         danceability=(0.72, 0.06), loudness=(-5.5, 1.2),
         track_pool=("HUMBLE.", "Sicko Mode", "Lose Yourself", "Mo Bamba",
                     "Till I Collapse", "Power", "Stronger")),
    Mood("late_night_downbeat",
         tempo=(92.0, 9.0), valence=(0.28, 0.10), energy=(0.42, 0.10),
         danceability=(0.50, 0.09), loudness=(-12.0, 2.0),
         track_pool=("Motion Sickness", "Night Shift", "Liability",
                     "Youth", "Saturn", "Fade Into You")),
)

ARTISTS_BY_MOOD = {
    "cooldown_ambient": ("Marconi Union", "Brian Eno", "Arvo Pärt",
                         "Aphex Twin", "Ludovico Einaudi", "Max Richter"),
    "lofi_focus": ("Sia", "beabadoobee", "Cigarettes After Sex",
                   "Men I Trust", "Japanese Breakfast", "Cavetown"),
    "indie_chill": ("The Kooks", "Arctic Monkeys", "Florence + The Machine",
                    "MGMT", "MGMT", "FKA twigs"),
    "hype_gym": ("Kendrick Lamar", "Travis Scott", "Eminem", "Sheck Wes",
                 "Eminem", "Kanye West", "Kanye West"),
    "late_night_downbeat": ("Phoebe Bridgers", "Lucy Dacus", "Lorde",
                            "Daughter", "Sleeping At Last", "Mazzy Star"),
}


def _track_uri() -> str:
    """Return a plausible Spotify track URI."""
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "spotify:track:" + "".join(random.choices(chars, k=22))


def _sample_clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(np.clip(value, lo, hi))


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------
def generate_spotify_history(
    workouts_csv: str | Path,
    output_dir: str | Path,
    session_minutes_after_workout: tuple[int, int] = (20, 90),
    extra_daily_sessions: int = 2,
    years_back: int = 3,
) -> tuple[Path, Path]:
    """
    Build synthetic Streaming_History_Audio_*.json files matching the schema
    and a sidecar audio-features CSV.

    Args:
        workouts_csv: Path to whoop_workouts_clean.csv (must have 'workout_end_ts').
        output_dir: Where to write the synthetic Spotify files.
        session_minutes_after_workout: Range of post-workout listening duration.
        extra_daily_sessions: Non-workout random sessions per day (commute, etc).
        years_back: How far back to generate non-workout sessions.

    Returns:
        (json_path, features_csv_path)
    """
    workouts_csv = Path(workouts_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load workouts for anchoring ---
    wk = pd.read_csv(workouts_csv)
    # Find the end-time column flexibly
    end_col = next((c for c in wk.columns
                    if "end" in c.lower() and ("time" in c.lower() or "ts" in c.lower())),
                   None)
    if end_col is None:
        raise KeyError(f"Could not find workout end-time column in {workouts_csv}. "
                       f"Columns: {list(wk.columns)}")
    wk[end_col] = pd.to_datetime(wk[end_col], utc=True, errors="coerce")
    wk = wk.dropna(subset=[end_col]).reset_index(drop=True)

    events: list[dict] = []
    features: list[dict] = []

    # --- Post-workout sessions (primary data of interest) ---
    for _, row in wk.iterrows():
        end_ts = row[end_col]
        mood = random.choice(MOODS)
        session_len_min = rng.integers(*session_minutes_after_workout)
        # Tracks last 3-5 min each
        n_tracks = max(1, int(session_len_min // rng.integers(3, 6)))
        start = end_ts + timedelta(minutes=int(rng.integers(2, 10)))

        for _ in range(n_tracks):
            e, f = _generate_track_event(start, mood, source="post_workout")
            events.append(e); features.append(f)
            start += timedelta(seconds=e["ms_played"] / 1000 + random.randint(0, 20))

    # --- Background daily sessions (ambient listening) ---
    if not events:
        raise RuntimeError("No workout events — synthetic generation aborted.")
    oldest = min(e_["ts_dt"] for e_ in [{"ts_dt": pd.to_datetime(e["ts"])} for e in events])
    newest = max(e_["ts_dt"] for e_ in [{"ts_dt": pd.to_datetime(e["ts"])} for e in events])
    day = oldest.floor("D")
    while day <= newest:
        for _ in range(extra_daily_sessions):
            mood = random.choice(MOODS)
            t = day + timedelta(
                hours=int(rng.integers(7, 23)),
                minutes=int(rng.integers(0, 60)),
            )
            for _ in range(int(rng.integers(3, 9))):
                e, f = _generate_track_event(t, mood, source="ambient")
                events.append(e); features.append(f)
                t += timedelta(seconds=e["ms_played"] / 1000 + random.randint(0, 45))
        day += timedelta(days=1)

    # --- Sort by timestamp and drop helper cols ---
    events.sort(key=lambda e: e["ts"])
    for e in events:
        e.pop("ts_dt", None)

    # --- Write JSON shards by year (mimics real export) ---
    df_ev = pd.DataFrame(events)
    df_ev["year"] = pd.to_datetime(df_ev["ts"]).dt.year
    written: list[Path] = []
    for i, (yr, chunk) in enumerate(df_ev.groupby("year")):
        out = output_dir / f"Streaming_History_Audio_{yr}_{i}.json"
        chunk_recs = chunk.drop(columns=["year"]).to_dict(orient="records")
        out.write_text(json.dumps(chunk_recs, indent=2), encoding="utf-8")
        written.append(out)

    # --- Write audio-features sidecar CSV ---
    feat_df = pd.DataFrame(features)
    feat_df["__synthetic__"] = True
    feat_path = output_dir / "synthetic_audio_features.csv"
    feat_df.to_csv(feat_path, index=False)

    # --- README ---
    readme = output_dir / "README_SYNTHETIC.md"
    readme.write_text(
        "# SYNTHETIC Spotify data — placeholder only\n\n"
        f"Generated: {datetime.now(timezone.utc).isoformat()}\n"
        f"Seed: {SEED}\n"
        f"Events: {len(events)}\n"
        f"Files: {len(written)}\n\n"
        "These files are **fabricated** for pipeline development. They match "
        "the real GDPR Streaming_History_Audio schema field-for-field, so the "
        "same parser can consume them. When the real Spotify export arrives, "
        "delete this directory and replace with the real JSON files.\n\n"
        "Every track also appears in `synthetic_audio_features.csv` with the "
        "`__synthetic__=True` flag so no synthetic row can be silently "
        "mistaken for real data downstream.\n",
        encoding="utf-8",
    )

    return written[0], feat_path


def _generate_track_event(
    start: datetime, mood: Mood, source: str
) -> tuple[dict, dict]:
    """Generate one listen event matching the Spotify export schema, plus its features."""
    # Duration: 60-300s, biased by energy
    ms_played = int(rng.integers(60_000, 300_000))
    track = random.choice(mood.track_pool)
    artist = random.choice(ARTISTS_BY_MOOD[mood.name])
    uri = _track_uri()
    ts = (start + timedelta(seconds=ms_played / 1000)).astimezone(timezone.utc)

    event = {
        "ts": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ts_dt": ts,
        "platform": random.choice(["iOS 17.4", "OS X 14.4.1", "Android"]),
        "ms_played": ms_played,
        "conn_country": "GB",
        "master_metadata_track_name": track,
        "master_metadata_album_artist_name": artist,
        "master_metadata_album_album_name": f"{artist} — Album",
        "spotify_track_uri": uri,
        "reason_start": random.choice(["clickrow", "trackdone", "fwdbtn"]),
        "reason_end": random.choice(["trackdone", "fwdbtn", "endplay"]),
        "shuffle": random.choice([True, False]),
        "skipped": ms_played < 30_000,
        "offline": False,
        "incognito_mode": False,
    }

    features = {
        "spotify_track_uri": uri,
        "track_name": track,
        "artist_name": artist,
        "mood_archetype": mood.name,
        "source": source,
        "tempo": round(_sample_clip(rng.normal(*mood.tempo), 40, 220), 1),
        "valence": round(_sample_clip(rng.normal(*mood.valence)), 3),
        "energy": round(_sample_clip(rng.normal(*mood.energy)), 3),
        "danceability": round(_sample_clip(rng.normal(*mood.danceability)), 3),
        "loudness": round(float(rng.normal(*mood.loudness)), 2),
        "duration_ms": ms_played,
    }
    return event, features


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    wk_csv = root / "data" / "processed" / "whoop_workouts_clean.csv"
    out_dir = root / "data" / "spotify_synthetic"
    j, f = generate_spotify_history(wk_csv, out_dir)
    print(f"Wrote synthetic Spotify export → {out_dir}")
    print(f"  first JSON shard: {j}")
    print(f"  audio-features:   {f}")

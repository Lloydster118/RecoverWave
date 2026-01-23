"""Generate synthetic Spotify-like listening history for development."""

import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


@dataclass
class Mood:
    label: str
    tempo_mu: float
    energy_mu: float
    valence_mu: float


MOODS = [
    Mood("calm", 80, 0.3, 0.5),
    Mood("upbeat", 120, 0.7, 0.7),
    Mood("intense", 140, 0.9, 0.5),
    Mood("reflective", 90, 0.4, 0.3),
]


def _track_uri() -> str:
    return f"spotify:track:{random.randint(10**10, 10**11)}"


def generate(days: int = 60, tracks_per_day: int = 18, out: Path = Path("data/spotify_synthetic/sample.json")) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    events = []
    start = datetime(2025, 1, 1)
    for d in range(days):
        # Pick a daily mood
        mood = random.choice(MOODS)
        for t in range(tracks_per_day):
            ts = start + timedelta(days=d, minutes=t * 30 + random.randint(0, 25))
            events.append({
                "ts": ts.isoformat() + "Z",
                "master_metadata_track_name": f"Track_{random.randint(1, 500)}",
                "master_metadata_album_artist_name": f"Artist_{random.randint(1, 80)}",
                "spotify_track_uri": _track_uri(),
                "ms_played": random.randint(60000, 240000),
                "_synthetic_tempo": mood.tempo_mu + random.gauss(0, 8),
                "_synthetic_energy": max(0, min(1, mood.energy_mu + random.gauss(0, 0.1))),
                "_synthetic_valence": max(0, min(1, mood.valence_mu + random.gauss(0, 0.15))),
            })
    out.write_text(json.dumps(events, indent=2))


if __name__ == "__main__":
    generate()

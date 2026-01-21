"""Generate synthetic Spotify-like listening history for development.

Used before the real Spotify Extended Streaming History export arrives.
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path


def generate(days: int = 30, tracks_per_day: int = 12, out: Path = Path("data/spotify_synthetic/sample.json")) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    events = []
    start = datetime(2025, 1, 1)
    for d in range(days):
        for t in range(tracks_per_day):
            ts = start + timedelta(days=d, minutes=t * 30 + random.randint(0, 25))
            events.append({
                "ts": ts.isoformat() + "Z",
                "master_metadata_track_name": f"Track_{random.randint(1, 200)}",
                "master_metadata_album_artist_name": f"Artist_{random.randint(1, 50)}",
                "spotify_track_uri": f"spotify:track:{random.randint(10**10, 10**11)}",
                "ms_played": random.randint(60000, 240000),
            })
    out.write_text(json.dumps(events, indent=2))


if __name__ == "__main__":
    generate()

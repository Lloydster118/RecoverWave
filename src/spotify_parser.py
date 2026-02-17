"""
spotify_parser.py — Parse Spotify Extended Streaming History JSON exports.

Spotify GDPR export delivers JSON files named like:
  Streaming_History_Audio_2023-2024_12.json

Each file contains an array of listen events with fields:
  - ts                                (UTC timestamp when stream ended)
  - master_metadata_track_name        (track name)
  - master_metadata_album_artist_name (artist)
  - master_metadata_album_album_name  (album)
  - spotify_track_uri                 (e.g. spotify:track:ABC123)
  - ms_played                         (milliseconds played)
  - reason_start / reason_end         (why playback started/stopped)
  - shuffle / skipped / offline       (booleans)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


class SpotifyParser:
    def __init__(self, data_dir: str, min_listen_seconds: int = 30):
        """
        Args:
            data_dir: Path to folder containing Spotify JSON exports.
            min_listen_seconds: Minimum play duration to count as a genuine listen.
                                Default 30s filters out skips and accidental plays.
        """
        self.data_dir = Path(data_dir)
        self.MIN_LISTEN_SECONDS = min_listen_seconds
        self.raw_df = None
        self.clean_df = None

# ── Quick test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/spotify"
    parser = SpotifyParser(data_dir)

    try:
        parser.load()
        df = parser.clean()
        print(f"\nSample events:")
        print(df[["end_time", "track_name", "artist_name", "listen_seconds"]].head(10))

        # Example: get listening after a specific time
        if len(df) > 0:
            sample_time = df.iloc[len(df) // 2]["end_time"]
            window = parser.get_listening_window(sample_time, window_hours=2.0)
            print(f"\nListening window after {sample_time}: {len(window)} tracks")
    except FileNotFoundError as e:
        print(e)

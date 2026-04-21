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
    """Load and clean Spotify extended streaming history."""

    # Minimum listen time in seconds to count as a "real" listen
    MIN_LISTEN_SECONDS = 30

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

    def load(self) -> pd.DataFrame:
        """Load all Streaming_History_Audio JSON files into a single DataFrame."""
        json_files = sorted(self.data_dir.glob("Streaming_History_Audio*.json"))

        if not json_files:
            # Also try the account data format (shorter history)
            json_files = sorted(self.data_dir.glob("StreamingHistory_music*.json"))

        if not json_files:
            raise FileNotFoundError(
                f"No Spotify streaming history files found in {self.data_dir}/\n"
                f"Expected files like: Streaming_History_Audio_2023-2024_0.json"
            )

        all_events = []
        for f in json_files:
            with open(f, "r", encoding="utf-8") as fp:
                events = json.load(fp)
                print(f"  Loaded {f.name}: {len(events)} events")
                all_events.extend(events)

        print(f"Total raw events: {len(all_events)}")
        self.raw_df = pd.DataFrame(all_events)
        return self.raw_df

    def clean(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Clean raw streaming history:
          1. Parse timestamps
          2. Filter out non-music (podcasts, videos)
          3. Remove skips / very short listens
          4. Extract track IDs
          5. Standardise column names

        Returns:
            Cleaned DataFrame with one row per genuine listen event.
        """
        if df is None:
            df = self.raw_df
        if df is None:
            raise ValueError("No data loaded — call load() first")

        df = df.copy()

        # ── Handle both extended and account data formats ──

        # Extended format uses 'ts', account format uses 'endTime'
        if "ts" in df.columns:
            df["end_time"] = pd.to_datetime(df["ts"], format="ISO8601", utc=True)
        elif "endTime" in df.columns:
            df["end_time"] = pd.to_datetime(df["endTime"], utc=True)

        # Track name
        if "master_metadata_track_name" in df.columns:
            df["track_name"] = df["master_metadata_track_name"]
            df["artist_name"] = df["master_metadata_album_artist_name"]
            df["album_name"] = df["master_metadata_album_album_name"]
        elif "trackName" in df.columns:
            df["track_name"] = df["trackName"]
            df["artist_name"] = df["artistName"]
            df["album_name"] = df.get("albumName")

        # Play duration
        if "ms_played" in df.columns:
            df["listen_seconds"] = df["ms_played"] / 1000
        elif "msPlayed" in df.columns:
            df["listen_seconds"] = df["msPlayed"] / 1000

        # Track URI (extended format only)
        if "spotify_track_uri" in df.columns:
            df["track_uri"] = df["spotify_track_uri"]
            df["track_id"] = df["track_uri"].str.replace("spotify:track:", "", regex=False)
        else:
            df["track_uri"] = None
            df["track_id"] = None

        # ── Filter ──

        # Remove rows with no track name (podcasts show as episode_name instead)
        df = df.dropna(subset=["track_name"])

        # Remove very short listens (skips, accidental plays)
        df = df[df["listen_seconds"] >= self.MIN_LISTEN_SECONDS]

        # Compute start time from end time and duration
        df["start_time"] = df["end_time"] - pd.to_timedelta(df["listen_seconds"], unit="s")

        # Extract date and hour for convenience
        df["date"] = df["end_time"].dt.date
        df["hour"] = df["end_time"].dt.hour

        # Keep useful columns
        keep_cols = [
            "start_time", "end_time", "date", "hour",
            "track_name", "artist_name", "album_name",
            "track_id", "track_uri", "listen_seconds",
        ]
        # Add extended-format columns if available
        for extra in ["reason_start", "reason_end", "shuffle", "skipped", "offline"]:
            if extra in df.columns:
                keep_cols.append(extra)

        self.clean_df = df[keep_cols].sort_values("end_time").reset_index(drop=True)

        print(f"Clean events: {len(self.clean_df)} "
              f"(filtered from {len(self.raw_df)}, "
              f"min {self.MIN_LISTEN_SECONDS}s listen time)")
        print(f"Date range: {self.clean_df['date'].min()} to {self.clean_df['date'].max()}")
        print(f"Unique tracks: {self.clean_df['track_id'].nunique()}")

        return self.clean_df

    def get_listening_window(
        self,
        after: pd.Timestamp,
        window_hours: float = 2.0,
    ) -> pd.DataFrame:
        """
        Get all listens within a time window after a given timestamp.

        This is the core method for RecoverWave — given a workout end time,
        retrieve the music listened to in the post-workout recovery window.

        Args:
            after: Start of window (e.g. workout end time). Must be tz-aware (UTC).
            window_hours: Duration of window in hours (default 2h).

        Returns:
            DataFrame of listen events within the window, ordered by time.
        """
        if self.clean_df is None:
            raise ValueError("No cleaned data — call clean() first")

        window_end = after + pd.Timedelta(hours=window_hours)

        mask = (self.clean_df["start_time"] >= after) & (self.clean_df["start_time"] < window_end)
        window = self.clean_df[mask].copy()

        # Add relative time (minutes since window start)
        window["minutes_after_workout"] = (
            (window["start_time"] - after).dt.total_seconds() / 60
        ).round(1)

        return window


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

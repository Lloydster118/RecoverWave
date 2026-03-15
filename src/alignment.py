"""
alignment.py — Temporal alignment pipeline for RecoverWave.

The core data pipeline:
  1. For each workout session, find the post-workout listening window
  2. Extract audio feature sequences from that window
  3. Link to the NEXT DAY's recovery score from Whoop
  4. Produce aligned dataset ready for modelling

Each row in the output represents:
  workout → [post-workout listening sequence] → next-day recovery outcome
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
from datetime import timedelta


class TemporalAligner:
    """
    Align workout sessions with post-workout listening and next-day recovery.

    The alignment logic:
      - Workout ends at time T
      - Listening window = [T, T + window_hours]
      - Recovery outcome = Whoop recovery score for the NEXT calendar day
        (since recovery is measured upon waking the following morning)
    """

    def __init__(self, window_hours: float = 2.0):
        """
        Args:
            window_hours: Hours after workout end to capture listening data.
        """
        self.window_hours = window_hours

    def align(
        self,
        workouts_df: pd.DataFrame,
        listening_df: pd.DataFrame,
        daily_timeline: pd.DataFrame,
        audio_features_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Build the aligned dataset.

        Args:
            workouts_df: Whoop workouts with 'workout_end_dt' column.
            listening_df: Spotify listening history (cleaned) with 'start_time', 'end_time'.
            daily_timeline: Whoop daily timeline with 'date' and 'recovery_score'.
            audio_features_df: Optional pre-merged audio features on listening_df.

        Returns:
            DataFrame with one row per workout, containing:
              - Workout metadata (strain, HR, duration)
              - Post-workout listening sequence features (aggregated)
              - Next-day recovery score (target variable)
        """
        if "workout_end_dt" not in workouts_df.columns:
            raise ValueError("workouts_df must have 'workout_end_dt' column")

        aligned_rows = []

        for idx, workout in workouts_df.iterrows():
            workout_end = workout["workout_end_dt"]

            if pd.isna(workout_end):
                continue

            # ── Get post-workout listening window ──
            window_start = workout_end
            window_end = workout_end + pd.Timedelta(hours=self.window_hours)

            # Ensure timezone compatibility
            if listening_df["start_time"].dt.tz is not None and workout_end.tzinfo is None:
                window_start = window_start.tz_localize("UTC")
                window_end = window_end.tz_localize("UTC")
            elif listening_df["start_time"].dt.tz is None and workout_end.tzinfo is not None:
                window_start = window_start.tz_localize(None)
                window_end = window_end.tz_localize(None)

            listening_window = listening_df[
                (listening_df["start_time"] >= window_start) &
                (listening_df["start_time"] < window_end)
            ]

            # ── Get next-day recovery score ──
            workout_date = workout_end.date() if hasattr(workout_end, 'date') else workout_end
            next_day = workout_date + timedelta(days=1)

            recovery_row = daily_timeline[daily_timeline["date"] == next_day]
            recovery_score = None
            if len(recovery_row) > 0:
                recovery_col = self._find_col(recovery_row, ["recovery_score", "recovery"])
                if recovery_col:
                    recovery_score = recovery_row[recovery_col].iloc[0]

            # ── Build feature row ──
            row = self._build_feature_row(workout, listening_window, recovery_score, audio_features_df)
            row["workout_date"] = str(workout_date)
            row["next_day_date"] = str(next_day)
            row["n_tracks_in_window"] = len(listening_window)
            row["has_listening_data"] = len(listening_window) > 0
            row["recovery_score"] = recovery_score

            aligned_rows.append(row)

        result = pd.DataFrame(aligned_rows)

        # Summary stats
        total = len(result)
        with_listening = result["has_listening_data"].sum()
        with_recovery = result["recovery_score"].notna().sum()
        complete = ((result["has_listening_data"]) & (result["recovery_score"].notna())).sum()

        print(f"\n{'='*60}")
        print(f"Alignment Summary:")
        print(f"  Total workouts:              {total}")
        print(f"  With post-workout listening:  {with_listening} ({100*with_listening/max(total,1):.0f}%)")
        print(f"  With next-day recovery:       {with_recovery} ({100*with_recovery/max(total,1):.0f}%)")
        print(f"  Complete rows (both):          {complete} ({100*complete/max(total,1):.0f}%)")
        print(f"{'='*60}")

        return result

    def _build_feature_row(
        self,
        workout: pd.Series,
        listening_window: pd.DataFrame,
        recovery_score: Optional[float],
        audio_features_df: Optional[pd.DataFrame],
    ) -> dict:
        """Build a single feature row from workout + listening data."""
        row = {}

        # ── Workout features ──
        for col in ["strain_score", "average_heart_rate", "max_heart_rate",
                     "kilojoules", "duration_minutes"]:
            if col in workout.index:
                row[f"workout_{col}"] = workout[col]

        # ── Listening sequence features ──
        if len(listening_window) == 0:
            # No listening data — fill with NaN
            for prefix in ["tempo", "energy", "valence", "danceability",
                           "loudness", "acousticness", "instrumentalness"]:
                for stat in ["mean", "std", "first", "last", "slope"]:
                    row[f"listening_{prefix}_{stat}"] = np.nan
            row["listening_total_minutes"] = 0
            row["listening_n_unique_artists"] = 0
            row["listening_n_tracks"] = 0
            return row

        # Basic listening stats
        row["listening_total_minutes"] = listening_window["listen_seconds"].sum() / 60
        row["listening_n_tracks"] = len(listening_window)
        row["listening_n_unique_artists"] = listening_window["artist_name"].nunique()

        # ── Audio feature sequence stats ──
        # If audio features are merged into the listening data
        feature_cols = ["tempo", "energy", "valence", "danceability",
                        "loudness", "acousticness", "instrumentalness",
                        "speechiness", "liveness"]

        available_features = [c for c in feature_cols if c in listening_window.columns]

        for feat in available_features:
            values = listening_window[feat].dropna()
            if len(values) == 0:
                for stat in ["mean", "std", "first", "last", "slope"]:
                    row[f"listening_{feat}_{stat}"] = np.nan
                continue

            row[f"listening_{feat}_mean"] = values.mean()
            row[f"listening_{feat}_std"] = values.std() if len(values) > 1 else 0
            row[f"listening_{feat}_first"] = values.iloc[0]
            row[f"listening_{feat}_last"] = values.iloc[-1]

            # Slope: direction of change across the window (increasing/decreasing)
            if len(values) > 1:
                x = np.arange(len(values))
                slope = np.polyfit(x, values.values, 1)[0]
                row[f"listening_{feat}_slope"] = slope
            else:
                row[f"listening_{feat}_slope"] = 0

        return row

    def build_sequences(
        self,
        aligned_df: pd.DataFrame,
        listening_df: pd.DataFrame,
        workouts_df: pd.DataFrame,
        max_seq_length: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build padded sequences for LSTM/Transformer input.

        Instead of aggregated features, this returns the raw temporal sequence
        of audio features for each workout's listening window.

        Args:
            aligned_df: Output from self.align()
            listening_df: Full cleaned listening history
            workouts_df: Whoop workouts
            max_seq_length: Maximum tracks per sequence (pad/truncate)

        Returns:
            Tuple of:
              - sequences: np.array of shape (n_workouts, max_seq_length, n_features)
              - targets: np.array of shape (n_workouts,) — recovery scores
              - masks: np.array of shape (n_workouts, max_seq_length) — True where real data
        """
        feature_cols = ["tempo", "energy", "valence", "danceability",
                        "loudness", "acousticness", "instrumentalness"]
        available = [c for c in feature_cols if c in listening_df.columns]

        if not available:
            raise ValueError(
                f"No audio features found in listening_df. "
                f"Available columns: {list(listening_df.columns)}"
            )

        n_features = len(available)
        sequences = []
        targets = []
        masks = []

        for _, row in aligned_df.iterrows():
            if pd.isna(row["recovery_score"]) or not row["has_listening_data"]:
                continue

            workout_date = pd.to_datetime(row["workout_date"]).date()
            workout_rows = workouts_df[workouts_df["workout_end_date"] == workout_date]

            if len(workout_rows) == 0:
                continue

            workout_end = workout_rows.iloc[0]["workout_end_dt"]
            window_start = workout_end
            window_end = workout_end + pd.Timedelta(hours=self.window_hours)

            # Get listening window
            if listening_df["start_time"].dt.tz is not None and window_start.tzinfo is None:
                window_start = window_start.tz_localize("UTC")
                window_end = window_end.tz_localize("UTC")

            window = listening_df[
                (listening_df["start_time"] >= window_start) &
                (listening_df["start_time"] < window_end)
            ][available].values

            # Pad or truncate
            seq = np.zeros((max_seq_length, n_features))
            mask = np.zeros(max_seq_length, dtype=bool)

            actual_len = min(len(window), max_seq_length)
            if actual_len > 0:
                seq[:actual_len] = window[:actual_len]
                mask[:actual_len] = True

            sequences.append(seq)
            targets.append(row["recovery_score"])
            masks.append(mask)

        print(f"Built {len(sequences)} sequences (max_len={max_seq_length}, "
              f"n_features={n_features})")

        return (
            np.array(sequences, dtype=np.float32),
            np.array(targets, dtype=np.float32),
            np.array(masks, dtype=bool),
        )

    @staticmethod
    def _find_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None


# ── Quick test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("TemporalAligner ready.")
    print("\nUsage:")
    print("  aligner = TemporalAligner(window_hours=2.0)")
    print("  aligned = aligner.align(workouts_df, listening_df, daily_timeline)")
    print("  sequences, targets, masks = aligner.build_sequences(aligned, listening_df, workouts_df)")

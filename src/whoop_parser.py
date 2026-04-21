"""
whoop_parser.py — Parse and clean Whoop CSV exports.

Whoop exports 4 CSV files via Settings > Data Export:
  - physiological_cycles.csv  (daily cycle: HRV, RHR, recovery score, strain)
  - workouts.csv              (each workout: start/end time, strain, avg/max HR, sport)
  - sleeps.csv                (each sleep: start/end, stages, efficiency, disturbances)
  - journal_entries.csv       (daily journal: stress, caffeine, alcohol, soreness, etc.)

This module loads, cleans, and merges them into a unified daily timeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


class WhoopParser:
    """Load and clean Whoop CSV exports into analysis-ready DataFrames."""

    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: Path to folder containing Whoop CSV exports.
        """
        self.data_dir = Path(data_dir)
        self.cycles = None
        self.workouts = None
        self.sleeps = None
        self.journal = None

    def load_all(self) -> dict:
        """Load all 4 Whoop CSVs. Returns dict of DataFrames."""
        self.cycles = self._load_cycles()
        self.workouts = self._load_workouts()
        self.sleeps = self._load_sleeps()
        self.journal = self._load_journal()

        print(f"Loaded: {len(self.cycles)} cycles, {len(self.workouts)} workouts, "
              f"{len(self.sleeps)} sleeps, {len(self.journal)} journal entries")

        return {
            "cycles": self.cycles,
            "workouts": self.workouts,
            "sleeps": self.sleeps,
            "journal": self.journal,
        }

    # ── Cycles (daily physiological summary) ──────────────────────────

    def _load_cycles(self) -> pd.DataFrame:
        """Parse physiological_cycles.csv."""
        path = self.data_dir / "physiological_cycles.csv"
        if not path.exists():
            raise FileNotFoundError(f"Expected file: {path}")

        df = pd.read_csv(path)
        df = self._normalise_columns(df)

        # Parse all datetime columns
        for col in df.columns:
            if any(k in col for k in ["time", "date", "onset"]):
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Extract the calendar date for merging (use cycle_start_time)
        if "cycle_start_time" in df.columns:
            df["date"] = df["cycle_start_time"].dt.date

        # Ensure numeric columns are float
        numeric_cols = [
            "recovery_score", "resting_heart_rate", "hrv_rmssd",
            "skin_temp", "blood_oxygen", "day_strain", "energy_burned",
            "max_hr", "average_hr", "sleep_performance", "respiratory_rate",
            "asleep_duration", "in_bed_duration", "light_sleep_duration",
            "deep_sws_duration", "rem_duration", "awake_duration",
            "sleep_need", "sleep_debt", "sleep_efficiency", "sleep_consistency",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    # ── Workouts ──────────────────────────────────────────────────────

    def _load_workouts(self) -> pd.DataFrame:
        """Parse workouts.csv."""
        path = self.data_dir / "workouts.csv"
        if not path.exists():
            raise FileNotFoundError(f"Expected file: {path}")

        df = pd.read_csv(path)
        df = self._normalise_columns(df)

        for col in df.columns:
            if any(k in col for k in ["time", "date", "onset"]):
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Canonical workout-end fields used by the rest of the pipeline
        if "workout_end_time" in df.columns:
            df["workout_end_dt"] = df["workout_end_time"]
            df["workout_end_date"] = df["workout_end_time"].dt.date
        if "workout_start_time" in df.columns:
            df["workout_start_dt"] = df["workout_start_time"]

        # Canonical strain column used throughout project
        if "activity_strain" in df.columns and "strain_score" not in df.columns:
            df["strain_score"] = df["activity_strain"]

        # Canonical HR columns
        if "average_hr" in df.columns and "average_heart_rate" not in df.columns:
            df["average_heart_rate"] = df["average_hr"]
        if "max_hr" in df.columns and "max_heart_rate" not in df.columns:
            df["max_heart_rate"] = df["max_hr"]

        for col in ["strain_score", "activity_strain", "average_heart_rate",
                    "max_heart_rate", "average_hr", "max_hr",
                    "duration", "energy_burned"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    # ── Sleeps ────────────────────────────────────────────────────────

    def _load_sleeps(self) -> pd.DataFrame:
        """Parse sleeps.csv."""
        path = self.data_dir / "sleeps.csv"
        if not path.exists():
            raise FileNotFoundError(f"Expected file: {path}")

        df = pd.read_csv(path)
        df = self._normalise_columns(df)

        for col in df.columns:
            if any(k in col for k in ["time", "date", "onset"]):
                df[col] = pd.to_datetime(df[col], errors="coerce")

        numeric_cols = [
            "sleep_performance", "respiratory_rate",
            "asleep_duration", "in_bed_duration",
            "light_sleep_duration", "deep_sws_duration",
            "rem_duration", "awake_duration",
            "sleep_need", "sleep_debt",
            "sleep_efficiency", "sleep_consistency",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Stage ratios (as fraction of asleep duration)
        if "asleep_duration" in df.columns:
            total = df["asleep_duration"].replace(0, np.nan)
            if "light_sleep_duration" in df.columns:
                df["light_ratio"] = df["light_sleep_duration"] / total
            if "deep_sws_duration" in df.columns:
                df["deep_ratio"] = df["deep_sws_duration"] / total
            if "rem_duration" in df.columns:
                df["rem_ratio"] = df["rem_duration"] / total

        return df

    # ── Journal Entries ───────────────────────────────────────────────

    def _load_journal(self) -> pd.DataFrame:
        """Parse journal_entries.csv."""
        path = self.data_dir / "journal_entries.csv"
        if not path.exists():
            print(f"Warning: {path} not found — journal data will be empty")
            return pd.DataFrame()

        df = pd.read_csv(path)
        df = self._normalise_columns(df)

        for col in df.columns:
            if any(k in col for k in ["time", "date"]):
                df[col] = pd.to_datetime(df[col], errors="coerce")

        if "cycle_start_time" in df.columns:
            df["date"] = df["cycle_start_time"].dt.date

        return df

    # ── Merged Daily Timeline ─────────────────────────────────────────

    def build_daily_timeline(self) -> pd.DataFrame:
        """
        Merge cycles, sleeps, and journal into a single daily DataFrame.
        Workouts are kept separate since there can be multiple per day.

        Returns:
            DataFrame indexed by date with recovery, sleep, and journal features.
        """
        if self.cycles is None:
            self.load_all()

        # Start with cycles (one row per day)
        if "date" not in self.cycles.columns:
            raise ValueError("Cycles DataFrame missing 'date' column — check CSV format")

        timeline = self.cycles.copy()

        # Aggregate sleeps to daily (take the main sleep per night)
        if len(self.sleeps) > 0:
            sleep_cols = [c for c in self.sleeps.columns
                          if any(k in c for k in ["ratio", "duration", "efficiency",
                                                   "performance", "disturbances", "respiratory"])]
            if sleep_cols:
                # Use the sleep that ends closest to the cycle start
                sleep_daily = self.sleeps.groupby(
                    self.sleeps.iloc[:, 0].dt.date  # group by first date-like column
                )[sleep_cols].first()
                sleep_daily.index.name = "date"
                timeline = timeline.merge(sleep_daily, on="date", how="left", suffixes=("", "_sleep"))

        # Merge journal entries
        if len(self.journal) > 0:
            date_col = self._find_column(self.journal, ["date", "created_at", "journal_date"])
            if date_col:
                self.journal["date"] = pd.to_datetime(self.journal[date_col]).dt.date
                journal_cols = [c for c in self.journal.columns if c != date_col and c != "date"]
                journal_daily = self.journal.groupby("date")[journal_cols].first()
                timeline = timeline.merge(journal_daily, on="date", how="left", suffixes=("", "_journal"))

        print(f"Daily timeline: {len(timeline)} days, {len(timeline.columns)} features")
        return timeline

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Normalise Whoop's human-readable column names to snake_case keys.

        Handles the 2025/2026 export format where columns look like
        'Heart rate variability (ms)' or 'Activity Strain'.
        """
        import re
        mapping = {
            # Core
            "Cycle start time": "cycle_start_time",
            "Cycle end time": "cycle_end_time",
            "Cycle timezone": "cycle_timezone",
            # Cycle metrics
            "Recovery score %": "recovery_score",
            "Resting heart rate (bpm)": "resting_heart_rate",
            "Heart rate variability (ms)": "hrv_rmssd",
            "Skin temp (celsius)": "skin_temp",
            "Blood oxygen %": "blood_oxygen",
            "Day Strain": "day_strain",
            "Energy burned (cal)": "energy_burned",
            "Max HR (bpm)": "max_hr",
            "Average HR (bpm)": "average_hr",
            # Sleep
            "Sleep onset": "sleep_onset",
            "Wake onset": "wake_onset",
            "Sleep performance %": "sleep_performance",
            "Respiratory rate (rpm)": "respiratory_rate",
            "Asleep duration (min)": "asleep_duration",
            "In bed duration (min)": "in_bed_duration",
            "Light sleep duration (min)": "light_sleep_duration",
            "Deep (SWS) duration (min)": "deep_sws_duration",
            "REM duration (min)": "rem_duration",
            "Awake duration (min)": "awake_duration",
            "Sleep need (min)": "sleep_need",
            "Sleep debt (min)": "sleep_debt",
            "Sleep efficiency %": "sleep_efficiency",
            "Sleep consistency %": "sleep_consistency",
            "Nap": "nap",
            # Workouts
            "Workout start time": "workout_start_time",
            "Workout end time": "workout_end_time",
            "Duration (min)": "duration",
            "Activity name": "activity_name",
            "Activity Strain": "activity_strain",
            "HR Zone 1 %": "hr_zone_1",
            "HR Zone 2 %": "hr_zone_2",
            "HR Zone 3 %": "hr_zone_3",
            "HR Zone 4 %": "hr_zone_4",
            "HR Zone 5 %": "hr_zone_5",
            "GPS enabled": "gps_enabled",
            # Journal
            "Question text": "question_text",
            "Answered yes": "answered_yes",
            "Notes": "notes",
        }
        new_cols = []
        for c in df.columns:
            if c in mapping:
                new_cols.append(mapping[c])
            else:
                # Fallback: snake_case anything we missed
                clean = re.sub(r"[^a-z0-9]+", "_", c.strip().lower()).strip("_")
                new_cols.append(clean)
        df.columns = new_cols
        return df

    @staticmethod
    def _find_column(df: pd.DataFrame, candidates: list) -> Optional[str]:
        """Return the first column name from candidates that exists in df."""
        for c in candidates:
            if c in df.columns:
                return c
        return None


# ── Quick test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/whoop"
    parser = WhoopParser(data_dir)

    try:
        data = parser.load_all()
        for name, df in data.items():
            print(f"\n{'='*60}")
            print(f"{name}: {df.shape[0]} rows × {df.shape[1]} columns")
            print(f"Columns: {list(df.columns[:10])}...")
            print(df.head(2))

        timeline = parser.build_daily_timeline()
        print(f"\nTimeline shape: {timeline.shape}")
    except FileNotFoundError as e:
        print(f"Place your Whoop CSVs in {data_dir}/")
        print(f"  Expected: physiological_cycles.csv, workouts.csv, sleeps.csv, journal_entries.csv")

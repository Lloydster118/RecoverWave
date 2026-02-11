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

    def _load_cycles(self):
        path = self.dir / "physiological_cycles.csv"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path)
        df["cycle_start"] = pd.to_datetime(df["Cycle start time"], utc=True)
        df["cycle_end"] = pd.to_datetime(df["Cycle end time"], utc=True)
        df["recovery_score"] = df["Recovery score %"]
        df["hrv"] = df["Heart rate variability (ms)"]
        df["rhr"] = df["Resting heart rate (bpm)"]
        df["strain"] = df["Day Strain"]
        return df[["cycle_start", "cycle_end", "recovery_score", "hrv", "rhr", "strain"]]

    def load_all(self):
        cycles = self._load_cycles()
        workouts = self._load_workouts()
        sleeps = self._load_sleeps()
        journal = self._load_journal()
        return {"cycles": cycles, "workouts": workouts, "sleeps": sleeps, "journal": journal}

    def _load_workouts(self):
        path = self.dir / "workouts.csv"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path)
        df["workout_start"] = pd.to_datetime(df["Workout start time"], utc=True)
        df["workout_end"] = pd.to_datetime(df["Workout end time"], utc=True)
        df["activity"] = df["Activity name"]
        df["strain"] = df["Activity Strain"]
        df["avg_hr"] = df["Average HR (bpm)"]
        df["max_hr"] = df["Max HR (bpm)"]
        return df[["workout_start", "workout_end", "activity", "strain", "avg_hr", "max_hr"]]

    def _load_sleeps(self):
        path = self.dir / "sleeps.csv"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path)
        df["sleep_start"] = pd.to_datetime(df["Sleep start time"], utc=True)
        df["sleep_end"] = pd.to_datetime(df["Sleep end time"], utc=True)
        df["in_bed_min"] = df["In bed (min)"]
        df["asleep_min"] = df["Asleep duration (min)"]
        df["light_min"] = df["Light sleep duration (min)"]
        df["deep_min"] = df["Deep (SWS) duration (min)"]
        df["rem_min"] = df["REM duration (min)"]
        df["awake_min"] = df["Awake duration (min)"]
        df["efficiency"] = df["Sleep efficiency %"]
        return df[["sleep_start", "sleep_end", "in_bed_min", "asleep_min", "light_min", "deep_min", "rem_min", "awake_min", "efficiency"]]

    def _load_journal(self):
        path = self.dir / "journal_entries.csv"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path)
        df["day"] = pd.to_datetime(df["Day"]).dt.date
        # Pivot question/answer into wide form
        wide = df.pivot_table(index="day", columns="Question", values="Answer", aggfunc="first").reset_index()
        # Coerce Yes/No to bool
        for col in wide.columns:
            if col == "day":
                continue
            if wide[col].dropna().isin(["Yes", "No"]).all():
                wide[col] = wide[col].map({"Yes": True, "No": False})
        return wide

    def build_daily_timeline(self):
        data = self.load_all()
        cycles = data["cycles"].copy()
        # Whoop cycle starts overnight - bucket by the END date so recovery scores align with the day they describe
        cycles["day"] = cycles["cycle_end"].dt.tz_convert("Europe/London").dt.date

        # Workout aggregates per day
        workouts = data["workouts"].copy()
        if len(workouts):
            workouts["day"] = workouts["workout_start"].dt.tz_convert("Europe/London").dt.date
            wk_daily = workouts.groupby("day").agg(workout_count=("activity", "count"),
                                                    workout_strain=("strain", "sum"),
                                                    workout_max_hr=("max_hr", "max")).reset_index()
        else:
            wk_daily = pd.DataFrame(columns=["day", "workout_count", "workout_strain", "workout_max_hr"])

        daily = cycles.merge(wk_daily, on="day", how="left").set_index("day")
        return daily

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

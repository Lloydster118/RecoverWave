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
    def __init__(self, window_hours: float = 2.0):
        """
        Args:
            window_hours: Hours after workout end to capture listening data.
        """
        self.window_hours = window_hours

# ── Quick test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("TemporalAligner ready.")
    print("\nUsage:")
    print("  aligner = TemporalAligner(window_hours=2.0)")
    print("  aligned = aligner.align(workouts_df, listening_df, daily_timeline)")
    print("  sequences, targets, masks = aligner.build_sequences(aligned, listening_df, workouts_df)")

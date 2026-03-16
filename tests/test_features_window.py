import pandas as pd
from datetime import datetime, timedelta
from src.alignment import TemporalAligner


def _make_listening(n=10, start="2025-01-01T18:00:00Z"):
    base = pd.Timestamp(start)
    return pd.DataFrame({
        "played_at": [base + timedelta(minutes=15 * i) for i in range(n)],
        "minutes_played": [3.0] * n,
        "track": [f"t{i}" for i in range(n)],
        "artist": ["a"] * n,
        "uri": [f"spotify:track:{i}" for i in range(n)],
    })


def test_window_picks_tracks_in_two_hours():
    aligner = TemporalAligner()
    listening = _make_listening(n=10)
    workouts = pd.DataFrame({
        "workout_start": [pd.Timestamp("2025-01-01T17:00:00Z")],
        "workout_end": [pd.Timestamp("2025-01-01T18:00:00Z")],
    })
    out = aligner.align_workouts_to_listening(workouts, listening)
    assert out["n_tracks"].iloc[0] == 8  # tracks 0-7 in 2hr window


def test_back_to_back_workouts_assign_to_earlier():
    """Two workouts within four hours of each other (brick session) must
    have their post-workout tracks attributed to the earlier workout, not
    double-counted or assigned to the later one. See notes/journal.md 28 Feb 2026."""
    aligner = TemporalAligner()
    listening = pd.DataFrame({
        "played_at": [
            pd.Timestamp("2025-06-14T08:10:00Z"),  # right after workout A
            pd.Timestamp("2025-06-14T08:30:00Z"),  # still in A's window
            pd.Timestamp("2025-06-14T10:45:00Z"),  # right after workout B
        ],
        "minutes_played": [3.0, 3.0, 3.0],
        "track": ["t1", "t2", "t3"],
        "artist": ["a", "a", "a"],
        "uri": ["spotify:track:1", "spotify:track:2", "spotify:track:3"],
    })
    workouts = pd.DataFrame({
        "workout_start": [
            pd.Timestamp("2025-06-14T07:00:00Z"),
            pd.Timestamp("2025-06-14T10:00:00Z"),
        ],
        "workout_end": [
            pd.Timestamp("2025-06-14T08:00:00Z"),
            pd.Timestamp("2025-06-14T10:30:00Z"),
        ],
    })
    out = aligner.align_workouts_to_listening(workouts, listening)
    # Workout A claims t1 and t2 (within its 2hr post-window before B starts)
    # Workout B claims t3 only
    assert out["n_tracks"].tolist() == [2, 1]

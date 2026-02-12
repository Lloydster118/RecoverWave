import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta
from src.whoop_parser import WhoopParser


def _setup(tmp_path):
    sample_dir = Path("data/whoop/sample")
    for fname, target in [
        ("cycles_sample.csv", "physiological_cycles.csv"),
        ("workouts_sample.csv", "workouts.csv"),
        ("sleeps_sample.csv", "sleeps.csv"),
        ("journal_sample.csv", "journal_entries.csv"),
    ]:
        src = sample_dir / fname
        if src.exists():
            (tmp_path / target).write_text(src.read_text())


def test_loads_cycles(tmp_path):
    _setup(tmp_path)
    parser = WhoopParser(tmp_path)
    cycles = parser._load_cycles()
    assert len(cycles) > 0
    assert "recovery_score" in cycles.columns


def test_loads_workouts(tmp_path):
    _setup(tmp_path)
    parser = WhoopParser(tmp_path)
    wk = parser._load_workouts()
    assert len(wk) > 0
    assert "strain" in wk.columns


def test_daily_timeline_aligns_workouts(tmp_path):
    _setup(tmp_path)
    parser = WhoopParser(tmp_path)
    daily = parser.build_daily_timeline()
    assert "recovery_score" in daily.columns
    assert "workout_count" in daily.columns


def test_dst_transition_lookback(tmp_path):
    """On the BST forward transition (30 March 2025) a cycle ending at 02:30 UTC
    must still be attributed to the prior calendar day under the 04:00 local cut-off,
    not silently dropped by a naive tz_convert."""
    lines = [
        "Cycle start time,Cycle end time,Recovery score %,HRV (ms),Resting HR (bpm),Sleep performance %",
        "2025-03-29T22:00:00Z,2025-03-30T01:30:00Z,62,45,52,80",
        "2025-03-30T22:15:00Z,2025-03-31T02:30:00Z,58,42,53,76",
    ]
    csv = chr(10).join(lines) + chr(10)
    (tmp_path / "physiological_cycles.csv").write_text(csv)
    parser = WhoopParser(tmp_path)
    cycles = parser._load_cycles()
    # Both cycles must survive the parse despite the DST forward jump
    assert len(cycles) == 2
    assert cycles["recovery_score"].tolist() == [62, 58]

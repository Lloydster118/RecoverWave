import pandas as pd
from pathlib import Path
from src.whoop_parser import WhoopParser


def test_loads_sample_cycles(tmp_path):
    sample = Path("data/whoop/sample/cycles_sample.csv").read_text()
    (tmp_path / "physiological_cycles.csv").write_text(sample)
    parser = WhoopParser(tmp_path)
    cycles = parser._load_cycles()
    assert len(cycles) > 0
    assert "recovery_score" in cycles.columns

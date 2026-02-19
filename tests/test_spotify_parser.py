import pandas as pd
from pathlib import Path
from src.spotify_parser import SpotifyParser
from src.synthetic_spotify import generate


def test_parses_synthetic(tmp_path):
    sample = tmp_path / "sample.json"
    generate(days=3, tracks_per_day=5, out=sample)
    parser = SpotifyParser(tmp_path)
    df = parser.load()
    assert len(df) > 0
    assert "ts" in df.columns

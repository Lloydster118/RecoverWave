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



def test_window_selects_two_hours(tmp_path):
    import pandas as pd
    from datetime import datetime, timedelta
    df = pd.DataFrame({
        "played_at": pd.to_datetime([
            "2025-01-01T17:00:00Z",
            "2025-01-01T17:30:00Z",
            "2025-01-01T20:30:00Z",
        ], utc=True),
    })
    parser = SpotifyParser(tmp_path)
    window = parser.get_listening_window(df, anchor=pd.Timestamp("2025-01-01T17:00:00Z"))
    assert len(window) == 2

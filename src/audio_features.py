"""Audio feature extraction - librosa-based.

Loading actual audio is expensive and we do not have direct file access to
Spotify tracks. This approach is unlikely to scale.
"""

import librosa
import numpy as np
from pathlib import Path


def extract_features(audio_path: Path) -> dict:
    y, sr = librosa.load(str(audio_path), mono=True, duration=30.0)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    rms = librosa.feature.rms(y=y).mean()
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    return {"tempo": float(tempo), "energy": float(rms), "centroid": float(centroid)}

"""Audio feature extraction via Reccobeats API.

Reccobeats provides a free endpoint for Spotify-track-id -> audio feature lookup
which restores the feature shape Spotify removed from its public API in 2024.
"""

import json
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Iterable


RECCO_BASE = "https://api.reccobeats.com/v1"
CACHE_PATH = Path("data/processed/audio_feature_cache.json")

"""
audio_features.py — Fetch track-level audio features from ReccoBeats
(replacement for the deprecated Spotify audio-features endpoint).

Why ReccoBeats: Spotify deprecated `/v1/audio-features` on 27 Nov 2024 for new
apps. ReccoBeats exposes the same fields (danceability, energy, valence, tempo,
loudness, key, mode, acousticness, instrumentalness, liveness, speechiness)
keyed by Spotify track ID, free, no API key required.

Flow:
    1. Batch-lookup Spotify track IDs -> ReccoBeats internal IDs (up to 40/call)
    2. Fetch audio-features per ReccoBeats ID (parallelised)
    3. Persist to on-disk cache so we never re-hit the API for the same track
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


BASE = "https://api.reccobeats.com/v1"
BATCH_SIZE = 40
TIMEOUT = 15
CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "audio_features_cache.json"
FEATURE_FIELDS = ("danceability", "energy", "valence", "tempo", "loudness",
                  "key", "mode", "acousticness", "instrumentalness",
                  "liveness", "speechiness")


# ─────────────────────────────────────────────────────────────────────
# Cache
# ─────────────────────────────────────────────────────────────────────
def load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=0))


# ─────────────────────────────────────────────────────────────────────
# Step 1: Spotify ID -> ReccoBeats internal ID
# ─────────────────────────────────────────────────────────────────────
def lookup_reccobeats_ids(spotify_ids: Iterable[str],
                          max_retries: int = 4) -> dict[str, str]:
    """Map Spotify track IDs -> ReccoBeats internal IDs (batched, with retry)."""
    ids = list({s for s in spotify_ids if s})
    mapping: dict[str, str] = {}
    failed_batches: list[list[str]] = []

    def _do_batch(chunk: list[str]) -> tuple[list[str], bool]:
        """Returns (resolved_ids, success). success=False means try again later."""
        url = f"{BASE}/track?ids={','.join(chunk)}"
        try:
            r = requests.get(url, timeout=TIMEOUT)
        except requests.RequestException:
            return [], False
        if r.status_code == 429 or r.status_code >= 500:
            return [], False
        if r.status_code != 200:
            # 4xx: don't retry, some IDs malformed
            return [], True
        resolved = []
        for t in r.json().get("content", []):
            href = t.get("href", "")
            sp_id = href.rsplit("/", 1)[-1] if href else None
            if sp_id:
                mapping[sp_id] = t["id"]
                resolved.append(sp_id)
        return resolved, True

    # Initial pass
    total = len(ids)
    for i in range(0, total, BATCH_SIZE):
        chunk = ids[i:i + BATCH_SIZE]
        _, ok = _do_batch(chunk)
        if not ok:
            failed_batches.append(chunk)
        if (i // BATCH_SIZE) % 10 == 0:
            print(f"    lookup: {i + len(chunk)}/{total}, resolved {len(mapping)}, failed batches {len(failed_batches)}")
        time.sleep(0.15)

    # Retry failed batches with backoff
    for attempt in range(1, max_retries + 1):
        if not failed_batches:
            break
        wait = 2 ** attempt
        print(f"    retry {attempt}: {len(failed_batches)} batches after {wait}s")
        time.sleep(wait)
        still_failed: list[list[str]] = []
        for chunk in failed_batches:
            _, ok = _do_batch(chunk)
            if not ok:
                still_failed.append(chunk)
            time.sleep(0.2)
        failed_batches = still_failed

    return mapping


# ─────────────────────────────────────────────────────────────────────
# Step 2: ReccoBeats ID -> audio features
# ─────────────────────────────────────────────────────────────────────
def fetch_features_for_reccobeats_id(rb_id: str, max_retries: int = 4) -> dict | None:
    for attempt in range(max_retries):
        try:
            r = requests.get(f"{BASE}/track/{rb_id}/audio-features", timeout=TIMEOUT)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None  # genuine miss
            # 429/5xx -> retry
            time.sleep(0.5 * (2 ** attempt))
        except requests.RequestException:
            time.sleep(0.5 * (2 ** attempt))
    return None


def fetch_all_features(
    spotify_ids: Iterable[str],
    max_workers: int = 4,
    progress_every: int = 200,
) -> dict[str, dict]:
    """Return Spotify track ID -> audio feature dict. Uses + updates cache."""
    cache = load_cache()
    want = [s for s in {s for s in spotify_ids if s} if s not in cache]
    print(f"Cache hits: {len(spotify_ids) - len(want)} / {len(spotify_ids)}")
    print(f"Need to fetch: {len(want)}")

    if not want:
        return {s: cache[s] for s in spotify_ids if s in cache}

    # Step 1: resolve to ReccoBeats IDs
    print("Resolving Spotify IDs -> ReccoBeats IDs…")
    sp_to_rb = lookup_reccobeats_ids(want)
    print(f"  Resolved {len(sp_to_rb)} / {len(want)} "
          f"({100 * len(sp_to_rb) / len(want):.1f}% coverage)")

    # Step 2: fetch features (lower concurrency to avoid rate limits)
    print("Fetching audio features…")
    rb_to_sp = {rb: sp for sp, rb in sp_to_rb.items()}
    done = 0
    failed: list[tuple[str, str]] = []  # (sp_id, rb_id) for retry
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_features_for_reccobeats_id, rb): rb
                   for rb in sp_to_rb.values()}
        for fut in as_completed(futures):
            rb = futures[fut]
            sp = rb_to_sp[rb]
            feat = fut.result()
            if feat:
                cache[sp] = {k: feat.get(k) for k in FEATURE_FIELDS}
            else:
                failed.append((sp, rb))
            done += 1
            if done % progress_every == 0:
                print(f"    {done}/{len(sp_to_rb)} (failed so far: {len(failed)})")
                save_cache(cache)

    # Serial retry of failures with backoff
    if failed:
        print(f"  Retrying {len(failed)} failed feature fetches serially…")
        for sp, rb in failed:
            feat = fetch_features_for_reccobeats_id(rb, max_retries=5)
            cache[sp] = {k: feat.get(k) for k in FEATURE_FIELDS} if feat else None
            time.sleep(0.15)

    # Mark unresolved tracks so we don't retry them
    for s in want:
        cache.setdefault(s, None)

    save_cache(cache)
    hits = sum(1 for s in spotify_ids if cache.get(s))
    print(f"Final coverage: {hits} / {len(spotify_ids)} "
          f"({100 * hits / len(spotify_ids):.1f}%)")
    return {s: cache[s] for s in spotify_ids if s in cache}


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    # Run against the post-workout track set
    sys.path.insert(0, str(Path(__file__).parent))
    from spotify_parser import SpotifyParser  # noqa: E402

    ROOT = Path(__file__).resolve().parents[1]
    parser = SpotifyParser(
        str(ROOT / "data/spotify/Spotify Extended Streaming History"))
    parser.load(); parser.clean()

    wk = pd.read_csv(ROOT / "data/processed/whoop_workouts_clean.csv")
    wk["end_utc"] = (pd.to_datetime(wk["workout_end_time"], errors="coerce")
                       .dt.tz_localize("Europe/London", ambiguous="NaT",
                                       nonexistent="NaT")
                       .dt.tz_convert("UTC"))
    wk = wk.dropna(subset=["end_utc"])

    all_tracks: set[str] = set()
    for _, w in wk.iterrows():
        win = parser.get_listening_window(w["end_utc"], window_hours=2.0)
        all_tracks.update(win["track_id"].dropna().unique())

    print(f"Unique tracks in post-workout windows: {len(all_tracks)}")
    features = fetch_all_features(all_tracks)

    # Quick diagnostic
    feat_df = pd.DataFrame({
        sp: f for sp, f in features.items() if f
    }).T
    feat_df.index.name = "track_id"
    out = ROOT / "data/processed/audio_features.csv"
    feat_df.to_csv(out)
    print(f"\nSaved {len(feat_df)} tracks with features → {out}")
    print(feat_df.describe().round(3).T)

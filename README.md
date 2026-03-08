# RecoverWave

**Modelling How Post-Workout Music Listening Patterns Influence Biometric Recovery**

Final year dissertation project for COM6001 (Pathway A).
Supervisor: Dr Shahadate Rezvy

## Research Question

Do the audio characteristics (tempo, energy, valence) of music listened to in
the two-hour window following a workout predict next-day Whoop recovery score?

## Status

- Spotify export requested 18 Jan, still awaiting delivery
- Whoop export to be requested closer to data freeze
- Pipeline development against synthetic data + Kaggle Spotify Audio Features

## Modules

- `src/whoop_parser.py` - parse Whoop CSV exports
- `src/spotify_parser.py` - parse Spotify Extended Streaming History JSONs
- `src/audio_features.py` - audio feature lookup via Reccobeats API
- `src/synthetic_spotify.py` - synthetic listening history generator

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

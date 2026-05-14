# SYNTHETIC Spotify data — placeholder only

Generated: 2026-04-21T19:25:11.895388+00:00
Seed: 20260421
Events: 17211
Files: 4

These files are **fabricated** for pipeline development. They match the real GDPR Streaming_History_Audio schema field-for-field, so the same parser can consume them. When the real Spotify export arrives, delete this directory and replace with the real JSON files.

Every track also appears in `synthetic_audio_features.csv` with the `__synthetic__=True` flag so no synthetic row can be silently mistaken for real data downstream.

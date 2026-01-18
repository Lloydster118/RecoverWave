# Data access plan

## Whoop
- Export available via account settings - up to 12 months
- Format: CSV files (cycles, sleeps, workouts, journal)
- Need to request - typically returns same day

## Spotify
- Request Extended Streaming History at spotify.com/account/privacy
- Takes up to 30 days to arrive
- Format: JSON files, one per ~12 months of history

## Plan B - if data is late or insufficient
- Public datasets:
  - PPG-DaLiA (HRV)
  - Kaggle Spotify Audio Features (130k tracks)
  - GTZAN (1000 tracks, audio features)
- Build synthetic data generator to develop pipeline against

## Status

- 18 Jan: requested Spotify Extended Streaming History via privacy page
- Waiting up to 30 days. Will work on synthetic data + public datasets in the meantime.

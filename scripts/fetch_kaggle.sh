#!/usr/bin/env bash
# Fetch public Kaggle Spotify Audio Features dataset for early prototyping
# while waiting for personal Spotify export.

set -e
mkdir -p data/public/kaggle_spotify
echo "Download SpotifyFeatures.csv manually from:"
echo "  https://www.kaggle.com/datasets/tomigelo/spotify-audio-features"
echo "and place in data/public/kaggle_spotify/"

"""Quick explore of Kaggle Spotify Audio Features dataset.

130k tracks with tempo, energy, valence, danceability etc.
Using this to understand the feature distributions while waiting for
the real Spotify export.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA = Path("data/public/kaggle_spotify/SpotifyFeatures.csv")


def main():
    if not DATA.exists():
        print(f"Place SpotifyFeatures.csv at {DATA} first")
        return
    df = pd.read_csv(DATA)
    print(f"Loaded {len(df):,} tracks across {df['genre'].nunique()} genres")
    print(df[['tempo', 'energy', 'valence', 'danceability']].describe())

    fig, axes = plt.subplots(1, 4, figsize=(16, 3))
    for ax, col in zip(axes, ['tempo', 'energy', 'valence', 'danceability']):
        df[col].hist(bins=40, ax=ax)
        ax.set_title(col)
    fig.tight_layout()
    fig.savefig("artefacts/kaggle_audio_distributions.png", dpi=100)


if __name__ == "__main__":
    main()

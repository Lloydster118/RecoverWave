# RecoverWave

**Modelling How Post-Workout Music Listening Patterns Influence Biometric Recovery**

BSc Final Year Project (COM6001, Pathway A — Software Engineering)
Buckinghamshire New University, 2025–26
Author: Harry Lloyd (22224922) · Supervisor: Dr Shahadate Rezvy

---

## Research Question

> Do the audio characteristics (tempo, energy, valence) of music listened to in the two-hour window following a workout predict next-day Whoop recovery score, and can a sequence model identify optimal "recovery listening" patterns?

## Project Structure

```
recoverwave/
├── src/                      # Core Python modules
│   ├── whoop_parser.py       # Parse and clean Whoop CSV exports
│   ├── spotify_parser.py     # Parse Spotify extended streaming history
│   ├── audio_features.py     # Audio feature extraction + cached lookup
│   ├── alignment.py          # Workout → 2h listening window alignment
│   ├── sleep_features.py     # Sleep architecture features
│   ├── post_workout_features.py
│   ├── build_modelling_dataset.py
│   ├── models.py             # Ridge, GBM, LSTM + Attention, baselines
│   ├── synthetic_spotify.py  # Synthetic Spotify generator (privacy-safe demo)
│   ├── results_figures.py    # Figure generation
│   ├── attention_visualisation.py
│   ├── architecture_diagram.py
│   └── save_artefacts.py
├── app/
│   └── streamlit_app.py      # Streamlit dashboard
├── notebooks/
│   ├── 01_whoop_eda.ipynb    # Exploratory analysis
│   └── 01_whoop_eda.py
├── tests/                    # Unit tests
├── data/                     # Personal data goes here (gitignored)
│   ├── whoop/                # Whoop CSV exports
│   ├── spotify/              # Spotify streaming history JSONs
│   ├── spotify_synthetic/    # Synthetic listening data for demo
│   └── processed/            # Pipeline outputs (gitignored)
├── models/                   # Trained model artefacts (gitignored)
├── artefacts/                # Generated figures and PDFs
├── requirements.txt
├── run_spotify_eda.py        # End-to-end EDA driver
└── README.md
```

## Setup

Tested on Python 3.11 (Linux, macOS, WSL).

```bash
git clone https://github.com/<your-username>/recoverwave.git
cd recoverwave

python -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate            # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

## Data

The project ingests two personal data sources. Neither is committed to the repo.

### 1. Whoop

Settings → My Account → Export Data. You receive a zip containing four CSVs. Place them in `data/whoop/`:

```
data/whoop/
├── physiological_cycles.csv
├── sleeps.csv
├── workouts.csv
└── journal_entries.csv
```

### 2. Spotify

Request your **Extended Streaming History** at <https://www.spotify.com/account/privacy>. Spotify emails the archive within ~30 days. Unzip and place every `Streaming_History_Audio_*.json` under:

```
data/spotify/Spotify Extended Streaming History/
```

If you want to demo the pipeline without real Spotify data, use the synthetic generator (see `data/spotify_synthetic/README_SYNTHETIC.md`).

## Reproducing the Results

The pipeline is deterministic given the seed. Run the modules in order:

```bash
# 1. Parse and clean raw exports
python -m src.whoop_parser
python -m src.spotify_parser

# 2. Audio feature extraction (cached after first run)
python -m src.audio_features

# 3. Temporal alignment + feature engineering
python -m src.alignment
python -m src.sleep_features
python -m src.post_workout_features
python -m src.build_modelling_dataset

# 4. Train and evaluate all models with 5-fold chronological CV
python -m src.models

# 5. Generate figures
python -m src.results_figures
python -m src.attention_visualisation
```

Outputs land in `data/processed/` (CSVs) and `data/processed/figures/` (PNGs).

## Dashboard

```bash
streamlit run app/streamlit_app.py
```

The dashboard surfaces recovery profiles, listening heatmaps, attention weights, and prediction-vs-actual plots over the same dataset.

## Tests

```bash
pytest tests/
```

## Tech Stack

- **Data wrangling**: Python 3.11, pandas, NumPy, SciPy
- **Audio features**: Librosa, Essentia
- **ML**: scikit-learn, XGBoost, LightGBM, PyTorch (LSTM + Bahdanau attention)
- **Statistical tests**: pingouin, statsmodels (mediation)
- **Visualisation**: Matplotlib, Seaborn, Plotly
- **Dashboard**: Streamlit

## Public Datasets Referenced

- [PPG-DaLiA](https://archive.ics.uci.edu/ml/datasets/PPG-DaLiA) — wrist-worn PPG, 15 subjects
- [EmoWear](https://www.nature.com/articles/s41597-024-03429-3) — 49 adults, emotion + physiology
- [Continuous HRV Dataset](https://doi.org/10.6084/m9.figshare.28509740) — 49 people × 4 weeks
- [Kaggle Spotify Audio Features](https://www.kaggle.com/datasets/tomigelo/spotify-audio-features) — 130k tracks
- [GTZAN](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) — 1,000 tracks across 10 genres

## Ethics and Privacy

- All personal data (`data/whoop/`, `data/spotify/`) is gitignored.
- The repository contains code and documentation only.
- A self-assessment ethics form is filed with the dissertation submission.
- The synthetic Spotify generator allows third parties to exercise the pipeline without exposure to personal listening history.

## Citation

If referencing this project, please cite:

> Lloyd, H. (2026) *RecoverWave: Modelling How Post-Workout Music Listening Patterns Influence Biometric Recovery*. BSc dissertation, Buckinghamshire New University.

## License

This project is released under the MIT License — see [LICENSE](LICENSE).

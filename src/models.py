"""
models.py — RecoverWave model suite.

Predicts next-day recovery score (0–100%) from a 7-day rolling window of
biometric, music, and workout features.

Models:
    Persistence       : ŷ_{t+1} = recovery_t                      (null model)
    SeasonalDoW       : ŷ_{t+1} = mean recovery on that weekday   (null #2)
    LinearRegression  : flat-feature linear regression
    GradientBoosting  : sklearn HistGradientBoostingRegressor
    LSTM              : 7-day → 1-day forecast with PyTorch

Evaluation: blocked time-series cross-validation (5 contiguous folds),
            primary metric MAE, secondary R².
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RNG_SEED = 20260504
np.random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────
# Feature config
# ─────────────────────────────────────────────────────────────────────
FEATURES = [
    # Biometric same-day
    "recovery_score", "resting_heart_rate", "hrv_rmssd",
    "skin_temp", "blood_oxygen", "day_strain",
    "sleep_performance", "respiratory_rate", "asleep_duration",
    "sleep_efficiency", "sleep_consistency",
    "rem_duration", "deep_sws_duration", "light_sleep_duration",
    "awake_duration", "sleep_debt",
    # Music aggregates
    "n_tracks", "total_listen_minutes", "unique_artists",
    "tempo_mean", "valence_mean", "energy_mean",
    "danceability_mean", "instrumentalness_mean",
    "loudness_mean", "mood_diversity",
    # Workout signals
    "had_workout", "pw_n_tracks", "pw_listen_minutes",
    "pw_tempo_mean", "pw_valence_mean", "pw_energy_mean",
    "pw_instrumentalness_mean", "pw_activity_strain",
    # Lags
    "recovery_lag1", "recovery_lag2",
]

TARGET = "next_day_recovery"
SEQ_LEN = 7


# ─────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────


def load_modelling_data() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data/processed/modelling_dataset.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    keep = ["date", TARGET] + [c for c in FEATURES if c in df.columns]
    df = df[keep].copy()
    # Median-impute missing values then forward-fill any remaining gaps
    for c in [c for c in df.columns if c not in {"date", TARGET}]:
        df[c] = df[c].fillna(df[c].median())
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    return df


def make_blocked_folds(n: int, n_folds: int = 5):
    """Yields (train_idx, test_idx) for purely chronological blocks.

    Fold k uses [0..start_test) for train and [start_test..end_test) for test.
    No leakage: test always strictly follows train.
    """
    test_size = n // (n_folds + 1)
    for k in range(n_folds):
        start_test = (k + 1) * test_size
        end_test = min(start_test + test_size, n)
        if end_test - start_test < 30:
            continue
        train_idx = np.arange(0, start_test)
        test_idx = np.arange(start_test, end_test)
        yield k, train_idx, test_idx


def predict_persistence(df: pd.DataFrame, train_idx, test_idx) -> np.ndarray:
    return df.loc[test_idx, "recovery_score"].to_numpy()


def predict_seasonal_dow(df: pd.DataFrame, train_idx, test_idx) -> np.ndarray:
    train = df.iloc[train_idx].copy()
    train["dow"] = train["date"].dt.dayofweek
    means = train.groupby("dow")[TARGET].mean()
    test = df.iloc[test_idx].copy()
    test["dow"] = test["date"].dt.dayofweek
    return test["dow"].map(means).fillna(train[TARGET].mean()).to_numpy()


@dataclass
class FoldResult:
    fold: int
    model: str
    n_test: int
    mae: float
    rmse: float
    r2: float


def _score(fold: int, name: str, y_true, y_pred) -> FoldResult:
    return FoldResult(
        fold=fold, model=name, n_test=len(y_true),
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        r2=float(r2_score(y_true, y_pred)),
    )


def main():
    df = load_modelling_data()
    n_features = 5  # ridge only uses a small initial feature set
    X = df[["hrv", "rhr", "strain", "n_tracks", "minutes_listened"]].fillna(0).values
    y = df["next_day_recovery"].values

    folds = make_blocked_folds(len(df))
    results = []
    for fi, (tr, te) in enumerate(folds):
        # Persistence
        y_pred = predict_persistence(df, tr, te)
        results.append(_score(fi, "persistence", y[te], y_pred))
        # Seasonal DoW
        y_pred = predict_seasonal_dow(df, tr, te)
        results.append(_score(fi, "seasonal_dow", y[te], y_pred))
        # Ridge
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(X[tr])
        m = Ridge(alpha=1.0).fit(sc.transform(X[tr]), y[tr])
        y_pred = m.predict(sc.transform(X[te]))
        results.append(_score(fi, "ridge", y[te], y_pred))

    out = pd.DataFrame([r.__dict__ for r in results])
    print(out.groupby("model")[["mae", "r"]].agg(["mean", "std"]))
    out.to_csv("data/processed/cv_results.csv", index=False)

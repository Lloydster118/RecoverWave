"""
save_artefacts.py — train final models on full data and persist them
for the Streamlit dashboard.

Saves:
    artefacts/scaler.pkl                StandardScaler fit on full dataset
    artefacts/ridge.pkl                 Ridge regression
    artefacts/gbm.pkl                   HistGradientBoostingRegressor
    artefacts/lstm.pt                   PyTorch LSTM weights
    artefacts/feature_columns.json      ordered list of feature names used at inference
    artefacts/lstm_config.json          architecture hyper-parameters
    artefacts/training_summary.json     metadata + CV summary
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from models import (DEVICE, FEATURES, RNG_SEED, RecoveryLSTM, SEQ_LEN, TARGET,
                    build_sequences, load_modelling_data, train_lstm)

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artefacts"
ART.mkdir(exist_ok=True)

print("Loading dataset…")
df = load_modelling_data()
feat_cols = [c for c in FEATURES if c in df.columns]
X = df[feat_cols].to_numpy(dtype="float32")
y = df[TARGET].to_numpy(dtype="float32")
print(f"  {len(df)} rows, {len(feat_cols)} features")

print("Fitting scaler on full dataset…")
scaler = StandardScaler().fit(X)
Xs = scaler.transform(X)

print("Training Ridge…")
ridge = Ridge(alpha=1.0, random_state=RNG_SEED).fit(Xs, y)

print("Training Gradient Boosting…")
gbm = HistGradientBoostingRegressor(
    max_iter=300, max_depth=5, learning_rate=0.05,
    random_state=RNG_SEED).fit(X, y)

print("Training LSTM (with last 15 % held out for early stopping)…")
val_size = max(30, int(0.15 * len(Xs)))
tr_end = len(Xs) - val_size
X_tr_seq, y_tr_seq = build_sequences(Xs[:tr_end], y[:tr_end])
X_va_seq, y_va_seq = build_sequences(Xs[tr_end - SEQ_LEN:], y[tr_end - SEQ_LEN:])
lstm = train_lstm(X_tr_seq, y_tr_seq, X_va_seq, y_va_seq,
                  n_features=Xs.shape[1])

print("Persisting artefacts…")
with open(ART / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open(ART / "ridge.pkl", "wb") as f:
    pickle.dump(ridge, f)
with open(ART / "gbm.pkl", "wb") as f:
    pickle.dump(gbm, f)
torch.save(lstm.state_dict(), ART / "lstm.pt")

with open(ART / "feature_columns.json", "w") as f:
    json.dump(feat_cols, f, indent=2)

with open(ART / "lstm_config.json", "w") as f:
    json.dump({
        "n_features": Xs.shape[1],
        "hidden": 64,
        "num_layers": 2,
        "dropout": 0.2,
        "seq_len": SEQ_LEN,
        "device": DEVICE,
    }, f, indent=2)

# Add CV summary into the metadata bundle
summary = pd.read_csv(ROOT / "data/processed/cv_summary.csv")
with open(ART / "training_summary.json", "w") as f:
    json.dump({
        "rng_seed": RNG_SEED,
        "n_samples": len(df),
        "n_features": len(feat_cols),
        "date_range": [str(df.date.min().date()), str(df.date.max().date())],
        "target": TARGET,
        "primary_metric": "MAE",
        "cv_summary": summary.to_dict(orient="records"),
    }, f, indent=2)

print("Saved:")
for p in sorted(ART.iterdir()):
    print(f"  {p.name}  ({p.stat().st_size:,} bytes)")

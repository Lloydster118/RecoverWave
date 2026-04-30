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


# ─────────────────────────────────────────────────────────────────────
# Baselines
# ─────────────────────────────────────────────────────────────────────
def predict_persistence(df: pd.DataFrame, train_idx, test_idx) -> np.ndarray:
    return df.loc[test_idx, "recovery_score"].to_numpy()


def predict_seasonal_dow(df: pd.DataFrame, train_idx, test_idx) -> np.ndarray:
    train = df.iloc[train_idx].copy()
    train["dow"] = train["date"].dt.dayofweek
    means = train.groupby("dow")[TARGET].mean()
    test = df.iloc[test_idx].copy()
    test["dow"] = test["date"].dt.dayofweek
    return test["dow"].map(means).fillna(train[TARGET].mean()).to_numpy()


# ─────────────────────────────────────────────────────────────────────
# LSTM
# ─────────────────────────────────────────────────────────────────────
class AdditiveAttention(nn.Module):
    """Bahdanau-style additive attention over the LSTM time axis.

    Given an LSTM output sequence H ∈ ℝ^{B × T × D} this layer learns a
    scalar score e_t = vᵀ tanh(W_h h_t + b) for each step t, normalises
    them with softmax over time, and returns the convex combination
    c = Σ_t α_t h_t together with the attention weights α.
    """

    def __init__(self, hidden: int):
        super().__init__()
        self.W = nn.Linear(hidden, hidden, bias=True)
        self.v = nn.Linear(hidden, 1, bias=False)

    def forward(self, H):  # H: (B, T, D)
        scores = self.v(torch.tanh(self.W(H))).squeeze(-1)  # (B, T)
        weights = torch.softmax(scores, dim=1)              # (B, T)
        context = torch.bmm(weights.unsqueeze(1), H).squeeze(1)  # (B, D)
        return context, weights


class RecoveryLSTM(nn.Module):
    """Sequence-to-one regressor: 7-day window → next-day recovery.

    Architecture: 2-layer LSTM → additive attention pooling → MLP head.
    """

    def __init__(self, n_features: int, hidden: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden,
                            num_layers=num_layers, dropout=dropout,
                            batch_first=True)
        self.attn = AdditiveAttention(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x, return_weights: bool = False):
        out, _ = self.lstm(x)
        context, weights = self.attn(out)
        y = self.head(context).squeeze(-1)
        if return_weights:
            return y, weights
        return y


def build_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = SEQ_LEN):
    """Slide a window of length seq_len; target is the day immediately after the window."""
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.array(Xs, dtype="float32"), np.array(ys, dtype="float32")


def train_lstm(X_train, y_train, X_val, y_val,
               n_features: int, epochs: int = 60, lr: float = 1e-3,
               batch_size: int = 32, patience: int = 8):
    model = RecoveryLSTM(n_features).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.L1Loss()  # train on MAE directly

    Xt = torch.tensor(X_train).to(DEVICE)
    yt = torch.tensor(y_train).to(DEVICE)
    Xv = torch.tensor(X_val).to(DEVICE)
    yv = torch.tensor(y_val).to(DEVICE)

    best_val = float("inf"); best_state = None; bad = 0
    n = len(Xt)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            pred = model(Xt[idx])
            loss = loss_fn(pred, yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v_pred = model(Xv)
            v_loss = loss_fn(v_pred, yv).item()
        if v_loss < best_val - 1e-3:
            best_val = v_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ─────────────────────────────────────────────────────────────────────
# Main CV loop
# ─────────────────────────────────────────────────────────────────────
@dataclass
class FoldResult:
    fold: int
    model: str
    n_test: int
    mae: float
    rmse: float
    r2: float


def main():
    print("Loading modelling dataset…")
    df = load_modelling_data()
    print(f"  rows: {len(df)}    date: {df.date.min().date()} → {df.date.max().date()}")
    feat_cols = [c for c in FEATURES if c in df.columns]
    print(f"  features: {len(feat_cols)}")

    X_all = df[feat_cols].to_numpy(dtype="float32")
    y_all = df[TARGET].to_numpy(dtype="float32")

    all_results: list[FoldResult] = []
    all_preds: list[pd.DataFrame] = []

    for k, tr, te in make_blocked_folds(len(df), n_folds=5):
        date_tr = (df.date.iloc[tr].min().date(), df.date.iloc[tr].max().date())
        date_te = (df.date.iloc[te].min().date(), df.date.iloc[te].max().date())
        print(f"\n── Fold {k}  train {date_tr[0]}\u2192{date_tr[1]} ({len(tr)})  "
              f"test {date_te[0]}\u2192{date_te[1]} ({len(te)})")

        y_test = y_all[te]
        fold_preds = {"date": df.date.iloc[te].dt.date.to_numpy(),
                      "actual": y_test}

        # Persistence
        p = predict_persistence(df, tr, te)
        all_results.append(_score(k, "Persistence", y_test, p))
        fold_preds["Persistence"] = p

        # Seasonal day-of-week
        p = predict_seasonal_dow(df, tr, te)
        all_results.append(_score(k, "SeasonalDoW", y_test, p))
        fold_preds["SeasonalDoW"] = p

        # Scale features (fit on train only)
        scaler = StandardScaler().fit(X_all[tr])
        Xtr = scaler.transform(X_all[tr])
        Xte = scaler.transform(X_all[te])

        # Ridge regression
        ridge = Ridge(alpha=1.0, random_state=RNG_SEED).fit(Xtr, y_all[tr])
        p = ridge.predict(Xte)
        all_results.append(_score(k, "Ridge", y_test, p))
        fold_preds["Ridge"] = p

        # Gradient Boosting
        gbm = HistGradientBoostingRegressor(
            max_iter=300, max_depth=5, learning_rate=0.05,
            random_state=RNG_SEED).fit(X_all[tr], y_all[tr])
        p = gbm.predict(X_all[te])
        all_results.append(_score(k, "GBM", y_test, p))
        fold_preds["GBM"] = p

        # LSTM (uses scaled features, sequence-windowed)
        # Carve a small validation slice from the END of the training window
        val_size = max(30, int(0.15 * len(tr)))
        tr_inner_end = len(tr) - val_size
        Xs_tr_seq, ys_tr_seq = build_sequences(Xtr[:tr_inner_end],
                                               y_all[tr][:tr_inner_end])
        Xs_val_seq, ys_val_seq = build_sequences(Xtr[tr_inner_end - SEQ_LEN:],
                                                  y_all[tr][tr_inner_end - SEQ_LEN:])
        # Build test sequences using the LAST SEQ_LEN train days as warm-up
        test_input = np.concatenate([Xtr[-SEQ_LEN:], Xte], axis=0)
        test_target = np.concatenate([y_all[tr][-SEQ_LEN:], y_test], axis=0)
        Xs_te_seq, ys_te_seq = build_sequences(test_input, test_target)

        model = train_lstm(Xs_tr_seq, ys_tr_seq,
                           Xs_val_seq, ys_val_seq, n_features=Xtr.shape[1])
        model.eval()
        with torch.no_grad():
            p = model(torch.tensor(Xs_te_seq).to(DEVICE)).cpu().numpy()
        all_results.append(_score(k, "LSTM", y_test, p))
        fold_preds["LSTM"] = p

        all_preds.append(pd.DataFrame(fold_preds).assign(fold=k))

    # ──────────────────────────────────────────────────────────────
    # Save and summarise
    # ──────────────────────────────────────────────────────────────
    res_df = pd.DataFrame([r.__dict__ for r in all_results])
    res_path = ROOT / "data/processed/cv_results.csv"
    res_df.to_csv(res_path, index=False)

    summary = (res_df.groupby("model")
                     .agg(mae_mean=("mae", "mean"), mae_std=("mae", "std"),
                          rmse_mean=("rmse", "mean"),
                          r2_mean=("r2", "mean"), r2_std=("r2", "std"))
                     .sort_values("mae_mean"))
    sum_path = ROOT / "data/processed/cv_summary.csv"
    summary.to_csv(sum_path)

    preds = pd.concat(all_preds, ignore_index=True)
    pred_path = ROOT / "data/processed/cv_predictions.csv"
    preds.to_csv(pred_path, index=False)

    print("\n============================================================")
    print("CROSS-VALIDATION SUMMARY  (5 chronological folds)")
    print("============================================================")
    print(summary.round(3).to_string())
    print(f"\nSaved fold-level results → {res_path}")
    print(f"Saved per-model summary → {sum_path}")
    print(f"Saved predictions      → {pred_path}")


def _score(fold: int, name: str, y_true, y_pred) -> FoldResult:
    return FoldResult(
        fold=fold, model=name, n_test=len(y_true),
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        r2=float(r2_score(y_true, y_pred)),
    )


if __name__ == "__main__":
    main()

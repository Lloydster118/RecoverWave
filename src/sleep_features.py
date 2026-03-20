"""
sleep_features.py — Add post-workout sleep features to the workout-feature table
to test the indirect pathway: music → sleep timing/quality → next-day recovery.

For each workout we attach:
  - bedtime_local            (clock time of sleep onset, hours past noon — wraps cleanly)
  - bedtime_delta_personal   (offset vs the participant's median bedtime, hours)
  - sleep_onset_lag_h        (hours from workout end to sleep onset)
  - sleep_duration_h         (asleep duration of the post-workout night)
  - sleep_efficiency         (% from Whoop)
  - sleep_performance        (% from Whoop)
  - sleep_consistency        (% from Whoop)
  - rem_minutes / deep_minutes / light_minutes / awake_minutes
  - is_late_bedtime          (1 if bedtime later than personal 75th percentile)

Then re-runs correlations against next-day recovery.

Outputs:
  data/processed/workout_features_with_sleep.csv
  data/processed/correlations_with_sleep.csv
  figures/fig_14_sleep_correlations.png
  figures/fig_15_indirect_pathway.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]


# ─────────────────────────────────────────────────────────────────────
# Style (matches earlier figures)
# ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.family": "DejaVu Sans",
    "font.size": 10, "axes.edgecolor": "#28251D", "axes.labelcolor": "#28251D",
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})
ACCENT = "#01696F"
ACCENT2 = "#A84B2F"
MUTED = "#7A7974"


# ─────────────────────────────────────────────────────────────────────
# 1. Load and clean the sleeps table
# ─────────────────────────────────────────────────────────────────────


def load_sleeps() -> pd.DataFrame:
    sl = pd.read_csv(ROOT / "data/whoop/sleeps.csv")
    # Drop naps — only main sleeps are relevant for recovery
    if "Nap" in sl.columns:
        sl = sl[sl["Nap"].astype(str).str.lower() != "true"].copy()

    # Parse timestamps — Whoop stores UTC ISO strings
    sl["sleep_onset"] = pd.to_datetime(sl["Sleep onset"], utc=True, errors="coerce")
    sl["wake_onset"] = pd.to_datetime(sl["Wake onset"], utc=True, errors="coerce")
    sl = sl.dropna(subset=["sleep_onset", "wake_onset"]).reset_index(drop=True)

    # Convert to local clock for bedtime feature
    sl["sleep_onset_local"] = sl["sleep_onset"].dt.tz_convert("Europe/London")
    # Express bedtime as decimal hours past midday (so 23:00 -> 11.0,
    # 02:00 -> 14.0). Wraps cleanly across midnight.
    h = sl["sleep_onset_local"].dt.hour + sl["sleep_onset_local"].dt.minute / 60
    sl["bedtime_hours_past_noon"] = (h - 12.0) % 24

    # Personal-norm reference
    median_bed = sl["bedtime_hours_past_noon"].median()
    p75_bed = sl["bedtime_hours_past_noon"].quantile(0.75)
    sl["bedtime_delta_personal"] = sl["bedtime_hours_past_noon"] - median_bed
    sl["is_late_bedtime"] = (sl["bedtime_hours_past_noon"] > p75_bed).astype(int)

    # Rename for cleanliness
    sl = sl.rename(columns={
        "Asleep duration (min)": "asleep_min",
        "In bed duration (min)": "in_bed_min",
        "Sleep efficiency %": "sleep_efficiency",
        "Sleep performance %": "sleep_performance",
        "Sleep consistency %": "sleep_consistency",
        "Light sleep duration (min)": "light_min",
        "Deep (SWS) duration (min)": "deep_min",
        "REM duration (min)": "rem_min",
        "Awake duration (min)": "awake_min",
        "Respiratory rate (rpm)": "respiratory_rate",
    })
    sl["sleep_duration_h"] = sl["asleep_min"] / 60
    return sl


def attach_post_workout_sleep(workouts: pd.DataFrame, sleeps: pd.DataFrame,
                              max_lag_h: float = 24.0) -> pd.DataFrame:
    """For each workout end, find the first sleep_onset within `max_lag_h` hours."""
    sleeps_sorted = sleeps.sort_values("sleep_onset").reset_index(drop=True)
    # Convert tz-aware datetimes to UTC-naive ns since epoch (uniform unit)
    onset_ns = (sleeps_sorted["sleep_onset"].dt.tz_convert("UTC")
                                              .dt.tz_localize(None)
                                              .astype("datetime64[ns]")
                                              .astype("int64")
                                              .to_numpy())

    rows = []
    for _, w in workouts.iterrows():
        end = pd.Timestamp(w["workout_end_utc"])
        if end.tz is None:
            end = end.tz_localize("UTC")
        end_ns = end.tz_convert("UTC").tz_localize(None).value
        idx = np.searchsorted(onset_ns, end_ns, side="right")
        if idx >= len(sleeps_sorted):
            rows.append(None); continue
        sl = sleeps_sorted.iloc[idx]
        lag_h = (sl["sleep_onset"] - end).total_seconds() / 3600
        if lag_h > max_lag_h:
            rows.append(None); continue
        rows.append({
            "sleep_onset_lag_h": lag_h,
            "bedtime_hours_past_noon": sl["bedtime_hours_past_noon"],
            "bedtime_delta_personal": sl["bedtime_delta_personal"],
            "is_late_bedtime": sl["is_late_bedtime"],
            "sleep_duration_h": sl["sleep_duration_h"],
            "sleep_efficiency": sl["sleep_efficiency"],
            "sleep_performance": sl["sleep_performance"],
            "sleep_consistency": sl["sleep_consistency"],
            "rem_min": sl["rem_min"],
            "deep_min": sl["deep_min"],
            "light_min": sl["light_min"],
            "awake_min": sl["awake_min"],
            "respiratory_rate": sl["respiratory_rate"],
        })

    # Replace None entries with empty dicts so DataFrame builds with NaN rows
    rows = [r if r is not None else {} for r in rows]
    sleep_df = pd.DataFrame(rows)
    return pd.concat([workouts.reset_index(drop=True),
                      sleep_df.reset_index(drop=True)], axis=1)


def correlate(df: pd.DataFrame, target: str = "next_day_recovery") -> pd.DataFrame:
    trainable = df.dropna(subset=[target]).copy()
    candidates = [c for c in trainable.columns
                  if c not in {target, "workout_end_utc", "workout_end_local",
                               "activity"}
                  and trainable[c].dtype != "O"]
    out = []
    for c in candidates:
        pair = trainable[[c, target]].dropna()
        if len(pair) < 20:
            continue
        r, p = stats.pearsonr(pair[c], pair[target])
        rho, p_s = stats.spearmanr(pair[c], pair[target])
        out.append({"feature": c, "n": len(pair),
                    "pearson_r": r, "pearson_p": p,
                    "spearman_r": rho, "spearman_p": p_s})
    return pd.DataFrame(out).sort_values("pearson_r", key=abs, ascending=False)


def mediation_snapshot(df: pd.DataFrame) -> dict:
    """
    Quick Baron & Kenny–style check for whether sleep mediates the link
    between post-workout listening behaviour and recovery.

    Predictor (X): n_tracks (the strongest 'music quantity' signal)
    Mediator (M): sleep_duration_h
    Outcome (Y): next_day_recovery

    Reports:
      a    : X -> M
      b    : M -> Y (controlling for X)  [via partial correlation]
      c    : X -> Y (total effect)
      c'   : X -> Y controlling for M     [direct effect]
    """
    import statsmodels.api as sm
    use = df.dropna(subset=["n_tracks", "sleep_duration_h", "next_day_recovery"]).copy()
    if len(use) < 30:
        return {"error": "too few rows for mediation"}

    X = sm.add_constant(use["n_tracks"])
    a_model = sm.OLS(use["sleep_duration_h"], X).fit()
    a = a_model.params["n_tracks"]

    Xy = sm.add_constant(use["n_tracks"])
    c_model = sm.OLS(use["next_day_recovery"], Xy).fit()
    c = c_model.params["n_tracks"]

    XM = sm.add_constant(use[["n_tracks", "sleep_duration_h"]])
    cprime_model = sm.OLS(use["next_day_recovery"], XM).fit()
    cprime = cprime_model.params["n_tracks"]
    b = cprime_model.params["sleep_duration_h"]

    indirect = a * b
    proportion_mediated = (indirect / c) if abs(c) > 1e-6 else np.nan

    return {
        "n": len(use),
        "a (X->M)": a, "b (M->Y|X)": b,
        "c (total X->Y)": c, "c' (direct X->Y|M)": cprime,
        "indirect a*b": indirect,
        "proportion mediated": proportion_mediated,
    }


if __name__ == "__main__":
    main()

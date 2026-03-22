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


# ─────────────────────────────────────────────────────────────────────
# 2. For each workout, find the next sleep onset
# ─────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────
# 3. Re-run correlations
# ─────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────
# 4. Mediation snapshot — does sleep mediate music → recovery?
# ─────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────
# 5. Figures
# ─────────────────────────────────────────────────────────────────────
def plot_sleep_correlations(res: pd.DataFrame, out_path: Path):
    """Two-panel ranked-bar with right-side r/p/n table (same pattern as fig_13)."""
    top = res.head(20).reset_index(drop=True)
    n = len(top)
    fig, (ax_bar, ax_tbl) = plt.subplots(
        1, 2, figsize=(12, 7),
        gridspec_kw={"width_ratios": [3, 1.3], "wspace": 0.05},
    )
    y = np.arange(n)[::-1]
    colors = [ACCENT if r >= 0 else ACCENT2 for r in top["pearson_r"]]
    ax_bar.barh(y, top["pearson_r"], color=colors, alpha=0.88)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(top["feature"].str.replace("_", " "), fontsize=10)
    ax_bar.axvline(0, color="#28251D", lw=0.6)
    ax_bar.set_xlabel("Pearson r (vs next-day recovery)")
    ax_bar.set_title(
        "Combined music + sleep features ranked by |r| with next-day recovery",
        pad=12, loc="left",
    )
    xmax = max(abs(top["pearson_r"].max()), abs(top["pearson_r"].min())) * 1.15
    ax_bar.set_xlim(-xmax, xmax)

    ax_tbl.set_xlim(0, 1); ax_tbl.set_ylim(ax_bar.get_ylim()); ax_tbl.axis("off")
    ax_tbl.text(0.08, n - 0.5, "r", fontsize=10, fontweight="bold", color="#28251D")
    ax_tbl.text(0.45, n - 0.5, "p", fontsize=10, fontweight="bold", color="#28251D")
    ax_tbl.text(0.80, n - 0.5, "n", fontsize=10, fontweight="bold", color="#28251D")
    for i, row in top.iterrows():
        yi = n - 1 - i
        sig = "*" if row["pearson_p"] < 0.05 else " "
        ax_tbl.text(0.08, yi, f"{row['pearson_r']:+.3f}", fontsize=10, va="center",
                    color="#28251D",
                    fontweight="bold" if row["pearson_p"] < 0.05 else "normal")
        ax_tbl.text(0.45, yi, f"{row['pearson_p']:.3f}{sig}", fontsize=10,
                    va="center", color="#28251D")
        ax_tbl.text(0.80, yi, f"{int(row['n'])}", fontsize=10, va="center", color=MUTED)

    fig.text(0.5, 0.02,
             "* p < 0.05    |    Teal = positive correlation, red = negative",
             ha="center", fontsize=9, color=MUTED)
    plt.savefig(out_path, bbox_inches="tight"); plt.close()


def plot_indirect_pathway(df: pd.DataFrame, mediation: dict, out_path: Path):
    """Three scatter panels: music→sleep, sleep→recovery, music→recovery (raw + adjusted)."""
    use = df.dropna(subset=["n_tracks", "sleep_duration_h", "next_day_recovery"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))

    pairs = [
        (axes[0], use["n_tracks"], use["sleep_duration_h"],
         "Post-workout n_tracks", "Sleep duration (h)",
         f"a path: music \u2192 sleep\nslope = {mediation['a (X->M)']:+.4f} h / track"),
        (axes[1], use["sleep_duration_h"], use["next_day_recovery"],
         "Sleep duration (h)", "Next-day recovery %",
         f"b path: sleep \u2192 recovery\nslope = {mediation['b (M->Y|X)']:+.2f} % / hour"),
        (axes[2], use["n_tracks"], use["next_day_recovery"],
         "Post-workout n_tracks", "Next-day recovery %",
         f"c path: music \u2192 recovery (total)\nslope = {mediation['c (total X->Y)']:+.3f} % / track"),
    ]
    for ax, x, y, xlab, ylab, title in pairs:
        ax.scatter(x, y, s=14, alpha=0.35, color=ACCENT)
        slope, intercept, r, p, _ = stats.linregress(x, y)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, intercept + slope * xs, color=ACCENT2, lw=1.6)
        ax.set_xlabel(xlab); ax.set_ylabel(ylab)
        ax.set_title(f"{title}\nr = {r:+.3f}, p = {p:.3f}", fontsize=10)

    pct = mediation["proportion mediated"]
    pct_str = f"{pct:.1%}" if not np.isnan(pct) else "n/a"
    fig.suptitle(
        "Indirect-pathway test: does sleep duration mediate "
        f"music \u2192 recovery? (proportion mediated \u2248 {pct_str}, n = {mediation['n']})",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight"); plt.close()


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    print("Loading workout features and sleeps…")
    wkf = pd.read_csv(ROOT / "data/processed/workout_features.csv")
    wkf["workout_end_utc"] = pd.to_datetime(wkf["workout_end_utc"], utc=True,
                                            errors="coerce")
    sleeps = load_sleeps()
    print(f"  Workouts: {len(wkf)}")
    print(f"  Main sleeps (naps removed): {len(sleeps)}")

    print("\nMatching workouts to next sleep…")
    merged = attach_post_workout_sleep(wkf, sleeps)
    matched = merged["sleep_duration_h"].notna().sum()
    print(f"  Workouts with a matched post-workout sleep: {matched} / {len(merged)}")

    out_csv = ROOT / "data/processed/workout_features_with_sleep.csv"
    merged.to_csv(out_csv, index=False)
    print(f"Saved → {out_csv}")

    print("\nRunning correlations (music + sleep features)…")
    res = correlate(merged)
    res_csv = ROOT / "data/processed/correlations_with_sleep.csv"
    res.to_csv(res_csv, index=False)
    print(f"Saved → {res_csv}")

    print("\n=== Top 20 features by |Pearson r| vs next-day recovery ===")
    print(res.head(20).to_string(
        index=False,
        formatters={"pearson_r": "{:+.3f}".format, "pearson_p": "{:.3f}".format,
                    "spearman_r": "{:+.3f}".format, "spearman_p": "{:.3f}".format},
    ))

    print("\nMediation snapshot (music \u2192 sleep \u2192 recovery)…")
    med = mediation_snapshot(merged)
    for k, v in med.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\nGenerating figures…")
    FIG = ROOT / "data/processed/figures"
    plot_sleep_correlations(res, FIG / "fig_14_sleep_correlations.png")
    print("  fig_14_sleep_correlations.png")
    plot_indirect_pathway(merged, med, FIG / "fig_15_indirect_pathway.png")
    print("  fig_15_indirect_pathway.png")


if __name__ == "__main__":
    main()

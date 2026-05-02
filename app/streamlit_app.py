"""
RecoverWave — Streamlit Dashboard

Interactive dashboard for exploring the relationship between
post-workout music listening and biometric recovery.

Run with: streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.whoop_parser import WhoopParser
from src.spotify_parser import SpotifyParser
from src.alignment import TemporalAligner

# ── Page config ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="RecoverWave",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("RecoverWave")
st.markdown(
    "**Modelling how post-workout music listening patterns "
    "influence biometric recovery**"
)

# ── Sidebar: Data Upload ──────────────────────────────────────────────

st.sidebar.header("Data Sources")

st.sidebar.subheader("1. Whoop Data")
whoop_dir = st.sidebar.text_input(
    "Whoop CSV directory",
    value="data/whoop",
    help="Path to folder containing your Whoop CSV exports "
         "(physiological_cycles.csv, workouts.csv, sleeps.csv, journal_entries.csv)",
)

st.sidebar.subheader("2. Spotify Data")
spotify_dir = st.sidebar.text_input(
    "Spotify JSON directory",
    value="data/spotify",
    help="Path to folder containing your Spotify extended streaming history JSONs",
)

st.sidebar.subheader("3. Audio Features")
features_path = st.sidebar.text_input(
    "Pre-computed features CSV (optional)",
    value="data/public/SpotifyAudioFeatures.csv",
    help="Kaggle Spotify Audio Features dataset for BPM/energy/valence lookup",
)

window_hours = st.sidebar.slider(
    "Post-workout listening window (hours)",
    min_value=0.5, max_value=6.0, value=2.0, step=0.5,
)

# ── Load Data ─────────────────────────────────────────────────────────

@st.cache_data
def load_whoop(data_dir):
    parser = WhoopParser(data_dir)
    data = parser.load_all()
    timeline = parser.build_daily_timeline()
    return data, timeline

@st.cache_data
def load_spotify(data_dir):
    parser = SpotifyParser(data_dir)
    parser.load()
    return parser.clean()

load_data = st.sidebar.button("Load Data", type="primary")

if load_data or "whoop_data" in st.session_state:
    try:
        if load_data:
            with st.spinner("Loading Whoop data..."):
                whoop_data, timeline = load_whoop(whoop_dir)
                st.session_state["whoop_data"] = whoop_data
                st.session_state["timeline"] = timeline

            with st.spinner("Loading Spotify data..."):
                listening = load_spotify(spotify_dir)
                st.session_state["listening"] = listening

        whoop_data = st.session_state.get("whoop_data", {})
        timeline = st.session_state.get("timeline", pd.DataFrame())
        listening = st.session_state.get("listening", pd.DataFrame())

        # ── Overview Tab ──────────────────────────────────────────

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Overview", "Alignment", "Model Results", "Attention Viz", "Recovery Profile"
        ])

        with tab1:
            st.header("Data Overview")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Whoop Days", len(timeline))
            with col2:
                st.metric("Workouts", len(whoop_data.get("workouts", [])))
            with col3:
                st.metric("Spotify Listens", len(listening))
            with col4:
                if "recovery_score" in timeline.columns:
                    avg_recovery = timeline["recovery_score"].mean()
                    st.metric("Avg Recovery", f"{avg_recovery:.0f}%")

            # Recovery score over time
            if "date" in timeline.columns and "recovery_score" in timeline.columns:
                st.subheader("Recovery Score Over Time")
                fig = px.line(
                    timeline, x="date", y="recovery_score",
                    title="Daily Recovery Score",
                    labels={"recovery_score": "Recovery %", "date": "Date"},
                )
                fig.add_hline(y=67, line_dash="dash", line_color="green",
                              annotation_text="Green zone (67%+)")
                fig.add_hline(y=33, line_dash="dash", line_color="red",
                              annotation_text="Red zone (<33%)")
                st.plotly_chart(fig, use_container_width=True)

            # Listening activity over time
            if len(listening) > 0:
                st.subheader("Daily Listening Activity")
                daily_listens = listening.groupby("date").agg(
                    n_tracks=("track_name", "count"),
                    total_minutes=("listen_seconds", lambda x: x.sum()/60),
                ).reset_index()
                fig2 = px.bar(
                    daily_listens, x="date", y="total_minutes",
                    title="Daily Listening Time",
                    labels={"total_minutes": "Minutes", "date": "Date"},
                )
                st.plotly_chart(fig2, use_container_width=True)

        # ── Alignment Tab ─────────────────────────────────────────

        with tab2:
            st.header("Workout ↔ Listening Alignment")
            st.markdown(
                f"Matching each workout to music listened within "
                f"**{window_hours} hours** afterwards, "
                f"then linking to **next-day recovery score**."
            )

            if len(whoop_data.get("workouts", [])) > 0 and len(listening) > 0:
                aligner = TemporalAligner(window_hours=window_hours)
                aligned = aligner.align(
                    whoop_data["workouts"], listening, timeline
                )
                st.session_state["aligned"] = aligned

                st.dataframe(
                    aligned[["workout_date", "n_tracks_in_window",
                             "has_listening_data", "recovery_score"]].head(20),
                    use_container_width=True,
                )

                # Scatter: tracks listened vs recovery
                complete = aligned[
                    aligned["has_listening_data"] & aligned["recovery_score"].notna()
                ]
                if len(complete) > 0:
                    fig3 = px.scatter(
                        complete, x="n_tracks_in_window", y="recovery_score",
                        trendline="ols",
                        title="Post-Workout Tracks vs Next-Day Recovery",
                        labels={
                            "n_tracks_in_window": "Tracks in Listening Window",
                            "recovery_score": "Next-Day Recovery %",
                        },
                    )
                    st.plotly_chart(fig3, use_container_width=True)

        # ── Model Results Tab ─────────────────────────────────────

        with tab3:
            st.header("Model Performance")
            st.markdown("Compare baseline (RF/XGBoost) vs sequence (LSTM) models.")
            st.info("Train models from the notebook, then load results here.")

            # Placeholder for model comparison table
            st.markdown("### Model Comparison")
            placeholder_results = pd.DataFrame({
                "Model": ["Random Forest", "XGBoost", "LSTM + Attention"],
                "MAE": ["—", "—", "—"],
                "RMSE": ["—", "—", "—"],
                "R²": ["—", "—", "—"],
            })
            st.table(placeholder_results)

        # ── Attention Visualisation Tab ───────────────────────────

        with tab4:
            st.header("Attention Weights")
            st.markdown(
                "Which tracks in the post-workout listening window "
                "did the LSTM attend to most when predicting recovery?"
            )
            st.info(
                "After training the LSTM, attention weights will be "
                "visualised here showing which listening moments matter most."
            )

            # Example placeholder visualisation
            st.markdown("### Example: Attention Over Listening Sequence")
            example_attn = np.random.dirichlet(np.ones(10))
            fig4 = go.Figure(go.Bar(
                x=[f"Track {i+1}" for i in range(10)],
                y=example_attn,
                marker_color="rgb(55, 83, 109)",
            ))
            fig4.update_layout(
                title="Attention Weights (placeholder — will show real data after training)",
                xaxis_title="Track in Post-Workout Window",
                yaxis_title="Attention Weight",
            )
            st.plotly_chart(fig4, use_container_width=True)

        # ── Recovery Profile Tab ──────────────────────────────────

        with tab5:
            st.header("Your Recovery Listening Profile")
            st.markdown(
                "The ideal audio characteristics for your post-workout "
                "listening based on what correlates with higher recovery."
            )
            st.info("This will be populated after model training.")

            # Placeholder profile
            st.markdown("### Ideal Post-Workout Audio Profile")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Optimal BPM Range", "—")
                st.metric("Energy Level", "—")
            with col2:
                st.metric("Valence (Mood)", "—")
                st.metric("Acousticness", "—")
            with col3:
                st.metric("Best Window", f"{window_hours}h")
                st.metric("Recommended Tracks", "—")

    except FileNotFoundError as e:
        st.error(f"Data not found: {e}")
        st.info(
            "**Setup instructions:**\n"
            "1. Export your Whoop data (app → Settings → Data Export) → place CSVs in `data/whoop/`\n"
            "2. Download your Spotify extended streaming history → place JSONs in `data/spotify/`\n"
            "3. (Optional) Download Kaggle Spotify Audio Features → place in `data/public/`"
        )

else:
    st.info("👈 Configure data paths in the sidebar and click **Load Data** to begin.")

    st.markdown("---")
    st.markdown(
        "### How RecoverWave Works\n\n"
        "1. **Whoop** provides your workout strain, sleep architecture, and recovery scores\n"
        "2. **Spotify** provides timestamped listening history with track identifiers\n"
        "3. We **align** each workout with post-workout music listening (configurable window)\n"
        "4. Audio features (BPM, energy, valence) are extracted for each track\n"
        "5. ML models predict next-day recovery from the listening sequence\n"
        "6. **Attention weights** reveal which listening moments matter most\n"
        "7. A **recovery profile** shows your optimal post-workout audio characteristics"
    )

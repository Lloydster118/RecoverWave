"""RecoverWave dashboard - skeleton.

Currently shows synthetic data. Will plug into the real pipeline once
Whoop and Spotify data arrive.
"""

import streamlit as st
import pandas as pd
from pathlib import Path


st.set_page_config(page_title="RecoverWave", layout="wide")
st.title("RecoverWave")
st.caption("Post-workout music and biometric recovery - work in progress")

st.warning("Running on synthetic data only - awaiting real Whoop/Spotify export.")

st.header("Daily recovery (synthetic)")
n = st.slider("Days", 7, 90, 30)
df = pd.DataFrame({
    "recovery": [50 + 15 * (i % 7 == 0) for i in range(n)],
    "day": pd.date_range("2025-01-01", periods=n),
})
st.line_chart(df.set_index("day"))

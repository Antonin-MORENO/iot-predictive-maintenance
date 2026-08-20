"""
Interactive predictive maintenance dashboard.

The user picks an engine and clicks "Run": the dashboard replays that
engine's cycles one by one, calls the Azure scoring endpoint for each,
and builds the diagnostic live.

Two-level alerting to avoid false alarms from noise:
- "possible anomaly" (orange): a single cycle exceeds the threshold, but
  it may just be sensor noise.
- "confirmed anomaly" (red): CONFIRM_WINDOW consecutive cycles stay above
  the threshold, indicating a persistent, real degradation.

Run:
    streamlit run src/dashboard/dashboard.py
"""

import os
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

SCORING_URL = os.getenv("SCORING_ENDPOINT_URL")
DATA_FILE = Path(__file__).resolve().parents[2] / "data" / "train_FD001.txt"

COLUMN_NAMES = (
    ["unit_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)
USEFUL_SENSORS = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_6", "sensor_7", "sensor_8",
    "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14",
    "sensor_15", "sensor_17", "sensor_20", "sensor_21",
]

CONFIRM_WINDOW = 5  # consecutive over-threshold cycles required to confirm

st.set_page_config(page_title="Predictive Maintenance", page_icon="🛩️", layout="wide")


@st.cache_data
def load_dataset() -> pd.DataFrame:
    if not DATA_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(DATA_FILE, sep=r"\s+", header=None, names=COLUMN_NAMES)


def score_cycle(row: pd.Series) -> dict | None:
    sensors = {c: float(row[c]) for c in USEFUL_SENSORS}
    try:
        resp = requests.post(SCORING_URL, json={"sensors": sensors}, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        st.error(f"Scoring request failed: {e}")
        return None


st.title("🛩️ Turbofan Predictive Maintenance")
st.caption("Pick an engine and run it — each cycle is scored live by an autoencoder deployed on Azure.")

df = load_dataset()
if df.empty:
    st.error("Dataset not found. Place train_FD001.txt in the data/ folder.")
    st.stop()
if not SCORING_URL:
    st.error("SCORING_ENDPOINT_URL is missing from your .env file.")
    st.stop()

# --- Controls ---
engines = sorted(df["unit_id"].unique())
col_a, col_b, col_c = st.columns([1, 1, 2])
selected = col_a.selectbox("Engine", engines, index=0)
speed = col_b.select_slider("Speed", options=["Slow", "Normal", "Fast"], value="Normal")
run = col_c.button("▶️ Run engine", type="primary", use_container_width=True)

delay = {"Slow": 0.4, "Normal": 0.15, "Fast": 0.02}[speed]

st.caption(
    f"An alert is **confirmed** only after {CONFIRM_WINDOW} consecutive cycles above the "
    "threshold. A single spike that falls back down is flagged as *possible* (noise), not confirmed."
)

metric_row = st.empty()
chart_area = st.empty()
alert_area = st.empty()

if run:
    engine_df = df[df["unit_id"] == selected].sort_values("cycle")
    history = []
    consecutive_over = 0      # running count of consecutive over-threshold cycles
    confirmed = False         # becomes True once a real anomaly is confirmed
    confirmed_at = None
    possible_count = 0

    for _, row in engine_df.iterrows():
        result = score_cycle(row)
        if result is None:
            break

        error = result["reconstruction_error"]
        threshold = result["threshold"]
        over = error > threshold

        # Track consecutive over-threshold streak
        if over:
            consecutive_over += 1
        else:
            consecutive_over = 0

        # Classify this cycle
        if consecutive_over >= CONFIRM_WINDOW:
            level = "confirmed"
            if not confirmed:
                confirmed = True
                confirmed_at = int(row["cycle"])
        elif over:
            level = "possible"
            possible_count += 1
        else:
            level = "ok"

        history.append({
            "cycle": int(row["cycle"]),
            "error": error,
            "threshold": threshold,
            "level": level,
        })

        latest = history[-1]
        if level == "confirmed":
            status = "🔴 CONFIRMED ANOMALY"
        elif level == "possible":
            status = "🟠 Possible anomaly"
        else:
            status = "✅ Healthy"

        # Metrics
        with metric_row.container():
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Cycle", latest["cycle"])
            m2.metric("Consecutive over threshold", consecutive_over)
            m3.metric("Reconstruction error", f"{error:.3f}")
            m4.metric("Status", status)

        # Chart
        hist_df = pd.DataFrame(history)
        chart_df = hist_df.set_index("cycle")[["error", "threshold"]]
        chart_area.line_chart(chart_df, color=["#1f77b4", "#d62728"])

        # Alert banner (priority: confirmed > possible > clear)
        if confirmed:
            alert_area.error(
                f"🔴 Confirmed anomaly since cycle {confirmed_at} — "
                f"{CONFIRM_WINDOW}+ consecutive cycles above threshold. Persistent degradation."
            )
        elif level == "possible":
            alert_area.warning(
                f"🟠 Possible anomaly at cycle {latest['cycle']} — single spike above threshold "
                "(may be noise, not yet confirmed)."
            )
        else:
            alert_area.empty()

        time.sleep(delay)

    # Final summary
    if confirmed:
        st.error(f"Run complete — engine {selected}: confirmed anomaly from cycle {confirmed_at} onward.")
    else:
        st.success(f"Run complete — engine {selected}: no confirmed anomaly ({possible_count} isolated spikes ignored).")
else:
    st.info("Select an engine and click **Run engine** to start the live diagnostic.")

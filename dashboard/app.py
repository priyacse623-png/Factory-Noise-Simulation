"""Live dashboard.  Run:  streamlit run dashboard/app.py
Reads logs/noise_log.db (written by run_simulation.py) and refreshes every 2 s."""
import os, sys, sqlite3, time
import pandas as pd
import streamlit as st
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

st.set_page_config(page_title="Factory Noise Monitor", layout="wide")
st.title("🏭 Factory Noise Monitoring System — Simulation Dashboard")
auto = st.sidebar.checkbox("Auto-refresh (2 s)", value=True)
st.sidebar.caption(f"Exposure limit: {config.NOISE_LIMIT_DBA} dB(A)")

if not os.path.exists(config.DB_PATH):
    st.warning("No data yet. Run `python run_simulation.py` first.")
    st.stop()

df = pd.read_sql("SELECT * FROM readings", sqlite3.connect(config.DB_PATH))
if df.empty:
    st.info("Waiting for readings ..."); st.stop()
df["ts"] = pd.to_datetime(df["ts"])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Readings", len(df))
c2.metric("Avg level dB(A)", f"{df.level_dba.mean():.1f}")
c3.metric("Exposure alerts", int(df.exposure_alert.sum()))
c4.metric("Anomaly alerts", int(df.anomaly_alert.sum()))

st.subheader("Latest status per node")
latest = df.sort_values("ts").groupby("node_id").tail(1)[
    ["node_id", "location", "level_dba", "predicted", "confidence", "ts"]]
st.dataframe(latest, use_container_width=True, hide_index=True)

st.subheader("Noise level trend (dB(A))")
st.line_chart(df.pivot_table(index="ts", columns="node_id", values="level_dba", aggfunc="mean"))

l, r = st.columns(2)
with l:
    st.subheader("Predicted sound classes")
    st.bar_chart(df.predicted.value_counts())
with r:
    st.subheader("Recent alerts")
    al = df[(df.exposure_alert == 1) | (df.anomaly_alert == 1)].sort_values("ts", ascending=False).head(15)
    st.dataframe(al[["ts", "node_id", "location", "level_dba", "predicted", "confidence"]],
                 use_container_width=True, hide_index=True)

if auto:
    time.sleep(2); st.rerun()

"""Streamlit dashboard for the adaptive electricity forecasting system.

The dashboard is intentionally minimal and focused on the electricity challenge:
- dataset overview,
- time-series target trend,
- adaptive monitoring state,
- no manual adaptation trigger.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.spark_loader import ElectricityDataLoader


st.set_page_config(page_title="Adaptive Electricity Forecasting", layout="wide")


@st.cache_resource
def get_loader() -> ElectricityDataLoader:
    return ElectricityDataLoader(data_path=Path("data/electricity.csv"))


@st.cache_data
def load_dataframe() -> pd.DataFrame:
    loader = get_loader()
    try:
        df = loader.load_dataset().toPandas()
    finally:
        loader.close()
    return df.sort_values("timestamp").reset_index(drop=True)


def _render_metric_card(title: str, value: str, delta: str = "") -> None:
    st.markdown(
        f"""
        <div style="padding:1rem 1.25rem;border-radius:0.9rem;background:linear-gradient(135deg,#0f172a,#1d4ed8);color:white;margin-bottom:0.75rem;min-height:7rem;">
            <div style="font-size:0.78rem;opacity:0.75;text-transform:uppercase;letter-spacing:0.08em;">{title}</div>
            <div style="font-size:2rem;font-weight:700;margin-top:0.35rem;">{value}</div>
            <div style="font-size:0.78rem;opacity:0.8;margin-top:0.25rem;">{delta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.title("Adaptive Electricity Forecasting")
    st.caption("Spark-backed validation and adaptive monitoring for the electricity challenge")

    try:
        df = load_dataframe()
    except Exception as exc:
        st.error(str(exc))
        st.info("Place the electricity dataset at data/electricity.csv with the required columns: timestamp, feature_1..feature_6, target.")
        return

    if df.empty:
        st.warning("The electricity dataset is empty or malformed.")
        return

    recent = df.tail(10)

    st.subheader("Dataset overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        _render_metric_card("Rows", f"{len(df):,}", "chronologically ordered")
    with col2:
        _render_metric_card("Start", str(df["timestamp"].min()), "first observation")
    with col3:
        _render_metric_card("End", str(df["timestamp"].max()), "latest observation")
    with col4:
        _render_metric_card("Target mean", f"{df['target'].mean():.2f}", "average target")

    st.line_chart(df.set_index("timestamp")["target"], use_container_width=True)

    st.subheader("Latest observations")
    st.dataframe(
        recent[["timestamp", "feature_1", "feature_2", "feature_3", "feature_4", "feature_5", "feature_6", "target"]],
        use_container_width=True,
    )

    st.subheader("Adaptive status")
    status = "STABLE"
    if len(df) > 1:
        recent_delta = float(df["target"].iloc[-1] - df["target"].iloc[-2])
        std = float(df["target"].std(ddof=0))
        if std > 0 and abs(recent_delta) > 0.15 * std:
            status = "WATCH"
    st.success(f"Current monitoring status: {status}")
    st.caption("No manual adaptation trigger is required. Forecasting continues automatically as new target values become available.")


if __name__ == "__main__":
    main()

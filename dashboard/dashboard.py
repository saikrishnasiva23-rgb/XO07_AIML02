"""Streamlit dashboard for the adaptive retail demand forecasting system.

This module intentionally integrates the existing backend modules instead of
reimplementing the forecasting logic. It supports:
- adaptive retail forecasting via AdaptiveRetailSystem
- inventory planning via InventoryManager
- local storage via RetailDatabase
- metrics evaluation via ForecastMetrics
- external evaluation data via EvaluationAPIClient

The dashboard defaults to a demo mode when no evaluation API is configured.
"""

from __future__ import annotations

import math
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.api_client import EvaluationAPIClient
from src.database import RetailDatabase
from src.inventory import InventoryManager
from src.metrics import ForecastMetrics
from src.system import AdaptiveRetailSystem


st.set_page_config(page_title="Adaptive Retail Intelligence", layout="wide")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_series(values: Iterable[Any]) -> List[float]:
    result: List[float] = []
    for value in values:
        try:
            result.append(float(value))
        except (TypeError, ValueError):
            continue
    return result


@st.cache_resource
def get_database() -> RetailDatabase:
    return RetailDatabase()


@st.cache_resource
def get_inventory_manager() -> InventoryManager:
    return InventoryManager()


@st.cache_resource
def get_metrics() -> ForecastMetrics:
    return ForecastMetrics(rolling_window=7)


@st.cache_resource
def get_api_client() -> Optional[EvaluationAPIClient]:
    base_url = (
        os.getenv("EVALUATION_API_BASE_URL")
        or os.getenv("PS02_EVALUATION_URL")
        or os.getenv("EVALUATION_URL")
    )
    api_key = (
        os.getenv("EVALUATION_API_KEY")
        or os.getenv("PS02_API_KEY")
        or os.getenv("API_KEY")
    )
    if not base_url:
        return None
    return EvaluationAPIClient(base_url=base_url, api_key=api_key)


def _seed_demo_database() -> List[Dict[str, Any]]:
    db = get_database()
    products = db.get_all_products()
    if products:
        return products

    product_specs = {
        "SKU-100": {"name": "Urban Essentials Tee", "category": "Apparel", "stock": 185.0, "price": 29.99},
        "SKU-101": {"name": "Trail Runner Sneakers", "category": "Footwear", "stock": 140.0, "price": 89.99},
        "SKU-102": {"name": "Smart Water Bottle", "category": "Lifestyle", "stock": 220.0, "price": 24.99},
        "SKU-103": {"name": "Home Office Lamp", "category": "Home", "stock": 96.0, "price": 54.99},
    }

    for product_id, meta in product_specs.items():
        db.add_product(
            product_id=product_id,
            product_name=meta["name"],
            category=meta["category"],
            current_stock=meta["stock"],
            unit_price=meta["price"],
            reorder_level=max(20.0, meta["stock"] * 0.15),
            supplier="Northline Supply",
        )

    stable = [100, 102, 101, 103, 99, 104, 102, 101, 100, 103, 105, 104, 106, 103, 101]
    anomaly = [150]
    persistent = [150, 155, 152, 158, 160, 165, 170, 175, 178, 181]
    product_history = {
        "SKU-100": stable + anomaly + persistent,
        "SKU-101": [90, 92, 91, 95, 94, 98, 96, 97, 99, 101, 103, 105, 107, 104, 102] + [140] + [142, 145, 148, 151, 153, 160, 165, 168, 170],
        "SKU-102": [110, 112, 111, 108, 109, 114, 116, 118, 117, 115, 120, 122, 124, 121, 119] + [170] + [169, 175, 177, 180, 182, 188, 192, 194, 198],
        "SKU-103": [70, 72, 71, 74, 76, 77, 75, 79, 78, 80, 83, 82, 85, 87, 88] + [120] + [123, 126, 130, 134, 138, 141, 145, 149, 152],
    }

    now = datetime.now()
    for product_id, demand_series in product_history.items():
        for offset, value in enumerate(demand_series):
            db.record_demand(
                product_id=product_id,
                demand=value,
                timestamp=(now - timedelta(days=len(demand_series) - offset)).isoformat(timespec="seconds"),
            )

    return db.get_all_products()


def _get_history_for_product(product_id: str) -> List[float]:
    db = get_database()
    demand_rows = db.get_demand_history(product_id=product_id, limit=60)
    values = _safe_series(row["demand"] for row in reversed(demand_rows))
    if values:
        return values

    base = {
        "SKU-100": [100, 102, 101, 103, 99, 104, 102, 101, 100, 103],
        "SKU-101": [90, 92, 91, 95, 94, 98, 96, 97, 99, 101],
        "SKU-102": [110, 112, 111, 108, 109, 114, 116, 118, 117, 115],
        "SKU-103": [70, 72, 71, 74, 76, 77, 75, 79, 78, 80],
    }
    return base.get(product_id, [100, 102, 101, 103, 99])


def _product_dataframe(products: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    db = get_database()
    for product in products:
        product_id = product["product_id"]
        history = db.get_demand_history(product_id=product_id, limit=30)
        history_values = [float(row["demand"]) for row in history]
        if not history_values:
            history_values = _get_history_for_product(product_id)
        recent = history_values[-7:]
        total = sum(history_values)
        stock = float(product.get("current_stock", 0.0))
        price = float(product.get("unit_price", 0.0))
        forecast = _estimate_product_forecast(product_id, history_values)
        try:
            recommendation = _inventory_recommendation(product_id, history_values, stock)
        except Exception:
            recommendation = None
        rows.append(
            {
                "product_id": product_id,
                "product_name": product["product_name"],
                "category": product.get("category") or "General",
                "current_stock": stock,
                "unit_price": price,
                "recent_demand": float(np.mean(recent)) if recent else 0.0,
                "total_demand": float(total),
                "forecast": float(forecast),
                "stock_out_risk": float(recommendation["stock_out_risk"]) if recommendation else 0.0,
                "inventory_status": recommendation["inventory_status"] if recommendation else "Unknown",
                "recommended_reorder": float(recommendation["recommended_reorder_quantity"]) if recommendation else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _estimate_product_forecast(product_id: str, history: List[float]) -> float:
    if len(history) < 2:
        return float(history[-1]) if history else 0.0
    system = AdaptiveRetailSystem()
    system.initialize(history[: min(len(history), 10)])
    try:
        return float(system.forecast()["forecast"])
    except Exception:
        return float(np.mean(history[-3:])) if history else 0.0


def _inventory_recommendation(product_id: str, history: List[float], current_stock: float) -> Dict[str, Any]:
    manager = get_inventory_manager()
    recommendation = manager.recommend_for_product(
        product_id=product_id,
        historical_demand=history,
        current_stock=current_stock,
    )
    return recommendation.to_dict()


def _status_badge(label: str) -> str:
    label = str(label).upper()
    palette = {
        "STABLE": "success",
        "WATCH": "warning",
        "POSSIBLE CHANGE": "warning",
        "PERSISTENT CHANGE": "error",
        "ADAPTING": "error",
        "CRITICAL": "error",
        "REPLENISH": "warning",
        "OVERSTOCKED": "info",
        "HEALTHY": "success",
    }
    color = palette.get(label, "primary")
    return f'<span style="display:inline-block;padding:0.25rem 0.6rem;border-radius:999px;background:{color};color:white;font-weight:700;">{label}</span>'


def _build_demo_sequence(product_id: str) -> List[float]:
    base_map = {
        "SKU-100": [100, 102, 101, 103, 99, 104, 102, 101, 100, 103, 105, 104, 106, 103, 101, 150, 150, 155, 152, 158, 160, 165, 170, 175, 178],
        "SKU-101": [90, 92, 91, 95, 94, 98, 96, 97, 99, 101, 103, 105, 107, 104, 102, 140, 142, 145, 148, 151, 153, 160, 165, 168, 170],
        "SKU-102": [110, 112, 111, 108, 109, 114, 116, 118, 117, 115, 120, 122, 124, 121, 119, 170, 169, 175, 177, 180, 182, 188, 192, 194, 198],
        "SKU-103": [70, 72, 71, 74, 76, 77, 75, 79, 78, 80, 83, 82, 85, 87, 88, 120, 123, 126, 130, 134, 138, 141, 145, 149, 152],
    }
    return base_map.get(product_id, [100, 102, 101, 103, 99, 104, 102, 101, 100, 103, 150, 150, 155, 152, 158, 160, 165])


def _init_demo_state(product_id: str) -> Dict[str, Any]:
    state = st.session_state.setdefault("demo_state", {})
    state.setdefault(product_id, {})
    bucket = state[product_id]
    if not bucket:
        history = _get_history_for_product(product_id)[:10]
        system = AdaptiveRetailSystem()
        system.initialize(history)
        bucket["system"] = system
        bucket["history"] = list(history)
        bucket["sequence"] = _build_demo_sequence(product_id)[len(history):]
        bucket["trace"] = []
        bucket["last_status"] = "INITIALIZED"
    return bucket


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


def _render_forecast_chart(history: List[float], forecast_trace: List[Dict[str, Any]]) -> go.Figure:
    fig = go.Figure()
    actual_x = list(range(len(history)))
    actual_y = list(history)
    fig.add_trace(go.Scatter(x=actual_x, y=actual_y, mode="lines+markers", name="Historical demand", line={"color": "#38bdf8", "width": 3}, marker={"size": 7}))

    if forecast_trace:
        forecast_x = [len(history) - 1 + idx + 1 for idx, _ in enumerate(forecast_trace)]
        forecast_y = [float(item.get("forecast", 0.0)) for item in forecast_trace]
        actual_points = [float(item.get("actual", 0.0)) for item in forecast_trace]
        fig.add_trace(go.Scatter(x=forecast_x, y=forecast_y, mode="lines+markers", name="Forecast", line={"color": "#f59e0b", "width": 3}, marker={"size": 7}))
        fig.add_trace(go.Scatter(x=forecast_x, y=actual_points, mode="markers", name="Actual observations", marker={"color": "#10b981", "size": 8}))

    fig.update_layout(
        title="Demand vs forecast",
        height=420,
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
        hovermode="x unified",
    )
    return fig


def _render_error_chart(forecast_trace: List[Dict[str, Any]]) -> go.Figure:
    fig = go.Figure()
    if forecast_trace:
        x = list(range(len(forecast_trace)))
        errors = [float(item.get("error", 0.0)) for item in forecast_trace]
        drift = [float(item.get("drift_score", 0.0)) for item in forecast_trace]
        fig.add_trace(go.Scatter(x=x, y=errors, mode="lines+markers", name="Forecast error", line={"color": "#ef4444", "width": 3}, marker={"size": 6}))
        fig.add_trace(go.Scatter(x=x, y=drift, mode="lines+markers", name="Drift score", line={"color": "#8b5cf6", "width": 2}, marker={"size": 5}))
    fig.update_layout(title="Forecast error and drift trend", template="plotly_white", height=320)
    return fig


def _render_model_weights_chart(system: AdaptiveRetailSystem) -> go.Figure:
    status = system.get_status()["models"]
    labels = list(status.keys())
    weights = [float(status[name].get("weight", 0.0)) for name in labels]
    fig = go.Figure(go.Bar(x=labels, y=weights, marker={"color": ["#60a5fa", "#a78bfa", "#f472b6", "#34d399"]}))
    fig.update_layout(title="Dynamic model weights", template="plotly_white", height=330)
    return fig


def _render_inventory_risk_chart(recommendation: Dict[str, Any]) -> go.Figure:
    labels = ["Stock-out risk", "Excess inventory risk"]
    values = [float(recommendation.get("stock_out_risk", 0.0)), float(recommendation.get("excess_inventory_risk", 0.0))]
    fig = go.Figure(go.Bar(x=labels, y=values, marker={"color": ["#f87171", "#60a5fa"]}))
    fig.update_layout(title="Inventory risk", template="plotly_white", height=300)
    return fig


def _render_revenue_chart(df: pd.DataFrame) -> go.Figure:
    if df.empty or not (df["unit_price"] > 0).any():
        fig = go.Figure()
        fig.update_layout(title="Revenue data unavailable", template="plotly_white", height=250)
        return fig
    revenue = df.assign(revenue=df["total_demand"] * df["unit_price"])
    top = revenue.sort_values("revenue", ascending=False).head(10)
    fig = go.Figure(go.Bar(x=top["product_name"], y=top["revenue"], marker={"color": "#14b8a6"}))
    fig.update_layout(title="Revenue contribution by product", template="plotly_white", height=350)
    return fig


def _get_adaptation_events(system: AdaptiveRetailSystem) -> List[Dict[str, Any]]:
    events = system.get_adaptation_events()
    if events:
        return events
    db = get_database()
    rows = db.get_adaptation_events(limit=20)
    return rows


def _current_mode() -> str:
    mode = st.session_state.get("mode", "Demo Mode")
    if not isinstance(mode, str):
        return "Demo Mode"
    return mode


def _render_system_status(system: AdaptiveRetailSystem) -> Dict[str, Any]:
    status = system.get_status()
    monitor = status["monitor"]
    adaptation = status["adaptation"]
    models = status["models"]
    return {
        "initialized": status.get("initialized", False),
        "step": status.get("step", 0),
        "data_points": status.get("data_points", 0),
        "drift_score": float(monitor.get("drift_score", 0.0)),
        "status": (status.get("last_forecast") and "ACTIVE") or "READY",
        "change_detected": bool(monitor.get("consecutive_drift", 0) > 0),
        "persistent_change": bool(monitor.get("consecutive_drift", 0) >= 3),
        "adaptation_count": int(adaptation.get("adaptation_count", 0)),
        "last_adaptation": adaptation.get("last_adaptation"),
        "model_weights": models,
    }


def _render_dashboard() -> None:
    db = get_database()
    products = _seed_demo_database()
    df = _product_dataframe(products)
    selected_product = st.sidebar.selectbox("Product", df["product_id"].tolist(), index=0)
    product_row = df[df["product_id"] == selected_product].iloc[0]
    mode = st.sidebar.radio("Operational mode", ["Demo Mode", "Live Mode"], index=0)
    st.session_state["mode"] = mode
    forecast_horizon = st.sidebar.slider("Forecast horizon", min_value=1, max_value=12, value=3)
    st.sidebar.button("Refresh dashboard", use_container_width=True)

    st.title("Adaptive Retail Intelligence")
    st.caption("Retail Data → Adaptive Forecast → Drift Detection → Adaptation → Inventory Decision")

    if mode == "Live Mode":
        client = get_api_client()
        if client is None:
            st.warning("No live evaluation API endpoint is configured. The dashboard is running in Demo Mode using local retail data.")
            mode = "Demo Mode"
        else:
            health = client.check_health()
            if health.get("ok"):
                st.success("Live evaluation API connected")
            else:
                st.warning("Live evaluation API unavailable. Demo Mode remains active for monitoring continuity.")
                mode = "Demo Mode"
    else:
        st.info("Demo Mode is active. The system is using locally generated retail demand patterns to simulate the full adaptive loop.")

    product_bucket = _init_demo_state(selected_product)
    system = product_bucket["system"]
    history = system.get_history()
    trace = product_bucket["trace"]

    if st.sidebar.button("Reset simulation", use_container_width=True):
        product_bucket = _init_demo_state(selected_product)
        st.session_state["demo_state"][selected_product] = product_bucket
        system = product_bucket["system"]
        history = system.get_history()
        trace = []
        st.rerun()

    if st.sidebar.button("Process next observation", use_container_width=True):
        sequence = product_bucket["sequence"]
        if sequence:
            actual = float(sequence.pop(0))
            result = system.process_observation(actual)
            forecast = result["forecast"]["forecast"]
            drift_score = float(result["observation"]["drift"]["score"])
            status = result["observation"]["system_status"]
            error = float(result["observation"]["error"])
            adaptation = bool(result["observation"]["adaptation"]["triggered"])
            trace.append(
                {
                    "step": len(trace) + 1,
                    "actual": actual,
                    "forecast": forecast,
                    "error": error,
                    "drift_score": drift_score,
                    "status": status,
                    "adapted": adaptation,
                }
            )
            product_bucket["trace"] = trace
            product_bucket["sequence"] = sequence
            product_bucket["history"] = system.get_history()
            product_bucket["last_status"] = status
            st.session_state["demo_state"][selected_product] = product_bucket
            st.rerun()
        else:
            st.info("The demo sequence is complete. Reset the simulation to start a fresh demand shift cycle.")

    if st.sidebar.button("Run stream", use_container_width=True):
        for _ in range(min(3, len(product_bucket["sequence"]))):
            if not product_bucket["sequence"]:
                break
            actual = float(product_bucket["sequence"].pop(0))
            result = system.process_observation(actual)
            trace.append(
                {
                    "step": len(trace) + 1,
                    "actual": actual,
                    "forecast": result["forecast"]["forecast"],
                    "error": float(result["observation"]["error"]),
                    "drift_score": float(result["observation"]["drift"]["score"]),
                    "status": result["observation"]["system_status"],
                    "adapted": bool(result["observation"]["adaptation"]["triggered"]),
                }
            )
        product_bucket["trace"] = trace
        product_bucket["sequence"] = product_bucket["sequence"]
        product_bucket["history"] = system.get_history()
        st.session_state["demo_state"][selected_product] = product_bucket
        st.rerun()

    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            _render_metric_card("Total Products", f"{len(products)}", "Catalog")
        with col2:
            demand_today = sum(row["demand"] for row in db.get_recent_demand(product_id=selected_product, limit=7))
            _render_metric_card("Today's Demand", f"{demand_today:.0f}", "Units")
        with col3:
            forecasted = float(_estimate_product_forecast(selected_product, history))
            _render_metric_card("Forecasted Demand", f"{forecasted:.1f}", f"{forecast_horizon}-step horizon")
        with col4:
            _render_metric_card("Current Inventory", f"{product_row['current_stock']:.0f}", "Units")

        col5, col6, col7, col8 = st.columns(4)
        with col5:
            low_stock = int((df["inventory_status"] == "Replenish").sum() + (df["inventory_status"] == "Critical").sum())
            _render_metric_card("Low Stock Products", f"{low_stock}", "Products")
        with col6:
            risk = float(_inventory_recommendation(selected_product, history, float(product_row["current_stock"]))["stock_out_risk"]) * 100.0
            _render_metric_card("Stock-out Risk", f"{risk:.1f}%", "Probability")
        with col7:
            metrics = get_metrics()
            recent_history = history[-10:]
            if len(recent_history) >= 2:
                errors = [abs(recent_history[i] - (recent_history[i-1] if i > 0 else recent_history[0])) for i in range(len(recent_history))]
                avg_err = float(np.mean(errors)) if errors else 0.0
            else:
                avg_err = 0.0
            _render_metric_card("Average Forecast Error", f"{avg_err:.1f}", "Units / step")
        with col8:
            adaptation_events = _get_adaptation_events(system)
            _render_metric_card("Adaptations Triggered", f"{len(adaptation_events)}", "Events")

    st.subheader("Demand forecast")
    chart_col, insight_col = st.columns([2.5, 1])
    with chart_col:
        fig = _render_forecast_chart(history, trace)
        st.plotly_chart(fig, use_container_width=True)
    with insight_col:
        status_summary = _render_system_status(system)
        st.markdown("<b>Adaptive AI status</b>", unsafe_allow_html=True)
        status_label = system.get_status()["monitor"]["drift_score"]
        st.markdown(_status_badge(status_summary["status"]), unsafe_allow_html=True)
        st.write(f"Drift score: {status_summary['drift_score']:.3f}")
        st.write(f"Persistent change: {str(system.get_status()['monitor'].get('consecutive_drift',0) >= 3).lower()}")
        st.write(f"Data points: {status_summary['data_points']}")
        st.write(f"Adaptations: {status_summary['adaptation_count']}")

    st.subheader("Adaptive AI status")
    system_status = system.get_status()
    monitor_state = system_status["monitor"]
    adaptation_state = system_status["adaptation"]
    model_state = system_status["models"]

    stat_cols = st.columns(6)
    with stat_cols[0]:
        st.metric("System status", "ACTIVE")
    with stat_cols[1]:
        st.metric("Drift score", f"{float(monitor_state.get('drift_score', 0.0)):.3f}")
    with stat_cols[2]:
        st.metric("Change detection", "ON" if monitor_state.get("consecutive_drift", 0) > 0 else "OFF")
    with stat_cols[3]:
        st.metric("Persistence", "YES" if monitor_state.get("consecutive_drift", 0) >= 3 else "NO")
    with stat_cols[4]:
        st.metric("Adaptation", "YES" if adaptation_state.get("adaptation_count", 0) > 0 else "NO")
    with stat_cols[5]:
        st.metric("Last adaptation", adaptation_state.get("last_adaptation") or "N/A")

    st.subheader("Model intelligence")
    model_col, error_col = st.columns([1.7, 1.3])
    with model_col:
        st.plotly_chart(_render_model_weights_chart(system), use_container_width=True)
    with error_col:
        model_errors = pd.DataFrame(
            [
                {"Model": name, "Recent error": float(values.get("recent_error", 0.0)), "Weight": float(values.get("weight", 0.0))}
                for name, values in model_state.items()
            ]
        )
        st.dataframe(model_errors, use_container_width=True)

    st.subheader("Inventory intelligence")
    recommendation = _inventory_recommendation(selected_product, history, float(product_row["current_stock"]))
    inv_cols = st.columns(6)
    with inv_cols[0]:
        st.metric("Current stock", f"{recommendation['current_stock']:.1f}")
    with inv_cols[1]:
        st.metric("Forecasted demand", f"{recommendation['forecasted_demand']:.1f}")
    with inv_cols[2]:
        st.metric("Safety stock", f"{recommendation['safety_stock']:.1f}")
    with inv_cols[3]:
        st.metric("Reorder point", f"{recommendation['reorder_point']:.1f}")
    with inv_cols[4]:
        st.metric("Reorder qty", f"{recommendation['recommended_reorder_quantity']:.1f}")
    with inv_cols[5]:
        st.metric("Stock-out risk", f"{recommendation['stock_out_risk'] * 100:.1f}%")

    alert = recommendation["inventory_status"].upper()
    st.markdown(_status_badge(alert), unsafe_allow_html=True)
    st.plotly_chart(_render_inventory_risk_chart(recommendation), use_container_width=True)

    st.subheader("Product analytics")
    analytics = df[["product_id", "product_name", "category", "current_stock", "forecast", "recent_demand", "stock_out_risk", "inventory_status", "recommended_reorder"]]
    st.dataframe(analytics, use_container_width=True, hide_index=True)

    st.subheader("Sales & revenue")
    if df.empty or not (df["unit_price"] > 0).any():
        st.info("Revenue data is not available in the current database. The dashboard is intentionally avoiding fabricated sales metrics.")
    else:
        revenue_chart = _render_revenue_chart(df)
        st.plotly_chart(revenue_chart, use_container_width=True)
        total_sales = float((df["total_demand"] * df["unit_price"]).sum())
        expected_sales = float((df["forecast"] * df["unit_price"]).sum())
        st.write(f"Total sales: {total_sales:,.2f}")
        st.write(f"Expected sales: {expected_sales:,.2f}")

    st.subheader("Demand intelligence")
    recent_window = history[-7:]
    change = float(recent_window[-1] - recent_window[0]) if recent_window else 0.0
    direction = "Increasing" if change > 0 else "Decreasing" if change < 0 else "Stable"
    volatility = float(np.std(recent_window)) if recent_window else 0.0
    st.write(f"Demand trend: {direction} by {abs(change):.1f} units over the last 7 periods.")
    st.write(f"Recent demand change: {change:.1f} units")
    st.write(f"Volatility: {volatility:.2f}")
    st.write(f"Persistent change detected: {str(monitor_state.get('consecutive_drift', 0) >= 3).lower()}")
    st.write(f"Adaptation recommendation: {'Adopt recent-window retraining' if monitor_state.get('consecutive_drift', 0) >= 3 else 'Monitor for continued drift'}")

    st.subheader("Adaptation timeline")
    event_df = pd.DataFrame(_get_adaptation_events(system))
    if event_df.empty:
        st.info("No adaptation events have been recorded yet in the active forecast lifecycle.")
    else:
        st.dataframe(event_df, use_container_width=True, hide_index=True)

    st.subheader("System monitoring")
    api_client = get_api_client()
    if api_client is not None:
        api_status = api_client.check_health()
        api_ok = bool(api_status.get("ok"))
        api_message = api_status.get("message") or "Healthy"
    else:
        api_ok = False
        api_message = "Demo mode active; no live API configured"

    monitor_cols = st.columns(6)
    with monitor_cols[0]:
        st.metric("Evaluation API", "OK" if api_ok else "DEMO")
    with monitor_cols[1]:
        st.metric("Last communication", api_message[:24] if api_message else "N/A")
    with monitor_cols[2]:
        st.metric("Observations processed", str(sum(len(db.get_demand_history(product_id=product["product_id"])) for product in products)))
    with monitor_cols[3]:
        st.metric("Forecasting status", "ACTIVE" if system.get_status()["initialized"] else "WAITING")
    with monitor_cols[4]:
        st.metric("Database", "READY")
    with monitor_cols[5]:
        st.metric("Last update", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    if mode == "Demo Mode":
        st.caption("Demo data is synthetic retail data used only for visualization and validation. It is not evaluation-server data.")


def main() -> None:
    _render_dashboard()


if __name__ == "__main__":
    main()

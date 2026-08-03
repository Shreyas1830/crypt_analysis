"""
dashboard/app.py
-----------------
Streamlit dashboard for the Crypto Market Analytics Platform.

Run with:
    streamlit run app.py     (from the project root - see root app.py, which
                               simply imports and runs this module)

Sections:
    - Sidebar filters (coin, date range, comparison mode, dark mode toggle)
    - KPI cards
    - Market structure charts (treemap, sunburst, bubble)
    - Per-coin deep dive (price line, candlestick, rolling volatility)
    - Correlation heatmap
    - Anomaly dashboard
    - Forecast dashboard
    - Download buttons (CSV / Excel / PDF)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running `streamlit run dashboard/app.py` directly from repo root
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from src.anomaly_detection import run_anomaly_pipeline
from src.cleaning import run_cleaning_pipeline
from src.database import Database
from src.feature_engineering import build_feature_set
from src.forecasting import forecast_arima, forecast_linear_regression
from src.pipeline import run_pipeline_once
from src.reporting import compute_kpis, generate_insights
from src import visualization as viz

st.set_page_config(page_title="Crypto Market Analytics", layout="wide", page_icon="\U0001FA99")


# --------------------------------------------------------------------------- #
# Data loading (cached so the dashboard doesn't re-run the full pipeline on
# every widget interaction - only on manual refresh or TTL expiry)
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=300)
def load_data() -> pd.DataFrame:
    db = Database()
    history = db.read_all()
    if history.empty:
        return history
    clean_df, _ = run_cleaning_pipeline(history)
    feature_df = build_feature_set(clean_df)
    final_df = run_anomaly_pipeline(feature_df)
    return final_df


def refresh_data() -> None:
    with st.spinner("Fetching latest market data..."):
        result = run_pipeline_once(generate_reports=False)
    if result.get("status") == "success":
        st.success(f"Refreshed. {result.get('rows_inserted', 0)} new rows inserted.")
        st.cache_data.clear()
    else:
        st.error("Refresh failed - check logs/pipeline.log for details.")


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("\U0001FA99 Crypto Analytics")
if st.sidebar.button("\U0001F504 Refresh live data"):
    refresh_data()

dark_mode = st.sidebar.toggle("Dark mode", value=False)
if dark_mode:
    st.markdown(
        "<style>body, .stApp {background-color:#0e1117; color:#e6e6e6;}</style>",
        unsafe_allow_html=True,
    )

df = load_data()

if df.empty:
    st.warning(
        "No data yet. Run `python src/pipeline.py` once (or click Refresh in the "
        "sidebar) to populate the database before using the dashboard."
    )
    st.stop()

all_symbols = sorted(df["symbol"].unique().tolist())
selected_symbol = st.sidebar.selectbox("Coin", all_symbols, index=all_symbols.index("BTC") if "BTC" in all_symbols else 0)

comparison_mode = st.sidebar.checkbox("Comparison mode")
compare_symbol = None
if comparison_mode:
    compare_symbol = st.sidebar.selectbox("Compare against", [s for s in all_symbols if s != selected_symbol])

min_date, max_date = df["snapshot_time"].min(), df["snapshot_time"].max()
date_range = st.sidebar.date_input("Date range", value=(min_date.date(), max_date.date()))
search_term = st.sidebar.text_input("Search coin by name/symbol")

latest_snapshot = df[df["snapshot_time"] == df["snapshot_time"].max()].copy()
if search_term:
    mask = latest_snapshot["symbol"].str.contains(search_term, case=False) | latest_snapshot["name"].str.contains(
        search_term, case=False
    )
    latest_snapshot = latest_snapshot[mask]

# --------------------------------------------------------------------------- #
# KPI Cards
# --------------------------------------------------------------------------- #
st.title("Cryptocurrency Market Analytics Platform")
st.caption("Live market intelligence: trends, anomalies, forecasts, and rankings across the top tracked coins.")

kpis = compute_kpis(df[df["snapshot_time"] == df["snapshot_time"].max()])
kpi_cols = st.columns(len(kpis))
for col, kpi in zip(kpi_cols, kpis):
    col.metric(kpi["label"], kpi["value"])

st.divider()

# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tab_overview, tab_deepdive, tab_correlation, tab_anomaly, tab_forecast, tab_rankings, tab_downloads = st.tabs(
    ["Market Overview", "Coin Deep Dive", "Correlations", "Anomalies", "Forecasts", "Rankings", "Downloads"]
)

with tab_overview:
    c1, c2 = st.columns(2)
    c1.plotly_chart(viz.treemap_market_cap(latest_snapshot), use_container_width=True)
    c2.plotly_chart(viz.sunburst_dominance(latest_snapshot), use_container_width=True)

    st.plotly_chart(viz.bubble_chart(latest_snapshot), use_container_width=True)

    g1, g2 = st.columns(2)
    fig_gain, fig_lose = viz.top_gainers_losers_bar(latest_snapshot)
    g1.plotly_chart(fig_gain, use_container_width=True)
    g2.plotly_chart(fig_lose, use_container_width=True)

with tab_deepdive:
    st.subheader(f"{selected_symbol} Deep Dive")
    c1, c2 = st.columns(2)
    c1.plotly_chart(viz.price_line_chart(df, selected_symbol), use_container_width=True)
    c2.plotly_chart(viz.price_area_chart(df, selected_symbol), use_container_width=True)

    ohlc = viz.resample_to_ohlc(df, selected_symbol, rule="1H")
    if not ohlc.empty:
        st.plotly_chart(viz.candlestick_chart(ohlc, selected_symbol), use_container_width=True)

    st.plotly_chart(viz.rolling_volatility_chart(df, selected_symbol), use_container_width=True)

    if comparison_mode and compare_symbol:
        st.subheader(f"Comparison: {selected_symbol} vs {compare_symbol}")
        compare_df = df[df["symbol"].isin([selected_symbol, compare_symbol])]
        import plotly.express as px

        fig = px.line(compare_df, x="snapshot_time", y="price", color="symbol", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

with tab_correlation:
    numeric_cols = ["price", "market_cap", "volume_24h", "percent_change_24h", "circulating_supply"]
    st.plotly_chart(viz.correlation_heatmap(latest_snapshot, numeric_cols), use_container_width=True)

with tab_anomaly:
    anomalies = df[df["is_any_anomaly"]] if "is_any_anomaly" in df.columns else pd.DataFrame()
    st.metric("Total anomalies flagged (all history)", len(anomalies))
    if not anomalies.empty:
        st.dataframe(
            anomalies[["snapshot_time", "symbol", "price", "percent_change_24h", "anomaly_score"]].sort_values(
                "snapshot_time", ascending=False
            ),
            use_container_width=True,
        )
    else:
        st.info("No anomalies detected yet in the collected history.")

with tab_forecast:
    st.subheader(f"{selected_symbol} Price Forecast")
    coin_series = df[df["symbol"] == selected_symbol].sort_values("snapshot_time").set_index("snapshot_time")["price"]

    if len(coin_series.dropna()) < 20:
        st.info("Not enough historical snapshots yet for a reliable forecast (need 20+). Let the scheduler run longer.")
    else:
        method = st.radio("Model", ["Linear Regression", "ARIMA"], horizontal=True)
        steps = st.slider("Steps ahead (30-min intervals)", 3, 48, 12)

        if method == "Linear Regression":
            result = forecast_linear_regression(coin_series, steps_ahead=steps)
        else:
            result = forecast_arima(coin_series, steps_ahead=steps)

        st.line_chart(pd.concat([coin_series.tail(50).reset_index(drop=True), result.forecast.reset_index(drop=True)], axis=1))
        st.write("Backtest metrics:", result.metrics)

with tab_rankings:
    rank_metric = st.selectbox(
        "Rank by", ["percent_change_24h", "volume_24h", "market_cap", "volatility", "volume_to_marketcap_ratio"]
    )
    ranked = latest_snapshot.sort_values(rank_metric, ascending=False)[
        ["symbol", "name", "price", rank_metric]
    ].reset_index(drop=True)
    st.dataframe(ranked, use_container_width=True)

with tab_downloads:
    st.write("Export the current latest snapshot:")
    st.download_button("Download CSV", latest_snapshot.to_csv(index=False), file_name="crypto_snapshot.csv")

    import io

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        latest_snapshot.to_excel(writer, index=False, sheet_name="Snapshot")
    st.download_button(
        "Download Excel", buffer.getvalue(), file_name="crypto_snapshot.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.caption("PDF/HTML reports with KPIs and narrative insights are generated by src/reporting.py "
               "and saved to the reports/ folder on each scheduled run.")

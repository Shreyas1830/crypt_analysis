"""
src/visualization.py
---------------------
Reusable Plotly chart-builder functions shared by the Streamlit dashboard
and the automated report generator, so every chart is defined exactly
once. Matplotlib/Seaborn variants are included for the static EDA charts
that go into PDF reports and notebooks.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

TEMPLATE = "plotly_white"


# --------------------------------------------------------------------------- #
# KPI-adjacent line/area/candlestick charts
# --------------------------------------------------------------------------- #
def price_line_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    coin_df = df[df["symbol"] == symbol].sort_values("snapshot_time")
    fig = px.line(coin_df, x="snapshot_time", y="price", title=f"{symbol} Price Over Time", template=TEMPLATE)
    fig.update_layout(yaxis_title="Price (USD)", xaxis_title="Time")
    return fig


def price_area_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    coin_df = df[df["symbol"] == symbol].sort_values("snapshot_time")
    fig = px.area(coin_df, x="snapshot_time", y="price", title=f"{symbol} Price (Area)", template=TEMPLATE)
    return fig


def candlestick_chart(ohlc_df: pd.DataFrame, symbol: str) -> go.Figure:
    """ohlc_df must contain: snapshot_time (bucketed), open, high, low, close."""
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=ohlc_df["snapshot_time"],
                open=ohlc_df["open"],
                high=ohlc_df["high"],
                low=ohlc_df["low"],
                close=ohlc_df["close"],
            )
        ]
    )
    fig.update_layout(title=f"{symbol} OHLC", template=TEMPLATE)
    return fig


def resample_to_ohlc(df: pd.DataFrame, symbol: str, rule: str = "1H") -> pd.DataFrame:
    coin_df = df[df["symbol"] == symbol].set_index("snapshot_time").sort_index()
    ohlc = coin_df["price"].resample(rule).ohlc()
    return ohlc.reset_index()


# --------------------------------------------------------------------------- #
# Market-structure charts
# --------------------------------------------------------------------------- #
def treemap_market_cap(latest_df: pd.DataFrame, top_n: int = 30) -> go.Figure:
    top = latest_df.nlargest(top_n, "market_cap")
    fig = px.treemap(top, path=["symbol"], values="market_cap", color="percent_change_24h",
                      color_continuous_scale="RdYlGn", title="Market Cap Treemap (Top 30)", template=TEMPLATE)
    return fig


def sunburst_dominance(latest_df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    top = latest_df.nlargest(top_n, "market_cap").copy()
    top["group"] = "Market"
    fig = px.sunburst(top, path=["group", "symbol"], values="market_cap", template=TEMPLATE,
                       title="Market Dominance Sunburst")
    return fig


def bubble_chart(latest_df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        latest_df, x="volume_24h", y="percent_change_24h", size="market_cap", color="symbol",
        hover_name="name", log_x=True, template=TEMPLATE, title="Volume vs 24h Change (bubble = market cap)",
    )
    return fig


def correlation_heatmap(df: pd.DataFrame, columns: list[str]) -> go.Figure:
    corr = df[columns].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu", zmin=-1, zmax=1,
                     title="Correlation Heatmap", template=TEMPLATE)
    return fig


def top_gainers_losers_bar(latest_df: pd.DataFrame, top_n: int = 10) -> tuple[go.Figure, go.Figure]:
    gainers = latest_df.nlargest(top_n, "percent_change_24h")
    losers = latest_df.nsmallest(top_n, "percent_change_24h")
    fig_gain = px.bar(gainers, x="symbol", y="percent_change_24h", title="Top Gainers (24h)",
                       color_discrete_sequence=["#2ecc71"], template=TEMPLATE)
    fig_lose = px.bar(losers, x="symbol", y="percent_change_24h", title="Top Losers (24h)",
                       color_discrete_sequence=["#e74c3c"], template=TEMPLATE)
    return fig_gain, fig_lose


def rolling_volatility_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    coin_df = df[df["symbol"] == symbol].sort_values("snapshot_time")
    fig = px.line(coin_df, x="snapshot_time", y="rolling_volatility_24", template=TEMPLATE,
                  title=f"{symbol} Rolling 24-Period Volatility")
    return fig


def calendar_heatmap_matplotlib(df: pd.DataFrame, symbol: str):
    """Matplotlib/Seaborn calendar-style heatmap (day x hour average return)
    for use in static PDF/Excel reports."""
    coin_df = df[df["symbol"] == symbol].copy()
    pivot = coin_df.pivot_table(index="weekday", columns="hour", values="percent_change_1h", aggfunc="mean")
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = pivot.reindex([d for d in order if d in pivot.index])

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(pivot, cmap="RdYlGn", center=0, ax=ax, cbar_kws={"label": "Avg % change"})
    ax.set_title(f"{symbol}: Avg Hourly Return by Weekday/Hour")
    return fig


def missing_values_heatmap(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df.isna(), cbar=False, cmap="viridis", ax=ax)
    ax.set_title("Missing Value Heatmap")
    return fig


def distribution_grid(df: pd.DataFrame, columns: list[str]):
    fig, axes = plt.subplots(1, len(columns), figsize=(5 * len(columns), 4))
    if len(columns) == 1:
        axes = [axes]
    for ax, col in zip(axes, columns):
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        ax.set_title(f"Distribution: {col}")
    fig.tight_layout()
    return fig

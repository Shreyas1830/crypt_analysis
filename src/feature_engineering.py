"""
src/feature_engineering.py
---------------------------
Builds all derived analytical features on top of the cleaned snapshot
history: timestamp features, rolling statistics, returns/volatility,
momentum & ranking metrics, and market-structure ratios (dominance,
market share, volume/market-cap ratio, etc).

All rolling features are computed *per coin*, ordered by snapshot_time,
since the raw table is a stack of repeated snapshots over time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import settings
from src.logger import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Timestamp engineering
# --------------------------------------------------------------------------- #
def add_timestamp_features(df: pd.DataFrame, ts_col: str = "snapshot_time") -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df[ts_col], utc=True)
    df["hour"] = ts.dt.hour
    df["day"] = ts.dt.day
    df["week"] = ts.dt.isocalendar().week.astype(int)
    df["month"] = ts.dt.month
    df["quarter"] = ts.dt.quarter
    df["year"] = ts.dt.year
    df["weekday"] = ts.dt.day_name()
    df["is_weekend"] = ts.dt.dayofweek.isin([5, 6])
    df["is_business_hour"] = ts.dt.hour.between(9, 17)
    return df


# --------------------------------------------------------------------------- #
# Per-coin rolling / return features
# --------------------------------------------------------------------------- #
def _per_coin_return_features(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("snapshot_time").copy()

    group["hourly_return"] = group["price"].pct_change(periods=2)  # ~30min cadence -> 2 steps/hr
    group["daily_return"] = group["price"].pct_change(periods=48)  # 48 steps/day
    group["weekly_return"] = group["price"].pct_change(periods=336)  # 336 steps/week

    for w in settings.ROLLING_WINDOWS:
        group[f"rolling_mean_{w}"] = group["price"].rolling(w, min_periods=1).mean()
        group[f"rolling_median_{w}"] = group["price"].rolling(w, min_periods=1).median()
        group[f"rolling_std_{w}"] = group["price"].rolling(w, min_periods=1).std()

    group["moving_average_12"] = group["price"].rolling(12, min_periods=1).mean()
    group["ema_12"] = group["price"].ewm(span=12, adjust=False).mean()

    group["price_momentum"] = group["price"].diff()
    group["price_acceleration"] = group["price_momentum"].diff()
    group["price_change_velocity"] = group["price"].diff() / group["snapshot_time"].diff().dt.total_seconds().replace(0, np.nan)

    group["volatility"] = group["hourly_return"].rolling(12, min_periods=2).std()
    group["rolling_volatility_24"] = group["hourly_return"].rolling(24, min_periods=2).std()

    group["volume_growth"] = group["volume_24h"].pct_change()
    group["market_cap_growth"] = group["market_cap"].pct_change()

    # Simplified rolling Sharpe: mean(return) / std(return) over trailing window
    roll_ret_mean = group["hourly_return"].rolling(24, min_periods=2).mean()
    roll_ret_std = group["hourly_return"].rolling(24, min_periods=2).std()
    group["sharpe_ratio_simplified"] = (roll_ret_mean / roll_ret_std).replace([np.inf, -np.inf], np.nan)

    # Momentum score: normalized blend of short vs long return
    group["momentum_score"] = group["hourly_return"].fillna(0) * 0.4 + group["daily_return"].fillna(0) * 0.6

    return group


def add_return_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["symbol", "snapshot_time"]).copy()
    result = df.groupby("symbol", group_keys=False).apply(_per_coin_return_features)
    if "symbol" not in result.columns:
        result["symbol"] = df.loc[result.index, "symbol"].values
    return result


# --------------------------------------------------------------------------- #
# Cross-sectional (per-snapshot, across coins) features
# --------------------------------------------------------------------------- #
def add_market_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    total_mcap = df.groupby("snapshot_time")["market_cap"].transform("sum")
    total_vol = df.groupby("snapshot_time")["volume_24h"].transform("sum")

    df["market_dominance"] = df["market_cap"] / total_mcap
    df["market_cap_share"] = df["market_dominance"]  # alias, same metric
    df["volume_share"] = df["volume_24h"] / total_vol
    df["volume_to_marketcap_ratio"] = df["volume_24h"] / df["market_cap"]

    df["price_rank"] = df.groupby("snapshot_time")["price"].rank(ascending=False, method="min")

    # Relative strength vs. BTC (or the top-ranked coin if BTC absent in a snapshot)
    def _relative_strength(group: pd.DataFrame) -> pd.Series:
        btc_row = group[group["symbol"] == "BTC"]
        benchmark_return = (
            btc_row["percent_change_24h"].iloc[0] if not btc_row.empty else group["percent_change_24h"].median()
        )
        return group["percent_change_24h"] - benchmark_return

    df["relative_strength_vs_btc"] = df.groupby("snapshot_time", group_keys=False).apply(_relative_strength)

    return df


def add_rolling_correlation(df: pd.DataFrame, symbol_a: str = "BTC", symbol_b: str = "ETH", window: int = 24) -> pd.Series:
    """Rolling correlation between two coins' returns over time - returned as
    a standalone series indexed by snapshot_time for charting purposes."""
    pivot = df.pivot_table(index="snapshot_time", columns="symbol", values="hourly_return")
    if symbol_a not in pivot.columns or symbol_b not in pivot.columns:
        logger.warning("Rolling correlation requested for missing symbols: %s / %s", symbol_a, symbol_b)
        return pd.Series(dtype=float)
    return pivot[symbol_a].rolling(window).corr(pivot[symbol_b])


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def build_feature_set(clean_df: pd.DataFrame) -> pd.DataFrame:
    df = add_timestamp_features(clean_df)
    df = add_return_and_rolling_features(df)
    df = add_market_structure_features(df)
    logger.info("Feature engineering complete: %d rows, %d columns", *df.shape)
    return df

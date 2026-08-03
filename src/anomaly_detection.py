"""
src/anomaly_detection.py
-------------------------
Detects market anomalies: flash crashes, price/volume spikes, and
market-cap anomalies. Combines statistical (Z-score) and ML-based
(Isolation Forest, Local Outlier Factor) approaches, per coin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from config import settings
from src.logger import get_logger

logger = get_logger(__name__)


def flag_flash_crashes(df: pd.DataFrame, drop_threshold: float = -0.08) -> pd.DataFrame:
    """A flash crash = a single-step price return below `drop_threshold`
    (default -8%) within one snapshot interval (~30 min)."""
    df = df.copy()
    df["is_flash_crash"] = df.groupby("symbol")["price"].pct_change() < drop_threshold
    return df


def flag_price_spikes(df: pd.DataFrame, spike_threshold: float = 0.10) -> pd.DataFrame:
    df = df.copy()
    df["is_price_spike"] = df.groupby("symbol")["price"].pct_change() > spike_threshold
    return df


def flag_volume_spikes(df: pd.DataFrame, z_threshold: float | None = None) -> pd.DataFrame:
    z_threshold = z_threshold or settings.ZSCORE_THRESHOLD
    df = df.copy()

    def _zscore(s: pd.Series) -> pd.Series:
        std = s.std()
        return (s - s.mean()) / std if std and not np.isnan(std) else pd.Series(0, index=s.index)

    df["volume_zscore"] = df.groupby("symbol")["volume_24h"].transform(_zscore)
    df["is_volume_spike"] = df["volume_zscore"].abs() > z_threshold
    return df


def flag_market_cap_anomalies(df: pd.DataFrame, z_threshold: float | None = None) -> pd.DataFrame:
    z_threshold = z_threshold or settings.ZSCORE_THRESHOLD
    df = df.copy()

    def _zscore(s: pd.Series) -> pd.Series:
        std = s.std()
        return (s - s.mean()) / std if std and not np.isnan(std) else pd.Series(0, index=s.index)

    df["market_cap_zscore"] = df.groupby("symbol")["market_cap"].transform(_zscore)
    df["is_market_cap_anomaly"] = df["market_cap_zscore"].abs() > z_threshold
    return df


def flag_ml_anomalies(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Cross-sectional anomaly detection (per snapshot, across coins) using
    Isolation Forest and Local Outlier Factor."""
    columns = columns or ["price", "market_cap", "volume_24h", "percent_change_24h"]
    df = df.copy()
    df["is_isolation_forest_anomaly"] = False
    df["is_lof_anomaly"] = False

    for snapshot_time, group in df.groupby("snapshot_time"):
        if len(group) < 10:
            continue  # not enough points for a meaningful model
        data = group[columns].fillna(group[columns].median())

        iso = IsolationForest(contamination=settings.ISOLATION_FOREST_CONTAMINATION, random_state=42)
        iso_preds = iso.fit_predict(data) == -1

        lof = LocalOutlierFactor(n_neighbors=min(20, len(group) - 1), contamination=settings.ISOLATION_FOREST_CONTAMINATION)
        lof_preds = lof.fit_predict(data) == -1

        df.loc[group.index, "is_isolation_forest_anomaly"] = iso_preds
        df.loc[group.index, "is_lof_anomaly"] = lof_preds

    return df


def run_anomaly_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = flag_flash_crashes(df)
    df = flag_price_spikes(df)
    df = flag_volume_spikes(df)
    df = flag_market_cap_anomalies(df)
    df = flag_ml_anomalies(df)

    anomaly_cols = [
        "is_flash_crash",
        "is_price_spike",
        "is_volume_spike",
        "is_market_cap_anomaly",
        "is_isolation_forest_anomaly",
        "is_lof_anomaly",
    ]
    df["anomaly_score"] = df[anomaly_cols].sum(axis=1)
    df["is_any_anomaly"] = df["anomaly_score"] > 0

    n_anomalies = int(df["is_any_anomaly"].sum())
    logger.info("Anomaly detection complete: %d anomalous rows flagged.", n_anomalies)
    return df

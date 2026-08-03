"""
src/clustering.py
------------------
Clusters cryptocurrencies by behavior (volatility, returns, market cap,
volume) using KMeans, Agglomerative (hierarchical), and DBSCAN. Uses the
latest snapshot's engineered features, standardized before clustering.
"""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler

from src.logger import get_logger

logger = get_logger(__name__)

CLUSTER_FEATURES = [
    "price",
    "market_cap",
    "volume_24h",
    "percent_change_24h",
    "percent_change_7d",
    "volatility",
    "market_dominance",
]


def _prepare_matrix(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, StandardScaler]:
    available = [f for f in features if f in df.columns]
    matrix = df[available].fillna(df[available].median())
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    return pd.DataFrame(scaled, columns=available, index=df.index), scaler


def cluster_kmeans(df: pd.DataFrame, n_clusters: int = 5) -> pd.Series:
    matrix, _ = _prepare_matrix(df, CLUSTER_FEATURES)
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(matrix)
    return pd.Series(labels, index=df.index, name="kmeans_cluster")


def cluster_hierarchical(df: pd.DataFrame, n_clusters: int = 5) -> pd.Series:
    matrix, _ = _prepare_matrix(df, CLUSTER_FEATURES)
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(matrix)
    return pd.Series(labels, index=df.index, name="hierarchical_cluster")


def cluster_dbscan(df: pd.DataFrame, eps: float = 1.2, min_samples: int = 4) -> pd.Series:
    matrix, _ = _prepare_matrix(df, CLUSTER_FEATURES)
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(matrix)  # -1 = noise/outlier coin
    return pd.Series(labels, index=df.index, name="dbscan_cluster")


def run_all_clustering(latest_snapshot_df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
    """Takes ONE snapshot (cross-section across coins) and returns it with
    three cluster-label columns attached."""
    df = latest_snapshot_df.copy()
    df["kmeans_cluster"] = cluster_kmeans(df, n_clusters)
    df["hierarchical_cluster"] = cluster_hierarchical(df, n_clusters)
    df["dbscan_cluster"] = cluster_dbscan(df)
    logger.info("Clustering complete on %d coins.", len(df))
    return df

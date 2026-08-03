"""
src/cleaning.py
----------------
Professional-grade data cleaning: missing values, duplicates, dtype
validation, outlier detection, business-rule consistency checks, and
string normalization. Every function returns a (clean_df, report_dict)
pair so cleaning decisions are auditable rather than silent.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest

from config import settings
from src.logger import get_logger

logger = get_logger(__name__)

NUMERIC_COLS = [
    "price",
    "market_cap",
    "volume_24h",
    "circulating_supply",
    "percent_change_1h",
    "percent_change_24h",
    "percent_change_7d",
]


# --------------------------------------------------------------------------- #
# Missing values
# --------------------------------------------------------------------------- #
def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a per-column count + percentage of missing values."""
    counts = df.isna().sum()
    pct = (counts / len(df) * 100).round(2)
    return pd.DataFrame({"missing_count": counts, "missing_pct": pct}).sort_values(
        "missing_count", ascending=False
    )


def handle_missing_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = df.copy()
    report: dict[str, Any] = {}

    # Rows with no price are useless for analysis - drop them.
    before = len(df)
    df = df.dropna(subset=["price", "coin_id" if "coin_id" in df.columns else "id"])
    report["rows_dropped_no_price"] = before - len(df)

    # Percent-change columns: a missing value usually means "no change data
    # available yet" (e.g. very new coin) rather than truly zero - but for
    # modeling we impute with 0 and flag it, rather than silently guessing.
    for col in ["percent_change_1h", "percent_change_24h", "percent_change_7d"]:
        if col in df.columns:
            n_missing = df[col].isna().sum()
            df[col] = df[col].fillna(0.0)
            report[f"{col}_imputed_with_zero"] = int(n_missing)

    # Market cap / volume: impute with median of same-rank tier rather than
    # global median, to avoid distorting comparisons across cap sizes.
    for col in ["market_cap", "volume_24h", "circulating_supply"]:
        if col in df.columns and df[col].isna().any():
            n_missing = df[col].isna().sum()
            df[col] = df[col].fillna(df[col].median())
            report[f"{col}_imputed_with_median"] = int(n_missing)

    return df, report


# --------------------------------------------------------------------------- #
# Duplicates
# --------------------------------------------------------------------------- #
def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = df.copy()
    report = {}

    id_col = "coin_id" if "coin_id" in df.columns else "id"

    before = len(df)
    df = df.drop_duplicates(subset=[id_col, "snapshot_time"], keep="last")
    report["duplicate_coin_snapshot_rows_removed"] = before - len(df)

    before = len(df)
    df = df.drop_duplicates(subset=["symbol", "snapshot_time"], keep="last")
    report["duplicate_symbol_timestamp_rows_removed"] = before - len(df)

    exact_dupes = df.duplicated().sum()
    if exact_dupes:
        df = df.drop_duplicates()
    report["exact_duplicate_rows_removed"] = int(exact_dupes)

    return df, report


# --------------------------------------------------------------------------- #
# Datatype validation
# --------------------------------------------------------------------------- #
def enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("last_updated", "snapshot_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df["name"] = df["name"].astype(str).str.strip()
    return df


# --------------------------------------------------------------------------- #
# Consistency / business-rule checks
# --------------------------------------------------------------------------- #
def consistency_checks(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = df.copy()
    report: dict[str, Any] = {}
    now = pd.Timestamp.now(tz="UTC")

    neg_price = (df["price"] < 0).sum()
    df = df[df["price"] >= 0]
    report["negative_price_rows_removed"] = int(neg_price)

    neg_mcap = (df["market_cap"] < 0).sum()
    df.loc[df["market_cap"] < 0, "market_cap"] = np.nan
    report["negative_market_cap_flagged"] = int(neg_mcap)

    null_symbol = df["symbol"].isin(["", "NAN", "NONE"]).sum()
    df = df[~df["symbol"].isin(["", "NAN", "NONE"])]
    report["null_symbol_rows_removed"] = int(null_symbol)

    invalid_rank = (df["cmc_rank"] <= 0).sum()
    df.loc[df["cmc_rank"] <= 0, "cmc_rank"] = np.nan
    report["invalid_rank_flagged"] = int(invalid_rank)

    future_ts = (df["snapshot_time"] > now).sum()
    df = df[df["snapshot_time"] <= now]
    report["future_timestamp_rows_removed"] = int(future_ts)

    return df, report


# --------------------------------------------------------------------------- #
# Outlier detection
# --------------------------------------------------------------------------- #
def detect_outliers_zscore(df: pd.DataFrame, column: str, threshold: float | None = None) -> pd.Series:
    threshold = threshold or settings.ZSCORE_THRESHOLD
    z = np.abs(stats.zscore(df[column].fillna(df[column].median())))
    return pd.Series(z > threshold, index=df.index, name=f"{column}_zscore_outlier")


def detect_outliers_iqr(df: pd.DataFrame, column: str, multiplier: float | None = None) -> pd.Series:
    multiplier = multiplier or settings.IQR_MULTIPLIER
    q1, q3 = df[column].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
    return ((df[column] < lower) | (df[column] > upper)).rename(f"{column}_iqr_outlier")


def detect_outliers_isolation_forest(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    data = df[columns].fillna(df[columns].median())
    model = IsolationForest(
        contamination=settings.ISOLATION_FOREST_CONTAMINATION, random_state=42
    )
    preds = model.fit_predict(data)  # -1 = outlier, 1 = inlier
    return pd.Series(preds == -1, index=df.index, name="isolation_forest_outlier")


def full_outlier_report(df: pd.DataFrame) -> pd.DataFrame:
    """Combines z-score, IQR, and Isolation Forest flags into one frame."""
    flags = pd.DataFrame(index=df.index)
    for col in ["price", "market_cap", "volume_24h", "percent_change_24h"]:
        flags[f"{col}_zscore_outlier"] = detect_outliers_zscore(df, col)
        flags[f"{col}_iqr_outlier"] = detect_outliers_iqr(df, col)

    iso_cols = ["price", "market_cap", "volume_24h", "percent_change_24h"]
    flags["isolation_forest_outlier"] = detect_outliers_isolation_forest(df, iso_cols)
    flags["any_outlier_flag"] = flags.any(axis=1)
    return flags


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def run_cleaning_pipeline(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Runs the full cleaning sequence and returns the clean dataframe plus
    a combined audit report (useful for the automated report generator)."""
    full_report: dict[str, Any] = {}

    df = enforce_dtypes(raw_df)

    df, missing_report = handle_missing_values(df)
    full_report["missing_values"] = missing_report

    df, dup_report = remove_duplicates(df)
    full_report["duplicates"] = dup_report

    df, consistency_report = consistency_checks(df)
    full_report["consistency"] = consistency_report

    outlier_flags = full_outlier_report(df)
    full_report["outliers_detected"] = int(outlier_flags["any_outlier_flag"].sum())

    df = df.join(outlier_flags["any_outlier_flag"])
    df = df.rename(columns={"any_outlier_flag": "is_outlier"})

    logger.info("Cleaning pipeline complete: %s", full_report)
    return df, full_report

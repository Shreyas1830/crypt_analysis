"""
src/api.py
----------
Extraction layer. Pulls the top-N cryptocurrencies from the CoinMarketCap
sandbox API, with automatic retry/backoff and a CoinGecko fallback (free,
no key) if CMC is unreachable. Validates the response schema before
handing data downstream, and never crashes the scheduler on a single
failed run.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
import requests
from requests.exceptions import RequestException

from config import settings
from src.logger import get_logger

logger = get_logger(__name__)

# Columns every downstream module expects, regardless of which upstream
# provider produced the data. This is the "schema contract" of the pipeline.
REQUIRED_COLUMNS = [
    "id",
    "name",
    "symbol",
    "cmc_rank",
    "price",
    "market_cap",
    "volume_24h",
    "circulating_supply",
    "percent_change_1h",
    "percent_change_24h",
    "percent_change_7d",
    "last_updated",
    "snapshot_time",
]


class CryptoAPIClient:
    """Fetches current market snapshots with retry + provider fallback."""

    def __init__(self) -> None:
        self.cmc_cfg = settings.CMC
        self.gecko_cfg = settings.COINGECKO

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def fetch_snapshot(self) -> Optional[pd.DataFrame]:
        """Try CMC sandbox first, fall back to CoinGecko. Returns a
        validated, schema-normalized DataFrame or None if both fail."""
        df = self._fetch_with_retry(self._fetch_cmc, provider_name="CoinMarketCap")
        if df is None:
            logger.warning("CMC fetch failed after retries -> falling back to CoinGecko")
            df = self._fetch_with_retry(self._fetch_coingecko, provider_name="CoinGecko")

        if df is None:
            logger.error("Both providers failed. No data fetched this cycle.")
            return None

        df = self._validate_schema(df)
        return df

    # ------------------------------------------------------------------ #
    # Retry / backoff wrapper
    # ------------------------------------------------------------------ #
    def _fetch_with_retry(self, fetch_fn, provider_name: str) -> Optional[pd.DataFrame]:
        cfg = self.cmc_cfg if provider_name == "CoinMarketCap" else self.gecko_cfg
        for attempt in range(1, cfg.max_retries + 1):
            try:
                logger.info("Fetching from %s (attempt %d/%d)", provider_name, attempt, cfg.max_retries)
                return fetch_fn()
            except RequestException as exc:
                wait = cfg.backoff_factor**attempt
                logger.warning(
                    "%s request failed (%s). Retrying in %.1fs...", provider_name, exc, wait
                )
                time.sleep(wait)
            except (KeyError, ValueError) as exc:
                logger.error("%s returned malformed data: %s", provider_name, exc)
                return None
        logger.error("%s exhausted all %d retries.", provider_name, cfg.max_retries)
        return None

    # ------------------------------------------------------------------ #
    # Provider-specific fetch + normalize logic
    # ------------------------------------------------------------------ #
    def _fetch_cmc(self) -> pd.DataFrame:
        cfg = self.cmc_cfg
        url = f"{cfg.base_url}{cfg.listings_endpoint}"
        headers = {"X-CMC_PRO_API_KEY": cfg.api_key, "Accept": "application/json"}
        params = {"start": "1", "limit": str(cfg.top_n), "convert": cfg.convert}

        resp = requests.get(url, headers=headers, params=params, timeout=cfg.timeout_seconds)
        resp.raise_for_status()
        payload: dict[str, Any] = resp.json()

        rows = []
        snapshot_time = datetime.now(timezone.utc).isoformat()
        for coin in payload["data"]:
            quote = coin["quote"][cfg.convert]
            rows.append(
                {
                    "id": coin["id"],
                    "name": coin["name"],
                    "symbol": coin["symbol"],
                    "cmc_rank": coin.get("cmc_rank"),
                    "price": quote.get("price"),
                    "market_cap": quote.get("market_cap"),
                    "volume_24h": quote.get("volume_24h"),
                    "circulating_supply": coin.get("circulating_supply"),
                    "percent_change_1h": quote.get("percent_change_1h"),
                    "percent_change_24h": quote.get("percent_change_24h"),
                    "percent_change_7d": quote.get("percent_change_7d"),
                    "last_updated": quote.get("last_updated"),
                    "snapshot_time": snapshot_time,
                }
            )
        return pd.DataFrame(rows)

    def _fetch_coingecko(self) -> pd.DataFrame:
        cfg = self.gecko_cfg
        url = f"{cfg.base_url}{cfg.markets_endpoint}"
        params = {
            "vs_currency": cfg.vs_currency,
            "order": "market_cap_desc",
            "per_page": str(cfg.top_n),
            "page": "1",
            "price_change_percentage": "1h,24h,7d",
        }
        resp = requests.get(url, params=params, timeout=cfg.timeout_seconds)
        resp.raise_for_status()
        payload = resp.json()

        snapshot_time = datetime.now(timezone.utc).isoformat()
        rows = []
        for coin in payload:
            rows.append(
                {
                    "id": coin["id"],
                    "name": coin["name"],
                    "symbol": coin["symbol"].upper(),
                    "cmc_rank": coin.get("market_cap_rank"),
                    "price": coin.get("current_price"),
                    "market_cap": coin.get("market_cap"),
                    "volume_24h": coin.get("total_volume"),
                    "circulating_supply": coin.get("circulating_supply"),
                    "percent_change_1h": coin.get("price_change_percentage_1h_in_currency"),
                    "percent_change_24h": coin.get("price_change_percentage_24h_in_currency"),
                    "percent_change_7d": coin.get("price_change_percentage_7d_in_currency"),
                    "last_updated": coin.get("last_updated"),
                    "snapshot_time": snapshot_time,
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # Schema validation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_schema(df: pd.DataFrame) -> pd.DataFrame:
        missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(f"API response missing required columns: {missing_cols}")
        # Enforce dtypes early so downstream modules can trust the contract.
        numeric_cols = [
            "price",
            "market_cap",
            "volume_24h",
            "circulating_supply",
            "percent_change_1h",
            "percent_change_24h",
            "percent_change_7d",
            "cmc_rank",
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["id"] = df["id"].astype(str).str.strip()
        df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce", utc=True)
        df["snapshot_time"] = pd.to_datetime(df["snapshot_time"], errors="coerce", utc=True)
        return df[REQUIRED_COLUMNS]


if __name__ == "__main__":
    client = CryptoAPIClient()
    snapshot = client.fetch_snapshot()
    if snapshot is not None:
        print(snapshot.head(10))
        print(f"\nFetched {len(snapshot)} rows.")

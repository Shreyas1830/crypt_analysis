"""
config/settings.py
-------------------
Centralized configuration for the Crypto Analytics Platform.
All secrets are read from environment variables (see .env.example) so that
no credentials are ever hard-coded or committed to version control.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a .env file if present (does nothing in prod if absent)
load_dotenv()

# --------------------------------------------------------------------------- #
# Path configuration
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARCHIVE_DATA_DIR = DATA_DIR / "archive"
REPORTS_DIR = ROOT_DIR / "reports"
LOG_DIR = ROOT_DIR / "logs"

for _dir in (RAW_DATA_DIR, PROCESSED_DATA_DIR, ARCHIVE_DATA_DIR, REPORTS_DIR, LOG_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# API configuration (CoinMarketCap sandbox by default, CoinGecko as a free
# fallback that needs no key at all)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CMCConfig:
    # The CMC *sandbox* environment returns realistic mock data and does not
    # count against a production quota - perfect for building/testing a
    # portfolio project without needing a paid key.
    base_url: str = "https://sandbox-api.coinmarketcap.com/v1"
    listings_endpoint: str = "/cryptocurrency/listings/latest"
    # The sandbox accepts the documented demo key. Override via env var
    # CMC_API_KEY if you move to the production API.
    api_key: str = field(
        default_factory=lambda: os.getenv(
            "CMC_API_KEY", "b54bcf4d-1bca-4e8e-9a24-22ff2c3d462c"  # public sandbox key
        )
    )
    top_n: int = int(os.getenv("CMC_TOP_N", "100"))
    convert: str = os.getenv("CMC_CONVERT", "USD")
    timeout_seconds: int = 15
    max_retries: int = 5
    backoff_factor: float = 1.5  # exponential backoff: 1.5, 3, 6, 12, 24 seconds


@dataclass(frozen=True)
class CoinGeckoConfig:
    """Free fallback API - no key required, used automatically if the
    CMC sandbox is unreachable or CMC_API_KEY is not set."""

    base_url: str = "https://api.coingecko.com/api/v3"
    markets_endpoint: str = "/coins/markets"
    top_n: int = int(os.getenv("CMC_TOP_N", "100"))
    vs_currency: str = "usd"
    timeout_seconds: int = 15
    max_retries: int = 5
    backoff_factor: float = 1.5


CMC = CMCConfig()
COINGECKO = CoinGeckoConfig()

# --------------------------------------------------------------------------- #
# Database configuration
# --------------------------------------------------------------------------- #
DB_ENGINE = os.getenv("DB_ENGINE", "sqlite")  # "sqlite" or "postgresql"
SQLITE_PATH = DATA_DIR / "crypto_market.db"

# Fill these via env vars if DB_ENGINE == "postgresql"
POSTGRES_URI = os.getenv(
    "POSTGRES_URI", "postgresql+psycopg2://user:password@localhost:5432/crypto_analytics"
)


def get_database_url() -> str:
    if DB_ENGINE == "postgresql":
        return POSTGRES_URI
    return f"sqlite:///{SQLITE_PATH}"


# --------------------------------------------------------------------------- #
# Scheduler configuration
# --------------------------------------------------------------------------- #
FETCH_INTERVAL_MINUTES = int(os.getenv("FETCH_INTERVAL_MINUTES", "30"))

# --------------------------------------------------------------------------- #
# Logging configuration
# --------------------------------------------------------------------------- #
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = LOG_DIR / "pipeline.log"

# --------------------------------------------------------------------------- #
# Anomaly / feature engineering thresholds
# --------------------------------------------------------------------------- #
ZSCORE_THRESHOLD = float(os.getenv("ZSCORE_THRESHOLD", "3.0"))
IQR_MULTIPLIER = float(os.getenv("IQR_MULTIPLIER", "1.5"))
ISOLATION_FOREST_CONTAMINATION = float(os.getenv("ISOLATION_FOREST_CONTAMINATION", "0.03"))
ROLLING_WINDOWS = (3, 6, 12, 24)  # in number of snapshots (30-min cadence -> 1.5h,3h,6h,12h

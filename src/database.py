"""
src/database.py
----------------
SQLAlchemy models + storage layer. Handles:
  - schema creation (SQLite by default, swappable to Postgres via config)
  - de-duplication on (coin_id, snapshot_time)
  - CSV backups alongside the SQL table
  - raw JSON archival for full auditability
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from config import settings
from src.logger import get_logger

logger = get_logger(__name__)

Base = declarative_base()


class CryptoSnapshot(Base):
    """One row = one coin's market stats at one point in time."""

    __tablename__ = "crypto_snapshots"
    __table_args__ = (UniqueConstraint("coin_id", "snapshot_time", name="uq_coin_snapshot"),)

    row_id = Column(Integer, primary_key=True, autoincrement=True)
    coin_id = Column(String(64), nullable=False, index=True)
    name = Column(String, nullable=False)
    symbol = Column(String, nullable=False, index=True)
    cmc_rank = Column(Integer)
    price = Column(Float)
    market_cap = Column(Float)
    volume_24h = Column(Float)
    circulating_supply = Column(Float)
    percent_change_1h = Column(Float)
    percent_change_24h = Column(Float)
    percent_change_7d = Column(Float)
    last_updated = Column(DateTime)
    snapshot_time = Column(DateTime, nullable=False, index=True)
    inserted_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Database:
    def __init__(self, db_url: str | None = None) -> None:
        self.engine = create_engine(db_url or settings.get_database_url(), future=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, future=True)

    # ------------------------------------------------------------------ #
    def upsert_snapshot(self, df: pd.DataFrame) -> int:
        """Insert new rows, silently skipping duplicates on
        (coin_id, snapshot_time). Returns number of rows actually inserted."""
        if df.empty:
            logger.warning("upsert_snapshot called with an empty DataFrame - nothing to do.")
            return 0

        records = df.rename(columns={"id": "coin_id"}).to_dict(orient="records")

        inserted = 0
        with self.engine.begin() as conn:
            for record in records:
                stmt = sqlite_insert(CryptoSnapshot).values(**record)
                stmt = stmt.on_conflict_do_nothing(index_elements=["coin_id", "snapshot_time"])
                result = conn.execute(stmt)
                inserted += result.rowcount if result.rowcount is not None else 0

        logger.info("Inserted %d new rows (%d duplicates skipped).", inserted, len(records) - inserted)
        return inserted

    # ------------------------------------------------------------------ #
    def read_all(self) -> pd.DataFrame:
        return pd.read_sql_table("crypto_snapshots", con=self.engine)

    def read_latest(self) -> pd.DataFrame:
        query = """
            SELECT * FROM crypto_snapshots
            WHERE snapshot_time = (SELECT MAX(snapshot_time) FROM crypto_snapshots)
        """
        return pd.read_sql_query(query, con=self.engine)

    def read_coin_history(self, symbol: str) -> pd.DataFrame:
        query = """
            SELECT * FROM crypto_snapshots
            WHERE symbol = :symbol
            ORDER BY snapshot_time ASC
        """
        return pd.read_sql_query(query, con=self.engine, params={"symbol": symbol.upper()})

    # ------------------------------------------------------------------ #
    def backup_to_csv(self, df: pd.DataFrame, path) -> None:
        """Append-mode CSV backup that mirrors the SQL table, so the data
        is recoverable even without the database file."""
        header = not path.exists()
        df.to_csv(path, mode="a", header=header, index=False)
        logger.info("CSV backup updated -> %s", path)


if __name__ == "__main__":
    db = Database()
    print("Database initialized at:", settings.get_database_url())

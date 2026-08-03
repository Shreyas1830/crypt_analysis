"""
src/pipeline.py
----------------
Orchestrates one full end-to-end pipeline run:
  fetch -> store -> clean -> feature-engineer -> anomaly-detect -> report

This is what the scheduler calls every FETCH_INTERVAL_MINUTES, and what
app.py / the Streamlit dashboard call for an on-demand refresh.
"""

from __future__ import annotations

from config import settings
from src.anomaly_detection import run_anomaly_pipeline
from src.api import CryptoAPIClient
from src.cleaning import run_cleaning_pipeline
from src.database import Database
from src.feature_engineering import build_feature_set
from src.logger import get_logger
from src.reporting import generate_all_reports

logger = get_logger(__name__)


def run_pipeline_once(generate_reports: bool = False) -> dict:
    """Executes a single fetch-clean-feature-anomaly cycle. Returns a
    summary dict describing what happened, for logging/monitoring."""
    summary: dict = {"status": "failed"}

    client = CryptoAPIClient()
    raw_df = client.fetch_snapshot()
    if raw_df is None or raw_df.empty:
        logger.error("Pipeline aborted: no data returned from any provider this cycle.")
        return summary

    db = Database()
    inserted = db.upsert_snapshot(raw_df)
    db.backup_to_csv(raw_df, settings.RAW_DATA_DIR / "snapshots_backup.csv")

    history_df = db.read_all()
    clean_df, clean_report = run_cleaning_pipeline(history_df)
    feature_df = build_feature_set(clean_df)
    final_df = run_anomaly_pipeline(feature_df)

    final_df.to_csv(settings.PROCESSED_DATA_DIR / "processed_latest.csv", index=False)

    latest_snapshot = final_df[final_df["snapshot_time"] == final_df["snapshot_time"].max()]

    summary = {
        "status": "success",
        "rows_inserted": inserted,
        "total_rows_in_db": len(history_df),
        "coins_in_latest_snapshot": len(latest_snapshot),
        "cleaning_report": clean_report,
        "anomalies_flagged": int(latest_snapshot["is_any_anomaly"].sum()) if "is_any_anomaly" in latest_snapshot.columns else 0,
    }

    if generate_reports:
        report_paths = generate_all_reports(latest_snapshot, final_df, period="Daily")
        summary["reports"] = {k: str(v) if v else None for k, v in report_paths.items()}

    logger.info("Pipeline run complete: %s", summary)
    return summary


if __name__ == "__main__":
    result = run_pipeline_once(generate_reports=True)
    print(result)

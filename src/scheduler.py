"""
src/scheduler.py
-----------------
Runs the pipeline automatically every FETCH_INTERVAL_MINUTES using
APScheduler (BlockingScheduler). Designed to run as a long-lived
background process (systemd service, Docker container, or `nohup python
src/scheduler.py &`) separate from the Streamlit dashboard process.

Includes a misfire grace period and a job-error listener so a single
failed run never kills the whole scheduler.
"""

from __future__ import annotations

from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED
from apscheduler.schedulers.blocking import BlockingScheduler

from config import settings
from src.logger import get_logger
from src.pipeline import run_pipeline_once

logger = get_logger(__name__)


def scheduled_job() -> None:
    logger.info("Scheduled pipeline run starting...")
    result = run_pipeline_once(generate_reports=False)
    if result.get("status") != "success":
        logger.error("Scheduled run did NOT complete successfully: %s", result)
    else:
        logger.info(
            "Scheduled run OK -> inserted %d new rows, %d anomalies flagged.",
            result.get("rows_inserted", 0),
            result.get("anomalies_flagged", 0),
        )


def _job_listener(event) -> None:
    if event.exception:
        logger.error("Job crashed: %s", event.exception)
    else:
        logger.debug("Job finished cleanly.")


def start_scheduler() -> None:
    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_listener(_job_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)

    scheduler.add_job(
        scheduled_job,
        trigger="interval",
        minutes=settings.FETCH_INTERVAL_MINUTES,
        next_run_time=None,  # runs immediately on start, then every interval
        misfire_grace_time=120,
        max_instances=1,
        coalesce=True,
    )

    logger.info("Scheduler started - fetching every %d minute(s).", settings.FETCH_INTERVAL_MINUTES)
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped by user.")


if __name__ == "__main__":
    start_scheduler()

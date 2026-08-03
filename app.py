"""
app.py
------
Root entry point. Two ways to use this project:

1. Run the Streamlit dashboard:
       streamlit run app.py

2. Run a one-off pipeline cycle from the CLI (fetch -> clean -> feature ->
   anomaly -> optional reports), useful for cron/Task Scheduler or a quick
   manual test:
       python app.py --once
       python app.py --once --with-reports

3. Start the always-on 30-minute scheduler (separate long-lived process):
       python src/scheduler.py
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Crypto Analytics Platform entry point")
    parser.add_argument("--once", action="store_true", help="Run a single pipeline cycle and exit")
    parser.add_argument("--with-reports", action="store_true", help="Generate HTML/PDF/Excel/CSV reports too")
    args = parser.parse_args()

    if args.once:
        from src.pipeline import run_pipeline_once

        result = run_pipeline_once(generate_reports=args.with_reports)
        print(result)
        return

    # No CLI flag -> assume we were launched via `streamlit run app.py`
    runpy.run_path(str(Path(__file__).resolve().parent / "dashboard" / "app.py"), run_name="__main__")


if __name__ == "__main__":
    main()

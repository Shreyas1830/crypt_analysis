"""
src/reporting.py
-----------------
Generates automated daily/weekly/monthly reports in HTML, PDF, Excel, and
CSV formats, including KPIs, top movers, and narrative insights. Designed
to be called from the scheduler right after a fetch cycle, or on-demand
from the Streamlit dashboard's "Download report" buttons.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from jinja2 import Template

from config import settings
from src.logger import get_logger

logger = get_logger(__name__)

HTML_TEMPLATE = Template(
    """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Crypto Market Report - {{ period }}</title>
<style>
  body { font-family: Arial, sans-serif; margin: 40px; color: #1a1a2e; }
  h1 { color: #0f3460; }
  .kpi-grid { display: flex; gap: 20px; margin: 20px 0; flex-wrap: wrap; }
  .kpi-card { background: #f4f6fb; border-radius: 10px; padding: 16px 24px; min-width: 160px; }
  .kpi-card h3 { margin: 0 0 8px 0; font-size: 13px; color: #555; text-transform: uppercase; }
  .kpi-card p { margin: 0; font-size: 22px; font-weight: bold; color: #0f3460; }
  table { border-collapse: collapse; width: 100%; margin-top: 10px; }
  th, td { border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 13px; }
  th { background-color: #0f3460; color: white; }
  .insights { background: #eef7ee; border-left: 4px solid #2ecc71; padding: 12px 20px; margin-top: 20px;}
</style>
</head>
<body>
  <h1>Cryptocurrency Market Report</h1>
  <p><strong>Period:</strong> {{ period }} &nbsp;|&nbsp; <strong>Generated:</strong> {{ generated_at }}</p>

  <div class="kpi-grid">
    {% for kpi in kpis %}
    <div class="kpi-card"><h3>{{ kpi.label }}</h3><p>{{ kpi.value }}</p></div>
    {% endfor %}
  </div>

  <h2>Top Gainers (24h)</h2>
  {{ gainers_table }}

  <h2>Top Losers (24h)</h2>
  {{ losers_table }}

  <div class="insights">
    <h2>Automated Insights</h2>
    <ul>
    {% for insight in insights %}
      <li>{{ insight }}</li>
    {% endfor %}
    </ul>
  </div>
</body>
</html>
"""
)


def compute_kpis(latest_df: pd.DataFrame) -> list[dict[str, str]]:
    btc_row = latest_df[latest_df["symbol"] == "BTC"]
    btc_price = f"${btc_row['price'].iloc[0]:,.2f}" if not btc_row.empty else "N/A"

    total_mcap = latest_df["market_cap"].sum()
    total_vol = latest_df["volume_24h"].sum()
    top_gainer = latest_df.loc[latest_df["percent_change_24h"].idxmax()]
    top_loser = latest_df.loc[latest_df["percent_change_24h"].idxmin()]
    most_volatile = (
        latest_df.loc[latest_df["volatility"].idxmax()] if "volatility" in latest_df.columns and latest_df["volatility"].notna().any() else None
    )
    largest_cap = latest_df.loc[latest_df["market_cap"].idxmax()]

    kpis = [
        {"label": "BTC Price", "value": btc_price},
        {"label": "Total Market Cap", "value": f"${total_mcap:,.0f}"},
        {"label": "24H Volume", "value": f"${total_vol:,.0f}"},
        {"label": "Number of Coins", "value": str(len(latest_df))},
        {"label": "Top Gainer", "value": f"{top_gainer['symbol']} ({top_gainer['percent_change_24h']:.2f}%)"},
        {"label": "Top Loser", "value": f"{top_loser['symbol']} ({top_loser['percent_change_24h']:.2f}%)"},
        {"label": "Largest Market Cap", "value": f"{largest_cap['symbol']}"},
    ]
    if most_volatile is not None:
        kpis.append({"label": "Most Volatile", "value": f"{most_volatile['symbol']}"})
    return kpis


def generate_insights(latest_df: pd.DataFrame) -> list[str]:
    insights = []
    avg_change = latest_df["percent_change_24h"].mean()
    direction = "up" if avg_change > 0 else "down"
    insights.append(f"The market is broadly {direction} on average ({avg_change:.2f}% over 24h across tracked coins).")

    top3 = latest_df.nlargest(3, "market_dominance") if "market_dominance" in latest_df.columns else latest_df.nlargest(3, "market_cap")
    combined_dominance = top3["market_dominance"].sum() * 100 if "market_dominance" in top3.columns else None
    if combined_dominance:
        insights.append(f"The top 3 coins by market cap ({', '.join(top3['symbol'])}) control {combined_dominance:.1f}% of total tracked market cap.")

    if "is_any_anomaly" in latest_df.columns:
        n_anom = int(latest_df["is_any_anomaly"].sum())
        if n_anom:
            insights.append(f"{n_anom} coin(s) triggered anomaly flags in the latest snapshot (flash crash, spike, or statistical outlier).")

    high_vol_ratio = latest_df["volume_to_marketcap_ratio"].mean() if "volume_to_marketcap_ratio" in latest_df.columns else None
    if high_vol_ratio:
        insights.append(f"Average volume/market-cap ratio is {high_vol_ratio:.2f}, indicating overall market liquidity.")

    return insights


def render_html_report(latest_df: pd.DataFrame, period: str = "Daily") -> str:
    kpis = compute_kpis(latest_df)
    insights = generate_insights(latest_df)

    gainers = latest_df.nlargest(10, "percent_change_24h")[["symbol", "name", "price", "percent_change_24h"]]
    losers = latest_df.nsmallest(10, "percent_change_24h")[["symbol", "name", "price", "percent_change_24h"]]

    html = HTML_TEMPLATE.render(
        period=period,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        kpis=kpis,
        gainers_table=gainers.to_html(index=False, float_format=lambda x: f"{x:,.2f}"),
        losers_table=losers.to_html(index=False, float_format=lambda x: f"{x:,.2f}"),
        insights=insights,
    )
    return html


def save_html_report(latest_df: pd.DataFrame, period: str = "Daily") -> Path:
    html = render_html_report(latest_df, period)
    filename = f"crypto_report_{period.lower()}_{datetime.now(timezone.utc):%Y%m%d_%H%M}.html"
    path = settings.REPORTS_DIR / filename
    path.write_text(html, encoding="utf-8")
    logger.info("HTML report saved -> %s", path)
    return path


def save_pdf_report(latest_df: pd.DataFrame, period: str = "Daily") -> Path | None:
    """Renders the same HTML template to PDF via WeasyPrint. Falls back
    gracefully (returns None + logs a warning) if system deps for
    WeasyPrint aren't installed - HTML/Excel/CSV still work regardless."""
    html = render_html_report(latest_df, period)
    filename = f"crypto_report_{period.lower()}_{datetime.now(timezone.utc):%Y%m%d_%H%M}.pdf"
    path = settings.REPORTS_DIR / filename
    try:
        from weasyprint import HTML

        HTML(string=html).write_pdf(str(path))
        logger.info("PDF report saved -> %s", path)
        return path
    except Exception as exc:  # broad on purpose - PDF is a nice-to-have, not critical path
        logger.warning("PDF generation skipped (%s). Install weasyprint + system deps to enable.", exc)
        return None


def _make_excel_safe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            if df[col].dt.tz is not None:
                df[col] = df[col].dt.tz_convert(None)
    return df


def save_excel_report(latest_df: pd.DataFrame, history_df: pd.DataFrame | None = None) -> Path:
    filename = f"crypto_report_{datetime.now(timezone.utc):%Y%m%d_%H%M}.xlsx"
    path = settings.REPORTS_DIR / filename

    latest_safe = _make_excel_safe(latest_df)
    history_safe = _make_excel_safe(history_df) if history_df is not None else None

    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        latest_safe.to_excel(writer, sheet_name="Latest Snapshot", index=False)
        latest_safe.nlargest(20, "percent_change_24h").to_excel(writer, sheet_name="Top Gainers", index=False)
        latest_safe.nsmallest(20, "percent_change_24h").to_excel(writer, sheet_name="Top Losers", index=False)
        latest_safe.nlargest(20, "market_cap").to_excel(writer, sheet_name="Top Market Cap", index=False)
        if history_safe is not None:
            history_safe.to_excel(writer, sheet_name="Full History", index=False)

    logger.info("Excel report saved -> %s", path)
    return path


def save_csv_report(latest_df: pd.DataFrame) -> Path:
    filename = f"crypto_report_{datetime.now(timezone.utc):%Y%m%d_%H%M}.csv"
    path = settings.REPORTS_DIR / filename
    latest_df.to_csv(path, index=False)
    logger.info("CSV report saved -> %s", path)
    return path


def generate_all_reports(latest_df: pd.DataFrame, history_df: pd.DataFrame | None = None, period: str = "Daily") -> dict[str, Path | None]:
    return {
        "html": save_html_report(latest_df, period),
        "pdf": save_pdf_report(latest_df, period),
        "excel": save_excel_report(latest_df, history_df),
        "csv": save_csv_report(latest_df),
    }

# 🪙 Crypto Market Analytics Platform

An end-to-end, production-style data analytics pipeline for cryptocurrency
markets: automated extraction, storage, cleaning, feature engineering,
EDA, forecasting, clustering, anomaly detection, an interactive Streamlit
dashboard, SQL analytics, and automated multi-format reporting.

Built to demonstrate the full skill set expected of a data analyst /
analytics engineer: ETL design, data quality engineering, statistics,
time-series modeling, unsupervised ML, dashboarding, and stakeholder
reporting — not just "call an API and make a chart."

---

## 1. Architecture

```
                     ┌─────────────────────┐
                     │  CoinMarketCap       │
                     │  Sandbox API         │  (CoinGecko = free fallback)
                     └──────────┬───────────┘
                                │ requests + retry/backoff
                                ▼
                     ┌─────────────────────┐
                     │   src/api.py         │  schema validation
                     └──────────┬───────────┘
                                ▼
                     ┌─────────────────────┐
                     │  src/database.py     │  SQLAlchemy → SQLite
                     │  (+ CSV backup)       │  de-dup on (coin_id, ts)
                     └──────────┬───────────┘
                                ▼
                     ┌─────────────────────┐
                     │  src/cleaning.py      │  missing values, dupes,
                     │                       │  dtypes, outliers, rules
                     └──────────┬───────────┘
                                ▼
                     ┌─────────────────────┐
                     │ feature_engineering  │  returns, rolling stats,
                     │       .py            │  momentum, dominance
                     └──────────┬───────────┘
                                ▼
              ┌─────────────────┴─────────────────┐
              ▼                                   ▼
   ┌─────────────────────┐            ┌─────────────────────┐
   │ anomaly_detection.py │            │  forecasting.py /     │
   │ clustering.py         │            │  clustering.py        │
   └──────────┬───────────┘            └──────────┬───────────┘
              └─────────────────┬─────────────────┘
                                 ▼
                     ┌─────────────────────┐
                     │  reporting.py         │  HTML / PDF / Excel / CSV
                     └──────────┬───────────┘
                                ▼
                     ┌─────────────────────┐
                     │ dashboard/app.py      │  Streamlit UI
                     │ (Plotly charts)       │
                     └─────────────────────┘

   src/scheduler.py (APScheduler) drives the fetch → ... → report cycle
   every FETCH_INTERVAL_MINUTES (default 30) as a standalone process.
```

**Data flow contract:** every stage takes a DataFrame in, returns a
DataFrame out, and logs an audit trail (`src/logger.py` → `logs/pipeline.log`).
This is what makes the pipeline debuggable at 3 AM when a scheduled run
fails on a bad API payload.

---

## 2. Project Structure

```
Crypto-Analytics/
├── data/
│   ├── raw/            # raw CSV backups mirroring the SQL table
│   ├── processed/       # cleaned + feature-engineered snapshots
│   └── archive/          # (reserved for periodic full-history archives)
├── notebooks/            # exploratory analysis (add your own .ipynb here)
├── src/
│   ├── api.py                  # extraction (CMC sandbox + CoinGecko fallback)
│   ├── database.py             # SQLAlchemy models + storage layer
│   ├── cleaning.py             # missing values, dupes, dtypes, outliers, rules
│   ├── feature_engineering.py   # timestamps, returns, rolling stats, dominance
│   ├── anomaly_detection.py     # flash crashes, spikes, IsolationForest, LOF
│   ├── clustering.py            # KMeans / Hierarchical / DBSCAN
│   ├── forecasting.py           # Linear Regression / ARIMA / SARIMA / Prophet stub
│   ├── visualization.py         # shared Plotly + Matplotlib chart builders
│   ├── reporting.py             # HTML / PDF / Excel / CSV report generation
│   ├── pipeline.py              # orchestrates one full fetch→report cycle
│   ├── scheduler.py             # APScheduler - runs pipeline every 30 min
│   └── logger.py                # shared logging config
├── dashboard/
│   └── app.py                   # Streamlit dashboard (imported by root app.py)
├── reports/                       # auto-generated HTML/PDF/Excel/CSV reports
├── config/
│   ├── settings.py               # all configuration (env-var driven)
│   └── .env.example
├── sql/
│   └── analysis_queries.sql       # 13 analytical queries + views + indexes
├── requirements.txt
├── app.py                         # root entry point (dashboard or CLI pipeline run)
└── README.md
```

---

## 3. Setup Guide

```bash
# 1. Clone and enter the project
git clone <your-fork-url> Crypto-Analytics
cd Crypto-Analytics

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp config/.env.example .env
# (A public CMC sandbox key already ships as a default in settings.py,
#  so the project works out of the box - swap in your own key to be safe.)

# 5. Run one full pipeline cycle to populate the database
python app.py --once --with-reports

# 6. Launch the dashboard
streamlit run app.py

# 7. (Optional) start the always-on 30-minute scheduler in a separate process
python src/scheduler.py
```

### Data source note
This project targets the **CoinMarketCap sandbox API**
(`sandbox-api.coinmarketcap.com`), which returns realistic mock market
data without consuming a production API quota — ideal for a portfolio
project. `src/api.py` automatically falls back to the **free CoinGecko
API** (no key required) if the sandbox is unreachable, so the pipeline
never has a hard dependency on a paid key.

---

## 4. Data Dictionary

| Column | Type | Description |
|---|---|---|
| `coin_id` | int | Provider's unique coin identifier |
| `name` | str | Full coin name (e.g. "Bitcoin") |
| `symbol` | str | Ticker symbol, uppercased (e.g. "BTC") |
| `cmc_rank` | int | Market cap rank at snapshot time |
| `price` | float | Spot price in USD |
| `market_cap` | float | Market capitalization in USD |
| `volume_24h` | float | Trailing 24h trading volume in USD |
| `circulating_supply` | float | Circulating token supply |
| `percent_change_1h/24h/7d` | float | Percent price change over each window |
| `last_updated` | datetime | Provider's last-updated timestamp |
| `snapshot_time` | datetime | Timestamp our pipeline captured this row (UTC) |
| `is_outlier` | bool | Flagged by combined Z-score/IQR/IsolationForest cleaning check |

## 5. Feature Dictionary (subset — see `src/feature_engineering.py` for full list)

| Feature | Meaning |
|---|---|
| `hourly_return` / `daily_return` / `weekly_return` | % price change over each window |
| `rolling_mean_{3,6,12,24}` / `rolling_median_*` / `rolling_std_*` | Rolling stats over N snapshots |
| `ema_12` | 12-period exponential moving average |
| `price_momentum` / `price_acceleration` | 1st/2nd derivative of price |
| `volatility` / `rolling_volatility_24` | Rolling std of returns |
| `sharpe_ratio_simplified` | Rolling mean(return)/std(return) |
| `momentum_score` | Weighted blend of short + long return |
| `market_dominance` / `market_cap_share` | Coin's market cap ÷ total tracked market cap |
| `volume_share` | Coin's volume ÷ total tracked volume |
| `volume_to_marketcap_ratio` | Liquidity proxy |
| `relative_strength_vs_btc` | 24h return minus BTC's 24h return |
| `price_rank` | Cross-sectional rank by price at that snapshot |

---

## 6. Business Questions Answered

- Which cryptocurrencies gained/lost the most, and how consistently?
- Which coins are most volatile, and does that change by hour/weekday?
- What's the relationship between market cap, volume, and price movement?
- Which coins consistently outperform Bitcoin? (see SQL query #13)
- Which assets show abnormal trading activity (flash crashes, volume spikes)?
- How concentrated is the market (dominance of top 3 vs. the rest)?

---

## 7. Known Scope Limitations (and how to extend)

This repo is a strong, fully-functional **core** of the platform described
in the brief. A few of the most resource-heavy/optional items were
intentionally left as documented extension points rather than built out,
since they depend on external paid tools or heavy system installs:

| Item | Status | To enable |
|---|---|---|
| Facebook Prophet forecasts | Stub in `forecasting.py` | `pip install prophet` (needs `cmdstanpy`) |
| XGBoost / LightGBM / CatBoost classifiers | Not included | Add to `requirements.txt`, extend `forecasting.py` with a classification target (e.g., next-hour up/down) |
| Power BI (.pbix) dashboard | Not included | Point Power BI's SQLite/ODBC connector at `data/crypto_market.db`, or import `reports/*.xlsx` |
| PostgreSQL | Supported by config, not provisioned | Set `DB_ENGINE=postgresql` and `POSTGRES_URI` in `.env` |
| Cloud deployment (Streamlit Cloud/Render/Railway/HF Spaces) | Not deployed | See deployment guide below |
| Geographical exchange-location charts | Not included | CMC/CoinGecko market-listing endpoints (not the `listings/latest` endpoint used here) would be needed |

### Deployment guide (Streamlit Cloud, the fastest path)
1. Push this repo to GitHub (make sure `.env` is in `.gitignore`, not committed).
2. Go to share.streamlit.io → "New app" → select the repo, branch, and `app.py` as the entry point.
3. Under "Advanced settings → Secrets", paste the contents of your `.env` file.
4. Deploy. Note: Streamlit Cloud's free tier is stateless storage — for durable historical data, use a hosted Postgres (e.g. Supabase/Neon free tier) instead of local SQLite in production.
5. Run `src/scheduler.py` separately (e.g. as a Railway/Render background worker, or a GitHub Actions cron job hitting `python app.py --once`) since Streamlit Cloud doesn't run long-lived background schedulers itself.

---

## 8. Resume Bullet Points

- Designed and built an end-to-end cryptocurrency analytics pipeline in Python, automating extraction, storage, cleaning, and feature engineering for 100 assets refreshed every 30 minutes.
- Implemented a fault-tolerant ETL layer with exponential-backoff retries, a secondary API fallback, and schema validation, achieving zero-downtime data collection across provider outages.
- Engineered 30+ time-series and cross-sectional features (returns, rolling volatility, momentum, market dominance) to support forecasting and anomaly detection.
- Built a multi-model anomaly detection system (Z-score, IQR, Isolation Forest, Local Outlier Factor) to flag flash crashes, volume spikes, and market-cap anomalies in near real time.
- Developed ARIMA/SARIMA and regression-based forecasting models for price and market-cap prediction, with backtested RMSE/MAE/MAPE/R² evaluation.
- Designed a 13-query analytical SQL library (CTEs, window functions, views, indexes) answering ranking, volatility, and consistency-of-growth business questions.
- Shipped an interactive Streamlit + Plotly dashboard with KPI cards, drill-downs, comparison mode, and automated HTML/PDF/Excel/CSV report generation.

## 9. Interview Questions Based on This Project

1. Walk me through what happens between an API call failing and the pipeline still producing a usable dashboard that day.
2. Why did you choose Isolation Forest *and* Z-score/IQR for outlier detection instead of just one method?
3. How do you decide whether ARIMA or SARIMA is the right model for a given coin's price series?
4. Your rolling features are computed per-coin, ordered by time. What could go wrong if the underlying data has gaps (e.g., a missed 30-minute fetch), and how would you detect/handle that?
5. How would you re-architect the storage layer if you needed second-by-second data instead of 30-minute snapshots?
6. What's the risk of using global thresholds (e.g., a flat volatility Z-score cutoff) across coins of very different market caps, and how would you fix it?
7. How would you validate that your "consistently outperforms Bitcoin" SQL query isn't just capturing survivorship bias?

## 10. STAR-format Explanation (Interview Answer Template)

**Situation:** Existing tutorials on crypto data pipelines stopped at basic
API pulls and a couple of charts — not representative of what a data
analyst actually ships.

**Task:** Build a production-style analytics platform covering the full
lifecycle: reliable ingestion, rigorous cleaning, feature engineering,
statistical/ML analysis, and stakeholder-ready reporting.

**Action:** I designed a modular pipeline (extraction → storage → cleaning
→ feature engineering → anomaly detection/forecasting → reporting) with
retry logic, schema validation, an audited cleaning pipeline (missing
values, duplicates, outliers via three different methods), 30+ engineered
features, ARIMA/regression forecasting with backtested metrics, and a
Streamlit dashboard with KPI cards and drill-down analytics — plus a
13-query SQL library and automated multi-format reporting.

**Result:** A fully automated, self-healing analytics platform that
refreshes every 30 minutes, flags market anomalies without manual
review, and produces daily reports — demonstrating end-to-end ownership
from raw API to executive-ready insight.

---

## 11. Tech Stack

Python · Pandas · NumPy · Matplotlib · Seaborn · Plotly · Scikit-learn ·
SciPy · Statsmodels · Requests · APScheduler · SQLAlchemy · SQLite
(Postgres-ready) · Streamlit · Jinja2 · XlsxWriter · WeasyPrint (optional
PDF) · Git/GitHub

-- =============================================================================
-- sql/analysis_queries.sql
-- Analytical SQL library for the crypto_snapshots table.
-- Compatible with SQLite (default). Swap window-function syntax as needed
-- if migrating to PostgreSQL (nearly identical - CTEs/window funcs both
-- support standard ANSI SQL here).
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1. Top 10 gainers in the most recent snapshot
-- -----------------------------------------------------------------------------
SELECT symbol, name, price, percent_change_24h
FROM crypto_snapshots
WHERE snapshot_time = (SELECT MAX(snapshot_time) FROM crypto_snapshots)
ORDER BY percent_change_24h DESC
LIMIT 10;

-- -----------------------------------------------------------------------------
-- 2. Top 10 losers in the most recent snapshot
-- -----------------------------------------------------------------------------
SELECT symbol, name, price, percent_change_24h
FROM crypto_snapshots
WHERE snapshot_time = (SELECT MAX(snapshot_time) FROM crypto_snapshots)
ORDER BY percent_change_24h ASC
LIMIT 10;

-- -----------------------------------------------------------------------------
-- 3. Highest average 24h volume per coin (all-time, in this DB)
-- -----------------------------------------------------------------------------
SELECT symbol, name, AVG(volume_24h) AS avg_volume_24h
FROM crypto_snapshots
GROUP BY symbol, name
ORDER BY avg_volume_24h DESC
LIMIT 20;

-- -----------------------------------------------------------------------------
-- 4. Largest volatility (std dev of price) per coin
--    SQLite has no native STDDEV, so we compute it manually via variance.
-- -----------------------------------------------------------------------------
WITH stats AS (
    SELECT
        symbol,
        AVG(price) AS mean_price,
        COUNT(*) AS n
    FROM crypto_snapshots
    GROUP BY symbol
),
variance_calc AS (
    SELECT
        s.symbol,
        s.n,
        SUM((c.price - s.mean_price) * (c.price - s.mean_price)) / NULLIF(s.n - 1, 0) AS variance
    FROM crypto_snapshots c
    JOIN stats s ON c.symbol = s.symbol
    GROUP BY s.symbol
)
SELECT symbol, SQRT(variance) AS price_volatility
FROM variance_calc
WHERE n > 5
ORDER BY price_volatility DESC
LIMIT 20;

-- -----------------------------------------------------------------------------
-- 5. Coins growing consistently (positive return in every recorded snapshot
--    over the last 20 snapshots for that coin)
-- -----------------------------------------------------------------------------
WITH ranked AS (
    SELECT
        symbol,
        snapshot_time,
        price,
        LAG(price) OVER (PARTITION BY symbol ORDER BY snapshot_time) AS prev_price,
        ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY snapshot_time DESC) AS rn
    FROM crypto_snapshots
),
recent AS (
    SELECT *, (price - prev_price) AS delta
    FROM ranked
    WHERE rn <= 20 AND prev_price IS NOT NULL
)
SELECT symbol, COUNT(*) AS snapshots_checked, SUM(CASE WHEN delta > 0 THEN 1 ELSE 0 END) AS positive_moves
FROM recent
GROUP BY symbol
HAVING positive_moves = snapshots_checked
ORDER BY snapshots_checked DESC;

-- -----------------------------------------------------------------------------
-- 6. Average daily return per coin
-- -----------------------------------------------------------------------------
SELECT symbol, AVG(percent_change_24h) AS avg_daily_return
FROM crypto_snapshots
GROUP BY symbol
ORDER BY avg_daily_return DESC
LIMIT 20;

-- -----------------------------------------------------------------------------
-- 7. Monthly trend: average price and market cap per coin per month
-- -----------------------------------------------------------------------------
SELECT
    symbol,
    strftime('%Y-%m', snapshot_time) AS year_month,
    AVG(price) AS avg_price,
    AVG(market_cap) AS avg_market_cap
FROM crypto_snapshots
GROUP BY symbol, year_month
ORDER BY symbol, year_month;

-- -----------------------------------------------------------------------------
-- 8. Ranking changes: compare cmc_rank between the two most recent snapshots
-- -----------------------------------------------------------------------------
WITH last_two AS (
    SELECT DISTINCT snapshot_time
    FROM crypto_snapshots
    ORDER BY snapshot_time DESC
    LIMIT 2
),
tagged AS (
    SELECT c.*, RANK() OVER (ORDER BY c.snapshot_time DESC) AS snap_rank
    FROM crypto_snapshots c
    JOIN last_two lt ON c.snapshot_time = lt.snapshot_time
)
SELECT
    curr.symbol,
    prev.cmc_rank AS previous_rank,
    curr.cmc_rank AS current_rank,
    (prev.cmc_rank - curr.cmc_rank) AS rank_change
FROM tagged curr
JOIN tagged prev ON curr.symbol = prev.symbol AND prev.snap_rank = 2
WHERE curr.snap_rank = 1
ORDER BY rank_change DESC;

-- -----------------------------------------------------------------------------
-- 9. Market cap trend with rolling 5-snapshot average (window function)
-- -----------------------------------------------------------------------------
SELECT
    symbol,
    snapshot_time,
    market_cap,
    AVG(market_cap) OVER (
        PARTITION BY symbol ORDER BY snapshot_time
        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
    ) AS rolling_avg_market_cap_5
FROM crypto_snapshots
ORDER BY symbol, snapshot_time;

-- -----------------------------------------------------------------------------
-- 10. Reusable VIEW: latest snapshot only (used heavily by the dashboard)
-- -----------------------------------------------------------------------------
DROP VIEW IF EXISTS v_latest_snapshot;
CREATE VIEW v_latest_snapshot AS
SELECT *
FROM crypto_snapshots
WHERE snapshot_time = (SELECT MAX(snapshot_time) FROM crypto_snapshots);

-- -----------------------------------------------------------------------------
-- 11. Reusable VIEW: daily OHLC-style summary per coin
-- -----------------------------------------------------------------------------
DROP VIEW IF EXISTS v_daily_summary;
CREATE VIEW v_daily_summary AS
SELECT
    symbol,
    DATE(snapshot_time) AS trade_date,
    MIN(price) AS low,
    MAX(price) AS high,
    AVG(price) AS avg_price,
    AVG(volume_24h) AS avg_volume,
    AVG(market_cap) AS avg_market_cap
FROM crypto_snapshots
GROUP BY symbol, trade_date;

-- -----------------------------------------------------------------------------
-- 12. Indexes for query performance (idempotent - SQLite ignores if exists)
-- -----------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_symbol_time ON crypto_snapshots (symbol, snapshot_time);
CREATE INDEX IF NOT EXISTS idx_snapshot_time ON crypto_snapshots (snapshot_time);
CREATE INDEX IF NOT EXISTS idx_coin_id ON crypto_snapshots (coin_id);

-- -----------------------------------------------------------------------------
-- 13. Which cryptocurrencies consistently outperform Bitcoin?
-- -----------------------------------------------------------------------------
WITH btc AS (
    SELECT snapshot_time, percent_change_24h AS btc_change
    FROM crypto_snapshots
    WHERE symbol = 'BTC'
)
SELECT
    c.symbol,
    COUNT(*) AS snapshots_outperformed,
    AVG(c.percent_change_24h - b.btc_change) AS avg_outperformance
FROM crypto_snapshots c
JOIN btc b ON c.snapshot_time = b.snapshot_time
WHERE c.symbol != 'BTC' AND c.percent_change_24h > b.btc_change
GROUP BY c.symbol
ORDER BY avg_outperformance DESC
LIMIT 20;

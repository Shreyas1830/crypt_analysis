"""
src/forecasting.py
-------------------
Time-series forecasting for price, market cap, and volume. Implements
Linear Regression (trend baseline), ARIMA and SARIMA (statsmodels).
Facebook Prophet is intentionally left optional (heavy system
dependency on cmdstan) - see `forecast_with_prophet` for a drop-in stub.

Also includes decomposition, ACF/PACF, and stationarity testing, since
these should inform which model is trustworthy for a given coin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, adfuller, pacf

from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ForecastResult:
    model_name: str
    forecast: pd.Series
    conf_int: pd.DataFrame | None
    metrics: dict[str, float]


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def test_stationarity(series: pd.Series) -> dict[str, float]:
    """Augmented Dickey-Fuller test. p-value < 0.05 => series is stationary."""
    clean = series.dropna()
    stat, p_value, *_ = adfuller(clean)
    return {"adf_statistic": float(stat), "p_value": float(p_value), "is_stationary": p_value < 0.05}


def decompose_series(series: pd.Series, period: int = 48) -> dict[str, pd.Series]:
    """Classical decomposition into trend / seasonal / residual. `period`
    defaults to 48 snapshots (~1 day at 30-minute cadence)."""
    clean = series.dropna()
    if len(clean) < period * 2:
        logger.warning("Not enough data points for decomposition (need >= %d).", period * 2)
        return {}
    result = seasonal_decompose(clean, model="additive", period=period, extrapolate_trend="freq")
    return {"trend": result.trend, "seasonal": result.seasonal, "residual": result.resid}


def autocorrelation(series: pd.Series, n_lags: int = 40) -> dict[str, np.ndarray]:
    clean = series.dropna()
    return {
        "acf": acf(clean, nlags=n_lags, fft=True),
        "pacf": pacf(clean, nlags=n_lags),
    }


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def forecast_linear_regression(series: pd.Series, steps_ahead: int = 12) -> ForecastResult:
    clean = series.dropna().reset_index(drop=True)
    X = np.arange(len(clean)).reshape(-1, 1)
    y = clean.values

    split = int(len(clean) * 0.8)
    model = LinearRegression().fit(X[:split], y[:split])

    y_pred_test = model.predict(X[split:])
    metrics = _regression_metrics(y[split:], y_pred_test) if len(y[split:]) else {}

    model_full = LinearRegression().fit(X, y)
    future_X = np.arange(len(clean), len(clean) + steps_ahead).reshape(-1, 1)
    forecast = pd.Series(model_full.predict(future_X), name="linear_regression_forecast")

    return ForecastResult("LinearRegression", forecast, None, metrics)


def forecast_arima(series: pd.Series, order: tuple[int, int, int] = (2, 1, 2), steps_ahead: int = 12) -> ForecastResult:
    clean = series.dropna()
    split = int(len(clean) * 0.8)
    train, test = clean[:split], clean[split:]

    model = ARIMA(train, order=order).fit()
    metrics = {}
    if len(test):
        test_pred = model.forecast(steps=len(test))
        metrics = _regression_metrics(test.values, test_pred.values)

    full_model = ARIMA(clean, order=order).fit()
    forecast_res = full_model.get_forecast(steps=steps_ahead)
    forecast = forecast_res.predicted_mean.rename("arima_forecast")
    conf_int = forecast_res.conf_int()

    return ForecastResult("ARIMA", forecast, conf_int, metrics)


def forecast_sarima(
    series: pd.Series,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 48),
    steps_ahead: int = 12,
) -> ForecastResult:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    clean = series.dropna()
    split = int(len(clean) * 0.8)
    train, test = clean[:split], clean[split:]

    model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                     enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    metrics = {}
    if len(test):
        test_pred = model.forecast(steps=len(test))
        metrics = _regression_metrics(test.values, test_pred.values)

    full_model = SARIMAX(clean, order=order, seasonal_order=seasonal_order,
                          enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    forecast_res = full_model.get_forecast(steps=steps_ahead)
    forecast = forecast_res.predicted_mean.rename("sarima_forecast")
    conf_int = forecast_res.conf_int()

    return ForecastResult("SARIMA", forecast, conf_int, metrics)


def forecast_with_prophet(series: pd.Series, steps_ahead: int = 12) -> ForecastResult | None:
    """Optional. Requires `pip install prophet` (needs cmdstanpy/cmdstan
    installed on the system). Left as a clean drop-in for users who want
    it - not run by default to keep the base install lightweight."""
    try:
        from prophet import Prophet
    except ImportError:
        logger.warning("Prophet not installed - skipping. Run `pip install prophet` to enable.")
        return None

    df = series.dropna().reset_index()
    df.columns = ["ds", "y"]
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=steps_ahead, freq="30min")
    forecast_df = model.predict(future)
    forecast = forecast_df.set_index("ds")["yhat"].tail(steps_ahead)
    conf_int = forecast_df.set_index("ds")[["yhat_lower", "yhat_upper"]].tail(steps_ahead)
    return ForecastResult("Prophet", forecast, conf_int, {})


# --------------------------------------------------------------------------- #
def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
    }

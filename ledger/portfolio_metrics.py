"""Canonical daily portfolio aggregation for paper/shadow evaluation.

All inputs are decimal net returns.  Multiple signals on the same day are
equal-weighted, missing crypto calendar days are cash (0%), and the equity
curve includes the initial capital before the first return.
"""
from __future__ import annotations

import math
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


CRYPTO_PERIODS_PER_YEAR = 365
KST = ZoneInfo("Asia/Seoul")


def normalize_kst_date(value=None) -> pd.Timestamp:
    """Return a timezone-naive KST calendar date for an as-of value.

    Ledger ``date`` columns are KST trading/calendar dates stored without a
    timezone.  A timezone-aware instant is converted to KST before taking its
    date; a naive value is interpreted as KST.  ``None`` means today's KST
    date.  Returning a naive normalized timestamp keeps comparisons with the
    persisted date-only columns unambiguous.
    """
    if value is None:
        timestamp = pd.Timestamp.now(tz=KST)
    else:
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("asof must be a valid scalar date or timestamp") from exc
    if pd.isna(timestamp):
        raise ValueError("asof must be a valid scalar date or timestamp")
    if not isinstance(timestamp, pd.Timestamp):
        raise ValueError("asof must be a valid scalar date or timestamp")
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(KST)
    else:
        timestamp = timestamp.tz_convert(KST)
    return timestamp.normalize().tz_localize(None)


def _normalize_kst_dates(values: pd.Series) -> pd.Series:
    """Normalize mixed naive/aware timestamps to timezone-naive KST dates."""
    def _one(value):
        if value is None or (
            isinstance(value, float) and math.isnan(value)
        ):
            return pd.NaT
        try:
            return normalize_kst_date(value)
        except (TypeError, ValueError):
            return pd.NaT

    return pd.Series(values, copy=False).map(_one)


def daily_equal_weight(
    dates: pd.Series,
    returns: pd.Series,
    *,
    include_no_trade_days: bool = True,
    calendar_end=None,
) -> pd.Series:
    """Equal-weight signals by day and represent inactive days as cash.

    ``calendar_end`` is the inclusive KST as-of cutoff.  Rows after it are
    excluded before aggregation, and the cash calendar extends through it.
    A timezone-aware instant is converted to KST before its calendar date is
    taken; naive values are interpreted as KST.
    """
    if len(dates) != len(returns):
        raise ValueError("dates and returns must have the same length")
    # Pair rows positionally.  Pandas' DataFrame constructor aligns Series by
    # index label, which can silently discard or cross-pair observations when
    # callers pass filtered/reset Series with different indexes.
    date_values = pd.Series(dates, copy=False).reset_index(drop=True)
    return_values = pd.Series(returns, copy=False).reset_index(drop=True)
    frame = pd.DataFrame({
        "date": _normalize_kst_dates(date_values),
        "return": pd.to_numeric(return_values, errors="coerce"),
    }).dropna()
    frame = frame[np.isfinite(frame["return"])]
    requested_end = None
    if calendar_end is not None:
        requested_end = normalize_kst_date(calendar_end)
        frame = frame[frame["date"] <= requested_end]
    if frame.empty:
        return pd.Series(dtype=float)
    daily = frame.groupby("date", sort=True)["return"].mean()
    if include_no_trade_days:
        end = daily.index.max()
        if requested_end is not None:
            end = requested_end
        calendar = pd.date_range(daily.index.min(), end, freq="D")
        daily = daily.reindex(calendar, fill_value=0.0)
    daily.index.name = "date"
    return daily.astype(float)


def equity_curve(daily_returns: pd.Series, *, initial_equity: float = 1.0) -> pd.Series:
    daily = pd.to_numeric(daily_returns, errors="coerce").dropna().sort_index()
    if daily.empty:
        return pd.Series([float(initial_equity)], index=pd.Index(["initial"]))
    first = daily.index[0]
    initial_index = (
        first - pd.Timedelta(days=1)
        if isinstance(first, (pd.Timestamp, np.datetime64))
        else "initial"
    )
    compounded = float(initial_equity) * (1.0 + daily).cumprod()
    initial = pd.Series([float(initial_equity)], index=[initial_index])
    return pd.concat([initial, compounded])


def max_drawdown_from_equity(equity: pd.Series) -> float:
    values = pd.to_numeric(equity, errors="coerce").dropna()
    if values.empty:
        return 0.0
    peak = values.cummax()
    return float((values / peak - 1.0).min())


def annualized_sharpe(
    daily_returns: pd.Series,
    *,
    periods_per_year: int = CRYPTO_PERIODS_PER_YEAR,
) -> float:
    daily = pd.to_numeric(daily_returns, errors="coerce").dropna()
    if len(daily) < 2:
        return 0.0
    std = float(daily.std(ddof=1))
    if not math.isfinite(std) or std <= 0:
        return 0.0
    return float(daily.mean() / std * math.sqrt(periods_per_year))


def annualized_sortino(
    daily_returns: pd.Series,
    *,
    periods_per_year: int = CRYPTO_PERIODS_PER_YEAR,
) -> float:
    daily = pd.to_numeric(daily_returns, errors="coerce").dropna()
    if daily.empty:
        return 0.0
    downside = np.minimum(daily.to_numpy(dtype=float), 0.0)
    downside_deviation = float(np.sqrt(np.mean(np.square(downside))))
    if not math.isfinite(downside_deviation) or downside_deviation <= 0:
        return 0.0
    return float(
        daily.mean() / downside_deviation * math.sqrt(periods_per_year)
    )


def summarize_daily(daily_returns: pd.Series) -> dict:
    daily = pd.to_numeric(daily_returns, errors="coerce").dropna()
    if daily.empty:
        return {
            "n_calendar_days": 0,
            "cumulative_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ann": 0.0,
            "sortino_ann": 0.0,
        }
    equity = equity_curve(daily)
    cumulative = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    return {
        "n_calendar_days": int(len(daily)),
        "cumulative_return": cumulative,
        "max_drawdown": max_drawdown_from_equity(equity),
        "sharpe_ann": annualized_sharpe(daily),
        "sortino_ann": annualized_sortino(daily),
    }


def date_cluster_bootstrap(
    daily_returns: pd.Series,
    *,
    n_iter: int = 1000,
    seed: int = 42,
) -> dict:
    """Resample whole signal dates after within-day equal weighting."""
    daily = pd.to_numeric(daily_returns, errors="coerce").dropna().to_numpy(dtype=float)
    n = len(daily)
    if n < 5:
        return {"n_days": n, "n_iter": n_iter, "note": "n_days<5: skipped"}
    rng = np.random.default_rng(seed)
    mean_samples = np.empty(n_iter)
    cumulative_samples = np.empty(n_iter)
    for i in range(n_iter):
        sample = daily[rng.integers(0, n, size=n)]
        mean_samples[i] = sample.mean()
        cumulative_samples[i] = np.prod(1.0 + sample) - 1.0
    return {
        "n_days": n,
        "n_iter": n_iter,
        "mean_return_ci95": [
            float(np.quantile(mean_samples, 0.025)),
            float(np.quantile(mean_samples, 0.975)),
        ],
        "cumulative_return_ci95": [
            float(np.quantile(cumulative_samples, 0.025)),
            float(np.quantile(cumulative_samples, 0.975)),
        ],
    }

from __future__ import annotations

import pandas as pd
import pytest

from ledger.portfolio_metrics import (
    annualized_sharpe,
    annualized_sortino,
    daily_equal_weight,
    date_cluster_bootstrap,
    equity_curve,
    max_drawdown_from_equity,
    summarize_daily,
)


def test_same_day_signals_are_equal_weighted_and_cash_days_are_zero():
    dates = pd.Series(["2026-07-01", "2026-07-01", "2026-07-03"])
    returns = pd.Series([0.10, -0.10, 0.03])

    daily = daily_equal_weight(dates, returns)

    assert daily.tolist() == pytest.approx([0.0, 0.0, 0.03])
    assert daily.index.tolist() == list(pd.date_range("2026-07-01", "2026-07-03"))


def test_dates_and_returns_are_paired_positionally_not_by_series_index():
    dates = pd.Series(
        ["2026-07-01", "2026-07-02"],
        index=["date-a", "date-b"],
    )
    returns = pd.Series([0.01, 0.02], index=["return-a", "return-b"])

    daily = daily_equal_weight(dates, returns)

    assert daily.index.tolist() == list(pd.date_range("2026-07-01", "2026-07-02"))
    assert daily.tolist() == pytest.approx([0.01, 0.02])


def test_dates_and_returns_require_matching_lengths():
    with pytest.raises(ValueError, match="same length"):
        daily_equal_weight(
            pd.Series(["2026-07-01"]),
            pd.Series([0.01, 0.02]),
        )


def test_calendar_end_adds_trailing_cash_days_and_truncates_future_observations():
    dates = pd.Series(["2026-07-01", "2026-07-03"])
    returns = pd.Series([0.01, 0.03])

    extended = daily_equal_weight(
        dates,
        returns,
        calendar_end="2026-07-05T18:30:00+09:00",
    )
    truncated = daily_equal_weight(
        dates,
        returns,
        calendar_end="2026-07-02",
    )

    assert extended.index.tolist() == list(
        pd.date_range("2026-07-01", "2026-07-05")
    )
    assert extended.tolist() == pytest.approx([0.01, 0.0, 0.03, 0.0, 0.0])
    assert truncated.index.tolist() == list(
        pd.date_range("2026-07-01", "2026-07-02")
    )
    assert truncated.tolist() == pytest.approx([0.01, 0.0])


def test_calendar_end_is_a_hard_asof_cutoff_for_future_rows():
    dates = pd.Series(["2026-07-24", "2026-07-26"])
    returns = pd.Series([0.01, 9.99])

    daily = daily_equal_weight(
        dates,
        returns,
        calendar_end="2026-07-25",
    )

    assert daily.index.tolist() == list(pd.date_range("2026-07-24", "2026-07-25"))
    assert daily.tolist() == pytest.approx([0.01, 0.0])


def test_timezone_aware_dates_are_cut_on_kst_calendar_boundary():
    dates = pd.Series([
        "2026-07-25T14:59:59+00:00",  # 2026-07-25 23:59:59 KST
        "2026-07-25T15:00:00+00:00",  # 2026-07-26 00:00:00 KST
    ])
    returns = pd.Series([0.01, 9.99])

    daily = daily_equal_weight(
        dates,
        returns,
        calendar_end="2026-07-25",
    )

    assert daily.index.tolist() == [pd.Timestamp("2026-07-25")]
    assert daily.tolist() == pytest.approx([0.01])


def test_nonfinite_returns_are_excluded_instead_of_poisoning_metrics():
    dates = pd.Series(["2026-07-23", "2026-07-24", "2026-07-25"])
    returns = pd.Series([0.01, float("inf"), float("-inf")])

    daily = daily_equal_weight(
        dates,
        returns,
        calendar_end="2026-07-25",
    )

    assert daily.tolist() == pytest.approx([0.01, 0.0, 0.0])


def test_first_day_loss_is_a_drawdown_from_initial_equity():
    daily = pd.Series([-0.10, 0.05], index=pd.date_range("2026-07-01", periods=2))

    equity = equity_curve(daily, initial_equity=100.0)

    assert equity.tolist() == pytest.approx([100.0, 90.0, 94.5])
    assert max_drawdown_from_equity(equity) == pytest.approx(-0.10)
    assert summarize_daily(daily)["max_drawdown"] == pytest.approx(-0.10)


def test_sharpe_uses_crypto_365_day_annualization():
    daily = pd.Series([0.01, -0.005, 0.02, 0.0])
    expected = daily.mean() / daily.std(ddof=1) * (365 ** 0.5)

    assert annualized_sharpe(daily) == pytest.approx(expected)


def test_sortino_uses_zero_target_downside_deviation_and_365_days():
    daily = pd.Series([0.01, -0.005, 0.02, 0.0])
    downside_deviation = ((0.005 ** 2) / len(daily)) ** 0.5
    expected = daily.mean() / downside_deviation * (365 ** 0.5)

    assert annualized_sortino(daily) == pytest.approx(expected)


def test_date_cluster_bootstrap_operates_on_days():
    daily = pd.Series(
        [0.01, -0.01, 0.02, -0.005, 0.003],
        index=pd.date_range("2026-07-01", periods=5),
    )

    out = date_cluster_bootstrap(daily, n_iter=50, seed=7)

    assert out["n_days"] == 5
    assert len(out["mean_return_ci95"]) == 2
    assert len(out["cumulative_return_ci95"]) == 2

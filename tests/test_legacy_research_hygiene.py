from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from scripts import binance_leadlag_v1 as binance
from scripts import cc_filtered_multiday_v1 as filtered_multiday
from scripts import cc_sustained_label_v1 as sustained
from scripts import ch_sustainability_v1 as sustainability
from scripts import downside_head_riskreward_v1 as downside
from scripts import recall_universe_recommender_v1 as recall
from scripts import univariate_precursor_lift_v1 as univariate
from scripts.recommender_downside_exit_v1 import downside_metrics, execution_start


def _daily_candles(periods: int = 40) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2026-06-17 09:00:00",
        periods=periods,
        freq="D",
    )
    close = np.linspace(100.0, 140.0, periods)
    return pd.DataFrame(
        {
            "market": "KRW-TEST",
            "timestamp": timestamps,
            "open": close - 1.0,
            "high": close + 5.0,
            "low": close - 5.0,
            "close": close,
            "volume": np.linspace(1_000.0, 2_000.0, periods),
            "quote_volume": np.linspace(100_000.0, 200_000.0, periods),
        }
    )


def test_upbit_completion_cutoff_changes_exactly_at_0900_kst() -> None:
    timestamps = pd.Series(
        pd.to_datetime(
            [
                "2026-07-24 09:00:00",
                "2026-07-25 09:00:00",
                "2026-07-26 09:00:00",
            ]
        )
    )

    before_open = univariate.completed_upbit_d1_mask(
        timestamps, "2026-07-26 08:59:59+09:00"
    )
    at_open = univariate.completed_upbit_d1_mask(
        timestamps, "2026-07-26 09:00:00+09:00"
    )

    assert before_open.tolist() == [True, False, False]
    assert at_open.tolist() == [True, True, False]


def test_unfinished_upbit_row_keeps_features_but_masks_targets() -> None:
    candles = _daily_candles()
    out = univariate.build_market_features(
        candles,
        asof_kst="2026-07-26 22:00:00+09:00",
    )

    last = out.iloc[-1]
    assert last["timestamp"] == pd.Timestamp("2026-07-26 09:00:00")
    assert pd.notna(last["f_ret_1d"])
    assert last[
        ["lab_pump20", "lab_pump15", "lab_pumpc20", "intraday_high_ret"]
    ].isna().all()
    assert pd.notna(out.iloc[-2]["lab_pump20"])


def test_expanding_purged_folds_have_disjoint_test_dates() -> None:
    dates = np.array(
        [d.date() for d in pd.date_range("2026-01-01", periods=60, freq="D")]
    )
    folds = univariate.expanding_purged_date_folds(
        dates,
        n_folds=5,
        embargo_days=2,
    )

    assert len(folds) == 5
    seen: set[date] = set()
    for train_dates, test_dates in folds:
        assert train_dates.isdisjoint(test_dates)
        assert seen.isdisjoint(test_dates)
        assert min(test_dates) > max(train_dates)
        seen.update(test_dates)

    assert recall.make_folds(dates, 5, 2) == folds


def test_downside_and_forward_labels_require_completed_targets() -> None:
    candles = _daily_candles()
    asof = "2026-07-26 22:00:00+09:00"

    down = downside._add_outcome_labels(candles, asof)
    assert down.iloc[-1][
        ["up_high_ret", "down_low_ret", "eod_ret", "lab_up_10", "lab_dn_05"]
    ].isna().all()
    assert pd.notna(down.iloc[-2]["lab_up_10"])

    labels = sustained._add_all_labels(candles, asof)
    assert labels.iloc[-1][
        ["eod_ret", "lab_sus3", "lab_up_10", "lab_dn_05"]
    ].isna().all()
    assert pd.isna(labels.iloc[-2]["lab_fwd1"])
    assert pd.notna(labels.iloc[-3]["lab_fwd1"])


@pytest.mark.parametrize(
    "module",
    [sustainability, filtered_multiday],
)
def test_dump_labels_preserve_incomplete_targets_as_missing(module) -> None:
    panel = pd.DataFrame(
        {
            "up_high_ret": [0.06, np.nan],
            "eod_ret": [-0.03, np.nan],
        }
    )

    columns = module.add_dump_labels(panel)

    assert panel.loc[0, columns[0]] == 1.0
    assert panel.loc[1, columns].isna().all()


def test_filtered_multiday_rejects_cached_incomplete_outcomes() -> None:
    frame = pd.DataFrame(
        {
            "market": ["KRW-BTC"],
            "up_high_ret": [np.nan],
            "down_low_ret": [np.nan],
            "eod_ret": [np.nan],
            "lab_dump_B": [0.0],
        }
    )

    with pytest.raises(RuntimeError, match="incomplete outcome labels"):
        filtered_multiday._reject_contaminated_oos(frame)


def test_binance_current_utc_bar_is_not_completed() -> None:
    timestamps = pd.Series(
        pd.to_datetime(["2026-07-25 00:00:00", "2026-07-26 00:00:00"])
    )
    mask = binance.completed_binance_d1_mask(
        timestamps,
        "2026-07-26 13:00:00+00:00",
    )
    assert mask.tolist() == [True, False]


def test_binance_daily_bracket_uses_low_and_charges_cost() -> None:
    rows = pd.DataFrame(
        {
            "entry_open_D": [100.0, 100.0],
            "high_D": [106.0, 106.0],
            "low_D": [96.0, 98.0],
            "close_D": [104.0, 104.0],
        }
    )

    both_touched = binance.net_sim(
        rows,
        pd.Series([True, False]),
        cost=0.0015,
    )
    tp_only = binance.net_sim(
        rows,
        pd.Series([False, True]),
        cost=0.0015,
    )

    assert both_touched["gross_tpsl_mean"] == pytest.approx(-0.03)
    assert both_touched["net_tpsl_mean"] == pytest.approx(-0.0315)
    assert tp_only["gross_tpsl_mean"] == pytest.approx(0.05)
    assert tp_only["net_tpsl_mean"] == pytest.approx(0.0485)


def test_research_execution_starts_after_0905_signal() -> None:
    assert execution_start("2026-07-25") == pd.Timestamp(
        "2026-07-25 09:15:00"
    )


def test_research_sortino_uses_day_equal_weight_mean() -> None:
    trades = pd.DataFrame(
        {
            "date": [date(2026, 1, 1), date(2026, 1, 1), date(2026, 1, 2)],
            "net": [0.10, 0.10, -0.10],
            "outcome": ["eod", "eod", "eod"],
        }
    )

    metrics = downside_metrics(trades)

    assert metrics["net_mean"] == pytest.approx(1 / 30)
    assert metrics["sortino"] == pytest.approx(0.0)

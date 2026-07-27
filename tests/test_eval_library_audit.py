from __future__ import annotations

import pandas as pd
import pytest

from ledger.metrics import compute_mdd, compute_sharpe
import scripts.eval_library_audit_v1 as audit
from ledger.portfolio_metrics import annualized_sortino
from scripts.eval_library_audit_v1 import (
    PPY,
    RT,
    folds,
    net_metrics,
    oos_selected,
)


def test_walk_forward_test_windows_are_chronological_disjoint_and_embargoed():
    dates = list(pd.date_range("2025-01-01", periods=63, freq="D").date)

    generated = folds(dates, n=5, emb=3)

    assert len(generated) == 5
    seen_test_dates = set()
    for train_dates, test_dates in generated:
        ordered_train = sorted(train_dates)
        ordered_test = sorted(test_dates)

        assert train_dates
        assert test_dates
        assert train_dates.isdisjoint(test_dates)
        assert seen_test_dates.isdisjoint(test_dates)
        assert ordered_train[-1] < ordered_test[0]

        train_end_index = dates.index(ordered_train[-1])
        test_start_index = dates.index(ordered_test[0])
        assert test_start_index - train_end_index - 1 >= 3
        seen_test_dates.update(test_dates)

    for (_, left_test), (_, right_test) in zip(generated, generated[1:]):
        assert max(left_test) < min(right_test)
    assert max(generated[-1][1]) == dates[-1]


def test_build_filters_excluded_signal_markets_and_cold_start_rows_before_xs_rank(
    monkeypatch,
):
    dates = pd.date_range("2025-01-01", periods=71, freq="D")
    candles = pd.DataFrame(
        {
            "timestamp": dates,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 10.0,
            "quote_volume": 1000.0,
        }
    )
    loaded = []
    monkeypatch.setattr(
        audit,
        "list_markets",
        lambda _db: ["KRW-USDT", "KRW-IP", "KRW-BTC"],
    )

    def load_market(_db, market):
        loaded.append(market)
        return candles.copy()

    monkeypatch.setattr(audit, "load_candles", load_market)
    monkeypatch.setattr(
        audit,
        "build_market_features",
        lambda frame: pd.DataFrame(
            {
                "market": frame["market"].to_numpy(),
                "timestamp": frame["timestamp"].to_numpy(),
                audit.LABEL: 0.0,
            }
        ),
    )

    def assert_eligible(frame):
        assert frame["history_prior_bars"].tolist() == [70]
        return frame

    monkeypatch.setattr(audit, "add_cross_sectional", assert_eligible)

    panel = audit.build()

    assert loaded == ["KRW-BTC"]
    assert len(panel) == 1
    assert panel.iloc[0]["market"] == "KRW-BTC"


def test_build_excludes_the_still_open_upbit_daily_session(monkeypatch):
    dates = pd.date_range("2025-01-01 09:00:00", periods=72, freq="D")
    candles = pd.DataFrame(
        {
            "timestamp": dates,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 10.0,
            "quote_volume": 1000.0,
        }
    )
    monkeypatch.setattr(audit, "list_markets", lambda _db: ["KRW-BTC"])
    monkeypatch.setattr(
        audit,
        "load_candles",
        lambda _db, _market: candles.copy(),
    )
    monkeypatch.setattr(
        audit,
        "build_market_features",
        lambda frame: pd.DataFrame(
            {
                "market": frame["market"].to_numpy(),
                "timestamp": frame["timestamp"].to_numpy(),
                audit.LABEL: 0.0,
            }
        ),
    )
    monkeypatch.setattr(audit, "add_cross_sectional", lambda frame: frame)

    panel = audit.build(
        completed_through="2025-03-12",
    )

    assert panel["date"].max().isoformat() == "2025-03-12"
    assert "2025-03-13" not in set(map(str, panel["date"]))


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        ("2026-07-26T08:59:59+09:00", "2026-07-24"),
        ("2026-07-26T09:00:00+09:00", "2026-07-25"),
    ],
)
def test_completed_label_cutoff_tracks_upbit_session_boundary(now, expected):
    assert audit._completed_label_cutoff(now) == pd.Timestamp(expected)


@pytest.mark.parametrize(
    "dates",
    [
        [pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-01")],
        [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01")],
    ],
)
def test_walk_forward_rejects_non_chronological_or_duplicate_dates(dates):
    with pytest.raises(ValueError, match="strictly increasing and unique"):
        folds(dates, n=1, emb=0)


def test_sparse_policy_metrics_include_zero_position_calendar_days():
    rows = pd.DataFrame(
        [
            {
                "date": "2025-01-02",
                "o": 100.0,
                "h": 111.0,
                "l": 99.0,
                "cl": 105.0,
            },
            {
                "date": "2025-01-10",
                "o": 100.0,
                "h": 102.0,
                "l": 96.0,
                "cl": 96.0,
            },
        ]
    )
    rows.attrs["evaluation_start"] = "2025-01-01"
    rows.attrs["evaluation_end"] = "2025-01-11"

    result = net_metrics(rows, "tp_eod")

    assert result is not None
    assert result["n_trades"] == 2
    assert result["n_days"] == 11
    assert result["n_calendar_days"] == 11
    assert result["n_trade_days"] == 2
    assert result["n_zero_position_days"] == 9
    assert result["calendar_start"] == "2025-01-01"
    assert result["calendar_end"] == "2025-01-11"

    trade_returns = pd.Series(
        [0.10 - RT, -0.04 - RT],
        index=pd.to_datetime(["2025-01-02", "2025-01-10"]),
    )
    calendar_returns = trade_returns.reindex(
        pd.date_range("2025-01-01", "2025-01-11", freq="D"),
        fill_value=0.0,
    )
    calendar_equity = (1.0 + calendar_returns).cumprod()

    assert result["net_mean_per_trade"] == pytest.approx(trade_returns.mean())
    assert result["sharpe"] == pytest.approx(
        compute_sharpe(calendar_returns, PPY)
    )
    assert result["trade_day_only_sharpe"] == pytest.approx(
        compute_sharpe(trade_returns, PPY)
    )
    assert abs(result["sharpe"]) < abs(result["trade_day_only_sharpe"])
    assert result["annualized_return"] == pytest.approx(
        (1.0 + result["cum_return"]) ** (PPY / len(calendar_returns)) - 1.0
    )
    assert result["annualized_arithmetic_return"] == pytest.approx(
        calendar_returns.mean() * PPY
    )
    assert result["trade_day_only_annualized_arithmetic_return"] == pytest.approx(
        trade_returns.mean() * PPY
    )
    assert result["cum_return"] == pytest.approx(calendar_equity.iloc[-1] - 1.0)
    assert result["mdd"] == pytest.approx(
        compute_mdd(calendar_equity, initial_equity=1.0)
    )
    assert result["trade_day_only_cum_return"] == pytest.approx(
        result["cum_return"]
    )
    assert result["trade_day_only_mdd"] == pytest.approx(result["mdd"])
    assert result["tp_rate"] == pytest.approx(0.5)
    assert result["sl_rate"] == pytest.approx(0.0)
    assert result["sortino"] == pytest.approx(
        annualized_sortino(calendar_returns)
    )


def test_sparse_oos_fold_is_kept_without_minimum_selection_gate(monkeypatch):
    train_dates = pd.date_range("2025-01-01", periods=4, freq="D").date
    test_dates = pd.date_range("2025-01-10", periods=2, freq="D").date
    rows = []
    for day in train_dates:
        rows.extend(
            {
                "date": day,
                "regime": "bull_quiet",
                "feat": float(index),
                "o": 100.0,
                "h": 101.0,
                "l": 99.0,
                "cl": 100.0,
                audit.LABEL: 0.0,
            }
            for index in range(100)
        )
    for day in test_dates:
        rows.extend(
            {
                "date": day,
                "regime": "bull_quiet",
                "feat": 100.0 if index == 0 and day == test_dates[0] else 0.0,
                "o": 100.0,
                "h": 101.0,
                "l": 99.0,
                "cl": 100.0,
                audit.LABEL: 0.0,
            }
            for index in range(50)
        )
    panel = pd.DataFrame(rows)
    monkeypatch.setattr(
        audit,
        "folds",
        lambda _dates: [(set(train_dates), set(test_dates))],
    )

    selected = oos_selected(
        panel,
        "feat",
        "high",
        "bull_quiet",
    )

    assert selected is not None
    assert len(selected) == 1
    assert selected.attrs["evaluation_start"] == min(test_dates)
    assert selected.attrs["evaluation_end"] == max(test_dates)


def test_sparse_regime_test_fold_is_not_deleted_ex_post(monkeypatch):
    train_dates = pd.date_range("2025-01-01", periods=4, freq="D").date
    test_dates = pd.date_range("2025-01-10", periods=3, freq="D").date
    rows = []
    for day in train_dates:
        rows.extend(
            {
                "date": day,
                "regime": "bear_volatile",
                "feat": float(index),
                "o": 100.0,
                "h": 101.0,
                "l": 99.0,
                "cl": 100.0,
                audit.LABEL: 0.0,
            }
            for index in range(50)
        )
    for day in test_dates:
        rows.extend(
            {
                "date": day,
                "regime": "bear_volatile",
                "feat": -1.0,
                "o": 100.0,
                "h": 101.0,
                "l": 99.0,
                "cl": 100.0,
                audit.LABEL: 0.0,
            }
            for _ in range(15)
        )
    panel = pd.DataFrame(rows)
    monkeypatch.setattr(
        audit,
        "folds",
        lambda _dates: [(set(train_dates), set(test_dates))],
    )

    selected = oos_selected(
        panel,
        "feat",
        "high",
        "bear_volatile",
    )
    result = net_metrics(selected, "sl_first")

    assert selected is not None
    assert selected.empty
    assert selected.attrs["evaluation_start"] == min(test_dates)
    assert selected.attrs["evaluation_end"] == max(test_dates)
    assert result is not None
    assert result["n_calendar_days"] == 3
    assert result["n_zero_position_days"] == 3


def test_zero_selection_oos_horizon_returns_all_cash_metrics(monkeypatch):
    train_dates = pd.date_range("2025-01-01", periods=4, freq="D").date
    test_dates = pd.date_range("2025-01-10", periods=3, freq="D").date
    rows = []
    for day in train_dates:
        rows.extend(
            {
                "date": day,
                "regime": "bear_quiet",
                "feat": float(index),
                "o": 100.0,
                "h": 101.0,
                "l": 99.0,
                "cl": 100.0,
                audit.LABEL: 0.0,
            }
            for index in range(100)
        )
    for day in test_dates:
        rows.extend(
            {
                "date": day,
                "regime": "bear_quiet",
                "feat": -1.0,
                "o": 100.0,
                "h": 101.0,
                "l": 99.0,
                "cl": 100.0,
                audit.LABEL: 0.0,
            }
            for _ in range(40)
        )
    panel = pd.DataFrame(rows)
    monkeypatch.setattr(
        audit,
        "folds",
        lambda _dates: [(set(train_dates), set(test_dates))],
    )

    selected = oos_selected(
        panel,
        "feat",
        "high",
        "bear_quiet",
    )
    result = net_metrics(selected, "sl_first")

    assert selected is not None
    assert selected.empty
    assert result is not None
    assert result["n_trades"] == 0
    assert result["n_calendar_days"] == 3
    assert result["n_zero_position_days"] == 3
    assert result["cum_return"] == 0.0
    assert result["sharpe"] == 0.0


def test_metrics_reject_calendar_bounds_that_drop_observed_rows():
    rows = pd.DataFrame(
        [
            {
                "date": "2025-01-01",
                "o": 100.0,
                "h": 111.0,
                "l": 99.0,
                "cl": 105.0,
            },
            {
                "date": "2025-01-05",
                "o": 100.0,
                "h": 101.0,
                "l": 99.0,
                "cl": 100.0,
            },
        ]
    )
    rows.attrs["evaluation_start"] = "2025-01-02"
    rows.attrs["evaluation_end"] = "2025-01-05"

    with pytest.raises(
        ValueError, match="evaluation calendar must contain every observed row"
    ):
        net_metrics(rows, "tp_eod")


def test_eval_csv_requires_content_bound_json_sidecar(tmp_path):
    csv_path = tmp_path / "eval.csv"
    csv_path.write_text("pattern,sharpe\np,1.0\n", encoding="utf-8")

    with pytest.raises(audit.EvalArtifactError, match="sidecar"):
        audit.load_eval_library_artifact(
            csv_path,
            require_current=False,
        )


def test_eval_sidecar_rejects_csv_and_input_db_changes(tmp_path):
    database = tmp_path / "d1.db"
    database.write_bytes(b"db-v1")
    manifest = audit.build_eval_input_manifest(
        "2026-07-25",
        db_path=database,
    )
    frame = pd.DataFrame(
        [
            {
                "pattern": "ret_7d",
                "regime": "bull_quiet",
                "exit": "sl_first",
                "sharpe": 1.25,
            }
        ]
    )
    payload = audit.build_eval_artifact(
        frame,
        asof="2026-07-25",
        leak_summary={
            "regime_d1_ok": True,
            "feature_shift_ok": True,
            "label_match": 1.0,
        },
        input_manifest=manifest,
    )
    csv_path = tmp_path / "eval.csv"
    json_path = tmp_path / "eval.json"
    audit.atomic_write_bytes(csv_path, audit._eval_csv_bytes(payload))
    audit.atomic_write_json(json_path, payload)

    loaded = audit.load_eval_library_artifact(
        csv_path,
        json_path=json_path,
        db_path=database,
    )
    assert loaded["run_id"] == payload["run_id"]

    csv_path.write_text(
        "pattern,regime,exit,sharpe\nret_7d,bull_quiet,sl_first,9.0\n",
        encoding="utf-8",
    )
    with pytest.raises(audit.EvalArtifactError, match="CSV/JSON"):
        audit.load_eval_library_artifact(
            csv_path,
            json_path=json_path,
            db_path=database,
        )

    audit.atomic_write_bytes(csv_path, audit._eval_csv_bytes(payload))
    database.write_bytes(b"db-v2")
    with pytest.raises(audit.EvalArtifactError, match="input DB"):
        audit.load_eval_library_artifact(
            csv_path,
            json_path=json_path,
            db_path=database,
        )


def test_eval_sidecar_rejects_nonstandard_json_number(tmp_path):
    database = tmp_path / "d1.db"
    database.write_bytes(b"db")
    manifest = audit.build_eval_input_manifest(
        "2026-07-25",
        db_path=database,
    )
    payload = audit.build_eval_artifact(
        pd.DataFrame([{"pattern": "ret_7d", "sharpe": 1.0}]),
        asof="2026-07-25",
        leak_summary={
            "regime_d1_ok": True,
            "feature_shift_ok": True,
            "label_match": 1.0,
        },
        input_manifest=manifest,
    )
    csv_path = tmp_path / "eval.csv"
    json_path = tmp_path / "eval.json"
    audit.atomic_write_bytes(csv_path, audit._eval_csv_bytes(payload))
    audit.atomic_write_json(json_path, payload)
    raw = json_path.read_text(encoding="utf-8").replace(
        '"label_match": 1.0',
        '"label_match": NaN',
        1,
    )
    json_path.write_text(raw, encoding="utf-8")

    with pytest.raises(audit.EvalArtifactError):
        audit.load_eval_library_artifact(
            csv_path,
            json_path=json_path,
            db_path=database,
        )


def test_eval_cli_routes_explicit_paths_without_ignoring_arguments(
    tmp_path,
    monkeypatch,
):
    calls = []
    database = tmp_path / "candles.db"
    csv_path = tmp_path / "eval.csv"
    json_path = tmp_path / "eval.json"
    monkeypatch.setattr(
        audit,
        "run",
        lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(
        audit.sys,
        "argv",
        [
            "eval_library_audit_v1.py",
            "--db",
            str(database),
            "--out-csv",
            str(csv_path),
            "--out-json",
            str(json_path),
        ],
    )

    audit.main()

    assert calls == [
        {
                "db_path": str(database),
                "output_csv": str(csv_path),
                "output_json": str(json_path),
                "completed_through": None,
            }
        ]


def test_eval_cli_help_never_runs_the_expensive_audit(monkeypatch):
    calls = []
    monkeypatch.setattr(
        audit,
        "run",
        lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(
        audit.sys,
        "argv",
        ["eval_library_audit_v1.py", "--help"],
    )

    with pytest.raises(SystemExit) as exc_info:
        audit.main()

    assert exc_info.value.code == 0
    assert calls == []

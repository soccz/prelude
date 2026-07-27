from __future__ import annotations

import json
import logging
import sqlite3

import pandas as pd
import pytest

from ledger.path_quality import PathAssessment
from scripts import close_recommend_ledger as closer
from scripts.recommend_today import RECOMMEND_LEDGER_COLS


M15_SCHEMA = """
CREATE TABLE candles (
    market TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    PRIMARY KEY (market, timestamp)
)
"""

D1_SCHEMA = M15_SCHEMA


def _grid() -> pd.DatetimeIndex:
    return pd.date_range("2026-07-01 09:00:00", periods=96, freq="15min")


def _flat_rows(market, timestamps, price):
    return [
        (
            market,
            ts.strftime("%Y-%m-%d %H:%M:%S"),
            price,
            price,
            price,
            price,
        )
        for ts in timestamps
    ]


def _write_db(path, schema, rows):
    with sqlite3.connect(path) as conn:
        conn.execute(schema)
        conn.executemany("INSERT INTO candles VALUES (?,?,?,?,?,?)", rows)


def _ledger_row(**values):
    row = {c: pd.NA for c in RECOMMEND_LEDGER_COLS}
    row.update(
        {
            "date": "2026-07-01",
            "coin": "KRW-ALT",
            "rank": 1,
            "entry_open": 100.0,
            "sl_pct": -0.03,
            "tp_pct": 0.05,
            "status": "open",
        }
    )
    row.update(values)
    return row


def _patch_databases(monkeypatch, tmp_path, target_rows):
    m15 = tmp_path / "m15.db"
    d1 = tmp_path / "d1.db"
    grid = _grid()
    _write_db(
        m15,
        M15_SCHEMA,
        _flat_rows(
            "KRW-BTC",
            grid.append(
                pd.DatetimeIndex([
                    grid[-1] + pd.Timedelta(minutes=15),
                    grid[-1] + pd.Timedelta(minutes=30),
                ])
            ),
            200.0,
        )
        + target_rows,
    )
    _write_db(
        d1,
        D1_SCHEMA,
        [("KRW-ALT", "2026-07-01 09:00:00", 100.0, 125.0, 90.0, 100.0)],
    )
    monkeypatch.setattr(closer, "M15_DB", str(m15))
    monkeypatch.setattr(closer, "D1_DB", str(d1))


def test_daily_label_read_never_creates_a_missing_database(
    monkeypatch,
    tmp_path,
):
    missing = tmp_path / "missing" / "d1.db"
    monkeypatch.setattr(closer, "D1_DB", str(missing))

    with pytest.raises(FileNotFoundError):
        closer._daily_pump20("KRW-ALT", pd.Timestamp("2026-07-01"))

    assert not missing.exists()
    assert not missing.parent.exists()


def test_canonical_noop_is_revalidated_under_ledger_lock(
    monkeypatch,
    tmp_path,
) -> None:
    ledger_path = tmp_path / "shadow_ledger_recommend.csv"
    lock_held = False

    class _Lock:
        def __enter__(self):
            nonlocal lock_held
            lock_held = True

        def __exit__(self, *_args):
            nonlocal lock_held
            lock_held = False

    monkeypatch.setattr(closer, "ledger_lock", lambda _path: _Lock())

    def validate(**_kwargs):
        assert lock_held is True
        return "skip-zero-pick"

    monkeypatch.setattr(closer, "validate_close_input", validate)

    closer.close_recommend_ledger(
        str(ledger_path),
        pd.Timestamp("2026-07-03"),
        False,
        logging.getLogger("test"),
        decision_date=pd.Timestamp("2026-07-01"),
        cohort="r1-open",
        expected_mode="skip-zero-pick",
        output_root=tmp_path,
    )

    assert not ledger_path.exists()


def test_canonical_mode_change_under_lock_fails_closed(
    monkeypatch,
    tmp_path,
) -> None:
    ledger_path = tmp_path / "shadow_ledger_recommend.csv"
    monkeypatch.setattr(
        closer,
        "validate_close_input",
        lambda **_kwargs: "close",
    )

    with pytest.raises(closer.CloseInputError, match="mode changed"):
        closer.close_recommend_ledger(
            str(ledger_path),
            pd.Timestamp("2026-07-03"),
            False,
            logging.getLogger("test"),
            decision_date=pd.Timestamp("2026-07-01"),
            cohort="r1-open",
            expected_mode="skip-zero-pick",
            output_root=tmp_path,
        )


def test_close_uses_sent_at_slice_records_quality_and_cost_once(monkeypatch, tmp_path):
    grid = _grid()
    target = _flat_rows(
        "KRW-ALT",
        grid.append(pd.DatetimeIndex([grid[-1] + pd.Timedelta(minutes=15)])),
        100.0,
    )
    # 알림 전 09:00 봉에서 SL/TP가 동시에 터져도 sent_at=09:10이면 평가에서 제외한다.
    target[0] = (
        "KRW-ALT",
        grid[0].strftime("%Y-%m-%d %H:%M:%S"),
        100.0,
        106.0,
        96.0,
        100.0,
    )
    _patch_databases(monkeypatch, tmp_path, target)

    ledger_path = tmp_path / "ledger.csv"
    rows = [
        _ledger_row(
            entry_open=90.0,
            delivery_ok=True,
            sent_at="2026-07-01T00:10:00+00:00",
        ),
        _ledger_row(
            coin="KRW-NOT-SENT",
            rank=2,
            delivery_ok=False,
            sent_at=pd.NA,
        ),
        _ledger_row(
            date="2026-06-30",
            coin="KRW-OLD",
            rank=1,
            status="closed",
            exit_reason="TP",
            realized_pct=12.34,
        ),
    ]
    pd.DataFrame(rows, columns=RECOMMEND_LEDGER_COLS).to_csv(ledger_path, index=False)

    closer.close_recommend_ledger(
        str(ledger_path),
        pd.Timestamp("2026-07-03"),
        False,
        logging.getLogger("test"),
        decision_date=pd.Timestamp("2026-07-01"),
    )

    out = pd.read_csv(ledger_path)
    closed = out[out["coin"] == "KRW-ALT"].iloc[0]
    assert closed["status"] == "closed"
    assert closed["exit_reason"] == "EOD"
    assert closed["realized_pct"] == pytest.approx(-0.15)
    assert closed["entry_open"] == pytest.approx(90.0)
    assert closed["execution_entry_open"] == pytest.approx(100.0)
    assert bool(closed["path_complete"]) is True
    assert closed["path_quality"] == "complete"
    assert closed["raw_bars"] == 96
    assert closed["expected_bars"] == 96
    assert closed["flat_filled_bars"] == 0
    assert closed["benchmark_bars"] == 96
    assert closed["path_start_at"] == "2026-07-01T09:15:00"
    assert closed["entry_observable_at"] == "2026-07-01T09:15:00"
    assert closed["entry_price_source"] == "15m_open_at_or_after_delivery"
    assert closed["path_used_bars"] == 96
    assert closed["pump20_hit"] == 1
    assert closed["post_send_pump20_hit"] == 0
    assert "both_recorded" in closed["pump20_label_basis"]

    not_sent = out[out["coin"] == "KRW-NOT-SENT"].iloc[0]
    assert not_sent["status"] == "not_delivered"
    assert not_sent["path_quality"] == "delivery_failed"
    assert pd.isna(not_sent["realized_pct"])

    old = out[out["coin"] == "KRW-OLD"].iloc[0]
    assert old["status"] == "closed"
    assert old["exit_reason"] == "TP"
    assert old["realized_pct"] == pytest.approx(12.34)
    assert pd.isna(old["path_quality"])


def test_sent_at_window_includes_final_15_minutes_of_full_24_hours(
    monkeypatch, tmp_path
):
    grid = _grid()
    final = grid[-1] + pd.Timedelta(minutes=15)
    target = _flat_rows(
        "KRW-ALT",
        grid.append(pd.DatetimeIndex([final])),
        100.0,
    )
    target[-1] = (
        "KRW-ALT",
        final.strftime("%Y-%m-%d %H:%M:%S"),
        100.0,
        106.0,
        100.0,
        105.0,
    )
    _patch_databases(monkeypatch, tmp_path, target)
    ledger_path = tmp_path / "ledger.csv"
    pd.DataFrame(
        [
            _ledger_row(
                delivery_ok=True,
                sent_at="2026-07-01T00:10:00+00:00",
            )
        ],
        columns=RECOMMEND_LEDGER_COLS,
    ).to_csv(ledger_path, index=False)

    closer.close_recommend_ledger(
        str(ledger_path),
        pd.Timestamp("2026-07-03"),
        False,
        logging.getLogger("test"),
        decision_date=pd.Timestamp("2026-07-01"),
    )

    row = pd.read_csv(ledger_path).iloc[0]
    assert row["status"] == "closed"
    assert row["path_used_bars"] == 96
    assert row["exit_reason"] == "TP"
    assert row["realized_pct"] == pytest.approx(4.85)


@pytest.mark.parametrize(
    ("delivery_ok", "sent_at"),
    [
        (True, pd.NA),
        (True, "2026-07-01T09:10:00"),
        ("unknown", pd.NA),
    ],
)
def test_invalid_delivery_metadata_never_falls_back_to_day_open(
    monkeypatch, tmp_path, delivery_ok, sent_at
):
    grid = _grid()
    _patch_databases(
        monkeypatch,
        tmp_path,
        _flat_rows("KRW-ALT", grid, 100.0),
    )
    ledger_path = tmp_path / "ledger.csv"
    pd.DataFrame(
        [_ledger_row(delivery_ok=delivery_ok, sent_at=sent_at)],
        columns=RECOMMEND_LEDGER_COLS,
    ).to_csv(ledger_path, index=False)

    closer.close_recommend_ledger(
        str(ledger_path),
        pd.Timestamp("2026-07-03"),
        False,
        logging.getLogger("test"),
        decision_date=pd.Timestamp("2026-07-01"),
    )

    row = pd.read_csv(ledger_path).iloc[0]
    assert row["status"] == "no_data"
    assert row["path_quality"] == "invalid_delivery_metadata"
    assert pd.isna(row["realized_pct"])


def test_record_only_forward_row_starts_after_immutable_decision(
    monkeypatch,
    tmp_path,
):
    observed_starts: list[pd.Timestamp | None] = []

    def fake_path(_coin, _date, *, start_at=None):
        observed_starts.append(start_at)
        assert start_at == pd.Timestamp("2026-07-27 09:15:00")
        timestamps = tuple(
            pd.date_range(start_at, periods=96, freq="15min")
        )
        return PathAssessment(
            bars=[(100.0, 100.0, 100.0, 100.0)] * 96,
            timestamps=timestamps,
            path_complete=True,
            path_quality="complete",
            raw_bars=96,
            benchmark_bars=96,
        )

    monkeypatch.setattr(closer, "_load_15m_path", fake_path)
    monkeypatch.setattr(
        closer,
        "_daily_pump20",
        lambda *_args, **_kwargs: {"status": "ok", "pump20_hit": 0},
    )
    ledger_path = tmp_path / "ledger.csv"
    pd.DataFrame(
        [
            _ledger_row(
                date="2026-07-27",
                delivery_ok=pd.NA,
                sent_at=pd.NA,
                decision_completed_at="2026-07-27T00:10:01+00:00",
            )
        ],
        columns=RECOMMEND_LEDGER_COLS,
    ).to_csv(ledger_path, index=False)

    closer.close_recommend_ledger(
        str(ledger_path),
        pd.Timestamp("2026-07-29"),
        False,
        logging.getLogger("test"),
        decision_date=pd.Timestamp("2026-07-27"),
    )

    row = pd.read_csv(ledger_path).iloc[0]
    assert observed_starts == [pd.Timestamp("2026-07-27 09:15:00")]
    assert row["status"] == "closed"
    assert row["path_start_at"] == "2026-07-27T09:15:00"
    assert row["entry_price_source"] == "15m_open_at_or_after_decision"


def test_server_second_truncation_uses_later_decision_completion(
    monkeypatch,
    tmp_path,
):
    observed_starts: list[pd.Timestamp | None] = []

    def fake_path(_coin, _date, *, start_at=None):
        observed_starts.append(start_at)
        return PathAssessment(
            bars=[(100.0, 100.0, 100.0, 100.0)] * 96,
            timestamps=tuple(
                pd.date_range(start_at, periods=96, freq="15min")
            ),
            path_complete=True,
            path_quality="complete",
            raw_bars=96,
            benchmark_bars=96,
        )

    monkeypatch.setattr(closer, "_load_15m_path", fake_path)
    monkeypatch.setattr(
        closer,
        "_daily_pump20",
        lambda *_args, **_kwargs: {"status": "ok", "pump20_hit": 0},
    )
    ledger_path = tmp_path / "ledger.csv"
    pd.DataFrame(
        [
            _ledger_row(
                date="2026-07-27",
                delivery_ok=True,
                sent_at="2026-07-27T00:10:01+00:00",
                decision_completed_at=(
                    "2026-07-27T00:10:01.700000+00:00"
                ),
            )
        ],
        columns=RECOMMEND_LEDGER_COLS,
    ).to_csv(ledger_path, index=False)

    closer.close_recommend_ledger(
        str(ledger_path),
        pd.Timestamp("2026-07-29"),
        False,
        logging.getLogger("test"),
        decision_date=pd.Timestamp("2026-07-27"),
    )

    row = pd.read_csv(ledger_path).iloc[0]
    assert observed_starts == [pd.Timestamp("2026-07-27 09:15:00")]
    assert row["status"] == "closed"
    assert row["path_start_at"] == "2026-07-27T09:15:00"


def test_server_time_one_second_before_decision_still_fails_closed(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        closer,
        "_load_15m_path",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid chronology must fail before path loading"
        ),
    )
    ledger_path = tmp_path / "ledger.csv"
    pd.DataFrame(
        [
            _ledger_row(
                date="2026-07-27",
                delivery_ok=True,
                sent_at="2026-07-27T00:10:00+00:00",
                decision_completed_at="2026-07-27T00:10:01+00:00",
            )
        ],
        columns=RECOMMEND_LEDGER_COLS,
    ).to_csv(ledger_path, index=False)

    closer.close_recommend_ledger(
        str(ledger_path),
        pd.Timestamp("2026-07-29"),
        False,
        logging.getLogger("test"),
        decision_date=pd.Timestamp("2026-07-27"),
    )

    row = pd.read_csv(ledger_path).iloc[0]
    assert row["status"] == "no_data"
    assert row["path_quality"] == "invalid_delivery_metadata"


def test_post_activation_row_without_decision_time_fails_closed(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        closer,
        "_load_15m_path",
        lambda *_args, **_kwargs: pytest.fail(
            "missing decision time must fail before path evaluation"
        ),
    )
    ledger_path = tmp_path / "ledger.csv"
    pd.DataFrame(
        [
            _ledger_row(
                date="2026-07-27",
                delivery_ok=pd.NA,
                sent_at=pd.NA,
                decision_completed_at=pd.NA,
            )
        ],
        columns=RECOMMEND_LEDGER_COLS,
    ).to_csv(ledger_path, index=False)

    closer.close_recommend_ledger(
        str(ledger_path),
        pd.Timestamp("2026-07-29"),
        False,
        logging.getLogger("test"),
        decision_date=pd.Timestamp("2026-07-27"),
    )

    row = pd.read_csv(ledger_path).iloc[0]
    assert row["status"] == "no_data"
    assert row["path_quality"] == "invalid_delivery_metadata"
    assert pd.isna(row["realized_pct"])


def test_same_bar_sl_remains_first_and_flat_gap_is_audited(monkeypatch, tmp_path):
    grid = _grid()
    target = _flat_rows("KRW-ALT", grid.delete(10), 100.0)
    target[0] = (
        "KRW-ALT",
        grid[0].strftime("%Y-%m-%d %H:%M:%S"),
        100.0,
        106.0,
        96.0,
        100.0,
    )
    _patch_databases(monkeypatch, tmp_path, target)
    ledger_path = tmp_path / "ledger.csv"
    pd.DataFrame([_ledger_row()], columns=RECOMMEND_LEDGER_COLS).to_csv(
        ledger_path, index=False
    )

    closer.close_recommend_ledger(
        str(ledger_path),
        pd.Timestamp("2026-07-03"),
        False,
        logging.getLogger("test"),
        decision_date=pd.Timestamp("2026-07-01"),
    )

    row = pd.read_csv(ledger_path).iloc[0]
    assert row["status"] == "closed"
    assert row["exit_reason"] == "SL"
    assert row["realized_pct"] == pytest.approx(-3.15)
    assert row["path_quality"] == "flat_filled"
    assert row["raw_bars"] == 95
    assert row["flat_filled_bars"] == 1
    assert row["path_used_bars"] == 96


def test_benchmark_gap_defers_close_and_persists_audit_metadata(monkeypatch, tmp_path):
    grid = _grid()
    m15 = tmp_path / "m15.db"
    d1 = tmp_path / "d1.db"
    _write_db(
        m15,
        M15_SCHEMA,
        _flat_rows(
            "KRW-BTC",
            grid.delete(40).append(
                pd.DatetimeIndex([grid[-1] + pd.Timedelta(minutes=15)])
            ),
            200.0,
        )
        + _flat_rows("KRW-ALT", grid, 100.0),
    )
    _write_db(
        d1,
        D1_SCHEMA,
        [("KRW-ALT", "2026-07-01 09:00:00", 100.0, 110.0, 90.0, 100.0)],
    )
    monkeypatch.setattr(closer, "M15_DB", str(m15))
    monkeypatch.setattr(closer, "D1_DB", str(d1))

    ledger_path = tmp_path / "ledger.csv"
    pd.DataFrame([_ledger_row()], columns=RECOMMEND_LEDGER_COLS).to_csv(
        ledger_path, index=False
    )

    closer.close_recommend_ledger(
        str(ledger_path),
        pd.Timestamp("2026-07-03"),
        False,
        logging.getLogger("test"),
        decision_date=pd.Timestamp("2026-07-01"),
    )

    row = pd.read_csv(ledger_path).iloc[0]
    assert row["status"] == "no_data"
    assert bool(row["path_complete"]) is False
    assert row["path_quality"] == "benchmark_gap"
    assert row["raw_bars"] == 96
    assert row["benchmark_bars"] == 95
    assert pd.isna(row["realized_pct"])


def test_exact_decision_date_never_closes_ungated_backlog(
    monkeypatch,
    tmp_path,
):
    timestamps = tuple(
        pd.date_range("2026-07-02 09:00:00", periods=96, freq="15min")
    )
    assessment = PathAssessment(
        bars=[(100.0, 100.0, 100.0, 100.0)] * 96,
        timestamps=timestamps,
        path_complete=True,
        path_quality="complete",
        raw_bars=96,
        benchmark_bars=96,
    )
    monkeypatch.setattr(closer, "_load_15m_path", lambda *_a, **_k: assessment)
    monkeypatch.setattr(
        closer,
        "_daily_pump20",
        lambda *_a, **_k: {"status": "ok", "pump20_hit": 0},
    )
    ledger_path = tmp_path / "ledger.csv"
    rows = [
        _ledger_row(date="2026-07-01", coin="KRW-OLD"),
        _ledger_row(date="2026-07-02", coin="KRW-TARGET"),
    ]
    pd.DataFrame(rows, columns=RECOMMEND_LEDGER_COLS).to_csv(
        ledger_path,
        index=False,
    )

    closer.close_recommend_ledger(
        str(ledger_path),
        pd.Timestamp("2026-07-03"),
        False,
        logging.getLogger("test"),
        decision_date=pd.Timestamp("2026-07-02"),
    )

    out = pd.read_csv(ledger_path).set_index("coin")
    assert out.loc["KRW-TARGET", "status"] == "closed"
    assert out.loc["KRW-OLD", "status"] == "open"
    assert pd.isna(out.loc["KRW-OLD", "realized_pct"])


def test_decision_date_newer_than_cutoff_is_rejected(tmp_path):
    ledger_path = tmp_path / "ledger.csv"
    pd.DataFrame(
        [_ledger_row(date="2026-07-03")],
        columns=RECOMMEND_LEDGER_COLS,
    ).to_csv(ledger_path, index=False)

    with pytest.raises(ValueError, match="newer than close cutoff"):
        closer.close_recommend_ledger(
            str(ledger_path),
            pd.Timestamp("2026-07-03"),
            False,
            logging.getLogger("test"),
            decision_date=pd.Timestamp("2026-07-03"),
        )


def test_no_decision_revalidated_under_lock_writes_marker(
    monkeypatch,
    tmp_path,
) -> None:
    ledger_path = tmp_path / "shadow_ledger_recommend.csv"

    def validate(**_kwargs):
        raise closer.MissingCloseEvidenceError("no evidence")

    monkeypatch.setattr(closer, "validate_close_input", validate)

    closer.close_recommend_ledger(
        str(ledger_path),
        pd.Timestamp("2026-07-28"),
        False,
        logging.getLogger("test"),
        decision_date=pd.Timestamp("2026-07-27"),
        cohort="r1-open",
        expected_mode="skip-no-decision",
        output_root=tmp_path,
    )

    marker = tmp_path / "close_no_decision" / "r1-open" / "2026-07-27.json"
    assert marker.exists()
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["asof"] == "2026-07-27"
    assert payload["cohort"] == "r1-open"
    assert not ledger_path.exists()


def test_no_decision_under_lock_rejects_lingering_ledger_row(
    monkeypatch,
    tmp_path,
) -> None:
    ledger_path = tmp_path / "shadow_ledger_recommend.csv"
    ledger_path.write_text(
        "date,status\n2026-07-27,closed\n",
        encoding="utf-8",
    )

    def validate(**_kwargs):
        raise closer.MissingCloseEvidenceError("no evidence")

    monkeypatch.setattr(closer, "validate_close_input", validate)

    with pytest.raises(closer.MissingCloseEvidenceError):
        closer.close_recommend_ledger(
            str(ledger_path),
            pd.Timestamp("2026-07-28"),
            False,
            logging.getLogger("test"),
            decision_date=pd.Timestamp("2026-07-27"),
            cohort="r1-open",
            expected_mode="skip-no-decision",
            output_root=tmp_path,
        )

    assert not (tmp_path / "close_no_decision").exists()

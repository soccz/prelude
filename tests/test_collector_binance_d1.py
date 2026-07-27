from __future__ import annotations

import sys
from datetime import datetime, timezone

import pandas as pd
import pytest

import data.collector_binance_d1 as collector


class _Exchange:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def fetch_ohlcv(self, *_args, **_kwargs):
        self.calls += 1
        value = next(self.responses)
        if isinstance(value, Exception):
            raise value
        return value


def test_fetch_page_retries_none_then_returns_rows(monkeypatch):
    exchange = _Exchange([None, [[1, 2, 3, 4, 5, 6]]])
    sleeps = []
    monkeypatch.setattr(collector, "get_exchange", lambda: exchange)
    monkeypatch.setattr(collector.time, "sleep", sleeps.append)

    rows = collector._fetch_page("BTC/USDT", 0)

    assert rows == [[1, 2, 3, 4, 5, 6]]
    assert exchange.calls == 2
    assert sleeps == [collector.RETRY_BACKOFF]


def test_fetch_page_raises_after_retry_exhaustion_without_final_sleep(monkeypatch):
    exchange = _Exchange([RuntimeError("x"), None, RuntimeError("z")])
    sleeps = []
    monkeypatch.setattr(collector, "get_exchange", lambda: exchange)
    monkeypatch.setattr(collector.time, "sleep", sleeps.append)

    with pytest.raises(collector.FetchPageError, match="failed after 3 attempts"):
        collector._fetch_page("BTC/USDT", 0)

    assert exchange.calls == collector.RETRY_MAX
    assert sleeps == [
        collector.RETRY_BACKOFF,
        collector.RETRY_BACKOFF ** 2,
    ]


def test_fetch_page_retries_empty_then_returns_rows(monkeypatch):
    exchange = _Exchange([[], [[1, 2, 3, 4, 5, 6]]])
    sleeps = []
    monkeypatch.setattr(collector, "get_exchange", lambda: exchange)
    monkeypatch.setattr(collector.time, "sleep", sleeps.append)

    assert collector._fetch_page("BTC/USDT", 0) == [[1, 2, 3, 4, 5, 6]]
    assert exchange.calls == 2
    assert sleeps == [collector.RETRY_BACKOFF]


def test_fetch_page_raises_after_empty_retry_exhaustion(monkeypatch):
    exchange = _Exchange([[], [], []])
    sleeps = []
    monkeypatch.setattr(collector, "get_exchange", lambda: exchange)
    monkeypatch.setattr(collector.time, "sleep", sleeps.append)

    with pytest.raises(collector.FetchPageError, match="empty page"):
        collector._fetch_page("BTC/USDT", 0)

    assert exchange.calls == collector.RETRY_MAX
    assert sleeps == [
        collector.RETRY_BACKOFF,
        collector.RETRY_BACKOFF ** 2,
    ]


def test_collect_market_never_accepts_empty_page(monkeypatch, tmp_path):
    monkeypatch.setattr(collector, "init_db", lambda _path: None)
    monkeypatch.setattr(
        collector,
        "latest_timestamp",
        lambda _path, _market: None,
    )
    monkeypatch.setattr(collector, "_fetch_page", lambda _pair, _since: [])

    with pytest.raises(collector.FetchPageError, match="empty page"):
        collector.collect_market(
            "BTC",
            days=1,
            db_path=tmp_path / "binance.db",
        )


def test_collect_market_rejects_nonpositive_days_before_db_mutation(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        collector,
        "init_db",
        lambda _path: pytest.fail("must reject before DB mutation"),
    )

    with pytest.raises(ValueError, match="days must be positive"):
        collector.collect_market(
            "BTC",
            days=0,
            db_path=tmp_path / "binance.db",
        )


def test_collect_market_preserves_unicode_binance_base_identity(
    monkeypatch,
    tmp_path,
):
    calls = {}
    monkeypatch.setattr(collector, "init_db", lambda _path: None)
    monkeypatch.setattr(
        collector,
        "latest_timestamp",
        lambda _path, _market: None,
    )

    def fetch(pair, _since):
        calls["pair"] = pair
        current_day_ms = int(
            pd.Timestamp.now(tz="UTC").floor("D").timestamp() * 1000
        )
        return [[current_day_ms, 100.0, 101.0, 99.0, 100.5, 10.0]]

    def save(_path, frame, market):
        calls["market"] = market
        return len(frame)

    monkeypatch.setattr(collector, "_fetch_page", fetch)
    monkeypatch.setattr(collector, "save_candles", save)

    saved = collector.collect_market(
        "币安人生",
        days=1,
        db_path=tmp_path / "binance.db",
    )

    assert saved == 1
    assert calls == {
        "pair": "币安人生/USDT",
        "market": "BINANCE-币安人生USDT",
    }


def test_utc_naive_epoch_conversion_is_independent_of_host_timezone():
    assert collector._utc_naive_to_epoch_ms(
        datetime(1970, 1, 2)
    ) == 86_400_000
    assert collector._utc_naive_to_epoch_ms(
        datetime(1970, 1, 2, tzinfo=timezone.utc)
    ) == 86_400_000


def test_partial_history_resumes_from_requested_boundary(
    monkeypatch,
    tmp_path,
):
    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            value = cls(2026, 7, 26, 12, tzinfo=timezone.utc)
            return value if tz is not None else value.replace(tzinfo=None)

    observed_since: list[int] = []
    monkeypatch.setattr(collector, "datetime", FrozenDatetime)
    monkeypatch.setattr(collector, "init_db", lambda _path: None)
    monkeypatch.setattr(
        collector,
        "latest_timestamp",
        lambda _path, _market: pd.Timestamp("2026-07-26 00:00:00"),
    )
    monkeypatch.setattr(
        collector,
        "oldest_timestamp",
        lambda _path, _market: pd.Timestamp("2026-07-25 00:00:00"),
    )

    def fetch(_pair, since_ms):
        observed_since.append(since_ms)
        return [
            [
                int(pd.Timestamp("2026-07-26", tz="UTC").timestamp() * 1000),
                100.0,
                101.0,
                99.0,
                100.5,
                10.0,
            ]
        ]

    monkeypatch.setattr(collector, "_fetch_page", fetch)
    monkeypatch.setattr(
        collector,
        "save_candles",
        lambda _path, frame, _market: len(frame),
    )

    collector.collect_market(
        "BTC",
        days=10,
        db_path=tmp_path / "binance.db",
    )

    expected = collector._utc_naive_to_epoch_ms(
        datetime(2026, 7, 16)
    )
    assert observed_since == [expected]


def test_short_page_does_not_silently_end_before_current_boundary(
    monkeypatch,
    tmp_path,
):
    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            value = cls(2026, 7, 26, 12, tzinfo=timezone.utc)
            return value if tz is not None else value.replace(tzinfo=None)

    observed_since: list[int] = []
    page_starts = iter(
        [
            datetime(2026, 7, 24),
            datetime(2026, 7, 26),
        ]
    )
    monkeypatch.setattr(collector, "datetime", FrozenDatetime)
    monkeypatch.setattr(collector, "init_db", lambda _path: None)
    monkeypatch.setattr(
        collector,
        "latest_timestamp",
        lambda _path, _market: None,
    )

    def fetch(_pair, since_ms):
        observed_since.append(since_ms)
        timestamp_ms = collector._utc_naive_to_epoch_ms(next(page_starts))
        return [[timestamp_ms, 100.0, 101.0, 99.0, 100.5, 10.0]]

    monkeypatch.setattr(collector, "_fetch_page", fetch)
    monkeypatch.setattr(
        collector,
        "save_candles",
        lambda _path, frame, _market: len(frame),
    )
    monkeypatch.setattr(collector.time, "sleep", lambda *_args: None)

    assert collector.collect_market(
        "BTC",
        days=3,
        db_path=tmp_path / "binance.db",
    ) == 2
    assert len(observed_since) == 2
    assert observed_since[1] == collector._utc_naive_to_epoch_ms(
        datetime(2026, 7, 25)
    )


def test_nonprogressing_binance_page_fails_before_write(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(collector, "init_db", lambda _path: None)
    monkeypatch.setattr(
        collector,
        "latest_timestamp",
        lambda _path, _market: None,
    )

    def stale_page(_pair, since_ms):
        return [[since_ms - 1, 100.0, 101.0, 99.0, 100.5, 10.0]]

    monkeypatch.setattr(collector, "_fetch_page", stale_page)
    monkeypatch.setattr(
        collector,
        "save_candles",
        lambda *_args, **_kwargs: pytest.fail("invalid page must not be written"),
    )

    with pytest.raises(collector.FetchPageError, match="no forward progress"):
        collector.collect_market(
            "BTC",
            days=1,
            db_path=tmp_path / "binance.db",
        )


def test_main_all_returns_nonzero_when_any_market_failed(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collector_binance_d1.py",
            "--all",
            "--days",
            "3",
            "--db",
            str(tmp_path / "binance.db"),
        ],
    )
    monkeypatch.setattr(
        collector,
        "get_binance_usdt_symbols",
        lambda: ["BTC", "ETH"],
    )
    monkeypatch.setattr(
        collector,
        "collect_all",
        lambda *_args, **_kwargs: {"BTC": 2, "ETH": -1},
    )
    monkeypatch.setattr(collector, "stats", lambda _path: pd.DataFrame())

    assert collector.main() == 1


def test_cli_collection_modes_are_required_and_mutually_exclusive(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["collector_binance_d1.py", "--all", "--coin", "BTC"],
    )

    with pytest.raises(SystemExit) as exc:
        collector.main()

    assert exc.value.code == 2

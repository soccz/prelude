from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from data import collector_d1
from data import collector_15m_upbit, collector_4h
from data.market_universe import STABLECOIN_KRW_MARKETS


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        (
            datetime(2026, 7, 30, 8, 59, 59),
            datetime(2026, 7, 29, 9),
        ),
        (
            datetime(2026, 7, 30, 9),
            datetime(2026, 7, 30, 9),
        ),
        (
            datetime(2026, 7, 30, 9, 0, 1),
            datetime(2026, 7, 30, 9),
        ),
        (
            datetime(2026, 7, 29, 23, 59, 59, tzinfo=timezone.utc),
            datetime(2026, 7, 29, 9),
        ),
        (
            datetime(2026, 7, 30, 0, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 7, 30, 9),
        ),
    ],
)
def test_current_d1_boundary_uses_kst_started_candle(now, expected):
    result = collector_d1.current_d1_boundary(now)

    assert result == expected
    assert result.tzinfo is None


def test_markets_at_d1_boundary_requires_exact_sqlite_timestamp(tmp_path):
    db_path = tmp_path / "upbit_d1.db"

    def candle(at: datetime) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1.0],
                "value": [100.5],
            },
            index=pd.DatetimeIndex([at]),
        )

    boundary = datetime(2026, 7, 30, 9)
    collector_d1.save_candles(
        db_path,
        candle(boundary),
        "KRW-EXACT",
    )
    collector_d1.save_candles(
        db_path,
        candle(boundary - timedelta(days=1)),
        "KRW-PREVIOUS",
    )
    collector_d1.save_candles(
        db_path,
        candle(boundary + timedelta(seconds=1)),
        "KRW-AFTER",
    )

    assert collector_d1._markets_at_d1_boundary_readonly(
        db_path,
        boundary,
    ) == {"KRW-EXACT"}


def test_refresh_targets_only_missing_and_rechecks_exact_boundary(
    monkeypatch,
    tmp_path,
):
    now = datetime(2026, 7, 30, 9, 8)
    db_path = tmp_path / "upbit_d1.db"
    exact_reads = iter(
        [
            {"KRW-BTC"},
            {"KRW-BTC", "KRW-AERO"},
        ]
    )
    queries: list[tuple[Path, datetime]] = []
    calls: list[tuple[list[str], int, Path]] = []

    monkeypatch.setattr(collector_d1, "init_db", lambda _path: None)

    def fake_exact(path, boundary):
        queries.append((path, boundary))
        return next(exact_reads)

    def fake_collect(markets, days, db_path):
        calls.append((list(markets), days, db_path))
        return {"KRW-AERO": 1, "KRW-LAYER": 0}

    monkeypatch.setattr(
        collector_d1,
        "_markets_at_d1_boundary_readonly",
        fake_exact,
    )
    monkeypatch.setattr(collector_d1, "collect_all", fake_collect)

    result = collector_d1.refresh_current_d1_boundary(
        db_path=db_path,
        now=now,
        live_markets=[
            "KRW-LAYER",
            "KRW-BTC",
            "KRW-AERO",
            "KRW-AERO",
        ],
    )

    boundary = datetime(2026, 7, 30, 9)
    assert calls == [(["KRW-AERO", "KRW-LAYER"], 1, db_path)]
    assert queries == [(db_path, boundary), (db_path, boundary)]
    assert result.live_count == 3
    assert result.missing_before == ("KRW-AERO", "KRW-LAYER")
    assert result.unresolved_after == ("KRW-LAYER",)
    assert result.failed_markets == ()


def test_refresh_no_missing_never_collects(monkeypatch, tmp_path):
    monkeypatch.setattr(collector_d1, "init_db", lambda _path: None)
    monkeypatch.setattr(
        collector_d1,
        "_markets_at_d1_boundary_readonly",
        lambda *_args: {"KRW-BTC"},
    )
    monkeypatch.setattr(
        collector_d1,
        "collect_all",
        lambda *_args, **_kwargs: pytest.fail("must not collect"),
    )

    result = collector_d1.refresh_current_d1_boundary(
        db_path=tmp_path / "upbit_d1.db",
        now=datetime(2026, 7, 30, 9, 8),
        live_markets=["KRW-BTC"],
    )

    assert result.results == {}
    assert result.missing_before == ()
    assert result.unresolved_after == ()
    assert result.failed_markets == ()


def test_refresh_separates_fetch_failure_from_unresolved(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(collector_d1, "init_db", lambda _path: None)
    monkeypatch.setattr(
        collector_d1,
        "_markets_at_d1_boundary_readonly",
        lambda *_args: {"KRW-BTC"},
    )
    monkeypatch.setattr(
        collector_d1,
        "collect_all",
        lambda *_args, **_kwargs: {
            "KRW-AERO": -1,
            "KRW-LAYER": 0,
        },
    )

    result = collector_d1.refresh_current_d1_boundary(
        db_path=tmp_path / "upbit_d1.db",
        now=datetime(2026, 7, 30, 9, 8),
        live_markets=["KRW-BTC", "KRW-AERO", "KRW-LAYER"],
    )

    assert result.failed_markets == ("KRW-AERO",)
    assert result.unresolved_after == ("KRW-AERO", "KRW-LAYER")


def test_refresh_defers_broad_gap_to_fail_closed_health_gate(
    monkeypatch,
    tmp_path,
):
    markets = [
        f"KRW-X{index:03d}"
        for index in range(
            collector_d1.MAX_CURRENT_BOUNDARY_REFRESH_MARKETS + 1
        )
    ]
    monkeypatch.setattr(collector_d1, "init_db", lambda _path: None)
    monkeypatch.setattr(
        collector_d1,
        "_markets_at_d1_boundary_readonly",
        lambda *_args: set(),
    )
    monkeypatch.setattr(
        collector_d1,
        "collect_all",
        lambda *_args, **_kwargs: pytest.fail("broad retry must be deferred"),
    )

    result = collector_d1.refresh_current_d1_boundary(
        db_path=tmp_path / "upbit_d1.db",
        now=datetime(2026, 7, 30, 9, 8),
        live_markets=markets,
    )

    assert result.results == {}
    assert result.failed_markets == ()
    assert result.unresolved_after == tuple(markets)


def test_refresh_cli_reports_unresolved_without_full_stats(
    monkeypatch,
    tmp_path,
    capsys,
):
    db_path = tmp_path / "upbit_d1.db"
    result = collector_d1.CurrentBoundaryRefresh(
        boundary=datetime(2026, 7, 30, 9),
        live_count=2,
        missing_before=("KRW-AERO",),
        unresolved_after=("KRW-AERO",),
        results={"KRW-AERO": 0},
        failed_markets=(),
    )
    monkeypatch.setattr(
        collector_d1,
        "refresh_current_d1_boundary",
        lambda **_kwargs: result,
    )
    monkeypatch.setattr(
        collector_d1,
        "stats",
        lambda *_args: pytest.fail("refresh must not scan full DB stats"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collector_d1.py",
            "--refresh-current-boundary",
            "--db",
            str(db_path),
        ],
    )

    assert collector_d1.main() == 0
    output = capsys.readouterr().out
    assert "missing_before=1" in output
    assert "attempted=1" in output
    assert "failed=0" in output
    assert "unresolved=1" in output
    assert "final health gate decides" in output


def test_refresh_cli_is_mutually_exclusive_with_update(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collector_d1.py",
            "--update",
            "--refresh-current-boundary",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        collector_d1.main()

    assert exc_info.value.code == 2


def test_update_reconciles_new_live_markets_before_existing(monkeypatch, tmp_path):
    now = datetime(2026, 7, 26, 9, 5)
    db_ranges = {
        "KRW-BTC": (now - timedelta(days=1000), now - timedelta(days=1)),
        "KRW-DELISTED": (now - timedelta(days=2000), now - timedelta(days=30)),
    }
    calls: list[tuple[list[str], int, Path]] = []

    monkeypatch.setattr(collector_d1, "_now_kst_naive", lambda: now)
    monkeypatch.setattr(
        collector_d1,
        "get_krw_markets",
        lambda: ["KRW-BTC", "KRW-NEW"],
    )
    monkeypatch.setattr(
        collector_d1,
        "market_timestamp_ranges_readonly",
        lambda _db_path: dict(db_ranges),
    )

    def fake_collect_all(markets, days, db_path):
        market_list = list(markets)
        calls.append((market_list, days, db_path))
        for market in market_list:
            db_ranges[market] = (
                now - timedelta(days=days),
                now - timedelta(days=1),
            )
        return {market: 1 for market in market_list}

    monkeypatch.setattr(collector_d1, "collect_all", fake_collect_all)

    db_path = tmp_path / "upbit_d1.db"
    result = collector_d1.update_existing(
        db_path=db_path,
        new_market_days=365,
    )

    assert calls == [
        (["KRW-NEW"], 365, db_path),
        (["KRW-BTC"], 7, db_path),
    ]
    assert isinstance(result, dict)
    assert result == {
        "KRW-NEW": 1,
        "KRW-BTC": 1,
    }
    assert result.coverage.new_markets == ("KRW-NEW",)
    assert result.coverage.backfill_markets == ("KRW-NEW",)
    assert result.coverage.inactive_db_markets == ("KRW-DELISTED",)
    assert result.coverage.missing_after == ()
    assert result.coverage.ratio_before == 0.5
    assert result.coverage.ratio_after == 1.0


def test_update_reports_failed_new_market_as_missing(monkeypatch, tmp_path):
    now = datetime(2026, 7, 26, 9, 5)
    db_ranges = {
        "KRW-BTC": (now - timedelta(days=2000), now - timedelta(days=1)),
    }

    monkeypatch.setattr(collector_d1, "_now_kst_naive", lambda: now)
    monkeypatch.setattr(
        collector_d1,
        "get_krw_markets",
        lambda: ["KRW-BTC", "KRW-NEW"],
    )
    monkeypatch.setattr(
        collector_d1,
        "market_timestamp_ranges_readonly",
        lambda _db_path: dict(db_ranges),
    )

    def fake_collect_all(markets, days, db_path):
        del days, db_path
        market_list = list(markets)
        if market_list == ["KRW-NEW"]:
            return {"KRW-NEW": -1}
        return {market: 1 for market in market_list}

    monkeypatch.setattr(collector_d1, "collect_all", fake_collect_all)

    result = collector_d1.update_existing(db_path=tmp_path / "upbit_d1.db")

    assert result.coverage.covered_after_count == 1
    assert result.coverage.missing_after == ("KRW-NEW",)
    assert result.coverage.failed_markets == ("KRW-NEW",)
    assert result.coverage.ratio_after == 0.5


def test_update_retries_partial_new_market_by_oldest_coverage(monkeypatch, tmp_path):
    """부분 저장 뒤 실패해 market이 존재해도 다음 update는 3년 backfill을 재개."""
    now = datetime(2026, 7, 26, 9, 5)
    db_ranges = {
        "KRW-BTC": (now - timedelta(days=2000), now - timedelta(days=1)),
        # 전 실행에서 최근 30일만 저장된 신규 market.
        "KRW-NEW": (now - timedelta(days=30), now - timedelta(days=1)),
        "KRW-DELISTED": (now - timedelta(days=3000), now - timedelta(days=100)),
    }
    calls: list[tuple[list[str], int]] = []

    monkeypatch.setattr(collector_d1, "_now_kst_naive", lambda: now)
    monkeypatch.setattr(
        collector_d1,
        "get_krw_markets",
        lambda: ["KRW-BTC", "KRW-NEW"],
    )
    monkeypatch.setattr(
        collector_d1,
        "market_timestamp_ranges_readonly",
        lambda _db_path: dict(db_ranges),
    )

    def fake_collect_all(markets, days, db_path):
        del db_path
        market_list = list(markets)
        calls.append((market_list, days))
        if market_list == ["KRW-NEW"]:
            return {"KRW-NEW": -1}
        return {market: 1 for market in market_list}

    monkeypatch.setattr(collector_d1, "collect_all", fake_collect_all)

    result = collector_d1.update_existing(
        db_path=tmp_path / "upbit_d1.db",
        new_market_days=365 * 3,
    )

    assert calls == [
        (["KRW-NEW"], 365 * 3),
        (["KRW-BTC"], 7),
    ]
    assert result.coverage.new_markets == ()
    assert result.coverage.backfill_markets == ("KRW-NEW",)
    assert result.coverage.inactive_db_markets == ("KRW-DELISTED",)
    assert result.coverage.failed_markets == ("KRW-NEW",)


def test_get_krw_markets_rejects_empty_live_response(monkeypatch):
    monkeypatch.setattr(collector_d1.pyupbit, "get_tickers", lambda **_kwargs: None)
    monkeypatch.setattr(collector_d1.time, "sleep", lambda *_args: None)

    with pytest.raises(RuntimeError, match="request failed"):
        collector_d1.get_krw_markets()


def test_get_krw_markets_retries_transient_discovery_failure(monkeypatch):
    responses = iter(
        [
            RuntimeError("transient"),
            None,
            ["KRW-BTC"],
        ]
    )
    sleeps: list[float] = []

    def get_tickers(**_kwargs):
        value = next(responses)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(collector_d1.pyupbit, "get_tickers", get_tickers)
    monkeypatch.setattr(
        collector_d1,
        "_markets_with_ticker_snapshot",
        lambda markets: list(markets),
    )
    monkeypatch.setattr(
        collector_d1.time,
        "sleep",
        lambda seconds: sleeps.append(seconds),
    )

    assert collector_d1.get_krw_markets() == ["KRW-BTC"]
    assert sleeps == [
        collector_d1.RETRY_BACKOFF,
        collector_d1.RETRY_BACKOFF**2,
    ]


@pytest.mark.parametrize(
    "payload",
    [
        [],
        "KRW-BTC",
        ["KRW-BTC", "KRW-BTC"],
        ["BTC-KRW"],
        ["KRW-btc"],
        [None],
        ["KRW-USDT", "KRW-USDC"],
    ],
)
def test_get_krw_markets_validates_market_list_contract(monkeypatch, payload):
    monkeypatch.setattr(
        collector_d1.pyupbit,
        "get_tickers",
        lambda **_kwargs: payload,
    )

    with pytest.raises(RuntimeError):
        collector_d1.get_krw_markets()


def test_get_krw_markets_uses_central_stablecoin_policy_and_logs_identities(
    caplog,
    monkeypatch,
):
    raw_markets = [
        "KRW-BTC",
        "KRW-WBTC",
        "KRW-WETH",
        *sorted(STABLECOIN_KRW_MARKETS),
    ]
    monkeypatch.setattr(
        collector_d1.pyupbit,
        "get_tickers",
        lambda **_kwargs: raw_markets,
    )
    monkeypatch.setattr(
        collector_d1,
        "_markets_with_ticker_snapshot",
        lambda markets: list(markets),
    )

    with caplog.at_level("INFO", logger="collector_d1"):
        result = collector_d1.get_krw_markets()

    assert result == ["KRW-BTC", "KRW-WBTC", "KRW-WETH"]
    assert (
        "Excluded 5 stablecoin KRW market(s): "
        "KRW-USD1, KRW-USDC, KRW-USDE, KRW-USDS, KRW-USDT"
    ) in caplog.text
    assert "Active signal-eligible KRW markets: 3" in caplog.text


def test_get_krw_markets_can_preserve_stablecoin_identities_for_audit(
    caplog,
    monkeypatch,
):
    raw_markets = [
        "KRW-BTC",
        "KRW-WBTC",
        *sorted(STABLECOIN_KRW_MARKETS),
    ]
    monkeypatch.setattr(
        collector_d1.pyupbit,
        "get_tickers",
        lambda **_kwargs: raw_markets,
    )
    monkeypatch.setattr(
        collector_d1,
        "_markets_with_ticker_snapshot",
        lambda markets: list(markets),
    )

    with caplog.at_level("INFO", logger="collector_d1"):
        result = collector_d1.get_krw_markets(
            include_stablecoins_for_audit=True
        )

    assert result == sorted(raw_markets)
    assert "stablecoin_identities_retained=5" in caplog.text
    assert "Excluded 5 stablecoin" not in caplog.text


def test_intraday_collectors_reuse_d1_active_market_policy():
    assert collector_4h.get_krw_markets is collector_d1.get_krw_markets
    assert collector_15m_upbit.get_krw_markets is collector_d1.get_krw_markets


def test_get_krw_markets_excludes_listed_pair_without_trade_snapshot(
    monkeypatch,
):
    requested: list[str] = []

    class FakeResponse:
        def __init__(self, markets):
            self.markets = markets

        def raise_for_status(self):
            return None

        def json(self):
            return [
                {"market": "KRW-BTC", "trade_price": 100.0}
            ] if "KRW-BTC" in self.markets else []

    def fake_get(_url, *, params, **_kwargs):
        markets = params["markets"].split(",")
        requested.extend(markets)
        return FakeResponse(markets)

    monkeypatch.setattr(
        collector_d1.pyupbit,
        "get_tickers",
        lambda **_kwargs: ["KRW-BTC", "KRW-EUL", "KRW-USDT"],
    )
    monkeypatch.setattr(collector_d1.requests, "get", fake_get)

    assert collector_d1.get_krw_markets() == ["KRW-BTC"]
    assert requested == ["KRW-BTC", "KRW-EUL", "KRW-EUL"]


def test_partial_ticker_batch_rechecks_omitted_market_before_exclusion(
    monkeypatch,
):
    calls: list[tuple[str, ...]] = []

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    def fake_get(_url, *, params, **_kwargs):
        batch = tuple(params["markets"].split(","))
        calls.append(batch)
        if len(batch) > 1:
            payload = [{"market": "KRW-BTC", "trade_price": 100.0}]
        else:
            payload = [{"market": batch[0], "trade_price": 50.0}]
        return FakeResponse(payload)

    monkeypatch.setattr(collector_d1.requests, "get", fake_get)
    monkeypatch.setattr(collector_d1.time, "sleep", lambda *_args: None)

    assert collector_d1._markets_with_ticker_snapshot(
        ["KRW-BTC", "KRW-ETH"]
    ) == ["KRW-BTC", "KRW-ETH"]
    assert calls == [
        ("KRW-BTC", "KRW-ETH"),
        ("KRW-ETH",),
    ]


def test_ticker_snapshot_retries_transport_failure_with_backoff(monkeypatch):
    calls = 0
    sleeps: list[float] = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return [{"market": "KRW-BTC", "trade_price": 100.0}]

    def fake_get(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise collector_d1.requests.ConnectionError("transient")
        return FakeResponse()

    monkeypatch.setattr(collector_d1.requests, "get", fake_get)
    monkeypatch.setattr(
        collector_d1.time,
        "sleep",
        lambda seconds: sleeps.append(seconds),
    )

    assert collector_d1._markets_with_ticker_snapshot(["KRW-BTC"]) == [
        "KRW-BTC"
    ]
    assert calls == 3
    assert sleeps == [
        collector_d1.RETRY_BACKOFF,
        collector_d1.RETRY_BACKOFF**2,
    ]


def test_ticker_snapshot_rejects_cross_batch_identity(monkeypatch):
    markets = [f"KRW-X{i}" for i in range(101)]

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    responses = iter(
        [
            # KRW-X100 belongs to the next request and must not be accepted
            # merely because it exists in the global requested universe.
            FakeResponse([{"market": "KRW-X100", "trade_price": 1.0}]),
        ]
    )
    monkeypatch.setattr(
        collector_d1.requests,
        "get",
        lambda *_args, **_kwargs: next(responses),
    )

    with pytest.raises(RuntimeError, match="identity violation"):
        collector_d1._markets_with_ticker_snapshot(markets)


def test_ticker_404_batch_is_rate_limited_and_split_to_filter_pending_pair(
    monkeypatch,
):
    calls: list[tuple[str, ...]] = []
    sleeps: list[float] = []

    class FakeResponse:
        def __init__(self, batch):
            self.batch = tuple(batch)
            self.status_code = (
                404 if "KRW-PENDING" in self.batch else 200
            )

        def raise_for_status(self):
            if self.status_code == 404:
                raise collector_d1.requests.HTTPError(
                    "404 pending ticker",
                    response=self,
                )

        def json(self):
            return [
                {"market": market, "trade_price": 100.0}
                for market in self.batch
            ]

    def fake_get(_url, *, params, **_kwargs):
        batch = tuple(params["markets"].split(","))
        calls.append(batch)
        return FakeResponse(batch)

    monkeypatch.setattr(collector_d1.requests, "get", fake_get)
    monkeypatch.setattr(
        collector_d1.time,
        "sleep",
        lambda seconds: sleeps.append(seconds),
    )

    active = collector_d1._markets_with_ticker_snapshot(
        ["KRW-BTC", "KRW-PENDING"]
    )

    assert active == ["KRW-BTC"]
    assert calls.count(("KRW-BTC", "KRW-PENDING")) == (
        collector_d1.RETRY_MAX
    )
    assert calls.count(("KRW-PENDING",)) == collector_d1.RETRY_MAX
    assert calls.count(("KRW-BTC",)) == 1
    assert collector_d1.SLEEP_BETWEEN_TICKER_FALLBACKS in sleeps


def test_ticker_429_never_shrinks_live_universe(monkeypatch):
    class FakeResponse:
        status_code = 429

        def raise_for_status(self):
            raise collector_d1.requests.HTTPError(
                "rate limited",
                response=self,
            )

        def json(self):
            raise AssertionError("429 body must not be accepted")

    monkeypatch.setattr(
        collector_d1.requests,
        "get",
        lambda *_args, **_kwargs: FakeResponse(),
    )
    monkeypatch.setattr(collector_d1.time, "sleep", lambda *_args: None)

    with pytest.raises(RuntimeError, match="failed after retries"):
        collector_d1._markets_with_ticker_snapshot(
            ["KRW-BTC", "KRW-PENDING"]
        )


@pytest.mark.parametrize(
    "module",
    [collector_d1, collector_4h, collector_15m_upbit],
)
@pytest.mark.parametrize("failure_mode", ["raise", "none"])
def test_fetch_retry_exhaustion_is_not_treated_as_normal_empty_page(
    monkeypatch, module, failure_mode
):
    calls = 0

    def fail_fetch(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if failure_mode == "none":
            return None
        raise RuntimeError("network")

    monkeypatch.setattr(module.pyupbit, "get_ohlcv", fail_fetch)
    monkeypatch.setattr(
        module,
        "_is_confirmed_empty_page",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(module.time, "sleep", lambda *_args: None)

    with pytest.raises(collector_d1.FetchPageError, match="retries failed"):
        module._fetch_page("KRW-BTC", collector_d1.datetime.now())
    assert calls == module.RETRY_MAX


@pytest.mark.parametrize(
    "module",
    [collector_d1, collector_4h, collector_15m_upbit],
)
def test_pyupbit_none_is_terminal_only_when_rest_confirms_empty(
    monkeypatch, module
):
    calls = 0

    def none_fetch(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return None

    monkeypatch.setattr(module.pyupbit, "get_ohlcv", none_fetch)
    monkeypatch.setattr(
        module,
        "_is_confirmed_empty_page",
        lambda *_args, **_kwargs: True,
    )

    assert module._fetch_page("KRW-PENDING", datetime.now()) is None
    assert calls == 1


def test_raw_empty_probe_accepts_only_http_200_empty_list(monkeypatch):
    class FakeResponse:
        def __init__(self, payload, *, fail=False):
            self.payload = payload
            self.fail = fail

        def raise_for_status(self):
            if self.fail:
                raise collector_d1.requests.HTTPError("503")

        def json(self):
            return self.payload

    now = datetime(2026, 7, 26, 10, 5)
    monkeypatch.setattr(
        collector_d1.requests,
        "get",
        lambda *_args, **_kwargs: FakeResponse([]),
    )
    assert collector_d1._is_confirmed_empty_page(
        "KRW-PENDING", now, 200, "minute15"
    )

    monkeypatch.setattr(
        collector_d1.requests,
        "get",
        lambda *_args, **_kwargs: FakeResponse([{"market": "KRW-BTC"}]),
    )
    assert not collector_d1._is_confirmed_empty_page(
        "KRW-BTC", now, 200, "minute15"
    )

    monkeypatch.setattr(
        collector_d1.requests,
        "get",
        lambda *_args, **_kwargs: FakeResponse([], fail=True),
    )
    assert not collector_d1._is_confirmed_empty_page(
        "KRW-BTC", now, 200, "minute15"
    )


@pytest.mark.parametrize(
    "module",
    [collector_d1, collector_4h, collector_15m_upbit],
)
def test_upbit_fetch_converts_kst_wall_clock_to_utc_api_boundary(
    monkeypatch,
    module,
):
    observed: list[str] = []
    frame = pd.DataFrame(
        {"close": [1.0]},
        index=pd.DatetimeIndex([datetime(2026, 7, 26, 9, 0)]),
    )

    def fake_get_ohlcv(*_args, **kwargs):
        observed.append(kwargs["to"])
        return frame

    monkeypatch.setattr(module.pyupbit, "get_ohlcv", fake_get_ohlcv)

    assert module._fetch_page(
        "KRW-BTC",
        datetime(2026, 7, 26, 9, 5),
    ) is frame
    assert observed == ["20260726 000500"]


def test_raw_empty_probe_uses_same_utc_boundary_as_pyupbit(monkeypatch):
    observed: list[str] = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return []

    def fake_get(*_args, **kwargs):
        observed.append(kwargs["params"]["to"])
        return FakeResponse()

    monkeypatch.setattr(collector_d1.requests, "get", fake_get)

    assert collector_d1._is_confirmed_empty_page(
        "KRW-PENDING",
        datetime(2026, 7, 26, 9, 5),
        200,
        "day",
    )
    assert observed == ["2026-07-26 00:05:00"]


@pytest.mark.parametrize(
    "module",
    [collector_d1, collector_4h, collector_15m_upbit],
)
def test_fetch_empty_dataframe_remains_normal_terminal_page(monkeypatch, module):
    calls = 0

    def empty_fetch(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return pd.DataFrame()

    monkeypatch.setattr(module.pyupbit, "get_ohlcv", empty_fetch)
    monkeypatch.setattr(
        module,
        "_is_confirmed_empty_page",
        lambda *_args, **_kwargs: True,
    )

    assert module._fetch_page("KRW-BTC", collector_d1.datetime.now()) is None
    assert calls == 1


def test_top_market_snapshot_failure_is_not_empty_success(monkeypatch):
    monkeypatch.setattr(collector_d1, "get_krw_markets", lambda: ["KRW-BTC"])
    monkeypatch.setattr(collector_d1, "_fetch_page", lambda *_args, **_kwargs: None)

    with pytest.raises(collector_d1.FetchPageError, match="no eligible"):
        collector_d1.get_top_markets(1)


def test_top_market_uses_last_completed_kst_day_not_partial_current_day(
    monkeypatch,
):
    now = datetime(2026, 7, 26, 9, 5)
    index = pd.DatetimeIndex(
        [
            datetime(2026, 7, 24, 9),
            datetime(2026, 7, 25, 9),
            datetime(2026, 7, 26, 9),  # 현재 진행 중 D1 candle
        ]
    )
    frames = {
        # A는 현재 부분봉만 비정상적으로 크다.
        "KRW-A": pd.DataFrame({"value": [10.0, 20.0, 1000.0]}, index=index),
        # B가 마지막 완결 일봉의 실제 top이다.
        "KRW-B": pd.DataFrame({"value": [10.0, 100.0, 1.0]}, index=index),
    }
    monkeypatch.setattr(
        collector_d1, "get_krw_markets", lambda: ["KRW-A", "KRW-B"]
    )
    monkeypatch.setattr(
        collector_d1,
        "_fetch_page",
        lambda market, *_args, **_kwargs: frames[market],
    )
    sleeps: list[float] = []
    monkeypatch.setattr(
        collector_d1.time,
        "sleep",
        lambda seconds: sleeps.append(seconds),
    )

    assert collector_d1.get_top_markets(1, now=now) == ["KRW-B"]
    assert sleeps == [
        collector_d1.SLEEP_BETWEEN_TOP_MARKETS,
        collector_d1.SLEEP_BETWEEN_TOP_MARKETS,
    ]


def test_top_market_skips_new_pair_without_completed_daily_candle(monkeypatch):
    now = datetime(2026, 7, 26, 9, 5)
    frames = {
        "KRW-BTC": pd.DataFrame(
            {"value": [100.0]},
            index=pd.DatetimeIndex([datetime(2026, 7, 25, 9)]),
        ),
        "KRW-NEW": pd.DataFrame(
            {"value": [1000.0]},
            index=pd.DatetimeIndex([datetime(2026, 7, 26, 9)]),
        ),
    }
    monkeypatch.setattr(
        collector_d1,
        "get_krw_markets",
        lambda: ["KRW-BTC", "KRW-NEW"],
    )
    monkeypatch.setattr(
        collector_d1,
        "_fetch_page",
        lambda market, *_args, **_kwargs: frames[market],
    )
    monkeypatch.setattr(collector_d1.time, "sleep", lambda *_args: None)

    assert collector_d1.get_top_markets(2, now=now) == ["KRW-BTC"]


def test_collect_market_treats_empty_recent_page_as_live_failure(
    monkeypatch, tmp_path
):
    now = datetime(2026, 7, 26, 9, 5)
    monkeypatch.setattr(collector_d1, "_now_kst_naive", lambda: now)
    monkeypatch.setattr(collector_d1, "init_db", lambda _db_path: None)
    monkeypatch.setattr(
        collector_d1,
        "latest_timestamp",
        lambda _db_path, _market: pd.Timestamp("2026-07-25 09:00:00"),
    )
    monkeypatch.setattr(
        collector_d1,
        "oldest_timestamp",
        lambda _db_path, _market: pd.Timestamp("2024-01-01 09:00:00"),
    )
    monkeypatch.setattr(
        collector_d1, "_fetch_page", lambda *_args, **_kwargs: None
    )

    with pytest.raises(collector_d1.FetchPageError, match="empty recent D1"):
        collector_d1.collect_market(
            "KRW-BTC", days=365, db_path=tmp_path / "upbit_d1.db"
        )


def test_d1_collect_market_rejects_nonpositive_days_before_io(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        collector_d1,
        "init_db",
        lambda _db_path: pytest.fail("must reject before DB mutation"),
    )

    with pytest.raises(ValueError, match="days must be positive"):
        collector_d1.collect_market(
            "KRW-BTC",
            days=0,
            db_path=tmp_path / "upbit_d1.db",
        )


@pytest.mark.parametrize(
    ("module", "interval"),
    [
        (collector_4h, "4h"),
        (collector_15m_upbit, "15m"),
    ],
)
def test_intraday_collectors_treat_empty_recent_page_as_live_failure(
    monkeypatch, tmp_path, module, interval
):
    now = datetime(2026, 7, 26, 9, 5)
    monkeypatch.setattr(module, "_now_kst_naive", lambda: now)
    monkeypatch.setattr(module, "init_db", lambda _db_path: None)
    monkeypatch.setattr(
        module,
        "latest_timestamp",
        lambda _db_path, _market: pd.Timestamp("2026-07-25 09:00:00"),
    )
    monkeypatch.setattr(module, "_fetch_page", lambda *_args, **_kwargs: None)

    with pytest.raises(
        collector_d1.FetchPageError,
        match=f"empty recent {interval}",
    ):
        module.collect_market(
            "KRW-BTC", days=365, db_path=tmp_path / f"upbit_{interval}.db"
        )


@pytest.mark.parametrize("module", [collector_4h, collector_15m_upbit])
def test_intraday_cli_collection_modes_are_mutually_exclusive(monkeypatch, module):
    monkeypatch.setattr(
        sys,
        "argv",
        [module.__name__, "--all", "--coin", "KRW-BTC"],
    )

    with pytest.raises(SystemExit) as exc:
        module.main()

    assert exc.value.code == 2


def test_cli_collection_modes_are_mutually_exclusive(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["collector_d1", "--all", "--update"],
    )

    with pytest.raises(SystemExit) as exc:
        collector_d1.main()

    assert exc.value.code == 2

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from data.database import connect_readonly
from ledger.path_quality import (
    EXPECTED_BARS,
    FIRST_HOUR_EXPECTED_BARS,
    FOUR_HOUR_EXPECTED_BARS,
    PathDataError,
    assess_15m_path,
    assess_15m_window,
    assess_15m_windows,
    assess_4h_path,
    assess_first_hour_path,
    next_bar_boundary,
)


SCHEMA = """
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


def _grid() -> pd.DatetimeIndex:
    return pd.date_range("2026-07-01 09:00:00", periods=EXPECTED_BARS, freq="15min")


@pytest.mark.parametrize(
    ("observed", "expected"),
    [
        ("2026-07-01 09:10:00+09:00", "2026-07-01 09:15:00+09:00"),
        ("2026-07-01 09:15:00+09:00", "2026-07-01 09:30:00+09:00"),
        ("2026-07-01 09:15:00.000001+09:00", "2026-07-01 09:30:00+09:00"),
    ],
)
def test_next_bar_boundary_is_strictly_after_observation(
    observed,
    expected,
):
    assert next_bar_boundary(observed) == pd.Timestamp(expected)


def _with_next_boundary(grid: pd.DatetimeIndex) -> pd.DatetimeIndex:
    freq = grid[1] - grid[0]
    return grid.append(pd.DatetimeIndex([grid[-1] + freq]))


def _make_db(tmp_path, rows):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "m15.db"
    with sqlite3.connect(path) as conn:
        conn.execute(SCHEMA)
        conn.executemany(
            "INSERT INTO candles VALUES (?,?,?,?,?,?)",
            rows,
        )
    return path


def _rows(market, timestamps, *, price=100.0):
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


def test_missing_database_is_read_only_and_never_created(tmp_path):
    missing = tmp_path / "missing" / "m15.db"

    with pytest.raises(PathDataError, match="read-only mode"):
        assess_15m_path("KRW-ALT", "2026-07-01", db_path=missing)

    assert not missing.exists()
    assert not missing.parent.exists()


def test_path_database_never_follows_a_final_component_symlink(tmp_path):
    target = _make_db(
        tmp_path / "target",
        _rows("KRW-BTC", _with_next_boundary(_grid()), price=200.0)
        + _rows("KRW-ALT", _grid()),
    )
    link = tmp_path / "linked.db"
    link.symlink_to(target)

    with pytest.raises(PathDataError, match="read-only mode"):
        assess_15m_path("KRW-ALT", "2026-07-01", db_path=link)


def test_invalid_database_schema_fails_closed(tmp_path):
    db = tmp_path / "invalid.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE unrelated (value TEXT)")

    with pytest.raises(PathDataError, match="read-only mode"):
        assess_15m_path("KRW-ALT", "2026-07-01", db_path=db)


def test_complete_path_keeps_all_raw_bars(tmp_path):
    grid = _grid()
    db = _make_db(
        tmp_path,
        _rows("KRW-BTC", _with_next_boundary(grid), price=200.0)
        + _rows("KRW-ALT", grid),
    )

    result = assess_15m_path("KRW-ALT", "2026-07-01", db_path=db)

    assert result.path_complete is True
    assert result.path_quality == "complete"
    assert result.reason == "complete"
    assert result.raw_bars == result.expected_bars == EXPECTED_BARS
    assert result.benchmark_bars == EXPECTED_BARS
    assert result.flat_filled_bars == 0
    assert len(result.bars) == len(result.timestamps) == EXPECTED_BARS

    aware = assess_15m_path(
        "KRW-ALT",
        pd.Timestamp("2026-07-01T00:00:00+00:00"),
        db_path=db,
    )
    assert aware.path_complete is True
    assert aware.timestamps[0] == pd.Timestamp("2026-07-01 09:00:00")


def test_execution_window_keeps_full_96_bars_after_0915(tmp_path):
    grid = pd.date_range(
        "2026-07-01 09:15:00",
        periods=EXPECTED_BARS,
        freq="15min",
    )
    db = _make_db(
        tmp_path,
        _rows("KRW-BTC", _with_next_boundary(grid), price=200.0)
        + _rows("KRW-ALT", grid),
    )

    result = assess_15m_window(
        "KRW-ALT",
        pd.Timestamp("2026-07-01T09:15:00+09:00"),
        db_path=db,
    )

    assert result.path_complete is True
    assert result.expected_bars == EXPECTED_BARS
    assert len(result.bars) == len(result.timestamps) == EXPECTED_BARS
    assert result.timestamps[0] == pd.Timestamp("2026-07-01 09:15:00")
    assert result.timestamps[-1] == pd.Timestamp("2026-07-02 09:00:00")


def test_execution_window_reuses_caller_owned_readonly_connection(tmp_path):
    grid = pd.date_range(
        "2026-07-01 09:15:00",
        periods=EXPECTED_BARS,
        freq="15min",
    )
    db = _make_db(
        tmp_path,
        _rows("KRW-BTC", _with_next_boundary(grid), price=200.0)
        + _rows("KRW-ALT", grid),
    )

    with connect_readonly(db) as connection:
        first = assess_15m_window(
            "KRW-ALT",
            "2026-07-01 09:15:00",
            connection=connection,
        )
        second = assess_15m_window(
            "KRW-ALT",
            "2026-07-01 09:15:00",
            connection=connection,
        )
        assert connection.execute("SELECT 1").fetchone() == (1,)

    assert first == second
    assert first.path_complete is True


def test_downside_loader_forwards_the_existing_connection(monkeypatch):
    from types import SimpleNamespace

    import scripts.downside_aware_recommender_v1 as downside

    connection = object()
    bars = [(100.0, 101.0, 99.0, 100.0)]
    observed = {}

    def assess(market, start_at, *, connection):
        observed.update(
            {
                "market": market,
                "start_at": start_at,
                "connection": connection,
            }
        )
        return SimpleNamespace(path_complete=True, bars=bars)

    monkeypatch.setattr(downside, "assess_15m_window", assess)

    assert downside.load_15m_window(
        connection,
        "KRW-ALT",
        "2026-07-01",
    ) == bars
    assert observed == {
        "market": "KRW-ALT",
        "start_at": pd.Timestamp("2026-07-01 09:15:00"),
        "connection": connection,
    }


def test_shared_path_loader_uses_bulk_assessments(monkeypatch):
    from types import SimpleNamespace

    import scripts.recommender_downside_exit_v1 as downside_exit

    observed = {}
    complete_bars = [(100.0, 101.0, 99.0, 100.0)]

    def assess(pairs, *, db_path):
        observed["pairs"] = pairs.copy()
        observed["db_path"] = db_path
        return {
            ("KRW-BTC", pd.Timestamp("2026-07-01").date()): SimpleNamespace(
                path_complete=True,
                bars=complete_bars,
            ),
            ("KRW-ETH", pd.Timestamp("2026-07-02").date()): SimpleNamespace(
                path_complete=False,
                bars=[],
            ),
        }

    monkeypatch.setattr(downside_exit, "assess_15m_windows", assess)
    pairs = pd.DataFrame(
        {
            "market": ["KRW-BTC", "KRW-ETH"],
            "date": ["2026-07-01", "2026-07-02"],
        }
    )

    paths = downside_exit.load_paths(pairs, db_path="paths.db")

    pd.testing.assert_frame_equal(observed["pairs"], pairs)
    assert observed["db_path"] == "paths.db"
    assert paths == {
        ("KRW-BTC", pd.Timestamp("2026-07-01").date()): complete_bars
    }


def test_realized_path_loader_batches_dates_and_discards_raw_paths(monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace

    import scripts.r2_challenger_compare_v1 as r2

    connection = object()
    opened = []
    observed_dates = []
    complete_bars = [(100.0, 101.0, 99.0, 100.0)]

    @contextmanager
    def connect(path):
        opened.append(path)
        yield connection

    def assess(pairs, *, connection):
        assert connection is not None
        dates = tuple(sorted(pairs["date"].unique()))
        observed_dates.append(dates)
        return {
            (
                row.market,
                pd.Timestamp(row.date).date(),
            ): SimpleNamespace(
                path_complete=row.market != "KRW-INCOMPLETE",
                bars=complete_bars,
            )
            for row in pairs.itertuples(index=False)
        }

    monkeypatch.setattr(r2, "connect_readonly", connect)
    monkeypatch.setattr(r2, "assess_15m_windows", assess)
    monkeypatch.setattr(
        r2,
        "realize_net",
        lambda bars: (bars[0][0] / 100 - 1, "eod", 0.0),
    )
    pairs = pd.DataFrame(
        {
            "market": [
                "KRW-A",
                "KRW-INCOMPLETE",
                "KRW-B",
                "KRW-C",
            ],
            "date": [
                "2026-07-01",
                "2026-07-01",
                "2026-07-02",
                "2026-07-03",
            ],
        }
    )

    realized = r2.load_realized_paths(
        pairs,
        db_path="paths.db",
        date_batch_size=2,
    )

    assert opened == ["paths.db"]
    assert observed_dates == [
        ("2026-07-01", "2026-07-02"),
        ("2026-07-03",),
    ]
    assert set(realized) == {
        ("KRW-A", pd.Timestamp("2026-07-01").date()),
        ("KRW-B", pd.Timestamp("2026-07-02").date()),
        ("KRW-C", pd.Timestamp("2026-07-03").date()),
    }


def test_bulk_execution_windows_match_canonical_single_assessments(tmp_path):
    grid = pd.date_range(
        "2026-07-01 09:15:00",
        periods=EXPECTED_BARS,
        freq="15min",
    )
    prior = grid[0] - pd.Timedelta(minutes=15)
    complete = _rows("KRW-COMPLETE", grid)
    internal_gap = _rows("KRW-GAP", grid.delete(17), price=110.0)
    opening_gap = _rows("KRW-OPEN-GAP", [prior], price=99.0)
    opening_gap += _rows("KRW-OPEN-GAP", grid[3:], price=120.0)
    no_prior = _rows("KRW-NO-PRIOR", grid[1:], price=130.0)
    off_grid = _rows("KRW-OFF-GRID", grid, price=140.0)
    off_grid += _rows(
        "KRW-OFF-GRID",
        [grid[10] + pd.Timedelta(minutes=1)],
        price=140.0,
    )
    db = _make_db(
        tmp_path,
        _rows("KRW-BTC", _with_next_boundary(grid), price=200.0)
        + complete
        + internal_gap
        + opening_gap
        + no_prior
        + off_grid,
    )
    markets = [
        "KRW-COMPLETE",
        "KRW-GAP",
        "KRW-OPEN-GAP",
        "KRW-NO-PRIOR",
        "KRW-NO-OBSERVATIONS",
        "KRW-OFF-GRID",
    ]
    pairs = pd.DataFrame(
        {
            "market": markets + ["KRW-COMPLETE"],
            "date": ["2026-07-01"] * (len(markets) + 1),
        }
    )

    bulk = assess_15m_windows(pairs, db_path=db)

    assert len(bulk) == len(markets)
    for market in markets:
        single = assess_15m_window(
            market,
            "2026-07-01 09:15:00",
            db_path=db,
        )
        assert bulk[(market, pd.Timestamp("2026-07-01").date())] == single


def test_bulk_execution_windows_match_incomplete_benchmark_contract(tmp_path):
    grid = pd.date_range(
        "2026-07-01 09:15:00",
        periods=EXPECTED_BARS,
        freq="15min",
    )
    db = _make_db(
        tmp_path,
        _rows(
            "KRW-BTC",
            _with_next_boundary(grid).delete(20),
            price=200.0,
        )
        + _rows("KRW-ALT", grid),
    )
    pairs = pd.DataFrame(
        {"market": ["KRW-ALT"], "date": ["2026-07-01"]}
    )

    bulk = assess_15m_windows(pairs, db_path=db)
    single = assess_15m_window(
        "KRW-ALT",
        "2026-07-01 09:15:00",
        db_path=db,
    )

    assert bulk[("KRW-ALT", pd.Timestamp("2026-07-01").date())] == single
    assert single.path_quality == "benchmark_gap"


def test_bulk_query_count_scales_with_dates_not_pairs(tmp_path):
    first_grid = pd.date_range(
        "2026-07-01 09:15:00",
        periods=EXPECTED_BARS,
        freq="15min",
    )
    second_grid = pd.date_range(
        "2026-07-02 09:15:00",
        periods=EXPECTED_BARS,
        freq="15min",
    )
    markets = [f"KRW-ALT-{i:02d}" for i in range(20)]
    rows = _rows(
        "KRW-BTC",
        first_grid.append(second_grid).append(
            pd.DatetimeIndex(
                [second_grid[-1] + pd.Timedelta(minutes=15)]
            )
        ),
        price=200.0,
    )
    for market in markets:
        rows += _rows(market, first_grid)
        rows += _rows(market, second_grid)
    db = _make_db(tmp_path, rows)
    pairs = pd.DataFrame(
        {
            "market": markets + markets,
            "date": ["2026-07-01"] * len(markets)
            + ["2026-07-02"] * len(markets),
        }
    )
    statements = []

    with connect_readonly(db) as connection:
        connection.set_trace_callback(statements.append)
        results = assess_15m_windows(pairs, connection=connection)

    data_queries = [
        statement
        for statement in statements
        if statement.lstrip().upper().startswith(("SELECT", "WITH"))
    ]
    assert len(results) == 2 * len(markets)
    assert len(data_queries) == 3  # one horizon + one bulk query per date


def test_execution_window_rejects_writable_caller_connection(tmp_path):
    grid = pd.date_range(
        "2026-07-01 09:15:00",
        periods=EXPECTED_BARS,
        freq="15min",
    )
    db = _make_db(
        tmp_path,
        _rows("KRW-BTC", _with_next_boundary(grid), price=200.0)
        + _rows("KRW-ALT", grid),
    )

    with sqlite3.connect(db) as connection:
        with pytest.raises(PathDataError, match="must be query-only"):
            assess_15m_window(
                "KRW-ALT",
                "2026-07-01 09:15:00",
                connection=connection,
            )


def test_execution_window_rejects_off_grid_start(tmp_path):
    grid = _grid()
    db = _make_db(
        tmp_path,
        _rows("KRW-BTC", _with_next_boundary(grid), price=200.0)
        + _rows("KRW-ALT", grid),
    )

    with pytest.raises(ValueError, match="start_at must be aligned"):
        assess_15m_window(
            "KRW-ALT",
            "2026-07-01 09:10:00",
            db_path=db,
        )


def test_target_only_gap_is_flat_filled_from_previous_close(tmp_path):
    grid = _grid()
    missing = grid[17]
    target_grid = grid.delete(17)
    target = _rows("KRW-ALT", target_grid)
    # gap 직전 close를 눈에 띄는 값으로 바꿔 synthetic OHLC가 이를 잇는지 고정한다.
    prior_idx = 16
    target[prior_idx] = (
        "KRW-ALT",
        grid[prior_idx].strftime("%Y-%m-%d %H:%M:%S"),
        100.0,
        123.0,
        99.0,
        123.0,
    )
    db = _make_db(
        tmp_path,
        _rows("KRW-BTC", _with_next_boundary(grid), price=200.0) + target,
    )

    result = assess_15m_path("KRW-ALT", "2026-07-01", db_path=db)

    assert result.path_complete is True
    assert result.path_quality == "flat_filled"
    assert result.raw_bars == 95
    assert result.flat_filled_bars == 1
    assert result.timestamps[17] == missing
    assert result.bars[17] == (123.0, 123.0, 123.0, 123.0)


def test_opening_and_ending_no_trade_gaps_are_flat_filled_when_btc_is_complete(tmp_path):
    grid = _grid()
    prior = pd.Timestamp("2026-07-01 08:45:00")
    target_grid = grid[3:90]
    rows = _rows("KRW-BTC", _with_next_boundary(grid), price=200.0)
    rows += _rows("KRW-ALT", [prior], price=99.0)
    rows += _rows("KRW-ALT", target_grid, price=100.0)
    db = _make_db(tmp_path, rows)

    result = assess_15m_path("KRW-ALT", "2026-07-01", db_path=db)

    assert result.path_complete is True
    assert result.path_quality == "flat_filled"
    assert result.raw_bars == len(target_grid)
    assert result.flat_filled_bars == EXPECTED_BARS - len(target_grid)
    assert result.bars[0] == (99.0, 99.0, 99.0, 99.0)
    assert result.bars[2] == (99.0, 99.0, 99.0, 99.0)
    assert result.bars[90] == (100.0, 100.0, 100.0, 100.0)
    assert len(result.bars) == EXPECTED_BARS


def test_benchmark_internal_gap_is_collection_failure_not_flat_fill(tmp_path):
    grid = _grid()
    benchmark = _rows(
        "KRW-BTC",
        _with_next_boundary(grid).delete(20),
        price=200.0,
    )
    db = _make_db(tmp_path, benchmark + _rows("KRW-ALT", grid))

    result = assess_15m_path("KRW-ALT", "2026-07-01", db_path=db)

    assert result.path_complete is False
    assert result.path_quality == "benchmark_gap"
    assert result.benchmark_bars == 95
    assert result.bars == []


def test_benchmark_start_and_end_horizon_incomplete_are_deferred(tmp_path):
    grid = _grid()

    db_start = _make_db(
        tmp_path / "start",
        _rows(
            "KRW-BTC",
            _with_next_boundary(grid)[1:],
            price=200.0,
        )
        + _rows("KRW-ALT", grid),
    )
    start_result = assess_15m_path("KRW-ALT", "2026-07-01", db_path=db_start)
    assert start_result.path_complete is False
    assert start_result.path_quality == "db_horizon_start_incomplete"

    db_end = _make_db(
        tmp_path / "end",
        _rows("KRW-BTC", grid[:-1], price=200.0) + _rows("KRW-ALT", grid),
    )
    end_result = assess_15m_path("KRW-ALT", "2026-07-01", db_path=db_end)
    assert end_result.path_complete is False
    assert end_result.path_quality == "db_horizon_end_incomplete"

    # 96번째 08:45 봉이 있어도 다음 09:00 boundary가 없으면 그 08:45
    # 봉은 진행 중 저장본일 수 있으므로 아직 마감으로 보지 않는다.
    db_unfinalized = _make_db(
        tmp_path / "unfinalized",
        _rows("KRW-BTC", grid, price=200.0) + _rows("KRW-ALT", grid),
    )
    unfinalized = assess_15m_path(
        "KRW-ALT", "2026-07-01", db_path=db_unfinalized
    )
    assert unfinalized.path_complete is False
    assert unfinalized.path_quality == "db_horizon_end_incomplete"


def test_missing_first_target_bar_without_prior_close_is_deferred(tmp_path):
    grid = _grid()
    db = _make_db(
        tmp_path,
        _rows("KRW-BTC", _with_next_boundary(grid), price=200.0)
        + _rows("KRW-ALT", grid[1:]),
    )

    result = assess_15m_path("KRW-ALT", "2026-07-01", db_path=db)

    assert result.path_complete is False
    assert result.path_quality == "market_start_horizon_incomplete"
    assert result.raw_bars == 95
    assert result.bars == []


def test_prior_close_cannot_turn_full_target_outage_into_flat_path(tmp_path):
    grid = _grid()
    prior = pd.Timestamp("2026-07-01 08:45:00")
    db = _make_db(
        tmp_path,
        _rows("KRW-BTC", _with_next_boundary(grid), price=200.0)
        + _rows("KRW-ALT", [prior], price=99.0),
    )

    result = assess_15m_path("KRW-ALT", "2026-07-01", db_path=db)

    assert result.path_complete is False
    assert result.path_quality == "target_no_observations"
    assert result.raw_bars == 0
    assert result.bars == []


def test_invalid_target_ohlc_is_never_silently_flat_filled(tmp_path):
    grid = _grid()
    target = _rows("KRW-ALT", grid)
    target[5] = (
        "KRW-ALT",
        grid[5].strftime("%Y-%m-%d %H:%M:%S"),
        100.0,
        90.0,
        95.0,
        100.0,
    )
    db = _make_db(
        tmp_path,
        _rows("KRW-BTC", _with_next_boundary(grid), price=200.0) + target,
    )

    result = assess_15m_path("KRW-ALT", "2026-07-01", db_path=db)

    assert result.path_complete is False
    assert result.path_quality == "invalid_target_ohlc"
    assert result.bars == []


def test_non_numeric_target_ohlc_is_incomplete_not_an_uncaught_parse_error(
    tmp_path,
):
    grid = _grid()
    target = _rows("KRW-ALT", grid)
    target[5] = (
        "KRW-ALT",
        grid[5].strftime("%Y-%m-%d %H:%M:%S"),
        "not-a-price",
        100.0,
        100.0,
        100.0,
    )
    db = _make_db(
        tmp_path,
        _rows("KRW-BTC", _with_next_boundary(grid), price=200.0) + target,
    )

    result = assess_15m_path("KRW-ALT", "2026-07-01", db_path=db)

    assert result.path_complete is False
    assert result.path_quality == "invalid_target_ohlc"
    assert result.bars == []


def test_4h_and_first_hour_use_reference_complete_grid_with_target_flat_fill(
    tmp_path,
):
    grid_4h = pd.date_range(
        "2026-07-01 09:00:00",
        periods=FOUR_HOUR_EXPECTED_BARS,
        freq="4h",
    )
    db_4h = _make_db(
        tmp_path / "4h",
        _rows("KRW-BTC", _with_next_boundary(grid_4h), price=200.0)
        + _rows("KRW-ALT", grid_4h.delete(2)),
    )
    four_hour = assess_4h_path("KRW-ALT", "2026-07-01", db_path=db_4h)
    assert four_hour.path_complete is True
    assert four_hour.expected_bars == FOUR_HOUR_EXPECTED_BARS
    assert four_hour.raw_bars == 5
    assert four_hour.flat_filled_bars == 1
    assert len(four_hour.bars) == FOUR_HOUR_EXPECTED_BARS

    grid_1h = pd.date_range(
        "2026-07-01 09:00:00",
        periods=FIRST_HOUR_EXPECTED_BARS,
        freq="15min",
    )
    db_1h = _make_db(
        tmp_path / "1h",
        _rows("KRW-BTC", _with_next_boundary(grid_1h), price=200.0)
        + _rows("KRW-ALT", grid_1h.delete(1)),
    )
    first_hour = assess_first_hour_path(
        "KRW-ALT", "2026-07-01", db_path=db_1h
    )
    assert first_hour.path_complete is True
    assert first_hour.expected_bars == FIRST_HOUR_EXPECTED_BARS
    assert first_hour.flat_filled_bars == 1


def test_4h_reference_gap_is_never_closed(tmp_path):
    grid = pd.date_range(
        "2026-07-01 09:00:00",
        periods=FOUR_HOUR_EXPECTED_BARS,
        freq="4h",
    )
    db = _make_db(
        tmp_path,
        _rows(
            "KRW-BTC",
            _with_next_boundary(grid).delete(3),
            price=200.0,
        )
        + _rows("KRW-ALT", grid),
    )

    result = assess_4h_path("KRW-ALT", "2026-07-01", db_path=db)

    assert result.path_complete is False
    assert result.path_quality == "benchmark_gap"
    assert result.bars == []

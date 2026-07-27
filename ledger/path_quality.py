"""15분 경로 완결성 판정과 무체결 봉의 보수적 flat 재구성.

업비트 분봉 API는 거래가 없었던 구간의 봉을 만들지 않는다. 따라서 단순히
``len(bars) == 96`` 을 요구하면 저유동 종목을 체계적으로 제외하게 된다.

이 모듈은 다음 규칙으로 기본 일봉 경로 `[D 09:00, D+1 09:00)` 또는
호출자가 지정한 정렬 시각부터 정확히 24시간인 경로를 만든다.

* KRW-BTC가 96개 기준 timestamp를 모두 가지며 DB horizon도 창 전체를
  덮을 때만 해당 날짜가 수집 완료됐다고 본다.
* 그 상태에서 대상 종목에만 빠진 timestamp는 무체결로 간주하고 직전 close로
  OHLC를 flat-fill한다.
* KRW-BTC 기준봉 누락, DB 시작/종료 horizon 부족, 첫 flat-fill에 필요한 대상
  종목의 직전 close 부재, 비정상 timestamp/OHLC는 incomplete로 반환한다.

청산 여부를 결정하는 모듈일 뿐 신호·주문·텔레그램 동작은 포함하지 않는다.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pandas as pd

from data.database import connect_readonly


BAR_FREQ = pd.Timedelta(minutes=15)
EXPECTED_BARS = 24 * 60 // 15
FOUR_HOUR_FREQ = pd.Timedelta(hours=4)
FOUR_HOUR_EXPECTED_BARS = 6
FIRST_HOUR_EXPECTED_BARS = 4
BENCHMARK_MARKET = "KRW-BTC"

PATH_QUALITY_COLS = [
    "path_complete",
    "path_quality",
    "raw_bars",
    "expected_bars",
    "flat_filled_bars",
    "benchmark_bars",
    "path_start_at",
    "path_used_bars",
]

# 발송 후 실행 가능 경로에만 필요한 추천 원장 전용 감사 컬럼.
EXECUTION_QUALITY_COLS = PATH_QUALITY_COLS + [
    "entry_observable_at",
    "entry_price_source",
]


class PathDataError(RuntimeError):
    """The path database cannot be opened or queried read-only."""


def next_bar_boundary(
    value: pd.Timestamp | str,
    *,
    bar_freq: pd.Timedelta = BAR_FREQ,
) -> pd.Timestamp:
    """Return the first candle boundary strictly after an observed timestamp.

    A receipt recorded exactly on a boundary cannot prove that the user could
    transact at that candle's already-formed opening print.  ``floor + freq``
    therefore stays conservative at exact boundaries while matching ``ceil``
    for timestamps between boundaries.
    """
    if bar_freq <= pd.Timedelta(0):
        raise ValueError("bar_freq must be positive")
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("execution timestamp must not be NaT")
    return timestamp.floor(bar_freq) + bar_freq


@contextmanager
def _path_connection(
    db_path: str | Path | None,
    connection: sqlite3.Connection | None,
) -> Iterator[sqlite3.Connection]:
    """Use one caller-owned query-only connection or open a hardened one."""
    if (db_path is None) == (connection is None):
        raise ValueError("provide exactly one of db_path or connection")
    if connection is not None:
        try:
            query_only = connection.execute("PRAGMA query_only").fetchone()
        except sqlite3.Error as exc:
            raise PathDataError(
                "provided path database connection is unavailable"
            ) from exc
        if query_only != (1,):
            raise PathDataError(
                "provided path database connection must be query-only"
            )
        try:
            yield connection
        except sqlite3.Error as exc:
            raise PathDataError(
                "provided path database connection is unavailable"
            ) from exc
        return

    assert db_path is not None
    try:
        with connect_readonly(db_path) as opened:
            yield opened
    except PathDataError:
        raise
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
        raise PathDataError(
            f"path database is unavailable in read-only mode: {db_path}"
        ) from exc


@dataclass(frozen=True)
class PathAssessment:
    """재구성된 경로와 ledger에 기록할 감사 메타데이터."""

    bars: list[tuple[float, float, float, float]]
    timestamps: tuple[pd.Timestamp, ...]
    path_complete: bool
    path_quality: str
    raw_bars: int
    expected_bars: int = EXPECTED_BARS
    flat_filled_bars: int = 0
    benchmark_bars: int = 0

    @property
    def reason(self) -> str:
        """호출자가 incomplete 사유를 읽을 때 쓰는 ``path_quality`` 별칭."""
        return self.path_quality

    def metadata(self) -> dict:
        return {
            "path_complete": self.path_complete,
            "path_quality": self.path_quality,
            "raw_bars": self.raw_bars,
            "expected_bars": self.expected_bars,
            "flat_filled_bars": self.flat_filled_bars,
            "benchmark_bars": self.benchmark_bars,
        }


def _window(
    date: pd.Timestamp,
    *,
    bar_freq: pd.Timedelta = BAR_FREQ,
    expected_bars: int = EXPECTED_BARS,
) -> tuple[pd.Timestamp, pd.DatetimeIndex]:
    day = pd.Timestamp(date)
    if day.tzinfo is not None:
        day = day.tz_convert("Asia/Seoul").tz_localize(None)
    day = day.normalize()
    start = day + pd.Timedelta(hours=9)
    expected = pd.date_range(start=start, periods=expected_bars, freq=bar_freq)
    return start, expected


def _query_rows(
    conn: sqlite3.Connection,
    market: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[tuple]:
    return conn.execute(
        """
        SELECT timestamp, open, high, low, close
        FROM candles
        WHERE market=? AND timestamp>=? AND timestamp<?
        ORDER BY timestamp
        """,
        (
            market,
            start.strftime("%Y-%m-%d %H:%M:%S"),
            end.strftime("%Y-%m-%d %H:%M:%S"),
        ),
    ).fetchall()


def _horizon(
    conn: sqlite3.Connection,
    market: str,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    row = conn.execute(
        "SELECT MIN(timestamp), MAX(timestamp) FROM candles WHERE market=?",
        (market,),
    ).fetchone()
    if not row or row[0] is None or row[1] is None:
        return None, None
    return pd.Timestamp(row[0]), pd.Timestamp(row[1])


def _prior_close(
    conn: sqlite3.Connection,
    market: str,
    start: pd.Timestamp,
) -> float | None:
    row = conn.execute(
        """
        SELECT close FROM candles
        WHERE market=? AND timestamp<?
        ORDER BY timestamp DESC LIMIT 1
        """,
        (market, start.strftime("%Y-%m-%d %H:%M:%S")),
    ).fetchone()
    if not row:
        return None
    try:
        value = float(row[0])
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) and value > 0 else None


def _incomplete(
    quality: str,
    *,
    timestamps: tuple[pd.Timestamp, ...],
    raw_bars: int,
    benchmark_bars: int,
    expected_bars: int,
) -> PathAssessment:
    return PathAssessment(
        bars=[],
        timestamps=timestamps,
        path_complete=False,
        path_quality=quality,
        raw_bars=raw_bars,
        expected_bars=expected_bars,
        benchmark_bars=benchmark_bars,
    )


def _assess_loaded_path(
    *,
    expected: pd.DatetimeIndex,
    target_rows: list[tuple],
    benchmark_rows: list[tuple],
    benchmark_start: pd.Timestamp | None,
    benchmark_end: pd.Timestamp | None,
    prior_close_getter: Callable[[], float | None],
) -> PathAssessment:
    """Apply the canonical completeness contract to already-loaded rows."""
    expected_bars = len(expected)
    expected_set = set(expected)
    timestamps = tuple(expected)
    end = expected[0] + (expected[1] - expected[0]) * expected_bars
    raw_bars = len(target_rows)
    benchmark_bars = len(benchmark_rows)

    if benchmark_start is None or benchmark_end is None:
        return _incomplete(
            "benchmark_missing",
            timestamps=timestamps,
            raw_bars=raw_bars,
            benchmark_bars=benchmark_bars,
            expected_bars=expected_bars,
        )
    if benchmark_start > expected[0]:
        return _incomplete(
            "db_horizon_start_incomplete",
            timestamps=timestamps,
            raw_bars=raw_bars,
            benchmark_bars=benchmark_bars,
            expected_bars=expected_bars,
        )
    # 마지막 expected timestamp의 봉이 "존재"하는 것만으로는 마감 증거가
    # 아니다. 다음 grid boundary의 봉까지 DB에 보여야 마지막 봉이 끝났음을
    # 확인할 수 있다(예: 08:50에 저장된 진행 중 08:45 15m 봉 방지).
    if benchmark_end < end:
        return _incomplete(
            "db_horizon_end_incomplete",
            timestamps=timestamps,
            raw_bars=raw_bars,
            benchmark_bars=benchmark_bars,
            expected_bars=expected_bars,
        )

    try:
        benchmark_ts = [pd.Timestamp(r[0]) for r in benchmark_rows]
    except (TypeError, ValueError):
        return _incomplete(
            "benchmark_off_grid",
            timestamps=timestamps,
            raw_bars=raw_bars,
            benchmark_bars=benchmark_bars,
            expected_bars=expected_bars,
        )
    if any(ts not in expected_set for ts in benchmark_ts):
        return _incomplete(
            "benchmark_off_grid",
            timestamps=timestamps,
            raw_bars=raw_bars,
            benchmark_bars=benchmark_bars,
            expected_bars=expected_bars,
        )
    if (
        len(benchmark_ts) != expected_bars
        or len(set(benchmark_ts)) != len(benchmark_ts)
        or set(benchmark_ts) != expected_set
    ):
        return _incomplete(
            "benchmark_gap",
            timestamps=timestamps,
            raw_bars=raw_bars,
            benchmark_bars=benchmark_bars,
            expected_bars=expected_bars,
        )

    # A full window with zero target observations is indistinguishable from
    # a per-market collector outage.  A prior close alone is not evidence
    # that the market genuinely had no trades for 24 hours, so fail closed
    # instead of fabricating a zero-return path.
    if raw_bars == 0:
        return _incomplete(
            "target_no_observations",
            timestamps=timestamps,
            raw_bars=raw_bars,
            benchmark_bars=benchmark_bars,
            expected_bars=expected_bars,
        )

    target_by_ts: dict[pd.Timestamp, tuple[float, float, float, float]] = {}
    for timestamp, o, h, low, close in target_rows:
        try:
            ts = pd.Timestamp(timestamp)
        except (TypeError, ValueError):
            return _incomplete(
                "target_off_grid",
                timestamps=timestamps,
                raw_bars=raw_bars,
                benchmark_bars=benchmark_bars,
                expected_bars=expected_bars,
            )
        if ts not in expected_set or ts in target_by_ts:
            return _incomplete(
                "target_off_grid",
                timestamps=timestamps,
                raw_bars=raw_bars,
                benchmark_bars=benchmark_bars,
                expected_bars=expected_bars,
            )
        try:
            bar: tuple[float, float, float, float] = (
                float(o),
                float(h),
                float(low),
                float(close),
            )
        except (TypeError, ValueError):
            return _incomplete(
                "invalid_target_ohlc",
                timestamps=timestamps,
                raw_bars=raw_bars,
                benchmark_bars=benchmark_bars,
                expected_bars=expected_bars,
            )
        if (
            not all(np.isfinite(v) and v > 0 for v in bar)
            or bar[1] < max(bar[0], bar[3])
            or bar[2] > min(bar[0], bar[3])
            or bar[1] < bar[2]
        ):
            return _incomplete(
                "invalid_target_ohlc",
                timestamps=timestamps,
                raw_bars=raw_bars,
                benchmark_bars=benchmark_bars,
                expected_bars=expected_bars,
            )
        target_by_ts[ts] = bar

    previous_close = None
    if expected[0] not in target_by_ts:
        previous_close = prior_close_getter()
        if previous_close is None:
            return _incomplete(
                "market_start_horizon_incomplete",
                timestamps=timestamps,
                raw_bars=raw_bars,
                benchmark_bars=benchmark_bars,
                expected_bars=expected_bars,
            )

    bars: list[tuple[float, float, float, float]] = []
    flat_filled = 0
    for ts in expected:
        existing_bar = target_by_ts.get(ts)
        if existing_bar is None:
            # 첫 timestamp가 비었다면 위에서 직전 close 존재를 확인했다. 이후 gap은
            # 바로 앞 실제/합성 봉 close를 이어 받아 거래 없음(flat)을 표현한다.
            if previous_close is None:
                raise PathDataError(
                    "path reconstruction lost prior close after "
                    "completeness validation"
                )
            bar = (previous_close, previous_close, previous_close, previous_close)
            flat_filled += 1
        else:
            bar = existing_bar
        bars.append(bar)
        previous_close = bar[3]

    quality = "complete" if flat_filled == 0 else "flat_filled"
    return PathAssessment(
        bars=bars,
        timestamps=timestamps,
        path_complete=True,
        path_quality=quality,
        raw_bars=raw_bars,
        expected_bars=expected_bars,
        flat_filled_bars=flat_filled,
        benchmark_bars=benchmark_bars,
    )


def _assess_path(
    market: str,
    date: pd.Timestamp,
    *,
    db_path: str | Path | None,
    connection: sqlite3.Connection | None = None,
    benchmark_market: str = BENCHMARK_MARKET,
    bar_freq: pd.Timedelta,
    expected_bars: int,
    start_at: pd.Timestamp | None = None,
) -> PathAssessment:
    """고정 그리드 경로를 판정하고 정상 무체결 gap만 flat-fill한다.

    반환 ``bars``는 ``path_complete=True``일 때만 ``expected_bars``개다. incomplete 결과는
    빈 ``bars``를 돌려 청산 코드가 실수로 부분 경로를 평가하지 못하게 한다.
    """
    if expected_bars <= 0 or bar_freq <= pd.Timedelta(0):
        raise ValueError("bar_freq and expected_bars must be positive")
    if start_at is None:
        start, expected = _window(
            date,
            bar_freq=bar_freq,
            expected_bars=expected_bars,
        )
    else:
        start = pd.Timestamp(start_at)
        if start.tzinfo is not None:
            start = start.tz_convert("Asia/Seoul").tz_localize(None)
        if start != start.floor(bar_freq):
            raise ValueError(
                f"start_at must be aligned to {bar_freq}: {start_at!r}"
            )
        expected = pd.date_range(
            start=start,
            periods=expected_bars,
            freq=bar_freq,
        )
    end = start + bar_freq * expected_bars

    with _path_connection(db_path, connection) as conn:
        target_rows = _query_rows(conn, market, start, end)
        benchmark_rows = _query_rows(conn, benchmark_market, start, end)
        benchmark_start, benchmark_end = _horizon(conn, benchmark_market)
        return _assess_loaded_path(
            expected=expected,
            target_rows=target_rows,
            benchmark_rows=benchmark_rows,
            benchmark_start=benchmark_start,
            benchmark_end=benchmark_end,
            prior_close_getter=lambda: _prior_close(conn, market, start),
        )


def _bulk_window_rows(
    conn: sqlite3.Connection,
    markets: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[dict[str, list[tuple]], dict[str, float | None]]:
    """Load one date's window and each market's prior close in one query."""
    placeholders = ",".join("(?)" for _ in markets)
    rows = conn.execute(
        f"""
        WITH requested(market) AS (VALUES {placeholders}),
        latest_prior AS (
            SELECT r.market,
                   (
                       SELECT c.timestamp
                       FROM candles AS c
                       WHERE c.market = r.market AND c.timestamp < ?
                       ORDER BY c.timestamp DESC
                       LIMIT 1
                   ) AS timestamp
            FROM requested AS r
        ),
        window_rows AS (
            SELECT c.market, c.timestamp, c.open, c.high, c.low, c.close,
                   0 AS is_prior
            FROM candles AS c
            JOIN requested AS r ON r.market = c.market
            WHERE c.timestamp >= ? AND c.timestamp < ?
        ),
        prior_rows AS (
            SELECT c.market, c.timestamp, c.open, c.high, c.low, c.close,
                   1 AS is_prior
            FROM candles AS c
            JOIN latest_prior AS p
              ON p.market = c.market AND p.timestamp = c.timestamp
        )
        SELECT * FROM window_rows
        UNION ALL
        SELECT * FROM prior_rows
        ORDER BY 1, 2, 7
        """,
        (
            *markets,
            start.strftime("%Y-%m-%d %H:%M:%S"),
            start.strftime("%Y-%m-%d %H:%M:%S"),
            end.strftime("%Y-%m-%d %H:%M:%S"),
        ),
    ).fetchall()

    window_rows: dict[str, list[tuple]] = {market: [] for market in markets}
    prior_closes: dict[str, float | None] = {}
    for market, timestamp, o, h, low, close, is_prior in rows:
        if is_prior:
            try:
                value = float(close)
            except (TypeError, ValueError):
                value = np.nan
            prior_closes[market] = (
                value if np.isfinite(value) and value > 0 else None
            )
        else:
            window_rows[market].append((timestamp, o, h, low, close))
    return window_rows, prior_closes


def assess_15m_windows(
    pairs: pd.DataFrame,
    *,
    db_path: str | Path | None = None,
    connection: sqlite3.Connection | None = None,
    market_col: str = "market",
    date_col: str = "date",
    benchmark_market: str = BENCHMARK_MARKET,
    start_offset: pd.Timedelta = pd.Timedelta(hours=9, minutes=15),
) -> dict[tuple[str, Date], PathAssessment]:
    """Assess many date/market execution windows with one SQL query per date.

    Duplicate pairs are evaluated once. Dates are interpreted as KST calendar
    dates and returned as ``(market, datetime.date)`` keys.
    """
    missing = {market_col, date_col}.difference(pairs.columns)
    if missing:
        raise ValueError(f"pairs missing required columns: {sorted(missing)}")
    if start_offset < pd.Timedelta(0) or start_offset >= pd.Timedelta(days=1):
        raise ValueError("start_offset must be within one calendar day")

    grouped: dict[pd.Timestamp, set[str]] = {}
    for market_value, date_value in pairs[[market_col, date_col]].itertuples(
        index=False,
        name=None,
    ):
        if not isinstance(market_value, str) or not market_value.strip():
            raise ValueError("market values must be non-empty strings")
        day = pd.Timestamp(date_value)
        if pd.isna(day):
            raise ValueError("date values must not be NaT")
        if day.tzinfo is not None:
            day = day.tz_convert("Asia/Seoul").tz_localize(None)
        grouped.setdefault(day.normalize(), set()).add(market_value.strip())

    if not grouped:
        return {}

    results: dict[tuple[str, Date], PathAssessment] = {}
    with _path_connection(db_path, connection) as conn:
        benchmark_start, benchmark_end = _horizon(conn, benchmark_market)
        for day, requested_markets in sorted(grouped.items()):
            start = day + start_offset
            if start != start.floor(BAR_FREQ):
                raise ValueError(
                    f"execution window must be aligned to {BAR_FREQ}: {start}"
                )
            end = start + BAR_FREQ * EXPECTED_BARS
            expected = pd.date_range(
                start=start,
                periods=EXPECTED_BARS,
                freq=BAR_FREQ,
            )
            markets = sorted(requested_markets | {benchmark_market})
            rows_by_market, prior_closes = _bulk_window_rows(
                conn,
                markets,
                start,
                end,
            )
            benchmark_rows = rows_by_market[benchmark_market]
            for market in sorted(requested_markets):
                results[(market, day.date())] = _assess_loaded_path(
                    expected=expected,
                    target_rows=rows_by_market[market],
                    benchmark_rows=benchmark_rows,
                    benchmark_start=benchmark_start,
                    benchmark_end=benchmark_end,
                    prior_close_getter=lambda m=market: prior_closes.get(m),
                )
    return results


def assess_15m_path(
    market: str,
    date: pd.Timestamp,
    *,
    db_path: str | Path,
    benchmark_market: str = BENCHMARK_MARKET,
) -> PathAssessment:
    """[D 09:00, D+1 09:00)의 96개 15분 경로."""
    return _assess_path(
        market,
        date,
        db_path=db_path,
        benchmark_market=benchmark_market,
        bar_freq=BAR_FREQ,
        expected_bars=EXPECTED_BARS,
    )


def assess_15m_window(
    market: str,
    start_at: pd.Timestamp,
    *,
    db_path: str | Path | None = None,
    connection: sqlite3.Connection | None = None,
    benchmark_market: str = BENCHMARK_MARKET,
) -> PathAssessment:
    """``start_at``부터 정확히 24시간인 96개 15분 경로.

    알림 전달 후 첫 실행 가능 봉이 09:15라면 ``[D 09:15, D+1 09:15)``를
    평가한다. ``assess_15m_path``의 KST 09:00 일봉 경로 계약은 그대로 둔다.
    """
    return _assess_path(
        market,
        pd.Timestamp(start_at),
        db_path=db_path,
        connection=connection,
        benchmark_market=benchmark_market,
        bar_freq=BAR_FREQ,
        expected_bars=EXPECTED_BARS,
        start_at=pd.Timestamp(start_at),
    )


def assess_4h_path(
    market: str,
    date: pd.Timestamp,
    *,
    db_path: str | Path,
    benchmark_market: str = BENCHMARK_MARKET,
) -> PathAssessment:
    """[D 09:00, D+1 09:00)의 6개 4시간 경로."""
    return _assess_path(
        market,
        date,
        db_path=db_path,
        benchmark_market=benchmark_market,
        bar_freq=FOUR_HOUR_FREQ,
        expected_bars=FOUR_HOUR_EXPECTED_BARS,
    )


def assess_first_hour_path(
    market: str,
    date: pd.Timestamp,
    *,
    db_path: str | Path,
    benchmark_market: str = BENCHMARK_MARKET,
) -> PathAssessment:
    """[D 09:00, D 10:00)의 4개 15분 경로."""
    return _assess_path(
        market,
        date,
        db_path=db_path,
        benchmark_market=benchmark_market,
        bar_freq=BAR_FREQ,
        expected_bars=FIRST_HOUR_EXPECTED_BARS,
    )

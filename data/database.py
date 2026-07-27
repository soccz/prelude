"""sqlite 헬퍼 — candles 저장 / 조회.

설계: interval 별로 DB 파일 분리 (upbit_d1.db / upbit_4h.db / binance_1h.db).
호출자가 db_path 명시. 한 곳에서 모든 sqlite 로직 관리.

Adapted from: gan_t/data/database.py
Changes:
  - 단순화 (단일 candles 테이블, generic interval)
  - pyupbit DataFrame 직접 받음
  - timestamp는 timezone-naive 거래소 원시 경계
    (Upbit=KST 09:00, Binance=UTC 00:00; 호출자가 거래소별 의미를 보존)
"""
from __future__ import annotations

import os
import sqlite3
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ============================================================================
# 스키마
# ============================================================================
SCHEMA = """
CREATE TABLE IF NOT EXISTS candles (
    market TEXT NOT NULL,
    timestamp DATETIME NOT NULL,    -- timezone-naive exchange timestamp
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,           -- 코인 단위 거래량
    quote_volume REAL,              -- KRW 단위 거래대금 (24h universe 선정용)
    PRIMARY KEY (market, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_candles_market ON candles(market);
CREATE INDEX IF NOT EXISTS idx_candles_timestamp ON candles(timestamp);
"""
_MARKET_ID_PUNCTUATION = frozenset("-._:")
_MAX_MARKET_ID_LENGTH = 128


def _valid_market_identifier(value: object) -> bool:
    """Accept exchange Unicode symbols without accepting ambiguous controls."""
    return (
        isinstance(value, str)
        and 0 < len(value) <= _MAX_MARKET_ID_LENGTH
        and value == value.strip()
        and all(
            char.isalnum() or char in _MARKET_ID_PUNCTUATION
            for char in value
        )
    )


# ============================================================================
# 연결 / 초기화
# ============================================================================
@contextmanager
def connect(db_path: str | Path):
    """실제 regular-file inode에 고정된 writable SQLite 연결."""
    path = Path(os.path.abspath(Path(db_path)))
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    conn: sqlite3.Connection | None = None
    try:
        fd_stat = os.fstat(fd)
        path_stat = path.lstat()
        if (
            not stat.S_ISREG(fd_stat.st_mode)
            or (path_stat.st_dev, path_stat.st_ino)
            != (fd_stat.st_dev, fd_stat.st_ino)
        ):
            raise RuntimeError(f"SQLite DB changed while opening: {path}")

        fd_uri = Path(f"/proc/self/fd/{fd}").as_uri()
        conn = sqlite3.connect(f"{fd_uri}?mode=rw", uri=True)
        path_after_connect = path.lstat()
        if (path_after_connect.st_dev, path_after_connect.st_ino) != (
            fd_stat.st_dev,
            fd_stat.st_ino,
        ):
            raise RuntimeError(f"SQLite DB changed while connecting: {path}")
        yield conn
        conn.commit()
    finally:
        if conn is not None:
            conn.close()
        os.close(fd)


@contextmanager
def connect_readonly(db_path: str | Path):
    """기존 SQLite 파일을 생성·변형하지 않고 연다.

    health/preflight 같은 관측 경로에서 일반 ``connect``/``init_db`` 를 쓰면
    경로 오타나 파일 유실 자체가 빈 정상 DB 생성으로 가려질 수 있다. SQLite URI의
    ``mode=ro`` 를 사용해 missing/invalid DB를 호출자에게 그대로 실패로 전달한다.
    """
    # Keep the lexical path instead of resolving it: the project root itself
    # may intentionally be reached through a parent symlink, while the DB
    # filename must still identify a real regular file (not a final-component
    # symlink).  Open it first with O_NOFOLLOW, then make SQLite duplicate that
    # stable inode through /proc/self/fd.  This closes the common
    # lstat->replace->sqlite-open TOCTOU window.
    path = Path(os.path.abspath(Path(db_path)))
    try:
        path_before = path.lstat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"SQLite DB does not exist: {path}") from exc
    if not stat.S_ISREG(path_before.st_mode):
        raise ValueError(
            f"SQLite DB must be a regular file, not a symlink/device: {path}"
        )

    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    conn: sqlite3.Connection | None = None
    try:
        fd_stat = os.fstat(fd)
        path_after_open = path.lstat()
        if (
            not stat.S_ISREG(fd_stat.st_mode)
            or (path_before.st_dev, path_before.st_ino)
            != (fd_stat.st_dev, fd_stat.st_ino)
            or (path_after_open.st_dev, path_after_open.st_ino)
            != (fd_stat.st_dev, fd_stat.st_ino)
        ):
            raise RuntimeError(
                f"SQLite DB changed while opening read-only: {path}"
            )

        fd_uri = Path(f"/proc/self/fd/{fd}").as_uri()
        conn = sqlite3.connect(f"{fd_uri}?mode=ro", uri=True)
        conn.execute("PRAGMA query_only = ON")

        path_after_connect = path.lstat()
        if (path_after_connect.st_dev, path_after_connect.st_ino) != (
            fd_stat.st_dev,
            fd_stat.st_ino,
        ):
            raise RuntimeError(
                f"SQLite DB changed while connecting read-only: {path}"
            )
        yield conn
    finally:
        if conn is not None:
            conn.close()
        os.close(fd)


def init_db(db_path: str | Path) -> None:
    """DB 파일 + 스키마 초기화 (없으면 생성, 있으면 IF NOT EXISTS 라 안전)."""
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)


# ============================================================================
# 저장
# ============================================================================
def save_candles(db_path: str | Path, df: pd.DataFrame, market: str) -> int:
    """
    candles DataFrame 을 DB 에 저장 (UPSERT).

    df columns 기대:
        timestamp (datetime, timezone-naive), open, high, low, close, volume, quote_volume
    또는 pyupbit get_ohlcv 결과 (index 가 timestamp, columns 가 open/high/low/close/volume)
    + value column (KRW 거래대금) → quote_volume 으로 매핑

    return: 저장된 행 수
    """
    if df is None or len(df) == 0:
        return 0
    if not _valid_market_identifier(market):
        raise ValueError(f"invalid market identifier: {market!r}")

    # pyupbit format 호환 처리
    df = df.copy()
    if "timestamp" not in df.columns:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                "timestamp column missing and index is not a DatetimeIndex"
            )
        df = df.reset_index()
        if "index" in df.columns and "timestamp" not in df.columns:
            df = df.rename(columns={"index": "timestamp"})

    # pyupbit 의 'value' 컬럼 → quote_volume
    if "value" in df.columns and "quote_volume" not in df.columns:
        df = df.rename(columns={"value": "quote_volume"})
    if "quote_volume" not in df.columns:
        df["quote_volume"] = None

    # timestamp 타입 정규화.
    # 각 collector 가 만든 exchange timestamp 를 timezone-naive 로 저장한다.
    # Upbit 는 KST candle boundary, Binance 는 UTC candle boundary 를 사용한다.
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"candle columns missing: {missing}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    if df["timestamp"].isna().any():
        raise ValueError("candle timestamp contains NaT")
    try:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    except AttributeError as exc:
        raise ValueError("candle timestamp values are not datetime-like") from exc
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    if df["timestamp"].duplicated().any():
        duplicates = sorted(df.loc[df["timestamp"].duplicated(), "timestamp"].unique())
        raise ValueError(f"duplicate candle timestamps in input: {duplicates[:3]}")

    for column in ("open", "high", "low", "close", "volume", "quote_volume"):
        df[column] = pd.to_numeric(df[column], errors="raise")
    prices = df[["open", "high", "low", "close"]]
    if not np.isfinite(prices.to_numpy(dtype=float)).all():
        raise ValueError("candle OHLC contains NaN or infinity")
    if (prices <= 0).any().any():
        raise ValueError("candle OHLC must be positive")
    if (
        (df["high"] < df[["open", "close"]].max(axis=1)).any()
        or (df["low"] > df[["open", "close"]].min(axis=1)).any()
        or (df["high"] < df["low"]).any()
    ):
        raise ValueError("candle OHLC relationship is invalid")
    if (
        not np.isfinite(df["volume"].to_numpy(dtype=float)).all()
        or (df["volume"] < 0).any()
    ):
        raise ValueError("candle volume must be finite and non-negative")
    quote_volume = df["quote_volume"]
    finite_quote = quote_volume.dropna().to_numpy(dtype=float)
    if (
        not np.isfinite(finite_quote).all()
        or (quote_volume.dropna() < 0).any()
    ):
        raise ValueError("candle quote_volume must be null or finite and non-negative")

    df["market"] = market
    cols = ["market", "timestamp", "open", "high", "low", "close", "volume", "quote_volume"]
    df = df[cols]

    with connect(db_path) as conn:
        # Keep first-use schema creation and the hot-path UPSERT on one
        # connection.  Collectors already initialize once per market; opening
        # a second SQLite connection for every 15m page was pure overhead.
        conn.executescript(SCHEMA)
        # UPSERT (sqlite 3.24+) — 중복 (market, timestamp) 시 update
        conn.executemany(
            """
            INSERT INTO candles (market, timestamp, open, high, low, close, volume, quote_volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(market, timestamp) DO UPDATE SET
                open = excluded.open,
                high = excluded.high,
                low = excluded.low,
                close = excluded.close,
                volume = excluded.volume,
                quote_volume = excluded.quote_volume
            """,
            df.itertuples(index=False, name=None),
        )

    return len(df)


# ============================================================================
# 조회
# ============================================================================
def load_candles(
    db_path: str | Path,
    market: str,
    since: Optional[str | pd.Timestamp] = None,
    until: Optional[str | pd.Timestamp] = None,
) -> pd.DataFrame:
    """특정 마켓의 candles 조회. timestamp 오름차순."""
    init_db(db_path)
    query = "SELECT timestamp, open, high, low, close, volume, quote_volume FROM candles WHERE market = ?"
    params: list = [market]
    if since is not None:
        query += " AND timestamp >= ?"
        params.append(str(pd.to_datetime(since)))
    if until is not None:
        query += " AND timestamp < ?"
        params.append(str(pd.to_datetime(until)))
    query += " ORDER BY timestamp ASC"

    with connect(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=params, parse_dates=["timestamp"])
    return df


def list_markets(db_path: str | Path) -> list[str]:
    """DB 에 저장된 모든 market 목록."""
    init_db(db_path)
    with connect(db_path) as conn:
        rows = conn.execute("SELECT DISTINCT market FROM candles ORDER BY market").fetchall()
    return [r[0] for r in rows]


def latest_timestamp(db_path: str | Path, market: str) -> Optional[pd.Timestamp]:
    """특정 market 의 최신 candle timestamp (timezone-naive exchange timestamp)."""
    init_db(db_path)
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT MAX(timestamp) FROM candles WHERE market = ?", (market,)
        ).fetchone()
    if row and row[0]:
        return pd.to_datetime(row[0])
    return None


def oldest_timestamp(db_path: str | Path, market: str) -> Optional[pd.Timestamp]:
    """특정 market 의 가장 오래된 timezone-naive exchange timestamp."""
    init_db(db_path)
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT MIN(timestamp) FROM candles WHERE market = ?", (market,)
        ).fetchone()
    if row and row[0]:
        return pd.to_datetime(row[0])
    return None


def market_timestamp_ranges_readonly(
    db_path: str | Path,
) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """market별 ``(oldest, latest)``를 단일 grouped query로 읽는다.

    DB 파일·디렉터리·스키마를 생성하지 않는다. 관측 중 파일이 없거나 schema가
    손상됐으면 예외를 내어 health gate가 fail closed 하도록 한다.
    """
    with connect_readonly(db_path) as conn:
        rows = conn.execute(
            """
            SELECT market, MIN(timestamp), MAX(timestamp)
            FROM candles
            GROUP BY market
            ORDER BY market
            """
        ).fetchall()

    ranges: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for market, oldest, latest in rows:
        if oldest is None or latest is None:
            continue
        ranges[str(market)] = (pd.Timestamp(oldest), pd.Timestamp(latest))
    return ranges


def stats(db_path: str | Path) -> pd.DataFrame:
    """DB 전체 통계: market 별 row 수, 최신 / 최오래된 timestamp."""
    init_db(db_path)
    with connect(db_path) as conn:
        df = pd.read_sql_query(
            """
            SELECT
                market,
                COUNT(*) AS rows,
                MIN(timestamp) AS oldest,
                MAX(timestamp) AS latest
            FROM candles
            GROUP BY market
            ORDER BY market
            """,
            conn,
            parse_dates=["oldest", "latest"],
        )
    return df


# ============================================================================
# 직접 실행 시 — DB 상태 출력
# ============================================================================
if __name__ == "__main__":
    import sys

    db_path = sys.argv[1] if len(sys.argv) > 1 else "data/upbit_d1.db"
    print(f"=== {db_path} ===")
    s = stats(db_path)
    if len(s) == 0:
        print("(empty)")
    else:
        print(s.to_string(index=False))
        print(f"\nTotal markets: {len(s)}, Total rows: {s['rows'].sum():,}")

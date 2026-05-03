"""sqlite 헬퍼 — candles 저장 / 조회.

설계: interval 별로 DB 파일 분리 (upbit_d1.db / upbit_4h.db / binance_1h.db).
호출자가 db_path 명시. 한 곳에서 모든 sqlite 로직 관리.

Adapted from: gan_t/data/database.py
Changes:
  - 단순화 (단일 candles 테이블, generic interval)
  - pyupbit DataFrame 직접 받음
  - timestamp 항상 UTC (KST 09:00 = UTC 00:00 일봉 마감 — SIGNAL.md §1.3)
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import pandas as pd

# ============================================================================
# 스키마
# ============================================================================
SCHEMA = """
CREATE TABLE IF NOT EXISTS candles (
    market TEXT NOT NULL,
    timestamp DATETIME NOT NULL,    -- UTC, ISO 8601 (예 '2026-05-03 00:00:00')
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


# ============================================================================
# 연결 / 초기화
# ============================================================================
@contextmanager
def connect(db_path: str | Path):
    """sqlite 연결 context manager."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


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
        timestamp (datetime, UTC), open, high, low, close, volume, quote_volume
    또는 pyupbit get_ohlcv 결과 (index 가 timestamp, columns 가 open/high/low/close/volume)
    + value column (KRW 거래대금) → quote_volume 으로 매핑

    return: 저장된 행 수
    """
    if df is None or len(df) == 0:
        return 0

    init_db(db_path)

    # pyupbit format 호환 처리
    df = df.copy()
    if df.index.name in ("timestamp", None) and "timestamp" not in df.columns:
        df = df.reset_index()
        if "index" in df.columns and "timestamp" not in df.columns:
            df = df.rename(columns={"index": "timestamp"})

    # pyupbit 의 'value' 컬럼 → quote_volume
    if "value" in df.columns and "quote_volume" not in df.columns:
        df = df.rename(columns={"value": "quote_volume"})
    if "quote_volume" not in df.columns:
        df["quote_volume"] = None

    df["market"] = market

    # timestamp 타입 정규화 (UTC)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    cols = ["market", "timestamp", "open", "high", "low", "close", "volume", "quote_volume"]
    df = df[cols]

    with connect(db_path) as conn:
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
    """특정 market 의 최신 candle timestamp (UTC)."""
    init_db(db_path)
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT MAX(timestamp) FROM candles WHERE market = ?", (market,)
        ).fetchone()
    if row and row[0]:
        return pd.to_datetime(row[0])
    return None


def oldest_timestamp(db_path: str | Path, market: str) -> Optional[pd.Timestamp]:
    """특정 market 의 가장 오래된 candle timestamp (UTC)."""
    init_db(db_path)
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT MIN(timestamp) FROM candles WHERE market = ?", (market,)
        ).fetchone()
    if row and row[0]:
        return pd.to_datetime(row[0])
    return None


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

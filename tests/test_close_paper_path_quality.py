from __future__ import annotations

import sqlite3

import pandas as pd

from scripts.close_paper_ledger import compute_realized


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


def _rows(market: str, timestamps: pd.DatetimeIndex, price: float) -> list[tuple]:
    return [
        (
            market,
            timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            price,
            price,
            price,
            price,
        )
        for timestamp in timestamps
    ]


def _db(tmp_path, benchmark: pd.DatetimeIndex, target: pd.DatetimeIndex):
    path = tmp_path / "4h.db"
    benchmark = benchmark.append(
        pd.DatetimeIndex([benchmark[-1] + pd.Timedelta(hours=4)])
    )
    with sqlite3.connect(path) as conn:
        conn.execute(SCHEMA)
        conn.executemany(
            "INSERT INTO candles VALUES (?,?,?,?,?,?)",
            _rows("KRW-BTC", benchmark, 200.0)
            + _rows("KRW-ALT", target, 100.0),
        )
    return path


def test_distribution_close_flat_fills_target_only_4h_gap(tmp_path):
    grid = pd.date_range("2026-07-01 09:00:00", periods=6, freq="4h")
    db = _db(tmp_path, grid, grid.delete(2))

    result = compute_realized("KRW-ALT", pd.Timestamp("2026-07-01"), str(db))

    assert result["status"] == "closed"
    assert result["path_complete"] is True
    assert result["path_quality"] == "flat_filled"
    assert result["raw_bars"] == 5
    assert result["flat_filled_bars"] == 1
    assert result["path_used_bars"] == 6
    assert result["next_close_return_pct"] == 0.0


def test_distribution_close_defers_when_btc_4h_grid_is_incomplete(tmp_path):
    grid = pd.date_range("2026-07-01 09:00:00", periods=6, freq="4h")
    db = _db(tmp_path, grid.delete(3), grid)

    result = compute_realized("KRW-ALT", pd.Timestamp("2026-07-01"), str(db))

    assert result["status"] == "no_data"
    assert result["path_complete"] is False
    assert result["path_quality"] == "benchmark_gap"

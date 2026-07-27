from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data import database
from data.database import (
    connect_readonly,
    list_markets,
    load_candles,
    save_candles,
)


def _valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [10.0, 11.0],
            "value": [1000.0, 1111.0],
        },
        index=pd.DatetimeIndex(
            ["2026-07-25 09:00:00", "2026-07-26 09:00:00"]
        ),
    )


def test_save_candles_accepts_valid_datetime_index_and_round_trips(tmp_path):
    path = tmp_path / "candles.db"
    frame = _valid_frame()

    assert save_candles(path, frame, "KRW-BTC") == 2
    loaded = load_candles(path, "KRW-BTC")

    assert len(loaded) == 2
    assert loaded["timestamp"].dt.strftime("%F %T").tolist() == [
        "2026-07-25 09:00:00",
        "2026-07-26 09:00:00",
    ]
    assert loaded["quote_volume"].tolist() == [1000.0, 1111.0]


def test_save_candles_accepts_exchange_unicode_market_identity(tmp_path):
    path = tmp_path / "binance.db"
    market = "BINANCE-币安人生USDT"

    assert save_candles(path, _valid_frame(), market) == 2

    loaded = load_candles(path, market)
    assert len(loaded) == 2
    assert list_markets(path) == [market]


def test_save_candles_rejects_range_index_as_timestamp(tmp_path):
    frame = _valid_frame().reset_index(drop=True)
    path = tmp_path / "candles.db"
    with pytest.raises(ValueError, match="timestamp column missing"):
        save_candles(path, frame, "KRW-BTC")
    assert not path.exists()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda frame: frame.__setitem__("high", [98.0, 103.0]), "relationship"),
        (lambda frame: frame.__setitem__("close", [np.inf, 102.0]), "NaN or infinity"),
        (lambda frame: frame.__setitem__("volume", [-1.0, 11.0]), "volume"),
        (lambda frame: frame.__setitem__("value", [np.nan, -1.0]), "quote_volume"),
    ],
)
def test_save_candles_rejects_invalid_numeric_contract(
    tmp_path,
    mutate,
    message,
):
    frame = _valid_frame()
    mutate(frame)
    with pytest.raises(ValueError, match=message):
        save_candles(tmp_path / "candles.db", frame, "KRW-BTC")


def test_save_candles_rejects_duplicate_timestamp_and_bad_market(tmp_path):
    frame = _valid_frame()
    frame.index = pd.DatetimeIndex([frame.index[0], frame.index[0]])
    with pytest.raises(ValueError, match="duplicate candle timestamps"):
        save_candles(tmp_path / "candles.db", frame, "KRW-BTC")

    with pytest.raises(ValueError, match="market identifier"):
        save_candles(tmp_path / "candles.db", _valid_frame(), "KRW-BTC;DROP")

    for market in ("", " KRW-BTC", "KRW-BTC\n", "KRW/BTC"):
        with pytest.raises(ValueError, match="market identifier"):
            save_candles(tmp_path / "candles.db", _valid_frame(), market)


def test_readonly_connection_does_not_create_missing_database(tmp_path):
    path = tmp_path / "missing.db"
    with pytest.raises(FileNotFoundError):
        with connect_readonly(path):
            pass
    assert not path.exists()


def test_readonly_connection_rejects_missing_schema_without_mutation(tmp_path):
    path = tmp_path / "empty.db"
    with sqlite3.connect(path):
        pass
    before = path.read_bytes()
    with connect_readonly(path) as conn:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("SELECT COUNT(*) FROM candles").fetchone()
    assert path.read_bytes() == before


def test_readonly_connection_rejects_final_component_symlink(tmp_path):
    target = tmp_path / "real.db"
    save_candles(target, _valid_frame(), "KRW-BTC")
    link = tmp_path / "linked.db"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="regular file"):
        with connect_readonly(link):
            pass

    with sqlite3.connect(target) as conn:
        assert conn.execute("SELECT COUNT(*) FROM candles").fetchone() == (2,)


def test_save_candles_never_follows_database_symlink(tmp_path):
    target = tmp_path / "real.db"
    save_candles(target, _valid_frame(), "KRW-BTC")
    link = tmp_path / "linked.db"
    link.symlink_to(target)

    with pytest.raises(OSError):
        save_candles(link, _valid_frame(), "KRW-ETH")

    with sqlite3.connect(target) as conn:
        assert conn.execute(
            "SELECT DISTINCT market FROM candles ORDER BY market"
        ).fetchall() == [("KRW-BTC",)]


def test_readonly_connection_detects_path_replacement_during_open(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "candles.db"
    replacement = tmp_path / "replacement.db"
    save_candles(path, _valid_frame(), "KRW-BTC")
    save_candles(replacement, _valid_frame().iloc[:1], "KRW-ETH")
    actual_open = database.os.open
    replaced = False

    def replace_after_open(path_value, flags, *args):
        nonlocal replaced
        fd = actual_open(path_value, flags, *args)
        if Path(path_value) == path and not replaced:
            replaced = True
            database.os.replace(replacement, path)
        return fd

    monkeypatch.setattr(database.os, "open", replace_after_open)

    with pytest.raises(RuntimeError, match="changed while opening"):
        with connect_readonly(path):
            pass


def test_recommend_open_lookup_never_creates_a_missing_database(
    tmp_path,
    monkeypatch,
):
    from signals import recommend

    path = tmp_path / "missing" / "upbit_d1.db"
    monkeypatch.setattr(recommend, "DB_PATH", str(path))

    with pytest.raises(FileNotFoundError):
        recommend._load_asof_open(
            pd.Timestamp("2026-07-26"),
            {"KRW-BTC"},
        )

    assert not path.exists()
    assert not path.parent.exists()

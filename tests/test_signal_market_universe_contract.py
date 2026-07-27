from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data.market_universe import (
    signal_eligible_markets,
)
from scripts import predict_preopen_trigger
from scripts import predict_today_distribution
from signals import predict
from signals import retrain
from signals.features import assemble_training_panel


class _StopAfterPanel(RuntimeError):
    pass


_ROOT = Path(__file__).resolve().parents[1]
_RAW_LIST_MARKET_ALLOWLIST = {
    "ops/preflight.py",
    "scripts/backfill_1h_qa.py",
}


def _rows(count: int = 101) -> pd.DataFrame:
    return pd.DataFrame({"row": range(count)})


def test_signal_eligible_markets_preserves_order_and_removes_aliases() -> None:
    markets = ["KRW-IP", "KRW-BTC", "KRW-USDT", "KRW-DATA", "BINANCE-BTCUSDT"]

    assert signal_eligible_markets(markets) == [
        "KRW-BTC",
        "KRW-DATA",
        "BINANCE-BTCUSDT",
    ]


def test_training_panel_fails_closed_on_excluded_market() -> None:
    with pytest.raises(ValueError, match=r"excluded signal markets: KRW-IP"):
        assemble_training_panel(
            {"KRW-IP": pd.DataFrame()},
            pd.DataFrame(),
        )


def test_signal_code_filters_every_direct_market_listing() -> None:
    offenders: list[str] = []
    for directory in ("ops", "scripts", "signals"):
        for path in (_ROOT / directory).rglob("*.py"):
            relative = path.relative_to(_ROOT).as_posix()
            source = path.read_text(encoding="utf-8")
            if "list_markets" not in source:
                continue
            if relative in _RAW_LIST_MARKET_ALLOWLIST:
                continue
            if (
                "signal_eligible_markets" not in source
                and "is_excluded_signal_market" not in source
            ):
                offenders.append(relative)

    assert offenders == []


def test_distribution_builder_excludes_alias_before_panel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: list[str] = []
    monkeypatch.setattr(
        predict_today_distribution,
        "list_markets",
        lambda _db: ["KRW-IP", "KRW-DATA"],
    )
    monkeypatch.setattr(
        predict_today_distribution,
        "load_candles",
        lambda *_args, **_kwargs: _rows(31),
    )

    def capture(candles, _btc, **_kwargs):
        captured.extend(candles)
        return pd.DataFrame()

    monkeypatch.setattr(
        predict_today_distribution,
        "assemble_training_panel",
        capture,
    )

    result = predict_today_distribution.build_panel_for_asof(
        "upbit.db",
        str(tmp_path / "missing-binance.db"),
        pd.Timestamp("2026-07-26 09:05:00"),
    )

    assert result.empty
    assert captured == ["KRW-DATA"]


def test_preopen_builder_excludes_alias_from_daily_and_intraday(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    daily_markets: list[str] = []
    intraday_markets: list[str] = []
    monkeypatch.setattr(
        predict_preopen_trigger,
        "list_markets",
        lambda _db: ["KRW-IP", "KRW-DATA"],
    )
    monkeypatch.setattr(
        predict_preopen_trigger,
        "load_candles",
        lambda *_args, **_kwargs: _rows(),
    )

    def capture_daily(candles, _btc, **_kwargs):
        daily_markets.extend(candles)
        return pd.DataFrame(
            {
                "market": ["KRW-DATA"],
                "timestamp": [pd.Timestamp("2026-07-25 09:00:00")],
                "quote_volume": [1.0],
            }
        )

    def capture_intraday(candles, _btc):
        intraday_markets.extend(candles)
        return pd.DataFrame()

    monkeypatch.setattr(
        predict_preopen_trigger,
        "assemble_training_panel",
        capture_daily,
    )
    monkeypatch.setattr(
        predict_preopen_trigger,
        "build_15m_precursor",
        capture_intraday,
    )

    latest, precursor = predict_preopen_trigger.build_panel_for_asof(
        "upbit-d1.db",
        "upbit-15m.db",
        str(tmp_path / "missing-binance.db"),
        pd.Timestamp("2026-07-26 08:55:00"),
    )

    assert latest["market"].tolist() == ["KRW-DATA"]
    assert precursor.empty
    assert daily_markets == ["KRW-DATA"]
    assert intraday_markets == ["KRW-DATA"]


def test_manual_predict_and_retrain_exclude_alias_before_panel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen: list[list[str]] = []
    listed = ["KRW-IP", "KRW-DATA"]

    monkeypatch.setattr(predict, "list_markets", lambda _db: listed)
    monkeypatch.setattr(
        predict,
        "load_candles",
        lambda *_args, **_kwargs: _rows(31),
    )

    def capture_predict(candles, _btc, **_kwargs):
        seen.append(list(candles))
        return pd.DataFrame()

    monkeypatch.setattr(predict, "assemble_training_panel", capture_predict)
    assert predict.predict_today(
        object(),
        upbit_db="upbit.db",
        binance_db=None,
        asof=pd.Timestamp("2026-07-26 09:05:00"),
    ).empty

    monkeypatch.setattr(retrain, "list_markets", lambda _db: listed)
    monkeypatch.setattr(
        retrain,
        "load_candles",
        lambda *_args, **_kwargs: _rows(31),
    )

    def capture_retrain(candles, _btc, **_kwargs):
        seen.append(list(candles))
        raise _StopAfterPanel

    monkeypatch.setattr(retrain, "assemble_training_panel", capture_retrain)
    with pytest.raises(_StopAfterPanel):
        retrain.run_retrain(
            upbit_db="upbit.db",
            binance_db=tmp_path / "missing-binance.db",
            history_path=tmp_path / "history.json",
            archive_dir=tmp_path / "archive",
        )

    assert seen == [["KRW-DATA"], ["KRW-DATA"]]
    assert not predict.is_tradable_market("KRW-IP")
    assert predict.is_tradable_market("KRW-DATA")

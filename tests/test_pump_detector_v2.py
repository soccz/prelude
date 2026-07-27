"""PUMP hunter v2 — registry 잠금 + 메시지 정직성 + volsurge 계산 검증."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import signals.pump_detector_v2 as pump_detector_v2  # noqa: E402
import signals.pump_detector_v1 as pump_detector_v1  # noqa: E402
from data.market_universe import SIGNAL_EXCLUDED_KRW_MARKETS  # noqa: E402
from signals.model_registry import MODELS  # noqa: E402
from signals.pump_detector_v2 import (  # noqa: E402
    BN_VOL_LOOKBACK,
    BN_VOL_SURGE_MIN,
    OOS_HIT_PCT,
    OOS_NET_TP5SL3_PCT,
    ROC7_RANK_MIN,
    binance_volsurge_for_date,
    krw_to_binance,
    score_pump_v2_candidates,
)
from scripts.pump_detector_v2_today import build_message  # noqa: E402


def test_registry_has_v2_as_challenger_only():
    """v2 는 champion 승격 금지 (challenger_only) — radar telegram 과 별개 보장."""
    spec = next((m for m in MODELS if m.id == "pump_hunter_v2"), None)
    assert spec is not None, "pump_hunter_v2 가 MODELS 에 없음"
    assert spec.challenger_only is True
    assert spec.ledger_path == "output/shadow_ledger_pump_hunter_v2.csv"
    assert "pump_detector_v2" in spec.predict_ref


def test_symbol_mapping_excludes_signal_markets():
    assert krw_to_binance("KRW-BTC") == "BINANCE-BTCUSDT"
    for market in SIGNAL_EXCLUDED_KRW_MARKETS:
        assert krw_to_binance(market) is None
    assert krw_to_binance("KRW-WBTC") == "BINANCE-WBTCUSDT"
    assert krw_to_binance("KRW-WETH") == "BINANCE-WETHUSDT"
    assert krw_to_binance("BTC") is None
    assert krw_to_binance("FOO-KRW-BTC") is None


def test_rule_thresholds_match_validated_research():
    """임계가 검증된 값 (binance_leadlag_v1) 에서 무단 변경되지 않게 고정."""
    assert ROC7_RANK_MIN == 0.85
    assert BN_VOL_SURGE_MIN == 1.5


def test_message_includes_honest_disclosure():
    """radar 메시지에 (a) hit 엣지 (b) 자동 net 음수 고지 (c) 자동주문 없음이 반드시 포함."""
    res = {
        "asof": "2026-06-12",
        "btc_regime": "bear_quiet",
        "binance_status": "ok",
        "n_candidates": 1,
        "candidates": [{
            "market": "KRW-TEST", "rank": 1, "score": 0.9, "entry_open": 100.0,
            "roc_7d": 12.0, "roc_7d_rank": 0.95, "atr_pct_14": 0.08,
            "log_return_1d": 0.01, "b_vol_surge": 3.2, "b_ret_1d": 0.05,
            "liq_rank_daily": 10, "btc_regime": "bear_quiet",
            "rule_id": "roc7_rank+bn_volsurge",
        }],
    }
    msg = build_message(res)
    assert f"{OOS_HIT_PCT}%" in msg          # hit 엣지 명시
    assert f"{OOS_NET_TP5SL3_PCT}%" in msg   # 자동 net 음수 정직 고지
    assert "자동 주문 없음" in msg
    assert "TEST" in msg and "surge 3.2" in msg


def test_message_stale_binance_warns():
    res = {"asof": "2026-06-12", "btc_regime": "bear_quiet",
           "binance_status": "binance_stale (latest=2026-05-03, need=2026-06-11)",
           "n_candidates": 0, "candidates": []}
    msg = build_message(res)
    assert "binance 데이터 문제" in msg


def test_message_quiet_when_no_candidates():
    res = {"asof": "2026-06-12", "btc_regime": "bull_quiet",
           "binance_status": "ok", "n_candidates": 0, "candidates": []}
    msg = build_message(res)
    assert "rule fire 없음" in msg


def _binance_daily(end: str, periods: int = BN_VOL_LOOKBACK + 1) -> pd.DataFrame:
    index = pd.date_range(end=end, periods=periods, freq="D")
    return pd.DataFrame(
        {
            "timestamp": index,
            "close": [100.0 + i for i in range(periods)],
            "quote_volume": [1000.0] * (periods - 1) + [2000.0],
        }
    )


def _upbit_daily_frame(
    *,
    daily_growth: float,
    quote_volume: float,
) -> pd.DataFrame:
    periods = 30
    timestamps = pd.date_range(
        end="2026-07-26 09:00:00",
        periods=periods,
        freq="D",
    )
    close = 100.0 * np.power(1.0 + daily_growth, np.arange(periods))
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": quote_volume / close,
            "quote_volume": np.full(periods, quote_volume),
        }
    )


@pytest.mark.parametrize(
    "excluded_market",
    sorted(SIGNAL_EXCLUDED_KRW_MARKETS),
)
def test_excluded_market_extremes_cannot_change_v2_universe_or_candidates(
    monkeypatch,
    excluded_market,
):
    frames = {
        "KRW-A": _upbit_daily_frame(
            daily_growth=0.05,
            quote_volume=300.0,
        ),
        "KRW-B": _upbit_daily_frame(
            daily_growth=0.03,
            quote_volume=200.0,
        ),
        "KRW-C": _upbit_daily_frame(
            daily_growth=0.01,
            quote_volume=100.0,
        ),
        excluded_market: _upbit_daily_frame(
            daily_growth=0.90,
            quote_volume=1e15,
        ),
    }
    market_sets = iter(
        [
            ["KRW-A", "KRW-B", "KRW-C"],
            [excluded_market, "KRW-A", "KRW-B", "KRW-C"],
        ]
    )
    monkeypatch.setattr(
        pump_detector_v1,
        "list_markets",
        lambda _db: next(market_sets),
    )
    monkeypatch.setattr(
        pump_detector_v1,
        "load_candles",
        lambda _db, market: frames[market].copy(),
    )
    monkeypatch.setattr(
        pump_detector_v1,
        "_btc_regime_for_feature_date",
        lambda *_args: "bull_quiet",
    )

    def fake_binance(_feature_date, needed, _db):
        excluded_bn_market = (
            f"BINANCE-{excluded_market.removeprefix('KRW-')}USDT"
        )
        assert excluded_bn_market not in needed
        return (
            pd.DataFrame(
                {
                    "bn_market": sorted(needed),
                    "b_vol_surge": [2.0] * len(needed),
                    "b_ret_1d": [0.01] * len(needed),
                }
            ),
            "ok",
        )

    monkeypatch.setattr(
        pump_detector_v2,
        "binance_volsurge_for_date",
        fake_binance,
    )

    baseline = score_pump_v2_candidates(
        "2026-07-26",
        db_path="unused.db",
        binance_db="unused-binance.db",
        top_universe=2,
        limit_markets=2,
    )
    attacked = score_pump_v2_candidates(
        "2026-07-26",
        db_path="unused.db",
        binance_db="unused-binance.db",
        top_universe=2,
        limit_markets=2,
    )

    assert baseline == attacked
    assert baseline["universe_n"] == 2
    assert [row["market"] for row in baseline["candidates"]] == ["KRW-A"]


@pytest.mark.parametrize("excluded_market", ["KRW-USDT", "KRW-IP"])
def test_v2_fails_closed_if_upstream_ranked_frame_contains_excluded_market(
    monkeypatch,
    excluded_market,
):
    monkeypatch.setattr(
        pump_detector_v2,
        "build_feature_frame",
        lambda *_args, **_kwargs: pd.DataFrame(
            [{"market": excluded_market}]
        ),
    )

    with pytest.raises(
        RuntimeError,
        match=rf"excluded signal market.*{excluded_market}",
    ):
        score_pump_v2_candidates("2026-07-26")


def test_binance_volsurge_requires_every_needed_symbol_at_feature_date(
    monkeypatch,
):
    frames = {
        "BINANCE-BTCUSDT": _binance_daily("2026-07-25"),
        # ETH만 feature_date보다 하루 stale.
        "BINANCE-ETHUSDT": _binance_daily("2026-07-24"),
    }
    monkeypatch.setattr(
        pump_detector_v2,
        "load_candles",
        lambda _db, market: frames[market],
    )

    result, status = binance_volsurge_for_date(
        "2026-07-25",
        {"BINANCE-BTCUSDT", "BINANCE-ETHUSDT"},
        "unused.db",
    )

    assert result.empty
    assert status.startswith("binance_partial")
    assert "ready=1/2" in status
    assert "BINANCE-ETHUSDT:feature_date_missing" in status


def test_partial_binance_data_fails_closed_to_zero_candidates(monkeypatch):
    feature_date = "2026-07-25 09:00:00"
    frame = pd.DataFrame(
        [
            {
                "market": "KRW-BTC",
                "feature_date": feature_date,
                "btc_regime": "bull_quiet",
                "roc_7d_rank": 0.99,
            },
            {
                "market": "KRW-ETH",
                "feature_date": feature_date,
                "btc_regime": "bull_quiet",
                "roc_7d_rank": 0.98,
            },
        ]
    )
    frames = {
        "BINANCE-BTCUSDT": _binance_daily("2026-07-25"),
        "BINANCE-ETHUSDT": _binance_daily("2026-07-24"),
    }
    monkeypatch.setattr(
        pump_detector_v2,
        "build_feature_frame",
        lambda *_args, **_kwargs: frame,
    )
    monkeypatch.setattr(
        pump_detector_v2,
        "load_candles",
        lambda _db, market: frames[market],
    )

    result = score_pump_v2_candidates(
        "2026-07-26",
        binance_db="unused.db",
    )

    assert result["binance_status"].startswith("binance_partial")
    assert result["n_candidates"] == 0
    assert result["candidates"] == []


def test_binance_volsurge_accepts_exact_feature_date_plus_20_days(
    monkeypatch,
):
    frames = {
        "BINANCE-BTCUSDT": _binance_daily("2026-07-25"),
        "BINANCE-ETHUSDT": _binance_daily("2026-07-25"),
    }
    monkeypatch.setattr(
        pump_detector_v2,
        "load_candles",
        lambda _db, market: frames[market],
    )

    result, status = binance_volsurge_for_date(
        "2026-07-25",
        set(frames),
        "unused.db",
    )

    assert status == "ok"
    assert set(result["bn_market"]) == set(frames)
    assert result["b_vol_surge"].tolist() == pytest.approx([2.0, 2.0])


def test_binance_feature_date_normalizes_aware_kst_business_day(
    monkeypatch,
):
    monkeypatch.setattr(
        pump_detector_v2,
        "load_candles",
        lambda _db, _market: _binance_daily("2026-07-25"),
    )

    result, status = binance_volsurge_for_date(
        pd.Timestamp("2026-07-25 09:00:00", tz="Asia/Seoul"),
        {"BINANCE-BTCUSDT"},
        "unused.db",
    )

    assert status == "ok"
    assert result.iloc[0]["b_vol_surge"] == pytest.approx(2.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"top_universe": 0},
        {"max_candidates": 0},
        {"limit_markets": 0},
    ],
)
def test_score_rejects_nonpositive_limits_before_loading_data(kwargs):
    with pytest.raises(ValueError):
        score_pump_v2_candidates("2026-07-26", **kwargs)

"""PUMP hunter v2 — registry 잠금 + 메시지 정직성 + volsurge 계산 검증."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signals.model_registry import MODELS  # noqa: E402
from signals.pump_detector_v2 import (  # noqa: E402
    BN_VOL_SURGE_MIN,
    OOS_HIT_PCT,
    OOS_NET_TP5SL3_PCT,
    ROC7_RANK_MIN,
    krw_to_binance,
)
from scripts.pump_detector_v2_today import build_message  # noqa: E402


def test_registry_has_v2_as_challenger_only():
    """v2 는 champion 승격 금지 (challenger_only) — radar telegram 과 별개 보장."""
    spec = next((m for m in MODELS if m.id == "pump_hunter_v2"), None)
    assert spec is not None, "pump_hunter_v2 가 MODELS 에 없음"
    assert spec.challenger_only is True
    assert spec.ledger_path == "output/shadow_ledger_pump_hunter_v2.csv"
    assert "pump_detector_v2" in spec.predict_ref


def test_symbol_mapping_excludes_stables():
    assert krw_to_binance("KRW-BTC") == "BINANCE-BTCUSDT"
    assert krw_to_binance("KRW-USDT") is None
    assert krw_to_binance("KRW-USDC") is None


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

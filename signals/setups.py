"""Setup Library v1 — Setup Discovery v1 결과 기반 룰 정의.

setup discovery 결과 (output/setup_discovery_v1.csv) 에서 robust + interpretable
한 leaf 들을 named setup 으로 인코딩.

각 setup 은:
  detect_fn(row) -> bool   # 패널 행 1개에 룰 적용
  past_stats               # 과거 검증 통계 (alert 표시용)

S04 는 setup 이 아니라 BTC context (regime) — get_btc_context() 사용.
"""
from __future__ import annotations

from typing import Callable

# ============================================================================
# Setup definitions
# ============================================================================
SETUP_LIBRARY: dict[str, dict] = {
    "S01": {
        "name": "high-vol momentum",
        "desc": "ATR 높음 + 전일 상승 + 3d ROC 상위 (h2/h6 공통)",
        "heads_supported": ["h2_hit_3_4h", "h6_hit_5_24h"],
        "rule_text": "atr_pct_14 > 0.06 AND log_return_1d > 0.04 AND roc_3d (rank) > 0.93",
        "detect_fn": lambda r: (
            r.get("atr_pct_14", 0) > 0.06
            and r.get("log_return_1d", 0) > 0.04
            and r.get("roc_3d", 0) > 0.93
        ),
        "past_stats": {
            "h2_lift": 4.09, "h2_prec_pct": 47.5, "h2_n": 552, "h2_fail_eod_pct": -2.5,
            "h6_lift": 2.63, "h6_prec_pct": 48.6, "h6_n": 479, "h6_fail_eod_pct": -4.2,
        },
    },
    "S02": {
        "name": "strong yesterday move",
        "desc": "전일 강한 상승 (log_return_1d > 5%) — h5 핵심 robust setup",
        "heads_supported": ["h5_tail_20"],
        "rule_text": "log_return_1d > 0.048 (depth=1, fold 1: lift 4.07)",
        "detect_fn": lambda r: r.get("log_return_1d", 0) > 0.048,
        "past_stats": {
            "h5_lift": 4.07, "h5_prec_pct": 6.5, "h5_n": 1408, "h5_fail_eod_pct": -0.65,
        },
    },
    "S03": {
        "name": "vol expansion + 5d momentum",
        "desc": "5d 변동성 상위 + 5/7d return 상위 + 전일 상승 (h5 보조)",
        "heads_supported": ["h5_tail_20"],
        "rule_text": "vol_5d (rank) > 0.83 AND return_7d (rank) > 0.91 AND log_return_1d > 0.059",
        "detect_fn": lambda r: (
            r.get("vol_5d", 0) > 0.83
            and r.get("return_7d", 0) > 0.91
            and r.get("log_return_1d", 0) > 0.059
        ),
        "past_stats": {
            "h5_lift": 6.81, "h5_prec_pct": 8.05, "h5_n": 298, "h5_fail_eod_pct": -1.89,
        },
    },
    "S04": {
        # BTC context — not a setup, but tracked
        "name": "BTC bull context",
        "desc": "BTC regime ∈ {bull_quiet, bull_volatile}",
        "heads_supported": [],
        "rule_text": "btc_regime in {bull_quiet, bull_volatile}",
        "detect_fn": lambda r: r.get("btc_regime", "") in {"bull_quiet", "bull_volatile"},
        "past_stats": {"context_only": True},
    },
}


def detect_setups(row) -> list[str]:
    """Apply all setup rules to a single panel row, return list of fired setup IDs."""
    fired = []
    for setup_id, sdef in SETUP_LIBRARY.items():
        try:
            if sdef["detect_fn"](row):
                fired.append(setup_id)
        except Exception:
            continue
    return fired


def get_setup_summary(setup_ids: list[str]) -> str:
    """Format setup ids as readable string for alerts."""
    if not setup_ids:
        return "(no setup)"
    parts = []
    for s_id in setup_ids:
        if s_id in SETUP_LIBRARY:
            parts.append(f"{s_id} {SETUP_LIBRARY[s_id]['name']}")
    return ", ".join(parts)

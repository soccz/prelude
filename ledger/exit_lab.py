"""Exit lab — 같은 15m 경로에 청산 변형 N 개를 병렬 가상 평가 (record-only).

배경: bear_quiet 경로 연구 (output/bear_quiet_path_exit_v1.csv) 에서 유일하게
마진이 두꺼운 양수 경로는 TP+10% / SL 없음 / EOD fallback (+0.29%/trade) 였다.
그런데 운영 shadow 청산 (scripts/close_recommend_ledger.py) 은 TP+5% / SL-3%
하나만 기록한다 — 펌프가 오후~밤에 완성되는 구조에서 SL-3% 가 70% 를 조기
절단하면, detector 가 "잘못된 잣대의 음수" 로 기각될 위험이 있다.

어느 잣대가 맞는지는 forward 데이터가 결정한다 (§2.5). 그래서 close 시점에
같은 15m 경로로 변형들을 나란히 기록한다. 알림/주문 영향 0 — 가상 평가만.

같은 봉 동시 터치 규약: SL 먼저 (보수) — scripts/recommender_downside_exit_v1.
simulate_path 와 동일 semantics. tests/test_exit_lab.py 가 두 구현의 동등성을
고정한다 (한쪽이 바뀌면 테스트가 강제로 sync 시킴).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ledger.config import ROUND_TRIP_COST_PCT


@dataclass(frozen=True)
class ExitSpec:
    """청산 변형 정의. tp/sl 은 진입가 대비 비율 (양수 magnitude), None=비활성."""
    tp: Optional[float]
    sl: Optional[float]
    label: str = ""


# 기록할 변형 (운영 기본 TP5/SL3 는 realized_pct/exit_reason 에 이미 있음 — 중복 X).
# 모든 숫자 placeholder — bear_quiet 연구 + 비대칭 bracket sweep 초기값 (§2.5).
# 2026-06-11 1차 backfill 결과 noSL 군이 전부 더 나빴음 (SL 이 하방을 지킴) →
# 2차로 bracket 비대칭 (SL 더 타이트 / 더 와이드 / TP 연장) 을 라이브로 결판.
EXIT_VARIANTS: dict[str, ExitSpec] = {
    "tp10_nosl": ExitSpec(tp=0.10, sl=None, label="TP+10% / SL없음 / EOD (bear_quiet 연구 최적)"),
    "tp5_nosl": ExitSpec(tp=0.05, sl=None, label="TP+5% / SL없음 / EOD"),
    "eod": ExitSpec(tp=None, sl=None, label="순수 EOD hold (기준선)"),
    "tp5_sl2": ExitSpec(tp=0.05, sl=0.02, label="TP+5% / SL-2% (출혈 더 빨리 절단)"),
    "tp10_sl5": ExitSpec(tp=0.10, sl=0.05, label="TP+10% / SL-5% (사용자 -5% 수용 anchor + 늦은 펌프 여유)"),
    "tp8_sl3": ExitSpec(tp=0.08, sl=0.03, label="TP+8% / SL-3% (같은 SL, 승자 더 끌기)"),
}

# ledger 에 추가되는 컬럼 (close 시점에 채움; open row 는 NA).
EXIT_LAB_COLS = [
    "exit_tp10_nosl_pct", "exit_tp10_nosl_reason",
    "exit_tp5_nosl_pct", "exit_tp5_nosl_reason",
    "exit_eod_pct",
    "exit_tp5_sl2_pct", "exit_tp5_sl2_reason",
    "exit_tp10_sl5_pct", "exit_tp10_sl5_reason",
    "exit_tp8_sl3_pct", "exit_tp8_sl3_reason",
    "path_max_pct", "path_min_pct",
]


def walk_path(bars: list, sl: Optional[float], tp: Optional[float]) -> tuple[float, str]:
    """15m 봉 시간순 walk → (gross_return, outcome). 진입가 = 첫 봉 open.

    scripts/recommender_downside_exit_v1.simulate_path(bars, sl, tp, trail=None)
    의 미러 (trail 미지원 버전). 같은 봉 SL·TP 동시 → SL 먼저 (보수).
    """
    if not bars:
        return np.nan, "nodata"
    entry = bars[0][0]
    if not np.isfinite(entry) or entry <= 0:
        return np.nan, "nodata"
    sl_px = entry * (1 - sl) if sl is not None else None
    tp_px = entry * (1 + tp) if tp is not None else None
    for (o, h, l, c) in bars:
        if sl_px is not None and l <= sl_px:
            return -sl, "sl"
        if tp_px is not None and h >= tp_px:
            return tp, "tp"
    return bars[-1][3] / entry - 1.0, "eod"


_REASON = {"sl": "SL", "tp": "TP", "eod": "EOD"}


def evaluate_exit_variants(
    bars: list,
    round_trip_cost: float = ROUND_TRIP_COST_PCT,
) -> Optional[dict]:
    """경로 1 개에 EXIT_VARIANTS 전부 + path 극값을 평가해 ledger 컬럼 dict 반환.

    반환 pct 는 모두 net % (왕복 비용 차감 — realized_pct 와 동일 규약).
    bars 가 비거나 진입가가 무효면 None.
    """
    if not bars:
        return None
    entry = bars[0][0]
    if not np.isfinite(entry) or entry <= 0:
        return None

    out: dict = {}
    for name, spec in EXIT_VARIANTS.items():
        gross, outcome = walk_path(bars, spec.sl, spec.tp)
        if not np.isfinite(gross):
            return None
        net_pct = round((gross - round_trip_cost) * 100, 4)
        out[f"exit_{name}_pct"] = net_pct
        if name != "eod":  # eod 변형은 reason 이 항상 EOD — 컬럼 생략
            out[f"exit_{name}_reason"] = _REASON.get(outcome, outcome.upper())

    highs = [b[1] for b in bars]
    lows = [b[2] for b in bars]
    out["path_max_pct"] = round((max(highs) / entry - 1.0) * 100, 4)
    out["path_min_pct"] = round((min(lows) / entry - 1.0) * 100, 4)
    return out

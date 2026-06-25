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

from ledger.config import FEE_PCT, ROUND_TRIP_COST_PCT, SLIPPAGE_PCT


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


# ============================================================================
# 델타-사다리 청산 (delta-ladder partial exit) — Thorp/BSM dynamic hedging 합성
# ----------------------------------------------------------------------------
# 영상(Veritasium "The Trillion Dollar Equation") 의 dynamic 델타헤지를 현물에서
# 합성: 가격이 오를수록 보유분을 arm 단위로 잘라 익절(이익 잠금)하고, hard floor
# 에서 잔여 전량 청산(프리미엄 상한). exit_rules 6변형 REJECT 의 핵심 실패모드
# ('1트레이드=1청산점' + SL 확대로 14x deep-loss)를 정면 회피 — floor 는 절대
# 안 넓히고 상방에서만 연속 축소. record-only 가상 평가 (자동주문 X).
#
# walk_path 규약 계승: 진입가=첫 봉 open, arm/floor 체결가=entry 기준 정확 비율
# (봉 high 로 부풀리지 않음 — 룩어헤드 차단), 같은 봉 floor 먼저(보수), EOD=마지막
# close. 단일-arm 환원(arms=(tp,), fractions=(1.0,), floor=sl) → walk_path 와
# gross 비트동일 (tests/test_exit_lab.py 가 강제).
#
# 비용 (정직 모델): 업비트 수수료는 명목금액 비례 → 매도를 N tranche 로 쪼개도
# 매도수수료 합 = full 1회와 동일. 'tranche 당 0.15% 고정'은 잘못된 비관(명시 기각).
#   fee_total  = FEE_PCT × 2                         (매수 1 + 매도 Σfrac=1)
#   slip_base  = base_slip × 2
#   slip_extra = max(0, n_sell-1) × extra_slip_per_tranche  (주문별 슬리피지 노브)
# extra=0 이면 왕복비용 = ROUND_TRIP_COST_PCT = 챔피언 TP/SL 과 비트동일(공정비교).
# ============================================================================


@dataclass(frozen=True)
class LadderSpec:
    """사다리 청산 정의. arms 오름차순 양수 비율, fractions 합=1, floor 양수 magnitude."""
    arms: tuple
    fractions: tuple
    floor: Optional[float]
    label: str = ""


def walk_ladder_path(
    bars: list,
    arms: tuple = (0.02, 0.04, 0.06),
    fractions: tuple = (1 / 3, 1 / 3, 1 / 3),
    floor: Optional[float] = 0.05,
) -> tuple[float, str, int]:
    """15m 봉 시간순 walk → (gross_weighted_return, outcome, n_sell_orders).

    진입가 = bars[0][0]. arm 체결 = entry*(1+arm) 정확, floor 잔여 = entry*(1-floor)
    (수익률 -floor), EOD 잔여 = 마지막 봉 close. 같은 봉 floor 먼저(보수) — 봉 high 가
    arm 을 위로 찍었어도 floor 가 같은 봉이면 잔여 전량 -floor (낙관적 룩어헤드 차단).
    이전 봉에서 확정된 tranche 는 보존(floor 가 소급 취소 X).

    outcome ∈ {nodata, floor, partial_floor, tp_full, partial_eod, eod}.
    n_sell_orders = 체결된 매도 주문 수 (비용 분해용; 같은 봉 멀티arm 은 보수적으로 각각 1).
    """
    if not bars:
        return np.nan, "nodata", 0
    entry = bars[0][0]
    if not np.isfinite(entry) or entry <= 0:
        return np.nan, "nodata", 0
    if len(arms) != len(fractions):
        raise ValueError("arms 와 fractions 길이 불일치")
    if any(arms[i] >= arms[i + 1] for i in range(len(arms) - 1)):
        raise ValueError("arms 는 오름차순(strict) 이어야 함")
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions 합이 1.0 이 아님")

    floor_px = entry * (1 - floor) if floor is not None else None
    arm_px = [entry * (1 + a) for a in arms]
    remaining = list(range(len(arms)))
    realized = 0.0
    n_sell = 0
    any_armed = False

    for (o, h, l, c) in bars:
        # 1) floor 먼저 (보수): 잔여가 남아있을 때만, 잔여 전량 -floor 로 청산하고 종료.
        if remaining and floor_px is not None and l <= floor_px:
            for i in remaining:
                realized += fractions[i] * (-floor)
            n_sell += 1
            return realized, ("partial_floor" if any_armed else "floor"), n_sell
        # 2) 멀티arm: 낮은 arm 부터 순차, 각 tranche 는 자기 arm 정확 비율로 잠금.
        for i in remaining[:]:
            if h >= arm_px[i]:
                realized += fractions[i] * arms[i]
                remaining.remove(i)
                n_sell += 1
                any_armed = True
        if not remaining:
            break  # 전량 청산 — 이후 봉 무시 (walk_path 의 즉시 return 과 동치)

    # 3) EOD: 미청산 잔여 = 마지막 봉 close.
    if remaining:
        eod_ret = bars[-1][3] / entry - 1.0
        for i in remaining:
            realized += fractions[i] * eod_ret
        n_sell += 1
        return realized, ("partial_eod" if any_armed else "eod"), n_sell
    return realized, "tp_full", n_sell


def ladder_cost(
    n_sell: int,
    fee_pct: float = FEE_PCT,
    base_slip_pct: float = SLIPPAGE_PCT,
    extra_slip_per_tranche_pct: float = 0.0,
) -> float:
    """사다리 왕복비용 (decimal). fee/slip_base 는 명목비례(분할 무관), slip_extra 만 n_sell 의존.

    n_sell ≤ 1 → ROUND_TRIP_COST_PCT 와 동일 (단일 청산·챔피언과 공정비교 보장).
    """
    fee_total = fee_pct * 2.0
    slip_base = base_slip_pct * 2.0
    slip_extra = max(0, n_sell - 1) * extra_slip_per_tranche_pct
    return fee_total + slip_base + slip_extra


def evaluate_ladder_variant(
    bars: list,
    arms: tuple = (0.02, 0.04, 0.06),
    fractions: tuple = (1 / 3, 1 / 3, 1 / 3),
    floor: Optional[float] = 0.05,
    fee_pct: float = FEE_PCT,
    base_slip_pct: float = SLIPPAGE_PCT,
    extra_slip_per_tranche_pct: float = 0.0,
) -> Optional[dict]:
    """경로 1 개에 사다리 변형을 평가 → net/gross/reason/n_sell/cost dict. bars 무효면 None.

    net = gross_weighted - 비용 (왕복, 명목비례 + 주문별 슬리피지). pct·decimal 모두 제공.
    """
    g, reason, n_sell = walk_ladder_path(bars, arms, fractions, floor)
    if not np.isfinite(g):
        return None
    cost = ladder_cost(n_sell, fee_pct, base_slip_pct, extra_slip_per_tranche_pct)
    net = g - cost
    return {
        "net": net,
        "net_pct": round(net * 100, 4),
        "gross_pct": round(g * 100, 4),
        "reason": reason,
        "n_sell": int(n_sell),
        "cost_pct": round(cost * 100, 4),
    }


# 사다리 outcome → downside_metrics(simulate_path 어휘) 매핑 (지표 호환).
LADDER_OUTCOME_MAP = {
    "floor": "sl",
    "partial_floor": "sl",
    "tp_full": "tp",
    "partial_eod": "eod",
    "eod": "eod",
}

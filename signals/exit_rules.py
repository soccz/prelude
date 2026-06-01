"""Parametrized exit rules — exit-discipline 실험(track B2) 산출물 (self-contained).

★ 판정(2026-06-01, quant-evaluator): **REJECT.** R1 진입 고정 위 청산만 바꿔선 net+하방
  동시개선 불가(6룰 중 0개 만족). vol 은 net 미끼(Δnet 비유의) + deep-loss 14배(유의) →
  하방-우선 사용자엔 독. baseline(R1 -3%SL/+5%TP/EOD)이 최선 → 챔피언 유지, 이 모듈 라이브
  미배선. 남긴 이유: 검증된 leak-free 청산 시뮬레이터(causal, mismatch=0) — 다음 진입-quality
  실험에서 재사용. net lever 는 청산 아닌 진입 quality(pump-then-dump 꼬리)에 있음.

챔피언(R1) 진입은 그대로 두고 **청산만** 바꾸는 청산 함수.
챔피언 close 경로(`scripts/close_recommend_ledger.py` 의 `simulate_path`, -3%SL/+5%TP/EOD)는
이 모듈을 import 하지 않으며 불변이다. 이 모듈은 challenger ledger
(`output/shadow_ledger_recommend_exitsmart.csv`) 청산에만 쓰인다(ops 단계에서 배선).

★ LEAK 방어 (양보불가): 각 봉의 청산 판단은 그 봉 시점까지 정보만 쓴다.
  - 트레일링: 현재까지 running_high (미래 고점 X), arm 도 현재까지 도달여부.
  - time-stop: 경과 봉 수.  vol: 진입 시점(D-1) ATR (진입 결정과 같은 정보집합).
  - 같은 봉 SL·TP 동시 → SL 먼저(보수).  시간 오름차순 walk → 미래 봉 미리보기 0.
  근거 백테스트: scripts/exit_challenger_compare_v1.py (prefix-replay causal 검증 0 mismatch).

거래비용: net = gross - 0.0015 (왕복 0.15%) — 호출측(ledger close)에서 차감.
"""
from __future__ import annotations

import numpy as np

BARS_PER_HOUR = 4         # 15m
CHAMP_SL = 0.03
CHAMP_TP = 0.05

# 기본 exit 룰 = 'baseline'(R1 챔피언 청산 -3%SL/+5%TP/EOD). evaluator 판정(REJECT)에 따라
# 'vol'(net 미끼·비유의 + deep-loss 14배·유의, 하방-우선엔 독)을 기본에서 내림. 이 모듈은
# 라이브 미배선이라 무해하나, 혹시 호출돼도 안전한 baseline 이 기본이 되도록.
DEFAULT_EXIT_RULE = "baseline"


def rule_params(exit_rule: str, regime: str = "", atr_pct: float | None = None) -> dict:
    """exit_rule + (regime, D-1 ATR%) → simulate_exit kwargs. atr_pct 는 leak-free (D-1)."""
    if exit_rule == "baseline":            # 챔피언 청산 (참조용)
        return dict(hard_sl=CHAMP_SL, tp=CHAMP_TP)
    if exit_rule == "trail":               # +3% arm 후 고점-2% 트레일 + 3% floor
        return dict(hard_sl=CHAMP_SL, trail_arm=0.03, trail=0.02, tp=None)
    if exit_rule == "time":                # 챔피언 SL/TP + 첫 1h time-stop
        return dict(hard_sl=CHAMP_SL, tp=CHAMP_TP, time_stop_h=1.0)
    if exit_rule == "vol":                 # SL=clip(2*ATR,1.5%,5%), TP=clip(4*ATR,3%,10%)
        a = atr_pct if (atr_pct is not None and np.isfinite(atr_pct)) else 0.025
        return dict(hard_sl=float(np.clip(2.0 * a, 0.015, 0.05)),
                    tp=float(np.clip(4.0 * a, 0.03, 0.10)))
    if exit_rule == "vol_tight":           # vol 이되 SL cap = 챔피언 3% (deep-loss 차단)
        a = atr_pct if (atr_pct is not None and np.isfinite(atr_pct)) else 0.025
        return dict(hard_sl=float(np.clip(2.0 * a, 0.015, CHAMP_SL)),
                    tp=float(np.clip(4.0 * a, 0.03, 0.10)))
    if exit_rule == "regime":              # regime 별
        if regime == "bear_volatile":
            return dict(hard_sl=0.02, tp=CHAMP_TP, time_stop_h=1.0)
        if regime == "bear_quiet":
            return dict(hard_sl=CHAMP_SL, tp=CHAMP_TP, time_stop_h=2.0)
        return dict(hard_sl=0.04, tp=0.08)
    raise ValueError(f"unknown exit_rule: {exit_rule}")


def simulate_exit(bars, *, hard_sl=None, tp=None, trail_arm=None, trail=None,
                  time_stop_h=None):
    """15m 봉 시간순 walk → (gross_return, outcome, hold_h, low_excursion).

    우선순위 (같은 봉 동시 → 보수=SL 먼저):
      1) hard SL  : low  <= entry*(1-hard_sl)
      2) trail SL : arm(high>=entry*(1+trail_arm)) 발동 후 low <= running_high*(1-trail)
      3) TP       : high >= entry*(1+tp)
      4) time-stop: 진입 후 time_stop_h 경과 봉 종가 청산
    아무것도 안 걸리면 EOD = 마지막 봉 close.
    low_excursion = 진입대비 그날 최저 도달(진단; gross 청산과 별개).
    """
    if not bars:
        return np.nan, "nodata", np.nan, np.nan
    entry = bars[0][0]
    if not np.isfinite(entry) or entry <= 0:
        return np.nan, "nodata", np.nan, np.nan
    sl_px = entry * (1 - hard_sl) if hard_sl is not None else None
    tp_px = entry * (1 + tp) if tp is not None else None
    arm_px = entry * (1 + trail_arm) if (trail is not None and trail_arm is not None) else None
    ts_bars = int(round(time_stop_h * BARS_PER_HOUR)) if time_stop_h is not None else None
    running_high = entry
    running_low = entry
    armed = (trail is not None and trail_arm is None)
    for i, (o, h, l, c) in enumerate(bars):
        running_low = min(running_low, l)
        if h > running_high:
            running_high = h
        if arm_px is not None and not armed and h >= arm_px:
            armed = True
        trail_px = running_high * (1 - trail) if (trail is not None and armed) else None
        if sl_px is not None and l <= sl_px:
            return -hard_sl, "sl", (i + 1) / BARS_PER_HOUR, running_low / entry - 1.0
        if trail_px is not None and l <= trail_px:
            return trail_px / entry - 1.0, "trail", (i + 1) / BARS_PER_HOUR, running_low / entry - 1.0
        if tp_px is not None and h >= tp_px:
            return tp, "tp", (i + 1) / BARS_PER_HOUR, running_low / entry - 1.0
        if ts_bars is not None and (i + 1) >= ts_bars:
            return c / entry - 1.0, "timestop", (i + 1) / BARS_PER_HOUR, running_low / entry - 1.0
    return bars[-1][3] / entry - 1.0, "eod", len(bars) / BARS_PER_HOUR, running_low / entry - 1.0


def simulate_exit_by_rule(bars, exit_rule: str, regime: str = "",
                          atr_pct: float | None = None):
    """exit_rule 이름으로 청산 시뮬 (ledger close 에서 호출하는 진입점)."""
    return simulate_exit(bars, **rule_params(exit_rule, regime, atr_pct))

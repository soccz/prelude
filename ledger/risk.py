"""Kill switch / 리스크 한도 (LEDGER §8).

가상 ledger 한정 — 사용자 본인 매매에는 강제 X.

- 일일 손실 -3% 발동 → 다음 날 자동 침묵 (가상 진입 X)
- MDD -15% 발동 → 7일 cool-down
- 사용자 명시적 "재개" 명령 시 침묵 해제
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import json

from ledger.config import COOLDOWN_DAYS, DAILY_LOSS_LIMIT_PCT, MDD_LIMIT_PCT


# ============================================================================
# Risk state (output/risk_state.json)
# ============================================================================
@dataclass
class RiskState:
    """현재 리스크 상태."""
    is_active: bool = True
    silenced_until: Optional[str] = None     # YYYY-MM-DD
    trigger_reason: Optional[str] = None
    last_daily_pnl_pct: Optional[float] = None
    current_mdd_pct: Optional[float] = None

    @classmethod
    def load(cls, path: Path | str = "output/risk_state.json") -> "RiskState":
        path = Path(path)
        if not path.exists():
            return cls()
        with open(path) as f:
            return cls(**json.load(f))

    def save(self, path: Path | str = "output/risk_state.json"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)


# ============================================================================
# 발동 체크
# ============================================================================
def check_daily_loss(daily_pnl_pct: float) -> bool:
    """일일 가상 PnL -3% 이하면 발동."""
    return daily_pnl_pct <= -DAILY_LOSS_LIMIT_PCT


def check_mdd(current_mdd_pct: float) -> bool:
    """누적 MDD -15% 이하면 발동."""
    return current_mdd_pct <= -MDD_LIMIT_PCT


# ============================================================================
# 발동 처리
# ============================================================================
def trigger_daily_loss(state: RiskState, daily_pnl_pct: float, today: datetime) -> RiskState:
    """일일 한도 발동 → 다음 날 침묵."""
    state.is_active = False
    state.silenced_until = (today + timedelta(days=1)).strftime("%Y-%m-%d")
    state.trigger_reason = f"DAILY_LOSS_LIMIT (-{abs(daily_pnl_pct)*100:.1f}%)"
    state.last_daily_pnl_pct = daily_pnl_pct
    return state


def trigger_mdd(state: RiskState, mdd_pct: float, today: datetime) -> RiskState:
    """MDD 한도 발동 → 7일 cool-down."""
    state.is_active = False
    state.silenced_until = (today + timedelta(days=COOLDOWN_DAYS)).strftime("%Y-%m-%d")
    state.trigger_reason = f"MDD_LIMIT (-{abs(mdd_pct)*100:.1f}%)"
    state.current_mdd_pct = mdd_pct
    return state


def maybe_resume(state: RiskState, today: datetime) -> RiskState:
    """silenced_until 지나면 자동 재개."""
    if state.is_active:
        return state
    if state.silenced_until is None:
        return state
    if today.strftime("%Y-%m-%d") > state.silenced_until:
        state.is_active = True
        state.silenced_until = None
        state.trigger_reason = None
    return state


# ============================================================================
# 종합 (매일 KST 09:30 cron 호출)
# ============================================================================
def evaluate_risk(
    daily_pnl_pct: float,
    current_mdd_pct: float,
    today: Optional[datetime] = None,
    state_path: Path | str = "output/risk_state.json",
) -> RiskState:
    """
    어제 가상 ledger 결과 → risk state 갱신 + 저장.

    return: 갱신된 RiskState
    """
    today = today or datetime.now()
    state = RiskState.load(state_path)
    state = maybe_resume(state, today)

    if check_daily_loss(daily_pnl_pct):
        state = trigger_daily_loss(state, daily_pnl_pct, today)
    elif check_mdd(current_mdd_pct):
        state = trigger_mdd(state, current_mdd_pct, today)

    state.save(state_path)
    return state

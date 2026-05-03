"""Drift detector — 시그널 정확도 + 분포 안정성 감시.

OPS §4 — 매일 KST 00:00 cron 호출.

체크:
  - per-feature IC: 24h vs 7d MA (sign flip / 50% drop)
  - 모델 prediction 분포: 24h vs 7d (KS test)
  - hit rate: 7d vs 30d

발동 시:
  - WARN: 알림 계속하되 가상 사이즈 50% cut
  - FREEZE: 가상 진입 X + 다음 retrain 강제 트리거
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


# ============================================================================
# 임계값 (placeholder, CLAUDE.md §2.5)
# ============================================================================
HALF_DROP_RATIO = 0.5             # 24h IC < 7d MA × 0.5
HIT_RATE_DROP = 0.15              # 7d hit rate < 30d - 0.15
KS_PVALUE_THRESHOLD = 0.01        # 분포 차이 유의

WINDOW_24H = 1
WINDOW_7D = 7
WINDOW_30D = 30


# ============================================================================
# Drift state (output/drift_state.json)
# ============================================================================
@dataclass
class DriftState:
    state: str = "OK"               # 'OK' / 'WARN' / 'FREEZE'
    triggers: list = None
    last_check: Optional[str] = None
    details: dict = None

    def __post_init__(self):
        if self.triggers is None:
            self.triggers = []
        if self.details is None:
            self.details = {}

    @classmethod
    def load(cls, path: Path | str = "output/drift_state.json") -> "DriftState":
        path = Path(path)
        if not path.exists():
            return cls()
        with open(path) as f:
            d = json.load(f)
        return cls(**d)

    def save(self, path: Path | str = "output/drift_state.json"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2, default=str)


# ============================================================================
# IC 계산 (Spearman, P(≥10%) vs 실제 hit)
# ============================================================================
def daily_ic(predictions: pd.DataFrame, actuals: pd.DataFrame, threshold: float = 0.10) -> pd.Series:
    """일별 cross-sectional Spearman IC (P(≥10%) vs actual hit).

    predictions: { date, coin, p_ge_10 }
    actuals:     { date, coin, max_return }

    return: Series indexed by date
    """
    if len(predictions) == 0 or len(actuals) == 0:
        return pd.Series(dtype=float)

    merged = predictions.merge(actuals, on=["date", "coin"], how="inner")
    if len(merged) == 0:
        return pd.Series(dtype=float)

    merged["actual_hit"] = (merged["max_return"] >= threshold).astype(int)

    daily_ics = []
    for date, g in merged.groupby("date"):
        if len(g) < 5:
            continue
        # rank corr (binary 변수라 분산 작아도 OK)
        ic = g[["p_ge_10", "actual_hit"]].corr(method="spearman").iloc[0, 1]
        daily_ics.append({"date": date, "ic": ic})

    return pd.DataFrame(daily_ics).set_index("date")["ic"]


# ============================================================================
# Drift 트리거 체크
# ============================================================================
def check_sign_flip(ic_24h: float, ic_7d_ma: float) -> bool:
    """7d MA 와 24h IC 부호가 바뀌었는가."""
    if pd.isna(ic_24h) or pd.isna(ic_7d_ma):
        return False
    return (ic_24h * ic_7d_ma) < 0


def check_half_drop(ic_24h: float, ic_7d_ma: float, ratio: float = HALF_DROP_RATIO) -> bool:
    """24h IC < 7d MA × ratio (예: 0.5)."""
    if pd.isna(ic_24h) or pd.isna(ic_7d_ma) or ic_7d_ma <= 0:
        return False
    return ic_24h < ic_7d_ma * ratio


def check_hit_rate_drop(
    hit_7d: float, hit_30d: float, threshold: float = HIT_RATE_DROP
) -> bool:
    """7d hit rate < 30d - threshold."""
    if pd.isna(hit_7d) or pd.isna(hit_30d):
        return False
    return hit_7d < hit_30d - threshold


def check_distribution_shift(
    probs_24h: np.ndarray, probs_7d: np.ndarray, p_threshold: float = KS_PVALUE_THRESHOLD
) -> tuple[bool, float]:
    """모델 prediction 분포가 24h vs 7d 차이 유의한가 (KS test)."""
    if len(probs_24h) < 10 or len(probs_7d) < 50:
        return False, 1.0
    stat, p = ks_2samp(probs_24h, probs_7d)
    return p < p_threshold, float(p)


# ============================================================================
# 종합 drift 체크 (매일 KST 00:00 cron)
# ============================================================================
def evaluate_drift(
    ic_history: pd.Series,           # date-indexed IC series
    hit_history: pd.Series,           # date-indexed hit rate (P(≥10%) actual)
    pred_probs_24h: Optional[np.ndarray] = None,
    pred_probs_7d: Optional[np.ndarray] = None,
    asof: Optional[datetime] = None,
    state_path: Path | str = "output/drift_state.json",
) -> DriftState:
    """
    종합 drift 체크 + state 저장.
    """
    asof = asof or datetime.now()
    state = DriftState.load(state_path)
    triggers = []
    details = {}

    # IC 비교 (24h vs 7d MA)
    if len(ic_history) >= WINDOW_7D:
        ic_24h = float(ic_history.iloc[-1]) if len(ic_history) > 0 else np.nan
        ic_7d_ma = float(ic_history.iloc[-WINDOW_7D:].mean())
        details["ic_24h"] = ic_24h
        details["ic_7d_ma"] = ic_7d_ma

        if check_sign_flip(ic_24h, ic_7d_ma):
            triggers.append("SIGN_FLIP")
        if check_half_drop(ic_24h, ic_7d_ma):
            triggers.append("HALF_DROP")

    # Hit rate 비교 (7d vs 30d)
    if len(hit_history) >= WINDOW_30D:
        hit_7d = float(hit_history.iloc[-WINDOW_7D:].mean())
        hit_30d = float(hit_history.iloc[-WINDOW_30D:].mean())
        details["hit_7d"] = hit_7d
        details["hit_30d"] = hit_30d

        if check_hit_rate_drop(hit_7d, hit_30d):
            triggers.append("HIT_RATE_DROP")

    # 분포 shift
    if pred_probs_24h is not None and pred_probs_7d is not None:
        shift, p_value = check_distribution_shift(pred_probs_24h, pred_probs_7d)
        details["dist_ks_pvalue"] = p_value
        if shift:
            triggers.append("DIST_SHIFT")

    # State 결정
    if not triggers:
        state.state = "OK"
    elif "SIGN_FLIP" in triggers or "HALF_DROP" in triggers:
        state.state = "FREEZE"
    else:
        state.state = "WARN"

    state.triggers = triggers
    state.last_check = asof.isoformat()
    state.details = details
    state.save(state_path)
    return state

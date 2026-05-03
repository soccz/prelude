"""가상 포지션 사이징 (LEDGER §2).

옵션:
  - equal_weight: top-K 균등 1/K
  - prob_weighted: P(≥10%) 비례
  - kelly_quarter: quarter-Kelly (이론적, 추정 오차 민감)

BTC regime 따라 max_total_exposure 동적 조정.
per-position cap 적용 (집중 방어).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ledger.config import (
    K_POSITIONS,
    MAX_POSITION_SIZE_PCT,
    MAX_TOTAL_EXPOSURE_PCT,
    REGIME_EXPOSURE_CAP,
    MIN_P_GE_5_FOR_RECOMMEND,
    SIZING_RULE,
)


# ============================================================================
# Top-K 후보 선택 + BTC regime gate
# ============================================================================
def select_candidates(
    predictions: pd.DataFrame,
    btc_regime: str,
    k: int = K_POSITIONS,
    min_p_ge_5: float = MIN_P_GE_5_FOR_RECOMMEND,
) -> pd.DataFrame:
    """
    예측 분포 → top-K 후보.

    1. P(≥+5%) ≥ min_p_ge_5 필터
    2. P(≥+10%) 정렬 기준 top-K
    3. bear_volatile regime → 빈 DataFrame (침묵)
    """
    # bear_volatile 침묵
    if REGIME_EXPOSURE_CAP.get(btc_regime, 0) <= 0:
        return predictions.iloc[0:0]  # empty

    candidates = predictions[predictions["p_ge_5"] >= min_p_ge_5]
    return candidates.sort_values("p_ge_10", ascending=False).head(k)


# ============================================================================
# 사이징 룰
# ============================================================================
def equal_weight(candidates: pd.DataFrame, total_exposure: float) -> pd.DataFrame:
    """K-개 균등 1/K."""
    if len(candidates) == 0:
        return candidates
    out = candidates.copy()
    out["position_size_pct"] = total_exposure / len(candidates)
    return out


def prob_weighted(candidates: pd.DataFrame, total_exposure: float, prob_col: str = "p_ge_10") -> pd.DataFrame:
    """P(≥10%) 비례."""
    if len(candidates) == 0:
        return candidates
    out = candidates.copy()
    weights = out[prob_col].clip(lower=0)
    if weights.sum() == 0:
        out["position_size_pct"] = 0
    else:
        out["position_size_pct"] = (weights / weights.sum()) * total_exposure
    return out


def kelly_quarter(
    candidates: pd.DataFrame,
    total_exposure: float,
    win_prob_col: str = "p_ge_10",
    win_pct: float = 0.10,
    loss_pct: float = 0.05,
) -> pd.DataFrame:
    """
    quarter-Kelly: f* = (p × b - q) / b, 1/4 적용.
    p = win_prob, q = 1-p, b = win/loss ratio.
    """
    if len(candidates) == 0:
        return candidates
    out = candidates.copy()
    p = out[win_prob_col].clip(0.001, 0.999)
    q = 1 - p
    b = win_pct / loss_pct
    f = (p * b - q) / b
    f = f.clip(lower=0) * 0.25  # quarter
    if f.sum() == 0:
        out["position_size_pct"] = 0
    else:
        out["position_size_pct"] = (f / f.sum()) * total_exposure
    return out


# ============================================================================
# Cap 적용
# ============================================================================
def apply_caps(sized: pd.DataFrame, max_per_position: float = MAX_POSITION_SIZE_PCT) -> pd.DataFrame:
    """per-position cap. 초과 분 균등 재분배."""
    if len(sized) == 0:
        return sized

    out = sized.copy()
    over = out["position_size_pct"] > max_per_position
    if not over.any():
        return out

    excess = (out.loc[over, "position_size_pct"] - max_per_position).sum()
    out.loc[over, "position_size_pct"] = max_per_position
    under = ~over
    if under.sum() > 0 and excess > 0:
        # 남은 코인에 균등 재분배 (cap 안 넘는 한도)
        room = (max_per_position - out.loc[under, "position_size_pct"]).clip(lower=0)
        if room.sum() > 0:
            add = (room / room.sum()) * min(excess, room.sum())
            out.loc[under, "position_size_pct"] += add
    return out


# ============================================================================
# Pipeline
# ============================================================================
def size_positions(
    predictions: pd.DataFrame,
    btc_regime: str,
    sizing_rule: str = SIZING_RULE,
    k: int = K_POSITIONS,
) -> pd.DataFrame:
    """
    예측 → top-K 후보 → 사이징 → cap.

    return: { coin, position_size_pct, ... } DataFrame
    """
    candidates = select_candidates(predictions, btc_regime, k=k)
    if len(candidates) == 0:
        return candidates

    total_exposure = REGIME_EXPOSURE_CAP.get(btc_regime, MAX_TOTAL_EXPOSURE_PCT)

    if sizing_rule == "equal":
        sized = equal_weight(candidates, total_exposure)
    elif sizing_rule == "prob_weighted":
        sized = prob_weighted(candidates, total_exposure)
    elif sizing_rule == "kelly_quarter":
        sized = kelly_quarter(candidates, total_exposure)
    else:
        raise ValueError(f"unknown sizing rule: {sizing_rule}")

    sized = apply_caps(sized)
    return sized

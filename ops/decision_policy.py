"""Operational decision policy for idea validation.

The model remains a candidate generator. This layer decides whether each
candidate is an active paper recommendation, watch-only experiment, or silence.
All thresholds here are placeholders and should be judged by net live paper PnL.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from ledger.config import ROUND_TRIP_COST_PCT
from signals.bucket_calibration import BucketCalibrator


ACTIVE = "ACTIVE"
WATCH_ONLY = "WATCH_ONLY"
SILENCE = "SILENCE"
POLICY_ID = "setup_quality_policy_v1"
POLICY_VERSION = "2026-05-25.1"
POLICY_SUMMARY = (
    "Distribution promotes A_TRIPLE/S03-quality candidates by BTC regime and "
    "h6 edge proxy; preopen stays watch-first until live edge improves."
)


def _as_setup_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, str):
        return [x for x in value.replace(",", "+").split("+") if x]
    if isinstance(value, Iterable):
        return [str(x) for x in value if str(x)]
    return []


def setup_quality(primary_setups) -> str:
    """Compact setup tier for experiment attribution."""
    setups = set(_as_setup_list(primary_setups))
    if {"S01", "S02", "S03"}.issubset(setups):
        return "A_TRIPLE"
    if "S03" in setups:
        return "B_S03"
    if setups:
        return "C_PRIMARY"
    return "NO_PRIMARY"


def _calibrated_distribution_h6(row: pd.Series, calibrator: Optional[BucketCalibrator]) -> dict:
    raw = float(row.get("p_h6_hit_5_24h", 0) or 0)
    if calibrator is not None:
        info = calibrator.lookup("h6", raw)
        if info is not None:
            return {
                "calibrated_hit_pct": float(info["actual_hit_pct"]),
                "calibrated_bucket": int(info["bucket"]) + 1,
                "calibration_source": "bucket_h6",
            }
    return {
        "calibrated_hit_pct": raw * 100,
        "calibrated_bucket": np.nan,
        "calibration_source": "raw_h6_fallback",
    }


def _edge_proxy_pct(hit_pct: float, target_pct: float, fail_loss_pct: float) -> float:
    """Simple net EV proxy in percentage points.

    This is intentionally conservative and not a trading guarantee. It is stored
    for attribution and later replacement by path-aware realized distributions.
    """
    p = max(0.0, min(1.0, float(hit_pct) / 100.0))
    return p * target_pct - (1.0 - p) * fail_loss_pct - ROUND_TRIP_COST_PCT * 100


def _distribution_decision(row: pd.Series, btc_regime: str) -> dict:
    q = row["setup_quality"]
    hit = float(row.get("calibrated_hit_pct", 0) or 0)
    edge = float(row.get("expected_edge_pct", 0) or 0)

    if btc_regime == "bear_volatile":
        return {
            "decision": SILENCE,
            "blocked_reason": "btc_bear_volatile",
            "decision_reason": "BTC bear_volatile: risk-off silence",
        }
    if q == "NO_PRIMARY":
        return {
            "decision": SILENCE,
            "blocked_reason": "no_primary_setup",
            "decision_reason": "No S01/S02/S03 primary setup",
        }
    if btc_regime == "bear_quiet":
        if q == "A_TRIPLE" and hit >= 55 and edge >= 0:
            return {
                "decision": ACTIVE,
                "blocked_reason": "",
                "decision_reason": "bear_quiet but A_TRIPLE with positive h6 edge proxy",
            }
        return {
            "decision": WATCH_ONLY,
            "blocked_reason": "bear_quiet_watch",
            "decision_reason": "bear_quiet: record candidate but do not promote unless strongest tier",
        }
    if q in {"A_TRIPLE", "B_S03"} and hit >= 52 and edge >= 0:
        return {
            "decision": ACTIVE,
            "blocked_reason": "",
            "decision_reason": "S03-quality setup with positive h6 edge proxy",
        }
    if q == "C_PRIMARY":
        return {
            "decision": WATCH_ONLY,
            "blocked_reason": "setup_without_s03",
            "decision_reason": "Primary setup exists but lacks S03 quality tier",
        }
    return {
        "decision": WATCH_ONLY,
        "blocked_reason": "edge_below_active_threshold",
        "decision_reason": "Candidate retained for experiment, not active",
    }


def apply_distribution_policy(
    candidates: pd.DataFrame,
    btc_regime: str,
    calibrator: Optional[BucketCalibrator] = None,
) -> pd.DataFrame:
    """Add ACTIVE/WATCH_ONLY/SILENCE metadata to distribution candidates."""
    if len(candidates) == 0:
        return candidates.copy()

    out = candidates.copy()
    if "alert_rank" in out.columns:
        out["source_rank"] = out["alert_rank"]
    else:
        out["source_rank"] = range(1, len(out) + 1)

    primary = out["primary_setups"] if "primary_setups" in out.columns else pd.Series([[]] * len(out), index=out.index)
    out["setup_quality"] = primary.apply(setup_quality)
    if "btc_regime" in out.columns:
        out["btc_regime"] = out["btc_regime"].fillna(btc_regime)
    else:
        out["btc_regime"] = btc_regime

    calib_rows = out.apply(lambda r: _calibrated_distribution_h6(r, calibrator), axis=1)
    for col in ["calibrated_hit_pct", "calibrated_bucket", "calibration_source"]:
        out[col] = [r[col] for r in calib_rows]
    out["expected_edge_pct"] = out["calibrated_hit_pct"].apply(
        lambda x: _edge_proxy_pct(x, target_pct=5.0, fail_loss_pct=5.0)
    )

    decisions = out.apply(lambda r: _distribution_decision(r, str(r.get("btc_regime", btc_regime))), axis=1)
    for col in ["decision", "blocked_reason", "decision_reason"]:
        out[col] = [d[col] for d in decisions]

    out["idea_id"] = out.apply(
        lambda r: f"dist_h6_{str(r['setup_quality']).lower()}_{str(r.get('btc_regime', btc_regime))}_v1",
        axis=1,
    )
    out["policy_id"] = POLICY_ID
    out["policy_version"] = POLICY_VERSION
    return out


def apply_preopen_policy(candidates: pd.DataFrame, btc_regime: str) -> pd.DataFrame:
    """Add ACTIVE/WATCH_ONLY/SILENCE metadata to pre-open candidates."""
    if len(candidates) == 0:
        return candidates.copy()

    out = candidates.copy()
    if "alert_rank" in out.columns:
        out["source_rank"] = out["alert_rank"]
    else:
        out["source_rank"] = range(1, len(out) + 1)
    if "btc_regime" in out.columns:
        out["btc_regime"] = out["btc_regime"].fillna(btc_regime)
    else:
        out["btc_regime"] = btc_regime
    out["setup_quality"] = "PREOPEN"
    out["calibrated_hit_pct"] = out.get("p_first1h_5pct", pd.Series([0] * len(out), index=out.index)).fillna(0) * 100
    out["calibrated_bucket"] = np.nan
    out["calibration_source"] = "raw_first1h_5_fallback"
    out["expected_edge_pct"] = out["calibrated_hit_pct"].apply(
        lambda x: _edge_proxy_pct(x, target_pct=5.0, fail_loss_pct=3.0)
    )

    decisions = []
    for _, r in out.iterrows():
        regime = str(r.get("btc_regime", btc_regime))
        p_1h_5 = float(r.get("p_first1h_5pct", 0) or 0)
        composite = float(r.get("composite", 0) or 0)
        if regime == "bear_volatile":
            d = {
                "decision": SILENCE,
                "blocked_reason": "btc_bear_volatile",
                "decision_reason": "BTC bear_volatile: pre-open silence",
            }
        elif regime == "bear_quiet":
            d = {
                "decision": WATCH_ONLY,
                "blocked_reason": "bear_quiet_watch",
                "decision_reason": "pre-open bear_quiet stays watch-only until live edge improves",
            }
        elif composite >= 1.5 and p_1h_5 >= 0.35:
            d = {
                "decision": ACTIVE,
                "blocked_reason": "",
                "decision_reason": "bull regime with strong first1h_5 and composite",
            }
        else:
            d = {
                "decision": WATCH_ONLY,
                "blocked_reason": "preopen_threshold_watch",
                "decision_reason": "pre-open retained for validation, below active threshold",
            }
        decisions.append(d)

    for col in ["decision", "blocked_reason", "decision_reason"]:
        out[col] = [d[col] for d in decisions]
    out["idea_id"] = out.apply(
        lambda r: f"preopen_first1h5_{str(r.get('btc_regime', btc_regime))}_v1",
        axis=1,
    )
    out["policy_id"] = POLICY_ID
    out["policy_version"] = POLICY_VERSION
    return out


def active_only(decisions: pd.DataFrame) -> pd.DataFrame:
    """Return ACTIVE rows with contiguous alert ranks for paper ledger/Telegram."""
    if len(decisions) == 0 or "decision" not in decisions.columns:
        return decisions.copy()
    active = decisions[decisions["decision"] == ACTIVE].copy()
    active["alert_rank"] = range(1, len(active) + 1)
    return active


def decision_counts(decisions: pd.DataFrame) -> dict:
    if len(decisions) == 0 or "decision" not in decisions.columns:
        return {}
    return {str(k): int(v) for k, v in decisions["decision"].value_counts().sort_index().items()}

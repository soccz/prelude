"""Shadow ledger for non-active idea validation candidates."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SHADOW_LEDGER_COLS = [
    "date", "channel", "coin",
    "decision", "idea_id", "setup_quality", "blocked_reason", "decision_reason",
    "policy_id", "policy_version",
    "base_decision", "meta_policy_id", "meta_policy_version",
    "meta_decision", "meta_reason",
    "trained_model_id", "trained_model_version", "trained_model_status",
    "trained_model_p_win", "trained_model_threshold",
    "confidence_score", "confidence_tier",
    "evidence_key", "evidence_n_closed", "evidence_net_pnl_sum_pct",
    "evidence_avg_net_pnl_pct", "evidence_tp5_hit_rate_pct", "evidence_win_rate_pct",
    "btc_regime", "btc_context", "setup_ids",
    "source_rank", "alert_rank", "composite_score",
    "calibrated_hit_pct", "calibrated_bucket", "calibration_source", "expected_edge_pct",
    "p_h2_3pct_4h", "p_h6_5pct_24h", "p_h5_20pct_tail",
    "p_first15_3pct", "p_first15_5pct",
    "p_first30_3pct", "p_first30_5pct",
    "p_first1h_3pct", "p_first1h_5pct",
    "entry_price_proxy",
    "next_open", "next_high", "next_low", "next_close",
    "next_max_return_pct", "next_min_return_pct", "next_close_return_pct",
    "hit_h2", "hit_h5", "hit_h6",
    "first_open",
    "first_15m_high", "first_30m_high", "first_1h_high",
    "first_15m_low", "first_30m_low", "first_1h_low",
    "first_1h_close",
    "first_1h_max_return_pct", "first_1h_min_return_pct", "first_1h_close_return_pct",
    "hit_first15_3pct", "hit_first15_5pct",
    "hit_first30_3pct", "hit_first30_5pct",
    "hit_first1h_3pct", "hit_first1h_5pct",
    "status", "notes",
]


def _setup_ids(row: pd.Series) -> str:
    value = row.get("primary_setups", row.get("setup_ids", ""))
    if isinstance(value, list):
        return "+".join(str(x) for x in value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value)


def _row_float(row: pd.Series, key: str):
    value = row.get(key, np.nan)
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _records(candidates: pd.DataFrame, asof: pd.Timestamp, channel: str) -> list[dict]:
    rows = []
    for _, r in candidates.iterrows():
        rows.append({
            "date": asof.strftime("%Y-%m-%d"),
            "channel": channel,
            "coin": r.get("market", r.get("coin", "")),
            "decision": r.get("decision", ""),
            "idea_id": r.get("idea_id", ""),
            "setup_quality": r.get("setup_quality", ""),
            "blocked_reason": r.get("blocked_reason", ""),
            "decision_reason": r.get("decision_reason", ""),
            "policy_id": r.get("policy_id", ""),
            "policy_version": r.get("policy_version", ""),
            "base_decision": r.get("base_decision", ""),
            "meta_policy_id": r.get("meta_policy_id", ""),
            "meta_policy_version": r.get("meta_policy_version", ""),
            "meta_decision": r.get("meta_decision", r.get("decision", "")),
            "meta_reason": r.get("meta_reason", ""),
            "trained_model_id": r.get("trained_model_id", ""),
            "trained_model_version": r.get("trained_model_version", ""),
            "trained_model_status": r.get("trained_model_status", ""),
            "trained_model_p_win": _row_float(r, "trained_model_p_win"),
            "trained_model_threshold": _row_float(r, "trained_model_threshold"),
            "confidence_score": _row_float(r, "confidence_score"),
            "confidence_tier": r.get("confidence_tier", ""),
            "evidence_key": r.get("evidence_key", ""),
            "evidence_n_closed": _row_float(r, "evidence_n_closed"),
            "evidence_net_pnl_sum_pct": _row_float(r, "evidence_net_pnl_sum_pct"),
            "evidence_avg_net_pnl_pct": _row_float(r, "evidence_avg_net_pnl_pct"),
            "evidence_tp5_hit_rate_pct": _row_float(r, "evidence_tp5_hit_rate_pct"),
            "evidence_win_rate_pct": _row_float(r, "evidence_win_rate_pct"),
            "btc_regime": r.get("btc_regime", "unknown"),
            "btc_context": r.get("btc_context", "—"),
            "setup_ids": _setup_ids(r),
            "source_rank": _row_float(r, "source_rank"),
            "alert_rank": _row_float(r, "alert_rank"),
            "composite_score": _row_float(r, "composite"),
            "calibrated_hit_pct": _row_float(r, "calibrated_hit_pct"),
            "calibrated_bucket": _row_float(r, "calibrated_bucket"),
            "calibration_source": r.get("calibration_source", ""),
            "expected_edge_pct": _row_float(r, "expected_edge_pct"),
            "p_h2_3pct_4h": _row_float(r, "p_h2_hit_3_4h"),
            "p_h6_5pct_24h": _row_float(r, "p_h6_hit_5_24h"),
            "p_h5_20pct_tail": _row_float(r, "p_h5_tail_20"),
            "p_first15_3pct": _row_float(r, "p_first15_3pct"),
            "p_first15_5pct": _row_float(r, "p_first15_5pct"),
            "p_first30_3pct": _row_float(r, "p_first30_3pct"),
            "p_first30_5pct": _row_float(r, "p_first30_5pct"),
            "p_first1h_3pct": _row_float(r, "p_first1h_3pct"),
            "p_first1h_5pct": _row_float(r, "p_first1h_5pct"),
            "entry_price_proxy": _row_float(r, "close"),
            "next_open": np.nan,
            "next_high": np.nan,
            "next_low": np.nan,
            "next_close": np.nan,
            "next_max_return_pct": np.nan,
            "next_min_return_pct": np.nan,
            "next_close_return_pct": np.nan,
            "hit_h2": np.nan,
            "hit_h5": np.nan,
            "hit_h6": np.nan,
            "first_open": np.nan,
            "first_15m_high": np.nan,
            "first_30m_high": np.nan,
            "first_1h_high": np.nan,
            "first_15m_low": np.nan,
            "first_30m_low": np.nan,
            "first_1h_low": np.nan,
            "first_1h_close": np.nan,
            "first_1h_max_return_pct": np.nan,
            "first_1h_min_return_pct": np.nan,
            "first_1h_close_return_pct": np.nan,
            "hit_first15_3pct": np.nan,
            "hit_first15_5pct": np.nan,
            "hit_first30_3pct": np.nan,
            "hit_first30_5pct": np.nan,
            "hit_first1h_3pct": np.nan,
            "hit_first1h_5pct": np.nan,
            # "entered" means close-out pending for an observed candidate, not a real trade.
            "status": "entered",
            "notes": "",
        })
    return rows


def append_shadow_ledger(
    candidates: pd.DataFrame,
    asof: pd.Timestamp,
    ledger_path: str,
    channel: str,
) -> int:
    """Append one shadow snapshot per date/channel."""
    if len(candidates) == 0:
        return 0

    new_df = pd.DataFrame(_records(candidates, asof, channel))[SHADOW_LEDGER_COLS]
    p = Path(ledger_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if p.exists():
        existing = pd.read_csv(p)
        if "channel" not in existing.columns:
            existing["channel"] = channel
        same_snapshot = (
            (existing["date"].astype(str) == asof.strftime("%Y-%m-%d"))
            & (existing["channel"].astype(str) == channel)
        )
        if bool(same_snapshot.any()):
            return 0
        combined = pd.concat([existing, new_df], ignore_index=True)
        ordered = [c for c in SHADOW_LEDGER_COLS if c in combined.columns]
        extras = [c for c in combined.columns if c not in ordered]
        combined[ordered + extras].to_csv(p, index=False)
        return len(new_df)

    new_df.to_csv(p, index=False)
    return len(new_df)

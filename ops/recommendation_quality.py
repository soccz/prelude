"""Recommendation quality meta-filter.

This layer uses only historical closed paper/shadow ledger rows. It does not
create new model scores; it decides whether an already generated ACTIVE
candidate has enough realized evidence to stay ACTIVE.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ledger.config import ROUND_TRIP_COST_PCT
from ops.decision_policy import ACTIVE, WATCH_ONLY, setup_quality


MIN_META_CLOSED = 8
STRONG_META_CLOSED = 20
META_POLICY_ID = "recommendation_quality_meta_v1"
META_POLICY_VERSION = "2026-05-25.1"
DEFAULT_META_MODEL_DIR = "signals/models/ckpt/recommendation_quality_v1"

NUMERIC_META_FEATURES = [
    "expected_edge_pct", "calibrated_hit_pct", "source_rank", "alert_rank",
    "composite_score",
    "p_h2_3pct_4h", "p_h6_5pct_24h", "p_h5_20pct_tail",
    "p_first15_3pct", "p_first15_5pct",
    "p_first30_3pct", "p_first30_5pct",
    "p_first1h_3pct", "p_first1h_5pct",
    "log_return_1d_pct", "atr_pct_14_pct",
    "return_5d_rank", "return_7d_rank", "vol_5d_rank", "roc_3d_rank",
    "vol5_rank", "return5_rank",
]
CATEGORICAL_META_FEATURES = [
    "channel", "setup_quality", "btc_regime", "btc_context", "calibration_source",
]
META_FEATURES = NUMERIC_META_FEATURES + CATEGORICAL_META_FEATURES
FEATURE_ALIASES = {
    "composite_score": ["composite_score", "composite"],
    "p_h2_3pct_4h": ["p_h2_3pct_4h", "p_h2_hit_3_4h"],
    "p_h6_5pct_24h": ["p_h6_5pct_24h", "p_h6_hit_5_24h"],
    "p_h5_20pct_tail": ["p_h5_20pct_tail", "p_h5_tail_20"],
    "return_5d_rank": ["return_5d_rank", "return_5d"],
    "return_7d_rank": ["return_7d_rank", "return_7d"],
    "vol_5d_rank": ["vol_5d_rank", "vol_5d"],
    "roc_3d_rank": ["roc_3d_rank", "roc_3d"],
    "vol5_rank": ["vol5_rank", "vol_5d"],
    "return5_rank": ["return5_rank", "return_5d"],
}

META_COLS = [
    "base_decision",
    "meta_policy_id", "meta_policy_version",
    "meta_decision", "meta_reason",
    "trained_model_id", "trained_model_version", "trained_model_status",
    "trained_model_p_win", "trained_model_threshold",
    "confidence_score", "confidence_tier",
    "evidence_key", "evidence_n_closed",
    "evidence_net_pnl_sum_pct", "evidence_avg_net_pnl_pct",
    "evidence_tp5_hit_rate_pct", "evidence_win_rate_pct",
]


@dataclass(frozen=True)
class Evidence:
    key: str = "none"
    n_closed: int = 0
    net_pnl_sum_pct: float = np.nan
    avg_net_pnl_pct: float = np.nan
    tp5_hit_rate_pct: float = np.nan
    win_rate_pct: float = np.nan


def build_meta_feature_frame(df: pd.DataFrame, channel: str | None = None) -> pd.DataFrame:
    """Build aligned train/inference features for the recommendation meta model."""
    out = pd.DataFrame(index=df.index)
    for col in NUMERIC_META_FEATURES:
        aliases = FEATURE_ALIASES.get(col, [col])
        series = None
        for alias in aliases:
            if alias in df.columns:
                series = df[alias]
                break
        out[col] = pd.to_numeric(series, errors="coerce") if series is not None else np.nan

    for col in CATEGORICAL_META_FEATURES:
        aliases = FEATURE_ALIASES.get(col, [col])
        series = None
        for alias in aliases:
            if alias in df.columns:
                series = df[alias]
                break
        if series is None:
            if col == "channel" and channel is not None:
                out[col] = channel
            else:
                out[col] = "unknown"
        else:
            out[col] = series.fillna("unknown").astype(str)
    if channel is not None:
        out["channel"] = out["channel"].replace("", channel).fillna(channel)
    return out[META_FEATURES]


def load_trained_meta_model(model_dir: str | Path = DEFAULT_META_MODEL_DIR) -> dict | None:
    """Load trained meta-label artifact if present."""
    p = Path(model_dir)
    meta_path = p / "meta.json"
    model_path = p / "model.pkl"
    if not meta_path.exists() or not model_path.exists():
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return {"meta": meta, "model": model}


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(v):
        return default
    return v


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, float) and np.isnan(value):
        return default
    text = str(value)
    return text if text else default


def _net_pnl_pct(max_ret_pct: Any, close_ret_pct: Any) -> float:
    max_ret = _safe_float(max_ret_pct)
    close_ret = _safe_float(close_ret_pct)
    if np.isnan(max_ret):
        return np.nan
    if max_ret >= 5.0:
        gross = 5.0
    elif np.isnan(close_ret):
        return np.nan
    else:
        gross = close_ret
    return gross - ROUND_TRIP_COST_PCT * 100


def _load_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def _legacy_setup(row: pd.Series, channel: str) -> str:
    existing = _safe_str(row.get("setup_quality"))
    if existing:
        return existing
    if channel == "preopen":
        return "PREOPEN"
    return setup_quality(_safe_str(row.get("setup_ids")).split("+"))


def _legacy_idea(row: pd.Series, channel: str) -> str:
    existing = _safe_str(row.get("idea_id"))
    if existing:
        return existing
    q = str(row.get("setup_quality", "") or _legacy_setup(row, channel)).lower()
    regime = _safe_str(row.get("btc_regime"), "unknown")
    return f"legacy_{channel}_{q}_{regime}_v1"


def normalize_history(df: pd.DataFrame, channel: str, source: str) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame()
    out = df.copy()
    out["channel"] = out.get("channel", channel)
    out["channel"] = out["channel"].fillna(channel).astype(str)
    out["source"] = source
    out["date"] = out["date"].astype(str)
    out["coin"] = out["coin"].astype(str)
    out["decision"] = out.get("decision", ACTIVE)
    out["decision"] = out["decision"].fillna(ACTIVE).astype(str)
    out["btc_regime"] = out.get("btc_regime", "unknown")
    out["btc_regime"] = out["btc_regime"].fillna("unknown").astype(str)
    out["status"] = out.get("status", "")
    out["status"] = out["status"].fillna("").astype(str)
    out["setup_quality"] = out.apply(lambda r: _legacy_setup(r, channel), axis=1)
    out["idea_id"] = out.apply(lambda r: _legacy_idea(r, channel), axis=1)
    out["date_dt"] = pd.to_datetime(out["date"], errors="coerce")

    if channel == "distribution":
        out["max_return_pct"] = pd.to_numeric(out.get("next_max_return_pct"), errors="coerce")
        out["close_return_pct"] = pd.to_numeric(out.get("next_close_return_pct"), errors="coerce")
    else:
        out["max_return_pct"] = pd.to_numeric(out.get("first_1h_max_return_pct"), errors="coerce")
        out["close_return_pct"] = pd.to_numeric(out.get("first_1h_close_return_pct"), errors="coerce")

    out["net_pnl_pct"] = out.apply(
        lambda r: _net_pnl_pct(r.get("max_return_pct"), r.get("close_return_pct")),
        axis=1,
    )
    out["tp5_hit"] = np.where(out["max_return_pct"].notna(), out["max_return_pct"] >= 5.0, np.nan)
    out["net_win"] = np.where(out["net_pnl_pct"].notna(), out["net_pnl_pct"] > 0, np.nan)
    return out


def load_channel_history(paper_ledger: str, shadow_ledger: str, channel: str) -> pd.DataFrame:
    """Load one channel's paper + shadow rows without double-counting duplicates."""
    paper = normalize_history(_load_csv(paper_ledger), channel, "paper")
    shadow = normalize_history(_load_csv(shadow_ledger), channel, "shadow")
    hist = pd.concat([shadow, paper], ignore_index=True, sort=False)
    if len(hist) == 0:
        return hist

    hist["_closed_priority"] = hist["net_pnl_pct"].notna().astype(int)
    hist["_source_priority"] = hist["source"].map({"shadow": 1, "paper": 0}).fillna(0)
    hist = hist.sort_values(
        ["date", "channel", "coin", "_closed_priority", "_source_priority"],
        ascending=[True, True, True, False, False],
    )
    hist = hist.drop_duplicates(["date", "channel", "coin"], keep="first")
    return hist.drop(columns=["_closed_priority", "_source_priority"], errors="ignore")


def _closed_before(history: pd.DataFrame, asof: pd.Timestamp, channel: str) -> pd.DataFrame:
    if len(history) == 0:
        return history
    cutoff = pd.Timestamp(asof).normalize()
    closed = history[
        (history["channel"].astype(str) == channel)
        & history["date_dt"].notna()
        & (history["date_dt"] < cutoff)
        & history["net_pnl_pct"].notna()
    ].copy()
    return closed


def _summarize(grp: pd.DataFrame, key: str) -> Evidence:
    n = int(len(grp))
    if n == 0:
        return Evidence(key=key)
    return Evidence(
        key=key,
        n_closed=n,
        net_pnl_sum_pct=float(grp["net_pnl_pct"].sum()),
        avg_net_pnl_pct=float(grp["net_pnl_pct"].mean()),
        tp5_hit_rate_pct=float(grp["tp5_hit"].mean() * 100),
        win_rate_pct=float(grp["net_win"].mean() * 100),
    )


def _key_values(row: pd.Series, channel: str) -> list[tuple[str, tuple]]:
    idea = _safe_str(row.get("idea_id"))
    setup_q = _safe_str(row.get("setup_quality"), "UNKNOWN")
    regime = _safe_str(row.get("btc_regime"), "unknown")
    keys = []
    if idea:
        keys.append(("idea", (channel, idea)))
    keys.extend([
        ("setup_regime", (channel, setup_q, regime)),
        ("setup", (channel, setup_q)),
        ("channel", (channel,)),
    ])
    return keys


def _evidence_maps(closed: pd.DataFrame) -> dict[str, dict[tuple, Evidence]]:
    maps: dict[str, dict[tuple, Evidence]] = {
        "idea": {},
        "setup_regime": {},
        "setup": {},
        "channel": {},
    }
    if len(closed) == 0:
        return maps

    for key, grp in closed.groupby(["channel", "idea_id"], dropna=False):
        maps["idea"][tuple(key)] = _summarize(grp, f"idea:{'|'.join(map(str, key))}")
    for key, grp in closed.groupby(["channel", "setup_quality", "btc_regime"], dropna=False):
        maps["setup_regime"][tuple(key)] = _summarize(grp, f"setup_regime:{'|'.join(map(str, key))}")
    for key, grp in closed.groupby(["channel", "setup_quality"], dropna=False):
        maps["setup"][tuple(key)] = _summarize(grp, f"setup:{'|'.join(map(str, key))}")
    for key, grp in closed.groupby(["channel"], dropna=False):
        key_tuple = (key,) if not isinstance(key, tuple) else tuple(key)
        maps["channel"][key_tuple] = _summarize(grp, f"channel:{'|'.join(map(str, key_tuple))}")
    return maps


def _select_evidence(row: pd.Series, maps: dict[str, dict[tuple, Evidence]], channel: str) -> Evidence:
    fallback = Evidence()
    for level, values in _key_values(row, channel):
        ev = maps.get(level, {}).get(values)
        if ev is None:
            continue
        if ev.n_closed >= MIN_META_CLOSED:
            return ev
        if ev.n_closed > fallback.n_closed:
            fallback = ev
    return fallback


def _confidence(edge_pct: float, ev: Evidence, base_decision: str) -> tuple[float, str, str]:
    if base_decision != ACTIVE:
        return 30.0, "NOT_ACTIVE", "base policy is not active"
    edge_bonus = max(-10.0, min(15.0, _safe_float(edge_pct, 0.0) * 3.0))
    if ev.n_closed == 0:
        return 52.0 + edge_bonus, "COLLECT", "no closed evidence yet"
    if ev.n_closed < MIN_META_CLOSED:
        return 55.0 + edge_bonus, "LOW_SAMPLE", f"only {ev.n_closed} closed evidence rows"
    if ev.avg_net_pnl_pct < -0.25 and ev.net_pnl_sum_pct < 0:
        return 25.0 + edge_bonus, "DOWNRANK", "historical matched evidence is negative"
    if ev.tp5_hit_rate_pct < 35.0 and ev.avg_net_pnl_pct < 0:
        return 30.0 + edge_bonus, "DOWNRANK", "matched evidence has low TP5 hit rate and negative avg"
    if ev.n_closed >= STRONG_META_CLOSED and ev.avg_net_pnl_pct > 0 and ev.tp5_hit_rate_pct >= 45.0:
        return 85.0 + edge_bonus, "HIGH", "matched evidence is positive with enough samples"
    if ev.avg_net_pnl_pct > 0:
        return 72.0 + edge_bonus, "EVIDENCE_OK", "matched evidence is positive"
    return 58.0 + edge_bonus, "WATCH_EVIDENCE", "matched evidence is mixed"


def _predict_trained_model(row: pd.Series, channel: str, trained_model: dict | None) -> dict:
    if not trained_model:
        return {
            "trained_model_id": "",
            "trained_model_version": "",
            "trained_model_status": "MISSING",
            "trained_model_p_win": np.nan,
            "trained_model_threshold": np.nan,
        }
    meta = trained_model.get("meta", {})
    X = build_meta_feature_frame(pd.DataFrame([row.to_dict()]), channel=channel)
    model = trained_model["model"]
    p = float(model.predict_proba(X)[:, 1][0])
    status = "DEPLOYED" if bool(meta.get("deployable", False)) else "COLLECT"
    return {
        "trained_model_id": meta.get("model_id", "recommendation_quality_v1"),
        "trained_model_version": meta.get("model_version", ""),
        "trained_model_status": status,
        "trained_model_p_win": p,
        "trained_model_threshold": _safe_float(meta.get("threshold"), 0.5),
    }


def apply_recommendation_quality(
    decisions: pd.DataFrame,
    history: pd.DataFrame,
    asof: pd.Timestamp,
    channel: str,
    *,
    enabled: bool = True,
    trained_model: dict | None = None,
) -> pd.DataFrame:
    """Attach confidence metadata and demote weak historical groups."""
    out = decisions.copy()
    if len(out) == 0:
        return out

    out["base_decision"] = out.get("decision", "")
    if not enabled:
        out["meta_policy_id"] = META_POLICY_ID
        out["meta_policy_version"] = META_POLICY_VERSION
        out["meta_decision"] = out["decision"]
        out["meta_reason"] = "meta filter disabled"
        out["trained_model_id"] = ""
        out["trained_model_version"] = ""
        out["trained_model_status"] = "DISABLED"
        out["trained_model_p_win"] = np.nan
        out["trained_model_threshold"] = np.nan
        out["confidence_score"] = np.nan
        out["confidence_tier"] = "DISABLED"
        out["evidence_key"] = "disabled"
        out["evidence_n_closed"] = 0
        out["evidence_net_pnl_sum_pct"] = np.nan
        out["evidence_avg_net_pnl_pct"] = np.nan
        out["evidence_tp5_hit_rate_pct"] = np.nan
        out["evidence_win_rate_pct"] = np.nan
        return out

    closed = _closed_before(history, asof, channel)
    maps = _evidence_maps(closed)
    meta_rows = []
    final_decisions = []
    final_blocked = []
    final_reasons = []

    for _, row in out.iterrows():
        base = _safe_str(row.get("decision"))
        ev = _select_evidence(row, maps, channel)
        score, tier, reason = _confidence(row.get("expected_edge_pct"), ev, base)
        model_info = _predict_trained_model(row, channel, trained_model)
        final = base
        blocked = _safe_str(row.get("blocked_reason"))
        decision_reason = _safe_str(row.get("decision_reason"))

        model_p = _safe_float(model_info["trained_model_p_win"])
        model_threshold = _safe_float(model_info["trained_model_threshold"])
        if (
            base == ACTIVE
            and model_info["trained_model_status"] == "DEPLOYED"
            and not np.isnan(model_p)
            and not np.isnan(model_threshold)
        ):
            score = model_p * 100
            if model_p < model_threshold:
                tier = "MODEL_DOWNRANK"
                reason = f"trained meta model p_win {model_p:.2f} below threshold {model_threshold:.2f}"
            else:
                tier = "MODEL_OK"
                reason = f"trained meta model p_win {model_p:.2f} above threshold {model_threshold:.2f}"

        if base == ACTIVE and tier == "DOWNRANK":
            final = WATCH_ONLY
            blocked = "meta_filter_negative_evidence"
            decision_reason = f"{decision_reason}; meta: {reason}" if decision_reason else f"meta: {reason}"
        elif base == ACTIVE and tier == "MODEL_DOWNRANK":
            final = WATCH_ONLY
            blocked = "trained_meta_model_downrank"
            decision_reason = f"{decision_reason}; meta: {reason}" if decision_reason else f"meta: {reason}"
        elif base == ACTIVE:
            decision_reason = f"{decision_reason}; meta: {reason}" if decision_reason else f"meta: {reason}"

        final_decisions.append(final)
        final_blocked.append(blocked)
        final_reasons.append(decision_reason)
        meta_rows.append({
            "meta_policy_id": META_POLICY_ID,
            "meta_policy_version": META_POLICY_VERSION,
            "meta_decision": final,
            "meta_reason": reason,
            **model_info,
            "confidence_score": round(max(0.0, min(100.0, score)), 2),
            "confidence_tier": tier,
            "evidence_key": ev.key,
            "evidence_n_closed": int(ev.n_closed),
            "evidence_net_pnl_sum_pct": ev.net_pnl_sum_pct,
            "evidence_avg_net_pnl_pct": ev.avg_net_pnl_pct,
            "evidence_tp5_hit_rate_pct": ev.tp5_hit_rate_pct,
            "evidence_win_rate_pct": ev.win_rate_pct,
        })

    meta_df = pd.DataFrame(meta_rows, index=out.index)
    for col in meta_df.columns:
        out[col] = meta_df[col]
    out["decision"] = final_decisions
    out["blocked_reason"] = final_blocked
    out["decision_reason"] = final_reasons
    return out

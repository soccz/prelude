"""Idea validation report for ACTIVE/WATCH_ONLY/SILENCE candidates.

This is the portfolio/experiment layer: combine active paper ledgers and shadow
candidate ledgers, compute net TP5-style results, and attribute performance by
idea_id, decision, setup quality, and BTC regime.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ledger.config import ROUND_TRIP_COST_PCT
from ops.decision_policy import ACTIVE, SILENCE, WATCH_ONLY, setup_quality
from ops.policy_gate import evaluate_policy_gate
from ops.recommendation_quality import DEFAULT_META_MODEL_DIR


DIST_REALIZED_COLS = [
    "next_open", "next_high", "next_low", "next_close",
    "next_max_return_pct", "next_min_return_pct", "next_close_return_pct",
    "hit_h2", "hit_h5", "hit_h6", "status", "notes",
]
PREOPEN_REALIZED_COLS = [
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
MERGE_KEY = ["date", "channel", "coin"]


def _load_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def _safe_str(value, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, float) and np.isnan(value):
        return default
    s = str(value)
    return s if s else default


def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = np.nan
    return out


def _legacy_setup_quality(row: pd.Series, channel: str) -> str:
    if "setup_quality" in row and _safe_str(row.get("setup_quality")):
        return _safe_str(row.get("setup_quality"))
    if channel == "preopen":
        return "PREOPEN"
    return setup_quality(_safe_str(row.get("setup_ids")).split("+"))


def _legacy_idea_id(row: pd.Series, channel: str) -> str:
    existing = _safe_str(row.get("idea_id"))
    if existing:
        return existing
    q = str(row.get("setup_quality", "")).lower() or "unknown"
    regime = _safe_str(row.get("btc_regime"), "unknown")
    if channel == "preopen":
        return f"legacy_preopen_{regime}_v1"
    return f"legacy_distribution_{q}_{regime}_v1"


def normalize_ledger(df: pd.DataFrame, channel: str, source: str) -> pd.DataFrame:
    """Normalize old/new paper or shadow ledger rows to one candidate schema."""
    if len(df) == 0:
        return pd.DataFrame()

    realized_cols = DIST_REALIZED_COLS if channel == "distribution" else PREOPEN_REALIZED_COLS
    out = _ensure_cols(df, realized_cols).copy()
    out["channel"] = out.get("channel", channel)
    out["channel"] = out["channel"].fillna(channel).astype(str)
    out["source"] = source
    out["date"] = out["date"].astype(str)
    out["coin"] = out["coin"].astype(str)
    out["decision"] = out.get("decision", ACTIVE)
    out["decision"] = out["decision"].fillna(ACTIVE).astype(str)
    out["btc_regime"] = out.get("btc_regime", "unknown")
    out["btc_regime"] = out["btc_regime"].fillna("unknown").astype(str)
    out["status"] = out["status"].fillna("").astype(str)
    out["setup_quality"] = out.apply(lambda r: _legacy_setup_quality(r, channel), axis=1)
    out["idea_id"] = out.apply(lambda r: _legacy_idea_id(r, channel), axis=1)
    if "blocked_reason" not in out.columns:
        out["blocked_reason"] = ""
    if "decision_reason" not in out.columns:
        out["decision_reason"] = ""
    return out


def _fill_realized_from_paper(shadow: pd.DataFrame, paper: pd.DataFrame, channel: str) -> pd.DataFrame:
    """Prefer shadow metadata, but fill realized fields from matching active paper rows."""
    if len(shadow) == 0 or len(paper) == 0:
        return shadow

    realized_cols = DIST_REALIZED_COLS if channel == "distribution" else PREOPEN_REALIZED_COLS
    paper_idx = paper.drop_duplicates(MERGE_KEY).set_index(MERGE_KEY)
    out = shadow.copy()
    for idx, row in out.iterrows():
        key = tuple(row[k] for k in MERGE_KEY)
        if key not in paper_idx.index:
            continue
        p_row = paper_idx.loc[key]
        for col in realized_cols:
            current = row.get(col, np.nan)
            if col == "status" and current != "closed" and p_row.get(col, current) == "closed":
                out.at[idx, col] = p_row.get(col, current)
            elif pd.isna(current) or current == "":
                out.at[idx, col] = p_row.get(col, current)
    return out


def combine_channel(paper: pd.DataFrame, shadow: pd.DataFrame, channel: str) -> pd.DataFrame:
    """Combine shadow candidates with legacy active paper rows without duplicates."""
    paper_n = normalize_ledger(paper, channel=channel, source="paper")
    shadow_n = normalize_ledger(shadow, channel=channel, source="shadow")
    if len(shadow_n) == 0:
        return paper_n

    shadow_n = _fill_realized_from_paper(shadow_n, paper_n, channel)
    shadow_keys = set(map(tuple, shadow_n[MERGE_KEY].astype(str).values.tolist()))
    legacy_paper = paper_n[
        ~paper_n[MERGE_KEY].astype(str).apply(lambda r: tuple(r), axis=1).isin(shadow_keys)
    ]
    return pd.concat([shadow_n, legacy_paper], ignore_index=True, sort=False)


def load_candidate_ledger(args) -> pd.DataFrame:
    dist = combine_channel(
        _load_csv(args.paper_ledger),
        _load_csv(args.shadow_ledger_distribution),
        "distribution",
    )
    preopen = combine_channel(
        _load_csv(args.paper_ledger_preopen),
        _load_csv(args.shadow_ledger_preopen),
        "preopen",
    )
    return pd.concat([dist, preopen], ignore_index=True, sort=False)


def _net_pnl_pct(max_ret_pct, close_ret_pct) -> float:
    max_ret = pd.to_numeric(max_ret_pct, errors="coerce")
    close_ret = pd.to_numeric(close_ret_pct, errors="coerce")
    if pd.isna(max_ret):
        return np.nan
    if max_ret >= 5.0:
        gross = 5.0
    elif pd.isna(close_ret):
        return np.nan
    else:
        gross = float(close_ret)
    return gross - ROUND_TRIP_COST_PCT * 100


def add_result_columns(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return df.copy()
    out = df.copy()
    out["date_dt"] = pd.to_datetime(out["date"], errors="coerce")
    out["is_closed"] = out["status"].astype(str).eq("closed")
    out["net_pnl_pct"] = np.nan
    out["tp5_hit"] = np.nan
    out["max_return_pct"] = np.nan
    out["min_return_pct"] = np.nan
    out["close_return_pct"] = np.nan
    out["policy_decision_v1"] = out.apply(policy_replay_decision, axis=1)

    dist = out["channel"].eq("distribution")
    pre = out["channel"].eq("preopen")

    if dist.any():
        out.loc[dist, "max_return_pct"] = pd.to_numeric(out.loc[dist, "next_max_return_pct"], errors="coerce")
        out.loc[dist, "min_return_pct"] = pd.to_numeric(out.loc[dist, "next_min_return_pct"], errors="coerce")
        out.loc[dist, "close_return_pct"] = pd.to_numeric(out.loc[dist, "next_close_return_pct"], errors="coerce")
    if pre.any():
        out.loc[pre, "max_return_pct"] = pd.to_numeric(out.loc[pre, "first_1h_max_return_pct"], errors="coerce")
        out.loc[pre, "min_return_pct"] = pd.to_numeric(out.loc[pre, "first_1h_min_return_pct"], errors="coerce")
        out.loc[pre, "close_return_pct"] = pd.to_numeric(out.loc[pre, "first_1h_close_return_pct"], errors="coerce")

    valid = out["is_closed"] & out["max_return_pct"].notna()
    out.loc[valid, "tp5_hit"] = (out.loc[valid, "max_return_pct"] >= 5.0).astype(int)
    out.loc[valid, "net_pnl_pct"] = out.loc[valid].apply(
        lambda r: _net_pnl_pct(r["max_return_pct"], r["close_return_pct"]),
        axis=1,
    )
    out.loc[valid, "net_win"] = (out.loc[valid, "net_pnl_pct"] > 0).astype(int)
    return out


def policy_replay_decision(row: pd.Series) -> str:
    """Replay the current setup-quality policy on historical candidate rows.

    This uses only decision-time metadata already in the ledger: channel, setup
    tier, BTC regime, composite score, and model head scores where present.
    """
    channel = _safe_str(row.get("channel"))
    regime = _safe_str(row.get("btc_regime"), "unknown")
    setup_q = _safe_str(row.get("setup_quality"), "NO_PRIMARY")

    if channel == "distribution":
        if regime == "bear_volatile":
            return SILENCE
        if setup_q == "A_TRIPLE":
            return ACTIVE
        if setup_q in {"B_S03", "C_PRIMARY"}:
            return WATCH_ONLY
        return SILENCE

    if channel == "preopen":
        if regime == "bear_volatile":
            return SILENCE
        if regime == "bear_quiet":
            return WATCH_ONLY
        composite = pd.to_numeric(row.get("composite_score", row.get("composite", np.nan)), errors="coerce")
        p_1h_5 = pd.to_numeric(row.get("p_first1h_5pct", np.nan), errors="coerce")
        if pd.notna(composite) and pd.notna(p_1h_5) and composite >= 1.5 and p_1h_5 >= 0.35:
            return ACTIVE
        return WATCH_ONLY

    return WATCH_ONLY


def _maybe_float(value, digits: int = 4):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(v) or np.isinf(v):
        return None
    return round(v, digits)


def _load_meta_model_card(model_dir: str | Path = DEFAULT_META_MODEL_DIR) -> dict:
    meta_path = Path(model_dir) / "meta.json"
    if not meta_path.exists():
        return {"available": False, "reason": "trained recommendation meta model not found"}
    with open(meta_path) as f:
        meta = json.load(f)
    return {
        "available": True,
        "model_id": meta.get("model_id"),
        "model_version": meta.get("model_version"),
        "deployable": bool(meta.get("deployable", False)),
        "reason": meta.get("reason"),
        "target": meta.get("target"),
        "threshold": meta.get("threshold"),
        "n_samples": meta.get("n_samples"),
        "n_holdout": meta.get("n_holdout"),
        "holdout_metrics": meta.get("holdout_metrics"),
        "holdout_threshold_stats": meta.get("holdout_threshold_stats"),
        "top_coefficients": meta.get("top_coefficients", [])[:10],
    }


def _daily_sharpe_and_mdd(closed: pd.DataFrame) -> tuple[float | None, float | None]:
    if len(closed) == 0:
        return None, None
    daily = closed.groupby(closed["date_dt"].dt.date)["net_pnl_pct"].sum().sort_index()
    if len(daily) < 2:
        sharpe = None
    else:
        std = daily.std(ddof=1)
        sharpe = (daily.mean() / std) * np.sqrt(252) if std and std > 0 else None
    cum = daily.cumsum()
    mdd = float((cum - cum.cummax()).min()) if len(cum) else None
    return _maybe_float(sharpe), _maybe_float(mdd)


def _bootstrap_pnl_ci(values: pd.Series, n_iter: int = 1000, seed: int = 42) -> dict:
    """Trade-level bootstrap CI for sum/average net PnL in percentage points."""
    pnl = pd.to_numeric(values, errors="coerce").dropna().reset_index(drop=True)
    n = len(pnl)
    if n < 5:
        return {"n": int(n), "note": "n<5: skipped"}
    rng = np.random.default_rng(seed)
    sum_samples = np.empty(n_iter)
    mean_samples = np.empty(n_iter)
    arr = pnl.values
    for i in range(n_iter):
        sample = arr[rng.integers(0, n, size=n)]
        sum_samples[i] = sample.sum()
        mean_samples[i] = sample.mean()
    return {
        "n": int(n),
        "n_iter": int(n_iter),
        "sum_pnl_ci95_pct": [
            _maybe_float(np.quantile(sum_samples, 0.025)),
            _maybe_float(np.quantile(sum_samples, 0.975)),
        ],
        "avg_pnl_ci95_pct": [
            _maybe_float(np.quantile(mean_samples, 0.025)),
            _maybe_float(np.quantile(mean_samples, 0.975)),
        ],
    }


def summarize_group(df: pd.DataFrame, group_cols: list[str], dimension: str) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame()
    rows = []
    for keys, grp in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        closed = grp[grp["net_pnl_pct"].notna()].copy()
        sharpe, mdd = _daily_sharpe_and_mdd(closed)
        row = {col: key for col, key in zip(group_cols, keys)}
        row.update({
            "dimension": dimension,
            "n_candidates": int(len(grp)),
            "n_closed": int(len(closed)),
            "start_date": str(grp["date"].min()),
            "end_date": str(grp["date"].max()),
            "tp5_hit_rate_pct": _maybe_float(closed["tp5_hit"].mean() * 100 if len(closed) else np.nan),
            "win_rate_pct": _maybe_float(closed["net_win"].mean() * 100 if len(closed) else np.nan),
            "net_pnl_sum_pct": _maybe_float(closed["net_pnl_pct"].sum() if len(closed) else np.nan),
            "avg_net_pnl_pct": _maybe_float(closed["net_pnl_pct"].mean() if len(closed) else np.nan),
            "median_net_pnl_pct": _maybe_float(closed["net_pnl_pct"].median() if len(closed) else np.nan),
            "avg_max_return_pct": _maybe_float(closed["max_return_pct"].mean() if len(closed) else np.nan),
            "worst_min_return_pct": _maybe_float(closed["min_return_pct"].min() if len(closed) else np.nan),
            "avg_close_return_pct": _maybe_float(closed["close_return_pct"].mean() if len(closed) else np.nan),
            "daily_sharpe_ann": sharpe,
            "max_drawdown_pct": mdd,
        })
        if "hit_h6" in closed.columns:
            row["hit_h6_rate_pct"] = _maybe_float(pd.to_numeric(closed["hit_h6"], errors="coerce").mean() * 100)
        if "hit_first1h_5pct" in closed.columns:
            row["hit_first1h_5pct_rate_pct"] = _maybe_float(
                pd.to_numeric(closed["hit_first1h_5pct"], errors="coerce").mean() * 100
            )
        row["evidence_tier"] = evidence_tier(row)
        rows.append(row)
    return pd.DataFrame(rows)


def evidence_tier(row: dict) -> str:
    n_closed = int(row.get("n_closed") or 0)
    avg = row.get("avg_net_pnl_pct")
    mdd = row.get("max_drawdown_pct")
    if n_closed < 20:
        return "COLLECT"
    if avg is not None and avg > 0 and (mdd is None or mdd > -25):
        return "PROMOTE_CANDIDATE"
    if avg is not None and avg < 0:
        return "DEMOTE_CANDIDATE"
    return "WATCH"


def _closed_stats(df: pd.DataFrame) -> dict:
    closed = df[df["net_pnl_pct"].notna()].copy()
    sharpe, mdd = _daily_sharpe_and_mdd(closed)
    bootstrap = _bootstrap_pnl_ci(closed["net_pnl_pct"]) if len(closed) else {"n": 0, "note": "empty"}
    return {
        "n_candidates": int(len(df)),
        "n_closed": int(len(closed)),
        "net_pnl_sum_pct": _maybe_float(closed["net_pnl_pct"].sum() if len(closed) else np.nan),
        "avg_net_pnl_pct": _maybe_float(closed["net_pnl_pct"].mean() if len(closed) else np.nan),
        "tp5_hit_rate_pct": _maybe_float(closed["tp5_hit"].mean() * 100 if len(closed) else np.nan),
        "win_rate_pct": _maybe_float(closed["net_win"].mean() * 100 if len(closed) else np.nan),
        "max_drawdown_pct": mdd,
        "daily_sharpe_ann": sharpe,
        "bootstrap": bootstrap,
    }


def _period_split_stats(df: pd.DataFrame) -> list[dict]:
    """Split closed rows by date into early/late halves for overfit sanity checks."""
    closed = df[df["net_pnl_pct"].notna()].copy().sort_values("date_dt")
    if len(closed) == 0:
        return []
    dates = sorted(closed["date_dt"].dropna().dt.date.unique().tolist())
    if len(dates) < 2:
        return [{"period": "all", **_closed_stats(closed)}]

    mid = max(1, len(dates) // 2)
    early_dates = set(dates[:mid])
    late_dates = set(dates[mid:])
    return [
        {
            "period": "early",
            "start_date": str(min(early_dates)),
            "end_date": str(max(early_dates)),
            **_closed_stats(closed[closed["date_dt"].dt.date.isin(early_dates)]),
        },
        {
            "period": "late",
            "start_date": str(min(late_dates)),
            "end_date": str(max(late_dates)),
            **_closed_stats(closed[closed["date_dt"].dt.date.isin(late_dates)]),
        },
    ]


def build_policy_replay(enriched: pd.DataFrame) -> dict:
    """Observed ACTIVE vs current policy replay on the same closed ledger."""
    if len(enriched) == 0:
        return {}

    rows = []
    for channel in ["all"] + sorted(enriched["channel"].dropna().astype(str).unique().tolist()):
        sub = enriched if channel == "all" else enriched[enriched["channel"].astype(str) == channel]
        observed = sub[sub["decision"].astype(str) == ACTIVE]
        replay = sub[sub["policy_decision_v1"].astype(str) == ACTIVE]
        observed_stats = _closed_stats(observed)
        replay_stats = _closed_stats(replay)
        rows.append({
            "channel": channel,
            "policy_id": "setup_quality_policy_v1",
            "observed_active": observed_stats,
            "replay_active": replay_stats,
            "replay_period_stability": _period_split_stats(replay),
            "delta_n_closed": int(replay_stats["n_closed"] - observed_stats["n_closed"]),
            "delta_net_pnl_sum_pct": _maybe_float(
                (replay_stats["net_pnl_sum_pct"] or 0) - (observed_stats["net_pnl_sum_pct"] or 0)
            ),
            "notes": (
                "Replay uses setup_quality/BTC regime only; it is an audit view, "
                "not a claim of out-of-sample performance."
            ),
        })
    return {"rows": rows}


def build_policy_recommendations(enriched: pd.DataFrame) -> list[dict]:
    """Generate action-oriented recommendations from replay evidence."""
    if len(enriched) == 0:
        return []
    recs = []
    replay = build_policy_replay(enriched)
    for row in replay.get("rows", []):
        channel = row["channel"]
        if channel == "all":
            continue
        observed = row["observed_active"]
        active = row["replay_active"]
        delta = row["delta_net_pnl_sum_pct"]
        late = next((p for p in row["replay_period_stability"] if p["period"] == "late"), None)
        late_ok = late is None or (late.get("avg_net_pnl_pct") is not None and late["avg_net_pnl_pct"] > 0)
        if active["n_closed"] >= 20 and active.get("avg_net_pnl_pct", 0) > 0 and delta > 0 and late_ok:
            recs.append({
                "channel": channel,
                "action": "ADOPT_REPLAY_ACTIVE_FILTER",
                "reason": "replay active filter improves net PnL with enough closed samples and positive late-period average",
                "observed_net_pnl_sum_pct": observed.get("net_pnl_sum_pct"),
                "replay_net_pnl_sum_pct": active.get("net_pnl_sum_pct"),
                "delta_net_pnl_sum_pct": delta,
                "n_closed": active["n_closed"],
            })
        elif observed["n_closed"] >= 20 and observed.get("net_pnl_sum_pct", 0) < 0 and active["n_closed"] == 0:
            recs.append({
                "channel": channel,
                "action": "DEMOTE_TO_WATCH_ONLY",
                "reason": "observed active book is negative and replay policy has no active candidates",
                "observed_net_pnl_sum_pct": observed.get("net_pnl_sum_pct"),
                "avoided_loss_pct": _maybe_float(abs(observed.get("net_pnl_sum_pct") or 0)),
                "n_closed": observed["n_closed"],
            })
        else:
            recs.append({
                "channel": channel,
                "action": "KEEP_COLLECTING",
                "reason": "evidence is not strong enough for automatic promotion/demotion",
                "observed_net_pnl_sum_pct": observed.get("net_pnl_sum_pct"),
                "replay_net_pnl_sum_pct": active.get("net_pnl_sum_pct"),
                "n_closed": active["n_closed"],
            })
    return recs


def build_recommendation_quality(enriched: pd.DataFrame) -> dict:
    """Summarize the live recommendation-quality meta layer."""
    trained_meta = _load_meta_model_card()
    if len(enriched) == 0 or "confidence_tier" not in enriched.columns:
        return {
            "available": False,
            "note": "confidence_tier is not populated yet; next prediction run will add meta-filter evidence.",
            "trained_meta_model": trained_meta,
            "rows": [],
        }
    rows = []
    meta = enriched.copy()
    meta["confidence_tier"] = meta["confidence_tier"].fillna("UNSET").astype(str)
    for keys, grp in meta.groupby(["channel", "confidence_tier"], dropna=False):
        channel, tier = keys
        closed_stats = _closed_stats(grp)
        scores = pd.to_numeric(grp.get("confidence_score"), errors="coerce").dropna()
        rows.append({
            "channel": str(channel),
            "confidence_tier": str(tier),
            "n_candidates": int(len(grp)),
            "avg_confidence_score": _maybe_float(scores.mean() if len(scores) else np.nan),
            "n_closed": closed_stats["n_closed"],
            "net_pnl_sum_pct": closed_stats["net_pnl_sum_pct"],
            "avg_net_pnl_pct": closed_stats["avg_net_pnl_pct"],
            "tp5_hit_rate_pct": closed_stats["tp5_hit_rate_pct"],
            "win_rate_pct": closed_stats["win_rate_pct"],
        })
    rows.sort(key=lambda r: (r["channel"], r["confidence_tier"]))
    return {
        "available": True,
        "policy": "recommendation_quality_meta_v1",
        "trained_meta_model": trained_meta,
        "purpose": "Demote historically weak ACTIVE candidates to WATCH_ONLY before Telegram/paper ledger.",
        "rows": rows,
    }


def build_model_card(enriched: pd.DataFrame) -> dict:
    """Portfolio-facing model card for the current operating design."""
    closed = enriched[enriched["net_pnl_pct"].notna()].copy() if len(enriched) else pd.DataFrame()
    replay = build_policy_replay(enriched)
    trained_meta = _load_meta_model_card()
    return {
        "name": "prelude AI quant recommendation assistant",
        "version": "policy-gated-beta-2026-05-25",
        "intended_use": "Personal KRW crypto trading recommendations; no automatic real-money orders.",
        "prediction_target": "Intraday TP5-style upside candidates from pre-open and distribution heads.",
        "decision_layers": [
            "XGBoost candidate generation",
            "setup/regime decision policy",
            "model + send-policy competition audit",
            "historical recommendation-quality meta-filter",
            "paper/shadow ledger validation",
            "policy gate for promotion/demotion",
        ],
        "risk_controls": [
            "look-ahead guard by as-of data cut",
            "ACTIVE-only Telegram; WATCH/SILENCE retained for dashboard evidence",
            "0.15% round-trip transaction cost in reports",
            "paper/shadow close-out before dashboard publication",
            "no automatic exchange order path",
        ],
        "current_evidence": {
            "n_candidates": int(len(enriched)),
            "n_closed": int(len(closed)),
            "net_pnl_sum_pct": _maybe_float(closed["net_pnl_pct"].sum() if len(closed) else np.nan),
            "tp5_hit_rate_pct": _maybe_float(closed["tp5_hit"].mean() * 100 if len(closed) else np.nan),
            "policy_replay_rows": replay.get("rows", []),
        },
        "trained_meta_model": trained_meta,
        "limitations": [
            "Crypto regimes can shift abruptly; live paper evidence outranks replay evidence.",
            "Small shadow-ledger groups remain COLLECT until enough closed samples exist.",
            "Meta-filter can reduce bad recommendations but cannot guarantee profit.",
        ],
        "methodology_references": [
            {
                "label": "Bailey & Lopez de Prado - Deflated Sharpe Ratio / backtest overfitting",
                "url": "https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf",
            },
            {
                "label": "Lopez de Prado publications - financial ML validation methods",
                "url": "https://www.quantresearch.org/Publications.htm",
            },
            {
                "label": "Tree SHAP documentation - explainable tree model attribution",
                "url": "https://shap-community.readthedocs.io/en/latest/generated/shap.explainers.Tree.html",
            },
        ],
    }


def _load_policy_competition(path: str | Path = "output/policy_competition_summary.json") -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def build_report(candidates: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    enriched = add_result_columns(candidates)
    replay_active = enriched[enriched.get("policy_decision_v1", pd.Series(dtype=str)).astype(str) == ACTIVE]
    tables = [
        summarize_group(enriched, ["channel"], "channel"),
        summarize_group(enriched, ["channel", "decision"], "decision"),
        summarize_group(enriched, ["channel", "policy_decision_v1"], "policy_replay_decision"),
        summarize_group(replay_active, ["channel"], "policy_replay_active"),
        summarize_group(enriched, ["channel", "setup_quality"], "setup_quality"),
        summarize_group(enriched, ["channel", "btc_regime"], "btc_regime"),
        summarize_group(enriched, ["channel", "decision", "idea_id", "setup_quality", "btc_regime"], "idea"),
    ]
    non_empty_tables = [
        t.dropna(axis=1, how="all")
        for t in tables
        if len(t) > 0
    ]
    summary_df = pd.concat(non_empty_tables, ignore_index=True, sort=False) if non_empty_tables else pd.DataFrame()
    if len(summary_df):
        summary_df = summary_df.sort_values(
            ["dimension", "channel", "net_pnl_sum_pct", "n_closed"],
            ascending=[True, True, False, False],
            na_position="last",
        ).reset_index(drop=True)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "round_trip_cost_pct": round(ROUND_TRIP_COST_PCT * 100, 4),
        "n_candidates": int(len(enriched)),
        "n_closed": int(enriched["net_pnl_pct"].notna().sum()) if len(enriched) else 0,
        "model_card": build_model_card(enriched),
        "recommendation_quality": build_recommendation_quality(enriched),
        "policy_replay": build_policy_replay(enriched),
        "policy_recommendations": build_policy_recommendations(enriched),
        "policy_competition": _load_policy_competition(),
        "tables": {
            "summary": summary_df.to_dict(orient="records"),
        },
    }
    payload["policy_gate"] = evaluate_policy_gate(payload)
    return summary_df, payload


def write_outputs(summary_df: pd.DataFrame, payload: dict, out_csv: str | Path, out_json: str | Path) -> None:
    csv_path = Path(out_csv)
    json_path = Path(out_json)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-ledger", default="output/paper_ledger.csv")
    parser.add_argument("--paper-ledger-preopen", default="output/paper_ledger_preopen.csv")
    parser.add_argument("--shadow-ledger-distribution", default="output/shadow_ledger_distribution.csv")
    parser.add_argument("--shadow-ledger-preopen", default="output/shadow_ledger_preopen.csv")
    parser.add_argument("--out-csv", default="output/idea_validation_summary.csv")
    parser.add_argument("--out-json", default="output/idea_validation_summary.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("idea-validation")

    candidates = load_candidate_ledger(args)
    summary_df, payload = build_report(candidates)
    write_outputs(summary_df, payload, args.out_csv, args.out_json)
    log.info("saved %s and %s", args.out_csv, args.out_json)
    print(
        "idea validation: "
        f"candidates={payload['n_candidates']} closed={payload['n_closed']} "
        f"rows={len(summary_df)}"
    )


if __name__ == "__main__":
    main()

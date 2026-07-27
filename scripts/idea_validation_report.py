"""Idea validation report for ACTIVE/WATCH_ONLY/SILENCE candidates.

This is the portfolio/experiment layer: combine active paper ledgers and shadow
candidate ledgers, compute net TP5-style results, and attribute performance by
idea_id, decision, setup quality, and BTC regime.
"""
from __future__ import annotations

import argparse
import logging
import platform
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ops.artifact_provenance import (  # noqa: E402
    ArtifactSourceChangedError,
    ArtifactValidationError,
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    file_set_identity,
    manifest_digest_matches,
    resolve_identity_path,
    sha256_bytes,
    strict_json_object,
    with_manifest_digest,
)
from ops.policy_competition import (  # noqa: E402
    POLICY_DB,
    PolicyArtifactError,
    load_policy_artifact,
)
from ledger.config import ROUND_TRIP_COST_PCT
from ledger.portfolio_metrics import (
    daily_equal_weight,
    date_cluster_bootstrap,
    normalize_kst_date,
    summarize_daily,
)
from ops.decision_policy import ACTIVE, SILENCE, WATCH_ONLY, setup_quality
from ops.policy_gate import evaluate_policy_gate
from ops.recommendation_quality import (
    DEFAULT_META_MODEL_DIR,
    inspect_trained_meta_model,
)


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
DEFAULT_POLICY_COMPETITION_PATH = "output/policy_competition_summary.json"
_ROOT = Path(__file__).resolve().parent.parent
IDEA_ARTIFACT_SCHEMA = "idea_validation.v2"
IDEA_INPUT_MANIFEST_SCHEMA_VERSION = 3
IDEA_GENERATOR_SOURCES = (
    "scripts/idea_validation_report.py",
    "ops/artifact_provenance.py",
    "ops/policy_competition.py",
    "ops/policy_gate.py",
    "ops/decision_policy.py",
    "ops/recommendation_quality.py",
    "ledger/config.py",
    "ledger/portfolio_metrics.py",
)


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


def _idea_manifest_contract() -> dict[str, Any]:
    return {
        "round_trip_cost_pct": ROUND_TRIP_COST_PCT,
        "merge_key": MERGE_KEY,
        "distribution_realized_columns": DIST_REALIZED_COLS,
        "preopen_realized_columns": PREOPEN_REALIZED_COLS,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }


def build_input_manifest(
    args,
    *,
    policy_competition_path: str | Path = DEFAULT_POLICY_COMPETITION_PATH,
    policy_db_path: str | Path = POLICY_DB,
    meta_model_dir: str | Path = DEFAULT_META_MODEL_DIR,
) -> dict:
    """Content-bound input lineage for a reproducible report build."""
    policy_json_path = Path(policy_competition_path)
    paths = {
        "paper_ledger_distribution": args.paper_ledger,
        "paper_ledger_preopen": args.paper_ledger_preopen,
        "shadow_ledger_distribution": args.shadow_ledger_distribution,
        "shadow_ledger_preopen": args.shadow_ledger_preopen,
        "policy_competition_json": policy_json_path,
        "policy_competition_csv": policy_json_path.with_suffix(".csv"),
        "policy_competition_db": policy_db_path,
        "policy_competition_db_wal": Path(f"{policy_db_path}-wal"),
        "policy_competition_db_journal": Path(
            f"{policy_db_path}-journal"
        ),
        "recommendation_meta": Path(meta_model_dir) / "meta.json",
    }
    rooted_paths = {
        name: resolve_identity_path(str(path), root=_ROOT)
        for name, path in paths.items()
    }
    captured_files = file_set_identity(
        {
            **{
                f"input:{name}": path
                for name, path in rooted_paths.items()
            },
            **{
                f"generator:{relative}": _ROOT / relative
                for relative in IDEA_GENERATOR_SOURCES
            },
        },
        root=_ROOT,
    )
    files = {
        name: captured_files[f"input:{name}"]
        for name in rooted_paths
    }
    generator_sources = {
        relative: captured_files[f"generator:{relative}"]
        for relative in IDEA_GENERATOR_SOURCES
    }
    missing_sources = [
        name
        for name, identity in generator_sources.items()
        if not identity["exists"]
    ]
    if missing_sources:
        raise FileNotFoundError(
            f"idea-validation generator source missing: {missing_sources}"
        )
    manifest = {
        "schema_version": IDEA_INPUT_MANIFEST_SCHEMA_VERSION,
        "files": files,
        "generator_sources": generator_sources,
        "contract": _idea_manifest_contract(),
    }
    return with_manifest_digest(manifest)


def input_manifest_matches_current(lineage: dict[str, Any]) -> bool:
    """Re-hash every report input and generator source; reject legacy manifests."""
    if (
        not isinstance(lineage, dict)
        or lineage.get("schema_version")
        != IDEA_INPUT_MANIFEST_SCHEMA_VERSION
        or not manifest_digest_matches(lineage)
        or lineage.get("contract") != _idea_manifest_contract()
    ):
        return False
    identity_groups = (
        lineage.get("files"),
        lineage.get("generator_sources"),
    )
    if any(not isinstance(group, dict) or not group for group in identity_groups):
        return False
    try:
        expected: dict[str, dict[str, Any]] = {}
        sources: dict[str, Path] = {}
        for group_index, group in enumerate(identity_groups):
            if not isinstance(group, dict):
                return False
            for name, identity in group.items():
                if not isinstance(identity, dict):
                    return False
                path_value = identity.get("path")
                if not isinstance(path_value, str):
                    return False
                key = f"{group_index}:{name}"
                expected[key] = identity
                sources[key] = resolve_identity_path(
                    path_value,
                    root=_ROOT,
                )
        if file_set_identity(sources, root=_ROOT) != expected:
            return False
    except (
        OSError,
        TypeError,
        ValueError,
        ArtifactSourceChangedError,
    ):
        return False
    return True


def report_payload_digest(payload: dict) -> str:
    canonical_payload = {
        key: value for key, value in payload.items() if key != "payload_sha256"
    }
    return sha256_bytes(canonical_json_bytes(_json_safe(canonical_payload)))


def _json_safe(value: Any) -> Any:
    """Normalize pandas/numpy values to strict RFC JSON without silent NaN."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime, date, np.datetime64)):
        timestamp = pd.Timestamp(value)
        return None if pd.isna(timestamp) else timestamp.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("idea JSON object keys must be strings")
            normalized[key] = _json_safe(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    raise TypeError(
        f"idea payload contains unsupported JSON value: {type(value).__name__}"
    )


class IdeaArtifactError(ArtifactValidationError):
    """The idea-validation report is malformed, stale, or unverifiable."""


def validate_idea_validation_payload(
    payload: dict[str, Any],
    *,
    asof=None,
    require_current: bool = True,
) -> None:
    if payload.get("schema") != IDEA_ARTIFACT_SCHEMA:
        raise IdeaArtifactError("unsupported idea-validation artifact schema")
    artifact_value = payload.get("asof")
    if not isinstance(artifact_value, str) or not artifact_value.strip():
        raise IdeaArtifactError(
            "idea-validation artifact asof is invalid"
        )
    try:
        artifact_asof = normalize_kst_date(artifact_value)
    except (TypeError, ValueError) as exc:
        raise IdeaArtifactError(
            "idea-validation artifact asof is invalid"
        ) from exc
    if asof is not None and artifact_asof > normalize_kst_date(asof):
        raise IdeaArtifactError("idea-validation artifact is future-dated")
    generated_value = payload.get("generated_at_utc")
    try:
        generated = pd.Timestamp(generated_value)
    except (TypeError, ValueError) as exc:
        raise IdeaArtifactError(
            "idea-validation generated_at_utc is invalid"
        ) from exc
    if generated.tzinfo is None:
        raise IdeaArtifactError(
            "idea-validation generated_at_utc must be timezone-aware"
        )
    lineage = payload.get("input_lineage")
    if not isinstance(lineage, dict) or not manifest_digest_matches(lineage):
        raise IdeaArtifactError(
            "idea-validation input lineage is invalid"
        )
    if require_current and not input_manifest_matches_current(lineage):
        raise IdeaArtifactError(
            "idea-validation input or generator lineage is stale"
        )
    checksum = payload.get("payload_sha256")
    if (
        not isinstance(checksum, str)
        or checksum != report_payload_digest(payload)
    ):
        raise IdeaArtifactError(
            "idea-validation payload checksum mismatch"
        )
    try:
        canonical_json_bytes(payload)
    except (TypeError, ValueError) as exc:
        raise IdeaArtifactError(
            "idea-validation payload is not strict JSON"
        ) from exc


def load_idea_validation_artifact(
    path: str | Path,
    *,
    asof=None,
    require_current: bool = True,
) -> dict[str, Any]:
    try:
        payload = strict_json_object(path)
    except ArtifactValidationError as exc:
        raise IdeaArtifactError(str(exc)) from exc
    validate_idea_validation_payload(
        payload,
        asof=asof,
        require_current=require_current,
    )
    return payload


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


def _strict_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1"}
    return False


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
    out["net_win"] = np.nan
    out["promotion_eligible"] = False
    out["outcome_contract"] = "unavailable"
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

    # Existing idea ledgers mostly expose only max/min/close aggregates.  A
    # max>=TP does not tell whether SL was touched first, so that TP-only lens
    # remains diagnostic and must never feed a promotion decision.
    proxy_valid = out["is_closed"] & out["max_return_pct"].notna()
    out.loc[proxy_valid, "tp5_hit"] = (
        out.loc[proxy_valid, "max_return_pct"] >= 5.0
    ).astype(int)
    out.loc[proxy_valid, "net_pnl_pct"] = out.loc[proxy_valid].apply(
        lambda r: _net_pnl_pct(r["max_return_pct"], r["close_return_pct"]),
        axis=1,
    )
    out.loc[proxy_valid, "outcome_contract"] = (
        "tp5_high_then_eod_proxy_unordered"
    )

    if {"exit_reason", "realized_pct", "path_complete"} <= set(out.columns):
        reasons = out["exit_reason"].astype(str).str.upper()
        realized = pd.to_numeric(out["realized_pct"], errors="coerce")
        complete = out["path_complete"].map(_strict_bool)
        ordered = (
            out["is_closed"]
            & complete
            & reasons.isin({"TP", "SL", "EOD"})
            & realized.notna()
        )
        out.loc[ordered, "net_pnl_pct"] = realized.loc[ordered]
        out.loc[ordered, "tp5_hit"] = reasons.loc[ordered].eq("TP").astype(int)
        out.loc[ordered, "promotion_eligible"] = True
        out.loc[ordered, "outcome_contract"] = (
            "tp5_sl3_ordered_first_passage_net"
        )

    valid_net = out["net_pnl_pct"].notna()
    out.loc[valid_net, "net_win"] = (
        out.loc[valid_net, "net_pnl_pct"] > 0
    ).astype(int)
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


def _load_meta_model_card(
    model_dir: str | Path = DEFAULT_META_MODEL_DIR,
    *,
    asof=None,
) -> dict:
    try:
        inspected = inspect_trained_meta_model(model_dir)
    except (ArtifactValidationError, ArtifactSourceChangedError):
        return {
            "available": False,
            "reason": "trained recommendation meta artifact invalid",
        }
    if inspected is None:
        return {"available": False, "reason": "trained recommendation meta model not found"}
    meta = inspected["meta"]
    artifact_status = str(inspected["artifact_status"])
    cutoff = normalize_kst_date(asof)
    date_range = meta.get("date_range")
    trained_through = date_range.get("end") if isinstance(date_range, dict) else None
    built_at = meta.get("built_at")
    if not trained_through or not built_at:
        return {
            "available": False,
            "reason": "trained recommendation meta model lacks as-of provenance",
        }
    try:
        trained_date = normalize_kst_date(trained_through)
        built_date = normalize_kst_date(built_at)
    except ValueError:
        return {
            "available": False,
            "reason": "trained recommendation meta model has invalid as-of provenance",
        }
    if trained_date > cutoff or built_date > cutoff:
        return {
            "available": False,
            "reason": (
                "trained recommendation meta model is future-dated for report asof "
                f"{cutoff.date()}"
            ),
            "trained_through": str(trained_date.date()),
            "built_at": str(built_at),
        }
    return {
        "available": True,
        "model_id": meta.get("model_id"),
        "model_version": meta.get("model_version"),
        "deployable": bool(meta.get("deployable", False)),
        "declared_deployable": bool(meta.get("declared_deployable", False)),
        "artifact_status": artifact_status,
        "reason": meta.get("reason"),
        "target": meta.get("target"),
        "threshold": meta.get("threshold"),
        "n_samples": meta.get("n_samples"),
        "n_holdout": meta.get("n_holdout"),
        "trained_through": str(trained_date.date()),
        "built_at": str(built_at),
        "holdout_metrics": meta.get("holdout_metrics"),
        "holdout_threshold_stats": meta.get("holdout_threshold_stats"),
        "top_coefficients": meta.get("top_coefficients", [])[:10],
    }


def _daily_sharpe_and_mdd(closed: pd.DataFrame) -> tuple[float | None, float | None]:
    if len(closed) == 0:
        return None, None
    daily = daily_equal_weight(
        closed["date_dt"],
        pd.to_numeric(closed["net_pnl_pct"], errors="coerce") / 100.0,
    )
    summary = summarize_daily(daily)
    sharpe = summary["sharpe_ann"] if len(daily) >= 2 else None
    return _maybe_float(sharpe), _maybe_float(summary["max_drawdown"] * 100)


def _bootstrap_pnl_ci(closed: pd.DataFrame, n_iter: int = 1000, seed: int = 42) -> dict:
    """Date-cluster CI after within-day equal weighting."""
    pnl = pd.to_numeric(closed["net_pnl_pct"], errors="coerce")
    valid = pnl.notna() & closed["date_dt"].notna()
    pnl = pnl.loc[valid]
    n = len(pnl)
    daily = daily_equal_weight(
        closed.loc[valid, "date_dt"],
        pnl / 100.0,
        include_no_trade_days=False,
    )
    boot = date_cluster_bootstrap(daily, n_iter=n_iter, seed=seed)
    if "mean_return_ci95" not in boot:
        return {
            "n": int(n),
            "n_days": int(boot["n_days"]),
            "note": boot["note"],
        }
    return {
        "n": int(n),
        "n_days": int(boot["n_days"]),
        "n_iter": int(n_iter),
        "method": "date_cluster_equal_weight",
        "sum_pnl_ci95_pct": [
            _maybe_float(v * 100) for v in boot["cumulative_return_ci95"]
        ],
        "avg_pnl_ci95_pct": [
            _maybe_float(v * 100) for v in boot["mean_return_ci95"]
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
        promotion_closed = closed[
            closed["promotion_eligible"].map(_strict_bool)
        ].copy()
        sharpe, mdd = _daily_sharpe_and_mdd(closed)
        promotion_sharpe, promotion_mdd = _daily_sharpe_and_mdd(
            promotion_closed
        )
        daily = daily_equal_weight(
            closed["date_dt"],
            pd.to_numeric(closed["net_pnl_pct"], errors="coerce") / 100.0,
        )
        portfolio = summarize_daily(daily)
        row = {col: key for col, key in zip(group_cols, keys)}
        row.update({
            "dimension": dimension,
            "n_candidates": int(len(grp)),
            "n_closed": int(len(closed)),
            "n_promotion_eligible_closed": int(len(promotion_closed)),
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
            "portfolio_cum_return_pct": _maybe_float(
                portfolio["cumulative_return"] * 100
            ),
            "portfolio_calendar_days": portfolio["n_calendar_days"],
            "promotion_net_pnl_sum_pct": _maybe_float(
                promotion_closed["net_pnl_pct"].sum()
                if len(promotion_closed)
                else np.nan
            ),
            "promotion_avg_net_pnl_pct": _maybe_float(
                promotion_closed["net_pnl_pct"].mean()
                if len(promotion_closed)
                else np.nan
            ),
            "promotion_daily_sharpe_ann": promotion_sharpe,
            "promotion_max_drawdown_pct": promotion_mdd,
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
    n_closed = int(row.get("n_promotion_eligible_closed") or 0)
    avg = row.get("promotion_avg_net_pnl_pct")
    mdd = row.get("promotion_max_drawdown_pct")
    if n_closed == 0:
        return "DIAGNOSTIC_ONLY"
    if n_closed < 20:
        return "COLLECT"
    if avg is not None and avg > 0 and (mdd is None or mdd > -25):
        return "PROMOTE_CANDIDATE"
    if avg is not None and avg < 0:
        return "DEMOTE_CANDIDATE"
    return "WATCH"


def _closed_stats_core(df: pd.DataFrame) -> dict:
    closed = df[df["net_pnl_pct"].notna()].copy()
    sharpe, mdd = _daily_sharpe_and_mdd(closed)
    bootstrap = _bootstrap_pnl_ci(closed) if len(closed) else {"n": 0, "note": "empty"}
    daily = daily_equal_weight(
        closed["date_dt"],
        pd.to_numeric(closed["net_pnl_pct"], errors="coerce") / 100.0,
    )
    portfolio = summarize_daily(daily)
    return {
        "n_candidates": int(len(df)),
        "n_closed": int(len(closed)),
        "net_pnl_sum_pct": _maybe_float(closed["net_pnl_pct"].sum() if len(closed) else np.nan),
        "avg_net_pnl_pct": _maybe_float(closed["net_pnl_pct"].mean() if len(closed) else np.nan),
        "tp5_hit_rate_pct": _maybe_float(closed["tp5_hit"].mean() * 100 if len(closed) else np.nan),
        "win_rate_pct": _maybe_float(closed["net_win"].mean() * 100 if len(closed) else np.nan),
        "max_drawdown_pct": mdd,
        "daily_sharpe_ann": sharpe,
        "portfolio_cum_return_pct": _maybe_float(
            portfolio["cumulative_return"] * 100
        ),
        "portfolio_calendar_days": portfolio["n_calendar_days"],
        "bootstrap": bootstrap,
    }


def _closed_stats(df: pd.DataFrame) -> dict:
    closed = df[df["net_pnl_pct"].notna()].copy()
    stats = _closed_stats_core(closed)
    if "promotion_eligible" in closed.columns:
        promotion_closed = closed[
            closed["promotion_eligible"].map(_strict_bool)
        ].copy()
    else:
        promotion_closed = closed.iloc[0:0].copy()
    promotion = _closed_stats_core(promotion_closed)
    promotion.update({
        "eligible": bool(len(promotion_closed)),
        "required_contract": "tp5_sl3_ordered_first_passage_net",
    })
    stats["promotion_evidence"] = promotion
    stats["diagnostic_outcome_contracts"] = {
        str(key): int(value)
        for key, value in closed.get(
            "outcome_contract",
            pd.Series(index=closed.index, dtype=str),
        ).value_counts().items()
    }
    return stats


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
        observed_promotion = observed_stats["promotion_evidence"]
        replay_promotion = replay_stats["promotion_evidence"]
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
            "promotion_delta_net_pnl_sum_pct": _maybe_float(
                (replay_promotion["net_pnl_sum_pct"] or 0)
                - (observed_promotion["net_pnl_sum_pct"] or 0)
            ),
            "notes": (
                "Replay uses setup_quality/BTC regime only. Aggregate-high "
                "TP-only results are diagnostic; promotion requires complete "
                "ordered TP5/SL3 first-passage outcomes."
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
        active_promotion = active.get("promotion_evidence") or {}
        delta = row.get("promotion_delta_net_pnl_sum_pct") or 0
        late = next((p for p in row["replay_period_stability"] if p["period"] == "late"), None)
        late_promotion = (late or {}).get("promotion_evidence") or {}
        late_ok = (
            late is None
            or (
                late_promotion.get("avg_net_pnl_pct") is not None
                and late_promotion["avg_net_pnl_pct"] > 0
            )
        )
        if (
            active_promotion.get("n_closed", 0) >= 20
            and active_promotion.get("avg_net_pnl_pct", 0) > 0
            and delta > 0
            and late_ok
        ):
            recs.append({
                "channel": channel,
                "action": "ADOPT_REPLAY_ACTIVE_FILTER",
                "reason": "replay active filter improves net PnL with enough closed samples and positive late-period average",
                "observed_net_pnl_sum_pct": observed.get("net_pnl_sum_pct"),
                "replay_net_pnl_sum_pct": active_promotion.get(
                    "net_pnl_sum_pct"
                ),
                "delta_net_pnl_sum_pct": delta,
                "n_closed": active_promotion["n_closed"],
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
                "replay_net_pnl_sum_pct": active_promotion.get(
                    "net_pnl_sum_pct"
                ),
                "n_closed": active_promotion.get("n_closed", 0),
            })
    return recs


def build_recommendation_quality(
    enriched: pd.DataFrame,
    *,
    trained_meta: dict | None = None,
) -> dict:
    """Summarize the live recommendation-quality meta layer."""
    trained_meta = trained_meta or {
        "available": False,
        "reason": "trained recommendation meta model not evaluated",
    }
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


def build_model_card(
    enriched: pd.DataFrame,
    *,
    trained_meta: dict | None = None,
) -> dict:
    """Portfolio-facing model card for the current operating design."""
    closed = enriched[enriched["net_pnl_pct"].notna()].copy() if len(enriched) else pd.DataFrame()
    replay = build_policy_replay(enriched)
    trained_meta = trained_meta or {
        "available": False,
        "reason": "trained recommendation meta model not evaluated",
    }
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


def _load_policy_competition(
    path: str | Path = DEFAULT_POLICY_COMPETITION_PATH,
    *,
    asof=None,
    db_path: str | Path = POLICY_DB,
) -> dict | None:
    p = Path(path)
    cutoff = normalize_kst_date(asof)
    try:
        payload = load_policy_artifact(
            p,
            csv_path=p.with_suffix(".csv"),
            db_path=Path(db_path),
            asof=cutoff,
            require_current=True,
        )
    except PolicyArtifactError:
        return None
    try:
        artifact_asof = normalize_kst_date(payload["asof"])
    except (KeyError, ValueError):
        return None
    payload = dict(payload)
    payload["asof"] = str(artifact_asof.date())
    return payload


def build_report(
    candidates: pd.DataFrame,
    *,
    asof=None,
    input_manifest: dict | None = None,
    policy_competition_path: str | Path = DEFAULT_POLICY_COMPETITION_PATH,
    meta_model_dir: str | Path = DEFAULT_META_MODEL_DIR,
) -> tuple[pd.DataFrame, dict]:
    cutoff = normalize_kst_date(asof)
    source = candidates.reset_index(drop=True).copy()
    if "date" in source.columns:
        parsed_dates = source["date"].map(
            lambda value: _try_normalize_date(value)
        )
    else:
        parsed_dates = pd.Series(pd.NaT, index=source.index, dtype="datetime64[ns]")
    invalid_date = parsed_dates.isna()
    future_date = parsed_dates.notna() & (parsed_dates > cutoff)
    source = source.loc[~invalid_date & ~future_date].copy()
    if len(source):
        source["date"] = parsed_dates.loc[source.index].dt.strftime("%Y-%m-%d")

    enriched = add_result_columns(source)
    trained_meta = _load_meta_model_card(meta_model_dir, asof=cutoff)
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

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload = {
        "schema": IDEA_ARTIFACT_SCHEMA,
        "asof": str(cutoff.date()),
        "generated_at": generated_at,
        "generated_at_utc": generated_at,
        "cutoff_timezone": "Asia/Seoul",
        "cutoff_exclusions": {
            "future_date_rows": int(future_date.sum()),
            "invalid_date_rows": int(invalid_date.sum()),
        },
        "input_lineage": input_manifest or {
            "schema_version": 0,
            "kind": "in_memory",
            "row_count": int(len(candidates)),
        },
        "round_trip_cost_pct": round(ROUND_TRIP_COST_PCT * 100, 4),
        "n_candidates": int(len(enriched)),
        "n_closed": int(enriched["net_pnl_pct"].notna().sum()) if len(enriched) else 0,
        "model_card": build_model_card(enriched, trained_meta=trained_meta),
        "recommendation_quality": build_recommendation_quality(
            enriched,
            trained_meta=trained_meta,
        ),
        "policy_replay": build_policy_replay(enriched),
        "policy_recommendations": build_policy_recommendations(enriched),
        "policy_competition": _load_policy_competition(
            policy_competition_path,
            asof=cutoff,
        ),
        "tables": {
            "summary": summary_df.to_dict(orient="records"),
        },
    }
    payload["policy_gate"] = evaluate_policy_gate(payload)
    payload = _json_safe(payload)
    payload["payload_sha256"] = report_payload_digest(payload)
    return summary_df, payload


def _try_normalize_date(value):
    if value is None or (
        isinstance(value, float) and np.isnan(value)
    ) or str(value).strip() == "":
        return pd.NaT
    try:
        return normalize_kst_date(value)
    except (TypeError, ValueError):
        return pd.NaT


def write_outputs(summary_df: pd.DataFrame, payload: dict, out_csv: str | Path, out_json: str | Path) -> None:
    csv_path = Path(out_csv)
    json_path = Path(out_json)
    normalized = _json_safe(payload)
    validate_idea_validation_payload(normalized, require_current=True)
    csv_bytes = summary_df.to_csv(
        index=False,
        lineterminator="\n",
    ).encode("utf-8")
    atomic_write_bytes(csv_path, csv_bytes)
    atomic_write_json(json_path, normalized)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-ledger", default="output/paper_ledger.csv")
    parser.add_argument("--paper-ledger-preopen", default="output/paper_ledger_preopen.csv")
    parser.add_argument("--shadow-ledger-distribution", default="output/shadow_ledger_distribution.csv")
    parser.add_argument("--shadow-ledger-preopen", default="output/shadow_ledger_preopen.csv")
    parser.add_argument(
        "--policy-competition-json",
        default=DEFAULT_POLICY_COMPETITION_PATH,
    )
    parser.add_argument("--asof", help="inclusive KST cutoff (default=today KST)")
    parser.add_argument("--out-csv", default="output/idea_validation_summary.csv")
    parser.add_argument("--out-json", default="output/idea_validation_summary.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("idea-validation")

    cutoff = normalize_kst_date(args.asof)
    before = build_input_manifest(
        args,
        policy_competition_path=args.policy_competition_json,
    )
    candidates = load_candidate_ledger(args)
    summary_df, payload = build_report(
        candidates,
        asof=cutoff,
        input_manifest=before,
        policy_competition_path=args.policy_competition_json,
    )
    after = build_input_manifest(
        args,
        policy_competition_path=args.policy_competition_json,
    )
    if before != after:
        raise RuntimeError("idea validation inputs changed during report build")
    write_outputs(summary_df, payload, args.out_csv, args.out_json)
    if build_input_manifest(
        args,
        policy_competition_path=args.policy_competition_json,
    ) != before:
        raise RuntimeError(
            "idea validation inputs changed during report publication"
        )
    log.info("saved %s and %s", args.out_csv, args.out_json)
    print(
        "idea validation: "
        f"candidates={payload['n_candidates']} closed={payload['n_closed']} "
        f"rows={len(summary_df)}"
    )


if __name__ == "__main__":
    main()

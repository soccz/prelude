"""Read-only readiness export for future R1 experiments; never fits a model.

Only frozen decision-time feature values are exported.  Targets remain the
existing 24-hour forward labels, and chronological folds keep every coin and
slot of a date together.  These already-observed dates are not a fresh holdout.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ledger.config import ROUND_TRIP_COST_PCT
from ops.artifact_provenance import file_set_identity
from ops.recommendation_evidence import (
    EvidenceError,
    EvidenceUnavailable,
    discover_r1_snapshots,
    load_recommendation_evidence,
)

ALLOWED_FEATURES = (
    "f_qv_surge_30d", "f_qv_surge_7d", "f_qv_ma7_vs_ma30", "f_vol_surge_7d",
    "f_bounce_off_7d_low", "f_ret_1d", "f_ret_3d", "f_ret_7d", "f_ret_14d",
    "f_roc_3d", "f_roc_7d", "f_atr_pct_14", "f_rv_7d", "f_rv_21d",
    "f_log_qv", "f_pos_in_20d_range", "f_drawdown_7d", "f_rsi_14",
    "f_up_streak", "f_ret3d_xs_decile", "f_ret7d_xs_decile",
    "f_atr_xs_decile", "f_qvsurge_xs_decile", "f_qv_rank_pct",
)
OUTCOME_FIELDS = (
    "actual_entry_open", "mfe", "mae", "eod_return_net", "up5", "up10",
    "up20", "dn3", "dn5", "dn10", "tp5_sl3_first_passage",
    "tp5_before_sl3", "tp5_sl3_return_net",
)
BINARY_OUTCOMES = ("up5", "up10", "up20", "dn3", "dn5", "dn10")
REPORT_SCHEMA = "recommend_training_readiness.v1"
ROOT = Path(__file__).resolve().parent.parent
GENERATOR_SOURCES = (
    "signals/recommend_training_data.py", "scripts/build_recommend_training_data.py",
    "ops/recommendation_evidence.py", "signals/recommend_snapshot.py",
    "signals/recommend_score_labels.py", "notifier/delivery_receipt.py",
    "ops/artifact_provenance.py", "ledger/path_quality.py", "ledger/config.py",
)
KNOWN_FEATURE_ALIASES = (("f_ret_3d", "f_roc_3d"), ("f_ret_7d", "f_roc_7d"))


class TrainingDataError(EvidenceError):
    """The export cannot safely represent its frozen inputs."""


def _aware(value: Any, field: str) -> datetime:
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise TrainingDataError(f"invalid {field}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise TrainingDataError(f"{field} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _source_manifest() -> dict:
    files = file_set_identity({name: ROOT / name for name in GENERATOR_SOURCES}, root=ROOT)
    if not all(item["exists"] for item in files.values()):
        raise TrainingDataError("generator source is missing")
    return {"files": files, "sha256": _digest(files)}


def _features(candidate: dict, columns: list[str]) -> dict:
    if len(columns) != len(ALLOWED_FEATURES) or set(columns) != set(ALLOWED_FEATURES):
        raise TrainingDataError("snapshot features do not match the frozen 24-feature allowlist")
    values = candidate.get("feature_values")
    if not isinstance(values, dict) or set(values) != set(ALLOWED_FEATURES):
        raise TrainingDataError("candidate feature keys do not match the allowlist")
    for name, value in values.items():
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise TrainingDataError(f"invalid stored feature {name}")
    # Do not impute, recompute from today's DB, append scores, or fit transforms.
    return {name: values[name] for name in ALLOWED_FEATURES}


def _export_evidence(evidence: dict, now: datetime) -> tuple[list[dict], list[dict]]:
    snapshot, label, receipt = (evidence[name] for name in ("snapshot", "label", "receipt"))
    asof = date.fromisoformat(snapshot["asof"])
    feature_asof = date.fromisoformat(snapshot["feature_asof"])
    started = _aware(snapshot["decision_started_at"], "decision_started_at")
    completed = _aware(snapshot["decision_completed_at"], "decision_completed_at")
    end = _aware(label["path_window_end"], "path_window_end")
    labeled = _aware(label["labeled_at"], "labeled_at")
    available = max(end, labeled)
    if max(started, completed, available) > now:
        raise TrainingDataError("evidence contains future timestamps")
    labels = {row["coin"]: row for row in label["rows"]}
    selected = {row["coin"] for row in snapshot["top3"]}
    rows, excluded = [], []
    for candidate in snapshot["universe"]:
        features = _features(candidate, snapshot["feature_columns"])
        outcome = labels[candidate["coin"]]
        if outcome["label_status"] == "halted_no_observations":
            excluded.append({"snapshot_id": snapshot["snapshot_id"], "coin": candidate["coin"],
                             "reason": "halted_no_observations"})
            continue
        if outcome["label_status"] != "labeled":
            raise TrainingDataError("accepted evidence contains an unlabeled candidate")
        if not all(name in outcome for name in OUTCOME_FIELDS):
            raise TrainingDataError("canonical outcome fields are missing")
        rows.append({
            "row_id": f"{snapshot['snapshot_id']}:{candidate['coin']}",
            "model_features": features,
            "outcomes": {name: outcome[name] for name in OUTCOME_FIELDS},
            "metadata": {
                "date": asof.isoformat(), "slot": snapshot["slot"],
                "coin": candidate["coin"], "rank": candidate["rank"],
                "was_delivered": candidate["coin"] in selected,
                "snapshot_id": snapshot["snapshot_id"],
                "snapshot_payload_sha256": snapshot["payload_sha256"],
                "label_payload_sha256": label["label_payload_sha256"],
                "model_id": snapshot["model"]["id"],
                "decision_started_at": started.isoformat(),
                "decision_completed_at": completed.isoformat(),
                "sent_at": receipt["sent_at"],
                "execution_start_at": label["execution_start_at"],
                "outcome_end_at": end.isoformat(),
                "label_created_at": labeled.isoformat(),
                "label_available_at": available.isoformat(),
                "label_availability_basis": "max(path_window_end,current_artifact_labeled_at)",
                "nominal_feature_row_date": feature_asof.isoformat(),
                "nominal_shift1_input_bar_date": (feature_asof - timedelta(days=1)).isoformat(),
                "actual_latest_input_timestamp": None,
                "actual_input_freshness": "unknown_per_coin_timestamp_not_recorded",
                "path_quality": outcome["path_quality"],
                "raw_bars": outcome["raw_bars"],
                "flat_filled_bars": outcome["flat_filled_bars"],
            },
        })
    return rows, excluded


def _sample_summary(rows: list[dict]) -> dict:
    return {
        "rows": len(rows),
        "dates": len({r["metadata"]["date"] for r in rows}),
        "rows_by_slot": dict(sorted(Counter(r["metadata"]["slot"] for r in rows).items())),
        "delivered_rows": sum(r["metadata"]["was_delivered"] for r in rows),
        "outcome_classes": {
            name: {"positive": sum(r["outcomes"][name] is True for r in rows),
                   "negative": sum(r["outcomes"][name] is False for r in rows)}
            for name in BINARY_OUTCOMES
        },
    }


def expanding_readiness_splits(
    rows: list[dict], *, min_train_dates: int = 10, validation_dates: int = 5,
) -> list[dict]:
    """Make global-date folds, purging outcomes not yet demonstrably available.

    Defaults are diagnostic initial values, not model promotion requirements.
    One late label removes its whole training date, including the other slot.
    """
    if (isinstance(min_train_dates, bool) or not isinstance(min_train_dates, int)
            or min_train_dates < 1 or isinstance(validation_dates, bool)
            or not isinstance(validation_dates, int) or validation_dates < 1):
        raise TrainingDataError("split date counts must be positive integers")
    by_date: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_date[row["metadata"]["date"]].append(row)
    dates = sorted(by_date)
    available = {
        day: max(_aware(r["metadata"]["label_available_at"], "label_available_at")
                 for r in day_rows)
        for day, day_rows in by_date.items()
    }
    folds = []
    for offset in range(min_train_dates, len(dates), validation_dates):
        val_dates = dates[offset:offset + validation_dates]
        val_rows = [r for day in val_dates for r in by_date[day]]
        decision = min(_aware(r["metadata"]["decision_started_at"], "decision_started_at")
                       for r in val_rows)
        train_dates = [day for day in dates[:offset] if available[day] < decision]
        purged = [day for day in dates[:offset] if available[day] >= decision]
        train_rows = [r for day in train_dates for r in by_date[day]]
        latest = max((available[day] for day in train_dates), default=None)
        if latest is not None and latest >= decision:
            raise TrainingDataError("training label availability overlaps validation")
        folds.append({
            "fold": len(folds) + 1,
            "status": "split_ready" if len(train_dates) >= min_train_dates else "insufficient_train_dates",
            "train_dates": train_dates, "validation_dates": val_dates,
            "purged_train_dates": purged,
            "train": _sample_summary(train_rows), "validation": _sample_summary(val_rows),
            "train_max_label_available_at": latest.isoformat() if latest else None,
            "validation_min_decision_started_at": decision.isoformat(),
            "is_untouched_holdout": False,
        })
    return folds


def build_training_readiness(
    *, snapshot_root: str | Path, label_root: str | Path, receipt_root: str | Path,
    start_date: date, end_date: date, now: datetime,
    slots: tuple[str, ...] = ("open", "preopen"),
    min_train_dates: int = 10, validation_dates: int = 5,
) -> dict:
    now = _aware(now, "now")
    if not slots or len(set(slots)) != len(slots) or set(slots) - {"open", "preopen"}:
        raise TrainingDataError("slots must be unique open/preopen values")
    if (type(start_date) is not date or type(end_date) is not date or start_date > end_date):
        raise TrainingDataError("invalid requested date range")
    # Validate split arguments even if no snapshots exist.
    expanding_readiness_splits([], min_train_dates=min_train_dates, validation_dates=validation_dates)
    generator_sources = _source_manifest()
    paths = discover_r1_snapshots(snapshot_root, start_date=start_date, end_date=end_date, slots=slots)
    rows, excluded, manifests = [], [], []
    keys, seen_dates = set(), set()
    for path in paths:
        try:
            evidence = load_recommendation_evidence(
                path, label_root=label_root, receipt_root=receipt_root, now=now,
            )
        except EvidenceUnavailable as exc:
            excluded.append({"snapshot_path": str(path), "reason": str(exc)})
            continue
        snapshot = evidence["snapshot"]
        identity = (snapshot["asof"], snapshot["slot"])
        if identity in seen_dates:
            raise TrainingDataError("duplicate R1 date/slot evidence")
        seen_dates.add(identity)
        exported, row_excluded = _export_evidence(evidence, now)
        for row in exported:
            key = (row["metadata"]["date"], row["metadata"]["slot"], row["metadata"]["coin"])
            if key in keys:
                raise TrainingDataError("duplicate training row identity")
            keys.add(key)
        rows.extend(exported)
        excluded.extend(row_excluded)
        manifests.append(evidence["manifest"])
    rows.sort(key=lambda r: (r["metadata"]["date"], r["metadata"]["slot"], r["metadata"]["rank"]))
    folds = expanding_readiness_splits(rows, min_train_dates=min_train_dates, validation_dates=validation_dates)
    ready = sum(f["status"] == "split_ready" for f in folds)
    aliases = []
    for left, right in KNOWN_FEATURE_ALIASES:
        comparable = [r["model_features"] for r in rows
                      if r["model_features"][left] is not None and r["model_features"][right] is not None]
        aliases.append({"columns": [left, right], "comparable_rows": len(comparable),
                        "identical_rows": sum(values[left] == values[right] for values in comparable),
                        "note": "known return/ROC aliases; not two independent sources of information"})
    report = {
        "schema": REPORT_SCHEMA, "as_of": now.isoformat(),
        "status": "split_ready" if ready else "insufficient_history",
        "model_fitted": False, "deployable": False, "promotion_status": "NOT_EVALUATED",
        "purpose": "frozen R1 data and chronological-split readiness only",
        "feature_columns": list(ALLOWED_FEATURES), "outcome_columns": list(OUTCOME_FIELDS),
        "feature_schema_sha256": _digest(list(ALLOWED_FEATURES)),
        "requested": {"start_date": start_date.isoformat(), "end_date": end_date.isoformat(), "slots": list(slots)},
        "methodology": {
            "cohort": "successful-receipt R1 full universe; was_delivered marks the original Top3",
            "features": "stored values only; no imputation, recomputation, scores, or outcome features",
            "slot_handling": "slot preserved; all coins and slots grouped globally by date",
            "split_rule": "max(train outcome_end,label_created_at) < min(validation decision_started_at)",
            "availability_note": "current artifact creation is conservative; earlier availability after regeneration is not inferred",
            "min_train_dates_initial": min_train_dates, "validation_dates_initial": validation_dates,
            "threshold_note": "diagnostic initial values, not fitted choices or promotion gates",
            "is_untouched_holdout": False,
            "holdout_note": "already-observed history; future untouched validation is still required",
            "return_unit": "fraction", "round_trip_cost_fraction": ROUND_TRIP_COST_PCT,
            "cost_note": "copied existing net labels; no additional deduction",
        },
        "summary": {**_sample_summary(rows), "snapshots_discovered": len(paths),
                    "snapshots_included": len(manifests), "folds": len(folds), "ready_folds": ready,
                    "by_slot": {slot: _sample_summary([r for r in rows if r["metadata"]["slot"] == slot])
                                for slot in slots}},
        "feature_missing_counts": {name: sum(r["model_features"][name] is None for r in rows)
                                   for name in ALLOWED_FEATURES},
        "excluded": excluded, "input_manifests": manifests,
        "generator_sources": generator_sources, "feature_alias_diagnostics": aliases,
        "folds": folds, "rows": rows,
    }
    if _source_manifest() != generator_sources:
        raise TrainingDataError("generator sources changed during export")
    report["report_payload_sha256"] = _digest(report)
    return report


def readiness_summary(report: dict) -> dict:
    """Small stdout view; the optional export contains the full frozen rows."""
    return {key: value for key, value in report.items()
            if key not in {"rows", "input_manifests"}}

"""Audit frozen preopen target-window coverage; no fitting, DB access or sending.

This is a readiness diagnostic, not a target export or a model experiment.
Current-day outcomes must come from the same coin on the previous CALENDAR
day's preopen window, never from a nearby date or the 09:15 open window.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
os.environ["PRELUDE_FORBID_TELEGRAM"] = "1"
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from ops.artifact_provenance import (  # noqa: E402
    canonical_json_bytes, file_set_identity, sha256_bytes, strict_json_object,
)
from scripts.build_recommend_training_data import write_new_report  # noqa: E402
from signals.recommend_experiment_data import load_experiment_data  # noqa: E402
from signals.recommend_slot_diagnostics import (  # noqa: E402
    _validate, load_slot_metadata,
)
from signals.recommend_training_data import ALLOWED_FEATURES, GENERATOR_SOURCES  # noqa: E402

CONFIG = {
    "version": "recommend_horizon_coverage.v1",
    "source_slot": "preopen", "prior_calendar_days": 1,
    "window_start_kst": "09:00:00", "window_hours": 24,
    "features": list(ALLOWED_FEATURES),
    "missing_policy": "report every original candidate; never widen date or slot",
    "folds": "existing frozen readiness folds; label availability audit only",
    "model_fitted": False, "training_ready": False,
}
DESIGN_SCHEMA = "recommend_horizon_coverage_design.v1"
DESIGN_ID = "r1_horizon_coverage_20260907_v1"
SOURCES = tuple(dict.fromkeys((
    "scripts/audit_recommend_horizon_data.py", "signals/recommend_experiment_data.py",
    "signals/recommend_slot_diagnostics.py", "signals/recommend_experiment_eval.py",
    *GENERATOR_SOURCES,
)))


def _same(left: object, right: object, reason: str) -> None:
    if canonical_json_bytes(left) != canonical_json_bytes(right):
        raise ValueError(reason)


def _digest(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _aware(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return stamp.tz_convert("UTC")


def _features(row: dict) -> dict:
    return {name: None if pd.isna(row[name]) else row[name] for name in ALLOWED_FEATURES}


def analyze_coverage(frame: pd.DataFrame, metadata: dict, folds: list[dict]) -> dict:
    """Preserve the whole candidate population; inspect identity and timing only."""
    work, _ = _validate(frame, metadata)
    if not work["label_status"].isin(["labeled", "halted_no_observations"]).all():
        raise ValueError("unsupported label status")
    if not work["label_available"].eq(work["label_status"].eq("labeled")).all():
        raise ValueError("label availability contradicts status")
    records = work.to_dict("records")
    indexed = {(r["date"], r["slot"], r["coin"]): r for r in records}
    slots = {(m["date"], m["slot"]): m for m in metadata.values()}
    rows = []
    for row in records:
        if row["slot"] != "preopen":
            continue
        day = date.fromisoformat(row["date"])
        previous = (day - timedelta(days=1)).isoformat()
        expected_start = pd.Timestamp(previous, tz="Asia/Seoul") + pd.Timedelta(hours=9)
        expected_end = expected_start + pd.Timedelta(days=1)
        current_meta = metadata[row["snapshot_id"]]
        if (_aware(current_meta["execution_start_at"]) != expected_end
                or _aware(current_meta["outcome_end_at"]) != expected_end + pd.Timedelta(days=1)):
            raise ValueError("current preopen window is not exact calendar 09:00/24h")
        prior_meta = slots.get((previous, "preopen"))
        prior = indexed.get((previous, "preopen", row["coin"]))
        if prior_meta is not None and (
            _aware(prior_meta["execution_start_at"]) != expected_start
            or _aware(prior_meta["outcome_end_at"]) != expected_end
        ):
            raise ValueError("prior preopen window is not exact calendar 09:00/24h")
        if prior_meta is None:
            control_status = "missing_prior_preopen_snapshot"
        elif prior is None:
            control_status = "coin_absent_from_prior_preopen_universe"
        elif prior["label_status"] == "halted_no_observations":
            control_status = "prior_target_halted_no_observations"
        elif prior["label_status"] == "labeled" and prior["label_available"]:
            control_status = "available"
        else:
            raise ValueError("unsupported prior label status")
        if row["label_status"] not in {"labeled", "halted_no_observations"}:
            raise ValueError("unsupported current label status")
        current_available = row["label_status"] == "labeled" and bool(row["label_available"])
        paired = control_status == "available" and current_available
        available = max(_aware(prior["label_available_at"]), _aware(row["label_available_at"])) if paired else None
        # This is only a test of the tempting prior-open input substitution.
        # It never substitutes its features or its incompatible outcome window.
        prior_open = indexed.get((previous, "open", row["coin"]))
        rows.append({
            "date": row["date"], "snapshot_id": row["snapshot_id"],
            "coin": row["coin"], "rank": int(row["rank"]),
            "was_delivered": bool(row["was_delivered"]),
            "feature_values_sha256": _digest(_features(row)),
            "control_date": previous,
            "control_window_start": expected_start.isoformat(),
            "control_window_end": expected_end.isoformat(),
            "control_status": control_status,
            "control_snapshot_id": prior_meta["snapshot_id"] if prior_meta else None,
            "control_rank": int(prior["rank"]) if prior is not None else None,
            "current_target_status": row["label_status"],
            "pair_available": paired,
            "pair_available_at": available.isoformat() if available is not None else None,
            "prior_open_candidate_exists": prior_open is not None,
            "prior_open_features_exactly_equal": (
                canonical_json_bytes(_features(prior_open)) == canonical_json_bytes(_features(row))
                if prior_open is not None else None
            ),
        })
    daily = []
    for day in sorted({r["date"] for r in rows}):
        group = [r for r in rows if r["date"] == day]
        daily.append({"date": day, "candidates": len(group),
                      "current_labeled": sum(r["current_target_status"] == "labeled" for r in group),
                      "control_available": sum(r["control_status"] == "available" for r in group),
                      "pair_available": sum(r["pair_available"] for r in group),
                      "control_status_counts": dict(sorted(Counter(r["control_status"] for r in group).items()))})
    fold_audit = []
    for fold in folds:
        boundary = _aware(fold["validation_min_decision_started_at"])
        train_dates = set(fold["train_dates"])
        validation_dates = set(fold["validation_dates"])
        if train_dates & validation_dates:
            raise ValueError("fold train and validation dates overlap")
        validation_metadata = [m for m in metadata.values() if m["date"] in validation_dates]
        if not validation_metadata or boundary != min(
            _aware(m["decision_started_at"]) for m in validation_metadata
        ):
            raise ValueError("fold boundary differs from actual validation decisions")
        train = [r for r in rows if r["date"] in train_dates]
        validation = [r for r in rows if r["date"] in validation_dates]
        if any(_aware(metadata[r["snapshot_id"]]["decision_started_at"]) >= boundary for r in train):
            raise ValueError("training decision is not before validation")
        eligible = [r for r in train if r["pair_available"] and _aware(r["pair_available_at"]) < boundary]
        fold_audit.append({
            "fold": fold["fold"], "original_fold_status": fold["status"],
            "validation_min_decision_started_at": boundary.isoformat(),
            "train_candidates": len(train), "pair_available": sum(r["pair_available"] for r in train),
            "pair_available_before_validation": len(eligible),
            "eligible_dates": sorted({r["date"] for r in eligible}),
            "eligible_identity_sha256": _digest([(r["snapshot_id"], r["coin"], r["rank"]) for r in eligible]),
            "validation_candidates_unchanged": len(validation),
            "validation_dates": sorted({r["date"] for r in validation}),
            "training_ready": False,
        })
    return {
        "status": "coverage_audit_completed", "training_ready": False,
        "candidate_rows": len(rows), "dates": len(daily),
        "current_labeled": sum(r["current_target_status"] == "labeled" for r in rows),
        "control_status_counts": dict(sorted(Counter(r["control_status"] for r in rows).items())),
        "paired_rows": sum(r["pair_available"] for r in rows),
        "dates_with_full_control_coverage": sum(d["control_available"] == d["candidates"] for d in daily),
        "prior_open_candidate_rows": sum(r["prior_open_candidate_exists"] for r in rows),
        "prior_open_exact_feature_rows": sum(r["prior_open_features_exactly_equal"] is True for r in rows),
        "daily": daily, "folds": fold_audit, "rows": rows,
        "limitations": [
            "No model fitted, targets exported, or new class-rate/performance analysis; inherited readiness counts are provenance only.",
            "An intersection of consecutive-day universes is not the complete original training population.",
            "Stored preopen canonical paths are timing proxies, not proof of equality to original D1 OHLC targets.",
            "A row's own decision time is not the later training cutoff; both target availabilities must precede validation.",
            "Prior-open feature equality is exact saved-value equality, not full-precision historical reconstruction.",
            "Target-only paired fitting still needs a reviewed common raw panel, features, head configuration and source lineage.",
        ],
    }


def run_audit(input_path: Path, design_path: Path) -> dict:
    paths = {"input": input_path, "design": design_path, **{name: ROOT / name for name in SOURCES}}
    before = file_set_identity(paths, root=ROOT)
    if not all(item["exists"] for item in before.values()):
        raise ValueError("missing audit input, design or source")
    design = strict_json_object(design_path)
    _same({key: design.get(key) for key in ("schema", "design_id", "scope", "input_sha256", "configuration")},
          {"schema": DESIGN_SCHEMA, "design_id": DESIGN_ID, "scope": "read_only_horizon_coverage",
           "input_sha256": before["input"]["sha256"], "configuration": CONFIG}, "unreviewed audit design")
    dataset = load_experiment_data(input_path)
    frame_before = dataset["frame"].copy(deep=True)
    folds_before = canonical_json_bytes(dataset["folds"])
    identities = dataset["provenance"]["files"]
    data_paths = {name: path if (path := Path(item["path"])).is_absolute() else ROOT / path
                  for name, item in identities.items()}

    def check() -> None:
        _same(file_set_identity(paths, root=ROOT), before, "audit input/design/source changed")
        _same(file_set_identity(data_paths, root=ROOT), identities, "frozen evidence changed")
        _same(CONFIG, design["configuration"], "runtime configuration changed")
        if not dataset["frame"].equals(frame_before):
            raise ValueError("audit input frame changed")
        if canonical_json_bytes(dataset["folds"]) != folds_before:
            raise ValueError("audit input folds changed")

    check()
    metadata = load_slot_metadata(dataset)
    metadata_before = canonical_json_bytes(metadata)
    check()
    audit = analyze_coverage(dataset["frame"], metadata, dataset["folds"])
    _same(metadata, json.loads(metadata_before), "audit metadata changed")
    report = {"schema": CONFIG["version"], "created_at": datetime.now(timezone.utc).isoformat(),
              "scope": "read_only_horizon_coverage", "model_fitted": False, "deployable": False,
              "live_model_changed": False, "promotion_status": "NOT_EVALUATED",
              "is_untouched_holdout": False, "design": design, "generator_files": before,
              "data_provenance": dataset["provenance"], "data_summary": dataset["summary"], "audit": audit}
    report["report_payload_sha256"] = _digest(report)
    check()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--design", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="optional NEW project-local report")
    args = parser.parse_args(argv)
    try:
        report = run_audit(args.input, args.design)
        if args.output:
            write_new_report(args.output, report, protected_roots=(args.input, args.design))
        print(json.dumps({key: value for key, value in report["audit"].items()
                          if key not in {"rows", "daily", "folds", "limitations"}}, allow_nan=False))
        return 0
    except (ValueError, RuntimeError, KeyError, TypeError, OSError) as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc), "model_fitted": False}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Rebind a frozen readiness export without collecting, fitting, or writing.

The ranking population includes halted candidates omitted by the training
export.  Only ``ALLOWED_FEATURES`` are model inputs: availability, outcomes,
delivery flags, ranks, and frozen baseline scores are never selection features.
"""
from __future__ import annotations

import math
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from ops.artifact_provenance import (
    canonical_json_bytes,
    file_identity,
    file_set_identity,
    resolve_identity_path,
    sha256_bytes,
    strict_json_object,
)
from ops.recommendation_evidence import EvidenceError, load_recommendation_evidence
from signals import recommend_training_data as training

ROOT = Path(__file__).resolve().parent.parent
ALLOWED_FEATURES = training.ALLOWED_FEATURES
OUTCOME_FIELDS = training.OUTCOME_FIELDS
BASELINE_FIELDS = ("p_up10", "p_dn5", "p_dn10", "exp_downside", "rr_ratio")
METADATA_FIELDS = (
    "date", "slot", "coin", "rank", "snapshot_id", "decision_started_at",
    "label_available_at", "outcome_end_at", "was_delivered", "label_status",
    "label_available",
)


class ExperimentDataError(EvidenceError):
    """The frozen report and its current exact inputs cannot be rebound."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ExperimentDataError(reason)


def _same(actual: Any, expected: Any, reason: str) -> None:
    # Python equality would equate True/1 and 1/1.0 in a tampered export.
    _require(canonical_json_bytes(actual) == canonical_json_bytes(expected), reason)


def _paths_from_manifest(manifest: dict) -> dict[str, Path]:
    files = manifest["files"]
    _require(isinstance(files, dict) and set(files) == {"snapshot", "label", "receipt"},
             "invalid evidence file manifest")
    paths = {}
    for role, identity in files.items():
        _require(isinstance(identity, dict) and identity.get("exists") is True,
                 f"missing frozen {role} identity")
        value = identity.get("path")
        _require(isinstance(value, str) and bool(value), f"invalid {role} path")
        paths[role] = resolve_identity_path(value, root=ROOT)
    return paths


def _score_values(candidate: dict) -> dict:
    scores = {}
    for name in BASELINE_FIELDS:
        value = candidate[name]
        _require(not isinstance(value, bool) and isinstance(value, (int, float))
                 and math.isfinite(value), f"invalid frozen baseline score: {name}")
        if name.startswith("p_"):
            _require(0 <= value <= 1, f"baseline probability out of range: {name}")
        scores[name] = value
    return scores


def _frame_rows(evidence: dict, exported: list[dict]) -> list[dict]:
    snapshot, label = evidence["snapshot"], evidence["label"]
    sid = snapshot["snapshot_id"]
    indexed = {
        (row["metadata"]["snapshot_id"], row["metadata"]["coin"], row["metadata"]["rank"]): row
        for row in exported
    }
    _require(len(indexed) == len(exported), "duplicate exported candidate join")
    outcomes = {(row["coin"], row["rank"]): row for row in label["rows"]}
    candidates = {(row["coin"], row["rank"]): row for row in snapshot["universe"]}
    _require(len(outcomes) == len(label["rows"]), "duplicate label candidate join")
    _require(len(candidates) == len(snapshot["universe"]), "duplicate snapshot candidate join")
    _require(candidates.keys() == outcomes.keys(), "missing candidate outcome join")
    delivered = {row["coin"] for row in snapshot["top3"]}
    end = training._aware(label["path_window_end"], "path_window_end")
    available = max(end, training._aware(label["labeled_at"], "labeled_at"))
    started = training._aware(snapshot["decision_started_at"], "decision_started_at")
    rows, matched = [], set()
    for (coin, rank), candidate in candidates.items():
        key = (sid, coin, rank)
        outcome = outcomes[(coin, rank)]
        status = outcome["label_status"]
        _require(status in {"labeled", "halted_no_observations"}, "unsupported label status")
        if status == "labeled":
            _require(key in indexed, "missing labeled candidate export join")
            row = indexed[key]
            features, targets = row["model_features"], row["outcomes"]
            matched.add(key)
        else:
            _require(key not in indexed, "halted candidate appeared in training export")
            features = training._features(candidate, snapshot["feature_columns"])
            # Do not promote stale/flat/missing outcomes to a negative target.
            targets = {name: float("nan") for name in OUTCOME_FIELDS}
        rows.append({
            **features,
            "date": snapshot["asof"], "slot": snapshot["slot"], "coin": coin,
            "rank": rank, "snapshot_id": sid,
            "decision_started_at": started.isoformat(),
            "label_available_at": available.isoformat(), "outcome_end_at": end.isoformat(),
            "was_delivered": coin in delivered, "label_status": status,
            "label_available": status == "labeled",
            **_score_values(candidate), **targets,
        })
    _require(matched == indexed.keys(), "unmatched training export row")
    return rows


def load_experiment_data(readiness_path: Path) -> dict:
    """Return a full-candidate frame and verified chronological folds.

    No new dates are discovered. Every source must match the frozen report at
    both ends of this read, and labels are checked against its original cutoff.
    ``provenance.files`` can be rehashed after a caller's longer experiment.
    """
    try:
        return _load_experiment_data(Path(readiness_path).absolute())
    except ExperimentDataError:
        raise
    except Exception as exc:
        raise ExperimentDataError(f"experiment data invalid: {type(exc).__name__}: {exc}") from exc


def _load_experiment_data(report_path: Path) -> dict:
    report_before = file_identity(report_path, root=ROOT)
    _require(report_before["exists"], "readiness report is missing")
    report = strict_json_object(report_path)
    digest = report.get("report_payload_sha256")
    body = {key: value for key, value in report.items() if key != "report_payload_sha256"}
    _require(digest == sha256_bytes(canonical_json_bytes(body)), "readiness report checksum mismatch")
    _require(report["schema"] == training.REPORT_SCHEMA, "unsupported readiness schema")
    _require(report["model_fitted"] is False and report["deployable"] is False
             and report["promotion_status"] == "NOT_EVALUATED", "invalid readiness-only contract")
    _same(report["feature_columns"], list(ALLOWED_FEATURES), "feature allowlist mismatch")
    _same(report["outcome_columns"], list(OUTCOME_FIELDS), "outcome allowlist mismatch")
    _require(report["feature_schema_sha256"] == training._digest(list(ALLOWED_FEATURES)),
             "feature schema checksum mismatch")
    now = training._aware(report["as_of"], "as_of")
    requested = report["requested"]
    start, end = (date.fromisoformat(requested[name]) for name in ("start_date", "end_date"))
    slots = requested["slots"]
    _require(start <= end and isinstance(slots, list) and bool(slots)
             and len(set(slots)) == len(slots) and not set(slots) - {"open", "preopen"},
             "invalid requested cohort")
    methodology = report["methodology"]
    _require(methodology["is_untouched_holdout"] is False, "observed history is not untouched holdout")
    _same(methodology["return_unit"], "fraction", "return unit mismatch")
    _same(methodology["round_trip_cost_fraction"], training.ROUND_TRIP_COST_PCT, "cost contract mismatch")
    split_args = {
        "min_train_dates": methodology["min_train_dates_initial"],
        "validation_dates": methodology["validation_dates_initial"],
    }
    training.expanding_readiness_splits([], **split_args)
    sources = training._source_manifest()
    _same(sources, report["generator_sources"], "generator sources no longer match frozen export")
    manifests = report["input_manifests"]
    _require(isinstance(manifests, list), "input manifests must be a list")
    paths_by_manifest = [_paths_from_manifest(manifest) for manifest in manifests]
    paths = {"readiness_report": report_path}
    paths.update({f"generator:{name}": training.ROOT / name for name in training.GENERATOR_SOURCES})
    for index, evidence_paths in enumerate(paths_by_manifest):
        paths.update({f"evidence:{index}:{role}": path for role, path in evidence_paths.items()})
    before = file_set_identity(paths, root=ROOT)
    _same(before["readiness_report"], report_before, "readiness report changed during read")
    for name, identity in sources["files"].items():
        _same(before[f"generator:{name}"], identity, "generator changed before rebinding")

    exported, halted, records = [], [], []
    seen_slots, seen_snapshot_ids, seen_candidates = set(), set(), set()
    for index, (manifest, evidence_paths) in enumerate(zip(manifests, paths_by_manifest)):
        for role, identity in manifest["files"].items():
            _same(before[f"evidence:{index}:{role}"], identity, f"frozen {role} bytes or path changed")
        evidence = load_recommendation_evidence(
            evidence_paths["snapshot"], label_root=evidence_paths["label"].parent.parent,
            receipt_root=evidence_paths["receipt"].parent.parent, now=now,
        )
        _same(evidence["manifest"], manifest, "rebound evidence manifest mismatch")
        snapshot = evidence["snapshot"]
        identity = (snapshot["asof"], snapshot["slot"])
        _require(start <= date.fromisoformat(identity[0]) <= end and identity[1] in slots,
                 "snapshot outside frozen requested cohort")
        _require(identity not in seen_slots and snapshot["snapshot_id"] not in seen_snapshot_ids,
                 "duplicate snapshot date/slot or ID")
        seen_slots.add(identity)
        seen_snapshot_ids.add(snapshot["snapshot_id"])
        rows, excluded = training._export_evidence(evidence, now)
        exported.extend(rows)
        halted.extend(excluded)
        for row in _frame_rows(evidence, rows):
            key = (row["date"], row["slot"], row["coin"])
            _require(key not in seen_candidates, "duplicate full-universe candidate")
            seen_candidates.add(key)
            records.append(row)

    exported.sort(key=lambda r: (r["metadata"]["date"], r["metadata"]["slot"], r["metadata"]["rank"]))
    _same(exported, report["rows"], "frozen export rows differ from rebound evidence")
    exclusions = report["excluded"]
    _require(isinstance(exclusions, list), "excluded must be a list")
    row_exclusions, snapshot_exclusions = [], []
    for exclusion in exclusions:
        _require(isinstance(exclusion, dict), "invalid exclusion")
        if "snapshot_id" in exclusion:
            row_exclusions.append(exclusion)
        else:
            _require(set(exclusion) == {"snapshot_path", "reason"}
                     and all(isinstance(value, str) and value for value in exclusion.values()),
                     "invalid unavailable snapshot exclusion")
            snapshot_exclusions.append(exclusion)
    _same(halted, row_exclusions, "halt exclusions differ from rebound evidence")
    folds = training.expanding_readiness_splits(exported, **split_args)
    _same(folds, report["folds"], "chronological folds differ from frozen export")
    ready = sum(fold["status"] == "split_ready" for fold in folds)
    _same(report["status"], "split_ready" if ready else "insufficient_history", "readiness status mismatch")
    expected_summary = {
        **training._sample_summary(exported),
        "snapshots_discovered": len(manifests) + len(snapshot_exclusions),
        "snapshots_included": len(manifests), "folds": len(folds), "ready_folds": ready,
        "by_slot": {slot: training._sample_summary([r for r in exported if r["metadata"]["slot"] == slot])
                    for slot in slots},
    }
    _same(expected_summary, report["summary"], "readiness summary mismatch")
    _same({name: sum(r["model_features"][name] is None for r in exported) for name in ALLOWED_FEATURES},
          report["feature_missing_counts"], "feature missing counts mismatch")
    records.sort(key=lambda row: (row["date"], row["slot"], row["rank"]))
    frame = pd.DataFrame(records, columns=(*ALLOWED_FEATURES, *METADATA_FIELDS,
                                          *BASELINE_FIELDS, *OUTCOME_FIELDS))
    after = file_set_identity(paths, root=ROOT)
    _same(after, before, "input files changed during experiment data read")
    return {
        "frame": frame, "folds": folds, "as_of": report["as_of"],
        "provenance": {
            "root": str(ROOT), "files": before, "files_sha256": training._digest(before),
            "report_file_sha256": before["readiness_report"]["sha256"],
            "report_payload_sha256": digest, "frozen_as_of": report["as_of"],
            "generator_sources": sources, "input_manifests": manifests,
        },
        "summary": {
            "rows": len(records), "labeled_rows": len(exported), "unlabeled_rows": len(halted),
            "snapshots": len(manifests), "dates": len({row["date"] for row in records}),
            "rows_by_slot": dict(sorted(Counter(row["slot"] for row in records).items())),
            "readiness_summary": expected_summary,
            "ranking_population": "all frozen snapshot candidates, including halted unlabeled rows",
            "training_population": "label_available rows only; availability is not a model feature",
            "is_untouched_holdout": False,
        },
    }

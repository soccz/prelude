"""Replay frozen reports to repair P/Q/R measurement coverage; never fit or collect."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
os.environ["PRELUDE_FORBID_TELEGRAM"] = "1"
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from ops.artifact_provenance import (  # noqa: E402
    canonical_json_bytes, file_set_identity, resolve_identity_path, sha256_bytes, strict_json_object,
)
from ops.recommendation_evidence import load_recommendation_evidence  # noqa: E402
from scripts.build_recommend_training_data import write_new_report  # noqa: E402
from scripts.evaluate_recommend_entry_delay import PARITY_FIELDS, SOURCE_FILES, _parity_differences  # noqa: E402
from signals import recommend_slot_diagnostics as slot  # noqa: E402

DESIGN_SCHEMA = "recommend_slot_supplement_design.v1"
DESIGN_ID = "r1_slot_coverage_20260907_v1"
SCOPE = "offline_slot_coverage_supplement_only"
SUPPLEMENT_CONFIG = {
    "version": "recommend_slot_coverage_supplement.v1", "n_boot": 1000, "seed": 42,
    "delays_minutes": [0, 15, 30], "picks_per_leg": 3,
    "Q_rule": "original preopen Top3; unique stored cell exactly matching open start AND end",
    "selection_rule": "unchanged actual delivered snapshot Top3; never insert synthetic candidates",
    "missing_policy": "only required windows; never filter by common_eligible or primary_eligible",
    "parity_fields": list(PARITY_FIELDS), "canonical_parity": "all original delivered picks",
    "Q_overlap_parity": "all original open-label overlaps, not only original eight evaluable dates",
    "statistics": deepcopy(slot.SLOT_CONFIG), "raw_OHLC_reverified": False,
}
SOURCES = ("scripts/complete_recommend_slot_diagnostics.py", "signals/recommend_slot_diagnostics.py",
           "scripts/build_recommend_training_data.py", "ops/artifact_provenance.py",
           "ops/recommendation_evidence.py", "scripts/evaluate_recommend_entry_delay.py")


def _same(left: object, right: object, reason: str) -> None:
    if canonical_json_bytes(left) != canonical_json_bytes(right):
        raise ValueError(reason)


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _checksum(report: dict) -> None:
    body = {key: value for key, value in report.items() if key != "report_payload_sha256"}
    _require(report.get("report_payload_sha256") == sha256_bytes(canonical_json_bytes(body)), "report checksum mismatch")


def _key(record: dict) -> tuple:
    _require(type(record["rank"]) is int and record["rank"] > 0, "invalid record rank")
    slot._day(record["asof"])
    _require(record["slot"] in {"preopen", "open"}, "invalid record slot")
    _require(all(isinstance(record[field], str) and record[field] for field in ("snapshot_id", "coin")), "invalid record identity")
    return tuple(record[field] for field in ("snapshot_id", "coin", "rank", "asof", "slot"))


def _cell_valid(cell: dict, start: pd.Timestamp, cutoff: pd.Timestamp) -> None:
    end = start + pd.Timedelta(days=1)
    _require(slot._time(cell["start_at"], "delay start") == start
             and slot._time(cell["end_at"], "delay end") == end, "delay window mismatch")
    _require(end <= cutoff, "future delay window")
    status = cell["label_status"]
    if status != "labeled":
        _require(status in {"path_incomplete", "invalid_complete_path", "halted_no_observations"}
                 and not cell.get("outcomes"), "invalid unavailable delay cell")
        return
    _require(cell.get("path_complete") is True, "labeled delay path is incomplete")
    outcome = cell["outcomes"]
    _require(set(PARITY_FIELDS) <= set(outcome), "delay outcome fields missing")
    for field in ("up5", "up10", "up20", "dn3", "dn5", "dn10"):
        _require(type(outcome[field]) is bool, "invalid binary delay outcome")
    for field in ("actual_entry_open", "mfe", "mae", "eod_return_gross", "eod_return_net", "tp5_sl3_return_gross", "tp5_sl3_return_net"):
        value = outcome[field]
        _require(type(value) in (int, float) and math.isfinite(value), "invalid numeric delay outcome")
    _require(outcome["actual_entry_open"] > 0 and outcome["mfe"] >= 0 and -1 <= outcome["mae"] <= 0, "invalid delay excursion or entry")
    for field, threshold in (("up5", .05), ("up10", .10), ("up20", .20)):
        _require(outcome[field] == (outcome["mfe"] >= threshold), "upside event/excursion mismatch")
    for field, threshold in (("dn3", -.03), ("dn5", -.05), ("dn10", -.10)):
        _require(outcome[field] == (outcome["mae"] <= threshold), "downside event/excursion mismatch")
    for name in ("eod_return", "tp5_sl3_return"):
        _require(math.isclose(outcome[f"{name}_gross"] - .0015, outcome[f"{name}_net"], abs_tol=1e-12, rel_tol=1e-10), "delay net-cost mismatch")
    passage = outcome["tp5_sl3_first_passage"]
    expected_before = {"neither": None, "tp_first": True, "sl_first": False, "sl_first_same_bar": False}
    _require(passage in expected_before and outcome["tp5_before_sl3"] is expected_before[passage], "invalid first-passage outcome")
    if passage == "neither":
        _require(outcome["first_passage_bar"] is None and outcome["first_passage_at"] is None, "neither has first passage")
    else:
        bar = outcome["first_passage_bar"]
        _require(type(bar) is int and 0 <= bar < 96, "invalid first passage bar")
        stamp = pd.Timestamp(outcome["first_passage_at"])
        stamp = stamp.tz_localize("Asia/Seoul") if stamp.tzinfo is None else stamp.tz_convert("Asia/Seoul")
        _require(stamp == start + pd.Timedelta(minutes=15 * bar), "first passage timestamp mismatch")


def _parity(cell: dict, label: dict, row: dict, context: str) -> None:
    _require(cell["label_status"] == row["label_status"] == "labeled", f"{context}: canonical label unavailable")
    _require(slot._time(cell["start_at"], "start") == slot._time(label["execution_start_at"], "canonical start")
             and slot._time(cell["end_at"], "end") == slot._time(label["path_window_end"], "canonical end"), f"{context}: entry/window mismatch")
    _require(set(PARITY_FIELDS) <= set(row) and set(PARITY_FIELDS) <= set(cell["outcomes"]), f"{context}: missing parity fields")
    _require(not _parity_differences(cell["outcomes"], row), f"{context}: outcome parity mismatch")


def _vector(cell: dict) -> np.ndarray:
    out = cell["outcomes"]
    values = {**out, "whole_path_safe_up10": out["up10"] and not out["dn5"]}
    return np.array([float(values[name]) for name in slot.METRICS])


def complete_slots(diagnostics: dict, delays: dict, evidence_by_sid: dict) -> dict:
    """Pure replay over verified documents; never creates rows in a universe."""
    _require(diagnostics["schema"] == "recommend_selection_diagnostics.v1"
             and delays["schema"] == "recommend_entry_delay.v1" and delays["status"] == "ok"
             and not delays["errors"], "unsupported or failed source report")
    contract = delays["contract"]
    for field, expected in {"ranking": "R1", "cohort": "actual_delivered_snapshot_top3",
                            "delays_minutes": [0, 15, 30], "horizon": "each_start_plus_24h",
                            "round_trip_cost_fraction": .0015, "return_unit": "fraction"}.items():
        _same(contract[field], expected, f"delay contract mismatch: {field}")
    frozen = diagnostics["data_provenance"]["input_manifests"]
    indexed_manifests = {}
    for manifest in delays["inputs"]:
        sid = manifest["snapshot_id"]
        _require(sid not in indexed_manifests, "duplicate delay input manifest")
        indexed_manifests[sid] = manifest
    _same(indexed_manifests, {item["snapshot_id"]: item for item in frozen}, "delay inputs do not match frozen diagnostic inputs")
    _require(len(frozen) == len(indexed_manifests) and set(evidence_by_sid) == set(indexed_manifests), "missing or duplicate frozen snapshot")
    cutoff = slot._time(delays["created_at"], "delay report cutoff")
    expected, groups, zero = {}, {}, {}
    for sid, evidence in evidence_by_sid.items():
        _same(evidence["manifest"], indexed_manifests[sid], "rebound manifest mismatch")
        snapshot, label = evidence["snapshot"], evidence["label"]
        _require(snapshot["snapshot_id"] == sid and len(snapshot["top3"]) == 3, "original three-pick identity missing")
        identity = (snapshot["asof"], snapshot["slot"])
        _require(identity not in groups, "duplicate original date/slot")
        groups[identity] = evidence
        label_rows = {row["coin"]: row for row in label["rows"]}
        for candidate in snapshot["top3"]:
            key = _key({"snapshot_id": sid, "asof": snapshot["asof"], "slot": snapshot["slot"],
                        "coin": candidate["coin"], "rank": candidate["rank"]})
            _require(key not in expected, "duplicate expected original pick")
            expected[key] = (label, label_rows[candidate["coin"]])
    records = {}
    for record in delays["records"]:
        key = _key(record)
        _require(key not in records, "duplicate delay record identity")
        records[key] = record
    _require(records.keys() == expected.keys(), "missing or extra original-pick delay records")
    for key, record in records.items():
        label, canonical = expected[key]
        base = slot._time(label["execution_start_at"], "canonical entry")
        cells = record["delays"]
        _require(isinstance(cells, dict) and "0" in cells and set(cells) <= {"0", "15", "30"}, "invalid delay keys or missing zero")
        for minutes, cell in cells.items():
            _cell_valid(cell, base + pd.Timedelta(minutes=int(minutes)), cutoff)
        _parity(cells["0"], label, canonical, "canonical zero")
        zero[key] = cells["0"]
    included, excluded, daily, matrices = [], [], [], []
    overlap_count = 0
    for day in sorted({identity[0] for identity in groups}):
        if not all((day, name) in groups for name in ("preopen", "open")):
            excluded.append({"date": day, "reason": "missing_slot", "coins": []})
            continue
        pre, opened = groups[(day, "preopen")], groups[(day, "open")]
        pkeys, rkeys = [], []
        for evidence, keys in ((pre, pkeys), (opened, rkeys)):
            snap = evidence["snapshot"]
            keys.extend(_key({"snapshot_id": snap["snapshot_id"], "asof": day, "slot": snap["slot"],
                              "coin": row["coin"], "rank": row["rank"]}) for row in snap["top3"])
        target_start = slot._time(opened["label"]["execution_start_at"], "open entry")
        target_end = slot._time(opened["label"]["path_window_end"], "open end")
        _require(target_start >= slot._time(pre["label"]["execution_start_at"], "preopen entry"), "backward evaluation window forbidden")
        open_labels = {row["coin"]: row for row in opened["label"]["rows"]}
        q, missing = [], []
        for key in pkeys:
            candidates = [(minutes, cell) for minutes, cell in records[key]["delays"].items()
                          if slot._time(cell["start_at"], "start") == target_start
                          and slot._time(cell["end_at"], "end") == target_end]
            _require(len(candidates) <= 1, "multiple exact-window Q cells")
            if not candidates:
                missing.append({"coin": key[1], "reason": "exact_open_window_not_recorded"})
                continue
            minutes, cell = candidates[0]
            # Every overlap must agree, even if another pick makes this date unusable.
            if key[1] in open_labels:
                _parity(cell, opened["label"], open_labels[key[1]], "Q open overlap")
                overlap_count += 1
            if cell["label_status"] != "labeled":
                missing.append({"coin": key[1], "reason": "required_Q_window_unavailable"})
                continue
            q.append((key[1], minutes, cell))
        if missing:
            excluded.append({"date": day, "reason": "required_Q_window_missing_or_unavailable", "details": missing,
                             "coins": sorted(item["coin"] for item in missing)})
            continue
        legs = np.stack((np.mean([_vector(zero[key]) for key in pkeys], axis=0),
                         np.mean([_vector(cell) for _, _, cell in q], axis=0),
                         np.mean([_vector(zero[key]) for key in rkeys], axis=0)))
        values = slot._parts(legs)
        slot._additivity_error(values)
        membership = {"date": day, "preopen_snapshot_id": pkeys[0][0], "open_snapshot_id": rkeys[0][0],
                      "P": [key[1] for key in pkeys], "Q": [key[1] for key in pkeys], "R": [key[1] for key in rkeys]}
        included.append(membership)
        daily.append({**membership, "metrics": slot._named(values),
                      "Q_sources": [{"coin": coin, "delay_minutes": int(minutes), "start_at": cell["start_at"],
                                     "end_at": cell["end_at"], "source": "frozen_preopen_pick_delay_record"} for coin, minutes, cell in q]})
        matrices.append(values)
    by_day = {row["date"]: row for row in daily}
    original_days = set()
    for row in diagnostics["slots"]["daily"]:
        day = row["date"]
        _require(day not in original_days and day in by_day, "original evaluable date lost or duplicated")
        original_days.add(day)
        for field in ("P", "Q", "R", "preopen_snapshot_id", "open_snapshot_id"):
            _same(by_day[day][field], row[field], "original three-leg membership changed")
        for part in slot.PARTS:
            for metric in slot.METRICS:
                _require(math.isclose(by_day[day]["metrics"][part][metric], row["metrics"][part][metric], rel_tol=1e-10, abs_tol=1e-12),
                         "original three-leg daily value changed")
    _same(sorted(original_days), sorted(diagnostics["slots"]["cohort"]["date_list"]), "original cohort/daily mismatch")
    matrix = np.stack(matrices) if matrices else np.empty((0, len(slot.PARTS), len(slot.METRICS)))
    cohort_hash = sha256_bytes(canonical_json_bytes(included))
    result = {
        "schema": SUPPLEMENT_CONFIG["version"], "scope": SCOPE,
        "status": "historical_coverage_supplement" if included else "no_common_evaluable_dates",
        "deployable": False, "model_fitted": False, "live_model_changed": False,
        "is_untouched_holdout": False, "config": deepcopy(SUPPLEMENT_CONFIG),
        "cohort": {"cohort_id": f"slot-supplement-{cohort_hash[:20]}", "cohort_sha256": cohort_hash,
                   "dates": len(included), "date_list": [row["date"] for row in included],
                   "picks_per_leg": 3, "pick_rows_per_leg": 3 * len(included), "membership": included},
        "coverage": {"input_snapshots": len(groups), "original_pick_records": len(records), "input_dates": len({day for day, _ in groups}),
                     "original_evaluable_dates": len(original_days), "included_dates": len(included),
                     "excluded_dates": len(excluded), "excluded": excluded},
        "parity": {"canonical_zero_picks_checked": len(zero), "Q_open_overlap_picks_checked": overlap_count,
                   "original_daily_dates_checked": len(original_days), "fields": list(PARITY_FIELDS), "status": "passed"},
        "daily": daily, "summary": slot._joint_summary(matrix, SUPPLEMENT_CONFIG["n_boot"], SUPPLEMENT_CONFIG["seed"]),
        "leave_one_date_out": [{"omitted_date": included[i]["date"], "remaining_dates": len(matrix) - 1,
                                "means": slot._named(np.delete(matrix, i, axis=0).mean(axis=0))}
                               for i in range(len(matrix))] if len(matrix) > 1 else [],
        "source_cutoffs": {"diagnostic_frozen_as_of": diagnostics["data_provenance"]["frozen_as_of"],
                           "delay_created_at": delays["created_at"]},
        "interpretation": ["Historical replay appendix repairing measurement coverage; no new selection rule or optimized delay.",
                           "No synthetic candidates or scores are added to either original universe.",
                           "Only the needed Q window controls availability; common_eligible/primary_eligible are not filters.",
                           "Raw OHLC bodies are absent. Cached path hashes are provenance metadata, not independent verification of DB observations.",
                           "Separate frozen source cutoffs are retained; the supplement is not a new untouched forward test.",
                           "P/Q/R use the same dates and weights. Window/selection arithmetic is not a causal freshness effect."]}
    canonical_json_bytes(result)
    return result


def run_supplement(diagnostics_path: Path, delays_path: Path, design_path: Path) -> dict:
    paths = {"diagnostics": diagnostics_path, "delays": delays_path, "design": design_path,
             **{name: ROOT / name for name in SOURCES}}
    before = file_set_identity(paths, root=ROOT)
    _require(all(item["exists"] for item in before.values()), "supplement input/design/source missing")
    design = strict_json_object(design_path)
    _require(design["schema"] == DESIGN_SCHEMA and design["design_id"] == DESIGN_ID and design["scope"] == SCOPE, "unapproved supplement design")
    _same(design["input_sha256"], {name: before[name]["sha256"] for name in ("diagnostics", "delays")}, "design input hash mismatch")
    _same(design["configuration"], SUPPLEMENT_CONFIG, "supplement configuration mismatch")
    diagnostics, delays = strict_json_object(diagnostics_path), strict_json_object(delays_path)
    _checksum(diagnostics)
    _checksum(delays)
    identities = {}
    for prefix, files in (("diagnostic_generator", diagnostics["generator_files"]),
                          ("diagnostic_evidence", diagnostics["data_provenance"]["files"]), ("delay_code", delays["code"])):
        for name, item in files.items():
            identities[f"{prefix}:{name}"] = item
    _require(set(delays["code"]) == set(SOURCE_FILES), "delay generator manifest incomplete")
    bound_paths = {key: resolve_identity_path(item["path"], root=ROOT) for key, item in identities.items()}
    _same(file_set_identity(bound_paths, root=ROOT), identities, "frozen report evidence or generator changed")
    evidence = {}
    cutoff = min(slot._time(diagnostics["data_provenance"]["frozen_as_of"], "diagnostic cutoff"),
                 slot._time(delays["created_at"], "delay cutoff"))
    for manifest in diagnostics["data_provenance"]["input_manifests"]:
        files = {key: resolve_identity_path(item["path"], root=ROOT) for key, item in manifest["files"].items()}
        rebound = load_recommendation_evidence(files["snapshot"], label_root=files["label"].parent.parent,
                                               receipt_root=files["receipt"].parent.parent, now=cutoff.to_pydatetime())
        _same(rebound["manifest"], manifest, "strict original evidence rebind failed")
        sid = manifest["snapshot_id"]
        _require(sid not in evidence, "duplicate frozen evidence ID")
        evidence[sid] = rebound
    _same(file_set_identity(paths, root=ROOT), before, "supplement sources changed before replay")
    result = complete_slots(diagnostics, delays, evidence)
    result.update(design=design, generator_files=before, source_file_identities=identities,
                  input_report_hashes={name: before[name]["sha256"] for name in ("diagnostics", "delays")})
    result["report_payload_sha256"] = sha256_bytes(canonical_json_bytes(result))
    _same(file_set_identity(bound_paths, root=ROOT), identities, "frozen evidence changed during replay")
    _same(file_set_identity(paths, root=ROOT), before, "supplement source/input changed during replay or serialization")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("diagnostics", "delays", "design"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="optional NEW project-local JSON; never overwrite")
    args = parser.parse_args(argv)
    try:
        report = run_supplement(args.diagnostics, args.delays, args.design)
        if args.output is not None:
            write_new_report(args.output, report, protected_roots=(args.diagnostics, args.delays, args.design))
        print(json.dumps({key: report[key] for key in ("status", "deployable", "coverage", "parity", "summary")}, ensure_ascii=False, allow_nan=False))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc), "deployable": False}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

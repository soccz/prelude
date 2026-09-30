"""Extend the frozen September 7 upper-head comparison onto fixed new dates.

Offline only. Freeze a new design with --prepare-design before running --design.
Neither the original study nor any production artifact is modified.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PRELUDE_FORBID_TELEGRAM"] = "1"
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ops.artifact_provenance import (  # noqa: E402
    canonical_json_bytes, file_set_identity, sha256_bytes, strict_json_object,
)
from scripts.build_recommend_training_data import write_new_report  # noqa: E402
from scripts.compare_recommend_upper import SOURCES as ORIGINAL_SOURCES  # noqa: E402
from signals.recommend_experiment_data import load_experiment_data  # noqa: E402
from signals.recommend_experiment_eval import evaluate_comparison  # noqa: E402
from signals.recommend_upper_experiment import (  # noqa: E402
    EXPERIMENT_CONFIG, run_upper_comparison,
)

REFERENCE = "_workspace/recommend_upper_comparison_20260907_v2.json"
SOURCES = (*ORIGINAL_SOURCES, "scripts/extend_recommend_upper.py")
SPEC = {
    "design_id": "r1_upper_extension_20260930_v1",
    "scope": "offline_comparison_only",
    "requested": {"start_date": "2026-07-26", "end_date": "2026-09-29",
                  "slots": ["open", "preopen"]},
    "evaluation_window": {"start_date": "2026-09-08", "end_date": "2026-09-29"},
    "primary_slot": "open", "secondary_slot": "preopen",
    "statistics": {"n_boot": 1000, "seed": 42},
    "no_tuning": True, "automatic_adoption": False,
    "is_untouched_holdout": False,
}


def _same(left: object, right: object, reason: str) -> None:
    if canonical_json_bytes(left) != canonical_json_bytes(right):
        raise ValueError(reason)


def _files(input_path: Path) -> dict[str, Path]:
    return {"input": input_path, "reference": ROOT / REFERENCE,
            **{name: ROOT / name for name in SOURCES}}


def prepare_design(input_path: Path) -> dict:
    """Bind the unchanged original kernel and the exact extended input, no fit."""
    paths = _files(input_path)
    bundle = file_set_identity(paths, root=ROOT)
    if not all(item["exists"] for item in bundle.values()):
        raise ValueError("extension input, reference, or source is missing")
    reference = strict_json_object(paths["reference"])
    digest = reference.pop("report_payload_sha256", None)
    if digest != sha256_bytes(canonical_json_bytes(reference)):
        raise ValueError("original report payload digest mismatch")
    if (reference.get("schema") != "recommend_upper_comparison.v1"
            or reference.get("design_id") != "r1_upper_compare_20260907_v1"):
        raise ValueError("wrong original comparison")
    _same(reference["configuration"], EXPERIMENT_CONFIG, "original model configuration changed")
    for name in ORIGINAL_SOURCES:
        if bundle[name]["sha256"] != reference["generator_files"][name]["sha256"]:
            raise ValueError(f"original study source changed: {name}")
    readiness = strict_json_object(input_path)
    _same(readiness.get("requested"), SPEC["requested"], "extension history range changed")
    _same(file_set_identity(paths, root=ROOT), bundle, "inputs changed during design preparation")
    return {
        "schema": "recommend_upper_extension_design.v1",
        "specification": SPEC, "model_config": EXPERIMENT_CONFIG,
        "frozen_bundle": bundle,
    }


def _window_frame(frame):
    window = SPEC["evaluation_window"]
    return frame.loc[frame["date"].between(window["start_date"], window["end_date"])].copy()


def _coverage(frame, predictions, evaluation: dict) -> dict:
    window = SPEC["evaluation_window"]
    first, last = (date.fromisoformat(window[key]) for key in ("start_date", "end_date"))
    expected = {(str(first + timedelta(days=i)), slot)
                for i in range((last - first).days + 1) for slot in SPEC["requested"]["slots"]}
    available = set(_window_frame(frame)[["date", "slot"]].itertuples(index=False, name=None))
    predicted = set(predictions[["date", "slot"]].itertuples(index=False, name=None))
    if not predicted <= available <= expected:
        raise ValueError("extension population is outside the fixed window")
    def pairs(items):
        return [{"date": day, "slot": slot} for day, slot in sorted(items)]

    return {
        "expected_calendar_dates": (last - first).days + 1,
        "expected_date_slots": len(expected),
        "evidence_date_slots": len(available), "predicted_date_slots": len(predicted),
        "missing_evidence": pairs(expected - available),
        "no_outer_prediction": pairs(available - predicted),
        "comparison_exclusions": evaluation["excluded"],
        "common_cohort": evaluation["coverage"],
        "missing_is_not_zero_return": True,
    }


def run_extension(input_path: Path, design_path: Path) -> dict:
    design_identity = file_set_identity({"design": design_path}, root=ROOT)
    design = strict_json_object(design_path)
    _same(design, prepare_design(input_path), "frozen extension design or inputs changed")
    paths = {**_files(input_path), "design": design_path}
    before = {**design["frozen_bundle"], **design_identity}
    dataset = load_experiment_data(input_path)
    evidence = dataset["provenance"]["files"]
    evidence_paths = {name: ROOT / item["path"] for name, item in evidence.items()}

    def unchanged(stage):
        _same(file_set_identity(paths, root=ROOT), before, f"study inputs changed {stage}")
        _same(file_set_identity(evidence_paths, root=ROOT), evidence,
              f"frozen evidence changed {stage}")

    unchanged("before fitting")
    started = datetime.now(timezone.utc).isoformat()
    experiment = run_upper_comparison(dataset["frame"], dataset["folds"])
    _same(experiment["config"], EXPERIMENT_CONFIG, "runtime model configuration changed")
    predictions = _window_frame(experiment["predictions"])
    evaluation = evaluate_comparison(predictions, **SPEC["statistics"])
    report = {
        "schema": "recommend_upper_extension.v1", "design_id": SPEC["design_id"],
        "status": "historical_extension_completed",
        "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
        "deployable": False, "promotion_status": "NOT_EVALUATED",
        "live_model_changed": False, "model_artifact_saved": False,
        "is_untouched_holdout": False, "automatic_adoption": False,
        "interpretation": [
            "Fixed old candidates on later dates; no parameter sweep or automatic adoption.",
            "Historical replay, not forward: aggregate outcomes were already reviewed.",
            "Open is primary; preopen is a separate secondary comparison, not a fallback winner.",
            "Earlier dates are training history only, not added to extension performance.",
            "B calibrates saved scores, not raw-head OOF; C is not architecture-only ablation.",
            "Net hypothetical pick outcomes are not actual portfolio returns.",
        ],
        "design": design, "generator_files": before,
        "versions": {name: importlib.metadata.version(name) for name in (
            "numpy", "pandas", "scipy", "scikit-learn", "xgboost")},
        "data_provenance": dataset["provenance"], "data_summary": dataset["summary"],
        "configuration": experiment["config"], "parity": experiment["parity"],
        "fold_audit": experiment["fold_audit"],
        "window_coverage": _coverage(dataset["frame"], predictions, evaluation),
        "evaluation": evaluation,
        "predictions": predictions.astype(object).where(predictions.notna(), None).to_dict(orient="records"),
    }
    report["report_payload_sha256"] = sha256_bytes(canonical_json_bytes(report))
    unchanged("during execution or serialization")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-design", action="store_true")
    mode.add_argument("--design", type=Path)
    parser.add_argument("--output", type=Path, help="NEW project-local JSON; never overwrite")
    args = parser.parse_args(argv)
    try:
        if args.prepare_design:
            if args.output is None:
                raise ValueError("design preparation requires a new --output")
            report = prepare_design(args.input)
        else:
            report = run_extension(args.input, args.design)
        if args.output is not None:
            protected = (args.input, ROOT / REFERENCE, *(ROOT / name for name in SOURCES))
            if args.design is not None:
                protected += (args.design,)
            write_new_report(args.output, report, root=ROOT, protected_roots=protected)
        summary = {key: report[key] for key in (
            "schema", "status", "deployable", "promotion_status", "window_coverage",
        ) if key in report}
        summary["output"] = str(args.output) if args.output else None
        if "evaluation" in report:
            summary["slots"] = {
                slot: {"status": value["status"], "dates": value["dates"],
                       "absolute": {arm: item["mean"] for arm, item in value.get("absolute", {}).items()},
                       "delta_vs_A": {arm: item["mean"] for arm, item in value.get("delta_vs_A", {}).items()}}
                for slot, value in report["evaluation"]["slots"].items()
            }
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc), "deployable": False},
                         ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

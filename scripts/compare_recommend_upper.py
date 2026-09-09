"""Run one frozen, offline R1 upper-head comparison; never deploy or send.

The design and input hashes are checked before fitting. Every result remains
historical development evidence, not an untouched holdout or a live strategy.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PRELUDE_FORBID_TELEGRAM"] = "1"
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ops.artifact_provenance import (  # noqa: E402
    canonical_json_bytes,
    file_set_identity,
    sha256_bytes,
    strict_json_object,
)
from scripts.build_recommend_training_data import write_new_report  # noqa: E402
from signals.recommend_experiment_data import load_experiment_data  # noqa: E402
from signals.recommend_experiment_eval import evaluate_comparison  # noqa: E402
from signals.recommend_training_data import GENERATOR_SOURCES  # noqa: E402
from signals.recommend_upper_experiment import (  # noqa: E402
    EXPERIMENT_CONFIG,
    run_upper_comparison,
)

DESIGN_SCHEMA = "recommend_upper_design.v1"
DESIGN_ID = "r1_upper_compare_20260907_v1"
SOURCES = tuple(dict.fromkeys((
    "scripts/compare_recommend_upper.py",
    "signals/recommend_experiment_data.py",
    "signals/recommend_experiment_eval.py",
    "signals/recommend_upper_experiment.py",
    "signals/recommend.py",
    *GENERATOR_SOURCES,
)))


def _same(left: object, right: object, reason: str) -> None:
    if canonical_json_bytes(left) != canonical_json_bytes(right):
        raise ValueError(reason)


def _paths_from_identities(identities: dict) -> dict[str, Path]:
    result = {}
    for name, item in identities.items():
        value = Path(item["path"])
        result[name] = value if value.is_absolute() else ROOT / value
    return result


def run_experiment(input_path: Path, design_path: Path) -> dict:
    sources = {name: ROOT / name for name in SOURCES}
    before = file_set_identity({"input": input_path, "design": design_path, **sources}, root=ROOT)
    if not all(item["exists"] for item in before.values()):
        raise ValueError("experiment input, design, or generator source is missing")
    design = strict_json_object(design_path)
    if (design.get("schema") != DESIGN_SCHEMA or design.get("design_id") != DESIGN_ID
            or design.get("scope") != "offline_comparison_only"):
        raise ValueError("unapproved offline experiment design")
    if design.get("input_sha256") != before["input"]["sha256"]:
        raise ValueError("frozen input does not match the reviewed design")
    _same(design.get("model_config"), EXPERIMENT_CONFIG, "model configuration differs from reviewed design")
    _same(design.get("statistics"), {"n_boot": 1000, "seed": 42},
          "statistics configuration differs from reviewed design")
    dataset = load_experiment_data(input_path)
    data_files = dataset["provenance"]["files"]
    _same(file_set_identity(_paths_from_identities(data_files), root=ROOT), data_files,
          "frozen evidence changed before fitting")
    _same(file_set_identity({"input": input_path, "design": design_path, **sources}, root=ROOT), before,
          "experiment input, design, or source changed before fitting")
    started = datetime.now(timezone.utc).isoformat()
    experiment = run_upper_comparison(dataset["frame"], dataset["folds"])
    _same(experiment["config"], EXPERIMENT_CONFIG, "runtime model configuration changed")
    evaluation = evaluate_comparison(experiment["predictions"], **design["statistics"])
    # Preserve float precision: DataFrame.to_json's default 10 decimal places
    # can move excursions across a label barrier or alter close ranking ties.
    prediction_frame = experiment["predictions"]
    predictions = prediction_frame.astype(object).where(
        prediction_frame.notna(), None,
    ).to_dict(orient="records")
    report = {
        "schema": "recommend_upper_comparison.v1",
        "design_id": DESIGN_ID, "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "status": "historical_comparison_completed",
        "deployable": False, "promotion_status": "NOT_EVALUATED",
        "live_model_changed": False, "model_artifact_saved": False,
        "is_untouched_holdout": False,
        "scope": "offline_comparison_only",
        "interpretation": [
            "Historical development replay, not prospective challenger performance.",
            "B recalibrates saved probabilities; it does not repair the existing raw head OOF.",
            "C changes training features/history/target window; it is not an architecture-only ablation.",
            "Prediction NaNs are unavailable experiments, never silently replaced with baseline scores.",
            "Returns are hypothetical pick-level net outcomes, not the user's actual portfolio PnL.",
        ],
        "design": design, "generator_files": before,
        "versions": {name: importlib.metadata.version(name)
                     for name in ("numpy", "pandas", "scipy", "scikit-learn", "xgboost")},
        "data_provenance": dataset["provenance"], "data_summary": dataset["summary"],
        "configuration": experiment["config"], "parity": experiment["parity"],
        "fold_audit": experiment["fold_audit"], "evaluation": evaluation,
        "predictions": predictions,
    }
    report["report_payload_sha256"] = sha256_bytes(canonical_json_bytes(report))
    _same(file_set_identity(_paths_from_identities(data_files), root=ROOT), data_files,
          "frozen evidence changed during execution or serialization")
    _same(file_set_identity({"input": input_path, "design": design_path, **sources}, root=ROOT), before,
          "experiment input, design, or source changed during execution or serialization")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="frozen readiness JSON")
    parser.add_argument("--design", type=Path, required=True, help="reviewed fixed-design JSON")
    parser.add_argument("--output", type=Path, help="optional NEW project-local JSON; never overwrite")
    args = parser.parse_args(argv)
    try:
        report = run_experiment(args.input, args.design)
        if args.output is not None:
            write_new_report(args.output, report, protected_roots=(args.input, args.design))
        evaluation = report["evaluation"]
        summary = {key: report[key] for key in (
            "status", "deployable", "promotion_status", "data_summary",
        )}
        summary["parity"] = {key: value for key, value in report["parity"].items()
                             if key != "snapshots_audit"}
        summary["evaluation"] = {key: evaluation.get(key) for key in (
            "status", "data_availability", "input", "coverage",
        )}
        summary["evaluation"]["slots"] = {
            slot: {"status": result["status"], "dates": result["dates"],
                   "absolute": {arm: item["mean"] for arm, item in result.get("absolute", {}).items()},
                   "delta_vs_A": {arm: item["mean"] for arm, item in result.get("delta_vs_A", {}).items()}}
            for slot, result in evaluation.get("slots", {}).items()
        }
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc), "deployable": False},
                         ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Diagnose frozen R1 selection and slot differences, without fitting or sending.

The reviewed design is fixed before outcomes are analyzed. These historical
diagnostics are neither a new forward trial nor authority to change live R1.
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
from signals.recommend_selection_diagnostics import (  # noqa: E402
    SELECTION_CONFIG,
    analyze_selection,
)
from signals.recommend_slot_diagnostics import (  # noqa: E402
    SLOT_CONFIG,
    analyze_slots,
    load_slot_metadata,
)
from signals.recommend_training_data import GENERATOR_SOURCES  # noqa: E402

DESIGN_SCHEMA = "recommend_selection_design.v1"
DESIGN_ID = "r1_selection_slots_20260907_v1"
SCOPE = "offline_diagnostics_only"
STATISTICS = {"n_boot": 1000, "seed": 42}
SOURCES = tuple(dict.fromkeys((
    "scripts/diagnose_recommend_selection.py",
    "scripts/build_recommend_training_data.py",
    "signals/recommend_experiment_data.py",
    "signals/recommend_experiment_eval.py",
    "signals/recommend_selection_diagnostics.py",
    "signals/recommend_slot_diagnostics.py",
    "signals/recommend.py",
    *GENERATOR_SOURCES,
)))


def _same(left: object, right: object, reason: str) -> None:
    if canonical_json_bytes(left) != canonical_json_bytes(right):
        raise ValueError(reason)


def _configuration() -> dict:
    return {"selection": SELECTION_CONFIG, "slots": SLOT_CONFIG, "statistics": STATISTICS}


def _paths(identities: dict) -> dict[str, Path]:
    return {name: path if (path := Path(item["path"])).is_absolute() else ROOT / path
            for name, item in identities.items()}


def run_diagnostics(input_path: Path, design_path: Path) -> dict:
    sources = {name: ROOT / name for name in SOURCES}
    paths = {"input": input_path, "design": design_path, **sources}
    before = file_set_identity(paths, root=ROOT)
    if not all(item["exists"] for item in before.values()):
        raise ValueError("diagnostic input, design, or generator source is missing")
    design = strict_json_object(design_path)
    if (design.get("schema") != DESIGN_SCHEMA or design.get("design_id") != DESIGN_ID
            or design.get("scope") != SCOPE):
        raise ValueError("unreviewed diagnostic design")
    _same(design.get("input_sha256"), before["input"]["sha256"],
          "frozen input differs from reviewed design")
    _same(design.get("configuration"), _configuration(),
          "diagnostic configuration differs from reviewed design")
    started = datetime.now(timezone.utc).isoformat()
    dataset = load_experiment_data(input_path)
    frame_before = dataset["frame"].copy(deep=True)
    data_files = dataset["provenance"]["files"]

    def verify_inputs(stage: str) -> None:
        _same(file_set_identity(paths, root=ROOT), before,
              f"input, design, or source changed {stage}")
        _same(file_set_identity(_paths(data_files), root=ROOT), data_files,
              f"frozen evidence changed {stage}")
        _same(_configuration(), design["configuration"],
              f"runtime configuration changed {stage}")

    verify_inputs("before metadata loading")
    metadata = load_slot_metadata(dataset)
    metadata_before = canonical_json_bytes(metadata)

    def verify_memory(stage: str) -> None:
        if not dataset["frame"].equals(frame_before):
            raise ValueError(f"diagnostic frame changed {stage}")
        if canonical_json_bytes(metadata) != metadata_before:
            raise ValueError(f"slot metadata changed {stage}")

    verify_inputs("before analysis")
    verify_memory("before analysis")
    selection = analyze_selection(dataset["frame"], **STATISTICS)
    verify_memory("during selection analysis")
    slots = analyze_slots(dataset["frame"], metadata, **STATISTICS)
    verify_memory("during slot analysis")
    report = {
        "schema": "recommend_selection_diagnostics.v1",
        "design_id": DESIGN_ID, "scope": SCOPE,
        "status": "historical_diagnostics_completed",
        "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
        "model_fitted": False, "deployable": False, "live_model_changed": False,
        "promotion_status": "NOT_EVALUATED", "is_untouched_holdout": False,
        "design": design, "configuration": _configuration(),
        "generator_files": before, "data_provenance": dataset["provenance"],
        "data_summary": dataset["summary"],
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas")},
        "interpretation": [
            "Full frozen historical cohort; overlaps the earlier upper-head experiment.",
            "Selection and slot diagnostics have separately identified evaluable cohorts.",
            "Selected-subset AUC differences do not establish ranking loss.",
            "Slot differences are arithmetic window/selection decompositions, not causal freshness effects.",
            "Existing net pick-return proxies are not actual user trades or portfolio returns.",
            "No alternative policy, threshold search, model fitting, or automatic adoption.",
        ],
        "selection": selection, "slots": slots, "slot_metadata": metadata,
    }
    report["report_payload_sha256"] = sha256_bytes(canonical_json_bytes(report))
    verify_inputs("during analysis or serialization")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="frozen readiness JSON")
    parser.add_argument("--design", type=Path, required=True, help="reviewed fixed-design JSON")
    parser.add_argument("--output", type=Path, help="optional NEW project-local JSON; never overwrite")
    args = parser.parse_args(argv)
    try:
        report = run_diagnostics(args.input, args.design)
        if args.output is not None:
            write_new_report(args.output, report, protected_roots=(args.input, args.design))
        summary = {key: report[key] for key in (
            "status", "model_fitted", "deployable", "promotion_status", "data_summary",
        )}
        for name in ("selection", "slots"):
            summary[name] = {key: report[name].get(key) for key in ("status", "coverage")}
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc), "deployable": False},
                         ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

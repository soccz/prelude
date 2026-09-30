"""One fixed score-scale ablation of saved upper-head predictions, offline only."""
from __future__ import annotations

import argparse
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

import pandas as pd  # noqa: E402

from ops.artifact_provenance import (  # noqa: E402
    canonical_json_bytes, file_set_identity, sha256_bytes, strict_json_object,
)
from scripts.build_recommend_training_data import write_new_report  # noqa: E402
from signals.recommend_score_alignment import (  # noqa: E402
    CONFIG, evaluate_alignment, transform_predictions,
)

INPUTS = {
    "earlier": "_workspace/recommend_upper_comparison_20260907_v2.json",
    "later": "_workspace/recommend_upper_extension_20260930_v1.json",
}
CONTRACTS = {
    "earlier": {"schema": "recommend_upper_comparison.v1", "design_id": "r1_upper_compare_20260907_v1",
                "start": "2026-08-18", "end": "2026-09-06"},
    "later": {"schema": "recommend_upper_extension.v1", "design_id": "r1_upper_extension_20260930_v1",
              "start": "2026-09-08", "end": "2026-09-29"},
}
SOURCES = (
    "scripts/compare_recommend_score_alignment.py", "signals/recommend_score_alignment.py",
    "signals/recommend_experiment_eval.py", "ops/artifact_provenance.py",
    "scripts/build_recommend_training_data.py",
)
SPEC = {
    "design_id": "r1_score_alignment_20260930_v1", "scope": "offline_ablation_only",
    "primary_slot": "open", "secondary_slot": "preopen", "segments_pooled": False,
    "new_candidates": 1, "parameter_sweep": False, "model_fitted": False,
    "is_untouched_holdout": False, "statistics": {"n_boot": 1000, "seed": 42},
}


def _same(left, right, reason):
    if canonical_json_bytes(left) != canonical_json_bytes(right):
        raise ValueError(reason)


def _paths(identities):
    return {name: ROOT / item["path"] for name, item in identities.items()}


def _load_inputs():
    paths = {**{name: ROOT / path for name, path in INPUTS.items()},
             **{name: ROOT / name for name in SOURCES}}
    before = file_set_identity(paths, root=ROOT)
    if not all(item["exists"] for item in before.values()):
        raise ValueError("missing score alignment input or source")
    reports = {}
    for name, relative in INPUTS.items():
        report = strict_json_object(ROOT / relative)
        digest = report.pop("report_payload_sha256", None)
        if digest != sha256_bytes(canonical_json_bytes(report)):
            raise ValueError(f"source report digest mismatch: {name}")
        if any(report.get(key) != CONTRACTS[name][key] for key in ("schema", "design_id")):
            raise ValueError("wrong source report contract")
        for group in ("generator_files", "data_provenance"):
            files = report[group] if group == "generator_files" else report[group]["files"]
            _same(file_set_identity(_paths(files), root=ROOT), files, "source report evidence changed")
            before.update({f"{name}:{group}:{key}": value for key, value in files.items()})
        reports[name] = report
    _same(file_set_identity(_paths(before), root=ROOT), before, "inputs changed during loading")
    return reports, before


def prepare_design():
    _, bundle = _load_inputs()
    return {"schema": "recommend_score_alignment_design.v1", "specification": SPEC,
            "configuration": CONFIG, "contracts": CONTRACTS, "frozen_bundle": bundle}


def _coverage(frame, contract, evaluation):
    start, end = (date.fromisoformat(contract[key]) for key in ("start", "end"))
    expected = {(str(start + timedelta(days=i)), slot) for i in range((end - start).days + 1)
                for slot in ("open", "preopen")}
    present = set(frame[["date", "slot"]].itertuples(index=False, name=None))
    if not present <= expected:
        raise ValueError("prediction dates outside fixed segment")
    return {"expected_date_slots": len(expected), "predicted_date_slots": len(present),
            "missing_predictions": [{"date": day, "slot": slot} for day, slot in sorted(expected - present)],
            "common_cohort": evaluation["coverage"], "exclusions": evaluation["excluded"]}


def _raw_parity(original, evaluated):
    source = {(row["date"], row["slot"]): row for row in original["selection_audit"]}
    for row in evaluated["selection_audit"]:
        prior = source[(row["date"], row["slot"])]
        _same(row["selected_coins"]["A"], prior["selected_coins"]["A"], "original R1 selection changed")
        _same(row["selected_coins"]["B"], prior["selected_coins"]["C"], "original C selection changed")
    return {"status": "passed", "date_slots": len(evaluated["selection_audit"])}


def run_comparison(design_path):
    design_identity = file_set_identity({"design": design_path}, root=ROOT)
    design = strict_json_object(design_path)
    inputs, bundle = _load_inputs()
    _same(design, {"schema": "recommend_score_alignment_design.v1", "specification": SPEC,
                   "configuration": CONFIG, "contracts": CONTRACTS, "frozen_bundle": bundle},
          "frozen score alignment design changed")
    before = {**bundle, **design_identity}
    _same(file_set_identity(_paths(before), root=ROOT), before, "inputs changed before comparison")
    started = datetime.now(timezone.utc).isoformat()
    segments = {}
    for name, source in inputs.items():
        frame = pd.DataFrame(source["predictions"])
        # Whole candidate population is transformed, including structurally unlabeled rows.
        transformed, audit = transform_predictions(frame)
        evaluation = evaluate_alignment(transformed, **SPEC["statistics"])
        segments[name] = {
            "coverage": _coverage(frame, CONTRACTS[name], evaluation),
            "original_selection_parity": _raw_parity(source["evaluation"], evaluation),
            "transformation_audit": audit, "evaluation": evaluation,
            "predictions": transformed.astype(object).where(transformed.notna(), None).to_dict(orient="records"),
        }
    result = {
        "schema": "recommend_score_alignment_comparison.v1", "design_id": SPEC["design_id"],
        "status": "historical_ablation_completed", "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(), "scope": SPEC["scope"],
        "deployable": False, "live_model_changed": False, "model_fitted": False,
        "model_artifact_saved": False, "automatic_adoption": False,
        "promotion_status": "NOT_EVALUATED", "is_untouched_holdout": False,
        "design": design, "generator_files": before, "segments": segments,
        "interpretation": [
            "One new candidate, proposed after seeing both periods; neither is untouched holdout.",
            "Earlier/later and open/preopen are separate, not pooled for a favorable verdict.",
            "Evaluator A/B/C denote original R1 / original upper C / aligned upper D.",
            "D is a contemporaneous cross-sectional ranking score, not calibrated probability.",
            "No new fit, source model change, new label, forward record, deployment or order.",
            "Net pick outcomes include existing costs once; not actual portfolio performance.",
        ],
    }
    result["report_payload_sha256"] = sha256_bytes(canonical_json_bytes(result))
    _same(file_set_identity(_paths(before), root=ROOT), before, "inputs changed during comparison or serialization")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-design", action="store_true")
    mode.add_argument("--design", type=Path)
    parser.add_argument("--output", type=Path, help="optional NEW project-local JSON; never overwrite")
    args = parser.parse_args(argv)
    try:
        if args.prepare_design and args.output is None:
            raise ValueError("design preparation requires a new --output")
        result = prepare_design() if args.prepare_design else run_comparison(args.design)
        if args.output is not None:
            identities = result.get("generator_files", result.get("frozen_bundle", {}))
            write_new_report(args.output, result, root=ROOT,
                             protected_roots=tuple(_paths(identities).values()))
        summary = {key: result[key] for key in ("schema", "status", "deployable") if key in result}
        if "segments" in result:
            summary["segments"] = {
                name: {"coverage": value["coverage"], "slots": {
                    slot: {"dates": data["dates"], "absolute": {
                        arm: metrics["mean"] for arm, metrics in data.get("absolute", {}).items()},
                        "D_minus_A": data.get("delta_vs_A", {}).get("C", {}).get("mean"),
                        "D_minus_C": data.get("aligned_D_minus_original_C", {}).get("mean")}
                    for slot, data in value["evaluation"]["slots"].items()}}
                for name, value in result["segments"].items()
            }
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc), "deployable": False},
                         ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

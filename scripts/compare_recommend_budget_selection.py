"""Compare one fixed score-budget selector on two saved prediction periods."""

from __future__ import annotations

import argparse
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

import pandas as pd  # noqa: E402
from scripts import compare_recommend_score_alignment as evidence  # noqa: E402
from scripts.build_recommend_training_data import write_new_report  # noqa: E402
from ops.artifact_provenance import (  # noqa: E402
    canonical_json_bytes,
    file_set_identity,
    sha256_bytes,
    strict_json_object,
)
from signals.recommend_budget_selection import CONFIG, evaluate_budget  # noqa: E402

SOURCES = (
    "scripts/compare_recommend_budget_selection.py",
    "signals/recommend_budget_selection.py",
)
SPEC = {
    "design_id": "r1_score_budget_20260930_v1",
    "scope": "offline_selection_only",
    "primary_slot": "open",
    "secondary_slot": "preopen",
    "segments_pooled": False,
    "new_candidates": 1,
    "model_fitted": False,
    "parameter_sweep": False,
    "is_untouched_holdout": False,
    "statistics": {"n_boot": 1000, "seed": 42},
}


def _paths(identities):
    return {name: ROOT / item["path"] for name, item in identities.items()}


def _load():
    sources = file_set_identity({name: ROOT / name for name in SOURCES}, root=ROOT)
    if not all(row["exists"] for row in sources.values()):
        raise ValueError("missing budget-selection source")
    inputs, bundle = evidence._load_inputs()
    bundle.update(sources)
    evidence._same(
        file_set_identity(_paths(bundle), root=ROOT),
        bundle,
        "inputs changed during loading",
    )
    return inputs, bundle


def prepare_design():
    _, bundle = _load()
    return {
        "schema": "recommend_budget_design.v1",
        "specification": SPEC,
        "configuration": CONFIG,
        "contracts": evidence.CONTRACTS,
        "frozen_bundle": bundle,
    }


def run_comparison(design_path):
    identity = file_set_identity({"design": design_path}, root=ROOT)
    design = strict_json_object(design_path)
    inputs, bundle = _load()
    evidence._same(
        design,
        {
            "schema": "recommend_budget_design.v1",
            "specification": SPEC,
            "configuration": CONFIG,
            "contracts": evidence.CONTRACTS,
            "frozen_bundle": bundle,
        },
        "frozen budget design changed",
    )
    before = {**bundle, **identity}
    evidence._same(
        file_set_identity(_paths(before), root=ROOT),
        before,
        "inputs changed before evaluation",
    )
    started = datetime.now(timezone.utc).isoformat()
    segments = {}
    for name, source in inputs.items():
        frame = pd.DataFrame(source["predictions"])
        evaluation = evaluate_budget(frame, **SPEC["statistics"])
        original = {
            (r["date"], r["slot"]): r["selected_coins"]["A"]
            for r in source["evaluation"]["selection_audit"]
        }
        for row in evaluation["selection_audit"]:
            evidence._same(
                row["selected_coins"]["A"],
                original[(row["date"], row["slot"])],
                "original R1 selection changed",
            )
        segments[name] = {
            "coverage": evidence._coverage(frame, evidence.CONTRACTS[name], evaluation),
            "evaluation": evaluation,
            "original_A_parity": "passed",
        }
    result = {
        "schema": "recommend_budget_comparison.v1",
        "design_id": SPEC["design_id"],
        "status": "historical_comparison_completed",
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "deployable": False,
        "model_fitted": False,
        "live_model_changed": False,
        "model_artifact_saved": False,
        "automatic_adoption": False,
        "promotion_status": "NOT_EVALUATED",
        "is_untouched_holdout": False,
        "design": design,
        "generator_files": before,
        "segments": segments,
        "interpretation": [
            "One new selector on already-observed periods; not untouched holdout or prospective performance.",
            "Original R1 scores define constraints, not guarantees of realized downside/upside.",
            "No new labels, model fitting, automatic training, deployment or orders.",
            "Full candidate set including unlabeled candidates is selected before outcome inspection.",
            "Net hypothetical pick outcomes include existing costs once, not portfolio performance.",
        ],
    }
    result["report_payload_sha256"] = sha256_bytes(canonical_json_bytes(result))
    evidence._same(
        file_set_identity(_paths(before), root=ROOT),
        before,
        "inputs changed during execution or serialization",
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-design", action="store_true")
    mode.add_argument("--design", type=Path)
    parser.add_argument(
        "--output", type=Path, help="NEW project-local JSON; never overwrite"
    )
    args = parser.parse_args(argv)
    try:
        if args.prepare_design and args.output is None:
            raise ValueError("design preparation requires a new --output")
        report = (
            prepare_design() if args.prepare_design else run_comparison(args.design)
        )
        if args.output is not None:
            identities = report.get("generator_files", report.get("frozen_bundle", {}))
            write_new_report(
                args.output,
                report,
                root=ROOT,
                protected_roots=tuple(_paths(identities).values()),
            )
        summary = {
            key: report[key]
            for key in ("schema", "status", "deployable")
            if key in report
        }
        if "segments" in report:
            summary["segments"] = {
                name: {
                    "coverage": segment["coverage"],
                    "slots": {
                        slot: {
                            "dates": value["dates"],
                            "changed_dates": value.get("changed_dates"),
                            "absolute": {
                                arm: data["mean"]
                                for arm, data in value.get("absolute", {}).items()
                            },
                            "delta_vs_A": value.get("delta_vs_A", {}).get("mean"),
                        }
                        for slot, value in segment["evaluation"]["slots"].items()
                    },
                }
                for name, segment in report["segments"].items()
            }
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(
            json.dumps(
                {"status": "blocked", "reason": str(exc), "deployable": False},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

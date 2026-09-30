"""Manual fixed L1 Top3 replay from the already verified endpoint artifact."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PRELUDE_FORBID_TELEGRAM"] = "1"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ops.artifact_provenance import canonical_json_bytes, file_identity, sha256_bytes, strict_json_object  # noqa: E402
from scripts import review_recommend_book_pressure as endpoint  # noqa: E402
from scripts.build_recommend_training_data import write_new_report  # noqa: E402
from signals import recommend_book_top3 as selector  # noqa: E402

SOURCES = ("scripts/compare_recommend_book_top3.py", "signals/recommend_book_top3.py", "ledger/portfolio_metrics.py")


def _source(path):
    before = file_identity(path, root=ROOT)
    report = strict_json_object(path)
    inputs = endpoint.inputs
    inputs.native._require(report["schema"] == "recommend_book_pressure_review.v1"
                           and report["deployable"] is report["live_model_changed"] is report["new_trial_recorded"] is False,
                           "invalid endpoint scope")
    payload = {k: v for k, v in report.items() if k != "report_payload_sha256"}
    inputs.native._require(inputs.native._time(report["generated_at"]) <= datetime.now(timezone.utc), "future endpoint report")
    inputs._same(report["report_payload_sha256"], sha256_bytes(canonical_json_bytes(payload)), "endpoint seal mismatch")
    inputs._same(report["design"], endpoint.prepare_design(ROOT / report["design"]["source_design"]), "endpoint lineage changed")
    inputs._same(before, file_identity(path, root=ROOT), "endpoint changed while reading")
    return report, before


def prepare_design(path):
    report, identity = _source(path)
    items = endpoint.inputs.review.forward._identities([*report["design"]["input_files"], identity,
              *[file_identity(ROOT / p, root=ROOT) for p in SOURCES]])
    endpoint.inputs._verify(items)
    return {"schema": "recommend_book_top3_design.v1", "configuration": selector.CONFIG,
            "endpoint_report": str(Path(path).resolve().relative_to(ROOT)), "input_files": items,
            "statistics": {"n_boot": 1000, "seed": 42}}


def run_comparison(design_path):
    inputs = endpoint.inputs
    identity = file_identity(design_path, root=ROOT)
    design = strict_json_object(design_path)
    inputs._same(design, prepare_design(ROOT / design["endpoint_report"]), "frozen selector design changed")
    endpoint_report, _ = _source(ROOT / design["endpoint_report"])
    original_design = strict_json_object(ROOT / endpoint_report["design"]["source_design"])
    days, used = inputs.load_inputs(ROOT / original_design["input_path"])
    inputs._same(used, original_design["input_files"], "native input cohort changed")
    source = strict_json_object(ROOT / original_design["input_path"])["evaluations"]["shortlist"]
    audits = {r["date"]: r for r in source["dates"]}
    quotes = {r["date"]: r["quotes"] for r in endpoint_report["endpoints"]}
    inputs.native._require(len(quotes) == len(endpoint_report["endpoints"]), "duplicate quote date")
    for day in days:
        if "rows" not in day:
            continue
        q = quotes[day["date"]]
        inputs.native._require(set(q) == {r["coin"] for r in day["rows"]}, "endpoint universe mismatch")
        for row in day["rows"]:
            point = q[row["coin"]]
            inputs._same({k: row[k] for k in ("spread_fraction", "book_event_age_seconds")},
                         {k: point[k] for k in ("spread_fraction", "book_event_age_seconds")}, "quote parity mismatch")
            row.update(point)
        label = strict_json_object(inputs.native._path(audits[day["date"]]["evidence_manifest"]["files"]["label"]["path"]))
        inputs.native._require(label["round_trip_cost_fraction"] == .0015 and label["return_unit"] == "fraction", "cost/unit changed")
        day.update(entry_at=label["path_window_start"], end_at=label["path_window_end"])
    result = selector.evaluate(days, **design["statistics"])
    originals = {r["date"]: r for r in source["primary_paired"]["daily"]}
    for row in result["daily"]:
        inputs._same(row["selection"]["control"], audits[row["date"]]["control_top3"], "original Top3 parity failed")
        inputs._same(row["metrics"]["control"], originals[row["date"]]["control"], "original metric parity failed")
    output = {"schema": "recommend_book_top3_comparison.v1", "design": design,
              "generated_at": datetime.now(timezone.utc).isoformat(), "deployable": False,
              "new_trial_recorded": False, "live_model_changed": False,
              "endpoint_source": "frozen_previously_raw_verified_artifact_no_raw_reparse_this_run",
              "original_R1_parity": "passed", "analysis": result}
    inputs._verify([*design["input_files"], identity])
    output["report_payload_sha256"] = sha256_bytes(canonical_json_bytes(output))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-design", type=Path, metavar="ENDPOINT_REPORT")
    mode.add_argument("--design", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.prepare_design and not args.output:
            raise ValueError("design requires new output")
        result = prepare_design(args.prepare_design) if args.prepare_design else run_comparison(args.design)
        if args.output:
            design = result.get("design", result)
            protected = tuple(endpoint.inputs.native._path(i["path"]) for i in design["input_files"])
            write_new_report(args.output, result, root=ROOT,
                             protected_roots=protected + ((args.design,) if args.design else ()))
        print(json.dumps({"schema": result["schema"], "deployable": False,
                          "segments": result.get("analysis", {}).get("segments")}, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

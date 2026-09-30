"""Manual outcome-blind L1 endpoint audit; new immutable files only, no live writes."""
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

from data.upbit_microstructure import iter_raw_records  # noqa: E402
from ops.artifact_provenance import canonical_json_bytes, file_identity, sha256_bytes, strict_json_object  # noqa: E402
from scripts import review_recommend_spread as inputs  # noqa: E402
from scripts.build_recommend_training_data import write_new_report  # noqa: E402
from signals import recommend_book_pressure as book  # noqa: E402
from signals import recommend_microstructure as raw  # noqa: E402

SOURCES = ("signals/recommend_book_pressure.py", "scripts/review_recommend_book_pressure.py",
           "signals/recommend_spread_diagnostics.py", "scripts/review_recommend_spread.py")


def prepare_design(source_design):
    """Reuse the frozen input lineage, without reading/aggregating outcomes."""
    source_identity = file_identity(source_design, root=ROOT)
    source = strict_json_object(source_design)
    inputs.native._require(source["schema"] == "recommend_spread_design.v1", "wrong source design")
    inputs._same(source["configuration"], inputs.CONFIG, "source configuration changed")
    bound = inputs.review.forward._identities([*source["input_files"], source_identity,
        *[file_identity(ROOT / p, root=ROOT) for p in SOURCES]])
    inputs._verify(bound)
    return {"schema": "recommend_book_pressure_design.v1", "configuration": book.CONFIG,
            "source_design": str(Path(source_design).resolve().relative_to(ROOT)), "input_files": bound,
            "statistics": {"n_boot": 1000, "seed": 42}}


def run_review(design_path):
    identity = file_identity(design_path, root=ROOT)
    design = strict_json_object(design_path)
    inputs.native._require(design["schema"] == "recommend_book_pressure_design.v1", "wrong design")
    inputs._same(design, prepare_design(ROOT / design["source_design"]), "frozen design changed")
    source = strict_json_object(ROOT / design["source_design"])
    days, used = inputs.load_inputs(ROOT / source["input_path"])
    inputs._same(used, source["input_files"], "base input lineage changed")
    bound = {str(inputs.native._path(i["path"]).resolve()): i for i in design["input_files"]}

    def checked(path):
        path = inputs.native._path(path)
        inputs._same(file_identity(path, root=ROOT), bound[str(path.resolve())], "unbound/changed input")
        return path

    extracted = []
    for day in days:
        if "rows" not in day:
            continue
        score = strict_json_object(checked(inputs.trial.DEFAULT_TRIAL_ROOT / day["date"] / inputs.trial.TRIAL_ID / "score.json"))
        snapshot, feature = [strict_json_object(checked(i["path"])) for i in score["source_inputs"]]
        manifest = strict_json_object(checked(feature["provenance"]["files"][1]["path"]))
        coins = {r["coin"] for r in snapshot["universe"]}
        stream = manifest["streams"]["orderbook"]
        inputs.native._require(manifest["subscription"]["orderbook_depth"]["requested"] == 1, "not depth1")
        records = raw._validated_records(iter_raw_records(checked(stream["artifact"]["path"])),
                                          "orderbook", manifest, set(manifest["universe"]["markets"]))
        quotes = book.endpoint_quotes(records, cutoff=raw._ns(snapshot["decision_started_at"]), coins=coins)
        for row in day["rows"]:
            q = quotes.get(row["coin"])
            if q is None:
                row.update(pressure=None, l1_notional=None)
                continue
            inputs._same({k: q[k] for k in ("spread_fraction", "book_event_age_seconds")},
                         {k: row[k] for k in ("spread_fraction", "book_event_age_seconds")}, "endpoint parity failed")
            row.update(q)
        extracted.append({"date": day["date"], "quotes": quotes})
        print(json.dumps({"parsed_date": day["date"], "quotes": len(quotes)}), file=sys.stderr, flush=True)
    result = {"schema": "recommend_book_pressure_review.v1", "design": design,
              "generated_at": datetime.now(timezone.utc).isoformat(), "deployable": False,
              "live_model_changed": False, "new_trial_recorded": False,
              "endpoints": extracted, "analysis": book.analyze(days, **design["statistics"])}
    inputs._verify([*design["input_files"], identity])
    result["report_payload_sha256"] = sha256_bytes(canonical_json_bytes(result))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-design", type=Path, metavar="FROZEN_SOURCE_DESIGN")
    mode.add_argument("--design", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.prepare_design and not args.output:
            raise ValueError("design requires new output")
        result = prepare_design(args.prepare_design) if args.prepare_design else run_review(args.design)
        if args.output:
            manifest = result.get("design", result)
            protected = tuple(inputs.native._path(i["path"]) for i in manifest["input_files"])
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

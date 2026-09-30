"""Manual, read-only spread-information audit; never connects to live ranking.

First freeze --prepare-design, then run --design. Optional output is a NEW
project-local JSON only. Existing review/native/raw/labels are never replaced.
"""

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

from ops import recommend_trial_review as review  # noqa: E402
from ops.artifact_provenance import (  # noqa: E402
    canonical_json_bytes,
    file_identity,
    sha256_bytes,
    strict_json_object,
)
from ops.recommendation_evidence import load_recommendation_evidence  # noqa: E402
from scripts.build_recommend_training_data import write_new_report  # noqa: E402
from signals import recommend_microstructure_trial as native  # noqa: E402
from signals import recommend_trade_shortlist_trial as trial  # noqa: E402
from signals import recommend_trade_shortlist_eval as evaluator  # noqa: E402
from signals.recommend_spread_diagnostics import CONFIG, analyze  # noqa: E402

SOURCES = (
    "scripts/review_recommend_spread.py",
    "signals/recommend_spread_diagnostics.py",
    "signals/recommend_experiment_eval.py",
    "scripts/build_recommend_training_data.py",
)


def _same(left, right, reason):
    native._require(canonical_json_bytes(left) == canonical_json_bytes(right), reason)


def _verify(items):
    # Includes compressed raw evidence, not just the cached review summary.
    review.forward._verify(items)


def load_inputs(path):
    before = file_identity(path, root=ROOT)
    document = strict_json_object(path)
    review._check_document(document, review.SCHEMA)
    _same(document["generator_sources"], review._sources(), "review sources changed")
    native._require(
        document["through_date"] >= CONFIG["end_date"], "review history too short"
    )
    evaluation = document["evaluations"]["shortlist"]
    review._check_document(evaluation, evaluator.EVALUATION_SCHEMA)
    native._require(
        evaluation["generated_at"] == document["generated_at"]
        and native._time(document["generated_at"]) <= datetime.now(timezone.utc),
        "mixed or future review clock",
    )
    _same(evaluation["config"], evaluator.TRIAL_CONFIG, "native configuration changed")
    _same(
        evaluation["generator_sources"],
        evaluator._generator_sources(),
        "native sources changed",
    )
    native._require(
        evaluation["inputs_unchanged"] is True
        and evaluation["deployable"] is False
        and evaluation["automatic_promotion"] is False,
        "invalid evaluation scope",
    )
    items = [
        before,
        *document["generator_sources"],
        *evaluation["generator_sources"],
        *evaluation["trial_inputs"],
        *evaluation["evidence_inputs"],
        *[file_identity(ROOT / source, root=ROOT) for source in SOURCES],
    ]
    items = review.forward._identities(items)
    _verify(items)
    bound = {str(native._path(item["path"]).resolve()): item for item in items}

    def checked_file(value):
        target = native._path(value)
        identity = bound.get(str(target.resolve()))
        native._require(identity is not None and identity["exists"], "unbound input")
        _same(file_identity(target, root=ROOT), identity, "bound input changed")
        return target

    days = []
    for audit in evaluation["dates"]:
        if not CONFIG["start_date"] <= audit["date"] <= CONFIG["end_date"]:
            continue
        day = {"date": audit["date"]}
        days.append(day)
        if audit["status"] != "prospective_comparable":
            day["unavailable_reason"] = audit["status"] + ":" + str(audit.get("reason"))
            continue
        score_path = checked_file(
            trial.DEFAULT_TRIAL_ROOT / audit["date"] / trial.TRIAL_ID / "score.json"
        )
        score = strict_json_object(score_path)
        trial._checked(score, trial.SCORE_SCHEMA)
        native._require(
            score["asof"] == audit["date"]
            and score["snapshot_id"] == audit["snapshot_id"],
            "native identity changed",
        )
        snapshot_path, feature_path = [
            checked_file(item["path"]) for item in score["source_inputs"]
        ]
        # Existing parser checks capture identity, source hashes, raw checksums,
        # both event/receive cutoffs and universe. No raw reparsing is claimed.
        snapshot, feature, _plan, sources = native._load_inputs(
            snapshot_path, feature_path
        )
        _same(sources, score["source_inputs"], "native feature generation changed")
        _same(
            trial.plan_trade_shortlist(snapshot, feature),
            score["plan"],
            "native plan mismatch",
        )
        evidence = load_recommendation_evidence(
            snapshot_path,
            label_root=ROOT / "output/recommend_score_labels",
            receipt_root=ROOT / "output/recommend_receipts",
            now=native._time(document["generated_at"]),
        )
        _same(
            evidence["manifest"],
            audit["evidence_manifest"],
            "label or delivery evidence changed",
        )
        labels = {r["coin"]: r for r in evidence["label"]["rows"]}
        features = {r["coin"]: r for r in feature["rows"]}
        native._require(
            set(labels) == set(features) == {r["coin"] for r in snapshot["universe"]},
            "candidate universe mismatch",
        )
        native._require(
            feature["feature_evidence_valid"] is True, "unavailable feature evidence"
        )
        day["rows"] = []
        for candidate in snapshot["universe"]:
            coin = candidate["coin"]
            micro, label = features[coin], labels[coin]
            native._require(micro["rank"] == candidate["rank"], "rank mismatch")
            day["rows"].append(
                {
                    "coin": coin,
                    "rank": candidate["rank"],
                    "f_log_qv": candidate["feature_values"].get("f_log_qv"),
                    "f_atr_pct_14": candidate["feature_values"].get("f_atr_pct_14"),
                    **micro["quality_diagnostics"],
                    **{
                        name: label[name]
                        for name in (
                            "label_status",
                            "up10",
                            "dn5",
                            "mfe",
                            "mae",
                            "eod_return_net",
                            "tp5_sl3_return_net",
                        )
                    },
                }
            )
    _verify(items)
    return days, items


def prepare_design(path):
    _days, inputs = load_inputs(path)
    return {
        "schema": "recommend_spread_design.v1",
        "configuration": CONFIG,
        "input_path": str(Path(path).resolve().relative_to(ROOT.resolve())),
        "input_files": inputs,
        "statistics": {"n_boot": 1000, "seed": 42},
    }


def run_review(design_path):
    design_identity = file_identity(design_path, root=ROOT)
    design = strict_json_object(design_path)
    native._require(
        design["schema"] == "recommend_spread_design.v1", "design schema mismatch"
    )
    _same(design["configuration"], CONFIG, "frozen configuration changed")
    _same(design["statistics"], {"n_boot": 1000, "seed": 42}, "statistics changed")
    # Verify frozen bytes before loading/reading any outcomes for evaluation.
    _verify(design["input_files"])
    days, inputs = load_inputs(ROOT / design["input_path"])
    _same(inputs, design["input_files"], "frozen inputs changed")
    result = {
        "schema": "recommend_spread_review.v1",
        "design": design,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "deployable": False,
        "live_model_changed": False,
        "new_trial_recorded": False,
        "analysis": analyze(days, **design["statistics"]),
    }
    _verify([*inputs, design_identity])
    result["report_payload_sha256"] = sha256_bytes(canonical_json_bytes(result))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-design", action="store_true")
    mode.add_argument("--design", type=Path)
    parser.add_argument(
        "--input", type=Path, default=ROOT / "output/recommend_trial_review.json"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.prepare_design and args.output is None:
            raise ValueError("design preparation requires new --output")
        report = (
            prepare_design(args.input)
            if args.prepare_design
            else run_review(args.design)
        )
        if args.output is not None:
            inputs = report.get(
                "input_files", report.get("design", {}).get("input_files", [])
            )
            protected = tuple(native._path(item["path"]) for item in inputs)
            write_new_report(
                args.output,
                report,
                root=ROOT,
                protected_roots=protected + ((args.design,) if args.design else ()),
            )
        print(
            json.dumps(
                {
                    "schema": report["schema"],
                    "deployable": False,
                    "segments": report.get("analysis", {}).get("segments"),
                    "excluded": report.get("analysis", {}).get("excluded"),
                    "output": str(args.output) if args.output else None,
                },
                ensure_ascii=False,
                allow_nan=False,
            )
        )
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(
            json.dumps({"status": "blocked", "reason": str(exc)}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

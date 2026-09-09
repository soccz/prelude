"""One offline matched-sample calibration comparison from pinned local evidence.

The final head, target dates, features and expected downside are held fixed.
Past-only block scores are used for probability calibration, not as a claim of
new prospective evidence. No DB, network, delivery, schedule or deployment work.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
os.environ["PRELUDE_FORBID_TELEGRAM"] = "1"
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from ops.artifact_provenance import (  # noqa: E402
    canonical_json_bytes, file_set_identity, sha256_bytes, sha256_file, strict_json_object,
)
from scripts.build_recommend_training_data import write_new_report  # noqa: E402
from scripts.compare_recommend_horizon import (  # noqa: E402
    _check_manifest, _check_seal, _encode_frame, _same, _seal, load_bundle,
    reviewed_design as reference_design,
)
from signals.recommend_calibration_eval import EVALUATION_CONFIG, evaluate_calibration  # noqa: E402
from signals.recommend_calibration_experiment import EXPERIMENT_CONFIG, run_calibration_experiment  # noqa: E402
from signals.recommend_calibration_model import KERNEL_CONFIG  # noqa: E402

BUNDLE_SHA256 = "473f21b7efc0d196f07a060bf1192ea8c09078e4071159c44a07cf5429389225"
REFERENCE_SHA256 = "f271f3ffcb633334185b6ced62f3f05f761f834fccfe594d430cebb83358b654"
DESIGN_ID = "r1_preopen_matched_calibration_20260907_v1"
SOURCES = (
    "scripts/compare_recommend_calibration.py", "signals/recommend_calibration_model.py",
    "signals/recommend_calibration_experiment.py", "signals/recommend_calibration_eval.py",
)
PACKAGES = ("numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "xgboost")


def reviewed_design() -> dict:
    return {
        "schema": "recommend_calibration_design.v1", "design_id": DESIGN_ID,
        "scope": "offline_comparison_only", "bundle_sha256": BUNDLE_SHA256,
        "reference_sha256": REFERENCE_SHA256,
        "approval": "user requested design through definite conclusion for the proposed offline calibration comparison; no deployment",
        "kernel_config": KERNEL_CONFIG, "experiment_config": EXPERIMENT_CONFIG,
        "evaluation_config": EVALUATION_CONFIG, "statistics": {"n_boot": 4000, "seed": 42},
        "settings_in_this_run": 1, "maximum_fits": 335,
        "primary_contrast": "O_minus_R", "sample_window_contrast": "R_minus_F",
        "practical_total_contrast": "O_minus_F", "actual_A_role": "practical_reference_not_pure_control",
        "knowledge_time_basis": "calendar_maturity_only; historical_ingestion_unknown",
        "target_contract": "unchanged original B same-row D1; canonical preopen outcome window differs",
        "is_untouched_holdout": False, "prior_reuse_of_dates": True,
        "interpretation": "OOT score protocol includes model-age/training-volume/distribution effects; not a pure overfit-removal causal effect",
        "stopping_rule": "one fixed comparison; reject, offline-promising or insufficient; no outcome-driven sweep or promotion",
    }


def _records(frame: pd.DataFrame) -> list[dict]:
    return frame.astype(object).where(frame.notna(), None).to_dict("records")


def _capture(paths: dict) -> dict:
    manifest = {"root": str(ROOT), "files": file_set_identity(paths, root=ROOT)}
    if not all(item["exists"] for item in manifest["files"].values()):
        raise ValueError("missing experiment input or source")
    return manifest


def load_inputs(bundle_path: Path, reference_path: Path) -> tuple:
    _same(sha256_file(reference_path), REFERENCE_SHA256, "reference whole-file checksum mismatch")
    reference = strict_json_object(reference_path)
    _check_seal(reference, "report_payload_sha256")
    _same(reference["design"], reference_design(), "old reference design changed")
    _check_manifest(reference["generator_manifest"])
    _check_manifest(reference["data_provenance"])
    bundle, training, prediction = load_bundle(bundle_path, BUNDLE_SHA256, reference["design"])
    _same(reference["bundle_file_sha256"], BUNDLE_SHA256, "reference bound to a different bundle")
    _same(reference["bundle_payload_sha256"], bundle["bundle_payload_sha256"], "reference bundle payload mismatch")
    _same(reference["versions"], {name: importlib.metadata.version(name) for name in PACKAGES},
          "installed packages differ from the fixed reference")
    # Full-precision original outcomes and predictor values must remain identical.
    original = [{key: row[key] for key in prediction.columns} for row in reference["predictions"]]
    _same(original, _records(prediction), "reference original candidate rows differ from bundle")
    _same(sha256_file(reference_path), REFERENCE_SHA256, "reference changed during deserialization")
    for manifest in (reference["generator_manifest"], reference["data_provenance"],
                     bundle["generator_manifest"], bundle["data_provenance"]):
        _check_manifest(manifest)
    return bundle, training, prediction, reference


def run_experiment(bundle_path: Path, reference_path: Path, design_path: Path, *, progress=None) -> dict:
    manifest = _capture({"bundle": bundle_path, "reference": reference_path, "design": design_path,
                         **{name: ROOT / name for name in SOURCES}})
    design = strict_json_object(design_path)
    _same(design, reviewed_design(), "design differs from the reviewed fixed configuration")
    bundle, training, prediction, reference = load_inputs(bundle_path, reference_path)
    metadata = bundle["decision_metadata"]
    metadata_before = canonical_json_bytes(metadata)
    reference_before = sha256_bytes(canonical_json_bytes(reference))
    _check_manifest(manifest)
    for item in (bundle, reference):
        _check_manifest(item["generator_manifest"])
        _check_manifest(item["data_provenance"])
    started = datetime.now(timezone.utc).isoformat()
    experiment = run_calibration_experiment(training, prediction, metadata, reference, progress=progress)
    _same(experiment["configuration"], design["experiment_config"], "runtime experiment configuration changed")
    if canonical_json_bytes(metadata) != metadata_before:
        raise ValueError("decision metadata mutated during fitting")
    _same(sha256_bytes(canonical_json_bytes(reference)), reference_before, "reference mutated during fitting")
    for name, frame in (("training", training), ("prediction", prediction)):
        _same(_encode_frame(frame)["sha256"], bundle[name]["sha256"], f"{name} frame mutated during fitting")
    fitted = experiment["predictions"]
    _same(_records(fitted.loc[:, prediction.columns]), _records(prediction), "fitted original candidates mutated")
    fitted_before = fitted.copy(deep=True)
    evaluation = evaluate_calibration(fitted, **design["statistics"])
    pd.testing.assert_frame_equal(fitted, fitted_before, check_exact=True)
    _same(evaluation["configuration"], design["evaluation_config"], "runtime evaluation configuration changed")
    fit_count = experiment["fit_count"]
    if type(fit_count) is not int or not 0 <= fit_count <= design["maximum_fits"]:
        raise ValueError("invalid or excessive fit count")
    all_audits = [*experiment["inner_audit"], *experiment["outer_audit"]]
    if sum(audit["fit_count"] for audit in all_audits) != fit_count:
        raise ValueError("fit count does not match stage audits")
    report = _seal({
        "schema": "recommend_calibration_comparison.v1", "design_id": DESIGN_ID,
        "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
        "status": "controlled_retrospective_completed", "scope": "offline_comparison_only",
        "deployable": False, "promotion_status": "NOT_EVALUATED", "live_model_changed": False,
        "model_artifact_saved": False, "is_untouched_holdout": False,
        "knowledge_time_basis": design["knowledge_time_basis"], "fit_count": fit_count,
        "design": design, "generator_manifest": manifest,
        "bundle_file_sha256": BUNDLE_SHA256, "reference_file_sha256": REFERENCE_SHA256,
        "bundle_payload_sha256": bundle["bundle_payload_sha256"],
        "reference_payload_sha256": reference["report_payload_sha256"],
        "data_provenance": bundle["data_provenance"], "data_audit": bundle["data_audit"],
        "versions": {name: importlib.metadata.version(name) for name in PACKAGES},
        "inner_audit": experiment["inner_audit"], "outer_audit": experiment["outer_audit"],
        "oof_cache": _encode_frame(experiment["oof_cache"]),
        "resub_cache": _encode_frame(experiment["resub_cache"]),
        "evaluation": evaluation, "predictions": _records(fitted),
        "interpretation": [
            "F reproduces prior B; R restricts calibration rows; O uses identical rows with past-only block scores.",
            "The final raw model and expected downside are shared, not newly optimized per calibration arm.",
            "Seven-calendar-day inner frozen models are not daily inner refits; final outer heads refit daily.",
            "Targets retain original B D1 timing; canonical preopen path/entry and EPS differences are not repaired here.",
            "Repeatedly observed development dates and unknown historical ingestion prevent a fresh forward claim.",
            "Probability calibration scores alone do not establish better low-downside/high-upside Top3 picks.",
            "Original pick-level net outcomes already include costs; not user trades or a compounded portfolio.",
        ],
    }, "report_payload_sha256")
    for item in (bundle, reference):
        _check_manifest(item["generator_manifest"])
        _check_manifest(item["data_provenance"])
    _check_manifest(manifest)
    _same(design, reviewed_design(), "runtime configuration changed during execution or serialization")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="NEW project-local JSON; never overwrite")
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError("output already exists; never overwrite")
        report = run_experiment(args.bundle, args.reference, args.design,
                                progress=lambda item: print(json.dumps(item), file=sys.stderr, flush=True))
        write_new_report(args.output, report, root=ROOT,
                         protected_roots=(args.bundle, args.reference, args.design))
        print(json.dumps({"status": report["status"], "fit_count": report["fit_count"],
                          "output": str(args.output), "sha256": sha256_file(args.output),
                          "primary_dates": report["evaluation"]["primary"]["dates"],
                          "supplemental_dates": report["evaluation"]["supplemental"]["dates"],
                          "deployable": False}, ensure_ascii=False))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, AssertionError) as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc), "deployable": False}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

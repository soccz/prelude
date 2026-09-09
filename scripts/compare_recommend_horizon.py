"""Prepare immutable inputs, then run one offline daily target-calendar comparison.

No live DB access during fitting, no saved model, no schedule or delivery action.
Preparation and fitting are separate commands; the run pins the published bundle
by a caller-supplied whole-file SHA256, not merely its internal checksum.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import importlib.metadata
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import sys
import time

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
from signals.recommend_horizon_data import DATA_CONFIG, prepare_horizon_data  # noqa: E402
from signals.recommend_horizon_eval import EVALUATION_CONFIG, evaluate_horizon  # noqa: E402
from signals.recommend_horizon_model import MODEL_CONFIG, fit_daily_pair  # noqa: E402
from signals.recommend_training_data import ALLOWED_FEATURES  # noqa: E402

DESIGN_ID = "r1_preopen_target_calendar_20260907_v1"
SOURCES = (
    "scripts/compare_recommend_horizon.py", "signals/recommend_horizon_data.py",
    "signals/recommend_horizon_model.py", "signals/recommend_horizon_eval.py",
    "signals/recommend_experiment_eval.py", "scripts/build_recommend_training_data.py",
)


def reviewed_design() -> dict:
    return {
        "schema": "recommend_horizon_design.v1", "design_id": DESIGN_ID,
        "scope": "offline_comparison_only",
        "approval": "user approved offline target-calendar alignment and comparative retraining; no deployment",
        "input_sha256": "7a420ef479f90eda8879301bb68b51e38dfb1c7afe7b7cdf450e07f956e3b2fa",
        "data_config": DATA_CONFIG, "model_config": MODEL_CONFIG,
        "evaluation_config": EVALUATION_CONFIG, "statistics": {"n_boot": 4000, "seed": 42},
        "daily_decisions": 35, "maximum_fits": 350, "settings_in_this_run": 1,
        "primary_contrast": "C_minus_B", "actual_A_role": "practical_reference_not_isolated_control",
        "knowledge_time_basis": "calendar_maturity_only; historical_ingestion_unknown",
        "is_untouched_holdout": False, "prior_reuse_of_dates": True,
        "stopping_rule": "one fixed comparison; no outcome-driven sweep, fallback, or live promotion",
    }


def _same(actual, expected, reason: str) -> None:
    if canonical_json_bytes(actual) != canonical_json_bytes(expected):
        raise ValueError(reason)


def _seal(report: dict, key: str) -> dict:
    return {**report, key: sha256_bytes(canonical_json_bytes(report))}


def _check_seal(report: dict, key: str) -> None:
    _same(report.get(key), sha256_bytes(canonical_json_bytes({k: v for k, v in report.items() if k != key})),
          f"{key} mismatch")


def _manifest(paths: dict) -> dict:
    files = file_set_identity(paths, root=ROOT)
    if not all(item["exists"] for item in files.values()):
        raise ValueError("missing source or input")
    return {"root": str(ROOT), "files": files}


def _check_manifest(manifest: dict) -> None:
    root = Path(manifest["root"])
    paths = {key: Path(item["path"]) if Path(item["path"]).is_absolute() else root / item["path"]
             for key, item in manifest["files"].items()}
    _same(file_set_identity(paths, root=root), manifest["files"], "source/evidence/input changed")


def _design(path: Path) -> dict:
    design = strict_json_object(path)
    _same(design, reviewed_design(), "design differs from fixed reviewed configuration")
    return design


def _encode_frame(frame: pd.DataFrame) -> dict:
    stream = BytesIO()
    frame.to_parquet(stream, index=False, engine="pyarrow", compression="zstd")
    raw = stream.getvalue()
    encoded = {"encoding": "parquet_zstd_base64", "sha256": sha256_bytes(raw),
               "rows": len(frame), "columns": list(frame.columns),
               "column_index_dtype": str(frame.columns.dtype), "column_index_name": frame.columns.name,
               "dtypes": [str(dtype) for dtype in frame.dtypes],
               "data": base64.b64encode(raw).decode("ascii")}
    pd.testing.assert_frame_equal(frame.reset_index(drop=True), _decode_frame(encoded), check_exact=True)
    return encoded


def _decode_frame(encoded: dict) -> pd.DataFrame:
    if encoded["encoding"] != "parquet_zstd_base64":
        raise ValueError("unknown frame encoding")
    raw = base64.b64decode(encoded["data"], validate=True)
    _same(sha256_bytes(raw), encoded["sha256"], "encoded frame checksum mismatch")
    # Pandas 3 otherwise silently promotes object strings (including nullable
    # target-end timestamps) to StringDtype and changes None to NaN. Disable
    # inference, then restore only the explicitly recorded StringDtype columns.
    with pd.option_context("future.infer_string", False):
        frame = pd.read_parquet(BytesIO(raw), engine="pyarrow")
    _same(list(frame.columns), encoded["columns"], "encoded columns mismatch")
    frame.columns = pd.Index(encoded["columns"], dtype=encoded["column_index_dtype"],
                             name=encoded["column_index_name"])
    for name, dtype in zip(encoded["columns"], encoded["dtypes"], strict=True):
        if dtype == "str" and name in frame and frame[name].dtype == object:
            frame[name] = frame[name].astype("str")
    _same([len(frame), list(frame.columns), [str(dtype) for dtype in frame.dtypes]],
          [encoded["rows"], encoded["columns"], encoded["dtypes"]], "encoded frame schema mismatch")
    return frame


def prepare_bundle(input_path: Path, audit_path: Path, design_path: Path) -> dict:
    manifest = _manifest({"readiness": input_path, "audit": audit_path, "design": design_path,
                          **{name: ROOT / name for name in SOURCES}})
    design = _design(design_path)
    _same(sha256_file(input_path), design["input_sha256"], "readiness checksum mismatch")
    dataset = prepare_horizon_data(input_path, audit_path)
    bundle = _seal({
        "schema": "recommend_horizon_bundle.v1", "design": design,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generator_manifest": manifest, "data_provenance": dataset["provenance"],
        "training": _encode_frame(dataset["training_frame"]),
        "prediction": _encode_frame(dataset["prediction_frame"]),
        "decision_metadata": dataset["decision_metadata"], "data_audit": dataset["data_audit"],
    }, "bundle_payload_sha256")
    _check_manifest(manifest)
    _check_manifest(dataset["provenance"])
    return bundle


def load_bundle(path: Path, expected_sha256: str, design: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    _same(sha256_file(path), expected_sha256, "bundle whole-file checksum mismatch")
    bundle = strict_json_object(path)
    _check_seal(bundle, "bundle_payload_sha256")
    if bundle.get("schema") != "recommend_horizon_bundle.v1":
        raise ValueError("unknown bundle schema")
    _same(bundle["design"], design, "bundle/design mismatch")
    _check_manifest(bundle["generator_manifest"])
    _check_manifest(bundle["data_provenance"])
    training, prediction = _decode_frame(bundle["training"]), _decode_frame(bundle["prediction"])
    _same(sha256_file(path), expected_sha256, "bundle changed during deserialization")
    return bundle, training, prediction


def run_experiment(bundle_path: Path, expected_sha256: str, design_path: Path, *, progress=None) -> dict:
    manifest = _manifest({"bundle": bundle_path, "design": design_path,
                          **{name: ROOT / name for name in SOURCES}})
    design = _design(design_path)
    bundle, training, prediction = load_bundle(bundle_path, expected_sha256, design)
    metadata = bundle["decision_metadata"]
    if (prediction.empty or not prediction.slot.eq("preopen").all()
            or prediction.snapshot_id.nunique() != design["daily_decisions"]
            or set(prediction.snapshot_id) != set(metadata)):
        raise ValueError("frozen decision population mismatch")
    _check_manifest(manifest)
    started = datetime.now(timezone.utc).isoformat()
    metadata_before = canonical_json_bytes(metadata)
    _check_manifest(bundle["generator_manifest"])
    _check_manifest(bundle["data_provenance"])
    pieces, audits = [], []
    for i, (sid, group) in enumerate(prediction.groupby("snapshot_id", sort=False), 1):
        decision = metadata[sid]
        _same(sorted(group.date.unique().tolist()), [decision["date"]], "decision date mismatch")
        tick = time.monotonic()
        if progress:
            progress({"event": "fit_started", "number": i, "date": decision["date"]})
        result = fit_daily_pair(training, group.loc[:, ALLOWED_FEATURES].copy(deep=True), decision=decision)
        audit, scores = result["audit"], result["scores"]
        _same(audit["configuration"], design["model_config"], "runtime model configuration changed")
        if not scores.index.equals(group.index) or set(scores.columns) & set(group.columns):
            raise ValueError("prediction score alignment/column collision")
        if audit["fit_count"] not in (0, 10):
            raise ValueError("partial or excessive daily fit count")
        pieces.append(pd.concat([group, scores], axis=1))
        audits.append({"snapshot_id": sid, **audit})
        if progress:
            progress({"event": "fit_finished", "number": i, "date": decision["date"],
                      "fit_count": audit["fit_count"], "seconds": round(time.monotonic() - tick, 2)})
    if canonical_json_bytes(metadata) != metadata_before:
        raise ValueError("decision metadata mutated during fitting")
    _same(_encode_frame(training)["sha256"], bundle["training"]["sha256"], "training frame mutated during fitting")
    _same(_encode_frame(prediction)["sha256"], bundle["prediction"]["sha256"], "prediction frame mutated during fitting")
    fitted = pd.concat(pieces).sort_values(["date", "rank"], kind="stable").reset_index(drop=True)
    fit_count = sum(item["fit_count"] for item in audits)
    if fit_count > design["maximum_fits"]:
        raise ValueError("exceeded fixed fit budget")
    fitted_before = fitted.copy(deep=True)
    evaluation = evaluate_horizon(fitted, **design["statistics"])
    pd.testing.assert_frame_equal(fitted, fitted_before, check_exact=True)
    _same(evaluation["configuration"], design["evaluation_config"], "runtime evaluation configuration changed")
    report = _seal({
        "schema": "recommend_horizon_comparison.v1", "design_id": DESIGN_ID,
        "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
        "status": "controlled_retrospective_completed", "scope": "offline_comparison_only",
        "deployable": False, "promotion_status": "NOT_EVALUATED", "live_model_changed": False,
        "model_artifact_saved": False, "is_untouched_holdout": False,
        "knowledge_time_basis": design["knowledge_time_basis"], "fit_count": fit_count,
        "design": design, "generator_manifest": manifest,
        "bundle_file_sha256": expected_sha256, "bundle_payload_sha256": bundle["bundle_payload_sha256"],
        "data_provenance": bundle["data_provenance"], "data_audit": bundle["data_audit"],
        "versions": {name: importlib.metadata.version(name)
                     for name in ("numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "xgboost")},
        "daily_fit_audit": audits, "evaluation": evaluation,
        "predictions": fitted.astype(object).where(fitted.notna(), None).to_dict("records"),
        "interpretation": [
            "C-minus-B changes target calendar and target-derived calibration; common B class weights are not optimized for C.",
            "Existing fitted-score bucket calibration is not OOF and retains its overfitting limitations.",
            "Historical ingestion state and historical full-precision training rows are not proven.",
            "A is a delivered practical reference, not a pure target-calendar control.",
            "Canonical 15m outcomes retain existing EPS/entry discrepancies with D1 training definitions.",
            "Returns already include original transaction costs; these are paper pick outcomes, not user portfolio PnL.",
            "Repeatedly observed dates are development evidence; no result authorizes deployment.",
        ],
    }, "report_payload_sha256")
    _check_manifest(bundle["generator_manifest"])
    _check_manifest(bundle["data_provenance"])
    _check_manifest(manifest)
    _same(design, reviewed_design(), "runtime configuration changed before serialization completed")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="read-only DB reconstruction; no model fitting")
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--audit", type=Path, required=True)
    run = commands.add_parser("run", help="fixed offline fits from published bundle; no DB access")
    run.add_argument("--bundle", type=Path, required=True)
    run.add_argument("--bundle-sha256", required=True)
    for command in (prepare, run):
        command.add_argument("--design", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True, help="NEW project-local JSON")
    args = parser.parse_args(argv)
    # The existing loader emits per-row record-only nesting warnings. Keep errors;
    # the old model's independent-head nesting limitation is unchanged, not repaired.
    logging.basicConfig(level=logging.ERROR)
    try:
        if args.output.exists():
            raise ValueError("output already exists; never overwrite")
        if args.command == "prepare":
            report = prepare_bundle(args.input, args.audit, args.design)
            protected = (args.input, args.audit, args.design)
        else:
            report = run_experiment(args.bundle, args.bundle_sha256, args.design,
                                    progress=lambda item: print(json.dumps(item), file=sys.stderr, flush=True))
            protected = (args.bundle, args.design)
        write_new_report(args.output, report, root=ROOT, protected_roots=protected)
        print(json.dumps({"status": "prepared_no_fits" if args.command == "prepare" else report["status"],
                          "output": str(args.output), "sha256": sha256_file(args.output),
                          "fit_count": report.get("fit_count", 0), "deployable": False}, ensure_ascii=False))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, AssertionError) as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc), "deployable": False}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

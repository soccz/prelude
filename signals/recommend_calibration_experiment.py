"""Pure F/R/O calibration orchestration over one immutable historical bundle.

F must exactly reproduce the preceding B. R and O use the same recent rows and
labels, but fitted versus genuinely past-model raw scores. Missing OOF evidence
never shrinks the pool or silently substitutes the reference arm.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib

import numpy as np
import pandas as pd

from ops.artifact_provenance import canonical_json_bytes
from scripts.downside_head_riskreward_v1 import _apply_calib, _oof_bucket_calib
from signals.recommend_calibration_model import KERNEL_CONFIG, fit_final_reference, fit_inner_block
from signals.recommend_horizon_model import MODEL_CONFIG
from signals.recommend_training_data import ALLOWED_FEATURES

EXPERIMENT_CONFIG = {
    "version": "recommend_calibration_experiment.v1", "arms": ["F", "R", "O"],
    "calibration_calendar_days": 180, "inner_block_calendar_days": 7,
    "inner_anchor": "2026-01-23", "maximum_inner_end_exclusive": "2026-08-31",
    "minimum_inner_training_dates": 365, "embargo_calendar_days": 5,
    "calibration_buckets": 10, "matched_min_rows": 150,
    "matched_min_positive": 12, "matched_min_negative": 12, "matched_min_dates": 5,
    "maximum_inner_blocks": 32, "maximum_outer_decisions": 35, "maximum_fits": 335,
    "calibration_interval": "[cutoff_exclusive-180d, cutoff_exclusive)",
    "pool": "all recent common final-training rows; identical ordered keys/B labels for R/O; never intersect away missing OOF",
    "unavailable_policy": "both R/O all scores/raw/expected_downside NaN; F never substituted",
    "raw_test_policy": "fitted F/R/O use identical final raw probabilities",
    "expected_downside_policy": "fitted F/R/O copy F expected downside",
    "fit_order": "all inner blocks before chronological outer reference fits",
}
HEADS = tuple(MODEL_CONFIG["thresholds"])
RAW_COLUMNS = tuple(f"{head}_raw" for head in HEADS)
KEYS = ["feature_date", "coin"]
HELD_COLUMNS = [*KEYS, "input_end_at", "B_target_end_at", *ALLOWED_FEATURES]


class CalibrationExperimentError(ValueError):
    """Invalid evidence, changed inputs or a violated fixed experiment contract."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise CalibrationExperimentError(reason)


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _records(frame: pd.DataFrame) -> list[dict]:
    return frame.astype(object).where(frame.notna(), None).to_dict("records")


def _matrix_digest(frame: pd.DataFrame) -> str:
    values = np.array(frame.to_numpy(), dtype="<f8", order="C", copy=True)
    values[np.isnan(values)] = np.nan
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _same(actual: object, expected: object, reason: str) -> None:
    _require(canonical_json_bytes(actual) == canonical_json_bytes(expected), reason)


def _time(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    _require(pd.notna(stamp) and stamp.tzinfo is not None, "timestamp must be timezone-aware")
    return stamp.tz_convert("UTC")


def _reference_rows(prediction: pd.DataFrame, reference_report: dict) -> dict:
    required = {"date", "slot", "snapshot_id", "coin", "rank", *ALLOWED_FEATURES}
    _require(isinstance(prediction, pd.DataFrame) and prediction.columns.is_unique
             and required <= set(prediction) and not prediction.empty, "invalid frozen prediction schema")
    _require(prediction.slot.eq("preopen").all() and prediction.index.is_unique,
             "invalid prediction slot/index")
    _require(not prediction.duplicated(["snapshot_id", "coin"]).any(), "duplicate prediction key")
    references = {}
    for row in reference_report["predictions"]:
        key = (row["snapshot_id"], row["coin"])
        _require(key not in references, "duplicate reference prediction key")
        references[key] = row
    _require(set(references) == set(zip(prediction.snapshot_id, prediction.coin)), "reference population differs")
    for row in _records(prediction):
        source = references[row["snapshot_id"], row["coin"]]
        _same(row, {name: source[name] for name in row}, "original prediction/label/feature differs from reference")
    return references


def _common_training(training: pd.DataFrame, decision: dict) -> pd.DataFrame:
    ends = [pd.to_datetime(training[name], utc=True) for name in ("B_target_end_at", "C_target_end_at")]
    finite = np.isfinite(training[["B_high_ret", "B_low_ret", "C_high_ret", "C_low_ret"]]).all(axis=1)
    mask = (training.feature_date.lt(decision["cutoff_exclusive"]) & training.C_target_present & finite
            & ends[0].lt(_time(decision["decision_started_at"])) & ends[1].lt(_time(decision["decision_started_at"])))
    return training.loc[mask].sort_values(KEYS, kind="stable").reset_index(drop=True)


def _validate_raw(raw: pd.DataFrame, index: pd.Index, *, available: bool) -> None:
    _require(isinstance(raw, pd.DataFrame) and list(raw.columns) == list(RAW_COLUMNS)
             and raw.index.equals(index), "raw-score schema/index mismatch")
    values = raw.to_numpy(dtype=float)
    if available:
        _require(np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(), "invalid raw probabilities")
    else:
        _require(np.isnan(values).all(), "unavailable inner model returned nonmissing raw scores")


def _record_map(edges, mapping, base: float) -> dict:
    return {"edges": np.asarray(edges, dtype=float).tolist(),
            "hit_map": {str(key): float(value) for key, value in mapping.items()}, "base": float(base)}


def run_calibration_experiment(training: pd.DataFrame, prediction: pd.DataFrame,
                               decision_metadata: dict, reference_report: dict, *, progress=None) -> dict:
    """Run all inner caches, then fixed daily reference fits and matched maps.

    Input/output are in-memory objects. No canonical test outcomes or identities
    reach either learner's feature argument. Historical score availability is a
    simulated information time, not a claim these predictions were saved then.
    """
    config = deepcopy(EXPERIMENT_CONFIG)
    references = _reference_rows(prediction, reference_report)
    _require(isinstance(training, pd.DataFrame) and training.columns.is_unique
             and not training.duplicated(KEYS).any(), "invalid training schema or duplicate keys")
    work = training.copy(deep=True).sort_values(KEYS, kind="stable").reset_index(drop=True)
    original_train = work.copy(deep=True)
    frozen = prediction.copy(deep=True).sort_values(["date", "rank"], kind="stable")
    original_frozen = frozen.copy(deep=True)
    metadata = deepcopy(decision_metadata)
    metadata_before = canonical_json_bytes(metadata)
    _require(set(metadata) == set(frozen.snapshot_id), "decision metadata population mismatch")
    _require(len(metadata) <= config["maximum_outer_decisions"], "too many outer decisions")
    audit_by_sid = {row["snapshot_id"]: row for row in reference_report["daily_fit_audit"]}
    _require(len(audit_by_sid) == len(reference_report["daily_fit_audit"])
             and set(audit_by_sid) == set(metadata), "reference daily audit mismatch")
    cutoffs = [pd.Timestamp(value["cutoff_exclusive"]) for value in metadata.values()]
    anchor = min(cutoffs) - pd.Timedelta(days=config["calibration_calendar_days"])
    limit = max(cutoffs)
    _require(anchor == pd.Timestamp(config["inner_anchor"])
             and limit <= pd.Timestamp(config["maximum_inner_end_exclusive"]), "unreviewed calibration calendar")
    for sid, group in frozen.groupby("snapshot_id", sort=False):
        decision = metadata[sid]
        _same(group.date.unique().tolist(), [decision["date"]], "decision date mismatch")
        _same(decision, audit_by_sid[sid]["decision"], "reference decision mismatch")
    inner_audit, outer_audit, cache_parts, resub_parts, prediction_parts = [], [], [], [], []
    fit_count = 0

    def unchanged() -> None:
        _require(work.equals(original_train) and frozen.equals(original_frozen), "experiment input frame mutated")
        _require(canonical_json_bytes(metadata) == metadata_before, "experiment decision metadata mutated")
        _same(EXPERIMENT_CONFIG, config, "runtime experiment configuration mutated")

    block_start = anchor
    while block_start < limit:
        block_end = min(block_start + pd.Timedelta(days=config["inner_block_calendar_days"]), limit)
        block_id = f"inner-{len(inner_audit) + 1:03d}"
        _require(len(inner_audit) < config["maximum_inner_blocks"], "too many inner blocks")
        held = work.loc[work.feature_date.ge(str(block_start.date()))
                        & work.feature_date.lt(str(block_end.date())), HELD_COLUMNS].copy(deep=True)
        before_held = held.copy(deep=True)
        if progress:
            progress({"event": "inner_started", "block_id": block_id, "rows": len(held)})
        if len(held):
            result = fit_inner_block(work, held, block_start=str(block_start.date()), block_end_exclusive=str(block_end.date()))
            raw, audit = result["raw_scores"], result["audit"]
            _require(audit["fit_count"] in (0, 5), "partial inner fit")
            available = audit["fit_count"] == 5
            _same(audit["configuration"], KERNEL_CONFIG, "inner kernel configuration differs")
            _require(audit["status"] == ("fitted" if available else "unavailable"), "inner status/fit mismatch")
            _require(not available or audit["training_dates"] >= config["minimum_inner_training_dates"], "insufficient inner training dates")
            _validate_raw(raw, held.index, available=available)
        else:
            raw = pd.DataFrame(np.nan, index=held.index, columns=RAW_COLUMNS)
            audit = {"status": "unavailable", "fit_count": 0, "reason": "empty_held_block"}
            available = False
        _require(held.equals(before_held), "held prediction inputs mutated")
        unchanged()
        fit_count += audit["fit_count"]
        origin = (block_start.tz_localize("Asia/Seoul") + pd.Timedelta(hours=9)).tz_convert("UTC")
        expected_timing = [{"feature_date": day, "coin": coin,
            "raw_prediction_available_at": (pd.Timestamp(day, tz="Asia/Seoul") + pd.Timedelta(hours=9)).tz_convert("UTC").isoformat()}
            for day, coin in zip(held.feature_date, held.coin, strict=True)]
        if len(held):
            _same(audit["model_fit_origin_at"], origin.isoformat(), "inner fit origin mismatch")
            _same(audit["prediction_timing"], expected_timing, "inner score availability/key mismatch")
            _same(audit["cutoff_exclusive"], str((block_start-pd.Timedelta(days=config["embargo_calendar_days"])).date()), "inner embargo cutoff mismatch")
        piece = held[KEYS].copy()
        piece["block_id"] = block_id
        piece["fit_origin"] = origin.isoformat()
        piece["prediction_available_at"] = [row["raw_prediction_available_at"] for row in expected_timing]
        piece["target_end_at"] = held.B_target_end_at
        piece["status"] = "fitted" if available else "unavailable"
        piece = pd.concat([piece, raw], axis=1)
        cache_parts.append(piece)
        inner_audit.append({"block_id": block_id, "block_start": str(block_start.date()),
                            "block_end_exclusive": str(block_end.date()), "held_rows": len(held),
                            "fit_origin": origin.isoformat(), **audit})
        if progress:
            progress({"event": "inner_finished", "block_id": block_id, "fit_count": audit["fit_count"]})
        block_start = block_end
    oof_cache = pd.concat(cache_parts, ignore_index=True)
    _require(not oof_cache.duplicated(KEYS).any(), "duplicate OOF cache key")
    cache_lookup = oof_cache.set_index(KEYS)
    source_score_columns = [*[f"{head}_B" for head in HEADS], "exp_downside_B", *[f"{head}_B_raw" for head in HEADS]]
    for sid, group in frozen.groupby("snapshot_id", sort=False):
        decision = metadata[sid]
        original_decision = deepcopy(decision)
        reference_scores = pd.DataFrame([{name: references[sid, coin][name] for name in source_score_columns}
                                         for coin in group.coin], index=group.index)
        features = group.loc[:, ALLOWED_FEATURES].copy(deep=True)
        features_before = features.copy(deep=True)
        if progress:
            progress({"event": "outer_started", "snapshot_id": sid, "date": decision["date"]})
        fitted = fit_final_reference(work, features, decision=decision,
                                     reference_scores=reference_scores, reference_audit=deepcopy(audit_by_sid[sid]))
        unchanged()
        _require(features.equals(features_before) and decision == original_decision, "final learner inputs mutated")
        audit, raw_train, keys, scores = (fitted[name] for name in ("audit", "raw_training", "training_keys", "scores"))
        _require(audit["fit_count"] == 5, "F reference must complete exactly five fits")
        _same(audit["configuration"], KERNEL_CONFIG, "final kernel configuration differs")
        _require(audit["status"] == "fitted" and audit["reference_parity"] == "exact_all_checks_passed", "F reference parity not confirmed")
        fit_count += 5
        _require(fit_count <= config["maximum_fits"], "exceeded fit budget")
        selected = _common_training(work, decision)
        _require(list(keys.columns) == KEYS and keys.index.equals(pd.RangeIndex(len(selected)))
                 and keys.equals(selected[KEYS]), "reference training keys differ from complete common training population")
        _validate_raw(raw_train, keys.index, available=True)
        _require(scores.index.equals(group.index), "final score index mismatch")
        _validate_raw(scores.loc[:, RAW_COLUMNS], group.index, available=True)
        F = scores.rename(columns={f"{head}_raw": f"{head}_F_raw" for head in HEADS}).copy(deep=True)
        _require(set(F.columns) == {name for head in HEADS for name in (f"{head}_F", f"{head}_F_raw")} | {"exp_downside_F"},
                 "unexpected F score columns")
        start = pd.Timestamp(decision["cutoff_exclusive"]) - pd.Timedelta(days=config["calibration_calendar_days"])
        recent_mask = selected.feature_date.ge(str(start.date()))
        recent = selected.loc[recent_mask].copy()
        resub = raw_train.loc[recent_mask].copy()
        recent_keys = recent[KEYS]
        lookup_keys = pd.MultiIndex.from_frame(recent_keys)
        matched_oof = cache_lookup.reindex(lookup_keys)
        reasons = []
        missing = matched_oof.block_id.isna()
        finite_oof = np.isfinite(matched_oof.loc[:, RAW_COLUMNS].to_numpy(dtype=float)).all(axis=1)
        if missing.any():
            reasons.append("missing_oof_rows")
        if not finite_oof.all() or not matched_oof.status.eq("fitted").all():
            reasons.append("unavailable_oof_rows")
        if len(recent) < config["matched_min_rows"] or recent.feature_date.nunique() < config["matched_min_dates"]:
            reasons.append("insufficient_matched_rows_or_dates")
        if not missing.any():
            target_end = pd.to_datetime(matched_oof.target_end_at, utc=True)
            score_available = pd.to_datetime(matched_oof.prediction_available_at, utc=True)
            _require((target_end.to_numpy() == pd.to_datetime(recent.B_target_end_at, utc=True).to_numpy()).all(), "OOF target binding mismatch")
            _require(target_end.lt(_time(decision["decision_started_at"])).all()
                     and score_available.lt(_time(decision["decision_started_at"])).all(), "calibration inputs not available before decision")
        labels, guards = {}, {}
        for head, threshold in MODEL_CONFIG["thresholds"].items():
            values = recent["B_high_ret" if threshold > 0 else "B_low_ret"].to_numpy()
            y = ((values >= threshold) if threshold > 0 else (values <= threshold)).astype(np.uint8)
            labels[head] = y
            positive, negative = int(y.sum()), len(y) - int(y.sum())
            guards[head] = {"positive": positive, "negative": negative,
                            "label_sha256": hashlib.sha256(y.tobytes()).hexdigest()}
            if positive < config["matched_min_positive"] or negative < config["matched_min_negative"]:
                reasons.append(f"insufficient_matched_classes:{head}")
        maps = {"R": {}, "O": {}}
        if not reasons:
            for arm, raw in (("R", resub), ("O", matched_oof)):
                for head in HEADS:
                    edges, mapping, base = _oof_bucket_calib(raw[f"{head}_raw"].to_numpy(), labels[head], config["calibration_buckets"])
                    if edges is None or mapping is None:
                        reasons.append(f"missing_bucket_map:{arm}:{head}")
                    else:
                        maps[arm][head] = _record_map(edges, mapping, base)
        result_rows = group.copy(deep=True)
        result_rows = pd.concat([result_rows, F], axis=1)
        result_rows["F_status"] = "fitted"
        for arm in ("R", "O"):
            for head in HEADS:
                raw = F[f"{head}_F_raw"].to_numpy()
                if reasons:
                    result_rows[f"{head}_{arm}"] = np.nan
                    result_rows[f"{head}_{arm}_raw"] = np.nan
                else:
                    record = maps[arm][head]
                    result_rows[f"{head}_{arm}"] = _apply_calib(raw, record["edges"],
                        {int(key): value for key, value in record["hit_map"].items()}, record["base"])
                    result_rows[f"{head}_{arm}_raw"] = raw
            result_rows[f"exp_downside_{arm}"] = np.nan if reasons else F.exp_downside_F
            result_rows[f"{arm}_status"] = "unavailable" if reasons else "fitted"
        prediction_parts.append(result_rows)
        resub_piece = pd.concat([recent_keys, resub], axis=1)
        resub_piece.insert(0, "snapshot_id", sid)
        resub_piece.insert(0, "outer_date", decision["date"])
        resub_parts.append(resub_piece)
        outer_audit.append({"snapshot_id": sid, "date": decision["date"], "fit_count": 5,
            "reference_fit": audit, "F_status": "fitted", "R_status": "unavailable" if reasons else "fitted",
            "O_status": "unavailable" if reasons else "fitted", "failures": reasons,
            "calibration_start": str(start.date()), "calibration_end_exclusive": decision["cutoff_exclusive"],
            "calibration_rows": len(recent), "calibration_dates": int(recent.feature_date.nunique()),
            "calibration_keys_sha256": _digest(_records(recent_keys)), "matched_head_counts": guards,
            "missing_oof_rows": int(missing.sum()), "nonfinite_oof_rows": int((~finite_oof).sum()),
            "resub_raw_sha256": _matrix_digest(resub),
            "oof_raw_sha256": _matrix_digest(matched_oof.loc[:, RAW_COLUMNS]), "maps": maps})
        if progress:
            progress({"event": "outer_finished", "snapshot_id": sid, "date": decision["date"],
                      "fit_count": 5, "matched_status": "unavailable" if reasons else "fitted"})
    unchanged()
    predictions = pd.concat(prediction_parts).sort_values(["date", "rank"], kind="stable").reset_index(drop=True)
    _same(_records(predictions.loc[:, frozen.columns]), _records(frozen.reset_index(drop=True)), "original test columns mutated")
    return {"predictions": predictions, "inner_audit": inner_audit, "outer_audit": outer_audit,
            "oof_cache": oof_cache, "resub_cache": pd.concat(resub_parts, ignore_index=True),
            "configuration": config, "fit_count": fit_count}

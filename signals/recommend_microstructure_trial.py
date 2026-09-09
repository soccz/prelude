"""One record-only prospective tie-break trial; no fitting or live dispatch.

The original R1 universe and Top3 never change. A second, immutable ranking
uses trade flow ONLY inside an exact stored-score tie crossing ranks 3/4.
Publication and evaluation are separate: a failed evaluation cannot rewrite
the forecast or manufacture a missing publication receipt after a crash.
"""
from __future__ import annotations

import math
import os
import stat
import uuid
from contextlib import contextmanager
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from notifier.delivery_receipt import DEFAULT_RECEIPT_ROOT
from ops.artifact_provenance import (
    canonical_json_bytes, file_identity, sha256_bytes, strict_json_object,
    strict_json_object_bytes,
)
from ops.recommendation_evidence import (
    EvidenceUnavailable, aware_time, load_recommendation_evidence,
)
from signals.recommend_experiment_eval import (
    METRICS, _metric_rows, _named, _summary, _validate_outcomes,
)
from signals.recommend_microstructure import (
    FEATURE, MICROSTRUCTURE_CONFIG, SCHEMA as FEATURE_SCHEMA,
    _capture_quality, _ns, _snapshot as validate_micro_snapshot,
)
from signals.recommend_snapshot import load_snapshot

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRIAL_ROOT = ROOT / "output/recommend_microstructure_trials"
TRIAL_ID = "r1_boundary_trade_imbalance_v1"
SCORE_SCHEMA = "recommend_microstructure_trial_score.v1"
COMMIT_SCHEMA = "recommend_microstructure_trial_commit.v1"
EVALUATION_SCHEMA = "recommend_microstructure_trial_evaluation.v1"
TIE_FIELDS = ("rr_ratio", "p_up10", "p_dn5", "p_dn10", "exp_downside")
TRIAL_CONFIG = {
    "trial_id": TRIAL_ID, "slot": "open", "universe_size": 100, "top_k": 3,
    "exact_stored_tie_fields": list(TIE_FIELDS),
    "rule": "only_contiguous_exact_tie_block_crossing_rank3_rank4",
    "tie_break": [f"{FEATURE}:descending", "original_rank:ascending"],
    "no_tolerance_or_weight_sweep": True, "missing_required_feature": "unavailable",
    "no_op_days_in_primary": True, "feature": FEATURE, "lookback_seconds": 300,
    "prospective_deadline": "score_durable_observed_at < canonical_execution_start",
    "prospective_start_asof": "2026-09-08", "scheduled_capture_start_kst": "08:45:00",
    "automatic_promotion": False, "places_orders": False,
}
GENERATOR_SOURCES = (
    "signals/recommend_microstructure_trial.py", "signals/recommend_microstructure.py",
    "data/upbit_microstructure.py", "signals/recommend_snapshot.py",
    "ops/artifact_provenance.py", "ops/recommendation_evidence.py",
    "signals/recommend_experiment_eval.py", "ledger/config.py",
    "notifier/delivery_receipt.py",
)


class MicrostructureTrialError(ValueError):
    """Corrupt/conflicting evidence blocks the operation without overwriting it."""


def _require(value, reason):
    if not value:
        raise MicrostructureTrialError(reason)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _now():
    return datetime.now(timezone.utc)


def _time(value):
    try:
        return aware_time(value).astimezone(timezone.utc)
    except ValueError as exc:
        raise MicrostructureTrialError("aware timestamp required") from exc


def _seal(document):
    return {**document, "payload_sha256": sha256_bytes(canonical_json_bytes(document))}


def _checked(document, schema):
    _require(document.get("schema") == schema, "unsupported trial schema")
    expected = _seal({key: value for key, value in document.items() if key != "payload_sha256"})
    _require(document == expected, "trial payload checksum mismatch")
    _require(document.get("trial_id") == TRIAL_ID, "trial id mismatch")
    return document


def _path(value):
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _verify_identities(items):
    _require(isinstance(items, list) and bool(items), "empty source manifest")
    names = []
    for item in items:
        _require(isinstance(item, dict) and item.get("exists") is True,
                 "source manifest requires existing regular files")
        names.append(str(_path(item["path"]).resolve()))
        _require(file_identity(_path(item["path"]), root=ROOT) == item,
                 f"source changed: {item['path']}")
    _require(len(names) == len(set(names)), "duplicate source manifest path")


def _generator_sources():
    return [file_identity(ROOT / relative, root=ROOT) for relative in GENERATOR_SOURCES]


def _remember_inputs(collected, items):
    for item in items:
        key = str(_path(item["path"]).absolute())
        _require(key not in collected or collected[key] == item, "input changed between records")
        collected[key] = item


def _validate_feature(snapshot, feature):
    candidates, cutoff, start = validate_micro_snapshot(snapshot)
    _require(feature.get("schema") == FEATURE_SCHEMA, "unsupported feature schema")
    _require(feature.get("config") == MICROSTRUCTURE_CONFIG, "feature config mismatch")
    identity = {key: snapshot[key] for key in (
        "snapshot_id", "payload_sha256", "asof", "slot", "decision_started_at"
    )}
    _require(feature.get("snapshot") == identity, "feature snapshot binding mismatch")
    window = feature.get("window") or {}
    _require(window.get("start_at_ns") == start
             and window.get("end_exclusive_at_ns") == cutoff
             and _ns(window["start_at"]) == start
             and _ns(window["end_exclusive_at"]) == cutoff,
             "feature window mismatch")
    _require(type(feature.get("feature_evidence_valid")) is bool, "invalid evidence flag")
    reasons = feature.get("quality_reasons")
    _require(isinstance(reasons, list) and reasons == sorted(set(reasons))
             and all(isinstance(reason, str) and reason for reason in reasons)
             and feature["feature_evidence_valid"] == (not reasons), "feature quality flag/reasons mismatch")
    _require(feature.get("experiment_readiness") == "not_assessed"
             and feature.get("effect_status") == "not_evaluated", "unexpected feature promotion claim")
    rows = feature.get("rows")
    _require(isinstance(rows, list) and len(rows) == 100, "feature requires original 100 rows")
    by_coin = {row["coin"]: row for row in rows}
    _require(len(by_coin) == 100 and set(by_coin) == {row["coin"] for row in candidates},
             "feature universe mismatch")
    available = []
    for candidate in candidates:
        row = by_coin[candidate["coin"]]
        _require(type(row["rank"]) is int and row["rank"] == candidate["rank"],
                 "feature original rank mismatch")
        for name in TIE_FIELDS:
            _require(_finite(candidate.get(name)), f"invalid stored tie field: {name}")
        count, bid, ask = row["observed_trade_count"], row["bid_notional"], row["ask_notional"]
        _require(type(count) is int and count >= 0 and _finite(bid) and bid >= 0
                 and _finite(ask) and ask >= 0 and math.isfinite(bid + ask), "invalid trade totals")
        value, status = row[FEATURE], row["feature_status"]
        if feature["feature_evidence_valid"] and bid + ask > 0:
            _require(status == "available" and count > 0 and _finite(value) and -1 <= value <= 1
                     and math.isclose(value, (bid - ask) / (bid + ask), abs_tol=1e-12),
                     "feature value/totals mismatch")
            available.append(candidate["coin"])
        elif feature["feature_evidence_valid"]:
            _require(status == "no_observed_trades" and count == 0 and value is None,
                     "zero trades must have a null feature")
        else:
            _require(status == "unavailable" and value is None, "invalid capture must not expose usable features")
    _require(feature.get("feature_available_rows") == len(available)
             and feature.get("feature_coverage") == len(available) / 100,
             "feature coverage mismatch")
    coverage = feature.get("coverage") or {}
    _require(coverage.get("recorded_rows") == 100
             and coverage.get("feature_available_rows") == len(available)
             and coverage.get("available_coins") == available
             and coverage.get("no_observed_trade_coins") == [row["coin"] for row in candidates
                  if by_coin[row["coin"]]["observed_trade_count"] == 0], "feature coverage detail mismatch")
    for name in ("missing_capture_coins", "missing_initial_book_coins"):
        missing = coverage.get(name)
        _require(isinstance(missing, list) and len(missing) == len(set(missing))
                 and set(missing).issubset(by_coin), "invalid missing feature coverage")
        _require(not feature["feature_evidence_valid"] or not missing,
                 "valid feature with missing capture/book coverage")
    return candidates, by_coin


def plan_microstructure_trial(snapshot, feature_document):
    """Pure, outcome-blind plan; exact stored ties only, never approximate ties."""
    try:
        candidates, features = _validate_feature(snapshot, feature_document)
        def key(row):
            return tuple(row[name] for name in TIE_FIELDS)
        block = []
        if key(candidates[2]) == key(candidates[3]):
            left, right = 2, 3
            while left and key(candidates[left - 1]) == key(candidates[2]):
                left -= 1
            while right + 1 < len(candidates) and key(candidates[right + 1]) == key(candidates[2]):
                right += 1
            block = candidates[left:right + 1]
        missing = [row["coin"] for row in block if features[row["coin"]][FEATURE] is None]
        reason = ("feature_evidence_invalid" if not feature_document["feature_evidence_valid"]
                  else "required_tie_feature_unavailable" if missing else None)
        control = [row["coin"] for row in candidates]
        challenger = None if reason else list(control)
        if challenger is not None and block:
            ordered = sorted(block, key=lambda row: (-features[row["coin"]][FEATURE], row["rank"]))
            challenger[block[0]["rank"] - 1:block[-1]["rank"]] = [row["coin"] for row in ordered]
        changed_picks = len(set(challenger[:3]) - set(control[:3])) if challenger else None
        return {
            "status": "unavailable" if reason else "planned", "reason": reason,
            "control_ranking": control, "challenger_ranking": challenger,
            "control_top3": control[:3], "challenger_top3": challenger[:3] if challenger else None,
            "boundary_opportunity": bool(block), "boundary_block": [row["coin"] for row in block],
            "required_missing_features": missing, "changed_picks": changed_picks,
            "is_no_op": changed_picks == 0 if changed_picks is not None else None,
            "recorded_candidates": 100,
            "feature_available_rows": feature_document["feature_available_rows"],
            "feature_coverage": feature_document["feature_coverage"],
            "candidate_inputs": [
                {"coin": row["coin"], "rank": row["rank"],
                 **{name: row[name] for name in TIE_FIELDS}, FEATURE: features[row["coin"]][FEATURE]}
                for row in candidates
            ],
            "note": "Exact stored-score ties are not proof of equal unrounded production scores.",
        }
    except MicrostructureTrialError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise MicrostructureTrialError(f"invalid trial inputs: {type(exc).__name__}") from exc


def _load_inputs(snapshot_path, feature_path):
    paths = [Path(snapshot_path), Path(feature_path)]
    before = [file_identity(path, root=ROOT) for path in paths]
    _require(all(item["exists"] for item in before), "snapshot or feature file missing")
    snapshot = load_snapshot(paths[0], slot="open", ranking="R1", model_id="recommend_r1_open")
    feature = strict_json_object(paths[1])
    provenance = feature.get("provenance") or {}
    _require(provenance.get("inputs_unchanged") is True, "unverified feature inputs")
    feature_sources = provenance.get("files")
    _verify_identities(feature_sources)
    expected_generators = {
        (ROOT / name).resolve() for name in (
            "signals/recommend_microstructure.py", "data/upbit_microstructure.py",
            "signals/recommend_snapshot.py", "ops/artifact_provenance.py",
        )
    }
    generator_sources = provenance.get("generator_sources") or []
    _require({_path(item["path"]).resolve() for item in generator_sources} == expected_generators,
             "feature generator manifest mismatch")
    _require(all(item in feature_sources for item in generator_sources), "unbound feature generators")
    _require(any(_path(item["path"]).resolve() == paths[0].resolve()
                 and item["sha256"] == before[0]["sha256"] for item in feature_sources),
             "feature source snapshot mismatch")
    _require(len(feature_sources) >= 6 and feature_sources[0] == before[0]
             and feature_sources[2:6] == generator_sources, "feature provenance layout mismatch")
    manifest_path = _path(feature_sources[1]["path"])
    manifest = strict_json_object(manifest_path)
    _require(sha256_bytes(canonical_json_bytes(manifest)) == provenance.get("capture_manifest_payload_sha256")
             and manifest.get("capture_id") == feature.get("capture_id"), "capture manifest binding mismatch")
    candidates, cutoff, start = validate_micro_snapshot(snapshot)
    reasons, _markets, missing, _deadline = _capture_quality(snapshot, manifest, candidates, cutoff, start)
    _require(set(reasons).issubset(feature["quality_reasons"])
             and missing == feature["coverage"]["missing_capture_coins"], "capture quality evidence mismatch")
    cutoff_source = manifest.get("cutoff_source") or {}
    if cutoff_source.get("kind") == "recommend_snapshot":
        _require(_path(cutoff_source["path"]).resolve() == paths[0].resolve()
                 and cutoff_source.get("file_sha256") == before[0]["sha256"], "capture snapshot file binding mismatch")
    expected_files = list(feature_sources[:6])
    raw_artifacts = {}
    for channel in ("trade", "orderbook"):
        artifact = manifest["streams"][channel].get("artifact")
        raw_artifacts[channel] = artifact
        if artifact and artifact.get("exists") is True and artifact.get("writer_state") == "finalized":
            raw_path = _path(artifact["path"])
            _require(raw_path.resolve() == (manifest_path.parent / f"{channel}.jsonl.gz").resolve(),
                     "capture raw path mismatch")
            identity = file_identity(raw_path, root=ROOT)
            _require(identity.get("sha256") == artifact.get("sha256")
                     and identity.get("size") == artifact.get("size_bytes"), "capture raw file binding mismatch")
            expected_files.append(identity)
    _require(feature_sources == expected_files and provenance.get("raw_artifacts") == raw_artifacts,
             "feature raw provenance graph mismatch")
    plan = plan_microstructure_trial(snapshot, feature)
    after = [file_identity(path, root=ROOT) for path in paths]
    _require(before == after, "snapshot or feature changed while reading")
    _verify_identities(feature_sources)
    return snapshot, feature, plan, before


@contextmanager
def _directory(root, parts=(), *, create=False):
    """Pin root/children with no-follow descriptors, retaining root-parent aliases."""
    root = Path(os.path.abspath(root))
    anchor, names = root.parent, [root.name, *parts]
    while not anchor.exists() and not anchor.is_symlink():
        names.insert(0, anchor.name)
        anchor = anchor.parent
    anchor = anchor.resolve(strict=True)
    held = []
    try:
        fd = os.open(anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        held.append((anchor, fd))
        for name in names:
            _require(name not in {"", ".", ".."} and "/" not in name, "invalid output component")
            if create:
                try:
                    os.mkdir(name, 0o700, dir_fd=fd)
                except FileExistsError:
                    pass
                os.fsync(fd)
            try:
                child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            except FileNotFoundError:
                if not create:
                    yield None
                    return
                raise
            anchor, fd = anchor / name, child
            held.append((anchor, fd))
        def verify():
            for path, handle in held:
                observed, opened = path.lstat(), os.fstat(handle)
                _require(stat.S_ISDIR(observed.st_mode)
                         and (observed.st_dev, observed.st_ino) == (opened.st_dev, opened.st_ino),
                         "trial directory changed")
        verify()
        yield (anchor, fd, verify)
        verify()
    finally:
        for _, fd in reversed(held):
            os.close(fd)


def _read_document(directory, name):
    _, fd, verify = directory
    verify()
    try:
        handle = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
    except FileNotFoundError:
        return None
    with os.fdopen(handle, "rb") as stream:
        before = os.fstat(stream.fileno())
        _require(stat.S_ISREG(before.st_mode), "trial artifact is not a regular file")
        content = stream.read(4 * 1024 * 1024 + 1)
        _require(len(content) <= 4 * 1024 * 1024, "trial artifact is too large")
        after = os.fstat(stream.fileno())
    current = os.stat(name, dir_fd=fd, follow_symlinks=False)
    _require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
             == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
             == (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns),
             "trial artifact changed while reading")
    verify()
    return strict_json_object_bytes(content)


def _publish_new(directory, name, document):
    _, fd, verify = directory
    verify()
    temporary = f".{name}.{uuid.uuid4().hex}.tmp"
    handle = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(canonical_json_bytes(document) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, name, src_dir_fd=fd, dst_dir_fd=fd, follow_symlinks=False)
        os.fsync(fd)
        verify()
    finally:
        os.unlink(temporary, dir_fd=fd)
        os.fsync(fd)


def write_new_trial_report(path, report):
    """Write a new report durably; existing files are never replaced."""
    path = Path(path)
    with _directory(path.parent, create=True) as directory:
        _publish_new(directory, path.name, report)


def _read_record(directory, *, now):
    artifact_paths = [directory[0] / name for name in ("score.json", "commit.json")]
    artifact_before = [file_identity(path, root=ROOT) for path in artifact_paths]
    score, commit = _read_document(directory, "score.json"), _read_document(directory, "commit.json")
    _require(artifact_before == [file_identity(path, root=ROOT) for path in artifact_paths],
             "trial artifacts changed while reading")
    if score is None:
        _require(commit is None, "orphan commit without score")
        return {"status": "missing", "reason": "trial_score_missing"}
    _checked(score, SCORE_SCHEMA)
    _require(score.get("config") == TRIAL_CONFIG, "trial config mismatch")
    _require(directory[0].parent.name == score["asof"], "trial directory date mismatch")
    _require(score.get("slot") == "open", "trial slot mismatch")
    _require(score.get("deployable") is False and score.get("effect_status") == "not_evaluated",
             "unexpected trial promotion claim")
    _require(score.get("generator_sources") == _generator_sources(), "trial generator changed")
    _verify_identities(score["source_inputs"])
    snapshot, feature, plan, sources = _load_inputs(
        _path(score["source_inputs"][0]["path"]), _path(score["source_inputs"][1]["path"])
    )
    _require(sources == score["source_inputs"] and plan == score["plan"], "trial plan/source mismatch")
    _require(score["snapshot_id"] == snapshot["snapshot_id"]
             and score["asof"] == snapshot["asof"], "trial snapshot identity mismatch")
    planned = _time(score["planned_at"])
    _require(_time(snapshot["decision_completed_at"]) <= planned <= now, "trial planning clock invalid")
    result = {"score": score, "snapshot": snapshot, "feature": feature,
              "trial_artifacts": artifact_before,
              "score_path": str(directory[0] / "score.json"), "commit_path": str(directory[0] / "commit.json")}
    if commit is None:
        return {**result, "status": "uncertain", "reason": "score_without_durable_commit_receipt"}
    _checked(commit, COMMIT_SCHEMA)
    _require(commit["score_payload_sha256"] == score["payload_sha256"]
             and commit["snapshot_id"] == snapshot["snapshot_id"]
             and commit["asof"] == snapshot["asof"], "commit/score binding mismatch")
    _require(commit.get("clock_contract") == "observed_after_score_file_and_directory_fsync_not_commit_completion",
             "unsupported publication clock contract")
    _require(planned <= _time(commit["score_durable_observed_at"]) <= now, "durable score clock invalid")
    return {**result, "commit": commit, "status": "committed", "reason": None}


def record_microstructure_trial(snapshot_path, feature_path, *, output_root=DEFAULT_TRIAL_ROOT, now_fn=_now):
    """Publish once; crash gaps remain uncertain and cannot be retroactively repaired."""
    snapshot, _feature, plan, sources = _load_inputs(snapshot_path, feature_path)
    planned = _time(now_fn())
    _require(_time(snapshot["decision_completed_at"]) <= planned, "planning before snapshot completion")
    score = _seal({
        "schema": SCORE_SCHEMA, "trial_id": TRIAL_ID, "config": TRIAL_CONFIG,
        "asof": snapshot["asof"], "slot": "open", "snapshot_id": snapshot["snapshot_id"],
        "planned_at": planned.isoformat(), "source_inputs": sources,
        "generator_sources": _generator_sources(), "plan": plan,
        "deployable": False, "effect_status": "not_evaluated",
    })
    with _directory(output_root, (snapshot["asof"], TRIAL_ID), create=True) as directory:
        existing = _read_record(directory, now=planned)
        if existing["status"] != "missing":
            _require(existing["score"]["source_inputs"] == sources, "existing trial has different feature/snapshot inputs")
            return {"status": existing["status"], "reused": True, "reason": existing["reason"],
                    "score_path": existing["score_path"], "commit_path": existing["commit_path"],
                    "plan_status": existing["score"]["plan"]["status"]}
        try:
            _publish_new(directory, "score.json", score)
        except FileExistsError:
            # Another publisher owns the first score. Never issue its receipt.
            return {"status": "uncertain", "reused": True, "reason": "concurrent_score_publication",
                    "score_path": str(directory[0] / "score.json"),
                    "commit_path": str(directory[0] / "commit.json"), "plan_status": plan["status"]}
        durable = _time(now_fn())
        _require(durable >= planned, "clock moved backwards after score durability")
        _verify_identities(sources)
        _require(score["generator_sources"] == _generator_sources(), "generator changed during publication")
        commit = _seal({
            "schema": COMMIT_SCHEMA, "trial_id": TRIAL_ID,
            "asof": snapshot["asof"], "snapshot_id": snapshot["snapshot_id"],
            "score_payload_sha256": score["payload_sha256"],
            "score_durable_observed_at": durable.isoformat(),
            "clock_contract": "observed_after_score_file_and_directory_fsync_not_commit_completion",
        })
        _publish_new(directory, "commit.json", commit)
        return {"status": "committed", "reused": False, "reason": None,
                "score_path": str(directory[0] / "score.json"),
                "commit_path": str(directory[0] / "commit.json"),
                "score_durable_observed_at": durable.isoformat(),
                "plan_status": plan["status"], "changed_picks": plan["changed_picks"],
                "prospective_status": "pending_canonical_delivery_entry_check"}


def _cohort_summary(rows, *, n_boot, seed):
    if not rows:
        return {"n_dates": 0, "effect_status": "not_evaluated", "metrics": None}
    control = np.asarray([row["control"] for row in rows])
    challenger = np.asarray([row["challenger"] for row in rows])
    difference = challenger - control
    return {
        "n_dates": len(rows), "n_no_op_dates": sum(row["changed_picks"] == 0 for row in rows),
        "n_changed_dates": sum(row["changed_picks"] > 0 for row in rows),
        "changed_picks": sum(row["changed_picks"] for row in rows),
        "picks_per_arm": 3 * len(rows), "effect_status": "descriptive_only_not_a_promotion_verdict",
        "metrics": {"control": _summary(control, n_boot, seed),
                    "challenger": _summary(challenger, n_boot, seed),
                    "challenger_minus_control": _summary(difference, n_boot, seed)},
        "daily": [{"date": row["date"], "changed_picks": row["changed_picks"],
                   "control": _named(np.asarray(row["control"])),
                   "challenger": _named(np.asarray(row["challenger"])),
                   "challenger_minus_control": _named(delta)} for row, delta in zip(rows, difference, strict=True)],
        "leave_one_date_out": [
            {"omitted_date": row["date"], "mean_delta": _named(np.delete(difference, i, axis=0).mean(axis=0))}
            for i, row in enumerate(rows) if len(rows) > 1
        ],
    }


def evaluate_microstructure_trials(trial_root=DEFAULT_TRIAL_ROOT, *, label_root=ROOT / "output/recommend_score_labels",
                                  receipt_root=DEFAULT_RECEIPT_ROOT, now=None,
                                  n_boot=1000, seed=42):
    """Read-only cumulative diagnostics of already committed trial predictions."""
    now = _time(_now() if now is None else now)
    _require(type(n_boot) is int and 1 <= n_boot <= 100_000, "invalid n_boot")
    _require(type(seed) is int and 0 <= seed < 2**64, "invalid seed")
    sources_before = _generator_sources()
    root = Path(trial_root)
    _require(not root.is_symlink(), "trial root is symlink")
    audits, paired, full, trial_inputs, evidence_inputs = [], [], [], [], {}
    days = []
    if root.exists():
        _require(root.is_dir(), "trial root is not a directory")
        for path in sorted(root.iterdir()):
            try:
                day = date.fromisoformat(path.name)
            except ValueError:
                continue
            _require(path.name == day.isoformat() and not path.is_symlink() and path.is_dir(), "invalid trial date directory")
            days.append(day.isoformat())
    local_now = now.astimezone(timezone(timedelta(hours=9)))
    launch = date.fromisoformat(TRIAL_CONFIG["prospective_start_asof"])
    expected = [launch + timedelta(days=i) for i in range(max(0, (local_now.date() - launch).days + 1))]
    due = [day.isoformat() for day in expected if day < local_now.date()
           or local_now.time() >= time.fromisoformat(TRIAL_CONFIG["scheduled_capture_start_kst"])]
    not_due = [day.isoformat() for day in expected if day.isoformat() not in due]
    for day in sorted(set(days) | set(due) | set(not_due)):
        with _directory(root, (day, TRIAL_ID), create=False) as directory:
            record = _read_record(directory, now=now) if directory else {"status": "missing", "reason": "trial_score_missing"}
        audit = {"date": day, "record_status": record["status"], "status": record["status"], "reason": record["reason"]}
        audits.append(audit)
        trial_inputs.extend(record.get("trial_artifacts", []))
        if "feature" in record:
            _remember_inputs(evidence_inputs, record["score"]["source_inputs"])
            _remember_inputs(evidence_inputs, record["feature"]["provenance"]["files"])
        if day in not_due and record["status"] == "missing":
            audit.update(status="not_due", reason="scheduled_capture_has_not_started")
            continue
        if record["status"] != "committed":
            continue
        score, commit = record["score"], record["commit"]
        plan = score["plan"]
        audit.update(snapshot_id=score["snapshot_id"], plan_status=plan["status"],
                     control_top3=plan["control_top3"], challenger_top3=plan["challenger_top3"],
                     changed_picks=plan["changed_picks"], boundary_opportunity=plan["boundary_opportunity"],
                     feature_coverage=plan["feature_coverage"], score_durable_observed_at=commit["score_durable_observed_at"])
        if day < launch.isoformat():
            audit.update(status="prelaunch_replay", reason="before_frozen_prospective_launch_date")
            continue
        if plan["status"] != "planned":
            audit.update(status="unavailable", reason=plan["reason"])
            continue
        try:
            evidence = load_recommendation_evidence(
                _path(score["source_inputs"][0]["path"]), label_root=label_root,
                receipt_root=receipt_root, now=now,
            )
        except EvidenceUnavailable as exc:
            reason = str(exc)
            pending = reason in {"label_missing", "receipt_missing_delivery_unknown", "label_not_available_as_of_now"} or reason.startswith("label_status=")
            audit.update(status="pending" if pending else "unavailable", reason=reason)
            continue
        _require(evidence["snapshot"]["snapshot_id"] == score["snapshot_id"], "evaluation snapshot changed")
        _remember_inputs(evidence_inputs, evidence["manifest"]["files"].values())
        audit["evidence_manifest"] = evidence["manifest"]
        entry = _time(evidence["label"]["execution_start_at"])
        audit["canonical_execution_start_at"] = entry.isoformat()
        if _time(commit["score_durable_observed_at"]) >= entry:
            audit.update(status="late", reason="score_not_durable_before_canonical_entry")
            continue
        frame = pd.DataFrame(evidence["label"]["rows"]).sort_values("coin").reset_index(drop=True)
        indices = {row.coin: i for i, row in frame.iterrows()}
        choices = {arm: sorted(indices[coin] for coin in plan[f"{arm}_top3"]) for arm in ("control", "challenger")}
        # Membership is fixed in the published score, before any label inspection.
        _validate_outcomes(frame)
        needed = set(choices["control"]) | set(choices["challenger"])
        missing = [str(frame.at[i, "coin"]) for i in sorted(needed) if frame.at[i, "label_status"] != "labeled"]
        if missing:
            audit.update(status="unavailable", reason="selected_outcome_unavailable", missing_selected=missing)
            continue
        row = {"date": day, "changed_picks": plan["changed_picks"],
               **{arm: _metric_rows(frame, picks).mean(axis=0).tolist() for arm, picks in choices.items()}}
        paired.append(row)
        audit.update(status="prospective_comparable", reason=None)
        if frame["label_status"].eq("labeled").all():
            baseline = _metric_rows(frame, frame.index.tolist()).mean(axis=0)
            full.append({**row, "baseline": baseline.tolist()})
            audit["full_universe_baseline_status"] = "available"
        else:
            audit["full_universe_baseline_status"] = "unavailable"
            audit["full_universe_missing"] = frame.loc[frame["label_status"] != "labeled", "coin"].tolist()
    primary = _cohort_summary(paired, n_boot=n_boot, seed=seed)
    baseline = _cohort_summary(full, n_boot=n_boot, seed=seed)
    if full:
        values = np.asarray([row["baseline"] for row in full])
        baseline["full_universe"] = _summary(values, n_boot, seed)
        for arm in ("control", "challenger"):
            baseline[f"{arm}_minus_full_universe"] = _summary(
                np.asarray([row[arm] for row in full]) - values, n_boot, seed
            )
    _require(sources_before == _generator_sources(), "evaluation generator changed")
    _require(trial_inputs == [file_identity(_path(item["path"]), root=ROOT) for item in trial_inputs],
             "trial inputs changed during evaluation")
    _require(list(evidence_inputs.values()) == [file_identity(_path(item["path"]), root=ROOT)
                                              for item in evidence_inputs.values()],
             "feature or canonical evidence changed during evaluation")
    return _seal({
        "schema": EVALUATION_SCHEMA, "trial_id": TRIAL_ID, "config": TRIAL_CONFIG,
        "generated_at": now.isoformat(), "generator_sources": sources_before,
        "trial_inputs": trial_inputs, "inputs_unchanged": True,
        "evidence_inputs": list(evidence_inputs.values()),
        "status": "evaluated_descriptively" if paired else "no_comparable_prospective_dates",
        "effect_status": primary["effect_status"], "deployable": False, "automatic_promotion": False,
        "n_boot": n_boot, "seed": seed, "metrics": list(METRICS),
        "coverage": {"recorded_date_directories": len(days), "paired_dates": len(paired),
                     "prospective_start_asof": launch.isoformat(), "calendar_expected_dates": due,
                     "calendar_expected_count": len(due), "calendar_not_due_dates": not_due,
                     "calendar_missing_dates": [row["date"] for row in audits
                         if row["date"] in due and row["status"] == "missing"],
                     "calendar_missing_count": sum(row["date"] in due and row["status"] == "missing" for row in audits),
                     "states": {state: sum(row["status"] == state for row in audits)
                                for state in sorted({row["status"] for row in audits})}},
        "primary_paired": primary, "full_universe_common": baseline, "dates": audits,
        "methodology": {
            "primary": "same-date control/challenger Top3; include no-op dates; outcomes never reselect picks",
            "cost": "copy canonical cost-adjusted pick returns without a second subtraction",
            "uncertainty": "IID date and 3 observed-date blocks; fewer than 5 dates gives no CI",
            "limits": "not actual trades; no portfolio Sharpe/compounding; descriptive monitoring, not automatic adoption",
            "clock": "local UTC observed after score durability; receipt timestamp is not its own durable completion time",
            "confounding": "Equal stored risk scores do not prove equal volatility or liquidity; adoption requires separate validation.",
            "missing": "Calendar absence means missing as of observation, not confirmed no-signal; no R1 decision is substituted.",
        },
    })

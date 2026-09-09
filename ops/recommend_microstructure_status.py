"""Bounded read-only operational check, NOT full raw/prospective validation.

No capture, recovery, publication, label join, model fit, or notification here.
The canonical research evaluator remains responsible for raw/source hashes,
delivery-entry eligibility and performance. A successful check proves only the
document bindings and raw-file presence/size observed during this check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from ops.artifact_provenance import canonical_json_bytes, strict_json_object_bytes
from signals import recommend_microstructure as feature_contract
from signals import recommend_microstructure_trial as trial

ROOT = Path(__file__).resolve().parent.parent
KST = ZoneInfo("Asia/Seoul")
# Installed Sep 8 AFTER the morning window; never change the research denominator.
OPERATIONAL_START = date(2026, 9, 9)
DUE_KST = time(10)  # Initial allowance: 08:45 + 2700s + 90s stop + grace.
MAX_DOCUMENT_BYTES = 4 * 1024 * 1024


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _token(value):
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _declared_graph(items, expected):
    """Check the native graph shape, not current generator/raw content hashes."""
    _require(
        isinstance(items, list) and len(items) == len(expected),
        "source graph length mismatch",
    )
    for item, path in zip(items, expected, strict=True):
        _require(
            isinstance(item, dict)
            and item.get("exists") is True
            and type(item.get("size")) is int
            and item["size"] >= 0
            and feature_contract._sha(item.get("sha256"))
            and trial._path(item["path"]).resolve() == Path(path).resolve(),
            "source graph identity mismatch",
        )


class _ReadSet:
    """Pin directories and bound documents; raw streams are stat-only."""

    def __init__(self):
        self.observed = {}
        self.identities = {}

    def observe(self, path):
        path = Path(path)
        try:
            value = path.lstat()
        except FileNotFoundError:
            value = None
        token = _token(value) if value else None
        _require(
            path not in self.observed or self.observed[path] == token,
            f"evidence changed during check: {path.name}",
        )
        self.observed[path] = token
        return value

    def directory(self, path):
        value = self.observe(path)
        _require(
            value is None or stat.S_ISDIR(value.st_mode), "unsafe evidence directory"
        )
        return value is not None

    def document(self, path):
        path = Path(path)
        before = self.observe(path)
        if before is None:
            return None
        _require(
            stat.S_ISREG(before.st_mode), "document must be a regular non-symlink file"
        )
        _require(
            before.st_size <= MAX_DOCUMENT_BYTES, "document exceeds bounded read limit"
        )
        with trial._directory(path.parent, create=False) as directory:
            _require(directory is not None, "document directory disappeared")
            fd = os.open(
                path.name,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=directory[1],
            )
            with os.fdopen(fd, "rb") as handle:
                _require(
                    _token(os.fstat(handle.fileno())) == _token(before),
                    "document replaced",
                )
                content = handle.read(MAX_DOCUMENT_BYTES + 1)
                _require(
                    len(content) <= MAX_DOCUMENT_BYTES, "document grew beyond limit"
                )
                _require(
                    _token(os.fstat(handle.fileno())) == _token(before),
                    "document changed",
                )
        self.observe(path)
        self.identities[path.resolve()] = (
            len(content),
            hashlib.sha256(content).hexdigest(),
        )
        return strict_json_object_bytes(content)

    def bound(self, item, expected):
        _require(
            isinstance(item, dict) and item.get("exists") is True,
            "missing file identity",
        )
        actual = trial._path(item["path"]).resolve()
        _require(actual == Path(expected).resolve(), "bound file location mismatch")
        _require(
            (item.get("size"), item.get("sha256")) == self.identities[actual],
            "bound document checksum/size mismatch",
        )

    def unchanged(self):
        for path in list(self.observed):
            self.observe(path)


def _documents_status(reader, *, capture, snapshot_path, trial_dir, day, now):
    manifest_path = capture / "manifest.json"
    manifest = reader.document(manifest_path)
    if manifest is None:
        return "capture_unfinished", "capture directory exists without final manifest"
    _require(
        manifest.get("asof") == day and manifest.get("capture_id") == capture.name,
        "capture date/directory identity mismatch",
    )
    _require(
        manifest.get("schema_version") == feature_contract.MANIFEST_SCHEMA_VERSION,
        "unsupported capture schema",
    )
    if manifest.get("complete") is not True:
        return "capture_incomplete", str(
            manifest.get("stop_reason", "completion unconfirmed")
        )
    snapshot = reader.document(snapshot_path)
    if snapshot is None:
        return "upstream_snapshot_missing", "original open R1 snapshot missing"
    _require(snapshot.get("asof") == day, "snapshot date mismatch")
    candidates, cutoff, start = feature_contract._snapshot(snapshot)
    reasons, _markets, _missing, _deadline = feature_contract._capture_quality(
        snapshot, manifest, candidates, cutoff, start
    )
    _require(
        feature_contract._ns(now.isoformat()) >= manifest["ended_at_ns"],
        "capture ends in the future",
    )
    _require(
        manifest["ended_at_ns"]
        >= feature_contract._ns(snapshot["decision_completed_at"]),
        "capture ended before original snapshot completion",
    )
    if reasons:
        return "capture_quality_invalid", ",".join(reasons)
    source = manifest["cutoff_source"]
    _require(
        trial._path(source["path"]).resolve() == snapshot_path.resolve()
        and source.get("file_sha256") == reader.identities[snapshot_path.resolve()][1],
        "capture snapshot binding mismatch",
    )
    raw = {}
    for channel in ("trade", "orderbook"):
        artifact = manifest["streams"][channel]["artifact"]
        raw[channel] = artifact
        path = capture / f"{channel}.jsonl.gz"
        _require(
            trial._path(artifact["path"]).resolve() == path.resolve(),
            "raw path mismatch",
        )
        value = reader.observe(path)
        _require(
            value is not None and stat.S_ISREG(value.st_mode),
            "raw stream missing or unsafe",
        )
        _require(
            artifact.get("exists") is True
            and artifact.get("writer_state") == "finalized"
            and value.st_size == artifact.get("size_bytes"),
            "raw stream not finalized or size mismatch",
        )
    feature_path = capture / "recommend_features.json"
    feature = reader.document(feature_path)
    if feature is None:
        return "feature_missing", "capture finished but feature sidecar missing"
    trial._validate_feature(snapshot, feature)
    provenance = feature["provenance"]
    _require(
        provenance.get("inputs_unchanged") is True
        and feature.get("capture_id") == capture.name
        and provenance.get("capture_manifest_payload_sha256")
        == hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
        and provenance.get("raw_artifacts") == raw,
        "feature capture binding mismatch",
    )
    reader.bound(provenance["files"][0], snapshot_path)
    reader.bound(provenance["files"][1], manifest_path)
    generator_paths = [
        trial.ROOT / name
        for name in (
            "signals/recommend_microstructure.py",
            "data/upbit_microstructure.py",
            "signals/recommend_snapshot.py",
            "ops/artifact_provenance.py",
        )
    ]
    _declared_graph(
        provenance["files"],
        [
            snapshot_path,
            manifest_path,
            *generator_paths,
            capture / "trade.jsonl.gz",
            capture / "orderbook.jsonl.gz",
        ],
    )
    _require(
        provenance.get("generator_sources") == provenance["files"][2:6],
        "feature generator graph mismatch",
    )
    for item, channel in zip(
        provenance["files"][6:], ("trade", "orderbook"), strict=True
    ):
        _require(
            item["size"] == raw[channel]["size_bytes"]
            and item["sha256"] == raw[channel]["sha256"],
            "declared raw graph mismatch",
        )
    if not feature["feature_evidence_valid"]:
        return "feature_quality_invalid", ",".join(feature["quality_reasons"])
    score = reader.document(trial_dir / "score.json")
    commit = reader.document(trial_dir / "commit.json")
    if score is None:
        return (
            ("publication_uncertain", "commit without score")
            if commit
            else ("score_missing", "valid feature exists without trial score")
        )
    trial._checked(score, trial.SCORE_SCHEMA)
    _require(
        score.get("config") == trial.TRIAL_CONFIG
        and score.get("asof") == day
        and score.get("slot") == "open"
        and score.get("snapshot_id") == snapshot["snapshot_id"]
        and score.get("deployable") is False
        and score.get("effect_status") == "not_evaluated",
        "score contract mismatch",
    )
    _declared_graph(
        score.get("generator_sources"),
        [trial.ROOT / name for name in trial.GENERATOR_SOURCES],
    )
    _require(len(score["source_inputs"]) == 2, "score input graph mismatch")
    reader.bound(score["source_inputs"][0], snapshot_path)
    reader.bound(score["source_inputs"][1], feature_path)
    plan = trial.plan_microstructure_trial(snapshot, feature)
    _require(score.get("plan") == plan, "stored plan differs from fixed trial rule")
    planned = trial._time(score["planned_at"])
    _require(
        trial._time(snapshot["decision_completed_at"]) <= planned <= now,
        "score planning clock mismatch",
    )
    _require(
        manifest["ended_at_ns"] <= feature_contract._ns(planned.isoformat()),
        "score planned before capture ended",
    )
    if commit is None:
        return (
            "publication_uncertain",
            "score exists without durable commit receipt; never reconstruct",
        )
    trial._checked(commit, trial.COMMIT_SCHEMA)
    _require(
        commit.get("asof") == day
        and commit.get("snapshot_id") == snapshot["snapshot_id"]
        and commit.get("score_payload_sha256") == score["payload_sha256"]
        and commit.get("clock_contract")
        == "observed_after_score_file_and_directory_fsync_not_commit_completion"
        and planned <= trial._time(commit["score_durable_observed_at"]) <= now,
        "commit binding/clock mismatch",
    )
    evaluation = reader.document(capture / "trial_evaluation.json")
    if evaluation is None:
        return (
            "recorded_evaluation_missing",
            "score committed but session evaluation file missing",
        )
    trial._checked(evaluation, trial.EVALUATION_SCHEMA)
    _require(
        evaluation.get("deployable") is False
        and evaluation.get("automatic_promotion") is False
        and evaluation.get("inputs_unchanged") is True
        and evaluation.get("config") == trial.TRIAL_CONFIG,
        "evaluation contract mismatch",
    )
    for name in ("score.json", "commit.json"):
        path = trial_dir / name
        matches = [
            item
            for item in evaluation["trial_inputs"]
            if trial._path(item["path"]).resolve() == path.resolve()
        ]
        _require(len(matches) == 1, "session evaluation does not bind current trial")
        reader.bound(matches[0], path)
    matches = [item for item in evaluation["dates"] if item.get("date") == day]
    _require(
        len(matches) == 1
        and matches[0].get("record_status") == "committed"
        and matches[0].get("snapshot_id") == snapshot["snapshot_id"]
        and matches[0].get("plan_status") == plan["status"],
        "session evaluation missing current committed date",
    )
    generated = trial._time(evaluation["generated_at"])
    _require(
        trial._time(commit["score_durable_observed_at"]) <= generated <= now
        and generated.astimezone(KST).date().isoformat() == day,
        "session evaluation date/clock mismatch",
    )
    if plan["status"] == "unavailable":
        return "complete_unavailable", plan["reason"]
    return (
        ("complete_noop", "valid recorded no-op")
        if plan["is_no_op"]
        else ("complete_changed", "trial picks recorded; real recommendation unchanged")
    )


def inspect_daily_status(
    *, now=None, capture_root=None, snapshot_root=None, trial_root=None, _asof=None
):
    now = (
        datetime.now(KST)
        if now is None
        else datetime.fromisoformat(now)
        if isinstance(now, str)
        else now
    )
    _require(
        now.tzinfo is not None and now.utcoffset() is not None, "aware now required"
    )
    now = now.astimezone(KST)
    asof = now.date() if _asof is None else _asof
    _require(asof <= now.date(), "cannot inspect a future date")
    day = asof.isoformat()
    report = {
        "schema": "recommend_microstructure_operational_status.v1",
        "asof": day,
        "checked_at": now.isoformat(),
        "operational_start_asof": OPERATIONAL_START.isoformat(),
        "research_start_asof": trial.TRIAL_CONFIG["prospective_start_asof"],
        "due_at": datetime.combine(asof, DUE_KST, KST).isoformat(),
        "scope": "bounded_metadata_bindings_and_raw_presence_size_only",
        "raw_content_hashes_checked": False,
        "generator_sources_checked": False,
        "prospective_status": "not_checked",
        "effect_status": "not_evaluated",
        "deployable": False,
    }
    if asof < OPERATIONAL_START:
        return {
            **report,
            "status": "not_started",
            "reason": "before first installed morning run",
            "attention_required": False,
        }
    if asof == now.date() and now.time() < DUE_KST:
        previous = inspect_daily_status(
            now=now,
            capture_root=capture_root,
            snapshot_root=snapshot_root,
            trial_root=trial_root,
            _asof=asof - timedelta(days=1),
        )
        return {
            **report,
            "status": "waiting",
            "reason": "before operational completion deadline",
            "previous_due": previous,
            "attention_required": previous["attention_required"],
        }
    capture_root = (
        Path(capture_root)
        if capture_root is not None
        else ROOT / "data/microstructure/upbit"
    )
    snapshot_root = (
        Path(snapshot_root)
        if snapshot_root is not None
        else ROOT / "output/recommend_snapshots"
    )
    trial_root = (
        Path(trial_root)
        if trial_root is not None
        else ROOT / "output/recommend_microstructure_trials"
    )
    reader = _ReadSet()
    try:
        # Observe every data-root descendant: don't follow replaced/symlink dates.
        for root, components in (
            (capture_root, day.split("-")),
            (snapshot_root, [day]),
            (trial_root, [day, trial.TRIAL_ID]),
        ):
            reader.directory(root)
            for component in components:
                root = root / component
                reader.directory(root)
        capture_day = capture_root.joinpath(*day.split("-"))
        captures = []
        if reader.directory(capture_day):
            with os.scandir(capture_day) as entries:
                for entry in entries:
                    _require(len(captures) < 16, "too many daily capture entries")
                    path = Path(entry.path)
                    _require(
                        reader.directory(path), "unexpected non-directory capture entry"
                    )
                    captures.append(path)
        if not captures:
            status, reason = (
                "capture_missing",
                "no daily capture directory; execution not evidenced",
            )
        elif len(captures) != 1:
            status, reason = (
                "capture_ambiguous",
                "multiple daily captures; no automatic winner selection",
            )
        else:
            status, reason = _documents_status(
                reader,
                capture=captures[0],
                snapshot_path=snapshot_root / day / "open_r1.json",
                trial_dir=trial_root / day / trial.TRIAL_ID,
                day=day,
                now=now,
            )
        reader.unchanged()
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        IndexError,
        AttributeError,
        RuntimeError,
    ) as exc:
        status, reason = "evidence_invalid", f"{type(exc).__name__}: {exc}"
    return {
        **report,
        "status": status,
        "reason": reason,
        "attention_required": status
        not in {"complete_noop", "complete_changed", "complete_unavailable"},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("json", "text"), default="json")
    parser.add_argument(
        "--now", help="Aware ISO timestamp for read-only incident replay"
    )
    args = parser.parse_args(argv)
    try:
        report = inspect_daily_status(now=args.now)
    except (ValueError, TypeError, OSError) as exc:
        print(f"microstructure status probe error: {type(exc).__name__}: {exc}")
        return 2
    if args.format == "json":
        print(json.dumps(report, ensure_ascii=False, allow_nan=False, sort_keys=True))
    else:
        print(
            f"microstructure {report['asof']}: {report['status']} — {report['reason']} "
            "[metadata check only; raw hashes/prospective eligibility/performance not checked]"
        )
        if "previous_due" in report:
            previous = report["previous_due"]
            print(
                f"microstructure latest due {previous['asof']}: {previous['status']} — {previous['reason']}"
            )
    return 1 if report["attention_required"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Independent post-label L1 replay, immutable daily evidence and bounded refresh.

No pre-entry score is claimed. No fitting, dispatch, order or live selection.
Initialization must precede the fixed future cohort. Inspection never repairs.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import uuid
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from data.upbit_microstructure import iter_raw_records
from notifier.delivery_receipt import DeliveryReceiptError, read_delivery_receipt
from ops.artifact_provenance import (
    canonical_json_bytes,
    file_identity,
    strict_json_object,
)
from ops.recommendation_evidence import (
    EvidenceUnavailable,
    load_recommendation_evidence,
)
from signals import recommend_book_pressure as book
from signals import recommend_book_validation as policy
from signals import recommend_microstructure as raw
from signals import recommend_microstructure_trial as native
from signals import recommend_trade_shortlist_trial as trial
from signals.recommend_regime_replay import version_from_snapshot
from signals.recommend_score_labels import FORWARD_PROVENANCE_COHORT, _artifact_digest
from signals.recommend_snapshot import load_snapshot

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "output/recommend_book_validation"
KST = ZoneInfo("Asia/Seoul")
SCHEMA = "recommend_book_validation_report.v1"
DESIGN_SCHEMA = "recommend_book_validation_design.v1"
DAY_SCHEMA = "recommend_book_validation_day.v1"
SOURCE_NAMES = (
    "ops/recommend_book_validation.py",
    "signals/recommend_book_validation.py",
    "signals/recommend_book_top3.py",
    "signals/recommend_book_pressure.py",
    "signals/recommend_spread_diagnostics.py",
    "signals/recommend_experiment_eval.py",
    "ops/recommendation_evidence.py",
    "signals/recommend_score_labels.py",
    "signals/recommend_snapshot.py",
    "notifier/delivery_receipt.py",
    "signals/recommend_regime_replay.py",
    "signals/recommend_trade_shortlist_trial.py",
    "signals/recommend_microstructure_trial.py",
    "signals/recommend_microstructure.py",
    "data/upbit_microstructure.py",
    "ops/artifact_provenance.py",
    "ledger/portfolio_metrics.py",
    "ledger/path_quality.py",
)


def _sources():
    items = [file_identity(ROOT / p, root=ROOT) for p in SOURCE_NAMES]
    native._require(all(i["exists"] for i in items), "validation source missing")
    return items


def _same(a, b, reason):
    native._require(canonical_json_bytes(a) == canonical_json_bytes(b), reason)


def _checked(doc, schema):
    native._require(doc.get("schema") == schema, "validation schema mismatch")
    _same(
        doc,
        native._seal({k: v for k, v in doc.items() if k != "payload_sha256"}),
        "validation seal mismatch",
    )


def _path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _verify(items):
    for item in items:
        _same(
            file_identity(_path(item["path"]), root=ROOT),
            item,
            "validation input changed",
        )


def _identities(items):
    result = {}
    for i in items:
        key = str(_path(i["path"]).resolve())
        native._require(
            key not in result or result[key] == i, "mixed validation generations"
        )
        result[key] = i
    return list(result.values())


def _root(root):
    root = Path(root).absolute()
    resolved = root.resolve()
    native._require(
        resolved == ROOT.resolve() / "output/recommend_book_validation"
        or resolved.is_relative_to(ROOT.resolve() / "_workspace"),
        "validation output must use its own namespace or private workspace",
    )
    native._require(not root.is_symlink(), "validation root is symlink")
    return root


def read_design(root, now):
    with native._directory(_root(root), create=False) as directory:
        native._require(directory is not None, "validation design missing")
        design = native._read_document(directory, "design.json")
    native._require(design is not None, "validation design missing")
    _checked(design, DESIGN_SCHEMA)
    _same(design["config"], policy.CONFIG, "validation configuration changed")
    _same(design["generator_sources"], _sources(), "validation sources changed")
    frozen = native._time(design["frozen_at"])
    native._require(
        frozen <= now and frozen < datetime.combine(policy.START, time(), tzinfo=KST),
        "design was not frozen before cohort start",
    )
    _verify([design["reference_snapshot"]])
    reference = load_snapshot(
        _path(design["reference_snapshot"]["path"]), slot="open", ranking="R1"
    )
    native._require(
        design["r1_version"] is not None
        and version_from_snapshot(reference) == design["r1_version"],
        "R1 reference version changed",
    )
    return design


def initialize(reference_path, *, root=DEFAULT_ROOT, now=None):
    now = native._time(now or datetime.now(KST))
    native._require(
        now < datetime.combine(policy.START, time(), tzinfo=KST),
        "too late to initialize this cohort",
    )
    identity = file_identity(reference_path, root=ROOT)
    reference = load_snapshot(reference_path, slot="open", ranking="R1")
    version = version_from_snapshot(reference)
    native._require(
        version is not None and native._time(reference["created_at"]) <= now,
        "invalid reference snapshot",
    )
    design = native._seal(
        {
            "schema": DESIGN_SCHEMA,
            "config": policy.CONFIG,
            "frozen_at": now.isoformat(),
            "r1_version": version,
            "reference_snapshot": identity,
            "generator_sources": _sources(),
        }
    )
    _verify([identity, *design["generator_sources"]])
    native.write_new_trial_report(_root(root) / "design.json", design)
    _same(read_design(root, now), design, "design publication mismatch")
    return design


def extract_day(
    day,
    now,
    *,
    expected_version,
    shortlist_root=None,
    label_root=None,
    receipt_root=None,
    before_raw=None,
):
    """Real extraction; the caller alone decides whether this is a cohort date."""
    record = trial.read_trade_shortlist_record(
        shortlist_root or trial.DEFAULT_TRIAL_ROOT, day, now
    )
    if record["status"] != "committed":
        raise EvidenceUnavailable("native_record_" + record["status"])
    snapshot, feature = record["snapshot"], record["feature"]
    native._require(
        snapshot["asof"] == day and version_from_snapshot(snapshot) == expected_version,
        "R1 version/date changed within cohort",
    )
    if feature["feature_evidence_valid"] is not True:
        raise EvidenceUnavailable("capture_evidence_invalid")
    # Check availability metadata before spending the scarce raw budget. Do not
    # inspect outcome values or pass them to selection. Canonical label joining
    # still follows the outcome-blind plan below. Otherwise two permanently
    # unavailable early dates could starve every later date on every refresh.
    label_root = label_root or ROOT / "output/recommend_score_labels"
    receipt_root = receipt_root or ROOT / "output/recommend_receipts"
    label_path = Path(label_root) / day / "open_r1.json"
    receipt_path = Path(receipt_root) / day / "open_r1.json"
    readiness_inputs = [file_identity(p, root=ROOT) for p in (label_path, receipt_path)]
    if not readiness_inputs[0]["exists"]:
        raise EvidenceUnavailable("label_missing")
    try:
        receipt = read_delivery_receipt(snapshot, root=Path(receipt_root))
    except DeliveryReceiptError as exc:
        raise ValueError("receipt readiness check failed") from exc
    if receipt is None or receipt["delivery_ok"] is not True:
        raise EvidenceUnavailable("receipt_not_confirmed_success")
    native._require(not label_path.is_symlink(), "label is symlink")
    metadata = strict_json_object(label_path)
    native._require(
        metadata.get("label_payload_sha256") == _artifact_digest(metadata),
        "label metadata checksum mismatch",
    )
    if "round_trip_cost_fraction" not in metadata or "label_code" not in metadata:
        raise EvidenceUnavailable("legacy_label_contract")
    if metadata["artifact_status"] != "complete":
        raise EvidenceUnavailable("label_not_complete")
    if (
        metadata.get("provenance_cohort") != FORWARD_PROVENANCE_COHORT
        or metadata.get("forward_eligible") is not True
    ):
        raise EvidenceUnavailable("not_forward_observed")
    if max(native._time(metadata[k]) for k in ("path_window_end", "labeled_at")) > now:
        raise EvidenceUnavailable("label_not_available_as_of_now")
    del metadata  # No result values are used to create the plan.
    _verify(readiness_inputs)
    items = _identities(
        [
            *readiness_inputs,
            *record["trial_artifacts"],
            *record["source_inputs"],
            *feature["provenance"]["files"],
        ]
    )
    _verify(items)
    manifest = strict_json_object(_path(feature["provenance"]["files"][1]["path"]))
    native._require(
        manifest["subscription"]["orderbook_depth"]["requested"] == 1, "not depth1"
    )
    stream = manifest["streams"]["orderbook"]["artifact"]
    if before_raw:
        before_raw()
    validated = raw._validated_records(
        iter_raw_records(_path(stream["path"])),
        "orderbook",
        manifest,
        set(manifest["universe"]["markets"]),
    )
    quotes = book.endpoint_quotes(
        validated,
        cutoff=raw._ns(snapshot["decision_started_at"]),
        coins={r["coin"] for r in snapshot["universe"]},
    )
    micros = {r["coin"]: r for r in feature["rows"]}
    predictors = []
    for c in snapshot["universe"]:
        coin = c["coin"]
        if coin not in quotes:
            raise EvidenceUnavailable("missing_endpoint_quote")
        q = quotes[coin]
        _same(
            {k: q[k] for k in ("spread_fraction", "book_event_age_seconds")},
            {
                k: micros[coin]["quality_diagnostics"][k]
                for k in ("spread_fraction", "book_event_age_seconds")
            },
            "endpoint parity mismatch",
        )
        predictors.append(
            {
                "coin": coin,
                "rank": c["rank"],
                **{k: c["feature_values"].get(k) for k in ("f_log_qv", "f_atr_pct_14")},
                **q,
            }
        )
    plan = policy.fixed.select_top3(predictors)  # Before loading any outcomes.
    native._require(
        plan["control"] == [c["coin"] for c in snapshot["top3"]],
        "original R1 parity mismatch",
    )
    evidence = load_recommendation_evidence(
        _path(record["source_inputs"][0]["path"]),
        label_root=label_root,
        receipt_root=receipt_root,
        now=now,
    )
    label = evidence["label"]
    native._require(
        label["round_trip_cost_fraction"] == 0.0015
        and label["return_unit"] == "fraction",
        "cost/unit mismatch",
    )
    outcomes = [
        {
            "coin": r["coin"],
            "label_status": r["label_status"],
            **{k: r[k] for k in book.OUTCOMES},
        }
        for r in label["rows"]
    ]
    items = _identities([*items, *evidence["manifest"]["files"].values()])
    result = policy.evaluate_day(
        day,
        predictors,
        outcomes,
        entry_at=label["path_window_start"],
        end_at=label["path_window_end"],
    )
    _verify(items)
    return {
        "date": day,
        "predictors": predictors,
        "outcomes": outcomes,
        "result": result,
        "input_files": items,
        "r1_version": expected_version,
    }


def _read_day(directory, day, design, now):
    doc = native._read_document(directory, day + ".json")
    if doc is None:
        return None
    _checked(doc, DAY_SCHEMA)
    native._require(
        doc["date"] == day
        and doc["design_sha256"] == design["payload_sha256"]
        and doc["r1_version"] == design["r1_version"]
        and native._time(design["frozen_at"])
        <= native._time(doc["generated_at"])
        <= now,
        "daily validation identity/clock mismatch",
    )
    _verify(doc["input_files"])
    native._require(bool(doc["input_files"]), "daily inputs missing")
    result = doc["result"]
    native._require(
        native._time(result["end_at"]) <= native._time(doc["generated_at"]),
        "premature daily outcome",
    )
    computed = policy.evaluate_day(
        day,
        doc["predictors"],
        doc["outcomes"],
        entry_at=result["entry_at"],
        end_at=result["end_at"],
    )
    _same(computed, result, "cached selection/arithmetic mismatch")
    return doc


def _report(design, rows, daily, now):
    statuses = [r["status"] for r in rows]
    attention = any(s not in {"evaluated", "excluded"} for s in statuses)
    return native._seal(
        {
            "schema": SCHEMA,
            "generated_at": now.isoformat(),
            "asof": now.astimezone(KST).date().isoformat(),
            "design_sha256": design["payload_sha256"],
            "config": policy.CONFIG,
            "status": "waiting_for_first_outcome"
            if not rows
            else "incomplete"
            if attention
            else "evaluated",
            "attention_required": attention,
            "calendar": rows,
            "daily": daily,
            "summary": policy.summarize(daily),
            "deployable": False,
            "automatic_promotion": False,
            "prospective_pick_records": 0,
            "scope": policy.CONFIG["evaluation_kind"],
        }
    )


class _BudgetExhausted(Exception):
    pass


def _replace_report(directory, report):
    """Replace only the derived report through the pinned, no-follow directory."""
    name = ".report-" + uuid.uuid4().hex + ".json"
    native._publish_new(directory, name, report)
    try:
        directory[2]()
        os.replace(
            name, "report.json", src_dir_fd=directory[1], dst_dir_fd=directory[1]
        )
        os.fsync(directory[1])
        directory[2]()
    finally:
        try:
            os.unlink(name, dir_fd=directory[1])
        except FileNotFoundError:
            pass


def refresh(*, root=DEFAULT_ROOT, now=None):
    now = native._time(now or datetime.now(KST))
    root = _root(root)
    design = read_design(root, now)
    with native._directory(root, create=False) as directory:
        # Serialize only this new research namespace; never take the live lock.
        fd = os.open(
            ".refresh.lock",
            os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory[1],
        )
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            rows, daily, new_dates = [], [], 0

            def before_raw():
                nonlocal new_dates
                if new_dates >= policy.CONFIG["max_new_dates_per_run"]:
                    raise _BudgetExhausted()
                new_dates += 1

            for day in policy.due_dates(now):
                doc = _read_day(directory, day, design, now)
                if doc is None:
                    # Do not spend the raw parsing budget on absent labels.
                    if not (
                        ROOT / "output/recommend_score_labels" / day / "open_r1.json"
                    ).is_file():
                        rows.append(
                            {
                                "date": day,
                                "status": "pending",
                                "reason": "label_missing",
                            }
                        )
                        continue
                    try:
                        bundle = extract_day(
                            day,
                            now,
                            expected_version=design["r1_version"],
                            before_raw=before_raw,
                        )
                    except _BudgetExhausted:
                        rows.append(
                            {
                                "date": day,
                                "status": "deferred",
                                "reason": "per_run_budget",
                            }
                        )
                        continue
                    except EvidenceUnavailable as exc:
                        rows.append(
                            {"date": day, "status": "pending", "reason": str(exc)}
                        )
                        continue
                    doc = native._seal(
                        {
                            "schema": DAY_SCHEMA,
                            "generated_at": now.isoformat(),
                            "design_sha256": design["payload_sha256"],
                            **bundle,
                        }
                    )
                    native._publish_new(directory, day + ".json", doc)
                    _same(
                        _read_day(directory, day, design, now),
                        doc,
                        "daily publication mismatch",
                    )
                daily.append(doc["result"])
                rows.append(
                    {
                        "date": day,
                        "status": doc["result"]["status"],
                        "reason": doc["result"]["reason"],
                    }
                )
            _same(design, read_design(root, now), "design changed during refresh")
            report = _report(design, rows, daily, now)
            existing = native._read_document(directory, "report.json")
            if existing is not None:
                _checked(existing, SCHEMA)
            _replace_report(directory, report)
            directory[2]()
            return report
        finally:
            os.close(fd)


def inspect(*, root=DEFAULT_ROOT, now=None):
    now = native._time(now or datetime.now(KST))
    design = read_design(root, now)
    with native._directory(_root(root), create=False) as directory:
        report = native._read_document(directory, "report.json")
        native._require(report is not None, "validation report missing")
        _checked(report, SCHEMA)
        generated = native._time(report["generated_at"])
        native._require(
            generated <= now and now - generated <= timedelta(hours=6),
            "validation report stale/future",
        )
        if now.astimezone(KST).time().replace(tzinfo=None) >= time(10, 10):
            native._require(
                report["asof"] == now.astimezone(KST).date().isoformat(),
                "validation day overdue",
            )
        expected = policy.due_dates(generated)
        native._require(
            [r["date"] for r in report["calendar"]] == expected,
            "validation calendar mismatch",
        )
        daily = []
        for row in report["calendar"]:
            native._require(
                row["status"] in {"evaluated", "excluded", "pending", "deferred"},
                "unknown validation state",
            )
            doc = _read_day(directory, row["date"], design, now)
            if row["status"] in {"evaluated", "excluded"}:
                native._require(
                    doc is not None
                    and doc["result"]["status"] == row["status"]
                    and doc["result"]["reason"] == row["reason"],
                    "cached day missing/mismatch",
                )
                daily.append(doc["result"])
            else:
                native._require(doc is None, "report omits a cached day")
        _same(
            report,
            _report(design, report["calendar"], daily, generated),
            "validation aggregates mismatch",
        )
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--initialize", type=Path, metavar="REFERENCE_SNAPSHOT")
    mode.add_argument("--refresh", action="store_true")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)
    try:
        if args.initialize:
            initialize(args.initialize, root=args.root)
            print("book validation design frozen; no live changes")
            return 0
        report = refresh(root=args.root) if args.refresh else inspect(root=args.root)
        print(
            json.dumps(report, ensure_ascii=False, allow_nan=False)
            if args.format == "json"
            else f"book validation: {report['status']}; dates={report['summary']['paired_dates']}; "
            f"attention={report['attention_required']}; fixed-rule replay, NOT pre-entry records"
        )
        return 0 if args.refresh else int(report["attention_required"])
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"book validation unavailable: {type(exc).__name__}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""L1 pre-entry publication, mature evaluation and read-only review status.

Separate from the frozen post-label replay. No CLI clock override, training,
dispatch, automatic promotion or orders. Only this module's namespace is written.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
from datetime import datetime, timedelta, time as wall_time
from pathlib import Path

from data.upbit_microstructure import iter_raw_records
from ledger.path_quality import next_bar_boundary
from notifier.delivery_receipt import read_delivery_receipt, receipt_path
from ops import recommend_book_execution as execution
from ops import recommend_book_validation as replay
from ops.artifact_provenance import file_identity, strict_json_object
from ops.recommendation_evidence import (
    EvidenceUnavailable,
    load_recommendation_evidence,
)
from signals import recommend_book_readiness as policy
from signals import recommend_microstructure_trial as native
from signals import recommend_trade_shortlist_trial as trial
from signals.recommend_regime_replay import context_from_snapshot, version_from_snapshot
from signals.recommend_score_labels import path_window
from signals.recommend_snapshot import load_snapshot

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "output/recommend_book_forward"
KST = replay.KST
DESIGN = "recommend_book_forward_design.v1"
SCORE = "recommend_book_forward_score.v1"
COMMIT = "recommend_book_forward_commit.v1"
DAY = "recommend_book_forward_day.v1"
REPORT = "recommend_book_forward_report.v1"
SOURCES = sorted(
    set(replay.SOURCE_NAMES)
    | set(execution.delay.SOURCE_FILES)
    | {
        "ops/recommend_book_forward.py",
        "ops/recommend_book_execution.py",
        "signals/recommend_book_readiness.py",
        "scripts/daily_run_distribution.sh",
    }
)
require, same, checked = native._require, replay._same, replay._checked


def _now():
    return datetime.now(KST)


def _path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _root(value):
    path = Path(value).absolute()
    require(
        path.resolve() == ROOT.resolve() / "output/recommend_book_forward"
        or path.resolve().is_relative_to(ROOT.resolve() / "_workspace"),
        "forward output must use its own namespace or private workspace",
    )
    require(not path.is_symlink(), "forward root is symlink")
    return path


def _sources():
    items = [file_identity(ROOT / p, root=ROOT) for p in SOURCES]
    require(all(i["exists"] for i in items), "forward source missing")
    return items


def _verify(items):
    for item in items:
        same(
            file_identity(_path(item["path"]), root=ROOT), item, "forward input changed"
        )


def initialize(reference, *, root=DEFAULT_ROOT, now=None):
    now = native._time(now or _now())
    require(
        now.astimezone(KST).date() < policy.START,
        "too late to initialize forward cohort",
    )
    identity = file_identity(reference, root=ROOT)
    snapshot = load_snapshot(reference)
    require(native._time(snapshot["created_at"]) <= now, "future reference snapshot")
    version = version_from_snapshot(snapshot)
    require(version is not None, "unknown R1 version")
    doc = native._seal(
        {
            "schema": DESIGN,
            "frozen_at": now.isoformat(),
            "config": policy.CONFIG,
            "sources": _sources(),
            "reference": identity,
            "r1_version": version,
        }
    )
    _verify([identity])
    with native._directory(_root(root), create=True) as directory:
        native._publish_new(directory, "design.json", doc)
    return doc


def design(root, now):
    with native._directory(_root(root), create=False) as directory:
        require(directory is not None, "forward design missing")
        doc = native._read_document(directory, "design.json")
    require(doc is not None, "forward design missing")
    checked(doc, DESIGN)
    require(
        native._time(doc["frozen_at"]) <= now
        and native._time(doc["frozen_at"]).astimezone(KST).date() < policy.START,
        "invalid design clock",
    )
    same(doc["config"], policy.CONFIG, "forward configuration changed")
    same(doc["sources"], _sources(), "forward source changed")
    _verify([doc["reference"]])
    require(
        version_from_snapshot(load_snapshot(_path(doc["reference"]["path"])))
        == doc["r1_version"],
        "reference version mismatch",
    )
    return doc


def predictors(day, now, *, expected_version, shortlist_root=None, receipt_root=None):
    """Only decision-time raw quotes and morning artifacts; never reads labels."""
    record = trial.read_trade_shortlist_record(
        shortlist_root or trial.DEFAULT_TRIAL_ROOT, day, now
    )
    if record["status"] != "committed":
        raise EvidenceUnavailable("native_record_not_committed")
    snapshot, feature = record["snapshot"], record["feature"]
    require(
        snapshot["asof"] == day and version_from_snapshot(snapshot) == expected_version,
        "R1 version/date changed",
    )
    if feature["feature_evidence_valid"] is not True:
        raise EvidenceUnavailable("capture_evidence_invalid")
    receipt_root = receipt_root or ROOT / "output/recommend_receipts"
    receipt_identity = file_identity(
        receipt_path(snapshot, root=receipt_root), root=ROOT
    )
    receipt = read_delivery_receipt(snapshot, root=receipt_root)
    if receipt is None or receipt["delivery_ok"] is not True:
        raise EvidenceUnavailable("successful_receipt_missing")
    require(
        all(
            native._time(receipt[k]) <= now
            for k in ("sent_at", "attempted_at", "recorded_at")
        ),
        "future receipt",
    )
    require(
        native._time(snapshot["decision_completed_at"])
        <= native._time(receipt["attempted_at"]),
        "receipt before decision",
    )
    entry = max(path_window(day)[0], next_bar_boundary(receipt["sent_at"]))
    entry = native._time(entry)
    if now >= entry:
        raise EvidenceUnavailable("publication_deadline_passed_no_backfill")
    items = replay._identities(
        [
            *record["trial_artifacts"],
            *record["source_inputs"],
            *feature["provenance"]["files"],
            receipt_identity,
        ]
    )
    _verify(items)
    manifest = strict_json_object(_path(feature["provenance"]["files"][1]["path"]))
    require(manifest["subscription"]["orderbook_depth"]["requested"] == 1, "not depth1")
    records = replay.raw._validated_records(
        iter_raw_records(_path(manifest["streams"]["orderbook"]["artifact"]["path"])),
        "orderbook",
        manifest,
        set(manifest["universe"]["markets"]),
    )
    quotes = replay.book.endpoint_quotes(
        records,
        cutoff=replay.raw._ns(snapshot["decision_started_at"]),
        coins={r["coin"] for r in snapshot["universe"]},
    )
    micros = {r["coin"]: r for r in feature["rows"]}
    rows = []
    for candidate in snapshot["universe"]:
        coin = candidate["coin"]
        if coin not in quotes:
            raise EvidenceUnavailable("endpoint_quote_missing")
        quote = quotes[coin]
        for key in ("spread_fraction", "book_event_age_seconds"):
            same(
                quote[key],
                micros[coin]["quality_diagnostics"][key],
                "endpoint parity mismatch",
            )
        rows.append(
            {
                "coin": coin,
                "rank": candidate["rank"],
                **{
                    k: candidate["feature_values"].get(k)
                    for k in ("f_log_qv", "f_atr_pct_14")
                },
                **quote,
            }
        )
    selection = replay.policy.fixed.select_top3(rows)
    same(
        selection["control"],
        [r["coin"] for r in snapshot["top3"]],
        "original R1 mismatch",
    )
    _verify(items)
    return {
        "snapshot_id": snapshot["snapshot_id"],
        "snapshot_path": record["source_inputs"][0]["path"],
        "entry_at": entry.isoformat(),
        "predictors": rows,
        "selection": selection,
        "context": context_from_snapshot(snapshot),
        "input_files": items,
        "receipt_root": str(Path(receipt_root).absolute()),
    }


def read_score(day, *, root=DEFAULT_ROOT, now=None, frozen=None):
    now = native._time(now or _now())
    frozen = frozen or design(root, now)
    day = trial._day(day)
    require(day >= policy.START.isoformat(), "pre-cohort record")
    with native._directory(_root(root), ("scores", day), create=False) as directory:
        if directory is None:
            return {"status": "missing", "score": None, "artifacts": []}
        score = native._read_document(directory, "score.json")
        commit = native._read_document(directory, "commit.json")
        if score is None:
            require(commit is None, "orphan commit")
            return {"status": "missing", "score": None, "artifacts": []}
        artifacts = [
            file_identity(directory[0] / n, root=ROOT)
            for n in ("score.json", "commit.json")
        ]
        checked(score, SCORE)
        planned = native._time(score["planned_at"])
        require(
            score["date"] == day
            and score["design_sha256"] == frozen["payload_sha256"]
            and planned.astimezone(KST).date().isoformat() == day
            and planned <= now,
            "score identity/clock mismatch",
        )
        _verify(score["input_files"])
        snapshot = load_snapshot(_path(score["snapshot_path"]))
        require(
            snapshot["snapshot_id"] == score["snapshot_id"]
            and snapshot["asof"] == day
            and version_from_snapshot(snapshot) == frozen["r1_version"]
            and native._time(snapshot["decision_completed_at"]) <= planned,
            "score snapshot mismatch",
        )
        saved = {r["coin"]: r for r in score["predictors"]}
        require(
            set(saved) == {r["coin"] for r in snapshot["universe"]},
            "score universe mismatch",
        )
        for r in snapshot["universe"]:
            same(
                {k: saved[r["coin"]][k] for k in ("rank", "f_log_qv", "f_atr_pct_14")},
                {
                    "rank": r["rank"],
                    **{
                        k: r["feature_values"].get(k)
                        for k in ("f_log_qv", "f_atr_pct_14")
                    },
                },
                "score predictor snapshot mismatch",
            )
        same(score["context"], context_from_snapshot(snapshot), "context mismatch")
        same(
            score["selection"],
            replay.policy.fixed.select_top3(score["predictors"]),
            "score selection mismatch",
        )
        receipt = read_delivery_receipt(snapshot, root=Path(score["receipt_root"]))
        require(
            receipt is not None and receipt["delivery_ok"] is True,
            "score receipt missing",
        )
        require(
            all(
                native._time(receipt[k]) <= planned
                for k in ("sent_at", "attempted_at", "recorded_at")
            ),
            "score planned before receipt",
        )
        for path in (
            _path(score["snapshot_path"]),
            receipt_path(snapshot, root=Path(score["receipt_root"])),
        ):
            require(
                file_identity(path, root=ROOT) in score["input_files"],
                "score omits primary identity",
            )
        expected = native._time(
            max(path_window(day)[0], next_bar_boundary(receipt["sent_at"]))
        )
        require(expected == native._time(score["entry_at"]), "score entry mismatch")
        status = "uncertain"
        if commit is not None:
            checked(commit, COMMIT)
            durable = native._time(commit["score_durable_observed_at"])
            require(
                commit["date"] == day
                and commit["score_payload_sha256"] == score["payload_sha256"]
                and commit["clock_contract"] == trial.CLOCK_CONTRACT
                and planned <= durable <= now,
                "commit binding/clock mismatch",
            )
            status = "ready" if durable < expected else "late"
        _verify(artifacts)
        return {"status": status, "score": score, "artifacts": artifacts}


def record(*, root=DEFAULT_ROOT, shortlist_root=None, receipt_root=None, now_fn=_now):
    started = native._time(now_fn())
    frozen = design(root, started)
    day = started.astimezone(KST).date()
    if day < policy.START:
        return {"status": "not_started"}
    existing = read_score(day, root=root, now=started, frozen=frozen)
    if existing["status"] != "missing":
        return {"status": existing["status"], "reused": True}
    data = predictors(
        day.isoformat(),
        started,
        expected_version=frozen["r1_version"],
        shortlist_root=shortlist_root,
        receipt_root=receipt_root,
    )
    planned = native._time(now_fn())
    require(
        started <= planned and planned.astimezone(KST).date() == day,
        "planning clock moved",
    )
    same(frozen, design(root, planned), "design changed while planning")
    score = native._seal(
        {
            "schema": SCORE,
            "date": day.isoformat(),
            "planned_at": planned.isoformat(),
            "design_sha256": frozen["payload_sha256"],
            **data,
        }
    )
    with native._directory(
        _root(root), ("scores", day.isoformat()), create=True
    ) as directory:
        try:
            native._publish_new(directory, "score.json", score)
        except FileExistsError:
            return {"status": "uncertain", "reused": True}
        durable = native._time(now_fn())
        require(durable >= planned, "publication clock moved backwards")
        _verify(data["input_files"])
        same(frozen, design(root, durable), "design changed during publication")
        native._publish_new(
            directory,
            "commit.json",
            native._seal(
                {
                    "schema": COMMIT,
                    "date": day.isoformat(),
                    "score_payload_sha256": score["payload_sha256"],
                    "score_durable_observed_at": durable.isoformat(),
                    "clock_contract": trial.CLOCK_CONTRACT,
                }
            ),
        )
    return {
        "status": "ready" if durable < native._time(data["entry_at"]) else "late",
        "reused": False,
    }


def evaluate_day(recorded, *, now, label_root=None, db_path=None):
    score = recorded["score"]
    evidence = load_recommendation_evidence(
        _path(score["snapshot_path"]),
        label_root=label_root or ROOT / "output/recommend_score_labels",
        receipt_root=Path(score["receipt_root"]),
        now=now,
    )
    label = evidence["label"]
    require(
        label["round_trip_cost_fraction"] == policy.CONFIG["round_trip_cost_fraction"]
        and label["return_unit"] == "fraction",
        "forward cost/unit mismatch",
    )
    outcomes = [
        {
            "coin": r["coin"],
            "label_status": r["label_status"],
            **{k: r[k] for k in replay.book.OUTCOMES},
        }
        for r in label["rows"]
    ]
    result = replay.policy.evaluate_day(
        score["date"],
        score["predictors"],
        outcomes,
        entry_at=label["path_window_start"],
        end_at=label["path_window_end"],
    )
    same(result["selection"], score["selection"], "forward replay parity mismatch")
    require(
        native._time(result["entry_at"]) == native._time(score["entry_at"]),
        "canonical entry changed",
    )
    diagnostic = (
        execution.evaluate(
            score, evidence, db_path=db_path or ROOT / "data/upbit_15m.db", now=now
        )
        if result["status"] == "evaluated"
        else {"status": "excluded"}
    )
    items = replay._identities(
        [
            *recorded["artifacts"],
            *score["input_files"],
            *evidence["manifest"]["files"].values(),
        ]
    )
    _verify(items)
    return {
        "date": score["date"],
        "result": result,
        "context": score["context"],
        "outcomes": outcomes,
        "execution": diagnostic,
        "input_files": items,
        "label_identity": evidence["manifest"]["files"]["label"],
        "score_payload_sha256": score["payload_sha256"],
    }


def _read_day(directory, day, frozen, now, *, root):
    doc = native._read_document(directory, day + ".json")
    if doc is None:
        return None
    checked(doc, DAY)
    require(
        doc["date"] == day
        and doc["design_sha256"] == frozen["payload_sha256"]
        and native._time(doc["result"]["end_at"])
        <= native._time(doc["generated_at"])
        <= now,
        "daily forward identity/clock mismatch",
    )
    recorded = read_score(day, root=root, now=now, frozen=frozen)
    require(recorded["status"] == "ready", "daily score not pre-entry")
    score = recorded["score"]
    require(
        doc["score_payload_sha256"] == score["payload_sha256"],
        "daily score binding mismatch",
    )
    _verify(doc["input_files"])
    require(
        all(
            item in doc["input_files"]
            for item in recorded["artifacts"] + score["input_files"]
        ),
        "daily cache omits score inputs",
    )
    # Cheap canonical read: cached outcomes are not allowed to become their own
    # truth merely by re-sealing all derived means together.
    require(
        doc["label_identity"] in doc["input_files"],
        "daily canonical label identity missing",
    )
    label = strict_json_object(_path(doc["label_identity"]["path"]))
    require(
        label["snapshot_id"] == score["snapshot_id"]
        and label["asof"] == day
        and label["label_payload_sha256"] == replay._artifact_digest(label),
        "daily label mismatch",
    )
    same(
        doc["outcomes"],
        [
            {
                "coin": r["coin"],
                "label_status": r["label_status"],
                **{k: r[k] for k in replay.book.OUTCOMES},
            }
            for r in label["rows"]
        ],
        "cached outcomes differ from canonical labels",
    )
    same(doc["context"], score["context"], "daily context mismatch")
    r = doc["result"]
    same(
        r,
        replay.policy.evaluate_day(
            day,
            score["predictors"],
            doc["outcomes"],
            entry_at=r["entry_at"],
            end_at=r["end_at"],
        ),
        "daily arithmetic mismatch",
    )
    if r["status"] == "evaluated":
        e = doc["execution"]
        require(e["status"] == "evaluated", "execution missing")
        same(
            e["metrics"],
            execution.aggregate(e["paths"], score["selection"]),
            "execution arithmetic mismatch",
        )
        for arm in ("control", "challenger", "challenger_minus_control"):
            for metric in execution.METRICS:
                require(
                    abs(e["metrics"]["0"][arm][metric] - r["metrics"][arm][metric])
                    < 1e-12,
                    "zero-delay aggregate parity mismatch",
                )
    return doc


def _report(frozen, calendar, daily, now):
    attention = any(r["status"] not in {"evaluated", "excluded"} for r in calendar)
    reviewed = policy.review(daily)
    coverage = reviewed["summary"]["paired_dates"] / len(calendar) if calendar else None
    reviewed["observed_date_coverage"] = coverage
    if calendar:
        covered = coverage >= policy.CONFIG["min_available_date_fraction"]
        reviewed["checks"]["observed_date_coverage"] = covered
        if not covered:
            reviewed["warnings"].append("missing_dates_may_bias_available_sample")
            if reviewed["verdict"] == "review_candidate":
                reviewed.update(
                    verdict="continue_observing",
                    reason="insufficient_calendar_coverage",
                )
    return native._seal(
        {
            "schema": REPORT,
            "generated_at": now.isoformat(),
            "asof": now.astimezone(KST).date().isoformat(),
            "design_sha256": frozen["payload_sha256"],
            "calendar": calendar,
            "attention_required": attention,
            "review": reviewed,
            "automatic_promotion": False,
            "scope": "pre_entry_paper_not_actual_trades",
        }
    )


def refresh(*, root=DEFAULT_ROOT, now=None):
    now = native._time(now or _now())
    frozen = design(root, now)
    with native._directory(_root(root), create=False) as directory:
        fd = os.open(
            ".refresh.lock",
            os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory[1],
        )
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            calendar, daily, new = [], [], 0
            for day in policy.due_dates(now):
                doc = _read_day(directory, day, frozen, now, root=root)
                if doc is None:
                    recorded = read_score(day, root=root, now=now, frozen=frozen)
                    if recorded["status"] != "ready":
                        calendar.append({"date": day, "status": recorded["status"]})
                        continue
                    if new >= policy.CONFIG["max_new_dates_per_run"]:
                        calendar.append({"date": day, "status": "deferred"})
                        continue
                    try:
                        bundle = evaluate_day(recorded, now=now)
                    except EvidenceUnavailable:
                        calendar.append({"date": day, "status": "pending"})
                        continue
                    doc = native._seal(
                        {
                            "schema": DAY,
                            "generated_at": now.isoformat(),
                            "design_sha256": frozen["payload_sha256"],
                            **bundle,
                        }
                    )
                    native._publish_new(directory, day + ".json", doc)
                    _read_day(directory, day, frozen, now, root=root)
                    new += 1
                daily.append(doc)
                calendar.append({"date": day, "status": doc["result"]["status"]})
            same(frozen, design(root, now), "design changed during refresh")
            report = _report(frozen, calendar, daily, now)
            replay._replace_report(directory, report)
            return report
        finally:
            os.close(fd)


def inspect(*, root=DEFAULT_ROOT, now=None):
    now = native._time(now or _now())
    frozen = design(root, now)
    with native._directory(_root(root), create=False) as directory:
        report = native._read_document(directory, "report.json")
        require(report is not None, "forward report missing")
        checked(report, REPORT)
        generated = native._time(report["generated_at"])
        require(
            generated <= now and now - generated <= timedelta(hours=6),
            "forward report stale/future",
        )
        if now.astimezone(KST).time().replace(tzinfo=None) >= wall_time(10, 10):
            require(
                report["asof"] == now.astimezone(KST).date().isoformat(),
                "forward day overdue",
            )
        require(
            [r["date"] for r in report["calendar"]] == policy.due_dates(generated),
            "forward calendar mismatch",
        )
        daily = []
        for row in report["calendar"]:
            require(
                row["status"]
                in {
                    "evaluated",
                    "excluded",
                    "pending",
                    "deferred",
                    "missing",
                    "late",
                    "uncertain",
                },
                "unknown forward status",
            )
            doc = _read_day(directory, row["date"], frozen, now, root=root)
            if row["status"] in {"evaluated", "excluded"}:
                require(
                    doc is not None and doc["result"]["status"] == row["status"],
                    "forward day cache mismatch",
                )
                daily.append(doc)
            else:
                require(doc is None, "omitted forward day cache")
        same(
            report,
            _report(frozen, report["calendar"], daily, generated),
            "forward report arithmetic mismatch",
        )
    today = now.astimezone(KST).date()
    score_status = "not_due"
    if today >= policy.START and now.astimezone(KST).time().replace(
        tzinfo=None
    ) >= wall_time(9, 20):
        score_status = read_score(today, root=root, now=now, frozen=frozen)["status"]
    return {
        "report": report,
        "today_record": score_status,
        "attention_required": report["attention_required"]
        or score_status not in {"ready", "not_due"},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--initialize", type=Path)
    action.add_argument("--record", action="store_true")
    action.add_argument("--refresh", action="store_true")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--format", choices=("json", "text"), default="json")
    args = parser.parse_args(argv)
    try:
        if args.initialize:
            result = initialize(args.initialize, root=args.root)
        elif args.record:
            # Wait only for the independent collector's immutable morning files.
            # The shell imposes a 120s total bound; no detached process or retry after entry.
            deadline = time.monotonic() + 45
            while True:
                try:
                    result = record(root=args.root)
                    break
                except EvidenceUnavailable as exc:
                    if (
                        str(exc) != "native_record_not_committed"
                        or time.monotonic() >= deadline
                    ):
                        raise
                    time.sleep(1)
        elif args.refresh:
            result = refresh(root=args.root)
        else:
            result = inspect(root=args.root)
        print(
            json.dumps(result, ensure_ascii=False)
            if args.format == "json"
            else f"L1 forward: {result.get('status', result.get('report', result).get('review', {}).get('verdict', 'initialized'))}; attention={result.get('attention_required', False)}"
        )
        return int(
            result.get("attention_required", False)
            or result.get("status") in {"late", "uncertain"}
        )
    except Exception as exc:
        print(json.dumps({"status": "unavailable", "error_type": type(exc).__name__}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

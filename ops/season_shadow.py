"""Day-ahead conditional choices, joined to independently committed morning Top3s.

There is no sender, scorer, retraining, expiry date, or CLI backdating option.
The close can evaluate a choice, but cannot create yesterday's choice/commit.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from ops import recommend_regime_forward as forward
from ops import recommend_trial_review as review
from ops.artifact_provenance import atomic_write_json, file_identity, strict_json_object
from signals import recommend_microstructure_trial as native
from signals import recommend_regime_replay as policy
from signals.recommend_experiment_eval import _summary

ROOT = forward.ROOT
STORE = ROOT / "output/season_shadow"
START = date(2026, 11, 25)
STATES = (
    "broad:expanding",
    "broad:contracting",
    "narrow:expanding",
    "narrow:contracting",
)
SCHEMA = "r1_day_ahead_season.v1"


def now():
    return datetime.now(timezone.utc)


def check(ok, message):
    if not ok:
        raise ValueError(message)


def sources():
    return [
        file_identity(ROOT / rel, root=ROOT)
        for rel in (
            "ops/season_shadow.py",
            "signals/recommend_regime_replay.py",
            "ops/recommend_regime_forward.py",
            "signals/recommend_trade_shortlist_eval.py",
        )
    ]


def checked(doc, kind):
    check(
        doc is not None
        and doc.get("schema") == SCHEMA + "." + kind
        and doc
        == native._seal({k: v for k, v in doc.items() if k != "payload_sha256"}),
        f"invalid {kind} evidence",
    )
    return doc


def choice_table(history, cutoff, target):
    """Only the last N relevant records can enter either original selector."""
    cutoff = policy.aware(cutoff)
    known = sorted(
        (
            r
            for r in history
            if r["date"] < target and policy.aware(r["label_available_at"]) < cutoff
        ),
        key=lambda r: r["date"],
    )
    table = {}
    limit = policy.CONFIG["history_observations"]
    for version in sorted({r["version"] for r in known if r["version"] is not None}):
        rows = [r for r in known if r["version"] == version]
        table[version] = {}
        for state in STATES:
            matched = [r for r in rows if r["context"]["state"] == state]
            recent = {r["date"]: r for r in rows[-limit:] + matched[-limit:]}
            # Scenario query, not a claim about tomorrow's clock, BTC or coins.
            case = dict(
                date=target,
                version=version,
                context={"state": state},
                decision_at=cutoff.isoformat(),
                entry_at=(cutoff + timedelta(days=1)).isoformat(),
                predictors_available_at=[cutoff.isoformat()],
                outcomes=None,
                label_available_at=None,
                changed_picks=0,
            )
            table[version][state] = policy.selection_plans([*recent.values(), case])[
                -1
            ]["choices"]
    return table


def read_ticket(root, day):
    with native._directory(root, (day,), create=False) as directory:
        if directory is None:
            return None
        plan = native._read_document(directory, "plan.json")
        commit = native._read_document(directory, "commit.json")
    if plan is None and commit is None:
        return None
    plan = checked(plan, "plan")
    if commit is None:
        return {"state": "uncertain"}
    checked(commit, "commit")
    durable = policy.aware(commit["durable_at"])
    check(
        plan["target"] == commit["target"] == day
        and commit["plan_sha256"] == plan["payload_sha256"]
        and policy.aware(plan["cutoff"]) <= policy.aware(plan["prepared_at"]) <= durable
        and durable.astimezone(forward.KST).date()
        == date.fromisoformat(day) - timedelta(days=1)
        and plan["sources"] == sources()
        and plan["policy"] == policy.CONFIG
        and plan["unseen"] == "control",
        "invalid plan source, day or commit",
    )
    return {"state": "committed", "plan": plan, "durable": durable}


def prepare(root, evaluation, history, clock=now):
    prepared = policy.aware(clock())
    today = prepared.astimezone(forward.KST).date()
    target = (today + timedelta(days=1)).isoformat()
    cutoff = policy.aware(evaluation["generated_at"])
    check(
        cutoff.astimezone(forward.KST).date() == today and cutoff <= prepared,
        "cannot prepare from a stale/future evaluation",
    )
    existing = read_ticket(root, target)
    if existing:
        check(
            existing["state"] == "committed", "partial plan: never complete it on retry"
        )
        return target
    plan = native._seal(
        dict(
            schema=SCHEMA + ".plan",
            target=target,
            prepared_at=prepared.isoformat(),
            cutoff=cutoff.isoformat(),
            sources=sources(),
            policy=policy.CONFIG,
            unseen="control",
            native_evaluation_sha256=evaluation["payload_sha256"],
            table=choice_table(history, cutoff, target),
        )
    )
    with native._directory(root, (target,), create=True) as directory:
        native._publish_new(directory, "plan.json", plan)
        durable = policy.aware(clock())
        check(
            durable >= prepared and durable.astimezone(forward.KST).date() == today,
            "day/clock changed after durable plan",
        )
        native._publish_new(
            directory,
            "commit.json",
            native._seal(
                dict(
                    schema=SCHEMA + ".commit",
                    target=target,
                    plan_sha256=plan["payload_sha256"],
                    durable_at=durable.isoformat(),
                )
            ),
        )
    return target


def verify_native_pair(evaluation, audit, record, observed):
    day = record["date"]
    bound = forward._bound_record(
        evaluation, day, forward.shortlist.DEFAULT_TRIAL_ROOT, observed
    )
    snapshot = bound["snapshot"]
    receipt = forward.read_delivery_receipt(snapshot, root=forward.DEFAULT_RECEIPT_ROOT)
    path = forward.receipt_path(snapshot, root=forward.DEFAULT_RECEIPT_ROOT)
    check(
        receipt
        and receipt["delivery_ok"] is True
        and file_identity(path, root=ROOT)
        == audit["evidence_manifest"]["files"]["receipt"],
        "canonical receipt mismatch",
    )
    entry = max(
        forward.path_window(day)[0], forward.next_bar_boundary(receipt["sent_at"])
    )
    check(
        snapshot["snapshot_id"] == audit["snapshot_id"]
        and record["version"] == policy.version_from_snapshot(snapshot)
        and record["context"] == policy.context_from_snapshot(snapshot)
        and bound["plan"]["status"] == "planned"
        and all(
            bound["plan"][key] == audit[key]
            for key in ("control_top3", "challenger_top3")
        )
        and bound["commit"]["score_durable_observed_at"]
        == audit["score_durable_observed_at"]
        and policy.aware(entry) == policy.aware(record["entry_at"]),
        "morning evidence mismatch",
    )


def score_day(ticket, audit, record, history):
    # Native evaluation rechecks both frozen Top3s and labels. Its history
    # record derives version and breadth/turnover from that very snapshot.
    entry = policy.aware(record["entry_at"])
    receipt = strict_json_object(
        native._path(audit["evidence_manifest"]["files"]["receipt"]["path"])
    )
    check(
        receipt["delivery_ok"] is True
        and receipt["snapshot_id"] == audit["snapshot_id"]
        and receipt["slot"] == "open"
        and receipt["asof"] == record["date"],
        "receipt is not for this decision",
    )
    latest = max(
        ticket["durable"],
        policy.aware(audit["score_durable_observed_at"]),
        policy.aware(receipt["recorded_at"]),
        *map(policy.aware, record["predictors_available_at"]),
    )
    if latest >= entry:
        return {"status": "late"}
    state = record["context"]["state"]
    choices = ticket["plan"]["table"].get(record["version"], {}).get(state, {})
    arms = {
        name: choices[name]["arm"] if choices else "control" for name in policy.POLICIES
    }
    check(
        arms["fixed_r1"] == "control"
        and set(arms.values()) <= {"control", "challenger"},
        "invalid policy arm",
    )
    # Independent check against the actual snapshot and the original rule,
    # never choose from today's returns or from a newly revised historical label.
    past = [
        r
        for r in history
        if r["date"] < record["date"]
        and r["version"] == record["version"]
        and policy.aware(r["label_available_at"])
        < policy.aware(ticket["plan"]["cutoff"])
    ]
    past.sort(key=lambda r: r["date"])
    limit = policy.CONFIG["history_observations"]
    relevant = {
        r["date"]: r
        for r in past[-limit:]
        + [r for r in past if r["context"]["state"] == state][-limit:]
    }
    observation = {
        **record,
        "decision_at": latest.isoformat(),
        "outcomes": None,
        "label_available_at": None,
    }
    expected = policy.selection_plans([*relevant.values(), observation])[-1]["choices"]
    check(
        arms == {name: c["arm"] for name, c in expected.items()}
        and (not choices or choices == expected),
        "precommitted choice differs from verified history",
    )
    selected = {name: audit[arm + "_top3"] for name, arm in arms.items()}
    changes = {
        name: len(set(coins) - set(audit["control_top3"]))
        for name, coins in selected.items()
    }
    return dict(
        status="comparable",
        context=state,
        known_before_entry_at=latest.isoformat(),
        snapshot_id=audit["snapshot_id"],
        plan_sha256=ticket["plan"]["payload_sha256"],
        selected=selected,
        changed=changes,
        arms=arms,
        values={name: record["outcomes"][arm] for name, arm in arms.items()},
    )


def summarize(days):
    paired = [row for row in days if row["status"] == "comparable"]
    result = {}
    for name in policy.POLICIES:

        def stats(rows):
            if not rows:
                return None
            differences = np.array(
                [
                    policy._vector(r["values"][name])
                    - policy._vector(r["values"]["fixed_r1"])
                    for r in rows
                ]
            )
            return _summary(differences, 1000, 42)

        changed = [row for row in paired if row["changed"][name]]
        result[name] = dict(
            paired_dates=len(paired),
            changed_dates=len(changed),
            changed_picks=sum(r["changed"][name] for r in changed),
            difference_all_dates=stats(paired),
            difference_changed_dates=stats(changed),
        )
    return result


def native_evaluation(review_path, observed):
    """A different legacy trial timing out cannot indefinitely starve this one."""
    today = observed.astimezone(forward.KST).date()
    source = file_identity(review_path, root=ROOT)
    if source["exists"]:
        report = strict_json_object(review_path)
        review._check_document(report, review.SCHEMA)
        through = date.fromisoformat(report["through_date"])
        check(through < today, "future close review")
        if through == today - timedelta(days=1):
            check(
                report["deployable"] is False
                and report["generator_sources"] == review._sources(),
                "invalid close review sources",
            )
            return report["evaluations"]["shortlist"], source
    # Only missing/stale reports are replaceable. Do not hide corruption.
    return forward.evaluator.evaluate_trade_shortlist_trials(now=observed), None


def refresh(*, root=STORE, review_path=review.DEFAULT_OUTPUT):
    observed = now()
    today = observed.astimezone(forward.KST).date()
    evaluation, source = native_evaluation(review_path, observed)
    cutoff = forward._evaluation_contract(evaluation, observed)
    check(cutoff.astimezone(forward.KST).date() == today, "native evaluation is stale")
    records = forward.history_records(evaluation, before_date=today.isoformat())
    check(
        source is None or source == file_identity(review_path, root=ROOT),
        "close evidence changed",
    )
    target = prepare(root, evaluation, records)
    by_day = {r["date"]: r for r in records}
    audits = {a["date"]: a for a in evaluation["dates"]}
    days = []
    for index in range((today - START).days):
        day = (START + timedelta(days=index)).isoformat()
        try:
            ticket = read_ticket(root, day)
            audit = audits.get(day, {})
            if not ticket or ticket["state"] != "committed":
                row = {"status": "missing_plan" if not ticket else "uncertain"}
            elif day not in by_day or audit.get("status") != "prospective_comparable":
                row = {
                    "status": "unavailable",
                    "reason": audit.get("reason")
                    or audit.get("status")
                    or "no_native_record",
                }
            else:
                verify_native_pair(evaluation, audit, by_day[day], observed)
                row = score_day(ticket, audit, by_day[day], records)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            row = {"status": "invalid_evidence", "reason": type(exc).__name__}
        days.append({"date": day, **row})
    result = native._seal(
        dict(
            schema=SCHEMA + ".review",
            asof=today.isoformat(),
            next_plan=target,
            automatic_promotion=False,
            deployable=False,
            scope="day_ahead_policy_and_native_pre_entry_picks_not_live_trades",
            source=source,
            native_evaluation_sha256=evaluation["payload_sha256"],
            status_counts=dict(Counter(r["status"] for r in days)),
            days=days,
            policies=summarize(days),
        )
    )
    check(
        source is None or source == file_identity(review_path, root=ROOT),
        "close review changed before publication",
    )
    atomic_write_json(Path(root) / "review.json", result)
    return result


def inspect(root=STORE):
    today = now().astimezone(forward.KST).date()
    result = checked(strict_json_object(Path(root) / "review.json"), "review")
    target = (today + timedelta(days=1)).isoformat()
    ticket = read_ticket(root, target)
    check(
        result["asof"] == today.isoformat()
        and result["next_plan"] == target
        and ticket
        and ticket["state"] == "committed"
        and result["deployable"] is False
        and result["automatic_promotion"] is False,
        "missing/stale next-day season plan or review",
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="read only: check the review and tomorrow's plan",
    )
    result = inspect() if parser.parse_args().check else refresh()
    print(
        f"day-ahead season: next={result['next_plan']} closed={result['status_counts']} no promotion"
    )

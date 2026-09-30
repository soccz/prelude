"""Post-label trial review, separate from immutable morning predictions.

Refresh reads the native evidence chains and atomically replaces only its own
derived report. The default probe never evaluates, fits, sends, or selects.
Performance is descriptive: successful evaluation is not successful trading.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from ops.artifact_provenance import atomic_write_json, file_identity, strict_json_object
from signals import recommend_microstructure_trial as boundary
from signals import recommend_trade_shortlist_eval as shortlist
from signals import recommend_regime_replay as regime
from ops import recommend_regime_forward as forward

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "output/recommend_trial_review.json"
SCHEMA = "recommend_trial_review.v1"
KST = ZoneInfo("Asia/Seoul")
DUE = time(10, 25)  # Initial close grace; tune from observed close runtime.


def _sources():
    paths = set(boundary.GENERATOR_SOURCES) | set(shortlist.GENERATOR_SOURCES)
    paths.update({"ops/recommend_trial_review.py", "signals/recommend_regime_replay.py",
                  "signals/recommend_experiment_eval.py"})
    paths.update(forward.GENERATOR_SOURCES)
    return [file_identity(ROOT / path, root=ROOT) for path in sorted(paths)]


def _check_document(document, schema):
    boundary._require(document.get("schema") == schema, "review schema mismatch")
    expected = boundary._seal(
        {key: value for key, value in document.items() if key != "payload_sha256"}
    )
    boundary._require(document == expected, "review payload checksum mismatch")


def summarize_evaluation(evaluation, through_date):
    """Keep historical exclusions visible; never count pending as zero return."""
    primary = evaluation["primary_paired"]
    rows = evaluation["dates"]
    latest = [row for row in rows if row["date"] == through_date]
    launch = evaluation["coverage"]["prospective_start_asof"]
    latest_status = (
        "not_started"
        if through_date < launch
        else latest[0]["status"] if len(latest) == 1 else "missing"
    )
    return {
        "trial_id": evaluation["trial_id"],
        "latest_due_date": through_date,
        "latest_due_status": latest_status,
        "evaluation_ready": latest_status in {"prospective_comparable", "not_started"},
        "paired_dates": primary["n_dates"],
        "changed_dates": primary.get("n_changed_dates", 0),
        "no_op_dates": primary.get("n_no_op_dates", 0),
        "changed_picks": primary.get("changed_picks", 0),
        "picks_per_arm": primary.get("picks_per_arm", 0),
        "metrics": primary["metrics"],
        "historical_exclusions": [
            {key: row.get(key) for key in ("date", "status", "reason")}
            for row in rows
            if launch <= row["date"] <= through_date
            and row["status"] != "prospective_comparable"
        ],
        "awaiting_outcomes": [
            row["date"]
            for row in rows
            if row["date"] > through_date and row["status"] == "pending"
        ],
        "effect_status": evaluation["effect_status"],
        "deployable": False,
    }


def selection_diagnostics(evaluation):
    """Explain frozen choices, without searching for a winning switching rule.

    Regimes come from the actual predecision snapshot. Swaps are pick-weighted
    diagnostics, not the primary equally weighted daily effect. Availability uses
    the label completion timestamp, not just a date less than the decision date.
    """
    audits = {row["date"]: row for row in evaluation["dates"]}
    records, added, removed = [], [], []
    for daily in evaluation["primary_paired"].get("daily", []):
        audit = audits[daily["date"]]
        files = audit["evidence_manifest"]["files"]
        snapshot = strict_json_object(boundary._path(files["snapshot"]["path"]))
        label = strict_json_object(boundary._path(files["label"]["path"]))
        records.append(
            {
                **daily,
                "regime": snapshot.get("btc_regime") or "unknown",
                "decision_at": snapshot["decision_started_at"],
                "available_at": max(
                    boundary._time(label[key])
                    for key in ("path_window_end", "labeled_at")
                ),
            }
        )
        by_coin = {row["coin"]: row for row in label["rows"]}
        control, challenger = (
            set(audit[f"{arm}_top3"]) for arm in ("control", "challenger")
        )
        for destination, coins in (
            (added, challenger - control),
            (removed, control - challenger),
        ):
            destination.extend(by_coin[coin] for coin in sorted(coins))
    regimes = {}
    for regime_name in sorted({row["regime"] for row in records}):
        rows = [row for row in records if row["regime"] == regime_name]
        regimes[regime_name] = {
            "n_dates": len(rows),
            "means": {
                arm: {
                    metric: sum(row[arm][metric] for row in rows) / len(rows)
                    for metric in evaluation["metrics"]
                }
                for arm in ("control", "challenger", "challenger_minus_control")
            },
        }
    availability = []
    for row in records:
        prior = [
            past
            for past in records
            if past["date"] < row["date"]
            and past["available_at"] < boundary._time(row["decision_at"])
        ]
        availability.append(
            {
                "date": row["date"],
                "eligible_prior_dates": len(prior),
                "same_regime_prior_dates": sum(
                    past["regime"] == row["regime"] for past in prior
                ),
                "latest_eligible_date": max(
                    (past["date"] for past in prior), default=None
                ),
            }
        )
    return {
        "scope": "descriptive_only_not_a_test_of_adaptive_switching",
        "switching_status": "not_tested_no_policy_selected",
        "regime_source": "frozen_predecision_btc_regime_not_recomputed",
        "regime_date_counts": dict(
            sorted(Counter(row["regime"] for row in records).items())
        ),
        "by_regime": regimes,
        "past_only_availability": availability,
        "availability_rule": "max(path_window_end,labeled_at) < decision_started_at; no same-day outcomes",
        "swaps": {
            "weighting": "pick_weighted_diagnostic_not_primary_daily_effect",
            **{
                name: {
                    "n_picks": len(rows),
                    "means": {
                        key: (
                            sum(float(row[key]) for row in rows) / len(rows)
                            if rows
                            else None
                        )
                        for key in ("up10", "dn5", "eod_return_net", "mae")
                    },
                }
                for name, rows in (("added", added), ("removed", removed))
            },
        },
        "automatic_promotion": False,
    }


def adaptive_replay(evaluation, *, n_boot=1000):
    """Adapt validated native common evidence without changing its cohort/picks."""
    audits = {row["date"]: row for row in evaluation["dates"]}
    records = []
    for daily in evaluation["primary_paired"].get("daily", []):
        audit = audits[daily["date"]]
        files = audit["evidence_manifest"]["files"]
        snapshot = strict_json_object(boundary._path(files["snapshot"]["path"]))
        label = strict_json_object(boundary._path(files["label"]["path"]))
        predictors = [snapshot["created_at"], snapshot["decision_completed_at"],
                      audit["score_durable_observed_at"]]
        records.append({
            "date": daily["date"], "version": regime.version_from_snapshot(snapshot),
            "context": regime.context_from_snapshot(snapshot),
            "decision_at": max(predictors, key=regime.aware),
            "predictors_available_at": predictors,
            "entry_at": audit["canonical_execution_start_at"],
            "label_available_at": max(label["path_window_end"], label["labeled_at"], key=regime.aware),
            "outcomes": {arm: daily[arm] for arm in ("control", "challenger")},
            "changed_picks": daily["changed_picks"],
        })
    result = regime.replay(records, n_boot=n_boot)
    result["native_trial_id"] = evaluation["trial_id"]
    result["native_excluded_dates"] = [
        {key: row.get(key) for key in ("date", "status", "reason")}
        for row in evaluation["dates"] if row["status"] != "prospective_comparable"
    ]
    return result


def build_review(*, now=None, n_boot=1000):
    observed = boundary._time(datetime.now(timezone.utc) if now is None else now)
    through = (observed.astimezone(KST).date() - timedelta(days=1)).isoformat()
    before = _sources()
    evaluations = {
        "boundary": boundary.evaluate_microstructure_trials(
            now=observed, n_boot=n_boot
        ),
        "shortlist": shortlist.evaluate_trade_shortlist_trials(
            now=observed, n_boot=n_boot
        ),
    }
    diagnostics = {
        name: selection_diagnostics(report) for name, report in evaluations.items()
    }
    adaptive = adaptive_replay(evaluations["shortlist"], n_boot=n_boot)
    prospective = forward.evaluate_forward(evaluations["shortlist"], now=observed, n_boot=n_boot)
    boundary._require(before == _sources(), "review sources changed during evaluation")
    # The two evaluators read at different times. Recheck their common immutable
    # evidence once before publishing, so a mixed generation cannot appear fresh.
    inputs = list(prospective["input_files"])
    for evaluation in evaluations.values():
        inputs.extend(evaluation["trial_inputs"] + evaluation["evidence_inputs"])
    forward._verify(inputs)
    summary = {
        name: summarize_evaluation(report, through)
        for name, report in evaluations.items()
    }
    return boundary._seal(
        {
            "schema": SCHEMA,
            "generated_at": observed.isoformat(),
            "through_date": through,
            "generator_sources": before,
            "status": (
                "evaluated"
                if all(row["evaluation_ready"] for row in summary.values())
                and _forward_ready(prospective, through)
                else "incomplete"
            ),
            "deployable": False,
            "automatic_promotion": False,
            "scope": "post_label_descriptive_review_no_live_selection",
            "summary": summary,
            "selection_diagnostics": diagnostics,
            "adaptive_replay": adaptive,
            "forward_evaluation": prospective,
            "evaluations": evaluations,
        }
    )


def _forward_ready(report, through):
    if through < forward.START.isoformat():
        return True
    latest = [row for row in report["dates"] if row["date"] == through]
    # A missed write-once decision cannot be repaired. Keep old gaps in the
    # audit, but do not turn one missed day into an unrecoverable daily alarm.
    return len(latest) == 1 and all(row["status"] == "prospective_comparable" or (
        row["record_status"] == "committed" and row["status"] == "unavailable" and row["reason"] is None
    ) for row in latest)


def inspect_review(path=DEFAULT_OUTPUT, *, now=None):
    """Bounded local report probe, not a rehash of raw capture history."""
    observed = boundary._time(
        datetime.now(timezone.utc) if now is None else now
    ).astimezone(KST)
    due_day = (
        observed.date()
        if observed.time() >= DUE
        else observed.date() - timedelta(days=1)
    )
    expected = (due_day - timedelta(days=1)).isoformat()
    result = {
        "checked_at": observed.isoformat(),
        "expected_through_date": expected,
        "scope": "report_integrity_freshness_and_maturity_not_raw_reverification",
        "deployable": False,
        "attention_required": True,
    }
    try:
        path = Path(path)
        boundary._require(not path.is_symlink(), "review report is symlink")
        before = file_identity(path, root=ROOT)
        report = strict_json_object(path)
        _check_document(report, SCHEMA)
        generated = boundary._time(report["generated_at"]).astimezone(KST)
        boundary._require(generated <= observed, "future review")
        boundary._require(generated.date() >= due_day, "stale review")
        through = (generated.date() - timedelta(days=1)).isoformat()
        boundary._require(
            report["through_date"] == through and through >= expected,
            "review date mismatch",
        )
        boundary._require(
            report["generator_sources"] == _sources(), "review code changed"
        )
        boundary._require(
            report["deployable"] is False and report["automatic_promotion"] is False,
            "review cannot promote",
        )
        boundary._require(
            set(report["evaluations"]) == {"boundary", "shortlist"},
            "missing trial evaluation",
        )
        expected_summary = {}
        for name, evaluation in report["evaluations"].items():
            module = boundary if name == "boundary" else shortlist
            _check_document(evaluation, module.EVALUATION_SCHEMA)
            boundary._require(
                evaluation["generated_at"] == report["generated_at"]
                and evaluation["trial_id"] == module.TRIAL_ID
                and evaluation["config"] == module.TRIAL_CONFIG
                and evaluation["generator_sources"] == module._generator_sources()
                and evaluation["inputs_unchanged"] is True
                and evaluation["deployable"] is False
                and evaluation["automatic_promotion"] is False,
                "evaluation contract mismatch",
            )
            expected_summary[name] = summarize_evaluation(evaluation, through)
        boundary._require(
            report["summary"] == expected_summary, "review summary mismatch"
        )
        adaptive = report["adaptive_replay"]
        boundary._require(
            adaptive["config"] == regime.CONFIG
            and adaptive["deployable"] is False
            and adaptive["automatic_promotion"] is False
            and adaptive["is_untouched_holdout"] is False
            and adaptive["prospective_policy_records"] == 0
            and set(adaptive["policies"]) == set(regime.POLICIES),
            "adaptive replay contract mismatch",
        )
        prospective = report["forward_evaluation"]
        expected_forward_dates = [
            (forward.START + timedelta(days=i)).isoformat()
            for i in range(max(0, (generated.date() - forward.START).days + 1))
        ]
        boundary._require(prospective["config"] == forward.CONFIG
                          and prospective["deployable"] is False
                          and prospective["automatic_promotion"] is False
                          and [row["date"] for row in prospective["dates"]] == expected_forward_dates
                          and set(prospective["policies"]) == set(regime.POLICIES)
                          and prospective["paired_dates"] == sum(row["status"] == "prospective_comparable" for row in prospective["dates"]),
                          "forward evaluation contract mismatch")
        ready = all(row["evaluation_ready"] for row in expected_summary.values()) and _forward_ready(prospective, through)
        boundary._require(
            report["status"] == ("evaluated" if ready else "incomplete"),
            "review status mismatch",
        )
        boundary._require(
            before == file_identity(path, root=ROOT), "review changed during check"
        )
        return {
            **result,
            "status": report["status"],
            "generated_at": report["generated_at"],
            "through_date": through,
            "summary": expected_summary,
            "forward": {
                "prospective_policy_records": prospective["prospective_policy_records"],
                "paired_dates": prospective["paired_dates"],
                "changed_picks": {name: value["changed_picks"] for name, value in prospective["policies"].items()},
                "historical_exclusions": [
                    {key: row[key] for key in ("date", "status", "reason")}
                    for row in prospective["dates"]
                    if row["date"] <= through and row["status"] != "prospective_comparable"
                ],
            },
            "attention_required": not ready,
            "reason": None if ready else "latest mature outcomes unavailable",
        }
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, IndexError) as exc:
        return {
            **result,
            "status": "unavailable",
            "reason": f"{type(exc).__name__}: {exc}",
        }


def format_review(report):
    lines = [
        f"trial review: {report['status']} — {report.get('reason') or 'descriptive only; no promotion'}"
    ]
    for name, row in report.get("summary", {}).items():
        lines.append(
            f"  {name}: through={row['latest_due_date']} maturity={row['latest_due_status']} "
            f"paired={row['paired_dates']} changed={row['changed_dates']} "
            f"noop={row['no_op_dates']} changed_picks={row['changed_picks']}"
        )
        metrics = row["metrics"]
        if metrics:
            control, challenger = (
                metrics[arm]["mean"] for arm in ("control", "challenger")
            )
            lines.append(
                f"    net_eod={control['eod_return_net']:.4%}->{challenger['eod_return_net']:.4%} "
                f"dn5={control['dn5']:.2%}->{challenger['dn5']:.2%} "
                f"up10={control['up10']:.2%}->{challenger['up10']:.2%}"
            )
    if "forward" in report:
        prospective = report["forward"]
        lines.append(f"  regime forward: ready={prospective['prospective_policy_records']} "
                     f"paired={prospective['paired_dates']} changed_picks={prospective['changed_picks']} "
                     f"historical_exclusions={len(prospective['historical_exclusions'])}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="evaluate and atomically replace only this derived report",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)
    try:
        if args.refresh:
            # Only our own derived generation may be replaced. A typo in
            # --output must never overwrite a snapshot or morning trial report.
            boundary._require(not args.output.is_symlink(), "review output is symlink")
            if args.output.exists():
                _check_document(strict_json_object(args.output), SCHEMA)
            report = build_review()
            atomic_write_json(args.output, report)
        report = inspect_review(args.output)
        print(
            json.dumps(report, ensure_ascii=False, allow_nan=False)
            if args.format == "json"
            else format_review(report)
        )
        return int(report["attention_required"])
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"trial review failed: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

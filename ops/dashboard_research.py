"""Bounded, public-safe projection of the post-label research review.

This reads the existing integrity/freshness probe and checks saved forward
aggregates against its same-generation derived daily rows. It never evaluates
raw data, repairs evidence, chooses a policy, trains, sends, or promotes a model.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import sys
from datetime import datetime, timedelta

from ops.artifact_provenance import file_identity, strict_json_object, strict_json_object_bytes
from ops.dashboard_current import KST, ROOT, _day, _require, _timestamp

SCHEMA = "prelude_dashboard_research.v2"
LEGACY_SCHEMA = "prelude_dashboard_research.v1"
METRICS = ("eod_return_net", "dn5", "up10")
POLICIES = ("fixed_r1", "recent", "same_context")
STATES = {"evaluated", "incomplete", "unavailable", "historical_not_observed"}


def read_review(*, now, path=None):
    """Bind the native probe and extra fields to one unchanged report, read-only."""
    from ops import recommend_trial_review as review

    path = review.DEFAULT_OUTPUT if path is None else path
    before = file_identity(path, root=ROOT)
    source = review.inspect_review(path, now=now)
    if source["status"] not in {"evaluated", "incomplete"}:
        return source
    report = strict_json_object(path)
    _require(before == file_identity(path, root=ROOT))
    _require(all(report[key] == source[key] for key in (
        "status", "generated_at", "through_date", "summary", "deployable",
    )))
    prospective = report["forward_evaluation"]
    policies = _checked_forward(prospective, report["evaluations"]["shortlist"],
                                generated_at=report["generated_at"], through=report["through_date"])
    counters = {
        "prospective_policy_records": prospective["prospective_policy_records"],
        "paired_dates": prospective["paired_dates"],
        "changed_picks": {name: row["changed_picks"] for name, row in prospective["policies"].items()},
        "historical_exclusions": [
            {key: row[key] for key in ("date", "status", "reason")}
            for row in prospective["dates"]
            if row["date"] <= report["through_date"] and row["status"] != "prospective_comparable"
        ],
    }
    _require(source["forward"] == counters)
    return {**source, "forward": {**counters, "policies": policies}}


def _checked_forward(report, native, *, generated_at, through):
    """Check only stored derived rows; never reopen raw evidence or reselect."""
    import numpy as np

    from ops import recommend_regime_forward as forward
    from signals.recommend_experiment_eval import METRICS as native_metrics, _summary

    _require(report["schema"] == "recommend_regime_forward_evaluation.v1"
             and report["trial_id"] == forward.TRIAL_ID and report["config"] == forward.CONFIG
             and report["scope"] == "pre_entry_frozen_policies_on_canonical_net_pick_proxies_not_portfolio"
             and report["deployable"] is False and report["automatic_promotion"] is False)
    n_boot, seed = report["n_boot"], report["seed"]
    _count(n_boot)
    _count(seed)
    _require(n_boot > 0)
    dates, daily = report["dates"], report["daily"]
    generated = datetime.fromisoformat(generated_at).astimezone(KST).date()
    expected = [(forward.START + timedelta(days=i)).isoformat()
                for i in range(max(0, (generated - forward.START).days + 1))]
    _require([row["date"] for row in dates] == expected)
    comparable = [row for row in dates if row["status"] == "prospective_comparable"]
    _require([row["date"] for row in daily] == [row["date"] for row in comparable])
    for key, expected_count in (
        ("paired_dates", len(daily)),
        ("committed_policy_records", sum(row["record_status"] == "committed" for row in dates)),
        ("prospective_policy_records", sum(row["eligibility"] == "ready" for row in dates)),
    ):
        _count(report[key])
        _require(report[key] == expected_count)
    native_daily = native["primary_paired"].get("daily", [])
    originals = {row["date"]: row for row in native_daily}
    _require(len(originals) == len(native_daily))
    _require(set(report["policies"]) == set(POLICIES))
    for row, audit in zip(daily, comparable, strict=True):
        _require(forward.START.isoformat() <= _day(row["date"]) <= through
                 and audit["record_status"] == "committed" and audit["eligibility"] == "ready"
                 and audit["reason"] is None)
        original = originals[row["date"]]
        _require(set(row["values"]) == set(row["changed_picks"]) == set(audit["choices"]) == set(POLICIES)
                 and row["changed_picks"] == audit["changed_picks"])
        for name in POLICIES:
            arm = audit["choices"][name]["arm"]
            _require(arm in {"control", "challenger"} and (name != "fixed_r1" or arm == "control"))
            values, changed = row["values"][name], row["changed_picks"][name]
            _count(changed)
            _require(changed <= 3 and changed == (original["changed_picks"] if arm == "challenger" else 0))
            _require(set(values) == set(native_metrics) and values == original[arm])
            for metric, value in values.items():
                _number(value)
                if metric in {"up10", "dn5", "whole_path_safe_up10"}:
                    _require(0 <= value <= 1)
                elif metric == "mae":
                    _require(-1 <= value <= 0)
            if changed == 0:
                _require(values == row["values"]["fixed_r1"])
    # A report-controlled bootstrap count cannot allocate unbounded arrays in
    # the bounded dashboard child. This is a display budget, not a policy gate.
    _require(n_boot * max(1, len(daily)) <= 2_000_000)
    base = np.asarray([[row["values"]["fixed_r1"][key] for key in native_metrics] for row in daily])
    result = {}
    for name in POLICIES:
        values = np.asarray([[row["values"][name][key] for key in native_metrics] for row in daily])
        changed = sum(row["changed_picks"][name] for row in daily)
        changed_dates = sum(row["changed_picks"][name] > 0 for row in daily)
        expected_policy = {
            "n_dates": len(daily), "changed_picks": changed, "changed_dates": changed_dates,
            "metrics": _summary(values, n_boot, seed) if len(daily) else None,
            "minus_fixed_r1": _summary(values - base, n_boot, seed) if len(daily) else None,
            "effect_status": "descriptive_forward_effect" if changed else "no_effect_observations",
        }
        _require(report["policies"][name] == expected_policy)
        means, difference = expected_policy["metrics"], expected_policy["minus_fixed_r1"]
        interval = difference["observed_date_block3_ci95"] if difference is not None else None
        result[name] = {
            **{key: expected_policy[key] for key in ("n_dates", "changed_picks", "changed_dates", "effect_status")},
            "no_op_dates": len(daily) - changed_dates, "picks_per_arm": len(daily) * 3,
            "metrics": {key: means["mean"][key] for key in METRICS} if means is not None else None,
            "minus_fixed_r1": {key: difference["mean"][key] for key in METRICS} if difference is not None else None,
            "difference_block_ci95": {key: interval[key] for key in METRICS} if interval is not None else None,
        }
    return result


def _empty(asof, now, status="unavailable"):
    return {
        "schema": SCHEMA, "asof": asof, "observed_at": now.isoformat(),
        "status": status, "attention_required": True,
        "report_generated_at": None, "through_date": None,
        "scope": "post_label_descriptive_not_promotion_or_portfolio",
        "automatic_promotion": False, "trials": {}, "forward": None,
    }


def project_review(source, *, asof, now):
    """Allowlist aggregates only; never expose paths, reasons or picked coins."""
    result = _empty(asof, now)
    if source["status"] not in {"evaluated", "incomplete"}:
        return result
    _require(source["checked_at"] == now.isoformat() and source["deployable"] is False)
    result.update(status=source["status"], attention_required=source["attention_required"],
                  report_generated_at=source["generated_at"], through_date=source["through_date"])
    for name in ("boundary", "shortlist"):
        row = source["summary"][name]
        item = {key: row[key] for key in (
            "paired_dates", "changed_dates", "no_op_dates", "changed_picks", "picks_per_arm",
        )}
        item.update(excluded_dates=len(row["historical_exclusions"]),
                    pending_dates=len(row["awaiting_outcomes"]), metrics=None)
        if row["metrics"] is not None:
            item["metrics"] = {
                arm: {metric: row["metrics"][arm]["mean"][metric] for metric in METRICS}
                for arm in ("control", "challenger", "challenger_minus_control")
            }
            interval = row["metrics"]["challenger_minus_control"]["observed_date_block3_ci95"]
            item["metrics"]["net_difference_block_ci95"] = (
                interval["eod_return_net"] if interval is not None else None
            )
        result["trials"][name] = item
    forward = source["forward"]
    result["forward"] = {
        "start_asof": "2026-10-01",
        "ready_records": forward["prospective_policy_records"],
        "paired_dates": forward["paired_dates"],
        "excluded_dates": len(forward["historical_exclusions"]),
        "changed_picks": {name: forward["changed_picks"][name] for name in POLICIES},
        "policies": copy.deepcopy(forward["policies"]),
    }
    validate_research_progress(result, asof=asof, now=now)
    return result


def _count(value):
    _require(type(value) is int and value >= 0)


def _number(value):
    _require(type(value) in (int, float) and math.isfinite(value))


def validate_research_progress(payload, *, asof, now):
    """Validate encrypted assets too, including unavailable != zero outcomes."""
    _require(type(payload) is dict and set(payload) == set(_empty(asof, now)))
    _require(payload["schema"] in {SCHEMA, LEGACY_SCHEMA} and payload["asof"] == _day(asof))
    observed = datetime.fromisoformat(_timestamp(payload["observed_at"], now=now)).astimezone(KST)
    _require(asof <= observed.date().isoformat())
    _require(now - observed <= timedelta(hours=6))
    _require(payload["scope"] == "post_label_descriptive_not_promotion_or_portfolio"
             and payload["automatic_promotion"] is False)
    _require(payload["status"] in STATES and type(payload["attention_required"]) is bool)
    _require(payload["attention_required"] == (payload["status"] != "evaluated"))
    if payload["status"] in {"unavailable", "historical_not_observed"}:
        _require(payload["report_generated_at"] is None and payload["through_date"] is None
                 and payload["trials"] == {} and payload["forward"] is None)
        if payload["status"] == "historical_not_observed":
            _require(asof < observed.astimezone(KST).date().isoformat())
        return
    _require(asof == observed.astimezone(KST).date().isoformat())
    generated = datetime.fromisoformat(_timestamp(payload["report_generated_at"], now=observed))
    _require(payload["through_date"] == (generated.astimezone(KST).date() - timedelta(days=1)).isoformat())
    # Use exactly the native review's due clock, not a new research cutoff.
    from ops.recommend_trial_review import DUE
    from signals.recommend_experiment_eval import MIN_CI_DATES

    due = observed.date() if observed.time() >= DUE else observed.date() - timedelta(days=1)
    _require(generated.astimezone(KST).date() >= due)
    _require(type(payload["trials"]) is dict and set(payload["trials"]) == {"boundary", "shortlist"})
    for row in payload["trials"].values():
        _require(type(row) is dict and set(row) == {
            "paired_dates", "changed_dates", "no_op_dates", "changed_picks", "picks_per_arm",
            "excluded_dates", "pending_dates", "metrics",
        })
        for key, value in row.items():
            if key != "metrics":
                _count(value)
        n = row["paired_dates"]
        _require(row["changed_dates"] + row["no_op_dates"] == n
                 and row["picks_per_arm"] == n * 3
                 and row["changed_dates"] <= row["changed_picks"] <= row["changed_dates"] * 3)
        metrics = row["metrics"]
        if n == 0:
            _require(metrics is None)
            continue
        _require(type(metrics) is dict and set(metrics) == {
            "control", "challenger", "challenger_minus_control", "net_difference_block_ci95",
        })
        for arm in ("control", "challenger", "challenger_minus_control"):
            _require(type(metrics[arm]) is dict and set(metrics[arm]) == set(METRICS))
            for metric, value in metrics[arm].items():
                _number(value)
                if metric != "eod_return_net":
                    _require((-1 if arm == "challenger_minus_control" else 0) <= value <= 1)
        for metric in METRICS:
            _require(math.isclose(metrics["challenger"][metric] - metrics["control"][metric],
                                  metrics["challenger_minus_control"][metric], abs_tol=1e-12))
        interval = metrics["net_difference_block_ci95"]
        if n < MIN_CI_DATES:
            _require(interval is None)
        else:
            _require(type(interval) is list and len(interval) == 2)
            for bound in interval:
                _number(bound)
            _require(interval[0] <= interval[1])
    forward = payload["forward"]
    forward_fields = {
        "start_asof", "ready_records", "paired_dates", "excluded_dates", "changed_picks",
    }
    if payload["schema"] == SCHEMA:
        forward_fields.add("policies")
    _require(type(forward) is dict and set(forward) == forward_fields and forward["start_asof"] == "2026-10-01")
    for key in ("ready_records", "paired_dates", "excluded_dates"):
        _count(forward[key])
    _require(forward["paired_dates"] <= forward["ready_records"])
    _require(type(forward["changed_picks"]) is dict and set(forward["changed_picks"]) == set(POLICIES))
    for value in forward["changed_picks"].values():
        _count(value)
        _require(value <= forward["paired_dates"] * 3)
    _require(forward["changed_picks"]["fixed_r1"] == 0)
    if asof < forward["start_asof"]:
        _require(forward["ready_records"] == forward["paired_dates"] == forward["excluded_dates"] == 0)
    if payload["schema"] == SCHEMA:
        _validate_forward_policies(forward, min_ci_dates=MIN_CI_DATES)


def _validate_forward_policies(forward, *, min_ci_dates):
    """Public invariants also apply after decryption, without a local report."""
    policies = forward["policies"]
    _require(type(policies) is dict and set(policies) == set(POLICIES))
    for name, row in policies.items():
        _require(type(row) is dict and set(row) == {
            "n_dates", "changed_dates", "no_op_dates", "changed_picks", "picks_per_arm",
            "effect_status", "metrics", "minus_fixed_r1", "difference_block_ci95",
        })
        for key in ("n_dates", "changed_dates", "no_op_dates", "changed_picks", "picks_per_arm"):
            _count(row[key])
        n, changed = row["n_dates"], row["changed_picks"]
        _require(n == forward["paired_dates"] and changed == forward["changed_picks"][name]
                 and row["changed_dates"] + row["no_op_dates"] == n and row["picks_per_arm"] == n * 3
                 and row["changed_dates"] <= changed <= row["changed_dates"] * 3)
        _require(row["effect_status"] == ("descriptive_forward_effect" if changed else "no_effect_observations"))
        if n == 0:
            _require(row["metrics"] is row["minus_fixed_r1"] is row["difference_block_ci95"] is None)
            continue
        for field in ("metrics", "minus_fixed_r1"):
            values = row[field]
            _require(type(values) is dict and set(values) == set(METRICS))
            for metric, value in values.items():
                _number(value)
                if metric != "eod_return_net":
                    _require((-1 if field == "minus_fixed_r1" else 0) <= value <= 1)
        for metric in METRICS:
            base = policies["fixed_r1"]["metrics"][metric]
            _require(math.isclose(row["metrics"][metric] - base, row["minus_fixed_r1"][metric], abs_tol=1e-12))
            if changed == 0:
                _require(row["metrics"][metric] == base and row["minus_fixed_r1"][metric] == 0)
        interval = row["difference_block_ci95"]
        if n < min_ci_dates:
            _require(interval is None)
        else:
            _require(type(interval) is dict and set(interval) == set(METRICS))
            for metric, bounds in interval.items():
                _require(type(bounds) is list and len(bounds) == 2)
                for bound in bounds:
                    _number(bound)
                    if metric != "eod_return_net":
                        _require(-1 <= bound <= 1)
                _require(bounds[0] <= bounds[1])
                if changed == 0:
                    _require(bounds == [0, 0])


def build_research_progress(*, asof, now=None):
    now = (now or datetime.now(KST)).astimezone(KST)
    _day(asof)
    _require(asof <= now.date().isoformat())
    if asof < now.date().isoformat():
        return _empty(asof, now, "historical_not_observed")
    try:
        child = subprocess.run(
            [sys.executable, "-B", "-m", "ops.dashboard_research", "--probe", "--now", now.isoformat()],
            cwd=ROOT, capture_output=True, timeout=20, check=False,
        )
        _require(child.returncode == 0 and len(child.stdout) <= 32768)
        result = strict_json_object_bytes(child.stdout)
        validate_research_progress(result, asof=asof, now=now)
        _require(result["observed_at"] == now.isoformat())
        return result
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, subprocess.TimeoutExpired):
        return _empty(asof, now)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true", required=True)
    parser.add_argument("--now", required=True)
    args = parser.parse_args(argv)
    now = datetime.fromisoformat(args.now)
    _require(now.tzinfo is not None)
    now = now.astimezone(KST)
    result = _empty(now.date().isoformat(), now)
    try:
        result = project_review(read_review(now=now), asof=result["asof"], now=now)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, IndexError):
        pass  # Fixed unavailable projection; never publish private exception text.
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

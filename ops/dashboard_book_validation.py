"""Bounded, allowlisted view of fixed-rule new-date replay; never live promotion."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timedelta

from ops.artifact_provenance import strict_json_object_bytes
from ops.dashboard_current import KST, ROOT, _day, _require, _timestamp
from ops.dashboard_research import _count, _number
from signals.recommend_book_validation import CONFIG, due_dates

SCHEMA = "prelude_dashboard_book_validation.v1"
METRICS = ("eod_return_net", "dn5", "up10")
ARMS = (
    "control",
    "challenger",
    "matched_random_expectation",
    "full_universe",
    "challenger_minus_control",
    "challenger_minus_matched_random_expectation",
)
STATES = {
    "waiting_for_first_outcome",
    "evaluated",
    "incomplete",
    "unavailable",
    "historical_not_observed",
}
COUNTS = (
    "expected_dates",
    "paired_dates",
    "excluded_dates",
    "pending_dates",
    "deferred_dates",
    "changed_dates",
    "no_op_dates",
    "changed_picks",
    "picks_per_arm",
    "outside_top10",
)


def _empty(asof, now, status="unavailable"):
    return {
        "schema": SCHEMA,
        "asof": asof,
        "observed_at": now.isoformat(),
        "status": status,
        "attention_required": True,
        "report_generated_at": None,
        "start_date": CONFIG["start_date"],
        "end_date": CONFIG["end_date"],
        "scope": CONFIG["evaluation_kind"],
        "automatic_promotion": False,
        "prospective_pick_records": 0,
        "counts": None,
        "metrics": None,
        "difference_ci95": None,
    }


def project_report(report, *, now):
    """Caller must supply the independently checked native inspection result."""
    result = _empty(now.astimezone(KST).date().isoformat(), now)
    _require(
        report["scope"] == CONFIG["evaluation_kind"]
        and report["deployable"] is False
        and report["automatic_promotion"] is False
        and report["prospective_pick_records"] == 0
    )
    summary = report["summary"]
    result.update(
        status=report["status"],
        attention_required=report["attention_required"],
        report_generated_at=report["generated_at"],
        counts={
            **{k: summary[k] for k in COUNTS[1:] if k in summary},
            "expected_dates": len(report["calendar"]),
            **{
                s + "_dates": sum(r["status"] == s for r in report["calendar"])
                for s in ("excluded", "pending", "deferred")
            },
        },
    )
    if summary["metrics"] is not None:
        result["metrics"] = {
            arm: {key: summary["metrics"][arm]["mean"][key] for key in METRICS}
            for arm in ARMS
        }
        interval = summary["metrics"]["challenger_minus_control"][
            "observed_date_block3_ci95"
        ]
        result["difference_ci95"] = (
            {k: interval[k] for k in METRICS} if interval is not None else None
        )
    validate_book_validation(result, asof=result["asof"], now=now)
    return result


def validate_book_validation(payload, *, asof, now):
    _require(type(payload) is dict and set(payload) == set(_empty(asof, now)))
    _require(payload["schema"] == SCHEMA and payload["asof"] == _day(asof))
    observed = datetime.fromisoformat(
        _timestamp(payload["observed_at"], now=now)
    ).astimezone(KST)
    _require(
        now - observed <= timedelta(hours=6) and asof <= observed.date().isoformat()
    )
    _require(
        payload["scope"] == CONFIG["evaluation_kind"]
        and payload["automatic_promotion"] is False
        and type(payload["prospective_pick_records"]) is int
        and payload["prospective_pick_records"] == 0
        and payload["start_date"] == CONFIG["start_date"]
        and payload["end_date"] == CONFIG["end_date"]
    )
    status = payload["status"]
    _require(status in STATES and type(payload["attention_required"]) is bool)
    _require(
        payload["attention_required"]
        == (status in {"unavailable", "incomplete", "historical_not_observed"})
    )
    if status in {"unavailable", "historical_not_observed"}:
        _require(
            payload["report_generated_at"]
            is payload["counts"]
            is payload["metrics"]
            is payload["difference_ci95"]
            is None
        )
        _require(
            (asof < observed.date().isoformat())
            == (status == "historical_not_observed")
        )
        return
    _require(asof == observed.date().isoformat())
    generated = datetime.fromisoformat(
        _timestamp(payload["report_generated_at"], now=observed)
    ).astimezone(KST)
    _require(observed - generated <= timedelta(hours=6))
    if (observed.hour, observed.minute) >= (10, 10):
        _require(generated.date() == observed.date())
    counts = payload["counts"]
    _require(type(counts) is dict and set(counts) == set(COUNTS))
    for value in counts.values():
        _count(value)
    n = counts["paired_dates"]
    expected = len(due_dates(generated))
    _require(
        counts["expected_dates"] == expected
        and expected
        == sum(
            counts[k]
            for k in (
                "paired_dates",
                "excluded_dates",
                "pending_dates",
                "deferred_dates",
            )
        )
        and counts["changed_dates"] + counts["no_op_dates"] == n
        and counts["picks_per_arm"] == n * 3
        and counts["changed_dates"]
        <= counts["changed_picks"]
        <= counts["changed_dates"] * 3
        and counts["outside_top10"] <= counts["changed_picks"]
    )
    _require(
        status
        == (
            "waiting_for_first_outcome"
            if expected == 0
            else "incomplete"
            if counts["pending_dates"] + counts["deferred_dates"]
            else "evaluated"
        )
    )
    values, interval = payload["metrics"], payload["difference_ci95"]
    if n == 0:
        _require(values is interval is None)
        return
    _require(type(values) is dict and set(values) == set(ARMS))
    for arm in ARMS:
        _require(type(values[arm]) is dict and set(values[arm]) == set(METRICS))
        for key, value in values[arm].items():
            _number(value)
            if key != "eod_return_net":
                _require((-1 if "minus" in arm else 0) <= value <= 1)
    for base in ("control", "matched_random_expectation"):
        for key in METRICS:
            _require(
                math.isclose(
                    values["challenger"][key] - values[base][key],
                    values["challenger_minus_" + base][key],
                    abs_tol=1e-12,
                )
            )
    if counts["changed_picks"] == 0:
        _require(values["control"] == values["challenger"])
    if n < 5:
        _require(interval is None)
    else:
        _require(type(interval) is dict and set(interval) == set(METRICS))
        for key, bounds in interval.items():
            _require(type(bounds) is list and len(bounds) == 2)
            for value in bounds:
                _number(value)
                if key != "eod_return_net":
                    _require(-1 <= value <= 1)
            _require(bounds[0] <= bounds[1])
            if counts["changed_picks"] == 0:
                _require(bounds == [0, 0])


def build_book_validation(*, asof, now=None):
    now = (now or datetime.now(KST)).astimezone(KST)
    _require(_day(asof) <= now.date().isoformat())
    if asof < now.date().isoformat():
        return _empty(asof, now, "historical_not_observed")
    try:
        child = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "ops.dashboard_book_validation",
                "--now",
                now.isoformat(),
            ],
            cwd=ROOT,
            capture_output=True,
            timeout=30,
            check=False,
        )
        _require(child.returncode == 0 and len(child.stdout) <= 32768)
        result = strict_json_object_bytes(child.stdout)
        validate_book_validation(result, asof=asof, now=now)
        _require(result["observed_at"] == now.isoformat())
        return result
    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        TypeError,
        subprocess.TimeoutExpired,
    ):
        return _empty(asof, now)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--now", required=True)
    args = parser.parse_args(argv)
    now = datetime.fromisoformat(args.now)
    _require(now.utcoffset() is not None)
    now = now.astimezone(KST)
    result = _empty(now.date().isoformat(), now)
    try:
        from ops.recommend_book_validation import inspect

        result = project_report(inspect(now=now), now=now)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, IndexError):
        pass
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

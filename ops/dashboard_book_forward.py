"""Small, public-safe readiness projection. Never expose coins, paths or errors."""

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timedelta

from ops.artifact_provenance import strict_json_object_bytes
from ops.dashboard_current import KST, ROOT, _day, _require, _timestamp
from signals.recommend_book_readiness import CONFIG

SCHEMA = "prelude_dashboard_book_forward.v1"
VERDICTS = {
    "waiting",
    "continue_observing",
    "review_candidate",
    "do_not_adopt",
    "blocked",
}
RECORDS = {"not_due", "ready", "missing", "late", "uncertain"}
METRICS = ("eod_return_net", "dn5", "up10")
CHECKS = {
    "review_sample",
    "changed_dates",
    "joint_down_up_safe_net",
    "positive_challenger_net",
    "beats_matched_net",
    "chronological_consistency",
    "leave_one_date_out_net",
    "context_coverage",
    "context_consistency",
    "execution_coverage",
    "delayed_and_extra_cost",
    "observed_date_coverage",
}


def empty(asof, now, status="unavailable"):
    return {
        "schema": SCHEMA,
        "asof": asof,
        "observed_at": now.isoformat(),
        "status": status,
        "attention_required": True,
        "generated_at": None,
        "today_record": None,
        "paired_dates": None,
        "changed_dates": None,
        "execution_dates": None,
        "expected_dates": None,
        "verdict": None,
        "checks": None,
        "difference": None,
        "automatic_promotion": False,
        "scope": "pre_entry_paper_not_actual_trades",
    }


def project(native, *, now):
    report, review = native["report"], native["report"]["review"]
    _require(report["automatic_promotion"] is False and review["deployable"] is False)
    result = empty(now.date().isoformat(), now)
    summary = review["summary"]
    result.update(
        status="observed",
        attention_required=native["attention_required"],
        generated_at=report["generated_at"],
        today_record=native["today_record"],
        paired_dates=summary["paired_dates"],
        changed_dates=summary["changed_dates"],
        execution_dates=review["execution"]["paired_dates"],
        expected_dates=len(report["calendar"]),
        verdict=review["verdict"],
        checks=review["checks"],
        difference=(
            {
                k: summary["metrics"]["challenger_minus_control"]["mean"][k]
                for k in METRICS
            }
            if summary["metrics"] is not None
            else None
        ),
    )
    validate_book_forward(result, asof=result["asof"], now=now)
    return result


def validate_book_forward(value, *, asof, now):
    _require(type(value) is dict and set(value) == set(empty(asof, now)))
    _require(
        value["schema"] == SCHEMA
        and value["asof"] == _day(asof)
        and value["automatic_promotion"] is False
        and type(value["attention_required"]) is bool
        and value["scope"] == "pre_entry_paper_not_actual_trades"
    )
    observed = datetime.fromisoformat(
        _timestamp(value["observed_at"], now=now)
    ).astimezone(KST)
    _require(
        now - observed <= timedelta(hours=6) and asof <= observed.date().isoformat()
    )
    _require(value["status"] in {"observed", "unavailable", "historical_not_observed"})
    if value["status"] != "observed":
        expected = empty(asof, observed, value["status"])
        expected["observed_at"] = value["observed_at"]  # Valid UTC/KST offsets are equivalent.
        _require(value == expected)
        _require(
            (asof < observed.date().isoformat())
            == (value["status"] == "historical_not_observed")
        )
        return
    _require(asof == observed.date().isoformat())
    generated = datetime.fromisoformat(
        _timestamp(value["generated_at"], now=observed)
    ).astimezone(KST)
    _require(observed - generated <= timedelta(hours=6))
    if (observed.hour, observed.minute) >= (10, 10):
        _require(generated.date() == observed.date())
    _require(value["today_record"] in RECORDS and value["verdict"] in VERDICTS)
    for field in ("paired_dates", "changed_dates", "execution_dates", "expected_dates"):
        _require(type(value[field]) is int and value[field] >= 0)
    n = value["paired_dates"]
    _require(
        value["changed_dates"] <= n <= value["expected_dates"]
        and value["execution_dates"] <= n
    )
    _require(
        type(value["checks"]) is dict
        and set(value["checks"]) <= CHECKS
        and all(type(v) is bool for v in value["checks"].values())
    )
    if not n:
        _require(
            value["difference"] is None and value["verdict"] in {"waiting", "blocked"}
        )
    else:
        delta = value["difference"]
        _require(type(delta) is dict and set(delta) == set(METRICS))
        for k, number in delta.items():
            _require(type(number) in (int, float) and math.isfinite(number))
            if k != "eod_return_net":
                _require(-1 <= number <= 1)
        _require(value["verdict"] != "waiting")
    if value["verdict"] == "review_candidate":
        _require(
            n >= CONFIG["min_paired_dates"]
            and set(value["checks"]) == CHECKS
            and all(value["checks"].values())
        )
    if value["today_record"] not in {"ready", "not_due"}:
        _require(value["attention_required"])


def build_book_forward(*, asof, now=None):
    now = (now or datetime.now(KST)).astimezone(KST)
    _require(_day(asof) <= now.date().isoformat())
    if asof < now.date().isoformat():
        return empty(asof, now, "historical_not_observed")
    try:
        child = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "ops.dashboard_book_forward",
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
        validate_book_forward(result, asof=asof, now=now)
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
        return empty(asof, now)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--now", required=True)
    args = parser.parse_args(argv)
    now = datetime.fromisoformat(args.now)
    _require(now.utcoffset() is not None)
    now = now.astimezone(KST)
    result = empty(now.date().isoformat(), now)
    try:
        from ops.recommend_book_forward import inspect

        result = project(inspect(now=now), now=now)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, IndexError):
        pass
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

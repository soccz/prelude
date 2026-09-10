"""Read-only check of the latest daily systemd selftest execution.

Only a completed successful run that started and ended today in KST is healthy.
A later successful same-day rerun clears this latest-run check; earlier failure
alerts and the system journal remain the historical evidence. This is not a
test runner, a recommendation gate, or a claim about the current code revision.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

UNIT = "prelude-selftest.service"
PROBE_TIMEOUT_SECONDS = 8
KST = ZoneInfo("Asia/Seoul")
PROPERTIES = (
    "Id",
    "LoadState",
    "ActiveState",
    "SubState",
    "Result",
    "ExecMainCode",
    "ExecMainStatus",
    "ExecMainStartTimestamp",
    "ExecMainExitTimestamp",
)


def _parse_properties(output: str) -> dict[str, str]:
    properties: dict[str, str] = {}
    for line in output.splitlines():
        key, separator, value = line.partition("=")
        if not separator or key not in PROPERTIES or key in properties:
            raise ValueError("invalid systemctl property response")
        properties[key] = value
    if set(properties) != set(PROPERTIES):
        raise ValueError("incomplete systemctl property response")
    return properties


def _timestamp(value: str) -> datetime:
    # systemctl inherits LC_ALL=C and TZ=UTC below. Do not interpret server
    # locale/timezone strings with the caller's local timezone.
    parsed = datetime.strptime(value, "%a %Y-%m-%d %H:%M:%S UTC").replace(
        tzinfo=timezone.utc
    )
    if parsed.strftime("%a %Y-%m-%d %H:%M:%S UTC") != value:
        raise ValueError("noncanonical systemctl timestamp")
    return parsed


def inspect_selftest(*, now: datetime | None = None) -> dict:
    if now is not None and (now.tzinfo is None or now.utcoffset() is None):
        raise ValueError("selftest status clock must be timezone-aware")
    command = ["systemctl", "show", UNIT, "--no-pager"]
    for name in PROPERTIES:
        command.extend(("--property", name))
    properties = None
    probe_error = None
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
            check=False,
            env={**os.environ, "LC_ALL": "C", "TZ": "UTC"},
        )
        if result.returncode:
            probe_error = f"systemctl exit={result.returncode}"
        else:
            properties = _parse_properties(result.stdout)
    except subprocess.TimeoutExpired:
        probe_error = "systemctl timeout"
    except (OSError, ValueError, UnicodeError) as exc:
        # Never copy arbitrary process stderr or OS exception text into alerts.
        probe_error = f"systemctl response unavailable ({type(exc).__name__})"

    observed = (now or datetime.now(timezone.utc)).astimezone(KST)
    report = {
        "unit": UNIT,
        "checked_at": observed.isoformat(),
        "asof": observed.date().isoformat(),
        "state": "probe_error",
        "attention_required": True,
        "reason": probe_error,
        "started_at": None,
        "completed_at": None,
        "policy": "latest_run_started_and_completed_successfully_today_kst",
    }
    if probe_error is not None:
        return report
    assert properties is not None

    def finish(state: str, reason: str) -> dict:
        report.update(state=state, reason=reason, attention_required=state != "passed")
        return report

    if properties["Id"] != UNIT or properties["LoadState"] != "loaded":
        return finish("invalid_state", "expected selftest unit is not loaded")
    active, substate = properties["ActiveState"], properties["SubState"]
    if active in {"activating", "active", "reloading", "deactivating"}:
        return finish("in_progress", "selftest has not completed successfully")
    if active not in {"inactive", "failed"}:
        return finish("invalid_state", "unknown selftest active state")
    if (
        not properties["Result"]
        or not properties["ExecMainCode"].isdigit()
        or not properties["ExecMainStatus"].isdigit()
    ):
        return finish("invalid_state", "missing selftest completion fields")
    if active == "failed" or properties["Result"] != "success":
        return finish("failed", "latest selftest execution failed")
    if substate != "dead":
        return finish("invalid_state", "selftest completion state is inconsistent")
    if not properties["ExecMainStartTimestamp"]:
        return finish("not_run", "selftest has no recorded execution")
    try:
        started = _timestamp(properties["ExecMainStartTimestamp"])
        completed = _timestamp(properties["ExecMainExitTimestamp"])
    except ValueError:
        return finish("invalid_state", "selftest execution timestamps are invalid")
    report.update(
        started_at=started.astimezone(KST).isoformat(),
        completed_at=completed.astimezone(KST).isoformat(),
    )
    if started > completed or completed > observed:
        return finish("invalid_state", "selftest execution chronology is inconsistent")
    if properties["ExecMainCode"] != "1" or properties["ExecMainStatus"] != "0":
        return finish("invalid_state", "selftest success contradicts process exit")
    if (
        started.astimezone(KST).date() != observed.date()
        or completed.astimezone(KST).date() != observed.date()
    ):
        return finish("stale", "no successful selftest run started and ended today")
    return finish("passed", "latest selftest run completed successfully today")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("text", "json"), default="json")
    args = parser.parse_args(argv)
    report = inspect_selftest()
    if args.format == "json":
        print(json.dumps(report, ensure_ascii=False))
    else:
        print(f"selftest {report['asof']}: {report['state']} — {report['reason']}")
    return 1 if report["attention_required"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

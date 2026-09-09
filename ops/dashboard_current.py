"""Public-safe current-operation projection; no scores, outcomes, or mutations.

The historical dashboard channels retain their old definitions. This separate
versioned object reports actual R1 delivery evidence and record-only research.
Research probes run in bounded child processes: a slow/broken experiment must
not hide the delivery cards or claim that an experiment has zero observations.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from notifier.delivery_receipt import _LIVE_SEND_WINDOWS
from ops.artifact_provenance import strict_json_object_bytes
from ops.recommendation_status import ATTENTION_STATES, build_recommendation_status

ROOT = Path(__file__).resolve().parent.parent
KST = ZoneInfo("Asia/Seoul")
SCHEMA = "prelude_dashboard_current.v1"
LIVE_STATES = frozenset(
    {
        "waiting",
        "pending",
        "delivered_candidates",
        "delivered_empty",
        "not_delivered",
        "delivery_uncertain",
        "invalid_evidence",
        "missing_decision",
        "probe_unavailable",
    }
)
RESEARCH_STATES = frozenset(
    {
        "not_started",
        "waiting",
        "capture_missing",
        "capture_ambiguous",
        "capture_unfinished",
        "capture_incomplete",
        "upstream_snapshot_missing",
        "capture_quality_invalid",
        "feature_missing",
        "feature_quality_invalid",
        "trial_missing",
        "trial_uncertain",
        "evaluation_missing",
        "publication_uncertain",
        "score_missing",
        "recorded_evaluation_missing",
        "missing",
        "uncertain",
        "evidence_invalid",
        "complete_noop",
        "complete_changed",
        "complete_unavailable",
        "probe_unavailable",
        "historical_not_observed",
    }
)
PROBES = {
    "microstructure": (
        "ops.recommend_microstructure_status",
        "recommend_microstructure_operational_status.v1",
        "2026-09-09",
    ),
    "trade_shortlist": (
        "ops.recommend_trade_shortlist_status",
        "recommend_trade_shortlist_operational_status.v1",
        "2026-09-10",
    ),
}


def _require(ok: bool) -> None:
    if not ok:
        # Do not put arbitrary source values, paths, or exception text in output.
        raise ValueError("invalid current dashboard contract")


def _timestamp(value, *, now: datetime, nullable: bool = False):
    if value is None and nullable:
        return None
    _require(type(value) is str)
    parsed = datetime.fromisoformat(value)
    _require(parsed.tzinfo is not None and parsed.utcoffset() is not None)
    _require(parsed <= now)
    return value


def _day(value):
    _require(type(value) is str and date.fromisoformat(value).isoformat() == value)
    return value


def _validate_live_row(row, *, slot: str, asof: str, observed: datetime) -> None:
    _require(
        type(row) is dict
        and set(row)
        == {
            "state",
            "candidate_count",
            "sent_at",
            "decision_completed_at",
            "attention_required",
            "read_confirmation",
            "actual_input_age_seconds",
        }
    )
    state = row["state"]
    _require(state in LIVE_STATES and type(row["attention_required"]) is bool)
    _require(
        row["attention_required"]
        == (state in ATTENTION_STATES or state == "probe_unavailable")
    )
    count = row["candidate_count"]
    _require(count is None or type(count) is int and 0 <= count <= 3)
    sent = _timestamp(row["sent_at"], now=observed, nullable=True)
    completed = _timestamp(row["decision_completed_at"], now=observed, nullable=True)
    day = date.fromisoformat(asof)
    start_wall, end_wall = _LIVE_SEND_WINDOWS[slot]
    start = datetime.combine(day, start_wall, KST)
    end = datetime.combine(day, end_wall, KST)
    for value in (sent, completed):
        if value is not None:
            _require(start <= datetime.fromisoformat(value) < end)
    if sent is not None and completed is not None:
        # Telegram server timestamps have whole-second precision. Preserve the
        # native receipt's floor rule, not a stricter fractional clock ordering.
        _require(
            datetime.fromisoformat(completed).replace(microsecond=0)
            <= datetime.fromisoformat(sent)
        )
    _require(
        row["read_confirmation"] == "unavailable"
        and row["actual_input_age_seconds"] is None
    )
    if state in {"delivered_candidates", "delivered_empty"}:
        _require(sent is not None and completed is not None and count is not None)
        _require((count == 0) == (state == "delivered_empty"))


def _research_attention(state: str, previous=None) -> bool:
    _require(state in RESEARCH_STATES)
    if state == "waiting":
        _require(type(previous) is dict and previous["state"] != "waiting")
        expected = _research_attention(previous["state"])
        _require(type(previous["attention_required"]) is bool)
        _require(previous["attention_required"] == expected)
        return expected
    _require(previous is None)
    return state not in {
        "not_started", "complete_noop", "complete_changed", "complete_unavailable"
    }


def _validate_research_row(row, *, name: str, asof: str, observed: datetime) -> None:
    _require(
        type(row) is dict
        and set(row)
        == {
            "state",
            "attention_required",
            "asof",
            "start_asof",
            "checked_at",
            "previous_due",
            "effect_status",
            "deployable",
        }
    )
    _require(row["asof"] == asof and row["start_asof"] == PROBES[name][2])
    _require(type(row["attention_required"]) is bool)
    _require(row["effect_status"] == "not_evaluated" and row["deployable"] is False)
    checked = _timestamp(row["checked_at"], now=observed, nullable=True)
    previous = row["previous_due"]
    if previous is not None:
        _require(
            type(previous) is dict
            and set(previous) == {"asof", "state", "attention_required"}
        )
        _require(
            previous["asof"]
            == (date.fromisoformat(asof) - timedelta(days=1)).isoformat()
        )
    _require(row["attention_required"] == _research_attention(row["state"], previous))
    if row["state"] in {"probe_unavailable", "historical_not_observed"}:
        _require(checked is None)
        if row["state"] == "historical_not_observed":
            _require(date.fromisoformat(asof) < observed.astimezone(KST).date())
    else:
        _require(checked == observed.isoformat())
        _require(asof == observed.astimezone(KST).date().isoformat())


def _run_probe(module: str, now: datetime) -> dict:
    result = subprocess.run(
        [sys.executable, "-B", "-m", module, "--now", now.isoformat()],
        cwd=ROOT,
        capture_output=True,
        timeout=30,
        check=False,
    )
    _require(result.returncode in (0, 1) and len(result.stdout) <= 65536)
    report = strict_json_object_bytes(result.stdout)
    _require(type(report.get("attention_required")) is bool)
    _require(result.returncode == int(report["attention_required"]))
    return report


def _research(name: str, *, asof: str, now: datetime) -> dict:
    module, schema, start = PROBES[name]
    row = {
        "state": "probe_unavailable",
        "attention_required": True,
        "asof": asof,
        "start_asof": start,
        "checked_at": None,
        "previous_due": None,
        "effect_status": "not_evaluated",
        "deployable": False,
    }
    if asof != now.date().isoformat():
        return {**row, "state": "historical_not_observed"}
    try:
        source = _run_probe(module, now)
        _require(source["schema"] == schema and source["asof"] == asof)
        _require(source["effect_status"] == "not_evaluated")
        _require(source["deployable"] is False)
        _require(source["status"] in RESEARCH_STATES)
        _require(type(source["attention_required"]) is bool)
        checked = _timestamp(source["checked_at"], now=now)
        _require(checked == now.isoformat())
        previous = source.get("previous_due")
        if previous is not None:
            _require(source["status"] == "waiting")
            _require(previous["schema"] == schema)
            _require(previous["asof"] == (now.date() - timedelta(days=1)).isoformat())
            _require(previous["status"] in RESEARCH_STATES)
            _require(type(previous["attention_required"]) is bool)
            _require(previous["checked_at"] == checked)
            _require(previous["effect_status"] == "not_evaluated")
            _require(previous["deployable"] is False)
            previous = {
                "asof": previous["asof"],
                "state": previous["status"],
                "attention_required": previous["attention_required"],
            }
        projected = {
            **row,
            "state": source["status"],
            "checked_at": checked,
            "attention_required": source["attention_required"],
            "previous_due": previous,
        }
        _validate_research_row(projected, name=name, asof=asof, observed=now)
        return projected
    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        TypeError,
        subprocess.TimeoutExpired,
    ):
        return row


def build_current_system(*, asof: str, now: datetime | None = None) -> dict:
    now = datetime.now(KST) if now is None else now
    _require(now.tzinfo is not None and now.utcoffset() is not None)
    now = now.astimezone(KST)
    _require(date.fromisoformat(_day(asof)) <= now.date())
    live = {
        slot: {
            "state": "probe_unavailable",
            "candidate_count": None,
            "sent_at": None,
            "decision_completed_at": None,
            "attention_required": True,
            "read_confirmation": "unavailable",
            "actual_input_age_seconds": None,
        }
        for slot in ("preopen", "open")
    }
    try:
        report = build_recommendation_status(asof, now=now)
        _require(report["asof"] == asof)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError):
        report = None
    if report is not None:
        for slot in live:
            try:
                source = report["slots"][slot]
                state = source["state"]
                projected = {
                    **live[slot],
                    "state": state,
                    "candidate_count": source["candidate_count"],
                    "sent_at": source["sent_at"],
                    "decision_completed_at": source["decision_completed_at"],
                    "attention_required": state in ATTENTION_STATES
                    or state == "probe_unavailable",
                }
                _validate_live_row(projected, slot=slot, asof=asof, observed=now)
                live[slot] = projected
            except (OSError, ValueError, RuntimeError, KeyError, TypeError):
                # One malformed slot cannot hide its independent sibling.
                # Commit only a fully validated projection; unknown is not zero.
                pass
    result = {
        "schema": SCHEMA,
        "asof": asof,
        "observed_at": now.isoformat(),
        "automatic_orders": False,
        "automatic_promotion": False,
        "live": live,
        "research": {name: _research(name, asof=asof, now=now) for name in PROBES},
        "legacy": {
            "pump_v2": "terminal_kill",
            "legacy_channels": ["distribution", "preopen"],
        },
    }
    validate_current_system(result, asof=asof, now=now)
    return result


def validate_current_system(payload, *, asof: str, now: datetime) -> None:
    """Strict allowlist; unknown/private fields cannot quietly enter publication."""
    _require(
        type(payload) is dict
        and set(payload)
        == {
            "schema",
            "asof",
            "observed_at",
            "automatic_orders",
            "automatic_promotion",
            "live",
            "research",
            "legacy",
        }
    )
    _require(payload["schema"] == SCHEMA and payload["asof"] == asof)
    observed = datetime.fromisoformat(_timestamp(payload["observed_at"], now=now))
    _require(now - observed <= timedelta(hours=6))
    _require(date.fromisoformat(_day(asof)) <= observed.astimezone(KST).date())
    _require(
        payload["automatic_orders"] is False and payload["automatic_promotion"] is False
    )
    _require(
        type(payload["live"]) is dict and set(payload["live"]) == {"preopen", "open"}
    )
    for slot, row in payload["live"].items():
        _validate_live_row(row, slot=slot, asof=asof, observed=observed)
    _require(
        type(payload["research"]) is dict and set(payload["research"]) == set(PROBES)
    )
    for name, row in payload["research"].items():
        _validate_research_row(row, name=name, asof=asof, observed=observed)
    _require(
        payload["legacy"]
        == {
            "pump_v2": "terminal_kill",
            "legacy_channels": ["distribution", "preopen"],
        }
    )

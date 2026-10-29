"""One persistent incident state for operational Telegram notifications.

Recommendations never pass through this module. Research diagnostics are kept
in the state file/logs. Only newly observed important incidents and recoveries
are notified. An unsuccessful delivery never advances the acknowledged state.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from notifier.telegram import send_telegram_with_receipt, telegram_error_is_ambiguous
from ops.artifact_provenance import atomic_write_bytes, strict_json_object
from ops.file_lock import file_lock

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE = ROOT / "output/operations_notifications.json"
SCHEMA = "prelude_operations_notifications.v1"
LABELS = {
    "recommendation": "추천 발송·기록 확인 필요",
    "recommendation_preopen": "장전 추천 발송 확인 필요",
    "recommendation_open": "장후 추천 발송 확인 필요",
    "publish": "대시보드 갱신 확인 필요",
    "backup": "백업 확인 필요",
    "selftest": "자동 점검 확인 필요",
    "data": "데이터 무결성 확인 필요",
    "disk": "저장 공간 확인 필요",
    "close": "추천 결과 기록 확인 필요",
    "monitor": "운영 모니터 확인 필요",
    "other": "운영 상태 확인 필요",
}
RESEARCH = (
    "microstructure",
    "shortlist",
    "post-label",
    "fixed L1",
    "regime forward",
    "L1 forward",
    "research snapshot",
)
UNITS = {
    "prelude-preopen.service": "recommendation_preopen",
    "prelude-distribution.service": "recommendation_open",
    "prelude-publish-dashboard.service": "publish",
    "prelude-backup.service": "backup",
    "prelude-selftest.service": "selftest",
    "prelude-close.service": "research:close",
    "prelude-preopen-close.service": "research:preopen-close",
    "prelude-microstructure.service": "research:microstructure",
    "prelude-heartbeat.service": "monitor",
}


def _issue_key(text: str) -> str:
    for name in RESEARCH:
        if name in text:
            return "research:" + name
    for token, key in (
        ("snapshot chain", "recommendation"),
        ("no-decision", "recommendation"),
        ("publish", "publish"),
        ("backup", "backup"),
        ("selftest", "selftest"),
        ("disk", "disk"),
        ("integrity", "data"),
        ("스키마", "data"),
        (".db", "data"),
    ):
        if token in text:
            return key
    return "other"


def notify(
    observations: dict[str, list[str]],
    *,
    watched: set[str] | None = None,
    state_path: Path = DEFAULT_STATE,
    sender: Callable | None = None,
    now: datetime | None = None,
) -> str:
    """Return sent/suppressed/recorded/uncertain; raise on known delivery failure.

    watched declares which absent incidents were actually checked and healthy.
    OnFailure reports only one observation and cannot clear unrelated incidents.
    """
    clock = now or datetime.now(timezone.utc)
    if clock.tzinfo is None:
        raise ValueError("notification time must be timezone-aware")
    if any(k not in LABELS and not k.startswith("research:") for k in observations):
        raise ValueError("unknown notification incident key")
    path = Path(state_path)
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError("notification state must not be a symlink")
    with file_lock(path.with_suffix(".lock")):
        if path.exists():
            state = strict_json_object(path)
            if (
                state.get("schema") != SCHEMA
                or not isinstance(state.get("active"), dict)
                or not isinstance(state.get("acknowledged"), list)
            ):
                raise ValueError("notification state schema invalid")
        else:
            state = {"schema": SCHEMA, "active": {}, "acknowledged": []}
        active = dict(state["active"])
        previous_active = set(active)
        for key in watched or set():
            active.pop(key, None)
        active.update({k: list(v) for k, v in observations.items()})
        current = {k for k in active if not k.startswith("research:")}
        acknowledged = set(state["acknowledged"]) - (current - previous_active)
        uncertain = (
            set(state.get("uncertain_target", []))
            | set(state.get("pending_target", []))
        ) & current
        new = sorted(current - acknowledged - uncertain)
        recovered = sorted(acknowledged - current)
        state.update(active=active, observed_at=clock.isoformat())
        result = (
            "uncertain"
            if uncertain
            else "recorded"
            if observations and not current
            else "suppressed"
        )
        if new or recovered:
            # If server acceptance was ambiguous, do not replay the same state.
            # Keep it explicitly unconfirmed rather than claiming delivery.
            target = sorted(current)
            if (
                state.get("uncertain_target") == target
                or state.get("pending_target") == target
            ):
                state["uncertain_target"] = target
                result = "uncertain"
            else:
                lines = ["⚠️ prelude 운영 알림" if new else "✅ prelude 운영 복구"]
                lines += ["• " + LABELS[k] for k in new]
                lines += [
                    "• 복구: " + LABELS[k].replace(" 확인 필요", "") for k in recovered
                ]
                lines.append(
                    "같은 문제는 반복 알림하지 않습니다. 상세 점검은 서버 기록·대시보드에서 확인하세요."
                )
                # Write-ahead intent prevents replay after a process crash in
                # the send -> acknowledgement window. Unconfirmed is not sent.
                state["pending_target"] = target
                atomic_write_bytes(
                    path, (json.dumps(state, ensure_ascii=False) + "\n").encode()
                )
                delivery = (sender or send_telegram_with_receipt)("\n".join(lines))
                ok = delivery if isinstance(delivery, bool) else delivery.delivery_ok
                error = None if isinstance(delivery, bool) else delivery.error
                if ok:
                    state["acknowledged"] = sorted((acknowledged & current) | set(new))
                    if uncertain:
                        state["uncertain_target"] = sorted(uncertain)
                    else:
                        state.pop("uncertain_target", None)
                    state.pop("pending_target", None)
                    state["notified_at"] = clock.isoformat()
                    result = "sent"
                elif error and telegram_error_is_ambiguous(error):
                    state["uncertain_target"] = target
                    result = "uncertain"
                else:
                    state.pop("pending_target", None)
                    state["last_delivery_failed_at"] = clock.isoformat()
                    atomic_write_bytes(
                        path, (json.dumps(state, ensure_ascii=False) + "\n").encode()
                    )
                    raise RuntimeError("operational notification delivery failed")
        if not current and not recovered:
            state.pop("uncertain_target", None)
            state.pop("pending_target", None)
        atomic_write_bytes(
            path, (json.dumps(state, ensure_ascii=False) + "\n").encode()
        )
        return result


def heartbeat(message: str, *, delivery_status: dict | None = None, **kwargs) -> str:
    observations: dict[str, list[str]] = {}
    for line in message.splitlines():
        if line.startswith("- "):
            detail = line[2:]
            observations.setdefault(_issue_key(detail), []).append(detail)
    # These checks run on every heartbeat, so absence is a healthy observation.
    watched = {
        "recommendation",
        "publish",
        "backup",
        "selftest",
        "data",
        "disk",
        "other",
        "monitor",
    }
    watched.update("research:" + name for name in RESEARCH)
    if delivery_status is not None:
        delivery_problems = {}
        for slot, row in delivery_status.get("slots", {}).items():
            if slot not in {"preopen", "open"}:
                continue
            key = "recommendation_" + slot
            if row.get("state") in {"delivered_candidates", "delivered_empty"}:
                watched.add(key)
            elif row.get("attention_required") or row.get("state") in {
                "missing_decision",
                "invalid_evidence",
                "not_delivered",
                "delivery_uncertain",
            }:
                watched.add(key)
                delivery_problems[key] = [row.get("reason", "delivery needs checking")]
        if delivery_problems:
            observations.pop("recommendation", None)
            observations.update(delivery_problems)
    return notify(observations, watched=watched, **kwargs)


def failure(unit: str, **kwargs) -> str:
    if unit not in UNITS:
        raise ValueError("unknown prelude service")
    key = UNITS[unit]
    if key in {"recommendation_preopen", "recommendation_open"}:
        # A legacy/research stage may fail AFTER a verified R1 was delivered.
        # That must not tell the user the actual recommendation failed.
        from ops.recommendation_status import KST, build_recommendation_status

        clock = kwargs.get("now") or datetime.now(KST)
        try:
            report = build_recommendation_status(
                clock.astimezone(KST).date().isoformat(), now=clock
            )
            slot = "preopen" if unit == "prelude-preopen.service" else "open"
            if report["slots"][slot]["state"] in {
                "delivered_candidates",
                "delivered_empty",
            }:
                key = "research:" + unit
        except Exception:
            pass  # Missing/invalid evidence remains important.
    return notify({key: [unit]}, **kwargs)


def notify_heartbeat_from_environment() -> None:
    from ops.recommendation_status import KST, build_recommendation_status

    now = datetime.now(KST)
    try:
        report = build_recommendation_status(now.date().isoformat(), now=now)
    except Exception:
        report = None
    print(
        "operations notification:",
        heartbeat(os.environ.get("HEARTBEAT_MESSAGE", ""), delivery_status=report),
    )

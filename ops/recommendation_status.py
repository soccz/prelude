"""Read-only R1 delivery status from local snapshots, receipts and attempts.

This report never scores, retries a send, creates a receipt, or takes a writer
lock. Missing evidence is not evidence of zero candidates or non-delivery.
"""
from __future__ import annotations

import stat
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from notifier.delivery_attempt import inspect_delivery_attempt
from notifier.delivery_receipt import (
    DEFAULT_RECEIPT_ROOT,
    RECEIPT_INTEGRITY_ACTIVATION_DATE,
    DeliveryReceiptError,
    _LIVE_SEND_WINDOWS,
    read_delivery_receipt,
    receipt_path,
)
from notifier.telegram import telegram_error_is_ambiguous
from signals.recommend_snapshot import (
    DEFAULT_SNAPSHOT_ROOT,
    SNAPSHOT_SCHEMA_VERSION,
    SnapshotError,
    load_snapshot,
    snapshot_path,
)

KST = ZoneInfo("Asia/Seoul")
REPORT_SCHEMA = "recommendation_status.v2"
SLOTS = ("preopen", "open")
# Reporting the supported identity does not authorize a new live model.
APPROVED_RANK_BASIS = "R1_riskreward(de-corr head)"
APPROVED_SCORE_SCHEMA = "recommend_score.v2"
APPROVED_RULE_VERSION = "r1_riskreward_v1"
ATTENTION_STATES = frozenset(
    {"not_delivered", "delivery_uncertain", "invalid_evidence", "missing_decision"}
)
NOTICES = (
    "전달 확인은 Telegram 서버 수락 증거이며 사용자의 읽음 확인이 아닙니다.",
    "후보 0개는 저장된 후보가 없다는 뜻이며 품질 조건 탈락을 뜻하지 않습니다.",
    "표시 확률의 실전 보정 신뢰도는 보장되지 않으며 상승·하락 가능성은 별도 추정값입니다.",
    "09:00 시가는 참고 가격이며 현재 시세나 실제 체결 가능한 가격이 아닙니다.",
    "피처 행 날짜와 실제 입력봉 날짜는 다릅니다. 코인별 실제 마지막 입력봉 시각은 이 증거에 없습니다.",
    "미완료 발송 시도는 미발송으로 단정하지 않습니다. 중복 방지를 위해 자동 재발송을 보류합니다.",
)


def _canonical_day(value: str) -> date:
    if not isinstance(value, str):
        raise ValueError("asof must be a canonical YYYY-MM-DD string")
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError("asof must be a canonical YYYY-MM-DD string")
    return parsed


def _path_state(path: Path) -> str:
    """Use lstat so dangling links are invalid evidence, not absent evidence."""
    try:
        parent = path.parent.lstat()
    except FileNotFoundError:
        return "missing"
    if not stat.S_ISDIR(parent.st_mode):
        raise ValueError("evidence parent must be a real directory")
    try:
        value = path.lstat()
    except FileNotFoundError:
        return "missing"
    if not stat.S_ISREG(value.st_mode):
        raise ValueError("evidence must be a regular non-symlink file")
    return "present"


def _timestamp(value: str, *, now: datetime) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("evidence timestamp must be timezone-aware")
    if parsed > now:
        raise ValueError("evidence timestamp is after observed_at")
    return parsed.astimezone(KST)


def _set_state(row: dict, state: str, delivery: str, reason: str) -> dict:
    row.update(state=state, delivery_state=delivery, reason=reason)
    return row


def _slot_status(
    day: date,
    slot: str,
    now: datetime,
    snapshot_root: Path,
    receipt_root: Path,
) -> dict[str, Any]:
    start_wall, end_wall = _LIVE_SEND_WINDOWS[slot]
    start = datetime.combine(day, start_wall, tzinfo=KST)
    end = datetime.combine(day, end_wall, tzinfo=KST)
    source = snapshot_path(day.isoformat(), slot, root=snapshot_root)
    receipt = receipt_path({"asof": day.isoformat(), "slot": slot}, root=receipt_root)
    row: dict[str, Any] = {
        "slot": slot,
        "artifact_state": "missing",
        "snapshot_state": "missing",
        "receipt_state": "missing",
        "candidate_count": None,
        "snapshot_path": str(source),
        "receipt_path": str(receipt),
        "send_window_start": start.isoformat(),
        "send_window_end_exclusive": end.isoformat(),
        "snapshot_id": None,
        "decision_started_at": None,
        "decision_completed_at": None,
        "decision_age_seconds": None,
        "sent_at": None,
        "delivery_age_seconds": None,
        "delivered_chunk_count": None,
        "expected_chunk_count": None,
        "feature_row_date": None,
        "nominal_input_candle_date": None,
        "actual_input_candle_date": None,
        "actual_input_age_seconds": None,
        "input_date_basis": "nominal_shift1; per_coin_actual_not_recorded",
        "reference_price_kind": (
            "09:00_open_reference" if slot == "open" else "unavailable_before_open"
        ),
        "read_confirmation": "unavailable",
        "approval_identity_matches": None,
        "attempt_state": "not_checked",
        "latest_attempt_id": None,
        "attempt_count": None,
        "attempted_at": None,
    }
    try:
        row["snapshot_state"] = _path_state(source)
        row["receipt_state"] = _path_state(receipt)
        # Resolve directory aliases only after rejecting a symlinked artifact
        # itself. The same physical snapshot must keep the producer's canonical
        # path identity; a copy at a different location still fails its receipt.
        source = source.resolve()
        row["snapshot_path"] = str(source)
        row["receipt_path"] = str(receipt.resolve())
        if row["snapshot_state"] == "missing":
            attempt_directory = receipt.parent / f"{receipt.stem}.attempts"
            try:
                attempt_directory.lstat()
            except FileNotFoundError:
                pass
            else:
                row.update(artifact_state="invalid", attempt_state="invalid")
                return _set_state(row, "invalid_evidence", "unknown", "orphan_attempt_journal")
            if row["receipt_state"] != "missing":
                row["artifact_state"] = "invalid"
                return _set_state(row, "invalid_evidence", "unknown", "orphan_receipt")
            if now < start:
                return _set_state(row, "waiting", "not_observed", "before_send_window")
            if now < end:
                return _set_state(row, "pending", "not_observed", "awaiting_local_evidence")
            return _set_state(row, "missing_decision", "unknown", "no_snapshot_or_receipt")

        snapshot = load_snapshot(
            source, asof=day.isoformat(), slot=slot, ranking="R1",
            model_id=f"recommend_r1_{slot}",
        )
        started = _timestamp(snapshot["decision_started_at"], now=now)
        completed = _timestamp(snapshot["decision_completed_at"], now=now)
        _timestamp(snapshot["created_at"], now=now)
        if not start <= started <= completed < end:
            raise ValueError("snapshot decision is outside its slot window")
        feature_day = _canonical_day(snapshot["feature_date"])
        approved = (
            snapshot["snapshot_schema"] == SNAPSHOT_SCHEMA_VERSION
            and snapshot["rank_basis"] == APPROVED_RANK_BASIS
            and snapshot["score_schema_version"] == APPROVED_SCORE_SCHEMA
            and snapshot["rule_version"] == APPROVED_RULE_VERSION
        )
        row.update(
            artifact_state="valid",
            snapshot_state="valid",
            candidate_count=len(snapshot["top3"]),
            snapshot_id=snapshot["snapshot_id"],
            decision_started_at=started.isoformat(),
            decision_completed_at=completed.isoformat(),
            decision_age_seconds=(now - completed).total_seconds(),
            feature_row_date=feature_day.isoformat(),
            nominal_input_candle_date=(feature_day - timedelta(days=1)).isoformat(),
            approval_identity_matches=approved,
        )
        if not approved:
            row["artifact_state"] = "invalid"
            return _set_state(
                row, "invalid_evidence", "unknown", "unsupported_current_r1_identity"
            )
        # Never rewrite snapshot_path to make a relocated receipt pass validation.
        attempt = inspect_delivery_attempt(snapshot, receipt_root=receipt_root, now=now)
        row.update(
            attempt_state=attempt["state"],
            latest_attempt_id=attempt["latest_attempt_id"],
            attempt_count=attempt["attempt_count"],
            attempted_at=attempt["attempted_at"],
        )
        if attempt["state"] == "invalid":
            row["artifact_state"] = "invalid"
            return _set_state(row, "invalid_evidence", "unknown", "attempt_journal_validation_failed")
        result = read_delivery_receipt(snapshot, root=receipt_root)
        confirmed_attempt = inspect_delivery_attempt(snapshot, receipt_root=receipt_root, now=now)
        if confirmed_attempt != attempt:
            # No writer lock is taken. If publication overlapped this read,
            # report uncertainty rather than combining an old receipt with a
            # new attempt. A later, stable observation can resolve it.
            row.update(
                attempt_state=confirmed_attempt["state"],
                latest_attempt_id=confirmed_attempt["latest_attempt_id"],
                attempt_count=confirmed_attempt["attempt_count"],
                attempted_at=confirmed_attempt["attempted_at"],
            )
            if confirmed_attempt["state"] == "invalid":
                row["artifact_state"] = "invalid"
                return _set_state(row, "invalid_evidence", "unknown", "attempt_journal_validation_failed")
            return _set_state(row, "delivery_uncertain", "uncertain", "delivery_evidence_changed_during_read")
        if result is None:
            row["receipt_state"] = "missing"
            if attempt["state"] == "pending":
                return _set_state(row, "delivery_uncertain", "uncertain", "unresolved_delivery_attempt")
            if now < end:
                return _set_state(row, "pending", "not_observed", "awaiting_receipt")
            return _set_state(
                row, "delivery_uncertain", "uncertain", "receipt_missing_after_deadline"
            )
        row["receipt_state"] = "valid"
        attempted = _timestamp(result["attempted_at"], now=now)
        _timestamp(result["recorded_at"], now=now)
        if attempted < completed:
            raise ValueError("delivery attempt precedes snapshot completion")
        for message in result.get("telegram_messages", []):
            _timestamp(message["server_date"], now=now)
        if result["sent_at"] is not None:
            sent_at = _timestamp(result["sent_at"], now=now)
            row.update(
                sent_at=sent_at.isoformat(),
                delivery_age_seconds=(now - sent_at).total_seconds(),
            )
        if attempt["state"] == "pending":
            # The receipt may describe an earlier rejected attempt, not the
            # latest request. Never turn pending evidence into retry permission.
            return _set_state(row, "delivery_uncertain", "uncertain", "unresolved_delivery_attempt")
        if day < RECEIPT_INTEGRITY_ACTIVATION_DATE:
            return _set_state(
                row, "delivery_uncertain", "uncertain", "legacy_local_receipt_only"
            )
        chunks = result["telegram_messages"]
        row.update(delivered_chunk_count=len(chunks), expected_chunk_count=result["chunk_count"])
        if result["delivery_ok"]:
            state = "delivered_candidates" if row["candidate_count"] else "delivered_empty"
            return _set_state(row, state, "server_accepted", "validated_server_receipt")
        if chunks or telegram_error_is_ambiguous(result.get("error")):
            return _set_state(
                row, "delivery_uncertain", "uncertain", "partial_or_ambiguous_delivery"
            )
        return _set_state(row, "not_delivered", "failed", "validated_rejection_without_chunks")
    except (OSError, SnapshotError, DeliveryReceiptError, ValueError, TypeError, KeyError) as exc:
        row["artifact_state"] = "invalid"
        # Do not expose arbitrary receipt contents or exception text in the report.
        row["validation_error_type"] = type(exc).__name__
        return _set_state(row, "invalid_evidence", "unknown", "evidence_validation_failed")


def build_recommendation_status(
    asof: str,
    *,
    now: datetime,
    snapshot_root: str | Path | None = None,
    receipt_root: str | Path | None = None,
) -> dict[str, Any]:
    """Report both R1 slots without creating files or contacting any service."""
    day = _canonical_day(asof)
    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be a timezone-aware datetime")
    observed = now.astimezone(KST)
    snapshots = Path(snapshot_root) if snapshot_root is not None else DEFAULT_SNAPSHOT_ROOT
    receipts = Path(receipt_root) if receipt_root is not None else DEFAULT_RECEIPT_ROOT
    slots = {
        slot: _slot_status(day, slot, observed, snapshots, receipts)
        for slot in SLOTS
    }
    return {
        "schema": REPORT_SCHEMA,
        "asof": day.isoformat(),
        "observed_at": observed.isoformat(),
        "evidence_scope": "local_snapshot_receipt_and_attempt_journal_only",
        "attention_required": any(row["state"] in ATTENTION_STATES for row in slots.values()),
        "slots": slots,
        "notices": list(NOTICES),
    }


def format_recommendation_status(report: dict[str, Any]) -> str:
    """Human-readable view of the same evidence, never a new recommendation."""
    labels = {
        "waiting": "발송 가능 시간 전",
        "pending": "시간창 내 증거 대기 (실행 중 여부는 확인하지 않음)",
        "delivered_candidates": "후보 메시지 서버 수락 확인",
        "delivered_empty": "후보 0개 메시지 서버 수락 확인",
        "not_delivered": "실패 영수증 확인 (수락된 메시지 없음)",
        "delivery_uncertain": "전달 여부 확인 필요",
        "invalid_evidence": "증거 검증 실패",
        "missing_decision": "판단·전달 증거 없음",
    }
    lines = [f"R1 상태 {report['asof']} / 확인 시각 {report['observed_at']}"]
    for slot, row in report["slots"].items():
        count = "확인 불가" if row["candidate_count"] is None else str(row["candidate_count"])
        lines.append(f"- {slot}: {labels[row['state']]} / 저장 후보 수 {count} / {row['reason']}")
        if row["sent_at"]:
            lines.append(f"  서버 수락 시각: {row['sent_at']}")
        if row["feature_row_date"]:
            lines.append(
                f"  피처 행: {row['feature_row_date']} / 통상 입력봉: "
                f"{row['nominal_input_candle_date']} / 실제 마지막 입력봉: 확인 불가"
            )
    lines.extend(f"※ {notice}" for notice in report["notices"])
    return "\n".join(lines) + "\n"

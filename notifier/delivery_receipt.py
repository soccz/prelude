"""추천 레이더 Telegram delivery 결과의 원자적 sidecar 기록.

시그널 snapshot이나 shadow ledger를 수정하지 않고, 알림 계층의 전달 시도 결과만
``output/recommend_receipts`` 아래에 남긴다.
"""
from __future__ import annotations

import hashlib
import re
from contextlib import contextmanager
from datetime import date, datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from typing import Iterator

from notifier.telegram import (
    TelegramSendResult,
    telegram_error_is_ambiguous,
    validate_telegram_send_result,
)
from ops.artifact_provenance import (
    ArtifactValidationError,
    atomic_write_json,
    manifest_digest_matches,
    strict_json_object,
    with_manifest_digest,
)
from ops.file_lock import FileLockError, file_lock

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RECEIPT_ROOT = _ROOT / "output" / "recommend_receipts"
RECEIPT_SCHEMA_VERSION = "recommend_delivery_receipt.v1"
RECEIPT_INTEGRITY_ACTIVATION_DATE = date(2026, 7, 27)
_INTEGRITY_FIELD = "integrity_sha256"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_RECEIPT_RANKING_RE = re.compile(
    r"(?:R1|R2|A1|champion_change_[0-9a-f]{1,64})"
)
_SERVER_CLOCK_SKEW_SECONDS = 5
_KST = timezone(timedelta(hours=9))
_LIVE_SEND_WINDOWS = {
    "preopen": (
        datetime.min.time().replace(hour=8, minute=45),
        datetime.min.time().replace(hour=9),
    ),
    "open": (
        datetime.min.time().replace(hour=9),
        datetime.min.time().replace(hour=9, minute=21),
    ),
}


class DeliveryReceiptError(RuntimeError):
    """Receipt가 손상됐거나 요청한 snapshot과 일치하지 않을 때 발생."""


def _snapshot_day(snapshot: dict, path: Path) -> date:
    raw = snapshot.get("asof")
    if not isinstance(raw, str):
        raise DeliveryReceiptError(f"snapshot asof is invalid ({path})")
    try:
        parsed = date.fromisoformat(raw)
    except ValueError as exc:
        raise DeliveryReceiptError(
            f"snapshot asof is invalid: {raw!r} ({path})"
        ) from exc
    if parsed.isoformat() != raw:
        raise DeliveryReceiptError(
            f"snapshot asof is not canonical: {raw!r} ({path})"
        )
    return parsed


def receipt_path(snapshot: dict, *, root: str | Path | None = None) -> Path:
    """Return a receipt path only from canonical, filename-safe identity data."""
    base = Path(root) if root is not None else DEFAULT_RECEIPT_ROOT
    if not isinstance(snapshot, dict):
        raise DeliveryReceiptError("snapshot must be an object")
    decision_day = _snapshot_day(snapshot, base)
    slot = snapshot.get("slot")
    if not isinstance(slot, str) or slot not in _LIVE_SEND_WINDOWS:
        raise DeliveryReceiptError(f"invalid snapshot slot: {slot!r}")

    model = snapshot.get("model")
    if model is None:
        model = {}
    if not isinstance(model, dict):
        raise DeliveryReceiptError("snapshot model must be an object")
    ranking = model.get("ranking", "R1")
    if (
        not isinstance(ranking, str)
        or _RECEIPT_RANKING_RE.fullmatch(ranking) is None
    ):
        raise DeliveryReceiptError(
            f"invalid snapshot ranking: {ranking!r}"
        )

    request = snapshot.get("request")
    if request is None:
        request = {}
    if not isinstance(request, dict):
        raise DeliveryReceiptError("snapshot request must be an object")
    limit_markets = request.get("limit_markets")
    if limit_markets is None:
        suffix = ""
    else:
        if (
            isinstance(limit_markets, bool)
            or not isinstance(limit_markets, int)
            or limit_markets <= 0
        ):
            raise DeliveryReceiptError(
                f"invalid snapshot limit_markets: {limit_markets!r}"
            )
        suffix = f".limit{limit_markets}"
    return (
        base
        / decision_day.isoformat()
        / f"{slot}_{ranking.lower()}{suffix}.json"
    )


def _aware_timestamp(document: dict, field: str, path: Path) -> datetime:
    value = document.get(field)
    if not isinstance(value, str) or not value.strip():
        raise DeliveryReceiptError(
            f"receipt {field} must be a non-empty ISO-8601 string ({path})"
        )
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise DeliveryReceiptError(
            f"receipt {field} is not ISO-8601: {value!r} ({path})"
        ) from exc
    if parsed.tzinfo is None:
        raise DeliveryReceiptError(
            f"receipt {field} must be timezone-aware: {value!r} ({path})"
        )
    return parsed


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def validate_telegram_transport_evidence(
    document: dict,
    *,
    decision_day: date,
    live_start: dt_time,
    live_end: dt_time,
    delivery_ok: bool,
    attempted_at: datetime,
    sent_at: datetime | None,
    recorded_at: datetime,
    path: Path,
) -> None:
    message_sha256 = document.get("message_sha256")
    chat_id_sha256 = document.get("chat_id_sha256")
    chunk_count = document.get("chunk_count")
    messages = document.get("telegram_messages")
    if (
        not isinstance(message_sha256, str)
        or _SHA256_RE.fullmatch(message_sha256) is None
    ):
        raise DeliveryReceiptError(f"invalid receipt message_sha256 ({path})")
    if (
        chat_id_sha256 is not None
        and (
            not isinstance(chat_id_sha256, str)
            or _SHA256_RE.fullmatch(chat_id_sha256) is None
        )
    ):
        raise DeliveryReceiptError(f"invalid receipt chat_id_sha256 ({path})")
    if (
        not isinstance(chunk_count, int)
        or isinstance(chunk_count, bool)
        or chunk_count <= 0
    ):
        raise DeliveryReceiptError(f"invalid receipt chunk_count ({path})")
    if not isinstance(messages, list):
        raise DeliveryReceiptError(
            f"receipt telegram_messages must be a list ({path})"
        )
    if delivery_ok:
        if chat_id_sha256 is None or len(messages) != chunk_count:
            raise DeliveryReceiptError(
                f"successful receipt transport evidence is incomplete ({path})"
            )
    elif len(messages) >= chunk_count:
        raise DeliveryReceiptError(
            f"failed receipt cannot claim every chunk delivered ({path})"
        )
    if messages and chat_id_sha256 is None:
        raise DeliveryReceiptError(
            f"delivered chunks require a chat identity ({path})"
        )

    ids: set[int] = set()
    observed_dates: list[datetime] = []
    attempted_kst = attempted_at.astimezone(_KST)
    attempted_wall = attempted_kst.timetz().replace(tzinfo=None)
    if (
        attempted_kst.date() != decision_day
        or not live_start <= attempted_wall < live_end
    ):
        raise DeliveryReceiptError(
            f"receipt attempted_at outside live send window ({path})"
        )
    attempted_floor = attempted_at.astimezone(timezone.utc).replace(
        microsecond=0
    )
    recorded_limit = recorded_at.astimezone(timezone.utc) + timedelta(
        seconds=_SERVER_CLOCK_SKEW_SECONDS
    )
    for index, item in enumerate(messages):
        if not isinstance(item, dict) or set(item) != {
            "message_id",
            "server_date",
            "text_sha256",
        }:
            raise DeliveryReceiptError(
                f"receipt telegram_messages[{index}] fields mismatch ({path})"
            )
        message_id = item.get("message_id")
        if (
            not isinstance(message_id, int)
            or isinstance(message_id, bool)
            or message_id <= 0
            or message_id in ids
        ):
            raise DeliveryReceiptError(
                f"invalid/duplicate Telegram message_id ({path})"
            )
        ids.add(message_id)
        text_sha256 = item.get("text_sha256")
        if (
            not isinstance(text_sha256, str)
            or _SHA256_RE.fullmatch(text_sha256) is None
        ):
            raise DeliveryReceiptError(
                f"invalid Telegram text_sha256 ({path})"
            )
        server_date = _aware_timestamp(item, "server_date", path)
        if (
            server_date.utcoffset() != timezone.utc.utcoffset(None)
            or server_date.isoformat() != item["server_date"]
        ):
            raise DeliveryReceiptError(
                f"Telegram server_date must be canonical UTC ({path})"
            )
        server_utc = server_date.astimezone(timezone.utc)
        if not attempted_floor <= server_utc <= recorded_limit:
            raise DeliveryReceiptError(
                f"Telegram server_date chronology is invalid ({path})"
            )
        server_kst = server_utc.astimezone(_KST)
        wall = server_kst.timetz().replace(tzinfo=None)
        if (
            server_kst.date() != decision_day
            or not live_start <= wall < live_end
        ):
            raise DeliveryReceiptError(
                f"Telegram server_date outside live send window ({path})"
            )
        if observed_dates and server_utc < observed_dates[-1]:
            raise DeliveryReceiptError(
                f"Telegram server dates are out of chunk order ({path})"
            )
        observed_dates.append(server_utc)

    if delivery_ok:
        if sent_at is None or sent_at != max(observed_dates):
            raise DeliveryReceiptError(
                f"successful receipt sent_at is not server-bound ({path})"
            )
    elif sent_at is not None:
        raise DeliveryReceiptError(
            f"failed receipt must not include sent_at ({path})"
        )


def _validate_document(document: dict, snapshot: dict, path: Path) -> None:
    decision_day = _snapshot_day(snapshot, path)
    expected_fields = {
        "schema",
        "snapshot_id",
        "snapshot_path",
        "asof",
        "slot",
        "model_id",
        "attempted_at",
        "sent_at",
        "delivery_ok",
        "error",
        "recorded_at",
    }
    if decision_day >= RECEIPT_INTEGRITY_ACTIVATION_DATE:
        expected_fields.update(
            {
                _INTEGRITY_FIELD,
                "message_sha256",
                "chat_id_sha256",
                "chunk_count",
                "telegram_messages",
            }
        )
    actual_fields = set(document)
    if actual_fields != expected_fields:
        raise DeliveryReceiptError(
            "receipt fields mismatch: "
            f"missing={sorted(expected_fields - actual_fields)} "
            f"unexpected={sorted(actual_fields - expected_fields)} ({path})"
        )
    if (
        decision_day >= RECEIPT_INTEGRITY_ACTIVATION_DATE
        and not manifest_digest_matches(
            document,
            digest_key=_INTEGRITY_FIELD,
        )
    ):
        raise DeliveryReceiptError(
            f"receipt outer integrity mismatch ({path})"
        )
    if document.get("schema") != RECEIPT_SCHEMA_VERSION:
        raise DeliveryReceiptError(
            f"unsupported receipt schema: {document.get('schema')!r} ({path})"
        )
    for field in ("snapshot_id", "asof", "slot"):
        expected = snapshot.get(field)
        if document.get(field) != expected:
            raise DeliveryReceiptError(
                f"receipt identity mismatch {field}: expected={expected!r} "
                f"actual={document.get(field)!r} ({path})"
            )
    for field, expected in (
        ("snapshot_path", snapshot.get("snapshot_path")),
        ("model_id", (snapshot.get("model") or {}).get("id")),
    ):
        if document.get(field) != expected:
            raise DeliveryReceiptError(
                f"receipt provenance mismatch {field}: expected={expected!r} "
                f"actual={document.get(field)!r} ({path})"
            )
    delivery_ok = document.get("delivery_ok")
    if not isinstance(delivery_ok, bool):
        raise DeliveryReceiptError(
            f"receipt delivery_ok must be bool, got {delivery_ok!r} ({path})"
        )

    attempted_at = _aware_timestamp(document, "attempted_at", path)
    recorded_at = _aware_timestamp(document, "recorded_at", path)
    sent_at_value = document.get("sent_at")
    sent_at = (
        _aware_timestamp(document, "sent_at", path)
        if sent_at_value is not None
        else None
    )
    if delivery_ok and sent_at is None:
        raise DeliveryReceiptError(
            f"successful receipt must include sent_at ({path})"
        )
    if not delivery_ok and sent_at is not None:
        raise DeliveryReceiptError(
            f"failed receipt must not include sent_at ({path})"
        )
    if attempted_at > recorded_at:
        raise DeliveryReceiptError(
            f"receipt attempted_at is after recorded_at ({path})"
        )
    if (
        decision_day < RECEIPT_INTEGRITY_ACTIVATION_DATE
        and sent_at is not None
        and not (attempted_at <= sent_at <= recorded_at)
    ):
        raise DeliveryReceiptError(
            f"receipt sent_at chronology is invalid ({path})"
        )
    error = document.get("error")
    if error is not None and not isinstance(error, str):
        raise DeliveryReceiptError(
            f"receipt error must be string or null, got {error!r} ({path})"
        )
    if decision_day >= RECEIPT_INTEGRITY_ACTIVATION_DATE:
        if delivery_ok and error is not None:
            raise DeliveryReceiptError(
                f"successful receipt must not include an error ({path})"
            )
        if not delivery_ok and (
            not isinstance(error, str) or not error.strip()
        ):
            raise DeliveryReceiptError(
                f"failed receipt must include a non-empty error ({path})"
            )
        try:
            live_start, live_end = _LIVE_SEND_WINDOWS[str(snapshot.get("slot"))]
        except KeyError as exc:
            raise DeliveryReceiptError(
                f"unsupported receipt live slot: "
                f"{snapshot.get('slot')!r} ({path})"
            ) from exc
        validate_telegram_transport_evidence(
            document,
            decision_day=decision_day,
            live_start=live_start,
            live_end=live_end,
            delivery_ok=delivery_ok,
            attempted_at=attempted_at,
            sent_at=sent_at,
            recorded_at=recorded_at,
            path=path,
        )


def _atomic_write(path: Path, document: dict) -> None:
    try:
        atomic_write_json(path, document)
    except (OSError, TypeError, ValueError) as exc:
        raise DeliveryReceiptError(
            f"receipt cannot be written safely: {path}"
        ) from exc


@contextmanager
def _exclusive_receipt_lock(path: Path) -> Iterator[None]:
    """동일 receipt의 read-check-write 전체를 프로세스 간 직렬화한다."""
    lock_path = path.with_name(f".{path.name}.lock")
    try:
        with file_lock(lock_path):
            yield
    except FileLockError as exc:
        raise DeliveryReceiptError(
            f"receipt lock cannot be opened safely: {lock_path}"
        ) from exc


def write_delivery_receipt(
    snapshot: dict,
    *,
    delivery_ok: bool,
    attempted_at: str,
    sent_at: str | None,
    error: str | None = None,
    telegram_result: TelegramSendResult | None = None,
    message: str | None = None,
    root: str | Path | None = None,
) -> Path:
    """실제 발송 결과를 sidecar에 원자 저장한다.

    동일 snapshot이 한 번이라도 성공적으로 전달됐다면 이후 재시도 실패가 그 성공
    증거를 덮어쓰지 않는다. 최초 성공 시각은 실행 귀속의 보수적 기준이므로 유지한다.
    """
    if not isinstance(delivery_ok, bool):
        raise DeliveryReceiptError(
            f"delivery_ok must be bool, got {delivery_ok!r}"
        )
    path = receipt_path(snapshot, root=root)
    with _exclusive_receipt_lock(path):
        existing = read_delivery_receipt(snapshot, root=root)
        if existing and existing["delivery_ok"]:
            return path
        if existing and (
            existing.get("telegram_messages")
            or telegram_error_is_ambiguous(existing.get("error"))
        ):
            raise DeliveryReceiptError(
                "partial/ambiguous Telegram delivery requires reconciliation; "
                "automatic overwrite/retry is unsafe"
            )

        document = {
            "schema": RECEIPT_SCHEMA_VERSION,
            "snapshot_id": snapshot.get("snapshot_id"),
            "snapshot_path": snapshot.get("snapshot_path"),
            "asof": snapshot.get("asof"),
            "slot": snapshot.get("slot"),
            "model_id": (snapshot.get("model") or {}).get("id"),
            "attempted_at": attempted_at,
            "sent_at": sent_at,
            "delivery_ok": delivery_ok,
            "error": error,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        decision_day = _snapshot_day(snapshot, path)
        if decision_day >= RECEIPT_INTEGRITY_ACTIVATION_DATE:
            if telegram_result is None or not isinstance(message, str):
                raise DeliveryReceiptError(
                    "post-activation receipt requires Telegram server evidence"
                )
            if telegram_result.dry_run:
                raise DeliveryReceiptError(
                    "dry-run transport cannot create a delivery receipt"
                )
            try:
                validate_telegram_send_result(telegram_result, message)
            except ValueError as exc:
                raise DeliveryReceiptError(
                    "Telegram server evidence does not match message"
                ) from exc
            if telegram_result.delivery_ok != delivery_ok:
                raise DeliveryReceiptError(
                    "delivery_ok does not match Telegram server evidence"
                )
            if telegram_result.message_sha256 != _sha256_text(message):
                raise DeliveryReceiptError(
                    "message does not match Telegram server evidence"
                )
            evidence_error = telegram_result.error
            if error != evidence_error:
                raise DeliveryReceiptError(
                    "error does not match Telegram server evidence"
                )
            document.update(
                {
                    "message_sha256": telegram_result.message_sha256,
                    "chat_id_sha256": telegram_result.chat_id_sha256,
                    "chunk_count": telegram_result.chunk_count,
                    "telegram_messages": [
                        item.as_dict()
                        for item in telegram_result.telegram_messages
                    ],
                }
            )
            document = with_manifest_digest(
                document,
                digest_key=_INTEGRITY_FIELD,
            )
        _validate_document(document, snapshot, path)
        _atomic_write(path, document)
    return path


def read_delivery_receipt(
    snapshot: dict,
    *,
    root: str | Path | None = None,
) -> dict | None:
    """snapshot의 검증된 delivery receipt를 읽는다.

    발송 시도가 없어서 파일이 없는 것은 정상이며 ``None``을 반환한다. 파일이 있는데
    schema/identity/type이 맞지 않으면 잘못된 실행 시각을 ledger에 넣지 않도록 예외를
    발생시킨다.
    """
    path = receipt_path(snapshot, root=root)
    if path.is_symlink():
        raise DeliveryReceiptError(
            f"receipt must be a regular non-symlink file: {path}"
        )
    if not path.exists():
        return None
    try:
        document = strict_json_object(path)
    except (OSError, ArtifactValidationError) as exc:
        raise DeliveryReceiptError(f"receipt read failed: {path}: {exc}") from exc

    _validate_document(document, snapshot, path)

    result = dict(document)
    result["receipt_path"] = str(path)
    return result

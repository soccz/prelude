"""텔레그램 봇 — 메시지 전송.

설정 (.env 또는 환경변수):
  TELEGRAM_BOT_TOKEN
  TELEGRAM_CHAT_ID

사용:
    from notifier.telegram import send_telegram
    send_telegram("hello")
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import requests  # type: ignore[import-untyped]

# Load only the three supported runtime keys.  ``.env`` is configuration
# data, never shell source, and an unsafe existing file must fail closed.
from ops.runtime_env import load_runtime_env

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RUNTIME_ENV = _PROJECT_ROOT / ".env"
if _RUNTIME_ENV.exists() or _RUNTIME_ENV.is_symlink():
    load_runtime_env(_RUNTIME_ENV)

logger = logging.getLogger("telegram")

_MAX_CHUNK_UTF16_UNITS = 4000
_REQUEST_TIMEOUT_SECONDS = 10.0
_MAX_RETRY_SLEEP_SECONDS = 30.0
AMBIGUOUS_DELIVERY_ERROR_PREFIX = "ambiguous Telegram acceptance: "


# ============================================================================
# 설정
# ============================================================================
def get_token() -> Optional[str]:
    return os.environ.get("TELEGRAM_BOT_TOKEN")


def get_chat_id() -> Optional[str]:
    return os.environ.get("TELEGRAM_CHAT_ID")


def _safe_error(e: Exception, token: Optional[str]) -> str:
    text = str(e)
    if token:
        text = text.replace(token, "<telegram-token>")
    return text


class _TelegramResponseError(RuntimeError):
    """HTTP 성공처럼 보여도 Bot API 응답 계약이 실패했을 때 발생."""

    def __init__(
        self,
        message: str,
        *,
        acceptance_ambiguous: bool = False,
    ) -> None:
        super().__init__(message)
        self.acceptance_ambiguous = acceptance_ambiguous


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TelegramServerMessage:
    """Bot API가 실제로 수락했다고 증명한 한 개 chunk."""

    message_id: int
    server_date: str
    text_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "message_id": self.message_id,
            "server_date": self.server_date,
            "text_sha256": self.text_sha256,
        }


@dataclass(frozen=True)
class TelegramSendResult:
    """상세 전송 결과.

    ``telegram_messages``는 요청 순서대로 검증을 마친 chunk만 담는다. 따라서
    ``delivery_ok=False``인데 목록이 비어 있지 않으면 부분 전달이다.
    """

    delivery_ok: bool
    message_sha256: str
    chunk_count: int
    chat_id_sha256: str | None
    telegram_messages: tuple[TelegramServerMessage, ...]
    error: str | None
    dry_run: bool = False


def _validated_server_message(
    response: requests.Response,
    *,
    expected_text: str,
    expected_chat_id: str,
    parse_mode: str | None,
    deadline: datetime | None,
) -> TelegramServerMessage:
    """Bot API 성공 응답을 요청한 chunk/대상과 대조해 검증한다."""
    response.raise_for_status()
    try:
        payload: Any = response.json()
    except (TypeError, ValueError) as exc:
        raise _TelegramResponseError(
            "telegram response is not valid JSON",
            acceptance_ambiguous=True,
        ) from exc
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        raise _TelegramResponseError(
            "telegram response ok is not true",
            acceptance_ambiguous=not (
                isinstance(payload, dict)
                and payload.get("ok") is False
            ),
        )
    result = payload.get("result")
    if not isinstance(result, dict):
        raise _TelegramResponseError(
            "telegram response has no result object",
            acceptance_ambiguous=True,
        )
    message_id = result.get("message_id")
    if (
        not isinstance(message_id, int)
        or isinstance(message_id, bool)
        or message_id <= 0
    ):
        raise _TelegramResponseError(
            "telegram response has no valid message_id",
            acceptance_ambiguous=True,
        )
    server_epoch = result.get("date")
    if (
        not isinstance(server_epoch, int)
        or isinstance(server_epoch, bool)
        or server_epoch <= 0
    ):
        raise _TelegramResponseError(
            "telegram response has no valid server date",
            acceptance_ambiguous=True,
        )
    try:
        server_date = datetime.fromtimestamp(server_epoch, tz=timezone.utc)
    except (OverflowError, OSError, ValueError) as exc:
        raise _TelegramResponseError(
            "telegram response server date is out of range",
            acceptance_ambiguous=True,
        ) from exc
    if (
        deadline is not None
        and server_date >= deadline.astimezone(timezone.utc)
    ):
        raise _TelegramResponseError(
            "telegram response server date reached or crossed send deadline",
            acceptance_ambiguous=True,
        )
    response_text = result.get("text")
    # With parse_mode Telegram returns rendered text plus entities, not the
    # original markup bytes. The successful response still binds the request
    # through this call's payload, chat id, message id, and server date; exact
    # echo comparison is valid only for plain-text sends.  Telegram strips
    # leading/trailing whitespace before echoing, so a chunk that ends in a
    # newline comes back stripped even though it was accepted verbatim —
    # treat the server-trimmed echo as a match (2026-07-27 heartbeat false
    # ambiguous), while any interior mismatch stays ambiguous.
    if (
        (
            parse_mode is None
            and response_text != expected_text
            and (
                not isinstance(response_text, str)
                or response_text != expected_text.strip()
            )
        )
        or (parse_mode is not None and not isinstance(response_text, str))
    ):
        raise _TelegramResponseError(
            "telegram response text does not match requested chunk",
            acceptance_ambiguous=True,
        )
    chat = result.get("chat")
    actual_chat_id = chat.get("id") if isinstance(chat, dict) else None
    if (
        isinstance(actual_chat_id, bool)
        or not isinstance(actual_chat_id, (int, str))
        or str(actual_chat_id) != expected_chat_id
    ):
        raise _TelegramResponseError(
            "telegram response chat id does not match requested chat",
            acceptance_ambiguous=True,
        )
    return TelegramServerMessage(
        message_id=message_id,
        server_date=server_date.isoformat(),
        text_sha256=_sha256_text(expected_text),
    )


def _telegram_retry_after(response: requests.Response | None) -> float | None:
    """429 응답의 Bot API parameters.retry_after를 안전하게 읽는다."""
    if response is None or getattr(response, "status_code", None) != 429:
        return None
    try:
        payload = response.json()
        parameters = payload.get("parameters") if isinstance(payload, dict) else None
        value = parameters.get("retry_after") if isinstance(parameters, dict) else None
        if value is None:
            return None
        delay = float(value)
    except (AttributeError, TypeError, ValueError):
        return None
    if not math.isfinite(delay) or delay < 0:
        return None
    return delay


def _http_rejection_is_terminal(
    response: requests.Response | None,
) -> bool:
    """Return whether Telegram explicitly rejected a request permanently.

    A 429 response is a rate-limit rejection that is safe to retry. HTTP 408
    is excluded because acceptance is ambiguous, so neither an in-call retry
    nor a later automatic receipt overwrite is safe. Other 4xx responses
    describe an invalid request, credential, or destination and retrying the
    same payload only delays the operational failure signal.
    """
    status_code = getattr(response, "status_code", None)
    return (
        isinstance(status_code, int)
        and 400 <= status_code < 500
        and status_code not in {408, 429}
    )


def telegram_error_is_ambiguous(error: object) -> bool:
    return (
        isinstance(error, str)
        and error.startswith(AMBIGUOUS_DELIVERY_ERROR_PREFIX)
    )


def _acceptance_is_ambiguous(
    error: Exception,
    response: requests.Response | None,
) -> bool:
    if isinstance(error, _TelegramResponseError):
        return error.acceptance_ambiguous
    if response is None:
        return True
    status_code = getattr(response, "status_code", None)
    return (
        isinstance(status_code, int)
        and (status_code == 408 or status_code >= 500)
    )


def _transport_error_text(
    error: Exception,
    *,
    token: str | None,
    response: requests.Response | None,
) -> str:
    detail = f"{type(error).__name__}: {_safe_error(error, token)}"
    if _acceptance_is_ambiguous(error, response):
        return f"{AMBIGUOUS_DELIVERY_ERROR_PREFIX}{detail}"
    return detail


def _utf16_units(text: str) -> int:
    """Telegram 길이 계산에 쓰는 UTF-16 code-unit 수."""
    return sum(2 if ord(char) > 0xFFFF else 1 for char in text)


def _split_message_utf16(
    message: str,
    *,
    max_units: int = _MAX_CHUNK_UTF16_UNITS,
) -> list[str]:
    """Surrogate pair를 자르지 않고 UTF-16 단위 상한 이하로 분할한다."""
    if max_units < 2:
        raise ValueError("telegram chunk limit must be at least 2 UTF-16 units")

    chunks: list[str] = []
    current: list[str] = []
    current_units = 0
    for char in message:
        char_units = 2 if ord(char) > 0xFFFF else 1
        if current and current_units + char_units > max_units:
            chunks.append("".join(current))
            current = []
            current_units = 0
        current.append(char)
        current_units += char_units
    if current:
        chunks.append("".join(current))
    return chunks


def validate_telegram_send_result(
    send_result: TelegramSendResult,
    message: str,
) -> None:
    """상세 결과가 원문과 각 UTF-16 chunk에 정확히 결합됐는지 검증한다."""
    if not isinstance(send_result, TelegramSendResult):
        raise ValueError("Telegram send result has an invalid type")
    if send_result.dry_run:
        raise ValueError("dry-run result is not delivery evidence")
    chunks = _split_message_utf16(message)
    if (
        send_result.message_sha256 != _sha256_text(message)
        or not isinstance(send_result.chunk_count, int)
        or isinstance(send_result.chunk_count, bool)
        or send_result.chunk_count != len(chunks)
        or send_result.chunk_count <= 0
    ):
        raise ValueError("Telegram send result message contract mismatch")
    if send_result.chat_id_sha256 is not None and (
        len(send_result.chat_id_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in send_result.chat_id_sha256
        )
    ):
        raise ValueError("Telegram send result chat digest is invalid")
    delivered = send_result.telegram_messages
    if not isinstance(delivered, tuple):
        raise ValueError("Telegram send result messages must be immutable")
    if send_result.delivery_ok:
        if (
            len(delivered) != len(chunks)
            or send_result.chat_id_sha256 is None
            or send_result.error is not None
        ):
            raise ValueError("Telegram success evidence is incomplete")
    elif (
        len(delivered) >= len(chunks)
        or not isinstance(send_result.error, str)
        or not send_result.error.strip()
    ):
        raise ValueError("Telegram failure evidence is inconsistent")
    if delivered and send_result.chat_id_sha256 is None:
        raise ValueError("Telegram delivered chunks have no chat identity")

    message_ids: set[int] = set()
    prior_date: datetime | None = None
    for index, item in enumerate(delivered):
        if not isinstance(item, TelegramServerMessage):
            raise ValueError("Telegram send result message has invalid type")
        if (
            not isinstance(item.message_id, int)
            or isinstance(item.message_id, bool)
            or item.message_id <= 0
            or item.message_id in message_ids
        ):
            raise ValueError("Telegram send result message_id is invalid")
        message_ids.add(item.message_id)
        if item.text_sha256 != _sha256_text(chunks[index]):
            raise ValueError("Telegram send result chunk digest mismatch")
        try:
            server_date = datetime.fromisoformat(item.server_date)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Telegram send result server date is invalid"
            ) from exc
        if (
            server_date.tzinfo is None
            or server_date.utcoffset() != timezone.utc.utcoffset(None)
            or server_date.isoformat() != item.server_date
            or (prior_date is not None and server_date < prior_date)
        ):
            raise ValueError(
                "Telegram send result server chronology is invalid"
            )
        prior_date = server_date


def _remaining_seconds(
    deadline: datetime,
    clock: Callable[[], datetime],
) -> float:
    observed = clock()
    if (
        not isinstance(observed, datetime)
        or observed.tzinfo is None
        or observed.utcoffset() is None
    ):
        raise ValueError("telegram deadline clock must return an aware datetime")
    return (
        deadline.astimezone(timezone.utc)
        - observed.astimezone(timezone.utc)
    ).total_seconds()


def _log_terminal_failure(
    *,
    chunk_index: int,
    chunk_count: int,
    attempts: int,
    reason: str,
) -> None:
    delivered_chunks = chunk_index - 1
    if delivered_chunks:
        logger.error(
            "telegram partial delivery: %s/%s chunk(s) delivered; "
            "chunk %s stopped after %s attempt(s): %s",
            delivered_chunks,
            chunk_count,
            chunk_index,
            attempts,
            reason,
        )
    else:
        logger.error(
            "telegram send FAILED: chunk %s/%s stopped after "
            "%s attempt(s); no chunks delivered: %s",
            chunk_index,
            chunk_count,
            attempts,
            reason,
        )


# ============================================================================
# 전송
# ============================================================================
def send_telegram_with_receipt(
    message: str,
    chat_id: Optional[str] = None,
    token: Optional[str] = None,
    parse_mode: Optional[str] = None,
    retries: int = 3,
    backoff: float = 2.0,
    dry_run: bool = False,
    deadline: datetime | None = None,
    clock: Callable[[], datetime] | None = None,
) -> TelegramSendResult:
    """
    텔레그램 메시지를 전송하고 서버가 확인한 chunk 증거를 반환한다.

    parse_mode: 'Markdown' / 'HTML' / None (plain)
    retries: 실패 시 재시도
    deadline: 이 aware 시각 이후에는 새 HTTP attempt/retry를 시작하지 않음
    clock: deadline 검사용 aware clock (테스트 주입용)

    """
    message_for_digest = message if isinstance(message, str) else ""
    message_sha256 = _sha256_text(message_for_digest)
    chunks: list[str] = []
    resolved_chat_id: str | None = None

    def result(
        *,
        delivery_ok: bool,
        error: str | None,
        delivered: list[TelegramServerMessage] | None = None,
        is_dry_run: bool = False,
    ) -> TelegramSendResult:
        return TelegramSendResult(
            delivery_ok=delivery_ok,
            message_sha256=message_sha256,
            chunk_count=len(chunks),
            chat_id_sha256=(
                _sha256_text(resolved_chat_id)
                if resolved_chat_id is not None
                else None
            ),
            telegram_messages=tuple(delivered or ()),
            error=error,
            dry_run=is_dry_run,
        )

    if not isinstance(message, str) or not message.strip():
        logger.error("telegram message is empty — 전송 거부")
        return result(delivery_ok=False, error="telegram message is empty")
    if (
        not isinstance(retries, int)
        or isinstance(retries, bool)
        or retries <= 0
    ):
        logger.error("telegram retries must be a positive integer")
        return result(
            delivery_ok=False,
            error="telegram retries must be a positive integer",
        )
    try:
        backoff_value = float(backoff)
    except (TypeError, ValueError):
        logger.error("telegram backoff must be a finite non-negative number")
        return result(
            delivery_ok=False,
            error="telegram backoff must be a finite non-negative number",
        )
    if not math.isfinite(backoff_value) or backoff_value < 0:
        logger.error("telegram backoff must be a finite non-negative number")
        return result(
            delivery_ok=False,
            error="telegram backoff must be a finite non-negative number",
        )

    token = token or get_token()
    chat_id = chat_id or get_chat_id()
    resolved_chat_id = str(chat_id) if chat_id is not None else None
    chunks = _split_message_utf16(message)

    if dry_run:
        print("=" * 60)
        chat_status = "configured" if resolved_chat_id else "missing"
        print(f"[telegram dry-run] chat_id={chat_status}")
        print("-" * 60)
        print(message)
        print("=" * 60)
        return result(delivery_ok=True, error=None, is_dry_run=True)
    deadline_clock = clock or (lambda: datetime.now(timezone.utc))
    if deadline is not None:
        if (
            not isinstance(deadline, datetime)
            or deadline.tzinfo is None
            or deadline.utcoffset() is None
        ):
            logger.error("telegram deadline must be a timezone-aware datetime")
            return result(
                delivery_ok=False,
                error="telegram deadline must be timezone-aware",
            )
    if not token or not chat_id:
        logger.error("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID 미설정 — 전송 실패")
        return result(
            delivery_ok=False,
            error="telegram credentials are missing",
        )

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    # Bot API의 4096-char 상한에 안전 여유를 두되, Python code point 수가
    # 아닌 UTF-16 code unit으로 계산한다. Emoji(astral code point)는 2 unit.
    chunk_count = len(chunks)
    delivered: list[TelegramServerMessage] = []
    for chunk_index, chunk in enumerate(chunks, start=1):
        payload = {"chat_id": chat_id, "text": chunk}
        if parse_mode:
            payload["parse_mode"] = parse_mode

        for attempt in range(1, retries + 1):
            response: requests.Response | None = None
            request_timeout = _REQUEST_TIMEOUT_SECONDS
            if deadline is not None:
                try:
                    remaining = _remaining_seconds(deadline, deadline_clock)
                except (OverflowError, TypeError, ValueError) as exc:
                    _log_terminal_failure(
                        chunk_index=chunk_index,
                        chunk_count=chunk_count,
                        attempts=attempt - 1,
                        reason=f"invalid deadline clock: {exc}",
                    )
                    return result(
                        delivery_ok=False,
                        error=f"invalid deadline clock: {exc}",
                        delivered=delivered,
                    )
                if remaining <= 0:
                    _log_terminal_failure(
                        chunk_index=chunk_index,
                        chunk_count=chunk_count,
                        attempts=attempt - 1,
                        reason="send deadline reached",
                    )
                    return result(
                        delivery_ok=False,
                        error="send deadline reached",
                        delivered=delivered,
                    )
                request_timeout = min(request_timeout, remaining)
            try:
                response = requests.post(
                    url,
                    data=payload,
                    timeout=request_timeout,
                )
                server_message = _validated_server_message(
                    response,
                    expected_text=chunk,
                    expected_chat_id=str(chat_id),
                    parse_mode=parse_mode if parse_mode else None,
                    deadline=deadline,
                )
                if any(
                    prior.message_id == server_message.message_id
                    for prior in delivered
                ):
                    raise _TelegramResponseError(
                        "telegram response reused a message_id",
                        acceptance_ambiguous=True,
                    )
                if (
                    delivered
                    and datetime.fromisoformat(server_message.server_date)
                    < datetime.fromisoformat(delivered[-1].server_date)
                ):
                    raise _TelegramResponseError(
                        "telegram response server dates are out of chunk order",
                        acceptance_ambiguous=True,
                    )
                delivered.append(server_message)
                break
            except Exception as e:
                safe_reason = _safe_error(e, token)
                transport_error = _transport_error_text(
                    e,
                    token=token,
                    response=response,
                )
                logger.warning(
                    "telegram chunk %s/%s attempt %s/%s fail: %s",
                    chunk_index,
                    chunk_count,
                    attempt,
                    retries,
                    safe_reason,
                )
                if _acceptance_is_ambiguous(e, response):
                    _log_terminal_failure(
                        chunk_index=chunk_index,
                        chunk_count=chunk_count,
                        attempts=attempt,
                        reason="Telegram acceptance is ambiguous",
                    )
                    return result(
                        delivery_ok=False,
                        error=transport_error,
                        delivered=delivered,
                    )
                if isinstance(e, _TelegramResponseError):
                    _log_terminal_failure(
                        chunk_index=chunk_index,
                        chunk_count=chunk_count,
                        attempts=attempt,
                        reason="Bot API explicitly rejected the request",
                    )
                    return result(
                        delivery_ok=False,
                        error=transport_error,
                        delivered=delivered,
                    )
                if _http_rejection_is_terminal(response):
                    status_code = getattr(response, "status_code", "unknown")
                    reason = (
                        "Telegram permanently rejected the request "
                        f"(HTTP {status_code})"
                    )
                    _log_terminal_failure(
                        chunk_index=chunk_index,
                        chunk_count=chunk_count,
                        attempts=attempt,
                        reason=reason,
                    )
                    return result(
                        delivery_ok=False,
                        error=transport_error,
                        delivered=delivered,
                    )
                if attempt < retries:
                    retry_after = _telegram_retry_after(response)
                    try:
                        delay = (
                            retry_after
                            if retry_after is not None
                            else backoff_value ** attempt
                        )
                    except OverflowError:
                        delay = math.inf
                    if retry_after is not None:
                        logger.warning(
                            "telegram rate limited; Retry-After %.3fs 적용",
                            delay,
                        )
                    if (
                        not math.isfinite(delay)
                        or delay > _MAX_RETRY_SLEEP_SECONDS
                    ):
                        reason = (
                            "retry delay exceeds "
                            f"{_MAX_RETRY_SLEEP_SECONDS:.0f}s safety cap"
                        )
                        _log_terminal_failure(
                            chunk_index=chunk_index,
                            chunk_count=chunk_count,
                            attempts=attempt,
                            reason=reason,
                        )
                        return result(
                            delivery_ok=False,
                            error=reason,
                            delivered=delivered,
                        )
                    if deadline is not None:
                        try:
                            remaining = _remaining_seconds(
                                deadline,
                                deadline_clock,
                            )
                        except (OverflowError, TypeError, ValueError) as exc:
                            _log_terminal_failure(
                                chunk_index=chunk_index,
                                chunk_count=chunk_count,
                                attempts=attempt,
                                reason=f"invalid deadline clock: {exc}",
                            )
                            return result(
                                delivery_ok=False,
                                error=f"invalid deadline clock: {exc}",
                                delivered=delivered,
                            )
                        if delay >= remaining:
                            _log_terminal_failure(
                                chunk_index=chunk_index,
                                chunk_count=chunk_count,
                                attempts=attempt,
                                reason=(
                                    "retry delay would reach or cross "
                                    "send deadline"
                                ),
                            )
                            return result(
                                delivery_ok=False,
                                error=transport_error,
                                delivered=delivered,
                            )
                    time.sleep(delay)
                else:
                    _log_terminal_failure(
                        chunk_index=chunk_index,
                        chunk_count=chunk_count,
                        attempts=retries,
                        reason="retry budget exhausted",
                    )
                    return result(
                        delivery_ok=False,
                        error=transport_error,
                        delivered=delivered,
                    )

    return result(
        delivery_ok=True,
        error=None,
        delivered=delivered,
    )


def send_telegram(
    message: str,
    chat_id: Optional[str] = None,
    token: Optional[str] = None,
    parse_mode: Optional[str] = None,
    retries: int = 3,
    backoff: float = 2.0,
    dry_run: bool = False,
    deadline: datetime | None = None,
    clock: Callable[[], datetime] | None = None,
) -> bool:
    """기존 호출자용 bool 호환 API."""
    return send_telegram_with_receipt(
        message,
        chat_id=chat_id,
        token=token,
        parse_mode=parse_mode,
        retries=retries,
        backoff=backoff,
        dry_run=dry_run,
        deadline=deadline,
        clock=clock,
    ).delivery_ok


# ============================================================================
# 직접 실행 — 봇 셋업 / 테스트
# ============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--message", default="prelude test message")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    token = get_token()
    chat_id = get_chat_id()
    if not args.dry_run and (not token or not chat_id):
        print("ERROR: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID 환경변수 미설정")
        print("설정: export TELEGRAM_BOT_TOKEN=...")
        print("       export TELEGRAM_CHAT_ID=...")
        print("       또는 .env 파일")
        sys.exit(1)

    ok = send_telegram(args.message, dry_run=args.dry_run)
    print("OK" if ok else "FAIL")

from __future__ import annotations

import logging
import os
import runpy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import requests

import notifier.telegram as telegram
from notifier.telegram import (
    _safe_error,
    _split_message_utf16,
    _utf16_units,
    send_telegram,
    send_telegram_with_receipt,
    telegram_error_is_ambiguous,
)


@pytest.fixture(autouse=True)
def _allow_mocked_sends(monkeypatch):
    # 이 모듈은 발송 경로 자체를 mock 된 requests 로 검증한다 —
    # 전역 kill-switch(tests/conftest.py)를 in-process 한정 해제.
    # 해제 상태에서도 mock 누락이 실 API 호출로 이어지지 않게 가장 낮은
    # HTTP 경계를 기본 폭탄으로 봉쇄하고, 각 HTTP 테스트만 명시 대체한다.
    monkeypatch.delenv("PRELUDE_FORBID_TELEGRAM", raising=False)

    def forbid_unmocked_post(*_args, **_kwargs):
        pytest.fail("unmocked Telegram HTTP transport reached")

    monkeypatch.setattr(telegram.requests, "post", forbid_unmocked_post)


@pytest.mark.parametrize("inherited", ("", "0", "disabled"))
def test_global_guard_overrides_untrusted_inherited_value(
    monkeypatch,
    inherited,
):
    monkeypatch.setenv("PRELUDE_FORBID_TELEGRAM", inherited)

    runpy.run_path(
        str(Path(__file__).resolve().parent / "conftest.py"),
        run_name="__prelude_conftest_guard_test__",
    )

    assert os.environ["PRELUDE_FORBID_TELEGRAM"] == "1"


class _Response:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"{self.status_code} error",
                response=self,
            )

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _ok_payload(
    text: str,
    *,
    message_id: int = 7,
    chat_id: int | str = 456,
    server_date: datetime | None = None,
) -> dict:
    observed = server_date or datetime(
        2026,
        7,
        25,
        9,
        20,
        58,
        tzinfo=timezone.utc,
    )
    return {
        "ok": True,
        "result": {
            "message_id": message_id,
            "date": int(observed.timestamp()),
            "text": text,
            "chat": {"id": chat_id},
        },
    }


def test_safe_error_redacts_telegram_token():
    token = "123:SECRET"
    err = RuntimeError(f"https://api.telegram.org/bot{token}/sendMessage failed")

    text = _safe_error(err, token)

    assert token not in text
    assert "<telegram-token>" in text


def test_missing_credentials_is_failure(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    assert send_telegram("not delivered") is False


def test_explicit_dry_run_succeeds_without_credentials(monkeypatch, capsys):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    assert send_telegram("preview", dry_run=True) is True
    assert "chat_id=missing" in capsys.readouterr().out


def test_dry_run_does_not_print_raw_chat_id(capsys):
    secret_chat_id = "-1001234567890"

    assert send_telegram(
        "preview",
        chat_id=secret_chat_id,
        dry_run=True,
    ) is True
    output = capsys.readouterr().out
    assert "chat_id=configured" in output
    assert secret_chat_id not in output


@pytest.mark.parametrize(
    ("message", "retries", "backoff"),
    [
        ("", 3, 2.0),
        ("   ", 3, 2.0),
        ("message", 0, 2.0),
        ("message", -1, 2.0),
        ("message", 1, -0.1),
        ("message", 1, float("nan")),
        ("message", 1, float("inf")),
    ],
)
def test_invalid_send_input_fails_closed_without_http(
    monkeypatch, message, retries, backoff
):
    def unexpected_post(*args, **kwargs):
        raise AssertionError("invalid input must not reach Telegram")

    monkeypatch.setattr(telegram.requests, "post", unexpected_post)

    assert send_telegram(
        message,
        token="123:SECRET",
        chat_id="456",
        retries=retries,
        backoff=backoff,
    ) is False


@pytest.mark.parametrize(
    "payload",
    [
        {"ok": False, "description": "rejected"},
        {"ok": True},
        {"ok": True, "result": {}},
        {"ok": True, "result": {"message_id": 0}},
        {"ok": True, "result": {"message_id": True}},
        {"ok": 1, "result": {"message_id": 7}},
        [],
    ],
)
def test_http_200_requires_true_ok_and_valid_message_id(monkeypatch, payload):
    monkeypatch.setattr(
        telegram.requests,
        "post",
        lambda *args, **kwargs: _Response(200, payload),
    )

    assert send_telegram(
        "message",
        token="123:SECRET",
        chat_id="456",
        retries=1,
    ) is False


def test_valid_bot_api_response_is_success(monkeypatch):
    monkeypatch.setattr(
        telegram.requests,
        "post",
        lambda *args, **kwargs: _Response(200, _ok_payload("message")),
    )

    assert send_telegram(
        "message",
        token="123:SECRET",
        chat_id="456",
        retries=1,
    ) is True


def test_formatted_send_accepts_rendered_server_text(monkeypatch):
    monkeypatch.setattr(
        telegram.requests,
        "post",
        lambda *args, **kwargs: _Response(
            200,
            _ok_payload("rendered"),
        ),
    )

    result = send_telegram_with_receipt(
        "<b>rendered</b>",
        token="123:SECRET",
        chat_id="456",
        parse_mode="HTML",
        retries=1,
    )

    assert result.delivery_ok is True
    assert result.telegram_messages[0].text_sha256 == telegram._sha256_text(
        "<b>rendered</b>"
    )


def test_utf16_chunking_keeps_emoji_pairs_within_telegram_limit():
    message = ("가" * 3999) + "😀" + ("나" * 2)

    chunks = _split_message_utf16(message)

    assert "".join(chunks) == message
    assert len(chunks) == 2
    assert [_utf16_units(chunk) for chunk in chunks] == [3999, 4]
    assert all(_utf16_units(chunk) <= 4000 for chunk in chunks)


def test_utf16_chunks_are_the_exact_http_payloads(monkeypatch):
    message = "😀" * 2001
    payloads: list[str] = []

    def fake_post(*_args, **kwargs):
        payloads.append(kwargs["data"]["text"])
        return _Response(
            200,
            _ok_payload(
                kwargs["data"]["text"],
                message_id=len(payloads),
            ),
        )

    monkeypatch.setattr(telegram.requests, "post", fake_post)

    assert send_telegram(
        message,
        token="123:SECRET",
        chat_id="456",
        retries=1,
    )
    assert "".join(payloads) == message
    assert [_utf16_units(chunk) for chunk in payloads] == [4000, 2]


def test_429_honors_retry_after_before_retry(monkeypatch):
    responses = iter(
        [
            _Response(
                429,
                {
                    "ok": False,
                    "parameters": {"retry_after": 7},
                },
            ),
            _Response(200, _ok_payload("message", message_id=8)),
        ]
    )
    sleeps: list[float] = []
    monkeypatch.setattr(
        telegram.requests,
        "post",
        lambda *args, **kwargs: next(responses),
    )
    monkeypatch.setattr(telegram.time, "sleep", sleeps.append)

    assert send_telegram(
        "message",
        token="123:SECRET",
        chat_id="456",
        retries=2,
        backoff=2.0,
    ) is True
    assert sleeps == [7.0]


def test_oversized_retry_after_fails_without_sleep_or_repost(monkeypatch):
    posts = 0
    sleeps: list[float] = []

    def fake_post(*_args, **_kwargs):
        nonlocal posts
        posts += 1
        return _Response(
            429,
            {
                "ok": False,
                "parameters": {"retry_after": 31},
            },
        )

    monkeypatch.setattr(telegram.requests, "post", fake_post)
    monkeypatch.setattr(telegram.time, "sleep", sleeps.append)

    result = send_telegram_with_receipt(
        "message",
        token="123:SECRET",
        chat_id="456",
        retries=2,
    )

    assert result.delivery_ok is False
    assert result.error == "retry delay exceeds 30s safety cap"
    assert posts == 1
    assert sleeps == []


def test_expired_deadline_blocks_before_first_http_attempt(monkeypatch):
    now = datetime(2026, 7, 25, 9, 21, tzinfo=timezone.utc)
    calls: list[str] = []
    monkeypatch.setattr(
        telegram.requests,
        "post",
        lambda *_args, **_kwargs: calls.append("post"),
    )

    assert send_telegram(
        "message",
        token="123:SECRET",
        chat_id="456",
        deadline=now,
        clock=lambda: now,
    ) is False
    assert calls == []


def test_request_timeout_is_capped_at_deadline_remaining(monkeypatch):
    now = datetime(2026, 7, 25, 9, 20, 58, tzinfo=timezone.utc)
    observed_timeouts: list[float] = []

    def fake_post(*_args, **kwargs):
        observed_timeouts.append(kwargs["timeout"])
        return _Response(
            200,
            _ok_payload(
                "message",
                message_id=11,
                server_date=now,
            ),
        )

    monkeypatch.setattr(telegram.requests, "post", fake_post)

    assert send_telegram(
        "message",
        token="123:SECRET",
        chat_id="456",
        deadline=now + timedelta(seconds=2),
        clock=lambda: now,
    )
    assert observed_timeouts == [2.0]


def test_retry_after_crossing_deadline_neither_sleeps_nor_reposts(monkeypatch):
    now = datetime(2026, 7, 25, 9, 20, 55, tzinfo=timezone.utc)
    posts = 0
    sleeps: list[float] = []

    def fake_post(*_args, **_kwargs):
        nonlocal posts
        posts += 1
        return _Response(
            429,
            {
                "ok": False,
                "parameters": {"retry_after": 5},
            },
        )

    monkeypatch.setattr(telegram.requests, "post", fake_post)
    monkeypatch.setattr(telegram.time, "sleep", sleeps.append)

    assert send_telegram(
        "message",
        token="123:SECRET",
        chat_id="456",
        retries=2,
        deadline=now + timedelta(seconds=5),
        clock=lambda: now,
    ) is False
    assert posts == 1
    assert sleeps == []


def test_deadline_after_first_chunk_reports_partial_delivery(
    monkeypatch,
    caplog,
):
    before = datetime(2026, 7, 25, 9, 20, 59, tzinfo=timezone.utc)
    deadline = before + timedelta(seconds=1)
    clock_values = iter(
        [
            before,  # first chunk attempt
            deadline,  # second chunk attempt
        ]
    )
    posts = 0

    def fake_post(*_args, **_kwargs):
        nonlocal posts
        posts += 1
        return _Response(
            200,
            _ok_payload(
                "x" * 4000,
                message_id=posts,
                server_date=before,
            ),
        )

    monkeypatch.setattr(telegram.requests, "post", fake_post)

    with caplog.at_level(logging.ERROR, logger="telegram"):
        ok = send_telegram(
            "x" * 4001,
            token="123:SECRET",
            chat_id="456",
            retries=1,
            deadline=deadline,
            clock=lambda: next(clock_values),
        )

    assert ok is False
    assert posts == 1
    assert "partial delivery" in caplog.text
    assert "1/2 chunk(s) delivered" in caplog.text


@pytest.mark.parametrize(
    ("deadline", "clock"),
    [
        (datetime(2026, 7, 25, 9, 21), lambda: datetime.now(timezone.utc)),
        (
            datetime(2026, 7, 25, 9, 21, tzinfo=timezone.utc),
            lambda: datetime(2026, 7, 25, 9, 20),
        ),
    ],
)
def test_deadline_requires_aware_values_without_http(
    monkeypatch,
    deadline,
    clock,
):
    calls: list[str] = []
    monkeypatch.setattr(
        telegram.requests,
        "post",
        lambda *_args, **_kwargs: calls.append("post"),
    )

    assert send_telegram(
        "message",
        token="123:SECRET",
        chat_id="456",
        deadline=deadline,
        clock=clock,
    ) is False
    assert calls == []


def test_long_message_partial_delivery_returns_false_and_logs(
    monkeypatch, caplog
):
    responses = iter(
        [
            _Response(
                200,
                _ok_payload("x" * 4000, message_id=9),
            ),
            _Response(200, {"ok": False, "description": "rejected"}),
        ]
    )
    monkeypatch.setattr(
        telegram.requests,
        "post",
        lambda *args, **kwargs: next(responses),
    )

    with caplog.at_level(logging.ERROR, logger="telegram"):
        ok = send_telegram(
            "x" * 4001,
            token="123:SECRET",
            chat_id="456",
            retries=1,
        )

    assert ok is False
    assert "partial delivery" in caplog.text
    assert "1/2 chunk(s) delivered" in caplog.text


def test_detailed_result_binds_each_chunk_text_chat_and_server_metadata(
    monkeypatch,
):
    message = "x" * 4001
    next_id = 40

    def fake_post(*_args, **kwargs):
        nonlocal next_id
        next_id += 1
        return _Response(
            200,
            _ok_payload(
                kwargs["data"]["text"],
                message_id=next_id,
            ),
        )

    monkeypatch.setattr(telegram.requests, "post", fake_post)

    result = send_telegram_with_receipt(
        message,
        token="123:SECRET",
        chat_id="456",
        retries=1,
    )

    assert result.delivery_ok is True
    assert result.chunk_count == 2
    assert [item.message_id for item in result.telegram_messages] == [41, 42]
    assert [item.text_sha256 for item in result.telegram_messages] == [
        telegram._sha256_text("x" * 4000),
        telegram._sha256_text("x"),
    ]
    assert result.message_sha256 == telegram._sha256_text(message)
    assert result.chat_id_sha256 == telegram._sha256_text("456")
    assert result.error is None


@pytest.mark.parametrize(
    "payload",
    [
        _ok_payload("different text"),
        _ok_payload("message", chat_id=999),
    ],
)
def test_success_response_must_echo_exact_text_and_chat(
    monkeypatch,
    payload,
):
    monkeypatch.setattr(
        telegram.requests,
        "post",
        lambda *_args, **_kwargs: _Response(200, payload),
    )

    result = send_telegram_with_receipt(
        "message",
        token="123:SECRET",
        chat_id="456",
        retries=1,
    )

    assert result.delivery_ok is False
    assert result.telegram_messages == ()
    assert result.error
    assert telegram_error_is_ambiguous(result.error)


def test_detailed_partial_failure_preserves_accepted_chunk_metadata(
    monkeypatch,
):
    responses = iter(
        [
            _Response(200, _ok_payload("x" * 4000, message_id=71)),
            _Response(200, {"ok": False, "description": "rejected"}),
        ]
    )
    monkeypatch.setattr(
        telegram.requests,
        "post",
        lambda *_args, **_kwargs: next(responses),
    )

    result = send_telegram_with_receipt(
        "x" * 4001,
        token="123:SECRET",
        chat_id="456",
        retries=1,
    )

    assert result.delivery_ok is False
    assert result.chunk_count == 2
    assert [item.message_id for item in result.telegram_messages] == [71]
    assert result.error


def test_server_date_at_deadline_is_rejected_without_retry(monkeypatch):
    before = datetime(2026, 7, 25, 9, 20, 59, tzinfo=timezone.utc)
    deadline = before + timedelta(seconds=1)
    posts = 0

    def fake_post(*_args, **_kwargs):
        nonlocal posts
        posts += 1
        return _Response(
            200,
            _ok_payload(
                "message",
                server_date=deadline,
            ),
        )

    monkeypatch.setattr(telegram.requests, "post", fake_post)

    result = send_telegram_with_receipt(
        "message",
        token="123:SECRET",
        chat_id="456",
        retries=3,
        deadline=deadline,
        clock=lambda: before,
    )

    assert result.delivery_ok is False
    assert result.telegram_messages == ()
    assert posts == 1


def test_reused_message_id_preserves_only_prior_verified_chunk(
    monkeypatch,
):
    payloads = iter(
        [
            _ok_payload("x" * 4000, message_id=91),
            _ok_payload("x", message_id=91),
        ]
    )
    monkeypatch.setattr(
        telegram.requests,
        "post",
        lambda *_args, **_kwargs: _Response(200, next(payloads)),
    )

    result = send_telegram_with_receipt(
        "x" * 4001,
        token="123:SECRET",
        chat_id="456",
        retries=3,
    )

    assert result.delivery_ok is False
    assert [item.message_id for item in result.telegram_messages] == [91]
    assert telegram_error_is_ambiguous(result.error)


def test_ambiguous_http_200_contract_failure_is_never_retried(
    monkeypatch,
):
    posts = 0

    def fake_post(*_args, **_kwargs):
        nonlocal posts
        posts += 1
        return _Response(200, _ok_payload("wrong text"))

    monkeypatch.setattr(telegram.requests, "post", fake_post)

    result = send_telegram_with_receipt(
        "message",
        token="123:SECRET",
        chat_id="456",
        retries=3,
    )

    assert result.delivery_ok is False
    assert telegram_error_is_ambiguous(result.error)
    assert posts == 1


@pytest.mark.parametrize(
    "failure",
    [
        requests.ReadTimeout("response timed out"),
        _Response(408, {"ok": False, "description": "request timed out"}),
        _Response(500, {"ok": False, "description": "server error"}),
    ],
)
def test_ambiguous_transport_failure_is_never_retried(
    monkeypatch,
    failure,
):
    posts = 0

    def fake_post(*_args, **_kwargs):
        nonlocal posts
        posts += 1
        if isinstance(failure, Exception):
            raise failure
        return failure

    monkeypatch.setattr(telegram.requests, "post", fake_post)

    result = send_telegram_with_receipt(
        "message",
        token="123:SECRET",
        chat_id="456",
        retries=3,
    )

    assert result.delivery_ok is False
    assert telegram_error_is_ambiguous(result.error)
    assert posts == 1


def test_explicit_bot_rejection_remains_nonambiguous(monkeypatch):
    posts = 0

    def fake_post(*_args, **_kwargs):
        nonlocal posts
        posts += 1
        return _Response(200, {"ok": False, "description": "rejected"})

    monkeypatch.setattr(telegram.requests, "post", fake_post)

    result = send_telegram_with_receipt(
        "message",
        token="123:SECRET",
        chat_id="456",
        retries=3,
    )

    assert result.delivery_ok is False
    assert not telegram_error_is_ambiguous(result.error)
    assert posts == 1


@pytest.mark.parametrize("status_code", [400, 401, 403, 409])
def test_permanent_http_rejection_is_never_retried(
    monkeypatch,
    status_code,
):
    posts = 0
    sleeps: list[float] = []

    def fake_post(*_args, **_kwargs):
        nonlocal posts
        posts += 1
        return _Response(
            status_code,
            {"ok": False, "description": "permanent rejection"},
        )

    monkeypatch.setattr(telegram.requests, "post", fake_post)
    monkeypatch.setattr(telegram.time, "sleep", sleeps.append)

    result = send_telegram_with_receipt(
        "message",
        token="123:SECRET",
        chat_id="456",
        retries=3,
    )

    assert result.delivery_ok is False
    assert not telegram_error_is_ambiguous(result.error)
    assert posts == 1
    assert sleeps == []


def test_server_trimmed_echo_of_trailing_whitespace_is_accepted(monkeypatch):
    # Telegram은 앞뒤 공백/개행을 strip해 에코한다 — trailing newline 메시지가
    # ambiguous로 오분류돼 heartbeat가 거짓 FAIL 나던 2026-07-27 회귀 방지.
    message = "heartbeat alert\n- issue one\n"

    monkeypatch.setattr(
        telegram.requests,
        "post",
        lambda *_args, **kwargs: _Response(
            200,
            _ok_payload(kwargs["data"]["text"].strip()),
        ),
    )

    result = send_telegram_with_receipt(
        message,
        token="123:SECRET",
        chat_id="456",
        retries=1,
    )

    assert result.delivery_ok is True
    assert result.error is None


def test_interior_text_mismatch_stays_ambiguous_after_trim_allowance(
    monkeypatch,
):
    monkeypatch.setattr(
        telegram.requests,
        "post",
        lambda *_args, **_kwargs: _Response(
            200,
            _ok_payload("heartbeat alert\n- tampered"),
        ),
    )

    result = send_telegram_with_receipt(
        "heartbeat alert\n- issue one\n",
        token="123:SECRET",
        chat_id="456",
        retries=1,
    )

    assert result.delivery_ok is False
    assert telegram_error_is_ambiguous(result.error)


def test_forbid_env_blocks_send_without_network(monkeypatch):
    # 전역 kill-switch 계약 — 네트워크 계층에 절대 도달하지 않는다.
    monkeypatch.setenv("PRELUDE_FORBID_TELEGRAM", "1")
    monkeypatch.setattr(
        telegram.requests,
        "post",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("network must not be touched")
        ),
    )

    result = send_telegram_with_receipt(
        "must not leave the process",
        token="123:SECRET",
        chat_id="456",
        retries=3,
    )

    assert result.delivery_ok is False
    assert "PRELUDE_FORBID_TELEGRAM" in result.error

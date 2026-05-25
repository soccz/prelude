from __future__ import annotations

from notifier.telegram import _safe_error


def test_safe_error_redacts_telegram_token():
    token = "123:SECRET"
    err = RuntimeError(f"https://api.telegram.org/bot{token}/sendMessage failed")

    text = _safe_error(err, token)

    assert token not in text
    assert "<telegram-token>" in text

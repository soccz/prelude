from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import hashes, hmac as crypto_hmac
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from scripts.build_dashboard import (
    DEFAULT_OUT_DIR,
    _encrypt_payload,
    resolve_dashboard_passphrase,
)


def test_manual_dashboard_default_stays_inside_project_output():
    default = Path(DEFAULT_OUT_DIR)

    assert not default.is_absolute()
    assert default.parts[:1] == ("output",)


def test_dashboard_encryption_authenticates_and_round_trips_unicode_payload():
    source = {"message": "프렐류드 🚀", "value": 1.25, "ok": True}
    passphrase = "test-passphrase"

    encrypted = _encrypt_payload(source, passphrase)
    salt = base64.b64decode(encrypted["salt"])
    iv = base64.b64decode(encrypted["iv"])
    ciphertext = base64.b64decode(encrypted["ct"])
    mac = base64.b64decode(encrypted["mac"])

    keymat = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=64,
        salt=salt,
        iterations=encrypted["iterations"],
    ).derive(passphrase.encode("utf-8"))
    aes_key, mac_key = keymat[:32], keymat[32:]

    verifier = crypto_hmac.HMAC(mac_key, hashes.SHA256())
    verifier.update(salt + iv + ciphertext)
    verifier.verify(mac)

    decryptor = Cipher(
        algorithms.AES(aes_key),
        modes.CBC(iv),
    ).decryptor()
    padded = decryptor.update(ciphertext) + decryptor.finalize()
    pad_len = padded[-1]
    assert 1 <= pad_len <= 16
    assert padded[-pad_len:] == bytes([pad_len]) * pad_len
    decoded = json.loads(padded[:-pad_len].decode("utf-8"))
    assert decoded == source


def test_dashboard_passphrase_has_no_public_default(monkeypatch):
    monkeypatch.delenv("PRELUDE_DASHBOARD_PIN", raising=False)

    with pytest.raises(ValueError, match="requires --pin"):
        resolve_dashboard_passphrase()


@pytest.mark.parametrize(
    "value",
    [
        "",
        "996",
        " leading-secret",
        "trailing-secret ",
    ],
)
def test_dashboard_passphrase_rejects_weak_or_ambiguous_values(
    monkeypatch,
    value,
):
    monkeypatch.setenv("PRELUDE_DASHBOARD_PIN", value)

    with pytest.raises(ValueError):
        resolve_dashboard_passphrase()


def test_dashboard_passphrase_accepts_four_digit_pin(monkeypatch):
    # 2026-07-28 사용자 명시 승인 — 최소 길이 12 → 4 완화 계약 고정.
    monkeypatch.setenv("PRELUDE_DASHBOARD_PIN", "9963")

    assert resolve_dashboard_passphrase() == "9963"


def test_dashboard_passphrase_accepts_explicit_or_environment_secret(
    monkeypatch,
):
    monkeypatch.setenv(
        "PRELUDE_DASHBOARD_PIN",
        "environment-secret-2026",
    )

    assert resolve_dashboard_passphrase() == "environment-secret-2026"
    assert (
        resolve_dashboard_passphrase("explicit-secret-2026")
        == "explicit-secret-2026"
    )

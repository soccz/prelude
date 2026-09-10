from __future__ import annotations

import copy
import hashlib
from datetime import datetime
from types import SimpleNamespace

import pytest

import notifier.delivery_receipt as receipts
import notifier.telegram as telegram
import scripts.recommend_send as sender


@pytest.mark.parametrize(
    "value, expected",
    [
        (0, "0%"),
        (-0.0, "0%"),
        (1e-12, "<1%"),
        (0.0014, "<1%"),
        (0.0037, "<1%"),
        (0.005, "<1%"),
        (0.009999, "<1%"),
        (0.01, "1%"),
        (0.0175, "2%"),
        (0.2726, "27%"),
        (1, "100%"),
        ("0.0037", "<1%"),
        (None, "—"),
        (float("nan"), "—"),
        (float("inf"), "—"),
        (float("-inf"), "—"),
        (-0.0001, "—"),
        (1.0001, "—"),
        (True, "—"),
        (False, "—"),
        ("invalid", "—"),
        (10**400, "—"),
    ],
)
def test_probability_display_preserves_small_nonzero_risk(value, expected):
    assert sender._pct(value) == expected


def _radar(slot="open"):
    return {
        "asof": "2026-09-10",
        "slot": slot,
        "btc_regime": "bull_volatile",
        "universe_n": 100,
        "top3": [
            {
                "coin": "KRW-TEST",
                "rank": 1,
                "entry_open": 52.6,
                "p_up5": 0.3058,
                "p_up10": 0.2726,
                "p_up20": 0.0037,
                "p_dn5": 0.0175,
                "p_dn10": 0.0014,
                "exp_downside": -0.0311,
            }
        ],
    }


@pytest.mark.parametrize("slot", ["open", "preopen"])
def test_radar_changes_only_probability_presentation(slot):
    snapshot = _radar(slot)
    before = copy.deepcopy(snapshot)

    message = sender.format_radar(snapshot, slot)

    assert snapshot == before
    assert "#1 TEST" in message
    assert "≥5% 31% · ≥10% 27% · ≥20% <1%" in message
    assert "≤-5% 2% · ≤-10% <1% · E[하방] -3.1%" in message
    assert "안전 보장 아님" in message
    assert "현재 체결가격 아님" in message
    assert "&lt;" not in message  # The live transport uses plain text, not HTML.


def test_radar_small_probability_is_sent_as_plain_text(monkeypatch):
    message = sender.format_radar(_radar(), "open")
    clock = datetime.fromisoformat("2026-09-10T00:07:59+00:00")
    payloads = []

    def fake_post(_url, *, data, timeout):
        payloads.append(data)
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "ok": True,
                "result": {
                    "message_id": 1,
                    "date": int(clock.timestamp()),
                    "chat": {"id": 456},
                    "text": data["text"],
                },
            },
        )

    # Only this in-process, fake-credential, mocked-HTTP transport may run.
    monkeypatch.setattr(telegram.requests, "post", fake_post)
    monkeypatch.setattr(telegram, "get_token", lambda: "test-token")
    monkeypatch.setattr(telegram, "get_chat_id", lambda: "456")
    monkeypatch.setattr(sender, "_now_kst", lambda: clock)
    monkeypatch.delenv("PRELUDE_FORBID_TELEGRAM")

    ok, error, sent_at, result = sender._send_live_transport(
        message, asof="2026-09-10", slot="open"
    )

    assert ok and error is None and sent_at == clock.isoformat()
    assert len(payloads) == 1
    assert payloads[0]["text"] == message and "<1%" in message
    assert "parse_mode" not in payloads[0]
    assert result.message_sha256 == hashlib.sha256(message.encode()).hexdigest()


def test_existing_delivery_evidence_survives_a_display_change(tmp_path, monkeypatch):
    snapshot = _radar()
    snapshot.update(
        snapshot_id="recommend-test-display-change",
        snapshot_path=str(tmp_path / "snapshots/2026-09-10/open_r1.json"),
        snapshot_schema=sender.SNAPSHOT_SCHEMA_VERSION,
        rank_basis=sender.APPROVED_LIVE_R1_RANK_BASIS,
        score_schema_version=sender.APPROVED_LIVE_SCORE_SCHEMA,
        rule_version=sender.APPROVED_LIVE_R1_RULE_VERSION,
        ranking="R1",
        model={"id": "recommend_r1_open", "ranking": "R1"},
        request={"asof": "2026-09-10", "slot": "open", "ranking": "R1", "limit_markets": None},
        decision_started_at="2026-09-10T00:05:00+00:00",
        decision_completed_at="2026-09-10T00:07:58+00:00",
    )
    clock = datetime.fromisoformat("2026-09-10T00:07:59+00:00")

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock.astimezone(tz) if tz is not None else clock.replace(tzinfo=None)

    monkeypatch.setattr(sender, "_now_kst", lambda: clock)
    monkeypatch.setattr(receipts, "datetime", FixedDatetime)
    monkeypatch.setattr(sender, "maybe_notify_champion_change", lambda *_a, **_k: None)
    calls = []

    def accepted(message, **_kwargs):
        calls.append(message)
        digest = hashlib.sha256(message.encode()).hexdigest()
        return telegram.TelegramSendResult(
            delivery_ok=True,
            message_sha256=digest,
            chunk_count=1,
            chat_id_sha256=hashlib.sha256(b"456").hexdigest(),
            telegram_messages=(telegram.TelegramServerMessage(1, clock.isoformat(), digest),),
            error=None,
        )

    monkeypatch.setattr(sender, "send_telegram_with_receipt", accepted)
    current_message = sender.format_radar(snapshot, "open")
    old_message = current_message.replace("<1%", "0%")
    assert current_message != old_message
    root = tmp_path / "receipts"
    assert sender._send_and_record(snapshot, old_message, slot="open", receipt_root=root)
    before = {path: path.read_bytes() for path in root.rglob("*.json")}

    assert sender._send_and_record(snapshot, current_message, slot="open", receipt_root=root)

    assert calls == [old_message]
    assert {path: path.read_bytes() for path in root.rglob("*.json")} == before
    receipt = receipts.read_delivery_receipt(snapshot, root=root)
    assert receipt["delivery_ok"] is True
    assert receipt["message_sha256"] == hashlib.sha256(old_message.encode()).hexdigest()

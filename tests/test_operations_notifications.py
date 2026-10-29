from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from ops.operations_notifications import heartbeat, notify


def test_new_duplicate_recovery_and_recurrence(tmp_path):
    sent = []
    path = tmp_path / "state.json"

    def sender(text):
        sent.append(text)
        return True

    kw = dict(state_path=path, sender=sender)
    assert heartbeat("- publish today missing", **kw) == "sent"
    assert heartbeat("- publish another date missing", **kw) == "suppressed"
    assert len(sent) == 1
    assert heartbeat("", **kw) == "sent"
    assert "복구" in sent[-1]
    assert heartbeat("", **kw) == "suppressed"
    assert heartbeat("- publish today missing", **kw) == "sent"
    assert len(sent) == 3


def test_research_details_are_recorded_without_telegram(tmp_path):
    def forbidden(_):
        raise AssertionError("research-only issue sent a message")

    path = tmp_path / "state.json"
    message = "\n".join(
        "- " + item + " probe FAIL"
        for item in (
            "microstructure",
            "shortlist",
            "post-label",
            "fixed L1",
            "regime forward",
            "L1 forward",
            "research snapshot",
        )
    )
    assert heartbeat(message, state_path=path, sender=forbidden) == "recorded"
    state = json.loads(path.read_text())
    assert len(state["active"]) == 7
    assert not state["acknowledged"]


def test_service_and_heartbeat_share_incident_and_new_core_fault_still_notifies(
    tmp_path,
):
    sent = []
    kw = dict(
        state_path=tmp_path / "state.json",
        sender=lambda text: sent.append(text) or True,
    )
    notify({"publish": ["prelude-publish-dashboard.service"]}, **kw)
    heartbeat("- publish no session", **kw)
    assert len(sent) == 1
    heartbeat("- publish no session\n- disk 95%", **kw)
    assert len(sent) == 2 and "저장 공간" in sent[-1]


def test_failed_delivery_is_not_acknowledged_and_can_retry(tmp_path):
    path = tmp_path / "state.json"
    with pytest.raises(RuntimeError, match="delivery failed"):
        heartbeat("- publish missing", state_path=path, sender=lambda _: False)
    assert json.loads(path.read_text())["acknowledged"] == []
    assert (
        heartbeat("- publish missing", state_path=path, sender=lambda _: True) == "sent"
    )


def test_ambiguous_acceptance_is_not_replayed_or_claimed_success(tmp_path):
    sent = []

    def ambiguous(text):
        sent.append(text)
        return SimpleNamespace(
            delivery_ok=False, error="ambiguous Telegram acceptance: timeout"
        )

    path = tmp_path / "state.json"
    kw = dict(state_path=path, sender=ambiguous)
    assert heartbeat("- publish missing", **kw) == "uncertain"
    assert heartbeat("- publish missing", **kw) == "uncertain"
    assert len(sent) == 1
    assert json.loads(path.read_text())["acknowledged"] == []


def test_partial_observation_cannot_recover_an_unchecked_service(tmp_path):
    path = tmp_path / "state.json"
    kw = dict(state_path=path, sender=lambda _: True)
    notify({"publish": ["failed"]}, **kw)
    notify({"backup": ["failed"]}, **kw)
    assert set(json.loads(path.read_text())["acknowledged"]) == {"publish", "backup"}


def test_uncertain_incident_does_not_mask_new_fault_or_later_recurrence(tmp_path):
    path = tmp_path / "state.json"
    heartbeat(
        "- publish missing",
        state_path=path,
        sender=lambda _: SimpleNamespace(
            delivery_ok=False, error="ambiguous Telegram acceptance: timeout"
        ),
    )
    sent = []
    kw = dict(state_path=path, sender=lambda text: sent.append(text) or True)
    heartbeat("- publish missing\n- disk 95%", **kw)
    assert len(sent) == 1 and "저장 공간" in sent[0] and "• 대시보드" not in sent[0]
    assert json.loads(path.read_text())["acknowledged"] == ["disk"]
    heartbeat("", **kw)
    heartbeat("- publish missing", **kw)
    assert "• 대시보드" in sent[-1]


def test_crash_after_send_intent_does_not_replay_on_restart(tmp_path):
    path = tmp_path / "state.json"

    def crash(_):
        raise ConnectionError("acceptance not known")

    with pytest.raises(ConnectionError):
        heartbeat("- publish missing", state_path=path, sender=crash)

    def forbidden(_):
        raise AssertionError("unconfirmed intent replayed")

    assert (
        heartbeat("- publish missing", state_path=path, sender=forbidden) == "uncertain"
    )
    assert json.loads(path.read_text())["acknowledged"] == []


def test_distinct_delivery_slots_and_recovery_share_service_incident(tmp_path):
    sent = []
    kw = dict(
        state_path=tmp_path / "state.json",
        sender=lambda text: sent.append(text) or True,
    )
    notify({"recommendation_preopen": ["failed"]}, **kw)
    report = {
        "slots": {
            "preopen": {"state": "missing_decision"},
            "open": {"state": "missing_decision"},
        }
    }
    heartbeat("- recommend snapshot chain probe FAIL", delivery_status=report, **kw)
    assert len(sent) == 2 and "장후" in sent[-1]
    report["slots"]["preopen"]["state"] = "delivered_candidates"
    report["slots"]["open"]["state"] = "delivered_empty"
    heartbeat("", delivery_status=report, **kw)
    assert len(sent) == 3 and "복구" in sent[-1]


def test_unsafe_or_corrupt_state_does_not_silently_reset(tmp_path):
    target = tmp_path / "original.json"
    target.write_text("{}")
    link = tmp_path / "state.json"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        heartbeat("- publish missing", state_path=link, sender=lambda _: True)
    with pytest.raises(ValueError, match="schema"):
        heartbeat("- publish missing", state_path=target, sender=lambda _: True)
    with pytest.raises(ValueError, match="timezone"):
        heartbeat(
            "",
            state_path=tmp_path / "other.json",
            now=datetime(2026, 10, 29),
            sender=lambda _: True,
        )
    assert datetime.now(timezone.utc).tzinfo is not None

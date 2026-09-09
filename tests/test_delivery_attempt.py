"""Hermetic crash/restart checks for the durable pre-send attempt journal."""
from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import stat

import pytest
import requests

from notifier import delivery_attempt as attempts
from notifier import delivery_receipt as receipts
from notifier.telegram import TelegramSendResult, TelegramServerMessage


NOW = datetime(2026, 9, 7, 0, 5, 2, tzinfo=timezone.utc)
MESSAGE = "independent journal fixture"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    monkeypatch.setenv("PRELUDE_FORBID_TELEGRAM", "1")

    def forbidden(*_args, **_kwargs):
        pytest.fail("network access is forbidden in journal tests")

    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)


def _snapshot(tmp_path: Path) -> dict:
    return {
        "snapshot_id": "recommend-journal-test",
        "snapshot_path": str(tmp_path / "snapshots/2026-09-07/open_r1.json"),
        "asof": "2026-09-07", "slot": "open", "ranking": "R1",
        "model": {"id": "recommend_r1_open", "ranking": "R1"},
    }


def _journal_path(snapshot: dict, root: Path) -> Path:
    path = receipts.receipt_path(snapshot, root=root)
    return path.parent / f"{path.stem}.attempts"


def _tree(root: Path) -> dict:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*") if path.is_file() and not path.is_symlink()
    }


def _write_receipt(monkeypatch, snapshot, root, *, at=NOW, kind="delivered", message=MESSAGE):
    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            value = at + timedelta(seconds=2)
            return value if tz is None else value.astimezone(tz)

    monkeypatch.setattr(receipts, "datetime", FrozenDatetime)
    message_sha = hashlib.sha256(message.encode()).hexdigest()
    success = kind == "delivered"
    transport = TelegramSendResult(
        delivery_ok=success, message_sha256=message_sha,
        chunk_count=1, chat_id_sha256=hashlib.sha256(b"test-chat").hexdigest(),
        telegram_messages=(TelegramServerMessage(
            message_id=101, server_date=(at + timedelta(seconds=1)).isoformat(),
            text_sha256=message_sha,
        ),) if success else (),
        error=None if success else (
            "ambiguous Telegram acceptance: transport interrupted"
            if kind == "ambiguous" else "Telegram HTTP 400: rejected"
        ),
    )
    receipts.write_delivery_receipt(
        snapshot, delivery_ok=success, attempted_at=at.isoformat(),
        sent_at=(at + timedelta(seconds=1)).isoformat() if success else None,
        error=transport.error, telegram_result=transport, message=message, root=root,
    )
    receipt = receipts.read_delivery_receipt(snapshot, root=root)
    assert receipt is not None
    return receipt


def _inspect(snapshot, root, *, now=NOW + timedelta(minutes=1)):
    return attempts.inspect_delivery_attempt(snapshot, receipt_root=root, now=now)


def test_absent_inspection_creates_no_directory_or_lock(tmp_path):
    snapshot = _snapshot(tmp_path)
    root = tmp_path / "does-not-exist"
    before = list(tmp_path.rglob("*"))
    result = _inspect(snapshot, root)
    assert result["state"] == "absent"
    assert result["attempt_count"] == 0
    assert not root.exists()
    assert list(tmp_path.rglob("*")) == before


def test_begin_is_durable_pending_after_context_restart(tmp_path):
    snapshot, root = _snapshot(tmp_path), tmp_path / "receipts"
    original = copy.deepcopy(snapshot)
    with attempts.delivery_attempt_journal(snapshot, receipt_root=root) as journal:
        intent = journal.begin(MESSAGE, now=NOW, receipt=None)
        assert journal.inspect(now=NOW)["state"] == "pending"
    before = _tree(root)
    restarted = _inspect(snapshot, root)
    assert restarted["state"] == "pending"
    assert restarted["attempt_count"] == 1
    assert restarted["attempted_at"] == NOW.isoformat()
    assert restarted["latest_attempt_id"] == intent["attempt_id"]
    assert _tree(root) == before
    assert snapshot == original


def test_pending_restart_refuses_second_begin(tmp_path):
    snapshot, root = _snapshot(tmp_path), tmp_path / "receipts"
    with attempts.delivery_attempt_journal(snapshot, receipt_root=root) as journal:
        journal.begin(MESSAGE, now=NOW, receipt=None)
    before = _tree(root)
    with pytest.raises((ValueError, RuntimeError)):
        with attempts.delivery_attempt_journal(snapshot, receipt_root=root) as journal:
            journal.begin(MESSAGE, now=NOW + timedelta(seconds=10), receipt=None)
    assert _tree(root) == before
    assert _inspect(snapshot, root)["state"] == "pending"


@pytest.mark.parametrize("kind", ["delivered", "clear_failure", "ambiguous"])
def test_complete_binds_the_exact_new_receipt(monkeypatch, tmp_path, kind):
    snapshot, root = _snapshot(tmp_path), tmp_path / "receipts"
    with attempts.delivery_attempt_journal(snapshot, receipt_root=root) as journal:
        intent = journal.begin(MESSAGE, now=NOW, receipt=None)
        receipt = _write_receipt(monkeypatch, snapshot, root, kind=kind)
        journal.complete(intent, receipt=receipt, now=NOW + timedelta(seconds=3))
    before = _tree(root)
    result = _inspect(snapshot, root)
    assert result["state"] == "resolved"
    assert result["resolution"] == kind
    assert result["attempt_count"] == 1
    assert _tree(root) == before


def test_old_failed_receipt_does_not_resolve_new_pending(monkeypatch, tmp_path):
    snapshot, root = _snapshot(tmp_path), tmp_path / "receipts"
    old_receipt = _write_receipt(monkeypatch, snapshot, root, kind="clear_failure")
    at = NOW + timedelta(seconds=10)
    with attempts.delivery_attempt_journal(snapshot, receipt_root=root) as journal:
        journal.begin(MESSAGE, now=at, receipt=old_receipt)
    before = _tree(root)
    result = _inspect(snapshot, root)
    assert result["state"] == "pending"
    assert result["resolution"] is None
    with pytest.raises((ValueError, RuntimeError)):
        with attempts.delivery_attempt_journal(snapshot, receipt_root=root) as journal:
            journal.begin(MESSAGE, now=at + timedelta(seconds=10), receipt=old_receipt)
    assert _tree(root) == before


def test_terminal_clear_failure_allows_one_new_attempt(monkeypatch, tmp_path):
    snapshot, root = _snapshot(tmp_path), tmp_path / "receipts"
    with attempts.delivery_attempt_journal(snapshot, receipt_root=root) as journal:
        first = journal.begin(MESSAGE, now=NOW, receipt=None)
        old = _write_receipt(monkeypatch, snapshot, root, kind="clear_failure")
        journal.complete(first, receipt=old, now=NOW + timedelta(seconds=3))
    first_files = _tree(_journal_path(snapshot, root))
    with attempts.delivery_attempt_journal(snapshot, receipt_root=root) as journal:
        second = journal.begin(MESSAGE, now=NOW + timedelta(seconds=10), receipt=old)
    assert second["attempt_id"] != first["attempt_id"]
    result = _inspect(snapshot, root)
    assert result["state"] == "pending" and result["attempt_count"] == 2
    current = _tree(_journal_path(snapshot, root))
    assert all(current[key] == value for key, value in first_files.items())


@pytest.mark.parametrize("kind", ["delivered", "ambiguous"])
def test_terminal_delivery_or_ambiguity_never_allows_new_begin(monkeypatch, tmp_path, kind):
    snapshot, root = _snapshot(tmp_path), tmp_path / "receipts"
    with attempts.delivery_attempt_journal(snapshot, receipt_root=root) as journal:
        intent = journal.begin(MESSAGE, now=NOW, receipt=None)
        receipt = _write_receipt(monkeypatch, snapshot, root, kind=kind)
        journal.complete(intent, receipt=receipt, now=NOW + timedelta(seconds=3))
    with pytest.raises((ValueError, RuntimeError)):
        with attempts.delivery_attempt_journal(snapshot, receipt_root=root) as journal:
            journal.begin(MESSAGE, now=NOW + timedelta(seconds=10), receipt=receipt)


@pytest.mark.parametrize("kind", ["regular", "directory"])
def test_fsync_failure_never_reaches_api_boundary(monkeypatch, tmp_path, kind):
    snapshot, root = _snapshot(tmp_path), tmp_path / "receipts"
    original_fsync = os.fsync
    injected = []
    api_calls = []

    def failing_fsync(fd):
        mode = os.fstat(fd).st_mode
        matches = stat.S_ISREG(mode) if kind == "regular" else stat.S_ISDIR(mode)
        if matches:
            injected.append(fd)
            raise OSError("injected fsync failure")
        return original_fsync(fd)

    monkeypatch.setattr(os, "fsync", failing_fsync)
    with pytest.raises((OSError, ValueError, RuntimeError)):
        with attempts.delivery_attempt_journal(snapshot, receipt_root=root) as journal:
            journal.begin(MESSAGE, now=NOW, receipt=None)
            api_calls.append("would call Telegram only after begin returns")
    assert injected
    assert api_calls == []


@pytest.mark.parametrize("mutation", ["truncated", "changed_message", "extra_file"])
def test_inspector_rejects_corrupt_or_unknown_journal_without_writing(tmp_path, mutation):
    snapshot, root = _snapshot(tmp_path), tmp_path / "receipts"
    with attempts.delivery_attempt_journal(snapshot, receipt_root=root) as journal:
        journal.begin(MESSAGE, now=NOW, receipt=None)
    directory = _journal_path(snapshot, root)
    intent_path = directory / "000001.intent.json"
    if mutation == "truncated":
        intent_path.write_text("{", encoding="utf-8")
    elif mutation == "changed_message":
        content = json.loads(intent_path.read_bytes())
        content["message_sha256"] = "0" * 64
        intent_path.write_text(json.dumps(content), encoding="utf-8")
    else:
        (directory / "000003.intent.json").write_bytes(intent_path.read_bytes())
    before = _tree(root)
    assert _inspect(snapshot, root)["state"] == "invalid"
    assert _tree(root) == before
    with pytest.raises((ValueError, RuntimeError)):
        with attempts.delivery_attempt_journal(snapshot, receipt_root=root) as journal:
            journal.begin(MESSAGE, now=NOW + timedelta(seconds=10), receipt=None)
    assert _tree(root) == before


@pytest.mark.parametrize("target", ["directory", "intent"])
def test_symlinked_journal_is_invalid_and_does_not_touch_target(tmp_path, target):
    snapshot, root = _snapshot(tmp_path), tmp_path / "receipts"
    directory = _journal_path(snapshot, root)
    outside = tmp_path / "unrelated"
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_text("preserve", encoding="utf-8")
    if target == "directory":
        directory.parent.mkdir(parents=True)
        directory.symlink_to(outside, target_is_directory=True)
    else:
        directory.mkdir(parents=True)
        (directory / "000001.intent.json").symlink_to(sentinel)
    before = _tree(outside)
    assert _inspect(snapshot, root)["state"] == "invalid"
    with pytest.raises((OSError, ValueError, RuntimeError)):
        with attempts.delivery_attempt_journal(snapshot, receipt_root=root) as journal:
            journal.begin(MESSAGE, now=NOW, receipt=None)
    assert _tree(outside) == before


def test_success_without_journal_remains_legacy_receipt_evidence(monkeypatch, tmp_path):
    snapshot, root = _snapshot(tmp_path), tmp_path / "receipts"
    receipt = _write_receipt(monkeypatch, snapshot, root)
    before = _tree(root)
    assert _inspect(snapshot, root)["state"] == "absent"
    assert not _journal_path(snapshot, root).exists()
    assert _tree(root) == before
    with pytest.raises(attempts.DeliveryAttemptError, match="forbids retry"):
        with attempts.delivery_attempt_journal(snapshot, receipt_root=root) as journal:
            journal.begin(MESSAGE, now=NOW + timedelta(seconds=10), receipt=receipt)


def test_named_fifo_is_rejected_without_blocking_or_writing(monkeypatch, tmp_path):
    snapshot, root = _snapshot(tmp_path), tmp_path / "receipts"
    directory = _journal_path(snapshot, root)
    directory.mkdir(parents=True)
    fifo = directory / "000001.intent.json"
    os.mkfifo(fifo)
    original_open = os.open
    checked = []

    def checked_open(path, flags, *args, **kwargs):
        if str(path).endswith("000001.intent.json"):
            # Assert before the real call so this regression cannot hang if
            # O_NONBLOCK is accidentally removed from the implementation.
            assert flags & os.O_NONBLOCK
            checked.append(str(path))
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", checked_open)
    before = sorted(str(path) for path in root.rglob("*"))
    assert _inspect(snapshot, root)["state"] == "invalid"
    assert checked
    assert stat.S_ISFIFO(fifo.lstat().st_mode)
    assert sorted(str(path) for path in root.rglob("*")) == before


@pytest.mark.parametrize("kind", ["delivered", "clear_failure", "ambiguous"])
def test_receipt_durable_result_missing_is_resolved_readonly(monkeypatch, tmp_path, kind):
    snapshot, root = _snapshot(tmp_path), tmp_path / "receipts"
    with attempts.delivery_attempt_journal(snapshot, receipt_root=root) as journal:
        journal.begin(MESSAGE, now=NOW, receipt=None)
        _write_receipt(monkeypatch, snapshot, root, kind=kind)
        # Simulate termination after receipt fsync, before journal.complete.
    before = _tree(root)
    result = _inspect(snapshot, root)
    assert result["state"] == "resolved" and result["resolution"] == kind
    assert not (_journal_path(snapshot, root) / "000001.result.json").exists()
    assert _tree(root) == before


def test_new_clear_failure_without_result_is_archived_before_retry(monkeypatch, tmp_path):
    snapshot, root = _snapshot(tmp_path), tmp_path / "receipts"
    with attempts.delivery_attempt_journal(snapshot, receipt_root=root) as journal:
        first = journal.begin(MESSAGE, now=NOW, receipt=None)
        receipt = _write_receipt(monkeypatch, snapshot, root, kind="clear_failure")
    with attempts.delivery_attempt_journal(snapshot, receipt_root=root) as journal:
        second = journal.begin(MESSAGE, now=NOW + timedelta(seconds=10), receipt=receipt)
    result_path = _journal_path(snapshot, root) / "000001.result.json"
    archived = json.loads(result_path.read_bytes())
    assert archived["attempt_id"] == first["attempt_id"]
    assert second["previous_result_sha256"] == archived["integrity_sha256"]
    assert _inspect(snapshot, root)["state"] == "pending"
    assert _inspect(snapshot, root)["attempt_count"] == 2


def test_result_publication_failure_does_not_erase_success_receipt(monkeypatch, tmp_path):
    snapshot, root = _snapshot(tmp_path), tmp_path / "receipts"
    with attempts.delivery_attempt_journal(snapshot, receipt_root=root) as journal:
        intent = journal.begin(MESSAGE, now=NOW, receipt=None)
        receipt = _write_receipt(monkeypatch, snapshot, root)
        before = receipts.receipt_path(snapshot, root=root).read_bytes()

        def failed_publish(*_args, **_kwargs):
            raise OSError("injected result publication failure")

        monkeypatch.setattr(journal, "_publish", failed_publish)
        with pytest.raises(OSError, match="publication"):
            journal.complete(intent, receipt=receipt, now=NOW + timedelta(seconds=3))
    assert receipts.receipt_path(snapshot, root=root).read_bytes() == before
    state = _inspect(snapshot, root)
    assert state["state"] == "resolved" and state["resolution"] == "delivered"
    with pytest.raises(attempts.DeliveryAttemptError, match="forbids retry"):
        with attempts.delivery_attempt_journal(snapshot, receipt_root=root) as journal:
            journal.begin(MESSAGE, now=NOW + timedelta(seconds=10), receipt=receipt)


@pytest.mark.parametrize("offset", [-1, 0])
def test_retry_clock_must_advance_beyond_previous_receipt(monkeypatch, tmp_path, offset):
    snapshot, root = _snapshot(tmp_path), tmp_path / "receipts"
    receipt = _write_receipt(monkeypatch, snapshot, root, kind="clear_failure")
    now = datetime.fromisoformat(receipt["recorded_at"]) + timedelta(microseconds=offset)
    with pytest.raises(attempts.DeliveryAttemptError, match="clock must advance"):
        with attempts.delivery_attempt_journal(snapshot, receipt_root=root) as journal:
            journal.begin(MESSAGE, now=now, receipt=receipt)
    assert _inspect(snapshot, root)["attempt_count"] == 0


def test_begin_syncs_new_ancestor_entries_and_intent_before_return(monkeypatch, tmp_path):
    snapshot, root = _snapshot(tmp_path), tmp_path / "receipts"
    original_fsync = os.fsync
    events = []

    def traced_fsync(fd):
        info = os.fstat(fd)
        events.append((stat.S_ISDIR(info.st_mode), info.st_dev, info.st_ino))
        return original_fsync(fd)

    monkeypatch.setattr(os, "fsync", traced_fsync)
    with attempts.delivery_attempt_journal(snapshot, receipt_root=root) as journal:
        journal.begin(MESSAGE, now=NOW, receipt=None)
        before_api = list(events)
    directory = _journal_path(snapshot, root)
    for path in (tmp_path, root, directory.parent, directory):
        info = path.stat()
        assert (True, info.st_dev, info.st_ino) in before_api
    assert any(not item[0] for item in before_api)
    assert before_api[-1][0] is True


def test_intent_publication_failure_never_returns_authorization(monkeypatch, tmp_path):
    snapshot, root = _snapshot(tmp_path), tmp_path / "receipts"
    api_calls = []

    def fail_link(*_args, **_kwargs):
        raise OSError("injected immutable publication failure")

    monkeypatch.setattr(os, "link", fail_link)
    with pytest.raises(OSError, match="publication"):
        with attempts.delivery_attempt_journal(snapshot, receipt_root=root) as journal:
            journal.begin(MESSAGE, now=NOW, receipt=None)
            api_calls.append(True)
    assert api_calls == []
    assert _inspect(snapshot, root)["state"] == "absent"


@pytest.mark.parametrize("mutation", ["identity", "prior_hash", "sequence", "future"])
def test_resealed_intent_semantic_corruption_is_invalid(tmp_path, mutation):
    snapshot, root = _snapshot(tmp_path), tmp_path / "receipts"
    with attempts.delivery_attempt_journal(snapshot, receipt_root=root) as journal:
        journal.begin(MESSAGE, now=NOW, receipt=None)
    path = _journal_path(snapshot, root) / "000001.intent.json"
    content = json.loads(path.read_bytes())
    if mutation == "identity":
        content["identity"]["snapshot_id"] = "different-snapshot"
    elif mutation == "prior_hash":
        content["previous_receipt_sha256"] = "0" * 64
    elif mutation == "sequence":
        content["sequence"] = True
    else:
        content["attempted_at"] = (NOW + timedelta(minutes=5)).isoformat()
    payload = {key: value for key, value in content.items() if key != "integrity_sha256"}
    content["integrity_sha256"] = hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
    path.write_text(json.dumps(content), encoding="utf-8")
    before = _tree(root)
    assert _inspect(snapshot, root)["state"] == "invalid"
    assert _tree(root) == before


def test_naive_inspection_clock_is_invalid_without_writing(tmp_path):
    snapshot, root = _snapshot(tmp_path), tmp_path / "missing"
    result = _inspect(snapshot, root, now=NOW.replace(tzinfo=None))
    assert result["state"] == "invalid"
    assert "timezone-aware" in result["reason"]
    assert not root.exists()


@pytest.mark.parametrize("change", ["attempt_id", "sequence"])
def test_complete_cannot_mutate_the_saved_intent(monkeypatch, tmp_path, change):
    snapshot, root = _snapshot(tmp_path), tmp_path / "receipts"
    with attempts.delivery_attempt_journal(snapshot, receipt_root=root) as journal:
        intent = journal.begin(MESSAGE, now=NOW, receipt=None)
        receipt = _write_receipt(monkeypatch, snapshot, root)
        altered = copy.deepcopy(intent)
        altered[change] = "0" * 32 if change == "attempt_id" else 2
        before = _tree(root)
        with pytest.raises(attempts.DeliveryAttemptError, match="latest saved intent"):
            journal.complete(altered, receipt=receipt, now=NOW + timedelta(seconds=3))
        assert _tree(root) == before

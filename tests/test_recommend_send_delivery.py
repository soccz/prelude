from __future__ import annotations

import json
import hashlib
import multiprocessing
import os
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import notifier.delivery_receipt as receipt_module
import ops.champion_selector as champion_selector
import scripts.recommend_send as recommend_send
from notifier.delivery_receipt import read_delivery_receipt
from notifier.telegram import TelegramSendResult, TelegramServerMessage


@pytest.fixture(autouse=True)
def _allow_mocked_sends(monkeypatch):
    # 이 모듈은 발송 경로 자체를 mock 된 requests 로 검증한다 —
    # 전역 kill-switch(tests/conftest.py)를 in-process 한정 해제.
    # subprocess 를 띄우는 테스트는 이 모듈에 두지 말 것.
    monkeypatch.delenv("PRELUDE_FORBID_TELEGRAM", raising=False)

    def forbid_unmocked_transport(*_args, **_kwargs):
        pytest.fail("unmocked Telegram transport reached")

    monkeypatch.setattr(
        recommend_send,
        "send_telegram",
        forbid_unmocked_transport,
    )
    monkeypatch.setattr(
        recommend_send,
        "send_telegram_with_receipt",
        forbid_unmocked_transport,
    )


def test_live_resolver_refuses_missing_champion_state(monkeypatch):
    monkeypatch.setattr(recommend_send, "get_champion", lambda *_a, **_k: None)

    with pytest.raises(RuntimeError, match="live send refused"):
        recommend_send.resolve_champion(
            "open",
            expected_asof="2026-07-26",
        )


def test_delivery_receipt_reader_rejects_symlink(tmp_path):
    snapshot = _snapshot(tmp_path)
    receipt_root = tmp_path / "receipts"
    path = receipt_module.receipt_path(snapshot, root=receipt_root)
    path.parent.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    path.symlink_to(outside)

    with pytest.raises(
        receipt_module.DeliveryReceiptError,
        match="regular non-symlink",
    ):
        read_delivery_receipt(snapshot, root=receipt_root)


def test_delivery_receipt_lock_rejects_symlink_without_touching_target(
    tmp_path,
):
    snapshot = _snapshot(tmp_path)
    receipt_root = tmp_path / "receipts"
    path = receipt_module.receipt_path(snapshot, root=receipt_root)
    path.parent.mkdir(parents=True)
    target = tmp_path / "outside.lock"
    target.write_text("do not touch", encoding="utf-8")
    lock_path = path.with_name(f".{path.name}.lock")
    lock_path.symlink_to(target)

    with pytest.raises(receipt_module.DeliveryReceiptError, match="lock"):
        with receipt_module._exclusive_receipt_lock(path):
            pass

    assert target.read_text(encoding="utf-8") == "do not touch"


def test_delivery_receipt_writer_rejects_symlink_without_touching_target(
    tmp_path,
):
    path = tmp_path / "receipt.json"
    target = tmp_path / "outside.json"
    target.write_text("do not touch", encoding="utf-8")
    path.symlink_to(target)

    with pytest.raises(
        receipt_module.DeliveryReceiptError,
        match="written safely",
    ):
        receipt_module._atomic_write(path, {"delivery_ok": True})

    assert target.read_text(encoding="utf-8") == "do not touch"


@pytest.mark.parametrize("operation", ["lock", "write"])
def test_delivery_receipt_rejects_symlinked_parent(tmp_path, operation):
    actual_parent = tmp_path / "outside"
    actual_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(actual_parent, target_is_directory=True)
    path = linked_parent / "receipt.json"

    with pytest.raises(receipt_module.DeliveryReceiptError):
        if operation == "lock":
            with receipt_module._exclusive_receipt_lock(path):
                pass
        else:
            receipt_module._atomic_write(path, {"delivery_ok": True})

    assert list(actual_parent.iterdir()) == []


def test_postactivation_receipt_rejects_attempt_outside_live_window(
    tmp_path,
):
    attempted_at = datetime.fromisoformat("2026-07-27T03:00:00+00:00")

    with pytest.raises(
        receipt_module.DeliveryReceiptError,
        match="attempted_at outside live send window",
    ):
        receipt_module.validate_telegram_transport_evidence(
            {
                "message_sha256": "a" * 64,
                "chat_id_sha256": None,
                "chunk_count": 1,
                "telegram_messages": [],
            },
            decision_day=receipt_module.date(2026, 7, 27),
            live_start=receipt_module.dt_time(9, 0),
            live_end=receipt_module.dt_time(9, 21),
            delivery_ok=False,
            attempted_at=attempted_at,
            sent_at=None,
            recorded_at=attempted_at,
            path=tmp_path / "receipt.json",
        )


def _snapshot(root: Path) -> dict:
    return {
        "asof": "2026-07-25",
        "slot": "open",
        "snapshot_id": "recommend-test-2026-07-25-open",
        "snapshot_path": str(root / "snapshots" / "2026-07-25" / "open_r1.json"),
        "btc_regime": "neutral",
        "universe_n": 0,
        "calibration_source": "test",
        "rank_basis": recommend_send.APPROVED_LIVE_R1_RANK_BASIS,
        "snapshot_schema": recommend_send.SNAPSHOT_SCHEMA_VERSION,
        "score_schema_version": recommend_send.APPROVED_LIVE_SCORE_SCHEMA,
        "rule_version": recommend_send.APPROVED_LIVE_R1_RULE_VERSION,
        "decision_started_at": "2026-07-25T00:05:00+00:00",
        "decision_completed_at": "2026-07-25T00:05:00+00:00",
        "ranking": "R1",
        "request": {
            "asof": "2026-07-25",
            "slot": "open",
            "ranking": "R1",
            "limit_markets": None,
        },
        "top3": [],
        "model": {
            "id": "recommend_r1_open",
            "ranking": "R1",
        },
    }


def _telegram_result(
    message: str,
    *,
    delivery_ok: bool = True,
    server_dates: tuple[str, ...] = (
        "2026-07-27T00:05:01+00:00",
    ),
    error: str | None = None,
) -> TelegramSendResult:
    chunks = [message[index:index + 4000] for index in range(0, len(message), 4000)]
    delivered = tuple(
        TelegramServerMessage(
            message_id=100 + index,
            server_date=server_date,
            text_sha256=hashlib.sha256(
                chunks[index - 1].encode("utf-8")
            ).hexdigest(),
        )
        for index, server_date in enumerate(server_dates, start=1)
    )
    return TelegramSendResult(
        delivery_ok=delivery_ok,
        message_sha256=hashlib.sha256(message.encode("utf-8")).hexdigest(),
        chunk_count=len(chunks),
        chat_id_sha256=hashlib.sha256(b"456").hexdigest(),
        telegram_messages=delivered,
        error=error,
    )


def _champion_state(asof: str, history: list[dict]) -> dict:
    state = {
        "schema_version": champion_selector.STATE_SCHEMA_VERSION,
        "asof": asof,
        "updated_at": f"{asof}T00:30:00+00:00",
        "config": champion_selector._expected_config(),
        "slots": {
            "open": {
                "champion_id": "recommend_r1_open",
                "since": asof,
                "is_fallback": False,
                "metric": None,
                "reason": "test",
            },
            "preopen": {
                "champion_id": "recommend_r1_preopen",
                "since": asof,
                "is_fallback": True,
                "metric": None,
                "reason": "test",
            },
        },
        "streaks": {},
        "history": history,
    }
    state["payload_sha256"] = champion_selector._state_digest(state)
    return state


def _stub_dispatch(monkeypatch, snapshot: dict) -> None:
    spec = SimpleNamespace(
        id="recommend_r1_open",
        predict_ref="signals.recommend:score_candidates",
    )
    monkeypatch.setattr(
        recommend_send,
        "resolve_champion",
        lambda slot, **_kwargs: (spec, False, "test"),
    )
    monkeypatch.setattr(
        recommend_send,
        "call_predict",
        lambda *args, **kwargs: snapshot,
    )
    monkeypatch.setattr(
        recommend_send,
        "maybe_notify_champion_change",
        lambda *args, **kwargs: None,
    )
    hour, minute = (8, 50) if snapshot["slot"] == "preopen" else (9, 5)
    if snapshot["slot"] == "preopen":
        snapshot["request"]["slot"] = "preopen"
        snapshot["decision_started_at"] = (
            f"{snapshot['asof']}T08:50:00+09:00"
        )
        snapshot["decision_completed_at"] = (
            f"{snapshot['asof']}T08:50:01+09:00"
        )
    decision_day = datetime.fromisoformat(snapshot["asof"])
    monkeypatch.setattr(
        recommend_send,
        "_now_kst",
        lambda: decision_day.replace(
            hour=hour,
            minute=minute,
            second=2,
            tzinfo=recommend_send.KST,
        ),
    )


def _concurrent_send_worker(
    snapshot: dict,
    receipt_root: str,
    start: Any,
    results: Any,
    api_calls: Any,
) -> None:
    """Run one isolated sender with spawn-safe in-child test doubles."""
    # spawn 자식은 부모의 kill-switch-해제 fixture 환경을 물려받고 부모의
    # monkeypatch 는 하나도 살아남지 않는다 — 아래 수동 stub 이 커버하지
    # 못하는 어떤 경로(post-activation receipt 전송 등)도 실 텔레그램에
    # 닿지 못하게 자식 첫 줄에서 재봉쇄한다.
    os.environ["PRELUDE_FORBID_TELEGRAM"] = "1"
    spec = SimpleNamespace(
        id="recommend_r1_open",
        predict_ref="signals.recommend:score_candidates",
    )
    recommend_send.resolve_champion = (
        lambda slot, **_kwargs: (spec, False, "test")
    )
    recommend_send.call_predict = lambda *args, **kwargs: snapshot
    recommend_send.maybe_notify_champion_change = (
        lambda *args, **kwargs: None
    )
    decision_day = datetime.fromisoformat(snapshot["asof"])
    recommend_send._now_kst = lambda: decision_day.replace(
        hour=9,
        minute=5,
        second=2,
        tzinfo=recommend_send.KST,
    )

    def fake_send(message, **_kwargs):
        with api_calls.get_lock():
            api_calls.value += 1
        time.sleep(0.15)
        return True

    recommend_send.send_telegram = fake_send
    start.wait()
    try:
        result = recommend_send.send_recommendation(
            snapshot["asof"],
            snapshot["slot"],
            receipt_root=receipt_root,
        )
    except Exception as exc:  # pragma: no cover - surfaced in parent assertion
        results.put(("error", repr(exc)))
        raise
    results.put(("ok", result))


def test_sequential_rerun_sends_same_snapshot_exactly_once(
    tmp_path, monkeypatch
):
    snapshot = _snapshot(tmp_path)
    receipt_root = tmp_path / "receipts"
    calls: list[str] = []
    _stub_dispatch(monkeypatch, snapshot)
    monkeypatch.setattr(
        recommend_send,
        "send_telegram",
        lambda message, **kwargs: calls.append(message) or True,
    )

    assert recommend_send.send_recommendation(
        snapshot["asof"], snapshot["slot"], receipt_root=receipt_root
    )
    assert recommend_send.send_recommendation(
        snapshot["asof"], snapshot["slot"], receipt_root=receipt_root
    )

    assert len(calls) == 1
    receipt = read_delivery_receipt(snapshot, root=receipt_root)
    assert receipt is not None
    assert receipt["delivery_ok"] is True


def test_dry_run_is_not_skipped_by_persistent_success_receipt(
    tmp_path, monkeypatch
):
    snapshot = _snapshot(tmp_path)
    receipt_root = tmp_path / "receipts"
    calls: list[bool] = []
    _stub_dispatch(monkeypatch, snapshot)

    def fake_send(message, *, dry_run=False, **_kwargs):
        calls.append(dry_run)
        return True

    monkeypatch.setattr(recommend_send, "send_telegram", fake_send)

    assert recommend_send.send_recommendation(
        snapshot["asof"], snapshot["slot"], receipt_root=receipt_root
    )
    assert recommend_send.send_recommendation(
        snapshot["asof"],
        snapshot["slot"],
        dry_run=True,
        receipt_root=receipt_root,
    )

    assert calls == [False, True]


@pytest.mark.parametrize(
    "rank_basis",
    [
        None,
        "",
        "score_rank(fallback)",
        "R1_riskreward(de-corr head, degraded)",
    ],
)
def test_real_send_rejects_unapproved_r1_rank_basis(
    tmp_path,
    monkeypatch,
    rank_basis,
):
    snapshot = _snapshot(tmp_path)
    snapshot["rank_basis"] = rank_basis
    calls: list[str] = []
    _stub_dispatch(monkeypatch, snapshot)
    monkeypatch.setattr(
        recommend_send,
        "send_telegram",
        lambda *_args, **_kwargs: calls.append("send") or True,
    )
    monkeypatch.setattr(
        recommend_send,
        "send_telegram_with_receipt",
        lambda *_args, **_kwargs: calls.append("send-with-receipt"),
    )

    with pytest.raises(RuntimeError, match="unapproved live rank basis"):
        recommend_send.send_recommendation(
            snapshot["asof"],
            snapshot["slot"],
            receipt_root=tmp_path / "receipts",
        )

    assert calls == []
    assert not (tmp_path / "receipts").exists()


def test_dry_run_preserves_fallback_rank_diagnostic(tmp_path, monkeypatch):
    snapshot = _snapshot(tmp_path)
    snapshot["rank_basis"] = "score_rank(fallback)"
    calls: list[bool] = []
    _stub_dispatch(monkeypatch, snapshot)
    monkeypatch.setattr(
        recommend_send,
        "send_telegram",
        lambda _message, *, dry_run=False: calls.append(dry_run) or True,
    )

    assert recommend_send.send_recommendation(
        snapshot["asof"],
        snapshot["slot"],
        dry_run=True,
        receipt_root=tmp_path / "receipts",
    )
    assert calls == [True]
    assert not (tmp_path / "receipts").exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "snapshot_schema",
            "recommend_snapshot.v1",
            "legacy/unversioned snapshot",
        ),
        (
            "decision_completed_at",
            "2026-07-25T09:30:00+09:00",
            "outside open decision window",
        ),
        (
            "decision_completed_at",
            "2026-07-25T09:10:00+09:00",
            "future-dated",
        ),
    ],
)
def test_real_send_rejects_nonforward_snapshot_provenance(
    tmp_path,
    monkeypatch,
    field,
    value,
    message,
):
    snapshot = _snapshot(tmp_path)
    snapshot[field] = value
    calls: list[str] = []
    _stub_dispatch(monkeypatch, snapshot)
    # Apply the adversarial mutation after the helper has normalised slot
    # timestamps.
    snapshot[field] = value
    monkeypatch.setattr(
        recommend_send,
        "send_telegram",
        lambda *_args, **_kwargs: calls.append("send") or True,
    )

    with pytest.raises(RuntimeError, match=message):
        recommend_send.send_recommendation(
            snapshot["asof"],
            snapshot["slot"],
            receipt_root=tmp_path / "receipts",
        )

    assert calls == []
    assert not (tmp_path / "receipts").exists()


@pytest.mark.parametrize(
    ("top_level_ranking", "model_ranking"),
    [
        ("R2", "R1"),
        ("R1", "R2"),
        (None, "R1"),
    ],
)
def test_real_send_rejects_ranking_identity_drift(
    tmp_path,
    monkeypatch,
    top_level_ranking,
    model_ranking,
):
    snapshot = _snapshot(tmp_path)
    snapshot["ranking"] = top_level_ranking
    snapshot["model"]["ranking"] = model_ranking
    calls: list[str] = []
    _stub_dispatch(monkeypatch, snapshot)
    monkeypatch.setattr(
        recommend_send,
        "send_telegram",
        lambda *_args, **_kwargs: calls.append("send") or True,
    )

    with pytest.raises(RuntimeError, match="unapproved live ranking identity"):
        recommend_send.send_recommendation(
            snapshot["asof"],
            snapshot["slot"],
            receipt_root=tmp_path / "receipts",
        )

    assert calls == []
    assert not (tmp_path / "receipts").exists()


def test_failed_receipt_is_retried_until_one_success(tmp_path, monkeypatch):
    snapshot = _snapshot(tmp_path)
    receipt_root = tmp_path / "receipts"
    outcomes = iter([False, True])
    calls = 0
    _stub_dispatch(monkeypatch, snapshot)

    def fake_send(message, **_kwargs):
        nonlocal calls
        calls += 1
        return next(outcomes)

    monkeypatch.setattr(recommend_send, "send_telegram", fake_send)

    assert not recommend_send.send_recommendation(
        snapshot["asof"], snapshot["slot"], receipt_root=receipt_root
    )
    assert recommend_send.send_recommendation(
        snapshot["asof"], snapshot["slot"], receipt_root=receipt_root
    )

    assert calls == 2
    receipt = read_delivery_receipt(snapshot, root=receipt_root)
    assert receipt is not None
    assert receipt["delivery_ok"] is True


def test_false_transport_result_records_explicit_failure_reason(
    tmp_path,
    monkeypatch,
):
    snapshot = _snapshot(tmp_path)
    receipt_root = tmp_path / "receipts"
    _stub_dispatch(monkeypatch, snapshot)
    monkeypatch.setattr(
        recommend_send,
        "send_telegram",
        lambda *_args, **_kwargs: False,
    )

    assert not recommend_send.send_recommendation(
        snapshot["asof"],
        snapshot["slot"],
        receipt_root=receipt_root,
    )

    receipt = read_delivery_receipt(snapshot, root=receipt_root)
    assert receipt is not None
    assert receipt["delivery_ok"] is False
    assert receipt["sent_at"] is None
    assert receipt["error"] == "send_telegram returned false"


def test_sender_rejects_postactivation_receipt_outer_tamper(
    tmp_path,
    monkeypatch,
):
    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            value = datetime(
                2026,
                7,
                27,
                0,
                5,
                2,
                tzinfo=receipt_module.timezone.utc,
            )
            return value if tz is None else value.astimezone(tz)

    snapshot = _snapshot(tmp_path)
    snapshot.update(
        {
            "asof": "2026-07-27",
            "snapshot_id": "recommend-test-2026-07-27-open",
            "decision_started_at": "2026-07-27T00:05:00+00:00",
            "decision_completed_at": "2026-07-27T00:05:01+00:00",
        }
    )
    snapshot["request"]["asof"] = "2026-07-27"
    snapshot["snapshot_path"] = str(
        tmp_path / "snapshots/2026-07-27/open_r1.json"
    )
    receipt_root = tmp_path / "receipts"
    monkeypatch.setattr(receipt_module, "datetime", FixedDatetime)
    path = recommend_send.write_delivery_receipt(
        snapshot,
        delivery_ok=True,
        attempted_at="2026-07-27T00:05:00+00:00",
        sent_at="2026-07-27T00:05:01+00:00",
        telegram_result=_telegram_result("radar"),
        message="radar",
        root=receipt_root,
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    document["sent_at"] = "2026-07-27T00:05:00.500000+00:00"
    path.write_text(json.dumps(document), encoding="utf-8")
    _stub_dispatch(monkeypatch, snapshot)
    calls: list[str] = []
    monkeypatch.setattr(
        recommend_send,
        "send_telegram",
        lambda *_args, **_kwargs: calls.append("send") or True,
    )
    monkeypatch.setattr(
        recommend_send,
        "send_telegram_with_receipt",
        lambda *_args, **_kwargs: calls.append("send-with-receipt"),
    )

    with pytest.raises(
        receipt_module.DeliveryReceiptError,
        match="outer integrity mismatch",
    ):
        recommend_send.send_recommendation(
            "2026-07-27",
            "open",
            receipt_root=receipt_root,
        )

    assert calls == []


def test_postactivation_sender_records_server_bound_transport_evidence(
    tmp_path,
    monkeypatch,
):
    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            value = datetime.fromisoformat(
                "2026-07-27T00:05:01.500000+00:00"
            )
            return value if tz is None else value.astimezone(tz)

    snapshot = _snapshot(tmp_path)
    snapshot.update(
        {
            "asof": "2026-07-27",
            "snapshot_id": "recommend-test-2026-07-27-open",
            "decision_started_at": "2026-07-27T00:05:00+00:00",
            "decision_completed_at": "2026-07-27T00:05:01+00:00",
        }
    )
    snapshot["request"]["asof"] = snapshot["asof"]
    snapshot["snapshot_path"] = str(
        tmp_path / "snapshots/2026-07-27/open_r1.json"
    )
    _stub_dispatch(monkeypatch, snapshot)
    monkeypatch.setattr(recommend_send, "datetime", FixedDatetime)
    monkeypatch.setattr(receipt_module, "datetime", FixedDatetime)
    messages: list[str] = []

    def fake_detailed(message, **_kwargs):
        messages.append(message)
        return _telegram_result(
            message,
            server_dates=("2026-07-27T00:05:01+00:00",),
        )

    monkeypatch.setattr(
        recommend_send,
        "send_telegram_with_receipt",
        fake_detailed,
    )
    monkeypatch.setattr(
        recommend_send,
        "send_telegram",
        lambda *_args, **_kwargs: pytest.fail(
            "post-activation live send must use detailed transport"
        ),
    )

    assert recommend_send.send_recommendation(
        snapshot["asof"],
        snapshot["slot"],
        receipt_root=tmp_path / "receipts",
    )

    receipt = read_delivery_receipt(
        snapshot,
        root=tmp_path / "receipts",
    )
    assert receipt is not None
    assert len(messages) == 1
    assert receipt["message_sha256"] == hashlib.sha256(
        messages[0].encode()
    ).hexdigest()
    assert receipt["chunk_count"] == 1
    assert receipt["sent_at"] == "2026-07-27T00:05:01+00:00"
    assert receipt["telegram_messages"][0]["message_id"] == 101


def test_postactivation_partial_delivery_is_preserved_and_not_retried(
    tmp_path,
    monkeypatch,
):
    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            value = datetime.fromisoformat(
                "2026-07-27T00:05:01.500000+00:00"
            )
            return value if tz is None else value.astimezone(tz)

    snapshot = _snapshot(tmp_path)
    snapshot.update(
        {
            "asof": "2026-07-27",
            "snapshot_id": "recommend-test-2026-07-27-open",
            "decision_started_at": "2026-07-27T00:05:00+00:00",
            "decision_completed_at": "2026-07-27T00:05:01+00:00",
        }
    )
    snapshot["request"]["asof"] = snapshot["asof"]
    _stub_dispatch(monkeypatch, snapshot)
    monkeypatch.setattr(recommend_send, "datetime", FixedDatetime)
    monkeypatch.setattr(receipt_module, "datetime", FixedDatetime)
    calls = 0
    message = "x" * 4001

    def fake_detailed(actual_message, **_kwargs):
        nonlocal calls
        calls += 1
        return _telegram_result(
            actual_message,
            delivery_ok=False,
            server_dates=("2026-07-27T00:05:01+00:00",),
            error="second chunk rejected",
        )

    monkeypatch.setattr(
        recommend_send,
        "send_telegram_with_receipt",
        fake_detailed,
    )
    receipt_root = tmp_path / "receipts"

    assert not recommend_send._send_and_record(
        snapshot,
        message,
        slot="open",
        receipt_root=receipt_root,
    )
    receipt = read_delivery_receipt(snapshot, root=receipt_root)
    assert receipt is not None
    assert receipt["delivery_ok"] is False
    assert receipt["sent_at"] is None
    assert len(receipt["telegram_messages"]) == 1

    with pytest.raises(RuntimeError, match="partial/ambiguous delivery"):
        recommend_send._send_and_record(
            snapshot,
            message,
            slot="open",
            receipt_root=receipt_root,
        )
    assert calls == 1


def test_postactivation_ambiguous_acceptance_is_not_retried(
    tmp_path,
    monkeypatch,
):
    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            value = datetime.fromisoformat(
                "2026-07-27T00:05:01.500000+00:00"
            )
            return value if tz is None else value.astimezone(tz)

    snapshot = _snapshot(tmp_path)
    snapshot.update(
        {
            "asof": "2026-07-27",
            "snapshot_id": "recommend-test-2026-07-27-open",
            "decision_started_at": "2026-07-27T00:05:00+00:00",
            "decision_completed_at": "2026-07-27T00:05:01+00:00",
        }
    )
    snapshot["request"]["asof"] = snapshot["asof"]
    _stub_dispatch(monkeypatch, snapshot)
    monkeypatch.setattr(recommend_send, "datetime", FixedDatetime)
    monkeypatch.setattr(receipt_module, "datetime", FixedDatetime)
    calls = 0
    message = "radar"

    def fake_detailed(actual_message, **_kwargs):
        nonlocal calls
        calls += 1
        return _telegram_result(
            actual_message,
            delivery_ok=False,
            server_dates=(),
            error=(
                "ambiguous Telegram acceptance: "
                "response text mismatch"
            ),
        )

    monkeypatch.setattr(
        recommend_send,
        "send_telegram_with_receipt",
        fake_detailed,
    )
    receipt_root = tmp_path / "receipts"

    assert not recommend_send._send_and_record(
        snapshot,
        message,
        slot="open",
        receipt_root=receipt_root,
    )
    receipt = read_delivery_receipt(snapshot, root=receipt_root)
    assert receipt is not None
    assert receipt["telegram_messages"] == []
    assert recommend_send.telegram_error_is_ambiguous(receipt["error"])

    with pytest.raises(RuntimeError, match="partial/ambiguous delivery"):
        recommend_send._send_and_record(
            snapshot,
            message,
            slot="open",
            receipt_root=receipt_root,
        )
    assert calls == 1


def test_concurrent_processes_send_same_snapshot_exactly_once(tmp_path):
    context = multiprocessing.get_context("spawn")
    snapshot = _snapshot(tmp_path)
    receipt_root = tmp_path / "receipts"
    api_calls = context.Value("i", 0)
    worker_count = 2
    start = context.Barrier(worker_count + 1)
    results = context.Queue()
    processes = [
        context.Process(
            target=_concurrent_send_worker,
            args=(
                snapshot,
                str(receipt_root),
                start,
                results,
                api_calls,
            ),
        )
        for _ in range(worker_count)
    ]
    try:
        for process in processes:
            process.start()
        start.wait(timeout=10)
        for process in processes:
            process.join(timeout=10)
        assert [process.exitcode for process in processes] == [0, 0]
        assert sorted(results.get(timeout=2) for _ in processes) == [
            ("ok", True),
            ("ok", True),
        ]
        assert api_calls.value == 1
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        results.close()
        results.join_thread()


def test_crash_after_api_success_before_receipt_can_duplicate(
    tmp_path, monkeypatch
):
    """Bot API에 idempotency key가 없어 이 crash window는 정직하게 남는다."""

    class SimulatedCrash(RuntimeError):
        pass

    snapshot = _snapshot(tmp_path)
    receipt_root = tmp_path / "receipts"
    actual_write = recommend_send.write_delivery_receipt
    write_calls = 0
    api_calls = 0
    _stub_dispatch(monkeypatch, snapshot)

    def fake_send(message, **_kwargs):
        nonlocal api_calls
        api_calls += 1
        return True

    def crash_once(*args, **kwargs):
        nonlocal write_calls
        write_calls += 1
        if write_calls == 1:
            raise SimulatedCrash("process died after Telegram accepted the message")
        return actual_write(*args, **kwargs)

    monkeypatch.setattr(recommend_send, "send_telegram", fake_send)
    monkeypatch.setattr(recommend_send, "write_delivery_receipt", crash_once)

    with pytest.raises(SimulatedCrash):
        recommend_send.send_recommendation(
            snapshot["asof"],
            snapshot["slot"],
            receipt_root=receipt_root,
        )
    assert recommend_send.send_recommendation(
        snapshot["asof"], snapshot["slot"], receipt_root=receipt_root
    )

    assert api_calls == 2
    receipt = read_delivery_receipt(snapshot, root=receipt_root)
    assert receipt is not None
    assert receipt["delivery_ok"] is True


@pytest.mark.parametrize(
    ("now", "message"),
    [
        (
            datetime(
                2026,
                7,
                26,
                9,
                5,
                tzinfo=recommend_send.KST,
            ),
            "stale live send rejected",
        ),
        (
            datetime(
                2026,
                7,
                25,
                14,
                0,
                tzinfo=recommend_send.KST,
            ),
            "outside open live send window",
        ),
    ],
)
def test_real_send_rejects_wrong_day_or_late_window_before_scoring(
    tmp_path, monkeypatch, now, message
):
    snapshot = _snapshot(tmp_path)
    calls: list[str] = []
    _stub_dispatch(monkeypatch, snapshot)
    monkeypatch.setattr(recommend_send, "_now_kst", lambda: now)
    monkeypatch.setattr(
        recommend_send,
        "call_predict",
        lambda *_args, **_kwargs: calls.append("score") or snapshot,
    )
    monkeypatch.setattr(
        recommend_send,
        "send_telegram",
        lambda *_args, **_kwargs: calls.append("send") or True,
    )

    with pytest.raises(RuntimeError, match=message):
        recommend_send.send_recommendation(
            snapshot["asof"],
            snapshot["slot"],
            snapshot_root=tmp_path / "snapshots",
            receipt_root=tmp_path / "receipts",
        )

    assert calls == []
    assert not (tmp_path / "snapshots").exists()
    assert not (tmp_path / "receipts").exists()


def test_real_send_rejects_market_limited_snapshot_before_scoring(
    tmp_path,
    monkeypatch,
):
    snapshot = _snapshot(tmp_path)
    calls: list[str] = []
    _stub_dispatch(monkeypatch, snapshot)
    monkeypatch.setattr(
        recommend_send,
        "call_predict",
        lambda *_args, **_kwargs: calls.append("score") or snapshot,
    )
    # 가드가 회귀하면 실 전송 경로로 떨어진다(이 모듈은 kill-switch 해제) —
    # 이웃 window 테스트처럼 transport 를 recorder 로 봉쇄하고 0건 단언.
    monkeypatch.setattr(
        recommend_send,
        "send_telegram",
        lambda *_a, **_k: calls.append("send") or True,
    )
    monkeypatch.setattr(
        recommend_send,
        "send_telegram_with_receipt",
        lambda *_a, **_k: calls.append("receipt") or None,
    )

    with pytest.raises(RuntimeError, match="dry-run-only"):
        recommend_send.send_recommendation(
            snapshot["asof"],
            snapshot["slot"],
            limit_markets=10,
            receipt_root=tmp_path / "receipts",
        )

    assert calls == []
    assert not (tmp_path / "receipts").exists()


@pytest.mark.parametrize(
    ("slot", "start", "before_side_effect"),
    [
        (
            "preopen",
            datetime(2026, 7, 25, 8, 59, 59, tzinfo=recommend_send.KST),
            datetime(2026, 7, 25, 9, 0, tzinfo=recommend_send.KST),
        ),
        (
            "open",
            datetime(2026, 7, 25, 9, 20, 59, tzinfo=recommend_send.KST),
            datetime(2026, 7, 25, 9, 21, tzinfo=recommend_send.KST),
        ),
    ],
)
def test_live_send_rechecks_window_after_scoring_before_telegram(
    tmp_path,
    monkeypatch,
    slot,
    start,
    before_side_effect,
):
    snapshot = _snapshot(tmp_path)
    snapshot["slot"] = slot
    clock = iter([start, start, before_side_effect])
    calls: list[str] = []
    _stub_dispatch(monkeypatch, snapshot)
    monkeypatch.setattr(recommend_send, "_now_kst", lambda: next(clock))
    monkeypatch.setattr(
        recommend_send,
        "send_telegram",
        lambda *_args, **_kwargs: calls.append("send") or True,
    )

    with pytest.raises(RuntimeError, match=f"outside {slot} live send window"):
        recommend_send.send_recommendation(
            snapshot["asof"],
            slot,
            receipt_root=tmp_path / "receipts",
        )

    assert calls == []
    assert not list((tmp_path / "receipts").rglob("*.json"))


@pytest.mark.parametrize("state_asof", ["2026-07-25", "2026-07-24"])
def test_champion_notice_receipt_prevents_duplicate_on_recommend_retry(
    tmp_path,
    monkeypatch,
    state_asof,
):
    snapshot = _snapshot(tmp_path)
    state_path = tmp_path / "champion_state.json"
    state_path.write_text(
        json.dumps(
            _champion_state(
                state_asof,
                [
                    {
                        "asof": state_asof,
                        "slot": snapshot["slot"],
                        "from": "recommend-old",
                        "to": "recommend_r1_open",
                        "reason": "test promotion",
                    }
                ],
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(recommend_send, "CHAMPION_STATE_PATH", state_path)
    monkeypatch.setattr(
        recommend_send,
        "_now_kst",
        lambda: datetime(
            2026,
            7,
            25,
            9,
            5,
            tzinfo=recommend_send.KST,
        ),
    )
    messages: list[str] = []
    send_options: list[dict] = []
    recommendation_attempts = 0

    def fake_send(message, **kwargs):
        nonlocal recommendation_attempts
        messages.append(message)
        send_options.append(kwargs)
        if message == "recommendation":
            recommendation_attempts += 1
            if recommendation_attempts == 1:
                raise RuntimeError("recommendation transport failed")
        return True

    monkeypatch.setattr(recommend_send, "send_telegram", fake_send)
    receipt_root = tmp_path / "receipts"

    with pytest.raises(RuntimeError, match="recommendation transport failed"):
        recommend_send._send_and_record(
            snapshot,
            "recommendation",
            slot=snapshot["slot"],
            receipt_root=receipt_root,
        )
    assert recommend_send._send_and_record(
        snapshot,
        "recommendation",
        slot=snapshot["slot"],
        receipt_root=receipt_root,
    )

    notices = [
        message for message in messages if message.startswith("ℹ️ 챔피언")
    ]
    assert len(notices) == 1
    assert recommendation_attempts == 2
    assert len(send_options) == 3
    assert all(
        options["deadline"]
        == datetime(
            2026,
            7,
            25,
            9,
            21,
            tzinfo=recommend_send.KST,
        )
        for options in send_options
    )
    assert all(
        options["clock"] is recommend_send._now_kst
        for options in send_options
    )
    receipt = read_delivery_receipt(snapshot, root=receipt_root)
    assert receipt is not None
    assert receipt["delivery_ok"] is True


def test_champion_notice_rejects_duplicate_key_state_without_send(
    tmp_path,
    monkeypatch,
):
    state_path = tmp_path / "champion_state.json"
    payload = json.dumps(_champion_state("2026-07-25", []))
    state_path.write_text(
        payload.replace(
            '"asof": "2026-07-25"',
            '"asof": "2026-07-25", "asof": "2026-07-25"',
            1,
        ),
        encoding="utf-8",
    )
    calls: list[str] = []
    monkeypatch.setattr(recommend_send, "CHAMPION_STATE_PATH", state_path)
    monkeypatch.setattr(
        recommend_send,
        "send_telegram",
        lambda *_args, **_kwargs: calls.append("send") or True,
    )

    assert recommend_send.maybe_notify_champion_change(
        "open",
        dry_run=False,
        receipt_root=tmp_path / "receipts",
        live_asof="2026-07-25",
    ) is None
    assert calls == []


def test_champion_notice_rejects_symlinked_state_without_send(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "state-target.json"
    target.write_text(
        json.dumps(_champion_state("2026-07-25", [])),
        encoding="utf-8",
    )
    state_path = tmp_path / "champion_state.json"
    state_path.symlink_to(target)
    calls: list[str] = []
    monkeypatch.setattr(recommend_send, "CHAMPION_STATE_PATH", state_path)
    monkeypatch.setattr(
        recommend_send,
        "send_telegram",
        lambda *_args, **_kwargs: calls.append("send") or True,
    )

    assert recommend_send.maybe_notify_champion_change(
        "open",
        dry_run=False,
        receipt_root=tmp_path / "receipts",
        live_asof="2026-07-25",
    ) is None
    assert calls == []


def test_champion_notice_missing_validated_asof_fails_closed_without_assert(
    tmp_path,
    monkeypatch,
):
    state_path = tmp_path / "champion_state.json"
    state_path.write_text("{}", encoding="utf-8")
    calls: list[str] = []
    monkeypatch.setattr(recommend_send, "CHAMPION_STATE_PATH", state_path)
    monkeypatch.setattr(
        recommend_send,
        "send_telegram",
        lambda *_args, **_kwargs: calls.append("send") or True,
    )

    assert recommend_send.maybe_notify_champion_change(
        "open",
        dry_run=False,
        receipt_root=tmp_path / "receipts",
        live_asof="2026-07-25",
    ) is None
    assert calls == []


def test_primary_radar_is_durably_recorded_before_auxiliary_notice(
    tmp_path,
    monkeypatch,
):
    snapshot = _snapshot(tmp_path)
    receipt_root = tmp_path / "receipts"
    events: list[str] = []
    send_options: list[dict] = []

    def fake_send(_message, **kwargs):
        send_options.append(kwargs)
        events.append("primary")
        return True

    monkeypatch.setattr(
        recommend_send,
        "send_telegram",
        fake_send,
    )

    def fake_notice(*_args, **_kwargs):
        receipt = read_delivery_receipt(snapshot, root=receipt_root)
        assert receipt is not None
        assert receipt["delivery_ok"] is True
        events.append("notice")

    monkeypatch.setattr(
        recommend_send,
        "maybe_notify_champion_change",
        fake_notice,
    )
    monkeypatch.setattr(
        recommend_send,
        "_now_kst",
        lambda: datetime(
            2026,
            7,
            25,
            9,
            5,
            tzinfo=recommend_send.KST,
        ),
    )

    assert recommend_send._send_and_record(
        snapshot,
        "recommendation",
        slot="open",
        receipt_root=receipt_root,
    )
    assert events == ["primary", "notice"]
    assert send_options == [
        {
            "deadline": datetime(
                2026,
                7,
                25,
                9,
                21,
                tzinfo=recommend_send.KST,
            ),
            "clock": recommend_send._now_kst,
        }
    ]


@pytest.mark.parametrize(
    ("slot", "hour", "minute", "allowed"),
    [
        ("preopen", 8, 44, False),
        ("preopen", 8, 45, True),
        ("preopen", 8, 59, True),
        ("preopen", 9, 0, False),
        ("open", 8, 59, False),
        ("open", 9, 0, True),
        ("open", 9, 20, True),
        ("open", 9, 21, False),
    ],
)
def test_live_send_slot_window_boundaries(slot, hour, minute, allowed):
    now = datetime(
        2026,
        7,
        25,
        hour,
        minute,
        tzinfo=recommend_send.KST,
    )
    if allowed:
        recommend_send._assert_live_send_window(
            "2026-07-25",
            slot,
            now=now,
        )
    else:
        with pytest.raises(RuntimeError, match="outside"):
            recommend_send._assert_live_send_window(
                "2026-07-25",
                slot,
                now=now,
            )

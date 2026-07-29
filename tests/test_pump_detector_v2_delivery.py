from __future__ import annotations

import json
import hashlib
import math
import multiprocessing
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

import scripts.pump_detector_v2_today as runner
from notifier.telegram import TelegramSendResult, TelegramServerMessage


@pytest.fixture(autouse=True)
def _allow_mocked_sends(monkeypatch, tmp_path):
    # 이 모듈은 발송 경로 자체를 mock 된 requests 로 검증한다 —
    # 전역 kill-switch(tests/conftest.py)를 in-process 한정 해제.
    # subprocess 를 띄우는 테스트는 이 모듈에 두지 말 것.
    # guard 회귀가 나도 실제 forward output에 decision/receipt/ledger를
    # 기록하지 못하도록 모든 기본 경로도 테스트별 tmp로 격리한다.
    monkeypatch.delenv("PRELUDE_FORBID_TELEGRAM", raising=False)
    monkeypatch.setattr(
        runner,
        "PUMP_V2_LEDGER",
        str(tmp_path / "default-pump-v2-ledger.csv"),
    )
    monkeypatch.setattr(
        runner,
        "PUMP_V2_RECEIPT_ROOT",
        str(tmp_path / "default-pump-v2-receipts"),
    )
    monkeypatch.setattr(
        runner,
        "PUMP_V2_DECISION_ROOT",
        str(tmp_path / "default-pump-v2-decisions"),
    )

    def forbid_unmocked_transport(*_args, **_kwargs):
        pytest.fail("unmocked Telegram transport reached")

    monkeypatch.setattr(runner, "send_telegram", forbid_unmocked_transport)
    monkeypatch.setattr(
        runner,
        "send_telegram_with_receipt",
        forbid_unmocked_transport,
    )


def _telegram_result(
    message: str,
    *,
    delivery_ok: bool = True,
    server_dates: tuple[str, ...] = (
        "2026-07-27T00:05:00+00:00",
    ),
    error: str | None = None,
) -> TelegramSendResult:
    chunks = [
        message[index:index + 4000]
        for index in range(0, len(message), 4000)
    ]
    delivered = tuple(
        TelegramServerMessage(
            message_id=100 + index,
            server_date=server_date,
            text_sha256=hashlib.sha256(
                chunks[index - 1].encode()
            ).hexdigest(),
        )
        for index, server_date in enumerate(server_dates, start=1)
    )
    return TelegramSendResult(
        delivery_ok=delivery_ok,
        message_sha256=hashlib.sha256(message.encode()).hexdigest(),
        chunk_count=len(chunks),
        chat_id_sha256=hashlib.sha256(b"456").hexdigest(),
        telegram_messages=delivered,
        error=error,
    )


@pytest.fixture(autouse=True)
def _isolate_terminal_verdict(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner,
        "RADAR_VERDICT_PATH",
        tmp_path / "radar-terminal.json",
    )


def _result(*, coin: str = "KRW-TEST") -> dict:
    return {
        "asof": "2026-07-26",
        "model_id": "pump_hunter_v2",
        "rule_version": "pump_detector_v2",
        "rule": runner.PUMP_V2_RULE,
        "feature_date": "2026-07-25 09:00:00",
        "btc_regime": "bear_quiet",
        "universe_n": 100,
        "binance_status": "ok",
        "n_candidates": 1,
        "candidates": [
            {
                "market": coin,
                "rank": 1,
                "score": 0.9,
                "entry_open": 100.0,
                "roc_7d": 12.0,
                "roc_7d_rank": 0.95,
                "atr_pct_14": 0.08,
                "log_return_1d": 0.01,
                "b_vol_surge": 3.2,
                "b_ret_1d": 0.05,
                "liq_rank_daily": 10,
                "btc_regime": "bear_quiet",
                "rule_id": "roc7_rank+bn_volsurge",
            }
        ],
        "oos": {
            "hit_pct": 8.1,
            "baseline_hit_pct": 5.6,
            "base_rate_pct": 1.4,
            "net_tp5sl3_pct": -0.36,
        },
    }


def _run_main(
    monkeypatch,
    ledger: Path,
    receipts: Path,
    result: dict,
    *,
    send: bool = True,
    dry_run: bool = False,
) -> int:
    decisions = receipts.parent / "decisions"
    monkeypatch.setattr(runner, "PUMP_V2_LEDGER", str(ledger))
    monkeypatch.setattr(runner, "PUMP_V2_RECEIPT_ROOT", str(receipts))
    monkeypatch.setattr(runner, "PUMP_V2_DECISION_ROOT", str(decisions))
    args = [
        "pump_detector_v2_today.py",
        "--asof",
        result["asof"],
        "--ledger",
        str(ledger),
        "--receipt-root",
        str(receipts),
        "--decision-root",
        str(decisions),
    ]
    if send:
        args.append("--send-telegram")
    if dry_run:
        args.append("--dry-run")
    monkeypatch.setattr(sys, "argv", args)
    decision_day = datetime.fromisoformat(result["asof"])
    monkeypatch.setattr(
        runner,
        "_now_kst",
        lambda: decision_day.replace(
            hour=9,
            minute=15,
            tzinfo=runner.KST,
        ),
    )
    monkeypatch.setattr(runner, "score_pump_v2_candidates", lambda *_a, **_k: result)
    return runner.main()


def _append_in_child(ledger: str, result: dict, start) -> None:
    # spawn 자식은 이 모듈의 kill-switch-해제 fixture 환경을 물려받는다 —
    # 어떤 코드 경로도 실 텔레그램에 닿지 못하게 자식에서 재봉쇄한다.
    os.environ["PRELUDE_FORBID_TELEGRAM"] = "1"
    start.wait()
    runner.append_ledger(
        result,
        ledger,
        False,
        delivery_ok=False,
        sent_at=None,
    )


def test_failed_delivery_is_recorded_but_never_open(monkeypatch, tmp_path):
    ledger = tmp_path / "v2.csv"
    receipts = tmp_path / "receipts"
    monkeypatch.setattr(runner, "send_telegram", lambda _message: False)

    assert _run_main(monkeypatch, ledger, receipts, _result()) == 1

    row = pd.read_csv(ledger).iloc[0]
    assert row["status"] == "not_delivered"
    assert bool(row["delivery_ok"]) is False
    assert pd.isna(row["sent_at"])
    receipt = json.loads((receipts / "2026-07-26.json").read_text())
    assert receipt["delivery_ok"] is False
    assert receipt["sent_at"] is None
    assert receipt["decision"]["candidates"][0]["market"] == "KRW-TEST"


def test_failed_send_can_retry_then_promote_without_duplicate(monkeypatch, tmp_path):
    ledger = tmp_path / "v2.csv"
    receipts = tmp_path / "receipts"
    calls: list[str] = []
    outcomes = iter([False, True])

    def fake_send(message: str) -> bool:
        calls.append(message)
        return next(outcomes)

    monkeypatch.setattr(runner, "send_telegram", fake_send)
    result = _result()

    assert _run_main(monkeypatch, ledger, receipts, result) == 1
    assert _run_main(monkeypatch, ledger, receipts, result) == 0
    assert _run_main(monkeypatch, ledger, receipts, result) == 0

    rows = pd.read_csv(ledger)
    assert len(rows) == 1
    assert rows.loc[0, "status"] == "open"
    assert bool(rows.loc[0, "delivery_ok"]) is True
    assert pd.notna(rows.loc[0, "sent_at"])
    assert len(calls) == 2


def test_dry_run_dominates_send_flag_and_has_no_side_effect(monkeypatch, tmp_path):
    ledger = tmp_path / "v2.csv"
    receipts = tmp_path / "receipts"
    calls: list[str] = []
    monkeypatch.setattr(runner, "send_telegram", lambda message: calls.append(message))

    assert _run_main(
        monkeypatch,
        ledger,
        receipts,
        _result(),
        dry_run=True,
    ) == 0

    assert calls == []
    assert not ledger.exists()
    assert not receipts.exists()
    assert not (receipts.parent / "decisions").exists()


def test_successful_day_rejects_changed_decision_without_second_send(
    monkeypatch,
    tmp_path,
):
    ledger = tmp_path / "v2.csv"
    receipts = tmp_path / "receipts"
    calls: list[str] = []
    monkeypatch.setattr(
        runner,
        "send_telegram",
        lambda message: calls.append(message) or True,
    )

    assert _run_main(monkeypatch, ledger, receipts, _result()) == 0
    assert _run_main(
        monkeypatch,
        ledger,
        receipts,
        _result(coin="KRW-OTHER"),
    ) == 1

    assert len(calls) == 1
    assert pd.read_csv(ledger)["coin"].tolist() == ["KRW-TEST"]


def test_failed_attempt_rejects_changed_decision_without_second_send(
    monkeypatch,
    tmp_path,
):
    receipts = tmp_path / "receipts"
    calls: list[str] = []
    monkeypatch.setattr(
        runner,
        "send_telegram",
        lambda message: calls.append(message) or False,
    )

    first = _result()
    first_receipt = runner.deliver_once(
        first,
        "first",
        receipt_root=receipts,
    )
    assert first_receipt["delivery_ok"] is False

    with pytest.raises(RuntimeError, match="different v2 decision"):
        runner.deliver_once(
            _result(coin="KRW-OTHER"),
            "changed",
            receipt_root=receipts,
        )

    assert calls == ["first"]


@pytest.mark.parametrize("corruption", ["duplicate_key", "nan"])
def test_existing_v2_receipt_uses_strict_json(
    monkeypatch,
    tmp_path,
    corruption,
):
    receipts = tmp_path / "receipts"
    monkeypatch.setattr(runner, "send_telegram", lambda _message: False)
    result = _result()
    runner.deliver_once(result, "first", receipt_root=receipts)
    path = receipts / f"{result['asof']}.json"
    raw = path.read_text(encoding="utf-8")
    if corruption == "duplicate_key":
        raw = raw.replace(
            '"delivery_ok": false',
            '"delivery_ok": false, "delivery_ok": false',
            1,
        )
    else:
        raw = raw.replace(
            '"delivery_ok": false',
            '"delivery_ok": NaN',
            1,
        )
    path.write_text(raw, encoding="utf-8")

    with pytest.raises(RuntimeError, match="receipt read failed"):
        runner.deliver_once(result, "retry", receipt_root=receipts)


def test_manual_record_only_run_is_explicitly_not_delivered(monkeypatch, tmp_path):
    ledger = tmp_path / "v2.csv"

    assert _run_main(
        monkeypatch,
        ledger,
        tmp_path / "receipts",
        _result(),
        send=False,
    ) == 0

    row = pd.read_csv(ledger).iloc[0]
    assert row["status"] == "not_delivered"
    assert bool(row["delivery_ok"]) is False


def test_atomic_replace_failure_preserves_existing_ledger(monkeypatch, tmp_path):
    ledger = tmp_path / "v2.csv"
    ledger.write_text("date,coin,status\n2026-07-25,KRW-OLD,closed\n")
    before = ledger.read_bytes()
    monkeypatch.setattr(
        "ledger.csv_store.os.replace",
        lambda *_a, **_kw: (_ for _ in ()).throw(OSError("boom")),
    )

    try:
        runner.append_ledger(
            _result(),
            str(ledger),
            False,
            delivery_ok=False,
            sent_at=None,
        )
    except OSError:
        pass
    else:
        raise AssertionError("replace failure must propagate")

    assert ledger.read_bytes() == before


def test_same_day_ledger_content_must_match_before_idempotent_retry(tmp_path):
    ledger = tmp_path / "v2.csv"
    result = _result()
    runner.append_ledger(
        result,
        str(ledger),
        False,
        delivery_ok=False,
        sent_at=None,
        receipt_path="decision.json",
    )
    rows = pd.read_csv(ledger)
    rows.loc[0, "score"] = 0.1
    rows.to_csv(ledger, index=False)
    before = ledger.read_bytes()

    with pytest.raises(RuntimeError, match="immutable decision row conflict"):
        runner.append_ledger(
            result,
            str(ledger),
            False,
            delivery_ok=False,
            sent_at=None,
            receipt_path="decision.json",
        )

    assert ledger.read_bytes() == before


def test_same_day_ledger_rejects_snapshot_identity_or_path_change(tmp_path):
    ledger = tmp_path / "v2.csv"
    result = _result()
    runner.append_ledger(
        result,
        str(ledger),
        False,
        delivery_ok=False,
        sent_at=None,
        receipt_path="decision.json",
    )

    rows = pd.read_csv(ledger)
    rows.loc[0, "snapshot_id"] = "wrong-id"
    rows.to_csv(ledger, index=False)
    before = ledger.read_bytes()
    with pytest.raises(RuntimeError, match="snapshot identity conflict"):
        runner.append_ledger(
            result,
            str(ledger),
            False,
            delivery_ok=False,
            sent_at=None,
            receipt_path="decision.json",
        )
    assert ledger.read_bytes() == before

    rows.loc[0, "snapshot_id"] = runner._decision_id(result)
    rows.loc[0, "snapshot_path"] = "other-decision.json"
    rows.to_csv(ledger, index=False)
    before = ledger.read_bytes()
    with pytest.raises(RuntimeError, match="snapshot path conflict"):
        runner.append_ledger(
            result,
            str(ledger),
            False,
            delivery_ok=False,
            sent_at=None,
            receipt_path="decision.json",
        )
    assert ledger.read_bytes() == before


def test_same_day_ledger_rejects_incoherent_delivery_state(tmp_path):
    ledger = tmp_path / "v2.csv"
    result = _result()
    runner.append_ledger(
        result,
        str(ledger),
        False,
        delivery_ok=False,
        sent_at=None,
    )
    rows = pd.read_csv(ledger)
    rows.loc[0, "status"] = "open"
    rows.to_csv(ledger, index=False)
    before = ledger.read_bytes()

    with pytest.raises(RuntimeError, match="delivered status metadata conflict"):
        runner.append_ledger(
            result,
            str(ledger),
            False,
            delivery_ok=False,
            sent_at=None,
        )

    assert ledger.read_bytes() == before


def test_same_day_ledger_rejects_changed_success_timestamp(tmp_path):
    ledger = tmp_path / "v2.csv"
    result = _result()
    first_sent_at = "2026-07-26T00:05:00+00:00"
    runner.append_ledger(
        result,
        str(ledger),
        False,
        delivery_ok=True,
        sent_at=first_sent_at,
    )
    before = ledger.read_bytes()

    with pytest.raises(RuntimeError, match="sent_at conflict"):
        runner.append_ledger(
            result,
            str(ledger),
            False,
            delivery_ok=True,
            sent_at="2026-07-26T00:06:00+00:00",
        )

    assert ledger.read_bytes() == before


@pytest.mark.parametrize("sent_at", ["2026-07-26T00:05:00", "not-a-time"])
def test_ledger_rejects_invalid_success_timestamp(tmp_path, sent_at):
    ledger = tmp_path / "v2.csv"

    with pytest.raises(RuntimeError, match="sent_at"):
        runner.append_ledger(
            _result(),
            str(ledger),
            False,
            delivery_ok=True,
            sent_at=sent_at,
        )

    assert not ledger.exists()


def test_concurrent_same_decision_append_is_idempotent(tmp_path):
    context = multiprocessing.get_context("spawn")
    ledger = tmp_path / "v2.csv"
    worker_count = 4
    start = context.Barrier(worker_count + 1)
    processes = [
        context.Process(
            target=_append_in_child,
            args=(str(ledger), _result(), start),
        )
        for _ in range(worker_count)
    ]
    try:
        for process in processes:
            process.start()
        start.wait(timeout=10)
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)

    rows = pd.read_csv(ledger)
    assert len(rows) == 1
    assert rows["coin"].tolist() == ["KRW-TEST"]


def test_zero_candidate_run_persists_observed_decision_day(monkeypatch, tmp_path):
    result = _result()
    result["n_candidates"] = 0
    result["candidates"] = []
    ledger = tmp_path / "v2.csv"
    receipts = tmp_path / "receipts"
    # 무소음 정책이 회귀하면 실 전송 경로로 떨어진다 — 이 모듈은
    # kill-switch 를 지우므로 반드시 recorder 로 봉쇄하고 0건을 단언한다.
    forbidden_sends: list = []
    monkeypatch.setattr(
        runner,
        "send_telegram",
        lambda *a, **k: forbidden_sends.append("send") or True,
    )
    monkeypatch.setattr(
        runner,
        "send_telegram_with_receipt",
        lambda *a, **k: forbidden_sends.append("receipt") or None,
    )

    assert _run_main(
        monkeypatch,
        ledger,
        receipts,
        result,
        send=True,
    ) == 0

    assert forbidden_sends == []

    manifest = tmp_path / "decisions" / "2026-07-26.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["schema"] == runner.PUMP_V2_DECISION_SCHEMA
    assert payload["decision"]["n_candidates"] == 0
    assert (
        payload["decision"]["execution_provenance"]["evidence_class"]
        == "canonical_forward"
    )
    assert not receipts.exists()
    assert not ledger.exists()


def test_live_run_rejects_source_change_during_scoring(
    monkeypatch,
    tmp_path,
):
    result = _result()
    result["asof"] = "2026-07-27"
    result["feature_date"] = "2026-07-26 09:00:00"

    def inputs(digest_character: str) -> dict:
        def identity(path: str) -> dict:
            return {
                "path": path,
                "bytes": 1,
                "sha256": digest_character * 64,
            }

        return {
            "sources": {
                path: identity(path)
                for path in runner._SOURCE_PATHS
            },
            "data": {
                "data/upbit_d1.db": identity("data/upbit_d1.db"),
                "data/binance_d1.db": identity("data/binance_d1.db"),
            },
        }

    observed = iter([inputs("a"), inputs("b")])
    monkeypatch.setattr(
        runner,
        "_current_forward_inputs",
        lambda: next(observed),
    )
    ledger = tmp_path / "v2.csv"

    assert _run_main(
        monkeypatch,
        ledger,
        tmp_path / "receipts",
        result,
        send=False,
    ) == 1
    assert not ledger.exists()
    assert not (tmp_path / "decisions" / "2026-07-27.json").exists()


def test_decision_manifest_rejects_same_day_changed_payload(tmp_path):
    first = _result()
    path = runner.persist_decision(first, decision_root=tmp_path)
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["schema"] == runner.PUMP_V2_LEGACY_DECISION_SCHEMA
    runner._validate_decision_document(persisted, first, path)

    with pytest.raises(RuntimeError, match="different v2 decision"):
        runner.persist_decision(
            _result(coin="KRW-OTHER"),
            decision_root=tmp_path,
        )

    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["decision"]["candidates"][0]["market"] == "KRW-TEST"


@pytest.mark.parametrize("corruption", ["duplicate_key", "nan"])
def test_existing_v2_decision_manifest_uses_strict_json(
    tmp_path,
    corruption,
):
    result = _result()
    path = runner.persist_decision(result, decision_root=tmp_path)
    raw = path.read_text(encoding="utf-8")
    if corruption == "duplicate_key":
        raw = raw.replace(
            f'"schema": "{runner.PUMP_V2_LEGACY_DECISION_SCHEMA}"',
            (
                f'"schema": "{runner.PUMP_V2_LEGACY_DECISION_SCHEMA}", '
                f'"schema": "{runner.PUMP_V2_LEGACY_DECISION_SCHEMA}"'
            ),
            1,
        )
    else:
        raw = raw.replace('"universe_n": 100', '"universe_n": NaN', 1)
    path.write_text(raw, encoding="utf-8")

    with pytest.raises(RuntimeError, match="decision read failed"):
        runner.persist_decision(result, decision_root=tmp_path)


def test_dangling_receipt_symlink_blocks_send_before_transport(
    tmp_path,
    monkeypatch,
):
    receipt_root = tmp_path / "receipts"
    receipt_root.mkdir()
    receipt_path = receipt_root / "2026-07-26.json"
    receipt_path.symlink_to(tmp_path / "missing-receipt.json")
    calls: list[str] = []
    monkeypatch.setattr(
        runner,
        "send_telegram",
        lambda message: calls.append(message) or True,
    )

    with pytest.raises(RuntimeError, match="receipt read failed"):
        runner.deliver_once(
            _result(),
            "radar",
            receipt_root=receipt_root,
        )

    assert calls == []
    assert receipt_path.is_symlink()


def test_forward_v2_manifest_and_receipt_seal_outer_chronology(
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
                tzinfo=timezone.utc,
            )
            return value if tz is None else value.astimezone(tz)

    result = _result()
    result["asof"] = "2026-07-27"
    result["feature_date"] = "2026-07-26 09:00:00"
    result = runner._with_forward_provenance(result)
    monkeypatch.setattr(runner, "datetime", FixedDatetime)
    monkeypatch.setattr(
        runner,
        "send_telegram_with_receipt",
        lambda _message, **_kwargs: _telegram_result("radar"),
    )

    decision_path = runner.persist_decision(
        result,
        decision_root=tmp_path / "decisions",
    )
    decision_payload = json.loads(
        decision_path.read_text(encoding="utf-8")
    )
    receipt_payload = runner.deliver_once(
        result,
        "radar",
        receipt_root=tmp_path / "receipts",
    )

    for payload in (decision_payload, receipt_payload):
        assert len(payload["integrity_sha256"]) == 64
        assert runner.manifest_digest_matches(
            payload,
            digest_key="integrity_sha256",
        )

    unsigned_decision = dict(decision_payload)
    unsigned_decision.pop("integrity_sha256")
    with pytest.raises(RuntimeError, match="outer schema mismatch"):
        runner._validate_decision_document(
            unsigned_decision,
            result,
            decision_path,
        )

    receipt_payload["sent_at"] = "2026-07-27T00:06:00+00:00"
    with pytest.raises(RuntimeError, match="outer integrity mismatch"):
        runner._validate_receipt(
            receipt_payload,
            result,
            tmp_path / "receipts/2026-07-27.json",
        )


def test_forward_partial_delivery_is_preserved_and_not_retried(
    tmp_path,
    monkeypatch,
):
    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            value = datetime.fromisoformat(
                "2026-07-27T00:05:00.500000+00:00"
            )
            return value if tz is None else value.astimezone(tz)

    result = _result()
    result["asof"] = "2026-07-27"
    result["feature_date"] = "2026-07-26 09:00:00"
    result = runner._with_forward_provenance(result)
    monkeypatch.setattr(runner, "datetime", FixedDatetime)
    calls = 0

    def fake_detailed(message, **_kwargs):
        nonlocal calls
        calls += 1
        return _telegram_result(
            message,
            delivery_ok=False,
            server_dates=("2026-07-27T00:05:00+00:00",),
            error="second chunk rejected",
        )

    monkeypatch.setattr(
        runner,
        "send_telegram_with_receipt",
        fake_detailed,
    )
    receipt_root = tmp_path / "receipts"
    message = "x" * 4001

    receipt = runner.deliver_once(
        result,
        message,
        receipt_root=receipt_root,
    )

    assert receipt["delivery_ok"] is False
    assert receipt["sent_at"] is None
    assert len(receipt["telegram_messages"]) == 1
    assert runner.manifest_digest_matches(
        receipt,
        digest_key="integrity_sha256",
    )
    with pytest.raises(RuntimeError, match="partial/ambiguous Telegram"):
        runner.deliver_once(
            result,
            message,
            receipt_root=receipt_root,
        )
    assert calls == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("n_candidates", 2),
        ("universe_n", 0),
    ],
)
def test_invalid_decision_is_rejected_before_persistence(
    tmp_path, field, value
):
    result = _result()
    result[field] = value

    with pytest.raises(ValueError):
        runner.persist_decision(result, decision_root=tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_healthy_zero_pick_rejects_empty_universe(tmp_path):
    result = _result()
    result["n_candidates"] = 0
    result["candidates"] = []
    result["universe_n"] = 0

    with pytest.raises(ValueError, match="universe_n must be positive"):
        runner.persist_decision(result, decision_root=tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rank", True),
        ("market", "KRW-bad"),
        ("score", math.nan),
        ("entry_open", 0.0),
        ("roc_7d_rank", 1.1),
        ("liq_rank_daily", 0),
        ("btc_regime", "unknown"),
        ("rule_id", ""),
    ],
)
def test_invalid_candidate_contract_is_rejected_before_persistence(
    tmp_path, field, value
):
    result = _result()
    result["candidates"][0][field] = value

    with pytest.raises(ValueError):
        runner.persist_decision(result, decision_root=tmp_path)

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("location", "field", "value", "message"),
    [
        ("decision", "rule", "changed-rule", "invalid v2 rule"),
        ("candidate", "rule_id", "changed-rule", "rule_id is invalid"),
        (
            "candidate",
            "btc_regime",
            "bull_quiet",
            "candidate/decision btc_regime mismatch",
        ),
        ("oos", "hit_pct", 99.0, "provenance mismatch"),
    ],
)
def test_decision_rejects_changed_rule_or_provenance(
    tmp_path,
    location,
    field,
    value,
    message,
):
    result = _result()
    target = (
        result
        if location == "decision"
        else result["candidates"][0]
        if location == "candidate"
        else result["oos"]
    )
    target[field] = value

    with pytest.raises(ValueError, match=message):
        runner.persist_decision(result, decision_root=tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_decision_requires_prior_completed_kst_feature_day(tmp_path):
    result = _result()
    result["feature_date"] = "2026-07-26 09:00:00"

    with pytest.raises(ValueError, match="prior KST D1"):
        runner.persist_decision(result, decision_root=tmp_path)


def test_stale_zero_pick_manifest_preserves_failure_evidence(tmp_path):
    result = _result()
    result["binance_status"] = "binance_partial"
    result["n_candidates"] = 0
    result["candidates"] = []
    result.pop("feature_date")
    result.pop("btc_regime")

    path = runner.persist_decision(result, decision_root=tmp_path)

    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["decision"]["binance_status"] == "binance_partial"
    assert persisted["decision"]["n_candidates"] == 0


def test_main_rejects_scorer_asof_mismatch_before_any_side_effect(
    monkeypatch, tmp_path
):
    result = _result()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pump_detector_v2_today.py",
            "--asof",
            "2026-07-25",
            "--ledger",
            str(tmp_path / "ledger.csv"),
            "--decision-root",
            str(tmp_path / "decisions"),
            "--receipt-root",
            str(tmp_path / "receipts"),
            "--send-telegram",
        ],
    )
    monkeypatch.setattr(
        runner,
        "score_pump_v2_candidates",
        lambda *_args, **_kwargs: result,
    )
    monkeypatch.setattr(
        runner,
        "_now_kst",
        lambda: datetime(
            2026,
            7,
            25,
            9,
            15,
            tzinfo=runner.KST,
        ),
    )
    sends: list[str] = []
    monkeypatch.setattr(
        runner,
        "send_telegram",
        lambda message: sends.append(message) or True,
    )

    assert runner.main() == 1
    assert sends == []
    assert [
        path
        for path in tmp_path.iterdir()
        if not path.name.endswith(".lock")
    ] == []


def test_live_run_rechecks_window_after_scoring_before_telegram(
    monkeypatch,
    tmp_path,
):
    result = _result()
    ledger = tmp_path / "v2.csv"
    receipts = tmp_path / "receipts"
    decisions = tmp_path / "decisions"
    monkeypatch.setattr(runner, "PUMP_V2_LEDGER", str(ledger))
    monkeypatch.setattr(runner, "PUMP_V2_RECEIPT_ROOT", str(receipts))
    monkeypatch.setattr(runner, "PUMP_V2_DECISION_ROOT", str(decisions))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pump_detector_v2_today.py",
            "--asof",
            result["asof"],
            "--ledger",
            str(ledger),
            "--receipt-root",
            str(receipts),
            "--decision-root",
            str(decisions),
            "--send-telegram",
        ],
    )
    clock = iter(
        [
            datetime(
                2026,
                7,
                26,
                9,
                30,
                59,
                tzinfo=runner.KST,
            ),
            datetime(
                2026,
                7,
                26,
                9,
                31,
                tzinfo=runner.KST,
            ),
        ]
    )
    calls: list[str] = []
    monkeypatch.setattr(runner, "_now_kst", lambda: next(clock))
    monkeypatch.setattr(
        runner,
        "score_pump_v2_candidates",
        lambda *_args, **_kwargs: result,
    )
    monkeypatch.setattr(
        runner,
        "send_telegram",
        lambda message: calls.append(message) or True,
    )

    assert runner.main() == 1
    assert calls == []
    assert not list(receipts.glob("*.json"))
    assert not ledger.exists()
    assert not (decisions / f"{result['asof']}.json").exists()


@pytest.mark.parametrize(
    "now",
    [
        datetime(2026, 7, 27, 9, 15, tzinfo=runner.KST),
        datetime(2026, 7, 26, 14, 0, tzinfo=runner.KST),
    ],
)
def test_main_rejects_stale_or_late_live_run_before_scoring(
    monkeypatch, tmp_path, now
):
    calls: list[str] = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pump_detector_v2_today.py",
            "--asof",
            "2026-07-26",
            "--ledger",
            str(tmp_path / "ledger.csv"),
            "--decision-root",
            str(tmp_path / "decisions"),
            "--send-telegram",
        ],
    )
    monkeypatch.setattr(runner, "_now_kst", lambda: now)
    monkeypatch.setattr(
        runner,
        "score_pump_v2_candidates",
        lambda *_args, **_kwargs: calls.append("score") or _result(),
    )
    monkeypatch.setattr(
        runner,
        "send_telegram",
        lambda *_args, **_kwargs: calls.append("send") or True,
    )

    assert runner.main() == 1
    assert calls == []
    assert list(tmp_path.iterdir()) == []


def test_main_rejects_custom_live_outputs_before_scoring(
    monkeypatch,
    tmp_path,
):
    calls: list[str] = []
    monkeypatch.setattr(
        runner,
        "send_telegram",
        lambda *_args, **_kwargs: calls.append("send"),
    )
    monkeypatch.setattr(
        runner,
        "send_telegram_with_receipt",
        lambda *_args, **_kwargs: calls.append("send-with-receipt"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pump_detector_v2_today.py",
            "--asof",
            "2026-07-26",
            "--ledger",
            str(tmp_path / "custom.csv"),
            "--send-telegram",
        ],
    )
    monkeypatch.setattr(
        runner,
        "_now_kst",
        lambda: datetime(2026, 7, 26, 9, 15, tzinfo=runner.KST),
    )
    monkeypatch.setattr(
        runner,
        "score_pump_v2_candidates",
        lambda *_args, **_kwargs: calls.append("score") or _result(),
    )

    assert runner.main() == 1
    assert calls == []
    assert list(tmp_path.iterdir()) == []


def test_main_rejects_custom_live_candidate_limit_before_scoring(
    monkeypatch,
):
    calls: list[str] = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pump_detector_v2_today.py",
            "--asof",
            "2026-07-26",
            "--max-candidates",
            str(runner.MAX_CANDIDATES + 1),
        ],
    )
    monkeypatch.setattr(
        runner,
        "_now_kst",
        lambda: datetime(2026, 7, 26, 9, 15, tzinfo=runner.KST),
    )
    monkeypatch.setattr(
        runner,
        "score_pump_v2_candidates",
        lambda *_args, **_kwargs: calls.append("score") or _result(),
    )

    assert runner.main() == 1
    assert calls == []


def test_post_activation_legacy_decision_cannot_be_persisted(tmp_path):
    result = _result()
    result["asof"] = "2026-07-27"
    result["feature_date"] = "2026-07-26 09:00:00"

    with pytest.raises(RuntimeError, match="legacy decision.*not forward-valid"):
        runner.persist_decision(result, decision_root=tmp_path)

    assert [
        path
        for path in tmp_path.iterdir()
        if not path.name.endswith(".lock")
    ] == []

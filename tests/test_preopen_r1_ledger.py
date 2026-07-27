from __future__ import annotations

import json
import hashlib
import threading
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

import notifier.delivery_receipt as delivery_receipt
import scripts.recommend_today as recommend_today
from notifier.delivery_receipt import (
    DeliveryReceiptError,
    read_delivery_receipt,
    receipt_path,
    write_delivery_receipt,
)
from notifier.telegram import TelegramSendResult, TelegramServerMessage
from signals.model_registry import get_model


def _preopen_snapshot(asof: str = "2026-07-25") -> dict:
    candidate = {
        "coin": "KRW-TEST",
        "rank": 1,
        "score": 0.9,
        "pump_prob": 0.05,
        "pump_prob_pct": "5.0%",
        "dump_risk_flag": False,
        "btc_regime": "neutral",
        "entry_open": None,
        "sl": -0.03,
        "tp": 0.05,
        "p_up5": 0.3,
        "p_up10": 0.1,
        "p_up20": 0.05,
        "p_dn5": 0.08,
        "p_dn10": 0.02,
        "exp_downside": -0.02,
        "rr_ratio": 1.25,
    }
    return {
        "asof": asof,
        "slot": "preopen",
        "btc_regime": "neutral",
        "universe_n": 1,
        "calibration_source": "bucket_score_pump20",
        "n_history_dates": 100,
        "snapshot_id": f"recommend-test-{asof}",
        "snapshot_path": f"output/recommend_snapshots/{asof}/preopen_r1.json",
        "decision_completed_at": f"{asof}T08:50:01+09:00",
        "top3": [candidate],
    }


def _telegram_result(message: str, server_date: str) -> TelegramSendResult:
    return TelegramSendResult(
        delivery_ok=True,
        message_sha256=hashlib.sha256(message.encode()).hexdigest(),
        chunk_count=1,
        chat_id_sha256=hashlib.sha256(b"456").hexdigest(),
        telegram_messages=(
            TelegramServerMessage(
                message_id=101,
                server_date=server_date,
                text_sha256=hashlib.sha256(message.encode()).hexdigest(),
            ),
        ),
        error=None,
    )
def test_r1_default_ledger_is_split_by_slot():
    assert (
        recommend_today.default_ledger_path("R1", "open")
        == recommend_today.SHADOW_RECOMMEND_LEDGER
    )
    assert (
        recommend_today.default_ledger_path("R1", "preopen")
        == recommend_today.SHADOW_RECOMMEND_LEDGER_PREOPEN
    )
    assert (
        recommend_today.SHADOW_RECOMMEND_LEDGER_PREOPEN
        != recommend_today.SHADOW_RECOMMEND_LEDGER
    )


def test_model_registry_uses_the_preopen_writer_path():
    spec = get_model("recommend_r1_preopen")

    assert spec.ledger_path == recommend_today.SHADOW_RECOMMEND_LEDGER_PREOPEN
    assert spec.slots == ["preopen"]


@pytest.mark.parametrize(
    ("asof", "limit_markets", "snapshot_root", "message"),
    [
        ("2026-07-24", None, None, "historical/future"),
        ("2026-07-25", 10, None, "market-limited"),
        ("2026-07-25", None, "custom-snapshots", "custom evidence roots"),
    ],
)
def test_canonical_recommend_ledger_rejects_replay_or_development_inputs(
    monkeypatch,
    asof,
    limit_markets,
    snapshot_root,
    message,
):
    monkeypatch.setattr(recommend_today, "_today_kst", lambda: "2026-07-25")
    monkeypatch.setattr(
        recommend_today,
        "get_or_create_recommend_snapshot",
        lambda *_args, **_kwargs: pytest.fail(
            "canonical request must fail before scoring"
        ),
    )

    with pytest.raises(RuntimeError, match=message):
        recommend_today.append_today(
            asof,
            ledger_path=recommend_today.SHADOW_RECOMMEND_LEDGER,
            limit_markets=limit_markets,
            snapshot_root=snapshot_root,
            require_receipt=True,
        )


def test_canonical_r1_ledger_requires_delivery_receipt_contract(monkeypatch):
    monkeypatch.setattr(recommend_today, "_today_kst", lambda: "2026-07-25")
    monkeypatch.setattr(
        recommend_today,
        "get_or_create_recommend_snapshot",
        lambda *_args, **_kwargs: pytest.fail(
            "canonical request must fail before scoring"
        ),
    )

    with pytest.raises(RuntimeError, match="requires a successful delivery"):
        recommend_today.append_today(
            "2026-07-25",
            ledger_path=recommend_today.SHADOW_RECOMMEND_LEDGER,
        )


def test_canonical_ledger_cannot_be_reused_by_another_ranking(monkeypatch):
    monkeypatch.setattr(recommend_today, "_today_kst", lambda: "2026-07-25")

    with pytest.raises(RuntimeError, match="ledger/ranking/slot mismatch"):
        recommend_today.append_today(
            "2026-07-25",
            ranking="R2",
            ledger_path=recommend_today.SHADOW_RECOMMEND_LEDGER,
        )


def test_canonical_forward_snapshot_requires_current_schema_and_slot_window():
    result = {
        "snapshot_schema": recommend_today.SNAPSHOT_SCHEMA_VERSION,
        "request": {
            "asof": "2026-07-25",
            "slot": "open",
            "ranking": "R2",
            "limit_markets": None,
        },
        "model": {
            "id": "recommend_r2_open",
            "ranking": "R2",
        },
        "score_schema_version": "recommend_score.v2",
        "rule_version": "r2_downside_penalized_v1",
        "rank_basis": "R2_penalized(λ=1.0, de-corr head)",
        "snapshot_path": str(
            recommend_today.snapshot_path("2026-07-25", "open", "R2", None)
        ),
        "decision_started_at": "2026-07-25T00:05:00+00:00",
        "decision_completed_at": "2026-07-25T00:05:05+00:00",
    }
    recommend_today._assert_canonical_forward_snapshot(
        result,
        asof="2026-07-25",
        ranking="R2",
        slot="open",
    )

    result["snapshot_schema"] = "recommend_snapshot.v1"
    with pytest.raises(RuntimeError, match="legacy recommend snapshot"):
        recommend_today._assert_canonical_forward_snapshot(
            result,
            asof="2026-07-25",
            ranking="R2",
            slot="open",
        )

    result["snapshot_schema"] = recommend_today.SNAPSHOT_SCHEMA_VERSION
    result["decision_completed_at"] = "2026-07-25T00:21:00+00:00"
    with pytest.raises(RuntimeError, match="outside open decision window"):
        recommend_today._assert_canonical_forward_snapshot(
            result,
            asof="2026-07-25",
            ranking="R2",
            slot="open",
        )

    result["decision_completed_at"] = "2026-07-25T00:05:05+00:00"
    result["rank_basis"] = "score_rank(fallback)"
    with pytest.raises(RuntimeError, match="degraded/fallback ranking"):
        recommend_today._assert_canonical_forward_snapshot(
            result,
            asof="2026-07-25",
            ranking="R2",
            slot="open",
        )


def test_append_preopen_r1_writes_only_dedicated_ledger(
    tmp_path, monkeypatch
):
    open_ledger = tmp_path / "open.csv"
    preopen_ledger = tmp_path / "preopen.csv"
    monkeypatch.setattr(
        recommend_today,
        "SHADOW_RECOMMEND_LEDGER",
        str(open_ledger),
    )
    monkeypatch.setattr(
        recommend_today,
        "SHADOW_RECOMMEND_LEDGER_PREOPEN",
        str(preopen_ledger),
    )
    monkeypatch.setattr(
        recommend_today,
        "get_or_create_recommend_snapshot",
        lambda *args, **kwargs: _preopen_snapshot(),
    )

    receipt_root = tmp_path / "receipts"
    recommend_today.append_today(
        "2026-07-25", slot="preopen", receipt_root=receipt_root
    )
    recommend_today.append_today(
        "2026-07-25", slot="preopen", receipt_root=receipt_root
    )

    assert preopen_ledger.exists()
    assert not open_ledger.exists()
    rows = pd.read_csv(preopen_ledger)
    assert len(rows) == 1
    assert rows.loc[0, "date"] == "2026-07-25"
    assert rows.loc[0, "coin"] == "KRW-TEST"
    assert rows.loc[0, "status"] == "open"
    assert rows.loc[0, "snapshot_id"] == "recommend-test-2026-07-25"
    assert rows.loc[0, "decision_completed_at"] == (
        "2026-07-25T08:50:01+09:00"
    )
    assert pd.isna(rows.loc[0, "delivery_ok"])
    assert pd.isna(rows.loc[0, "sent_at"])


@pytest.mark.parametrize("slot", ["preopen", "open"])
def test_r1_ledger_records_validated_delivery_receipt(
    tmp_path, monkeypatch, slot
):
    snapshot = _preopen_snapshot()
    snapshot["slot"] = slot
    snapshot["snapshot_path"] = f"output/recommend_snapshots/2026-07-25/{slot}_r1.json"
    ledger = tmp_path / f"{slot}.csv"
    receipt_root = tmp_path / "receipts"
    sent_at = "2026-07-25T00:00:05+00:00"
    write_delivery_receipt(
        snapshot,
        delivery_ok=True,
        attempted_at="2026-07-25T00:00:04+00:00",
        sent_at=sent_at,
        root=receipt_root,
    )
    monkeypatch.setattr(
        recommend_today,
        "get_or_create_recommend_snapshot",
        lambda *args, **kwargs: snapshot,
    )

    recommend_today.append_today(
        "2026-07-25",
        slot=slot,
        ledger_path=str(ledger),
        receipt_root=receipt_root,
    )

    row = pd.read_csv(ledger).iloc[0]
    assert row["snapshot_id"] == snapshot["snapshot_id"]
    assert row["snapshot_path"] == snapshot["snapshot_path"]
    assert bool(row["delivery_ok"]) is True
    assert row["sent_at"] == sent_at


def test_delivery_receipt_reader_rejects_wrong_snapshot_identity(tmp_path):
    snapshot = _preopen_snapshot()
    receipt_root = tmp_path / "receipts"
    path = write_delivery_receipt(
        snapshot,
        delivery_ok=False,
        attempted_at="2026-07-25T00:00:04+00:00",
        sent_at=None,
        root=receipt_root,
    )
    text = path.read_text().replace(
        "recommend-test-2026-07-25",
        "recommend-other-2026-07-25",
    )
    path.write_text(text)

    with pytest.raises(DeliveryReceiptError, match="snapshot_id"):
        read_delivery_receipt(snapshot, root=receipt_root)


def test_post_activation_receipt_seals_full_outer_chronology(
    tmp_path,
    monkeypatch,
):
    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            value = datetime(
                2026,
                7,
                26,
                23,
                50,
                6,
                tzinfo=timezone.utc,
            )
            return value if tz is None else value.astimezone(tz)

    monkeypatch.setattr(delivery_receipt, "datetime", FixedDatetime)
    snapshot = _preopen_snapshot("2026-07-27")
    path = write_delivery_receipt(
        snapshot,
        delivery_ok=True,
        attempted_at="2026-07-26T23:50:04+00:00",
        sent_at="2026-07-26T23:50:05+00:00",
        telegram_result=_telegram_result(
            "preopen radar",
            "2026-07-26T23:50:05+00:00",
        ),
        message="preopen radar",
        root=tmp_path,
    )
    document = json.loads(path.read_text(encoding="utf-8"))

    assert len(document["integrity_sha256"]) == 64
    assert delivery_receipt.manifest_digest_matches(
        document,
        digest_key="integrity_sha256",
    )
    assert document["message_sha256"] == hashlib.sha256(
        b"preopen radar"
    ).hexdigest()
    assert document["chunk_count"] == 1
    assert document["telegram_messages"] == [
        {
            "message_id": 101,
            "server_date": "2026-07-26T23:50:05+00:00",
            "text_sha256": hashlib.sha256(b"preopen radar").hexdigest(),
        }
    ]
    assert document["sent_at"] == document["telegram_messages"][0][
        "server_date"
    ]

    document["attempted_at"] = "2026-07-26T23:50:03+00:00"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(DeliveryReceiptError, match="outer integrity mismatch"):
        read_delivery_receipt(snapshot, root=tmp_path)


@pytest.mark.parametrize(
    "tamper",
    [
        "message_digest",
        "chat_digest",
        "chunk_count_bool",
        "messages_not_list",
        "success_incomplete",
        "nested_fields",
        "message_id",
        "text_digest",
        "server_not_utc",
        "server_before_attempt",
        "server_outside_window",
        "sent_not_server_max",
        "success_with_error",
        "failure_without_error",
        "failure_claims_all_chunks",
        "partial_without_chat",
    ],
)
def test_resealed_postactivation_receipt_still_enforces_semantics(
    tmp_path,
    monkeypatch,
    tamper,
):
    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            value = datetime.fromisoformat(
                "2026-07-26T23:50:06+00:00"
            )
            return value if tz is None else value.astimezone(tz)

    monkeypatch.setattr(delivery_receipt, "datetime", FixedDatetime)
    snapshot = _preopen_snapshot("2026-07-27")
    path = write_delivery_receipt(
        snapshot,
        delivery_ok=True,
        attempted_at="2026-07-26T23:50:04.900000+00:00",
        sent_at="2026-07-26T23:50:04+00:00",
        telegram_result=_telegram_result(
            "preopen radar",
            "2026-07-26T23:50:04+00:00",
        ),
        message="preopen radar",
        root=tmp_path,
    )
    document = json.loads(path.read_text(encoding="utf-8"))

    if tamper == "message_digest":
        document["message_sha256"] = "x" * 64
    elif tamper == "chat_digest":
        document["chat_id_sha256"] = "x" * 64
    elif tamper == "chunk_count_bool":
        document["chunk_count"] = True
    elif tamper == "messages_not_list":
        document["telegram_messages"] = {}
    elif tamper == "success_incomplete":
        document["telegram_messages"] = []
    elif tamper == "nested_fields":
        document["telegram_messages"][0]["unexpected"] = True
    elif tamper == "message_id":
        document["telegram_messages"][0]["message_id"] = 0
    elif tamper == "text_digest":
        document["telegram_messages"][0]["text_sha256"] = "x" * 64
    elif tamper == "server_not_utc":
        document["telegram_messages"][0]["server_date"] = (
            "2026-07-27T08:50:04+09:00"
        )
    elif tamper == "server_before_attempt":
        document["telegram_messages"][0]["server_date"] = (
            "2026-07-26T23:50:03+00:00"
        )
        document["sent_at"] = "2026-07-26T23:50:03+00:00"
    elif tamper == "server_outside_window":
        document["telegram_messages"][0]["server_date"] = (
            "2026-07-27T00:05:04+00:00"
        )
        document["sent_at"] = "2026-07-27T00:05:04+00:00"
        document["recorded_at"] = "2026-07-27T00:05:06+00:00"
    elif tamper == "sent_not_server_max":
        document["sent_at"] = "2026-07-26T23:50:05+00:00"
    elif tamper == "success_with_error":
        document["error"] = "unexpected"
    elif tamper == "failure_without_error":
        document["delivery_ok"] = False
        document["sent_at"] = None
        document["telegram_messages"] = []
    elif tamper == "failure_claims_all_chunks":
        document["delivery_ok"] = False
        document["sent_at"] = None
        document["error"] = "failed"
    else:
        document["delivery_ok"] = False
        document["sent_at"] = None
        document["error"] = "second chunk failed"
        document["chunk_count"] = 2
        document["chat_id_sha256"] = None

    document.pop("integrity_sha256")
    document = delivery_receipt.with_manifest_digest(
        document,
        digest_key="integrity_sha256",
    )
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(DeliveryReceiptError):
        read_delivery_receipt(snapshot, root=tmp_path)


@pytest.mark.parametrize("delivery_ok", [None, 0, 1, "false"])
def test_delivery_receipt_writer_rejects_non_boolean_status(
    tmp_path,
    delivery_ok,
):
    with pytest.raises(DeliveryReceiptError, match="must be bool"):
        write_delivery_receipt(
            _preopen_snapshot(),
            delivery_ok=delivery_ok,
            attempted_at="2026-07-25T00:00:04+00:00",
            sent_at=None,
            root=tmp_path,
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("field", "tampered"),
    [
        ("snapshot_path", "output/recommend_snapshots/other.json"),
        ("model_id", "other-model"),
    ],
)
def test_delivery_receipt_reader_rejects_wrong_provenance(
    tmp_path,
    field,
    tampered,
):
    snapshot = _preopen_snapshot()
    receipt_root = tmp_path / "receipts"
    path = write_delivery_receipt(
        snapshot,
        delivery_ok=True,
        attempted_at="2026-07-25T00:00:04+00:00",
        sent_at="2026-07-25T00:00:05+00:00",
        root=receipt_root,
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    document[field] = tampered
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(DeliveryReceiptError, match=field):
        read_delivery_receipt(snapshot, root=receipt_root)


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("duplicate", "duplicate JSON object key"),
        ("nan", "non-standard JSON numeric constant"),
        ("array", "must be a JSON object"),
        ("extra", "fields mismatch"),
        ("missing", "fields mismatch"),
    ],
)
def test_delivery_receipt_reader_rejects_noncanonical_json_contract(
    tmp_path,
    tamper,
    message,
):
    snapshot = _preopen_snapshot()
    path = write_delivery_receipt(
        snapshot,
        delivery_ok=False,
        attempted_at="2026-07-25T00:00:04+00:00",
        sent_at=None,
        error="transport failed",
        root=tmp_path,
    )
    text = path.read_text(encoding="utf-8")
    if tamper == "duplicate":
        text = text.replace(
            '"schema":',
            '"schema": "ambiguous",\n  "schema":',
            1,
        )
    elif tamper == "nan":
        text = text.replace('"error": "transport failed"', '"error": NaN')
    elif tamper == "array":
        text = "[]\n"
    else:
        document = json.loads(text)
        if tamper == "extra":
            document["unattributed"] = True
        else:
            del document["model_id"]
        text = json.dumps(document)
    path.write_text(text, encoding="utf-8")

    with pytest.raises(DeliveryReceiptError, match=message):
        read_delivery_receipt(snapshot, root=tmp_path)


def test_market_limited_receipt_cannot_overwrite_full_universe_receipt(tmp_path):
    full = _preopen_snapshot()
    full["request"] = {"limit_markets": None}
    limited = {
        **full,
        "snapshot_id": "recommend-limited",
        "request": {"limit_markets": 5},
    }

    assert receipt_path(full, root=tmp_path).name == "preopen_r1.json"
    assert receipt_path(limited, root=tmp_path).name == "preopen_r1.limit5.json"

    write_delivery_receipt(
        full,
        delivery_ok=False,
        attempted_at="2026-07-25T00:00:04+00:00",
        sent_at=None,
        root=tmp_path,
    )
    write_delivery_receipt(
        limited,
        delivery_ok=False,
        attempted_at="2026-07-25T00:00:04+00:00",
        sent_at=None,
        root=tmp_path,
    )
    assert receipt_path(full, root=tmp_path).exists()
    assert receipt_path(limited, root=tmp_path).exists()


@pytest.mark.parametrize(
    ("identity", "value", "message"),
    [
        ("asof", "../2026-07-25", "asof"),
        ("asof", "2026-7-25", "asof"),
        ("slot", "../open", "slot"),
        ("ranking", "../../escape", "ranking"),
        ("ranking", "champion_change_not-hex", "ranking"),
    ],
)
def test_receipt_path_rejects_noncanonical_or_unsafe_identity_before_io(
    tmp_path,
    identity,
    value,
    message,
):
    snapshot = _preopen_snapshot()
    if identity == "ranking":
        snapshot["model"] = {"ranking": value}
    else:
        snapshot[identity] = value

    with pytest.raises(DeliveryReceiptError, match=message):
        read_delivery_receipt(snapshot, root=tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_receipt_path_accepts_digest_bound_champion_notice(tmp_path):
    snapshot = _preopen_snapshot()
    snapshot["model"] = {"ranking": "champion_change_0123abcdef99"}

    path = receipt_path(snapshot, root=tmp_path)

    assert path == (
        tmp_path
        / "2026-07-25"
        / "preopen_champion_change_0123abcdef99.json"
    )


def test_delivery_receipt_rejects_naive_or_impossible_timestamp_chain(tmp_path):
    snapshot = _preopen_snapshot()
    path = write_delivery_receipt(
        snapshot,
        delivery_ok=True,
        attempted_at="2026-07-25T00:00:04+00:00",
        sent_at="2026-07-25T00:00:05+00:00",
        root=tmp_path,
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    document["sent_at"] = "2026-07-25T00:00:05"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(DeliveryReceiptError, match="timezone-aware"):
        read_delivery_receipt(snapshot, root=tmp_path)

    document["sent_at"] = "2026-07-25T00:00:03+00:00"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(DeliveryReceiptError, match="chronology"):
        read_delivery_receipt(snapshot, root=tmp_path)


def test_active_r1_requires_valid_receipt_before_ledger_append(
    tmp_path, monkeypatch
):
    snapshot = _preopen_snapshot()
    ledger = tmp_path / "preopen.csv"
    receipt_root = tmp_path / "receipts"
    monkeypatch.setattr(
        recommend_today,
        "get_or_create_recommend_snapshot",
        lambda *args, **kwargs: snapshot,
    )

    with pytest.raises(RuntimeError, match="receipt missing"):
        recommend_today.append_today(
            "2026-07-25",
            slot="preopen",
            ledger_path=str(ledger),
            receipt_root=receipt_root,
            require_receipt=True,
        )
    assert not ledger.exists()

    path = write_delivery_receipt(
        snapshot,
        delivery_ok=False,
        attempted_at="2026-07-25T00:00:04+00:00",
        sent_at=None,
        root=receipt_root,
    )
    with pytest.raises(RuntimeError, match="not successful"):
        recommend_today.append_today(
            "2026-07-25",
            slot="preopen",
            ledger_path=str(ledger),
            receipt_root=receipt_root,
            require_receipt=True,
        )
    assert not ledger.exists()

    path.write_text(path.read_text().replace(snapshot["snapshot_id"], "wrong"))
    with pytest.raises(RuntimeError, match="receipt invalid"):
        recommend_today.append_today(
            "2026-07-25",
            slot="preopen",
            ledger_path=str(ledger),
            receipt_root=receipt_root,
            require_receipt=True,
        )
    assert not ledger.exists()


def test_delivery_success_is_monotonic_and_retry_backfills_existing_ledger(
    tmp_path, monkeypatch
):
    snapshot = _preopen_snapshot()
    ledger = tmp_path / "preopen.csv"
    receipt_root = tmp_path / "receipts"
    monkeypatch.setattr(
        recommend_today,
        "get_or_create_recommend_snapshot",
        lambda *args, **kwargs: snapshot,
    )

    write_delivery_receipt(
        snapshot,
        delivery_ok=False,
        attempted_at="2026-07-25T00:00:01+00:00",
        sent_at=None,
        root=receipt_root,
    )
    recommend_today.append_today(
        "2026-07-25",
        slot="preopen",
        ledger_path=str(ledger),
        receipt_root=receipt_root,
    )
    assert pd.read_csv(ledger).loc[0, "delivery_ok"] == False  # noqa: E712

    first_success = "2026-07-25T00:00:05+00:00"
    write_delivery_receipt(
        snapshot,
        delivery_ok=True,
        attempted_at="2026-07-25T00:00:04+00:00",
        sent_at=first_success,
        root=receipt_root,
    )
    recommend_today.append_today(
        "2026-07-25",
        slot="preopen",
        ledger_path=str(ledger),
        receipt_root=receipt_root,
    )

    # 성공 뒤 실패한 재시도가 최초 성공 증거를 지우지 않는다.
    write_delivery_receipt(
        snapshot,
        delivery_ok=False,
        attempted_at="2026-07-25T00:01:00+00:00",
        sent_at=None,
        root=receipt_root,
    )
    receipt = read_delivery_receipt(snapshot, root=receipt_root)
    row = pd.read_csv(ledger).iloc[0]
    assert receipt["delivery_ok"] is True
    assert receipt["sent_at"] == first_success
    assert bool(row["delivery_ok"]) is True
    assert row["sent_at"] == first_success


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("coin", "candidate identity conflict"),
        ("score", "column=score"),
        ("snapshot_id", "snapshot_id conflict"),
        ("delivery_ok", "delivery_ok invalid"),
    ],
)
def test_same_date_legacy_row_must_match_immutable_snapshot_before_backfill(
    tmp_path, monkeypatch, mutation, match
):
    snapshot = _preopen_snapshot()
    ledger = tmp_path / "preopen.csv"
    receipt_root = tmp_path / "receipts"
    monkeypatch.setattr(
        recommend_today,
        "get_or_create_recommend_snapshot",
        lambda *args, **kwargs: snapshot,
    )
    recommend_today.append_today(
        snapshot["asof"],
        slot="preopen",
        ledger_path=str(ledger),
        receipt_root=receipt_root,
    )

    legacy = pd.read_csv(ledger)
    legacy["snapshot_path"] = pd.NA
    legacy["delivery_ok"] = pd.NA
    legacy["sent_at"] = pd.NA
    if mutation == "coin":
        legacy.loc[0, "coin"] = "KRW-OTHER"
        legacy["snapshot_id"] = pd.NA
    elif mutation == "score":
        legacy.loc[0, "score"] = 0.1
        legacy["snapshot_id"] = pd.NA
    elif mutation == "snapshot_id":
        legacy.loc[0, "snapshot_id"] = "recommend-wrong"
    else:
        legacy.loc[0, "delivery_ok"] = "maybe"
    legacy.to_csv(ledger, index=False)
    before = ledger.read_bytes()

    write_delivery_receipt(
        snapshot,
        delivery_ok=True,
        attempted_at="2026-07-25T00:00:04+00:00",
        sent_at="2026-07-25T00:00:05+00:00",
        root=receipt_root,
    )
    with pytest.raises(RuntimeError, match=match):
        recommend_today.append_today(
            snapshot["asof"],
            slot="preopen",
            ledger_path=str(ledger),
            receipt_root=receipt_root,
        )

    assert ledger.read_bytes() == before


def test_same_date_partial_legacy_row_cannot_gain_snapshot_provenance(
    tmp_path, monkeypatch
):
    snapshot = _preopen_snapshot()
    ledger = tmp_path / "preopen.csv"
    monkeypatch.setattr(
        recommend_today,
        "get_or_create_recommend_snapshot",
        lambda *args, **kwargs: snapshot,
    )
    recommend_today.append_today(
        snapshot["asof"],
        slot="preopen",
        ledger_path=str(ledger),
    )
    partial = pd.read_csv(ledger).drop(columns=["score"])
    partial["snapshot_id"] = pd.NA
    partial["snapshot_path"] = pd.NA
    partial.to_csv(ledger, index=False)
    before = ledger.read_bytes()

    with pytest.raises(RuntimeError, match="column=score"):
        recommend_today.append_today(
            snapshot["asof"],
            slot="preopen",
            ledger_path=str(ledger),
        )

    assert ledger.read_bytes() == before


def test_delivery_success_is_monotonic_under_concurrent_writers(
    tmp_path, monkeypatch
):
    snapshot = _preopen_snapshot()
    receipt_root = tmp_path / "receipts"
    failure_entered_write = threading.Event()
    release_failure = threading.Event()
    success_completed = threading.Event()
    errors: list[BaseException] = []
    original_atomic_write = delivery_receipt._atomic_write

    def controlled_atomic_write(path, document):
        if not document["delivery_ok"]:
            failure_entered_write.set()
            if not release_failure.wait(timeout=5):
                raise TimeoutError("test did not release failed receipt writer")
        original_atomic_write(path, document)

    monkeypatch.setattr(
        delivery_receipt, "_atomic_write", controlled_atomic_write
    )

    def write_failure():
        try:
            write_delivery_receipt(
                snapshot,
                delivery_ok=False,
                attempted_at="2026-07-25T00:00:01+00:00",
                sent_at=None,
                root=receipt_root,
            )
        except BaseException as exc:
            errors.append(exc)

    first_success = "2026-07-25T00:00:05+00:00"

    def write_success():
        try:
            write_delivery_receipt(
                snapshot,
                delivery_ok=True,
                attempted_at="2026-07-25T00:00:04+00:00",
                sent_at=first_success,
                root=receipt_root,
            )
        except BaseException as exc:
            errors.append(exc)
        finally:
            success_completed.set()

    failure_thread = threading.Thread(target=write_failure)
    success_thread = threading.Thread(target=write_success)
    failure_thread.start()
    assert failure_entered_write.wait(timeout=2)
    success_thread.start()

    # 실패 writer가 read-check-write lock을 잡은 동안 성공 writer는 대기해야 한다.
    assert not success_completed.wait(timeout=0.5)
    release_failure.set()
    failure_thread.join(timeout=5)
    success_thread.join(timeout=5)

    assert not failure_thread.is_alive()
    assert not success_thread.is_alive()
    assert not errors
    receipt = read_delivery_receipt(snapshot, root=receipt_root)
    assert receipt["delivery_ok"] is True
    assert receipt["sent_at"] == first_success


def test_append_preserves_existing_closed_path_metadata_columns(
    tmp_path, monkeypatch
):
    ledger = tmp_path / "recommend.csv"
    receipt_root = tmp_path / "receipts"
    first_snapshot = _preopen_snapshot("2026-07-25")
    second_snapshot = _preopen_snapshot("2026-07-26")
    snapshots = iter([first_snapshot, second_snapshot])
    monkeypatch.setattr(
        recommend_today,
        "get_or_create_recommend_snapshot",
        lambda *args, **kwargs: next(snapshots),
    )

    recommend_today.append_today(
        "2026-07-25",
        slot="preopen",
        ledger_path=str(ledger),
        receipt_root=receipt_root,
    )
    closed = pd.read_csv(ledger)
    extras = ["exit_tp10_net_pct", "path_complete", "path_status"]
    closed["status"] = "closed"
    closed["exit_price"] = 107.275
    closed["exit_reason"] = "EOD"
    closed["exit_tp10_net_pct"] = [7.125]
    closed["path_complete"] = [True]
    closed["path_status"] = ["flat_filled_complete"]
    closed.to_csv(ledger, index=False)
    old_extra_values = closed.loc[0, extras].copy()

    recommend_today.append_today(
        "2026-07-26",
        slot="preopen",
        ledger_path=str(ledger),
        receipt_root=receipt_root,
    )

    rows = pd.read_csv(ledger)
    assert list(rows.columns) == recommend_today.RECOMMEND_LEDGER_COLS + extras
    pd.testing.assert_series_equal(
        rows.loc[0, extras],
        old_extra_values,
        check_names=False,
    )
    assert rows.loc[0, "status"] == "closed"
    assert rows.loc[0, "exit_price"] == 107.275
    assert rows.loc[0, "exit_reason"] == "EOD"
    assert rows.loc[1, extras].isna().all()


def test_daily_preopen_records_and_closes_dedicated_r1_ledger():
    run_text = Path("scripts/daily_run_preopen.sh").read_text()
    close_text = Path("scripts/daily_close_preopen.sh").read_text()

    assert "python scripts/recommend_send.py --slot preopen" in run_text
    assert "python scripts/recommend_today.py --slot preopen" in run_text
    assert "--slot preopen --dry-run" not in run_text
    assert "python -m ops.close_input_gate" in close_text
    assert "--cohort r1-preopen" in close_text
    assert "python scripts/close_recommend_ledger.py" in close_text
    assert "--ledger output/shadow_ledger_recommend_preopen.csv" in close_text
    assert '--decision-date "$decision_date"' in close_text

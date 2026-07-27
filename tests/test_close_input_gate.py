from __future__ import annotations

import csv
import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

import notifier.delivery_receipt as recommend_receipt
import ops.close_input_gate as gate
import scripts.pump_detector_today as pump_v1
import scripts.pump_detector_v2_today as pump_v2
from notifier.telegram import TelegramSendResult, TelegramServerMessage


def _telegram_result(message: str, server_date: str) -> TelegramSendResult:
    digest = hashlib.sha256(message.encode()).hexdigest()
    return TelegramSendResult(
        delivery_ok=True,
        message_sha256=digest,
        chunk_count=1,
        chat_id_sha256=hashlib.sha256(b"456").hexdigest(),
        telegram_messages=(
            TelegramServerMessage(
                message_id=101,
                server_date=server_date,
                text_sha256=digest,
            ),
        ),
        error=None,
    )


def _expected(
    tmp_path: Path,
    *,
    candidates: tuple[tuple[str, int], ...],
) -> gate.ExpectedCohort:
    return gate.ExpectedCohort(
        asof="2026-07-25",
        evidence_id="immutable-id",
        evidence_path=tmp_path / "output" / "decision.json",
        candidates=candidates,
    )


def _pump_v1_decision(*, with_candidate: bool) -> dict:
    candidates = []
    if with_candidate:
        candidates.append(
            {
                "market": "KRW-AAA",
                "rank": 1,
                "score": 0.91,
                "estimated_pump20_prob": 0.064,
                "dump_risk_flag": False,
                "btc_regime": "bull_quiet",
                "entry_open": 100.0,
                "rule_id": "roc7_rank_pump20",
                "liq_rank_daily": 1,
                "roc_7d": 0.2,
                "roc_7d_rank": 0.91,
                "atr_pct_14": 0.08,
                "log_return_1d": 0.04,
                "pump20_rule": True,
                "pump15_rule": False,
                "estimated_pump15_prob": None,
                "overheated_flag": False,
            }
        )
    return {
        "asof": "2026-07-25",
        "feature_date": "2026-07-24",
        "model_id": "pump_hunter",
        "rule_version": "pump_detector_v1",
        "top_universe": 100,
        "universe_n": 100,
        "n_candidates": len(candidates),
        "rules": {"pump20": "rule-20", "pump15": "rule-15"},
        "candidates": candidates,
    }


def _write_v2_receipt(
    output: Path,
    *,
    asof: str,
    delivery_ok: bool = True,
) -> Path:
    decision_day = date.fromisoformat(asof)
    decision = {
        "asof": asof,
        "feature_date": str(decision_day - timedelta(days=1)),
        "model_id": "pump_hunter_v2",
        "rule_version": "pump_detector_v2",
        "rule": pump_v2.PUMP_V2_RULE,
        "btc_regime": "bear_quiet",
        "universe_n": 100,
        "n_candidates": 0,
        "binance_status": "binance_partial (ready=12/15)",
        "oos": dict(pump_v2.PUMP_V2_OOS),
        "candidates": [],
    }
    receipt = {
        "schema": pump_v2.PUMP_V2_RECEIPT_SCHEMA,
        "asof": asof,
        "decision_id": pump_v2._decision_id(decision),
        "decision": decision,
        "delivery_ok": delivery_ok,
        "attempted_at": f"{asof}T00:00:00+00:00",
        "sent_at": f"{asof}T00:00:01+00:00" if delivery_ok else None,
        "recorded_at": f"{asof}T00:00:02+00:00",
        "error": None if delivery_ok else "delivery failed",
    }
    if decision_day >= pump_v2.FORWARD_EVIDENCE_ACTIVATION_DATE:
        message = f"pump-v2:{asof}"
        digest = hashlib.sha256(message.encode()).hexdigest()
        receipt.update(
            {
                "message_sha256": digest,
                "chat_id_sha256": hashlib.sha256(b"456").hexdigest(),
                "chunk_count": 1,
                "telegram_messages": (
                    [
                        {
                            "message_id": 101,
                            "server_date": f"{asof}T00:00:01+00:00",
                            "text_sha256": digest,
                        }
                    ]
                    if delivery_ok
                    else []
                ),
            }
        )
        receipt = pump_v2._with_outer_integrity(receipt)
    path = output / "pump_v2_receipts" / f"{asof}.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(receipt, sort_keys=True),
        encoding="utf-8",
    )
    return path


def test_contract_activation_boundary_is_next_unattended_decision_day(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"

    assert gate.CLOSE_EVIDENCE_ACTIVATION_DATE == date(2026, 7, 27)
    assert gate.validate_close_input(
        asof="2026-07-26",
        cohort="pump-v1",
        output_root=output,
    ) == "skip-legacy-unverifiable"
    with pytest.raises(
        gate.MissingCloseEvidenceError,
        match="canonical same-day evidence missing",
    ):
        gate.validate_close_input(
            asof="2026-07-27",
            cohort="pump-v1",
            output_root=output,
        )


def test_post_activation_close_rejects_legacy_recommend_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "output"
    evidence = output / "recommend_snapshots/2026-07-27/open_r2.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        gate,
        "load_snapshot",
        lambda *_args, **_kwargs: {
            "asof": "2026-07-27",
            "snapshot_schema": "recommend_snapshot.v1",
            "snapshot_id": "legacy-id",
            "calibration_source": "legacy",
            "top3": [],
        },
    )

    with pytest.raises(
        gate.CloseInputError,
        match="legacy schema after close-evidence activation",
    ):
        gate._load_expected(
            asof="2026-07-27",
            cohort="r2",
            output_root=output,
        )


def test_post_activation_close_rejects_receipt_outer_tamper(
    tmp_path: Path,
    monkeypatch,
) -> None:
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
                tzinfo=timezone.utc,
            )
            return value if tz is None else value.astimezone(tz)

    output = tmp_path / "output"
    evidence = output / "recommend_snapshots/2026-07-27/open_r2.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("{}", encoding="utf-8")
    snapshot = {
        "asof": "2026-07-27",
        "slot": "open",
        "snapshot_schema": gate.SNAPSHOT_SCHEMA_VERSION,
        "snapshot_id": "recommend-forward-r2",
        "snapshot_path": str(evidence),
        "calibration_source": "test",
        "top3": [],
        "model": {"id": "recommend_r2_open", "ranking": "R2"},
        "request": {"limit_markets": None},
    }
    monkeypatch.setattr(gate, "load_snapshot", lambda *_args, **_kwargs: snapshot)
    monkeypatch.setattr(recommend_receipt, "datetime", FixedDatetime)
    receipt_path = recommend_receipt.write_delivery_receipt(
        snapshot,
        delivery_ok=True,
        attempted_at="2026-07-27T00:05:00+00:00",
        sent_at="2026-07-27T00:05:01+00:00",
        telegram_result=_telegram_result(
            "open radar",
            "2026-07-27T00:05:01+00:00",
        ),
        message="open radar",
        root=output / "recommend_receipts",
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["sent_at"] = "2026-07-27T00:05:00.500000+00:00"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(
        gate.CloseInputError,
        match="canonical delivery receipt invalid",
    ):
        gate._load_expected(
            asof="2026-07-27",
            cohort="r2",
            output_root=output,
        )


def test_pre_activation_skip_does_not_blanket_ignore_present_bad_evidence(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    path = output / "pump_v1_decisions" / "2026-07-26.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(gate.CloseInputError, match="decision payload missing"):
        gate.validate_close_input(
            asof="2026-07-26",
            cohort="pump-v1",
            output_root=output,
        )


@pytest.mark.parametrize(
    "body",
    [
        '{"schema":"first","schema":"second"}',
        '{"schema":NaN}',
        "[]",
    ],
)
def test_canonical_close_evidence_rejects_ambiguous_json(
    tmp_path: Path,
    body: str,
) -> None:
    output = tmp_path / "output"
    path = output / "pump_v1_decisions" / "2026-07-26.json"
    path.parent.mkdir(parents=True)
    path.write_text(body, encoding="utf-8")

    with pytest.raises(gate.CloseInputError, match="canonical evidence"):
        gate.validate_close_input(
            asof="2026-07-26",
            cohort="pump-v1",
            output_root=output,
        )


def test_canonical_close_evidence_rejects_symlink(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    target = output / "outside.json"
    target.parent.mkdir(parents=True)
    target.write_text("{}", encoding="utf-8")
    path = output / "pump_v1_decisions" / "2026-07-26.json"
    path.parent.mkdir(parents=True)
    path.symlink_to(target)

    with pytest.raises(gate.CloseInputError, match="canonical evidence"):
        gate.validate_close_input(
            asof="2026-07-26",
            cohort="pump-v1",
            output_root=output,
        )


def test_pre_activation_pending_row_without_evidence_fails_closed(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "shadow_ledger_pump_hunter.csv").write_text(
        "date,status\n2026-07-26,open\n",
        encoding="utf-8",
    )

    with pytest.raises(
        gate.CloseInputError,
        match="pending ledger rows lack canonical evidence",
    ):
        gate.validate_close_input(
            asof="2026-07-26",
            cohort="pump-v1",
            output_root=output,
        )


def test_pre_activation_v2_missing_manifest_accepts_only_valid_success_receipt(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    receipt_path = _write_v2_receipt(output, asof="2026-07-26")

    assert gate.validate_close_input(
        asof="2026-07-26",
        cohort="pump-v2",
        output_root=output,
    ) == "skip-legacy-unverifiable"

    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["decision_id"] = "pump-v2-tampered"
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(gate.CloseInputError, match="v2 receipt invalid"):
        gate.validate_close_input(
            asof="2026-07-26",
            cohort="pump-v2",
            output_root=output,
        )


def test_valid_receipt_is_not_a_post_activation_manifest_substitute(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    _write_v2_receipt(output, asof="2026-07-27")

    with pytest.raises(gate.MissingCloseEvidenceError):
        gate.validate_close_input(
            asof="2026-07-27",
            cohort="pump-v2",
            output_root=output,
        )


def test_pre_activation_failed_v2_receipt_is_not_derivation_evidence(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    _write_v2_receipt(
        output,
        asof="2026-07-26",
        delivery_ok=False,
    )

    with pytest.raises(
        gate.CloseInputError,
        match="not a successful same-day receipt",
    ):
        gate.validate_close_input(
            asof="2026-07-26",
            cohort="pump-v2",
            output_root=output,
        )


def test_clean_start_missing_ledger_is_healthy_only_for_canonical_zero_pick(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    pump_v1.persist_decision(
        _pump_v1_decision(with_candidate=False),
        decision_root=output / "pump_v1_decisions",
    )

    assert gate.validate_close_input(
        asof="2026-07-25",
        cohort="pump-v1",
        output_root=output,
    ) == "skip-zero-pick"


def test_clean_start_candidate_bearing_missing_ledger_fails_closed(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    pump_v1.persist_decision(
        _pump_v1_decision(with_candidate=True),
        decision_root=output / "pump_v1_decisions",
    )

    with pytest.raises(
        gate.CloseInputError,
        match="candidate-bearing ledger missing",
    ):
        gate.validate_close_input(
            asof="2026-07-25",
            cohort="pump-v1",
            output_root=output,
        )


def test_zero_pick_gate_rejects_symlinked_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    evidence = output / "decision.json"
    evidence.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        gate,
        "_load_expected",
        lambda **_kwargs: gate.ExpectedCohort(
            asof="2026-07-25",
            evidence_id="immutable-id",
            evidence_path=evidence.resolve(),
            candidates=(),
        ),
    )
    outside = tmp_path / "outside.csv"
    outside.write_text(
        "date,coin,rank,snapshot_id,snapshot_path,status\n",
        encoding="utf-8",
    )
    (output / "shadow_ledger_pump_hunter.csv").symlink_to(outside)

    with pytest.raises(gate.CloseInputError, match="ledger is unreadable"):
        gate.validate_close_input(
            asof="2026-07-25",
            cohort="pump-v1",
            output_root=output,
        )


def test_existing_ledger_must_match_candidate_and_evidence_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    evidence = output / "decision.json"
    evidence.write_text("{}", encoding="utf-8")
    expected = gate.ExpectedCohort(
        asof="2026-07-25",
        evidence_id="immutable-id",
        evidence_path=evidence.resolve(),
        candidates=(("KRW-AAA", 1),),
    )
    monkeypatch.setattr(gate, "_load_expected", lambda **_kwargs: expected)
    ledger = output / "shadow_ledger_pump_hunter.csv"
    ledger.write_text(
        "date,coin,rank,snapshot_id,snapshot_path,status\n"
        f"2026-07-25,KRW-AAA,1,immutable-id,{evidence},open\n",
        encoding="utf-8",
    )

    assert gate.validate_close_input(
        asof="2026-07-25",
        cohort="pump-v1",
        output_root=output,
    ) == "close"

    ledger.write_text(
        "date,coin,rank,snapshot_id,snapshot_path,status\n"
        f"2026-07-25,KRW-BBB,1,immutable-id,{evidence},open\n",
        encoding="utf-8",
    )
    with pytest.raises(gate.CloseInputError, match="candidate mismatch"):
        gate.validate_close_input(
            asof="2026-07-25",
            cohort="pump-v1",
            output_root=output,
        )


@pytest.mark.parametrize(
    ("field", "tampered"),
    [
        ("score", "0.12"),
        ("pump_prob", "0.99"),
        ("entry_open", "101.0"),
        ("sl_pct", "-0.30"),
        ("tp_pct", "0.50"),
        ("delivery_ok", ""),
        ("sent_at", ""),
    ],
)
def test_close_gate_rejects_signal_or_delivery_metadata_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    tampered: str,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    evidence = output / "decision.json"
    evidence.write_text("{}", encoding="utf-8")
    immutable_row = {
        "date": "2026-07-25",
        "coin": "KRW-AAA",
        "rank": 1,
        "score": 0.91,
        "pump_prob": 0.064,
        "entry_open": 100.0,
        "sl_pct": -0.03,
        "tp_pct": 0.05,
    }
    expected = gate.ExpectedCohort(
        asof="2026-07-25",
        evidence_id="immutable-id",
        evidence_path=evidence.resolve(),
        candidates=(("KRW-AAA", 1),),
        immutable_rows=(immutable_row,),
        validate_delivery_metadata=True,
        delivery_ok=True,
        sent_at="2026-07-25T00:05:00+00:00",
    )
    monkeypatch.setattr(gate, "_load_expected", lambda **_kwargs: expected)
    row = {
        **{key: str(value) for key, value in immutable_row.items()},
        "snapshot_id": "immutable-id",
        "snapshot_path": str(evidence),
        "delivery_ok": "True",
        "sent_at": "2026-07-25T00:05:00+00:00",
        "status": "open",
    }
    row[field] = tampered
    ledger = output / "shadow_ledger_recommend.csv"
    with ledger.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    with pytest.raises(gate.CloseInputError, match="mismatch"):
        gate.validate_close_input(
            asof="2026-07-25",
            cohort="r1-open",
            output_root=output,
        )


def test_close_gate_accepts_exact_signal_and_delivery_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    evidence = output / "decision.json"
    evidence.write_text("{}", encoding="utf-8")
    immutable_row = {
        "date": "2026-07-25",
        "coin": "KRW-AAA",
        "rank": 1,
        "score": 0.91,
        "entry_open": 100.0,
        "sl_pct": -0.03,
        "tp_pct": 0.05,
        "p_up5": None,
    }
    expected = gate.ExpectedCohort(
        asof="2026-07-25",
        evidence_id="immutable-id",
        evidence_path=evidence.resolve(),
        candidates=(("KRW-AAA", 1),),
        immutable_rows=(immutable_row,),
        validate_delivery_metadata=True,
        delivery_ok=True,
        sent_at="2026-07-25T00:05:00+00:00",
    )
    monkeypatch.setattr(gate, "_load_expected", lambda **_kwargs: expected)
    row = {
        **{
            key: "" if value is None else str(value)
            for key, value in immutable_row.items()
        },
        "snapshot_id": "immutable-id",
        "snapshot_path": str(evidence),
        "delivery_ok": "True",
        "sent_at": "2026-07-25T00:05:00+00:00",
        "status": "open",
    }
    ledger = output / "shadow_ledger_recommend.csv"
    with ledger.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    assert gate.validate_close_input(
        asof="2026-07-25",
        cohort="r1-open",
        output_root=output,
    ) == "close"


def test_close_plan_validates_yesterday_and_every_pending_backlog_date(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    ledger = output / "shadow_ledger_recommend.csv"
    ledger.write_text(
        "date,status\n"
        "2026-07-23,closed\n"
        "2026-07-24,no_data\n"
        "2026-07-25,open\n"
        "2026-07-27,open\n",
        encoding="utf-8",
    )
    calls = []

    def fake_validate(**kwargs):
        calls.append(kwargs["asof"])
        return "close"

    monkeypatch.setattr(gate, "validate_close_input", fake_validate)

    assert gate.validate_close_plan(
        through_asof="2026-07-26",
        cohort="r1-open",
        output_root=output,
    ) == (
        ("2026-07-24", "close"),
        ("2026-07-25", "close"),
        ("2026-07-26", "close"),
    )
    assert calls == ["2026-07-24", "2026-07-25", "2026-07-26"]


def test_close_plan_rejects_noncanonical_pending_date(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    ledger = output / "shadow_ledger_recommend.csv"
    ledger.write_text(
        "date,status\n2026-7-24,open\n",
        encoding="utf-8",
    )

    with pytest.raises(gate.CloseInputError, match="date invalid"):
        gate.validate_close_plan(
            through_asof="2026-07-26",
            cohort="r1-open",
            output_root=output,
        )


def test_close_plan_rejects_unknown_status_instead_of_hiding_row(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    ledger = output / "shadow_ledger_recommend.csv"
    ledger.write_text(
        "date,status\n2026-07-24,opne\n",
        encoding="utf-8",
    )

    with pytest.raises(gate.CloseInputError, match="invalid status"):
        gate.validate_close_plan(
            through_asof="2026-07-26",
            cohort="r1-open",
            output_root=output,
        )

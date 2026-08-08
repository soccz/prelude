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
from ops.radar_verdict import record_terminal_verdict, terminal_candidate


def _write_terminal_kill(
    output: Path,
) -> dict:
    scorecard = {
        "status": "early_kill",
        "closed_n": 9,
        "mean_net_pct": -0.09658888888888893,
        "ci95": [-2.3315745735651023, 2.1383967957873247],
        "per_day_t": 0.028541508607332546,
        "regimes": ["bear_quiet"],
        "criteria": {
            "n>=200": False,
            "mean>0": False,
            "CI_0_제외": False,
            "2레짐_or_t>=2": False,
        },
        "criteria_met": 0,
        "early_kill_breached": True,
        "terminal_metric_values": {
            "mean_net_pct": -0.09658888888888893,
            "ci95": [-2.3315745735651023, 2.1383967957873247],
            "per_day_t": 0.028541508607332546,
        },
    }
    candidate = terminal_candidate(
        scorecard,
        asof=date(2026, 8, 5),
        recorded_at=datetime.fromisoformat(
            "2026-08-05T01:30:43.423095+00:00"
        ),
    )
    assert candidate is not None
    return record_terminal_verdict(
        candidate,
        path=output / "radar_terminal_verdict.json",
    )


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


def test_pre_activation_pending_row_without_evidence_stays_legacy_open(
    tmp_path: Path,
) -> None:
    # 계약 이전 pending 행 + 증거 전무 = 영원히 검증 불가.  매일 close 전체를
    # fail 시키는 대신 legacy-unverifiable 로 열어 둔다 (행은 원장에 그대로
    # 보이고 verified 통계에는 인증 행만 들어간다) — 2026-07-28 회귀 수리.
    output = tmp_path / "output"
    output.mkdir()
    (output / "shadow_ledger_pump_hunter.csv").write_text(
        "date,status\n2026-07-26,open\n",
        encoding="utf-8",
    )

    assert gate.validate_close_input(
        asof="2026-07-26",
        cohort="pump-v1",
        output_root=output,
    ) == "skip-legacy-unverifiable"


def test_post_activation_pending_row_without_evidence_still_fails(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "shadow_ledger_pump_hunter.csv").write_text(
        "date,status\n2026-07-27,open\n",
        encoding="utf-8",
    )

    with pytest.raises(gate.MissingCloseEvidenceError):
        gate.validate_close_input(
            asof="2026-07-27",
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


def _seam_ledger_setup(
    tmp_path: Path,
    monkeypatch,
    *,
    asof: str,
    blank_seam_column: bool = False,
    extra_expected_column: str | None = None,
):
    """Old-schema ledger row + new-schema immutable evidence for one day."""
    output = tmp_path / "output"
    output.mkdir()
    evidence = output / "decision.json"
    evidence.write_text("{}", encoding="utf-8")
    immutable_row = {
        "date": asof,
        "coin": "KRW-AAA",
        "rank": 1,
        "score": 0.91,
        "entry_open": 100.0,
        "decision_completed_at": f"{asof}T00:08:00+00:00",
    }
    if extra_expected_column is not None:
        immutable_row[extra_expected_column] = "expected-value"
    expected = gate.ExpectedCohort(
        asof=asof,
        evidence_id="immutable-id",
        evidence_path=evidence.resolve(),
        candidates=(("KRW-AAA", 1),),
        immutable_rows=(immutable_row,),
        validate_delivery_metadata=True,
        delivery_ok=True,
        sent_at=f"{asof}T00:05:00+00:00",
    )
    monkeypatch.setattr(gate, "_load_expected", lambda **_kwargs: expected)
    row = {
        "date": asof,
        "coin": "KRW-AAA",
        "rank": "1",
        "score": "0.91",
        "entry_open": "100.0",
        "snapshot_id": "immutable-id",
        "snapshot_path": str(evidence),
        "delivery_ok": "True",
        "sent_at": f"{asof}T00:05:00+00:00",
        "status": "open",
    }
    if blank_seam_column:
        row["decision_completed_at"] = ""
    ledger = output / "shadow_ledger_recommend.csv"
    with ledger.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return output, ledger


def test_pre_activation_seam_tolerates_old_ledger_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 07-26 이음매: snapshot/receipt 는 신형인데 원장 행은 구스키마 writer 가
    # 이미 아침에 적재한 상태.  없는 컬럼은 관용, 있는 컬럼 값은 그대로 강제.
    output, _ = _seam_ledger_setup(tmp_path, monkeypatch, asof="2026-07-25")

    assert gate.validate_close_input(
        asof="2026-07-25",
        cohort="r1-open",
        output_root=output,
    ) == "close"


def test_pre_activation_seam_still_rejects_present_value_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, ledger = _seam_ledger_setup(
        tmp_path,
        monkeypatch,
        asof="2026-07-25",
    )
    text = ledger.read_text(encoding="utf-8").replace("0.91", "0.95")
    ledger.write_text(text, encoding="utf-8")

    with pytest.raises(gate.CloseInputError, match="immutable value mismatch"):
        gate.validate_close_input(
            asof="2026-07-25",
            cohort="r1-open",
            output_root=output,
        )


def test_post_activation_missing_schema_columns_still_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, _ = _seam_ledger_setup(tmp_path, monkeypatch, asof="2026-07-27")

    with pytest.raises(
        gate.CloseInputError,
        match="immutable schema missing",
    ):
        gate.validate_close_input(
            asof="2026-07-27",
            cohort="r1-open",
            output_root=output,
        )


def test_close_plan_records_evidence_free_target_day_as_no_decision(
    tmp_path: Path,
) -> None:
    # 발송 파이프가 그날 아예 죽어 snapshot 도 원장 행도 없는 post-activation
    # 날짜는 백로그 전체를 오염시키지 않고 no-decision 으로만 기록한다.
    output = tmp_path / "output"
    output.mkdir()
    ledger = output / "shadow_ledger_recommend.csv"
    ledger.write_text(
        "date,status\n2026-07-26,closed\n",
        encoding="utf-8",
    )

    assert gate.validate_close_plan(
        through_asof="2026-07-27",
        cohort="r1-open",
        output_root=output,
    ) == (("2026-07-27", "skip-no-decision"),)


def test_post_kill_pump_v2_absence_is_terminal_noop(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    _write_terminal_kill(output)

    assert gate.validate_close_input(
        asof="2026-08-07",
        cohort="pump-v2",
        output_root=output,
    ) == "skip-terminal-kill"
    assert gate.validate_close_plan(
        through_asof="2026-08-07",
        cohort="pump-v2",
        output_root=output,
    ) == (
        ("2026-08-06", "skip-terminal-kill"),
        ("2026-08-07", "skip-terminal-kill"),
    )
    assert not gate.is_no_decision_day(
        asof="2026-08-07",
        cohort="pump-v2",
        output_root=output,
    )


def test_close_plan_does_not_repeat_valid_policy_noop_marker(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    _write_terminal_kill(output)
    mode, state = gate.validate_policy_noop(
        asof="2026-08-06",
        cohort="pump-v2",
        output_root=output,
    )
    marker = gate.policy_noop_marker_path(
        output_root=output,
        cohort="pump-v2",
        asof="2026-08-06",
        mode=mode,
    )
    marker.parent.mkdir(parents=True)
    marker.write_text(
        json.dumps(
            {
                **gate.policy_noop_marker_identity(
                    cohort="pump-v2",
                    asof="2026-08-06",
                    mode=mode,
                    terminal_state=state,
                ),
                "recorded_at": "2026-08-08T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    assert gate.validate_close_plan(
        through_asof="2026-08-07",
        cohort="pump-v2",
        output_root=output,
    ) == (("2026-08-07", "skip-terminal-kill"),)


def test_post_retirement_missing_terminal_pair_fails_closed(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()

    with pytest.raises(gate.CloseInputError, match="pair missing"):
        gate.validate_close_input(
            asof="2026-08-07",
            cohort="pump-v2",
            output_root=output,
        )


@pytest.mark.parametrize(
    "artifact",
    ("decision", "receipt", "ledger"),
)
def test_post_kill_pump_v2_rejects_lingering_artifacts(
    tmp_path: Path,
    artifact: str,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    _write_terminal_kill(output)
    if artifact == "decision":
        path = output / "pump_v2_decisions/2026-08-07.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}", encoding="utf-8")
    elif artifact == "receipt":
        path = output / "pump_v2_receipts/2026-08-07.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}", encoding="utf-8")
    else:
        path = output / "shadow_ledger_pump_hunter_v2.csv"
        path.write_text(
            "date,status\n2026-08-07,open\n",
            encoding="utf-8",
        )

    with pytest.raises(gate.CloseInputError, match="after terminal KILL"):
        gate.validate_close_input(
            asof="2026-08-07",
            cohort="pump-v2",
            output_root=output,
        )


def test_terminal_noop_rejects_tampered_anchor(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    _write_terminal_kill(output)
    anchor = output / "radar_terminal_verdict.json.anchor"
    anchor.write_text("{}", encoding="utf-8")

    with pytest.raises(
        gate.CloseInputError,
        match="state/anchor validation failed",
    ):
        gate.validate_close_input(
            asof="2026-08-07",
            cohort="pump-v2",
            output_root=output,
        )


@pytest.mark.parametrize("cohort", ("r1-open", "r1-preopen"))
def test_r1_shared_kill_gap_is_bounded_policy_noop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cohort: str,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    _write_terminal_kill(output)
    spec = gate.COHORTS[cohort]
    evidence = output / spec.evidence_name.format(asof="2026-08-07")
    evidence.parent.mkdir(parents=True)
    evidence.write_text("{}", encoding="utf-8")

    def expected_without_delivery(**kwargs):
        assert kwargs["require_successful_delivery"] is False
        return gate.ExpectedCohort(
            asof="2026-08-07",
            evidence_id="snapshot-id",
            evidence_path=evidence,
            candidates=(("KRW-TEST", 1),),
        )

    monkeypatch.setattr(gate, "_load_expected", expected_without_delivery)

    assert gate.validate_close_input(
        asof="2026-08-07",
        cohort=cohort,
        output_root=output,
    ) == "skip-policy-blocked"


def test_r1_resume_restores_strict_receipt_requirement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    _write_terminal_kill(output)

    def require_receipt(**kwargs):
        assert kwargs.get("require_successful_delivery") is None
        raise gate.CloseInputError("requires a successful canonical receipt")

    monkeypatch.setattr(gate, "_load_expected", require_receipt)

    with pytest.raises(
        gate.CloseInputError,
        match="requires a successful canonical receipt",
    ):
        gate.validate_close_input(
            asof=gate.R1_RESUME_ASOF.isoformat(),
            cohort="r1-open",
            output_root=output,
        )


@pytest.mark.parametrize("cohort", ("r1-open", "r2", "a1", "pump-v1"))
def test_non_v2_close_is_independent_of_corrupt_v2_terminal_after_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cohort: str,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "radar_terminal_verdict.json").write_text(
        "{",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        gate,
        "_load_expected",
        lambda **_kwargs: gate.ExpectedCohort(
            asof=gate.R1_RESUME_ASOF.isoformat(),
            evidence_id="evidence-id",
            evidence_path=output / "evidence.json",
            candidates=(),
        ),
    )

    assert gate.validate_close_input(
        asof=gate.R1_RESUME_ASOF.isoformat(),
        cohort=cohort,
        output_root=output,
    ) == "skip-zero-pick"


@pytest.mark.parametrize("cohort", ("r1-open", "pump-v2"))
def test_pre_retirement_close_does_not_read_later_terminal_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cohort: str,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "radar_terminal_verdict.json").write_text("{", encoding="utf-8")
    monkeypatch.setattr(
        gate,
        "_load_expected",
        lambda **_kwargs: gate.ExpectedCohort(
            asof="2026-07-30",
            evidence_id="evidence-id",
            evidence_path=output / "evidence.json",
            candidates=(),
        ),
    )

    assert gate.validate_close_input(
        asof="2026-07-30",
        cohort=cohort,
        output_root=output,
    ) == "skip-zero-pick"


@pytest.mark.parametrize("cohort", ("r1-open", "r1-preopen"))
def test_resumed_r1_plan_uses_recorded_seam_markers_without_v2_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cohort: str,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "radar_terminal_verdict.json").write_text("{", encoding="utf-8")
    terminal_identity = {
        "verdict_id": gate.PUMP_V2_TERMINAL_VERDICT_ID,
        "effective_asof": gate.PUMP_V2_TERMINAL_EFFECTIVE_ASOF.isoformat(),
    }
    for day in ("2026-08-06", "2026-08-07", "2026-08-08"):
        marker = gate.policy_noop_marker_path(
            output_root=output,
            cohort=cohort,
            asof=day,
            mode="skip-policy-blocked",
        )
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps(
                {
                    **gate.policy_noop_marker_identity(
                        cohort=cohort,
                        asof=day,
                        mode="skip-policy-blocked",
                        terminal_state=terminal_identity,
                    ),
                    "recorded_at": "2026-08-08T00:00:00+00:00",
                }
            ),
            encoding="utf-8",
        )
    monkeypatch.setattr(
        gate,
        "_load_expected",
        lambda **_kwargs: gate.ExpectedCohort(
            asof="2026-08-09",
            evidence_id="evidence-id",
            evidence_path=output / "evidence.json",
            candidates=(),
        ),
    )

    assert gate.validate_close_plan(
        through_asof="2026-08-09",
        cohort=cohort,
        output_root=output,
    ) == (("2026-08-09", "skip-zero-pick"),)


@pytest.mark.parametrize("cohort", ("r1-open", "r1-preopen"))
@pytest.mark.parametrize(
    ("artifact", "error"),
    (("ledger", "ledger rows"), ("receipt", "delivery receipt")),
)
def test_recorded_r1_policy_seam_rejects_retroactive_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cohort: str,
    artifact: str,
    error: str,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    # Completed R1 seam markers deliberately remain usable if the retired-v2
    # pair later becomes unavailable, but they must not hide retroactive PnL
    # or a fabricated delivery receipt.
    (output / "radar_terminal_verdict.json").write_text(
        "{",
        encoding="utf-8",
    )
    terminal_identity = {
        "verdict_id": gate.PUMP_V2_TERMINAL_VERDICT_ID,
        "effective_asof": gate.PUMP_V2_TERMINAL_EFFECTIVE_ASOF.isoformat(),
    }
    for day in ("2026-08-06", "2026-08-07", "2026-08-08"):
        marker = gate.policy_noop_marker_path(
            output_root=output,
            cohort=cohort,
            asof=day,
            mode="skip-policy-blocked",
        )
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps(
                {
                    **gate.policy_noop_marker_identity(
                        cohort=cohort,
                        asof=day,
                        mode="skip-policy-blocked",
                        terminal_state=terminal_identity,
                    ),
                    "recorded_at": "2026-08-08T00:00:00+00:00",
                }
            ),
            encoding="utf-8",
        )

    spec = gate.COHORTS[cohort]
    if artifact == "ledger":
        (output / spec.ledger_name).write_text(
            "date,status\n2026-08-06,closed\n",
            encoding="utf-8",
        )
    else:
        evidence = output / spec.evidence_name.format(asof="2026-08-06")
        receipt = output / "recommend_receipts/2026-08-06" / evidence.name
        receipt.parent.mkdir(parents=True)
        receipt.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        gate,
        "_load_expected",
        lambda **_kwargs: gate.ExpectedCohort(
            asof="2026-08-09",
            evidence_id="evidence-id",
            evidence_path=output / "evidence.json",
            candidates=(),
        ),
    )

    with pytest.raises(gate.CloseInputError, match=error):
        gate.validate_close_plan(
            through_asof="2026-08-09",
            cohort=cohort,
            output_root=output,
        )


def test_recorded_terminal_skip_rejects_later_pump_v2_decision(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    state = _write_terminal_kill(output)
    marker = gate.policy_noop_marker_path(
        output_root=output,
        cohort="pump-v2",
        asof="2026-08-06",
        mode="skip-terminal-kill",
    )
    marker.parent.mkdir(parents=True)
    marker.write_text(
        json.dumps(
            {
                **gate.policy_noop_marker_identity(
                    cohort="pump-v2",
                    asof="2026-08-06",
                    mode="skip-terminal-kill",
                    terminal_state=state,
                ),
                "recorded_at": "2026-08-08T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    decision = output / "pump_v2_decisions/2026-08-06.json"
    decision.parent.mkdir(parents=True)
    decision.write_text("{}", encoding="utf-8")

    with pytest.raises(gate.CloseInputError, match="canonical artifacts"):
        gate.validate_close_plan(
            through_asof="2026-08-06",
            cohort="pump-v2",
            output_root=output,
        )


def test_close_plan_pending_rows_without_evidence_still_fail_closed(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    ledger = output / "shadow_ledger_recommend.csv"
    ledger.write_text(
        "date,status\n2026-07-27,open\n",
        encoding="utf-8",
    )

    with pytest.raises(gate.MissingCloseEvidenceError):
        gate.validate_close_plan(
            through_asof="2026-07-27",
            cohort="r1-open",
            output_root=output,
        )


def test_pre_activation_seam_tolerates_blank_backfilled_seam_column(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 07-28 아침 append 가 백로그 행에 seam 컬럼을 pd.NA(공란)로 채워도
    # allowlist 컬럼에 한해 관용한다 (C2 재발 방지).
    output, _ = _seam_ledger_setup(
        tmp_path,
        monkeypatch,
        asof="2026-07-25",
        blank_seam_column=True,
    )

    assert gate.validate_close_input(
        asof="2026-07-25",
        cohort="r1-open",
        output_root=output,
    ) == "close"


def test_pre_activation_seam_rejects_non_allowlisted_missing_column(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, _ = _seam_ledger_setup(
        tmp_path,
        monkeypatch,
        asof="2026-07-25",
        extra_expected_column="decision_started_at",
    )

    with pytest.raises(
        gate.CloseInputError,
        match="immutable schema missing",
    ):
        gate.validate_close_input(
            asof="2026-07-25",
            cohort="r1-open",
            output_root=output,
        )


def test_pre_activation_seam_rejects_blanked_non_allowlisted_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, ledger = _seam_ledger_setup(
        tmp_path,
        monkeypatch,
        asof="2026-07-25",
    )
    text = ledger.read_text(encoding="utf-8").replace("0.91", "")
    ledger.write_text(text, encoding="utf-8")

    with pytest.raises(gate.CloseInputError, match="immutable value mismatch"):
        gate.validate_close_input(
            asof="2026-07-25",
            cohort="r1-open",
            output_root=output,
        )


@pytest.mark.parametrize("status", ["closed", "not_delivered"])
def test_no_decision_rejects_any_status_ledger_row(
    tmp_path: Path,
    status: str,
) -> None:
    # closed/not_delivered 행이 있는 날은 결정이 존재했던 날 — 정본 증거
    # 부재는 무결성 실패이지 조용한 skip 대상이 아니다 (C1).
    output = tmp_path / "output"
    output.mkdir()
    ledger = output / "shadow_ledger_recommend.csv"
    ledger.write_text(
        f"date,status\n2026-07-27,{status}\n",
        encoding="utf-8",
    )

    with pytest.raises(gate.MissingCloseEvidenceError):
        gate.validate_close_plan(
            through_asof="2026-07-27",
            cohort="r1-open",
            output_root=output,
        )


def test_no_decision_rejects_lingering_delivery_receipt(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    ledger = output / "shadow_ledger_recommend.csv"
    ledger.write_text(
        "date,status\n2026-07-26,closed\n",
        encoding="utf-8",
    )
    receipt_dir = output / "recommend_receipts" / "2026-07-27"
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "open_r1.json").write_text("{}", encoding="utf-8")

    with pytest.raises(gate.MissingCloseEvidenceError):
        gate.validate_close_plan(
            through_asof="2026-07-27",
            cohort="r1-open",
            output_root=output,
        )


def test_no_decision_ledger_reader_rejects_duplicate_header(
    tmp_path: Path,
) -> None:
    # 중복 date 헤더로 실제 행을 가리는 원장은 형제 리더와 동일하게 거부 —
    # 락 하 최종 권위 지점에서 skip-no-decision 위장을 막는다 (N1).
    output = tmp_path / "output"
    output.mkdir()
    ledger = output / "shadow_ledger_recommend.csv"
    ledger.write_text(
        "date,coin,rank,snapshot_id,snapshot_path,status,date\n"
        "2026-07-27,KRW-AAA,1,snap-1,output/x.json,closed,2020-01-01\n",
        encoding="utf-8",
    )

    with pytest.raises(gate.CloseInputError, match="identity schema invalid"):
        gate.is_no_decision_day(
            asof="2026-07-27",
            cohort="r1-open",
            output_root=output,
        )


def test_no_decision_ledger_reader_rejects_symlink(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    real = tmp_path / "elsewhere.csv"
    real.write_text("date,status\n", encoding="utf-8")
    (output / "shadow_ledger_recommend.csv").symlink_to(real)

    with pytest.raises(gate.CloseInputError):
        gate.is_no_decision_day(
            asof="2026-07-27",
            cohort="r1-open",
            output_root=output,
        )

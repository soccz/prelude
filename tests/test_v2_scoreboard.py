from __future__ import annotations

import csv
import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from ledger.config import ROUND_TRIP_COST_PP
import scripts.pump_detector_v2_today as v2_runner
import scripts.v2_scoreboard as scoreboard_module
from scripts.v2_scoreboard import build_scoreboard, run_scoreboard


@pytest.fixture(autouse=True)
def _isolate_terminal_verdict(tmp_path, monkeypatch):
    monkeypatch.setattr(
        scoreboard_module,
        "TERMINAL_STATE",
        tmp_path / "radar-terminal.json",
    )


def _row(
    day: str,
    realized: float,
    regime: str = "bear_quiet",
    coin: str = "KRW-TEST",
) -> dict:
    return {
        "date": day,
        "coin": coin,
        "status": "closed",
        "realized_pct": str(realized),
        "btc_regime": regime,
    }


def _write_verified_inputs(
    root: Path,
    rows: list[dict],
) -> tuple[Path, Path, Path]:
    ledger = root / "v2.csv"
    decisions = root / "decisions"
    receipts = root / "receipts"
    ledger_rows: list[dict] = []
    by_day: dict[str, list[dict]] = {}
    for row in rows:
        by_day.setdefault(str(row["date"]), []).append(row)

    for day, day_rows in by_day.items():
        feature_day = date.fromisoformat(day) - timedelta(days=1)
        candidates = []
        for rank, row in enumerate(day_rows, 1):
            candidates.append(
                {
                    "market": row["coin"],
                    "rank": rank,
                    "score": round(0.9 - rank / 100, 4),
                    "entry_open": 100.0 + rank,
                    "roc_7d": 12.0,
                    "roc_7d_rank": 0.95,
                    "atr_pct_14": 0.08,
                    "log_return_1d": 0.01,
                    "b_vol_surge": 3.2,
                    "b_ret_1d": 0.05,
                    "liq_rank_daily": float(rank),
                    "btc_regime": row.get("btc_regime", "bear_quiet"),
                    "rule_id": v2_runner.PUMP_V2_RULE_ID,
                }
            )
        decision = {
            "asof": day,
            "model_id": "pump_hunter_v2",
            "rule_version": "pump_detector_v2",
            "rule": v2_runner.PUMP_V2_RULE,
            "feature_date": f"{feature_day.isoformat()} 09:00:00",
            "btc_regime": day_rows[0].get("btc_regime", "bear_quiet"),
            "universe_n": 100,
            "binance_status": "ok",
            "n_candidates": len(candidates),
            "candidates": candidates,
            "oos": dict(v2_runner.PUMP_V2_OOS),
        }
        decision = v2_runner._with_forward_provenance(decision)
        manifest = {
            "schema": v2_runner.PUMP_V2_DECISION_SCHEMA,
            "asof": day,
            "decision_id": v2_runner._decision_id(decision),
            "decision": decision,
            "recorded_at": f"{day}T00:04:59+00:00",
        }
        manifest = v2_runner._with_outer_integrity(manifest)
        decisions.mkdir(parents=True, exist_ok=True)
        decision_path = decisions / f"{day}.json"
        decision_path.write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
        decision_id = v2_runner._decision_id(decision)
        attempted_at = f"{day}T00:05:00+00:00"
        sent_at = f"{day}T00:05:01+00:00"
        message_digest = hashlib.sha256(b"pump-v2 radar").hexdigest()
        receipt = {
            "schema": v2_runner.PUMP_V2_RECEIPT_SCHEMA,
            "asof": day,
            "decision_id": decision_id,
            "decision": decision,
            "delivery_ok": True,
            "attempted_at": attempted_at,
            "sent_at": sent_at,
            "recorded_at": f"{day}T00:05:02+00:00",
            "error": None,
            "message_sha256": message_digest,
            "chat_id_sha256": hashlib.sha256(b"456").hexdigest(),
            "chunk_count": 1,
            "telegram_messages": [
                {
                    "message_id": 101,
                    "server_date": sent_at,
                    "text_sha256": message_digest,
                }
            ],
        }
        receipt = v2_runner._with_outer_integrity(receipt)
        receipts.mkdir(parents=True, exist_ok=True)
        (receipts / f"{day}.json").write_text(
            json.dumps(receipt),
            encoding="utf-8",
        )
        for candidate, row in zip(candidates, day_rows):
            ledger_rows.append(
                {
                    **row,
                    "rank": candidate["rank"],
                    "score": candidate["score"],
                    "pump_prob": v2_runner.OOS_HIT_PCT / 100.0,
                    "pump_prob_pct": f"{v2_runner.OOS_HIT_PCT:.1f}%",
                    "dump_risk_flag": False,
                    "entry_open": candidate["entry_open"],
                    "sl_pct": v2_runner.SL_PCT,
                    "tp_pct": v2_runner.TP_PCT,
                    "calibration_source": "binance_leadlag_v1_oos",
                    "snapshot_id": decision_id,
                    "snapshot_path": str(decision_path),
                    "decision_completed_at": manifest["recorded_at"],
                    "delivery_ok": True,
                    "sent_at": sent_at,
                    "p_up20": v2_runner.OOS_HIT_PCT / 100.0,
                    "model_id": decision["model_id"],
                    "rule_version": decision["rule_version"],
                    "rule_id": candidate["rule_id"],
                    "feature_date": decision["feature_date"],
                    "liq_rank_daily": candidate["liq_rank_daily"],
                    "roc_7d": candidate["roc_7d"],
                    "roc_7d_rank": candidate["roc_7d_rank"],
                    "atr_pct_14": candidate["atr_pct_14"],
                    "log_return_1d": candidate["log_return_1d"],
                    "b_vol_surge": candidate["b_vol_surge"],
                    "b_ret_1d": candidate["b_ret_1d"],
                }
            )

    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ledger_rows[0]))
        writer.writeheader()
        writer.writerows(ledger_rows)
    return ledger, decisions, receipts


def test_realized_pct_is_not_charged_cost_twice():
    rows = [_row("2026-07-01", 0.3), _row("2026-07-02", 0.1)]

    out = build_scoreboard(rows, today=date(2026, 7, 25))

    assert out["mean_net_pct"] == pytest.approx(0.2)
    assert out["mean_net_pct"] != pytest.approx(0.2 - ROUND_TRIP_COST_PP)
    assert out["cost_already_deducted"] is True
    assert out["round_trip_cost_pp"] == pytest.approx(ROUND_TRIP_COST_PP)


def test_day_equal_metric_is_companion_not_preregistered_replacement():
    rows = [
        _row("2026-07-01", 1.0, coin="KRW-A"),
        _row("2026-07-01", 1.0, coin="KRW-B"),
        _row("2026-07-01", 1.0, coin="KRW-C"),
        _row("2026-07-02", -1.0, "bear_volatile"),
    ]

    out = build_scoreboard(rows, today=date(2026, 7, 25))

    assert out["mean_net_pct"] == pytest.approx(0.5)
    assert out["day_equal_mean_net_pct"] == pytest.approx(0.0)
    assert out["preregistered_metric"] == "per_trade_mean_net_pct"
    assert "diagnostic only" in out["companion_metric_note"]


def test_open_no_data_and_blank_rows_are_ignored():
    rows = [
        _row("2026-07-01", 0.2),
        _row("2026-07-02", 0.4),
        {
            "date": "2026-07-03",
            "coin": "KRW-OPEN",
            "status": "open",
            "realized_pct": "",
        },
        {
            "date": "2026-07-04",
            "coin": "KRW-NO-DATA",
            "status": "no_data",
            "realized_pct": "",
        },
    ]

    out = build_scoreboard(rows, today=date(2026, 7, 25))

    assert out["closed_n"] == 2
    assert out["mean_net_pct"] == pytest.approx(0.3)


def test_verified_no_data_row_is_accepted_and_not_scored(tmp_path):
    output = tmp_path / "scoreboard.json"
    ledger, decisions, receipts = _write_verified_inputs(
        tmp_path,
        [
            {
                "date": "2026-07-27",
                "coin": "KRW-NODATA",
                "status": "no_data",
                "realized_pct": "",
                "btc_regime": "bear_quiet",
            }
        ],
    )

    code, payload = run_scoreboard(
        ledger=ledger,
        output=output,
        today=date(2026, 7, 28),
        decision_root=decisions,
        receipt_root=receipts,
    )

    assert code == 0
    assert payload["status"] == "insufficient_verified_evidence"
    assert payload["closed_n"] == 0


def test_single_negative_closed_trade_triggers_immediate_early_kill():
    out = build_scoreboard(
        [_row("2026-07-01", -0.01)],
        today=date(2026, 7, 25),
    )

    assert out["closed_n"] == 1
    assert out["mean_net_pct"] == pytest.approx(-0.01)
    assert out["ci95"] is None
    assert out["early_kill_breached"] is True
    assert out["status"] == "early_kill"


def test_unknown_regime_does_not_satisfy_two_regime_gate():
    rows = [
        _row("2026-07-01", 1.0, "?"),
        _row("2026-07-02", 1.0, "unknown"),
    ]

    out = build_scoreboard(rows, today=date(2026, 7, 25))

    assert out["regimes"] == []
    assert out["criteria"]["2레짐_or_t>=2"] is False


def test_missing_ledger_replaces_stale_scoreboard_and_fails_closed(tmp_path):
    output = tmp_path / "scoreboard.json"
    output.write_text('{"status": "active", "closed_n": 999}')

    code, payload = run_scoreboard(
        ledger=tmp_path / "missing.csv",
        output=output,
        today=date(2026, 7, 25),
    )

    assert code == 2
    assert payload["status"] == "ledger_missing"
    persisted = json.loads(output.read_text())
    assert persisted["status"] == "ledger_missing"
    assert persisted["closed_n"] == 0


def test_insufficient_sample_is_always_persisted(tmp_path):
    output = tmp_path / "scoreboard.json"
    ledger, decisions, receipts = _write_verified_inputs(
        tmp_path,
        [_row("2026-07-27", 0.2)],
    )
    output.write_text('{"status": "active", "closed_n": 999}')

    code, payload = run_scoreboard(
        ledger=ledger,
        output=output,
        today=date(2026, 7, 28),
        decision_root=decisions,
        receipt_root=receipts,
    )

    assert code == 0
    assert payload["status"] == "insufficient_sample"
    assert json.loads(output.read_text())["closed_n"] == 1


@pytest.mark.parametrize("bad", ["nan", "inf", "not-a-number"])
def test_invalid_closed_return_replaces_stale_output_and_fails_closed(
    tmp_path,
    bad,
):
    ledger = tmp_path / "v2.csv"
    output = tmp_path / "scoreboard.json"
    ledger.write_text(
        "date,coin,status,realized_pct,btc_regime\n"
        f"2026-07-01,KRW-TEST,closed,{bad},bear_quiet\n"
    )

    code, payload = run_scoreboard(
        ledger=ledger,
        output=output,
        today=date(2026, 7, 25),
    )

    assert code == 2
    assert payload["status"] == "ledger_invalid"
    assert json.loads(output.read_text())["status"] == "ledger_invalid"


@pytest.mark.parametrize(
    "bad_row",
    [
        {
            "date": "2026-07-01",
            "coin": "KRW-A",
            "status": "closed",
            "realized_pct": "",
        },
        {
            "date": "2026-07-25",
            "coin": "KRW-A",
            "status": "closed",
            "realized_pct": "1.0",
        },
        {
            "date": "2026-07-01",
            "coin": "KRW-A",
            "status": "unknown",
            "realized_pct": "",
        },
        {
            "date": "2026-07-01",
            "coin": "KRW-A",
            "status": "open",
            "realized_pct": "1.0",
        },
    ],
)
def test_invalid_row_contract_fails_closed(bad_row):
    with pytest.raises(ValueError):
        build_scoreboard([bad_row], today=date(2026, 7, 25))


def test_duplicate_closed_position_fails_closed():
    rows = [
        _row("2026-07-01", 1.0, coin="KRW-A"),
        _row("2026-07-01", -1.0, coin="KRW-A"),
    ]

    with pytest.raises(ValueError, match="duplicate closed position"):
        build_scoreboard(rows, today=date(2026, 7, 25))


@pytest.mark.parametrize(
    "rows",
    [
        [
            {
                "date": "2026-07-01",
                "coin": "KRW-A",
                "status": "open",
                "realized_pct": "",
            },
            {
                "date": "2026-07-01",
                "coin": "KRW-A",
                "status": "open",
                "realized_pct": "",
            },
        ],
        [
            _row("2026-07-01", 1.0, coin="KRW-A"),
            {
                "date": "2026-07-01",
                "coin": "KRW-A",
                "status": "open",
                "realized_pct": "",
            },
        ],
    ],
)
def test_duplicate_position_across_any_status_fails_closed(rows):
    with pytest.raises(ValueError, match="duplicate"):
        build_scoreboard(rows, today=date(2026, 7, 25))


def test_future_open_position_fails_closed():
    with pytest.raises(ValueError, match="after scoreboard asof"):
        build_scoreboard(
            [
                {
                    "date": "2026-07-26",
                    "coin": "KRW-A",
                    "status": "open",
                    "realized_pct": "",
                }
            ],
            today=date(2026, 7, 25),
        )


def test_two_trades_on_one_day_prints_without_missing_t_stat_crash(
    tmp_path, capsys
):
    output = tmp_path / "scoreboard.json"
    ledger, decisions, receipts = _write_verified_inputs(
        tmp_path,
        [
            _row("2026-07-27", 1.0, coin="KRW-A"),
            _row("2026-07-27", 1.0, coin="KRW-B"),
        ],
    )

    code, payload = run_scoreboard(
        ledger=ledger,
        output=output,
        today=date(2026, 7, 28),
        decision_root=decisions,
        receipt_root=receipts,
    )

    assert code == 0
    assert payload["closed_n"] == 2
    assert payload["per_day_t"] is None
    assert "t_day=n/a" in capsys.readouterr().out


@pytest.mark.parametrize(
    "body",
    [
        (
            "date,coin,status,realized_pct,status\n"
            "2026-07-01,KRW-A,closed,1.0,closed\n"
        ),
        (
            "date,coin,status,realized_pct\n"
            "2026-07-01,KRW-A,closed,1.0,unexpected\n"
        ),
    ],
)
def test_malformed_csv_shape_replaces_stale_output_and_fails_closed(
    tmp_path, body
):
    ledger = tmp_path / "v2.csv"
    output = tmp_path / "scoreboard.json"
    ledger.write_text(body, encoding="utf-8")
    output.write_text('{"status":"active"}', encoding="utf-8")

    code, payload = run_scoreboard(
        ledger=ledger,
        output=output,
        today=date(2026, 7, 25),
    )

    assert code == 2
    assert payload["status"] == "ledger_invalid"
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == (
        "ledger_invalid"
    )


def test_fake_closed_row_without_canonical_evidence_fails_closed(tmp_path):
    ledger = tmp_path / "v2.csv"
    output = tmp_path / "scoreboard.json"
    ledger.write_text(
        "date,coin,status,realized_pct,btc_regime\n"
        "2026-07-27,KRW-FAKE,closed,99.0,bear_quiet\n",
        encoding="utf-8",
    )

    code, payload = run_scoreboard(
        ledger=ledger,
        output=output,
        today=date(2026, 7, 28),
        decision_root=tmp_path / "decisions",
        receipt_root=tmp_path / "receipts",
    )

    assert code == 2
    assert payload["status"] == "provenance_invalid"
    assert payload["closed_n"] == 0


def test_scoreboard_rejects_symlinked_ledger_input(tmp_path):
    source = tmp_path / "outside.csv"
    source.write_text(
        "date,coin,status,realized_pct,btc_regime\n"
        "2026-07-01,KRW-LEGACY,closed,1.0,bear_quiet\n",
        encoding="utf-8",
    )
    ledger = tmp_path / "v2.csv"
    ledger.symlink_to(source)
    output = tmp_path / "scoreboard.json"

    code, payload = run_scoreboard(
        ledger=ledger,
        output=output,
        today=date(2026, 7, 28),
        decision_root=tmp_path / "decisions",
        receipt_root=tmp_path / "receipts",
    )

    assert code == 2
    assert payload["status"] == "ledger_invalid"
    assert ledger.is_symlink()
    assert source.read_text(encoding="utf-8").startswith("date,coin")


def test_precontract_legacy_row_is_diagnostic_only(tmp_path):
    ledger = tmp_path / "v2.csv"
    output = tmp_path / "scoreboard.json"
    ledger.write_text(
        "date,coin,status,realized_pct,btc_regime\n"
        "2026-07-01,KRW-LEGACY,closed,1.0,bear_quiet\n",
        encoding="utf-8",
    )

    code, payload = run_scoreboard(
        ledger=ledger,
        output=output,
        today=date(2026, 7, 28),
        decision_root=tmp_path / "decisions",
        receipt_root=tmp_path / "receipts",
    )

    assert code == 0
    assert payload["closed_n"] == 0
    assert payload["mean_net_pct"] is None
    assert payload["status"] == "insufficient_verified_evidence"
    assert payload["terminal_evidence_eligible"] is False
    assert payload["terminal_state"]["status"] == "pending"
    assert payload["legacy_diagnostic"]["closed_n"] == 1
    assert payload["legacy_diagnostic"]["mean_net_pct"] == pytest.approx(1.0)
    assert payload["legacy_diagnostic"]["operational"] is False
    assert payload["evidence_scope"] == (
        "verified_post_contract_closed_positions_only"
    )
    assert payload["provenance"]["healthy_dates"] == []
    assert payload["provenance"]["legacy_scorecard_rows"] == 1
    assert payload["provenance"]["legacy_baseline_verified"] is False


def test_legacy_only_deadline_kills_on_verified_zero_without_scoring_legacy(
    tmp_path,
):
    ledger = tmp_path / "v2.csv"
    output = tmp_path / "scoreboard.json"
    terminal = tmp_path / "terminal.json"
    ledger.write_text(
        "date,coin,status,realized_pct,btc_regime\n"
        "2026-07-01,KRW-LEGACY,closed,-99.0,bear_quiet\n",
        encoding="utf-8",
    )

    code, payload = run_scoreboard(
        ledger=ledger,
        output=output,
        today=date(2026, 9, 1),
        decision_root=tmp_path / "decisions",
        receipt_root=tmp_path / "receipts",
        terminal_state=terminal,
        verdict_recorded_at=datetime(
            2026,
            9,
            1,
            tzinfo=timezone.utc,
        ),
    )

    assert code == 21
    assert payload["status"] == "judgment_kill"
    assert payload["closed_n"] == 0
    assert payload["mean_net_pct"] is None
    assert payload["terminal_state"]["verdict"] == "kill"
    assert payload["legacy_diagnostic"]["mean_net_pct"] == pytest.approx(-99.0)
    assert terminal.exists()


def test_underpowered_verified_deadline_uses_frozen_kill_rule(
    tmp_path,
):
    output = tmp_path / "scoreboard.json"
    terminal = tmp_path / "terminal.json"
    ledger, decisions, receipts = _write_verified_inputs(
        tmp_path,
        [_row("2026-07-27", 1.0)],
    )

    code, payload = run_scoreboard(
        ledger=ledger,
        output=output,
        today=date(2026, 9, 1),
        decision_root=decisions,
        receipt_root=receipts,
        terminal_state=terminal,
        verdict_recorded_at=datetime(
            2026,
            9,
            1,
            tzinfo=timezone.utc,
        ),
    )

    assert code == 21
    assert payload["closed_n"] == 1
    assert payload["status"] == "judgment_kill"
    assert payload["terminal_evidence_eligible"] is True
    assert payload["terminal_state"]["verdict"] == "kill"
    assert payload["terminal_verdict"] == "kill"
    assert terminal.exists()


def test_canonical_legacy_baseline_mismatch_fails_closed(
    tmp_path,
    monkeypatch,
):
    ledger = tmp_path / "v2.csv"
    output = tmp_path / "scoreboard.json"
    ledger.write_text(
        "date,coin,status,realized_pct,btc_regime\n"
        "2026-07-01,KRW-TAMPERED,closed,99.0,bear_quiet\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(scoreboard_module, "LEDGER", ledger)

    code, payload = run_scoreboard(
        ledger=ledger,
        output=output,
        today=date(2026, 7, 28),
        decision_root=tmp_path / "decisions",
        receipt_root=tmp_path / "receipts",
    )

    assert code == 2
    assert payload["status"] == "provenance_invalid"
    assert "legacy v2 scorecard baseline mismatch" in payload["error"]


@pytest.mark.parametrize(
    "tamper",
    [
        "candidate",
        "decision_duplicate_key",
        "decision_nan",
        "decision_and_ledger_recorded_at",
        "receipt_id",
        "receipt_duplicate_key",
        "receipt_and_ledger_sent_at",
        "delivery_failed",
        "decision_recorded_wrong_day",
        "decision_completed_at",
        "receipt_attempted_wrong_day",
    ],
)
def test_closed_row_provenance_mismatch_fails_closed(tmp_path, tamper):
    output = tmp_path / "scoreboard.json"
    ledger, decisions, receipts = _write_verified_inputs(
        tmp_path,
        [_row("2026-07-27", 1.0)],
    )
    if tamper == "candidate":
        body = ledger.read_text(encoding="utf-8").replace(
            "KRW-TEST",
            "KRW-FAKE",
        )
        ledger.write_text(body, encoding="utf-8")
    elif tamper in {"decision_duplicate_key", "decision_nan"}:
        decision_path = decisions / "2026-07-27.json"
        raw = decision_path.read_text(encoding="utf-8")
        if tamper == "decision_duplicate_key":
            raw = raw.replace(
                '"schema": "pump_v2_decision.v2"',
                (
                    '"schema": "pump_v2_decision.v2", '
                    '"schema": "pump_v2_decision.v2"'
                ),
                1,
            )
        else:
            raw = raw.replace('"universe_n": 100', '"universe_n": NaN', 1)
        decision_path.write_text(raw, encoding="utf-8")
    elif tamper == "decision_recorded_wrong_day":
        decision_path = decisions / "2026-07-27.json"
        manifest = json.loads(decision_path.read_text(encoding="utf-8"))
        manifest["recorded_at"] = "2026-07-28T00:05:00+00:00"
        decision_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif tamper == "decision_and_ledger_recorded_at":
        decision_path = decisions / "2026-07-27.json"
        manifest = json.loads(decision_path.read_text(encoding="utf-8"))
        replacement = "2026-07-27T00:05:30+00:00"
        manifest["recorded_at"] = replacement
        decision_path.write_text(json.dumps(manifest), encoding="utf-8")
        ledger.write_text(
            ledger.read_text(encoding="utf-8").replace(
                "2026-07-27T00:04:59+00:00",
                replacement,
            ),
            encoding="utf-8",
        )
    elif tamper == "decision_completed_at":
        body = ledger.read_text(encoding="utf-8").replace(
            "2026-07-27T00:04:59+00:00",
            "2026-07-27T00:04:58+00:00",
        )
        ledger.write_text(body, encoding="utf-8")
    else:
        receipt_path = receipts / "2026-07-27.json"
        if tamper == "receipt_duplicate_key":
            raw = receipt_path.read_text(encoding="utf-8").replace(
                '"delivery_ok": true',
                '"delivery_ok": true, "delivery_ok": true',
                1,
            )
            receipt_path.write_text(raw, encoding="utf-8")
            receipt = None
        else:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if tamper == "receipt_id":
            assert receipt is not None
            receipt["decision_id"] = "pump-v2-wrong"
        elif tamper == "delivery_failed":
            assert receipt is not None
            receipt["delivery_ok"] = False
            receipt["sent_at"] = None
            receipt["error"] = "delivery failed"
        elif tamper == "receipt_and_ledger_sent_at":
            assert receipt is not None
            replacement = "2026-07-27T00:05:00.500000+00:00"
            receipt["sent_at"] = replacement
            ledger.write_text(
                ledger.read_text(encoding="utf-8").replace(
                    "2026-07-27T00:05:01+00:00",
                    replacement,
                ),
                encoding="utf-8",
            )
        elif tamper == "receipt_attempted_wrong_day":
            assert receipt is not None
            receipt["attempted_at"] = "2026-07-28T00:05:00+00:00"
            receipt["sent_at"] = "2026-07-28T00:05:01+00:00"
            receipt["recorded_at"] = "2026-07-28T00:05:02+00:00"
        if receipt is not None:
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    code, payload = run_scoreboard(
        ledger=ledger,
        output=output,
        today=date(2026, 7, 28),
        decision_root=decisions,
        receipt_root=receipts,
    )

    assert code == 2
    assert payload["status"] == "provenance_invalid"

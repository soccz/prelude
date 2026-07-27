from __future__ import annotations

import json
import sqlite3
import threading
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

import ops.policy_competition as competition
import scripts.pump_detector_today as v1_runner
import scripts.pump_detector_v2_today as v2_runner
from ops.policy_competition import (
    POLICY_ARTIFACT_SCHEMA,
    POLICY_INPUT_MANIFEST_SCHEMA,
    POLICY_ROW_COLUMNS,
    POLICIES,
    PolicyArtifactError,
    PolicySpec,
    _atomic_write_json,
    _best_participant,
    _consensus_rows,
    _evaluate_rows,
    _finalize_policy_payload,
    _policy_config,
    _policy_csv_bytes,
    _selected_pump20_hits,
    _source_observed_dates,
    _write_sqlite,
    load_policy_artifact,
    _true_series,
)
from ops.artifact_provenance import atomic_write_bytes, with_manifest_digest
from signals.model_registry import MetricSource, ModelSpec


def _artifact_payload(
    rows: list[dict] | None = None,
    *,
    asof: str = "2026-06-04",
) -> dict:
    config = _policy_config()
    manifest = with_manifest_digest(
        {
            "schema": POLICY_INPUT_MANIFEST_SCHEMA,
            "asof": asof,
            "files": {},
            "decision_roots": {},
            "generator_sources": {},
            "contract": {"config": config},
        }
    )
    return _finalize_policy_payload(
        {
            "asof": asof,
            "generated_at_utc": f"{asof}T00:00:00+00:00",
            "input_manifest": manifest,
            "config": config,
            "radar_terminal": {"status": "not_due"},
            "rows": rows or [],
            "exit_lab": [],
        }
    )


def _complete_policy_row(**updates) -> dict:
    row = {
        "asof": "2026-06-04",
        "participant_id": "r1:top_all",
        "source_id": "r1",
        "policy_id": "top_all",
        "objective": "baseline",
        "description": "all",
        "n_closed": 3,
        "n_days": 1,
        "n_selected_days": 1,
        "net_sum_pct": 1.0,
        "net_mean_pct": 0.3333,
        "deep_loss_freq_pct": 0.0,
        "sl_rate_pct": 33.3333,
        "tp_rate_pct": 33.3333,
        "pump20_precision_pct": 50.0,
        "post_send_pump20_precision_pct": None,
        "post_send_label_n": 0,
        "pump20_recall_pct": 25.0,
        "pump20_captured": 1,
        "pump20_actual": 4,
        "pump_days": 1,
        "pump_days_any_captured": 1,
        "pump_day_capture_rate_pct": 100.0,
        "recall_date_basis": "ledger_rows",
        "pump20_label_basis": "full_day_D1_high_over_open_proxy",
    }
    row.update(updates)
    assert set(row) == set(POLICY_ROW_COLUMNS)
    return row


def test_policy_competition_reports_pump_recall_not_just_precision():
    rows = pd.DataFrame([
        {
            "date": pd.Timestamp("2026-06-01"),
            "coin": "KRW-AAA",
            "net_pct": 4.85,
            "selected_pump20_hit": 1,
            "exit_reason": "TP",
        },
        {
            "date": pd.Timestamp("2026-06-01"),
            "coin": "KRW-BBB",
            "net_pct": -3.15,
            "selected_pump20_hit": 0,
            "exit_reason": "SL",
        },
        {
            "date": pd.Timestamp("2026-06-02"),
            "coin": "KRW-DDD",
            "net_pct": 0.5,
            "selected_pump20_hit": 1,
            "exit_reason": "EOD",
        },
    ])
    actual_pumps = {
        pd.Timestamp("2026-06-01"): {"KRW-AAA", "KRW-CCC"},
        pd.Timestamp("2026-06-02"): {"KRW-DDD"},
    }

    metric = _evaluate_rows(
        rows,
        participant_id="test:top_all",
        source_id="test",
        policy=PolicySpec("top_all", "all", "pump_recall", _true_series),
        asof=pd.Timestamp("2026-06-03"),
        actual_pumps=actual_pumps,
    )

    assert metric["pump20_precision_pct"] == 66.6667
    assert metric["pump20_recall_pct"] == 66.6667
    assert metric["pump20_captured"] == 2
    assert metric["pump20_actual"] == 3
    assert metric["pump_day_capture_rate_pct"] == 100.0
    assert metric["net_mean_pct"] == 0.7333
    assert metric["sl_rate_pct"] == 33.3333


def test_post_send_precision_is_separate_from_full_day_discovery_proxy():
    rows = pd.DataFrame([
        {
            "date": pd.Timestamp("2026-06-01"),
            "coin": "KRW-AAA",
            "net_pct": 1.0,
            "selected_pump20_hit": 1,
            "post_send_pump20_hit": 0,
            "exit_reason": "TP",
        },
        {
            "date": pd.Timestamp("2026-06-01"),
            "coin": "KRW-BBB",
            "net_pct": -1.0,
            "selected_pump20_hit": 1,
            "post_send_pump20_hit": 1,
            "exit_reason": "SL",
        },
    ])

    metric = _evaluate_rows(
        rows,
        participant_id="test:top_all",
        source_id="test",
        policy=PolicySpec("top_all", "all", "pump_recall", _true_series),
        asof=pd.Timestamp("2026-06-03"),
        actual_pumps={pd.Timestamp("2026-06-01"): {"KRW-AAA", "KRW-BBB"}},
    )

    assert metric["pump20_precision_pct"] == 100.0
    assert metric["post_send_pump20_precision_pct"] == 50.0
    assert metric["post_send_label_n"] == 2
    assert "not post-send" in metric["pump20_label_basis"]


def test_selected_pump_hits_are_loaded_in_one_batched_query(tmp_path, monkeypatch):
    db_path = tmp_path / "candles.db"
    con = sqlite3.connect(db_path)
    con.execute(
        "CREATE TABLE candles "
        "(timestamp TEXT NOT NULL, market TEXT NOT NULL, open REAL, high REAL)"
    )
    con.executemany(
        "INSERT INTO candles VALUES (?, ?, ?, ?)",
        [
            ("2026-06-01 09:00:00", "KRW-A", 100.0, 121.0),
            ("2026-06-01 09:00:00", "KRW-B", 100.0, 119.0),
            ("2026-06-02 09:00:00", "KRW-C", 0.0, 200.0),
        ],
    )
    con.commit()
    con.close()

    statements = []
    real_connect = sqlite3.connect

    def tracked_connect(*args, **kwargs):
        tracked = real_connect(*args, **kwargs)
        tracked.set_trace_callback(statements.append)
        return tracked

    monkeypatch.setattr(competition.sqlite3, "connect", tracked_connect)
    result = _selected_pump20_hits(
        pd.DataFrame(
            {
                "date": [
                    pd.Timestamp("2026-06-01"),
                    pd.Timestamp("2026-06-01"),
                    pd.Timestamp("2026-06-02"),
                    pd.Timestamp("2026-06-03"),
                ],
                "coin": ["KRW-A", "KRW-B", "KRW-C", "KRW-MISSING"],
            }
        ),
        db_path,
    )

    assert result["selected_pump20_hit"].tolist()[:2] == [1, 0]
    assert result["selected_pump20_hit"].isna().tolist()[2:] == [True, True]
    assert sum("WITH requested" in statement for statement in statements) == 1


def test_policy_recall_counts_evaluation_day_with_zero_selected_rows():
    rows = pd.DataFrame([
        {
            "date": pd.Timestamp("2026-06-01"),
            "coin": "KRW-AAA",
            "net_pct": 4.85,
            "selected_pump20_hit": 1,
            "exit_reason": "TP",
        },
    ])
    actual_pumps = {
        pd.Timestamp("2026-06-01"): {"KRW-AAA"},
        pd.Timestamp("2026-06-02"): {"KRW-BBB"},
    }

    metric = _evaluate_rows(
        rows,
        participant_id="test:strict",
        source_id="test",
        policy=PolicySpec("strict", "strict", "pump_recall", _true_series),
        asof=pd.Timestamp("2026-06-03"),
        actual_pumps=actual_pumps,
        evaluation_dates=[
            pd.Timestamp("2026-06-01"),
            pd.Timestamp("2026-06-02"),
        ],
    )

    assert metric["pump20_recall_pct"] == 50.0
    assert metric["pump_days"] == 2
    assert metric["pump_days_any_captured"] == 1
    assert metric["n_days"] == 2
    assert metric["n_selected_days"] == 1


def test_no_dump_policy_filters_flagged_rows():
    rows = pd.DataFrame([
        {"rank": 1, "dump_risk_flag": False},
        {"rank": 2, "dump_risk_flag": True},
        {"rank": 3, "dump_risk_flag": "False"},
        {"rank": 4, "dump_risk_flag": False},
        {"rank": 2, "dump_risk_flag": None},
        {"rank": 3, "dump_risk_flag": "unknown"},
    ])
    no_dump = next(p for p in POLICIES if p.policy_id == "no_dump_top3")

    mask = no_dump.predicate(rows)

    assert mask.tolist() == [True, False, True, False, False, False]


def test_policy_predicates_reject_invalid_numeric_boundaries():
    rows = pd.DataFrame([
        {"rank": 0, "rr_ratio": float("inf"), "p_up20": float("inf")},
        {"rank": 1.5, "rr_ratio": 0.75, "p_up20": 0.03},
        {"rank": 2, "rr_ratio": 0.80, "p_up20": 1.01},
        {"rank": 1, "rr_ratio": 0.80, "p_up20": 0.04},
    ])
    top2 = next(p for p in POLICIES if p.policy_id == "top2_only")
    rr = next(p for p in POLICIES if p.policy_id == "rr_ge_0_75")
    pump = next(
        p for p in POLICIES if p.policy_id == "pump_prob_ge_3pct"
    )

    assert top2.predicate(rows).tolist() == [False, False, True, True]
    assert rr.predicate(rows).tolist() == [False, True, True, True]
    assert pump.predicate(rows).tolist() == [False, True, False, True]


def test_consensus_rows_require_two_sources_for_same_coin_and_date():
    base = {
        "date": pd.Timestamp("2026-06-01"),
        "net_pct": 1.0,
        "selected_pump20_hit": 0,
        "rank": 1,
    }
    r1 = pd.DataFrame([
        {**base, "coin": "KRW-AAA", "model_id": "recommend_r1_open"},
        {**base, "coin": "KRW-BBB", "model_id": "recommend_r1_open", "rank": 2},
    ])
    r2 = pd.DataFrame([
        {**base, "coin": "KRW-AAA", "model_id": "recommend_r2_open"},
        {**base, "coin": "KRW-CCC", "model_id": "recommend_r2_open", "rank": 2},
    ])
    a1 = pd.DataFrame([
        {**base, "coin": "KRW-DDD", "model_id": "recommend_r1_sustain_open"},
    ])

    out = _consensus_rows({
        "recommend_r1_open": r1,
        "recommend_r2_open": r2,
        "recommend_r1_sustain_open": a1,
    })

    assert out["coin"].tolist() == ["KRW-AAA"]
    assert out.iloc[0]["source_count"] == 2
    assert out.iloc[0]["model_id"] == "consensus_2of3"


def test_policy_competition_persists_sqlite_snapshot(tmp_path):
    db = tmp_path / "policy_competition.db"
    payload = _artifact_payload([_complete_policy_row()])

    _write_sqlite(payload, db)

    import sqlite3
    con = sqlite3.connect(db)
    try:
        run = con.execute(
            "SELECT row_count, best_pump_participant, best_net_participant "
            "FROM policy_competition_runs WHERE asof='2026-06-04'"
        ).fetchone()
        rows = con.execute("SELECT COUNT(*) FROM policy_competition_latest_rows").fetchone()[0]
    finally:
        con.close()

    assert run == (1, "r1:top_all", "r1:top_all")
    assert rows == 1


def test_zero_metric_beats_negative_metric_when_selecting_best():
    rows = [
        {
            "participant_id": "zero",
            "n_closed": 1,
            "pump20_recall_pct": 0.0,
            "net_mean_pct": 0.0,
        },
        {
            "participant_id": "negative",
            "n_closed": 100,
            "pump20_recall_pct": -1.0,
            "net_mean_pct": 100.0,
        },
    ]

    assert _best_participant(rows, "pump20_recall_pct") == "zero"


def test_sqlite_legacy_schema_is_migrated_with_new_audit_columns(tmp_path):
    db = tmp_path / "legacy.db"
    con = sqlite3.connect(db)
    try:
        con.execute(
            """
            CREATE TABLE policy_competition_rows (
                asof TEXT NOT NULL, participant_id TEXT NOT NULL,
                source_id TEXT NOT NULL, policy_id TEXT NOT NULL,
                objective TEXT NOT NULL, description TEXT,
                n_closed INTEGER NOT NULL, n_days INTEGER NOT NULL,
                net_sum_pct REAL, net_mean_pct REAL,
                deep_loss_freq_pct REAL, sl_rate_pct REAL, tp_rate_pct REAL,
                pump20_precision_pct REAL, pump20_recall_pct REAL,
                pump20_captured INTEGER NOT NULL, pump20_actual INTEGER NOT NULL,
                pump_days INTEGER NOT NULL,
                pump_days_any_captured INTEGER NOT NULL,
                pump_day_capture_rate_pct REAL,
                generated_at_utc TEXT NOT NULL,
                PRIMARY KEY (asof, participant_id)
            )
            """
        )
        con.commit()
    finally:
        con.close()

    payload = _artifact_payload()
    _write_sqlite(payload, db)

    con = sqlite3.connect(db)
    try:
        columns = {
            row[1]
            for row in con.execute(
                "PRAGMA table_info(policy_competition_rows)"
            )
        }
    finally:
        con.close()
    assert {
        "n_selected_days",
        "post_send_pump20_precision_pct",
        "post_send_label_n",
        "recall_date_basis",
        "run_id",
        "payload_sha256",
        "pump20_label_basis",
    } <= columns


def test_policy_triplet_loader_rejects_any_cross_artifact_mismatch(tmp_path):
    json_path = tmp_path / "policy.json"
    csv_path = tmp_path / "policy.csv"
    db_path = tmp_path / "policy.db"
    payload = _artifact_payload([_complete_policy_row()])
    _atomic_write_json(json_path, payload)
    atomic_write_bytes(csv_path, _policy_csv_bytes(payload))
    _write_sqlite(payload, db_path)

    loaded = load_policy_artifact(
        json_path,
        csv_path=csv_path,
        db_path=db_path,
        require_current=False,
    )
    assert loaded["schema"] == POLICY_ARTIFACT_SCHEMA
    assert loaded["run_id"] == payload["run_id"]

    csv_path.write_bytes(csv_path.read_bytes().replace(b"r1:top_all", b"r1:tampered"))
    with pytest.raises(PolicyArtifactError, match="JSON/CSV"):
        load_policy_artifact(
            json_path,
            csv_path=csv_path,
            db_path=db_path,
            require_current=False,
        )

    atomic_write_bytes(csv_path, _policy_csv_bytes(payload))
    with sqlite3.connect(db_path) as con:
        con.execute(
            "UPDATE policy_competition_runs SET run_id='wrong' WHERE asof=?",
            (payload["asof"],),
        )
    with pytest.raises(PolicyArtifactError, match="JSON/SQLite"):
        load_policy_artifact(
            json_path,
            csv_path=csv_path,
            db_path=db_path,
            require_current=False,
        )


@pytest.mark.parametrize("malformation", ["duplicate", "nan"])
def test_policy_loader_rejects_noncanonical_json(
    malformation,
    tmp_path,
):
    json_path = tmp_path / "policy.json"
    csv_path = tmp_path / "policy.csv"
    db_path = tmp_path / "policy.db"
    payload = _artifact_payload()
    _atomic_write_json(json_path, payload)
    atomic_write_bytes(csv_path, _policy_csv_bytes(payload))
    _write_sqlite(payload, db_path)
    raw = json_path.read_text(encoding="utf-8")
    if malformation == "duplicate":
        raw = raw.replace(
            f'"schema": "{POLICY_ARTIFACT_SCHEMA}"',
            f'"schema": "{POLICY_ARTIFACT_SCHEMA}", '
            f'"schema": "{POLICY_ARTIFACT_SCHEMA}"',
            1,
        )
    else:
        raw = raw.replace(
            '"pump20_threshold": 0.2',
            '"pump20_threshold": NaN',
            1,
        )
    json_path.write_text(raw, encoding="utf-8")

    with pytest.raises(PolicyArtifactError):
        load_policy_artifact(
            json_path,
            csv_path=csv_path,
            db_path=db_path,
            require_current=False,
        )


def test_policy_loader_fails_closed_after_input_bytes_change(
    monkeypatch,
    tmp_path,
):
    ledger = tmp_path / "ledger.csv"
    ledger.write_text("date,status,coin,realized_pct\n", encoding="utf-8")
    spec = ModelSpec(
        id="test_model",
        name="test",
        ledger_path=str(ledger),
        slots=["open"],
        metric=MetricSource(
            status_col="status",
            closed_value="closed",
            date_col="date",
            realized_pct_col="realized_pct",
        ),
        predict_ref="signals.recommend:score_candidates",
    )
    monkeypatch.setattr(competition, "MODELS", [spec])
    candle_db = tmp_path / "d1.db"
    candle_db.write_bytes(b"db-v1")
    roots = {
        "snapshot_root": tmp_path / "snapshots",
        "pump_v1_decision_root": tmp_path / "v1",
        "pump_v2_decision_root": tmp_path / "v2",
        "pump_v2_receipt_root": tmp_path / "receipts",
    }
    radar = tmp_path / "radar.json"
    manifest = competition.build_policy_input_manifest(
        "2026-06-04",
        candle_db=candle_db,
        radar_verdict_path=radar,
        **roots,
    )
    payload = competition._finalize_policy_payload(
        {
            "asof": "2026-06-04",
            "generated_at_utc": "2026-06-04T00:00:00+00:00",
            "input_manifest": manifest,
            "config": competition._policy_config(),
            "radar_terminal": {"status": "not_due"},
            "rows": [],
            "exit_lab": [],
        }
    )
    json_path = tmp_path / "policy.json"
    csv_path = tmp_path / "policy.csv"
    db_path = tmp_path / "policy.db"
    competition._atomic_write_json(json_path, payload)
    atomic_write_bytes(csv_path, competition._policy_csv_bytes(payload))
    competition._write_sqlite(payload, db_path)

    assert load_policy_artifact(
        json_path,
        csv_path=csv_path,
        db_path=db_path,
        candle_db=candle_db,
        radar_verdict_path=radar,
        **roots,
    )["run_id"] == payload["run_id"]

    ledger.write_text(
        "date,status,coin,realized_pct\n2026-06-01,closed,KRW-A,1\n",
        encoding="utf-8",
    )
    with pytest.raises(PolicyArtifactError, match="input bytes"):
        load_policy_artifact(
            json_path,
            csv_path=csv_path,
            db_path=db_path,
            candle_db=candle_db,
            radar_verdict_path=radar,
            **roots,
        )


def test_policy_run_aborts_before_publish_when_ledger_mutates_mid_compute(
    monkeypatch,
    tmp_path,
):
    ledger = tmp_path / "ledger.csv"
    header = "date,status,coin,realized_pct\n"
    ledger.write_text(header, encoding="utf-8")
    spec = ModelSpec(
        id="test_model",
        name="test",
        ledger_path=str(ledger),
        slots=["open"],
        metric=MetricSource(
            status_col="status",
            closed_value="closed",
            date_col="date",
            realized_pct_col="realized_pct",
        ),
        predict_ref="signals.recommend:score_candidates",
    )
    monkeypatch.setattr(competition, "MODELS", [spec])
    original_load = competition._load_model_rows

    def load_then_mutate(*args, **kwargs):
        rows = original_load(*args, **kwargs)
        ledger.write_text(
            header + "2026-06-04,open,KRW-A,\n",
            encoding="utf-8",
        )
        return rows

    monkeypatch.setattr(
        competition,
        "_load_model_rows",
        load_then_mutate,
    )
    candle_db = tmp_path / "d1.db"
    candle_db.write_bytes(b"stable-input")
    output_json = tmp_path / "policy.json"
    output_csv = tmp_path / "policy.csv"
    output_db = tmp_path / "policy.db"

    with pytest.raises(
        RuntimeError,
        match="changed while the artifact was being computed",
    ):
        competition.run(
            pd.Timestamp("2026-06-04"),
            output_csv=output_csv,
            output_json=output_json,
            db_path=output_db,
            candle_db=candle_db,
            snapshot_root=tmp_path / "snapshots",
            pump_v1_decision_root=tmp_path / "v1",
            pump_v2_decision_root=tmp_path / "v2",
            pump_v2_receipt_root=tmp_path / "receipts",
            radar_verdict_path=tmp_path / "radar.json",
        )

    assert not output_json.exists()
    assert not output_csv.exists()
    assert not output_db.exists()


def test_policy_reader_cannot_observe_csv_json_sqlite_mid_publish(
    monkeypatch,
    tmp_path,
):
    json_path = tmp_path / "policy.json"
    csv_path = tmp_path / "policy.csv"
    db_path = tmp_path / "policy.db"
    previous = _artifact_payload([_complete_policy_row()])
    _atomic_write_json(json_path, previous)
    atomic_write_bytes(csv_path, _policy_csv_bytes(previous))
    _write_sqlite(previous, db_path)

    monkeypatch.setattr(competition, "MODELS", [])
    candle_db = tmp_path / "d1.db"
    candle_db.write_bytes(b"stable-input")
    roots = {
        "snapshot_root": tmp_path / "snapshots",
        "pump_v1_decision_root": tmp_path / "v1",
        "pump_v2_decision_root": tmp_path / "v2",
        "pump_v2_receipt_root": tmp_path / "receipts",
        "radar_verdict_path": tmp_path / "radar.json",
    }
    original_write_sqlite = competition._write_sqlite
    sqlite_stage_entered = threading.Event()
    allow_sqlite_finish = threading.Event()
    reader_done = threading.Event()
    writer_results: list[dict] = []
    reader_results: list[dict] = []
    errors: list[BaseException] = []

    def pause_before_sqlite(payload, destination):
        sqlite_stage_entered.set()
        if not allow_sqlite_finish.wait(timeout=5):
            raise RuntimeError("test timed out before SQLite release")
        original_write_sqlite(payload, destination)

    monkeypatch.setattr(
        competition,
        "_write_sqlite",
        pause_before_sqlite,
    )

    def writer():
        try:
            writer_results.append(
                competition.run(
                    pd.Timestamp("2026-06-04"),
                    output_csv=csv_path,
                    output_json=json_path,
                    db_path=db_path,
                    candle_db=candle_db,
                    **roots,
                )
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    def reader():
        try:
            reader_results.append(
                load_policy_artifact(
                    json_path,
                    csv_path=csv_path,
                    db_path=db_path,
                    require_current=False,
                )
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)
        finally:
            reader_done.set()

    writer_thread = threading.Thread(target=writer)
    writer_thread.start()
    assert sqlite_stage_entered.wait(timeout=5)

    reader_thread = threading.Thread(target=reader)
    reader_thread.start()
    assert not reader_done.wait(timeout=0.2)

    allow_sqlite_finish.set()
    writer_thread.join(timeout=5)
    reader_thread.join(timeout=5)

    assert not writer_thread.is_alive()
    assert not reader_thread.is_alive()
    assert errors == []
    assert reader_results[0]["run_id"] == writer_results[0]["run_id"]
    assert reader_results[0]["rows"] == []


def test_source_dates_union_ledger_and_validated_snapshot_but_exclude_asof_future(
    monkeypatch,
    tmp_path,
):
    ledger = tmp_path / "ledger.csv"
    ledger.write_text(
        "date,status,realized_pct,coin\n"
        "2026-06-01,closed,1.0,KRW-A\n"
        "2026-06-03,open,,KRW-B\n"
        "2026-06-04,open,,KRW-C\n"
    )
    spec = ModelSpec(
        id="recommend_r1_open",
        name="test",
        ledger_path=str(ledger),
        slots=["open"],
        metric=MetricSource(
            status_col="status",
            closed_value="closed",
            date_col="date",
            realized_pct_col="realized_pct",
        ),
        predict_ref="signals.recommend:score_candidates",
    )
    snapshots = tmp_path / "snapshots"
    historical = snapshots / "2026-06-02" / "open_r1.json"
    current = snapshots / "2026-06-03" / "open_r1.json"
    future = snapshots / "2026-06-04" / "open_r1.json"
    for path in (historical, current, future):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    loaded: list[Path] = []

    def fake_load(path, **_kwargs):
        loaded.append(Path(path))
        return {"asof": Path(path).parent.name}

    monkeypatch.setattr(competition, "load_snapshot", fake_load)

    dates, basis = _source_observed_dates(
        spec,
        pd.Timestamp("2026-06-03"),
        snapshot_root=snapshots,
    )

    assert dates == [
        pd.Timestamp("2026-06-01"),
        pd.Timestamp("2026-06-02"),
    ]
    assert basis == "ledger_rows+validated_score_snapshots"
    assert loaded == [historical]


def _write_pump_v2_decision(
    root: Path,
    day: str,
    *,
    binance_status: str = "ok",
    candidates: list[dict] | None = None,
) -> Path:
    candidate_rows = []
    for rank, partial in enumerate(candidates or [], 1):
        candidate_rows.append({
            "market": partial.get("market", f"KRW-T{rank}"),
            "rank": rank,
            "score": partial.get("score", 0.9 - rank / 100),
            "entry_open": partial.get("entry_open", 1000.0 + rank),
            "roc_7d": partial.get("roc_7d", 0.15),
            "roc_7d_rank": partial.get("roc_7d_rank", 0.99),
            "atr_pct_14": partial.get("atr_pct_14", 0.04),
            "log_return_1d": partial.get("log_return_1d", 0.01),
            "b_vol_surge": partial.get("b_vol_surge", 3.0),
            "b_ret_1d": partial.get("b_ret_1d", 0.02),
            "liq_rank_daily": partial.get("liq_rank_daily", rank),
            "btc_regime": partial.get("btc_regime", "bull_quiet"),
            "rule_id": v2_runner.PUMP_V2_RULE_ID,
        })
    feature_day = (
        pd.Timestamp(day) - pd.Timedelta(days=1)
    ).strftime("%Y-%m-%dT09:00:00")
    decision = {
        "asof": day,
        "model_id": "pump_hunter_v2",
        "rule_version": "pump_detector_v2",
        "rule": v2_runner.PUMP_V2_RULE,
        "universe_n": 100,
        "binance_status": binance_status,
        "feature_date": feature_day,
        "btc_regime": "bull_quiet",
        "n_candidates": len(candidate_rows),
        "candidates": candidate_rows,
        "oos": dict(v2_runner.PUMP_V2_OOS),
    }
    decision = v2_runner._with_forward_provenance(decision)
    payload = {
        "schema": v2_runner.PUMP_V2_DECISION_SCHEMA,
        "asof": day,
        "decision_id": v2_runner._decision_id(decision),
        "decision": decision,
        "recorded_at": f"{day}T00:04:59+00:00",
    }
    if date.fromisoformat(day) >= v2_runner.FORWARD_EVIDENCE_ACTIVATION_DATE:
        payload = v2_runner._with_outer_integrity(payload)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{day}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_pump_v1_decision(
    root: Path,
    day: str,
    *,
    candidates: list[str] | None = None,
    forward_provenance: bool = True,
) -> Path:
    candidate_rows = [
        {
            "market": market,
            "rank": rank,
            "score": 0.9 - rank / 100,
            "entry_open": 1000.0 + rank,
            "roc_7d_rank": 0.9,
            "btc_regime": "bull_quiet",
            "rule_id": "roc7_rank_pump20",
        }
        for rank, market in enumerate(candidates or [], 1)
    ]
    decision = {
        "asof": day,
        "feature_date": str(
            (pd.Timestamp(day) - pd.Timedelta(days=1)).date()
        ),
        "model_id": "pump_hunter",
        "rule_version": "pump_detector_v1",
        "top_universe": 100,
        "universe_n": 100,
        "n_candidates": len(candidate_rows),
        "rules": {"pump20": "rule-20", "pump15": "rule-15"},
        "candidates": candidate_rows,
    }
    if forward_provenance:
        decision = v1_runner._with_forward_provenance(decision)
    payload = {
        "schema": (
            v1_runner.PUMP_V1_DECISION_SCHEMA
            if forward_provenance
            else v1_runner.PUMP_V1_LEGACY_DECISION_SCHEMA
        ),
        "asof": day,
        "decision_id": v1_runner._decision_id(decision),
        "decision": decision,
        "recorded_at": f"{day}T00:05:00+00:00",
    }
    if date.fromisoformat(day) >= v1_runner.FORWARD_EVIDENCE_ACTIVATION_DATE:
        payload = v1_runner._with_outer_integrity(payload)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{day}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_pump_v1_zero_selection_manifest_enters_recall_denominator(
    tmp_path,
):
    legacy_ledger = tmp_path / "v1.csv"
    legacy_ledger.write_text(
        "date,status,realized_pct,coin\n"
        "2026-07-25,closed,99.0,KRW-LEGACY\n",
        encoding="utf-8",
    )
    spec = ModelSpec(
        id="pump_hunter",
        name="test",
        ledger_path=str(legacy_ledger),
        slots=["open"],
        metric=MetricSource(
            status_col="status",
            closed_value="closed",
            date_col="date",
            realized_pct_col="realized_pct",
        ),
        predict_ref="signals.pump_detector_v1:score_pump_candidates",
    )
    decisions = tmp_path / "decisions"
    _write_pump_v1_decision(
        decisions,
        "2026-07-26",
        forward_provenance=False,
    )
    _write_pump_v1_decision(decisions, "2026-07-27")

    dates, basis = _source_observed_dates(
        spec,
        pd.Timestamp("2026-07-28"),
        pump_v1_decision_root=decisions,
    )

    assert dates == [pd.Timestamp("2026-07-27")]
    assert basis == (
        "post_contract_validated_forward_decisions;"
        "legacy_dates_excluded"
    )


def test_pump_v1_post_contract_manifest_without_provenance_fails_closed(
    tmp_path,
):
    spec = ModelSpec(
        id="pump_hunter",
        name="test",
        ledger_path=str(tmp_path / "missing.csv"),
        slots=["open"],
        metric=MetricSource(
            status_col="status",
            closed_value="closed",
            date_col="date",
            realized_pct_col="realized_pct",
        ),
        predict_ref="signals.pump_detector_v1:score_pump_candidates",
    )
    decisions = tmp_path / "decisions"
    _write_pump_v1_decision(
        decisions,
        "2026-07-27",
        forward_provenance=False,
    )

    with pytest.raises(
        ValueError,
        match="unsupported pump v1 decision schema",
    ):
        _source_observed_dates(
            spec,
            pd.Timestamp("2026-07-28"),
            pump_v1_decision_root=decisions,
        )


@pytest.mark.parametrize(
    "corruption",
    ["duplicate_top", "duplicate_nested", "nan", "symlink"],
)
def test_pump_v1_manifest_reader_is_strict_and_non_symlink(
    tmp_path,
    corruption,
):
    spec = ModelSpec(
        id="pump_hunter",
        name="test",
        ledger_path=str(tmp_path / "missing.csv"),
        slots=["open"],
        metric=MetricSource(
            status_col="status",
            closed_value="closed",
            date_col="date",
            realized_pct_col="realized_pct",
        ),
        predict_ref="signals.pump_detector_v1:score_pump_candidates",
    )
    decisions = tmp_path / "decisions"
    manifest = _write_pump_v1_decision(decisions, "2026-07-27")
    raw = manifest.read_text(encoding="utf-8")
    if corruption == "duplicate_top":
        raw = raw.replace(
            '"asof": "2026-07-27"',
            '"asof": "2026-07-27", "asof": "2026-07-27"',
            1,
        )
        manifest.write_text(raw, encoding="utf-8")
    elif corruption == "duplicate_nested":
        raw = raw.replace(
            '"model_id": "pump_hunter"',
            '"model_id": "pump_hunter", "model_id": "pump_hunter"',
            1,
        )
        manifest.write_text(raw, encoding="utf-8")
    elif corruption == "nan":
        raw = raw.replace('"universe_n": 100', '"universe_n": NaN', 1)
        manifest.write_text(raw, encoding="utf-8")
    else:
        outside = tmp_path / "outside.json"
        outside.write_text(raw, encoding="utf-8")
        manifest.unlink()
        manifest.symlink_to(outside)

    with pytest.raises(ValueError, match="invalid decision manifest"):
        _source_observed_dates(
            spec,
            pd.Timestamp("2026-07-28"),
            pump_v1_decision_root=decisions,
        )


def test_actual_pump_denominator_excludes_signal_markets(tmp_path):
    db = tmp_path / "d1.db"
    con = sqlite3.connect(db)
    try:
        con.execute(
            "CREATE TABLE candles ("
            "market TEXT, timestamp TEXT, open REAL, high REAL)"
        )
        con.executemany(
            "INSERT INTO candles VALUES (?, ?, ?, ?)",
            [
                ("KRW-USDT", "2026-07-27 09:00:00", 100.0, 130.0),
                ("KRW-IP", "2026-07-27 09:00:00", 100.0, 140.0),
                ("KRW-REAL", "2026-07-27 09:00:00", 100.0, 125.0),
                ("KRW-FLAT", "2026-07-27 09:00:00", 100.0, 110.0),
                ("BTC-NONKRW", "2026-07-27 09:00:00", 100.0, 150.0),
            ],
        )
        con.commit()
    finally:
        con.close()

    actual = competition._actual_pumps_for_dates(
        [pd.Timestamp("2026-07-27")],
        db_path=db,
    )

    assert actual == {
        pd.Timestamp("2026-07-27"): {"KRW-REAL"},
    }


def test_actual_pump_denominator_batches_observed_dates(
    tmp_path,
    monkeypatch,
):
    db = tmp_path / "d1.db"
    con = sqlite3.connect(db)
    try:
        con.execute(
            "CREATE TABLE candles ("
            "market TEXT, timestamp TEXT, open REAL, high REAL)"
        )
        con.executemany(
            "INSERT INTO candles VALUES (?, ?, ?, ?)",
            [
                ("KRW-A", "2026-07-27 09:00:00", 100.0, 125.0),
                ("KRW-B", "2026-07-28 09:00:00", 100.0, 110.0),
            ],
        )
        con.commit()
    finally:
        con.close()

    statements = []
    real_connect = sqlite3.connect

    def tracked_connect(*args, **kwargs):
        tracked = real_connect(*args, **kwargs)
        tracked.set_trace_callback(statements.append)
        return tracked

    monkeypatch.setattr(competition.sqlite3, "connect", tracked_connect)
    actual = competition._actual_pumps_for_dates(
        [
            pd.Timestamp("2026-07-27"),
            pd.Timestamp("2026-07-28"),
            pd.Timestamp("2026-07-29"),
        ],
        db_path=db,
    )

    assert actual == {
        pd.Timestamp("2026-07-27"): {"KRW-A"},
        pd.Timestamp("2026-07-28"): set(),
        pd.Timestamp("2026-07-29"): set(),
    }
    assert sum(
        statement.startswith("SELECT timestamp, market")
        for statement in statements
    ) == 1


def test_pump_v2_zero_selection_manifest_enters_recall_denominator(tmp_path):
    spec = ModelSpec(
        id="pump_hunter_v2",
        name="test",
        ledger_path=str(tmp_path / "missing.csv"),
        slots=["open"],
        metric=MetricSource(
            status_col="status",
            closed_value="closed",
            date_col="date",
            realized_pct_col="realized_pct",
        ),
        predict_ref="signals.pump_detector_v2:score_pump_v2_candidates",
    )
    decisions = tmp_path / "decisions"
    _write_pump_v2_decision(decisions, "2026-07-27")
    # Current/future artifacts aren't inputs to a completed-date evaluation,
    # even if corrupt or only partially written.
    (decisions / "2026-07-28.json").write_text("{", encoding="utf-8")

    dates, basis = _source_observed_dates(
        spec,
        pd.Timestamp("2026-07-28"),
        snapshot_root=tmp_path / "snapshots",
        pump_v2_decision_root=decisions,
        pump_v2_receipt_root=tmp_path / "receipts",
    )

    assert dates == [pd.Timestamp("2026-07-27")]
    assert basis == (
        "post_contract_validated_healthy_decisions"
        "(delivery_independent)+delivery_verified_closed_rows;"
        "legacy_dates_excluded"
    )


def test_pump_v2_historical_manifest_tamper_fails_closed(tmp_path):
    spec = ModelSpec(
        id="pump_hunter_v2",
        name="test",
        ledger_path=str(tmp_path / "missing.csv"),
        slots=["open"],
        metric=MetricSource(
            status_col="status",
            closed_value="closed",
            date_col="date",
            realized_pct_col="realized_pct",
        ),
        predict_ref="signals.pump_detector_v2:score_pump_v2_candidates",
    )
    decisions = tmp_path / "decisions"
    path = _write_pump_v2_decision(decisions, "2026-07-27")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["decision"]["universe_n"] = 999
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        _source_observed_dates(
            spec,
            pd.Timestamp("2026-07-28"),
            snapshot_root=tmp_path / "snapshots",
            pump_v2_decision_root=decisions,
            pump_v2_receipt_root=tmp_path / "receipts",
        )


def test_pump_v2_stale_zero_pick_is_preserved_but_not_recall_denominator(
    tmp_path,
):
    spec = ModelSpec(
        id="pump_hunter_v2",
        name="test",
        ledger_path=str(tmp_path / "missing.csv"),
        slots=["open"],
        metric=MetricSource(
            status_col="status",
            closed_value="closed",
            date_col="date",
            realized_pct_col="realized_pct",
        ),
        predict_ref="signals.pump_detector_v2:score_pump_v2_candidates",
    )
    decisions = tmp_path / "decisions"
    _write_pump_v2_decision(
        decisions,
        "2026-07-27",
        binance_status="binance_partial",
    )

    dates, basis = _source_observed_dates(
        spec,
        pd.Timestamp("2026-07-28"),
        snapshot_root=tmp_path / "snapshots",
        pump_v2_decision_root=decisions,
        pump_v2_receipt_root=tmp_path / "receipts",
    )

    assert dates == []
    assert basis == (
        "post_contract_validated_healthy_decisions"
        "(delivery_independent)+delivery_verified_closed_rows;"
        "legacy_dates_excluded"
    )


def test_pump_v2_candidate_delivery_failure_stays_in_recall_denominator(
    tmp_path,
):
    ledger = tmp_path / "v2.csv"
    spec = ModelSpec(
        id="pump_hunter_v2",
        name="test",
        ledger_path=str(ledger),
        slots=["open"],
        metric=MetricSource(
            status_col="status",
            closed_value="closed",
            date_col="date",
            realized_pct_col="realized_pct",
        ),
        predict_ref="signals.pump_detector_v2:score_pump_v2_candidates",
    )
    decisions = tmp_path / "decisions"
    decision_path = _write_pump_v2_decision(
        decisions,
        "2026-07-27",
        candidates=[{"market": "KRW-B", "rank": 1}],
    )
    decision = json.loads(
        decision_path.read_text(encoding="utf-8")
    )["decision"]
    v2_runner.append_ledger(
        decision,
        str(ledger),
        False,
        decision_completed_at="2026-07-27T00:04:59+00:00",
        delivery_ok=False,
        receipt_path=str(decision_path),
    )

    dates, basis = _source_observed_dates(
        spec,
        pd.Timestamp("2026-07-29"),
        snapshot_root=tmp_path / "snapshots",
        pump_v2_decision_root=decisions,
        pump_v2_receipt_root=tmp_path / "receipts",
    )

    assert dates == [pd.Timestamp("2026-07-27")]
    assert basis == (
        "post_contract_validated_healthy_decisions"
        "(delivery_independent)+delivery_verified_closed_rows;"
        "legacy_dates_excluded"
    )


def test_policy_rejects_fake_v2_closed_row_without_canonical_evidence(
    tmp_path,
):
    ledger = tmp_path / "v2.csv"
    ledger.write_text(
        "date,status,realized_pct,coin\n"
        "2026-07-27,closed,9.9,KRW-FAKE\n",
        encoding="utf-8",
    )
    spec = ModelSpec(
        id="pump_hunter_v2",
        name="test",
        ledger_path=str(ledger),
        slots=["open"],
        metric=MetricSource(
            status_col="status",
            closed_value="closed",
            date_col="date",
            realized_pct_col="realized_pct",
        ),
        predict_ref="signals.pump_detector_v2:score_pump_v2_candidates",
    )

    with pytest.raises(
        ValueError,
        match="no canonical decision manifest",
    ):
        competition._load_model_rows(
            spec,
            pd.Timestamp("2026-07-28"),
            pump_v2_decision_root=tmp_path / "decisions",
            pump_v2_receipt_root=tmp_path / "receipts",
        )


def test_policy_excludes_precontract_v2_rows_from_forward_metrics(tmp_path):
    ledger = tmp_path / "v2.csv"
    ledger.write_text(
        "date,status,realized_pct,coin,btc_regime\n"
        "2026-07-25,closed,99.0,KRW-LEGACY,bear_quiet\n",
        encoding="utf-8",
    )
    spec = ModelSpec(
        id="pump_hunter_v2",
        name="test",
        ledger_path=str(ledger),
        slots=["open"],
        metric=MetricSource(
            status_col="status",
            closed_value="closed",
            date_col="date",
            realized_pct_col="realized_pct",
        ),
        predict_ref="signals.pump_detector_v2:score_pump_v2_candidates",
    )

    rows = competition._load_model_rows(
        spec,
        pd.Timestamp("2026-07-28"),
        pump_v2_decision_root=tmp_path / "decisions",
        pump_v2_receipt_root=tmp_path / "receipts",
    )

    assert rows.empty


def test_policy_rejects_v2_ledger_candidate_not_in_manifest(tmp_path):
    ledger = tmp_path / "v2.csv"
    decisions = tmp_path / "decisions"
    decision_path = _write_pump_v2_decision(
        decisions,
        "2026-07-27",
        candidates=[{"market": "KRW-REAL"}],
    )
    decision = json.loads(
        decision_path.read_text(encoding="utf-8")
    )["decision"]
    v2_runner.append_ledger(
        decision,
        str(ledger),
        False,
        decision_completed_at="2026-07-27T00:04:59+00:00",
        delivery_ok=False,
        receipt_path=str(decision_path),
    )
    frame = pd.read_csv(ledger)
    frame.loc[0, "coin"] = "KRW-FAKE"
    frame.to_csv(ledger, index=False)
    spec = ModelSpec(
        id="pump_hunter_v2",
        name="test",
        ledger_path=str(ledger),
        slots=["open"],
        metric=MetricSource(
            status_col="status",
            closed_value="closed",
            date_col="date",
            realized_pct_col="realized_pct",
        ),
        predict_ref="signals.pump_detector_v2:score_pump_v2_candidates",
    )

    with pytest.raises(ValueError, match="not in decision manifest"):
        competition._load_model_rows(
            spec,
            pd.Timestamp("2026-07-28"),
            pump_v2_decision_root=decisions,
            pump_v2_receipt_root=tmp_path / "receipts",
        )

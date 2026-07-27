from __future__ import annotations

import json
import sqlite3

import pandas as pd
import pytest

import ops.champion_selector as selector
import ops.policy_competition as competition
import scripts.build_findings_dashboard as findings
from ops.artifact_provenance import atomic_write_bytes
from scripts.build_findings_dashboard import (
    FindingsDataError,
    build_champion_leaderboard,
    build_policy_competition_panel,
    verify_backtest_pumps,
)


def _policy_payload(asof: str) -> dict:
    row = {
        column: None for column in competition.POLICY_ROW_COLUMNS
    }
    row.update(
        {
            "asof": asof,
            "participant_id": "safe",
            "source_id": "recommend_r1_open",
            "policy_id": "top_all",
            "objective": "baseline",
            "description": "test",
            "n_closed": 1,
            "n_days": 1,
            "n_selected_days": 1,
            "net_mean_pct": 1.0,
            "pump20_recall_pct": 1.0,
            "pump20_captured": 1,
            "pump20_actual": 1,
            "pump_days": 1,
            "pump_days_any_captured": 1,
            "post_send_label_n": 0,
            "recall_date_basis": "ledger_rows",
            "pump20_label_basis": "full_day_proxy",
        }
    )
    config = competition._policy_config()
    manifest = competition.build_policy_input_manifest(asof)
    return competition._finalize_policy_payload(
        {
            "asof": asof,
            "generated_at_utc": f"{asof}T01:00:00+00:00",
            "input_manifest": manifest,
            "config": config,
            "radar_terminal": {"status": "not_due"},
            "rows": [row],
            "exit_lab": [],
        }
    )


def _champion_state(asof: str) -> dict:
    state = {
        "schema_version": selector.STATE_SCHEMA_VERSION,
        "asof": asof,
        "updated_at": f"{asof}T00:00:00+00:00",
        "config": selector._expected_config(),
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
        "history": [],
    }
    state["payload_sha256"] = selector._state_digest(state)
    return state


def _write_policy_triplet(summary, database, payload):
    competition._atomic_write_json(summary, payload)
    atomic_write_bytes(
        summary.with_suffix(".csv"),
        competition._policy_csv_bytes(payload),
    )
    competition._write_sqlite(payload, database)


def test_policy_panel_rejects_future_summary_artifact(tmp_path):
    summary = tmp_path / "summary.json"
    db = tmp_path / "competition.db"
    _write_policy_triplet(summary, db, _policy_payload("2026-07-26"))

    with pytest.raises(FindingsDataError, match="triplet"):
        build_policy_competition_panel(
            str(summary),
            str(db),
            asof="2026-07-25",
        )


def test_policy_panel_binds_database_metadata_to_summary_asof(tmp_path):
    summary = tmp_path / "summary.json"
    db = tmp_path / "competition.db"
    payload = _policy_payload("2026-07-20")
    _write_policy_triplet(summary, db, payload)

    panel = build_policy_competition_panel(
        str(summary),
        str(db),
        asof="2026-07-25",
    )

    assert panel is not None
    assert panel["database"]["latest_asof"] == "2026-07-20"
    assert panel["database"]["latest_row_count"] == 1


def test_future_champion_state_fails_closed(tmp_path, monkeypatch):
    state = tmp_path / "champion_state.json"
    state.write_text(
        json.dumps(_champion_state("2026-07-26")),
        encoding="utf-8",
    )
    monkeypatch.setattr("ops.champion_selector.STATE_PATH", state)

    with pytest.raises(FindingsDataError, match="missing or invalid"):
        build_champion_leaderboard(asof="2026-07-25")


def test_champion_state_requires_canonical_checksum_and_config(
    tmp_path,
    monkeypatch,
):
    state_path = tmp_path / "champion_state.json"
    state = _champion_state("2026-07-25")
    state["config"]["rolling_n"] = 999
    state["payload_sha256"] = selector._state_digest(state)
    state_path.write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr("ops.champion_selector.STATE_PATH", state_path)

    with pytest.raises(FindingsDataError, match="missing or invalid"):
        build_champion_leaderboard(asof="2026-07-25")


@pytest.mark.parametrize("malformation", ["duplicate", "nan"])
def test_findings_json_reader_rejects_ambiguous_json(
    malformation,
    tmp_path,
):
    path = tmp_path / "source.json"
    raw = '{"value":1}'
    if malformation == "duplicate":
        raw = '{"value":1,"value":2}'
    else:
        raw = '{"value":NaN}'
    path.write_text(raw, encoding="utf-8")

    with pytest.raises(FindingsDataError, match="invalid findings JSON"):
        findings._read_json_artifact(str(path))


@pytest.mark.parametrize("malformation", ["duplicate", "nan"])
def test_findings_champion_loader_rejects_ambiguous_json(
    malformation,
    tmp_path,
    monkeypatch,
):
    state_path = tmp_path / "champion_state.json"
    raw = json.dumps(_champion_state("2026-07-25"))
    if malformation == "duplicate":
        raw = raw.replace(
            '"asof": "2026-07-25"',
            '"asof": "2026-07-25", "asof": "2026-07-25"',
            1,
        )
    else:
        raw = raw.replace(
            '"history": []',
            '"history": [], "metric": NaN',
            1,
        )
    state_path.write_text(raw, encoding="utf-8")
    monkeypatch.setattr("ops.champion_selector.STATE_PATH", state_path)

    with pytest.raises(FindingsDataError, match="missing or invalid"):
        build_champion_leaderboard(asof="2026-07-25")


def test_policy_panel_requires_matching_database_snapshot(tmp_path):
    summary = tmp_path / "summary.json"
    db = tmp_path / "competition.db"
    payload = _policy_payload("2026-07-20")
    _write_policy_triplet(summary, db, payload)
    with sqlite3.connect(db) as con:
        con.execute(
            "UPDATE policy_competition_runs SET row_count=2 WHERE asof=?",
            ("2026-07-20",),
        )

    with pytest.raises(FindingsDataError, match="triplet"):
        build_policy_competition_panel(
            str(summary),
            str(db),
            asof="2026-07-20",
        )


def test_policy_panel_rejects_fractional_closed_count(tmp_path):
    summary = tmp_path / "summary.json"
    db = tmp_path / "competition.db"
    payload = _policy_payload("2026-07-20")
    _write_policy_triplet(summary, db, payload)
    payload["rows"][0]["n_closed"] = 1.5
    payload["run_id"] = competition._policy_run_identity(payload)
    payload["payload_sha256"] = competition.payload_digest(payload)
    competition._atomic_write_json(summary, payload)

    with pytest.raises(FindingsDataError, match="triplet"):
        build_policy_competition_panel(
            str(summary),
            str(db),
            asof="2026-07-20",
        )


def _stub_noncore_panels(monkeypatch):
    monkeypatch.setattr(findings, "verify_backtest_pumps", lambda *a, **k: [])
    monkeypatch.setattr(
        findings,
        "build_magnitude_curve",
        lambda **kwargs: {},
    )
    monkeypatch.setattr(findings, "build_risk_reward_panel", lambda: {})
    monkeypatch.setattr(findings, "build_calibration_panel", lambda: {})
    monkeypatch.setattr(findings, "build_precision_at3_panel", lambda: {})
    monkeypatch.setattr(findings, "build_precursor_lift_panel", lambda: {})
    monkeypatch.setattr(findings, "build_regime_baserate_panel", lambda: {})


def test_canonical_payload_fails_when_core_panel_is_invalid(monkeypatch):
    _stub_noncore_panels(monkeypatch)

    def fail_champion(**kwargs):
        raise FindingsDataError("bad champion")

    monkeypatch.setattr(findings, "build_champion_leaderboard", fail_champion)

    with pytest.raises(FindingsDataError, match="requires valid champion"):
        findings.build_payload("unused.db", asof="2026-07-25")


def test_canonical_payload_requires_policy_snapshot_at_cutoff(monkeypatch):
    _stub_noncore_panels(monkeypatch)
    monkeypatch.setattr(
        findings,
        "build_champion_leaderboard",
        lambda **kwargs: {
            "rows": [],
            "current_champion": {"open": "r1", "preopen": "r1"},
        },
    )
    monkeypatch.setattr(
        findings,
        "build_policy_competition_panel",
        lambda **kwargs: {"asof": "2026-07-24"},
    )

    with pytest.raises(
        FindingsDataError,
        match="requires current policy competition",
    ):
        findings.build_payload("unused.db", asof="2026-07-25")


def test_preview_mode_explicitly_allows_missing_core_panels(monkeypatch):
    _stub_noncore_panels(monkeypatch)

    def fail_core(**kwargs):
        raise FindingsDataError("missing")

    monkeypatch.setattr(findings, "build_champion_leaderboard", fail_core)
    monkeypatch.setattr(findings, "build_policy_competition_panel", fail_core)

    payload = findings.build_payload(
        "unused.db",
        asof="2026-07-25",
        allow_missing_core=True,
    )

    assert payload["champion_leaderboard"] is None
    assert payload["policy_competition"] is None


def test_backtest_db_rows_after_asof_are_excluded(tmp_path, monkeypatch):
    db = tmp_path / "d1.db"
    con = sqlite3.connect(db)
    try:
        con.execute(
            "CREATE TABLE candles (market TEXT, timestamp TEXT, open REAL, high REAL)"
        )
        con.executemany(
            "INSERT INTO candles VALUES (?, ?, ?, ?)",
                [
                    ("KRW-PAST", "2026-07-24 09:00:00", 100.0, 110.0),
                    ("KRW-BAD", "2026-07-24 09:00:00", "bad", 999.0),
                    ("KRW-FUTURE", "2026-07-26 09:00:00", 100.0, 999.0),
            ],
        )
        con.commit()
    finally:
        con.close()
    monkeypatch.setattr(
        "scripts.build_findings_dashboard.BACKTEST_PUMPS",
            [
                {"market": "KRW-PAST", "date": "2026-07-24"},
                {"market": "KRW-BAD", "date": "2026-07-24"},
                {"market": "KRW-FUTURE", "date": "2026-07-26"},
        ],
    )

    rows = verify_backtest_pumps(str(db), asof="2026-07-25")

    assert [row["coin"] for row in rows] == ["PAST"]


def test_calibration_and_precision_panels_read_current_artifacts(
    tmp_path,
    monkeypatch,
):
    legacy = tmp_path / "legacy.json"
    calibration = tmp_path / "calibration.csv"
    metrics = tmp_path / "metrics.csv"
    legacy.write_text(
        json.dumps(
            {
                "h5": {
                    "top_bucket_mean_pred_pct": 70.0,
                    "top_bucket_actual_hit_pct": 10.0,
                    "top_bucket_n": 20,
                }
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "b": 9,
                "pred_prob": 0.125,
                "actual_hit": 0.2,
                "n": 40,
                "score": "cal_composite",
                "label": "lab_pump20",
            }
        ]
    ).to_csv(calibration, index=False)
    pd.DataFrame(
        [
            {
                "score": "cal_composite",
                "label": "lab_pump20",
                "regime": "ALL",
                "K": 3,
                "n_days": 10,
                "n_picks": 30,
                "precision_at_k": 0.3,
                "base_rate": 0.1,
                "lift": 3.0,
            }
        ]
    ).to_csv(metrics, index=False)
    monkeypatch.setattr(findings, "LEGACY_CALIBRATION_JSON", str(legacy))
    monkeypatch.setattr(findings, "SCORER_CALIBRATION_CSV", str(calibration))
    monkeypatch.setattr(findings, "SCORER_METRICS_CSV", str(metrics))

    calibration_panel = findings.build_calibration_panel()
    precision_panel = findings.build_precision_at3_panel()

    assert calibration_panel["research_scorer"]["pred_pct"] == 12.5
    assert calibration_panel["research_scorer"]["actual_pct"] == 20.0
    assert calibration_panel["research_scorer"]["overconfidence_pp"] == -7.5
    assert precision_panel["full_oos_pct"] == 30.0
    assert precision_panel["full_oos_base_pct"] == 10.0
    assert precision_panel["full_oos_lift"] == 3.0
    assert calibration_panel["sources"][1]["path"] == str(calibration)

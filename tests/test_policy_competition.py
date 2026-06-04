from __future__ import annotations

import pandas as pd

from ops.policy_competition import (
    POLICIES,
    PolicySpec,
    _consensus_rows,
    _evaluate_rows,
    _write_sqlite,
    _true_series,
)


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


def test_no_dump_policy_filters_flagged_rows():
    rows = pd.DataFrame([
        {"rank": 1, "dump_risk_flag": False},
        {"rank": 2, "dump_risk_flag": True},
        {"rank": 3, "dump_risk_flag": "False"},
        {"rank": 4, "dump_risk_flag": False},
    ])
    no_dump = next(p for p in POLICIES if p.policy_id == "no_dump_top3")

    mask = no_dump.predicate(rows)

    assert mask.tolist() == [True, False, True, False]


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
    payload = {
        "asof": "2026-06-04",
        "generated_at_utc": "2026-06-04T00:00:00+00:00",
        "config": {
            "pump20_threshold": 0.20,
            "deep_loss_pct": -5.0,
            "round_trip_cost_pct": 0.15,
        },
        "rows": [
            {
                "participant_id": "r1:top_all",
                "source_id": "r1",
                "policy_id": "top_all",
                "objective": "baseline",
                "description": "all",
                "n_closed": 3,
                "n_days": 1,
                "net_sum_pct": 1.0,
                "net_mean_pct": 0.3333,
                "deep_loss_freq_pct": 0.0,
                "sl_rate_pct": 33.3333,
                "tp_rate_pct": 33.3333,
                "pump20_precision_pct": 50.0,
                "pump20_recall_pct": 25.0,
                "pump20_captured": 1,
                "pump20_actual": 4,
                "pump_days": 1,
                "pump_days_any_captured": 1,
                "pump_day_capture_rate_pct": 100.0,
            }
        ],
    }

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

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

import ops.champion_selector as selector
from ledger.config import ROUND_TRIP_COST_PCT
from ops.artifact_provenance import ArtifactValidationError
from scripts.build_dashboard import (
    _build_champion_gate_payload,
    _build_pump_hunter_payload,
    _daily_pnl_from_closed,
    _load_or_empty,
    _load_optional_artifact,
    _rows_through_asof,
    _virtual_pnl_per_alert,
    _write_json,
    compute_bootstrap_ci,
    compute_distribution_summary,
    compute_factor_regression,
    compute_quant_metrics,
    compute_recommend_summary,
    compute_tp_sweep,
    cumulative_pnl_series,
)
from scripts.idea_validation_report import (
    IDEA_ARTIFACT_SCHEMA,
    build_input_manifest,
    report_payload_digest,
)


def test_dashboard_equal_weights_multiple_alerts_on_same_day():
    closed = pd.DataFrame([
        {"date": "2026-07-01", "max_ret": 4.0, "close_ret": 10.0},
        {"date": "2026-07-01", "max_ret": 4.0, "close_ret": -10.0},
        {"date": "2026-07-03", "max_ret": 6.0, "close_ret": 0.0},
    ])

    daily = _daily_pnl_from_closed(
        closed,
        "max_ret",
        "close_ret",
        asof="2026-07-05",
    )

    assert daily.iloc[0] == pytest.approx(-ROUND_TRIP_COST_PCT)
    assert daily.iloc[1] == pytest.approx(0.0)  # no-alert cash day
    assert daily.iloc[2] == pytest.approx(0.05 - ROUND_TRIP_COST_PCT)
    assert daily.iloc[3:].tolist() == pytest.approx([0.0, 0.0])
    assert daily.index[-1] == pd.Timestamp("2026-07-05")


def test_bootstrap_clusters_signal_dates_without_inventing_cash_clusters():
    closed = pd.DataFrame([
        {"date": "2026-07-01", "max_ret": 6.0, "close_ret": 0.0},
        {"date": "2026-07-03", "max_ret": 1.0, "close_ret": -1.0},
    ])

    out = compute_bootstrap_ci(closed, "max_ret", "close_ret")

    assert out["n_days"] == 2


def test_dashboard_mdd_includes_first_day_loss_and_uses_365_sharpe():
    closed = pd.DataFrame([
        {"date": "2026-07-01", "max_ret": 1.0, "close_ret": -10.0},
        {"date": "2026-07-02", "max_ret": 6.0, "close_ret": 1.0},
    ])

    out = compute_quant_metrics(
        closed,
        "max_ret",
        "close_ret",
        asof="2026-07-05",
    )

    assert out["max_drawdown_pct"] == pytest.approx(
        (-0.10 - ROUND_TRIP_COST_PCT) * 100
    )
    assert out["n_days"] == 5
    assert out["n_signal_days"] == 2


def test_tp_sweep_cumulative_is_day_equal_weight_not_signal_sum():
    closed = pd.DataFrame([
        {"date": "2026-07-01", "max_ret": 1.0, "close_ret": 10.0},
        {"date": "2026-07-01", "max_ret": 1.0, "close_ret": 0.0},
    ])

    out = compute_tp_sweep(
        closed,
        "max_ret",
        "close_ret",
        tp_list=[],
        include_no_tp=True,
        cost=0.0,
    )

    assert out[0]["cum_pnl_pct"] == pytest.approx(5.0)


def test_dashboard_pnl_requires_close_when_take_profit_was_not_observed():
    assert pd.isna(_virtual_pnl_per_alert(1.0, None))
    assert pd.isna(_virtual_pnl_per_alert(1.0, float("nan")))
    assert pd.isna(_virtual_pnl_per_alert(float("inf"), 3.0))
    assert _virtual_pnl_per_alert(6.0, None) == pytest.approx(
        0.05 - ROUND_TRIP_COST_PCT
    )

    closed = pd.DataFrame(
        [
            {"date": "2026-07-01", "max_ret": None, "close_ret": 10.0},
            {"date": "2026-07-02", "max_ret": 6.0, "close_ret": None},
            {"date": "2026-07-03", "max_ret": float("inf"), "close_ret": 3.0},
            {"date": "2026-07-04", "max_ret": 1.0, "close_ret": float("inf")},
        ]
    )
    sweep = compute_tp_sweep(closed, "max_ret", "close_ret")

    assert sweep[0]["n_trades"] == 1
    assert sweep[-1]["n_trades"] == 2


def test_no_tp_sweep_needs_only_finite_close_evidence():
    closed = pd.DataFrame(
        [
            {"date": "2026-07-01", "max_ret": None, "close_ret": 2.0},
            {
                "date": "2026-07-02",
                "max_ret": None,
                "close_ret": float("inf"),
            },
        ]
    )

    sweep = compute_tp_sweep(
        closed,
        "max_ret",
        "close_ret",
        tp_list=(0.05,),
        include_no_tp=True,
    )

    assert sweep[0]["n_trades"] == 0
    assert sweep[0]["tp_hit_rate_pct"] is None
    assert sweep[-1]["n_trades"] == 1


def test_dashboard_cumulative_series_extends_flat_through_asof():
    ledger = pd.DataFrame([
        {
            "date": "2026-07-01",
            "status": "closed",
            "max_ret": 6.0,
            "close_ret": 0.0,
        },
        {
            "date": "2026-07-03",
            "status": "closed",
            "max_ret": 1.0,
            "close_ret": -1.0,
        },
    ])

    rows = cumulative_pnl_series(
        ledger,
        "max_ret",
        "close_ret",
        asof="2026-07-05",
    )

    assert len(rows) == 5
    assert rows[-1]["date"] == "2026-07-05 00:00:00"
    assert rows[-2]["daily_pnl_pct"] == pytest.approx(0.0)
    assert rows[-1]["daily_pnl_pct"] == pytest.approx(0.0)
    assert rows[-1]["cum_pnl_pct"] == pytest.approx(rows[-3]["cum_pnl_pct"])


def test_factor_regression_recovers_daily_returns_from_legacy_wealth():
    dates = pd.date_range("2026-07-01", periods=12)
    returns = [
        0.0, 0.010, -0.005, 0.015, 0.002, -0.010,
        0.008, 0.004, -0.003, 0.012, -0.007, 0.006,
    ]
    closed = pd.DataFrame([
        {
            "date": str(date.date()),
            "max_ret": 1.0,
            # After standard cost, strategy return equals BTC's daily return.
            "close_ret": ret * 100 + ROUND_TRIP_COST_PCT * 100,
        }
        for date, ret in zip(dates, returns)
    ])
    btc_series = []
    wealth = 1.0
    for date, ret in zip(dates, returns):
        wealth *= 1.0 + ret
        btc_series.append({
            "date": str(date.date()),
            "btc_cum_pct": (wealth - 1.0) * 100,
        })

    out = compute_factor_regression(
        closed, "max_ret", "close_ret", btc_series
    )

    assert out["beta"] == pytest.approx(1.0)
    assert out["alpha_ann_pct"] == pytest.approx(0.0)


def test_recommend_summary_uses_day_equal_compounding_and_separates_hit_basis():
    ledger = pd.DataFrame([
        {
            "date": "2026-07-01",
            "coin": "KRW-A",
            "rank": 1,
            "status": "closed",
            "realized_pct": 20.0,
            "pump20_hit": 1,
            "post_send_pump20_hit": 0,
        },
        {
            "date": "2026-07-01",
            "coin": "KRW-B",
            "rank": 2,
            "status": "closed",
            "realized_pct": 0.0,
            "pump20_hit": 0,
            "post_send_pump20_hit": 0,
        },
        {
            "date": "2026-07-02",
            "coin": "KRW-C",
            "rank": 1,
            "status": "closed",
            "realized_pct": -10.0,
            "pump20_hit": 0,
            "post_send_pump20_hit": 1,
        },
    ])

    out = compute_recommend_summary(ledger, asof="2026-07-05")

    assert out["legacy_per_trade_sum_net_pct"] == pytest.approx(10.0)
    assert out["cum_net_pnl_pct"] == pytest.approx(-1.0)
    assert out["n_signal_days"] == 2
    assert out["n_calendar_days"] == 5
    assert out["max_drawdown_pct"] == pytest.approx(-10.0)
    assert out["pump20_hit_rate_pct"] == pytest.approx(100 / 3)
    assert out["post_send_pump20_hit_rate_pct"] == pytest.approx(100 / 3)
    assert out["pump20_hit_basis"] == "legacy_full_day_D1"
    assert out["post_send_pump20_hit_basis"] == "sent_at_after_15m_path"


def test_distribution_summary_uses_one_asof_cohort_for_every_headline():
    ledger = pd.DataFrame([
        {
            "date": "2026-07-24",
            "coin": "KRW-SAFE",
            "status": "closed",
            "next_max_return_pct": 1.0,
            "next_min_return_pct": -2.0,
            "next_close_return_pct": -1.0,
            "next_open": 100.0,
            "next_high": 101.0,
            "next_low": 98.0,
            "next_close": 99.0,
            "hit_h2": 0,
            "hit_h6": 0,
            "hit_h5": 0,
        },
        {
            "date": "2026-07-26",
            "coin": "KRW-FUTURE",
            "status": "closed",
            "next_max_return_pct": 999.0,
            "next_min_return_pct": 999.0,
            "next_close_return_pct": 999.0,
            "next_open": 100.0,
            "next_high": 1099.0,
            "next_low": 1099.0,
            "next_close": 1099.0,
            "hit_h2": 1,
            "hit_h6": 1,
            "hit_h5": 1,
        },
    ])

    out = compute_distribution_summary(ledger, asof="2026-07-25")

    assert out["n_alerts_total"] == 1
    assert out["n_closed"] == 1
    assert out["last_alert_date"] == "2026-07-24"
    assert out["avg_max_return_pct"] == pytest.approx(1.0)
    assert out["hit_h6_pct"] == pytest.approx(0.0)
    assert out["virtual"]["n_trades"] == 1
    assert out["quant"]["n_trades"] == 1


def test_recommend_summary_excludes_future_closed_and_future_radar():
    ledger = pd.DataFrame([
        {
            "date": "2026-07-24",
            "coin": "KRW-SAFE",
            "rank": 1,
            "status": "closed",
            "realized_pct": -1.0,
            "pump20_hit": 0,
            "post_send_pump20_hit": 0,
        },
        {
            "date": "2026-07-26",
            "coin": "KRW-FUTURE",
            "rank": 1,
            "status": "closed",
            "realized_pct": 999.0,
            "pump20_hit": 1,
            "post_send_pump20_hit": 1,
        },
    ])

    out = compute_recommend_summary(ledger, asof="2026-07-25")

    assert out["n_alerts_total"] == 1
    assert out["n_closed"] == 1
    assert out["latest_radar_date"] == "2026-07-24"
    assert [row["coin"] for row in out["latest_radar"]] == ["SAFE"]
    assert out["avg_net_realized_pct"] == pytest.approx(-1.0)
    assert out["pump20_hit_rate_pct"] == pytest.approx(0.0)


def test_dashboard_row_cutoff_converts_utc_instants_to_kst_dates():
    rows = pd.DataFrame([
        {"date": "2026-07-25T14:59:59+00:00", "value": "safe"},
        {"date": "2026-07-25T15:00:00+00:00", "value": "future"},
    ])

    safe = _rows_through_asof(rows, "2026-07-25")

    assert safe["value"].tolist() == ["safe"]
    assert safe["date"].tolist() == ["2026-07-25"]


def test_dashboard_optional_artifacts_fail_closed_on_future_or_missing_lineage(
    tmp_path,
):
    future_idea = {
        "asof": "2026-07-26",
        "input_lineage": {"schema_version": 1, "kind": "test"},
    }
    future_idea["payload_sha256"] = report_payload_digest(future_idea)
    future_path = tmp_path / "future_idea.json"
    future_path.write_text(json.dumps(future_idea), encoding="utf-8")
    legacy_path = tmp_path / "legacy_idea.json"
    legacy_path.write_text(json.dumps({"n_closed": 999}), encoding="utf-8")

    assert _load_optional_artifact(
        future_path,
        asof="2026-07-25",
        kind="idea_validation",
    ) is None
    assert _load_optional_artifact(
        legacy_path,
        asof="2026-07-25",
        kind="idea_validation",
    ) is None


def test_dashboard_idea_artifact_fails_closed_when_source_bytes_change(tmp_path):
    ledger = tmp_path / "paper.csv"
    ledger.write_text("date,coin\n2026-07-25,KRW-BTC\n", encoding="utf-8")
    args = SimpleNamespace(
        paper_ledger=ledger,
        paper_ledger_preopen=tmp_path / "preopen.csv",
        shadow_ledger_distribution=tmp_path / "shadow_distribution.csv",
        shadow_ledger_preopen=tmp_path / "shadow_preopen.csv",
    )
    lineage = build_input_manifest(
        args,
        policy_competition_path=tmp_path / "policy.json",
        meta_model_dir=tmp_path / "model",
    )
    payload = {
        "schema": IDEA_ARTIFACT_SCHEMA,
        "asof": "2026-07-25",
        "generated_at_utc": "2026-07-25T00:00:00+00:00",
        "input_lineage": lineage,
        "n_closed": 1,
    }
    payload["payload_sha256"] = report_payload_digest(payload)
    artifact = tmp_path / "idea.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    assert _load_optional_artifact(
        artifact,
        asof="2026-07-25",
        kind="idea_validation",
    ) is not None

    ledger.write_text(
        "date,coin\n2026-07-25,KRW-ETH\n",
        encoding="utf-8",
    )
    assert _load_optional_artifact(
        artifact,
        asof="2026-07-25",
        kind="idea_validation",
    ) is None


def test_dashboard_meta_artifact_trained_after_asof_is_excluded(tmp_path):
    path = tmp_path / "meta.json"
    path.write_text(
        json.dumps(
            {
                "built_at": "2026-07-26T01:00:00+09:00",
                "date_range": {"end": "2026-07-26"},
                "deployable": True,
            }
        ),
        encoding="utf-8",
    )

    assert _load_optional_artifact(
        path,
        asof="2026-07-25",
        kind="recommendation_meta",
    ) is None


@pytest.mark.parametrize("malformation", ["duplicate", "nan"])
def test_dashboard_meta_artifact_rejects_ambiguous_json(
    malformation,
    tmp_path,
):
    path = tmp_path / "meta.json"
    raw = (
        '{"built_at":"2026-07-25T01:00:00+09:00",'
        '"date_range":{"end":"2026-07-25"},'
        '"deployable":true}'
    )
    if malformation == "duplicate":
        raw = raw.replace(
            '"deployable":true',
            '"deployable":true,"deployable":false',
        )
    else:
        raw = raw.replace(
            '"deployable":true',
            '"deployable":true,"metric":NaN',
        )
    path.write_text(raw, encoding="utf-8")

    assert _load_optional_artifact(
        path,
        asof="2026-07-25",
        kind="recommendation_meta",
    ) is None


def test_dashboard_champion_gate_requires_canonical_validated_state(
    tmp_path,
):
    path = tmp_path / "champion.json"
    path.write_text(
        json.dumps(
            {
                "asof": "2026-07-25",
                "config": {"min_closed": 30},
                "slots": {},
            }
        ),
        encoding="utf-8",
    )
    assert _build_champion_gate_payload(
        path,
        asof="2026-07-25",
    ) is None

    state = {
        "schema_version": selector.STATE_SCHEMA_VERSION,
        "asof": "2026-07-25",
        "updated_at": "2026-07-25T00:00:00+00:00",
        "config": selector._expected_config(),
        "slots": {
            "open": {
                "champion_id": "recommend_r1_open",
                "since": "2026-07-25",
                "is_fallback": False,
                "metric": None,
                "reason": "test",
            },
            "preopen": {
                "champion_id": "recommend_r1_preopen",
                "since": "2026-07-25",
                "is_fallback": True,
                "metric": None,
                "reason": "test",
            },
        },
        "streaks": {},
        "history": [],
    }
    state["payload_sha256"] = selector._state_digest(state)
    selector._atomic_write_state(path, state)

    gate = _build_champion_gate_payload(
        path,
        asof="2026-07-25",
    )
    assert gate is not None
    assert {row["slot"] for row in gate["slots"]} == {
        "open",
        "preopen",
    }


def test_pump_panel_and_champion_gate_exclude_future_state(tmp_path):
    pump_path = tmp_path / "pump.csv"
    pd.DataFrame([
        {
            "date": "2026-07-24",
            "coin": "KRW-SAFE",
            "rank": 1,
            "status": "closed",
            "pump20_hit": 0,
        },
        {
            "date": "2026-07-26",
            "coin": "KRW-FUTURE",
            "rank": 1,
            "status": "closed",
            "pump20_hit": 1,
        },
    ]).to_csv(pump_path, index=False)
    state_path = tmp_path / "champion.json"
    state_path.write_text(
        json.dumps(
            {
                "asof": "2026-07-26",
                "config": {"min_closed": 30},
                "slots": {},
            }
        ),
        encoding="utf-8",
    )

    pump = _build_pump_hunter_payload(pump_path, asof="2026-07-25")

    assert pump is not None
    assert pump["latest_date"] == "2026-07-24"
    assert [row["coin"] for row in pump["watchlist"]] == ["SAFE"]
    assert pump["rows_closed"] == 1
    assert _build_champion_gate_payload(
        state_path,
        asof="2026-07-25",
    ) is None


def test_dashboard_json_replace_failure_preserves_last_complete_artifact(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "summary.json"
    target.write_text('{"generation":"old"}', encoding="utf-8")

    def fail_replace(*_args, **_kwargs):
        raise OSError("injected replace failure")

    monkeypatch.setattr("scripts.build_dashboard.os.replace", fail_replace)

    with pytest.raises(OSError, match="injected replace failure"):
        _write_json(target, {"generation": "new"})

    assert json.loads(target.read_text(encoding="utf-8")) == {
        "generation": "old"
    }
    assert list(tmp_path.glob(".*.tmp")) == []


def test_dashboard_json_writer_rejects_symlink_target(tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text('{"generation":"outside"}\n', encoding="utf-8")
    target = tmp_path / "summary.json"
    target.symlink_to(outside)

    with pytest.raises(OSError, match="regular file"):
        _write_json(target, {"generation": "new"})

    assert json.loads(outside.read_text(encoding="utf-8")) == {
        "generation": "outside"
    }


def test_dashboard_csv_reader_rejects_symlinked_parent(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "ledger.csv").write_text(
        "date,status\n2026-07-26,closed\n",
        encoding="utf-8",
    )
    linked_parent = tmp_path / "linked-output"
    linked_parent.symlink_to(outside, target_is_directory=True)

    with pytest.raises(
        ArtifactValidationError,
        match="parent must be a real directory",
    ):
        _load_or_empty(linked_parent / "ledger.csv")

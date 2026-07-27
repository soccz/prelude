from __future__ import annotations

import json

import pandas as pd

import scripts.idea_validation_report as idea_report
from scripts.idea_validation_report import (
    ACTIVE,
    add_result_columns,
    build_input_manifest,
    build_report,
    combine_channel,
    input_manifest_matches_current,
    load_candidate_ledger,
    write_outputs,
)


def test_combine_channel_prefers_shadow_metadata_and_fills_paper_realized():
    paper = pd.DataFrame([
        {
            "date": "2026-05-25",
            "coin": "KRW-AAA",
            "setup_ids": "S01+S02+S03",
            "btc_regime": "bear_quiet",
            "next_max_return_pct": 6.0,
            "next_min_return_pct": -1.0,
            "next_close_return_pct": 2.0,
            "hit_h6": 1,
            "status": "closed",
        }
    ])
    shadow = pd.DataFrame([
        {
            "date": "2026-05-25",
            "channel": "distribution",
            "coin": "KRW-AAA",
            "decision": "WATCH_ONLY",
            "idea_id": "dist_test",
            "setup_quality": "A_TRIPLE",
            "btc_regime": "bear_quiet",
            "status": "entered",
        }
    ])

    combined = combine_channel(paper, shadow, "distribution")

    assert len(combined) == 1
    row = combined.iloc[0]
    assert row["decision"] == "WATCH_ONLY"
    assert row["idea_id"] == "dist_test"
    assert row["status"] == "closed"
    assert row["next_max_return_pct"] == 6.0


def test_build_report_scores_distribution_and_preopen_groups():
    candidates = pd.DataFrame([
        {
            "date": "2026-05-25",
            "channel": "distribution",
            "coin": "KRW-AAA",
            "decision": ACTIVE,
            "idea_id": "dist_a",
            "setup_quality": "A_TRIPLE",
            "btc_regime": "bear_quiet",
            "next_max_return_pct": 6.0,
            "next_min_return_pct": -1.0,
            "next_close_return_pct": 2.0,
            "hit_h6": 1,
            "status": "closed",
        },
        {
            "date": "2026-05-25",
            "channel": "preopen",
            "coin": "KRW-BBB",
            "decision": "WATCH_ONLY",
            "idea_id": "pre_b",
            "setup_quality": "PREOPEN",
            "btc_regime": "bear_quiet",
            "first_1h_max_return_pct": 1.0,
            "first_1h_min_return_pct": -2.0,
            "first_1h_close_return_pct": -1.0,
            "hit_first1h_5pct": 0,
            "status": "closed",
        },
        {
            "date": "2026-05-26",
            "channel": "distribution",
            "coin": "KRW-CCC",
            "decision": ACTIVE,
            "idea_id": "dist_c",
            "setup_quality": "C_PRIMARY",
            "btc_regime": "bear_quiet",
            "next_max_return_pct": 1.0,
            "next_min_return_pct": -5.0,
            "next_close_return_pct": -3.0,
            "hit_h6": 0,
            "status": "closed",
        },
    ])

    summary, payload = build_report(candidates, asof="2026-05-26")

    assert payload["n_candidates"] == 3
    assert payload["n_closed"] == 3
    dist_idea = summary[(summary["dimension"] == "idea") & (summary["idea_id"] == "dist_a")].iloc[0]
    pre_idea = summary[(summary["dimension"] == "idea") & (summary["idea_id"] == "pre_b")].iloc[0]
    replay_dist = summary[
        (summary["dimension"] == "policy_replay_active")
        & (summary["channel"] == "distribution")
    ].iloc[0]
    assert dist_idea["tp5_hit_rate_pct"] == 100.0
    assert dist_idea["net_pnl_sum_pct"] > 0
    assert pre_idea["tp5_hit_rate_pct"] == 0.0
    assert pre_idea["net_pnl_sum_pct"] < 0
    assert replay_dist["n_closed"] == 1
    replay_all = next(r for r in payload["policy_replay"]["rows"] if r["channel"] == "all")
    assert replay_all["observed_active"]["n_closed"] == 2
    assert replay_all["replay_active"]["n_closed"] == 1
    assert replay_all["delta_net_pnl_sum_pct"] > 0
    assert replay_all["replay_active"]["bootstrap"]["n"] == 1
    assert payload["policy_recommendations"]
    assert "policy_gate" in payload


def test_unordered_tp_and_sl_touch_is_diagnostic_only():
    candidates = pd.DataFrame([
        {
            "date": "2026-07-25",
            "channel": "distribution",
            "coin": "KRW-BOTH",
            "decision": ACTIVE,
            "setup_quality": "A_TRIPLE",
            "btc_regime": "bull_quiet",
            "next_max_return_pct": 6.0,
            "next_min_return_pct": -4.0,
            "next_close_return_pct": 2.0,
            "status": "closed",
            "path_complete": True,
        }
    ])

    row = add_result_columns(candidates).iloc[0]

    assert row["net_pnl_pct"] > 0
    assert not bool(row["promotion_eligible"])
    assert row["outcome_contract"] == "tp5_high_then_eod_proxy_unordered"


def test_ordered_first_passage_net_overrides_optimistic_tp_proxy():
    candidates = pd.DataFrame([
        {
            "date": "2026-07-25",
            "channel": "distribution",
            "coin": "KRW-SL-FIRST",
            "decision": ACTIVE,
            "setup_quality": "A_TRIPLE",
            "btc_regime": "bull_quiet",
            "next_max_return_pct": 6.0,
            "next_min_return_pct": -4.0,
            "next_close_return_pct": 2.0,
            "status": "closed",
            "path_complete": True,
            "exit_reason": "SL",
            "realized_pct": -3.15,
        }
    ])

    row = add_result_columns(candidates).iloc[0]

    assert row["net_pnl_pct"] == -3.15
    assert row["tp5_hit"] == 0
    assert bool(row["promotion_eligible"])
    assert row["outcome_contract"] == "tp5_sl3_ordered_first_passage_net"


def test_build_report_excludes_future_and_invalid_candidate_dates():
    candidates = pd.DataFrame([
        {
            "date": "2026-07-24",
            "channel": "distribution",
            "coin": "KRW-PAST",
            "decision": ACTIVE,
            "idea_id": "past",
            "setup_quality": "A_TRIPLE",
            "btc_regime": "bear_quiet",
            "next_max_return_pct": 6.0,
            "next_min_return_pct": -1.0,
            "next_close_return_pct": 2.0,
            "status": "closed",
        },
        {
            "date": "2026-07-26",
            "channel": "distribution",
            "coin": "KRW-FUTURE",
            "decision": ACTIVE,
            "idea_id": "future",
            "setup_quality": "A_TRIPLE",
            "btc_regime": "bear_quiet",
            "next_max_return_pct": 99.0,
            "next_min_return_pct": 0.0,
            "next_close_return_pct": 99.0,
            "status": "closed",
        },
        {
            "date": "not-a-date",
            "channel": "distribution",
            "coin": "KRW-INVALID",
            "decision": ACTIVE,
            "idea_id": "invalid",
            "setup_quality": "A_TRIPLE",
            "btc_regime": "bear_quiet",
            "next_max_return_pct": 99.0,
            "next_min_return_pct": 0.0,
            "next_close_return_pct": 99.0,
            "status": "closed",
        },
    ])

    summary, payload = build_report(candidates, asof="2026-07-25")

    assert payload["asof"] == "2026-07-25"
    assert payload["n_candidates"] == 1
    assert payload["n_closed"] == 1
    assert payload["cutoff_exclusions"] == {
        "future_date_rows": 1,
        "invalid_date_rows": 1,
    }
    assert set(summary["idea_id"].dropna()) == {"past"}


def test_build_report_excludes_future_trained_meta_artifact(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "meta.json").write_text(
        json.dumps(
            {
                "model_id": "future-model",
                "model_version": "1",
                "built_at": "2026-07-26T10:00:00+09:00",
                "date_range": {"start": "2026-01-01", "end": "2026-07-26"},
                "deployable": True,
            }
        ),
        encoding="utf-8",
    )

    _, payload = build_report(
        pd.DataFrame(),
        asof="2026-07-25",
        meta_model_dir=model_dir,
    )

    meta = payload["model_card"]["trained_meta_model"]
    assert meta["available"] is False
    assert "future-dated" in meta["reason"]


def test_build_report_never_labels_legacy_unbound_pickle_deployable(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "meta.json").write_text(
        json.dumps(
            {
                "model_id": "legacy-model",
                "model_version": "1",
                "built_at": "2026-07-24T10:00:00+09:00",
                "date_range": {"start": "2026-01-01", "end": "2026-07-24"},
                "deployable": True,
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "model.pkl").write_bytes(b"must not be executed")

    _, payload = build_report(
        pd.DataFrame(),
        asof="2026-07-25",
        meta_model_dir=model_dir,
    )

    meta = payload["model_card"]["trained_meta_model"]
    assert meta["available"] is True
    assert meta["deployable"] is False
    assert meta["declared_deployable"] is True
    assert meta["artifact_status"] == "LEGACY_UNBOUND"


def test_cli_helpers_load_and_write_outputs(tmp_path):
    paper_dist = tmp_path / "paper_dist.csv"
    paper_pre = tmp_path / "paper_pre.csv"
    shadow_dist = tmp_path / "shadow_dist.csv"
    shadow_pre = tmp_path / "shadow_pre.csv"
    out_csv = tmp_path / "summary.csv"
    out_json = tmp_path / "summary.json"

    pd.DataFrame([
        {
            "date": "2026-05-25",
            "coin": "KRW-AAA",
            "setup_ids": "S02",
            "btc_regime": "bull_quiet",
            "next_max_return_pct": 4.0,
            "next_min_return_pct": -1.0,
            "next_close_return_pct": 3.0,
            "status": "closed",
        }
    ]).to_csv(paper_dist, index=False)
    pd.DataFrame(columns=["date", "coin"]).to_csv(paper_pre, index=False)
    pd.DataFrame(columns=["date", "coin"]).to_csv(shadow_dist, index=False)
    pd.DataFrame(columns=["date", "coin"]).to_csv(shadow_pre, index=False)

    class Args:
        paper_ledger = str(paper_dist)
        paper_ledger_preopen = str(paper_pre)
        shadow_ledger_distribution = str(shadow_dist)
        shadow_ledger_preopen = str(shadow_pre)

    candidates = load_candidate_ledger(Args)
    lineage = build_input_manifest(
        Args,
        policy_competition_path=tmp_path / "policy.json",
        policy_db_path=tmp_path / "policy.db",
        meta_model_dir=tmp_path / "model",
    )
    summary, payload = build_report(
        candidates,
        asof="2026-05-25",
        input_manifest=lineage,
        policy_competition_path=tmp_path / "policy.json",
        meta_model_dir=tmp_path / "model",
    )
    write_outputs(summary, payload, out_csv, out_json)

    assert out_csv.exists()
    assert out_json.exists()
    raw_json = out_json.read_text()
    assert "NaN" not in raw_json
    assert "Infinity" not in raw_json
    loaded = json.loads(raw_json)
    assert loaded["n_candidates"] == 1


def test_input_manifest_binds_generator_source_bytes(
    monkeypatch,
    tmp_path,
):
    generator = tmp_path / "generator.py"
    generator.write_text("VERSION = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        idea_report,
        "IDEA_GENERATOR_SOURCES",
        (str(generator),),
    )

    class Args:
        paper_ledger = str(tmp_path / "paper.csv")
        paper_ledger_preopen = str(tmp_path / "preopen.csv")
        shadow_ledger_distribution = str(tmp_path / "shadow_dist.csv")
        shadow_ledger_preopen = str(tmp_path / "shadow_pre.csv")

    manifest = build_input_manifest(
        Args,
        policy_competition_path=tmp_path / "policy.json",
        policy_db_path=tmp_path / "policy.db",
        meta_model_dir=tmp_path / "model",
    )

    assert manifest["schema_version"] == 3
    assert input_manifest_matches_current(manifest) is True

    generator.write_text("VERSION = 2\n", encoding="utf-8")
    assert input_manifest_matches_current(manifest) is False

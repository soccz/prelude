from __future__ import annotations

import json

import pandas as pd

from scripts.idea_validation_report import (
    ACTIVE,
    build_report,
    combine_channel,
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

    summary, payload = build_report(candidates)

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
    summary, payload = build_report(candidates)
    write_outputs(summary, payload, out_csv, out_json)

    assert out_csv.exists()
    assert out_json.exists()
    loaded = json.loads(out_json.read_text())
    assert loaded["n_candidates"] == 1

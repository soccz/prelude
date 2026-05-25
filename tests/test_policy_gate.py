from __future__ import annotations

from ops.policy_gate import evaluate_policy_gate


def test_policy_gate_promotes_paper_only_until_shadow_confirms():
    payload = {
        "policy_replay": {
            "rows": [
                {
                    "channel": "distribution",
                    "policy_id": "setup_quality_policy_v1",
                    "observed_active": {"n_closed": 100, "net_pnl_sum_pct": -20.0},
                    "replay_active": {
                        "n_closed": 32,
                        "net_pnl_sum_pct": 70.0,
                        "avg_net_pnl_pct": 2.1,
                        "bootstrap": {"avg_pnl_ci95_pct": [0.5, 3.7]},
                    },
                    "replay_period_stability": [
                        {"period": "early", "avg_net_pnl_pct": 3.0},
                        {"period": "late", "avg_net_pnl_pct": 1.0},
                    ],
                    "delta_net_pnl_sum_pct": 90.0,
                }
            ]
        },
        "tables": {"summary": []},
    }

    gate = evaluate_policy_gate(payload)

    row = gate["rows"][0]
    assert row["state"] == "PROMOTE_PAPER"
    assert row["action"] == "ADOPT_ACTIVE_FILTER_IN_PAPER_AND_ALERTS"
    assert row["confidence"] == "MEDIUM_REPLAY_ONLY"


def test_policy_gate_demotes_negative_preopen():
    payload = {
        "policy_replay": {
            "rows": [
                {
                    "channel": "preopen",
                    "observed_active": {"n_closed": 88, "net_pnl_sum_pct": -40.0},
                    "replay_active": {"n_closed": 0, "net_pnl_sum_pct": None},
                    "replay_period_stability": [],
                    "delta_net_pnl_sum_pct": 40.0,
                }
            ]
        },
        "tables": {"summary": []},
    }

    gate = evaluate_policy_gate(payload)

    row = gate["rows"][0]
    assert row["state"] == "DEMOTE"
    assert row["action"] == "WATCH_ONLY"

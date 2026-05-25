from __future__ import annotations

import json

from scripts.build_idea_validation_html import build_html, render_html


def _payload():
    return {
        "generated_at": "2026-05-25T22:00:00",
        "round_trip_cost_pct": 0.15,
        "model_card": {
            "name": "prelude AI quant recommendation assistant",
            "version": "policy-gated-beta-2026-05-25",
            "intended_use": "Personal recommendations",
            "decision_layers": ["candidate generation", "meta-filter"],
            "risk_controls": ["ACTIVE-only Telegram"],
            "current_evidence": {"n_closed": 32, "net_pnl_sum_pct": 71.7, "tp5_hit_rate_pct": 62.5},
            "trained_meta_model": {
                "available": True,
                "model_id": "recommendation_quality_meta_label_v1",
                "deployable": False,
                "reason": "selected holdout too small",
                "holdout_metrics": {
                    "n": 41,
                    "positive_rate_pct": 24.39,
                    "auc": 0.516,
                    "average_precision": 0.365,
                    "brier": 0.228,
                    "net_pnl_sum_pct": -59.2,
                },
                "holdout_threshold_stats": {
                    "n_selected": 1,
                    "precision_pct": 100.0,
                    "net_pnl_sum_pct": 4.85,
                    "avg_net_pnl_pct": 4.85,
                },
                "top_coefficients": [
                    {"feature": "cat__setup_quality_A_TRIPLE", "coef": 0.801492}
                ],
            },
            "methodology_references": [{"label": "DSR", "url": "https://example.com"}],
        },
        "recommendation_quality": {
            "available": True,
            "rows": [
                {
                    "channel": "distribution",
                    "confidence_tier": "EVIDENCE_OK",
                    "n_candidates": 3,
                    "avg_confidence_score": 78.4,
                    "n_closed": 2,
                    "net_pnl_sum_pct": 4.0,
                    "tp5_hit_rate_pct": 50.0,
                }
            ],
        },
        "policy_gate": {
            "rows": [
                {
                    "channel": "distribution",
                    "state": "PROMOTE_PAPER",
                    "action": "ADOPT_ACTIVE_FILTER_IN_PAPER_AND_ALERTS",
                    "confidence": "MEDIUM_REPLAY_ONLY",
                    "replay_n_closed": 32,
                    "replay_net_pnl_sum_pct": 71.7472,
                    "delta_net_pnl_sum_pct": 116.774,
                    "replay_bootstrap_avg_ci95_low_pct": 0.8895,
                    "replay_late_avg_net_pnl_pct": 1.0977,
                    "reasons": ["reason"],
                }
            ]
        },
        "policy_replay": {
            "rows": [
                {
                    "channel": "distribution",
                    "observed_active": {"n_closed": 132, "net_pnl_sum_pct": -45.0268},
                    "replay_active": {"n_closed": 32, "net_pnl_sum_pct": 71.7472},
                    "delta_net_pnl_sum_pct": 116.774,
                }
            ]
        },
        "tables": {
            "summary": [
                {
                    "dimension": "idea",
                    "channel": "distribution",
                    "idea_id": "dist_test",
                    "setup_quality": "A_TRIPLE",
                    "btc_regime": "bear_quiet",
                    "n_closed": 32,
                    "tp5_hit_rate_pct": 62.5,
                    "net_pnl_sum_pct": 71.7,
                    "avg_net_pnl_pct": 2.2,
                    "evidence_tier": "PROMOTE_CANDIDATE",
                }
            ]
        },
    }


def test_render_html_contains_gate_and_idea_table():
    html = render_html(_payload())

    assert "Prelude Idea Validation" in html
    assert "PROMOTE_PAPER" in html
    assert "dist_test" in html
    assert "A_TRIPLE" in html
    assert "Model Card" in html
    assert "Recommendation Accuracy" in html
    assert "SHADOW" in html
    assert "cat__setup_quality_A_TRIPLE" in html
    assert "Recommendation Quality" in html
    assert "EVIDENCE_OK" in html


def test_build_html_writes_file(tmp_path):
    input_json = tmp_path / "idea.json"
    out_html = tmp_path / "idea.html"
    input_json.write_text(json.dumps(_payload()), encoding="utf-8")

    build_html(input_json, out_html)

    assert out_html.exists()
    assert "Policy Replay" in out_html.read_text(encoding="utf-8")

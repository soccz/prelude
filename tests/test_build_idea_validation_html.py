from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts.build_idea_validation_html import build_html, render_html
from scripts.idea_validation_report import (
    IDEA_ARTIFACT_SCHEMA,
    build_input_manifest,
    report_payload_digest,
)


def _payload():
    missing = "/definitely-missing/prelude-test-artifact"
    lineage = build_input_manifest(
        SimpleNamespace(
            paper_ledger=f"{missing}-paper.csv",
            paper_ledger_preopen=f"{missing}-preopen.csv",
            shadow_ledger_distribution=f"{missing}-shadow-dist.csv",
            shadow_ledger_preopen=f"{missing}-shadow-preopen.csv",
        ),
        policy_competition_path=f"{missing}-policy.json",
        policy_db_path=f"{missing}-policy.db",
        meta_model_dir=f"{missing}-model",
    )
    payload = {
        "schema": IDEA_ARTIFACT_SCHEMA,
        "asof": "2026-05-25",
        "input_lineage": lineage,
        "generated_at": "2026-05-25T22:00:00",
        "generated_at_utc": "2026-05-25T13:00:00+00:00",
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
                "artifact_status": "LEGACY_UNBOUND",
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
        "policy_competition": {
            "asof": "2026-05-25",
            "config": {"pump20_threshold": 0.20},
            "rows": [
                {
                    "asof": "2026-05-25",
                    "participant_id": "distribution_engine:top_all",
                    "objective": "baseline",
                    "n_closed": 147,
                    "n_days": 24,
                    "net_mean_pct": -0.7157,
                    "deep_loss_freq_pct": 31.2925,
                    "pump20_precision_pct": 10.2041,
                    "pump20_recall_pct": 17.8571,
                    "pump20_captured": 15,
                    "pump20_actual": 84,
                }
            ],
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
    payload["payload_sha256"] = report_payload_digest(payload)
    return payload


def test_render_html_contains_gate_and_idea_table():
    html = render_html(_payload())

    assert "Prelude Idea Validation" in html
    assert "PROMOTE_PAPER" in html
    assert "dist_test" in html
    assert "A_TRIPLE" in html
    assert "Model Card" in html
    assert "Recommendation Accuracy" in html
    assert "LEGACY_UNBOUND" in html
    assert "SHADOW" not in html
    assert "cat__setup_quality_A_TRIPLE" in html
    assert "Recommendation Quality" in html
    assert "EVIDENCE_OK" in html
    assert "Policy Competition" in html
    assert "distribution_engine:top_all" in html
    assert "15 / 84" in html


def test_build_html_writes_file(tmp_path):
    input_json = tmp_path / "idea.json"
    out_html = tmp_path / "idea.html"
    input_json.write_text(json.dumps(_payload()), encoding="utf-8")

    build_html(input_json, out_html, asof="2026-05-25")

    assert out_html.exists()
    assert "Policy Replay" in out_html.read_text(encoding="utf-8")


def test_build_html_replaces_output_with_invalid_page_for_future_artifact(tmp_path):
    input_json = tmp_path / "idea.json"
    out_html = tmp_path / "idea.html"
    payload = _payload()
    payload["asof"] = "2026-05-26"
    input_json.write_text(json.dumps(payload), encoding="utf-8")
    out_html.write_text("stale future metrics", encoding="utf-8")

    try:
        build_html(input_json, out_html, asof="2026-05-25")
    except ValueError as exc:
        assert "future" in str(exc)
    else:
        raise AssertionError("future artifact must fail closed")

    rendered = out_html.read_text(encoding="utf-8")
    assert "INVALID AS-OF ARTIFACT" in rendered
    assert "stale future metrics" not in rendered


def test_build_html_rejects_unattributed_legacy_artifact(tmp_path):
    input_json = tmp_path / "idea.json"
    out_html = tmp_path / "idea.html"
    payload = _payload()
    payload.pop("asof")
    input_json.write_text(json.dumps(payload), encoding="utf-8")

    try:
        build_html(input_json, out_html, asof="2026-05-25")
    except ValueError as exc:
        assert "asof is invalid" in str(exc)
    else:
        raise AssertionError("artifact without asof must fail closed")

    assert "INVALID AS-OF ARTIFACT" in out_html.read_text(encoding="utf-8")


def test_future_embedded_policy_artifact_is_excluded():
    payload = _payload()
    payload["policy_competition"]["asof"] = "2026-05-26"
    payload["payload_sha256"] = report_payload_digest(payload)

    rendered = render_html(payload, asof="2026-05-25")

    assert "distribution_engine:top_all" not in rendered
    assert "No policy competition artifact yet." in rendered


def test_checksum_tampering_replaces_report_with_invalid_page(tmp_path):
    input_json = tmp_path / "idea.json"
    out_html = tmp_path / "idea.html"
    payload = _payload()
    payload["n_closed"] = 999_999
    input_json.write_text(json.dumps(payload), encoding="utf-8")

    try:
        build_html(input_json, out_html, asof="2026-05-25")
    except ValueError as exc:
        assert "checksum mismatch" in str(exc)
    else:
        raise AssertionError("tampered artifact must fail closed")

    rendered = out_html.read_text(encoding="utf-8")
    assert "INVALID AS-OF ARTIFACT" in rendered
    assert "999999" not in rendered


def test_non_object_json_replaces_report_with_invalid_page(tmp_path):
    input_json = tmp_path / "idea.json"
    out_html = tmp_path / "idea.html"
    input_json.write_text("[]", encoding="utf-8")
    out_html.write_text("stale metrics", encoding="utf-8")

    try:
        build_html(input_json, out_html, asof="2026-05-25")
    except ValueError as exc:
        assert "JSON object" in str(exc)
    else:
        raise AssertionError("non-object artifact must fail closed")

    rendered = out_html.read_text(encoding="utf-8")
    assert "INVALID AS-OF ARTIFACT" in rendered
    assert "stale metrics" not in rendered


def test_duplicate_key_and_nonstandard_nan_are_rejected(tmp_path):
    payload = _payload()
    canonical = json.dumps(payload)
    malformed_documents = [
        canonical.replace(
            '"asof": "2026-05-25"',
            '"asof": "2026-05-25", "asof": "2026-05-25"',
            1,
        ),
        canonical.replace(
            '"round_trip_cost_pct": 0.15',
            '"round_trip_cost_pct": NaN',
            1,
        ),
    ]

    for index, raw in enumerate(malformed_documents):
        input_json = tmp_path / f"idea-{index}.json"
        out_html = tmp_path / f"idea-{index}.html"
        input_json.write_text(raw, encoding="utf-8")

        with pytest.raises(ValueError):
            build_html(input_json, out_html, asof="2026-05-25")

        assert "INVALID AS-OF ARTIFACT" in out_html.read_text(
            encoding="utf-8"
        )

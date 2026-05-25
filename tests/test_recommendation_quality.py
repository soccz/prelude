from __future__ import annotations

import pandas as pd
import numpy as np

from ops.decision_policy import ACTIVE, WATCH_ONLY
from ops.recommendation_quality import (
    apply_recommendation_quality,
    normalize_history,
)


def _active_candidate() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "market": "KRW-AAA",
            "decision": ACTIVE,
            "idea_id": "dist_h6_b_s03_bull_quiet_v1",
            "setup_quality": "B_S03",
            "btc_regime": "bull_quiet",
            "expected_edge_pct": 1.0,
            "decision_reason": "base active",
        }
    ])


class _FakeModel:
    def __init__(self, p_win: float):
        self.p_win = p_win

    def predict_proba(self, X):
        return np.array([[1.0 - self.p_win, self.p_win] for _ in range(len(X))])


def test_recommendation_quality_demotes_negative_matched_evidence():
    raw_history = pd.DataFrame([
        {
            "date": f"2026-05-{day:02d}",
            "coin": f"KRW-{day}",
            "decision": ACTIVE,
            "idea_id": "dist_h6_b_s03_bull_quiet_v1",
            "setup_quality": "B_S03",
            "btc_regime": "bull_quiet",
            "next_max_return_pct": 1.0,
            "next_close_return_pct": -2.0,
            "status": "closed",
        }
        for day in range(1, 9)
    ])
    history = normalize_history(raw_history, "distribution", "paper")

    out = apply_recommendation_quality(
        _active_candidate(),
        history,
        pd.Timestamp("2026-05-25 09:05"),
        "distribution",
    )

    row = out.iloc[0]
    assert row["base_decision"] == ACTIVE
    assert row["decision"] == WATCH_ONLY
    assert row["confidence_tier"] == "DOWNRANK"
    assert row["blocked_reason"] == "meta_filter_negative_evidence"
    assert row["evidence_n_closed"] == 8


def test_recommendation_quality_keeps_active_without_closed_history():
    out = apply_recommendation_quality(
        _active_candidate(),
        pd.DataFrame(),
        pd.Timestamp("2026-05-25 09:05"),
        "distribution",
    )

    row = out.iloc[0]
    assert row["decision"] == ACTIVE
    assert row["confidence_tier"] == "COLLECT"
    assert row["evidence_n_closed"] == 0


def test_recommendation_quality_ignores_same_day_history():
    raw_history = pd.DataFrame([
        {
            "date": "2026-05-25",
            "coin": f"KRW-{i}",
            "decision": ACTIVE,
            "idea_id": "dist_h6_b_s03_bull_quiet_v1",
            "setup_quality": "B_S03",
            "btc_regime": "bull_quiet",
            "next_max_return_pct": 1.0,
            "next_close_return_pct": -5.0,
            "status": "closed",
        }
        for i in range(8)
    ])
    history = normalize_history(raw_history, "distribution", "paper")

    out = apply_recommendation_quality(
        _active_candidate(),
        history,
        pd.Timestamp("2026-05-25 09:05"),
        "distribution",
    )

    assert out.iloc[0]["decision"] == ACTIVE
    assert out.iloc[0]["confidence_tier"] == "COLLECT"


def test_recommendation_quality_records_non_deployed_model_score_without_demoting():
    artifact = {
        "meta": {
            "model_id": "meta_test",
            "model_version": "v1",
            "deployable": False,
            "threshold": 0.6,
        },
        "model": _FakeModel(0.1),
    }

    out = apply_recommendation_quality(
        _active_candidate(),
        pd.DataFrame(),
        pd.Timestamp("2026-05-25 09:05"),
        "distribution",
        trained_model=artifact,
    )

    row = out.iloc[0]
    assert row["decision"] == ACTIVE
    assert row["trained_model_status"] == "COLLECT"
    assert row["trained_model_p_win"] == 0.1


def test_recommendation_quality_deployed_model_can_downrank():
    artifact = {
        "meta": {
            "model_id": "meta_test",
            "model_version": "v1",
            "deployable": True,
            "threshold": 0.6,
        },
        "model": _FakeModel(0.1),
    }

    out = apply_recommendation_quality(
        _active_candidate(),
        pd.DataFrame(),
        pd.Timestamp("2026-05-25 09:05"),
        "distribution",
        trained_model=artifact,
    )

    row = out.iloc[0]
    assert row["decision"] == WATCH_ONLY
    assert row["confidence_tier"] == "MODEL_DOWNRANK"
    assert row["blocked_reason"] == "trained_meta_model_downrank"

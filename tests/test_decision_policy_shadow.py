from __future__ import annotations

import pandas as pd

from ledger.shadow import append_shadow_ledger
from ops.decision_policy import (
    ACTIVE,
    SILENCE,
    WATCH_ONLY,
    active_only,
    apply_distribution_policy,
    apply_preopen_policy,
    setup_quality,
)


def test_setup_quality_tiers():
    assert setup_quality(["S01", "S02", "S03"]) == "A_TRIPLE"
    assert setup_quality(["S02", "S03"]) == "B_S03"
    assert setup_quality(["S02"]) == "C_PRIMARY"
    assert setup_quality([]) == "NO_PRIMARY"


def test_distribution_policy_replaces_bear_hard_silence_with_watch_and_active():
    candidates = pd.DataFrame([
        {
            "market": "KRW-AAA",
            "primary_setups": ["S01", "S02", "S03"],
            "btc_regime": "bear_quiet",
            "p_h6_hit_5_24h": 0.90,
            "composite": 3.0,
            "alert_rank": 1,
        },
        {
            "market": "KRW-BBB",
            "primary_setups": ["S02"],
            "btc_regime": "bear_quiet",
            "p_h6_hit_5_24h": 0.90,
            "composite": 2.0,
            "alert_rank": 2,
        },
    ])

    decisions = apply_distribution_policy(candidates, btc_regime="bear_quiet")

    assert decisions["decision"].tolist() == [ACTIVE, WATCH_ONLY]
    assert decisions["setup_quality"].tolist() == ["A_TRIPLE", "C_PRIMARY"]
    assert active_only(decisions)["market"].tolist() == ["KRW-AAA"]


def test_distribution_policy_silences_bear_volatile():
    candidates = pd.DataFrame([
        {
            "market": "KRW-AAA",
            "primary_setups": ["S01", "S02", "S03"],
            "p_h6_hit_5_24h": 0.90,
            "composite": 3.0,
        },
    ])

    decisions = apply_distribution_policy(candidates, btc_regime="bear_volatile")

    assert decisions["decision"].tolist() == [SILENCE]
    assert decisions["blocked_reason"].tolist() == ["btc_bear_volatile"]
    assert active_only(decisions).empty


def test_preopen_policy_is_watch_first_in_bear_quiet():
    candidates = pd.DataFrame([
        {
            "market": "KRW-AAA",
            "p_first1h_5pct": 0.80,
            "composite": 2.0,
            "alert_rank": 1,
        },
    ])

    decisions = apply_preopen_policy(candidates, btc_regime="bear_quiet")

    assert decisions["decision"].tolist() == [WATCH_ONLY]
    assert active_only(decisions).empty


def test_shadow_ledger_keeps_one_snapshot_per_date_channel(tmp_path):
    candidates = apply_distribution_policy(
        pd.DataFrame([
            {
                "market": "KRW-AAA",
                "primary_setups": ["S01", "S02", "S03"],
                "btc_regime": "bear_quiet",
                "p_h6_hit_5_24h": 0.90,
                "p_h2_hit_3_4h": 0.80,
                "p_h5_tail_20": 0.20,
                "composite": 3.0,
                "alert_rank": 1,
            }
        ]),
        btc_regime="bear_quiet",
    )
    path = tmp_path / "shadow.csv"
    asof = pd.Timestamp("2026-05-25 09:05")

    assert append_shadow_ledger(candidates, asof, str(path), "distribution") == 1
    assert append_shadow_ledger(candidates, asof, str(path), "distribution") == 0

    ledger = pd.read_csv(path)
    assert ledger[["date", "channel", "coin", "decision", "idea_id"]].to_dict("records") == [
        {
            "date": "2026-05-25",
            "channel": "distribution",
            "coin": "KRW-AAA",
            "decision": ACTIVE,
            "idea_id": "dist_h6_a_triple_bear_quiet_v1",
        }
    ]

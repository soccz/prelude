from __future__ import annotations

import pandas as pd

from signals.model_registry import get_model
from signals.pump_detector_v1 import apply_pump_rules


def test_pump_rule_detector_fires_on_mined_roc_and_atr_rule():
    frame = pd.DataFrame([
        {
            "market": "KRW-A",
            "roc_7d_rank": 0.86,
            "atr_pct_14_rank": 0.50,
            "atr_pct_14": 0.05,
            "log_return_1d": 0.02,
        },
        {
            "market": "KRW-B",
            "roc_7d_rank": 0.90,
            "atr_pct_14_rank": 0.90,
            "atr_pct_14": 0.08,
            "log_return_1d": 0.05,
        },
        {
            "market": "KRW-C",
            "roc_7d_rank": 0.70,
            "atr_pct_14_rank": 0.95,
            "atr_pct_14": 0.09,
            "log_return_1d": 0.01,
        },
    ])

    out = apply_pump_rules(frame, max_candidates=20)

    assert out["market"].tolist() == ["KRW-B", "KRW-A"]
    assert out.loc[out["market"] == "KRW-A", "pump20_rule"].item()
    assert out.loc[out["market"] == "KRW-B", "pump15_rule"].item()
    assert "KRW-C" not in set(out["market"])


def test_model_registry_has_pump_hunter_as_challenger_only():
    spec = get_model("pump_hunter")

    assert spec.ledger_path == "output/shadow_ledger_pump_hunter.csv"
    assert spec.slots == ["open"]
    assert spec.challenger_only is True
    assert spec.predict_ref == "signals.pump_detector_v1:score_pump_candidates"


def test_pump_detector_ledger_append_is_idempotent(tmp_path, monkeypatch):
    import scripts.pump_detector_today as mod

    def fake_score(*args, **kwargs):
        return {
            "asof": "2026-06-04",
            "feature_date": "2026-06-03",
            "model_id": "pump_hunter",
            "rule_version": "pump_detector_v1",
            "universe_n": 100,
            "n_candidates": 1,
            "candidates": [
                {
                    "market": "KRW-PUMP",
                    "rank": 1,
                    "score": 0.93,
                    "estimated_pump20_prob": 0.064,
                    "dump_risk_flag": False,
                    "btc_regime": "bull_quiet",
                    "entry_open": 100.0,
                    "rule_id": "roc7_rank_pump20",
                    "liq_rank_daily": 12,
                    "roc_7d": 55.0,
                    "roc_7d_rank": 0.91,
                    "atr_pct_14": 0.08,
                    "log_return_1d": 0.04,
                    "pump20_rule": True,
                    "pump15_rule": False,
                    "estimated_pump15_prob": None,
                    "overheated_flag": False,
                }
            ],
        }

    monkeypatch.setattr(mod, "score_pump_candidates", fake_score)
    ledger = tmp_path / "shadow_ledger_pump_hunter.csv"

    mod.append_today("2026-06-04", ledger_path=str(ledger))
    mod.append_today("2026-06-04", ledger_path=str(ledger))

    df = pd.read_csv(ledger)
    assert len(df) == 1
    assert df.iloc[0]["coin"] == "KRW-PUMP"
    assert df.iloc[0]["model_id"] == "pump_hunter"
    assert df.iloc[0]["status"] == "open"

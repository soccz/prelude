"""Stored prediction display only: no inference, re-ranking or imputation."""
import numpy as np
import pandas as pd
import pytest

from scripts.build_dashboard import _recommend_prediction_evidence, compute_recommend_summary


def test_projection_keeps_stored_ratio_and_original_rank_with_asof():
    rows = [dict(date="2026-10-01", coin="KRW-A", rank=2, status="open",
                 score=.99, p_up10=.2764, p_dn5=.3236, p_dn10=.0258,
                 rr_ratio=.8542, dump_risk_flag=None),
            dict(date="2026-10-01", coin="KRW-B", rank=1, status="open",
                 score=.1, p_up10=.2764, p_dn5=.2686, p_dn10=.0344,
                 rr_ratio=1.0291, dump_risk_flag="False"),
            dict(date="2026-10-02", coin="KRW-FUTURE", rank=1, status="open")]
    ledger = pd.DataFrame(rows)
    before = ledger.copy(deep=True)
    radar = compute_recommend_summary(ledger, asof="2026-10-01")["latest_radar"]
    assert [(r["coin"], r["rank"]) for r in radar] == [("B", 1), ("A", 2)]
    assert radar[0]["prediction_evidence"]["rr_ratio"] == 1.0291
    assert radar[0]["prediction_evidence"]["rr_ratio"] != .2764 / .2686
    assert radar[0]["dump_risk_flag"] is False
    assert radar[1]["dump_risk_flag"] is None
    pd.testing.assert_frame_equal(ledger, before)


@pytest.mark.parametrize("value", [None, "", "nan", np.nan, np.inf, -np.inf,
                                     True, False, np.bool_(True), -.01, 1.001,
                                     "25%", "1,0", {}, []])
def test_invalid_probabilities_are_unavailable_not_zero(value):
    out = _recommend_prediction_evidence({k: value for k in ("p_up10", "p_dn5", "p_dn10")})
    assert all(out[k] is None for k in ("p_up10", "p_dn5", "p_dn10"))
    assert out["rr_ratio"] is None


def test_zero_boundary_legacy_missing_and_private_fields():
    out = _recommend_prediction_evidence(dict(p_up10="0", p_dn5="1", p_dn10=0,
                                               rr_ratio=0, snapshot_path="private", token="secret"))
    assert [out[k] for k in ("p_up10", "p_dn5", "p_dn10", "rr_ratio")] == [0, 1, 0, 0]
    assert set(out) == {"schema", "source", "basis", "p_up10", "p_dn5", "p_dn10", "rr_ratio"}
    assert _recommend_prediction_evidence({})["p_up10"] is None
    assert out["basis"] == "day_D_0900_KST_open_high_low"


def test_inconsistent_downside_is_not_silently_repaired():
    out = _recommend_prediction_evidence(dict(p_up10=.2, p_dn5=.1, p_dn10=.3, rr_ratio=2))
    assert out["p_up10"] == .2
    assert all(out[k] is None for k in ("p_dn5", "p_dn10", "rr_ratio"))

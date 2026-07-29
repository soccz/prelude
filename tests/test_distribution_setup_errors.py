from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import pandas as pd
import pytest

from signals.distribution_engine import DistributionEngine
from signals.setups import SETUP_LIBRARY, detect_setups


def _engine(rule: Callable[[Any], bool]) -> DistributionEngine:
    return DistributionEngine(
        models={},
        feature_cols=[],
        head_meta={},
        setup_library={"S01": {"detect_fn": rule}},
    )


def _scored(markets: tuple[str, ...] = ("KRW-BAD", "KRW-GOOD")) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "market": market,
                "liq_rank_daily": rank,
                "p_h2_hit_3_4h": 0.5,
                "p_h5_tail_20": 0.1,
                "p_h6_hit_5_24h": 0.4,
            }
            for rank, market in enumerate(markets, start=1)
        ]
    )


def test_normal_setup_matches_and_diagnose_counts_are_unchanged() -> None:
    row = pd.Series(
        {
            "market": "KRW-GOOD",
            "atr_pct_14": 0.07,
            "log_return_1d": 0.06,
            "roc_3d": 0.95,
            "vol_5d": 0.90,
            "return_7d": 0.95,
            "btc_regime": "bull_quiet",
        }
    )

    assert detect_setups(row) == ["S01", "S02", "S03", "S04"]

    engine = DistributionEngine(
        models={},
        feature_cols=[],
        head_meta={},
        setup_library=SETUP_LIBRARY,
    )
    diagnose = engine.diagnose(pd.DataFrame([row]))

    assert diagnose["setup_fire_counts"] == {
        "S01": 1,
        "S02": 1,
        "S03": 1,
        "S04": 1,
    }
    assert diagnose["setup_error_counts"] == {
        "S01": 0,
        "S02": 0,
        "S03": 0,
        "S04": 0,
    }


def test_partial_rule_errors_keep_healthy_results_and_are_run_local(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def partial_rule(row: pd.Series) -> bool:
        if row["market"] == "KRW-BAD":
            raise KeyError("schema drift")
        return True

    engine = _engine(partial_rule)
    scored = _scored()

    with caplog.at_level(logging.WARNING, logger="signals.setups"):
        diagnose = engine.diagnose(scored)

    assert diagnose["setup_fire_counts"] == {"S01": 1}
    assert diagnose["setup_error_counts"] == {"S01": 1}
    assert "setup rule S01 failed for 1/2 rows" in caplog.text

    alerts = engine.assemble_alerts(scored, require_setup=False)
    setups_by_market = alerts.set_index("market")["setups"].to_dict()
    assert setups_by_market == {
        "KRW-BAD": [],
        "KRW-GOOD": ["S01"],
    }

    # A subsequent evaluation gets a fresh counter; the prior error cannot leak.
    next_diagnose = engine.diagnose(_scored(("KRW-GOOD",)))
    assert next_diagnose["setup_error_counts"] == {"S01": 0}
    assert next_diagnose["setup_fire_counts"] == {"S01": 1}


@pytest.mark.parametrize("method_name", ("diagnose", "assemble_alerts"))
def test_rule_failing_on_every_evaluated_row_fails_loud(
    method_name: str,
) -> None:
    def broken_rule(_row: pd.Series) -> bool:
        raise KeyError("renamed_feature")

    engine = _engine(broken_rule)

    with pytest.raises(
        RuntimeError,
        match=r"setup rule failed for every evaluated row: "
        r"S01=2/2 \(KeyError: 'renamed_feature'\)",
    ):
        getattr(engine, method_name)(_scored())

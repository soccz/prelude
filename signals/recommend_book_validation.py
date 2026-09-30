"""Fixed-rule new-date replay; never claim pre-entry recording or promotion."""

from datetime import date, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from signals import recommend_book_top3 as fixed
from signals.recommend_experiment_eval import (
    METRICS,
    _metric_rows,
    _named,
    _summary,
    _validate_outcomes,
)

START, END = date(2026, 10, 1), date(2026, 10, 30)
CONFIG = {
    "schema": "recommend_book_validation_config.v1",
    "start_date": START.isoformat(),
    "end_date": END.isoformat(),
    "slot": "open",
    "selector": fixed.CONFIG.copy(),
    "evaluation_kind": "fixed_rule_out_of_time_replay_not_pre_entry_record",
    "round_trip_cost_fraction": 0.0015,
    "new_selectors": 0,
    "n_boot": 1000,
    "seed": 42,
    "automatic_promotion": False,
    "places_orders": False,
    "max_new_dates_per_run": 2,
}


def due_dates(now):
    fixed.book._require(now.utcoffset() is not None, "aware observation time required")
    through = min(
        END, now.astimezone(ZoneInfo("Asia/Seoul")).date() - timedelta(days=1)
    )
    return [
        (START + timedelta(days=i)).isoformat()
        for i in range(max(0, (through - START).days + 1))
    ]


def evaluate_day(day, predictors, outcomes, *, entry_at, end_at):
    """Fix the selection before validating/joining outcomes; never reselect."""
    start, end = pd.Timestamp(entry_at), pd.Timestamp(end_at)
    fixed.book._require(
        start.tzinfo is not None
        and end.tzinfo is not None
        and start.tz_convert("Asia/Seoul").date().isoformat() == day
        and end - start == pd.Timedelta(days=1),
        "comparison requires the same-date full 24h interval",
    )
    plan = fixed.select_top3(predictors)
    frame = pd.DataFrame(outcomes).set_index("coin")
    fixed.book._require(
        frame.index.is_unique and set(frame.index) == {r["coin"] for r in predictors},
        "label universe mismatch",
    )
    _validate_outcomes(frame)
    result = {"date": day, "entry_at": entry_at, "end_at": end_at, "selection": plan}
    if not frame.label_status.eq("labeled").all():
        return {
            **result,
            "status": "excluded",
            "reason": "incomplete_full_universe_labels",
            "metrics": None,
        }
    vectors = {
        arm: _metric_rows(frame, plan[arm]).mean(0) for arm in ("control", "challenger")
    }
    vectors["matched_random_expectation"] = sum(
        _metric_rows(frame, p["coins"]).mean(0) * p["quota"] / 3 for p in plan["pools"]
    )
    vectors["full_universe"] = _metric_rows(frame, sorted(frame.index)).mean(0)
    for baseline in ("control", "matched_random_expectation"):
        vectors[f"challenger_minus_{baseline}"] = (
            vectors["challenger"] - vectors[baseline]
        )
    return {
        **result,
        "status": "evaluated",
        "reason": None,
        "metrics": {arm: _named(v) for arm, v in vectors.items()},
    }


def summarize(daily):
    fixed.book._require(
        len({d["date"] for d in daily}) == len(daily), "duplicate comparison dates"
    )
    fixed.book._require(
        all(d["status"] in {"evaluated", "excluded"} for d in daily),
        "unknown comparison status",
    )
    records = sorted(
        (d for d in daily if d["status"] == "evaluated"), key=lambda d: d["date"]
    )
    n = len(records)
    result = {
        "paired_dates": n,
        "picks_per_arm": n * 3,
        "changed_dates": sum(d["selection"]["changed_picks"] > 0 for d in records),
        "changed_picks": sum(d["selection"]["changed_picks"] for d in records),
        "outside_top10": sum(d["selection"]["selected_outside_top10"] for d in records),
        "metrics": None,
        "basket_proxy": None,
    }
    result["no_op_dates"] = n - result["changed_dates"]
    if records:
        result["metrics"] = {
            arm: _summary(
                np.array([[r["metrics"][arm][m] for m in METRICS] for r in records]),
                CONFIG["n_boot"],
                CONFIG["seed"],
            )
            for arm in records[0]["metrics"]
        }
        result["basket_proxy"] = {
            arm: fixed._basket(records, arm) for arm in ("control", "challenger")
        }
    return result

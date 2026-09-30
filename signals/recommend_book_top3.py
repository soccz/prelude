"""One offline L1 selector preserving R1's liquidity/ATR cell counts.

The rule is outcome-blind. This is development replay, not a live policy.
"""
from collections import Counter

import numpy as np
import pandas as pd

from ledger.portfolio_metrics import summarize_daily
from signals import recommend_book_pressure as book
from signals.recommend_experiment_eval import METRICS, _metric_rows, _named, _summary, _validate_outcomes

CONFIG = {
    "schema": "recommend_book_top3.v1", "start_date": "2026-09-10", "end_date": "2026-09-29",
    "split_date": "2026-09-20", "slot": "open", "universe_size": 100, "top_k": 3,
    "cells": "daily_log_qv_ATR_average_rank_quartiles",
    "quota": "preserve_original_R1_Top3_count_per_cell",
    "selection": "highest_endpoint_L1_pressure_without_replacement_per_cell",
    "ties": "original_Top3_members_first_then_original_rank",
    "matched_baseline": "exact_expected_mean_uniform_without_replacement_same_cell_quotas",
    "missing": "exclude_whole_date_not_reselect_or_zero_fill",
    "new_selectors": 1, "parameter_sweep": False, "model_fitted": False,
    "is_untouched_holdout": False, "deployable": False,
}


def select_top3(rows):
    # Reuse the frozen endpoint audit's complete predictor/identity guards.
    # Its diagnostic pairs are NOT used to select recommendations.
    book.pairing_plan(rows)
    frame = pd.DataFrame(rows).sort_values("rank").set_index("coin")
    quartiles = {f: np.clip(np.ceil(frame[f].rank(method="average", pct=True)*4).astype(int)-1, 0, 3)
                 for f in ("f_log_qv", "f_atr_pct_14")}
    cells = {c: tuple(int(q.at[c]) for q in quartiles.values()) for c in frame.index}
    control = list(frame.index[:3])
    quotas = Counter(cells[c] for c in control)
    selected, pools = [], []
    for cell, count in sorted(quotas.items()):
        pool = [c for c in frame.index if cells[c] == cell]
        ordered = sorted(pool, key=lambda c: (-frame.at[c, "pressure"], c not in control, frame.at[c, "rank"]))
        selected.extend(ordered[:count])
        pools.append({"cell": list(cell), "quota": count, "coins": pool})
    selected.sort(key=lambda c: frame.at[c, "rank"])
    book._require(len(set(selected)) == 3 and Counter(cells[c] for c in selected) == quotas, "quota mismatch")
    return {"control": control, "challenger": selected, "pools": pools,
            "selected_original_ranks": [int(frame.at[c, "rank"]) for c in selected],
            "changed_picks": len(set(selected)-set(control)),
            "selected_outside_top10": int(sum(frame.at[c, "rank"] > 10 for c in selected))}


def _basket(records, arm):
    """Only contiguous nonoverlapping 24h returns can be chained; never fill gaps."""
    if not records:
        return {"status": "no_dates", "metrics": None}
    intervals = []
    for row in records:
        try:
            start, end = (pd.Timestamp(row[k]) for k in ("entry_at", "end_at"))
        except (KeyError, TypeError, ValueError):
            return {"status": "unproven_intervals", "metrics": None}
        if start.tzinfo is None or end.tzinfo is None or end-start != pd.Timedelta(days=1):
            return {"status": "unproven_intervals", "metrics": None}
        if start.tz_convert("Asia/Seoul").date().isoformat() != row["date"]:
            return {"status": "unproven_intervals", "metrics": None}
        intervals.append((start, end))
    if any(a[1] != b[0] for a, b in zip(intervals, intervals[1:])):
        return {"status": "gaps_or_overlaps", "metrics": None}
    values = pd.Series([r["metrics"][arm]["eod_return_net"] for r in records],
                       index=pd.to_datetime([r["date"] for r in records]))
    book._require(np.isfinite(values).all() and (values > -1).all(), "invalid basket return")
    return {"status": "hypothetical_daily_rebalanced_equal3_not_actual_portfolio",
            "metrics": {**summarize_daily(values), "mean_daily_net": float(values.mean()),
                        "positive_day_rate": float((values > 0).mean()), "periods_per_year": 365}}


def evaluate(days, *, n_boot=1000, seed=42):
    book._require(type(n_boot) is int and 1 <= n_boot <= 100_000, "invalid bootstrap count")
    book._require(type(seed) is int and 0 <= seed < 2**64, "invalid seed")
    expected = pd.date_range(CONFIG["start_date"], CONFIG["end_date"]).strftime("%Y-%m-%d").tolist()
    index = {d["date"]: d for d in days}
    book._require(len(index) == len(days) and set(index) <= set(expected), "duplicate or outside dates")
    daily, excluded = [], []
    for date in expected:
        day = index.get(date, {"unavailable_reason": "missing_evidence"})
        rows = day.get("rows", [])
        reason = day.get("unavailable_reason")
        if reason or any(r.get(k) is None for r in rows for k in book.PREDICTORS):
            excluded.append({"date": date, "reason": reason or "missing_predictor"})
            continue
        plan = select_top3(rows)  # All original candidates, before inspecting any outcome.
        frame = pd.DataFrame(rows).set_index("coin")
        _validate_outcomes(frame)
        if not frame.label_status.eq("labeled").all():
            excluded.append({"date": date, "reason": "incomplete_full_universe_labels", "selection": plan})
            continue
        vectors = {arm: _metric_rows(frame, plan[arm]).mean(axis=0) for arm in ("control", "challenger")}
        vectors["matched_random_expectation"] = sum(
            _metric_rows(frame, pool["coins"]).mean(axis=0)*pool["quota"]/3 for pool in plan["pools"])
        vectors["full_universe"] = _metric_rows(frame, list(frame.index)).mean(axis=0)
        for baseline in ("control", "matched_random_expectation"):
            vectors[f"challenger_minus_{baseline}"] = vectors["challenger"]-vectors[baseline]
        added = [c for c in plan["challenger"] if c not in plan["control"]]
        removed = [c for c in plan["control"] if c not in plan["challenger"]]
        changed = {s: None if not picks else _named(_metric_rows(frame, picks).mean(axis=0))
                   for s, picks in (("added", added), ("removed", removed))}
        daily.append({"date": date, "entry_at": day.get("entry_at"), "end_at": day.get("end_at"),
                      "selection": plan, "metrics": {arm: _named(v) for arm, v in vectors.items()},
                      "replacement_metrics": changed,
                      "context": {arm: {k: float(frame.loc[plan[arm], k].mean()) for k in book.PREDICTORS}
                                  for arm in ("control", "challenger")}})
    segments = {}
    for name, records in (("earlier", [d for d in daily if d["date"] < CONFIG["split_date"]]),
                          ("later", [d for d in daily if d["date"] >= CONFIG["split_date"]]), ("all", daily)):
        segment = {"dates": len(records), "picks_per_arm": len(records)*3,
                   "changed_dates": sum(r["selection"]["changed_picks"] > 0 for r in records),
                   "changed_picks": sum(r["selection"]["changed_picks"] for r in records),
                   "selected_outside_top10": sum(r["selection"]["selected_outside_top10"] for r in records)}
        segment["no_op_dates"] = len(records)-segment["changed_dates"]
        if records:
            arrays = {arm: np.array([[r["metrics"][arm][m] for m in METRICS] for r in records])
                      for arm in records[0]["metrics"]}
            segment["metrics"] = {arm: _summary(a, n_boot, seed) for arm, a in arrays.items()}
            delta = arrays["challenger_minus_control"]
            segment["leave_one_date_out"] = [{"omitted_date": r["date"], "mean": _named(np.delete(delta,i,axis=0).mean(0))}
                                               for i,r in enumerate(records)] if len(records) > 1 else []
            segment["basket_proxy"] = {arm: _basket(records, arm) for arm in ("control", "challenger")}
            changed = [r for r in records if r["selection"]["changed_picks"]]
            segment["replacement_date_equal_means"] = {
                arm: _named(np.array([[r["replacement_metrics"][arm][m] for m in METRICS] for r in changed]).mean(0))
                if changed else None for arm in ("added", "removed")}
        segments[name] = segment
    return {"configuration": CONFIG.copy(), "daily": daily, "excluded": excluded, "segments": segments,
            "deployable": False, "is_untouched_holdout": False, "model_fitted": False,
            "interpretation": ["One rule chosen after the prior observational study on these same dates.",
                               "No fit or tuning; not fresh out-of-sample or prospective policy evidence.",
                               "Cell quotas are coarse context matching, not downside guarantees.",
                               "Existing net labels subtract 0.15% once; actual execution remains unverified.",
                               "Random baseline is an exact expected daily mean, not a sampled portfolio."]}

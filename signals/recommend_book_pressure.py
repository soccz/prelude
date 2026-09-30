"""Offline L1 resting-quote information audit, never a production selector."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from signals.recommend_experiment_eval import METRICS, _metric_rows, _named, _summary
from signals.recommend_spread_diagnostics import OUTCOMES, _finite, _require

CONFIG = {
    "schema": "recommend_book_pressure.v1", "start_date": "2026-09-10", "end_date": "2026-09-29",
    "split_date": "2026-09-20", "universe_size": 100, "slot": "open",
    "feature": "endpoint_L1_bid_minus_ask_notional_over_total",
    "cutoff": "last_ingress_with_event_and_receive_strictly_before_decision",
    "direction": "high_minus_low", "matching": "daily_ATR_log_qv_average_rank_quartiles",
    "pairing": "ascending_pressure_then_original_rank_outside_in_disjoint",
    "ties_and_odd_middle": "omit", "missing": "exclude_whole_date_not_zero",
    "weighting": "equal_pairs_within_date_equal_dates_between_dates",
    "new_feature_hypotheses": 1, "parameter_sweep": False, "is_untouched_holdout": False,
    "deployable": False,
}
PREDICTORS = ("rank", "pressure", "l1_notional", "f_log_qv", "f_atr_pct_14",
              "spread_fraction", "book_event_age_seconds")


def endpoint_quotes(records, *, cutoff, coins):
    """Consume validated (envelope,payload) pairs; do not deduplicate timestamps."""
    _require(type(cutoff) is int and cutoff > 0, "invalid integer cutoff")
    last = {}
    for record, payload in records:
        _require(payload is not None, "invalid raw book event")
        event, received = record["event_at_ms"], record["received_at_ns"]
        _require(type(event) is int and type(received) is int, "invalid integer clock")
        if event * 1_000_000 >= cutoff or received >= cutoff or record["market"] not in coins:
            continue
        units = payload["orderbook_units"]
        _require(len(units) == 1, "experiment requires depth1")
        unit = units[0]
        for field in ("bid_price", "ask_price", "bid_size", "ask_size"):
            _require(_finite(unit[field]) and unit[field] > 0, "invalid L1 quote")
        bid, ask = unit["bid_price"], unit["ask_price"]
        _require(bid < ask, "crossed or locked book")
        buy, sell = bid * unit["bid_size"], ask * unit["ask_size"]
        total = buy + sell
        _require(math.isfinite(total) and total > 0, "invalid L1 notional")
        last[record["market"]] = {
            "pressure": (buy-sell)/total, "l1_notional": total,
            "spread_fraction": (ask-bid)/((ask+bid)/2),
            "book_event_age_seconds": (cutoff-event*1_000_000)/1_000_000_000,
            "event_at_ms": event, "received_at_ns": received, "ingress_seq": record["ingress_seq"],
        }
    return last


def pairing_plan(rows):
    """No access to labels; stable identities, no replacement or cutoff search."""
    _require(len(rows) == CONFIG["universe_size"], "incomplete original universe")
    _require(len({r["coin"] for r in rows}) == len(rows)
             and all(isinstance(r["coin"], str) and r["coin"] for r in rows), "invalid identities")
    for row in rows:
        _require(all(_finite(row.get(k)) for k in PREDICTORS), "invalid predictor")
        _require(-1 < row["pressure"] < 1 and row["l1_notional"] > 0
                 and 0 <= row["spread_fraction"] < 2 and row["book_event_age_seconds"] > 0
                 and row["f_atr_pct_14"] >= 0, "invalid predictor range")
    _require(sorted(r["rank"] for r in rows) == list(range(1, 101)), "invalid original rank")
    frame = pd.DataFrame(rows).set_index("coin")
    cells = pd.DataFrame({f: np.clip(np.ceil(frame[f].rank(method="average", pct=True)*4).astype(int)-1, 0, 3)
                          for f in ("f_log_qv", "f_atr_pct_14")})
    pairs, ties, middle = [], 0, 0
    for cell, group in cells.groupby(list(cells.columns), sort=True):
        ordered = sorted(group.index, key=lambda c: (frame.at[c, "pressure"], frame.at[c, "rank"]))
        middle += len(ordered) % 2
        for i in range(len(ordered)//2):
            low, high = ordered[i], ordered[-i-1]
            if frame.at[low, "pressure"] == frame.at[high, "pressure"]:
                ties += 1
            else:
                pairs.append({"high": high, "low": low, "cell": [int(v) for v in cell]})
    return {"pairs": pairs, "equal_pressure_pairs": ties, "unpaired_middle_rows": middle,
            "observed_cells": len(cells.drop_duplicates())}


def analyze(days, *, n_boot=1000, seed=42):
    _require(type(n_boot) is int and 1 <= n_boot <= 100_000, "invalid bootstrap count")
    _require(type(seed) is int and 0 <= seed < 2**64, "invalid seed")
    expected = pd.date_range(CONFIG["start_date"], CONFIG["end_date"]).strftime("%Y-%m-%d").tolist()
    indexed = {d["date"]: d for d in days}
    _require(len(indexed) == len(days) and set(indexed) <= set(expected), "duplicate or outside date")
    daily, excluded = [], []
    for date in expected:
        day = indexed.get(date, {"unavailable_reason": "missing_evidence"})
        reason = day.get("unavailable_reason")
        rows = day.get("rows", [])
        if reason or any(r.get(k) is None for r in rows for k in PREDICTORS):
            excluded.append({"date": date, "reason": reason or "missing_predictor"})
            continue
        plan = pairing_plan(rows)
        _require(all(r["label_status"] in {"labeled", "halted_no_observations"} for r in rows), "invalid label status")
        if any(r["label_status"] != "labeled" for r in rows) or not plan["pairs"]:
            excluded.append({"date": date, "reason": "incomplete_labels" if plan["pairs"] else "no_unequal_pairs",
                             "plan": plan})
            continue
        for r in rows:
            _require(all(_finite(r[k]) or (k in {"up10", "dn5"} and type(r[k]) is bool)
                         for k in OUTCOMES), "invalid outcome")
            _require(r["up10"] in (0, 1) and r["dn5"] in (0, 1)
                     and r["mfe"] >= 0 and -1 <= r["mae"] <= 0
                     and bool(r["up10"]) == (r["mfe"] >= .1)
                     and bool(r["dn5"]) == (r["mae"] <= -.05), "event/excursion mismatch")
        frame = pd.DataFrame(rows).set_index("coin")
        sides = {s: [p[s] for p in plan["pairs"]] for s in ("high", "low")}
        values = {s: _metric_rows(frame, picks).mean(axis=0) for s, picks in sides.items()}
        values["delta"] = values["high"]-values["low"]
        daily.append({"date": date, **plan, "n_pairs": len(plan["pairs"]),
                      "metrics": {s: _named(v) for s, v in values.items()},
                      "context": {s: {k: float(frame.loc[picks, k].mean()) for k in PREDICTORS}
                                  for s, picks in sides.items()},
                      "book_age_max_seconds": float(frame.book_event_age_seconds.max())})
    segments = {}
    for name, subset in (("earlier", [d for d in daily if d["date"] < CONFIG["split_date"]]),
                         ("later", [d for d in daily if d["date"] >= CONFIG["split_date"]]), ("all", daily)):
        segment = {"dates": len(subset), "pairs": sum(d["n_pairs"] for d in subset)}
        if subset:
            arrays = {s: np.array([[d["metrics"][s][k] for k in METRICS] for d in subset])
                      for s in ("high", "low", "delta")}
            segment["metrics"] = {s: _summary(a, n_boot, seed) for s, a in arrays.items()}
            segment["leave_one_date_out"] = [{"omitted_date": d["date"],
                "mean": _named(np.delete(arrays["delta"], i, axis=0).mean(axis=0))}
                for i, d in enumerate(subset)] if len(subset) > 1 else []
        segments[name] = segment
    return {"configuration": CONFIG.copy(), "daily": daily, "excluded": excluded, "segments": segments,
            "expected_dates": len(expected), "automatic_adoption": False, "model_fitted": False,
            "effect_status": "observational_not_a_recommendation_policy",
            "interpretation": ["Resting L1 orders, not trades, net inflows or full depth; cancellable/spoofable.",
                               "Existing 24h labels include 0.15% costs once; no cost re-estimation.",
                               "Coarse matching leaves residual confounding; not causal or fresh forward."]}

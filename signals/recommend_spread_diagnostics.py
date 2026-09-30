"""Outcome-blind, within-liquidity/ATR spread contrasts; not a selector.

No learned cutoffs, training, portfolio simulation, or production imports.
Pairs are diagnostic comparisons, not hypothetical recommended Top3 picks.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd

from signals.recommend_experiment_eval import METRICS, _metric_rows, _named, _summary

CONFIG = {
    "schema": "recommend_spread_diagnostics.v1",
    "start_date": "2026-09-10",
    "end_date": "2026-09-29",
    "split_date": "2026-09-20",
    "slot": "open",
    "universe_size": 100,
    "feature": "spread_fraction",
    "direction": "narrow_minus_wide",
    "matching": "within_day_ATR_and_log_qv_average_rank_quartiles",
    "pairing": "ascending_spread_then_original_rank_outside_in_disjoint_pairs",
    "equal_spread_pairs": "omit",
    "odd_middle": "omit",
    "missing_predictor_or_outcome": "exclude_whole_date_not_zero",
    "weighting": "equal_pairs_within_date_equal_dates_between_dates",
    "new_feature_hypotheses": 1,
    "parameter_sweep": False,
    "is_untouched_holdout": False,
    "deployable": False,
}
PREDICTORS = (
    "rank",
    "spread_fraction",
    "book_event_age_seconds",
    "f_log_qv",
    "f_atr_pct_14",
)
OUTCOMES = ("up10", "dn5", "mfe", "mae", "eod_return_net", "tp5_sl3_return_net")


def _finite(value):
    return (
        not isinstance(value, (bool, np.bool_))
        and isinstance(value, (int, float, np.number))
        and math.isfinite(value)
    )


def _require(value, reason):
    if not value:
        raise ValueError(reason)


def pairing_plan(rows):
    """Reads identity/predictors only; labels and other dates cannot change pairs."""
    frame = pd.DataFrame(rows).copy()
    _require(set(("coin", *PREDICTORS)) <= set(frame), "missing predictor columns")
    _require(len(frame) == CONFIG["universe_size"], "incomplete original universe")
    _require(
        frame.coin.map(lambda x: isinstance(x, str) and bool(x)).all()
        and frame.coin.is_unique,
        "invalid coin identity",
    )
    for name in PREDICTORS:
        _require(
            frame[name].map(_finite).all(), f"missing or invalid predictor: {name}"
        )
    _require(
        sorted(frame["rank"]) == list(range(1, len(frame) + 1)),
        "invalid original ranks",
    )
    _require(
        frame.spread_fraction.between(0, 2, inclusive="left").all(), "invalid spread"
    )
    _require(frame.book_event_age_seconds.gt(0).all(), "invalid book age")
    _require(frame.f_atr_pct_14.ge(0).all(), "negative ATR")
    frame = frame.sort_values("rank").set_index("coin", drop=False)
    cells = pd.DataFrame(
        {
            field: np.clip(
                np.ceil(frame[field].rank(method="average", pct=True) * 4).astype(int)
                - 1,
                0,
                3,
            )
            for field in ("f_log_qv", "f_atr_pct_14")
        }
    )
    pairs, equal, middle = [], 0, 0
    for cell, members in cells.groupby(list(cells.columns), sort=True):
        ordered = sorted(
            members.index,
            key=lambda c: (frame.at[c, "spread_fraction"], frame.at[c, "rank"]),
        )
        middle += len(ordered) % 2
        for left, right in zip(
            ordered[: len(ordered) // 2],
            reversed(ordered[len(ordered) - len(ordered) // 2 :]),
            strict=True,
        ):
            if frame.at[left, "spread_fraction"] == frame.at[right, "spread_fraction"]:
                equal += 1
            else:
                pairs.append(
                    {"narrow": left, "wide": right, "cell": [int(v) for v in cell]}
                )
    return {
        "pairs": pairs,
        "equal_spread_pairs": equal,
        "unpaired_middle_rows": middle,
        "observed_cells": len(cells.drop_duplicates()),
    }


def analyze(days, *, n_boot=1000, seed=42):
    _require(type(n_boot) is int and 1 <= n_boot <= 100_000, "invalid n_boot")
    _require(type(seed) is int and 0 <= seed < 2**64, "invalid seed")
    expected = (
        pd.date_range(CONFIG["start_date"], CONFIG["end_date"])
        .strftime("%Y-%m-%d")
        .tolist()
    )
    _require(len({d["date"] for d in days}) == len(days), "duplicate date")
    _require(
        all(date.fromisoformat(d["date"]).isoformat() in expected for d in days),
        "date outside fixed window",
    )
    indexed = {d["date"]: d for d in days}
    daily, excluded = [], []
    for day in expected:
        document = indexed.get(day)
        if document is None or document.get("unavailable_reason"):
            excluded.append(
                {
                    "date": day,
                    "reason": (document or {}).get(
                        "unavailable_reason", "missing_evidence"
                    ),
                }
            )
            continue
        rows = document["rows"]
        # Invalid numbers are corruption, not a silent quality-based filter.
        if any(
            r.get("spread_fraction") is None
            or r.get("book_event_age_seconds") is None
            or r.get("f_log_qv") is None
            or r.get("f_atr_pct_14") is None
            for r in rows
        ):
            excluded.append({"date": day, "reason": "missing_predictor"})
            continue
        plan = pairing_plan(rows)
        frame = pd.DataFrame(rows).set_index("coin", drop=False)
        _require(
            frame.label_status.isin(["labeled", "halted_no_observations"]).all(),
            "unknown label status",
        )
        if not frame.label_status.eq("labeled").all():
            excluded.append(
                {"date": day, "reason": "incomplete_full_universe_labels", "plan": plan}
            )
            continue
        for row in rows:
            for name in OUTCOMES:
                _require(
                    (
                        _finite(row[name])
                        or (name in ("up10", "dn5") and type(row[name]) is bool)
                    ),
                    f"invalid outcome: {name}",
                )
            _require(row["up10"] in (0, 1) and row["dn5"] in (0, 1), "invalid event")
            _require(row["mfe"] >= 0 and -1 <= row["mae"] <= 0, "invalid excursion")
            _require(
                bool(row["up10"]) == (row["mfe"] >= 0.1)
                and bool(row["dn5"]) == (row["mae"] <= -0.05),
                "event/excursion mismatch",
            )
        if not plan["pairs"]:
            excluded.append(
                {"date": day, "reason": "no_unequal_spread_pairs", "plan": plan}
            )
            continue
        sides = {side: [p[side] for p in plan["pairs"]] for side in ("narrow", "wide")}
        metrics = {
            side: _metric_rows(frame, coins).mean(axis=0)
            for side, coins in sides.items()
        }
        context = {
            side: {name: float(frame.loc[coins, name].mean()) for name in PREDICTORS}
            for side, coins in sides.items()
        }
        daily.append(
            {
                "date": day,
                **plan,
                "n_pairs": len(plan["pairs"]),
                "metrics": {
                    **{side: _named(v) for side, v in metrics.items()},
                    "delta": _named(metrics["narrow"] - metrics["wide"]),
                },
                "context": context,
                "book_age_max_seconds": float(frame.book_event_age_seconds.max()),
            }
        )
    summaries = {}
    for name, subset in (
        ("earlier", [d for d in daily if d["date"] < CONFIG["split_date"]]),
        ("later", [d for d in daily if d["date"] >= CONFIG["split_date"]]),
        ("all", daily),
    ):
        summary = {"dates": len(subset), "pairs": sum(d["n_pairs"] for d in subset)}
        if subset:
            summary["metrics"] = {
                side: _summary(
                    np.array(
                        [[d["metrics"][side][m] for m in METRICS] for d in subset]
                    ),
                    n_boot,
                    seed,
                )
                for side in ("narrow", "wide", "delta")
            }
            differences = np.array(
                [[d["metrics"]["delta"][m] for m in METRICS] for d in subset]
            )
            summary["leave_one_date_out"] = (
                [
                    {
                        "omitted_date": d["date"],
                        "mean": _named(np.delete(differences, i, axis=0).mean(axis=0)),
                    }
                    for i, d in enumerate(subset)
                ]
                if len(subset) > 1
                else []
            )
        summaries[name] = summary
    return {
        "configuration": CONFIG.copy(),
        "daily": daily,
        "excluded": excluded,
        "segments": summaries,
        "expected_dates": len(expected),
        "effect_status": "observational_not_a_recommendation_policy",
        "automatic_adoption": False,
        "model_fitted": False,
        "interpretation": [
            "Existing net outcomes include 0.15% costs once; no spread-cost subtraction.",
            "Decision-time quotes do not prove entry-time spread, depth or realized slippage.",
            "Coarse matching does not remove residual volatility/liquidity confounding.",
            "Already-observed data and one new hypothesis; not prospective performance.",
        ],
    }

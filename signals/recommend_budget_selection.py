"""Offline whole-universe Top3 selection within the original R1 score budgets.

Score budgets are not guarantees of realized risk. No labels enter selection.
"""

from itertools import combinations
from math import fsum, isfinite
from numbers import Real

import numpy as np

from signals.recommend_experiment_eval import (
    METRICS,
    _metric_rows,
    _named,
    _selection_plans,
    _sensitivity,
    _summary,
    _validate_outcomes,
    _validate_predictors,
)

CONFIG = {
    "schema": "recommend_budget_selection.v1",
    "top_k": 3,
    "universe": "all_original_candidates_including_unlabeled",
    "objective": "maximize_sum_original_upper_C",
    "constraints": ["sum_A_upper>=original_Top3", "sum_p_dn5<=original_Top3"],
    "tie_break": "original_Top3_if_objective_tied_else_original_rank_tuple",
    "arithmetic": "math.fsum_no_epsilon",
    "search": "exhaustive_three_combinations",
    "new_candidates": 1,
    "model_fitted": False,
    "parameter_sweep": False,
    "is_risk_guarantee": False,
    "actual_selection_changed": False,
}


def select_budgeted(rows):
    """Solve the exact fixed three-name problem; ignore all outcome fields."""
    if not isinstance(rows, list) or len(rows) < 3:
        raise ValueError("at least three original candidates required")
    if any(
        not isinstance(r["rank"], Real)
        or isinstance(r["rank"], (bool, np.bool_))
        or not isfinite(r["rank"])
        or not isinstance(r["coin"], str)
        or not r["coin"]
        for r in rows
    ):
        raise ValueError("invalid candidate identity")
    rows = sorted(rows, key=lambda row: row["rank"])
    if [r["rank"] for r in rows] != list(range(1, len(rows) + 1)) or len(
        {r["coin"] for r in rows}
    ) != len(rows):
        raise ValueError("incomplete or duplicate candidate population")
    fields = ("p_up10_A", "p_dn5", "p_up10_C")
    for row in rows:
        for field in fields:
            v = row[field]
            if (
                not isinstance(v, Real)
                or isinstance(v, (bool, np.bool_))
                or not isfinite(v)
                or not 0 <= v <= 1
            ):
                raise ValueError("invalid candidate score")
    upper, down, objective = ([float(row[field]) for row in rows] for field in fields)
    budget_up, budget_down = fsum(upper[:3]), fsum(down[:3])
    baseline_objective = fsum(objective[:3])
    best, best_objective = (0, 1, 2), baseline_objective
    feasible = examined = 0
    for i, j, k in combinations(range(len(rows)), 3):
        examined += 1
        if fsum((down[i], down[j], down[k])) > budget_down:
            continue
        if fsum((upper[i], upper[j], upper[k])) < budget_up:
            continue
        feasible += 1
        value = fsum((objective[i], objective[j], objective[k]))
        # Enumeration is lexicographic; equal objectives never create churn.
        if value > best_objective:
            best, best_objective = (i, j, k), value
    assert feasible > 0  # Original three are feasible, including exact boundaries.
    return {
        "selected_coins": [rows[i]["coin"] for i in best],
        "selected_original_ranks": [i + 1 for i in best],
        "original_upper_sum": budget_up,
        "original_down_sum": budget_down,
        "selected_upper_sum": fsum(upper[i] for i in best),
        "selected_down_sum": fsum(down[i] for i in best),
        "original_C_sum": baseline_objective,
        "selected_C_sum": best_objective,
        "feasible_combinations": feasible,
        "examined_combinations": examined,
        "changed_picks": len(set(best) - {0, 1, 2}),
    }


def evaluate_budget(frame, *, n_boot=1000, seed=42):
    """Evaluate actual chosen indices; never fabricate probabilities to rank E."""
    if (
        type(n_boot) is not int
        or not 1 <= n_boot <= 100_000
        or type(seed) is not int
        or seed < 0
    ):
        raise ValueError("invalid bootstrap settings")
    frame = _validate_predictors(frame)
    plans = _selection_plans(frame)  # Original A parity, predictor-only.
    for plan in plans:
        group = frame.loc[plan["indices"]]
        selection = None
        if group["C_status"].iloc[0] == "ok":
            selection = select_budgeted(
                group[["coin", "rank", "p_up10_A", "p_dn5", "p_up10_C"]].to_dict(
                    orient="records"
                )
            )
        index = dict(zip(group.coin, group.index, strict=True))
        selected = {
            "A": plan["selected"]["A"],
            "E": None
            if selection is None
            else [index[c] for c in selection["selected_coins"]],
        }
        quartiles = {
            f: np.clip(
                np.ceil(group[f].rank(method="average", pct=True) * 4).astype(int) - 1,
                0,
                3,
            )
            for f in ("f_log_qv", "f_atr_pct_14")
        }
        cells = {
            i: tuple(int(values.at[i]) for values in quartiles.values())
            for i in group.index
        }
        plan.update(
            selected=selected,
            budget=selection,
            matched={
                arm: None
                if picks is None
                else [[j for j in group.index if cells[j] == cells[i]] for i in picks]
                for arm, picks in selected.items()
            },
        )
    # Only now inspect whether each already-selected candidate has an outcome.
    _validate_outcomes(frame)
    valid, excluded, daily, audits = [], [], [], []
    for plan in plans:
        reasons = []
        for arm, picks in plan["selected"].items():
            if picks is None:
                reasons.append(f"arm_unavailable:{arm}")
                continue
            for i, pool in zip(picks, plan["matched"][arm], strict=True):
                if len(pool) < 3:
                    reasons.append(
                        f"matched_cell_unsupported:{arm}:{frame.at[i, 'coin']}:{len(pool)}"
                    )
        if any(frame.at[i, "label_status"] != "labeled" for i in plan["indices"]):
            reasons.append("full_universe_label_unavailable")
        audits.append(
            {
                "date": plan["date"],
                "slot": plan["slot"],
                "fold": plan["fold"],
                "selected_coins": {
                    arm: None if picks is None else frame.loc[picks, "coin"].tolist()
                    for arm, picks in plan["selected"].items()
                },
                "budget": plan["budget"],
                "evaluable": not reasons,
                "matched_pool_sizes": {
                    arm: None if pools is None else [len(pool) for pool in pools]
                    for arm, pools in plan["matched"].items()
                },
            }
        )
        if reasons:
            excluded.append(
                {"date": plan["date"], "slot": plan["slot"], "reasons": reasons}
            )
            continue
        valid.append(plan)
        daily.append(
            {
                "date": plan["date"],
                "slot": plan["slot"],
                "fold": plan["fold"],
                "arms": {
                    arm: _named(_metric_rows(frame, picks).mean(0))
                    for arm, picks in plan["selected"].items()
                },
                "full_universe": _named(_metric_rows(frame, plan["indices"]).mean(0)),
                "matched": {
                    arm: _named(
                        np.mean(
                            [_metric_rows(frame, pool).mean(0) for pool in pools],
                            axis=0,
                        )
                    )
                    for arm, pools in plan["matched"].items()
                },
            }
        )
    result = {
        "schema": "recommend_budget_evaluation.v1",
        "configuration": CONFIG,
        "status": "historical_comparison" if valid else "no_common_evaluable_dates",
        "deployable": False,
        "is_untouched_holdout": False,
        "coverage": {
            "included_date_slots": len(valid),
            "excluded_date_slots": len(excluded),
        },
        "selection_audit": audits,
        "excluded": excluded,
        "daily": daily,
        "slots": {},
        "methodology": {
            "population": "all candidates fixed before outcomes; full-universe labels and matched min-pool3 required",
            "weighting": "three picks equal weight then dates equal weight; slots separate",
            "matching": "same snapshot log_qv/ATR average-percentile joint quartiles; three pools equal weight",
            "returns": "existing net fractional labels; costs already deducted once; not portfolio returns",
            "uncertainty": "paired IID-date and observed-date block3 percentile95, diagnostic only",
            "score_constraints": "predicted score sums are not guarantees of realized risk or upside",
        },
    }
    for slot in sorted(frame.slot.unique()):
        records = [r for r in daily if r["slot"] == slot]
        if not records:
            result["slots"][slot] = {"dates": 0, "status": "no_common_evaluable_dates"}
            continue
        values = {
            arm: np.array([[r["arms"][arm][m] for m in METRICS] for r in records])
            for arm in ("A", "E")
        }
        delta = values["E"] - values["A"]
        selected_plans = [p for p in valid if p["slot"] == slot]
        result["slots"][slot] = {
            "dates": len(records),
            "picks_per_arm": 3 * len(records),
            "included_dates": [r["date"] for r in records],
            "changed_dates": sum(
                p["budget"]["changed_picks"] > 0 for p in selected_plans
            ),
            "changed_picks": sum(p["budget"]["changed_picks"] for p in selected_plans),
            "absolute": {
                arm: _summary(value, n_boot, seed) for arm, value in values.items()
            },
            "delta_vs_A": _summary(delta, n_boot, seed),
            "within_vol_liquidity_lift": {
                arm: _summary(
                    values[arm]
                    - np.array(
                        [[r["matched"][arm][m] for m in METRICS] for r in records]
                    ),
                    n_boot,
                    seed,
                )
                for arm in ("A", "E")
            },
            "sensitivity_vs_A": _sensitivity(frame, selected_plans, "E", delta),
            "folds": [
                {
                    "fold": fold,
                    "dates": sum(r["fold"] == fold for r in records),
                    "delta_vs_A": _named(
                        delta[[r["fold"] == fold for r in records]].mean(0)
                    ),
                }
                for fold in sorted({r["fold"] for r in records})
            ],
        }
    return result

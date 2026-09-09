"""R1-only frozen-history diagnosis, with no fitting, alternative policy or I/O.

All ranking, matching and score geometry are fixed before outcome inspection.
Selection effects are historical paired differences, not causal effects or proof
that a different ranking rule would improve live recommendations.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date
import hashlib
import json

import numpy as np
import pandas as pd

from signals.recommend_experiment_eval import (
    METRICS, OUTCOMES, ExperimentEvaluationError,
    _finite, _metric_rows, _named, _summary, _validate_outcomes,
)

SELECTION_CONFIG = {
    "schema": "recommend_selection_diagnostics.v1", "cohort": "full_frozen_history",
    "slots": ["open", "preopen"],
    "ranking": {"ratio_floor": 0.001, "top_k": 3,
                "keys": ["ratio descending", "p_dn10 ascending", "p_up10 descending",
                         "exp_downside descending", "original_rank ascending"]},
    "matching": {"features": ["f_log_qv", "f_atr_pct_14"], "quartiles": 4,
                 "percentile_rank": "average", "min_pool": 3,
                 "includes_selected": True, "each_selected_pool_weight": 1 / 3},
    "rank_bands": [{"name": "1-3", "lower": 1, "upper": 3},
                   {"name": "4-10", "lower": 4, "upper": 10},
                   {"name": "11-30", "lower": 11, "upper": 30},
                   {"name": "31+", "lower": 31, "upper": None}],
    "metrics": list(METRICS),
    "log_decomposition": {"center": "snapshot_population_mean_log",
                          "zero_up": "whole_snapshot_log_diagnostic_unavailable",
                          "additivity_absolute_tolerance": 1e-12},
    "ci": {"min_dates": 5, "moving_block_observed_dates": 3, "level": 0.95},
    "model_fitted": False, "new_policy_evaluated": False,
    "threshold_sweep": False, "live_changes": False,
}
PREDICTORS = ("date", "slot", "coin", "snapshot_id", "rank", "was_delivered",
              "p_up10", "p_dn5", "p_dn10", "exp_downside", "rr_ratio",
              "f_log_qv", "f_atr_pct_14")
_NUMERIC_PREDICTORS = ("rank", "p_up10", "p_dn5", "p_dn10", "exp_downside",
                       "rr_ratio", "f_log_qv", "f_atr_pct_14")


class SelectionDiagnosticsError(ValueError):
    """Invalid evidence blocks the complete diagnosis before publication."""


def _require(condition, reason):
    if not condition:
        raise SelectionDiagnosticsError(reason)


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _validate_predictors(frame):
    _require(isinstance(frame, pd.DataFrame) and not frame.empty, "empty or invalid frame")
    _require(frame.columns.is_unique, "duplicate columns")
    required = set(PREDICTORS + OUTCOMES) | {"label_status"}
    _require(required <= set(frame.columns), f"missing columns: {sorted(required - set(frame.columns))}")
    # Only names, not values, of outcome columns are inspected in this phase.
    scores = frame[list(PREDICTORS)].copy(deep=True)
    for field in ("date", "slot", "coin", "snapshot_id"):
        _require(scores[field].map(lambda value: isinstance(value, str) and bool(value)).all(), f"invalid {field}")
    for day in scores["date"].unique():
        try:
            _require(date.fromisoformat(day).isoformat() == day, "noncanonical date")
        except ValueError as exc:
            raise SelectionDiagnosticsError("invalid date") from exc
    _require(scores["slot"].isin(SELECTION_CONFIG["slots"]).all(), "invalid slot")
    _require(scores["was_delivered"].map(lambda value: isinstance(value, (bool, np.bool_))).all(), "invalid delivery flag")
    for field in _NUMERIC_PREDICTORS:
        _require(scores[field].map(_finite).all(), f"invalid predictor: {field}")
    for field in ("p_up10", "p_dn5", "p_dn10"):
        _require(scores[field].between(0, 1).all(), f"invalid probability: {field}")
    _require(scores["rr_ratio"].ge(0).all(), "negative frozen ratio")
    _require((scores["rank"].gt(0) & scores["rank"].eq(np.floor(scores["rank"]))).all(), "invalid rank")
    _require(not scores.duplicated(["date", "slot", "coin"]).any(), "duplicate candidate")
    for _, group in scores.groupby(["date", "slot"], sort=True):
        _require(group["snapshot_id"].nunique() == 1, "mixed snapshot date/slot")
        _require(sorted(group["rank"]) == list(range(1, len(group) + 1)), "incomplete candidate ranks")
        _require(sorted(group.loc[group["was_delivered"], "rank"]) == [1, 2, 3], "actual Top3 missing or inconsistent")
    _require(scores.groupby("snapshot_id")[["date", "slot"]].nunique().eq(1).all().all(), "snapshot reused across date/slot")
    return scores.sort_values(["date", "slot", "rank"], kind="stable").reset_index(drop=True)


def _tie_summary(values, selected):
    counts = values.value_counts()
    return {"distinct_values": len(counts), "tied_rows": int(counts.loc[counts.gt(1)].sum()),
            "largest_tie": int(counts.max()),
            "top3_tied_rows": sum(int(counts.at[values.at[i]]) > 1 for i in selected)}


def _geometry(group, selected, ranking, effective_down, ratio):
    up_pct = group["p_up10"].rank(method="average", pct=True)
    low_down_pct = (-effective_down).rank(method="average", pct=True)
    positive = group["p_up10"].gt(0).all()
    log_audit = {"status": "available" if positive else "unavailable_zero_up_probability",
                 "zero_up_rows": int(group["p_up10"].eq(0).sum()), "population_rows": len(group)}
    up_center = down_center = rr_center = None
    if positive:
        up = np.log(group["p_up10"].astype(float))
        low_down = -np.log(effective_down.astype(float))
        log_rr = up + low_down
        up_center, down_center = up - up.mean(), low_down - low_down.mean()
        rr_center = log_rr - log_rr.mean()
        residual = float((up_center + down_center - rr_center).abs().max())
        _require(residual <= SELECTION_CONFIG["log_decomposition"]["additivity_absolute_tolerance"], "log decomposition additivity failed")
        log_audit.update(max_additivity_residual=residual, mean_log_up=float(up.mean()),
                         mean_negative_log_effective_down=float(low_down.mean()),
                         mean_log_ratio=float(log_rr.mean()))
    reconstructed = {index: rank for rank, index in enumerate(ranking, 1)}
    candidates = [{"coin": group.at[i, "coin"], "original_rank": int(group.at[i, "rank"]),
                   "reconstructed_rank": reconstructed[i], "selected": i in selected,
                   "up_percentile": float(up_pct.at[i]), "low_down_percentile": float(low_down_pct.at[i]),
                   "p_up10": float(group.at[i, "p_up10"]), "p_dn5": float(group.at[i, "p_dn5"]),
                   "effective_p_dn5": float(effective_down.at[i]), "computed_ratio": float(ratio.at[i]),
                   "up_log_contribution": float(up_center.at[i]) if positive else None,
                   "low_down_log_contribution": float(down_center.at[i]) if positive else None,
                   "centered_log_ratio": float(rr_center.at[i]) if positive else None}
                  for i in group.index]
    boundary = {"status": "no_unselected_candidate"}
    if len(ranking) > 3:
        chosen, next_index = ranking[2:4]
        keys = ("computed_ratio", "p_dn10", "p_up10", "exp_downside", "original_rank")
        left = (ratio.at[chosen], group.at[chosen, "p_dn10"], group.at[chosen, "p_up10"],
                group.at[chosen, "exp_downside"], group.at[chosen, "rank"])
        right = (ratio.at[next_index], group.at[next_index, "p_dn10"], group.at[next_index, "p_up10"],
                 group.at[next_index, "exp_downside"], group.at[next_index, "rank"])
        boundary = {"status": "available", "rank3_coin": group.at[chosen, "coin"],
                    "original_rank4_coin": group.loc[group["rank"].eq(4), "coin"].iloc[0],
                    "nearest_unselected_coin": group.at[next_index, "coin"],
                    "nearest_unselected_original_rank": int(group.at[next_index, "rank"]),
                    "ratio_margin": float(ratio.at[chosen] - ratio.at[next_index]),
                    "decisive_key": next(key for key, a, b in zip(keys, left, right, strict=True) if a != b),
                    "ratio_tie_rows_at_cutoff": int(ratio.eq(ratio.at[chosen]).sum())}
    ties = {"p_up10": _tie_summary(group["p_up10"], selected),
            "effective_p_dn5": _tie_summary(effective_down, selected),
            "computed_ratio": _tie_summary(ratio, selected),
            "frozen_ratio": _tie_summary(group["rr_ratio"], selected)}
    return {"candidates": candidates, "ties": ties, "boundary": boundary, "log_decomposition": log_audit,
            "downside_floor_rows": int(group["p_dn5"].lt(SELECTION_CONFIG["ranking"]["ratio_floor"]).sum()),
            "downside_at_floor_rows": int(group["p_dn5"].eq(SELECTION_CONFIG["ranking"]["ratio_floor"]).sum()),
            "full_order_matches": ranking == group.index.tolist()}


def _build_plans(scores):
    """Input deliberately contains no outcomes or future availability flags."""
    _require(set(scores.columns) == set(PREDICTORS), "selection accepts only predictor columns")
    plans = []
    for (day, slot), group in scores.groupby(["date", "slot"], sort=True):
        selected = group.loc[group["was_delivered"]].index.tolist()
        effective_down = group["p_dn5"].clip(lower=SELECTION_CONFIG["ranking"]["ratio_floor"])
        ratio = group["p_up10"] / effective_down
        ranking = sorted(group.index, key=lambda i: (-float(ratio.at[i]), float(group.at[i, "p_dn10"]),
                                                    -float(group.at[i, "p_up10"]), -float(group.at[i, "exp_downside"]),
                                                    int(group.at[i, "rank"])))
        _require(ranking[:3] == selected, f"identity Top3 parity failed: {day}/{slot}")
        quartiles = {feature: np.clip(np.ceil(group[feature].rank(method="average", pct=True) * 4).astype(int) - 1, 0, 3)
                     for feature in SELECTION_CONFIG["matching"]["features"]}
        cells = {i: tuple(int(quartiles[feature].at[i]) for feature in SELECTION_CONFIG["matching"]["features"])
                 for i in group.index}
        matched = [[i for i in group.index if cells[i] == cells[chosen]] for chosen in selected]
        bands = {band["name"]: [i for i in group.index if group.at[i, "rank"] >= band["lower"]
                                and (band["upper"] is None or group.at[i, "rank"] <= band["upper"])]
                 for band in SELECTION_CONFIG["rank_bands"]}
        plans.append({"date": day, "slot": slot, "snapshot_id": group["snapshot_id"].iloc[0],
                      "indices": group.index.tolist(), "selected": selected, "matched": matched,
                      "cells": cells, "bands": bands,
                      "geometry": _geometry(group, selected, ranking, effective_down, ratio)})
    return plans


def _cohort(frame, plans, selection):
    membership, distinct, coins = [], set(), set()
    total_memberships = 0
    for plan in plans:
        if selection == "matched_expectation":
            pools = plan["matched"]
        else:
            indices = (plan["selected"] if selection == "actual_top3" else plan["indices"]
                       if selection == "full_universe" else plan["bands"][selection])
            pools = [indices]
        if not any(pools):
            continue
        payload = {"date": plan["date"], "slot": plan["slot"], "snapshot_id": plan["snapshot_id"], "pools": []}
        for pool in pools:
            total_memberships += len(pool)
            distinct.update(pool)
            coins.update(frame.loc[pool, "coin"])
            payload["pools"].append({"outer_weight": 1 / len(pools), "member_weight": 1 / len(pool),
                                     "members": [{"coin": frame.at[i, "coin"], "rank": int(frame.at[i, "rank"])} for i in pool]})
        membership.append(payload)
    digest = _digest({"selection": selection, "membership": membership})
    date_membership = [{key: item[key] for key in ("date", "slot", "snapshot_id")} for item in membership]
    return {"cohort_id": f"{selection}:{digest[:16]}", "cohort_sha256": digest,
            "date_cohort_sha256": _digest(date_membership), "date_slots": date_membership,
            "denominators": {"date_slots": len(membership), "dates": len({item["date"] for item in membership}),
                             "candidate_memberships": total_memberships, "distinct_candidate_rows": len(distinct),
                             "distinct_coins": len(coins)},
            "hash_scope": "ordered decision-time identities and within-date weights, not outcome values"}


def _lodo(plans, differences):
    if len(plans) < 2:
        return {"status": "insufficient_dates", "rows": []}
    return {"status": "available", "rows": [
        {"omitted_date": plan["date"], "remaining_dates": len(plans) - 1,
         "mean_delta": _named(np.delete(differences, index, axis=0).mean(axis=0))}
        for index, plan in enumerate(plans)]}


def analyze_selection(frame: pd.DataFrame, *, n_boot: int = 1000, seed: int = 42) -> dict:
    """Diagnose actual frozen R1 picks only. Never mutate inputs or fit models."""
    _require(type(n_boot) is int and 1 <= n_boot <= 100_000, "n_boot must be an integer in [1,100000]")
    _require(type(seed) is int and seed >= 0, "seed must be a nonnegative integer")
    scores = _validate_predictors(frame)
    plans = _build_plans(scores)
    # Outcome access starts here, after all plans AND structural audits exist.
    data = frame.sort_values(["date", "slot", "rank"], kind="stable").reset_index(drop=True).copy(deep=True)
    try:
        _validate_outcomes(data)
    except ExperimentEvaluationError as exc:
        raise SelectionDiagnosticsError(str(exc)) from exc
    valid, excluded, daily, selection_audit = [], [], [], []
    for plan in plans:
        reasons = []
        unavailable = data.loc[plan["indices"]].loc[lambda rows: rows["label_status"].ne("labeled"), "coin"].tolist()
        if unavailable:
            reasons.append({"kind": "full_universe_label_unavailable", "coins": unavailable})
        for index, pool in zip(plan["selected"], plan["matched"], strict=True):
            if len(pool) < SELECTION_CONFIG["matching"]["min_pool"]:
                reasons.append({"kind": "matched_cell_unsupported", "coin": data.at[index, "coin"],
                                "pool_size": len(pool), "pool_coins": data.loc[pool, "coin"].tolist()})
        identity = {key: plan[key] for key in ("date", "slot", "snapshot_id")}
        selection_audit.append({**identity, "selected_coins": data.loc[plan["selected"], "coin"].tolist(),
                                "matched_pools": [{"anchor_coin": data.at[i, "coin"], "cell": list(plan["cells"][i]),
                                                   "coins": data.loc[pool, "coin"].tolist(), "size": len(pool)}
                                                  for i, pool in zip(plan["selected"], plan["matched"], strict=True)],
                                "geometry": plan["geometry"], "performance_evaluable": not reasons})
        if reasons:
            excluded.append({**identity, "reasons": reasons})
            continue
        valid.append(plan)
        daily.append({**identity, "actual_top3": _named(_metric_rows(data, plan["selected"]).mean(axis=0)),
                      "full_universe": _named(_metric_rows(data, plan["indices"]).mean(axis=0)),
                      "matched_expectation": _named(np.mean([_metric_rows(data, pool).mean(axis=0) for pool in plan["matched"]], axis=0)),
                      "rank_bands": {name: {"rows": len(indices), "mean": _named(_metric_rows(data, indices).mean(axis=0)) if indices else None}
                                     for name, indices in plan["bands"].items()}})
    report = {"schema": SELECTION_CONFIG["schema"], "status": "historical_diagnostic" if valid else "no_common_evaluable_dates",
              "execution_status": "completed", "model_fitted": False, "deployable": False,
              "promotion_status": "NOT_EVALUATED", "automatic_adoption": False,
              "config": deepcopy(SELECTION_CONFIG), "parameters": {"n_boot": n_boot, "seed": seed},
              "input": {"candidate_rows": len(data), "date_slots": len(plans), "dates": int(data["date"].nunique()),
                        "delivered_rows": int(data["was_delivered"].sum())},
              "coverage": {"included_date_slots": len(valid), "excluded_date_slots": len(excluded)},
              "parity": {"status": "passed", "top3_order_passed": len(plans),
                         "full_order_mismatches": sum(not plan["geometry"]["full_order_matches"] for plan in plans)},
              "methodology": {
                  "cohort": "full frozen history, R1-only; not the prior three-arm outer-validation common cohort",
                  "performance_cohort": "per-slot intersection of actual Top3, full-universe and supported matched expectations; identical dates for all three legs",
                  "structure_cohort": "all frozen candidates including future-halted rows; no outcome or label-status access before selection, matching and geometry are fixed",
                  "matching": "same-snapshot average-percentile joint quartiles; includes selected coins; three pools equally weighted; no adaptive widening",
                  "weighting": "candidate mean within date/slot, then equal dates; slots never pooled; matched memberships may overlap and are not independent observations",
                  "ranking": "no-op Top3 order parity required; lower-rank rounding discrepancies are diagnostic, not a full-precision reconstruction",
                  "boundary": "actual rank3 versus nearest reconstructed unselected; original rank4 also identified",
                  "log": "log(p_up10)-log(max(p_dn5,0.001)), centered on complete snapshot population; additive score geometry, not causal or independent contributions; any zero p_up10 disables that snapshot's log decomposition",
                  "bands": "fixed original-rank groups, descriptive only; not an optimal K search or alternative selection rule",
                  "returns": "existing cost-adjusted fractional pick proxies; no extra 0.15% deduction, portfolio compounding, Sharpe, drawdown or actual fills",
                  "whole_path_safe_up10": "MFE>=0.10 and MAE>-0.05 over canonical 24h; not TP-first passage",
                  "metric_units": "rates and returns are fractions; MAE is a nonpositive excursion fraction",
                  "ci": "date-paired mean differences; IID dates and non-circular blocks of 3 observed dates, not necessarily consecutive calendar days; diagnostic only",
                  "auc": "no selected-versus-universe AUC comparison is used to infer selection loss",
                  "verdict": "observed history, not untouched validation; no causal claim, tuning, model training or automatic policy decision",
              }, "selection_audit": selection_audit, "excluded": excluded, "daily": daily, "slots": {}}
    for slot in sorted(data["slot"].unique()):
        all_plans = [plan for plan in plans if plan["slot"] == slot]
        slot_plans = [plan for plan in valid if plan["slot"] == slot]
        records = [row for row in daily if row["slot"] == slot]
        log_plans = [plan for plan in all_plans if plan["geometry"]["log_decomposition"]["status"] == "available"]
        chosen_geometry = [[row for row in plan["geometry"]["candidates"] if row["selected"]] for plan in all_plans]
        geometry = {"mean_top3_up_percentile": float(np.mean([np.mean([row["up_percentile"] for row in rows]) for rows in chosen_geometry])),
                    "mean_top3_low_down_percentile": float(np.mean([np.mean([row["low_down_percentile"] for row in rows]) for rows in chosen_geometry])),
                    "log_cohort": _cohort(data, log_plans, "full_universe"),
                    "log_unavailable_snapshots": [{key: plan[key] for key in ("date", "slot", "snapshot_id")} for plan in all_plans if plan not in log_plans],
                    "mean_top3_up_log_contribution": None, "mean_top3_low_down_log_contribution": None}
        if log_plans:
            for name in ("up_log_contribution", "low_down_log_contribution"):
                geometry[f"mean_top3_{name}"] = float(np.mean([
                    np.mean([row[name] for row in plan["geometry"]["candidates"] if row["selected"]]) for plan in log_plans]))
        slot_report = {"status": "historical_diagnostic" if records else "no_common_evaluable_dates",
                       "structural_cohort": _cohort(data, all_plans, "full_universe"), "geometry": geometry,
                       "excluded": [row for row in excluded if row["slot"] == slot],
                       "cohorts": {name: _cohort(data, slot_plans, name) for name in ("actual_top3", "full_universe", "matched_expectation")},
                       "absolute": {}, "deltas": {}, "rank_bands": {}}
        report["slots"][slot] = slot_report
        if not records:
            continue
        values = {name: np.array([[row[name][metric] for metric in METRICS] for row in records])
                  for name in ("actual_top3", "full_universe", "matched_expectation")}
        slot_report["absolute"] = {name: _summary(value, n_boot, seed) for name, value in values.items()}
        for baseline in ("full_universe", "matched_expectation"):
            difference = values["actual_top3"] - values[baseline]
            slot_report["deltas"][f"top3_minus_{baseline}"] = {
                **_summary(difference, n_boot, seed), "leave_one_date_out": _lodo(slot_plans, difference),
                "selected_cohort_id": slot_report["cohorts"]["actual_top3"]["cohort_id"],
                "baseline_cohort_id": slot_report["cohorts"][baseline]["cohort_id"],
                "date_cohort_sha256": slot_report["cohorts"][baseline]["date_cohort_sha256"]}
        for band in SELECTION_CONFIG["rank_bands"]:
            name = band["name"]
            band_rows = [row["rank_bands"][name] for row in records if row["rank_bands"][name]["rows"]]
            slot_report["rank_bands"][name] = {"cohort": _cohort(data, slot_plans, name), "descriptive_only": True,
                "status": "available" if band_rows else "empty_band", "dates": len(band_rows),
                "candidate_rows": sum(row["rows"] for row in band_rows),
                "day_equal_mean": _named(np.mean([[row["mean"][metric] for metric in METRICS] for row in band_rows], axis=0)) if band_rows else None}
    return report

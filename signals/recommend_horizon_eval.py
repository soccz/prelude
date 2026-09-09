"""All-head, fixed-cohort retrospective evaluation; no fitting or IO.

Selections and matching pools are fixed using predictors before outcomes are
checked. The primary baseline-common cohort and selected-only supplement must
never be pooled or swapped after seeing results.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date
import hashlib
import json
from numbers import Real

import numpy as np
import pandas as pd

from signals.recommend_experiment_eval import (
    METRICS, OUTCOMES, _classification, _metric_rows, _named, _summary,
    _validate_outcomes,
)

ARMS = ("A", "B", "C")
HEADS = ("up5", "up10", "up20", "dn5", "dn10")
SCORES = tuple(f"p_{head}" for head in HEADS) + ("exp_downside",)
CONTRASTS = {"C_minus_B": ("C", "B"), "B_minus_A": ("B", "A"), "C_minus_A": ("C", "A")}
EVALUATION_CONFIG = {
    "version": "recommend_horizon_eval.v1", "slot": "preopen", "top_k": 3,
    "ranking": ["-p_up10/max(p_dn5,0.001)", "p_dn10", "-p_up10", "-exp_downside", "original_rank"],
    "ratio_floor": .001, "matched_min_pool": 3,
    "matching": "same-date log_qv/ATR average-percentile joint quartiles; expected random mean includes picks",
    "primary": "baseline_common", "supplemental": "selected_only",
    "primary_contrast": "C_minus_B", "secondary_contrasts": ["B_minus_A", "C_minus_A"],
    "metrics": list(METRICS), "return_unit": "fraction", "cost": "already net; no additional deduction",
    "weighting": "equal three picks within date, then equal dates",
    "bootstrap": ["iid_date", "noncircular_block3_observed_dates"],
    "n_boot_default": 4000, "seed_default": 42, "minimum_ci_dates_initial": 5,
    "chronological_group_dates": 5, "group_note": "diagnostic groups, not outer CV or OOF; models refit daily",
    "settings_in_this_experiment": 1, "fit_count_is_not_trial_count": True,
    "is_untouched_holdout": False, "automatic_promotion": False,
}


class HorizonEvaluationError(ValueError):
    """Inconsistent evidence invalidates the complete comparison."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise HorizonEvaluationError(reason)


def _finite(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) and bool(np.isfinite(value))


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _column(name: str, arm: str) -> str:
    return name if arm == "A" else f"{name}_{arm}"


def _validate(frame: pd.DataFrame) -> pd.DataFrame:
    _require(isinstance(frame, pd.DataFrame) and not frame.empty and frame.columns.is_unique, "empty frame or duplicate columns")
    required = set(OUTCOMES) | set(HEADS) | {
        "date", "slot", "coin", "snapshot_id", "rank", "was_delivered", "label_status",
        "p_up10", "p_dn5", "p_dn10", "exp_downside", "rr_ratio", "f_log_qv", "f_atr_pct_14",
        "B_status", "C_status",
    } | {f"{name}_{arm}" for arm in ("B", "C") for name in SCORES}
    _require(required <= set(frame), f"missing columns: {sorted(required - set(frame))}")
    work = frame.copy(deep=True).reset_index(drop=True)
    for field in ("date", "slot", "coin", "snapshot_id"):
        _require(work[field].map(lambda x: isinstance(x, str) and bool(x)).all(), f"invalid {field}")
    for day in work.date.unique():
        _require(date.fromisoformat(day).isoformat() == day, "date must be canonical ISO")
    _require(work.slot.eq("preopen").all(), "only frozen preopen is supported")
    _require(not work.duplicated(["date", "coin"]).any(), "duplicate date/coin")
    _require(work.was_delivered.map(lambda x: isinstance(x, (bool, np.bool_))).all(), "was_delivered must be boolean")
    for name in ("rank", "p_up10", "p_dn5", "p_dn10", "exp_downside", "rr_ratio", "f_log_qv", "f_atr_pct_14"):
        _require(work[name].map(_finite).all(), f"invalid predictor: {name}")
    _require(work.rr_ratio.ge(0).all() and work.f_atr_pct_14.ge(0).all(), "negative ratio or ATR")
    for arm in ARMS:
        if arm == "A":
            available = pd.Series(True, index=work.index)
            names = [name for name in SCORES if name in work]
        else:
            status = f"{arm}_status"
            _require(work[status].isin(["fitted", "unavailable"]).all(), f"invalid {arm} status")
            _require(work.groupby("date")[status].nunique().eq(1).all(), f"mixed date arm status: {arm}")
            available = work[status].eq("fitted")
            names = [f"{name}_{arm}" for name in SCORES]
            raw_names = [f"p_{head}_{arm}_raw" for head in HEADS]
            present = [name for name in raw_names if name in work]
            _require(not present or len(present) == len(raw_names), f"partial raw-head columns: {arm}")
            names += present
        for name in names:
            _require(work.loc[available, name].map(_finite).all(), f"invalid predictor: {name}")
            lo, hi = (-1, 0) if name.startswith("exp_downside") else (0, 1)
            _require(work.loc[available, name].between(lo, hi).all(), f"predictor out of range: {name}")
            _require(work.loc[~available, name].isna().all(), f"unavailable arm has nonmissing scores: {name}")
    for _, group in work.groupby("date", sort=True):
        _require(group.snapshot_id.nunique() == 1, "mixed snapshot within date")
        _require(sorted(group['rank']) == list(range(1, len(group) + 1)), "ranks must be complete and unique")
        _require(group.loc[group.was_delivered, "rank"].sort_values().tolist() == [1, 2, 3], "actual Top3 identity invalid")
    _require(work.groupby("snapshot_id").date.nunique().eq(1).all(), "snapshot reused across dates")
    return work.sort_values(["date", "rank"], kind="stable").reset_index(drop=True)


def _rank_all_heads(group: pd.DataFrame, arm: str) -> list[int]:
    return sorted(group.index, key=lambda i: (
        -float(group.at[i, _column("p_up10", arm)]) / max(float(group.at[i, _column("p_dn5", arm)]), .001),
        float(group.at[i, _column("p_dn10", arm)]), -float(group.at[i, _column("p_up10", arm)]),
        -float(group.at[i, _column("exp_downside", arm)]), int(group.at[i, "rank"]),
    ))[:3]


def _plans(frame: pd.DataFrame) -> list[dict]:
    """Only predictor/identity columns may be read in this stage."""
    plans = []
    for number, (day, group) in enumerate(frame.groupby("date", sort=True)):
        actual = group.loc[group.was_delivered].index.tolist()
        _require(_rank_all_heads(group, "A") == actual, f"A identity Top3 parity failed: {day}")
        selected = {"A": actual, **{arm: _rank_all_heads(group, arm) if group[f"{arm}_status"].iloc[0] == "fitted" else None
                                    for arm in ("B", "C")}}
        bands = [np.clip(np.ceil(group[name].rank(method="average", pct=True) * 4).astype(int) - 1, 0, 3)
                 for name in ("f_log_qv", "f_atr_pct_14")]
        cells = {i: tuple(int(band.at[i]) for band in bands) for i in group.index}
        matched = {arm: None if picks is None else [[j for j in group.index if cells[j] == cells[i]] for i in picks]
                   for arm, picks in selected.items()}
        plans.append({"date": day, "snapshot_id": group.snapshot_id.iloc[0], "chronological_group": number // 5 + 1,
                      "indices": group.index.tolist(), "selected": selected, "matched": matched})
    return plans


def _validate_extra_outcomes(frame: pd.DataFrame) -> None:
    _validate_outcomes(frame)
    for _, row in frame.iterrows():
        for head, threshold in (("up5", .05), ("up20", .20), ("dn10", -.10)):
            value = row[head]
            if row.label_status == "halted_no_observations":
                _require(pd.isna(value), f"halted row has {head}")
            else:
                _require((isinstance(value, (bool, np.bool_)) or _finite(value)) and value in (0, 1), f"invalid {head}")
                expected = row.mfe >= threshold if head.startswith("up") else row.mae <= threshold
                _require(bool(value) == expected, f"{head}/excursion mismatch")


def _probabilities(frame: pd.DataFrame, indices: list[int], arm: str) -> dict:
    labeled = [i for i in indices if frame.at[i, "label_status"] == "labeled"]
    result = {"candidate_rows": len(indices), "labeled_rows": len(labeled),
              "unlabeled_rows": len(indices) - len(labeled), "heads": {}, "raw_heads": {}}
    for head in HEADS:
        for target, name in (("heads", _column(f"p_{head}", arm)), ("raw_heads", f"p_{head}_{arm}_raw")):
            result[target][head] = (_classification(frame.loc[labeled, head].to_numpy(float), frame.loc[labeled, name].to_numpy(float))
                                    if name in frame and labeled else {"status": "not_recorded_or_no_labels"})
    return result


def _sensitivity(frame: pd.DataFrame, plans: list[dict], positive: str, negative: str, differences: np.ndarray) -> dict:
    n = len(plans)
    contributions = {}
    for plan in plans:
        for arm, sign in ((positive, 1), (negative, -1)):
            for i, value in zip(plan["selected"][arm], _metric_rows(frame, plan["selected"][arm]), strict=True):
                contributions.setdefault(frame.at[i, "coin"], np.zeros(len(METRICS)))
                contributions[frame.at[i, "coin"]] += sign * value / (3 * n)
    summed = np.sum(list(contributions.values()), axis=0)
    _require(np.allclose(summed, differences.mean(axis=0), atol=1e-12, rtol=1e-10), "coin contribution sum mismatch")
    return {
        "leave_one_date_out": [{"omitted_date": plan["date"], "mean_delta": _named(np.delete(differences, i, axis=0).mean(axis=0))}
                               for i, plan in enumerate(plans)] if n > 1 else [],
        "coin_contributions": {coin: _named(values) for coin, values in sorted(contributions.items())},
        "sum_coin_contributions": _named(summed),
        "max_absolute_day_contribution": {metric: {"date": plans[int(np.argmax(np.abs(differences[:, j])))]["date"],
                                                    "contribution_to_mean": float(differences[int(np.argmax(np.abs(differences[:, j]))), j] / n)}
                                           for j, metric in enumerate(METRICS)},
        "max_absolute_coin_contribution": {metric: {"coin": max(contributions, key=lambda c: abs(contributions[c][j])),
                                                     "contribution_to_mean": float(contributions[max(contributions, key=lambda c: abs(contributions[c][j]))][j])}
                                            for j, metric in enumerate(METRICS)},
    }


def _cohort(frame: pd.DataFrame, plans: list[dict], excluded: list[dict], *, baseline: bool, n_boot: int, seed: int) -> dict:
    name = "baseline_common" if baseline else "selected_only"
    membership = [{"date": p["date"], "snapshot_id": p["snapshot_id"],
                   "selected": {arm: frame.loc[p["selected"][arm], "coin"].tolist() for arm in ARMS}}
                  for p in plans]
    digest = _digest({"cohort": name, "membership": membership})
    result = {"name": name, "cohort_id": f"{name}-{digest[:20]}", "cohort_sha256": digest,
              "dates": len(plans), "date_list": [p["date"] for p in plans], "membership": membership,
              "picks_per_arm": 3 * len(plans), "excluded": excluded,
              "status": "historical_comparison" if plans else "no_common_evaluable_dates"}
    if not plans:
        return result
    values = {arm: np.array([_metric_rows(frame, p["selected"][arm]).mean(axis=0) for p in plans]) for arm in ARMS}
    all_indices = [i for p in plans for i in p["indices"]]
    result.update(
        daily=[{"date": p["date"], "chronological_group": p["chronological_group"],
                "arms": {arm: _named(values[arm][i]) for arm in ARMS}} for i, p in enumerate(plans)],
        absolute={arm: _summary(values[arm], n_boot, seed) for arm in ARMS},
        contrasts={key: _summary(values[pos] - values[neg], n_boot, seed) for key, (pos, neg) in CONTRASTS.items()},
        sensitivity={key: _sensitivity(frame, plans, pos, neg, values[pos] - values[neg]) for key, (pos, neg) in CONTRASTS.items()},
        probability_diagnostics={arm: {"all_labeled_candidates": _probabilities(frame, all_indices, arm),
                                      "selected_top3": _probabilities(frame, [i for p in plans for i in p["selected"][arm]], arm)} for arm in ARMS},
    )
    result["chronological_groups"] = []
    for group in sorted({p["chronological_group"] for p in plans}):
        mask = np.array([p["chronological_group"] == group for p in plans])
        result["chronological_groups"].append({"group": group, "dates": int(mask.sum()),
            "date_list": [p["date"] for p, keep in zip(plans, mask) if keep],
            "arms": {arm: _named(values[arm][mask].mean(axis=0)) for arm in ARMS},
            "contrasts": {key: _named((values[pos] - values[neg])[mask].mean(axis=0)) for key, (pos, neg) in CONTRASTS.items()}})
    if baseline:
        full = np.array([_metric_rows(frame, p["indices"]).mean(axis=0) for p in plans])
        matched = {arm: np.array([np.mean([_metric_rows(frame, pool).mean(axis=0) for pool in p["matched"][arm]], axis=0)
                                 for p in plans]) for arm in ARMS}
        result.update(full_universe=_summary(full, n_boot, seed),
                      matched={arm: _summary(matched[arm], n_boot, seed) for arm in ARMS},
                      delta_vs_full_universe={arm: _summary(values[arm] - full, n_boot, seed) for arm in ARMS},
                      within_vol_liquidity_lift={arm: _summary(values[arm] - matched[arm], n_boot, seed) for arm in ARMS})
        for i, daily in enumerate(result["daily"]):
            daily.update(full_universe=_named(full[i]), matched={arm: _named(matched[arm][i]) for arm in ARMS})
    return result


def evaluate_horizon(predictions: pd.DataFrame, *, n_boot: int = 4000, seed: int = 42) -> dict:
    """Compare all-head scores on two predeclared common historical cohorts."""
    _require(type(n_boot) is int and 1 <= n_boot <= 100_000, "invalid n_boot")
    _require(type(seed) is int and seed >= 0, "invalid seed")
    frame = _validate(predictions)
    plans = _plans(frame)
    _validate_extra_outcomes(frame)
    accepted = {"primary": [], "supplemental": []}
    excluded = {"primary": [], "supplemental": []}
    audit = []
    for plan in plans:
        selected_reasons, baseline_reasons = [], []
        for arm in ARMS:
            picks = plan["selected"][arm]
            if picks is None:
                selected_reasons.append(f"arm_unavailable:{arm}")
                continue
            selected_reasons.extend(f"selected_label_unavailable:{arm}:{frame.at[i, 'coin']}" for i in picks if frame.at[i, "label_status"] != "labeled")
            for i, pool in zip(picks, plan["matched"][arm], strict=True):
                if len(pool) < 3:
                    baseline_reasons.append(f"matched_cell_unsupported:{arm}:{frame.at[i, 'coin']}:{len(pool)}")
        missing = [frame.at[i, "coin"] for i in plan["indices"] if frame.at[i, "label_status"] != "labeled"]
        if missing:
            baseline_reasons.append("full_universe_label_unavailable:" + ",".join(missing))
        for name, reasons in (("primary", selected_reasons + baseline_reasons), ("supplemental", selected_reasons)):
            if reasons:
                excluded[name].append({"date": plan["date"], "snapshot_id": plan["snapshot_id"], "reasons": reasons})
            else:
                accepted[name].append(plan)
        audit.append({"date": plan["date"], "snapshot_id": plan["snapshot_id"], "chronological_group": plan["chronological_group"],
                      "universe_rows": len(plan["indices"]),
                      "selected_coins": {arm: None if picks is None else frame.loc[picks, "coin"].tolist() for arm, picks in plan["selected"].items()},
                      "matched_pool_sizes": {arm: None if pools is None else [len(pool) for pool in pools] for arm, pools in plan["matched"].items()}})
    result = {"schema": "recommend_horizon_eval.v1", "status": "historical_comparison" if accepted["primary"] else "no_primary_common_dates",
              "execution_status": "completed", "data_availability": "primary_available" if accepted["primary"] else "insufficient_primary",
              "deployable": False, "promotion_status": "NOT_EVALUATED", "automatic_promotion": False,
              "is_untouched_holdout": False, "configuration": deepcopy(EVALUATION_CONFIG), "n_boot": n_boot, "seed": seed,
              "input": {"rows": len(frame), "dates": len(plans), "actual_delivered_rows": int(frame.was_delivered.sum())},
              "selection_audit": audit,
              "limitations": ["Repeatedly observed retrospective history, not a new forward test.",
                               "Pick returns already include costs; not actual trades, compounded portfolio returns or portfolio Sharpe.",
                               "Primary and supplemental populations differ; do not pool or choose the more favorable result.",
                               "Five-observed-date groups and block3 need not be contiguous calendar days.",
                               "Probability diagnostics are pooled descriptive measures; selected and universe AUC are not directly comparable.",
                               "One fixed setting here does not erase earlier research on the same dates; daily fits are not independent trials."]}
    for name in accepted:
        result[name] = _cohort(frame, accepted[name], excluded[name], baseline=name == "primary", n_boot=n_boot, seed=seed)
    json.dumps(result, allow_nan=False)
    return result

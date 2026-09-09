"""Offline, date-paired evaluation of frozen R1 and two upside experiments.

Selection reads only decision-time predictors. Outcome availability is checked
after every arm and matched pool has been selected; it never changes a pick.
Returns are existing cost-adjusted pick proxies, not a traded portfolio.
"""
from __future__ import annotations

from datetime import date
from numbers import Real

import numpy as np
import pandas as pd


ARMS = ("A", "B", "C")
METRICS = ("up10", "dn5", "whole_path_safe_up10", "tp5_sl3_return_net",
           "eod_return_net", "mae")
RATIO_FLOOR = 0.001
MATCH_MIN_POOL = 3
MIN_CI_DATES = 5
PREDICTORS = ("rank", "fold", "p_up10", "p_dn5", "p_dn10", "exp_downside",
              "rr_ratio", "p_up10_A", "p_up10_B", "p_up10_C",
              "f_log_qv", "f_atr_pct_14")
OUTCOMES = ("up10", "dn5", "mfe", "mae", "eod_return_net", "tp5_sl3_return_net")


class ExperimentEvaluationError(ValueError):
    """Invalid evidence blocks the complete comparison, not individual rows."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ExperimentEvaluationError(reason)


def _finite(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) and bool(np.isfinite(value))


def _validate_predictors(predictions: pd.DataFrame) -> pd.DataFrame:
    _require(isinstance(predictions, pd.DataFrame) and not predictions.empty, "empty or invalid prediction frame")
    required = set(PREDICTORS + OUTCOMES) | {"date", "slot", "coin", "snapshot_id", "was_delivered", "label_status"}
    _require(required.issubset(predictions.columns), f"missing columns: {sorted(required - set(predictions.columns))}")
    _require(predictions.columns.is_unique, "duplicate columns")
    frame = predictions.copy(deep=True).reset_index(drop=True)
    for field in ("date", "slot", "coin", "snapshot_id"):
        _require(frame[field].map(lambda x: isinstance(x, str) and bool(x)).all(), f"invalid {field}")
    for day in frame["date"].unique():
        try:
            _require(date.fromisoformat(day).isoformat() == day, "date must be ISO YYYY-MM-DD")
        except ValueError as exc:
            raise ExperimentEvaluationError("date must be ISO YYYY-MM-DD") from exc
    _require(frame["slot"].isin(["open", "preopen"]).all(), "invalid slot")
    _require(~frame.duplicated(["date", "slot", "coin"]).any(), "duplicate date/slot/coin")
    _require(frame["was_delivered"].map(lambda x: isinstance(x, (bool, np.bool_))).all(), "was_delivered must be boolean")
    for name in PREDICTORS:
        if name in ("p_up10_B", "p_up10_C"):
            continue
        _require(frame[name].map(_finite).all(), f"non-finite or invalid predictor: {name}")
    for name in ("p_up10", "p_dn5", "p_dn10", "p_up10_A"):
        _require(frame[name].between(0, 1).all(), f"invalid probability: {name}")
    for arm in ("B", "C"):
        status, probability = f"{arm}_status", f"p_up10_{arm}"
        if status not in frame:
            frame[status] = "ok"
        _require(frame[status].isin(["ok", "unavailable"]).all(), f"invalid arm status: {arm}")
        _require(frame.groupby(["fold", "slot"])[status].nunique().eq(1).all(), f"mixed fold/slot arm status: {arm}")
        available = frame[status].eq("ok")
        _require(frame.loc[available, probability].map(_finite).all(), f"non-finite or invalid predictor: {probability}")
        _require(frame.loc[available, probability].between(0, 1).all(), f"invalid probability: {probability}")
        _require(frame.loc[~available, probability].isna().all(), f"unavailable arm must contain only missing probabilities: {arm}")
    _require((frame["p_up10_A"] == frame["p_up10"]).all(), "A must preserve the frozen p_up10")
    _require((frame["rr_ratio"] >= 0).all(), "negative ratio")
    for name in ("rank", "fold"):
        _require(((frame[name] > 0) & (frame[name] == np.floor(frame[name]))).all(), f"invalid {name}")
    for _, group in frame.groupby(["date", "slot"], sort=True):
        _require(group["snapshot_id"].nunique() == group["fold"].nunique() == 1, "mixed snapshot or fold in date/slot")
        _require(sorted(group["rank"].tolist()) == list(range(1, len(group) + 1)), "candidate ranks must be complete and unique")
        _require(group["was_delivered"].sum() == 3, "exactly three actual delivered picks required")
        _require(set(group.loc[group["was_delivered"], "rank"]) == {1, 2, 3}, "delivered picks must match original Top3")
    _require(frame.groupby("date")["fold"].nunique().eq(1).all(), "cross-slot date assigned to different folds")
    _require(frame.groupby("snapshot_id")[["date", "slot"]].nunique().eq(1).all().all(), "snapshot reused across dates/slots")
    # Stable aggregation order also makes the serialized report reproducible
    # when the caller supplies the same evidence in a different row order.
    return frame.sort_values(["date", "slot", "rank"], kind="stable").reset_index(drop=True)


def _rank(group: pd.DataFrame, arm: str) -> list[int]:
    return sorted(group.index, key=lambda i: (
        -float(group.at[i, f"p_up10_{arm}"]) / max(float(group.at[i, "p_dn5"]), RATIO_FLOOR),
        float(group.at[i, "p_dn10"]), -float(group.at[i, f"p_up10_{arm}"]),
        -float(group.at[i, "exp_downside"]), int(group.at[i, "rank"]),
    ))[:3]


def _selection_plans(frame: pd.DataFrame) -> list[dict]:
    """This stage does not read any outcome or label-status column."""
    plans = []
    for (day, slot), group in frame.groupby(["date", "slot"], sort=True):
        actual = group.loc[group["was_delivered"]].sort_values("rank").index.tolist()
        _require(_rank(group, "A") == actual, f"identity Top3 parity failed: {day}/{slot}")
        choices = {"A": actual, **{arm: _rank(group, arm) if group[f"{arm}_status"].iloc[0] == "ok" else None
                                  for arm in ("B", "C")}}
        quartiles = {}
        for feature in ("f_log_qv", "f_atr_pct_14"):
            pct = group[feature].rank(method="average", pct=True)
            quartiles[feature] = np.clip(np.ceil(pct * 4).astype(int) - 1, 0, 3)
        cells = {i: (int(quartiles["f_log_qv"].at[i]), int(quartiles["f_atr_pct_14"].at[i])) for i in group.index}
        pools = {arm: None if picks is None else [[j for j in group.index if cells[j] == cells[i]] for i in picks]
                 for arm, picks in choices.items()}
        plans.append({"date": day, "slot": slot, "fold": int(group["fold"].iloc[0]),
                      "indices": group.index.tolist(), "selected": choices, "matched": pools})
    return plans


def _validate_outcomes(frame: pd.DataFrame) -> None:
    _require(frame["label_status"].isin(["labeled", "halted_no_observations"]).all(), "unknown label_status")
    for i, row in frame.iterrows():
        if row["label_status"] == "halted_no_observations":
            _require(all(pd.isna(row[name]) for name in OUTCOMES), f"halted row has outcomes: {i}")
            continue
        for name in OUTCOMES:
            value = row[name]
            valid = isinstance(value, (bool, np.bool_)) if name in ("up10", "dn5") else False
            _require(valid or _finite(value), f"invalid labeled outcome: {i}/{name}")
        _require(row["up10"] in (0, 1) and row["dn5"] in (0, 1), "invalid binary outcome")
        _require(row["mfe"] >= 0 and -1 <= row["mae"] <= 0, "invalid excursion")
        _require(bool(row["up10"]) == (row["mfe"] >= 0.10), "up10/MFE mismatch")
        _require(bool(row["dn5"]) == (row["mae"] <= -0.05), "dn5/MAE mismatch")


def _metric_rows(frame: pd.DataFrame, indices: list[int]) -> np.ndarray:
    rows = frame.loc[indices]
    return np.column_stack((rows["up10"].astype(float), rows["dn5"].astype(float),
                            (rows["up10"].astype(bool) & ~rows["dn5"].astype(bool)).astype(float),
                            rows["tp5_sl3_return_net"].astype(float), rows["eod_return_net"].astype(float),
                            rows["mae"].astype(float)))


def _named(values: np.ndarray) -> dict:
    return {metric: float(value) for metric, value in zip(METRICS, values, strict=True)}


def _summary(values: np.ndarray, n_boot: int, seed: int) -> dict:
    n = len(values)
    result = {"dates": n, "mean": _named(values.mean(axis=0)),
              "ci_status": "available" if n >= MIN_CI_DATES else "insufficient_dates",
              "iid_date_ci95": None, "observed_date_block3_ci95": None}
    if n < MIN_CI_DATES:
        return result
    rng = np.random.default_rng(seed)
    iid = values[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    length = min(3, n)
    starts = rng.integers(0, n - length + 1, size=(n_boot, int(np.ceil(n / length))))
    indices = (starts[:, :, None] + np.arange(length)).reshape(n_boot, -1)[:, :n]
    blocks = values[indices].mean(axis=1)
    for name, samples in (("iid_date_ci95", iid), ("observed_date_block3_ci95", blocks)):
        low, high = np.quantile(samples, [0.025, 0.975], axis=0)
        result[name] = {key: [float(a), float(b)] for key, a, b in zip(METRICS, low, high, strict=True)}
    return result


def _classification(y: np.ndarray, probability: np.ndarray) -> dict:
    n, positive = len(y), int(y.sum())
    auc = None
    if 0 < positive < n:
        ranks = pd.Series(probability).rank(method="average").to_numpy()
        auc = float((ranks[y == 1].sum() - positive * (positive + 1) / 2) / (positive * (n - positive)))
    return {"rows": n, "positive": positive, "negative": n - positive,
            "mean_prediction": float(probability.mean()), "observed_rate": float(y.mean()),
            "auc": auc, "auc_status": "available" if auc is not None else "single_class",
            "brier": float(np.mean((probability - y) ** 2))}


def _diagnostics(frame: pd.DataFrame, indices: list[int], arm: str) -> dict:
    rows = frame.loc[indices]
    return {"dates": int(rows["date"].nunique()),
            "up10": _classification(rows["up10"].to_numpy(float), rows[f"p_up10_{arm}"].to_numpy(float)),
            "dn5": _classification(rows["dn5"].to_numpy(float), rows["p_dn5"].to_numpy(float))}


def _sensitivity(frame: pd.DataFrame, plans: list[dict], arm: str, differences: np.ndarray) -> dict:
    n = len(plans)
    leave_one = []
    if n > 1:
        for i, plan in enumerate(plans):
            leave_one.append({"omitted_date": plan["date"],
                              "mean_delta": _named(np.delete(differences, i, axis=0).mean(axis=0))})
    contributions: dict[str, np.ndarray] = {}
    for plan in plans:
        for choice, sign in ((arm, 1), ("A", -1)):
            picks = plan["selected"][choice]
            for i, value in zip(picks, _metric_rows(frame, picks), strict=True):
                coin = frame.at[i, "coin"]
                contributions.setdefault(coin, np.zeros(len(METRICS)))
                contributions[coin] += sign * value / (3 * n)
    max_day, max_coin = {}, {}
    for j, metric in enumerate(METRICS):
        i = int(np.argmax(np.abs(differences[:, j])))
        max_day[metric] = {"date": plans[i]["date"], "daily_delta": float(differences[i, j]),
                           "contribution_to_mean_delta": float(differences[i, j] / n)}
        coin = max(contributions, key=lambda c: abs(contributions[c][j]))
        max_coin[metric] = {"coin": coin, "contribution_to_mean_delta": float(contributions[coin][j]),
                            "sum_all_coin_contributions": float(sum(v[j] for v in contributions.values()))}
    return {"leave_one_date_out": leave_one, "max_absolute_day_contribution": max_day,
            "max_absolute_coin_contribution": max_coin}


def evaluate_comparison(predictions: pd.DataFrame, *, n_boot: int = 1000, seed: int = 42) -> dict:
    """Evaluate historical arms on a strict common date/slot cohort, without IO."""
    _require(type(n_boot) is int and 1 <= n_boot <= 100_000, "n_boot must be in [1,100000]")
    _require(type(seed) is int and seed >= 0, "seed must be a nonnegative integer")
    frame = _validate_predictors(predictions)
    plans = _selection_plans(frame)
    _validate_outcomes(frame)
    valid, excluded, audit, daily = [], [], [], []
    for plan in plans:
        reasons = []
        for arm in ARMS:
            if plan["selected"][arm] is None:
                reasons.append(f"arm_unavailable:{arm}")
                continue
            for index, pool in zip(plan["selected"][arm], plan["matched"][arm], strict=True):
                if len(pool) < MATCH_MIN_POOL:
                    reasons.append(f"matched_cell_unsupported:{arm}:{frame.at[index, 'coin']}:{len(pool)}")
                if frame.at[index, "label_status"] != "labeled":
                    reasons.append(f"selected_label_unavailable:{arm}:{frame.at[index, 'coin']}")
                if any(frame.at[i, "label_status"] != "labeled" for i in pool):
                    reasons.append(f"matched_label_unavailable:{arm}:{frame.at[index, 'coin']}")
        unavailable = [frame.at[i, "coin"] for i in plan["indices"] if frame.at[i, "label_status"] != "labeled"]
        if unavailable:
            reasons.append("full_universe_label_unavailable:" + ",".join(unavailable))
        audit.append({"date": plan["date"], "slot": plan["slot"], "fold": plan["fold"],
                      "selected_coins": {arm: None if plan["selected"][arm] is None else frame.loc[plan["selected"][arm], "coin"].tolist() for arm in ARMS},
                      "matched_pool_sizes": {arm: None if plan["matched"][arm] is None else [len(pool) for pool in plan["matched"][arm]] for arm in ARMS},
                      "universe_rows": len(plan["indices"]), "evaluable": not reasons})
        if reasons:
            excluded.append({"date": plan["date"], "slot": plan["slot"], "fold": plan["fold"], "reasons": reasons})
            continue
        valid.append(plan)
        daily.append({"date": plan["date"], "slot": plan["slot"], "fold": plan["fold"],
                      "arms": {arm: _named(_metric_rows(frame, plan["selected"][arm]).mean(axis=0)) for arm in ARMS},
                      "full_universe": _named(_metric_rows(frame, plan["indices"]).mean(axis=0)),
                      "matched": {arm: _named(np.mean([_metric_rows(frame, pool).mean(axis=0)
                                                       for pool in plan["matched"][arm]], axis=0)) for arm in ARMS}})
    report = {
        "schema": "recommend_experiment_eval.v1", "status": "historical_comparison" if valid else "no_common_evaluable_dates",
        "execution_status": "completed",
        "data_availability": "historical_common_cohort_available" if valid else "insufficient_no_common_dates",
        "deployable": False, "promotion_status": "NOT_EVALUATED", "automatic_adoption": False,
        "input": {"rows": len(frame), "dates": int(frame["date"].nunique()), "date_slots": len(plans),
                  "folds": sorted(int(x) for x in frame["fold"].unique()), "actual_delivered_rows": int(frame["was_delivered"].sum())},
        "coverage": {"included_date_slots": len(valid), "excluded_date_slots": len(excluded)},
        "methodology": {
            "cohort": "all arms and all baselines share the same fully evaluable date/slot intersection",
            "selection": "all picks, arm availability and matching pools fixed before any outcome or label-status is read",
            "model_availability": "declared unavailable fold/slot arms have only missing probabilities and exclude that date/slot for all comparisons; undeclared or inconsistent missingness is invalid",
            "ranking": "A=actual Top3; B/C=(-p_up10/max(p_dn5,0.001),p_dn10,-p_up10,-exp_downside,original_rank)",
            "identity_top3_parity": "required for every input snapshot; full-precision full-universe ordering is not recovered",
            "matching": "same-snapshot log_qv and ATR average-percentile joint quartiles; expected random mean includes selected coins; each of three pools weighted 1/3",
            "quartile_boundaries": "(0,.25],(.25,.5],(.5,.75],(.75,1]; no label-dependent eligibility or widening",
            "matching_min_pool_initial": MATCH_MIN_POOL,
            "weighting": "equal three picks within date/slot, then equal dates; slots reported separately",
            "returns": "existing net fractional pick-return proxies; no extra cost deduction, compounding, portfolio Sharpe or drawdown",
            "metric_units": "rates and returns are fractions, not percentage points; MAE is a nonpositive excursion fraction",
            "favorable_delta_direction": {metric: "lower" if metric == "dn5" else "higher" for metric in METRICS},
            "whole_path_safe_up10": "MFE>=0.10 and MAE>-0.05 over the complete canonical 24h; not first passage",
            "ci": "paired mean deltas; IID dates and non-circular moving blocks of 3 observed dates; percentile95; diagnostic only",
            "block_caveat": "observed dates may be irregularly spaced; 3 observations do not necessarily mean 3 consecutive calendar days",
            "min_ci_dates_initial": MIN_CI_DATES, "bootstraps": n_boot, "seed": seed,
            "auc": "descriptive pooled AUC/Brier on common dates; all-candidate and selected-Top3 cohorts separate; no IID-coin confidence claim",
            "monotonic_B": "within a snapshot strictly monotonic recalibration preserves upside AUC except clipping ties; pooled across differently calibrated folds may change",
            "holdout": "already-observed historical comparison, not an untouched holdout or actual forward test of B/C",
            "input_scope": "outer validation predictions only; input dates are not the full development-history date count",
            "configuration": "fixed evaluation contract; no sweeps, adaptive thresholds or automatic verdict",
        },
        "selection_audit": audit, "excluded": excluded, "daily": daily, "slots": {},
    }
    for slot in sorted(frame["slot"].unique()):
        slot_plans = [p for p in valid if p["slot"] == slot]
        records = [r for r in daily if r["slot"] == slot]
        if not records:
            report["slots"][slot] = {"status": "no_common_evaluable_dates", "dates": 0}
            continue
        values = {arm: np.array([[r["arms"][arm][m] for m in METRICS] for r in records]) for arm in ARMS}
        full = np.array([[r["full_universe"][m] for m in METRICS] for r in records])
        matched = {arm: np.array([[r["matched"][arm][m] for m in METRICS] for r in records]) for arm in ARMS}
        all_indices = [i for p in slot_plans for i in p["indices"]]
        folds = []
        for fold in sorted({r["fold"] for r in records}):
            mask = np.array([r["fold"] == fold for r in records])
            folds.append({"fold": fold, "dates": int(mask.sum()),
                          "arms": {arm: _named(values[arm][mask].mean(axis=0)) for arm in ARMS},
                          "delta_vs_A": {arm: _named((values[arm] - values["A"])[mask].mean(axis=0)) for arm in ("B", "C")}})
        report["slots"][slot] = {
            "status": "historical_comparison", "dates": len(records), "picks_per_arm": 3 * len(records),
            "included_dates": [r["date"] for r in records], "all_candidate_rows": len(all_indices),
            "absolute": {arm: _summary(values[arm], n_boot, seed) for arm in ARMS},
            "delta_vs_A": {arm: _summary(values[arm] - values["A"], n_boot, seed) for arm in ("B", "C")},
            "full_universe": _summary(full, n_boot, seed),
            "matched": {arm: _summary(matched[arm], n_boot, seed) for arm in ARMS},
            "delta_vs_full_universe": {arm: _summary(values[arm] - full, n_boot, seed) for arm in ARMS},
            "within_vol_liquidity_lift": {arm: _summary(values[arm] - matched[arm], n_boot, seed) for arm in ARMS},
            "probability_diagnostics": {arm: {
                "all_candidates": _diagnostics(frame, all_indices, arm),
                "selected_top3": _diagnostics(frame, [i for p in slot_plans for i in p["selected"][arm]], arm),
            } for arm in ARMS},
            "folds": folds,
            "sensitivity_vs_A": {arm: _sensitivity(frame, slot_plans, arm, values[arm] - values["A"]) for arm in ("B", "C")},
        }
    return report

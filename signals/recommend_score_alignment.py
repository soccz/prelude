"""One outcome-blind score-scale ablation; never a calibrated probability.

Use C's within-snapshot order and A's contemporaneously available score levels.
This is a cross-sectional transformation, not a fitted quantile threshold.
"""
from __future__ import annotations

from math import fsum
from numbers import Real

import numpy as np
import pandas as pd

from signals.recommend_experiment_eval import METRICS, _summary, evaluate_comparison

CONFIG = {
    "schema": "recommend_score_alignment.v1",
    "method": "sorted_reference_levels_with_source_tie_group_mean",
    "group": ["date", "slot"],
    "source": "saved_C_calibrated_upper_score", "reference": "same_snapshot_A_upper_score",
    "score_is_calibrated_probability": False, "fitted_parameters": 0,
    "new_candidates": 1, "sweep": False, "top_k": 3,
    "ratio_floor": 0.001, "unchanged_downside_head": True,
    "arm_mapping": {"A": "original_R1", "B": "original_upper_C", "C": "aligned_upper_D"},
}
KEYS = ["date", "slot", "coin", "snapshot_id", "rank", "fold"]


def align_scores(source, reference) -> np.ndarray:
    """Preserve source ties/order and reference mean; use neither outcomes nor IO."""
    arrays = []
    for values in (source, reference):
        values = np.asarray(values, dtype=object)
        if (values.ndim != 1 or not len(values)
                or any(not isinstance(v, Real) or isinstance(v, (bool, np.bool_))
                       or not np.isfinite(v) or not 0 <= v <= 1 for v in values)):
            raise ValueError("score arrays must be nonempty finite numbers in [0,1]")
        arrays.append(values.astype(float))
    raw, target = arrays
    if len(raw) != len(target):
        raise ValueError("source/reference population sizes differ")
    order, levels = np.argsort(raw, kind="stable"), np.sort(target)
    result = np.empty(len(raw), dtype=float)
    start = 0
    while start < len(raw):
        end = start + 1
        while end < len(raw) and raw[order[end]] == raw[order[start]]:
            end += 1
        # Identical reference levels stay bit-exact, including the A->A control.
        value = (levels[start] if levels[start] == levels[end - 1]
                 else fsum(levels[start:end]) / (end - start))
        result[order[start:end]] = value
        start = end
    return result


def transform_predictions(predictions: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Return evaluator columns A=R1, B=old C, C=aligned D; preserve other data."""
    required = KEYS + ["p_up10_A", "p_up10_C", "C_status"]
    if (not isinstance(predictions, pd.DataFrame) or predictions.empty
            or not predictions.columns.is_unique or not set(required) <= set(predictions)):
        raise ValueError("missing or invalid alignment predictor frame")
    output = predictions.copy(deep=True).reset_index(drop=True)
    scores = output[required].copy(deep=True)
    if scores[KEYS].isna().any().any() or scores.duplicated(["date", "slot", "coin"]).any():
        raise ValueError("missing or duplicate candidate identity")
    output["p_up10_B"] = scores["p_up10_C"]
    output["B_status"] = scores["C_status"]
    if "p_up10_C_raw" in output:
        output = output.rename(columns={"p_up10_C_raw": "original_upper_C_raw"})
    audit = []
    for (day, slot), group in scores.groupby(["date", "slot"], sort=True):
        if (group["snapshot_id"].nunique() != 1 or group["fold"].nunique() != 1
                or sorted(group["rank"].tolist()) != list(range(1, len(group) + 1))):
            raise ValueError("incomplete or mixed snapshot population")
        if group["C_status"].nunique() != 1:
            raise ValueError("mixed source status")
        status = group["C_status"].iloc[0]
        reference = group["p_up10_A"].to_numpy()
        # Validate reference even when C is unavailable; never heal missing A.
        align_scores(reference, reference)
        item = {"date": day, "slot": slot, "candidates": len(group), "status": status}
        if status == "unavailable":
            if not group["p_up10_C"].isna().all():
                raise ValueError("unavailable C must have only missing scores")
            audit.append(item)
            continue
        if status != "ok":
            raise ValueError("invalid source status")
        raw = group["p_up10_C"].to_numpy()
        aligned = align_scores(raw, reference)
        output.loc[group.index, "p_up10_C"] = aligned
        item.update(reference_mean=float(np.mean(reference)), source_mean=float(np.mean(raw)),
                    aligned_mean=float(np.mean(aligned)), source_unique=int(len(np.unique(raw))),
                    aligned_unique=int(len(np.unique(aligned))),
                    interpretation="order-preserving ranking score, not calibrated probability")
        audit.append(item)
    return output, audit


def evaluate_alignment(predictions: pd.DataFrame, *, n_boot: int = 1000, seed: int = 42) -> dict:
    """Same paired/matched evaluation contract, explicit new arm semantics."""
    evaluation = evaluate_comparison(predictions, n_boot=n_boot, seed=seed)
    evaluation["arm_mapping"] = CONFIG["arm_mapping"]
    methodology = evaluation["methodology"]
    methodology.pop("monotonic_B", None)
    methodology["alignment"] = CONFIG
    methodology["auc"] = "Descriptive score AUC only; mapped D is not a calibrated probability."
    for slot, result in evaluation["slots"].items():
        if not result["dates"]:
            continue
        diagnostics = result.pop("probability_diagnostics")
        for arm in diagnostics.values():
            for cohort in arm.values():
                for event in ("up10", "dn5"):
                    cohort[event].pop("brier", None)
                    cohort[event]["mean_score"] = cohort[event].pop("mean_prediction")
        result["score_diagnostics"] = diagnostics
        rows = [row for row in evaluation["daily"] if row["slot"] == slot]
        differences = np.array([[row["arms"]["C"][m] - row["arms"]["B"][m]
                                 for m in METRICS] for row in rows])
        result["aligned_D_minus_original_C"] = _summary(differences, n_boot, seed)
    return evaluation

"""One offline mechanical prototype, not an additional prospective trial.

Inputs must be trusted native snapshot/feature documents: the caller owns the
file graph, raw-data and source-hash checks. Reusing the existing pure validator
checks their in-memory contract but does NOT establish a durable prediction.
No files, labels, fitted models, live rankings or trial registries are changed.
"""
from __future__ import annotations

import copy
import math

from signals.recommend_microstructure import FEATURE
from signals.recommend_microstructure_trial import MicrostructureTrialError, _validate_feature

SCHEMA = "recommend_trade_shortlist_plan.v1"
SHORTLIST_CONFIG = {
    "scope": "offline_mechanical_prototype_only",
    "slot": "open",
    "universe_size": 100,
    "shortlist_size": 10,
    "shortlist_rule": "original_R1_ranks_1_through_10",
    "top_k": 3,
    "selection_order": [f"{FEATURE}:descending", "original_rank:ascending"],
    "feature": FEATURE,
    "lookback_seconds": 300,
    "missing_shortlist_feature": "whole_plan_unavailable_no_refill_or_imputation",
    "non_shortlist_null": "allowed_when_global_feature_evidence_valid",
    "saturation_policy": "diagnostic_only_no_min_count_or_notional_filter",
    "shortlist_size_basis": "top_decile_of_100_initial_not_outcome_optimized",
    "changes_model_or_labels": False,
    "changes_live_selection": False,
    "publishes_prospective_score": False,
    "automatic_promotion": False,
}


class TradeShortlistError(ValueError):
    """Contradictory or malformed inputs are not a missing observation."""


def _stored_context(candidate, name):
    """Optional context only; never impute, reconstruct or use it for ranking."""
    values = candidate.get("feature_values")
    value = values.get(name) if isinstance(values, dict) else None
    return value if type(value) in (int, float) and math.isfinite(value) else None


def plan_trade_shortlist(snapshot, feature_document):
    """Rank a fixed ten-name shortlist by one feature without observing outcomes.

Every shortlist feature is required before choosing any names. Missing features
outside that original shortlist do not alter its eligibility. Negative values,
genuine zero and saturated +/-1 values are all ranked literally; a single tiny
one-sided trade can therefore outrank substantial two-sided flow. This known
limitation is reported, not silently repaired with an untested filter.
"""
    try:
        candidates, features = _validate_feature(snapshot, feature_document)
        shortlist = candidates[:10]
        missing = [row["coin"] for row in shortlist if features[row["coin"]][FEATURE] is None]
        reason = (
            "feature_evidence_invalid" if not feature_document["feature_evidence_valid"]
            else "required_shortlist_feature_unavailable" if missing else None
        )
        control = [row["coin"] for row in candidates]
        challenger = None
        if reason is None:
            selected_order = sorted(
                shortlist, key=lambda row: (-features[row["coin"]][FEATURE], row["rank"])
            )
            challenger = [row["coin"] for row in selected_order] + control[10:]
        changed = len(set(challenger[:3]) - set(control[:3])) if challenger is not None else None
        candidate_inputs = []
        for row in candidates:
            feature_row = features[row["coin"]]
            value = feature_row[FEATURE]
            candidate_inputs.append({
                "coin": row["coin"], "original_rank": row["rank"], FEATURE: value,
                "observed_trade_count": feature_row["observed_trade_count"],
                "bid_notional": feature_row["bid_notional"],
                "ask_notional": feature_row["ask_notional"],
                "total_notional": feature_row["bid_notional"] + feature_row["ask_notional"],
                "saturated": value is not None and abs(value) == 1,
                "single_trade": feature_row["observed_trade_count"] == 1,
                "stored_f_atr_pct_14": _stored_context(row, "f_atr_pct_14"),
                "stored_f_log_qv": _stored_context(row, "f_log_qv"),
            })
        selected_coins = set(challenger[:3]) if challenger is not None else set()
        diagnostics = {
            "shortlist_saturated_coins": [r["coin"] for r in candidate_inputs[:10] if r["saturated"]],
            "shortlist_single_trade_coins": [r["coin"] for r in candidate_inputs[:10] if r["single_trade"]],
            "selected_saturated_coins": [r["coin"] for r in candidate_inputs
                                          if r["coin"] in selected_coins and r["saturated"]],
            "selected_single_trade_coins": [r["coin"] for r in candidate_inputs
                                             if r["coin"] in selected_coins and r["single_trade"]],
            "context_source": "stored_snapshot_only_not_reconstructed_or_used_for_selection",
        }
        return {
            "schema": SCHEMA, "config": copy.deepcopy(SHORTLIST_CONFIG),
            "snapshot": copy.deepcopy(feature_document["snapshot"]),
            "status": "unavailable" if reason else "planned", "reason": reason,
            "control_ranking": control, "challenger_ranking": challenger,
            "control_top3": control[:3],
            "challenger_top3": challenger[:3] if challenger is not None else None,
            "shortlist": control[:10], "required_missing_features": missing,
            "changed_picks": changed, "is_no_op": changed == 0 if changed is not None else None,
            "coverage": {
                "recorded_candidates": 100, "shortlist_candidates": 10,
                "shortlist_available_features": 10 - len(missing),
                "feature_available_rows": feature_document["feature_available_rows"],
                "feature_coverage": feature_document["feature_coverage"],
            },
            "candidate_inputs": candidate_inputs, "diagnostics": diagnostics,
            "evidence_scope": "trusted_in_memory_documents_file_graph_and_raw_not_reverified",
            "experiment_readiness": "not_assessed", "effect_status": "not_evaluated",
            "prospective_prediction": False, "deployable": False,
            "limitations": [
                "Shortlist membership does not preserve realized downside, volatility or liquidity.",
                "A single tiny one-sided trade can have imbalance +/-1; this is not strong-flow evidence.",
                "Missing-shortlist days are unavailable, not zero-return or no-op observations.",
                "Later offline plans are not proof of scores durably available before entry.",
            ],
        }
    except MicrostructureTrialError as exc:
        raise TradeShortlistError(str(exc)) from exc
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise TradeShortlistError(f"invalid shortlist inputs: {type(exc).__name__}") from exc

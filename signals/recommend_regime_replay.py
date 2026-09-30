"""Causal, offline policy replay. No fitting, live selection, I/O or promotion.

Contexts use frozen universe features. Policies can see only completed past
labels of the same source version; today's outcome never determines its plan.
The already-observed replay cohort is not an untouched or prospective test.
"""

from __future__ import annotations

from collections import Counter
from datetime import date, datetime, timedelta, timezone
from statistics import median

import numpy as np

from signals.recommend_experiment_eval import METRICS, _finite, _summary

CONFIG = {
    "version": "r1_regime_replay_v1",
    "history_observations": 10,
    "minimum_observations": 5,
    "breadth_cut": 0.5,
    "turnover_ratio_cut": 1.0,
    "threshold_status": "initial_values_fixed_before_replay_no_sweep",
    "selection_rule": "higher_net_eod_and_no_worse_dn5_up10_mae",
}
POLICIES = ("fixed_r1", "recent", "same_context")
FEATURES = {
    "breadth": "f_ret_1d",
    "turnover_ratio": "f_qv_ma7_vs_ma30",
    "atr": "f_atr_pct_14",
    "range_position": "f_pos_in_20d_range",
}


def aware(value):
    result = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    if result.utcoffset() is None:
        raise ValueError("replay timestamps must be timezone-aware")
    return result.astimezone(timezone.utc)


def context_from_snapshot(snapshot):
    """Never silently shrink the breadth/turnover universe for missing values."""
    universe = snapshot["universe"]
    coins = [row["coin"] for row in universe]
    if len(coins) != len(set(coins)):
        raise ValueError("duplicate context coin")
    values, missing = {}, {}
    for name, feature in FEATURES.items():
        raw = [row.get("feature_values", {}).get(feature) for row in universe]
        if any(value is not None and not _finite(value) for value in raw):
            raise ValueError(f"invalid context feature: {feature}")
        finite = [float(value) for value in raw if value is not None]
        missing[name] = len(raw) - len(finite)
        values[name] = (
            (sum(value > 0 for value in finite) / len(finite)
             if name == "breadth" else median(finite))
            if finite else None
        )
    valid = bool(universe) and not (missing["breadth"] or missing["turnover_ratio"])
    state = None
    if valid:
        breadth = "broad" if values["breadth"] >= CONFIG["breadth_cut"] else "narrow"
        turnover = "expanding" if values["turnover_ratio"] >= CONFIG["turnover_ratio_cut"] else "contracting"
        state = f"{breadth}:{turnover}"
    return {
        "state": state, "universe_size": len(universe), "values": values,
        "missing": missing, "btc_regime": snapshot.get("btc_regime", "unknown"),
        "source": "frozen_snapshot_universe_not_current_database",
    }


def version_from_snapshot(snapshot):
    """Unknown source identity is never treated as a shared trainable version."""
    source = snapshot.get("code", {}).get("score_source_sha256")
    parts = [snapshot.get("model", {}).get("id"), snapshot.get("slot"),
             source, snapshot.get("rule_version"), snapshot.get("score_schema_version")]
    if (not isinstance(source, str) or len(source) != 64
            or any(char not in "0123456789abcdef" for char in source)
            or any(value is None or value == "" for value in parts)):
        return None
    return "|".join(str(value) for value in parts)


def _vector(metrics):
    if any(not _finite(metrics[key]) for key in METRICS):
        raise ValueError("non-finite daily metric")
    if any(not 0 <= metrics[key] <= 1 for key in ("up10", "dn5", "whole_path_safe_up10")):
        raise ValueError("invalid daily event rate")
    if not -1 <= metrics["mae"] <= 0:
        raise ValueError("invalid daily MAE")
    return np.array([metrics[key] for key in METRICS], dtype=float)


def _delta(history):
    values = np.array([_vector(row["outcomes"]["challenger"])
                       - _vector(row["outcomes"]["control"]) for row in history])
    return dict(zip(METRICS, values.mean(axis=0).tolist(), strict=True))


def selection_plans(records):
    """Plan first. Only strictly earlier, available labels may enter a choice.

    A record may have outcomes=None: its plan still exists without a result.
    No future-derived quantile thresholds or pooling across slots/versions.
    """
    rows = sorted(records, key=lambda row: row["date"])
    if len({row["date"] for row in rows}) != len(rows):
        raise ValueError("replay requires one common-slot decision per date")
    for row in rows:
        if date.fromisoformat(row["date"]).isoformat() != row["date"]:
            raise ValueError("noncanonical replay date")
        decision, entry = aware(row["decision_at"]), aware(row["entry_at"])
        if decision >= entry:
            raise ValueError("scores not available before common entry")
        if not row["predictors_available_at"] or any(aware(stamp) > decision for stamp in row["predictors_available_at"]):
            raise ValueError("future predictor at decision")
        if row.get("outcomes") is not None and aware(row["label_available_at"]) < entry + timedelta(days=1):
            raise ValueError("24-hour outcome available before path ends")
        if type(row["changed_picks"]) is not int or not 0 <= row["changed_picks"] <= 3:
            raise ValueError("invalid changed-pick count")
    plans = []
    for index, row in enumerate(rows):
        decision = aware(row["decision_at"])
        prior = [past for past in rows[:index]
                 if row["version"] is not None and past["version"] == row["version"]
                 and past.get("outcomes") is not None
                 and aware(past["label_available_at"]) < decision]
        choices = {"fixed_r1": {"arm": "control", "reason": "fixed", "history_dates": []}}
        for policy in POLICIES[1:]:
            history = [past for past in prior if policy == "recent"
                       or past["context"]["state"] == row["context"]["state"]]
            history = history[-CONFIG["history_observations"]:]
            delta, arm = None, "control"
            if row["version"] is None:
                reason = "unknown_source_version"
            elif row["context"]["state"] is None:
                reason = "unknown_context"
            elif len(history) < CONFIG["minimum_observations"]:
                reason = "insufficient_completed_history"
            else:
                delta = _delta(history)
                wins = (delta["eod_return_net"] > 0 and delta["dn5"] <= 0
                        and delta["up10"] >= 0 and delta["mae"] >= 0)
                arm, reason = ("challenger", "past_joint_dominance") if wins else ("control", "no_past_joint_dominance")
            choices[policy] = {
                "arm": arm, "reason": reason, "history_dates": [past["date"] for past in history],
                "latest_label_available_at": max((past["label_available_at"] for past in history),
                                                key=aware, default=None),
                "past_mean_delta": delta,
            }
        plans.append({"date": row["date"], "decision_at": row["decision_at"],
                      "entry_at": row["entry_at"], "version": row["version"],
                      "context": row["context"], "choices": choices})
    return plans


def replay(records, *, n_boot=1000, seed=42):
    if type(n_boot) is not int or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    plans = selection_plans(records)
    by_day = {row["date"]: row for row in records}
    paired = [plan for plan in plans if by_day[plan["date"]].get("outcomes") is not None]
    base = np.array([_vector(by_day[plan["date"]]["outcomes"]["control"]) for plan in paired])
    policies = {}
    for policy in POLICIES:
        choices = [plan["choices"][policy]["arm"] for plan in plans]
        values = np.array([_vector(by_day[plan["date"]]["outcomes"][plan["choices"][policy]["arm"]])
                           for plan in paired])
        changed = sum(by_day[plan["date"]]["changed_picks"] for plan in paired
                      if plan["choices"][policy]["arm"] == "challenger")
        policies[policy] = {
            "n_dates": len(paired), "planned_dates": len(plans),
            "challenger_dates": sum(arm == "challenger" for arm in choices),
            "arm_transitions_observed_dates": sum(a != b for a, b in zip(choices, choices[1:])),
            "changed_picks_evaluated": changed,
            "effect_status": "same_picks_as_fixed_r1_no_effect_observations" if not changed else "exploratory_paired_effect",
            "reason_counts": dict(Counter(plan["choices"][policy]["reason"] for plan in plans)),
            "metrics": _summary(values, n_boot, seed) if len(values) else None,
            "minus_fixed_r1": _summary(values - base, n_boot, seed) if len(values) else None,
        }
    return {
        "schema": "recommend_regime_replay.v1", "config": dict(CONFIG),
        "deployable": False, "automatic_promotion": False, "model_fitted": False,
        "is_untouched_holdout": False, "prospective_policy_records": 0,
        "additional_selector_hypotheses": 2,
        "scope": "retrospective_causal_replay_on_native_common_evidence_not_live_selection",
        "clock": "max(snapshot_created,decision_completed,challenger_score_durable) < canonical_entry",
        "history_rule": "same_version_and_max(path_window_end,labeled_at) < virtual_decision",
        "cost_basis": "existing_net_24h_pick_proxies_no_second_deduction_not_portfolio_returns",
        "cohort_warning": "native common-complete dates only; unavailable outcomes are not zero; no fresh holdout",
        "pending_dates": [plan["date"] for plan in plans if plan not in paired],
        "context_counts": dict(Counter(plan["context"]["state"] or "unknown" for plan in plans)),
        "policies": policies, "plans": plans,
    }


def failure_summary(records, *, n_boot=1000, seed=42):
    """Daily R1 versus its saved universe; descriptive, not liquidity-matched."""
    if type(n_boot) is not int or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    groups = {}
    for row in sorted(records, key=lambda item: (item["date"], item["slot"])):
        key = (row["slot"], row["version"], row["context"]["state"])
        groups.setdefault(key, []).append(row)
    result = []
    for (slot, version, state), rows in groups.items():
        control = np.array([_vector(row["outcomes"]["control"]) for row in rows])
        universe = np.array([_vector(row["outcomes"]["universe"]) for row in rows])
        result.append({
            "slot": slot, "version": version, "state": state, "n_dates": len(rows),
            "dates": [row["date"] for row in rows],
            "r1": _summary(control, n_boot, seed),
            "universe": _summary(universe, n_boot, seed),
            "r1_minus_universe": _summary(control - universe, n_boot, seed),
            "negative_r1_eod_dates": sum(row["outcomes"]["control"]["eod_return_net"] < 0 for row in rows),
        })
    return {
        "scope": "descriptive_correlation_not_failure_cause_or_filter_validation",
        "weighting": "equal_date_within_slot_and_source_version_not_pooled_coin_rows",
        "baseline": "same_frozen_universe_labeled_candidates_not_liquidity_matched",
        "groups": result,
    }

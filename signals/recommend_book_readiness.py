"""Fixed, descriptive review criteria; no fitting or automatic adoption.

Thirty dates and other counts are initial review minima, not power guarantees.
Uncertainty intervals inform a human review; they are not a rejection gate.
"""

from datetime import date, timedelta

import numpy as np

from signals import recommend_book_validation as comparison
from signals.recommend_experiment_eval import METRICS, _summary

START = date(2026, 10, 2)
CONFIG = {
    "schema": "recommend_book_forward_config.v2",
    "start_date": START.isoformat(),
    "revision_reason": "native_readiness_wait_after_20261001_missing_record",
    "supersedes_design_sha256": "0592e7718c8f062fb93f98cb0f9424d7fbbdd6e935ddec409f1d24d02326cef9",
    "native_readiness_wait_seconds": 180,
    "slot": "open",
    "selector": comparison.CONFIG["selector"].copy(),
    "min_paired_dates": 30,
    "min_changed_dates": 10,
    "min_dates_per_context": 5,
    "min_contexts": 2,
    "min_available_date_fraction": 0.8,
    "delays_minutes": [0, 15, 30],
    "extra_challenger_cost_fraction": 0.0015,
    "round_trip_cost_fraction": 0.0015,
    "threshold_status": "initial_review_minima_not_power_or_execution_guarantees",
    "n_boot": 1000,
    "seed": 42,
    "max_new_dates_per_run": 2,
    "automatic_promotion": False,
    "places_orders": False,
    "new_selectors": 0,
}


def due_dates(now):
    comparison.fixed.book._require(now.utcoffset() is not None, "aware clock required")
    through = now.astimezone(comparison.ZoneInfo("Asia/Seoul")).date() - timedelta(
        days=1
    )
    return [
        (START + timedelta(days=i)).isoformat()
        for i in range(max(0, (through - START).days + 1))
    ]


def joint(delta, *, strict_down=True):
    return (
        delta["eod_return_net"] > 0
        and (delta["dn5"] < 0 if strict_down else delta["dn5"] <= 0)
        and delta["up10"] >= 0
        and delta["whole_path_safe_up10"] >= 0
    )


def _mean(rows, arm="challenger_minus_control"):
    return {m: float(np.mean([r["metrics"][arm][m] for r in rows])) for m in METRICS}


def execution_summary(daily):
    # Identical full-day cohort across both arms AND every delay.
    complete = [d for d in daily if d["execution"]["status"] == "evaluated"]
    result = {"paired_dates": len(complete), "delays": None}
    if complete:
        result["delays"] = {}
        for delay in CONFIG["delays_minutes"]:
            key = str(delay)
            result["delays"][key] = {
                arm: _summary(
                    np.array(
                        [
                            [r["execution"]["metrics"][key][arm][m] for m in METRICS]
                            for r in complete
                        ]
                    ),
                    CONFIG["n_boot"],
                    CONFIG["seed"],
                )
                for arm in ("control", "challenger", "challenger_minus_control")
            }
    return result


def review(daily, *, integrity_ok=True):
    rows = sorted(
        (d for d in daily if d["result"]["status"] == "evaluated"),
        key=lambda d: d["date"],
    )
    results = [d["result"] for d in rows]
    summary = comparison.summarize(results)
    execution = execution_summary(rows)
    contexts = {}
    for state in sorted({d["context"]["state"] or "unknown" for d in rows}):
        values = [
            d["result"] for d in rows if (d["context"]["state"] or "unknown") == state
        ]
        contexts[state] = {"paired_dates": len(values), "difference": _mean(values)}
    segments = {}
    if len(rows) >= 2:
        middle = len(rows) // 2
        for key, values in (("earlier", results[:middle]), ("later", results[middle:])):
            segments[key] = {"paired_dates": len(values), "difference": _mean(values)}
    leave_one_out = None
    if len(rows) >= 2:
        leave_one_out = min(
            _mean(results[:i] + results[i + 1 :])["eod_return_net"]
            for i in range(len(rows))
        )
    checks, warnings = {}, []
    if results:
        means = summary["metrics"]
        delta = means["challenger_minus_control"]["mean"]
        eligible = [
            c
            for state, c in contexts.items()
            if state != "unknown"
            and c["paired_dates"] >= CONFIG["min_dates_per_context"]
        ]
        checks = {
            "review_sample": len(rows) >= CONFIG["min_paired_dates"],
            "changed_dates": summary["changed_dates"] >= CONFIG["min_changed_dates"],
            "joint_down_up_safe_net": joint(delta),
            "positive_challenger_net": means["challenger"]["mean"]["eod_return_net"]
            > 0,
            "beats_matched_net": means["challenger_minus_matched_random_expectation"][
                "mean"
            ]["eod_return_net"]
            > 0,
            "chronological_consistency": bool(segments)
            and all(
                joint(s["difference"], strict_down=False) for s in segments.values()
            ),
            "leave_one_date_out_net": leave_one_out is not None and leave_one_out > 0,
            "context_coverage": len(eligible) >= CONFIG["min_contexts"],
            "context_consistency": bool(eligible)
            and all(joint(c["difference"], strict_down=False) for c in eligible),
            "execution_coverage": execution["paired_dates"] == len(rows),
            "delayed_and_extra_cost": False,
        }
        if execution["delays"]:
            extra = CONFIG["extra_challenger_cost_fraction"]
            checks["delayed_and_extra_cost"] = all(
                joint(d["challenger_minus_control"]["mean"], strict_down=False)
                and d["challenger_minus_control"]["mean"]["eod_return_net"] > extra
                and d["challenger"]["mean"]["eod_return_net"] > extra
                for d in execution["delays"].values()
            )
        ci = means["challenger_minus_control"]["observed_date_block3_ci95"]
        if ci is None or ci["eod_return_net"][0] <= 0 <= ci["eod_return_net"][1]:
            warnings.append("net_difference_uncertain_not_a_standalone_rejection")
    if not integrity_ok:
        verdict, reason = "blocked", "evidence_integrity_or_operational_gap"
    elif not rows:
        verdict, reason = "waiting", "no_mature_pre_entry_records"
    elif not checks["review_sample"] or not checks["changed_dates"]:
        verdict, reason = "continue_observing", "initial_review_sample_not_reached"
    elif not checks["joint_down_up_safe_net"] or not checks["positive_challenger_net"]:
        verdict, reason = (
            "do_not_adopt",
            "joint_practical_goal_not_met_in_observed_sample",
        )
    elif all(checks.values()):
        verdict, reason = (
            "review_candidate",
            "practical_checks_pass_human_promotion_review_required",
        )
    else:
        verdict, reason = (
            "continue_observing",
            "robustness_or_coverage_not_yet_sufficient",
        )
    return {
        "verdict": verdict,
        "reason": reason,
        "checks": checks,
        "warnings": warnings,
        "summary": summary,
        "execution": execution,
        "contexts": contexts,
        "segments": segments,
        "leave_one_date_out_net_min": leave_one_out,
        "automatic_promotion": False,
        "deployable": False,
    }

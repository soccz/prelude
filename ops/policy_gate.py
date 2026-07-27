"""Policy promotion gate for the idea validation report.

This module turns replay/live evidence into an explicit operational state. It is
deliberately conservative: replay can justify paper-policy adoption, but high
confidence requires future shadow/live samples.
"""
from __future__ import annotations

from datetime import datetime, timezone


GATE_ID = "policy_gate_v1"
GATE_VERSION = "2026-05-25.1"

MIN_REPLAY_ACTIVE_TRADES = 30
MIN_LIVE_SHADOW_TRADES_FOR_HIGH_CONFIDENCE = 20


def _bootstrap_avg_low(stats: dict):
    boot = stats.get("bootstrap") or {}
    ci = boot.get("avg_pnl_ci95_pct") or []
    return ci[0] if len(ci) == 2 else None


def _late_avg(row: dict):
    late = next((p for p in row.get("replay_period_stability", []) if p.get("period") == "late"), None)
    promotion = {} if late is None else (late.get("promotion_evidence") or {})
    if promotion.get("eligible") is not True:
        return None
    return promotion.get("avg_net_pnl_pct")


def _shadow_closed_count(summary_rows: list[dict], channel: str) -> int:
    # Future shadow rows will appear as their own decision/idea groups. Until
    # then the gate marks confidence as replay-only.
    n = 0
    for row in summary_rows:
        if row.get("dimension") != "decision":
            continue
        if row.get("channel") != channel:
            continue
        if row.get("decision") in {"WATCH_ONLY", "SILENCE"}:
            n += int(row.get("n_promotion_eligible_closed") or 0)
    return n


def evaluate_policy_gate(report_payload: dict) -> dict:
    """Return explicit gate state per channel from idea_validation payload."""
    summary_rows = report_payload.get("tables", {}).get("summary", [])
    replay_rows = report_payload.get("policy_replay", {}).get("rows", [])
    out_rows = []

    for row in replay_rows:
        channel = row.get("channel")
        if channel == "all":
            continue
        observed = row.get("observed_active", {})
        replay = row.get("replay_active", {})
        promotion = replay.get("promotion_evidence") or {}
        promotion_eligible = promotion.get("eligible") is True
        delta = row.get("promotion_delta_net_pnl_sum_pct") or 0
        replay_n = int(promotion.get("n_closed") or 0)
        replay_avg = promotion.get("avg_net_pnl_pct")
        replay_net = promotion.get("net_pnl_sum_pct")
        ci_low = _bootstrap_avg_low(promotion)
        late_avg = _late_avg(row)
        shadow_n = _shadow_closed_count(summary_rows, channel)

        state = "COLLECT"
        action = "KEEP_COLLECTING"
        confidence = "LOW"
        reasons = []

        if channel == "preopen" and int(observed.get("n_closed") or 0) >= 50 and (observed.get("net_pnl_sum_pct") or 0) < 0:
            state = "DEMOTE"
            action = "WATCH_ONLY"
            confidence = "MEDIUM_REPLAY_ONLY"
            reasons.append("observed active preopen book is materially negative")
        elif (
            promotion_eligible
            and replay_n >= MIN_REPLAY_ACTIVE_TRADES
            and replay_avg is not None and replay_avg > 0
            and replay_net is not None and replay_net > 0
            and delta > 0
            and late_avg is not None and late_avg > 0
            and ci_low is not None and ci_low > 0
        ):
            state = "PROMOTE_PAPER"
            action = "ADOPT_ACTIVE_FILTER_IN_PAPER_AND_ALERTS"
            confidence = "HIGH" if shadow_n >= MIN_LIVE_SHADOW_TRADES_FOR_HIGH_CONFIDENCE else "MEDIUM_REPLAY_ONLY"
            reasons.append("replay active filter is positive overall, positive in late split, and bootstrap avg CI is above zero")
            if confidence != "HIGH":
                reasons.append("needs future shadow/live closed samples for high confidence")
        else:
            reasons.append("promotion gate not met")

        out_rows.append({
            "channel": channel,
            "state": state,
            "action": action,
            "confidence": confidence,
            "policy_id": row.get("policy_id", "setup_quality_policy_v1"),
            "replay_n_closed": replay_n,
            "replay_net_pnl_sum_pct": replay_net,
            "replay_avg_net_pnl_pct": replay_avg,
            "replay_bootstrap_avg_ci95_low_pct": ci_low,
            "replay_late_avg_net_pnl_pct": late_avg,
            "observed_net_pnl_sum_pct": observed.get("net_pnl_sum_pct"),
            "delta_net_pnl_sum_pct": row.get(
                "promotion_delta_net_pnl_sum_pct"
            ),
            "diagnostic_replay_n_closed": replay.get("n_closed"),
            "diagnostic_replay_net_pnl_sum_pct": replay.get(
                "net_pnl_sum_pct"
            ),
            "shadow_closed_for_confidence": shadow_n,
            "reasons": reasons,
        })

    return {
        "gate_id": GATE_ID,
        "gate_version": GATE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": out_rows,
    }

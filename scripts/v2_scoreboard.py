#!/usr/bin/env python3
"""pump_hunter_v2 시한부 판정(2026-09-01) 일일 자동 채점.

사전등록 블록(PHASES.md, 커밋 1d46a9c) 기준을 그대로 기계화:
  - 승격 기준: closed n>=200 AND mean net>0 AND 95% CI 0 제외 AND (2레짐 관측 OR per-day t>=2)
  - 조기 KILL: closed 누적 mean net < 0 전환 시 즉시
  - 2026-09-01: 4개 기준 전부 충족이면 GO, 아니면 표본 부족도 포함해 KILL
운영 판정은 2026-07-27 이후 canonical provenance가 검증된 CLOSED만 사용한다.
과거 legacy는 별도 진단으로만 남고 위 GO/KILL 표본에 합산하지 않는다.
비용: ``realized_pct`` 는 close_recommend_ledger 에서 이미 왕복 0.15%를
차감한 net 값이다. 여기서 다시 빼지 않는다.

출력: 한 줄 요약 + 일일 scoreboard + 별도 불변 terminal verdict.
종료코드: 0=정상/GO / 21=terminal KILL / 2=입력·provenance 오류.
cron 아님 — heartbeat.sh가 매일 호출.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from contextlib import contextmanager
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Iterator
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ledger.config import ROUND_TRIP_COST_PP  # noqa: E402
from ledger.csv_store import ledger_lock  # noqa: E402
from ops.artifact_provenance import atomic_write_json  # noqa: E402
from ops.close_input_gate import (  # noqa: E402
    CLOSE_EVIDENCE_ACTIVATION_DATE,
)
from ops.file_lock import file_lock  # noqa: E402
from ops.v2_provenance import (  # noqa: E402
    DEFAULT_DECISION_ROOT,
    DEFAULT_LEDGER_PATH,
    DEFAULT_RECEIPT_ROOT,
    V2ProvenanceError,
    validate_v2_provenance,
)
from ops.radar_verdict import (  # noqa: E402
    JUDGMENT_DAY,
    RADAR_TERMINAL_STATE,
    RadarVerdictError,
    load_terminal_verdict,
    record_terminal_verdict,
    recover_terminal_verdict,
    terminal_candidate,
)

LEDGER = DEFAULT_LEDGER_PATH
OUT = ROOT / "output" / "v2_scoreboard.json"
DECISION_ROOT = DEFAULT_DECISION_ROOT
RECEIPT_ROOT = DEFAULT_RECEIPT_ROOT
TERMINAL_STATE = RADAR_TERMINAL_STATE
VALID_REGIMES = {
    "bull_quiet",
    "bull_volatile",
    "bear_quiet",
    "bear_volatile",
}
VALID_STATUSES = {"open", "no_data", "not_delivered", "closed"}
KST = ZoneInfo("Asia/Seoul")


def _mean_ci95(values: list[float]) -> tuple[float, tuple[float, float]]:
    mean = statistics.mean(values)
    se = statistics.stdev(values) / math.sqrt(len(values))
    return mean, (mean - 1.96 * se, mean + 1.96 * se)


def build_scoreboard(rows: list[dict], *, today: date | None = None) -> dict:
    """Build the frozen v2 scorecard plus a new day-equal companion metric.

    The preregistered decision remains per trade.  ``day_equal_*`` is reported
    alongside it and must not be substituted retroactively for the 2026-09-01
    decision rule.
    """
    asof = today or datetime.now(KST).date()
    closed: list[dict] = []
    seen_positions: set[tuple[date, str]] = set()
    for row_number, row in enumerate(rows, 2):
        if not isinstance(row, dict):
            raise ValueError(f"ledger row {row_number} is not an object")
        status = str(row.get("status", ""))
        if status not in VALID_STATUSES:
            raise ValueError(f"invalid ledger status: {status!r}")
        try:
            decision_date = date.fromisoformat(str(row.get("date", "")))
        except ValueError as exc:
            raise ValueError(
                f"invalid ledger date: {row.get('date')!r}"
            ) from exc
        coin = str(row.get("coin", "")).strip()
        if not coin:
            raise ValueError("ledger row is missing coin")
        key = (decision_date, coin)
        if key in seen_positions:
            duplicate_kind = "closed" if status == "closed" else "ledger"
            raise ValueError(
                f"duplicate {duplicate_kind} position: {decision_date}/{coin}"
            )
        seen_positions.add(key)

        realized = row.get("realized_pct")
        if status != "closed":
            if realized not in (None, ""):
                raise ValueError(
                    f"{status} row unexpectedly has realized_pct={realized!r}"
                )
            if decision_date > asof:
                raise ValueError(
                    f"{status} date must not be after scoreboard asof: "
                    f"{decision_date} > {asof}"
                )
            continue
        if realized in (None, ""):
            raise ValueError("closed row is missing realized_pct")
        if decision_date >= asof:
            raise ValueError(
                f"closed date must be before scoreboard asof: "
                f"{decision_date} >= {asof}"
            )
        closed.append(row)

    parsed_closed: list[tuple[dict, float]] = []
    for row in closed:
        try:
            value = float(row["realized_pct"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid closed realized_pct: {row.get('realized_pct')!r}"
            ) from exc
        if not math.isfinite(value):
            raise ValueError(
                f"non-finite closed realized_pct: {row.get('realized_pct')!r}"
            )
        parsed_closed.append((row, value))

    # A delayed first run after the deadline must reproduce the exact
    # 2026-09-01 cutoff, not admit Sep-01+ cohorts that became CLOSED later.
    post_judgment_closed = 0
    if asof > JUDGMENT_DAY:
        frozen_closed: list[tuple[dict, float]] = []
        for row, value in parsed_closed:
            decision_date = date.fromisoformat(str(row["date"]))
            if decision_date < JUDGMENT_DAY:
                frozen_closed.append((row, value))
            else:
                post_judgment_closed += 1
        parsed_closed = frozen_closed

    # realized_pct is already net of ROUND_TRIP_COST_PP in the closer.
    closed = [row for row, _value in parsed_closed]
    nets = [value for _row, value in parsed_closed]
    n = len(nets)
    base = {
        "asof": asof.isoformat(),
        "closed_n": n,
        "cost_already_deducted": True,
        "round_trip_cost_pp": ROUND_TRIP_COST_PP,
        "preregistered_metric": "per_trade_mean_net_pct",
        "companion_metric_note": (
            "day_equal fields are diagnostic only; do not replace the frozen "
            "2026-09-01 per-trade decision metric retroactively"
        ),
        "terminal_evidence_cutoff_exclusive": JUDGMENT_DAY.isoformat(),
        "post_judgment_closed_excluded": post_judgment_closed,
    }
    if n == 0:
        criteria = {
            "n>=200": False,
            "mean>0": False,
            "CI_0_제외": False,
            "2레짐_or_t>=2": False,
        }
        deadline_reached = asof >= JUDGMENT_DAY
        return {
            **base,
            "status": (
                "judgment_kill"
                if deadline_reached
                else "insufficient_sample"
            ),
            "mean_net_pct": None,
            "ci95": None,
            "per_day_t": None,
            "trade_days": 0,
            "day_equal_mean_net_pct": None,
            "day_cluster_ci95": None,
            "regimes": [],
            "criteria": criteria,
            "criteria_met": 0,
            "early_kill_breached": False,
            "days_to_judgment": (JUDGMENT_DAY - asof).days,
            "terminal_verdict": "kill" if deadline_reached else None,
            "terminal_reason": (
                "one_or_more_frozen_criteria_not_met_at_deadline"
                if deadline_reached
                else None
            ),
            "terminal_metric_values": {
                "mean_net_pct": None,
                "ci95": None,
                "per_day_t": None,
            },
        }

    mean = statistics.mean(nets)
    ci = _mean_ci95(nets)[1] if n >= 2 else None

    # Per-day companion metric: equal weight per signal date.
    by_day = defaultdict(list)
    for r, v in zip(closed, nets):
        by_day[r["date"]].append(v)
    daily = [statistics.mean(v) for v in by_day.values()]
    k = len(daily)
    if k >= 2:
        day_mean, day_ci = _mean_ci95(daily)
        day_sd = statistics.stdev(daily)
        t_daily = day_mean / (day_sd / math.sqrt(k)) if day_sd > 0 else 0.0
    else:
        day_mean = daily[0] if daily else 0.0
        day_ci = None
        t_daily = None

    regimes = sorted(
        {
            str(r.get("btc_regime"))
            for r in closed
            if str(r.get("btc_regime")) in VALID_REGIMES
        }
    )

    crit = {
        "n>=200": n >= 200,
        "mean>0": mean > 0,
        "CI_0_제외": ci is not None and ci[0] > 0,
        "2레짐_or_t>=2": (
            len(regimes) >= 2
            or (t_daily is not None and t_daily >= 2)
        ),
    }
    early_kill = mean < 0
    days_left = (JUDGMENT_DAY - asof).days
    deadline_reached = asof >= JUDGMENT_DAY
    all_met = all(crit.values())
    if deadline_reached:
        status = "judgment_go" if all_met else "judgment_kill"
        terminal_verdict = "go" if all_met else "kill"
        terminal_reason = (
            "all_frozen_criteria_met_at_deadline"
            if all_met
            else "one_or_more_frozen_criteria_not_met_at_deadline"
        )
    elif early_kill:
        status = "early_kill"
        terminal_verdict = "kill"
        terminal_reason = "cumulative_mean_net_below_zero_before_deadline"
    else:
        status = "insufficient_sample" if n < 2 else "active"
        terminal_verdict = None
        terminal_reason = None

    return {
        **base,
        "status": status,
        "closed_n": n, "mean_net_pct": round(mean, 4),
        "ci95": (
            [round(ci[0], 4), round(ci[1], 4)]
            if ci is not None
            else None
        ),
        "per_day_t": round(t_daily, 3) if t_daily is not None else None,
        "trade_days": k,
        "day_equal_mean_net_pct": round(day_mean, 4),
        "day_cluster_ci95": (
            [round(day_ci[0], 4), round(day_ci[1], 4)]
            if day_ci is not None
            else None
        ),
        "regimes": regimes, "criteria": crit,
        "criteria_met": sum(crit.values()),
        "early_kill_breached": early_kill,
        "days_to_judgment": days_left,
        "terminal_verdict": terminal_verdict,
        "terminal_reason": terminal_reason,
        "terminal_metric_values": {
            "mean_net_pct": mean,
            "ci95": list(ci) if ci is not None else None,
            "per_day_t": t_daily,
        },
    }


def _legacy_diagnostic(rows: list[dict], *, today: date) -> dict:
    """Summarize pre-contract rows without exposing a decision verdict."""
    scorecard = build_scoreboard(rows, today=today)
    return {
        "scope": "pre_contract_legacy_rows_only",
        "operational": False,
        "terminal_eligible": False,
        "contract_cutoff_exclusive": (
            CLOSE_EVIDENCE_ACTIVATION_DATE.isoformat()
        ),
        "closed_n": scorecard["closed_n"],
        "mean_net_pct": scorecard["mean_net_pct"],
        "ci95": scorecard["ci95"],
        "trade_days": scorecard["trade_days"],
        "day_equal_mean_net_pct": scorecard["day_equal_mean_net_pct"],
        "day_cluster_ci95": scorecard["day_cluster_ci95"],
        "per_day_t": scorecard["per_day_t"],
        "regimes": scorecard["regimes"],
    }


@contextmanager
def _exclusive_output_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(path.with_name(f".{path.name}.lock")):
        yield


def _atomic_write(path: Path, payload: dict) -> None:
    atomic_write_json(path, payload)


def _error_payload(status: str, message: str, *, today: date | None) -> dict:
    payload = build_scoreboard([], today=today)
    payload.update(
        {
            "status": status,
            "error": message,
            "terminal_evidence_eligible": False,
            "terminal_verdict": None,
            "terminal_reason": "verified_post_contract_evidence_unavailable",
        }
    )
    return payload


def _apply_terminal_state(
    payload: dict,
    *,
    asof: date,
    terminal_state: Path,
    verdict_recorded_at: datetime | None,
) -> dict | None:
    """Resolve once and project the immutable verdict onto today's report."""
    if payload.get("asof") != asof.isoformat():
        raise RadarVerdictError(
            "scorecard/terminal asof identity mismatch"
        )
    scorecard_status = str(payload.get("status", "unknown"))
    state = load_terminal_verdict(terminal_state)
    if (
        state is None
        and payload.get("terminal_evidence_eligible") is not False
    ):
        candidate = terminal_candidate(
            payload,
            asof=asof,
            recorded_at=verdict_recorded_at,
        )
        if candidate is not None:
            state = record_terminal_verdict(
                candidate,
                path=terminal_state,
            )

    payload["scorecard_status"] = scorecard_status
    if state is None:
        payload["terminal_state"] = {
            "status": "pending",
            "verdict": None,
            "judgment_day": JUDGMENT_DAY.isoformat(),
            "effective": False,
        }
        return None

    effective_asof = date.fromisoformat(state["effective_asof"])
    if asof >= JUDGMENT_DAY and effective_asof > asof:
        raise RadarVerdictError(
            "terminal verdict is future-dated relative to deadline scorecard"
        )
    effective = effective_asof <= asof
    payload["terminal_state"] = {
        "status": state["status"],
        "verdict": state["verdict"],
        "verdict_id": state["verdict_id"],
        "effective_asof": state["effective_asof"],
        "judgment_day": state["judgment_day"],
        "reason": state["reason"],
        "effective": effective,
    }
    if effective:
        payload["status"] = state["status"]
        payload["terminal_verdict"] = state["verdict"]
        payload["terminal_reason"] = state["reason"]
    return state if effective else None


def _finalize_payload(
    payload: dict,
    *,
    output: Path,
    asof: date,
    terminal_state: Path,
    base_code: int,
    verdict_recorded_at: datetime | None,
) -> tuple[int, dict]:
    try:
        effective_state = _apply_terminal_state(
            payload,
            asof=asof,
            terminal_state=terminal_state,
            verdict_recorded_at=verdict_recorded_at,
        )
    except RadarVerdictError as exc:
        terminal_error = f"{type(exc).__name__}: {exc}"
        forced_kill = None
        if (
            asof >= JUDGMENT_DAY
            and payload.get("terminal_evidence_eligible") is not False
        ):
            recovery_scorecard = _error_payload(
                "terminal_state_invalid",
                terminal_error,
                today=asof,
            )
            forced_kill = terminal_candidate(
                recovery_scorecard,
                asof=asof,
                recorded_at=verdict_recorded_at,
            )
        try:
            recover_terminal_verdict(
                path=terminal_state,
                forced_kill=forced_kill,
            )
            effective_state = _apply_terminal_state(
                payload,
                asof=asof,
                terminal_state=terminal_state,
                verdict_recorded_at=verdict_recorded_at,
            )
        except RadarVerdictError as recovery_exc:
            if "error" in payload:
                payload["scorecard_error"] = payload["error"]
            recovery_error = (
                f"{type(recovery_exc).__name__}: {recovery_exc}"
            )
            payload["scorecard_status"] = str(
                payload.get("status", "unknown")
            )
            payload["status"] = "terminal_state_invalid"
            payload["error"] = recovery_error
            payload["terminal_state"] = {
                "status": "invalid",
                "verdict": None,
                "effective": False,
                "error": recovery_error,
                "initial_error": terminal_error,
            }
            _atomic_write(output, payload)
            return 2, payload

        payload["terminal_recovery"] = {
            "recovered": True,
            "initial_error": terminal_error,
        }
        _atomic_write(output, payload)
        if effective_state is not None and effective_state["verdict"] == "kill":
            return 21, payload
        return base_code, payload

    _atomic_write(output, payload)
    if effective_state is not None and effective_state["verdict"] == "kill":
        return 21, payload
    return base_code, payload


def run_scoreboard(
    *,
    ledger: Path = LEDGER,
    output: Path = OUT,
    today: date | None = None,
    decision_root: Path = DECISION_ROOT,
    receipt_root: Path = RECEIPT_ROOT,
    terminal_state: Path | None = None,
    verdict_recorded_at: datetime | None = None,
) -> tuple[int, dict]:
    """Read, score and atomically replace the canonical scoreboard artifact."""
    terminal_path = TERMINAL_STATE if terminal_state is None else terminal_state
    with _exclusive_output_lock(output):
        # Capture exactly once after lock acquisition.  A process queued over
        # KST midnight must use the new day consistently for build + terminal.
        asof = today or datetime.now(KST).date()
        if not ledger.exists():
            payload = _error_payload(
                "ledger_missing",
                f"ledger missing: {ledger}",
                today=asof,
            )
            return _finalize_payload(
                payload,
                output=output,
                asof=asof,
                terminal_state=terminal_path,
                base_code=2,
                verdict_recorded_at=verdict_recorded_at,
            )
        try:
            with ledger_lock(ledger), ledger.open(
                newline="",
                encoding="utf-8",
            ) as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    raise ValueError("ledger header missing")
                required = {"date", "coin", "status", "realized_pct"}
                missing = sorted(required - set(reader.fieldnames))
                if missing:
                    raise ValueError(
                        f"ledger columns missing: {','.join(missing)}"
                    )
                if len(set(reader.fieldnames)) != len(reader.fieldnames):
                    raise ValueError("ledger contains duplicate column names")
                rows = []
                for line_number, row in enumerate(reader, 2):
                    if None in row:
                        raise ValueError(
                            f"ledger row {line_number} has extra fields"
                        )
                    rows.append(row)
                # Validate the complete ledger before provenance filtering.
                # This result is deliberately discarded: legacy rows must
                # never leak into the operational or terminal scorecard.
                build_scoreboard(rows, today=asof)
        except (OSError, UnicodeError, csv.Error, ValueError) as exc:
            payload = _error_payload(
                "ledger_invalid",
                f"{type(exc).__name__}: {exc}",
                today=asof,
            )
            return _finalize_payload(
                payload,
                output=output,
                asof=asof,
                terminal_state=terminal_path,
                base_code=2,
                verdict_recorded_at=verdict_recorded_at,
            )
        try:
            provenance = validate_v2_provenance(
                rows,
                asof=asof,
                decision_root=decision_root,
                receipt_root=receipt_root,
                enforce_legacy_baseline=(
                    ledger.resolve() == LEDGER.resolve()
                ),
            )
        except V2ProvenanceError as exc:
            payload = _error_payload(
                "provenance_invalid",
                f"{type(exc).__name__}: {exc}",
                today=asof,
            )
            return _finalize_payload(
                payload,
                output=output,
                asof=asof,
                terminal_state=terminal_path,
                base_code=2,
                verdict_recorded_at=verdict_recorded_at,
            )
        verified_keys = set(provenance.verified_closed_positions)
        verified_rows = [
            row
            for row in rows
            if (
                date.fromisoformat(str(row["date"])),
                str(row["coin"]).strip(),
            )
            in verified_keys
        ]
        if len(verified_rows) != len(verified_keys):
            payload = _error_payload(
                "provenance_invalid",
                "verified v2 position projection is incomplete",
                today=asof,
            )
            return _finalize_payload(
                payload,
                output=output,
                asof=asof,
                terminal_state=terminal_path,
                base_code=2,
                verdict_recorded_at=verdict_recorded_at,
            )
        legacy_rows = [
            row
            for row in rows
            if date.fromisoformat(str(row["date"]))
            < CLOSE_EVIDENCE_ACTIVATION_DATE
        ]
        payload = build_scoreboard(verified_rows, today=asof)
        payload["evidence_scope"] = (
            "verified_post_contract_closed_positions_only"
        )
        payload["terminal_evidence_eligible"] = (
            bool(verified_rows)
            or asof >= JUDGMENT_DAY
        )
        if not verified_rows and asof < JUDGMENT_DAY:
            # Pre-deadline legacy history cannot resolve the forward verdict.
            payload.update(
                {
                    "status": "insufficient_verified_evidence",
                    "terminal_verdict": None,
                    "terminal_reason": (
                        "no_verified_post_contract_closed_positions"
                    ),
                }
            )
        payload["legacy_diagnostic"] = _legacy_diagnostic(
            legacy_rows,
            today=asof,
        )
        payload["provenance"] = {
            "healthy_dates": [
                value.isoformat() for value in provenance.healthy_dates
            ],
            "healthy_zero_pick_dates": [
                value.isoformat()
                for value in provenance.healthy_zero_pick_dates
            ],
            "healthy_candidate_dates": [
                value.isoformat()
                for value in provenance.healthy_candidate_dates
            ],
            "delivery_success_dates": [
                value.isoformat()
                for value in provenance.delivery_success_dates
            ],
            "delivery_failed_dates": [
                value.isoformat()
                for value in provenance.delivery_failed_dates
            ],
            "verified_closed_positions": len(
                provenance.verified_closed_positions
            ),
            "legacy_scorecard_rows": provenance.legacy_scorecard_rows,
            "legacy_scorecard_sha256": (
                provenance.legacy_scorecard_sha256
            ),
            "legacy_baseline_verified": (
                provenance.legacy_baseline_verified
            ),
        }
        code, payload = _finalize_payload(
            payload,
            output=output,
            asof=asof,
            terminal_state=terminal_path,
            base_code=0,
            verdict_recorded_at=verdict_recorded_at,
        )

    if code == 21:
        mean = payload["mean_net_pct"]
        mean_text = f"{mean:+.3f}%" if mean is not None else "n/a"
        status = payload["status"]
        print(
            f"[v2-scoreboard] n={payload['closed_n']} "
            f"mean_net={mean_text} ⛔ {status} — radar 전체 KILL"
        )
        return 21, payload
    if code == 2:
        return code, payload
    if payload["status"] == "judgment_go":
        print(
            f"[v2-scoreboard] n={payload['closed_n']} "
            "✅ judgment_go — frozen criteria 4/4"
        )
        return 0, payload
    if payload["closed_n"] < 2:
        print(
            f"[v2-scoreboard] closed n={payload['closed_n']} — "
            "표본 부족, 채점 보류"
        )
        return 0, payload

    mean = payload["mean_net_pct"]
    ci = payload["ci95"]
    t_daily = payload["per_day_t"]
    days_left = payload["days_to_judgment"]
    n = payload["closed_n"]
    criteria_met = payload["criteria_met"]
    day_mean = payload["day_equal_mean_net_pct"]
    t_daily_text = (
        f"{t_daily:.2f}" if t_daily is not None else "n/a"
    )
    line = (f"[v2-scoreboard] D-{days_left} n={n} mean_net={mean:+.3f}% "
            f"CI[{ci[0]:+.3f},{ci[1]:+.3f}] day_eq={day_mean:+.3f}% "
            f"t_day={t_daily_text} 기준충족 {criteria_met}/4")
    print(line)
    return 0, payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()
    # Canonical CLI is intentionally non-configurable.  Tests and offline
    # diagnostics inject paths through run_scoreboard(), never production CLI.
    code, payload = run_scoreboard()
    if code == 2:
        print(f"[v2-scoreboard] {payload['error']}")
    return code


if __name__ == "__main__":
    sys.exit(main())

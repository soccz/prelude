"""Read-only, paired entry-delay diagnostics for actually delivered R1 picks.

Every delay receives a fresh 96-bar horizon. This measures sensitivity to a
later assumed entry; it does not select an optimal entry or certify alert TTL.
Canonical snapshots, receipts, labels, ledgers and databases are never written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import sys
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.database import connect_readonly  # noqa: E402
from ledger.config import ROUND_TRIP_COST_PCT  # noqa: E402
from ledger.path_quality import assess_15m_window  # noqa: E402
from ops.artifact_provenance import file_set_identity  # noqa: E402
from ops.recommendation_evidence import (  # noqa: E402
    EvidenceError,
    EvidenceUnavailable,
    discover_r1_snapshots,
    load_recommendation_evidence,
)
from signals.recommend_score_labels import _label_candidate  # noqa: E402

REPORT_SCHEMA = "recommend_entry_delay.v1"
DEFAULT_DELAYS = (0, 15, 30)  # Initial diagnostic grid, not an optimized policy.
PARITY_FIELDS = (
    "actual_entry_open", "mfe", "mae", "eod_return_gross", "eod_return_net",
    "up5", "up10", "up20", "dn3", "dn5", "dn10", "tp5_before_sl3",
    "tp5_sl3_first_passage", "tp5_sl3_return_gross", "tp5_sl3_return_net",
    "first_passage_bar", "first_passage_at",
)
METRICS = (
    "whole_path_safe_up10", "up10", "dn5", "tp5_first", "sl3_first",
    "neither", "tp5_sl3_return_net", "eod_return_net", "mfe", "mae",
)
SOURCE_FILES = (
    "scripts/evaluate_recommend_entry_delay.py", "ops/recommendation_evidence.py",
    "signals/recommend_snapshot.py", "signals/recommend_score_labels.py",
    "notifier/delivery_receipt.py", "notifier/telegram.py",
    "ops/artifact_provenance.py", "ops/file_lock.py",
    "ledger/path_quality.py", "ledger/config.py", "data/database.py",
)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _aware(value: datetime | str | pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("now/execution timestamps must be timezone-aware")
    return stamp.tz_convert("Asia/Seoul")


def _validate_options(delays: tuple[int, ...], n_boot: int,
                      min_days: int, seed: int) -> None:
    if (not delays or delays[0] != 0 or len(delays) > 17
            or any(type(x) is not int or x < 0 or x > 240 or x % 15 for x in delays)
            or tuple(sorted(set(delays))) != delays):
        raise ValueError("delays must start at 0 and increase in 15-minute steps, <=240")
    if type(n_boot) is not int or not 100 <= n_boot <= 100_000:
        raise ValueError("n_boot must be an integer in [100, 100000]")
    if type(min_days) is not int or min_days < 2:
        raise ValueError("min_days must be an integer >=2")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")


def _path_manifest(conn: sqlite3.Connection, market: str,
                   start: pd.Timestamp) -> dict:
    """Hash the same read transaction used by the path assessment.

    Include every target/BTC OHLC row, each prior close, the next BTC boundary
    witness and the BTC horizon used by the completeness check. DB-wide byte
    hashing is deliberately unnecessary for a small window diagnostic.
    """
    start = start.tz_convert("Asia/Seoul").tz_localize(None)
    end = start + pd.Timedelta(days=1)
    start_sql, end_sql = (x.strftime("%Y-%m-%d %H:%M:%S") for x in (start, end))
    markets = sorted({market, "KRW-BTC"})
    rows: list[tuple] = []
    prior: list[tuple] = []
    for current in markets:
        rows.extend(conn.execute(
            "SELECT market,timestamp,open,high,low,close FROM candles "
            "WHERE market=? AND timestamp>=? AND timestamp<? ORDER BY timestamp",
            (current, start_sql, end_sql),
        ).fetchall())
        previous = conn.execute(
            "SELECT market,timestamp,open,high,low,close FROM candles "
            "WHERE market=? AND timestamp<? ORDER BY timestamp DESC LIMIT 1",
            (current, start_sql),
        ).fetchone()
        if previous is not None:
            prior.append(previous)
    boundary = conn.execute(
        "SELECT market,timestamp,open,high,low,close FROM candles "
        "WHERE market='KRW-BTC' AND timestamp>=? ORDER BY timestamp LIMIT 1",
        (end_sql,),
    ).fetchone()
    horizon = conn.execute(
        "SELECT MIN(timestamp),MAX(timestamp) FROM candles WHERE market='KRW-BTC'"
    ).fetchone()
    raw = {"rows": rows, "prior": prior, "benchmark_next": boundary,
           "benchmark_horizon": horizon}
    return {
        "market": market, "start_at": start.isoformat(), "end_at": end.isoformat(),
        "window_rows": len(rows), "prior_rows": len(prior),
        "benchmark_next_timestamp": boundary[1] if boundary else None,
        "benchmark_horizon": list(horizon),
        "raw_input_sha256": hashlib.sha256(_json_bytes(raw)).hexdigest(),
        "hash_contract": "ordered_OHLC_window+prior+BTC_next_boundary+BTC_horizon.v1",
    }


def _parity_differences(actual: dict, canonical: dict) -> list[dict]:
    differences = []
    for field in PARITY_FIELDS:
        left, right = actual.get(field), canonical.get(field)
        if field == "first_passage_at" and left is not None and right is not None:
            # Canonical values use exchange-local naive bar timestamps.
            same = pd.Timestamp(left) == pd.Timestamp(right)
        elif isinstance(left, bool) or isinstance(right, bool):
            same = type(left) is type(right) and left == right
        elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
            same = math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12)
        else:
            same = left == right
        if not same:
            differences.append({"field": field, "canonical": right, "recomputed": left})
    return differences


def _metrics(outcome: dict) -> dict[str, float]:
    passage = outcome["tp5_sl3_first_passage"]
    return {
        "whole_path_safe_up10": float(outcome["up10"] and not outcome["dn5"]),
        "up10": float(outcome["up10"]), "dn5": float(outcome["dn5"]),
        "tp5_first": float(passage == "tp_first"),
        "sl3_first": float(passage in {"sl_first", "sl_first_same_bar"}),
        "neither": float(passage == "neither"),
        **{key: float(outcome[key]) for key in (
            "tp5_sl3_return_net", "eod_return_net", "mfe", "mae")},
    }


def _summarize(records: list[dict], delays: tuple[int, ...], *,
               n_boot: int, min_days: int, seed: int) -> dict:
    """Average picks within a date, then bootstrap paired date-level deltas."""
    if not records:
        return {"status": "insufficient", "n_days": 0, "n_picks": 0, "delays": {}}
    daily: dict[str, list[dict]] = {}
    for row in records:
        daily.setdefault(row["asof"], []).append(row)
    days = sorted(daily)
    # Chunk the resamples so a large requested bootstrap never allocates B x N.
    result = {"status": "ok" if len(days) >= min_days else "insufficient",
              "n_days": len(days), "n_picks": len(records), "dates": days,
              "cluster": "date", "weighting": "mean_within_date_then_equal_dates",
              "delays": {}}
    baseline = {
        metric: np.array([np.mean([r["delays"]["0"]["metrics"][metric]
                                  for r in daily[day]]) for day in days])
        for metric in METRICS
    }
    for delay in delays:
        metrics = {}
        for metric in METRICS:
            values = np.array([
                np.mean([r["delays"][str(delay)]["metrics"][metric] for r in daily[day]])
                for day in days
            ])
            delta = values - baseline[metric]
            ci = None
            if len(days) >= min_days:
                # Reset per metric: the same resampled dates are used for all
                # outcomes and delays, retaining their paired interpretation.
                rng = np.random.default_rng(seed)
                samples = []
                for offset in range(0, n_boot, 1000):
                    indices = rng.integers(0, len(days), size=(min(1000, n_boot-offset), len(days)))
                    samples.extend(delta[indices].mean(axis=1).tolist())
                ci = [float(x) for x in np.quantile(samples, (0.025, 0.975))]
            metrics[metric] = {
                "day_equal_mean": float(values.mean()),
                "delta_vs_zero": float(delta.mean()), "delta_ci95": ci,
                "ci_reason": None if ci is not None else f"n_days<{min_days}",
            }
        result["delays"][str(delay)] = {"metrics": metrics}
    return result


def evaluate_entry_delay(
    *, snapshot_root: str | Path, label_root: str | Path, receipt_root: str | Path,
    db_path: str | Path, start_date: date, end_date: date,
    now: datetime | None = None, slots: tuple[str, ...] = ("open", "preopen"),
    delays: tuple[int, ...] = DEFAULT_DELAYS, n_boot: int = 1000,
    min_days: int = 5, seed: int = 42,
) -> dict:
    _validate_options(delays, n_boot, min_days, seed)
    now_kst = _aware(now or datetime.now(timezone.utc))
    if start_date > end_date:
        raise ValueError("start_date must not exceed end_date")
    if not slots or len(set(slots)) != len(slots) or set(slots) - {"open", "preopen"}:
        raise ValueError("slots must be distinct open/preopen values")
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA, "status": "insufficient", "created_at": now_kst.isoformat(),
        "requested_period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
        "contract": {
            "ranking": "R1", "cohort": "actual_delivered_snapshot_top3",
            "delays_minutes": list(delays), "grid_status": "initial_diagnostic_not_optimized",
            "baseline": "canonical_execution_start_at", "horizon_bars": 96,
            "bar_minutes": 15, "horizon": "each_start_plus_24h",
            "round_trip_cost_fraction": ROUND_TRIP_COST_PCT, "return_unit": "fraction",
            "n_boot": n_boot, "seed": seed, "min_days_for_ci": min_days,
            "primary": "all_original_top3_complete_at_all_delays_per_slot_date",
            "secondary": "same_pick_complete_at_all_delays",
            "whole_path_safe_up10": "MFE>=10% AND MAE>-5% over full 24h",
            "tp5_first": "TP5 before SL3; not whole-path safety; same-bar is SL-first",
            "claims_excluded": ["optimal_entry_strategy", "5_minute_precision", "alert_TTL_certification"],
        },
        "inputs": [], "exclusions": [], "errors": [], "records": [], "channels": {},
        "database": {"path": str(Path(db_path)), "access": "read_only_single_transaction"},
        "code": file_set_identity({rel: _ROOT / rel for rel in SOURCE_FILES}, root=_ROOT),
    }
    try:
        sources = discover_r1_snapshots(Path(snapshot_root), start_date=start_date,
                                       end_date=end_date, slots=slots)
    except EvidenceUnavailable as exc:
        report["exclusions"].append({"snapshot_path": str(snapshot_root), "reason": str(exc)})
        return _finish(report)
    except EvidenceError as exc:
        report["errors"].append({"snapshot_path": str(snapshot_root), "reason": str(exc)})
        report["status"] = "blocked_evidence_integrity"
        return _finish(report)
    accepted = []
    for source in sources:
        try:
            evidence = load_recommendation_evidence(
                source, label_root=Path(label_root), receipt_root=Path(receipt_root),
                now=now_kst.to_pydatetime(),
            )
        except EvidenceUnavailable as exc:
            report["exclusions"].append({"snapshot_path": str(source), "reason": str(exc)})
            continue
        except EvidenceError as exc:
            report["errors"].append({"snapshot_path": str(source), "reason": str(exc)})
            continue
        report["inputs"].append(evidence["manifest"])
        accepted.append(evidence)
    if report["errors"]:
        report["status"] = "blocked_evidence_integrity"
        return _finish(report)
    if not accepted:
        return _finish(report)

    try:
        with connect_readonly(db_path) as conn:
            conn.execute("BEGIN")
            report["database"]["schema_version"] = conn.execute("PRAGMA schema_version").fetchone()[0]
            for evidence in accepted:
                snapshot, label = evidence["snapshot"], evidence["label"]
                identity = {"asof": snapshot["asof"], "slot": snapshot["slot"],
                            "snapshot_id": snapshot["snapshot_id"]}
                start = _aware(label["execution_start_at"])
                latest_end = start + pd.Timedelta(minutes=delays[-1], days=1)
                if now_kst < latest_end:
                    report["exclusions"].append({**identity, "reason": "latest_delay_not_mature",
                                                 "latest_window_end": latest_end.isoformat()})
                    continue
                top3 = snapshot["top3"]
                if len(top3) != 3:
                    report["exclusions"].append({**identity, "reason": "not_three_original_picks"})
                    continue
                indexed = {(r["coin"], r["rank"]): r for r in label["rows"]}
                day_records = []
                for candidate in top3:
                    canonical = indexed[(candidate["coin"], candidate["rank"])]
                    pick = {**identity, "coin": candidate["coin"], "rank": candidate["rank"],
                            "delays": {}, "common_eligible": True,
                            "canonical_parity": "not_checked", "primary_eligible": False}
                    if canonical["label_status"] != "labeled" or not canonical["path_complete"]:
                        pick["common_eligible"] = False
                        pick["excluded_reason"] = f"canonical_{canonical['label_status']}"
                        report["exclusions"].append({**identity, "coin": candidate["coin"],
                                                     "reason": pick["excluded_reason"]})
                        day_records.append(pick)
                        continue
                    for delay in delays:
                        delayed_start = start + pd.Timedelta(minutes=delay)
                        assessment = assess_15m_window(candidate["coin"], delayed_start, connection=conn)
                        raw_input = _path_manifest(conn, candidate["coin"], delayed_start)
                        execution = {"execution_start_at": delayed_start.isoformat()}
                        outcome = _label_candidate(
                            candidate, assessment, snapshot_id=snapshot["snapshot_id"],
                            snapshot_hash=snapshot["payload_sha256"], execution=execution,
                        )
                        item = {"start_at": delayed_start.isoformat(),
                                "end_at": (delayed_start + pd.Timedelta(days=1)).isoformat(),
                                "raw_input": raw_input, **assessment.metadata(),
                                "label_status": outcome["label_status"]}
                        if outcome["label_status"] == "labeled":
                            item["outcomes"] = {k: outcome[k] for k in PARITY_FIELDS}
                            item["metrics"] = _metrics(outcome)
                        else:
                            pick["common_eligible"] = False
                            report["exclusions"].append({**identity, "coin": candidate["coin"],
                                                         "delay_minutes": delay,
                                                         "reason": assessment.reason})
                        if delay == 0:
                            differences = (_parity_differences(outcome, canonical)
                                           if outcome["label_status"] == "labeled"
                                           else [{"field": "label_status", "canonical": "labeled",
                                                  "recomputed": outcome["label_status"]}])
                            pick["canonical_parity"] = "failed" if differences else "passed"
                            if differences:
                                report["errors"].append({**identity, "coin": candidate["coin"],
                                                         "reason": "canonical_zero_delay_mismatch",
                                                         "differences": differences})
                        pick["delays"][str(delay)] = item
                    day_records.append(pick)
                primary_ok = len(day_records) == 3 and all(r["common_eligible"] for r in day_records)
                for pick in day_records:
                    pick["primary_eligible"] = primary_ok
                report["records"].extend(day_records)
                if not primary_ok:
                    report["exclusions"].append({**identity, "reason": "primary_top3_incomplete",
                                                 "common_picks": sum(r["common_eligible"] for r in day_records)})
            conn.rollback()
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
        report["errors"].append({"reason": "path_evaluation_failed", "detail": str(exc)})
    if report["errors"]:
        report["status"] = "blocked_canonical_parity_or_path_integrity"
        return _finish(report)
    for slot in slots:
        records = [r for r in report["records"] if r["slot"] == slot]
        primary = [r for r in records if r["primary_eligible"]]
        secondary = [r for r in records if r["common_eligible"]]
        report["channels"][slot] = {
            "primary": _summarize(primary, delays, n_boot=n_boot, min_days=min_days, seed=seed),
            "secondary": _summarize(secondary, delays, n_boot=n_boot, min_days=min_days, seed=seed),
            "observed_pick_records": len(records),
            "primary_excluded_pick_records": len(records) - len(primary),
            "secondary_excluded_pick_records": len(records) - len(secondary),
        }
    report["status"] = ("ok" if any(c["primary"]["status"] == "ok"
                                    for c in report["channels"].values()) else "insufficient")
    return _finish(report)


def _finish(report: dict) -> dict:
    code_after = file_set_identity({rel: _ROOT / rel for rel in SOURCE_FILES}, root=_ROOT)
    if code_after != report["code"]:
        report["errors"].append({"reason": "evaluation_sources_changed_during_read"})
        report["status"] = "blocked_source_integrity"
        report["channels"] = {}
    report["exclusion_counts"] = dict(Counter(item["reason"] for item in report["exclusions"]))
    report["canonical_parity_counts"] = dict(Counter(
        item["canonical_parity"] for item in report["records"]
    ))
    report["report_payload_sha256"] = hashlib.sha256(_json_bytes(report)).hexdigest()
    return report


def write_report_exclusive(report: dict, output: Path, *, protected_roots: tuple[Path, ...]) -> None:
    target = output.resolve()
    canonical_roots = tuple(_ROOT / "output" / name for name in (
        "recommend_snapshots", "recommend_score_labels", "recommend_receipts",
    )) + (_ROOT / "data/upbit_15m.db",)
    if any(target.is_relative_to(root.resolve()) for root in (*canonical_roots, *protected_roots)):
        raise ValueError("report output must be outside canonical input roots")
    # Exclusive creation also rejects existing symlinks. Never mkdir or replace.
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path, default=_ROOT / "output/recommend_snapshots")
    parser.add_argument("--label-root", type=Path, default=_ROOT / "output/recommend_score_labels")
    parser.add_argument("--receipt-root", type=Path, default=_ROOT / "output/recommend_receipts")
    parser.add_argument("--db", type=Path, default=_ROOT / "data/upbit_15m.db")
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--now", type=datetime.fromisoformat,
                        help="aware ISO analysis clock for reproducible maturity checks")
    parser.add_argument("--slot", choices=("open", "preopen", "both"), default="both")
    parser.add_argument("--delays", default="0,15,30", help="initial diagnostic grid; integer minutes")
    parser.add_argument("--bootstraps", type=int, default=1000)
    parser.add_argument("--min-days", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="new exclusive report file; default stdout")
    args = parser.parse_args()
    try:
        report = evaluate_entry_delay(
            snapshot_root=args.snapshot_root, label_root=args.label_root, receipt_root=args.receipt_root,
            db_path=args.db, start_date=args.start, end_date=args.end,
            now=args.now,
            slots=("open", "preopen") if args.slot == "both" else (args.slot,),
            delays=tuple(int(x) for x in args.delays.split(",")),
            n_boot=args.bootstraps, min_days=args.min_days, seed=args.seed,
        )
        if args.output:
            write_report_exclusive(report, args.output, protected_roots=(
                args.snapshot_root, args.label_root, args.receipt_root, args.db,
            ))
            print(json.dumps({"status": report["status"], "output": str(args.output)}))
        else:
            print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return 2 if report["status"].startswith("blocked") else 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"entry-delay evaluation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

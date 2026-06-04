"""Policy competition - evaluate model + send-policy variants on forward CLOSED rows.

This is the second layer after ``champion_selector``:

* champion_selector answers: "which scorer has the best forward PnL/downside?"
* policy_competition answers: "under which send policy would this scorer have
  produced the best tradeable or pump-watchlist stream?"

It is deliberately record-only. It sends no Telegram messages, places no orders,
and changes no champion state. The output is a daily audit artifact for deciding
whether a policy should become LIVE / WATCH / SILENT later.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from signals.model_registry import MODELS, ModelSpec  # noqa: E402

OUT_CSV = _ROOT / "output" / "policy_competition_summary.csv"
OUT_JSON = _ROOT / "output" / "policy_competition_summary.json"
D1_DB = _ROOT / "data" / "upbit_d1.db"
POLICY_DB = _ROOT / "data" / "policy_competition.db"

ROUND_TRIP_COST_PCT = 0.15
PUMP20_THRESH = 0.20
DEEP_LOSS_PCT = -5.0

DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS policy_competition_runs (
    asof TEXT PRIMARY KEY,
    generated_at_utc TEXT NOT NULL,
    pump20_threshold REAL NOT NULL,
    deep_loss_pct REAL NOT NULL,
    round_trip_cost_pct REAL NOT NULL,
    row_count INTEGER NOT NULL,
    best_pump_participant TEXT,
    best_net_participant TEXT,
    updated_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS policy_competition_rows (
    asof TEXT NOT NULL,
    participant_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    policy_id TEXT NOT NULL,
    objective TEXT NOT NULL,
    description TEXT,
    n_closed INTEGER NOT NULL,
    n_days INTEGER NOT NULL,
    net_sum_pct REAL,
    net_mean_pct REAL,
    deep_loss_freq_pct REAL,
    sl_rate_pct REAL,
    tp_rate_pct REAL,
    pump20_precision_pct REAL,
    pump20_recall_pct REAL,
    pump20_captured INTEGER NOT NULL,
    pump20_actual INTEGER NOT NULL,
    pump_days INTEGER NOT NULL,
    pump_days_any_captured INTEGER NOT NULL,
    pump_day_capture_rate_pct REAL,
    generated_at_utc TEXT NOT NULL,
    PRIMARY KEY (asof, participant_id)
);

CREATE INDEX IF NOT EXISTS idx_policy_competition_rows_source
ON policy_competition_rows(source_id, policy_id);

CREATE VIEW IF NOT EXISTS policy_competition_latest_rows AS
SELECT *
FROM policy_competition_rows
WHERE asof = (SELECT MAX(asof) FROM policy_competition_runs);
"""


@dataclass(frozen=True)
class PolicySpec:
    policy_id: str
    description: str
    objective: str
    predicate: Callable[[pd.DataFrame], pd.Series]


def _true_series(df: pd.DataFrame) -> pd.Series:
    return pd.Series(True, index=df.index)


def _rank_le(n: int) -> Callable[[pd.DataFrame], pd.Series]:
    def pred(df: pd.DataFrame) -> pd.Series:
        rank = pd.to_numeric(df.get("rank"), errors="coerce")
        return rank.notna() & (rank <= n)

    return pred


def _no_dump(df: pd.DataFrame) -> pd.Series:
    if "dump_risk_flag" not in df.columns:
        return pd.Series(False, index=df.index)
    s = df["dump_risk_flag"].astype(str).str.lower()
    return ~s.isin(["true", "1", "yes"])


def _rr_min(threshold: float) -> Callable[[pd.DataFrame], pd.Series]:
    def pred(df: pd.DataFrame) -> pd.Series:
        if "rr_ratio" not in df.columns:
            return pd.Series(False, index=df.index)
        rr = pd.to_numeric(df["rr_ratio"], errors="coerce")
        return rr.notna() & (rr >= threshold)

    return pred


def _pump_prob_min(threshold: float) -> Callable[[pd.DataFrame], pd.Series]:
    def pred(df: pd.DataFrame) -> pd.Series:
        col = "p_up20" if "p_up20" in df.columns else "pump_prob"
        if col not in df.columns:
            return pd.Series(False, index=df.index)
        prob = pd.to_numeric(df[col], errors="coerce")
        return prob.notna() & (prob >= threshold)

    return pred


POLICIES: list[PolicySpec] = [
    PolicySpec(
        "top_all",
        "Use every recorded pick from that source ledger.",
        "baseline",
        _true_series,
    ),
    PolicySpec(
        "top1_only",
        "Send only rank #1.",
        "precision_pnl",
        _rank_le(1),
    ),
    PolicySpec(
        "top2_only",
        "Send only ranks #1-2.",
        "precision_pnl",
        _rank_le(2),
    ),
    PolicySpec(
        "no_dump_top3",
        "Send top-3 only when dump_risk_flag is false.",
        "downside_control",
        lambda df: _rank_le(3)(df) & _no_dump(df),
    ),
    PolicySpec(
        "rr_ge_0_75",
        "Send only picks with rr_ratio >= 0.75.",
        "downside_control",
        _rr_min(0.75),
    ),
    PolicySpec(
        "pump_prob_ge_3pct",
        "Pump watchlist: require calibrated P(+20%) >= 3%.",
        "pump_recall",
        _pump_prob_min(0.03),
    ),
]


def _rank_col(df: pd.DataFrame) -> str | None:
    for col in ("rank", "alert_rank", "source_rank"):
        if col in df.columns:
            return col
    return None


def _coin_col(df: pd.DataFrame) -> str | None:
    for col in ("coin", "market"):
        if col in df.columns:
            return col
    return None


def _load_model_rows(spec: ModelSpec, asof: pd.Timestamp) -> pd.DataFrame:
    path = spec.abs_ledger_path()
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    metric = spec.metric
    needed = [metric.status_col, metric.date_col, metric.realized_pct_col]
    if any(col not in df.columns for col in needed):
        return pd.DataFrame()
    coin_col = _coin_col(df)
    if coin_col is None:
        return pd.DataFrame()

    out = df[df[metric.status_col].astype(str) == metric.closed_value].copy()
    if out.empty:
        return out

    out["date"] = pd.to_datetime(out[metric.date_col], errors="coerce").dt.normalize()
    out = out[out["date"].notna() & (out["date"] <= asof)]
    if out.empty:
        return out

    out["coin"] = out[coin_col].astype(str)
    out["model_id"] = spec.id
    out["source_name"] = spec.name

    realized = pd.to_numeric(out[metric.realized_pct_col], errors="coerce")
    if not metric.cost_already_deducted:
        realized = realized - ROUND_TRIP_COST_PCT
    out["net_pct"] = realized

    hit = _selected_pump20_hits(out[["date", "coin"]])
    out = out.merge(hit, on=["date", "coin"], how="left")

    rank_col = _rank_col(out)
    if rank_col and rank_col != "rank":
        out["rank"] = pd.to_numeric(out[rank_col], errors="coerce")
    elif "rank" in out.columns:
        out["rank"] = pd.to_numeric(out["rank"], errors="coerce")
    else:
        out["rank"] = np.nan

    return out


def _selected_pump20_hits(keys: pd.DataFrame, db_path: Path = D1_DB) -> pd.DataFrame:
    if keys.empty or not db_path.exists():
        return pd.DataFrame(columns=["date", "coin", "selected_pump20_hit"])
    unique = keys.drop_duplicates(["date", "coin"]).copy()
    rows = []
    con = sqlite3.connect(db_path)
    try:
        for _, item in unique.iterrows():
            date = pd.Timestamp(item["date"])
            coin = str(item["coin"])
            ts = date.strftime("%Y-%m-%d 09:00:00")
            row = con.execute(
                "SELECT open, high FROM candles WHERE market=? AND timestamp=?",
                (coin, ts),
            ).fetchone()
            hit = np.nan
            if row and float(row[0]) > 0:
                hit = int(float(row[1]) / float(row[0]) - 1.0 >= PUMP20_THRESH)
            rows.append({"date": date, "coin": coin, "selected_pump20_hit": hit})
    finally:
        con.close()
    return pd.DataFrame(rows)


def _actual_pumps_for_dates(dates: list[pd.Timestamp],
                            db_path: Path = D1_DB) -> dict[pd.Timestamp, set[str]]:
    out: dict[pd.Timestamp, set[str]] = {}
    if not dates or not db_path.exists():
        return out
    con = sqlite3.connect(db_path)
    try:
        for d in sorted(set(pd.Timestamp(x).normalize() for x in dates)):
            ts = d.strftime("%Y-%m-%d 09:00:00")
            rows = con.execute(
                "SELECT market FROM candles WHERE timestamp=? AND open > 0 "
                "AND (high / open - 1.0) >= ?",
                (ts, PUMP20_THRESH),
            ).fetchall()
            out[d] = {str(r[0]) for r in rows}
    finally:
        con.close()
    return out


def _safe_pct(x: float | None) -> float | None:
    if x is None or not np.isfinite(x):
        return None
    return round(float(x), 4)


def _best_participant(rows: list[dict], metric: str) -> str | None:
    candidates = [
        r for r in rows
        if r.get("n_closed", 0) > 0 and r.get(metric) is not None
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda r: (
            float(r.get(metric) or float("-inf")),
            float(r.get("net_mean_pct") or float("-inf")),
            int(r.get("n_closed") or 0),
        ),
        reverse=True,
    )
    return str(candidates[0].get("participant_id"))


def _write_sqlite(payload: dict, db_path: Path = POLICY_DB) -> None:
    """Persist the latest policy competition snapshot for dashboard/ops audit.

    Same-day reruns replace the asof snapshot. This keeps the DB small and makes
    heartbeat/dashboard consumers read one canonical row set per trading day.
    """
    rows = payload.get("rows", [])
    db_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    con = sqlite3.connect(db_path)
    try:
        con.executescript(DB_SCHEMA)
        con.execute(
            """
            INSERT INTO policy_competition_runs (
                asof, generated_at_utc, pump20_threshold, deep_loss_pct,
                round_trip_cost_pct, row_count, best_pump_participant,
                best_net_participant, updated_at_utc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(asof) DO UPDATE SET
                generated_at_utc=excluded.generated_at_utc,
                pump20_threshold=excluded.pump20_threshold,
                deep_loss_pct=excluded.deep_loss_pct,
                round_trip_cost_pct=excluded.round_trip_cost_pct,
                row_count=excluded.row_count,
                best_pump_participant=excluded.best_pump_participant,
                best_net_participant=excluded.best_net_participant,
                updated_at_utc=excluded.updated_at_utc
            """,
            (
                payload["asof"],
                payload["generated_at_utc"],
                float(payload["config"]["pump20_threshold"]),
                float(payload["config"]["deep_loss_pct"]),
                float(payload["config"]["round_trip_cost_pct"]),
                len(rows),
                _best_participant(rows, "pump20_recall_pct"),
                _best_participant(rows, "net_mean_pct"),
                now,
            ),
        )
        con.execute("DELETE FROM policy_competition_rows WHERE asof=?", (payload["asof"],))
        con.executemany(
            """
            INSERT INTO policy_competition_rows (
                asof, participant_id, source_id, policy_id, objective, description,
                n_closed, n_days, net_sum_pct, net_mean_pct, deep_loss_freq_pct,
                sl_rate_pct, tp_rate_pct, pump20_precision_pct, pump20_recall_pct,
                pump20_captured, pump20_actual, pump_days, pump_days_any_captured,
                pump_day_capture_rate_pct, generated_at_utc
            )
            VALUES (
                :asof, :participant_id, :source_id, :policy_id, :objective,
                :description, :n_closed, :n_days, :net_sum_pct, :net_mean_pct,
                :deep_loss_freq_pct, :sl_rate_pct, :tp_rate_pct,
                :pump20_precision_pct, :pump20_recall_pct, :pump20_captured,
                :pump20_actual, :pump_days, :pump_days_any_captured,
                :pump_day_capture_rate_pct, :generated_at_utc
            )
            """,
            [
                {
                    **row,
                    "asof": payload["asof"],
                    "generated_at_utc": payload["generated_at_utc"],
                }
                for row in rows
            ],
        )
        con.commit()
    finally:
        con.close()


def _evaluate_rows(rows: pd.DataFrame, *, participant_id: str, source_id: str,
                   policy: PolicySpec, asof: pd.Timestamp,
                   actual_pumps: dict[pd.Timestamp, set[str]]) -> dict:
    selected = rows.copy()
    selected = selected[pd.to_numeric(selected["net_pct"], errors="coerce").notna()]
    selected["net_pct"] = pd.to_numeric(selected["net_pct"], errors="coerce")

    n_closed = int(len(selected))
    n_days = int(selected["date"].nunique()) if n_closed else 0

    if n_closed:
        net_sum = float(selected["net_pct"].sum())
        net_mean = float(selected["net_pct"].mean())
        deep_loss = float((selected["net_pct"] <= DEEP_LOSS_PCT).mean())
        pump_precision = float(pd.to_numeric(
            selected["selected_pump20_hit"], errors="coerce").dropna().mean())
        exit_reason = selected.get("exit_reason", pd.Series(index=selected.index, dtype=object))
        tp_rate = float((exit_reason.astype(str).str.upper() == "TP").mean())
        sl_rate = float((exit_reason.astype(str).str.upper() == "SL").mean())
    else:
        net_sum = net_mean = deep_loss = pump_precision = tp_rate = sl_rate = np.nan

    total_actual = 0
    captured = 0
    days_with_pump = 0
    days_any_captured = 0
    for date, day_rows in selected.groupby("date"):
        d = pd.Timestamp(date).normalize()
        pumps = actual_pumps.get(d, set())
        if not pumps:
            continue
        days_with_pump += 1
        total_actual += len(pumps)
        picked = set(day_rows["coin"].astype(str))
        day_captured = len(picked & pumps)
        captured += day_captured
        if day_captured:
            days_any_captured += 1

    pump_recall = captured / total_actual if total_actual else np.nan
    any_capture = days_any_captured / days_with_pump if days_with_pump else np.nan

    return {
        "asof": str(asof.date()),
        "participant_id": participant_id,
        "source_id": source_id,
        "policy_id": policy.policy_id,
        "objective": policy.objective,
        "description": policy.description,
        "n_closed": n_closed,
        "n_days": n_days,
        "net_sum_pct": _safe_pct(net_sum),
        "net_mean_pct": _safe_pct(net_mean),
        "deep_loss_freq_pct": _safe_pct(deep_loss * 100.0 if np.isfinite(deep_loss) else np.nan),
        "sl_rate_pct": _safe_pct(sl_rate * 100.0 if np.isfinite(sl_rate) else np.nan),
        "tp_rate_pct": _safe_pct(tp_rate * 100.0 if np.isfinite(tp_rate) else np.nan),
        "pump20_precision_pct": _safe_pct(
            pump_precision * 100.0 if np.isfinite(pump_precision) else np.nan),
        "pump20_recall_pct": _safe_pct(
            pump_recall * 100.0 if np.isfinite(pump_recall) else np.nan),
        "pump20_captured": int(captured),
        "pump20_actual": int(total_actual),
        "pump_days": int(days_with_pump),
        "pump_days_any_captured": int(days_any_captured),
        "pump_day_capture_rate_pct": _safe_pct(
            any_capture * 100.0 if np.isfinite(any_capture) else np.nan),
    }


def _consensus_rows(model_rows: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for model_id in ("recommend_r1_open", "recommend_r2_open", "recommend_r1_sustain_open"):
        df = model_rows.get(model_id)
        if df is not None and not df.empty:
            frames.append(df.copy())
    if len(frames) < 2:
        return pd.DataFrame()

    pool = pd.concat(frames, ignore_index=True)
    counts = (
        pool.groupby(["date", "coin"], as_index=False)
        .agg(source_count=("model_id", "nunique"), rank=("rank", "min"))
    )
    consensus = counts[counts["source_count"] >= 2]
    if consensus.empty:
        return consensus

    # Keep one realized path per date/coin. Realized is identical for the same
    # entry/SL/TP path; prefer R1 then R2 then A1 for stable metadata.
    priority = {
        "recommend_r1_open": 0,
        "recommend_r2_open": 1,
        "recommend_r1_sustain_open": 2,
    }
    pool["priority"] = pool["model_id"].map(priority).fillna(99)
    pool = pool.sort_values(["date", "coin", "priority"])
    first = pool.drop_duplicates(["date", "coin"], keep="first")
    out = consensus.merge(first.drop(columns=["rank"], errors="ignore"),
                          on=["date", "coin"], how="left")
    out["model_id"] = "consensus_2of3"
    out["source_name"] = "Consensus of at least 2 among R1/R2/A1"
    return out


def run(asof: pd.Timestamp | None = None, *, output_csv: Path = OUT_CSV,
        output_json: Path = OUT_JSON, db_path: Path | None = POLICY_DB) -> dict:
    asof = (asof or pd.Timestamp.now()).normalize()
    model_rows = {spec.id: _load_model_rows(spec, asof) for spec in MODELS}
    all_dates = sorted({
        pd.Timestamp(d).normalize()
        for df in model_rows.values() if df is not None and not df.empty
        for d in df["date"].dropna().unique()
    })
    actual_pumps = _actual_pumps_for_dates(all_dates)

    rows = []
    for spec in MODELS:
        df = model_rows.get(spec.id, pd.DataFrame())
        if df is None or df.empty:
            continue
        for policy in POLICIES:
            mask = policy.predicate(df)
            selected = df[mask.fillna(False)].copy()
            rows.append(_evaluate_rows(
                selected,
                participant_id=f"{spec.id}:{policy.policy_id}",
                source_id=spec.id,
                policy=policy,
                asof=asof,
                actual_pumps=actual_pumps,
            ))

    consensus = _consensus_rows(model_rows)
    if not consensus.empty:
        policy = PolicySpec(
            "consensus_2of3",
            "Send only coins selected by at least two of R1/R2/A1.",
            "consensus_precision",
            _true_series,
        )
        rows.append(_evaluate_rows(
            consensus,
            participant_id="consensus_2of3",
            source_id="R1/R2/A1",
            policy=policy,
            asof=asof,
            actual_pumps=actual_pumps,
        ))

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["objective", "pump20_recall_pct", "net_mean_pct", "n_closed"],
            ascending=[True, False, False, False],
            na_position="last",
        )

    rows_payload = (
        json.loads(summary.to_json(orient="records"))
        if not summary.empty else []
    )

    payload = {
        "asof": str(asof.date()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "pump20_threshold": PUMP20_THRESH,
            "deep_loss_pct": DEEP_LOSS_PCT,
            "round_trip_cost_pct": ROUND_TRIP_COST_PCT,
            "note": (
                "Record-only policy competition. It evaluates CLOSED forward rows. "
                "pump20_recall uses all KRW daily candles in upbit_d1.db for the same dates."
            ),
        },
        "rows": rows_payload,
    }

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    tmp_csv = output_csv.with_suffix(output_csv.suffix + ".tmp")
    tmp_json = output_json.with_suffix(output_json.suffix + ".tmp")
    summary.to_csv(tmp_csv, index=False)
    tmp_csv.replace(output_csv)
    with open(tmp_json, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)
    tmp_json.replace(output_json)
    if db_path is not None:
        _write_sqlite(payload, db_path)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate model + send-policy competition")
    ap.add_argument("--asof", type=str, default=None, help="YYYY-MM-DD (default=today)")
    ap.add_argument("--out-csv", type=Path, default=OUT_CSV)
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    ap.add_argument("--db", type=Path, default=POLICY_DB)
    ap.add_argument("--no-db", action="store_true", help="Do not persist SQLite audit DB")
    args = ap.parse_args()

    asof = pd.Timestamp(args.asof).normalize() if args.asof else pd.Timestamp.now().normalize()
    payload = run(
        asof,
        output_csv=args.out_csv,
        output_json=args.out_json,
        db_path=None if args.no_db else args.db,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

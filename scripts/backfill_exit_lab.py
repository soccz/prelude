"""Exit lab backfill — 이미 closed 된 shadow ledger rows 에 청산 변형 컬럼 채우기.

close_recommend_ledger.py 가 close 시점에 exit lab 컬럼을 채우도록 배선됐지만
(2026-06-11), 그 이전에 closed 된 rows 는 비어있다. 같은 15m 경로 데이터가
data/upbit_15m.db 에 남아있으므로 동일 평가를 소급 적용한다.

record-only — 알림/주문/모델 영향 0. realized_pct (운영 TP5/SL3) 는 건드리지 않음.

사용:
    python scripts/backfill_exit_lab.py            # 4 ledger 모두
    python scripts/backfill_exit_lab.py --dry-run  # 저장 X, 요약만
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ledger.config import ROUND_TRIP_COST_PCT  # noqa: E402
from ledger.exit_lab import EXIT_LAB_COLS, evaluate_exit_variants  # noqa: E402

M15_DB = "data/upbit_15m.db"

LEDGERS = [
    ("recommend_r1", "output/shadow_ledger_recommend.csv"),
    ("recommend_r2", "output/shadow_ledger_recommend_r2.csv"),
    ("recommend_a1", "output/shadow_ledger_recommend_sustain.csv"),
    ("pump_hunter", "output/shadow_ledger_pump_hunter.csv"),
]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("backfill_exit_lab")


def _load_15m_path(conn: sqlite3.Connection, coin: str, date: pd.Timestamp) -> list:
    """[09:00 D, 09:00 D+1) 의 15m 봉 — close_recommend_ledger 와 동일 윈도."""
    start = pd.Timestamp(date).strftime("%Y-%m-%d 09:00:00")
    end = (pd.Timestamp(date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d 09:00:00")
    return conn.execute(
        "SELECT open,high,low,close FROM candles WHERE market=? AND "
        "timestamp>=? AND timestamp<? ORDER BY timestamp", (coin, start, end)
    ).fetchall()


def backfill_ledger(name: str, path: str, conn: sqlite3.Connection, dry_run: bool) -> dict | None:
    p = Path(path)
    if not p.exists():
        log.warning("%s: ledger 없음 (%s) — skip", name, path)
        return None
    df = pd.read_csv(p)
    if df.empty or "status" not in df.columns:
        log.warning("%s: 빈 ledger — skip", name)
        return None
    for c in EXIT_LAB_COLS:
        if c not in df.columns:
            df[c] = pd.NA
        df[c] = df[c].astype(object)

    closed = df["status"].astype(str) == "closed"
    missing = closed & df["exit_tp10_nosl_pct"].isna()
    n_target = int(missing.sum())
    n_filled = 0
    n_no_bars = 0
    for idx, r in df[missing].iterrows():
        bars = _load_15m_path(conn, r["coin"], pd.to_datetime(r["date"]))
        lab = evaluate_exit_variants(bars, round_trip_cost=ROUND_TRIP_COST_PCT)
        if lab is None:
            n_no_bars += 1
            continue
        for k, v in lab.items():
            df.at[idx, k] = v
        n_filled += 1

    if not dry_run and n_filled:
        df.to_csv(p, index=False)
    log.info("%s: closed %d | backfill 대상 %d | 채움 %d | 경로없음 %d%s",
             name, int(closed.sum()), n_target, n_filled, n_no_bars,
             " (dry-run, 저장 X)" if dry_run else "")

    # 변형 비교 요약 (이 ledger 의 closed + lab 채워진 rows)
    have = df[closed & df["exit_tp10_nosl_pct"].notna()].copy()
    if have.empty:
        return None
    summary = {"model": name, "n": len(have)}
    base = pd.to_numeric(have["realized_pct"], errors="coerce")
    summary["tp5_sl3_net_mean"] = round(float(base.mean()), 3)
    summary["tp5_sl3_net_sum"] = round(float(base.sum()), 2)
    for v in ["tp10_nosl", "tp5_nosl", "eod"]:
        col = pd.to_numeric(have[f"exit_{v}_pct"], errors="coerce")
        summary[f"{v}_net_mean"] = round(float(col.mean()), 3)
        summary[f"{v}_net_sum"] = round(float(col.sum()), 2)
        summary[f"{v}_deep_loss_n"] = int((col <= -5.0).sum())
    summary["tp5_sl3_deep_loss_n"] = int((base <= -5.0).sum())
    return summary


def main():
    ap = argparse.ArgumentParser(description="exit lab 변형 컬럼 backfill (record-only)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    conn = sqlite3.connect(M15_DB)
    summaries = []
    for name, path in LEDGERS:
        s = backfill_ledger(name, path, conn, args.dry_run)
        if s:
            summaries.append(s)
    conn.close()

    if summaries:
        out = pd.DataFrame(summaries)
        cols = ["model", "n",
                "tp5_sl3_net_mean", "tp10_nosl_net_mean", "tp5_nosl_net_mean", "eod_net_mean",
                "tp5_sl3_net_sum", "tp10_nosl_net_sum", "tp5_nosl_net_sum", "eod_net_sum",
                "tp5_sl3_deep_loss_n", "tp10_nosl_deep_loss_n", "tp5_nosl_deep_loss_n", "eod_deep_loss_n"]
        out = out[[c for c in cols if c in out.columns]]
        print("\n=== exit lab 변형 비교 (net %, 왕복 0.15% 차감) ===")
        print(out.to_string(index=False))
        out.to_csv("output/exit_lab_backfill_summary.csv", index=False)
        print("\nsaved output/exit_lab_backfill_summary.csv")


if __name__ == "__main__":
    main()

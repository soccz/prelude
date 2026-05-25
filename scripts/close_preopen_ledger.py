"""Pre-open paper ledger close-out — first 15m/30m/1h realized 로 채움.

흐름:
  1. paper_ledger_preopen.csv 의 status="entered" 행 로드
  2. 각 행의 (coin, date) 에서 date 09:00 부터 첫 15m/30m/1h bars 조회
  3. realized 채움:
     first_open, first_15m_high, first_30m_high, first_1h_high
     hit_first15_3pct, hit_first15_5pct, ...
  4. status → "closed" + 저장

운영:
  매일 10:05 cron. 09:00 시작 첫 1시간 bar (4개) 마감 후 닫는다.
  4개 15m bar 가 아직 없으면 status 를 "entered" 로 유지해 다음 실행에서 retry.

사용:
    python scripts/close_preopen_ledger.py
    python scripts/close_preopen_ledger.py --asof 2026-05-06
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.database import load_candles
from scripts.predict_preopen_trigger import PAPER_LEDGER_COLS


DEFAULT_SHADOW_LEDGER = "output/shadow_ledger_preopen.csv"


def compute_realized_preopen(coin: str, label_date: pd.Timestamp,
                              candles_15m_db: str = "data/upbit_15m.db") -> dict:
    """coin 의 label_date 09:00 첫 N 15m bars 로 realized 계산."""
    df = load_candles(candles_15m_db, coin)
    if df is None or len(df) == 0:
        return {"status": "no_data"}
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    target_date = pd.to_datetime(label_date).date()
    target_ts_start = pd.Timestamp(target_date) + pd.Timedelta(hours=9)
    target_ts_end = target_ts_start + pd.Timedelta(hours=1)
    bars = df[(df["timestamp"] >= target_ts_start) &
              (df["timestamp"] < target_ts_end)].sort_values("timestamp")

    if len(bars) < 1:
        return {"status": "no_data"}

    first_bar = bars.iloc[0]
    if first_bar["timestamp"] != target_ts_start:
        return {"status": "no_data", "notes": f"first bar timestamp mismatch: {first_bar['timestamp']}"}

    opn = float(first_bar["open"])
    if opn <= 0:
        return {"status": "bad_open"}

    if len(bars) < 4:
        return {"status": "pending", "notes": f"only {len(bars)} 15m bars available"}

    highs = bars["high"].values.astype(float)
    lows = bars["low"].values.astype(float)
    closes = bars["close"].values.astype(float)

    h15 = float(highs[0])
    h30 = float(highs[:2].max())
    h1h = float(highs[:4].max())
    l15 = float(lows[0])
    l30 = float(lows[:2].min())
    l1h = float(lows[:4].min())
    c1h = float(closes[3])

    return {
        "status": "closed",
        "first_open": opn,
        "first_15m_high": h15,
        "first_30m_high": h30,
        "first_1h_high": h1h,
        "first_15m_low": l15,
        "first_30m_low": l30,
        "first_1h_low": l1h,
        "first_1h_close": c1h,
        "first_1h_max_return_pct": (h1h / opn - 1) * 100,
        "first_1h_min_return_pct": (l1h / opn - 1) * 100,
        "first_1h_close_return_pct": (c1h / opn - 1) * 100,
        "hit_first15_3pct": int(h15 >= opn * 1.03),
        "hit_first15_5pct": int(h15 >= opn * 1.05),
        "hit_first30_3pct": int(h30 >= opn * 1.03),
        "hit_first30_5pct": int(h30 >= opn * 1.05),
        "hit_first1h_3pct": int(h1h >= opn * 1.03),
        "hit_first1h_5pct": int(h1h >= opn * 1.05),
        "n_bars_available": int(len(bars)),
    }


def close_ledger_file(
    ledger_path: str,
    candles_15m_db: str,
    asof: pd.Timestamp,
    dry_run: bool,
    log: logging.Logger,
    *,
    required: bool = True,
) -> None:
    p = Path(ledger_path)
    if not p.exists():
        msg = f"ledger missing (no entries yet): {p}"
        if required:
            log.warning(msg)
        else:
            log.info(msg)
        return

    ledger = pd.read_csv(p)
    for col in PAPER_LEDGER_COLS:
        if col not in ledger.columns:
            ledger[col] = np.nan
    ledger["status"] = ledger["status"].fillna("").astype(str)
    ledger["notes"] = ledger["notes"].fillna("").astype(str)
    log.info(f"ledger: {len(ledger)} rows total")

    # For pre-open, can close SAME DAY (15m bar at 09:00 closes at 09:15, available immediately).
    # cutoff = asof.date() (today is closeable since 09:30 service runs after 09:15+).
    cutoff_date = asof.date()

    entered_mask = ledger["status"].isin(["entered", "no_data"])
    n_entered = int(entered_mask.sum())
    log.info(f"  entered/retry: {n_entered}, closing date <= {cutoff_date}")

    if n_entered == 0:
        log.info("nothing to close")
        return

    n_closed = 0
    n_no_data = 0
    rows_summary = []
    for idx, r in ledger[entered_mask].iterrows():
        entry_date = pd.to_datetime(r["date"]).date()
        if entry_date > cutoff_date:
            continue

        result = compute_realized_preopen(r["coin"], entry_date, candles_15m_db)
        if result["status"] == "no_data":
            ledger.at[idx, "status"] = "no_data"
            ledger.at[idx, "notes"] = result.get("notes", "no 15m data")
            n_no_data += 1
            continue
        if result["status"] == "pending":
            ledger.at[idx, "notes"] = result.get("notes", "pending")
            continue
        if result["status"] != "closed":
            continue

        for k, v in result.items():
            if k in ("status", "n_bars_available"):
                continue
            ledger.at[idx, k] = v
        ledger.at[idx, "status"] = "closed"
        n_closed += 1
        rows_summary.append({
            "date": r["date"], "coin": r["coin"],
            "h15_3": result["hit_first15_3pct"], "h15_5": result["hit_first15_5pct"],
            "h1h_3": result.get("hit_first1h_3pct"), "h1h_5": result.get("hit_first1h_5pct"),
        })

    log.info(f"  closed: {n_closed}, no_data: {n_no_data}")

    if not dry_run:
        ordered = [c for c in PAPER_LEDGER_COLS if c in ledger.columns]
        extras = [c for c in ledger.columns if c not in ordered]
        ledger = ledger[ordered + extras]
        ledger.to_csv(p, index=False)
        log.info(f"saved {p}")

    if rows_summary:
        df = pd.DataFrame(rows_summary)
        print(f"\n=== Closed (n={len(df)}) ===")
        print(df.to_string(index=False))
        for h in ["h15_3", "h15_5", "h1h_3", "h1h_5"]:
            r = df[h].dropna().mean()
            print(f"  hit_{h}: {r*100:.1f}% ({int(df[h].sum())}/{len(df[h].dropna())})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-ledger", default="output/paper_ledger_preopen.csv")
    parser.add_argument("--shadow-ledger", default=DEFAULT_SHADOW_LEDGER)
    parser.add_argument("--skip-shadow-ledger", action="store_true")
    parser.add_argument("--upbit-15m", default="data/upbit_15m.db")
    parser.add_argument("--asof", type=str, help="기준 시점 (default=now)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("close-preopen")

    asof = pd.Timestamp(args.asof) if args.asof else pd.Timestamp.now()
    close_ledger_file(args.paper_ledger, args.upbit_15m, asof, args.dry_run, log, required=True)

    if not args.skip_shadow_ledger and Path(args.shadow_ledger) != Path(args.paper_ledger):
        close_ledger_file(args.shadow_ledger, args.upbit_15m, asof, args.dry_run, log, required=False)


if __name__ == "__main__":
    main()

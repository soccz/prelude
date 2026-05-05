"""1h Upbit backfill quality check (사용자 권장 체크리스트).

체크 항목:
  1. market 별 row count 분포 (3년 ≈ 26k 기대; 신규 상장 코인은 적음)
  2. timestamp min/max 범위
  3. KST hour 분포가 0~23 고르게 — 특정 시간대 누락 detect
  4. (market, timestamp) 중복 detect
  5. 누락 시간 (gap) detect — 24h 안 1h bar 가 24개 미만인 일자
  6. 샘플 수동 확인: KRW-BTC, KRW-XRP 두 개 — 08:00 snapshot bar 가 실제 07:00 timestamp 인지

사용:
    python scripts/backfill_1h_qa.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.database import list_markets, load_candles, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/upbit_1h.db")
    parser.add_argument("--samples", nargs="+", default=["KRW-BTC", "KRW-XRP", "KRW-ETH"])
    parser.add_argument("--min-rows", type=int, default=10000,
                        help="이 미만 row 시 의심 (3년 기대 ~26k)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("qa")

    print("=== 1h Upbit Backfill QA ===\n")

    p = Path(args.db)
    if not p.exists():
        log.error(f"DB missing: {p}")
        sys.exit(1)

    markets = list_markets(args.db)
    print(f"Total markets: {len(markets)}\n")

    # === 1. Row count distribution ===
    print("1. Row count distribution per market:")
    s = stats(args.db)
    print(s.describe()[["rows"]].to_string())
    print(f"\n  markets < {args.min_rows} rows: {(s['rows'] < args.min_rows).sum()}")
    if (s['rows'] < args.min_rows).sum() > 0:
        print("  (likely newly listed, not necessarily a problem)")
        print(s[s["rows"] < args.min_rows].sort_values("rows").head(10).to_string(index=False))

    # === 2. Timestamp range ===
    print(f"\n2. Timestamp range:")
    print(f"   earliest oldest: {s['oldest'].min()}")
    print(f"   latest oldest:   {s['oldest'].max()}")
    print(f"   earliest latest: {s['latest'].min()}")
    print(f"   latest latest:   {s['latest'].max()}")

    # === 3-5. KST hour distribution + duplicates + gaps (sample first 30 markets for speed) ===
    print("\n3-5. Detailed checks on first 30 markets (sample)")
    sample_markets = markets[:30]
    hour_counter_global = Counter()
    n_dup_total = 0
    n_gap_days = 0

    for m in sample_markets:
        df = load_candles(args.db, m)
        if df is None or len(df) < 24:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        # hour distribution
        hours = df["timestamp"].dt.hour
        for h in hours:
            hour_counter_global[h] += 1
        # duplicates
        n_dup = df.duplicated(subset=["timestamp"]).sum()
        n_dup_total += int(n_dup)
        # gap days (count days where # 1h bars != 24)
        df["bar_date"] = (df["timestamp"] - pd.Timedelta(hours=9)).dt.date
        bars_per_day = df.groupby("bar_date").size()
        # exclude first/last day (likely partial)
        if len(bars_per_day) > 2:
            mid = bars_per_day.iloc[1:-1]
            n_gap = (mid != 24).sum()
            n_gap_days += int(n_gap)

    print(f"\n3. KST hour distribution (sum across 30 markets):")
    for h in sorted(hour_counter_global.keys()):
        bar = "█" * (hour_counter_global[h] // (max(hour_counter_global.values()) // 50 + 1))
        print(f"   {h:02d}h: {hour_counter_global[h]:>6,}  {bar}")
    if min(hour_counter_global.values()) < max(hour_counter_global.values()) * 0.85:
        print("   ⚠️ uneven distribution — some hours under-represented")
    else:
        print("   ✓ hours evenly distributed")

    print(f"\n4. Duplicate (market, timestamp) rows: {n_dup_total} (sample 30 markets)")
    print(f"5. Gap days (mid days with != 24 bars): {n_gap_days}")

    # === 6. Manual sample check ===
    print(f"\n6. Sample check — verify 08:00 snapshot semantics:")
    for m in args.samples:
        df = load_candles(args.db, m)
        if df is None or len(df) == 0:
            print(f"   {m}: NO DATA")
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        # find a recent date with all bars present
        df["bar_date"] = (df["timestamp"] - pd.Timedelta(hours=9)).dt.date
        recent_dates = sorted(df["bar_date"].unique())[-5:]
        print(f"\n   --- {m} ---")
        for date in recent_dates:
            sub = df[df["bar_date"] == date].sort_values("timestamp")
            n = len(sub)
            target_ts = pd.Timestamp(date) + pd.Timedelta(days=1)
            target_ts = target_ts.replace(hour=7)  # 07:00 of next day = 08:00 snapshot bar
            bar_at_07 = sub[sub["timestamp"] == target_ts]
            if len(bar_at_07) > 0:
                r = bar_at_07.iloc[0]
                print(f"     bar_date {date}: {n} bars total | 07:00 bar found: "
                      f"O={r['open']:.4f} H={r['high']:.4f} L={r['low']:.4f} C={r['close']:.4f} V={r['volume']:.2f}")
            else:
                print(f"     bar_date {date}: {n} bars total | 07:00 bar MISSING")

    # === 7. Markets with too few rows ===
    print(f"\n7. Markets that may need re-fetch (row count anomaly):")
    suspicious = s[s["rows"] < 1000]  # <1000 rows = <40 days
    if len(suspicious) > 0:
        print(suspicious.to_string(index=False))
    else:
        print("   None — all markets have ≥ 1000 rows (~40 days)")

    print("\n=== QA done ===")


if __name__ == "__main__":
    main()

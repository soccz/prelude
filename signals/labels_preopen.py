"""Pre-open Trigger Labels — 09:00 직후 첫 15분 펌프 hit.

타겟: KST 09:00 시작 daily candle 의 **첫 15m bar (09:00-09:15)** 의 high.

Labels:
  hit_first15_3pct: bar.high >= bar.open * 1.03
  hit_first15_5pct: bar.high >= bar.open * 1.05
  hit_first30_3pct: max(high of first 2 bars) >= open * 1.03
  hit_first30_5pct: max(high of first 2 bars) >= open * 1.05
  hit_first1h_3pct: max(high of first 4 bars) >= open * 1.03

base rate (전체 KRW 대상 추정):
  first15 +3%: ~5%
  first15 +5%: ~3%
  first30 +3%: ~8%
  first1h +3%: ~12%

distribution beta 의 h2 (4h +3%) 보다 더 좁음.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 15m bars: 4 per hour, 96 per day
# bar at "X+1 09:00" timestamp = covers KST 09:00-09:15 of X+1 day
# This is the FIRST bar of X+1 daily candle


PREOPEN_HEADS = {
    "first15_3pct": {"n_bars": 1, "target_pct": 3, "name": "09:00-09:15 안 +3%"},
    "first15_5pct": {"n_bars": 1, "target_pct": 5, "name": "09:00-09:15 안 +5%"},
    "first30_3pct": {"n_bars": 2, "target_pct": 3, "name": "09:00-09:30 안 +3%"},
    "first30_5pct": {"n_bars": 2, "target_pct": 5, "name": "09:00-09:30 안 +5%"},
    "first1h_3pct": {"n_bars": 4, "target_pct": 3, "name": "09:00-10:00 안 +3%"},
    "first1h_5pct": {"n_bars": 4, "target_pct": 5, "name": "09:00-10:00 안 +5%"},
}


def compute_preopen_labels(
    candles_15m: dict,
    target_dates: list = None,
) -> pd.DataFrame:
    """For each (market, day X), compute labels from first N 15m bars after 09:00 X+1.

    Returns DataFrame with columns:
      market, label_date (= X+1 day, when the candle starts),
      first_open, first_15m_high, first_30m_high, first_1h_high,
      hit_first15_3pct, hit_first15_5pct, hit_first30_3pct, ...
    """
    rows = []
    for market, df in candles_15m.items():
        if df is None or len(df) < 4:
            continue
        df = df.sort_values("timestamp").copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Find all "09:00 KST" bars — these are first bar of each daily candle
        first_bars = df[(df["timestamp"].dt.hour == 9) & (df["timestamp"].dt.minute == 0)]

        for _, fb in first_bars.iterrows():
            ts = fb["timestamp"]
            label_date = ts.date()  # the day this candle represents
            opn = float(fb["open"])
            if opn <= 0:
                continue
            # Get next 4 bars (1 hour) from this point
            window = df[(df["timestamp"] >= ts) &
                         (df["timestamp"] < ts + pd.Timedelta(hours=1))].sort_values("timestamp")
            if len(window) < 1:
                continue
            highs = window["high"].values.astype(float)

            row = {
                "market": market,
                "label_date": label_date,
                "first_open": opn,
                "first_15m_high": float(highs[0]) if len(highs) >= 1 else np.nan,
                "first_30m_high": float(highs[:2].max()) if len(highs) >= 2 else np.nan,
                "first_1h_high": float(highs[:4].max()) if len(highs) >= 4 else np.nan,
            }

            for head_id, cfg in PREOPEN_HEADS.items():
                n_bars = cfg["n_bars"]
                tgt = cfg["target_pct"]
                if len(highs) >= n_bars:
                    max_high = float(highs[:n_bars].max())
                    row[head_id] = int(max_high >= opn * (1 + tgt / 100))
                else:
                    row[head_id] = np.nan

            rows.append(row)

    return pd.DataFrame(rows)

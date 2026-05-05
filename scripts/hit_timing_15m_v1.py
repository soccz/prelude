"""15m hit timing audit for the 09:05 alert schedule.

Purpose:
  The current Stage 1 timer runs at KST 09:05 because closed daily candles are
  available and train/live alignment is clean. This audit answers the separate
  execution question: how often does the target move already start inside the
  first 15m candle after KST 09:00?

Interpretation:
  A 15m candle cannot distinguish 09:00-09:05 from 09:05-09:15. Therefore the
  "first_15m" share is an upper bound on what a 09:05 alert could miss.

Outputs:
  output/hit_timing_15m_v1.csv

Usage:
  python scripts/hit_timing_15m_v1.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.database import list_markets, load_candles
from signals.precursors import FIFTEEN_M_HIT_THRESHOLDS, build_15m_event_table, cached_frame


def summarize_segment(events: pd.DataFrame, segment: str) -> pd.DataFrame:
    rows = []
    n_samples = len(events)
    if n_samples == 0:
        return pd.DataFrame()

    for name, threshold in FIFTEEN_M_HIT_THRESHOLDS.items():
        idx_col = f"{name}_first_bar15"
        hit_col = f"{name}_hit"
        hits = events[events[hit_col] == 1].copy()
        n_hits = len(hits)
        if n_hits == 0:
            rows.append({
                "segment": segment,
                "threshold_pct": threshold * 100,
                "n_samples": n_samples,
                "n_hits": 0,
                "hit_rate_pct": 0.0,
                "first_15m_share_of_hits_pct": np.nan,
                "first_30m_share_of_hits_pct": np.nan,
                "first_1h_share_of_hits_pct": np.nan,
                "first_4h_share_of_hits_pct": np.nan,
                "median_first_hit_minute": np.nan,
                "p75_first_hit_minute": np.nan,
            })
            continue

        idx = hits[idx_col].astype(float)
        rows.append({
            "segment": segment,
            "threshold_pct": threshold * 100,
            "n_samples": n_samples,
            "n_hits": n_hits,
            "hit_rate_pct": n_hits / n_samples * 100,
            "first_15m_share_of_hits_pct": (idx == 0).mean() * 100,
            "first_30m_share_of_hits_pct": (idx < 2).mean() * 100,
            "first_1h_share_of_hits_pct": (idx < 4).mean() * 100,
            "first_4h_share_of_hits_pct": (idx < 16).mean() * 100,
            "median_first_hit_minute": float(np.nanmedian(idx * 15)),
            "p75_first_hit_minute": float(np.nanpercentile(idx * 15, 75)),
        })
    return pd.DataFrame(rows)


def load_alert_subset(events: pd.DataFrame, paper_ledger: str) -> pd.DataFrame:
    path = Path(paper_ledger)
    if not path.exists():
        return pd.DataFrame()
    ledger = pd.read_csv(path)
    if len(ledger) == 0 or not {"date", "coin"}.issubset(ledger.columns):
        return pd.DataFrame()
    ledger = ledger[["date", "coin", "setup_ids", "btc_regime", "alert_rank"]].copy()
    ledger["date"] = ledger["date"].astype(str)
    ledger["coin"] = ledger["coin"].astype(str)
    return ledger.merge(events, on=["date", "coin"], how="inner")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upbit-15m", default="data/upbit_15m.db")
    parser.add_argument("--paper-ledger", default="output/paper_ledger_backfill.csv")
    parser.add_argument("--cache-path", default="output/cache/hit_timing_15m_events.pkl")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--min-bars", type=int, default=80)
    parser.add_argument("--out-csv", default="output/hit_timing_15m_v1.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("hit_timing")

    events = cached_frame(
        args.cache_path,
        lambda: build_15m_event_table(
            {m: load_candles(args.upbit_15m, m)
             for m in list_markets(args.upbit_15m) if m.startswith("KRW-")},
            min_bars=args.min_bars,
        ),
        refresh=args.refresh_cache,
    )
    log.info("event rows=%s, dates=%s -> %s",
             f"{len(events):,}", events["date"].min(), events["date"].max())

    summaries = [summarize_segment(events, "all_market_days")]

    alerts = load_alert_subset(events, args.paper_ledger)
    if len(alerts) > 0:
        log.info("matched alert rows=%s", f"{len(alerts):,}")
        summaries.append(summarize_segment(alerts, "distribution_alerts"))
        for setup in ["S01", "S02", "S03"]:
            sub = alerts[alerts["setup_ids"].fillna("").str.contains(setup, regex=False)]
            if len(sub) > 0:
                summaries.append(summarize_segment(sub, f"alerts_{setup}"))

    out = pd.concat([s for s in summaries if len(s) > 0], ignore_index=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    log.info("saved %s", args.out_csv)

    print("=== 15m Hit Timing Audit v1 ===")
    print("first_15m = 09:00-09:15 candle; this is an upper bound for 09:05 lateness.")
    cols = [
        "segment", "threshold_pct", "n_samples", "n_hits", "hit_rate_pct",
        "first_15m_share_of_hits_pct", "first_30m_share_of_hits_pct",
        "first_1h_share_of_hits_pct", "first_4h_share_of_hits_pct",
        "median_first_hit_minute", "p75_first_hit_minute",
    ]
    print(out[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()

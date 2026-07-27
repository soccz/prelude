"""15m path-aware TP/SL backtest.

목적:
  4h path backtest 는 같은 bar 안 TP/SL 순서를 알 수 없어 SL-first 보수 가정이
  과하게 나쁠 수 있다. 15m path 로 최근 1년 구간에서 SL 룰을 재검증한다.

입력:
  output/baseline_showdown_trades_v1.csv

출력:
  output/path_backtest_15m_v1.csv
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
from data.market_universe import signal_eligible_markets
from scripts.path_backtest_distribution_v1 import RULES, simulate_path, summarize


def build_15m_path_table(upbit_15m: str) -> pd.DataFrame:
    rows = []
    markets = [
        market
        for market in signal_eligible_markets(list_markets(upbit_15m))
        if market.startswith("KRW-")
    ]
    for market in markets:
        df = load_candles(upbit_15m, market)
        if df is None or len(df) == 0:
            continue
        df = df.sort_values("timestamp").copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = (df["timestamp"] - pd.Timedelta(hours=9)).dt.date.astype(str)
        for date, g in df.groupby("date", sort=False):
            g2 = g.sort_values("timestamp")
            if len(g2) < 12:
                continue
            highs = g2["high"].values.astype(float)
            lows = g2["low"].values.astype(float)
            opens = g2["open"].values.astype(float)
            closes = g2["close"].values.astype(float)
            n = min(len(g2), 96)
            rows.append({
                "coin": market,
                "date": date,
                "open_4h": float(opens[0]),
                "close_4h": float(closes[min(n, len(closes)) - 1]),
                "highs": highs[:n],
                "lows": lows[:n],
                "n_bars": n,
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades", default="output/baseline_showdown_trades_v1.csv")
    parser.add_argument("--upbit-15m", default="data/upbit_15m.db")
    parser.add_argument("--out-csv", default="output/path_backtest_15m_v1.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("path15m")

    trades = pd.read_csv(args.trades)
    paths = build_15m_path_table(args.upbit_15m)
    log.info("trades=%s paths=%s", f"{len(trades):,}", f"{len(paths):,}")
    df = trades.merge(paths, on=["coin", "date"], how="inner")
    log.info("matched trades=%s, date range=%s -> %s", f"{len(df):,}", df["date"].min(), df["date"].max())
    df["date_dt"] = pd.to_datetime(df["date"])

    chunks = []
    for rule, (tp, sl) in RULES.items():
        for cost in [0.0015, 0.0020]:
            tmp = df[["strategy", "date", "date_dt", "coin", "open_4h", "close_4h", "highs", "lows", "n_bars"]].copy()
            tmp["rule"] = rule
            tmp["cost_pct"] = cost * 100
            tmp["net"] = tmp.apply(lambda r: simulate_path(r, tp, sl, cost), axis=1)
            chunks.append(tmp.dropna(subset=["net"]))
    all_results = pd.concat(chunks, ignore_index=True)
    summary = summarize(all_results)
    summary = summary.sort_values(["rule", "cost_pct", "annualized_sharpe"], ascending=[True, True, False])
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_csv, index=False)
    log.info("saved %s", args.out_csv)

    print("=== 15m Path Backtest v1 ===")
    cols = [
        "strategy", "rule", "cost_pct", "n_trades", "n_days", "avg_trade_net_pct",
        "pos_trade_pct", "cum_return_pct", "annualized_sharpe", "mdd_pct",
    ]
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    print("\n--- Best per strategy (cost 0.15%) ---")
    best = summary[summary["cost_pct"] == 0.15].sort_values(
        ["strategy", "annualized_sharpe"], ascending=[True, False]
    ).groupby("strategy", as_index=False).head(1)
    print(best[cols].sort_values("annualized_sharpe", ascending=False).to_string(
        index=False, float_format=lambda x: f"{x:+.3f}"
    ))


if __name__ == "__main__":
    main()

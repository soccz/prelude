"""Path-aware TP/SL backtest for baseline showdown trades.

입력:
  output/baseline_showdown_trades_v1.csv

역할:
  같은 trade set 에 대해 4h OHLC path 로 TP/SL first-touch 를 보수적으로 계산한다.
  같은 4h bar 안에서 TP 와 SL 이 동시에 닿으면 SL first 로 본다.

룰:
  TP3_only, TP5_only, TP3_SL3, TP5_SL3, TP5_SL5, TP10_SL5, TP20_only

주의:
  4h 내부 순서는 알 수 없으므로 SL-first 는 보수적 근사다.
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
from signals.features import compute_btc_features
from scripts.backfill_paper_ledger import build_4h_panel_for_labels


RULES = {
    "TP3_only": (0.03, None),
    "TP5_only": (0.05, None),
    "TP20_only": (0.20, None),
    "TP3_SL3": (0.03, -0.03),
    "TP5_SL3": (0.05, -0.03),
    "TP5_SL5": (0.05, -0.05),
    "TP10_SL5": (0.10, -0.05),
}


def build_path_table(upbit_d1: str, upbit_4h: str) -> pd.DataFrame:
    btc_d1 = load_candles(upbit_d1, "KRW-BTC")
    btc_feat = compute_btc_features(btc_d1.copy())
    btc_feat["date_only"] = pd.to_datetime(btc_feat["timestamp"]).dt.date
    btc_regime_map = dict(zip(btc_feat["date_only"], btc_feat["btc_regime"]))
    krw_4h = [
        market
        for market in signal_eligible_markets(list_markets(upbit_4h))
        if market.startswith("KRW-")
    ]
    candles_4h = {m: load_candles(upbit_4h, m) for m in krw_4h}
    candles_4h = {k: v for k, v in candles_4h.items() if v is not None and len(v) > 0}
    paths = build_4h_panel_for_labels(candles_4h, btc_regime_map)
    paths = paths.rename(columns={"date_only": "date", "market": "coin"})
    paths["date"] = paths["date"].astype(str)
    return paths[["coin", "date", "open_4h", "close_4h", "highs", "lows", "n_bars"]]


def simulate_path(row: pd.Series, tp: float, sl: float | None, cost: float) -> float:
    opn = float(row["open_4h"])
    close = float(row["close_4h"])
    if not np.isfinite(opn) or opn <= 0 or not np.isfinite(close):
        return np.nan
    highs = np.asarray(row["highs"], dtype=float)
    lows = np.asarray(row["lows"], dtype=float)
    n = int(row.get("n_bars", len(highs)) or len(highs))
    highs = highs[:n]
    lows = lows[:n]
    tp_px = opn * (1 + tp)
    sl_px = opn * (1 + sl) if sl is not None else None
    for hi, lo in zip(highs, lows):
        if not np.isfinite(hi) or not np.isfinite(lo):
            continue
        # conservative: if same 4h bar touches both, assume SL first.
        if sl_px is not None and lo <= sl_px:
            return sl - cost
        if hi >= tp_px:
            return tp - cost
    return (close / opn - 1) - cost


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (strategy, rule, cost), g in df.groupby(["strategy", "rule", "cost_pct"], sort=False):
        daily_n = g.groupby("date_dt").size().rename("n")
        gg = g.join(daily_n, on="date_dt")
        daily = (gg["net"] / gg["n"]).groupby(gg["date_dt"]).sum().sort_index()
        eq = (1 + daily).cumprod()
        sharpe = 0.0
        if len(daily) > 1 and daily.std() > 0:
            sharpe = float(daily.mean() / daily.std() * np.sqrt(365))
        mdd = ((eq - eq.cummax()) / eq.cummax()).min() if len(eq) else np.nan
        rows.append({
            "strategy": strategy,
            "rule": rule,
            "cost_pct": cost,
            "n_trades": int(len(g)),
            "n_days": int(daily.index.nunique()),
            "avg_trade_net_pct": float(g["net"].mean() * 100),
            "pos_trade_pct": float((g["net"] > 0).mean() * 100),
            "cum_return_pct": float((eq.iloc[-1] - 1) * 100) if len(eq) else np.nan,
            "annualized_sharpe": sharpe,
            "mdd_pct": float(mdd * 100) if pd.notna(mdd) else np.nan,
            "worst_day_pct": float(daily.min() * 100) if len(daily) else np.nan,
        })
    return pd.DataFrame(rows)


def daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (strategy, rule, cost), g in df.groupby(["strategy", "rule", "cost_pct"], sort=False):
        daily_n = g.groupby("date_dt").size().rename("n")
        gg = g.join(daily_n, on="date_dt")
        daily = (gg["net"] / gg["n"]).groupby(gg["date_dt"]).sum().sort_index()
        for dt, ret in daily.items():
            rows.append({
                "strategy": strategy,
                "rule": rule,
                "cost_pct": cost,
                "date": dt.date().isoformat(),
                "daily_ret": float(ret),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades", default="output/baseline_showdown_trades_v1.csv")
    parser.add_argument("--upbit-d1", default="data/upbit_d1.db")
    parser.add_argument("--upbit-4h", default="data/upbit_4h.db")
    parser.add_argument("--out-csv", default="output/path_backtest_distribution_v1.csv")
    parser.add_argument("--out-daily", default="output/path_backtest_distribution_daily_v1.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("pathbt")

    trades = pd.read_csv(args.trades)
    log.info("trades: %s", f"{len(trades):,}")
    paths = build_path_table(args.upbit_d1, args.upbit_4h)
    log.info("paths: %s", f"{len(paths):,}")
    df = trades.merge(paths, on=["coin", "date"], how="left")
    missing = int(df["open_4h"].isna().sum())
    if missing:
        log.warning("missing path rows: %s", missing)
    df = df.dropna(subset=["open_4h"]).copy()
    df["date_dt"] = pd.to_datetime(df["date"])

    rows = []
    for rule, (tp, sl) in RULES.items():
        for cost in [0.0015, 0.0020]:
            tmp = df[["strategy", "date", "date_dt", "coin", "open_4h", "close_4h", "highs", "lows", "n_bars"]].copy()
            tmp["rule"] = rule
            tmp["cost_pct"] = cost * 100
            tmp["net"] = tmp.apply(lambda r: simulate_path(r, tp, sl, cost), axis=1)
            rows.append(tmp.dropna(subset=["net"]))
    all_results = pd.concat(rows, ignore_index=True)
    summary = summarize(all_results)
    daily = daily_returns(all_results)
    summary = summary.sort_values(["rule", "cost_pct", "annualized_sharpe"], ascending=[True, True, False])
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_csv, index=False)
    daily.to_csv(args.out_daily, index=False)
    log.info("saved %s", args.out_csv)
    log.info("saved %s", args.out_daily)

    print("=== Path-aware Backtest v1 (4h, SL-first conservative) ===")
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

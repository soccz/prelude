"""Baseline showdown for distribution beta.

목적:
  distribution beta 가 단순 baseline 을 이기는지 확인한다.

설계:
  - 기준 날짜/추천 개수 K 는 paper_ledger_backfill 의 model alert 를 그대로 사용.
  - 각 날짜마다 같은 K 개를 다음 전략으로 선택한다.
      distribution_beta: 실제 backfill alert
      random_top100:     top100 universe 안 deterministic random
      momentum_1d:       log_return_1d 상위
      atr_14:            atr_pct_14 상위
      setup_only:        S01/S02/S03 setup score 상위
      setup_momentum:    setup score + momentum
      vol_momentum:      vol_5d + return_7d + log_return_1d
  - realized label 은 4h label panel 에서 가져온다.
  - TP3/TP5/TP20/tiered + EOD PnL 을 equal-size per day 로 계산한다.

주의:
  tiered 는 h5/h6/h2 hit 여부만 보고 큰 TP 우선으로 가정하므로 optimistic upper bound.
  production 로직은 바꾸지 않는 audit 전용 스크립트다.
"""
from __future__ import annotations

import argparse
import logging
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.database import list_markets, load_candles
from signals.features import assemble_training_panel, compute_btc_features
from signals.labels_distribution import compute_distribution_labels
from signals.setups import detect_setups

from scripts.backfill_paper_ledger import build_4h_panel_for_labels


RULES = ["TP3", "TP5", "TP20", "tiered"]
COSTS = [0.0015, 0.0020]


def deterministic_random_score(date_value, market: str) -> float:
    key = f"{date_value}|{market}".encode()
    return zlib.crc32(key) / 2**32


def per_trade_gross_return(row: pd.Series, rule: str) -> float:
    max_ret = row.get("next_max_return_pct", np.nan)
    eod_ret = row.get("next_close_return_pct", np.nan)
    if pd.isna(max_ret) or pd.isna(eod_ret):
        return np.nan
    max_ret = float(max_ret) / 100
    eod_ret = float(eod_ret) / 100
    if rule == "TP3":
        return 0.03 if max_ret >= 0.03 else eod_ret
    if rule == "TP5":
        return 0.05 if max_ret >= 0.05 else eod_ret
    if rule == "TP20":
        return 0.20 if max_ret >= 0.20 else eod_ret
    if rule == "tiered":
        if int(row.get("hit_h5", 0) or 0) == 1:
            return 0.20
        if int(row.get("hit_h6", 0) or 0) == 1:
            return 0.05
        if int(row.get("hit_h2", 0) or 0) == 1:
            return 0.03
        return eod_ret
    raise ValueError(rule)


def backtest_strategy(trades: pd.DataFrame, rule: str, cost: float) -> dict:
    df = trades.copy()
    df["gross"] = df.apply(lambda r: per_trade_gross_return(r, rule), axis=1)
    n_missing = int(df["gross"].isna().sum())
    df = df.dropna(subset=["gross"]).copy()
    if len(df) == 0:
        return {}
    df["net"] = df["gross"] - cost
    df["date_dt"] = pd.to_datetime(df["date"])
    daily_n = df.groupby("date_dt").size().rename("n_alerts")
    df = df.join(daily_n, on="date_dt")
    df["weighted_ret"] = df["net"] / df["n_alerts"]
    daily = df.groupby("date_dt")["weighted_ret"].sum().sort_index()
    eq = (1 + daily).cumprod()
    mdd = ((eq - eq.cummax()) / eq.cummax()).min() if len(eq) else np.nan
    sharpe = 0.0
    if len(daily) > 1 and daily.std() > 0:
        sharpe = float(daily.mean() / daily.std() * np.sqrt(365))
    return {
        "strategy": str(df["strategy"].iloc[0]),
        "rule": rule,
        "cost_pct": cost * 100,
        "n_trades": int(len(df)),
        "n_days": int(daily.index.nunique()),
        "avg_alerts_per_day": float(daily_n.mean()),
        "n_missing_realized": n_missing,
        "avg_trade_net_pct": float(df["net"].mean() * 100),
        "pos_trade_pct": float((df["net"] > 0).mean() * 100),
        "cum_return_pct": float((eq.iloc[-1] - 1) * 100) if len(eq) else np.nan,
        "annualized_sharpe": sharpe,
        "mdd_pct": float(mdd * 100) if pd.notna(mdd) else np.nan,
        "worst_day_pct": float(daily.min() * 100) if len(daily) else np.nan,
        "best_day_pct": float(daily.max() * 100) if len(daily) else np.nan,
    }


def build_labeled_panel(upbit_d1: str, upbit_4h: str, binance_d1: str) -> pd.DataFrame:
    krw = list_markets(upbit_d1)
    candles_d1 = {m: load_candles(upbit_d1, m) for m in krw}
    if Path(binance_d1).exists():
        for m in list_markets(binance_d1):
            candles_d1[m] = load_candles(binance_d1, m)
    candles_d1 = {k: v for k, v in candles_d1.items() if v is not None and len(v) > 30}
    btc_d1 = load_candles(upbit_d1, "KRW-BTC")
    panel = assemble_training_panel(candles_d1, btc_d1, normalize=True)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values(["market", "timestamp"]).reset_index(drop=True)
    panel["date_only"] = panel["timestamp"].dt.date
    panel["quote_volume_d1"] = panel.get("quote_volume", np.nan)

    btc_feat = compute_btc_features(btc_d1.copy())
    btc_feat["date_only"] = pd.to_datetime(btc_feat["timestamp"]).dt.date
    btc_regime_map = dict(zip(btc_feat["date_only"], btc_feat["btc_regime"]))
    krw_4h = [m for m in list_markets(upbit_4h) if m.startswith("KRW-")]
    candles_4h = {m: load_candles(upbit_4h, m) for m in krw_4h}
    candles_4h = {k: v for k, v in candles_4h.items() if v is not None and len(v) > 0}
    panel_4h = build_4h_panel_for_labels(candles_4h, btc_regime_map)
    label_df = compute_distribution_labels(panel_4h)
    label_df["market"] = panel_4h["market"].values
    label_df["label_date"] = panel_4h["date_only"].values
    label_df["eod_ret_next"] = panel_4h["eod_ret_4h"].values
    label_df["max_ret_next"] = panel_4h["max_ret_4h"].values
    label_df["date_only"] = (pd.to_datetime(label_df["label_date"]) - pd.Timedelta(days=1)).dt.date

    full = panel.merge(label_df, on=["market", "date_only"], how="inner")
    full = full[full["market"].str.startswith("KRW-")].copy()
    full["liq_rank_daily"] = full.groupby("date_only")["quote_volume_d1"].rank(
        method="dense", ascending=False, na_option="bottom"
    )
    full["setups"] = full.apply(detect_setups, axis=1)
    full["primary_setups"] = full["setups"].apply(lambda xs: [x for x in xs if x != "S04"])
    full["setup_ids"] = full["primary_setups"].apply(lambda xs: "+".join(xs) if xs else "—")
    full["setup_score"] = full["primary_setups"].apply(
        lambda xs: (1.0 if "S01" in xs else 0.0)
        + (1.0 if "S02" in xs else 0.0)
        + (2.0 if "S03" in xs else 0.0)
    )
    full["target_date"] = pd.to_datetime(full["label_date"]).dt.date.astype(str)
    full["next_max_return_pct"] = full["max_ret_next"] * 100
    full["next_close_return_pct"] = full["eod_ret_next"] * 100
    full["hit_h2"] = full["h2_hit_3_4h"].fillna(0).astype(int)
    full["hit_h5"] = full["h5_tail_20"].fillna(0).astype(int)
    full["hit_h6"] = full["h6_hit_5_24h"].fillna(0).astype(int)
    return full


def select_baseline(full: pd.DataFrame, target_ks: pd.Series, strategy: str,
                    universe: str) -> pd.DataFrame:
    sub = full[full["target_date"].isin(target_ks.index)].copy()
    if universe.startswith("top"):
        n = int(universe.replace("top", ""))
        sub = sub[sub["liq_rank_daily"] <= n].copy()

    if strategy == "random_top100":
        sub["rank_score"] = [
            deterministic_random_score(d, m) for d, m in zip(sub["target_date"], sub["market"])
        ]
    elif strategy == "momentum_1d":
        sub["rank_score"] = sub["log_return_1d"].fillna(-999)
    elif strategy == "atr_14":
        sub["rank_score"] = sub["atr_pct_14"].fillna(-999)
    elif strategy == "setup_only":
        sub["rank_score"] = sub["setup_score"].fillna(0) * 10 + sub["log_return_1d"].fillna(0)
        sub = sub[sub["setup_score"] > 0]
    elif strategy == "setup_momentum":
        sub["rank_score"] = (
            sub["setup_score"].fillna(0) * 10
            + sub["log_return_1d"].fillna(0) * 5
            + sub["return_7d"].fillna(0)
        )
        sub = sub[sub["setup_score"] > 0]
    elif strategy == "vol_momentum":
        sub["rank_score"] = (
            sub["vol_5d"].fillna(0)
            + sub["return_7d"].fillna(0)
            + sub["roc_3d"].fillna(0)
            + sub["log_return_1d"].fillna(0) * 3
        )
    else:
        raise ValueError(strategy)

    out = []
    for date, day in sub.groupby("target_date", sort=True):
        k = int(target_ks.loc[date])
        if k <= 0:
            continue
        pick = day.sort_values("rank_score", ascending=False).head(k).copy()
        pick["alert_rank"] = range(1, len(pick) + 1)
        out.append(pick)
    if not out:
        return pd.DataFrame()
    picked = pd.concat(out, ignore_index=True)
    return pd.DataFrame({
        "strategy": strategy,
        "date": picked["target_date"],
        "coin": picked["market"],
        "setup_ids": picked["setup_ids"],
        "btc_regime": picked.get("btc_regime", "unknown"),
        "alert_rank": picked["alert_rank"],
        "next_max_return_pct": picked["next_max_return_pct"],
        "next_close_return_pct": picked["next_close_return_pct"],
        "hit_h2": picked["hit_h2"],
        "hit_h5": picked["hit_h5"],
        "hit_h6": picked["hit_h6"],
    })


def model_trades_from_ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    df = ledger[ledger["status"] == "closed"].copy()
    out = pd.DataFrame({
        "strategy": "distribution_beta",
        "date": df["date"].astype(str),
        "coin": df["coin"],
        "setup_ids": df.get("setup_ids", "—"),
        "btc_regime": df.get("btc_regime", "unknown"),
        "alert_rank": df.get("alert_rank", np.nan),
        "next_max_return_pct": df["next_max_return_pct"],
        "next_close_return_pct": df["next_close_return_pct"],
        "hit_h2": df["hit_h2"],
        "hit_h5": df["hit_h5"],
        "hit_h6": df["hit_h6"],
    })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-ledger", default="output/paper_ledger_backfill.csv")
    parser.add_argument("--upbit-d1", default="data/upbit_d1.db")
    parser.add_argument("--upbit-4h", default="data/upbit_4h.db")
    parser.add_argument("--binance-d1", default="data/binance_d1.db")
    parser.add_argument("--universe", default="top100")
    parser.add_argument("--out-summary", default="output/baseline_showdown_v1.csv")
    parser.add_argument("--out-trades", default="output/baseline_showdown_trades_v1.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("baseline")

    ledger = pd.read_csv(args.paper_ledger)
    ledger = ledger[ledger["status"] == "closed"].copy()
    target_ks = ledger.groupby(ledger["date"].astype(str)).size()
    log.info("model ledger: %s trades / %s days", f"{len(ledger):,}", len(target_ks))

    log.info("building labeled panel...")
    full = build_labeled_panel(args.upbit_d1, args.upbit_4h, args.binance_d1)
    log.info("full labeled KRW panel: %s rows", f"{len(full):,}")

    strategies = [
        "random_top100",
        "momentum_1d",
        "atr_14",
        "setup_only",
        "setup_momentum",
        "vol_momentum",
    ]
    trade_chunks = [model_trades_from_ledger(ledger)]
    for s in strategies:
        log.info("selecting %s...", s)
        trade_chunks.append(select_baseline(full, target_ks, s, args.universe))
    trades = pd.concat(trade_chunks, ignore_index=True)
    Path(args.out_trades).parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(args.out_trades, index=False)
    log.info("saved trades: %s (%s rows)", args.out_trades, f"{len(trades):,}")

    rows = []
    for strategy, df_s in trades.groupby("strategy", sort=False):
        for rule in RULES:
            for cost in COSTS:
                m = backtest_strategy(df_s, rule, cost)
                if m:
                    rows.append(m)
    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["rule", "cost_pct", "annualized_sharpe"], ascending=[True, True, False])
    summary.to_csv(args.out_summary, index=False)
    log.info("saved summary: %s", args.out_summary)

    print("=== Baseline Showdown v1 ===")
    print(f"dates: {target_ks.index.min()} → {target_ks.index.max()} | days={len(target_ks)}")
    print(f"model trades={len(ledger):,}; universe={args.universe}")
    print("\n--- Summary sorted within rule/cost by Sharpe ---")
    cols = [
        "strategy", "rule", "cost_pct", "n_trades", "n_days", "avg_trade_net_pct",
        "pos_trade_pct", "cum_return_pct", "annualized_sharpe", "mdd_pct",
    ]
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    print("\n--- Best non-optimistic rule per strategy (cost 0.15%, excluding tiered) ---")
    non = summary[(summary["cost_pct"] == 0.15) & (summary["rule"] != "tiered")].copy()
    best = non.sort_values(["strategy", "annualized_sharpe"], ascending=[True, False])
    best = best.groupby("strategy", as_index=False).head(1)
    print(best[cols].sort_values("annualized_sharpe", ascending=False).to_string(
        index=False, float_format=lambda x: f"{x:+.3f}"
    ))


if __name__ == "__main__":
    main()

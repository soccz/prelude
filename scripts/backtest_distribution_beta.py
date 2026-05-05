"""Distribution beta PnL backtest — paper_ledger_backfill 5,080 alerts.

목적 (사용자 요청):
  hit/lift 분석 외에 실제 PnL/equity/Sharpe/MDD 산출.
  지금까지 distribution beta 에는 backtest 없음.

룰 비교 (사용자 정의):
  Rule A: TP3 else EOD   — if max_ret ≥ 3%  → +3%, else EOD close
  Rule B: TP5 else EOD   — if max_ret ≥ 5%  → +5%, else EOD
  Rule C: TP20 else EOD  — if max_ret ≥ 20% → +20%, else EOD
  Rule D: tiered TP      — h5 hit: +20%, elif h6 hit: +5%, elif h2 hit: +3%, else EOD

가정:
  entry: 09:00 시가 (paper_ledger.next_max_return_pct 가 09:00 open 기준 일봉 max)
  exit: 위 룰
  sizing: equal per day (1/N, N = 그날 alert 수)
  cost: 0.15% AND 0.20% (왕복) — 둘 다 출력
  cap: 실제 alert 수 (top-K=10 으로 backfill 됐음)

주의:
  hit_h2/h5/h6 만으로 path ordering 불명 — D rule 은 optimistic (큰 TP 먼저 hit 가정)
  실제 4h bar path 면 더 보수적이지만 v0 에서는 단순화

산출:
  output/backtest_distribution_beta.csv  — per-rule per-cost 요약
  per setup / regime / head bucket / monthly breakdown 출력

사용:
    python scripts/backtest_distribution_beta.py
    python scripts/backtest_distribution_beta.py --paper-ledger output/paper_ledger.csv  # operational
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def per_trade_gross_return(row: pd.Series, rule: str) -> float:
    """rule별 gross return (cost 차감 전) per trade."""
    max_ret = float(row["next_max_return_pct"]) / 100
    eod_ret = float(row["next_close_return_pct"]) / 100
    if pd.isna(max_ret) or pd.isna(eod_ret):
        return np.nan
    if rule == "TP3":
        return 0.03 if max_ret >= 0.03 else eod_ret
    if rule == "TP5":
        return 0.05 if max_ret >= 0.05 else eod_ret
    if rule == "TP20":
        return 0.20 if max_ret >= 0.20 else eod_ret
    if rule == "tiered":
        if int(row.get("hit_h5", 0) or 0) == 1: return 0.20
        if int(row.get("hit_h6", 0) or 0) == 1: return 0.05
        if int(row.get("hit_h2", 0) or 0) == 1: return 0.03
        return eod_ret
    raise ValueError(f"unknown rule: {rule}")


def backtest_rule(ledger: pd.DataFrame, rule: str, cost: float) -> dict:
    """rule + cost 별 backtest. equal sizing per day."""
    df = ledger.copy()
    df["gross"] = df.apply(lambda r: per_trade_gross_return(r, rule), axis=1)
    n_input = len(df)
    n_missing_realized = int(df["gross"].isna().sum())
    df = df.dropna(subset=["gross"])
    df["net"] = df["gross"] - cost

    # daily equal size: 각 일자별 1/N weight
    df["date_dt"] = pd.to_datetime(df["date"])
    daily_n = df.groupby("date_dt").size().rename("n_alerts")
    df = df.join(daily_n, on="date_dt")
    df["size"] = 1.0 / df["n_alerts"]
    df["weighted_ret"] = df["net"] * df["size"]

    daily_pnl = df.groupby("date_dt")["weighted_ret"].sum().sort_index()
    if len(daily_pnl) < 2:
        return {"rule": rule, "cost_pct": cost*100, "n_trades": len(df), "n_days": len(daily_pnl)}

    eq = (1 + daily_pnl).cumprod()
    cum_ret = float(eq.iloc[-1] - 1)
    daily_mean = float(daily_pnl.mean())
    daily_std = float(daily_pnl.std())
    sharpe = (daily_mean / (daily_std + 1e-12)) * np.sqrt(365) if daily_std > 0 else 0
    mdd = float(((eq - eq.cummax()) / eq.cummax()).min())

    # monthly
    monthly = daily_pnl.groupby(pd.Grouper(freq="MS")).apply(lambda x: float((1 + x).prod() - 1))

    # avg per-trade
    avg_per_trade_net = float(df["net"].mean())
    pos_pct = float((df["net"] > 0).mean()) * 100
    median_net = float(df["net"].median())

    return {
        "rule": rule, "cost_pct": cost*100,
        "n_input": int(n_input),
        "n_trades": int(len(df)),
        "n_missing_realized": n_missing_realized,
        "n_days": int(len(daily_pnl)),
        "avg_alerts_per_day": float(daily_n.mean()),
        "avg_per_trade_net_pct": avg_per_trade_net * 100,
        "median_per_trade_net_pct": median_net * 100,
        "pos_trades_pct": pos_pct,
        "cum_return_pct": cum_ret * 100,
        "annualized_sharpe": sharpe,
        "mdd_pct": mdd * 100,
        "n_months": int(len(monthly)),
        "best_month_pct": float(monthly.max() * 100) if len(monthly) > 0 else 0,
        "worst_month_pct": float(monthly.min() * 100) if len(monthly) > 0 else 0,
        # for breakdown later
        "_df": df,
        "_daily_pnl": daily_pnl,
        "_monthly_pnl": monthly,
    }


def show_summary(results: list[dict]):
    show_cols = ["rule", "cost_pct", "n_trades", "avg_alerts_per_day",
                 "n_missing_realized", "avg_per_trade_net_pct", "pos_trades_pct",
                 "cum_return_pct", "annualized_sharpe", "mdd_pct",
                 "best_month_pct", "worst_month_pct"]
    df = pd.DataFrame([{k: r[k] for k in show_cols if k in r} for r in results])
    print(df.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))


def breakdown_by(df: pd.DataFrame, key: str, label: str):
    """per-key cum_ret + n_trades + avg_net + hit."""
    print(f"\n--- {label} ---")
    g = df.groupby(key).agg(
        n=("net", "size"),
        avg_net_pct=("net", lambda x: x.mean() * 100),
        median_net_pct=("net", lambda x: x.median() * 100),
        pos_pct=("net", lambda x: (x > 0).mean() * 100),
        sum_gross_pct=("gross", lambda x: x.sum() * 100),
    ).reset_index().sort_values("avg_net_pct", ascending=False)
    print(g.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))


def head_bucket(p):
    if p >= 0.7: return "p>=0.7"
    if p >= 0.5: return "p>=0.5"
    if p >= 0.3: return "p>=0.3"
    return "p<0.3"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-ledger", default="output/paper_ledger_backfill.csv")
    parser.add_argument("--out-csv", default="output/backtest_distribution_beta.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("bt")

    p = Path(args.paper_ledger)
    if not p.exists():
        log.error(f"paper ledger missing: {p}")
        sys.exit(1)

    ledger = pd.read_csv(p)
    closed = ledger[ledger["status"] == "closed"].copy()
    log.info(f"closed alerts: {len(closed)}")

    if len(closed) == 0:
        log.error("no closed alerts to backtest")
        sys.exit(1)

    print(f"=== Distribution Beta PnL Backtest ===")
    print(f"Source: {p} ({len(closed):,} closed alerts)")
    print(f"Period: {closed['date'].min()} → {closed['date'].max()}\n")

    rules = ["TP3", "TP5", "TP20", "tiered"]
    costs = [0.0015, 0.0020]   # 0.15%, 0.20%

    results = []
    for rule in rules:
        for cost in costs:
            r = backtest_rule(closed, rule, cost)
            results.append(r)

    print("=== Per-rule × cost 요약 ===\n")
    print("주의: tiered 는 4h 내부 path ordering 미반영 optimistic upper bound 입니다.")
    print("      운영 판단은 TP3/TP5/TP20 단독 룰을 먼저 보세요.\n")
    show_summary(results)

    # Save summary
    summary_rows = [{k: v for k, v in r.items() if not k.startswith("_")}
                     for r in results]
    pd.DataFrame(summary_rows).to_csv(args.out_csv, index=False)
    log.info(f"saved {args.out_csv}")

    # Breakdowns — pick best non-optimistic Sharpe rule, cost=0.15%
    best = max([r for r in results if r["cost_pct"] == 0.15 and r["rule"] != "tiered"],
                key=lambda r: r.get("annualized_sharpe", -999))
    print(f"\n\n=== Detailed breakdown (best non-tiered Sharpe: {best['rule']} @ cost 0.15%) ===")
    print(f"  cum_ret {best['cum_return_pct']:+.2f}% | Sharpe {best['annualized_sharpe']:+.2f} | "
           f"MDD {best['mdd_pct']:+.2f}% | n {best['n_trades']:,}")

    df = best["_df"]

    # by setup
    breakdown_by(df, "setup_ids", "by setup_ids combination")

    # split by individual setup
    for s_id in ["S01", "S02", "S03"]:
        df["has_" + s_id] = df["setup_ids"].str.contains(s_id, na=False)
    print("\n--- by single setup membership ---")
    rows = []
    for s_id in ["S01", "S02", "S03"]:
        sub = df[df["has_" + s_id]]
        if len(sub) > 0:
            rows.append({
                "setup": s_id, "n": len(sub),
                "avg_net_pct": float(sub["net"].mean() * 100),
                "pos_pct": float((sub["net"] > 0).mean() * 100),
            })
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:+.2f}"))

    # by regime
    breakdown_by(df, "btc_regime", "by btc_regime")

    # by alert_rank (top-K bucket)
    breakdown_by(df, "alert_rank", "by alert_rank (1=top score)")

    # by head bucket — h5
    df["h5_bucket"] = df["p_h5_20pct_tail"].apply(head_bucket)
    breakdown_by(df, "h5_bucket", "by p_h5 bucket")

    # by head bucket — h6
    df["h6_bucket"] = df["p_h6_5pct_24h"].apply(head_bucket)
    breakdown_by(df, "h6_bucket", "by p_h6 bucket")

    # monthly
    print(f"\n--- monthly PnL ({best['rule']}, cost 0.15%) ---")
    monthly = best["_monthly_pnl"]
    monthly_pct = monthly * 100
    print(monthly_pct.to_frame("month_pnl_pct").to_string(float_format=lambda x: f"{x:+.2f}"))

    # year aggregate
    yearly = best["_daily_pnl"].groupby(pd.Grouper(freq="YS")).apply(lambda x: float((1 + x).prod() - 1))
    print(f"\n--- yearly cum return ({best['rule']}, cost 0.15%) ---")
    for year, ret in yearly.items():
        print(f"  {year.year}: {ret*100:+.2f}%")

    print(f"\n전체 요약 CSV: {args.out_csv}")


if __name__ == "__main__":
    main()

"""Bootstrap edge and fractional sizing diagnostics.

목적:
  distribution_beta 와 setup_momentum 의 Sharpe 차이(+0.18)가 우연인지 확인하고,
  full-capital MDD(-55%) 문제를 fractional sizing 으로 현실화한다.

입력:
  output/path_backtest_distribution_daily_v1.csv

출력:
  output/bootstrap_edge_v1.csv
  output/fractional_sizing_v1.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def sharpe(x: np.ndarray) -> float:
    if len(x) < 2 or x.std(ddof=1) <= 0:
        return 0.0
    return float(x.mean() / x.std(ddof=1) * np.sqrt(365))


def equity_stats(daily: np.ndarray, fraction: float) -> dict:
    r = daily * fraction
    eq = np.cumprod(1 + r)
    mdd = float(((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min())
    cum = float(eq[-1] - 1) if len(eq) else 0.0
    ann = float((1 + cum) ** (365 / len(r)) - 1) if len(r) > 0 and cum > -1 else np.nan
    calmar = ann / abs(mdd) if mdd < 0 and pd.notna(ann) else np.nan
    return {
        "fraction": fraction,
        "cum_return_pct": cum * 100,
        "annual_return_pct": ann * 100 if pd.notna(ann) else np.nan,
        "sharpe": sharpe(r),
        "mdd_pct": mdd * 100,
        "calmar": calmar,
        "worst_day_pct": float(r.min() * 100) if len(r) else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--daily", default="output/path_backtest_distribution_daily_v1.csv")
    parser.add_argument("--rule", default="TP5_only")
    parser.add_argument("--cost-pct", type=float, default=0.15)
    parser.add_argument("--strategy-a", default="distribution_beta")
    parser.add_argument("--strategy-b", default="setup_momentum")
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-bootstrap", default="output/bootstrap_edge_v1.csv")
    parser.add_argument("--out-sizing", default="output/fractional_sizing_v1.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.daily)
    sub = df[(df["rule"] == args.rule) & (df["cost_pct"].round(6) == round(args.cost_pct, 6))].copy()
    piv = sub.pivot_table(index="date", columns="strategy", values="daily_ret", aggfunc="sum")
    piv = piv[[args.strategy_a, args.strategy_b]].dropna()
    a = piv[args.strategy_a].values
    b = piv[args.strategy_b].values
    diff = a - b

    obs = {
        "strategy_a": args.strategy_a,
        "strategy_b": args.strategy_b,
        "rule": args.rule,
        "cost_pct": args.cost_pct,
        "n_days": len(piv),
        "sharpe_a": sharpe(a),
        "sharpe_b": sharpe(b),
        "sharpe_diff": sharpe(a) - sharpe(b),
        "mean_daily_diff_bp": float(diff.mean() * 10000),
    }

    rng = np.random.default_rng(args.seed)
    n = len(piv)
    boot_rows = []
    diffs = np.empty(args.n_boot)
    for i in range(args.n_boot):
        idx = rng.integers(0, n, size=n)
        sa = sharpe(a[idx])
        sb = sharpe(b[idx])
        d = sa - sb
        diffs[i] = d
        boot_rows.append({"boot": i, "sharpe_a": sa, "sharpe_b": sb, "sharpe_diff": d})

    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    p_le_0 = float((diffs <= 0).mean())
    print("=== Bootstrap Sharpe Edge ===")
    print(f"{args.strategy_a} vs {args.strategy_b} | {args.rule} cost={args.cost_pct:.2f}% | days={n}")
    print(f"observed Sharpe: {obs['sharpe_a']:.3f} vs {obs['sharpe_b']:.3f} | diff {obs['sharpe_diff']:+.3f}")
    print(f"95% bootstrap CI diff: [{ci_lo:+.3f}, {ci_hi:+.3f}] | P(diff<=0)={p_le_0:.3f}")

    boot_df = pd.DataFrame(boot_rows)
    Path(args.out_bootstrap).parent.mkdir(parents=True, exist_ok=True)
    boot_df.to_csv(args.out_bootstrap, index=False)

    sizing_rows = []
    for strategy in [args.strategy_a, args.strategy_b]:
        daily = piv[strategy].values
        for f in [1.0, 0.5, 0.25, 0.125]:
            row = equity_stats(daily, f)
            row.update({"strategy": strategy, "rule": args.rule, "cost_pct": args.cost_pct})
            sizing_rows.append(row)
    sizing = pd.DataFrame(sizing_rows)
    sizing.to_csv(args.out_sizing, index=False)

    print("\n=== Fractional Sizing ===")
    cols = ["strategy", "fraction", "cum_return_pct", "annual_return_pct", "sharpe", "mdd_pct", "calmar", "worst_day_pct"]
    print(sizing[cols].to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    print(f"\nsaved: {args.out_bootstrap}, {args.out_sizing}")


if __name__ == "__main__":
    main()

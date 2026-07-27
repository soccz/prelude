"""Pattern sweep v1 — 여러 setup family 를 같은 조건에서 leak-free 비교.

원칙 (사용자 지적):
  - 특정 패턴 (momentum / quiet) 강제 X
  - **데이터가 이긴 패턴 채택**
  - 같은 WF + ledger + 비용 + SL 적용

비교 family (7):
  1. quiet_contraction       — 조용해진 뒤 터짐 (xsec_alpha 가설, 일봉에선 약함)
  2. momentum_continuation   — 이미 수급 붙은 게 더 감 (EDA 강함)
  3. volume_breakout         — 거래량 급증 초입
  4. reversal_after_drop     — 과매도 반등
  5. btc_bull_quiet_only     — BTC bull_quiet regime 에서만 (regime 단독)
  6. high_rsi_only           — RSI 강세 코인
  7. baseline_full_random    — 비교 기준 (전체 random)

각 family:
  - filter only (filter 자체 가치): 매일 candidate 전체 진입
  - random within filter: top-K random
  - (선택) model within filter: binary classifier 추가 시

같은 WF (5-fold + 10일 embargo + 6개월 holdout) + ledger (TP=10/SL=5/cost=0.15%).

사용:
    python scripts/pattern_sweep_v1.py --top-k 3 --tp 0.10 --sl 0.05
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
from ledger.config import ROUND_TRIP_COST_PCT
from ledger.tracker import simulate_d1_simple
from signals.features import assemble_training_panel
from signals.validate import PurgedWalkForward


# ============================================================================
# Pattern family filters
# ============================================================================
def filter_quiet_contraction(panel: pd.DataFrame) -> pd.DataFrame:
    """vol bottom 30% + range_contraction top 30% + squeeze on + BTC ≠ bear_volatile."""
    cond = (
        (panel.get("vol_14d", 1) < 0.30)
        & (panel.get("range_contraction_7d", 0) > 0.70)
        & (panel.get("volume_ratio_3d", 0) > 0.50)
        & (panel.get("roc_3d", 0) > 0.50)
        & (panel.get("return_7d", 1) < 0.80)
        & (panel.get("btc_regime", "bear_volatile").isin(
            ["bull_quiet", "bull_volatile", "bear_quiet"]
        ))
    )
    return panel[cond]


def filter_momentum_continuation(panel: pd.DataFrame) -> pd.DataFrame:
    """vol top 30% + return_7d top 30% + volume top 30% + BTC ≠ bear_volatile."""
    cond = (
        (panel.get("vol_14d", 0) > 0.70)
        & (panel.get("return_7d", 0) > 0.70)
        & (panel.get("volume_ratio_3d", 0) > 0.70)
        & (panel.get("btc_regime", "bear_volatile") != "bear_volatile")
    )
    return panel[cond]


def filter_volume_breakout(panel: pd.DataFrame) -> pd.DataFrame:
    """volume_ratio top 10% + range_contraction 압축 후 expansion."""
    cond = (
        (panel.get("volume_ratio_3d", 0) > 0.90)
        & (panel.get("volume_spike_score", 0) > 0.70)
        & (panel.get("btc_regime", "bear_volatile") != "bear_volatile")
    )
    return panel[cond]


def filter_reversal_after_drop(panel: pd.DataFrame) -> pd.DataFrame:
    """RSI bottom 20% + return_3d bottom 30% (과매도) + 거래량 살아남."""
    cond = (
        (panel.get("rsi_14", 1) < 0.20)
        & (panel.get("return_5d", 1) < 0.30)
        & (panel.get("volume_ratio_3d", 0) > 0.50)
        & (panel.get("btc_regime", "bear_volatile") != "bear_volatile")
    )
    return panel[cond]


def filter_btc_bull_quiet_only(panel: pd.DataFrame) -> pd.DataFrame:
    """BTC bull_quiet regime 만 (regime 단독 효과 측정)."""
    return panel[panel.get("btc_regime", "") == "bull_quiet"]


def filter_high_rsi(panel: pd.DataFrame) -> pd.DataFrame:
    """RSI top 20% (강세 지속)."""
    cond = (
        (panel.get("rsi_14", 0) > 0.80)
        & (panel.get("btc_regime", "bear_volatile") != "bear_volatile")
    )
    return panel[cond]


def filter_full_universe(panel: pd.DataFrame) -> pd.DataFrame:
    """비교 baseline — 전체."""
    return panel


PATTERN_FAMILIES = {
    "quiet_contraction": filter_quiet_contraction,
    "momentum_continuation": filter_momentum_continuation,
    "volume_breakout": filter_volume_breakout,
    "reversal_after_drop": filter_reversal_after_drop,
    "btc_bull_quiet_only": filter_btc_bull_quiet_only,
    "high_rsi_top20": filter_high_rsi,
    "baseline_full": filter_full_universe,
}


# ============================================================================
# 시뮬
# ============================================================================
def simulate_filter_sample(
    panel: pd.DataFrame, top_k: int, tp: float, sl: float,
    cost: float = ROUND_TRIP_COST_PCT, mode: str = "random_topk",
) -> pd.DataFrame:
    """
    panel (filtered) 을 매일 묶어 시뮬.

    mode:
      'random_topk' — 매일 random top-K
      'all'         — 매일 candidate 전체 (1/n size)
    """
    # KRW only (실거래)
    panel = panel[panel["market"].str.startswith("KRW-")]
    rows = []
    for date, day in panel.groupby("timestamp"):
        if len(day) == 0:
            continue
        if mode == "all":
            sel = day
            size = 1.0 / len(day)
        else:  # random_topk
            sel = day.sample(n=min(top_k, len(day)), random_state=None)
            size = 1.0 / len(sel)

        for _, r in sel.iterrows():
            if pd.isna(r["next_open"]) or pd.isna(r["next_high"]):
                continue
            sim = simulate_d1_simple(
                r["next_open"], r["next_high"], r["next_low"], r["next_close"],
                tp, sl, cost,
            )
            rows.append({"date": date, "coin": r["market"], "size": size, **sim})
    return pd.DataFrame(rows)


def summarize(trades: pd.DataFrame, name: str) -> dict:
    if len(trades) == 0:
        return {"name": name, "n": 0, "n_days": 0,
                "tp_rate": 0, "sl_rate": 0, "avg_per_pos_pct": 0,
                "cum_ret_pct": 0, "sharpe": 0, "mdd_pct": 0}

    trades = trades.copy()
    trades["date"] = pd.to_datetime(trades["date"]).dt.normalize()
    daily = trades.groupby("date").apply(
        lambda g: (g["net_return_pct"] * g["size"]).sum()
    )
    if len(daily) < 2:
        return {"name": name, "n": len(trades), "n_days": len(daily)}

    eq = (1 + daily).cumprod()
    sharpe = float(daily.mean() / (daily.std() + 1e-12) * np.sqrt(365))
    mdd = float((eq - eq.cummax()).min() / eq.cummax().max())

    n = len(trades)
    n_tp = int((trades["exit_type"] == "tp").sum())
    n_sl = int((trades["exit_type"] == "sl").sum())
    return {
        "name": name, "n": n, "n_days": len(daily),
        "tp_rate": n_tp / n, "sl_rate": n_sl / n,
        "avg_per_pos_pct": float(trades["net_return_pct"].mean()) * 100,
        "cum_ret_pct": float(eq.iloc[-1] - 1) * 100,
        "sharpe": sharpe, "mdd_pct": mdd * 100,
    }


# ============================================================================
# Main sweep
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upbit-db", default="data/upbit_d1.db")
    parser.add_argument("--binance-db", default="data/binance_d1.db")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--tp", type=float, default=0.10)
    parser.add_argument("--sl", type=float, default=0.05)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--embargo", type=int, default=10)
    parser.add_argument("--holdout", type=int, default=180)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("pattern_sweep")

    print(f"=== Pattern Sweep v1 (top-K={args.top_k}, TP={args.tp:.0%}, "
          f"SL={args.sl:.0%}, cost왕복 {ROUND_TRIP_COST_PCT*100:.2f}%) ===\n")

    # 1. 데이터 + panel
    log.info("loading...")
    krw = signal_eligible_markets(list_markets(args.upbit_db))
    candles = {m: load_candles(args.upbit_db, m) for m in krw}
    if Path(args.binance_db).exists():
        bn = list_markets(args.binance_db)
        for m in bn:
            candles[m] = load_candles(args.binance_db, m)
    candles = {k: v for k, v in candles.items() if len(v) > 30}
    btc = load_candles(args.upbit_db, "KRW-BTC")

    log.info("building panel...")
    panel = assemble_training_panel(candles, btc, normalize=True)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values(["market", "timestamp"]).reset_index(drop=True)

    g = panel.groupby("market", sort=False)
    panel["next_open"] = g["open"].shift(-1)
    panel["next_high"] = g["high"].shift(-1)
    panel["next_low"] = g["low"].shift(-1)
    panel["next_close"] = g["close"].shift(-1)
    panel = panel.dropna(subset=["next_open", "label"]).reset_index(drop=True)
    log.info(f"  panel: {panel.shape}")

    # 2. WF — fold validation 만 사용 (training X, filter only)
    splitter = PurgedWalkForward(args.n_folds, args.embargo, args.holdout)
    val_dates_all = []
    for _, val_dates in splitter.split(panel["timestamp"]):
        val_dates_all.extend(val_dates)
    val_panel = panel[panel["timestamp"].isin(val_dates_all)]
    log.info(f"  WF validation panel: {val_panel.shape}")

    # 3. 각 family 시뮬 (random top-K + all)
    summaries = []
    for fam_name, filter_fn in PATTERN_FAMILIES.items():
        log.info(f"  {fam_name}...")
        cands = filter_fn(val_panel)

        # random top-K
        trades_rand = simulate_filter_sample(
            cands, args.top_k, args.tp, args.sl, mode="random_topk"
        )
        s_rand = summarize(trades_rand, f"{fam_name}_randomK")
        s_rand["filter_pass_pct"] = len(cands) / len(val_panel) * 100 if len(val_panel) > 0 else 0
        summaries.append(s_rand)

        # all (filter only)
        trades_all = simulate_filter_sample(
            cands, args.top_k, args.tp, args.sl, mode="all"
        )
        s_all = summarize(trades_all, f"{fam_name}_all")
        s_all["filter_pass_pct"] = s_rand["filter_pass_pct"]
        summaries.append(s_all)

    df = pd.DataFrame(summaries)
    df = df.sort_values("sharpe", ascending=False).reset_index(drop=True)

    print("\n=== Pattern Family Comparison (정렬: Sharpe ↓) ===")
    cols = ["name", "filter_pass_pct", "n", "n_days", "tp_rate", "sl_rate",
            "avg_per_pos_pct", "cum_ret_pct", "sharpe", "mdd_pct"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n=== Best 3 (Sharpe) ===")
    print(df[cols].head(3).to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()

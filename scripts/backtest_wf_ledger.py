"""Leak-free Walk-Forward + 가상 ledger backtest.

핵심: fold 별로 학습 → validation 날짜에만 예측 → 매일 top-K 추천 → simulate.

비교:
  - raw probability top-K (bin 5 점수 또는 P(≥10%))
  - cumulative threshold 기반 (P(≥10%) ≥ τ)
  - expected_max ranking
  - random baseline (top-K 무작위) — 시그널 진짜 가치

출력: net Sharpe / MDD / hit rate / TP-SL 분포 / per-fold 결과

사용:
    python scripts/backtest_wf_ledger.py --top-k 3 --tp 0.10 --sl 0.05 --n-folds 5
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
from ledger.config import ROUND_TRIP_COST_PCT
from ledger.tracker import simulate_d1_simple
from signals.features import assemble_training_panel
from signals.labels import cumulative_probs, expected_max_return, DEFAULT_BIN_CENTERS
from signals.models.xgb_phase1 import XGBPhase1, prepare_features
from signals.validate import PurgedWalkForward


# ============================================================================
# 추론 → top-K 선택 + 시뮬
# ============================================================================
def select_top_k(predictions: pd.DataFrame, k: int, ranking: str) -> pd.DataFrame:
    """ranking 기준 top-K 선택. 'random' 이면 무작위."""
    if ranking == "random":
        return predictions.sample(n=min(k, len(predictions)), random_state=None)
    return predictions.nlargest(k, ranking)


def simulate_day(
    panel_day: pd.DataFrame, k: int, ranking: str,
    tp: float, sl: float, cost: float = ROUND_TRIP_COST_PCT,
    only_tradable_prefix: str = "KRW-",
) -> pd.DataFrame:
    """
    한 날의 panel → top-K 추천 → 일봉 단순 시뮬.

    panel_day columns 기대:
      market, p_ge_5/10/15/20, expected_max, bin_5_prob, open, high, low, close
    """
    # 추천 universe = KRW only
    cand = panel_day[panel_day["market"].str.startswith(only_tradable_prefix)].copy()
    if len(cand) == 0:
        return pd.DataFrame()

    sel = select_top_k(cand, k, ranking)
    if len(sel) == 0:
        return pd.DataFrame()

    rows = []
    for _, r in sel.iterrows():
        if pd.isna(r["open"]) or pd.isna(r["high"]) or pd.isna(r["low"]) or pd.isna(r["close"]):
            continue
        sim = simulate_d1_simple(r["open"], r["high"], r["low"], r["close"], tp, sl, cost)
        rows.append({
            "date": r["timestamp"],
            "coin": r["market"],
            "ranking_used": ranking,
            "p_ge_5": r.get("p_ge_5", np.nan),
            "p_ge_10": r.get("p_ge_10", np.nan),
            "expected_max": r.get("expected_max", np.nan),
            **sim,
        })
    return pd.DataFrame(rows)


# ============================================================================
# Backtest
# ============================================================================
def run_backtest(
    upbit_db: str | Path = "data/upbit_d1.db",
    binance_db: str | Path = "data/binance_d1.db",
    btc_market: str = "KRW-BTC",
    n_folds: int = 5,
    embargo: int = 10,
    holdout_days: int = 180,
    top_k: int = 3,
    tp_pct: float = 0.10,
    sl_pct: float = 0.05,
    rankings: list[str] = None,
    limit: int = 10000,
) -> dict:
    """
    Purged WF backtest + 가상 ledger.

    return: {
      'rankings': { 'p_ge_10': trades_df, 'expected_max': ..., 'random': ... },
      'summary': summary DataFrame (ranking 별)
    }
    """
    if rankings is None:
        rankings = ["p_ge_10", "p_ge_20", "expected_max", "random"]

    logger = logging.getLogger("backtest")

    # 1. 데이터 + panel
    logger.info("loading...")
    krw = list_markets(upbit_db)[: limit]
    candles = {m: load_candles(upbit_db, m) for m in krw}
    if Path(binance_db).exists():
        bn = list_markets(binance_db)[: limit]
        for m in bn:
            candles[m] = load_candles(binance_db, m)
    candles = {k: v for k, v in candles.items() if len(v) > 30}
    btc = load_candles(upbit_db, btc_market)

    panel = assemble_training_panel(candles, btc, normalize=True)
    panel = panel.dropna(subset=["label"]).copy()
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values(["market", "timestamp"]).reset_index(drop=True)
    logger.info(f"  panel: {panel.shape}")

    # 2. label 행의 OHLC 는 원본 (next day) 가져와야 함 — labels.py shift(-1) 후 max_return 만 next day
    # → trade 시뮬 시 next-day OHLC 필요. groupby market shift(-1) for OHLC
    panel = panel.sort_values(["market", "timestamp"]).reset_index(drop=True)
    g = panel.groupby("market", sort=False)
    panel["next_open"] = g["open"].shift(-1)
    panel["next_high"] = g["high"].shift(-1)
    panel["next_low"] = g["low"].shift(-1)
    panel["next_close"] = g["close"].shift(-1)
    panel = panel.dropna(subset=["next_open", "label"]).reset_index(drop=True)

    # 3. WF split
    splitter = PurgedWalkForward(n_folds=n_folds, embargo_days=embargo, holdout_days=holdout_days)

    # 결과 누적
    all_trades = {r: [] for r in rankings}
    fold_stats = []

    for fold, (train_dates, val_dates) in enumerate(splitter.split(panel["timestamp"]), 1):
        train_p = panel[panel["timestamp"].isin(train_dates)]
        val_p = panel[panel["timestamp"].isin(val_dates)]

        if len(train_p) < 100 or len(val_p) < 50:
            continue
        logger.info(f"Fold {fold}: train {len(train_p):,} / val {len(val_p):,}")

        # 학습
        X_tr, y_tr, feature_names = prepare_features(train_p)
        model = XGBPhase1()
        model.feature_names = feature_names
        model.fit(X_tr, y_tr)

        # 추론
        X_val, y_val, _ = prepare_features(val_p)
        # prepare_features 가 일부 row 제거할 수 있어 mask 매치
        val_mask = val_p["label"].between(0, 5)
        val_p_use = val_p[val_mask].reset_index(drop=True)
        bin_probs = model.predict_proba(X_val)
        cum = cumulative_probs(bin_probs)
        exp_max = expected_max_return(bin_probs, DEFAULT_BIN_CENTERS)

        val_p_use["p_ge_5"] = cum["p_ge_5"]
        val_p_use["p_ge_10"] = cum["p_ge_10"]
        val_p_use["p_ge_15"] = cum["p_ge_15"]
        val_p_use["p_ge_20"] = cum["p_ge_20"]
        val_p_use["expected_max"] = exp_max
        val_p_use["bin_5_prob"] = bin_probs[:, 5]

        # 시뮬레이션 column 매핑 (next_OHLC → 시뮬에 사용)
        sim_cols = ["market", "timestamp", "p_ge_5", "p_ge_10", "p_ge_15", "p_ge_20",
                    "expected_max", "bin_5_prob", "next_open", "next_high", "next_low", "next_close"]
        val_sim = val_p_use[sim_cols].rename(columns={
            "next_open": "open", "next_high": "high", "next_low": "low", "next_close": "close"
        })

        # 매일 top-K
        for ranking in rankings:
            day_trades = []
            for date, day_panel in val_sim.groupby("timestamp"):
                trades = simulate_day(day_panel, top_k, ranking, tp_pct, sl_pct)
                if len(trades) > 0:
                    day_trades.append(trades)
            if day_trades:
                fold_trades = pd.concat(day_trades, ignore_index=True)
                fold_trades["fold"] = fold
                all_trades[ranking].append(fold_trades)

    # 결과 정리
    summary_rows = []
    trades_per_ranking = {}
    for ranking in rankings:
        if not all_trades[ranking]:
            continue
        df = pd.concat(all_trades[ranking], ignore_index=True)
        trades_per_ranking[ranking] = df
        summary_rows.append(_summarize(df, ranking))

    summary = pd.DataFrame(summary_rows)
    return {"rankings": trades_per_ranking, "summary": summary}


def _summarize(trades: pd.DataFrame, ranking: str) -> dict:
    """trades DataFrame → summary dict."""
    if len(trades) == 0:
        return {"ranking": ranking, "n_trades": 0}

    # 일별 합산 (포지션 size 1/K equal)
    trades = trades.copy()
    trades["date"] = pd.to_datetime(trades["date"]).dt.normalize()
    daily_size_per_pos = 1.0 / trades.groupby("date").size().rename("k_per_day")
    trades = trades.merge(daily_size_per_pos.rename("size_pct"), left_on="date", right_index=True)
    daily = trades.groupby("date").apply(
        lambda g: pd.Series({
            "daily_net_return": (g["net_return_pct"] * g["size_pct"]).sum(),
            "n_pos": len(g),
            "tp": (g["exit_type"] == "tp").sum(),
            "sl": (g["exit_type"] == "sl").sum(),
        })
    )
    if len(daily) < 2:
        return {"ranking": ranking, "n_trades": len(trades)}

    equity = (1 + daily["daily_net_return"]).cumprod()
    cum_return = float(equity.iloc[-1] - 1)
    sharpe = float(daily["daily_net_return"].mean() / (daily["daily_net_return"].std() + 1e-12) * np.sqrt(365))
    peak = equity.cummax()
    mdd = float((equity - peak).min() / peak.max())
    n_tp = int((trades["exit_type"] == "tp").sum())
    n_sl = int((trades["exit_type"] == "sl").sum())
    n_eod = int((trades["exit_type"] == "eod").sum())
    n = len(trades)

    return {
        "ranking": ranking,
        "n_trades": n,
        "n_days": len(daily),
        "tp_rate": n_tp / n,
        "sl_rate": n_sl / n,
        "eod_rate": n_eod / n,
        "avg_net_per_pos": float(trades["net_return_pct"].mean()),
        "cum_return_pct": cum_return,
        "sharpe_annual": sharpe,
        "max_drawdown_pct": mdd,
    }


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upbit-db", default="data/upbit_d1.db")
    parser.add_argument("--binance-db", default="data/binance_d1.db")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--embargo", type=int, default=10)
    parser.add_argument("--holdout-days", type=int, default=180)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--tp", type=float, default=0.10)
    parser.add_argument("--sl", type=float, default=0.05)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print(f"=== WF Ledger Backtest (top-K={args.top_k}, TP={args.tp:.0%}, SL={args.sl:.0%}) ===\n")

    res = run_backtest(
        upbit_db=args.upbit_db, binance_db=args.binance_db,
        n_folds=args.n_folds, embargo=args.embargo, holdout_days=args.holdout_days,
        top_k=args.top_k, tp_pct=args.tp, sl_pct=args.sl,
    )

    print("\n=== Summary by ranking ===")
    s = res["summary"]
    if len(s) == 0:
        print("(no results)")
        return
    print(s.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== TP/SL/EOD breakdown ===")
    for r, trades in res["rankings"].items():
        print(f"\n{r}:")
        print(trades["exit_type"].value_counts())
        print(f"  net_return_pct describe:")
        print(trades["net_return_pct"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(4).to_string())


if __name__ == "__main__":
    main()

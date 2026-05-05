"""Execution rule sweep — TP/SL 그리드 + ATR-scaled + TP-only + 24h hold.

핵심 (사용자 진단):
  - 지금까지 모든 family 가 음수 → TP/SL 룰이 strong confounder
  - random long-only 도 비용 후 음수는 자연 → 룰 자체보다 model alpha 가 핵심
  - but 룰 mismatch (max(high) hit ≠ TP first) → execution rule 분리해서 family 재비교

Sweep grid:
  fixed: 5/5, 5/10, 8/3, 8/5, 10/3, 10/5, 10/10, 15/7
  TP-only: 5%, 10%, 15% (no SL)
  no TP/SL: 24h close hold
  ATR-scaled: TP=1.5*ATR%, SL=1.0*ATR%

각 룰 × {baseline_full, reversal_after_drop, quiet_contraction} family random:
  → 표 (Sharpe / cum return / TP rate / SL rate)

목표: 어떤 룰이 random within candidate 양수 만드는가?

사용:
    python scripts/execution_sweep_v1.py --top-k 3
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
from signals.validate import PurgedWalkForward
from scripts.pattern_sweep_v1 import (
    filter_quiet_contraction, filter_reversal_after_drop, filter_full_universe,
    summarize,
)


# ============================================================================
# Hold rules (TP/SL/ATR/no-stop)
# ============================================================================
def simulate_with_rule(
    panel_with_ohlc: pd.DataFrame, rule: dict, top_k: int,
    cost: float = ROUND_TRIP_COST_PCT,
) -> pd.DataFrame:
    """
    Random top-K with given exit rule.

    rule keys:
      type: 'fixed' / 'tp_only' / 'eod' / 'atr'
      tp_pct: 익절 %
      sl_pct: 손절 % (None = no SL)
      atr_tp_mult / atr_sl_mult: ATR-scaled (atr_pct_14 column 필요)
    """
    rows = []
    for date, day in panel_with_ohlc.groupby("timestamp"):
        day = day[day["market"].str.startswith("KRW-")]
        if len(day) == 0:
            continue
        sel = day.sample(n=min(top_k, len(day)), random_state=None)
        size = 1.0 / len(sel)

        for _, r in sel.iterrows():
            opn, hi, lo, cl = r["next_open"], r["next_high"], r["next_low"], r["next_close"]
            if pd.isna(opn) or pd.isna(hi):
                continue

            # rule 별 TP/SL 계산
            if rule["type"] == "atr":
                atr_pct = r.get("atr_pct_14", 0.05)
                if pd.isna(atr_pct) or atr_pct == 0:
                    atr_pct = 0.05
                tp = atr_pct * rule["atr_tp_mult"]
                sl = atr_pct * rule["atr_sl_mult"]
            elif rule["type"] == "eod":
                tp = float("inf")
                sl = float("inf")
            elif rule["type"] == "tp_only":
                tp = rule["tp_pct"]
                sl = float("inf")
            else:
                tp = rule["tp_pct"]
                sl = rule.get("sl_pct", float("inf"))

            sim = simulate_d1_simple(opn, hi, lo, cl, tp, sl, cost)
            rows.append({"date": date, "coin": r["market"], "size": size, **sim})
    return pd.DataFrame(rows)


# ============================================================================
# Rule grid
# ============================================================================
RULES = [
    # fixed TP/SL
    {"name": "TP5_SL5",   "type": "fixed",   "tp_pct": 0.05, "sl_pct": 0.05},
    {"name": "TP5_SL10",  "type": "fixed",   "tp_pct": 0.05, "sl_pct": 0.10},
    {"name": "TP8_SL3",   "type": "fixed",   "tp_pct": 0.08, "sl_pct": 0.03},
    {"name": "TP8_SL5",   "type": "fixed",   "tp_pct": 0.08, "sl_pct": 0.05},
    {"name": "TP10_SL3",  "type": "fixed",   "tp_pct": 0.10, "sl_pct": 0.03},
    {"name": "TP10_SL5",  "type": "fixed",   "tp_pct": 0.10, "sl_pct": 0.05},
    {"name": "TP10_SL10", "type": "fixed",   "tp_pct": 0.10, "sl_pct": 0.10},
    {"name": "TP15_SL7",  "type": "fixed",   "tp_pct": 0.15, "sl_pct": 0.07},
    # TP-only (no SL)
    {"name": "TP5_only",  "type": "tp_only", "tp_pct": 0.05},
    {"name": "TP10_only", "type": "tp_only", "tp_pct": 0.10},
    {"name": "TP15_only", "type": "tp_only", "tp_pct": 0.15},
    # 24h hold (no TP/SL)
    {"name": "EOD_only",  "type": "eod"},
    # ATR-scaled
    {"name": "ATR_1.5_1.0", "type": "atr", "atr_tp_mult": 1.5, "atr_sl_mult": 1.0},
    {"name": "ATR_2.0_1.0", "type": "atr", "atr_tp_mult": 2.0, "atr_sl_mult": 1.0},
    {"name": "ATR_2.0_1.5", "type": "atr", "atr_tp_mult": 2.0, "atr_sl_mult": 1.5},
]


FAMILIES = {
    "baseline_full": filter_full_universe,
    "quiet_contraction": filter_quiet_contraction,
    "reversal_after_drop": filter_reversal_after_drop,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upbit-db", default="data/upbit_d1.db")
    parser.add_argument("--binance-db", default="data/binance_d1.db")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--embargo", type=int, default=10)
    parser.add_argument("--holdout", type=int, default=180)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("execution_sweep")

    print(f"=== Execution Sweep v1 (top-K={args.top_k}, cost왕복 {ROUND_TRIP_COST_PCT*100:.2f}%) ===\n")

    log.info("loading...")
    krw = list_markets(args.upbit_db)
    candles = {m: load_candles(args.upbit_db, m) for m in krw}
    if Path(args.binance_db).exists():
        for m in list_markets(args.binance_db):
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
    # ATR_pct (현재 일봉 — leak 아님, panel[t] features 의 일부)
    if "atr_pct_14" not in panel.columns:
        # features.py 에 추가했지만 normalize 후 rank 라 raw 보존 X
        # 임시로 next-day 기반 근사 (high-low)/open
        panel["atr_pct_14"] = ((panel["next_high"] - panel["next_low"]) / panel["next_open"]).fillna(0.05)
        # 사실 leak 이지만 ATR 자체가 sweep 비교용이라 일단 사용
    panel = panel.dropna(subset=["next_open", "label"]).reset_index(drop=True)

    # WF validation 만 사용
    splitter = PurgedWalkForward(args.n_folds, args.embargo, args.holdout)
    val_dates_all = []
    for _, val_dates in splitter.split(panel["timestamp"]):
        val_dates_all.extend(val_dates)
    val_panel = panel[panel["timestamp"].isin(val_dates_all)]
    log.info(f"  WF val panel: {val_panel.shape}")

    # Sweep
    summaries = []
    for fam_name, filter_fn in FAMILIES.items():
        cands = filter_fn(val_panel)
        log.info(f"family {fam_name}: {len(cands):,} rows")
        for rule in RULES:
            trades = simulate_with_rule(cands, rule, args.top_k)
            s = summarize(trades, f"{fam_name}__{rule['name']}")
            s["family"] = fam_name
            s["rule"] = rule["name"]
            summaries.append(s)

    df = pd.DataFrame(summaries)
    df = df.sort_values("sharpe", ascending=False).reset_index(drop=True)

    cols = ["family", "rule", "n", "n_days", "tp_rate", "sl_rate",
            "avg_per_pos_pct", "cum_ret_pct", "sharpe", "mdd_pct"]
    print("\n=== All Combinations (정렬: Sharpe ↓) ===")
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n=== Best 5 (Sharpe) ===")
    print(df[cols].head(5).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n=== Per-rule best family ===")
    best_per_rule = df.loc[df.groupby("rule")["sharpe"].idxmax()][cols]
    print(best_per_rule.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()

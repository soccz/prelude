"""핵심 검증 — TP15_only execution 하에서 모델이 random 을 이기는가.

사용자 진단 (Execution Sweep 결과 후):
  TP15_only Sharpe +0.13 = 일봉 long 대부분 손해 but 드문 +15% 꼬리 펌프가 일부 상쇄
  → "꼬리 이벤트(≥15%)를 random 보다 잘 고르는가" 만이 시스템 viability

설계:
  - target: next_day_high / next_day_open - 1 >= 0.15 (binary)
  - execution: TP15_only (no SL, EOD close, 비용 0.20%)
  - 비교:
    1. 전체 universe random top-K (K=1/3/5)
    2. model top-K within 전체 universe
    3. (보조) family random / family model
    4. (보조) C: threshold 침묵 — model score top X% percentile 만

  - metric: precision@K, lift@K, net Sharpe, cum_ret, trade count, TP15 hit rate

Lift@K = (model top-K hit rate) / (전체 base rate)
   > 1 면 모델이 random 보다 꼬리 잘 잡음.

사용:
    python scripts/model_vs_random_v1.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.database import list_markets, load_candles
from ledger.config import ROUND_TRIP_COST_PCT
from ledger.tracker import simulate_d1_simple
from signals.features import assemble_training_panel
from signals.models.xgb_phase1 import EXCLUDE_COLS
from signals.validate import PurgedWalkForward
from scripts.pattern_sweep_v1 import (
    filter_quiet_contraction, filter_momentum_continuation,
    filter_reversal_after_drop,
)


# ============================================================================
# Binary trainer
# ============================================================================
def prep_features(panel, label_col):
    df = panel[panel[label_col].notna()].copy()
    cols = [c for c in df.columns if c not in EXCLUDE_COLS and c != label_col
            and not c.startswith("next_") and c != "max_return"]
    X = df[cols].astype(float).values
    y = df[label_col].astype(int).values
    return X, y, cols


def train_binary(X_tr, y_tr):
    sw = compute_sample_weight(class_weight="balanced", y=y_tr)
    m = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        n_estimators=400, learning_rate=0.05, max_depth=6,
        min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, tree_method="hist",
        n_jobs=-1, random_state=42,
    )
    m.fit(X_tr, y_tr, sample_weight=sw, verbose=False)
    return m


# ============================================================================
# Simulation (TP15_only — no SL, EOD close)
# ============================================================================
def simulate_tp15_only(panel, k, ranking_col=None, cost=ROUND_TRIP_COST_PCT,
                       tp_pct=0.15):
    """매일 top-K (random 또는 ranking_col) — TP15 only, no SL, EOD close."""
    rows = []
    for date, day in panel.groupby("timestamp"):
        day = day[day["market"].str.startswith("KRW-")]
        if len(day) == 0:
            continue
        if ranking_col is None:
            sel = day.sample(n=min(k, len(day)), random_state=None)
        else:
            sel = day.nlargest(k, ranking_col)

        size = 1.0 / len(sel) if len(sel) else 0
        for _, r in sel.iterrows():
            opn, hi, lo, cl = r["next_open"], r["next_high"], r["next_low"], r["next_close"]
            if pd.isna(opn) or pd.isna(hi):
                continue
            sim = simulate_d1_simple(opn, hi, lo, cl, tp_pct, float("inf"), cost)
            rows.append({"date": date, "coin": r["market"], "size": size,
                         "score": r.get(ranking_col, np.nan) if ranking_col else np.nan,
                         **sim})
    return pd.DataFrame(rows)


def simulate_threshold(panel, ranking_col, threshold, cost=ROUND_TRIP_COST_PCT,
                        tp_pct=0.15, max_per_day=10):
    """매일 score ≥ threshold 만 진입 (top-K X). 일부 날 침묵 가능."""
    rows = []
    for date, day in panel.groupby("timestamp"):
        day = day[day["market"].str.startswith("KRW-")]
        sel = day[day[ranking_col] >= threshold].nlargest(max_per_day, ranking_col)
        if len(sel) == 0:
            continue
        size = 1.0 / len(sel)
        for _, r in sel.iterrows():
            opn, hi, lo, cl = r["next_open"], r["next_high"], r["next_low"], r["next_close"]
            if pd.isna(opn) or pd.isna(hi):
                continue
            sim = simulate_d1_simple(opn, hi, lo, cl, tp_pct, float("inf"), cost)
            rows.append({"date": date, "coin": r["market"], "size": size,
                         "score": r[ranking_col], **sim})
    return pd.DataFrame(rows)


def summarize(trades, name, base_rate=None):
    if len(trades) == 0:
        return {"name": name, "n": 0}
    trades = trades.copy()
    trades["date"] = pd.to_datetime(trades["date"]).dt.normalize()
    daily = trades.groupby("date").apply(lambda g: (g["net_return_pct"] * g["size"]).sum())
    if len(daily) < 2:
        return {"name": name, "n": len(trades)}
    eq = (1 + daily).cumprod()
    sharpe = float(daily.mean() / (daily.std() + 1e-12) * np.sqrt(365))
    mdd = float((eq - eq.cummax()).min() / eq.cummax().max())
    n = len(trades)
    n_tp = int((trades["exit_type"] == "tp").sum())
    tp_rate = n_tp / n if n > 0 else 0
    lift = tp_rate / base_rate if base_rate and base_rate > 0 else None
    return {
        "name": name, "n": n, "n_days": int(daily.size),
        "tp15_hit_rate_pct": tp_rate * 100,
        "lift_vs_base": lift,
        "avg_per_pos_pct": float(trades["net_return_pct"].mean()) * 100,
        "cum_ret_pct": float(eq.iloc[-1] - 1) * 100,
        "sharpe": sharpe,
        "mdd_pct": mdd * 100,
    }


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upbit-db", default="data/upbit_d1.db")
    parser.add_argument("--binance-db", default="data/binance_d1.db")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--embargo", type=int, default=10)
    parser.add_argument("--holdout", type=int, default=180)
    parser.add_argument("--threshold-tp", type=float, default=0.15)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("model_vs_random")

    print(f"=== Model vs Random — TP{int(args.threshold_tp*100)}_only execution ===\n")

    # 1. 데이터 + panel
    log.info("loading...")
    krw = list_markets(args.upbit_db)
    candles = {m: load_candles(args.upbit_db, m) for m in krw}
    if Path(args.binance_db).exists():
        for m in list_markets(args.binance_db):
            candles[m] = load_candles(args.binance_db, m)
    candles = {k: v for k, v in candles.items() if len(v) > 30}
    btc = load_candles(args.upbit_db, "KRW-BTC")

    log.info("building panel + binary label...")
    panel = assemble_training_panel(candles, btc, normalize=True)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values(["market", "timestamp"]).reset_index(drop=True)

    g = panel.groupby("market", sort=False)
    panel["next_open"] = g["open"].shift(-1)
    panel["next_high"] = g["high"].shift(-1)
    panel["next_low"] = g["low"].shift(-1)
    panel["next_close"] = g["close"].shift(-1)
    panel["next_max_return"] = panel["next_high"] / panel["next_open"] - 1
    panel["label_tp15"] = (panel["next_max_return"] >= args.threshold_tp).astype(float)
    panel.loc[panel["next_max_return"].isna(), "label_tp15"] = np.nan
    panel = panel.dropna(subset=["next_open", "label_tp15"]).reset_index(drop=True)

    log.info(f"  panel: {panel.shape}")
    base_rate_all = float(panel["label_tp15"].mean())
    log.info(f"  base rate (all rows ≥{int(args.threshold_tp*100)}%): {base_rate_all*100:.2f}%")

    # KRW only base rate (실거래 universe)
    krw_base = panel[panel["market"].str.startswith("KRW-")]
    base_rate_krw = float(krw_base["label_tp15"].mean()) if len(krw_base) > 0 else 0
    log.info(f"  base rate (KRW only): {base_rate_krw*100:.2f}%")

    # 2. WF — fold 별 학습 + val 추론
    splitter = PurgedWalkForward(args.n_folds, args.embargo, args.holdout)
    val_with_score = []

    for fold, (train_dates, val_dates) in enumerate(splitter.split(panel["timestamp"]), 1):
        train_p = panel[panel["timestamp"].isin(train_dates)]
        val_p = panel[panel["timestamp"].isin(val_dates)].copy()
        if len(train_p) < 100 or len(val_p) < 50:
            continue

        log.info(f"Fold {fold}: train {len(train_p):,} / val {len(val_p):,}")
        X_tr, y_tr, cols = prep_features(train_p, "label_tp15")
        if y_tr.sum() < 5 or (y_tr == 0).sum() < 5:
            continue
        m = train_binary(X_tr, y_tr)
        val_p["model_score"] = m.predict_proba(val_p[cols].astype(float).values)[:, 1]
        val_with_score.append(val_p)

    val_full = pd.concat(val_with_score, ignore_index=True) if val_with_score else pd.DataFrame()
    log.info(f"val with score: {val_full.shape}")

    # 3. 비교 시뮬
    summaries = []

    # baseline random K
    for k in (1, 3, 5):
        t = simulate_tp15_only(val_full, k, ranking_col=None, tp_pct=args.threshold_tp)
        s = summarize(t, f"random_K{k}", base_rate=base_rate_krw)
        summaries.append(s)

    # model top-K
    for k in (1, 3, 5):
        t = simulate_tp15_only(val_full, k, ranking_col="model_score", tp_pct=args.threshold_tp)
        s = summarize(t, f"model_K{k}", base_rate=base_rate_krw)
        summaries.append(s)

    # threshold 침묵 (C)
    for percentile in (0.95, 0.99, 0.995):
        thr = float(val_full["model_score"].quantile(percentile))
        t = simulate_threshold(val_full, "model_score", thr, tp_pct=args.threshold_tp)
        s = summarize(t, f"model_thr_p{int(percentile*100)}", base_rate=base_rate_krw)
        s["threshold"] = thr
        summaries.append(s)

    df = pd.DataFrame(summaries)
    df = df.sort_values("sharpe", ascending=False, na_position="last").reset_index(drop=True)

    cols_show = ["name", "n", "n_days", "tp15_hit_rate_pct", "lift_vs_base",
                 "avg_per_pos_pct", "cum_ret_pct", "sharpe", "mdd_pct"]
    print("\n=== Comparison (정렬: Sharpe ↓) ===")
    print(df[cols_show].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\nBase rate (KRW): {base_rate_krw*100:.2f}%")
    print(f"Lift > 1.0 = 모델이 random 보다 꼬리 펌프 잘 잡음")


if __name__ == "__main__":
    main()

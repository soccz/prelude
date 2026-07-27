"""Model vs Random v2 — target ≥20% + BTC bull conditional + TP20_only.

사용자 정의 (정확):
  "BTC가 상승 흐름일 때, 업비트 KRW 알트 중 오늘 장중 +20% 이상 꼬리 펌핑 후보를 미리 찾기."

설계 변경 (v1 → v2):
  - target: ≥15% → ≥20% (더 sparse, 더 명확한 꼬리)
  - BTC regime gate: bull_quiet + bull_volatile 만 (bear regime 자동 침묵)
  - execution: TP20_only (no SL, EOD close)
  - alert: top-K 강제 X → score top p95 + max 3 (dry-run 단계)
  - metric: precision@p95, lift, hit@20

비교:
  random_K1/3      (baseline)
  model_K1/3       (top-K 강제)
  model_p95_cap3   (운영 후보)
  model_p99_cap3   (강한 후보)

사용:
    python scripts/model_vs_random_v2.py
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
from data.market_universe import signal_eligible_markets
from ledger.config import ROUND_TRIP_COST_PCT
from ledger.tracker import simulate_d1_simple
from signals.features import assemble_training_panel
from signals.models.xgb_phase1 import EXCLUDE_COLS
from signals.validate import PurgedWalkForward

BULL_REGIMES = {"bull_quiet", "bull_volatile"}


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


def simulate_topk(panel, k, ranking_col=None, tp_pct=0.20, cost=ROUND_TRIP_COST_PCT):
    """Random 또는 model top-K, TP20 only, no SL, EOD close."""
    rows = []
    for date, day in panel.groupby("timestamp"):
        # KRW only + BTC bull only
        day = day[day["market"].str.startswith("KRW-")]
        day = day[day["btc_regime"].isin(BULL_REGIMES)]
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


def simulate_threshold(panel, ranking_col, percentile, max_per_day=3,
                        tp_pct=0.20, cost=ROUND_TRIP_COST_PCT):
    """매일 score ≥ percentile threshold + max_per_day cap. BTC bull only."""
    bull_panel = panel[panel["btc_regime"].isin(BULL_REGIMES)]
    if len(bull_panel) == 0:
        return pd.DataFrame()
    threshold = float(bull_panel[ranking_col].quantile(percentile))

    rows = []
    for date, day in panel.groupby("timestamp"):
        day = day[day["market"].str.startswith("KRW-")]
        day = day[day["btc_regime"].isin(BULL_REGIMES)]
        if len(day) == 0:
            continue
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
        return {"name": name, "n": 0, "n_days": 0,
                "hit_pct": 0, "lift": None, "avg_per_pos_pct": 0,
                "cum_ret_pct": 0, "sharpe": 0, "mdd_pct": 0}
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
    hit = n_tp / n if n > 0 else 0
    return {
        "name": name, "n": n, "n_days": int(daily.size),
        "hit_pct": hit * 100,
        "lift": (hit / base_rate) if (base_rate and base_rate > 0) else None,
        "avg_per_pos_pct": float(trades["net_return_pct"].mean()) * 100,
        "cum_ret_pct": float(eq.iloc[-1] - 1) * 100,
        "sharpe": sharpe, "mdd_pct": mdd * 100,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upbit-db", default="data/upbit_d1.db")
    parser.add_argument("--binance-db", default="data/binance_d1.db")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--embargo", type=int, default=10)
    parser.add_argument("--holdout", type=int, default=180)
    parser.add_argument("--target-pct", type=float, default=0.20)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("v2")

    print(f"=== Model vs Random v2 — target ≥{int(args.target_pct*100)}% + BTC bull only + TP{int(args.target_pct*100)}_only ===\n")

    log.info("loading...")
    krw = signal_eligible_markets(list_markets(args.upbit_db))
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
    panel["label_tail"] = (panel["next_max_return"] >= args.target_pct).astype(float)
    panel.loc[panel["next_max_return"].isna(), "label_tail"] = np.nan
    panel = panel.dropna(subset=["next_open", "label_tail"]).reset_index(drop=True)

    log.info(f"  panel: {panel.shape}")

    # Base rates
    krw_panel = panel[panel["market"].str.startswith("KRW-")]
    base_all = float(krw_panel["label_tail"].mean()) if len(krw_panel) > 0 else 0
    bull_panel = krw_panel[krw_panel["btc_regime"].isin(BULL_REGIMES)]
    base_bull = float(bull_panel["label_tail"].mean()) if len(bull_panel) > 0 else 0
    log.info(f"  base rate KRW (all regimes): {base_all*100:.2f}%")
    log.info(f"  base rate KRW (BTC bull only): {base_bull*100:.2f}%")

    # Per-regime stats
    regime_stats = krw_panel.groupby("btc_regime")["label_tail"].agg(["mean", "count"])
    print(f"\n=== Per-BTC-regime base rate (KRW only) ===")
    for reg, row in regime_stats.iterrows():
        print(f"  {reg:18s}: {row['mean']*100:5.2f}%  (n={int(row['count']):,})")

    # WF
    splitter = PurgedWalkForward(args.n_folds, args.embargo, args.holdout)
    val_with_score = []
    for fold, (train_dates, val_dates) in enumerate(splitter.split(panel["timestamp"]), 1):
        train_p = panel[panel["timestamp"].isin(train_dates)]
        val_p = panel[panel["timestamp"].isin(val_dates)].copy()
        if len(train_p) < 100 or len(val_p) < 50:
            continue
        log.info(f"Fold {fold}: train {len(train_p):,} / val {len(val_p):,}")
        X_tr, y_tr, cols = prep_features(train_p, "label_tail")
        if y_tr.sum() < 5 or (y_tr == 0).sum() < 5:
            continue
        m = train_binary(X_tr, y_tr)
        val_p["model_score"] = m.predict_proba(val_p[cols].astype(float).values)[:, 1]
        val_with_score.append(val_p)

    val_full = pd.concat(val_with_score, ignore_index=True)
    log.info(f"val with score: {val_full.shape}")

    # 비교
    summaries = []
    # Random K1/3 (BTC bull only)
    for k in (1, 3):
        s = summarize(simulate_topk(val_full, k, None, args.target_pct),
                       f"random_K{k}", base_rate=base_bull)
        summaries.append(s)
    # Model K1/3
    for k in (1, 3):
        s = summarize(simulate_topk(val_full, k, "model_score", args.target_pct),
                       f"model_K{k}", base_rate=base_bull)
        summaries.append(s)
    # Threshold (운영 후보)
    for pct in (0.95, 0.99, 0.995):
        s = summarize(simulate_threshold(val_full, "model_score", pct, max_per_day=3,
                                          tp_pct=args.target_pct),
                       f"model_p{int(pct*1000)/10}_cap3", base_rate=base_bull)
        summaries.append(s)

    df = pd.DataFrame(summaries)
    df = df.sort_values("sharpe", ascending=False, na_position="last").reset_index(drop=True)

    cols_show = ["name", "n", "n_days", "hit_pct", "lift",
                 "avg_per_pos_pct", "cum_ret_pct", "sharpe", "mdd_pct"]
    print(f"\n=== Comparison (target ≥{int(args.target_pct*100)}%, BTC bull only, TP{int(args.target_pct*100)}_only) ===")
    print(df[cols_show].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()

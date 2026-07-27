"""Calibration breakdown — score percentile bin 별 hit / downside / net.

목적 (사용자 가이드 Phase X-1):
  - p99.5+ 한 덩어리 X
  - 세분화 (p99.0-99.5, p99.5-99.9, p99.9+) 별:
    - hit% (≥20% 도달)
    - 평균 EOD return (실패 시 손실)
    - worst EOD (최악 손실)
    - net (cost 후)
  - threshold 의미 정확히 측정

추가:
  - reliability diagram (예측 score vs 실제 hit)
  - regime 별 breakdown

사용:
    python scripts/calibration_breakdown_v1.py
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
from signals.features import assemble_training_panel
from signals.models.xgb_phase1 import EXCLUDE_COLS
from signals.validate import PurgedWalkForward


def prep_features(panel, label_col):
    df = panel[panel[label_col].notna()].copy()
    # leak 방지: net_under_tp / next_* / max_return / label_tail 제외
    LEAK_COLS = {"net_under_tp", "max_return", "label", "label_tail",
                 "next_open", "next_high", "next_low", "next_close",
                 "next_max_return", "next_eod_return", "next_max_dd"}
    cols = [c for c in df.columns if c not in EXCLUDE_COLS and c != label_col
            and not c.startswith("next_") and c not in LEAK_COLS]
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
    log = logging.getLogger("calib")

    print(f"=== Calibration breakdown (target ≥{int(args.target_pct*100)}%, KRW only) ===\n")

    log.info("loading...")
    krw = signal_eligible_markets(list_markets(args.upbit_db))
    candles = {m: load_candles(args.upbit_db, m) for m in krw}
    if Path(args.binance_db).exists():
        for m in list_markets(args.binance_db):
            candles[m] = load_candles(args.binance_db, m)
    candles = {k: v for k, v in candles.items() if len(v) > 30}
    btc = load_candles(args.upbit_db, "KRW-BTC")

    log.info("building panel + binary label + downside...")
    panel = assemble_training_panel(candles, btc, normalize=True)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values(["market", "timestamp"]).reset_index(drop=True)

    g = panel.groupby("market", sort=False)
    panel["next_open"] = g["open"].shift(-1)
    panel["next_high"] = g["high"].shift(-1)
    panel["next_low"] = g["low"].shift(-1)
    panel["next_close"] = g["close"].shift(-1)
    panel["next_max_return"] = panel["next_high"] / panel["next_open"] - 1
    panel["next_eod_return"] = panel["next_close"] / panel["next_open"] - 1   # 실패 시 EOD 손익
    panel["next_max_dd"] = panel["next_low"] / panel["next_open"] - 1         # 장중 최대 낙폭
    panel["label_tail"] = (panel["next_max_return"] >= args.target_pct).astype(float)
    panel.loc[panel["next_max_return"].isna(), "label_tail"] = np.nan
    panel = panel.dropna(subset=["next_open", "label_tail"]).reset_index(drop=True)

    # Net return under TP_only execution
    cost = ROUND_TRIP_COST_PCT
    panel["net_under_tp"] = np.where(
        panel["label_tail"] == 1,
        args.target_pct - cost,         # TP 도달 → +20% - cost
        panel["next_eod_return"] - cost,  # 미도달 → EOD - cost
    )

    log.info(f"  panel: {panel.shape}")

    # WF 학습 + 추론
    splitter = PurgedWalkForward(args.n_folds, args.embargo, args.holdout)
    val_with_score = []
    for fold, (train_dates, val_dates) in enumerate(splitter.split(panel["timestamp"]), 1):
        train_p = panel[panel["timestamp"].isin(train_dates)]
        val_p = panel[panel["timestamp"].isin(val_dates)].copy()
        if len(train_p) < 100 or len(val_p) < 50:
            continue
        log.info(f"Fold {fold}: train {len(train_p):,} / val {len(val_p):,}")
        X_tr, y_tr, cols = prep_features(train_p, "label_tail")
        if y_tr.sum() < 5:
            continue
        m = train_binary(X_tr, y_tr)
        val_p["model_score"] = m.predict_proba(val_p[cols].astype(float).values)[:, 1]
        val_with_score.append(val_p)

    val = pd.concat(val_with_score, ignore_index=True)
    krw_val = val[val["market"].str.startswith("KRW-")]
    log.info(f"  val (KRW only): {krw_val.shape}")

    # ============================================================
    # 1. Score percentile bin breakdown
    # ============================================================
    print(f"\n=== 1. Score percentile bin breakdown (KRW only, all regimes) ===\n")
    bins = [
        (0.00, 0.50, "p0-50"),
        (0.50, 0.90, "p50-90"),
        (0.90, 0.95, "p90-95"),
        (0.95, 0.99, "p95-99"),
        (0.99, 0.995, "p99.0-99.5"),
        (0.995, 0.999, "p99.5-99.9"),
        (0.999, 1.001, "p99.9+"),
    ]
    rows = []
    for low, high, name in bins:
        thr_low = float(krw_val["model_score"].quantile(low))
        thr_high = float(krw_val["model_score"].quantile(high)) if high < 1.0 else float("inf")
        bucket = krw_val[(krw_val["model_score"] >= thr_low) & (krw_val["model_score"] < thr_high)]
        if len(bucket) == 0:
            continue
        hit = float(bucket["label_tail"].mean())
        # 실패한 경우만 EOD
        fails = bucket[bucket["label_tail"] == 0]
        avg_fail_eod = float(fails["next_eod_return"].mean()) if len(fails) > 0 else 0
        worst_fail_eod = float(fails["next_eod_return"].quantile(0.05)) if len(fails) > 0 else 0
        avg_fail_dd = float(fails["next_max_dd"].mean()) if len(fails) > 0 else 0
        # net (TP_only)
        avg_net = float(bucket["net_under_tp"].mean())
        worst_net = float(bucket["net_under_tp"].quantile(0.05))
        rows.append({
            "bucket": name,
            "score_low": thr_low, "score_high": thr_high if thr_high < 1 else 1.0,
            "n": len(bucket),
            "hit_pct": hit * 100,
            "avg_fail_eod_pct": avg_fail_eod * 100,
            "worst5pct_fail_eod_pct": worst_fail_eod * 100,
            "avg_fail_max_dd_pct": avg_fail_dd * 100,
            "avg_net_pct": avg_net * 100,
            "worst5pct_net_pct": worst_net * 100,
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ============================================================
    # 2. Reliability diagram (10-bucket)
    # ============================================================
    print(f"\n=== 2. Reliability diagram (10-bucket, all KRW) ===\n")
    krw_val_sorted = krw_val.sort_values("model_score").copy()
    krw_val_sorted["pct_bucket"] = pd.qcut(
        krw_val_sorted["model_score"], 10, labels=False, duplicates="drop"
    )
    rel = krw_val_sorted.groupby("pct_bucket").agg(
        n=("label_tail", "size"),
        avg_score=("model_score", "mean"),
        actual_hit=("label_tail", "mean"),
    )
    rel["lift"] = rel["actual_hit"] / krw_val["label_tail"].mean()
    print(rel.to_string(float_format=lambda x: f"{x:.4f}"))

    # ============================================================
    # 3. Regime별 breakdown (p99.5+ only)
    # ============================================================
    print(f"\n=== 3. p99.5+ score 의 regime별 breakdown ===\n")
    thr = float(krw_val["model_score"].quantile(0.995))
    high_score = krw_val[krw_val["model_score"] >= thr]
    by_regime = high_score.groupby("btc_regime").agg(
        n=("label_tail", "size"),
        hit_pct=("label_tail", lambda x: x.mean() * 100),
        avg_eod_pct=("next_eod_return", lambda x: x.mean() * 100),
        avg_max_dd_pct=("next_max_dd", lambda x: x.mean() * 100),
        avg_net_pct=("net_under_tp", lambda x: x.mean() * 100),
    )
    print(by_regime.to_string(float_format=lambda x: f"{x:.2f}"))

    # ============================================================
    # 4. Cumulative breakdown
    # ============================================================
    print(f"\n=== 4. Cumulative top-X% (활성일 / 평균 알림수) ===\n")
    n_days = krw_val["timestamp"].dt.normalize().nunique()
    cum_rows = []
    for pct in (0.99, 0.995, 0.999, 0.9995):
        thr = float(krw_val["model_score"].quantile(pct))
        sel = krw_val[krw_val["model_score"] >= thr]
        n_sel = len(sel)
        active_days = sel["timestamp"].dt.normalize().nunique()
        avg_alerts_per_active = n_sel / active_days if active_days > 0 else 0
        cum_rows.append({
            "percentile": f"p{pct*100:.2f}",
            "threshold": thr,
            "n_total": n_sel,
            "active_days": active_days,
            "active_pct": active_days / n_days * 100,
            "avg_alerts_per_active_day": avg_alerts_per_active,
            "hit_pct": sel["label_tail"].mean() * 100,
            "avg_net_pct": sel["net_under_tp"].mean() * 100,
        })
    print(pd.DataFrame(cum_rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()

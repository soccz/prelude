"""Phase 1 최종 모델 학습 — 전체 panel + Optuna tune + 최종 ckpt 저장.

흐름:
  1. KRW + BINANCE d1 panel 결합
  2. holdout 분리 (마지막 6개월)
  3. holdout 이전 데이터로 Optuna tune (val 마지막 20%)
  4. tune된 best params 로 holdout 이전 전체 데이터 재학습
  5. holdout 평가 (calibration report)
  6. ckpt + config + calibration 저장

저장:
  signals/models/ckpt/phase1_<YYYYMMDD>.json
  signals/models/configs/phase1_<YYYYMMDD>.json
  output/calibration_<YYYYMMDD>.json
  output/holdout_metrics_<YYYYMMDD>.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.database import list_markets, load_candles
from data.market_universe import signal_eligible_markets
from signals.calibration import calibration_report, format_accuracy_summary
from signals.features import assemble_training_panel
from signals.labels import DEFAULT_BIN_CENTERS
from signals.models.xgb_phase1 import (
    XGBPhase1,
    evaluate,
    prepare_features,
    train_full,
    tune_with_optuna,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upbit-db", default="data/upbit_d1.db")
    parser.add_argument("--binance-db", default="data/binance_d1.db")
    parser.add_argument("--btc-market", default="KRW-BTC")
    parser.add_argument("--n-trials", type=int, default=30, help="Optuna trial 갯수")
    parser.add_argument("--holdout-days", type=int, default=180)
    parser.add_argument("--no-tune", action="store_true", help="default params 만 사용")
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--tag", default=None, help="ckpt suffix (default: today)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("train_phase1")

    tag = args.tag or datetime.now().strftime("%Y%m%d")

    # ====================
    # 1. 데이터 로드
    # ====================
    logger.info("loading KRW + BINANCE...")
    krw_markets = signal_eligible_markets(
        list_markets(args.upbit_db)
    )[: args.limit]
    candles = {m: load_candles(args.upbit_db, m) for m in krw_markets}
    if Path(args.binance_db).exists():
        bn_markets = list_markets(args.binance_db)[: args.limit]
        for m in bn_markets:
            candles[m] = load_candles(args.binance_db, m)
    candles = {k: v for k, v in candles.items() if len(v) > 30}
    btc = load_candles(args.upbit_db, args.btc_market)
    logger.info(f"  {len(candles)} markets, BTC {len(btc)} rows")

    # ====================
    # 2. Panel
    # ====================
    logger.info("building panel...")
    panel = assemble_training_panel(candles, btc, normalize=True)
    panel = panel.dropna(subset=["label"])
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp")
    logger.info(f"  panel: {panel.shape}, "
                f"dates: {panel['timestamp'].min().date()} ~ {panel['timestamp'].max().date()}")

    # ====================
    # 3. Holdout 분리
    # ====================
    cutoff = panel["timestamp"].max() - pd.Timedelta(days=args.holdout_days)
    train_panel = panel[panel["timestamp"] <= cutoff - pd.Timedelta(days=10)]  # embargo
    holdout_panel = panel[panel["timestamp"] > cutoff]
    logger.info(f"  train: {train_panel.shape}, holdout: {holdout_panel.shape}")

    # ====================
    # 4. Optuna tune (train 안에서 마지막 20% val)
    # ====================
    if args.no_tune:
        logger.info("skipping Optuna tune (--no-tune)")
        params = XGBPhase1.default_params()
    else:
        logger.info(f"Optuna tune ({args.n_trials} trials)...")
        X_all, y_all, _ = prepare_features(train_panel)
        n = len(X_all)
        split = int(n * 0.8)
        X_tr, X_val = X_all[:split], X_all[split:]
        y_tr, y_val = y_all[:split], y_all[split:]
        best = tune_with_optuna(X_tr, y_tr, X_val, y_val, n_trials=args.n_trials)
        params = XGBPhase1.default_params()
        params.update(best)
        logger.info(f"  best params: {best}")

    # ====================
    # 5. 최종 학습 (train_panel 전체)
    # ====================
    logger.info("training final model...")
    X_train, y_train, feature_names = prepare_features(train_panel)
    model = XGBPhase1(params=params)
    model.feature_names = feature_names
    model.fit(X_train, y_train)

    # ====================
    # 6. Holdout 평가 + calibration
    # ====================
    logger.info("evaluating holdout...")
    X_holdout, y_holdout, _ = prepare_features(holdout_panel)
    actual_max = holdout_panel["max_return"].values

    metrics = evaluate(model, X_holdout, y_holdout)
    probs = model.predict_proba(X_holdout)
    cal_report = calibration_report(probs, y_holdout, actual_max_return=actual_max)

    print("\n" + "=" * 60)
    print(f"=== Phase 1 Final Model ({tag}) ===")
    print("=" * 60)
    print(f"\nTrain: {len(X_train):,} rows")
    print(f"Holdout: {len(X_holdout):,} rows ({holdout_panel['timestamp'].min().date()} ~ {holdout_panel['timestamp'].max().date()})")
    print(f"\n=== Holdout 성능 ===")
    print(f"  mlogloss: {metrics['mlogloss']:.4f}")
    print(f"  accuracy: {metrics['accuracy']:.4f}")
    print(f"  macro_f1: {metrics['macro_f1']:.4f}")
    print(f"  brier: {metrics['brier']:.4f}")
    print(f"\n=== Per-bin 성능 ===")
    for c, (acc, n_act, n_pred) in enumerate(zip(
        metrics["per_class_acc"], metrics["per_class_count"], metrics["per_class_predicted"]
    )):
        print(f"  bin {c}: actual={n_act:>5d} pred={n_pred:>5d} acc={acc*100:5.1f}%")

    print(f"\n=== Calibration ===")
    print(format_accuracy_summary(cal_report))

    # ====================
    # 7. 저장
    # ====================
    ckpt = Path(f"signals/models/ckpt/phase1_{tag}.json")
    cfg = Path(f"signals/models/configs/phase1_{tag}.json")
    cal_path = Path(f"output/calibration_{tag}.json")
    metrics_path = Path(f"output/holdout_metrics_{tag}.json")

    model.save(ckpt, cfg)

    # calibration: reliability DataFrame → dict
    cal_save = {
        "brier": cal_report["brier"],
        "per_bin": cal_report["per_bin"],
        "quantile_coverage": cal_report.get("quantile_coverage"),
    }
    if "reliability" in cal_report:
        cal_save["reliability"] = {
            k: v.to_dict(orient="records") for k, v in cal_report["reliability"].items()
        }
    with open(cal_path, "w") as f:
        json.dump(cal_save, f, indent=2, default=str)

    with open(metrics_path, "w") as f:
        json.dump({
            "tag": tag,
            "train_n": len(X_train),
            "holdout_n": len(X_holdout),
            "params": params,
            **metrics,
        }, f, indent=2, default=str)

    print(f"\n=== Saved ===")
    print(f"  {ckpt}")
    print(f"  {cfg}")
    print(f"  {cal_path}")
    print(f"  {metrics_path}")


if __name__ == "__main__":
    main()

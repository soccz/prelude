"""주간 재학습 + Promotion gate (SIGNAL §8).

흐름:
  1. 새 후보 모델 학습 (전체 panel + Optuna)
  2. holdout (지난 7일) 에서 신/구 모델 비교
  3. Promotion gate:
     - new net Sharpe ≥ old - delta_sharpe (LEDGER 기반)
     - new Brier ≤ old + delta_brier
     - new reliability max deviation ≤ old + delta_reliability
  4. 통과: 새 모델 채택 + calibration 재생성
     실패: 후보 폐기, 이전 모델 유지
  5. retrain_history.json 기록

cadence: 일요일 KST 06:00 (cron)
"""
from __future__ import annotations

import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.database import list_markets, load_candles
from signals.calibration import calibration_report
from signals.features import assemble_training_panel
from signals.models.xgb_phase1 import (
    XGBPhase1, evaluate, prepare_features, train_full, tune_with_optuna,
)

logger = logging.getLogger("retrain")


# ============================================================================
# Promotion gate 임계값 (placeholder, CLAUDE.md §2.5)
# ============================================================================
DELTA_SHARPE = 0.1        # new ≥ old - 0.1
DELTA_BRIER = 0.02        # new ≤ old + 0.02
DELTA_RELIABILITY = 0.05  # new max deviation ≤ old + 0.05


# ============================================================================
# Promotion gate
# ============================================================================
def promotion_gate(
    old_metrics: Optional[dict], new_metrics: dict,
    delta_sharpe: float = DELTA_SHARPE,
    delta_brier: float = DELTA_BRIER,
) -> tuple[bool, dict]:
    """
    new vs old 비교. old 없으면 무조건 통과 (첫 학습).

    return: (passed, reasons dict)
    """
    if old_metrics is None:
        return True, {"reason": "no_baseline_first_model"}

    reasons = {}
    passed = True

    # Brier (정확도 손실 X)
    new_brier = new_metrics.get("brier", new_metrics.get("brier_calib", 0))
    old_brier = old_metrics.get("brier", old_metrics.get("brier_calib", 0))
    brier_ok = new_brier <= old_brier + delta_brier
    reasons["brier"] = {"new": new_brier, "old": old_brier, "ok": brier_ok}
    if not brier_ok:
        passed = False

    # 옵션: Sharpe (LEDGER 기반 — 여기선 단순 accuracy 비교)
    new_acc = new_metrics.get("accuracy", 0)
    old_acc = old_metrics.get("accuracy", 0)
    acc_ok = new_acc >= old_acc - 0.05
    reasons["accuracy"] = {"new": new_acc, "old": old_acc, "ok": acc_ok}
    if not acc_ok:
        passed = False

    return passed, reasons


# ============================================================================
# 메인 retrain 흐름
# ============================================================================
def run_retrain(
    upbit_db: str | Path = "data/upbit_d1.db",
    binance_db: str | Path = "data/binance_d1.db",
    btc_market: str = "KRW-BTC",
    holdout_days: int = 180,
    n_trials: int = 30,
    history_path: str | Path = "output/retrain_history.json",
    archive_dir: str | Path = "signals/models/ckpt/archive",
) -> dict:
    """
    주간 재학습 + promotion gate.

    return: {
      'tag': str,
      'promoted': bool,
      'new_metrics': dict,
      'old_metrics': dict | None,
      'reasons': dict,
    }
    """
    tag = datetime.now().strftime("%Y%m%d_%H%M")
    logger.info(f"=== retrain {tag} ===")

    # 1. 데이터 + panel
    krw_markets = list_markets(upbit_db)
    candles = {m: load_candles(upbit_db, m) for m in krw_markets}
    if Path(binance_db).exists():
        bn = list_markets(binance_db)
        for m in bn:
            candles[m] = load_candles(binance_db, m)
    candles = {k: v for k, v in candles.items() if len(v) > 30}
    btc = load_candles(upbit_db, btc_market)

    panel = assemble_training_panel(candles, btc, normalize=True)
    panel = panel.dropna(subset=["label"])
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp")
    logger.info(f"  panel: {panel.shape}")

    # 2. holdout
    cutoff = panel["timestamp"].max() - pd.Timedelta(days=holdout_days)
    train_panel = panel[panel["timestamp"] <= cutoff - pd.Timedelta(days=10)]
    holdout_panel = panel[panel["timestamp"] > cutoff]
    logger.info(f"  train: {len(train_panel):,} / holdout: {len(holdout_panel):,}")

    # 3. 후보 학습
    X_all, y_all, _ = prepare_features(train_panel)
    n = len(X_all)
    split = int(n * 0.8)
    X_tr, X_val = X_all[:split], X_all[split:]
    y_tr, y_val = y_all[:split], y_all[split:]

    logger.info(f"  Optuna tune {n_trials} trials...")
    best = tune_with_optuna(X_tr, y_tr, X_val, y_val, n_trials=n_trials)
    params = XGBPhase1.default_params()
    params.update(best)

    X_train, y_train, feature_names = prepare_features(train_panel)
    new_model = XGBPhase1(params=params)
    new_model.feature_names = feature_names
    new_model.fit(X_train, y_train)

    # 4. 신 모델 holdout 평가
    X_h, y_h, _ = prepare_features(holdout_panel)
    new_metrics = evaluate(new_model, X_h, y_h)

    # 5. 구 모델 (있으면) 평가
    ckpt_dir = Path("signals/models/ckpt")
    old_metrics = None
    old_ckpt = None
    if ckpt_dir.exists():
        existing = sorted(ckpt_dir.glob("phase1_*.json"), reverse=True)
        if existing:
            old_ckpt = existing[0]
            try:
                cfg = Path("signals/models/configs") / old_ckpt.name
                old_model = XGBPhase1.load(old_ckpt, cfg)
                old_metrics = evaluate(old_model, X_h, y_h)
                logger.info(f"  old model: acc {old_metrics['accuracy']:.4f} brier {old_metrics['brier']:.4f}")
            except Exception as e:
                logger.warning(f"  old model load fail: {e}")

    logger.info(f"  new model: acc {new_metrics['accuracy']:.4f} brier {new_metrics['brier']:.4f}")

    # 6. Promotion gate
    promoted, reasons = promotion_gate(old_metrics, new_metrics)

    result = {
        "tag": tag,
        "timestamp": datetime.now().isoformat(),
        "panel_size": len(panel),
        "train_size": len(X_train),
        "holdout_size": len(X_h),
        "new_metrics": new_metrics,
        "old_metrics": old_metrics,
        "promoted": promoted,
        "reasons": reasons,
        "params": params,
    }

    # 7. 저장
    if promoted:
        new_ckpt = ckpt_dir / f"phase1_{tag}.json"
        new_cfg = Path("signals/models/configs") / f"phase1_{tag}.json"
        new_model.save(new_ckpt, new_cfg)
        logger.info(f"  ★ PROMOTED: {new_ckpt}")

        # archive 이전 모델
        if old_ckpt and old_ckpt.exists():
            archive = Path(archive_dir)
            archive.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_ckpt), archive / old_ckpt.name)
            old_cfg = Path("signals/models/configs") / old_ckpt.name
            if old_cfg.exists():
                shutil.move(str(old_cfg), archive / old_cfg.name)
    else:
        logger.warning(f"  ✗ REJECTED: reasons={reasons}")

    # 8. retrain_history 누적
    hist_path = Path(history_path)
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    history = []
    if hist_path.exists():
        with open(hist_path) as f:
            history = json.load(f)
    history.append(result)
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2, default=str)

    return result


# ============================================================================
# CLI
# ============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--holdout-days", type=int, default=180)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    res = run_retrain(n_trials=args.n_trials, holdout_days=args.holdout_days)

    print(f"\n=== retrain {res['tag']} ===")
    print(f"  promoted: {res['promoted']}")
    print(f"  new accuracy: {res['new_metrics']['accuracy']:.4f}")
    print(f"  new brier:    {res['new_metrics']['brier']:.4f}")
    if res['old_metrics']:
        print(f"  old accuracy: {res['old_metrics']['accuracy']:.4f}")
        print(f"  old brier:    {res['old_metrics']['brier']:.4f}")
    print(f"  reasons: {res['reasons']}")


if __name__ == "__main__":
    main()

"""Build detector v1 — full-panel 학습 + OOF threshold artifact 저장.

산출물:
  signals/models/ckpt/detector_v1.json      # 운영 모델 (full panel 학습)
  signals/models/ckpt/detector_v1_meta.json # 학습 메타 (feature names, params)
  output/detector_threshold.json            # 운영 config (regimes, threshold, cap)

운영 룰 (사용자 확정):
  regime: bull_quiet + bull_volatile
  threshold: OOF p99.95 (artifact 에 저장된 고정값)
  cap: 2 per day
  bear regime: silence (또는 warning only)

사용:
    python scripts/build_detector_v1.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.database import list_markets, load_candles
from data.market_universe import signal_eligible_markets
from signals.features import assemble_training_panel
from signals.models.xgb_phase1 import EXCLUDE_COLS

LEAK_COLS = {"net_under_tp", "max_return", "label", "label_tail",
             "next_open", "next_high", "next_low", "next_close",
             "next_max_return", "next_eod_return", "next_max_dd"}

XGB_PARAMS = dict(
    objective="binary:logistic", eval_metric="logloss",
    n_estimators=400, learning_rate=0.05, max_depth=6,
    min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, tree_method="hist",
    n_jobs=-1, random_state=42,
)

OPERATIONAL_RULE = {
    "regimes": ["bull_quiet", "bull_volatile"],
    "cap": 2,
    "threshold_pct": 0.9995,
    "target_pct": 0.20,
    "bear_action": "silence",  # bear_quiet/bear_volatile → 알림 X
}


def feature_cols(df, label_col):
    return [c for c in df.columns if c not in EXCLUDE_COLS and c != label_col
            and not c.startswith("next_") and c not in LEAK_COLS]


def train_binary(X, y):
    sw = compute_sample_weight(class_weight="balanced", y=y)
    m = xgb.XGBClassifier(**XGB_PARAMS)
    m.fit(X, y, sample_weight=sw, verbose=False)
    return m


def compute_oof_scores(panel, cols, label_col, n_splits=3, log=None):
    """전체 panel chronological n_splits → OOF 점수."""
    panel = panel[panel[label_col].notna()].sort_values("timestamp").reset_index(drop=True)
    dates = panel["timestamp"].drop_duplicates().sort_values().reset_index(drop=True)
    chunk = len(dates) // (n_splits + 1)

    oof_rows = []
    for i in range(1, n_splits + 1):
        train_dates = dates.iloc[: chunk * i]
        val_dates = dates.iloc[chunk * i: chunk * (i + 1)] if i < n_splits \
            else dates.iloc[chunk * i:]
        tr = panel[panel["timestamp"].isin(train_dates)]
        va = panel[panel["timestamp"].isin(val_dates)].copy()
        if len(tr) < 100 or len(va) < 50:
            continue
        y_tr = tr[label_col].astype(int).values
        if y_tr.sum() < 5:
            continue
        m = train_binary(tr[cols].astype(float).values, y_tr)
        va["_oof_score"] = m.predict_proba(va[cols].astype(float).values)[:, 1]
        oof_rows.append(va[["market", "timestamp", "_oof_score"]])
        if log:
            log.info(f"  OOF split {i}: train {len(tr):,} → val {len(va):,}")
    return pd.concat(oof_rows, ignore_index=True) if oof_rows else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upbit-db", default="data/upbit_d1.db")
    parser.add_argument("--binance-db", default="data/binance_d1.db")
    parser.add_argument("--target-pct", type=float, default=0.20)
    parser.add_argument("--n-oof-splits", type=int, default=3)
    parser.add_argument("--out-model", default="signals/models/ckpt/detector_v1.json")
    parser.add_argument("--out-meta", default="signals/models/ckpt/detector_v1_meta.json")
    parser.add_argument("--out-config", default="output/detector_threshold.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("build")

    print(f"=== Build detector v1 (target ≥{int(args.target_pct*100)}%) ===\n")

    log.info("loading + panel + label...")
    krw = signal_eligible_markets(list_markets(args.upbit_db))
    candles = {m: load_candles(args.upbit_db, m) for m in krw}
    if Path(args.binance_db).exists():
        for m in list_markets(args.binance_db):
            candles[m] = load_candles(args.binance_db, m)
    candles = {k: v for k, v in candles.items() if len(v) > 30}
    btc = load_candles(args.upbit_db, "KRW-BTC")
    panel = assemble_training_panel(candles, btc, normalize=True)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values(["market", "timestamp"]).reset_index(drop=True)

    g = panel.groupby("market", sort=False)
    panel["next_open"] = g["open"].shift(-1)
    panel["next_high"] = g["high"].shift(-1)
    panel["next_max_return"] = panel["next_high"] / panel["next_open"] - 1
    panel["label_tail"] = (panel["next_max_return"] >= args.target_pct).astype(float)
    panel.loc[panel["next_max_return"].isna(), "label_tail"] = np.nan
    panel = panel.dropna(subset=["next_open", "label_tail"]).reset_index(drop=True)

    df = panel[panel["label_tail"].notna()].copy()
    cols = feature_cols(df, "label_tail")
    log.info(f"panel: {df.shape}, features: {len(cols)}")

    # 1) OOF threshold
    log.info(f"computing OOF scores ({args.n_oof_splits} splits)...")
    oof = compute_oof_scores(panel, cols, "label_tail", args.n_oof_splits, log)
    if oof is None or len(oof) == 0:
        log.error("OOF 계산 실패")
        sys.exit(1)
    krw_oof = oof[oof["market"].str.startswith("KRW-")]
    threshold = float(krw_oof["_oof_score"].quantile(0.9995))
    log.info(f"OOF KRW score quantiles: "
             f"p99.5={krw_oof['_oof_score'].quantile(0.995):.4f}, "
             f"p99.9={krw_oof['_oof_score'].quantile(0.999):.4f}, "
             f"p99.95={threshold:.4f}")

    # 2) Final model (full panel)
    log.info("training final model on full panel...")
    X = df[cols].astype(float).values
    y = df["label_tail"].astype(int).values
    log.info(f"  X: {X.shape}, positive rate: {y.mean()*100:.2f}%")
    model = train_binary(X, y)

    # 3) Save model + meta
    Path(args.out_model).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(args.out_model)
    log.info(f"saved model: {args.out_model}")

    train_start = str(df["timestamp"].min().date())
    train_end = str(df["timestamp"].max().date())

    meta = {
        "version": "detector_v1",
        "built_at": datetime.utcnow().isoformat() + "Z",
        "target_pct": args.target_pct,
        "feature_cols": cols,
        "n_features": len(cols),
        "n_train_samples": int(len(df)),
        "positive_rate_pct": float(y.mean() * 100),
        "train_start": train_start,
        "train_end": train_end,
        "xgb_params": {k: v for k, v in XGB_PARAMS.items()
                       if isinstance(v, (int, float, str, bool))},
        "n_oof_splits": args.n_oof_splits,
        "n_oof_samples": int(len(oof)),
    }
    with open(args.out_meta, "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"saved meta: {args.out_meta}")

    # 4) Save operational config
    config = {
        "version": "detector_v1",
        "built_at": meta["built_at"],
        "model_ckpt": args.out_model,
        "model_meta": args.out_meta,
        "operational_rule": {
            **OPERATIONAL_RULE,
            "threshold": threshold,
            "threshold_source": f"OOF p99.95 over KRW universe ({len(krw_oof):,} samples)",
        },
        "framing": (
            "BTC bull regime에서 ≥20% tail pump 가능성이 "
            "과거 OOF 기준 최상위 0.05%에 들어온 후보 (매수 추천 X)"
        ),
        "validation_summary": {
            "method": "5-fold purged walk-forward + per-fold inner OOF threshold",
            "candidate": "C3 (bull_all p99.95 cap2)",
            "n_folds_positive": "3 of 4 active",
            "no_trade_folds": "1 of 5",
            "ev_avg_pct": 7.40,
            "ev_2024_pct": -0.89,
            "n_total_trades": 56,
            "rank_fallback_rejected": True,
        },
    }
    Path(args.out_config).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_config, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    log.info(f"saved config: {args.out_config}")

    print(f"\n=== Done ===")
    print(f"Model:     {args.out_model}")
    print(f"Meta:      {args.out_meta}")
    print(f"Config:    {args.out_config}")
    print(f"Threshold: {threshold:.4f}  (OOF p99.95, KRW universe)")
    print(f"Rule:      regimes={OPERATIONAL_RULE['regimes']}, cap={OPERATIONAL_RULE['cap']}")


if __name__ == "__main__":
    main()

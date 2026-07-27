"""Pre-open Trigger v1 — 학습 + WF lift 검증 + final artifact 저장.

목적:
  08:55 KST 알림용 모델. 09:00 직후 첫 15m/30m/1h 펌프 예측.
  사용자 audit 결과 (first15_3pct top1% precision 38.2%, base 5.1% → lift 7.5x)
  정당화됨.

흐름:
  1. 15m precursor (08:30 snapshot) 계산
  2. preopen labels (X+1 09:00 첫 N bars hit) 계산
  3. join: precursor(X) + labels(X+1)
  5. WF 5-fold 학습 (head 별 binary XGBoost)
  6. lift @top0.5/1/2% 측정
  7. Final model = full panel 학습, artifact 저장

산출물:
  signals/models/ckpt/preopen_v1/<head>.json     # head 별 model
  signals/models/ckpt/preopen_v1/meta.json
  output/preopen_trigger_v1_validation.csv

사용:
    python scripts/build_preopen_trigger_v1.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.database import list_markets, load_candles
from data.market_universe import signal_eligible_markets
from signals.labels_preopen import PREOPEN_HEADS, compute_preopen_labels
from signals.precursors import FIFTEEN_M_FEATURES as PRECURSOR_15M_FEATURES
from signals.precursors import build_15m_precursor
from signals.validate import PurgedWalkForward

XGB_PARAMS = dict(
    objective="binary:logistic", eval_metric="logloss",
    n_estimators=400, learning_rate=0.05, max_depth=6,
    min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, tree_method="hist",
    n_jobs=-1, random_state=42,
)

TARGET_HEADS = list(PREOPEN_HEADS.keys())
TOPK_PCTS = [0.005, 0.01, 0.02]


def train_binary(X, y):
    sw = compute_sample_weight(class_weight="balanced", y=y)
    m = xgb.XGBClassifier(**XGB_PARAMS)
    m.fit(X, y, sample_weight=sw, verbose=False)
    return m


def lift_at_topk(scores, labels, pct):
    if len(scores) == 0 or labels.sum() == 0:
        return None
    base = labels.mean()
    n_top = max(1, int(len(scores) * pct))
    idx = np.argsort(-scores)[:n_top]
    prec = float(labels[idx].mean())
    return {"prec_pct": prec * 100, "lift": prec / base if base > 0 else None,
            "n_top": n_top, "base_pct": base * 100}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upbit-d1", default="data/upbit_d1.db")
    parser.add_argument("--upbit-15m", default="data/upbit_15m.db")
    parser.add_argument("--binance-d1", default="data/binance_d1.db")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--embargo", type=int, default=10)
    parser.add_argument("--holdout", type=int, default=180)
    parser.add_argument("--min-rows-per-date", type=int, default=100)
    parser.add_argument("--out-dir", default="signals/models/ckpt/preopen_v1")
    parser.add_argument("--out-validation", default="output/preopen_trigger_v1_validation.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("preopen")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Pre-open Trigger v1 — {len(TARGET_HEADS)} heads ===\n")
    print(f"Heads: {TARGET_HEADS}")
    print("Target: 09:00 직후 첫 15m/30m/1h 펌프 hit\n")

    # === 1) 15m candles → precursor + preopen labels ===
    log.info("loading 15m candles...")
    krw_15m = [
        m
        for m in signal_eligible_markets(list_markets(args.upbit_15m))
        if m.startswith("KRW-")
    ]
    candles_15m = {m: load_candles(args.upbit_15m, m) for m in krw_15m}
    candles_15m = {k: v for k, v in candles_15m.items() if v is not None and len(v) > 100}
    log.info(f"  15m markets: {len(candles_15m)}")

    log.info("computing 15m precursor (08:30 snapshot) + preopen labels...")
    btc_15m = candles_15m.get("KRW-BTC", pd.DataFrame())
    precursor_df = build_15m_precursor(candles_15m, btc_15m)
    log.info(f"  precursor rows: {len(precursor_df)}")

    label_df = compute_preopen_labels(candles_15m)
    log.info(f"  label rows: {len(label_df)}")

    # === 2) Join ===
    # precursor.date_only = X (snapshot at 08:30 X+1)
    # labels.label_date = X+1 (the day being predicted)
    # → join: precursor(X) ⨝ labels(label_date=X+1, shifted to X for join)
    #
    # Intentional: no daily D-1 features in the model. At 08:55, the D-1
    # daily candle has 5 minutes left and would create train/live mismatch.
    label_df["date_only"] = (pd.to_datetime(label_df["label_date"]) - pd.Timedelta(days=1)).dt.date

    full = precursor_df.merge(
        label_df.drop(columns=["label_date"]),
        on=["market", "date_only"],
        how="inner",
    )
    full["timestamp"] = pd.to_datetime(full["date_only"]) + pd.Timedelta(days=1)
    before = len(full)
    per_date = full.groupby("date_only").size()
    good_dates = per_date[per_date >= args.min_rows_per_date].index
    full = full[full["date_only"].isin(good_dates)].copy()
    log.info(f"  joined: {full.shape}")
    log.info(
        "  date filter: %s -> %s rows, dates=%s (min_rows_per_date=%s)",
        f"{before:,}", f"{len(full):,}", len(good_dates), args.min_rows_per_date,
    )

    feature_cols = list(PRECURSOR_15M_FEATURES)
    log.info(f"  features: {len(feature_cols)} (15m precursor only)")

    # === 4) WF train + per-head per-universe lift ===
    splitter = PurgedWalkForward(args.n_folds, args.embargo, args.holdout)
    fold_data = []
    for fold, (train_dates, val_dates) in enumerate(splitter.split(full["timestamp"]), 1):
        train_p = full[full["timestamp"].isin(train_dates)]
        val_p = full[full["timestamp"].isin(val_dates)].copy()
        if len(train_p) < 100 or len(val_p) < 50:
            continue
        log.info(f"Fold {fold}: train {len(train_p):,} / val {len(val_p):,}")
        fold_data.append((fold, train_p, val_p))

    log.info(f"\nWF training {len(TARGET_HEADS)} heads × {len(fold_data)} folds...")
    val_rows = []
    for head_name in TARGET_HEADS:
        log.info(f"\n--- head: {head_name} ({PREOPEN_HEADS[head_name]['name']}) ---")
        for fold, train_p, val_p in fold_data:
            t0 = time.time()
            df_tr = train_p[train_p[head_name].notna()].copy()
            df_va = val_p[val_p[head_name].notna()].copy()
            if len(df_tr) < 100:
                continue
            X_tr = df_tr[feature_cols].astype(float).values
            y_tr = df_tr[head_name].astype(int).values
            if y_tr.sum() < 10 or (y_tr == 0).sum() < 10:
                log.info(f"  fold {fold}: degenerate")
                continue
            X_va = df_va[feature_cols].astype(float).values
            y_va = df_va[head_name].astype(int).values

            m = train_binary(X_tr, y_tr)
            scores = m.predict_proba(X_va)[:, 1]

            # KRW only
            krw_mask = df_va["market"].str.startswith("KRW-").values
            sc_krw = scores[krw_mask]
            y_krw = y_va[krw_mask]
            base = float(y_krw.mean()) * 100 if len(y_krw) > 0 else 0
            rec = {"head": head_name, "fold": fold,
                    "n_val": len(y_krw), "base_pct": base}
            for pct in TOPK_PCTS:
                lk = lift_at_topk(sc_krw, y_krw, pct)
                if lk:
                    rec[f"prec@top{pct*100:g}pct"] = lk["prec_pct"]
                    rec[f"lift@top{pct*100:g}pct"] = lk["lift"]
                    rec[f"n_top{pct*100:g}pct"] = lk["n_top"]
            val_rows.append(rec)
            log.info(f"  fold {fold}: trained ({time.time()-t0:.1f}s)")

    val_df = pd.DataFrame(val_rows)
    Path(args.out_validation).parent.mkdir(parents=True, exist_ok=True)
    val_df.to_csv(args.out_validation, index=False)
    log.info(f"saved {args.out_validation}")

    # === 5) Final model per head ===
    log.info("\ntraining final models on full panel...")
    final_meta = {
        "version": "preopen_v1",
        "built_at": datetime.utcnow().isoformat() + "Z",
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "heads": PREOPEN_HEADS,
        "xgb_params": {k: v for k, v in XGB_PARAMS.items() if isinstance(v, (int, float, str, bool))},
        "head_positive_rate": {},
        "n_train_samples": {},
    }
    for head_name in TARGET_HEADS:
        df = full[full[head_name].notna()].copy()
        X = df[feature_cols].astype(float).values
        y = df[head_name].astype(int).values
        if y.sum() < 10:
            log.warning(f"  {head_name}: too few positives — skip")
            continue
        m = train_binary(X, y)
        out_path = out_dir / f"{head_name}.json"
        m.save_model(str(out_path))
        final_meta["head_positive_rate"][head_name] = float(y.mean() * 100)
        final_meta["n_train_samples"][head_name] = int(len(df))
        log.info(f"  {head_name}: saved {out_path} (pos_rate {y.mean()*100:.2f}%)")

    meta_path = out_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(final_meta, f, indent=2, ensure_ascii=False, default=str)
    log.info(f"saved {meta_path}")

    # === 6) Summary ===
    print(f"\n{'='*100}")
    print("WF Lift Summary — KRW only, mean across folds")
    print(f"{'='*100}\n")
    if len(val_df) > 0:
        agg = val_df.groupby("head").agg(
            n_folds=("fold", "nunique"),
            base_pct=("base_pct", "mean"),
            **{f"lift@top{pct*100:g}pct": (f"lift@top{pct*100:g}pct", "mean") for pct in TOPK_PCTS},
            **{f"prec@top{pct*100:g}pct": (f"prec@top{pct*100:g}pct", "mean") for pct in TOPK_PCTS},
        ).reset_index()
        head_order = list(PREOPEN_HEADS.keys())
        agg["_h"] = agg["head"].map({h: i for i, h in enumerate(head_order)})
        agg = agg.sort_values("_h").drop(columns=["_h"])
        print(agg.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print(f"\nDone. Model: {out_dir}, Meta: {meta_path}")


if __name__ == "__main__":
    main()

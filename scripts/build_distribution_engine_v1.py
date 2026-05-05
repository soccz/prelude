"""Distribution Engine v1 — 7 head 학습 + universe 별 lift 검증.

흐름:
  1. 일봉 panel (features) 빌드 — detector_v1 과 동일 path
  2. 4h 봉으로 7 head label 계산 (signals/labels_distribution)
  3. (market, date) 기준 join → features + 7 head labels
  4. WF 5-fold per head → train binary XGBoost, val 점수
  5. Universe (top50/top100/all) 별 lift@topK + calibration + fail EOD
  6. Final 모델 (full panel) 저장 → signals/models/ckpt/dist_engine_v1/<head>.json

운영 원칙:
  - universe 는 inference filter (학습은 모든 KRW)
  - regime gate 는 head 별로 다를 수 있음 (기본 bull_all 평가)

산출물:
  signals/models/ckpt/dist_engine_v1/<head>.json     # head 별 model
  signals/models/ckpt/dist_engine_v1/meta.json       # feature_cols, params, head config
  output/distribution_engine_v1_validation.csv       # WF lift 검증 결과

사용:
    python scripts/build_distribution_engine_v1.py
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
from signals.features import assemble_training_panel, compute_btc_features
from signals.labels_distribution import HEADS, compute_distribution_labels, MAX_BARS
from signals.models.xgb_phase1 import EXCLUDE_COLS
from signals.validate import PurgedWalkForward


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

LIQ_TIERS = ["all", "top50", "top100"]
TOPK_PCTS = [0.001, 0.005, 0.01, 0.05]  # top 0.1%, 0.5%, 1%, 5% — lift@K


def feature_cols_from(df, label_cols):
    EXTRA_DROP = {"quote_volume_d1", "date_only", "liq_rank_daily"}
    cols = []
    for c in df.columns:
        if c in EXCLUDE_COLS or c in label_cols or c in HEADS:
            continue
        if c in LEAK_COLS or c in EXTRA_DROP:
            continue
        if c.startswith("next_"):
            continue
        # numeric 만 — date 등 non-numeric drop
        dt = df[c].dtype
        if dt == object or "datetime" in str(dt):
            continue
        cols.append(c)
    return cols


def build_4h_panel_for_labels(candles_4h: dict, btc_regime_map: dict) -> pd.DataFrame:
    """4h bars → (market, date) padded arrays + regime."""
    rows = []
    for market, df in candles_4h.items():
        if df is None or len(df) < 1:
            continue
        df = df.sort_values("timestamp").copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["bar_date"] = (df["timestamp"] - pd.Timedelta(hours=9)).dt.date
        for date, g in df.groupby("bar_date", sort=False):
            g2 = g.sort_values("timestamp")
            n = min(len(g2), MAX_BARS)
            if n < 1:
                continue
            opens = g2["open"].values.astype(float)
            closes = g2["close"].values.astype(float)
            highs = g2["high"].values.astype(float)[:MAX_BARS]
            lows = g2["low"].values.astype(float)[:MAX_BARS]
            highs_p = np.full(MAX_BARS, np.nan)
            lows_p = np.full(MAX_BARS, np.nan)
            highs_p[:n] = highs[:n]
            lows_p[:n] = lows[:n]
            rows.append({
                "market": market,
                "date_only": date,
                "open_4h": float(opens[0]),
                "close_4h": float(closes[-1]),
                "highs": highs_p,
                "lows": lows_p,
                "btc_regime_4h": btc_regime_map.get(date, "unknown"),
                "n_bars": n,
            })
    return pd.DataFrame(rows)


def train_binary(X, y):
    sw = compute_sample_weight(class_weight="balanced", y=y)
    m = xgb.XGBClassifier(**XGB_PARAMS)
    m.fit(X, y, sample_weight=sw, verbose=False)
    return m


def lift_at_topk(scores: np.ndarray, labels: np.ndarray, topk_pct: float):
    """top-K% precision / lift over base rate. base_rate = labels.mean()."""
    if len(scores) == 0 or labels.sum() == 0:
        return None, None, None
    base = labels.mean()
    n_top = max(1, int(len(scores) * topk_pct))
    idx = np.argsort(-scores)[:n_top]
    precision = float(labels[idx].mean())
    lift = precision / base if base > 0 else None
    return precision, lift, n_top


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upbit-d1", default="data/upbit_d1.db")
    parser.add_argument("--upbit-4h", default="data/upbit_4h.db")
    parser.add_argument("--binance-d1", default="data/binance_d1.db")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--embargo", type=int, default=10)
    parser.add_argument("--holdout", type=int, default=180)
    parser.add_argument("--out-dir", default="signals/models/ckpt/dist_engine_v1")
    parser.add_argument("--out-validation", default="output/distribution_engine_v1_validation.csv")
    parser.add_argument("--limit-markets", type=int, default=None,
                        help="개발용 — KRW 일부만 사용")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("dist-eng")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Distribution Engine v1 — {len(HEADS)} heads ===\n")
    print(f"Heads: {list(HEADS.keys())}")
    print(f"Universe tiers (eval): {LIQ_TIERS}\n")

    # =========================================================================
    # 1) Daily panel (features)
    # =========================================================================
    log.info("loading daily candles for features...")
    krw = list_markets(args.upbit_d1)
    if args.limit_markets:
        krw = krw[: args.limit_markets]
    candles_d1 = {m: load_candles(args.upbit_d1, m) for m in krw}
    if Path(args.binance_d1).exists():
        for m in list_markets(args.binance_d1):
            candles_d1[m] = load_candles(args.binance_d1, m)
    candles_d1 = {k: v for k, v in candles_d1.items() if v is not None and len(v) > 30}
    btc_d1 = load_candles(args.upbit_d1, "KRW-BTC")
    log.info(f"  d1 markets: {len(candles_d1)}")

    log.info("building daily panel (features)...")
    panel = assemble_training_panel(candles_d1, btc_d1, normalize=True)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values(["market", "timestamp"]).reset_index(drop=True)
    panel["date_only"] = panel["timestamp"].dt.date
    log.info(f"  daily panel: {panel.shape}")

    # also need quote_volume per (market, date) for universe ranking
    panel["quote_volume_d1"] = panel.get("quote_volume", np.nan)
    if "quote_volume" not in panel.columns:
        log.warning("  no quote_volume in panel — universe filter may be limited")

    # =========================================================================
    # 2) 4h panel + labels
    # =========================================================================
    log.info("loading 4h candles for labels...")
    krw_4h = [m for m in list_markets(args.upbit_4h) if m.startswith("KRW-")]
    if args.limit_markets:
        krw_4h = krw_4h[: args.limit_markets]
    candles_4h = {m: load_candles(args.upbit_4h, m) for m in krw_4h}
    candles_4h = {k: v for k, v in candles_4h.items() if v is not None and len(v) > 0}
    log.info(f"  4h markets: {len(candles_4h)}")

    btc_feat = compute_btc_features(btc_d1.copy())
    btc_feat["date_only"] = pd.to_datetime(btc_feat["timestamp"]).dt.date
    btc_regime_map = dict(zip(btc_feat["date_only"], btc_feat["btc_regime"]))

    log.info("building 4h panel + computing labels...")
    panel_4h = build_4h_panel_for_labels(candles_4h, btc_regime_map)
    log.info(f"  4h panel: {panel_4h.shape}")
    label_df = compute_distribution_labels(panel_4h)
    label_df["market"] = panel_4h["market"].values
    label_df["label_date"] = panel_4h["date_only"].values
    log.info("  labels computed for 7 heads")

    # === LEAK FIX ===
    # features[t]  = t일 일봉까지의 정보
    # label[t]     = t+1일 4h path 으로 측정해야 leak-free
    # 구현: label_df 의 label_date 를 feature_date = label_date - 1day 로 매핑하여 join
    label_df["date_only"] = (pd.to_datetime(label_df["label_date"]) - pd.Timedelta(days=1)).dt.date
    log.info(f"  applied next-day shift: feature_date = label_date - 1day (leak fix)")

    # join panel (features) ← labels (다음 일봉 4h path)
    log.info("joining features + next-day labels...")
    full = panel.merge(
        label_df.drop(columns=["label_date"]),
        on=["market", "date_only"], how="inner"
    )
    log.info(f"  joined: {full.shape}, head positive rates (next-day):")
    for h in HEADS:
        pr = float(full[h].mean()) * 100 if h in full.columns else 0
        log.info(f"    {h}: {pr:.2f}%")

    # daily quote_volume rank (universe)
    full["liq_rank_daily"] = full.groupby("date_only")["quote_volume_d1"].rank(
        method="dense", ascending=False, na_option="bottom"
    )

    # =========================================================================
    # 3) WF train + per-head + per-universe lift
    # =========================================================================
    feature_cols = feature_cols_from(full, label_cols=list(HEADS.keys()))
    log.info(f"  features: {len(feature_cols)}")

    splitter = PurgedWalkForward(args.n_folds, args.embargo, args.holdout)
    val_records = []  # per (head, fold, universe)

    log.info("WF training per head...")
    fold_data = []
    for fold, (train_dates, val_dates) in enumerate(splitter.split(full["timestamp"]), 1):
        train_p = full[full["timestamp"].isin(train_dates)]
        val_p = full[full["timestamp"].isin(val_dates)].copy()
        if len(train_p) < 100 or len(val_p) < 50:
            continue
        log.info(f"Fold {fold}: train {len(train_p):,} / val {len(val_p):,}")
        fold_data.append((fold, train_p, val_p))

    for head_name, head_cfg in HEADS.items():
        log.info(f"\n--- head: {head_name} ({head_cfg['name']}) ---")
        for fold, train_p, val_p in fold_data:
            t0 = time.time()
            df_tr = train_p[train_p[head_name].notna()].copy()
            df_va = val_p[val_p[head_name].notna()].copy()
            if len(df_tr) < 100:
                continue
            X_tr = df_tr[feature_cols].astype(float).values
            y_tr = df_tr[head_name].astype(int).values
            if y_tr.sum() < 10 or (y_tr == 0).sum() < 10:
                log.info(f"  fold {fold}: degenerate labels — skip")
                continue
            X_va = df_va[feature_cols].astype(float).values
            y_va = df_va[head_name].astype(int).values

            m = train_binary(X_tr, y_tr)
            scores = m.predict_proba(X_va)[:, 1]
            df_va = df_va.assign(_score=scores)

            # eval per universe (KRW only for liquidity)
            for liq in LIQ_TIERS:
                if liq == "all":
                    sub = df_va[df_va["market"].str.startswith("KRW-")]
                else:
                    n = int(liq.replace("top", ""))
                    sub = df_va[(df_va["market"].str.startswith("KRW-")) &
                                (df_va["liq_rank_daily"] <= n)]
                if len(sub) < 50:
                    continue
                base = float(sub[head_name].mean()) * 100
                rec = {"head": head_name, "fold": fold, "universe": liq,
                       "n_val": len(sub), "base_rate_pct": base}
                for pct in TOPK_PCTS:
                    prec, lift, ntop = lift_at_topk(sub["_score"].values, sub[head_name].values, pct)
                    if prec is not None:
                        rec[f"prec@top{pct*100:g}pct"] = prec * 100
                        rec[f"lift@top{pct*100:g}pct"] = lift
                        rec[f"n_top{pct*100:g}pct"] = ntop
                val_records.append(rec)
            elapsed = time.time() - t0
            log.info(f"  fold {fold}: trained + eval ({elapsed:.1f}s)")

    val_df = pd.DataFrame(val_records)
    Path(args.out_validation).parent.mkdir(parents=True, exist_ok=True)
    val_df.to_csv(args.out_validation, index=False)
    log.info(f"\nsaved {args.out_validation}")

    # =========================================================================
    # 4) Final model per head (full panel)
    # =========================================================================
    log.info("\ntraining final models on full panel...")
    final_meta = {
        "version": "dist_engine_v1",
        "built_at": datetime.utcnow().isoformat() + "Z",
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "heads": HEADS,
        "liq_tiers_evaluated": LIQ_TIERS,
        "xgb_params": {k: v for k, v in XGB_PARAMS.items()
                       if isinstance(v, (int, float, str, bool))},
        "head_positive_rate": {},
        "n_train_samples": {},
    }
    for head_name in HEADS:
        df = full[full[head_name].notna()].copy()
        X = df[feature_cols].astype(float).values
        y = df[head_name].astype(int).values
        if y.sum() < 10:
            log.warning(f"  {head_name}: too few positives — skip final")
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

    # =========================================================================
    # 5) 요약 출력
    # =========================================================================
    print("\n" + "=" * 110)
    print("WF Validation Summary — mean lift across folds (universe × head)")
    print("=" * 110)

    if len(val_df) > 0:
        # mean lift across folds per (head, universe)
        agg = val_df.groupby(["head", "universe"]).agg(
            n_folds=("fold", "nunique"),
            base_rate_pct=("base_rate_pct", "mean"),
            **{f"prec@top{pct*100:g}pct_mean": (f"prec@top{pct*100:g}pct", "mean") for pct in TOPK_PCTS},
            **{f"lift@top{pct*100:g}pct_mean": (f"lift@top{pct*100:g}pct", "mean") for pct in TOPK_PCTS},
        ).reset_index()
        # 정렬: head 순서 유지
        head_order = list(HEADS.keys())
        agg["_h_idx"] = agg["head"].map({h: i for i, h in enumerate(head_order)})
        agg = agg.sort_values(["_h_idx", "universe"]).drop(columns=["_h_idx"])
        cols = ["head", "universe", "n_folds", "base_rate_pct"] + \
               [f"lift@top{pct*100:g}pct_mean" for pct in TOPK_PCTS] + \
               [f"prec@top{pct*100:g}pct_mean" for pct in TOPK_PCTS]
        print(agg[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print(f"\n=== Done ===")
    print(f"Models:     {out_dir}")
    print(f"Meta:       {meta_path}")
    print(f"Validation: {args.out_validation}")


if __name__ == "__main__":
    main()

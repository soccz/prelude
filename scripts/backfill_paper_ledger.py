"""Backfill paper ledger — 과거 데이터 OOF 시뮬로 paper ledger 채우기.

목적:
  operational paper_ledger 는 천천히 차오름. calibration 신뢰도 위해 즉시 backfill.
  WF OOF 로 학습 (in-sample leak 없음, 진짜 라이브 분포 근사).

흐름:
  1. panel + labels (leak fix) 빌드 — sweep_dist_engine_v1.py 와 동일
  2. WF 5-fold per head → val 점수
  3. val OOF 합치기 → per (market, date) row 별 score for h2/h5/h6
  4. 각 val date 별로:
     - top100 KRW universe filter
     - setup 매칭 (signals.setups)
     - composite ranking + top-K
     - alert 행 만들고 status="closed" + label 직접 활용 (hit_h2/h5/h6)
  5. output/paper_ledger_backfill.csv 저장 (operational 과 분리)

산출물:
  output/paper_ledger_backfill.csv  — backfill rows (전부 closed)

사용:
    python scripts/backfill_paper_ledger.py
    python scripts/backfill_paper_ledger.py --top-k 10
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
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
from signals.setups import detect_setups, SETUP_LIBRARY
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


def feature_cols_from(df, label_cols):
    EXTRA_DROP = {"quote_volume_d1", "date_only", "liq_rank_daily",
                   "highs", "lows", "n_bars", "btc_regime_4h", "label_date",
                   "eod_ret_next"}
    cols = []
    for c in df.columns:
        if c in EXCLUDE_COLS or c in label_cols or c in HEADS:
            continue
        if c in LEAK_COLS or c in EXTRA_DROP:
            continue
        if c.startswith("next_"):
            continue
        dt = df[c].dtype
        if dt == object or "datetime" in str(dt):
            continue
        cols.append(c)
    return cols


def build_4h_panel_for_labels(candles_4h, btc_regime_map):
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
            highs_p = np.full(MAX_BARS, np.nan); lows_p = np.full(MAX_BARS, np.nan)
            highs_p[:n] = highs[:n]; lows_p[:n] = lows[:n]
            rows.append({
                "market": market, "date_only": date,
                "open_4h": float(opens[0]), "close_4h": float(closes[-1]),
                "highs": highs_p, "lows": lows_p,
                "btc_regime_4h": btc_regime_map.get(date, "unknown"),
                "n_bars": n,
                "eod_ret_4h": float(closes[-1]) / float(opens[0]) - 1 if opens[0] > 0 else np.nan,
                "max_ret_4h": float(highs[:n].max()) / float(opens[0]) - 1 if opens[0] > 0 else np.nan,
            })
    return pd.DataFrame(rows)


def train_binary(X, y):
    sw = compute_sample_weight(class_weight="balanced", y=y)
    m = xgb.XGBClassifier(**XGB_PARAMS)
    m.fit(X, y, sample_weight=sw, verbose=False)
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upbit-d1", default="data/upbit_d1.db")
    parser.add_argument("--upbit-4h", default="data/upbit_4h.db")
    parser.add_argument("--binance-d1", default="data/binance_d1.db")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--embargo", type=int, default=10)
    parser.add_argument("--holdout", type=int, default=180)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--universe", default="top100", choices=["top50", "top100", "all"])
    parser.add_argument("--out-csv", default="output/paper_ledger_backfill.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("backfill")

    print(f"=== Backfill paper ledger — WF OOF 시뮬 (universe={args.universe}, top-k={args.top_k}) ===\n")

    # ===== load data =====
    log.info("loading panel + labels...")
    krw = list_markets(args.upbit_d1)
    candles_d1 = {m: load_candles(args.upbit_d1, m) for m in krw}
    if Path(args.binance_d1).exists():
        for m in list_markets(args.binance_d1):
            candles_d1[m] = load_candles(args.binance_d1, m)
    candles_d1 = {k: v for k, v in candles_d1.items() if v is not None and len(v) > 30}
    btc_d1 = load_candles(args.upbit_d1, "KRW-BTC")
    panel = assemble_training_panel(candles_d1, btc_d1, normalize=True)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values(["market", "timestamp"]).reset_index(drop=True)
    panel["date_only"] = panel["timestamp"].dt.date
    panel["quote_volume_d1"] = panel.get("quote_volume", np.nan)

    krw_4h = [m for m in list_markets(args.upbit_4h) if m.startswith("KRW-")]
    candles_4h = {m: load_candles(args.upbit_4h, m) for m in krw_4h}
    candles_4h = {k: v for k, v in candles_4h.items() if v is not None and len(v) > 0}
    btc_feat = compute_btc_features(btc_d1.copy())
    btc_feat["date_only"] = pd.to_datetime(btc_feat["timestamp"]).dt.date
    btc_regime_map = dict(zip(btc_feat["date_only"], btc_feat["btc_regime"]))
    panel_4h = build_4h_panel_for_labels(candles_4h, btc_regime_map)
    label_df = compute_distribution_labels(panel_4h)
    label_df["market"] = panel_4h["market"].values
    label_df["label_date"] = panel_4h["date_only"].values
    label_df["eod_ret_next"] = panel_4h["eod_ret_4h"].values
    label_df["max_ret_next"] = panel_4h["max_ret_4h"].values
    # leak fix
    label_df["date_only"] = (pd.to_datetime(label_df["label_date"]) - pd.Timedelta(days=1)).dt.date

    # keep label_date (= predicted target day) for paper_ledger.date
    full = panel.merge(label_df, on=["market", "date_only"], how="inner")
    full["liq_rank_daily"] = full.groupby("date_only")["quote_volume_d1"].rank(
        method="dense", ascending=False, na_option="bottom"
    )
    feature_cols = feature_cols_from(full, list(HEADS.keys()) + ["eod_ret_next", "max_ret_next"])
    log.info(f"  joined: {full.shape}, features: {len(feature_cols)}")

    # ===== WF + per-head val OOF score =====
    splitter = PurgedWalkForward(args.n_folds, args.embargo, args.holdout)
    fold_data = []
    for fold, (train_dates, val_dates) in enumerate(splitter.split(full["timestamp"]), 1):
        train_p = full[full["timestamp"].isin(train_dates)]
        val_p = full[full["timestamp"].isin(val_dates)].copy()
        if len(train_p) < 100 or len(val_p) < 50:
            continue
        log.info(f"Fold {fold}: train {len(train_p):,} / val {len(val_p):,}")
        fold_data.append((fold, train_p, val_p))

    log.info(f"WF training all 7 heads × {len(fold_data)} folds...")
    val_scored_chunks = []  # 각 fold 의 val_p (head 별 score 컬럼 추가)
    for fold, train_p, val_p in fold_data:
        t0 = time.time()
        val_p_scored = val_p.copy()
        for head_name in HEADS:
            df_tr = train_p[train_p[head_name].notna()]
            if len(df_tr) < 100:
                continue
            X_tr = df_tr[feature_cols].astype(float).values
            y_tr = df_tr[head_name].astype(int).values
            if y_tr.sum() < 10 or (y_tr == 0).sum() < 10:
                continue
            X_va = val_p_scored[feature_cols].astype(float).values
            m = train_binary(X_tr, y_tr)
            val_p_scored[f"p_{head_name}"] = m.predict_proba(X_va)[:, 1]
        val_scored_chunks.append(val_p_scored)
        log.info(f"  fold {fold} done ({time.time()-t0:.1f}s)")

    full_oof = pd.concat(val_scored_chunks, ignore_index=True)
    full_oof = full_oof[full_oof["market"].str.startswith("KRW-")]
    log.info(f"OOF KRW: {len(full_oof):,}")

    # ===== universe + setup + assemble alerts per date =====
    log.info("assembling alerts per date...")
    if args.universe.startswith("top"):
        n = int(args.universe.replace("top", ""))
        full_oof = full_oof[full_oof["liq_rank_daily"] <= n]
        log.info(f"  after universe filter ({args.universe}): {len(full_oof):,}")

    full_oof["setups"] = full_oof.apply(detect_setups, axis=1)
    full_oof["primary_setups"] = full_oof["setups"].apply(
        lambda lst: [s for s in lst if s != "S04"]
    )
    full_oof["btc_context"] = full_oof["setups"].apply(
        lambda lst: "S04" if "S04" in lst else "—"
    )

    has_setup = full_oof[full_oof["primary_setups"].apply(len) > 0]
    log.info(f"  with primary setup fired: {len(has_setup):,}")

    # composite
    has_setup = has_setup.copy()
    has_setup["composite"] = (
        has_setup.get("p_h2_hit_3_4h", 0).fillna(0)
        + has_setup.get("p_h6_hit_5_24h", 0).fillna(0)
        + 5 * has_setup.get("p_h5_tail_20", 0).fillna(0)
    )

    # per-day top-K
    has_setup["alert_rank"] = has_setup.groupby("date_only")["composite"].rank(
        method="first", ascending=False
    )
    alerts = has_setup[has_setup["alert_rank"] <= args.top_k].copy()
    log.info(f"  alerts (top-{args.top_k} per day): {len(alerts):,}")

    # ===== build backfill ledger rows =====
    rows = []
    for _, r in alerts.iterrows():
        rows.append({
            "date": str(r["label_date"]),  # 예측 대상 day (label_date = feature_date + 1)
            "coin": r["market"],
            "setup_ids": "+".join(r["primary_setups"]),
            "btc_regime": r.get("btc_regime", "unknown"),
            "btc_context": r.get("btc_context", "—"),
            "p_h2_3pct_4h": float(r.get("p_h2_hit_3_4h", 0) or 0),
            "p_h6_5pct_24h": float(r.get("p_h6_hit_5_24h", 0) or 0),
            "p_h5_20pct_tail": float(r.get("p_h5_tail_20", 0) or 0),
            "p_h1_stable_4h": float(r.get("p_h1_stable_3_4h", 0) or 0),
            "p_h7_mid_7_24h": float(r.get("p_h7_mid_7_24h", 0) or 0),
            "log_return_1d_pct": float(r.get("log_return_1d", 0) or 0) * 100,
            "atr_pct_14_pct": float(r.get("atr_pct_14", 0) or 0) * 100,
            "return_5d_rank": float(r.get("return_5d", 0) or 0),
            "return_7d_rank": float(r.get("return_7d", 0) or 0),
            "vol_5d_rank": float(r.get("vol_5d", 0) or 0),
            "roc_3d_rank": float(r.get("roc_3d", 0) or 0),
            "composite_score": float(r["composite"]),
            "alert_rank": int(r["alert_rank"]),
            # realized = labels (we already know)
            "next_max_return_pct": float(r.get("max_ret_next", np.nan) or np.nan) * 100 if pd.notna(r.get("max_ret_next")) else np.nan,
            "next_close_return_pct": float(r.get("eod_ret_next", np.nan) or np.nan) * 100 if pd.notna(r.get("eod_ret_next")) else np.nan,
            "hit_h2": int(r.get("h2_hit_3_4h", 0) or 0),
            "hit_h5": int(r.get("h5_tail_20", 0) or 0),
            "hit_h6": int(r.get("h6_hit_5_24h", 0) or 0),
            "status": "closed",
            "notes": "backfill_oof",
        })

    df_out = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.out_csv, index=False)
    log.info(f"saved {args.out_csv}: {len(df_out):,} rows")

    # ===== summary =====
    print(f"\n=== Backfill summary ({len(df_out):,} alerts) ===")
    print(f"date range: {df_out['date'].min()} → {df_out['date'].max()}")
    print(f"\nrealized hit rate (overall):")
    for h in ["h2", "h6", "h5"]:
        col = f"hit_{h}"
        n = len(df_out)
        h_rate = float(df_out[col].mean()) * 100
        print(f"  hit_{h}: {h_rate:.2f}% ({int(df_out[col].sum())}/{n})")

    print(f"\nper-setup hit rate:")
    setups_seen = set()
    for s in df_out["setup_ids"].fillna(""):
        for s_id in s.split("+"):
            if s_id:
                setups_seen.add(s_id)
    for s_id in sorted(setups_seen):
        mask = df_out["setup_ids"].str.contains(s_id, na=False)
        n = int(mask.sum())
        if n == 0: continue
        sub = df_out[mask]
        print(f"  {s_id} (n={n}): hit_h2 {sub['hit_h2'].mean()*100:.1f}%, "
              f"hit_h6 {sub['hit_h6'].mean()*100:.1f}%, hit_h5 {sub['hit_h5'].mean()*100:.1f}%, "
              f"avg max_ret {sub['next_max_return_pct'].mean():.2f}%, "
              f"avg close_ret {sub['next_close_return_pct'].mean():.2f}%")

    print(f"\nper-regime hit rate:")
    for reg in df_out["btc_regime"].unique():
        mask = df_out["btc_regime"] == reg
        n = int(mask.sum())
        if n < 10: continue
        sub = df_out[mask]
        print(f"  {reg} (n={n}): hit_h2 {sub['hit_h2'].mean()*100:.1f}%, "
              f"hit_h6 {sub['hit_h6'].mean()*100:.1f}%, hit_h5 {sub['hit_h5'].mean()*100:.1f}%")


if __name__ == "__main__":
    main()

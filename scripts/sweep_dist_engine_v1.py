"""Sweep Distribution Engine v1 — head × regime × universe × topK.

목적 (사용자 A 단계):
  각 head별:
    - regime: all / bull_all / bull_quiet / bull_volatile / bear_all
    - universe: all / top50 / top100
    - topK_pct: 0.1 / 0.5 / 1 / 2 (h5 처럼 sweet spot 찾기)
  per cell:
    base_rate, precision, lift, n_alerts, active_days, avg_fail_eod

특히 확인:
  - h2/h6 가 regime gate 필요한가? (base 높은 hit-type)
  - h5 가 p99.5~p99.95 어디가 최적인가? (top0.5 > top0.1 발견 추적)
  - regime gate 가 lift 더 올리는 head 식별

흐름:
  1. panel + labels (leak fix 적용)
  2. WF 5-fold per head → val OOF score 누적
  3. (head × regime × universe × topK) 격자 평가
  4. CSV + 핵심 표 출력

사용:
    python scripts/sweep_dist_engine_v1.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.database import list_markets, load_candles
from data.market_universe import signal_eligible_markets
from signals.features import assemble_training_panel, compute_btc_features
from signals.labels_distribution import HEADS, MAX_BARS, compute_distribution_labels
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

REGIME_SUBSETS = {
    "all":           None,
    "bull_all":      ["bull_quiet", "bull_volatile"],
    "bull_quiet":    ["bull_quiet"],
    "bull_volatile": ["bull_volatile"],
    "bear_all":      ["bear_quiet", "bear_volatile"],
}
LIQ_TIERS = ["all", "top100", "top50"]
TOPK_PCTS = [0.001, 0.005, 0.01, 0.02]  # 0.1 / 0.5 / 1 / 2 %


def feature_cols_from(df, label_cols):
    EXTRA_DROP = {"quote_volume_d1", "date_only", "liq_rank_daily",
                   "highs", "lows", "n_bars", "btc_regime_4h", "label_date"}
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


def build_4h_panel_for_labels(candles_4h: dict, btc_regime_map: dict) -> pd.DataFrame:
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
                "eod_ret_4h": float(closes[-1]) / float(opens[0]) - 1 if opens[0] > 0 else np.nan,
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
    parser.add_argument("--out-csv", default="output/dist_engine_v1_sweep.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("sweep")

    print(f"=== Distribution Engine v1 — sweep (head × regime × universe × topK) ===\n")

    # data
    log.info("loading daily + 4h...")
    krw = signal_eligible_markets(list_markets(args.upbit_d1))
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

    krw_4h = [
        m
        for m in signal_eligible_markets(list_markets(args.upbit_4h))
        if m.startswith("KRW-")
    ]
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
    # leak fix: feature_date = label_date - 1day
    label_df["date_only"] = (pd.to_datetime(label_df["label_date"]) - pd.Timedelta(days=1)).dt.date

    full = panel.merge(
        label_df.drop(columns=["label_date"]),
        on=["market", "date_only"], how="inner"
    )
    log.info(f"  joined: {full.shape}")

    full["liq_rank_daily"] = full.groupby("date_only")["quote_volume_d1"].rank(
        method="dense", ascending=False, na_option="bottom"
    )

    feature_cols = feature_cols_from(full, list(HEADS.keys()) + ["eod_ret_next"])
    log.info(f"  features: {len(feature_cols)}")

    # WF + per-head val OOF score
    log.info("WF 5-fold per head...")
    splitter = PurgedWalkForward(args.n_folds, args.embargo, args.holdout)
    fold_data = []
    for fold, (train_dates, val_dates) in enumerate(splitter.split(full["timestamp"]), 1):
        train_p = full[full["timestamp"].isin(train_dates)]
        val_p = full[full["timestamp"].isin(val_dates)].copy()
        if len(train_p) < 100 or len(val_p) < 50:
            continue
        log.info(f"Fold {fold}: train {len(train_p):,} / val {len(val_p):,}")
        fold_data.append((fold, train_p, val_p))

    # accumulate val OOF: for each row in val (across folds, no overlap), per head score
    oof_rows = []
    for head_name in HEADS:
        log.info(f"  head: {head_name}")
        for fold, train_p, val_p in fold_data:
            df_tr = train_p[train_p[head_name].notna()].copy()
            df_va = val_p[val_p[head_name].notna()].copy()
            if len(df_tr) < 100:
                continue
            X_tr = df_tr[feature_cols].astype(float).values
            y_tr = df_tr[head_name].astype(int).values
            if y_tr.sum() < 10 or (y_tr == 0).sum() < 10:
                continue
            X_va = df_va[feature_cols].astype(float).values
            m = train_binary(X_tr, y_tr)
            scores = m.predict_proba(X_va)[:, 1]
            for i, (_, r) in enumerate(df_va.iterrows()):
                oof_rows.append({
                    "head": head_name,
                    "fold": fold,
                    "market": r["market"],
                    "date_only": r["date_only"],
                    "btc_regime": r.get("btc_regime", "unknown"),
                    "liq_rank_daily": r.get("liq_rank_daily", np.nan),
                    "label": r[head_name],
                    "score": scores[i],
                    "eod_ret_next": r.get("eod_ret_next", np.nan),
                })

    oof = pd.DataFrame(oof_rows)
    log.info(f"oof: {oof.shape}")

    # Sweep
    log.info("sweeping head × regime × universe × topK...")
    sweep_rows = []
    for head_name in HEADS:
        oof_h = oof[oof["head"] == head_name]
        if len(oof_h) == 0:
            continue
        for regime_label, regs in REGIME_SUBSETS.items():
            if regs is None:
                m_reg = np.ones(len(oof_h), dtype=bool)
            else:
                m_reg = oof_h["btc_regime"].isin(regs).values
            for liq_label in LIQ_TIERS:
                if liq_label == "all":
                    m_liq = np.ones(len(oof_h), dtype=bool)
                else:
                    n = int(liq_label.replace("top", ""))
                    m_liq = (oof_h["liq_rank_daily"] <= n).values
                m_combo = m_reg & m_liq
                sub = oof_h[m_combo]
                n_sub = len(sub)
                if n_sub < 50:
                    continue
                base = float(sub["label"].mean())
                if base == 0:
                    continue
                # active days = unique dates in sub
                total_days = sub["date_only"].nunique()
                for pct in TOPK_PCTS:
                    n_top = max(1, int(n_sub * pct))
                    top = sub.nlargest(n_top, "score")
                    prec = float(top["label"].mean())
                    lift = prec / base if base > 0 else np.nan
                    fails = top[top["label"] == 0]
                    avg_fail_eod = float(fails["eod_ret_next"].mean() * 100) if len(fails) > 0 else np.nan
                    worst5_fail_eod = float(fails["eod_ret_next"].quantile(0.05) * 100) if len(fails) >= 20 else np.nan
                    active_days = top["date_only"].nunique()
                    sweep_rows.append({
                        "head": head_name,
                        "regime": regime_label,
                        "liq": liq_label,
                        "topK_pct": pct * 100,
                        "n_sub": n_sub,
                        "base_rate_pct": base * 100,
                        "n_top": n_top,
                        "precision_pct": prec * 100,
                        "lift": lift,
                        "active_days": active_days,
                        "total_days": total_days,
                        "active_pct_of_period": active_days / total_days * 100 if total_days > 0 else 0,
                        "avg_fail_eod_pct": avg_fail_eod,
                        "worst5_fail_eod_pct": worst5_fail_eod,
                    })

    df = pd.DataFrame(sweep_rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    log.info(f"saved {args.out_csv}")

    # ========================================================================
    # 핵심 출력
    # ========================================================================
    show_cols = ["head", "regime", "liq", "topK_pct", "n_sub", "base_rate_pct",
                 "precision_pct", "lift", "active_days", "active_pct_of_period",
                 "avg_fail_eod_pct"]

    def show(title, df_sel):
        print(f"\n=== {title} ===")
        if len(df_sel) == 0:
            print("(no rows)")
            return
        print(df_sel[show_cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # 1. 각 head 별 best (lift) 조합
    print(f"\n{'='*130}")
    print("1. Head 별 best lift (top liq=all, all regime/topK 비교)")
    for head_name in HEADS:
        sub = df[(df["head"] == head_name) & (df["liq"] == "all")].sort_values("lift", ascending=False)
        show(f"{head_name}", sub.head(8))

    # 2. h5 sweet spot 검증 (top0.1~top2)
    print(f"\n{'='*130}")
    show("2. h5 tail sweet spot — bull_all, top100",
         df[(df["head"] == "h5_tail_20") & (df["regime"] == "bull_all") &
            (df["liq"] == "top100")].sort_values("topK_pct"))
    show("2b. h5 tail — all regime, all liq (top0.1~top2)",
         df[(df["head"] == "h5_tail_20") & (df["regime"] == "all") &
            (df["liq"] == "all")].sort_values("topK_pct"))

    # 3. h2/h6 regime gate 효과 — all vs bull_all 비교
    print(f"\n{'='*130}")
    print("3. h2/h6 regime gate 필요한가?")
    for h in ["h2_hit_3_4h", "h6_hit_5_24h"]:
        show(f"  {h} — top0.5 across regimes (liq=all)",
             df[(df["head"] == h) & (df["topK_pct"] == 0.5) & (df["liq"] == "all")])

    # 4. all heads × bull_all × top100 × top0.5 (운영 후보 한 큐)
    print(f"\n{'='*130}")
    show("4. 운영 후보 시뮬 — all heads × bull_all × top100 × topK varied",
         df[(df["regime"] == "bull_all") & (df["liq"] == "top100") &
            (df["topK_pct"].isin([0.1, 0.5, 1.0]))].sort_values(["head", "topK_pct"]))


if __name__ == "__main__":
    main()

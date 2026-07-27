"""Pre-open Trigger Discovery — Phase A: 4h precursor feature 빠른 가설 검증.

가설:
  KST 08:30 직전 sub-daily 흐름이 오늘 09:00 이후 펌프를 예고.

Phase A 한계:
  4h 데이터로는 "08:30 직전" 정확 재현 불가.
  T-3.5h precursor (= bar at KST 01-05, closed at 05:00) 가 최선.
  실제 사용자 가설 검증은 Phase B (1h collector) 부터.

Phase A 의 역할:
  - 신규 데이터 X, 가설의 방향성 빠르게 확인
  - 4h precursor 도 lift 양수 → Phase B 강력 추진
  - 4h 음수 → 그래도 Phase B 한 번 시도 (4h 가 거칠어서일 수 있음)

설계:
  panel row 일자 X 의 daily features (existing) + bar_5 (= bar at KST 01-05 of X+1) precursor features.
  target = h2/h5/h6 labels for daily X+1 (after leak fix, same as existing).

precursor features (bar_5 = T-3.5h before pump trigger 시점):
  pre_vol_ratio       : bar_5.vol / mean(bar_1..4 vol)
  pre_price_change    : bar_5.close / bar_5.open - 1
  pre_range_pct       : (bar_5.high - bar_5.low) / bar_5.open
  pre_close_position  : bar_5.close 위치 (0=low, 1=high)
  pre_day_to_bar5_max : daily X 의 09:00 → 05:00 X+1 까지 max 상승률
  pre_day_to_bar5_dd  : daily X 의 09:00 → 05:00 X+1 까지 max 낙폭
  pre_bar5_close_vs_day_high : 직전 bar 의 close 가 day 의 high 근처인지

비교:
  Model A (baseline) : existing daily features only (= distribution engine v1)
  Model B (precursor): A + bar_5 precursor features
  per head h2/h5/h6  : lift @ top0.5%, top1% — A vs B

사용:
    python scripts/precursor_4h_v0.py
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
from data.market_universe import signal_eligible_markets
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

TARGET_HEADS = ["h2_hit_3_4h", "h5_tail_20", "h6_hit_5_24h"]
PRECURSOR_FEATURES = [
    "pre_vol_ratio",
    "pre_price_change",
    "pre_range_pct",
    "pre_close_position",
    "pre_day_to_bar5_max",
    "pre_day_to_bar5_dd",
    "pre_bar5_close_vs_day_high",
]
TOPK_PCTS = [0.005, 0.01, 0.02]


def build_4h_panel(candles_4h, btc_regime_map):
    """4h bars per (market, bar_date) → padded arrays + precursor features."""
    rows = []
    for market, df in candles_4h.items():
        if df is None or len(df) < 1:
            continue
        df = df.sort_values("timestamp").copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["bar_date"] = (df["timestamp"] - pd.Timedelta(hours=9)).dt.date
        for date, g in df.groupby("bar_date", sort=False):
            g2 = g.sort_values("timestamp").reset_index(drop=True)
            n = min(len(g2), MAX_BARS)
            if n < 5:
                # need at least bar_5 to compute precursor
                continue
            opens = g2["open"].values.astype(float)
            closes = g2["close"].values.astype(float)
            highs = g2["high"].values.astype(float)
            lows = g2["low"].values.astype(float)
            vols = g2["volume"].values.astype(float) if "volume" in g2.columns else np.zeros(n)
            highs_p = np.full(MAX_BARS, np.nan)
            lows_p = np.full(MAX_BARS, np.nan)
            highs_p[:n] = highs[:n]
            lows_p[:n] = lows[:n]

            # bar_5 (index 4) = covers 01-05 KST of next day, closed at 05:00 next day
            # T-3.5h precursor for 09:00 next day prediction
            day_open = float(opens[0])
            bar5 = {
                "open": float(opens[4]), "close": float(closes[4]),
                "high": float(highs[4]), "low": float(lows[4]),
                "volume": float(vols[4]),
            }
            bar1_4_vols = vols[:4]
            mean_vol_1_4 = float(bar1_4_vols.mean()) if len(bar1_4_vols) > 0 else 1e-9
            day_max_to_bar5 = float(highs[:5].max())
            day_min_to_bar5 = float(lows[:5].min())

            pre_vol_ratio = bar5["volume"] / max(mean_vol_1_4, 1e-9)
            pre_price_change = bar5["close"] / max(bar5["open"], 1e-9) - 1
            pre_range_pct = (bar5["high"] - bar5["low"]) / max(bar5["open"], 1e-9)
            denom_pos = max(bar5["high"] - bar5["low"], 1e-9)
            pre_close_position = (bar5["close"] - bar5["low"]) / denom_pos
            pre_day_to_bar5_max = day_max_to_bar5 / max(day_open, 1e-9) - 1
            pre_day_to_bar5_dd = day_min_to_bar5 / max(day_open, 1e-9) - 1
            pre_bar5_close_vs_day_high = bar5["close"] / max(day_max_to_bar5, 1e-9) - 1

            rows.append({
                "market": market,
                "date_only": date,
                "open_4h": float(opens[0]),
                "close_4h": float(closes[-1]),
                "highs": highs_p, "lows": lows_p,
                "btc_regime_4h": btc_regime_map.get(date, "unknown"),
                "n_bars": n,
                "pre_vol_ratio": pre_vol_ratio,
                "pre_price_change": pre_price_change,
                "pre_range_pct": pre_range_pct,
                "pre_close_position": pre_close_position,
                "pre_day_to_bar5_max": pre_day_to_bar5_max,
                "pre_day_to_bar5_dd": pre_day_to_bar5_dd,
                "pre_bar5_close_vs_day_high": pre_bar5_close_vs_day_high,
            })
    return pd.DataFrame(rows)


def feature_cols_baseline(df, label_cols):
    EXTRA_DROP = {"quote_volume_d1", "date_only", "liq_rank_daily",
                   "highs", "lows", "n_bars", "btc_regime_4h", "label_date",
                   "eod_ret_next", "max_ret_next"} | set(PRECURSOR_FEATURES)
    cols = []
    for c in df.columns:
        if c in EXCLUDE_COLS or c in label_cols or c in HEADS:
            continue
        if c in LEAK_COLS or c in EXTRA_DROP:
            continue
        if c.startswith("next_"):
            continue
        dt = df[c].dtype
        if pd.api.types.is_object_dtype(dt) or "datetime" in str(dt):
            continue
        cols.append(c)
    return cols


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
    parser.add_argument("--upbit-4h", default="data/upbit_4h.db")
    parser.add_argument("--binance-d1", default="data/binance_d1.db")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--embargo", type=int, default=10)
    parser.add_argument("--holdout", type=int, default=180)
    parser.add_argument("--out-csv", default="output/precursor_4h_v0.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("precursor")

    print("=== Phase A — 4h precursor (T-3.5h, bar_5 KST 01-05) quick hypothesis test ===\n")
    print("Phase A 의 의미: 신규 데이터 X. 음수여도 Phase B 진입 폐기 X.")
    print(f"Heads: {TARGET_HEADS}")
    print(f"Precursor features ({len(PRECURSOR_FEATURES)}): {PRECURSOR_FEATURES}\n")

    # data
    log.info("loading panel + labels...")
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
    panel_4h = build_4h_panel(candles_4h, btc_regime_map)
    log.info(f"  4h panel rows: {len(panel_4h)}")

    # === LEAK FIX ===
    # label: panel_4h.date_only X 의 bars 로 측정 → leak 방지 위해 -1day shift (panel row X-1 에 attach)
    label_df = compute_distribution_labels(panel_4h)
    label_df["market"] = panel_4h["market"].values
    label_df["label_date"] = panel_4h["date_only"].values
    label_df["date_only"] = (pd.to_datetime(label_df["label_date"]) - pd.Timedelta(days=1)).dt.date

    # precursor: panel_4h.date_only X 의 bar_5 로 측정 → panel row X (= 같은 day) 에 attach
    # = 다음 09:00 (= X+1 day 시작) 예측 시 어제 late intraday 정보. SHIFT 없음.
    precursor_df = panel_4h[["market", "date_only"] + PRECURSOR_FEATURES].copy()
    log.info(f"  precursor df: {precursor_df.shape}, label df: {label_df.shape}")

    full = panel.merge(precursor_df, on=["market", "date_only"], how="inner")
    full = full.merge(label_df.drop(columns=["label_date"]),
                      on=["market", "date_only"], how="inner")
    log.info(f"  joined: {full.shape}")

    base_cols = feature_cols_baseline(full, list(HEADS.keys()))
    plus_cols = base_cols + PRECURSOR_FEATURES
    log.info(f"  baseline features: {len(base_cols)}, +precursor: {len(plus_cols)}")

    # WF
    splitter = PurgedWalkForward(args.n_folds, args.embargo, args.holdout)
    fold_data = []
    for fold, (train_dates, val_dates) in enumerate(splitter.split(full["timestamp"]), 1):
        train_p = full[full["timestamp"].isin(train_dates)]
        val_p = full[full["timestamp"].isin(val_dates)].copy()
        if len(train_p) < 100 or len(val_p) < 50:
            continue
        log.info(f"Fold {fold}: train {len(train_p):,} / val {len(val_p):,}")
        fold_data.append((fold, train_p, val_p))

    rows = []
    for head in TARGET_HEADS:
        log.info(f"\n--- head: {head} ---")
        for fold, train_p, val_p in fold_data:
            df_tr = train_p[train_p[head].notna()].copy()
            df_va = val_p[val_p[head].notna()].copy()
            if len(df_tr) < 100:
                continue
            y_tr = df_tr[head].astype(int).values
            y_va = df_va[head].astype(int).values
            if y_tr.sum() < 10 or (y_tr == 0).sum() < 10:
                continue

            # Model A — baseline only
            t0 = time.time()
            X_tr_a = df_tr[base_cols].astype(float).values
            X_va_a = df_va[base_cols].astype(float).values
            m_a = train_binary(X_tr_a, y_tr)
            sc_a = m_a.predict_proba(X_va_a)[:, 1]

            # Model B — baseline + precursor
            X_tr_b = df_tr[plus_cols].astype(float).values
            X_va_b = df_va[plus_cols].astype(float).values
            m_b = train_binary(X_tr_b, y_tr)
            sc_b = m_b.predict_proba(X_va_b)[:, 1]

            for pct in TOPK_PCTS:
                a = lift_at_topk(sc_a, y_va, pct)
                b = lift_at_topk(sc_b, y_va, pct)
                if a is None or b is None:
                    continue
                rows.append({
                    "head": head, "fold": fold, "topK_pct": pct * 100,
                    "n_val": len(y_va),
                    "base_pct": a["base_pct"],
                    "A_baseline_lift": a["lift"], "A_baseline_prec_pct": a["prec_pct"],
                    "B_plus_precursor_lift": b["lift"], "B_plus_precursor_prec_pct": b["prec_pct"],
                    "delta_lift": (b["lift"] - a["lift"]) if (b["lift"] and a["lift"]) else None,
                    "delta_prec_pp": b["prec_pct"] - a["prec_pct"],
                })
            log.info(f"  fold {fold}: A vs B trained ({time.time()-t0:.1f}s)")

    df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    log.info(f"saved {args.out_csv}")

    # ===== summary =====
    print("\n" + "=" * 130)
    print("Per-head A (baseline) vs B (+precursor) — mean across folds")
    print("=" * 130)
    show = ["head", "topK_pct", "base_pct",
            "A_baseline_lift", "B_plus_precursor_lift", "delta_lift",
            "A_baseline_prec_pct", "B_plus_precursor_prec_pct", "delta_prec_pp"]
    summary = df.groupby(["head", "topK_pct"]).agg({
        "base_pct": "mean",
        "A_baseline_lift": "mean",
        "B_plus_precursor_lift": "mean",
        "delta_lift": "mean",
        "A_baseline_prec_pct": "mean",
        "B_plus_precursor_prec_pct": "mean",
        "delta_prec_pp": "mean",
    }).reset_index()
    print(summary[show].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n" + "=" * 130)
    print("Verdict (사용자 기준):")
    print("  - delta_lift > 0 면 4h precursor 도 lift 추가 → Phase B (1h collector) 강력 추진")
    print("  - delta_lift ≈ 0 또는 음수 면 4h 가 거칠어서일 수 있음 → 그래도 Phase B 한 번 시도")


if __name__ == "__main__":
    main()

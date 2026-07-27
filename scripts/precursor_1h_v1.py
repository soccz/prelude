"""Pre-open Trigger Discovery — Phase B v1: 1h precursor + cumulative day features.

v0 결과 (1h, 9 features, no cumulative):
  h2 +0.28~0.43, h5 +1.00~1.63, h6 +0.11~0.25
  Phase A (4h) 와 비슷하거나 약함. **공정 비교 X** — 4h 에 cumulative day
  features (day_to_bar5_max 등) 가 있었음. 1h v0 에는 없었음.

v1 변경 (사용자 권장):
  + 누적 day-progress features 추가 (4h 의 cumulative pattern 1h 로 이식):
    cum_max_ret_to_08h     : 09:00 ~ 08:00 누적 max return
    cum_dd_to_08h          : 09:00 ~ 08:00 누적 max drawdown
    close07_vs_cum_high    : 07-08 close 가 누적 high 근처인지
    close07_vs_day_open    : 07-08 close vs 일봉 시가
    range_5_to_8_expansion : 05-08 3h range / 1h ATR 비
    cum_volume_to_08h      : 09:00 ~ 08:00 누적 vol vs 24h 평균 sum

해석 기준:
  v1 1h 가 4h Phase A 보다 lift 좋음 → 사용자 가설 정당화 → 15m 진행
  v1 1h 도 4h 와 비슷 → 4h 가 sweet spot, finer scale 가치 X

Phase B v1 가설:
  KST 08:00 snapshot (= bar at X+1 07:00 closed at 08:00) 의 precursor 가
  X+1 day 의 펌프 (h2/h5/h6) lift 추가하는가?

설계:
  panel row date X (features = X day daily candle close)
  + 1h precursor from bar at "X+1 07:00" (covers KST 07-08 of X+1, closed 08:00)
  → predict X+1 daily candle's h2/h5/h6 (after leak fix join)

  precursor 와 panel 모두 "X 날 까지" 의 정보 → leak-free
  precursor 의 bar_date = X (= panel.date_only) → no shift

precursor features (08:00 snapshot, T-1h before 09:00 prediction):
  pre1h_ret              : 07-08 1h return
  pre1h_vol_ratio        : 07-08 1h vol / mean(prior 24h vol)
  pre1h_range_pct        : (high-low)/open
  pre1h_close_position   : (close-low)/(high-low)
  pre3h_ret              : 05-08 3h return (07:00 close / 04:00 open)
  pre3h_vol_ratio        : sum(05-06, 06-07, 07-08 vol) / mean(prior 24h vol) × 3
  breakout_24h_high      : 07-08 close / max(prior 24h high) - 1
  pre1h_btc_ret          : BTC 1h return (07-08)
  pre1h_breadth          : % of KRW coins with positive 1h return

비교:
  Model A: baseline daily features only (existing distribution engine)
  Model B: baseline + 1h precursor features
  per head h2/h5/h6: lift @ top0.5%/1%/2% — A vs B

사용:
    python scripts/precursor_1h_v0.py
    (선결: data/upbit_1h.db 백필 완료)
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
    # micro (v0 retained)
    "pre1h_ret", "pre1h_vol_ratio", "pre1h_range_pct", "pre1h_close_position",
    "pre3h_ret", "pre3h_vol_ratio", "breakout_24h_high",
    "pre1h_btc_ret", "pre1h_breadth",
    # cumulative day (v1 added — 4h Phase A 의 cumulative pattern 이식)
    "cum_max_ret_to_08h", "cum_dd_to_08h",
    "close07_vs_cum_high", "close07_vs_day_open",
    "range_5_to_8_expansion", "cum_volume_to_08h",
]
TOPK_PCTS = [0.005, 0.01, 0.02]


def build_1h_precursor(candles_1h: dict, btc_df: pd.DataFrame) -> pd.DataFrame:
    """For each (market, bar_date X), compute features from 08:00 snapshot.

    bar at "X+1 07:00" (timestamp) covers KST 07-08 of X+1, closed 08:00.
    bar_date X (since (X+1 07:00 - 9h).date() = X with KST naive timestamp).
    """
    # BTC 1h returns (per timestamp)
    btc = btc_df.sort_values("timestamp").copy()
    btc["timestamp"] = pd.to_datetime(btc["timestamp"])
    btc["btc_1h_ret"] = btc["close"] / btc["close"].shift(1) - 1
    btc_ret_map = dict(zip(btc["timestamp"], btc["btc_1h_ret"]))

    # First pass: compute per-market per-bar_date features (no breadth yet)
    rows = []
    for market, df in candles_1h.items():
        if df is None or len(df) < 30:
            continue
        df = df.sort_values("timestamp").copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["bar_date"] = (df["timestamp"] - pd.Timedelta(hours=9)).dt.date

        # rolling 24h prior vol mean (exclusive of current bar)
        df["vol_24h_mean_prior"] = df["volume"].rolling(24, min_periods=12).mean().shift(1)
        df["high_24h_prior"] = df["high"].rolling(24, min_periods=12).max().shift(1)

        for date, g in df.groupby("bar_date", sort=False):
            g = g.sort_values("timestamp").reset_index(drop=True)
            target_ts_date = pd.Timestamp(date) + pd.Timedelta(days=1)
            target_ts = target_ts_date.replace(hour=7)
            bar = g[g["timestamp"] == target_ts]
            if len(bar) == 0:
                continue
            r = bar.iloc[0]
            # 3h aggregate: bars at 05, 06, 07 of next-day calendar (= last 3 1h bars before 08:00)
            three_h = g[(g["timestamp"].dt.hour.isin([5, 6, 7])) &
                         (g["timestamp"].dt.date == target_ts_date.date())]
            if len(three_h) < 1:
                continue

            opn = float(r["open"]) if r["open"] > 0 else np.nan
            cls = float(r["close"])
            hi = float(r["high"])
            lo = float(r["low"])
            vol = float(r["volume"])
            vol_24h_mean = float(r["vol_24h_mean_prior"])
            high_24h_prior = float(r["high_24h_prior"])

            # micro features (v0)
            pre1h_ret = (cls / opn - 1) if pd.notna(opn) else np.nan
            pre1h_vol_ratio = (vol / vol_24h_mean) if (pd.notna(vol_24h_mean) and vol_24h_mean > 0) else np.nan
            pre1h_range_pct = ((hi - lo) / opn) if (pd.notna(opn) and opn > 0) else np.nan
            denom = max(hi - lo, 1e-9)
            pre1h_close_position = (cls - lo) / denom

            three_h_first_open = float(three_h.iloc[0]["open"]) if three_h.iloc[0]["open"] > 0 else np.nan
            pre3h_ret = (cls / three_h_first_open - 1) if pd.notna(three_h_first_open) else np.nan
            three_h_vol_sum = float(three_h["volume"].sum())
            pre3h_vol_ratio = (three_h_vol_sum / (vol_24h_mean * 3)) if (pd.notna(vol_24h_mean) and vol_24h_mean > 0) else np.nan

            breakout_24h_high = (cls / high_24h_prior - 1) if (pd.notna(high_24h_prior) and high_24h_prior > 0) else np.nan
            pre1h_btc_ret = btc_ret_map.get(target_ts, np.nan)

            # === v1 added: cumulative day-progress (09:00 of X day → 08:00 of X+1) ===
            # bars in current bar_date X up to and including target_ts (= bars 0..22 = 23 hours)
            day_bars_until_08h = g[g["timestamp"] <= target_ts]
            if len(day_bars_until_08h) >= 1:
                day_open = float(day_bars_until_08h.iloc[0]["open"]) if day_bars_until_08h.iloc[0]["open"] > 0 else np.nan
                cum_high = float(day_bars_until_08h["high"].max())
                cum_low = float(day_bars_until_08h["low"].min())
                cum_vol = float(day_bars_until_08h["volume"].sum())
            else:
                day_open = np.nan
                cum_high = np.nan
                cum_low = np.nan
                cum_vol = np.nan

            cum_max_ret_to_08h = (cum_high / day_open - 1) if (pd.notna(day_open) and day_open > 0) else np.nan
            cum_dd_to_08h = (cum_low / day_open - 1) if (pd.notna(day_open) and day_open > 0) else np.nan
            close07_vs_cum_high = (cls / cum_high - 1) if (pd.notna(cum_high) and cum_high > 0) else np.nan
            close07_vs_day_open = (cls / day_open - 1) if (pd.notna(day_open) and day_open > 0) else np.nan
            range_5_to_8_expansion = ((three_h["high"].max() - three_h["low"].min()) / opn) \
                if (pd.notna(opn) and opn > 0 and len(three_h) > 0) else np.nan
            cum_volume_to_08h = (cum_vol / (vol_24h_mean * 23)) \
                if (pd.notna(vol_24h_mean) and vol_24h_mean > 0) else np.nan

            rows.append({
                "market": market, "date_only": date, "ts": target_ts,
                "pre1h_ret": pre1h_ret,
                "pre1h_vol_ratio": pre1h_vol_ratio,
                "pre1h_range_pct": pre1h_range_pct,
                "pre1h_close_position": pre1h_close_position,
                "pre3h_ret": pre3h_ret,
                "pre3h_vol_ratio": pre3h_vol_ratio,
                "breakout_24h_high": breakout_24h_high,
                "pre1h_btc_ret": pre1h_btc_ret,
                # v1 cumulative
                "cum_max_ret_to_08h": cum_max_ret_to_08h,
                "cum_dd_to_08h": cum_dd_to_08h,
                "close07_vs_cum_high": close07_vs_cum_high,
                "close07_vs_day_open": close07_vs_day_open,
                "range_5_to_8_expansion": range_5_to_8_expansion,
                "cum_volume_to_08h": cum_volume_to_08h,
            })
    df_pre = pd.DataFrame(rows)
    if len(df_pre) == 0:
        return df_pre

    # Second pass: breadth = per-ts fraction of markets with pre1h_ret > 0
    df_pre["breadth_indicator"] = (df_pre["pre1h_ret"] > 0).astype(float)
    breadth = df_pre.groupby("ts")["breadth_indicator"].mean().rename("pre1h_breadth")
    df_pre = df_pre.merge(breadth, on="ts", how="left").drop(columns=["breadth_indicator", "ts"])
    return df_pre


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


def build_4h_panel_for_labels(candles_4h, btc_regime_map):
    """기존 distribution engine 의 label panel 빌드 (precursor_4h_v0 와 동일)."""
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
                "market": market, "date_only": date,
                "open_4h": float(opens[0]), "close_4h": float(closes[-1]),
                "highs": highs_p, "lows": lows_p,
                "btc_regime_4h": btc_regime_map.get(date, "unknown"),
                "n_bars": n,
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upbit-d1", default="data/upbit_d1.db")
    parser.add_argument("--upbit-4h", default="data/upbit_4h.db")
    parser.add_argument("--upbit-1h", default="data/upbit_1h.db")
    parser.add_argument("--binance-d1", default="data/binance_d1.db")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--embargo", type=int, default=10)
    parser.add_argument("--holdout", type=int, default=180)
    parser.add_argument("--out-csv", default="output/precursor_1h_v1.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("precursor1h")

    print("=== Phase B — 1h Upbit precursor (T-1h, bar @ KST 07-08) ===\n")
    print("As-of: KST 08:00 (closed 07-08 1h bar 사용)")
    print(f"Heads: {TARGET_HEADS}")
    print(f"Precursor features ({len(PRECURSOR_FEATURES)}): {PRECURSOR_FEATURES}\n")

    # === 1) daily panel + label ===
    log.info("loading daily panel...")
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

    # === 2) labels (4h based, leak fix) ===
    log.info("loading 4h candles + computing labels...")
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
    label_df["date_only"] = (pd.to_datetime(label_df["label_date"]) - pd.Timedelta(days=1)).dt.date

    # === 3) 1h precursor ===
    log.info("loading 1h candles + computing precursor (08:00 snapshot)...")
    krw_1h = [
        m
        for m in signal_eligible_markets(list_markets(args.upbit_1h))
        if m.startswith("KRW-")
    ]
    candles_1h = {m: load_candles(args.upbit_1h, m) for m in krw_1h}
    candles_1h = {k: v for k, v in candles_1h.items() if v is not None and len(v) > 30}
    log.info(f"  1h markets: {len(candles_1h)}")
    btc_1h = candles_1h.get("KRW-BTC", pd.DataFrame())
    precursor_df = build_1h_precursor(candles_1h, btc_1h)
    log.info(f"  precursor rows: {len(precursor_df)}")

    # === 4) merge: panel + precursor (no shift) + label (shifted) ===
    full = panel.merge(precursor_df, on=["market", "date_only"], how="inner")
    full = full.merge(label_df.drop(columns=["label_date"]),
                      on=["market", "date_only"], how="inner")
    log.info(f"  joined: {full.shape}")

    base_cols = feature_cols_baseline(full, list(HEADS.keys()))
    plus_cols = base_cols + PRECURSOR_FEATURES
    log.info(f"  baseline features: {len(base_cols)}, +precursor: {len(plus_cols)}")

    # === 5) WF: A vs B ===
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

            t0 = time.time()
            X_tr_a = df_tr[base_cols].astype(float).values
            X_va_a = df_va[base_cols].astype(float).values
            m_a = train_binary(X_tr_a, y_tr)
            sc_a = m_a.predict_proba(X_va_a)[:, 1]

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

    print("\n" + "=" * 130)
    print("Per-head A (baseline daily) vs B (+1h precursor) — mean across folds")
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
    print("비교 vs Phase A (4h, T-3.5h):")
    print("  Phase A: h2 +0.53, h5 +1.94, h6 ≈ 0 (top0.5%)")
    print("  Phase B (이번): 위 표")
    print("  → Phase B 가 더 큰 lift 면 1h scale 이 사용자 가설 (08:30 trigger) 의미 있음 → Phase C (15m) 정당화")


if __name__ == "__main__":
    main()

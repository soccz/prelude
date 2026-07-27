"""Common-period precursor ablation.

목적:
  Phase A/B/C 결과를 공정하게 비교한다. 기존 비교는 4h/1h 는 긴 기간,
  15m 는 최근 약 1년만 사용해서 h5 rare tail 비교에 sample-size confound 가 컸다.

이 스크립트는 4h, 1h, 15m precursor 가 모두 존재하는 동일 row 집합에서
다음 feature set 을 같은 fold 로 비교한다.

  - daily
  - daily + 4h
  - daily + 1h
  - daily + 15m
  - daily + 1h + 15m
  - daily + 4h + 1h + 15m

출력:
  - per-fold raw rows: output/ablation_common_period_v1.csv
  - weighted summary: stdout

주의:
  production 알림 로직을 바꾸지 않는다. 이건 algorithm audit 전용이다.
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
from signals.labels_distribution import HEADS, compute_distribution_labels
from signals.models.xgb_phase1 import EXCLUDE_COLS
from signals.precursors import (
    FIFTEEN_M_FEATURES as PRE15M,
    FOUR_H_FEATURES as PRE4H,
    ONE_H_FEATURES as PRE1H,
    build_15m_precursor,
    build_1h_precursor,
    build_4h_label_panel,
    build_4h_precursor,
    cached_frame,
    prefix_columns,
)
from signals.validate import PurgedWalkForward


TARGET_HEADS = ["h2_hit_3_4h", "h5_tail_20", "h6_hit_5_24h"]
TOPK_PCTS = [0.005, 0.01, 0.02]

LEAK_COLS = {
    "net_under_tp", "max_return", "label", "label_tail",
    "next_open", "next_high", "next_low", "next_close",
    "next_max_return", "next_eod_return", "next_max_dd",
}

XGB_PARAMS = dict(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=350,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
)


def feature_cols_daily(df: pd.DataFrame, label_cols: list[str], precursor_cols: set[str]) -> list[str]:
    drop = {
        "quote_volume_d1", "date_only", "liq_rank_daily",
        "highs", "lows", "n_bars", "btc_regime_4h", "label_date",
        "eod_ret_next", "max_ret_next",
    } | precursor_cols
    cols: list[str] = []
    for c in df.columns:
        if c in EXCLUDE_COLS or c in label_cols or c in HEADS:
            continue
        if c in LEAK_COLS or c in drop:
            continue
        if c.startswith("next_"):
            continue
        dt = df[c].dtype
        if pd.api.types.is_object_dtype(dt) or "datetime" in str(dt):
            continue
        cols.append(c)
    return cols


def train_binary(X: np.ndarray, y: np.ndarray) -> xgb.XGBClassifier:
    sw = compute_sample_weight(class_weight="balanced", y=y)
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X, y, sample_weight=sw, verbose=False)
    return model


def eval_topk(scores: np.ndarray, labels: np.ndarray, pct: float) -> dict | None:
    if len(scores) == 0:
        return None
    pos = int(labels.sum())
    if pos == 0:
        return None
    n_top = max(1, int(len(scores) * pct))
    idx = np.argsort(-scores)[:n_top]
    top_pos = int(labels[idx].sum())
    base = pos / len(labels)
    prec = top_pos / n_top
    return {
        "n_val": int(len(labels)),
        "n_pos": pos,
        "n_top": int(n_top),
        "top_pos": top_pos,
        "base_pct": base * 100,
        "prec_pct": prec * 100,
        "lift": (prec / base) if base > 0 else np.nan,
    }


def weighted_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (head, feature_set, top), g in df.groupby(["head", "feature_set", "topK_pct"], sort=False):
        n_val = int(g["n_val"].sum())
        n_pos = int(g["n_pos"].sum())
        n_top = int(g["n_top"].sum())
        top_pos = int(g["top_pos"].sum())
        base = n_pos / n_val if n_val else np.nan
        prec = top_pos / n_top if n_top else np.nan
        rows.append({
            "head": head,
            "feature_set": feature_set,
            "topK_pct": top,
            "folds": int(g["fold"].nunique()),
            "n_val": n_val,
            "base_pct": base * 100,
            "prec_pct": prec * 100,
            "lift": (prec / base) if base > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def load_daily_panel(upbit_d1: str, binance_d1: str) -> pd.DataFrame:
    krw = signal_eligible_markets(list_markets(upbit_d1))
    candles_d1 = {m: load_candles(upbit_d1, m) for m in krw}
    if Path(binance_d1).exists():
        for m in list_markets(binance_d1):
            candles_d1[m] = load_candles(binance_d1, m)
    candles_d1 = {k: v for k, v in candles_d1.items() if v is not None and len(v) > 30}
    btc_d1 = load_candles(upbit_d1, "KRW-BTC")
    panel = assemble_training_panel(candles_d1, btc_d1, normalize=True)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values(["market", "timestamp"]).reset_index(drop=True)
    panel["date_only"] = panel["timestamp"].dt.date
    panel["quote_volume_d1"] = panel.get("quote_volume", np.nan)
    return panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upbit-d1", default="data/upbit_d1.db")
    parser.add_argument("--upbit-4h", default="data/upbit_4h.db")
    parser.add_argument("--upbit-1h", default="data/upbit_1h.db")
    parser.add_argument("--upbit-15m", default="data/upbit_15m.db")
    parser.add_argument("--binance-d1", default="data/binance_d1.db")
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--embargo", type=int, default=10)
    parser.add_argument("--holdout", type=int, default=0)
    parser.add_argument("--min-rows-per-date", type=int, default=100)
    parser.add_argument("--cache-dir", default="output/cache")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--out-csv", default="output/ablation_common_period_v1.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("ablation")

    print("=== Common-period ablation v1 ===")
    print("feature sets: daily / +4h / +1h / +15m / +1h+15m / all")
    print(f"heads: {TARGET_HEADS}\n")

    log.info("loading daily panel...")
    panel = load_daily_panel(args.upbit_d1, args.binance_d1)
    btc_d1 = load_candles(args.upbit_d1, "KRW-BTC")
    btc_feat = compute_btc_features(btc_d1.copy())
    btc_feat["date_only"] = pd.to_datetime(btc_feat["timestamp"]).dt.date
    btc_regime_map = dict(zip(btc_feat["date_only"], btc_feat["btc_regime"]))
    cache_dir = Path(args.cache_dir)

    log.info("loading 4h labels + 4h precursor...")
    krw_4h = [
        m
        for m in signal_eligible_markets(list_markets(args.upbit_4h))
        if m.startswith("KRW-")
    ]
    candles_4h = {m: load_candles(args.upbit_4h, m) for m in krw_4h}
    candles_4h = {k: v for k, v in candles_4h.items() if v is not None and len(v) > 0}
    panel_4h_label = cached_frame(
        cache_dir / "panel_4h_label.pkl",
        lambda: build_4h_label_panel(candles_4h, btc_regime_map),
        refresh=args.refresh_cache,
    )
    label_df = compute_distribution_labels(panel_4h_label)
    label_df["market"] = panel_4h_label["market"].values
    label_df["label_date"] = panel_4h_label["date_only"].values
    label_df["date_only"] = (pd.to_datetime(label_df["label_date"]) - pd.Timedelta(days=1)).dt.date

    pre4 = cached_frame(
        cache_dir / "precursor_4h.pkl",
        lambda: build_4h_precursor(candles_4h, btc_regime_map),
        refresh=args.refresh_cache,
    )
    pre4, pre4_cols = prefix_columns(pre4, PRE4H, "p4h_")

    log.info("loading 1h precursor...")
    krw_1h = [
        m
        for m in signal_eligible_markets(list_markets(args.upbit_1h))
        if m.startswith("KRW-")
    ]
    candles_1h = {m: load_candles(args.upbit_1h, m) for m in krw_1h}
    candles_1h = {k: v for k, v in candles_1h.items() if v is not None and len(v) > 30}
    pre1 = cached_frame(
        cache_dir / "precursor_1h_v1.pkl",
        lambda: build_1h_precursor(candles_1h, candles_1h.get("KRW-BTC", pd.DataFrame())),
        refresh=args.refresh_cache,
    )
    pre1, pre1_cols = prefix_columns(pre1, PRE1H, "p1h_")

    log.info("loading 15m precursor...")
    krw_15m = [
        m
        for m in signal_eligible_markets(list_markets(args.upbit_15m))
        if m.startswith("KRW-")
    ]
    candles_15m = {m: load_candles(args.upbit_15m, m) for m in krw_15m}
    candles_15m = {k: v for k, v in candles_15m.items() if v is not None and len(v) > 100}
    pre15 = cached_frame(
        cache_dir / "precursor_15m_v0.pkl",
        lambda: build_15m_precursor(candles_15m, candles_15m.get("KRW-BTC", pd.DataFrame())),
        refresh=args.refresh_cache,
    )
    pre15, pre15_cols = prefix_columns(pre15, PRE15M, "p15m_")

    log.info("merging common rows...")
    full = panel.merge(label_df.drop(columns=["label_date"]), on=["market", "date_only"], how="inner")
    full = full.merge(pre4, on=["market", "date_only"], how="inner")
    full = full.merge(pre1, on=["market", "date_only"], how="inner")
    full = full.merge(pre15, on=["market", "date_only"], how="inner")
    full["timestamp"] = pd.to_datetime(full["timestamp"])

    per_date = full.groupby("date_only").size()
    good_dates = per_date[per_date >= args.min_rows_per_date].index
    before = len(full)
    full = full[full["date_only"].isin(good_dates)].copy()
    log.info(
        "common rows: %s -> %s after min_rows_per_date=%s; dates=%s",
        f"{before:,}", f"{len(full):,}", args.min_rows_per_date, len(good_dates),
    )
    if len(full) < 1000:
        raise SystemExit("too few common rows; lower --min-rows-per-date or inspect 15m coverage")

    all_precursor_cols = set(pre4_cols + pre1_cols + pre15_cols)
    daily_cols = feature_cols_daily(full, list(HEADS.keys()), all_precursor_cols)
    feature_sets = {
        "daily": daily_cols,
        "daily+4h": daily_cols + pre4_cols,
        "daily+1h": daily_cols + pre1_cols,
        "daily+15m": daily_cols + pre15_cols,
        "daily+1h+15m": daily_cols + pre1_cols + pre15_cols,
        "daily+all": daily_cols + pre4_cols + pre1_cols + pre15_cols,
    }
    log.info(
        "feature counts: %s",
        {k: len(v) for k, v in feature_sets.items()},
    )

    splitter = PurgedWalkForward(
        n_folds=args.n_folds,
        embargo_days=args.embargo,
        holdout_days=args.holdout,
    )
    fold_data = []
    for fold, (train_dates, val_dates) in enumerate(splitter.split(full["timestamp"]), 1):
        train_p = full[full["timestamp"].isin(train_dates)]
        val_p = full[full["timestamp"].isin(val_dates)]
        if len(train_p) < 500 or len(val_p) < 200:
            log.info("skip fold %s: train=%s val=%s", fold, len(train_p), len(val_p))
            continue
        log.info(
            "Fold %s: train %s / val %s (%s ~ %s)",
            fold, f"{len(train_p):,}", f"{len(val_p):,}",
            pd.to_datetime(val_dates[0]).date(), pd.to_datetime(val_dates[-1]).date(),
        )
        fold_data.append((fold, train_p, val_p))

    rows = []
    for head in TARGET_HEADS:
        for feature_set, cols in feature_sets.items():
            log.info("--- head=%s feature_set=%s ---", head, feature_set)
            for fold, train_p, val_p in fold_data:
                df_tr = train_p[train_p[head].notna()].copy()
                df_va = val_p[val_p[head].notna()].copy()
                y_tr = df_tr[head].astype(int).values
                y_va = df_va[head].astype(int).values
                if y_tr.sum() < 10 or (y_tr == 0).sum() < 10 or y_va.sum() < 3:
                    log.info(
                        "  skip fold %s: train_pos=%s val_pos=%s",
                        fold, int(y_tr.sum()), int(y_va.sum()),
                    )
                    continue
                t0 = time.time()
                model = train_binary(df_tr[cols].astype(float).values, y_tr)
                scores = model.predict_proba(df_va[cols].astype(float).values)[:, 1]
                for pct in TOPK_PCTS:
                    m = eval_topk(scores, y_va, pct)
                    if m is None:
                        continue
                    rows.append({
                        "head": head,
                        "feature_set": feature_set,
                        "fold": fold,
                        "topK_pct": pct * 100,
                        **m,
                    })
                log.info("  fold %s done in %.1fs", fold, time.time() - t0)

    out = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    log.info("saved %s (%s rows)", args.out_csv, len(out))

    summary = weighted_summary(out)
    print("\n=== Weighted summary ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n=== Best per head/topK ===")
    best = summary.sort_values(["head", "topK_pct", "lift"], ascending=[True, True, False])
    best = best.groupby(["head", "topK_pct"], as_index=False).head(3)
    print(best.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()

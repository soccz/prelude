"""Pre-open first-15m trigger model audit.

Question:
  If 09:05 can miss a material share of TP3/TP5 moves, can a leak-free
  08:55/pre-open model predict the first 09:00-09:15 hit?

Strict as-of design:
  For target daily candle date D (KST D 09:00 -> D+1 09:00), at D 08:55:
    - daily candle D-1 is still in progress until D 09:00, so it is NOT used.
    - the last fully closed daily candle is D-2.
    - 15m precursor features use D-1 08:30 snapshot, already closed.

Feature sets:
  - prev_daily: D-2 closed daily features
  - preopen_15m: D-1 08:30 15m/30m/1h/3h/cumulative precursor features
  - prev_daily+preopen_15m

Targets:
  - first15_t3: +3% is first hit in the 09:00-09:15 candle
  - first15_t5 / first15_t10 / first15_t20

This is research/audit only. It does not change production timers.
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
from signals.features import assemble_training_panel
from signals.models.xgb_phase1 import EXCLUDE_COLS
from signals.precursors import (
    FIFTEEN_M_FEATURES,
    FIFTEEN_M_HIT_THRESHOLDS,
    build_15m_event_table,
    build_15m_precursor,
    cached_frame,
    prefix_columns,
)
from signals.validate import PurgedWalkForward


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
    max_depth=5,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
)

TOPK_PCTS = [0.005, 0.01, 0.02, 0.05]


def load_daily_panel(upbit_d1: str, binance_d1: str) -> pd.DataFrame:
    krw = list_markets(upbit_d1)
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
    return panel


def safe_daily_feature_cols(df: pd.DataFrame) -> list[str]:
    drop = {"date_only"}
    cols: list[str] = []
    for c in df.columns:
        if c in EXCLUDE_COLS or c in LEAK_COLS or c in drop:
            continue
        if c.startswith("next_"):
            continue
        dt = df[c].dtype
        if dt == object or "datetime" in str(dt):
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
    for (target, feature_set, top), g in df.groupby(["target", "feature_set", "topK_pct"], sort=False):
        n_val = int(g["n_val"].sum())
        n_pos = int(g["n_pos"].sum())
        n_top = int(g["n_top"].sum())
        top_pos = int(g["top_pos"].sum())
        base = n_pos / n_val if n_val else np.nan
        prec = top_pos / n_top if n_top else np.nan
        rows.append({
            "target": target,
            "feature_set": feature_set,
            "topK_pct": top,
            "folds": int(g["fold"].nunique()),
            "n_val": n_val,
            "base_pct": base * 100,
            "prec_pct": prec * 100,
            "lift": (prec / base) if base > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upbit-d1", default="data/upbit_d1.db")
    parser.add_argument("--upbit-15m", default="data/upbit_15m.db")
    parser.add_argument("--binance-d1", default="data/binance_d1.db")
    parser.add_argument("--cache-dir", default="output/cache")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--embargo", type=int, default=10)
    parser.add_argument("--holdout", type=int, default=0)
    parser.add_argument("--min-rows-per-date", type=int, default=100)
    parser.add_argument("--out-csv", default="output/preopen_first15m_model_v1.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("preopen")
    cache_dir = Path(args.cache_dir)

    print("=== Pre-open first-15m model audit v1 ===")
    print("as-of: D-2 closed daily + D-1 08:30 precursor -> D 09:00-09:15 first hit\n")

    log.info("loading daily panel...")
    panel = load_daily_panel(args.upbit_d1, args.binance_d1)
    daily_cols = safe_daily_feature_cols(panel)
    daily = panel[["market", "date_only"] + daily_cols].copy()
    daily["label_date"] = (pd.to_datetime(daily["date_only"]) + pd.Timedelta(days=2)).dt.date.astype(str)
    daily = daily.drop(columns=["date_only"])
    daily = daily.rename(columns={c: f"d2_{c}" for c in daily_cols})
    daily_cols_prefixed = [f"d2_{c}" for c in daily_cols]

    log.info("loading 15m candles...")
    krw_15m = [m for m in list_markets(args.upbit_15m) if m.startswith("KRW-")]
    candles_15m = {m: load_candles(args.upbit_15m, m) for m in krw_15m}
    candles_15m = {k: v for k, v in candles_15m.items() if v is not None and len(v) > 100}

    events = cached_frame(
        cache_dir / "hit_timing_15m_events.pkl",
        lambda: build_15m_event_table(candles_15m),
        refresh=args.refresh_cache,
    )
    events = events.rename(columns={"coin": "market"})
    events["label_date"] = events["date"].astype(str)
    events["timestamp"] = pd.to_datetime(events["label_date"])
    for name in FIFTEEN_M_HIT_THRESHOLDS:
        events[f"first15_{name}"] = (events[f"{name}_first_bar15"] == 0).astype(int)

    pre15 = cached_frame(
        cache_dir / "precursor_15m_v0.pkl",
        lambda: build_15m_precursor(candles_15m, candles_15m.get("KRW-BTC", pd.DataFrame())),
        refresh=args.refresh_cache,
    )
    pre15["label_date"] = (pd.to_datetime(pre15["date_only"]) + pd.Timedelta(days=1)).dt.date.astype(str)
    pre15 = pre15.drop(columns=["date_only"])
    pre15, pre15_cols = prefix_columns(pre15, FIFTEEN_M_FEATURES, "p15m_")

    full = events.merge(daily, on=["market", "label_date"], how="inner")
    full = full.merge(pre15, on=["market", "label_date"], how="inner")
    before = len(full)
    per_date = full.groupby("label_date").size()
    good_dates = per_date[per_date >= args.min_rows_per_date].index
    full = full[full["label_date"].isin(good_dates)].copy()
    log.info(
        "joined rows: %s -> %s after min_rows_per_date=%s; dates=%s",
        f"{before:,}", f"{len(full):,}", args.min_rows_per_date, len(good_dates),
    )

    feature_sets = {
        "prev_daily": daily_cols_prefixed,
        "preopen_15m": pre15_cols,
        "prev_daily+preopen_15m": daily_cols_prefixed + pre15_cols,
    }
    targets = [f"first15_{name}" for name in FIFTEEN_M_HIT_THRESHOLDS]
    log.info("feature counts: %s", {k: len(v) for k, v in feature_sets.items()})

    splitter = PurgedWalkForward(args.n_folds, args.embargo, args.holdout)
    folds = []
    for fold, (train_dates, val_dates) in enumerate(splitter.split(full["timestamp"]), 1):
        train_p = full[full["timestamp"].isin(train_dates)].copy()
        val_p = full[full["timestamp"].isin(val_dates)].copy()
        if len(train_p) < 100 or len(val_p) < 50:
            continue
        log.info(
            "Fold %s: train %s / val %s (%s -> %s)",
            fold, f"{len(train_p):,}", f"{len(val_p):,}",
            val_p["label_date"].min(), val_p["label_date"].max(),
        )
        folds.append((fold, train_p, val_p))

    rows = []
    for target in targets:
        for set_name, cols in feature_sets.items():
            log.info("--- target=%s feature_set=%s ---", target, set_name)
            for fold, train_p, val_p in folds:
                y_tr = train_p[target].astype(int).to_numpy()
                y_va = val_p[target].astype(int).to_numpy()
                if y_tr.sum() < 10 or y_va.sum() == 0:
                    log.info("  fold %s skipped (train_pos=%s val_pos=%s)", fold, int(y_tr.sum()), int(y_va.sum()))
                    continue
                t0 = time.time()
                model = train_binary(train_p[cols].astype(float).to_numpy(), y_tr)
                scores = model.predict_proba(val_p[cols].astype(float).to_numpy())[:, 1]
                for pct in TOPK_PCTS:
                    ev = eval_topk(scores, y_va, pct)
                    if ev is None:
                        continue
                    rows.append({
                        "target": target,
                        "feature_set": set_name,
                        "fold": fold,
                        "topK_pct": pct * 100,
                        **ev,
                    })
                log.info("  fold %s done in %.1fs", fold, time.time() - t0)

    raw = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.out_csv, index=False)
    log.info("saved %s (%s rows)", args.out_csv, len(raw))

    summary = weighted_summary(raw)
    print("\n=== Weighted summary ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n=== Best per target/topK ===")
    best = summary.sort_values(
        ["target", "topK_pct", "lift"], ascending=[True, True, False]
    ).groupby(["target", "topK_pct"], as_index=False).head(3)
    print(best.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()

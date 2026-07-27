"""Fold stability v3 — train-OOF threshold (overfit 보정).

v2 → v3 변경:
  - 문제: train direct p99.95 → val 에 거의 도달 못 함 (XGBoost overfit)
    Fold1: train_p99.95=0.982, val_p99.95=0.935, val>=train_thr 단 1개
  - 해결: train 안에서 3-fold time CV → OOF train score → p99.95
    realistic 분포에서 threshold 뽑기 (val 분포에 가깝게 calibrated)
  - 운영 원칙 유지: threshold = 학습 기간에 fix, val/live 에서 재계산 X

OOF threshold 절차 (per outer fold):
  1. train 을 시간순 3 chunk 로 split
  2. (chunk1+2) → chunk3 OOF, (chunk1+3) 같은 식으로 3회
     → simpler: chronological 3-split, 각 inner fold 는 그 이전 데이터로 학습
  3. concat OOF train score → p99.95 / p99.90 quantile
  4. 그 threshold 로 val 에 적용

사용:
    python scripts/fold_stability_v3.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.database import list_markets, load_candles
from data.market_universe import signal_eligible_markets
from ledger.config import ROUND_TRIP_COST_PCT
from signals.features import assemble_training_panel
from signals.models.xgb_phase1 import EXCLUDE_COLS
from signals.validate import PurgedWalkForward

CANDIDATES = [
    # A: OOF threshold (메인)
    {"name": "C1: bull_quiet p99.95 cap1", "mode": "thr", "regimes": ["bull_quiet"], "thr_pct": 0.9995, "cap": 1},
    {"name": "C2: bull_quiet p99.95 cap2", "mode": "thr", "regimes": ["bull_quiet"], "thr_pct": 0.9995, "cap": 2},
    {"name": "C3: bull_all p99.95 cap2", "mode": "thr", "regimes": ["bull_quiet", "bull_volatile"], "thr_pct": 0.9995, "cap": 2},
    {"name": "C4: bull_all p99.95 cap3", "mode": "thr", "regimes": ["bull_quiet", "bull_volatile"], "thr_pct": 0.9995, "cap": 3},
    {"name": "C5: all_except_bear_quiet p99.95 cap3", "mode": "thr",
     "regimes": ["bull_quiet", "bull_volatile", "bear_volatile"], "thr_pct": 0.9995, "cap": 3},
    {"name": "C6 참고: bull_quiet p99.90 cap1", "mode": "thr", "regimes": ["bull_quiet"], "thr_pct": 0.999, "cap": 1},
    {"name": "C7 참고: bull_quiet p99.90 cap2", "mode": "thr", "regimes": ["bull_quiet"], "thr_pct": 0.999, "cap": 2},
    # B: rank-based fallback baseline (no threshold, regime + per-day top-K)
    {"name": "C8 fallback: bull_quiet rank top1", "mode": "rank", "regimes": ["bull_quiet"], "cap": 1},
    {"name": "C9 fallback: bull_quiet rank top2", "mode": "rank", "regimes": ["bull_quiet"], "cap": 2},
    {"name": "C10 fallback: bull_all rank top1", "mode": "rank", "regimes": ["bull_quiet", "bull_volatile"], "cap": 1},
]


LEAK_COLS = {"net_under_tp", "max_return", "label", "label_tail",
             "next_open", "next_high", "next_low", "next_close",
             "next_max_return", "next_eod_return", "next_max_dd"}


def feature_cols(df, label_col):
    return [c for c in df.columns if c not in EXCLUDE_COLS and c != label_col
            and not c.startswith("next_") and c not in LEAK_COLS]


def train_binary(X_tr, y_tr):
    sw = compute_sample_weight(class_weight="balanced", y=y_tr)
    m = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        n_estimators=400, learning_rate=0.05, max_depth=6,
        min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, tree_method="hist",
        n_jobs=-1, random_state=42,
    )
    m.fit(X_tr, y_tr, sample_weight=sw, verbose=False)
    return m


def compute_oof_train_threshold(train_p, cols, label_col, n_inner=3, log=None):
    """train 안에서 chronological n_inner-fold → OOF score → KRW p99.95/p99.90 quantile."""
    train_p = train_p[train_p[label_col].notna()].sort_values("timestamp").reset_index(drop=True)
    dates = train_p["timestamp"].drop_duplicates().sort_values().reset_index(drop=True)
    if len(dates) < n_inner * 2:
        # too small → fallback to direct (no OOF)
        X = train_p[cols].astype(float).values
        y = train_p[label_col].astype(int).values
        m = train_binary(X, y)
        scores = m.predict_proba(X)[:, 1]
        return _quantiles_krw(train_p.assign(_s=scores), "_s")

    # split dates into n_inner+1 chunks; each inner fold: train on chunks[:i], predict on chunks[i]
    # i in {1, ..., n_inner} → n_inner OOF chunks
    chunk_size = len(dates) // (n_inner + 1)
    oof_rows = []
    for i in range(1, n_inner + 1):
        train_dates_inner = dates.iloc[: chunk_size * i]
        val_dates_inner = dates.iloc[chunk_size * i: chunk_size * (i + 1)] if i < n_inner \
            else dates.iloc[chunk_size * i:]
        tr_in = train_p[train_p["timestamp"].isin(train_dates_inner)]
        va_in = train_p[train_p["timestamp"].isin(val_dates_inner)].copy()
        if len(tr_in) < 100 or len(va_in) < 50:
            continue
        y_in = tr_in[label_col].astype(int).values
        if y_in.sum() < 5:
            continue
        X_in = tr_in[cols].astype(float).values
        m_in = train_binary(X_in, y_in)
        va_in["_oof_score"] = m_in.predict_proba(va_in[cols].astype(float).values)[:, 1]
        oof_rows.append(va_in[["market", "timestamp", "_oof_score"]])
        if log:
            log.info(f"    inner {i}: train {len(tr_in):,} / oof {len(va_in):,}")
    if not oof_rows:
        return None
    oof_df = pd.concat(oof_rows, ignore_index=True)
    return _quantiles_krw(oof_df, "_oof_score")


def _quantiles_krw(df, score_col):
    krw = df[df["market"].str.startswith("KRW-")]
    if len(krw) == 0:
        return None
    return {
        "p99.95": float(krw[score_col].quantile(0.9995)),
        "p99.90": float(krw[score_col].quantile(0.999)),
    }


def evaluate_fold_op(val_p, regimes, threshold, cap, target_pct=0.20,
                      cost=ROUND_TRIP_COST_PCT, mode="thr"):
    """mode='thr': score >= threshold + nlargest(cap)
       mode='rank': nlargest(cap) only (no threshold)"""
    sub = val_p[val_p["btc_regime"].isin(regimes)]
    sub = sub[sub["market"].str.startswith("KRW-")]
    if len(sub) == 0:
        return pd.DataFrame(), {"n": 0, "no_trade": True}

    rows = []
    for date, day in sub.groupby("timestamp"):
        if mode == "rank":
            sel = day.nlargest(cap, "model_score")
        else:
            sel = day[day["model_score"] >= threshold].nlargest(cap, "model_score")
        if len(sel) == 0:
            continue
        size = 1.0 / len(sel)
        for _, r in sel.iterrows():
            opn = r["next_open"]
            if pd.isna(opn) or opn <= 0:
                continue
            net = (target_pct - cost) if r["label_tail"] == 1 \
                  else ((r["next_eod_return"] if pd.notna(r["next_eod_return"]) else 0) - cost)
            rows.append({
                "date": pd.to_datetime(date), "coin": r["market"],
                "label": r["label_tail"], "eod": r["next_eod_return"],
                "size": size, "net": net,
            })
    trades = pd.DataFrame(rows)
    if len(trades) == 0:
        return trades, {"n": 0, "no_trade": True}

    n = len(trades)
    active = trades["date"].dt.normalize().nunique()
    hit = float(trades["label"].mean())
    ev = float(trades["net"].mean())
    fails = trades[trades["label"] == 0]
    worst5 = float(fails["eod"].quantile(0.05)) if len(fails) > 0 else 0
    worst_t = float(trades["net"].min())
    daily = trades.groupby(trades["date"].dt.normalize()).apply(lambda g: (g["net"] * g["size"]).sum())
    cum = float((1 + daily).cumprod().iloc[-1] - 1) if len(daily) > 0 else 0

    return trades, {
        "n": n, "active": active, "no_trade": False,
        "hit_pct": hit * 100, "EV_pct": ev * 100,
        "worst5_fail_eod_pct": worst5 * 100,
        "worst_trade_net_pct": worst_t * 100,
        "cum_ret_pct": cum * 100,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upbit-db", default="data/upbit_d1.db")
    parser.add_argument("--binance-db", default="data/binance_d1.db")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--embargo", type=int, default=10)
    parser.add_argument("--holdout", type=int, default=180)
    parser.add_argument("--target-pct", type=float, default=0.20)
    parser.add_argument("--n-inner", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("v3")

    print(f"=== Fold Stability v3 — train-OOF threshold (leak-free, overfit-corrected) ===\n")
    print(f"target ≥{int(args.target_pct*100)}%")
    print(f"threshold = train 안 {args.n_inner}-fold OOF score 의 KRW p99.95/p99.90 quantile")
    print(f"global score + regime gate (regime 내부 quantile X)\n")

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
    panel["next_low"] = g["low"].shift(-1)
    panel["next_close"] = g["close"].shift(-1)
    panel["next_max_return"] = panel["next_high"] / panel["next_open"] - 1
    panel["next_eod_return"] = panel["next_close"] / panel["next_open"] - 1
    panel["label_tail"] = (panel["next_max_return"] >= args.target_pct).astype(float)
    panel.loc[panel["next_max_return"].isna(), "label_tail"] = np.nan
    panel = panel.dropna(subset=["next_open", "label_tail"]).reset_index(drop=True)

    splitter = PurgedWalkForward(args.n_folds, args.embargo, args.holdout)

    fold_data = {}
    for fold, (train_dates, val_dates) in enumerate(splitter.split(panel["timestamp"]), 1):
        train_p = panel[panel["timestamp"].isin(train_dates)]
        val_p = panel[panel["timestamp"].isin(val_dates)].copy()
        if len(train_p) < 100 or len(val_p) < 50:
            continue
        log.info(f"Fold {fold}: train {len(train_p):,} / val {len(val_p):,}")
        df = train_p[train_p["label_tail"].notna()].copy()
        cols = feature_cols(df, "label_tail")

        # 1) OOF threshold from train
        thresholds = compute_oof_train_threshold(train_p, cols, "label_tail",
                                                  n_inner=args.n_inner, log=log)
        if thresholds is None:
            log.warning(f"  Fold {fold}: OOF threshold 계산 실패 — skip")
            continue
        log.info(f"  OOF train threshold: p99.95={thresholds['p99.95']:.4f}, p99.90={thresholds['p99.90']:.4f}")

        # 2) Final model on full train, predict on val
        X_tr = df[cols].astype(float).values
        y_tr = df["label_tail"].astype(int).values
        if y_tr.sum() < 5:
            continue
        m = train_binary(X_tr, y_tr)
        val_p["model_score"] = m.predict_proba(val_p[cols].astype(float).values)[:, 1]
        log.info(f"  val score: max={val_p['model_score'].max():.4f}, "
                 f"p99.95={val_p['model_score'].quantile(0.9995):.4f}")

        fold_data[fold] = {
            "val_p": val_p,
            "thr_p99_95": thresholds["p99.95"],
            "thr_p99_9": thresholds["p99.90"],
        }

    print()
    for cand in CANDIDATES:
        print(f"\n{'='*80}")
        print(f"## {cand['name']}")
        if cand["mode"] == "thr":
            print(f"   mode=THR, regimes={cand['regimes']}, threshold p{cand['thr_pct']*100:.2f}, cap={cand['cap']}")
        else:
            print(f"   mode=RANK (no threshold), regimes={cand['regimes']}, cap={cand['cap']}")
        print(f"{'='*80}")

        fold_rows = []
        all_trades = []
        for fold, fd in fold_data.items():
            val_p = fd["val_p"]
            if cand["mode"] == "thr":
                thr_key = "thr_p99_95" if cand["thr_pct"] >= 0.9995 else "thr_p99_9"
                threshold = fd[thr_key]
            else:
                threshold = float("-inf")

            trades, metrics = evaluate_fold_op(
                val_p, cand["regimes"], threshold, cand["cap"], args.target_pct,
                mode=cand["mode"],
            )
            metrics["fold"] = fold
            metrics["thr_used"] = threshold
            fold_rows.append(metrics)
            if not metrics["no_trade"]:
                trades["fold"] = fold
                all_trades.append(trades)

        fold_df = pd.DataFrame(fold_rows)
        cols_show = ["fold", "thr_used", "n", "active", "hit_pct", "EV_pct",
                     "worst5_fail_eod_pct", "worst_trade_net_pct", "cum_ret_pct"]
        for c in cols_show:
            if c not in fold_df.columns:
                fold_df[c] = 0
        print("\n  Per-fold (OOF threshold):")
        print(fold_df[cols_show].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

        no_trade_folds = int(fold_df["no_trade"].sum())
        active_folds = fold_df[~fold_df["no_trade"]]
        if len(active_folds) == 0:
            print(f"\n  ❌ no_trade in all folds")
            continue

        evs = active_folds["EV_pct"].tolist()
        n_pos = sum(1 for ev in evs if ev > 0)
        n_neg = sum(1 for ev in evs if ev <= 0)
        avg_ev = float(np.mean(evs))
        std_ev = float(np.std(evs))

        passes = []
        if no_trade_folds <= 1 and n_pos >= 3:
            passes.append("3+folds_pos")
        if no_trade_folds <= 1 and n_pos >= 2 and len(active_folds) <= 4:
            passes.append("2+folds_pos_borderline")

        if all_trades:
            all_t = pd.concat(all_trades, ignore_index=True)
            all_t["year"] = pd.to_datetime(all_t["date"]).dt.year
            year_sum = all_t.groupby("year").agg(
                n=("net", "size"),
                hit_pct=("label", lambda x: x.mean() * 100),
                EV_pct=("net", lambda x: x.mean() * 100),
            ).round(2)
            print("\n  연도별:")
            print(year_sum.to_string())
            if 2024 in year_sum.index:
                ev_2024 = year_sum.loc[2024, "EV_pct"]
                if ev_2024 > -3:
                    passes.append("2024_resilient")
                print(f"  2024 EV: {ev_2024:+.2f}%")

        print(f"\n  안정성: no_trade {no_trade_folds}/5, pos {n_pos}, neg {n_neg}")
        print(f"  EV 평균 {avg_ev:+.2f}%, std {std_ev:.2f}")
        print(f"  통과 항목: {passes if passes else '없음'}")

    print(f"\n{'='*80}")
    print(f"=== 통과 기준 (사용자 v2) ===")
    print(f"{'='*80}\n")
    print("✅ 통과:")
    print("   - no_trade fold ≤ 1")
    print("   - 3+ fold EV 양수 (또는 2+ fold + 4 active)")
    print("   - 2024 EV ≥ -3%")
    print("   - C6/C7 (p99.90) ≤ C1/C2 (p99.95) 비교")


if __name__ == "__main__":
    main()

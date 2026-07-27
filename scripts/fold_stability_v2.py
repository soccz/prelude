"""Fold stability v2 — operational threshold (leak-free).

v1 → v2 변경 (사용자 caveats):
  - threshold = train fold 의 KRW score quantile (per-fold)
    → 운영 룰과 동일 (val fold 에 미래 정보 X)
  - global score (전체 KRW) p99.95 + regime gate 분리
    (regime 내부 quantile X — 정의 통일)
  - no-trade fold 명시 표시
  - 통과 기준 (완화):
    * bull_quiet + p99.95 + cap1/2 가 3+ fold EV 양수
    * 2024 약세 구간 크게 안 망함
    * p99.90 ≤ p99.95 또는 비슷
    * all_except/bull_all 불안정 시 제외

운영 gate (사용자 정의):
  - bull_quiet 우선 후보
  - 비교: bull_quiet+bull_volatile, all_except_bear_quiet, bull_all
  - cap 1/2/3
  - threshold p99.95 기본, p99.90 참고

사용:
    python scripts/fold_stability_v2.py
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

# ============================================================================
# 후보 조합 (사용자 v2 정의)
# ============================================================================
CANDIDATES = [
    {"name": "C1: bull_quiet p99.95 cap1", "regimes": ["bull_quiet"], "thr_pct": 0.9995, "cap": 1},
    {"name": "C2: bull_quiet p99.95 cap2", "regimes": ["bull_quiet"], "thr_pct": 0.9995, "cap": 2},
    {"name": "C3: bull_all p99.95 cap2", "regimes": ["bull_quiet", "bull_volatile"], "thr_pct": 0.9995, "cap": 2},
    {"name": "C4: bull_all p99.95 cap3", "regimes": ["bull_quiet", "bull_volatile"], "thr_pct": 0.9995, "cap": 3},
    {"name": "C5: all_except_bear_quiet p99.95 cap3",
     "regimes": ["bull_quiet", "bull_volatile", "bear_volatile"], "thr_pct": 0.9995, "cap": 3},
    {"name": "C6 참고: bull_quiet p99.90 cap1", "regimes": ["bull_quiet"], "thr_pct": 0.999, "cap": 1},
    {"name": "C7 참고: bull_quiet p99.90 cap2", "regimes": ["bull_quiet"], "thr_pct": 0.999, "cap": 2},
]


def prep_features(panel, label_col):
    df = panel[panel[label_col].notna()].copy()
    LEAK_COLS = {"net_under_tp", "max_return", "label", "label_tail",
                 "next_open", "next_high", "next_low", "next_close",
                 "next_max_return", "next_eod_return", "next_max_dd"}
    cols = [c for c in df.columns if c not in EXCLUDE_COLS and c != label_col
            and not c.startswith("next_") and c not in LEAK_COLS]
    X = df[cols].astype(float).values
    y = df[label_col].astype(int).values
    return X, y, cols


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


def evaluate_fold_op(val_p, regimes, threshold, cap, target_pct=0.20,
                      cost=ROUND_TRIP_COST_PCT):
    """fold 별 operational evaluation. threshold 는 외부에서 주입."""
    sub = val_p[val_p["btc_regime"].isin(regimes)]
    sub = sub[sub["market"].str.startswith("KRW-")]
    if len(sub) == 0:
        return pd.DataFrame(), {"n": 0, "no_trade": True}

    rows = []
    for date, day in sub.groupby("timestamp"):
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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("v2")

    print(f"=== Fold Stability v2 — operational threshold (leak-free) ===\n")
    print(f"target ≥{int(args.target_pct*100)}%, threshold = train fold KRW score quantile (per-fold)")
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

    # Per-fold: train + val + threshold from train
    fold_data = {}
    for fold, (train_dates, val_dates) in enumerate(splitter.split(panel["timestamp"]), 1):
        train_p = panel[panel["timestamp"].isin(train_dates)]
        val_p = panel[panel["timestamp"].isin(val_dates)].copy()
        if len(train_p) < 100 or len(val_p) < 50:
            continue
        log.info(f"Fold {fold}: train {len(train_p):,} / val {len(val_p):,}")
        X_tr, y_tr, cols = prep_features(train_p, "label_tail")
        if y_tr.sum() < 5:
            continue
        m = train_binary(X_tr, y_tr)
        # train score (operational threshold 용)
        train_p_full = train_p[train_p["label_tail"].notna()].copy()
        train_p_full["model_score"] = m.predict_proba(train_p_full[cols].astype(float).values)[:, 1]
        # val score
        val_p["model_score"] = m.predict_proba(val_p[cols].astype(float).values)[:, 1]

        # train KRW score quantile (operational threshold)
        train_krw = train_p_full[train_p_full["market"].str.startswith("KRW-")]
        train_thr_p99_95 = float(train_krw["model_score"].quantile(0.9995))
        train_thr_p99_9 = float(train_krw["model_score"].quantile(0.999))

        fold_data[fold] = {
            "val_p": val_p,
            "thr_p99_95": train_thr_p99_95,
            "thr_p99_9": train_thr_p99_9,
        }

    # 후보 별 분석
    for cand in CANDIDATES:
        print(f"\n{'='*80}")
        print(f"## {cand['name']}")
        print(f"   regimes={cand['regimes']}, threshold p{cand['thr_pct']*100:.2f}, cap={cand['cap']}")
        print(f"{'='*80}")

        fold_rows = []
        all_trades = []
        for fold, fd in fold_data.items():
            val_p = fd["val_p"]
            thr_key = "thr_p99_95" if cand["thr_pct"] >= 0.9995 else "thr_p99_9"
            threshold = fd[thr_key]

            trades, m = evaluate_fold_op(val_p, cand["regimes"], threshold, cand["cap"], args.target_pct)
            m["fold"] = fold
            m["thr_used"] = threshold
            fold_rows.append(m)
            if not m["no_trade"]:
                trades["fold"] = fold
                all_trades.append(trades)

        fold_df = pd.DataFrame(fold_rows)

        # display
        cols_show = ["fold", "thr_used", "n", "active", "hit_pct", "EV_pct",
                     "worst5_fail_eod_pct", "worst_trade_net_pct", "cum_ret_pct"]
        # missing 처리
        for c in cols_show:
            if c not in fold_df.columns:
                fold_df[c] = 0
        print("\n  Per-fold (operational threshold):")
        print(fold_df[cols_show].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

        # 안정성 평가
        no_trade_folds = int(fold_df["no_trade"].sum())
        active_folds = fold_df[~fold_df["no_trade"]]
        if len(active_folds) == 0:
            print(f"\n  ❌ no_trade in all folds")
            continue

        evs = active_folds["EV_pct"].tolist()
        n_pos = sum(1 for ev in evs if ev > 0)
        n_neg = sum(1 for ev in evs if ev <= 0)
        avg_ev = np.mean(evs)
        std_ev = np.std(evs)

        # 통과 기준 (사용자 v2 완화):
        passes = []
        if no_trade_folds <= 1 and n_pos >= 3:
            passes.append("3+folds_pos")
        if no_trade_folds <= 1 and n_pos >= 2 and len(active_folds) <= 4:
            passes.append("2+folds_pos_borderline")

        if all_trades:
            all_t = pd.concat(all_trades, ignore_index=True)
            all_t["year"] = pd.to_datetime(all_t["date"]).dt.year
            year_sum = all_t.groupby("year").agg(
                n=("net", "size"), hit_pct=("label", lambda x: x.mean() * 100),
                EV_pct=("net", lambda x: x.mean() * 100),
            ).round(2)
            print("\n  연도별:")
            print(year_sum.to_string())

            # 2024 (약세) 별도 체크
            if 2024 in year_sum.index:
                ev_2024 = year_sum.loc[2024, "EV_pct"]
                if ev_2024 > -3:
                    passes.append("2024_resilient")
                print(f"  2024 EV: {ev_2024:+.2f}%")

        print(f"\n  안정성: no_trade {no_trade_folds}/5, pos {n_pos}, neg {n_neg}")
        print(f"  EV 평균 {avg_ev:+.2f}%, std {std_ev:.2f}")
        print(f"  통과 항목: {passes if passes else '없음'}")

    print(f"\n{'='*80}")
    print(f"=== 통과 기준 요약 (사용자 v2) ===")
    print(f"{'='*80}\n")
    print("✅ 통과:")
    print("   - no_trade fold ≤ 1")
    print("   - 3+ fold EV 양수 (또는 2+ fold + 4 active)")
    print("   - 2024 EV ≥ -3%")
    print("   - C6/C7 (p99.90) ≤ C1/C2 (p99.95) 비교")


if __name__ == "__main__":
    main()

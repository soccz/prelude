"""Fold / Year stability 검증 — 5 후보 조합의 안정성 분석.

목적 (사용자 가이드 Phase X-2-D):
  Sweep 결과 강하지만 표본 작음 (active 20-45일).
  Optuna 전에 fold 별 / 연도별 분포 확인 → 한 fold 의 우연 X 입증.

5 후보 조합:
  1. bull_quiet × p99.95 × cap1
  2. bull_quiet × p99.95 × cap2
  3. all_except_bear_quiet × p99.95 × cap3
  4. bull_all × p99.95 × cap3
  5. (참고) bull_quiet × p99.90 × cap1 (active 더 많음)

각 조합 분석:
  - fold별 n / active / hit% / EV / worst trade
  - fold별 cum_ret 분배 (한 fold 가 전체 의 80% 차지하면 의심)
  - 연도/분기별 분포
  - 운영 안정성 점수 (3+ fold 양수 + 음수 fold 작음)

채택 기준 (사용자 새로):
  ✅: 3+ fold 양수 EV
  ✅: 음수 fold 의 |EV| < 양수 fold 의 평균 EV
  ⚠: 한 fold 만 양수
  ❌: 모든 fold 음수

사용:
    python scripts/fold_stability_v1.py
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
# 5 후보 조합
# ============================================================================
CANDIDATES = [
    {"name": "C1: bull_quiet p99.95 cap1",
     "regimes": ["bull_quiet"], "thr_pct": 0.9995, "cap": 1},
    {"name": "C2: bull_quiet p99.95 cap2",
     "regimes": ["bull_quiet"], "thr_pct": 0.9995, "cap": 2},
    {"name": "C3: all_except_bear_quiet p99.95 cap3",
     "regimes": ["bull_quiet", "bull_volatile", "bear_volatile"],
     "thr_pct": 0.9995, "cap": 3},
    {"name": "C4: bull_all p99.95 cap3",
     "regimes": ["bull_quiet", "bull_volatile"], "thr_pct": 0.9995, "cap": 3},
    {"name": "C5: bull_quiet p99.90 cap1 (참고)",
     "regimes": ["bull_quiet"], "thr_pct": 0.999, "cap": 1},
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


def evaluate_fold(val_p, regimes, threshold, cap, target_pct=0.20,
                   cost=ROUND_TRIP_COST_PCT):
    """한 fold 의 trades + metrics."""
    sub = val_p[val_p["btc_regime"].isin(regimes)]
    sub = sub[sub["market"].str.startswith("KRW-")]
    if len(sub) == 0:
        return pd.DataFrame(), {}

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
        return trades, {}

    n = len(trades)
    active = trades["date"].dt.normalize().nunique()
    hit = float(trades["label"].mean())
    ev = float(trades["net"].mean())
    fails = trades[trades["label"] == 0]
    worst5 = float(fails["eod"].quantile(0.05)) if len(fails) > 0 else 0
    worst_trade = float(trades["net"].min())

    daily = trades.groupby(trades["date"].dt.normalize()).apply(lambda g: (g["net"] * g["size"]).sum())
    cum = float((1 + daily).cumprod().iloc[-1] - 1) if len(daily) > 0 else 0

    return trades, {
        "n": n, "active": active,
        "hit_pct": hit * 100, "EV_pct": ev * 100,
        "worst5_fail_eod_pct": worst5 * 100,
        "worst_trade_net_pct": worst_trade * 100,
        "cum_ret_pct": cum * 100,
    }


def stability_score(fold_evs: list, year_evs: dict | None = None) -> tuple[str, str]:
    """안정성 점수.
    return: (mark, reason)
    """
    if not fold_evs:
        return "❌", "no folds"
    n_pos = sum(1 for ev in fold_evs if ev > 0)
    n_neg = sum(1 for ev in fold_evs if ev <= 0)
    pos_evs = [ev for ev in fold_evs if ev > 0]
    neg_evs = [ev for ev in fold_evs if ev < 0]

    if n_pos == 0:
        return "❌", "all folds negative"
    if n_pos == 1 and len(fold_evs) > 1:
        return "⚠", f"only 1/{len(fold_evs)} folds positive"
    if n_pos >= 3:
        # 음수 fold 의 |EV| < 양수 fold 의 평균
        if neg_evs:
            avg_pos = np.mean(pos_evs)
            max_neg = abs(min(neg_evs))
            if max_neg > avg_pos:
                return "⚠", f"neg fold |EV|={max_neg:.2f} > avg pos {avg_pos:.2f}"
        return "✅", f"{n_pos}/{len(fold_evs)} folds positive"
    if n_pos == 2 and len(fold_evs) <= 4:
        return "⚠", f"{n_pos}/{len(fold_evs)} folds positive (borderline)"
    return "⚠", f"{n_pos}/{len(fold_evs)}"


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
    log = logging.getLogger("stability")

    print(f"=== Fold/Year Stability — 5 후보 조합 ===\n")

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

    # WF 학습 + score (fold 별 보존)
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
        val_p["model_score"] = m.predict_proba(val_p[cols].astype(float).values)[:, 1]
        fold_data[fold] = val_p

    # 각 후보 × 각 fold 분석
    for cand in CANDIDATES:
        print(f"\n{'='*80}")
        print(f"## {cand['name']}")
        print(f"   regimes={cand['regimes']}, threshold=p{cand['thr_pct']*100:.2f}, cap={cand['cap']}")
        print(f"{'='*80}")

        # Per-fold metric
        fold_rows = []
        all_trades = []
        # Threshold = 모든 fold 의 KRW val 의 quantile (전체 기준 — 운영과 동일)
        all_val = pd.concat(fold_data.values(), ignore_index=True)
        sub_all = all_val[all_val["btc_regime"].isin(cand["regimes"])]
        sub_all = sub_all[sub_all["market"].str.startswith("KRW-")]
        if len(sub_all) == 0:
            print("  (regime 안 맞음)")
            continue
        threshold = float(sub_all["model_score"].quantile(cand["thr_pct"]))

        for fold, val_p in fold_data.items():
            trades, m = evaluate_fold(val_p, cand["regimes"], threshold, cand["cap"], args.target_pct)
            if not m:
                continue
            m["fold"] = fold
            fold_rows.append(m)
            if len(trades) > 0:
                trades["fold"] = fold
                all_trades.append(trades)

        if not fold_rows:
            print("  (trades 없음)")
            continue

        fold_df = pd.DataFrame(fold_rows)
        cols_show = ["fold", "n", "active", "hit_pct", "EV_pct",
                     "worst5_fail_eod_pct", "worst_trade_net_pct", "cum_ret_pct"]
        print("\n  Per-fold metrics:")
        print(fold_df[cols_show].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

        # 안정성 평가
        fold_evs = fold_df["EV_pct"].tolist()
        mark, reason = stability_score(fold_evs)
        print(f"\n  안정성: {mark} {reason}")
        print(f"  EV 평균: {np.mean(fold_evs):.2f}%  std: {np.std(fold_evs):.2f}%")

        # 연도별 분포
        if all_trades:
            all_t = pd.concat(all_trades, ignore_index=True)
            all_t["year"] = pd.to_datetime(all_t["date"]).dt.year
            year_summary = all_t.groupby("year").agg(
                n=("net", "size"),
                hit_pct=("label", lambda x: x.mean() * 100),
                EV_pct=("net", lambda x: x.mean() * 100),
                worst_trade=("net", "min"),
            ).round(2)
            print("\n  연도별:")
            print(year_summary.to_string())

            # 한 fold 의 cum_ret 비중 — 전체 cum 의 X% 가 fold 1 에서?
            if len(fold_df) > 1:
                total_cum = fold_df["cum_ret_pct"].sum()
                if total_cum > 0:
                    print("\n  fold별 cum_ret 비중 (전체 합 기준):")
                    for _, r in fold_df.iterrows():
                        if r["cum_ret_pct"] > 0:
                            pct = r["cum_ret_pct"] / total_cum * 100
                            print(f"    Fold {int(r['fold'])}: {r['cum_ret_pct']:+.2f}% ({pct:.1f}%)")
                        else:
                            print(f"    Fold {int(r['fold'])}: {r['cum_ret_pct']:+.2f}%")

    print(f"\n{'='*80}")
    print("=== 운영안 가설 (D 통과 후) ===")
    print(f"{'='*80}\n")
    print("기본 침묵: bear_quiet (D 분석 외 — 별도 침묵)")
    print("🔥 very rare: bull_quiet × p99.95 × cap1/2")
    print("✅ active tail: all_except_bear_quiet × p99.95 × cap3")
    print("⚠  high-risk warning: bear_volatile 포함 후보 → DD 경고 표시")


if __name__ == "__main__":
    main()

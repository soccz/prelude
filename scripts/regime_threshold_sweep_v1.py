"""Regime × Threshold × Cap sweep — 운영 sweet spot 찾기.

목적 (사용자 가이드 Phase X-2):
  v1b 에서 모델 score 자체보다 **운영 조건이 성과를 갈라버림** 확인.
  Optuna 이전에 sweet spot 먼저 잡아야 함.

격자:
  regime: bull_quiet / bull_volatile / bull_all / bull+bear_volatile /
          all_except_bear_quiet / all
  threshold: p99.25 / p99.5 / p99.75 / p99.9 / p99.95
  cap: 1 / 2 / 3
  → 6 × 5 × 3 = 90 조합

각 조합 metric:
  n_trades, active_days, alerts_per_active_day,
  hit%, avg_fail_eod%, worst_5pct_fail_eod%,
  EV%, sharpe, fold variance (활성일/EV 의 fold 별 std)

채택 기준 (사용자 명시):
  EV ≥ +0.20%
  hit ≥ 18% OR fail EOD 충분히 작음
  active_days ≥ 100 근처
  worst 5% fail EOD 과도하지 않음
  연도/fold 한 구간에 몰리지 않음

운영 tier 가설:
  기본: bull_all + p99.75~99.9 sweet spot
  강한: p99.95+ "🔥 very rare"
  침묵: bear_quiet
  위험 경고: bear_volatile (별도 tier)

사용:
    python scripts/regime_threshold_sweep_v1.py
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
from ledger.config import ROUND_TRIP_COST_PCT
from signals.features import assemble_training_panel
from signals.models.xgb_phase1 import EXCLUDE_COLS
from signals.validate import PurgedWalkForward


# ============================================================================
# Regime sets
# ============================================================================
REGIME_SETS = {
    "bull_quiet": ["bull_quiet"],
    "bull_volatile": ["bull_volatile"],
    "bull_all": ["bull_quiet", "bull_volatile"],
    "bull+bear_volatile": ["bull_quiet", "bull_volatile", "bear_volatile"],
    "all_except_bear_quiet": ["bull_quiet", "bull_volatile", "bear_volatile"],
    "all": ["bull_quiet", "bull_volatile", "bear_quiet", "bear_volatile"],
}

THRESHOLDS = [0.9925, 0.995, 0.9975, 0.999, 0.9995]
CAPS = [1, 2, 3]


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


def evaluate_combo(panel, regimes, threshold_pct, cap, target_pct=0.20,
                    cost=ROUND_TRIP_COST_PCT):
    """
    한 조합 (regime + threshold + cap) 평가.

    return: dict of metrics
    """
    # Regime filter
    sub = panel[panel["btc_regime"].isin(regimes)]
    if len(sub) == 0:
        return None

    # Threshold (regime 안에서 quantile, train-only X — fold 평균 score 사용 가정)
    threshold = float(sub["model_score"].quantile(threshold_pct))

    # 매일 score >= threshold + cap top
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
            # net under TP_only
            if r["label_tail"] == 1:
                net = target_pct - cost
            else:
                eod_ret = r["next_eod_return"]
                net = (eod_ret if pd.notna(eod_ret) else 0) - cost
            rows.append({
                "date": pd.to_datetime(date),
                "coin": r["market"],
                "label": r["label_tail"],
                "eod": r["next_eod_return"],
                "max_dd": r["next_max_dd"],
                "size": size,
                "net": net,
                "fold_id": r.get("fold_id", -1),
            })

    trades = pd.DataFrame(rows)
    if len(trades) == 0:
        return None

    # Aggregates
    n = len(trades)
    active_days = trades["date"].dt.normalize().nunique()
    n_total_days = sub["timestamp"].dt.normalize().nunique()
    avg_alerts = n / active_days if active_days > 0 else 0
    hit_rate = float(trades["label"].mean())
    fails = trades[trades["label"] == 0]
    avg_fail_eod = float(fails["eod"].mean()) if len(fails) > 0 else 0
    worst_fail = float(fails["eod"].quantile(0.05)) if len(fails) > 0 else 0

    avg_net = float(trades["net"].mean())  # = EV per pos
    daily_net = trades.groupby(trades["date"].dt.normalize()).apply(
        lambda g: (g["net"] * g["size"]).sum()
    )
    sharpe = float(daily_net.mean() / (daily_net.std() + 1e-12) * np.sqrt(365)) if len(daily_net) >= 2 else 0
    eq = (1 + daily_net).cumprod()
    cum_ret = float(eq.iloc[-1] - 1) if len(eq) > 0 else 0
    mdd = float((eq - eq.cummax()).min() / eq.cummax().max()) if len(eq) > 0 else 0

    # fold variance — fold 별 EV / hit
    fold_var = None
    if "fold_id" in trades.columns and trades["fold_id"].nunique() > 1:
        fold_evs = trades.groupby("fold_id")["net"].mean()
        fold_var = float(fold_evs.std())

    return {
        "n_trades": n,
        "active_days": active_days,
        "active_pct": active_days / n_total_days * 100 if n_total_days > 0 else 0,
        "alerts_per_active": avg_alerts,
        "hit_pct": hit_rate * 100,
        "avg_fail_eod_pct": avg_fail_eod * 100,
        "worst5_fail_eod_pct": worst_fail * 100,
        "EV_pct": avg_net * 100,
        "sharpe": sharpe,
        "cum_ret_pct": cum_ret * 100,
        "mdd_pct": mdd * 100,
        "fold_EV_std": fold_var * 100 if fold_var else None,
    }


def check_criteria(m: dict) -> tuple[bool, str]:
    """채택 기준 체크 (사용자 명시)."""
    reasons = []
    passed = True
    if m["EV_pct"] < 0.20:
        passed = False
        reasons.append(f"EV<0.20")
    if m["hit_pct"] < 18 and m["avg_fail_eod_pct"] < -2.0:
        # fail EOD 충분히 작으면 hit 낮아도 OK
        passed = False
        reasons.append(f"hit<18 + fail_eod<-2")
    if m["active_days"] < 50:  # 100 너무 strict, 50 으로 완화
        passed = False
        reasons.append(f"active<50")
    if m["worst5_fail_eod_pct"] < -5:
        passed = False
        reasons.append(f"worst<-5")
    return passed, ",".join(reasons) if reasons else "OK"


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
    log = logging.getLogger("sweep")

    print(f"=== Regime × Threshold × Cap sweep ===\n")
    print(f"target ≥{int(args.target_pct*100)}%, TP{int(args.target_pct*100)}_only, cost {ROUND_TRIP_COST_PCT*100:.2f}%\n")

    log.info("loading...")
    krw = list_markets(args.upbit_db)
    candles = {m: load_candles(args.upbit_db, m) for m in krw}
    if Path(args.binance_db).exists():
        for m in list_markets(args.binance_db):
            candles[m] = load_candles(args.binance_db, m)
    candles = {k: v for k, v in candles.items() if len(v) > 30}
    btc = load_candles(args.upbit_db, "KRW-BTC")

    log.info("building panel...")
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
    panel["next_max_dd"] = panel["next_low"] / panel["next_open"] - 1
    panel["label_tail"] = (panel["next_max_return"] >= args.target_pct).astype(float)
    panel.loc[panel["next_max_return"].isna(), "label_tail"] = np.nan
    panel = panel.dropna(subset=["next_open", "label_tail"]).reset_index(drop=True)

    splitter = PurgedWalkForward(args.n_folds, args.embargo, args.holdout)
    val_with_score = []
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
        val_p["fold_id"] = fold
        val_with_score.append(val_p)

    val = pd.concat(val_with_score, ignore_index=True)
    krw_val = val[val["market"].str.startswith("KRW-")]
    log.info(f"val (KRW only): {krw_val.shape}")

    # Sweep
    results = []
    for reg_name, regimes in REGIME_SETS.items():
        for thr in THRESHOLDS:
            for cap in CAPS:
                m = evaluate_combo(krw_val, regimes, thr, cap, args.target_pct)
                if m is None:
                    continue
                m["regime"] = reg_name
                m["threshold"] = f"p{thr*100:.2f}"
                m["cap"] = cap
                passed, reason = check_criteria(m)
                m["pass"] = "✅" if passed else "❌"
                m["reason"] = reason
                results.append(m)

    df = pd.DataFrame(results)
    df = df.sort_values("EV_pct", ascending=False).reset_index(drop=True)

    cols_show = ["regime", "threshold", "cap", "n_trades", "active_days", "active_pct",
                 "alerts_per_active", "hit_pct", "avg_fail_eod_pct", "worst5_fail_eod_pct",
                 "EV_pct", "sharpe", "cum_ret_pct", "pass", "reason"]

    print(f"\n=== ALL combinations (정렬: EV ↓), 90 조합 ===")
    print(df[cols_show].head(20).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print(f"\n=== ✅ 채택 기준 통과 ===")
    passed = df[df["pass"] == "✅"]
    if len(passed) > 0:
        print(passed[cols_show].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    else:
        print("(통과 조합 없음)")

    print(f"\n=== Top 3 EV (전체) ===")
    print(df[cols_show].head(3).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print(f"\n=== Per-regime best EV ===")
    best_per_regime = df.loc[df.groupby("regime")["EV_pct"].idxmax()][cols_show]
    print(best_per_regime.sort_values("EV_pct", ascending=False).to_string(
        index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()

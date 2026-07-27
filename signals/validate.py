"""Purged Walk-Forward Cross-Validation.

설계 (SIGNAL.md §7, CLAUDE.md §2.5 양보 X 위생):
  - 5-fold WF
  - embargo: 10 일 (라벨 horizon 1d × 보수 마진)
  - 최종 holdout: 마지막 6 개월 절대 락
  - look-ahead 누수 방어 (panel 시간순 정렬)

핵심 함수:
  - PurgedWalkForward: 학습/검증 fold yield
  - run_wf_cv(panel, n_folds, embargo_days) → fold 별 metrics
  - run_wf_cv_with_holdout(panel) → fold + 최종 holdout
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

# 직접 실행용
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ============================================================================
# Purged Walk-Forward
# ============================================================================
@dataclass
class PurgedWalkForward:
    """
    시간 순 panel 의 5-fold WF + embargo.

    n_folds: 폴드 갯수 (default 5)
    embargo_days: train 끝 ~ val 시작 사이 빈 구간 (look-ahead 방어)
    holdout_days: 마지막 N일은 fold 에서 제외 (held-out test set)
    expand_train: True 면 train 누적 확장 (anchored), False 면 sliding
    """
    n_folds: int = 5
    embargo_days: int = 10
    holdout_days: int = 180
    expand_train: bool = True

    def split(self, timestamps: pd.Series) -> Iterable[tuple]:
        """
        timestamps: panel 의 timestamp Series (정렬 안 됐을 수도)

        yield: (train_idx, val_idx) per fold
        """
        ts = pd.to_datetime(timestamps).sort_values()
        ts_unique = ts.unique()
        n_dates = len(ts_unique)

        if n_dates < self.n_folds * 2 + self.embargo_days:
            raise ValueError(f"timestamps too few ({n_dates}) for {self.n_folds} folds")

        # holdout 분리 (마지막 N일 제외)
        holdout_cutoff_dt = ts_unique[-1] - pd.Timedelta(days=self.holdout_days)
        cv_dates = ts_unique[ts_unique <= holdout_cutoff_dt]
        n_cv = len(cv_dates)

        # fold 별 val window 크기
        fold_size = n_cv // (self.n_folds + 1)  # +1 for first train

        for fold in range(self.n_folds):
            # val window: cv_dates 의 [start..end]
            val_start_idx = (fold + 1) * fold_size
            val_end_idx = (fold + 2) * fold_size
            val_dates = cv_dates[val_start_idx:val_end_idx]

            if len(val_dates) == 0:
                continue

            val_start_dt = val_dates[0]
            embargo_dt = val_start_dt - pd.Timedelta(days=self.embargo_days)

            if self.expand_train:
                train_dates = cv_dates[cv_dates <= embargo_dt]
            else:
                # sliding window — 같은 fold_size 만큼 train
                tr_end_idx = max(0, val_start_idx - 1)
                tr_start_idx = max(0, tr_end_idx - fold_size)
                train_dates = cv_dates[tr_start_idx:tr_end_idx + 1]
                # embargo 추가 적용
                train_dates = train_dates[train_dates <= embargo_dt]

            yield train_dates, val_dates

    def held_out(self, timestamps: pd.Series) -> pd.Series:
        """최종 holdout 기간."""
        ts = pd.to_datetime(timestamps).sort_values().unique()
        cutoff = ts[-1] - pd.Timedelta(days=self.holdout_days)
        return pd.Series([t for t in ts if t > cutoff])


# ============================================================================
# WF 실행
# ============================================================================
def run_wf_cv(
    panel: pd.DataFrame,
    n_folds: int = 5,
    embargo_days: int = 10,
    holdout_days: int = 180,
    train_func=None,
    eval_func=None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    panel 에 Purged WF 적용 + train_func / eval_func 호출.

    train_func(train_panel) → model
    eval_func(model, val_panel) → metrics dict

    return: fold 별 metrics DataFrame (mean / std 포함)
    """
    from signals.models.xgb_phase1 import train_full, evaluate, prepare_features
    from signals.calibration import calibration_report

    if train_func is None:
        def train_func(p):
            m, _ = train_full(p, val_ratio=0.0001, tune=False)
            return m

    if eval_func is None:
        def eval_func(m, p):
            X, y, _ = prepare_features(p)
            metrics = evaluate(m, X, y)
            actual_max = p.dropna(subset=["label"])["max_return"].values
            probs = m.predict_proba(X)
            cal = calibration_report(probs, y, actual_max_return=actual_max)
            return {**metrics, "brier_calib": cal["brier"]}

    splitter = PurgedWalkForward(
        n_folds=n_folds, embargo_days=embargo_days, holdout_days=holdout_days
    )
    panel = panel.dropna(subset=["label"]).copy()
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])

    results = []
    for fold, (train_dates, val_dates) in enumerate(splitter.split(panel["timestamp"]), 1):
        train_panel = panel[panel["timestamp"].isin(train_dates)]
        val_panel = panel[panel["timestamp"].isin(val_dates)]

        if len(train_panel) < 100 or len(val_panel) < 50:
            continue

        if verbose:
            print(f"  Fold {fold}: train {len(train_panel):>6,} ({train_dates[0].date()}~{train_dates[-1].date()}) "
                  f"| val {len(val_panel):>6,} ({val_dates[0].date()}~{val_dates[-1].date()})")

        try:
            model = train_func(train_panel)
            metrics = eval_func(model, val_panel)
            metrics["fold"] = fold
            metrics["train_n"] = len(train_panel)
            metrics["val_n"] = len(val_panel)
            results.append(metrics)
        except Exception as e:
            if verbose:
                print(f"    Fold {fold} FAIL: {e}")

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    return df


def run_wf_with_holdout(
    panel: pd.DataFrame,
    n_folds: int = 5,
    embargo_days: int = 10,
    holdout_days: int = 180,
    verbose: bool = True,
) -> dict:
    """
    WF CV + 최종 holdout 평가.

    return: { 'cv_results': DataFrame, 'cv_summary': dict, 'holdout': dict, 'holdout_model': model }
    """
    from signals.models.xgb_phase1 import train_full, evaluate, prepare_features
    from signals.calibration import calibration_report

    panel = panel.dropna(subset=["label"]).copy()
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])

    # 1. CV
    cv_results = run_wf_cv(panel, n_folds, embargo_days, holdout_days, verbose=verbose)
    if len(cv_results) == 0:
        return {"cv_results": cv_results, "holdout": None}

    # 2. 최종 holdout
    splitter = PurgedWalkForward(n_folds, embargo_days, holdout_days)
    holdout_dates = splitter.held_out(panel["timestamp"])
    cutoff = pd.to_datetime(holdout_dates.iloc[0])
    train_panel = panel[panel["timestamp"] < cutoff - pd.Timedelta(days=embargo_days)]
    holdout_panel = panel[panel["timestamp"] >= cutoff]

    if verbose:
        print(f"\n  Final: train {len(train_panel):>6,} (~{cutoff.date()}) "
              f"| holdout {len(holdout_panel):>6,}")

    model, _ = train_full(train_panel, val_ratio=0.0001, tune=False)
    X, y, _ = prepare_features(holdout_panel)
    metrics = evaluate(model, X, y)
    actual_max = holdout_panel["max_return"].values
    probs = model.predict_proba(X)
    cal = calibration_report(probs, y, actual_max_return=actual_max)

    holdout = {**metrics, "brier_calib": cal["brier"], "n": len(holdout_panel)}

    cv_summary = {
        "mean_mlogloss": float(cv_results["mlogloss"].mean()),
        "std_mlogloss": float(cv_results["mlogloss"].std()),
        "mean_accuracy": float(cv_results["accuracy"].mean()),
        "mean_macro_f1": float(cv_results["macro_f1"].mean()),
        "mean_brier": float(cv_results["brier"].mean()),
    }

    return {
        "cv_results": cv_results,
        "cv_summary": cv_summary,
        "holdout": holdout,
        "holdout_model": model,
    }


# ============================================================================
# 직접 실행
# ============================================================================
if __name__ == "__main__":
    import argparse

    from data.database import list_markets, load_candles
    from data.market_universe import signal_eligible_markets
    from signals.features import assemble_training_panel

    parser = argparse.ArgumentParser()
    parser.add_argument("--upbit-db", default="data/upbit_d1.db")
    parser.add_argument("--binance-db", default="data/binance_d1.db")
    parser.add_argument("--btc-market", default="KRW-BTC")
    parser.add_argument("--limit", type=int, default=50, help="샘플 코인 (속도용)")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--embargo", type=int, default=10)
    parser.add_argument("--holdout", type=int, default=180)
    args = parser.parse_args()

    print(f"loading {args.limit} KRW + 일부 BINANCE markets...")
    krw_markets = signal_eligible_markets(list_markets(args.upbit_db))[
        : args.limit
    ]
    candles = {m: load_candles(args.upbit_db, m) for m in krw_markets}
    if Path(args.binance_db).exists():
        bn_markets = list_markets(args.binance_db)[: args.limit]
        for m in bn_markets:
            candles[m] = load_candles(args.binance_db, m)

    candles = {k: v for k, v in candles.items() if len(v) > 30}
    btc = load_candles(args.upbit_db, args.btc_market)

    print(f"building panel from {len(candles)} markets...")
    panel = assemble_training_panel(candles, btc, normalize=True)
    print(f"panel: {panel.shape}, dates: {panel['timestamp'].min().date()} ~ {panel['timestamp'].max().date()}")

    print(f"\nWF CV ({args.n_folds} folds, embargo {args.embargo}d, holdout {args.holdout}d)...")
    out = run_wf_with_holdout(
        panel, n_folds=args.n_folds, embargo_days=args.embargo, holdout_days=args.holdout
    )

    print("\n=== CV summary ===")
    if out.get("cv_summary"):
        for k, v in out["cv_summary"].items():
            print(f"  {k}: {v:.4f}")

    print("\n=== CV per-fold ===")
    if len(out["cv_results"]) > 0:
        cols = ["fold", "train_n", "val_n", "mlogloss", "accuracy", "macro_f1", "brier"]
        print(out["cv_results"][cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== Final holdout ===")
    if out.get("holdout"):
        for k, v in out["holdout"].items():
            if isinstance(v, list):
                continue
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

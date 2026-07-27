"""Calibration — multi-class 분포 모델의 정확도 측정 + 보정.

설계 (SIGNAL.md §5):
  - Reliability diagram (cumulative 확률 별 실제 적중률)
  - Brier score (multi-class)
  - Quantile coverage (CI 가 실제로 N% 커버하는지)
  - per-bin accuracy
  - 정확도 알림 표 (사용자 신뢰)
  - (옵션) Isotonic regression 보정

핵심 클래스 / 함수:
  - reliability_diagram(p_cum, actual, threshold) — bucket 별 실제 적중률
  - brier_score_multi(probs, y) — multi-class Brier
  - quantile_coverage(ci_low, ci_high, actual) — CI 커버리지
  - per_class_accuracy(probs, y) — bin 별 정확도
  - calibration_report(model, X, y, bin_centers, bins) — 종합 표
  - IsotonicCalibrator (옵션) — cumulative 확률 보정
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


# ============================================================================
# Reliability diagram
# ============================================================================
def reliability_diagram(
    p_cum: np.ndarray,
    actual_hit: np.ndarray,
    n_buckets: int = 10,
) -> pd.DataFrame:
    """
    cumulative 확률의 reliability.

    p_cum: 예측 확률 (예: P(max ≥ 10%)) shape (N,)
    actual_hit: 실제 적중 여부 (0/1) shape (N,)
    n_buckets: 확률 bucket 갯수 (예: 10 = 0~10%, 10~20%, ...)

    return: bucket 별 (pred_avg, actual_rate, n)
    """
    bucket_edges = np.linspace(0, 1, n_buckets + 1)
    rows = []
    for low, high in zip(bucket_edges[:-1], bucket_edges[1:]):
        mask = (p_cum >= low) & (p_cum < high) if high < 1.0 else (p_cum >= low) & (p_cum <= 1.0)
        n = int(mask.sum())
        if n == 0:
            continue
        pred_avg = float(p_cum[mask].mean())
        actual_rate = float(actual_hit[mask].mean())
        rows.append({
            "bucket_low": float(low),
            "bucket_high": float(high),
            "pred_avg": pred_avg,
            "actual_rate": actual_rate,
            "n": n,
            "abs_deviation": abs(pred_avg - actual_rate),
        })
    return pd.DataFrame(rows)


# ============================================================================
# Brier score
# ============================================================================
def brier_score_multi(probs: np.ndarray, y: np.ndarray) -> float:
    """multi-class Brier (낮을수록 정확)."""
    n_class = probs.shape[1]
    one_hot = np.eye(n_class)[y]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


# ============================================================================
# Quantile coverage
# ============================================================================
def quantile_coverage(
    ci_low: np.ndarray, ci_high: np.ndarray, actual: np.ndarray, target: float = 0.95
) -> dict:
    """CI 가 실제로 N% 커버하는지."""
    in_ci = (actual >= ci_low) & (actual <= ci_high)
    coverage = float(in_ci.mean())
    return {
        "coverage": coverage,
        "target": target,
        "deviation": coverage - target,  # > 0: 너무 넓음, < 0: 너무 좁음
        "n": int(len(actual)),
    }


# ============================================================================
# Per-class / Per-bin accuracy
# ============================================================================
def per_class_accuracy(probs: np.ndarray, y: np.ndarray) -> dict:
    """각 bin 별 적중률 + sample 수."""
    pred = probs.argmax(axis=1)
    n_class = probs.shape[1]
    return {
        f"bin_{c}": {
            "n_actual": int((y == c).sum()),
            "n_predicted": int((pred == c).sum()),
            "accuracy": float((pred[y == c] == c).mean()) if (y == c).any() else 0.0,
            "precision": float((y[pred == c] == c).mean()) if (pred == c).any() else 0.0,
        }
        for c in range(n_class)
    }


# ============================================================================
# 종합 calibration report
# ============================================================================
def calibration_report(
    probs: np.ndarray,
    y: np.ndarray,
    actual_max_return: Optional[np.ndarray] = None,
    bin_centers: Optional[np.ndarray] = None,
    cumulative_thresholds: tuple = (0.05, 0.10, 0.15, 0.20),
) -> dict:
    """
    종합 calibration report (SIGNAL §5.4).

    return: {
      'brier': float,
      'per_bin': { bin_X: {n_actual, n_predicted, accuracy, precision}, ... },
      'reliability': { 'p_ge_5': DataFrame, 'p_ge_10': ..., ... },
      'quantile_coverage': { ... } (actual_max_return 있을 때만),
    }
    """
    from signals.labels import (
        cumulative_probs,
        approx_ci_from_bins, DEFAULT_BIN_CENTERS,
    )
    if bin_centers is None:
        bin_centers = DEFAULT_BIN_CENTERS

    report = {
        "brier": brier_score_multi(probs, y),
        "per_bin": per_class_accuracy(probs, y),
    }

    # reliability per cumulative threshold
    cum = cumulative_probs(probs)
    reliability = {}
    if actual_max_return is not None:
        for t in cumulative_thresholds:
            key = f"p_ge_{int(t * 100)}"
            actual_hit = (actual_max_return >= t).astype(int)
            reliability[key] = reliability_diagram(cum[key], actual_hit)
        report["reliability"] = reliability

        # quantile coverage
        ci_low, ci_high = approx_ci_from_bins(probs, bin_centers)
        report["quantile_coverage"] = quantile_coverage(
            ci_low, ci_high, actual_max_return, target=0.95
        )

    return report


# ============================================================================
# Isotonic regression 보정 (옵션)
# ============================================================================
@dataclass
class IsotonicCalibrator:
    """
    cumulative 확률 별 isotonic 보정 (예측 50% → 실제 40% 면 isotonic 으로 매핑).
    각 cutoff (5%, 10%, 15%, 20%) 별 IsotonicRegression 학습.
    """
    cumulative_thresholds: tuple = (0.05, 0.10, 0.15, 0.20)
    calibrators: dict = field(default_factory=dict)

    def fit(self, probs: np.ndarray, actual_max_return: np.ndarray):
        from signals.labels import cumulative_probs
        cum = cumulative_probs(probs)
        for t in self.cumulative_thresholds:
            key = f"p_ge_{int(t * 100)}"
            actual_hit = (actual_max_return >= t).astype(int)
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(cum[key], actual_hit)
            self.calibrators[key] = iso
        return self

    def transform(self, probs: np.ndarray) -> dict:
        """raw cumulative → isotonic-calibrated cumulative."""
        from signals.labels import cumulative_probs
        cum = cumulative_probs(probs)
        out = {}
        for key, iso in self.calibrators.items():
            out[key] = iso.predict(cum[key])
        return out


# ============================================================================
# 알림용 정확도 요약 텍스트
# ============================================================================
def format_accuracy_summary(report: dict, cumulative_thresholds: tuple = (0.05, 0.10, 0.15, 0.20)) -> str:
    """텔레그램 / NOTES 용 사용자 친화 정확도 요약."""
    lines = ["=== 시스템 정확도 ==="]
    if "reliability" in report:
        for t in cumulative_thresholds:
            key = f"p_ge_{int(t * 100)}"
            rel = report["reliability"].get(key)
            if rel is None or len(rel) == 0:
                continue
            # 가장 large bucket 의 pred vs actual
            best = rel.iloc[rel["n"].idxmax()]
            mark = "✓" if best["abs_deviation"] < 0.05 else "⚠"
            lines.append(
                f"  ≥+{int(t*100)}% 예측 {best['pred_avg']*100:.0f}% → 실제 {best['actual_rate']*100:.0f}% {mark}"
            )

    if "quantile_coverage" in report:
        qc = report["quantile_coverage"]
        mark = "✓" if abs(qc["deviation"]) < 0.05 else "⚠"
        lines.append(f"  95% CI 커버: {qc['coverage']*100:.0f}% (목표 95%) {mark}")

    if "brier" in report:
        lines.append(f"  Brier: {report['brier']:.3f} (낮을수록 좋음)")

    return "\n".join(lines)


# ============================================================================
# 직접 실행 — 학습된 모델로 calibration test
# ============================================================================
if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data.database import list_markets, load_candles
    from data.market_universe import signal_eligible_markets
    from signals.features import assemble_training_panel
    from signals.models.xgb_phase1 import prepare_features, train_full

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/upbit_d1.db")
    parser.add_argument("--btc-market", default="KRW-BTC")
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    print(f"loading {args.limit} markets...")
    markets = signal_eligible_markets(list_markets(args.db))[: args.limit]
    candles = {m: load_candles(args.db, m) for m in markets}
    candles = {k: v for k, v in candles.items() if len(v) > 0}
    btc = load_candles(args.db, args.btc_market)

    print("building panel + training...")
    panel = assemble_training_panel(candles, btc, normalize=True)
    panel = panel.sort_values("timestamp")
    n = len(panel)
    split = int(n * 0.8)
    train_panel = panel.iloc[:split]
    val_panel = panel.iloc[split:]

    model, _ = train_full(train_panel, val_ratio=0.1)
    X_val, y_val, _ = prepare_features(val_panel)
    actual_max = val_panel.dropna(subset=["label"])["max_return"].values

    probs = model.predict_proba(X_val)

    print("\n=== calibration report ===")
    report = calibration_report(probs, y_val, actual_max_return=actual_max)
    print(format_accuracy_summary(report))

    print("\n=== reliability (P(≥10%)) ===")
    if "reliability" in report:
        rel = report["reliability"]["p_ge_10"]
        print(rel.to_string(index=False) if len(rel) > 0 else "(empty)")

    print(f"\nBrier: {report['brier']:.4f}")
    print("\nPer-bin:")
    for k, v in report["per_bin"].items():
        print(f"  {k}: actual={v['n_actual']:>5d} pred={v['n_predicted']:>5d} "
              f"acc={v['accuracy']*100:5.1f}% prec={v['precision']*100:5.1f}%")

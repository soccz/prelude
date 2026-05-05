"""Multi-class 라벨 — 오늘 일봉의 max(high)/open 분포 기반.

설계 (SIGNAL.md §2):
  - 타겟: max(high)/open 의 multi-class bin (사용자 옵션 4)
  - 안정성 조건 X (가상 ledger 익절/손절 시뮬에서 처리, LEDGER §3)
  - bin 경계는 placeholder (CLAUDE.md §2.5) — EDA 후 사용자 컨펌
  - panel 학습용 (KRW + BINANCE 모두 동일 함수)

핵심 함수:
  - today_pump_label(open, high_max, bins) — single 코인 single 일봉
  - label_panel(df_d1, bins) — 전체 panel DataFrame
  - cumulative_probs(bin_probs) — bin 확률 → P(≥5%/10%/15%/20%)
  - expected_max_return(bin_probs, bin_centers) — 기대값
  - approx_ci_from_bins(bin_probs, bin_centers, alpha) — CI 근사
  - label_distribution(df) — bin 분포 통계 (EDA)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ============================================================================
# 디폴트 bin 경계 (placeholder, CLAUDE.md §2.5)
#
# Upbit KRW 252 코인 × 3년 EDA 결과 (2026-05-03):
#   max(high) ≥ open 이 100% — 음봉 bin 무의미 → 제거
#   분포 매우 skewed (median 2.16%, 99%=26%)
#   → bin 세분화 (A 옵션, 20루프 결정)
# ============================================================================
DEFAULT_BINS = (0.02, 0.05, 0.10, 0.15, 0.20)
# bin 0: max < 2%       (~50%)  "거의 안 움직임"
# bin 1: 2%   ≤ max < 5%   (~28%)  "조금 움직임"
# bin 2: 5%   ≤ max < 10%  (~15%)  "의미 있는 펌프"
# bin 3: 10%  ≤ max < 15%  (~4%)
# bin 4: 15%  ≤ max < 20%  (~1.6%)
# bin 5: 20%  ≤ max        (~1.8%)
#
# Fallback 계획 (라이브 4주 후 데이터 보고):
#   - bin 4/5 정확도 낮으면 → 합쳐서 5-bin
#   - bin 0/1 SHAP 기여 같으면 → 합쳐서 5-bin

# bin 중간값 (기대값 계산용)
# bin 0: < 2% → 평균 ~1% (실측 1.8%)
# bin 5: ≥20% → outlier 큰 코인 평균 37.7%, 보수적 30% 사용
DEFAULT_BIN_CENTERS = np.array([0.01, 0.035, 0.075, 0.125, 0.175, 0.30])


# ============================================================================
# Single 라벨
# ============================================================================
def today_pump_label(
    open_today: float,
    high_max_today: float,
    bins: tuple[float, ...] = DEFAULT_BINS,
) -> int:
    """
    오늘 일봉의 max_return = max(high)/open - 1 을 multi-class bin 으로.

    bin 0: max < bins[0]               (음봉)
    bin i: bins[i-1] ≤ max < bins[i]   (i = 1..len(bins)-1)
    bin len(bins): max ≥ bins[-1]      (최상위)
    """
    if open_today <= 0:
        return 0  # 비정상 데이터 → 음봉 처리
    max_return = high_max_today / open_today - 1
    for i, b in enumerate(bins):
        if max_return < b:
            return i
    return len(bins)


# ============================================================================
# Panel 라벨 (DataFrame)
# ============================================================================
def label_panel(
    df_d1: pd.DataFrame,
    bins: tuple[float, ...] = DEFAULT_BINS,
    label_col: str = "label",
    return_col: str = "max_return",
) -> pd.DataFrame:
    """
    DataFrame (panel) 에 multi-class 라벨 추가 — market별 shift(-1).

    panel[t] features = t 시점 (어제까지 본 정보)
    panel[t] label    = t+1 시점 (내일 일봉의 max_return)

    즉 "어제 보고 오늘 예측" 구조 (추론 흐름과 일치).
    look-ahead leak 방지 (CLAUDE.md §2.5 위생).

    df_d1 columns 기대:
        market, timestamp, open, high, low, close, volume, ...

    추가 column:
        max_return : (next day high) / (next day open) - 1
        label      : multi-class bin (0 ~ len(bins))

    참고: 마지막 row 의 max_return / label 은 NaN
          (다음 일봉 없음) → prepare_features 에서 dropna(label) 자동 제거.
    """
    df = df_d1.copy()

    # ===== 핵심: market별 shift(-1) =====
    # 단순 shift(-1) 는 다른 코인 데이터로 shift 됨 → 반드시 groupby
    if "market" not in df.columns:
        raise ValueError("label_panel requires 'market' column")
    df = df.sort_values(["market", "timestamp"]).reset_index(drop=True)
    g = df.groupby("market", sort=False)
    target_open = g["open"].shift(-1)
    target_high = g["high"].shift(-1)
    df[return_col] = target_high / target_open - 1

    bin_arr = np.array([-np.inf] + list(bins) + [np.inf])
    # NaN safe: digitize 가 NaN 처리 못하니 별도
    valid = df[return_col].notna()
    df[label_col] = np.nan
    df.loc[valid, label_col] = np.digitize(
        df.loc[valid, return_col].values, bin_arr
    ) - 1

    # 비정상 (next open <= 0) 처리
    bad = (target_open <= 0) | target_open.isna()
    df.loc[bad, label_col] = np.nan
    df.loc[bad, return_col] = np.nan

    return df


# ============================================================================
# Path-aware 라벨 (실행 룰 인식 — TP 먼저 도달 vs SL 먼저)
# ============================================================================
def label_panel_path_aware(
    df_d1: pd.DataFrame,
    df_4h: pd.DataFrame,
    tp_pct: float = 0.10,
    sl_pct: float = 0.05,
    sl_priority: bool = True,
    label_col: str = "label_path_tp_first",
    return_col: str = "net_return_under_rule",
) -> pd.DataFrame:
    """
    Path-aware 라벨 — 다음 일봉의 4h 봉 path 기반.

    panel[t] features = t 시점 (어제까지)
    panel[t] label    = t+1 일봉의 4h path 시뮬 결과:
        1 = TP_pct 먼저 도달 (TP first)
        0 = SL_pct 먼저 도달 또는 안 도달 (eod close)
    panel[t] net_return_under_rule = 같은 룰의 실제 net return (cost 미차감)

    df_4h: 4h 봉 panel (timestamp, market, open, high, low, close)
    same_bar 정책: sl_priority=True 면 같은 4h 봉 둘 다 시 SL 먼저 (보수)
    """
    df = df_d1.copy()
    if "market" not in df.columns:
        raise ValueError("requires market column")
    df = df.sort_values(["market", "timestamp"]).reset_index(drop=True)

    # next-day OHLC (시뮬용)
    g = df.groupby("market", sort=False)
    next_open = g["open"].shift(-1)
    next_date = g["timestamp"].shift(-1)

    # 4h panel index (market, day)
    df_4h = df_4h.copy()
    df_4h["timestamp"] = pd.to_datetime(df_4h["timestamp"])
    df_4h["day"] = df_4h["timestamp"].dt.normalize()
    g4 = df_4h.groupby(["market", "day"])

    labels = np.full(len(df), np.nan)
    rets = np.full(len(df), np.nan)

    for i, (market, ts, opn) in enumerate(zip(df["market"], next_date, next_open)):
        if pd.isna(opn) or opn <= 0 or pd.isna(ts):
            continue
        day_key = (market, pd.to_datetime(ts).normalize())
        if day_key not in g4.groups:
            continue
        bars = df_4h.loc[g4.groups[day_key]].sort_values("timestamp")
        if len(bars) == 0:
            continue

        tp_price = opn * (1 + tp_pct)
        sl_price = opn * (1 - sl_pct)
        ret = None
        outcome = 0  # default: not TP first

        for _, b in bars.iterrows():
            tp_hit = b["high"] >= tp_price
            sl_hit = b["low"] <= sl_price
            if tp_hit and sl_hit:
                if sl_priority:
                    ret = (sl_price - opn) / opn
                else:
                    ret = (tp_price - opn) / opn
                    outcome = 1
                break
            if tp_hit:
                ret = (tp_price - opn) / opn
                outcome = 1
                break
            if sl_hit:
                ret = (sl_price - opn) / opn
                break

        if ret is None:
            # eod close
            ret = (bars["close"].iloc[-1] - opn) / opn

        labels[i] = outcome
        rets[i] = ret

    df[label_col] = labels
    df[return_col] = rets
    return df


# ============================================================================
# Binary 라벨 (setup detector v2 — momentum-continuation)
# ============================================================================
def label_panel_binary(
    df_d1: pd.DataFrame,
    threshold: float = 0.10,
    label_col: str = "label_binary",
    return_col: str = "max_return",
) -> pd.DataFrame:
    """
    Binary 라벨: next_day_max(high) / next_day_open - 1 >= threshold → 1, else 0.

    panel[t] features = t 시점 (어제까지 본 정보)
    panel[t] label    = t+1 일봉의 max(high) ≥ threshold 도달 yes/no

    market별 shift(-1) — coin 간 leak 방지.

    추가 column:
        max_return    : (next day high) / (next day open) - 1 (이미 있으면 재사용)
        label_binary  : 0 / 1
    """
    df = df_d1.copy()
    if "market" not in df.columns:
        raise ValueError("label_panel_binary requires 'market' column")
    df = df.sort_values(["market", "timestamp"]).reset_index(drop=True)

    # max_return 없으면 계산
    if return_col not in df.columns:
        g = df.groupby("market", sort=False)
        target_open = g["open"].shift(-1)
        target_high = g["high"].shift(-1)
        df[return_col] = target_high / target_open - 1
        bad = (target_open <= 0) | target_open.isna()
        df.loc[bad, return_col] = np.nan

    # binary 라벨
    df[label_col] = (df[return_col] >= threshold).astype(float)
    # max_return NaN → label NaN
    df.loc[df[return_col].isna(), label_col] = np.nan

    return df


# ============================================================================
# 모델 출력 변환 (raw bin probs → user-facing distribution)
# ============================================================================
def cumulative_probs(bin_probs: np.ndarray, bins: tuple[float, ...] = DEFAULT_BINS) -> dict:
    """
    XGBoost predict_proba 결과 (n_samples, n_bins) → cumulative dict.

    return:
      {
        'p_ge_5':  ...,   # P(max ≥ 5%)  = sum(bin >= bin_index_for_5pct)
        'p_ge_10': ...,
        'p_ge_15': ...,
        'p_ge_20': ...,
      }
    """
    bin_probs = np.atleast_2d(bin_probs)

    # bin index 계산: 5% 가 어느 bin 부터인가?
    def bin_index_at_or_above(threshold: float) -> int:
        for i, b in enumerate(bins):
            if b >= threshold:
                return i + 1  # bin i+1 이 [bins[i], bins[i+1]) 이므로
        return len(bins) + 1

    return {
        f"p_ge_{int(t * 100)}": bin_probs[:, bin_index_at_or_above(t):].sum(axis=1)
        for t in (0.05, 0.10, 0.15, 0.20)
    }


def expected_max_return(
    bin_probs: np.ndarray, bin_centers: np.ndarray = DEFAULT_BIN_CENTERS
) -> np.ndarray:
    """
    bin 중간값 가중 평균 = 기대 max_return.

    bin_probs: (n_samples, n_bins)
    bin_centers: (n_bins,)
    """
    bin_probs = np.atleast_2d(bin_probs)
    return (bin_probs * bin_centers).sum(axis=1)


def approx_ci_from_bins(
    bin_probs: np.ndarray,
    bin_centers: np.ndarray = DEFAULT_BIN_CENTERS,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    bin 확률 분포 → 근사 quantile CI.

    각 sample 의 bin 분포에서 cdf 누적 → alpha/2 와 1-alpha/2 quantile bin 의 중심값.
    """
    bin_probs = np.atleast_2d(bin_probs)
    cdf = np.cumsum(bin_probs, axis=1)

    low_q = alpha / 2
    high_q = 1 - alpha / 2

    ci_low = np.empty(bin_probs.shape[0])
    ci_high = np.empty(bin_probs.shape[0])
    for i in range(bin_probs.shape[0]):
        # low quantile bin
        low_idx = np.searchsorted(cdf[i], low_q)
        low_idx = min(low_idx, len(bin_centers) - 1)
        ci_low[i] = bin_centers[low_idx]
        # high quantile bin
        high_idx = np.searchsorted(cdf[i], high_q)
        high_idx = min(high_idx, len(bin_centers) - 1)
        ci_high[i] = bin_centers[high_idx]
    return ci_low, ci_high


# ============================================================================
# EDA — bin 분포 통계
# ============================================================================
def label_distribution(
    df_with_label: pd.DataFrame,
    label_col: str = "label",
    return_col: str = "max_return",
) -> pd.DataFrame:
    """
    panel 라벨 분포 통계 (EDA — bin 경계 결정용).

    return: bin 별 row, count, ratio, 평균 max_return, std
    """
    g = df_with_label.groupby(label_col).agg(
        count=(return_col, "size"),
        mean_max_return=(return_col, "mean"),
        std_max_return=(return_col, "std"),
        median_max_return=(return_col, "median"),
    )
    g["ratio"] = g["count"] / g["count"].sum()
    return g.round(4)


# ============================================================================
# 직접 실행 — DB 에서 라벨 분포 EDA
# ============================================================================
if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data.database import load_candles, list_markets

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/upbit_d1.db")
    parser.add_argument("--coin", help="단일 코인 (없으면 전체)")
    parser.add_argument("--bins", help="bin 경계 (콤마, 예: 0.0,0.05,0.10,0.15,0.20)")
    args = parser.parse_args()

    bins = tuple(float(x) for x in args.bins.split(",")) if args.bins else DEFAULT_BINS
    print(f"bins: {bins}")

    if args.coin:
        df = load_candles(args.db, args.coin)
    else:
        # 전체 코인 panel
        markets = list_markets(args.db)
        print(f"loading {len(markets)} markets...")
        dfs = []
        for m in markets:
            d = load_candles(args.db, m)
            if len(d) > 0:
                d["market"] = m
                dfs.append(d)
        df = pd.concat(dfs, ignore_index=True)

    print(f"\nTotal rows: {len(df):,}")

    df = label_panel(df, bins=bins)
    print("\n=== 라벨 분포 ===")
    print(label_distribution(df).to_string())

    print("\n=== max_return 통계 ===")
    print(df["max_return"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).round(4))

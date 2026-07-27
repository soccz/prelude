"""피처 생성 — alt multi-lookback + BTC regime + panel 결합.

설계 (SIGNAL.md §3):
  - 알트 multi-lookback {3, 5, 7, 14, 21}일 — return / vol / range_contraction
  - BTC regime (전 코인 공통) — return_Nd / ma_distance / intensity / 4-state
  - cross-sectional rank normalization (코인별 피처)
  - rolling z-score (BTC 피처)
  - **fillna(0) 절대 X** (gan_t known gap)

핵심 함수:
  - compute_alt_features(df_d1)      — 단일 코인 시계열 피처
  - compute_btc_features(btc_d1)     — BTC 만 (전 코인 공통값)
  - compute_panel_features(...)      — 모든 코인 시계열 + BTC join
  - cross_sectional_normalize(panel) — per-day rank norm
  - assemble_training_panel(...)     — 학습용 panel + 라벨
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from data.market_universe import (
    signal_eligible_markets,
    signal_market_exclusions,
)

# ============================================================================
# Lookback 격자 (placeholder, CLAUDE.md §2.5)
# ============================================================================
LOOKBACKS = (3, 5, 7, 14, 21)
BTC_LOOKBACKS = (1, 3, 5, 7, 14, 21)
BTC_MA_WINDOW = 200          # SIGNAL §3.2 — placeholder
BTC_RV_WINDOW = 30
BTC_INTENSITY_LOOKBACK = 252  # 1년 분위
INTENSITY_THRESHOLD = 0.5

EPS = 1e-12


# ============================================================================
# 단일 코인 시계열 피처
# ============================================================================
def compute_alt_features(
    df: pd.DataFrame, lookbacks: tuple[int, ...] = LOOKBACKS
) -> pd.DataFrame:
    """
    한 코인의 일봉 DataFrame → 시계열 피처.

    df columns 기대: timestamp (sorted), open, high, low, close, volume, quote_volume

    추가 columns:
      log_return_1d
      return_{N}d, vol_{N}d for N in lookbacks
      range_contraction_{N}d for N in (3, 7, 14)
      vol_inv_7d
      volume_ratio_{N}d for N in (3, 7)
      volume_spike_score
      rsi_14, macd, macdhist
      bb_position, squeeze_on
      roc_3d, roc_7d
    """
    out = df.copy()
    out = out.sort_values("timestamp").reset_index(drop=True)

    close = out["close"]
    high = out["high"]
    low = out["low"]
    volume = out["volume"]

    # === 가격 수익률 ===
    out["log_return_1d"] = np.log(close / close.shift(1) + EPS)

    for N in lookbacks:
        out[f"return_{N}d"] = close / close.shift(N) - 1

    # === 변동성 (실현 표준편차) ===
    log_ret = out["log_return_1d"]
    for N in lookbacks:
        out[f"vol_{N}d"] = log_ret.rolling(N, min_periods=max(2, N // 2)).std()
    out["vol_inv_7d"] = 1.0 / (out["vol_7d"] + EPS)

    # === 고저폭 압축 (range_contraction) ===
    # 최근 N일 high-low 범위 / 그 이전 N일 범위
    for N in (3, 7, 14):
        recent_range = (high.rolling(N).max() - low.rolling(N).min()) / (close + EPS)
        prev_range = (
            high.shift(N).rolling(N).max() - low.shift(N).rolling(N).min()
        ) / (close.shift(N) + EPS)
        out[f"range_contraction_{N}d"] = prev_range / (recent_range + EPS)
        # > 1 이면 최근이 더 좁음 (= "조용해짐"), < 1 이면 더 넓음

    # === 거래량 ===
    for N in (3, 7):
        ma = volume.rolling(N).mean()
        out[f"volume_ratio_{N}d"] = volume / (ma + EPS)
    # volume_spike_score: pct_change 의 표준화
    vol_pct = volume.pct_change()
    out["volume_spike_score"] = vol_pct / (vol_pct.rolling(20).std() + EPS)

    # === 기술지표 (간단 구현, pandas-ta 의존성 없이) ===
    # RSI 14
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(14, min_periods=7).mean()
    avg_loss = loss.rolling(14, min_periods=7).mean()
    rs = avg_gain / (avg_loss + EPS)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    macd_signal = out["macd"].ewm(span=9, adjust=False).mean()
    out["macdhist"] = out["macd"] - macd_signal

    # Bollinger position + squeeze
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    out["bb_position"] = (close - bb_lower) / (bb_upper - bb_lower + EPS)
    bb_bandwidth = (bb_upper - bb_lower) / (bb_mid + EPS)
    out["squeeze_on"] = (
        bb_bandwidth.rolling(24, min_periods=10).min() == bb_bandwidth
    ).astype(int)

    # ROC
    out["roc_3d"] = close.pct_change(3) * 100
    out["roc_7d"] = close.pct_change(7) * 100

    # ATR (volatility-scaled TP/SL 용)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    out["atr_14"] = tr.rolling(14).mean()
    out["atr_pct_14"] = out["atr_14"] / (close + EPS)  # close 대비 ATR 비율

    return out


# ============================================================================
# BTC regime 피처 (전 코인 공통)
# ============================================================================
def compute_btc_features(btc_d1: pd.DataFrame) -> pd.DataFrame:
    """
    BTC 일봉 → BTC regime 피처 (전 코인 공통).

    df columns 기대: timestamp, open, high, low, close, volume

    추가 columns:
      btc_return_{N}d for N in BTC_LOOKBACKS
      btc_ma_distance        (= (close - MA200)/MA200)
      btc_rv_30d
      btc_intensity_d        (= rv 30 의 252일 분위, 0~1)
      btc_regime             (4-state: bull_quiet/bull_volatile/bear_quiet/bear_volatile)
      btc_regime_id          (0/1/2/3 — FiLM 용)
    """
    df = btc_d1.copy().sort_values("timestamp").reset_index(drop=True)
    close = df["close"]
    log_ret = np.log(close / close.shift(1) + EPS)

    # 추세 (return + MA distance)
    for N in BTC_LOOKBACKS:
        df[f"btc_return_{N}d"] = close / close.shift(N) - 1

    ma = close.rolling(BTC_MA_WINDOW).mean()
    df["btc_ma_distance"] = (close - ma) / (ma + EPS)

    # 변동성 + intensity
    df["btc_rv_30d"] = log_ret.rolling(BTC_RV_WINDOW).std()
    df["btc_intensity_d"] = (
        df["btc_rv_30d"].rolling(BTC_INTENSITY_LOOKBACK).rank(pct=True)
    )

    # 4-state regime
    bull = df["btc_ma_distance"] > 0
    volatile = df["btc_intensity_d"] > INTENSITY_THRESHOLD

    def _regime(b, v):
        if pd.isna(b) or pd.isna(v):
            return "unknown"
        if b and not v:
            return "bull_quiet"
        if b and v:
            return "bull_volatile"
        if not b and not v:
            return "bear_quiet"
        return "bear_volatile"

    df["btc_regime"] = [_regime(b, v) for b, v in zip(bull, volatile)]
    regime_id_map = {
        "bull_quiet": 0,
        "bull_volatile": 1,
        "bear_quiet": 2,
        "bear_volatile": 3,
        "unknown": -1,
    }
    df["btc_regime_id"] = df["btc_regime"].map(regime_id_map)

    # 학습용으로 BTC OHLC 등 원본 가격 컬럼 제거 (피처만 남김)
    btc_feature_cols = [c for c in df.columns if c.startswith("btc_") or c == "timestamp"]
    return df[btc_feature_cols]


# ============================================================================
# Panel features 결합
# ============================================================================
def compute_panel_features(
    candles_by_market: dict[str, pd.DataFrame],
    btc_features_df: pd.DataFrame,
    lookbacks: tuple[int, ...] = LOOKBACKS,
) -> pd.DataFrame:
    """
    여러 코인의 시계열 피처 + BTC regime join → 전체 panel.

    candles_by_market: {market_id: df_d1}  (예: 'KRW-BTC' or 'BINANCE-BTCUSDT')
    btc_features_df: compute_btc_features 결과 (timestamp 기준)

    return: long-format panel
      columns: market, timestamp, open, high, low, close, volume,
               <alt features>, <btc features>
    """
    panels = []
    for market, df in candles_by_market.items():
        if len(df) < max(lookbacks) + 5:
            continue
        feat = compute_alt_features(df, lookbacks=lookbacks)
        feat["market"] = market
        panels.append(feat)

    if not panels:
        return pd.DataFrame()

    panel = pd.concat(panels, ignore_index=True)

    # BTC features merge (timestamp 기준)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    btc_features_df = btc_features_df.copy()
    btc_features_df["timestamp"] = pd.to_datetime(btc_features_df["timestamp"])
    panel = panel.merge(btc_features_df, on="timestamp", how="left")

    return panel


# ============================================================================
# Cross-sectional normalization
# ============================================================================
def cross_sectional_normalize(
    panel: pd.DataFrame,
    alt_cols: list[str],
    rolling_z_cols: Optional[list[str]] = None,
    z_window: int = 60,
) -> pd.DataFrame:
    """
    panel 정규화 (SIGNAL.md §3.4).

    - alt_cols: 코인별 피처 → per-day cross-sectional rank (0~1)
    - rolling_z_cols: BTC regime 같은 전역 피처 → rolling z-score (시간축)
    - **fillna(0) 절대 X** (그대로 NaN 유지 — XGBoost native 처리)
    """
    out = panel.copy()

    # cross-sectional rank (per-day)
    for col in alt_cols:
        if col in out.columns:
            out[col] = out.groupby("timestamp")[col].rank(pct=True)

    # rolling z (per-feature, time axis)
    if rolling_z_cols:
        for col in rolling_z_cols:
            if col in out.columns:
                # BTC features 는 timestamp 별 동일 값이라 그냥 시간축으로
                tmp = out.drop_duplicates("timestamp").sort_values("timestamp")
                ma = tmp[col].rolling(z_window).mean()
                std = tmp[col].rolling(z_window).std()
                z = (tmp[col] - ma) / (std + EPS)
                z_map = dict(zip(tmp["timestamp"], z))
                out[f"{col}_z"] = out["timestamp"].map(z_map)

    return out


# ============================================================================
# 학습용 panel 조립 (label + features)
# ============================================================================
def assemble_training_panel(
    candles_by_market: dict[str, pd.DataFrame],
    btc_d1: pd.DataFrame,
    bins: tuple[float, ...] = (0.02, 0.05, 0.10, 0.15, 0.20),
    normalize: bool = True,
) -> pd.DataFrame:
    """
    학습용 panel 한 번에 조립.

    candles_by_market: 모든 코인 일봉 (KRW-* + BINANCE-* 다 가능)
    btc_d1: BTC 일봉 (regime 피처용)

    return: panel with label + features + market + timestamp
    """
    from signals.labels import label_panel  # 순환 import 방지

    excluded = signal_market_exclusions(candles_by_market)
    if excluded:
        raise ValueError(
            "candles_by_market contains excluded signal markets: "
            + ", ".join(excluded)
        )

    # 1. BTC features
    btc_feat = compute_btc_features(btc_d1)

    # 2. Panel features
    panel = compute_panel_features(candles_by_market, btc_feat)
    if len(panel) == 0:
        return panel

    # 3. Label
    panel = label_panel(panel, bins=bins)

    # 4. Normalize (선택)
    if normalize:
        # 코인별 피처 (rank)
        rank_cols = [c for c in panel.columns if any(
            c.startswith(p) for p in (
                "return_", "vol_", "range_contraction_", "vol_inv_",
                "volume_ratio_", "volume_spike", "rsi_", "macd",
                "bb_position", "roc_",
            )
        )]
        z_cols = [c for c in panel.columns if c.startswith("btc_") and c != "btc_regime"]
        panel = cross_sectional_normalize(
            panel, alt_cols=rank_cols, rolling_z_cols=z_cols
        )

    return panel


# ============================================================================
# 직접 실행 — panel 검증
# ============================================================================
if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data.database import load_candles, list_markets

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/upbit_d1.db")
    parser.add_argument("--btc-market", default="KRW-BTC")
    parser.add_argument("--limit", type=int, default=10, help="검증용 샘플 코인 갯수")
    args = parser.parse_args()

    print("loading...")
    markets = signal_eligible_markets(list_markets(args.db))[: args.limit]
    candles = {}
    for m in markets:
        d = load_candles(args.db, m)
        if len(d) > 0:
            candles[m] = d

    btc = load_candles(args.db, args.btc_market)

    print(f"\nbuilding panel from {len(candles)} markets ({len(btc)} BTC rows)...")
    panel = assemble_training_panel(candles, btc, normalize=True)

    print(f"\nPanel shape: {panel.shape}")
    print(f"Columns ({len(panel.columns)}):")
    print("  ", ", ".join(panel.columns.tolist()))
    print("\n=== 라벨 분포 ===")
    print(panel["label"].value_counts().sort_index())
    print("\n=== 샘플 (5 행) ===")
    sample_cols = [
        "market", "timestamp", "label", "return_5d", "range_contraction_7d",
        "btc_return_5d", "btc_intensity_d", "btc_regime",
    ]
    print(panel[sample_cols].tail(5).to_string(index=False))

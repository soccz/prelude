"""Distribution Engine — multi-head label 정의.

Core heads (사용자 확정 2026-05-03):
  h1_stable_3_4h    (T=3, C=2, DD=2, H=4)   — 즉발 안정
  h2_hit_3_4h       (T=3, hit-only, H=4)    — 즉발 hit
  h3_eod_3_24h      (T=3, C=3, DD=3, H=24)  — 종가 +3% 유지
  h4_eod_5_24h      (T=5, C=3, DD=3, H=24)  — 5% 종가 유지
  h5_tail_20        (T=20, hit-only, H=24)  — rare tail (= detector_v1)

Aux heads:
  h6_hit_5_24h      (T=5, hit-only, H=24)   — 5% hit-only
  h7_mid_7_24h      (T=7, C=0, DD=3, H=24)  — 7% mid-strong

label 정의는 4h 봉 기반 (label_space_discovery_v1/v2 와 동일):
  hit:        첫 4h 봉의 high >= open*(1+T/100), 누적 시간 <= H
  pass_dd:    bars[0..hit_idx] 의 모든 low >= open*(1-DD/100)
  pass_close: close_t >= open*(1+C/100)
  success = hit AND pass_dd AND pass_close
  hit-only label = hit (DD/C 무관)

사용:
    from signals.labels_distribution import HEADS, compute_distribution_labels
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAX_BARS = 6  # 24h / 4h


HEADS = {
    "h1_stable_3_4h":  {"T": 3,  "C": 2, "DD": 2, "H": 4,  "kind": "success", "name": "즉발 +3% 안정"},
    "h2_hit_3_4h":     {"T": 3,  "C": 0, "DD": 999, "H": 4,  "kind": "hit",     "name": "즉발 +3% hit"},
    "h3_eod_3_24h":    {"T": 3,  "C": 3, "DD": 3, "H": 24, "kind": "success", "name": "종가 +3% 유지"},
    "h4_eod_5_24h":    {"T": 5,  "C": 3, "DD": 3, "H": 24, "kind": "success", "name": "종가 +5% 유지"},
    "h5_tail_20":      {"T": 20, "C": 0, "DD": 999, "H": 24, "kind": "hit",     "name": "rare +20% tail"},
    # aux
    "h6_hit_5_24h":    {"T": 5,  "C": 0, "DD": 999, "H": 24, "kind": "hit",     "name": "+5% hit"},
    "h7_mid_7_24h":    {"T": 7,  "C": 0, "DD": 3, "H": 24, "kind": "success", "name": "+7% mid-strong"},
}

CORE_HEADS = ["h1_stable_3_4h", "h2_hit_3_4h", "h3_eod_3_24h", "h4_eod_5_24h", "h5_tail_20"]
AUX_HEADS = ["h6_hit_5_24h", "h7_mid_7_24h"]


def _compute_one_label(highs: np.ndarray, lows: np.ndarray,
                        opens: np.ndarray, closes: np.ndarray,
                        T: float, C: float, DD: float, H: int, kind: str) -> np.ndarray:
    """vectorized label per row. returns float array (1.0/0.0/NaN)."""
    n_bars_in_H = max(1, H // 4)
    n_bars_in_H = min(n_bars_in_H, MAX_BARS)
    target = opens * (1 + T / 100)
    stop = opens * (1 - DD / 100)

    hit_mask = highs[:, :n_bars_in_H] >= target[:, None]
    hit_any = hit_mask.any(axis=1)

    if kind == "hit":
        label = hit_any.astype(float)
        # NaN 처리: opens 가 NaN 이면 라벨 NaN
        label[np.isnan(opens)] = np.nan
        return label

    # success kind
    first_hit = np.argmax(hit_mask, axis=1)
    first_hit[~hit_any] = -1

    N = len(opens)
    pre_hit_low = np.full(N, np.nan)
    for i in range(n_bars_in_H):
        rows_i = (first_hit == i)
        if rows_i.any():
            pre_hit_low[rows_i] = np.nanmin(lows[rows_i, : i + 1], axis=1)

    pass_dd = pre_hit_low >= stop
    pass_close = closes >= opens * (1 + C / 100)
    success = (hit_any & pass_dd & pass_close).astype(float)
    success[np.isnan(opens)] = np.nan
    return success


def compute_distribution_labels(panel_4h_bars: pd.DataFrame) -> pd.DataFrame:
    """panel_4h_bars: rows = (market, date) with columns:
         open, close, highs (np.ndarray of MAX_BARS), lows (np.ndarray of MAX_BARS)

    returns DataFrame with one column per head (head_name → 0/1/NaN).
    """
    highs = np.vstack(panel_4h_bars["highs"].values)
    lows = np.vstack(panel_4h_bars["lows"].values)
    open_col = "open_4h" if "open_4h" in panel_4h_bars.columns else "open"
    close_col = "close_4h" if "close_4h" in panel_4h_bars.columns else "close"
    opens = panel_4h_bars[open_col].values.astype(float)
    closes = panel_4h_bars[close_col].values.astype(float)

    out = {}
    for head_name, cfg in HEADS.items():
        out[head_name] = _compute_one_label(
            highs, lows, opens, closes,
            cfg["T"], cfg["C"], cfg["DD"], cfg["H"], cfg["kind"]
        )
    return pd.DataFrame(out, index=panel_4h_bars.index)

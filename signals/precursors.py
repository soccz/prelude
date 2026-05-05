"""Sub-daily precursor feature builders.

This module centralizes the 4h/1h/15m precursor builders that were first
prototyped in research scripts. Keeping them here avoids script-to-script
imports and makes ablation/cache workflows repeatable.

All returned rows use:
  - market
  - date_only = daily feature date X

Labels for X+1 must still be shifted by the caller. Precursor rows are not
shifted because they represent late intraday information available before the
next KST 09:00 decision point.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from signals.labels_distribution import MAX_BARS


FOUR_H_FEATURES = [
    "pre_vol_ratio",
    "pre_price_change",
    "pre_range_pct",
    "pre_close_position",
    "pre_day_to_bar5_max",
    "pre_day_to_bar5_dd",
    "pre_bar5_close_vs_day_high",
]

ONE_H_FEATURES = [
    "pre1h_ret",
    "pre1h_vol_ratio",
    "pre1h_range_pct",
    "pre1h_close_position",
    "pre3h_ret",
    "pre3h_vol_ratio",
    "breakout_24h_high",
    "pre1h_btc_ret",
    "pre1h_breadth",
    "cum_max_ret_to_08h",
    "cum_dd_to_08h",
    "close07_vs_cum_high",
    "close07_vs_day_open",
    "range_5_to_8_expansion",
    "cum_volume_to_08h",
]

FIFTEEN_M_FEATURES = [
    "pre15m_ret",
    "pre15m_vol_ratio",
    "pre15m_range_pct",
    "pre15m_close_position",
    "pre30m_ret",
    "pre30m_vol_ratio",
    "pre1h_ret",
    "pre1h_vol_ratio",
    "pre3h_ret",
    "pre3h_vol_ratio",
    "pre3h_range_pct",
    "breakout_24h_high",
    "pre15m_btc_ret",
    "pre15m_breadth",
    "cum_max_ret_to_0830",
    "cum_dd_to_0830",
    "close0830_vs_cum_high",
    "close0830_vs_day_open",
    "cum_volume_to_0830",
]

FIFTEEN_M_HIT_THRESHOLDS = {
    "t3": 0.03,
    "t5": 0.05,
    "t10": 0.10,
    "t20": 0.20,
}


def cached_frame(path: str | Path, build_fn: Callable[[], pd.DataFrame],
                 refresh: bool = False) -> pd.DataFrame:
    """Load a cached DataFrame or build and cache it.

    Pickle is intentional here: these research frames include Python date
    objects and, for label/path panels, ndarray columns. No optional parquet
    dependency is required.
    """
    p = Path(path)
    if p.exists() and not refresh:
        return pd.read_pickle(p)
    df = build_fn()
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(p)
    return df


def prefix_columns(df: pd.DataFrame, cols: list[str], prefix: str) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    rename = {c: f"{prefix}{c}" for c in cols}
    out = out.rename(columns=rename)
    return out, list(rename.values())


def build_15m_event_table(candles_15m: dict, min_bars: int = 80) -> pd.DataFrame:
    """Build one row per (market, KST daily candle date) with first-hit timing.

    The daily candle date is the date of the KST 09:00 open. For example,
    timestamp 2026-05-06 08:45 belongs to daily candle date 2026-05-05.
    """
    rows = []
    for market, df in candles_15m.items():
        if df is None or len(df) == 0:
            continue
        df = df.sort_values("timestamp").copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = (df["timestamp"] - pd.Timedelta(hours=9)).dt.date.astype(str)

        for date, g in df.groupby("date", sort=False):
            g = g.sort_values("timestamp").reset_index(drop=True)
            if len(g) < min_bars:
                continue
            first_ts = pd.Timestamp(g.iloc[0]["timestamp"])
            if first_ts.hour != 9 or first_ts.minute != 0:
                # Incomplete left edge: cannot trust first-hit timing.
                continue

            n = min(len(g), 96)
            opens = g["open"].iloc[:n].to_numpy(dtype=float)
            highs = g["high"].iloc[:n].to_numpy(dtype=float)
            closes = g["close"].iloc[:n].to_numpy(dtype=float)
            if len(opens) == 0 or opens[0] <= 0:
                continue

            open0 = float(opens[0])
            row = {
                "date": str(date),
                "coin": market,
                "n_bars": int(n),
                "open": open0,
                "max_return_pct": (float(np.nanmax(highs)) / open0 - 1) * 100,
                "close_return_pct": (float(closes[-1]) / open0 - 1) * 100,
            }
            for name, threshold in FIFTEEN_M_HIT_THRESHOLDS.items():
                hit_idx = np.flatnonzero(highs >= open0 * (1 + threshold))
                if len(hit_idx) == 0:
                    row[f"{name}_hit"] = 0
                    row[f"{name}_first_bar15"] = np.nan
                    row[f"{name}_first_minute"] = np.nan
                else:
                    idx = int(hit_idx[0])
                    row[f"{name}_hit"] = 1
                    row[f"{name}_first_bar15"] = idx
                    row[f"{name}_first_minute"] = idx * 15
            rows.append(row)

    return pd.DataFrame(rows)


def build_4h_label_panel(candles_4h: dict, btc_regime_map: dict) -> pd.DataFrame:
    """Build 4h padded bars for distribution labels/path backtests."""
    rows = []
    for market, df in candles_4h.items():
        if df is None or len(df) < 1:
            continue
        df = df.sort_values("timestamp").copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["bar_date"] = (df["timestamp"] - pd.Timedelta(hours=9)).dt.date
        for date, g in df.groupby("bar_date", sort=False):
            g2 = g.sort_values("timestamp").reset_index(drop=True)
            n = min(len(g2), MAX_BARS)
            if n < 1:
                continue
            opens = g2["open"].values.astype(float)
            closes = g2["close"].values.astype(float)
            highs = g2["high"].values.astype(float)[:MAX_BARS]
            lows = g2["low"].values.astype(float)[:MAX_BARS]
            highs_p = np.full(MAX_BARS, np.nan)
            lows_p = np.full(MAX_BARS, np.nan)
            highs_p[:n] = highs[:n]
            lows_p[:n] = lows[:n]
            open0 = float(opens[0])
            rows.append({
                "market": market,
                "date_only": date,
                "open_4h": open0,
                "close_4h": float(closes[min(n, len(closes)) - 1]),
                "highs": highs_p,
                "lows": lows_p,
                "btc_regime_4h": btc_regime_map.get(date, "unknown"),
                "n_bars": n,
                "eod_ret_4h": float(closes[min(n, len(closes)) - 1]) / open0 - 1 if open0 > 0 else np.nan,
                "max_ret_4h": float(highs[:n].max()) / open0 - 1 if open0 > 0 else np.nan,
            })
    return pd.DataFrame(rows)


def build_4h_precursor(candles_4h: dict, btc_regime_map: dict) -> pd.DataFrame:
    """Build 4h T-3.5h precursor features from bar index 4."""
    rows = []
    for market, df in candles_4h.items():
        if df is None or len(df) < 1:
            continue
        df = df.sort_values("timestamp").copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["bar_date"] = (df["timestamp"] - pd.Timedelta(hours=9)).dt.date
        for date, g in df.groupby("bar_date", sort=False):
            g2 = g.sort_values("timestamp").reset_index(drop=True)
            n = min(len(g2), MAX_BARS)
            if n < 5:
                continue
            opens = g2["open"].values.astype(float)
            closes = g2["close"].values.astype(float)
            highs = g2["high"].values.astype(float)
            lows = g2["low"].values.astype(float)
            vols = g2["volume"].values.astype(float) if "volume" in g2.columns else np.zeros(n)

            day_open = float(opens[0])
            bar_open = float(opens[4])
            bar_close = float(closes[4])
            bar_high = float(highs[4])
            bar_low = float(lows[4])
            bar_vol = float(vols[4])
            mean_vol_1_4 = float(vols[:4].mean()) if len(vols[:4]) > 0 else 1e-9
            day_max_to_bar5 = float(highs[:5].max())
            day_min_to_bar5 = float(lows[:5].min())
            denom_pos = max(bar_high - bar_low, 1e-9)

            rows.append({
                "market": market,
                "date_only": date,
                "pre_vol_ratio": bar_vol / max(mean_vol_1_4, 1e-9),
                "pre_price_change": bar_close / max(bar_open, 1e-9) - 1,
                "pre_range_pct": (bar_high - bar_low) / max(bar_open, 1e-9),
                "pre_close_position": (bar_close - bar_low) / denom_pos,
                "pre_day_to_bar5_max": day_max_to_bar5 / max(day_open, 1e-9) - 1,
                "pre_day_to_bar5_dd": day_min_to_bar5 / max(day_open, 1e-9) - 1,
                "pre_bar5_close_vs_day_high": bar_close / max(day_max_to_bar5, 1e-9) - 1,
            })
    return pd.DataFrame(rows)


def build_1h_precursor(candles_1h: dict, btc_df: pd.DataFrame) -> pd.DataFrame:
    """Build KST 08:00 snapshot precursor features."""
    btc = btc_df.sort_values("timestamp").copy()
    btc["timestamp"] = pd.to_datetime(btc["timestamp"])
    btc["btc_1h_ret"] = btc["close"] / btc["close"].shift(1) - 1
    btc_ret_map = dict(zip(btc["timestamp"], btc["btc_1h_ret"]))

    rows = []
    for market, df in candles_1h.items():
        if df is None or len(df) < 30:
            continue
        df = df.sort_values("timestamp").copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["bar_date"] = (df["timestamp"] - pd.Timedelta(hours=9)).dt.date
        df["vol_24h_mean_prior"] = df["volume"].rolling(24, min_periods=12).mean().shift(1)
        df["high_24h_prior"] = df["high"].rolling(24, min_periods=12).max().shift(1)

        for date, g in df.groupby("bar_date", sort=False):
            g = g.sort_values("timestamp").reset_index(drop=True)
            target_ts_date = pd.Timestamp(date) + pd.Timedelta(days=1)
            target_ts = target_ts_date.replace(hour=7)
            bar = g[g["timestamp"] == target_ts]
            if len(bar) == 0:
                continue
            r = bar.iloc[0]
            three_h = g[(g["timestamp"].dt.hour.isin([5, 6, 7])) &
                        (g["timestamp"].dt.date == target_ts_date.date())]
            if len(three_h) < 1:
                continue

            opn = float(r["open"]) if r["open"] > 0 else np.nan
            cls = float(r["close"])
            hi = float(r["high"])
            lo = float(r["low"])
            vol = float(r["volume"])
            vol_24h_mean = float(r["vol_24h_mean_prior"])
            high_24h_prior = float(r["high_24h_prior"])

            pre1h_ret = (cls / opn - 1) if pd.notna(opn) else np.nan
            pre1h_vol_ratio = (vol / vol_24h_mean) if (pd.notna(vol_24h_mean) and vol_24h_mean > 0) else np.nan
            pre1h_range_pct = ((hi - lo) / opn) if (pd.notna(opn) and opn > 0) else np.nan
            pre1h_close_position = (cls - lo) / max(hi - lo, 1e-9)

            three_h_first_open = float(three_h.iloc[0]["open"]) if three_h.iloc[0]["open"] > 0 else np.nan
            pre3h_ret = (cls / three_h_first_open - 1) if pd.notna(three_h_first_open) else np.nan
            three_h_vol_sum = float(three_h["volume"].sum())
            pre3h_vol_ratio = (three_h_vol_sum / (vol_24h_mean * 3)) if (
                pd.notna(vol_24h_mean) and vol_24h_mean > 0
            ) else np.nan
            breakout_24h_high = (cls / high_24h_prior - 1) if (
                pd.notna(high_24h_prior) and high_24h_prior > 0
            ) else np.nan
            pre1h_btc_ret = btc_ret_map.get(target_ts, np.nan)

            day_bars_until_08h = g[g["timestamp"] <= target_ts]
            if len(day_bars_until_08h) >= 1:
                day_open = float(day_bars_until_08h.iloc[0]["open"]) if day_bars_until_08h.iloc[0]["open"] > 0 else np.nan
                cum_high = float(day_bars_until_08h["high"].max())
                cum_low = float(day_bars_until_08h["low"].min())
                cum_vol = float(day_bars_until_08h["volume"].sum())
            else:
                day_open = np.nan
                cum_high = np.nan
                cum_low = np.nan
                cum_vol = np.nan

            cum_max_ret_to_08h = (cum_high / day_open - 1) if (pd.notna(day_open) and day_open > 0) else np.nan
            cum_dd_to_08h = (cum_low / day_open - 1) if (pd.notna(day_open) and day_open > 0) else np.nan
            close07_vs_cum_high = (cls / cum_high - 1) if (pd.notna(cum_high) and cum_high > 0) else np.nan
            close07_vs_day_open = (cls / day_open - 1) if (pd.notna(day_open) and day_open > 0) else np.nan
            range_5_to_8_expansion = ((three_h["high"].max() - three_h["low"].min()) / opn) if (
                pd.notna(opn) and opn > 0 and len(three_h) > 0
            ) else np.nan
            cum_volume_to_08h = (cum_vol / (vol_24h_mean * 23)) if (
                pd.notna(vol_24h_mean) and vol_24h_mean > 0
            ) else np.nan

            rows.append({
                "market": market,
                "date_only": date,
                "ts": target_ts,
                "pre1h_ret": pre1h_ret,
                "pre1h_vol_ratio": pre1h_vol_ratio,
                "pre1h_range_pct": pre1h_range_pct,
                "pre1h_close_position": pre1h_close_position,
                "pre3h_ret": pre3h_ret,
                "pre3h_vol_ratio": pre3h_vol_ratio,
                "breakout_24h_high": breakout_24h_high,
                "pre1h_btc_ret": pre1h_btc_ret,
                "cum_max_ret_to_08h": cum_max_ret_to_08h,
                "cum_dd_to_08h": cum_dd_to_08h,
                "close07_vs_cum_high": close07_vs_cum_high,
                "close07_vs_day_open": close07_vs_day_open,
                "range_5_to_8_expansion": range_5_to_8_expansion,
                "cum_volume_to_08h": cum_volume_to_08h,
            })
    df_pre = pd.DataFrame(rows)
    if len(df_pre) == 0:
        return df_pre
    df_pre["breadth_indicator"] = (df_pre["pre1h_ret"] > 0).astype(float)
    breadth = df_pre.groupby("ts")["breadth_indicator"].mean().rename("pre1h_breadth")
    return df_pre.merge(breadth, on="ts", how="left").drop(columns=["breadth_indicator", "ts"])


def build_15m_precursor(candles_15m: dict, btc_df: pd.DataFrame) -> pd.DataFrame:
    """Build KST 08:30 snapshot precursor features."""
    btc = btc_df.sort_values("timestamp").copy()
    btc["timestamp"] = pd.to_datetime(btc["timestamp"])
    btc["btc_1h_ret_to_bar"] = btc["close"] / btc["open"].shift(3) - 1
    btc_ret_map = dict(zip(btc["timestamp"], btc["btc_1h_ret_to_bar"]))

    rows = []
    for market, df in candles_15m.items():
        if df is None or len(df) < 100:
            continue
        df = df.sort_values("timestamp").copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["bar_date"] = (df["timestamp"] - pd.Timedelta(hours=9)).dt.date
        df["vol_24h_mean_prior"] = df["volume"].rolling(96, min_periods=48).mean().shift(1)
        df["high_24h_prior"] = df["high"].rolling(96, min_periods=48).max().shift(1)

        for date, g in df.groupby("bar_date", sort=False):
            g = g.sort_values("timestamp").reset_index(drop=True)
            target_ts_date = pd.Timestamp(date) + pd.Timedelta(days=1)
            target_ts = target_ts_date.replace(hour=8, minute=15)
            bar = g[g["timestamp"] == target_ts]
            if len(bar) == 0:
                continue
            r = bar.iloc[0]

            win_30m = g[(g["timestamp"] >= target_ts_date.replace(hour=8, minute=0)) & (g["timestamp"] <= target_ts)]
            win_1h = g[(g["timestamp"] >= target_ts_date.replace(hour=7, minute=30)) & (g["timestamp"] <= target_ts)]
            win_3h = g[(g["timestamp"] >= target_ts_date.replace(hour=5, minute=30)) & (g["timestamp"] <= target_ts)]
            day_bars = g[g["timestamp"] <= target_ts]

            opn = float(r["open"]) if r["open"] > 0 else np.nan
            cls = float(r["close"])
            hi = float(r["high"])
            lo = float(r["low"])
            vol = float(r["volume"])
            vol_24h_mean = float(r["vol_24h_mean_prior"]) if pd.notna(r["vol_24h_mean_prior"]) else np.nan
            high_24h_prior = float(r["high_24h_prior"]) if pd.notna(r["high_24h_prior"]) else np.nan

            pre15m_ret = (cls / opn - 1) if pd.notna(opn) else np.nan
            pre15m_vol_ratio = (vol / vol_24h_mean) if (pd.notna(vol_24h_mean) and vol_24h_mean > 0) else np.nan
            pre15m_range_pct = ((hi - lo) / opn) if (pd.notna(opn) and opn > 0) else np.nan
            pre15m_close_position = (cls - lo) / max(hi - lo, 1e-9)

            if len(win_30m) >= 1:
                w30_open = float(win_30m.iloc[0]["open"])
                pre30m_ret = (cls / w30_open - 1) if w30_open > 0 else np.nan
                pre30m_vol_ratio = (float(win_30m["volume"].sum()) / (vol_24h_mean * 2)) if (
                    pd.notna(vol_24h_mean) and vol_24h_mean > 0
                ) else np.nan
            else:
                pre30m_ret = np.nan
                pre30m_vol_ratio = np.nan

            if len(win_1h) >= 1:
                w1h_open = float(win_1h.iloc[0]["open"])
                pre1h_ret = (cls / w1h_open - 1) if w1h_open > 0 else np.nan
                pre1h_vol_ratio = (float(win_1h["volume"].sum()) / (vol_24h_mean * 4)) if (
                    pd.notna(vol_24h_mean) and vol_24h_mean > 0
                ) else np.nan
            else:
                pre1h_ret = np.nan
                pre1h_vol_ratio = np.nan

            if len(win_3h) >= 1:
                w3h_open = float(win_3h.iloc[0]["open"])
                pre3h_ret = (cls / w3h_open - 1) if w3h_open > 0 else np.nan
                pre3h_vol_sum = float(win_3h["volume"].sum())
                pre3h_vol_ratio = (pre3h_vol_sum / (vol_24h_mean * 12)) if (
                    pd.notna(vol_24h_mean) and vol_24h_mean > 0
                ) else np.nan
                pre3h_range_pct = ((float(win_3h["high"].max()) - float(win_3h["low"].min())) / w3h_open) if w3h_open > 0 else np.nan
            else:
                pre3h_ret = np.nan
                pre3h_vol_ratio = np.nan
                pre3h_range_pct = np.nan

            breakout_24h_high = (cls / high_24h_prior - 1) if (
                pd.notna(high_24h_prior) and high_24h_prior > 0
            ) else np.nan
            pre15m_btc_ret = btc_ret_map.get(target_ts, np.nan)

            if len(day_bars) >= 1:
                day_open = float(day_bars.iloc[0]["open"]) if day_bars.iloc[0]["open"] > 0 else np.nan
                cum_high = float(day_bars["high"].max())
                cum_low = float(day_bars["low"].min())
                cum_vol = float(day_bars["volume"].sum())
            else:
                day_open = np.nan
                cum_high = np.nan
                cum_low = np.nan
                cum_vol = np.nan

            rows.append({
                "market": market,
                "date_only": date,
                "ts": target_ts,
                "pre15m_ret": pre15m_ret,
                "pre15m_vol_ratio": pre15m_vol_ratio,
                "pre15m_range_pct": pre15m_range_pct,
                "pre15m_close_position": pre15m_close_position,
                "pre30m_ret": pre30m_ret,
                "pre30m_vol_ratio": pre30m_vol_ratio,
                "pre1h_ret": pre1h_ret,
                "pre1h_vol_ratio": pre1h_vol_ratio,
                "pre3h_ret": pre3h_ret,
                "pre3h_vol_ratio": pre3h_vol_ratio,
                "pre3h_range_pct": pre3h_range_pct,
                "breakout_24h_high": breakout_24h_high,
                "pre15m_btc_ret": pre15m_btc_ret,
                "cum_max_ret_to_0830": (cum_high / day_open - 1) if (pd.notna(day_open) and day_open > 0) else np.nan,
                "cum_dd_to_0830": (cum_low / day_open - 1) if (pd.notna(day_open) and day_open > 0) else np.nan,
                "close0830_vs_cum_high": (cls / cum_high - 1) if (pd.notna(cum_high) and cum_high > 0) else np.nan,
                "close0830_vs_day_open": (cls / day_open - 1) if (pd.notna(day_open) and day_open > 0) else np.nan,
                "cum_volume_to_0830": (cum_vol / (vol_24h_mean * 94)) if (
                    pd.notna(vol_24h_mean) and vol_24h_mean > 0
                ) else np.nan,
            })
    df_pre = pd.DataFrame(rows)
    if len(df_pre) == 0:
        return df_pre
    df_pre["breadth_indicator"] = (df_pre["pre15m_ret"] > 0).astype(float)
    breadth = df_pre.groupby("ts")["breadth_indicator"].mean().rename("pre15m_breadth")
    return df_pre.merge(breadth, on="ts", how="left").drop(columns=["breadth_indicator", "ts"])

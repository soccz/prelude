"""Label Space Discovery v1 — stable_move 라벨 sweep (vectorized).

목적 (사용자 Distribution Engine 방향):
  - 단일 ≥20% tail target 강요 X
  - 사용자 actionable 한 라벨을 데이터에서 발견
  - 4 조건 동시 만족: not too sparse, lift 가능, fail 손실 작음, 사용자 대응 가능

stable_move(T, C, DD, H):
  bars_today (4h × ≤6) 로 평가
  open_t = 첫 4h 봉 open, close_t = 마지막 4h 봉 close
  hit:        첫 i 번째 4h 봉의 high >= open*(1+T/100), i*4 <= H
  pass_dd:    bars[0..hit_i] 의 모든 low >= open*(1-DD/100)
  pass_close: close_t >= open*(1+C/100)
  success = hit AND pass_dd AND pass_close

Sweep:
  T  : +3, +5, +7, +10, +15, +20    (6)
  C  :  0, +2, +3, +5                (4)
  DD : 2, 3, 5                       (3) → 의미 -2/-3/-5%
  H  : 4h, 8h, 12h, 24h              (4)
  Total: 288 조합

평가 metric (per combo, per regime_subset):
  base_rate (success rate)
  hit_rate (hit only, regardless of pass_dd/close)
  avg_no_hit_eod (실패 시 EOD), worst5
  avg_post_hit_fail_eod (hit 했는데 dd/close fail)
  avg_hit_time, avg_pre_hit_dd

사용:
    python scripts/label_space_discovery_v1.py
    python scripts/label_space_discovery_v1.py --limit-markets 30  # 개발용
"""
from __future__ import annotations

import argparse
import logging
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.database import list_markets, load_candles
from signals.features import compute_btc_features


SWEEP_T = [3, 5, 7, 10, 15, 20]
SWEEP_C = [0, 2, 3, 5]
SWEEP_DD = [2, 3, 5]
SWEEP_H = [4, 8, 12, 24]
MAX_BARS = 6  # 24h / 4h


def build_arrays(candles_4h: dict[str, pd.DataFrame],
                 btc_regime_map: dict, log) -> dict:
    """모든 (coin, day) 를 padded 2D 배열로 빌드."""
    rows_market = []
    rows_date = []
    rows_open = []
    rows_close = []
    rows_regime = []
    highs_pad = []
    lows_pad = []
    n_bars_arr = []

    for market, df in candles_4h.items():
        if df is None or len(df) < 1:
            continue
        df = df.sort_values("timestamp").copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        # KST 09:00 기준 day boundary: subtract 9h
        df["bar_date"] = (df["timestamp"] - pd.Timedelta(hours=9)).dt.date
        for date, g in df.groupby("bar_date", sort=False):
            g2 = g.sort_values("timestamp")
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

            rows_market.append(market)
            rows_date.append(date)
            rows_open.append(float(opens[0]))
            rows_close.append(float(closes[-1]))
            rows_regime.append(btc_regime_map.get(date, "unknown"))
            highs_pad.append(highs_p)
            lows_pad.append(lows_p)
            n_bars_arr.append(n)

    return {
        "market": np.array(rows_market),
        "date": np.array(rows_date),
        "open": np.array(rows_open, dtype=float),
        "close": np.array(rows_close, dtype=float),
        "regime": np.array(rows_regime),
        "highs": np.vstack(highs_pad),  # (N, 6)
        "lows": np.vstack(lows_pad),    # (N, 6)
        "n_bars": np.array(n_bars_arr),
    }


def evaluate_combo_vectorized(arr: dict, T: float, C: float, DD: float, H: int):
    """전체 panel 에 한 combo 적용 → per-row dict 반환."""
    open_t = arr["open"]
    close_t = arr["close"]
    highs = arr["highs"]
    lows = arr["lows"]
    n_bars = arr["n_bars"]
    N = len(open_t)

    n_bars_in_H = max(1, H // 4)
    n_bars_in_H = min(n_bars_in_H, MAX_BARS)
    target = open_t * (1 + T / 100)
    stop = open_t * (1 - DD / 100)
    eod_ret = close_t / open_t - 1

    # hit_mask: (N, n_bars_in_H) — but consider only bars within n_bars
    hit_mask = (highs[:, :n_bars_in_H] >= target[:, None])
    # bars beyond n_bars are NaN → comparison False
    hit_any = hit_mask.any(axis=1)
    first_hit = np.argmax(hit_mask, axis=1)  # 0 when no hit
    first_hit[~hit_any] = -1

    # cum_min_low up to first_hit: build cum min over [:i+1]
    # safer: for hit rows, compute min(lows[i, :first_hit[i]+1])
    pre_hit_low = np.full(N, np.nan)
    for i in range(n_bars_in_H):
        rows_with_hit_at_i = (first_hit == i)
        if rows_with_hit_at_i.any():
            pre_hit_low[rows_with_hit_at_i] = np.nanmin(
                lows[rows_with_hit_at_i, : i + 1], axis=1
            )
    pre_hit_dd = pre_hit_low / open_t - 1

    pass_dd = pre_hit_low >= stop
    pass_close = close_t >= open_t * (1 + C / 100)
    success = hit_any & pass_dd & pass_close

    return {
        "hit": hit_any,
        "success": success,
        "eod_ret": eod_ret,
        "first_hit_bar": first_hit,
        "pre_hit_dd": pre_hit_dd,
        "pass_dd": pass_dd,
        "pass_close": pass_close,
    }


def aggregate_combo(arr: dict, regime_subset, ev: dict, T, C, DD, H):
    if regime_subset is None:
        mask = np.ones(len(ev["hit"]), dtype=bool)
        rs_label = "all"
    else:
        mask = np.isin(arr["regime"], regime_subset)
        rs_label = regime_subset if isinstance(regime_subset, str) else "_".join(regime_subset)
    n_total = int(mask.sum())
    if n_total == 0:
        return None
    hit_m = ev["hit"][mask]
    succ_m = ev["success"][mask]
    eod_m = ev["eod_ret"][mask]
    fh_m = ev["first_hit_bar"][mask]
    dd_m = ev["pre_hit_dd"][mask]

    no_hit_mask_local = ~hit_m
    post_hit_fail_local = hit_m & (~succ_m)

    n_hit = int(hit_m.sum())
    n_success = int(succ_m.sum())

    return {
        "regime_subset": rs_label,
        "T": T, "C": C, "DD": DD, "H": H,
        "n_total": n_total,
        "n_hit": n_hit,
        "n_success": n_success,
        "base_rate_pct": n_success / n_total * 100,
        "hit_rate_pct": n_hit / n_total * 100,
        "avg_no_hit_eod_pct": float(eod_m[no_hit_mask_local].mean() * 100) if no_hit_mask_local.any() else np.nan,
        "worst5_no_hit_eod_pct": float(np.quantile(eod_m[no_hit_mask_local], 0.05) * 100) if no_hit_mask_local.any() else np.nan,
        "avg_post_hit_fail_eod_pct": float(eod_m[post_hit_fail_local].mean() * 100) if post_hit_fail_local.any() else np.nan,
        "avg_hit_time_h": float((fh_m[succ_m] + 1).mean() * 4) if succ_m.any() else np.nan,
        "avg_pre_hit_dd_pct_when_success": float(dd_m[succ_m].mean() * 100) if succ_m.any() else np.nan,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upbit-4h", default="data/upbit_4h.db")
    parser.add_argument("--upbit-d1", default="data/upbit_d1.db")
    parser.add_argument("--out-csv", default="output/label_discovery_v1.csv")
    parser.add_argument("--limit-markets", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("label-disc")

    print(f"=== Label Space Discovery v1 — stable_move sweep (vectorized) ===\n")
    n_combos = len(SWEEP_T) * len(SWEEP_C) * len(SWEEP_DD) * len(SWEEP_H)
    print(f"Sweep: T{len(SWEEP_T)} × C{len(SWEEP_C)} × DD{len(SWEEP_DD)} × H{len(SWEEP_H)} "
          f"= {n_combos} 조합")
    print(f"Regime subsets: all, bull_all (bull_quiet+bull_volatile)\n")

    log.info("loading BTC daily for regime...")
    btc_d1 = load_candles(args.upbit_d1, "KRW-BTC")
    btc_d1["timestamp"] = pd.to_datetime(btc_d1["timestamp"])
    btc_feat = compute_btc_features(btc_d1)
    btc_feat["date"] = btc_feat["timestamp"].dt.date
    btc_regime_map = dict(zip(btc_feat["date"], btc_feat["btc_regime"]))
    log.info(f"  BTC regime dates: {len(btc_regime_map):,}")

    log.info("loading KRW 4h candles...")
    krw = [m for m in list_markets(args.upbit_4h) if m.startswith("KRW-")]
    if args.limit_markets:
        krw = krw[: args.limit_markets]
    candles_4h = {m: load_candles(args.upbit_4h, m) for m in krw}
    candles_4h = {k: v for k, v in candles_4h.items() if v is not None and len(v) >= 1}
    log.info(f"  KRW 4h markets: {len(candles_4h)}")

    log.info("building padded arrays...")
    arr = build_arrays(candles_4h, btc_regime_map, log)
    N = len(arr["open"])
    log.info(f"  panel rows: {N:,}")
    log.info(f"  regime distribution: {pd.Series(arr['regime']).value_counts().to_dict()}")

    log.info(f"sweeping {n_combos} combos × 2 regime subsets...")
    rows = []
    bull_regimes = ["bull_quiet", "bull_volatile"]
    combos = list(product(SWEEP_T, SWEEP_C, SWEEP_DD, SWEEP_H))
    for i, (T, C, DD, H) in enumerate(combos, 1):
        ev = evaluate_combo_vectorized(arr, T, C, DD, H)
        for rs in (None, bull_regimes):
            agg = aggregate_combo(arr, rs, ev, T, C, DD, H)
            if agg:
                rows.append(agg)
        if i % 50 == 0:
            log.info(f"  {i}/{n_combos}")

    df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    log.info(f"saved {args.out_csv}")

    # ============================================================
    # Top combos under different criteria
    # ============================================================
    show_cols = ["regime_subset", "T", "C", "DD", "H",
                 "base_rate_pct", "hit_rate_pct",
                 "avg_no_hit_eod_pct", "worst5_no_hit_eod_pct",
                 "avg_post_hit_fail_eod_pct",
                 "avg_hit_time_h", "avg_pre_hit_dd_pct_when_success"]

    def show(title, df_sel):
        print(f"\n=== {title} ===")
        if len(df_sel) == 0:
            print("(no rows)")
            return
        print(df_sel[show_cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print(f"\n{'='*100}")
    show("Top 15 by base_rate — bull_all 만, base ≥ 5%, hit_time ≤ 12h",
         df[(df["regime_subset"] == "bull_all_bull_quiet_bull_volatile") |
            (df["regime_subset"] == "bull_quiet_bull_volatile")][lambda x:
            (x["base_rate_pct"] >= 5) & (x["avg_hit_time_h"] <= 12)
         ].sort_values("base_rate_pct", ascending=False).head(15))

    show("Top 15 (smallest |worst5 fail|) — bull_all, base ≥ 3%",
         df[(df["regime_subset"] == "bull_quiet_bull_volatile") &
            (df["base_rate_pct"] >= 3)
         ].sort_values("worst5_no_hit_eod_pct", ascending=False).head(15))

    show("Composite — bull_all, base ≥ 5%, worst5 ≥ -5%, hit_time ≤ 12h",
         df[(df["regime_subset"] == "bull_quiet_bull_volatile") &
            (df["base_rate_pct"] >= 5) &
            (df["worst5_no_hit_eod_pct"] >= -5) &
            (df["avg_hit_time_h"] <= 12)
         ].sort_values("base_rate_pct", ascending=False).head(15))

    print(f"\n전체 결과: {len(df)} rows → {args.out_csv}")


if __name__ == "__main__":
    main()

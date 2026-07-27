"""Label Space Discovery v2 — composite 기준 재조정 + regime lift + liquidity split.

v1 → v2 변경 (사용자 가이드):
  A. Composite 기준 재조정
     - worst5 ≥ -5% 폐기 (이 시장 자연 바닥 -7%)
     - worst5 ≥ -8% 또는 avg_fail_eod 중심
     - first_hit_bar 분포 추가 (즉발 vs 느린형 명시)

  B. Regime lift 분석
     - all / bull_all / bull_quiet / bull_volatile 비교
     - lift = base_in_subset / base_in_all
     - bull_quiet vs bull_volatile 어디 라벨이 더 잘 작동하는지

  C. Liquidity split
     - quote_volume per-day rank → top50 / top100 / top200 / all
     - low-cap 알트가 끌어올린 효과 분리
     - 사용자 실제 진입 가능한 zone 만 actionable

평가 단위:
  combo (T,C,DD,H) × regime_subset × liquidity_tier = 288 × 4 × 4 = 4,608 rows

사용:
    python scripts/label_space_discovery_v2.py
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
from data.market_universe import signal_eligible_markets
from signals.features import compute_btc_features


SWEEP_T = [3, 5, 7, 10, 15, 20]
SWEEP_C = [0, 2, 3, 5]
SWEEP_DD = [2, 3, 5]
SWEEP_H = [4, 8, 12, 24]
MAX_BARS = 6

REGIME_SUBSETS = {
    "all": None,
    "bull_all": ["bull_quiet", "bull_volatile"],
    "bull_quiet": ["bull_quiet"],
    "bull_volatile": ["bull_volatile"],
}

LIQ_TIERS = ["top50", "top100", "top200", "all"]


def build_arrays(candles_4h: dict[str, pd.DataFrame],
                 candles_d1: dict[str, pd.DataFrame],
                 btc_regime_map: dict, log) -> dict:
    """4h panel + daily quote_volume join → padded arrays."""
    rows = []
    for market, df in candles_4h.items():
        if df is None or len(df) < 1:
            continue
        df = df.sort_values("timestamp").copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["bar_date"] = (df["timestamp"] - pd.Timedelta(hours=9)).dt.date

        # daily quote_volume (sum of 4h bars or from d1)
        d1 = candles_d1.get(market)
        d1_qv = {}
        if d1 is not None and "quote_volume" in d1.columns:
            d1 = d1.copy()
            d1["timestamp"] = pd.to_datetime(d1["timestamp"])
            d1["bar_date"] = (d1["timestamp"] - pd.Timedelta(hours=9)).dt.date
            d1_qv = dict(zip(d1["bar_date"], d1["quote_volume"].astype(float)))

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

            # daily quote_volume — d1 value if available, else sum of 4h bars
            if date in d1_qv:
                qv = d1_qv[date]
            elif "quote_volume" in g2.columns:
                qv = float(g2["quote_volume"].sum())
            else:
                qv = np.nan

            rows.append({
                "market": market,
                "date": date,
                "open": float(opens[0]),
                "close": float(closes[-1]),
                "regime": btc_regime_map.get(date, "unknown"),
                "highs": highs_p,
                "lows": lows_p,
                "n_bars": n,
                "quote_volume": qv,
            })
    df_panel = pd.DataFrame(rows)
    log.info(f"  panel rows: {len(df_panel):,}")

    # liquidity tier per date: rank by quote_volume per (date), take top N
    df_panel["liq_rank"] = df_panel.groupby("date")["quote_volume"].rank(
        method="dense", ascending=False, na_option="bottom"
    )
    return df_panel


def evaluate_combo_vectorized(highs: np.ndarray, lows: np.ndarray,
                              opens: np.ndarray, closes: np.ndarray,
                              T, C, DD, H):
    """fast vectorized evaluator. returns dict of arrays."""
    N = len(opens)
    n_bars_in_H = max(1, H // 4)
    n_bars_in_H = min(n_bars_in_H, MAX_BARS)
    target = opens * (1 + T / 100)
    stop = opens * (1 - DD / 100)
    eod_ret = closes / opens - 1

    hit_mask = highs[:, :n_bars_in_H] >= target[:, None]
    hit_any = hit_mask.any(axis=1)
    first_hit = np.argmax(hit_mask, axis=1)
    first_hit[~hit_any] = -1

    pre_hit_low = np.full(N, np.nan)
    for i in range(n_bars_in_H):
        rows_i = (first_hit == i)
        if rows_i.any():
            pre_hit_low[rows_i] = np.nanmin(lows[rows_i, : i + 1], axis=1)
    pre_hit_dd = pre_hit_low / opens - 1

    pass_dd = pre_hit_low >= stop
    pass_close = closes >= opens * (1 + C / 100)
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


def aggregate(panel: pd.DataFrame, ev: dict, mask: np.ndarray, T, C, DD, H,
              regime_label, liq_label, baseline_base_rate=None):
    n_total = int(mask.sum())
    if n_total == 0:
        return None
    hit = ev["hit"][mask]
    succ = ev["success"][mask]
    eod = ev["eod_ret"][mask]
    fh = ev["first_hit_bar"][mask]
    dd = ev["pre_hit_dd"][mask]

    no_hit = ~hit
    post_hit_fail = hit & ~succ
    n_hit = int(hit.sum())
    n_success = int(succ.sum())

    base_rate_pct = n_success / n_total * 100
    lift_vs_all = (base_rate_pct / baseline_base_rate) if baseline_base_rate and baseline_base_rate > 0 else np.nan

    # first_hit_bar distribution among hits
    fh_hits = fh[hit]
    fh_dist = {f"hit_bar_{i}_pct": float((fh_hits == i).sum() / max(len(fh_hits), 1) * 100)
               for i in range(MAX_BARS)}

    return {
        "regime_subset": regime_label,
        "liq_tier": liq_label,
        "T": T, "C": C, "DD": DD, "H": H,
        "n_total": n_total,
        "n_hit": n_hit,
        "n_success": n_success,
        "base_rate_pct": base_rate_pct,
        "hit_rate_pct": n_hit / n_total * 100,
        "lift_vs_all": lift_vs_all,
        "avg_no_hit_eod_pct": float(eod[no_hit].mean() * 100) if no_hit.any() else np.nan,
        "worst5_no_hit_eod_pct": float(np.quantile(eod[no_hit], 0.05) * 100) if no_hit.any() else np.nan,
        "avg_post_hit_fail_eod_pct": float(eod[post_hit_fail].mean() * 100) if post_hit_fail.any() else np.nan,
        "worst5_post_hit_fail_eod_pct": float(np.quantile(eod[post_hit_fail], 0.05) * 100) if post_hit_fail.sum() >= 20 else np.nan,
        "avg_hit_time_h_success": float((fh[succ] + 1).mean() * 4) if succ.any() else np.nan,
        "avg_pre_hit_dd_pct_success": float(dd[succ].mean() * 100) if succ.any() else np.nan,
        **fh_dist,
    }


def liq_mask(panel: pd.DataFrame, liq_label: str) -> np.ndarray:
    if liq_label == "all":
        return np.ones(len(panel), dtype=bool)
    n = int(liq_label.replace("top", ""))
    return (panel["liq_rank"] <= n).values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upbit-4h", default="data/upbit_4h.db")
    parser.add_argument("--upbit-d1", default="data/upbit_d1.db")
    parser.add_argument("--out-csv", default="output/label_discovery_v2.csv")
    parser.add_argument("--limit-markets", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("ldv2")

    print(f"=== Label Space Discovery v2 — composite + regime lift + liquidity ===\n")
    n_combos = len(SWEEP_T) * len(SWEEP_C) * len(SWEEP_DD) * len(SWEEP_H)
    print(f"Sweep: {n_combos} combos × {len(REGIME_SUBSETS)} regime × {len(LIQ_TIERS)} liq "
          f"= {n_combos * len(REGIME_SUBSETS) * len(LIQ_TIERS):,} rows\n")

    log.info("loading BTC daily for regime...")
    btc_d1 = load_candles(args.upbit_d1, "KRW-BTC")
    btc_d1["timestamp"] = pd.to_datetime(btc_d1["timestamp"])
    btc_feat = compute_btc_features(btc_d1)
    btc_feat["date"] = btc_feat["timestamp"].dt.date
    btc_regime_map = dict(zip(btc_feat["date"], btc_feat["btc_regime"]))

    log.info("loading KRW 4h + d1 candles...")
    krw = [
        market
        for market in signal_eligible_markets(list_markets(args.upbit_4h))
        if market.startswith("KRW-")
    ]
    if args.limit_markets:
        krw = krw[: args.limit_markets]
    candles_4h = {m: load_candles(args.upbit_4h, m) for m in krw}
    candles_d1 = {m: load_candles(args.upbit_d1, m) for m in krw}
    log.info(f"  KRW markets: {len(candles_4h)}")

    log.info("building panel + liquidity rank...")
    panel = build_arrays(candles_4h, candles_d1, btc_regime_map, log)
    highs = np.vstack(panel["highs"].values)
    lows = np.vstack(panel["lows"].values)
    opens = panel["open"].values
    closes = panel["close"].values
    regime_arr = panel["regime"].values

    log.info(f"  liquidity rank distribution (sample): {panel['liq_rank'].describe().to_dict()}")
    log.info(f"  regime: {pd.Series(regime_arr).value_counts().to_dict()}")

    # liq masks (precompute)
    liq_masks = {tier: liq_mask(panel, tier) for tier in LIQ_TIERS}
    # regime masks
    regime_masks = {label: (np.ones(len(panel), dtype=bool) if regs is None
                            else np.isin(regime_arr, regs))
                    for label, regs in REGIME_SUBSETS.items()}

    log.info(f"sweeping {n_combos} combos...")
    rows = []
    combos = list(product(SWEEP_T, SWEEP_C, SWEEP_DD, SWEEP_H))
    for i, (T, C, DD, H) in enumerate(combos, 1):
        ev = evaluate_combo_vectorized(highs, lows, opens, closes, T, C, DD, H)
        # baseline = all regime / all liq
        base_all = aggregate(panel, ev,
                              regime_masks["all"] & liq_masks["all"],
                              T, C, DD, H, "all", "all")
        baseline_rate = base_all["base_rate_pct"] if base_all else None

        for regime_label, r_mask in regime_masks.items():
            for liq_label in LIQ_TIERS:
                if regime_label == "all" and liq_label == "all":
                    rows.append(base_all)
                    continue
                m = r_mask & liq_masks[liq_label]
                agg = aggregate(panel, ev, m, T, C, DD, H,
                                 regime_label, liq_label,
                                 baseline_base_rate=baseline_rate)
                if agg:
                    rows.append(agg)
        if i % 50 == 0:
            log.info(f"  {i}/{n_combos}")

    df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    log.info(f"saved {args.out_csv}")

    # ========================================================================
    # 분석 출력
    # ========================================================================
    show_cols = ["regime_subset", "liq_tier", "T", "C", "DD", "H",
                 "base_rate_pct", "hit_rate_pct", "lift_vs_all",
                 "avg_no_hit_eod_pct", "worst5_no_hit_eod_pct",
                 "avg_post_hit_fail_eod_pct",
                 "avg_hit_time_h_success", "avg_pre_hit_dd_pct_success"]

    def show(title, df_sel):
        print(f"\n=== {title} ===")
        if len(df_sel) == 0:
            print("(no rows)")
            return
        print(df_sel[show_cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # A. composite 재조정 (worst5 >= -8%, hit_time<=12h)
    print(f"\n{'='*120}")
    show("A1. Composite — bull_all × top100 × base ≥ 5% × worst5 ≥ -8% × hit_time ≤ 12h",
         df[(df["regime_subset"] == "bull_all") &
            (df["liq_tier"] == "top100") &
            (df["base_rate_pct"] >= 5) &
            (df["worst5_no_hit_eod_pct"] >= -8) &
            (df["avg_hit_time_h_success"] <= 12)
         ].sort_values("base_rate_pct", ascending=False).head(15))

    show("A2. Composite — bull_all × top100 × base ≥ 5% × hit_time ≤ 12h (worst5 cut 폐기)",
         df[(df["regime_subset"] == "bull_all") &
            (df["liq_tier"] == "top100") &
            (df["base_rate_pct"] >= 5) &
            (df["avg_hit_time_h_success"] <= 12)
         ].sort_values("avg_post_hit_fail_eod_pct", ascending=False).head(15))

    # B. regime lift — bull_all vs all (top100 만)
    print(f"\n{'='*120}")
    show("B1. Top 15 lift (bull_all vs all) at top100, base ≥ 5%",
         df[(df["regime_subset"] == "bull_all") &
            (df["liq_tier"] == "top100") &
            (df["base_rate_pct"] >= 5)
         ].sort_values("lift_vs_all", ascending=False).head(15))

    show("B2. bull_quiet 만 — top100 base ≥ 5%",
         df[(df["regime_subset"] == "bull_quiet") &
            (df["liq_tier"] == "top100") &
            (df["base_rate_pct"] >= 5)
         ].sort_values("base_rate_pct", ascending=False).head(10))

    show("B3. bull_volatile 만 — top100 base ≥ 5%",
         df[(df["regime_subset"] == "bull_volatile") &
            (df["liq_tier"] == "top100") &
            (df["base_rate_pct"] >= 5)
         ].sort_values("base_rate_pct", ascending=False).head(10))

    # C. liquidity split — 같은 (T=3,C=2,DD=3,H=24) 가 top50/100/200/all 에서 어떻게 변하는지
    print(f"\n{'='*120}")
    print("C. Liquidity split (사용자 actionable zone 검증)\n")
    sample_combos = [(3, 0, 5, 24), (3, 2, 3, 24), (3, 3, 3, 24),
                     (5, 0, 5, 24), (5, 3, 3, 24),
                     (3, 0, 5, 4), (3, 2, 2, 4),
                     (10, 0, 5, 24), (20, 0, 5, 24)]
    for T, C, DD, H in sample_combos:
        sub = df[(df["regime_subset"] == "bull_all") &
                 (df["T"] == T) & (df["C"] == C) & (df["DD"] == DD) & (df["H"] == H)]
        if len(sub) > 0:
            show(f"  combo (T={T},C={C},DD={DD},H={H}) — bull_all × all liq tier",
                 sub.sort_values("liq_tier"))

    # D. detector_v1 위치 (T=20)
    print(f"\n{'='*120}")
    show("D. detector_v1 영역 (T=20, bull_all)",
         df[(df["regime_subset"] == "bull_all") &
            (df["T"] == 20)].sort_values(["liq_tier", "C", "DD", "H"]).head(15))

    # E. hit_time 분포 (사용자 즉발 vs 느린형 분리 위해)
    print(f"\n{'='*120}")
    print("E. first_hit_bar 분포 (bull_all × top100 × 대표 combo)\n")
    hb_cols = ["regime_subset", "liq_tier", "T", "C", "DD", "H",
               "hit_rate_pct"] + [f"hit_bar_{i}_pct" for i in range(MAX_BARS)]
    sample = df[(df["regime_subset"] == "bull_all") &
                (df["liq_tier"] == "top100") &
                (df["C"] == 0) & (df["DD"] == 5) &
                (df["H"] == 24)]
    if len(sample) > 0:
        print(sample[hb_cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print(f"\n전체 결과: {len(df):,} rows → {args.out_csv}")


if __name__ == "__main__":
    main()

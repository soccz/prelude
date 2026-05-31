"""Ignition-following backtest (angle-3: cold-start intraday micro-dynamics).

Hypothesis
----------
Daily precursors miss ~half of +20% pumps (cold-start first spikes). But the
+20% high of a pump day is hit at median bar ~32 (~8h) — i.e. there is intraday
runway AFTER an early ignition. If we detect an "ignition" from the first K
completed 15m bars after the KST 09:00 open (volume explosion + price surge),
and ENTER on the NEXT bar's open (so the signal uses only completed-bar info),
we can *follow* the move rather than predict it.

This is fast-following, not prediction. We do NOT claim to predict +20% at 09:00.

Leak guards (this project has 2 same-day-leak incidents)
--------------------------------------------------------
- Signal uses bars [0 .. K-1] only (completed). Entry = open of bar K.
- Exit path uses bars [K .. end] highs/lows for TP/SL/time-stop simulation,
  but the entry DECISION never sees bar K+ data.
- Day boundary = KST 09:00 (verified: d1 open == 15m 09:00 open exactly).
- We drop any (market,date) whose first bar is not exactly 09:00 (incomplete
  left edge -> first-bar timing untrustworthy).
- Warm/cold tag uses D-1 daily features (shift(1) per market) only.
- 0.15% round-trip cost charged on every realized trade.
- Walk-forward by calendar time; thresholds picked on TRAIN only, applied OOS.

Usage:
    python scripts/ignition_following_v1.py --K 4 --tp 0.08 --sl 0.04
"""
from __future__ import annotations
import argparse, sqlite3, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D1 = ROOT / "data" / "upbit_d1.db"
M15 = ROOT / "data" / "upbit_15m.db"
OUT = ROOT / "output"
COST = 0.0015  # round-trip


def load_daily_with_precursors() -> pd.DataFrame:
    c = sqlite3.connect(D1)
    d = pd.read_sql("SELECT market,timestamp,open,high,low,close,quote_volume FROM candles",
                    c, parse_dates=["timestamp"])
    d = d.sort_values(["market", "timestamp"]).reset_index(drop=True)
    g = d.groupby("market", sort=False)
    # D-1 daily precursors (shift(1) per market => strictly past)
    d["ret_1d_prev"] = g["close"].pct_change().groupby(d["market"]).shift(1)
    d["qv_ma30_prev"] = g["quote_volume"].transform(
        lambda s: s.rolling(30, min_periods=10).mean().shift(1))
    d["qv_prev"] = g["quote_volume"].shift(1)
    d["qv_surge_30d"] = d["qv_prev"] / d["qv_ma30_prev"]
    d["close_prev"] = g["close"].shift(1)
    d["close_prev8"] = g["close"].shift(8)
    d["ret_7d_prev"] = d["close_prev"] / d["close_prev8"] - 1.0
    # warm = any strong daily precursor at D-1
    d["warm"] = (((d["qv_surge_30d"] >= 2.0).fillna(False)) |
                 ((d["ret_7d_prev"] >= 0.15).fillna(False)) |
                 ((d["ret_1d_prev"] >= 0.05).fillna(False))).astype(int)
    d["date"] = d["timestamp"].dt.strftime("%Y-%m-%d")
    # daily liquidity rank (D-1) so we can study cap tiers without leak
    d["qv_rank_prev"] = d.groupby("date")["qv_prev"].rank(pct=True)
    return d[["market", "date", "warm", "qv_surge_30d", "ret_7d_prev",
              "ret_1d_prev", "qv_rank_prev"]]


def build_ignition_panel(K: int, daily: pd.DataFrame) -> pd.DataFrame:
    """One row per clean (market, KST-day). Signal features from bars[0..K-1].
    Exit-path stats from bars[K..end]. No entry-decision uses bar>=K."""
    m = sqlite3.connect(M15)
    df = pd.read_sql("SELECT market,timestamp,open,high,low,close,volume,quote_volume FROM candles",
                     m, parse_dates=["timestamp"])
    df = df.sort_values(["market", "timestamp"]).reset_index(drop=True)
    df["date"] = (df["timestamp"] - pd.Timedelta(hours=9)).dt.strftime("%Y-%m-%d")
    rows = []
    for (mkt, date), g in df.groupby(["market", "date"], sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        first = g.iloc[0]["timestamp"]
        if first.hour != 9 or first.minute != 0:
            continue
        if len(g) < K + 8:  # need K signal bars + some runway
            continue
        o0 = float(g.iloc[0]["open"])
        if o0 <= 0:
            continue
        sig = g.iloc[:K]
        # ----- signal features (completed bars only) -----
        sig_high = float(sig["high"].max())
        sig_close = float(sig.iloc[-1]["close"])
        sig_low = float(sig["low"].min())
        early_max_ret = sig_high / o0 - 1.0          # max surge in window
        early_close_ret = sig_close / o0 - 1.0       # close-of-window vs open
        sig_vol = float(sig["volume"].sum())
        sig_qv = float(sig["quote_volume"].sum())
        # green-bar fraction & last-bar momentum
        green_frac = float((sig["close"].to_numpy() > sig["open"].to_numpy()).mean())
        last_bar_ret = float(sig.iloc[-1]["close"] / sig.iloc[-1]["open"] - 1.0)
        close_pos = (sig_close - sig_low) / max(sig_high - sig_low, 1e-9)
        # ----- entry & exit path (bars[K..]) -----
        entry = float(g.iloc[K]["open"])
        if entry <= 0:
            continue
        post = g.iloc[K:]
        post_high = post["high"].to_numpy(float)
        post_low = post["low"].to_numpy(float)
        post_close = float(post.iloc[-1]["close"])
        rows.append({
            "market": mkt, "date": date, "open0": o0, "entry": entry,
            "early_max_ret": early_max_ret, "early_close_ret": early_close_ret,
            "green_frac": green_frac, "last_bar_ret": last_bar_ret,
            "close_pos": close_pos, "sig_vol": sig_vol, "sig_qv": sig_qv,
            # store post path compactly for vectorized TP/SL later
            "post_high_arr": post_high, "post_low_arr": post_low,
            "post_close": post_close,
            "entry_to_dayhigh": float(post_high.max()) / entry - 1.0,
            "entry_to_dayclose": post_close / entry - 1.0,
        })
    panel = pd.DataFrame(rows)
    # 24h-volume baseline per (market): use sig_vol vs trailing median of sig_vol
    panel = panel.sort_values(["market", "date"]).reset_index(drop=True)
    panel["sig_vol_med30"] = panel.groupby("market")["sig_vol"].transform(
        lambda s: s.rolling(30, min_periods=10).median().shift(1))
    panel["vol_ratio"] = panel["sig_vol"] / panel["sig_vol_med30"]
    panel = panel.merge(daily, on=["market", "date"], how="left")
    return panel


def simulate_trade(post_high, post_low, entry, tp, sl):
    """Path-aware exit. Conservative: within a bar, if both TP and SL touched,
    assume SL first (pessimistic). Returns gross return fraction."""
    tp_px = entry * (1 + tp)
    sl_px = entry * (1 - sl)
    for h, l in zip(post_high, post_low):
        hit_sl = l <= sl_px
        hit_tp = h >= tp_px
        if hit_sl:
            return -sl
        if hit_tp:
            return tp
    # neither: exit at day close
    return post_low.size and None  # placeholder; handled by caller via post_close


def run_backtest(panel, K, tp, sl, entry_rules, n_folds=3, embargo_days=2):
    """Walk-forward. entry_rules: dict of threshold params chosen on TRAIN only."""
    panel = panel.dropna(subset=["vol_ratio"]).copy()
    panel["dt"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values("dt").reset_index(drop=True)
    dates = panel["dt"].sort_values().unique()
    fold_edges = np.linspace(0, len(dates), n_folds + 2, dtype=int)

    results = []
    fold_summ = []
    for f in range(n_folds):
        tr_end = dates[fold_edges[f + 1] - 1]
        te_start = dates[min(fold_edges[f + 1] + embargo_days, len(dates) - 1)]
        te_end = dates[fold_edges[f + 2] - 1]
        tr = panel[panel["dt"] <= tr_end]
        te = panel[(panel["dt"] >= te_start) & (panel["dt"] <= te_end)]
        if len(tr) < 200 or len(te) < 50:
            continue
        # ----- pick thresholds on TRAIN ONLY -----
        # ignition = early surge + volume explosion + bullish close
        em_thr = tr["early_max_ret"].quantile(entry_rules["em_q"])
        vr_thr = tr["vol_ratio"].quantile(entry_rules["vr_q"])
        em_thr = max(em_thr, entry_rules["em_floor"])  # floor: require real surge

        for split_name, sub in [("test", te)]:
            fire = sub[(sub["early_max_ret"] >= em_thr) &
                       (sub["vol_ratio"] >= vr_thr) &
                       (sub["green_frac"] >= entry_rules["green_min"]) &
                       (sub["close_pos"] >= entry_rules["close_pos_min"])].copy()
            for _, r in fire.iterrows():
                g = simulate_trade(r["post_high_arr"], r["post_low_arr"], r["entry"], tp, sl)
                if g is None:
                    g = r["entry_to_dayclose"]  # time-stop at day close
                net = g - COST
                results.append({"fold": f, "market": r["market"], "date": r["date"],
                                "warm": r["warm"], "qv_rank_prev": r["qv_rank_prev"],
                                "gross": g, "net": net,
                                "early_max_ret": r["early_max_ret"], "vol_ratio": r["vol_ratio"]})
            n_fire = len(fire)
            fold_summ.append({"fold": f, "tr_n": len(tr), "te_n": len(te),
                              "em_thr": em_thr, "vr_thr": vr_thr,
                              "n_fire": n_fire,
                              "fire_rate": n_fire / max(len(te), 1)})
    res = pd.DataFrame(results)
    fs = pd.DataFrame(fold_summ)
    return res, fs


def summarize(res, label=""):
    if len(res) == 0:
        print(f"  [{label}] no trades fired.")
        return {}
    n = len(res)
    net = res["net"]
    win = (net > 0).mean()
    avg = net.mean()
    tot = net.sum()
    # daily-aggregated sharpe (group by date, equal-weight that day's fires)
    daily = res.groupby("date")["net"].mean()
    sharpe = (daily.mean() / daily.std() * np.sqrt(365)) if daily.std() > 0 else 0.0
    summ = {"n": n, "win_rate": round(win, 3), "avg_net": round(avg, 4),
            "total_net": round(tot, 3), "sharpe_daily": round(sharpe, 2),
            "n_pump20_caught": int((res["gross"] >= 0.20).sum())}
    print(f"  [{label}] n={n} win={win:.1%} avg_net={avg:+.3%} "
          f"total_net={tot:+.2f} sharpe(daily)={sharpe:+.2f}")
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=4, help="signal window in 15m bars (4=1h)")
    ap.add_argument("--tp", type=float, default=0.08)
    ap.add_argument("--sl", type=float, default=0.04)
    ap.add_argument("--em-q", type=float, default=0.97, help="early_max_ret train quantile")
    ap.add_argument("--em-floor", type=float, default=0.05, help="min surge to call ignition")
    ap.add_argument("--vr-q", type=float, default=0.90, help="vol_ratio train quantile")
    ap.add_argument("--green-min", type=float, default=0.5)
    ap.add_argument("--close-pos-min", type=float, default=0.5)
    ap.add_argument("--n-folds", type=int, default=3)
    ap.add_argument("--cache", default=str(ROOT / "output" / "_ignition_panel_cache.pkl"))
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    cache = Path(args.cache.replace(".pkl", f"_K{args.K}.pkl"))
    if cache.exists() and not args.refresh:
        print(f"[load] {cache}")
        panel = pd.read_pickle(cache)
    else:
        print("[build] daily precursors...")
        daily = load_daily_with_precursors()
        print("[build] ignition panel from 15m (this takes a minute)...")
        panel = build_ignition_panel(args.K, daily)
        panel.to_pickle(cache)
        print(f"[cache] {cache}  rows={len(panel)}")

    print(f"\npanel rows (clean coin-days): {len(panel)}")
    print(f"date range: {panel['date'].min()} .. {panel['date'].max()}")

    entry_rules = {"em_q": args.em_q, "em_floor": args.em_floor, "vr_q": args.vr_q,
                   "green_min": args.green_min, "close_pos_min": args.close_pos_min}
    res, fs = run_backtest(panel, args.K, args.tp, args.sl, entry_rules, n_folds=args.n_folds)
    print("\n=== fold summary ===")
    if len(fs):
        print(fs.to_string(index=False))
    print(f"\n=== OOS net results (K={args.K} tp={args.tp} sl={args.sl}) ===")
    overall = summarize(res, "ALL")
    if len(res):
        summarize(res[res["warm"] == 1], "WARM")
        summarize(res[res["warm"] == 0], "COLD")
        summarize(res[res["qv_rank_prev"] >= 0.8], "top20%-liq")
        summarize(res[res["qv_rank_prev"] < 0.5], "bot50%-liq")
        tag = f"K{args.K}_tp{args.tp}_sl{args.sl}"
        res.to_csv(OUT / f"ignition_following_v1_{tag}_trades.csv", index=False)
        print(f"\n[saved] output/ignition_following_v1_{tag}_trades.csv")


if __name__ == "__main__":
    main()

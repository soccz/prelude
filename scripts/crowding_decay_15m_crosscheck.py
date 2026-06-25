#!/usr/bin/env python
"""crowding_decay 15m TP5/SL3 cross-check (evaluator 적대 감사).

honest 지표는 EOD open->close 인데 운영 청산은 TP5/SL3 bracket 이다.
EOD 효과가 intraday 덤프(-3% SL 이 잡는) 때문이면 운영에선 약해진다.
같은 backfill 픽(date, coin)으로 15m 경로를 로드해 walk_path(sl=0.03, tp=0.05) 로
재계산하고, HIGH vs LOW crowd net 차를 day-clustered bootstrap 으로 다시 본다.

핵심 비교: EOD-from-15m vs backfill EOD (sanity), 그리고 bracket net 차이가
EOD net 차이만큼 살아남는가.
"""
from __future__ import annotations
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

BACKFILL = Path("output/paper_ledger_backfill.csv")
M15_DB = "data/upbit_15m.db"
COST = 0.0015
DEEP = -0.05
SEED = 42
N_BOOT = 2000
CROWD_COLS = ["return_7d_rank", "roc_3d_rank", "return_5d_rank"]
OPERATING_REGIMES = ["bull_quiet", "bull_volatile"]


def _minmax(s):
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if hi > lo else s * 0.0


def walk_path(bars, sl, tp):
    """시간순 walk. 진입=첫봉 open. 같은봉 SL·TP 동시 -> SL 먼저(보수). 없으면 EOD."""
    if not bars:
        return np.nan, "nodata"
    entry = bars[0][0]
    if not np.isfinite(entry) or entry <= 0:
        return np.nan, "nodata"
    sl_px = entry * (1 - sl) if sl is not None else None
    tp_px = entry * (1 + tp) if tp is not None else None
    for (o, h, l, c) in bars:
        if sl_px is not None and l <= sl_px:
            return -sl, "sl"
        if tp_px is not None and h >= tp_px:
            return tp, "tp"
    return bars[-1][3] / entry - 1.0, "eod"


def load_paths(pairs):
    conn = sqlite3.connect(M15_DB)
    paths = {}
    for m, d in pairs:
        dt = pd.Timestamp(d)
        start = dt.strftime("%Y-%m-%d 09:00:00")
        end = (dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d 09:00:00")
        rows = conn.execute(
            "SELECT open,high,low,close FROM candles WHERE market=? AND "
            "timestamp>=? AND timestamp<? ORDER BY timestamp", (m, start, end)
        ).fetchall()
        if rows:
            paths[(m, d)] = rows
    conn.close()
    return paths


def day_cluster_boot(d, col_q, value_col, hi="high", lo="low"):
    rng = np.random.default_rng(SEED)
    sub = d[d[col_q].isin([hi, lo])].copy()
    is_hi = (sub[col_q] == hi).astype(float)
    is_lo = (sub[col_q] == lo).astype(float)
    deep = (sub[value_col] < DEEP).astype(float)
    sub = sub.assign(is_hi=is_hi, is_lo=is_lo,
                     v_hi=sub[value_col] * is_hi, v_lo=sub[value_col] * is_lo,
                     deep_hi=deep * is_hi, deep_lo=deep * is_lo)
    agg = sub.groupby("date")[["is_hi", "is_lo", "v_hi", "v_lo", "deep_hi", "deep_lo"]].sum()
    nd = len(agg)
    if nd < 8:
        return dict(net_diff=np.nan, net_ci=[np.nan, np.nan], deep_diff=np.nan, deep_ci=[np.nan, np.nan])
    n_hi, n_lo = agg["is_hi"].values, agg["is_lo"].values
    s_hi, s_lo = agg["v_hi"].values, agg["v_lo"].values
    d_hi, d_lo = agg["deep_hi"].values, agg["deep_lo"].values
    idx = rng.integers(0, nd, size=(N_BOOT, nd))

    def ratio(num, den):
        N = num[idx].sum(axis=1)
        D = den[idx].sum(axis=1)
        return np.where(D > 0, N / np.where(D > 0, D, 1.0), np.nan)

    net_diffs = ratio(s_hi, n_hi) - ratio(s_lo, n_lo)
    deep_diffs = ratio(d_hi, n_hi) - ratio(d_lo, n_lo)

    def ci(a):
        a = a[~np.isnan(a)]
        return float(a.mean()), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

    nm, nlo, nhi = ci(net_diffs)
    dm, dlo, dhi = ci(deep_diffs)
    return dict(net_diff=nm, net_ci=[nlo, nhi], deep_diff=dm, deep_ci=[dlo, dhi])


def perm_within_date(df, value_col, n_perm=2000):
    np.random.seed(SEED)
    net = df[value_col].values
    q = df["crowd_q"].astype(str).values
    obs = net[q == "high"].mean() - net[q == "low"].mean()
    grp = df.groupby("date").indices
    cnt = 0
    for _ in range(n_perm):
        pq = q.copy()
        for _, pos in grp.items():
            pq[pos] = np.random.permutation(pq[pos])
        d = net[pq == "high"].mean() - net[pq == "low"].mean()
        if d <= obs:
            cnt += 1
    return float(obs), float(cnt / n_perm)


def main():
    df = pd.read_csv(BACKFILL)
    df["net_eod"] = df["next_close_return_pct"] / 100.0 - COST
    df = df.dropna(subset=["net_eod"]).reset_index(drop=True)
    avail = [c for c in CROWD_COLS if c in df.columns]
    norm = pd.DataFrame({c: _minmax(df[c]) for c in avail})
    df["crowd_index"] = norm.mean(axis=1)
    df = df.dropna(subset=["crowd_index"]).reset_index(drop=True)
    df["crowd_q"] = pd.qcut(df["crowd_index"], 3, labels=["low", "mid", "high"], duplicates="drop")

    # load 15m paths
    pairs = list(zip(df["coin"], df["date"]))
    print(f"loading 15m paths for {len(pairs)} picks...")
    paths = load_paths(pairs)
    print(f"  paths found: {len(paths)} ({100*len(paths)/len(pairs):.1f}% coverage)")

    # bracket + eod-from-15m
    out = {"tp5_sl3": (0.05, 0.03), "eod15": (None, None), "tp5_sl5": (0.05, 0.05),
           "tp10_sl3": (0.10, 0.03), "noSL_eod": (None, None)}
    recs = []
    for _, r in df.iterrows():
        key = (r["coin"], r["date"])
        bars = paths.get(key)
        rec = {"date": r["date"], "coin": r["coin"], "crowd_q": r["crowd_q"],
               "btc_regime": r["btc_regime"], "net_eod_d1": r["net_eod"], "has_path": bars is not None}
        if bars:
            for name, (tp, sl) in out.items():
                g, oc = walk_path(bars, sl, tp)
                rec[f"net_{name}"] = g - COST if np.isfinite(g) else np.nan
                rec[f"oc_{name}"] = oc
        recs.append(rec)
    res = pd.DataFrame(recs)
    cov = res[res["has_path"]].copy()
    print(f"\ncovered subset n={len(cov)} dates={cov['date'].nunique()}")

    # sanity: eod-from-15m vs backfill eod
    valid = cov.dropna(subset=["net_eod15"])
    corr = valid["net_eod15"].corr(valid["net_eod_d1"])
    mae = (valid["net_eod15"] - valid["net_eod_d1"]).abs().mean()
    print(f"SANITY eod15 vs backfill eod: corr={corr:.4f} MAE={mae:.4f} "
          f"(mean eod15={valid['net_eod15'].mean():+.4f} vs d1={valid['net_eod_d1'].mean():+.4f})")

    print("\n===== covered subset: HIGH-LOW net차 by exit policy =====")
    print(f"{'policy':<12} {'LOW net':>9} {'HIGH net':>9} {'net차':>9} {'CI95':>22} "
          f"{'deep차':>8} {'perm_p':>7}")
    for name in ["net_eod_d1", "net_eod15", "net_tp5_sl3", "net_tp5_sl5", "net_tp10_sl3", "net_noSL_eod"]:
        sub = cov.dropna(subset=[name])
        lo_m = sub[sub.crowd_q == "low"][name].mean()
        hi_m = sub[sub.crowd_q == "high"][name].mean()
        b = day_cluster_boot(sub, "crowd_q", name)
        _, p = perm_within_date(sub, name, n_perm=2000)
        star = " *0제외" if (not np.isnan(b["net_ci"][1]) and b["net_ci"][1] < 0) else ""
        print(f"{name:<12} {lo_m:>+9.4f} {hi_m:>+9.4f} {b['net_diff']:>+9.4f} "
              f"[{b['net_ci'][0]:+.4f},{b['net_ci'][1]:+.4f}]{star:>8} "
              f"{b['deep_diff']:>+8.4f} {p:>7.4f}")

    print("\n===== bracket TP5/SL3: bucket net + deep + outcome mix =====")
    for q in ["low", "mid", "high"]:
        sub = cov[cov.crowd_q == q].dropna(subset=["net_tp5_sl3"])
        ocmix = sub["oc_tp5_sl3"].value_counts(normalize=True).to_dict()
        print(f"  {q:<4} n={len(sub):<5} net={sub['net_tp5_sl3'].mean():+.4f} "
              f"hit={(sub['net_tp5_sl3']>0).mean():.3f} deep={(sub['net_tp5_sl3']<DEEP).mean():.3f} "
              f"| SL={ocmix.get('sl',0):.2f} TP={ocmix.get('tp',0):.2f} EOD={ocmix.get('eod',0):.2f}")

    print("\n===== 발사regime(bull_quiet+bull_volatile) TP5/SL3 net차 =====")
    op = cov[cov.btc_regime.isin(OPERATING_REGIMES)]
    for name in ["net_eod_d1", "net_tp5_sl3"]:
        sub = op.dropna(subset=[name])
        b = day_cluster_boot(sub, "crowd_q", name)
        lo_m = sub[sub.crowd_q == "low"][name].mean()
        hi_m = sub[sub.crowd_q == "high"][name].mean()
        star = " *0제외" if (not np.isnan(b["net_ci"][1]) and b["net_ci"][1] < 0) else ""
        print(f"  {name:<12} LOW={lo_m:+.4f} HIGH={hi_m:+.4f} net차={b['net_diff']:+.4f} "
              f"CI[{b['net_ci'][0]:+.4f},{b['net_ci'][1]:+.4f}]{star}")

    print("\n===== DOWNRANK under TP5/SL3 (발사regime) =====")
    op2 = op.dropna(subset=["net_tp5_sl3"])
    allm = op2["net_tp5_sl3"].mean()
    keep = op2[op2.crowd_q != "high"]["net_tp5_sl3"].mean()
    dropped = op2[op2.crowd_q == "high"]
    print(f"  전체 n={len(op2)} net={allm:+.4f} deep={(op2['net_tp5_sl3']<DEEP).mean():.3f}")
    print(f"  high강등후 n={len(op2[op2.crowd_q!='high'])} net={keep:+.4f} "
          f"deep={(op2[op2.crowd_q!='high']['net_tp5_sl3']<DEEP).mean():.3f} "
          f"| 강등 {len(dropped)}픽 net={dropped['net_tp5_sl3'].mean():+.4f} "
          f"deep={(dropped['net_tp5_sl3']<DEEP).mean():.3f}")

    res.to_csv("output/crowding_decay_15m_crosscheck.csv", index=False)
    print("\nwrote output/crowding_decay_15m_crosscheck.csv")


if __name__ == "__main__":
    main()

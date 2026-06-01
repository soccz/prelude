"""quant-evaluator independent verification of track B2 exit challenger.

Re-aggregates from output/exit_challenger_trades_v1.csv (does NOT trust researcher table),
runs day-block paired bootstrap CI95 on Δnet and Δdeep-loss vs baseline, and an
independent prefix-replay causal-leak check on simulate_exit.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from signals.exit_rules import simulate_exit, rule_params, simulate_exit_by_rule  # noqa

TRADES = "output/exit_challenger_trades_v1.csv"
RULES = ["baseline", "trail", "time", "vol", "vol_tight", "regime"]


def daily_series(t, rule):
    d = t[t["rule"] == rule]
    g = d.groupby("date")
    return pd.DataFrame({
        "net": g["net"].mean(),
        "lowm5": g["low_excursion"].apply(lambda s: (s.values <= -0.05).mean()),
        "lowm10": g["low_excursion"].apply(lambda s: (s.values <= -0.10).mean()),
        "netm5": g["net"].apply(lambda s: (s.values <= -0.05).mean()),
    })


def ci(arr):
    a = np.array(arr)
    return np.nanmean(a), np.nanpercentile(a, 2.5), np.nanpercentile(a, 97.5)


def main():
    t = pd.read_csv(TRADES)
    base = daily_series(t, "baseline")
    days = base.index.values
    ndays = len(days)
    print(f"n days={ndays}  n picks/rule={len(t[t['rule']=='baseline'])}")

    rng = np.random.RandomState(42)
    B = 5000

    print("\n=== Δ vs baseline (paired day-block bootstrap, B=%d) ===" % B)
    print(f"{'rule':10s} | {'Δnet daily mean [CI95]':38s} | {'Δlow<=-5% rate [CI95]':36s} | Δlow<=-10%")
    print("-" * 130)
    for r in RULES[1:]:
        other = daily_series(t, r).reindex(base.index)
        d_net = (other["net"] - base["net"]).values
        d_l5 = (other["lowm5"] - base["lowm5"]).values
        d_l10 = (other["lowm10"] - base["lowm10"]).values
        bn, bl5, bl10 = [], [], []
        for _ in range(B):
            idx = rng.randint(0, ndays, ndays)
            bn.append(np.nanmean(d_net[idx]))
            bl5.append(np.nanmean(d_l5[idx]))
            bl10.append(np.nanmean(d_l10[idx]))
        (mn, ln, un) = ci(bn)
        (ml, ll, ul) = ci(bl5)
        (m10, l10, u10) = ci(bl10)
        sn = "SIG+" if ln > 0 else ("SIG-" if un < 0 else "n.s.")
        sl = "WORSE*" if ll > 0 else ("better*" if ul < 0 else "n.s.")
        print(f"{r:10s} | {mn:+.6f} [{ln:+.6f},{un:+.6f}] {sn:4s} | "
              f"{ml:+.5f} [{ll:+.5f},{ul:+.5f}] {sl:7s} | {m10:+.5f}[{l10:+.5f},{u10:+.5f}]")

    print("\n=== Absolute daily-mean net per rule, day-block bootstrap CI95 ===")
    for r in RULES:
        s = daily_series(t, r)["net"].values
        bs = [np.mean(s[rng.randint(0, ndays, ndays)]) for _ in range(B)]
        lo, hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
        print(f"{r:10s} net={np.mean(s):+.6f} CI95=[{lo:+.6f},{hi:+.6f}] "
              f"-> {'POS' if lo > 0 else 'NOT-POS'}")

    # ---- bear_quiet subset (researcher claims vol net-positive there) ----
    print("\n=== bear_quiet regime subset (researcher claim: vol net-positive) ===")
    for r in ["baseline", "vol"]:
        d = t[(t["rule"] == r) & (t["regime"] == "bear_quiet")]
        if len(d) == 0:
            print(f"{r}: no bear_quiet rows"); continue
        dl = d.groupby("date")["net"].mean()
        nd = len(dl)
        bs = [np.mean(dl.values[rng.randint(0, nd, nd)]) for _ in range(B)]
        eq = (1 + dl.sort_index()).cumprod()
        print(f"{r:10s} bear_quiet n={len(d)} days={nd} net={d['net'].mean():+.6f} "
              f"cum={eq.iloc[-1]-1:+.4f} CI95=[{np.percentile(bs,2.5):+.6f},{np.percentile(bs,97.5):+.6f}] "
              f"low<=-5%={(d['low_excursion'].values<=-0.05).mean():.4f}")

    # ---- independent prefix-replay causal check ----
    print("\n=== INDEPENDENT prefix-replay causal-leak check ===")
    causal_check(t)


def causal_check(t):
    """Synthetic adversarial paths: confirm no future-bar can change an exit decision.
    Build paths with a known future spike/dip AFTER the exit trigger; the exit must
    not move. Also test same-bar SL+TP -> SL first (conservative)."""
    rng = np.random.RandomState(7)
    fails = 0
    tested = 0
    for _ in range(400):
        entry = 100.0
        n = rng.randint(8, 96)  # up to a day of 15m bars
        bars = []
        px = entry
        for i in range(n):
            o = px
            # random walk OHLC
            hi = o * (1 + abs(rng.normal(0, 0.01)))
            lo = o * (1 - abs(rng.normal(0, 0.01)))
            c = o * (1 + rng.normal(0, 0.008))
            hi = max(hi, o, c); lo = min(lo, o, c)
            bars.append((o, hi, lo, c))
            px = c
        atr = abs(rng.normal(0.025, 0.01))
        for rule in RULES:
            kw = rule_params(rule, regime=rng.choice(["bull_quiet", "bear_volatile", "bear_quiet"]), atr_pct=atr)
            g_full, oc_full, hold_full, _ = simulate_exit(bars, **kw)
            # prefix replay: grow bars one at a time, take FIRST stopping decision
            g_pref, oc_pref = None, None
            for L in range(1, len(bars) + 1):
                gg, occ, _, _ = simulate_exit(bars[:L], **kw)
                if occ in ("sl", "trail", "tp", "timestop"):
                    g_pref, oc_pref = gg, occ
                    break
            if g_pref is None:
                g_pref, oc_pref, _, _ = simulate_exit(bars, **kw)
            tested += 1
            if oc_pref != oc_full or not np.isclose(g_pref, g_full, atol=1e-9):
                fails += 1
    print(f"  random-path prefix-replay: {fails} mismatch / {tested} (rule x path); must be 0")

    # same-bar SL+TP precedence: a single bar that hits BOTH -3% low and +5% high
    bar = [(100.0, 106.0, 96.0, 101.0)]  # low 96 (-4%) <= SL(-3%), high 106 (+6%) >= TP(+5%)
    g, oc, _, _ = simulate_exit(bar, hard_sl=0.03, tp=0.05)
    print(f"  same-bar SL+TP -> outcome={oc} gross={g:+.4f} (expect 'sl', -0.03 = conservative)")

    # future-spike invariance: SL hits at bar 2; add huge spike at bar 5 -> exit must not change
    bars_a = [(100, 101, 99, 100), (100, 100, 96, 97), (97, 130, 95, 120), (120, 140, 100, 130)]
    ga, oca, _, _ = simulate_exit(bars_a, hard_sl=0.03, tp=0.05)
    bars_b = bars_a[:2]  # truncate before the future spike
    gb, ocb, _, _ = simulate_exit(bars_b, hard_sl=0.03, tp=0.05)
    print(f"  future-spike invariance: full={oca}/{ga:+.4f} prefix={ocb}/{gb:+.4f} "
          f"(must match -> future spike ignored)")


if __name__ == "__main__":
    main()

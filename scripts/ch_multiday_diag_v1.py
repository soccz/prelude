#!/usr/bin/env python3
"""ch_multiday_diag_v1 — Track B3 정직성/베타분리 진단.

(1) 유일한 net_mean>0 셀(N=5 trail_dd0.08)이 진짜 broad 엣지인지 소수 극단winner
    degeneracy인지: top-winner 제거 시 net.
(2) 베타 분리: 같은 N일 윈도우 시장바스켓/BTC 보유수익 대비 excess, picks 가 시장을
    이기는 빈도.
(3) holdN N=1 이 R1 1일 베이스라인(no-SL EOD)과 정합한지 sanity.
(4) deep-loss(<=-5%) 빈도가 N 에 따라 어떻게 늘어나는지(하방 노출 trade-off).
"""
from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "output"
t = pd.read_csv(OUT / "ch_multiday_picks_v1.csv")


def show(c, tag):
    net = c["net"].dropna().values
    if len(net) == 0:
        print(tag, "EMPTY"); return
    am = int(net.argmax())
    print(f"\n[{tag}] n={len(net)} net_mean={net.mean():+.4f} median={np.median(net):+.4f} "
          f"frac>0={(net>0).mean():.3f}")
    print(f"  top5 net winners: {np.round(np.sort(net)[-5:],3)}")
    print(f"  net_mean drop top1={np.delete(net, am).mean():+.4f}  "
          f"drop top5={np.sort(net)[:-5].mean():+.4f}  "
          f"drop top1%={np.sort(net)[:-max(1,len(net)//100)].mean():+.4f}")
    ex = c["excess_mkt"].dropna().values
    if len(ex):
        print(f"  excess_over_mkt mean={ex.mean():+.4f}  picks_beat_mkt_frac={(ex>0).mean():.3f}")
    print(f"  mkt_window_ret mean={c['mkt_window_ret'].mean():+.4f}  "
          f"btc_window_ret mean={c['btc_window_ret'].mean():+.4f}")
    print(f"  BTC entry frac={ (c['market']=='KRW-BTC').mean():.3f}  "
          f"deep_loss(<=-5%) freq={(net<=-0.05).mean():.3f}")


# (1)+(2) the only positive cell
pos = t[(t["n_hold"] == 5) & (t["variant"] == "trail") & (t["p1"] == 0.08)]
show(pos, "N=5 trail_dd0.08 (유일 net>0 셀)")

# (3) holdN N=1 sanity (baseline 정합)
h1 = t[(t["n_hold"] == 1) & (t["variant"] == "holdN")]
show(h1, "N=1 holdN (1일 종가 = baseline 참조)")

# (4) holdN deep-loss escalation across N
print("\n[deep-loss(<=-5%) freq by N — holdN, 하방 노출 trade-off]")
for n in [1, 2, 3, 5]:
    c = t[(t["n_hold"] == n) & (t["variant"] == "holdN")]["net"].dropna().values
    if len(c):
        print(f"  N={n}: deep_loss_freq={(c<=-0.05).mean():.3f}  worst={c.min():+.3f}  "
              f"net_mean={c.mean():+.4f}")

# (5) best-Sharpe cells overall (block) — 전부 음수여도 가장 덜 나쁜 것
comp = pd.read_csv(OUT / "ch_multiday_compare_v1.csv")
print("\n[block-Sharpe 상위 6 (전반 비교)]")
for _, r in comp.sort_values("sharpe_block", ascending=False).head(6).iterrows():
    print(f"  N={int(r['n_hold'])} {r['variant']:<22} net={r['net_mean']:+.4f} "
          f"Sh_blk={r['sharpe_block']:+.2f} deep={r['deep_loss_freq']:.3f} "
          f"exMkt={r['excess_mkt_mean']:+.4f} n_blk={int(r['n_blocks'])}")

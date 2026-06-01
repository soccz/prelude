#!/usr/bin/env python3
"""Independent adversarial re-aggregation of Track B3 multiday challenger.

quant-evaluator 가 researcher 주장(보유 길수록 net·하방 악화, 유일 양수셀=베타잔여)을
picks dump 에서 독립 재집계하고, bootstrap CI95 / deflate / baseline 정합 / sim-artifact
양방향 검토를 수행한다. 새 파일만 작성(라이브 코드 불변).
"""
from __future__ import annotations
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
COST = 0.0015
np.random.seed(7)


def reagg():
    p = pd.read_csv(OUT / "ch_multiday_picks_v1.csv")
    comp = pd.read_csv(OUT / "ch_multiday_compare_v1.csv")
    print("dump variants:", sorted(p.variant.unique()), "n_holds:", sorted(p.n_hold.unique()))
    print("dump rows:", len(p))

    print("\n=== [1] INDEP re-agg holdN from picks dump vs compare CSV ===")
    print(f"{'N':>2} {'n':>5} {'net_mean':>10} {'net_med':>10} {'hit':>7} {'deep':>7} {'worst':>9} | compare net_mean")
    for n in [1, 2, 3, 5]:
        d = p[(p.variant == "holdN") & (p.n_hold == n)].dropna(subset=["net"])
        net = d.net.values
        cm = comp[(comp.vfamily == "holdN") & (comp.n_hold == n)].net_mean.values
        cm = cm[0] if len(cm) else np.nan
        print(f"{n:>2} {len(net):>5} {net.mean():>+10.6f} {np.median(net):>+10.6f} "
              f"{(net > 0).mean():>7.4f} {(net <= -0.05).mean():>7.4f} {net.min():>+9.6f} | "
              f"{cm:>+10.6f} match={abs(net.mean()-cm) < 1e-9}")

    print("\n=== [1b] excess-over-market across ALL 36 cells (beta-residual) ===")
    print(f"  cells with excess_mkt_mean>0: {(comp.excess_mkt_mean>0).sum()} of {len(comp)}")
    print(f"  excess_mkt_mean range: {comp.excess_mkt_mean.min():+.5f} to {comp.excess_mkt_mean.max():+.5f}")
    print(f"  cells with excess_pos_frac>0.5 (picks beat market >half): {(comp.excess_mkt_pos_frac>0.5).sum()}")

    print("\n=== [2] INDEP re-agg trail_dd0.08 (variant=trail, p1=0.08) by N ===")
    t = p[(p.variant == "trail") & (np.isclose(p.p1, 0.08))]
    for n in [1, 2, 3, 5]:
        d = t[t.n_hold == n].dropna(subset=["net"])
        net = d.net.values
        ex = d.excess_mkt.dropna().values
        cm = comp[(comp.variant == "trail_dd0.08") & (comp.n_hold == n)]
        cmn = cm.net_mean.values[0] if len(cm) else np.nan
        print(f"N={n}: n={len(net)} net_mean={net.mean():+.6f} (compare {cmn:+.6f} "
              f"match={abs(net.mean()-cmn) < 1e-9}) median={np.median(net):+.6f} "
              f"hit={(net > 0).mean():.4f} deep={(net <= -0.05).mean():.4f} "
              f"worst={net.min():+.6f} excess={ex.mean():+.6f} ex_pos={(ex > 0).mean():.4f}")

    return p, comp


def baseline_check(p):
    """R1 N=1 holdN net 이 r2_challenger_picks_v1.csv 의 R1_ratio net 과 byte-일치하는지."""
    print("\n=== [3] baseline byte-consistency: N=1 holdN vs r2 R1_ratio net ===")
    r2 = pd.read_csv(OUT / "r2_challenger_picks_v1.csv")
    r2 = r2[r2.policy == "R1_ratio"].copy()
    r2["entry_date"] = pd.to_datetime(r2["date"]).dt.date.astype(str)
    # r2 net 은 1일 진입가->? . compare 의 N1 holdN gross = lastClose/open - 1, net = gross-0.0015
    d1 = p[(p.variant == "holdN") & (p.n_hold == 1)].copy()
    d1["entry_date"] = pd.to_datetime(d1["entry_date"]).dt.date.astype(str)
    m = d1.merge(r2[["entry_date", "market", "net"]], on=["entry_date", "market"],
                 how="left", suffixes=("_b3", "_r2"))
    print(f"  r2 R1_ratio rows={len(r2)}  N1 holdN dump rows={len(d1)}  merged={m.net_r2.notna().sum()}")
    print(f"  r2 R1_ratio net mean = {r2.net.mean():+.6f}")
    print(f"  B3 N1 holdN gross mean (no cost) = {d1.gross.mean():+.6f}")
    print(f"  B3 N1 holdN net mean (cost {COST}) = {d1.net.mean():+.6f}")
    # r2 net: cost 차감 여부 확인
    diff = (m.gross - m.net_r2).dropna()
    print(f"  median(B3.gross - r2.net) = {diff.median():+.6f}  "
          f"mean = {diff.mean():+.6f}  (0이면 r2.net=gross; 0.0015면 r2.net 이미 net)")
    # row-level identity on gross vs r2 net
    eq_gross = np.isclose(m.gross, m.net_r2, atol=1e-9).mean()
    eq_net = np.isclose(m.net_b3, m.net_r2, atol=1e-9).mean()
    print(f"  frac rows B3.gross == r2.net : {eq_gross:.4f}")
    print(f"  frac rows B3.net   == r2.net : {eq_net:.4f}")


def bootstrap_ci(p, comp):
    """유일 양수셀 N5 trail0.08: 거래일/non-overlap 블록 bootstrap CI95(net)."""
    print("\n=== [4] bootstrap CI95 net — N5 trail_dd0.08 (the only positive cell) ===")
    d = p[(p.variant == "trail") & (np.isclose(p.p1, 0.08)) & (p.n_hold == 5)].dropna(subset=["net"]).copy()
    d["entry_date"] = pd.to_datetime(d["entry_date"]).dt.date
    net = d.net.values
    print(f"  point net_mean = {net.mean():+.6f}  n_trades={len(net)}")

    # (a) trade-level naive bootstrap (overstates n -> optimistic)
    B = 5000
    means = np.array([net[np.random.randint(0, len(net), len(net))].mean() for _ in range(B)])
    print(f"  (a) trade-level bootstrap CI95 = [{np.percentile(means,2.5):+.6f}, {np.percentile(means,97.5):+.6f}]")

    # (b) entry-day cluster bootstrap (resample whole entry-days)
    daygrp = {k: v.net.values for k, v in d.groupby("entry_date")}
    days = list(daygrp.keys())
    means_d = []
    for _ in range(B):
        samp = np.random.choice(len(days), len(days), replace=True)
        vals = np.concatenate([daygrp[days[i]] for i in samp])
        means_d.append(vals.mean())
    means_d = np.array(means_d)
    print(f"  (b) entry-day cluster bootstrap CI95 = [{np.percentile(means_d,2.5):+.6f}, {np.percentile(means_d,97.5):+.6f}]")

    # (c) non-overlap block bootstrap (block = 5 consecutive entry days)
    days_sorted = sorted(days)
    block_of = {dt: i // 5 for i, dt in enumerate(days_sorted)}
    d["block"] = d.entry_date.map(block_of)
    blkgrp = {k: v.net.values for k, v in d.groupby("block")}
    blocks = list(blkgrp.keys())
    means_b = []
    for _ in range(B):
        samp = np.random.choice(len(blocks), len(blocks), replace=True)
        vals = np.concatenate([blkgrp[blocks[i]] for i in samp])
        means_b.append(vals.mean())
    means_b = np.array(means_b)
    lo, hi = np.percentile(means_b, 2.5), np.percentile(means_b, 97.5)
    print(f"  (c) non-overlap block({len(blocks)}) bootstrap CI95 = [{lo:+.6f}, {hi:+.6f}]")
    print(f"      P(net_mean>0) under block bootstrap = {(means_b>0).mean():.4f}")
    print(f"      => CI95 excludes 0? {lo>0 or hi<0}")

    # degeneracy: top1% removal
    k1 = max(1, int(np.ceil(0.01 * len(net))))
    trimmed = np.sort(net)[:-k1]
    print(f"  degeneracy: remove top1% ({k1} trades) -> net_mean = {trimmed.mean():+.6f} (researcher claim -0.0035)")
    # top5 winners
    top5 = np.sort(net)[-5:]
    print(f"  top5 winners net = {top5}")


def deflate(comp):
    """36-trial selection deflate: block-Sharpe 의 PSR / DSR (사후 표기)."""
    print("\n=== [5] selection deflate (사후표기, §2.3) ===")
    from scipy.stats import norm
    # block-Sharpe of best cell, annualized. Convert to per-block SR for DSR.
    row = comp[(comp.variant == "trail_dd0.08") & (comp.n_hold == 5)].iloc[0]
    sr_ann = row.sharpe_block
    n_blocks = int(row.n_blocks)
    # de-annualize: sharpe_block = sr_per_block * sqrt(blocks_per_year), blocks_per_year=365/5=73
    bpy = 365 / 5
    sr_block = sr_ann / np.sqrt(bpy)
    print(f"  N5 trail0.08: sharpe_block(ann)={sr_ann:+.4f}  n_blocks={n_blocks}  sr_per_block={sr_block:+.5f}")
    # All 36 block-Sharpes (annualized) -> variance across trials for DSR E[max]
    srs_ann = comp.sharpe_block.dropna().values
    srs_block = srs_ann / np.sqrt(bpy)  # rough; bpy varies by N but use as proxy magnitude
    N_trials = 36
    var_sr = np.var(srs_block, ddof=1)
    euler = 0.5772156649
    emax = (np.sqrt(var_sr) * ((1 - euler) * norm.ppf(1 - 1.0 / N_trials)
            + euler * norm.ppf(1 - 1.0 / (N_trials * np.e))))
    print(f"  trials N={N_trials}  Var(SR_block across trials)={var_sr:.6f}  E[max SR]_expected={emax:+.5f}")
    # PSR vs 0, DSR vs expected max (Bailey & Lopez de Prado 2014), assume skew~0 kurt~3 proxy
    T = n_blocks
    def psr(sr, sr0, T):
        return float(norm.cdf((sr - sr0) * np.sqrt(T - 1)))
    print(f"  PSR(SR_block>0)  = {psr(sr_block, 0.0, T):.4f}")
    print(f"  DSR(SR_block>E[max]) = {psr(sr_block, emax, T):.4f}  (DSR<0.5 => indistinguishable from selection noise)")


def gap_sim_artifact(p):
    """sim-artifact 양방향 검토: 동봉 SL-우선 보수성 + 일봉 갭 영향, deep-loss 필연성."""
    print("\n=== [6] sim-artifact 양방향 검토 ===")
    # deep-loss increase: is it purely 'more exposure = more -5% touches'?
    print("  (b) deep-loss 증가가 단순 노출증가 통계필연인지 — holdN net 분포의 좌측꼬리")
    for n in [1, 2, 3, 5]:
        d = p[(p.variant == "holdN") & (p.n_hold == n)].dropna(subset=["net"])
        net = d.net.values
        print(f"    N={n}: P(net<=-0.05)={(net<=-0.05).mean():.4f} P(net<=-0.10)={(net<=-0.10).mean():.4f} "
              f"P(net<=-0.20)={(net<=-0.20).mean():.4f} mean_left_tail(<=-0.05)={net[net<=-0.05].mean():+.4f}")
    # gap-down severity: how many holdN losses are worse than -SL band would allow (gap risk)
    print("  (a) 갭리스크 — holdN(스탑없음) worst 가 trail/bracket 보다 깊은지 (스탑이 갭에서 못 막음)")
    for n in [5]:
        for var, p1 in [("holdN", None), ("trail", 0.08), ("bracket", 0.05)]:
            if var == "bracket":
                d = p[(p.variant == "bracket") & (np.isclose(p.p2, 0.05)) & (p.n_hold == n)].dropna(subset=["net"])
            elif var == "trail":
                d = p[(p.variant == "trail") & (np.isclose(p.p1, 0.08)) & (p.n_hold == n)].dropna(subset=["net"])
            else:
                d = p[(p.variant == "holdN") & (p.n_hold == n)].dropna(subset=["net"])
            if len(d):
                net = d.net.values
                print(f"    N={n} {var}: worst={net.min():+.4f} P(<=-0.10)={(net<=-0.10).mean():.4f} "
                      f"n_below_SLband(net<-0.06)={(net<-0.06).sum()}")


def main():
    p, comp = reagg()
    baseline_check(p)
    bootstrap_ci(p, comp)
    try:
        deflate(comp)
    except Exception as e:
        print("  deflate skipped:", e)
    gap_sim_artifact(p)


if __name__ == "__main__":
    main()

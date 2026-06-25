#!/usr/bin/env python
"""crowding_decay_v1 — Newton 과확장 함정(이미 오른 코인=칼받기) honest net 검증.

영상(Veritasium "The Trillion Dollar Equation"): Newton 의 South Sea 버블 물타기 실패 —
"천체 운동은 계산해도 사람의 광기는 계산 못 한다". 가설: D-1 까지 이미 많이 오른/쏠린 픽일수록
forward honest net↓ · deep_loss↑ (늦게 들어가는 칼받기).

데이터: output/paper_ledger_backfill.csv (5033 OOF 픽, 754일). honest 청산 = next_close EOD
(max-bracket 환상 회피 — binance lead-lag 를 죽인 일봉 낙관근사 금지). 비용 왕복 0.15% 차감.
leak: crowding 프록시(return_7d_rank/roc_3d_rank/return_5d_rank)는 D-1 feature(.shift(1) 패널),
next_close 는 D 종가 — 입력 t-1 / 타겟 t 분리.

판정축(메모리 user-downside-first / radar-not-strategy): 이건 진입 재정렬(과열 픽 DOWNRANK),
exit 아님. day-quality(REJECT, 날 전체 침묵)와 달리 코인×과열 미시 강등.
★핵심 트랩(과거 eval 지적): 거의 모든 버킷 net 음수 → "과열 net<0"는 trivially 참.
진짜 게이트 = (1) 비과열(low) 버킷이 유의하게 더 나은가(differential 실재, bootstrap CI 0 제외),
(2) btc_regime 통제 후에도 유지되는가(regime proxy 아님), (3) composite_score 와 중복 아닌가
(같은 score 밴드 안에서도 separate 하는가 — 모델이 이미 반영했으면 추가가치 0).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("crowding")

BACKFILL = Path("output/paper_ledger_backfill.csv")
OUT_DIR = Path("output")
COST = 0.0015
DEEP = -0.05
SEED = 42
N_BOOT = 2000
# 과확장 프록시 (이미 오른/쏠림). vol_5d_rank 는 변동성이라 index 제외, robustness 로만 보고.
CROWD_COLS = ["return_7d_rank", "roc_3d_rank", "return_5d_rank"]
OPERATING_REGIMES = ["bull_quiet", "bull_volatile"]   # 실제 발사하는 regime


def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if hi > lo else s * 0.0


def perm_test_within_date(df: pd.DataFrame, n_perm: int = 2000) -> dict:
    """within-date permutation: 날짜 안에서 crowd_q 라벨을 무작위 셔플 → HIGH-LOW net diff 귀무분포.

    day-quality(REJECT) 가 죽은 바로 그 시험. 날짜 구조·버킷 크기 보존하고 crowd↔net 연결만 끊음.
    p = 관측 diff 이하(더 극단 음수)인 셔플 비율. 작을수록 진짜 신호.
    """
    np.random.seed(SEED)
    net = df["net"].values
    q = df["crowd_q"].astype(str).values
    obs = net[q == "high"].mean() - net[q == "low"].mean()
    grp = df.groupby("date").indices  # date -> row positions
    cnt = 0
    perm_diffs = []
    for _ in range(n_perm):
        pq = q.copy()
        for _, pos in grp.items():
            pq[pos] = np.random.permutation(pq[pos])
        d = net[pq == "high"].mean() - net[pq == "low"].mean()
        perm_diffs.append(d)
        if d <= obs:
            cnt += 1
    return dict(observed=float(obs), p_value=float(cnt / n_perm),
                perm_mean=float(np.mean(perm_diffs)), perm_std=float(np.std(perm_diffs)))


def bucket_metrics(d: pd.DataFrame) -> dict:
    net = d["net"].values
    return dict(n=int(len(d)), net=float(np.mean(net)),
                hit=float(np.mean(net > 0)), deep=float(np.mean(net < DEEP)),
                p25=float(np.percentile(net, 25)) if len(net) else np.nan)


def day_cluster_boot(d: pd.DataFrame, col_q: str, hi="high", lo="low", rng=None) -> dict:
    """day-clustered bootstrap (벡터화): 날짜 단위 재표집 → HIGH-LO net·deep differential CI95.

    같은 날의 픽은 상관 → 날짜를 cluster 로 재표집(독립성 보존). 날짜별 집계를 선계산 후
    (N_BOOT, n_dates) fancy-indexing 으로 ratio estimator(Σnet/Σn) 를 한 번에 계산.
    """
    rng = rng or np.random.default_rng(SEED)
    sub = d[d[col_q].isin([hi, lo])].copy()
    is_hi = (sub[col_q] == hi).astype(float)
    is_lo = (sub[col_q] == lo).astype(float)
    deep = (sub["net"] < DEEP).astype(float)
    sub = sub.assign(is_hi=is_hi, is_lo=is_lo,
                     net_hi=sub["net"] * is_hi, net_lo=sub["net"] * is_lo,
                     deep_hi=deep * is_hi, deep_lo=deep * is_lo)
    agg = sub.groupby("date")[["is_hi", "is_lo", "net_hi", "net_lo", "deep_hi", "deep_lo"]].sum()
    nd = len(agg)
    if nd < 8:
        return dict(net_diff=np.nan, net_ci=[np.nan, np.nan], deep_diff=np.nan, deep_ci=[np.nan, np.nan])
    n_hi, n_lo = agg["is_hi"].values, agg["is_lo"].values
    s_hi, s_lo = agg["net_hi"].values, agg["net_lo"].values
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
        if not len(a):
            return np.nan, np.nan, np.nan
        return float(a.mean()), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

    nm, nlo, nhi = ci(net_diffs)
    dm, dlo, dhi = ci(deep_diffs)
    return dict(net_diff=nm, net_ci=[nlo, nhi], deep_diff=dm, deep_ci=[dlo, dhi])


def main():
    df = pd.read_csv(BACKFILL)
    df["net"] = df["next_close_return_pct"] / 100.0 - COST   # honest EOD 청산
    df = df.dropna(subset=["net"]).reset_index(drop=True)

    # crowding index = 사용 가능한 과확장 rank 평균 (단일 프록시 cherry-pick 회피)
    avail = [c for c in CROWD_COLS if c in df.columns]
    norm = pd.DataFrame({c: _minmax(df[c]) for c in avail})
    df["crowd_index"] = norm.mean(axis=1)
    df = df.dropna(subset=["crowd_index"]).reset_index(drop=True)
    df["crowd_q"] = pd.qcut(df["crowd_index"], 3, labels=["low", "mid", "high"], duplicates="drop")

    rng = np.random.default_rng(SEED)
    rows = []
    log.info("총 %d 픽, %d일. honest net_mean=%.4f hit=%.3f deep=%.3f (crowd_index = %s 평균)",
             len(df), df["date"].nunique(), df.net.mean(), (df.net > 0).mean(),
             (df.net < DEEP).mean(), "+".join(avail))

    # ---------- (A) 전체 crowd 3분위 ----------
    log.info("\n===== (A) crowd_index 3분위 (high=이미 오름/쏠림) =====")
    for q in ["low", "mid", "high"]:
        m = bucket_metrics(df[df.crowd_q == q])
        log.info("  %-4s n=%-5d net=%+.4f hit=%.3f deep=%.3f", q, m["n"], m["net"], m["hit"], m["deep"])
        rows.append(dict(strat="ALL", group=q, **m))
    boot_all = day_cluster_boot(df, "crowd_q", rng=rng)
    nd_lo, nd_hi = boot_all["net_ci"]
    dd_lo, dd_hi = boot_all["deep_ci"]
    log.info("  [bootstrap HIGH-LOW] net차=%+.4f CI95[%+.4f,%+.4f]%s | deep차=%+.4f CI95[%+.4f,%+.4f]%s",
             boot_all["net_diff"], nd_lo, nd_hi, "  ★CI 0 제외" if nd_hi < 0 else "",
             boot_all["deep_diff"], dd_lo, dd_hi, "  ★CI 0 제외" if dd_lo > 0 else "")
    perm = perm_test_within_date(df)
    log.info("  [permutation within-date] 관측 net차=%+.4f, 귀무평균=%+.4f, p=%.4f%s  (day-quality 는 p=0.43 으로 죽음)",
             perm["observed"], perm["perm_mean"], perm["p_value"],
             "  ★유의(p<0.05)" if perm["p_value"] < 0.05 else "")

    # ---------- (B) regime 통제 ----------
    log.info("\n===== (B) btc_regime 통제 (crowd 효과가 regime proxy 인가?) =====")
    regime_boot = {}
    for reg in df["btc_regime"].dropna().unique():
        sub = df[df.btc_regime == reg]
        if sub["crowd_q"].nunique() < 3 or len(sub) < 60:
            continue
        lo_m = bucket_metrics(sub[sub.crowd_q == "low"])
        hi_m = bucket_metrics(sub[sub.crowd_q == "high"])
        rb = day_cluster_boot(sub, "crowd_q", rng=rng)
        nm, (nlo, nhi) = rb["net_diff"], rb["net_ci"]
        op = "★발사regime" if reg in OPERATING_REGIMES else ""
        log.info("  %-14s n=%-4d LOW net=%+.4f deep=%.3f | HIGH net=%+.4f deep=%.3f | "
                 "net차=%+.4f CI[%+.4f,%+.4f]%s %s",
                 reg, len(sub), lo_m["net"], lo_m["deep"], hi_m["net"], hi_m["deep"],
                 nm, nlo, nhi, "✓" if (not np.isnan(nhi) and nhi < 0) else "", op)
        regime_boot[reg] = dict(n=len(sub), low_net=lo_m["net"], high_net=hi_m["net"],
                                low_deep=lo_m["deep"], high_deep=hi_m["deep"],
                                net_diff=nm, net_ci=[nlo, nhi], operating=reg in OPERATING_REGIMES)
        for q in ["low", "mid", "high"]:
            rows.append(dict(strat=f"regime={reg}", group=q, **bucket_metrics(sub[sub.crowd_q == q])))

    # ---------- (C) composite_score 중복성 (모델이 이미 반영했나?) ----------
    log.info("\n===== (C) composite_score 밴드 안에서도 crowd 가 separate 하나? (중복성) =====")
    df["score_q"] = pd.qcut(df["composite_score"].rank(method="first"), 3,
                            labels=["s_lo", "s_mid", "s_hi"])
    score_sep = {}
    for sq in ["s_lo", "s_mid", "s_hi"]:
        sub = df[df.score_q == sq]
        if sub["crowd_q"].nunique() < 3:
            continue
        lo_m = bucket_metrics(sub[sub.crowd_q == "low"])
        hi_m = bucket_metrics(sub[sub.crowd_q == "high"])
        log.info("  score=%-5s n=%-4d  LOW net=%+.4f | HIGH net=%+.4f | net차=%+.4f",
                 sq, len(sub), lo_m["net"], hi_m["net"], hi_m["net"] - lo_m["net"])
        score_sep[sq] = dict(low_net=lo_m["net"], high_net=hi_m["net"],
                             diff=hi_m["net"] - lo_m["net"])
    # crowd 가 높은 픽을 모델이 좋아하나? (corr)
    cs_corr = float(df[["crowd_index", "composite_score"]].corr().iloc[0, 1])
    log.info("  corr(crowd_index, composite_score)=%+.3f  (양수면 모델이 과열코인을 더 높게 점수 = 교정 여지)", cs_corr)

    # ---------- (D) DOWNRANK 효과 (과열 top 1/3 강등 시 발사regime net) ----------
    log.info("\n===== (D) DOWNRANK 효과: 발사regime 에서 high-crowd 제거 시 =====")
    op = df[df.btc_regime.isin(OPERATING_REGIMES)]
    op_all = bucket_metrics(op)
    op_keep = bucket_metrics(op[op.crowd_q != "high"])
    dropped = op[op.crowd_q == "high"]
    log.info("  발사regime 전체: n=%d net=%+.4f deep=%.3f", op_all["n"], op_all["net"], op_all["deep"])
    log.info("  high-crowd 강등 후: n=%d net=%+.4f deep=%.3f  (강등 %d픽 net=%+.4f deep=%.3f)",
             op_keep["n"], op_keep["net"], op_keep["deep"], len(dropped),
             dropped["net"].mean(), (dropped["net"] < DEEP).mean())
    downrank = dict(op_all=op_all, op_keep=op_keep,
                    dropped_n=len(dropped), dropped_net=float(dropped["net"].mean()),
                    dropped_deep=float((dropped["net"] < DEEP).mean()))

    # ---------- 저장 ----------
    res = pd.DataFrame(rows)
    res_path = OUT_DIR / "crowding_decay_v1.csv"
    res.to_csv(res_path, index=False)
    coverage = {
        "data": str(BACKFILL), "n_picks": int(len(df)), "days": int(df["date"].nunique()),
        "honest_exit": "next_close EOD (max-bracket 금지)", "cost": COST,
        "crowd_index_cols": avail, "deep_thresh": DEEP,
        "regime_dist": {k: int(v) for k, v in df["btc_regime"].value_counts().items()},
        "bootstrap": {"n_boot": N_BOOT, "cluster": "date", "seed": SEED},
        "result_all": boot_all, "permutation": perm, "result_by_regime": regime_boot,
        "score_redundancy": score_sep, "crowd_score_corr": cs_corr,
        "downrank_effect": downrank,
        "verdict_axis": "진입 재정렬(과열 DOWNRANK). 게이트: (1)bootstrap HIGH-LOW net CI 0 제외, "
                        "(2)발사regime 안에서도 유지, (3)score 밴드 안에서도 separate. radar-not-strategy.",
    }
    (OUT_DIR / "crowding_decay_coverage_v1.json").write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2, default=float))
    log.info("\nwrote %s + crowding_decay_coverage_v1.json", res_path)

    # ---------- 게이트 판정 ----------
    g1 = (not np.isnan(boot_all["net_ci"][1])) and boot_all["net_ci"][1] < 0   # 전체 net차 CI<0
    g1d = (not np.isnan(boot_all["deep_ci"][0])) and boot_all["deep_ci"][0] > 0  # deep차 CI>0
    op_regs = [r for r, v in regime_boot.items() if v["operating"]]
    g2 = any((not np.isnan(regime_boot[r]["net_ci"][1])) and regime_boot[r]["net_ci"][1] < 0
             for r in op_regs)
    g3 = sum(1 for v in score_sep.values() if v["diff"] < -0.002) >= 2   # ≥2 score밴드서 분리
    log.info("\n===== 게이트 =====")
    log.info("  G1 전체 HIGH-LOW net차 CI<0: %s | deep차 CI>0: %s", g1, g1d)
    log.info("  G2 발사regime 안 유지(net CI<0): %s", g2)
    log.info("  G3 ≥2 score밴드서 crowd separate: %s", g3)
    verdict = ("SHADOW (진입 DOWNRANK 후보)" if (g1 and g2 and g3)
               else "WEAK (일부 게이트 미달 — 추가검증)" if (g1 or g2) else "REJECT")
    log.info("  → 판정: %s", verdict)
    log.info("  (record-only. 실제 DOWNRANK 배선 = recommendation_quality evidence stratum, 사용자 컨펌)")


if __name__ == "__main__":
    main()

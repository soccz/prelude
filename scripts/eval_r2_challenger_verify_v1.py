"""quant-evaluator 독립 재검증 — R2 challenger vs R1 champion.

researcher CSV 를 그대로 믿지 않고 picks dump 에서 net·하방을 재집계하고,
거래일 블록 부트스트랩으로 R2-R1 하방 Δ 유의성(CI95 0 포함 여부)을 본다.
또한 사용자 요청대로 SL 끈 EOD net 분포로 Sortino/CVaR@5%/gap-down 빈도를
정직한 하방지표로 R1 vs R2 재계산한다.

self-contained (prelude 내부 데이터만; gan_t/xsec_alpha/fin import 0).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
PICKS = _ROOT / "output" / "r2_challenger_picks_v1.csv"
DEEP = -0.05
RNG = np.random.default_rng(20260601)


def downside_stats(net: np.ndarray, eod: np.ndarray, daily: pd.Series) -> dict:
    """하방-우선 정직 지표 묶음."""
    n = len(net)
    sd = daily.std()
    dn = daily[daily < 0]
    dstd = dn.std() if len(dn) > 1 else np.nan
    k5 = max(1, int(np.ceil(0.05 * n)))
    cvar95_net = float(np.sort(net)[:k5].mean())
    # no-SL EOD 분포 기준 (정렬키 본질 하방, SL 가리기 전)
    eod = eod[np.isfinite(eod)]
    k5e = max(1, int(np.ceil(0.05 * len(eod)))) if len(eod) else 1
    cvar95_eod = float(np.sort(eod)[:k5e].mean()) if len(eod) else np.nan
    eod_dn = eod[eod < 0]
    eod_dstd = eod_dn.std() if len(eod_dn) > 1 else np.nan
    return dict(
        n=n,
        net_mean=float(net.mean()),
        sharpe=float(net.mean() / sd * np.sqrt(365)) if sd > 0 else np.nan,
        sortino_net=float(net.mean() / dstd * np.sqrt(365)) if dstd and dstd > 0 else np.nan,
        cvar95_net=cvar95_net,
        worst_net=float(net.min()),
        # SL 끈 EOD = 정렬키 본질 하방
        eod_mean=float(eod.mean()) if len(eod) else np.nan,
        eod_sortino=float(eod.mean() / eod_dstd * np.sqrt(365)) if eod_dstd and eod_dstd > 0 else np.nan,
        eod_cvar95=cvar95_eod,
        eod_worst=float(eod.min()) if len(eod) else np.nan,
        eod_deep_freq=float((eod <= DEEP).mean()) if len(eod) else np.nan,   # P(EOD net<=-5%)
        eod_gapdn10_freq=float((eod <= -0.10).mean()) if len(eod) else np.nan,  # 깊은 갭다운
        eod_p05=float(np.percentile(eod, 5)) if len(eod) else np.nan,
        eod_median=float(np.median(eod)) if len(eod) else np.nan,
    )


def per_day_metric(df: pd.DataFrame, value: str, agg) -> pd.Series:
    """거래일별 지표 (블록 부트스트랩 단위 = 거래일)."""
    return df.groupby("date")[value].apply(agg)


def block_bootstrap_diff(r2_day: pd.Series, r1_day: pd.Series,
                          n_boot: int = 5000) -> tuple:
    """거래일 단위 페어 부트스트랩으로 mean(R2)-mean(R1) CI95.
    같은 날 top-3 는 상관 → 거래일을 블록(=리샘플 단위)으로."""
    common = r2_day.index.intersection(r1_day.index)
    a = r2_day.loc[common].values
    b = r1_day.loc[common].values
    diff = a - b
    nd = len(diff)
    obs = float(diff.mean())
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = RNG.integers(0, nd, nd)
        boots[i] = diff[idx].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_ge0 = float((boots >= 0).mean())   # R2 가 R1 보다 큰 비율
    return obs, float(lo), float(hi), p_ge0, nd


def main():
    df = pd.read_csv(PICKS, parse_dates=["date"])
    out = {}
    g = {p: x for p, x in df.groupby("policy")}
    print("policies:", list(g.keys()))
    lam_key = [k for k in g if k.startswith("R2_lam")][0]
    r1 = g["R1_ratio"].dropna(subset=["net"]).copy()
    r2 = g[lam_key].dropna(subset=["net"]).copy()
    print(f"\n=== picks: R1 n={len(r1)} days={r1['date'].nunique()} | "
          f"{lam_key} n={len(r2)} days={r2['date'].nunique()} ===")

    for name, d in [("R1_ratio", r1), (lam_key, r2)]:
        s = downside_stats(d["net"].values, d["eod_net"].values if "eod_net" in d else np.array([]),
                           d.groupby("date")["net"].mean())
        # eod_net 이 picks dump 에 없으면 net 기준만
        out[name] = s
        print(f"\n--- {name} (재집계, n={s['n']}) ---")
        for k, v in s.items():
            if k == "n":
                continue
            print(f"  {k:18s} {v:+.5f}" if isinstance(v, float) and np.isfinite(v) else f"  {k:18s} {v}")

    # eod_net 컬럼 부재 시 안내
    if "eod_net" not in df.columns:
        print("\n[주의] picks dump 에 eod_net 컬럼 없음 → no-SL EOD 하방지표는 net 기반 근사 불가.")
        print("       net(SL/TP/EOD 경로) 기준 분포로만 재집계. compare CSV 의 deep_loss_freq_noSL 는 별도 검증 필요.")

    # --- 블록 부트스트랩: net_mean Δ, deep(net<=-5%) Δ, sl-freq 대용(net 기반) ---
    print("\n=== 거래일 블록 부트스트랩 (R2 - R1), CI95 ===")
    # 1) 일평균 net
    r1_net_day = per_day_metric(r1, "net", np.mean)
    r2_net_day = per_day_metric(r2, "net", np.mean)
    obs, lo, hi, p, nd = block_bootstrap_diff(r2_net_day, r1_net_day)
    print(f"[net_mean Δ]      obs={obs:+.5f}  CI95=[{lo:+.5f}, {hi:+.5f}]  "
          f"P(Δ>=0)={p:.3f}  n_days={nd}  {'유의(0밖)' if (lo>0 or hi<0) else '비유의(0포함)'}")

    # 2) 일별 deep-loss freq (net<=-5%)
    r1_deep_day = per_day_metric(r1, "net", lambda x: float((x <= DEEP).mean()))
    r2_deep_day = per_day_metric(r2, "net", lambda x: float((x <= DEEP).mean()))
    obs, lo, hi, p, nd = block_bootstrap_diff(r2_deep_day, r1_deep_day)
    print(f"[deep(net<=-5%)Δ] obs={obs:+.5f}  CI95=[{lo:+.5f}, {hi:+.5f}]  "
          f"P(Δ>=0)={p:.3f}  n_days={nd}  {'유의(0밖)' if (lo>0 or hi<0) else '비유의(0포함)'}")

    # 3) 일별 SL stop-out freq (outcome=='sl')
    if "outcome" in df.columns:
        r1["is_sl"] = (r1["outcome"] == "sl").astype(float)
        r2["is_sl"] = (r2["outcome"] == "sl").astype(float)
        r1_sl_day = per_day_metric(r1, "is_sl", np.mean)
        r2_sl_day = per_day_metric(r2, "is_sl", np.mean)
        obs, lo, hi, p, nd = block_bootstrap_diff(r2_sl_day, r1_sl_day)
        print(f"[%SL Δ]           obs={obs:+.5f}  CI95=[{lo:+.5f}, {hi:+.5f}]  "
              f"P(Δ>=0)={p:.3f}  n_days={nd}  {'유의(0밖)' if (lo>0 or hi<0) else '비유의(0포함)'}")

    # 4) 일별 pump20 precision Δ (상방 비용)
    if "pump20_hit" in df.columns:
        r1_p_day = per_day_metric(r1, "pump20_hit", np.mean)
        r2_p_day = per_day_metric(r2, "pump20_hit", np.mean)
        obs, lo, hi, p, nd = block_bootstrap_diff(r2_p_day, r1_p_day)
        print(f"[prec@3(pump20)Δ] obs={obs:+.5f}  CI95=[{lo:+.5f}, {hi:+.5f}]  "
              f"P(Δ>=0)={p:.3f}  n_days={nd}  (음수=R2 상방 포기)")

    # --- worst floor 확인 (하드 SL claim) ---
    print(f"\n=== worst-net floor 확인 ===")
    print(f"  R1 worst net = {r1['net'].min():+.5f}  (n net<=-5% = {(r1['net']<=DEEP).sum()})")
    print(f"  {lam_key} worst net = {r2['net'].min():+.5f}  (n net<=-5% = {(r2['net']<=DEEP).sum()})")
    print(f"  R1 min outcome dist among net<=-3.1%: ",
          r1[r1['net'] <= -0.031]['outcome'].value_counts().to_dict() if 'outcome' in r1 else 'n/a')

    # --- 같은 날 픽 겹침/차이 (재정렬 isolate 확인) ---
    print(f"\n=== 같은 OOS 행 공유 검증 ===")
    print(f"  R1 dates={r1['date'].nunique()}  R2 dates={r2['date'].nunique()}  "
          f"공통날={r1['date'].isin(r2['date']).sum()>0}")

    # --- no-SL 본질 하방: down_low_ret(open->intraday low) 는 dump 에 있음 ---
    #     SL 끄면 그 픽이 장중 얼마나 빠졌나 = 정렬키 본질 하방 품질 (SL 가리기 전).
    print(f"\n=== no-SL 본질 하방 (down_low_ret = open->장중최저, dump 에 존재) ===")
    for name, d in [("R1_ratio", r1), (lam_key, r2)]:
        dl = d["down_low_ret"].dropna()
        print(f"  {name}: mean={dl.mean():+.4f} median={dl.median():+.4f} "
              f"p05={np.percentile(dl,5):+.4f} worst={dl.min():+.4f} | "
              f"P(low<=-3%)={(dl<=-0.03).mean():.4f} P(low<=-5%)={(dl<=-0.05).mean():.4f} "
              f"P(low<=-10%)={(dl<=-0.10).mean():.4f}")
    # 블록 부트스트랩: P(장중저가<=-5%) Δ, P(<=-10%) Δ (정직 하방, SL 무관)
    print(f"\n=== no-SL 본질 하방 블록 부트스트랩 (R2 - R1), CI95 ===")
    for thr, lbl in [(-0.03, "P(low<=-3%)"), (-0.05, "P(low<=-5%)"), (-0.10, "P(low<=-10%)")]:
        r1_day = per_day_metric(r1, "down_low_ret", lambda x: float((x <= thr).mean()))
        r2_day = per_day_metric(r2, "down_low_ret", lambda x: float((x <= thr).mean()))
        obs, lo, hi, p, nd = block_bootstrap_diff(r2_day, r1_day)
        print(f"  [{lbl} Δ] obs={obs:+.5f}  CI95=[{lo:+.5f}, {hi:+.5f}]  "
              f"P(Δ>=0)={p:.3f}  {'유의(0밖)' if (lo>0 or hi<0) else '비유의(0포함)'}")
    # mean down_low_ret Δ (평균적으로 덜 빠지나)
    r1_dl = per_day_metric(r1, "down_low_ret", np.mean)
    r2_dl = per_day_metric(r2, "down_low_ret", np.mean)
    obs, lo, hi, p, nd = block_bootstrap_diff(r2_dl, r1_dl)
    print(f"  [mean down_low_ret Δ] obs={obs:+.5f}  CI95=[{lo:+.5f}, {hi:+.5f}]  "
          f"P(Δ>=0)={p:.3f}  {'유의(0밖, 양수=R2 덜빠짐)' if (lo>0 or hi<0) else '비유의(0포함)'}")

    # --- selection: λ grid 5개 중 best 가 in-sample 인지 — 각 λ 의 동일 OOS net/하방 ---
    print(f"\n=== selection deflate 참고: dump 에는 R1·best-λ 만 있음 ===")
    print(f"  compare CSV 가 5 trial(R1+λ4) 전수 기록 — best λ 는 동일 OOS 에서 선정(in-sample λ).")
    print(f"  → DSR/forward 권고는 판정 카드에서.")


if __name__ == "__main__":
    main()

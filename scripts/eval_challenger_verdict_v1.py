"""quant-evaluator — A1/A2a/A3 challenger 독립 재집계 + 거래일-블록 bootstrap CI95.

researcher compare CSV 를 믿지 않고 picks dump 에서 직접 재집계한다.
- net 비용 0.15% 는 dump 의 net 컬럼에 이미 차감됨 (realize_net = gross - 0.0015).
- 지표: net_mean(trade-eq), cum(day-eq), %SL, deepNoSL(eod_net<=-5%), P(min<=-5%), hit, prec@pump20.
- bootstrap: 거래일(date) 블록 복원추출 B=5000. 각 부트표본 = 거래일 index 복원추출, 그 날의
  모든 픽 포함. Δ(challenger - R1) per-trade 통계의 CI95 + P(Δ>0).
  ★ 속도: 거래일별로 net/sl/eod/min/hit/pump20 합과 카운트를 미리 집계 → 부트는 정수 인덱스
    합산만 (groupby/concat 루프 제거). day-eq cum 은 부트표본 내 날짜순이 의미 없으므로
    'day-eq mean 의 곱 누적' 대신 day-eq net 의 산술평균(=균등가중 일수익) Δ 로 근사 보고
    (cum 자체는 point estimate 로 별도 표기, 부트는 net_mean·하방에 집중).
self-contained. 외부 폴더 import 없음.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "output"
DEEP = -0.05
RNG = np.random.default_rng(42)
B = 5000


def per_trade_stats(df: pd.DataFrame) -> dict:
    d = df.dropna(subset=["net"])
    n = len(d)
    if n == 0:
        return {}
    net = d["net"].values
    oc = d["outcome"].astype(str).values if "outcome" in d.columns else np.array([])
    eod = d["eod_net"].dropna().values if "eod_net" in d.columns else np.array([])
    minr = d["down_low_ret"].dropna().values if "down_low_ret" in d.columns else np.array([])
    p20 = d["pump20_hit"].dropna().values if "pump20_hit" in d.columns else np.array([])
    return dict(
        n=n, net_mean=float(net.mean()),
        pct_sl=float((oc == "sl").mean()) if len(oc) else np.nan,
        deepNoSL=float((eod <= DEEP).mean()) if len(eod) else np.nan,
        p_min_le5=float((minr <= -0.05).mean()) if len(minr) else np.nan,
        hit=float((net > 0).mean()),
        prec20=float(p20.mean()) if len(p20) else np.nan,
        cum=day_eq_cum(d),
    )


def day_eq_cum(df: pd.DataFrame) -> float:
    d = df.dropna(subset=["net"])
    if len(d) == 0:
        return np.nan
    daily = d.groupby("date")["net"].mean().sort_index()
    return float((1 + daily).cumprod().iloc[-1] - 1.0)


def _day_agg(df: pd.DataFrame, all_days: list, metrics: list[str]):
    """거래일별 (분자합, 분모카운트) 행렬. 부트에서 정수 인덱스로 빠르게 합산."""
    d = df.dropna(subset=["net"]).copy()
    di = {dt: i for i, dt in enumerate(all_days)}
    d["_d"] = d["date"].map(di)
    nd = len(all_days)
    num = {m: np.zeros(nd) for m in metrics}
    cnt = {m: np.zeros(nd) for m in metrics}
    daymean = np.full(nd, np.nan)        # day-eq net mean (cum 용)
    g = d.groupby("_d")
    for di_, sub in g:
        di_ = int(di_)
        net = sub["net"].values
        daymean[di_] = net.mean()
        if "net_mean" in metrics:
            num["net_mean"][di_] = net.sum(); cnt["net_mean"][di_] = len(net)
        if "hit" in metrics:
            num["hit"][di_] = (net > 0).sum(); cnt["hit"][di_] = len(net)
        if "pct_sl" in metrics and "outcome" in sub.columns:
            oc = sub["outcome"].astype(str).values
            num["pct_sl"][di_] = (oc == "sl").sum(); cnt["pct_sl"][di_] = len(oc)
        if "deepNoSL" in metrics and "eod_net" in sub.columns:
            eod = sub["eod_net"].dropna().values
            num["deepNoSL"][di_] = (eod <= DEEP).sum(); cnt["deepNoSL"][di_] = len(eod)
        if "p_min_le5" in metrics and "down_low_ret" in sub.columns:
            mn = sub["down_low_ret"].dropna().values
            num["p_min_le5"][di_] = (mn <= -0.05).sum(); cnt["p_min_le5"][di_] = len(mn)
        if "prec20" in metrics and "pump20_hit" in sub.columns:
            p2 = sub["pump20_hit"].dropna().values
            num["prec20"][di_] = p2.sum(); cnt["prec20"][di_] = len(p2)
    return num, cnt, daymean


def block_bootstrap_delta(base, chal, metrics):
    all_days = sorted(set(base["date"]) | set(chal["date"]))
    nd = len(all_days)
    bn, bc, bdm = _day_agg(base, all_days, metrics)
    cn, cc, cdm = _day_agg(chal, all_days, metrics)
    res = {}
    idx_mat = RNG.integers(0, nd, size=(B, nd))   # B 부트 × nd 일
    for m in metrics:
        deltas = np.empty(B)
        for b in range(B):
            ix = idx_mat[b]
            bnum = bn[m][ix].sum(); bcnt = bc[m][ix].sum()
            cnum = cn[m][ix].sum(); ccnt = cc[m][ix].sum()
            bv = bnum / bcnt if bcnt > 0 else np.nan
            cv = cnum / ccnt if ccnt > 0 else np.nan
            deltas[b] = cv - bv
        a = deltas[np.isfinite(deltas)]
        res[m] = dict(lo=float(np.percentile(a, 2.5)), hi=float(np.percentile(a, 97.5)),
                      mean=float(a.mean()), p_gt0=float((a > 0).mean()))
    # cum (day-eq): 부트표본 일평균 net 의 곱 누적
    cum_d = np.empty(B)
    for b in range(B):
        ix = idx_mat[b]
        bd = bdm[ix]; cd = cdm[ix]
        bd = bd[np.isfinite(bd)]; cd = cd[np.isfinite(cd)]
        cum_b = np.prod(1 + bd) - 1 if len(bd) else np.nan
        cum_c = np.prod(1 + cd) - 1 if len(cd) else np.nan
        cum_d[b] = cum_c - cum_b
    a = cum_d[np.isfinite(cum_d)]
    res["cum"] = dict(lo=float(np.percentile(a, 2.5)), hi=float(np.percentile(a, 97.5)),
                      mean=float(a.mean()), p_gt0=float((a > 0).mean()))
    return res


def load_picks(path, policy_col, base_val, chal_val):
    df = pd.read_csv(OUT / path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df[df[policy_col] == base_val].copy(), df[df[policy_col] == chal_val].copy()


def report(name, base, chal, metrics):
    print(f"\n{'='*72}\n{name}   (n_base={len(base.dropna(subset=['net']))} "
          f"n_chal={len(chal.dropna(subset=['net']))})")
    sb, sc = per_trade_stats(base), per_trade_stats(chal)
    def line(lbl, k, good):
        vb, vc = sb.get(k, np.nan), sc.get(k, np.nan)
        if not (np.isfinite(vb) and np.isfinite(vc)):
            return
        print(f"  {lbl:12s} {vb:+.5f} -> {vc:+.5f}  (Δ{vc-vb:+.5f}, good={good})")
    line("net_mean", "net_mean", "+")
    line("%SL", "pct_sl", "-")
    line("deepNoSL", "deepNoSL", "-")
    line("P(min<=-5%)", "p_min_le5", "-")
    line("hit", "hit", "+")
    line("prec20", "prec20", "+")
    line("cum(day-eq)", "cum", "+")
    boot = block_bootstrap_delta(base, chal, metrics)
    print(f"  bootstrap Δ CI95 (B={B}, date-block resample):")
    for m in metrics + ["cum"]:
        r = boot[m]
        good = "+" if m in ("net_mean", "hit", "prec20", "cum") else "-"
        excl0 = (r['lo'] > 0) or (r['hi'] < 0)
        tag = " *CI EXCLUDES 0*" if excl0 else " (spans 0)"
        print(f"    Δ{m:11s} [{r['lo']:+.5f}, {r['hi']:+.5f}] mean={r['mean']:+.5f} "
              f"P(Δ>0)={r['p_gt0']:.3f} good={good}{tag}")
    return boot


def main():
    b, c = load_picks("ch_sustainability_picks_v1.csv", "policy", "R1_baseline", "A1_sustain")
    report("A1 sustainability (best=dump_B q0.6)", b, c,
           ["net_mean", "pct_sl", "deepNoSL", "p_min_le5", "hit", "prec20"])

    b2, c2 = load_picks("ch_regime_split_picks_v1.csv", "policy", "R1_baseline", "A2a")
    report("A2a regime-rank (ALL regimes)", b2, c2,
           ["net_mean", "pct_sl", "deepNoSL", "hit"])

    df2 = pd.read_csv(OUT / "ch_regime_split_picks_v1.csv")
    df2["date"] = pd.to_datetime(df2["date"]).dt.date
    bq_b = df2[(df2.policy == "R1_baseline") & (df2.regime == "bear_quiet")].copy()
    bq_c = df2[(df2.policy == "A2a") & (df2.regime == "bear_quiet")].copy()
    report("A2a bear_quiet 단독 (cum +0.040 주장 검증, n=208/100일)", bq_b, bq_c,
           ["net_mean", "pct_sl", "deepNoSL", "hit"])

    b3, c3 = load_picks("ch_features_picks_v1.csv", "variant", "BASE", "BREADTH_LIQ")
    report("A3 BREADTH_LIQ", b3, c3,
           ["net_mean", "pct_sl", "p_min_le5", "hit"])
    print("\nDONE.")


if __name__ == "__main__":
    main()

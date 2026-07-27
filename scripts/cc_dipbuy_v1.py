"""C2 dip-buy (oversold-bounce) challenger vs R1 champion — OOS net 비교 (증거 생성).

목적 (signal-researcher 트랙 C2):
  현 챔피언 R1(모멘텀/거래대금 급증 선행 = "오르는 거 따라가기")에 *정반대 regime* 으로
  도전하는 챌린저 C2(과매도 후 반등 = mean-reversion 진입)를 **동일 OOS 기간·동일
  유니버스(top100 D-1)·동일 head 파이프·동일 청산경로**에서 net(왕복 0.15% 차감)으로 비교.

  R1 (champion, byte-identical to r2 baseline):
    - 전 top100 유니버스에서 rr_ratio = P(>=+10% up) / max(P(<=-5% down), eps) 정렬 top-3.
  C2 (dip-buy challenger, 새 entry regime):
    - top100 중 **과매도 게이지** 통과 코인(깊은 하락 + RSI 낮음 + 7d-low 에서 *반전 시작*)
      으로 유니버스를 먼저 *필터* 한 뒤 그 subset 안에서 정렬 top-3.
    - R1 이 모멘텀(상승 지속)을 따라가는 반면 C2 는 하락이 멈추고 반등이 시작되는 코인을
      고른다 → 진입 성격이 직교 → net 분포·R1 과의 상관이 다를 수 있다(분산 포트 가치).

★ 왜 R2 와 다른가 (핵심):
  - R2 는 *같은 top100 OOS 행*에서 정렬키만 교체(R1 의 후처리, 천장 공유).
  - C2 는 **entry 유니버스 자체를 oversold subset 으로 교체** → R1 이 거의 안 보는 행을 본다.
    (R1 의 bounce_off_7d_low feature 는 이미 있으나, C2 는 *오버솔드 맥락*에서 게이트로 재구성.)

★ 청산/평가 = R1 과 100% 동일 (EDA-hit ≠ ledger 함정 회피):
  - realized path = recommend.py / r2_challenger 와 동일한 15m SL -3% / TP +5% / EOD 경로청산
    (simulate_path) net of 0.15%. head 확률·fold·calibration 파이프 R1 과 완전 공유.

★ LEAK 방어 (same-day leak 2번 전적 — 양보 X):
  - 과매도 게이지 = build_market_features 의 .shift(1) feature (D-1 까지)만 사용.
    LEAK_COLS / next_* 제외. 라벨(bounce/up/down) = day-D open 대비 high/low/close (미래, 타겟).
  - head 학습/calibration = expanding train(과거 fold)에서만 fit, test fold 적용만.
    per-fold OOF bucket calibration (raw prob 과신 금지).
  - purged walk-forward + embargo=5 (downside_head_riskreward_v1.walk_forward_heads 재사용).
  - 청산 15m 경로는 진입일 D 의 in-trade outcome → leak 아님 (진입 결정은 D-1 까지).
  - oversold 게이트는 D-1 feature 의 결정론적 필터 → 새 leak 유입 구조적 0.

self-contained: prelude 내부 모듈만 import (gan_t/xsec_alpha/fin import 0).

사용:
    python scripts/cc_dipbuy_v1.py
    python scripts/cc_dipbuy_v1.py --limit-markets 80   # 개발
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# R1 과 동일한 leak-free head 파이프 재사용 (recommend.py 가 쓰는 바로 그 빌더).
from scripts.downside_head_riskreward_v1 import (  # noqa: E402
    build_panel,
    add_cross_sectional,
    attach_btc_regime,
    walk_forward_heads,
    _feats,
    UP_THRESH,
    DN_THRESH,
)
# 라이브 ledger 와 동일한 15m SL/TP/EOD 경로청산.
from scripts.recommender_downside_exit_v1 import (  # noqa: E402
    load_paths as load_execution_paths,
    simulate_path,
)

M15_DB = str(_ROOT / "data" / "upbit_15m.db")
OUT = _ROOT / "output"

ROUND_TRIP_COST = 0.0015     # 왕복 0.15%
HARD_SL = 0.03               # -3% 손절 (R1 / r2_challenger 와 동일)
TP = 0.05                    # +5% 익절
DEEP_LOSS = -0.05            # 사용자 하방 수용선: net 이보다 깊으면 '나쁜 사건'

UP_ANCHOR = "lab_up_10"      # P(>=+10%)  — R1 정렬 분자
DN_ANCHOR = "lab_dn_05"      # P(<=-5%)   — R1 정렬 분모
RR_EPS = 1e-3

TOP_K = 3
UNIVERSE_TOP_N = 100         # 라이브 R1 유니버스 = 정적 top100 (D-1)
N_FOLDS = 6
EMBARGO = 5

# ---- C2 과매도 게이지 (전부 D-1 feature, placeholder — 데이터가 결정) ----
# 과매도 = 깊게 떨어졌고(ret_7d/14d, drawdown_7d) RSI 낮고, 7d-low 에서 *살짝 반등 시작*
# (bounce_off_7d_low 가 바닥은 아님 = 떨어지는 칼이 아니라 반전 초입).
# 게이트 격자(여러 강도) — sweep 으로 trade-off 본다. selection deflate 위해 조합수 기록.
RSI_GRID = [40.0, 35.0, 30.0]            # RSI_14 <= cut (낮을수록 강한 과매도)
RET7D_GRID = [-0.08, -0.12, -0.18]       # 최근 7d 수익률 <= cut (깊은 하락)
# bounce_off_7d_low: c/7d_low - 1. 너무 0 이면 아직 바닥(칼), 너무 크면 이미 반등 끝.
# 반전 *초입* band: [BOUNCE_LO, BOUNCE_HI] (placeholder).
BOUNCE_LO = 0.01
BOUNCE_HI = 0.12

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("cc_dipbuy")


# ============================================================================
# 1. 15m 경로 로딩 + 진입일 D 경로청산 (라이브 ledger 와 동일)
# ============================================================================
def load_paths(pairs: pd.DataFrame) -> dict:
    return load_execution_paths(pairs, db_path=M15_DB)


def realize_net(bars: list) -> tuple[float, str, float]:
    """SL -3% / TP +5% / EOD 경로청산 → net = gross - 왕복비용.
    추가로 no-SL EOD net 반환 (SL 끄면 그 픽이 본질적으로 얼마나 빠졌나 = '떨어지는 칼' 검사)."""
    gross, outcome = simulate_path(bars, HARD_SL, TP, None)
    eod_gross, _ = simulate_path(bars, None, None, None)
    if not np.isfinite(gross):
        return np.nan, "nodata", np.nan
    eod_net = (eod_gross - ROUND_TRIP_COST) if np.isfinite(eod_gross) else np.nan
    return gross - ROUND_TRIP_COST, outcome, eod_net


# ============================================================================
# 2. 하방-우선 net 지표 + 일별 net 시계열 (R1 상관 계산용)
# ============================================================================
def net_metrics(trades: pd.DataFrame) -> dict:
    d = trades.dropna(subset=["net"]).copy()
    n = len(d)
    if n == 0:
        return {}
    net = d["net"].values
    daily = d.groupby("date")["net"].mean().sort_index()
    eq = (1 + daily).cumprod()
    peak = eq.cummax()
    mdd = float(((eq - peak) / peak).min())
    cum = float(eq.iloc[-1] - 1.0)
    trade_mu = float(net.mean())
    daily_mu = float(daily.mean())
    sd = float(daily.std())
    dstd = float(daily[daily < 0].std()) if (daily < 0).any() else np.nan
    k5 = max(1, int(np.ceil(0.05 * n)))
    cvar95 = float(np.sort(net)[:k5].mean())
    oc = d["outcome"]
    eod = d["eod_net"].dropna().values if "eod_net" in d.columns else np.array([])
    return dict(
        n=int(n),
        n_days=int(d["date"].nunique()),
        pct_sl=float((oc == "sl").mean()),
        deep_loss_freq_noSL=float((eod <= DEEP_LOSS).mean()) if len(eod) else np.nan,
        net_mean=trade_mu,
        net_median=float(np.median(net)),
        hit=float((net > 0).mean()),
        precision_pump20=float(d["pump20_hit"].dropna().mean())
        if d["pump20_hit"].notna().any() else np.nan,
        deep_loss_freq=float((net <= DEEP_LOSS).mean()),
        cvar95=cvar95,
        worst=float(net.min()),
        mdd=mdd,
        cum=cum,
        sharpe=float(daily_mu / sd * np.sqrt(365)) if sd and sd > 0 else np.nan,
        sortino=float(daily_mu / dstd * np.sqrt(365)) if dstd and dstd > 0 else np.nan,
        pct_tp=float((oc == "tp").mean()),
        pct_eod=float((oc == "eod").mean()),
    )


def daily_net_series(trades: pd.DataFrame) -> pd.Series:
    d = trades.dropna(subset=["net"])
    return d.groupby("date")["net"].mean().sort_index()


def regime_breakdown(trades: pd.DataFrame, tag: str) -> pd.DataFrame:
    rows = []
    for reg, g in trades.dropna(subset=["net"]).groupby("regime"):
        m = net_metrics(g)
        if not m:
            continue
        m.update(policy=tag, regime=str(reg))
        rows.append(m)
    return pd.DataFrame(rows)


# ============================================================================
# 3. 정렬 / 게이트
# ============================================================================
def topk_picks(d: pd.DataFrame, score_col: str, k: int) -> pd.DataFrame:
    """각 날 score_col 내림차순 top-k. tie-break 하방-우선(R1 정신)."""
    s = d.sort_values(
        ["date", score_col, "p_lab_dn_10", "p_lab_up_10", "exp_downside"],
        ascending=[True, False, True, False, False])
    return s.groupby("date").head(k).copy()


def oversold_mask(oos: pd.DataFrame, rsi_cut, ret7_cut) -> pd.Series:
    """C2 과매도 게이트 (전부 D-1 feature). 깊은 하락 + 낮은 RSI + 7d-low 반전 초입."""
    rsi = oos["f_rsi_14"]
    ret7 = oos["f_ret_7d"]
    bounce = oos["f_bounce_off_7d_low"]
    m = (rsi <= rsi_cut) & (ret7 <= ret7_cut) & \
        (bounce >= BOUNCE_LO) & (bounce <= BOUNCE_HI)
    return m.fillna(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-markets", type=int, default=None)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    # --- 1) leak-free panel + head 라벨 (R1 과 동일 빌더), top100 유니버스 ---
    panel = build_panel(args.limit_markets)
    panel = add_cross_sectional(panel)
    panel = attach_btc_regime(panel)
    panel = panel[panel["f_qv_rank"] <= UNIVERSE_TOP_N].copy()
    panel["in_universe"] = True
    feats = _feats(panel)
    log.info("panel(top%d) rows=%d markets=%d dates=%d feats=%d",
             UNIVERSE_TOP_N, len(panel), panel["market"].nunique(),
             panel["date"].nunique(), len(feats))

    # bounce 라벨(day-D high/open >= +5%)은 이미 lab_up_05 와 동일 — 별 head 불필요.
    # head 학습은 R1 과 동일 라벨공간 (up/down) 으로 full top100 에서 (positives 확보).
    up_labels = [f"lab_{n}" for n in UP_THRESH]
    dn_labels = [f"lab_{n}" for n in DN_THRESH]
    label_cols = up_labels + dn_labels

    # --- 2) purged WF heads (R1 과 동일 빌더, per-fold OOF bucket calib) ---
    log.info("=== walk-forward heads (R1 동일 빌더, top100) ===")
    oos = walk_forward_heads(panel, feats, label_cols, N_FOLDS, EMBARGO)
    if oos.empty:
        log.error("no OOS folds — abort"); sys.exit(1)
    up = f"p_{UP_ANCHOR}"; dn = f"p_{DN_ANCHOR}"
    oos = oos.dropna(subset=[up, dn]).copy()
    log.info("OOS rows=%d dates %s..%s folds=%s",
             len(oos), oos["date"].min(), oos["date"].max(),
             sorted(oos["fold"].unique()))

    # --- 3) 15m 경로 커버리지 ---
    conn = sqlite3.connect(M15_DB)
    m15_min = conn.execute("SELECT MIN(timestamp) FROM candles").fetchone()[0]
    conn.close()
    m15_start = pd.Timestamp(m15_min).date()
    oos = oos[oos["date"] >= m15_start].reset_index(drop=True)

    # C2 게이트가 고를 수 있는 후보 + R1 top-3 후보 모두 net 필요 → 전 OOS 행 net 부착.
    pairs = oos[["market", "date"]].drop_duplicates()
    bars_map = load_paths(pairs)
    oos = oos[oos.apply(lambda r: (r["market"], r["date"]) in bars_map,
                        axis=1)].reset_index(drop=True)
    nets, outs, eods, p20 = [], [], [], []
    for _, r in oos.iterrows():
        net, oc, eod_net = realize_net(bars_map[(r["market"], r["date"])])
        nets.append(net); outs.append(oc); eods.append(eod_net)
        p20.append(1 if (r["up_high_ret"] >= 0.20) else 0)
    oos["net"] = nets; oos["outcome"] = outs; oos["eod_net"] = eods
    oos["pump20_hit"] = p20
    oos = oos.dropna(subset=["net"]).reset_index(drop=True)
    log.info("OOS with 15m net path: rows=%d dates=%d", len(oos), oos["date"].nunique())

    # --- 4) 정렬키 ---
    oos["R1"] = oos[up] / np.maximum(oos[dn], RR_EPS)
    # C2 정렬 후보: subset 안에서 bounce 확률(p_up_05) 정렬 / R1 비율 정렬 둘 다 본다.
    p_bounce = "p_lab_up_05"

    # --- 5) EDA: 과매도 base rate (selection 투명성) ---
    log.info("=== 과매도 게이지 EDA (OOS top100, D-1 feature) ===")
    base_net = float(oos["net"].mean())
    base_deep = float((oos["eod_net"].dropna() <= DEEP_LOSS).mean())
    log.info("  전체 top100 OOS: n=%d net_mean=%+.4f noSL_deep=%.3f", len(oos), base_net, base_deep)

    # --- 6) R1 baseline (동일 OOS 행, 전 유니버스) ---
    r1_picks = topk_picks(oos, "R1", TOP_K)
    r1_m = net_metrics(r1_picks); r1_m.update(policy="R1_champion", param="full_top100_rr", gate="none")
    r1_daily = daily_net_series(r1_picks)

    # --- 7) C2 게이트 sweep (조합수 기록 — selection deflate) ---
    rows = [r1_m]
    n_combos = 1   # R1
    c2_variants = {}   # tag -> (picks, daily)
    for rsi_cut in RSI_GRID:
        for ret7_cut in RET7D_GRID:
            mask = oversold_mask(oos, rsi_cut, ret7_cut)
            sub = oos[mask]
            n_combos += 2   # subset 당 2 정렬(rr, bounce)
            n_days_avail = sub["date"].nunique()
            cand_per_day = (len(sub) / n_days_avail) if n_days_avail else 0
            if len(sub) < 60 or n_days_avail < 30:
                log.info("  gate rsi<=%.0f ret7<=%.2f: subset n=%d days=%d (TOO SPARSE, skip)",
                         rsi_cut, ret7_cut, len(sub), n_days_avail)
                continue
            for skey, sname in [("R1", "rr"), (p_bounce, "bounce")]:
                picks = topk_picks(sub, skey, TOP_K)
                m = net_metrics(picks)
                if not m:
                    continue
                tag = f"C2_rsi{int(rsi_cut)}_ret7{int(ret7_cut*100)}_{sname}"
                m.update(policy="C2_dipbuy", param=f"rsi<={rsi_cut};ret7<={ret7_cut};sort={sname}",
                         gate=f"oversold", cand_per_day=round(cand_per_day, 2))
                rows.append(m)
                c2_variants[tag] = (picks, daily_net_series(picks))
            log.info("  gate rsi<=%.0f ret7<=%.2f: subset n=%d days=%d cand/day=%.2f",
                     rsi_cut, ret7_cut, len(sub), n_days_avail, cand_per_day)

    res = pd.DataFrame(rows)
    lead = ["policy", "param", "gate", "cand_per_day", "n", "n_days",
            "pct_sl", "deep_loss_freq_noSL", "net_mean", "net_median", "hit",
            "precision_pump20", "deep_loss_freq", "cvar95", "worst", "mdd",
            "cum", "sharpe", "sortino", "pct_tp", "pct_eod"]
    res = res.reindex(columns=[c for c in lead if c in res.columns])
    res_path = OUT / "cc_dipbuy_compare_v1.csv"
    res.to_csv(res_path, index=False)

    # --- 8) 상관 + regime 분해: C2 best (net_mean 최고) vs R1 ---
    c2_rows = res[res["policy"] == "C2_dipbuy"].copy()
    corr_rows = []
    best_tag = None
    if not c2_rows.empty and c2_variants:
        # net_mean 최고 C2 변형 (동률이면 noSL_deep 낮은 쪽)
        c2_rows["_sort"] = list(zip(-c2_rows["net_mean"], c2_rows["deep_loss_freq_noSL"].fillna(1)))
        # tag 복원: param 으로 매칭
        def tag_of(p):
            for t in c2_variants:
                mm = res[(res["policy"] == "C2_dipbuy") & (res["param"] == p)]
                if not mm.empty:
                    return t if t.endswith(("rr", "bounce")) else None
            return None
        # 간단: c2_variants 의 daily 와 r1_daily 상관 전부 계산
        for tag, (picks, dly) in c2_variants.items():
            j = pd.concat([r1_daily.rename("r1"), dly.rename("c2")], axis=1).dropna()
            corr = float(j["r1"].corr(j["c2"])) if len(j) >= 10 else np.nan
            overlap_days = int(len(j))
            mm = net_metrics(picks)
            corr_rows.append(dict(tag=tag, n=mm["n"], net_mean=mm["net_mean"],
                                  sharpe=mm["sharpe"], noSL_deep=mm["deep_loss_freq_noSL"],
                                  corr_with_R1=corr, overlap_days=overlap_days))
        corr_df = pd.DataFrame(corr_rows).sort_values("net_mean", ascending=False)
        corr_df.to_csv(OUT / "cc_dipbuy_corr_v1.csv", index=False)
        best_tag = corr_df.iloc[0]["tag"]

    # regime 분해: R1 + best C2
    reg_frames = [regime_breakdown(r1_picks, "R1_champion")]
    if best_tag is not None:
        best_picks = c2_variants[best_tag][0]
        reg_frames.append(regime_breakdown(best_picks, f"C2_{best_tag}"))
    reg_df = pd.concat([f for f in reg_frames if not f.empty], ignore_index=True)
    reg_cols = ["policy", "regime", "n", "n_days", "pct_sl", "deep_loss_freq_noSL",
                "net_mean", "hit", "precision_pump20", "cvar95", "mdd", "cum", "sharpe"]
    reg_df = reg_df.reindex(columns=[c for c in reg_cols if c in reg_df.columns])
    reg_df.to_csv(OUT / "cc_dipbuy_regime_v1.csv", index=False)

    # --- 9) picks dump (감사용): R1 + best C2 side-by-side ---
    dump_cols = ["date", "market", "regime", "fold", up, dn, "p_lab_up_05",
                 "p_lab_dn_10", "exp_downside", "R1", "f_rsi_14", "f_ret_7d",
                 "f_ret_14d", "f_bounce_off_7d_low", "f_drawdown_7d",
                 "net", "outcome", "eod_net", "pump20_hit", "up_high_ret", "down_low_ret"]
    dump_cols = [c for c in dump_cols if c in oos.columns]
    dumps = [r1_picks[dump_cols].assign(policy="R1_champion")]
    if best_tag is not None:
        dumps.append(c2_variants[best_tag][0][dump_cols].assign(policy=f"C2_{best_tag}"))
    pd.concat(dumps, ignore_index=True).to_csv(OUT / "cc_dipbuy_picks_v1.csv", index=False)

    # --- 10) coverage ---
    cov = dict(
        oos_rows=int(len(oos)), oos_dates=int(oos["date"].nunique()),
        oos_window=[str(oos["date"].min()), str(oos["date"].max())],
        universe="static_top100_then_oversold_gate", n_folds=N_FOLDS, embargo=EMBARGO,
        n_combos_tried=int(n_combos),
        rsi_grid=RSI_GRID, ret7d_grid=RET7D_GRID,
        bounce_band=[BOUNCE_LO, BOUNCE_HI],
        exit_path="15m SL-3%/TP+5%/EOD net 0.15%",
        best_c2_tag=str(best_tag) if best_tag else None,
        regime_dist={k: int(v) for k, v in oos["regime"].value_counts().items()},
    )
    (OUT / "cc_dipbuy_coverage_v1.json").write_text(json.dumps(cov, indent=2))

    # ---- 콘솔 요약 ----
    log.info("\n===== C2 dip-buy vs R1 champion (top-3, OOS net 0.15%, 15m SL/TP/EOD) =====")
    log.info("  policy        param                         n   days %%SL  noSLdeep net_mean hit  Sharpe  Sortino MaxDD")
    for _, r in res.iterrows():
        log.info("  %-12s %-30s %4d %4d %.3f %.3f  %+.4f %.2f %s %s %+.3f",
                 r["policy"], str(r["param"])[:30], int(r["n"]), int(r["n_days"]),
                 r["pct_sl"], r["deep_loss_freq_noSL"] if pd.notna(r["deep_loss_freq_noSL"]) else float("nan"),
                 r["net_mean"], r["hit"],
                 f"{r['sharpe']:+.2f}" if pd.notna(r["sharpe"]) else " nan ",
                 f"{r['sortino']:+.2f}" if pd.notna(r["sortino"]) else " nan ",
                 r["mdd"])
    if corr_rows:
        log.info("\n===== C2 변형별 R1 상관 (분산 포트 가치: |corr| 낮을수록 보완) =====")
        for _, r in pd.DataFrame(corr_rows).sort_values("net_mean", ascending=False).iterrows():
            log.info("  %-28s net=%+.4f Sharpe=%s corr_R1=%s (overlap %d일)",
                     r["tag"], r["net_mean"],
                     f"{r['sharpe']:+.2f}" if pd.notna(r["sharpe"]) else "nan",
                     f"{r['corr_with_R1']:+.3f}" if pd.notna(r["corr_with_R1"]) else "nan",
                     int(r["overlap_days"]))
    log.info("DONE. wrote %s + cc_dipbuy_{corr,regime,picks,coverage}_v1.*", res_path)


if __name__ == "__main__":
    main()

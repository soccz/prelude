"""Challenger A1 — sustainability-filter (진입-quality) vs R1 champion (증거 생성).

[배경] R1 (risk-reward 레이더, top-3 일일) 의 진입 엣지는 real(lift 3-6x) 인데
net breakeven~음수 = pump-then-dump. 진짜 lever = '진입 quality'. R1 픽 중 일부는
장중 펌프(+X% 고가) 후 종가 음수/SL = 펌프-후-덤프.

[A1 가설] D-1 시점 신호로 "지속 펌프 vs 펌프-후-덤프" 를 가른다. R1 top-3 에
**dump-prone 점수(p_dumped)** 를 매겨 강등/배제하고 다음 R1 후보로 교체(R1 진입집합 위
re-selection). traded 픽의 net/하방이 실제 개선되는지 본다(덜 거래로만 net 올리는
R2 degeneracy 와 구분).

[설계 — R1 baseline 충실성]
  R1 baseline 은 r2_challenger_compare_v1 과 **동일 빌더·동일 유니버스(top100)·동일
  15m SL-3%/TP+5%/EOD 청산경로·동일 head(walk_forward_heads)** 로 재현 →
  output/r2_challenger_compare_v1.csv 의 R1_ratio row 와 일치해야 한다(자가검증).

[A1 sustainability head — leak-free]
  - 라벨(day-D 타겟, 미래 → 학습 feature 에 안 섞임): "dumped" 후보 2종 비교.
      lab_dump_A = (high/open-1 >= 0.05) AND (close/open-1 < 0)         (펌프 후 종가 음수)
      lab_dump_B = (high/open-1 >= 0.05) AND (close/open-1 < -0.02)     (펌프 후 명확 음수)
    (참고로 lab_dn_05 = day-D 장중저가 <= -5% 도 함께 head 로 학습돼 있음 → 비교축)
  - 피처 = R1 head 와 **완전히 동일한 _feats(panel)** (D-1 까지, LEAK_COLS/next_* 제외).
    이미 dump_risk 게이지 구성요소(f_ret_7d 과열, f_log_qv board-top)와
    bear_volatile regime decile 이 PRECURSOR_FEATURES 안에 들어있음.
  - 모델/calibration = walk_forward_heads (R1 과 동일: 6-fold purged WF + embargo 5,
    per-fold OOF bucket calibration, train-only fit, test fold 적용만). fold 경계 동일.

[A1 re-selection 규칙 (R1 진입집합 위)]
  각 날: R1_ratio 로 universe 정렬 → 후보 풀에서 p_dumped 가 cutoff 초과인 픽을 강등,
  통과 픽으로 top-K(=3) 채우되 부족하면 다음 R1 후보로 교체. cutoff 는 **train-only**
  (각 fold test 의 cutoff = 그 fold 이전 train OOS 의 p_dumped 분위) → OOF, leak X.
  cutoff 분위 grid (q) 와 dumped 라벨(A/B) 조합을 sweep(조합 수 기록 → deflate).

[비교 — 동일 OOS·net 0.15% 차감·15m 경로]
  R1 baseline vs A1(라벨×cutoff) :
    net_mean, hit, Sharpe/Sortino, MaxDD, cum, CVaR95,
    %SL(라이브 경로 나쁜 사건), deep_loss_freq_noSL(SL 끈 본질 하방), P(min<=-5/-10%),
    precision@3(pump20), 그리고 **픽 수 / 커버리지(거래일) / 대형주(qv_rank) 쏠림**(degeneracy).

[★ 4대 위생 — 양보 X]
  - look-ahead: 입력 feature ≤ D-1 (build_market_features shift(1)), 라벨 day-D(미래),
    head/cutoff 모두 train-only fit (test fold 적용만), embargo=5.
  - 유니버스 시간정합: top100 = D-1 qv_rank (build_market_features f_universe_qv=qv.shift(1)).
  - 비용: 왕복 0.15% 차감 net.
  - 자동주문 X: 증거 생성만, 라이브/레지스트리 파일 미편집.

★ 병렬 안전: NEW 파일만. 공유 라이브 파일(recommend.py/model_registry.py/daily_*.sh) 미편집.

사용:
    python scripts/ch_sustainability_v1.py
    python scripts/ch_sustainability_v1.py --limit-markets 60   # 개발(빠른 smoke)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.database import connect_readonly  # noqa: E402
# R1 과 동일한 leak-free head 파이프 재사용 (recommend.py / r2_challenger 와 동일 빌더).
from scripts.downside_head_riskreward_v1 import (  # noqa: E402
    build_panel,
    add_cross_sectional,
    attach_btc_regime,
    walk_forward_heads,
    _feats,
    UP_THRESH,
    DN_THRESH,
)
# r2_challenger 와 동일한 15m 경로청산 + net 지표 (라이브 ledger 손익경로).
from scripts.r2_challenger_compare_v1 import (  # noqa: E402
    load_realized_paths,
    net_metrics,
    UP_ANCHOR,       # "lab_up_10"
    DN_ANCHOR,       # "lab_dn_05"
    RR_EPS,
    TOP_K,
    UNIVERSE_TOP_N,
    N_FOLDS,
    EMBARGO,
    M15_DB,
)

OUT = _ROOT / "output"
EPS = 1e-12

# A1 sustainability ("dumped") 라벨 — day-D 타겟(미래). 학습 feature 에 안 섞임.
#   dump_A: 장중 +5% 이상 찍고 종가 음수      (펌프-후-덤프 코어 정의)
#   dump_B: 장중 +5% 이상 찍고 종가 -2% 미만  (더 명확한 덤프)
# placeholder (CLAUDE.md §2.5): 데이터가 더 좋은 정의 주면 변경.
DUMP_LABELS = {
    "dump_A": dict(up=0.05, eod=0.00),
    "dump_B": dict(up=0.05, eod=-0.02),
}
# re-selection cutoff 분위 grid (train-only OOF p_dumped 분위 — 그 분위 초과 강등).
# 높을수록 덜 걸러냄(보수적). placeholder.
CUTOFF_Q_GRID = [0.60, 0.70, 0.80, 0.90]

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("ch_sustain")


# ============================================================================
# 1. sustainability 라벨 부착 (day-D outcome → build_panel 이 이미 outcome 보유)
# ============================================================================
def add_dump_labels(panel: pd.DataFrame) -> list:
    """build_panel 이 부착한 up_high_ret/eod_ret(day-D 타겟)로 dumped 라벨 생성.
    이 라벨은 head 의 학습 타겟(미래)이지 입력 feature 가 아니다 → leak X."""
    cols = []
    for name, sp in DUMP_LABELS.items():
        lab = f"lab_{name}"
        complete = panel[["up_high_ret", "eod_ret"]].notna().all(axis=1)
        panel[lab] = (
            (panel["up_high_ret"] >= sp["up"])
            & (panel["eod_ret"] < sp["eod"])
        ).astype(float)
        panel.loc[~complete, lab] = np.nan
        cols.append(lab)
    return cols


# ============================================================================
# 2. R1 baseline 정렬 + tie-break (r2_challenger topk_picks 와 동일 정신)
# ============================================================================
def _r1_order(oos: pd.DataFrame) -> pd.DataFrame:
    """R1 = p_up10/max(p_dn5,eps) 내림차순. tie-break: p_dn10↑작게→p_up10↑크게→exp_downside↑.
    각 날 내 정렬된 rank 를 r1_rank 로 부여(1=최상위)."""
    up = f"p_{UP_ANCHOR}"
    dn = f"p_{DN_ANCHOR}"
    d = oos.copy()
    d["R1"] = d[up] / np.maximum(d[dn], RR_EPS)
    d = d.sort_values(
        ["date", "R1", "p_lab_dn_10", up, "exp_downside"],
        ascending=[True, False, True, False, False])
    d["r1_rank"] = d.groupby("date").cumcount() + 1
    return d


def topk_r1(d_ordered: pd.DataFrame, k: int) -> pd.DataFrame:
    """R1 baseline top-k (r2_challenger_compare 의 R1_ratio 와 동일)."""
    return d_ordered[d_ordered["r1_rank"] <= k].copy()


# ============================================================================
# 3. A1 re-selection — dump-prone 강등 후 다음 R1 후보로 교체 (train-only cutoff)
# ============================================================================
def per_fold_cutoff(d_ordered: pd.DataFrame, dump_col: str, q: float) -> dict:
    """각 fold k 의 cutoff = fold<k 의 (train OOS) p_dumped 의 q 분위.
    fold 0 은 이전 train OOS 가 없으므로 cutoff=+inf(강등 안 함, R1 그대로).
    → 모든 cutoff 는 test fold 데이터를 안 본다(OOF). leak X."""
    cutoffs = {}
    folds = sorted(d_ordered["fold"].unique())
    for k in folds:
        prior = d_ordered[d_ordered["fold"] < k]
        if len(prior) < 200 or prior[dump_col].notna().sum() < 50:
            cutoffs[k] = np.inf   # train 부족 → 강등 안 함(R1 그대로). leak X.
        else:
            cutoffs[k] = float(prior[dump_col].quantile(q))
    return cutoffs


def a1_reselect(d_ordered: pd.DataFrame, dump_col: str, q: float, k: int) -> pd.DataFrame:
    """각 날: R1 순위대로 후보를 보되, p_dumped > fold-cutoff 인 픽은 강등.
    통과 픽으로 top-k 채우고, 부족하면 강등된 픽으로 채운다(=R1 후보 풀 내 재선택).

    반환 플래그:
      a1_filled_from_demoted = 최종 선택된 픽 자신이 dump-prone 인데 풀 부족으로 못 비킴.
      a1_substituted = 그 픽이 R1 top-k 집합에는 없는 픽(=교체로 새로 들어온 픽).
        ★ 진짜 '교체율' 지표 — degeneracy 진단의 핵심(덜 거래 vs 진짜 교체)."""
    cutoffs = per_fold_cutoff(d_ordered, dump_col, q)
    d = d_ordered.copy()
    d["_cut"] = d["fold"].map(cutoffs)
    # dump-prone = p_dumped > cutoff. NaN p_dumped 는 통과로 본다.
    d["_dumpprone"] = (d[dump_col] > d["_cut"]).fillna(False)
    sel_frames = []
    for dt, g in d.groupby("date"):
        g = g.sort_values("r1_rank")
        r1_set = set(g[g["r1_rank"] <= k]["market"])   # 그 날 R1 top-k 시장집합
        passed = g[~g["_dumpprone"]]
        rejected = g[g["_dumpprone"]]
        need_from_rej = max(0, k - len(passed.head(k)))
        chosen = pd.concat([passed.head(k), rejected.head(need_from_rej)])
        chosen = chosen.sort_values("r1_rank").head(k).copy()
        chosen["a1_filled_from_demoted"] = chosen["_dumpprone"].values
        # R1 top-k 에 없던 시장이 들어왔으면 교체된 픽.
        chosen["a1_substituted"] = ~chosen["market"].isin(r1_set)
        sel_frames.append(chosen)
    out = pd.concat(sel_frames, ignore_index=True)
    return out


# ============================================================================
# 4. degeneracy 진단 (R2 교훈: 덜 거래 vs 진짜 개선)
# ============================================================================
def degeneracy_stats(picks: pd.DataFrame, all_dates: int) -> dict:
    d = picks.dropna(subset=["net"]).copy()
    if len(d) == 0:
        return {}
    # 대형주 쏠림: f_qv_rank (D-1 거래대금 순위, 1=최대). 낮을수록 대형주.
    qvr = d["f_qv_rank"] if "f_qv_rank" in d.columns else pd.Series(np.nan, index=d.index)
    daily_n = d.groupby("date").size()
    return dict(
        n_picks=int(len(d)),
        n_days_traded=int(d["date"].nunique()),
        coverage=float(d["date"].nunique() / all_dates) if all_dates else np.nan,
        mean_picks_per_day=float(daily_n.mean()),
        median_qv_rank=float(qvr.median()) if qvr.notna().any() else np.nan,
        frac_top10_qv=float((qvr <= 10).mean()) if qvr.notna().any() else np.nan,
        frac_top20_qv=float((qvr <= 20).mean()) if qvr.notna().any() else np.nan,
        n_unique_markets=int(d["market"].nunique()),
    )


def daily_downside(picks: pd.DataFrame) -> dict:
    """daily-candle 기반 보조 하방(15m 경로와 별개): P(min<=-5/-10%)."""
    d = picks.dropna(subset=["down_low_ret"]).copy()
    if len(d) == 0:
        return {}
    minr = d["down_low_ret"].values
    return dict(
        p_min_le_5=float((minr <= -0.05).mean()),
        p_min_le_10=float((minr <= -0.10).mean()),
        mean_min_ret=float(minr.mean()),
    )


# ============================================================================
# main
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-markets", type=int, default=None)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    # --- 1) leak-free panel (R1 과 동일 빌더) ---
    panel = build_panel(args.limit_markets)   # up_high_ret/down_low_ret/eod_ret 부착됨
    panel = add_cross_sectional(panel)
    panel = attach_btc_regime(panel)
    # 라이브 R1 유니버스 = 정적 top100 (recommend.py / r2_challenger 와 동일).
    panel = panel[panel["f_qv_rank"] <= UNIVERSE_TOP_N].copy()
    panel["in_universe"] = True
    feats = _feats(panel)

    up_labels = [f"lab_{n}" for n in UP_THRESH]   # lab_up_05/10/15/20
    dn_labels = [f"lab_{n}" for n in DN_THRESH]    # lab_dn_03/05/10
    dump_cols = add_dump_labels(panel)             # lab_dump_A/B
    label_cols = up_labels + dn_labels + dump_cols
    log.info("panel(top%d) rows=%d markets=%d dates=%d feats=%d | dump base: %s",
             UNIVERSE_TOP_N, len(panel), panel["market"].nunique(),
             panel["date"].nunique(), len(feats),
             {c: round(float(panel[c].mean()), 4) for c in dump_cols})

    # --- 2) purged WF heads (R1 동일 빌더; up/dn + dump head 동시 학습) ---
    log.info("=== walk-forward heads (R1 동일 빌더 + sustainability head) ===")
    oos = walk_forward_heads(panel, feats, label_cols, N_FOLDS, EMBARGO)
    if oos.empty:
        log.error("no OOS folds — abort")
        sys.exit(1)
    up = f"p_{UP_ANCHOR}"
    dn = f"p_{DN_ANCHOR}"
    oos = oos.dropna(
        subset=[up, dn, "up_high_ret", "down_low_ret", "eod_ret", *dump_cols]
    ).copy()
    log.info("OOS rows=%d dates %s..%s folds=%s",
             len(oos), oos["date"].min(), oos["date"].max(),
             sorted(oos["fold"].unique()))

    # --- 3) 15m 경로 net (r2_challenger 와 동일 simulate_path) ---
    with connect_readonly(M15_DB) as connection:
        m15_min = connection.execute(
            "SELECT MIN(timestamp) FROM candles"
        ).fetchone()[0]
    m15_start = pd.Timestamp(m15_min).date()
    oos = oos[oos["date"] >= m15_start].reset_index(drop=True)
    pairs = oos[["market", "date"]].drop_duplicates()
    realized = load_realized_paths(pairs)
    keys = list(zip(oos["market"], oos["date"], strict=True))
    complete = np.fromiter(
        (key in realized for key in keys),
        dtype=bool,
        count=len(keys),
    )
    oos = oos.loc[complete].reset_index(drop=True)
    realized_rows = [
        realized[key]
        for key, keep in zip(keys, complete, strict=True)
        if keep
    ]
    oos[["net", "outcome", "eod_net"]] = pd.DataFrame(
        realized_rows,
        index=oos.index,
    )
    oos["pump20_hit"] = (oos["up_high_ret"] >= 0.20).astype(int)
    oos = oos.dropna(subset=["net"]).reset_index(drop=True)
    all_dates = oos["date"].nunique()
    log.info("OOS w/ 15m net: rows=%d dates=%d (outcome %s)",
             len(oos), all_dates, oos["outcome"].value_counts().to_dict())

    # --- 4) R1 baseline (충실 재현) ---
    ordered = _r1_order(oos)
    r1_picks = topk_r1(ordered, TOP_K)
    rows = []
    m = net_metrics(r1_picks)
    m.update(degeneracy_stats(r1_picks, all_dates))
    m.update(daily_downside(r1_picks))
    m.update(policy="R1_baseline", dump_label="-", cutoff_q="-", K=TOP_K,
             frac_substituted=0.0, frac_kept_dumpprone=0.0)
    rows.append(m)
    log.info("R1 baseline: n=%d net_mean=%+.4f %%SL=%.3f deepNoSL=%.3f hit=%.2f cum=%+.3f",
             m["n_picks"], m["net_mean"], m["pct_sl"], m["deep_loss_freq_noSL"],
             m["hit"], m["cum"])

    # --- 5) A1 re-selection sweep (dump 라벨 × cutoff 분위) ---
    n_combos = len(DUMP_LABELS) * len(CUTOFF_Q_GRID)
    log.info("SELECTION: A1 조합 %d개 (dump 라벨 %d × cutoff 분위 %d). deflate 기록용.",
             n_combos, len(DUMP_LABELS), len(CUTOFF_Q_GRID))
    for dname in DUMP_LABELS:
        dcol = f"p_lab_{dname}"
        if dcol not in ordered.columns:
            log.warning("missing %s — skip", dcol)
            continue
        for q in CUTOFF_Q_GRID:
            picks = a1_reselect(ordered, dcol, q, TOP_K)
            m = net_metrics(picks)
            m.update(degeneracy_stats(picks, all_dates))
            m.update(daily_downside(picks))
            m.update(policy="A1_sustain", dump_label=dname, cutoff_q=q, K=TOP_K,
                     frac_substituted=float(picks["a1_substituted"].mean()),
                     frac_kept_dumpprone=float(picks["a1_filled_from_demoted"].mean()))
            rows.append(m)
            log.info("  A1 %s q=%.2f: n=%d sub=%.2f net_mean=%+.4f %%SL=%.3f "
                     "deepNoSL=%.3f P(min<=-5%%)=%.3f prec20=%.3f cum=%+.3f topqv=%.2f",
                     dname, q, m["n_picks"], m["frac_substituted"], m["net_mean"],
                     m["pct_sl"], m["deep_loss_freq_noSL"],
                     m.get("p_min_le_5", np.nan), m["precision_pump20"],
                     m["cum"], m.get("frac_top10_qv", np.nan))

    res = pd.DataFrame(rows)
    lead = ["policy", "dump_label", "cutoff_q", "K", "n_picks", "n_days_traded",
            "coverage", "mean_picks_per_day", "frac_substituted", "frac_kept_dumpprone",
            "pct_sl", "deep_loss_freq_noSL", "net_mean", "hit", "precision_pump20",
            "p_min_le_5", "p_min_le_10", "mean_min_ret",
            "net_median", "cvar95", "worst", "mdd", "cum", "sharpe", "sortino",
            "pct_tp", "pct_eod", "median_qv_rank", "frac_top10_qv", "frac_top20_qv",
            "n_unique_markets"]
    res = res[[c for c in lead if c in res.columns]]
    res_path = OUT / "ch_sustainability_compare_v1.csv"
    res.to_csv(res_path, index=False)

    # --- 6) best A1 선정 (하방-우선: %SL ↓ → deepNoSL ↓ → net_mean ↑), R1 대비 ---
    r1row = res[res["policy"] == "R1_baseline"].iloc[0]
    a1 = res[res["policy"] == "A1_sustain"].copy()
    a1_sorted = a1.sort_values(["pct_sl", "deep_loss_freq_noSL", "net_mean"],
                               ascending=[True, True, False])
    bestrow = a1_sorted.iloc[0] if len(a1_sorted) else None
    a1_netbest = a1.sort_values("net_mean", ascending=False).iloc[0] if len(a1) else None

    cov = dict(
        oos_rows=int(len(oos)), oos_dates=int(all_dates),
        oos_window=[str(oos["date"].min()), str(oos["date"].max())],
        universe="static_top100", n_folds=N_FOLDS, embargo=EMBARGO,
        n_a1_combos=int(n_combos),
        dump_labels=DUMP_LABELS, cutoff_q_grid=CUTOFF_Q_GRID,
        exit_path="15m SL-3%/TP+5%/EOD net 0.15%",
        r1_baseline=dict(net_mean=float(r1row["net_mean"]),
                         pct_sl=float(r1row["pct_sl"]),
                         deep_loss_freq_noSL=float(r1row["deep_loss_freq_noSL"]),
                         hit=float(r1row["hit"]), cum=float(r1row["cum"]),
                         n_picks=int(r1row["n_picks"]),
                         coverage=float(r1row["coverage"]),
                         p_min_le_5=float(r1row["p_min_le_5"]),
                         precision_pump20=float(r1row["precision_pump20"])),
        best_downside_a1=(dict(dump_label=bestrow["dump_label"],
                               cutoff_q=float(bestrow["cutoff_q"]),
                               net_mean=float(bestrow["net_mean"]),
                               pct_sl=float(bestrow["pct_sl"]),
                               deep_loss_freq_noSL=float(bestrow["deep_loss_freq_noSL"]),
                               p_min_le_5=float(bestrow["p_min_le_5"]),
                               precision_pump20=float(bestrow["precision_pump20"]),
                               cum=float(bestrow["cum"]),
                               n_picks=int(bestrow["n_picks"]),
                               coverage=float(bestrow["coverage"]),
                               frac_substituted=float(bestrow["frac_substituted"]))
                          if bestrow is not None else None),
        best_netmean_a1=(dict(dump_label=a1_netbest["dump_label"],
                              cutoff_q=float(a1_netbest["cutoff_q"]),
                              net_mean=float(a1_netbest["net_mean"]),
                              pct_sl=float(a1_netbest["pct_sl"]),
                              n_picks=int(a1_netbest["n_picks"]),
                              coverage=float(a1_netbest["coverage"]))
                         if a1_netbest is not None else None),
        regime_dist={k: int(v) for k, v in oos["regime"].value_counts().items()},
    )
    (OUT / "ch_sustainability_coverage_v1.json").write_text(json.dumps(cov, indent=2))

    # --- 7) 픽 dump (감사용): R1 vs best-downside A1 side-by-side ---
    if bestrow is not None:
        bcol = f"p_lab_{bestrow['dump_label']}"
        a1best_picks = a1_reselect(ordered, bcol, float(bestrow["cutoff_q"]), TOP_K)
        dump_keep = ["date", "market", "regime", "fold", "r1_rank", up, dn,
                     "p_lab_dn_10", bcol, "exp_downside", "R1", "f_qv_rank",
                     "net", "outcome", "eod_net", "pump20_hit",
                     "up_high_ret", "down_low_ret", "a1_filled_from_demoted"]
        dump_keep = [c for c in dump_keep if c in a1best_picks.columns]
        r1_out = r1_picks[[c for c in dump_keep if c in r1_picks.columns]].assign(policy="R1_baseline")
        a1_out = a1best_picks[dump_keep].assign(policy="A1_sustain")
        pd.concat([r1_out, a1_out], ignore_index=True).to_csv(
            OUT / "ch_sustainability_picks_v1.csv", index=False)

    # ---- 콘솔 요약 ----
    log.info("\n===== R1 baseline vs A1 sustainability (top-3, OOS net 0.15%, 15m 경로) =====")
    for _, r in res.iterrows():
        log.info("  %-11s %-6s q=%-5s n=%4d cov=%.2f sub=%s %%SL=%.3f deepNoSL=%.3f "
                 "net=%+.4f hit=%.2f prec20=%.3f Pmin5=%.3f cum=%+.3f topqv=%s",
                 r["policy"], str(r["dump_label"]), str(r["cutoff_q"]),
                 int(r["n_picks"]), r["coverage"],
                 f"{r['frac_substituted']:.2f}",
                 r["pct_sl"], r["deep_loss_freq_noSL"], r["net_mean"], r["hit"],
                 r["precision_pump20"], r["p_min_le_5"], r["cum"],
                 f"{r['frac_top10_qv']:.2f}" if pd.notna(r.get("frac_top10_qv")) else "-")
    if bestrow is not None:
        log.info("\n===== Δ(best-downside A1 - R1) =====  best A1 = %s q=%.2f",
                 bestrow["dump_label"], bestrow["cutoff_q"])
        log.info("  Δnet_mean %+.4f | Δ%%SL %+.3f | ΔdeepNoSL %+.3f | ΔP(min<=-5%%) %+.3f | "
                 "Δprec20 %+.3f | Δn_picks %+d | Δcoverage %+.3f",
                 bestrow["net_mean"] - r1row["net_mean"],
                 bestrow["pct_sl"] - r1row["pct_sl"],
                 bestrow["deep_loss_freq_noSL"] - r1row["deep_loss_freq_noSL"],
                 bestrow["p_min_le_5"] - r1row["p_min_le_5"],
                 bestrow["precision_pump20"] - r1row["precision_pump20"],
                 int(bestrow["n_picks"]) - int(r1row["n_picks"]),
                 bestrow["coverage"] - r1row["coverage"])
    log.info("DONE. wrote %s + ch_sustainability_{picks,coverage}_v1.*", res_path)


if __name__ == "__main__":
    main()

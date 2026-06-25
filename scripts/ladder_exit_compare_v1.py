#!/usr/bin/env python
"""ladder_exit_compare_v1 — 델타-사다리 청산 vs 챔피언 4-cell ablation (record-only).

영상(Veritasium "The Trillion Dollar Equation")의 Thorp/BSM dynamic 델타헤지를 현물에서
합성한 델타-사다리(arm +2/+4/+6%에서 1/3씩 부분익절, hard floor 잔여 전량)를, 챔피언
TP5/SL3 와 같은 OOF 픽셋·같은 15m 경로·같은 비용규약으로 나란히 평가한다. 자동주문 0.

설계: _workspace/ladder_exit_design_v1.md (prelude-delta-ladder-design 워크플로우, 8 에이전트).

4-cell ablation (floor 효과 vs convexity 효과 분리):
  A (챔피언)  = walk_path(sl=0.03, tp=0.05)         단일 청산 TP5/SL3
  L3          = ladder(arms 2/4/6, floor=0.03)       사다리, floor 3% (A와 floor 매칭)
  L5          = ladder(arms 2/4/6, floor=0.05)       사다리, floor 5% (사용자 -5% anchor)
  S5          = walk_path(sl=0.05, tp=0.05)          단일 청산 TP5/SL5 (L5와 floor 매칭)

효과 분리 (★ downside_metrics 실측 정의 기준 정정):
  - convexity 효과(부분청산) = L3−A (floor 3 고정) AND L5−S5 (floor 5 고정).
    floor 매칭이라 p_loss_lt_5·cvar95 는 거의 핀(동어반복) → net_mean·hit·mdd·h_gt_+5·중간손실분포로 측정.
  - floor 효과(하방폭 트레이드오프) = L5−L3 (사다리) AND S5−A (단일).
    floor 가 -3→-5 로 넓어짐 → worst·p_loss_lt_5 는 악화(넓은 floor 가 더 깊게 손실),
    대신 whipsaw 감소로 net_mean 개선 가능. 알려진 SL폭 트레이드오프.

leak 방어: 픽 = collect_oof_picks(purged WF OOF), 진입=D open, 경로=[09:00 D, 09:00 D+1),
arm/floor 체결가 = entry 기준 정확 비율(봉 high 부풀림 X), 같은 봉 floor 먼저(보수).
비용: 명목비례(분할이 수수료 안 늘림) — ladder_cost. extra_slip 노브 + conservative tier 로 강건성.

판정축: net 양전이 목적 아님(이 픽셋은 96-cell 전수 sweep 에서 전부 net 음수).
"덜 잃되 여전히 잃음" = 하방 재분배 lever 인가(deep_loss·CVaR↓를 net 비악화로). radar-not-strategy.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ledger.config import (  # noqa: E402
    CONSERVATIVE_SLIPPAGE_PCT,
    FEE_PCT,
    ROUND_TRIP_COST_PCT,
    SLIPPAGE_PCT,
)
from ledger.exit_lab import (  # noqa: E402
    LADDER_OUTCOME_MAP,
    ladder_cost,
    walk_ladder_path,
    walk_path,
)
from scripts.recommender_downside_exit_v1 import (  # noqa: E402
    K_LIST,
    M15_DB,
    add_score_and_universe,
    attach_btc_regime,
    build_panel,
    collect_oof_picks,
    downside_metrics,
    load_paths,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ladder_compare")

OUT_DIR = Path("output")
ARMS = (0.02, 0.04, 0.06)
FRACS = (1 / 3, 1 / 3, 1 / 3)
CELLS = ["A", "L3", "L5", "S5"]
CELL_FLOOR = {"A": 0.03, "L3": 0.03, "L5": 0.05, "S5": 0.05}
CELL_KIND = {"A": "single", "L3": "ladder", "L5": "ladder", "S5": "single"}
# 비용 tier: (이름, fee, base_slip). conservative 는 base_slip=0.2% 편도.
COST_TIERS = [("base", FEE_PCT, SLIPPAGE_PCT),
              ("conservative", FEE_PCT, CONSERVATIVE_SLIPPAGE_PCT)]
EXTRAS = [0.0, 0.0002, 0.0005]   # 주문별 추가 슬리피지 (사다리만 영향)

METRIC_COLS = ["n", "p_loss_lt_5", "cvar95", "worst", "mdd",
               "net_mean", "cum", "hit", "sortino",
               "mean_win", "p90_net",   # 상방 보존 측정 (h_gt_+5 는 전 cell 0.0=死문 — net cap<+5%)
               "h_lt_-10", "h_-10_to_-5", "h_-5_to_0", "h_0_to_+5", "h_gt_+5"]


def cell_trade(bars: list) -> dict:
    """경로 1개에 4 cell 전부 → {cell: (gross, n_sell, outcome)}. 비용 미적용(gross)."""
    out = {}
    g, o = walk_path(bars, 0.03, 0.05);  out["A"] = (g, 1, o)
    g, o = walk_path(bars, 0.05, 0.05);  out["S5"] = (g, 1, o)
    g, r, n = walk_ladder_path(bars, ARMS, FRACS, 0.03); out["L3"] = (g, n, LADDER_OUTCOME_MAP[r])
    g, r, n = walk_ladder_path(bars, ARMS, FRACS, 0.05); out["L5"] = (g, n, LADDER_OUTCOME_MAP[r])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-markets", type=int, default=None)
    args = ap.parse_args()
    OUT_DIR.mkdir(exist_ok=True)

    # --- leak-free panel + score + 유니버스 + regime (recommender 재사용) ---
    panel = build_panel(args.limit_markets)
    panel = add_score_and_universe(panel)
    panel = attach_btc_regime(panel)
    picks = collect_oof_picks(panel, label="lab_pump20")
    log.info("panel rows=%d  OOF picks=%d over %d days",
             len(panel), len(picks), picks["date"].nunique())

    # --- 15m 경로 커버리지 (recommender main 과 동일 절차) ---
    conn = sqlite3.connect(M15_DB)
    m15_min, m15_max = conn.execute(
        "SELECT MIN(timestamp), MAX(timestamp) FROM candles").fetchone()
    conn.close()
    m15_start = pd.Timestamp(m15_min).date()
    picks_in15 = picks[picks["date"] >= m15_start].reset_index(drop=True)
    pairs = picks_in15[["market", "date"]].drop_duplicates()
    bars_map = load_paths(pairs)
    picks_in15 = picks_in15[picks_in15.apply(
        lambda r: (r["market"], r["date"]) in bars_map, axis=1)].reset_index(drop=True)
    log.info("15m window %s -> %s  picks w/ path=%d over %d days  regime=%s",
             m15_min, m15_max, len(picks_in15), picks_in15["date"].nunique(),
             picks_in15["regime"].value_counts().to_dict())

    rows = []
    perfold_rows = []
    for K in K_LIST:
        sub = picks_in15[picks_in15["rank"] <= K].reset_index(drop=True)
        # 경로별 gross/n_sell/outcome 1회 선계산 (비용과 무관) → cell_recs[cell]
        cell_recs = {c: [] for c in CELLS}
        for _, r in sub.iterrows():
            ct = cell_trade(bars_map[(r["market"], r["date"])])
            for c in CELLS:
                g, n_sell, oc = ct[c]
                if np.isfinite(g):
                    cell_recs[c].append((r["date"], int(r["fold"]), g, n_sell, oc))

        for tier_name, fee, bslip in COST_TIERS:
            for extra in EXTRAS:   # base·conservative 둘 다 extra 스윕 (사다리 분할 페널티 보수성)
                for c in CELLS:
                    recs = cell_recs[c]
                    if not recs:
                        continue
                    td = pd.DataFrame(recs, columns=["date", "fold", "gross", "n_sell", "outcome"])
                    td["net"] = td["gross"] - td["n_sell"].map(
                        lambda n: ladder_cost(n, fee, bslip, extra))
                    mt = downside_metrics(td)
                    wins = td["net"][td["net"] > 0]
                    mt["mean_win"] = round(float(wins.mean()), 5) if len(wins) else 0.0
                    mt["p90_net"] = round(float(td["net"].quantile(0.90)), 5)
                    mt.update(K=K, cell=c, kind=CELL_KIND[c], floor=CELL_FLOOR[c],
                              cost_tier=tier_name, extra_slip=extra,
                              mean_n_sell=round(float(td["n_sell"].mean()), 3))
                    rows.append(mt)

                    # per-fold (primary config: base, extra=0 만 — 부호 안정성 G5)
                    if tier_name == "base" and extra == 0.0:
                        for fold, fd in td.groupby("fold"):
                            fm = downside_metrics(fd)
                            perfold_rows.append(dict(
                                K=K, cell=c, fold=int(fold), n=fm.get("n", 0),
                                net_mean=fm.get("net_mean", np.nan),
                                p_loss_lt_5=fm.get("p_loss_lt_5", np.nan),
                                cvar95=fm.get("cvar95", np.nan),
                                mdd=fm.get("mdd", np.nan)))

    res = pd.DataFrame(rows)
    lead = ["K", "cell", "kind", "floor", "cost_tier", "extra_slip", "mean_n_sell"] + METRIC_COLS
    res = res[[c for c in lead if c in res.columns]]
    res = res.sort_values(["K", "cost_tier", "extra_slip", "cell"]).reset_index(drop=True)
    res_path = OUT_DIR / "ladder_exit_compare_v1.csv"
    res.to_csv(res_path, index=False)
    perfold = pd.DataFrame(perfold_rows)
    perfold_path = OUT_DIR / "ladder_exit_perfold_v1.csv"
    perfold.to_csv(perfold_path, index=False)
    log.info("wrote %s (%d rows) + %s (%d rows)",
             res_path, len(res), perfold_path, len(perfold))

    # ===================== 핵심 비교 (primary: K=3, base, extra=0) =====================
    def cell_row(K, cell, tier="base", extra=0.0):
        m = res[(res.K == K) & (res.cell == cell) & (res.cost_tier == tier)
                & (res.extra_slip == extra)]
        return m.iloc[0].to_dict() if len(m) else None

    K0 = 3
    A, L3, L5, S5 = (cell_row(K0, c) for c in ["A", "L3", "L5", "S5"])

    def delta(x, y, keys):
        return {k: round(float(x[k]) - float(y[k]), 5) for k in keys}

    conv_keys = ["net_mean", "hit", "mean_win", "p90_net", "mdd", "h_-5_to_0", "cvar95", "p_loss_lt_5"]
    floor_keys = ["p_loss_lt_5", "cvar95", "worst", "net_mean", "mdd"]
    comparison = {}
    if all(v is not None for v in [A, L3, L5, S5]):
        comparison = {
            "convexity_L3_vs_A": delta(L3, A, conv_keys),     # floor 3 고정
            "convexity_L5_vs_S5": delta(L5, S5, conv_keys),   # floor 5 고정
            "floor_L5_vs_L3": delta(L5, L3, floor_keys),      # 사다리 floor 3→5
            "floor_S5_vs_A": delta(S5, A, floor_keys),        # 단일 floor 3→5
            "levels_K3_base": {c: {k: round(float(d[k]), 5) for k in METRIC_COLS}
                               for c, d in [("A", A), ("L3", L3), ("L5", L5), ("S5", S5)]},
        }

    # per-fold net_mean 부호 안정성 (G5)
    fold_sign = {}
    if len(perfold):
        for c in CELLS:
            sub = perfold[(perfold.K == K0) & (perfold.cell == c)]
            fold_sign[c] = {
                "net_mean_by_fold": [round(float(v), 5) for v in sub.sort_values("fold")["net_mean"]],
                "n_pos_folds": int((sub["net_mean"] > 0).sum()),
            }

    coverage = {
        "design_doc": "_workspace/ladder_exit_design_v1.md",
        "picks_total": int(len(picks)),
        "picks_with_15m_path": int(len(picks_in15)),
        "days": int(picks_in15["date"].nunique()),
        "date_range_15m": [str(m15_min), str(m15_max)],
        "regime_dist": {k: int(v) for k, v in picks_in15["regime"].value_counts().items()},
        "embargo_days_in_picks": 5,   # recall_universe EMBARGO=5 — 정직 표기 (10d 명목 금지)
        "n_folds": int(picks_in15["fold"].nunique()),
        "arms": list(ARMS), "fractions": [round(f, 4) for f in FRACS],
        "cost_model": {
            "fee_pct_per_side": FEE_PCT, "base_slippage_pct_per_side": SLIPPAGE_PCT,
            "round_trip_base": ROUND_TRIP_COST_PCT,
            "conservative_slippage_pct_per_side": CONSERVATIVE_SLIPPAGE_PCT,
            "extra_slip_per_tranche_grid": EXTRAS,
            "note": "수수료 명목비례 — tranche 분할이 fee 안 늘림. single-exit cell 은 n_sell=1 → 챔피언과 비트동일 비용.",
        },
        "trials_note": "primary 1셀(arms 2/4/6, fractions thirds) 고정. DSR trials 누적 산정 시 "
                       "본 12셀 + prior 96(recommender_downside_exit) + 8(ch_sustainability) ≥ 116 (사후 deflation 보고용, §2.3 게이트 강제 X).",
        "comparison_K3_base": comparison,
        "fold_sign_K3": fold_sign,
        "verdict_axis": "net 음수 픽셋 — 'convexity(L3 vs A): net 비악화로 하방 재분배?' + 'floor(L5 vs L3): whipsaw 이득 vs worst 악화'. record-only.",
    }
    cov_path = OUT_DIR / "ladder_exit_coverage_v1.json"
    cov_path.write_text(json.dumps(coverage, ensure_ascii=False, indent=2))
    log.info("wrote %s", cov_path)

    # ===================== 콘솔 리포트 =====================
    log.info("\n===== LEVELS (K=3, base cost, extra_slip=0) =====")
    show = res[(res.K == K0) & (res.cost_tier == "base") & (res.extra_slip == 0.0)]
    log.info("%-4s %-6s floor n=%-5s  net_mean   hit    p<-5%%   cvar95   mean_win  p90_net  mdd     n_sell",
             "cell", "kind", "")
    for _, r in show.iterrows():
        log.info("%-4s %-6s %.2f  n=%-5d  %+.4f  %.3f  %.3f  %+.4f  %+.4f   %+.4f  %+.3f  %.2f",
                 r["cell"], r["kind"], r["floor"], int(r["n"]), r["net_mean"], r["hit"],
                 r["p_loss_lt_5"], r["cvar95"], r["mean_win"], r["p90_net"], r["mdd"], r["mean_n_sell"])

    if comparison:
        log.info("\n===== CONVEXITY 효과 (부분청산, floor 매칭) =====")
        log.info("L3 vs A  (floor 3): %s", comparison["convexity_L3_vs_A"])
        log.info("L5 vs S5 (floor 5): %s", comparison["convexity_L5_vs_S5"])
        log.info("→ Δmean_win<0 = 부분청산이 평균 승자를 깎음(상방 절단). floor5 에선 Δp_loss_lt_5<0 = deep-loss 재분배.")
        log.info("  net 비악화면 '평균 승자 희생 ↔ deep-loss 빈도↓' 의 하방-우선 reshaping (downside-first 정합).")
        log.info("\n===== FLOOR 효과 (하방폭 트레이드오프) =====")
        log.info("L5 vs L3 (사다리 3→5): %s", comparison["floor_L5_vs_L3"])
        log.info("S5 vs A  (단일 3→5):   %s", comparison["floor_S5_vs_A"])
        log.info("→ floor 넓힘: worst↓·p<-5%%↑(더 깊은 손실) 가 정상. net_mean↑면 whipsaw 회피 이득.")

    if fold_sign:
        log.info("\n===== per-fold net_mean 부호 안정성 (K=3, base) =====")
        for c in CELLS:
            log.info("  %-4s folds=%s  (n_pos=%d/%d)", c,
                     fold_sign[c]["net_mean_by_fold"], fold_sign[c]["n_pos_folds"],
                     len(fold_sign[c]["net_mean_by_fold"]))

    log.info("\n[판정 가이드] net 음수 픽셋 — ADOPT 라도 record-only(라이브 교체는 사용자 컨펌). "
             "floor5 전제 L5 vs S5: Δp_loss_lt_5≤-0.05 & Δnet≥-0.001 면 하방 재분배 가치(mean_win 희생은 명시). "
             "conservative+extra5bp 에서 net 0.1%p 초과 악화면 REJECT.")


if __name__ == "__main__":
    main()

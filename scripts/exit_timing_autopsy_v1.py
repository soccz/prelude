#!/usr/bin/env python
"""exit_timing_autopsy_v1 — "언제 파는 게 최적인가" 정조준.

동기: self_impact 에서 발사(ACTIVE) 코인이 장중 fwd_max +9.9% 까지 튀고 EOD -5.85% 로 덤프.
엣지는 '무엇을 사느냐'(진입)에 이미 있고, 잃는 건 '언제 파느냐'. 96-cell 은 이미 hold-to-EOD
(net -2.1%) 가 최악, TP5 가 -0.14% 까지 복구, trailing 은 패배를 보여줌. 남은 frontier:
TP5/SL8 의 net(≈breakeven) 을 champion 의 deep-loss(0) 와 함께 얻을 수 있나? = 부분익절(사다리).

(A) 경로 autopsy: 장중 max 분포·스파이크 타이밍·"스파이크가 유지되나 덤프되나"(touched+5% → EOD).
(B) 청산 토너먼트: hold/champ/bestnet + 사다리 + partial-into-strength(half@5 rest run) 를 같은
    OOF 픽셋·15m 경로·honest 비용으로. net↔deep frontier 에서 champion 을 개선하는 게 있나.

leak: 모든 청산 룰 causal(미래 봉 미열람), 진입=첫봉 open, 체결=정확비율, 같은봉 floor/SL 먼저.
honest 비용 ladder_cost(명목비례). picks=collect_oof_picks(purged WF OOF). 판정 record-only.
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

from ledger.exit_lab import ladder_cost, walk_ladder_path, walk_path  # noqa: E402
from scripts.recommender_downside_exit_v1 import (  # noqa: E402
    K_LIST, M15_DB, add_score_and_universe, attach_btc_regime, build_panel,
    collect_oof_picks, downside_metrics, load_paths, simulate_path,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("autopsy")
OUT_DIR = Path("output")
OPERATING = ["bull_quiet", "bull_volatile"]
BIG = 10.0  # 절대 안 닿는 arm (그 tranche 는 EOD/floor 로 흘림)

# 청산 정책: name -> (kind, params). kind ∈ {path, ladder, trail}
POLICIES = {
    "hold_eod":        ("path", dict(sl=None, tp=None)),
    "champ_tp5_sl3":   ("path", dict(sl=0.03, tp=0.05)),     # 현 라이브
    "bestnet_tp5_sl8": ("path", dict(sl=0.08, tp=0.05)),     # 96-cell best net
    # autopsy 가설: 고점이 초반·스파이크 안 유지 → 더 일찍/낮게 익절이 더 잡나? (deep 유지 위해 SL3 고정)
    "tp2_sl3":         ("path", dict(sl=0.03, tp=0.02)),
    "tp3_sl3":         ("path", dict(sl=0.03, tp=0.03)),
    "tp4_sl3":         ("path", dict(sl=0.03, tp=0.04)),
    "ladder_123_fl3":  ("ladder", dict(arms=(0.01, 0.02, 0.03), fr=(1/3, 1/3, 1/3), fl=0.03)),
    "ladder_246_fl5":  ("ladder", dict(arms=(0.02, 0.04, 0.06), fr=(1/3, 1/3, 1/3), fl=0.05)),
    "ladder_246_fl3":  ("ladder", dict(arms=(0.02, 0.04, 0.06), fr=(1/3, 1/3, 1/3), fl=0.03)),
    "half5_rest10_fl5": ("ladder", dict(arms=(0.05, 0.10), fr=(0.5, 0.5), fl=0.05)),   # 반 @+5, 반 @+10/EOD
    "half5_restEOD_fl3": ("ladder", dict(arms=(0.05, BIG), fr=(0.5, 0.5), fl=0.03)),  # 반 @+5, 반 EOD, floor3
    "third5_rest_fl3": ("ladder", dict(arms=(0.05, BIG), fr=(1/3, 2/3), fl=0.03)),    # 1/3 @+5, 2/3 EOD floor3
    "trail5":          ("trail", dict(sl=None, tp=None, trail=0.05)),
}


def run_policy(bars, kind, p):
    """→ (gross, n_sell, outcome) outcome ∈ downside_metrics 어휘(sl/tp/trail/eod)."""
    if kind == "path":
        g, o = walk_path(bars, p["sl"], p["tp"])
        return g, 1, o
    if kind == "trail":
        g, o = simulate_path(bars, p["sl"], p["tp"], p["trail"])
        return g, 1, o
    g, r, n = walk_ladder_path(bars, p["arms"], p["fr"], p["fl"])
    m = {"floor": "sl", "partial_floor": "sl", "tp_full": "tp", "partial_eod": "eod", "eod": "eod"}
    return g, n, m[r]


def path_features(bars):
    o = bars[0][0]
    highs = np.array([b[1] for b in bars]); lows = np.array([b[2] for b in bars])
    n = len(bars)
    max_ret = highs.max() / o - 1.0
    min_ret = lows.min() / o - 1.0
    eod_ret = bars[-1][3] / o - 1.0
    bar_max = int(highs.argmax())
    # 첫 +K% 도달 봉 (causal)
    def first_cross(k):
        idx = np.where(highs >= o * (1 + k))[0]
        return int(idx[0]) if len(idx) else -1
    return dict(n=n, max_ret=max_ret, min_ret=min_ret, eod_ret=eod_ret,
                t_max=bar_max / max(1, n - 1),
                touched5=max_ret >= 0.05, touched10=max_ret >= 0.10,
                fc5=first_cross(0.05))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-markets", type=int, default=None)
    args = ap.parse_args()
    OUT_DIR.mkdir(exist_ok=True)

    panel = attach_btc_regime(add_score_and_universe(build_panel(args.limit_markets)))
    picks = collect_oof_picks(panel, label="lab_pump20")
    conn = sqlite3.connect(M15_DB)
    m15_start = pd.Timestamp(conn.execute("SELECT MIN(timestamp) FROM candles").fetchone()[0]).date()
    conn.close()
    picks = picks[picks["date"] >= m15_start].reset_index(drop=True)
    bars_map = load_paths(picks[["market", "date"]].drop_duplicates())
    picks = picks[picks.apply(lambda r: (r["market"], r["date"]) in bars_map, axis=1)].reset_index(drop=True)
    sub = picks[picks["rank"] <= 3].reset_index(drop=True)
    log.info("picks(K=3) w/ 15m path: %d coin-days, %d days, regime=%s",
             len(sub), sub["date"].nunique(), sub["regime"].value_counts().to_dict())

    # ---------- (A) 경로 autopsy ----------
    feats = pd.DataFrame([path_features(bars_map[(r["market"], r["date"])]) for _, r in sub.iterrows()])
    feats["regime"] = sub["regime"].values
    feats["rank1"] = (sub["rank"] == 1).values
    log.info("\n===== (A) 장중 경로 autopsy (K=3) =====")
    for tag, d in [("ALL", feats), ("rank1(최강픽)", feats[feats.rank1]),
                   ("bull_regime", feats[feats.regime.isin(OPERATING)])]:
        t5 = d[d.touched5]
        log.info("  %-14s n=%-5d max_ret p50=%+.3f p90=%+.3f | P(touch+5%%)=%.3f P(+10%%)=%.3f | "
                 "EOD p50=%+.3f | t_max중앙=%.2f",
                 tag, len(d), d.max_ret.median(), d.max_ret.quantile(0.9),
                 d.touched5.mean(), d.touched10.mean(), d.eod_ret.median(), d.t_max.median())
        if len(t5):
            log.info("       └ touched+5%% 후: EOD p50=%+.3f, P(EOD>0)=%.3f  ← 스파이크 유지 vs 덤프",
                     t5.eod_ret.median(), (t5.eod_ret > 0).mean())
    autopsy = {
        "n": int(len(feats)), "max_p50": float(feats.max_ret.median()),
        "max_p90": float(feats.max_ret.quantile(0.9)),
        "p_touch5": float(feats.touched5.mean()), "p_touch10": float(feats.touched10.mean()),
        "eod_p50": float(feats.eod_ret.median()), "t_max_med": float(feats.t_max.median()),
        "touched5_eod_p50": float(feats[feats.touched5].eod_ret.median()),
        "touched5_eod_pos": float((feats[feats.touched5].eod_ret > 0).mean()),
    }

    # ---------- (B) 청산 토너먼트 ----------
    log.info("\n===== (B) 청산 토너먼트 (K=3, honest net, 명목비례 비용) =====")
    log.info("  %-18s net      deep   cvar95   hit    mean_win  p90     mdd      n_sell", "policy")
    rows = []
    for name, (kind, p) in POLICIES.items():
        recs = []
        for _, r in sub.iterrows():
            bars = bars_map[(r["market"], r["date"])]
            g, n_sell, oc = run_policy(bars, kind, p)
            if not np.isfinite(g):
                continue
            recs.append((r["date"], g - ladder_cost(n_sell), oc, n_sell))
        td = pd.DataFrame(recs, columns=["date", "net", "outcome", "n_sell"])
        m = downside_metrics(td)
        wins = td["net"][td["net"] > 0]
        mw = float(wins.mean()) if len(wins) else 0.0
        p90 = float(td["net"].quantile(0.9))
        mns = float(td["n_sell"].mean())
        log.info("  %-18s %+.4f  %.3f  %+.4f  %.3f  %+.4f  %+.4f  %+.3f  %.2f",
                 name, m["net_mean"], m["p_loss_lt_5"], m["cvar95"], m["hit"], mw, p90, m["mdd"], mns)
        rows.append(dict(policy=name, kind=kind, net_mean=m["net_mean"], p_loss_lt_5=m["p_loss_lt_5"],
                         cvar95=m["cvar95"], hit=m["hit"], mean_win=round(mw, 5), p90=round(p90, 5),
                         mdd=m["mdd"], h_gt_5=m["h_gt_+5"], mean_n_sell=round(mns, 3),
                         pct_eod=m.get("pct_eod", np.nan)))

    res = pd.DataFrame(rows).sort_values("net_mean", ascending=False)
    res.to_csv(OUT_DIR / "exit_timing_autopsy_v1.csv", index=False)

    # frontier 판정: champion 대비 net↑ & deep≤ 인 정책
    champ = res[res.policy == "champ_tp5_sl3"].iloc[0]
    dominators = res[(res.net_mean > champ.net_mean + 0.0005) & (res.p_loss_lt_5 <= champ.p_loss_lt_5 + 0.02)]
    log.info("\n===== frontier 판정 (champion TP5/SL3 net=%+.4f deep=%.3f 대비) =====",
             champ.net_mean, champ.p_loss_lt_5)
    if len(dominators):
        for _, r in dominators.iterrows():
            log.info("  ★ %-18s net=%+.4f(Δ%+.4f) deep=%.3f(Δ%+.3f) — champion 개선 후보",
                     r.policy, r.net_mean, r.net_mean - champ.net_mean,
                     r.p_loss_lt_5, r.p_loss_lt_5 - champ.p_loss_lt_5)
    else:
        log.info("  champion 을 net↑·deep≤ 로 동시 개선하는 정책 없음 (frontier 위 champion 이 효율적)")

    coverage = {
        "picks_k3": int(len(sub)), "days": int(sub["date"].nunique()),
        "regime_dist": {k: int(v) for k, v in sub["regime"].value_counts().items()},
        "autopsy": autopsy, "policies": rows,
        "champion": {"net": float(champ.net_mean), "deep": float(champ.p_loss_lt_5)},
        "dominators": dominators["policy"].tolist(),
        "note": "honest 15m 경로. hold-to-EOD 가 최악·TP5 가 복구를 96-cell 이 이미 보임. 여기선 "
                "부분익절(사다리)이 net↔deep frontier 에서 champion 을 개선하는지. record-only.",
    }
    (OUT_DIR / "exit_timing_autopsy_coverage_v1.json").write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2, default=float))
    log.info("\nwrote output/exit_timing_autopsy_v1.csv + coverage_v1.json")


if __name__ == "__main__":
    main()

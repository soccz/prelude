"""Downside-first exit re-optimization for the rank-mean recommender top-K picks.

사용자 명시 선호(이 워크플로의 단일 질문):
  "상승 보장보다 하락 위험 적은 게 더 좋다. 평균 -5% 정도 손실은 괜찮다."
  → 1순위 지표를 평균/Sharpe 가 아니라 **하방위험**으로:
    - P(loss < -5%)   : -5% 보다 깊은 손실의 빈도 (사용자가 가장 싫어하는 사건)
    - CVaR95          : 최악 5% trade 의 평균 손실
    - worst_trade     : 단일 최악 trade
    - MaxDD           : day-equal-weight 누적 곡선의 최대 낙폭
    - %stopped (SL)   : 하드 SL 에 잘린 비율
    - loss histogram  : [<-10%, -10~-5%, -5~0%, 0~+5%, >+5%] 비율
    부수(포기하는 상승): 평균/누적/hit/Sortino.

대상 = recall_universe_recommender_v1 의 rank-mean 스코어러가 매일 OOF 로 뽑는
       cross-section top-K(K=3 메인, 5 보조) 픽. 15m 경로가 있는 구간만.
진입 = 알림 후 첫 실행가능 15m open(09:15 KST).
경로 = [09:15 D, 09:15 D+1) 의 완결된 96개 15m 봉.
청산정책 그리드를 15m 봉 시간순 walk 로 시뮬 후 하방위험으로 재집계.

★ LEAK 방어 (same-day leak 2번 전적):
  - 진입 선택: recall recommender 와 동일한 leak-free 파이프 재사용.
      * feature = build_market_features 의 market 별 .shift(1) (D-1 까지).
      * score = D-1 feature 의 cross-section rank-mean (hand-pick/weight tuning X).
      * 유니버스(dynamic: top100 qv OR qv_surge_7d>=3x) 컷도 D-1 정보.
      * fold = purged walk-forward(embargo 5d). top-K 는 **test fold(OOF)** 에서만.
        train fold 픽은 집계에서 제외 → in-sample 청산 최적화 selection bias 차단.
  - day-D 경로: in-trade outcome (leak 아님). 미래 봉 미리보기 X — 15m 봉을
    시간 오름차순으로만 순회. 같은 봉 안 SL·TP 동시 → SL 먼저(보수).
  - 청산정책 그리드는 모든 trade 에 동일 적용 (정책 선택이 데이터를 미리 보지 않음).

거래비용: 왕복 0.15% 차감(net = gross - 0.0015). spot-only(공매도 X).
사이징: day-equal-weight (하루 안 K 픽 평균 → MaxDD/누적이 per-trade 클러스터링에
        과장 안 되게). per-trade 통계는 trade pool 그대로.

selection 기록: K{3,5} × policy grid (하드SL × TP × trailing). 아래 GRID 크기 로그.

사용:
    python scripts/recommender_downside_exit_v1.py
    python scripts/recommender_downside_exit_v1.py --limit-markets 60   # 개발
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# recall recommender 의 leak-free 파이프 재사용 (self-contained: prelude 내부 모듈만)
from scripts.recall_universe_recommender_v1 import (  # noqa: E402
    build_panel,
    add_score_and_universe,
    attach_btc_regime,
    make_folds,
    N_FOLDS,
    EMBARGO,
)
from ledger.path_quality import assess_15m_windows  # noqa: E402

D1_DB = "data/upbit_d1.db"
M15_DB = "data/upbit_15m.db"
OUT_DIR = Path("output")
ROUND_TRIP_COST = 0.0015
BARS_PER_HOUR = 4
DEEP_LOSS_THRESH = -0.05   # 사용자 기준: 이보다 깊은 손실이 '나쁜 사건'
EXECUTION_OFFSET = pd.Timedelta(hours=9, minutes=15)

K_LIST = [3, 5]            # 메인 3, 보조 5

# 청산정책 그리드 (사용자 요청 + trailing 보조). placeholder — 데이터가 더 좋은 값 주면 변경.
HARD_SL = [None, 0.03, 0.05, 0.08]
TP_CAP = [None, 0.05, 0.08, 0.10]
TRAIL = [None, 0.05, 0.08]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("rec_downside")


# ============================================================================
# 1. OOF 일별 top-K 픽 수집 (recall recommender 와 동일한 leak-free fold)
# ============================================================================
def collect_oof_picks(panel: pd.DataFrame, label: str = "lab_pump20") -> pd.DataFrame:
    """test fold(OOF) 에서 dynamic 유니버스 cross-section top-K(max(K_LIST)) 픽.

    반환 row: market, date(=진입일 D), score, regime, fold, rank(1=최고점), in_dynamic.
    각 날 top max(K) 만 모아두고, 집계 시 K 별로 rank<=K 슬라이스.
    """
    dates = np.sort(panel["date"].unique())
    folds = make_folds(dates, N_FOLDS, EMBARGO)
    kmax = max(K_LIST)
    picks = []
    for fi, (tr_dates, te_dates) in enumerate(folds):
        te = panel[panel["date"].isin(te_dates)].dropna(subset=["score", label]).copy()
        tr = panel[panel["date"].isin(tr_dates)].dropna(subset=["score", label])
        if len(tr) < 500 or len(te) < 200:
            continue
        te = te[te["in_dynamic"] == 1]
        for d, g in te.groupby("date"):
            topk = g.nlargest(kmax, "score").copy()
            topk["rank"] = np.arange(1, len(topk) + 1)
            topk["fold"] = fi
            picks.append(topk[["market", "date", "score", "regime", "fold", "rank"]])
    if not picks:
        return pd.DataFrame(columns=["market", "date", "score", "regime", "fold", "rank"])
    return pd.concat(picks, ignore_index=True)


# ============================================================================
# 2. 15m 실행 경로 로딩 ([09:15 D, 09:15 D+1))
# ============================================================================
def execution_start(date: pd.Timestamp | str) -> pd.Timestamp:
    """First executable 15m boundary after the 09:05 recommendation."""
    return pd.Timestamp(date).normalize() + EXECUTION_OFFSET


def load_paths(
    pick_pairs: pd.DataFrame,
    *,
    db_path: str | Path = M15_DB,
) -> dict:
    """Load only complete [D 09:15, D+1 09:15) execution paths."""
    assessments = assess_15m_windows(pick_pairs, db_path=db_path)
    return {
        key: assessment.bars
        for key, assessment in assessments.items()
        if assessment.path_complete
    }


# ============================================================================
# 3. 경로순서 청산 시뮬 (시간순 walk, leak 없음)
# ============================================================================
def simulate_path(bars: list, hard_sl, tp, trail) -> tuple[float, str]:
    """15m 봉 시간순 walk → (gross_return, outcome). 진입가=첫봉 open.
    우선순위(같은 봉 동시 → 보수): 1)hard SL 2)trailing SL 3)TP. 없으면 EOD close.
    """
    if not bars:
        return np.nan, "nodata"
    entry = bars[0][0]
    if not np.isfinite(entry) or entry <= 0:
        return np.nan, "nodata"
    sl_px = entry * (1 - hard_sl) if hard_sl is not None else None
    tp_px = entry * (1 + tp) if tp is not None else None
    running_high = entry
    for (o, h, low, c) in bars:
        if trail is not None and h > running_high:
            running_high = h
        trail_px = running_high * (1 - trail) if trail is not None else None
        if sl_px is not None and low <= sl_px:
            return -hard_sl, "sl"
        if trail_px is not None and low <= trail_px:
            return trail_px / entry - 1.0, "trail"
        if tp_px is not None and h >= tp_px:
            return tp, "tp"
    return bars[-1][3] / entry - 1.0, "eod"


# ============================================================================
# 4. 하방위험 우선 집계
# ============================================================================
LOSS_BINS = [-np.inf, -0.10, -0.05, 0.0, 0.05, np.inf]
LOSS_BIN_NAMES = ["lt_-10", "-10_to_-5", "-5_to_0", "0_to_+5", "gt_+5"]


def downside_metrics(trades: pd.DataFrame) -> dict:
    """trade DataFrame(date, net, outcome) → 하방위험 우선 지표.
    net = gross - 왕복비용. day-equal-weight 로 누적/MaxDD.
    """
    n = len(trades)
    if n == 0:
        return {}
    net = trades["net"].values
    # --- 하방위험 (1순위) ---
    p_deep = float((net < DEEP_LOSS_THRESH).mean())           # P(loss < -5%)
    k5 = max(1, int(np.ceil(0.05 * n)))
    worst5 = np.sort(net)[:k5]
    cvar95 = float(worst5.mean())                             # 최악 5% 평균
    worst = float(net.min())
    losses = net[net < 0]
    # day-equal-weight 누적/MaxDD
    daily = trades.groupby("date")["net"].mean().sort_index()
    downside_std = float(
        np.sqrt(np.mean(np.minimum(daily.to_numpy(dtype=float), 0.0) ** 2))
    )
    eq = (1 + daily).cumprod()
    peak = eq.cummax()
    mdd = float(((eq - peak) / peak).min())
    cum = float(eq.iloc[-1] - 1.0)
    trade_mu = float(net.mean())
    daily_mu = float(daily.mean())
    sortino = (
        float(daily_mu / downside_std * np.sqrt(365))
        if downside_std > 0
        else np.nan
    )
    # loss histogram
    hist = pd.cut(trades["net"], bins=LOSS_BINS, labels=LOSS_BIN_NAMES)
    hist_frac = hist.value_counts(normalize=True).reindex(LOSS_BIN_NAMES).fillna(0.0)
    oc = trades["outcome"]
    out = dict(
        n=int(n),
        p_loss_lt_5=p_deep,
        cvar95=cvar95,
        worst=worst,
        mdd=mdd,
        pct_stopped_sl=float((oc == "sl").mean()),
        pct_trail=float((oc == "trail").mean()),
        pct_tp=float((oc == "tp").mean()),
        pct_eod=float((oc == "eod").mean()),
        # 부수(포기하는 상승)
        net_mean=trade_mu,
        cum=cum,
        hit=float((net > 0).mean()),
        sortino=sortino,
        median=float(np.median(net)),
        avg_loss=float(losses.mean()) if len(losses) else 0.0,
        pct_loss=float((net < 0).mean()),
    )
    for bn in LOSS_BIN_NAMES:
        out[f"h_{bn}"] = float(hist_frac[bn])
    return out


# ============================================================================
# main
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-markets", type=int, default=None)
    args = ap.parse_args()
    OUT_DIR.mkdir(exist_ok=True)

    # --- leak-free panel + score + 유니버스 + regime (recall recommender 재사용) ---
    panel = build_panel(args.limit_markets)
    panel = add_score_and_universe(panel)
    panel = attach_btc_regime(panel)
    log.info("panel rows=%d markets=%d dates=%d",
             len(panel), panel["market"].nunique(), panel["date"].nunique())

    picks = collect_oof_picks(panel, label="lab_pump20")
    log.info("OOF picks (top-%d/day, dynamic uni): %d coin-days over %d days",
             max(K_LIST), len(picks), picks["date"].nunique())

    # --- 15m 경로 커버리지 ---
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
    log.info("15m window: %s -> %s  (start_date=%s)", m15_min, m15_max, m15_start)
    log.info("picks with 15m path: %d coin-days over %d days (regime dist: %s)",
             len(picks_in15), picks_in15["date"].nunique(),
             picks_in15["regime"].value_counts().to_dict())

    grid = list(product(HARD_SL, TP_CAP, TRAIL))
    log.info("SELECTION: K%s × policy-grid(%d: SL%d×TP%d×trail%d) = %d combos",
             K_LIST, len(grid), len(HARD_SL), len(TP_CAP), len(TRAIL),
             len(K_LIST) * len(grid))

    # --- 정책별 returns 캐시 (모든 픽에 대해 1회 시뮬, K 슬라이스는 사후) ---
    # outcome/gross 는 (market,date,policy) 함수 → 픽 rank 와 무관. 픽마다 1회.
    rows = []
    for K in K_LIST:
        sub = picks_in15[picks_in15["rank"] <= K].reset_index(drop=True)
        for (sl, tp, tr) in grid:
            recs = []
            for _, r in sub.iterrows():
                bars = bars_map[(r["market"], r["date"])]
                gross, outcome = simulate_path(bars, sl, tp, tr)
                if not np.isfinite(gross):
                    continue
                recs.append((r["date"], gross - ROUND_TRIP_COST, outcome))
            if not recs:
                continue
            td = pd.DataFrame(recs, columns=["date", "net", "outcome"])
            mt = downside_metrics(td)
            mt.update(K=K,
                      hard_sl=(sl if sl is not None else "none"),
                      tp=(tp if tp is not None else "none"),
                      trail=(tr if tr is not None else "none"))
            rows.append(mt)

    res = pd.DataFrame(rows)
    lead = ["K", "hard_sl", "tp", "trail", "n",
            "p_loss_lt_5", "cvar95", "worst", "mdd",
            "pct_stopped_sl", "pct_trail", "pct_tp", "pct_eod",
            "net_mean", "cum", "hit", "sortino", "median", "avg_loss", "pct_loss",
            "h_lt_-10", "h_-10_to_-5", "h_-5_to_0", "h_0_to_+5", "h_gt_+5"]
    res = res[lead]
    # 하방위험 우선 정렬: P(loss<-5%) ↑(작게), CVaR95 ↑(큰=덜손실)
    res = res.sort_values(["K", "p_loss_lt_5", "cvar95"], ascending=[True, True, False])
    res_path = OUT_DIR / "recommender_downside_exit_v1.csv"
    res.to_csv(res_path, index=False)
    log.info("wrote %s (%d rows)", res_path, len(res))

    # --- 핵심 비교: 하드 -5% SL (TP none, trail none) vs noSL+EOD ---
    cmp_rows = []
    for K in K_LIST:
        rk = res[res["K"] == K]
        nosl = rk[(rk["hard_sl"] == "none") & (rk["tp"] == "none") & (rk["trail"] == "none")]
        sl5 = rk[(rk["hard_sl"] == 0.05) & (rk["tp"] == "none") & (rk["trail"] == "none")]
        for tag, df in [("noSL_EOD", nosl), ("hardSL5_EOD", sl5)]:
            if len(df):
                rr = df.iloc[0].to_dict()
                rr["compare_tag"] = tag
                cmp_rows.append(rr)
    cmp_df = pd.DataFrame(cmp_rows)
    cmp_df.to_csv(OUT_DIR / "recommender_downside_core_compare_v1.csv", index=False)

    # --- 최악 trade(붕괴) 캡 확인: noSL EOD 기준 worst 픽들 ---
    deep_rows = []
    sub3 = picks_in15[picks_in15["rank"] <= max(K_LIST)].drop_duplicates(
        subset=["market", "date"]).reset_index(drop=True)
    for _, r in sub3.iterrows():
        bars = bars_map[(r["market"], r["date"])]
        g_eod, _ = simulate_path(bars, None, None, None)
        g_sl5, oc5 = simulate_path(bars, 0.05, None, None)
        if np.isfinite(g_eod):
            deep_rows.append(dict(market=r["market"], date=str(r["date"]),
                                  regime=r["regime"], rank=int(r["rank"]),
                                  net_eod=g_eod - ROUND_TRIP_COST,
                                  net_sl5=g_sl5 - ROUND_TRIP_COST, sl5_outcome=oc5))
    deep_df = pd.DataFrame(deep_rows).sort_values("net_eod").head(25)
    deep_df.to_csv(OUT_DIR / "recommender_downside_worst_trades_v1.csv", index=False)

    cov = dict(
        n_picks_top_kmax=int(len(picks_in15)),
        n_days=int(picks_in15["date"].nunique()),
        kmax=int(max(K_LIST)),
        m15_window=[m15_min, m15_max],
        regime_dist={k: int(v) for k, v in picks_in15["regime"].value_counts().items()},
        n_combos=int(len(K_LIST) * len(grid)),
    )
    (OUT_DIR / "recommender_downside_coverage_v1.json").write_text(json.dumps(cov, indent=2))

    # ---- 콘솔 요약 ----
    log.info("\n===== CORE COMPARE: hard -5%% SL vs noSL+EOD (TP/trail none) =====")
    for _, r in cmp_df.iterrows():
        log.info("  K=%d %-12s | P(loss<-5%%)=%.3f CVaR95=%+.4f worst=%+.4f MaxDD=%.3f "
                 "%%SL=%.2f | net/trade=%+.4f cum=%+.3f hit=%.2f Sortino=%s n=%d",
                 int(r["K"]), r["compare_tag"], r["p_loss_lt_5"], r["cvar95"],
                 r["worst"], r["mdd"], r["pct_stopped_sl"], r["net_mean"], r["cum"],
                 r["hit"], f"{r['sortino']:+.2f}" if pd.notna(r["sortino"]) else "nan",
                 int(r["n"]))

    log.info("\n===== TIGHTEST DOWNSIDE policies (per K, by P(loss<-5%%) then CVaR95) =====")
    for K in K_LIST:
        log.info("--- K=%d ---", K)
        for _, r in res[res["K"] == K].head(6).iterrows():
            log.info("  SL=%-4s TP=%-4s trail=%-4s | P(loss<-5%%)=%.3f CVaR95=%+.4f "
                     "worst=%+.4f MaxDD=%.3f %%SL=%.2f | hist[<-10:%.2f -10/-5:%.2f "
                     "-5/0:%.2f 0/+5:%.2f >+5:%.2f] | net=%+.4f cum=%+.3f hit=%.2f",
                     str(r["hard_sl"]), str(r["tp"]), str(r["trail"]),
                     r["p_loss_lt_5"], r["cvar95"], r["worst"], r["mdd"],
                     r["pct_stopped_sl"], r["h_lt_-10"], r["h_-10_to_-5"],
                     r["h_-5_to_0"], r["h_0_to_+5"], r["h_gt_+5"],
                     r["net_mean"], r["cum"], r["hit"])

    log.info("\n===== WORST 8 trades (noSL EOD) — does -5%% SL cap them? =====")
    for _, r in deep_df.head(8).iterrows():
        log.info("  %-14s %s rank%d %-13s | EOD=%+.3f  SL5=%+.3f (%s)",
                 r["market"], r["date"], int(r["rank"]), r["regime"],
                 r["net_eod"], r["net_sl5"], r["sl5_outcome"])


if __name__ == "__main__":
    main()

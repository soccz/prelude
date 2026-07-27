"""Downside-aware filter for pump-recommendation picks (Angle B).

사용자 선호 (명시): "상승 보장보다 하락 위험 적은 게 더 좋다. 평균 -5% 정도 손실은 괜찮다."
→ 평가 1순위 = 하방위험 (per-trade 손실분포, P(loss<-5%), CVaR95, MaxDD, worst,
  Sortino). 평균/누적은 '포기하는 상승'으로 부수 보고. 하드 -5% SL 이 이제 바람직.

단일 질문:
  추천기(recommendation_scorer_v1, cal_xgb top-K)가 뽑은 픽 중, 다음날 펌프 후
  **덜 무너지는(낮은 하방위험)** 코인을 진입 D-1 정보만으로 사전에 가릴 수 있나?
  그 필터로 추천을 좁히면 precision(급등 적중)은 유지하면서 deep-loss 빈도/CVaR 가
  줄어드나?

접근:
  1. recommendation_scorer 의 leak-free WF OOS 를 그대로 재현(cal_xgb daily top-K 픽).
     → 이게 '필터 전' 추천 모집단.
  2. 각 픽의 다음날(=진입일 D) **하방 경로**를 15m 봉 시간순 walk 로 해소:
       - open→intraday min  (장중 최저, 봉내 SL 닿는 깊이)
       - open→close         (EOD)
       - 하드 -5% SL 적용 net (각 손실을 -5% 로 캡, 사용자 선호)
     deep-dump 정의: intraday_min <= -10%  OR  close <= -5%.
  3. D-1 선행조건으로 deep-dump vs sustain 분리 (train-only 로 cut 학습 → OOS 적용,
     selection bias 방어). 후보: 과열/소진(ret_7d/up_streak 극단), 유동성(qv_rank/log_qv),
     regime(bear_volatile), ATR 과도(atr_pct_14 / atr_xs_decile).
  4. dump_risk 점수(train logit, D-1 feature) 를 학습 → downside-aware 재랭크
       score_DS = pump_prob - lambda * dump_risk
     필터(상위 dump_risk 컷) / 다운랭크 후 top-K 의 하방을 필터 전과 정량 비교.

★ LEAK 방어 (same-day leak 2번 전적):
  - 진입선택 = recommendation_scorer 의 cal_xgb (모든 feature D-1 shift, calibration
    train-only). 그대로 import 재사용.
  - dump_risk 라벨/cut/logit = **train fold(과거) 에서만 fit**, OOS test 에 적용만.
  - 경로 = day-D [09:15 D, 09:15 D+1) 15m 봉 시간순 단일패스. 미래봉 미리보기 X.
    봉내 SL·TP 동시 → SL 먼저(보수). intraday_min/close 는 in-trade outcome (leak X).
  - 라벨/유니버스/사이징/알림 변경 X — 이건 검증 (사용자 컨펌 영역 안 건드림).

거래비용 0.15% 왕복, spot-only. 사이징중립 = day-equal-weight 집계(per-trade Sharpe
는 day clustering 과장 주의 → 하방지표는 per-trade 분포로 본다).

사용:
    python scripts/downside_aware_recommender_v1.py
    python scripts/downside_aware_recommender_v1.py --limit-markets 80   # 개발
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.database import connect_readonly, list_markets, load_candles  # noqa: E402
from data.market_universe import is_excluded_signal_market  # noqa: E402
from scripts.univariate_precursor_lift_v1 import (  # noqa: E402
    build_market_features,
    add_cross_sectional,
)
from scripts.regime_split_precursor_v1 import attach_btc_regime  # noqa: E402
from scripts.recommendation_scorer_v1 import (  # noqa: E402
    apply_dynamic_universe,
    walk_forward_scores,
    PRECURSOR_FEATURES,
    MAIN_LABEL,
)
from ledger.path_quality import assess_15m_window  # noqa: E402
from scripts.recommender_downside_exit_v1 import execution_start  # noqa: E402

D1_DB = "data/upbit_d1.db"
M15_DB = "data/upbit_15m.db"
OUT_DIR = Path("output")
EPS = 1e-12
ROUND_TRIP_COST = 0.0015
BARS_PER_HOUR = 4

# 추천 모집단: scorer 의 best = cal_xgb (OOS K=3 기준 best, 기존 산출과 일치)
SCORE_COL = "cal_xgb"
TOP_K = 5                         # 픽 모집단 (K=3/5 둘 다 본다, 기본 5)
HARD_SL = 0.05                    # 사용자 선호: 하드 -5% SL (손실 캡)

# deep-dump 정의 (사용자 제시): 장중 -10% 이하 OR 종가 -5% 이하
DEEP_INTRADAY = -0.10
DEEP_CLOSE = -0.05

# dump_risk 재랭크 lambda 그리드 (placeholder — OOS 하방개선 비례로 선택)
LAMBDA_GRID = [0.0, 0.5, 1.0, 1.5, 2.0]
# dump_risk 상위 컷 필터 그리드 (이 분위 이상이면 추천에서 제외)
DUMP_CUT_GRID = [None, 0.5, 0.667, 0.75, 0.8]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("ds_aware")


# ============================================================================
# 1. Panel + scorer OOS (leak-free, 그대로 재사용)
# ============================================================================
def build_panel(limit_markets):
    markets = [
        market
        for market in list_markets(D1_DB)
        if not is_excluded_signal_market(str(market))
    ]
    if limit_markets:
        markets = markets[:limit_markets]
    log.info("loading %d markets", len(markets))
    frames = []
    for i, m in enumerate(markets):
        df = load_candles(D1_DB, m)
        if df is None or len(df) < 70:
            continue
        df = df.copy()
        df["market"] = m
        frames.append(build_market_features(df))
        if (i + 1) % 60 == 0:
            log.info("  %d/%d", i + 1, len(markets))
    panel = pd.concat(frames, ignore_index=True)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel["date"] = panel["timestamp"].dt.date
    panel = panel.sort_values(["date", "market"]).reset_index(drop=True)
    return panel


# ============================================================================
# 2. 15m 하방 경로: open→intraday min, open→close, 하드 -5% SL net
# ============================================================================
def load_15m_window(conn, market, dt):
    assessment = assess_15m_window(
        market,
        execution_start(dt),
        connection=conn,
    )
    return assessment.bars if assessment.path_complete else []


def path_downside(bars, hard_sl):
    """15m 봉 시간순 walk → 하방 메트릭.

    반환 dict:
      open_min : open→intraday min (장중 최저 수익률, <=0 보통)
      open_close: open→close (EOD)
      sl_net   : 하드 -sl SL 적용 net (봉내 SL 닿으면 -sl, 아니면 EOD), 비용 차감
      eod_net  : SL 없는 EOD net, 비용 차감
      stopped  : SL 닿았나 (1/0)
      hit10    : 장중 +10% 닿았나 (펌프 적중, upside 참고)
    """
    if not bars:
        return None
    entry = bars[0][0]
    if not np.isfinite(entry) or entry <= 0:
        return None
    sl_px = entry * (1 - hard_sl)
    running_min = entry
    running_high = entry
    stopped = False
    sl_gross = None
    for o, h, low, c in bars:
        if low < running_min:
            running_min = low
        if h > running_high:
            running_high = h
        if (not stopped) and low <= sl_px:
            stopped = True
            sl_gross = -hard_sl   # 봉내 SL 우선 (보수)
    last_c = bars[-1][3]
    open_min = running_min / entry - 1.0
    open_close = last_c / entry - 1.0
    open_high = running_high / entry - 1.0
    if sl_gross is None:
        sl_gross = open_close
    return dict(
        open_min=float(open_min),
        open_close=float(open_close),
        open_high=float(open_high),
        sl_net=float(sl_gross - ROUND_TRIP_COST),
        eod_net=float(open_close - ROUND_TRIP_COST),
        stopped=int(stopped),
        hit10=int(open_high >= 0.10),
    )


def attach_paths(picks: pd.DataFrame, m15_start) -> pd.DataFrame:
    """픽들에 15m 하방경로 부착. 15m 윈도우 내 픽만."""
    picks = picks[picks["date"] >= m15_start].copy()
    recs = []
    cache = {}
    with connect_readonly(M15_DB) as conn:
        for _, r in picks.iterrows():
            key = (r["market"], r["date"])
            if key not in cache:
                cache[key] = load_15m_window(conn, r["market"], r["date"])
            dd = path_downside(cache[key], HARD_SL)
            if dd is None:
                continue
            rec = r.to_dict()
            rec.update(dd)
            recs.append(rec)
    return pd.DataFrame(recs)


# ============================================================================
# 3. 하방 지표 (per-trade 분포 우선 — 사용자 선호)
# ============================================================================
def downside_block(df: pd.DataFrame, net_col: str) -> dict:
    """net_col(=실현손익) 의 하방 지표 + day-equal-weight cum/MaxDD."""
    if len(df) == 0:
        return {}
    x = df[net_col].dropna().values
    n = len(x)
    losses = x[x < 0]
    # 사용자 선호: "평균 -5% 정도 손실은 괜찮다" → 의미있는 deep-loss 선은 -5% SL 캡보다
    # '더 깊은' 손실. 하드SL net 은 -5% gross - 0.15% 비용 = -5.15% 에 캡되므로:
    #   p_deep5      : net < -5% (SL trade 는 -5.15% 라 여기 포함 — 'SL 또는 더깊음')
    #   p_beyond_sl  : net < -5.5% (= SL 캡을 '넘어선' 손실 빈도. 하드SL이면 ~0 이어야)
    p_deep5 = float((x < -0.05).mean())
    p_beyond_sl = float((x < -0.055).mean())
    # CVaR95 = 최악 5% 평균손실 (per-trade)
    q05 = np.quantile(x, 0.05)
    tail = x[x <= q05]
    cvar95 = float(tail.mean()) if len(tail) else float(q05)
    worst = float(x.min())
    # Sortino (하방편차만, periods=365, day-equal-weight daily)
    daily = df.groupby("date")[net_col].mean().sort_index()
    mu = daily.mean()
    downside = daily[daily < 0]
    dd_std = downside.std()
    sortino = float(mu / dd_std * np.sqrt(365)) if dd_std and dd_std > 0 else np.nan
    sig = daily.std()
    sharpe = float(mu / sig * np.sqrt(365)) if sig and sig > 0 else np.nan
    eq = (1 + daily).cumprod()
    peak = eq.cummax()
    mdd = float(((eq - peak) / peak).min())
    cum = float(eq.iloc[-1] - 1.0)
    return dict(
        n=int(n),
        net_mean=float(x.mean()),
        net_median=float(np.median(x)),
        hit=float((x > 0).mean()),
        p_loss_lt_5=p_deep5,
        p_beyond_sl=p_beyond_sl,
        cvar95=cvar95,
        worst=worst,
        loss_rate=float((x < 0).mean()),
        avg_loss=float(losses.mean()) if len(losses) else 0.0,
        mdd=mdd,
        cum=cum,
        sharpe=sharpe,
        sortino=sortino,
        stop_rate=float(df["stopped"].mean()) if "stopped" in df else np.nan,
        hit10_rate=float(df["hit10"].mean()) if "hit10" in df else np.nan,
    )


# ============================================================================
# 4. D-1 선행조건 분석: deep-dump vs sustain 분리 (전체 픽 모집단에서 진단)
# ============================================================================
SEPARATOR_FEATS = [
    # 과열/소진
    "f_ret_7d", "f_ret_3d", "f_ret_14d", "f_up_streak", "f_rsi_14",
    "f_ret7d_xs_decile", "f_ret3d_xs_decile",
    # 유동성
    "f_qv_rank", "f_log_qv", "f_qv_rank_pct",
    # 변동성/ATR 과도
    "f_atr_pct_14", "f_atr_xs_decile", "f_rv_7d", "f_rv_21d",
    "f_qv_surge_7d", "f_qv_surge_30d",
    # 위치
    "f_pos_in_20d_range", "f_dist_from_20d_high", "f_bounce_off_7d_low",
    "f_drawdown_7d",
]


def separator_diagnostics(picks: pd.DataFrame) -> pd.DataFrame:
    """deep-dump 픽 vs sustain 픽의 D-1 feature 평균/중앙 비교 (전 표본 진단).
    selection bias 주의: 이건 '어디를 볼지'의 EDA. cut 학습은 train-only(아래)."""
    picks = picks.copy()
    picks["deep_dump"] = ((picks["open_min"] <= DEEP_INTRADAY) |
                          (picks["open_close"] <= DEEP_CLOSE)).astype(int)
    rows = []
    dd = picks[picks["deep_dump"] == 1]
    su = picks[picks["deep_dump"] == 0]
    for f in SEPARATOR_FEATS:
        if f not in picks.columns:
            continue
        a = dd[f].dropna()
        b = su[f].dropna()
        if len(a) < 5 or len(b) < 5:
            continue
        # AUC (deep_dump 를 양성으로 보는 단변량 분리력, dir 무관)
        from scipy.stats import mannwhitneyu
        try:
            u, _ = mannwhitneyu(a, b, alternative="two-sided")
            auc = u / (len(a) * len(b))
        except Exception:
            auc = np.nan
        rows.append(dict(
            feature=f,
            deep_mean=float(a.mean()), sust_mean=float(b.mean()),
            deep_med=float(a.median()), sust_med=float(b.median()),
            auc_dump=float(auc),
            sep=float(abs(auc - 0.5) * 2),  # |2*AUC-1| = 분리력
        ))
    out = pd.DataFrame(rows).sort_values("sep", ascending=False)
    return out, picks


# ============================================================================
# 5. dump_risk 모델 (train-only logit) → OOS dump_risk 점수
#   라벨 = deep_dump (intraday<=-10% OR close<=-5%). feature = D-1 SEPARATOR_FEATS.
#   per-fold: train fold 픽에서 fit → test fold 픽에 적용 (OOS). leak X.
# ============================================================================
def fit_dump_risk_oos(picks: pd.DataFrame, fold_col="fold") -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    feats = [f for f in SEPARATOR_FEATS if f in picks.columns]
    picks = picks.copy()
    picks["deep_dump"] = ((picks["open_min"] <= DEEP_INTRADAY) |
                          (picks["open_close"] <= DEEP_CLOSE)).astype(int)
    folds = sorted(picks[fold_col].dropna().unique())
    out = []
    for k in folds:
        te = picks[picks[fold_col] == k]
        tr = picks[picks[fold_col] < k]            # expanding: 과거 fold 만 train
        if len(tr) < 60 or tr["deep_dump"].sum() < 8 or te.empty:
            # train 부족 → dump_risk = 0.5 (정보없음, 중립)
            t = te.copy()
            t["dump_risk"] = 0.5
            out.append(t)
            continue
        Xtr = tr[feats].replace([np.inf, -np.inf], np.nan)
        med = Xtr.median()
        Xtr = Xtr.fillna(med).values
        ytr = tr["deep_dump"].values
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        clf = LogisticRegression(max_iter=500, class_weight="balanced", C=0.5)
        clf.fit(Xtr_s, ytr)
        Xte = te[feats].replace([np.inf, -np.inf], np.nan).fillna(med).values
        risk = clf.predict_proba(sc.transform(Xte))[:, 1]
        t = te.copy()
        t["dump_risk"] = risk
        out.append(t)
    res = pd.concat(out, ignore_index=True)
    return res, feats


# ============================================================================
# 6. 재랭크 / 필터 적용 후 하방 비교 (필터 전 vs 후)
# ============================================================================
def _combo_block(top: pd.DataFrame) -> dict:
    """top-K 픽의 하방 지표. SL net(사용자 정책) + no-SL(딥테일 가시화) 둘 다.

    하드 -5% SL 하에선 CVaR95/worst/P(loss<-5%) 가 SL 캡(-5.15%)에 묶여 정책간
    구분이 안 됨 → SL 정책의 discriminating downside 는 stop_rate/loss_rate/cum.
    필터가 '얼마나 깊은 덤프를 피하나'는 no-SL(open_min/eod_net) 에서만 보임:
      no_cvar95/no_worst (eod_net), p_min_lt_10 (장중 -10% 닿은 빈도=deep-dump 회피력).
    """
    blk = downside_block(top, "sl_net")            # SL 정책 (사용자 선호)
    # no-SL deep-tail 가시화
    no = downside_block(top, "eod_net")
    blk["no_cvar95"] = no.get("cvar95", np.nan)    # SL 없을 때 최악5% (EOD)
    blk["no_worst"] = no.get("worst", np.nan)
    blk["no_cum"] = no.get("cum", np.nan)
    # 장중 -10% 이하 닿은 빈도 (deep intraday dump 회피력)
    om = top["open_min"].dropna()
    blk["p_min_lt_10"] = float((om <= DEEP_INTRADAY).mean()) if len(om) else np.nan
    blk["p_close_lt_5"] = float((top["open_close"].dropna() <= DEEP_CLOSE).mean()) \
        if "open_close" in top else np.nan
    blk["deep_dump_rate"] = float(top["deep_dump"].mean()) if "deep_dump" in top else np.nan
    return blk


def rerank_and_compare(scored: pd.DataFrame, k: int) -> pd.DataFrame:
    """각 (lambda, dump_cut) 조합으로 daily 재랭크/필터 후 top-K 하방 비교.
    하드 -5% SL net 정책 기준(사용자 선호) + no-SL deep-tail 가시화."""
    rows = []
    # baseline = pump_prob 만으로 top-K (필터 전)
    base = (scored.sort_values(["date", SCORE_COL], ascending=[True, False])
            .groupby("date").head(k))
    b = _combo_block(base)
    b.update(policy="baseline_topK", k=k, lam="-", dump_cut="-",
             prec_pump=float(base[MAIN_LABEL].mean()))
    rows.append(b)
    for lam in LAMBDA_GRID:
        for cut in DUMP_CUT_GRID:
            d = scored.copy()
            d["score_ds"] = d[SCORE_COL] - lam * d["dump_risk"]
            if cut is not None:
                thr = d["dump_risk"].quantile(cut)
                d = d[d["dump_risk"] <= thr]
            if d.empty:
                continue
            top = (d.sort_values(["date", "score_ds"], ascending=[True, False])
                   .groupby("date").head(k))
            if len(top) < 20:
                continue
            blk = _combo_block(top)
            blk.update(policy="ds_rerank", k=k, lam=lam,
                       dump_cut=(cut if cut is not None else "none"),
                       prec_pump=float(top[MAIN_LABEL].mean()))
            rows.append(blk)
    return pd.DataFrame(rows)


# ============================================================================
# main
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-markets", type=int, default=None)
    ap.add_argument("--universe-top", type=int, default=100)
    ap.add_argument("--surge-cut", type=float, default=3.0)
    ap.add_argument("--n-folds", type=int, default=6)
    args = ap.parse_args()
    OUT_DIR.mkdir(exist_ok=True)

    panel = build_panel(args.limit_markets)
    panel = add_cross_sectional(panel)
    panel = attach_btc_regime(panel)
    feats = [f for f in PRECURSOR_FEATURES if f in panel.columns]
    uni = apply_dynamic_universe(panel, args.universe_top, args.surge_cut)
    log.info("panel rows=%d uni rows=%d markets=%d dates=%d",
             len(panel), len(uni), panel["market"].nunique(), panel["date"].nunique())

    # ---- leak-free WF OOS (scorer 와 동일) ----
    log.info("=== walk-forward OOS scoring (cal_xgb, reuse scorer) ===")
    oos = walk_forward_scores(uni, feats, MAIN_LABEL, n_folds=args.n_folds)
    if oos.empty or SCORE_COL not in oos.columns:
        log.error("no OOS / no %s — abort", SCORE_COL)
        sys.exit(1)
    oos = oos.dropna(subset=[SCORE_COL]).copy()
    log.info("OOS rows=%d dates %s..%s", len(oos), oos["date"].min(), oos["date"].max())

    # ---- 픽 모집단: daily top-K (필터 전 추천) ----
    # K 큰 쪽으로 픽 모은 뒤(top-8) 경로 부착 → 재랭크 단계에서 K 별 head
    PICK_POOL = 8
    pool = (oos.sort_values(["date", SCORE_COL], ascending=[True, False])
            .groupby("date").head(PICK_POOL).reset_index(drop=True))
    log.info("pick pool (daily top-%d): rows=%d", PICK_POOL, len(pool))

    # ---- 15m 하방 경로 부착 ----
    with connect_readonly(M15_DB) as conn:
        m15_min = conn.execute(
            "SELECT MIN(timestamp) FROM candles"
        ).fetchone()[0]
    m15_start = pd.Timestamp(m15_min).date()
    picks = attach_paths(pool, m15_start)
    n_total = pool[pool["date"] >= m15_start].shape[0]
    log.info("15m coverage: window>=%s ; pool-in-window=%d ; with-path=%d (%.1f%%)",
             m15_start, n_total, len(picks),
             100 * len(picks) / max(n_total, 1))
    regime_dist = picks["regime"].value_counts().to_dict()
    log.info("pick regime dist: %s", regime_dist)

    # ---- 진단: deep-dump vs sustain D-1 선행조건 ----
    sep, picks = separator_diagnostics(picks)
    sep.to_csv(OUT_DIR / "downside_aware_separators_v1.csv", index=False)
    log.info("\n===== D-1 separators: deep-dump vs sustain (|2AUC-1| desc) =====")
    log.info("  deep_dump n=%d / sustain n=%d (deep = intraday<=%.0f%% OR close<=%.0f%%)",
             int(picks["deep_dump"].sum()), int((1 - picks["deep_dump"]).sum()),
             100 * DEEP_INTRADAY, 100 * DEEP_CLOSE)
    for _, r in sep.head(12).iterrows():
        log.info("  %-22s sep=%.3f auc=%.3f deep_med=%+.3f sust_med=%+.3f",
                 r["feature"], r["sep"], r["auc_dump"], r["deep_med"], r["sust_med"])

    # ---- dump_risk OOS (train-only logit) ----
    scored, used_feats = fit_dump_risk_oos(picks)
    log.info("\ndump_risk logit feats=%d ; OOS scored rows=%d ; "
             "dump_risk mean=%.3f (deep base rate=%.3f)",
             len(used_feats), len(scored), scored["dump_risk"].mean(),
             scored["deep_dump"].mean())

    # dump_risk 자체의 분리력 (OOS): risk 상위 vs 하위 deep-dump rate
    sc2 = scored.dropna(subset=["dump_risk"]).copy()
    hi = sc2[sc2["dump_risk"] >= sc2["dump_risk"].quantile(0.667)]
    lo = sc2[sc2["dump_risk"] <= sc2["dump_risk"].quantile(0.333)]
    log.info("dump_risk OOS separation: top-tercile deep=%.3f vs bottom-tercile deep=%.3f",
             hi["deep_dump"].mean(), lo["deep_dump"].mean())

    # ---- 필터 전/후 하방 비교 (K=3,5) ----
    cmp_all = []
    for k in (3, 5):
        cmp_all.append(rerank_and_compare(scored, k))
    cmp = pd.concat(cmp_all, ignore_index=True)
    cmp_cols = ["policy", "k", "lam", "dump_cut", "n", "prec_pump",
                "hit", "hit10_rate",
                # SL 정책 하방 (사용자 선호 정책)
                "stop_rate", "loss_rate", "p_loss_lt_5", "p_beyond_sl",
                "cvar95", "worst", "net_mean", "net_median", "mdd", "cum",
                "sharpe", "sortino",
                # no-SL deep-tail 가시화 (필터의 덤프회피력)
                "deep_dump_rate", "p_min_lt_10", "p_close_lt_5",
                "no_cvar95", "no_worst", "no_cum"]
    cmp = cmp[[c for c in cmp_cols if c in cmp.columns]]
    cmp.to_csv(OUT_DIR / "downside_aware_compare_v1.csv", index=False)
    log.info("wrote downside_aware_compare_v1.csv (%d rows)", len(cmp))

    # ---- 픽 레벨 경로 산출 (감사용) ----
    keep_cols = ["market", "date", "fold", "regime", SCORE_COL, MAIN_LABEL,
                 "dump_risk", "deep_dump", "open_min", "open_close", "open_high",
                 "sl_net", "eod_net", "stopped", "hit10"] + used_feats
    keep_cols = [c for c in keep_cols if c in scored.columns]
    scored[keep_cols].to_csv(OUT_DIR / "downside_aware_picks_v1.csv", index=False)

    # ---- 콘솔: 필터 전/후 하방 (K=3,5) ----
    # 하드 -5% SL 하에선 CVaR95/worst 가 SL캡에 묶임 → 정책간 구분은
    #   (a) deep_dump_rate / p_min_lt_10 (덤프 빈도, no-SL 진실)
    #   (b) stop_rate (SL 발동 빈도)
    #   (c) no_cvar95 (SL 껐을 때 최악5%)  로 본다.
    for k in (3, 5):
        log.info("\n===== DOWNSIDE: filter before/after (K=%d) =====", k)
        kk = cmp[cmp["k"] == k]
        base = kk[kk["policy"] == "baseline_topK"]
        if not base.empty:
            r = base.iloc[0]
            log.info("  [BASELINE] n=%d prec_pump=%.3f hit=%.3f hit10=%.3f | "
                     "deep_dump=%.3f p(min<-10%%)=%.3f stop=%.3f | "
                     "no_CVaR95=%.3f no_worst=%.3f | SL: net=%+.4f cum=%+.3f Sortino=%.2f",
                     int(r["n"]), r["prec_pump"], r["hit"], r["hit10_rate"],
                     r["deep_dump_rate"], r["p_min_lt_10"], r["stop_rate"],
                     r["no_cvar95"], r["no_worst"], r["net_mean"], r["cum"], r["sortino"])
        cand = kk[kk["policy"] == "ds_rerank"].copy()
        if base.empty or cand.empty:
            continue
        bprec = base.iloc[0]["prec_pump"]
        # 하방 개선 best: deep_dump_rate 최소, prec_pump 유지(baseline-0.02 이상)
        cand = cand[cand["prec_pump"] >= bprec - 0.02]
        cand = cand.sort_values("deep_dump_rate", ascending=True)
        log.info("  --- DS-rerank (prec_pump >= baseline-0.02), lowest deep_dump_rate first ---")
        for _, r in cand.head(8).iterrows():
            log.info("  lam=%-4s cut=%-5s n=%d prec_pump=%.3f hit=%.3f hit10=%.3f | "
                     "deep_dump=%.3f p(min<-10%%)=%.3f stop=%.3f | "
                     "no_CVaR95=%.3f no_worst=%.3f | SL: net=%+.4f cum=%+.3f Sortino=%.2f",
                     str(r["lam"]), str(r["dump_cut"]), int(r["n"]), r["prec_pump"],
                     r["hit"], r["hit10_rate"], r["deep_dump_rate"], r["p_min_lt_10"],
                     r["stop_rate"], r["no_cvar95"], r["no_worst"], r["net_mean"],
                     r["cum"], r["sortino"])

    # ---- 참고: SL 없을 때 baseline 하방 (사용자 선호 대비 위해) ----
    base_pool = (scored.sort_values(["date", SCORE_COL], ascending=[True, False])
                 .groupby("date").head(5))
    noSL = downside_block(base_pool, "eod_net")
    log.info("\n===== REFERENCE: baseline K=5, NO hard SL (EOD net) =====")
    log.info("  P(loss<-5%%)=%.3f CVaR95=%.3f worst=%.3f loss_rate=%.3f MaxDD=%.3f "
             "net_mean=%+.4f cum=%+.3f",
             noSL["p_loss_lt_5"], noSL["cvar95"], noSL["worst"], noSL["loss_rate"],
             noSL["mdd"], noSL["net_mean"], noSL["cum"])
    log.info("DONE.")


if __name__ == "__main__":
    main()

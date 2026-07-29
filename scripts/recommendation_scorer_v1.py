"""Daily pump-probability recommendation scorer (Angle A — 추천 레이더).

목표 (사용자 명시 재설정): 사람이 판단해 매매하는 **추천 레이더**. 1순위 = 추천 품질
(급등을 잘 짚기). 검증된 leak-free 선행패턴을 결합해 코인별 **일일 급등확률
(0~1, calibrated)** 을 매기고 top-K 를 뽑는다. precision@K / recall / lift 가
1순위 지표. net PnL/Sharpe 는 사후 참고만 (사용자가 직접 청산).

펌프 라벨 (추천 적중 기준, 진입일 D):
  pump20 = (high_D / open_D - 1) >= 0.20   (메인)
  pump15 = (high_D / open_D - 1) >= 0.15   (보조)
  → D-1 까지 정보로 D 를 예측 → leak-free.

세 스코어러 비교 (OOS 로 더 나은 쪽 채택):
  (a) composite : 검증된 선행패턴들의 cross-sectional rank 가중합 (해석 가능, 학습 X)
  (b) logit     : leak-free 로지스틱 회귀 P(pump20 | D-1 features)
  (c) xgb       : leak-free XGBoost P(pump20 | D-1 features)

★ LEAK 방어 (이 프로젝트 same-day leak 2번 전적):
  - 모든 feature 는 build_market_features 에서 market 별 .shift(1) → day D row 는
    D-1 까지 정보. cross-sectional rank 입력도 D-1 feature.
  - 라벨 pump20/pump15 는 day D 의 open/high 만 (타겟, 미래). feature 에 안 섞임.
  - BTC regime 은 D-1 시점 (attach_btc_regime, 무관 시계열 1일 shift).
  - 모델/임계/calibration 은 **expanding train (과거) 에서만 fit**, test fold (미래)
    에는 적용만. calibration bin 도 train 에서 fit → OOS 에서 historical hit 매핑.
    (rare-event raw prob 과신 금지 — 이 프로젝트 +20% tail 90% pred vs 11.6% actual 전적.)
  - 유니버스: 동적 (D-1 qv top-N OR qv_surge_7d >= 3x). 정적 top100 대비 펌프 recall ↑
    (직전 발굴 +7.9pp). 유니버스 컷도 D-1 정보.

산출 (output/):
  - recommendation_scorer_oos_metrics_v1.csv : 스코어러×K×regime OOS precision@K/recall/lift
  - recommendation_scorer_recent_picks_v1.csv: 최근 N 거래일 top-K 추천 + 실제 펌프 여부
  - recommendation_scorer_today_v1.csv       : 최신 가용일 top 후보 + calibrated 급등확률
  - recommendation_scorer_calibration_v1.csv : 확률 bucket calibration (예측 vs 실제 hit)

사용:
    python scripts/recommendation_scorer_v1.py
    python scripts/recommendation_scorer_v1.py --limit-markets 60   # 개발용
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.database import list_markets, load_candles  # noqa: E402
from data.market_universe import is_excluded_signal_market  # noqa: E402
# 재사용: 동일한 leak-free feature 빌더 (univariate_precursor_lift_v1 검증됨)
from scripts.univariate_precursor_lift_v1 import (  # noqa: E402
    build_market_features,
    add_cross_sectional,
)
from scripts.regime_split_precursor_v1 import attach_btc_regime  # noqa: E402

DB_PATH = "data/upbit_d1.db"
OUT_DIR = Path("output")
EPS = 1e-12

log = logging.getLogger("rec_scorer")

REGIMES = ["bull_quiet", "bull_volatile", "bear_quiet", "bear_volatile"]
MAIN_LABEL = "lab_pump20"
AUX_LABEL = "lab_pump15"

# 검증된 선행패턴 (univariate_precursor_lift_v1.csv 의 OOS lift 상위, dir=high).
# composite 스코어러 + 모델 feature 공통 base set. 전부 D-1 feature.
PRECURSOR_FEATURES = [
    "f_qv_surge_30d", "f_qv_surge_7d", "f_qv_ma7_vs_ma30", "f_vol_surge_7d",
    "f_bounce_off_7d_low", "f_ret_1d", "f_ret_3d", "f_ret_7d", "f_ret_14d",
    "f_roc_3d", "f_roc_7d", "f_atr_pct_14", "f_rv_7d", "f_rv_21d",
    "f_log_qv", "f_pos_in_20d_range", "f_drawdown_7d", "f_rsi_14",
    "f_up_streak",
    # cross-sectional deciles (당일 panel 내, D-1 입력)
    "f_ret3d_xs_decile", "f_ret7d_xs_decile", "f_atr_xs_decile",
    "f_qvsurge_xs_decile", "f_qv_rank_pct",
]

# composite 가중합에 쓸 핵심 cross-sectional rank (검증된 dir=high, 높을수록 펌프 쪽).
# 가중치는 OOS lift 비례 (placeholder — 데이터가 더 좋은 값 주면 변경).
COMPOSITE_RANKS = {
    "f_qv_surge_30d": 3.35,
    "f_bounce_off_7d_low": 3.18,
    "f_ret_3d": 3.03,
    "f_qv_surge_7d": 2.88,
    "f_ret_7d": 2.95,
    "f_vol_surge_7d": 2.79,
    "f_atr_pct_14": 2.60,
    "f_rv_7d": 2.60,
    "f_log_qv": 2.71,
}


# ============================================================================
# 1. Panel build (D-1 features) + day-D 라벨 + regime + 동적 유니버스
# ============================================================================
def build_panel(limit_markets: int | None) -> pd.DataFrame:
    markets = [
        market
        for market in list_markets(DB_PATH)
        if not is_excluded_signal_market(str(market))
    ]
    if limit_markets:
        markets = markets[:limit_markets]
    log.info("loading %d markets", len(markets))
    frames = []
    for i, m in enumerate(markets):
        df = load_candles(DB_PATH, m)
        if df is None or len(df) < 70:
            continue
        df = df.copy()
        df["market"] = m
        frames.append(build_market_features(df))
        if (i + 1) % 50 == 0:
            log.info("  %d/%d markets", i + 1, len(markets))
    panel = pd.concat(frames, ignore_index=True)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel["date"] = panel["timestamp"].dt.date
    panel = panel.sort_values(["date", "market"]).reset_index(drop=True)
    return panel


def apply_dynamic_universe(panel: pd.DataFrame, top_n: int,
                           surge_cut: float) -> pd.DataFrame:
    """동적 유니버스: D-1 qv top-N  OR  qv_surge_7d >= surge_cut.

    둘 다 D-1 정보 (f_qv_rank, f_qv_surge_7d 는 shift 된 값). 정적 top100 대비
    펌프 recall 을 끌어올린다 (직전 발굴 +7.9pp).
    """
    in_top = panel["f_qv_rank"] <= top_n
    in_surge = panel["f_qv_surge_7d"] >= surge_cut
    mask = (in_top | in_surge).fillna(False)
    out = panel[mask].copy()
    out["in_universe"] = True
    return out


# ============================================================================
# 2. Scorers — composite / logit / xgb. 전부 train(과거)→test(미래) 분리.
# ============================================================================
def composite_score(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """검증된 선행 rank 의 가중합. 학습 없음(해석 가능). test 의 그날 panel 내에서
    cross-sectional pct-rank 후 가중합. 입력 feature 는 모두 D-1 → leak 아님."""
    score = np.zeros(len(test))
    test = test.copy()
    for feat, w in COMPOSITE_RANKS.items():
        if feat not in test.columns:
            continue
        # 그날(date) 내 pct-rank (높을수록 펌프 쪽, dir=high 검증됨)
        r = test.groupby("date")[feat].rank(pct=True)
        score += w * r.fillna(0.5).values
    return score


def _prep_xy(df: pd.DataFrame, feats: list[str], label: str):
    X = df[feats].copy()
    med = X.median(numeric_only=True)
    X = X.fillna(med)
    X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    y = df[label].values
    return X, y, med


def logit_score(train: pd.DataFrame, test: pd.DataFrame,
                feats: list[str], label: str) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    Xtr, ytr, med = _prep_xy(train, feats, label)
    if ytr.sum() < 20 or len(np.unique(ytr)) < 2:
        return np.full(len(test), np.nan)
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    clf = LogisticRegression(max_iter=500, class_weight="balanced", C=0.5)
    clf.fit(Xtr_s, ytr)
    Xte = test[feats].fillna(med).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    Xte_s = sc.transform(Xte)
    return clf.predict_proba(Xte_s)[:, 1]


def xgb_score(train: pd.DataFrame, test: pd.DataFrame,
              feats: list[str], label: str) -> np.ndarray:
    import xgboost as xgb
    Xtr, ytr, med = _prep_xy(train, feats, label)
    if ytr.sum() < 20 or len(np.unique(ytr)) < 2:
        return np.full(len(test), np.nan)
    pos = max(ytr.sum(), 1)
    neg = max(len(ytr) - ytr.sum(), 1)
    clf = xgb.XGBClassifier(
        n_estimators=250, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
        reg_lambda=2.0, scale_pos_weight=neg / pos,
        eval_metric="logloss", n_jobs=4, random_state=0,
    )
    clf.fit(Xtr, ytr)
    Xte = test[feats].fillna(med).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return clf.predict_proba(Xte)[:, 1]


# ============================================================================
# 3. Calibration — bucket-based historical hit (rare-event 정직).
#   train 에서 raw score → bin → bin 별 실제 hit rate. test 의 raw score 를
#   train bin 에 매핑해 calibrated prob 부여. (val/test 에서 fit 금지 = leak.)
# ============================================================================
def fit_calibration_bins(train_score: np.ndarray, train_y: np.ndarray,
                         n_bins: int = 20) -> pd.DataFrame:
    d = pd.DataFrame({"s": train_score, "y": train_y}).dropna()
    if len(d) < 200 or d["y"].sum() < 5:
        return pd.DataFrame()
    try:
        d["b"] = pd.qcut(d["s"].rank(method="first"), n_bins, labels=False)
    except Exception:
        return pd.DataFrame()
    g = d.groupby("b").agg(lo=("s", "min"), hi=("s", "max"),
                           hit=("y", "mean"), n=("y", "size")).reset_index()
    return g


def apply_calibration(test_score: np.ndarray, bins: pd.DataFrame) -> np.ndarray:
    if bins.empty:
        return np.full(len(test_score), np.nan)
    edges = bins["hi"].values
    hits = bins["hit"].values
    idx = np.searchsorted(edges, test_score, side="left")
    idx = np.clip(idx, 0, len(hits) - 1)
    out = hits[idx]
    out[np.isnan(test_score)] = np.nan
    return out


# ============================================================================
# 4. Walk-forward OOS: expanding train + embargo. per-fold score+calibrate.
# ============================================================================
def walk_forward_scores(panel: pd.DataFrame, feats: list[str], label: str,
                        n_folds: int = 6, embargo_days: int = 5,
                        min_train_frac: float = 0.35) -> pd.DataFrame:
    """각 fold: train(과거 expanding) → 3 스코어러로 test fold scoring + calibration.
    반환: test rows + raw_/cal_ score 컬럼 (모든 fold concat = full OOS)."""
    dates = np.sort(panel["date"].unique())
    n = len(dates)
    start = int(n * min_train_frac)
    fold_edges = np.linspace(start, n, n_folds + 1).astype(int)
    out_frames = []
    for k in range(n_folds):
        tr_end = fold_edges[k]
        te_start = tr_end + embargo_days
        te_end = fold_edges[k + 1]
        if te_start >= te_end:
            continue
        tr_dates = set(dates[:tr_end])
        te_dates = set(dates[te_start:te_end])
        tr = panel[panel["date"].isin(tr_dates)].copy()
        te = panel[panel["date"].isin(te_dates)].copy()
        if len(tr) < 1000 or len(te) < 100 or tr[label].sum() < 20:
            continue

        res = te.copy()
        res["fold"] = k
        # (a) composite
        res["raw_composite"] = composite_score(tr, te)
        cb = fit_calibration_bins(composite_score(tr, tr), tr[label].values)
        res["cal_composite"] = apply_calibration(res["raw_composite"].values, cb)
        # (b) logit
        res["raw_logit"] = logit_score(tr, te, feats, label)
        cb = fit_calibration_bins(
            logit_score(tr, tr, feats, label), tr[label].values)
        res["cal_logit"] = apply_calibration(res["raw_logit"].values, cb)
        # (c) xgb
        res["raw_xgb"] = xgb_score(tr, te, feats, label)
        cb = fit_calibration_bins(
            xgb_score(tr, tr, feats, label), tr[label].values)
        res["cal_xgb"] = apply_calibration(res["raw_xgb"].values, cb)

        out_frames.append(res)
        log.info("  fold %d: train=%d(<%s) test=%d(%s..%s) pos_tr=%d",
                 k, len(tr), dates[tr_end - 1], len(te),
                 dates[te_start], dates[te_end - 1], int(tr[label].sum()))
    if not out_frames:
        return pd.DataFrame()
    return pd.concat(out_frames, ignore_index=True)


# ============================================================================
# 5. precision@K / recall / lift (시간순 OOS, daily top-K).
# ============================================================================
def topk_metrics(oos: pd.DataFrame, score_col: str, label: str,
                 ks=(1, 2, 3, 5), regime: str | None = None) -> list[dict]:
    """일별 score top-K 추천 → precision@K. recall = 잡은 펌프 / 전체 펌프(유니버스 내).
    lift = precision@K / base_rate. regime 지정 시 그 regime 일자만."""
    d = oos.dropna(subset=[score_col, label]).copy()
    if regime is not None:
        d = d[d["regime"] == regime]
    if len(d) < 50:
        return []
    base = d[label].mean()
    total_pumps = d[label].sum()
    rows = []
    for K in ks:
        # 일별 top-K
        d = d.sort_values([ "date", score_col], ascending=[True, False])
        topk = d.groupby("date").head(K)
        if len(topk) == 0:
            continue
        prec = topk[label].mean()
        caught = topk[label].sum()
        n_days = topk["date"].nunique()
        rows.append({
            "score": score_col, "label": label,
            "regime": regime or "ALL", "K": K,
            "n_days": int(n_days), "n_picks": int(len(topk)),
            "precision_at_k": float(prec),
            "base_rate": float(base),
            "lift": float(prec / base) if base > 0 else np.nan,
            "recall": float(caught / total_pumps) if total_pumps > 0 else np.nan,
            "pumps_caught": int(caught), "total_pumps": int(total_pumps),
        })
    return rows


def calibration_table(oos: pd.DataFrame, score_col: str, label: str,
                      n_bins: int = 10) -> pd.DataFrame:
    d = oos.dropna(subset=[score_col, label]).copy()
    if len(d) < 200:
        return pd.DataFrame()
    try:
        d["b"] = pd.qcut(d[score_col].rank(method="first"), n_bins,
                         labels=False, duplicates="drop")
    except Exception:
        return pd.DataFrame()
    g = d.groupby("b").agg(
        pred_prob=(score_col, "mean"),
        actual_hit=(label, "mean"),
        n=(label, "size"),
    ).reset_index()
    g["score"] = score_col
    g["label"] = label
    return g


# ============================================================================
# 6. main
# ============================================================================
def _configure_cli_logging() -> None:
    """Configure stdout logging only for direct CLI execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main():
    _configure_cli_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-markets", type=int, default=None)
    ap.add_argument("--universe-top", type=int, default=100)
    ap.add_argument("--surge-cut", type=float, default=3.0)
    ap.add_argument("--n-folds", type=int, default=6)
    ap.add_argument("--recent-days", type=int, default=14)
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    panel = build_panel(args.limit_markets)
    panel = add_cross_sectional(panel)
    panel = attach_btc_regime(panel)
    log.info("panel rows=%d markets=%d dates=%d",
             len(panel), panel["market"].nunique(), panel["date"].nunique())

    feats = [f for f in PRECURSOR_FEATURES if f in panel.columns]
    log.info("model features: %d", len(feats))

    # 동적 유니버스 적용 (정적 top-N 결과는 recall_gap 비교용으로 따로 계산)
    full_panel = panel.copy()  # 유니버스 적용 전 (recall gap 측정용)
    uni = apply_dynamic_universe(panel, args.universe_top, args.surge_cut)
    log.info("dynamic universe rows=%d (%.1f%% of full) — top%d OR qv_surge>=%.1f",
             len(uni), 100 * len(uni) / len(panel), args.universe_top, args.surge_cut)
    log.info("base rate pump20 full=%.4f universe=%.4f",
             full_panel[MAIN_LABEL].mean(), uni[MAIN_LABEL].mean())

    # selection bias 기록
    log.info("SELECTION: scorers=3 (composite/logit/xgb) × labels=2 × "
             "calibration(bucket) ; features=%d (검증된 선행 set, hand-pick after data)",
             len(feats))

    # ---- walk-forward OOS scores (동적 유니버스 위) ----
    log.info("=== walk-forward OOS scoring (pump20) ===")
    oos = walk_forward_scores(uni, feats, MAIN_LABEL, n_folds=args.n_folds)
    if oos.empty:
        log.error("no OOS folds produced — aborting")
        sys.exit(1)
    log.info("OOS rows=%d, dates %s..%s",
             len(oos), oos["date"].min(), oos["date"].max())

    # ---- precision@K / recall / lift (스코어러 × cal/raw × regime) ----
    metric_rows = []
    score_cols = ["cal_composite", "cal_logit", "cal_xgb",
                  "raw_composite", "raw_logit", "raw_xgb"]
    for sc in score_cols:
        if sc not in oos.columns:
            continue
        metric_rows += topk_metrics(oos, sc, MAIN_LABEL, regime=None)
        for rg in REGIMES:
            metric_rows += topk_metrics(oos, sc, MAIN_LABEL, regime=rg)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT_DIR / "recommendation_scorer_oos_metrics_v1.csv", index=False)
    log.info("wrote recommendation_scorer_oos_metrics_v1.csv (%d rows)", len(metrics))

    # 콘솔: ALL regime, calibrated scorers, K=1,2,3,5
    log.info("\n===== OOS precision@K / recall / lift (ALL regime, calibrated) =====")
    al = metrics[(metrics["regime"] == "ALL") &
                 (metrics["score"].str.startswith("cal_"))]
    for sc in ["cal_composite", "cal_logit", "cal_xgb"]:
        ss = al[al["score"] == sc].sort_values("K")
        for _, r in ss.iterrows():
            log.info("  %-14s K=%d prec@K=%.3f lift=%.2f recall=%.3f "
                     "(caught %d/%d, %d days)",
                     sc, int(r["K"]), r["precision_at_k"], r["lift"],
                     r["recall"], int(r["pumps_caught"]),
                     int(r["total_pumps"]), int(r["n_days"]))

    # best scorer = ALL regime, K=3, calibrated, 최고 precision@K
    k3 = al[al["K"] == 3].sort_values("precision_at_k", ascending=False)
    best_score = k3.iloc[0]["score"] if len(k3) else "cal_composite"
    log.info("BEST scorer (ALL K=3): %s", best_score)

    # ---- regime별 요약 ----
    log.info("\n===== OOS precision@3 by regime (best=%s) =====", best_score)
    rg_tab = metrics[(metrics["score"] == best_score) & (metrics["K"] == 3)]
    for _, r in rg_tab.iterrows():
        log.info("  regime=%-14s prec@3=%.3f lift=%.2f recall=%.3f n_days=%d",
                 r["regime"], r["precision_at_k"], r["lift"], r["recall"],
                 int(r["n_days"]))

    # ---- calibration tables ----
    cal_frames = []
    for sc in ["cal_composite", "cal_logit", "cal_xgb"]:
        cal_frames.append(calibration_table(oos, sc, MAIN_LABEL))
    cal_df = pd.concat([c for c in cal_frames if not c.empty], ignore_index=True) \
        if any(not c.empty for c in cal_frames) else pd.DataFrame()
    if not cal_df.empty:
        cal_df.to_csv(OUT_DIR / "recommendation_scorer_calibration_v1.csv", index=False)
        log.info("wrote recommendation_scorer_calibration_v1.csv")
        log.info("\n===== calibration (best=%s): pred_prob vs actual_hit =====", best_score)
        cc = cal_df[cal_df["score"] == best_score].sort_values("pred_prob")
        for _, r in cc.iterrows():
            log.info("  bin pred=%.3f actual=%.3f n=%d",
                     r["pred_prob"], r["actual_hit"], int(r["n"]))

    # ---- 최근 N 거래일 top-K 추천 + 실제 펌프 여부 ----
    completed_oos = oos.dropna(subset=[MAIN_LABEL]).copy()
    recent_dates = sorted(completed_oos["date"].unique())[-args.recent_days:]
    recent = completed_oos[completed_oos["date"].isin(recent_dates)].copy()
    pick_rows = []
    for dt in recent_dates:
        day = recent[recent["date"] == dt].dropna(subset=[best_score])
        if day.empty:
            continue
        day = day.sort_values(best_score, ascending=False)
        for K in (3, 5):
            top = day.head(K)
            picks = []
            for _, r in top.iterrows():
                hit = int(r[MAIN_LABEL] == 1)
                ret = r.get("intraday_high_ret", np.nan)
                picks.append(f"{r['market'].replace('KRW-','')}"
                             f"({r[best_score]:.1%}{'/HIT' if hit else ''}"
                             f"+{ret:.1%})" if pd.notna(ret) else
                             f"{r['market'].replace('KRW-','')}({r[best_score]:.1%})")
            n_hit = int(top[MAIN_LABEL].sum())
            pick_rows.append({
                "date": str(dt), "K": K, "scorer": best_score,
                "n_hit": n_hit, "precision": n_hit / len(top),
                "picks": " | ".join(picks),
            })
    picks_df = pd.DataFrame(pick_rows)
    picks_df.to_csv(OUT_DIR / "recommendation_scorer_recent_picks_v1.csv", index=False)
    log.info("wrote recommendation_scorer_recent_picks_v1.csv (%d rows)", len(picks_df))
    log.info("\n===== 최근 %d 거래일 top-3 추천 + 실제 펌프 (best=%s) =====",
             args.recent_days, best_score)
    for _, r in picks_df[picks_df["K"] == 3].iterrows():
        log.info("  %s  hit=%d/3  %s", r["date"], r["n_hit"], r["picks"])

    # ---- 오늘 (최신 가용일) top 후보 ----
    # 최신 일자는 walk-forward 마지막 test fold 에 없을 수 있으므로, 전체 train
    # (마지막 가용일 직전까지) 으로 fit 한 모델로 최신일 panel 을 scoring.
    today_df = score_latest_day(uni, feats, MAIN_LABEL, best_score)
    if today_df is not None:
        today_df.to_csv(OUT_DIR / "recommendation_scorer_today_v1.csv", index=False)
        log.info("wrote recommendation_scorer_today_v1.csv")
        log.info("\n===== 오늘 (%s) top 후보 + calibrated 급등확률 (best=%s) =====",
                 today_df["date"].iloc[0], best_score)
        for _, r in today_df.head(10).iterrows():
            log.info("  %-14s prob=%.1f%%  (qv_surge7=%.2f ret3=%.1f%% bounce=%.2f)",
                     r["market"], 100 * r["pump_prob"],
                     r.get("f_qv_surge_7d", np.nan),
                     100 * r.get("f_ret_3d", np.nan) if pd.notna(r.get("f_ret_3d")) else np.nan,
                     r.get("f_bounce_off_7d_low", np.nan))

    # ---- recall gap: 정적 top100 vs 동적 유니버스 + 유니버스밖 펌프 ----
    recall_gap_report(panel, full_panel, args.universe_top, args.surge_cut)

    log.info("DONE.")


def score_latest_day(uni: pd.DataFrame, feats: list[str], label: str,
                     best_score: str) -> pd.DataFrame | None:
    """최신 가용일 panel 을 직전까지 train 으로 fit 한 모델로 scoring + calibration.
    leak: train 은 (최신일 - embargo) 이전, 최신일은 D-1 feature 만 본다."""
    dates = np.sort(uni["date"].unique())
    if len(dates) < 50:
        return None
    today = dates[-1]
    embargo = 5
    train_dates = set(dates[: max(0, len(dates) - 1 - embargo)])
    tr = uni[uni["date"].isin(train_dates)].copy()
    te = uni[uni["date"] == today].copy()
    if len(tr) < 1000 or len(te) == 0:
        return None
    kind = best_score.replace("cal_", "").replace("raw_", "")
    if kind == "composite":
        raw_te = composite_score(tr, te)
        raw_tr = composite_score(tr, tr)
    elif kind == "logit":
        raw_te = logit_score(tr, te, feats, label)
        raw_tr = logit_score(tr, tr, feats, label)
    else:
        raw_te = xgb_score(tr, te, feats, label)
        raw_tr = xgb_score(tr, tr, feats, label)
    bins = fit_calibration_bins(raw_tr, tr[label].values)
    cal_te = apply_calibration(raw_te, bins)
    te = te.copy()
    te["pump_prob"] = cal_te if not bins.empty else raw_te
    te["raw_score"] = raw_te
    keep = ["market", "date", "pump_prob", "raw_score", "regime",
            "f_qv_surge_7d", "f_qv_surge_30d", "f_ret_3d", "f_ret_7d",
            "f_bounce_off_7d_low", "f_atr_pct_14", "f_rv_7d", "f_qv_rank"]
    keep = [c for c in keep if c in te.columns]
    out = te[keep].sort_values("pump_prob", ascending=False).reset_index(drop=True)
    out["date"] = out["date"].astype(str)
    return out


def recall_gap_report(uni_panel: pd.DataFrame, full_panel: pd.DataFrame,
                      top_n: int, surge_cut: float):
    """동적 유니버스 recall vs 정적 top-N. 유니버스밖 펌프(놓친 것) 회수 표시."""
    full = full_panel.dropna(subset=[MAIN_LABEL]).copy()
    total_pumps = full[MAIN_LABEL].sum()
    static = full[full["f_qv_rank"] <= top_n]
    in_top = full["f_qv_rank"] <= top_n
    in_surge = full["f_qv_surge_7d"] >= surge_cut
    dyn = full[(in_top | in_surge).fillna(False)]
    static_caught = static[MAIN_LABEL].sum()
    dyn_caught = dyn[MAIN_LABEL].sum()
    log.info("\n===== RECALL GAP (pump20, full history) =====")
    log.info("  total pumps (all coins)       = %d", int(total_pumps))
    log.info("  static top%-3d recall          = %.3f (caught %d)",
             top_n, static_caught / total_pumps, int(static_caught))
    log.info("  dynamic (top%d OR surge>=%.1f) = %.3f (caught %d)  Δ=%+.1fpp",
             top_n, surge_cut, dyn_caught / total_pumps, int(dyn_caught),
             100 * (dyn_caught - static_caught) / total_pumps)
    # 유니버스밖 펌프 (정적/동적 둘 다 밖)
    outside = full[~(in_top | in_surge).fillna(False) & (full[MAIN_LABEL] == 1)]
    log.info("  pumps OUTSIDE dynamic universe = %d (%.1f%% — 회수 불가)",
             len(outside), 100 * len(outside) / total_pumps)
    # 유니버스밖 펌프 큰 것 몇 개 (회수 안 된 케이스 사례)
    if "intraday_high_ret" in outside.columns and len(outside):
        big = outside.sort_values("intraday_high_ret", ascending=False).head(5)
        for _, r in big.iterrows():
            log.info("    missed: %-12s %s +%.0f%% (qv_rank=%s surge7=%.2f)",
                     r["market"], str(r["date"]),
                     100 * r["intraday_high_ret"],
                     int(r["f_qv_rank"]) if pd.notna(r["f_qv_rank"]) else -1,
                     r["f_qv_surge_7d"] if pd.notna(r["f_qv_surge_7d"]) else -1)


if __name__ == "__main__":
    main()

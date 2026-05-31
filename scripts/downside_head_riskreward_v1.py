"""Downside head + risk-reward ranking (component=downside_rank).

사용자 refined 비전 (Task B):
  코인×일별로 **calibrated 하락 리스크**를 산출 +  multi-threshold **상승 분포** 를 산출하고,
  "하락확률 낮고 상승확률 높은" 코인을 cross-sectional 로 surface 하는 risk-reward 랭킹을 만든다.
  상승폭 기준(20% 등)은 고정 X — 분포로 열어두고 임계별 base rate + 달성가능성을 표기.

산출 (모두 leak-free, 임계별 동일 규율):
  UPSIDE head (day-D high/open):   P(>=+5%), P(>=+10%), P(>=+15%), P(>=+20%)
  DOWNSIDE head (day-D low/open):  P(<=-3%), P(<=-5%), P(<=-10%),  P(close<0)
  기대(중앙) 하방:  E[min_ret | <=0],  median(min_ret),  E[min_ret] (전체)
→ risk-reward 랭킹 3종 결합 비교:
    (R1) ratio       = P(>=10% up) / max(P(<=-5% down), eps)
    (R2) penalized   = P(>=10% up) - lam * P(<=-5% down)     (lam grid)
    (R3) gated       = P(>=10% up) 로 정렬하되 P(<=-5% down) > q-cut 인 코인 제외(필터)
  vs baseline = P(>=10% up) only (상승확률만).

검증:
  risk-reward top-K 픽이 upside-only top-K 대비
    - 실현 하방(다음날 min_ret, P(min<=-5%), P(min<=-10%), deep-dump rate, CVaR)이 실제 낮은가?
    - 상승확률(실현 pump rate, hit>=+10%)은 유지되나?
  + 임계별 OOS calibration reliability (pred bucket vs actual) — calibration 정직성.

★ LEAK 방어 (same-day leak 2번 전적 — 양보 X):
  - feature = build_market_features 의 market 별 .shift(1) (D-1 까지). LEAK_COLS/next_* 제외.
  - 라벨(up/down) = day-D open 대비 day-D high/low/close (타겟, 미래) — 학습 feature 에 안 섞임.
  - 모델/calibration = expanding train(과거 fold) 에서만 fit, test fold(미래) 적용만.
    임계별 calibration 은 train OOF bucket historical-hit (raw prob 과신 금지).
    각 임계 동일 규율 (multi-threshold 라도 same pipeline).
  - 오늘 scan = D-1 feature 만으로 추론 (라벨/미래봉 안 봄).
  - self-contained: prelude 내부 모듈만 import.

거래비용 0.15% 왕복 — 실현 net 지표 차감. 사이징중립 = day-equal-weight.
라벨/유니버스/사이징/알림 변경 X (이건 빌드/검증, 사용자 컨펌 영역 안 건드림).

사용:
    python scripts/downside_head_riskreward_v1.py
    python scripts/downside_head_riskreward_v1.py --limit-markets 80   # 개발
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

from data.database import list_markets, load_candles  # noqa: E402
from scripts.univariate_precursor_lift_v1 import (  # noqa: E402
    build_market_features,
    add_cross_sectional,
)
from scripts.recommendation_scorer_v1 import PRECURSOR_FEATURES  # noqa: E402
from scripts.regime_split_precursor_v1 import attach_btc_regime  # noqa: E402

D1_DB = str(_ROOT / "data" / "upbit_d1.db")
OUT = _ROOT / "output"
EPS = 1e-12
ROUND_TRIP_COST = 0.0015

# 학습에서 절대 제외 (LEAK_COLS + next_* + 라벨/내부 outcome)
LEAK_COLS = {
    "lab_pump20", "lab_pump15", "lab_pumpc20", "intraday_high_ret",
    "up_high_ret", "down_low_ret", "eod_ret",
    "lab_up_05", "lab_up_10", "lab_up_15", "lab_up_20",
    "lab_dn_03", "lab_dn_05", "lab_dn_10", "lab_close_neg",
}

# 상승 임계 grid (분포로 열어둠 — 고정 X). day-D high/open - 1 기준.
UP_THRESH = {"up_05": 0.05, "up_10": 0.10, "up_15": 0.15, "up_20": 0.20}
# 하락 임계 grid. day-D min(low)/open - 1 기준 (close<0 은 close/open).
DN_THRESH = {"dn_03": -0.03, "dn_05": -0.05, "dn_10": -0.10}

# risk-reward 랭킹: 상승은 +10%, 하방은 -5% 를 anchor (사용자: ~-5% 손실 수용)
UP_ANCHOR = "up_10"
DN_ANCHOR = "dn_05"

LAMBDA_GRID = [0.5, 1.0, 2.0, 3.0]          # R2 penalized
GATE_CUT_GRID = [0.6, 0.7, 0.8]             # R3 gate (downside prob 상위 분위 제외)
TOP_KS = (3, 5)

N_FOLDS = 6
EMBARGO = 5
CAL_BUCKETS = 10
UNIVERSE_TOP = 100
SURGE_CUT = 3.0

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("downside_rank")


# ============================================================================
# 1. Panel — leak-free D-1 features + day-D up/down 라벨
# ============================================================================
def _add_outcome_labels(g: pd.DataFrame) -> pd.DataFrame:
    """한 코인 일봉에 day-D 진입(open) 대비 outcome 라벨 부착 (타겟=미래, leak X).
    build_market_features 가 이미 feature 를 shift(1) 했으므로 여기 outcome 은
    같은 row(day-D) 의 open/high/low/close 로 만든다 → feature(D-1) 와 시점분리."""
    o = g["open"].values
    h = g["high"].values
    l = g["low"].values
    c = g["close"].values
    g = g.copy()
    g["up_high_ret"] = h / (o + EPS) - 1.0     # 진입(open)→장중 최고
    g["down_low_ret"] = l / (o + EPS) - 1.0    # 진입(open)→장중 최저 (<=0 보통)
    g["eod_ret"] = c / (o + EPS) - 1.0         # 진입(open)→종가
    for name, thr in UP_THRESH.items():
        g[f"lab_{name}"] = (g["up_high_ret"] >= thr).astype(float)
    for name, thr in DN_THRESH.items():
        g[f"lab_{name}"] = (g["down_low_ret"] <= thr).astype(float)
    g["lab_close_neg"] = (g["eod_ret"] < 0).astype(float)
    return g


def build_panel(limit_markets):
    markets = list_markets(D1_DB)
    if limit_markets:
        markets = markets[:limit_markets]
    log.info("loading %d markets", len(markets))
    frames = []
    for i, m in enumerate(markets):
        df = load_candles(D1_DB, m)
        if df is None or len(df) < 70:
            continue
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["market"] = m
        feat = build_market_features(df)          # D-1 feature + pump labels
        # day-D outcome (open 대비) 라벨을 같은 timestamp 로 join
        oc = _add_outcome_labels(df)[
            ["timestamp", "up_high_ret", "down_low_ret", "eod_ret"]
            + [f"lab_{n}" for n in UP_THRESH]
            + [f"lab_{n}" for n in DN_THRESH]
            + ["lab_close_neg"]
        ]
        feat = feat.merge(oc, on="timestamp", how="left")
        frames.append(feat)
        if (i + 1) % 60 == 0:
            log.info("  %d/%d", i + 1, len(markets))
    panel = pd.concat(frames, ignore_index=True)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel["date"] = panel["timestamp"].dt.date
    panel = panel.sort_values(["date", "market"]).reset_index(drop=True)
    return panel


def apply_universe(panel):
    in_top = panel["f_qv_rank"] <= UNIVERSE_TOP
    in_surge = panel["f_qv_surge_7d"] >= SURGE_CUT
    mask = (in_top | in_surge).fillna(False)
    out = panel[mask].copy()
    out["in_universe"] = True
    return out


# ============================================================================
# 2. Head model (XGBoost) + per-threshold per-fold OOF bucket calibration
# ============================================================================
def _feats(panel):
    cand = [c for c in PRECURSOR_FEATURES if c in panel.columns and c not in LEAK_COLS]
    return cand


def _xgb_fit_predict(Xtr, ytr, Xte):
    import xgboost as xgb
    pos = ytr.sum()
    if pos < 12 or len(np.unique(ytr)) < 2:
        return None, None
    spw = float((len(ytr) - pos) / max(pos, 1))
    m = xgb.XGBClassifier(
        n_estimators=180, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_lambda=1.5, scale_pos_weight=spw, n_jobs=4,
        eval_metric="logloss", tree_method="hist",
    )
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1], m


def _oof_bucket_calib(raw_train, y_train, n_buckets):
    """train OOF raw score → bucket actual-hit map (rare-event raw 과신 금지).
    반환 (edges, hit_map, base). edges = bucket 상한(quantile)."""
    df = pd.DataFrame({"s": raw_train, "y": y_train}).dropna()
    base = float(df["y"].mean()) if len(df) else np.nan
    if len(df) < 150 or df["y"].sum() < 5:
        return None, None, base
    try:
        bk = pd.qcut(df["s"].rank(method="first"), n_buckets,
                     labels=False, duplicates="drop")
    except ValueError:
        return None, None, base
    df = df.assign(bk=bk)
    g = df.groupby("bk").agg(hi=("s", "max"), hit=("y", "mean")).sort_index()
    return g["hi"].values, g["hit"].to_dict(), base


def _apply_calib(scores, edges, hit_map, base):
    if edges is None or hit_map is None:
        return np.full(len(scores), base, dtype=float)
    idx = np.searchsorted(edges, scores, side="left")
    idx = np.clip(idx, 0, len(edges) - 1)
    return np.array([hit_map.get(int(b), base) for b in idx], dtype=float)


def walk_forward_heads(panel, feats, label_cols, n_folds, embargo):
    """expanding WF. 각 fold·각 임계: train OOF→XGB→test cal_prob.
    calibration = train-only OOF bucket. test fold 는 적용만 (leak X).
    또한 down_low_ret 의 expected/median downside head (분위회귀 대용: train 분포)도
    각 fold test 에 train 의 조건부 기대값으로 부착(여기선 회귀 head)."""
    dates = np.sort(panel["date"].unique())
    n = len(dates)
    start = int(n * 0.35)
    edges_f = np.linspace(start, n, n_folds + 1).astype(int)
    out = []
    for k in range(n_folds):
        tr_end = edges_f[k]
        te_start = tr_end + embargo
        te_end = edges_f[k + 1]
        if te_start >= te_end:
            continue
        tr_d = set(dates[:tr_end]); te_d = set(dates[te_start:te_end])
        tr = panel[panel["date"].isin(tr_d)].copy()
        te = panel[panel["date"].isin(te_d)].copy()
        if len(tr) < 1000 or len(te) < 100:
            continue
        Xtr = tr[feats].replace([np.inf, -np.inf], np.nan)
        med = Xtr.median()
        Xtr = Xtr.fillna(med).values
        Xte = te[feats].replace([np.inf, -np.inf], np.nan).fillna(med).values
        res = te.copy()
        res["fold"] = k
        for lc in label_cols:
            y = tr[lc].values
            raw_te, m = _xgb_fit_predict(Xtr, y, Xte)
            if raw_te is None:
                res[f"p_{lc}"] = float(np.nanmean(y))
                res[f"raw_{lc}"] = np.nan
                continue
            raw_tr = m.predict_proba(Xtr)[:, 1]
            ed, hm, base = _oof_bucket_calib(raw_tr, y, CAL_BUCKETS)
            res[f"raw_{lc}"] = raw_te
            res[f"p_{lc}"] = _apply_calib(raw_te, ed, hm, base)
        # 기대/중앙 하방: train 의 down_low_ret 를 raw down-prob bucket 으로 조건부 평균
        # (간단 leak-free 회귀 head). dn_05 raw 가 없으면 train 전역 통계.
        if "raw_lab_dn_05" in res.columns and res["raw_lab_dn_05"].notna().any():
            # train 의 raw dn_05 → down_low_ret 조건부 통계
            ytr_raw, mtmp = _xgb_fit_predict(Xtr, tr["lab_dn_05"].values, Xtr)
            if ytr_raw is not None:
                tdf = pd.DataFrame({"s": ytr_raw, "lr": tr["down_low_ret"].values}).dropna()
                try:
                    tdf["bk"] = pd.qcut(tdf["s"].rank(method="first"), CAL_BUCKETS,
                                        labels=False, duplicates="drop")
                    gg = tdf.groupby("bk").agg(hi=("s", "max"),
                                               mean_lr=("lr", "mean"),
                                               med_lr=("lr", "median"))
                    edd = gg["hi"].values
                    mmap = gg["mean_lr"].to_dict(); medmap = gg["med_lr"].to_dict()
                    g_base = float(tdf["lr"].mean()); g_med = float(tdf["lr"].median())
                    idx = np.clip(np.searchsorted(edd, res["raw_lab_dn_05"].values,
                                                  side="left"), 0, len(edd) - 1)
                    res["exp_downside"] = [mmap.get(int(b), g_base) for b in idx]
                    res["med_downside"] = [medmap.get(int(b), g_med) for b in idx]
                except ValueError:
                    res["exp_downside"] = float(tr["down_low_ret"].mean())
                    res["med_downside"] = float(tr["down_low_ret"].median())
        out.append(res)
        log.info("  fold %d train=%d(<%s) test=%d(%s..%s)", k, len(tr),
                 dates[tr_end - 1], len(te), dates[te_start], dates[te_end - 1])
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


# ============================================================================
# 3. 실현 net path (daily candle) — day-equal-weight 하방 지표
# ============================================================================
def _net_eod(row):
    return row["eod_ret"] - ROUND_TRIP_COST


def _downside_metrics(df):
    """top-K 픽 모집단의 실현 하방 + 상승 유지 지표 (사이징중립 day-eq)."""
    d = df.dropna(subset=["down_low_ret", "eod_ret"]).copy()
    if len(d) == 0:
        return {}
    minr = d["down_low_ret"].values
    eod = d["eod_ret"].values
    netc = eod - ROUND_TRIP_COST
    q05 = np.quantile(eod, 0.05)
    cvar = float(eod[eod <= q05].mean()) if (eod <= q05).any() else float(q05)
    # day-equal-weight cum/MaxDD/Sortino on EOD net (cost 차감)
    daily = pd.DataFrame({"date": d["date"].values, "r": netc}).groupby("date")["r"].mean().sort_index()
    mu = daily.mean(); dstd = daily[daily < 0].std(); sd = daily.std()
    eq = (1 + daily).cumprod(); peak = eq.cummax()
    return dict(
        n=int(len(d)),
        realized_pump10=float((d["up_high_ret"] >= 0.10).mean()),
        realized_pump20=float((d["up_high_ret"] >= 0.20).mean()),
        hit_eod_pos=float((eod > 0).mean()),
        p_min_le_5=float((minr <= -0.05).mean()),
        p_min_le_10=float((minr <= -0.10).mean()),
        deep_dump_rate=float(((minr <= -0.10) | (eod <= -0.05)).mean()),
        mean_min_ret=float(minr.mean()),
        median_min_ret=float(np.median(minr)),
        mean_eod_net=float(netc.mean()),
        median_eod_net=float(np.median(netc)),
        cvar95_eod=cvar,
        worst_eod=float(eod.min()),
        mdd=float(((eq - peak) / peak).min()),
        cum_eod_net=float(eq.iloc[-1] - 1.0),
        sortino=float(mu / dstd * np.sqrt(365)) if dstd and dstd > 0 else np.nan,
        sharpe=float(mu / sd * np.sqrt(365)) if sd and sd > 0 else np.nan,
    )


# ============================================================================
# 4. risk-reward 랭킹 3종 비교 vs upside-only baseline
# ============================================================================
def _topk(d, score_col, k):
    return (d.sort_values(["date", score_col], ascending=[True, False])
            .groupby("date").head(k))


def compare_rankings(oos, k):
    up = f"p_lab_{UP_ANCHOR}"; dn = f"p_lab_{DN_ANCHOR}"
    d = oos.dropna(subset=[up, dn]).copy()
    rows = []
    # baseline: upside-only
    base = _topk(d, up, k)
    b = _downside_metrics(base); b.update(policy="upside_only", k=k, param="-")
    rows.append(b)
    # R1 ratio
    d["rr_ratio"] = d[up] / np.maximum(d[dn], 1e-3)
    t = _topk(d, "rr_ratio", k)
    m = _downside_metrics(t); m.update(policy="R1_ratio", k=k, param="-"); rows.append(m)
    # R2 penalized
    for lam in LAMBDA_GRID:
        d["rr_pen"] = d[up] - lam * d[dn]
        t = _topk(d, "rr_pen", k)
        m = _downside_metrics(t); m.update(policy="R2_penalized", k=k, param=f"lam={lam}")
        rows.append(m)
    # R3 gate: downside prob 상위 분위 제외 후 upside 정렬
    for q in GATE_CUT_GRID:
        thr = d[dn].quantile(q)
        dd = d[d[dn] <= thr]
        if dd.empty:
            continue
        t = _topk(dd, up, k)
        if len(t) < 20:
            continue
        m = _downside_metrics(t); m.update(policy="R3_gate", k=k, param=f"qcut={q}")
        rows.append(m)
    return pd.DataFrame(rows)


# ============================================================================
# 5. calibration reliability (임계별 OOS pred vs actual)
# ============================================================================
def reliability(oos, label_cols):
    rows = []
    for lc in label_cols:
        pc = f"p_{lc}"
        if pc not in oos.columns:
            continue
        d = oos.dropna(subset=[pc, lc]).copy()
        if len(d) < 200:
            continue
        try:
            d["bk"] = pd.qcut(d[pc].rank(method="first"), 5, labels=False, duplicates="drop")
        except ValueError:
            continue
        for bk, gg in d.groupby("bk"):
            rows.append(dict(
                label=lc, bucket=int(bk), n=int(len(gg)),
                pred_mean=float(gg[pc].mean()),
                actual=float(gg[lc].mean()),
            ))
    return pd.DataFrame(rows)


# ============================================================================
# 6. 오늘 scan (D-1 feature 만으로 추론, 최신 train fit)
# ============================================================================
def today_scan(panel, feats, label_cols, asof_date):
    """전 과거(< asof-embargo) 로 head 학습 → asof 유니버스 코인 추론.
    leak X: feature 는 D-1, 라벨/미래봉 안 봄. calibration 은 train OOF."""
    cutoff = pd.Timestamp(asof_date) - pd.Timedelta(days=EMBARGO)
    tr = panel[(panel["date"] < cutoff.date()) & panel["in_universe"]].copy()
    te = panel[(panel["date"] == asof_date) & panel["in_universe"]].copy()
    if te.empty:
        log.warning("asof %s not in universe panel", asof_date)
        return pd.DataFrame()
    Xtr = tr[feats].replace([np.inf, -np.inf], np.nan)
    med = Xtr.median(); Xtr = Xtr.fillna(med).values
    Xte = te[feats].replace([np.inf, -np.inf], np.nan).fillna(med).values
    out = te[["market", "date", "regime"]].copy()
    for lc in label_cols:
        y = tr[lc].values
        raw_te, m = _xgb_fit_predict(Xtr, y, Xte)
        if raw_te is None:
            out[f"p_{lc}"] = float(np.nanmean(y)); continue
        raw_tr = m.predict_proba(Xtr)[:, 1]
        ed, hm, base = _oof_bucket_calib(raw_tr, y, CAL_BUCKETS)
        out[f"p_{lc}"] = _apply_calib(raw_te, ed, hm, base)
    # expected downside (조건부)
    ytr_raw, mtmp = _xgb_fit_predict(Xtr, tr["lab_dn_05"].values, Xtr)
    if ytr_raw is not None:
        tdf = pd.DataFrame({"s": ytr_raw, "lr": tr["down_low_ret"].values}).dropna()
        try:
            tdf["bk"] = pd.qcut(tdf["s"].rank(method="first"), CAL_BUCKETS,
                                labels=False, duplicates="drop")
            gg = tdf.groupby("bk").agg(hi=("s", "max"), mean_lr=("lr", "mean"),
                                       med_lr=("lr", "median"))
            edd = gg["hi"].values; mmap = gg["mean_lr"].to_dict(); medmap = gg["med_lr"].to_dict()
            raw_te_dn = mtmp.predict_proba(Xte)[:, 1]
            idx = np.clip(np.searchsorted(edd, raw_te_dn, side="left"), 0, len(edd) - 1)
            out["exp_downside"] = [mmap.get(int(b), float(tdf["lr"].mean())) for b in idx]
            out["med_downside"] = [medmap.get(int(b), float(tdf["lr"].median())) for b in idx]
        except ValueError:
            out["exp_downside"] = float(tr["down_low_ret"].mean())
            out["med_downside"] = float(tr["down_low_ret"].median())
    # risk-reward 점수 (R2 lam=2 anchor — 검증 best 로 갱신될 placeholder)
    up = f"p_lab_{UP_ANCHOR}"; dn = f"p_lab_{DN_ANCHOR}"
    out["rr_ratio"] = out[up] / np.maximum(out[dn], 1e-3)
    out["rr_pen_lam2"] = out[up] - 2.0 * out[dn]
    out["upside_only_rank"] = out[up].rank(ascending=False, method="min").astype(int)
    out["rr_rank"] = out["rr_pen_lam2"].rank(ascending=False, method="min").astype(int)
    return out.sort_values("rr_pen_lam2", ascending=False).reset_index(drop=True)


# ============================================================================
# main
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-markets", type=int, default=None)
    ap.add_argument("--asof", type=str, default="2026-05-31")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    panel = build_panel(args.limit_markets)
    panel = add_cross_sectional(panel)
    panel = attach_btc_regime(panel)
    panel = apply_universe(panel)
    feats = _feats(panel)
    log.info("panel(universe) rows=%d markets=%d dates=%d feats=%d",
             len(panel), panel["market"].nunique(), panel["date"].nunique(), len(feats))

    up_labels = [f"lab_{n}" for n in UP_THRESH]
    dn_labels = [f"lab_{n}" for n in DN_THRESH] + ["lab_close_neg"]
    label_cols = up_labels + dn_labels

    # base rates (임계별, 유니버스 내 전체)
    log.info("=== base rates (universe, day-D open-anchored) ===")
    br = {}
    for lc in label_cols:
        br[lc] = float(panel[lc].mean())
        log.info("  %-14s base=%.4f", lc, br[lc])

    # ---- WF OOS heads ----
    log.info("=== walk-forward heads (XGB + per-threshold OOF bucket calib) ===")
    oos = walk_forward_heads(panel, feats, label_cols, N_FOLDS, EMBARGO)
    if oos.empty:
        log.error("no OOS folds — abort"); sys.exit(1)
    log.info("OOS rows=%d dates %s..%s", len(oos), oos["date"].min(), oos["date"].max())

    # ---- calibration reliability ----
    rel = reliability(oos, label_cols)
    rel.to_csv(OUT / "downside_head_reliability_v1.csv", index=False)
    log.info("=== OOS calibration reliability (pred vs actual, 5-bin) ===")
    for lc in label_cols:
        sub = rel[rel["label"] == lc]
        if sub.empty:
            continue
        top = sub.sort_values("bucket").iloc[-1]
        bot = sub.sort_values("bucket").iloc[0]
        log.info("  %-14s topbin pred=%.3f act=%.3f (n=%d) | botbin pred=%.3f act=%.3f",
                 lc, top["pred_mean"], top["actual"], int(top["n"]),
                 bot["pred_mean"], bot["actual"])

    # ---- risk-reward ranking compare ----
    cmp_all = [compare_rankings(oos, k) for k in TOP_KS]
    cmp = pd.concat(cmp_all, ignore_index=True)
    cols = ["policy", "k", "param", "n", "realized_pump10", "realized_pump20",
            "hit_eod_pos", "p_min_le_5", "p_min_le_10", "deep_dump_rate",
            "mean_min_ret", "median_min_ret", "mean_eod_net", "median_eod_net",
            "cvar95_eod", "worst_eod", "mdd", "cum_eod_net", "sortino", "sharpe"]
    cmp = cmp[[c for c in cols if c in cmp.columns]]
    cmp.to_csv(OUT / "downside_head_riskreward_compare_v1.csv", index=False)
    log.info("=== risk-reward vs upside-only (per K) ===")
    for k in TOP_KS:
        kk = cmp[cmp["k"] == k]
        log.info("--- K=%d ---", k)
        for _, r in kk.iterrows():
            log.info("  %-13s %-9s n=%-5d pump10=%.3f | p(min<=-5%%)=%.3f "
                     "p(min<=-10%%)=%.3f deep=%.3f | meanMin=%+.3f CVaR=%+.3f "
                     "EODnet=%+.4f cum=%+.3f sortino=%.2f",
                     r["policy"], str(r["param"]), int(r["n"]), r["realized_pump10"],
                     r["p_min_le_5"], r["p_min_le_10"], r["deep_dump_rate"],
                     r["mean_min_ret"], r["cvar95_eod"], r["mean_eod_net"],
                     r["cum_eod_net"], r["sortino"])

    # ---- OOS picks 산출 (감사용) ----
    keep = ["market", "date", "fold", "regime"] + [f"p_{l}" for l in label_cols] + \
           ["exp_downside", "med_downside", "up_high_ret", "down_low_ret", "eod_ret"]
    keep = [c for c in keep if c in oos.columns]
    oos[keep].to_csv(OUT / "downside_head_oos_picks_v1.csv", index=False)

    # ---- 오늘 scan ----
    log.info("=== today scan asof=%s ===", args.asof)
    scan = today_scan(panel, feats, label_cols, pd.Timestamp(args.asof).date())
    if not scan.empty:
        scan.to_csv(OUT / "downside_head_today_v1.csv", index=False)
        up = f"p_lab_{UP_ANCHOR}"; dn = f"p_lab_{DN_ANCHOR}"
        log.info("today universe n=%d. top-10 by rr_pen(lam=2):", len(scan))
        for _, r in scan.head(10).iterrows():
            log.info("  %-14s rr=%d(up=%d) P+5=%.3f P+10=%.3f P+15=%.3f P+20=%.3f | "
                     "P-3=%.3f P-5=%.3f P-10=%.3f Pcl<0=%.3f | E[min]=%+.3f med[min]=%+.3f | %s",
                     r["market"], int(r["rr_rank"]), int(r["upside_only_rank"]),
                     r["p_lab_up_05"], r["p_lab_up_10"], r["p_lab_up_15"], r["p_lab_up_20"],
                     r["p_lab_dn_03"], r["p_lab_dn_05"], r["p_lab_dn_10"], r["p_lab_close_neg"],
                     r.get("exp_downside", np.nan), r.get("med_downside", np.nan),
                     r.get("regime", "?"))

    # base rate json 저장
    with open(OUT / "downside_head_baserates_v1.json", "w") as f:
        json.dump({"base_rates": br,
                   "up_thresh": UP_THRESH, "dn_thresh": DN_THRESH,
                   "n_universe_rows": int(len(panel)),
                   "asof": args.asof}, f, indent=2)
    log.info("DONE. wrote downside_head_{reliability,riskreward_compare,oos_picks,today,baserates}_v1.*")


if __name__ == "__main__":
    main()

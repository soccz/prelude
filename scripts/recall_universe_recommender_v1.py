"""Recall-maximizing recommender evaluation (Angle B).

목표 재설정(사용자): 이 시스템은 사람이 판단해 매매하는 *추천 레이더*다.
1순위 = 추천 품질(급등을 잘 찾기): precision@K / recall / 급등확률 calibration.
net-PnL/Sharpe는 사용자가 직접 청산하므로 사후 참고만(net 음수가 추천을 막지 않음).
leak 방어는 여전히 양보 X.

이 스크립트가 답하는 4가지:
  1. 동적 유니버스: 정적 top100 vs (top100 OR qv_surge_7d>=3x).
     유니버스밖 펌프를 얼마나 회수(recall)하나 + 후보수(cand) trade-off.
  2. 언제 더 추천? regime(bear_volatile 펌프 최다) + breadth clustering
     (어제 풍부->오늘 풍부)으로 K를 조절하면 precision@K/recall이 오르나.
  3. precision-recall 커브: K(추천 개수)와 확률 임계를 바꿔가며.
  4. warm/cold 분리: 추천기가 잡는 warm 펌프 recall vs cold(첫 스파이크) 한계.

라벨(추천 적중 기준, day D = 진입일):
  pump20 = (high_D/open_D - 1) >= 0.20 (주)
  pump15 = (high_D/open_D - 1) >= 0.15 (보조)

★ LEAK 방어 (same-day leak 2번 전적):
  - 모든 feature 는 build_market_features 의 .shift(1) 결과 (D-1까지). 재사용.
  - BTC regime / breadth 는 시계열 1일 shift 후 day D 에 D-1 값만 join.
  - 추천 score = D-1 feature 의 cross-section rank 집계 (라벨은 day D, 시점 분리).
  - calibration(점수->급등확률) bucket map 은 walk-forward train fold 안에서만
    적합 -> test fold 에 적용 (OOF). selection bias 방어: 점수 가중치는 hand-pick X,
    확립된 OOS precursor 동일가중 rank-mean (조합 1개).

검증:
  - Purged walk-forward (embargo 5d), 전역 fold 경계 공유.
  - precision@K / recall 모두 test fold(OOS)에서만 집계.
  - net PnL 은 진단치로만 (open_D 진입, TP/SL/EOD, 왕복 0.15% 차감).

사용:
    python scripts/recall_universe_recommender_v1.py
    python scripts/recall_universe_recommender_v1.py --limit-markets 60  # 개발
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.database import list_markets, load_candles
from data.market_universe import is_excluded_signal_market
from signals.features import compute_btc_features
from scripts.univariate_precursor_lift_v1 import (
    build_market_features,
    add_cross_sectional,
    expanding_purged_date_folds,
)

DB_PATH = "data/upbit_d1.db"
OUT_DIR = Path("output")
EPS = 1e-12
ROUND_TRIP_COST = 0.0015
TP_PCT = 0.10
SL_PCT = 0.05

log = logging.getLogger("recall_recommender")

# 확립된 OOS-validated precursors (univariate_precursor_lift_v1.csv, oos_lift>=2.1).
# 동일가중 rank-mean -> hand-pick/weight tuning 없음 (selection 1조합).
SCORE_FEATURES = [
    "f_qv_surge_30d",       # OOS lift ~3.0
    "f_bounce_off_7d_low",  # ~3.1 (반등)
    "f_ret_7d",             # short momentum ~2.8
    "f_ret_3d",             # ~2.9
    "f_qv_surge_7d",        # ~2.5
    "f_rv_7d",              # 변동성 ~2.5
    "f_atr_pct_14",         # ATR ~2.2
    "f_log_qv",             # 유동성 baseline
]
QV_SURGE_DYNAMIC = 3.0      # 동적 유니버스 진입 임계 (qv_surge_7d >= 3x)
TOPN_STATIC = 100           # 정적 유니버스 크기 (D-1 qv rank)
N_FOLDS = 5
EMBARGO = 5
CAL_BUCKETS = 10            # calibration 분위 버킷


# ============================================================================
# 1. Panel build (leak-free) + day-D OHLC (청산 진단용) + BTC regime/breadth
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
        feat = build_market_features(df)
        # day-D OHLC (라벨/청산용, 미래 시점). build_market_features 가 안 주므로 별도 부착.
        g = df.sort_values("timestamp").reset_index(drop=True)
        feat = feat.reset_index(drop=True)
        feat["open_D"] = g["open"].values
        feat["high_D"] = g["high"].values
        feat["low_D"] = g["low"].values
        feat["close_D"] = g["close"].values
        frames.append(feat)
        if (i + 1) % 50 == 0:
            log.info("  %d/%d markets", i + 1, len(markets))
    panel = pd.concat(frames, ignore_index=True)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel["date"] = panel["timestamp"].dt.date
    panel = panel.sort_values(["date", "market"]).reset_index(drop=True)
    return panel


def attach_btc_regime(panel: pd.DataFrame) -> pd.DataFrame:
    """BTC regime D-1 join (leak 방어 1일 shift)."""
    btc = load_candles(DB_PATH, "KRW-BTC")
    bf = compute_btc_features(btc)[["timestamp", "btc_regime"]].copy()
    bf["timestamp"] = pd.to_datetime(bf["timestamp"])
    bf = bf.sort_values("timestamp").reset_index(drop=True)
    bf["regime"] = bf["btc_regime"].shift(1)   # D-1 regime
    return panel.merge(bf[["timestamp", "regime"]], on="timestamp", how="left")


def attach_breadth_dm1(panel: pd.DataFrame) -> pd.DataFrame:
    """어제(D-1) 펌프 풍부도(breadth) 를 day D 에 부착 (clustering 신호, leak-free).

    breadth_dm1 = D-1 에 실제로 pump15 한 코인 비율. day D 의 라벨이 아니라
    이미 끝난 D-1 의 실현 사건이므로 leak 아님.
    """
    daily = panel.groupby("date").agg(
        n_live=("lab_pump15", "size"),
        n_pump15=("lab_pump15", "sum"),
        n_pump20=("lab_pump20", "sum"),
    ).reset_index()
    daily["breadth15"] = daily["n_pump15"] / daily["n_live"].clip(lower=1)
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["breadth15_dm1"] = daily["breadth15"].shift(1)
    daily["n_pump15_dm1"] = daily["n_pump15"].shift(1)
    return panel.merge(
        daily[["date", "breadth15_dm1", "n_pump15_dm1"]], on="date", how="left"
    )


# ============================================================================
# 2. 추천 점수 (D-1 feature cross-section rank-mean) + 유니버스 플래그
# ============================================================================
def add_score_and_universe(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy()
    # 각 feature 를 그날 cross-section percentile rank (높을수록 펌프 후보)
    rank_cols = []
    for f in SCORE_FEATURES:
        rc = f"rk_{f}"
        p[rc] = p.groupby("date")[f].rank(pct=True)
        rank_cols.append(rc)
    p["score"] = p[rank_cols].mean(axis=1)   # 동일가중 rank-mean (0~1)

    # 유니버스 플래그 (D-1 정보)
    p["qv_rank"] = p.groupby("date")["f_universe_qv"].rank(ascending=False, method="min")
    p["in_static"] = (p["qv_rank"] <= TOPN_STATIC).astype(int)
    p["in_dynamic"] = (
        (p["qv_rank"] <= TOPN_STATIC) | (p["f_qv_surge_7d"] >= QV_SURGE_DYNAMIC)
    ).astype(int)
    return p


# ============================================================================
# 3. Walk-forward fold 경계
# ============================================================================
def make_folds(dates: np.ndarray, n_folds: int, embargo: int):
    return expanding_purged_date_folds(dates, n_folds, embargo)


# ============================================================================
# 4. Calibration: train fold 점수->급등확률 bucket map -> test 에 적용 (OOF)
# ============================================================================
def fit_calibration(tr: pd.DataFrame, label: str, n_buckets: int):
    """train fold 안에서 score quantile bucket 별 historical hit rate."""
    try:
        tr = tr.copy()
        tr["bk"] = pd.qcut(tr["score"], n_buckets, labels=False, duplicates="drop")
    except ValueError:
        return None, None
    hit = tr.groupby("bk")[label].mean()
    edges = tr.groupby("bk")["score"].max().sort_index().values
    return hit.to_dict(), edges


def apply_calibration(te: pd.DataFrame, hit_map, edges, label_default: float):
    if hit_map is None:
        return np.full(len(te), label_default)
    bk = np.searchsorted(edges, te["score"].values, side="left")
    bk = np.clip(bk, 0, len(edges) - 1)
    return np.array([hit_map.get(int(b), label_default) for b in bk])


# ============================================================================
# 5. 평가: 유니버스별 recall + precision@K + breadth-aware K
# ============================================================================
def evaluate(panel: pd.DataFrame, label: str):
    dates = np.sort(panel["date"].unique())
    folds = make_folds(dates, N_FOLDS, EMBARGO)
    log.info("label=%s folds=%d dates=%d", label, len(folds), len(dates))

    K_LIST = [1, 2, 3, 5, 10]
    universes = ["static", "dynamic"]
    # 누적 컬렉션
    rows_pak = []          # precision@K / recall per (universe, K)
    rows_thr = []          # probability-threshold sweep
    rows_breadth = []      # breadth-aware adaptive K
    warmcold = {"warm_hit": 0, "warm_total": 0, "cold_hit": 0, "cold_total": 0}

    # 전역 test pool (OOF score 부여)
    scored_test = []

    # rare event(base ~1.7%) 이므로 calibrated 급등확률은 작은 값. 실제 스케일로.
    THRESH_LIST = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15]

    for fi, (tr_dates, te_dates) in enumerate(folds):
        tr = panel[panel["date"].isin(tr_dates)].dropna(subset=["score", label])
        te = panel[panel["date"].isin(te_dates)].dropna(subset=["score", label]).copy()
        if len(tr) < 500 or len(te) < 200:
            continue
        base = tr[label].mean()
        hit_map, edges = fit_calibration(tr, label, CAL_BUCKETS)
        te["pump_prob"] = apply_calibration(te, hit_map, edges, base)
        te["fold"] = fi
        scored_test.append(te)

    if not scored_test:
        return None
    allte = pd.concat(scored_test, ignore_index=True)

    # ---- (A) 유니버스별 recall + precision@K ----
    for uni in universes:
        flag = "in_static" if uni == "static" else "in_dynamic"
        # 전체 펌프 (유니버스 무관) = recall 분모. 유니버스밖 펌프 회수 측정.
        total_pumps = int(allte[label].sum())
        for K in K_LIST:
            hits = 0
            recommended = 0
            captured_pumps = 0
            n_days = 0
            for d, g in allte.groupby("date"):
                cand = g[g[flag] == 1]
                if cand.empty:
                    continue
                topk = cand.nlargest(K, "score")
                hits += int(topk[label].sum())
                recommended += len(topk)
                captured_pumps += int(topk[label].sum())
                n_days += 1
            prec = hits / recommended if recommended else np.nan
            recall = captured_pumps / total_pumps if total_pumps else np.nan
            rows_pak.append({
                "universe": uni, "K": K, "label": label,
                "precision_at_k": prec, "recall_all_pumps": recall,
                "n_recommended": recommended, "n_days": n_days,
                "avg_cand_per_day": recommended / n_days if n_days else np.nan,
                "total_pumps_oos": total_pumps,
            })

    # ---- (A2) 유니버스 후보 풀 자체의 recall (K=inf, 유니버스 안 모든 코인 추천 시) ----
    for uni in universes:
        flag = "in_static" if uni == "static" else "in_dynamic"
        in_uni = allte[allte[flag] == 1]
        total_pumps = int(allte[label].sum())
        rows_pak.append({
            "universe": f"{uni}_poolmax", "K": "all_in_universe", "label": label,
            "precision_at_k": in_uni[label].mean(),
            "recall_all_pumps": int(in_uni[label].sum()) / total_pumps if total_pumps else np.nan,
            "n_recommended": len(in_uni),
            "n_days": allte["date"].nunique(),
            "avg_cand_per_day": len(in_uni) / allte["date"].nunique(),
            "total_pumps_oos": total_pumps,
        })

    # ---- (B) probability-threshold sweep (dynamic universe) ----
    dyn = allte[allte["in_dynamic"] == 1]
    total_pumps = int(allte[label].sum())
    for thr in THRESH_LIST:
        sel = dyn[dyn["pump_prob"] >= thr]
        rows_thr.append({
            "label": label, "prob_threshold": thr,
            "precision": sel[label].mean() if len(sel) else np.nan,
            "recall_all_pumps": int(sel[label].sum()) / total_pumps if total_pumps else np.nan,
            "n_selected": len(sel),
            "avg_per_day": len(sel) / allte["date"].nunique(),
        })

    # ---- (C) breadth-aware adaptive K vs fixed K (dynamic universe) ----
    # 어제 풍부(breadth15_dm1 상위 tertile)면 K=5, 보통 K=3, 희박이면 K=1.
    b = allte.dropna(subset=["breadth15_dm1"])
    if len(b) > 0:
        q_hi = b["breadth15_dm1"].quantile(0.66)
        q_lo = b["breadth15_dm1"].quantile(0.33)

        def adaptive_K(bd):
            if bd >= q_hi:
                return 5
            if bd <= q_lo:
                return 1
            return 3

        for mode in ["adaptive", "fixed3"]:
            hits = recommended = 0
            for d, g in b[b["in_dynamic"] == 1].groupby("date"):
                bd = g["breadth15_dm1"].iloc[0]
                K = adaptive_K(bd) if mode == "adaptive" else 3
                topk = g.nlargest(K, "score")
                hits += int(topk[label].sum())
                recommended += len(topk)
            rows_breadth.append({
                "label": label, "mode": mode,
                "precision": hits / recommended if recommended else np.nan,
                "recall_all_pumps": hits / total_pumps if total_pumps else np.nan,
                "n_recommended": recommended,
            })

    # ---- (D) warm/cold split (dynamic universe top-3 추천 기준) ----
    # warm = D-1 에 선행 점수가 이미 높았던 펌프(잡을 수 있음).
    # cold = 선행 거의 없이 day D 첫 스파이크. score 하위면 cold proxy.
    # 추천기는 score top-K 를 추천하므로, "잡힌 펌프"는 정의상 warm.
    # warm 펌프 = (해당일 score 상위 30% 안에 든 펌프), cold = 하위 70%.
    for d, g in allte.groupby("date"):
        if g["in_dynamic"].sum() == 0:
            continue
        cand = g[g["in_dynamic"] == 1].copy()
        cut = cand["score"].quantile(0.70)
        pumps = g[g[label] == 1]
        for _, pr in pumps.iterrows():
            is_warm = (pr["in_dynamic"] == 1) and (pr["score"] >= cut)
            if is_warm:
                warmcold["warm_total"] += 1
                # top-3 안에 들었나
                top3 = cand.nlargest(3, "score")
                if pr["market"] in set(top3["market"]):
                    warmcold["warm_hit"] += 1
            else:
                warmcold["cold_total"] += 1
                top3 = cand.nlargest(3, "score")
                if pr["market"] in set(top3["market"]):
                    warmcold["cold_hit"] += 1

    return {
        "pak": rows_pak, "thr": rows_thr, "breadth": rows_breadth,
        "warmcold": warmcold, "allte": allte,
    }


# ============================================================================
# 6. 최근 N거래일 추천 회고 (dynamic universe, calibrated top-K)
#    전 데이터로 calibration 적합 후 최근일 추천 -> 실제 다음날 펌프% 대조.
#    (회고이므로 마지막 fold OOF 가 아니라 "최신일은 미래 라벨 관측" 인정 — 회고용)
# ============================================================================
def recent_recommendations(panel: pd.DataFrame, label: str, n_days: int, top_k: int):
    # 회고 전용: 아직 닫히지 않은 최신 D1 row를 miss로 세지 않는다.
    p = panel.dropna(subset=["score", label]).copy()
    dates = np.sort(p["date"].unique())
    # calibration 은 마지막 n_days 직전까지로만 적합 (추천일 D-1까지 정보 정합)
    cutoff = dates[-(n_days + 1)] if len(dates) > n_days + 1 else dates[0]
    tr = p[p["date"] <= cutoff].dropna(subset=[label])
    base = tr[label].mean()
    hit_map, edges = fit_calibration(tr, label, CAL_BUCKETS)

    recs = []
    recent_dates = dates[-n_days:]
    for d in recent_dates:
        g = p[(p["date"] == d) & (p["in_dynamic"] == 1)].copy()
        if g.empty:
            continue
        g["pump_prob"] = apply_calibration(g, hit_map, edges, base)
        topk = g.nlargest(top_k, "score")
        items = []
        for _, r in topk.iterrows():
            actual = r["intraday_high_ret"]
            hit = "HIT" if r[label] == 1 else "miss"
            actual_str = f"{actual*100:+.1f}%" if np.isfinite(actual) else "NA"
            items.append(
                f"{r['market'].replace('KRW-','')}(p={r['pump_prob']*100:.0f}% "
                f"act={actual_str} {hit})"
            )
        n_hit = int(topk[label].sum())
        recs.append({
            "date": str(d), "n_rec": len(topk), "n_hit": n_hit,
            "items": items,
        })
    return recs


# ============================================================================
# 7. net PnL 진단 (사후 참고만, 추천을 막지 않음)
# ============================================================================
def net_pnl_diag(rows: pd.DataFrame):
    trades = []
    for _, r in rows.iterrows():
        o, hi, lo, cl = r["open_D"], r["high_D"], r["low_D"], r["close_D"]
        if not np.isfinite(o) or o <= 0:
            continue
        tp_px, sl_px = o * (1 + TP_PCT), o * (1 - SL_PCT)
        if lo <= sl_px:
            gross = -SL_PCT
        elif hi >= tp_px:
            gross = TP_PCT
        else:
            gross = cl / o - 1.0
        trades.append(gross - ROUND_TRIP_COST)
    if not trades:
        return {"net_mean": np.nan, "n": 0}
    return {"net_mean": float(np.mean(trades)), "n": len(trades)}


def _configure_cli_logging() -> None:
    """CLI 전용 로깅 — import 시 root 구성 금지(07-28/29 stdout 오염 클래스).
    이 모듈은 라이브 close 경로가 import 하므로 부작용이 곧 장애다."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main():
    _configure_cli_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-markets", type=int, default=None)
    ap.add_argument("--recent-days", type=int, default=14)
    ap.add_argument("--top-k-recent", type=int, default=3)
    args = ap.parse_args()

    log.info("=== build panel ===")
    panel = build_panel(args.limit_markets)
    panel = add_cross_sectional(panel)
    panel = attach_btc_regime(panel)
    panel = attach_breadth_dm1(panel)
    panel = add_score_and_universe(panel)
    log.info("panel rows=%d markets=%d dates=%d",
             len(panel), panel["market"].nunique(), panel["date"].nunique())

    all_pak, all_thr, all_breadth = [], [], []
    summary = {}
    for label in ["lab_pump20", "lab_pump15"]:
        res = evaluate(panel, label)
        if res is None:
            log.warning("no eval for %s", label)
            continue
        all_pak.extend(res["pak"])
        all_thr.extend(res["thr"])
        all_breadth.extend(res["breadth"])
        wc = res["warmcold"]
        warm_recall = wc["warm_hit"] / wc["warm_total"] if wc["warm_total"] else np.nan
        cold_recall = wc["cold_hit"] / wc["cold_total"] if wc["cold_total"] else np.nan
        summary[label] = {
            "warm_total": wc["warm_total"], "warm_recall_top3": warm_recall,
            "cold_total": wc["cold_total"], "cold_recall_top3": cold_recall,
        }
        # net diag: dynamic top-3 추천 (OOF test pool)
        allte = res["allte"]
        top3_rows = []
        for d, g in allte[allte["in_dynamic"] == 1].groupby("date"):
            top3_rows.append(g.nlargest(3, "score"))
        if top3_rows:
            np_diag = net_pnl_diag(pd.concat(top3_rows))
            summary[label]["net_diag_top3"] = np_diag

    pak_df = pd.DataFrame(all_pak)
    thr_df = pd.DataFrame(all_thr)
    breadth_df = pd.DataFrame(all_breadth)
    pak_df.to_csv(OUT_DIR / "recall_universe_pak_v1.csv", index=False)
    thr_df.to_csv(OUT_DIR / "recall_universe_threshold_v1.csv", index=False)
    breadth_df.to_csv(OUT_DIR / "recall_universe_breadth_v1.csv", index=False)

    # 최근 추천 회고 (주 라벨 pump20)
    recs = recent_recommendations(panel, "lab_pump20", args.recent_days, args.top_k_recent)
    with open(OUT_DIR / "recall_universe_recent_v1.json", "w") as f:
        json.dump({"recent": recs, "summary": summary}, f, indent=2, default=str)

    # ---- 콘솔 요약 ----
    log.info("\n========== PRECISION@K / RECALL (OOS) ==========")
    for label in ["lab_pump20", "lab_pump15"]:
        sub = pak_df[(pak_df["label"] == label)]
        log.info("--- %s ---", label)
        for _, r in sub.iterrows():
            log.info("  uni=%-16s K=%-14s prec@K=%.4f recall=%.4f cand/day=%.2f n_rec=%d",
                     r["universe"], str(r["K"]), r["precision_at_k"],
                     r["recall_all_pumps"], r["avg_cand_per_day"], r["n_recommended"])

    log.info("\n========== PROB THRESHOLD SWEEP (dynamic) ==========")
    for label in ["lab_pump20", "lab_pump15"]:
        sub = thr_df[thr_df["label"] == label]
        log.info("--- %s ---", label)
        for _, r in sub.iterrows():
            log.info("  thr=%.2f prec=%.4f recall=%.4f n_sel=%d /day=%.2f",
                     r["prob_threshold"], r["precision"], r["recall_all_pumps"],
                     r["n_selected"], r["avg_per_day"])

    log.info("\n========== BREADTH-AWARE ADAPTIVE K vs FIXED ==========")
    for _, r in breadth_df.iterrows():
        log.info("  [%s] mode=%-9s prec=%.4f recall=%.4f n_rec=%d",
                 r["label"], r["mode"], r["precision"], r["recall_all_pumps"],
                 r["n_recommended"])

    log.info("\n========== WARM/COLD (top-3 dynamic) ==========")
    for label, s in summary.items():
        log.info("  [%s] warm n=%d recall_top3=%.3f | cold n=%d recall_top3=%.3f | net_diag=%s",
                 label, s["warm_total"], s["warm_recall_top3"],
                 s["cold_total"], s["cold_recall_top3"], s.get("net_diag_top3"))

    log.info("\n========== RECENT %dd RECOMMENDATIONS (pump20, dynamic top-%d) ==========",
             args.recent_days, args.top_k_recent)
    for r in recs:
        log.info("  %s  hit=%d/%d  %s", r["date"], r["n_hit"], r["n_rec"], ", ".join(r["items"]))

    log.info("\nDONE. selection combos tried: 1 (equal-weight rank-mean of %d established precursors)",
             len(SCORE_FEATURES))


if __name__ == "__main__":
    main()

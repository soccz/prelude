"""Upside distribution head v1 — calibrated P(next-day max(high)/open >= T) for T in a grid,
plus downside risk P(min(low)/open <= -T) and expected downside, leak-free.

사용자 refined 비전 (이 워크플로의 단일 질문):
  이진 "≥20% 펌프 ✓/✗" 가 아니라 — 코인×일별로 **상승 확률분포** (P(≥3/5/7/10/15/20/30%))
  + **하락 리스크** (P(≤-5/-10%), 기대하방) 를 calibrated 로 매기고, 검증된 선행패턴
  (rank-mean score) 상위 코인을 cross-section 비교해 "상승확률 높고 하락확률 낮은" 걸 찾는다.
  상승폭 기준(20%)도 고정 말고 **데이터로 검증** — 각 임계 base rate + 선행상위에서 P 감소.

설계:
  입력 = 검증된 leak-free 선행 rank-mean score (signals.recommend / recall recommender 와 동일
         8 feature 동일가중 cross-section pct-rank 평균, 전부 D-1 까지).
  라벨 = 다음날(=day D) 일봉 경로:
         up_T   = (high_D / open_D - 1) >= T            (T in UP_GRID)
         dn_T   = (low_D  / open_D - 1) <= -T           (T in DN_GRID)
         dd_path = (low_D / open_D - 1)                 (open→intraday-min, 기대하방의 raw)
         (모두 day-D open/high/low — 미래 = 라벨. feature 는 .shift(1) 로 D-1.)
  calibration = **per-threshold purged walk-forward OOF**:
         각 fold 의 train(과거, embargo 전) 에서 score quantile bucket 별 실제 hit rate 를 적합 →
         test fold(미래) 에 적용. 모든 OOF prob 을 모아 reliability (pred vs actual).
         rare event raw 과신 금지 — bucket historical-hit 만.
  비교 = 동일 (market,date) 에서 distribution_engine_v1 7-head (h2/h6/h7/h5) 의 OOF-free
         (final model, 단 train-period 분리는 못 하므로 in-sample 경고) prob vs
         rank-mean per-threshold calibration 의 reliability/Brier 를 같은 universe 에서 비교.

★ LEAK 방어 (이 프로젝트 same-day leak 2번 전적 — 양보 X):
  - 모든 feature = build_market_features 의 market 별 .shift(1) (D-1 까지).
  - score = 그날 cross-section rank-mean (입력이 D-1 이라 leak 아님).
  - calibration bucket map = **각 fold train(과거) 에서만** 적합, test(미래) 적용 (OOF).
    embargo 로 train 종료와 test 시작 사이 gap → 인접일 누수 차단.
  - 라벨(up_T/dn_T)은 day-D open/high/low (미래) — feature 에 절대 안 섞임.
  - 유니버스 컷 = D-1 qv top-N (f_qv_rank).
  - today scan: asof 까지만 로드, asof row feature = D-1, 라벨 미관측 (NaN).

selection 기록: UP_GRID(7) × DN_GRID(2) 임계 = 9 calibration head. score/feature/universe 는
  recommender 에서 이미 검증된 것 재사용 (새 hand-pick 없음).

self-contained (gan_t/xsec_alpha import 금지, prelude 내부 모듈만).

산출물:
  output/upside_dist_base_rates_v1.csv      — 임계별 base rate + 선행상위(decile) P 감소
  output/upside_dist_calibration_v1.csv     — 임계별 OOF reliability (bucket pred vs actual)
  output/upside_dist_vs_engine_v1.csv       — rank-mean per-threshold vs 7-head Brier 비교
  output/upside_dist_today_<asof>.csv       — 오늘 코인별 상승분포 + 하락리스크 + 랭크

사용:
    python scripts/build_upside_dist_head_v1.py --asof 2026-05-31
    python scripts/build_upside_dist_head_v1.py --asof 2026-05-31 --limit-markets 60   # 개발
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

from data.database import list_markets, load_candles  # noqa: E402
from data.market_universe import signal_eligible_markets  # noqa: E402
from scripts.regime_split_precursor_v1 import attach_btc_regime  # noqa: E402
from scripts.univariate_precursor_lift_v1 import (  # noqa: E402
    add_cross_sectional,
    build_market_features,
)

DB_PATH = "data/upbit_d1.db"
OUT_DIR = Path("output")
EPS = 1e-12

# 검증된 leak-free 선행 8 feature (recommend.py / recall recommender 와 동일 set, 재사용)
SCORE_FEATURES = [
    "f_qv_surge_30d", "f_qv_surge_7d", "f_bounce_off_7d_low",
    "f_ret_3d", "f_ret_7d", "f_rv_7d", "f_atr_pct_14", "f_log_qv",
]

UP_GRID = [0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30]   # 상승 임계 (데이터로 검증)
DN_GRID = [0.05, 0.10]                                  # 하락 임계 (사용자 -5% 기준 + -10%)
UNIVERSE_TOP_N = 100        # 정적 유니버스 = D-1 qv top100 (recommender 와 동일)
CAL_BUCKETS = 10
N_FOLDS = 5
EMBARGO_DAYS = 5
MIN_HISTORY = 70

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("upside_dist")


# ============================================================================
# 1. Panel build (leak-free) + multi-threshold up/down labels
# ============================================================================
def build_panel(asof: pd.Timestamp | None, limit_markets: int | None) -> pd.DataFrame:
    markets = signal_eligible_markets(list_markets(DB_PATH))
    if limit_markets:
        markets = markets[:limit_markets]
    frames = []
    for i, m in enumerate(markets):
        df = load_candles(DB_PATH, m)
        if df is None or len(df) < MIN_HISTORY:
            continue
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if asof is not None:
            df = df[df["timestamp"].dt.normalize() <= asof]
        if len(df) < MIN_HISTORY:
            continue
        df["market"] = m
        feat = build_market_features(df)
        # ---- multi-threshold up/down 라벨 (day-D open/high/low = 미래 = 라벨) ----
        g = df.sort_values("timestamp").reset_index(drop=True)
        o, h, l = g["open"], g["high"], g["low"]
        up_path = (h / (o + EPS) - 1.0)      # open→intraday-high
        dn_path = (l / (o + EPS) - 1.0)      # open→intraday-min (<=0 보통)
        lab = pd.DataFrame({"timestamp": g["timestamp"]})
        for t in UP_GRID:
            lab[f"lab_up_{int(t*100)}"] = (up_path >= t).astype(float)
        for t in DN_GRID:
            lab[f"lab_dn_{int(t*100)}"] = (dn_path <= -t).astype(float)
        lab["dd_path"] = dn_path             # 기대하방 raw (open→min)
        lab["eod_ret"] = (g["close"] / (o + EPS) - 1.0)
        feat = feat.merge(lab, on="timestamp", how="left")
        frames.append(feat)
        if (i + 1) % 50 == 0:
            log.info("  %d markets loaded", i + 1)
    panel = pd.concat(frames, ignore_index=True)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel["date"] = panel["timestamp"].dt.date
    panel = panel.sort_values(["date", "market"]).reset_index(drop=True)
    return panel


def add_score(panel: pd.DataFrame) -> pd.DataFrame:
    """검증된 rank-mean score (그날 cross-section pct-rank 동일가중 평균, 입력 D-1)."""
    p = panel.copy()
    rank_cols = []
    for f in SCORE_FEATURES:
        rc = f"rk_{f}"
        p[rc] = p.groupby("date")[f].rank(pct=True)
        rank_cols.append(rc)
    p["score"] = p[rank_cols].fillna(0.5).mean(axis=1)
    p["in_universe"] = (p["f_qv_rank"] <= UNIVERSE_TOP_N).fillna(False)
    return p


# ============================================================================
# 2. Per-threshold bucket calibration (train-only) + OOF walk-forward
# ============================================================================
def fit_bucket_cal(train: pd.DataFrame, label: str, n_buckets: int):
    d = train.dropna(subset=["score", label])
    if len(d) < 200 or d[label].sum() < 5:
        return None, None, (float(d[label].mean()) if len(d) else np.nan)
    try:
        bk = pd.qcut(d["score"].rank(method="first"), n_buckets, labels=False, duplicates="drop")
    except ValueError:
        return None, None, float(d[label].mean())
    d = d.assign(bk=bk)
    g = d.groupby("bk").agg(hi=("score", "max"), hit=(label, "mean")).sort_index()
    return g["hit"].to_dict(), g["hi"].values, float(d[label].mean())


def apply_bucket_cal(scores: np.ndarray, hit_map, edges, base: float) -> np.ndarray:
    if hit_map is None or edges is None:
        return np.full(len(scores), base, dtype=float)
    idx = np.searchsorted(edges, scores, side="left")
    idx = np.clip(idx, 0, len(edges) - 1)
    return np.array([hit_map.get(int(b), base) for b in idx], dtype=float)


def make_folds(dates: np.ndarray, n_folds: int, embargo: int):
    """purged walk-forward (expanding train, embargo gap, test block)."""
    dates = np.sort(dates)
    fold_size = len(dates) // (n_folds + 1)
    folds = []
    for k in range(1, n_folds + 1):
        train_end = fold_size * k
        test_start = train_end + embargo
        test_end = min(fold_size * (k + 1) + train_end, len(dates))
        if test_start >= len(dates):
            break
        folds.append((set(dates[:train_end]), set(dates[test_start:test_end])))
    return folds


def oof_calibration(panel: pd.DataFrame, label: str, universe_only: bool = True):
    """purged WF: 각 fold train 에서 bucket cal 적합 → test 에 적용. OOF (pred, actual) 수집.
    returns DataFrame[pred, actual, fold] + reliability table."""
    pool = panel[panel["in_universe"]] if universe_only else panel
    pool = pool.dropna(subset=["score", label])
    dates = pool["date"].unique()
    folds = make_folds(dates, N_FOLDS, EMBARGO_DAYS)
    oof = []
    for fi, (tr_d, te_d) in enumerate(folds, 1):
        tr = pool[pool["date"].isin(tr_d)]
        te = pool[pool["date"].isin(te_d)]
        if len(tr) < 200 or len(te) < 50:
            continue
        hm, ed, base = fit_bucket_cal(tr, label, CAL_BUCKETS)
        pred = apply_bucket_cal(te["score"].values, hm, ed, base)
        oof.append(pd.DataFrame({"pred": pred, "actual": te[label].values, "fold": fi}))
    if not oof:
        return None, None
    oof = pd.concat(oof, ignore_index=True)
    return oof, reliability_table(oof)


def reliability_table(oof: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """pred 를 10 bin → bin 별 mean(pred) vs mean(actual). calibration reliability."""
    d = oof.dropna(subset=["pred", "actual"]).copy()
    if len(d) < 50:
        return pd.DataFrame()
    try:
        d["pb"] = pd.qcut(d["pred"].rank(method="first"), n_bins, labels=False, duplicates="drop")
    except ValueError:
        d["pb"] = 0
    g = d.groupby("pb").agg(pred_mean=("pred", "mean"), actual_mean=("actual", "mean"),
                            n=("actual", "size")).reset_index()
    return g


def brier(oof: pd.DataFrame) -> float:
    d = oof.dropna(subset=["pred", "actual"])
    if len(d) == 0:
        return np.nan
    return float(((d["pred"] - d["actual"]) ** 2).mean())


# ============================================================================
# 3. 7-head engine 비교 (같은 universe 에서 Brier — engine 은 final model 이라 in-sample 경고)
# ============================================================================
def compare_engine(panel: pd.DataFrame) -> pd.DataFrame:
    """distribution_engine_v1 7-head 의 head 별 prob 을 동일 universe 에서 매겨,
    가장 가까운 우리 임계 라벨에 대한 Brier 와 rank-mean OOF Brier 를 비교.

    주의: engine final model 은 full panel 학습이라 이 비교의 engine 쪽은 in-sample
    (optimistic). 그럼에도 'rank-mean per-threshold 가 적어도 in-sample engine 만큼
    calibrated 한가' 의 sanity check 로 본다 (quant-evaluator 가 OOF engine 재학습으로 확정)."""
    rows = []
    try:
        from signals.distribution_engine import DistributionEngine
        from signals.features import assemble_training_panel, compute_btc_features
    except Exception as e:
        log.warning("engine compare skipped (import fail): %s", e)
        return pd.DataFrame()
    # engine head ↔ 우리 임계 대응 (T% hit, H 무관하게 daily high 로 근사 비교)
    head_to_thresh = {
        "p_h2_hit_3_4h": "lab_up_3",
        "p_h6_hit_5_24h": "lab_up_5",
        "p_h7_mid_7_24h": "lab_up_7",
        "p_h5_tail_20": "lab_up_20",
    }
    try:
        eng = DistributionEngine.load()
    except Exception as e:
        log.warning("engine load fail: %s — compare skipped", e)
        return pd.DataFrame()

    # engine 은 자체 feature set 이 필요 → assemble_training_panel 로 별도 빌드 후 우리 라벨 join.
    try:
        krw = signal_eligible_markets(list_markets(DB_PATH))
        candles = {m: load_candles(DB_PATH, m) for m in krw}
        candles = {k: v for k, v in candles.items() if v is not None and len(v) > 30}
        btc = load_candles(DB_PATH, "KRW-BTC")
        epanel = assemble_training_panel(candles, btc, normalize=True)
        epanel["timestamp"] = pd.to_datetime(epanel["timestamp"])
        epanel["date"] = epanel["timestamp"].dt.date
    except Exception as e:
        log.warning("engine feature panel fail: %s — compare skipped", e)
        return pd.DataFrame()

    missing = [c for c in eng.feature_cols if c not in epanel.columns]
    if missing:
        log.warning("engine missing %d feature cols (%s...) — compare skipped",
                    len(missing), missing[:3])
        return pd.DataFrame()
    scored = eng.score_panel(epanel)
    # universe filter (top100 by quote_volume)
    if "quote_volume" in scored.columns:
        scored["liq_rank"] = scored.groupby("date")["quote_volume"].rank(
            ascending=False, method="min")
        scored = scored[scored["liq_rank"] <= UNIVERSE_TOP_N]

    merged = scored.merge(
        panel[["market", "date", "score", "in_universe"] + list(set(head_to_thresh.values()))],
        on=["market", "date"], how="inner")
    merged = merged[merged["in_universe"]]

    for head, lab in head_to_thresh.items():
        if head not in merged.columns:
            continue
        d = merged.dropna(subset=[head, lab])
        if len(d) < 100:
            continue
        eng_brier = float(((d[head] - d[lab]) ** 2).mean())
        # rank-mean OOF brier for the same threshold (전체 panel OOF)
        oof, _ = oof_calibration(panel, lab)
        rm_brier = brier(oof) if oof is not None else np.nan
        rows.append({
            "engine_head": head, "threshold": lab, "n": int(len(d)),
            "base_rate": float(d[lab].mean()),
            "engine_brier_INSAMPLE": eng_brier,
            "rankmean_brier_OOF": rm_brier,
            "engine_pred_mean": float(d[head].mean()),
            "actual_mean": float(d[lab].mean()),
        })
    return pd.DataFrame(rows)


# ============================================================================
# 4. Base rate + 선행상위(decile) P 감소 (상승폭 데이터 검증)
# ============================================================================
def base_rate_table(panel: pd.DataFrame) -> pd.DataFrame:
    """임계별 universe base rate + score top-decile 에서 P. '20% 고정' 검증 — 어느 임계까지
    선행상위가 base 대비 의미있는 lift 를 주나."""
    pool = panel[panel["in_universe"]].copy()
    # top-decile = 그날 cross-section score 상위 10%
    pool["score_decile"] = pool.groupby("date")["score"].rank(pct=True)
    top = pool[pool["score_decile"] >= 0.90]
    rows = []
    for t in UP_GRID:
        lab = f"lab_up_{int(t*100)}"
        base = float(pool[lab].mean())
        topd = float(top[lab].mean())
        rows.append({
            "threshold_pct": int(t * 100), "kind": "up",
            "base_rate": base, "top_decile_P": topd,
            "lift": (topd / base) if base > 0 else np.nan,
            "n_label": int(pool[lab].notna().sum()),
        })
    for t in DN_GRID:
        lab = f"lab_dn_{int(t*100)}"
        base = float(pool[lab].mean())
        topd = float(top[lab].mean())
        rows.append({
            "threshold_pct": int(t * 100), "kind": "down",
            "base_rate": base, "top_decile_P": topd,
            "lift": (topd / base) if base > 0 else np.nan,
            "n_label": int(pool[lab].notna().sum()),
        })
    return pd.DataFrame(rows)


# ============================================================================
# 5. Today scan — 코인별 상승분포 + 하락리스크 + cross-section 랭크
# ============================================================================
def today_scan(panel: pd.DataFrame, asof_date) -> pd.DataFrame:
    """asof 까지 전체로 임계별 bucket cal 적합(< asof-embargo) → asof universe 코인에 적용.
    상승확률분포 + 하락확률 + 기대하방 + 종합랭크 (상승확률 높음 AND 하락확률 낮음)."""
    cutoff = pd.Timestamp(asof_date) - pd.Timedelta(days=EMBARGO_DAYS)
    cutoff = cutoff.date()
    train = panel[(panel["date"] < cutoff) & panel["in_universe"]].copy()
    today = panel[(panel["date"] == asof_date) & panel["in_universe"]].copy()
    today = today.dropna(subset=["score"]).copy()
    if today.empty:
        return pd.DataFrame()

    out = today[["market", "score"]].copy()
    out["score"] = out["score"].round(4)

    # 상승확률분포 (임계별 calibrated)
    for t in UP_GRID:
        lab = f"lab_up_{int(t*100)}"
        hm, ed, base = fit_bucket_cal(train, lab, CAL_BUCKETS)
        out[f"P_up_{int(t*100)}"] = apply_bucket_cal(today["score"].values, hm, ed, base)
    # 하락확률 (임계별 calibrated)
    for t in DN_GRID:
        lab = f"lab_dn_{int(t*100)}"
        hm, ed, base = fit_bucket_cal(train, lab, CAL_BUCKETS)
        out[f"P_dn_{int(t*100)}"] = apply_bucket_cal(today["score"].values, hm, ed, base)
    # 기대하방 (open→min 의 score-bucket 평균; train 에서 bucket→mean(dd_path))
    out["exp_downside"] = _expected_downside(train, today)

    # ─────────────────────────────────────────────────────────────────────
    # 랭크 설계 (★ 중요 발견 반영):
    # 검증된 선행 score 는 '움직임' 을 산다 — 상위 score 코인은 P_up_5 高(42%)지만
    # P_dn_5 도 高(48%). score 자체는 상방-only 를 분리하지 못한다(대칭 변동성 선택).
    # net_edge = P_up - P_dn 로만 정렬하면 '조용한 저-score' 코인이 위로 와 사용자
    # 의도(상승확률 높음)와 반대가 된다. 그래서:
    #   - 1차 랭크(rank): 검증된 선행 score 내림차순 = recommend.py 와 동일한 cross-section
    #     ordering (상방 잠재력의 proven proxy). 분포는 magnitude readout 으로 부착.
    #   - 보조 랭크(rank_dsadj): 사용자 위험선호(하락 최소화 우선, -5% 수용) 반영 —
    #     downside-penalized upside index = P_up_5 - LAMBDA_DN * P_dn_10 (deep-loss 만 벌점).
    #     LAMBDA_DN=1.0 placeholder. quant-evaluator 가 ledger 경로로 lambda 검증.
    LAMBDA_DN = 1.0
    out["ds_adj_index"] = out["P_up_5"] - LAMBDA_DN * out["P_dn_10"]
    out["net_edge_5"] = out["P_up_5"] - out["P_dn_5"]   # 진단용(상-하 대칭성 노출)

    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    out.insert(1, "rank", np.arange(1, len(out) + 1))
    out["rank_dsadj"] = out["ds_adj_index"].rank(ascending=False, method="min").astype(int)
    return out


def _expected_downside(train: pd.DataFrame, today: pd.DataFrame) -> np.ndarray:
    """score-bucket 별 mean(dd_path) (open→min) 을 train 에서 적합 → today 적용. 기대하방(<=0)."""
    d = train.dropna(subset=["score", "dd_path"])
    if len(d) < 200:
        base = float(d["dd_path"].mean()) if len(d) else 0.0
        return np.full(len(today), base)
    try:
        bk = pd.qcut(d["score"].rank(method="first"), CAL_BUCKETS, labels=False, duplicates="drop")
    except ValueError:
        return np.full(len(today), float(d["dd_path"].mean()))
    d = d.assign(bk=bk)
    g = d.groupby("bk").agg(hi=("score", "max"), m=("dd_path", "mean")).sort_index()
    idx = np.searchsorted(g["hi"].values, today["score"].values, side="left")
    idx = np.clip(idx, 0, len(g) - 1)
    return np.array([g["m"].to_dict().get(int(b), d["dd_path"].mean()) for b in idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", type=str, default="2026-05-31")
    ap.add_argument("--limit-markets", type=int, default=None)
    args = ap.parse_args()
    OUT_DIR.mkdir(exist_ok=True)

    asof = pd.Timestamp(args.asof).normalize()
    log.info("building leak-free panel up to %s ...", asof.date())
    panel = build_panel(asof, args.limit_markets)
    panel = add_cross_sectional(panel)
    panel = attach_btc_regime(panel)
    panel = add_score(panel)
    log.info("panel rows=%d markets=%d dates=%d universe_rows=%d",
             len(panel), panel["market"].nunique(), panel["date"].nunique(),
             int(panel["in_universe"].sum()))

    n_heads = len(UP_GRID) + len(DN_GRID)
    log.info("SELECTION: %d calibration heads (UP %d + DN %d), reused validated 8-feat score "
             "(no new hand-pick)", n_heads, len(UP_GRID), len(DN_GRID))

    # --- (A) base rate + top-decile P 감소 (상승폭 데이터 검증) ---
    br = base_rate_table(panel)
    br.to_csv(OUT_DIR / "upside_dist_base_rates_v1.csv", index=False)
    log.info("=== base rate + top-decile P (universe top%d) ===", UNIVERSE_TOP_N)
    for _, r in br.iterrows():
        log.info("  %-4s %3d%%: base=%.4f top-decile=%.4f lift=%.2f (n=%d)",
                 r["kind"], int(r["threshold_pct"]), r["base_rate"], r["top_decile_P"],
                 r["lift"], int(r["n_label"]))

    # --- (B) per-threshold OOF calibration reliability ---
    cal_rows = []
    log.info("=== OOF calibration reliability (purged WF %d-fold, embargo %dd) ===",
             N_FOLDS, EMBARGO_DAYS)
    for t in UP_GRID:
        lab = f"lab_up_{int(t*100)}"
        oof, rel = oof_calibration(panel, lab)
        if oof is None:
            log.info("  up_%d%%: insufficient OOF", int(t*100))
            continue
        b = brier(oof)
        top_bin = rel.iloc[-1] if len(rel) else None
        log.info("  up_%-3d%% OOF n=%d Brier=%.4f | top-bin pred=%.3f actual=%.3f",
                 int(t*100), len(oof), b,
                 top_bin["pred_mean"] if top_bin is not None else np.nan,
                 top_bin["actual_mean"] if top_bin is not None else np.nan)
        for _, rr in rel.iterrows():
            cal_rows.append({"threshold": lab, "kind": "up", "bin": int(rr["pb"]),
                             "pred_mean": rr["pred_mean"], "actual_mean": rr["actual_mean"],
                             "n": int(rr["n"]), "brier": b})
    for t in DN_GRID:
        lab = f"lab_dn_{int(t*100)}"
        oof, rel = oof_calibration(panel, lab)
        if oof is None:
            log.info("  dn_%d%%: insufficient OOF", int(t*100))
            continue
        b = brier(oof)
        top_bin = rel.iloc[-1] if len(rel) else None
        log.info("  dn_%-3d%% OOF n=%d Brier=%.4f | top-bin pred=%.3f actual=%.3f",
                 int(t*100), len(oof), b,
                 top_bin["pred_mean"] if top_bin is not None else np.nan,
                 top_bin["actual_mean"] if top_bin is not None else np.nan)
        for _, rr in rel.iterrows():
            cal_rows.append({"threshold": lab, "kind": "down", "bin": int(rr["pb"]),
                             "pred_mean": rr["pred_mean"], "actual_mean": rr["actual_mean"],
                             "n": int(rr["n"]), "brier": b})
    pd.DataFrame(cal_rows).to_csv(OUT_DIR / "upside_dist_calibration_v1.csv", index=False)

    # --- (C) 7-head engine 비교 ---
    cmp = compare_engine(panel)
    if len(cmp):
        cmp.to_csv(OUT_DIR / "upside_dist_vs_engine_v1.csv", index=False)
        log.info("=== rank-mean per-threshold OOF vs 7-head engine (engine=IN-SAMPLE) ===")
        for _, r in cmp.iterrows():
            log.info("  %-18s %-9s base=%.3f | engine_brier(IS)=%.4f rankmean_brier(OOF)=%.4f"
                     " | engine_pred=%.3f actual=%.3f",
                     r["engine_head"], r["threshold"], r["base_rate"],
                     r["engine_brier_INSAMPLE"], r["rankmean_brier_OOF"],
                     r["engine_pred_mean"], r["actual_mean"])
    else:
        log.info("engine compare produced no rows (skipped/insufficient)")

    # --- (D) today scan ---
    asof_date = asof.date()
    if asof_date not in set(panel["date"]):
        log.error("asof %s not in panel (DB stale?)", asof_date)
    else:
        ts = today_scan(panel, asof_date)
        ts_path = OUT_DIR / f"upside_dist_today_{asof.strftime('%Y%m%d')}.csv"
        ts.to_csv(ts_path, index=False)
        log.info("=== TODAY %s — top-10 by validated precursor score ===", asof_date)
        for _, r in ts.head(10).iterrows():
            log.info("  #%d %-12s sc=%.3f | up5=%.3f up10=%.3f up15=%.3f up20=%.3f | "
                     "dn5=%.3f dn10=%.3f expDD=%.3f | dsadj#%d",
                     int(r["rank"]), r["market"], r["score"],
                     r.get("P_up_5", np.nan), r.get("P_up_10", np.nan),
                     r.get("P_up_15", np.nan), r.get("P_up_20", np.nan),
                     r.get("P_dn_5", np.nan), r.get("P_dn_10", np.nan),
                     r.get("exp_downside", np.nan), int(r.get("rank_dsadj", 0)))
        # ★ 상-하 대칭성 진단: 상위 score 코인의 평균 P_up_5 vs P_dn_5
        topq = ts.head(max(5, len(ts) // 10))
        log.info("SYMMETRY: top-score decile mean P_up_5=%.3f vs P_dn_5=%.3f "
                 "(검증된 선행은 변동성을 산다 — 상방-only 분리 못함)",
                 float(topq["P_up_5"].mean()), float(topq["P_dn_5"].mean()))
        log.info("wrote %s", ts_path)

    log.info("DONE.")


if __name__ == "__main__":
    main()

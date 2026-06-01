"""A3 challenger — richer D-1 context features for the entry head (NEW file only).

트랙 B3 (signal-researcher, 병렬 안전):
  챔피언 R1 의 entry head 는 코인-레벨 D-1 feature(PRECURSOR_FEATURES)만 본다.
  market 이 "지금 펌프-풍부/follow-through 장인가"라는 **시장-레벨 D-1 맥락**은 head 에
  전혀 안 들어간다(BTC regime 도 head feature set 엔 없음). A3 가설:
    pump-then-dump 의 상당수는 "죽은 장에서 혼자 솟구친" 고립 펌프다. 같은 entry
    확률이라도 시장이 광범위하게 상승 중(브레드스 높음)일 때의 펌프가 follow-through
    하고, 고립 펌프는 dump 한다. 따라서 이미 검증된 **시장-브레드스 D-1 맥락 feature**
    를 head feature 에 추가하면 진입 quality 가 올라가 net 이 개선될 수 있다.

  ★ 왜 시장-브레드스인가 (redundancy 회피):
    - liquidity_dynamics(qv_ratio_med_*, qv_zscore_30d, qv_rank_xs ...) 는 OOS lift 가
      크지만 R1 head 가 이미 f_qv_surge_30d/7d, f_qv_ma7_vs_ma30, f_log_qv, f_qv_rank_pct
      를 갖고 있어 **코인-레벨 유동성 축과 강하게 중복**된다(새 정보 적음).
    - market_breadth 는 head 에 전혀 없는 **시장-레벨(날짜당 1행) follow-through 축**이라
      진짜 새 정보다. market_breadth_feature_lift_v1.csv 에서 dir=high OOS lift 가
      breadth20_ma3 2.40 / breadth20_lag1 2.53 / qv_trend_7_30 2.17 / qv_zscore_30 2.02
      / breadth20_mom_3_14 1.86 등으로 살아있다(5/5 fold).
    - 그래도 A3 는 (a) breadth-only, (b) breadth+liq 두 변형을 모두 비교해 어느 축이
      net 에 lift 를 주는지(아니면 둘 다 noise 인지) 데이터로 가른다.

설계 (R1 baseline 과 byte-동일 비교 — 정렬키·청산경로·유니버스·OOS fold 공유):
  - panel/head/calibration/15m 청산경로/top-3 정렬(R1 ratio)은 r2_challenger_compare_v1
    및 downside_head_riskreward_v1 의 검증 빌더를 그대로 재사용.
  - 유일한 차이 = **head 의 feature 행렬에 새 D-1 맥락 feature 를 합친다**.
  - variant 비교: BASE(R1 기존 피처) / +BREADTH / +LIQ / +BREADTH+LIQ.
  - 같은 OOS·net 으로 net_mean·Sharpe·Sortino·MaxDD·hit·precision@3·deep-loss(%SL,
    no-SL) + 픽수/커버리지 + feature 기여도(gain importance)를 출력.

★ LEAK 방어 (same-day leak 2번 전적 — 양보 X, A3 핵심 감사 포인트):
  - market_breadth: build_market_features(scripts.market_breadth_discovery_v1) 가 모든
    raw 시계열을 .shift(1) 후 rolling → row(date=D) 의 breadth feature 는 D-1 까지만.
    그 frame 을 coin panel 에 **date 로 join** → (market,D) row 에 시장 D-1 맥락 부착.
    추가 shift 불필요(빌더 내부에서 이미 D-1). adv_ratio_raw/total_qv_raw(D 값)은
    절대 직접 join 안 함 — *_lag1 / *_ma* (전부 shift 된) 만 사용.
  - liquidity: build_features(scripts.liquidity_dynamics_discovery_v1) 의 feature 는
    "그 행 t 까지". 따라서 (market,D) 의 D-1 맥락은 **그 코인의 D-1 행** feature 다.
    → market 별로 .shift(1) 해서 D row 에 D-1 행 feature 를 붙인다(명시적 shift). qv_rank_xs
    같은 cross-sectional 도 D-1 행 값을 쓰므로 미래 안 봄.
  - 라벨(up/down/pump20) = day-D high/low/close (미래) — head 학습 feature 에 안 섞임.
  - per-fold OOF bucket calibration (train-only). test fold 적용만.
  - "성능이 너무 좋으면 leak 의심" — variant net 이 BASE 보다 비현실적으로 좋으면
    새 feature 시점부터 재감사(노트에 명시).

self-contained: prelude 내부 모듈만 import (gan_t/xsec_alpha/fin import 0).
NEW 파일만 — 공유 라이브(recommend.py/model_registry.py/daily_*.sh) 편집 X.

사용:
    python scripts/ch_features_v1.py
    python scripts/ch_features_v1.py --limit-markets 80   # 개발
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

# R1 과 동일한 leak-free head 파이프 (recommend.py 가 쓰는 바로 그 빌더).
from scripts.downside_head_riskreward_v1 import (  # noqa: E402
    build_panel,
    add_cross_sectional,
    attach_btc_regime,
    _feats,
    _xgb_fit_predict,
    _oof_bucket_calib,
    _apply_calib,
    UP_THRESH,
    DN_THRESH,
    CAL_BUCKETS,
)
# 라이브 ledger 와 동일한 15m SL/TP/EOD 경로청산.
from scripts.recommender_downside_exit_v1 import simulate_path  # noqa: E402
# 새 D-1 맥락 feature 소스 (검증된 leak-free 빌더 — 그대로 재사용, 수정 X).
import scripts.market_breadth_discovery_v1 as mb  # noqa: E402
import scripts.liquidity_dynamics_discovery_v1 as liq  # noqa: E402

D1_DB = str(_ROOT / "data" / "upbit_d1.db")
M15_DB = str(_ROOT / "data" / "upbit_15m.db")
OUT = _ROOT / "output"

ROUND_TRIP_COST = 0.0015
HARD_SL = 0.03
TP = 0.05
DEEP_LOSS = -0.05

UP_ANCHOR = "lab_up_10"     # P(>=+10%) numerator
DN_ANCHOR = "lab_dn_05"     # P(<=-5%) denominator (R1 ratio)
RR_EPS = 1e-3

TOP_K = 3
UNIVERSE_TOP_N = 100        # 라이브 R1 유니버스 = 정적 top100 (recommend.py 와 동일)
N_FOLDS = 6
EMBARGO = 5

# 새 시장-레벨 브레드스 D-1 feature (전부 빌더에서 .shift(1) 됨 → date join 안전).
# market_breadth_feature_lift_v1 의 dir=high OOS lift 상위 + 비중복 축.
BREADTH_FEATS = [
    "breadth20_lag1", "breadth20_ma3", "breadth20_ma7",
    "qv_trend_7_30", "qv_zscore_30", "breadth20_mom_3_14",
    "n_pump20_ma7", "adv_ratio_ma7",
]
# 새 코인-레벨 유동성 D-1 feature (market 별 명시 shift(1)). 중복 우려 있어 별도 변형.
LIQ_FEATS = [
    "qv_zscore_30d", "qv_ratio_med_14d", "qv_ratio_med_max_3d",
    "qv_rank_jump_3d", "qv_consec_up",
]

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("ch_features")


# ============================================================================
# 1. 새 D-1 맥락 feature 빌드 (전부 leak-free — 빌더 재사용)
# ============================================================================
def build_breadth_features() -> pd.DataFrame:
    """시장-레벨(날짜당 1행) D-1 브레드스 맥락. 모든 feature 는 빌더가 .shift(1) →
    row(date=D) 값은 D-1 까지. date 로 coin panel 에 join (추가 shift 불필요)."""
    df = mb.load_candles()
    g = mb.build_coin_pump_flags(df)
    panel = mb.build_daily_panel(g)
    btc = mb.build_btc_features(df)
    p = mb.build_market_features(panel, btc)   # *_lag1 / *_ma* 전부 D-1
    present = [c for c in BREADTH_FEATS if c in p.columns]
    out = p[["date"] + present].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    missing = [c for c in BREADTH_FEATS if c not in p.columns]
    if missing:
        log.warning("breadth feats missing in builder: %s", missing)
    return out.rename(columns={c: f"mb_{c}" for c in present})


def build_liq_features() -> pd.DataFrame:
    """코인-레벨 D-1 유동성 맥락. liq.build_features 의 feature 는 '그 행 t 까지'이므로
    (market,D) 의 D-1 맥락 = 그 코인의 **D-1 행** feature → market 별 .shift(1)."""
    df = liq.load_all(Path(D1_DB))
    feat = liq.build_features(df)   # row t = t 까지 feature
    feat = feat.sort_values(["market", "timestamp"]).reset_index(drop=True)
    present = [c for c in LIQ_FEATS if c in feat.columns]
    f = feat[["market", "timestamp"] + present].copy()
    # D row 에 D-1 행 feature 를 붙임 (명시적 leak shift). per-market shift(1).
    for c in present:
        f[f"liq_{c}"] = f.groupby("market")[c].shift(1)
    f["date"] = pd.to_datetime(f["timestamp"]).dt.date
    out = f[["market", "date"] + [f"liq_{c}" for c in present]]
    missing = [c for c in LIQ_FEATS if c not in feat.columns]
    if missing:
        log.warning("liq feats missing in builder: %s", missing)
    return out


def attach_context(panel: pd.DataFrame, breadth: pd.DataFrame,
                   liq_df: pd.DataFrame) -> tuple[pd.DataFrame, list, list]:
    """coin panel 에 시장-브레드스(date join) + 코인-유동성(market+date join) 부착.
    반환: (panel, mb_cols, liq_cols)."""
    p = panel.copy()
    p["date"] = pd.to_datetime(p["date"]).dt.date
    mb_cols = [c for c in breadth.columns if c.startswith("mb_")]
    liq_cols = [c for c in liq_df.columns if c.startswith("liq_")]
    p = p.merge(breadth, on="date", how="left")
    p = p.merge(liq_df, on=["market", "date"], how="left")
    return p, mb_cols, liq_cols


# ============================================================================
# 2. variant 별 head walk-forward (feature set 만 다름 — 나머지 동일)
# ============================================================================
def walk_forward_heads_feats(panel, feats, label_cols, n_folds, embargo):
    """downside_head_riskreward_v1.walk_forward_heads 와 동일 규율의 WF.
    유일한 차이: feats(=feature 행렬 컬럼)를 인자로 받아 variant 간 교체.
    per-fold OOF bucket calibration(train-only), test fold 적용만 — leak X."""
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
                continue
            raw_tr = m.predict_proba(Xtr)[:, 1]
            ed, hm, base = _oof_bucket_calib(raw_tr, y, CAL_BUCKETS)
            res[f"p_{lc}"] = _apply_calib(raw_te, ed, hm, base)
        out.append(res)
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


def feature_importance(panel, feats, label_col, n_folds, embargo) -> dict:
    """마지막 fold train 으로 단일 head(anchor up_10)의 gain importance 산출.
    새 feature 가 실제로 split 에 쓰이나(기여) 진단용 — 정렬효과와 별개."""
    import xgboost as xgb
    dates = np.sort(panel["date"].unique())
    n = len(dates)
    start = int(n * 0.35)
    edges_f = np.linspace(start, n, n_folds + 1).astype(int)
    tr_end = edges_f[-2]   # 마지막 fold 의 train 끝
    tr_d = set(dates[:tr_end])
    tr = panel[panel["date"].isin(tr_d)].copy()
    Xtr = tr[feats].replace([np.inf, -np.inf], np.nan)
    Xtr = Xtr.fillna(Xtr.median())
    y = tr[label_col].values
    pos = y.sum()
    if pos < 12:
        return {}
    spw = float((len(y) - pos) / max(pos, 1))
    m = xgb.XGBClassifier(
        n_estimators=180, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_lambda=1.5, scale_pos_weight=spw, n_jobs=4,
        eval_metric="logloss", tree_method="hist", random_state=42)
    m.fit(Xtr.values, y)
    booster = m.get_booster()
    booster.feature_names = list(feats)
    gain = booster.get_score(importance_type="gain")
    tot = sum(gain.values()) or 1.0
    return {f: gain.get(f, 0.0) / tot for f in feats}


# ============================================================================
# 3. 15m 경로 net (라이브 ledger 손익경로) — r2_challenger_compare_v1 와 동일
# ============================================================================
def load_paths(pairs: pd.DataFrame) -> dict:
    conn = sqlite3.connect(M15_DB)
    paths = {}
    for _, r in pairs.iterrows():
        m, dt = r["market"], pd.Timestamp(r["date"])
        s = dt.strftime("%Y-%m-%d 09:00:00")
        e = (dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d 09:00:00")
        rows = conn.execute(
            "SELECT open,high,low,close FROM candles WHERE market=? AND "
            "timestamp>=? AND timestamp<? ORDER BY timestamp", (m, s, e)).fetchall()
        if rows:
            paths[(m, dt.date())] = rows
    conn.close()
    return paths


def realize_net(bars: list):
    gross, outcome = simulate_path(bars, HARD_SL, TP, None)
    eod_gross, _ = simulate_path(bars, None, None, None)
    if not np.isfinite(gross):
        return np.nan, "nodata", np.nan
    eod_net = (eod_gross - ROUND_TRIP_COST) if np.isfinite(eod_gross) else np.nan
    return gross - ROUND_TRIP_COST, outcome, eod_net


def net_metrics(trades: pd.DataFrame) -> dict:
    d = trades.dropna(subset=["net"]).copy()
    n = len(d)
    if n == 0:
        return {}
    net = d["net"].values
    daily = d.groupby("date")["net"].mean().sort_index()
    eq = (1 + daily).cumprod(); peak = eq.cummax()
    mdd = float(((eq - peak) / peak).min())
    mu = float(net.mean()); sd = float(daily.std())
    dstd = float(daily[daily < 0].std()) if (daily < 0).any() else np.nan
    k5 = max(1, int(np.ceil(0.05 * n)))
    cvar95 = float(np.sort(net)[:k5].mean())
    oc = d["outcome"]
    eod = d["eod_net"].dropna().values if "eod_net" in d.columns else np.array([])
    return dict(
        n=int(n), n_days=int(d["date"].nunique()),
        pct_sl=float((oc == "sl").mean()),
        deep_loss_freq_noSL=float((eod <= DEEP_LOSS).mean()) if len(eod) else np.nan,
        net_mean=mu,
        hit=float((net > 0).mean()),
        precision_pump20=float(d["pump20_hit"].dropna().mean())
        if d["pump20_hit"].notna().any() else np.nan,
        deep_loss_freq=float((net <= DEEP_LOSS).mean()),
        net_median=float(np.median(net)),
        cvar95=cvar95, worst=float(net.min()), mdd=mdd,
        cum=float(eq.iloc[-1] - 1.0),
        sharpe=float(mu / sd * np.sqrt(365)) if sd and sd > 0 else np.nan,
        sortino=float(mu / dstd * np.sqrt(365)) if dstd and dstd > 0 else np.nan,
        pct_tp=float((oc == "tp").mean()), pct_eod=float((oc == "eod").mean()),
    )


def topk_picks(oos: pd.DataFrame, score_col: str, k: int) -> pd.DataFrame:
    """R1 정렬(ratio 내림차순) + 하방-우선 tie-break (recommend.py 와 동일 정신)."""
    s = oos.sort_values(
        ["date", score_col, "p_lab_dn_10", "p_lab_up_10", "exp_downside"],
        ascending=[True, False, True, False, False])
    return s.groupby("date").head(k).copy()


# ============================================================================
# main
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-markets", type=int, default=None)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    # --- 1) R1 과 동일한 leak-free coin panel + head 라벨 ---
    panel = build_panel(args.limit_markets)
    panel = add_cross_sectional(panel)
    panel = attach_btc_regime(panel)
    panel = panel[panel["f_qv_rank"] <= UNIVERSE_TOP_N].copy()
    panel["in_universe"] = True
    base_feats = _feats(panel)
    log.info("base panel(top%d) rows=%d markets=%d dates=%d base_feats=%d",
             UNIVERSE_TOP_N, len(panel), panel["market"].nunique(),
             panel["date"].nunique(), len(base_feats))

    # --- 2) 새 D-1 맥락 feature 부착 (leak-free 빌더 재사용) ---
    log.info("=== building NEW D-1 context features (breadth + liquidity) ===")
    breadth = build_breadth_features()
    liq_df = build_liq_features()
    panel, mb_cols, liq_cols = attach_context(panel, breadth, liq_df)
    log.info("attached breadth=%d cols, liq=%d cols. mb coverage=%.1f%% liq coverage=%.1f%%",
             len(mb_cols), len(liq_cols),
             100 * panel[mb_cols].notna().any(axis=1).mean() if mb_cols else 0,
             100 * panel[liq_cols].notna().any(axis=1).mean() if liq_cols else 0)

    up_labels = [f"lab_{n}" for n in UP_THRESH]
    dn_labels = [f"lab_{n}" for n in DN_THRESH]
    label_cols = up_labels + dn_labels

    # --- 3) variant feature set ---
    variants = {
        "BASE":           base_feats,
        "BREADTH":        base_feats + mb_cols,
        "LIQ":            base_feats + liq_cols,
        "BREADTH_LIQ":    base_feats + mb_cols + liq_cols,
    }
    n_combos = len(variants)   # selection deflate 기록 (variant 4개만 — hand-pick 없음).
    log.info("SELECTION: variant %d개 비교 (동일 OOS·유니버스·청산경로·R1 정렬). "
             "feature 격자=빌더 OOS lift 상위 hand-pick X(전부 포함).", n_combos)

    # --- 4) variant 별 WF heads ---
    log.info("=== variant WF heads ===")
    oos_by_variant = {}
    for name, feats in variants.items():
        feats = [f for f in feats if f in panel.columns]
        oos = walk_forward_heads_feats(panel, feats, label_cols, N_FOLDS, EMBARGO)
        if oos.empty:
            log.error("variant %s produced no OOS — abort", name); sys.exit(1)
        oos_by_variant[name] = oos
        log.info("  %-12s feats=%d OOS rows=%d dates=%d",
                 name, len(feats), len(oos), oos["date"].nunique())

    # exp_downside (tie-break 보조축) 는 head 라벨과 무관 — 없으면 0 으로 채움.
    # 모든 variant 동일 처리 → 공정. 정렬 1순위는 항상 R1 ratio.
    for name in oos_by_variant:
        if "exp_downside" not in oos_by_variant[name].columns:
            oos_by_variant[name]["exp_downside"] = 0.0

    # --- 5) 15m 경로 net 부착 (정렬키 무관 → (market,date) 당 1회 계산 후 공유) ---
    all_pairs = pd.concat(
        [o[["market", "date"]] for o in oos_by_variant.values()]
    ).drop_duplicates()
    conn = sqlite3.connect(M15_DB)
    m15_min = conn.execute("SELECT MIN(timestamp) FROM candles").fetchone()[0]
    conn.close()
    m15_start = pd.Timestamp(m15_min).date()
    all_pairs = all_pairs[all_pairs["date"] >= m15_start]
    bars_map = load_paths(all_pairs)
    log.info("15m paths loaded for %d (market,date) pairs (>= %s)",
             len(bars_map), m15_start)

    net_cache = {}
    for key, bars in bars_map.items():
        net, oc, eod_net = realize_net(bars)
        net_cache[key] = (net, oc, eod_net)

    def attach_net(oos):
        oos = oos[oos["date"] >= m15_start].reset_index(drop=True)
        keys = list(zip(oos["market"], oos["date"]))
        oos = oos[[k in net_cache for k in keys]].reset_index(drop=True)
        keys = list(zip(oos["market"], oos["date"]))
        oos["net"] = [net_cache[k][0] for k in keys]
        oos["outcome"] = [net_cache[k][1] for k in keys]
        oos["eod_net"] = [net_cache[k][2] for k in keys]
        oos["pump20_hit"] = [1 if v >= 0.20 else 0 for v in oos["up_high_ret"]]
        return oos.dropna(subset=["net"]).reset_index(drop=True)

    # --- 6) variant 별 R1 정렬 top-3 net 지표 ---
    up = f"p_{UP_ANCHOR}"; dn = f"p_{DN_ANCHOR}"
    rows = []
    picks_dump = []
    for name, oos in oos_by_variant.items():
        oos = oos.dropna(subset=[up, dn]).copy()
        oos = attach_net(oos)
        oos["R1"] = oos[up] / np.maximum(oos[dn], RR_EPS)
        picks = topk_picks(oos, "R1", TOP_K)
        m = net_metrics(picks)
        m.update(variant=name, ranking="R1_ratio", K=TOP_K,
                 n_feats=len([f for f in variants[name] if f in panel.columns]))
        rows.append(m)
        pk = picks[["date", "market", "regime", "fold", up, dn,
                    "R1", "net", "outcome", "pump20_hit", "up_high_ret",
                    "down_low_ret"]].copy()
        pk["variant"] = name
        picks_dump.append(pk)

    res = pd.DataFrame(rows)
    lead = ["variant", "ranking", "K", "n_feats", "n", "n_days",
            "pct_sl", "deep_loss_freq_noSL", "net_mean", "hit", "precision_pump20",
            "deep_loss_freq", "net_median", "cvar95", "worst", "mdd", "cum",
            "sharpe", "sortino", "pct_tp", "pct_eod"]
    res = res[[c for c in lead if c in res.columns]]
    res_path = OUT / "ch_features_compare_v1.csv"
    res.to_csv(res_path, index=False)
    pd.concat(picks_dump, ignore_index=True).to_csv(
        OUT / "ch_features_picks_v1.csv", index=False)

    # --- 7) feature 기여도 (anchor up_10 head, gain importance) ---
    log.info("=== feature importance (anchor=lab_up_10, gain) ===")
    imp_rows = []
    for name in ("BREADTH", "LIQ", "BREADTH_LIQ"):
        feats = [f for f in variants[name] if f in panel.columns]
        imp = feature_importance(panel, feats, "lab_up_10", N_FOLDS, EMBARGO)
        new_cols = [c for c in feats if c.startswith("mb_") or c.startswith("liq_")]
        new_share = sum(imp.get(c, 0.0) for c in new_cols)
        for f, g in sorted(imp.items(), key=lambda x: -x[1])[:12]:
            imp_rows.append(dict(variant=name, feature=f, gain_share=round(g, 4),
                                 is_new=(f.startswith("mb_") or f.startswith("liq_"))))
        log.info("  %-12s new-feature gain share = %.3f (sum over %d new feats)",
                 name, new_share, len(new_cols))
    pd.DataFrame(imp_rows).to_csv(OUT / "ch_features_importance_v1.csv", index=False)

    # --- 8) coverage + 콘솔 요약 ---
    base_row = res[res["variant"] == "BASE"].iloc[0]
    cov = dict(
        universe="static_top100", n_folds=N_FOLDS, embargo=EMBARGO,
        n_variants=int(n_combos), top_k=TOP_K,
        exit_path="15m SL-3%/TP+5%/EOD net 0.15%",
        breadth_feats=mb_cols, liq_feats=liq_cols,
        oos_dates_base=int(base_row["n_days"]), oos_picks_base=int(base_row["n"]),
        note="A3 richer-features challenger — leak-free D-1 context added to entry head",
    )
    (OUT / "ch_features_coverage_v1.json").write_text(json.dumps(cov, indent=2))

    log.info("\n===== A3: BASE vs +context (top-3, OOS net 0.15%%, 15m SL/TP/EOD) =====")
    log.info("  variant      feats  n   days  %%SL  deepNoSL net_mean hit  prec20  MaxDD   cum   Sharpe Sortino")
    for _, r in res.iterrows():
        log.info("  %-12s %4d %4d %4d  %.3f  %.3f  %+.4f %.2f  %s  %+.4f %+.3f  %s %s",
                 r["variant"], int(r["n_feats"]), int(r["n"]), int(r["n_days"]),
                 r["pct_sl"], r["deep_loss_freq_noSL"], r["net_mean"], r["hit"],
                 f"{r['precision_pump20']:.3f}" if pd.notna(r["precision_pump20"]) else " nan ",
                 r["mdd"], r["cum"],
                 f"{r['sharpe']:+.2f}" if pd.notna(r["sharpe"]) else "nan",
                 f"{r['sortino']:+.2f}" if pd.notna(r["sortino"]) else "nan")

    log.info("\n===== Δ vs BASE (음수 %%SL/deep = 하방 개선, 양수 net_mean = 상방 개선) =====")
    for _, r in res[res["variant"] != "BASE"].iterrows():
        log.info("  %-12s Δ%%SL %+.3f / ΔdeepNoSL %+.3f / Δnet_mean %+.4f / Δprec20 %+.4f / Δhit %+.3f / ΔSharpe %s",
                 r["variant"], r["pct_sl"] - base_row["pct_sl"],
                 r["deep_loss_freq_noSL"] - base_row["deep_loss_freq_noSL"],
                 r["net_mean"] - base_row["net_mean"],
                 (r["precision_pump20"] - base_row["precision_pump20"])
                 if pd.notna(r["precision_pump20"]) and pd.notna(base_row["precision_pump20"]) else float("nan"),
                 r["hit"] - base_row["hit"],
                 f"{r['sharpe'] - base_row['sharpe']:+.2f}"
                 if pd.notna(r["sharpe"]) and pd.notna(base_row["sharpe"]) else "nan")
    log.info("DONE. wrote %s + ch_features_picks_v1.csv + ch_features_importance_v1.csv "
             "+ ch_features_coverage_v1.json", res_path)


if __name__ == "__main__":
    main()

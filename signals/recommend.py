"""Production daily recommendation scorer — equal-weight rank-mean (SHADOW channel).

이 모듈은 검증 완료된 leak-free 선행 패턴 스코어러를 **프로덕션용 단일 asof 호출**로
정리한 것이다. 매일 KST 09:05 SHADOW(기록만) 채널에서 호출하는 것을 전제로 한다.

★★★ 이 모듈은 시그널 계산만 한다 (CLAUDE.md §2.2 책임 분리):
    - 텔레그램 발송 X / cron 등록 X / 업비트 자동주문·API key 절대 X (기록만).
    - 사이징/청산 실행은 ledger 책임. 여기서는 shadow 평가용 플랜(sl/tp)만 부착.

확정 스펙 (사용자, 변경 금지):
  - 스코어러 = equal-weight rank-mean: 8개 검증 leak-free 선행 feature 의
    그날 cross-section percentile rank 동일가중 평균 = score (상승 proxy, 부가표시 보존).
    XGB/logit 아님 (과신·과적합으로 탈락; rank-mean 이 precision@K 더 높음).
  - ★ 최종 top-K 정렬 = R1 risk-reward (downside-first). rr_ratio = P(≥10%)/max(P(≤-5%),eps)
    내림차순 → "하락확률 낮고 상승확률 높은" 코인 상단. P(≥10%)/P(≤-5%)/P(≤-10%) 는
    rank-mean score 가 아니라 de-correlated XGB head(D-1 feature 로 코인별 따로 학습)에서
    산출 → 저-하방 분리(NXPC/ZK/VET 상단, POKT/IN/INJ 강등). rank-mean 의 상·하방 대칭
    saturate(top-3 동일확률) 함정을 제거. head 는 downside_head_riskreward_v1 의 검증
    빌더(4/4 leak PASS) 를 포팅. score/pump_prob 는 부가표시로 보존.
  - 유니버스 = D-1 quote_volume top100 (정적). 동적 surge 는 이득 ~0.
  - 매일 top-3 추천 (R1 정렬).
  - 라벨(적중 기준) = pump20 = (high_D / open_D - 1) >= 0.20.
  - calibrated 급등확률 = 과거(asof 이전) score→pump20 bucket historical-hit.
    raw 90% 류 절대 금지 (이 프로젝트 +20% tail 90%→실제 11.6% 전적).
    top bin 은 정직하게 ~8~10% (≥20% 다음날) 로 표기됨.
  - dump_risk ⚠️ : D-1 게이지(ret_7d 극단 과열 + log_qv 고유동 board-top +
    bear_volatile regime)로 hi-risk(상위 1/3) bool.
  - 청산 플랜 = -3% 손절 + 5% 익절 (shadow 가상평가용). 진입 = day-D open(09:00).

★ LEAK 방어 (same-day leak 2번 전적 — 양보 X):
  - 모든 feature 는 build_market_features 의 market 별 .shift(1) 결과 (D-1 까지).
  - score 는 asof(=오늘) cross-section 의 rank-mean (라벨은 미래 — 학습에 안 섞임).
  - BTC regime 은 1일 shift 후 D-1 값만 (attach_btc_regime).
  - calibration bucket map 은 asof 이전(< asof) 데이터로만 적합 → asof 에 적용 (train-only).
  - 유니버스 컷도 D-1 qv (f_qv_rank).

재사용 (self-contained — gan_t/xsec_alpha import 금지, prelude 내부 모듈만):
  - 검증된 leak-free 빌더: scripts.univariate_precursor_lift_v1
      (build_market_features, add_cross_sectional)
  - regime D-1 join: scripts.regime_split_precursor_v1 (attach_btc_regime)

사용:
    from signals.recommend import score_candidates
    res = score_candidates("2026-05-31")   # -> {"asof", "top3":[...], ...}
또는 CLI 스모크:
    python -m signals.recommend --asof 2026-05-31
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# prelude 루트를 path 에 추가 (self-contained: prelude 내부 모듈만 import)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.database import list_markets, load_candles  # noqa: E402
from scripts.univariate_precursor_lift_v1 import (  # noqa: E402
    build_market_features,
    add_cross_sectional,
)
from scripts.regime_split_precursor_v1 import attach_btc_regime  # noqa: E402
# de-correlated leak-free 하락/상승 head (R1 risk-reward 정렬용) — 검증된 4/4 leak PASS 빌더.
# rank-mean score 와 달리 코인별 P(≤-5%)/P(≥10%)를 D-1 feature 로 따로 학습 → 저-하방 분리.
from scripts.downside_head_riskreward_v1 import (  # noqa: E402
    _feats as _dh_feats,
    _xgb_fit_predict as _dh_xgb,
    _oof_bucket_calib as _dh_calib_fit,
    _apply_calib as _dh_calib_apply,
)

log = logging.getLogger("recommend")

# --------------------------------------------------------------------------
# 확정 상수 (사용자 스펙 — 변경 시 컨펌 필요). 숫자는 placeholder 원칙이지만
# 라벨/유니버스/사이징은 사용자 확정값 그대로 사용.
# --------------------------------------------------------------------------
DB_PATH = str(_ROOT / "data" / "upbit_d1.db")
MIN_HISTORY = 70           # build_market_features 최소 history (60d 윈도 + 여유)

# 스코어러: 검증 leak-free 선행 8 feature 동일가중 rank-mean (recall_universe_recommender_v1 와 동일 set).
SCORE_FEATURES = [
    "f_qv_surge_30d",       # 거래대금 30d 급증
    "f_qv_surge_7d",        # 거래대금 7d 급증
    "f_bounce_off_7d_low",  # 7d 저점 대비 반등
    "f_ret_3d",             # 단기 모멘텀
    "f_ret_7d",             # 단기 모멘텀
    "f_rv_7d",              # 7d 실현변동성
    "f_atr_pct_14",         # ATR% (변동성 — universal first-split)
    "f_log_qv",             # 유동성 baseline
]

UNIVERSE_TOP_N = 100        # 정적 유니버스 = D-1 qv top100
TOP_K = 3                   # 매일 top-3 추천
MAIN_LABEL = "lab_pump20"   # 적중 기준: (high_D/open_D - 1) >= 0.20

# --- risk-reward 레이더 라벨 (전부 day-D 타겟 — score 에 안 섞임, leak 아님) ---
# upside magnitude: high_D/open_D - 1 의 임계 초과 (intraday_high_ret 재사용).
# base rate (FACTS, DB 실측): up5≈23% / up10≈7.5% / up20≈1.8%.
UPSIDE_LABELS = {
    "p_up5": ("intraday_high_ret", 0.05),    # P(고가가 +5% 이상)
    "p_up10": ("intraday_high_ret", 0.10),   # P(+10% 이상)
    "p_up20": ("intraday_high_ret", 0.20),   # P(+20% 이상) — pump20 과 동일 기준
}
# downside: low_D/open_D - 1 (intraday_low_ret) 의 임계 미만.
# upside-only 함정 방어 (FACTS): P(≤-5%) upside-only 0.53, deep-dump(≤-10%) 0.15.
DOWNSIDE_LABELS = {
    "p_dn5": 0.05,    # P(저가가 -5% 이하)
    "p_dn10": 0.10,   # P(-10% 이하) deep-dump
}
EXP_DOWNSIDE_LABEL = "intraday_low_ret"   # E[하방] = 조건부 음수 low excursion 기대값

# 청산 플랜 (shadow 가상평가용). 진입 = day-D open(09:00).
SL_PCT = -0.03              # -3% 손절
TP_PCT = 0.05               # +5% 익절

# calibration: 과거 score → pump20 bucket historical-hit (train-only, OOF 아님 —
# asof 이전 전체로 적합. rare-event raw 과신 금지: bucket hist hit 만 사용).
CAL_BUCKETS = 10
EMBARGO_DAYS = 5            # asof 직전 embargo (calibration train 종료를 asof-embargo 로)

# --- R1 risk-reward 정렬 (★ 최종 top-K 정렬 키 — 사용자 확정) ------------------
# 문제: rank-mean score → bucket calibration 은 상·하방 모두 같은 score 의 함수라
#   top-3 가 동일 bucket 에 몰려 p_up/p_dn 이 saturate(전부 같은 값)되고, 하방이
#   상방과 대칭이라 "저-하방" 코인이 분리 안 됨 (ID/XLM/INJ 처럼 고-하방이 상단).
# 해결: de-correlated XGB head 가 D-1 feature 로 P(≥10%)·P(≤-5%)·P(≤-10%)를 따로
#   학습 → 코인별로 하방이 실제로 갈림(NXPC vs POKT). 정렬키 = R1_ratio.
#   R1_ratio = p_up10 / max(p_dn5, RR_EPS) 내림차순 (상승 유지하며 하방 낮은 코인 상단).
# leak-free: head feature = D-1 (_dh_feats=LEAK_COLS/next_* 제외), 라벨은 day-D(미래),
#   train = asof-embargo 이전 universe (calibration 과 동일 cutoff). asof 는 추론만.
RR_UP_LABELS = {"p_up5": 0.05, "p_up10": 0.10, "p_up20": 0.20}   # high/open-1 >= thr
RR_DN_LABELS = {"p_dn5": 0.05, "p_dn10": 0.10}                   # low/open-1 <= -thr
RR_RATIO_UP = "p_up10"     # R1 numerator
RR_RATIO_DN = "p_dn5"      # R1 denominator (≤-5% — 사용자 ~-5% 손실 수용 anchor)
RR_EPS = 1e-3              # downside floor (downside_head_riskreward_v1 와 동일)

# dump_risk 게이지 구성요소 (D-1 정보). 상위 1/3 → hi-risk.
DUMP_OVERHEAT_FEAT = "f_ret_7d"     # 과열 (ret_7d 극단 상위)
DUMP_BOARDTOP_FEAT = "f_log_qv"     # 고유동 board-top
DUMP_HIRISK_TERTILE = 2.0 / 3.0     # 게이지 상위 1/3 = hi-risk
BEAR_VOLATILE = "bear_volatile"


# ==========================================================================
# 1. Panel build (leak-free) — 단일 asof 호출용. asof 까지의 일봉만 로드.
# ==========================================================================
def _build_panel(asof: pd.Timestamp, limit_markets: int | None = None) -> pd.DataFrame:
    """asof(=오늘) 까지의 일봉으로 leak-free panel 빌드.

    build_market_features 가 market 별 .shift(1) 로 D-1 까지 feature 를 만들고,
    라벨(pump20/pump15)은 day D open/high (미래) 로 만든다 → 시점 분리.
    asof row 의 라벨은 NaN/관측될 수 있으나 score/calibration 에 안 섞이게 처리.
    """
    markets = list_markets(DB_PATH)
    if limit_markets:
        markets = markets[:limit_markets]
    frames = []
    for m in markets:
        df = load_candles(DB_PATH, m)
        if df is None or len(df) < MIN_HISTORY:
            continue
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        # asof 이후 데이터는 절대 보지 않는다 (leak 방어). asof 당일 봉(09:00)은 포함
        # (그 봉의 feature 는 shift(1) 로 D-1 까지만 본다 → asof row feature = D-1).
        # asof 는 normalize(자정)되어 있으므로 그 날(date) 전체를 포함하도록 date 비교.
        df = df[df["timestamp"].dt.normalize() <= asof].copy()
        if len(df) < MIN_HISTORY:
            continue
        df["market"] = m
        feat = build_market_features(df)
        # day-D 하방 라벨 부착 (intraday_high_ret 은 builder 가 이미 부착).
        # intraday_low_ret = low_D/open_D - 1 (음수). day-D 타겟 → score 에 안 섞임.
        low_ret = (df["low"] / (df["open"] + 1e-12) - 1.0)
        feat = feat.merge(
            pd.DataFrame({"timestamp": df["timestamp"].values,
                          "market": df["market"].values,
                          "intraday_low_ret": low_ret.values}),
            on=["timestamp", "market"], how="left")
        frames.append(feat)
    if not frames:
        raise RuntimeError(f"no markets with >= {MIN_HISTORY} bars up to {asof.date()}")
    panel = pd.concat(frames, ignore_index=True)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel["date"] = panel["timestamp"].dt.date
    panel = panel.sort_values(["date", "market"]).reset_index(drop=True)
    return panel


# ==========================================================================
# 2. 스코어러 = equal-weight rank-mean (그날 cross-section pct-rank 동일가중 평균)
# ==========================================================================
def _add_score(panel: pd.DataFrame) -> pd.DataFrame:
    """각 SCORE_FEATURE 를 그날(date) cross-section pct-rank → 동일가중 평균 = score.
    입력 feature 는 전부 D-1 → leak 아님. NaN rank 는 0.5(중립)로 채움."""
    p = panel.copy()
    rank_cols = []
    for f in SCORE_FEATURES:
        if f not in p.columns:
            raise KeyError(f"score feature missing: {f}")
        rc = f"rk_{f}"
        p[rc] = p.groupby("date")[f].rank(pct=True)
        rank_cols.append(rc)
    p["score"] = p[rank_cols].fillna(0.5).mean(axis=1)
    return p


def _add_universe(panel: pd.DataFrame) -> pd.DataFrame:
    """정적 유니버스 = D-1 qv top100 (f_qv_rank <= 100). f_qv_rank 는 add_cross_sectional 산출."""
    p = panel.copy()
    p["in_universe"] = (p["f_qv_rank"] <= UNIVERSE_TOP_N).fillna(False)
    return p


# ==========================================================================
# 3. Calibration — asof 이전 score→pump20 bucket historical-hit (train-only).
# ==========================================================================
def _fit_calibration(train: pd.DataFrame, label: str, n_buckets: int):
    """asof-embargo 이전 train fold 안에서 score quantile bucket 별 실제 hit rate.
    rare-event raw prob 과신 금지 — bucket hist hit 만 반환."""
    d = train.dropna(subset=["score", label])
    if len(d) < 200 or d[label].sum() < 5:
        return None, None, np.nan
    try:
        bk = pd.qcut(d["score"].rank(method="first"), n_buckets,
                     labels=False, duplicates="drop")
    except ValueError:
        return None, None, d[label].mean()
    d = d.assign(bk=bk)
    g = d.groupby("bk").agg(hi=("score", "max"), hit=(label, "mean")).sort_index()
    return g["hit"].to_dict(), g["hi"].values, float(d[label].mean())


def _apply_calibration(scores: np.ndarray, hit_map, edges, base: float) -> np.ndarray:
    if hit_map is None or edges is None:
        return np.full(len(scores), base, dtype=float)
    idx = np.searchsorted(edges, scores, side="left")
    idx = np.clip(idx, 0, len(edges) - 1)
    return np.array([hit_map.get(int(b), base) for b in idx], dtype=float)


def _fit_binary_label(train: pd.DataFrame, ret_col: str, thr: float, sign: str,
                      n_buckets: int):
    """ret_col 의 임계(thr) 초과/미만 binary 라벨을 즉석 생성 후 bucket calibration.
    sign='ge' → ret >= thr (upside), sign='le' → ret <= -thr (downside).
    반환: (hit_map, edges, base) — _apply_calibration 과 동일 계약."""
    d = train.dropna(subset=["score", ret_col]).copy()
    if sign == "ge":
        d["_y"] = (d[ret_col] >= thr).astype(float)
    else:
        d["_y"] = (d[ret_col] <= -thr).astype(float)
    return _fit_calibration(d, "_y", n_buckets)


def _fit_expected_downside(train: pd.DataFrame, ret_col: str, n_buckets: int):
    """E[하방] = score bucket 별 day-D low excursion(intraday_low_ret) 평균.
    음수 기대값(보통 -0.0x). rare-event 과신 없음 — 단순 bucket mean."""
    d = train.dropna(subset=["score", ret_col])
    if len(d) < 200:
        return None, None, np.nan
    try:
        bk = pd.qcut(d["score"].rank(method="first"), n_buckets,
                     labels=False, duplicates="drop")
    except ValueError:
        return None, None, float(d[ret_col].mean())
    d = d.assign(bk=bk)
    g = d.groupby("bk").agg(hi=("score", "max"), m=(ret_col, "mean")).sort_index()
    return g["m"].to_dict(), g["hi"].values, float(d[ret_col].mean())


# ==========================================================================
# 3b. de-correlated 하락/상승 head (R1 risk-reward) — leak-free XGB.
#     rank-mean score 가 아니라 D-1 feature 로 P(≥X%)/P(≤-X%)를 코인별로 따로 학습 →
#     저-하방 코인이 실제로 분리됨 (NXPC/ZK/VET 상단, POKT/IN/INJ 강등).
# ==========================================================================
def _rr_outcome_labels(panel: pd.DataFrame) -> pd.DataFrame:
    """day-D open-anchored outcome 라벨을 패널에 부착 (타겟=미래, feature(D-1)와 시점분리).
    intraday_high_ret = high_D/open_D-1, intraday_low_ret = low_D/open_D-1 (이미 패널에 있음)
    로부터 임계 라벨 생성. downside_head_riskreward_v1._add_outcome_labels 와 동일 정의."""
    p = panel.copy()
    hi = p["intraday_high_ret"]
    lo = p["intraday_low_ret"]
    for k, thr in RR_UP_LABELS.items():
        p[f"lab_{k}"] = (hi >= thr).astype(float)
    for k, thr in RR_DN_LABELS.items():
        p[f"lab_{k}"] = (lo <= -thr).astype(float)
    return p


def _fit_rr_head(train: pd.DataFrame, asof_rows: pd.DataFrame, feats: list[str]):
    """train(asof-embargo 이전 universe) 으로 각 임계 head 를 XGB 학습 → asof 후보 추론.
    calibration = train OOF bucket historical-hit (rare-event raw 과신 금지, leak-free).
    반환: dict[label_key -> np.ndarray (asof_rows 길이)] + exp_downside array.

    ★ leak 방어:
      - feature = _dh_feats(LEAK_COLS/next_* 제외) → 전부 D-1.
      - 라벨 = day-D high/low (미래) — train 에서만 fit, asof 는 predict 만.
      - calibration bucket 도 train OOF 에서만 적합 → asof 적용 (train-only).
    """
    Xtr = train[feats].replace([np.inf, -np.inf], np.nan)
    med = Xtr.median()
    Xtr = Xtr.fillna(med).values
    Xte = asof_rows[feats].replace([np.inf, -np.inf], np.nan).fillna(med).values
    out = {}
    n = len(asof_rows)
    for k in list(RR_UP_LABELS) + list(RR_DN_LABELS):
        y = train[f"lab_{k}"].values
        raw_te, m = _dh_xgb(Xtr, y, Xte)
        if raw_te is None:
            out[k] = np.full(n, float(np.nanmean(y)), dtype=float)
            continue
        raw_tr = m.predict_proba(Xtr)[:, 1]
        ed, hm, base = _dh_calib_fit(raw_tr, y, CAL_BUCKETS)
        out[k] = _dh_calib_apply(raw_te, ed, hm, base)
    # E[하방] = dn5 raw bucket 별 train low excursion 조건부 평균 (leak-free 회귀 head).
    exp_dn = np.full(n, float(train["intraday_low_ret"].mean()), dtype=float)
    yraw, mtmp = _dh_xgb(Xtr, train["lab_p_dn5"].values, Xtr)
    if yraw is not None:
        tdf = pd.DataFrame({"s": yraw, "lr": train["intraday_low_ret"].values}).dropna()
        try:
            tdf["bk"] = pd.qcut(tdf["s"].rank(method="first"), CAL_BUCKETS,
                                labels=False, duplicates="drop")
            gg = tdf.groupby("bk").agg(hi=("s", "max"), m=("lr", "mean"))
            edd = gg["hi"].values
            mmap = gg["m"].to_dict()
            g_base = float(tdf["lr"].mean())
            raw_te_dn = mtmp.predict_proba(Xte)[:, 1]
            idx = np.clip(np.searchsorted(edd, raw_te_dn, side="left"), 0, len(edd) - 1)
            exp_dn = np.array([mmap.get(int(b), g_base) for b in idx], dtype=float)
        except ValueError:
            pass
    out["exp_downside"] = exp_dn
    return out


# ==========================================================================
# 4. dump_risk 게이지 — D-1 (ret_7d 극단 과열 + log_qv board-top + bear_volatile).
#    그날 cross-section 에서 게이지 상위 1/3 = hi-risk bool.
# ==========================================================================
def _add_dump_risk(panel: pd.DataFrame, asof_date) -> pd.DataFrame:
    """dump_risk_flag = D-1 게이지로 hi-risk(상위 1/3). leak-free:
    - 과열: ret_7d 의 cross-section pct-rank (높을수록 과열).
    - 고유동 board-top: log_qv 의 cross-section pct-rank.
    - bear_volatile regime: D-1 regime 이 bear_volatile 이면 게이지 가산.
    게이지 = (과열rank + boardtop_rank)/2 + bear_volatile bump. 상위 1/3 = hi-risk.

    ★ 게이지/tertile 은 **유니버스(top100) 내** cross-section 으로 계산한다.
    유니버스 자체가 고-log_qv board-top 집합이라 전체-board tertile 을 쓰면
    추천 후보 대부분이 hi-risk 가 된다(의미 없음). 추천 가능한 후보(=universe)
    안에서 상대적으로 과열/board-top 인 상위 1/3 을 플래그하는 게 배지 의도."""
    p = panel.copy()
    p["dump_gauge"] = np.nan
    p["dump_risk_flag"] = False
    day = (p["date"] == asof_date) & p["in_universe"]
    sub = p[day].copy()
    if sub.empty:
        return p
    overheat = sub[DUMP_OVERHEAT_FEAT].rank(pct=True)
    boardtop = sub[DUMP_BOARDTOP_FEAT].rank(pct=True)
    gauge = (overheat.fillna(0.5) + boardtop.fillna(0.5)) / 2.0
    # bear_volatile regime 이면 게이지 +0.15 bump (precision -1.4pp 영역으로 끌어올림)
    is_bear_vol = (sub["regime"] == BEAR_VOLATILE).fillna(False)
    gauge = gauge + np.where(is_bear_vol, 0.15, 0.0)
    # hi-risk = 게이지 상위 1/3. tie-robust: 게이지 pct-rank >= 2/3.
    # (>= quantile 은 tie 가 많을 때 1/3 을 넘어 과다 플래그 → pct-rank 로 정확히 상위 1/3.)
    grank = gauge.rank(pct=True)
    p.loc[day, "dump_gauge"] = gauge.values
    p.loc[day, "dump_risk_flag"] = (grank >= DUMP_HIRISK_TERTILE).values
    return p


# ==========================================================================
# 5. 공개 API — score_candidates(asof_date)
# ==========================================================================
def score_candidates(asof_date, limit_markets: int | None = None,
                     slot: str = "auto") -> dict:
    """asof 기준 D-1 까지 데이터로 top-3 추천을 반환 (post-open / pre-open 양쪽).

    두 호출 시점(slot):
      - **open** (09:05, post-open): asof 당일 일봉(09:00)이 이미 DB 에 있음.
        feature 는 그 봉의 .shift(1)=D-1 까지, 진입가 entry_open = asof open(09:00 실제값).
        기존 동작 그대로 (무변경 경로).
      - **preopen** (08:50, 개장 전): asof 당일 봉이 아직 없음(09:00 미개장).
        가장 최신 가용일(D-1)까지의 feature 로 '다음 거래일(asof)' 후보를 예측.
        진입가 entry_open = None (09:00 open 미존재 — 아직 거래 불가).

    slot="auto"(기본): asof 가 panel(일봉)에 있으면 post-open, 없으면 pre-open 으로
    자동 분기. 명시적으로 "open"/"preopen" 을 줄 수도 있다 (08:50/09:05 송신 분리용).

    Parameters
    ----------
    asof_date : str | datetime | date
        예측 대상일 D (KST 일봉 09:00 기준). post-open 이면 이 날의 open 이 진입가.
        pre-open 이면 이 날은 '다음 거래일'이고 진입가는 미확정(None).
    limit_markets : int | None
        개발용 마켓 수 제한 (None = 전체).
    slot : {"auto", "open", "preopen"}
        호출 시점. "auto" 면 asof 당일 봉 존재 여부로 자동 판정.

    Returns
    -------
    dict:
      {
        "asof": "YYYY-MM-DD",              # 예측 대상일 D
        "slot": "open" | "preopen",        # 실제 사용된 모드 (auto 해소 결과)
        "feature_date": "YYYY-MM-DD",      # feature 가 참조한 reference date
                                           #   open: = asof (그 봉 feature 는 shift(1)=D-1)
                                           #   preopen: = 최신 가용일 (= asof 직전 거래일)
        "btc_regime": <D-1 regime>,
        "universe_n": int,                 # 유니버스 내 후보 수
        "calibration_source": "bucket_score_pump20" | "base_rate",
        "n_history_dates": int,            # calibration train 일수
        "top3": [
          {
            "coin": "KRW-XXX", "rank": 1, "score": float,
            "pump_prob": float,            # calibrated (~0.06~0.10 top bin) = p_up20
            "pump_prob_pct": "8.5%",       # 정직 표기 (≥20% 다음날)
            "p_up5": float, "p_up10": float, "p_up20": float,  # P(고가 ≥5/10/20%)
            "p_dn5": float, "p_dn10": float,                   # P(저가 ≤-5/-10%)
            "exp_downside": float,         # E[하방] (음수 low excursion 기대값)
            "dump_risk_flag": bool,        # ⚠️ hi-risk (상위 1/3)
            "entry_open": float | None,    # open: asof open(09:00). preopen: None(미개장).
            "sl": -0.03, "tp": 0.05,       # shadow 청산 플랜
            "btc_regime": <D-1 regime>,
          }, ...
        ],
      }

    ★ LEAK 방어 (pre-open 도 post-open 과 동일하게 t-1 까지만):
      - feature 는 두 경로 모두 build_market_features 의 .shift(1) 결과(D-1 까지).
        pre-open 은 reference date 가 D-1(최신 가용일)이라 그 feature 는 D-2 까지.
        즉 pre-open feature 가 본 가장 미래 시점은 asof 의 전날 종가 — asof(미래)는 안 봄.
      - 라벨/실현(intraday_high/low_ret, pump20)은 모두 미래(asof day-D) → score/head 학습에
        안 섞임 (post-open 과 동일). pre-open 에서는 reference 일에 해당하는 미래 라벨도
        존재하지 않으므로 더더욱 leak 불가.
      - calibration/head train cutoff 도 reference date 기준 embargo (asof 미래 안 봄).
    """
    asof = pd.Timestamp(asof_date).normalize()
    if slot not in ("auto", "open", "preopen"):
        raise ValueError(f"slot must be auto|open|preopen, got {slot!r}")
    log.info("score_candidates asof=%s slot=%s", asof.date(), slot)

    # --- 1) leak-free panel (asof 까지 — pre-open 이면 asof 봉이 아직 없을 뿐) ---
    panel = _build_panel(asof, limit_markets=limit_markets)
    panel = add_cross_sectional(panel)
    panel = attach_btc_regime(panel)
    panel = _add_score(panel)
    panel = _add_universe(panel)

    asof_date = asof.date()
    panel_dates = set(panel["date"])
    asof_in_panel = asof_date in panel_dates

    # --- slot 해소: auto 면 asof 봉 존재 여부로 판정. 명시 slot 은 일관성 검증. ---
    if slot == "auto":
        resolved_slot = "open" if asof_in_panel else "preopen"
    elif slot == "open":
        if not asof_in_panel:
            raise RuntimeError(
                f"slot='open' 인데 asof {asof_date} 봉이 panel 에 없음 "
                f"(개장 전? DB stale? max date={max(panel_dates)}). "
                f"개장 전이면 slot='preopen' 또는 'auto' 를 사용.")
        resolved_slot = "open"
    else:  # slot == "preopen"
        resolved_slot = "preopen"

    # --- reference date (feature 가 참조하는 행의 date). ---
    #   open: asof 당일 행(그 행 feature 는 shift(1)=D-1 까지). entry_open = asof open.
    #   preopen: asof 당일 행이 없으니 최신 가용일(= asof 직전 거래일)을 reference 로.
    #            그 행 feature 는 그 날의 shift(1) → D-2 까지. asof(미래)는 절대 안 봄.
    #            entry_open 은 미확정(None) — 09:00 open 이 아직 존재하지 않음.
    if resolved_slot == "open":
        feat_date = asof_date
    else:
        feat_date = max(panel_dates)   # 최신 가용 reference (D-1)
        if feat_date >= asof_date:
            # asof 봉이 실제로 존재(=post-open 인데 preopen 강제) → 비일관. 막는다.
            raise RuntimeError(
                f"slot='preopen' 인데 reference {feat_date} >= asof {asof_date} "
                f"(asof 봉이 이미 존재). post-open 이면 slot='open'/'auto' 사용.")
        log.info("  pre-open: reference(feature) date=%s, predicting next trading day=%s",
                 feat_date, asof_date)

    # dump_risk / 후보 선택은 reference(feature) date 의 cross-section 에서 한다.
    #   open: feat_date == asof_date (기존과 동일).
    #   preopen: feat_date == 최신 가용일 (asof 봉이 없으므로 그날의 후보를 평가).
    panel = _add_dump_risk(panel, feat_date)

    # --- 2) entry_open — DB 에서 직접 (라벨 아님, 진입가) ---
    #   open: asof open(09:00) = 실제 진입가.
    #   preopen: 09:00 이 아직 미개장 → 진입가 미확정. open_map 비움(전부 None).
    if resolved_slot == "open":
        open_map = _load_asof_open(
            asof, set(panel.loc[panel["date"] == feat_date, "market"]))
    else:
        open_map = {}   # pre-open: entry_open 미확정 (None)

    # --- 3) calibration: (feat_date)-embargo 이전 데이터로만 적합 (train-only).
    #   ★ 유니버스(top100) 내 train 으로 적합 — 픽이 나오는 모집단과 동일해야
    #   top-bucket hit 이 실제 top100 픽이 겪는 확률을 정직하게 반영한다.
    #   (full panel 로 적합하면 모집단이 달라 prob 이 왜곡됨.)
    #   ★ leak: cutoff 는 feat_date(=feature reference) 기준 embargo. pre-open 이면
    #   feat_date < asof 라 train 이 asof 미래를 보는 일이 원천적으로 불가능.
    cutoff = (pd.Timestamp(feat_date) - pd.Timedelta(days=EMBARGO_DAYS)).date()
    train = panel[(panel["date"] < cutoff) & panel["in_universe"]].copy()
    hit_map, edges, base = _fit_calibration(train, MAIN_LABEL, CAL_BUCKETS)
    cal_source = "bucket_score_pump20" if hit_map is not None else "base_rate"
    if np.isnan(base):
        base = float(train[MAIN_LABEL].dropna().mean()) if len(train) else 0.0

    # --- risk-reward 레이더 멀티임계 calibration (전부 같은 train, train-only) ---
    # upside: P(≥5/10/20%) = high_D/open_D-1 임계 bucket-hit.
    # downside: P(≤-5/-10%) = low_D/open_D-1 임계 bucket-hit. E[하방] = low excursion mean.
    up_cal = {k: _fit_binary_label(train, col, thr, "ge", CAL_BUCKETS)
              for k, (col, thr) in UPSIDE_LABELS.items()}
    dn_cal = {k: _fit_binary_label(train, EXP_DOWNSIDE_LABEL, thr, "le", CAL_BUCKETS)
              for k, thr in DOWNSIDE_LABELS.items()}
    exp_dn_cal = _fit_expected_downside(train, EXP_DOWNSIDE_LABEL, CAL_BUCKETS)

    # --- 3b) de-correlated R1 head 학습 (★ 정렬키 출처) ---
    #   rank-mean score→bucket 은 상·하방이 대칭이라 top-3 가 saturate + 저-하방 미분리.
    #   D-1 feature 로 P(≥10%)/P(≤-5%)/P(≤-10%)를 코인별로 따로 XGB 학습 → 하방 분리.
    #   train = 같은 cutoff(asof-embargo 이전 universe), asof 는 추론만 → leak-free.
    rr_head_ok = False
    today_all = panel[(panel["date"] == feat_date) & panel["in_universe"]].copy()
    today_all = today_all.dropna(subset=["score"])
    try:
        dh_feats = [f for f in _dh_feats(panel) if f in panel.columns]
        train_lab = _rr_outcome_labels(train).dropna(
            subset=["intraday_high_ret", "intraday_low_ret"])
        if len(train_lab) >= 1000 and len(dh_feats) >= 5 and not today_all.empty:
            rr_pred = _fit_rr_head(train_lab, today_all, dh_feats)
            for k in list(RR_UP_LABELS) + list(RR_DN_LABELS):
                today_all[f"rr_{k}"] = rr_pred[k]
            today_all["rr_exp_downside"] = rr_pred["exp_downside"]
            rr_head_ok = True
        else:
            log.warning("RR head skipped (train=%d feats=%d today=%d) — fallback score-rank",
                        len(train_lab), len(dh_feats), len(today_all))
    except Exception as e:  # noqa: BLE001
        log.warning("RR head failed (%s) — fallback score-rank", e)

    # --- 4) 최종 정렬 = R1 risk-reward (de-correlated head). ---
    #   rr_ratio = p_up10 / max(p_dn5, eps) 내림차순. 동률은 연속 tie-break:
    #     1) p_dn10 작을수록(deep-dump 낮음), 2) p_up10 클수록, 3) exp_downside 높을수록
    #     (덜 깊은 하방). 셋 다 연속값이라 top-3 가 서로 다른 P/rr 을 갖게 함.
    #   head 실패(소표본 등) 시에만 기존 score 내림차순으로 fallback.
    today = today_all
    today["pump_prob"] = _apply_calibration(today["score"].values, hit_map, edges, base)
    if rr_head_ok:
        today["rr_ratio"] = today[f"rr_{RR_RATIO_UP}"] / np.maximum(
            today[f"rr_{RR_RATIO_DN}"], RR_EPS)
        today = today.sort_values(
            ["rr_ratio", "rr_p_dn10", "rr_p_up10", "rr_exp_downside"],
            ascending=[False, True, False, False])
        rank_basis = "R1_riskreward(de-corr head)"
    else:
        today["rr_ratio"] = np.nan
        today = today.sort_values("score", ascending=False)
        rank_basis = "score_rank(fallback)"

    btc_regime = _mode_regime(today)
    top = today.head(TOP_K).reset_index(drop=True)

    items = []
    for i, r in top.iterrows():
        coin = r["market"]
        eo = open_map.get(coin, np.nan)
        prob = float(r["pump_prob"]) if pd.notna(r["pump_prob"]) else float(base)
        s = float(r["score"])
        sc = np.array([s])
        if rr_head_ok:
            # de-correlated head 확률 (코인별 분리). p_up20 은 head 에 없으므로
            # score-bucket calibration 유지(희소 tail — rank-mean 이 더 정직).
            # p_up20 만 head 가 아니라 score-bucket calib → 소표본 시 base=NaN 가능 →
            # None 으로 가드(CSV 빈칸·JSON null 안전, NaN 토큰 방지).
            _pu20 = float(_apply_calibration(sc, *up_cal["p_up20"])[0])
            up = {"p_up5": round(float(r["rr_p_up5"]), 4),
                  "p_up10": round(float(r["rr_p_up10"]), 4),
                  "p_up20": round(_pu20, 4) if np.isfinite(_pu20) else None}
            dn = {"p_dn5": round(float(r["rr_p_dn5"]), 4),
                  "p_dn10": round(float(r["rr_p_dn10"]), 4)}
            e_down = round(float(r["rr_exp_downside"]), 4)
            rr_val = round(float(r["rr_ratio"]), 4) if pd.notna(r["rr_ratio"]) else None
        else:
            # fallback: 기존 score→bucket calibration (saturate 가능 — head 실패 시만).
            up = {k: round(float(_apply_calibration(sc, hm, ed, bs)[0]), 4)
                  for k, (hm, ed, bs) in up_cal.items()}
            dn = {k: round(float(_apply_calibration(sc, hm, ed, bs)[0]), 4)
                  for k, (hm, ed, bs) in dn_cal.items()}
            e_down = round(float(_apply_calibration(sc, *exp_dn_cal)[0]), 4)
            rr_val = None
        items.append({
            "coin": coin,
            "rank": int(i + 1),
            "score": round(s, 4),
            "pump_prob": round(prob, 4),
            "pump_prob_pct": f"{prob * 100:.1f}%",
            "rr_ratio": rr_val,       # R1 = p_up10/max(p_dn5,eps) — 최종 정렬키
            # --- risk-reward 레이더 축 (de-corr head, 정직 표기) ---
            "p_up5": up["p_up5"],     # P(≥5%)
            "p_up10": up["p_up10"],   # P(≥10%)
            "p_up20": up["p_up20"],   # P(≥20%)
            "p_dn5": dn["p_dn5"],     # P(≤-5%)
            "p_dn10": dn["p_dn10"],   # P(≤-10%) deep-dump
            "exp_downside": e_down,   # E[하방] (음수 low excursion)
            "dump_risk_flag": bool(r.get("dump_risk_flag", False)),
            "entry_open": float(eo) if np.isfinite(eo) else None,
            "sl": SL_PCT,
            "tp": TP_PCT,
            "btc_regime": str(r.get("regime", "unknown")),
        })

    return {
        "asof": str(asof_date),
        "slot": resolved_slot,             # "open" | "preopen" (auto 해소 결과)
        "feature_date": str(feat_date),    # feature reference date (open: =asof, preopen: D-1)
        "btc_regime": btc_regime,
        "universe_n": int(today.shape[0]),
        "calibration_source": cal_source,
        "rank_basis": rank_basis,          # "R1_riskreward(de-corr head)" | fallback
        "n_history_dates": int(train["date"].nunique()),
        "top3": items,
    }


def _load_asof_open(asof: pd.Timestamp, markets: set) -> dict:
    """asof 당일 open(09:00 진입가) 를 DB 에서 직접 로드. 진입가는 라벨이 아니라
    실제 거래 진입 시점 가격 → shadow ledger 진입가로 사용 (leak 아님: 09:00 진입)."""
    import sqlite3
    out = {}
    if not markets:
        return out
    conn = sqlite3.connect(DB_PATH)
    ts = asof.strftime("%Y-%m-%d 09:00:00")
    q = "SELECT market, open FROM candles WHERE timestamp = ? AND market IN ({})".format(
        ",".join("?" * len(markets)))
    for m, o in conn.execute(q, [ts, *markets]).fetchall():
        out[m] = o
    conn.close()
    return out


def _mode_regime(df: pd.DataFrame) -> str:
    if df.empty or "regime" not in df.columns:
        return "unknown"
    vc = df["regime"].dropna()
    return str(vc.mode().iloc[0]) if len(vc) else "unknown"


# ==========================================================================
# CLI 스모크
# ==========================================================================
def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)])
    ap = argparse.ArgumentParser(description="production rank-mean scorer (SHADOW)")
    ap.add_argument("--asof", type=str, required=True, help="YYYY-MM-DD (오늘)")
    ap.add_argument("--limit-markets", type=int, default=None)
    args = ap.parse_args()
    res = score_candidates(args.asof, limit_markets=args.limit_markets)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

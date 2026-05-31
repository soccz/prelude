"""Production daily recommendation scorer — equal-weight rank-mean (SHADOW channel).

이 모듈은 검증 완료된 leak-free 선행 패턴 스코어러를 **프로덕션용 단일 asof 호출**로
정리한 것이다. 매일 KST 09:05 SHADOW(기록만) 채널에서 호출하는 것을 전제로 한다.

★★★ 이 모듈은 시그널 계산만 한다 (CLAUDE.md §2.2 책임 분리):
    - 텔레그램 발송 X / cron 등록 X / 업비트 자동주문·API key 절대 X (기록만).
    - 사이징/청산 실행은 ledger 책임. 여기서는 shadow 평가용 플랜(sl/tp)만 부착.

확정 스펙 (사용자, 변경 금지):
  - 스코어러 = equal-weight rank-mean: 8개 검증 leak-free 선행 feature 의
    그날 cross-section percentile rank 동일가중 평균 = score. XGB/logit 아님
    (과신·과적합으로 탈락; rank-mean 이 precision@K 더 높음).
  - 유니버스 = D-1 quote_volume top100 (정적). 동적 surge 는 이득 ~0.
  - 매일 top-3 추천.
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

# 청산 플랜 (shadow 가상평가용). 진입 = day-D open(09:00).
SL_PCT = -0.03              # -3% 손절
TP_PCT = 0.05               # +5% 익절

# calibration: 과거 score → pump20 bucket historical-hit (train-only, OOF 아님 —
# asof 이전 전체로 적합. rare-event raw 과신 금지: bucket hist hit 만 사용).
CAL_BUCKETS = 10
EMBARGO_DAYS = 5            # asof 직전 embargo (calibration train 종료를 asof-embargo 로)

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
        frames.append(build_market_features(df))
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
def score_candidates(asof_date, limit_markets: int | None = None) -> dict:
    """asof(=오늘) 기준 D-1 까지 데이터로 top-3 추천을 반환.

    Parameters
    ----------
    asof_date : str | datetime | date
        오늘 날짜 (KST 일봉 09:00 기준). 이 날짜의 open 이 진입가(entry_open).
    limit_markets : int | None
        개발용 마켓 수 제한 (None = 전체).

    Returns
    -------
    dict:
      {
        "asof": "YYYY-MM-DD",
        "btc_regime": <D-1 regime>,
        "universe_n": int,                 # asof 유니버스 내 후보 수
        "calibration_source": "bucket_score_pump20" | "base_rate",
        "n_history_dates": int,            # calibration train 일수
        "top3": [
          {
            "coin": "KRW-XXX", "rank": 1, "score": float,
            "pump_prob": float,            # calibrated (~0.08~0.10 top bin)
            "pump_prob_pct": "8.5%",       # 정직 표기 (≥20% 다음날)
            "dump_risk_flag": bool,        # ⚠️ hi-risk (상위 1/3)
            "entry_open": float,           # asof open (09:00 진입가)
            "sl": -0.03, "tp": 0.05,       # shadow 청산 플랜
            "btc_regime": <D-1 regime>,
          }, ...
        ],
      }
    """
    asof = pd.Timestamp(asof_date).normalize()
    log.info("score_candidates asof=%s", asof.date())

    # --- 1) leak-free panel (asof 까지) ---
    panel = _build_panel(asof, limit_markets=limit_markets)
    panel = add_cross_sectional(panel)
    panel = attach_btc_regime(panel)
    panel = _add_score(panel)
    panel = _add_universe(panel)

    asof_date = asof.date()
    if asof_date not in set(panel["date"]):
        raise RuntimeError(
            f"asof {asof_date} not in panel (DB stale? max date={max(panel['date'])})")

    panel = _add_dump_risk(panel, asof_date)

    # --- 2) entry_open (asof open, 09:00 진입가) — DB 에서 직접 (라벨 아님, 진입가) ---
    open_map = _load_asof_open(asof, set(panel.loc[panel["date"] == asof_date, "market"]))

    # --- 3) calibration: asof-embargo 이전 데이터로만 적합 (train-only).
    #   ★ 유니버스(top100) 내 train 으로 적합 — 픽이 나오는 모집단과 동일해야
    #   top-bucket hit 이 실제 top100 픽이 겪는 확률을 정직하게 반영한다.
    #   (full panel 로 적합하면 모집단이 달라 prob 이 왜곡됨.) ---
    cutoff = (asof - pd.Timedelta(days=EMBARGO_DAYS)).date()
    train = panel[(panel["date"] < cutoff) & panel["in_universe"]].copy()
    hit_map, edges, base = _fit_calibration(train, MAIN_LABEL, CAL_BUCKETS)
    cal_source = "bucket_score_pump20" if hit_map is not None else "base_rate"
    if np.isnan(base):
        base = float(train[MAIN_LABEL].dropna().mean()) if len(train) else 0.0

    # --- 4) asof 유니버스 내 score 내림차순 top-3 + calibrated prob ---
    today = panel[(panel["date"] == asof_date) & panel["in_universe"]].copy()
    today = today.dropna(subset=["score"]).sort_values("score", ascending=False)
    today["pump_prob"] = _apply_calibration(today["score"].values, hit_map, edges, base)

    btc_regime = _mode_regime(today)
    top = today.head(TOP_K).reset_index(drop=True)

    items = []
    for i, r in top.iterrows():
        coin = r["market"]
        eo = open_map.get(coin, np.nan)
        prob = float(r["pump_prob"]) if pd.notna(r["pump_prob"]) else float(base)
        items.append({
            "coin": coin,
            "rank": int(i + 1),
            "score": round(float(r["score"]), 4),
            "pump_prob": round(prob, 4),
            "pump_prob_pct": f"{prob * 100:.1f}%",
            "dump_risk_flag": bool(r.get("dump_risk_flag", False)),
            "entry_open": float(eo) if np.isfinite(eo) else None,
            "sl": SL_PCT,
            "tp": TP_PCT,
            "btc_regime": str(r.get("regime", "unknown")),
        })

    return {
        "asof": str(asof_date),
        "btc_regime": btc_regime,
        "universe_n": int(today.shape[0]),
        "calibration_source": cal_source,
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

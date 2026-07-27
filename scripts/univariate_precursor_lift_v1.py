"""Univariate precursor lift screening (Angle A).

목표: "왜 이 코인이 day D 에 급등했나" 를 D-1 이전 데이터로만 계산한 후보 선행
피처들의 lift 로 역분석. 전수 스크리닝 → base rate 대비 lift / 표본수 /
top-decile precision / OOS(walk-forward) 안정성.

라벨(미래 = day D):
  pump20 = (high_D / open_D - 1) >= 0.20
  pump15 = (high_D / open_D - 1) >= 0.15
  (보조) pumpc20 = (close_D / open_D - 1) >= 0.20  # 종가 기준 안정 상승

★ LEAK 방어 (이 프로젝트 same-day leak 2번 전적):
  - 모든 precursor feature 는 market 별 시계열에서 .shift(1) 후 계산된 값만 사용.
    즉 day D 의 row 가 보는 모든 입력은 D-1 (어제) 까지의 정보.
  - day D 의 high/low/close 는 라벨(타겟) 계산에만 쓰이고 feature 엔 절대 안 들어감.
  - cross-sectional rank(volume_rank, momentum decile) 도 D-1 까지의 feature 로
    그날(D) panel 내에서 매김 → 라벨(D)과 시점 분리 유지.
    (rank 는 same-day cross-section 이지만 입력 피처 자체가 D-1 값이므로 leak 아님)
  - 유니버스/정규화: 각 D 의 cross-section 은 그 D 에 살아있던 코인 + D-1 feature 만.
    survivorship: 상폐 코인도 DB 에 있으면 그대로 포함(필터 X).

검증:
  - in-sample(전체) lift + walk-forward OOS lift(시간순 5 fold, embargo 5d).
  - selection bias: 시도한 feature 수를 로그에 기록.

사용:
    python scripts/univariate_precursor_lift_v1.py
    python scripts/univariate_precursor_lift_v1.py --limit-markets 40   # 개발용
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.database import list_markets, load_candles
from data.market_universe import is_excluded_signal_market

DB_PATH = "data/upbit_d1.db"
OUT_DIR = Path("output")
EPS = 1e-12
KST = ZoneInfo("Asia/Seoul")
UPBIT_D1_OPEN_HOUR = 9

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("univ_lift")


# ============================================================================
# 0. Shared research-time hygiene
# ============================================================================
def current_upbit_session_start(
    asof_kst: datetime | pd.Timestamp | str | None = None,
) -> pd.Timestamp:
    """Return the current Upbit D1 session start as a naive KST wall clock.

    Upbit stores D1 timestamps as timezone-naive KST 09:00 boundaries.  The row
    whose timestamp equals the current session start is still forming and must
    never be used as a realized target.  ``asof_kst`` exists so audits/tests can
    reproduce the exact cutoff instead of depending on the host clock.
    """
    if asof_kst is None:
        now = pd.Timestamp.now(tz=KST)
    else:
        now = pd.Timestamp(asof_kst)
        if now.tzinfo is None:
            now = now.tz_localize(KST)
        else:
            now = now.tz_convert(KST)
    session_start = now.normalize() + pd.Timedelta(hours=UPBIT_D1_OPEN_HOUR)
    if now < session_start:
        session_start -= pd.Timedelta(days=1)
    return session_start.tz_localize(None)


def completed_upbit_d1_mask(
    timestamps: pd.Series,
    asof_kst: datetime | pd.Timestamp | str | None = None,
) -> pd.Series:
    """True only for Upbit D1 rows whose full 24-hour session has closed."""
    ts = pd.to_datetime(timestamps, errors="coerce")
    return ts.notna() & (ts < current_upbit_session_start(asof_kst))


def expanding_purged_date_folds(
    dates: np.ndarray,
    n_folds: int,
    embargo_days: int,
) -> list[tuple[set, set]]:
    """Build disjoint expanding-window test folds with an index-day embargo.

    The previous research scripts accidentally added ``train_end`` twice when
    calculating ``test_end``.  That made later test windows overlap and counted
    some OOS dates two or three times.  This single implementation is shared by
    the legacy research chain so the boundary cannot drift again.
    """
    ordered = np.asarray(sorted(pd.unique(dates)))
    if n_folds <= 0:
        raise ValueError("n_folds must be positive")
    if embargo_days < 0:
        raise ValueError("embargo_days must be non-negative")
    fold_size = len(ordered) // (n_folds + 1)
    if fold_size <= embargo_days:
        return []

    folds: list[tuple[set, set]] = []
    for k in range(1, n_folds + 1):
        train_end = fold_size * k
        test_start = train_end + embargo_days
        test_end = len(ordered) if k == n_folds else fold_size * (k + 1)
        if test_start >= test_end:
            continue
        folds.append(
            (set(ordered[:train_end]), set(ordered[test_start:test_end]))
        )
    return folds


# ============================================================================
# 1. Panel build — 모든 feature 는 D-1 까지의 값 (leak-free)
# ============================================================================
def build_market_features(
    df: pd.DataFrame,
    asof_kst: datetime | pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """한 코인의 일봉 → (1) day-D 라벨, (2) D-1 이전 precursor feature.

    핵심 leak 규칙:
      - 라벨 (pump20/pump15/pumpc20): day D 의 open/high/close 사용 (타겟, OK).
      - precursor: 아래에서 만든 '오늘까지의' raw 지표를 market 별로 shift(1) →
        D row 에는 D-1 까지의 값만 남음.
    """
    g = df.sort_values("timestamp").reset_index(drop=True).copy()
    if "market" not in g.columns:
        g["market"] = g.attrs.get("_market", "UNKNOWN")
    open_, high, low, close = g["open"], g["high"], g["low"], g["close"]
    v = g["volume"]
    qv = g["quote_volume"].fillna(v * close)  # quote_volume 결측 시 근사

    # ---- 라벨 (day D, 타겟) ----
    g["lab_pump20"] = ((high / (open_ + EPS) - 1.0) >= 0.20).astype(float)
    g["lab_pump15"] = ((high / (open_ + EPS) - 1.0) >= 0.15).astype(float)
    g["lab_pumpc20"] = ((close / (open_ + EPS) - 1.0) >= 0.20).astype(float)
    g["intraday_high_ret"] = high / (open_ + EPS) - 1.0  # 진단용 (라벨 origin)

    # ---- '오늘까지' raw 지표 (아직 shift 안 함) ----
    raw = pd.DataFrame(index=g.index)
    log_ret = np.log(close / close.shift(1) + EPS)

    # 모멘텀 / ROC
    for N in (1, 3, 7, 14):
        raw[f"ret_{N}d"] = close / close.shift(N) - 1.0
    raw["roc_3d"] = close.pct_change(3)
    raw["roc_7d"] = close.pct_change(7)

    # 연속 상승일 (오늘 포함 streak)
    up = (close > close.shift(1)).astype(int)
    s = 0
    up_vals = up.values
    streak_vals = np.zeros(len(up_vals))
    for i in range(len(up_vals)):
        s = s + 1 if up_vals[i] == 1 else 0
        streak_vals[i] = s
    raw["up_streak"] = streak_vals

    # 변동성: ATR%, RV, 볼밴 폭
    tr = pd.concat(
        [
            (high - low),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    raw["atr_pct_14"] = tr.rolling(14, min_periods=7).mean() / (close + EPS)
    raw["rv_7d"] = log_ret.rolling(7, min_periods=4).std()
    raw["rv_21d"] = log_ret.rolling(21, min_periods=10).std()
    # range_contraction: 이전 7d 폭 / 최근 7d 폭 (>1 = 조용해짐)
    rng7 = (
        high.rolling(7).max() - low.rolling(7).min()
    ) / (close + EPS)
    rng7_prev = (
        high.shift(7).rolling(7).max() - low.shift(7).rolling(7).min()
    ) / (close.shift(7) + EPS)
    raw["range_contraction_7d"] = rng7_prev / (rng7 + EPS)
    bb_std = close.rolling(20).std()
    bb_mid = close.rolling(20).mean()
    raw["bb_width"] = (4 * bb_std) / (bb_mid + EPS)
    # 볼밴 squeeze: bb_width 가 최근 60일 분위 하위 (조용)
    raw["bb_width_pctile_60"] = raw["bb_width"].rolling(60, min_periods=30).rank(pct=True)

    # 거래대금 / 유동성 동학
    raw["qv"] = qv
    raw["log_qv"] = np.log(qv + 1.0)
    qv_ma7 = qv.rolling(7).mean()
    qv_ma30 = qv.rolling(30).mean()
    raw["qv_surge_7d"] = qv / (qv_ma7 + EPS)
    raw["qv_surge_30d"] = qv / (qv_ma30 + EPS)
    raw["qv_ma7_vs_ma30"] = qv_ma7 / (qv_ma30 + EPS)
    raw["vol_surge_7d"] = v / (v.rolling(7).mean() + EPS)
    # 거래대금 급증일 직후 플래그: 어제 qv_surge_7d 가 컸나 (shift 로 D-2 정보가 D-1 feature 됨)
    raw["qv_spiked_yday"] = (raw["qv_surge_7d"].shift(1) >= 3.0).astype(float)

    # 위치 (최근 N일 고점/저점 대비)
    hh20 = high.rolling(20).max()
    ll20 = low.rolling(20).min()
    raw["pos_in_20d_range"] = (close - ll20) / (hh20 - ll20 + EPS)
    raw["dist_from_20d_high"] = close / (hh20 + EPS) - 1.0
    hh60 = high.rolling(60).max()
    raw["dist_from_60d_high"] = close / (hh60 + EPS) - 1.0
    raw["near_20d_high"] = (raw["dist_from_20d_high"] >= -0.03).astype(float)
    # 직전 dump 후 반등: 최근 7d 최저 대비 현재 위치 + 7d 전 대비 하락 후
    raw["drawdown_7d"] = close / (high.rolling(7).max() + EPS) - 1.0
    raw["bounce_off_7d_low"] = close / (low.rolling(7).min() + EPS) - 1.0

    # RSI 14
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14, min_periods=7).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14, min_periods=7).mean()
    raw["rsi_14"] = 100 - 100 / (1 + gain / (loss + EPS))

    # ★ leak 방어 핵심: 모든 raw 지표를 1일 shift → D row 는 D-1 까지만 본다
    shifted = raw.shift(1)
    shifted.columns = [f"f_{x}" for x in shifted.columns]

    out = pd.concat(
        [g[["market", "timestamp", "lab_pump20", "lab_pump15", "lab_pumpc20",
            "intraday_high_ret"]], shifted], axis=1
    )
    # The current 09:00 KST row is a still-forming candle.  Keep its D-1
    # features for live/as-of scoring, but never treat its partial high/close as
    # a realized miss or outcome.
    complete = completed_upbit_d1_mask(out["timestamp"], asof_kst)
    target_cols = [
        "lab_pump20", "lab_pump15", "lab_pumpc20", "intraday_high_ret",
    ]
    out.loc[~complete, target_cols] = np.nan
    # day D 의 거래대금(어제 기준 유니버스 필터용): D-1 qv 를 별도 보관
    out["f_universe_qv"] = qv.shift(1)
    return out


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
        if df is None or len(df) < 70:  # 최소 history (60d 윈도 + 여유)
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


# ============================================================================
# 2. Cross-sectional rank features (그날 panel 내, 입력은 D-1 feature)
# ============================================================================
def add_cross_sectional(panel: pd.DataFrame) -> pd.DataFrame:
    """그날(D) 살아있는 코인들 사이 rank. 입력은 모두 D-1 feature 라 leak 아님."""
    p = panel.copy()
    # 거래대금 순위 (D-1 qv 기준) — rank 1 = 최대 거래대금
    p["f_qv_rank"] = p.groupby("date")["f_universe_qv"].rank(ascending=False, method="min")
    p["f_qv_rank_pct"] = p.groupby("date")["f_universe_qv"].rank(ascending=True, pct=True)
    # 거래대금 순위 상승 추세: 어제 rank 대비 (per market)
    p = p.sort_values(["market", "date"])
    p["f_qv_rank_chg_3d"] = p.groupby("market")["f_qv_rank"].diff(3) * -1  # +면 순위 상승
    # 모멘텀 decile (그날 cross-section)
    p["f_ret7d_xs_decile"] = p.groupby("date")["f_ret_7d"].rank(pct=True)
    p["f_ret3d_xs_decile"] = p.groupby("date")["f_ret_3d"].rank(pct=True)
    p["f_atr_xs_decile"] = p.groupby("date")["f_atr_pct_14"].rank(pct=True)
    p["f_qvsurge_xs_decile"] = p.groupby("date")["f_qv_surge_7d"].rank(pct=True)
    p = p.sort_values(["date", "market"]).reset_index(drop=True)
    return p


# ============================================================================
# 3. Univariate lift 스크리닝
# ============================================================================
def screen_feature(sub: pd.DataFrame, feat: str, label: str,
                   n_buckets: int = 10) -> dict | None:
    """한 feature 의 top-decile precision + lift."""
    d = sub[[feat, label]].dropna()
    if len(d) < 500:
        return None
    base = d[label].mean()
    if base <= 0:
        return None
    # decile (높을수록 펌프 잘 나는 쪽 = top decile)
    try:
        d["bucket"] = pd.qcut(d[feat].rank(method="first"), n_buckets, labels=False)
    except Exception:
        return None
    grp = d.groupby("bucket")[label].agg(["mean", "count"])
    top = grp.loc[grp.index.max()]
    bot = grp.loc[grp.index.min()]
    top_prec = top["mean"]
    # 어느 방향이 펌프 쪽? top vs bottom 중 큰 쪽을 신호 방향으로
    direction = "high" if top_prec >= bot["mean"] else "low"
    signal_prec = max(top_prec, bot["mean"])
    signal_n = top["count"] if direction == "high" else bot["count"]
    return {
        "feature": feat,
        "label": label,
        "n": int(len(d)),
        "base_rate": float(base),
        "top_decile_prec": float(top_prec),
        "bot_decile_prec": float(bot["mean"]),
        "signal_dir": direction,
        "signal_decile_prec": float(signal_prec),
        "signal_decile_n": int(signal_n),
        "lift": float(signal_prec / base),
    }


def walk_forward_lift(panel: pd.DataFrame, feat: str, label: str,
                      n_folds: int = 5, embargo_days: int = 5) -> dict:
    """시간순 OOS lift: train 절반에서 top-decile cutoff 정하고 test 에서 precision."""
    dates = np.sort(panel["date"].unique())
    oos_precs, oos_ns, base_rates = [], [], []
    d = panel[[feat, label, "date"]].dropna()
    if len(d) < 1000:
        return {"oos_lift": np.nan, "oos_prec": np.nan, "oos_n": 0, "n_folds_used": 0}
    for train_dates, test_dates in expanding_purged_date_folds(
        dates, n_folds, embargo_days
    ):
        tr = d[d["date"].isin(train_dates)]
        te = d[d["date"].isin(test_dates)]
        if len(tr) < 300 or len(te) < 200:
            continue
        # train 에서 top-decile cutoff (방향은 train 의 top/bot 비교)
        tr_top_cut = tr[feat].quantile(0.9)
        tr_bot_cut = tr[feat].quantile(0.1)
        top_rate = tr[tr[feat] >= tr_top_cut][label].mean()
        bot_rate = tr[tr[feat] <= tr_bot_cut][label].mean()
        if top_rate >= bot_rate:
            sel = te[te[feat] >= tr_top_cut]
        else:
            sel = te[te[feat] <= tr_bot_cut]
        if len(sel) < 30:
            continue
        oos_precs.append(sel[label].mean())
        oos_ns.append(len(sel))
        base_rates.append(te[label].mean())
    if not oos_precs:
        return {"oos_lift": np.nan, "oos_prec": np.nan, "oos_n": 0, "n_folds_used": 0}
    # n-weighted 평균
    w = np.array(oos_ns, dtype=float)
    oos_prec = float(np.average(oos_precs, weights=w))
    base = float(np.average(base_rates, weights=w))
    return {
        "oos_prec": oos_prec,
        "oos_base": base,
        "oos_lift": oos_prec / base if base > 0 else np.nan,
        "oos_n": int(w.sum()),
        "n_folds_used": len(oos_precs),
    }


CANDIDATE_FEATURES = [
    # 모멘텀
    "f_ret_1d", "f_ret_3d", "f_ret_7d", "f_ret_14d", "f_roc_3d", "f_roc_7d",
    "f_up_streak", "f_ret7d_xs_decile", "f_ret3d_xs_decile",
    # 변동성
    "f_atr_pct_14", "f_rv_7d", "f_rv_21d", "f_range_contraction_7d",
    "f_bb_width", "f_bb_width_pctile_60", "f_atr_xs_decile",
    # 거래대금 / 유동성
    "f_qv_surge_7d", "f_qv_surge_30d", "f_qv_ma7_vs_ma30", "f_vol_surge_7d",
    "f_qv_spiked_yday", "f_log_qv", "f_qv_rank", "f_qv_rank_pct",
    "f_qv_rank_chg_3d", "f_qvsurge_xs_decile",
    # 위치
    "f_pos_in_20d_range", "f_dist_from_20d_high", "f_dist_from_60d_high",
    "f_near_20d_high", "f_drawdown_7d", "f_bounce_off_7d_low",
    # 기타
    "f_rsi_14",
]
LABELS = ["lab_pump20", "lab_pump15", "lab_pumpc20"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-markets", type=int, default=None)
    ap.add_argument("--universe-top", type=int, default=None,
                    help="그날 D-1 거래대금 top-N 으로 유니버스 제한 (None=전체)")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    panel = build_panel(args.limit_markets)
    panel = add_cross_sectional(panel)
    log.info("panel rows=%d, markets=%d, dates=%d",
             len(panel), panel["market"].nunique(), panel["date"].nunique())

    # 유니버스 옵션 (D-1 qv 기준)
    if args.universe_top:
        panel = panel[panel["f_qv_rank"] <= args.universe_top].copy()
        log.info("universe top-%d applied → rows=%d", args.universe_top, len(panel))

    # 라벨 base rate
    for lab in LABELS:
        log.info("base rate %s = %.4f (n=%d)", lab, panel[lab].mean(), panel[lab].notna().sum())

    n_features = len(CANDIDATE_FEATURES)
    n_tries = n_features * len(LABELS)
    log.info("SELECTION BIAS: features=%d × labels=%d = %d univariate tries",
             n_features, len(LABELS), n_tries)

    rows = []
    for lab in LABELS:
        for feat in CANDIDATE_FEATURES:
            if feat not in panel.columns:
                continue
            res = screen_feature(panel, feat, lab)
            if res is None:
                continue
            wf = walk_forward_lift(panel, feat, lab)
            res.update(wf)
            rows.append(res)

    out = pd.DataFrame(rows)
    out = out.sort_values(["label", "oos_lift"], ascending=[True, False])
    out_path = OUT_DIR / "univariate_precursor_lift_v1.csv"
    out.to_csv(out_path, index=False)
    log.info("wrote %s (%d rows)", out_path, len(out))

    # 콘솔 요약: pump20 기준 OOS lift top
    for lab in LABELS:
        sub = out[out["label"] == lab].copy()
        sub = sub[sub["oos_n"] >= 200].sort_values("oos_lift", ascending=False)
        log.info("=== %s — top OOS lift (oos_n>=200) ===", lab)
        for _, r in sub.head(12).iterrows():
            log.info("  %-26s dir=%-4s IS_lift=%.2f OOS_lift=%.2f OOS_prec=%.3f base=%.3f n=%d folds=%d",
                     r["feature"], r["signal_dir"], r["lift"], r["oos_lift"],
                     r["oos_prec"], r["base_rate"], int(r["oos_n"]), int(r["n_folds_used"]))


if __name__ == "__main__":
    main()

"""Liquidity Dynamics Discovery v1 — angle="liquidity_dynamics" (Angle D).

질문: 펌프는 급등 전에 거래대금으로 자신을 예고하는가?
  큰 펌프의 44%가 top100 거래대금 유니버스 밖이라 신호 생성조차 안 됨 (recall gap).
  → 정적 top100 대신 "거래대금 급증 기반 동적 워치리스트" 가 사전에 그 펌프를 후보에 넣을 수 있었나?

라벨 (미래, day D):
  pump20 = (day D high / day D open - 1) >= 0.20
  pump15 = ... >= 0.15  (보조)
  ※ open 은 KST 09:00 일봉 시가. high 는 그날 장중 최고. 둘 다 day D.

선행조건 (precursor) — 반드시 D-1 및 이전 데이터로만:
  거래대금/거래량은 quote_volume (KRW 거래대금) 사용. 모든 rolling/rank/median 은 t = D-1 시점까지.
  - qv_ratio_med_{N}d   : D-1 거래대금 / 자기 N일 중앙값 (N in 7,14,30)
  - qv_ratio_med_max_3d : 최근 3일(D-1,D-2,D-3) 중 max(qv/medN)  — surge 가 며칠에 걸쳐 와도 잡음
  - qv_zscore_30d       : D-1 거래대금의 30일 rolling z-score
  - qv_rank_xs          : D-1 거래대금의 그날 전체 코인 cross-sectional rank (0~1, 동적 유니버스 위치)
  - qv_rank_jump_3d     : qv_rank_xs(D-1) - qv_rank_xs(D-4)  — 순위가 며칠새 급상승했나
  - qv_consec_up        : D-1 기준 거래대금 연속 증가일 수
  - in_top100_dm1       : D-1 거래대금 cross-sectional top100 안이었나 (정적 유니버스 baseline)
  - dollar_vol_dm1      : D-1 거래대금 절대값 (low-liquidity 필터용)

leak-free 보장:
  - 모든 feature 는 코인별 시계열에서 .shift() 후 rolling → 그 행의 feature 는 D-1 까지만 반영.
    (구현: feature 를 "그 행 t 까지" 로 계산 후, 라벨과 join 할 때 label 을 t+1 행으로 shift.
     즉 panel row 의 feat = t 일까지, label = t+1 일 pump. feature_date = label_date - 1.)
  - cross-sectional rank 도 각 날짜 t 의 그 시점 데이터로만 (미래 코인 안 봄).
  - day D 의 high/low/close 는 라벨 계산에만 쓰이고 feature 엔 절대 안 들어감.

검증:
  - PurgedWalkForward 와 동일 정신: 시간순 5-fold, train 구간에서 base rate/threshold 정하고
    test(OOS) 구간에서 lift/recall/precision 측정. embargo 5일.
  - selection bias: 시도한 feature 격자 / threshold 격자 수 기록.

산출물:
  output/liquidity_dynamics_discovery_v1.csv   — per (feature, threshold, fold) OOS 통계
  output/liquidity_dynamics_discovery_v1.log
  stdout 요약 + 이번주 케이스 커버리지.

사용:
    python scripts/liquidity_dynamics_discovery_v1.py
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "upbit_d1.db"
OUT_CSV = ROOT / "output" / "liquidity_dynamics_discovery_v1.csv"
OUT_LOG = ROOT / "output" / "liquidity_dynamics_discovery_v1.log"

log = logging.getLogger("liqdyn")

# 라벨 임계 (placeholder, CLAUDE.md §2.5)
PUMP_THRESHOLDS = {"pump20": 0.20, "pump15": 0.15}

# precursor 윈도우 (placeholder)
MED_WINDOWS = (7, 14, 30)
EPS = 1e-12

# 이번주 실제 급등 케이스 (사용자 제공) — 커버리지 확인용
RECENT_CASES = [
    # (market_symbol, approx_date, approx_ret, was_in_universe)
    ("ID", "2026-05-29", 0.53, False),
    ("ID", "2026-05-30", 0.39, False),
    ("GMT", "2026-05-29", 0.46, False),
    ("VTHO", "2026-05-29", 0.43, False),
    ("META", "2026-05-29", 0.36, False),
    ("ERA", "2026-05-29", 0.35, False),
    ("PROVE", "2026-05-29", 0.66, True),
    ("ALT", "2026-05-29", 0.51, True),
    ("PRL", "2026-05-29", 0.44, True),
    ("XLM", "2026-05-28", 0.32, True),
    ("WLD", "2026-05-29", 0.31, True),
]


def setup_logging():
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(OUT_LOG, mode="w")],
    )


# ============================================================================
# 1. 데이터 로드
# ============================================================================
def load_all(db_path: Path) -> pd.DataFrame:
    con = sqlite3.connect(str(db_path))
    df = pd.read_sql_query(
        "SELECT market, timestamp, open, high, low, close, volume, quote_volume FROM candles",
        con,
    )
    con.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.normalize()
    df = df.sort_values(["market", "timestamp"]).reset_index(drop=True)
    return df


# ============================================================================
# 2. precursor feature (per-market 시계열, 전부 t 까지 — leak-safe)
# ============================================================================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """각 (market, t) 행에 t 일까지의 거래대금 동학 feature 를 붙인다.

    중요: 이 함수는 라벨을 안 붙인다. feature 는 모두 '그 행 t 시점' 정보.
    라벨 join 시 label 을 t+1 로 shift 하므로 feature(t) → pump(t+1) 가 되어 D-1 leak-safe.
    """
    parts = []
    for market, g in df.groupby("market", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        qv = g["quote_volume"].astype(float)

        # 자기 N일 중앙값 (그 행 t 포함; t 까지 데이터만)
        for N in MED_WINDOWS:
            med = qv.rolling(N, min_periods=max(3, N // 2)).median()
            g[f"qv_ratio_med_{N}d"] = qv / (med + EPS)

        # 최근 3일(t, t-1, t-2) 중 max(qv/med14) — surge 가 며칠에 걸쳐도 잡음
        g["qv_ratio_med_max_3d"] = (
            g["qv_ratio_med_14d"].rolling(3, min_periods=1).max()
        )

        # 30일 rolling z-score
        m30 = qv.rolling(30, min_periods=10).mean()
        s30 = qv.rolling(30, min_periods=10).std()
        g["qv_zscore_30d"] = (qv - m30) / (s30 + EPS)

        # 연속 증가일 수 (t 기준)
        up = (qv > qv.shift(1)).astype(int)
        # streak: 누적 연속 1
        streak = up * (up.groupby((up != up.shift()).cumsum()).cumcount() + 1)
        g["qv_consec_up"] = streak

        # 절대 거래대금 (low-liq 필터)
        g["dollar_vol"] = qv

        # 일봉 시가/장중 — 라벨 계산용 (feature 아님, day D 정보)
        # 여기선 보관만; build_panel 에서 shift 처리
        parts.append(g)

    out = pd.concat(parts, ignore_index=True)

    # cross-sectional: 각 날짜 t 의 그 시점 거래대금 rank (미래 안 봄)
    out["qv_rank_xs"] = out.groupby("date")["quote_volume"].rank(pct=True)
    out["in_top100"] = out.groupby("date")["quote_volume"].rank(
        ascending=False, method="first"
    ) <= 100

    # rank jump: t 와 t-3 의 rank 차 (per market)
    out = out.sort_values(["market", "timestamp"]).reset_index(drop=True)
    out["qv_rank_jump_3d"] = out.groupby("market")["qv_rank_xs"].diff(3)
    return out


# ============================================================================
# 3. panel: feature(t) → label(t+1), leak-safe shift
# ============================================================================
def build_panel(feat: pd.DataFrame) -> pd.DataFrame:
    """feature 행 t 에, 다음 거래일 t+1 의 pump 라벨을 붙인다.

    feature_date = label_date - 1day 정신 (RESEARCH §6.2 leak-fix 와 동일).
    """
    rows = []
    feat = feat.sort_values(["market", "timestamp"]).reset_index(drop=True)
    for market, g in feat.groupby("market", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        # 다음 행(t+1) 의 day-D open/high
        next_open = g["open"].shift(-1)
        next_high = g["high"].shift(-1)
        next_ts = g["timestamp"].shift(-1)
        # 연속 거래일만 라벨 유효 (gap > 2일이면 제외)
        gap_days = (next_ts - g["timestamp"]).dt.days
        intraday_ret = next_high / (next_open + EPS) - 1.0
        g = g.copy()
        g["label_date"] = next_ts
        g["label_gap"] = gap_days
        g["next_intraday_ret"] = intraday_ret
        for name, thr in PUMP_THRESHOLDS.items():
            g[name] = (intraday_ret >= thr).astype(float)
        rows.append(g)
    panel = pd.concat(rows, ignore_index=True)
    # 라벨 없는 마지막 행 / gap 큰 행 제외
    panel = panel[panel["label_date"].notna()]
    panel = panel[(panel["label_gap"] >= 1) & (panel["label_gap"] <= 2)]
    return panel.reset_index(drop=True)


# ============================================================================
# 4. walk-forward OOS 평가
# ============================================================================
FEATURE_GRID = {
    "qv_ratio_med_7d":   [1.5, 2.0, 3.0, 5.0],
    "qv_ratio_med_14d":  [1.5, 2.0, 3.0, 5.0],
    "qv_ratio_med_30d":  [1.5, 2.0, 3.0, 5.0],
    "qv_ratio_med_max_3d": [2.0, 3.0, 5.0, 8.0],
    "qv_zscore_30d":     [1.0, 2.0, 3.0],
    "qv_rank_jump_3d":   [0.10, 0.20, 0.30],
    "qv_consec_up":      [2, 3, 4],
    "qv_rank_xs":        [0.80, 0.90, 0.95],  # 동적 유니버스 위치
}


def wf_folds(panel: pd.DataFrame, n_folds=5, embargo_days=5):
    """시간순 expanding train / 다음 구간 test. base/threshold 는 정보용,
    여기선 단일 임계 rule 의 OOS lift 만 보므로 train 구간은 분포 참고용.
    실제 평가는 test(OOS) 구간 통계."""
    dates = np.sort(panel["date"].unique())
    n = len(dates)
    fold_size = n // (n_folds + 1)
    folds = []
    for k in range(n_folds):
        train_end = fold_size * (k + 1)
        test_start = train_end + embargo_days
        test_end = min(train_end + fold_size + embargo_days, n)
        if test_start >= n:
            break
        train_dates = dates[:train_end]
        test_dates = dates[test_start:test_end]
        folds.append((train_dates, test_dates))
    return folds


def evaluate(panel: pd.DataFrame):
    results = []
    n_combos = 0
    for label in PUMP_THRESHOLDS:
        folds = wf_folds(panel)
        for feat, thrs in FEATURE_GRID.items():
            for thr in thrs:
                n_combos += 1
                fold_stats = []
                for fi, (tr_dates, te_dates) in enumerate(folds):
                    te = panel[panel["date"].isin(te_dates)]
                    te = te[te[feat].notna() & te[label].notna()]
                    if len(te) < 200:
                        continue
                    base = te[label].mean()
                    sel = te[te[feat] >= thr]
                    n_sel = len(sel)
                    if n_sel < 20:
                        continue
                    prec = sel[label].mean()
                    lift = prec / (base + EPS)
                    # recall: 모든 pump 중 이 rule 이 잡은 비율
                    n_pump = te[label].sum()
                    recall = sel[label].sum() / (n_pump + EPS) if n_pump > 0 else np.nan
                    fold_stats.append(dict(
                        fold=fi, base=base, prec=prec, lift=lift,
                        recall=recall, n_sel=n_sel, n_pump=int(n_pump),
                        sel_rate=n_sel / len(te),
                    ))
                if len(fold_stats) < 3:
                    continue
                fs = pd.DataFrame(fold_stats)
                results.append(dict(
                    label=label, feature=feat, threshold=thr,
                    n_folds_ok=len(fs),
                    base_mean=fs["base"].mean(),
                    prec_mean=fs["prec"].mean(),
                    lift_mean=fs["lift"].mean(),
                    lift_min=fs["lift"].min(),
                    lift_std=fs["lift"].std(),
                    recall_mean=fs["recall"].mean(),
                    sel_rate_mean=fs["sel_rate"].mean(),
                    n_sel_total=int(fs["n_sel"].sum()),
                    folds_lift_gt1=int((fs["lift"] > 1.0).sum()),
                ))
    return pd.DataFrame(results), n_combos


# ============================================================================
# 5. 동적 유니버스 recall 시뮬레이션
# ============================================================================
def recall_universe_compare(panel: pd.DataFrame, label="pump20", recent_only=False):
    """정적 top100 vs 동적(거래대금 surge OR top-rank) 유니버스가
    실제 pump 를 사전 후보에 얼마나 담았나 (recall) + 후보 규모(precision proxy).

    recent_only=True 면 현 유니버스 regime (그날 상장 코인 >=180) 만 — 사용자 진단
    (큰 펌프 44% top100 밖) 과 시점 일치. 초기 2023 (~95 코인) 은 top100 이 전부라
    historical recall 을 부풀리므로 분리 평가.
    """
    p = panel[panel[label].notna()].copy()
    if recent_only:
        ncoin = p.groupby("date")["market"].transform("nunique")
        p = p[ncoin >= 180]
    pumps = p[p[label] == 1.0]
    n_pump = len(pumps)
    out = {}

    def cov(mask_func, name):
        sel = p[mask_func(p)]
        n_sel = len(sel)
        caught = sel[label].sum()
        out[name] = dict(
            recall=caught / (n_pump + EPS),
            candidate_rate=n_sel / len(p),
            precision=caught / (n_sel + EPS),
            n_candidates=int(n_sel),
        )

    cov(lambda d: d["in_top100"], "static_top100")
    cov(lambda d: d["qv_rank_xs"] >= 0.80, "dynamic_rank_top20pct")
    cov(lambda d: d["qv_ratio_med_max_3d"] >= 3.0, "surge_3x_med14")
    cov(lambda d: d["in_top100"] | (d["qv_ratio_med_max_3d"] >= 3.0),
        "top100_OR_surge3x")
    cov(lambda d: d["in_top100"] | (d["qv_rank_jump_3d"] >= 0.20),
        "top100_OR_rankjump20")
    cov(lambda d: (d["qv_rank_xs"] >= 0.70) | (d["qv_ratio_med_max_3d"] >= 3.0),
        "rank70_OR_surge3x")
    return out, n_pump


# ============================================================================
# 6. 이번주 케이스 커버리지
# ============================================================================
def case_coverage(feat: pd.DataFrame):
    """RECENT_CASES 각각에 대해, 펌프 전날(D-1) precursor 값을 출력.
    market 심볼 → KRW-<SYM> 매핑."""
    rows = []
    for sym, date_s, ret, in_univ in RECENT_CASES:
        market = f"KRW-{sym}"
        D = pd.to_datetime(date_s).normalize()
        Dm1 = D - pd.Timedelta(days=1)
        g = feat[feat["market"] == market]
        if len(g) == 0:
            rows.append(dict(sym=sym, date=date_s, note="market not in DB"))
            continue
        # D-1 행 (없으면 가장 가까운 직전)
        cand = g[g["date"] <= Dm1].sort_values("date")
        if len(cand) == 0:
            rows.append(dict(sym=sym, date=date_s, note="no D-1 row"))
            continue
        r = cand.iloc[-1]
        rows.append(dict(
            sym=sym, date=date_s, ret=ret, in_univ_claimed=in_univ,
            dm1_date=str(r["date"].date()),
            qv_ratio_med_14d=round(float(r.get("qv_ratio_med_14d", np.nan)), 2),
            qv_ratio_med_max_3d=round(float(r.get("qv_ratio_med_max_3d", np.nan)), 2),
            qv_zscore_30d=round(float(r.get("qv_zscore_30d", np.nan)), 2),
            qv_rank_xs=round(float(r.get("qv_rank_xs", np.nan)), 3),
            qv_rank_jump_3d=round(float(r.get("qv_rank_jump_3d", np.nan)), 3),
            qv_consec_up=int(r.get("qv_consec_up", 0)),
            in_top100_dm1=bool(r.get("in_top100", False)),
        ))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DB))
    args = ap.parse_args()
    setup_logging()

    log.info("=== Liquidity Dynamics Discovery v1 (angle=liquidity_dynamics) ===")
    df = load_all(Path(args.db))
    log.info(f"loaded {len(df):,} rows, {df['market'].nunique()} markets, "
             f"{df['date'].min().date()}..{df['date'].max().date()}")

    feat = build_features(df)
    panel = build_panel(feat)
    log.info(f"panel: {len(panel):,} rows (feature_date=label_date-1, gap<=2d)")
    for name, thr in PUMP_THRESHOLDS.items():
        log.info(f"  {name} (>= {thr:.0%}) base rate = {panel[name].mean():.4%} "
                 f"({int(panel[name].sum())} events)")

    # 1. WF OOS feature lift
    res, n_combos = evaluate(panel)
    res = res.sort_values(["label", "lift_mean"], ascending=[True, False])
    res.to_csv(OUT_CSV, index=False)
    log.info(f"\nfeature/threshold combos tried: {n_combos} "
             f"(× {len(PUMP_THRESHOLDS)} labels), wrote {OUT_CSV}")
    log.info("\n=== TOP OOS lift (pump20) ===")
    top = res[res["label"] == "pump20"].head(12)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        log.info("\n" + top[["feature", "threshold", "base_mean", "prec_mean",
                              "lift_mean", "lift_min", "recall_mean",
                              "sel_rate_mean", "folds_lift_gt1"]].to_string(index=False))

    # 2. universe recall compare (ALL + RECENT regime; 사용자 진단과 시점 일치하려면 RECENT)
    for label in ("pump20", "pump15"):
        for recent in (False, True):
            cmp, n_pump = recall_universe_compare(panel, label, recent_only=recent)
            tag = "RECENT (n_coins>=180)" if recent else "ALL"
            log.info(f"\n=== Universe recall compare ({label}, {tag}, n_pump={n_pump}) ===")
            for name, d in cmp.items():
                log.info(f"  {name:28s} recall={d['recall']:.1%}  "
                         f"cand_rate={d['candidate_rate']:.1%}  "
                         f"prec={d['precision']:.3%}  n_cand={d['n_candidates']:,}")

    # 3. recent case coverage
    cc = case_coverage(feat)
    log.info("\n=== This-week pump case coverage (D-1 precursor values) ===")
    with pd.option_context("display.width", 240, "display.max_columns", 30):
        log.info("\n" + cc.to_string(index=False))

    log.info("\nDONE")


if __name__ == "__main__":
    main()

"""Path-ordered exit backtest for bear_quiet pump candidates (15m intrabar walk).

목적 (이 워크플로의 단일 질문):
  bear_quiet regime 펌프 후보(qv_surge_7d 1순위 + qv_surge_30d / bounce_off_7d_low
  / atr_xs_decile)를 day-D open 진입했을 때, 일봉 OHLC로는 미상이던 intrabar
  청산 경로(SL/TP 누가 먼저 닿았나)를 15m 봉을 **시간순으로 walk** 하며 해소하고,
  청산정책 그리드별 net 성과를 측정해 PROMISING_SHADOW→PORTFOLIO_GRADE 승급 근거를 댄다.

진입 신호 = bear_quiet 날(BTC D-1 ma_distance(200d)<0 & rv30 252d-intensity<=0.5)에
그날 cross-section top-decile 후보(train fold quantile cutoff).
진입 = day-D open(09:00 KST). 경로 = [09:00 D, 09:00 D+1) 의 ~96개 15m 봉.

★ LEAK 방어 (same-day leak 2번 전적):
  - 진입 결정(어느 코인/언제): D-1 까지 feature + D-1 regime (build_market_features의
    shift(1), attach_btc_regime의 regime shift(1)). top-decile cutoff = train fold
    quantile (OOF-style, test fold에 leak X).
  - day-D 경로: in-trade outcome 이므로 leak 아님. 단 미래 봉을 미리 보지 않고
    15m 봉을 시간 오름차순으로만 순회. 어떤 정책도 그날 종가/고저를 미리 안 봄.
  - 같은 15m 봉 안에서 TP·SL 동시 조건 → SL 먼저(보수). 일봉(4배 큰 봉)보다 미세하나
    여전히 봉 내부는 ambiguous → 보수 가정 유지.

거래비용: 왕복 0.15% 차감(진입+청산 합산, net = gross - 0.0015). spot-only.

사용:
    python scripts/bear_quiet_path_exit_v1.py
    python scripts/bear_quiet_path_exit_v1.py --limit-markets 60   # 개발
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.database import list_markets, load_candles
from data.market_universe import signal_eligible_markets
from signals.features import compute_btc_features
from scripts.univariate_precursor_lift_v1 import (
    build_market_features,
    add_cross_sectional,
)

D1_DB = "data/upbit_d1.db"
M15_DB = "data/upbit_15m.db"
OUT_DIR = Path("output")
EPS = 1e-12
ROUND_TRIP_COST = 0.0015
N_FOLDS = 5
EMBARGO_DAYS = 5
TOP_DECILE_Q = 0.90  # cross-section top-decile

# 1순위 + 보조 후보 (모두 'high' 방향이 펌프 쪽)
CANDIDATES = [
    "f_qv_surge_7d",       # 1순위
    "f_qv_surge_30d",
    "f_bounce_off_7d_low",
    "f_atr_xs_decile",
]

# ---- 청산정책 그리드 ----
HARD_SL = [None, 0.03, 0.05, 0.08]
TP = [None, 0.05, 0.08, 0.10, 0.15]
TIME_STOP_H = [None, 2.0, 4.0, 8.0]      # 무TP시 N시간 후 청산
TRAIL = [None, 0.05, 0.08, 0.10]          # 고점대비 -X% trailing
BARS_PER_HOUR = 4                          # 15m → 4봉/시간

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("bq_path")


# ============================================================================
# 1. d1 panel (D-1 features) + bear_quiet day flag
# ============================================================================
def build_panel(limit_markets: int | None) -> pd.DataFrame:
    markets = signal_eligible_markets(list_markets(D1_DB))
    if limit_markets:
        markets = markets[:limit_markets]
    log.info("d1 loading %d markets", len(markets))
    frames = []
    for i, m in enumerate(markets):
        df = load_candles(D1_DB, m)
        if df is None or len(df) < 70:
            continue
        df = df.copy()
        df["market"] = m
        frames.append(build_market_features(df))
    panel = pd.concat(frames, ignore_index=True)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel["date"] = panel["timestamp"].dt.date
    panel = panel.sort_values(["date", "market"]).reset_index(drop=True)
    return panel


def attach_bear_quiet(panel: pd.DataFrame) -> pd.DataFrame:
    """BTC regime을 D-1 시점으로 부착 (leak 방어 shift) → bear_quiet flag."""
    btc = load_candles(D1_DB, "KRW-BTC")
    bf = compute_btc_features(btc)[["timestamp", "btc_regime"]].copy()
    bf["timestamp"] = pd.to_datetime(bf["timestamp"])
    bf = bf.sort_values("timestamp").reset_index(drop=True)
    bf["regime_dm1"] = bf["btc_regime"].shift(1)   # day D는 D-1 regime을 본다
    bf = bf[["timestamp", "regime_dm1"]]
    p = panel.merge(bf, on="timestamp", how="left")
    p["is_bear_quiet"] = (p["regime_dm1"] == "bear_quiet").astype(float)
    return p


# ============================================================================
# 2. bear_quiet 날 후보 진입 row 수집 (walk-forward, train fold quantile)
# ============================================================================
def collect_entries(panel: pd.DataFrame, feat: str) -> pd.DataFrame:
    """train fold quantile cutoff로 bear_quiet test fold에서 top-decile 진입 수집.

    - cutoff(top-decile)는 train fold의 bear_quiet row에서 계산(OOF) → test fold 적용.
    - 반환 row: market/date(=진입일 D)/feat. 경로는 별도 15m join.
    """
    d = panel[panel["is_bear_quiet"] == 1.0][["market", "date", feat]].dropna(subset=[feat])
    all_dates = np.sort(panel["date"].unique())
    fold_size = len(all_dates) // (N_FOLDS + 1)
    sel = []
    for k in range(1, N_FOLDS + 1):
        train_end = fold_size * k
        test_start = train_end + EMBARGO_DAYS
        test_end = min(fold_size * (k + 1) + train_end, len(all_dates))
        if test_start >= len(all_dates):
            break
        train_dates = set(all_dates[:train_end])
        test_dates = set(all_dates[test_start:test_end])
        tr = d[d["date"].isin(train_dates)]
        te = d[d["date"].isin(test_dates)]
        if len(tr) < 50 or len(te) < 10:
            continue
        cut = tr[feat].quantile(TOP_DECILE_Q)
        s = te[te[feat] >= cut].copy()
        s["fold"] = k
        sel.append(s)
    if not sel:
        return pd.DataFrame(columns=["market", "date", feat, "fold"])
    return pd.concat(sel, ignore_index=True)


# ============================================================================
# 3. 15m 경로 로딩 (day-D = [09:00 D, 09:00 D+1))
# ============================================================================
def load_paths(entries: pd.DataFrame) -> dict:
    """(market, date) → 15m 봉 리스트 [(open,high,low,close), ...] 시간순.

    경로 = entry_date 09:00:00 (포함) ~ entry_date+1 09:00:00 (미포함).
    """
    conn = sqlite3.connect(M15_DB)
    paths = {}
    pairs = entries[["market", "date"]].drop_duplicates()
    for _, r in pairs.iterrows():
        m, dt = r["market"], pd.Timestamp(r["date"])
        start = dt.strftime("%Y-%m-%d 09:00:00")
        end = (dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d 09:00:00")
        rows = conn.execute(
            "SELECT open,high,low,close FROM candles WHERE market=? AND "
            "timestamp>=? AND timestamp<? ORDER BY timestamp", (m, start, end)
        ).fetchall()
        if rows:
            paths[(m, dt.date())] = rows
    conn.close()
    return paths


# ============================================================================
# 4. 경로순서 청산 시뮬 (시간순 walk, leak 없음)
# ============================================================================
def simulate_path(bars: list, hard_sl, tp, time_stop_h, trail) -> tuple[float, str, float]:
    """15m 봉을 시간순으로 walk 하며 청산정책 적용 → (gross_return, outcome, hold_hours).

    진입가 = 첫 봉 open (= day-D open). 각 봉 안에서:
      1) hard SL: low <= entry*(1-hard_sl) → -hard_sl 청산 (봉 내 SL 우선=보수)
      2) trailing SL: low <= running_high*(1-trail) → trailing 가격 청산
      3) TP: high >= entry*(1+tp) → +tp 청산
      4) time-stop: 진입후 time_stop_h 경과 봉 종가에서 무조건 청산
    아무것도 안 걸리면 EOD = 마지막 봉 close.
    같은 봉에서 SL과 TP 동시 → SL 먼저 (보수, 봉 내부 ambiguous).
    """
    if not bars:
        return np.nan, "nodata", np.nan
    entry = bars[0][0]
    if not np.isfinite(entry) or entry <= 0:
        return np.nan, "nodata", np.nan
    sl_px = entry * (1 - hard_sl) if hard_sl is not None else None
    tp_px = entry * (1 + tp) if tp is not None else None
    ts_bars = int(round(time_stop_h * BARS_PER_HOUR)) if time_stop_h is not None else None
    running_high = entry
    for i, (o, h, l, c) in enumerate(bars):
        # trailing 기준 고점 갱신 (이 봉 high 포함)
        if trail is not None and h > running_high:
            running_high = h
        trail_px = running_high * (1 - trail) if trail is not None else None
        # --- 청산 우선순위 (봉 내 동시 → SL 먼저=보수) ---
        # 1) hard SL
        if sl_px is not None and l <= sl_px:
            hold = (i + 1) / BARS_PER_HOUR
            return -hard_sl, "sl", hold
        # 2) trailing SL (hard SL보다 낮을 수 있음 — 둘 중 먼저 닿는 쪽; 봉단위라 같은 봉이면 trailing은 SL 다음)
        if trail_px is not None and l <= trail_px:
            hold = (i + 1) / BARS_PER_HOUR
            return trail_px / entry - 1.0, "trail", hold
        # 3) TP
        if tp_px is not None and h >= tp_px:
            hold = (i + 1) / BARS_PER_HOUR
            return tp, "tp", hold
        # 4) time-stop (이 봉이 time_stop 경계면 종가 청산)
        if ts_bars is not None and (i + 1) >= ts_bars:
            hold = (i + 1) / BARS_PER_HOUR
            return c / entry - 1.0, "timestop", hold
    # EOD
    last_c = bars[-1][3]
    return last_c / entry - 1.0, "eod", len(bars) / BARS_PER_HOUR


def net_metrics(trades: pd.DataFrame) -> dict:
    """trade DataFrame (date, net, outcome) → net 지표 (periods_per_year=365)."""
    if len(trades) == 0:
        return dict(net_mean=np.nan, net_sharpe=np.nan, hit=np.nan, n=0,
                    cum=np.nan, mdd=np.nan, tp_rate=np.nan, sl_rate=np.nan,
                    trail_rate=np.nan, ts_rate=np.nan, eod_rate=np.nan,
                    avg_hold=np.nan)
    daily = trades.groupby("date")["net"].mean().sort_index()
    mu, sig = daily.mean(), daily.std()
    sharpe = float(mu / sig * np.sqrt(365)) if sig and sig > 0 else np.nan
    eq = (1 + daily).cumprod()
    peak = eq.cummax()
    mdd = float(((eq - peak) / peak).min())
    cum = float(eq.iloc[-1] - 1.0)
    oc = trades["outcome"]
    return dict(
        net_mean=float(trades["net"].mean()),
        net_sharpe=sharpe,
        hit=float((trades["net"] > 0).mean()),
        n=int(len(trades)),
        cum=cum, mdd=mdd,
        tp_rate=float((oc == "tp").mean()),
        sl_rate=float((oc == "sl").mean()),
        trail_rate=float((oc == "trail").mean()),
        ts_rate=float((oc == "timestop").mean()),
        eod_rate=float((oc == "eod").mean()),
        avg_hold=float(trades["hold"].mean()),
    )


# ============================================================================
# 5. what-hit-first 진단 (sl_first 보수 가정이 정당한가)
# ============================================================================
def what_hit_first(bars_map: dict, entries_df: pd.DataFrame, feat: str,
                   tp: float, sl: float) -> dict:
    """+tp 와 -sl 둘 다 닿은 trade 중 실제로 뭐가 먼저였나 (15m 경로순)."""
    both, tp_first, sl_first = 0, 0, 0
    for _, r in entries_df.iterrows():
        bars = bars_map.get((r["market"], r["date"]))
        if not bars:
            continue
        entry = bars[0][0]
        if not np.isfinite(entry) or entry <= 0:
            continue
        tp_px, sl_px = entry * (1 + tp), entry * (1 - sl)
        hit_tp = any(h >= tp_px for (o, h, l, c) in bars)
        hit_sl = any(l <= sl_px for (o, h, l, c) in bars)
        if not (hit_tp and hit_sl):
            continue
        both += 1
        # 경로순서로 누가 먼저
        for (o, h, l, c) in bars:
            sl_now = l <= sl_px
            tp_now = h >= tp_px
            if sl_now and tp_now:
                sl_first += 1  # 같은 봉 → 보수
                break
            if sl_now:
                sl_first += 1
                break
            if tp_now:
                tp_first += 1
                break
    return dict(both=both, tp_first=tp_first, sl_first=sl_first,
                sl_first_share=(sl_first / both) if both else np.nan)


# ============================================================================
# main
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-markets", type=int, default=None)
    args = ap.parse_args()
    OUT_DIR.mkdir(exist_ok=True)

    panel = build_panel(args.limit_markets)
    panel = add_cross_sectional(panel)
    panel = attach_bear_quiet(panel)
    log.info("panel rows=%d markets=%d dates=%d",
             len(panel), panel["market"].nunique(), panel["date"].nunique())

    # 커버리지: d1 bear_quiet 표본 vs 15m 윈도우 겹침
    conn = sqlite3.connect(M15_DB)
    m15_min, m15_max = conn.execute(
        "SELECT MIN(timestamp), MAX(timestamp) FROM candles").fetchone()
    conn.close()
    m15_start_date = pd.Timestamp(m15_min).date()
    bq_days_all = panel[panel["is_bear_quiet"] == 1.0]["date"].nunique()
    bq_days_in15 = panel[(panel["is_bear_quiet"] == 1.0) &
                         (panel["date"] >= m15_start_date)]["date"].nunique()
    log.info("15m coverage: %s -> %s", m15_min, m15_max)
    log.info("bear_quiet days: full=%d, in-15m-window(>=%s)=%d",
             bq_days_all, m15_start_date, bq_days_in15)

    grid = list(product(HARD_SL, TP, TIME_STOP_H, TRAIL))
    log.info("SELECTION BIAS: candidates=%d × exit-grid=%d = %d (policy×pattern)",
             len(CANDIDATES), len(grid), len(CANDIDATES) * len(grid))

    all_rows = []
    whf_rows = []
    coverage_rows = []
    for feat in CANDIDATES:
        entries = collect_entries(panel, feat)
        if len(entries) == 0:
            log.info("[%s] no entries", feat)
            continue
        # 15m 윈도우 내 진입만 경로검증 가능
        entries = entries[entries["date"] >= m15_start_date].reset_index(drop=True)
        bars_map = load_paths(entries)
        # 실제 경로 있는 진입만
        entries = entries[entries.apply(
            lambda r: (r["market"], r["date"]) in bars_map, axis=1)].reset_index(drop=True)
        n_entry = len(entries)
        n_coinday = entries[["market", "date"]].drop_duplicates().shape[0]
        n_days = entries["date"].nunique()
        coverage_rows.append(dict(feature=feat, n_trades=n_entry,
                                  n_coinday=n_coinday, n_days=n_days))
        log.info("[%s] entries with 15m path: trades=%d coin-day=%d days=%d",
                 feat, n_entry, n_coinday, n_days)
        if n_entry < 20:
            log.info("[%s] too few — skip grid", feat)
            continue

        # what-hit-first (대표 TP+10/SL-5)
        whf = what_hit_first(bars_map, entries, feat, tp=0.10, sl=0.05)
        whf.update(feature=feat, tp=0.10, sl=0.05)
        whf_rows.append(whf)

        # 그리드 시뮬
        for (sl, tp, ts, tr) in grid:
            recs = []
            for _, r in entries.iterrows():
                bars = bars_map[(r["market"], r["date"])]
                gross, outcome, hold = simulate_path(bars, sl, tp, ts, tr)
                if not np.isfinite(gross):
                    continue
                recs.append((r["date"], gross - ROUND_TRIP_COST, outcome, hold))
            if not recs:
                continue
            td = pd.DataFrame(recs, columns=["date", "net", "outcome", "hold"])
            mt = net_metrics(td)
            mt.update(feature=feat,
                      hard_sl=(sl if sl is not None else "none"),
                      tp=(tp if tp is not None else "none"),
                      time_stop_h=(ts if ts is not None else "none"),
                      trail=(tr if tr is not None else "none"))
            all_rows.append(mt)

    res = pd.DataFrame(all_rows)
    cols = ["feature", "hard_sl", "tp", "time_stop_h", "trail", "n",
            "net_mean", "net_sharpe", "hit", "cum", "mdd",
            "tp_rate", "sl_rate", "trail_rate", "ts_rate", "eod_rate", "avg_hold"]
    res = res[cols].sort_values(["feature", "net_sharpe"], ascending=[True, False])
    res_path = OUT_DIR / "bear_quiet_path_exit_v1.csv"
    res.to_csv(res_path, index=False)
    log.info("wrote %s (%d rows)", res_path, len(res))

    whf_df = pd.DataFrame(whf_rows)
    whf_path = OUT_DIR / "bear_quiet_path_what_hit_first_v1.csv"
    whf_df.to_csv(whf_path, index=False)
    cov_df = pd.DataFrame(coverage_rows)
    cov_df.to_csv(OUT_DIR / "bear_quiet_path_coverage_v1.csv", index=False)

    # ---- 콘솔 요약 ----
    log.info("\n===== WHAT HIT FIRST (TP+10 vs SL-5, 15m path order) =====")
    for _, r in whf_df.iterrows():
        log.info("  %-22s both=%d tp_first=%d sl_first=%d (sl_first_share=%.2f)",
                 r["feature"], int(r["both"]), int(r["tp_first"]),
                 int(r["sl_first"]),
                 r["sl_first_share"] if pd.notna(r["sl_first_share"]) else float("nan"))

    log.info("\n===== TOP POLICIES BY net_sharpe (per candidate, n>=20) =====")
    for feat in CANDIDATES:
        ss = res[(res["feature"] == feat) & (res["n"] >= 20)].head(6)
        if len(ss) == 0:
            continue
        log.info("--- %s ---", feat)
        for _, r in ss.iterrows():
            log.info("  SL=%-4s TP=%-4s ts=%-4s trail=%-4s | net/trade=%+.4f "
                     "sharpe=%s hit=%.2f cum=%+.3f mdd=%.3f n=%d "
                     "[tp=%.2f sl=%.2f tr=%.2f ts=%.2f eod=%.2f hold=%.1fh]",
                     str(r["hard_sl"]), str(r["tp"]), str(r["time_stop_h"]),
                     str(r["trail"]), r["net_mean"],
                     f"{r['net_sharpe']:+.2f}" if pd.notna(r["net_sharpe"]) else " nan",
                     r["hit"], r["cum"], r["mdd"], int(r["n"]),
                     r["tp_rate"], r["sl_rate"], r["trail_rate"], r["ts_rate"],
                     r["eod_rate"], r["avg_hold"])

    # baseline EOD (SL/TP/ts/trail 모두 none) 별도 표기
    log.info("\n===== BASELINE EOD (no SL/TP/ts/trail) =====")
    base = res[(res["hard_sl"] == "none") & (res["tp"] == "none") &
               (res["time_stop_h"] == "none") & (res["trail"] == "none")]
    for _, r in base.iterrows():
        log.info("  %-22s net/trade=%+.4f sharpe=%s hit=%.2f cum=%+.3f mdd=%.3f n=%d",
                 r["feature"], r["net_mean"],
                 f"{r['net_sharpe']:+.2f}" if pd.notna(r["net_sharpe"]) else " nan",
                 r["hit"], r["cum"], r["mdd"], int(r["n"]))


if __name__ == "__main__":
    main()

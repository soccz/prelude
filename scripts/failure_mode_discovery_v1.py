"""Failure-mode / anti-pattern mining (Angle 4).

목표: 선행패턴이 깃발을 세웠는데 **돈을 잃는 경우**를 전체 히스토리로 mining.
핵심 현상 (RESEARCH §5.2): "≥20% high 를 찍어도 종가 유지 실패 (pump-then-dump)
→ 종가유지가 base rate 40% 깎음". TP 도달 전 붕괴.

이 스크립트가 답하려는 것:
  Q1. 깃발 후 pump-then-dump(high 찍고 EOD 음수) vs sustain 을 가르는
      D-1 이전 조건은? (소진상태/regime/저유동성?)
  Q2. 각 코어 패턴(qv_surge/bounce/momentum/ATR)의 false-positive 프로파일:
      안 터지는 다수는 어떤 D-1 특성인가?
  Q3. 어떤 exit(TP%/EOD/-SL)가 각 패턴의 net 을 살리나 (일봉 OHLC 한계 명시).

★★ LEAK 방어 (same-day leak 2번 전적) ★★
  - 모든 선행조건/regime 분류 = D-1 및 이전 데이터로만.
    build_market_features 가 raw 지표를 market 별 .shift(1) → f_* 는 전부 D-1 까지.
  - BTC regime 도 compute_btc_features 후 date 별 .shift(1) → day-D row 는 D-1 regime.
  - day-D high/low/close 는 (a) 라벨 (b) PnL 경로 시뮬에만 쓰인다. feature 엔 절대 X.
  - 패턴 fire 조건 cutoff 는 walk-forward train 구간에서만 결정 (OOS 평가).
  - selection bias: 시도한 패턴×regime×exit 조합 수를 로그/CSV 에 기록.

PnL 경로 모델 (일봉 OHLC 한계 명시):
  진입 = day-D open (깃발은 D-1 마감 후 → 09:00 시가 진입 가정).
  exit 정책별:
    EOD     : day-D close 청산.
    TP_EOD  : day-D high >= open*(1+TP) → TP 체결, 아니면 close 청산.
    TP_SL   : SL 우선(보수, ledger/config SL_PRIORITY). 같은 날 low<=open*(1-SL) → SL,
              else high>=open*(1+TP) → TP, else close.  (일봉이라 intrabar 순서 미상 →
              SL 우선 = 비관적. 15m path 가 있으면 정밀화 가능, 여기선 일봉 한계 명시.)
  net = gross_exit_ret - ROUND_TRIP_COST_PCT (0.15%).

사용:
    python scripts/failure_mode_discovery_v1.py
    python scripts/failure_mode_discovery_v1.py --limit-markets 40   # 개발
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.database import list_markets, load_candles
from data.market_universe import signal_eligible_markets
from signals.features import compute_btc_features
# 패널/피처 빌더는 angle-A 스크립트 재사용 (동일 leak 규율)
from scripts.univariate_precursor_lift_v1 import (
    build_market_features,
    add_cross_sectional,
)

DB_PATH = "data/upbit_d1.db"
OUT_DIR = Path("output")
EPS = 1e-12

# 거래/청산 파라미터 (placeholder — ledger/config 와 정합)
ROUND_TRIP_COST = 0.0015     # 0.15% 왕복 (수수료 0.1% + 슬리피지 0.05%)
TP_PCT = 0.10                # +10% 익절
SL_PCT = 0.05                # -5% 손절
SL_PRIORITY = True           # 같은 봉 TP/SL 동시 → SL 먼저 (보수)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("failmode")


# ============================================================================
# 1. Panel (leak-free f_* + day-D OHLC for outcome) + BTC regime (D-1)
# ============================================================================
def build_outcome_panel(limit_markets: int | None) -> pd.DataFrame:
    """univariate 스크립트의 build_market_features 를 쓰되, day-D OHLC 를
    outcome(PnL 경로) 계산용으로 추가 보관한다. feature 엔 안 섞는다."""
    markets = signal_eligible_markets(list_markets(DB_PATH))
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
        feat = build_market_features(df)  # leak-free f_* + lab_* + intraday_high_ret
        # ---- outcome 용 day-D OHLC (타겟 측, feature 아님) ----
        g = df.sort_values("timestamp").reset_index(drop=True)
        oc = pd.DataFrame({
            "market": m,
            "timestamp": pd.to_datetime(g["timestamp"]),
            "d_open": g["open"].values,
            "d_high": g["high"].values,
            "d_low": g["low"].values,
            "d_close": g["close"].values,
        })
        feat = feat.copy()
        feat["timestamp"] = pd.to_datetime(feat["timestamp"])
        merged = feat.merge(oc, on=["market", "timestamp"], how="left")
        frames.append(merged)
        if (i + 1) % 50 == 0:
            log.info("  %d/%d markets", i + 1, len(markets))
    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = panel["timestamp"].dt.date
    panel = panel.sort_values(["date", "market"]).reset_index(drop=True)
    return panel


def attach_btc_regime(panel: pd.DataFrame) -> pd.DataFrame:
    """BTC regime 을 D-1 기준으로 붙인다 (leak-free).
    compute_btc_features 는 그날까지의 regime → date 별 .shift(1) 후 merge."""
    btc = load_candles(DB_PATH, "KRW-BTC")
    btc = btc.sort_values("timestamp").reset_index(drop=True)
    bf = compute_btc_features(btc)  # btc_*, btc_regime, btc_regime_id
    bf["timestamp"] = pd.to_datetime(bf["timestamp"])
    bf["date"] = bf["timestamp"].dt.date
    bf = bf.sort_values("date").reset_index(drop=True)
    # ★ 그날 row 는 어제(D-1) regime 을 본다
    keep = ["btc_regime", "btc_ma_distance", "btc_intensity_d", "btc_return_7d"]
    for c in keep:
        bf[f"d1_{c}"] = bf[c].shift(1)
    bf_d1 = bf[["date"] + [f"d1_{c}" for c in keep]].copy()
    out = panel.merge(bf_d1, on="date", how="left")
    out = out.rename(columns={"d1_btc_regime": "btc_regime_d1"})
    return out


# ============================================================================
# 2. Outcome layer — flag worked? sustain vs dump? net PnL per exit policy
# ============================================================================
def add_outcomes(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy()
    o = p["d_open"]; h = p["d_high"]; l = p["d_low"]; c = p["d_close"]
    # 진입(open) 기준 경로
    p["o2h"] = h / (o + EPS) - 1.0     # 장중 최대 상승 (라벨 origin 과 동일 정의)
    p["o2c"] = c / (o + EPS) - 1.0     # 종가 수익 (sustain proxy)
    p["o2l"] = l / (o + EPS) - 1.0     # 장중 최대 하락

    # 깃발이 "방향성으로 맞았나" = 장중 +20%/15% 도달
    p["hit_high20"] = (p["o2h"] >= 0.20).astype(float)
    p["hit_high15"] = (p["o2h"] >= 0.15).astype(float)
    # ---- spike 후 결말 (RESEARCH §5.2: high 찍어도 종가유지 실패) ----
    cond_spike = p["o2h"] >= 0.15
    # retention = 종가가 그날 고점 대비 얼마나 유지됐나 (1=고점마감, 0=완전반납)
    # close-to-high ratio relative to the run-up: (o2c)/(o2h)
    p["spike_retention"] = np.where(cond_spike, p["o2c"] / (p["o2h"] + EPS), np.nan)
    # pump_then_dump (강): +15% high 후 종가 음수 (open 대비 손실로 마감)
    p["pump_then_dump"] = ((cond_spike) & (p["o2c"] < 0.0)).astype(float)
    # fade (약): +15% high 후 종가가 고점의 절반도 못 지킴 (retention<0.5)
    #   = §5.2 "종가유지 실패" 의 더 충실한 정의 (TP-before-SL 함정 영역)
    p["spike_fade"] = ((cond_spike) & (p["spike_retention"] < 0.5)).astype(float)
    # sustain: +15% high 찍고 종가도 +5% 이상 유지
    p["sustain"] = ((cond_spike) & (p["o2c"] >= 0.05)).astype(float)
    # spike 조건일 때만 dump/sustain 정의 (그 외 NaN 으로 분모 제어)
    p.loc[~cond_spike, ["pump_then_dump", "spike_fade", "sustain"]] = np.nan

    # ---- net PnL per exit policy ----
    tp = o * (1 + TP_PCT)
    sl = o * (1 - SL_PCT)
    # EOD
    p["net_eod"] = p["o2c"] - ROUND_TRIP_COST
    # TP_EOD: high 가 TP 도달 → +TP, else close
    hit_tp = (h >= tp)
    p["net_tp_eod"] = np.where(hit_tp, TP_PCT, p["o2c"]) - ROUND_TRIP_COST
    # TP_SL (SL 우선, 보수)
    hit_sl = (l <= sl)
    if SL_PRIORITY:
        ret_tpsl = np.where(hit_sl, -SL_PCT, np.where(hit_tp, TP_PCT, p["o2c"]))
    else:
        ret_tpsl = np.where(hit_tp, TP_PCT, np.where(hit_sl, -SL_PCT, p["o2c"]))
    p["net_tp_sl"] = ret_tpsl - ROUND_TRIP_COST
    return p


# ============================================================================
# 3. Core pattern fire flags (D-1 only) — walk-forward cutoff
# ============================================================================
# 각 코어 패턴: (feature, 방향). top-decile fire 를 walk-forward train 에서 결정.
CORE_PATTERNS = {
    "qv_surge_30d":  ("f_qv_surge_30d", "high"),
    "qv_surge_7d":   ("f_qv_surge_7d", "high"),
    "bounce_7d_low": ("f_bounce_off_7d_low", "high"),
    "mom_ret_7d":    ("f_ret_7d", "high"),
    "mom_ret_3d":    ("f_ret_3d", "high"),
    "atr_pct_14":    ("f_atr_pct_14", "high"),
    "rv_21d":        ("f_rv_21d", "high"),
}

# 깃발 후 dump 를 가르는 후보 D-1 컨디션 (exhaustion / regime / 유동성)
SPLIT_CONDS = [
    "f_ret_7d", "f_ret_3d", "f_ret_14d",   # 이미 많이 올랐나 (소진)
    "f_rsi_14", "f_up_streak",
    "f_dist_from_20d_high", "f_dist_from_60d_high", "f_pos_in_20d_range",
    "f_bounce_off_7d_low",
    "f_atr_pct_14", "f_rv_21d", "f_qv_surge_7d", "f_qv_surge_30d",
    "f_qv_rank", "f_log_qv",                # 유동성 (저유동성 = dump 위험?)
]


def wf_fire_mask(panel: pd.DataFrame, feat: str, direction: str,
                 decile: float, n_folds: int = 5, embargo_days: int = 5) -> pd.Series:
    """walk-forward: train 구간 quantile cutoff 로 test 구간만 fire 표시.
    train 구간 자체는 fire=NaN (OOS 만 평가). leak-free cutoff."""
    dates = np.sort(panel["date"].unique())
    fold_size = len(dates) // (n_folds + 1)
    fire = pd.Series(np.nan, index=panel.index)
    d = panel[[feat]].copy()
    for k in range(1, n_folds + 1):
        train_end_idx = fold_size * k
        test_start_idx = train_end_idx + embargo_days
        test_end_idx = min(fold_size * (k + 1) + train_end_idx, len(dates))
        if test_start_idx >= len(dates):
            break
        train_dates = set(dates[:train_end_idx])
        test_dates = set(dates[test_start_idx:test_end_idx])
        tr = panel[panel["date"].isin(train_dates)]
        tr_feat = tr[feat].dropna()
        if len(tr_feat) < 300:
            continue
        if direction == "high":
            cut = tr_feat.quantile(decile)
            mask_test = panel["date"].isin(test_dates) & (panel[feat] >= cut)
        else:
            cut = tr_feat.quantile(1 - decile)
            mask_test = panel["date"].isin(test_dates) & (panel[feat] <= cut)
        fire.loc[mask_test[mask_test].index] = 1.0
    return fire


def summarize(sub: pd.DataFrame, label: str, n: int) -> dict:
    """fire 한 표본 집합의 outcome 요약."""
    if len(sub) == 0:
        return {}
    o2h = sub["o2h"]
    spike = (o2h >= 0.15)
    n_spike = int(spike.sum())
    res = {
        "n_fire": int(len(sub)),
        "hit_high20_rate": float(sub["hit_high20"].mean()),
        "hit_high15_rate": float(sub["hit_high15"].mean()),
        "n_spike15": n_spike,
        # spike 중 dump / fade / sustain
        "pump_then_dump_rate": float(sub.loc[spike, "pump_then_dump"].mean()) if n_spike else np.nan,
        "spike_fade_rate": float(sub.loc[spike, "spike_fade"].mean()) if n_spike else np.nan,
        "sustain_rate": float(sub.loc[spike, "sustain"].mean()) if n_spike else np.nan,
        "mean_retention": float(sub.loc[spike, "spike_retention"].mean()) if n_spike else np.nan,
        # net PnL per exit (전체 fire 표본)
        "net_eod_mean": float(sub["net_eod"].mean()),
        "net_eod_med": float(sub["net_eod"].median()),
        "net_tp_eod_mean": float(sub["net_tp_eod"].mean()),
        "net_tp_sl_mean": float(sub["net_tp_sl"].mean()),
        # net win rate
        "net_eod_winrate": float((sub["net_eod"] > 0).mean()),
        "net_tp_sl_winrate": float((sub["net_tp_sl"] > 0).mean()),
        # 진단: 평균 장중 고점/저점/종가
        "mean_o2h": float(o2h.mean()),
        "mean_o2c": float(sub["o2c"].mean()),
        "mean_o2l": float(sub["o2l"].mean()),
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-markets", type=int, default=None)
    ap.add_argument("--decile", type=float, default=0.9,
                    help="패턴 fire top-decile cutoff (초기값 0.9)")
    ap.add_argument("--universe-top", type=int, default=None,
                    help="D-1 거래대금 top-N 유니버스 제한 (None=전체)")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    panel = build_outcome_panel(args.limit_markets)
    panel = add_cross_sectional(panel)
    panel = attach_btc_regime(panel)
    panel = add_outcomes(panel)
    # outcome 결측(마지막날 등) 제거
    panel = panel.dropna(subset=["d_open", "d_close", "o2h"]).reset_index(drop=True)
    log.info("panel rows=%d markets=%d dates=%d",
             len(panel), panel["market"].nunique(), panel["date"].nunique())

    if args.universe_top:
        panel = panel[panel["f_qv_rank"] <= args.universe_top].copy()
        log.info("universe top-%d → rows=%d", args.universe_top, len(panel))

    # 전체 base rate (참고)
    log.info("base: hit_high20=%.4f hit_high15=%.4f | among spike15: dump=%.3f sustain=%.3f",
             panel["hit_high20"].mean(), panel["hit_high15"].mean(),
             panel.loc[panel["o2h"] >= 0.15, "pump_then_dump"].mean(),
             panel.loc[panel["o2h"] >= 0.15, "sustain"].mean())
    log.info("base net (all rows): eod=%.4f tp_eod=%.4f tp_sl=%.4f",
             panel["net_eod"].mean(), panel["net_tp_eod"].mean(), panel["net_tp_sl"].mean())

    # ---- selection bias 기록 ----
    n_patterns = len(CORE_PATTERNS)
    regimes = ["ALL", "bull_quiet", "bull_volatile", "bear_quiet", "bear_volatile"]
    n_split = len(SPLIT_CONDS)
    log.info("SELECTION BIAS: patterns=%d × regimes=%d (Q1/Q2) ; "
             "split-conds=%d × patterns=%d (Q3 dump-separator) ; exits=3",
             n_patterns, len(regimes), n_split, n_patterns)

    # ========================================================================
    # Q2 + Q3: 패턴별 × regime별 fire outcome (false-positive + net per exit)
    # ========================================================================
    rows = []
    fire_cache = {}
    for pname, (feat, direction) in CORE_PATTERNS.items():
        fire = wf_fire_mask(panel, feat, direction, args.decile)
        fire_cache[pname] = fire
        fired = panel[fire == 1.0]
        for reg in regimes:
            if reg == "ALL":
                sub = fired
            else:
                sub = fired[fired["btc_regime_d1"] == reg]
            if len(sub) < 30:
                continue
            res = {"pattern": pname, "feature": feat, "regime": reg}
            res.update(summarize(sub, pname, len(sub)))
            rows.append(res)
    pat_df = pd.DataFrame(rows)
    pat_path = OUT_DIR / "failure_mode_pattern_regime_v1.csv"
    pat_df.to_csv(pat_path, index=False)
    log.info("wrote %s (%d rows)", pat_path, len(pat_df))

    # ========================================================================
    # Q1: 깃발(패턴 fire) 후 dump vs sustain 을 가르는 D-1 split 조건
    #   각 split-cond 를 패턴-fire 표본 내에서 high/low half 로 나눠
    #   sustain_rate, net_tp_sl 차이를 본다 (조건부 lift).
    # ========================================================================
    split_rows = []
    for pname, (feat, direction) in CORE_PATTERNS.items():
        fire = fire_cache[pname]
        fired = panel[fire == 1.0].copy()
        if len(fired) < 100:
            continue
        # spike 표본 (high 찍은 것들) 안에서 dump 분리
        spike = fired[fired["o2h"] >= 0.15].copy()
        for cond in SPLIT_CONDS:
            if cond not in fired.columns:
                continue
            d = fired[[cond, "net_tp_sl", "net_eod", "o2h", "o2c", "sustain", "pump_then_dump"]].dropna(subset=[cond])
            if len(d) < 80:
                continue
            med = d[cond].median()
            hi = d[d[cond] >= med]
            lo = d[d[cond] < med]
            if len(hi) < 30 or len(lo) < 30:
                continue
            # spike 내 dump 분리 (조건의 high half 가 dump 를 더 만드나)
            sp = spike[[cond, "sustain", "pump_then_dump", "net_tp_sl"]].dropna(subset=[cond])
            sp_dump_hi = sp_dump_lo = np.nan
            sp_sustain_hi = sp_sustain_lo = np.nan
            if len(sp) >= 60:
                m2 = sp[cond].median()
                sphi = sp[sp[cond] >= m2]; splo = sp[sp[cond] < m2]
                if len(sphi) >= 20 and len(splo) >= 20:
                    sp_dump_hi = float(sphi["pump_then_dump"].mean())
                    sp_dump_lo = float(splo["pump_then_dump"].mean())
                    sp_sustain_hi = float(sphi["sustain"].mean())
                    sp_sustain_lo = float(splo["sustain"].mean())
            split_rows.append({
                "pattern": pname, "split_cond": cond, "median_cut": float(med),
                "n_hi": int(len(hi)), "n_lo": int(len(lo)),
                # 전체 fire 표본: net_tp_sl high vs low
                "net_tp_sl_hi": float(hi["net_tp_sl"].mean()),
                "net_tp_sl_lo": float(lo["net_tp_sl"].mean()),
                "net_tp_sl_diff": float(hi["net_tp_sl"].mean() - lo["net_tp_sl"].mean()),
                "net_eod_hi": float(hi["net_eod"].mean()),
                "net_eod_lo": float(lo["net_eod"].mean()),
                # spike 내 dump/sustain 분리력
                "spike_dump_hi": sp_dump_hi, "spike_dump_lo": sp_dump_lo,
                "spike_sustain_hi": sp_sustain_hi, "spike_sustain_lo": sp_sustain_lo,
                "n_spike": int(len(sp)),
            })
    split_df = pd.DataFrame(split_rows)
    if len(split_df):
        split_df["abs_net_diff"] = split_df["net_tp_sl_diff"].abs()
        split_df = split_df.sort_values(["pattern", "abs_net_diff"], ascending=[True, False])
    split_path = OUT_DIR / "failure_mode_dump_separator_v1.csv"
    split_df.to_csv(split_path, index=False)
    log.info("wrote %s (%d rows)", split_path, len(split_df))

    # ---- 콘솔 요약 ----
    log.info("=== Q2/Q3: 패턴별 ALL-regime net per exit ===")
    for _, r in pat_df[pat_df["regime"] == "ALL"].iterrows():
        log.info("  %-14s n=%5d hit20=%.3f spike_dump=%.2f sustain=%.2f | "
                 "net eod=%+.4f tp_eod=%+.4f tp_sl=%+.4f (eod_win=%.2f)",
                 r["pattern"], int(r["n_fire"]), r["hit_high20_rate"],
                 r["pump_then_dump_rate"], r["sustain_rate"],
                 r["net_eod_mean"], r["net_tp_eod_mean"], r["net_tp_sl_mean"],
                 r["net_eod_winrate"])

    log.info("=== Q1: 상위 dump-separator (|net_tp_sl_diff| 큰 순, 패턴당 top2) ===")
    if len(split_df):
        for pname in CORE_PATTERNS:
            ss = split_df[split_df["pattern"] == pname].head(2)
            for _, r in ss.iterrows():
                log.info("  %-14s split=%-22s net_tp_sl hi=%+.4f lo=%+.4f diff=%+.4f | "
                         "spike_dump hi=%.2f lo=%.2f",
                         r["pattern"], r["split_cond"], r["net_tp_sl_hi"],
                         r["net_tp_sl_lo"], r["net_tp_sl_diff"],
                         r["spike_dump_hi"], r["spike_dump_lo"])


if __name__ == "__main__":
    main()

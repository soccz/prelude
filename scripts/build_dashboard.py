"""Dashboard JSON builder — paper_ledger 두 개 → soccz.github.io/projects/prelude/dashboard/data/*.json

산출물 (3개):
  summary.json   — 채널별 KPI (총 alert, hit rate, 가상 누적 PnL)
  history.json   — 전체 알림 행 (예측 + 실제 OHLC + 결과). 날짜 내림차순.
  accuracy.json  — rolling 30일 hit rate 시계열

가상 PnL 룰 (텔레그램 가이드와 동일):
  - 알림 1건 = 1단위 자본 (equal weight)
  - +5% 도달 시 익절, 아니면 EOD close (자동 손절 X)
  - 거래비용 ROUND_TRIP_COST_PCT 차감
  - cum_pnl_pct = sum of per-alert net_return (compounding 안 함, 단순합)

운영:
  매일 close cron 끝에 호출. JSON 만 갱신, html/JS 는 그대로 둔다.

사용:
    python scripts/build_dashboard.py
    python scripts/build_dashboard.py --out-dir <github.io path>
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ledger.config import ROUND_TRIP_COST_PCT
from scripts.idea_validation_report import (
    build_report as build_idea_validation_report,
    load_candidate_ledger as load_idea_candidate_ledger,
)


DEFAULT_OUT_DIR = "/home/soccz/22tb/soccz.github.io/projects/prelude/dashboard/data"
ROLLING_WINDOW_DAYS = 30
TP_PCT = 0.05  # 사용자 가이드: "5% 오르면 즉시 매도"
DEFAULT_PIN = "9963"
PBKDF2_ITERATIONS = 250000


# ============================================================================
# Per-channel metrics
# ============================================================================
def _virtual_pnl_per_alert(max_ret_pct: float, close_ret_pct: float) -> float:
    """텔레그램 가이드 룰: TP 5% 도달 시 5% 익절, 아니면 EOD close. 비용 차감.

    Returns: net return as decimal (e.g. 0.034 == +3.4%).
    """
    if pd.isna(max_ret_pct):
        return np.nan
    max_d = max_ret_pct / 100.0
    close_d = (close_ret_pct or 0.0) / 100.0
    gross = TP_PCT if max_d >= TP_PCT else close_d
    return gross - ROUND_TRIP_COST_PCT


def _per_alert_pnl_series(closed: pd.DataFrame, max_col: str, close_col: str) -> pd.Series:
    """알림별 net PnL (TP rule + 비용) — vectorized.

    gross = max_d >= TP_PCT ? TP_PCT : close_d
    net = gross - ROUND_TRIP_COST_PCT
    NaN max → drop.
    """
    if len(closed) == 0 or max_col not in closed.columns:
        return pd.Series(dtype=float)
    max_d = pd.to_numeric(closed[max_col], errors="coerce") / 100.0
    close_d = pd.to_numeric(closed.get(close_col, np.nan), errors="coerce") / 100.0
    valid = max_d.notna()
    gross = np.where(max_d >= TP_PCT, TP_PCT, close_d)
    pnl = pd.Series(gross - ROUND_TRIP_COST_PCT, index=closed.index)
    return pnl[valid].dropna()


def compute_quant_metrics(closed: pd.DataFrame, max_col: str, close_col: str) -> dict:
    """헤드라인 metric — Sharpe/Sortino/Calmar/Profit Factor/Expectancy.

    daily aggregation 으로 Sharpe (그날 net 의 평균 / std × √252).
    Calmar = annualized return / |MDD|.
    Profit Factor = sum(positive) / |sum(negative)|.
    Expectancy = avg per trade.
    """
    pnl = _per_alert_pnl_series(closed, max_col, close_col)
    if len(pnl) == 0:
        return {"n_trades": 0}

    closed = closed.copy()
    closed["pnl"] = closed.apply(
        lambda r: _virtual_pnl_per_alert(r[max_col], r[close_col]), axis=1
    )
    closed = closed.dropna(subset=["pnl"])
    closed["date_dt"] = pd.to_datetime(closed["date"])

    # daily aggregate (같은 날 알림 N 개면 그날 net = mean)
    daily = closed.groupby(closed["date_dt"].dt.date)["pnl"].sum().sort_index()
    n_days = len(daily)
    avg_daily = float(daily.mean())
    std_daily = float(daily.std(ddof=1)) if n_days > 1 else float("nan")
    downside_daily = daily[daily < 0]
    downside_std = (
        float(downside_daily.std(ddof=1)) if len(downside_daily) > 1 else float("nan")
    )

    sharpe = (avg_daily / std_daily) * np.sqrt(252) if std_daily and std_daily > 0 else float("nan")
    sortino = (avg_daily / downside_std) * np.sqrt(252) if downside_std and downside_std > 0 else float("nan")

    cum = daily.cumsum()
    mdd = float((cum - cum.cummax()).min())
    annualized_return = avg_daily * 252
    calmar = annualized_return / abs(mdd) if mdd < 0 else float("nan")

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    profit_factor = (
        float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() < 0 else float("nan")
    )
    expectancy = float(pnl.mean())
    avg_win = float(wins.mean()) if len(wins) else float("nan")
    avg_loss = float(losses.mean()) if len(losses) else float("nan")
    win_loss_ratio = float(abs(avg_win / avg_loss)) if avg_loss and not np.isnan(avg_loss) and avg_loss < 0 else float("nan")

    return {
        "n_trades": int(len(pnl)),
        "n_days": int(n_days),
        "n_wins": int(len(wins)),
        "n_losses": int(len(losses)),
        "sharpe_ann": _maybe_float(sharpe),
        "sortino_ann": _maybe_float(sortino),
        "calmar": _maybe_float(calmar),
        "profit_factor": _maybe_float(profit_factor),
        "expectancy_pct": _maybe_float(expectancy * 100),
        "avg_win_pct": _maybe_float(avg_win * 100) if len(wins) else None,
        "avg_loss_pct": _maybe_float(avg_loss * 100) if len(losses) else None,
        "win_loss_ratio": _maybe_float(win_loss_ratio),
        "annualized_return_pct": _maybe_float(annualized_return * 100),
    }


def _maybe_float(v):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return None
    return round(float(v), 4)


def compute_bootstrap_ci(closed: pd.DataFrame, max_col: str, close_col: str,
                         n_iter: int = 1000, seed: int = 42) -> dict:
    """Trade-level bootstrap 95% CI for cum PnL, mean PnL, Sharpe.

    표본이 28 정도라 CI 는 매우 넓을 것 — 그게 "정직한" 신호.
    """
    pnl = _per_alert_pnl_series(closed, max_col, close_col).reset_index(drop=True)
    n = len(pnl)
    if n < 5:
        return {"n_trades": int(n), "note": "n<5: bootstrap skipped"}
    rng = np.random.default_rng(seed)
    cum_samples = np.empty(n_iter)
    mean_samples = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        sample = pnl.iloc[idx].values
        cum_samples[i] = sample.sum()
        mean_samples[i] = sample.mean()
    return {
        "n_trades": int(n),
        "n_iter": int(n_iter),
        "cum_pnl_pct_ci95": [
            _maybe_float(np.quantile(cum_samples, 0.025) * 100),
            _maybe_float(np.quantile(cum_samples, 0.975) * 100),
        ],
        "mean_pnl_pct_ci95": [
            _maybe_float(np.quantile(mean_samples, 0.025) * 100),
            _maybe_float(np.quantile(mean_samples, 0.975) * 100),
        ],
    }


def _daily_pnl_from_closed(closed: pd.DataFrame, max_col: str, close_col: str) -> pd.Series:
    """closed 행 → 일별 net PnL series (TP rule + cost, vectorized).
    같은 날 N알림이면 sum.
    """
    if len(closed) == 0:
        return pd.Series(dtype=float)
    pnl = _per_alert_pnl_series(closed, max_col, close_col)
    if len(pnl) == 0:
        return pd.Series(dtype=float)
    dates = pd.to_datetime(closed.loc[pnl.index, "date"])
    sub = pd.DataFrame({"pnl": pnl.values, "date_d": dates.dt.date.values})
    daily = sub.groupby("date_d")["pnl"].sum().sort_index()
    daily.index = pd.to_datetime(daily.index)
    return daily


def compute_distribution_stats(closed: pd.DataFrame, max_col: str, close_col: str) -> dict:
    """업계 표준 — Vol(ann) / Skew / Kurt / VaR / CVaR / Tail Ratio /
    Recovery Factor / Ulcer Index / Common Sense Ratio / Win-Loss streak."""
    pnl = _per_alert_pnl_series(closed, max_col, close_col)
    if len(pnl) == 0:
        return {}
    daily = _daily_pnl_from_closed(closed, max_col, close_col)

    out = {}
    if len(daily) > 1:
        std_d = float(daily.std(ddof=1))
        out["volatility_ann_pct"] = _maybe_float(std_d * np.sqrt(252) * 100)
    out["skewness"] = _maybe_float(float(pnl.skew()) if len(pnl) > 2 else float("nan"))
    out["kurtosis_excess"] = _maybe_float(float(pnl.kurtosis()) if len(pnl) > 3 else float("nan"))

    # VaR / CVaR (95%) on per-trade returns
    if len(pnl) >= 5:
        var95 = float(pnl.quantile(0.05))
        tail = pnl[pnl <= var95]
        cvar95 = float(tail.mean()) if len(tail) else float("nan")
        out["var_95_pct"] = _maybe_float(var95 * 100)
        out["cvar_95_pct"] = _maybe_float(cvar95 * 100)
        # Tail Ratio (right / left tail abs ratio)
        right = abs(float(pnl.quantile(0.95)))
        left = abs(float(pnl.quantile(0.05)))
        out["tail_ratio"] = _maybe_float(right / left if left > 0 else float("nan"))

    # Recovery Factor + Ulcer + Common Sense
    cum = pnl.cumsum()
    if len(cum) and cum.cummax().abs().max() > 0:
        mdd = float((cum - cum.cummax()).min())
        out["recovery_factor"] = _maybe_float(float(cum.iloc[-1]) / abs(mdd) if mdd < 0 else float("nan"))
    # Ulcer index over per-trade equity curve
    if len(cum) > 0:
        dd_pct = (cum - cum.cummax())
        ulcer = float(np.sqrt(np.mean(dd_pct ** 2)))
        out["ulcer_index"] = _maybe_float(ulcer * 100)

    # Common Sense Ratio = profit factor × tail ratio
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    pf = (float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() < 0 else float("nan"))
    out["common_sense_ratio"] = _maybe_float(pf * out.get("tail_ratio", float("nan"))) if not np.isnan(pf) and out.get("tail_ratio") else None

    # Win/Loss streak
    sign_seq = np.sign(pnl.values)
    max_win, max_loss = 0, 0
    cur_w = cur_l = 0
    for s in sign_seq:
        if s > 0:
            cur_w += 1; cur_l = 0
            max_win = max(max_win, cur_w)
        elif s < 0:
            cur_l += 1; cur_w = 0
            max_loss = max(max_loss, cur_l)
        else:
            cur_w = cur_l = 0
    out["max_win_streak"] = int(max_win)
    out["max_loss_streak"] = int(max_loss)

    return out


def compute_psr_dsr(closed: pd.DataFrame, max_col: str, close_col: str,
                     n_trials: int = 50) -> dict:
    """Lopez de Prado 의 PSR / DSR / MinTRL.

    - PSR(SR*=0): "이 Sharpe 가 0 보다 클 확률".
    - DSR(SR_threshold from N trials): selection bias 보정.
    - MinTRL(α=0.95): 신뢰 95% 위해 필요한 표본 길이.

    공식 (per-period, non-annualized SR. wikipedia/Bailey 2014 동일):
      PSR = Φ((SR - SR*) × √(T-1) / √(1 - γ3·SR + (γ4-1)/4·SR²))
      γ3 = skew, γ4 = (excess kurt) + 3
    """
    from scipy.stats import norm
    pnl = _per_alert_pnl_series(closed, max_col, close_col)
    T = len(pnl)
    if T < 5:
        return {"n_trades": int(T), "note": "T<5: PSR/DSR skipped"}

    mu = float(pnl.mean())
    sd = float(pnl.std(ddof=1))
    if sd <= 0:
        return {"n_trades": int(T), "note": "zero variance"}
    sr = mu / sd  # per-trade Sharpe (non-annualized)

    skew = float(pnl.skew()) if T > 2 else 0.0
    kurt_excess = float(pnl.kurtosis()) if T > 3 else 0.0
    kurt = kurt_excess + 3.0
    # PSR variance term — Bailey 2014 (SR_obs 기반)
    psr_var = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    psr_var = max(psr_var, 1e-9)
    psr_z = (sr - 0.0) * np.sqrt(T - 1) / np.sqrt(psr_var)
    psr = float(norm.cdf(psr_z))

    # SR_threshold from N trials (False Strategy Theorem)
    # V[SR̂_n] ≈ (1/T) (independent trial assumption, Bailey 2014 §4.2)
    EULER = 0.5772156649
    if n_trials > 1:
        v_sr = 1.0 / T
        z1 = norm.ppf(1.0 - 1.0 / n_trials)
        z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        sr_threshold = np.sqrt(v_sr) * ((1 - EULER) * z1 + EULER * z2)
    else:
        sr_threshold = 0.0

    # DSR — same denominator as PSR (SR_obs based variance)
    dsr_z = (sr - sr_threshold) * np.sqrt(T - 1) / np.sqrt(psr_var)
    dsr = float(norm.cdf(dsr_z))

    # MinTRL @ alpha=0.95 — Wikipedia/Bailey 의 정확한 공식은
    # SR_threshold 기반 variance term:
    #   MinTRL = 1 + (1 − γ3·SR₀ + (γ4−1)/4·SR₀²) × (Φ⁻¹(α) / (SR_obs − SR₀))²
    alpha = 0.95
    z_alpha = norm.ppf(alpha)
    sr_diff = sr - sr_threshold
    if sr_diff > 0:
        var_threshold = 1.0 - skew * sr_threshold + (kurt - 1.0) / 4.0 * sr_threshold ** 2
        var_threshold = max(var_threshold, 1e-9)
        mintrl = 1.0 + var_threshold * (z_alpha / sr_diff) ** 2
    else:
        mintrl = float("inf")  # 현재 SR < threshold — 절대 도달 불가

    return {
        "n_trades": int(T),
        "sr_per_trade": _maybe_float(sr),
        "skew": _maybe_float(skew),
        "kurt_excess": _maybe_float(kurt_excess),
        "psr": _maybe_float(psr),
        "psr_pct": _maybe_float(psr * 100),
        "n_trials_assumed": int(n_trials),
        "sr_threshold": _maybe_float(sr_threshold),
        "dsr": _maybe_float(dsr),
        "dsr_pct": _maybe_float(dsr * 100),
        "min_trl_trades": _maybe_float(mintrl) if np.isfinite(mintrl) else None,
    }


def compute_benchmark_metrics(closed: pd.DataFrame, max_col: str, close_col: str,
                                btc_series: list[dict]) -> dict:
    """IR / Beta / Tracking Error vs BTC HODL daily returns.

    alert 없는 날의 strat daily PnL = 0 (cash) 가정 후 BTC 모든 날에 align.
    이래야 sparse alert 도 BTC 의 daily series 와 비교 가능.
    """
    if not btc_series or len(btc_series) < 5:
        return {}
    daily = _daily_pnl_from_closed(closed, max_col, close_col)
    if len(daily) == 0:
        return {}
    btc_df = pd.DataFrame(btc_series)
    btc_df["date_dt"] = pd.to_datetime(btc_df["date"])
    btc_df = btc_df.set_index("date_dt").sort_index()
    btc_pct = btc_df["btc_cum_pct"] / 100.0
    btc_daily = btc_pct.diff().dropna()
    if len(btc_daily) < 5:
        return {}

    # Reindex strat daily to BTC's date index — alert 없는 날 = 0 PnL
    strat_daily = daily.reindex(btc_daily.index, fill_value=0.0)

    excess = strat_daily - btc_daily
    te = float(excess.std(ddof=1))
    ir = float(excess.mean() / te * np.sqrt(252)) if te > 0 else float("nan")

    var_b = float(btc_daily.var(ddof=1))
    cov_sb = float(strat_daily.cov(btc_daily))
    beta = (cov_sb / var_b) if var_b > 0 else float("nan")
    corr = float(strat_daily.corr(btc_daily))

    return {
        "n_days_aligned": int(len(strat_daily)),
        "n_days_alert": int((strat_daily != 0).sum()),
        "information_ratio": _maybe_float(ir),
        "tracking_error_ann_pct": _maybe_float(te * np.sqrt(252) * 100),
        "beta_vs_btc": _maybe_float(beta),
        "correlation_btc": _maybe_float(corr),
    }


def compute_top_drawdowns(closed: pd.DataFrame, max_col: str, close_col: str,
                            top_n: int = 5) -> list[dict]:
    """누적 PnL 시계열의 top-N drawdown (peak/valley/recovery/depth/length)."""
    daily = _daily_pnl_from_closed(closed, max_col, close_col)
    if len(daily) == 0:
        return []
    cum = daily.cumsum()
    peak = cum.cummax()
    underwater = cum - peak  # ≤ 0
    drawdowns = []
    in_dd = False
    peak_date = peak_val = valley_date = valley_val = None
    for date, c, p in zip(cum.index, cum, peak):
        if c < p and not in_dd:
            in_dd = True
            peak_date = date - pd.Timedelta(days=1) if date > cum.index[0] else date
            peak_val = float(p)
            valley_date = date
            valley_val = float(c)
        elif in_dd:
            if c < valley_val:
                valley_date = date
                valley_val = float(c)
            if c >= p:
                drawdowns.append({
                    "peak_date": str(peak_date.date()) if peak_date else None,
                    "valley_date": str(valley_date.date()),
                    "recovery_date": str(date.date()),
                    "depth_pct": _maybe_float((valley_val - peak_val) * 100),
                    "length_days": int((date - peak_date).days) if peak_date else None,
                })
                in_dd = False
                peak_date = peak_val = valley_date = valley_val = None
    if in_dd:
        drawdowns.append({
            "peak_date": str(peak_date.date()) if peak_date else None,
            "valley_date": str(valley_date.date()),
            "recovery_date": None,
            "depth_pct": _maybe_float((valley_val - peak_val) * 100),
            "length_days": int((cum.index[-1] - peak_date).days) if peak_date else None,
        })
    drawdowns.sort(key=lambda x: x["depth_pct"] or 0)
    return drawdowns[:top_n]


def compute_underwater_series(closed: pd.DataFrame, max_col: str, close_col: str) -> list[dict]:
    """일별 underwater % (cum / peak - 1)."""
    daily = _daily_pnl_from_closed(closed, max_col, close_col)
    if len(daily) == 0:
        return []
    cum = daily.cumsum()
    peak = cum.cummax()
    return [
        {"date": str(d.date()), "underwater_pct": float(c - p) * 100}
        for d, c, p in zip(cum.index, cum, peak)
    ]


def compute_rolling_sharpe(closed: pd.DataFrame, max_col: str, close_col: str,
                            window: int = 30) -> list[dict]:
    """일별 rolling annualized Sharpe."""
    daily = _daily_pnl_from_closed(closed, max_col, close_col)
    if len(daily) < 5:
        return []
    rolling_mean = daily.rolling(window, min_periods=5).mean()
    rolling_std = daily.rolling(window, min_periods=5).std(ddof=1)
    rs = (rolling_mean / rolling_std) * np.sqrt(252)
    return [
        {"date": str(d.date()), "rolling_sharpe": _maybe_float(float(v))}
        for d, v in zip(rs.index, rs.values) if pd.notna(v)
    ]


def compute_monthly_returns(closed: pd.DataFrame, max_col: str, close_col: str) -> list[dict]:
    """year-month 누적 return — heatmap 용."""
    daily = _daily_pnl_from_closed(closed, max_col, close_col)
    if len(daily) == 0:
        return []
    monthly = daily.resample("ME").sum() * 100  # %
    return [
        {
            "year": int(d.year),
            "month": int(d.month),
            "return_pct": _maybe_float(float(v)),
        }
        for d, v in zip(monthly.index, monthly.values)
    ]


def compute_best_worst_trades(closed: pd.DataFrame, max_col: str, close_col: str,
                                top_n: int = 3) -> dict:
    """가상 PnL 기준 top/bottom N 알림."""
    if len(closed) == 0:
        return {"best": [], "worst": []}
    sub = closed.copy()
    sub["pnl"] = sub.apply(
        lambda r: _virtual_pnl_per_alert(r[max_col], r[close_col]), axis=1
    )
    sub = sub.dropna(subset=["pnl"])
    if len(sub) == 0:
        return {"best": [], "worst": []}

    def to_card(row, kind):
        return {
            "date": str(row.get("date")),
            "coin": str(row.get("coin", "")).replace("KRW-", ""),
            "setup": str(row.get("setup_ids", "") or ""),
            "regime": str(row.get("btc_regime", "")),
            "entry_price": _maybe_float(row.get("next_open") if "next_open" in row else row.get("entry_price_proxy")),
            "max_pct": _maybe_float(row.get(max_col)),
            "min_pct": _maybe_float(row.get("next_min_return_pct" if "next_min_return_pct" in row else "first_1h_min_return_pct")),
            "close_pct": _maybe_float(row.get(close_col)),
            "pnl_pct": _maybe_float(float(row["pnl"]) * 100),
            "kind": kind,
        }

    best = sub.nlargest(top_n, "pnl")
    worst = sub.nsmallest(top_n, "pnl")
    return {
        "best": [to_card(r, "best") for _, r in best.iterrows()],
        "worst": [to_card(r, "worst") for _, r in worst.iterrows()],
    }


def compute_return_distribution(closed: pd.DataFrame, max_col: str, close_col: str,
                                  bin_pct: float = 1.0) -> list[dict]:
    """per-trade return histogram bins (default 1%pt 너비)."""
    pnl = _per_alert_pnl_series(closed, max_col, close_col) * 100  # %
    if len(pnl) == 0:
        return []
    lo = np.floor(pnl.min() / bin_pct) * bin_pct
    hi = np.ceil(pnl.max() / bin_pct) * bin_pct
    edges = np.arange(lo, hi + bin_pct, bin_pct)
    if len(edges) < 2:
        return []
    counts, _ = np.histogram(pnl, bins=edges)
    return [
        {
            "bin_low": _maybe_float(float(edges[i])),
            "bin_high": _maybe_float(float(edges[i + 1])),
            "count": int(counts[i]),
        }
        for i in range(len(counts))
    ]


# ============================================================================
# A1. Calibration curve — predicted P vs actual hit rate by score decile
# ============================================================================
def compute_calibration_curve(closed: pd.DataFrame, prob_col: str, hit_col: str,
                                n_bins: int = 10) -> list[dict]:
    """score bin 별 expected (mean predicted P) vs actual (hit rate).
    diagonal 에 가까울수록 모델이 솔직 (calibrated).
    """
    if prob_col not in closed.columns or hit_col not in closed.columns:
        return []
    sub = closed[[prob_col, hit_col]].dropna()
    if len(sub) < n_bins:
        return []
    try:
        sub = sub.copy()
        sub["_bin"] = pd.qcut(sub[prob_col], q=n_bins,
                                labels=False, duplicates="drop")
    except Exception:
        return []
    out = []
    for b, grp in sub.groupby("_bin"):
        if len(grp) == 0:
            continue
        out.append({
            "bin": int(b),
            "n": int(len(grp)),
            "predicted_mean": _maybe_float(float(grp[prob_col].mean())),
            "actual_hit_rate": _maybe_float(float(grp[hit_col].mean())),
        })
    return out


# ============================================================================
# A2. Decay curve — alert 후 시간별 평균 return
# ============================================================================
def compute_decay_curve_dist(closed: pd.DataFrame) -> list[dict]:
    """distribution 의 decay — paper_ledger 의 next_open/high/low/close 활용.
    t=0 (entry) → t=24h_low → t=24h_close → t=24h_high (ordering for shape).
    실제 시간 정확치 X, 24h horizon 의 분포 모양만.
    """
    if len(closed) == 0:
        return []
    sub = closed.dropna(subset=["next_open"]).copy()
    if len(sub) == 0 or "next_high" not in sub.columns:
        return []
    sub = sub[sub["next_open"] > 0]
    if len(sub) == 0:
        return []
    return [
        {"t_label": "entry",      "avg_return_pct": 0.0, "n": int(len(sub))},
        {"t_label": "24h close",  "avg_return_pct": _maybe_float(((sub["next_close"] / sub["next_open"] - 1).mean() * 100)), "n": int(len(sub))},
        {"t_label": "24h max",    "avg_return_pct": _maybe_float(((sub["next_high"]  / sub["next_open"] - 1).mean() * 100)), "n": int(len(sub))},
        {"t_label": "24h min",    "avg_return_pct": _maybe_float(((sub["next_low"]   / sub["next_open"] - 1).mean() * 100)), "n": int(len(sub))},
    ]


def compute_decay_curve_pre(closed: pd.DataFrame) -> list[dict]:
    """preopen 은 first 15m / 30m / 1h 의 high/low/close 데이터 풍부."""
    if len(closed) == 0 or "first_open" not in closed.columns:
        return []
    sub = closed.dropna(subset=["first_open"]).copy()
    sub = sub[sub["first_open"] > 0]
    if len(sub) == 0:
        return []
    op = sub["first_open"]

    def avg(col):
        if col not in sub.columns:
            return None
        return _maybe_float(((sub[col] / op - 1).mean() * 100))

    return [
        {"t_label": "entry",       "avg_return_pct": 0.0,                "n": int(len(sub))},
        {"t_label": "15m max",     "avg_return_pct": avg("first_15m_high"), "n": int(len(sub))},
        {"t_label": "15m low",     "avg_return_pct": avg("first_15m_low"),  "n": int(len(sub))},
        {"t_label": "30m max",     "avg_return_pct": avg("first_30m_high"), "n": int(len(sub))},
        {"t_label": "30m low",     "avg_return_pct": avg("first_30m_low"),  "n": int(len(sub))},
        {"t_label": "1h max",      "avg_return_pct": avg("first_1h_high"),  "n": int(len(sub))},
        {"t_label": "1h low",      "avg_return_pct": avg("first_1h_low"),   "n": int(len(sub))},
        {"t_label": "1h close",    "avg_return_pct": avg("first_1h_close"), "n": int(len(sub))},
    ]


# ============================================================================
# A3. TP rule sweep — TP 룰 다양화별 누적
# ============================================================================
def compute_tp_sweep(closed: pd.DataFrame, max_col: str, close_col: str,
                      tp_list: list[float] = (0.03, 0.05, 0.07, 0.10),
                      include_no_tp: bool = True,
                      cost: float | None = None) -> list[dict]:
    if cost is None:
        cost = ROUND_TRIP_COST_PCT
    if len(closed) == 0:
        return []
    max_d = pd.to_numeric(closed[max_col], errors="coerce") / 100.0
    close_d = pd.to_numeric(closed[close_col], errors="coerce") / 100.0
    n_valid = int(max_d.notna().sum())
    if n_valid == 0:
        return []
    out = []
    for tp in tp_list:
        gross = np.where(max_d >= tp, tp, close_d)
        pnl = pd.Series(gross - cost, index=closed.index).dropna()
        out.append({
            "rule": f"{int(tp*100)}% TP",
            "tp_pct": _maybe_float(tp * 100),
            "n_trades": int(len(pnl)),
            "cum_pnl_pct": _maybe_float(float(pnl.sum() * 100)),
            "tp_hit_rate_pct": _maybe_float(float((max_d.dropna() >= tp).mean() * 100)),
        })
    if include_no_tp:
        pnl = (close_d - cost).dropna()
        out.append({
            "rule": "no TP (EOD)",
            "tp_pct": None,
            "n_trades": int(len(pnl)),
            "cum_pnl_pct": _maybe_float(float(pnl.sum() * 100)),
            "tp_hit_rate_pct": None,
        })
    return out


# ============================================================================
# A4. Cost sensitivity sweep
# ============================================================================
def compute_cost_sweep(closed: pd.DataFrame, max_col: str, close_col: str,
                        cost_list: list[float] = (0.0015, 0.003, 0.005, 0.01)) -> list[dict]:
    if len(closed) == 0:
        return []
    max_d = pd.to_numeric(closed[max_col], errors="coerce") / 100.0
    close_d = pd.to_numeric(closed[close_col], errors="coerce") / 100.0
    out = []
    for cost in cost_list:
        gross = np.where(max_d >= TP_PCT, TP_PCT, close_d)
        pnl = pd.Series(gross - cost, index=closed.index).dropna()
        out.append({
            "cost_pct": _maybe_float(cost * 100),
            "n_trades": int(len(pnl)),
            "cum_pnl_pct": _maybe_float(float(pnl.sum() * 100)),
            "avg_pnl_pct": _maybe_float(float(pnl.mean() * 100)),
        })
    return out


# ============================================================================
# A5. PBO simple — K-fold chronological split robustness
# ============================================================================
def compute_pbo_simple(closed: pd.DataFrame, max_col: str, close_col: str,
                        n_splits: int = 5) -> dict:
    """Lopez de Prado 의 정식 PBO 는 K-comb in/out ranking 통계 (multi-strategy
    필요). prelude 는 단일 strategy 라 단순 변형: 시간 chronological 5-fold
    각 fold 의 cum PnL 부호 일관성 + 분산.
    """
    pnl = _per_alert_pnl_series(closed, max_col, close_col)
    n = len(pnl)
    if n < n_splits * 2:
        return {"n_trades": int(n), "note": "fold split skipped (n<10)"}
    # chronological order — closed["date"] 기준
    sub = closed.copy()
    sub["pnl"] = sub.apply(
        lambda r: _virtual_pnl_per_alert(r[max_col], r[close_col]), axis=1
    )
    sub = sub.dropna(subset=["pnl"]).sort_values("date").reset_index(drop=True)
    fold_size = max(1, len(sub) // n_splits)
    folds = []
    for i in range(n_splits):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_splits - 1 else len(sub)
        chunk = sub.iloc[start:end]
        if len(chunk) == 0:
            continue
        folds.append({
            "fold": int(i + 1),
            "n": int(len(chunk)),
            "date_range": [str(chunk["date"].iloc[0]), str(chunk["date"].iloc[-1])],
            "cum_pnl_pct": _maybe_float(float(chunk["pnl"].sum() * 100)),
            "win_rate_pct": _maybe_float(float((chunk["pnl"] > 0).mean() * 100)),
        })
    pos_folds = sum(1 for f in folds if (f["cum_pnl_pct"] or 0) > 0)
    return {
        "n_trades": int(len(sub)),
        "n_folds": len(folds),
        "n_positive_folds": int(pos_folds),
        "consistency_pct": _maybe_float(pos_folds / len(folds) * 100) if folds else None,
        "folds": folds,
        "note": "chronological K-fold cum PnL — 정식 PBO 는 multi-strategy 필요, 여기는 단순 robustness check",
    }


# ============================================================================
# A6. Factor regression — strat_daily = α + β · btc_daily
# ============================================================================
def compute_factor_regression(closed: pd.DataFrame, max_col: str, close_col: str,
                                btc_series: list[dict]) -> dict:
    if not btc_series or len(btc_series) < 10:
        return {}
    daily = _daily_pnl_from_closed(closed, max_col, close_col)
    if len(daily) == 0:
        return {}
    btc_df = pd.DataFrame(btc_series)
    btc_df["date_dt"] = pd.to_datetime(btc_df["date"])
    btc_df = btc_df.set_index("date_dt").sort_index()
    btc_daily = (btc_df["btc_cum_pct"] / 100.0).diff().dropna()
    if len(btc_daily) < 10:
        return {}
    strat_daily = daily.reindex(btc_daily.index, fill_value=0.0)

    # OLS y = a + b*x
    x = btc_daily.values
    y = strat_daily.values
    n = len(x)
    if n < 10:
        return {}
    x_mean, y_mean = x.mean(), y.mean()
    x_var = ((x - x_mean) ** 2).sum()
    if x_var == 0:
        return {}
    beta = ((x - x_mean) * (y - y_mean)).sum() / x_var
    alpha = y_mean - beta * x_mean
    y_pred = alpha + beta * x
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y_mean) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    se_alpha = float(np.sqrt(ss_res / (n - 2) * (1 / n + x_mean ** 2 / x_var))) if n > 2 else float("nan")
    t_alpha = float(alpha / se_alpha) if se_alpha and se_alpha > 0 else float("nan")
    return {
        "n_days": int(n),
        "alpha_daily": _maybe_float(alpha),
        "alpha_ann_pct": _maybe_float(alpha * 252 * 100),
        "alpha_se": _maybe_float(se_alpha),
        "alpha_t_stat": _maybe_float(t_alpha),
        "alpha_significant": (abs(t_alpha) > 1.96) if not np.isnan(t_alpha) else None,
        "beta": _maybe_float(beta),
        "r_squared": _maybe_float(r2),
    }


# ============================================================================
# B7. Per-coin mini tear sheet
# ============================================================================
def compute_per_coin(closed: pd.DataFrame, max_col: str, close_col: str,
                       top_n: int = 10) -> list[dict]:
    if len(closed) == 0 or "coin" not in closed.columns:
        return []
    rows = []
    for coin, grp in closed.groupby("coin"):
        n = len(grp)
        max_d = pd.to_numeric(grp[max_col], errors="coerce") / 100.0
        close_d = pd.to_numeric(grp[close_col], errors="coerce") / 100.0
        gross = np.where(max_d >= TP_PCT, TP_PCT, close_d)
        pnl = pd.Series(gross - ROUND_TRIP_COST_PCT, index=grp.index).dropna()
        if len(pnl) == 0:
            continue
        avg_max = float(grp[max_col].dropna().mean()) if max_col in grp else float("nan")
        rows.append({
            "coin": str(coin).replace("KRW-", ""),
            "n_alerts": int(n),
            "avg_max_pct": _maybe_float(avg_max),
            "avg_pnl_pct": _maybe_float(float(pnl.mean() * 100)),
            "cum_pnl_pct": _maybe_float(float(pnl.sum() * 100)),
            "win_rate_pct": _maybe_float(float((pnl > 0).mean() * 100)),
        })
    rows.sort(key=lambda r: -r["n_alerts"])
    return rows[:top_n]


# ============================================================================
# B8. NOTES.md vs system — placeholder (NOTES 비어있으면 unavailable)
# ============================================================================
def parse_notes_md(notes_path: str = "NOTES.md") -> list[dict]:
    """NOTES.md 의 '## YYYY-MM-DD' 헤더 + '### 내가 진입' 블록 파싱.
    템플릿 따른 entry 만 추출. 비어있거나 파일 없으면 빈 list.
    """
    p = Path(notes_path)
    if not p.exists():
        return []
    text = p.read_text(encoding="utf-8")
    entries = []
    import re
    blocks = re.split(r"\n## (\d{4}-\d{2}-\d{2})", text)
    if len(blocks) < 3:
        return []
    for i in range(1, len(blocks), 2):
        date = blocks[i]
        body = blocks[i + 1] if i + 1 < len(blocks) else ""
        # '### 내가 진입' 다음 bullet 들
        m = re.search(r"### 내가 진입\s*\n((?:- .+\n?)+)", body)
        coins = []
        if m:
            for line in m.group(1).splitlines():
                cm = re.match(r"-\s*(\S+)\s*[:：]\s*(.+)", line)
                if cm:
                    coins.append({"coin": cm.group(1), "raw": cm.group(2).strip()})
        m2 = re.search(r"### 청산 결과.*?\n((?:- .+\n?)+)", body)
        results = []
        if m2:
            for line in m2.group(1).splitlines():
                rm = re.match(r"-\s*(\S+)\s*[:：]\s*([+-]?\d+\.?\d*)", line)
                if rm:
                    results.append({"coin": rm.group(1), "return_pct": float(rm.group(2))})
        if coins or results:
            entries.append({"date": date, "entries": coins, "results": results})
    return entries


def compute_notes_vs_system(notes_entries: list[dict], df_dist: pd.DataFrame,
                              df_pre: pd.DataFrame) -> dict:
    """user 매매 vs system 매매 비교 — NOTES 비어있으면 placeholder."""
    if not notes_entries:
        return {
            "available": False,
            "note": "NOTES.md 가 비어있거나 템플릿 미작성. 사용자가 NOTES 채우면 자동 파싱 + 비교 활성.",
        }
    # NOTES 의 results 의 cumulative
    user_pnl = []
    for e in notes_entries:
        for r in e.get("results", []):
            user_pnl.append({"date": e["date"], "coin": r["coin"], "return_pct": r["return_pct"]})
    return {
        "available": True,
        "n_entries": int(len(notes_entries)),
        "n_results": int(len(user_pnl)),
        "user_avg_pct": _maybe_float(float(np.mean([r["return_pct"] for r in user_pnl]))) if user_pnl else None,
        "user_cum_pct": _maybe_float(float(np.sum([r["return_pct"] for r in user_pnl]))) if user_pnl else None,
        "rows": user_pnl[-20:],
    }


# ============================================================================
# B9. Time-of-day (preopen) — 첫 N분 hit 분포
# ============================================================================
def compute_time_of_day_pre(closed_pre: pd.DataFrame) -> dict:
    """preopen closed 의 first 15m / 30m / 1h hit 분포 — 첫 몇 분에 hit 가 집중."""
    if len(closed_pre) == 0:
        return {}
    out = {}
    for col in ["hit_first15_3pct", "hit_first15_5pct",
                "hit_first30_3pct", "hit_first30_5pct",
                "hit_first1h_3pct", "hit_first1h_5pct"]:
        if col in closed_pre.columns:
            v = closed_pre[col].dropna()
            out[col] = _maybe_float(float(v.mean() * 100)) if len(v) else None
    return out


# ============================================================================
# B10. Coin × channel matrix
# ============================================================================
def compute_coin_universe_matrix(df_dist: pd.DataFrame, df_pre: pd.DataFrame,
                                    top_n: int = 15) -> list[dict]:
    if len(df_dist) == 0 and len(df_pre) == 0:
        return []
    counts = {}
    if "coin" in df_dist.columns:
        for c, n in df_dist["coin"].value_counts().items():
            counts.setdefault(str(c), {"dist": 0, "pre": 0})["dist"] = int(n)
    if "coin" in df_pre.columns:
        for c, n in df_pre["coin"].value_counts().items():
            counts.setdefault(str(c), {"dist": 0, "pre": 0})["pre"] = int(n)
    rows = [
        {"coin": k.replace("KRW-", ""), "dist": v["dist"], "pre": v["pre"], "total": v["dist"] + v["pre"]}
        for k, v in counts.items()
    ]
    rows.sort(key=lambda r: -r["total"])
    return rows[:top_n]


def compute_breakdown_by(closed: pd.DataFrame, group_col: str,
                          max_col: str, min_col: str, close_col: str,
                          hit_cols: list[tuple[str, str]] | None = None) -> list[dict]:
    """그룹 별 (n, avg_max%, avg_min%, avg_pnl%, hit_rate%).

    hit_cols: [(label, col), ...] — 표시할 hit 컬럼.
    """
    if len(closed) == 0 or group_col not in closed.columns:
        return []
    out = []
    for key, sub in closed.groupby(group_col):
        if pd.isna(key) or str(key).lower() in ("nan", "none", "unknown", ""):
            continue
        n = len(sub)
        if n == 0:
            continue
        avg_max = sub[max_col].dropna().mean()
        avg_min = sub[min_col].dropna().mean() if min_col in sub.columns else np.nan
        pnl = sub.apply(
            lambda r: _virtual_pnl_per_alert(r[max_col], r[close_col]), axis=1
        ).dropna()
        avg_pnl = pnl.mean() * 100 if len(pnl) else np.nan
        row = {
            "group": str(key),
            "n": int(n),
            "avg_max_pct": _maybe_float(avg_max),
            "avg_min_pct": _maybe_float(avg_min),
            "avg_pnl_pct": _maybe_float(avg_pnl),
        }
        if hit_cols:
            for label, col in hit_cols:
                if col in sub.columns:
                    vals = sub[col].dropna()
                    row[f"hit_{label}_pct"] = _maybe_float(vals.mean() * 100) if len(vals) else None
        out.append(row)
    out.sort(key=lambda r: -r["n"])
    return out


def compute_setup_breakdown(closed: pd.DataFrame, max_col: str, min_col: str,
                             close_col: str) -> list[dict]:
    """setup_ids 가 'S01+S02' 같은 문자열 → 개별 setup 으로 unfold."""
    if len(closed) == 0 or "setup_ids" not in closed.columns:
        return []
    expanded = []
    for _, r in closed.iterrows():
        ids = str(r.get("setup_ids", "") or "")
        for sid in ids.split("+"):
            sid = sid.strip()
            if sid:
                expanded.append((sid, r))
    if not expanded:
        return []
    setups_seen = sorted(set(x[0] for x in expanded))
    out = []
    for sid in setups_seen:
        rows = [r for s, r in expanded if s == sid]
        sub = pd.DataFrame(rows)
        n = len(sub)
        if n == 0:
            continue
        avg_max = sub[max_col].dropna().mean()
        avg_min = sub[min_col].dropna().mean() if min_col in sub.columns else np.nan
        pnl = sub.apply(
            lambda r: _virtual_pnl_per_alert(r[max_col], r[close_col]), axis=1
        ).dropna()
        avg_pnl = pnl.mean() * 100 if len(pnl) else np.nan
        out.append({
            "group": sid,
            "n": int(n),
            "avg_max_pct": _maybe_float(avg_max),
            "avg_min_pct": _maybe_float(avg_min),
            "avg_pnl_pct": _maybe_float(avg_pnl),
            "hit_h2_pct": _maybe_float(sub.get("hit_h2", pd.Series(dtype=float)).dropna().mean() * 100) if "hit_h2" in sub.columns else None,
            "hit_h6_pct": _maybe_float(sub.get("hit_h6", pd.Series(dtype=float)).dropna().mean() * 100) if "hit_h6" in sub.columns else None,
            "hit_h5_pct": _maybe_float(sub.get("hit_h5", pd.Series(dtype=float)).dropna().mean() * 100) if "hit_h5" in sub.columns else None,
        })
    out.sort(key=lambda r: -r["n"])
    return out


def compute_score_breakdown(closed: pd.DataFrame, score_col: str,
                             max_col: str, min_col: str, close_col: str,
                             n_buckets: int = 4) -> list[dict]:
    """composite_score 분위별 break-down (quartile 기본)."""
    if len(closed) == 0 or score_col not in closed.columns:
        return []
    sub = closed.dropna(subset=[score_col]).copy()
    if len(sub) < n_buckets:
        return []
    try:
        sub["_bucket"] = pd.qcut(
            sub[score_col], q=n_buckets, labels=[f"Q{i+1}" for i in range(n_buckets)],
            duplicates="drop",
        )
    except Exception:
        return []
    out = []
    for bucket, grp in sub.groupby("_bucket", observed=True):
        n = len(grp)
        if n == 0:
            continue
        avg_max = grp[max_col].dropna().mean()
        avg_min = grp[min_col].dropna().mean() if min_col in grp.columns else np.nan
        pnl = grp.apply(
            lambda r: _virtual_pnl_per_alert(r[max_col], r[close_col]), axis=1
        ).dropna()
        avg_pnl = pnl.mean() * 100 if len(pnl) else np.nan
        score_lo = float(grp[score_col].min())
        score_hi = float(grp[score_col].max())
        out.append({
            "group": str(bucket),
            "n": int(n),
            "score_range": [_maybe_float(score_lo), _maybe_float(score_hi)],
            "avg_max_pct": _maybe_float(avg_max),
            "avg_min_pct": _maybe_float(avg_min),
            "avg_pnl_pct": _maybe_float(avg_pnl),
        })
    return out


def compute_btc_benchmark(upbit_d1_db: str, start_date: str, end_date: str) -> list[dict]:
    """같은 기간 KRW-BTC HODL 누적 return — alpha vs beta 비교용.

    start/end 사이 일별 close 기준 close[t]/close[start] - 1 을 % 로.
    """
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from data.database import load_candles
    except Exception:
        return []
    try:
        df = load_candles(upbit_d1_db, "KRW-BTC")
    except Exception:
        return []
    if df is None or len(df) == 0:
        return []
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date.astype(str)
    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()
    sub = df[(df["timestamp"].dt.date >= start) & (df["timestamp"].dt.date <= end)].sort_values("timestamp")
    if len(sub) == 0:
        return []
    base = float(sub["close"].iloc[0])
    if base <= 0:
        return []
    return [
        {"date": str(row["date"]), "btc_cum_pct": float(row["close"] / base - 1) * 100}
        for _, row in sub.iterrows()
    ]


def compute_distribution_summary(df: pd.DataFrame, btc_series: list[dict] | None = None) -> dict:
    """distribution paper_ledger → KPI dict."""
    closed = df[df["status"].astype(str) == "closed"].copy()
    out = {
        "n_alerts_total": int(len(df)),
        "n_closed": int(len(closed)),
        "n_pending": int(len(df) - len(closed)),
    }
    if len(df) > 0:
        out["first_alert_date"] = str(pd.to_datetime(df["date"]).min().date())
        out["last_alert_date"] = str(pd.to_datetime(df["date"]).max().date())
    if len(closed) == 0:
        return out

    # Hit rate (head 정의 그대로)
    for k in ("hit_h2", "hit_h6", "hit_h5"):
        if k in closed.columns:
            out[f"{k}_pct"] = float(closed[k].dropna().mean() * 100)

    out["avg_max_return_pct"] = float(closed["next_max_return_pct"].dropna().mean())
    out["avg_min_return_pct"] = float(closed["next_min_return_pct"].dropna().mean())
    out["avg_close_return_pct"] = float(closed["next_close_return_pct"].dropna().mean())
    out["median_max_return_pct"] = float(closed["next_max_return_pct"].dropna().median())
    out["win_rate_pct"] = float(
        (closed["next_close_return_pct"].dropna() > 0).mean() * 100
    )

    # 가상 누적 PnL (TP5 룰)
    pnl = closed.apply(
        lambda r: _virtual_pnl_per_alert(
            r["next_max_return_pct"], r["next_close_return_pct"]
        ),
        axis=1,
    ).dropna()
    if len(pnl) > 0:
        cum = pnl.cumsum()
        out["virtual"] = {
            "rule": "5% TP / EOD close, equal weight, cost 0.15% 차감",
            "n_trades": int(len(pnl)),
            "cum_pnl_pct": float(cum.iloc[-1] * 100),
            "max_dd_pct": float((cum - cum.cummax()).min() * 100),
            "avg_pnl_per_trade_pct": float(pnl.mean() * 100),
            "tp_hit_rate_pct": float(
                (closed["next_max_return_pct"].dropna() / 100.0 >= TP_PCT).mean() * 100
            ),
        }
        out["quant"] = compute_quant_metrics(
            closed, "next_max_return_pct", "next_close_return_pct"
        )
        out["bootstrap"] = compute_bootstrap_ci(
            closed, "next_max_return_pct", "next_close_return_pct"
        )
        out["stratification"] = {
            "regime": compute_breakdown_by(
                closed, "btc_regime",
                "next_max_return_pct", "next_min_return_pct", "next_close_return_pct",
                hit_cols=[("h2", "hit_h2"), ("h6", "hit_h6"), ("h5", "hit_h5")],
            ),
            "setup": compute_setup_breakdown(
                closed, "next_max_return_pct", "next_min_return_pct", "next_close_return_pct",
            ),
            "score": compute_score_breakdown(
                closed, "composite_score",
                "next_max_return_pct", "next_min_return_pct", "next_close_return_pct",
            ),
        }
        # 신규 — 업계 표준 + Lopez de Prado
        out["stats"] = compute_distribution_stats(
            closed, "next_max_return_pct", "next_close_return_pct"
        )
        out["confidence"] = compute_psr_dsr(
            closed, "next_max_return_pct", "next_close_return_pct"
        )
        if btc_series:
            out["vs_benchmark"] = compute_benchmark_metrics(
                closed, "next_max_return_pct", "next_close_return_pct", btc_series
            )
        out["top_drawdowns"] = compute_top_drawdowns(
            closed, "next_max_return_pct", "next_close_return_pct"
        )
        out["best_worst"] = compute_best_worst_trades(
            closed, "next_max_return_pct", "next_close_return_pct"
        )
        out["return_distribution"] = compute_return_distribution(
            closed, "next_max_return_pct", "next_close_return_pct"
        )
        # 신규 — 풍성화 6 항목
        out["calibration"] = {
            "p_h2_vs_hit": compute_calibration_curve(closed, "p_h2_3pct_4h", "hit_h2"),
            "p_h6_vs_hit": compute_calibration_curve(closed, "p_h6_5pct_24h", "hit_h6"),
            "p_h5_vs_hit": compute_calibration_curve(closed, "p_h5_20pct_tail", "hit_h5"),
        }
        out["decay"] = compute_decay_curve_dist(closed)
        out["tp_sweep"] = compute_tp_sweep(
            closed, "next_max_return_pct", "next_close_return_pct"
        )
        out["cost_sweep"] = compute_cost_sweep(
            closed, "next_max_return_pct", "next_close_return_pct"
        )
        out["pbo"] = compute_pbo_simple(
            closed, "next_max_return_pct", "next_close_return_pct"
        )
        if btc_series:
            out["factor_regression"] = compute_factor_regression(
                closed, "next_max_return_pct", "next_close_return_pct", btc_series
            )
        out["per_coin"] = compute_per_coin(
            closed, "next_max_return_pct", "next_close_return_pct"
        )
    return out


def compute_preopen_summary(df: pd.DataFrame, btc_series: list[dict] | None = None) -> dict:
    """preopen paper_ledger → KPI dict (1h horizon)."""
    closed = df[df["status"].astype(str) == "closed"].copy()
    out = {
        "n_alerts_total": int(len(df)),
        "n_closed": int(len(closed)),
        "n_pending": int(len(df) - len(closed)),
    }
    if len(df) > 0:
        out["first_alert_date"] = str(pd.to_datetime(df["date"]).min().date())
        out["last_alert_date"] = str(pd.to_datetime(df["date"]).max().date())
    if len(closed) == 0:
        return out

    for k in (
        "hit_first15_3pct",
        "hit_first15_5pct",
        "hit_first1h_3pct",
        "hit_first1h_5pct",
    ):
        if k in closed.columns:
            out[f"{k}_pct"] = float(closed[k].dropna().mean() * 100)

    if "first_1h_max_return_pct" in closed.columns:
        out["avg_max_return_pct"] = float(closed["first_1h_max_return_pct"].dropna().mean())
        out["avg_min_return_pct"] = float(closed["first_1h_min_return_pct"].dropna().mean())
        out["avg_close_return_pct"] = float(
            closed["first_1h_close_return_pct"].dropna().mean()
        )
        # 가상 누적 PnL (1h 안 +5% 못 가면 1h close)
        pnl = closed.apply(
            lambda r: _virtual_pnl_per_alert(
                r["first_1h_max_return_pct"], r["first_1h_close_return_pct"]
            ),
            axis=1,
        ).dropna()
        if len(pnl) > 0:
            cum = pnl.cumsum()
            out["virtual"] = {
                "rule": "1h horizon: 5% TP / 1h close, equal weight, cost 0.15% 차감",
                "n_trades": int(len(pnl)),
                "cum_pnl_pct": float(cum.iloc[-1] * 100),
                "max_dd_pct": float((cum - cum.cummax()).min() * 100),
                "avg_pnl_per_trade_pct": float(pnl.mean() * 100),
                "tp_hit_rate_pct": float(
                    (closed["first_1h_max_return_pct"].dropna() / 100.0 >= TP_PCT).mean() * 100
                ),
            }
            out["quant"] = compute_quant_metrics(
                closed, "first_1h_max_return_pct", "first_1h_close_return_pct"
            )
            out["bootstrap"] = compute_bootstrap_ci(
                closed, "first_1h_max_return_pct", "first_1h_close_return_pct"
            )
            out["stratification"] = {
                "regime": compute_breakdown_by(
                    closed, "btc_regime",
                    "first_1h_max_return_pct", "first_1h_min_return_pct",
                    "first_1h_close_return_pct",
                    hit_cols=[
                        ("first15_3", "hit_first15_3pct"),
                        ("first15_5", "hit_first15_5pct"),
                        ("first1h_3", "hit_first1h_3pct"),
                        ("first1h_5", "hit_first1h_5pct"),
                    ],
                ),
                "score": compute_score_breakdown(
                    closed, "composite_score",
                    "first_1h_max_return_pct", "first_1h_min_return_pct",
                    "first_1h_close_return_pct",
                ),
            }
            out["stats"] = compute_distribution_stats(
                closed, "first_1h_max_return_pct", "first_1h_close_return_pct"
            )
            out["confidence"] = compute_psr_dsr(
                closed, "first_1h_max_return_pct", "first_1h_close_return_pct"
            )
            if btc_series:
                out["vs_benchmark"] = compute_benchmark_metrics(
                    closed, "first_1h_max_return_pct", "first_1h_close_return_pct", btc_series
                )
            out["top_drawdowns"] = compute_top_drawdowns(
                closed, "first_1h_max_return_pct", "first_1h_close_return_pct"
            )
            out["best_worst"] = compute_best_worst_trades(
                closed, "first_1h_max_return_pct", "first_1h_close_return_pct"
            )
            out["return_distribution"] = compute_return_distribution(
                closed, "first_1h_max_return_pct", "first_1h_close_return_pct"
            )
            out["calibration"] = {
                "p_first15_3_vs_hit": compute_calibration_curve(closed, "p_first15_3pct", "hit_first15_3pct"),
                "p_first15_5_vs_hit": compute_calibration_curve(closed, "p_first15_5pct", "hit_first15_5pct"),
                "p_first1h_3_vs_hit": compute_calibration_curve(closed, "p_first1h_3pct", "hit_first1h_3pct"),
                "p_first1h_5_vs_hit": compute_calibration_curve(closed, "p_first1h_5pct", "hit_first1h_5pct"),
            }
            out["decay"] = compute_decay_curve_pre(closed)
            out["tp_sweep"] = compute_tp_sweep(
                closed, "first_1h_max_return_pct", "first_1h_close_return_pct"
            )
            out["cost_sweep"] = compute_cost_sweep(
                closed, "first_1h_max_return_pct", "first_1h_close_return_pct"
            )
            out["pbo"] = compute_pbo_simple(
                closed, "first_1h_max_return_pct", "first_1h_close_return_pct"
            )
            if btc_series:
                out["factor_regression"] = compute_factor_regression(
                    closed, "first_1h_max_return_pct", "first_1h_close_return_pct", btc_series
                )
            out["per_coin"] = compute_per_coin(
                closed, "first_1h_max_return_pct", "first_1h_close_return_pct"
            )
            out["time_of_day"] = compute_time_of_day_pre(closed)
    return out


# ============================================================================
# History (combined rows)
# ============================================================================
def history_rows(df_dist: pd.DataFrame, df_preopen: pd.DataFrame) -> list[dict]:
    rows = []
    for _, r in df_dist.iterrows():
        max_pct = r.get("next_max_return_pct")
        close_pct = r.get("next_close_return_pct")
        rows.append({
            "date": str(r["date"]),
            "channel": "distribution",
            "coin": str(r["coin"]).replace("KRW-", ""),
            "setups": str(r.get("setup_ids", "") or ""),
            "btc_regime": str(r.get("btc_regime", "")),
            "entry_price": _safe_float(r.get("next_open")),
            "p_h2": _safe_float(r.get("p_h2_3pct_4h")),
            "p_h6": _safe_float(r.get("p_h6_5pct_24h")),
            "p_h5": _safe_float(r.get("p_h5_20pct_tail")),
            "composite": _safe_float(r.get("composite_score")),
            "next_max_pct": _safe_float(max_pct),
            "next_min_pct": _safe_float(r.get("next_min_return_pct")),
            "next_close_pct": _safe_float(close_pct),
            "hit_h2": _safe_int(r.get("hit_h2")),
            "hit_h6": _safe_int(r.get("hit_h6")),
            "hit_h5": _safe_int(r.get("hit_h5")),
            "virtual_pnl_pct": _maybe_pct(_virtual_pnl_per_alert(max_pct, close_pct)),
            "status": str(r.get("status", "")),
        })

    for _, r in df_preopen.iterrows():
        max_pct = r.get("first_1h_max_return_pct")
        close_pct = r.get("first_1h_close_return_pct")
        # preopen 알림 시점 가격 = entry_price_proxy (alert 시점 close).
        # 09:00 첫 봉 시가 (first_open) 와 약간 다를 수 있음.
        entry = r.get("entry_price_proxy")
        if pd.isna(entry):
            entry = r.get("first_open")
        rows.append({
            "date": str(r["date"]),
            "channel": "preopen",
            "coin": str(r["coin"]).replace("KRW-", ""),
            "setups": "",
            "btc_regime": str(r.get("btc_regime", "")),
            "entry_price": _safe_float(entry),
            "p_first15_3": _safe_float(r.get("p_first15_3pct")),
            "p_first15_5": _safe_float(r.get("p_first15_5pct")),
            "p_first1h_3": _safe_float(r.get("p_first1h_3pct")),
            "composite": _safe_float(r.get("composite_score")),
            "next_max_pct": _safe_float(max_pct),
            "next_min_pct": _safe_float(r.get("first_1h_min_return_pct")),
            "next_close_pct": _safe_float(close_pct),
            "hit_h2": _safe_int(r.get("hit_first1h_3pct")),
            "hit_h6": _safe_int(r.get("hit_first1h_5pct")),
            "hit_h5": None,
            "virtual_pnl_pct": _maybe_pct(_virtual_pnl_per_alert(max_pct, close_pct)),
            "status": str(r.get("status", "")),
        })

    rows.sort(key=lambda x: (x["date"], x["channel"], x["coin"]), reverse=True)
    return rows


def _safe_float(v):
    try:
        f = float(v)
        if not np.isfinite(f):
            return None
        return round(f, 4)
    except (TypeError, ValueError):
        return None


def _safe_int(v):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return int(v)
    except (TypeError, ValueError):
        return None


def _maybe_pct(v):
    if v is None or pd.isna(v):
        return None
    return round(v * 100, 3)


# ============================================================================
# Accuracy time series (rolling)
# ============================================================================
def rolling_accuracy(df: pd.DataFrame, hit_cols: list[str],
                      window_days: int = ROLLING_WINDOW_DAYS) -> list[dict]:
    """alert 시점 기준 rolling window 의 hit rate 시계열.

    각 date 마다 [date - window + 1 days, date] 의 alert 들을 모아 hit rate.
    """
    if len(df) == 0:
        return []
    closed = df[df["status"].astype(str) == "closed"].copy()
    if len(closed) == 0:
        return []
    closed["date"] = pd.to_datetime(closed["date"])
    dates = sorted(closed["date"].dt.normalize().unique())
    results = []
    for d in dates:
        start = d - pd.Timedelta(days=window_days - 1)
        window = closed[(closed["date"] >= start) & (closed["date"] <= d)]
        if len(window) == 0:
            continue
        row = {"date": str(d.date()), "n": int(len(window))}
        for c in hit_cols:
            if c in window.columns:
                vals = window[c].dropna()
                row[f"{c}_pct"] = float(vals.mean() * 100) if len(vals) else None
        # cum virtual pnl up to d
        pnl_col = "virtual_pnl_dec"
        if pnl_col not in window.columns:
            pass
        results.append(row)
    return results


def cumulative_pnl_series(df: pd.DataFrame,
                           max_col: str, close_col: str) -> list[dict]:
    """일자별 누적 가상 PnL 시계열.

    같은 날 알림 N 개면 그날 net = sum(per-alert net). cum = 일별 net 누적.
    """
    closed = df[df["status"].astype(str) == "closed"].copy()
    if len(closed) == 0:
        return []
    closed["date"] = pd.to_datetime(closed["date"])
    closed["pnl"] = closed.apply(
        lambda r: _virtual_pnl_per_alert(r[max_col], r[close_col]), axis=1
    )
    closed = closed.dropna(subset=["pnl"])
    if len(closed) == 0:
        return []
    daily = closed.groupby(closed["date"].dt.date)["pnl"].sum().sort_index()
    cum = daily.cumsum()
    return [
        {
            "date": str(d),
            "daily_pnl_pct": float(daily.loc[d] * 100),
            "cum_pnl_pct": float(cum.loc[d] * 100),
        }
        for d in cum.index
    ]


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-ledger", default="output/paper_ledger.csv")
    parser.add_argument("--paper-ledger-preopen", default="output/paper_ledger_preopen.csv")
    parser.add_argument("--shadow-ledger-distribution", default="output/shadow_ledger_distribution.csv")
    parser.add_argument("--shadow-ledger-preopen", default="output/shadow_ledger_preopen.csv")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--asof", type=str, help="기준 시점 (default=now)")
    parser.add_argument(
        "--pin",
        default=None,
        help="명시 암호화 PIN. 미지정 시 PRELUDE_DASHBOARD_PIN env "
             f"또는 default {DEFAULT_PIN}.",
    )
    parser.add_argument(
        "--no-encrypt",
        action="store_true",
        help="평문 출력 (테스트용 only). 라이브 publish 에는 절대 금지.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("dashboard")

    asof = pd.Timestamp(args.asof) if args.asof else pd.Timestamp.now()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # PIN 우선순위 — 빈 문자열을 절대 평문으로 해석하지 X.
    # 평문 원하면 명시적 --no-encrypt.
    if args.no_encrypt:
        pin = None
    else:
        pin = args.pin or os.environ.get("PRELUDE_DASHBOARD_PIN") or DEFAULT_PIN
        if not pin:  # extra guard
            pin = DEFAULT_PIN
    log.info(f"encryption: {'PIN ' + ('*' * len(pin)) if pin else 'OFF (plaintext)'}")

    df_dist = _load_or_empty(args.paper_ledger)
    df_pre = _load_or_empty(args.paper_ledger_preopen)
    log.info(f"loaded: distribution {len(df_dist)} rows, preopen {len(df_pre)} rows")

    # 0) BTC benchmark (summary + accuracy 둘 다 사용)
    all_dates = []
    for ledger in [df_dist, df_pre]:
        if "date" in ledger.columns and len(ledger) > 0:
            all_dates += pd.to_datetime(ledger["date"]).dt.strftime("%Y-%m-%d").tolist()
    btc_bench = []
    if all_dates:
        btc_bench = compute_btc_benchmark(
            "data/upbit_d1.db", min(all_dates), max(all_dates)
        )
    log.info(f"BTC benchmark: {len(btc_bench)} days")

    # 1) summary.json
    notes_entries = parse_notes_md("NOTES.md")
    notes_vs = compute_notes_vs_system(notes_entries, df_dist, df_pre)
    coin_matrix = compute_coin_universe_matrix(df_dist, df_pre)

    summary = {
        "asof": asof.isoformat(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "channels": {
            "distribution": compute_distribution_summary(df_dist, btc_series=btc_bench),
            "preopen": compute_preopen_summary(df_pre, btc_series=btc_bench),
        },
        "notes_vs_system": notes_vs,
        "coin_universe": coin_matrix,
    }
    _write_json(out_dir / "summary.json", summary, passphrase=pin)
    log.info(f"saved summary.json")

    # 2) history.json
    history = {
        "asof": asof.isoformat(),
        "rows": history_rows(df_dist, df_pre),
    }
    _write_json(out_dir / "history.json", history, passphrase=pin)
    log.info(f"saved history.json ({len(history['rows'])} rows)")

    # 3) accuracy.json — 시계열 (rolling + cum_pnl + underwater + rolling_sharpe + monthly)
    closed_dist = df_dist[df_dist["status"].astype(str) == "closed"].copy() if "status" in df_dist.columns else df_dist.iloc[0:0]
    closed_pre = df_pre[df_pre["status"].astype(str) == "closed"].copy() if "status" in df_pre.columns else df_pre.iloc[0:0]

    accuracy = {
        "asof": asof.isoformat(),
        "window_days": ROLLING_WINDOW_DAYS,
        "distribution": {
            "rolling": rolling_accuracy(
                df_dist, ["hit_h2", "hit_h6", "hit_h5"]
            ),
            "cum_pnl": cumulative_pnl_series(
                df_dist, "next_max_return_pct", "next_close_return_pct"
            ),
            "rolling_sharpe": compute_rolling_sharpe(
                closed_dist, "next_max_return_pct", "next_close_return_pct"
            ),
            "underwater": compute_underwater_series(
                closed_dist, "next_max_return_pct", "next_close_return_pct"
            ),
            "monthly_returns": compute_monthly_returns(
                closed_dist, "next_max_return_pct", "next_close_return_pct"
            ),
        },
        "preopen": {
            "rolling": rolling_accuracy(
                df_pre,
                ["hit_first15_3pct", "hit_first15_5pct",
                 "hit_first1h_3pct", "hit_first1h_5pct"],
            ),
            "cum_pnl": cumulative_pnl_series(
                df_pre, "first_1h_max_return_pct", "first_1h_close_return_pct"
            ),
            "rolling_sharpe": compute_rolling_sharpe(
                closed_pre, "first_1h_max_return_pct", "first_1h_close_return_pct"
            ),
            "underwater": compute_underwater_series(
                closed_pre, "first_1h_max_return_pct", "first_1h_close_return_pct"
            ),
            "monthly_returns": compute_monthly_returns(
                closed_pre, "first_1h_max_return_pct", "first_1h_close_return_pct"
            ),
        },
        "btc_benchmark": btc_bench,
    }
    _write_json(out_dir / "accuracy.json", accuracy, passphrase=pin)
    log.info(f"saved accuracy.json (btc benchmark: {len(btc_bench)} days)")

    # 4) idea_validation.json — ACTIVE/WATCH_ONLY/SILENCE attribution.
    idea_candidates = load_idea_candidate_ledger(args)
    _, idea_payload = build_idea_validation_report(idea_candidates)
    idea_payload["asof"] = asof.isoformat()
    _write_json(out_dir / "idea_validation.json", idea_payload, passphrase=pin)
    log.info(f"saved idea_validation.json ({idea_payload.get('n_candidates', 0)} candidates)")

    # quick stdout summary
    print()
    print("=== Dashboard summary ===")
    for ch, s in summary["channels"].items():
        v = s.get("virtual", {})
        print(
            f"  {ch:<13} alerts={s.get('n_alerts_total',0):>4} "
            f"closed={s.get('n_closed',0):>4} "
            f"cum_pnl={v.get('cum_pnl_pct', float('nan')):+.2f}% "
            f"avg_max={s.get('avg_max_return_pct', float('nan')):+.2f}% "
            f"avg_min={s.get('avg_min_return_pct', float('nan')):+.2f}%"
        )
    print(f"\nout_dir: {out_dir}")


def _load_or_empty(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "status" in df.columns:
        df["status"] = df["status"].fillna("").astype(str)
    return df


def _write_json(path: Path, payload: dict, passphrase: str | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if passphrase:
        payload = _encrypt_payload(payload, passphrase)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def _encrypt_payload(payload: dict, passphrase: str) -> dict:
    """PBKDF2-HMAC-SHA256 + AES-256-CBC + HMAC-SHA256(salt||iv||ct).

    Viewer 의 decryptPayload (papers index.html 동일) 와 호환. PIN brute-force
    완전 차단은 불가능 (4-digit + client-side) — 외부 일반인 차단 수준.
    """
    from cryptography.hazmat.primitives import hashes, hmac as crypto_hmac
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    plaintext = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
    salt = os.urandom(16)
    iv = os.urandom(16)
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=64,
        salt=salt,
        iterations=PBKDF2_ITERATIONS,
    )
    keymat = kdf.derive(passphrase.encode("utf-8"))
    aes_key, mac_key = keymat[:32], keymat[32:]

    pad_len = 16 - (len(plaintext) % 16)
    padded = plaintext + bytes([pad_len]) * pad_len
    cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv))
    ct = cipher.encryptor().update(padded) + cipher.encryptor().finalize()

    h = crypto_hmac.HMAC(mac_key, hashes.SHA256())
    h.update(salt + iv + ct)
    mac = h.finalize()

    def b64(b: bytes) -> str:
        return base64.b64encode(b).decode("ascii")

    return {
        "encrypted": True,
        "version": 1,
        "kdf": "PBKDF2-HMAC-SHA256",
        "cipher": "AES-256-CBC-HMAC-SHA256",
        "iterations": PBKDF2_ITERATIONS,
        "salt": b64(salt),
        "iv": b64(iv),
        "ct": b64(ct),
        "mac": b64(mac),
    }


if __name__ == "__main__":
    main()

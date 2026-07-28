"""Dashboard JSON builder — paper_ledger 두 개 → soccz.github.io/projects/prelude/dashboard/data/*.json

산출물 (3개):
  summary.json   — 채널별 KPI (총 alert, hit rate, 가상 누적 PnL)
  history.json   — 전체 알림 행 (예측 + 실제 OHLC + 결과). 날짜 내림차순.
  accuracy.json  — rolling 30일 hit rate 시계열

가상 PnL 룰 (텔레그램 가이드와 동일):
  - 같은 날 알림은 동일 비중, 알림 없는 날은 cash 0%
  - +5% 도달 시 익절, 아니면 EOD close (자동 손절 X)
  - 거래비용 ROUND_TRIP_COST_PCT 차감
  - cum_pnl_pct = 일별 동일비중 net return의 복리 누적

운영:
  매일 close cron 끝에 호출. JSON 만 갱신, html/JS 는 그대로 둔다.

사용:
    python scripts/build_dashboard.py
    python scripts/build_dashboard.py --out-dir <github.io path>
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import stat
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ledger.config import ROUND_TRIP_COST_PCT
from ledger.portfolio_metrics import (
    CRYPTO_PERIODS_PER_YEAR,
    daily_equal_weight,
    date_cluster_bootstrap,
    equity_curve,
    normalize_kst_date,
    summarize_daily,
)
from ops.artifact_provenance import (  # noqa: E402
    ArtifactSourceChangedError,
    ArtifactValidationError,
    atomic_write_bytes,
    file_identity,
    sha256_bytes,
    strict_json_object,
)
from ops.champion_selector import (  # noqa: E402
    ChampionStateError,
    load_champion_state_artifact,
)
from scripts.idea_validation_report import (
    IdeaArtifactError,
    build_input_manifest as build_idea_input_manifest,
    build_report as build_idea_validation_report,
    input_manifest_matches_current,
    load_idea_validation_artifact,
    load_candidate_ledger as load_idea_candidate_ledger,
    report_payload_digest,
    validate_idea_validation_payload,
)
from ops.policy_competition import (  # noqa: E402
    POLICY_DB,
    PolicyArtifactError,
    load_policy_artifact,
)
from scripts.policy_history import build_policy_evolution


# Manual builds stay self-contained. Production publish always supplies its
# private generated-data directory explicitly.
DEFAULT_OUT_DIR = "output/dashboard_preview"
ROLLING_WINDOW_DAYS = 30
TP_PCT = 0.05  # 사용자 가이드: "5% 오르면 즉시 매도"
PBKDF2_ITERATIONS = 250000
# 2026-07-28 사용자 명시 승인으로 12 → 4 완화 (짧은 PIN의 무차별 대입 취약을
# 인지하고 수용 — 개인용 대시보드). PBKDF2 반복은 그대로 유지한다.
MIN_DASHBOARD_PASSPHRASE_LENGTH = 4
DASHBOARD_GENERATION_ENV = "PRELUDE_DASHBOARD_GENERATION_ID"


def resolve_dashboard_passphrase(explicit: str | None = None) -> str:
    """Resolve a non-default dashboard secret or fail closed.

    The dashboard is a static encrypted artifact, so a short published default
    is equivalent to no meaningful access control.  Production builds must
    receive a secret explicitly through ``--pin`` or
    ``PRELUDE_DASHBOARD_PIN``.  Plaintext remains an explicit test-only mode.
    """
    value = explicit if explicit is not None else os.environ.get(
        "PRELUDE_DASHBOARD_PIN"
    )
    if value is None or not value:
        raise ValueError(
            "dashboard encryption requires --pin or PRELUDE_DASHBOARD_PIN"
        )
    if value != value.strip():
        raise ValueError(
            "dashboard passphrase must not have leading/trailing whitespace"
        )
    if len(value) < MIN_DASHBOARD_PASSPHRASE_LENGTH:
        raise ValueError(
            "dashboard passphrase must contain at least "
            f"{MIN_DASHBOARD_PASSPHRASE_LENGTH} characters"
        )
    return value


def resolve_dashboard_generation_id(explicit: str | None = None) -> str:
    """Return one canonical UUIDv4 shared by every asset in a publish."""
    value = explicit or os.environ.get(DASHBOARD_GENERATION_ENV)
    if value is None:
        return str(uuid.uuid4())
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ValueError(
            f"{DASHBOARD_GENERATION_ENV} must be a canonical UUIDv4"
        ) from exc
    if parsed.version != 4 or str(parsed) != value:
        raise ValueError(
            f"{DASHBOARD_GENERATION_ENV} must be a canonical UUIDv4"
        )
    return value


def _bind_idea_dashboard_generation(
    payload: dict[str, Any],
    generation_id: str,
) -> dict[str, Any]:
    bound = dict(payload)
    bound["dashboard_generation_id"] = generation_id
    bound["payload_sha256"] = report_payload_digest(bound)
    return bound


def _rows_through_asof(
    frame: pd.DataFrame,
    asof=None,
    *,
    date_col: str = "date",
) -> pd.DataFrame:
    """Return one canonical KST-date cohort at or before ``asof``.

    Dashboard ledgers store date-only KST values, but tests/backfills may carry
    timezone-aware instants.  Invalid or future-dated rows are excluded before
    *any* headline, radar, history, or performance calculation so all metrics
    share the same cohort.
    """
    if frame is None or not isinstance(frame, pd.DataFrame):
        return pd.DataFrame()
    out = frame.reset_index(drop=True).copy()
    if out.empty:
        return out
    if date_col not in out.columns:
        return out.iloc[0:0].copy()
    cutoff = normalize_kst_date(asof)

    def _parse(value):
        if value is None or (
            isinstance(value, float) and np.isnan(value)
        ) or str(value).strip() == "":
            return pd.NaT
        try:
            return normalize_kst_date(value)
        except (TypeError, ValueError):
            return pd.NaT

    dates = out[date_col].map(_parse)
    out = out.loc[dates.notna() & (dates <= cutoff)].copy()
    if len(out):
        out[date_col] = dates.loc[out.index].dt.strftime("%Y-%m-%d")
    return out.reset_index(drop=True)


# ============================================================================
# Per-channel metrics
# ============================================================================
def _virtual_pnl_per_alert(max_ret_pct: float, close_ret_pct: float) -> float:
    """텔레그램 가이드 룰: TP 5% 도달 시 5% 익절, 아니면 EOD close. 비용 차감.

    Returns: net return as decimal (e.g. 0.034 == +3.4%).
    """
    try:
        max_d = float(max_ret_pct) / 100.0
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(max_d):
        return np.nan
    if max_d >= TP_PCT:
        return TP_PCT - ROUND_TRIP_COST_PCT
    try:
        close_d = float(close_ret_pct) / 100.0
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(close_d):
        return np.nan
    return close_d - ROUND_TRIP_COST_PCT


def _net_pnl_series(
    max_values,
    close_values,
    *,
    tp: float = TP_PCT,
    cost: float = ROUND_TRIP_COST_PCT,
) -> pd.Series:
    """Return index-aligned net PnL without inventing missing path evidence."""
    max_d = pd.to_numeric(max_values, errors="coerce") / 100.0
    close_d = pd.to_numeric(close_values, errors="coerce") / 100.0
    gross = np.where(max_d >= tp, tp, close_d)
    pnl = pd.Series(gross - cost, index=max_d.index, dtype=float)
    invalid_path = ~np.isfinite(max_d)
    return pnl.mask(invalid_path | ~np.isfinite(pnl))


def _per_alert_pnl_series(closed: pd.DataFrame, max_col: str, close_col: str) -> pd.Series:
    """알림별 net PnL (TP rule + 비용) — vectorized.

    gross = max_d >= TP_PCT ? TP_PCT : close_d
    net = gross - ROUND_TRIP_COST_PCT
    NaN max → drop.
    """
    if len(closed) == 0 or max_col not in closed.columns:
        return pd.Series(dtype=float)
    return _net_pnl_series(
        closed[max_col],
        closed.get(close_col, pd.Series(np.nan, index=closed.index)),
    ).dropna()


def compute_quant_metrics(
    closed: pd.DataFrame,
    max_col: str,
    close_col: str,
    *,
    asof=None,
) -> dict:
    """헤드라인 metric — Sharpe/Sortino/Calmar/Profit Factor/Expectancy.

    일별 equal-weight aggregation 으로 Sharpe (평균 / std × √365).
    Calmar = compounded annualized return / |MDD|.
    Profit Factor = sum(positive) / |sum(negative)|.
    Expectancy = avg per trade.
    """
    closed = _rows_through_asof(closed, asof)
    pnl = _per_alert_pnl_series(closed, max_col, close_col)
    if len(pnl) == 0:
        return {"n_trades": 0}

    closed = closed.copy()
    closed["pnl"] = closed.apply(
        lambda r: _virtual_pnl_per_alert(r[max_col], r[close_col]), axis=1
    )
    closed = closed.dropna(subset=["pnl"])
    closed["date_dt"] = pd.to_datetime(closed["date"])

    daily = daily_equal_weight(
        closed["date_dt"],
        closed["pnl"],
        calendar_end=asof,
    )
    portfolio = summarize_daily(daily)
    n_days = len(daily)
    cumulative_return = portfolio["cumulative_return"]
    if n_days and 1.0 + cumulative_return > 0:
        annualized_return = (1.0 + cumulative_return) ** (
            CRYPTO_PERIODS_PER_YEAR / n_days
        ) - 1.0
    else:
        annualized_return = float("nan")
    mdd = portfolio["max_drawdown"]
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
        "n_signal_days": int(closed["date_dt"].dt.normalize().nunique()),
        "n_wins": int(len(wins)),
        "n_losses": int(len(losses)),
        "sharpe_ann": _maybe_float(portfolio["sharpe_ann"]),
        "sortino_ann": _maybe_float(portfolio["sortino_ann"]),
        "calmar": _maybe_float(calmar),
        "max_drawdown_pct": _maybe_float(mdd * 100),
        "cumulative_return_pct": _maybe_float(cumulative_return * 100),
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
    """Date-cluster bootstrap CI after within-day equal weighting."""
    pnl = _per_alert_pnl_series(closed, max_col, close_col).reset_index(drop=True)
    n = len(pnl)
    dates = pd.to_datetime(closed.loc[_per_alert_pnl_series(
        closed, max_col, close_col
    ).index, "date"])
    daily = daily_equal_weight(
        dates.reset_index(drop=True),
        pnl,
        include_no_trade_days=False,
    )
    boot = date_cluster_bootstrap(daily, n_iter=n_iter, seed=seed)
    if "mean_return_ci95" not in boot:
        return {
            "n_trades": int(n),
            "n_days": int(boot["n_days"]),
            "note": boot["note"],
        }
    return {
        "n_trades": int(n),
        "n_days": int(boot["n_days"]),
        "n_iter": int(n_iter),
        "method": "date_cluster_equal_weight",
        "cum_pnl_pct_ci95": [
            _maybe_float(v * 100) for v in boot["cumulative_return_ci95"]
        ],
        "mean_pnl_pct_ci95": [
            _maybe_float(v * 100) for v in boot["mean_return_ci95"]
        ],
    }


def _daily_pnl_from_closed(
    closed: pd.DataFrame,
    max_col: str,
    close_col: str,
    *,
    asof=None,
) -> pd.Series:
    """closed 행 → 일별 net PnL series (TP rule + cost, vectorized).
    같은 날 N알림이면 equal-weight mean, 무알림일은 cash=0.
    """
    closed = _rows_through_asof(closed, asof)
    if len(closed) == 0:
        return pd.Series(dtype=float)
    pnl = _per_alert_pnl_series(closed, max_col, close_col)
    if len(pnl) == 0:
        return pd.Series(dtype=float)
    dates = pd.to_datetime(closed.loc[pnl.index, "date"])
    return daily_equal_weight(dates, pnl, calendar_end=asof)


def _portfolio_cumulative_pct(dates: pd.Series, pnl: pd.Series) -> float | None:
    """Signal rows → same-day equal-weight, calendar-cash, compounded return."""
    if pnl.empty:
        return None
    daily = daily_equal_weight(dates.loc[pnl.index], pnl)
    return _maybe_float(summarize_daily(daily)["cumulative_return"] * 100)


def compute_distribution_stats(
    closed: pd.DataFrame,
    max_col: str,
    close_col: str,
    *,
    asof=None,
) -> dict:
    """업계 표준 — Vol(ann) / Skew / Kurt / VaR / CVaR / Tail Ratio /
    Recovery Factor / Ulcer Index / Common Sense Ratio / Win-Loss streak."""
    pnl = _per_alert_pnl_series(closed, max_col, close_col)
    if len(pnl) == 0:
        return {}
    daily = _daily_pnl_from_closed(
        closed,
        max_col,
        close_col,
        asof=asof,
    )

    out = {}
    if len(daily) > 1:
        std_d = float(daily.std(ddof=1))
        out["volatility_ann_pct"] = _maybe_float(
            std_d * np.sqrt(CRYPTO_PERIODS_PER_YEAR) * 100
        )
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

    # Recovery Factor + Ulcer over compounded daily equity (initial equity included).
    equity = equity_curve(daily)
    if len(equity) > 1:
        peak = equity.cummax()
        drawdown = equity / peak - 1.0
        mdd = float(drawdown.min())
        cumulative = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
        out["recovery_factor"] = _maybe_float(
            cumulative / abs(mdd) if mdd < 0 else float("nan")
        )
        out["ulcer_index"] = _maybe_float(
            float(np.sqrt(np.mean(drawdown ** 2))) * 100
        )

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
            cur_w += 1
            cur_l = 0
            max_win = max(max_win, cur_w)
        elif s < 0:
            cur_l += 1
            cur_w = 0
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


def _btc_daily_returns(btc_df: pd.DataFrame) -> pd.Series:
    """Normalize current and legacy benchmark payloads to decimal daily returns."""
    if "btc_daily_return_pct" in btc_df:
        return pd.to_numeric(
            btc_df["btc_daily_return_pct"], errors="coerce"
        ).div(100.0).dropna()
    if "btc_close" in btc_df:
        return pd.to_numeric(
            btc_df["btc_close"], errors="coerce"
        ).pct_change().dropna()
    # Legacy payload: cumulative percentage is wealth, so use pct_change rather
    # than a percentage-point difference.
    btc_wealth = 1.0 + pd.to_numeric(
        btc_df["btc_cum_pct"], errors="coerce"
    ).div(100.0)
    return btc_wealth.pct_change().dropna()


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
    btc_daily = _btc_daily_returns(btc_df)
    if len(btc_daily) < 5:
        return {}

    # Reindex strat daily to BTC's date index — alert 없는 날 = 0 PnL
    strat_daily = daily.reindex(btc_daily.index, fill_value=0.0)

    excess = strat_daily - btc_daily
    te = float(excess.std(ddof=1))
    ir = (
        float(excess.mean() / te * np.sqrt(CRYPTO_PERIODS_PER_YEAR))
        if te > 0 else float("nan")
    )

    var_b = float(btc_daily.var(ddof=1))
    cov_sb = float(strat_daily.cov(btc_daily))
    beta = (cov_sb / var_b) if var_b > 0 else float("nan")
    corr = float(strat_daily.corr(btc_daily))

    return {
        "n_days_aligned": int(len(strat_daily)),
        "n_days_alert": int((strat_daily != 0).sum()),
        "information_ratio": _maybe_float(ir),
        "tracking_error_ann_pct": _maybe_float(
            te * np.sqrt(CRYPTO_PERIODS_PER_YEAR) * 100
        ),
        "beta_vs_btc": _maybe_float(beta),
        "correlation_btc": _maybe_float(corr),
    }


def compute_top_drawdowns(closed: pd.DataFrame, max_col: str, close_col: str,
                            top_n: int = 5, *, asof=None) -> list[dict]:
    """누적 PnL 시계열의 top-N drawdown (peak/valley/recovery/depth/length)."""
    daily = _daily_pnl_from_closed(
        closed,
        max_col,
        close_col,
        asof=asof,
    )
    if len(daily) == 0:
        return []
    curve = equity_curve(daily)
    peak = curve.cummax()
    drawdowns: list[dict[str, Any]] = []
    in_dd = False
    peak_date: pd.Timestamp | None = None
    peak_val: float | None = None
    valley_date: pd.Timestamp | None = None
    valley_val: float | None = None
    for date, c, p in zip(curve.index[1:], curve.iloc[1:], peak.iloc[1:]):
        if c < p and not in_dd:
            in_dd = True
            peak_date = date - pd.Timedelta(days=1)
            peak_val = float(p)
            valley_date = date
            valley_val = float(c)
        elif (
            in_dd
            and peak_date is not None
            and peak_val is not None
            and valley_date is not None
            and valley_val is not None
        ):
            if c < valley_val:
                valley_date = date
                valley_val = float(c)
            if c >= p:
                drawdowns.append({
                    "peak_date": str(peak_date.date()),
                    "valley_date": str(valley_date.date()),
                    "recovery_date": str(date.date()),
                    "depth_pct": _maybe_float(
                        (valley_val / peak_val - 1.0) * 100
                    ),
                    "length_days": int((date - peak_date).days),
                })
                in_dd = False
                peak_date = peak_val = valley_date = valley_val = None
    if (
        in_dd
        and peak_date is not None
        and peak_val is not None
        and valley_date is not None
        and valley_val is not None
    ):
        drawdowns.append({
            "peak_date": str(peak_date.date()),
            "valley_date": str(valley_date.date()),
            "recovery_date": None,
            "depth_pct": _maybe_float((valley_val / peak_val - 1.0) * 100),
            "length_days": int((curve.index[-1] - peak_date).days),
        })
    drawdowns.sort(key=lambda x: x["depth_pct"] or 0)
    return drawdowns[:top_n]


def compute_underwater_series(
    closed: pd.DataFrame,
    max_col: str,
    close_col: str,
    *,
    asof=None,
) -> list[dict]:
    """일별 underwater % (cum / peak - 1)."""
    daily = _daily_pnl_from_closed(
        closed,
        max_col,
        close_col,
        asof=asof,
    )
    if len(daily) == 0:
        return []
    curve = equity_curve(daily)
    peak = curve.cummax()
    return [
        {"date": str(d.date()), "underwater_pct": float(c / p - 1.0) * 100}
        for d, c, p in zip(curve.index[1:], curve.iloc[1:], peak.iloc[1:])
    ]


def compute_rolling_sharpe(closed: pd.DataFrame, max_col: str, close_col: str,
                            window: int = 30, *, asof=None) -> list[dict]:
    """일별 rolling annualized Sharpe."""
    daily = _daily_pnl_from_closed(
        closed,
        max_col,
        close_col,
        asof=asof,
    )
    if len(daily) < 5:
        return []
    rolling_mean = daily.rolling(window, min_periods=5).mean()
    rolling_std = daily.rolling(window, min_periods=5).std(ddof=1)
    rs = (rolling_mean / rolling_std) * np.sqrt(CRYPTO_PERIODS_PER_YEAR)
    return [
        {"date": str(d.date()), "rolling_sharpe": _maybe_float(float(v))}
        for d, v in zip(rs.index, rs.values) if pd.notna(v)
    ]


def compute_monthly_returns(
    closed: pd.DataFrame,
    max_col: str,
    close_col: str,
    *,
    asof=None,
) -> list[dict]:
    """year-month 누적 return — heatmap 용."""
    daily = _daily_pnl_from_closed(
        closed,
        max_col,
        close_col,
        asof=asof,
    )
    if len(daily) == 0:
        return []
    monthly = daily.groupby(daily.index.to_period("M")).apply(
        lambda x: ((1.0 + x).prod() - 1.0) * 100
    )
    return [
        {
            "year": int(period.year),
            "month": int(period.month),
            "return_pct": _maybe_float(float(v)),
        }
        for period, v in zip(monthly.index, monthly.values)
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
                      tp_list: Sequence[float] = (0.03, 0.05, 0.07, 0.10),
                      include_no_tp: bool = True,
                      cost: float | None = None) -> list[dict]:
    if cost is None:
        cost = ROUND_TRIP_COST_PCT
    if len(closed) == 0:
        return []
    max_d = pd.to_numeric(closed[max_col], errors="coerce") / 100.0
    close_d = pd.to_numeric(closed[close_col], errors="coerce") / 100.0
    valid_max = max_d[np.isfinite(max_d)]
    out = []
    for tp in tp_list:
        pnl = _net_pnl_series(
            closed[max_col],
            closed[close_col],
            tp=tp,
            cost=cost,
        ).dropna()
        out.append({
            "rule": f"{int(tp*100)}% TP",
            "tp_pct": _maybe_float(tp * 100),
            "n_trades": int(len(pnl)),
            "cum_pnl_pct": _portfolio_cumulative_pct(closed["date"], pnl),
            "tp_hit_rate_pct": _maybe_float(
                float((valid_max >= tp).mean() * 100)
                if len(valid_max)
                else np.nan
            ),
        })
    if include_no_tp:
        pnl = (close_d - cost).mask(~np.isfinite(close_d)).dropna()
        out.append({
            "rule": "no TP (EOD)",
            "tp_pct": None,
            "n_trades": int(len(pnl)),
            "cum_pnl_pct": _portfolio_cumulative_pct(closed["date"], pnl),
            "tp_hit_rate_pct": None,
        })
    return out


# ============================================================================
# A4. Cost sensitivity sweep
# ============================================================================
def compute_cost_sweep(closed: pd.DataFrame, max_col: str, close_col: str,
                        cost_list: Sequence[float] = (0.0015, 0.003, 0.005, 0.01)) -> list[dict]:
    if len(closed) == 0:
        return []
    out = []
    for cost in cost_list:
        pnl = _net_pnl_series(
            closed[max_col],
            closed[close_col],
            cost=cost,
        ).dropna()
        out.append({
            "cost_pct": _maybe_float(cost * 100),
            "n_trades": int(len(pnl)),
            "cum_pnl_pct": _portfolio_cumulative_pct(closed["date"], pnl),
            "avg_pnl_pct": _maybe_float(float(pnl.mean() * 100)),
        })
    return out


# ============================================================================
# A5. PBO simple — K-fold chronological split robustness
# ============================================================================
def compute_pbo_simple(closed: pd.DataFrame, max_col: str, close_col: str,
                        n_splits: int = 5, *, asof=None) -> dict:
    """Lopez de Prado 의 정식 PBO 는 K-comb in/out ranking 통계 (multi-strategy
    필요). prelude 는 단일 strategy 라 단순 변형: 시간 chronological 5-fold
    각 fold 의 cum PnL 부호 일관성 + 분산.
    """
    pnl = _per_alert_pnl_series(closed, max_col, close_col)
    daily = daily_equal_weight(
        closed.loc[pnl.index, "date"],
        pnl,
        calendar_end=asof,
    )
    if len(daily) < n_splits * 2:
        return {
            "n_trades": int(len(pnl)),
            "n_days": int(len(daily)),
            "note": f"fold split skipped (n_days<{n_splits * 2})",
        }
    chunks = [
        daily.iloc[indexes]
        for indexes in np.array_split(np.arange(len(daily)), n_splits)
    ]
    folds = []
    for i, chunk in enumerate(chunks):
        if len(chunk) == 0:
            continue
        cumulative = summarize_daily(chunk)["cumulative_return"]
        folds.append({
            "fold": int(i + 1),
            "n": int(len(chunk)),
            "n_days": int(len(chunk)),
            "date_range": [
                str(pd.Timestamp(chunk.index[0]).date()),
                str(pd.Timestamp(chunk.index[-1]).date()),
            ],
            "cum_pnl_pct": _maybe_float(cumulative * 100),
            "win_rate_pct": _maybe_float(float((chunk > 0).mean() * 100)),
        })
    pos_folds = sum(1 for f in folds if (f["cum_pnl_pct"] or 0) > 0)
    return {
        "n_trades": int(len(pnl)),
        "n_days": int(len(daily)),
        "n_folds": len(folds),
        "n_positive_folds": int(pos_folds),
        "consistency_pct": _maybe_float(pos_folds / len(folds) * 100) if folds else None,
        "folds": folds,
        "note": (
            "chronological day-level K-fold compounded PnL — 정식 PBO 는 "
            "multi-strategy 필요, 여기는 단순 robustness check"
        ),
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
    btc_daily = _btc_daily_returns(btc_df)
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
        "alpha_ann_pct": _maybe_float(alpha * CRYPTO_PERIODS_PER_YEAR * 100),
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
        pnl = _net_pnl_series(grp[max_col], grp[close_col]).dropna()
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
    counts: dict[str, dict[str, int]] = {}
    if "coin" in df_dist.columns:
        for c, n in df_dist["coin"].value_counts().items():
            counts.setdefault(str(c), {"dist": 0, "pre": 0})["dist"] = int(n)
    if "coin" in df_pre.columns:
        for c, n in df_pre["coin"].value_counts().items():
            counts.setdefault(str(c), {"dist": 0, "pre": 0})["pre"] = int(n)
    rows: list[dict[str, Any]] = [
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
    previous_close = sub["close"].shift(1)
    daily_return = sub["close"].div(previous_close).sub(1.0)
    return [
        {
            "date": str(row["date"]),
            "btc_close": float(row["close"]),
            "btc_daily_return_pct": (
                None if pd.isna(daily_return.loc[idx])
                else float(daily_return.loc[idx]) * 100
            ),
            "btc_cum_pct": float(row["close"] / base - 1) * 100,
        }
        for idx, row in sub.iterrows()
    ]


def compute_distribution_summary(
    df: pd.DataFrame,
    btc_series: list[dict] | None = None,
    *,
    asof=None,
) -> dict:
    """distribution paper_ledger → KPI dict."""
    df = _rows_through_asof(df, asof)
    closed = df[df["status"].astype(str) == "closed"].copy()
    out: dict[str, Any] = {
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
        daily = _daily_pnl_from_closed(
            closed,
            "next_max_return_pct",
            "next_close_return_pct",
            asof=asof,
        )
        portfolio = summarize_daily(daily)
        out["virtual"] = {
            "rule": (
                "5% TP / EOD close, day equal-weight, compounded, "
                "cost 0.15% 차감"
            ),
            "n_trades": int(len(pnl)),
            "n_calendar_days": portfolio["n_calendar_days"],
            "cum_pnl_pct": float(portfolio["cumulative_return"] * 100),
            "max_dd_pct": float(portfolio["max_drawdown"] * 100),
            "avg_pnl_per_trade_pct": float(pnl.mean() * 100),
            "tp_hit_rate_pct": float(
                (closed["next_max_return_pct"].dropna() / 100.0 >= TP_PCT).mean() * 100
            ),
        }
        out["quant"] = compute_quant_metrics(
            closed,
            "next_max_return_pct",
            "next_close_return_pct",
            asof=asof,
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
            closed,
            "next_max_return_pct",
            "next_close_return_pct",
            asof=asof,
        )
        out["confidence"] = compute_psr_dsr(
            closed, "next_max_return_pct", "next_close_return_pct"
        )
        if btc_series:
            out["vs_benchmark"] = compute_benchmark_metrics(
                closed, "next_max_return_pct", "next_close_return_pct", btc_series
            )
        out["top_drawdowns"] = compute_top_drawdowns(
            closed,
            "next_max_return_pct",
            "next_close_return_pct",
            asof=asof,
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
            closed,
            "next_max_return_pct",
            "next_close_return_pct",
            asof=asof,
        )
        if btc_series:
            out["factor_regression"] = compute_factor_regression(
                closed, "next_max_return_pct", "next_close_return_pct", btc_series
            )
        out["per_coin"] = compute_per_coin(
            closed, "next_max_return_pct", "next_close_return_pct"
        )
    return out


def compute_preopen_summary(
    df: pd.DataFrame,
    btc_series: list[dict] | None = None,
    *,
    asof=None,
) -> dict:
    """preopen paper_ledger → KPI dict (1h horizon)."""
    df = _rows_through_asof(df, asof)
    closed = df[df["status"].astype(str) == "closed"].copy()
    out: dict[str, Any] = {
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
            daily = _daily_pnl_from_closed(
                closed,
                "first_1h_max_return_pct",
                "first_1h_close_return_pct",
                asof=asof,
            )
            portfolio = summarize_daily(daily)
            out["virtual"] = {
                "rule": (
                    "1h horizon: 5% TP / 1h close, day equal-weight, "
                    "compounded, cost 0.15% 차감"
                ),
                "n_trades": int(len(pnl)),
                "n_calendar_days": portfolio["n_calendar_days"],
                "cum_pnl_pct": float(portfolio["cumulative_return"] * 100),
                "max_dd_pct": float(portfolio["max_drawdown"] * 100),
                "avg_pnl_per_trade_pct": float(pnl.mean() * 100),
                "tp_hit_rate_pct": float(
                    (closed["first_1h_max_return_pct"].dropna() / 100.0 >= TP_PCT).mean() * 100
                ),
            }
            out["quant"] = compute_quant_metrics(
                closed,
                "first_1h_max_return_pct",
                "first_1h_close_return_pct",
                asof=asof,
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
                closed,
                "first_1h_max_return_pct",
                "first_1h_close_return_pct",
                asof=asof,
            )
            out["confidence"] = compute_psr_dsr(
                closed, "first_1h_max_return_pct", "first_1h_close_return_pct"
            )
            if btc_series:
                out["vs_benchmark"] = compute_benchmark_metrics(
                    closed, "first_1h_max_return_pct", "first_1h_close_return_pct", btc_series
                )
            out["top_drawdowns"] = compute_top_drawdowns(
                closed,
                "first_1h_max_return_pct",
                "first_1h_close_return_pct",
                asof=asof,
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
                closed,
                "first_1h_max_return_pct",
                "first_1h_close_return_pct",
                asof=asof,
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
# Recommend radar (SHADOW 스캐너) — 일일 출력 + net 실현 요약
# ============================================================================
def compute_recommend_summary(
    df_rec: pd.DataFrame,
    *,
    asof=None,
) -> dict:
    """SHADOW 추천 레이더 ledger(shadow_ledger_recommend.csv) → KPI dict.

    - latest_radar: 가장 최근 추천일의 top-3 (오늘 분포 — 대시보드 노출용).
    - closed 행 실현 요약: realized_pct 는 close_recommend_ledger.py 가 이미
      net(왕복 0.15% 차감) 로 기록 → 추가 차감 없이 그대로 사용 (ops-steward §0).
    """
    if df_rec is None or len(df_rec) == 0:
        return {"channel": "recommend", "n_alerts_total": 0}

    df = _rows_through_asof(df_rec, asof)
    out = {"channel": "recommend", "n_alerts_total": int(len(df))}
    if df.empty:
        return out
    df["date"] = df["date"].astype(str)
    out["first_alert_date"] = str(pd.to_datetime(df["date"]).min().date())
    out["last_alert_date"] = str(pd.to_datetime(df["date"]).max().date())

    # 최신 추천일 top-3 (오늘 분포). dump_risk⚠️ + calibrated 확률 그대로 노출.
    last_day = df["date"].max()
    today = df[df["date"] == last_day].sort_values("rank")
    radar = []
    for _, r in today.iterrows():
        radar.append({
            "date": str(r["date"]),
            "coin": str(r["coin"]).replace("KRW-", ""),
            "rank": _safe_int(r.get("rank")),
            "score": _safe_float(r.get("score")),
            "pump_prob_pct": _safe_float(r.get("pump_prob")),  # 0~1 → viewer 가 %
            "dump_risk_flag": bool(str(r.get("dump_risk_flag")).lower() == "true"),
            "btc_regime": str(r.get("btc_regime", "")),
            "entry_open": _safe_float(r.get("entry_open")),
        })
    out["latest_radar"] = radar
    out["latest_radar_date"] = str(last_day)

    # 실현(closed) 요약 — realized_pct는 이미 net(%p). 같은 날은 equal-weight,
    # 무추천일 cash, 복리·초기 equity·√365·날짜-cluster CI를 공통 엔진으로 계산한다.
    closed = df[df["status"].astype(str) == "closed"].copy()
    out["n_closed"] = int(len(closed))
    if len(closed):
        net_all = pd.to_numeric(closed["realized_pct"], errors="coerce")
        net = net_all.dropna()
        if len(net):
            dates = pd.to_datetime(closed.loc[net.index, "date"])
            daily = daily_equal_weight(
                dates,
                net / 100.0,
                calendar_end=asof,
            )
            signal_daily = daily_equal_weight(
                dates,
                net / 100.0,
                include_no_trade_days=False,
            )
            portfolio = summarize_daily(daily)
            cluster = date_cluster_bootstrap(signal_daily)
            out["avg_net_realized_pct"] = float(net.mean())
            out["cum_net_pnl_pct"] = float(
                portfolio["cumulative_return"] * 100
            )
            out["legacy_per_trade_sum_net_pct"] = float(net.sum())
            out["win_rate_pct"] = float((net > 0).mean() * 100)
            out["n_signal_days"] = int(len(signal_daily))
            out["n_calendar_days"] = portfolio["n_calendar_days"]
            out["max_drawdown_pct"] = float(
                portfolio["max_drawdown"] * 100
            )
            out["sharpe_ann"] = float(portfolio["sharpe_ann"])
            out["sortino_ann"] = float(portfolio["sortino_ann"])
            out["date_cluster_ci"] = {
                **cluster,
                "mean_return_ci95_pct": [
                    float(value * 100)
                    for value in cluster.get("mean_return_ci95", [])
                ],
                "cumulative_return_ci95_pct": [
                    float(value * 100)
                    for value in cluster.get("cumulative_return_ci95", [])
                ],
            }
        hits = pd.to_numeric(closed.get("pump20_hit"), errors="coerce").dropna()
        if len(hits):
            out["pump20_hit_rate_pct"] = float(hits.mean() * 100)
            out["pump20_hit_basis"] = "legacy_full_day_D1"
        if "post_send_pump20_hit" in closed.columns:
            post_send_hits = pd.to_numeric(
                closed["post_send_pump20_hit"], errors="coerce"
            ).dropna()
            if len(post_send_hits):
                out["post_send_pump20_hit_rate_pct"] = float(
                    post_send_hits.mean() * 100
                )
                out["post_send_pump20_hit_basis"] = "sent_at_after_15m_path"
        if "exit_reason" in closed.columns:
            out["exit_reason_counts"] = {
                str(k): int(v) for k, v in
                closed["exit_reason"].dropna().value_counts().items()
            }
    return out


# ============================================================================
# History (combined rows)
# ============================================================================
def history_rows(
    df_dist: pd.DataFrame,
    df_preopen: pd.DataFrame,
    df_rec: pd.DataFrame | None = None,
) -> list[dict]:
    rows = []
    # 현행 라이브 채널 R1 — 알림 히스토리 표의 정본 첫 채널 (2026-07-28 추가:
    # 뷰어가 legacy 2채널만 보여 '3일 전에 죽은 시스템'처럼 보이던 결함 수리).
    if df_rec is not None and len(df_rec):
        for _, r in df_rec.iterrows():
            realized = _safe_float(r.get("realized_pct"))
            rows.append({
                "date": str(r["date"]),
                "channel": "recommend",
                "coin": str(r["coin"]).replace("KRW-", ""),
                "setups": str(r.get("exit_reason", "") or ""),
                "btc_regime": str(r.get("btc_regime", "")),
                "entry_price": _safe_float(r.get("entry_open")),
                "p_up10": _safe_float(r.get("p_up10")),
                "p_dn5": _safe_float(r.get("p_dn5")),
                "composite": _safe_float(r.get("score")),
                "next_max_pct": _safe_float(r.get("max_return_pct")),
                "next_min_pct": _safe_float(r.get("min_return_pct")),
                "next_close_pct": realized,
                "hit_h2": None,
                "hit_h6": None,
                "hit_h5": None,
                "virtual_pnl_pct": realized,
                "status": str(r.get("status", "")),
                "decision": "R1 champion",
                "idea_id": "—",
                "setup_quality": "—",
                "calibrated_hit_pct": None,
                "expected_edge_pct": None,
                "decision_reason": "",
            })
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
            # 정책 layer (2026-05-25 추가) — graceful fallback for legacy rows
            "decision": str(r.get("decision", "")) or "—",
            "idea_id": str(r.get("idea_id", "")) or "—",
            "setup_quality": str(r.get("setup_quality", "")) or "—",
            "calibrated_hit_pct": _safe_float(r.get("calibrated_hit_pct")),
            "expected_edge_pct": _safe_float(r.get("expected_edge_pct")),
            "decision_reason": str(r.get("decision_reason", "")) or "",
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
            # 정책 layer
            "decision": str(r.get("decision", "")) or "—",
            "idea_id": str(r.get("idea_id", "")) or "—",
            "setup_quality": str(r.get("setup_quality", "")) or "—",
            "calibrated_hit_pct": _safe_float(r.get("calibrated_hit_pct")),
            "expected_edge_pct": _safe_float(r.get("expected_edge_pct")),
            "decision_reason": str(r.get("decision_reason", "")) or "",
        })

    rows.sort(key=lambda x: (x["date"], x["channel"], x["coin"]), reverse=True)
    return rows


def _safe_float(v):
    """어떤 값이든 안전하게 float 으로. ledger 가 '6.4%'·'1,234'·''·'nan' 같은 문자열을
    저장해도 크래시 없이 None 또는 숫자 반환 (대시보드 publish 가 한 셀 때문에 죽지 않게)."""
    try:
        if isinstance(v, str):
            v = v.replace("%", "").replace(",", "").strip()
            if v == "" or v.lower() in ("nan", "none", "null", "-"):
                return None
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
        if isinstance(v, str):
            v = v.replace("%", "").replace(",", "").strip()
            if v == "" or v.lower() in ("nan", "none", "null", "-"):
                return None
            v = float(v)   # '3.0' / '3%' → 3
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
                           max_col: str, close_col: str, *,
                           asof=None) -> list[dict]:
    """일자별 누적 가상 PnL 시계열.

    같은 날 알림 N 개면 equal-weight mean. cum = 복리 일별 net.
    """
    df = _rows_through_asof(df, asof)
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
    daily = daily_equal_weight(
        closed["date"],
        closed["pnl"],
        calendar_end=asof,
    )
    cum = (1.0 + daily).cumprod() - 1.0
    return [
        {
            "date": str(d),
            "daily_pnl_pct": float(daily.loc[d] * 100),
            "cum_pnl_pct": float(cum.loc[d] * 100),
        }
        for d in cum.index
    ]


def _load_optional_artifact(
    path: str | Path,
    *,
    asof=None,
    kind: str,
) -> dict | None:
    """Load an optional dashboard artifact only when its lineage is safe."""
    artifact_path = Path(path)
    cutoff = normalize_kst_date(asof)
    if kind == "policy_competition":
        try:
            return load_policy_artifact(
                artifact_path,
                csv_path=artifact_path.with_suffix(".csv"),
                db_path=POLICY_DB,
                asof=cutoff,
                require_current=True,
            )
        except PolicyArtifactError:
            return None
    if kind == "idea_validation":
        try:
            return load_idea_validation_artifact(
                artifact_path,
                asof=cutoff,
                require_current=True,
            )
        except IdeaArtifactError:
            return None
    try:
        payload = strict_json_object(artifact_path)
    except ArtifactValidationError:
        return None
    if kind == "recommendation_meta":
        date_range = payload.get("date_range")
        trained_through = (
            date_range.get("end") if isinstance(date_range, dict) else None
        )
        built_at = payload.get("built_at")
        if not trained_through or not built_at:
            # 새 ordered-증거 계약 이후 표본 0 → REJECTED 진단 아티팩트.
            # null 로 삼키면 뷰어가 침묵하므로 상태·사유만 최소 노출한다.
            status = payload.get("artifact_status")
            if status:
                return {
                    "artifact_status": str(status),
                    "reason": str(payload.get("reason") or ""),
                    "built_at": str(built_at or ""),
                }
            return None
        try:
            trained_date = normalize_kst_date(trained_through)
            built_date = normalize_kst_date(built_at)
        except ValueError:
            return None
        return payload if trained_date <= cutoff and built_date <= cutoff else None

    artifact_value = payload.get("asof")
    if not artifact_value:
        return None
    try:
        artifact_asof = normalize_kst_date(artifact_value)
    except ValueError:
        return None
    if artifact_asof > cutoff:
        return None

    safe = dict(payload)
    safe["asof"] = str(artifact_asof.date())
    return safe


def _input_lineage_matches_current(lineage: dict[str, Any]) -> bool:
    """Compatibility wrapper around the canonical idea-lineage validator."""
    return input_manifest_matches_current(lineage)


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-ledger", default="output/paper_ledger.csv")
    parser.add_argument("--paper-ledger-preopen", default="output/paper_ledger_preopen.csv")
    parser.add_argument("--shadow-ledger-distribution", default="output/shadow_ledger_distribution.csv")
    parser.add_argument("--shadow-ledger-preopen", default="output/shadow_ledger_preopen.csv")
    parser.add_argument("--shadow-ledger-recommend", default="output/shadow_ledger_recommend.csv")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--asof", type=str, help="기준 시점 (default=now)")
    parser.add_argument(
        "--pin",
        default=None,
        help=(
            "명시 암호화 passphrase. 미지정 시 PRELUDE_DASHBOARD_PIN env; "
            f"최소 {MIN_DASHBOARD_PASSPHRASE_LENGTH}자."
        ),
    )
    parser.add_argument(
        "--no-encrypt",
        action="store_true",
        help="평문 출력 (테스트용 only). 라이브 publish 에는 절대 금지.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("dashboard")

    # PIN 우선순위 — 빈 문자열을 절대 평문으로 해석하지 X.
    # 평문 원하면 명시적 --no-encrypt.
    if args.no_encrypt:
        pin = None
    else:
        try:
            pin = resolve_dashboard_passphrase(args.pin)
        except ValueError as exc:
            parser.error(str(exc))
    log.info(f"encryption: {'PIN ' + ('*' * len(pin)) if pin else 'OFF (plaintext)'}")
    try:
        generation_id = resolve_dashboard_generation_id()
    except ValueError as exc:
        parser.error(str(exc))

    asof = normalize_kst_date(args.asof)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_dist = _rows_through_asof(_load_or_empty(args.paper_ledger), asof)
    df_pre = _rows_through_asof(_load_or_empty(args.paper_ledger_preopen), asof)
    # policy_evolution 등 cross-cutting consumer 가 사용할 수 있게 가상 PnL 컬럼 부착.
    # distribution = next_max/next_close, preopen = first_1h_max/first_1h_close.
    for _ledger, _max, _close in [
        (df_dist, "next_max_return_pct", "next_close_return_pct"),
        (df_pre, "first_1h_max_return_pct", "first_1h_close_return_pct"),
    ]:
        if len(_ledger) and _max in _ledger.columns and "virtual_pnl_pct" not in _ledger.columns:
            close_values = _ledger.get(
                _close,
                pd.Series(np.nan, index=_ledger.index),
            )
            _ledger["virtual_pnl_pct"] = (
                _net_pnl_series(_ledger[_max], close_values) * 100.0
            )
    # SHADOW 추천 레이더 스캐너 일일 출력 — graceful (파일 없으면 빈 채널).
    df_rec = _rows_through_asof(
        _load_or_empty(args.shadow_ledger_recommend),
        asof,
    )
    log.info(f"loaded: distribution {len(df_dist)} rows, preopen {len(df_pre)} rows, "
             f"recommend {len(df_rec)} rows")

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

    # 신규 (5/25 정책 layer 추가 후) — idea_validation + meta model card 통합.
    # 산출물 파일이 없거나 as-of/lineage가 맞지 않으면 보수적으로 제외.
    idea_validation = _load_optional_artifact(
        "output/idea_validation_summary.json",
        asof=asof,
        kind="idea_validation",
    )
    if idea_validation:
        idea_validation = _bind_idea_dashboard_generation(
            idea_validation,
            generation_id,
        )
    meta_model = _load_optional_artifact(
        "output/recommendation_meta_validation.json",
        asof=asof,
        kind="recommendation_meta",
    )
    policy_competition = _load_optional_artifact(
        "output/policy_competition_summary.json",
        asof=asof,
        kind="policy_competition",
    )
    pump_hunter = _build_pump_hunter_payload(
        "output/shadow_ledger_pump_hunter.csv",
        asof=asof,
    )
    pump_hunter_v2 = _build_pump_hunter_payload(
        "output/shadow_ledger_pump_hunter_v2.csv",
        asof=asof,
    )
    champion_gate = _build_champion_gate_payload(
        "output/champion_state.json",
        asof=asof,
    )
    for artifact_name, artifact_path, artifact in (
        ("idea_validation", "output/idea_validation_summary.json", idea_validation),
        ("recommendation_meta", "output/recommendation_meta_validation.json", meta_model),
        ("policy_competition", "output/policy_competition_summary.json", policy_competition),
        ("champion_gate", "output/champion_state.json", champion_gate),
    ):
        if artifact is None and Path(artifact_path).exists():
            log.warning(
                "%s excluded: invalid, unattributed, or future-dated for KST asof %s",
                artifact_name,
                asof.date(),
            )

    summary = {
        "dashboard_generation_id": generation_id,
        "asof": asof.isoformat(),
        "asof_timezone": "Asia/Seoul",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "channels": {
            "distribution": compute_distribution_summary(
                df_dist,
                btc_series=btc_bench,
                asof=asof,
            ),
            "preopen": compute_preopen_summary(
                df_pre,
                btc_series=btc_bench,
                asof=asof,
            ),
            "recommend": compute_recommend_summary(df_rec, asof=asof),
        },
        "notes_vs_system": notes_vs,
        "coin_universe": coin_matrix,
        # 정책 layer (2026-05-25 추가) — narrative + 결과
        # events / live_summary 는 policy_history.py 에서 자동 로드 + paper_ledger 로 metrics 산출
        "policy_evolution": {
            "version": "2026-05-25.1",
            "policy_id": "setup_quality_policy_v1",
            **build_policy_evolution(df_dist, df_pre, asof=str(asof)),
            "rule_set": [
                {"channel": "distribution", "regime": "bear_volatile", "decision": "SILENCE"},
                {"channel": "distribution", "regime": "bear_quiet", "decision": "WATCH_ONLY (A_TRIPLE 만 ACTIVE)"},
                {"channel": "distribution", "regime": "bull*", "decision": "ACTIVE if S03-quality + h6 edge ≥ 0"},
                {"channel": "preopen", "regime": "bear*", "decision": "WATCH/SILENCE"},
                {"channel": "preopen", "regime": "bull*", "decision": "ACTIVE if composite ≥ 1.5 + p_1h_5 ≥ 0.35"},
            ],
        },
        "idea_validation": idea_validation,  # full summary (model_card / quality / replay / gate)
        "meta_model": meta_model,  # recommendation_quality_meta_label_v1 학습 결과
        "policy_competition": policy_competition,  # model + send-policy CLOSED forward audit (+exit_lab)
        "pump_hunter": pump_hunter,  # PUMP detector SHADOW — 오늘 watchlist + 일별 capture
        "pump_hunter_v2": pump_hunter_v2,  # 🎯 v2 radar (Binance volsurge) — watchlist + capture
        "champion_gate": champion_gate,  # 슬롯별 forward 검증 진행률 (n_days/MIN_CLOSED)
    }
    _write_json(out_dir / "summary.json", summary, passphrase=pin)
    log.info(f"saved summary.json (idea_validation={bool(idea_validation)}, meta_model={bool(meta_model)})")

    # 2) history.json
    history = {
        "dashboard_generation_id": generation_id,
        "asof": asof.isoformat(),
        "asof_timezone": "Asia/Seoul",
        "rows": history_rows(df_dist, df_pre, df_rec),
    }
    _write_json(out_dir / "history.json", history, passphrase=pin)
    log.info(f"saved history.json ({len(history['rows'])} rows)")

    # 3) accuracy.json — 시계열 (rolling + cum_pnl + underwater + rolling_sharpe + monthly)
    closed_dist = df_dist[df_dist["status"].astype(str) == "closed"].copy() if "status" in df_dist.columns else df_dist.iloc[0:0]
    closed_pre = df_pre[df_pre["status"].astype(str) == "closed"].copy() if "status" in df_pre.columns else df_pre.iloc[0:0]

    accuracy = {
        "dashboard_generation_id": generation_id,
        "asof": asof.isoformat(),
        "asof_timezone": "Asia/Seoul",
        "window_days": ROLLING_WINDOW_DAYS,
        "distribution": {
            "rolling": rolling_accuracy(
                df_dist, ["hit_h2", "hit_h6", "hit_h5"]
            ),
            "cum_pnl": cumulative_pnl_series(
                df_dist,
                "next_max_return_pct",
                "next_close_return_pct",
                asof=asof,
            ),
            "rolling_sharpe": compute_rolling_sharpe(
                closed_dist,
                "next_max_return_pct",
                "next_close_return_pct",
                asof=asof,
            ),
            "underwater": compute_underwater_series(
                closed_dist,
                "next_max_return_pct",
                "next_close_return_pct",
                asof=asof,
            ),
            "monthly_returns": compute_monthly_returns(
                closed_dist,
                "next_max_return_pct",
                "next_close_return_pct",
                asof=asof,
            ),
        },
        "preopen": {
            "rolling": rolling_accuracy(
                df_pre,
                ["hit_first15_3pct", "hit_first15_5pct",
                 "hit_first1h_3pct", "hit_first1h_5pct"],
            ),
            "cum_pnl": cumulative_pnl_series(
                df_pre,
                "first_1h_max_return_pct",
                "first_1h_close_return_pct",
                asof=asof,
            ),
            "rolling_sharpe": compute_rolling_sharpe(
                closed_pre,
                "first_1h_max_return_pct",
                "first_1h_close_return_pct",
                asof=asof,
            ),
            "underwater": compute_underwater_series(
                closed_pre,
                "first_1h_max_return_pct",
                "first_1h_close_return_pct",
                asof=asof,
            ),
            "monthly_returns": compute_monthly_returns(
                closed_pre,
                "first_1h_max_return_pct",
                "first_1h_close_return_pct",
                asof=asof,
            ),
        },
        "btc_benchmark": btc_bench,
    }
    _write_json(out_dir / "accuracy.json", accuracy, passphrase=pin)
    log.info(f"saved accuracy.json (btc benchmark: {len(btc_bench)} days)")

    # 4) idea_validation.json — ACTIVE/WATCH_ONLY/SILENCE attribution.
    idea_input_manifest = build_idea_input_manifest(args)
    idea_candidates = load_idea_candidate_ledger(args)
    _, idea_payload = build_idea_validation_report(
        idea_candidates,
        asof=asof,
        input_manifest=idea_input_manifest,
    )
    idea_payload = _bind_idea_dashboard_generation(
        idea_payload,
        generation_id,
    )
    if build_idea_input_manifest(args) != idea_input_manifest:
        raise RuntimeError("dashboard idea-validation inputs changed during build")
    validate_idea_validation_payload(
        idea_payload,
        asof=asof,
        require_current=True,
    )
    _write_json(out_dir / "idea_validation.json", idea_payload, passphrase=pin)
    log.info(f"saved idea_validation.json ({idea_payload.get('n_candidates', 0)} candidates)")

    # quick stdout summary
    print()
    print("=== Dashboard summary ===")
    for ch, s in summary["channels"].items():
        v = s.get("virtual", {})
        cumulative = v.get(
            "cum_pnl_pct",
            s.get("cum_net_pnl_pct", float("nan")),
        )
        print(
            f"  {ch:<13} alerts={s.get('n_alerts_total',0):>4} "
            f"closed={s.get('n_closed',0):>4} "
            f"cum_pnl={cumulative:+.2f}% "
            f"avg_max={s.get('avg_max_return_pct', float('nan')):+.2f}% "
            f"avg_min={s.get('avg_min_return_pct', float('nan')):+.2f}%"
        )
    print(f"\nout_dir: {out_dir}")


def _load_or_empty(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    raw = _read_stable_artifact_bytes(p)
    if raw is None:
        return pd.DataFrame()
    df = pd.read_csv(io.BytesIO(raw))
    if "status" in df.columns:
        df["status"] = df["status"].fillna("").astype(str)
    return df


def _build_pump_hunter_payload(
    ledger_path: str,
    *,
    asof=None,
) -> dict | None:
    """PUMP hunter SHADOW dashboard payload — 오늘 watchlist + 일별 capture 시계열.

    shadow_ledger_pump_hunter.csv 가 아직 없으면 None (cron 1회 후 생성).
    challenger_only — Telegram/ACTIVE 승격 금지가 코드 레벨에서 보장됨.
    """
    p = Path(ledger_path)
    raw = _read_stable_artifact_bytes(p)
    if raw is None:
        return {
            "status": "pending_first_cron",
            "note": "shadow_ledger_pump_hunter.csv 가 아직 없음 — 내일 09:05 cron 후 첫 row 생성",
        }
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except (pd.errors.ParserError, UnicodeError, ValueError):
        return None
    if df.empty or "date" not in df.columns:
        return {"status": "empty", "note": "ledger empty", "rows_total": 0}

    df = _rows_through_asof(df, asof)
    if df.empty:
        return {
            "status": "empty_asof",
            "note": "no valid ledger rows at or before dashboard asof",
            "rows_total": 0,
        }
    df["date"] = df["date"].astype(str)
    latest_date = df["date"].max()

    # 오늘 (가장 최근 date) 의 watchlist — rank 순으로 표시
    today_df = df[df["date"] == latest_date].copy()
    if "rank" in today_df.columns:
        today_df = today_df.sort_values("rank")
    watchlist = []
    for _, r in today_df.iterrows():
        # ★ 전부 _safe_float/_safe_int (bare float()/int() 금지) — ledger 가 '6.4%' 같은 % 문자열을
        #   저장해도 한 셀 때문에 build 가 죽지 않게. (2026-06-05 사고: pump_prob_pct='6.4%' → publish 크래시)
        item = {
            "rank": _safe_int(r.get("rank")),
            "coin": str(r.get("coin", "")).replace("KRW-", ""),
            "score": _safe_float(r.get("score")),
            "pump_prob_pct": _safe_float(r.get("pump_prob_pct")),   # '6.4%' → 6.4
            "rule_id": str(r.get("rule_id", "")),
            "roc_7d_rank": _safe_float(r.get("roc_7d_rank")),
            "atr_pct_14": _safe_float(r.get("atr_pct_14")),
            "log_return_1d": _safe_float(r.get("log_return_1d")),
            "status": str(r.get("status", "")),
        }
        # v2 전용 — Binance 거래량 surge (컬럼 있을 때만)
        if "b_vol_surge" in today_df.columns:
            item["b_vol_surge"] = _safe_float(r.get("b_vol_surge"))
        watchlist.append(item)

    # 일별 capture rate — closed row 만 (pump20_hit 컬럼이 있을 때)
    daily = []
    closed = df[df.get("status", "").astype(str) == "closed"] if "status" in df.columns else df.iloc[0:0]
    if len(closed) and "pump20_hit" in closed.columns:
        grp = closed.groupby("date").agg(
            n_picks=("coin", "count"),
            n_pump20_hit=("pump20_hit", lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).sum())),
        ).reset_index()
        grp["capture_pct"] = (grp["n_pump20_hit"] / grp["n_picks"].clip(lower=1)) * 100.0
        for _, r in grp.iterrows():
            daily.append({
                "date": str(r["date"]),
                "n_picks": int(r["n_picks"]),
                "n_pump20_hit": int(r["n_pump20_hit"]),
                "capture_pct": float(r["capture_pct"]),
            })

    # 누적 요약
    total_n = int(len(df))
    closed_n = int(len(closed))
    pump_hits = int(pd.to_numeric(closed.get("pump20_hit", pd.Series([])), errors="coerce").fillna(0).sum()) if closed_n else 0
    return {
        "status": "live",
        "latest_date": latest_date,
        "rows_total": total_n,
        "rows_closed": closed_n,
        "pump20_hits_total": pump_hits,
        "pump20_capture_pct": (pump_hits / closed_n * 100.0) if closed_n else None,
        "watchlist": watchlist,
        "daily_capture": daily,
        "note": "challenger_only — Telegram/ACTIVE 승격 금지 (코드 레벨 차단). record-only.",
    }


def _build_champion_gate_payload(
    state_path: str,
    *,
    asof=None,
) -> dict | None:
    """champion gate 진행률 — output/champion_state.json 에서 슬롯별 n_days/MIN_CLOSED.

    "언제쯤 fallback 을 벗어나 첫 실전 champion 판정이 가능한가" 를 사용자가
    한눈에 보도록. MIN_CLOSED 등 상수는 state 의 config 에 이미 기록돼 있음.
    """
    p = Path(state_path)
    cutoff = normalize_kst_date(asof)
    try:
        artifact = load_champion_state_artifact(
            p,
            expected_asof=cutoff,
        )
    except ChampionStateError:
        return None
    if artifact is None:
        return None
    state = artifact.payload
    state_asof = normalize_kst_date(state["asof"])
    cfg = state.get("config", {})
    if not isinstance(cfg, dict):
        return None
    try:
        min_closed = int(cfg.get("min_closed", 30))
    except (TypeError, ValueError):
        return None
    if min_closed < 0:
        return None
    slots_out = []
    slots = state.get("slots")
    if not isinstance(slots, dict):
        return None
    for slot, v in slots.items():
        if not isinstance(v, dict):
            continue
        m = v.get("metric") or {}
        if not isinstance(m, dict):
            continue
        last_date = m.get("last_date")
        if last_date:
            try:
                if normalize_kst_date(last_date) > cutoff:
                    continue
            except ValueError:
                continue
        since = v.get("since")
        if since:
            try:
                if normalize_kst_date(since) > cutoff:
                    continue
            except ValueError:
                continue
        try:
            n_days = int(m.get("n_days") or 0)
        except (TypeError, ValueError):
            continue
        if n_days < 0:
            continue
        slots_out.append({
            "slot": slot,
            "champion_id": v.get("champion_id"),
            "is_fallback": bool(v.get("is_fallback")),
            "since": v.get("since"),
            "n_days": n_days,
            "min_closed": min_closed,
            "progress_pct": round(min(100.0, n_days / min_closed * 100.0), 1) if min_closed else None,
            "days_remaining": max(0, min_closed - n_days),
        })
    return {
        "asof": str(state_asof.date()),
        "updated_at": state.get("updated_at"),
        "min_closed": min_closed,
        "hyst_k": cfg.get("hyst_k"),
        "slots": slots_out,
    }


def _sanitize_json(obj):
    """NaN/Infinity → None. Python json allows NaN; browser JSON.parse rejects."""
    import math
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    return obj


def _read_stable_artifact_bytes(path: Path) -> bytes | None:
    """Read one regular non-symlink artifact from a stable named generation."""
    try:
        parent_before = path.parent.lstat()
    except OSError as exc:
        raise ArtifactValidationError(
            f"dashboard input parent cannot be inspected: {path.parent}"
        ) from exc
    if not stat.S_ISDIR(parent_before.st_mode):
        raise ArtifactValidationError(
            f"dashboard input parent must be a real directory: {path.parent}"
        )
    try:
        before = file_identity(path, root=path.parent)
    except ArtifactSourceChangedError as exc:
        raise ArtifactValidationError(
            f"dashboard input is not a stable regular file: {path}"
        ) from exc
    if not before["exists"]:
        try:
            after = file_identity(path, root=path.parent)
            parent_after = path.parent.lstat()
        except ArtifactSourceChangedError as exc:
            raise ArtifactValidationError(
                f"dashboard input appeared unsafely during read: {path}"
            ) from exc
        except OSError as exc:
            raise ArtifactValidationError(
                f"dashboard input parent changed during read: {path.parent}"
            ) from exc
        if (
            before != after
            or (
                parent_before.st_dev,
                parent_before.st_ino,
                parent_before.st_mode,
                parent_before.st_mtime_ns,
                parent_before.st_ctime_ns,
            )
            != (
                parent_after.st_dev,
                parent_after.st_ino,
                parent_after.st_mode,
                parent_after.st_mtime_ns,
                parent_after.st_ctime_ns,
            )
        ):
            raise ArtifactValidationError(
                f"dashboard input appeared during read: {path}"
            )
        return None
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ArtifactValidationError(
            f"dashboard input cannot be read: {path}"
        ) from exc
    try:
        after = file_identity(path, root=path.parent)
    except ArtifactSourceChangedError as exc:
        raise ArtifactValidationError(
            f"dashboard input changed unsafely during read: {path}"
        ) from exc
    try:
        parent_after = path.parent.lstat()
    except OSError as exc:
        raise ArtifactValidationError(
            f"dashboard input parent changed during read: {path.parent}"
        ) from exc
    if (
        before != after
        or before.get("size") != len(raw)
        or before.get("sha256") != sha256_bytes(raw)
        or (
            parent_before.st_dev,
            parent_before.st_ino,
            parent_before.st_mode,
            parent_before.st_mtime_ns,
            parent_before.st_ctime_ns,
        )
        != (
            parent_after.st_dev,
            parent_after.st_ino,
            parent_after.st_mode,
            parent_after.st_mtime_ns,
            parent_after.st_ctime_ns,
        )
    ):
        raise ArtifactValidationError(
            f"dashboard input changed during read: {path}"
        )
    return raw


def _write_json(path: Path, payload: dict, passphrase: str | None = None):
    payload = _sanitize_json(payload)
    if passphrase:
        payload = _encrypt_payload(payload, passphrase)
    serialized = (
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=str,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    atomic_write_bytes(path, serialized)


def _encrypt_payload(payload: dict, passphrase: str) -> dict:
    """PBKDF2-HMAC-SHA256 + AES-256-CBC + HMAC-SHA256(salt||iv||ct).

    Viewer 의 decryptPayload (papers index.html 동일) 와 호환. client-side
    ciphertext라 offline brute-force 자체를 막을 수는 없으므로
    MIN_DASHBOARD_PASSPHRASE_LENGTH(현재 4 — 사용자 명시 승인 2026-07-28,
    짧은 PIN 의 무차별 대입 취약 수용) 이상 passphrase 를 강제한다.
    이는 서버 측 접근제어를 대체하지 않는다.
    """
    from cryptography.hazmat.primitives import hashes, hmac as crypto_hmac
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    plaintext = json.dumps(payload, ensure_ascii=False, default=str, allow_nan=False).encode("utf-8")
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
    encryptor = cipher.encryptor()
    ct = encryptor.update(padded) + encryptor.finalize()

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

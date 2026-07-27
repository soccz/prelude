"""성과 메트릭 (LEDGER §6).

ledger.csv → Sharpe / MDD / hit rate / 평균 hold 등.

핵심 함수:
  - compute_daily_pnl(ledger_df) — 일별 합산
  - compute_sharpe(daily_returns)
  - compute_mdd(equity_curve)
  - compute_summary(ledger_df) — 종합 dict
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ledger.config import INITIAL_CAPITAL_KRW
from ledger.portfolio_metrics import max_drawdown_from_equity


# ============================================================================
# 기본 통계
# ============================================================================
def compute_sharpe(daily_returns: pd.Series, periods_per_year: int = 365) -> float:
    """일별 수익률 → annualized Sharpe."""
    if len(daily_returns) < 2:
        return 0.0
    mu = daily_returns.mean()
    sigma = daily_returns.std()
    if sigma == 0:
        return 0.0
    return float(mu / sigma * np.sqrt(periods_per_year))


def compute_mdd(
    equity_curve: pd.Series,
    *,
    initial_equity: float | None = None,
) -> float:
    """최대 낙폭.

    ``initial_equity``를 주면 첫 관측 전 자본도 peak 후보에 포함한다. 입력
    curve의 단위(1.0 normalized vs KRW)를 추측하지 않도록 기본값은 기존
    호환 방식(None)이며, 운영 호출자는 시작 자본을 명시한다.
    """
    if len(equity_curve) == 0:
        return 0.0
    values = pd.to_numeric(equity_curve, errors="coerce").dropna()
    if initial_equity is not None:
        values = pd.concat([
            pd.Series([float(initial_equity)]),
            values.reset_index(drop=True),
        ], ignore_index=True)
    return max_drawdown_from_equity(values)


def compute_daily_pnl(ledger_df: pd.DataFrame) -> pd.DataFrame:
    """ledger 의 entry_date 기준 일별 net PnL 합산.

    return: { date, daily_net_return, daily_pnl_krw, equity_krw, drawdown }
    """
    if len(ledger_df) == 0:
        return pd.DataFrame()

    df = ledger_df.copy()
    df["entry_date"] = pd.to_datetime(df["entry_date"]).dt.normalize()

    # 한 날에 여러 포지션 → position_size_pct 가중 합
    daily = df.groupby("entry_date").apply(
        lambda g: pd.Series({
            "n_positions": len(g),
            "n_tp": int((g["exit_type"] == "tp").sum()),
            "n_sl": int((g["exit_type"] == "sl").sum()),
            "n_eod": int((g["exit_type"] == "eod").sum()),
            "daily_net_return": float((g["net_return_pct"] * g["position_size_pct"]).sum()),
            "avg_hold_hours": float(g["hold_hours"].mean()),
        })
    ).reset_index()

    # equity curve
    daily["equity_krw"] = INITIAL_CAPITAL_KRW * (1 + daily["daily_net_return"]).cumprod()
    daily["daily_pnl_krw"] = daily["equity_krw"].diff().fillna(daily["equity_krw"] - INITIAL_CAPITAL_KRW)
    daily["peak"] = daily["equity_krw"].cummax().clip(lower=INITIAL_CAPITAL_KRW)
    daily["drawdown"] = (daily["equity_krw"] - daily["peak"]) / daily["peak"]

    return daily


# ============================================================================
# 종합 요약
# ============================================================================
def compute_summary(ledger_df: pd.DataFrame) -> dict:
    """ledger 전체 요약."""
    if len(ledger_df) == 0:
        return {"n_positions": 0}

    daily = compute_daily_pnl(ledger_df)

    n = len(ledger_df)
    n_tp = int((ledger_df["exit_type"] == "tp").sum())
    n_sl = int((ledger_df["exit_type"] == "sl").sum())
    n_eod = int((ledger_df["exit_type"] == "eod").sum())

    cum_return = float(daily["equity_krw"].iloc[-1] / INITIAL_CAPITAL_KRW - 1) if len(daily) > 0 else 0.0
    cum_pnl_krw = float(daily["equity_krw"].iloc[-1] - INITIAL_CAPITAL_KRW) if len(daily) > 0 else 0.0
    sharpe = compute_sharpe(daily["daily_net_return"]) if len(daily) > 0 else 0.0
    mdd = (
        compute_mdd(daily["equity_krw"], initial_equity=INITIAL_CAPITAL_KRW)
        if len(daily) > 0 else 0.0
    )

    return {
        "n_positions": n,
        "n_days": int(daily["entry_date"].nunique()) if len(daily) > 0 else 0,
        "tp_hit_rate": n_tp / n if n > 0 else 0.0,
        "sl_hit_rate": n_sl / n if n > 0 else 0.0,
        "eod_rate": n_eod / n if n > 0 else 0.0,
        "avg_hold_hours": float(ledger_df["hold_hours"].mean()),
        "cum_return_pct": cum_return,
        "cum_pnl_krw": cum_pnl_krw,
        "current_equity_krw": float(daily["equity_krw"].iloc[-1]) if len(daily) > 0 else INITIAL_CAPITAL_KRW,
        "annualized_sharpe": sharpe,
        "max_drawdown_pct": mdd,
        "avg_net_return_per_position": float(ledger_df["net_return_pct"].mean()),
    }


# ============================================================================
# 텔레그램 / NOTES 포맷
# ============================================================================
def format_summary(summary: dict, period_label: str = "전체") -> str:
    """사용자 친화 요약 텍스트."""
    if summary.get("n_positions", 0) == 0:
        return f"=== {period_label} ===\n(가상 ledger 비어있음)"

    return (
        f"=== {period_label} ===\n"
        f"포지션 수: {summary['n_positions']} ({summary['n_days']}일)\n"
        f"누적 net PnL: {summary['cum_pnl_krw']/10000:.1f} 만원 "
        f"({summary['cum_return_pct']*100:+.2f}%)\n"
        f"현재 equity: {summary['current_equity_krw']/10000:.1f} 만원\n"
        f"Sharpe (annual): {summary['annualized_sharpe']:.2f}\n"
        f"MDD: {summary['max_drawdown_pct']*100:.2f}%\n"
        f"TP 적중: {summary['tp_hit_rate']*100:.1f}% | "
        f"SL 적중: {summary['sl_hit_rate']*100:.1f}% | "
        f"EOD: {summary['eod_rate']*100:.1f}%\n"
        f"평균 hold: {summary['avg_hold_hours']:.1f}h\n"
        f"포지션 당 평균 net: {summary['avg_net_return_per_position']*100:+.2f}%"
    )

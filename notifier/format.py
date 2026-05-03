"""텔레그램 메시지 포맷터 (OPS §3).

매일 KST 08:30 알림 + 어제 결과 + 가상 ledger + 시스템 정확도.

핵심:
  - format_daily_alert(predictions, ledger_yesterday, summary, calibration_text, asof, btc_regime)
  - tier_emoji(p_ge_5) — sigma-tier (🔥 / ✅ / ▫ / ·)
  - format_silence_alert(reason, asof) — 침묵 시
  - format_preflight_fail(failures, asof)
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd


# ============================================================================
# Tier emoji
# ============================================================================
def tier_emoji(p_ge_5: float, p_ge_10: float = None) -> str:
    """σ-tier (calibration 기반, placeholder cutoff)."""
    if p_ge_10 is not None:
        if p_ge_10 >= 0.5:
            return "🔥"
        if p_ge_10 >= 0.30:
            return "✅"
        if p_ge_5 >= 0.50:
            return "▫"
    else:
        if p_ge_5 >= 0.70:
            return "🔥"
        if p_ge_5 >= 0.55:
            return "✅"
        if p_ge_5 >= 0.40:
            return "▫"
    return "·"


# ============================================================================
# Memo (안정도 ⭐)
# ============================================================================
def stability_marks(coin_history_hit_rate: Optional[float]) -> str:
    """이 코인 과거 ledger TP 적중률 기반."""
    if coin_history_hit_rate is None:
        return ""
    if coin_history_hit_rate >= 0.70:
        return "⭐⭐⭐"
    if coin_history_hit_rate >= 0.60:
        return "⭐⭐"
    if coin_history_hit_rate >= 0.50:
        return "⭐"
    return ""


# ============================================================================
# 메인 메시지 (OPS §3.1)
# ============================================================================
def format_daily_alert(
    predictions: pd.DataFrame,
    ledger_yesterday: Optional[pd.DataFrame] = None,
    ledger_summary: Optional[dict] = None,
    calibration_text: Optional[str] = None,
    asof: Optional[datetime] = None,
    btc_regime: str = "unknown",
    universe_size: int = 0,
    top_k: int = 5,
    coin_hit_rates: Optional[dict] = None,
) -> str:
    """매일 KST 08:30 메인 알림."""
    asof = asof or datetime.now()
    date_str = asof.strftime("%Y-%m-%d")

    lines = [f"🌅 prelude {date_str} (KST 08:30)"]
    lines.append(f"BTC regime: {btc_regime} | universe: {universe_size} 코인")
    lines.append("")

    # 오늘 추천
    lines.append("━━━ 오늘 장중 펌프 분포 (top {}) ━━━".format(min(top_k, len(predictions))))
    if len(predictions) == 0:
        lines.append("  (오늘 시그널 없음 — 침묵)")
    else:
        for i, (_, row) in enumerate(predictions.head(top_k).iterrows()):
            coin = row["coin"]
            base = coin.replace("KRW-", "")
            tier = tier_emoji(row.get("p_ge_5", 0), row.get("p_ge_10", 0))
            stars = stability_marks((coin_hit_rates or {}).get(coin))
            p_str = f"{row['p_ge_5']*100:>3.0f}/{row['p_ge_10']*100:>3.0f}/{row['p_ge_15']*100:>3.0f}/{row['p_ge_20']*100:>3.0f}%"
            exp_max = row.get("expected_max", 0) * 100
            ci_low = row.get("ci_low", 0) * 100
            ci_high = row.get("ci_high", 0) * 100
            lines.append(
                f"{tier} {base:<8} P(≥+5/10/15/20%) = {p_str}\n"
                f"   기대 max {exp_max:+.1f}% | CI [{ci_low:+.0f},{ci_high:+.0f}] {stars}"
            )

    # 어제 결과
    if ledger_yesterday is not None and len(ledger_yesterday) > 0:
        lines.append("")
        lines.append("━━━ 어제 결과 ━━━")
        for _, row in ledger_yesterday.iterrows():
            coin = row["coin"].replace("KRW-", "")
            net = row.get("net_return_pct", 0) * 100
            ext = row.get("exit_type", "?")
            hold = row.get("hold_hours", 0)
            mark = "🎯" if ext == "tp" else ("❌" if ext == "sl" else "▫")
            sig_p = row.get("signal_p_ge_10", 0) * 100
            lines.append(
                f"  {coin:<8} {ext.upper()} {net:+.1f}% in {hold:.0f}h {mark} (예측 P(≥10%) {sig_p:.0f}%)"
            )

    # 가상 ledger 누적
    if ledger_summary:
        lines.append("")
        lines.append("━━━ 가상 ledger ━━━")
        cum_pct = ledger_summary.get("cum_return_pct", 0) * 100
        cum_krw = ledger_summary.get("cum_pnl_krw", 0) / 10000
        mdd = ledger_summary.get("max_drawdown_pct", 0) * 100
        sharpe = ledger_summary.get("annualized_sharpe", 0)
        tp_rate = ledger_summary.get("tp_hit_rate", 0) * 100
        sl_rate = ledger_summary.get("sl_hit_rate", 0) * 100
        avg_hold = ledger_summary.get("avg_hold_hours", 0)
        lines.append(
            f"누적 net {cum_krw:+.1f}만 ({cum_pct:+.2f}%)  MDD {mdd:.2f}%\n"
            f"Sharpe {sharpe:.2f} | TP {tp_rate:.0f}% | SL {sl_rate:.0f}% | 평균 hold {avg_hold:.1f}h"
        )

    # 정확도
    if calibration_text:
        lines.append("")
        lines.append("━━━ 시스템 정확도 (지난 7일) ━━━")
        lines.append(calibration_text)

    return "\n".join(lines)


# ============================================================================
# 침묵 / 실패 알림
# ============================================================================
def format_silence_alert(reason: str, asof: Optional[datetime] = None) -> str:
    asof = asof or datetime.now()
    return (
        f"🌅 prelude {asof.strftime('%Y-%m-%d')} (KST 08:30)\n"
        f"⚠️ 오늘 침묵 — 사유: {reason}\n"
        f"가상 ledger 진입 X. 사용자 본인 매매 판단."
    )


def format_preflight_fail(failures: list, details: dict, asof: Optional[datetime] = None) -> str:
    asof = asof or datetime.now()
    lines = [
        f"🌅 prelude {asof.strftime('%Y-%m-%d')} (KST 08:30)",
        f"⚠️ Preflight 실패 — {', '.join(failures)}",
    ]
    for f in failures:
        if f in details:
            lines.append(f"  {f}: {details[f]}")
    lines.append("가상 진입 X. 다음 cron 자동 재시도.")
    return "\n".join(lines)


def format_drift_alert(drift_state, asof: Optional[datetime] = None) -> str:
    asof = asof or datetime.now()
    lines = [
        f"⚠️ prelude {asof.strftime('%Y-%m-%d')} drift detected",
        f"State: {drift_state.state}",
        f"Triggers: {', '.join(drift_state.triggers)}",
    ]
    for k, v in (drift_state.details or {}).items():
        lines.append(f"  {k}: {v}")
    if drift_state.state == "FREEZE":
        lines.append("→ 가상 진입 X + retrain 강제 트리거")
    elif drift_state.state == "WARN":
        lines.append("→ 가상 사이즈 50% cut")
    return "\n".join(lines)

"""텔레그램 메시지 포맷터 (OPS §3).

매일 KST 09:05 알림 + 어제 결과 + 가상 ledger + 시스템 정확도.

핵심:
  - format_daily_alert(predictions, ledger_yesterday, summary, calibration_text, asof, btc_regime)
  - tier_emoji(p_ge_5) — sigma-tier (🔥 / ✅ / ▫ / ·)
  - format_silence_alert(reason, asof) — 침묵 시
  - format_preflight_fail(failures, asof)
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Optional

import pandas as pd


def _fmt_krw_price(p) -> str:
    """KRW 가격 포맷 — 가격대에 따라 소수점 자릿수 자동 조정."""
    try:
        p = float(p)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(p):
        return "—"
    if p >= 1000:
        return f"{p:,.0f}원"
    if p >= 100:
        return f"{p:.1f}원"
    if p >= 10:
        return f"{p:.2f}원"
    if p >= 1:
        return f"{p:.3f}원"
    if p >= 0.01:
        return f"{p:.4f}원"
    return f"{p:.6f}원"


# ============================================================================
# 공통 디자인 헬퍼 (preopen ↔ distribution 통일용)
# ============================================================================
_BTC_REGIME_KR = {
    "bull_quiet": "🟢 강세 안정",
    "bull_volatile": "🟢 강세 변동",
    "bear_quiet": "🔴 약세 안정",
    "bear_volatile": "🔴 약세 변동",
}


def btc_regime_kr(regime: str) -> str:
    """BTC regime key → 한글 + 이모지 라벨. 미지정 키는 그대로 반환."""
    return _BTC_REGIME_KR.get(str(regime), str(regime))


def composite_tier(composite: float, fire: float = 1.5, hot: float = 1.0) -> str:
    """composite score → tier 이모지 (preopen/distribution 공통).

    cutoff 는 placeholder. 두 채널 composite 분포가 다르면 호출부에서 인자 조정.
    """
    try:
        c = float(composite)
    except (TypeError, ValueError):
        return "👀"
    if c >= fire:
        return "🔥"
    if c >= hot:
        return "✨"
    return "👀"


def _finite_float(value, default=None):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(v):
        return default
    return v


def _fmt_pctp(value) -> str:
    v = _finite_float(value)
    if v is None:
        return "검증중"
    return f"{v:+.2f}%p"


def _fmt_pct(value) -> str:
    v = _finite_float(value)
    if v is None:
        return "검증중"
    return f"{v:.1f}%"


def _fmt_rank(value) -> str:
    v = _finite_float(value)
    if v is None or v <= 0:
        return "#?"
    return f"#{int(v)}"


def _fmt_conf(score, tier) -> str:
    v = _finite_float(score)
    t = str(tier or "").strip()
    if v is None and not t:
        return ""
    if v is None:
        return f"conf {t}"
    if t:
        return f"conf {v:.0f}/{t}"
    return f"conf {v:.0f}"


def _hit_label(source: str, fallback: str) -> str:
    source = str(source or "")
    if source.startswith("bucket"):
        return "검증 hit"
    return fallback


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
    """매일 KST 09:05 메인 알림."""
    asof = asof or datetime.now()
    date_str = asof.strftime("%Y-%m-%d")

    lines = [f"🌅 prelude {date_str} (KST 09:05)"]
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
        f"🌅 prelude {asof.strftime('%Y-%m-%d')} (KST 09:05)\n"
        f"⚠️ 오늘 침묵 — 사유: {reason}\n"
        f"가상 ledger 진입 X. 사용자 본인 매매 판단."
    )


def format_preflight_fail(failures: list, details: dict, asof: Optional[datetime] = None) -> str:
    asof = asof or datetime.now()
    lines = [
        f"🌅 prelude {asof.strftime('%Y-%m-%d')} (KST 09:05)",
        f"⚠️ Preflight 실패 — {', '.join(failures)}",
    ]
    for f in failures:
        if f in details:
            lines.append(f"  {f}: {details[f]}")
    lines.append("가상 진입 X. 다음 cron 자동 재시도.")
    return "\n".join(lines)


# ============================================================================
# Detector beta (binary tail detector v1) — Stage 1/2 알림
# ============================================================================
def format_detector_beta(
    alerts: pd.DataFrame,
    threshold: float,
    btc_regime: str,
    universe_size: int,
    diagnose: Optional[dict] = None,
    asof: Optional[datetime] = None,
    dry_run: bool = False,
) -> str:
    """Detector v1 beta 알림 — '매수 추천 X, ≥20% tail 후보' framing.

    silence-heavy: alerts 가 비면 침묵 메시지.
    framing 은 사용자 운영 원칙 그대로:
      'BTC bull regime 에서 ≥20% tail pump 가능성이 OOF 기준 최상위 0.05% 후보'
    """
    asof = asof or datetime.now()
    date_str = asof.strftime("%Y-%m-%d")
    header = f"🌅 prelude detector v1 {date_str} (KST 09:05)"
    if dry_run:
        header += "  [DRY-RUN]"

    lines = [header]
    lines.append(f"BTC regime: {btc_regime} | universe: {universe_size}")
    lines.append(f"threshold: {threshold:.4f} (OOF p99.95, 고정)")
    lines.append("")

    if len(alerts) == 0:
        lines.append("━━━ 침묵 ━━━")
        if btc_regime.startswith("bear"):
            lines.append("(BTC bear regime — 알림 비활성)")
        else:
            lines.append("(threshold 통과 후보 없음 — 오늘은 강한 tail 신호 X)")
    else:
        lines.append(f"━━━ ≥20% tail 후보 ({len(alerts)}건) ━━━")
        lines.append("※ 매수 추천 아님 / 실패 시 큰 손실 가능")
        for _, row in alerts.iterrows():
            coin = row["coin"].replace("KRW-", "")
            score = row["score"]
            rank = int(row.get("alert_rank", 0))
            margin = (score - threshold) * 100
            lines.append(
                f"🔍 {coin:<8} score {score:.4f}  (threshold +{margin:.2f}pp)  rank #{rank}"
            )

    if diagnose:
        lines.append("")
        lines.append("━━━ 진단 ━━━")
        lines.append(
            f"in_regime {diagnose.get('n_in_regime', 0):>4} / "
            f"above_thr {diagnose.get('n_above_threshold', 0):>3} / "
            f"both {diagnose.get('n_pass_both', 0)}"
        )
        lines.append(
            f"score max {diagnose.get('score_max', 0):.4f} | "
            f"p99 {diagnose.get('score_p99', 0):.4f} | "
            f"p99.5 {diagnose.get('score_p99_5', 0):.4f}"
        )

    lines.append("")
    lines.append("📎 framing: BTC bull regime 에서 ≥20% tail pump 가능성이")
    lines.append("   과거 OOF 기준 최상위 0.05% 후보. 사용자 본인 판단.")
    return "\n".join(lines)


# ============================================================================
# Pre-open trigger (08:55 KST) — 09:00 직후 펌프 후보 알림
# ============================================================================
def format_preopen_beta(
    alerts: pd.DataFrame,
    btc_regime: str = "unknown",
    universe_label: str = "top100",
    asof: Optional[datetime] = None,
    dry_run: bool = False,
    demoted: bool = False,
) -> str:
    """Pre-open trigger 알림 (08:55 KST).

    inputs (alerts row 기대 컬럼):
      market, composite, close, expected_edge_pct, calibrated_hit_pct,
      source_rank, decision_reason

    demoted=True 면 채널 전체가 WATCH_ONLY 강등 상태 — 알림 형식을 그렇게 표시.
    """
    asof = asof or datetime.now()
    date_str = asof.strftime("%Y-%m-%d")
    header = f"⚡ pre-open trigger {date_str} (KST 08:50)"
    if dry_run:
        header += "  [DRY-RUN]"

    lines = [header]
    lines.append(f"BTC: {btc_regime_kr(btc_regime)} | universe: {universe_label}")
    lines.append("")

    if demoted:
        lines.append("━━━ DEMOTED (shadow only) ━━━")
        lines.append("preopen 채널 강등 — 모든 후보 shadow_ledger 만 누적")
        lines.append("(2026-05-26 사용자 컨펌 · 누적 -40.8%, replay active 0건)")
        lines.append("· 검증 누적 후 재평가 — distribution 09:05 알림 참고")
        return "\n".join(lines)

    if len(alerts) == 0:
        lines.append("━━━ 침묵 ━━━")
        if str(btc_regime).startswith("bear"):
            lines.append("(BTC bear regime — pre-open trigger 비활성)")
        else:
            lines.append("(09:00 직후 펌프 후보 없음)")
        return "\n".join(lines)

    lines.append(f"━━━ 09:00 직후 펌프 후보 {len(alerts)}건 ━━━")
    lines.append("")
    for _, r in alerts.iterrows():
        coin = str(r["market"]).replace("KRW-", "")
        price = r.get("close", float("nan"))
        tier = composite_tier(r.get("composite", 0))
        edge = _fmt_pctp(r.get("expected_edge_pct"))
        hit = _fmt_pct(r.get("calibrated_hit_pct"))
        rank = _fmt_rank(r.get("source_rank", r.get("alert_rank")))
        reason = str(r.get("decision_reason", "") or "").strip()
        signal_label = _hit_label(r.get("calibration_source", ""), "1h +5 signal")
        conf = _fmt_conf(r.get("confidence_score"), r.get("confidence_tier"))
        conf_text = f" | {conf}" if conf else ""
        lines.append(f"{tier} {coin}  진입가 ≈ {_fmt_krw_price(price)}  [PREOPEN | rank {rank}]")
        lines.append(f"   ▸ edge {edge}{conf_text} | {signal_label} {hit}")
        if reason:
            lines.append(f"   ▸ policy: {reason}")
        lines.append("")

    lines.append("━━━ 사용 ━━━")
    lines.append("• 09:00 직후 진입, 5% 오르면 즉시 매도")
    lines.append("• 텔레그램은 ACTIVE만 발송, WATCH/SILENCE는 dashboard·ledger에 기록")
    lines.append("• 자동 실거래 주문 없음 — 사용자가 직접 판단")
    lines.append("• 09:05 distribution 알림과 함께 확인")
    return "\n".join(lines)


# ============================================================================
# Distribution Engine v1 (09:05 KST) — multi-head + setup 후보 알림
# ============================================================================
def format_distribution_beta(
    alerts: pd.DataFrame,
    btc_regime: str = "unknown",
    universe_label: str = "top100",
    universe_size: int = 0,
    asof: Optional[datetime] = None,
    dry_run: bool = False,
) -> str:
    """Distribution Engine beta 알림 (09:05 KST).

    inputs (alerts row 기대 컬럼):
      market, primary_setups, btc_context, setup_quality, expected_edge_pct,
      calibrated_hit_pct, source_rank, decision_reason, composite, close

    톤은 pre-open 과 통일: ⚡ 헤더, BTC 한글 라벨, composite tier (🔥/✨/👀),
    코인당 본문 3줄, 사용 가이드. 진단/Setup library 블록은 텔레그램에서 제거
    (호출부의 stdout/log JSON 이 동일 정보 보존).
    """
    asof = asof or datetime.now()
    date_str = asof.strftime("%Y-%m-%d")
    header = f"⚡ distribution {date_str} (KST 09:05)"
    if dry_run:
        header += "  [DRY-RUN]"

    lines = [header]
    lines.append(
        f"BTC: {btc_regime_kr(btc_regime)} | universe: {universe_label} ({universe_size})"
    )
    lines.append("")

    if len(alerts) == 0:
        lines.append("━━━ 침묵 ━━━")
        if str(btc_regime).startswith("bear"):
            lines.append("(BTC bear regime — distribution head 약세)")
        else:
            lines.append("(setup S01/S02/S03 fire 한 후보 없음)")
        return "\n".join(lines)

    lines.append(f"━━━ 상승 setup 후보 {len(alerts)}건 ━━━")
    lines.append("")
    for _, r in alerts.iterrows():
        coin = str(r["market"]).replace("KRW-", "")
        primary = r.get("primary_setups") or []
        if not isinstance(primary, list):
            primary = list(primary) if primary is not None else []
        setup_str = "+".join(primary) if primary else "—"
        ctx = r.get("btc_context", "—")
        ctx_suffix = f" / {ctx}" if ctx and ctx != "—" else ""

        price = r.get("close", float("nan"))
        tier = composite_tier(r.get("composite", 0))
        setup_quality = str(r.get("setup_quality", "") or "SETUP")
        edge = _fmt_pctp(r.get("expected_edge_pct"))
        hit = _fmt_pct(r.get("calibrated_hit_pct"))
        rank = _fmt_rank(r.get("source_rank", r.get("alert_rank")))
        reason = str(r.get("decision_reason", "") or "").strip()
        signal_label = _hit_label(r.get("calibration_source", ""), "h6 signal")
        conf = _fmt_conf(r.get("confidence_score"), r.get("confidence_tier"))
        conf_text = f" | {conf}" if conf else ""

        lines.append(
            f"{tier} {coin}  진입가 ≈ {_fmt_krw_price(price)}  [{setup_quality} | rank {rank}]"
        )
        lines.append(f"   ▸ edge {edge}{conf_text} | {signal_label} {hit} | setup {setup_str}{ctx_suffix}")
        if reason:
            lines.append(f"   ▸ policy: {reason}")
        lines.append("")

    lines.append("━━━ 사용 ━━━")
    lines.append("• 09:00 직후 또는 첫 4h 안 진입")
    lines.append("• 5% 오르면 즉시 매도, 자동 실거래 주문 없음")
    lines.append("• 텔레그램은 ACTIVE만 발송, WATCH/SILENCE는 dashboard·ledger에 기록")
    lines.append("• 08:50 pre-open 채널은 DEMOTED (shadow only) — distribution 만 신뢰")
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

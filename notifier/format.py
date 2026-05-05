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
# Distribution Engine v1 — multi-head + setup library 알림 (사용자 신규 방향)
# ============================================================================
def _bucket_hit(calibrator, head_id: str, score: float) -> tuple[float, int, int] | None:
    """Return calibrated hit pct + human decile if available."""
    if calibrator is None or not calibrator.is_loaded(head_id):
        return None
    info = calibrator.lookup(head_id, score)
    if info is None:
        return None
    return (
        float(info["actual_hit_pct"]),
        int(info["bucket"]) + 1,
        int(info["n_buckets_total"]),
    )


def _distribution_decision_line(calibrator, h2: float, h6: float, h5: float) -> str:
    """사용자 판단용 one-line contract.

    Backtest 결과상 distribution beta 는 TP3/TP5 에서 baseline 을 크게 이겼고,
    TP20 은 tail 정보로만 쓰는 게 더 정직하다. 그래서 알림도 h2/h6 를
    판단 중심으로 두고 h5 는 rare tail bonus 로 표시한다.
    """
    b2 = _bucket_hit(calibrator, "h2", h2)
    b6 = _bucket_hit(calibrator, "h6", h6)
    b5 = _bucket_hit(calibrator, "h5", h5)
    if not (b2 and b6 and b5):
        return "   판단  calibration unavailable — 판단 tier 숨김"

    tags = []
    if b2[0] >= 50 or b2[1] >= b2[2] - 1:
        tags.append("즉발 +3%")
    if b6[0] >= 50 or b6[1] >= b6[2] - 1:
        tags.append("24h +5%")
    if b5[0] >= 10 or b5[1] >= b5[2]:
        tags.append("+20% tail")
    if not tags:
        tags.append("관찰")

    # h5 는 base rate 가 낮아서 실제 hit 이 두 자리여도 rare event 로 취급한다.
    tail_note = "tail bonus" if "+20% tail" in tags else "tail 낮음"
    return f"   판단  {' / '.join(tags)}  |  우선: TP3/TP5, {tail_note}"


def format_distribution_beta(
    alerts: pd.DataFrame,
    diagnose: Optional[dict] = None,
    btc_regime: str = "unknown",
    universe_size: int = 0,
    universe_label: str = "top100",
    asof: Optional[datetime] = None,
    dry_run: bool = False,
    calibrator=None,  # signals.bucket_calibration.BucketCalibrator
    show_raw: bool = False,
) -> str:
    """Distribution Engine beta 알림.

    framing: 매수 추천 X. 상승 setup 후보 + 분포 + 실패 위험.
    구조 (사용자 prototype):
      - setup IDs (S01/S02/S03)
      - 확률 분포 (h2/h6/h5)
      - 위험 (실패 평균 EOD, setup worst5 fail, sustain head 미검증)
      - 근거 (전일 상승률, ATR, return rank, 유사 사례 수)
    """
    asof = asof or datetime.now()
    date_str = asof.strftime("%Y-%m-%d")
    header = f"🌅 prelude distribution beta {date_str} (KST 09:05)"
    if dry_run:
        header += "  [DRY-RUN]"

    lines = [header]
    lines.append(f"BTC: {btc_regime} | universe: {universe_label} ({universe_size})")
    lines.append("")

    if len(alerts) == 0:
        lines.append("━━━ 침묵 ━━━")
        if btc_regime.startswith("bear"):
            lines.append("(BTC bear regime — distribution head 약세)")
        else:
            lines.append("(setup S01/S02/S03 fire 한 후보 없음)")
        if diagnose:
            lines.append("")
            lines.append("━━━ 진단 ━━━")
            sc = diagnose.get("setup_fire_counts", {})
            lines.append(f"setup fire — S01:{sc.get('S01',0)} S02:{sc.get('S02',0)} "
                         f"S03:{sc.get('S03',0)} S04:{sc.get('S04',0)}")
            lines.append(
                f"p99 — h2:{diagnose.get('p_h2_p99_pct',0):.1f}% "
                f"h5:{diagnose.get('p_h5_p99_pct',0):.1f}% "
                f"h6:{diagnose.get('p_h6_p99_pct',0):.1f}%"
            )
        return "\n".join(lines)

    lines.append(f"━━━ 상승 setup 후보 ({len(alerts)}건) ━━━")
    lines.append("※ 매수 추천 아님 / 표시 = OOF backfill 기반 historical hit (raw probability X)")

    calib_ready = calibrator is not None and all(calibrator.is_loaded(h) for h in ["h2", "h5", "h6"])
    if not calib_ready:
        lines.append("⚠️  calibrator 미로드 — distribution score 숨김 (raw probability 표시 금지)")

    for _, row in alerts.iterrows():
        coin = row["market"].replace("KRW-", "")
        primary = row.get("primary_setups", [])
        ctx = row.get("btc_context", "—")
        setup_str = "+".join(primary) if primary else "—"
        ctx_suffix = f" / {ctx}" if ctx != "—" else ""

        s_h2 = float(row.get("p_h2_hit_3_4h", 0))
        s_h5 = float(row.get("p_h5_tail_20", 0))
        s_h6 = float(row.get("p_h6_hit_5_24h", 0))

        # evidence (raw values)
        atr = row.get("atr_pct_14", 0) * 100
        log_ret = row.get("log_return_1d", 0) * 100
        ret5_rank = row.get("return_5d", 0)
        ret7_rank = row.get("return_7d", 0)
        vol5_rank = row.get("vol_5d", 0)

        entry_price = _fmt_krw_price(row.get("close", float("nan")))

        lines.append("")
        lines.append(f"🔍 {coin}  진입가 ≈ {entry_price}  Setup: {setup_str}{ctx_suffix}")
        if calib_ready:
            lines.append(_distribution_decision_line(calibrator, s_h2, s_h6, s_h5))
            lines.append(f"   +3% in 4h   : {calibrator.format_for_alert('h2', s_h2, show_raw=show_raw)}")
            lines.append(f"   +5% in 24h  : {calibrator.format_for_alert('h6', s_h6, show_raw=show_raw)}")
            lines.append(f"   +20% tail   : {calibrator.format_for_alert('h5', s_h5, show_raw=show_raw)}")
        else:
            lines.append("   분포  calibration unavailable — raw score hidden")
        lines.append(
            f"   근거  전일 +{log_ret:.2f}% / ATR14 {atr:.2f}% / "
            f"vol5 rank {vol5_rank:.2f} / return5 rank {ret5_rank:.2f} / return7 rank {ret7_rank:.2f}"
        )

    lines.append("")
    lines.append("━━━ Setup library v1 (참고) ━━━")
    lines.append("S01 high-vol momentum   — 과거 h2 lift 4.09×, h6 lift 2.63×")
    lines.append("S02 strong yesterday    — 과거 h5 lift 4.07× (depth=1)")
    lines.append("S03 vol expansion 5d    — 과거 h5 lift 6.81×")
    lines.append("S04 BTC bull context    — context only (setup 아님)")
    lines.append("")
    lines.append("⚠️  종가 유지 head (h3/h4) 는 daily features 한계로 미검증")
    lines.append("⚠️  실거래는 사용자 본인 판단. 가상 paper ledger 만 자동 누적")

    if diagnose:
        lines.append("")
        lines.append("━━━ 진단 ━━━")
        sc = diagnose.get("setup_fire_counts", {})
        lines.append(
            f"universe {diagnose.get('n_universe_filtered',0)} → "
            f"setup fire S01:{sc.get('S01',0)} S02:{sc.get('S02',0)} "
            f"S03:{sc.get('S03',0)} (any primary: {diagnose.get('n_with_any_primary_setup',0)})"
        )
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

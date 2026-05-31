"""SHADOW 추천 스캐너 — 텔레그램 risk-reward 레이더 발송 (매일 2회: 08:50 / 09:05).

흐름:
  1. signals.recommend.score_candidates(asof) 호출 → leak-free top-3
     + 멀티임계 calibrated 확률 (P(≥5/10/20%) / P(≤-5/-10%) / E[하방]).
     최종 정렬 = R1 risk-reward (downside-first): rr_ratio = P(≥10%)/max(P(≤-5%),eps)
     내림차순. score(rank-mean) 는 후보 추리기·부가표시일 뿐 정렬키가 아니다.
  2. risk-reward 레이더 메시지로 포맷 (코인 | 상방확률 | 하방확률 | E[하방] + dump_risk⚠️
     + "자동매매X·본인판단·검증중" + "-3% SL / +5% TP" 가이드).
  3. notifier.telegram.send_telegram 으로 발송.

★★★ 이 채널은 SHADOW(검증중) 다 (CLAUDE.md §2.2/§3.1, ops-steward §0):
    - 자동주문·업비트 API key 절대 없음. 사람이 보고 본인 판단으로 매매.
    - 진입/SL/TP 는 shadow 가상평가용 플랜값일 뿐.
    - score_candidates 가 시그널 계산 전담 (이 스크립트는 포맷+발송만 = notifier 책임).
    - 기록(ledger append)은 scripts/recommend_today.py 책임 — 여기서는 발송만.

★ 정렬 = R1 risk-reward(de-corr 하락 head, downside-first): rr_ratio = P(≥10%)/max(P(≤-5%),eps)
  내림차순으로 top-K 를 고른다. P-prob 은 downside_head_riskreward_v1 의 de-correlated
  XGB head(코인별 D-1 feature 로 따로 학습)에서 산출 → 저-하방 분리.
  score(equal-weight rank-mean + train-only bucket calibration)는 후보 추리기·부가표시용
  부가값일 뿐, 최종 정렬키가 아니다. rank-mean 의 상·하방 대칭 saturate 함정을 R1 이 제거.

확률 정직성 (이 프로젝트 전적: +20% tail 90%→실제 11.6% 과신 사고):
  - p_up20 == pump_prob (둘 다 pump20 calibrated, top bin ~6~10%).
  - 모든 확률은 calibrated bucket-hit (raw softmax 아님). 헤드라인 90% 류 절대 없음.

cron/systemd timer 가 발사한다 (이 스크립트는 등록 X). 수동 스모크:
    python scripts/recommend_send.py --asof 2026-05-31 --dry-run   # 발송 X, 메시지 출력만
    python scripts/recommend_send.py                               # 오늘(KST), 실제 발송(타이머용)
    python scripts/recommend_send.py --slot preopen                # 08:50 슬롯 헤더
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signals.recommend import score_candidates  # noqa: E402
from notifier.telegram import send_telegram  # noqa: E402

KST = timezone(timedelta(hours=9))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("recommend_send")

# 슬롯별 헤더 시각 — cron 실제 발사 시각과 일치 (ops-steward §3: 08:50 / 09:05).
SLOT_TIME = {"preopen": "08:50 (개장 직전)", "open": "09:05 (개장 후)"}

_BEAR = {"bear", "bear_quiet", "bear_volatile"}


def _today_kst() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d")


def _regime_kr(regime: str) -> str:
    """notifier.format.btc_regime_kr 와 톤 통일 (self-contained 최소 매핑)."""
    m = {
        "bull": "강세", "bull_quiet": "강세(저변동)", "bull_volatile": "강세(고변동)",
        "bear": "약세", "bear_quiet": "약세(저변동)", "bear_volatile": "약세(고변동)",
        "neutral": "중립", "unknown": "—",
    }
    return m.get(str(regime), str(regime))


def _pct(x) -> str:
    """0~1 확률 → 정직 % 표기. None/NaN 은 '—'."""
    try:
        v = float(x)
        if v != v:  # NaN
            return "—"
        return f"{v * 100:.0f}%"
    except (TypeError, ValueError):
        return "—"


def _signed_pct(x) -> str:
    """E[하방] 같은 부호 있는 비율 → +/- % 표기."""
    try:
        v = float(x)
        if v != v:
            return "—"
        return f"{v * 100:+.1f}%"
    except (TypeError, ValueError):
        return "—"


# ==========================================================================
# risk-reward 레이더 메시지 포맷 (notifier 책임 — 알림 포맷 변경은 사용자 컨펌 게이트)
# ==========================================================================
def format_radar(res: dict, slot: str, *, dry_run: bool = False) -> str:
    asof = res.get("asof", "")
    regime = res.get("btc_regime", "unknown")
    slot_label = SLOT_TIME.get(slot, slot)
    header = f"🛰️ 추천 레이더 {asof} (KST {slot_label})"
    if dry_run:
        header += "  [DRY-RUN]"

    lines = [header]
    lines.append(
        f"BTC: {_regime_kr(regime)} | universe top100 ({res.get('universe_n', 0)})"
        f" | R1 risk-reward · downside-first · SHADOW(검증중)"
    )
    lines.append("")

    top3 = res.get("top3") or []
    if not top3:
        lines.append("━━━ 후보 없음 ━━━")
        if str(regime) in _BEAR:
            lines.append("(BTC 약세 regime — 추천 후보 없음, 정상 침묵)")
        else:
            lines.append("(유니버스 내 스코어 후보 없음)")
        return "\n".join(lines)

    lines.append(f"━━━ risk-reward 레이더 top{len(top3)} ━━━")
    lines.append("(상방=고가가 그만큼 갈 확률 / 하방=저가가 그만큼 빠질 확률, 둘 다 검증된 calibrated)")
    lines.append("")
    for it in top3:
        coin = str(it.get("coin", "")).replace("KRW-", "")
        warn = " ⚠️dump_risk" if it.get("dump_risk_flag") else ""
        entry = it.get("entry_open")
        entry_str = f"{entry:g}" if entry is not None else "—"
        lines.append(f"#{it.get('rank', '?')} {coin}  진입 ≈ {entry_str}{warn}")
        lines.append(
            f"   ▸ 상방  ≥5% {_pct(it.get('p_up5'))} · ≥10% {_pct(it.get('p_up10'))}"
            f" · ≥20% {_pct(it.get('p_up20'))}"
        )
        lines.append(
            f"   ▸ 하방  ≤-5% {_pct(it.get('p_dn5'))} · ≤-10% {_pct(it.get('p_dn10'))}"
            f" · E[하방] {_signed_pct(it.get('exp_downside'))}"
        )
        lines.append("")

    lines.append("━━━ 사용 ━━━")
    lines.append("• 자동매매 없음 — 알림만. 본인 판단으로 직접 매매")
    lines.append("• 가이드: 진입 09:00 open, -3% 손절(SL) / +5% 익절(TP)")
    lines.append("• 검증중(SHADOW) — 가상 ledger·dashboard 로 성과 추적, 실거래 주문 X")
    lines.append("• ⚠️dump_risk = 과열·고유동 board-top — 하방 클 수 있으니 사이즈 축소")
    return "\n".join(lines)


# ==========================================================================
# 발송
# ==========================================================================
def send_recommendation(asof: str, slot: str, *, dry_run: bool = False,
                        limit_markets: int | None = None) -> bool:
    res = score_candidates(asof, limit_markets=limit_markets)
    log.info("asof=%s slot=%s btc_regime=%s universe_n=%d calib=%s top=%d",
             res["asof"], slot, res["btc_regime"], res["universe_n"],
             res["calibration_source"], len(res["top3"]))
    msg = format_radar(res, slot, dry_run=dry_run)
    ok = send_telegram(msg, dry_run=dry_run)
    if not dry_run:
        log.info("telegram send %s", "OK" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser(
        description="SHADOW 추천 레이더 텔레그램 발송 (08:50 / 09:05)")
    ap.add_argument("--asof", type=str, default=None, help="YYYY-MM-DD (default=오늘 KST)")
    ap.add_argument("--slot", type=str, default="open", choices=["preopen", "open"],
                    help="발송 슬롯 (preopen=08:50 / open=09:05). 헤더 시각만 결정")
    ap.add_argument("--dry-run", action="store_true",
                    help="발송 X, 포맷된 메시지만 stdout 출력 (스모크)")
    ap.add_argument("--limit-markets", type=int, default=None, help="개발용 마켓 제한")
    args = ap.parse_args()

    asof = args.asof or _today_kst()
    ok = send_recommendation(asof, args.slot, dry_run=args.dry_run,
                             limit_markets=args.limit_markets)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

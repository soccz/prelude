"""SHADOW 추천 레이더 — champion-aware 텔레그램 발송 (매일 2회: 08:50 / 09:05).

★ champion/challenger 배선 (Stage 2): 이 dispatcher 는 *어느 모델로 발송할지* 를
  하드코딩하지 않는다. ops.champion_selector.get_champion(slot) 으로 slot 별 현 챔피언을
  읽고 → signals.model_registry.get_model(champion_id).predict_ref 를 동적 import 해서
  호출한다. 현재 챔피언 = recommend_r1_open (open·preopen 둘 다 fallback) 이라
  predict_ref = "signals.recommend:score_candidates" → score_candidates(asof, slot=slot)
  를 호출(= 기존 R1 레이더 동작 그대로). 새 챔피언이 선정되면 코드 변경 없이 그쪽 predict 가 불린다.

흐름:
  1. get_champion(slot) → champion_id + is_fallback + reason.
  2. get_model(champion_id).predict_ref 를 import → score_candidates(asof, slot=slot) 호출.
     leak-free top-3 + 멀티임계 calibrated 확률 (P(≥5/10/20%) / P(≤-5/-10%) / E[하방]).
  3. risk-reward 레이더 메시지로 포맷 (헤더에 champion_id + fallback 이면 "SHADOW fallback").
     pre-open slot 은 진입가 미확정 → "진입가 09:00 open(개장 후 확정)" 표기.
  4. notifier.telegram.send_telegram 으로 발송.

★★★ 이 채널은 SHADOW(검증중) 다 (CLAUDE.md §2.2/§3.1, ops-steward §0):
    - 자동주문·업비트 API key 절대 없음. 사람이 보고 본인 판단으로 매매.
    - 진입/SL/TP 는 shadow 가상평가용 플랜값일 뿐.
    - predict 함수가 시그널 계산 전담 (이 스크립트는 dispatch+포맷+발송만 = ops 책임).
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
    python scripts/recommend_send.py --asof 2026-05-31 --dry-run            # open slot, 발송 X
    python scripts/recommend_send.py --slot preopen --asof 2026-06-01 --dry-run  # 08:50 slot
    python scripts/recommend_send.py                                        # 오늘(KST) 실발송(타이머용)
"""
from __future__ import annotations

import argparse
import importlib
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from notifier.telegram import send_telegram  # noqa: E402
from ops.champion_selector import get_champion  # noqa: E402
from signals.model_registry import ModelSpec, fallback_model, get_model  # noqa: E402

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
# champion 해소 + predict_ref 동적 호출 (champion-aware dispatcher)
# ==========================================================================
def resolve_champion(slot: str) -> tuple[ModelSpec, bool, str]:
    """slot 의 현 챔피언 ModelSpec + is_fallback + reason 반환.

    champion_state.json 이 없거나 champion_id 미설정이면 model_registry 의 백테스트-최선
    fallback 으로 graceful 처리(항상 무언가 발송, SHADOW)."""
    entry = get_champion(slot)
    if entry and entry.get("champion_id"):
        spec = get_model(entry["champion_id"])
        return spec, bool(entry.get("is_fallback", False)), str(entry.get("reason", ""))
    # champion_state.json 부재/손상 → 레지스트리 fallback (R1).
    spec = fallback_model(slot)
    if spec is None:
        raise RuntimeError(f"slot={slot}: champion 도 fallback 모델도 없음")
    return spec, True, "champion_state.json 부재/미설정 → 레지스트리 fallback"


def call_predict(spec: ModelSpec, asof: str, slot: str,
                 limit_markets: int | None) -> dict:
    """spec.predict_ref ('module:obj' 또는 'module:Class.method') 를 동적 import 해서 호출.

    현 챔피언 R1 의 predict_ref = 'signals.recommend:score_candidates' →
    score_candidates(asof, limit_markets=..., slot=slot). 반환 dict 는 format_radar 가 소비.
    (distribution_engine 류 다른 시그니처 모델이 챔피언이 되면 그 모델 전용 어댑터를
    여기 추가한다 — 지금은 R1 만 라이브 챔피언이라 score_candidates 경로만.)"""
    mod_name, _, attr_path = spec.predict_ref.partition(":")
    mod = importlib.import_module(mod_name)
    obj = mod
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    # score_candidates(asof, limit_markets, slot) 시그니처 (현 라이브 챔피언).
    if spec.predict_ref == "signals.recommend:score_candidates":
        return obj(asof, limit_markets=limit_markets, slot=slot)
    raise NotImplementedError(
        f"predict_ref={spec.predict_ref} 발송 어댑터 미구현 — "
        f"이 모델이 라이브 챔피언이 되면 call_predict 에 어댑터 추가 필요")


# ==========================================================================
# risk-reward 레이더 메시지 포맷 (notifier 책임 — 알림 포맷 변경은 사용자 컨펌 게이트)
# ==========================================================================
def format_radar(res: dict, slot: str, *, dry_run: bool = False,
                 champion_id: str = "", is_fallback: bool = False) -> str:
    asof = res.get("asof", "")
    regime = res.get("btc_regime", "unknown")
    slot_label = SLOT_TIME.get(slot, slot)
    header = f"🛰️ 추천 레이더 {asof} (KST {slot_label})"
    if dry_run:
        header += "  [DRY-RUN]"

    # champion 표기 — champion_id + (fallback 이면) SHADOW fallback.
    champ_tag = f"champion: {champion_id}" if champion_id else "champion: —"
    if is_fallback:
        champ_tag += " · SHADOW fallback(백테스트-최선)"

    lines = [header]
    lines.append(champ_tag)
    lines.append(
        f"BTC: {_regime_kr(regime)} | universe top100 ({res.get('universe_n', 0)})"
        f" | risk-reward · downside-first · SHADOW(검증중)"
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

    # 진입가 표기: open=실제 09:00 open. preopen=미개장이라 미확정.
    resolved_slot = str(res.get("slot", slot))
    is_preopen = resolved_slot == "preopen"

    lines.append(f"━━━ risk-reward 레이더 top{len(top3)} ━━━")
    lines.append("(상방=고가가 그만큼 갈 확률 / 하방=저가가 그만큼 빠질 확률, 둘 다 검증된 calibrated)")
    lines.append("")
    for it in top3:
        coin = str(it.get("coin", "")).replace("KRW-", "")
        warn = " ⚠️dump_risk" if it.get("dump_risk_flag") else ""
        entry = it.get("entry_open")
        if is_preopen or entry is None:
            # pre-open(08:50): 09:00 미개장 → 진입가 미확정. None 을 "—" 로 보이지 않게.
            entry_line = "진입가 09:00 open(개장 후 확정)"
        else:
            entry_line = f"진입 ≈ {entry:g}"
        lines.append(f"#{it.get('rank', '?')} {coin}  {entry_line}{warn}")
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
# 챔피언 교체 통보 (task 2) — 막지 않고 알리기만.
# ==========================================================================
def maybe_notify_champion_change(slot: str, *, dry_run: bool = False) -> bool | None:
    """champion_state.json history 의 *마지막 교체 이벤트* 가 이 slot 에서 from!=to(실제 교체)
    면 1줄 통보 발송. 부팅(from=None)·교체 없음·이미 통보됨 케이스는 발송 안 함.

    멱등성: history 의 가장 최근 (asof, slot) from!=to 이벤트만 보고, 그게 champion_state 의
    asof 와 같을 때만 '오늘 교체'로 간주(매일 1회 close 후 셀렉터가 갱신). 발송 여부만 반환.
    ★ 통보는 알림일 뿐 — 발송을 막지 않는다(레이더는 별도로 나간다)."""
    state_path = Path(__file__).resolve().parent.parent / "output" / "champion_state.json"
    if not state_path.exists():
        return None
    import json
    try:
        with open(state_path) as f:
            st = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    asof = st.get("asof")
    # 이 slot 의, 오늘 asof 에 발생한, 실제 교체(from!=to & from is not None) 이벤트.
    evts = [e for e in st.get("history", [])
            if e.get("slot") == slot and e.get("asof") == asof
            and e.get("from") is not None and e.get("from") != e.get("to")]
    if not evts:
        return None
    e = evts[-1]
    msg = (f"ℹ️ 챔피언 {slot}: {e['from']} → {e['to']} "
           f"(사유: {str(e.get('reason', '')).strip()})")
    log.info("champion change notify: %s", msg)
    return send_telegram(msg, dry_run=dry_run)


# ==========================================================================
# 발송 (champion-aware)
# ==========================================================================
def send_recommendation(asof: str, slot: str, *, dry_run: bool = False,
                        limit_markets: int | None = None) -> bool:
    spec, is_fallback, reason = resolve_champion(slot)
    log.info("slot=%s champion=%s is_fallback=%s reason=%s",
             slot, spec.id, is_fallback, reason)

    # 챔피언 교체 통보 (있으면 1줄, 막지 않고 알리기만 — 지금은 교체 없음=통보 없음).
    maybe_notify_champion_change(slot, dry_run=dry_run)

    res = call_predict(spec, asof, slot, limit_markets)
    log.info("asof=%s slot=%s(resolved=%s) btc_regime=%s universe_n=%d calib=%s top=%d",
             res["asof"], slot, res.get("slot", slot), res["btc_regime"],
             res["universe_n"], res["calibration_source"], len(res["top3"]))
    msg = format_radar(res, slot, dry_run=dry_run,
                       champion_id=spec.id, is_fallback=is_fallback)
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

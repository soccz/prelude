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
     leak-free top-3 + 멀티임계 historical/resubstitution 추정치
     (P(≥5/10/20%) / P(≤-5/-10%) / E[하방]); strict calibrated 확률은 아님.
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
  - p_up20 == pump_prob (둘 다 train-only pump20 score-bucket historical estimate).
  - head와 bucket 모두 asof 이후를 보지는 않지만 inner OOF/strict forward 보정이
    아니므로 calibrated라고 주장할 수 없다. snapshot v2는 확률 중첩 위반을 차단한다.

cron/systemd timer 가 발사한다 (이 스크립트는 등록 X). 수동 스모크:
    python scripts/recommend_send.py --asof 2026-05-31 --dry-run            # open slot, 발송 X
    python scripts/recommend_send.py --slot preopen --asof 2026-06-01 --dry-run  # 08:50 slot
    python scripts/recommend_send.py                                        # 오늘(KST) 실발송(타이머용)
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import sys
from contextlib import contextmanager
from datetime import date, datetime, time, timezone, timedelta
from pathlib import Path
from typing import Callable, Iterator, cast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from notifier.delivery_receipt import (  # noqa: E402
    RECEIPT_INTEGRITY_ACTIVATION_DATE,
    read_delivery_receipt,
    receipt_path,
    write_delivery_receipt,
)
from notifier.telegram import (  # noqa: E402
    TelegramSendResult,
    send_telegram,
    send_telegram_with_receipt,
    telegram_error_is_ambiguous,
)
from ops.champion_selector import (  # noqa: E402
    ChampionStateError,
    get_champion,
    load_champion_state_artifact,
)
from ops.file_lock import file_lock  # noqa: E402
from signals.model_registry import ModelSpec, get_model  # noqa: E402
from signals.recommend_snapshot import get_or_create_recommend_snapshot  # noqa: E402
from signals.recommend_snapshot import SNAPSHOT_SCHEMA_VERSION  # noqa: E402

KST = timezone(timedelta(hours=9))

log = logging.getLogger("recommend_send")

# 슬롯별 헤더 시각 — cron 실제 발사 시각과 일치 (ops-steward §3: 08:50 / 09:05).
SLOT_TIME = {"preopen": "08:50 (개장 직전)", "open": "09:05 (개장 후)"}
LIVE_SEND_WINDOWS = {
    "preopen": (time(8, 45), time(9, 0)),
    "open": (time(9, 0), time(9, 21)),
}
APPROVED_LIVE_R1_RANK_BASIS = "R1_riskreward(de-corr head)"
APPROVED_LIVE_SCORE_SCHEMA = "recommend_score.v2"
APPROVED_LIVE_R1_RULE_VERSION = "r1_riskreward_v1"
CHAMPION_STATE_PATH = (
    Path(__file__).resolve().parent.parent / "output" / "champion_state.json"
)
_BEAR = {"bear", "bear_quiet", "bear_volatile"}


def _today_kst() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d")


def _now_kst() -> datetime:
    return datetime.now(KST)


def _assert_live_send_window(
    asof: str,
    slot: str,
    *,
    now: datetime | None = None,
) -> None:
    """Reject stale/wrong-slot real sends before scoring or side effects."""
    if slot not in LIVE_SEND_WINDOWS:
        raise ValueError(f"unsupported live send slot: {slot!r}")
    try:
        decision_day = date.fromisoformat(asof)
    except ValueError as exc:
        raise ValueError(f"invalid live send asof: {asof!r}") from exc
    observed = now or _now_kst()
    if observed.tzinfo is None:
        raise ValueError("live send clock must be timezone-aware")
    observed = observed.astimezone(KST)
    if decision_day != observed.date():
        raise RuntimeError(
            f"stale live send rejected: asof={decision_day} "
            f"today_kst={observed.date()}"
        )
    start, end = LIVE_SEND_WINDOWS[slot]
    wall_time = observed.timetz().replace(tzinfo=None)
    if not start <= wall_time < end:
        raise RuntimeError(
            f"outside {slot} live send window: "
            f"{wall_time.isoformat(timespec='seconds')} not in "
            f"[{start.isoformat(timespec='minutes')},"
            f"{end.isoformat(timespec='minutes')}) KST"
        )


def _live_send_deadline(asof: str, slot: str) -> datetime:
    """해당 decision day/slot의 배타적 종료 시각."""
    if slot not in LIVE_SEND_WINDOWS:
        raise ValueError(f"unsupported live send slot: {slot!r}")
    try:
        decision_day = date.fromisoformat(asof)
    except ValueError as exc:
        raise ValueError(f"invalid live send asof: {asof!r}") from exc
    return datetime.combine(
        decision_day,
        LIVE_SEND_WINDOWS[slot][1],
        tzinfo=KST,
    )


def _send_live_transport(
    message: str,
    *,
    asof: str,
    slot: str,
) -> tuple[bool, str | None, str | None, TelegramSendResult | None]:
    """Activation-aware Telegram adapter preserving the legacy bool seam."""
    deadline = _live_send_deadline(asof, slot)
    if date.fromisoformat(asof) < RECEIPT_INTEGRITY_ACTIVATION_DATE:
        ok = bool(
            send_telegram(
                message,
                deadline=deadline,
                clock=_now_kst,
            )
        )
        return (
            ok,
            None if ok else "send_telegram returned false",
            datetime.now(timezone.utc).isoformat() if ok else None,
            None,
        )

    transport = send_telegram_with_receipt(
        message,
        deadline=deadline,
        clock=_now_kst,
    )
    if transport.dry_run:
        raise RuntimeError("live Telegram transport unexpectedly used dry-run")
    sent_at = None
    if transport.delivery_ok:
        if not transport.telegram_messages:
            raise RuntimeError(
                "Telegram success omitted server delivery evidence"
            )
        sent_at = max(
            datetime.fromisoformat(item.server_date)
            for item in transport.telegram_messages
        ).isoformat()
    return (
        transport.delivery_ok,
        transport.error,
        sent_at,
        transport,
    )


def _assert_approved_live_ranking(
    res: dict,
    *,
    now: datetime | None = None,
) -> None:
    """Reject a diagnostic/degraded R1 snapshot from the live channel.

    The scorer deliberately emits ``score_rank(fallback)`` evidence when its
    de-correlated risk/reward head cannot be trained.  Keeping that snapshot is
    useful for dry-run diagnosis, but it is not the approved Telegram policy.
    """
    model = res.get("model")
    model_ranking = model.get("ranking") if isinstance(model, dict) else None
    if res.get("ranking") != "R1" or model_ranking != "R1":
        raise RuntimeError(
            "unapproved live ranking identity: "
            f"top_level={res.get('ranking')!r} model={model_ranking!r}"
        )
    rank_basis = res.get("rank_basis")
    if rank_basis != APPROVED_LIVE_R1_RANK_BASIS:
        raise RuntimeError(
            "unapproved live rank basis: "
            f"expected={APPROVED_LIVE_R1_RANK_BASIS!r} "
            f"resolved={rank_basis!r}"
        )
    if res.get("snapshot_schema") != SNAPSHOT_SCHEMA_VERSION:
        raise RuntimeError(
            "legacy/unversioned snapshot cannot enter the live radar"
        )
    if (
        res.get("score_schema_version") != APPROVED_LIVE_SCORE_SCHEMA
        or res.get("rule_version") != APPROVED_LIVE_R1_RULE_VERSION
    ):
        raise RuntimeError("unapproved live scorer/rule provenance")
    asof = res.get("asof")
    slot = res.get("slot")
    request = res.get("request")
    if (
        not isinstance(asof, str)
        or slot not in LIVE_SEND_WINDOWS
        or not isinstance(request, dict)
        or request.get("asof") != asof
        or request.get("slot") != slot
        or request.get("ranking") != "R1"
        or request.get("limit_markets") is not None
    ):
        raise RuntimeError("live snapshot request provenance mismatch")

    decision_day = date.fromisoformat(asof)
    start_wall, end_wall = LIVE_SEND_WINDOWS[cast(str, slot)]
    started: datetime | None = None
    completed: datetime | None = None
    for field in ("decision_started_at", "decision_completed_at"):
        raw = res.get(field)
        if not isinstance(raw, str):
            raise RuntimeError(f"live snapshot missing {field}")
        try:
            observed = datetime.fromisoformat(raw)
        except ValueError as exc:
            raise RuntimeError(f"live snapshot has invalid {field}") from exc
        if observed.tzinfo is None:
            raise RuntimeError(f"live snapshot {field} must be timezone-aware")
        observed_kst = observed.astimezone(KST)
        wall = observed_kst.timetz().replace(tzinfo=None)
        if (
            observed_kst.date() != decision_day
            or not start_wall <= wall < end_wall
        ):
            raise RuntimeError(
                f"live snapshot {field} outside {slot} decision window"
            )
        if started is None:
            started = observed
        elif observed < started:
            raise RuntimeError("live snapshot decision timestamp chronology invalid")
        else:
            completed = observed
    if now is not None:
        if now.tzinfo is None:
            raise RuntimeError("live snapshot comparison clock must be aware")
        if completed is None or completed > now:
            raise RuntimeError(
                "live snapshot decision completion is future-dated"
            )


def _assert_live_snapshot_boundary(
    snapshot: dict,
    slot: str,
    observed: datetime,
) -> None:
    """Revalidate policy identity and chronology at the API boundary."""
    _assert_live_send_window(
        str(snapshot.get("asof", "")),
        slot,
        now=observed,
    )
    _assert_approved_live_ranking(snapshot, now=observed)


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
def resolve_champion(
    slot: str,
    *,
    expected_asof: str | None = None,
) -> tuple[ModelSpec, bool, str]:
    """slot 의 현 챔피언 ModelSpec + is_fallback + reason 반환.

    champion_state.json 부재·손상·stale·미설정은 모두 발송을 fail-closed한다.
    selector가 명시적으로 기록한 ``is_fallback`` champion만 허용한다."""
    entry = get_champion(slot, expected_asof=expected_asof)
    if entry and entry.get("champion_id"):
        spec = get_model(entry["champion_id"])
        return spec, bool(entry.get("is_fallback", False)), str(entry.get("reason", ""))
    raise RuntimeError(
        f"slot={slot}: canonical champion state/slot entry missing; "
        "live send refused"
    )


def call_predict(spec: ModelSpec, asof: str, slot: str,
                 limit_markets: int | None, *,
                 snapshot_root: str | Path | None = None,
                 persist_snapshot: bool = True) -> dict:
    """spec.predict_ref ('module:obj' 또는 'module:Class.method') 를 동적 import 해서 호출.

    현 챔피언 R1 의 predict_ref = 'signals.recommend:score_candidates' →
    score_candidates(asof, limit_markets=..., slot=slot). 반환 dict 는 format_radar 가 소비.
    (distribution_engine 류 다른 시그니처 모델이 챔피언이 되면 그 모델 전용 어댑터를
    여기 추가한다 — 지금은 R1 만 라이브 챔피언이라 score_candidates 경로만.)"""
    mod_name, _, attr_path = spec.predict_ref.partition(":")
    mod = importlib.import_module(mod_name)
    obj: object = mod
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    # score_candidates(asof, limit_markets, slot) 시그니처 (현 라이브 챔피언).
    if spec.predict_ref == "signals.recommend:score_candidates":
        if not persist_snapshot:
            result = cast(Callable[..., dict], obj)(
                asof,
                limit_markets=limit_markets,
                slot=slot,
                ranking="R1",
            )
            result = dict(result)
            model = result.get("model")
            model_identity = dict(model) if isinstance(model, dict) else {}
            model_identity["id"] = spec.id
            result["model"] = model_identity
            return result
        return get_or_create_recommend_snapshot(
            asof,
            slot=slot,
            ranking="R1",
            limit_markets=limit_markets,
            model_id=spec.id,
            root=snapshot_root,
            scorer=cast(Callable[..., dict], obj),
        )
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
def maybe_notify_champion_change(
    slot: str,
    *,
    dry_run: bool = False,
    receipt_root: str | Path | None = None,
    live_asof: str | None = None,
) -> bool | None:
    """champion_state.json history 의 *마지막 교체 이벤트* 가 이 slot 에서 from!=to(실제 교체)
    면 1줄 통보 발송. 부팅(from=None)·교체 없음·이미 통보됨 케이스는 발송 안 함.

    멱등성: history 의 가장 최근 (asof, slot) from!=to 이벤트만 보고, 그게 champion_state 의
    asof 와 같을 때만 유효 교체로 간주한다. close 후 갱신된 전일 state가 다음날
    레이더에 쓰이는 정상 순서를 허용하되, 그보다 오래되거나 미래인 state는 거부한다.
    발송 여부만 반환.
    ★ 통보는 알림일 뿐 — 발송을 막지 않는다(레이더는 별도로 나간다)."""
    state_path = CHAMPION_STATE_PATH
    try:
        artifact = load_champion_state_artifact(
            state_path,
            expected_asof=live_asof,
        )
    except (OSError, ChampionStateError, ValueError) as exc:
        log.error("champion change state invalid: %s", exc)
        return None
    if artifact is None:
        return None
    st = artifact.payload
    asof = st.get("asof")
    if not isinstance(asof, str) or not asof:
        log.error("champion change state has no validated asof")
        return None
    history = st.get("history")
    if not isinstance(history, list):
        return None
    # 이 slot 의, 오늘 asof 에 발생한, 실제 교체(from!=to & from is not None) 이벤트.
    evts = [
        e
        for e in history
        if isinstance(e, dict)
        and e.get("slot") == slot
        and e.get("asof") == asof
        and isinstance(e.get("from"), str)
        and bool(e["from"])
        and isinstance(e.get("to"), str)
        and bool(e["to"])
        and e["from"] != e["to"]
    ]
    if not evts:
        return None
    e = evts[-1]
    msg = (f"ℹ️ 챔피언 {slot}: {e['from']} → {e['to']} "
           f"(사유: {str(e.get('reason', '')).strip()})")
    log.info("champion change notify: %s", msg)
    if dry_run:
        return send_telegram(msg, dry_run=True)

    event_payload = {
        "asof": asof,
        "slot": slot,
        "from": e["from"],
        "to": e["to"],
        "reason": str(e.get("reason", "")).strip(),
    }
    event_digest = hashlib.sha256(
        json.dumps(
            event_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    event_snapshot = {
        "snapshot_id": f"champion-change-{event_digest[:20]}",
        "snapshot_path": str(state_path),
        # The receipt contract follows the actual delivery day. The immutable
        # source event day remains bound in event_digest/snapshot_id.
        "asof": str(live_asof or asof),
        "slot": slot,
        "model": {
            "id": str(e["to"]),
            # Include the event digest in the receipt filename so multiple
            # genuine changes on one date cannot alias one another.
            "ranking": f"champion_change_{event_digest[:12]}",
        },
    }
    with _exclusive_snapshot_send_lock(
        event_snapshot,
        receipt_root=receipt_root,
    ):
        existing = read_delivery_receipt(
            event_snapshot,
            root=receipt_root,
        )
        if existing is not None and existing["delivery_ok"]:
            return True
        if existing is not None and (
            existing.get("telegram_messages")
            or telegram_error_is_ambiguous(existing.get("error"))
        ):
            raise RuntimeError(
                "champion notice receipt records partial/ambiguous delivery; "
                "automatic retry would duplicate delivered chunks"
            )
        if live_asof is not None:
            _assert_live_send_window(
                cast(str, live_asof),
                slot,
                now=_now_kst(),
            )
        attempted_at = datetime.now(timezone.utc).isoformat()
        try:
            ok, error, sent_at, transport = _send_live_transport(
                msg,
                asof=live_asof or asof,
                slot=slot,
            )
        except Exception as exc:  # noqa: BLE001
            ok = False
            error = f"{type(exc).__name__}: {exc}"
            sent_at = None
            transport = None
        if (
            date.fromisoformat(live_asof or asof)
            >= RECEIPT_INTEGRITY_ACTIVATION_DATE
            and transport is None
        ):
            raise RuntimeError(
                "post-activation Telegram attempt produced no server evidence"
            )
        write_delivery_receipt(
            event_snapshot,
            delivery_ok=ok,
            attempted_at=attempted_at,
            sent_at=sent_at,
            error=error,
            telegram_result=transport,
            message=msg,
            root=receipt_root,
        )
        return ok


# ==========================================================================
# 발송 (champion-aware)
# ==========================================================================
@contextmanager
def _exclusive_snapshot_send_lock(
    snapshot: dict,
    *,
    receipt_root: str | Path | None = None,
) -> Iterator[None]:
    """동일 snapshot의 receipt 확인→API 호출→receipt 저장을 직렬화."""
    receipt = receipt_path(snapshot, root=receipt_root)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    lock_path = receipt.with_name(f".{receipt.name}.send.lock")
    with file_lock(lock_path):
        yield


def _send_and_record(
    snapshot: dict,
    message: str,
    *,
    slot: str,
    receipt_root: str | Path | None,
) -> bool:
    """성공 receipt가 없는 snapshot만 발송하고 결과를 동일 lock 안에서 기록."""
    _assert_live_snapshot_boundary(snapshot, slot, _now_kst())
    with _exclusive_snapshot_send_lock(snapshot, receipt_root=receipt_root):
        existing = read_delivery_receipt(snapshot, root=receipt_root)
        if existing is not None and existing["delivery_ok"]:
            log.info(
                "telegram send skipped: snapshot already delivered → %s",
                existing["receipt_path"],
            )
            try:
                maybe_notify_champion_change(
                    slot,
                    dry_run=False,
                    receipt_root=receipt_root,
                    live_asof=str(snapshot.get("asof", "")),
                )
            except Exception as exc:  # noqa: BLE001
                log.error("champion change notification failed: %s", exc)
            return True
        if existing is not None and (
            existing.get("telegram_messages")
            or telegram_error_is_ambiguous(existing.get("error"))
        ):
            raise RuntimeError(
                "recommendation receipt records partial/ambiguous delivery; "
                "automatic retry would duplicate delivered chunks"
            )

        # Scoring can cross a slot boundary. Recheck immediately before the
        # primary recommendation API call. The auxiliary champion notice is
        # deliberately attempted only after this primary result is durably
        # recorded, so it can never consume the remaining slot window first.
        _assert_live_snapshot_boundary(snapshot, slot, _now_kst())
        attempted_at = datetime.now(timezone.utc).isoformat()
        try:
            ok, error, sent_at, transport = _send_live_transport(
                message,
                asof=str(snapshot.get("asof", "")),
                slot=slot,
            )
        except Exception as exc:
            if (
                date.fromisoformat(str(snapshot.get("asof", "")))
                < RECEIPT_INTEGRITY_ACTIVATION_DATE
            ):
                write_delivery_receipt(
                    snapshot,
                    delivery_ok=False,
                    attempted_at=attempted_at,
                    sent_at=None,
                    error=type(exc).__name__,
                    root=receipt_root,
                )
            raise

        # Telegram Bot API에는 client idempotency key가 없다. API가 메시지를
        # 수락한 직후 receipt fsync 전에 SIGKILL/전원 장애가 나면 다음 실행이
        # 중복 발송할 수 있다. 이 lock+receipt는 완료된 실행 및 동시 실행의
        # 중복만 제거하며, 불가피한 crash window를 성공으로 추정하지 않는다.
        receipt = write_delivery_receipt(
            snapshot,
            delivery_ok=bool(ok),
            attempted_at=attempted_at,
            sent_at=sent_at,
            error=error,
            telegram_result=transport,
            message=message,
            root=receipt_root,
        )
        log.info("telegram send %s", "OK" if ok else "FAIL")
        log.info("delivery receipt → %s", receipt)
        if ok:
            try:
                maybe_notify_champion_change(
                    slot,
                    dry_run=False,
                    receipt_root=receipt_root,
                    live_asof=str(snapshot.get("asof", "")),
                )
            except Exception as exc:  # noqa: BLE001
                # Champion notice is auxiliary; its evidence or transport
                # failure cannot retroactively invalidate the primary radar.
                log.error("champion change notification failed: %s", exc)
        return ok


def send_recommendation(asof: str, slot: str, *, dry_run: bool = False,
                        limit_markets: int | None = None,
                        snapshot_root: str | Path | None = None,
                        receipt_root: str | Path | None = None) -> bool:
    if not dry_run:
        if limit_markets is not None:
            raise RuntimeError(
                "--limit-markets is dry-run-only and cannot enter live radar"
            )
        _assert_live_send_window(asof, slot, now=_now_kst())
    spec, is_fallback, reason = resolve_champion(
        slot,
        expected_asof=asof,
    )
    log.info("slot=%s champion=%s is_fallback=%s reason=%s",
             slot, spec.id, is_fallback, reason)

    res = call_predict(
        spec,
        asof,
        slot,
        limit_markets,
        snapshot_root=snapshot_root,
        # A CLI preview must not reserve the canonical immutable identity.
        # Tests/research may still request a dedicated explicit root.
        persist_snapshot=not (dry_run and snapshot_root is None),
    )
    if res.get("asof") != asof or res.get("slot") != slot:
        raise RuntimeError(
            f"score snapshot identity mismatch: requested={asof}/{slot} "
            f"resolved={res.get('asof')}/{res.get('slot')}"
        )
    model = res.get("model")
    if not isinstance(model, dict) or model.get("id") != spec.id:
        raise RuntimeError(
            f"score snapshot model mismatch: "
            f"expected={spec.id} "
            f"resolved={model.get('id') if isinstance(model, dict) else None}"
        )
    if not dry_run:
        _assert_approved_live_ranking(res)
    log.info("asof=%s slot=%s(resolved=%s) btc_regime=%s universe_n=%d calib=%s top=%d",
             res["asof"], slot, res.get("slot", slot), res["btc_regime"],
             res["universe_n"], res["calibration_source"], len(res["top3"]))
    log.info("score snapshot id=%s path=%s",
             res.get("snapshot_id"), res.get("snapshot_path"))
    msg = format_radar(res, slot, dry_run=dry_run,
                       champion_id=spec.id, is_fallback=is_fallback)

    if dry_run:
        # preview는 영속 receipt의 성공 여부와 무관하게 항상 렌더링한다.
        maybe_notify_champion_change(slot, dry_run=True)
        return send_telegram(msg, dry_run=True)

    return _send_and_record(
        res,
        msg,
        slot=slot,
        receipt_root=receipt_root,
    )


def _configure_cli_logging() -> None:
    """CLI 실행 전용 로깅 구성 — 반드시 main() 에서만 호출할 것.

    import 시점에 root logger 를 stdout 으로 구성하면, 이 모듈을 import
    하는 다른 프로세스(close gate NUL 프로토콜, heartbeat 'ok' 프로브)의
    기계 파싱 stdout 이 오염된다 — 07-28/29 이틀 연속 라이브 장애의
    근본 원인 클래스라 import 부작용을 금지한다.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main():
    _configure_cli_logging()
    ap = argparse.ArgumentParser(
        description="SHADOW 추천 레이더 텔레그램 발송 (08:50 / 09:05)")
    ap.add_argument("--asof", type=str, default=None, help="YYYY-MM-DD (default=오늘 KST)")
    ap.add_argument("--slot", type=str, default="open", choices=["preopen", "open"],
                    help="발송 슬롯 (preopen=08:45~09:00 / open=09:00~09:21 KST gate)")
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

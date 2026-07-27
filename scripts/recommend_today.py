"""SHADOW 채널 일일 추천 기록 스크립트 (기록만 — 발송/주문/cron 전부 X).

흐름:
  1. signals.recommend.score_candidates(today) 호출 → leak-free rank-mean top-3
  2. top-3 를 output/shadow_ledger_recommend.csv 에 append (status='open')
  3. 청산 realized 는 다음날 scripts/close_recommend_ledger.py 가 채운다 (status='closed')

★★★ 이 스크립트는 SHADOW(기록만) 채널이다 (CLAUDE.md §3.1, ops-steward §0):
    - 텔레그램 발송 코드 없음.
    - cron/systemd timer 미등록 (수동/추후 사용자 sudo 후 등록).
    - 업비트 자동주문·API key 절대 없음. entry_open은 DB의 그날 09:00 open
      표시용 기준가일 뿐 — 실제 주문/fill X.
    - 사이징/실제 청산 시뮬은 ledger 책임 (여기서는 sl/tp 플랜값만 기록).

strict forward 평가는 Telegram receipt 뒤 다음 15분봉 open부터 시작한다.
청산 플랜 = -3% 손절 / +5% 익절 (shadow 가상평가용).
유니버스/라벨/사이징은 사용자 확정값 그대로 (signals.recommend 상수).

거래비용: realized PnL 차감은 청산 스크립트(close_recommend_ledger.py)에서 net 처리.

사용:
    python scripts/recommend_today.py                 # 오늘(KST) 기록
    python scripts/recommend_today.py --slot preopen  # 08:50 R1 전용 원장 기록
    python scripts/recommend_today.py --asof 2026-05-31 --dry-run
    python scripts/recommend_today.py --dry-run       # 기록 X, 출력만
    python scripts/recommend_today.py --limit-markets 120  # 개발용
"""
from __future__ import annotations

import argparse
import logging
import math
import numbers
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, cast

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from notifier.delivery_receipt import (  # noqa: E402
    DeliveryReceiptError,
    read_delivery_receipt,
)
from ledger.csv_store import atomic_write_csv, ledger_lock  # noqa: E402
from signals.recommend_snapshot import (  # noqa: E402
    DEFAULT_SNAPSHOT_ROOT,
    SNAPSHOT_SCHEMA_VERSION,
    get_or_create_recommend_snapshot,
    snapshot_path,
)

# SHADOW recommend ledger — 기존 shadow_ledger_distribution.csv 컨벤션 참고.
# date/coin/rank/score/pump_prob/dump_risk_flag/btc_regime/entry_open/sl_pct/tp_pct/status
# + 청산 후 채움(close_recommend_ledger.py): exit_price/exit_reason/realized_pct/pump20_hit/closed_at
SHADOW_RECOMMEND_LEDGER = "output/shadow_ledger_recommend.csv"
# 08:50 R1 전용 shadow ledger. 09:05 open R1 과 추천 시점·실행 가능성이 달라
# 같은 원장에 섞지 않는다. model_registry recommend_r1_preopen 경로와 일치해야 한다.
SHADOW_RECOMMEND_LEDGER_PREOPEN = "output/shadow_ledger_recommend_preopen.csv"
# R2 challenger 전용 shadow ledger (record-only, 챔피언과 분리 평가). gitignore
# shadow_ledger_* 패턴에 이미 걸림. 스키마는 R1 과 동일 (champion_selector 가
# 30거래일 후 R1 vs R2 를 같은 컬럼으로 비교 가능해야 하므로).
SHADOW_RECOMMEND_LEDGER_R2 = "output/shadow_ledger_recommend_r2.csv"
# A1 sustainability challenger 전용 shadow ledger (record-only, 챔피언/R2 와 분리 평가).
# gitignore shadow_ledger_* 패턴에 이미 걸림. 스키마는 R1 과 동일 (champion_selector 가
# forward CLOSED 로 R1 vs A1 을 같은 컬럼으로 하방-우선 비교 가능해야 하므로).
SHADOW_RECOMMEND_LEDGER_SUSTAIN = "output/shadow_ledger_recommend_sustain.csv"

RECOMMEND_LEDGER_COLS = [
    "date",            # 추천/평가 대상일 D
    "coin",            # KRW-XXX
    "rank",            # 1..3
    "score",           # rank-mean score (0~1)
    "pump_prob",       # train-only bucket historical estimate
    "pump_prob_pct",   # 표시 문자열 (예: "5.5%"), strict calibrated 아님
    "dump_risk_flag",  # ⚠️ hi-risk (상위 1/3) bool
    "btc_regime",      # D-1 regime
    "entry_open",      # day-D 09:00 reference price (실행 fill 아님)
    "sl_pct",          # 청산 플랜 손절 = -0.03
    "tp_pct",          # 청산 플랜 익절 = +0.05
    "status",          # 'open' → close 후 'closed'
    "calibration_source",  # bucket_score_pump20 | base_rate
    # --- 발송/재현 귀속 (recommend_snapshot + delivery receipt) ---
    "snapshot_id",     # 발송과 ledger가 공유한 immutable score snapshot
    "snapshot_path",   # snapshot JSON 경로
    "decision_completed_at",  # immutable scorer completion UTC ISO timestamp
    "delivery_ok",     # Telegram 전달 성공 여부. 미시도/receipt 없음 = null
    "sent_at",         # 실제 전달 성공 UTC ISO 시각. receipt 없음/실패 = null
    # --- 평가용 분포/하방 확률 (SHADOW→ADOPT calibration 정직성·rr 정렬 감사용) ---
    "p_up5",           # historical/resub P(고가 ≥+5%)
    "p_up10",          # historical/resub P(고가 ≥+10%) (R1 분자)
    "p_up20",          # train-only bucket P(고가 ≥+20%)
    "p_dn5",           # historical/resub P(저가 ≤-5%) (R1 분모)
    "p_dn10",          # P(저가 ≤-10%) deep-dump
    "exp_downside",    # E[하방] (음수 low excursion 기대값)
    "rr_ratio",        # R1 정렬키 = p_up10 / max(p_dn5, eps)
    # --- 청산 후 채움 (close_recommend_ledger.py) ---
    "exit_price",      # 청산가 (15m 경로)
    "exit_reason",     # SL | TP | EOD
    "realized_pct",    # net 실현 수익률 % (왕복 0.15% 차감 후)
    "pump20_hit",      # 일봉 라벨: (high_D/open_D - 1) >= 0.20 적중 여부 (0/1)
    "closed_at",       # 청산 기록 시각 (UTC iso)
]

KST = timezone(timedelta(hours=9))
_ROOT = Path(__file__).resolve().parent.parent
_CANONICAL_LEDGER_RELATIVE = {
    ("R1", "open"): "output/shadow_ledger_recommend.csv",
    ("R1", "preopen"): "output/shadow_ledger_recommend_preopen.csv",
    ("R2", "open"): "output/shadow_ledger_recommend_r2.csv",
    ("A1", "open"): "output/shadow_ledger_recommend_sustain.csv",
}
_CANONICAL_MODEL_IDS = {
    ("R1", "open"): "recommend_r1_open",
    ("R1", "preopen"): "recommend_r1_preopen",
    ("R2", "open"): "recommend_r2_open",
    ("A1", "open"): "recommend_r1_sustain_open",
}
_CANONICAL_RULE_VERSIONS = {
    "R1": "r1_riskreward_v1",
    "R2": "r2_downside_penalized_v1",
    "A1": "a1_sustainability_v1",
}
_CANONICAL_SCORE_SCHEMA_VERSION = "recommend_score.v2"
_FORWARD_RECORD_WINDOWS = {
    "preopen": (time(8, 45), time(9, 0)),
    "open": (time(9, 0), time(9, 21)),
}

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("recommend_today")

_SNAPSHOT_ROW_IDENTITY_COLS = [
    "date",
    "coin",
    "rank",
    "score",
    "pump_prob",
    "pump_prob_pct",
    "dump_risk_flag",
    "btc_regime",
    "entry_open",
    "sl_pct",
    "tp_pct",
    "calibration_source",
    "decision_completed_at",
    "p_up5",
    "p_up10",
    "p_up20",
    "p_dn5",
    "p_dn10",
    "exp_downside",
    "rr_ratio",
]


def _today_kst() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d")


def _resolved_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = _ROOT / candidate
    return candidate.resolve()


def _uses_canonical_snapshot_store(root: str | Path | None) -> bool:
    if root is None:
        return True
    return _resolved_project_path(root) == DEFAULT_SNAPSHOT_ROOT.resolve()


def _canonical_ledger_path(ranking: str, slot: str) -> Path:
    try:
        relative = _CANONICAL_LEDGER_RELATIVE[(ranking.upper(), slot)]
    except KeyError as exc:
        raise ValueError(
            f"unsupported canonical ranking/slot: {ranking!r}/{slot!r}"
        ) from exc
    return (_ROOT / relative).resolve()


def _assert_canonical_forward_write_request(
    asof: str,
    *,
    ranking: str,
    slot: str,
    ledger_path: str | Path,
    limit_markets: int | None,
    snapshot_root: str | Path | None,
    receipt_root: str | Path | None,
    require_receipt: bool,
) -> bool:
    """Protect official forward ledgers from replay/development parameters.

    Custom ledgers remain available for explicit research/replay work.  The
    canonical ledgers consumed by champion/policy evaluation accept only the
    current KST decision, full-universe/default evidence roots, and the exact
    registered ranking/slot mapping.
    """
    target = _resolved_project_path(ledger_path)
    canonical_targets = {
        (_ROOT / relative).resolve()
        for relative in _CANONICAL_LEDGER_RELATIVE.values()
    }
    if target not in canonical_targets:
        return False

    expected = _canonical_ledger_path(ranking, slot)
    if target != expected:
        raise RuntimeError(
            "canonical recommend ledger/ranking/slot mismatch: "
            f"target={target} expected={expected}"
        )
    try:
        decision_day = date.fromisoformat(asof)
    except ValueError as exc:
        raise ValueError(f"invalid canonical recommend asof: {asof!r}") from exc
    today = date.fromisoformat(_today_kst())
    if decision_day != today:
        raise RuntimeError(
            "historical/future canonical recommend write rejected: "
            f"asof={decision_day} today_kst={today}; use --dry-run or a "
            "noncanonical research ledger"
        )
    if limit_markets is not None:
        raise RuntimeError(
            "market-limited canonical recommend write rejected; "
            "--limit-markets is dry-run/research-only"
        )
    if snapshot_root is not None or receipt_root is not None:
        raise RuntimeError(
            "custom evidence roots cannot write a canonical recommend ledger"
        )
    if ranking.upper() == "R1" and not require_receipt:
        raise RuntimeError(
            "canonical active R1 ledger requires a successful delivery receipt"
        )
    return True


def _assert_canonical_forward_snapshot(
    result: dict,
    *,
    asof: str,
    ranking: str,
    slot: str,
) -> None:
    """Require current immutable provenance and an in-window decision cohort."""
    if result.get("snapshot_schema") != SNAPSHOT_SCHEMA_VERSION:
        raise RuntimeError(
            "legacy recommend snapshot cannot enter a canonical forward ledger"
        )
    request = result.get("request")
    ranking = ranking.upper()
    if (
        not isinstance(request, dict)
        or request.get("asof") != asof
        or request.get("slot") != slot
        or request.get("ranking") != ranking
        or request.get("limit_markets") is not None
    ):
        raise RuntimeError("canonical recommend snapshot request identity mismatch")
    model = result.get("model")
    if (
        not isinstance(model, dict)
        or model.get("id") != _CANONICAL_MODEL_IDS[(ranking, slot)]
        or model.get("ranking") != ranking
    ):
        raise RuntimeError("canonical recommend snapshot model identity mismatch")
    if (
        result.get("score_schema_version")
        != _CANONICAL_SCORE_SCHEMA_VERSION
        or result.get("rule_version") != _CANONICAL_RULE_VERSIONS[ranking]
    ):
        raise RuntimeError("canonical recommend snapshot scorer/rule mismatch")
    rank_basis = result.get("rank_basis")
    if ranking == "R1":
        rank_basis_ok = rank_basis == "R1_riskreward(de-corr head)"
    elif ranking == "R2":
        rank_basis_ok = rank_basis == "R2_penalized(λ=1.0, de-corr head)"
    else:
        rank_basis_ok = (
            isinstance(rank_basis, str)
            and rank_basis.startswith(
                "A1_sustain(dump_B, cutoff_q=0.6, de-corr head, demoted "
            )
            and rank_basis.endswith(")")
        )
    if not rank_basis_ok:
        raise RuntimeError(
            "degraded/fallback ranking cannot enter a canonical forward ledger"
        )

    expected_path = snapshot_path(
        asof,
        slot,
        ranking,
        None,
    ).resolve()
    raw_path = result.get("snapshot_path")
    if not isinstance(raw_path, str) or _resolved_project_path(raw_path) != expected_path:
        raise RuntimeError("canonical recommend snapshot path mismatch")

    start_wall, end_wall = _FORWARD_RECORD_WINDOWS[slot]
    decision_day = date.fromisoformat(asof)
    for field in ("decision_started_at", "decision_completed_at"):
        raw = result.get(field)
        if not isinstance(raw, str):
            raise RuntimeError(f"canonical recommend snapshot missing {field}")
        try:
            observed = datetime.fromisoformat(raw)
        except ValueError as exc:
            raise RuntimeError(
                f"canonical recommend snapshot has invalid {field}"
            ) from exc
        if observed.tzinfo is None:
            raise RuntimeError(
                f"canonical recommend snapshot {field} must be timezone-aware"
            )
        observed_kst = observed.astimezone(KST)
        wall = observed_kst.timetz().replace(tzinfo=None)
        if observed_kst.date() != decision_day or not start_wall <= wall < end_wall:
            raise RuntimeError(
                f"canonical recommend snapshot {field} outside {slot} "
                f"decision window: {observed_kst.isoformat()}"
            )


def default_ledger_path(ranking: str, slot: str) -> str:
    """ranking/slot 조합의 기본 ledger. R1 preopen/open은 반드시 분리한다."""
    ranking = ranking.upper()
    if slot not in {"preopen", "open"}:
        raise ValueError(f"slot must be preopen|open, got {slot!r}")
    if ranking == "R1":
        return (
            SHADOW_RECOMMEND_LEDGER_PREOPEN
            if slot == "preopen"
            else SHADOW_RECOMMEND_LEDGER
        )
    if ranking == "R2":
        return SHADOW_RECOMMEND_LEDGER_R2
    if ranking == "A1":
        return SHADOW_RECOMMEND_LEDGER_SUSTAIN
    raise ValueError(f"ranking must be R1|R2|A1, got {ranking!r}")


def _is_missing(value: object) -> bool:
    missing = pd.isna(value)
    try:
        return bool(missing)
    except (TypeError, ValueError):
        return False


def _identity_value_equal(actual: object, expected: object) -> bool:
    actual_missing = _is_missing(actual)
    expected_missing = _is_missing(expected)
    if actual_missing or expected_missing:
        return actual_missing and expected_missing
    if isinstance(expected, bool):
        if isinstance(actual, str):
            normalized = actual.strip().lower()
            if normalized not in {"true", "false"}:
                return False
            return (normalized == "true") is expected
        return bool(actual) is expected
    if isinstance(expected, numbers.Real):
        if isinstance(actual, bool):
            return False
        try:
            actual_number = float(cast(Any, actual))
            expected_number = float(expected)
        except (TypeError, ValueError):
            return False
        return (
            math.isfinite(actual_number)
            and math.isfinite(expected_number)
            and math.isclose(
                actual_number,
                expected_number,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        )
    return str(actual) == str(expected)


def _candidate_key(row: pd.Series, *, source: str) -> tuple[str, int]:
    coin = str(row.get("coin", "")).strip()
    try:
        rank_number = float(cast(Any, row.get("rank")))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{source} has invalid candidate rank") from exc
    if (
        not coin
        or not math.isfinite(rank_number)
        or not rank_number.is_integer()
        or rank_number < 1
    ):
        raise RuntimeError(f"{source} has invalid candidate identity")
    return coin, int(rank_number)


def _assert_existing_rows_match_snapshot(
    existing_rows: pd.DataFrame,
    snapshot_rows: pd.DataFrame,
    *,
    asof: str,
) -> None:
    """Prove a same-date ledger cohort is the immutable snapshot's cohort."""
    existing_by_key: dict[tuple[str, int], pd.Series] = {}
    expected_by_key: dict[tuple[str, int], pd.Series] = {}
    for source, frame, target in (
        ("existing ledger", existing_rows, existing_by_key),
        ("score snapshot", snapshot_rows, expected_by_key),
    ):
        for _, row in frame.iterrows():
            key = _candidate_key(row, source=source)
            if key in target:
                raise RuntimeError(
                    f"{source} has duplicate candidate key for asof={asof}: "
                    f"{key}"
                )
            target[key] = row
    if set(existing_by_key) != set(expected_by_key):
        raise RuntimeError(
            f"recommend snapshot candidate identity conflict for asof={asof}: "
            f"existing={sorted(existing_by_key)} "
            f"snapshot={sorted(expected_by_key)}"
        )
    for key, expected in expected_by_key.items():
        actual = existing_by_key[key]
        for column in _SNAPSHOT_ROW_IDENTITY_COLS:
            if not _identity_value_equal(actual.get(column), expected.get(column)):
                raise RuntimeError(
                    f"recommend snapshot row conflict for asof={asof} "
                    f"candidate={key} column={column}"
                )

    for column in ("snapshot_id", "snapshot_path"):
        expected_value = snapshot_rows.iloc[0].get(column)
        if _is_missing(expected_value):
            raise RuntimeError(f"score snapshot missing {column}")
        for actual_value in existing_rows[column]:
            if (
                not _is_missing(actual_value)
                and not _identity_value_equal(actual_value, expected_value)
            ):
                raise RuntimeError(
                    f"recommend {column} conflict for asof={asof}"
                )
    delivery_values: set[bool] = set()
    sent_values: set[str] = set()
    for _, row in existing_rows.iterrows():
        delivery_value = row.get("delivery_ok")
        sent_value = row.get("sent_at")
        if _is_missing(delivery_value):
            delivery = None
        else:
            normalized = str(delivery_value).strip().lower()
            if normalized not in {"true", "false"}:
                raise RuntimeError(
                    f"recommend delivery_ok invalid for asof={asof}"
                )
            delivery = normalized == "true"
        sent = None if _is_missing(sent_value) else str(sent_value)
        if sent is not None and delivery is not True:
            raise RuntimeError(
                f"recommend delivery metadata conflict for asof={asof}"
            )
        if delivery is True and sent is None:
            raise RuntimeError(
                f"recommend successful delivery missing sent_at for asof={asof}"
            )
        if delivery is not None:
            delivery_values.add(delivery)
        if sent is not None:
            sent_values.add(sent)
    if len(delivery_values) > 1 or len(sent_values) > 1:
        raise RuntimeError(
            f"recommend delivery cohort inconsistent for asof={asof}"
        )
    expected_sent_at = snapshot_rows.iloc[0].get("sent_at")
    if not _is_missing(expected_sent_at):
        for actual_sent_at in existing_rows["sent_at"]:
            if (
                not _is_missing(actual_sent_at)
                and not _identity_value_equal(actual_sent_at, expected_sent_at)
            ):
                raise RuntimeError(
                    f"recommend sent_at conflict for asof={asof}"
                )


def append_today(asof: str, *, dry_run: bool = False,
                 limit_markets: int | None = None,
                 ledger_path: str | None = None,
                 ranking: str = "R1",
                 slot: str = "open",
                 snapshot_root: str | Path | None = None,
                 receipt_root: str | Path | None = None,
                 require_receipt: bool = False) -> dict:
    """asof 기준 top-3 를 shadow recommend ledger 에 append (status='open').

    ranking : {"R1","R2","A1"} — "R1"(기본)은 챔피언 경로(기존과 무변경). "R2"는
      downside-penalized, "A1"은 sustainability-filter 챌린저 → 각각 반드시 자기
      record-only ledger(SHADOW_RECOMMEND_LEDGER_R2 / _SUSTAIN)와 함께 호출한다
      (챔피언 ledger 오염 금지). R1도 slot='preopen'이면 08:50 전용 원장을
      기본값으로 사용해 open 원장과 섞지 않는다."""
    ledger_path = ledger_path or default_ledger_path(ranking, slot)
    canonical_forward_write = False
    if not dry_run:
        canonical_forward_write = _assert_canonical_forward_write_request(
            asof,
            ranking=ranking,
            slot=slot,
            ledger_path=ledger_path,
            limit_markets=limit_markets,
            snapshot_root=snapshot_root,
            receipt_root=receipt_root,
            require_receipt=require_receipt,
        )
    if dry_run and _uses_canonical_snapshot_store(snapshot_root):
        # A preview outside the operating window must never create the
        # canonical same-day snapshot and thereby poison the later live run.
        # This remains true when a caller explicitly supplies the canonical
        # root (or an equivalent path alias).
        from signals.recommend import score_candidates

        res = score_candidates(
            asof,
            slot=slot,
            ranking=ranking,
            limit_markets=limit_markets,
        )
    else:
        res = get_or_create_recommend_snapshot(
            asof,
            slot=slot,
            ranking=ranking,
            limit_markets=limit_markets,
            root=snapshot_root,
        )
    if res.get("asof") != asof or res.get("slot") != slot:
        raise RuntimeError(
            f"score snapshot identity mismatch: requested={asof}/{slot} "
            f"resolved={res.get('asof')}/{res.get('slot')}"
        )
    if canonical_forward_write:
        _assert_canonical_forward_snapshot(
            res,
            asof=asof,
            ranking=ranking,
            slot=slot,
        )
    log.info("asof=%s btc_regime=%s universe_n=%d calib=%s n_hist_dates=%d top=%d",
             res["asof"], res["btc_regime"], res["universe_n"],
             res["calibration_source"], res["n_history_dates"], len(res["top3"]))
    log.info("score snapshot id=%s path=%s",
             res.get("snapshot_id"), res.get("snapshot_path"))

    if not res["top3"]:
        log.warning("top3 empty — nothing to record (universe 0?)")
        return res

    snapshot_id = res.get("snapshot_id")
    snapshot_path_value = res.get("snapshot_path")
    if bool(snapshot_id) != bool(snapshot_path_value):
        raise RuntimeError(
            "score snapshot identity is incomplete: snapshot_id/path must "
            "both be present or both be absent"
        )
    receipt = None
    if snapshot_id and snapshot_path_value and (not dry_run or require_receipt):
        try:
            receipt = read_delivery_receipt(res, root=receipt_root)
        except DeliveryReceiptError as exc:
            # 손상 receipt를 null로 바꾸면 closer가 09:00 전체 경로로 fail-open한다.
            # 잘못된 귀속을 영구 CLOSED하는 것보다 실행을 실패시켜 수리를 요구한다.
            raise RuntimeError(f"delivery receipt invalid: {exc}") from exc
    elif require_receipt:
        raise RuntimeError(
            "delivery receipt requires a persisted score snapshot"
        )
    if require_receipt:
        if receipt is None:
            raise RuntimeError(
                "delivery receipt missing for active R1 ledger attribution"
            )
        if not receipt["delivery_ok"]:
            raise RuntimeError(
                "delivery receipt is not successful for active R1 ledger attribution"
            )
    delivery_ok = receipt.get("delivery_ok") if receipt else pd.NA
    sent_at = receipt.get("sent_at") if receipt else pd.NA

    rows = []
    for item in res["top3"]:
        rows.append({
            "date": res["asof"],
            "coin": item["coin"],
            "rank": item["rank"],
            "score": item["score"],
            "pump_prob": item["pump_prob"],
            "pump_prob_pct": item["pump_prob_pct"],
            "dump_risk_flag": bool(item["dump_risk_flag"]),
            "btc_regime": item["btc_regime"],
            "entry_open": item["entry_open"],
            "sl_pct": item["sl"],     # -0.03
            "tp_pct": item["tp"],     # +0.05
            "status": "open",
            "calibration_source": res["calibration_source"],
            "snapshot_id": res.get("snapshot_id"),
            "snapshot_path": res.get("snapshot_path"),
            "decision_completed_at": res.get("decision_completed_at"),
            "delivery_ok": delivery_ok,
            "sent_at": sent_at,
            "p_up5": item.get("p_up5"),
            "p_up10": item.get("p_up10"),
            "p_up20": item.get("p_up20"),
            "p_dn5": item.get("p_dn5"),
            "p_dn10": item.get("p_dn10"),
            "exp_downside": item.get("exp_downside"),
            "rr_ratio": item.get("rr_ratio"),
            "exit_price": pd.NA,
            "exit_reason": pd.NA,
            "realized_pct": pd.NA,
            "pump20_hit": pd.NA,
            "closed_at": pd.NA,
        })
    new = pd.DataFrame(rows)[RECOMMEND_LEDGER_COLS]

    p = Path(ledger_path)
    if dry_run:
        # An ephemeral preview must not consume or validate canonical ledger
        # state. A prior same-day live row can legitimately differ because the
        # model is refit from current inputs; previewing that result is still
        # useful and must remain write-free.
        log.info("[dry-run] would append %d rows to %s", len(new), p)
        _print_rows(new)
        return res

    with ledger_lock(p):
        if p.exists():
            existing = pd.read_csv(p)
            # closer/path-quality가 뒤에 붙인 평가 컬럼은 과거 CLOSED 행의 감사
            # 증거다. 새 추천을 append할 때 기본 스키마로 잘라내지 않고, 기존
            # extra 컬럼의 상대 순서와 값을 그대로 보존한다.
            original_columns = list(existing.columns)
            for c in RECOMMEND_LEDGER_COLS:
                if c not in existing.columns:
                    existing[c] = pd.NA
            for c in ["snapshot_id", "snapshot_path", "delivery_ok", "sent_at"]:
                # 빈 문자열 열은 read_csv에서 float로 추론될 수 있다. 재발송 성공 시
                # ISO timestamp/bool을 안전하게 보강하도록 object로 고정한다.
                existing[c] = existing[c].astype(object)
            extra_columns = [
                c for c in original_columns if c not in RECOMMEND_LEDGER_COLS
            ]
            ledger_columns = RECOMMEND_LEDGER_COLS + extra_columns
            existing = existing[ledger_columns]
            for c in extra_columns:
                new[c] = pd.NA
            new = new[ledger_columns]
            # 같은 날 이미 기록돼 있으면 중복 append 방지 (idempotent).
            already = existing[(existing["date"].astype(str) == res["asof"])]
            if len(already) > 0:
                _assert_existing_rows_match_snapshot(
                    already,
                    new,
                    asof=res["asof"],
                )
                # 발송보다 ledger가 먼저 기록됐거나 구버전 행이면 receipt/snapshot metadata만
                # 빈 칸에 보강한다. 추천 행 자체는 절대 교체/중복 append하지 않는다.
                metadata_cols = [
                    "snapshot_id", "snapshot_path", "delivery_ok", "sent_at",
                ]
                metadata_updated = False
                for c in metadata_cols:
                    value = new.iloc[0][c]
                    if pd.isna(value):
                        continue
                    target_idx = already.index[existing.loc[already.index, c].isna()]
                    if c == "delivery_ok" and bool(value):
                        # 최초 시도 실패 후 동일 snapshot 재발송 성공은 성과 귀속을
                        # not_delivered로 영구 고정하지 말고 성공으로 단조 보강한다.
                        target_idx = already.index
                    if len(target_idx):
                        existing.loc[target_idx, c] = value
                        metadata_updated = True
                if metadata_updated and not dry_run:
                    atomic_write_csv(existing, p)
                    log.info("backfilled snapshot/delivery metadata → %s", p)
                log.warning(
                    "asof=%s already has %d rows in %s — skip append (idempotent)",
                    res["asof"], len(already), p,
                )
                _print_rows(existing.loc[already.index])
                return res
            combined = pd.concat([existing, new], ignore_index=True)
        else:
            combined = new

        atomic_write_csv(combined, p)
        log.info("appended %d rows → %s (total %d)", len(new), p, len(combined))
        _print_rows(new)
    return res


def _print_rows(df: pd.DataFrame) -> None:
    cols = ["date", "coin", "rank", "score", "pump_prob", "pump_prob_pct",
            "dump_risk_flag", "btc_regime", "entry_open", "sl_pct", "tp_pct", "status"]
    cols = [c for c in cols if c in df.columns]
    print("\n=== recorded top-3 (SHADOW, record-only) ===")
    print(df[cols].to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description="SHADOW 일일 추천 기록 (발송/주문/cron X)")
    ap.add_argument(
        "--asof",
        type=str,
        default=None,
        help="YYYY-MM-DD (default=오늘 KST; 과거 날짜는 --dry-run 전용)",
    )
    ap.add_argument("--limit-markets", type=int, default=None, help="개발용 마켓 제한")
    ap.add_argument("--dry-run", action="store_true", help="기록 X, 출력만")
    ap.add_argument("--ranking", type=str, default="R1", choices=["R1", "R2", "A1"],
                    help="정렬 모드. R1=챔피언(기본), R2=downside-penalized 챌린저, "
                         "A1=sustainability-filter 챌린저")
    ap.add_argument("--slot", type=str, default="open", choices=["preopen", "open"],
                    help="score snapshot 슬롯 (default=open). 발송과 같은 슬롯을 지정")
    ap.add_argument("--ledger", type=str, default=None,
                    help="ledger 경로 (default: R1 preopen/open 각각 전용 원장, "
                         "R2→shadow_ledger_recommend_r2.csv, "
                         "A1→shadow_ledger_recommend_sustain.csv; custom은 research 전용)")
    ap.add_argument(
        "--require-receipt",
        action="store_true",
        help="receipt가 없거나 손상되면 원장 기록을 거부(active R1 daily용)",
    )
    args = ap.parse_args()

    asof = args.asof or _today_kst()
    ledger = args.ledger or default_ledger_path(args.ranking, args.slot)
    append_today(asof, dry_run=args.dry_run, limit_markets=args.limit_markets,
                 ledger_path=ledger, ranking=args.ranking, slot=args.slot,
                 require_receipt=args.require_receipt)


if __name__ == "__main__":
    main()

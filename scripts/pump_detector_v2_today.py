"""PUMP hunter v2 daily runner — shadow ledger append + 텔레그램 radar 발사.

사용자 컨펌 (2026-06-11): "텔레그램으로 오게 제대로 완성" — v2 radar 메시지 추가.

발사 정책:
- 후보 ≥ 1 → 🎯 radar 메시지 (hit 엣지 + 자동 net 음수 정직 고지 포함)
- 후보 0 + binance stale → ⚠️ 데이터 stale 1줄 (운영 알림)
- 후보 0 + 정상 → 텔레그램 X (로그만 — 매일 5번째 빈 메시지 방지)

원장: output/shadow_ledger_pump_hunter_v2.csv (close_recommend_ledger 가
TP5/SL3 + exit lab 7 잣대 자동 청산 — record-only, 자동 주문 없음).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import sys
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, cast

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ledger.csv_store import atomic_write_csv, ledger_lock  # noqa: E402
from notifier.delivery_receipt import (  # noqa: E402
    DeliveryReceiptError,
    validate_telegram_transport_evidence,
)
from notifier.telegram import (  # noqa: E402
    TelegramSendResult,
    send_telegram,
    send_telegram_with_receipt,
    telegram_error_is_ambiguous,
    validate_telegram_send_result,
)
from ops.artifact_provenance import (  # noqa: E402
    ArtifactSourceChangedError,
    ArtifactValidationError,
    atomic_write_json,
    file_identity,
    manifest_digest_matches,
    strict_json_object,
    with_manifest_digest,
)
from ops.file_lock import file_lock  # noqa: E402
from ops.radar_verdict import (  # noqa: E402
    RADAR_TERMINAL_STATE,
    assert_radar_send_allowed,
    radar_send_guard,
)
from scripts.recommend_today import RECOMMEND_LEDGER_COLS  # noqa: E402
from signals.pump_detector_v1 import (  # noqa: E402
    DB_PATH as UPBIT_D1_DB,
)
from signals.pump_detector_v1 import UNIVERSE_TOP_N  # noqa: E402
from signals.pump_detector_v2 import (  # noqa: E402
    BINANCE_DB,
    BN_VOL_SURGE_MIN,
    MAX_CANDIDATES,
    OOS_BASELINE_HIT_PCT,
    OOS_BASE_RATE_PCT,
    OOS_HIT_PCT,
    OOS_NET_TP5SL3_PCT,
    ROC7_RANK_MIN,
    SL_PCT,
    TP_PCT,
    score_pump_v2_candidates,
)

PUMP_V2_LEDGER = "output/shadow_ledger_pump_hunter_v2.csv"
PUMP_V2_RECEIPT_ROOT = "output/pump_v2_receipts"
PUMP_V2_DECISION_ROOT = "output/pump_v2_decisions"
RADAR_VERDICT_PATH = RADAR_TERMINAL_STATE
PUMP_V2_RECEIPT_SCHEMA = "pump_v2_delivery_receipt.v1"
PUMP_V2_DECISION_SCHEMA = "pump_v2_decision.v2"
PUMP_V2_LEGACY_DECISION_SCHEMA = "pump_v2_decision.v1"
FORWARD_EVIDENCE_ACTIVATION_DATE = date(2026, 7, 27)
FORWARD_PROVENANCE_SCHEMA = "pump_forward_provenance.v1"
PUMP_V2_RULE = (
    f"roc_7d_rank > {ROC7_RANK_MIN} "
    f"AND b_vol_surge > {BN_VOL_SURGE_MIN}"
)
PUMP_V2_RULE_ID = "roc7_rank+bn_volsurge"
PUMP_V2_OOS = {
    "hit_pct": OOS_HIT_PCT,
    "baseline_hit_pct": OOS_BASELINE_HIT_PCT,
    "base_rate_pct": OOS_BASE_RATE_PCT,
    "net_tp5sl3_pct": OOS_NET_TP5SL3_PCT,
}

EXTRA_LEDGER_COLS = [
    "model_id", "rule_version", "rule_id", "feature_date",
    "liq_rank_daily", "roc_7d", "roc_7d_rank", "atr_pct_14", "log_return_1d",
    "b_vol_surge", "b_ret_1d",
]
PUMP_V2_LEDGER_COLS = RECOMMEND_LEDGER_COLS + EXTRA_LEDGER_COLS
_LEDGER_IDENTITY_COLS = [
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
    "p_up20",
    "model_id",
    "rule_version",
    "rule_id",
    "feature_date",
    "liq_rank_daily",
    "roc_7d",
    "roc_7d_rank",
    "atr_pct_14",
    "log_return_1d",
    "b_vol_surge",
    "b_ret_1d",
]
_INTEGRITY_FIELD = "integrity_sha256"
_DECISION_DOCUMENT_KEYS = {
    "schema",
    "asof",
    "decision_id",
    "decision",
    "recorded_at",
}
_RECEIPT_DOCUMENT_KEYS = {
    "schema",
    "asof",
    "decision_id",
    "decision",
    "delivery_ok",
    "attempted_at",
    "sent_at",
    "recorded_at",
    "error",
}
_POSTACTIVATION_RECEIPT_KEYS = {
    "message_sha256",
    "chat_id_sha256",
    "chunk_count",
    "telegram_messages",
}

KST = timezone(timedelta(hours=9))

log = logging.getLogger("pump_detector_v2_today")

BTC_KR = {
    "bull_quiet": "🟢 강세 안정", "bull_volatile": "🟡 강세 변동",
    "bear_quiet": "🔴 약세 안정", "bear_volatile": "🔴 약세 변동",
}
VALID_BTC_REGIMES = frozenset(BTC_KR)
LIVE_RUN_START = datetime.min.time().replace(hour=9)
LIVE_RUN_END = datetime.min.time().replace(hour=9, minute=31)
_MARKET_RE = re.compile(r"KRW-[A-Z0-9]+")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SOURCE_PATHS = (
    "scripts/pump_detector_v2_today.py",
    "signals/pump_detector_v2.py",
    "signals/pump_detector_v1.py",
    "signals/features.py",
    "data/database.py",
    "data/market_universe.py",
)
_CANDIDATE_FINITE_FIELDS = (
    "score",
    "entry_open",
    "roc_7d",
    "roc_7d_rank",
    "atr_pct_14",
    "log_return_1d",
    "b_vol_surge",
    "liq_rank_daily",
)
_OOS_FINITE_FIELDS = (
    "hit_pct",
    "baseline_hit_pct",
    "base_rate_pct",
    "net_tp5sl3_pct",
)


def _today_kst() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d")


def _now_kst() -> datetime:
    return datetime.now(KST)


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return path.resolve(strict=False)


def _assert_canonical_forward_request(
    *,
    ledger_path: str | Path,
    decision_root: str | Path,
    receipt_root: str | Path,
    max_candidates: int,
) -> None:
    """Reject custom live outputs/config that could poison forward evidence."""
    if _project_path(ledger_path) != _project_path(PUMP_V2_LEDGER):
        raise RuntimeError("v2 live ledger must be the canonical ledger")
    if _project_path(decision_root) != _project_path(PUMP_V2_DECISION_ROOT):
        raise RuntimeError("v2 live decision root must be the canonical root")
    if _project_path(receipt_root) != _project_path(PUMP_V2_RECEIPT_ROOT):
        raise RuntimeError("v2 live receipt root must be the canonical root")
    if max_candidates != MAX_CANDIDATES:
        raise RuntimeError(
            "v2 live max_candidates must use the official configuration"
        )


def _file_identity(path_value: str | Path) -> dict[str, object]:
    path = Path(path_value)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    try:
        identity = file_identity(path, root=_PROJECT_ROOT)
    except (OSError, ArtifactSourceChangedError) as exc:
        raise RuntimeError(f"v2 provenance source is unavailable: {path}") from exc
    if not identity.get("exists") or int(identity.get("size") or 0) <= 0:
        raise RuntimeError(f"v2 provenance source is invalid: {path}")
    return {
        "path": identity["path"],
        "bytes": identity["size"],
        "sha256": identity["sha256"],
    }


def _current_forward_inputs() -> dict[str, dict[str, object]]:
    return {
        "sources": {
            source: _file_identity(source)
            for source in _SOURCE_PATHS
        },
        "data": {
            "data/upbit_d1.db": _file_identity(UPBIT_D1_DB),
            "data/binance_d1.db": _file_identity(BINANCE_DB),
        },
    }


def _forward_provenance(decision: dict) -> dict[str, object]:
    inputs = _current_forward_inputs()
    return {
        "schema": FORWARD_PROVENANCE_SCHEMA,
        "evidence_class": "canonical_forward",
        "runner": "pump_detector_v2",
        "config": {
            "top_universe": UNIVERSE_TOP_N,
            "max_candidates": MAX_CANDIDATES,
            "limit_markets": None,
        },
        "decision_basis": {
            "asof": decision["asof"],
            "feature_date": decision.get("feature_date"),
            "universe_n": decision["universe_n"],
            "n_candidates": decision["n_candidates"],
            "binance_status": decision["binance_status"],
        },
        **inputs,
    }


def _with_forward_provenance(decision: dict) -> dict:
    enriched = dict(decision)
    enriched["execution_provenance"] = _forward_provenance(enriched)
    return enriched


def _validate_file_identities(
    value: object,
    *,
    expected_paths: set[str],
    field: str,
) -> None:
    if not isinstance(value, dict) or set(value) != expected_paths:
        raise ValueError(f"v2 {field} identities are invalid")
    for expected_path, identity in value.items():
        if (
            not isinstance(identity, dict)
            or identity.get("path") != expected_path
            or isinstance(identity.get("bytes"), bool)
            or not isinstance(identity.get("bytes"), int)
            or identity["bytes"] <= 0
            or not isinstance(identity.get("sha256"), str)
            or _SHA256_RE.fullmatch(identity["sha256"]) is None
        ):
            raise ValueError(
                f"v2 {field} identity is invalid: {expected_path}"
            )


def _validate_forward_provenance(decision: dict) -> None:
    provenance = decision.get("execution_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("v2 canonical forward provenance is missing")
    if (
        provenance.get("schema") != FORWARD_PROVENANCE_SCHEMA
        or provenance.get("evidence_class") != "canonical_forward"
        or provenance.get("runner") != "pump_detector_v2"
        or provenance.get("config")
        != {
            "top_universe": UNIVERSE_TOP_N,
            "max_candidates": MAX_CANDIDATES,
            "limit_markets": None,
        }
        or provenance.get("decision_basis")
        != {
            "asof": decision["asof"],
            "feature_date": decision.get("feature_date"),
            "universe_n": decision["universe_n"],
            "n_candidates": decision["n_candidates"],
            "binance_status": decision["binance_status"],
        }
    ):
        raise ValueError("v2 canonical forward provenance is invalid")
    _validate_file_identities(
        provenance.get("sources"),
        expected_paths=set(_SOURCE_PATHS),
        field="source",
    )
    _validate_file_identities(
        provenance.get("data"),
        expected_paths={"data/upbit_d1.db", "data/binance_d1.db"},
        field="data",
    )


def _assert_live_run_window(
    asof: str,
    *,
    now: datetime | None = None,
) -> None:
    try:
        decision_day = date.fromisoformat(asof)
    except ValueError as exc:
        raise ValueError(f"invalid v2 live asof: {asof!r}") from exc
    observed = now or _now_kst()
    if observed.tzinfo is None:
        raise ValueError("v2 live clock must be timezone-aware")
    observed = observed.astimezone(KST)
    if decision_day != observed.date():
        raise RuntimeError(
            f"stale v2 live run rejected: asof={decision_day} "
            f"today_kst={observed.date()}"
        )
    wall_time = observed.timetz().replace(tzinfo=None)
    if not LIVE_RUN_START <= wall_time < LIVE_RUN_END:
        raise RuntimeError(
            f"outside v2 live run window: "
            f"{wall_time.isoformat(timespec='seconds')} not in "
            "[09:00,09:31) KST"
        )


def _live_run_deadline(asof: str) -> datetime:
    try:
        decision_day = date.fromisoformat(asof)
    except ValueError as exc:
        raise ValueError(f"invalid v2 live asof: {asof!r}") from exc
    return datetime.combine(decision_day, LIVE_RUN_END, tzinfo=KST)


def _fmt_price(v) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    if v >= 1000:
        return f"{v:,.0f}원"
    if v >= 1:
        return f"{v:.1f}원"
    return f"{v:.4f}원"


def build_message(res: dict, dry_run: bool = False) -> str:
    date_str = res["asof"]
    header = f"🎯 PUMP 레이더 v2 {date_str} (KST 09:05)"
    if dry_run:
        header += "  [DRY-RUN]"
    lines = [header]
    regime = BTC_KR.get(str(res.get("btc_regime", "")), str(res.get("btc_regime", "—")))
    lines.append(f"BTC: {regime} | rule: 7d모멘텀 상위 + Binance 거래량 surge")
    lines.append("")

    cands = res.get("candidates", [])
    if not cands:
        if str(res.get("binance_status", "ok")) != "ok":
            lines.append(f"⚠️ binance 데이터 문제 — {res.get('binance_status')}")
        else:
            lines.append("━━━ 오늘 rule fire 없음 ━━━")
        return "\n".join(lines)

    lines.append(f"━━━ 급등 후보 {len(cands)}건 (검증 hit {OOS_HIT_PCT}% ≈ base {OOS_BASE_RATE_PCT}% 의 6배) ━━━")
    lines.append("")
    for c in cands:
        coin = str(c["market"]).replace("KRW-", "")
        lines.append(
            f"#{c['rank']} {coin}  기준가 ≈ {_fmt_price(c['entry_open'])}"
        )
        lines.append(
            f"   ▸ Binance surge {c['b_vol_surge']:.1f}× | 7d모멘텀 rank {c['roc_7d_rank']:.2f}"
            f" | 7d {c['roc_7d']:+.1f}%"
        )
    lines.append("")
    lines.append("━━━ 사용 ━━━")
    lines.append(f"• 검증 (최근 7개월 OOS): +20% 도달 {OOS_HIT_PCT}% — 약 12건 중 1건")
    lines.append(f"• ★ 자동 룰 (TP5/SL3) net 은 {OOS_NET_TP5SL3_PCT}% 음수 — 진입·청산은 본인 판단")
    lines.append("• 하락 위험 동반 — 사이즈 작게, 자동 주문 없음 (shadow 기록만)")
    return "\n".join(lines)


def _decision_id(res: dict) -> str:
    """후보 payload 자체를 고정하는 content-addressed 실행 ID."""
    encoded = json.dumps(
        res,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return f"pump-v2-{hashlib.sha256(encoded).hexdigest()[:20]}"


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"v2 {field} must be a finite number")
    try:
        parsed = float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"v2 {field} must be a finite number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"v2 {field} must be a finite number")
    return parsed


def _validate_decision_result(
    res: dict,
    *,
    expected_asof: str | None = None,
    max_candidates: int | None = None,
) -> None:
    if not isinstance(res, dict):
        raise ValueError("v2 decision must be an object")
    try:
        asof = date.fromisoformat(str(res.get("asof", "")))
    except ValueError as exc:
        raise ValueError(f"invalid v2 decision asof: {res.get('asof')!r}") from exc
    if expected_asof is not None and res["asof"] != expected_asof:
        raise ValueError(
            f"v2 scorer asof mismatch: requested={expected_asof} "
            f"returned={res['asof']}"
        )
    if res.get("model_id") != "pump_hunter_v2":
        raise ValueError(f"invalid v2 model_id: {res.get('model_id')!r}")
    if res.get("rule_version") != "pump_detector_v2":
        raise ValueError(
            f"invalid v2 rule_version: {res.get('rule_version')!r}"
        )
    if res.get("rule") != PUMP_V2_RULE:
        raise ValueError(f"invalid v2 rule: {res.get('rule')!r}")
    candidates = res.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("v2 candidates must be a list")
    n_candidates = res.get("n_candidates")
    if (
        isinstance(n_candidates, bool)
        or not isinstance(n_candidates, int)
        or n_candidates != len(candidates)
    ):
        raise ValueError("v2 n_candidates does not match candidates")
    if max_candidates is not None and n_candidates > max_candidates:
        raise ValueError(
            f"v2 scorer exceeded max_candidates: "
            f"{n_candidates}>{max_candidates}"
        )
    universe_n = res.get("universe_n")
    if (
        isinstance(universe_n, bool)
        or not isinstance(universe_n, int)
        or universe_n < n_candidates
    ):
        raise ValueError("invalid v2 universe_n")
    binance_status = res.get("binance_status")
    if not isinstance(binance_status, str) or not binance_status:
        raise ValueError("invalid v2 binance_status")
    if binance_status != "ok" and candidates:
        raise ValueError("v2 stale Binance decision cannot contain candidates")
    if binance_status == "ok":
        if universe_n <= 0:
            raise ValueError("v2 healthy decision universe_n must be positive")
        feature_value = res.get("feature_date")
        if not isinstance(feature_value, str):
            raise ValueError("v2 healthy decision is missing feature_date")
        try:
            feature_at = datetime.fromisoformat(feature_value)
        except ValueError as exc:
            raise ValueError("v2 feature_date is invalid") from exc
        if feature_at.tzinfo is not None:
            feature_at = feature_at.astimezone(KST).replace(tzinfo=None)
        if (
            feature_at.date() != asof - timedelta(days=1)
            or feature_at.time() != datetime.min.time().replace(hour=9)
        ):
            raise ValueError(
                "v2 feature_date must be the prior KST D1 09:00 candle"
            )
        if res.get("btc_regime") not in VALID_BTC_REGIMES:
            raise ValueError("v2 healthy decision has invalid btc_regime")

    markets: list[str] = []
    for expected_rank, candidate in enumerate(candidates, 1):
        if not isinstance(candidate, dict):
            raise ValueError("v2 candidate must be an object")
        market = candidate.get("market")
        if (
            not isinstance(market, str)
            or _MARKET_RE.fullmatch(market) is None
        ):
            raise ValueError("v2 candidate market identity is invalid")
        markets.append(market)
        rank = candidate.get("rank")
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank != expected_rank
        ):
            raise ValueError("v2 candidate ranks must be contiguous from one")
        values = {
            field: _finite_number(candidate.get(field), f"candidate.{field}")
            for field in _CANDIDATE_FINITE_FIELDS
        }
        if values["entry_open"] <= 0:
            raise ValueError("v2 candidate.entry_open must be positive")
        if not 0 <= values["score"] <= 1:
            raise ValueError("v2 candidate.score must be in [0, 1]")
        if not 0 <= values["roc_7d_rank"] <= 1:
            raise ValueError("v2 candidate.roc_7d_rank must be in [0, 1]")
        if values["atr_pct_14"] < 0 or values["b_vol_surge"] <= 0:
            raise ValueError("v2 candidate volatility fields are invalid")
        if values["liq_rank_daily"] < 1:
            raise ValueError("v2 candidate.liq_rank_daily must be >= 1")
        b_ret = candidate.get("b_ret_1d")
        if b_ret is not None:
            _finite_number(b_ret, "candidate.b_ret_1d")
        if candidate.get("btc_regime") not in VALID_BTC_REGIMES:
            raise ValueError("v2 candidate.btc_regime is invalid")
        if candidate["btc_regime"] != res.get("btc_regime"):
            raise ValueError("v2 candidate/decision btc_regime mismatch")
        if candidate.get("rule_id") != PUMP_V2_RULE_ID:
            raise ValueError("v2 candidate.rule_id is invalid")
    if len(set(markets)) != len(markets):
        raise ValueError("v2 candidate markets must be unique")

    oos = res.get("oos")
    if not isinstance(oos, dict):
        raise ValueError("v2 oos provenance must be an object")
    for field in _OOS_FINITE_FIELDS:
        actual = _finite_number(oos.get(field), f"oos.{field}")
        if actual != PUMP_V2_OOS[field]:
            raise ValueError(
                f"v2 oos.{field} provenance mismatch: "
                f"{actual!r} != {PUMP_V2_OOS[field]!r}"
            )
    if "execution_provenance" in res:
        _validate_forward_provenance(res)
    # Persisted JSON must never silently contain NaN or Infinity tokens.
    json.dumps(res, ensure_ascii=False, sort_keys=True, allow_nan=False)


@contextmanager
def _exclusive_path_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    with file_lock(lock_path):
        yield


def _atomic_json(path: Path, payload: dict) -> None:
    atomic_write_json(path, payload)


def _path_entry_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _with_outer_integrity(payload: dict) -> dict:
    """Seal transport/chronology fields independently of decision_id."""
    return with_manifest_digest(
        payload,
        digest_key=_INTEGRITY_FIELD,
    )


def _validate_outer_integrity(
    payload: dict,
    *,
    document_keys: set[str],
    decision_day: date,
    kind: str,
    path: Path,
) -> None:
    expected_keys = set(document_keys)
    if decision_day >= FORWARD_EVIDENCE_ACTIVATION_DATE:
        expected_keys.add(_INTEGRITY_FIELD)
        if kind == "receipt":
            expected_keys.update(_POSTACTIVATION_RECEIPT_KEYS)
    if set(payload) != expected_keys:
        raise RuntimeError(f"v2 {kind} outer schema mismatch: {path}")
    if (
        decision_day >= FORWARD_EVIDENCE_ACTIVATION_DATE
        and not manifest_digest_matches(
            payload,
            digest_key=_INTEGRITY_FIELD,
        )
    ):
        raise RuntimeError(f"v2 {kind} outer integrity mismatch: {path}")


def _parse_aware(value: object, field: str, path: Path) -> datetime:
    if not isinstance(value, str):
        raise RuntimeError(f"invalid v2 receipt {field}: {path}")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise RuntimeError(f"invalid v2 receipt {field}: {path}") from exc
    if parsed.tzinfo is None:
        raise RuntimeError(f"naive v2 receipt {field}: {path}")
    return parsed


def _validate_receipt(payload: dict, res: dict, path: Path) -> None:
    try:
        decision_day = date.fromisoformat(str(res.get("asof", "")))
    except ValueError as exc:
        raise RuntimeError(f"v2 receipt asof invalid: {path}") from exc
    _validate_outer_integrity(
        payload,
        document_keys=_RECEIPT_DOCUMENT_KEYS,
        decision_day=decision_day,
        kind="receipt",
        path=path,
    )
    if payload.get("schema") != PUMP_V2_RECEIPT_SCHEMA:
        raise RuntimeError(f"unsupported v2 receipt schema: {path}")
    if payload.get("asof") != res.get("asof"):
        raise RuntimeError(f"v2 receipt asof mismatch: {path}")
    decision = payload.get("decision")
    if not isinstance(decision, dict):
        raise RuntimeError(f"v2 receipt decision payload missing: {path}")
    _validate_decision_result(decision)
    if decision_day >= FORWARD_EVIDENCE_ACTIVATION_DATE:
        try:
            _validate_forward_provenance(decision)
        except ValueError as exc:
            raise RuntimeError(
                f"v2 receipt forward provenance invalid: {path}"
            ) from exc
    if payload.get("decision_id") != _decision_id(decision):
        raise RuntimeError(f"v2 receipt decision checksum mismatch: {path}")
    if not isinstance(payload.get("delivery_ok"), bool):
        raise RuntimeError(f"invalid v2 receipt delivery_ok: {path}")
    attempted = _parse_aware(payload.get("attempted_at"), "attempted_at", path)
    recorded = _parse_aware(payload.get("recorded_at"), "recorded_at", path)
    sent_value = payload.get("sent_at")
    sent = _parse_aware(sent_value, "sent_at", path) if sent_value else None
    if payload["delivery_ok"] != (sent is not None):
        raise RuntimeError(f"invalid v2 receipt sent_at contract: {path}")
    if attempted > recorded or (
        decision_day < FORWARD_EVIDENCE_ACTIVATION_DATE
        and sent is not None
        and not attempted <= sent <= recorded
    ):
        raise RuntimeError(f"invalid v2 receipt chronology: {path}")
    error = payload.get("error")
    if error is not None and not isinstance(error, str):
        raise RuntimeError(f"invalid v2 receipt error: {path}")
    if decision_day >= FORWARD_EVIDENCE_ACTIVATION_DATE:
        if payload["delivery_ok"] and error is not None:
            raise RuntimeError(f"successful v2 receipt has an error: {path}")
        if not payload["delivery_ok"] and (
            not isinstance(error, str) or not error.strip()
        ):
            raise RuntimeError(
                f"failed v2 receipt has no explicit error: {path}"
            )
        try:
            validate_telegram_transport_evidence(
                payload,
                decision_day=decision_day,
                live_start=LIVE_RUN_START,
                live_end=LIVE_RUN_END,
                delivery_ok=payload["delivery_ok"],
                attempted_at=attempted,
                sent_at=sent,
                recorded_at=recorded,
                path=path,
            )
        except DeliveryReceiptError as exc:
            raise RuntimeError(
                f"invalid v2 Telegram server evidence: {path}: {exc}"
            ) from exc


def _receipt_path(res: dict, receipt_root: str | Path) -> Path:
    return Path(receipt_root) / f"{res['asof']}.json"


def _decision_path(res: dict, decision_root: str | Path) -> Path:
    return Path(decision_root) / f"{res['asof']}.json"


def _validate_decision_document(payload: dict, res: dict, path: Path) -> None:
    decision = payload.get("decision")
    if not isinstance(decision, dict):
        raise RuntimeError(f"v2 decision payload missing: {path}")
    if (
        payload.get("asof") != decision.get("asof")
        or path.stem != decision.get("asof")
        or decision.get("asof") != res.get("asof")
    ):
        raise RuntimeError(f"v2 decision asof mismatch: {path}")
    expected_id = _decision_id(decision)
    if payload.get("decision_id") != expected_id:
        raise RuntimeError(f"v2 decision checksum mismatch: {path}")
    _validate_decision_result(decision)
    decision_day = date.fromisoformat(decision["asof"])
    _validate_outer_integrity(
        payload,
        document_keys=_DECISION_DOCUMENT_KEYS,
        decision_day=decision_day,
        kind="decision",
        path=path,
    )
    schema = payload.get("schema")
    if schema == PUMP_V2_DECISION_SCHEMA:
        try:
            _validate_forward_provenance(decision)
        except ValueError as exc:
            raise RuntimeError(
                f"v2 forward decision provenance invalid: {path}"
            ) from exc
    elif (
        schema == PUMP_V2_LEGACY_DECISION_SCHEMA
        and decision_day < FORWARD_EVIDENCE_ACTIVATION_DATE
    ):
        pass
    else:
        raise RuntimeError(f"unsupported v2 decision schema: {path}")
    if expected_id != _decision_id(res):
        raise RuntimeError(
            f"a different v2 decision already exists for asof={res['asof']}"
        )
    _validate_decision_result(res)
    recorded = _parse_aware(payload.get("recorded_at"), "recorded_at", path)
    if decision_day >= FORWARD_EVIDENCE_ACTIVATION_DATE:
        recorded_kst = recorded.astimezone(KST)
        wall_time = recorded_kst.timetz().replace(tzinfo=None)
        if (
            recorded_kst.date() != decision_day
            or not LIVE_RUN_START <= wall_time < LIVE_RUN_END
        ):
            raise RuntimeError(
                f"v2 decision recorded outside live window: {path}"
            )


def persist_decision(
    res: dict,
    *,
    decision_root: str | Path = PUMP_V2_DECISION_ROOT,
) -> Path:
    """Persist every non-dry scorer run, including a zero-selection day."""
    _validate_decision_result(res)
    decision_day = date.fromisoformat(res["asof"])
    if decision_day >= FORWARD_EVIDENCE_ACTIVATION_DATE:
        try:
            _validate_forward_provenance(res)
        except ValueError as exc:
            raise RuntimeError(
                "v2 legacy decision is not forward-valid"
            ) from exc
    path = _decision_path(res, decision_root)
    with _exclusive_path_lock(path):
        if _path_entry_exists(path):
            try:
                existing = strict_json_object(path)
            except ArtifactValidationError as exc:
                raise RuntimeError(f"v2 decision read failed: {path}") from exc
            _validate_decision_document(existing, res, path)
            return path
        if (
            decision_day >= FORWARD_EVIDENCE_ACTIVATION_DATE
            and res["execution_provenance"] != _forward_provenance(res)
        ):
            raise RuntimeError(
                "v2 forward provenance does not match current inputs"
            )
        payload = {
            "schema": (
                PUMP_V2_DECISION_SCHEMA
                if "execution_provenance" in res
                else PUMP_V2_LEGACY_DECISION_SCHEMA
            ),
            "asof": res["asof"],
            "decision_id": _decision_id(res),
            "decision": res,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        if decision_day >= FORWARD_EVIDENCE_ACTIVATION_DATE:
            payload = _with_outer_integrity(payload)
        _validate_decision_document(payload, res, path)
        _atomic_json(path, payload)
    return path


def deliver_once(
    res: dict,
    message: str,
    *,
    receipt_root: str | Path = PUMP_V2_RECEIPT_ROOT,
    live_asof: str | None = None,
    radar_verdict_path: str | Path | None = None,
) -> dict:
    """동일 결정 payload의 Telegram 부작용을 프로세스 간 한 번만 수행한다.

    Bot API가 성공을 반환한 직후 receipt fsync 전 프로세스가 죽는 극소 구간은
    외부 API에 idempotency key가 없어 제거할 수 없다. 그 외 재실행은 성공 receipt로
    차단하며, 실패 receipt는 재시도를 허용한다.
    """
    _validate_decision_result(res)
    decision_day = date.fromisoformat(str(res.get("asof", "")))
    if decision_day >= FORWARD_EVIDENCE_ACTIVATION_DATE:
        try:
            _validate_forward_provenance(res)
        except ValueError as exc:
            raise RuntimeError(
                "v2 delivery requires canonical forward provenance"
            ) from exc
    if not isinstance(message, str) or not message:
        raise ValueError("v2 delivery message must be non-empty")
    verdict_path = (
        RADAR_VERDICT_PATH
        if radar_verdict_path is None
        else radar_verdict_path
    )
    path = _receipt_path(res, receipt_root)
    decision_id = _decision_id(res)
    with _exclusive_path_lock(path):
        if _path_entry_exists(path):
            try:
                existing = strict_json_object(path)
            except ArtifactValidationError as exc:
                raise RuntimeError(f"v2 receipt read failed: {path}") from exc
            _validate_receipt(existing, res, path)
            if existing.get("decision_id") != decision_id:
                raise RuntimeError(
                    "a different v2 decision already has a delivery attempt "
                    f"for asof={res['asof']}"
                )
            if existing["delivery_ok"]:
                return existing
            if (
                decision_day >= FORWARD_EVIDENCE_ACTIVATION_DATE
                and (
                    existing.get("telegram_messages")
                    or telegram_error_is_ambiguous(existing.get("error"))
                )
            ):
                raise RuntimeError(
                    "v2 receipt records partial/ambiguous Telegram delivery; "
                    "automatic retry would duplicate delivered chunks"
                )

        with radar_send_guard(
            path=verdict_path,
            clock=_now_kst,
            boundary_check=(
                (
                    lambda observed: _assert_live_run_window(
                        cast(str, live_asof),
                        now=observed,
                    )
                )
                if live_asof is not None
                else None
            ),
        ):
            attempted_at = datetime.now(timezone.utc).isoformat()
            transport: TelegramSendResult | None = None
            try:
                if decision_day >= FORWARD_EVIDENCE_ACTIVATION_DATE:
                    transport = send_telegram_with_receipt(
                        message,
                        deadline=_live_run_deadline(str(res["asof"])),
                        clock=_now_kst,
                    )
                    delivery_ok = transport.delivery_ok
                    error = transport.error
                else:
                    delivery_ok = bool(send_telegram(message))
                    error = (
                        None
                        if delivery_ok
                        else "send_telegram returned false"
                    )
            except Exception as exc:  # noqa: BLE001
                if decision_day >= FORWARD_EVIDENCE_ACTIVATION_DATE:
                    raise
                delivery_ok = False
                error = f"{type(exc).__name__}: {exc}"
        sent_at = None
        if delivery_ok:
            if transport is not None:
                if not transport.telegram_messages:
                    raise RuntimeError(
                        "v2 Telegram success omitted server evidence"
                    )
                sent_at = max(
                    datetime.fromisoformat(item.server_date)
                    for item in transport.telegram_messages
                ).isoformat()
            else:
                sent_at = datetime.now(timezone.utc).isoformat()
        payload = {
            "schema": PUMP_V2_RECEIPT_SCHEMA,
            "asof": res["asof"],
            "decision_id": decision_id,
            "decision": res,
            "delivery_ok": delivery_ok,
            "attempted_at": attempted_at,
            "sent_at": sent_at,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "error": error,
        }
        if decision_day >= FORWARD_EVIDENCE_ACTIVATION_DATE:
            if transport is None or transport.dry_run:
                raise RuntimeError(
                    "v2 forward receipt requires Telegram server evidence"
                )
            try:
                validate_telegram_send_result(transport, message)
            except ValueError as exc:
                raise RuntimeError(
                    "v2 Telegram server evidence does not match message"
                ) from exc
            payload.update(
                {
                    "message_sha256": transport.message_sha256,
                    "chat_id_sha256": transport.chat_id_sha256,
                    "chunk_count": transport.chunk_count,
                    "telegram_messages": [
                        item.as_dict()
                        for item in transport.telegram_messages
                    ],
                }
            )
            payload = _with_outer_integrity(payload)
        _validate_receipt(payload, res, path)
        _atomic_json(path, payload)
        return payload


def _assert_ledger_rows_match_decision(
    existing: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    asof: str,
) -> None:
    if len(existing) != len(expected):
        raise RuntimeError(
            f"v2 decision row-count conflict for asof={asof}: "
            f"existing={len(existing)} expected={len(expected)}"
        )
    sort_columns = ["coin", "rank"]
    actual_identity = (
        existing[_LEDGER_IDENTITY_COLS]
        .sort_values(sort_columns)
        .reset_index(drop=True)
    )
    expected_identity = (
        expected[_LEDGER_IDENTITY_COLS]
        .sort_values(sort_columns)
        .reset_index(drop=True)
    )
    # CSV round-trips represent an absent object scalar as NaN while the
    # freshly built frame uses None.  They are the same missing value, not an
    # immutable-decision conflict.
    actual_identity = actual_identity.astype(object).where(
        pd.notna(actual_identity),
        None,
    )
    expected_identity = expected_identity.astype(object).where(
        pd.notna(expected_identity),
        None,
    )
    try:
        pd.testing.assert_frame_equal(
            actual_identity,
            expected_identity,
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as exc:
        raise RuntimeError(
            f"v2 immutable decision row conflict for asof={asof}"
        ) from exc


def _missing_scalar(value: object) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _parse_delivery_flag(value: object, *, asof: str) -> bool | None:
    if _missing_scalar(value):
        return None
    normalized = str(value).strip().lower()
    if normalized not in {"true", "false"}:
        raise RuntimeError(f"v2 delivery_ok invalid for asof={asof}")
    return normalized == "true"


def _validate_sent_at(value: object, *, asof: str) -> str | None:
    if _missing_scalar(value):
        return None
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"v2 sent_at invalid for asof={asof}")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise RuntimeError(f"v2 sent_at invalid for asof={asof}") from exc
    if parsed.tzinfo is None:
        raise RuntimeError(f"v2 sent_at must be timezone-aware for asof={asof}")
    return value


def _assert_ledger_delivery_state(
    rows: pd.DataFrame,
    *,
    asof: str,
    expected_delivery_ok: bool,
    expected_sent_at: str | None,
) -> None:
    cohort_flags: set[bool] = set()
    cohort_sent_at: set[str] = set()
    for _, row in rows.iterrows():
        status = str(row.get("status", ""))
        if status not in {"not_delivered", "open", "no_data", "closed"}:
            raise RuntimeError(
                f"v2 ledger status invalid for asof={asof}: {status!r}"
            )
        delivery = _parse_delivery_flag(row.get("delivery_ok"), asof=asof)
        sent = _validate_sent_at(row.get("sent_at"), asof=asof)
        if status == "not_delivered":
            if delivery is True or sent is not None:
                raise RuntimeError(
                    f"v2 not_delivered metadata conflict for asof={asof}"
                )
        elif delivery is not True or sent is None:
            raise RuntimeError(
                f"v2 delivered status metadata conflict for asof={asof}"
            )
        if delivery is not None:
            cohort_flags.add(delivery)
        if sent is not None:
            cohort_sent_at.add(sent)
    if len(cohort_flags) > 1 or len(cohort_sent_at) > 1:
        raise RuntimeError(f"v2 delivery cohort inconsistent for asof={asof}")
    if expected_delivery_ok:
        expected = _validate_sent_at(expected_sent_at, asof=asof)
        if expected is None:
            raise ValueError("delivery_ok and sent_at contract mismatch")
        if cohort_sent_at and cohort_sent_at != {expected}:
            raise RuntimeError(f"v2 sent_at conflict for asof={asof}")


def append_ledger(
    res: dict,
    ledger_path: str,
    dry_run: bool,
    *,
    decision_completed_at: str | None = None,
    delivery_ok: bool = False,
    sent_at: str | None = None,
    receipt_path: str | None = None,
) -> None:
    _validate_decision_result(res)
    cands = res.get("candidates", [])
    if not cands:
        log.info("no v2 candidates — ledger skip")
        return
    if delivery_ok != (sent_at is not None):
        raise ValueError("delivery_ok and sent_at contract mismatch")
    decision_day = date.fromisoformat(str(res.get("asof", "")))
    if decision_day >= FORWARD_EVIDENCE_ACTIVATION_DATE and not dry_run:
        try:
            _validate_forward_provenance(res)
        except ValueError as exc:
            raise RuntimeError(
                "canonical v2 ledger requires forward provenance"
            ) from exc
        if decision_completed_at is None:
            raise ValueError(
                "canonical v2 ledger requires immutable decision completion time"
            )
        _parse_aware(
            decision_completed_at,
            "decision_completed_at",
            Path(receipt_path or ledger_path),
        )
    if sent_at is not None:
        _validate_sent_at(sent_at, asof=res["asof"])
    decision_id = _decision_id(res)
    rows = []
    for c in cands:
        rows.append({
            "date": res["asof"], "coin": c["market"], "rank": c["rank"],
            "score": c["score"], "pump_prob": OOS_HIT_PCT / 100.0,
            "pump_prob_pct": f"{OOS_HIT_PCT:.1f}%",
            "dump_risk_flag": False, "btc_regime": c.get("btc_regime", "unknown"),
            "entry_open": c["entry_open"], "sl_pct": SL_PCT, "tp_pct": TP_PCT,
            "status": "open" if delivery_ok else "not_delivered",
            "calibration_source": "binance_leadlag_v1_oos",
            "snapshot_id": decision_id,
            "snapshot_path": receipt_path,
            "decision_completed_at": decision_completed_at,
            "delivery_ok": delivery_ok,
            "sent_at": sent_at,
            "p_up20": OOS_HIT_PCT / 100.0,
            "model_id": res["model_id"], "rule_version": res["rule_version"],
            "rule_id": c["rule_id"], "feature_date": res.get("feature_date"),
            "liq_rank_daily": c["liq_rank_daily"], "roc_7d": c["roc_7d"],
            "roc_7d_rank": c["roc_7d_rank"], "atr_pct_14": c["atr_pct_14"],
            "log_return_1d": c["log_return_1d"],
            "b_vol_surge": c["b_vol_surge"], "b_ret_1d": c.get("b_ret_1d"),
        })
    new = pd.DataFrame(rows)
    for col in PUMP_V2_LEDGER_COLS:
        if col not in new.columns:
            new[col] = pd.NA
    new = new[PUMP_V2_LEDGER_COLS]

    if dry_run:
        log.info("[dry-run] would append %d rows to %s", len(new), ledger_path)
        return

    p = Path(ledger_path)
    with ledger_lock(p):
        if p.exists():
            existing = pd.read_csv(p)
            for col in PUMP_V2_LEDGER_COLS:
                if col not in existing.columns:
                    existing[col] = pd.NA
            for col in (
                "snapshot_id",
                "snapshot_path",
                "delivery_ok",
                "sent_at",
            ):
                existing[col] = existing[col].astype(object)
            ordered = [c for c in PUMP_V2_LEDGER_COLS if c in existing.columns]
            extras = [c for c in existing.columns if c not in ordered]
            existing = existing[ordered + extras]
            same_day = existing["date"].astype(str) == res["asof"]
            if same_day.any():
                _assert_ledger_rows_match_decision(
                    existing.loc[same_day],
                    new,
                    asof=res["asof"],
                )
                _assert_ledger_delivery_state(
                    existing.loc[same_day],
                    asof=res["asof"],
                    expected_delivery_ok=delivery_ok,
                    expected_sent_at=sent_at,
                )
                ids = existing.loc[same_day, "snapshot_id"].dropna().astype(str)
                if not ids.empty and set(ids) != {decision_id}:
                    raise RuntimeError(
                        f"v2 snapshot identity conflict for asof={res['asof']}"
                    )
                if receipt_path is not None:
                    paths = (
                        existing.loc[same_day, "snapshot_path"]
                        .dropna()
                        .astype(str)
                    )
                    if not paths.empty and set(paths) != {receipt_path}:
                        raise RuntimeError(
                            f"v2 snapshot path conflict for asof={res['asof']}"
                        )
                if delivery_ok:
                    promotable = same_day & (
                        existing["status"].astype(str) == "not_delivered"
                    )
                    existing.loc[promotable, "status"] = "open"
                    existing.loc[same_day, "snapshot_id"] = decision_id
                    existing.loc[same_day, "snapshot_path"] = receipt_path
                    existing.loc[same_day, "delivery_ok"] = True
                    existing.loc[same_day, "sent_at"] = sent_at
                    atomic_write_csv(existing, p)
                    log.info("backfilled successful delivery metadata -> %s", p)
                else:
                    log.warning(
                        "asof=%s already in %s — skip append (idempotent)",
                        res["asof"],
                        p,
                    )
                return
            combined = pd.concat([existing, new], ignore_index=True)
        else:
            combined = new
        atomic_write_csv(combined, p)
        log.info("appended %d rows -> %s (total %d)", len(new), p, len(combined))


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


def main() -> int:
    _configure_cli_logging()
    ap = argparse.ArgumentParser(description="PUMP hunter v2 daily — shadow + telegram radar")
    ap.add_argument("--asof", type=str, default=None)
    ap.add_argument("--ledger", type=str, default=PUMP_V2_LEDGER)
    ap.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES)
    ap.add_argument("--send-telegram", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="ledger 기록 X")
    ap.add_argument(
        "--receipt-root",
        type=str,
        default=PUMP_V2_RECEIPT_ROOT,
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--decision-root",
        type=str,
        default=PUMP_V2_DECISION_ROOT,
        help=argparse.SUPPRESS,
    )
    args = ap.parse_args()
    if args.max_candidates <= 0:
        ap.error("--max-candidates must be > 0")

    asof = args.asof or _today_kst()
    if not args.dry_run:
        try:
            _assert_canonical_forward_request(
                ledger_path=args.ledger,
                decision_root=args.decision_root,
                receipt_root=args.receipt_root,
                max_candidates=args.max_candidates,
            )
            observed = _now_kst()
            _assert_live_run_window(asof, now=observed)
            assert_radar_send_allowed(
                path=RADAR_VERDICT_PATH,
                now=observed,
            )
        except (RuntimeError, ValueError) as exc:
            log.error("v2 live run rejected: %s", exc)
            return 1
    input_before = None
    if (
        not args.dry_run
        and date.fromisoformat(asof) >= FORWARD_EVIDENCE_ACTIVATION_DATE
    ):
        try:
            input_before = _current_forward_inputs()
        except RuntimeError as exc:
            log.error("forward provenance pre-snapshot failed: %s", exc)
            return 1
    res = score_pump_v2_candidates(asof, max_candidates=args.max_candidates)
    try:
        _validate_decision_result(
            res,
            expected_asof=asof,
            max_candidates=args.max_candidates,
        )
    except ValueError as exc:
        log.error("invalid scorer decision: %s", exc)
        return 1
    if not args.dry_run:
        try:
            _assert_live_run_window(asof, now=_now_kst())
            res = _with_forward_provenance(res)
            if input_before is not None and {
                "sources": res["execution_provenance"]["sources"],
                "data": res["execution_provenance"]["data"],
            } != input_before:
                raise RuntimeError(
                    "v2 source/data inputs changed during scoring"
                )
            _validate_decision_result(
                res,
                expected_asof=asof,
                max_candidates=args.max_candidates,
            )
        except (RuntimeError, ValueError) as exc:
            log.error("forward provenance construction failed: %s", exc)
            return 1
    log.info("asof=%s universe=%s binance=%s candidates=%d",
             res["asof"], res.get("universe_n"), res.get("binance_status"),
             res.get("n_candidates", 0))
    decision_path = None
    decision_completed_at = None
    if not args.dry_run:
        try:
            decision_path = persist_decision(
                res,
                decision_root=args.decision_root,
            )
            decision_manifest = strict_json_object(decision_path)
            _validate_decision_document(
                decision_manifest,
                res,
                decision_path,
            )
            decision_completed_at = str(
                decision_manifest["recorded_at"]
            )
        except Exception as exc:  # noqa: BLE001
            log.error("decision persistence failed: %s", exc)
            return 1

    msg = build_message(
        res,
        dry_run=args.dry_run or not args.send_telegram,
    )
    print(msg)

    has_cands = res.get("n_candidates", 0) > 0
    stale = str(res.get("binance_status", "ok")) != "ok"
    receipt = None
    should_send = args.send_telegram and not args.dry_run and (has_cands or stale)
    if should_send:
        try:
            receipt = deliver_once(
                res,
                msg,
                receipt_root=args.receipt_root,
                live_asof=asof,
                radar_verdict_path=RADAR_VERDICT_PATH,
            )
            if receipt["delivery_ok"]:
                log.info("telegram sent")
            else:
                log.error("telegram not sent")
        except Exception as e:  # noqa: BLE001
            log.error("telegram fail: %s", e)
            return 1
    elif args.send_telegram and not args.dry_run:
        log.info("telegram skipped: 후보 0 + binance ok (무소음 정책)")

    delivery_ok = bool(receipt and receipt["delivery_ok"])
    sent_at = receipt.get("sent_at") if receipt else None
    try:
        append_ledger(
            res,
            args.ledger,
            args.dry_run,
            decision_completed_at=decision_completed_at,
            delivery_ok=delivery_ok,
            sent_at=sent_at,
            receipt_path=(
                str(decision_path) if decision_path is not None else None
            ),
        )
    except Exception as exc:  # noqa: BLE001
        log.error("ledger append failed: %s", exc)
        return 1
    return 0 if receipt is None or receipt["delivery_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Strict pump-v2 decision/receipt/ledger provenance validation.

The scorer decision is the immutable source of candidate identity.  Delivery
receipts prove whether a candidate cohort reached Telegram, while the ledger
adds mutable close outcomes.  Consumers must not treat a CLOSED CSV row as
genuine unless all three layers agree.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, cast

from ops.artifact_provenance import (
    ArtifactValidationError,
    strict_json_object,
)
from ops.close_input_gate import CLOSE_EVIDENCE_ACTIVATION_DATE
from scripts.pump_detector_v2_today import (
    KST,
    LIVE_RUN_END,
    LIVE_RUN_START,
    OOS_HIT_PCT,
    PUMP_V2_DECISION_SCHEMA,
    PUMP_V2_RECEIPT_SCHEMA,
    SL_PCT,
    TP_PCT,
    _validate_decision_document,
    _validate_receipt,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LEDGER_PATH = (
    PROJECT_ROOT / "output" / "shadow_ledger_pump_hunter_v2.csv"
)
DEFAULT_DECISION_ROOT = PROJECT_ROOT / "output" / "pump_v2_decisions"
DEFAULT_RECEIPT_ROOT = PROJECT_ROOT / "output" / "pump_v2_receipts"
# The frozen 2026-09-01 scorecard predates canonical decision manifests.
# These are the exact scorecard-relevant fields of all 205 legacy CLOSED rows
# through 2026-07-25, locked when the evidence contract was activated.  New
# strict evidence starts on 2026-07-27; 2026-07-26 was a verified zero-pick
# legacy run and therefore added no ledger row.
LEGACY_SCORECARD_BASELINE_ROWS = 205
LEGACY_SCORECARD_BASELINE_SHA256 = (
    "ac01dddebbd1e70fe9d6f034292470982ecd237bd684422946b938fc82794451"
)


class V2ProvenanceError(ValueError):
    """Canonical pump-v2 evidence is missing, corrupt, or contradictory."""


@dataclass(frozen=True)
class V2ProvenanceAudit:
    healthy_dates: tuple[date, ...]
    healthy_zero_pick_dates: tuple[date, ...]
    healthy_candidate_dates: tuple[date, ...]
    delivery_success_dates: tuple[date, ...]
    delivery_failed_dates: tuple[date, ...]
    verified_closed_positions: tuple[tuple[date, str], ...]
    legacy_scorecard_rows: int
    legacy_scorecard_sha256: str
    legacy_baseline_verified: bool

    @property
    def recall_date_basis(self) -> str:
        return (
            "post_contract_validated_healthy_decisions"
            "(delivery_independent)+delivery_verified_closed_rows;"
            "legacy_dates_excluded"
        )


def _read_json(path: Path, *, kind: str) -> dict[str, Any]:
    try:
        payload = strict_json_object(path)
    except ArtifactValidationError as exc:
        raise V2ProvenanceError(f"invalid v2 {kind}: {path}") from exc
    return payload


def _path_day(path: Path, *, kind: str) -> date:
    try:
        return date.fromisoformat(path.stem)
    except ValueError as exc:
        raise V2ProvenanceError(
            f"invalid v2 {kind} filename: {path.name!r}"
        ) from exc


def _aware_timestamp(
    value: object,
    *,
    field: str,
    path: Path,
) -> datetime:
    if not isinstance(value, str):
        raise V2ProvenanceError(f"v2 {field} is invalid: {path}")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise V2ProvenanceError(f"v2 {field} is invalid: {path}") from exc
    if parsed.tzinfo is None:
        raise V2ProvenanceError(f"v2 {field} must be timezone-aware: {path}")
    return parsed


def _validate_decision_chronology(
    payload: Mapping[str, object],
    *,
    day: date,
    path: Path,
) -> datetime:
    recorded = _aware_timestamp(
        payload.get("recorded_at"),
        field="decision recorded_at",
        path=path,
    )
    local = recorded.astimezone(KST)
    wall_time = local.timetz().replace(tzinfo=None)
    if (
        local.date() != day
        or not LIVE_RUN_START <= wall_time < LIVE_RUN_END
    ):
        raise V2ProvenanceError(
            f"v2 decision recorded outside canonical live window: {path}"
        )
    return recorded


def _validate_receipt_chronology(
    payload: Mapping[str, object],
    *,
    day: date,
    decision_recorded_at: datetime,
    path: Path,
) -> None:
    attempted = _aware_timestamp(
        payload.get("attempted_at"),
        field="receipt attempted_at",
        path=path,
    )
    recorded = _aware_timestamp(
        payload.get("recorded_at"),
        field="receipt recorded_at",
        path=path,
    )
    sent_value = payload.get("sent_at")
    sent = (
        _aware_timestamp(
            sent_value,
            field="receipt sent_at",
            path=path,
        )
        if sent_value is not None
        else None
    )
    attempted_local = attempted.astimezone(KST)
    attempted_wall = attempted_local.timetz().replace(tzinfo=None)
    event_times = [attempted, recorded]
    if sent is not None:
        event_times.append(sent)
    if (
        attempted_local.date() != day
        or not LIVE_RUN_START <= attempted_wall < LIVE_RUN_END
        or any(value.astimezone(KST).date() != day for value in event_times)
        or decision_recorded_at > attempted
    ):
        raise V2ProvenanceError(
            f"v2 receipt chronology is outside canonical run: {path}"
        )


def _is_missing(value: object) -> bool:
    if value is None or value == "":
        return True
    return isinstance(value, float) and math.isnan(value)


def _as_bool(value: object, *, field: str, day: date) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise V2ProvenanceError(
        f"v2 ledger {field} invalid for {day}: {value!r}"
    )


def _equal_value(actual: object, expected: object) -> bool:
    if expected is None:
        return _is_missing(actual)
    if isinstance(expected, bool):
        try:
            return _as_bool(actual, field="identity", day=date.min) is expected
        except V2ProvenanceError:
            return False
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if isinstance(actual, bool) or _is_missing(actual):
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


def _resolved_path(value: object) -> Path | None:
    if _is_missing(value):
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve(strict=False)


def _candidate_key(row: Mapping[str, object], *, day: date) -> tuple[str, int]:
    coin = str(row.get("coin", "")).strip()
    rank_value = row.get("rank")
    try:
        rank_number = float(cast(Any, rank_value))
    except (TypeError, ValueError) as exc:
        raise V2ProvenanceError(
            f"v2 ledger candidate rank invalid for {day}: {rank_value!r}"
        ) from exc
    if (
        not coin
        or not math.isfinite(rank_number)
        or not rank_number.is_integer()
        or rank_number < 1
    ):
        raise V2ProvenanceError(
            f"v2 ledger candidate identity invalid for {day}"
        )
    return coin, int(rank_number)


def _validate_ledger_candidate(
    row: Mapping[str, object],
    *,
    day: date,
    candidate: Mapping[str, object],
    decision: Mapping[str, object],
    decision_recorded_at: datetime,
    decision_id: str,
    decision_path: Path,
    receipt: Mapping[str, object] | None,
) -> None:
    expected = {
        "coin": candidate.get("market"),
        "rank": candidate.get("rank"),
        "score": candidate.get("score"),
        "pump_prob": OOS_HIT_PCT / 100.0,
        "pump_prob_pct": f"{OOS_HIT_PCT:.1f}%",
        "dump_risk_flag": False,
        "btc_regime": candidate.get("btc_regime"),
        "entry_open": candidate.get("entry_open"),
        "sl_pct": SL_PCT,
        "tp_pct": TP_PCT,
        "calibration_source": "binance_leadlag_v1_oos",
        "snapshot_id": decision_id,
        "decision_completed_at": decision_recorded_at.isoformat(),
        "p_up20": OOS_HIT_PCT / 100.0,
        "model_id": decision.get("model_id"),
        "rule_version": decision.get("rule_version"),
        "rule_id": candidate.get("rule_id"),
        "feature_date": decision.get("feature_date"),
        "liq_rank_daily": candidate.get("liq_rank_daily"),
        "roc_7d": candidate.get("roc_7d"),
        "roc_7d_rank": candidate.get("roc_7d_rank"),
        "atr_pct_14": candidate.get("atr_pct_14"),
        "log_return_1d": candidate.get("log_return_1d"),
        "b_vol_surge": candidate.get("b_vol_surge"),
        "b_ret_1d": candidate.get("b_ret_1d"),
    }
    for field, expected_value in expected.items():
        if not _equal_value(row.get(field), expected_value):
            raise V2ProvenanceError(
                f"v2 ledger/decision mismatch for {day} "
                f"candidate={candidate.get('market')}/{candidate.get('rank')} "
                f"field={field}"
            )

    if _resolved_path(row.get("snapshot_path")) != decision_path.resolve(
        strict=False
    ):
        raise V2ProvenanceError(
            f"v2 ledger canonical decision path mismatch for {day}"
        )

    status = str(row.get("status", ""))
    if status not in {"not_delivered", "open", "no_data", "closed"}:
        raise V2ProvenanceError(
            f"v2 ledger status invalid for {day}: {status!r}"
        )
    delivery_ok = _as_bool(
        row.get("delivery_ok"),
        field="delivery_ok",
        day=day,
    )
    sent_at = None if _is_missing(row.get("sent_at")) else str(row["sent_at"])
    receipt_ok = bool(receipt and receipt.get("delivery_ok") is True)
    receipt_sent_at = (
        str(receipt.get("sent_at"))
        if receipt is not None and receipt.get("sent_at") is not None
        else None
    )
    if status == "not_delivered":
        if delivery_ok or sent_at is not None or receipt_ok:
            raise V2ProvenanceError(
                f"v2 not_delivered provenance conflict for {day}"
            )
    elif (
        not delivery_ok
        or not receipt_ok
        or sent_at is None
        or sent_at != receipt_sent_at
    ):
        raise V2ProvenanceError(
            f"v2 {status} row lacks successful matching receipt for {day}"
        )


def _legacy_scorecard_digest(
    rows: Iterable[Mapping[str, object]],
) -> tuple[int, str]:
    """Hash only the legacy fields consumed by the frozen scorecard."""
    canonical: list[tuple[str, str, str, str, str]] = []
    for row in rows:
        day = str(row.get("date", ""))
        coin = str(row.get("coin", "")).strip()
        status = str(row.get("status", ""))
        regime = str(row.get("btc_regime", ""))
        if not day or not coin or not status or not regime:
            raise V2ProvenanceError(
                "legacy v2 scorecard row has missing identity/status/regime"
            )
        try:
            realized = float(cast(Any, row.get("realized_pct")))
        except (TypeError, ValueError) as exc:
            raise V2ProvenanceError(
                f"legacy v2 scorecard return invalid: {day}/{coin}"
            ) from exc
        if not math.isfinite(realized):
            raise V2ProvenanceError(
                f"legacy v2 scorecard return non-finite: {day}/{coin}"
            )
        canonical.append(
            (day, coin, status, realized.hex(), regime)
        )
    canonical.sort()
    payload = json.dumps(
        canonical,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return len(canonical), hashlib.sha256(payload).hexdigest()


def validate_v2_provenance(
    ledger_rows: Iterable[Mapping[str, object]],
    *,
    asof: date,
    decision_root: Path = DEFAULT_DECISION_ROOT,
    receipt_root: Path = DEFAULT_RECEIPT_ROOT,
    activation_date: date = CLOSE_EVIDENCE_ACTIVATION_DATE,
    enforce_legacy_baseline: bool = False,
) -> V2ProvenanceAudit:
    """Validate every completed v2 decision cohort consumed by reporting.

    Healthy decision dates are observation opportunities independent of
    Telegram delivery.  CLOSED rows, however, require an exact successful
    receipt and immutable decision/candidate match.
    """
    decisions: dict[
        date,
        tuple[dict[str, Any], dict[str, Any], Path, datetime],
    ] = {}
    if decision_root.exists():
        for path in sorted(decision_root.glob("*.json")):
            day = _path_day(path, kind="decision")
            if day >= asof or day < activation_date:
                continue
            payload = _read_json(path, kind="decision")
            decision = payload.get("decision")
            if (
                payload.get("schema") != PUMP_V2_DECISION_SCHEMA
                or not isinstance(decision, dict)
            ):
                raise V2ProvenanceError(
                    f"v2 decision manifest identity invalid: {path}"
                )
            try:
                _validate_decision_document(payload, decision, path)
            except (RuntimeError, TypeError, ValueError) as exc:
                raise V2ProvenanceError(
                    f"v2 decision manifest invalid: {path}: {exc}"
                ) from exc
            decision_recorded_at = _validate_decision_chronology(
                payload,
                day=day,
                path=path,
            )
            decisions[day] = (
                payload,
                decision,
                path,
                decision_recorded_at,
            )

    receipts: dict[date, dict[str, Any]] = {}
    if receipt_root.exists():
        for path in sorted(receipt_root.glob("*.json")):
            day = _path_day(path, kind="receipt")
            if day >= asof or day < activation_date:
                continue
            decision_entry = decisions.get(day)
            if decision_entry is None:
                raise V2ProvenanceError(
                    f"orphan v2 receipt without decision manifest: {path}"
                )
            (
                manifest_payload,
                decision,
                _decision_path,
                decision_recorded_at,
            ) = decision_entry
            payload = _read_json(path, kind="receipt")
            if payload.get("schema") != PUMP_V2_RECEIPT_SCHEMA:
                raise V2ProvenanceError(
                    f"v2 receipt schema invalid: {path}"
                )
            try:
                _validate_receipt(payload, decision, path)
            except (RuntimeError, TypeError, ValueError) as exc:
                raise V2ProvenanceError(
                    f"v2 receipt invalid: {path}: {exc}"
                ) from exc
            if payload.get("decision_id") != manifest_payload.get("decision_id"):
                raise V2ProvenanceError(
                    f"v2 receipt/decision identity mismatch: {path}"
                )
            _validate_receipt_chronology(
                payload,
                day=day,
                decision_recorded_at=decision_recorded_at,
                path=path,
            )
            receipts[day] = payload

    rows_by_day: dict[date, list[Mapping[str, object]]] = {}
    legacy_rows: list[Mapping[str, object]] = []
    for row in ledger_rows:
        try:
            day = date.fromisoformat(str(row.get("date", "")))
        except ValueError as exc:
            raise V2ProvenanceError(
                f"v2 ledger date invalid: {row.get('date')!r}"
            ) from exc
        if day >= asof:
            continue
        if day < activation_date:
            legacy_rows.append(row)
            continue
        rows_by_day.setdefault(day, []).append(row)

    legacy_n, legacy_sha256 = _legacy_scorecard_digest(legacy_rows)
    legacy_verified = (
        legacy_n == LEGACY_SCORECARD_BASELINE_ROWS
        and legacy_sha256 == LEGACY_SCORECARD_BASELINE_SHA256
    )
    if enforce_legacy_baseline and not legacy_verified:
        raise V2ProvenanceError(
            "legacy v2 scorecard baseline mismatch: "
            f"rows={legacy_n}/{LEGACY_SCORECARD_BASELINE_ROWS} "
            f"sha256={legacy_sha256}/"
            f"{LEGACY_SCORECARD_BASELINE_SHA256}"
        )

    verified_closed: list[tuple[date, str]] = []
    for day, rows in rows_by_day.items():
        entry = decisions.get(day)
        if entry is None:
            raise V2ProvenanceError(
                f"v2 ledger cohort has no canonical decision manifest: {day}"
            )
        (
            manifest_payload,
            decision,
            decision_path,
            decision_recorded_at,
        ) = entry
        candidates = decision["candidates"]
        expected_candidates = {
            (str(candidate["market"]), int(candidate["rank"])): candidate
            for candidate in candidates
        }
        actual_keys: set[tuple[str, int]] = set()
        for row in rows:
            key = _candidate_key(row, day=day)
            if key in actual_keys:
                raise V2ProvenanceError(
                    f"duplicate v2 ledger candidate for {day}: {key}"
                )
            actual_keys.add(key)
            candidate = expected_candidates.get(key)
            if candidate is None:
                raise V2ProvenanceError(
                    f"v2 ledger candidate not in decision manifest: {day}/{key}"
                )
            _validate_ledger_candidate(
                row,
                day=day,
                candidate=candidate,
                decision=decision,
                decision_recorded_at=decision_recorded_at,
                decision_id=str(manifest_payload["decision_id"]),
                decision_path=decision_path,
                receipt=receipts.get(day),
            )
            if str(row.get("status")) == "closed":
                verified_closed.append((day, key[0]))
        if actual_keys != set(expected_candidates):
            raise V2ProvenanceError(
                f"v2 ledger/decision candidate cohort mismatch for {day}: "
                f"ledger={sorted(actual_keys)} "
                f"decision={sorted(expected_candidates)}"
            )

    healthy_dates: list[date] = []
    healthy_zero: list[date] = []
    healthy_candidates: list[date] = []
    delivered: list[date] = []
    delivery_failed: list[date] = []
    for day, (_payload, decision, _path, _recorded_at) in decisions.items():
        if decision.get("binance_status") != "ok":
            continue
        healthy_dates.append(day)
        candidates = decision["candidates"]
        if not candidates:
            healthy_zero.append(day)
            continue
        healthy_candidates.append(day)
        expected_keys = {
            (str(candidate["market"]), int(candidate["rank"]))
            for candidate in candidates
        }
        actual_keys = {
            _candidate_key(row, day=day)
            for row in rows_by_day.get(day, [])
        }
        if actual_keys != expected_keys:
            raise V2ProvenanceError(
                f"v2 healthy candidate decision lacks exact ledger cohort "
                f"for {day}"
            )
        receipt = receipts.get(day)
        if receipt is not None and receipt.get("delivery_ok") is True:
            delivered.append(day)
        else:
            delivery_failed.append(day)

    return V2ProvenanceAudit(
        healthy_dates=tuple(sorted(healthy_dates)),
        healthy_zero_pick_dates=tuple(sorted(healthy_zero)),
        healthy_candidate_dates=tuple(sorted(healthy_candidates)),
        delivery_success_dates=tuple(sorted(delivered)),
        delivery_failed_dates=tuple(sorted(delivery_failed)),
        verified_closed_positions=tuple(sorted(verified_closed)),
        legacy_scorecard_rows=legacy_n,
        legacy_scorecard_sha256=legacy_sha256,
        legacy_baseline_verified=legacy_verified,
    )

"""Fail-closed validation for daily shadow-ledger close inputs.

The immutable score snapshot/decision is the source of candidate identity.
A missing ledger is healthy only when that canonical evidence proves a
zero-candidate decision for the same day.  If candidates existed, the exact
same-day ledger cohort must be present before a close job can continue.
"""
from __future__ import annotations

# ruff: noqa: E402
# Protocol fd sealing intentionally runs before every project import below.

import argparse
import csv
import io
import logging
import math
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping, cast

# NUL 프로토콜 모드면 여기서 fd1 이 봉인된다.  아래 prelude 모듈들의
# import 시점 오염까지 막아야 하므로 반드시 이들보다 먼저 import 할 것.
from ops._stdout_seal import seal_nul_protocol_stdout

PROTOCOL_STDOUT = (
    seal_nul_protocol_stdout(sys.argv[1:])
    if __name__ == "__main__"
    else None
)

from notifier.delivery_receipt import (
    DeliveryReceiptError,
    read_delivery_receipt,
)
from ops.artifact_provenance import (
    ArtifactSourceChangedError,
    ArtifactValidationError,
    file_identity,
    sha256_bytes,
    strict_json_object,
)
from scripts.pump_detector_today import (
    _validate_decision_document as _validate_v1_decision,
)
from scripts.pump_detector_v2_today import (
    _validate_decision_document as _validate_v2_decision,
)
from scripts.pump_detector_v2_today import (
    _validate_receipt as _validate_v2_receipt,
)
from signals.recommend_snapshot import SNAPSHOT_SCHEMA_VERSION, load_snapshot
from signals.pump_detector_v1 import (
    EST_PUMP20_PROB as PUMP_V1_PROB,
)
from signals.pump_detector_v1 import SL_PCT as PUMP_V1_SL
from signals.pump_detector_v1 import TP_PCT as PUMP_V1_TP
from signals.pump_detector_v2 import SL_PCT as PUMP_V2_SL
from signals.pump_detector_v2 import TP_PCT as PUMP_V2_TP


PROJECT_ROOT = Path(__file__).resolve().parent.parent
# This contract landed after the 2026-07-26 daily runners had already
# completed.  2026-07-27 is therefore the first decision date on which every
# runner is required to emit canonical evidence.
CLOSE_EVIDENCE_ACTIVATION_DATE = date(2026, 7, 27)
ALLOWED_LEDGER_STATUSES = frozenset(
    {"open", "no_data", "not_delivered", "closed"}
)
# 2026-07-26 이음매(구 writer가 아침에 적재, immutable schema는 저녁에 도착)
# 하루에만 존재할 수 있는 컬럼.  pre-activation 행에서 이 컬럼이 아예 없거나
# 이후 append 가 pd.NA 로 채워 공란인 경우만 관용한다.  다른 컬럼은 예외 없음.
SEAM_TOLERATED_COLUMNS = frozenset({"decision_completed_at"})

log = logging.getLogger(__name__)
# 이 모듈의 stdout 은 NUL 레코드 프로토콜(--output-format nul)로 셸이 기계
# 파싱한다.  import 되는 다른 스크립트가 root logging 을 stdout 으로 구성해도
# 이 로거의 경고가 레코드 스트림을 오염시키지 않도록 stderr 로 고정한다
# (2026-07-28 09:30 close 3채널이 이 오염으로 전멸했던 회귀의 수리).
if not log.handlers:
    _stderr_handler = logging.StreamHandler(sys.stderr)
    _stderr_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    log.addHandler(_stderr_handler)
    log.propagate = False


class CloseInputError(RuntimeError):
    """Canonical evidence and the ledger cohort are missing or inconsistent."""


class MissingCloseEvidenceError(CloseInputError):
    """The canonical same-day artifact does not exist."""


@dataclass(frozen=True)
class CohortSpec:
    ledger_name: str
    evidence_kind: str
    evidence_name: str
    slot: str | None = None
    ranking: str | None = None
    model_id: str | None = None
    requires_successful_delivery: bool = False


@dataclass(frozen=True)
class ExpectedCohort:
    asof: str
    evidence_id: str
    evidence_path: Path
    candidates: tuple[tuple[str, int], ...]
    immutable_rows: tuple[Mapping[str, object], ...] = ()
    validate_delivery_metadata: bool = False
    delivery_ok: bool | None = None
    sent_at: str | None = None


COHORTS: dict[str, CohortSpec] = {
    "r1-open": CohortSpec(
        ledger_name="shadow_ledger_recommend.csv",
        evidence_kind="recommend_snapshot",
        evidence_name="recommend_snapshots/{asof}/open_r1.json",
        slot="open",
        ranking="R1",
        model_id="recommend_r1_open",
        requires_successful_delivery=True,
    ),
    "r1-preopen": CohortSpec(
        ledger_name="shadow_ledger_recommend_preopen.csv",
        evidence_kind="recommend_snapshot",
        evidence_name="recommend_snapshots/{asof}/preopen_r1.json",
        slot="preopen",
        ranking="R1",
        model_id="recommend_r1_preopen",
        requires_successful_delivery=True,
    ),
    "r2": CohortSpec(
        ledger_name="shadow_ledger_recommend_r2.csv",
        evidence_kind="recommend_snapshot",
        evidence_name="recommend_snapshots/{asof}/open_r2.json",
        slot="open",
        ranking="R2",
        model_id="recommend_r2_open",
    ),
    "a1": CohortSpec(
        ledger_name="shadow_ledger_recommend_sustain.csv",
        evidence_kind="recommend_snapshot",
        evidence_name="recommend_snapshots/{asof}/open_a1.json",
        slot="open",
        ranking="A1",
        model_id="recommend_r1_sustain_open",
    ),
    "pump-v1": CohortSpec(
        ledger_name="shadow_ledger_pump_hunter.csv",
        evidence_kind="pump_v1_decision",
        evidence_name="pump_v1_decisions/{asof}.json",
    ),
    "pump-v2": CohortSpec(
        ledger_name="shadow_ledger_pump_hunter_v2.csv",
        evidence_kind="pump_v2_decision",
        evidence_name="pump_v2_decisions/{asof}.json",
    ),
}


def _recommend_immutable_rows(snapshot: Mapping[str, object]) -> tuple[dict, ...]:
    """Reconstruct the signal-owned ledger columns from a validated snapshot."""
    candidates = cast(list[dict[str, object]], snapshot["top3"])
    calibration_source = snapshot["calibration_source"]
    rows = []
    for item in candidates:
        rows.append(
            {
                "date": snapshot["asof"],
                "coin": item["coin"],
                "rank": item["rank"],
                "score": item["score"],
                "pump_prob": item["pump_prob"],
                "pump_prob_pct": item["pump_prob_pct"],
                "dump_risk_flag": item["dump_risk_flag"],
                "btc_regime": item["btc_regime"],
                "entry_open": item["entry_open"],
                "sl_pct": item["sl"],
                "tp_pct": item["tp"],
                "calibration_source": calibration_source,
                "decision_completed_at": snapshot.get(
                    "decision_completed_at"
                ),
                "p_up5": item.get("p_up5"),
                "p_up10": item.get("p_up10"),
                "p_up20": item.get("p_up20"),
                "p_dn5": item.get("p_dn5"),
                "p_dn10": item.get("p_dn10"),
                "exp_downside": item.get("exp_downside"),
                "rr_ratio": item.get("rr_ratio"),
            }
        )
    return tuple(rows)


def _pump_v1_immutable_rows(
    decision: Mapping[str, object],
    *,
    decision_recorded_at: object,
) -> tuple[dict, ...]:
    """Reconstruct every decision-owned v1 ledger column."""
    candidates = cast(list[dict[str, object]], decision["candidates"])
    rows = []
    for item in candidates:
        pump_prob = float(
            cast(Any, item.get("estimated_pump20_prob", PUMP_V1_PROB))
        )
        rows.append(
            {
                "date": decision["asof"],
                "coin": item["market"],
                "rank": item["rank"],
                "score": item["score"],
                "pump_prob": pump_prob,
                "pump_prob_pct": f"{pump_prob * 100:.1f}%",
                "dump_risk_flag": item.get("dump_risk_flag", False),
                "btc_regime": item.get("btc_regime", "unknown"),
                "entry_open": item.get("entry_open"),
                "sl_pct": PUMP_V1_SL,
                "tp_pct": PUMP_V1_TP,
                "calibration_source": "pump_rule_discovery_v1_prior",
                "decision_completed_at": decision_recorded_at,
                "p_up5": None,
                "p_up10": None,
                "p_up20": pump_prob,
                "p_dn5": None,
                "p_dn10": None,
                "exp_downside": None,
                "rr_ratio": None,
                "model_id": decision["model_id"],
                "rule_version": decision["rule_version"],
                "rule_id": item.get("rule_id"),
                "feature_date": decision["feature_date"],
                "liq_rank_daily": item.get("liq_rank_daily"),
                "roc_7d": item.get("roc_7d"),
                "roc_7d_rank": item.get("roc_7d_rank"),
                "atr_pct_14": item.get("atr_pct_14"),
                "log_return_1d": item.get("log_return_1d"),
                "pump20_rule": item.get("pump20_rule", False),
                "pump15_rule": item.get("pump15_rule", False),
                "estimated_pump15_prob": item.get(
                    "estimated_pump15_prob"
                ),
                "overheated_flag": item.get("overheated_flag", False),
            }
        )
    return tuple(rows)


def _pump_v2_immutable_rows(
    decision: Mapping[str, object],
    *,
    decision_recorded_at: object,
) -> tuple[dict, ...]:
    """Reconstruct every decision-owned v2 ledger column."""
    candidates = cast(list[dict[str, object]], decision["candidates"])
    oos = cast(Mapping[str, object], decision["oos"])
    pump_prob = float(cast(Any, oos["hit_pct"])) / 100.0
    rows = []
    for item in candidates:
        rows.append(
            {
                "date": decision["asof"],
                "coin": item["market"],
                "rank": item["rank"],
                "score": item["score"],
                "pump_prob": pump_prob,
                "pump_prob_pct": f"{pump_prob * 100:.1f}%",
                "dump_risk_flag": False,
                "btc_regime": item.get("btc_regime", "unknown"),
                "entry_open": item["entry_open"],
                "sl_pct": PUMP_V2_SL,
                "tp_pct": PUMP_V2_TP,
                "calibration_source": "binance_leadlag_v1_oos",
                "decision_completed_at": decision_recorded_at,
                "p_up5": None,
                "p_up10": None,
                "p_up20": pump_prob,
                "p_dn5": None,
                "p_dn10": None,
                "exp_downside": None,
                "rr_ratio": None,
                "model_id": decision["model_id"],
                "rule_version": decision["rule_version"],
                "rule_id": item["rule_id"],
                "feature_date": decision.get("feature_date"),
                "liq_rank_daily": item["liq_rank_daily"],
                "roc_7d": item["roc_7d"],
                "roc_7d_rank": item["roc_7d_rank"],
                "atr_pct_14": item["atr_pct_14"],
                "log_return_1d": item["log_return_1d"],
                "b_vol_surge": item["b_vol_surge"],
                "b_ret_1d": item.get("b_ret_1d"),
            }
        )
    return tuple(rows)


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        return strict_json_object(path)
    except (OSError, ArtifactValidationError) as exc:
        raise CloseInputError(f"canonical evidence is unreadable: {path}") from exc


def _directory_entry_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise CloseInputError(
            f"canonical input cannot be inspected: {path}"
        ) from exc
    return True


def _stable_csv_text(path: Path) -> str:
    try:
        before = file_identity(path, root=path.parent)
        raw = path.read_bytes()
        after = file_identity(path, root=path.parent)
    except (OSError, ArtifactSourceChangedError) as exc:
        raise CloseInputError(f"ledger is unreadable: {path}") from exc
    if (
        not before.get("exists")
        or before != after
        or before.get("sha256") != sha256_bytes(raw)
    ):
        raise CloseInputError(f"ledger changed while reading: {path}")
    try:
        return raw.decode("utf-8")
    except UnicodeError as exc:
        raise CloseInputError(f"ledger is not UTF-8: {path}") from exc


def _candidate_key(
    candidate: Mapping[str, object],
    *,
    coin_field: str,
    source: str,
) -> tuple[str, int]:
    coin = candidate.get(coin_field)
    rank_value = candidate.get("rank")
    try:
        rank_number = float(cast(Any, rank_value))
    except (TypeError, ValueError) as exc:
        raise CloseInputError(f"{source} has an invalid candidate rank") from exc
    if (
        not isinstance(coin, str)
        or not coin
        or not math.isfinite(rank_number)
        or not rank_number.is_integer()
        or rank_number < 1
    ):
        raise CloseInputError(f"{source} has an invalid candidate identity")
    return coin, int(rank_number)


def _load_expected(
    *,
    asof: str,
    cohort: str,
    output_root: Path,
) -> ExpectedCohort:
    try:
        spec = COHORTS[cohort]
    except KeyError as exc:
        raise CloseInputError(f"unsupported close cohort: {cohort}") from exc
    evidence_path = output_root / spec.evidence_name.format(asof=asof)
    if not _directory_entry_exists(evidence_path):
        raise MissingCloseEvidenceError(
            f"{cohort} canonical same-day evidence missing: {evidence_path}"
        )

    delivery_ok: bool | None = None
    sent_at: str | None = None
    if spec.evidence_kind == "recommend_snapshot":
        try:
            snapshot = load_snapshot(
                evidence_path,
                asof=asof,
                slot=spec.slot,
                ranking=spec.ranking,
                limit_markets=None,
                model_id=spec.model_id,
            )
        except Exception as exc:
            raise CloseInputError(
                f"{cohort} canonical snapshot invalid: {evidence_path}"
            ) from exc
        if (
            date.fromisoformat(asof) >= CLOSE_EVIDENCE_ACTIVATION_DATE
            and snapshot.get("snapshot_schema") != SNAPSHOT_SCHEMA_VERSION
        ):
            raise CloseInputError(
                f"{cohort} canonical snapshot uses a legacy schema after "
                f"close-evidence activation: {evidence_path}"
            )
        candidates_raw = snapshot.get("top3")
        evidence_id = snapshot.get("snapshot_id")
        coin_field = "coin"
        immutable_rows = _recommend_immutable_rows(snapshot)
        try:
            receipt = read_delivery_receipt(
                snapshot,
                root=output_root / "recommend_receipts",
            )
        except DeliveryReceiptError as exc:
            raise CloseInputError(
                f"{cohort} canonical delivery receipt invalid"
            ) from exc
        if receipt is not None:
            delivery_ok = receipt["delivery_ok"]
            sent_at = receipt.get("sent_at")
        if spec.requires_successful_delivery and (
            delivery_ok is not True or sent_at is None
        ):
            raise CloseInputError(
                f"{cohort} requires a successful canonical delivery receipt "
                f"for {asof}"
            )
    else:
        payload = _strict_json(evidence_path)
        decision = payload.get("decision")
        if not isinstance(decision, dict):
            raise CloseInputError(
                f"{cohort} decision payload missing: {evidence_path}"
            )
        try:
            if spec.evidence_kind == "pump_v1_decision":
                validated = _validate_v1_decision(
                    payload,
                    decision,
                    evidence_path,
                )
            else:
                _validate_v2_decision(payload, decision, evidence_path)
                validated = decision
        except Exception as exc:
            raise CloseInputError(
                f"{cohort} canonical decision invalid: {evidence_path}"
            ) from exc
        if cohort == "pump-v2" and validated.get("binance_status") != "ok":
            raise CloseInputError(
                f"{cohort} decision is not healthy: "
                f"binance_status={validated.get('binance_status')!r}"
            )
        candidates_raw = validated.get("candidates")
        evidence_id = payload.get("decision_id")
        coin_field = "market"
        if spec.evidence_kind == "pump_v1_decision":
            immutable_rows = _pump_v1_immutable_rows(
                validated,
                decision_recorded_at=payload.get("recorded_at"),
            )
        else:
            immutable_rows = _pump_v2_immutable_rows(
                validated,
                decision_recorded_at=payload.get("recorded_at"),
            )
            receipt_path = (
                output_root / "pump_v2_receipts" / f"{asof}.json"
            )
            if _directory_entry_exists(receipt_path):
                receipt_payload = _strict_json(receipt_path)
                try:
                    _validate_v2_receipt(
                        receipt_payload,
                        validated,
                        receipt_path,
                    )
                except Exception as exc:
                    raise CloseInputError(
                        f"{cohort} canonical delivery receipt invalid: "
                        f"{receipt_path}"
                    ) from exc
                if (
                    receipt_payload.get("decision_id") != evidence_id
                    or receipt_payload.get("decision") != validated
                ):
                    raise CloseInputError(
                        f"{cohort} receipt/decision identity mismatch: "
                        f"{receipt_path}"
                    )
                delivery_ok = receipt_payload["delivery_ok"]
                sent_at = receipt_payload.get("sent_at")
            elif candidates_raw:
                raise CloseInputError(
                    f"{cohort} candidate-bearing delivery receipt missing: "
                    f"{receipt_path}"
                )

    if not isinstance(evidence_id, str) or not evidence_id:
        raise CloseInputError(f"{cohort} evidence id missing: {evidence_path}")
    if not isinstance(candidates_raw, list):
        raise CloseInputError(
            f"{cohort} canonical candidates are invalid: {evidence_path}"
        )
    candidates = tuple(
        _candidate_key(
            candidate,
            coin_field=coin_field,
            source=f"{cohort} evidence",
        )
        for candidate in candidates_raw
        if isinstance(candidate, dict)
    )
    if len(candidates) != len(candidates_raw) or len(set(candidates)) != len(
        candidates
    ):
        raise CloseInputError(
            f"{cohort} canonical candidate identity is not unique"
        )
    if [rank for _, rank in candidates] != list(range(1, len(candidates) + 1)):
        raise CloseInputError(
            f"{cohort} canonical candidate ranks are not contiguous"
        )
    return ExpectedCohort(
        asof=asof,
        evidence_id=evidence_id,
        evidence_path=evidence_path.resolve(strict=False),
        candidates=candidates,
        immutable_rows=immutable_rows,
        validate_delivery_metadata=True,
        delivery_ok=delivery_ok,
        sent_at=sent_at,
    )


def _validate_pre_activation_v2_receipt(
    *,
    asof: str,
    output_root: Path,
) -> None:
    """Authenticate a legacy embedded decision without promoting it.

    The receipt is only a reason to classify the missing-manifest day as
    legacy-unverifiable.  It never becomes canonical decision evidence and is
    never eligible for a forward-valid close.
    """
    receipt_path = output_root / "pump_v2_receipts" / f"{asof}.json"
    if not _directory_entry_exists(receipt_path):
        return
    payload = _strict_json(receipt_path)
    decision = payload.get("decision")
    if not isinstance(decision, dict):
        raise CloseInputError(
            f"pre-activation v2 receipt decision missing: {receipt_path}"
        )
    try:
        _validate_v2_receipt(payload, decision, receipt_path)
    except Exception as exc:
        raise CloseInputError(
            f"pre-activation v2 receipt invalid: {receipt_path}"
        ) from exc
    if (
        receipt_path.stem != asof
        or payload.get("asof") != asof
        or payload.get("delivery_ok") is not True
        or payload.get("sent_at") is None
    ):
        raise CloseInputError(
            f"pre-activation v2 receipt is not a successful same-day receipt: "
            f"{receipt_path}"
        )


def _read_target_rows(ledger_path: Path, asof: str) -> list[dict[str, str]]:
    try:
        with io.StringIO(
            _stable_csv_text(ledger_path),
            newline="",
        ) as handle:
            reader = csv.DictReader(handle, strict=True)
            fieldnames = reader.fieldnames
            if (
                fieldnames is None
                or len(fieldnames) != len(set(fieldnames))
                or not {
                    "date",
                    "coin",
                    "rank",
                    "snapshot_id",
                    "snapshot_path",
                    "status",
                }.issubset(fieldnames)
            ):
                raise CloseInputError(
                    f"ledger identity schema invalid: {ledger_path}"
                )
            rows = []
            for row in reader:
                if None in row:
                    raise CloseInputError(
                        f"ledger row has extra fields: {ledger_path}"
                    )
                if row.get("status") not in ALLOWED_LEDGER_STATUSES:
                    raise CloseInputError(
                        f"ledger row has invalid status: "
                        f"{row.get('status')!r} ({ledger_path})"
                    )
                if row.get("date") == asof:
                    rows.append(row)
            return rows
    except (OSError, csv.Error, UnicodeError) as exc:
        raise CloseInputError(f"ledger is unreadable: {ledger_path}") from exc


def _pending_ledger_dates(
    *,
    ledger_path: Path,
    through_asof: str,
) -> tuple[str, ...]:
    """Return every close-eligible backlog date without trusting pandas casts."""
    if not _directory_entry_exists(ledger_path):
        return ()
    try:
        with io.StringIO(
            _stable_csv_text(ledger_path),
            newline="",
        ) as handle:
            reader = csv.DictReader(handle, strict=True)
            fieldnames = reader.fieldnames
            if (
                fieldnames is None
                or len(fieldnames) != len(set(fieldnames))
                or not {"date", "status"}.issubset(fieldnames)
            ):
                raise CloseInputError(
                    f"ledger backlog schema invalid: {ledger_path}"
                )
            pending: set[str] = set()
            through = date.fromisoformat(through_asof)
            for row in reader:
                if None in row:
                    raise CloseInputError(
                        f"ledger row has extra fields: {ledger_path}"
                    )
                status = row.get("status")
                if status not in ALLOWED_LEDGER_STATUSES:
                    raise CloseInputError(
                        f"ledger row has invalid status: {status!r} "
                        f"({ledger_path})"
                    )
                if status not in {"open", "no_data"}:
                    continue
                raw_date = row.get("date", "")
                try:
                    parsed = date.fromisoformat(raw_date)
                except ValueError as exc:
                    raise CloseInputError(
                        f"ledger pending row date invalid: {raw_date!r} "
                        f"({ledger_path})"
                    ) from exc
                if str(parsed) != raw_date:
                    raise CloseInputError(
                        f"ledger pending row date is not canonical: "
                        f"{raw_date!r} ({ledger_path})"
                    )
                if parsed <= through:
                    pending.add(raw_date)
            return tuple(sorted(pending))
    except (OSError, csv.Error, UnicodeError) as exc:
        raise CloseInputError(f"ledger is unreadable: {ledger_path}") from exc


def validate_close_plan(
    *,
    through_asof: str,
    cohort: str,
    output_root: str | Path = PROJECT_ROOT / "output",
) -> tuple[tuple[str, str], ...]:
    """Validate yesterday plus each older retry row before any closer runs."""
    try:
        parsed = date.fromisoformat(through_asof)
    except ValueError as exc:
        raise CloseInputError(
            f"through_asof must be an ISO date: {through_asof!r}"
        ) from exc
    if str(parsed) != through_asof:
        raise CloseInputError(
            f"through_asof must be canonical ISO date: {through_asof!r}"
        )
    try:
        spec = COHORTS[cohort]
    except KeyError as exc:
        raise CloseInputError(f"unsupported close cohort: {cohort}") from exc
    root = Path(output_root)
    pending = set(
        _pending_ledger_dates(
            ledger_path=root / spec.ledger_name,
            through_asof=through_asof,
        )
    )
    # A zero-pick day has no ledger row, but its canonical decision still needs
    # authentication so a missing candidate-bearing ledger cannot pass silently.
    dates = set(pending)
    dates.add(through_asof)
    plan: list[tuple[str, str]] = []
    for decision_date in sorted(dates):
        try:
            mode = validate_close_input(
                asof=decision_date,
                cohort=cohort,
                output_root=root,
            )
        except MissingCloseEvidenceError:
            # Evidence is entirely absent AND the ledger holds no rows of any
            # status for the injected target day: the send pipeline never
            # produced a decision (and already failed loudly on its own unit).
            # There is nothing to close, so record the day explicitly instead
            # of poisoning every backlog date in this run.  Any ledger row or
            # lingering receipt still fails closed.
            if decision_date != through_asof or not is_no_decision_day(
                asof=decision_date,
                cohort=cohort,
                output_root=root,
            ):
                raise
            log.warning(
                "%s close: no canonical decision evidence, no receipt and no "
                "ledger rows for %s — recording no-decision day",
                cohort,
                decision_date,
            )
            mode = "skip-no-decision"
        plan.append((decision_date, mode))
    return tuple(plan)


def _no_decision_artifacts(
    asof: str,
    spec: CohortSpec,
    output_root: Path,
) -> tuple[Path, ...]:
    """Artifacts whose presence disproves a genuine no-decision day."""
    evidence = output_root / spec.evidence_name.format(asof=asof)
    paths = [evidence]
    if spec.evidence_kind == "recommend_snapshot":
        paths.append(
            output_root / "recommend_receipts" / asof / evidence.name
        )
    elif spec.evidence_kind == "pump_v2_decision":
        paths.append(output_root / "pump_v2_receipts" / f"{asof}.json")
    return tuple(paths)


def _ledger_has_rows_for(ledger_path: Path, asof: str) -> bool:
    """Any-status row presence check via the same hardened CSV entry point.

    Must share ``_stable_csv_text`` (O_NOFOLLOW, regular-file, change-detect)
    and the duplicate-header/extra-field rejection of the sibling readers —
    a permissive reader here would let a symlinked or header-shadowed ledger
    fake a no-decision day at the final under-lock authority.
    """
    if not _directory_entry_exists(ledger_path):
        return False
    try:
        with io.StringIO(
            _stable_csv_text(ledger_path),
            newline="",
        ) as handle:
            reader = csv.DictReader(handle, strict=True)
            fieldnames = reader.fieldnames
            if (
                fieldnames is None
                or len(fieldnames) != len(set(fieldnames))
                or "date" not in fieldnames
            ):
                raise CloseInputError(
                    f"ledger identity schema invalid: {ledger_path}"
                )
            for row in reader:
                if None in row:
                    raise CloseInputError(
                        f"ledger row has extra fields: {ledger_path}"
                    )
                raw = (row.get("date") or "").strip()
                try:
                    canonical = str(date.fromisoformat(raw))
                except ValueError as exc:
                    raise CloseInputError(
                        f"ledger date invalid: {raw!r} ({ledger_path})"
                    ) from exc
                if raw != canonical:
                    raise CloseInputError(
                        f"ledger date noncanonical: {raw!r} ({ledger_path})"
                    )
                if canonical == asof:
                    return True
    except (OSError, csv.Error, UnicodeError) as exc:
        raise CloseInputError(f"ledger is unreadable: {ledger_path}") from exc
    return False


def is_no_decision_day(
    *,
    asof: str,
    cohort: str,
    output_root: str | Path = PROJECT_ROOT / "output",
) -> bool:
    """True iff evidence, receipt, and ledger rows (any status) are all absent.

    The single shared predicate behind ``skip-no-decision``: the plan builder
    and the under-lock closer revalidation must agree on it, so neither may
    reimplement the check.  A closed/not_delivered row or a lingering receipt
    means a decision existed and the missing canonical artifact is an
    integrity failure, never a quiet skip.
    """
    spec = COHORTS.get(cohort)
    if spec is None:
        raise CloseInputError(f"unsupported close cohort: {cohort}")
    root = Path(output_root)
    if _ledger_has_rows_for(root / spec.ledger_name, asof):
        return False
    return not any(
        _directory_entry_exists(path)
        for path in _no_decision_artifacts(asof, spec, root)
    )


def _resolved_recorded_path(value: str, output_root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = output_root.parent / path
    return path.resolve(strict=False)


def _csv_value_matches(actual: str | None, expected: object) -> bool:
    """Compare a CSV scalar to canonical JSON without permissive coercion."""
    if expected is None:
        return actual is not None and actual.strip() == ""
    if actual is None or actual.strip() == "":
        return False
    if isinstance(expected, bool):
        return actual.strip().lower() == str(expected).lower()
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            parsed = float(actual)
            target = float(expected)
        except (TypeError, ValueError):
            return False
        return (
            math.isfinite(parsed)
            and math.isfinite(target)
            and math.isclose(parsed, target, rel_tol=1e-12, abs_tol=1e-12)
        )
    return actual == str(expected)


def _validate_immutable_row(
    *,
    cohort: str,
    asof: str,
    key: tuple[str, int],
    actual: Mapping[str, str],
    expected: Mapping[str, object],
    pre_activation: bool = False,
) -> None:
    missing_columns = sorted(set(expected) - set(actual))
    if missing_columns:
        if not pre_activation or not set(
            missing_columns
        ) <= SEAM_TOLERATED_COLUMNS:
            raise CloseInputError(
                f"{cohort} ledger immutable schema missing for {asof}: "
                f"{missing_columns}"
            )
        # Seam day: the ledger row was appended by the pre-contract writer
        # before the immutable schema landed, while snapshot/receipt already
        # exist in canonical form.  Only the allowlisted seam columns are
        # tolerated; every other column stays strictly enforced.
        log.warning(
            "%s ledger predates immutable schema for %s: tolerating missing "
            "columns %s (values of present columns remain enforced)",
            cohort,
            asof,
            missing_columns,
        )
    for field, expected_value in expected.items():
        if field not in actual:
            continue
        actual_value = actual.get(field)
        if (
            pre_activation
            and field in SEAM_TOLERATED_COLUMNS
            and (actual_value is None or not actual_value.strip())
        ):
            # A later appender backfills the seam column as pd.NA for the
            # pre-contract backlog row; the blank carries no information the
            # canonical snapshot does not already pin.
            log.warning(
                "%s ledger seam column %s blank for %s: tolerating "
                "pre-activation backfill artifact",
                cohort,
                field,
                asof,
            )
            continue
        if not _csv_value_matches(actual_value, expected_value):
            raise CloseInputError(
                f"{cohort} ledger immutable value mismatch for {asof}: "
                f"candidate={key} field={field}"
            )


def _validate_delivery_row(
    *,
    cohort: str,
    asof: str,
    key: tuple[str, int],
    row: Mapping[str, str],
    expected: ExpectedCohort,
) -> None:
    missing_columns = {"delivery_ok", "sent_at"} - set(row)
    if missing_columns:
        raise CloseInputError(
            f"{cohort} ledger delivery schema missing for {asof}: "
            f"{sorted(missing_columns)}"
        )
    if not _csv_value_matches(row.get("delivery_ok"), expected.delivery_ok):
        raise CloseInputError(
            f"{cohort} ledger delivery_ok mismatch for {asof}: {key}"
        )
    if not _csv_value_matches(row.get("sent_at"), expected.sent_at):
        raise CloseInputError(
            f"{cohort} ledger sent_at mismatch for {asof}: {key}"
        )


def validate_close_input(
    *,
    asof: str,
    cohort: str,
    output_root: str | Path = PROJECT_ROOT / "output",
) -> str:
    """Return a close/zero-pick/legacy-unverifiable mode after validation."""
    try:
        parsed_asof = str(date.fromisoformat(asof))
    except ValueError as exc:
        raise CloseInputError(f"asof must be an ISO date: {asof!r}") from exc
    if parsed_asof != asof:
        raise CloseInputError(f"asof must be canonical ISO date: {asof!r}")

    root = Path(output_root)
    spec = COHORTS.get(cohort)
    if spec is None:
        raise CloseInputError(f"unsupported close cohort: {cohort}")
    ledger_path = root / spec.ledger_name
    try:
        expected = _load_expected(
            asof=asof,
            cohort=cohort,
            output_root=root,
        )
    except MissingCloseEvidenceError:
        if date.fromisoformat(asof) >= CLOSE_EVIDENCE_ACTIVATION_DATE:
            raise
        if (
            _directory_entry_exists(ledger_path)
            and asof
            in _pending_ledger_dates(
                ledger_path=ledger_path,
                through_asof=asof,
            )
        ):
            # 계약 이전 날의 pending 행인데 정본 증거가 없다 — 영원히 검증
            # 불가능하므로 매일 close 를 죽이는 대신 legacy-unverifiable 로
            # 남긴다.  행은 open 상태로 원장에 그대로 보이고, forward/verified
            # 통계에는 어차피 인증된 행만 들어간다.  (pump-v1 2026-07-26:
            # decision manifest 는 07-27 활성부터 존재하는데 그날 close 체인이
            # 죽어 있어 잔존 — 이 독이 매일 close 전체를 fail 시키던 회귀 수리)
            log.warning(
                "%s: %s pre-activation pending rows have no canonical "
                "evidence and can never be verified — leaving them open as "
                "legacy-unverifiable",
                cohort,
                asof,
            )
            return "skip-legacy-unverifiable"
        # v1 has no receipt and must never be reconstructed.  A present v2
        # receipt is validated end-to-end, but remains legacy-unverifiable
        # rather than being promoted into a canonical decision manifest.
        if cohort == "pump-v2":
            _validate_pre_activation_v2_receipt(
                asof=asof,
                output_root=root,
            )
        return "skip-legacy-unverifiable"
    if not _directory_entry_exists(ledger_path):
        if expected.candidates:
            raise CloseInputError(
                f"{cohort} candidate-bearing ledger missing for {asof}: "
                f"expected={list(expected.candidates)}"
            )
        return "skip-zero-pick"

    rows = _read_target_rows(ledger_path, asof)
    actual_by_key: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        key = _candidate_key(
            row,
            coin_field="coin",
            source=f"{cohort} ledger",
        )
        if key in actual_by_key:
            raise CloseInputError(
                f"{cohort} ledger has duplicate candidate key for {asof}: {key}"
            )
        actual_by_key[key] = row
    if set(actual_by_key) != set(expected.candidates):
        raise CloseInputError(
            f"{cohort} ledger/evidence candidate mismatch for {asof}: "
            f"ledger={sorted(actual_by_key)} "
            f"evidence={sorted(expected.candidates)}"
        )
    immutable_by_key: dict[tuple[str, int], Mapping[str, object]] = {}
    for immutable_row in expected.immutable_rows:
        key = _candidate_key(
            immutable_row,
            coin_field="coin",
            source=f"{cohort} immutable evidence",
        )
        if key in immutable_by_key:
            raise CloseInputError(
                f"{cohort} immutable evidence has duplicate key: {key}"
            )
        immutable_by_key[key] = immutable_row
    if immutable_by_key and set(immutable_by_key) != set(expected.candidates):
        raise CloseInputError(
            f"{cohort} immutable evidence candidate mismatch for {asof}"
        )
    for key, row in actual_by_key.items():
        if row["snapshot_id"] != expected.evidence_id:
            raise CloseInputError(
                f"{cohort} ledger evidence id mismatch for {asof}: {key}"
            )
        if _resolved_recorded_path(
            row["snapshot_path"],
            root,
        ) != expected.evidence_path:
            raise CloseInputError(
                f"{cohort} ledger evidence path mismatch for {asof}: {key}"
            )
        expected_row = immutable_by_key.get(key)
        if expected_row is not None:
            _validate_immutable_row(
                cohort=cohort,
                asof=asof,
                key=key,
                actual=row,
                expected=expected_row,
                pre_activation=(
                    date.fromisoformat(asof) < CLOSE_EVIDENCE_ACTIVATION_DATE
                ),
            )
        if expected.validate_delivery_metadata:
            _validate_delivery_row(
                cohort=cohort,
                asof=asof,
                key=key,
                row=row,
                expected=expected,
            )
    return "close"


def _force_stderr_logging() -> None:
    """NUL 레코드 프로토콜 절대 방어 — 어떤 모듈이 root logging 을 stdout 으로
    구성해 두었더라도 이 프로세스에서는 stderr 로 강제한다."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        stream = getattr(handler, "stream", None)
        if stream is sys.stdout:
            handler.setStream(sys.stderr)
    if not root.handlers:
        logging.basicConfig(stream=sys.stderr)


def main() -> int:
    _force_stderr_logging()
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Validate immutable daily decision evidence before ledger close"
    )
    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument("--asof", help="single decision date YYYY-MM-DD")
    date_group.add_argument(
        "--through-asof",
        help="validate this date plus every pending backlog date",
    )
    parser.add_argument("--cohort", required=True, choices=sorted(COHORTS))
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "output"))
    parser.add_argument(
        "--output-format",
        choices=("text", "nul"),
        default="text",
        help=(
            "text for human-readable output; nul emits date/mode records plus "
            "a completion sentinel for shell consumers"
        ),
    )
    args = parser.parse_args()
    if args.output_format == "nul" and args.through_asof is None:
        parser.error("--output-format nul requires --through-asof")
    try:
        if args.through_asof is not None:
            plan = validate_close_plan(
                through_asof=args.through_asof,
                cohort=args.cohort,
                output_root=args.output_root,
            )
        else:
            plan = (
                (
                    args.asof,
                    validate_close_input(
                        asof=args.asof,
                        cohort=args.cohort,
                        output_root=args.output_root,
                    ),
                ),
            )
    except CloseInputError as exc:
        print(f"close input gate failed: {exc}", file=sys.stderr)
        return 2
    if args.through_asof is not None:
        if args.output_format == "nul":
            records = [
                item
                for decision_date, mode in plan
                for item in (decision_date, mode)
            ]
            records.append("__PRELUDE_CLOSE_PLAN_V1_OK__")
            payload = b"".join(
                record.encode("utf-8") + b"\0" for record in records
            )
            # 봉인된 프로토콜 fd 가 있으면 그쪽이 유일한 셸 파서 통로다.
            # (없으면 in-process 호출 — 테스트/라이브러리 — 이라 stdout.)
            protocol_out = (
                PROTOCOL_STDOUT
                if PROTOCOL_STDOUT is not None
                else sys.stdout.buffer
            )
            protocol_out.write(payload)
            protocol_out.flush()
        else:
            for decision_date, mode in plan:
                print(f"{decision_date}\t{mode}")
    else:
        print(plan[0][1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

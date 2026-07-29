"""Record-only daily PUMP hunter ledger append.

This script writes candidates from signals.pump_detector_v1 to a dedicated
shadow ledger. It sends no Telegram messages and places no orders.
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

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ledger.csv_store import atomic_write_csv, ledger_lock  # noqa: E402
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
from scripts.recommend_today import RECOMMEND_LEDGER_COLS  # noqa: E402
from signals.pump_detector_v1 import (  # noqa: E402
    DB_PATH,
    EST_PUMP20_PROB,
    MAX_CANDIDATES,
    SL_PCT,
    TP_PCT,
    UNIVERSE_TOP_N,
    score_pump_candidates,
)

PUMP_HUNTER_LEDGER = "output/shadow_ledger_pump_hunter.csv"
PUMP_V1_DECISION_ROOT = "output/pump_v1_decisions"
PUMP_V1_DECISION_SCHEMA = "pump_v1_decision.v2"
PUMP_V1_LEGACY_DECISION_SCHEMA = "pump_v1_decision.v1"
FORWARD_EVIDENCE_ACTIVATION_DATE = date(2026, 7, 27)
FORWARD_PROVENANCE_SCHEMA = "pump_forward_provenance.v1"
_MARKET_RE = re.compile(r"KRW-[A-Z0-9]+")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_INTEGRITY_FIELD = "integrity_sha256"
_DECISION_DOCUMENT_KEYS = {
    "schema",
    "asof",
    "decision_id",
    "decision",
    "recorded_at",
}
_SOURCE_PATHS = (
    "scripts/pump_detector_today.py",
    "signals/pump_detector_v1.py",
    "signals/features.py",
    "data/database.py",
    "data/market_universe.py",
)

EXTRA_LEDGER_COLS = [
    "model_id",
    "rule_version",
    "rule_id",
    "feature_date",
    "liq_rank_daily",
    "roc_7d",
    "roc_7d_rank",
    "atr_pct_14",
    "log_return_1d",
    "pump20_rule",
    "pump15_rule",
    "estimated_pump15_prob",
    "overheated_flag",
]
PUMP_LEDGER_COLS = RECOMMEND_LEDGER_COLS + EXTRA_LEDGER_COLS
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
    "snapshot_id",
    "snapshot_path",
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
    "pump20_rule",
    "pump15_rule",
    "estimated_pump15_prob",
    "overheated_flag",
]

KST = timezone(timedelta(hours=9))
LIVE_RUN_START = datetime.min.time().replace(hour=9)
LIVE_RUN_END = datetime.min.time().replace(hour=9, minute=31)

log = logging.getLogger("pump_detector_today")


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
    asof: str,
    *,
    ledger_path: str | Path,
    decision_root: str | Path,
    top_universe: int,
    max_candidates: int,
    limit_markets: int | None,
) -> None:
    """Reject any non-dry request that is not the official daily cohort."""
    try:
        decision_day = date.fromisoformat(asof)
    except ValueError as exc:
        raise ValueError(f"invalid pump v1 live asof: {asof!r}") from exc
    observed = _now_kst()
    if observed.tzinfo is None:
        raise ValueError("pump v1 live clock must be timezone-aware")
    observed = observed.astimezone(KST)
    if decision_day != observed.date():
        raise RuntimeError(
            "stale pump v1 live run rejected: "
            f"asof={decision_day} today_kst={observed.date()}"
        )
    wall_time = observed.timetz().replace(tzinfo=None)
    if not LIVE_RUN_START <= wall_time < LIVE_RUN_END:
        raise RuntimeError(
            "outside pump v1 live run window: "
            f"{wall_time.isoformat(timespec='seconds')} not in "
            "[09:00,09:31) KST"
        )
    if _project_path(ledger_path) != _project_path(PUMP_HUNTER_LEDGER):
        raise RuntimeError("pump v1 live ledger must be the canonical ledger")
    if _project_path(decision_root) != _project_path(PUMP_V1_DECISION_ROOT):
        raise RuntimeError(
            "pump v1 live decision root must be the canonical root"
        )
    if top_universe != UNIVERSE_TOP_N:
        raise RuntimeError(
            "pump v1 live top_universe must use the official configuration"
        )
    if max_candidates != MAX_CANDIDATES:
        raise RuntimeError(
            "pump v1 live max_candidates must use the official configuration"
        )
    if limit_markets is not None:
        raise RuntimeError("pump v1 live limit_markets is development-only")


def _file_identity(path_value: str | Path) -> dict[str, object]:
    path = Path(path_value)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    try:
        identity = file_identity(path, root=_PROJECT_ROOT)
    except (OSError, ArtifactSourceChangedError) as exc:
        raise RuntimeError(
            f"pump v1 provenance source is unavailable: {path}"
        ) from exc
    if not identity.get("exists") or int(identity.get("size") or 0) <= 0:
        raise RuntimeError(f"pump v1 provenance source is invalid: {path}")
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
            "data/upbit_d1.db": _file_identity(DB_PATH),
        },
    }


def _forward_provenance(decision: dict) -> dict[str, object]:
    inputs = _current_forward_inputs()
    return {
        "schema": FORWARD_PROVENANCE_SCHEMA,
        "evidence_class": "canonical_forward",
        "runner": "pump_detector_v1",
        "config": {
            "top_universe": UNIVERSE_TOP_N,
            "max_candidates": MAX_CANDIDATES,
            "limit_markets": None,
        },
        "decision_basis": {
            "asof": decision["asof"],
            "feature_date": decision["feature_date"],
            "universe_n": decision["universe_n"],
            "n_candidates": decision["n_candidates"],
        },
        **inputs,
    }


def _with_forward_provenance(decision: dict) -> dict:
    enriched = _canonical_decision(decision)
    enriched["execution_provenance"] = _forward_provenance(enriched)
    return enriched


def _validate_file_identities(
    value: object,
    *,
    expected_paths: set[str],
    field: str,
) -> None:
    if not isinstance(value, dict) or set(value) != expected_paths:
        raise ValueError(f"pump v1 {field} identities are invalid")
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
                f"pump v1 {field} identity is invalid: {expected_path}"
            )


def _validate_forward_provenance(decision: dict) -> None:
    provenance = decision.get("execution_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("pump v1 canonical forward provenance is missing")
    if (
        provenance.get("schema") != FORWARD_PROVENANCE_SCHEMA
        or provenance.get("evidence_class") != "canonical_forward"
        or provenance.get("runner") != "pump_detector_v1"
        or provenance.get("config")
        != {
            "top_universe": UNIVERSE_TOP_N,
            "max_candidates": MAX_CANDIDATES,
            "limit_markets": None,
        }
        or provenance.get("decision_basis")
        != {
            "asof": decision["asof"],
            "feature_date": decision["feature_date"],
            "universe_n": decision["universe_n"],
            "n_candidates": decision["n_candidates"],
        }
    ):
        raise ValueError("pump v1 canonical forward provenance is invalid")
    _validate_file_identities(
        provenance.get("sources"),
        expected_paths=set(_SOURCE_PATHS),
        field="source",
    )
    _validate_file_identities(
        provenance.get("data"),
        expected_paths={"data/upbit_d1.db"},
        field="data",
    )


def _json_safe(value: Any) -> Any:
    """Return a strict-JSON copy without numpy scalars or NaN tokens."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    return value


def _canonical_decision(res: dict) -> dict:
    decision = _json_safe(res)
    if not isinstance(decision, dict):
        raise ValueError("pump v1 decision must be an object")
    return decision


def _decision_id(res: dict) -> str:
    decision = _canonical_decision(res)
    encoded = json.dumps(
        decision,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return f"pump-v1-{hashlib.sha256(encoded).hexdigest()[:20]}"


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"pump v1 {field} must be a finite number")
    try:
        parsed = float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"pump v1 {field} must be a finite number"
        ) from exc
    if not math.isfinite(parsed):
        raise ValueError(f"pump v1 {field} must be a finite number")
    return parsed


def _validate_decision_result(
    res: dict,
    *,
    expected_asof: str | None = None,
) -> dict:
    """Validate the immutable v1 scorer result and return its JSON-safe copy."""
    decision = _canonical_decision(res)
    try:
        asof = pd.Timestamp(decision.get("asof")).date()
        feature_date = pd.Timestamp(decision.get("feature_date")).date()
    except (TypeError, ValueError) as exc:
        raise ValueError("pump v1 decision dates are invalid") from exc
    if str(asof) != decision.get("asof"):
        raise ValueError("pump v1 asof must be an ISO date")
    if expected_asof is not None and decision["asof"] != expected_asof:
        raise ValueError(
            "pump v1 scorer asof mismatch: "
            f"requested={expected_asof} returned={decision['asof']}"
        )
    if feature_date != asof - timedelta(days=1):
        raise ValueError("pump v1 feature_date must be exactly asof-1d")
    if decision.get("model_id") != "pump_hunter":
        raise ValueError("pump v1 model identity is invalid")
    if decision.get("rule_version") != "pump_detector_v1":
        raise ValueError("pump v1 rule version is invalid")
    rules = decision.get("rules")
    if (
        not isinstance(rules, dict)
        or not isinstance(rules.get("pump20"), str)
        or not rules["pump20"]
        or not isinstance(rules.get("pump15"), str)
        or not rules["pump15"]
    ):
        raise ValueError("pump v1 rule provenance is invalid")

    universe_n = decision.get("universe_n")
    top_universe = decision.get("top_universe")
    candidates = decision.get("candidates")
    n_candidates = decision.get("n_candidates")
    if (
        isinstance(universe_n, bool)
        or not isinstance(universe_n, int)
        or universe_n <= 0
    ):
        raise ValueError("pump v1 healthy universe_n must be positive")
    if (
        isinstance(top_universe, bool)
        or not isinstance(top_universe, int)
        or top_universe <= 0
        or universe_n > top_universe
    ):
        raise ValueError("pump v1 top-universe contract is invalid")
    if not isinstance(candidates, list):
        raise ValueError("pump v1 candidates must be a list")
    if (
        isinstance(n_candidates, bool)
        or not isinstance(n_candidates, int)
        or n_candidates != len(candidates)
        or n_candidates > universe_n
    ):
        raise ValueError("pump v1 candidate count is invalid")

    markets: list[str] = []
    for expected_rank, candidate in enumerate(candidates, 1):
        if not isinstance(candidate, dict):
            raise ValueError("pump v1 candidate must be an object")
        market = candidate.get("market")
        rank = candidate.get("rank")
        if (
            not isinstance(market, str)
            or _MARKET_RE.fullmatch(market) is None
        ):
            raise ValueError("pump v1 candidate market is invalid")
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank != expected_rank
        ):
            raise ValueError(
                "pump v1 candidate ranks must be contiguous from one"
            )
        score = _finite_number(candidate.get("score"), "candidate.score")
        entry_open = _finite_number(
            candidate.get("entry_open"),
            "candidate.entry_open",
        )
        roc_rank = _finite_number(
            candidate.get("roc_7d_rank"),
            "candidate.roc_7d_rank",
        )
        if not 0.0 <= score <= 1.0 or entry_open <= 0:
            raise ValueError("pump v1 candidate score/entry is invalid")
        if not 0.0 <= roc_rank <= 1.0:
            raise ValueError("pump v1 candidate roc rank is invalid")
        if not isinstance(candidate.get("btc_regime"), str):
            raise ValueError("pump v1 candidate BTC regime is invalid")
        if not isinstance(candidate.get("rule_id"), str):
            raise ValueError("pump v1 candidate rule identity is invalid")
        for probability_field in (
            "estimated_pump20_prob",
            "estimated_pump15_prob",
        ):
            if probability_field not in candidate:
                continue
            probability = candidate.get(probability_field)
            if probability is None:
                if probability_field == "estimated_pump20_prob":
                    raise ValueError(
                        "pump v1 candidate.estimated_pump20_prob "
                        "must be a finite number"
                    )
                continue
            parsed_probability = _finite_number(
                probability,
                f"candidate.{probability_field}",
            )
            if not 0.0 <= parsed_probability <= 1.0:
                raise ValueError(
                    f"pump v1 candidate.{probability_field} "
                    "must be in [0, 1]"
                )
        for boolean_field in (
            "dump_risk_flag",
            "pump20_rule",
            "pump15_rule",
            "overheated_flag",
        ):
            if (
                boolean_field in candidate
                and not isinstance(candidate[boolean_field], bool)
            ):
                raise ValueError(
                    f"pump v1 candidate.{boolean_field} must be boolean"
                )
        markets.append(market)
    if len(set(markets)) != len(markets):
        raise ValueError("pump v1 candidate markets must be unique")
    if "execution_provenance" in decision:
        _validate_forward_provenance(decision)
    json.dumps(decision, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return decision


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
    """Seal outer chronology separately from the inner decision_id."""
    return with_manifest_digest(
        payload,
        digest_key=_INTEGRITY_FIELD,
    )


def _validate_outer_integrity(
    payload: dict,
    *,
    decision_day: date,
    path: Path,
) -> None:
    expected_keys = set(_DECISION_DOCUMENT_KEYS)
    if decision_day >= FORWARD_EVIDENCE_ACTIVATION_DATE:
        expected_keys.add(_INTEGRITY_FIELD)
    if set(payload) != expected_keys:
        raise RuntimeError(
            f"pump v1 decision outer schema mismatch: {path}"
        )
    if (
        decision_day >= FORWARD_EVIDENCE_ACTIVATION_DATE
        and not manifest_digest_matches(
            payload,
            digest_key=_INTEGRITY_FIELD,
        )
    ):
        raise RuntimeError(
            f"pump v1 decision outer integrity mismatch: {path}"
        )


def _validate_decision_document(
    payload: dict,
    res: dict,
    path: Path,
) -> dict:
    decision = _validate_decision_result(res)
    stored = payload.get("decision")
    if not isinstance(stored, dict):
        raise RuntimeError(f"pump v1 decision payload missing: {path}")
    stored = _validate_decision_result(stored)
    decision_day = date.fromisoformat(stored["asof"])
    _validate_outer_integrity(
        payload,
        decision_day=decision_day,
        path=path,
    )
    schema = payload.get("schema")
    if schema == PUMP_V1_DECISION_SCHEMA:
        try:
            _validate_forward_provenance(stored)
        except ValueError as exc:
            raise RuntimeError(
                f"pump v1 forward decision provenance invalid: {path}"
            ) from exc
    elif (
        schema == PUMP_V1_LEGACY_DECISION_SCHEMA
        and decision_day < FORWARD_EVIDENCE_ACTIVATION_DATE
    ):
        pass
    else:
        raise RuntimeError(f"unsupported pump v1 decision schema: {path}")
    if (
        payload.get("asof") != stored["asof"]
        or path.stem != stored["asof"]
        or stored["asof"] != decision["asof"]
    ):
        raise RuntimeError(f"pump v1 decision asof mismatch: {path}")
    if stored != decision:
        raise RuntimeError(
            f"a different pump v1 decision already exists for {decision['asof']}"
        )
    expected_id = _decision_id(stored)
    if payload.get("decision_id") != expected_id:
        raise RuntimeError(f"pump v1 decision checksum mismatch: {path}")
    recorded_at = payload.get("recorded_at")
    try:
        recorded = datetime.fromisoformat(str(recorded_at))
    except ValueError as exc:
        raise RuntimeError(
            f"pump v1 decision recorded_at invalid: {path}"
        ) from exc
    if recorded.tzinfo is None:
        raise RuntimeError(
            f"pump v1 decision recorded_at must be timezone-aware: {path}"
        )
    if decision_day >= FORWARD_EVIDENCE_ACTIVATION_DATE:
        recorded_kst = recorded.astimezone(KST)
        wall_time = recorded_kst.timetz().replace(tzinfo=None)
        if (
            recorded_kst.date() != decision_day
            or not LIVE_RUN_START <= wall_time < LIVE_RUN_END
        ):
            raise RuntimeError(
                f"pump v1 decision recorded outside live window: {path}"
            )
    return decision


def persist_decision(
    res: dict,
    *,
    decision_root: str | Path = PUMP_V1_DECISION_ROOT,
) -> Path:
    """Persist every healthy production decision, including zero candidates."""
    decision = _validate_decision_result(res)
    decision_day = date.fromisoformat(decision["asof"])
    if decision_day >= FORWARD_EVIDENCE_ACTIVATION_DATE:
        try:
            _validate_forward_provenance(decision)
        except ValueError as exc:
            raise RuntimeError(
                "pump v1 legacy decision is not forward-valid"
            ) from exc
    path = Path(decision_root) / f"{decision['asof']}.json"
    with _exclusive_path_lock(path):
        if _path_entry_exists(path):
            try:
                existing = strict_json_object(path)
            except ArtifactValidationError as exc:
                raise RuntimeError(
                    f"pump v1 decision read failed: {path}"
                ) from exc
            _validate_decision_document(existing, decision, path)
            return path
        if (
            decision_day >= FORWARD_EVIDENCE_ACTIVATION_DATE
            and decision["execution_provenance"]
            != _forward_provenance(decision)
        ):
            raise RuntimeError(
                "pump v1 forward provenance does not match current inputs"
            )
        payload = {
            "schema": (
                PUMP_V1_DECISION_SCHEMA
                if "execution_provenance" in decision
                else PUMP_V1_LEGACY_DECISION_SCHEMA
            ),
            "asof": decision["asof"],
            "decision_id": _decision_id(decision),
            "decision": decision,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        if decision_day >= FORWARD_EVIDENCE_ACTIVATION_DATE:
            payload = _with_outer_integrity(payload)
        _validate_decision_document(payload, decision, path)
        _atomic_json(path, payload)
    return path


def _assert_ledger_rows_match_decision(
    existing: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    asof: str,
) -> None:
    if len(existing) != len(expected):
        raise RuntimeError(
            f"pump v1 decision row-count conflict for asof={asof}: "
            f"existing={len(existing)} expected={len(expected)}"
        )
    sort_columns = ["coin", "rank"]
    actual_identity = (
        existing[_LEDGER_IDENTITY_COLS]
        .sort_values(sort_columns)
        .reset_index(drop=True)
        .astype(object)
    )
    expected_identity = (
        expected[_LEDGER_IDENTITY_COLS]
        .sort_values(sort_columns)
        .reset_index(drop=True)
        .astype(object)
    )
    actual_identity = actual_identity.where(
        pd.notna(actual_identity),
        None,
    )
    expected_identity = expected_identity.where(
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
            f"pump v1 immutable decision row conflict for asof={asof}"
        ) from exc


def append_today(asof: str, *, dry_run: bool = False,
                 ledger_path: str = PUMP_HUNTER_LEDGER,
                 decision_root: str | Path = PUMP_V1_DECISION_ROOT,
                 top_universe: int = UNIVERSE_TOP_N,
                 max_candidates: int = MAX_CANDIDATES,
                 limit_markets: int | None = None) -> dict:
    if not dry_run:
        _assert_canonical_forward_request(
            asof,
            ledger_path=ledger_path,
            decision_root=decision_root,
            top_universe=top_universe,
            max_candidates=max_candidates,
            limit_markets=limit_markets,
        )
    input_before = None
    if (
        not dry_run
        and date.fromisoformat(asof) >= FORWARD_EVIDENCE_ACTIVATION_DATE
    ):
        input_before = _current_forward_inputs()
    res = score_pump_candidates(
        asof,
        top_universe=top_universe,
        max_candidates=max_candidates,
        limit_markets=limit_markets,
    )
    log.info(
        "asof=%s feature_date=%s universe_n=%d candidates=%d",
        res["asof"],
        res["feature_date"],
        res["universe_n"],
        res["n_candidates"],
    )
    decision = _validate_decision_result(res, expected_asof=asof)
    decision_path = None
    decision_recorded_at = None
    if not dry_run:
        decision = _with_forward_provenance(decision)
        if input_before is not None and {
            "sources": decision["execution_provenance"]["sources"],
            "data": decision["execution_provenance"]["data"],
        } != input_before:
            raise RuntimeError(
                "pump v1 source/data inputs changed during scoring"
            )
        _assert_canonical_forward_request(
            asof,
            ledger_path=ledger_path,
            decision_root=decision_root,
            top_universe=top_universe,
            max_candidates=max_candidates,
            limit_markets=limit_markets,
        )
        decision_path = persist_decision(
            decision,
            decision_root=decision_root,
        )
        try:
            decision_manifest = strict_json_object(decision_path)
        except ArtifactValidationError as exc:
            raise RuntimeError(
                f"pump v1 decision read-after-write failed: {decision_path}"
            ) from exc
        _validate_decision_document(
            decision_manifest,
            decision,
            decision_path,
        )
        decision_recorded_at = decision_manifest["recorded_at"]
        log.info(
            "decision id=%s path=%s",
            _decision_id(decision),
            decision_path,
        )

    candidates = decision.get("candidates", [])
    if not candidates:
        log.warning("no PUMP hunter candidates — zero-pick decision recorded")
        return decision

    rows = []
    for item in candidates:
        pump_prob = float(item.get("estimated_pump20_prob", EST_PUMP20_PROB))
        rows.append({
            "date": res["asof"],
            "coin": item["market"],
            "rank": int(item["rank"]),
            "score": float(item["score"]),
            "pump_prob": pump_prob,
            "pump_prob_pct": f"{pump_prob * 100:.1f}%",
            "dump_risk_flag": bool(item.get("dump_risk_flag", False)),
            "btc_regime": item.get("btc_regime", "unknown"),
            "entry_open": item.get("entry_open"),
            "sl_pct": SL_PCT,
            "tp_pct": TP_PCT,
            "status": "open",
            "calibration_source": "pump_rule_discovery_v1_prior",
            "snapshot_id": (
                _decision_id(decision) if decision_path is not None else pd.NA
            ),
            "snapshot_path": (
                str(decision_path) if decision_path is not None else pd.NA
            ),
            "decision_completed_at": (
                decision_recorded_at
                if decision_recorded_at is not None
                else pd.NA
            ),
            "p_up5": pd.NA,
            "p_up10": pd.NA,
            "p_up20": pump_prob,
            "p_dn5": pd.NA,
            "p_dn10": pd.NA,
            "exp_downside": pd.NA,
            "rr_ratio": pd.NA,
            "exit_price": pd.NA,
            "exit_reason": pd.NA,
            "realized_pct": pd.NA,
            "pump20_hit": pd.NA,
            "closed_at": pd.NA,
            "model_id": decision["model_id"],
            "rule_version": decision["rule_version"],
            "rule_id": item.get("rule_id"),
            "feature_date": res["feature_date"],
            "liq_rank_daily": item.get("liq_rank_daily"),
            "roc_7d": item.get("roc_7d"),
            "roc_7d_rank": item.get("roc_7d_rank"),
            "atr_pct_14": item.get("atr_pct_14"),
            "log_return_1d": item.get("log_return_1d"),
            "pump20_rule": bool(item.get("pump20_rule", False)),
            "pump15_rule": bool(item.get("pump15_rule", False)),
            "estimated_pump15_prob": item.get("estimated_pump15_prob"),
            "overheated_flag": bool(item.get("overheated_flag", False)),
        })

    new = pd.DataFrame(rows)
    for col in PUMP_LEDGER_COLS:
        if col not in new.columns:
            new[col] = pd.NA
    new = new[PUMP_LEDGER_COLS]

    if dry_run:
        log.info("[dry-run] would append %d rows to %s", len(new), ledger_path)
        _print_rows(new)
        return decision

    p = Path(ledger_path)
    with ledger_lock(p):
        if p.exists():
            existing = pd.read_csv(p)
            for col in PUMP_LEDGER_COLS:
                if col not in existing.columns:
                    existing[col] = pd.NA
            ordered = [c for c in PUMP_LEDGER_COLS if c in existing.columns]
            extras = [c for c in existing.columns if c not in ordered]
            existing = existing[ordered + extras]
            already = existing[existing["date"].astype(str) == res["asof"]]
            if len(already) > 0:
                _assert_ledger_rows_match_decision(
                    already,
                    new,
                    asof=res["asof"],
                )
                log.warning(
                    "asof=%s already has %d rows in %s — skip append (idempotent)",
                    decision["asof"],
                    len(already),
                    p,
                )
                _print_rows(already)
                return decision
            combined = pd.concat([existing, new], ignore_index=True)
        else:
            combined = new

        atomic_write_csv(combined, p)
        log.info("appended %d rows -> %s (total %d)", len(new), p, len(combined))
        _print_rows(new)
    return decision


def _print_rows(df: pd.DataFrame) -> None:
    cols = [
        "date",
        "coin",
        "rank",
        "score",
        "pump_prob_pct",
        "rule_id",
        "roc_7d_rank",
        "atr_pct_14",
        "log_return_1d",
        "status",
    ]
    cols = [c for c in cols if c in df.columns]
    print("\n=== PUMP hunter candidates (SHADOW, record-only) ===")
    print(df[cols].to_string(index=False))


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


def main() -> None:
    _configure_cli_logging()
    ap = argparse.ArgumentParser(description="PUMP hunter daily record-only ledger append")
    ap.add_argument("--asof", type=str, default=None, help="YYYY-MM-DD (default=today KST)")
    ap.add_argument("--ledger", type=str, default=PUMP_HUNTER_LEDGER)
    ap.add_argument(
        "--decision-root",
        type=str,
        default=PUMP_V1_DECISION_ROOT,
        help="immutable v1 decision evidence directory",
    )
    ap.add_argument("--top-universe", type=int, default=UNIVERSE_TOP_N)
    ap.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES)
    ap.add_argument("--limit-markets", type=int, default=None, help="development only")
    ap.add_argument("--dry-run", action="store_true", help="do not write ledger")
    args = ap.parse_args()

    append_today(
        args.asof or _today_kst(),
        dry_run=args.dry_run,
        ledger_path=args.ledger,
        decision_root=args.decision_root,
        top_universe=args.top_universe,
        max_candidates=args.max_candidates,
        limit_markets=args.limit_markets,
    )


if __name__ == "__main__":
    main()

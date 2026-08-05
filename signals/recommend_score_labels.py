"""전 유니버스 recommendation snapshot의 당일 잔여 경로 사후 라벨.

``signals.recommend_snapshot``이 저장한 약 100개 score 행 각각에 실제 15분 경로
결과를 붙인다. 경로 완결성과 무체결 flat-fill 판단은 오직
``ledger.path_quality.assess_15m_window``에 위임한다.

R1이 실제로 전달된 경우 진입 가능 시점은 delivery receipt의 ``sent_at``을
KST 15분 그리드로 올림한 시각이다. receipt가 없는 record-only challenger는
snapshot 생성 시각을 사용한다. 따라서 09:10에 도착한 알림이 09:00 봉을
소급해서 맞힌 것으로 기록되지 않는다.

목표일 밖에서 나중에 생성한 snapshot은 예정 슬롯 시각으로 경로 라벨을 만들 수는
있지만 ``provenance_cohort='scheduled_replay'``로 고정한다. 이는 관측된
``forward_observed``와 섞지 않기 위한 출처 표식이며 경로 계산 자체는 동일하다.

미성숙 경로는 artifact를 쓰지 않는다. 수집 장애 등 incomplete 경로는 수익 라벨을
만들지 않고 ``artifact_status='partial'``로 명시해 다음 실행에서 재시도할 수 있게 한다.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
import math
import os
import re
import sqlite3
import stat
import tempfile
import time
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Literal, cast, overload

import numpy as np
import pandas as pd

from ledger.config import ROUND_TRIP_COST_PCT
from ledger.path_quality import (
    PathAssessment,
    assess_15m_window,
    next_bar_boundary,
)
from notifier.delivery_receipt import (
    DEFAULT_RECEIPT_ROOT,
    DeliveryReceiptError,
    read_delivery_receipt,
)
from ops.artifact_provenance import (
    ArtifactSourceChangedError,
    ArtifactValidationError,
    canonical_json_bytes,
    file_set_identity,
    sha256_bytes,
    strict_json_object,
)
from ops.file_lock import FileLockError, file_lock
from signals.recommend_snapshot import load_snapshot

_ROOT = Path(__file__).resolve().parent.parent
M15_DB_PATH = _ROOT / "data" / "upbit_15m.db"
DEFAULT_LABEL_ROOT = _ROOT / "output" / "recommend_score_labels"
LABEL_SCHEMA_VERSION = "recommend_score_labels.v3"
KST = timezone(timedelta(hours=9))

TP5 = 0.05
SL3 = 0.03
ROUND_TRIP_COST = ROUND_TRIP_COST_PCT
FORWARD_PROVENANCE_COHORT = "forward_observed"
OFF_SCHEDULE_PROVENANCE_COHORT = "off_schedule_observed"
SCHEDULED_REPLAY_PROVENANCE_COHORT = "scheduled_replay"
_LABEL_SOURCE_FILES = (
    _ROOT / "signals" / "recommend_score_labels.py",
    _ROOT / "ledger" / "path_quality.py",
    _ROOT / "ledger" / "config.py",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MARKET_RE = re.compile(r"KRW-[A-Z0-9]+")

log = logging.getLogger(__name__)
# 진단 경고가 root logger 를 타고 stdout 으로 새면 NUL 레코드 프로토콜·헬스
# 프로브 캡처를 오염시킨다 (2026-07-29 close 3채널·heartbeat 도배 회귀).
# 이 모듈의 로그는 stderr 고정 + 전파 차단.
if not log.handlers:
    _stderr_handler = logging.StreamHandler(sys.stderr)
    _stderr_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    log.addHandler(_stderr_handler)
    log.propagate = False
_SNAPSHOT_ID_RE = re.compile(r"recommend-[0-9a-f]{20}")
_PERCENT_RE = re.compile(r"(?:100|[0-9]{1,2})\.[0-9]%")
_DISPLAY_PROBABILITY_TOLERANCE = 5.5e-4

_ARTIFACT_REQUIRED_KEYS = frozenset({
    "schema",
    "artifact_status",
    "return_unit",
    "round_trip_cost_fraction",
    "asof",
    "slot",
    "ranking",
    "feature_asof",
    "delivery_ok",
    "receipt_path",
    "execution_at",
    "execution_start_at",
    "execution_time_basis",
    "provenance_cohort",
    "forward_eligible",
    "scheduled_window_start",
    "scheduled_window_end",
    "snapshot_id",
    "snapshot_payload_sha256",
    "snapshot_path",
    "snapshot_model",
    "snapshot_rule",
    "snapshot_code",
    "snapshot_data",
    "label_code",
    "path_input",
    "path_window_start",
    "path_window_end",
    "target_day_window_start",
    "target_day_window_end",
    "labeled_at",
    "summary",
    "rows",
    "label_payload_sha256",
})
_ARTIFACT_OPTIONAL_KEYS = frozenset({"supersedes_label_payload_sha256"})

_CANDIDATE_KEYS = frozenset({
    "coin",
    "rank",
    "score",
    "pump_prob",
    "pump_prob_pct",
    "rr_ratio",
    "p_up5",
    "p_up10",
    "p_up20",
    "p_dn5",
    "p_dn10",
    "exp_downside",
    "dump_risk_flag",
    "entry_open",
    "sl",
    "tp",
    "btc_regime",
    "feature_values",
})
_EXECUTION_KEYS = frozenset({
    "delivery_ok",
    "receipt_path",
    "execution_at",
    "execution_start_at",
    "execution_time_basis",
    "provenance_cohort",
    "forward_eligible",
    "scheduled_window_start",
    "scheduled_window_end",
})
_PATH_KEYS = frozenset({
    "path_complete",
    "path_quality",
    "path_reason",
    "raw_bars",
    "expected_bars",
    "flat_filled_bars",
    "benchmark_bars",
    "path_used_bars",
    "label_status",
})
_OUTCOME_NUMBER_KEYS = frozenset({
    "actual_entry_open",
    "mfe",
    "mae",
    "eod_return",
    "eod_return_gross",
    "eod_return_net",
    "tp5_sl3_return_gross",
    "tp5_sl3_return_net",
})
_OUTCOME_BOOL_KEYS = frozenset({
    "up5",
    "up10",
    "up20",
    "dn3",
    "dn5",
    "dn10",
    "tp5_before_sl3",
})
_OUTCOME_KEYS = (
    _OUTCOME_NUMBER_KEYS
    | _OUTCOME_BOOL_KEYS
    | frozenset({
        "tp5_sl3_first_passage",
        "first_passage_bar",
        "first_passage_at",
    })
)
_ROW_REQUIRED_KEYS = (
    _CANDIDATE_KEYS
    | _EXECUTION_KEYS
    | _PATH_KEYS
    | _OUTCOME_KEYS
    | frozenset({"snapshot_id", "snapshot_payload_sha256"})
)
_PROVENANCE_COHORTS = frozenset({
    FORWARD_PROVENANCE_COHORT,
    OFF_SCHEDULE_PROVENANCE_COHORT,
    SCHEDULED_REPLAY_PROVENANCE_COHORT,
})
_LABEL_STATUSES = frozenset({
    "labeled",
    "path_incomplete",
    "no_executable_path",
    "invalid_complete_path",
    "assessment_error",
    # 거래정지/상폐로 창 전체 무봉이 업스트림에서 확인된 구조적 종결 상태.
    # incomplete 와 달리 재시도 대상이 아니며 artifact 완결을 막지 않는다
    # (2026-08-05 AERGO·AQT 정지가 close·publish 를 무기한 차단한 사고).
    "halted_no_observations",
})
_FIRST_PASSAGE_VALUES = frozenset({
    "tp_first",
    "sl_first",
    "sl_first_same_bar",
    "neither",
})
_LEGACY_ARTIFACT_REQUIRED_KEYS = frozenset({
    "schema",
    "artifact_status",
    "return_unit",
    "asof",
    "slot",
    "ranking",
    "feature_asof",
    "snapshot_id",
    "snapshot_payload_sha256",
    "snapshot_path",
    "path_window_start",
    "path_window_end",
    "labeled_at",
    "execution_time_basis",
    "summary",
    "rows",
    "label_payload_sha256",
})
_LEGACY_PROVENANCE_KEYS = frozenset({
    "provenance_cohort",
    "forward_eligible",
})
_LEGACY_ROW_REQUIRED_KEYS = frozenset({
    "coin",
    "rank",
    "score",
    "p_up10",
    "p_dn5",
    "up10",
    "dn5",
    "mfe",
    "mae",
    "eod_return",
    "tp5_sl3_first_passage",
    "tp5_sl3_return_net",
    "label_status",
    "path_complete",
    "path_quality",
    "delivery_ok",
    "execution_at",
    "execution_time_basis",
    "feature_values",
})
_LEGACY_ROW_OPTIONAL_KEYS = (
    _LEGACY_PROVENANCE_KEYS
    | frozenset({"net_return", "eod_return_net"})
)


class ScoreLabelError(RuntimeError):
    """라벨 artifact 또는 snapshot 계약 위반."""


def _normalise(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _normalise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise(v) for v in value]
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _artifact_digest(document: dict) -> str:
    payload = {
        k: v for k, v in document.items()
        if k not in {
            "labeled_at",
            "label_payload_sha256",
            "artifact_path",
            "artifact_reused",
            "written",
        }
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _artifact_error(path: Path, field: str, detail: str) -> ScoreLabelError:
    return ScoreLabelError(f"invalid label artifact {field}: {detail} ({path})")


def _expect(condition: bool, path: Path, field: str, detail: str) -> None:
    if not condition:
        raise _artifact_error(path, field, detail)


def _exact_keys(
    value: dict,
    required: frozenset[str],
    path: Path,
    field: str,
    *,
    optional: frozenset[str] = frozenset(),
) -> None:
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required - optional)
    _expect(
        not missing and not unknown,
        path,
        field,
        f"missing={missing}, unknown={unknown}",
    )


def _finite_json(value: Any, path: Path, field: str = "root") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        _expect(math.isfinite(value), path, field, "number must be finite")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _finite_json(item, path, f"{field}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _expect(isinstance(key, str), path, field, "keys must be strings")
            _finite_json(item, path, f"{field}.{key}")
        return
    raise _artifact_error(
        path,
        field,
        f"unsupported value type {type(value).__name__}",
    )


def _text(value: Any, path: Path, field: str) -> str:
    _expect(isinstance(value, str) and bool(value), path, field, "must be text")
    return value


def _boolean(value: Any, path: Path, field: str) -> bool:
    _expect(isinstance(value, bool), path, field, "must be bool")
    return value


@overload
def _number(
    value: Any,
    path: Path,
    field: str,
    *,
    nullable: Literal[False] = False,
) -> float: ...


@overload
def _number(
    value: Any,
    path: Path,
    field: str,
    *,
    nullable: Literal[True],
) -> float | None: ...


def _number(
    value: Any,
    path: Path,
    field: str,
    *,
    nullable: bool = False,
) -> float | None:
    if value is None and nullable:
        return None
    _expect(
        not isinstance(value, bool) and isinstance(value, (int, float)),
        path,
        field,
        "must be numeric",
    )
    parsed = float(value)
    _expect(math.isfinite(parsed), path, field, "must be finite")
    return parsed


@overload
def _uint(
    value: Any,
    path: Path,
    field: str,
    *,
    nullable: Literal[False] = False,
) -> int: ...


@overload
def _uint(
    value: Any,
    path: Path,
    field: str,
    *,
    nullable: Literal[True],
) -> int | None: ...


def _uint(
    value: Any,
    path: Path,
    field: str,
    *,
    nullable: bool = False,
) -> int | None:
    if value is None and nullable:
        return None
    _expect(
        not isinstance(value, bool) and isinstance(value, int) and value >= 0,
        path,
        field,
        "must be a non-negative integer",
    )
    return value


def _date_value(value: Any, path: Path, field: str) -> date:
    raw = _text(value, path, field)
    try:
        parsed = date.fromisoformat(raw)
    except ValueError as exc:
        raise _artifact_error(path, field, "must be an ISO date") from exc
    _expect(parsed.isoformat() == raw, path, field, "must be canonical ISO")
    return parsed


def _datetime_value(
    value: Any,
    path: Path,
    field: str,
    *,
    aware: bool,
) -> datetime:
    raw = _text(value, path, field)
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise _artifact_error(path, field, "must be an ISO datetime") from exc
    is_aware = parsed.tzinfo is not None and parsed.utcoffset() is not None
    _expect(is_aware == aware, path, field, "timezone contract mismatch")
    _expect(parsed.isoformat() == raw, path, field, "must be canonical ISO")
    return parsed


def _digest_value(value: Any, path: Path, field: str) -> str:
    digest = _text(value, path, field)
    _expect(
        _SHA256_RE.fullmatch(digest) is not None,
        path,
        field,
        "must be a lowercase SHA-256 digest",
    )
    return digest


def _validate_provenance(
    value: dict,
    path: Path,
    field: str,
    *,
    optional: bool,
) -> str:
    has_cohort = "provenance_cohort" in value
    has_flag = "forward_eligible" in value
    _expect(has_cohort == has_flag, path, field, "cohort/eligibility shape mismatch")
    if not has_cohort:
        _expect(
            optional
            and value.get("execution_time_basis")
            == "scheduled_slot_fallback_snapshot_outside_window",
            path,
            field,
            "missing provenance is allowed only for legacy scheduled replay",
        )
        return SCHEDULED_REPLAY_PROVENANCE_COHORT
    cohort = value["provenance_cohort"]
    _expect(cohort in _PROVENANCE_COHORTS, path, field, "unsupported cohort")
    flag = _boolean(value["forward_eligible"], path, f"{field}.forward_eligible")
    _expect(
        flag == (cohort == FORWARD_PROVENANCE_COHORT),
        path,
        field,
        "eligibility does not match cohort",
    )
    return cohort


def _validate_summary(
    summary: Any,
    rows: list[dict],
    status: str,
    path: Path,
    *,
    legacy: bool,
) -> None:
    _expect(isinstance(summary, dict), path, "summary", "must be an object")
    required = frozenset({
        "snapshot_universe_n",
        "rows",
        "labeled",
        "incomplete",
        "flat_filled",
    })
    # "halted" 는 2026-08-05 이후 artifact 에만 존재(정지종목 구조적 종결).
    # 구형 artifact 와의 호환을 위해 선택 키로 둔다.
    if "halted" in summary:
        required = required | {"halted"}
    _exact_keys(summary, required, path, "summary")
    observed = {
        key: _uint(value, path, f"summary.{key}")
        for key, value in summary.items()
    }
    if legacy:
        labeled = len(rows) if status == "complete" else 0
        flat_filled = 0
        halted = 0
    else:
        labeled = sum(row["label_status"] == "labeled" for row in rows)
        flat_filled = sum(row["path_quality"] == "flat_filled" for row in rows)
        halted = sum(
            row["label_status"] == "halted_no_observations" for row in rows
        )
    expected = {
        "snapshot_universe_n": len(rows),
        "rows": len(rows),
        "labeled": labeled,
        "incomplete": len(rows) - labeled - halted,
        "flat_filled": flat_filled,
    }
    if "halted" in summary:
        expected["halted"] = halted
    else:
        _expect(halted == 0, path, "summary", "halted rows without halted key")
    _expect(observed == expected, path, "summary", "does not match rows")
    if not legacy:
        expected_status = (
            "complete" if rows and labeled + halted == len(rows) else "partial"
        )
        _expect(status == expected_status, path, "artifact_status", "row mismatch")


def _validate_legacy_label_artifact(document: dict, path: Path) -> None:
    """Read-only compatibility for the exact early-v3 evaluation shape."""
    _exact_keys(
        document,
        _LEGACY_ARTIFACT_REQUIRED_KEYS,
        path,
        "root",
        optional=_LEGACY_PROVENANCE_KEYS,
    )
    _finite_json(document, path)
    _expect(document["schema"] == LABEL_SCHEMA_VERSION, path, "schema", "unsupported")
    status = document["artifact_status"]
    _expect(status in {"complete", "partial"}, path, "artifact_status", "invalid")
    _expect(document["return_unit"] == "fraction", path, "return_unit", "invalid")
    asof = _date_value(document["asof"], path, "asof")
    slot = document["slot"]
    _expect(slot in {"open", "preopen"}, path, "slot", "invalid")
    _expect(document["ranking"] in {"R1", "R2", "A1"}, path, "ranking", "invalid")
    feature_asof = _date_value(document["feature_asof"], path, "feature_asof")
    expected_feature = asof if slot == "open" else asof - timedelta(days=1)
    _expect(feature_asof == expected_feature, path, "feature_asof", "cutoff mismatch")
    for field in ("snapshot_id", "snapshot_payload_sha256", "snapshot_path"):
        _text(document[field], path, field)
    for field in ("path_window_start", "path_window_end", "labeled_at"):
        _datetime_value(document[field], path, field, aware=True)
    basis = _text(document["execution_time_basis"], path, "execution_time_basis")
    cohort = _validate_provenance(document, path, "provenance", optional=True)

    rows = document["rows"]
    _expect(isinstance(rows, list), path, "rows", "must be an array")
    coins: list[str] = []
    ranks: list[int] = []
    for index, row in enumerate(rows):
        field = f"rows[{index}]"
        _expect(isinstance(row, dict), path, field, "must be an object")
        _exact_keys(
            row,
            _LEGACY_ROW_REQUIRED_KEYS,
            path,
            field,
            optional=_LEGACY_ROW_OPTIONAL_KEYS,
        )
        coin = _text(row["coin"], path, f"{field}.coin")
        _expect(_MARKET_RE.fullmatch(coin) is not None, path, f"{field}.coin", "invalid")
        rank = _uint(row["rank"], path, f"{field}.rank")
        _expect(rank > 0, path, f"{field}.rank", "must be positive")
        score = _number(row["score"], path, f"{field}.score")
        p_up10 = _number(row["p_up10"], path, f"{field}.p_up10")
        p_dn5 = _number(row["p_dn5"], path, f"{field}.p_dn5")
        _expect(
            all(0 <= value <= 1 for value in (score, p_up10, p_dn5)),
            path,
            field,
            "scores must be in [0, 1]",
        )
        mfe = _number(row["mfe"], path, f"{field}.mfe")
        mae = _number(row["mae"], path, f"{field}.mae")
        eod = _number(row["eod_return"], path, f"{field}.eod_return")
        _expect(
            _boolean(row["up10"], path, f"{field}.up10") == (mfe >= 0.10)
            and _boolean(row["dn5"], path, f"{field}.dn5") == (mae <= -0.05),
            path,
            field,
            "outcome flags mismatch",
        )
        _expect(
            row["tp5_sl3_first_passage"] in _FIRST_PASSAGE_VALUES,
            path,
            f"{field}.tp5_sl3_first_passage",
            "invalid",
        )
        _number(row["tp5_sl3_return_net"], path, f"{field}.tp5_sl3_return_net")
        _expect(row["label_status"] == "labeled", path, f"{field}.label_status", "invalid")
        _expect(
            _boolean(row["path_complete"], path, f"{field}.path_complete"),
            path,
            f"{field}.path_complete",
            "must be true",
        )
        _text(row["path_quality"], path, f"{field}.path_quality")
        _boolean(row["delivery_ok"], path, f"{field}.delivery_ok")
        _datetime_value(row["execution_at"], path, f"{field}.execution_at", aware=True)
        _expect(row["execution_time_basis"] == basis, path, field, "basis mismatch")
        features = row["feature_values"]
        _expect(isinstance(features, dict), path, f"{field}.feature_values", "invalid")
        for name, value in features.items():
            _text(name, path, f"{field}.feature_values key")
            _number(value, path, f"{field}.feature_values.{name}", nullable=True)
        for name in ("net_return", "eod_return_net"):
            if name in row:
                net = _number(row[name], path, f"{field}.{name}")
                _expect(
                    math.isclose(net, eod - ROUND_TRIP_COST, abs_tol=1e-12),
                    path,
                    f"{field}.{name}",
                    "cost mismatch",
                )
        row_cohort = _validate_provenance(row, path, f"{field}.provenance", optional=True)
        _expect(row_cohort == cohort, path, f"{field}.provenance", "artifact mismatch")
        coins.append(coin)
        ranks.append(rank)
    _expect(ranks == list(range(1, len(rows) + 1)), path, "rows.rank", "not contiguous")
    _expect(len(coins) == len(set(coins)), path, "rows.coin", "duplicates")
    _validate_summary(document["summary"], rows, status, path, legacy=True)
    _digest_value(document["label_payload_sha256"], path, "label_payload_sha256")


def _validate_label_code(value: Any, path: Path) -> None:
    _expect(isinstance(value, dict), path, "label_code", "must be an object")
    _exact_keys(value, frozenset({"sha256", "files"}), path, "label_code")
    _digest_value(value["sha256"], path, "label_code.sha256")
    files = value["files"]
    _expect(isinstance(files, list), path, "label_code.files", "must be an array")
    expected = [str(source.relative_to(_ROOT)) for source in _LABEL_SOURCE_FILES]
    observed = []
    for index, item in enumerate(files):
        field = f"label_code.files[{index}]"
        _expect(isinstance(item, dict), path, field, "must be an object")
        _exact_keys(item, frozenset({"path", "sha256"}), path, field)
        observed.append(_text(item["path"], path, f"{field}.path"))
        _digest_value(item["sha256"], path, f"{field}.sha256")
    _expect(observed == expected, path, "label_code.files", "source set mismatch")


def _validate_modern_row(
    row: dict,
    document: dict,
    index: int,
    path: Path,
) -> tuple[str, int]:
    field = f"rows[{index}]"
    _exact_keys(row, _ROW_REQUIRED_KEYS, path, field)
    coin = _text(row["coin"], path, f"{field}.coin")
    _expect(_MARKET_RE.fullmatch(coin) is not None, path, f"{field}.coin", "invalid")
    rank = _uint(row["rank"], path, f"{field}.rank")
    _expect(rank > 0, path, f"{field}.rank", "must be positive")

    score = _number(row["score"], path, f"{field}.score")
    pump_prob = _number(row["pump_prob"], path, f"{field}.pump_prob")
    _expect(0 <= score <= 1 and 0 <= pump_prob <= 1, path, field, "score range")
    probability_text = row["pump_prob_pct"]
    _expect(
        isinstance(probability_text, str)
        and _PERCENT_RE.fullmatch(probability_text) is not None,
        path,
        f"{field}.pump_prob_pct",
        "probability format mismatch",
    )
    displayed_probability = float(probability_text[:-1]) / 100.0
    _expect(
        probability_text == f"{displayed_probability * 100:.1f}%"
        and abs(displayed_probability - pump_prob)
        <= _DISPLAY_PROBABILITY_TOLERANCE,
        path,
        f"{field}.pump_prob_pct",
        "probability mismatch",
    )
    probabilities = {
        name: _number(row[name], path, f"{field}.{name}", nullable=True)
        for name in ("p_up5", "p_up10", "p_up20", "p_dn5", "p_dn10")
    }
    _expect(
        all(value is None or 0 <= value <= 1 for value in probabilities.values()),
        path,
        field,
        "probability range",
    )
    p_up5 = probabilities["p_up5"]
    p_up10 = probabilities["p_up10"]
    p_up20 = probabilities["p_up20"]
    # 독립 head 라 포함관계가 구조적으로 보장되지 않는 알려진 모델 성질이며,
    # snapshot 검증(recommend_snapshot)과 동일하게 진단으로만 남긴다.  값의
    # 무결성은 artifact checksum 이 담당하고, 여기서 하드 실패로 두면 유니버스
    # 꼬리 코인 하나가 전 유니버스 forward 라벨 축적을 죽인다(2026-07-27 실증).
    if (
        p_up5 is not None
        and p_up10 is not None
        and p_up20 is not None
        and not p_up5 >= p_up10 >= p_up20
    ):
        log.warning(
            "label artifact %s: upside nesting violated (diagnostic only): %s",
            field,
            path,
        )
    if (
        probabilities["p_dn5"] is not None
        and probabilities["p_dn10"] is not None
        and probabilities["p_dn5"] < probabilities["p_dn10"]
    ):
        log.warning(
            "label artifact %s: downside nesting violated (diagnostic only): "
            "%s",
            field,
            path,
        )
    rr_ratio = _number(row["rr_ratio"], path, f"{field}.rr_ratio", nullable=True)
    downside = _number(row["exp_downside"], path, f"{field}.exp_downside", nullable=True)
    _expect(rr_ratio is None or rr_ratio >= 0, path, f"{field}.rr_ratio", "negative")
    _expect(downside is None or downside <= 0, path, f"{field}.exp_downside", "positive")
    _boolean(row["dump_risk_flag"], path, f"{field}.dump_risk_flag")
    entry = _number(row["entry_open"], path, f"{field}.entry_open", nullable=True)
    if document["slot"] == "open":
        _expect(entry is not None and entry > 0, path, f"{field}.entry_open", "invalid")
    else:
        _expect(entry is None, path, f"{field}.entry_open", "must be null")
    sl = _number(row["sl"], path, f"{field}.sl")
    tp = _number(row["tp"], path, f"{field}.tp")
    _expect(
        math.isclose(sl, -SL3, abs_tol=1e-12)
        and math.isclose(tp, TP5, abs_tol=1e-12),
        path,
        field,
        "exit contract mismatch",
    )
    _text(row["btc_regime"], path, f"{field}.btc_regime")
    features = row["feature_values"]
    _expect(isinstance(features, dict), path, f"{field}.feature_values", "invalid")
    for name, value in features.items():
        _text(name, path, f"{field}.feature_values key")
        _number(value, path, f"{field}.feature_values.{name}", nullable=True)

    _expect(
        row["snapshot_id"] == document["snapshot_id"]
        and row["snapshot_payload_sha256"] == document["snapshot_payload_sha256"],
        path,
        field,
        "snapshot identity mismatch",
    )
    for name in _EXECUTION_KEYS:
        _expect(row[name] == document[name], path, f"{field}.{name}", "artifact mismatch")

    status = row["label_status"]
    _expect(status in _LABEL_STATUSES, path, f"{field}.label_status", "invalid")
    complete = _boolean(row["path_complete"], path, f"{field}.path_complete")
    _text(row["path_quality"], path, f"{field}.path_quality")
    if row["path_reason"] is not None:
        _text(row["path_reason"], path, f"{field}.path_reason")
    counts = {
        name: _uint(row[name], path, f"{field}.{name}", nullable=True)
        for name in ("raw_bars", "expected_bars", "flat_filled_bars", "benchmark_bars")
    }
    used = _uint(row["path_used_bars"], path, f"{field}.path_used_bars")
    if status == "assessment_error":
        _expect(all(value is None for value in counts.values()), path, field, "count mismatch")
    else:
        _expect(all(value is not None for value in counts.values()), path, field, "missing counts")
        _expect(counts["expected_bars"] == 96, path, field, "expected_bars mismatch")
        _expect(
            all(value is None or value <= 96 for value in counts.values())
            and used <= 96,
            path,
            field,
            "bar count range",
        )
    _expect(
        (status == "labeled" and complete and used == 96)
        or (status == "no_executable_path" and complete)
        or (
            status
            in {
                "path_incomplete",
                "invalid_complete_path",
                "assessment_error",
                "halted_no_observations",
            }
            and not complete
        ),
        path,
        field,
        "path status mismatch",
    )

    numbers = {
        name: _number(row[name], path, f"{field}.{name}", nullable=True)
        for name in _OUTCOME_NUMBER_KEYS
    }
    bools = {}
    for name in _OUTCOME_BOOL_KEYS:
        value = row[name]
        _expect(value is None or isinstance(value, bool), path, f"{field}.{name}", "invalid")
        bools[name] = value
    first = row["tp5_sl3_first_passage"]
    first_bar = _uint(row["first_passage_bar"], path, f"{field}.first_passage_bar", nullable=True)
    first_at = row["first_passage_at"]

    if status != "labeled":
        _expect(all(row[name] is None for name in _OUTCOME_KEYS), path, field, "outcome leak")
        return coin, rank
    _expect(all(value is not None for value in numbers.values()), path, field, "missing outcome")
    complete_numbers = cast(dict[str, float], numbers)
    _expect(
        complete_numbers["actual_entry_open"] > 0,
        path,
        f"{field}.actual_entry_open",
        "invalid",
    )
    expected_flags = {
        "up5": complete_numbers["mfe"] >= 0.05,
        "up10": complete_numbers["mfe"] >= 0.10,
        "up20": complete_numbers["mfe"] >= 0.20,
        "dn3": complete_numbers["mae"] <= -0.03,
        "dn5": complete_numbers["mae"] <= -0.05,
        "dn10": complete_numbers["mae"] <= -0.10,
    }
    _expect(
        all(bools[name] == expected for name, expected in expected_flags.items()),
        path,
        field,
        "outcome flag mismatch",
    )
    _expect(
        math.isclose(
            complete_numbers["eod_return"],
            complete_numbers["eod_return_gross"],
            abs_tol=1e-12,
        )
        and math.isclose(
            complete_numbers["eod_return_net"],
            complete_numbers["eod_return_gross"] - ROUND_TRIP_COST,
            abs_tol=1e-12,
        ),
        path,
        field,
        "return mismatch",
    )
    _expect(first in _FIRST_PASSAGE_VALUES, path, f"{field}.tp5_sl3_first_passage", "invalid")
    if first == "neither":
        _expect(
            first_bar is None and first_at is None and bools["tp5_before_sl3"] is None,
            path,
            field,
            "first-passage mismatch",
        )
        gross = complete_numbers["eod_return_gross"]
    else:
        _expect(first_bar is not None and first_at is not None, path, field, "missing passage")
        parsed_first_at = _datetime_value(
            first_at,
            path,
            f"{field}.first_passage_at",
            aware=False,
        )
        _expect(
            first_bar < used,
            path,
            f"{field}.first_passage_bar",
            "outside labeled path",
        )
        execution_start = _datetime_value(
            row["execution_start_at"],
            path,
            f"{field}.execution_start_at",
            aware=True,
        )
        expected_first_at = (
            execution_start.astimezone(KST).replace(tzinfo=None)
            + timedelta(minutes=15 * first_bar)
        )
        _expect(
            parsed_first_at == expected_first_at,
            path,
            f"{field}.first_passage_at",
            "does not match passage bar",
        )
        before = first == "tp_first"
        _expect(bools["tp5_before_sl3"] is before, path, field, "direction mismatch")
        gross = TP5 if before else -SL3
    _expect(
        math.isclose(
            complete_numbers["tp5_sl3_return_gross"],
            gross,
            abs_tol=1e-12,
        )
        and math.isclose(
            complete_numbers["tp5_sl3_return_net"],
            gross - ROUND_TRIP_COST,
            abs_tol=1e-12,
        ),
        path,
        field,
        "first-passage return mismatch",
    )
    return coin, rank


def _validate_label_artifact(document: dict, path: Path) -> None:
    if "round_trip_cost_fraction" not in document:
        _validate_legacy_label_artifact(document, path)
        return
    _exact_keys(
        document,
        _ARTIFACT_REQUIRED_KEYS,
        path,
        "root",
        optional=_ARTIFACT_OPTIONAL_KEYS,
    )
    _finite_json(document, path)
    _expect(document["schema"] == LABEL_SCHEMA_VERSION, path, "schema", "unsupported")
    status = document["artifact_status"]
    _expect(status in {"complete", "partial"}, path, "artifact_status", "invalid")
    _expect(document["return_unit"] == "fraction", path, "return_unit", "invalid")
    cost = _number(document["round_trip_cost_fraction"], path, "round_trip_cost_fraction")
    _expect(
        math.isclose(cost, ROUND_TRIP_COST, abs_tol=1e-12),
        path,
        "round_trip_cost_fraction",
        "cost mismatch",
    )
    asof = _date_value(document["asof"], path, "asof")
    slot = document["slot"]
    ranking = document["ranking"]
    _expect(slot in {"open", "preopen"}, path, "slot", "invalid")
    _expect(ranking in {"R1", "R2", "A1"}, path, "ranking", "invalid")
    feature_asof = _date_value(document["feature_asof"], path, "feature_asof")
    expected_feature = asof if slot == "open" else asof - timedelta(days=1)
    _expect(feature_asof == expected_feature, path, "feature_asof", "cutoff mismatch")

    delivery = document["delivery_ok"]
    _expect(delivery is None or isinstance(delivery, bool), path, "delivery_ok", "invalid")
    receipt = document["receipt_path"]
    if receipt is not None:
        _text(receipt, path, "receipt_path")
    _expect((delivery is None) == (receipt is None), path, "receipt_path", "evidence mismatch")
    basis = _text(document["execution_time_basis"], path, "execution_time_basis")
    cohort = _validate_provenance(document, path, "provenance", optional=False)

    timestamp_fields = (
        "execution_at",
        "execution_start_at",
        "scheduled_window_start",
        "scheduled_window_end",
        "path_window_start",
        "path_window_end",
        "target_day_window_start",
        "target_day_window_end",
    )
    timestamps = {
        name: _datetime_value(document[name], path, name, aware=True)
        for name in timestamp_fields
    }
    _expect(
        all(value.utcoffset() == timedelta(hours=9) for value in timestamps.values()),
        path,
        "timestamps",
        "must use KST",
    )
    labeled_at = _datetime_value(document["labeled_at"], path, "labeled_at", aware=True)
    _expect(labeled_at.utcoffset() == timedelta(0), path, "labeled_at", "must use UTC")
    target_start = datetime(asof.year, asof.month, asof.day, 9, tzinfo=KST)
    _expect(
        timestamps["target_day_window_start"] == target_start
        and timestamps["target_day_window_end"] == target_start + timedelta(days=1),
        path,
        "target_day_window",
        "asof mismatch",
    )
    scheduled = (
        (target_start - timedelta(minutes=15), target_start)
        if slot == "preopen"
        else (target_start, target_start + timedelta(minutes=30))
    )
    _expect(
        (timestamps["scheduled_window_start"], timestamps["scheduled_window_end"])
        == scheduled,
        path,
        "scheduled_window",
        "slot mismatch",
    )
    _expect(
        timestamps["path_window_start"] == timestamps["execution_start_at"]
        and timestamps["path_window_end"]
        == timestamps["path_window_start"] + timedelta(days=1)
        and timestamps["execution_start_at"] > timestamps["execution_at"],
        path,
        "path_window",
        "execution mismatch",
    )
    _expect(labeled_at >= timestamps["path_window_end"], path, "labeled_at", "immature")
    on_schedule = scheduled[0] <= timestamps["execution_at"] < scheduled[1]
    _expect(
        (cohort == FORWARD_PROVENANCE_COHORT and on_schedule)
        or (cohort == OFF_SCHEDULE_PROVENANCE_COHORT and not on_schedule)
        or (
            cohort == SCHEDULED_REPLAY_PROVENANCE_COHORT
            and on_schedule
            and delivery is None
            and basis == "scheduled_slot_fallback_snapshot_outside_window"
        ),
        path,
        "provenance",
        "schedule mismatch",
    )

    snapshot_id = _text(document["snapshot_id"], path, "snapshot_id")
    _expect(_SNAPSHOT_ID_RE.fullmatch(snapshot_id) is not None, path, "snapshot_id", "invalid")
    _digest_value(document["snapshot_payload_sha256"], path, "snapshot_payload_sha256")
    _text(document["snapshot_path"], path, "snapshot_path")
    for name in ("snapshot_model", "snapshot_rule", "snapshot_code", "snapshot_data"):
        _expect(isinstance(document[name], dict), path, name, "must be an object")
    model = document["snapshot_model"]
    _expect(
        isinstance(model.get("id"), str)
        and bool(model["id"])
        and model.get("ranking") == ranking,
        path,
        "snapshot_model",
        "identity mismatch",
    )
    _validate_label_code(document["label_code"], path)

    rows = document["rows"]
    _expect(isinstance(rows, list), path, "rows", "must be an array")
    coins: list[str] = []
    ranks: list[int] = []
    for index, row in enumerate(rows):
        _expect(isinstance(row, dict), path, f"rows[{index}]", "must be an object")
        coin, rank = _validate_modern_row(row, document, index, path)
        coins.append(coin)
        ranks.append(rank)
    _expect(ranks == list(range(1, len(rows) + 1)), path, "rows.rank", "not contiguous")
    _expect(len(coins) == len(set(coins)), path, "rows.coin", "duplicates")

    path_input = document["path_input"]
    _expect(isinstance(path_input, dict), path, "path_input", "must be an object")
    exists = path_input.get("exists")
    _expect(isinstance(exists, bool), path, "path_input.exists", "must be bool")
    input_keys = {
        "path",
        "exists",
        "start_at",
        "end_at",
        "markets",
        "rows",
        "sha256",
    }
    if exists:
        input_keys.add("schema_version")
    _exact_keys(path_input, frozenset(input_keys), path, "path_input")
    _text(path_input["path"], path, "path_input.path")
    input_start = _datetime_value(
        path_input["start_at"],
        path,
        "path_input.start_at",
        aware=False,
    )
    input_end = _datetime_value(path_input["end_at"], path, "path_input.end_at", aware=False)
    expected_input_start = timestamps["path_window_start"].replace(tzinfo=None)
    _expect(
        input_start == expected_input_start
        and input_end == expected_input_start + timedelta(days=1),
        path,
        "path_input",
        "window mismatch",
    )
    expected_markets = sorted({*coins, "KRW-BTC"})
    markets = path_input["markets"]
    _expect(
        isinstance(markets, list)
        and markets == expected_markets
        and all(
            isinstance(market, str) and _MARKET_RE.fullmatch(market)
            for market in markets
        ),
        path,
        "path_input.markets",
        "universe mismatch",
    )
    input_rows = _uint(path_input["rows"], path, "path_input.rows")
    _digest_value(path_input["sha256"], path, "path_input.sha256")
    if exists:
        _uint(path_input["schema_version"], path, "path_input.schema_version")
    else:
        _expect(input_rows == 0, path, "path_input.rows", "absent DB must have zero rows")

    _validate_summary(document["summary"], rows, status, path, legacy=False)
    _digest_value(document["label_payload_sha256"], path, "label_payload_sha256")
    if "supersedes_label_payload_sha256" in document:
        _digest_value(
            document["supersedes_label_payload_sha256"],
            path,
            "supersedes_label_payload_sha256",
        )


def _atomic_write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(
                document,
                f,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                indent=2,
            )
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


@contextmanager
def _exclusive_label_lock(path: Path):
    lock_path = path.with_name(f".{path.name}.lock")
    try:
        with file_lock(lock_path):
            yield
    except FileLockError as exc:
        raise ScoreLabelError(
            f"label artifact lock is unsafe: {lock_path}"
        ) from exc


def _label_code_manifest() -> dict:
    sources = {
        str(path.relative_to(_ROOT)): path
        for path in _LABEL_SOURCE_FILES
    }
    try:
        identities = file_set_identity(sources, root=_ROOT)
    except (OSError, ArtifactSourceChangedError) as exc:
        raise ScoreLabelError(
            "label sources cannot be hashed safely"
        ) from exc
    files = []
    for relative in sources:
        identity = identities[relative]
        if not identity.get("exists"):
            raise ScoreLabelError(f"label source is missing: {relative}")
        files.append(
            {
                "path": relative,
                "sha256": identity["sha256"],
            }
        )
    return {
        "sha256": sha256_bytes(canonical_json_bytes(files)),
        "files": files,
    }


def _path_input_manifest(
    db_path: str | Path,
    *,
    markets: list[str],
    start_at: pd.Timestamp,
) -> dict:
    """Hash exactly the rows that can affect a 96-bar assessment.

    Besides the 24-hour target window, the next benchmark boundary proves the
    final candle is closed and one prior row per market supports a possible
    first-bar flat fill.
    """
    path = Path(db_path)
    start = pd.Timestamp(start_at).tz_localize(None)
    end = start + pd.Timedelta(days=1)
    universe = sorted({*markets, "KRW-BTC"})
    manifest: dict[str, Any] = {
        "path": str(path),
        "exists": False,
        "start_at": start.isoformat(),
        "end_at": end.isoformat(),
        "markets": universe,
    }
    try:
        before = path.lstat()
    except FileNotFoundError:
        manifest["sha256"] = hashlib.sha256(_canonical_bytes([])).hexdigest()
        manifest["rows"] = 0
        return manifest
    except OSError as exc:
        raise ScoreLabelError(
            f"path input manifest cannot inspect {path}: {exc}"
        ) from exc
    if not stat.S_ISREG(before.st_mode):
        raise ScoreLabelError(f"path input is not a regular file: {path}")
    manifest["exists"] = True

    placeholders = ",".join("?" for _ in universe)
    params: list[Any] = [
        *universe,
        start.strftime("%Y-%m-%d %H:%M:%S"),
        end.strftime("%Y-%m-%d %H:%M:%S"),
        *universe,
        start.strftime("%Y-%m-%d %H:%M:%S"),
        *universe,
        end.strftime("%Y-%m-%d %H:%M:%S"),
    ]
    # ``placeholders`` contains only a generated comma-list of ``?`` tokens.
    query = f"""
        SELECT market, timestamp, open, high, low, close
        FROM candles
        WHERE market IN ({placeholders})
          AND timestamp >= ? AND timestamp < ?
        UNION ALL
        SELECT c.market, c.timestamp, c.open, c.high, c.low, c.close
        FROM candles AS c
        JOIN (
            SELECT market, MAX(timestamp) AS timestamp
            FROM candles
            WHERE market IN ({placeholders}) AND timestamp < ?
            GROUP BY market
        ) AS prior
          ON prior.market = c.market AND prior.timestamp = c.timestamp
        UNION ALL
        SELECT c.market, c.timestamp, c.open, c.high, c.low, c.close
        FROM candles AS c
        JOIN (
            SELECT market, MIN(timestamp) AS timestamp
            FROM candles
            WHERE market IN ({placeholders}) AND timestamp >= ?
            GROUP BY market
        ) AS next_boundary
          ON next_boundary.market = c.market
         AND next_boundary.timestamp = c.timestamp
        ORDER BY market, timestamp
    """  # noqa: S608
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            connection.execute("BEGIN")
            rows = connection.execute(query, params).fetchall()
            schema_version = int(
                connection.execute("PRAGMA schema_version").fetchone()[0]
            )
            connection.rollback()
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise ScoreLabelError(
            f"path input manifest failed for {path}: {exc}"
        ) from exc
    try:
        after = path.lstat()
    except OSError as exc:
        raise ScoreLabelError(
            f"path input changed while manifest was captured: {path}"
        ) from exc
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(
        getattr(before, field) != getattr(after, field)
        for field in stable_fields
    ):
        raise ScoreLabelError(
            f"path input changed while manifest was captured: {path}"
        )
    manifest.update({
        "rows": len(rows),
        "schema_version": schema_version,
        "sha256": hashlib.sha256(_canonical_bytes(rows)).hexdigest(),
    })
    return manifest


def _kst_timestamp(value: str | datetime | pd.Timestamp | None) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp.now(tz=KST)
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize(KST)
    return ts.tz_convert(KST)


def path_window(asof: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    day = pd.Timestamp(asof).normalize()
    start = (day + pd.Timedelta(hours=9)).tz_localize(KST)
    return start, start + pd.Timedelta(days=1)


def _upstream_confirms_no_observations(
    market: str,
    window_start: pd.Timestamp,
) -> bool:
    """창(96×15m) 전체 무봉이 업스트림 사실인지 확인 — 불확실은 False.

    구조적 무봉으로 인정하는 업비트 응답은 정확히 세 가지다:
    ① HTTP 404 "Code not found" — 상폐(심볼 소멸; 2026-08-05 AERGO·AQT 실측)
    ② HTTP 200 + [] — 상장 유지이나 창 이전부터 거래 없음
    ③ HTTP 200 + 봉들이 전부 창 밖 — 창 시작 전에 거래 정지
    그 외(전송 오류·비정상 응답·창 내 봉 존재)는 전부 False 로 fail-closed —
    우리 수집 갭이면 다음 실행 재시도가 치유한다.
    """
    import requests

    from data.collector_d1 import _kst_wall_to_utc_naive

    start_naive = pd.Timestamp(window_start).tz_localize(None)
    end_naive = start_naive + pd.Timedelta(days=1)
    api_to = _kst_wall_to_utc_naive(end_naive.to_pydatetime())
    start_iso = start_naive.strftime("%Y-%m-%dT%H:%M:%S")
    end_iso = end_naive.strftime("%Y-%m-%dT%H:%M:%S")
    for attempt in range(1, 4):
        try:
            response = requests.get(
                "https://api.upbit.com/v1/candles/minutes/15",
                params={
                    "market": market,
                    "count": 96,
                    "to": api_to.strftime("%Y-%m-%d %H:%M:%S"),
                },
                timeout=10,
                headers={"Accept": "application/json"},
            )
        except requests.RequestException:
            time.sleep(1.5 ** attempt)
            continue
        if response.status_code == 404 and "Code not found" in response.text:
            return True
        if response.status_code != 200:
            time.sleep(1.5 ** attempt)
            continue
        try:
            payload = response.json()
        except ValueError:
            return False
        if not isinstance(payload, list):
            return False
        if not payload:
            return True
        kst_times = [
            item.get("candle_date_time_kst")
            for item in payload
            if isinstance(item, dict)
        ]
        if len(kst_times) != len(payload) or any(t is None for t in kst_times):
            return False
        return not any(start_iso <= t < end_iso for t in kst_times)
    return False


def _scheduled_execution_at(snapshot: dict, window_start: pd.Timestamp) -> pd.Timestamp:
    """과거 재생 snapshot처럼 생성시각이 목표일 밖일 때만 쓰는 명시적 fallback."""
    if snapshot.get("slot") == "preopen":
        return window_start - pd.Timedelta(minutes=10)
    return window_start + pd.Timedelta(minutes=5)


def _execution_metadata(
    snapshot: dict,
    *,
    receipt_root: str | Path | None,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> dict:
    """실제 전달 시각을 우선해 실행 가능한 첫 15분봉을 결정한다."""
    try:
        receipt = read_delivery_receipt(snapshot, root=receipt_root)
    except DeliveryReceiptError:
        # 손상 receipt를 snapshot 시각으로 조용히 대체하면 귀속 오류가 재발한다.
        raise

    delivery_ok = receipt.get("delivery_ok") if receipt else None
    if receipt and delivery_ok:
        execution_at = _kst_timestamp(receipt["sent_at"])
        basis = "delivery_sent_at"
    else:
        execution_at = _kst_timestamp(snapshot.get("created_at"))
        basis = (
            "snapshot_created_at_delivery_failed"
            if receipt
            else "snapshot_created_at_no_receipt"
        )

    # 오늘 생성한 과거일 snapshot만 예정 시각 replay로 이동한다. 실제 receipt가
    # 있는 늦은 알림은 그 실제 시각의 24시간 경로를 유지하되 기본 forward 통계에서
    # 분리한다. 그래야 늦은 실행을 09:05 진입으로 소급하는 누수를 만들지 않는다.
    outside_target_window = not (
        window_start - pd.Timedelta(hours=1) <= execution_at < window_end
    )
    scheduled_replay = receipt is None and outside_target_window
    if scheduled_replay:
        execution_at = _scheduled_execution_at(snapshot, window_start)
        basis = "scheduled_slot_fallback_snapshot_outside_window"

    if snapshot.get("slot") == "preopen":
        scheduled_start = window_start - pd.Timedelta(minutes=15)
        scheduled_end = window_start
    else:
        scheduled_start = window_start
        scheduled_end = window_start + pd.Timedelta(minutes=30)
    on_schedule = scheduled_start <= execution_at < scheduled_end

    execution_start = max(window_start, next_bar_boundary(execution_at))
    if scheduled_replay:
        provenance = SCHEDULED_REPLAY_PROVENANCE_COHORT
    elif on_schedule:
        provenance = FORWARD_PROVENANCE_COHORT
    else:
        provenance = OFF_SCHEDULE_PROVENANCE_COHORT
        basis = f"{basis}_off_schedule"
    if execution_start >= window_end and provenance == FORWARD_PROVENANCE_COHORT:
        raise ScoreLabelError(
            f"no executable path after {execution_start.isoformat()}"
        )
    return {
        "delivery_ok": delivery_ok,
        "receipt_path": receipt.get("receipt_path") if receipt else None,
        "execution_at": execution_at.isoformat(),
        "execution_start_at": execution_start.isoformat(),
        "execution_time_basis": basis,
        "provenance_cohort": provenance,
        "forward_eligible": provenance == FORWARD_PROVENANCE_COHORT,
        "scheduled_window_start": scheduled_start.isoformat(),
        "scheduled_window_end": scheduled_end.isoformat(),
    }


def label_artifact_path(
    snapshot_file: str | Path,
    snapshot: dict,
    *,
    output_root: str | Path | None = None,
) -> Path:
    root = Path(output_root) if output_root is not None else DEFAULT_LABEL_ROOT
    return root / str(snapshot["asof"]) / Path(snapshot_file).name


def load_label_artifact(path: str | Path) -> dict:
    p = Path(path)
    try:
        document = strict_json_object(p)
    except ArtifactValidationError as exc:
        raise ScoreLabelError(f"label artifact read failed: {p}: {exc}") from exc
    _validate_label_artifact(document, p)
    if document.get("label_payload_sha256") != _artifact_digest(document):
        raise ScoreLabelError(f"label artifact checksum mismatch: {p}")
    result = dict(document)
    result["artifact_path"] = str(p)
    result["artifact_reused"] = True
    result["written"] = False
    return result


def _first_passage(
    bars: np.ndarray,
    timestamps: tuple[pd.Timestamp, ...],
    entry: float,
) -> dict:
    tp_price = entry * (1.0 + TP5)
    sl_price = entry * (1.0 - SL3)
    for index, (_, high, low, _) in enumerate(bars):
        hit_tp = high >= tp_price
        hit_sl = low <= sl_price
        if hit_sl and hit_tp:
            outcome = "sl_first_same_bar"
            before = False
        elif hit_sl:
            outcome = "sl_first"
            before = False
        elif hit_tp:
            outcome = "tp_first"
            before = True
        else:
            continue
        timestamp = timestamps[index] if index < len(timestamps) else None
        return {
            "tp5_sl3_first_passage": outcome,
            "tp5_before_sl3": before,
            "tp5_sl3_return_gross": -SL3 if hit_sl else TP5,
            "tp5_sl3_return_net": (
                (-SL3 if hit_sl else TP5) - ROUND_TRIP_COST
            ),
            "first_passage_bar": int(index),
            "first_passage_at": (
                pd.Timestamp(timestamp).isoformat() if timestamp is not None else None
            ),
        }
    eod = float(bars[-1, 3] / entry - 1.0)
    return {
        "tp5_sl3_first_passage": "neither",
        "tp5_before_sl3": None,
        "tp5_sl3_return_gross": eod,
        "tp5_sl3_return_net": eod - ROUND_TRIP_COST,
        "first_passage_bar": None,
        "first_passage_at": None,
    }


def _empty_outcome() -> dict:
    return {
        "actual_entry_open": None,
        "mfe": None,
        "mae": None,
        "eod_return": None,
        "eod_return_gross": None,
        "eod_return_net": None,
        "up5": None,
        "up10": None,
        "up20": None,
        "dn3": None,
        "dn5": None,
        "dn10": None,
        "tp5_sl3_first_passage": None,
        "tp5_before_sl3": None,
        "tp5_sl3_return_gross": None,
        "tp5_sl3_return_net": None,
        "first_passage_bar": None,
        "first_passage_at": None,
    }


def _label_candidate(
    candidate: dict,
    assessment: PathAssessment,
    *,
    snapshot_id: str,
    snapshot_hash: str,
    execution: dict,
) -> dict:
    row = {
        **candidate,
        "snapshot_id": snapshot_id,
        "snapshot_payload_sha256": snapshot_hash,
        **execution,
        **assessment.metadata(),
    }
    if not assessment.path_complete:
        return {
            **row,
            "label_status": "path_incomplete",
            "path_reason": assessment.reason,
            "path_used_bars": 0,
            **_empty_outcome(),
        }

    execution_start = _kst_timestamp(
        execution["execution_start_at"]
    ).tz_localize(None)
    try:
        timestamps = tuple(
            (
                pd.Timestamp(timestamp).tz_convert(KST).tz_localize(None)
                if pd.Timestamp(timestamp).tzinfo is not None
                else pd.Timestamp(timestamp)
            )
            for timestamp in assessment.timestamps
        )
        raw_bars = np.asarray(assessment.bars, dtype=float)
    except (TypeError, ValueError):
        timestamps = ()
        raw_bars = np.empty((0, 4), dtype=float)
    expected_timestamps = tuple(
        pd.date_range(execution_start, periods=96, freq="15min")
    )
    if (
        assessment.expected_bars != 96
        or len(timestamps) != 96
        or timestamps != expected_timestamps
        or raw_bars.shape != (96, 4)
        or not np.isfinite(raw_bars).all()
        or np.any(raw_bars <= 0)
        or np.any(
            raw_bars[:, 1]
            < np.maximum(raw_bars[:, 0], raw_bars[:, 3])
        )
        or np.any(
            raw_bars[:, 2]
            > np.minimum(raw_bars[:, 0], raw_bars[:, 3])
        )
        or np.any(raw_bars[:, 1] < raw_bars[:, 2])
    ):
        return {
            **row,
            "path_complete": False,
            "label_status": "invalid_complete_path",
            "path_reason": "assessment_contract_invalid",
            "path_used_bars": 0,
            **_empty_outcome(),
        }
    selected = [
        (timestamp, bar)
        for timestamp, bar in zip(timestamps, raw_bars)
        if pd.Timestamp(timestamp) >= execution_start
    ]
    if not selected:
        return {
            **row,
            "label_status": "no_executable_path",
            "path_reason": "no_executable_path",
            "path_used_bars": 0,
            **_empty_outcome(),
        }

    selected_timestamps = tuple(timestamp for timestamp, _ in selected)
    bars = np.asarray([bar for _, bar in selected], dtype=float)

    entry = float(bars[0, 0])
    mfe = float(np.max(bars[:, 1]) / entry - 1.0)
    mae = float(np.min(bars[:, 2]) / entry - 1.0)
    eod = float(bars[-1, 3] / entry - 1.0)
    return {
        **row,
        "label_status": "labeled",
        "path_reason": assessment.reason,
        "path_used_bars": len(bars),
        "actual_entry_open": entry,
        "mfe": mfe,
        "mae": mae,
        "eod_return": eod,
        "eod_return_gross": eod,
        "eod_return_net": eod - ROUND_TRIP_COST,
        "up5": bool(mfe >= 0.05),
        "up10": bool(mfe >= 0.10),
        "up20": bool(mfe >= 0.20),
        "dn3": bool(mae <= -0.03),
        "dn5": bool(mae <= -0.05),
        "dn10": bool(mae <= -0.10),
        **_first_passage(bars, selected_timestamps, entry),
    }


def label_recommend_snapshot(
    snapshot_file: str | Path,
    *,
    output_root: str | Path | None = None,
    db_path: str | Path = M15_DB_PATH,
    now: str | datetime | pd.Timestamp | None = None,
    assessor: Callable[..., PathAssessment] = assess_15m_window,
    receipt_root: str | Path | None = DEFAULT_RECEIPT_ROOT,
    halt_prober: Callable[[str], bool] | None = None,
) -> dict:
    """snapshot 전 유니버스를 실행 가능 경로로 라벨링하고 atomic artifact를 반환한다.

    halt_prober(market) 는 창 전체 무봉(raw_bars=0)인 종목이 업스트림에서도
    무봉임을 확인하면 True — 그 행만 halted_no_observations 구조적 종결로
    분류한다. None 이면 실제 업비트 REST 확인기를 쓴다.
    """
    source = Path(snapshot_file)
    snapshot = load_snapshot(source)
    window_start, window_end = path_window(str(snapshot["asof"]))
    execution = _execution_metadata(
        snapshot,
        receipt_root=receipt_root,
        window_start=window_start,
        window_end=window_end,
    )
    execution_start = _kst_timestamp(execution["execution_start_at"])
    label_window_start = execution_start
    label_window_end = label_window_start + pd.Timedelta(days=1)
    now_kst = _kst_timestamp(now)
    target = label_artifact_path(source, snapshot, output_root=output_root)

    if now_kst < label_window_end:
        return {
            "schema": LABEL_SCHEMA_VERSION,
            "artifact_status": "not_mature",
            "reason": "24h_path_not_closed",
            "asof": snapshot["asof"],
            "slot": snapshot["slot"],
            "snapshot_id": snapshot["snapshot_id"],
            "path_window_start": label_window_start.isoformat(),
            "path_window_end": label_window_end.isoformat(),
            "target_day_window_start": window_start.isoformat(),
            "target_day_window_end": window_end.isoformat(),
            "now": now_kst.isoformat(),
            "rows": [],
            "artifact_path": str(target),
            "artifact_reused": False,
            "written": False,
        }

    markets = [
        str(candidate.get("coin", ""))
        for candidate in snapshot.get("universe") or []
    ]
    label_code = _label_code_manifest()
    path_input_before = _path_input_manifest(
        db_path,
        markets=markets,
        start_at=label_window_start,
    )

    existing = load_label_artifact(target) if target.exists() else None
    initial_existing_digest = (
        existing.get("label_payload_sha256") if existing is not None else None
    )
    existing_same_lineage = False
    if existing is not None:
        if existing.get("snapshot_id") != snapshot.get("snapshot_id"):
            raise ScoreLabelError(
                f"label target belongs to another snapshot: {target}"
            )
        existing_same_lineage = (
            existing.get("label_code") == label_code
            and existing.get("path_input") == path_input_before
        )
        if (
            existing.get("artifact_status") == "complete"
            and existing_same_lineage
        ):
            return existing

    snapshot_id = str(snapshot["snapshot_id"])
    snapshot_hash = str(snapshot["payload_sha256"])
    if halt_prober is None:
        def prober(market: str) -> bool:
            return _upstream_confirms_no_observations(
                market, label_window_start
            )
    else:
        prober = halt_prober
    halt_confirmed: dict[str, bool] = {}
    rows = []
    for candidate in snapshot.get("universe") or []:
        market = str(candidate.get("coin", ""))
        try:
            assessment = assessor(
                market,
                label_window_start,
                db_path=db_path,
            )
            row = _label_candidate(
                candidate,
                assessment,
                snapshot_id=snapshot_id,
                snapshot_hash=snapshot_hash,
                execution=execution,
            )
        except Exception as exc:  # 한 종목 오류가 나머지 forward 표본을 막지 않게 한다.
            row = {
                **candidate,
                "snapshot_id": snapshot_id,
                "snapshot_payload_sha256": snapshot_hash,
                **execution,
                "label_status": "assessment_error",
                "path_complete": False,
                "path_quality": "assessment_error",
                "path_reason": type(exc).__name__,
                "raw_bars": None,
                "expected_bars": None,
                "flat_filled_bars": None,
                "benchmark_bars": None,
                "path_used_bars": 0,
                **_empty_outcome(),
            }
        if (
            row.get("label_status") == "path_incomplete"
            and row.get("path_reason") == "target_no_observations"
        ):
            # 창 전체 무봉 — 거래정지/상폐면 업스트림에도 봉이 없다.
            # 업스트림 확인이 될 때만 구조적 종결(halted)로 재분류하고,
            # 아니면 incomplete 유지(수집 갭 — 다음 실행 재시도).
            if market not in halt_confirmed:
                try:
                    halt_confirmed[market] = bool(prober(market))
                except Exception:
                    halt_confirmed[market] = False
            if halt_confirmed[market]:
                row = {
                    **row,
                    "label_status": "halted_no_observations",
                    "path_quality": "halted_no_observations",
                    "path_reason": "upstream_confirmed_no_observations",
                }
        rows.append(_normalise(row))

    n_labeled = sum(r.get("label_status") == "labeled" for r in rows)
    n_halted = sum(
        r.get("label_status") == "halted_no_observations" for r in rows
    )
    n_incomplete = len(rows) - n_labeled - n_halted
    status = "complete" if rows and n_incomplete == 0 else "partial"
    path_input_after = _path_input_manifest(
        db_path,
        markets=markets,
        start_at=label_window_start,
    )
    if path_input_after != path_input_before:
        raise ScoreLabelError(
            "15m path input changed while label artifact was being computed"
        )
    if _label_code_manifest() != label_code:
        raise ScoreLabelError(
            "label source changed while label artifact was being computed"
        )
    document = {
        "schema": LABEL_SCHEMA_VERSION,
        "artifact_status": status,
        "return_unit": "fraction",
        "round_trip_cost_fraction": ROUND_TRIP_COST,
        "asof": snapshot["asof"],
        "slot": snapshot["slot"],
        "ranking": snapshot.get("ranking"),
        "feature_asof": snapshot.get("feature_asof"),
        **execution,
        "snapshot_id": snapshot_id,
        "snapshot_payload_sha256": snapshot_hash,
        "snapshot_path": str(source),
        "snapshot_model": snapshot.get("model"),
        "snapshot_rule": snapshot.get("rule"),
        "snapshot_code": snapshot.get("code"),
        "snapshot_data": snapshot.get("data"),
        "label_code": label_code,
        "path_input": path_input_before,
        "path_window_start": label_window_start.isoformat(),
        "path_window_end": label_window_end.isoformat(),
        "target_day_window_start": window_start.isoformat(),
        "target_day_window_end": window_end.isoformat(),
        "labeled_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "snapshot_universe_n": int(snapshot.get("universe_n", len(rows))),
            "rows": len(rows),
            "labeled": n_labeled,
            "incomplete": n_incomplete,
            "flat_filled": sum(
                r.get("path_quality") == "flat_filled" for r in rows
            ),
            **({"halted": n_halted} if n_halted else {}),
        },
        "rows": rows,
    }
    if initial_existing_digest is not None and not existing_same_lineage:
        document["supersedes_label_payload_sha256"] = initial_existing_digest
    digest = _artifact_digest(document)
    document["label_payload_sha256"] = digest
    _validate_label_artifact(document, target)

    with _exclusive_label_lock(target):
        current = load_label_artifact(target) if target.exists() else None
        if current is not None:
            if current.get("snapshot_id") != snapshot_id:
                raise ScoreLabelError(
                    f"label target belongs to another snapshot: {target}"
                )
            current_same_lineage = (
                current.get("label_code") == label_code
                and current.get("path_input") == path_input_before
            )
            if (
                current.get("artifact_status") == "complete"
                and current_same_lineage
            ):
                return current
            if current.get("label_payload_sha256") == digest:
                return current
            current_digest = current.get("label_payload_sha256")
            if (
                current_digest != initial_existing_digest
                and not current_same_lineage
            ):
                raise ScoreLabelError(
                    f"label lineage changed concurrently: {target}"
                )
        if _label_code_manifest() != label_code:
            raise ScoreLabelError(
                "label source changed before label artifact persistence"
            )
        _atomic_write_json(target, document)
    result = dict(document)
    result["artifact_path"] = str(target)
    result["artifact_reused"] = False
    result["written"] = True
    return result

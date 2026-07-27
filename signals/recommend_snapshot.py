"""R1/R2/A1 일일 score snapshot 저장소.

한 ``asof``/``slot``/``ranking`` 조합의 ``score_candidates`` 결과를 정확히 한 번
계산해 JSON으로 원자 저장하고, 발송과 shadow ledger가 같은 결과를 읽게 한다.

이 모듈은 신호 산출물의 영속화만 담당한다. 텔레그램 발송 결과는
``notifier.delivery_receipt``가 별도 sidecar에 기록하고, ledger CSV에는 손대지 않는다.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import sqlite3
import subprocess
import tempfile
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, cast

import numpy as np
import pandas as pd

from ops.artifact_provenance import (
    ArtifactSourceChangedError,
    ArtifactValidationError,
    canonical_json_bytes,
    file_set_identity,
    sha256_file,
    strict_json_object,
)
from ops.file_lock import FileLockError, file_lock

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SNAPSHOT_ROOT = _ROOT / "output" / "recommend_snapshots"
LEGACY_SNAPSHOT_SCHEMA_VERSION = "recommend_snapshot.v1"
SNAPSHOT_SCHEMA_VERSION = "recommend_snapshot.v2"
SUPPORTED_SNAPSHOT_SCHEMA_VERSIONS = frozenset(
    {LEGACY_SNAPSHOT_SCHEMA_VERSION, SNAPSHOT_SCHEMA_VERSION}
)
RR_RATIO_EPS = 1e-3
_FOUR_DECIMAL_HALF_UNIT = 5e-5
_DISPLAY_PROBABILITY_TOLERANCE = 5.5e-4

_SCORE_SOURCE_FILES_V1 = (
    "signals/recommend.py",
    "signals/features.py",
    "signals/model_registry.py",
    "data/database.py",
    "scripts/downside_head_riskreward_v1.py",
    "scripts/recommendation_scorer_v1.py",
    "scripts/univariate_precursor_lift_v1.py",
    "scripts/regime_split_precursor_v1.py",
)
_SCORE_SOURCE_FILES = (
    *_SCORE_SOURCE_FILES_V1[:4],
    "data/market_universe.py",
    *_SCORE_SOURCE_FILES_V1[4:],
)


class SnapshotError(RuntimeError):
    """Snapshot이 손상됐거나 요청 identity와 맞지 않을 때 발생."""


def _aware_datetime(value: object, field: str, path: Path) -> datetime:
    if not isinstance(value, str):
        raise SnapshotError(f"snapshot {field} missing or non-string: {path}")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise SnapshotError(f"snapshot {field} is not ISO-8601: {path}") from exc
    if parsed.tzinfo is None:
        raise SnapshotError(f"snapshot {field} must be timezone-aware: {path}")
    return parsed


def _finite_number(value: object, field: str, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise SnapshotError(f"snapshot {field} must be numeric: {path}")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise SnapshotError(f"snapshot {field} must be numeric: {path}") from exc
    if not math.isfinite(parsed):
        raise SnapshotError(f"snapshot {field} must be finite: {path}")
    return parsed


def _iso_date(value: object, field: str, path: Path) -> date:
    if not isinstance(value, str):
        raise SnapshotError(f"snapshot {field} missing or non-string: {path}")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise SnapshotError(f"snapshot {field} is not an ISO date: {path}") from exc


def _nonnegative_int(value: object, field: str, path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SnapshotError(
            f"snapshot {field} must be a non-negative integer: {path}"
        )
    return value


def _validate_snapshot_contract(document: dict, path: Path) -> None:
    if not isinstance(document, dict):
        raise SnapshotError(f"snapshot root must be an object: {path}")

    request = document.get("request")
    model = document.get("model")
    rule = document.get("rule")
    schema = document.get("schema")
    features = document.get("features")
    training = document.get("training")
    code = document.get("code")
    data = document.get("data")
    environment = document.get("environment")
    if not all(
        isinstance(value, dict)
        for value in (
            request,
            model,
            rule,
            schema,
            features,
            training,
            code,
            data,
            environment,
        )
    ):
        raise SnapshotError(f"snapshot identity metadata missing: {path}")
    request = cast(dict[str, Any], request)
    model = cast(dict[str, Any], model)
    rule = cast(dict[str, Any], rule)
    schema = cast(dict[str, Any], schema)
    features = cast(dict[str, Any], features)
    training = cast(dict[str, Any], training)
    code = cast(dict[str, Any], code)
    data = cast(dict[str, Any], data)
    environment = cast(dict[str, Any], environment)

    asof = _iso_date(request.get("asof"), "request.asof", path)
    feature_asof = _iso_date(document.get("feature_asof"), "feature_asof", path)
    if document.get("feature_date") != document.get("feature_asof"):
        raise SnapshotError(f"snapshot feature date metadata mismatch: {path}")
    slot = request.get("slot")
    if slot == "open" and feature_asof != asof:
        raise SnapshotError(f"snapshot open feature_asof must equal asof: {path}")
    if slot == "preopen" and feature_asof != asof - timedelta(days=1):
        raise SnapshotError(
            f"snapshot preopen feature_asof must be exactly asof-1d: {path}"
        )
    if slot not in {"open", "preopen"}:
        raise SnapshotError(f"snapshot slot invalid: {path}")

    ranking = request.get("ranking")
    if ranking not in {"R1", "R2", "A1"}:
        raise SnapshotError(f"snapshot request ranking invalid: {path}")
    limit_markets = request.get("limit_markets")
    if limit_markets is not None and (
        isinstance(limit_markets, bool)
        or not isinstance(limit_markets, int)
        or limit_markets <= 0
    ):
        raise SnapshotError(f"snapshot request market limit invalid: {path}")
    if document.get("ranking") != ranking:
        raise SnapshotError(f"snapshot top-level/request ranking mismatch: {path}")

    if not isinstance(model.get("id"), str) or not model["id"]:
        raise SnapshotError(f"snapshot model id missing: {path}")
    if model.get("ranking") != ranking:
        raise SnapshotError(f"snapshot model/request ranking mismatch: {path}")
    random_seed = model.get("random_seed")
    if (
        isinstance(random_seed, bool)
        or not isinstance(random_seed, int)
        or random_seed < 0
        or random_seed != document.get("model_random_seed")
    ):
        raise SnapshotError(f"snapshot model seed metadata mismatch: {path}")
    for field in ("predict_ref", "fit_mode"):
        if not isinstance(model.get(field), str) or not model[field]:
            raise SnapshotError(f"snapshot model {field} missing: {path}")
    if not isinstance(rule.get("version"), str) or not rule["version"]:
        raise SnapshotError(f"snapshot rule version missing: {path}")
    if rule.get("version") != document.get("rule_version"):
        raise SnapshotError(f"snapshot rule version metadata mismatch: {path}")
    if rule.get("rank_basis") != document.get("rank_basis"):
        raise SnapshotError(f"snapshot rank basis metadata mismatch: {path}")
    snapshot_schema = document.get("snapshot_schema")
    if (
        snapshot_schema not in SUPPORTED_SNAPSHOT_SCHEMA_VERSIONS
        or schema.get("snapshot") != snapshot_schema
        or not isinstance(
        schema.get("score"),
        str,
        )
    ):
        raise SnapshotError(f"snapshot schema metadata invalid: {path}")
    if schema.get("score") != document.get("score_schema_version"):
        raise SnapshotError(f"snapshot score schema metadata mismatch: {path}")
    if snapshot_schema == SNAPSHOT_SCHEMA_VERSION:
        rr_ratio_eps = _finite_number(
            rule.get("rr_ratio_eps"),
            "rule.rr_ratio_eps",
            path,
        )
        if rr_ratio_eps != RR_RATIO_EPS:
            raise SnapshotError(
                f"snapshot risk-reward epsilon metadata mismatch: {path}"
            )

    sha_pattern = re.compile(r"[0-9a-f]{64}")
    source_files = code.get("score_source_files")
    sources_dirty = code.get("score_sources_dirty")
    expected_source_files = (
        _SCORE_SOURCE_FILES_V1
        if snapshot_schema == LEGACY_SNAPSHOT_SCHEMA_VERSION
        else _SCORE_SOURCE_FILES
    )
    if (
        sha_pattern.fullmatch(str(code.get("score_source_sha256"))) is None
        or source_files != list(expected_source_files)
        or not isinstance(sources_dirty, (bool, type(None)))
    ):
        raise SnapshotError(f"snapshot code provenance invalid: {path}")
    git_commit = code.get("git_commit")
    if git_commit is not None and re.fullmatch(r"[0-9a-f]{40}", str(git_commit)) is None:
        raise SnapshotError(f"snapshot git commit invalid: {path}")

    if (
        data.get("path") != "data/upbit_d1.db"
        or data.get("exists") is not True
        or sha_pattern.fullmatch(str(data.get("sha256"))) is None
        or sha_pattern.fullmatch(str(data.get("manifest_id"))) is None
        or _nonnegative_int(data.get("rows"), "data.rows", path) <= 0
        or _nonnegative_int(data.get("markets"), "data.markets", path) <= 0
    ):
        raise SnapshotError(f"snapshot D1 provenance invalid: {path}")
    manifest_payload = {
        key: value for key, value in data.items() if key != "manifest_id"
    }
    if hashlib.sha256(_canonical_bytes(manifest_payload)).hexdigest() != data.get(
        "manifest_id"
    ):
        raise SnapshotError(f"snapshot D1 manifest checksum mismatch: {path}")
    packages = environment.get("packages")
    if (
        not isinstance(environment.get("python"), str)
        or not environment["python"]
        or not isinstance(environment.get("implementation"), str)
        or not isinstance(packages, dict)
    ):
        raise SnapshotError(f"snapshot environment provenance invalid: {path}")

    feature_columns = features.get("columns")
    if (
        not isinstance(feature_columns, list)
        or any(not isinstance(item, str) or not item for item in feature_columns)
        or len(set(feature_columns)) != len(feature_columns)
        or feature_columns != document.get("feature_columns")
    ):
        raise SnapshotError(f"snapshot feature column metadata invalid: {path}")
    forbidden_feature_names = {
        "label",
        "label_tail",
        "max_return",
        "net_under_tp",
        "next_open",
        "next_high",
        "next_low",
        "next_close",
        "next_max_return",
        "next_eod_return",
        "next_max_dd",
    }
    if any(
        not item.startswith("f_")
        or item in forbidden_feature_names
        or item.startswith("next_")
        for item in feature_columns
    ):
        raise SnapshotError(f"snapshot feature columns contain leak fields: {path}")

    training_start = training.get("start")
    training_end = training.get("end")
    cutoff = _iso_date(
        training.get("cutoff_exclusive"),
        "training.cutoff_exclusive",
        path,
    )
    embargo_days = _nonnegative_int(
        training.get("embargo_days"),
        "training.embargo_days",
        path,
    )
    training_rows = _nonnegative_int(training.get("rows"), "training.rows", path)
    training_dates = _nonnegative_int(
        training.get("dates"),
        "training.dates",
        path,
    )
    if embargo_days <= 0:
        raise SnapshotError(f"snapshot training embargo must be positive: {path}")
    if cutoff > feature_asof:
        raise SnapshotError(f"snapshot training cutoff follows feature date: {path}")
    if (feature_asof - cutoff).days != embargo_days:
        raise SnapshotError(f"snapshot training embargo metadata mismatch: {path}")
    if training_rows == 0 or training_dates == 0:
        raise SnapshotError(f"snapshot training sample is empty: {path}")
    start = _iso_date(training_start, "training.start", path)
    end = _iso_date(training_end, "training.end", path)
    if not start <= end < cutoff:
        raise SnapshotError(f"snapshot training chronology invalid: {path}")
    if document.get("n_history_dates") != training_dates:
        raise SnapshotError(f"snapshot training date count mismatch: {path}")

    top3 = document.get("top3")
    universe = document.get("universe")
    if not isinstance(top3, list) or not isinstance(universe, list):
        raise SnapshotError(f"snapshot missing top3/universe arrays: {path}")
    if len(top3) != min(3, len(universe)):
        raise SnapshotError(f"snapshot top3 size is inconsistent with universe: {path}")
    if universe[:len(top3)] != top3:
        raise SnapshotError(f"snapshot top3 must be the prefix of universe: {path}")
    universe_n = document.get("universe_n")
    if (
        isinstance(universe_n, bool)
        or not isinstance(universe_n, int)
        or universe_n != len(universe)
    ):
        raise SnapshotError(
            f"snapshot universe_n mismatch: expected={len(universe)} "
            f"actual={universe_n!r} ({path})"
        )
    if rule.get("top_k") != len(top3):
        raise SnapshotError(f"snapshot rule top_k mismatch: {path}")

    expected_ranks = list(range(1, len(universe) + 1))
    ranks: list[int] = []
    coins: list[str] = []
    sl_values: list[float] = []
    tp_values: list[float] = []
    for index, candidate in enumerate(universe, start=1):
        if not isinstance(candidate, dict):
            raise SnapshotError(f"snapshot candidate {index} is not an object: {path}")
        coin = candidate.get("coin")
        if not isinstance(coin, str) or re.fullmatch(r"KRW-[A-Z0-9]+", coin) is None:
            raise SnapshotError(
                f"snapshot candidate {index} has invalid coin={coin!r}: {path}"
            )
        rank = candidate.get("rank")
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise SnapshotError(
                f"snapshot candidate {index} has invalid rank={rank!r}: {path}"
            )
        score = _finite_number(candidate.get("score"), f"{coin}.score", path)
        if not 0.0 <= score <= 1.0:
            raise SnapshotError(f"snapshot {coin}.score outside [0,1]: {path}")
        probability = _finite_number(
            candidate.get("pump_prob"),
            f"{coin}.pump_prob",
            path,
        )
        if not 0.0 <= probability <= 1.0:
            raise SnapshotError(f"snapshot {coin}.pump_prob outside [0,1]: {path}")
        entry_open = candidate.get("entry_open")
        if slot == "open":
            if entry_open is None or _finite_number(
                entry_open,
                f"{coin}.entry_open",
                path,
            ) <= 0:
                raise SnapshotError(
                    f"snapshot open {coin}.entry_open must be positive: {path}"
                )
        elif entry_open is not None:
            raise SnapshotError(
                f"snapshot preopen {coin}.entry_open must be null: {path}"
            )
        parsed_probabilities: dict[str, float] = {}
        for probability_field in ("p_up5", "p_up10", "p_up20", "p_dn5", "p_dn10"):
            value = candidate.get(probability_field)
            if value is None:
                continue
            parsed = _finite_number(value, f"{coin}.{probability_field}", path)
            if not 0.0 <= parsed <= 1.0:
                raise SnapshotError(
                    f"snapshot {coin}.{probability_field} outside [0,1]: {path}"
                )
            parsed_probabilities[probability_field] = parsed
        if snapshot_schema == SNAPSHOT_SCHEMA_VERSION:
            required_probabilities = {
                "p_up5",
                "p_up10",
                "p_up20",
                "p_dn5",
                "p_dn10",
            }
            if set(parsed_probabilities) != required_probabilities:
                raise SnapshotError(
                    f"snapshot {coin} probability vector incomplete: {path}"
                )
            if not (
                parsed_probabilities["p_up5"]
                >= parsed_probabilities["p_up10"]
                >= parsed_probabilities["p_up20"]
                and parsed_probabilities["p_dn5"]
                >= parsed_probabilities["p_dn10"]
            ):
                raise SnapshotError(
                    f"snapshot {coin} probability nesting violated: {path}"
                )
        p_up20 = parsed_probabilities.get("p_up20")
        if p_up20 is not None and not math.isclose(
            probability,
            p_up20,
            abs_tol=5e-5,
        ):
            raise SnapshotError(
                f"snapshot {coin}.pump_prob/p_up20 mismatch: {path}"
            )
        probability_text = candidate.get("pump_prob_pct")
        if (
            not isinstance(probability_text, str)
            or re.fullmatch(r"(?:100|[0-9]{1,2})\.[0-9]%", probability_text)
            is None
        ):
            raise SnapshotError(f"snapshot {coin}.pump_prob_pct mismatch: {path}")
        displayed_probability = float(probability_text[:-1]) / 100.0
        if (
            probability_text != f"{displayed_probability * 100:.1f}%"
            or
            abs(displayed_probability - probability)
            > _DISPLAY_PROBABILITY_TOLERANCE
        ):
            raise SnapshotError(
                f"snapshot {coin}.pump_prob_pct mismatch: {path}"
            )
        if not isinstance(candidate.get("dump_risk_flag"), bool):
            raise SnapshotError(
                f"snapshot {coin}.dump_risk_flag must be bool: {path}"
            )
        if not isinstance(candidate.get("btc_regime"), str):
            raise SnapshotError(f"snapshot {coin}.btc_regime missing: {path}")
        expected_downside = candidate.get("exp_downside")
        if expected_downside is not None and _finite_number(
            expected_downside,
            f"{coin}.exp_downside",
            path,
        ) > 0:
            raise SnapshotError(
                f"snapshot {coin}.exp_downside must be non-positive: {path}"
            )
        rr_ratio = candidate.get("rr_ratio")
        parsed_rr_ratio = None
        if rr_ratio is not None:
            parsed_rr_ratio = _finite_number(
                rr_ratio,
                f"{coin}.rr_ratio",
                path,
            )
            if parsed_rr_ratio < 0:
                raise SnapshotError(
                    f"snapshot {coin}.rr_ratio must be non-negative: {path}"
                )
        if snapshot_schema == SNAPSHOT_SCHEMA_VERSION:
            if parsed_rr_ratio is None:
                raise SnapshotError(
                    f"snapshot {coin}.rr_ratio is required: {path}"
                )
            up10 = parsed_probabilities["p_up10"]
            dn5 = parsed_probabilities["p_dn5"]
            up10_low = max(0.0, up10 - _FOUR_DECIMAL_HALF_UNIT)
            up10_high = min(1.0, up10 + _FOUR_DECIMAL_HALF_UNIT)
            dn5_low = max(0.0, dn5 - _FOUR_DECIMAL_HALF_UNIT)
            dn5_high = min(1.0, dn5 + _FOUR_DECIMAL_HALF_UNIT)
            ratio_low = up10_low / max(dn5_high, RR_RATIO_EPS)
            ratio_high = up10_high / max(dn5_low, RR_RATIO_EPS)
            if not (
                ratio_low - _FOUR_DECIMAL_HALF_UNIT
                <= parsed_rr_ratio
                <= ratio_high + _FOUR_DECIMAL_HALF_UNIT
            ):
                raise SnapshotError(
                    f"snapshot {coin}.rr_ratio is inconsistent with "
                    f"p_up10/p_dn5: {path}"
                )
        sl = _finite_number(candidate.get("sl"), f"{coin}.sl", path)
        tp = _finite_number(candidate.get("tp"), f"{coin}.tp", path)
        if sl >= 0 or tp <= 0:
            raise SnapshotError(f"snapshot {coin} exit contract invalid: {path}")
        feature_values = candidate.get("feature_values")
        if not isinstance(feature_values, dict) or set(feature_values) != set(
            feature_columns
        ):
            raise SnapshotError(f"snapshot {coin}.feature_values invalid: {path}")
        for feature_name, value in feature_values.items():
            if value is not None:
                _finite_number(value, f"{coin}.{feature_name}", path)
        ranks.append(rank)
        coins.append(coin)
        sl_values.append(sl)
        tp_values.append(tp)
    if ranks != expected_ranks:
        raise SnapshotError(f"snapshot candidate ranks are not contiguous: {path}")
    if len(set(coins)) != len(coins):
        raise SnapshotError(f"snapshot contains duplicate candidate coins: {path}")
    if sl_values and (
        len(set(sl_values)) != 1
        or len(set(tp_values)) != 1
        or rule.get("sl_pct") != sl_values[0]
        or rule.get("tp_pct") != tp_values[0]
    ):
        raise SnapshotError(f"snapshot exit rule metadata mismatch: {path}")

    started = _aware_datetime(document.get("decision_started_at"), "decision_started_at", path)
    completed = _aware_datetime(
        document.get("decision_completed_at"),
        "decision_completed_at",
        path,
    )
    created = _aware_datetime(document.get("created_at"), "created_at", path)
    if started > completed or created != completed:
        raise SnapshotError(f"snapshot decision timestamp chronology invalid: {path}")


def _normalise(value: Any) -> Any:
    """numpy/pandas 값을 strict JSON 값으로 바꾼다(NaN/Inf는 null)."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (datetime, date, pd.Timestamp)):
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
        _normalise(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _document_digest(document: dict) -> str:
    payload = {
        k: v for k, v in document.items()
        if k not in {"created_at", "snapshot_id", "payload_sha256", "snapshot_path"}
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _source_sha256(
    source_files: tuple[str, ...] = _SCORE_SOURCE_FILES,
) -> str:
    paths = {rel: _ROOT / rel for rel in source_files}
    try:
        identities = file_set_identity(paths, root=_ROOT)
    except (OSError, ArtifactSourceChangedError) as exc:
        raise SnapshotError("score sources cannot be hashed safely") from exc
    manifest = []
    for rel in source_files:
        identity = identities[rel]
        if not identity.get("exists"):
            raise SnapshotError(f"score source is missing: {_ROOT / rel}")
        manifest.append(
            {
                "path": rel,
                "size": identity["size"],
                "sha256": identity["sha256"],
            }
        )
    return hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()


def _git_code_metadata() -> dict:
    commit = None
    dirty = None
    try:
        # Fixed git binary and constant arguments only.
        cp = subprocess.run(  # noqa: S603
            ["/usr/bin/git", "-C", str(_ROOT), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if cp.returncode == 0:
            commit = cp.stdout.strip() or None
        status = subprocess.run(  # noqa: S603
                [
                    "/usr/bin/git",
                "-C",
                str(_ROOT),
                "status",
                "--porcelain",
                "--",
                *_SCORE_SOURCE_FILES,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if status.returncode == 0:
            dirty = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "git_commit": commit,
        "score_sources_dirty": dirty,
        "score_source_sha256": _source_sha256(),
        "score_source_files": list(_SCORE_SOURCE_FILES),
    }


def _environment_metadata() -> dict:
    packages: dict[str, str | None] = {}
    for name in ("numpy", "pandas", "scikit-learn", "xgboost"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "packages": packages,
    }


def _data_metadata() -> dict:
    """D1 input manifest captured on both sides of scoring.

    The logical summary makes the lineage inspectable while byte hashes of the
    SQLite database and WAL (when present) make silent in-place corrections
    detectable.  ``get_or_create_recommend_snapshot`` rejects a run if this
    manifest changes while the scorer is executing.
    """
    path = _ROOT / "data" / "upbit_d1.db"
    manifest: dict[str, Any] = {
        "path": str(path.relative_to(_ROOT)),
        "exists": path.exists(),
    }
    if not path.exists():
        return manifest
    stat = path.stat()
    manifest.update({
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _file_sha256(path),
    })
    wal_path = Path(f"{path}-wal")
    if wal_path.exists():
        wal_stat = wal_path.stat()
        manifest["wal"] = {
            "size_bytes": wal_stat.st_size,
            "mtime_ns": wal_stat.st_mtime_ns,
            "sha256": _file_sha256(wal_path),
        }
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            row = con.execute(
                "SELECT COUNT(*), COUNT(DISTINCT market), "
                "MIN(timestamp), MAX(timestamp) FROM candles"
            ).fetchone()
            manifest.update({
                "rows": int(row[0]),
                "markets": int(row[1]),
                "min_timestamp": row[2],
                "max_timestamp": row[3],
                "page_count": int(con.execute("PRAGMA page_count").fetchone()[0]),
                "schema_version": int(
                    con.execute("PRAGMA schema_version").fetchone()[0]
                ),
            })
        finally:
            con.close()
    except (OSError, sqlite3.Error) as exc:
        manifest["manifest_error"] = type(exc).__name__
    manifest["manifest_id"] = hashlib.sha256(
        _canonical_bytes(manifest)
    ).hexdigest()
    return manifest


def _file_sha256(path: Path) -> str:
    try:
        return sha256_file(path)
    except (OSError, ArtifactSourceChangedError) as exc:
        raise SnapshotError(
            f"snapshot input cannot be hashed safely: {path}"
        ) from exc


def snapshot_path(
    asof: str,
    slot: str,
    ranking: str = "R1",
    limit_markets: int | None = None,
    *,
    root: str | Path | None = None,
) -> Path:
    """요청 identity에 대응하는 결정론적 JSON 경로."""
    asof_norm = str(pd.Timestamp(asof).date())
    if slot not in {"open", "preopen"}:
        raise ValueError(f"slot must be open|preopen, got {slot!r}")
    ranking_norm = ranking.upper()
    if ranking_norm not in {"R1", "R2", "A1"}:
        raise ValueError(f"ranking must be R1|R2|A1, got {ranking!r}")
    if limit_markets is not None and (
        isinstance(limit_markets, bool)
        or not isinstance(limit_markets, int)
        or limit_markets <= 0
    ):
        raise ValueError("limit_markets must be a positive integer when provided")
    suffix = "" if limit_markets is None else f".limit{limit_markets}"
    base = Path(root) if root is not None else DEFAULT_SNAPSHOT_ROOT
    return base / asof_norm / f"{slot}_{ranking_norm.lower()}{suffix}.json"


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    try:
        with file_lock(lock_path):
            yield
    except FileLockError as exc:
        raise SnapshotError(
            f"snapshot lock is unsafe: {lock_path}"
        ) from exc


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


def _default_model_id(ranking: str, slot: str) -> str:
    if ranking == "R1":
        return "recommend_r1_preopen" if slot == "preopen" else "recommend_r1_open"
    if ranking == "R2":
        return "recommend_r2_open"
    return "recommend_r1_sustain_open"


def _build_document(
    result: dict,
    *,
    ranking: str,
    limit_markets: int | None,
    model_id: str | None,
    created_at: str | None,
    decision_started_at: str | None = None,
    decision_completed_at: str | None = None,
    code_metadata: dict | None = None,
    data_metadata: dict | None = None,
    environment_metadata: dict | None = None,
) -> dict:
    normalised = _normalise(result)
    resolved_slot = str(normalised["slot"])
    resolved_ranking = str(normalised.get("ranking", ranking)).upper()
    top3 = normalised.get("top3") or []
    universe = normalised.get("universe") or []
    if universe[:len(top3)] != top3:
        raise SnapshotError("score result top3 must be the prefix of universe")

    document = {
        **normalised,
        "snapshot_schema": SNAPSHOT_SCHEMA_VERSION,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "decision_started_at": decision_started_at,
        "decision_completed_at": decision_completed_at,
        "feature_asof": normalised.get("feature_date"),
        "model": {
            "id": model_id or _default_model_id(resolved_ranking, resolved_slot),
            "predict_ref": "signals.recommend:score_candidates",
            "ranking": resolved_ranking,
            "fit_mode": "ephemeral_daily_fit",
            "random_seed": normalised.get("model_random_seed"),
        },
        "rule": {
            "version": normalised.get("rule_version"),
            "rank_basis": normalised.get("rank_basis"),
            "top_k": len(top3),
            "sl_pct": top3[0].get("sl") if top3 else None,
            "tp_pct": top3[0].get("tp") if top3 else None,
            "rr_ratio_eps": RR_RATIO_EPS,
        },
        "schema": {
            "snapshot": SNAPSHOT_SCHEMA_VERSION,
            "score": normalised.get("score_schema_version"),
        },
        "features": {
            "columns": normalised.get("feature_columns") or [],
        },
        "environment": environment_metadata or _environment_metadata(),
        "data": data_metadata or _data_metadata(),
        "request": {
            "asof": normalised["asof"],
            "slot": resolved_slot,
            "ranking": resolved_ranking,
            "limit_markets": limit_markets,
        },
        "code": code_metadata or _git_code_metadata(),
    }
    digest = _document_digest(document)
    document["payload_sha256"] = digest
    document["snapshot_id"] = f"recommend-{digest[:20]}"
    return document


def load_snapshot(
    path: str | Path,
    *,
    asof: str | None = None,
    slot: str | None = None,
    ranking: str | None = None,
    limit_markets: int | None = None,
    model_id: str | None = None,
) -> dict:
    """JSON을 읽고 schema, checksum, 요청 identity를 검증한다."""
    p = Path(path)
    try:
        document = strict_json_object(p)
    except ArtifactValidationError as exc:
        raise SnapshotError(f"snapshot read failed: {p}: {exc}") from exc

    if document.get("snapshot_schema") not in SUPPORTED_SNAPSHOT_SCHEMA_VERSIONS:
        raise SnapshotError(
            f"unsupported snapshot schema: {document.get('snapshot_schema')!r}"
        )
    expected_digest = _document_digest(document)
    if document.get("payload_sha256") != expected_digest:
        raise SnapshotError(f"snapshot checksum mismatch: {p}")
    expected_snapshot_id = f"recommend-{expected_digest[:20]}"
    if document.get("snapshot_id") != expected_snapshot_id:
        raise SnapshotError(
            f"snapshot_id mismatch: expected={expected_snapshot_id!r} "
            f"actual={document.get('snapshot_id')!r} ({p})"
        )
    _validate_snapshot_contract(document, p)

    expected = {
        "asof": str(pd.Timestamp(asof).date()) if asof is not None else None,
        "slot": slot,
        "ranking": ranking.upper() if ranking is not None else None,
        "limit_markets": limit_markets,
    }
    request = document.get("request") or {}
    for key in ("asof", "slot"):
        if document.get(key) != request.get(key):
            raise SnapshotError(
                f"snapshot top-level/request mismatch {key}: "
                f"{document.get(key)!r} != {request.get(key)!r} ({p})"
            )
    model = document.get("model") or {}
    if model.get("ranking") != request.get("ranking"):
        raise SnapshotError(
            "snapshot model/request ranking mismatch: "
            f"{model.get('ranking')!r} != {request.get('ranking')!r} ({p})"
        )
    if model_id is not None and model.get("id") != model_id:
        raise SnapshotError(
            f"snapshot model identity mismatch: expected={model_id!r} "
            f"actual={model.get('id')!r} ({p})"
        )
    for key, value in expected.items():
        if value is not None and request.get(key) != value:
            raise SnapshotError(
                f"snapshot identity mismatch {key}: expected={value!r} "
                f"actual={request.get(key)!r} ({p})"
            )
    # None도 production full-universe identity의 일부다.
    if limit_markets is None and request.get("limit_markets") is not None:
        raise SnapshotError(f"snapshot is market-limited, full universe requested: {p}")

    result = dict(document)
    try:
        display_path = p.relative_to(_ROOT)
    except ValueError:
        display_path = p
    result["snapshot_path"] = str(display_path)
    return result


def get_or_create_recommend_snapshot(
    asof: str,
    *,
    slot: str = "open",
    ranking: str = "R1",
    limit_markets: int | None = None,
    model_id: str | None = None,
    root: str | Path | None = None,
    scorer: Callable[..., dict] | None = None,
) -> dict:
    """동일 identity snapshot이 있으면 재사용하고, 없을 때만 scorer를 한 번 호출한다.

    ``slot='auto'``는 호환용이다. 이미 저장된 open을 우선 재사용하고, 없으면 preopen을
    찾는다. 신규 daily 배선은 반드시 명시적인 ``open``/``preopen``을 사용해야 한다.
    """
    asof_norm = str(pd.Timestamp(asof).date())
    ranking_norm = ranking.upper()
    if slot not in {"auto", "open", "preopen"}:
        raise ValueError(f"slot must be auto|open|preopen, got {slot!r}")

    if slot == "auto":
        for candidate_slot in ("open", "preopen"):
            candidate = snapshot_path(
                asof_norm,
                candidate_slot,
                ranking_norm,
                limit_markets,
                root=root,
            )
            if candidate.exists():
                return load_snapshot(
                    candidate,
                    asof=asof_norm,
                    slot=candidate_slot,
                    ranking=ranking_norm,
                    limit_markets=limit_markets,
                    model_id=model_id,
                )
    # ``auto`` can resolve to either explicit slot.  A slot-specific lock lets
    # ``auto`` and ``open`` (or ``preopen``) fit and replace the same target
    # concurrently.  Serialize one asof/ranking/market scope through a common
    # lock; the persisted slot files remain separate.
    open_target = snapshot_path(
        asof_norm,
        "open",
        ranking_norm,
        limit_markets,
        root=root,
    )
    limit_suffix = "" if limit_markets is None else f".limit{limit_markets}"
    lock_target = open_target.with_name(
        f".recommend_{ranking_norm.lower()}{limit_suffix}.snapshot"
    )

    with _exclusive_lock(lock_target):
        # 다른 프로세스가 lock 대기 중 생성했을 수 있으므로 반드시 재검사한다.
        slots = ("open", "preopen") if slot == "auto" else (slot,)
        for candidate_slot in slots:
            candidate = snapshot_path(
                asof_norm,
                candidate_slot,
                ranking_norm,
                limit_markets,
                root=root,
            )
            if candidate.exists():
                return load_snapshot(
                    candidate,
                    asof=asof_norm,
                    slot=candidate_slot,
                    ranking=ranking_norm,
                    limit_markets=limit_markets,
                    model_id=model_id,
                )

        if scorer is None:
            from signals.recommend import score_candidates

            scorer = score_candidates
        code_before = _git_code_metadata()
        data_before = _data_metadata()
        environment = _environment_metadata()
        decision_started_at = datetime.now(timezone.utc).isoformat()
        result = scorer(
            asof_norm,
            limit_markets=limit_markets,
            slot=slot,
            ranking=ranking_norm,
        )
        decision_completed_at = datetime.now(timezone.utc).isoformat()
        code_after = _git_code_metadata()
        data_after = _data_metadata()
        if code_after != code_before:
            raise SnapshotError("score source changed while snapshot was being computed")
        if data_after != data_before:
            raise SnapshotError("D1 input changed while snapshot was being computed")
        resolved_slot = str(result.get("slot"))
        if resolved_slot not in {"open", "preopen"}:
            raise SnapshotError(f"scorer returned invalid slot: {resolved_slot!r}")
        if slot != "auto" and resolved_slot != slot:
            raise SnapshotError(
                f"scorer returned slot={resolved_slot!r}, requested {slot!r}"
            )
        if str(pd.Timestamp(result.get("asof")).date()) != asof_norm:
            raise SnapshotError(
                f"scorer returned asof={result.get('asof')!r}, requested {asof_norm!r}"
            )
        result_ranking = str(result.get("ranking", ranking_norm)).upper()
        if result_ranking != ranking_norm:
            raise SnapshotError(
                f"scorer returned ranking={result_ranking!r}, "
                f"requested {ranking_norm!r}"
            )
        target = snapshot_path(
            asof_norm,
            resolved_slot,
            ranking_norm,
            limit_markets,
            root=root,
        )
        document = _build_document(
            result,
            ranking=ranking_norm,
            limit_markets=limit_markets,
            model_id=model_id,
            created_at=decision_completed_at,
            decision_started_at=decision_started_at,
            decision_completed_at=decision_completed_at,
            code_metadata=code_before,
            data_metadata=data_before,
            environment_metadata=environment,
        )
        _validate_snapshot_contract(document, target)
        _atomic_write_json(target, document)
        return load_snapshot(
            target,
            asof=asof_norm,
            slot=resolved_slot,
            ranking=ranking_norm,
            limit_markets=limit_markets,
            model_id=model_id,
        )

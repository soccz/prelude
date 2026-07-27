"""Recommendation quality meta-filter.

This layer uses only historical closed paper/shadow ledger rows. It does not
create new model scores; it decides whether an already generated ACTIVE
candidate has enough realized evidence to stay ACTIVE.
"""
from __future__ import annotations

import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from ledger.config import ROUND_TRIP_COST_PCT
from ops.artifact_provenance import (
    ArtifactSourceChangedError,
    ArtifactValidationError,
    canonical_json_bytes,
    file_identity,
    file_set_identity,
    manifest_digest_matches,
    payload_digest,
    sha256_bytes,
    strict_json_object_bytes,
)
from ops.decision_policy import ACTIVE, WATCH_ONLY, setup_quality


MIN_META_CLOSED = 8
STRONG_META_CLOSED = 20
META_POLICY_ID = "recommendation_quality_meta_v1"
META_POLICY_VERSION = "2026-05-25.1"
DEFAULT_META_MODEL_DIR = "signals/models/ckpt/recommendation_quality_v1"
DEFAULT_META_CANDIDATE_DIR = (
    "signals/models/ckpt/recommendation_quality_candidates/latest"
)
META_ARTIFACT_SCHEMA = "prelude.recommendation_quality_model.v2"
META_TRAINING_LINEAGE_SCHEMA = 1
MODEL_ID = "recommendation_quality_meta_label_v1"
MODEL_VERSION = "2026-05-25.1"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
META_LEDGER_INPUT_KEYS = (
    "paper_ledger_distribution",
    "paper_ledger_preopen",
    "shadow_ledger_distribution",
    "shadow_ledger_preopen",
)
META_TRAINING_GENERATOR_SOURCES = (
    "scripts/train_recommendation_meta.py",
    "scripts/idea_validation_report.py",
    "ops/recommendation_quality.py",
    "ops/artifact_provenance.py",
    "ledger/config.py",
    "ledger/csv_store.py",
)

# Pickle can execute code while loading.  A live artifact is therefore trusted
# only after its complete metadata digest has been explicitly pinned in source
# as part of a user-approved promotion.  Automated training must never edit
# this allow-list. Promotion also requires artifact_status=DEPLOYED and
# deployable=true in the content-bound metadata; copying candidate files alone
# cannot activate a model. A reviewer must verify the bound training inputs,
# strict row cohort, generator sources, and promotion evidence before pinning.
APPROVED_META_ARTIFACT_SHA256: dict[tuple[str, str], str] = {}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

NUMERIC_META_FEATURES = [
    "expected_edge_pct", "calibrated_hit_pct", "source_rank", "alert_rank",
    "composite_score",
    "p_h2_3pct_4h", "p_h6_5pct_24h", "p_h5_20pct_tail",
    "p_first15_3pct", "p_first15_5pct",
    "p_first30_3pct", "p_first30_5pct",
    "p_first1h_3pct", "p_first1h_5pct",
    "log_return_1d_pct", "atr_pct_14_pct",
    "return_5d_rank", "return_7d_rank", "vol_5d_rank", "roc_3d_rank",
    "vol5_rank", "return5_rank",
]
CATEGORICAL_META_FEATURES = [
    "channel", "setup_quality", "btc_regime", "btc_context", "calibration_source",
]
META_FEATURES = NUMERIC_META_FEATURES + CATEGORICAL_META_FEATURES
FEATURE_ALIASES = {
    "composite_score": ["composite_score", "composite"],
    "p_h2_3pct_4h": ["p_h2_3pct_4h", "p_h2_hit_3_4h"],
    "p_h6_5pct_24h": ["p_h6_5pct_24h", "p_h6_hit_5_24h"],
    "p_h5_20pct_tail": ["p_h5_20pct_tail", "p_h5_tail_20"],
    "return_5d_rank": ["return_5d_rank", "return_5d"],
    "return_7d_rank": ["return_7d_rank", "return_7d"],
    "vol_5d_rank": ["vol_5d_rank", "vol_5d"],
    "roc_3d_rank": ["roc_3d_rank", "roc_3d"],
    "vol5_rank": ["vol5_rank", "vol_5d"],
    "return5_rank": ["return5_rank", "return_5d"],
}

META_COLS = [
    "base_decision",
    "meta_policy_id", "meta_policy_version",
    "meta_decision", "meta_reason",
    "trained_model_id", "trained_model_version", "trained_model_status",
    "trained_model_p_win", "trained_model_threshold",
    "confidence_score", "confidence_tier",
    "evidence_key", "evidence_n_closed",
    "evidence_net_pnl_sum_pct", "evidence_avg_net_pnl_pct",
    "evidence_tp5_hit_rate_pct", "evidence_win_rate_pct",
]


@dataclass(frozen=True)
class Evidence:
    key: str = "none"
    n_closed: int = 0
    net_pnl_sum_pct: float = np.nan
    avg_net_pnl_pct: float = np.nan
    tp5_hit_rate_pct: float = np.nan
    win_rate_pct: float = np.nan


def build_meta_feature_frame(df: pd.DataFrame, channel: str | None = None) -> pd.DataFrame:
    """Build aligned train/inference features for the recommendation meta model."""
    out = pd.DataFrame(index=df.index)
    for col in NUMERIC_META_FEATURES:
        aliases = FEATURE_ALIASES.get(col, [col])
        series = None
        for alias in aliases:
            if alias in df.columns:
                series = df[alias]
                break
        out[col] = pd.to_numeric(series, errors="coerce") if series is not None else np.nan

    for col in CATEGORICAL_META_FEATURES:
        aliases = FEATURE_ALIASES.get(col, [col])
        series = None
        for alias in aliases:
            if alias in df.columns:
                series = df[alias]
                break
        if series is None:
            if col == "channel" and channel is not None:
                out[col] = channel
            else:
                out[col] = "unknown"
        else:
            out[col] = series.fillna("unknown").astype(str)
    if channel is not None:
        out["channel"] = out["channel"].replace("", channel).fillna(channel)
    return out[META_FEATURES]


def _stable_file_bytes(path: Path) -> bytes:
    """Read one regular non-symlink file and reject a concurrent replacement."""
    before = file_identity(path, root=path.parent)
    if not before.get("exists"):
        raise ArtifactValidationError(f"artifact file missing or not regular: {path}")
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ArtifactValidationError(f"artifact file unreadable: {path}") from exc
    after = file_identity(path, root=path.parent)
    digest = sha256_bytes(content)
    if before != after or digest != before.get("sha256"):
        raise ArtifactSourceChangedError(
            f"artifact file changed during read: {path}"
        )
    return content


def meta_feature_schema_sha256() -> str:
    contract = {
        "features": META_FEATURES,
        "numeric_features": NUMERIC_META_FEATURES,
        "categorical_features": CATEGORICAL_META_FEATURES,
    }
    return sha256_bytes(canonical_json_bytes(contract))


def meta_training_row_schema_sha256() -> str:
    contract = {
        "key_columns": ["date", "channel", "coin"],
        "target_columns": ["net_pnl_pct", "target_net_win"],
        "numeric_features": NUMERIC_META_FEATURES,
        "categorical_features": CATEGORICAL_META_FEATURES,
        "missing_numeric": None,
        "outcome_contract": "tp5_sl3_ordered_first_passage_net",
        "sort": ["date", "channel", "coin"],
    }
    return sha256_bytes(canonical_json_bytes(contract))


def meta_runtime_versions() -> dict[str, str]:
    """Versions whose ABI/serialization contracts affect pickle execution."""
    return {
        "python": ".".join(map(str, sys.version_info[:3])),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }


def _valid_file_identity(identity: object) -> bool:
    if not isinstance(identity, dict) or set(identity) != {
        "path",
        "exists",
        "size",
        "sha256",
    }:
        return False
    return (
        isinstance(identity["path"], str)
        and bool(identity["path"])
        and identity["exists"] is True
        and isinstance(identity["size"], int)
        and not isinstance(identity["size"], bool)
        and identity["size"] >= 0
        and isinstance(identity["sha256"], str)
        and _SHA256_RE.fullmatch(identity["sha256"]) is not None
    )


def _validate_identity_bundle(
    bundle: object,
    *,
    required_keys: tuple[str, ...],
    label: str,
) -> dict[str, dict[str, Any]]:
    if not isinstance(bundle, dict) or not manifest_digest_matches(
        bundle,
        digest_key="bundle_sha256",
    ):
        raise ArtifactValidationError(
            f"recommendation meta {label} bundle digest mismatch"
        )
    files = bundle.get("files")
    if not isinstance(files, dict) or set(files) != set(required_keys):
        raise ArtifactValidationError(
            f"recommendation meta {label} identities are incomplete"
        )
    if not all(_valid_file_identity(identity) for identity in files.values()):
        raise ArtifactValidationError(
            f"recommendation meta {label} identity is invalid"
        )
    return files


def _validate_training_lineage(meta: dict[str, Any]) -> None:
    lineage = meta.get("training_lineage")
    if (
        not isinstance(lineage, dict)
        or lineage.get("schema_version") != META_TRAINING_LINEAGE_SCHEMA
        or not manifest_digest_matches(lineage)
    ):
        raise ArtifactValidationError(
            "recommendation meta training lineage digest/schema mismatch"
        )
    _validate_identity_bundle(
        lineage.get("ledger_inputs"),
        required_keys=META_LEDGER_INPUT_KEYS,
        label="ledger input",
    )
    recorded_generators = _validate_identity_bundle(
        lineage.get("generator_sources"),
        required_keys=META_TRAINING_GENERATOR_SOURCES,
        label="generator source",
    )
    for relative, identity in recorded_generators.items():
        if identity["path"] != relative:
            raise ArtifactValidationError(
                "recommendation meta generator source path mismatch"
            )
    try:
        current_generators = file_set_identity(
            {
                relative: PROJECT_ROOT / relative
                for relative in META_TRAINING_GENERATOR_SOURCES
            },
            root=PROJECT_ROOT,
        )
    except (OSError, ArtifactSourceChangedError) as exc:
        raise ArtifactValidationError(
            "recommendation meta generator sources are unreadable or unstable"
        ) from exc
    if current_generators != recorded_generators:
        raise ArtifactValidationError(
            "recommendation meta generator source lineage is stale"
        )

    rows = lineage.get("training_rows")
    if not isinstance(rows, dict) or set(rows) != {
        "row_schema_sha256",
        "rows_sha256",
        "n_rows",
        "n_dates",
        "date_start",
        "date_end",
    }:
        raise ArtifactValidationError(
            "recommendation meta training row contract is incomplete"
        )
    if rows.get("row_schema_sha256") != meta_training_row_schema_sha256():
        raise ArtifactValidationError(
            "recommendation meta training row schema mismatch"
        )
    rows_digest = rows.get("rows_sha256")
    if not isinstance(rows_digest, str) or _SHA256_RE.fullmatch(rows_digest) is None:
        raise ArtifactValidationError(
            "recommendation meta training row digest is invalid"
        )
    n_rows = rows.get("n_rows")
    n_dates = rows.get("n_dates")
    meta_n_samples = meta.get("n_samples")
    if (
        not isinstance(n_rows, int)
        or isinstance(n_rows, bool)
        or n_rows < 0
        or not isinstance(n_dates, int)
        or isinstance(n_dates, bool)
        or n_dates < 0
        or n_dates > n_rows
        or not isinstance(meta_n_samples, int)
        or isinstance(meta_n_samples, bool)
        or n_rows != meta_n_samples
    ):
        raise ArtifactValidationError(
            "recommendation meta training row counts are invalid"
        )
    date_start = rows.get("date_start")
    date_end = rows.get("date_end")
    date_range = meta.get("date_range")
    if not isinstance(date_range, dict):
        raise ArtifactValidationError(
            "recommendation meta date_range is invalid"
        )
    if (
        date_range.get("start") != date_start
        or date_range.get("end") != date_end
    ):
        raise ArtifactValidationError(
            "recommendation meta training date lineage mismatch"
        )
    if n_rows == 0:
        if (
            n_dates != 0
            or date_start is not None
            or date_end is not None
            or rows_digest != sha256_bytes(canonical_json_bytes([]))
        ):
            raise ArtifactValidationError(
                "empty recommendation meta training dates are inconsistent"
            )
        return
    if (
        n_dates < 1
        or not isinstance(date_start, str)
        or not isinstance(date_end, str)
    ):
        raise ArtifactValidationError(
            "recommendation meta training date range is invalid"
        )
    try:
        start = pd.Timestamp(date_start)
        end = pd.Timestamp(date_end)
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(
            "recommendation meta training date range is invalid"
        ) from exc
    if (
        pd.isna(start)
        or pd.isna(end)
        or start.time() != pd.Timestamp(0).time()
        or end.time() != pd.Timestamp(0).time()
        or start > end
        or date_start != start.strftime("%Y-%m-%d")
        or date_end != end.strftime("%Y-%m-%d")
    ):
        raise ArtifactValidationError(
            "recommendation meta training date range is invalid"
        )


def _validate_common_meta(meta: dict[str, Any]) -> None:
    if meta.get("model_id") != MODEL_ID:
        raise ArtifactValidationError("unexpected recommendation meta model_id")
    if meta.get("model_version") != MODEL_VERSION:
        raise ArtifactValidationError("unexpected recommendation meta model_version")
    if type(meta.get("deployable")) is not bool:
        raise ArtifactValidationError("recommendation meta deployable must be boolean")
    if meta.get("target") != "net_win":
        raise ArtifactValidationError("unexpected recommendation meta target")
    for key, expected in (
        ("features", META_FEATURES),
        ("numeric_features", NUMERIC_META_FEATURES),
        ("categorical_features", CATEGORICAL_META_FEATURES),
    ):
        value = meta.get(key)
        if value != expected:
            raise ArtifactValidationError(
                f"recommendation meta {key} does not match live feature contract"
            )
    if meta.get("runtime_versions") != meta_runtime_versions():
        raise ArtifactValidationError(
            "recommendation meta runtime version contract mismatch"
        )
    _validate_training_lineage(meta)


def _validate_meta_state(meta: dict[str, Any]) -> None:
    status = meta.get("artifact_status")
    validation_passed = meta.get("validation_gate_passed")
    promotion_status = meta.get("promotion_status")
    state_contracts = {
        "REJECTED": (False, False, "NOT_ELIGIBLE"),
        "CANDIDATE": (True, False, "AWAITING_USER_APPROVAL"),
        "DEPLOYED": (True, True, "APPROVED"),
    }
    if status not in state_contracts:
        raise ArtifactValidationError(
            "unexpected recommendation meta artifact_status"
        )
    if type(validation_passed) is not bool:
        raise ArtifactValidationError(
            "recommendation meta validation_gate_passed must be boolean"
        )
    expected = state_contracts[str(status)]
    observed = (
        validation_passed,
        meta["deployable"],
        promotion_status,
    )
    if observed != expected:
        raise ArtifactValidationError(
            "recommendation meta promotion state is internally inconsistent"
        )

    model_file = meta.get("model_file")
    model_hash = meta.get("model_sha256")
    threshold = meta.get("threshold")
    if model_file is None and model_hash is None:
        if status != "REJECTED" or threshold is not None:
            raise ArtifactValidationError(
                "recommendation meta may omit model only for rejected no-model state"
            )
        return
    if model_file is None or model_hash is None:
        raise ArtifactValidationError(
            "recommendation meta model_file/model_sha256 must be present together"
        )
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not np.isfinite(float(threshold))
        or not 0.0 <= float(threshold) <= 1.0
    ):
        raise ArtifactValidationError(
            "recommendation meta threshold must be finite and within [0, 1]"
        )


def _validated_generation_path(model_dir: Path, meta: dict[str, Any]) -> Path:
    model_hash = meta.get("model_sha256")
    model_name = meta.get("model_file")
    if not isinstance(model_hash, str) or _SHA256_RE.fullmatch(model_hash) is None:
        raise ArtifactValidationError("recommendation meta model_sha256 is invalid")
    expected_name = f"model.{model_hash}.pkl"
    if model_name != expected_name:
        raise ArtifactValidationError(
            "recommendation meta model_file is not content-addressed"
        )
    model_path = model_dir / expected_name
    if model_path.parent != model_dir:
        raise ArtifactValidationError("recommendation meta model path escapes artifact directory")
    return model_path


def _inactive_artifact(meta: dict[str, Any], status: str) -> dict[str, Any]:
    diagnostic_meta = dict(meta)
    diagnostic_meta["declared_deployable"] = bool(meta.get("deployable", False))
    diagnostic_meta["deployable"] = False
    diagnostic_meta["runtime_artifact_status"] = status
    return {
        "meta": diagnostic_meta,
        "model": None,
        "artifact_status": status,
    }


def _read_trained_meta_model(
    model_dir: str | Path,
    *,
    execute_approved_pickle: bool,
) -> dict | None:
    """Read and validate one meta-label artifact bundle.

    Legacy/unapproved metadata is returned for diagnostics without loading its
    pickle. Invalid versioned artifacts raise and stop the caller.
    """
    model_root = Path(model_dir)
    meta_path = model_root / "meta.json"
    try:
        meta_path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ArtifactValidationError(
            f"recommendation meta cannot be inspected: {meta_path}"
        ) from exc

    meta_bytes = _stable_file_bytes(meta_path)
    meta = strict_json_object_bytes(meta_bytes, source=meta_path)

    if meta.get("artifact_schema") != META_ARTIFACT_SCHEMA:
        # Historical artifacts predate content binding and cannot cross the
        # pickle trust boundary. Keep their card visible, but never execute it.
        return _inactive_artifact(meta, "LEGACY_UNBOUND")

    _validate_common_meta(meta)
    _validate_meta_state(meta)
    if meta.get("feature_schema_sha256") != meta_feature_schema_sha256():
        raise ArtifactValidationError(
            "recommendation meta feature schema digest mismatch"
        )
    recorded_artifact_digest = meta.get("artifact_sha256")
    if (
        not isinstance(recorded_artifact_digest, str)
        or _SHA256_RE.fullmatch(recorded_artifact_digest) is None
        or recorded_artifact_digest
        != payload_digest(meta, digest_key="artifact_sha256")
    ):
        raise ArtifactValidationError(
            "recommendation meta artifact digest mismatch"
        )

    if meta.get("model_file") is None:
        return _inactive_artifact(meta, "REJECTED")

    model_path = _validated_generation_path(model_root, meta)
    model_bytes = _stable_file_bytes(model_path)
    if sha256_bytes(model_bytes) != meta["model_sha256"]:
        raise ArtifactValidationError("recommendation meta model hash mismatch")

    # Re-observe the pointer after reading its generation. This prevents a
    # concurrent publisher from mixing metadata from one generation with model
    # bytes from another.
    if _stable_file_bytes(meta_path) != meta_bytes:
        raise ArtifactSourceChangedError(
            f"recommendation meta changed during bundle read: {meta_path}"
        )

    approval_key = (str(meta["model_id"]), str(meta["model_version"]))
    approved_digest = APPROVED_META_ARTIFACT_SHA256.get(approval_key)
    if approved_digest != recorded_artifact_digest:
        return _inactive_artifact(meta, "UNAPPROVED")
    if not meta["deployable"]:
        return _inactive_artifact(meta, "COLLECT")

    if not execute_approved_pickle:
        inspected_meta = dict(meta)
        inspected_meta["declared_deployable"] = True
        inspected_meta["runtime_artifact_status"] = "DEPLOYED"
        return {
            "meta": inspected_meta,
            "model": None,
            "artifact_status": "DEPLOYED",
        }

    # This is intentionally the final step: no untrusted pickle bytes execute
    # before strict metadata, feature contract, content hash, coherent-pair,
    # and explicit promotion approval validation all succeed.
    model = pickle.loads(model_bytes)  # noqa: S301
    predict_proba = getattr(model, "predict_proba", None)
    if not callable(predict_proba):
        raise ArtifactValidationError(
            "approved recommendation meta model lacks predict_proba"
        )
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None or list(map(str, feature_names)) != META_FEATURES:
        raise ArtifactValidationError(
            "approved recommendation meta model feature names mismatch"
        )
    classes = getattr(model, "classes_", None)
    if classes is None or list(classes) != [0, 1]:
        raise ArtifactValidationError(
            "approved recommendation meta model class contract mismatch"
        )
    return {
        "meta": meta,
        "model": model,
        "artifact_status": "DEPLOYED",
    }


def inspect_trained_meta_model(
    model_dir: str | Path = DEFAULT_META_MODEL_DIR,
) -> dict | None:
    """Inspect a complete artifact without ever executing pickle bytes."""
    return _read_trained_meta_model(
        model_dir,
        execute_approved_pickle=False,
    )


def load_trained_meta_model(
    model_dir: str | Path = DEFAULT_META_MODEL_DIR,
    *,
    enabled: bool = True,
) -> dict | None:
    """Load an explicitly approved, content-bound meta-label artifact."""
    if type(enabled) is not bool:
        raise TypeError("enabled must be boolean")
    if not enabled:
        return None
    return _read_trained_meta_model(
        model_dir,
        execute_approved_pickle=True,
    )


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(v):
        return default
    return v


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and missing:
        return default
    text = str(value)
    return text if text else default


def _net_pnl_pct(max_ret_pct: Any, close_ret_pct: Any) -> float:
    max_ret = _safe_float(max_ret_pct)
    close_ret = _safe_float(close_ret_pct)
    if np.isnan(max_ret):
        return np.nan
    if max_ret >= 5.0:
        gross = 5.0
    elif np.isnan(close_ret):
        return np.nan
    else:
        gross = close_ret
    return gross - ROUND_TRIP_COST_PCT * 100


def _load_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def _legacy_setup(row: pd.Series, channel: str) -> str:
    existing = _safe_str(row.get("setup_quality"))
    if existing:
        return existing
    if channel == "preopen":
        return "PREOPEN"
    return setup_quality(_safe_str(row.get("setup_ids")).split("+"))


def _legacy_idea(row: pd.Series, channel: str) -> str:
    existing = _safe_str(row.get("idea_id"))
    if existing:
        return existing
    q = str(row.get("setup_quality", "") or _legacy_setup(row, channel)).lower()
    regime = _safe_str(row.get("btc_regime"), "unknown")
    return f"legacy_{channel}_{q}_{regime}_v1"


def normalize_history(df: pd.DataFrame, channel: str, source: str) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame()
    out = df.copy()
    out["channel"] = out.get("channel", channel)
    out["channel"] = out["channel"].fillna(channel).astype(str)
    out["source"] = source
    out["date"] = out["date"].astype(str)
    out["coin"] = out["coin"].astype(str)
    out["decision"] = out.get("decision", ACTIVE)
    out["decision"] = out["decision"].fillna(ACTIVE).astype(str)
    out["btc_regime"] = out.get("btc_regime", "unknown")
    out["btc_regime"] = out["btc_regime"].fillna("unknown").astype(str)
    out["status"] = out.get("status", "")
    out["status"] = out["status"].fillna("").astype(str)
    out["setup_quality"] = out.apply(lambda r: _legacy_setup(r, channel), axis=1)
    out["idea_id"] = out.apply(lambda r: _legacy_idea(r, channel), axis=1)
    out["date_dt"] = pd.to_datetime(out["date"], errors="coerce")

    if channel == "distribution":
        out["max_return_pct"] = pd.to_numeric(out.get("next_max_return_pct"), errors="coerce")
        out["close_return_pct"] = pd.to_numeric(out.get("next_close_return_pct"), errors="coerce")
    else:
        out["max_return_pct"] = pd.to_numeric(out.get("first_1h_max_return_pct"), errors="coerce")
        out["close_return_pct"] = pd.to_numeric(out.get("first_1h_close_return_pct"), errors="coerce")

    out["net_pnl_pct"] = out.apply(
        lambda r: _net_pnl_pct(r.get("max_return_pct"), r.get("close_return_pct")),
        axis=1,
    )
    out["tp5_hit"] = np.where(out["max_return_pct"].notna(), out["max_return_pct"] >= 5.0, np.nan)
    out["net_win"] = np.where(out["net_pnl_pct"].notna(), out["net_pnl_pct"] > 0, np.nan)
    return out


def load_channel_history(paper_ledger: str, shadow_ledger: str, channel: str) -> pd.DataFrame:
    """Load one channel's paper + shadow rows without double-counting duplicates."""
    paper = normalize_history(_load_csv(paper_ledger), channel, "paper")
    shadow = normalize_history(_load_csv(shadow_ledger), channel, "shadow")
    hist = pd.concat([shadow, paper], ignore_index=True, sort=False)
    if len(hist) == 0:
        return hist

    hist["_closed_priority"] = hist["net_pnl_pct"].notna().astype(int)
    hist["_source_priority"] = hist["source"].map({"shadow": 1, "paper": 0}).fillna(0)
    hist = hist.sort_values(
        ["date", "channel", "coin", "_closed_priority", "_source_priority"],
        ascending=[True, True, True, False, False],
    )
    hist = hist.drop_duplicates(["date", "channel", "coin"], keep="first")
    return hist.drop(columns=["_closed_priority", "_source_priority"], errors="ignore")


def _closed_before(history: pd.DataFrame, asof: pd.Timestamp, channel: str) -> pd.DataFrame:
    if len(history) == 0:
        return history
    cutoff = pd.Timestamp(asof).normalize()
    # Legacy rows may not carry a status, but an explicitly non-closed row
    # must not become realized evidence merely because partial outcome
    # columns were populated.
    status = (
        history["status"].astype(str).str.strip().str.lower()
        if "status" in history.columns
        else pd.Series("", index=history.index, dtype=str)
    )
    coin = (
        history["coin"].astype("string").str.strip().str.lower()
        if "coin" in history.columns
        else pd.Series("", index=history.index, dtype="string")
    )
    valid_coin = coin.notna() & ~coin.fillna("").isin(
        {"", "nan", "none", "<na>", "nat"}
    )
    closed = history[
        (history["channel"].astype(str) == channel)
        & history["date_dt"].notna()
        & (history["date_dt"] < cutoff)
        & status.isin({"", "closed"})
        & valid_coin
        & history["net_pnl_pct"].notna()
    ].copy()
    return closed


def _summarize(grp: pd.DataFrame, key: str) -> Evidence:
    n = int(len(grp))
    if n == 0:
        return Evidence(key=key)
    return Evidence(
        key=key,
        n_closed=n,
        net_pnl_sum_pct=float(grp["net_pnl_pct"].sum()),
        avg_net_pnl_pct=float(grp["net_pnl_pct"].mean()),
        tp5_hit_rate_pct=float(grp["tp5_hit"].mean() * 100),
        win_rate_pct=float(grp["net_win"].mean() * 100),
    )


def _key_values(
    row: pd.Series,
    channel: str,
) -> list[tuple[str, tuple[str, ...]]]:
    idea = _safe_str(row.get("idea_id"))
    setup_q = _safe_str(row.get("setup_quality"), "UNKNOWN")
    regime = _safe_str(row.get("btc_regime"), "unknown")
    keys: list[tuple[str, tuple[str, ...]]] = []
    if idea:
        keys.append(("idea", (channel, idea)))
    keys.extend([
        ("setup_regime", (channel, setup_q, regime)),
        ("setup", (channel, setup_q)),
        ("channel", (channel,)),
    ])
    return keys


def _evidence_maps(
    closed: pd.DataFrame,
) -> dict[str, dict[tuple[str, ...], Evidence]]:
    maps: dict[str, dict[tuple[str, ...], Evidence]] = {
        "idea": {},
        "setup_regime": {},
        "setup": {},
        "channel": {},
    }
    if len(closed) == 0:
        return maps

    for key, grp in closed.groupby(["channel", "idea_id"], dropna=False):
        maps["idea"][tuple(key)] = _summarize(grp, f"idea:{'|'.join(map(str, key))}")
    for key, grp in closed.groupby(["channel", "setup_quality", "btc_regime"], dropna=False):
        maps["setup_regime"][tuple(key)] = _summarize(grp, f"setup_regime:{'|'.join(map(str, key))}")
    for key, grp in closed.groupby(["channel", "setup_quality"], dropna=False):
        maps["setup"][tuple(key)] = _summarize(grp, f"setup:{'|'.join(map(str, key))}")
    for key, grp in closed.groupby(["channel"], dropna=False):
        key_tuple = (key,) if not isinstance(key, tuple) else tuple(key)
        maps["channel"][key_tuple] = _summarize(grp, f"channel:{'|'.join(map(str, key_tuple))}")
    return maps


def _select_evidence(
    row: pd.Series,
    maps: dict[str, dict[tuple[str, ...], Evidence]],
    channel: str,
) -> Evidence:
    fallback = Evidence()
    for level, values in _key_values(row, channel):
        ev = maps.get(level, {}).get(values)
        if ev is None:
            continue
        if ev.n_closed >= MIN_META_CLOSED:
            return ev
        if ev.n_closed > fallback.n_closed:
            fallback = ev
    return fallback


def _confidence(edge_pct: float, ev: Evidence, base_decision: str) -> tuple[float, str, str]:
    if base_decision != ACTIVE:
        return 30.0, "NOT_ACTIVE", "base policy is not active"
    edge_bonus = max(-10.0, min(15.0, _safe_float(edge_pct, 0.0) * 3.0))
    if ev.n_closed == 0:
        return 52.0 + edge_bonus, "COLLECT", "no closed evidence yet"
    if ev.n_closed < MIN_META_CLOSED:
        return 55.0 + edge_bonus, "LOW_SAMPLE", f"only {ev.n_closed} closed evidence rows"
    if ev.avg_net_pnl_pct < -0.25 and ev.net_pnl_sum_pct < 0:
        return 25.0 + edge_bonus, "DOWNRANK", "historical matched evidence is negative"
    if ev.tp5_hit_rate_pct < 35.0 and ev.avg_net_pnl_pct < 0:
        return 30.0 + edge_bonus, "DOWNRANK", "matched evidence has low TP5 hit rate and negative avg"
    if ev.n_closed >= STRONG_META_CLOSED and ev.avg_net_pnl_pct > 0 and ev.tp5_hit_rate_pct >= 45.0:
        return 85.0 + edge_bonus, "HIGH", "matched evidence is positive with enough samples"
    if ev.avg_net_pnl_pct > 0:
        return 72.0 + edge_bonus, "EVIDENCE_OK", "matched evidence is positive"
    return 58.0 + edge_bonus, "WATCH_EVIDENCE", "matched evidence is mixed"


def _predict_trained_model(row: pd.Series, channel: str, trained_model: dict | None) -> dict:
    if not trained_model:
        return {
            "trained_model_id": "",
            "trained_model_version": "",
            "trained_model_status": "MISSING",
            "trained_model_p_win": np.nan,
            "trained_model_threshold": np.nan,
        }
    meta = trained_model.get("meta", {})
    model = trained_model.get("model")
    artifact_status = _safe_str(trained_model.get("artifact_status"))
    if model is None:
        return {
            "trained_model_id": meta.get("model_id", ""),
            "trained_model_version": meta.get("model_version", ""),
            "trained_model_status": artifact_status or "COLLECT",
            "trained_model_p_win": np.nan,
            "trained_model_threshold": _safe_float(meta.get("threshold")),
        }
    X = build_meta_feature_frame(pd.DataFrame([row.to_dict()]), channel=channel)
    probabilities = np.asarray(model.predict_proba(X), dtype=float)
    if (
        probabilities.shape != (1, 2)
        or not np.isfinite(probabilities).all()
        or (probabilities < 0.0).any()
        or (probabilities > 1.0).any()
        or not np.isclose(probabilities.sum(axis=1), 1.0, atol=1e-8).all()
    ):
        raise ArtifactValidationError(
            "recommendation meta predict_proba returned invalid probabilities"
        )
    p = float(probabilities[0, 1])
    status = (
        "DEPLOYED"
        if bool(meta.get("deployable", False))
        and artifact_status in {"", "DEPLOYED"}
        else "COLLECT"
    )
    return {
        "trained_model_id": meta.get("model_id", "recommendation_quality_v1"),
        "trained_model_version": meta.get("model_version", ""),
        "trained_model_status": status,
        "trained_model_p_win": p,
        "trained_model_threshold": _safe_float(meta.get("threshold"), 0.5),
    }


def apply_recommendation_quality(
    decisions: pd.DataFrame,
    history: pd.DataFrame,
    asof: pd.Timestamp,
    channel: str,
    *,
    enabled: bool = True,
    trained_model: dict | None = None,
) -> pd.DataFrame:
    """Attach confidence metadata and demote weak historical groups."""
    out = decisions.copy()
    if len(out) == 0:
        return out

    out["base_decision"] = out.get("decision", "")
    if not enabled:
        out["meta_policy_id"] = META_POLICY_ID
        out["meta_policy_version"] = META_POLICY_VERSION
        out["meta_decision"] = out["decision"]
        out["meta_reason"] = "meta filter disabled"
        out["trained_model_id"] = ""
        out["trained_model_version"] = ""
        out["trained_model_status"] = "DISABLED"
        out["trained_model_p_win"] = np.nan
        out["trained_model_threshold"] = np.nan
        out["confidence_score"] = np.nan
        out["confidence_tier"] = "DISABLED"
        out["evidence_key"] = "disabled"
        out["evidence_n_closed"] = 0
        out["evidence_net_pnl_sum_pct"] = np.nan
        out["evidence_avg_net_pnl_pct"] = np.nan
        out["evidence_tp5_hit_rate_pct"] = np.nan
        out["evidence_win_rate_pct"] = np.nan
        return out

    closed = _closed_before(history, asof, channel)
    maps = _evidence_maps(closed)
    meta_rows = []
    final_decisions = []
    final_blocked = []
    final_reasons = []

    for _, row in out.iterrows():
        base = _safe_str(row.get("decision"))
        ev = _select_evidence(row, maps, channel)
        score, tier, reason = _confidence(row.get("expected_edge_pct"), ev, base)
        model_info = _predict_trained_model(row, channel, trained_model)
        final = base
        blocked = _safe_str(row.get("blocked_reason"))
        decision_reason = _safe_str(row.get("decision_reason"))

        model_p = _safe_float(model_info["trained_model_p_win"])
        model_threshold = _safe_float(model_info["trained_model_threshold"])
        if (
            base == ACTIVE
            and model_info["trained_model_status"] == "DEPLOYED"
            and not np.isnan(model_p)
            and not np.isnan(model_threshold)
        ):
            if model_p < model_threshold:
                score = model_p * 100
                tier = "MODEL_DOWNRANK"
                reason = f"trained meta model p_win {model_p:.2f} below threshold {model_threshold:.2f}"
            elif tier != "DOWNRANK":
                # A positive model score may confirm an ACTIVE candidate, but
                # it must not erase directly observed negative PnL evidence.
                score = model_p * 100
                tier = "MODEL_OK"
                reason = f"trained meta model p_win {model_p:.2f} above threshold {model_threshold:.2f}"

        if base == ACTIVE and tier == "DOWNRANK":
            final = WATCH_ONLY
            blocked = "meta_filter_negative_evidence"
            decision_reason = f"{decision_reason}; meta: {reason}" if decision_reason else f"meta: {reason}"
        elif base == ACTIVE and tier == "MODEL_DOWNRANK":
            final = WATCH_ONLY
            blocked = "trained_meta_model_downrank"
            decision_reason = f"{decision_reason}; meta: {reason}" if decision_reason else f"meta: {reason}"
        elif base == ACTIVE:
            decision_reason = f"{decision_reason}; meta: {reason}" if decision_reason else f"meta: {reason}"

        final_decisions.append(final)
        final_blocked.append(blocked)
        final_reasons.append(decision_reason)
        meta_rows.append({
            "meta_policy_id": META_POLICY_ID,
            "meta_policy_version": META_POLICY_VERSION,
            "meta_decision": final,
            "meta_reason": reason,
            **model_info,
            "confidence_score": round(max(0.0, min(100.0, score)), 2),
            "confidence_tier": tier,
            "evidence_key": ev.key,
            "evidence_n_closed": int(ev.n_closed),
            "evidence_net_pnl_sum_pct": ev.net_pnl_sum_pct,
            "evidence_avg_net_pnl_pct": ev.avg_net_pnl_pct,
            "evidence_tp5_hit_rate_pct": ev.tp5_hit_rate_pct,
            "evidence_win_rate_pct": ev.win_rate_pct,
        })

    meta_df = pd.DataFrame(meta_rows, index=out.index)
    for col in meta_df.columns:
        out[col] = meta_df[col]
    out["decision"] = final_decisions
    out["blocked_reason"] = final_blocked
    out["decision_reason"] = final_reasons
    return out

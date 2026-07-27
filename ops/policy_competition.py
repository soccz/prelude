"""Policy competition - evaluate model + send-policy variants on forward CLOSED rows.

This is the second layer after ``champion_selector``:

* champion_selector answers: "which scorer has the best forward PnL/downside?"
* policy_competition answers: "under which send policy would this scorer have
  produced the best tradeable or pump-watchlist stream?"

It is deliberately record-only. It sends no Telegram messages, places no orders,
and changes no champion state. The output is a daily audit artifact for deciding
whether a policy should become LIVE / WATCH / SILENT later.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sqlite3
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ledger.config import (  # noqa: E402
    ROUND_TRIP_COST_CONSERVATIVE_PP,
    ROUND_TRIP_COST_PP,
)
from ledger.exit_lab import EXIT_VARIANTS  # noqa: E402
from ops.artifact_provenance import (  # noqa: E402
    ArtifactSourceChangedError,
    ArtifactValidationError,
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    manifest_digest_matches,
    payload_digest,
    sha256_bytes,
    source_bundle_identity,
    strict_json_object,
    with_manifest_digest,
)
from ops.close_input_gate import (  # noqa: E402
    CLOSE_EVIDENCE_ACTIVATION_DATE,
)
from ops.file_lock import file_lock  # noqa: E402
from ops.radar_verdict import (  # noqa: E402
    RADAR_TERMINAL_STATE,
    verdict_status_for,
)
from ops.v2_provenance import (  # noqa: E402
    DEFAULT_LEDGER_PATH as DEFAULT_PUMP_V2_LEDGER_PATH,
    DEFAULT_RECEIPT_ROOT as DEFAULT_PUMP_V2_RECEIPT_ROOT,
    validate_v2_provenance,
)
from signals.model_registry import MODELS, ModelSpec  # noqa: E402
from data.market_universe import (  # noqa: E402
    is_excluded_signal_market,
)
from scripts.pump_detector_today import (  # noqa: E402
    _validate_decision_document as _validate_pump_v1_decision_document,
)
from signals.recommend_snapshot import (  # noqa: E402
    DEFAULT_SNAPSHOT_ROOT,
    load_snapshot,
)

OUT_CSV = _ROOT / "output" / "policy_competition_summary.csv"
OUT_JSON = _ROOT / "output" / "policy_competition_summary.json"
D1_DB = _ROOT / "data" / "upbit_d1.db"
POLICY_DB = _ROOT / "data" / "policy_competition.db"
PUMP_V1_DECISION_ROOT = _ROOT / "output" / "pump_v1_decisions"
PUMP_V2_DECISION_ROOT = _ROOT / "output" / "pump_v2_decisions"
PUMP_V2_RECEIPT_ROOT = DEFAULT_PUMP_V2_RECEIPT_ROOT
RADAR_VERDICT_PATH = RADAR_TERMINAL_STATE

# 왕복 거래비용 (%p 단위) — ledger/config.py 단일 출처 (= 0.15)
ROUND_TRIP_COST_PCT = ROUND_TRIP_COST_PP
PUMP20_THRESH = 0.20
DEEP_LOSS_PCT = -5.0
POLICY_ARTIFACT_SCHEMA = "policy_competition.v2"
POLICY_INPUT_MANIFEST_SCHEMA = "policy_competition_inputs.v2"
POLICY_CSV_METADATA_COLUMNS = [
    "artifact_schema",
    "run_id",
    "payload_sha256",
]
POLICY_ROW_COLUMNS = [
    "asof",
    "participant_id",
    "source_id",
    "policy_id",
    "objective",
    "description",
    "n_closed",
    "n_days",
    "n_selected_days",
    "net_sum_pct",
    "net_mean_pct",
    "deep_loss_freq_pct",
    "sl_rate_pct",
    "tp_rate_pct",
    "pump20_precision_pct",
    "post_send_pump20_precision_pct",
    "post_send_label_n",
    "pump20_recall_pct",
    "pump20_captured",
    "pump20_actual",
    "pump_days",
    "pump_days_any_captured",
    "pump_day_capture_rate_pct",
    "recall_date_basis",
    "pump20_label_basis",
]
POLICY_SQLITE_ROW_SELECT = """
SELECT
    asof, participant_id, source_id, policy_id, objective, description,
    n_closed, n_days, n_selected_days, net_sum_pct, net_mean_pct,
    deep_loss_freq_pct, sl_rate_pct, tp_rate_pct, pump20_precision_pct,
    post_send_pump20_precision_pct, post_send_label_n, pump20_recall_pct,
    pump20_captured, pump20_actual, pump_days, pump_days_any_captured,
    pump_day_capture_rate_pct, recall_date_basis, pump20_label_basis,
    run_id, payload_sha256
FROM policy_competition_rows
WHERE asof=?
ORDER BY participant_id
"""
POLICY_GENERATOR_SOURCES = (
    "ops/policy_competition.py",
    "ops/artifact_provenance.py",
    "ledger/config.py",
    "ledger/exit_lab.py",
    "ops/close_input_gate.py",
    "ops/radar_verdict.py",
    "ops/v2_provenance.py",
    "signals/model_registry.py",
    "data/market_universe.py",
    "scripts/pump_detector_today.py",
    "scripts/pump_detector_v2_today.py",
    "signals/pump_detector_v1.py",
    "signals/pump_detector_v2.py",
    "signals/recommend_snapshot.py",
)

DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS policy_competition_runs (
    asof TEXT PRIMARY KEY,
    artifact_schema TEXT NOT NULL,
    run_id TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    input_manifest_sha256 TEXT NOT NULL,
    generated_at_utc TEXT NOT NULL,
    pump20_threshold REAL NOT NULL,
    deep_loss_pct REAL NOT NULL,
    round_trip_cost_pct REAL NOT NULL,
    row_count INTEGER NOT NULL,
    best_pump_participant TEXT,
    best_net_participant TEXT,
    updated_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS policy_competition_rows (
    asof TEXT NOT NULL,
    run_id TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    participant_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    policy_id TEXT NOT NULL,
    objective TEXT NOT NULL,
    description TEXT,
    n_closed INTEGER NOT NULL,
    n_days INTEGER NOT NULL,
    n_selected_days INTEGER NOT NULL DEFAULT 0,
    net_sum_pct REAL,
    net_mean_pct REAL,
    deep_loss_freq_pct REAL,
    sl_rate_pct REAL,
    tp_rate_pct REAL,
    pump20_precision_pct REAL,
    post_send_pump20_precision_pct REAL,
    post_send_label_n INTEGER NOT NULL DEFAULT 0,
    pump20_recall_pct REAL,
    pump20_captured INTEGER NOT NULL,
    pump20_actual INTEGER NOT NULL,
    pump_days INTEGER NOT NULL,
    pump_days_any_captured INTEGER NOT NULL,
    pump_day_capture_rate_pct REAL,
    recall_date_basis TEXT NOT NULL DEFAULT 'ledger_rows',
    pump20_label_basis TEXT NOT NULL DEFAULT '',
    generated_at_utc TEXT NOT NULL,
    PRIMARY KEY (asof, participant_id)
);

CREATE INDEX IF NOT EXISTS idx_policy_competition_rows_source
ON policy_competition_rows(source_id, policy_id);

CREATE VIEW IF NOT EXISTS policy_competition_latest_rows AS
SELECT *
FROM policy_competition_rows
WHERE asof = (SELECT MAX(asof) FROM policy_competition_runs);
"""


@dataclass(frozen=True)
class PolicySpec:
    policy_id: str
    description: str
    objective: str
    predicate: Callable[[pd.DataFrame], pd.Series]


def _true_series(df: pd.DataFrame) -> pd.Series:
    return pd.Series(True, index=df.index)


def _rank_le(n: int) -> Callable[[pd.DataFrame], pd.Series]:
    def pred(df: pd.DataFrame) -> pd.Series:
        if "rank" not in df.columns:
            return pd.Series(False, index=df.index)
        rank = pd.to_numeric(df["rank"], errors="coerce")
        return (
            rank.notna()
            & np.isfinite(rank)
            & rank.ge(1)
            & rank.le(n)
            & rank.mod(1).eq(0)
        )

    return pred


def _no_dump(df: pd.DataFrame) -> pd.Series:
    if "dump_risk_flag" not in df.columns:
        return pd.Series(False, index=df.index)
    # Unknown/blank values are not affirmative evidence that dump risk is
    # absent.  The policy's contract says the flag must be false.
    s = df["dump_risk_flag"].astype(str).str.strip().str.lower()
    return s.isin(["false", "0", "no"])


def _rr_min(threshold: float) -> Callable[[pd.DataFrame], pd.Series]:
    def pred(df: pd.DataFrame) -> pd.Series:
        if "rr_ratio" not in df.columns:
            return pd.Series(False, index=df.index)
        rr = pd.to_numeric(df["rr_ratio"], errors="coerce")
        return rr.notna() & np.isfinite(rr) & (rr >= threshold)

    return pred


def _pump_prob_min(threshold: float) -> Callable[[pd.DataFrame], pd.Series]:
    def pred(df: pd.DataFrame) -> pd.Series:
        col = "p_up20" if "p_up20" in df.columns else "pump_prob"
        if col not in df.columns:
            return pd.Series(False, index=df.index)
        prob = pd.to_numeric(df[col], errors="coerce")
        return (
            prob.notna()
            & np.isfinite(prob)
            & prob.ge(threshold)
            & prob.le(1.0)
        )

    return pred


POLICIES: list[PolicySpec] = [
    PolicySpec(
        "top_all",
        "Use every recorded pick from that source ledger.",
        "baseline",
        _true_series,
    ),
    PolicySpec(
        "top1_only",
        "Send only rank #1.",
        "precision_pnl",
        _rank_le(1),
    ),
    PolicySpec(
        "top2_only",
        "Send only ranks #1-2.",
        "precision_pnl",
        _rank_le(2),
    ),
    PolicySpec(
        "no_dump_top3",
        "Send top-3 only when dump_risk_flag is false.",
        "downside_control",
        lambda df: _rank_le(3)(df) & _no_dump(df),
    ),
    PolicySpec(
        "rr_ge_0_75",
        "Send only picks with rr_ratio >= 0.75.",
        "downside_control",
        _rr_min(0.75),
    ),
    PolicySpec(
        "pump_prob_ge_3pct",
        "Pump watchlist: require train-only P(+20%) estimate >= 3% "
        "(not strict calibrated probability).",
        "pump_recall",
        _pump_prob_min(0.03),
    ),
]


def _rank_col(df: pd.DataFrame) -> str | None:
    for col in ("rank", "alert_rank", "source_rank"):
        if col in df.columns:
            return col
    return None


def _coin_col(df: pd.DataFrame) -> str | None:
    for col in ("coin", "market"):
        if col in df.columns:
            return col
    return None


def _is_canonical_v2_ledger(path: Path) -> bool:
    return path.resolve() == DEFAULT_PUMP_V2_LEDGER_PATH.resolve()


def _load_model_rows(
    spec: ModelSpec,
    asof: pd.Timestamp,
    *,
    candle_db: Path = D1_DB,
    pump_v1_decision_root: Path = PUMP_V1_DECISION_ROOT,
    pump_v2_decision_root: Path = PUMP_V2_DECISION_ROOT,
    pump_v2_receipt_root: Path = PUMP_V2_RECEIPT_ROOT,
) -> pd.DataFrame:
    path = spec.abs_ledger_path()
    v2_audit = None
    if not path.exists():
        if spec.id == "pump_hunter_v2":
            validate_v2_provenance(
                [],
                asof=asof.date(),
                decision_root=pump_v2_decision_root,
                receipt_root=pump_v2_receipt_root,
                enforce_legacy_baseline=_is_canonical_v2_ledger(path),
            )
        return pd.DataFrame()
    df = pd.read_csv(path)
    if spec.id == "pump_hunter_v2":
        v2_audit = validate_v2_provenance(
            df.to_dict(orient="records"),
            asof=asof.date(),
            decision_root=pump_v2_decision_root,
            receipt_root=pump_v2_receipt_root,
            enforce_legacy_baseline=_is_canonical_v2_ledger(path),
        )
    metric = spec.metric
    needed = [metric.status_col, metric.date_col, metric.realized_pct_col]
    if any(col not in df.columns for col in needed):
        return pd.DataFrame()
    coin_col = _coin_col(df)
    if coin_col is None:
        return pd.DataFrame()

    out = df[df[metric.status_col].astype(str) == metric.closed_value].copy()
    if out.empty:
        return out

    parsed_dates = pd.to_datetime(out[metric.date_col], errors="coerce")
    if parsed_dates.isna().any():
        raise ValueError(f"{spec.id}: CLOSED row has invalid date")
    out["date"] = parsed_dates.dt.normalize()
    # daily_close runs shortly after the next KST open. The asof trading day
    # itself is still incomplete and must never enter forward evaluation.
    out = out[out["date"] < asof]
    if out.empty:
        return out

    out["coin"] = out[coin_col].astype(str)
    if out["coin"].eq("").any() or out["coin"].eq("nan").any():
        raise ValueError(f"{spec.id}: CLOSED row has invalid coin")
    if out.duplicated(["date", "coin"]).any():
        raise ValueError(f"{spec.id}: duplicate CLOSED date/coin rows")
    if spec.id == "pump_hunter":
        out = _validated_pump_v1_closed_rows(
            out,
            asof=asof,
            decision_root=pump_v1_decision_root,
        )
        if out.empty:
            return out
    elif spec.id == "pump_hunter_v2":
        if v2_audit is None:
            raise AssertionError("v2 provenance audit is unavailable")
        verified = {
            (pd.Timestamp(day).normalize(), coin)
            for day, coin in v2_audit.verified_closed_positions
        }
        out = out[
            [
                (pd.Timestamp(row["date"]).normalize(), str(row["coin"]))
                in verified
                for _, row in out.iterrows()
            ]
        ].copy()
        if out.empty:
            return out
    out["model_id"] = spec.id
    out["source_name"] = spec.name

    realized = pd.to_numeric(out[metric.realized_pct_col], errors="coerce")
    if realized.isna().any() or not np.isfinite(realized.to_numpy(dtype=float)).all():
        raise ValueError(f"{spec.id}: CLOSED row has invalid realized return")
    if not metric.cost_already_deducted:
        realized = realized - ROUND_TRIP_COST_PCT
    out["net_pct"] = realized

    hit = _selected_pump20_hits(out[["date", "coin"]], candle_db)
    out = out.merge(hit, on=["date", "coin"], how="left")

    rank_col = _rank_col(out)
    if rank_col and rank_col != "rank":
        out["rank"] = pd.to_numeric(out[rank_col], errors="coerce")
    elif "rank" in out.columns:
        out["rank"] = pd.to_numeric(out["rank"], errors="coerce")
    else:
        out["rank"] = np.nan

    return out


_SNAPSHOT_IDENTITIES = {
    "recommend_r1_open": ("open", "R1"),
    "recommend_r2_open": ("open", "R2"),
    "recommend_r1_sustain_open": ("open", "A1"),
    "recommend_r1_preopen": ("preopen", "R1"),
}


def _pump_v1_manifest_map(
    root: Path,
    asof: pd.Timestamp,
    *,
    model_id: str,
) -> dict[pd.Timestamp, tuple[dict, Path, str]]:
    """Load only post-contract canonical-forward v1 decision manifests."""
    manifests: dict[pd.Timestamp, tuple[dict, Path, str]] = {}
    if not root.exists():
        return manifests
    for manifest in sorted(root.glob("*.json")):
        try:
            manifest_day = date.fromisoformat(manifest.stem)
        except ValueError as exc:
            raise ValueError(
                f"{model_id}: invalid decision manifest name {manifest.name!r}"
            ) from exc
        path_date = pd.Timestamp(manifest_day)
        if (
            manifest_day < CLOSE_EVIDENCE_ACTIVATION_DATE
            or path_date >= asof
        ):
            continue
        try:
            payload = strict_json_object(manifest)
        except ArtifactValidationError as exc:
            raise ValueError(
                f"{model_id}: invalid decision manifest {manifest}"
            ) from exc
        decision = payload.get("decision")
        if (
            not isinstance(decision, dict)
            or payload.get("asof") != manifest.stem
            or decision.get("asof") != manifest.stem
            or decision.get("model_id") != model_id
        ):
            raise ValueError(
                f"{model_id}: decision manifest identity mismatch {manifest}"
            )
        try:
            validated = _validate_pump_v1_decision_document(
                payload,
                decision,
                manifest,
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{model_id}: decision manifest invalid {manifest}: {exc}"
            ) from exc
        decision_id = payload.get("decision_id")
        if not isinstance(decision_id, str) or not decision_id:
            raise ValueError(
                f"{model_id}: decision id missing {manifest}"
            )
        manifests[path_date] = (validated, manifest, decision_id)
    return manifests


def _resolved_evidence_path(value: object) -> Path | None:
    if value is None or pd.isna(value):
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = _ROOT / path
    return path.resolve(strict=False)


def _validated_pump_v1_closed_rows(
    rows: pd.DataFrame,
    *,
    asof: pd.Timestamp,
    decision_root: Path,
) -> pd.DataFrame:
    """Exclude legacy rows and authenticate every post-contract CLOSED row."""
    activation = pd.Timestamp(CLOSE_EVIDENCE_ACTIVATION_DATE)
    current = rows[rows["date"] >= activation].copy()
    if current.empty:
        return current
    manifests = _pump_v1_manifest_map(
        decision_root,
        asof,
        model_id="pump_hunter",
    )
    for index, row in current.iterrows():
        day = pd.Timestamp(row["date"]).normalize()
        entry = manifests.get(day)
        if entry is None:
            raise ValueError(
                f"pump_hunter: CLOSED row has no canonical decision "
                f"manifest for {day.date()}"
            )
        decision, manifest_path, decision_id = entry
        try:
            rank_number = float(row.get("rank"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"pump_hunter: CLOSED row rank invalid for {day.date()}"
            ) from exc
        if (
            not math.isfinite(rank_number)
            or not rank_number.is_integer()
            or rank_number < 1
        ):
            raise ValueError(
                f"pump_hunter: CLOSED row rank invalid for {day.date()}"
            )
        candidate_keys = {
            (str(candidate["market"]), int(candidate["rank"]))
            for candidate in decision["candidates"]
        }
        row_key = (str(row["coin"]), int(rank_number))
        if row_key not in candidate_keys:
            raise ValueError(
                f"pump_hunter: CLOSED row is not in decision manifest: "
                f"{day.date()}/{row_key}"
            )
        if row.get("snapshot_id") != decision_id:
            raise ValueError(
                f"pump_hunter: CLOSED row decision id mismatch for {day.date()}"
            )
        if _resolved_evidence_path(row.get("snapshot_path")) != (
            manifest_path.resolve(strict=False)
        ):
            raise ValueError(
                f"pump_hunter: CLOSED row decision path mismatch for {day.date()}"
            )
    return current


def _source_observed_dates(
    spec: ModelSpec,
    asof: pd.Timestamp,
    *,
    snapshot_root: Path = DEFAULT_SNAPSHOT_ROOT,
    pump_v1_decision_root: Path = PUMP_V1_DECISION_ROOT,
    pump_v2_decision_root: Path = PUMP_V2_DECISION_ROOT,
    pump_v2_receipt_root: Path = PUMP_V2_RECEIPT_ROOT,
) -> tuple[list[pd.Timestamp], str]:
    """Return completed decision dates, preferring validated score snapshots.

    Ledger rows cover legacy sources but cannot represent a scorer run that
    selected zero candidates. Recommendation snapshots do, so their validated
    dates are unioned into the denominator when available.
    """
    if spec.id == "pump_hunter_v2":
        ledger_path = spec.abs_ledger_path()
        ledger_rows = (
            pd.read_csv(ledger_path).to_dict(orient="records")
            if ledger_path.exists()
            else []
        )
        audit = validate_v2_provenance(
            ledger_rows,
            asof=asof.date(),
            decision_root=pump_v2_decision_root,
            receipt_root=pump_v2_receipt_root,
            enforce_legacy_baseline=_is_canonical_v2_ledger(ledger_path),
        )
        return (
            [
                pd.Timestamp(value).normalize()
                for value in audit.healthy_dates
            ],
            audit.recall_date_basis,
        )
    if spec.id == "pump_hunter":
        manifests = _pump_v1_manifest_map(
            pump_v1_decision_root,
            asof,
            model_id=spec.id,
        )
        return (
            sorted(manifests),
            "post_contract_validated_forward_decisions;"
            "legacy_dates_excluded",
        )

    dates: set[pd.Timestamp] = set()
    basis = [
        "closed_ledger_rows"
        if spec.id == "pump_hunter_v2"
        else "ledger_rows"
    ]
    path = spec.abs_ledger_path()
    if path.exists():
        wanted = {spec.metric.date_col}
        if spec.id == "pump_hunter_v2":
            wanted.add(spec.metric.status_col)
        frame = pd.read_csv(path, usecols=lambda col: col in wanted)
        if spec.metric.date_col not in frame.columns:
            raise ValueError(f"{spec.id}: ledger date column missing")
        if spec.id == "pump_hunter_v2":
            if spec.metric.status_col not in frame.columns:
                raise ValueError(f"{spec.id}: ledger status column missing")
            frame = frame[
                frame[spec.metric.status_col].astype(str)
                == spec.metric.closed_value
            ]
        raw = frame[spec.metric.date_col]
        parsed = pd.to_datetime(raw, errors="coerce")
        invalid = raw.notna() & parsed.isna()
        if invalid.any():
            raise ValueError(f"{spec.id}: ledger has invalid decision date")
        dates.update(
            pd.Timestamp(value).normalize()
            for value in parsed.dropna()
            if pd.Timestamp(value).normalize() < asof
        )

    identity = _SNAPSHOT_IDENTITIES.get(spec.id)
    if identity and snapshot_root.exists():
        slot, ranking = identity
        filename = f"{slot}_{ranking.lower()}.json"
        matched = list(snapshot_root.glob(f"*/{filename}"))
        if matched:
            basis.append("validated_score_snapshots")
        for snapshot_file in matched:
            try:
                path_date = pd.Timestamp(snapshot_file.parent.name).normalize()
            except ValueError as exc:
                raise ValueError(
                    f"{spec.id}: invalid snapshot date directory "
                    f"{snapshot_file.parent.name!r}"
                ) from exc
            if path_date >= asof:
                continue
            document = load_snapshot(
                snapshot_file,
                asof=snapshot_file.parent.name,
                slot=slot,
                ranking=ranking,
                limit_markets=None,
                model_id=spec.id,
            )
            decision_date = pd.Timestamp(document["asof"]).normalize()
            dates.add(decision_date)

    return sorted(dates), "+".join(basis)


def _empty_model_rows() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "coin",
            "net_pct",
            "selected_pump20_hit",
            "post_send_pump20_hit",
            "rank",
            "dump_risk_flag",
            "rr_ratio",
            "p_up20",
            "pump_prob",
            "exit_reason",
            "model_id",
        ]
    )


def _selected_pump20_hits(keys: pd.DataFrame, db_path: Path = D1_DB) -> pd.DataFrame:
    if keys.empty or not db_path.exists():
        return pd.DataFrame(columns=["date", "coin", "selected_pump20_hit"])
    unique = keys.drop_duplicates(["date", "coin"]).copy()
    unique["date"] = pd.to_datetime(unique["date"], errors="raise").dt.normalize()
    requested = [
        (
            pd.Timestamp(item["date"]),
            str(item["coin"]),
            pd.Timestamp(item["date"]).strftime("%Y-%m-%d 09:00:00"),
        )
        for _, item in unique.iterrows()
    ]
    hits: dict[tuple[str, str], float | int] = {}
    con = sqlite3.connect(
        f"{db_path.resolve().as_uri()}?mode=ro",
        uri=True,
    )
    try:
        # Fetch selected keys in bounded batches instead of issuing one SQLite
        # query per recommendation row.  The VALUES CTE preserves exact-key
        # semantics while staying below SQLite's common 999-parameter limit.
        for start in range(0, len(requested), 400):
            batch = requested[start:start + 400]
            placeholders = ",".join("(?, ?)" for _ in batch)
            params = [
                value
                for _, coin, timestamp in batch
                for value in (timestamp, coin)
            ]
            query = (
                f"WITH requested(timestamp, market) AS (VALUES {placeholders}) "
                "SELECT requested.timestamp, requested.market, "
                "candles.open, candles.high "
                "FROM requested LEFT JOIN candles "
                "ON candles.timestamp=requested.timestamp "
                "AND candles.market=requested.market"
            )
            for timestamp, coin, open_value, high_value in con.execute(
                query,
                params,
            ):
                hit: float | int = np.nan
                if open_value is not None and high_value is not None:
                    open_number = float(open_value)
                    if open_number > 0:
                        hit = int(
                            float(high_value) / open_number - 1.0
                            >= PUMP20_THRESH
                        )
                hits[(str(timestamp), str(coin))] = hit
    finally:
        con.close()
    return pd.DataFrame([
        {
            "date": day,
            "coin": coin,
            "selected_pump20_hit": hits.get((timestamp, coin), np.nan),
        }
        for day, coin, timestamp in requested
    ])


def _actual_pumps_for_dates(dates: list[pd.Timestamp],
                            db_path: Path = D1_DB) -> dict[pd.Timestamp, set[str]]:
    normalized_dates = sorted(
        set(pd.Timestamp(value).normalize() for value in dates)
    )
    if not normalized_dates or not db_path.exists():
        return {}
    out: dict[pd.Timestamp, set[str]] = {
        value: set() for value in normalized_dates
    }
    timestamps = {
        value.strftime("%Y-%m-%d 09:00:00"): value
        for value in normalized_dates
    }
    con = sqlite3.connect(
        f"{db_path.resolve().as_uri()}?mode=ro",
        uri=True,
    )
    try:
        # Preserve the exact date denominator without one SQLite round trip
        # per trading day. Bounded IN batches stay below the usual SQLite
        # host-parameter limit.
        timestamp_values = list(timestamps)
        for start in range(0, len(timestamp_values), 900):
            batch = timestamp_values[start:start + 900]
            placeholders = ",".join("?" for _ in batch)
            rows = con.execute(
                "SELECT timestamp, market FROM candles "
                f"WHERE timestamp IN ({placeholders}) AND open > 0 "
                "AND (high / open - 1.0) >= ?",
                [*batch, PUMP20_THRESH],
            ).fetchall()
            for timestamp, market in rows:
                market_name = str(market)
                if (
                    market_name.startswith("KRW-")
                    and not is_excluded_signal_market(market_name)
                ):
                    out[timestamps[str(timestamp)]].add(market_name)
    finally:
        con.close()
    return out


def _safe_pct(x: float | None) -> float | None:
    if x is None or not np.isfinite(x):
        return None
    return round(float(x), 4)


class PolicyArtifactError(ArtifactValidationError):
    """The policy artifact triplet is incomplete, stale, or inconsistent."""


def _policy_config() -> dict[str, Any]:
    return {
        "pump20_threshold": PUMP20_THRESH,
        "deep_loss_pct": DEEP_LOSS_PCT,
        "round_trip_cost_pct": ROUND_TRIP_COST_PCT,
        "round_trip_cost_conservative_pct": ROUND_TRIP_COST_CONSERVATIVE_PP,
        "note": (
            "Record-only policy competition. It evaluates CLOSED forward rows. "
            "pump20_recall is a full-day D1 discovery proxy, not post-send "
            "tradeability. Its denominator uses all KRW daily candles for each "
            "validated source-observed completed date, including zero-pick policy "
            "days. post_send_pump20_precision_pct is reported separately when "
            "receipt-aligned 15m labels exist; post-send all-universe recall is not "
            "claimed without a matching denominator."
        ),
    }


def _policy_contract() -> dict[str, Any]:
    return {
        "models": [asdict(spec) for spec in MODELS],
        "policies": [
            {
                "policy_id": policy.policy_id,
                "description": policy.description,
                "objective": policy.objective,
            }
            for policy in POLICIES
        ],
        "config": _policy_config(),
        "exit_variants": {
            name: asdict(spec) for name, spec in EXIT_VARIANTS.items()
        },
        "snapshot_identities": {
            key: list(value) for key, value in _SNAPSHOT_IDENTITIES.items()
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sqlite": sqlite3.sqlite_version,
        },
    }


def build_policy_input_manifest(
    asof: pd.Timestamp | str,
    *,
    candle_db: Path | None = None,
    snapshot_root: Path = DEFAULT_SNAPSHOT_ROOT,
    pump_v1_decision_root: Path = PUMP_V1_DECISION_ROOT,
    pump_v2_decision_root: Path = PUMP_V2_DECISION_ROOT,
    pump_v2_receipt_root: Path = PUMP_V2_RECEIPT_ROOT,
    radar_verdict_path: Path = RADAR_VERDICT_PATH,
) -> dict[str, Any]:
    """Capture every mutable input and every direct semantic dependency."""
    normalized = _normalise_asof(pd.Timestamp(asof))
    daily_db = Path(candle_db) if candle_db is not None else Path(D1_DB)
    if not daily_db.is_file():
        raise FileNotFoundError(f"daily candle DB missing: {daily_db}")
    bundle = source_bundle_identity(
        files={
            **{
                f"generator:{relative}": _ROOT / relative
                for relative in POLICY_GENERATOR_SOURCES
            },
            "daily_candle_db:main": daily_db,
            "daily_candle_db:wal": Path(f"{daily_db}-wal"),
            "daily_candle_db:journal": Path(f"{daily_db}-journal"),
            "radar_terminal": radar_verdict_path,
            **{
                f"model_ledger:{spec.id}": spec.abs_ledger_path()
                for spec in MODELS
            },
        },
        trees={
            "recommend_snapshots": (snapshot_root, {".json"}),
            "pump_v1_decisions": (
                pump_v1_decision_root,
                {".json"},
            ),
            "pump_v2_decisions": (
                pump_v2_decision_root,
                {".json"},
            ),
            "pump_v2_receipts": (
                pump_v2_receipt_root,
                {".json"},
            ),
        },
        root=_ROOT,
    )
    captured_files = bundle["files"]
    generator_sources = {
        relative: captured_files[f"generator:{relative}"]
        for relative in POLICY_GENERATOR_SOURCES
    }
    daily_candle_files = {
        name: captured_files[f"daily_candle_db:{name}"]
        for name in ("main", "wal", "journal")
    }
    missing_sources = [
        name
        for name, identity in generator_sources.items()
        if not identity["exists"]
    ]
    if missing_sources:
        raise FileNotFoundError(
            f"policy generator source missing: {missing_sources}"
        )
    manifest = {
        "schema": POLICY_INPUT_MANIFEST_SCHEMA,
        "asof": str(normalized.date()),
        "files": {
            "daily_candle_db": daily_candle_files,
            "radar_terminal": captured_files["radar_terminal"],
            "model_ledgers": {
                spec.id: captured_files[f"model_ledger:{spec.id}"]
                for spec in MODELS
            },
        },
        "decision_roots": bundle["trees"],
        "generator_sources": generator_sources,
        "contract": _policy_contract(),
    }
    return with_manifest_digest(manifest)


def _policy_run_identity(payload: dict[str, Any]) -> str:
    body = {
        key: value
        for key, value in payload.items()
        if key not in {"generated_at_utc", "run_id", "payload_sha256"}
    }
    return "policy-" + sha256_bytes(canonical_json_bytes(body))[:32]


def _policy_csv_bytes(payload: dict[str, Any]) -> bytes:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise PolicyArtifactError("policy rows must be a list")
    frame = pd.DataFrame(rows).reindex(columns=POLICY_ROW_COLUMNS)
    for column, value in reversed(
        list(
            zip(
                POLICY_CSV_METADATA_COLUMNS,
                (
                    payload.get("schema"),
                    payload.get("run_id"),
                    payload.get("payload_sha256"),
                ),
            )
        )
    ):
        frame.insert(0, column, value)
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


@contextmanager
def _artifact_lock(path: Path, *, shared: bool) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(
        path.with_name(f".{path.name}.lock"),
        shared=shared,
    ):
        yield


def _atomic_write_json(path: Path, payload: dict) -> None:
    atomic_write_json(path, payload)


def _best_participant(rows: list[dict], metric: str) -> str | None:
    candidates = [
        r for r in rows
        if r.get("n_closed", 0) > 0 and r.get(metric) is not None
    ]
    if not candidates:
        return None

    def numeric(row: dict, key: str) -> float:
        value = row.get(key)
        if value is None:
            return float("-inf")
        parsed = float(value)
        return parsed if math.isfinite(parsed) else float("-inf")

    candidates.sort(
        key=lambda r: (
            numeric(r, metric),
            numeric(r, "net_mean_pct"),
            int(r.get("n_closed") or 0),
        ),
        reverse=True,
    )
    return str(candidates[0].get("participant_id"))


def _validate_policy_payload(payload: dict[str, Any]) -> None:
    if payload.get("schema") != POLICY_ARTIFACT_SCHEMA:
        raise PolicyArtifactError("unsupported policy artifact schema")
    try:
        artifact_asof = date.fromisoformat(str(payload["asof"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise PolicyArtifactError("policy artifact asof is invalid") from exc
    generated_at = payload.get("generated_at_utc")
    try:
        generated = datetime.fromisoformat(str(generated_at))
    except (TypeError, ValueError) as exc:
        raise PolicyArtifactError(
            "policy artifact generated_at_utc is invalid"
        ) from exc
    if generated.tzinfo is None:
        raise PolicyArtifactError(
            "policy artifact generated_at_utc must be timezone-aware"
        )

    manifest = payload.get("input_manifest")
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != POLICY_INPUT_MANIFEST_SCHEMA
        or manifest.get("asof") != artifact_asof.isoformat()
        or not manifest_digest_matches(manifest)
    ):
        raise PolicyArtifactError("policy input manifest is invalid")
    contract = manifest.get("contract")
    if not isinstance(contract, dict) or payload.get("config") != contract.get(
        "config"
    ):
        raise PolicyArtifactError(
            "policy config does not match its input contract"
        )
    config = payload["config"]
    if not isinstance(config, dict):
        raise PolicyArtifactError("policy config must be an object")
    for key in (
        "pump20_threshold",
        "deep_loss_pct",
        "round_trip_cost_pct",
        "round_trip_cost_conservative_pct",
    ):
        value = config.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise PolicyArtifactError(f"policy config is invalid: {key}")

    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise PolicyArtifactError("policy rows must be a list")
    participants: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != set(POLICY_ROW_COLUMNS):
            raise PolicyArtifactError(
                f"policy row schema mismatch at index {index}"
            )
        if row.get("asof") != artifact_asof.isoformat():
            raise PolicyArtifactError(
                f"policy row asof mismatch at index {index}"
            )
        participant = row.get("participant_id")
        if not isinstance(participant, str) or not participant:
            raise PolicyArtifactError(
                f"policy participant identity missing at index {index}"
            )
        if participant in participants:
            raise PolicyArtifactError(
                f"duplicate policy participant: {participant}"
            )
        participants.add(participant)
        for key in (
            "source_id",
            "policy_id",
            "objective",
            "description",
            "recall_date_basis",
            "pump20_label_basis",
        ):
            if not isinstance(row.get(key), str) or not row[key]:
                raise PolicyArtifactError(
                    f"policy row string field invalid at index {index}: {key}"
                )
        integer_fields = (
            "n_closed",
            "n_days",
            "n_selected_days",
            "post_send_label_n",
            "pump20_captured",
            "pump20_actual",
            "pump_days",
            "pump_days_any_captured",
        )
        for key in integer_fields:
            value = row.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise PolicyArtifactError(
                    f"policy row integer field invalid at index {index}: {key}"
                )
        if row["n_selected_days"] > row["n_days"]:
            raise PolicyArtifactError(
                f"policy selected days exceed observed days at index {index}"
            )
        if row["post_send_label_n"] > row["n_closed"]:
            raise PolicyArtifactError(
                f"policy post-send labels exceed closed rows at index {index}"
            )
        if row["pump20_captured"] > row["pump20_actual"]:
            raise PolicyArtifactError(
                f"policy captured pumps exceed actual pumps at index {index}"
            )
        if row["pump_days_any_captured"] > row["pump_days"]:
            raise PolicyArtifactError(
                f"policy captured days exceed pump days at index {index}"
            )
        numeric_fields = (
            "net_sum_pct",
            "net_mean_pct",
            "deep_loss_freq_pct",
            "sl_rate_pct",
            "tp_rate_pct",
            "pump20_precision_pct",
            "post_send_pump20_precision_pct",
            "pump20_recall_pct",
            "pump_day_capture_rate_pct",
        )
        for key in numeric_fields:
            value = row.get(key)
            if value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise PolicyArtifactError(
                    f"policy row numeric field invalid at index {index}: {key}"
                )
        for key in (
            "deep_loss_freq_pct",
            "sl_rate_pct",
            "tp_rate_pct",
            "pump20_precision_pct",
            "post_send_pump20_precision_pct",
            "pump20_recall_pct",
            "pump_day_capture_rate_pct",
        ):
            value = row.get(key)
            if value is not None and not 0.0 <= float(value) <= 100.0:
                raise PolicyArtifactError(
                    f"policy percentage out of range at index {index}: {key}"
                )
    if not isinstance(payload.get("exit_lab"), list):
        raise PolicyArtifactError("policy exit_lab must be a list")

    expected_run_id = _policy_run_identity(payload)
    if payload.get("run_id") != expected_run_id:
        raise PolicyArtifactError("policy run_id mismatch")
    recorded_digest = payload.get("payload_sha256")
    if (
        not isinstance(recorded_digest, str)
        or recorded_digest != payload_digest(payload)
    ):
        raise PolicyArtifactError("policy payload checksum mismatch")


def _finalize_policy_payload(
    body: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema": POLICY_ARTIFACT_SCHEMA,
        **body,
    }
    payload["run_id"] = _policy_run_identity(payload)
    payload["payload_sha256"] = payload_digest(payload)
    _validate_policy_payload(payload)
    return payload


def _write_sqlite(payload: dict, db_path: Path = POLICY_DB) -> None:
    """Persist the latest policy competition snapshot for dashboard/ops audit.

    Same-day reruns replace the asof snapshot. This keeps the DB small and makes
    heartbeat/dashboard consumers read one canonical row set per trading day.
    """
    _validate_policy_payload(payload)
    rows = payload["rows"]
    db_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    con = sqlite3.connect(db_path, timeout=30)
    try:
        con.execute("PRAGMA busy_timeout=30000")
        con.executescript(DB_SCHEMA)
        run_columns = {
            str(row[1])
            for row in con.execute(
                "PRAGMA table_info(policy_competition_runs)"
            ).fetchall()
        }
        run_migrations = {
            "artifact_schema": (
                "ALTER TABLE policy_competition_runs "
                "ADD COLUMN artifact_schema TEXT NOT NULL DEFAULT ''"
            ),
            "run_id": (
                "ALTER TABLE policy_competition_runs "
                "ADD COLUMN run_id TEXT NOT NULL DEFAULT ''"
            ),
            "payload_sha256": (
                "ALTER TABLE policy_competition_runs "
                "ADD COLUMN payload_sha256 TEXT NOT NULL DEFAULT ''"
            ),
            "input_manifest_sha256": (
                "ALTER TABLE policy_competition_runs "
                "ADD COLUMN input_manifest_sha256 TEXT NOT NULL DEFAULT ''"
            ),
        }
        for column, statement in run_migrations.items():
            if column not in run_columns:
                con.execute(statement)

        row_columns = {
            str(row[1])
            for row in con.execute(
                "PRAGMA table_info(policy_competition_rows)"
            ).fetchall()
        }
        migrations = {
            "n_selected_days": (
                "ALTER TABLE policy_competition_rows "
                "ADD COLUMN n_selected_days INTEGER NOT NULL DEFAULT 0"
            ),
            "post_send_pump20_precision_pct": (
                "ALTER TABLE policy_competition_rows "
                "ADD COLUMN post_send_pump20_precision_pct REAL"
            ),
            "post_send_label_n": (
                "ALTER TABLE policy_competition_rows "
                "ADD COLUMN post_send_label_n INTEGER NOT NULL DEFAULT 0"
            ),
            "recall_date_basis": (
                "ALTER TABLE policy_competition_rows "
                "ADD COLUMN recall_date_basis TEXT NOT NULL DEFAULT 'ledger_rows'"
            ),
            "run_id": (
                "ALTER TABLE policy_competition_rows "
                "ADD COLUMN run_id TEXT NOT NULL DEFAULT ''"
            ),
            "payload_sha256": (
                "ALTER TABLE policy_competition_rows "
                "ADD COLUMN payload_sha256 TEXT NOT NULL DEFAULT ''"
            ),
            "pump20_label_basis": (
                "ALTER TABLE policy_competition_rows "
                "ADD COLUMN pump20_label_basis TEXT NOT NULL DEFAULT ''"
            ),
        }
        for column, statement in migrations.items():
            if column not in row_columns:
                con.execute(statement)
        con.commit()
        con.execute("BEGIN IMMEDIATE")
        con.execute(
            """
            INSERT INTO policy_competition_runs (
                asof, artifact_schema, run_id, payload_sha256,
                input_manifest_sha256, generated_at_utc, pump20_threshold,
                deep_loss_pct, round_trip_cost_pct, row_count,
                best_pump_participant, best_net_participant, updated_at_utc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(asof) DO UPDATE SET
                artifact_schema=excluded.artifact_schema,
                run_id=excluded.run_id,
                payload_sha256=excluded.payload_sha256,
                input_manifest_sha256=excluded.input_manifest_sha256,
                generated_at_utc=excluded.generated_at_utc,
                pump20_threshold=excluded.pump20_threshold,
                deep_loss_pct=excluded.deep_loss_pct,
                round_trip_cost_pct=excluded.round_trip_cost_pct,
                row_count=excluded.row_count,
                best_pump_participant=excluded.best_pump_participant,
                best_net_participant=excluded.best_net_participant,
                updated_at_utc=excluded.updated_at_utc
            """,
            (
                payload["asof"],
                payload["schema"],
                payload["run_id"],
                payload["payload_sha256"],
                payload["input_manifest"]["manifest_sha256"],
                payload["generated_at_utc"],
                float(payload["config"]["pump20_threshold"]),
                float(payload["config"]["deep_loss_pct"]),
                float(payload["config"]["round_trip_cost_pct"]),
                len(rows),
                _best_participant(rows, "pump20_recall_pct"),
                _best_participant(rows, "net_mean_pct"),
                now,
            ),
        )
        con.execute("DELETE FROM policy_competition_rows WHERE asof=?", (payload["asof"],))
        con.executemany(
            """
            INSERT INTO policy_competition_rows (
                asof, run_id, payload_sha256, participant_id, source_id,
                policy_id, objective, description, n_closed, n_days,
                n_selected_days, net_sum_pct, net_mean_pct, deep_loss_freq_pct,
                sl_rate_pct, tp_rate_pct, pump20_precision_pct,
                post_send_pump20_precision_pct, post_send_label_n,
                pump20_recall_pct, pump20_captured, pump20_actual, pump_days,
                pump_days_any_captured, pump_day_capture_rate_pct,
                recall_date_basis, pump20_label_basis, generated_at_utc
            )
            VALUES (
                :asof, :run_id, :payload_sha256, :participant_id, :source_id,
                :policy_id, :objective, :description, :n_closed, :n_days,
                :n_selected_days,
                :net_sum_pct, :net_mean_pct, :deep_loss_freq_pct, :sl_rate_pct,
                :tp_rate_pct, :pump20_precision_pct,
                :post_send_pump20_precision_pct, :post_send_label_n,
                :pump20_recall_pct, :pump20_captured, :pump20_actual,
                :pump_days, :pump_days_any_captured,
                :pump_day_capture_rate_pct, :recall_date_basis,
                :pump20_label_basis, :generated_at_utc
            )
            """,
            [
                {
                    "n_selected_days": int(row.get("n_selected_days", 0)),
                    "post_send_pump20_precision_pct": row.get(
                        "post_send_pump20_precision_pct"
                    ),
                    "post_send_label_n": int(
                        row.get("post_send_label_n", 0)
                    ),
                    "recall_date_basis": str(
                        row.get("recall_date_basis", "legacy_unknown")
                    ),
                    **row,
                    "asof": payload["asof"],
                    "run_id": payload["run_id"],
                    "payload_sha256": payload["payload_sha256"],
                    "generated_at_utc": payload["generated_at_utc"],
                }
                for row in rows
            ],
        )
        con.commit()
    finally:
        con.close()


def _validate_policy_sqlite(
    payload: dict[str, Any],
    db_path: Path,
) -> dict[str, Any]:
    if not db_path.is_file():
        raise PolicyArtifactError(f"policy SQLite artifact missing: {db_path}")
    try:
        con = sqlite3.connect(
            f"{db_path.resolve().as_uri()}?mode=ro",
            uri=True,
            timeout=30,
        )
        try:
            con.execute("PRAGMA query_only=ON")
            quick_check = con.execute("PRAGMA quick_check").fetchone()
            if quick_check != ("ok",):
                raise PolicyArtifactError(
                    f"policy SQLite quick_check failed: {quick_check!r}"
                )
            run = con.execute(
                """
                SELECT artifact_schema, run_id, payload_sha256,
                       input_manifest_sha256, generated_at_utc, row_count,
                       best_pump_participant, best_net_participant
                FROM policy_competition_runs
                WHERE asof=?
                """,
                (payload["asof"],),
            ).fetchone()
            expected_run = (
                payload["schema"],
                payload["run_id"],
                payload["payload_sha256"],
                payload["input_manifest"]["manifest_sha256"],
                payload["generated_at_utc"],
                len(payload["rows"]),
                _best_participant(payload["rows"], "pump20_recall_pct"),
                _best_participant(payload["rows"], "net_mean_pct"),
            )
            if run != expected_run:
                raise PolicyArtifactError(
                    "policy JSON/SQLite run identity mismatch"
                )

            cursor = con.execute(
                POLICY_SQLITE_ROW_SELECT,
                (payload["asof"],),
            )
            actual_rows: list[dict[str, Any]] = []
            for values in cursor.fetchall():
                row = dict(zip(POLICY_ROW_COLUMNS, values[:-2]))
                if (
                    values[-2] != payload["run_id"]
                    or values[-1] != payload["payload_sha256"]
                ):
                    raise PolicyArtifactError(
                        "policy SQLite row run identity mismatch"
                    )
                actual_rows.append(row)
        finally:
            con.close()
    except PolicyArtifactError:
        raise
    except (OSError, sqlite3.Error) as exc:
        raise PolicyArtifactError(
            f"invalid policy SQLite artifact: {db_path}"
        ) from exc

    expected_rows = sorted(
        payload["rows"],
        key=lambda row: str(row["participant_id"]),
    )
    if canonical_json_bytes(actual_rows) != canonical_json_bytes(expected_rows):
        raise PolicyArtifactError("policy JSON/SQLite row payload mismatch")
    return {
        "path": str(db_path),
        "row_count": len(actual_rows),
        "run_id": payload["run_id"],
        "payload_sha256": payload["payload_sha256"],
    }


def load_policy_artifact(
    json_path: Path = OUT_JSON,
    *,
    csv_path: Path | None = None,
    db_path: Path = POLICY_DB,
    asof: pd.Timestamp | str | None = None,
    require_exact_asof: bool = False,
    require_current: bool = True,
    candle_db: Path | None = None,
    snapshot_root: Path = DEFAULT_SNAPSHOT_ROOT,
    pump_v1_decision_root: Path = PUMP_V1_DECISION_ROOT,
    pump_v2_decision_root: Path = PUMP_V2_DECISION_ROOT,
    pump_v2_receipt_root: Path = PUMP_V2_RECEIPT_ROOT,
    radar_verdict_path: Path = RADAR_VERDICT_PATH,
) -> dict[str, Any]:
    """Load the JSON/CSV/SQLite triplet only when all identities agree."""
    json_path = Path(json_path)
    csv_path = (
        Path(csv_path)
        if csv_path is not None
        else json_path.with_suffix(".csv")
    )
    db_path = Path(db_path)
    with _artifact_lock(json_path, shared=True):
        try:
            payload = strict_json_object(json_path)
            _validate_policy_payload(payload)
            actual_csv = csv_path.read_bytes()
        except PolicyArtifactError:
            raise
        except ArtifactValidationError as exc:
            raise PolicyArtifactError(str(exc)) from exc
        except OSError as exc:
            raise PolicyArtifactError(
                f"policy CSV artifact missing: {csv_path}"
            ) from exc

        expected_csv = _policy_csv_bytes(payload)
        if actual_csv != expected_csv:
            raise PolicyArtifactError("policy JSON/CSV payload mismatch")

        if asof is not None:
            cutoff = _normalise_asof(pd.Timestamp(asof))
            artifact_asof = _normalise_asof(pd.Timestamp(payload["asof"]))
            if artifact_asof > cutoff or (
                require_exact_asof and artifact_asof != cutoff
            ):
                raise PolicyArtifactError(
                    "policy artifact asof does not satisfy requested cutoff"
                )

        if require_current:
            try:
                current_manifest = build_policy_input_manifest(
                    payload["asof"],
                    candle_db=candle_db,
                    snapshot_root=snapshot_root,
                    pump_v1_decision_root=pump_v1_decision_root,
                    pump_v2_decision_root=pump_v2_decision_root,
                    pump_v2_receipt_root=pump_v2_receipt_root,
                    radar_verdict_path=radar_verdict_path,
                )
            except (OSError, ArtifactSourceChangedError) as exc:
                raise PolicyArtifactError(
                    "policy inputs required for current validation are missing"
                ) from exc
            if payload["input_manifest"] != current_manifest:
                raise PolicyArtifactError(
                    "policy input bytes or generator semantics changed"
                )

        _validate_policy_sqlite(payload, db_path)
    return payload


def _evaluate_rows(rows: pd.DataFrame, *, participant_id: str, source_id: str,
                   policy: PolicySpec, asof: pd.Timestamp,
                   actual_pumps: dict[pd.Timestamp, set[str]],
                   evaluation_dates: list[pd.Timestamp] | None = None,
                   recall_date_basis: str = "selected_closed_rows") -> dict:
    selected = rows.copy()
    selected = selected[pd.to_numeric(selected["net_pct"], errors="coerce").notna()]
    selected["net_pct"] = pd.to_numeric(selected["net_pct"], errors="coerce")

    n_closed = int(len(selected))
    selected_dates = {
        pd.Timestamp(date).normalize()
        for date in selected["date"].dropna().unique()
    }
    eval_dates = sorted({
        pd.Timestamp(date).normalize()
        for date in (
            evaluation_dates if evaluation_dates is not None else selected_dates
        )
    })
    n_selected_days = len(selected_dates)
    n_days = len(eval_dates)

    if n_closed:
        net_sum = float(selected["net_pct"].sum())
        net_mean = float(selected["net_pct"].mean())
        deep_loss = float((selected["net_pct"] <= DEEP_LOSS_PCT).mean())
        pump_precision = float(pd.to_numeric(
            selected["selected_pump20_hit"], errors="coerce").dropna().mean())
        exit_reason = selected.get("exit_reason", pd.Series(index=selected.index, dtype=object))
        tp_rate = float((exit_reason.astype(str).str.upper() == "TP").mean())
        sl_rate = float((exit_reason.astype(str).str.upper() == "SL").mean())
    else:
        net_sum = net_mean = deep_loss = pump_precision = tp_rate = sl_rate = np.nan

    if "post_send_pump20_hit" in selected.columns:
        post_send_values = pd.to_numeric(
            selected["post_send_pump20_hit"], errors="coerce"
        ).dropna()
        if not post_send_values.isin([0, 1]).all():
            raise ValueError(
                f"{participant_id}: invalid post_send_pump20_hit value"
            )
    else:
        post_send_values = pd.Series(dtype=float)
    post_send_precision = (
        float(post_send_values.mean()) if not post_send_values.empty else np.nan
    )

    total_actual = 0
    captured = 0
    days_with_pump = 0
    days_any_captured = 0
    picked_by_date = {
        pd.Timestamp(date).normalize(): set(day_rows["coin"].astype(str))
        for date, day_rows in selected.groupby("date")
    }
    # Recall must include source-observed dates on which a stricter policy chose
    # nothing. Iterating only selected.groupby("date") silently removed those
    # zero-pick days and inflated recall.
    for d in eval_dates:
        pumps = actual_pumps.get(d, set())
        if not pumps:
            continue
        days_with_pump += 1
        total_actual += len(pumps)
        picked = picked_by_date.get(d, set())
        day_captured = len(picked & pumps)
        captured += day_captured
        if day_captured:
            days_any_captured += 1

    pump_recall = captured / total_actual if total_actual else np.nan
    any_capture = days_any_captured / days_with_pump if days_with_pump else np.nan

    return {
        "asof": str(asof.date()),
        "participant_id": participant_id,
        "source_id": source_id,
        "policy_id": policy.policy_id,
        "objective": policy.objective,
        "description": policy.description,
        "n_closed": n_closed,
        "n_days": n_days,
        "n_selected_days": n_selected_days,
        "net_sum_pct": _safe_pct(net_sum),
        "net_mean_pct": _safe_pct(net_mean),
        "deep_loss_freq_pct": _safe_pct(deep_loss * 100.0 if np.isfinite(deep_loss) else np.nan),
        "sl_rate_pct": _safe_pct(sl_rate * 100.0 if np.isfinite(sl_rate) else np.nan),
        "tp_rate_pct": _safe_pct(tp_rate * 100.0 if np.isfinite(tp_rate) else np.nan),
        "pump20_precision_pct": _safe_pct(
            pump_precision * 100.0 if np.isfinite(pump_precision) else np.nan),
        "post_send_pump20_precision_pct": _safe_pct(
            post_send_precision * 100.0
            if np.isfinite(post_send_precision)
            else np.nan
        ),
        "post_send_label_n": int(len(post_send_values)),
        "pump20_recall_pct": _safe_pct(
            pump_recall * 100.0 if np.isfinite(pump_recall) else np.nan),
        "pump20_captured": int(captured),
        "pump20_actual": int(total_actual),
        "pump_days": int(days_with_pump),
        "pump_days_any_captured": int(days_any_captured),
        "pump_day_capture_rate_pct": _safe_pct(
            any_capture * 100.0 if np.isfinite(any_capture) else np.nan),
        "recall_date_basis": recall_date_basis,
        "pump20_label_basis": (
            "full_day_D1_high_over_open_proxy; not post-send tradeability"
        ),
    }


def _consensus_rows(model_rows: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for model_id in ("recommend_r1_open", "recommend_r2_open", "recommend_r1_sustain_open"):
        df = model_rows.get(model_id)
        if df is not None and not df.empty:
            frames.append(df.copy())
    if len(frames) < 2:
        return pd.DataFrame()

    pool = pd.concat(frames, ignore_index=True)
    counts = (
        pool.groupby(["date", "coin"], as_index=False)
        .agg(source_count=("model_id", "nunique"), rank=("rank", "min"))
    )
    consensus = counts[counts["source_count"] >= 2]
    if consensus.empty:
        return consensus

    # Keep one realized path per date/coin. Realized is identical for the same
    # entry/SL/TP path; prefer R1 then R2 then A1 for stable metadata.
    priority = {
        "recommend_r1_open": 0,
        "recommend_r2_open": 1,
        "recommend_r1_sustain_open": 2,
    }
    pool["priority"] = pool["model_id"].map(priority).fillna(99)
    pool = pool.sort_values(["date", "coin", "priority"])
    first = pool.drop_duplicates(["date", "coin"], keep="first")
    out = consensus.merge(first.drop(columns=["rank"], errors="ignore"),
                          on=["date", "coin"], how="left")
    out["model_id"] = "consensus_2of3"
    out["source_name"] = "Consensus of at least 2 among R1/R2/A1"
    return out


def _exit_lab_summary(model_rows: dict[str, pd.DataFrame]) -> list[dict]:
    """모델별 exit 변형 비교 (record-only) — ledger/exit_lab.py 가 close 시 기록한
    병렬 가상 청산 컬럼을 집계한다. 운영 기본 (TP5/SL3) 은 net_pct 그대로.

    보수 비용 (편도 0.2% 슬리피지) net 도 병기 — 표준 가정이 펌프 코인에서
    낙관적일 수 있어서 (ledger/config.py 참조).
    """
    cost_extra_pp = ROUND_TRIP_COST_CONSERVATIVE_PP - ROUND_TRIP_COST_PP
    variants = [("tp5_sl3", "net_pct", None)] + [
        (name, f"exit_{name}_pct", f"exit_{name}_reason" if name != "eod" else None)
        for name in EXIT_VARIANTS
    ]
    out: list[dict] = []
    for model_id, df in model_rows.items():
        if df is None or df.empty:
            continue
        # exit lab 컬럼이 아직 없는 ledger (예: 미배선 채널) 는 기본 변형만 집계
        for name, pct_col, reason_col in variants:
            if pct_col not in df.columns:
                continue
            vals = pd.to_numeric(df[pct_col], errors="coerce").dropna()
            if vals.empty:
                continue
            row = {
                "model_id": model_id,
                "variant": name,
                "n": int(len(vals)),
                "net_mean_pct": round(float(vals.mean()), 4),
                "net_sum_pct": round(float(vals.sum()), 2),
                "net_mean_conservative_pct": round(float(vals.mean()) - cost_extra_pp, 4),
                "deep_loss_freq_pct": round(float((vals <= DEEP_LOSS_PCT).mean() * 100), 2),
            }
            if reason_col and reason_col in df.columns:
                reasons = df.loc[vals.index, reason_col].astype(str)
                row["tp_rate_pct"] = round(float((reasons == "TP").mean() * 100), 2)
            elif name == "tp5_sl3" and "exit_reason" in df.columns:
                reasons = df.loc[vals.index, "exit_reason"].astype(str)
                row["tp_rate_pct"] = round(float((reasons == "TP").mean() * 100), 2)
                row["sl_rate_pct"] = round(float((reasons == "SL").mean() * 100), 2)
            out.append(row)
    return out


def _normalise_asof(asof: pd.Timestamp | None) -> pd.Timestamp:
    value = asof if asof is not None else pd.Timestamp.now(tz="Asia/Seoul")
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is not None:
        parsed = parsed.tz_convert("Asia/Seoul").tz_localize(None)
    return parsed.normalize()


def _run_unlocked(
    asof: pd.Timestamp | None = None,
    *,
    output_csv: Path = OUT_CSV,
    output_json: Path = OUT_JSON,
    db_path: Path | None = POLICY_DB,
    candle_db: Path | None = None,
    snapshot_root: Path = DEFAULT_SNAPSHOT_ROOT,
    pump_v1_decision_root: Path = PUMP_V1_DECISION_ROOT,
    pump_v2_decision_root: Path = PUMP_V2_DECISION_ROOT,
    pump_v2_receipt_root: Path = PUMP_V2_RECEIPT_ROOT,
    radar_verdict_path: Path = RADAR_VERDICT_PATH,
) -> dict:
    asof = _normalise_asof(asof)
    daily_db = Path(candle_db) if candle_db is not None else Path(D1_DB)
    input_manifest = build_policy_input_manifest(
        asof,
        candle_db=daily_db,
        snapshot_root=snapshot_root,
        pump_v1_decision_root=pump_v1_decision_root,
        pump_v2_decision_root=pump_v2_decision_root,
        pump_v2_receipt_root=pump_v2_receipt_root,
        radar_verdict_path=radar_verdict_path,
    )
    radar_terminal = verdict_status_for(
        asof.date(),
        path=radar_verdict_path,
    )
    model_rows = {
        spec.id: (
            _load_model_rows(
                spec,
                asof,
                candle_db=daily_db,
                pump_v1_decision_root=pump_v1_decision_root,
                pump_v2_decision_root=pump_v2_decision_root,
                pump_v2_receipt_root=pump_v2_receipt_root,
            )
            if spec.id in {"pump_hunter", "pump_hunter_v2"}
            else _load_model_rows(spec, asof, candle_db=daily_db)
        )
        for spec in MODELS
    }
    source_observations = {
        spec.id: _source_observed_dates(
            spec,
            asof,
            snapshot_root=snapshot_root,
            pump_v1_decision_root=pump_v1_decision_root,
            pump_v2_decision_root=pump_v2_decision_root,
            pump_v2_receipt_root=pump_v2_receipt_root,
        )
        for spec in MODELS
    }
    all_dates = sorted({
        date
        for dates, _basis in source_observations.values()
        for date in dates
    })
    actual_pumps = _actual_pumps_for_dates(all_dates, daily_db)

    rows = []
    for spec in MODELS:
        df = model_rows.get(spec.id, pd.DataFrame())
        source_dates, recall_basis = source_observations[spec.id]
        if (df is None or df.empty) and not source_dates:
            continue
        evaluation_frame = (
            df if df is not None and not df.empty else _empty_model_rows()
        )
        for policy in POLICIES:
            mask = policy.predicate(evaluation_frame)
            selected = evaluation_frame[mask.fillna(False)].copy()
            rows.append(_evaluate_rows(
                selected,
                participant_id=f"{spec.id}:{policy.policy_id}",
                source_id=spec.id,
                policy=policy,
                asof=asof,
                actual_pumps=actual_pumps,
                evaluation_dates=source_dates,
                recall_date_basis=recall_basis,
            ))

    consensus = _consensus_rows(model_rows)
    consensus_date_counts: dict[pd.Timestamp, int] = {}
    for model_id in (
        "recommend_r1_open",
        "recommend_r2_open",
        "recommend_r1_sustain_open",
    ):
        dates, _basis = source_observations.get(model_id, ([], ""))
        for decision_date in dates:
            normalized = pd.Timestamp(decision_date).normalize()
            consensus_date_counts[normalized] = (
                consensus_date_counts.get(normalized, 0) + 1
            )
    consensus_eval_dates = sorted(
        date for date, count in consensus_date_counts.items() if count >= 2
    )
    if consensus_eval_dates:
        policy = PolicySpec(
            "consensus_2of3",
            "Send only coins selected by at least two of R1/R2/A1.",
            "consensus_precision",
            _true_series,
        )
        rows.append(_evaluate_rows(
            consensus if not consensus.empty else _empty_model_rows(),
            participant_id="consensus_2of3",
            source_id="R1/R2/A1",
            policy=policy,
            asof=asof,
            actual_pumps=actual_pumps,
            evaluation_dates=consensus_eval_dates,
            recall_date_basis="2of3_validated_source_observed_dates",
        ))

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["objective", "pump20_recall_pct", "net_mean_pct", "n_closed"],
            ascending=[True, False, False, False],
            na_position="last",
        )

    rows_payload = (
        json.loads(summary.to_json(orient="records"))
        if not summary.empty else []
    )

    payload = _finalize_policy_payload({
        "asof": str(asof.date()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_manifest": input_manifest,
        "config": _policy_config(),
        "radar_terminal": radar_terminal,
        "rows": rows_payload,
        # exit 변형 비교 (ledger/exit_lab.py) — 같은 경로 병렬 가상 청산.
        "exit_lab": _exit_lab_summary(model_rows),
    })

    current_manifest = build_policy_input_manifest(
        asof,
        candle_db=daily_db,
        snapshot_root=snapshot_root,
        pump_v1_decision_root=pump_v1_decision_root,
        pump_v2_decision_root=pump_v2_decision_root,
        pump_v2_receipt_root=pump_v2_receipt_root,
        radar_verdict_path=radar_verdict_path,
    )
    if current_manifest != input_manifest:
        raise RuntimeError(
            "policy inputs changed while the artifact was being computed"
        )

    atomic_write_bytes(output_csv, _policy_csv_bytes(payload))
    _atomic_write_json(output_json, payload)
    if db_path is not None:
        _write_sqlite(payload, db_path)
    if build_policy_input_manifest(
        asof,
        candle_db=daily_db,
        snapshot_root=snapshot_root,
        pump_v1_decision_root=pump_v1_decision_root,
        pump_v2_decision_root=pump_v2_decision_root,
        pump_v2_receipt_root=pump_v2_receipt_root,
        radar_verdict_path=radar_verdict_path,
    ) != input_manifest:
        raise RuntimeError(
            "policy inputs changed while the artifact was being published"
        )
    return payload


def run(
    asof: pd.Timestamp | None = None,
    *,
    output_csv: Path = OUT_CSV,
    output_json: Path = OUT_JSON,
    db_path: Path | None = POLICY_DB,
    candle_db: Path | None = None,
    snapshot_root: Path = DEFAULT_SNAPSHOT_ROOT,
    pump_v1_decision_root: Path = PUMP_V1_DECISION_ROOT,
    pump_v2_decision_root: Path = PUMP_V2_DECISION_ROOT,
    pump_v2_receipt_root: Path = PUMP_V2_RECEIPT_ROOT,
    radar_verdict_path: Path = RADAR_VERDICT_PATH,
) -> dict:
    with _artifact_lock(output_json, shared=False):
        return _run_unlocked(
            asof,
            output_csv=output_csv,
            output_json=output_json,
            db_path=db_path,
            candle_db=candle_db,
            snapshot_root=snapshot_root,
            pump_v1_decision_root=pump_v1_decision_root,
            pump_v2_decision_root=pump_v2_decision_root,
            pump_v2_receipt_root=pump_v2_receipt_root,
            radar_verdict_path=radar_verdict_path,
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate model + send-policy competition")
    ap.add_argument("--asof", type=str, default=None, help="YYYY-MM-DD (default=today)")
    ap.add_argument("--out-csv", type=Path, default=OUT_CSV)
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    ap.add_argument("--db", type=Path, default=POLICY_DB)
    ap.add_argument("--candle-db", type=Path, default=D1_DB)
    ap.add_argument("--no-db", action="store_true", help="Do not persist SQLite audit DB")
    ap.add_argument(
        "--pump-v1-decision-root",
        type=Path,
        default=PUMP_V1_DECISION_ROOT,
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--pump-v2-decision-root",
        type=Path,
        default=PUMP_V2_DECISION_ROOT,
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--pump-v2-receipt-root",
        type=Path,
        default=PUMP_V2_RECEIPT_ROOT,
        help=argparse.SUPPRESS,
    )
    args = ap.parse_args()

    asof = pd.Timestamp(args.asof) if args.asof else None
    payload = run(
        asof,
        output_csv=args.out_csv,
        output_json=args.out_json,
        db_path=None if args.no_db else args.db,
        candle_db=args.candle_db,
        pump_v1_decision_root=args.pump_v1_decision_root,
        pump_v2_decision_root=args.pump_v2_decision_root,
        pump_v2_receipt_root=args.pump_v2_receipt_root,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

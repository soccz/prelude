"""Read-only, content-pinned D1 inputs for one paired target-window experiment.

This reconstructs present-day full-precision training inputs, not historical
ingestion state or historical model weights.  It neither fits nor writes.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from data.database import connect_readonly
from data.market_universe import signal_eligible_markets
from ops.artifact_provenance import (
    canonical_json_bytes, file_identity, file_set_identity, resolve_identity_path,
    sha256_bytes, strict_json_object,
)
from scripts.recommendation_scorer_v1 import PRECURSOR_FEATURES
from scripts.univariate_precursor_lift_v1 import (
    EPS, add_cross_sectional, build_market_features,
)
from signals.recommend_experiment_data import load_experiment_data
from signals.recommend_training_data import ALLOWED_FEATURES

ROOT = Path(__file__).resolve().parent.parent
RAW_COLUMNS = ("market", "timestamp", "open", "high", "low", "close", "volume", "quote_volume")
RAW_SQL = """SELECT market,timestamp,open,high,low,close,volume,quote_volume
FROM candles WHERE timestamp < '2026-09-07 00:00:00'
ORDER BY market,timestamp"""
DATA_CONFIG = {
    "schema": "recommend_horizon_data.v1",
    "reconstruction_audit_sha256": "04df21246a91ba43933cc2f00d78af1aed3069ac3d726a6818b9cdf0e132c1ec",
    "raw_sql_sha256": "37fbe422c30cc3e807c9054c66772c83ea556a43e912038331dd6a68c0d409f0",
    "build_asof": "2026-09-07T09:01:00+09:00",
    "minimum_prior_rows": 70,
    "universe_rank_max": 100,
    "embargo_calendar_days": 5,
    "target_return_denominator_epsilon": 1e-12,
    "prediction_precision_decimal_places": 8,
}
TRAIN_COLUMNS = (
    "feature_date", "coin", "input_end_at", "B_target_end_at", "C_target_end_at",
    "C_target_present", "B_high_ret", "B_low_ret", "C_high_ret", "C_low_ret",
    *ALLOWED_FEATURES,
)


class HorizonDataError(ValueError):
    """The frozen reconstruction contract cannot be reproduced safely."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HorizonDataError(message)


def _same(actual, expected, message: str) -> None:
    _require(canonical_json_bytes(actual) == canonical_json_bytes(expected), message)


def _aware(value, field: str) -> pd.Timestamp:
    result = pd.Timestamp(value)
    _require(pd.notna(result) and result.tzinfo is not None, f"{field} must be timezone-aware")
    return result.tz_convert("UTC")


def logical_rows_identity(rows) -> dict:
    """Same JSON-row hash domain as the independently frozen audit."""
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes({"columns": list(RAW_COLUMNS),
                                       "encoding": "JSON array per row plus LF"}) + b"\n")
    count = 0
    for row in rows:
        digest.update(canonical_json_bytes(list(row)) + b"\n")
        count += 1
    return {"sha256": digest.hexdigest(), "rows": count}


def _raw_panel(raw: pd.DataFrame, *, build_asof: str) -> pd.DataFrame:
    """Pure original feature transformations, plus explicitly keyed targets."""
    _require(set(RAW_COLUMNS).issubset(raw.columns) and len(raw) > 0, "missing raw OHLC input")
    raw = raw.loc[:, RAW_COLUMNS].copy(deep=True)
    _require(raw.market.map(lambda x: isinstance(x, str) and x.startswith("KRW-")).all(),
             "invalid market")
    raw["timestamp"] = pd.to_datetime(raw.timestamp)
    _require(raw.timestamp.dt.tz is None and raw.timestamp.notna().all(), "raw timestamp must be naive KST")
    _require(raw.timestamp.dt.strftime("%H:%M:%S").eq("09:00:00").all(), "raw boundary must be 09:00 KST")
    _require(not raw.duplicated(["market", "timestamp"]).any(), "duplicate raw candle key")
    for name in RAW_COLUMNS[2:]:
        values = raw[name]
        _require(not values.map(lambda x: isinstance(x, (bool, np.bool_))).any(), f"boolean raw {name}")
        numeric = pd.to_numeric(values, errors="raise")
        valid = np.isfinite(numeric) & (numeric > 0 if name in {"open", "high", "low", "close"} else numeric >= 0)
        if name == "quote_volume":
            valid |= numeric.isna()  # Original feature helper uses volume * close.
        _require(valid.all(), f"invalid raw {name}")
        raw[name] = numeric
    _require((raw.high >= raw[["open", "low", "close"]].max(axis=1)).all()
             and (raw.low <= raw[["open", "high", "close"]].min(axis=1)).all(), "inconsistent OHLC")
    raw = raw.sort_values(["market", "timestamp"], kind="stable").reset_index(drop=True)
    ends = (raw.timestamp + pd.Timedelta(days=1)).dt.tz_localize("Asia/Seoul")
    _require((ends <= _aware(build_asof, "build_asof")).all(), "raw contains an incomplete/future D1 candle")
    raw["input_end_at"] = raw.groupby("market").timestamp.shift(1) + pd.Timedelta(days=1)
    raw["B_high_ret"] = raw.high / (raw.open + EPS) - 1.0
    raw["B_low_ret"] = raw.low / (raw.open + EPS) - 1.0
    eligible = set(signal_eligible_markets(sorted(raw.market.unique())))
    frames = []
    for market, group in raw.groupby("market", sort=True):
        if market in eligible and len(group) > 70:
            features = build_market_features(group.reset_index(drop=True), asof_kst=build_asof)
            features["date"] = features.timestamp.dt.date
            frames.append(features.iloc[70:].copy())
    _require(bool(frames), "no markets with more than 70 raw rows")
    panel = add_cross_sectional(pd.concat(frames, ignore_index=True))
    panel = panel.merge(raw[["market", "timestamp", "input_end_at", "B_high_ret", "B_low_ret"]],
                        on=["market", "timestamp"], validate="one_to_one")
    following = raw[["market", "timestamp", "B_high_ret", "B_low_ret"]].rename(columns={
        "timestamp": "C_timestamp", "B_high_ret": "C_high_ret", "B_low_ret": "C_low_ret"})
    following["timestamp"] = following.C_timestamp - pd.Timedelta(days=1)
    panel = panel.merge(following, on=["market", "timestamp"], how="left", validate="one_to_one")
    panel["C_target_present"] = panel.C_timestamp.notna()
    _require((panel.input_end_at <= panel.timestamp).all(), "shifted input ends after target start")
    _require(not np.isinf(panel[list(ALLOWED_FEATURES)].to_numpy(dtype=float)).any(),
             "nonfinite reconstructed feature")
    return panel.sort_values(["date", "market"], kind="stable").reset_index(drop=True)


def _prediction_parity(panel: pd.DataFrame, frozen: pd.DataFrame) -> tuple[dict, list]:
    required = {*ALLOWED_FEATURES, "date", "slot", "coin", "rank", "snapshot_id", "decision_started_at"}
    _require(required.issubset(frozen.columns) and len(frozen) > 0, "missing frozen prediction inputs")
    _require(frozen.slot.eq("preopen").all(), "only frozen preopen is in scope")
    _require(not frozen.duplicated(["date", "coin"]).any(), "duplicate frozen candidate")
    metadata, parity, seen_ids = {}, [], set()
    for day, group in frozen.groupby("date", sort=True):
        stamp = pd.Timestamp(day)
        _require(isinstance(day, str) and stamp.strftime("%Y-%m-%d") == day, "invalid prediction date")
        _require(group.snapshot_id.nunique() == 1 and group.decision_started_at.nunique() == 1,
                 "inconsistent snapshot metadata")
        sid = group.snapshot_id.iloc[0]
        _require(isinstance(sid, str) and sid not in seen_ids, "duplicate/invalid snapshot ID")
        seen_ids.add(sid)
        _require(group["rank"].map(lambda x: isinstance(x, (int, np.integer))
                                  and not isinstance(x, (bool, np.bool_))).all()
                 and sorted(group["rank"].tolist()) == list(range(1, len(group) + 1)), "invalid candidate ranks")
        started = _aware(group.decision_started_at.iloc[0], "decision_started_at")
        kst = started.tz_convert("Asia/Seoul")
        _require(kst.strftime("%Y-%m-%d") == day and kst < stamp.tz_localize("Asia/Seoul") + pd.Timedelta(hours=9),
                 "preopen decision is outside its date/before-open window")
        reference = (stamp - pd.Timedelta(days=1)).date()
        rebuilt = panel.loc[panel.date.eq(reference) & panel.f_qv_rank.le(100)].set_index("market")
        _require(set(rebuilt.index) == set(group.coin), f"universe parity failed: {day}")
        for row in group.to_dict("records"):
            actual = rebuilt.loc[row["coin"]]
            _require(_aware(actual.input_end_at.tz_localize("Asia/Seoul"), "input_end") < started,
                     "prediction input not available before decision")
            for name in ALLOWED_FEATURES:
                a, b = row[name], actual[name]
                _require(not isinstance(a, (bool, np.bool_)), "boolean frozen feature")
                a_missing, b_missing = pd.isna(a), pd.isna(b)
                same = a_missing and b_missing
                if not a_missing and not b_missing:
                    same = np.isfinite(float(a)) and float(a) == round(float(b), 8)
                _require(bool(same), f"stored-feature parity failed: {day}/{row['coin']}/{name}")
        metadata[sid] = {"date": day, "decision_started_at": started.isoformat(),
                         "feature_row_date": str(reference),
                         "cutoff_exclusive": str(reference - pd.Timedelta(days=5))}
        parity.append({"date": day, "snapshot_id": sid, "candidate_rows": len(group),
                       "exact_round8_rows": len(group), "universe_equal": True})
    return metadata, parity


def common_training_mask(frame: pd.DataFrame, decision: dict) -> pd.Series:
    """Calendar maturity only; historical DB ingestion availability is unknown."""
    started = _aware(decision["decision_started_at"], "decision_started_at")
    return (frame.feature_date.lt(decision["cutoff_exclusive"]) & frame.C_target_present
            & pd.to_datetime(frame.input_end_at, utc=True).lt(started)
            & pd.to_datetime(frame.B_target_end_at, utc=True).lt(started)
            & pd.to_datetime(frame.C_target_end_at, utc=True).lt(started))


def build_horizon_frames(raw: pd.DataFrame, frozen_preopen: pd.DataFrame, *, build_asof: str) -> dict:
    """Pure reconstruction; arguments remain unchanged, no outcomes are features."""
    _require(list(PRECURSOR_FEATURES) == list(ALLOWED_FEATURES) and len(ALLOWED_FEATURES) == 24,
             "actual R1 feature allowlist changed")
    _require(EPS == DATA_CONFIG["target_return_denominator_epsilon"], "original epsilon changed")
    panel = _raw_panel(raw, build_asof=build_asof)
    frozen = frozen_preopen.copy(deep=True).sort_values(["date", "rank"], kind="stable").reset_index(drop=True)
    metadata, parity = _prediction_parity(panel, frozen)
    maximum_cutoff = max(row["cutoff_exclusive"] for row in metadata.values())
    training = panel.loc[panel.f_qv_rank.le(100) & panel.date.astype(str).lt(maximum_cutoff)].copy()
    training["feature_date"] = training.date.astype(str)
    training["coin"] = training.market
    for name, values in {
        "input_end_at": training.input_end_at,
        "B_target_end_at": training.timestamp + pd.Timedelta(days=1),
        "C_target_end_at": training.C_timestamp + pd.Timedelta(days=1),
    }.items():
        utc = values.dt.tz_localize("Asia/Seoul").dt.tz_convert("UTC")
        training[name] = pd.Series([value.isoformat() if pd.notna(value) else None for value in utc],
                                   index=training.index, dtype=object)
    training = training.loc[:, TRAIN_COLUMNS].reset_index(drop=True)
    common_audits = []
    for sid, decision in metadata.items():
        base = training.loc[training.feature_date.lt(decision["cutoff_exclusive"])]
        common = training.loc[common_training_mask(training, decision)]
        missing = base.loc[~base.C_target_present, ["feature_date", "coin"]]
        common_audits.append({"snapshot_id": sid, "date": decision["date"],
                              "base_top100_feature_rows": len(base), "common_training_rows": len(common),
                              "missing_exact_next_calendar_targets": missing.to_dict("records"),
                              "other_unavailable_rows": len(base) - len(common) - len(missing)})
    return {"training_frame": training, "prediction_frame": frozen,
            "decision_metadata": metadata,
            "data_audit": {"model_fitted": False, "historical_ingestion_proven": False,
                           "historical_fullprecision_training_proven": False,
                           "panel_rows": len(panel), "training_rows": len(training),
                           "feature_columns": list(ALLOWED_FEATURES), "parity": parity,
                           "common_training_by_decision": common_audits}}


def _bound_manifest(files: dict, root: Path) -> dict:
    paths = {name: resolve_identity_path(item["path"], root=root) for name, item in files.items()}
    _same(file_set_identity(paths, root=root), files, "frozen source/evidence content changed")
    return paths


def prepare_horizon_data(readiness_path: Path, reconstruction_audit_path: Path) -> dict:
    """Pin the audited SQL domain and every source before and after computing."""
    audit_path = Path(reconstruction_audit_path)
    identity = file_identity(audit_path, root=ROOT)
    _require(identity.get("sha256") == DATA_CONFIG["reconstruction_audit_sha256"],
             "reconstruction audit file checksum mismatch")
    audit = strict_json_object(audit_path)
    _same(file_identity(audit_path, root=ROOT), identity, "audit changed while reading")
    _same(audit["sql"]["raw_ohlc_query"], RAW_SQL, "audited SQL changed")
    _same(audit["database"]["logical_rows"]["sha256"], DATA_CONFIG["raw_sql_sha256"], "audit raw hash changed")
    paths = _bound_manifest(audit["sources"]["files"], ROOT)
    evidence_root = Path(audit["frozen_evidence"]["identity_root"])
    evidence_paths = _bound_manifest(audit["frozen_evidence"]["files"], evidence_root)
    expected_readiness = evidence_paths["readiness_report"]
    _require(Path(readiness_path).resolve() == expected_readiness.resolve(), "unexpected readiness path")
    frozen = load_experiment_data(expected_readiness)
    _same(frozen["provenance"]["files"], audit["frozen_evidence"]["files"], "rebound frozen evidence differs")
    all_paths = {**{f"source:{k}": v for k, v in paths.items()},
                 **{f"evidence:{k}": v for k, v in evidence_paths.items()},
                 "reconstruction_audit": audit_path, "horizon_builder": Path(__file__)}
    before = file_set_identity(all_paths, root=ROOT)
    with connect_readonly(ROOT / "data/upbit_d1.db") as connection:
        connection.execute("PRAGMA temp_store = MEMORY")
        connection.execute("BEGIN")
        rows = connection.execute(RAW_SQL).fetchall()
        logical = logical_rows_identity(rows)
        _same(logical, audit["database"]["logical_rows"], "raw D1 logical row hash changed")
        _same(logical_rows_identity(connection.execute(RAW_SQL)), logical, "SQL transaction identity changed")
    raw = pd.DataFrame.from_records(rows, columns=RAW_COLUMNS)
    prediction = frozen["frame"].loc[lambda frame: frame.slot.eq("preopen")].copy(deep=True)
    result = build_horizon_frames(raw, prediction, build_asof=DATA_CONFIG["build_asof"])
    parity = result["data_audit"]["parity"]
    _require(len(prediction) == 3500 and len(parity) == 35
             and all(row["candidate_rows"] == 100 for row in parity), "frozen 35-day population changed")
    _same([(row["date"], row["snapshot_id"]) for row in parity],
          [(row["date"], row["snapshot_id"]) for row in audit["parity"]["by_date"]], "frozen dates changed")
    checks = result["data_audit"]["common_training_by_decision"]
    for actual, expected in zip([checks[0], checks[-1]], audit["first_last_common_training_checks"]):
        _same([actual["date"], actual["base_top100_feature_rows"], actual["common_training_rows"]],
              [expected["asof"], expected["base_top100_feature_rows"], expected["common_training_rows"]],
              "common first/last training population changed")
    _same(file_set_identity(all_paths, root=ROOT), before, "inputs changed during horizon reconstruction")
    _same(file_identity(audit_path, root=ROOT), identity, "audit changed during reconstruction")
    _bound_manifest(audit["sources"]["files"], ROOT)
    _bound_manifest(audit["frozen_evidence"]["files"], evidence_root)
    result["data_audit"].update({"raw_logical_identity": logical, "config": dict(DATA_CONFIG),
                                  "d1_canonical_definition_agreement": audit["d1_vs_existing_preopen_canonical_labels"]})
    result["provenance"] = {"root": str(ROOT), "files": before,
                            "files_sha256": sha256_bytes(canonical_json_bytes(before)),
                            "source_evidence_before_after_identical": True,
                            "raw_logical_identity": logical,
                            "raw_sql": RAW_SQL, "database_mode": "ro/query_only/BEGIN/temp_store=MEMORY"}
    return result

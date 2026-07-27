"""Read-only independent recalculation for semivol_joint_challenger_v1.

The script rebuilds the four semivolatility columns from the D1 database,
checks persisted row contracts, and recalculates policy metrics and paired
date-cluster intervals with an auditor-owned seed.  It writes nothing.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parent.parent
PREFIX = ROOT / "output/semivol_joint_challenger_v1"
D1_DB = ROOT / "data/upbit_d1.db"
FIRST_PASSAGE_REFERENCE = (
    ROOT / "output/first_passage_head_challenger_v1_predictions.csv.gz"
)
FIXED = "label_fp_safe10"
SEED = 20260725
DRAWS = 20_000
SEMIVOL_FEATURES = (
    "downside_semivol_7",
    "downside_semivol_21",
    "upside_semivol_21",
    "semivol_asym_21",
)


def _current_completed_cutoff() -> pd.Timestamp:
    now = datetime.now(ZoneInfo("Asia/Seoul"))
    current_session = (now - timedelta(hours=9)).date()
    return pd.Timestamp(current_session - timedelta(days=1))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        if series.isna().any():
            raise ValueError(f"{series.name}: null boolean values")
        return series.astype(bool)
    normalized = series.astype("string").str.strip().str.lower()
    valid = normalized.isin({"true", "1", "yes", "false", "0", "no"})
    if not valid.all():
        bad = sorted(normalized[~valid].dropna().unique().tolist())
        raise ValueError(f"{series.name}: invalid boolean values: {bad[:5]}")
    return normalized.isin({"true", "1", "yes"})


def _ci(values: pd.Series | np.ndarray, offset: int) -> dict:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("bootstrap input contains missing or non-finite values")
    array = numeric.to_numpy(float)
    if not len(array):
        raise ValueError("bootstrap input is empty")
    rng = np.random.default_rng(SEED + offset)
    indices = rng.integers(
        0,
        len(array),
        size=(DRAWS, len(array)),
    )
    bootstrap = array[indices].mean(axis=1)
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "ci95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
    }


def _require_columns(
    frame: pd.DataFrame,
    columns: set[str],
    *,
    name: str,
) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{name}: missing columns: {missing}")


def _require_unique(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    name: str,
) -> None:
    duplicate = frame.duplicated(columns, keep=False)
    if duplicate.any():
        sample = frame.loc[duplicate, columns].head(3).to_dict("records")
        raise ValueError(f"{name}: duplicate keys {columns}: {sample}")


def _require_finite(
    frame: pd.DataFrame,
    columns: set[str],
    *,
    name: str,
) -> None:
    for column in sorted(columns):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"{name}: {column} contains missing/non-finite values")


def _file_snapshot(path: Path) -> dict:
    stat = path.stat()
    wal = Path(f"{path}-wal")
    wal_stat = wal.stat() if wal.exists() else None
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "wal_bytes": wal_stat.st_size if wal_stat else 0,
        "wal_mtime_ns": wal_stat.st_mtime_ns if wal_stat else None,
    }


def _connect_readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _daily(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["_sl"] = work["path_bracket_outcome"].eq("sl").astype(float)
    return work.groupby("date", sort=True).agg(
        safe=(FIXED, "mean"),
        up10=("path_up10", "mean"),
        dn5=("path_dn5", "mean"),
        sl=("_sl", "mean"),
        net=("path_bracket_net", "mean"),
        n=("market", "size"),
    )


def _policy_table(picks: pd.DataFrame) -> dict:
    result = {}
    for (scope, policy), group in picks.groupby(
        ["scope", "policy"],
        sort=True,
    ):
        daily = _daily(group)
        result[f"{scope}:{policy}"] = {
            "rows": int(len(group)),
            "dates": int(group["date"].nunique()),
            **{
                metric: float(daily[metric].mean())
                for metric in ("safe", "up10", "dn5", "sl", "net")
            },
            "absolute_net_date_ci": _ci(
                daily["net"],
                offset=len(result) + 1,
            ),
        }
    return result


def _paired(
    picks: pd.DataFrame,
    *,
    scope: str,
    primary: str,
    comparators: tuple[str, ...],
    offset: int,
) -> dict:
    scoped = picks[picks["scope"].eq(scope)]
    first = _daily(scoped[scoped["policy"].eq(primary)])
    output = {}
    for comparator_index, comparator in enumerate(comparators, start=1):
        second = _daily(scoped[scoped["policy"].eq(comparator)])
        common = first.join(
            second,
            how="inner",
            lsuffix="_primary",
            rsuffix="_baseline",
        )
        output[comparator] = {
            metric: _ci(
                common[f"{metric}_primary"]
                - common[f"{metric}_baseline"],
                offset=offset + comparator_index * 20 + metric_index,
            )
            for metric_index, metric in enumerate(
                ("safe", "up10", "dn5", "sl", "net")
            )
        }
    return output


def _auc(label: pd.Series, score: pd.Series) -> float | None:
    valid = label.notna() & score.notna()
    if valid.sum() < 2 or label[valid].nunique() < 2:
        return None
    return float(roc_auc_score(label[valid], score[valid]))


def _rebuild_semivol(predictions: pd.DataFrame) -> dict:
    with _connect_readonly(D1_DB) as connection:
        candles = pd.read_sql_query(
            """
            SELECT market, timestamp, close
            FROM candles
            ORDER BY market, timestamp
            """,
            connection,
        )
    candles["timestamp"] = pd.to_datetime(candles["timestamp"])
    if candles.empty or candles["timestamp"].isna().any():
        raise ValueError("D1 database: empty or invalid timestamps")
    _require_unique(candles, ["market", "timestamp"], name="D1 candles")
    candles["date"] = candles["timestamp"].dt.normalize()
    candles["close"] = pd.to_numeric(candles["close"], errors="coerce")
    _require_finite(candles, {"close"}, name="D1 candles")
    if (candles["close"] <= 0).any():
        raise ValueError("D1 database: non-positive closes")
    candles["return"] = candles.groupby("market", sort=False)[
        "close"
    ].pct_change(fill_method=None)
    candles["_down_sq"] = candles["return"].clip(upper=0.0).pow(2)
    candles["_up_sq"] = candles["return"].clip(lower=0.0).pow(2)

    def rolling_shift(
        series: pd.Series,
        window: int,
    ) -> pd.Series:
        return np.sqrt(
            series.rolling(window, min_periods=window).mean()
        ).shift(1)

    candles["recalc_downside_semivol_7"] = candles.groupby(
        "market",
        sort=False,
    )["_down_sq"].transform(lambda values: rolling_shift(values, 7))
    candles["recalc_downside_semivol_21"] = candles.groupby(
        "market",
        sort=False,
    )["_down_sq"].transform(lambda values: rolling_shift(values, 21))
    candles["recalc_upside_semivol_21"] = candles.groupby(
        "market",
        sort=False,
    )["_up_sq"].transform(lambda values: rolling_shift(values, 21))
    candles["recalc_semivol_asym_21"] = (
        candles["recalc_upside_semivol_21"]
        - candles["recalc_downside_semivol_21"]
    )
    candles["recalc_source_date"] = candles.groupby(
        "market",
        sort=False,
    )["date"].shift(1)

    unique = predictions[
        ["market", "date", "semivol_source_date", *SEMIVOL_FEATURES]
    ].copy()
    rebuilt = candles[
        [
            "market",
            "date",
            "recalc_source_date",
            "recalc_downside_semivol_7",
            "recalc_downside_semivol_21",
            "recalc_upside_semivol_21",
            "recalc_semivol_asym_21",
        ]
    ]
    merged = unique.merge(
        rebuilt,
        on=["market", "date"],
        how="left",
        validate="one_to_one",
    )
    rebuilt_columns = [
        "recalc_source_date",
        *[f"recalc_{feature}" for feature in SEMIVOL_FEATURES],
    ]
    if merged[rebuilt_columns].isna().any().any():
        raise ValueError("semivol rebuild: missing source rows or values")
    errors = {}
    for feature in SEMIVOL_FEATURES:
        errors[feature] = float(
            np.nanmax(
                np.abs(
                    pd.to_numeric(merged[feature]).to_numpy(float)
                    - pd.to_numeric(
                        merged[f"recalc_{feature}"]
                    ).to_numpy(float)
                )
            )
        )
    source = pd.to_datetime(merged["semivol_source_date"])
    rebuilt_source = pd.to_datetime(merged["recalc_source_date"])
    if source.isna().any() or rebuilt_source.isna().any():
        raise ValueError("semivol rebuild: invalid source dates")
    source_lag = (merged["date"] - source).dt.days
    if not source_lag.eq(1).all():
        raise ValueError("semivol rebuild: feature source is not exactly D-1")
    if (source != rebuilt_source).any():
        raise ValueError("semivol rebuild: persisted source date mismatch")
    if any(value > 1e-12 for value in errors.values()):
        raise ValueError(f"semivol rebuild: feature mismatch: {errors}")
    digest_columns = [
        "market", "date", "recalc_source_date",
        *[f"recalc_{feature}" for feature in SEMIVOL_FEATURES],
    ]
    digest_payload = (
        merged[digest_columns]
        .sort_values(["date", "market"])
        .to_csv(index=False, date_format="%Y-%m-%d", float_format="%.17g")
        .encode()
    )
    return {
        "rows": int(len(merged)),
        "missing_rebuilt_rows": int(
            merged["recalc_source_date"].isna().sum()
        ),
        "max_abs_error": errors,
        "source_date_mismatch": int(
            (source != rebuilt_source).sum()
        ),
        "source_lag_days_min": int(
            (merged["date"] - source).dt.days.min()
        ),
        "source_lag_days_max": int(
            (merged["date"] - source).dt.days.max()
        ),
        "rebuilt_values_sha256": hashlib.sha256(digest_payload).hexdigest(),
    }


def _core_reference(predictions: pd.DataFrame) -> dict:
    reference = pd.read_csv(
        FIRST_PASSAGE_REFERENCE,
        usecols=[
            "scope",
            "date",
            "market",
            "raw_fp_fixed_head",
            "p_fp_fixed_head",
        ],
        parse_dates=["date"],
        float_precision="round_trip",
    )
    _require_unique(
        reference,
        ["scope", "date", "market"],
        name="first-passage reference",
    )
    _require_finite(
        reference,
        {"raw_fp_fixed_head", "p_fp_fixed_head"},
        name="first-passage reference",
    )
    merged = predictions[
        [
            "scope",
            "date",
            "market",
            "raw_core_fp_head",
            "p_core_fp_head",
        ]
    ].merge(
        reference,
        on=["scope", "date", "market"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(predictions) or len(reference) != len(predictions):
        raise ValueError("core reference: cohort/key mismatch")
    _require_finite(
        merged,
        {
            "raw_core_fp_head", "p_core_fp_head",
            "raw_fp_fixed_head", "p_fp_fixed_head",
        },
        name="core reference comparison",
    )
    raw_error = float(
        (merged["raw_core_fp_head"] - merged["raw_fp_fixed_head"]).abs().max()
    )
    probability_error = float(
        (merged["p_core_fp_head"] - merged["p_fp_fixed_head"]).abs().max()
    )
    if raw_error > 1e-12 or probability_error > 1e-12:
        raise ValueError("core reference: baseline scores do not reproduce")
    return {
        "rows": int(len(merged)),
        "candidate_rows": int(len(predictions)),
        "raw_max_abs_delta": raw_error,
        "probability_max_abs_delta": probability_error,
        "reference_sha256": _sha(FIRST_PASSAGE_REFERENCE),
    }


def _pick_overlap(picks: pd.DataFrame) -> dict:
    scoped = picks[picks["scope"].eq("locked_holdout")]
    semivol = scoped[
        scoped["policy"].eq("semivol_joint_head")
    ].groupby("date")["market"].agg(set)
    core = scoped[
        scoped["policy"].eq("core_fp_head")
    ].groupby("date")["market"].agg(set)
    common = semivol.index.intersection(core.index)
    if common.empty:
        raise ValueError("pick overlap: no common semivol/core dates")
    overlap = np.array(
        [
            len(semivol.loc[date].intersection(core.loc[date]))
            for date in common
        ],
        dtype=float,
    )
    return {
        "dates": int(len(common)),
        "mean_shared_of_3": float(overlap.mean()),
        "identical_top3_date_rate": float((overlap == 3).mean()),
        "zero_overlap_date_rate": float((overlap == 0).mean()),
    }


def _time_order_audit(
    coverage: dict,
    predictions: pd.DataFrame,
) -> dict:
    entries = [
        *coverage["model_metadata"]["discovery"],
        *coverage["model_metadata"]["locked_holdout"],
    ]
    keys = []
    windows: dict[tuple[str, str], list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for entry in entries:
        key = (
            str(entry["scope"]),
            int(entry["outer_fold"]),
            str(entry["policy"]),
        )
        keys.append(key)
        train_end = pd.Timestamp(entry["train_end"])
        test_start = pd.Timestamp(entry["test_start"])
        test_end = pd.Timestamp(entry["test_end"])
        if train_end >= test_start or test_start > test_end:
            raise ValueError(f"coverage: train/test overlap or inverted range: {key}")
        if (test_start - train_end).days - 1 < 5:
            raise ValueError(f"coverage: outer embargo shorter than five dates: {key}")
        inner = entry.get("inner_folds", [])
        if len(inner) != 3:
            raise ValueError(f"coverage: expected three inner folds: {key}")
        previous_validation_end: pd.Timestamp | None = None
        for fold in inner:
            inner_train_end = pd.Timestamp(fold["train_end"])
            validation_start = pd.Timestamp(fold["validation_start"])
            validation_end = pd.Timestamp(fold["validation_end"])
            if inner_train_end >= validation_start or validation_start > validation_end:
                raise ValueError(f"coverage: inner train/validation overlap: {key}")
            if (validation_start - inner_train_end).days - 1 < 5:
                raise ValueError(f"coverage: inner embargo shorter than five dates: {key}")
            if (
                previous_validation_end is not None
                and validation_start <= previous_validation_end
            ):
                raise ValueError(f"coverage: overlapping inner folds: {key}")
            previous_validation_end = validation_end
        if entry["scope"] == "locked_holdout":
            embargoed = entry.get("embargoed_discovery_dates", [])
            if (
                int(entry.get("holdout_embargo_dates", -1)) != 5
                or len(embargoed) != 5
            ):
                raise ValueError(f"coverage: invalid holdout embargo: {key}")
            expected_embargo = list(
                pd.date_range(
                    train_end + pd.Timedelta(days=1),
                    periods=5,
                    freq="D",
                )
            )
            if [pd.Timestamp(value) for value in embargoed] != expected_embargo:
                raise ValueError(f"coverage: non-contiguous holdout embargo: {key}")
        scoped_dates = predictions.loc[
            predictions["scope"].eq(key[0])
            & predictions["fold"].eq(key[1]),
            "date",
        ]
        if scoped_dates.empty:
            raise ValueError(f"predictions: no rows for metadata key: {key}")
        if scoped_dates.min() != test_start or scoped_dates.max() != test_end:
            raise ValueError(f"predictions: metadata test window mismatch: {key}")
        windows.setdefault((key[0], key[2]), []).append((test_start, test_end))
    if len(keys) != len(set(keys)):
        raise ValueError("coverage: duplicate scope/fold/policy metadata")
    for (scope, policy), policy_windows in windows.items():
        ordered = sorted(set(policy_windows))
        if scope == "discovery_oof" and len(ordered) != 5:
            raise ValueError(f"coverage: {policy} does not have five outer folds")
        if any(current[0] <= previous[1] for previous, current in zip(ordered, ordered[1:])):
            raise ValueError(f"coverage: overlapping outer folds for {scope}:{policy}")
    return {
        "metadata_entries": int(len(entries)),
        "unique_scope_fold_policy_entries": int(len(set(keys))),
        "outer_time_order_violations": 0,
        "inner_time_order_violations": 0,
        "inner_fold_count_violations": 0,
        "holdout_embargo_count_violations": 0,
    }


def main() -> None:
    paths = {
        "predictions": Path(f"{PREFIX}_predictions.csv.gz"),
        "picks": Path(f"{PREFIX}_picks.csv.gz"),
        "summary": Path(f"{PREFIX}_summary.csv"),
        "paired": Path(f"{PREFIX}_paired.csv"),
        "auc": Path(f"{PREFIX}_auc.csv"),
        "folds": Path(f"{PREFIX}_folds.csv"),
        "coverage": Path(f"{PREFIX}_coverage.json"),
        "manifest": Path(f"{PREFIX}_manifest.json"),
    }
    missing_paths = [str(path) for path in paths.values() if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"missing audit inputs: {missing_paths}")
    if not D1_DB.is_file() or not FIRST_PASSAGE_REFERENCE.is_file():
        raise FileNotFoundError("missing D1 database or first-passage reference")
    d1_snapshot_before = _file_snapshot(D1_DB)
    predictions = pd.read_csv(
        paths["predictions"],
        parse_dates=["date", "semivol_source_date"],
        float_precision="round_trip",
    )
    picks = pd.read_csv(
        paths["picks"],
        parse_dates=["date", "semivol_source_date"],
        float_precision="round_trip",
    )
    required_common = {
        "market", "date", "scope", "fold", "path_complete",
        "history_prior_bars", FIXED, "fp_fixed_outcome",
        "path_bracket_outcome", "path_bracket_net", "path_eod_net",
        "path_up10", "path_dn5", "semivol_source_date",
        *SEMIVOL_FEATURES,
    }
    _require_columns(
        predictions,
        required_common
        | {
            "raw_core_fp_head", "p_core_fp_head",
            "raw_semivol_joint_head", "p_semivol_joint_head",
        },
        name="predictions",
    )
    _require_columns(
        picks,
        required_common | {"policy"},
        name="picks",
    )
    if predictions.empty or picks.empty:
        raise ValueError("prediction/pick artifact is empty")
    if predictions["date"].isna().any() or picks["date"].isna().any():
        raise ValueError("prediction/pick artifact contains invalid dates")
    _require_unique(
        predictions,
        ["scope", "date", "market"],
        name="predictions",
    )
    _require_unique(
        picks,
        ["scope", "date", "policy", "market"],
        name="picks",
    )
    for frame in (predictions, picks):
        frame["path_complete"] = _bool(frame["path_complete"])
    coverage = json.loads(paths["coverage"].read_text())
    manifest = json.loads(paths["manifest"].read_text())
    if manifest.get("schema") != "semivol_joint_challenger_v1_manifest":
        raise ValueError("manifest: unexpected schema")
    expected_scopes = {"discovery_oof", "locked_holdout"}
    if set(predictions["scope"]) != expected_scopes:
        raise ValueError("predictions: unexpected/missing scopes")
    expected_policies = {
        "R1_repaired", "core_fp_head", "liquidity_matched",
        "lowest_ATR", "semivol_joint_head",
    }
    for scope in expected_scopes:
        scoped = picks[picks["scope"].eq(scope)]
        if set(scoped["policy"]) != expected_policies:
            raise ValueError(f"picks: unexpected/missing policies in {scope}")
        counts = scoped.groupby(["date", "policy"]).size()
        if counts.empty or not counts.eq(3).all():
            raise ValueError(f"picks: {scope} is not an exact common Top3 cohort")
        if not scoped.groupby("date")["policy"].nunique().eq(5).all():
            raise ValueError(f"picks: {scope} has incomplete policy dates")
    if not picks["path_complete"].all():
        raise ValueError("picks: persisted cohort contains incomplete paths")
    _require_finite(
        predictions,
        {
            "history_prior_bars", *SEMIVOL_FEATURES,
            "raw_core_fp_head", "p_core_fp_head",
            "raw_semivol_joint_head", "p_semivol_joint_head",
        },
        name="predictions",
    )
    _require_finite(
        picks,
        {FIXED, "path_bracket_net", "path_eod_net", "path_up10", "path_dn5"},
        name="picks",
    )
    completed_cutoff = pd.Timestamp(
        coverage["dates"]["locked_holdout"]["end"]
    )
    if completed_cutoff > _current_completed_cutoff():
        raise ValueError("coverage: locked cutoff is in the future")
    if predictions["date"].max() != completed_cutoff:
        raise ValueError("predictions: max date disagrees with locked cutoff")
    if picks["date"].max() > completed_cutoff:
        raise ValueError("picks: contains rows after locked cutoff")
    expected_prediction_dates = int(coverage["baseline_alignment"]["after"]["n"])
    if predictions["date"].nunique() != expected_prediction_dates:
        raise ValueError("predictions: date count disagrees with coverage")
    if len(predictions) != expected_prediction_dates * 100:
        raise ValueError("predictions: full Top100 population is incomplete")

    prediction_counts = predictions.groupby(["scope", "date"]).size()
    pick_counts = picks.groupby(["scope", "date", "policy"]).size()
    policy_counts = picks.groupby(["scope", "date"])["policy"].nunique()
    complete = predictions[predictions["path_complete"]].copy()
    expected_net = np.where(
        complete["path_bracket_outcome"].eq("tp"),
        0.05 - 0.0015,
        np.where(
            complete["path_bracket_outcome"].eq("sl"),
            -0.03 - 0.0015,
            complete["path_eod_net"],
        ),
    )
    holdout_dates = sorted(
        predictions.loc[
            predictions["scope"].eq("locked_holdout"),
            "date",
        ].unique()
    )
    holdout_hash = hashlib.sha256(
        "\n".join(
            pd.Timestamp(value).strftime("%Y-%m-%d")
            for value in holdout_dates
        ).encode()
    ).hexdigest()

    manifest_hash_match = {
        name: (
            _sha(paths[name])
            == manifest["files"][name]["sha256"]
        )
        for name in (
            "predictions",
            "picks",
            "summary",
            "paired",
            "auc",
            "folds",
            "coverage",
        )
    }
    if not all(manifest_hash_match.values()):
        raise ValueError(f"manifest: artifact hash mismatch: {manifest_hash_match}")
    for name, expected in manifest["files"].items():
        artifact = paths.get(name)
        if artifact is None:
            continue
        if int(expected["bytes"]) != artifact.stat().st_size:
            raise ValueError(f"manifest: byte count mismatch for {name}")
        expected_path = ROOT / str(expected["path"])
        if expected_path.resolve() != artifact.resolve():
            raise ValueError(f"manifest: path mismatch for {name}")
    auc = {
        f"{scope}:{policy}": _auc(
            group[FIXED],
            group[f"raw_{policy}"],
        )
        for scope, scoped in predictions.groupby("scope", sort=True)
        for policy in ("core_fp_head", "semivol_joint_head")
        for group in [
            scoped[
                scoped["path_complete"]
                & scoped[FIXED].notna()
            ]
        ]
    }
    if any(value is None or not np.isfinite(value) for value in auc.values()):
        raise ValueError("AUC audit produced missing/non-finite values")
    cost_error = float(
        np.max(
            np.abs(
                complete["path_bracket_net"].to_numpy(float)
                - expected_net
            )
        )
    )
    if cost_error > 1e-12:
        raise ValueError(f"predictions: round-trip cost mismatch: {cost_error}")
    fixed_mismatch = int(
        (
            complete[FIXED].astype(int)
            != complete["fp_fixed_outcome"].eq("up_first").astype(int)
        ).sum()
    )
    if fixed_mismatch:
        raise ValueError("predictions: fixed first-passage label mismatch")
    if holdout_hash != coverage["dates"]["locked_holdout"]["dates_sha256"]:
        raise ValueError("coverage: locked holdout hash mismatch")
    feature_rebuild = _rebuild_semivol(predictions)
    d1_snapshot_after = _file_snapshot(D1_DB)
    if d1_snapshot_after != d1_snapshot_before:
        raise RuntimeError("D1 database changed during recalculation")
    core_reference = _core_reference(predictions)
    time_order = _time_order_audit(coverage, predictions)
    result = {
        "hashes": {name: _sha(path) for name, path in paths.items()},
        "input_provenance": {
            "d1_database_snapshot": d1_snapshot_before,
            "first_passage_reference_sha256": _sha(FIRST_PASSAGE_REFERENCE),
            "audit_source_sha256": _sha(Path(__file__)),
            "bootstrap_seed": SEED,
            "bootstrap_draws": DRAWS,
            "population_scope": (
                "all aligned discovery/locked prediction rows (exact PIT "
                "Top100 per date) plus exact common complete Top3 picks"
            ),
            "output_contract": (
                "read-only; no artifact writes; one final JSON document is "
                "emitted only after all checks pass"
            ),
        },
        "contracts": {
            "prediction_rows": int(len(predictions)),
            "prediction_scope_date_min": int(prediction_counts.min()),
            "prediction_scope_date_max": int(prediction_counts.max()),
            "pick_policy_date_exact3": bool((pick_counts == 3).all()),
            "pick_policy_count_min": int(policy_counts.min()),
            "pick_policy_count_max": int(policy_counts.max()),
            "persisted_picks_all_complete": bool(
                picks["path_complete"].all()
            ),
            "history_prior_min": int(
                predictions["history_prior_bars"].min()
            ),
            "history_prior_below70": int(
                (predictions["history_prior_bars"] < 70).sum()
            ),
            "fixed_label_mismatch": fixed_mismatch,
            "cost_once_max_abs_error": cost_error,
            "holdout_dates_n": int(len(holdout_dates)),
            "holdout_hash": holdout_hash,
            "coverage_holdout_hash": coverage["dates"][
                "locked_holdout"
            ]["dates_sha256"],
            "holdout_hash_matches": bool(
                holdout_hash
                == coverage["dates"]["locked_holdout"][
                    "dates_sha256"
                ]
            ),
            "manifest_hash_matches": manifest_hash_match,
            "only_model_score_variants": sorted(
                column.removeprefix("raw_")
                for column in predictions.columns
                if column.startswith("raw_")
            ),
            "completed_label_cutoff": completed_cutoff.strftime("%Y-%m-%d"),
        },
        "feature_rebuild": feature_rebuild,
        "core_reference": core_reference,
        "time_order": time_order,
        "auc": auc,
        "table": _policy_table(picks),
        "paired": {
            scope: _paired(
                picks,
                scope=scope,
                primary="semivol_joint_head",
                comparators=("core_fp_head", "R1_repaired"),
                offset=100 if scope == "discovery_oof" else 500,
            )
            for scope in ("discovery_oof", "locked_holdout")
        },
        "locked_pick_overlap_semivol_vs_core": _pick_overlap(picks),
        "evidence_boundary": {
            "reported_trial_count": coverage["fixed_trial"][
                "trial_count"
            ],
            "new_feature_bundle": coverage["fixed_trial"][
                "new_features"
            ],
            "holdout_contaminated": coverage["dates"][
                "locked_holdout"
            ]["contaminated_diagnostic_only"],
            "maximum_possible_verdict": coverage[
                "maximum_possible_verdict"
            ],
            "reported_verdict": coverage["verdict"],
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

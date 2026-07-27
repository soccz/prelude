"""R1 p_dn5 downside-veto challengers: locked holdout + paired-date audit.

This is a research-only harness.  It does not import or mutate the production
sender, ledger, registry, Telegram, or decision policy.

Question
--------
Can the useful R1 downside head (``p_lab_dn_05``) remove risky candidates
without destroying the already-weak upside ranking?

The current grid is deliberately small and fixed before evaluation:

* ``R1_baseline``: current p_up10 / p_dn5 ordering.
* ``veto_within_day_top_third``: remove the riskiest third inside each day's
  same 100-name candidate set, then preserve R1 order.
* ``veto_train_cal50``: scalar-calibrate p_dn5 on data strictly before the
  locked holdout, veto calibrated risk > 50%, then preserve R1 order.
* ``lexicographic_risk_first``: p_dn5 ascending, then R1 ordering.

The final 180 OOS dates are a locked historical holdout.  The absolute
calibration factor is fit once on pre-holdout OOS rows and frozen.  No
challenger result is used to choose a cutoff.  The two rank-only rules have no
fitted parameter.

Evaluation is reported twice and never pooled:

* all OOS daily-candle conservative approximation (SL wins same-day ambiguity);
* common-date, complete 15m paths using KRW-BTC completeness and target-only
  no-trade flat filling, with exact SL-first same-bar handling.  The execution
  window is the first tradable bar after the 09:10 decision:
  ``[D 09:15, D+1 09:15)``.

All policies use the exact same candidate set, Top 3, TP +5%, SL -3%, and
round-trip cost 0.15%.  Date-cluster paired bootstrap compares each challenger
with R1.

Outputs
-------
``output/downside_veto_challenger_v1_summary.csv``
``output/downside_veto_challenger_v1_paired_ci.csv``
``output/downside_veto_challenger_v1_folds.csv``
``output/downside_veto_challenger_v1_picks.csv``
``output/downside_veto_challenger_v1_coverage.json``
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.market_universe import is_excluded_signal_market  # noqa: E402
from ledger.path_quality import BAR_FREQ, EXPECTED_BARS  # noqa: E402
from ops.code_lineage import python_code_lineage  # noqa: E402
from scripts.recommender_downside_exit_v1 import simulate_path  # noqa: E402


INPUT_OOS = _ROOT / "output" / "cc_filtered_multiday_oos_v1.parquet"
M15_DB = _ROOT / "data" / "upbit_15m.db"
OUTPUT_PREFIX = _ROOT / "output" / "downside_veto_challenger_v1"
UPSTREAM_PRODUCER = _ROOT / "scripts" / "cc_filtered_multiday_v1.py"

TOP_K = 3
UNIVERSE_TOP_N = 100
RR_EPS = 1e-3
TP = 0.05
SL = 0.03
ROUND_TRIP_COST = 0.0015
ABSOLUTE_RISK_CUTOFF = 0.50
WITHIN_DAY_VETO_FRACTION = 1.0 / 3.0
LOCKED_HOLDOUT_DAYS = 180
BOOTSTRAP_DRAWS = 5_000
RANDOM_SEED = 42
BENCHMARK_MARKET = "KRW-BTC"
EXECUTION_START_HOUR = 9
EXECUTION_START_MINUTE = 15
RETURN_TOLERANCE = 1e-8

BASELINE = "R1_baseline"
CHALLENGERS = (
    "veto_within_day_top_third",
    "veto_train_cal50",
    "lexicographic_risk_first",
)
POLICIES = (BASELINE,) + CHALLENGERS

REQUIRED_COLUMNS = {
    "date",
    "market",
    "fold",
    "p_lab_up_10",
    "p_lab_dn_05",
    "p_lab_dn_10",
    "exp_downside",
    "up_high_ret",
    "down_low_ret",
    "eod_ret",
    "f_qv_rank",
}

NUMERIC_COLUMNS = REQUIRED_COLUMNS - {"date", "market"}
PROBABILITY_COLUMNS = {
    "p_lab_up_10",
    "p_lab_dn_05",
    "p_lab_dn_10",
}


@dataclass(frozen=True)
class BulkPath:
    bars: tuple[tuple[float, float, float, float], ...]
    complete: bool
    quality: str
    raw_bars: int
    flat_filled_bars: int
    benchmark_bars: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _code_lineage() -> dict:
    return python_code_lineage(entrypoint=Path(__file__), root=_ROOT)


def _completed_label_cutoff() -> object:
    """Latest fully closed KST 09:00-to-09:00 trading date."""
    now_kst = pd.Timestamp.now(tz="Asia/Seoul")
    current_session = (now_kst - pd.Timedelta(hours=9)).date()
    return (
        pd.Timestamp(current_session) - pd.Timedelta(days=1)
    ).date()


def _connect_readonly(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(path)
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


def _file_signature(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    wal_path = Path(f"{path}-wal")
    return {
        "path": str(path.resolve()),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _sha256(path),
        "wal": (
            {
                "bytes": int(wal_path.stat().st_size),
                "mtime_ns": int(wal_path.stat().st_mtime_ns),
                "sha256": _sha256(wal_path),
            }
            if wal_path.is_file()
            else None
        ),
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _artifact_paths(prefix: Path) -> dict[str, Path]:
    return {
        "summary": Path(f"{prefix}_summary.csv"),
        "paired_ci": Path(f"{prefix}_paired_ci.csv"),
        "folds": Path(f"{prefix}_folds.csv"),
        "picks": Path(f"{prefix}_picks.csv"),
        "coverage": Path(f"{prefix}_coverage.json"),
    }


def _verify_manifest(
    *,
    manifest_path: Path,
    expected: dict[str, Path],
) -> dict:
    if not manifest_path.is_file():
        raise RuntimeError(f"artifact manifest is missing: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("downside artifact manifest is unreadable") from exc
    if payload.get("schema") != "downside_veto_challenger_v1_manifest":
        raise RuntimeError("downside artifact manifest schema mismatch")
    files = payload.get("files")
    if not isinstance(files, dict) or set(files) != set(expected):
        raise RuntimeError("downside artifact manifest file set mismatch")
    for name, expected_path in expected.items():
        entry = files.get(name)
        if not isinstance(entry, dict):
            raise RuntimeError(f"downside manifest entry is invalid: {name}")
        recorded_path = Path(str(entry.get("path", "")))
        if not recorded_path.is_absolute():
            recorded_path = _ROOT / recorded_path
        if recorded_path.resolve() != expected_path.resolve():
            raise RuntimeError(f"downside manifest path mismatch: {name}")
        if not expected_path.is_file():
            raise RuntimeError(f"downside artifact is missing: {expected_path}")
        if int(entry.get("bytes", -1)) != expected_path.stat().st_size:
            raise RuntimeError(f"downside artifact size mismatch: {name}")
        if entry.get("sha256") != _sha256(expected_path):
            raise RuntimeError(f"downside artifact checksum mismatch: {name}")
    return payload


def validate_existing_artifacts(
    *,
    output_prefix: Path,
    input_oos: Path,
    m15_db: Path,
) -> dict:
    """Reject stale, tampered, or mixed-generation downside results."""
    expected = _artifact_paths(output_prefix)
    manifest_path = Path(f"{output_prefix}_manifest.json")
    _verify_manifest(manifest_path=manifest_path, expected=expected)
    try:
        coverage = json.loads(
            expected["coverage"].read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("downside coverage is unreadable") from exc
    if coverage.get("schema") != "downside_veto_challenger_v1":
        raise RuntimeError("downside coverage schema mismatch")
    if coverage.get("script_sha256") != _sha256(Path(__file__)):
        raise RuntimeError("downside artifacts were built by stale code")
    if coverage.get("code_lineage") != _code_lineage():
        raise RuntimeError("downside local code dependencies have changed")
    current_signatures = {
        "input_oos": _file_signature(input_oos),
        "m15_db": _file_signature(m15_db),
    }
    if coverage.get("source_signatures") != current_signatures:
        raise RuntimeError("downside source inputs have changed")
    return {
        "status": "valid",
        "schema": coverage["schema"],
        "output_prefix": str(output_prefix),
        "promotion_eligible": bool(
            coverage.get("wf_hygiene", {})
            .get("upstream_train_embargo_provenance", {})
            .get("promotion_eligible", False)
        ),
        "manifest_sha256": _sha256(manifest_path),
    }


def _publish_staged_files(
    staged_to_target: dict[Path, Path],
) -> None:
    """Publish one complete generation and restore every original on failure."""
    if not staged_to_target:
        raise ValueError("no staged artifacts to publish")
    parents = {target.parent.resolve() for target in staged_to_target.values()}
    if len(parents) != 1:
        raise ValueError("all generation artifacts must share one directory")
    output_dir = next(iter(parents))
    output_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = Path(
        tempfile.mkdtemp(prefix=".challenger-backup.", dir=output_dir)
    )
    backups: dict[Path, Path | None] = {}
    published: list[Path] = []
    remove_backup_dir = True
    try:
        for index, target in enumerate(staged_to_target.values()):
            if target.is_file():
                backup = backup_dir / f"{index:03d}.backup"
                shutil.copy2(target, backup)
                backups[target] = backup
            else:
                backups[target] = None
        for staged, target in staged_to_target.items():
            if not staged.is_file():
                raise RuntimeError(f"staged artifact is missing: {staged}")
            os.replace(staged, target)
            published.append(target)
        directory_fd = os.open(output_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        rollback_errors = []
        for target in reversed(published):
            backup_path = backups[target]
            try:
                if backup_path is None:
                    target.unlink(missing_ok=True)
                else:
                    os.replace(backup_path, target)
            except OSError as exc:
                rollback_errors.append(f"{target}: {exc}")
        try:
            directory_fd = os.open(output_dir, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError as exc:
            rollback_errors.append(f"{output_dir} fsync: {exc}")
        if rollback_errors:
            remove_backup_dir = False
            raise RuntimeError(
                "generation publish failed and rollback was incomplete: "
                + "; ".join(rollback_errors)
                + f"; recovery backups preserved at {backup_dir}"
            )
        raise
    finally:
        if remove_backup_dir:
            shutil.rmtree(backup_dir, ignore_errors=True)


def _read_oos(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        try:
            frame = pd.read_parquet(path)
        except ImportError as exc:
            raise RuntimeError(
                "reading the default OOS Parquet requires pyarrow or "
                "fastparquet in the project environment"
            ) from exc
    else:
        frame = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"OOS input missing columns: {missing}")
    if frame.empty:
        raise ValueError("OOS input is empty")

    frame = frame.copy()
    parsed_dates = pd.to_datetime(frame["date"], errors="raise")
    if parsed_dates.isna().any():
        raise ValueError("OOS input contains missing dates")
    frame["date"] = parsed_dates.dt.date
    if frame["date"].max() > _completed_label_cutoff():
        raise ValueError("OOS input contains a not-yet-completed label date")
    markets = frame["market"].astype("string")
    valid_markets = markets.str.fullmatch(r"KRW-[A-Z0-9]+", na=False)
    if not valid_markets.all():
        raise ValueError("OOS input contains invalid/non-canonical markets")
    frame["market"] = markets.astype(str)
    excluded = sorted(
        {
            market
            for market in frame["market"].unique()
            if is_excluded_signal_market(market)
        }
    )
    if excluded:
        raise RuntimeError(
            "OOS input contains excluded signal markets "
            f"{excluded}; regenerate the upstream C3 OOS artifact"
        )
    for column in sorted(NUMERIC_COLUMNS):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"OOS input contains nonfinite {column}")
        frame[column] = values
    for column in PROBABILITY_COLUMNS:
        if not frame[column].between(0.0, 1.0, inclusive="both").all():
            raise ValueError(f"OOS probability is outside [0,1]: {column}")
    fold = frame["fold"].to_numpy(dtype=float)
    if (fold < 0).any() or not np.equal(fold, np.floor(fold)).all():
        raise ValueError("OOS fold must be a nonnegative integer")
    frame["fold"] = fold.astype(int)
    rank = frame["f_qv_rank"].to_numpy(dtype=float)
    if (
        not np.equal(rank, np.floor(rank)).all()
        or (rank < 1).any()
        or (rank > UNIVERSE_TOP_N).any()
    ):
        raise ValueError("f_qv_rank must be an integer in [1,100]")
    frame["f_qv_rank"] = rank.astype(int)
    invalid_ohlc = (
        (frame["up_high_ret"] < -RETURN_TOLERANCE)
        | (frame["down_low_ret"] > RETURN_TOLERANCE)
        | (frame["down_low_ret"] <= -1)
        | (frame["eod_ret"] <= -1)
        | (
            frame["up_high_ret"] + RETURN_TOLERANCE
            < frame["eod_ret"]
        )
        | (
            frame["down_low_ret"] - RETURN_TOLERANCE
            > frame["eod_ret"]
        )
    )
    if invalid_ohlc.any():
        raise ValueError("OOS daily return columns violate OHLC ordering")
    frame = frame[frame["f_qv_rank"] <= 100].copy()
    frame = frame.sort_values(["date", "market"]).reset_index(drop=True)
    if frame.duplicated(["date", "market"]).any():
        raise ValueError("duplicate date/market rows in OOS input")
    counts = frame.groupby("date").size()
    if not (counts == 100).all():
        bad = counts[counts != 100].head().to_dict()
        raise ValueError(f"candidate set is not static Top 100 on every date: {bad}")
    ranks = frame.groupby("date")["f_qv_rank"].agg(
        lambda values: tuple(sorted(values))
    )
    expected_ranks = tuple(range(1, UNIVERSE_TOP_N + 1))
    if not ranks.map(lambda values: values == expected_ranks).all():
        raise ValueError("f_qv_rank is not an exact 1..100 permutation per date")
    folds_per_date = frame.groupby("date")["fold"].nunique()
    if not (folds_per_date == 1).all():
        raise ValueError("a date appears in more than one OOS fold")

    fold_ranges = (
        frame.groupby("fold")["date"]
        .agg(["min", "max"])
        .sort_index()
    )
    previous_end = None
    for _, row in fold_ranges.iterrows():
        if previous_end is not None and row["min"] <= previous_end:
            raise ValueError("walk-forward folds overlap or are not chronological")
        previous_end = row["max"]

    frame["R1"] = (
        frame["p_lab_up_10"]
        / np.maximum(frame["p_lab_dn_05"], RR_EPS)
    )
    return frame


def _r1_order(group: pd.DataFrame) -> pd.DataFrame:
    ordered = group.sort_values(
        ["R1", "p_lab_dn_10", "p_lab_up_10", "exp_downside"],
        ascending=[False, True, False, False],
    ).copy()
    ordered["r1_rank"] = np.arange(1, len(ordered) + 1)
    return ordered


def _preholdout_calibration(
    frame: pd.DataFrame,
    holdout_start: object,
) -> dict:
    train = frame[frame["date"] < holdout_start].copy()
    predicted = float(train["p_lab_dn_05"].mean())
    actual = float((train["down_low_ret"] <= -0.05).mean())
    if not np.isfinite(predicted) or predicted <= 0:
        raise ValueError("invalid pre-holdout p_dn5 mean")
    factor = actual / predicted
    # Guard only against corrupt input.  This bound is not tuned on outcomes.
    factor = float(np.clip(factor, 0.25, 4.0))
    return {
        "n_rows": int(len(train)),
        "n_dates": int(train["date"].nunique()),
        "end_date": str(train["date"].max()),
        "predicted_dn5_mean": predicted,
        "actual_dn5_rate": actual,
        "scalar_factor": factor,
        "absolute_cutoff": ABSOLUTE_RISK_CUTOFF,
    }


def _take_survivors(
    ordered: pd.DataFrame,
    veto_mask: pd.Series,
) -> tuple[pd.DataFrame, int]:
    passed = ordered[~veto_mask].sort_values("r1_rank")
    rejected = ordered[veto_mask].sort_values("r1_rank")
    chosen = passed.head(TOP_K)
    fallback_n = max(0, TOP_K - len(chosen))
    if fallback_n:
        chosen = pd.concat([chosen, rejected.head(fallback_n)])
    return chosen.sort_values("r1_rank").head(TOP_K).copy(), fallback_n


def _select_all(
    frame: pd.DataFrame,
    calibration: dict,
) -> pd.DataFrame:
    selected = []
    factor = float(calibration["scalar_factor"])

    for date, raw_group in frame.groupby("date", sort=True):
        group = _r1_order(raw_group)
        r1_set = set(group.head(TOP_K)["market"])

        baseline = group.head(TOP_K).copy()
        baseline["_veto_count"] = 0
        baseline["_fallback_count"] = 0
        baseline["_cal_p_dn5"] = np.clip(
            baseline["p_lab_dn_05"] * factor, 0.0, 1.0
        )
        policy_frames = [(BASELINE, baseline)]

        # Exact top third.  Discrete p_dn5 ties are resolved with downside-only
        # information and finally market name; no outcome column participates.
        n_veto = int(np.floor(len(group) * WITHIN_DAY_VETO_FRACTION))
        risk_order = group.sort_values(
            ["p_lab_dn_05", "p_lab_dn_10", "exp_downside", "market"],
            ascending=[False, False, True, True],
        )
        veto_indices = set(risk_order.head(n_veto).index)
        veto_mask = group.index.to_series().isin(veto_indices)
        pseudo, fallback_n = _take_survivors(group, veto_mask)
        pseudo["_veto_count"] = n_veto
        pseudo["_fallback_count"] = fallback_n
        pseudo["_cal_p_dn5"] = np.clip(
            pseudo["p_lab_dn_05"] * factor, 0.0, 1.0
        )
        policy_frames.append(("veto_within_day_top_third", pseudo))

        # One semantic cutoff, calibrated only on pre-holdout rows and frozen.
        calibrated_risk = np.clip(group["p_lab_dn_05"] * factor, 0.0, 1.0)
        absolute_mask = calibrated_risk > ABSOLUTE_RISK_CUTOFF
        absolute, fallback_n = _take_survivors(group, absolute_mask)
        absolute["_veto_count"] = int(absolute_mask.sum())
        absolute["_fallback_count"] = fallback_n
        absolute["_cal_p_dn5"] = np.clip(
            absolute["p_lab_dn_05"] * factor, 0.0, 1.0
        )
        policy_frames.append(("veto_train_cal50", absolute))

        # Deliberately extreme comparator: downside first, upside only inside
        # the lowest-risk bucket/ties.  It has no learned cutoff.
        lexicographic = group.sort_values(
            [
                "p_lab_dn_05",
                "p_lab_dn_10",
                "R1",
                "p_lab_up_10",
                "exp_downside",
            ],
            ascending=[True, True, False, False, False],
        ).head(TOP_K).copy()
        lexicographic["_veto_count"] = 0
        lexicographic["_fallback_count"] = 0
        lexicographic["_cal_p_dn5"] = np.clip(
            lexicographic["p_lab_dn_05"] * factor, 0.0, 1.0
        )
        policy_frames.append(("lexicographic_risk_first", lexicographic))

        for policy, picks in policy_frames:
            picks = picks.copy()
            picks["policy"] = policy
            picks["selection_rank"] = np.arange(1, len(picks) + 1)
            picks["candidate_count"] = len(group)
            picks["is_original_r1_top3"] = picks["market"].isin(r1_set)
            selected.append(picks)

    out = pd.concat(selected, ignore_index=True)
    counts = out.groupby(["date", "policy"]).size()
    if not (counts == TOP_K).all():
        raise AssertionError("every date/policy must select exactly Top 3")
    if out.duplicated(["date", "policy", "market"]).any():
        raise AssertionError("duplicate market inside policy/date")
    return out


def _window(date: object) -> tuple[pd.Timestamp, pd.DatetimeIndex]:
    start = pd.Timestamp(date).normalize() + pd.Timedelta(
        hours=EXECUTION_START_HOUR,
        minutes=EXECUTION_START_MINUTE,
    )
    expected = pd.date_range(start, periods=EXPECTED_BARS, freq=BAR_FREQ)
    return start, expected


def _valid_bar(values: tuple[float, float, float, float]) -> bool:
    o, high, low, close = values
    return bool(
        all(np.isfinite(v) and v > 0 for v in values)
        and high >= max(o, close)
        and low <= min(o, close)
        and high >= low
    )


def _bulk_paths(
    pairs: pd.DataFrame,
    db_path: Path,
) -> dict[tuple[str, object], BulkPath]:
    """Bulk equivalent of ledger.path_quality.assess_15m_path.

    A persistent connection and one benchmark check per date make this
    research backfill materially faster while preserving the same rules:
    exact KRW-BTC grid + closed horizon, target-only gaps flat-filled from the
    previous close, and no partial path use.
    """
    pairs = pairs[["market", "date"]].drop_duplicates().sort_values(
        ["date", "market"]
    )
    results: dict[tuple[str, object], BulkPath] = {}

    with _connect_readonly(db_path) as conn:
        horizon = conn.execute(
            "SELECT MIN(timestamp), MAX(timestamp) FROM candles WHERE market=?",
            (BENCHMARK_MARKET,),
        ).fetchone()
        benchmark_start = pd.Timestamp(horizon[0]) if horizon and horizon[0] else None
        benchmark_end = pd.Timestamp(horizon[1]) if horizon and horizon[1] else None

        benchmark_cache: dict[object, tuple[bool, str, int]] = {}
        for date in sorted(pairs["date"].unique()):
            start, expected = _window(date)
            end = start + BAR_FREQ * EXPECTED_BARS
            rows = conn.execute(
                """
                SELECT timestamp FROM candles
                WHERE market=? AND timestamp>=? AND timestamp<?
                ORDER BY timestamp
                """,
                (
                    BENCHMARK_MARKET,
                    start.strftime("%Y-%m-%d %H:%M:%S"),
                    end.strftime("%Y-%m-%d %H:%M:%S"),
                ),
            ).fetchall()
            timestamps = [pd.Timestamp(row[0]) for row in rows]
            expected_set = set(expected)
            if benchmark_start is None or benchmark_end is None:
                state = (False, "benchmark_missing", len(rows))
            elif benchmark_start > expected[0]:
                state = (False, "db_horizon_start_incomplete", len(rows))
            elif benchmark_end < end:
                state = (False, "db_horizon_end_incomplete", len(rows))
            elif any(ts not in expected_set for ts in timestamps):
                state = (False, "benchmark_off_grid", len(rows))
            elif set(timestamps) != expected_set:
                state = (False, "benchmark_gap", len(rows))
            else:
                state = (True, "complete", len(rows))
            benchmark_cache[date] = state

        for row in pairs.itertuples(index=False):
            market, date = str(row.market), row.date
            benchmark_ok, benchmark_quality, benchmark_bars = benchmark_cache[date]
            if not benchmark_ok:
                results[(market, date)] = BulkPath(
                    (), False, benchmark_quality, 0, 0, benchmark_bars
                )
                continue

            start, expected = _window(date)
            end = start + BAR_FREQ * EXPECTED_BARS
            rows = conn.execute(
                """
                SELECT timestamp, open, high, low, close
                FROM candles
                WHERE market=? AND timestamp>=? AND timestamp<?
                ORDER BY timestamp
                """,
                (
                    market,
                    start.strftime("%Y-%m-%d %H:%M:%S"),
                    end.strftime("%Y-%m-%d %H:%M:%S"),
                ),
            ).fetchall()
            expected_set = set(expected)
            by_timestamp: dict[
                pd.Timestamp, tuple[float, float, float, float]
            ] = {}
            invalid_quality = ""
            for timestamp, o, high, low, close in rows:
                ts = pd.Timestamp(timestamp)
                if ts not in expected_set or ts in by_timestamp:
                    invalid_quality = "target_off_grid"
                    break
                bar: tuple[float, float, float, float] = (
                    float(o),
                    float(high),
                    float(low),
                    float(close),
                )
                if not _valid_bar(bar):
                    invalid_quality = "invalid_target_ohlc"
                    break
                by_timestamp[ts] = bar
            if invalid_quality:
                results[(market, date)] = BulkPath(
                    (), False, invalid_quality, len(rows), 0, benchmark_bars
                )
                continue
            if not rows:
                results[(market, date)] = BulkPath(
                    (),
                    False,
                    "target_no_observations",
                    0,
                    0,
                    benchmark_bars,
                )
                continue

            previous_close = None
            if expected[0] not in by_timestamp:
                prior = conn.execute(
                    """
                    SELECT close FROM candles
                    WHERE market=? AND timestamp<?
                    ORDER BY timestamp DESC LIMIT 1
                    """,
                    (market, start.strftime("%Y-%m-%d %H:%M:%S")),
                ).fetchone()
                if prior:
                    value = float(prior[0])
                    if np.isfinite(value) and value > 0:
                        previous_close = value
                if previous_close is None:
                    results[(market, date)] = BulkPath(
                        (),
                        False,
                        "market_start_horizon_incomplete",
                        len(rows),
                        0,
                        benchmark_bars,
                    )
                    continue

            bars: list[tuple[float, float, float, float]] = []
            flat_filled = 0
            for timestamp in expected:
                current_bar = by_timestamp.get(timestamp)
                if current_bar is None:
                    if previous_close is None:
                        raise RuntimeError(
                            "path fill invariant violated: previous close is unavailable"
                        )
                    current_bar = (
                        previous_close,
                        previous_close,
                        previous_close,
                        previous_close,
                    )
                    flat_filled += 1
                bars.append(current_bar)
                previous_close = current_bar[3]
            results[(market, date)] = BulkPath(
                tuple(bars),
                True,
                "flat_filled" if flat_filled else "complete",
                len(rows),
                flat_filled,
                benchmark_bars,
            )
    return results


def _safe_up10_before_dn5(
    bars: tuple[tuple[float, float, float, float], ...],
) -> bool:
    if not bars:
        return False
    entry = bars[0][0]
    up = entry * 1.10
    down = entry * 0.95
    for _, high, low, _ in bars:
        hit_up = high >= up
        hit_down = low <= down
        if hit_down:
            # Same-bar ambiguity is downside first, consistent with SL logic.
            return False
        if hit_up:
            return True
    return False


def _path_outcomes(path: BulkPath) -> dict:
    if not path.complete:
        return {
            "path_complete": False,
            "path_quality": path.quality,
            "raw_bars": path.raw_bars,
            "flat_filled_bars": path.flat_filled_bars,
            "benchmark_bars": path.benchmark_bars,
        }
    bars = list(path.bars)
    gross, outcome = simulate_path(bars, SL, TP, None)
    eod_gross, _ = simulate_path(bars, None, None, None)
    entry = bars[0][0]
    mfe = max(bar[1] for bar in bars) / entry - 1.0
    mae = min(bar[2] for bar in bars) / entry - 1.0
    return {
        "path_complete": True,
        "path_quality": path.quality,
        "raw_bars": path.raw_bars,
        "flat_filled_bars": path.flat_filled_bars,
        "benchmark_bars": path.benchmark_bars,
        "m15_net": float(gross - ROUND_TRIP_COST),
        "m15_outcome": outcome,
        "m15_eod_net": float(eod_gross - ROUND_TRIP_COST),
        "m15_mfe": float(mfe),
        "m15_mae": float(mae),
        "m15_safe_up10": _safe_up10_before_dn5(path.bars),
    }


def _attach_path_outcomes(
    selected: pd.DataFrame,
    db_path: Path,
) -> pd.DataFrame:
    unique_pairs = selected[["market", "date"]].drop_duplicates()
    paths = _bulk_paths(unique_pairs, db_path)
    outcome_rows = []
    for row in unique_pairs.itertuples(index=False):
        values = {"market": row.market, "date": row.date}
        values.update(_path_outcomes(paths[(row.market, row.date)]))
        outcome_rows.append(values)
    outcomes = pd.DataFrame(outcome_rows)
    return selected.merge(outcomes, on=["market", "date"], how="left", validate="many_to_one")


def _add_d1_conservative_outcomes(selected: pd.DataFrame) -> pd.DataFrame:
    out = selected.copy()
    hit_sl = out["down_low_ret"] <= -SL
    hit_tp = out["up_high_ret"] >= TP
    gross = np.where(
        hit_sl,
        -SL,
        np.where(hit_tp, TP, out["eod_ret"]),
    )
    out["d1_net"] = gross - ROUND_TRIP_COST
    out["d1_outcome"] = np.where(hit_sl, "sl", np.where(hit_tp, "tp", "eod"))
    out["d1_eod_net"] = out["eod_ret"] - ROUND_TRIP_COST
    out["d1_mfe"] = out["up_high_ret"]
    out["d1_mae"] = out["down_low_ret"]
    # With one daily candle, both barriers are ambiguous; downside-first makes
    # "safe" require +10% without a -5% touch.
    out["d1_safe_up10"] = (
        (out["up_high_ret"] >= 0.10)
        & (out["down_low_ret"] > -0.05)
    )
    return out


def _common_complete_dates(selected: pd.DataFrame) -> set:
    quality = (
        selected.groupby(["date", "policy"])
        .agg(n=("market", "size"), complete=("path_complete", "sum"))
        .reset_index()
    )
    good = quality[(quality["n"] == TOP_K) & (quality["complete"] == TOP_K)]
    policy_count = good.groupby("date")["policy"].nunique()
    return set(policy_count[policy_count == len(POLICIES)].index)


def _standardize_cohort(
    selected: pd.DataFrame,
    prefix: str,
    cohort: str,
) -> pd.DataFrame:
    columns = {
        f"{prefix}_net": "net",
        f"{prefix}_outcome": "outcome",
        f"{prefix}_eod_net": "eod_net",
        f"{prefix}_mfe": "mfe",
        f"{prefix}_mae": "mae",
        f"{prefix}_safe_up10": "safe_up10",
    }
    out = selected.rename(columns=columns).copy()
    out["cohort"] = cohort
    return out


def _metrics(frame: pd.DataFrame, expected_days: int) -> dict:
    frame = frame.dropna(subset=["net"]).copy()
    daily = frame.groupby("date")["net"].mean().sort_index()
    equity = (1.0 + daily).cumprod()
    peak = equity.cummax()
    net = frame["net"].to_numpy(dtype=float)
    k5 = max(1, int(np.ceil(0.05 * len(net))))
    sd = float(daily.std(ddof=1)) if len(daily) > 1 else np.nan
    return {
        "n": int(len(frame)),
        "n_days": int(frame["date"].nunique()),
        "top3_coverage": (
            float(frame["date"].nunique() / expected_days)
            if expected_days
            else np.nan
        ),
        "avg_picks_per_day": (
            float(len(frame) / frame["date"].nunique())
            if frame["date"].nunique()
            else np.nan
        ),
        "replacement_rate_vs_r1": float(
            1.0 - frame["is_original_r1_top3"].mean()
        ),
        "candidate_veto_rate": float(
            frame.groupby("date")["_veto_count"].first().mean() / 100.0
        ),
        "fallback_pick_rate": float(
            frame.groupby("date")["_fallback_count"].first().sum()
            / max(len(frame), 1)
        ),
        "net_mean": float(net.mean()),
        "daily_net_mean": float(daily.mean()),
        "hit_rate": float((net > 0).mean()),
        "sl_first_rate": float((frame["outcome"] == "sl").mean()),
        "tp_first_rate": float((frame["outcome"] == "tp").mean()),
        "mfe10_rate": float((frame["mfe"] >= 0.10).mean()),
        "mfe20_rate": float((frame["mfe"] >= 0.20).mean()),
        "dn5_rate": float((frame["mae"] <= -0.05).mean()),
        "safe_up10_rate": float(frame["safe_up10"].astype(bool).mean()),
        "eod_deep_loss_rate": float((frame["eod_net"] <= -0.05).mean()),
        "mfe_mean": float(frame["mfe"].mean()),
        "mae_mean": float(frame["mae"].mean()),
        "cvar95": float(np.sort(net)[:k5].mean()),
        "cum_net": float(equity.iloc[-1] - 1.0),
        "max_drawdown": float(((equity - peak) / peak).min()),
        "sharpe_sqrt365": (
            float(daily.mean() / sd * np.sqrt(365.0))
            if np.isfinite(sd) and sd > 0
            else np.nan
        ),
    }


def _summary(
    cohorts: list[pd.DataFrame],
    expected_days: dict[str, int],
) -> pd.DataFrame:
    rows = []
    for cohort_frame in cohorts:
        cohort = str(cohort_frame["cohort"].iloc[0])
        for policy in POLICIES:
            part = cohort_frame[cohort_frame["policy"] == policy]
            row = _metrics(part, expected_days[cohort])
            row.update(cohort=cohort, policy=policy, segment="ALL")
            rows.append(row)
    summary = pd.DataFrame(rows)
    for cohort in summary["cohort"].unique():
        base_safe = float(
            summary[
                (summary["cohort"] == cohort)
                & (summary["policy"] == BASELINE)
            ]["safe_up10_rate"].iloc[0]
        )
        base_up10 = float(
            summary[
                (summary["cohort"] == cohort)
                & (summary["policy"] == BASELINE)
            ]["mfe10_rate"].iloc[0]
        )
        mask = summary["cohort"] == cohort
        summary.loc[mask, "safe_up10_retention_vs_r1"] = (
            summary.loc[mask, "safe_up10_rate"] / base_safe
            if base_safe > 0
            else np.nan
        )
        summary.loc[mask, "up10_retention_vs_r1"] = (
            summary.loc[mask, "mfe10_rate"] / base_up10
            if base_up10 > 0
            else np.nan
        )
    return summary


def _fold_summary(cohorts: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for cohort_frame in cohorts:
        cohort = str(cohort_frame["cohort"].iloc[0])
        for fold, fold_frame in cohort_frame.groupby("fold"):
            expected = int(fold_frame["date"].nunique())
            for policy in POLICIES:
                part = fold_frame[fold_frame["policy"] == policy]
                row = _metrics(part, expected)
                row.update(cohort=cohort, policy=policy, fold=int(fold))
                rows.append(row)
    return pd.DataFrame(rows)


def _daily_metric(frame: pd.DataFrame, metric: str) -> pd.Series:
    if metric == "net":
        values = frame["net"]
    elif metric == "hit_rate":
        values = (frame["net"] > 0).astype(float)
    elif metric == "sl_first_rate":
        values = (frame["outcome"] == "sl").astype(float)
    elif metric == "tp_first_rate":
        values = (frame["outcome"] == "tp").astype(float)
    elif metric == "mfe10_rate":
        values = (frame["mfe"] >= 0.10).astype(float)
    elif metric == "dn5_rate":
        values = (frame["mae"] <= -0.05).astype(float)
    elif metric == "safe_up10_rate":
        values = frame["safe_up10"].astype(float)
    elif metric == "eod_deep_loss_rate":
        values = (frame["eod_net"] <= -0.05).astype(float)
    else:
        raise KeyError(metric)
    return values.groupby(frame["date"]).mean().sort_index()


def _paired_ci(
    cohorts: list[pd.DataFrame],
    draws: int,
    seed: int,
) -> pd.DataFrame:
    metrics = (
        "net",
        "hit_rate",
        "sl_first_rate",
        "tp_first_rate",
        "mfe10_rate",
        "dn5_rate",
        "safe_up10_rate",
        "eod_deep_loss_rate",
    )
    lower_better = {
        "sl_first_rate",
        "dn5_rate",
        "eod_deep_loss_rate",
    }
    rows = []
    for cohort_frame in cohorts:
        cohort = str(cohort_frame["cohort"].iloc[0])
        baseline = cohort_frame[cohort_frame["policy"] == BASELINE]
        for policy in CHALLENGERS:
            challenger = cohort_frame[cohort_frame["policy"] == policy]
            for metric in metrics:
                base_daily = _daily_metric(baseline, metric)
                challenger_daily = _daily_metric(challenger, metric)
                common = base_daily.index.intersection(challenger_daily.index)
                delta = (
                    challenger_daily.loc[common].to_numpy()
                    - base_daily.loc[common].to_numpy()
                )
                if len(delta) == 0:
                    continue
                salt_payload = (
                    f"{seed}|{cohort}|{policy}|{metric}"
                ).encode("utf-8")
                stream_seed = int.from_bytes(
                    hashlib.sha256(salt_payload).digest()[:8],
                    "big",
                )
                rng = np.random.default_rng(stream_seed)
                indices = rng.integers(0, len(delta), size=(draws, len(delta)))
                boot = delta[indices].mean(axis=1)
                rows.append(
                    {
                        "cohort": cohort,
                        "challenger": policy,
                        "metric": metric,
                        "n_paired_dates": int(len(delta)),
                        "delta_challenger_minus_r1": float(delta.mean()),
                        "ci95_lo": float(np.percentile(boot, 2.5)),
                        "ci95_hi": float(np.percentile(boot, 97.5)),
                        "p_delta_gt_zero": float((boot > 0).mean()),
                        "direction": (
                            "lower_better" if metric in lower_better else "higher_better"
                        ),
                        "bootstrap_draws": int(draws),
                        "seed": int(seed),
                        "stream_seed": int(stream_seed),
                    }
                )
    return pd.DataFrame(rows)


def _cohort_slice(
    standardized: pd.DataFrame,
    cohort: str,
    dates: set,
) -> pd.DataFrame:
    out = standardized[standardized["date"].isin(dates)].copy()
    out["cohort"] = cohort
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-oos", type=Path, default=INPUT_OOS)
    parser.add_argument("--m15-db", type=Path, default=M15_DB)
    parser.add_argument("--output-prefix", type=Path, default=OUTPUT_PREFIX)
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--validate-existing",
        action="store_true",
        help="validate current artifacts and exit without rebuilding",
    )
    args = parser.parse_args()
    if args.validate_existing:
        audit = validate_existing_artifacts(
            output_prefix=args.output_prefix,
            input_oos=args.input_oos,
            m15_db=args.m15_db,
        )
        print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
        return
    if not 100 <= args.bootstrap_draws <= 100_000:
        raise ValueError("--bootstrap-draws must be in [100, 100000]")
    if not 0 <= args.seed <= np.iinfo(np.int64).max:
        raise ValueError("--seed must be in [0, 2**63-1]")
    if not args.m15_db.is_file():
        raise FileNotFoundError(args.m15_db)
    source_signatures = {
        "input_oos": _file_signature(args.input_oos),
        "m15_db": _file_signature(args.m15_db),
    }
    code_lineage = _code_lineage()

    frame = _read_oos(args.input_oos)
    dates = sorted(frame["date"].unique())
    if len(dates) <= LOCKED_HOLDOUT_DAYS:
        raise ValueError("not enough OOS dates for locked holdout")
    holdout_dates = set(dates[-LOCKED_HOLDOUT_DAYS:])
    holdout_start = min(holdout_dates)
    preholdout_dates = set(dates) - holdout_dates
    calibration = _preholdout_calibration(frame, holdout_start)

    selected = _select_all(frame, calibration)
    selected = _add_d1_conservative_outcomes(selected)
    selected = _attach_path_outcomes(selected, args.m15_db)
    common_dates = _common_complete_dates(selected)
    locked_common_dates = common_dates & holdout_dates

    d1_all = _standardize_cohort(
        selected, "d1", "d1_conservative_all_oos"
    )
    d1_locked = _cohort_slice(
        d1_all, "d1_conservative_locked_180", holdout_dates
    )
    m15_all = _standardize_cohort(
        selected[selected["date"].isin(common_dates)],
        "m15",
        "m15_common_complete_all",
    )
    m15_locked = _cohort_slice(
        m15_all, "m15_common_complete_locked_180", locked_common_dates
    )
    cohorts = [d1_all, d1_locked, m15_all, m15_locked]
    cohorts = [cohort for cohort in cohorts if len(cohort)]
    expected_days = {
        "d1_conservative_all_oos": len(dates),
        "d1_conservative_locked_180": len(holdout_dates),
        "m15_common_complete_all": len(common_dates),
        "m15_common_complete_locked_180": len(locked_common_dates),
    }

    summary = _summary(cohorts, expected_days)
    paired = _paired_ci(
        cohorts,
        draws=args.bootstrap_draws,
        seed=args.seed,
    )
    folds = _fold_summary(cohorts)
    if {
        "input_oos": _file_signature(args.input_oos),
        "m15_db": _file_signature(args.m15_db),
    } != source_signatures:
        raise RuntimeError("research input changed while challenger was running")
    if _code_lineage() != code_lineage:
        raise RuntimeError("local code changed while challenger was running")

    pick_columns = [
        "date",
        "market",
        "fold",
        "regime",
        "policy",
        "selection_rank",
        "r1_rank",
        "is_original_r1_top3",
        "candidate_count",
        "_veto_count",
        "_fallback_count",
        "R1",
        "p_lab_up_10",
        "p_lab_dn_05",
        "_cal_p_dn5",
        "p_lab_dn_10",
        "exp_downside",
        "f_qv_rank",
        "up_high_ret",
        "down_low_ret",
        "eod_ret",
        "d1_net",
        "d1_outcome",
        "d1_eod_net",
        "d1_safe_up10",
        "path_complete",
        "path_quality",
        "raw_bars",
        "flat_filled_bars",
        "benchmark_bars",
        "m15_net",
        "m15_outcome",
        "m15_eod_net",
        "m15_mfe",
        "m15_mae",
        "m15_safe_up10",
    ]
    selected["is_locked_holdout"] = selected["date"].isin(holdout_dates)
    selected["is_common_complete_date"] = selected["date"].isin(common_dates)
    selected["is_locked_common_complete_date"] = selected["date"].isin(
        locked_common_dates
    )
    pick_columns += [
        "is_locked_holdout",
        "is_common_complete_date",
        "is_locked_common_complete_date",
    ]
    picks = selected[pick_columns].copy()

    path_by_policy = {}
    for policy, group in selected.groupby("policy"):
        day_quality = group.groupby("date")["path_complete"].all()
        path_by_policy[policy] = {
            "complete_pick_rate": float(group["path_complete"].mean()),
            "complete_top3_days": int(day_quality.sum()),
            "all_days": int(day_quality.size),
            "flat_filled_pick_rate_among_complete": float(
                (
                    group.loc[group["path_complete"], "path_quality"]
                    == "flat_filled"
                ).mean()
            ),
        }

    coverage = {
        "schema": "downside_veto_challenger_v1",
        "created_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "research_only": True,
        "input_oos": str(args.input_oos),
        "input_oos_sha256": _sha256(args.input_oos),
        "m15_db": str(args.m15_db),
        "m15_db_sha256": _sha256(args.m15_db),
        "script_sha256": _sha256(Path(__file__)),
        "code_lineage": code_lineage,
        "source_signatures": source_signatures,
        "sources_before_and_after_identical": True,
        "oos_rows": int(len(frame)),
        "oos_dates": int(len(dates)),
        "oos_window": [str(min(dates)), str(max(dates))],
        "candidate_set": "static D-1 Top 100, identical for every policy/date",
        "top_k": TOP_K,
        "policies": list(POLICIES),
        "current_challenger_trials": len(CHALLENGERS),
        "current_grid_note": (
            "3 fixed challenger transforms; no threshold/parameter sweep"
        ),
        "related_prior_policy_variants_minimum": 15,
        "related_prior_note": (
            "R2 lambda variants 4 + R3 gate variants 3 + A1 variants 8; "
            "meta-filter is a separate previously attempted layer"
        ),
        "holdout": {
            "locked_days": LOCKED_HOLDOUT_DAYS,
            "start": str(holdout_start),
            "end": str(max(holdout_dates)),
            "preholdout_days": len(preholdout_dates),
            "rule_lock": (
                "q=1/3, calibrated absolute cutoff=0.50, and lexicographic "
                "ordering fixed before challenger evaluation"
            ),
        },
        "absolute_calibration": calibration,
        "wf_hygiene": {
            "input": (
                "6-fold purged walk-forward OOS probabilities, embargo=5, "
                "D-1 features and day-D outcomes"
            ),
            "holdout_calibration": (
                "single scalar factor fit only on dates before locked holdout; "
                "frozen throughout holdout"
            ),
            "within_day_veto": (
                "cross-sectional p_dn5 ranks from the same observable day only"
            ),
            "outcome_not_used_for_ranking": True,
            "row_level_checks": (
                "dates are single-fold, folds are chronological/nonoverlapping, "
                "and every fold-date has an exact finite Top100"
            ),
            "upstream_train_embargo_provenance": {
                "status": "legacy_artifact_not_cryptographically_bound",
                "producer_path": str(UPSTREAM_PRODUCER),
                "current_producer_sha256": (
                    _sha256(UPSTREAM_PRODUCER)
                    if UPSTREAM_PRODUCER.is_file()
                    else None
                ),
                "limitation": (
                    "the row artifact contains test fold IDs but no train-range "
                    "manifest, so its claimed upstream five-date embargo cannot "
                    "be independently reconstructed from this file"
                ),
                "promotion_eligible": False,
            },
        },
        "path_rule": (
            "[D 09:15,D+1 09:15) first-executable-bar window; "
            "KRW-BTC exact 96-bar completeness + closed next boundary; "
            "target-only no-trade gaps flat-filled; incomplete paths excluded "
            "through a common-date intersection across all policies"
        ),
        "common_complete_dates": {
            "n": len(common_dates),
            "start": str(min(common_dates)) if common_dates else None,
            "end": str(max(common_dates)) if common_dates else None,
        },
        "locked_common_complete_dates": {
            "n": len(locked_common_dates),
            "start": str(min(locked_common_dates)) if locked_common_dates else None,
            "end": str(max(locked_common_dates)) if locked_common_dates else None,
        },
        "path_coverage_by_policy": path_by_policy,
        "cost_and_path": {
            "round_trip_cost": ROUND_TRIP_COST,
            "tp": TP,
            "sl": SL,
            "same_bar_priority": "SL before TP",
            "daily_candle_cohort": (
                "conservative approximation only; if both daily barriers touch, "
                "SL is assigned first"
            ),
        },
        "bootstrap": {
            "paired_unit": "date",
            "draws": args.bootstrap_draws,
            "seed": args.seed,
        },
    }
    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    artifact_paths = _artifact_paths(prefix)
    manifest_path = Path(f"{prefix}_manifest.json")
    with tempfile.TemporaryDirectory(
        dir=prefix.parent,
        prefix=f".{prefix.name}.generation.",
    ) as stage_directory:
        stage_root = Path(stage_directory)
        staged = {
            name: stage_root / path.name
            for name, path in artifact_paths.items()
        }
        summary.to_csv(staged["summary"], index=False)
        paired.to_csv(staged["paired_ci"], index=False)
        folds.to_csv(staged["folds"], index=False)
        picks.to_csv(staged["picks"], index=False)
        _write_json(staged["coverage"], coverage)
        staged_manifest = stage_root / manifest_path.name
        _write_json(
            staged_manifest,
            {
                "schema": "downside_veto_challenger_v1_manifest",
                "files": {
                    name: {
                        "path": str(artifact_paths[name]),
                        "bytes": int(path.stat().st_size),
                        "sha256": _sha256(path),
                    }
                    for name, path in staged.items()
                },
            },
        )
        publish_map = {
            staged[name]: target
            for name, target in artifact_paths.items()
        }
        publish_map[staged_manifest] = manifest_path
        _publish_staged_files(publish_map)

    print(
        f"OOS={len(dates)}d, locked={len(holdout_dates)}d "
        f"({holdout_start}..{max(holdout_dates)}), "
        f"m15_common={len(common_dates)}d, "
        f"m15_locked_common={len(locked_common_dates)}d"
    )
    locked_rows = summary[
        summary["cohort"] == "m15_common_complete_locked_180"
    ][
        [
            "policy",
            "n",
            "net_mean",
            "hit_rate",
            "sl_first_rate",
            "dn5_rate",
            "mfe10_rate",
            "safe_up10_rate",
            "safe_up10_retention_vs_r1",
        ]
    ]
    print(locked_rows.to_string(index=False))
    print(f"wrote {prefix}_*.csv/json")


if __name__ == "__main__":
    main()

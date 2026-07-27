"""Read-only row-artifact recalculation for challenger quant audit v1.

This script intentionally writes nothing.  It rebuilds the locked/common
date-level metrics and bootstrap intervals independently of the three reports.
For the upside-head artifact, which does not persist path rows, it reconstructs
the exact 09:15 path cache from the read-only 15-minute database.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SEED = 20260725
DRAWS = 20_000
D1_DB = ROOT / "data/upbit_d1.db"
M15_DB = ROOT / "data/upbit_15m.db"
DOWNSIDE_COVERAGE = ROOT / "output/downside_veto_challenger_v1_coverage.json"
SAFEUP_COVERAGE = ROOT / "output/safeup_head_challenger_v1_coverage.json"
UPSIDE_COVERAGE = ROOT / "output/upside_head_challenger_v1_coverage.json"
CC_CANDIDATE = ROOT / "output/cc_filtered_multiday_oos_v1.parquet"
DOWNSIDE_SOURCE = ROOT / "scripts/downside_veto_challenger_v1.py"
SAFEUP_SOURCE = ROOT / "scripts/safeup_head_challenger_v1.py"
UPSIDE_SOURCE = ROOT / "scripts/upside_head_challenger_v1.py"


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


def _cluster_ci(delta: pd.Series, seed_offset: int = 0) -> dict:
    numeric = pd.to_numeric(delta, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("bootstrap input contains missing or non-finite values")
    values = numeric.to_numpy(float)
    if not len(values):
        raise ValueError("bootstrap input is empty")
    rng = np.random.default_rng(SEED + seed_offset)
    sample = rng.integers(0, len(values), size=(DRAWS, len(values)))
    boot = values[sample].mean(axis=1)
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "ci95": [
            float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)),
        ],
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(path: Path) -> dict:
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
    if frame.duplicated(columns).any():
        raise ValueError(f"{name}: duplicate keys {columns}")


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


def _completed_cutoff() -> pd.Timestamp:
    now = datetime.now(ZoneInfo("Asia/Seoul"))
    session_date = (now - timedelta(hours=9)).date()
    return pd.Timestamp(session_date - timedelta(days=1))


def _hash_frame(frame: pd.DataFrame, sort_columns: list[str]) -> str:
    payload = (
        frame.sort_values(sort_columns)
        .to_json(orient="records", date_format="iso", double_precision=15)
        .encode()
    )
    return hashlib.sha256(payload).hexdigest()


def _time_order_from_entries(
    entries: list[dict],
    *,
    fold_key: str,
    variant_key: str,
) -> dict:
    seen: set[tuple[str, int]] = set()
    windows: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for entry in entries:
        variant = str(entry[variant_key])
        if fold_key in entry:
            fold = int(entry[fold_key])
        elif entry.get("scope") == "locked_holdout":
            fold = -1
        else:
            raise ValueError(f"coverage: missing fold identifier for {variant}")
        key = (variant, fold)
        if key in seen:
            raise ValueError(f"coverage: duplicate fold metadata: {key}")
        seen.add(key)
        train_end = pd.Timestamp(entry["train_end"])
        test_start = pd.Timestamp(entry["test_start"])
        test_end = pd.Timestamp(entry["test_end"])
        embargo = int(entry.get("embargo_dates", entry.get("embargo_days", -1)))
        if train_end >= test_start or test_start > test_end:
            raise ValueError(f"coverage: train/test overlap: {key}")
        if embargo < 5 or (test_start - train_end).days - 1 < 5:
            raise ValueError(f"coverage: embargo shorter than five dates: {key}")
        windows.setdefault(variant, []).append((test_start, test_end))
    for variant, variant_windows in windows.items():
        ordered = sorted(variant_windows)
        if any(current[0] <= previous[1] for previous, current in zip(ordered, ordered[1:])):
            raise ValueError(f"coverage: overlapping test folds for {variant}")
    return {
        "metadata_entries": len(entries),
        "unique_variant_fold_entries": len(seen),
        "overlap_or_embargo_violations": 0,
    }


def _daily_path(frame: pd.DataFrame, *, prefix: str = "") -> pd.DataFrame:
    if frame.empty:
        raise ValueError("daily path cohort is empty")
    work = frame.copy()
    if prefix:
        net = f"{prefix}net"
        outcome = f"{prefix}outcome"
        mfe = f"{prefix}mfe"
        mae = f"{prefix}mae"
        safe = f"{prefix}safe_up10"
    else:
        net = "path_net"
        outcome = "path_outcome"
        mfe = "path_mfe"
        mae = "path_mae"
        safe = "path_safe_up10"
    _require_columns(
        work,
        {"date", net, outcome, mfe, mae, safe},
        name="daily path cohort",
    )
    _require_finite(work, {net, mfe, mae, safe}, name="daily path cohort")
    if not set(work[outcome]).issubset({"tp", "sl", "eod"}):
        raise ValueError(f"daily path cohort has invalid outcomes in {outcome}")
    work["_sl"] = work[outcome].eq("sl").astype(float)
    work["_tp"] = work[outcome].eq("tp").astype(float)
    work["_up10"] = work[mfe].ge(0.10).astype(float)
    work["_dn5"] = work[mae].le(-0.05).astype(float)
    work["_safe"] = work[safe].astype(float)
    return work.groupby("date", sort=True).agg(
        net=(net, "mean"),
        sl=("_sl", "mean"),
        tp=("_tp", "mean"),
        up10=("_up10", "mean"),
        dn5=("_dn5", "mean"),
        safe=("_safe", "mean"),
        n=(net, "size"),
    )


def downside() -> dict:
    from scripts import downside_veto_challenger_v1 as downside_module
    from scripts import safeup_head_challenger_v1 as safeup_module

    path = ROOT / "output/downside_veto_challenger_v1_picks.csv"
    frame = pd.read_csv(path, parse_dates=["date"])
    _require_columns(
        frame,
        {
            "market", "date", "fold", "policy", "selection_rank",
            "is_locked_holdout", "is_locked_common_complete_date",
            "path_complete", "m15_net", "m15_outcome", "m15_mfe",
            "m15_mae", "m15_safe_up10",
        },
        name="downside picks",
    )
    if frame.empty or frame["date"].isna().any():
        raise ValueError("downside picks are empty or have invalid dates")
    if frame["date"].max() > _completed_cutoff():
        raise ValueError("downside picks contain incomplete/future dates")
    _require_unique(
        frame,
        ["date", "policy", "market"],
        name="downside picks",
    )
    for column in (
        "is_locked_holdout",
        "is_locked_common_complete_date",
        "path_complete",
    ):
        frame[column] = _bool(frame[column])
    locked_all = frame[_bool(frame["is_locked_holdout"])].copy()
    frame = frame[_bool(frame["is_locked_common_complete_date"])].copy()
    expected_policies = {
        "R1_baseline",
        "lexicographic_risk_first",
        "veto_train_cal50",
        "veto_within_day_top_third",
    }
    if set(frame["policy"]) != expected_policies:
        raise ValueError("downside picks: unexpected/missing policies")
    full_counts = frame.groupby(["date", "policy"]).size()
    if not full_counts.eq(3).all():
        raise ValueError("downside picks: full persisted population is not Top3")
    if not frame.groupby("date")["policy"].nunique().eq(4).all():
        raise ValueError("downside picks: full policy population is incomplete")
    counts = frame.groupby(["date", "policy"]).size()
    if counts.empty or not counts.eq(3).all():
        raise ValueError("downside picks: locked cohort is not exact Top3")
    if not frame.groupby("date")["policy"].nunique().eq(4).all():
        raise ValueError("downside picks: locked common dates are incomplete")
    if not frame["path_complete"].all():
        raise ValueError("downside picks: locked cohort has incomplete paths")
    table = {}
    daily = {}
    for policy, group in frame.groupby("policy", sort=True):
        d = _daily_path(group, prefix="m15_")
        daily[policy] = d
        table[policy] = {
            "rows": int(len(group)),
            "dates": int(group["date"].nunique()),
            **{name: float(d[name].mean()) for name in ("net", "sl", "tp", "up10", "dn5", "safe")},
        }
    paired = {}
    base = daily["R1_baseline"]
    for offset, (policy, other) in enumerate(sorted(daily.items()), start=1):
        if policy == "R1_baseline":
            continue
        common = other.join(base, lsuffix="_candidate", rsuffix="_base", how="inner")
        paired[policy] = {
            metric: _cluster_ci(
                common[f"{metric}_candidate"] - common[f"{metric}_base"],
                seed_offset=offset * 10 + index,
            )
            for index, metric in enumerate(("net", "sl", "up10", "dn5", "safe"))
        }

    with _connect_readonly(D1_DB) as connection:
        history = pd.read_sql_query(
            "SELECT market, timestamp FROM candles",
            connection,
            parse_dates=["timestamp"],
        )
    history = history.sort_values(["market", "timestamp"])
    if history.empty or history["timestamp"].isna().any():
        raise ValueError("D1 history is empty or invalid")
    _require_unique(history, ["market", "timestamp"], name="D1 history")
    history["history_prior_bars"] = history.groupby("market").cumcount()
    history["date"] = history["timestamp"].dt.normalize()
    candidate = pd.read_parquet(
        CC_CANDIDATE,
        columns=["market", "date"],
    )
    candidate["date"] = pd.to_datetime(candidate["date"]).dt.normalize()
    _require_unique(candidate, ["market", "date"], name="candidate OOS panel")
    candidate_history = candidate.merge(
        history[["market", "date", "history_prior_bars"]],
        on=["market", "date"],
        how="left",
        validate="one_to_one",
    )
    if candidate_history["history_prior_bars"].isna().any():
        raise ValueError("candidate OOS panel is missing point-in-time history")
    holdout_dates = set(sorted(candidate_history["date"].unique())[-180:])
    candidate_locked = candidate_history[
        candidate_history["date"].isin(holdout_dates)
    ]
    selected_history = locked_all.merge(
        history[["market", "date", "history_prior_bars"]],
        on=["market", "date"],
        how="left",
        validate="many_to_one",
    )
    if selected_history["history_prior_bars"].isna().any():
        raise ValueError("selected downside rows are missing point-in-time history")

    wanted = locked_all[["market", "date"]].drop_duplicates()
    paths_0915, checked, checked_complete = safeup_module._bulk_execution_paths(
        wanted,
        ROOT / "data/upbit_15m.db",
    )
    outcomes = []
    for pair in wanted.itertuples(index=False):
        row = {"market": pair.market, "date": pair.date}
        row.update(
            downside_module._path_outcomes(
                paths_0915[(str(pair.market), pair.date)]
            )
        )
        outcomes.append(row)
    outcomes_frame = pd.DataFrame(outcomes)
    _require_unique(outcomes_frame, ["market", "date"], name="09:15 path cache")
    refreshed_columns = {
        key
        for row in outcomes
        for key in row
        if key not in {"market", "date"}
    }
    stored_paths = (
        locked_all[
            ["market", "date", *sorted(refreshed_columns)]
        ]
        .drop_duplicates()
        .sort_values(["date", "market"])
        .reset_index(drop=True)
    )
    if stored_paths.duplicated(["market", "date"]).any():
        raise ValueError("downside stored path outcomes disagree across policies")
    recalculated_paths = (
        outcomes_frame[
            ["market", "date", *sorted(refreshed_columns)]
        ]
        .sort_values(["date", "market"])
        .reset_index(drop=True)
    )
    try:
        pd.testing.assert_frame_equal(
            stored_paths,
            recalculated_paths,
            check_dtype=False,
            check_exact=False,
            rtol=0.0,
            atol=1e-12,
        )
    except AssertionError as exc:
        raise ValueError(
            "downside stored paths disagree with independent 09:15 recalculation"
        ) from exc
    selected_0915 = locked_all.drop(
        columns=[column for column in refreshed_columns if column in locked_all],
    ).merge(
        outcomes_frame,
        on=["market", "date"],
        how="left",
        validate="many_to_one",
    )
    quality = selected_0915.groupby(["date", "policy"], sort=True).agg(
        n=("market", "size"),
        complete=("path_complete", "sum"),
    )
    good = quality[(quality["n"] == 3) & (quality["complete"] == 3)].reset_index()
    policy_counts = good.groupby("date")["policy"].nunique()
    dates_0915 = set(policy_counts[policy_counts == 4].index)
    if not dates_0915:
        raise ValueError("downside 09:15 sensitivity has no common dates")
    cohort_0915 = selected_0915[
        selected_0915["date"].isin(dates_0915)
        & selected_0915["path_complete"]
    ]
    table_0915 = {}
    daily_0915 = {}
    for policy, group in cohort_0915.groupby("policy", sort=True):
        group = group.rename(
            columns={
                "m15_net": "path_net",
                "m15_outcome": "path_outcome",
                "m15_mfe": "path_mfe",
                "m15_mae": "path_mae",
                "m15_safe_up10": "path_safe_up10",
            }
        )
        d = _daily_path(group)
        daily_0915[policy] = d
        table_0915[policy] = {
            "rows": int(len(group)),
            "dates": int(group["date"].nunique()),
            **{
                name: float(d[name].mean())
                for name in ("net", "sl", "tp", "up10", "dn5", "safe")
            },
        }
    paired_0915 = {}
    base_0915 = daily_0915["R1_baseline"]
    for offset, (policy, other) in enumerate(sorted(daily_0915.items()), start=1):
        if policy == "R1_baseline":
            continue
        common = other.join(
            base_0915,
            lsuffix="_candidate",
            rsuffix="_base",
            how="inner",
        )
        paired_0915[policy] = {
            metric: _cluster_ci(
                common[f"{metric}_candidate"] - common[f"{metric}_base"],
                seed_offset=50 + offset * 10 + index,
            )
            for index, metric in enumerate(("net", "sl", "up10", "dn5", "safe"))
        }
    return {
        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "input_sha256": {
            "candidate_oos": _sha(CC_CANDIDATE),
            "coverage": _sha(DOWNSIDE_COVERAGE),
        },
        "locked_common_table": table,
        "paired_vs_R1": paired,
        "point_in_time_history_audit": {
            "all_oos_rows": int(len(candidate_history)),
            "all_oos_history_lt70": int(
                (candidate_history["history_prior_bars"] < 70).sum()
            ),
            "all_oos_dates_affected": int(
                candidate_history.loc[
                    candidate_history["history_prior_bars"] < 70, "date"
                ].nunique()
            ),
            "locked_rows": int(len(candidate_locked)),
            "locked_history_lt70": int(
                (candidate_locked["history_prior_bars"] < 70).sum()
            ),
            "locked_dates_affected": int(
                candidate_locked.loc[
                    candidate_locked["history_prior_bars"] < 70, "date"
                ].nunique()
            ),
            "locked_selected_by_policy": {
                policy: {
                    "rows": int(len(group)),
                    "history_lt70": int(
                        (group["history_prior_bars"] < 70).sum()
                    ),
                    "dates_affected": int(
                        group.loc[group["history_prior_bars"] < 70, "date"].nunique()
                    ),
                    "min_history": int(group["history_prior_bars"].min()),
                }
                for policy, group in selected_history.groupby("policy", sort=True)
            },
        },
        "independent_0915_path_recalculation": {
            "common_dates": int(len(dates_0915)),
            "canonical_crosschecks": int(checked),
            "canonical_complete_crosschecks": int(checked_complete),
            "table": table_0915,
            "paired_vs_R1": paired_0915,
            "path_cache_sha256": _hash_frame(
                outcomes_frame,
                ["date", "market"],
            ),
            "warning": (
                "The stored 09:15 paths match this independent recalculation, "
                "but the upstream history gate and in-sample calibration "
                "limitations remain."
            ),
        },
    }


SAFE_POLICIES = (
    "safeup_head",
    "up10_control",
    "R1_repaired",
    "R1_frozen_pattern",
    "monkey_seed42",
)


def _complete_policy_days(frame: pd.DataFrame, policies: tuple[str, ...]) -> set[pd.Timestamp]:
    scoped = frame[frame["policy"].isin(policies)].copy()
    quality = scoped.groupby(["date", "policy"], sort=True).agg(
        n=("market", "size"),
        complete=("path_complete", "sum"),
    )
    good = quality[(quality["n"] == 3) & (quality["complete"] == 3)].reset_index()
    counts = good.groupby("date")["policy"].nunique()
    return set(counts[counts == len(policies)].index)


def safeup() -> dict:
    path = ROOT / "output/safeup_head_challenger_v1_picks.csv.gz"
    frame = pd.read_csv(path, parse_dates=["date"])
    _require_columns(
        frame,
        {
            "market", "date", "scope", "fold", "policy", "selection_rank",
            "path_complete", "path_common_complete_date", "path_net",
            "path_outcome", "path_mfe", "path_mae", "path_safe_up10",
            "path_dn5", "path_up10",
        },
        name="safeup picks",
    )
    if frame.empty or frame["date"].isna().any():
        raise ValueError("safeup picks are empty or have invalid dates")
    if frame["date"].max() > _completed_cutoff():
        raise ValueError("safeup picks contain incomplete/future dates")
    _require_unique(
        frame,
        ["scope", "date", "policy", "market"],
        name="safeup picks",
    )
    frame["path_complete"] = _bool(frame["path_complete"])
    frame["path_common_complete_date"] = _bool(frame["path_common_complete_date"])
    locked = frame[frame["scope"].eq("locked_holdout")].copy()
    if set(frame["scope"]) != {"discovery_oof", "locked_holdout"}:
        raise ValueError("safeup picks: unexpected/missing scopes")
    expected = set(SAFE_POLICIES) | {"safeup_pareto_rank"}
    if set(locked["policy"]) != expected:
        raise ValueError("safeup picks: unexpected/missing locked policies")
    locked_counts = locked.groupby(["date", "policy"]).size()
    if not locked_counts.eq(3).all():
        raise ValueError("safeup picks: locked population is not exact Top3")
    if not locked.groupby("date")["policy"].nunique().eq(6).all():
        raise ValueError("safeup picks: locked policy population is incomplete")
    reported_dates = set(
        locked.loc[locked["path_common_complete_date"], "date"].unique()
    )
    exante_dates = _complete_policy_days(locked, SAFE_POLICIES)
    all_six_dates = _complete_policy_days(
        locked, SAFE_POLICIES + ("safeup_pareto_rank",)
    )
    if not exante_dates or not all_six_dates:
        raise ValueError("safeup picks: empty reconstructed common cohort")
    if reported_dates != all_six_dates:
        raise ValueError("safeup picks: reported common-date flag mismatch")

    def table_for(dates: set[pd.Timestamp], policies: tuple[str, ...]) -> dict:
        cohort = locked[
            locked["date"].isin(dates)
            & locked["policy"].isin(policies)
            & locked["path_complete"]
        ].copy()
        output = {}
        for policy, group in cohort.groupby("policy", sort=True):
            d = _daily_path(group)
            output[policy] = {
                "rows": int(len(group)),
                "dates": int(group["date"].nunique()),
                **{
                    name: float(d[name].mean())
                    for name in ("net", "sl", "tp", "up10", "dn5", "safe")
                },
            }
        return output

    pairwise = {}
    primary = locked[locked["policy"].eq("safeup_head") & locked["path_complete"]]
    primary_daily = _daily_path(primary)
    for offset, comparator in enumerate(
        [p for p in SAFE_POLICIES if p != "safeup_head"], start=1
    ):
        other = locked[locked["policy"].eq(comparator) & locked["path_complete"]]
        other_daily = _daily_path(other)
        common = primary_daily[primary_daily["n"].eq(3)].join(
            other_daily[other_daily["n"].eq(3)],
            how="inner",
            lsuffix="_candidate",
            rsuffix="_base",
        )
        pairwise[comparator] = {
            metric: _cluster_ci(
                common[f"{metric}_candidate"] - common[f"{metric}_base"],
                seed_offset=100 + offset * 10 + index,
            )
            for index, metric in enumerate(("net", "sl", "up10", "dn5", "safe"))
        }

    fold_table = {}
    complete = locked[locked["path_complete"]].copy()
    for policy in ("safeup_head", "R1_repaired", "R1_frozen_pattern", "up10_control"):
        group = complete[complete["policy"].eq(policy)]
        fold_table[policy] = {
            str(int(fold)): {
                "dates": int(part["date"].nunique()),
                "net": float(part["path_net"].mean()),
                "dn5": float(part["path_dn5"].mean()),
                "safe": float(part["path_safe_up10"].mean()),
            }
            for fold, part in group.groupby("fold", sort=True)
        }

    return {
        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "coverage_sha256": _sha(SAFEUP_COVERAGE),
        "reported_common_dates": len(reported_dates),
        "reconstructed_all_six_common_dates": len(all_six_dates),
        "reconstructed_exante_five_common_dates": len(exante_dates),
        "reported_six_policy_table": table_for(
            all_six_dates, SAFE_POLICIES + ("safeup_pareto_rank",)
        ),
        "exante_five_policy_table": table_for(exante_dates, SAFE_POLICIES),
        "pairwise_complete_dates_vs_safeup": pairwise,
        "locked_fold_table": fold_table,
    }


def _random_score(date: object, market: str, seed: int = 42) -> int:
    payload = f"{seed}|{date}|{market}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _top3(frame: pd.DataFrame, column: str, ascending: bool = False) -> pd.DataFrame:
    selected = (
        frame.sort_values(
            ["date", column, "market"],
            ascending=[True, ascending, True],
        )
        .groupby("date", sort=False)
        .head(3)
        .copy()
    )
    counts = selected.groupby("date").size()
    if counts.empty or not counts.eq(3).all():
        raise ValueError(f"Top3 selection is incomplete for {column}")
    return selected


def _upside_daily(
    picks: pd.DataFrame,
    cache: dict[tuple[str, object], dict],
) -> pd.DataFrame:
    rows = []
    for pick in picks.itertuples(index=False):
        item = cache.get((str(pick.market), pick.date), {})
        if not item.get("path_complete", False):
            continue
        rows.append(
            {
                "date": pick.date,
                "net": float(item["path_net"]),
                "sl": float(item["outcome"] == "sl"),
                "tp": float(item["outcome"] == "tp"),
                "up10": float(item["execution_up10"]),
                "dn5": float(item["execution_dn5"]),
                "safe": float(item["execution_safe_up10"]),
            }
        )
    if not rows:
        return pd.DataFrame()
    trades = pd.DataFrame(rows)
    counts = trades.groupby("date").size()
    trades = trades[trades["date"].isin(set(counts[counts == 3].index))]
    return trades.groupby("date", sort=True).mean(numeric_only=True)


def upside() -> dict:
    from scripts import upside_head_challenger_v1 as module

    path = ROOT / "output/upside_head_challenger_v1_predictions.csv.gz"
    coverage = json.loads(UPSIDE_COVERAGE.read_text(encoding="utf-8"))
    if coverage.get("schema") != "upside_head_challenger_v1":
        raise ValueError("upside coverage: unexpected schema")
    holdout_contract = coverage.get("locked_holdout")
    if not isinstance(holdout_contract, dict):
        raise ValueError("upside coverage: locked_holdout contract is missing")
    selected_variant = holdout_contract.get("selected_variant")
    evaluated_variants = holdout_contract.get("evaluated_model_variants")
    if (
        not isinstance(selected_variant, str)
        or not selected_variant
        or not isinstance(evaluated_variants, list)
        or not evaluated_variants
        or any(not isinstance(item, str) or not item for item in evaluated_variants)
    ):
        raise ValueError("upside coverage: invalid locked variant contract")
    expected_variants = set(evaluated_variants)
    if selected_variant not in expected_variants:
        raise ValueError("upside coverage: selected variant was not evaluated")

    frame = pd.read_csv(path, parse_dates=["date"])
    _require_columns(
        frame,
        {
            "market", "date", "fold", "variant", "evaluation_scope",
            "score_raw", "y_up10", "f_atr_pct_14", "f_qv_rank",
            "b_vol_missing",
        },
        name="upside predictions",
    )
    if frame.empty or frame["date"].isna().any():
        raise ValueError("upside predictions are empty or have invalid dates")
    if frame["date"].max() > _completed_cutoff():
        raise ValueError("upside predictions contain incomplete/future dates")
    _require_unique(
        frame,
        ["evaluation_scope", "variant", "date", "market"],
        name="upside predictions",
    )
    _require_finite(
        frame,
        {
            "fold", "score_raw", "y_up10", "f_atr_pct_14",
            "f_qv_rank", "b_vol_missing",
        },
        name="upside predictions",
    )
    if not frame["y_up10"].isin({0, 1}).all():
        raise ValueError("upside predictions: y_up10 is not binary")
    locked = frame[frame["evaluation_scope"].eq("locked_holdout")].copy()
    if set(locked["variant"]) != expected_variants:
        raise ValueError("upside predictions: unexpected/missing locked variants")
    locked_counts = locked.groupby(["variant", "date"]).size()
    if locked_counts.empty or not locked_counts.eq(100).all():
        raise ValueError("upside predictions: locked full Top100 population is incomplete")
    variants = {
        name: _top3(group, "score_raw")
        for name, group in locked.groupby("variant", sort=True)
    }
    reference = locked[locked["variant"].eq("cls_core")].copy()
    reference["random_score"] = [
        _random_score(date.date(), market)
        for date, market in zip(reference["date"], reference["market"])
    ]
    variants.update(
        {
            "random_seed42": _top3(reference, "random_score"),
            "atr_top3": _top3(reference, "f_atr_pct_14"),
            "liquidity_top3": _top3(reference, "f_qv_rank", ascending=True),
        }
    )
    wanted = pd.concat(
        [part[["market", "date"]] for part in variants.values()],
        ignore_index=True,
    ).drop_duplicates()
    cache, cache_meta = module.build_path_cache(wanted)
    if set(cache) != {
        (str(row.market), row.date)
        for row in wanted.itertuples(index=False)
    }:
        raise ValueError("upside path cache keys do not match requested pairs")
    cache_rows = []
    for (market, date), item in cache.items():
        cache_rows.append(
            {
                "market": market,
                "date": date,
                **item,
            }
        )
    cache_frame = pd.DataFrame(cache_rows)
    _require_unique(cache_frame, ["market", "date"], name="upside path cache")
    cache_meta["path_cache_sha256"] = _hash_frame(
        cache_frame,
        ["date", "market"],
    )
    daily = {name: _upside_daily(picks, cache) for name, picks in variants.items()}
    table = {
        name: {
            "dates": int(len(values)),
            **{
                metric: float(values[metric].mean())
                for metric in ("net", "sl", "tp", "up10", "dn5", "safe")
            },
        }
        for name, values in daily.items()
        if not values.empty
    }
    candidate = daily[selected_variant]
    paired = {}
    comparison_order = [
        *sorted(expected_variants - {selected_variant}),
        "random_seed42",
        "atr_top3",
        "liquidity_top3",
    ]
    for offset, baseline in enumerate(comparison_order, start=1):
        common = candidate.join(
            daily[baseline], how="inner", lsuffix="_candidate", rsuffix="_base"
        )
        paired[baseline] = {
            metric: _cluster_ci(
                common[f"{metric}_candidate"] - common[f"{metric}_base"],
                seed_offset=200 + offset * 10 + index,
            )
            for index, metric in enumerate(("net", "sl", "up10", "dn5", "safe"))
        }
    universe_counts = (
        locked.groupby(["variant", "date"]).size().groupby("variant").agg(["min", "max"])
    )
    aucs = {}
    for variant, group in locked.groupby("variant", sort=True):
        if group["y_up10"].nunique() < 2:
            raise ValueError(f"upside AUC has a single-class target: {variant}")
        aucs[variant] = {
            "rows": int(len(group)),
            "dates": int(group["date"].nunique()),
            "raw_auc": float(roc_auc_score(group["y_up10"], group["score_raw"])),
            "bvol_missing_rate": float(group["b_vol_missing"].mean()),
        }
    return {
        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "coverage_sha256": _sha(UPSIDE_COVERAGE),
        "universe_count_minmax": universe_counts.to_dict(orient="index"),
        "head_metrics": aucs,
        "path_cache": cache_meta,
        "locked_path_table": table,
        "selected_variant": selected_variant,
        "paired_vs_selected_variant": paired,
    }


def main() -> None:
    required = (
        D1_DB,
        M15_DB,
        DOWNSIDE_COVERAGE,
        SAFEUP_COVERAGE,
        UPSIDE_COVERAGE,
        CC_CANDIDATE,
        DOWNSIDE_SOURCE,
        SAFEUP_SOURCE,
        UPSIDE_SOURCE,
        ROOT / "output/downside_veto_challenger_v1_picks.csv",
        ROOT / "output/safeup_head_challenger_v1_picks.csv.gz",
        ROOT / "output/upside_head_challenger_v1_predictions.csv.gz",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing audit inputs: {missing}")
    database_snapshots_before = {
        D1_DB.name: _snapshot(D1_DB),
        M15_DB.name: _snapshot(M15_DB),
    }
    safeup_coverage = json.loads(SAFEUP_COVERAGE.read_text())
    upside_coverage = json.loads(UPSIDE_COVERAGE.read_text())
    safeup_entries = [
        *safeup_coverage["folds"]["discovery"],
        *safeup_coverage["folds"]["locked_holdout"],
    ]
    upside_entries = [
        *upside_coverage["folds"]["discovery"],
        *upside_coverage["folds"]["locked_holdout"],
    ]
    time_order = {
        "safeup": _time_order_from_entries(
            safeup_entries,
            fold_key="fold",
            variant_key="target",
        ),
        "upside": _time_order_from_entries(
            upside_entries,
            fold_key="fold",
            variant_key="variant",
        ),
        "downside": {
            "status": (
                "UNPROVABLE_FROM_PICKS: coverage declares purged WF embargo=5 "
                "but does not persist train/test date ranges"
            ),
        },
    }
    print("RECALC downside", flush=True)
    result = {"downside": downside()}
    print("RECALC safeup", flush=True)
    result["safeup"] = safeup()
    print("RECALC upside paths", flush=True)
    result["upside"] = upside()
    database_snapshots_after = {
        D1_DB.name: _snapshot(D1_DB),
        M15_DB.name: _snapshot(M15_DB),
    }
    if database_snapshots_after != database_snapshots_before:
        raise RuntimeError("a source database changed during recalculation")
    result["provenance"] = {
        "source_database_snapshots": database_snapshots_before,
        "coverage_sha256": {
            "downside": _sha(DOWNSIDE_COVERAGE),
            "safeup": _sha(SAFEUP_COVERAGE),
            "upside": _sha(UPSIDE_COVERAGE),
        },
        "source_sha256": {
            "downside": _sha(DOWNSIDE_SOURCE),
            "safeup": _sha(SAFEUP_SOURCE),
            "upside": _sha(UPSIDE_SOURCE),
            "audit": _sha(Path(__file__)),
        },
        "completed_label_cutoff": _completed_cutoff().strftime("%Y-%m-%d"),
        "bootstrap": {
            "unit": "trading date",
            "seed_base": SEED,
            "draws": DRAWS,
        },
        "time_order": time_order,
        "population_scope": {
            "downside": (
                "all persisted four-policy Top3 rows, full candidate parquet "
                "for PIT-history audit, and locked common complete paths"
            ),
            "safeup": (
                "all persisted discovery/locked picks; locked six-policy "
                "exact Top3 population and reconstructed common path dates"
            ),
            "upside": (
                "all locked variants declared by the current coverage "
                "contract, each with exact Top100 rows per date, plus "
                "deterministic control Top3 selections"
            ),
        },
        "output_contract": (
            "read-only and stdout-only; partial progress output is invalid "
            "unless the final JSON document is emitted and exit is zero"
        ),
    }
    print(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False),
        flush=True,
    )


if __name__ == "__main__":
    main()

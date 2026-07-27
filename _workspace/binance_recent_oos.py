"""Recent fixed-rule OOS audit for the Binance vol-surge detector.

The OOS start is derived from (and strictly follows) the last *label date* in
the persisted walk-forward artifact.  Only completed KST D1 labels and exact
96-bar 15-minute paths are admitted.  Bootstrap uncertainty is clustered by
trading date.  The script writes nothing.
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

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ledger.exit_lab import walk_path  # noqa: E402
from scripts.binance_leadlag_v1 import (  # noqa: E402
    build_binance_features,
    build_upbit_panel,
    krw_to_binance,
)

UPBIT_DB = ROOT / "data/upbit_d1.db"
BINANCE_DB = ROOT / "data/binance_d1.db"
M15_DB = ROOT / "data/upbit_15m.db"
OOF_ARTIFACT = ROOT / "output/binance_leadlag_v1_oof_picks.csv"
BINANCE_SOURCE = ROOT / "scripts/binance_leadlag_v1.py"
EXIT_SOURCE = ROOT / "ledger/exit_lab.py"
COST = 0.0015
EXITS = {
    "tp5_sl3": (0.03, 0.05),
    "tp5_sl2": (0.02, 0.05),
    "tp10_sl5": (0.05, 0.10),
    "eod": (None, None),
}
SEED = 7
DRAWS = 2_000
EXPECTED_BARS = 96


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


def _completed_label_cutoff(now: datetime | None = None) -> pd.Timestamp:
    current = now or datetime.now(ZoneInfo("Asia/Seoul"))
    if current.tzinfo is None:
        raise ValueError("completed-label cutoff requires a timezone-aware clock")
    current_kst_session = (current - timedelta(hours=9)).date()
    return pd.Timestamp(current_kst_session - timedelta(days=1))


def _evaluation_start() -> tuple[pd.Timestamp, pd.Timestamp]:
    artifact = pd.read_csv(
        OOF_ARTIFACT,
        usecols=["market", "timestamp"],
        parse_dates=["timestamp"],
    )
    if artifact.empty or artifact["timestamp"].isna().any():
        raise ValueError("OOF artifact is empty or has invalid timestamps")
    if artifact.duplicated(["market", "timestamp"]).any():
        raise ValueError("OOF artifact has duplicate market/timestamp keys")
    last_oof_label = artifact["timestamp"].max().normalize() + pd.Timedelta(days=1)
    # Starting at the last OOF label would reuse that target day.  The first
    # genuinely unseen label is one day later.
    return last_oof_label + pd.Timedelta(days=1), last_oof_label


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


def regime_series(upbit_db: Path) -> pd.Series:
    """Point-in-time BTC trend/volatility regime mapped from D-1 to label D."""
    with _connect_readonly(upbit_db) as connection:
        btc = pd.read_sql_query(
            """
            SELECT timestamp, close
            FROM candles
            WHERE market='KRW-BTC'
            ORDER BY timestamp
            """,
            connection,
        )
    btc["timestamp"] = pd.to_datetime(btc["timestamp"], errors="raise")
    if btc.empty or btc["timestamp"].duplicated().any():
        raise ValueError("BTC D1 history is empty or duplicated")
    _require_finite(btc, {"close"}, name="BTC D1 history")
    if (btc["close"] <= 0).any():
        raise ValueError("BTC D1 history has non-positive closes")
    btc["ret"] = btc["close"].pct_change(fill_method=None)
    btc["ma50"] = btc["close"].rolling(50, min_periods=50).mean()
    btc["rv20"] = btc["ret"].rolling(20, min_periods=20).std()
    # A full-sample median leaks future volatility regimes into earlier dates.
    # The expanding threshold only uses observations available by feature date.
    btc["rv_threshold_pit"] = (
        btc["rv20"].expanding(min_periods=20).median()
    )
    valid = btc[["ma50", "rv20", "rv_threshold_pit"]].notna().all(axis=1)
    regime = pd.Series("unknown", index=btc.index, dtype="string")
    trend = np.where(btc.loc[valid, "close"] >= btc.loc[valid, "ma50"], "bull", "bear")
    volatility = np.where(
        btc.loc[valid, "rv20"] >= btc.loc[valid, "rv_threshold_pit"],
        "volatile",
        "quiet",
    )
    regime.loc[valid] = (
        pd.Series(trend, index=btc.index[valid])
        + "_"
        + pd.Series(volatility, index=btc.index[valid])
    )
    label_dates = (btc["timestamp"].dt.normalize() + pd.Timedelta(days=1)).dt.date
    result = pd.Series(regime.to_numpy(), index=label_dates)
    if result.index.duplicated().any():
        raise ValueError("BTC regime map has duplicate label dates")
    return result


def _exact_path(
    connection: sqlite3.Connection,
    market: str,
    label_date: pd.Timestamp,
) -> list[tuple[float, float, float, float]] | None:
    start = label_date.normalize() + pd.Timedelta(hours=9)
    end = start + pd.Timedelta(days=1)
    rows = connection.execute(
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
    expected = pd.date_range(start, periods=EXPECTED_BARS, freq="15min")
    timestamps = [pd.Timestamp(row[0]) for row in rows]
    if len(rows) != EXPECTED_BARS or timestamps != list(expected):
        return None
    bars = [tuple(map(float, row[1:])) for row in rows]
    values = np.asarray(bars, dtype=float)
    if (
        not np.isfinite(values).all()
        or (values <= 0).any()
        or (values[:, 1] < np.maximum(values[:, 0], values[:, 3])).any()
        or (values[:, 2] > np.minimum(values[:, 0], values[:, 3])).any()
        or (values[:, 1] < values[:, 2]).any()
    ):
        raise ValueError(f"invalid 15m OHLC path: {market} {label_date.date()}")
    return bars


def _simulate(
    rows: pd.DataFrame,
    *,
    tag: str,
    connection: sqlite3.Connection,
) -> tuple[list[dict], dict]:
    records: list[dict] = []
    missing_paths = 0
    for row in rows.itertuples(index=False):
        bars = _exact_path(connection, str(row.market), pd.Timestamp(row.label_date))
        if bars is None:
            missing_paths += 1
            continue
        for exit_name, (stop_loss, take_profit) in EXITS.items():
            gross, _ = walk_path(bars, stop_loss, take_profit)
            if not np.isfinite(gross):
                raise ValueError(
                    f"non-finite path result: {row.market} {row.label_date}"
                )
            records.append(
                {
                    "market": str(row.market),
                    "label_date": pd.Timestamp(row.label_date),
                    "regime": str(row.regime),
                    "exit": exit_name,
                    "net": float(gross - COST),
                    "pump20": bool(float(row.pump_max_return) >= 0.20),
                }
            )
    result = pd.DataFrame(records)
    print(
        f"\n=== {tag}: picks={len(rows)}, exact_paths="
        f"{0 if result.empty else result[['market', 'label_date']].drop_duplicates().shape[0]}, "
        f"incomplete_paths={missing_paths} ===",
        flush=True,
    )
    if result.empty:
        raise ValueError(f"{tag}: no complete 15m paths")
    pair_counts = result.groupby(["market", "label_date"])["exit"].nunique()
    if not pair_counts.eq(len(EXITS)).all():
        raise ValueError(f"{tag}: exit policies do not share an exact path cohort")
    for offset, (exit_name, group) in enumerate(result.groupby("exit", sort=True), start=1):
        daily = group.groupby("label_date", sort=True)["net"].mean()
        values = daily.to_numpy(float)
        rng = np.random.default_rng(SEED + offset)
        indices = rng.integers(0, len(values), size=(DRAWS, len(values)))
        bootstrap = values[indices].mean(axis=1)
        low, high = np.quantile(bootstrap, [0.025, 0.975])
        print(
            f"  {exit_name:9s} net={values.mean() * 100:+.3f}% "
            f"date-CI95=[{low * 100:+.3f},{high * 100:+.3f}] "
            f"trade-win={100 * group['net'].gt(0).mean():.1f}% "
            f"dates={len(values)}",
            flush=True,
        )
    primary = result[result["exit"].eq("tp5_sl3")]
    print("  --- regime split (tp5_sl3; exact-path cohort) ---", flush=True)
    for regime, group in primary.groupby("regime", sort=True):
        print(
            f"  {regime:14s} n={len(group):4d} "
            f"net={group['net'].mean() * 100:+.3f}%",
            flush=True,
        )
    print(
        f"  pump20 hit exact-path cohort: {int(primary['pump20'].sum())}/"
        f"{len(primary)} ({primary['pump20'].mean() * 100:.1f}%)",
        flush=True,
    )
    canonical = (
        result.sort_values(["label_date", "market", "exit"])
        .to_json(orient="records", date_format="iso", double_precision=15)
        .encode()
    )
    return records, {
        "selected_rows": int(len(rows)),
        "complete_pairs": int(len(primary)),
        "incomplete_pairs": int(missing_paths),
        "path_result_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def main() -> None:
    required_paths = (
        UPBIT_DB,
        BINANCE_DB,
        M15_DB,
        OOF_ARTIFACT,
        BINANCE_SOURCE,
        EXIT_SOURCE,
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing audit inputs: {missing}")
    snapshots_before = {path.name: _snapshot(path) for path in required_paths[:3]}
    evaluation_start, last_oof_label = _evaluation_start()
    cutoff = _completed_label_cutoff()
    if evaluation_start > cutoff:
        raise ValueError("no completed labels strictly after the OOF artifact")

    print("[1] upbit panel ...", flush=True)
    up = build_upbit_panel(str(UPBIT_DB))
    required_up = {
        "market", "feature_date", "quote_volume_d1", "u_roc_7d",
        "pump_max_return",
    }
    missing_up = sorted(required_up.difference(up.columns))
    if missing_up or up.empty:
        raise ValueError(f"Upbit panel missing/empty: {missing_up}")
    up["feature_date"] = pd.to_datetime(up["feature_date"], errors="raise").dt.normalize()
    if up.duplicated(["market", "feature_date"]).any():
        raise ValueError("Upbit panel has duplicate market/feature_date rows")
    _require_finite(
        up,
        {"quote_volume_d1", "u_roc_7d", "pump_max_return"},
        name="Upbit panel",
    )
    up["label_date"] = up["feature_date"] + pd.Timedelta(days=1)
    up = up[
        up["label_date"].between(evaluation_start, cutoff, inclusive="both")
    ].copy()
    if up.empty:
        raise ValueError("Upbit panel has no completed strict-OOS rows")
    up = up.sort_values(
        ["feature_date", "quote_volume_d1", "market"],
        ascending=[True, False, True],
    )
    up["qv_rank"] = up.groupby("feature_date", sort=False).cumcount() + 1
    up = up[up["qv_rank"] <= 120].copy()
    if up.groupby("feature_date").size().max() > 120:
        raise ValueError("Upbit universe exceeds exact Top120")
    up["roc_7d_rank"] = up.groupby("feature_date")["u_roc_7d"].rank(pct=True)

    print("[2] binance features ...", flush=True)
    up["bn_market"] = up["market"].map(krw_to_binance)
    binance = build_binance_features(
        str(BINANCE_DB),
        set(up["bn_market"].dropna()),
    )
    if binance.empty:
        raise ValueError("Binance feature panel is empty")
    binance["feature_date"] = pd.to_datetime(
        binance["feature_date"],
        errors="raise",
    ).dt.normalize()
    if binance.duplicated(["bn_market", "feature_date"]).any():
        raise ValueError("Binance feature panel has duplicate keys")
    joined = up.merge(
        binance,
        on=["bn_market", "feature_date"],
        how="left",
        validate="many_to_one",
    )
    if joined["label_date"].min() < evaluation_start:
        raise ValueError("strict OOS cutoff violation")
    if joined["label_date"].max() > cutoff:
        raise ValueError("incomplete/future label admitted")
    available = np.isfinite(
        pd.to_numeric(joined["b_vol_surge"], errors="coerce").to_numpy(float)
    )
    print(
        f"    strict OOS {evaluation_start.date()}..{cutoff.date()} "
        f"(last OOF label={last_oof_label.date()}) rows={len(joined)}",
        flush=True,
    )
    print(f"    Binance join coverage={available.mean() * 100:.1f}%", flush=True)
    # Both policies use the same Binance-observable population.
    baseline_mask = joined["roc_7d_rank"].gt(0.85) & available
    rule_mask = baseline_mask & joined["b_vol_surge"].gt(1.5)
    regimes = regime_series(UPBIT_DB)
    joined["regime"] = joined["label_date"].dt.date.map(regimes)
    if joined.loc[baseline_mask, "regime"].isna().any():
        raise ValueError("missing BTC regime on baseline rows")

    with _connect_readonly(M15_DB) as connection:
        _, rule_meta = _simulate(
            joined[rule_mask],
            tag="base_AND_bn_volsurge (strict post-OOF)",
            connection=connection,
        )
        _, baseline_meta = _simulate(
            joined[baseline_mask],
            tag="baseline_roc7 (same Binance-available cohort)",
            connection=connection,
        )
    snapshots_after = {path.name: _snapshot(path) for path in required_paths[:3]}
    if snapshots_after != snapshots_before:
        raise RuntimeError("a source database changed during recalculation")
    provenance = {
        "input_sha256": {
            "oof_artifact": _sha(OOF_ARTIFACT),
            "binance_feature_source": _sha(BINANCE_SOURCE),
            "exit_path_source": _sha(EXIT_SOURCE),
            "audit_source": _sha(Path(__file__)),
        },
        "source_database_snapshots": snapshots_before,
        "last_oof_label_date": last_oof_label.strftime("%Y-%m-%d"),
        "strict_oos_start": evaluation_start.strftime("%Y-%m-%d"),
        "completed_label_cutoff": cutoff.strftime("%Y-%m-%d"),
        "cost_round_trip_once": COST,
        "bootstrap": {
            "unit": "trading date",
            "seed_base": SEED,
            "draws": DRAWS,
        },
        "path_contract": "exact [D 09:00,D+1 09:00) 96x15m bars",
        "path_cache": {
            "rule": rule_meta,
            "baseline": baseline_meta,
        },
        "population_scope": {
            "all_strict_oos_top120_rows": int(len(joined)),
            "binance_observable_rows": int(available.sum()),
            "baseline_fire_rows": int(baseline_mask.sum()),
            "rule_fire_rows": int(rule_mask.sum()),
            "comparison_contract": (
                "both rules originate from the same exact PIT Top120 and "
                "Binance-observable population"
            ),
        },
        "output_contract": (
            "read-only and stdout-only; partial progress output is invalid "
            "unless the final provenance document is emitted and exit is zero"
        ),
    }
    print("\n=== PROVENANCE / CONTRACTS ===")
    print(
        json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False),
        flush=True,
    )


if __name__ == "__main__":
    main()

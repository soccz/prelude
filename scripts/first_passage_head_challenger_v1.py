"""Direct 09:15 first-passage head challenger (research only).

The production alert, registry, Telegram formatter, and ledgers are not read
for mutation and are never written by this module.

Primary target
--------------
For the executable 96-bar window ``[D 09:15, D+1 09:15)``, predict from D-1
features whether +10% is reached before -5%.  Same-bar ambiguity is resolved
downside first.

Secondary target
----------------
An ATR-normalized first-passage label uses fixed, untuned barriers:
``up = 2 * D-1 ATR14`` and ``down = 1 * D-1 ATR14``.  No holdout statistic,
quantile, or fitted multiplier defines these barriers.

Both heads use the same fixed XGBoost specification, outer expanding
walk-forward with five-date embargo, and true inner expanding OOF isotonic
calibration.  Model fitting is single-threaded for byte-stable repeatability.
There is no hyperparameter sweep.

The final 180 dates on one shared benchmark-complete eligibility axis define
the SafeUp/R1, first-passage, and semivol historical comparison schedule.
Every row is joined by schedule hash, scope, fold, date, and market.  Discovery
chooses at most one of the two heads; only that head is fitted and evaluated
on the shared historical dates.  Re-aligning the schedule restores comparison
fairness but cannot make this already observed period virgin/preregistered
evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.safeup_head_challenger_v1 as safeup  # noqa: E402
from ledger.path_quality import EXPECTED_BARS  # noqa: E402
from ops.code_lineage import python_code_lineage  # noqa: E402
from scripts.recommender_downside_exit_v1 import simulate_path  # noqa: E402


D1_DB = ROOT / "data" / "upbit_d1.db"
M15_DB = ROOT / "data" / "upbit_15m.db"
BASELINE_PREDICTIONS = Path(
    f"{safeup.FP_SCHEDULE_PREFIX}_predictions.csv.gz"
)
OUT_PREFIX = ROOT / "output" / "first_passage_head_challenger_v1"

TOP_K = 3
MIN_PRIOR_HISTORY = 70
UNIVERSE_TOP_N = 100
OUTER_FOLDS = 5
INNER_FOLDS = 3
EMBARGO_DATES = 5
LOCKED_COMMON_DATES = 180
ROUND_TRIP_COST = 0.0015
TP5 = 0.05
SL3 = 0.03
MODEL_SEED = 42
BOOTSTRAP_DRAWS = 5_000
PATH_CACHE_SCHEMA = "first_passage_head_challenger_v1_path_panel_v3"
GZIP_COMPRESSION = {"method": "gzip", "mtime": 0}

FIXED_TARGET = "label_fp_safe10"
ATR_TARGET = "label_fp_atr"
HEADS = {
    "fp_fixed_head": FIXED_TARGET,
    "fp_atr_head": ATR_TARGET,
}
PRIMARY_HEAD = "fp_fixed_head"

FINAL_BASELINES = (
    "R1_repaired",
    "safeup_head",
    "monkey_seed42",
    "ATR_top3",
    "liquidity_matched",
)

BENCHMARK_FAILURES = {
    "benchmark_missing",
    "db_horizon_start_incomplete",
    "db_horizon_end_incomplete",
    "benchmark_off_grid",
    "benchmark_gap",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _code_lineage() -> dict:
    return python_code_lineage(entrypoint=Path(__file__), root=ROOT)


def _file_signature(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    signature = {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _sha256(path),
    }
    wal_path = Path(f"{path}-wal")
    if wal_path.is_file():
        wal_stat = wal_path.stat()
        signature["wal"] = {
            "size": int(wal_stat.st_size),
            "mtime_ns": int(wal_stat.st_mtime_ns),
            "sha256": _sha256(wal_path),
        }
    else:
        signature["wal"] = None
    return signature


def _connect_readonly(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(path)
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


def _path_cache_signature(
    *,
    d1_db: Path,
    m15_db: Path,
    limit_markets: int | None,
) -> dict:
    semantic_source = "\n".join(
        [
            inspect.getsource(_first_passage),
            inspect.getsource(_path_labels),
            inspect.getsource(safeup._bulk_execution_paths),
        ]
    )
    return {
        "schema": PATH_CACHE_SCHEMA,
        "d1_db": _file_signature(d1_db),
        "m15_db": _file_signature(m15_db),
        "safeup_script_sha256": _sha256(Path(safeup.__file__)),
        "code_lineage": _code_lineage(),
        "path_semantics_sha256": hashlib.sha256(
            semantic_source.encode("utf-8")
        ).hexdigest(),
        "completed_label_cutoff": str(
            safeup._completed_label_cutoff().date()
        ),
        "limit_markets": limit_markets,
        "minimum_prior_history": MIN_PRIOR_HISTORY,
        "universe_top_n": UNIVERSE_TOP_N,
        "expected_15m_bars": EXPECTED_BARS,
    }


def _stable_hash(date: object, market: str, salt: str) -> int:
    payload = f"{MODEL_SEED}|{salt}|{date}|{market}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _safe_auc(labels: pd.Series, scores: pd.Series) -> float:
    frame = pd.DataFrame({"y": labels, "s": scores}).dropna()
    if len(frame) == 0 or frame["y"].nunique() < 2:
        return np.nan
    return float(roc_auc_score(frame["y"], frame["s"]))


def _first_passage(
    bars: tuple[tuple[float, float, float, float], ...],
    *,
    up_barrier: float,
    down_barrier: float,
) -> tuple[int, str]:
    """Return 1 only when the upper barrier is observed first."""
    if not bars or up_barrier <= 0 or down_barrier <= 0:
        return 0, "invalid"
    entry = float(bars[0][0])
    upper = entry * (1.0 + up_barrier)
    lower = entry * (1.0 - down_barrier)
    for _, high, low, _ in bars:
        hit_down = float(low) <= lower
        hit_up = float(high) >= upper
        if hit_down:
            return 0, "down_first_same_bar" if hit_up else "down_first"
        if hit_up:
            return 1, "up_first"
    return 0, "neither"


def _path_labels(
    path: safeup.BulkPath,
    atr_pct: float,
) -> dict:
    base = {
        "path_complete": bool(path.complete),
        "path_quality": path.quality,
        "path_raw_bars": int(path.raw_bars),
        "path_flat_filled_bars": int(path.flat_filled_bars),
        "path_benchmark_bars": int(path.benchmark_bars),
        "benchmark_complete": bool(
            path.benchmark_bars == EXPECTED_BARS
            and path.quality not in BENCHMARK_FAILURES
        ),
    }
    if not path.complete:
        return base

    bars = path.bars
    fixed, fixed_outcome = _first_passage(
        bars,
        up_barrier=0.10,
        down_barrier=0.05,
    )
    atr = float(atr_pct)
    if not np.isfinite(atr) or atr <= 0:
        atr_label, atr_outcome = np.nan, "invalid_atr"
        atr_up, atr_down = np.nan, np.nan
    else:
        atr_up = 2.0 * atr
        atr_down = 1.0 * atr
        atr_label, atr_outcome = _first_passage(
            bars,
            up_barrier=atr_up,
            down_barrier=atr_down,
        )

    bracket_gross, bracket_outcome = simulate_path(
        list(bars), SL3, TP5, None
    )
    eod_gross, _ = simulate_path(list(bars), None, None, None)
    entry = float(bars[0][0])
    mfe = max(float(bar[1]) for bar in bars) / entry - 1.0
    mae = min(float(bar[2]) for bar in bars) / entry - 1.0
    return {
        **base,
        FIXED_TARGET: int(fixed),
        "fp_fixed_outcome": fixed_outcome,
        ATR_TARGET: atr_label,
        "fp_atr_outcome": atr_outcome,
        "atr_up_barrier": atr_up,
        "atr_down_barrier": atr_down,
        "path_up10": int(mfe >= 0.10),
        "path_dn5": int(mae <= -0.05),
        "path_mfe": float(mfe),
        "path_mae": float(mae),
        "path_mfe20": int(mfe >= 0.20),
        "path_bracket_outcome": bracket_outcome,
        "path_bracket_net": float(
            bracket_gross - ROUND_TRIP_COST
        ),
        "path_eod_net": float(eod_gross - ROUND_TRIP_COST),
    }


def build_labeled_panel(
    *,
    d1_db: Path,
    m15_db: Path,
    limit_markets: int | None,
) -> tuple[pd.DataFrame, dict]:
    """Build PIT Top100 D-1 features and exact post-alert path labels."""
    panel, panel_meta = safeup.prepare_panel(
        limit_markets,
        d1_db=d1_db,
    )
    counts = panel.groupby("date")["market"].transform("size")
    panel = panel[counts == UNIVERSE_TOP_N].copy()

    with _connect_readonly(m15_db) as connection:
        first_nonbenchmark = connection.execute(
            """
            SELECT MIN(timestamp) FROM candles
            WHERE market != ?
            """,
            (safeup.BENCHMARK_MARKET,),
        ).fetchone()[0]
    if first_nonbenchmark is None:
        raise RuntimeError("15m DB has no non-benchmark history")
    first_path_date = pd.Timestamp(first_nonbenchmark).date()
    panel = panel[panel["date"] >= first_path_date].copy()
    if panel.empty:
        raise RuntimeError("no Top100 dates overlap 15m history")

    paths, canonical_checked, canonical_complete_checked = (
        safeup._bulk_execution_paths(
            panel[["market", "date"]],
            m15_db,
        )
    )
    labels = []
    for row in panel[["market", "date", "f_atr_pct_14"]].itertuples(
        index=False
    ):
        values = {"market": str(row.market), "date": row.date}
        values.update(
            _path_labels(
                paths[(str(row.market), row.date)],
                float(row.f_atr_pct_14),
            )
        )
        labels.append(values)
    labeled = panel.merge(
        pd.DataFrame(labels),
        on=["market", "date"],
        how="left",
        validate="one_to_one",
    )
    labeled["vol_band"] = np.minimum(
        np.floor(
            labeled["f_atr_xs_decile"]
            .fillna(0.5)
            .clip(0, 1)
            * 5
        ),
        4,
    ).astype(int)

    date_state = labeled.groupby("date").agg(
        candidates=("market", "size"),
        benchmark_complete=("benchmark_complete", "all"),
        target_complete=("path_complete", "sum"),
    )
    if not (date_state["candidates"] == UNIVERSE_TOP_N).all():
        raise RuntimeError("labeled panel lost static Top100 contract")
    benchmark_dates = list(
        date_state[date_state["benchmark_complete"]].index
    )
    if len(benchmark_dates) <= LOCKED_COMMON_DATES + 60:
        raise RuntimeError(
            "not enough benchmark-complete dates for discovery and holdout"
        )
    meta = {
        "panel": panel_meta,
        "first_nonbenchmark_15m_date": str(first_path_date),
        "path_rows": int(len(labeled)),
        "path_dates": int(labeled["date"].nunique()),
        "benchmark_complete_dates": int(len(benchmark_dates)),
        "target_complete_rows": int(labeled["path_complete"].sum()),
        "target_incomplete_rows": int((~labeled["path_complete"]).sum()),
        "exact_100_target_complete_dates": int(
            (
                date_state["benchmark_complete"]
                & (date_state["target_complete"] == UNIVERSE_TOP_N)
            ).sum()
        ),
        "canonical_crosscheck_n": int(canonical_checked),
        "canonical_complete_crosscheck_n": int(
            canonical_complete_checked
        ),
        "path_quality_counts": {
            str(key): int(value)
            for key, value in labeled["path_quality"]
            .value_counts(dropna=False)
            .items()
        },
    }
    return labeled.sort_values(["date", "market"]).reset_index(
        drop=True
    ), meta


def _parse_bool_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        raise RuntimeError(f"path panel is missing {column}")
    values = frame[column]
    if values.dtype == bool:
        return values
    parsed = (
        values.astype("string")
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False})
    )
    if parsed.isna().any():
        raise RuntimeError(f"path panel has invalid {column} values")
    return parsed.astype(bool)


def _validate_labeled_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Fail closed before cached labels or features enter any model."""
    required = {
        "market",
        "date",
        "history_prior_bars",
        "f_qv_rank",
        "path_complete",
        "benchmark_complete",
        "path_quality",
        "path_raw_bars",
        "path_flat_filled_bars",
        "path_benchmark_bars",
        FIXED_TARGET,
        ATR_TARGET,
        "path_up10",
        "path_dn5",
        "path_bracket_net",
        "path_eod_net",
    }
    missing = sorted(required - set(panel.columns))
    if missing:
        raise RuntimeError(f"path panel columns missing: {missing}")
    if panel.empty:
        raise RuntimeError("path panel is empty")
    out = panel.copy()
    parsed_dates = pd.to_datetime(out["date"], errors="raise")
    if parsed_dates.isna().any():
        raise RuntimeError("path panel has missing dates")
    out["date"] = parsed_dates.dt.date
    if max(out["date"]) > safeup._completed_label_cutoff().date():
        raise RuntimeError("path panel extends beyond completed-label cutoff")
    if not out["market"].astype("string").str.fullmatch(
        r"KRW-[A-Z0-9]+", na=False
    ).all():
        raise RuntimeError("path panel has invalid/non-canonical markets")
    if out.duplicated(["date", "market"]).any():
        raise RuntimeError("path panel has duplicate market/date rows")
    out["path_complete"] = _parse_bool_column(out, "path_complete")
    out["benchmark_complete"] = _parse_bool_column(
        out, "benchmark_complete"
    )
    counts = out.groupby("date").size()
    if not (counts == UNIVERSE_TOP_N).all():
        raise RuntimeError("path panel is not exact Top100")
    ranks = pd.to_numeric(out["f_qv_rank"], errors="coerce")
    expected_ranks = tuple(range(1, UNIVERSE_TOP_N + 1))
    rank_sets = ranks.groupby(out["date"]).agg(
        lambda values: tuple(sorted(values.astype(int)))
        if values.notna().all()
        and np.equal(values, np.floor(values)).all()
        else ()
    )
    if not rank_sets.map(lambda values: values == expected_ranks).all():
        raise RuntimeError("path panel ranks are not exact 1..100 per date")
    history = pd.to_numeric(out["history_prior_bars"], errors="coerce")
    if (
        history.isna().any()
        or not np.isfinite(history).all()
        or int(history.min()) < MIN_PRIOR_HISTORY
    ):
        raise RuntimeError("path panel violates prior-history contract")
    benchmark_nunique = out.groupby("date")[
        "benchmark_complete"
    ].nunique()
    if not (benchmark_nunique == 1).all():
        raise RuntimeError("benchmark completeness differs within a date")
    complete = out["path_complete"]
    if (complete & ~out["benchmark_complete"]).any():
        raise RuntimeError("complete target path has incomplete benchmark")
    if not out.loc[complete, "path_quality"].isin(
        ["complete", "flat_filled"]
    ).all():
        raise RuntimeError("complete path rows have invalid quality")
    complete_numeric = [
        "path_raw_bars",
        "path_flat_filled_bars",
        "path_benchmark_bars",
        FIXED_TARGET,
        "path_up10",
        "path_dn5",
        "path_bracket_net",
        "path_eod_net",
    ]
    numeric = out.loc[complete, complete_numeric].apply(
        pd.to_numeric, errors="coerce"
    )
    if (
        numeric.isna().any().any()
        or not np.isfinite(numeric.to_numpy()).all()
    ):
        raise RuntimeError("complete path rows contain nonfinite outcomes")
    for column in (FIXED_TARGET, "path_up10", "path_dn5"):
        if not numeric[column].isin([0, 1]).all():
            raise RuntimeError(f"path panel has non-binary {column}")
    raw_bars = numeric["path_raw_bars"]
    flat_bars = numeric["path_flat_filled_bars"]
    if (
        (raw_bars <= 0).any()
        or (raw_bars > EXPECTED_BARS).any()
        or (flat_bars < 0).any()
        or ((raw_bars + flat_bars) != EXPECTED_BARS).any()
        or (numeric["path_benchmark_bars"] != EXPECTED_BARS).any()
    ):
        raise RuntimeError("complete path rows violate exact 96-bar contract")
    incomplete = ~complete
    if incomplete.any() and out.loc[
        incomplete, [FIXED_TARGET, ATR_TARGET]
    ].notna().any().any():
        raise RuntimeError("incomplete path rows contain model labels")
    return out.sort_values(["date", "market"]).reset_index(drop=True)


def load_or_build_labeled_panel(
    *,
    d1_db: Path,
    m15_db: Path,
    limit_markets: int | None,
    output_prefix: Path,
    rebuild: bool,
) -> tuple[pd.DataFrame, dict]:
    """Reuse only a source- and code-matched path panel cache."""
    cache_path = Path(f"{output_prefix}_path_panel.csv.gz")
    cache_meta_path = Path(f"{output_prefix}_path_panel_meta.json")
    signature = _path_cache_signature(
        d1_db=d1_db,
        m15_db=m15_db,
        limit_markets=limit_markets,
    )
    cache_hit = False
    if not rebuild and cache_path.exists() and cache_meta_path.exists():
        cached_meta = json.loads(
            cache_meta_path.read_text(encoding="utf-8")
        )
        cached_signature = dict(cached_meta.get("signature", {}))
        # The original cache metadata included the whole research script
        # hash.  Model-only edits must not invalidate a 15m path cache;
        # path-label semantic edits instead bump PATH_CACHE_SCHEMA.
        cached_signature.pop("script_sha256", None)
        cache_sha256 = cached_meta.get("cache_sha256")
        if (
            cached_signature == signature
            and isinstance(cache_sha256, str)
            and cache_sha256 == _sha256(cache_path)
        ):
            panel = pd.read_csv(
                cache_path,
                float_precision="round_trip",
            )
            panel = _validate_labeled_panel(panel)
            path_meta = cached_meta.get("path_meta")
            if not isinstance(path_meta, dict):
                raise RuntimeError("path cache metadata is missing path_meta")
            cache_hit = True
    if not cache_hit:
        panel, path_meta = build_labeled_panel(
            d1_db=d1_db,
            m15_db=m15_db,
            limit_markets=limit_markets,
        )
        panel = _validate_labeled_panel(panel)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=cache_path.parent,
            prefix=f".{cache_path.name}.",
            suffix=".csv.gz",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
        panel.to_csv(
            temporary_path,
            index=False,
            compression=GZIP_COMPRESSION,
            float_format="%.17g",
        )
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        signature_after = _path_cache_signature(
            d1_db=d1_db,
            m15_db=m15_db,
            limit_markets=limit_markets,
        )
        if signature_after != signature:
            temporary_path.unlink(missing_ok=True)
            raise RuntimeError("path-cache source changed during build")
        temporary_path.replace(cache_path)
        _write_json(
            cache_meta_path,
            {
                "signature": signature,
                "path_meta": path_meta,
                "cache_sha256": _sha256(cache_path),
            },
        )
    print(
        f"path_panel_cache={'hit' if cache_hit else 'built'} "
        f"rows={len(panel)} dates={panel['date'].nunique()}"
    )
    return panel, path_meta


def _history_contract(frame: pd.DataFrame, context: str) -> dict:
    if "history_prior_bars" not in frame:
        raise RuntimeError(f"{context}: history_prior_bars is missing")
    values = pd.to_numeric(
        frame["history_prior_bars"],
        errors="coerce",
    )
    missing = int(values.isna().sum())
    below = int((values < MIN_PRIOR_HISTORY).fillna(False).sum())
    violations = missing + below
    result = {
        "context": context,
        "rows": int(len(frame)),
        "minimum_required": MIN_PRIOR_HISTORY,
        "observed_min": (
            int(values.min()) if values.notna().any() else None
        ),
        "missing_count": missing,
        "below_minimum_count": below,
        "violation_count": violations,
    }
    if violations:
        raise RuntimeError(f"history contract failed: {result}")
    return result


def _date_coverage(frame: pd.DataFrame) -> dict:
    dates = sorted(frame["date"].dropna().unique())
    return {
        "n": int(len(dates)),
        "start": str(dates[0]) if dates else None,
        "end": str(dates[-1]) if dates else None,
        "dates_sha256": hashlib.sha256(
            "\n".join(map(str, dates)).encode("utf-8")
        ).hexdigest(),
    }


def barrier_diagnostics(panel: pd.DataFrame, dates: Iterable) -> dict:
    scoped = panel[
        panel["date"].isin(set(dates)) & panel["path_complete"]
    ].copy()
    up = pd.to_numeric(scoped["atr_up_barrier"], errors="coerce")
    down = pd.to_numeric(scoped["atr_down_barrier"], errors="coerce")
    valid = (
        np.isfinite(up)
        & np.isfinite(down)
        & (up > 0)
        & (down > 0)
    )

    def describe(values: pd.Series) -> dict:
        clean = values[np.isfinite(values)]
        return {
            "min": float(clean.min()) if len(clean) else None,
            "q01": float(clean.quantile(0.01)) if len(clean) else None,
            "median": float(clean.median()) if len(clean) else None,
            "q99": float(clean.quantile(0.99)) if len(clean) else None,
            "max": float(clean.max()) if len(clean) else None,
        }

    counts = {
        "nonfinite_or_nonpositive": int((~valid).sum()),
        "up_below_1pct": int((valid & (up < 0.01)).sum()),
        "up_above_50pct": int((valid & (up > 0.50)).sum()),
        "down_below_0_5pct": int((valid & (down < 0.005)).sum()),
        "down_above_25pct": int((valid & (down > 0.25)).sum()),
        "down_at_or_above_100pct": int(
            (valid & (down >= 1.0)).sum()
        ),
    }
    return {
        "rows": int(len(scoped)),
        "dates": int(scoped["date"].nunique()),
        "up_barrier": describe(up),
        "down_barrier": describe(down),
        "abnormal_or_extreme_counts": counts,
        "abnormal_or_extreme_rates": {
            key: (float(value / len(scoped)) if len(scoped) else None)
            for key, value in counts.items()
        },
        "caps_applied": False,
    }


def _expanding_splits(
    dates: Iterable,
    *,
    n_folds: int,
    minimum_warmup: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    return safeup._expanding_splits(
        dates,
        n_folds=n_folds,
        embargo=EMBARGO_DATES,
        warmup_fraction=0.35 if n_folds == OUTER_FOLDS else 0.45,
        minimum_warmup=minimum_warmup,
    )


def _fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    target: str,
) -> tuple[np.ndarray, None]:
    """Fixed single-thread XGBoost; equivalent to the baseline helper."""
    import xgboost as xgb

    train_ordered = train.sort_values(["date", "market"]).copy()
    x_train, x_test = safeup._matrix(
        train_ordered,
        test,
        features,
    )
    y_train = train_ordered[target].to_numpy(dtype=int)
    positives = int(y_train.sum())
    if positives < 20 or len(np.unique(y_train)) < 2:
        raise RuntimeError(
            f"insufficient target support: target={target} "
            f"positives={positives}"
        )
    model = xgb.XGBClassifier(
        n_estimators=180,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.5,
        scale_pos_weight=float(
            (len(y_train) - positives) / positives
        ),
        n_jobs=1,
        tree_method="hist",
        eval_metric="logloss",
        random_state=MODEL_SEED,
    )
    model.fit(x_train, y_train, verbose=False)
    raw = model.predict_proba(x_test)[:, 1]
    if not np.isfinite(raw).all():
        raise RuntimeError(f"model produced nonfinite scores for {target}")
    return np.asarray(raw, dtype=float), None


def _inner_oof(
    train: pd.DataFrame,
    features: list[str],
    target: str,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    scores = []
    labels = []
    metadata = []
    splits = _expanding_splits(
        train["date"],
        n_folds=INNER_FOLDS,
        minimum_warmup=60,
    )
    if len(splits) != INNER_FOLDS:
        raise RuntimeError(
            f"expected {INNER_FOLDS} inner folds, got {len(splits)}"
        )
    for fold, (train_dates, validation_dates) in enumerate(splits):
        inner_train = train[
            train["date"].isin(set(train_dates))
            & train[target].notna()
            & train["path_complete"]
        ].copy()
        validation = train[
            train["date"].isin(set(validation_dates))
            & train[target].notna()
            & train["path_complete"]
        ].copy()
        record = {
            "fold": int(fold),
            "train_start": str(inner_train["date"].min()),
            "train_end": str(inner_train["date"].max()),
            "validation_start": str(validation["date"].min()),
            "validation_end": str(validation["date"].max()),
            "train_rows": int(len(inner_train)),
            "validation_rows": int(len(validation)),
            "used_for_calibration": False,
        }
        if (
            len(inner_train) < 1_000
            or len(validation) < 100
            or inner_train[target].sum() < 20
        ):
            record["skip_reason"] = "insufficient_rows_or_positive_support"
            metadata.append(record)
            continue
        raw, _ = _fit_predict(
            inner_train,
            validation,
            features,
            target,
        )
        scores.append(raw)
        labels.append(validation[target].to_numpy(dtype=int))
        record["used_for_calibration"] = True
        record["skip_reason"] = None
        metadata.append(record)
    if len(scores) != len(splits):
        return np.array([]), np.array([], dtype=int), metadata
    return np.concatenate(scores), np.concatenate(labels), metadata


def _predict_target(
    train_all: pd.DataFrame,
    test_all: pd.DataFrame,
    features: list[str],
    target: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    train = train_all[
        train_all["path_complete"]
        & train_all[target].notna()
    ].copy()
    if len(train) < 1_000 or train[target].sum() < 20:
        raise RuntimeError(
            f"insufficient path labels for {target}: "
            f"n={len(train)} pos={train[target].sum()}"
        )
    inner_raw, inner_y, inner_meta = _inner_oof(
        train,
        features,
        target,
    )
    calibrator = safeup._fit_isotonic(inner_raw, inner_y)
    raw_test, _ = _fit_predict(
        train,
        test_all,
        features,
        target,
    )
    base_rate = float(train[target].mean())
    probability = safeup._apply_isotonic(
        calibrator,
        raw_test,
        base_rate,
    )
    order = np.argsort(raw_test)
    monotonic = bool(
        np.all(np.diff(probability[order]) >= -1e-12)
    )
    if not monotonic:
        raise RuntimeError("isotonic output is not monotone")
    if (
        not np.isfinite(probability).all()
        or (probability < 0).any()
        or (probability > 1).any()
    ):
        raise RuntimeError("calibrator produced invalid probabilities")
    return raw_test, probability, {
        "target": target,
        "train_rows": int(len(train)),
        "train_dates": int(train["date"].nunique()),
        "train_start": str(train["date"].min()),
        "train_end": str(train["date"].max()),
        "test_rows": int(len(test_all)),
        "test_dates": int(test_all["date"].nunique()),
        "test_start": str(test_all["date"].min()),
        "test_end": str(test_all["date"].max()),
        "train_positive_rate": base_rate,
        "inner_oof_rows": int(len(inner_raw)),
        "inner_oof_positive_rate": (
            float(inner_y.mean()) if len(inner_y) else np.nan
        ),
        "inner_folds": inner_meta,
        "inner_all_folds_used": bool(
            inner_meta
            and all(
                item.get("used_for_calibration", False)
                for item in inner_meta
            )
        ),
        "isotonic_fitted": calibrator is not None,
        "isotonic_monotone": monotonic,
    }


def _base_columns(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "date",
        "market",
        "regime",
        "history_prior_bars",
        "f_qv_rank",
        "f_atr_pct_14",
        "f_atr_xs_decile",
        "vol_band",
        "path_complete",
        "path_quality",
        "path_flat_filled_bars",
        FIXED_TARGET,
        "fp_fixed_outcome",
        ATR_TARGET,
        "fp_atr_outcome",
        "atr_up_barrier",
        "atr_down_barrier",
        "path_up10",
        "path_dn5",
        "path_mfe",
        "path_mae",
        "path_mfe20",
        "path_bracket_outcome",
        "path_bracket_net",
        "path_eod_net",
    ]
    return frame[columns].copy()


def run_discovery(
    panel: pd.DataFrame,
    discovery_dates: list,
    features: list[str],
    split_schedule_sha256: str,
) -> tuple[pd.DataFrame, list[dict]]:
    predictions = []
    metadata = []
    discovery = panel[panel["date"].isin(set(discovery_dates))]
    splits = _expanding_splits(
        discovery_dates,
        n_folds=OUTER_FOLDS,
        minimum_warmup=90,
    )
    if len(splits) != OUTER_FOLDS:
        raise RuntimeError(
            f"expected {OUTER_FOLDS} outer folds, got {len(splits)}"
        )
    for fold, (train_dates, test_dates) in enumerate(splits):
        train = discovery[
            discovery["date"].isin(set(train_dates))
        ].copy()
        test = discovery[
            discovery["date"].isin(set(test_dates))
        ].copy()
        if not (test.groupby("date").size() == UNIVERSE_TOP_N).all():
            raise RuntimeError("outer test candidate set is not Top100")
        result = _base_columns(test)
        result["split_schedule_sha256"] = split_schedule_sha256
        result["scope"] = "discovery_oof"
        result["fold"] = int(fold)
        for head, target in HEADS.items():
            raw, probability, meta = _predict_target(
                train,
                test,
                features,
                target,
            )
            result[f"raw_{head}"] = raw
            result[f"p_{head}"] = probability
            meta.update(
                scope="discovery_oof",
                outer_fold=int(fold),
                split_schedule_sha256=split_schedule_sha256,
            )
            metadata.append(meta)
        predictions.append(result)
    if not predictions:
        raise RuntimeError("outer discovery produced no predictions")
    return pd.concat(predictions, ignore_index=True), metadata


def run_locked_holdout(
    panel: pd.DataFrame,
    discovery_dates: list,
    holdout_dates: list,
    features: list[str],
    selected_head: str,
    split_schedule_sha256: str,
) -> tuple[pd.DataFrame, dict]:
    target = HEADS[selected_head]
    if len(discovery_dates) <= EMBARGO_DATES:
        raise RuntimeError("not enough discovery dates for holdout embargo")
    holdout_train_dates = discovery_dates[:-EMBARGO_DATES]
    train = panel[
        panel["date"].isin(set(holdout_train_dates))
    ].copy()
    test = panel[panel["date"].isin(set(holdout_dates))].copy()
    if not (test.groupby("date").size() == UNIVERSE_TOP_N).all():
        raise RuntimeError("holdout candidate set is not Top100")
    result = _base_columns(test)
    result["split_schedule_sha256"] = split_schedule_sha256
    result["scope"] = "locked_holdout"
    result["fold"] = -1
    raw, probability, metadata = _predict_target(
        train,
        test,
        features,
        target,
    )
    result[f"raw_{selected_head}"] = raw
    result[f"p_{selected_head}"] = probability
    metadata.update(
        scope="locked_holdout",
        outer_fold=-1,
        split_schedule_sha256=split_schedule_sha256,
        selected_head=selected_head,
        holdout_embargo_dates=EMBARGO_DATES,
        embargoed_discovery_dates=[
            str(value)
            for value in discovery_dates[-EMBARGO_DATES:]
        ],
    )
    return result, metadata


def attach_reproducible_baselines(
    predictions: pd.DataFrame,
    baseline_path: Path,
) -> tuple[pd.DataFrame, dict]:
    if not baseline_path.is_file():
        raise FileNotFoundError(baseline_path)
    if predictions.empty:
        raise RuntimeError("baseline alignment input is empty")
    identity = [
        "split_schedule_sha256",
        "scope",
        "fold",
        "date",
        "market",
    ]
    missing_identity = sorted(set(identity) - set(predictions.columns))
    if missing_identity:
        raise RuntimeError(
            f"prediction identity columns missing: {missing_identity}"
        )
    if predictions.duplicated(identity).any():
        raise RuntimeError("prediction frame has duplicate identity rows")
    required = [
        "split_schedule_sha256",
        "scope",
        "fold",
        "date",
        "market",
        "score_R1_repaired",
        "score_safeup_head",
    ]
    baseline = pd.read_csv(
        baseline_path,
        usecols=required,
        float_precision="round_trip",
    )
    baseline["date"] = pd.to_datetime(
        baseline["date"], errors="raise"
    ).dt.date
    if baseline["date"].isna().any():
        raise RuntimeError("baseline prediction artifact has missing dates")
    if not baseline["market"].astype("string").str.fullmatch(
        r"KRW-[A-Z0-9]+", na=False
    ).all():
        raise RuntimeError("baseline prediction artifact has invalid markets")
    expected_scopes = {"discovery_oof", "locked_holdout"}
    if set(baseline["scope"].astype(str).unique()) != expected_scopes:
        raise RuntimeError("baseline prediction artifact has invalid scopes")
    if baseline.duplicated(identity).any():
        raise RuntimeError("baseline prediction artifact has duplicates")
    candidate_hashes = set(
        predictions["split_schedule_sha256"].astype(str)
    )
    baseline_hashes = set(
        baseline["split_schedule_sha256"].astype(str)
    )
    if (
        len(candidate_hashes) != 1
        or baseline_hashes != candidate_hashes
    ):
        raise RuntimeError("baseline split-schedule hash mismatch")
    scores = baseline[
        ["score_R1_repaired", "score_safeup_head"]
    ].apply(pd.to_numeric, errors="coerce")
    if scores.isna().any().any() or not np.isfinite(
        scores.to_numpy()
    ).all():
        raise RuntimeError("baseline prediction artifact has nonfinite scores")
    baseline[["score_R1_repaired", "score_safeup_head"]] = scores
    before = _date_coverage(predictions)
    merged = predictions.merge(
        baseline,
        on=identity,
        how="left",
        validate="one_to_one",
    )
    missing_mask = (
        merged["score_R1_repaired"].isna()
        | merged["score_safeup_head"].isna()
    )
    if missing_mask.any():
        missing_by_date = missing_mask.groupby(merged["date"]).sum()
        raise RuntimeError(
            "baseline artifact does not exactly cover predictions: "
            f"{missing_by_date[missing_by_date > 0].head().to_dict()}"
        )
    counts = merged.groupby(["scope", "date"]).size()
    if len(counts) == 0 or not (counts == UNIVERSE_TOP_N).all():
        raise RuntimeError("baseline-aligned frame is not exact Top100")
    merged["score_ATR_top3"] = merged["f_atr_pct_14"]
    merged["score_monkey_seed42"] = [
        _stable_hash(date, str(market), "monkey")
        for date, market in zip(merged["date"], merged["market"])
    ]
    return merged, {
        "before": before,
        "after": _date_coverage(merged),
        "whole_dates_dropped": [],
        "whole_date_drop_count": 0,
        "partial_date_missing_count": 0,
        "exact_identity_coverage": True,
    }


def _top3(
    predictions: pd.DataFrame,
    score_column: str,
    policy: str,
) -> pd.DataFrame:
    ordered = predictions.sort_values(
        ["scope", "date", score_column, "market"],
        ascending=[True, True, False, True],
    )
    picks = (
        ordered.groupby(["scope", "date"], sort=False)
        .head(TOP_K)
        .copy()
    )
    picks["policy"] = policy
    picks["selection_rank"] = (
        picks.groupby(["scope", "date"]).cumcount() + 1
    )
    return picks


def _liquidity_matched(
    predictions: pd.DataFrame,
    anchor: pd.DataFrame,
) -> pd.DataFrame:
    """Deterministic random picks in the anchor's D-1 qv deciles."""
    rows = []
    for (scope, date), desired in anchor.groupby(["scope", "date"]):
        universe = predictions[
            (predictions["scope"] == scope)
            & (predictions["date"] == date)
        ].copy()
        universe["_liq_bin"] = (
            (universe["f_qv_rank"].astype(int) - 1) // 10
        )
        universe["_hash"] = [
            _stable_hash(date, str(market), "liquidity")
            for market in universe["market"]
        ]
        used: set[str] = set()
        chosen = []
        for qv_rank in desired.sort_values("selection_rank")[
            "f_qv_rank"
        ]:
            liquidity_bin = (int(qv_rank) - 1) // 10
            candidates = universe[
                (universe["_liq_bin"] == liquidity_bin)
                & (~universe["market"].isin(used))
            ].sort_values(["_hash", "market"])
            if candidates.empty:
                raise RuntimeError("empty liquidity-matched bin")
            row = candidates.iloc[[0]].copy()
            market = str(row["market"].iloc[0])
            used.add(market)
            chosen.append(row)
        matched = pd.concat(chosen, ignore_index=True)
        matched["policy"] = "liquidity_matched"
        matched["selection_rank"] = np.arange(1, TOP_K + 1)
        rows.append(matched)
    return pd.concat(rows, ignore_index=True)


def discovery_picks(predictions: pd.DataFrame) -> pd.DataFrame:
    picks = []
    for head in HEADS:
        picks.append(
            _top3(
                predictions,
                f"raw_{head}",
                head,
            )
        )
    picks.append(
        _top3(
            predictions,
            "score_R1_repaired",
            "R1_repaired",
        )
    )
    return pd.concat(picks, ignore_index=True)


def final_picks(
    predictions: pd.DataFrame,
    selected_head: str,
) -> pd.DataFrame:
    policy_scores = {
        selected_head: f"raw_{selected_head}",
        "R1_repaired": "score_R1_repaired",
        "safeup_head": "score_safeup_head",
        "monkey_seed42": "score_monkey_seed42",
        "ATR_top3": "score_ATR_top3",
    }
    picks = [
        _top3(predictions, score, policy)
        for policy, score in policy_scores.items()
    ]
    anchor = picks[0]
    picks.append(_liquidity_matched(predictions, anchor))
    out = pd.concat(picks, ignore_index=True)
    expected = len(policy_scores) + 1
    counts = out.groupby(["scope", "date", "policy"]).size()
    if not (counts == TOP_K).all():
        raise RuntimeError("final policy/date is not exact Top3")
    if out.groupby(["scope", "date"])["policy"].nunique().min() != expected:
        raise RuntimeError("final policy set is incomplete")
    return out


def _common_complete(picks: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    quality = (
        picks.groupby(["scope", "date", "policy"])
        .agg(n=("market", "size"), complete=("path_complete", "sum"))
        .reset_index()
    )
    good = quality[
        (quality["n"] == TOP_K)
        & (quality["complete"] == TOP_K)
    ]
    expected_policies = picks["policy"].nunique()
    common = (
        good.groupby(["scope", "date"])["policy"]
        .nunique()
        .reset_index(name="policies")
    )
    keys = set(
        common.loc[
            common["policies"] == expected_policies,
            ["scope", "date"],
        ].itertuples(index=False, name=None)
    )
    out = picks[
        [
            (scope, date) in keys
            for scope, date in zip(picks["scope"], picks["date"])
        ]
    ].copy()
    return out, {
        str(scope): int(
            sum(1 for key in keys if key[0] == scope)
        )
        for scope in picks["scope"].unique()
    }


def _daily_metric(frame: pd.DataFrame, metric: str) -> pd.Series:
    if metric == "safe_fp_rate":
        values = frame[FIXED_TARGET].astype(float)
    elif metric == "path_dn5_rate":
        values = frame["path_dn5"].astype(float)
    elif metric == "path_up10_rate":
        values = frame["path_up10"].astype(float)
    elif metric == "bracket_net":
        values = frame["path_bracket_net"].astype(float)
    elif metric == "sl_first_rate":
        values = (
            frame["path_bracket_outcome"] == "sl"
        ).astype(float)
    else:
        raise KeyError(metric)
    return values.groupby(frame["date"]).mean().sort_index()


def paired_bootstrap(
    picks: pd.DataFrame,
    *,
    primary: str,
    comparators: Iterable[str],
    scope: str,
    draws: int,
) -> pd.DataFrame:
    metrics = (
        "safe_fp_rate",
        "path_dn5_rate",
        "path_up10_rate",
        "bracket_net",
        "sl_first_rate",
    )
    rows = []
    scoped = picks[picks["scope"] == scope]
    for comparator in comparators:
        first = scoped[scoped["policy"] == primary]
        second = scoped[scoped["policy"] == comparator]
        for metric in metrics:
            a = _daily_metric(first, metric)
            b = _daily_metric(second, metric)
            common = a.index.intersection(b.index)
            delta = (
                a.loc[common].to_numpy()
                - b.loc[common].to_numpy()
            )
            if len(delta) == 0:
                continue
            salt = _stable_hash(scope, f"{primary}|{comparator}", metric)
            rng = np.random.default_rng(salt)
            indices = rng.integers(
                0,
                len(delta),
                size=(draws, len(delta)),
            )
            boot = delta[indices].mean(axis=1)
            lower_better = metric in (
                "path_dn5_rate",
                "sl_first_rate",
            )
            rows.append(
                {
                    "scope": scope,
                    "primary": primary,
                    "comparator": comparator,
                    "metric": metric,
                    "n_paired_dates": int(len(delta)),
                    "delta_primary_minus_comparator": float(
                        delta.mean()
                    ),
                    "ci95_lo": float(np.percentile(boot, 2.5)),
                    "ci95_hi": float(np.percentile(boot, 97.5)),
                    "p_delta_gt_zero": float((boot > 0).mean()),
                    "preferred_direction": (
                        "lower" if lower_better else "higher"
                    ),
                    "p_preferred_direction": float(
                        (boot < 0).mean()
                        if lower_better
                        else (boot > 0).mean()
                    ),
                    "bootstrap_draws": int(draws),
                }
            )
    return pd.DataFrame(rows)


def select_variant(
    discovery_common: pd.DataFrame,
    *,
    draws: int,
) -> tuple[str, pd.DataFrame, dict]:
    comparisons = []
    for head in HEADS:
        paired = paired_bootstrap(
            discovery_common,
            primary=head,
            comparators=["R1_repaired"],
            scope="discovery_oof",
            draws=draws,
        )
        comparisons.append(paired)
    paired_all = pd.concat(comparisons, ignore_index=True)
    candidates: list[dict] = []
    for head in HEADS:
        part = paired_all[
            (paired_all["primary"] == head)
            & (paired_all["comparator"] == "R1_repaired")
        ].set_index("metric")
        safe_delta = float(
            part.loc["safe_fp_rate", "delta_primary_minus_comparator"]
        )
        downside_delta = float(
            part.loc[
                "path_dn5_rate",
                "delta_primary_minus_comparator",
            ]
        )
        net_delta = float(
            part.loc["bracket_net", "delta_primary_minus_comparator"]
        )
        net_hi = float(part.loc["bracket_net", "ci95_hi"])
        eligible_flag = (
            safe_delta > 0
            and downside_delta <= 0
            and net_hi >= 0
        )
        candidates.append(
            {
                "head": head,
                "safe_delta": safe_delta,
                "downside_delta": downside_delta,
                "net_delta": net_delta,
                "net_ci95_hi": net_hi,
                "discovery_eligible": bool(eligible_flag),
            }
        )
    eligible_rows = [
        row for row in candidates if row["discovery_eligible"]
    ]
    if eligible_rows:
        selected = sorted(
            eligible_rows,
            key=lambda row: (
                row["safe_delta"],
                row["net_delta"],
                row["head"] == PRIMARY_HEAD,
            ),
            reverse=True,
        )[0]["head"]
        reason = "best eligible discovery safe first-passage delta"
    else:
        selected = PRIMARY_HEAD
        reason = "no candidate passed discovery screen; ex-ante primary fallback"
    return selected, paired_all, {
        "selection_rule": (
            "safe_fp point delta > 0, dn5 point delta <= 0, and "
            "net CI95 upper >= 0 versus repaired R1; then max safe delta"
        ),
        "candidates": candidates,
        "selected": selected,
        "reason": reason,
        "holdout_not_used": True,
    }


def _within_vol_auc(
    frame: pd.DataFrame,
    label: str,
    score: str,
) -> float:
    weighted = []
    for _, group in frame.groupby("vol_band"):
        auc = _safe_auc(group[label], group[score])
        if np.isfinite(auc):
            weighted.append((auc, len(group)))
    if not weighted:
        return np.nan
    return float(
        sum(auc * n for auc, n in weighted)
        / sum(n for _, n in weighted)
    )


def _daily_macro_auc(
    frame: pd.DataFrame,
    label: str,
    score: str,
) -> float:
    values = []
    for _, group in frame.groupby("date"):
        auc = _safe_auc(group[label], group[score])
        if np.isfinite(auc):
            values.append(auc)
    return float(np.mean(values)) if values else np.nan


def auc_metrics(
    predictions: pd.DataFrame,
    selected_head: str,
) -> pd.DataFrame:
    rows = []
    for scope in predictions["scope"].unique():
        scoped = predictions[
            (predictions["scope"] == scope)
            & predictions["path_complete"]
        ].copy()
        available_heads = [
            head
            for head in HEADS
            if f"raw_{head}" in scoped.columns
            and scoped[f"raw_{head}"].notna().any()
        ]
        for head in available_heads:
            own_label = HEADS[head]
            evaluation_labels = (
                (own_label,)
                if own_label == FIXED_TARGET
                else (own_label, FIXED_TARGET)
            )
            for evaluation_label in evaluation_labels:
                raw_col = f"raw_{head}"
                probability_col = f"p_{head}"
                valid = scoped.dropna(
                    subset=[
                        evaluation_label,
                        raw_col,
                        probability_col,
                    ]
                )
                if valid.empty:
                    continue
                rows.append(
                    {
                        "scope": scope,
                        "head": head,
                        "trained_target": own_label,
                        "evaluation_label": evaluation_label,
                        "n": int(len(valid)),
                        "dates": int(valid["date"].nunique()),
                        "base_rate": float(
                            valid[evaluation_label].mean()
                        ),
                        "raw_auc": _safe_auc(
                            valid[evaluation_label],
                            valid[raw_col],
                        ),
                        "probability_auc": _safe_auc(
                            valid[evaluation_label],
                            valid[probability_col],
                        ),
                        "daily_macro_auc": _daily_macro_auc(
                            valid,
                            evaluation_label,
                            raw_col,
                        ),
                        "within_vol_auc": _within_vol_auc(
                            valid,
                            evaluation_label,
                            raw_col,
                        ),
                        "brier": float(
                            brier_score_loss(
                                valid[evaluation_label],
                                valid[probability_col],
                            )
                        ),
                        "mean_probability": float(
                            valid[probability_col].mean()
                        ),
                        "score_atr_spearman": float(
                            valid[[raw_col, "f_atr_pct_14"]]
                            .corr(method="spearman")
                            .iloc[0, 1]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def policy_metrics(
    predictions: pd.DataFrame,
    common_picks: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for (scope, policy), group in common_picks.groupby(
        ["scope", "policy"]
    ):
        reference = predictions[
            (predictions["scope"] == scope)
            & predictions["date"].isin(set(group["date"]))
            & predictions["path_complete"]
        ]
        strata = (
            reference.groupby(["date", "vol_band"])[FIXED_TARGET]
            .mean()
            .reset_index(name="matched_safe_fp")
        )
        matched = group.merge(
            strata,
            on=["date", "vol_band"],
            how="left",
            validate="many_to_one",
        )
        daily = group.groupby("date")["path_bracket_net"].mean()
        equity = (1.0 + daily).cumprod()
        peak = equity.cummax()
        net = group["path_bracket_net"].to_numpy(dtype=float)
        k5 = max(1, int(np.ceil(len(net) * 0.05)))
        safe_rate = float(group[FIXED_TARGET].mean())
        matched_rate = float(matched["matched_safe_fp"].mean())
        rows.append(
            {
                "scope": scope,
                "policy": policy,
                "n": int(len(group)),
                "dates": int(group["date"].nunique()),
                "safe_fp_rate": safe_rate,
                "path_up10_rate": float(group["path_up10"].mean()),
                "path_dn5_rate": float(group["path_dn5"].mean()),
                "sl_first_rate": float(
                    (
                        group["path_bracket_outcome"] == "sl"
                    ).mean()
                ),
                "tp_first_rate": float(
                    (
                        group["path_bracket_outcome"] == "tp"
                    ).mean()
                ),
                "bracket_net_mean": float(net.mean()),
                "bracket_hit_rate": float((net > 0).mean()),
                "cvar95": float(np.sort(net)[:k5].mean()),
                "mfe20_rate": float(group["path_mfe20"].mean()),
                "cum_net": float(equity.iloc[-1] - 1.0),
                "max_drawdown": float(
                    ((equity - peak) / peak).min()
                ),
                "matched_vol_safe_fp": matched_rate,
                "safe_fp_lift_within_vol": (
                    safe_rate / matched_rate
                    if matched_rate > 0
                    else np.nan
                ),
                "mean_qv_rank": float(group["f_qv_rank"].mean()),
                "flat_filled_pick_rate": float(
                    (group["path_quality"] == "flat_filled").mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def fold_metrics(common_picks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scope, fold, policy), group in common_picks.groupby(
        ["scope", "fold", "policy"]
    ):
        rows.append(
            {
                "scope": scope,
                "fold": int(fold),
                "policy": policy,
                "n": int(len(group)),
                "dates": int(group["date"].nunique()),
                "safe_fp_rate": float(group[FIXED_TARGET].mean()),
                "path_dn5_rate": float(group["path_dn5"].mean()),
                "path_up10_rate": float(group["path_up10"].mean()),
                "sl_first_rate": float(
                    (
                        group["path_bracket_outcome"] == "sl"
                    ).mean()
                ),
                "bracket_net_mean": float(
                    group["path_bracket_net"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def holdout_gate(
    paired: pd.DataFrame,
    selected_head: str,
) -> dict:
    comparison = paired[
        (paired["scope"] == "locked_holdout")
        & (paired["primary"] == selected_head)
        & (paired["comparator"] == "R1_repaired")
    ].set_index("metric")
    downside_nonworse = bool(
        comparison.loc["path_dn5_rate", "ci95_hi"] <= 0
    )
    safe_higher = bool(
        comparison.loc["safe_fp_rate", "ci95_lo"] > 0
    )
    net_not_adverse = bool(
        comparison.loc["bracket_net", "ci95_lo"] >= 0
    )
    passed = downside_nonworse and safe_higher and net_not_adverse
    return {
        "comparison": "selected head versus repaired R1",
        "downside_nonworse": {
            "rule": "paired dn5 delta CI95 upper <= 0",
            "passed": downside_nonworse,
        },
        "safe_first_passage_higher": {
            "rule": "paired safe-FP delta CI95 lower > 0",
            "passed": safe_higher,
        },
        "net_uncertainty_not_adverse": {
            "rule": "paired net delta CI95 lower >= 0",
            "passed": net_not_adverse,
        },
        "all_required": passed,
        "verdict": (
            "SHADOW"
            if passed
            else "REJECT"
        ),
        "adopt_blocked_without_forward": True,
    }


def _write_json(path: Path, payload: dict) -> None:
    text = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
        temporary_path = Path(handle.name)
    temporary_path.replace(path)


def _manifest_artifacts(prefix: Path) -> dict[str, Path]:
    return {
        "predictions": Path(f"{prefix}_predictions.csv.gz"),
        "picks": Path(f"{prefix}_picks.csv.gz"),
        "summary": Path(f"{prefix}_summary.csv"),
        "auc": Path(f"{prefix}_auc.csv"),
        "paired": Path(f"{prefix}_paired.csv"),
        "folds": Path(f"{prefix}_folds.csv"),
        "coverage": Path(f"{prefix}_coverage.json"),
        "path_panel_cache": Path(f"{prefix}_path_panel.csv.gz"),
        "path_panel_cache_meta": Path(f"{prefix}_path_panel_meta.json"),
    }


def _output_prefix_from_predictions(path: Path) -> Path:
    suffix = "_predictions.csv.gz"
    if not path.name.endswith(suffix):
        raise RuntimeError(
            "first-passage reference must be a *_predictions.csv.gz artifact"
        )
    return path.with_name(path.name[: -len(suffix)])


def _verify_manifest(
    *,
    manifest_path: Path,
    schema: str,
    expected: dict[str, Path],
) -> dict:
    if not manifest_path.is_file():
        raise RuntimeError(f"artifact manifest is missing: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("artifact manifest is unreadable") from exc
    if payload.get("schema") != schema:
        raise RuntimeError("artifact manifest schema mismatch")
    files = payload.get("files")
    if not isinstance(files, dict) or set(files) != set(expected):
        raise RuntimeError("artifact manifest file set mismatch")
    for name, expected_path in expected.items():
        entry = files.get(name)
        if not isinstance(entry, dict):
            raise RuntimeError(f"manifest entry is invalid: {name}")
        recorded_path = Path(str(entry.get("path", "")))
        if not recorded_path.is_absolute():
            recorded_path = ROOT / recorded_path
        if recorded_path.resolve() != expected_path.resolve():
            raise RuntimeError(f"manifest path mismatch: {name}")
        if not expected_path.is_file():
            raise RuntimeError(f"artifact is missing: {expected_path}")
        if int(entry.get("bytes", -1)) != expected_path.stat().st_size:
            raise RuntimeError(f"artifact size mismatch: {name}")
        if entry.get("sha256") != _sha256(expected_path):
            raise RuntimeError(f"artifact checksum mismatch: {name}")
    return payload


def _publish_staged_files(
    staged_to_target: dict[Path, Path],
) -> None:
    """Publish a generation with rollback to every pre-existing target."""
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


@contextmanager
def _preserve_existing_files_on_failure(
    targets: list[Path],
) -> Iterator[None]:
    """Restore cache inputs if a long rebuild/evaluation fails later."""
    if not targets:
        yield
        return
    output_dir = targets[0].parent
    output_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = Path(
        tempfile.mkdtemp(prefix=".cache-backup.", dir=output_dir)
    )
    backups: dict[Path, Path | None] = {}
    remove_backup_dir = True
    try:
        for index, target in enumerate(targets):
            if target.is_file():
                backup = backup_dir / f"{index:03d}.backup"
                shutil.copy2(target, backup)
                backups[target] = backup
            else:
                backups[target] = None
        yield
    except BaseException:
        rollback_errors = []
        for target in targets:
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
                "evaluation failed and cache rollback was incomplete: "
                + "; ".join(rollback_errors)
                + f"; recovery backups preserved at {backup_dir}"
            )
        raise
    finally:
        if remove_backup_dir:
            shutil.rmtree(backup_dir, ignore_errors=True)


def _validate_safeup_baseline_lineage(
    *,
    baseline_predictions: Path,
    d1_db: Path,
    m15_db: Path,
) -> dict:
    baseline_prefix = safeup._output_prefix_from_predictions(
        baseline_predictions
    )
    suffix = "_fp_schedule"
    if not baseline_prefix.name.endswith(suffix):
        raise RuntimeError(
            "first-passage baseline is not a shared FP schedule artifact"
        )
    standalone_prefix = baseline_prefix.with_name(
        baseline_prefix.name[: -len(suffix)]
    )
    standalone_audit = safeup.validate_existing_artifacts(
        output_prefix=standalone_prefix,
        d1_db=d1_db,
        m15_db=m15_db,
    )
    schedule_audit = standalone_audit.get("fp_schedule_baseline")
    if not isinstance(schedule_audit, dict):
        raise RuntimeError(
            "validated SafeUp generation lacks the shared schedule"
        )
    if (
        Path(str(schedule_audit["output_prefix"])).resolve()
        != baseline_prefix.resolve()
    ):
        raise RuntimeError("shared baseline output-prefix mismatch")
    return {
        **schedule_audit,
        "standalone_safeup_manifest_sha256": standalone_audit[
            "manifest_sha256"
        ],
    }


def validate_existing_artifacts(
    *,
    output_prefix: Path,
    d1_db: Path,
    m15_db: Path,
    baseline_predictions: Path,
    limit_markets: int | None = None,
) -> dict:
    """Reject stale/tampered research outputs before report consumption."""
    expected = _manifest_artifacts(output_prefix)
    _verify_manifest(
        manifest_path=Path(f"{output_prefix}_manifest.json"),
        schema="first_passage_head_challenger_v1_manifest",
        expected=expected,
    )
    current_baseline_audit = _validate_safeup_baseline_lineage(
        baseline_predictions=baseline_predictions,
        d1_db=d1_db,
        m15_db=m15_db,
    )
    try:
        coverage = json.loads(
            expected["coverage"].read_text(encoding="utf-8")
        )
        cache_meta = json.loads(
            expected["path_panel_cache_meta"].read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("artifact metadata is unreadable") from exc
    if coverage.get("schema") != "first_passage_head_challenger_v1":
        raise RuntimeError("coverage schema mismatch")
    lineage = coverage.get("input_lineage")
    if not isinstance(lineage, dict):
        raise RuntimeError("coverage input lineage is missing")
    if lineage.get("script_sha256") != _sha256(Path(__file__)):
        raise RuntimeError("first-passage artifacts were built by stale code")
    if lineage.get("code_lineage") != _code_lineage():
        raise RuntimeError(
            "first-passage local code dependencies have changed"
        )
    if lineage.get("baseline_generation_audit") != current_baseline_audit:
        raise RuntimeError("first-passage baseline generation has changed")
    if lineage.get("baseline_source_signature") != _file_signature(
        baseline_predictions
    ):
        raise RuntimeError("first-passage baseline source has changed")
    expected_signature = _path_cache_signature(
        d1_db=d1_db,
        m15_db=m15_db,
        limit_markets=limit_markets,
    )
    if cache_meta.get("signature") != expected_signature:
        raise RuntimeError("path-panel source/code signature is stale")
    if cache_meta.get("cache_sha256") != _sha256(
        expected["path_panel_cache"]
    ):
        raise RuntimeError("path-panel cache checksum mismatch")
    panel = pd.read_csv(
        expected["path_panel_cache"],
        float_precision="round_trip",
    )
    panel = _validate_labeled_panel(panel)
    benchmark_dates = sorted(
        panel.loc[panel["benchmark_complete"], "date"].unique()
    )
    current_schedule = safeup.build_common_benchmark_schedule(
        benchmark_dates
    )[0]
    if coverage.get("shared_split_schedule") != current_schedule:
        raise RuntimeError("first-passage shared schedule is stale")
    if (
        current_schedule["split_schedule_sha256"]
        != current_baseline_audit["split_schedule_sha256"]
    ):
        raise RuntimeError(
            "first-passage/baseline shared schedule mismatch"
        )
    prediction_identity = pd.read_csv(
        expected["predictions"],
        usecols=[
            "split_schedule_sha256",
            "scope",
            "fold",
            "date",
            "market",
        ],
    )
    prediction_identity["date"] = pd.to_datetime(
        prediction_identity["date"], errors="raise"
    ).dt.date
    identity_columns = [
        "split_schedule_sha256",
        "scope",
        "fold",
        "date",
        "market",
    ]
    if prediction_identity.duplicated(identity_columns).any():
        raise RuntimeError(
            "first-passage predictions have duplicate identities"
        )
    if not prediction_identity["market"].astype("string").str.fullmatch(
        r"KRW-[A-Z0-9]+", na=False
    ).all():
        raise RuntimeError(
            "first-passage predictions contain invalid markets"
        )
    if set(
        prediction_identity["split_schedule_sha256"].astype(str)
    ) != {current_schedule["split_schedule_sha256"]}:
        raise RuntimeError(
            "first-passage prediction schedule hash mismatch"
        )
    expected_date_keys = {
        (
            str(record["scope"]),
            int(record["fold"]),
            pd.Timestamp(date).date(),
        )
        for record in current_schedule["folds"]
        for date in record["test_dates"]
    }
    observed_counts = prediction_identity.groupby(
        ["scope", "fold", "date"]
    ).size()
    if (
        set(observed_counts.index) != expected_date_keys
        or not (observed_counts == UNIVERSE_TOP_N).all()
    ):
        raise RuntimeError(
            "first-passage predictions do not exactly cover the schedule"
        )
    return {
        "status": "valid",
        "schema": coverage["schema"],
        "output_prefix": str(output_prefix),
        "rows": int(len(panel)),
        "dates": int(panel["date"].nunique()),
        "manifest_sha256": _sha256(
            Path(f"{output_prefix}_manifest.json")
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct 09:15 first-passage challenger"
    )
    parser.add_argument("--d1-db", type=Path, default=D1_DB)
    parser.add_argument("--m15-db", type=Path, default=M15_DB)
    parser.add_argument(
        "--baseline-predictions",
        type=Path,
        default=BASELINE_PREDICTIONS,
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=OUT_PREFIX,
    )
    parser.add_argument("--limit-markets", type=int, default=None)
    parser.add_argument(
        "--bootstrap-draws",
        type=int,
        default=BOOTSTRAP_DRAWS,
    )
    parser.add_argument(
        "--rebuild-path-cache",
        action="store_true",
        help="ignore a matching labeled path-panel cache",
    )
    parser.add_argument(
        "--validate-existing",
        action="store_true",
        help="validate current artifacts and exit without rebuilding",
    )
    return parser.parse_args()


def _run(args: argparse.Namespace) -> None:
    if args.validate_existing:
        audit = validate_existing_artifacts(
            output_prefix=args.output_prefix,
            d1_db=args.d1_db,
            m15_db=args.m15_db,
            baseline_predictions=args.baseline_predictions,
            limit_markets=args.limit_markets,
        )
        print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
        return
    if not 5_000 <= args.bootstrap_draws <= 100_000:
        raise ValueError("--bootstrap-draws must be in [5000, 100000]")
    if args.limit_markets is not None and args.limit_markets < UNIVERSE_TOP_N:
        raise ValueError("--limit-markets must be >= 100")
    if not args.d1_db.is_file():
        raise FileNotFoundError(args.d1_db)
    if not args.m15_db.is_file():
        raise FileNotFoundError(args.m15_db)
    if not args.baseline_predictions.exists():
        raise FileNotFoundError(args.baseline_predictions)
    baseline_lineage_audit = _validate_safeup_baseline_lineage(
        baseline_predictions=args.baseline_predictions,
        d1_db=args.d1_db,
        m15_db=args.m15_db,
    )
    baseline_source_signature = _file_signature(
        args.baseline_predictions
    )
    code_lineage = _code_lineage()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    panel, path_meta = load_or_build_labeled_panel(
        d1_db=args.d1_db,
        m15_db=args.m15_db,
        limit_markets=args.limit_markets,
        output_prefix=args.output_prefix,
        rebuild=args.rebuild_path_cache,
    )
    path_cache_path = Path(f"{args.output_prefix}_path_panel.csv.gz")
    path_cache_meta_path = Path(
        f"{args.output_prefix}_path_panel_meta.json"
    )
    path_cache_source_signatures = {
        "panel": _file_signature(path_cache_path),
        "metadata": _file_signature(path_cache_meta_path),
    }
    panel_history = _history_contract(panel, "final_labeled_panel")
    features = safeup._feature_columns(panel)
    benchmark_dates = sorted(
        panel.loc[panel["benchmark_complete"], "date"].unique()
    )
    (
        shared_schedule,
        _,
        _,
        holdout_date_array,
    ) = safeup.build_common_benchmark_schedule(
        benchmark_dates,
    )
    split_schedule_sha256 = str(
        shared_schedule["split_schedule_sha256"]
    )
    if (
        split_schedule_sha256
        != baseline_lineage_audit["split_schedule_sha256"]
        or shared_schedule["eligible_dates_sha256"]
        != baseline_lineage_audit["eligible_dates_sha256"]
        or shared_schedule["locked_holdout_dates_sha256"]
        != baseline_lineage_audit[
            "locked_holdout_dates_sha256"
        ]
    ):
        raise RuntimeError(
            "path panel and SafeUp baseline shared schedules differ"
        )
    discovery_dates = list(
        np.asarray(benchmark_dates, dtype=object)[
            :-LOCKED_COMMON_DATES
        ]
    )
    holdout_dates = list(holdout_date_array)
    holdout_hash = str(
        shared_schedule["locked_holdout_dates_sha256"]
    )

    # Only the discovery slice is passed to model/variant selection here.
    discovery, discovery_meta = run_discovery(
        panel,
        discovery_dates,
        features,
        split_schedule_sha256,
    )
    discovery, discovery_baseline_meta = attach_reproducible_baselines(
        discovery,
        args.baseline_predictions,
    )
    discovery_selected = discovery_picks(discovery)
    discovery_pick_history = _history_contract(
        discovery_selected,
        "all_discovery_policy_picks_before_path_intersection",
    )
    discovery_common, discovery_common_meta = _common_complete(
        discovery_selected
    )
    selected_head, discovery_paired, selection_meta = select_variant(
        discovery_common,
        draws=args.bootstrap_draws,
    )

    # The sealed dates are unlocked only after selected_head is fixed.
    holdout, holdout_meta = run_locked_holdout(
        panel,
        discovery_dates,
        holdout_dates,
        features,
        selected_head,
        split_schedule_sha256,
    )
    holdout, holdout_baseline_meta = attach_reproducible_baselines(
        holdout,
        args.baseline_predictions,
    )
    if _file_signature(
        args.baseline_predictions
    ) != baseline_source_signature:
        raise RuntimeError("baseline artifact changed during evaluation")
    if (
        _validate_safeup_baseline_lineage(
            baseline_predictions=args.baseline_predictions,
            d1_db=args.d1_db,
            m15_db=args.m15_db,
        )
        != baseline_lineage_audit
    ):
        raise RuntimeError("baseline generation changed during evaluation")
    if {
        "panel": _file_signature(path_cache_path),
        "metadata": _file_signature(path_cache_meta_path),
    } != path_cache_source_signatures:
        raise RuntimeError("path-panel cache changed during evaluation")
    if _code_lineage() != code_lineage:
        raise RuntimeError("local code changed during evaluation")
    final_selected = final_picks(holdout, selected_head)
    holdout_pick_history = _history_contract(
        final_selected,
        "all_holdout_policy_picks_before_path_intersection",
    )
    final_common, final_common_meta = _common_complete(final_selected)
    holdout_paired = paired_bootstrap(
        final_common,
        primary=selected_head,
        comparators=FINAL_BASELINES,
        scope="locked_holdout",
        draws=args.bootstrap_draws,
    )
    paired = pd.concat(
        [discovery_paired, holdout_paired],
        ignore_index=True,
    )

    predictions = pd.concat(
        [discovery, holdout],
        ignore_index=True,
        sort=False,
    )
    picks = pd.concat(
        [discovery_common, final_common],
        ignore_index=True,
        sort=False,
    )
    output_pick_history = _history_contract(
        picks,
        "all_output_policy_picks_after_common_path_intersection",
    )
    auc = auc_metrics(predictions, selected_head)
    summary = policy_metrics(predictions, picks)
    folds = fold_metrics(picks)
    gate = holdout_gate(paired, selected_head)

    prefix = args.output_prefix
    predictions_path = Path(f"{prefix}_predictions.csv.gz")
    picks_path = Path(f"{prefix}_picks.csv.gz")
    summary_path = Path(f"{prefix}_summary.csv")
    auc_path = Path(f"{prefix}_auc.csv")
    paired_path = Path(f"{prefix}_paired.csv")
    folds_path = Path(f"{prefix}_folds.csv")
    coverage = {
        "schema": "first_passage_head_challenger_v1",
        "research_only": True,
        "production_modified": False,
        "auto_order_code": False,
        "authorization_boundary": (
            "user-authorized separate challenger research during the old "
            "moratorium; cannot count as clean preregistration evidence"
        ),
        "targets": {
            "primary": (
                "09:15 execution-window +10% first before -5%; "
                "same-bar downside first"
            ),
            "secondary": (
                "09:15 execution-window 2*D-1 ATR14 first before "
                "1*D-1 ATR14; same-bar downside first"
            ),
            "atr_definition": (
                "per-row barriers are exactly up=2*f_atr_pct_14 and "
                "down=1*f_atr_pct_14; no cap, train anchor, or holdout tuning"
            ),
        },
        "trials": {
            "challenger_head_count": 2,
            "heads": HEADS,
            "hyperparameter_sweep": False,
            "model_spec_shared": (
                "XGBClassifier n_estimators=180,max_depth=4,lr=.05,"
                "subsample=.8,colsample=.8,min_child_weight=5,"
                "reg_lambda=1.5,seed=42,n_jobs=1,tree_method=hist"
            ),
        },
        "hygiene": {
            "all_d1_db_markets": True,
            "minimum_prior_history_before_universe": MIN_PRIOR_HISTORY,
            "universe": "same-date D-1 quote-volume Top100",
            "feature_boundary": "<= D-1",
            "outer": (
                f"{OUTER_FOLDS}-fold expanding WF, "
                f"{EMBARGO_DATES}-date embargo"
            ),
            "inner": (
                f"{INNER_FOLDS}-fold true expanding OOF isotonic, "
                f"{EMBARGO_DATES}-date embargo"
            ),
            "round_trip_cost_once": ROUND_TRIP_COST,
            "path": (
                "[D09:15,D+1 09:15), KRW-BTC exact 96-grid and "
                "closed boundary, target-only gaps flat-filled"
            ),
            "history_contract": {
                "panel": panel_history,
                "discovery_all_policy_picks": discovery_pick_history,
                "holdout_all_policy_picks": holdout_pick_history,
                "output_common_policy_picks": output_pick_history,
            },
        },
        "path_panel": path_meta,
        "input_lineage": {
            "d1_db": str(args.d1_db),
            "d1_db_sha256": _sha256(args.d1_db),
            "m15_db": str(args.m15_db),
            "m15_db_sha256": _sha256(args.m15_db),
            "baseline_predictions": str(args.baseline_predictions),
            "baseline_predictions_sha256": _sha256(
                args.baseline_predictions
            ),
            "path_panel_cache_sha256": _sha256(
                Path(f"{prefix}_path_panel.csv.gz")
            ),
            "path_panel_cache_meta_sha256": _sha256(
                Path(f"{prefix}_path_panel_meta.json")
            ),
            "script_sha256": _sha256(Path(__file__)),
            "code_lineage": code_lineage,
            "baseline_generation_audit": baseline_lineage_audit,
            "baseline_source_signature": baseline_source_signature,
            "path_cache_source_signatures": path_cache_source_signatures,
        },
        "atr_barrier_diagnostics": {
            "discovery": barrier_diagnostics(panel, discovery_dates),
            "locked_holdout": barrier_diagnostics(panel, holdout_dates),
        },
        "sealed_holdout": {
            "definition": (
                "last 180 dates on the shared exact benchmark-complete "
                "eligibility axis, identically refit for every comparator"
            ),
            "n": len(holdout_dates),
            "start": str(holdout_dates[0]),
            "end": str(holdout_dates[-1]),
            "dates_sha256": holdout_hash,
            "discovery_dates": len(discovery_dates),
            "unlocked_after_variant_selection": True,
            "virgin_or_clean_preregistered": False,
            "historical_holdout_contaminated": True,
            "comparison_fairness_restored_not_virginity": True,
        },
        "shared_split_schedule": shared_schedule,
        "variant_selection": selection_meta,
        "effective_dates": {
            "discovery_predictions": _date_coverage(discovery),
            "locked_holdout_predictions": _date_coverage(holdout),
            "discovery_common_complete": _date_coverage(
                discovery_common
            ),
            "locked_holdout_common_complete": _date_coverage(
                final_common
            ),
        },
        "discovery_common_complete_dates": discovery_common_meta,
        "holdout_common_complete_dates": final_common_meta,
        "model_metadata": {
            "discovery": discovery_meta,
            "holdout": holdout_meta,
        },
        "baselines": {
            "repaired_r1_and_safeup_source": str(
                args.baseline_predictions.relative_to(ROOT)
            ),
            "source_sha256": _sha256(args.baseline_predictions),
            "monkey": "SHA256(seed42,date,market)",
            "atr_top3": "highest D-1 ATR14",
            "liquidity_matched": (
                "deterministic SHA256 pick from each selected-head "
                "D-1 qv decile"
            ),
            "alignment": {
                "discovery": discovery_baseline_meta,
                "locked_holdout": holdout_baseline_meta,
            },
        },
        "gate": gate,
        "bootstrap": {
            "unit": "date with all compared Top3 paths complete",
            "draws": args.bootstrap_draws,
        },
        "artifacts": {
            "predictions": str(predictions_path.relative_to(ROOT)),
            "picks": str(picks_path.relative_to(ROOT)),
            "summary": str(summary_path.relative_to(ROOT)),
            "auc": str(auc_path.relative_to(ROOT)),
            "paired": str(paired_path.relative_to(ROOT)),
            "folds": str(folds_path.relative_to(ROOT)),
            "path_panel_cache": str(
                Path(f"{prefix}_path_panel.csv.gz").relative_to(ROOT)
            ),
            "path_panel_cache_meta": str(
                Path(f"{prefix}_path_panel_meta.json").relative_to(ROOT)
            ),
        },
    }
    artifact_paths = _manifest_artifacts(prefix)
    manifest_path = Path(f"{prefix}_manifest.json")
    result_names = (
        "predictions",
        "picks",
        "summary",
        "auc",
        "paired",
        "folds",
        "coverage",
    )
    with tempfile.TemporaryDirectory(
        dir=prefix.parent,
        prefix=f".{prefix.name}.generation.",
    ) as stage_directory:
        stage_root = Path(stage_directory)
        staged = {
            name: stage_root / artifact_paths[name].name
            for name in result_names
        }
        predictions.to_csv(
            staged["predictions"],
            index=False,
            compression=GZIP_COMPRESSION,
            float_format="%.17g",
        )
        picks.to_csv(
            staged["picks"],
            index=False,
            compression=GZIP_COMPRESSION,
            float_format="%.17g",
        )
        summary.to_csv(staged["summary"], index=False)
        auc.to_csv(staged["auc"], index=False)
        paired.to_csv(staged["paired"], index=False)
        folds.to_csv(staged["folds"], index=False)
        _write_json(staged["coverage"], coverage)
        staged_manifest = stage_root / manifest_path.name
        _write_json(
            staged_manifest,
            {
                "schema": "first_passage_head_challenger_v1_manifest",
                "gzip_mtime": 0,
                "files": {
                    name: {
                        "path": str(target),
                        "bytes": int(
                            staged.get(name, target).stat().st_size
                        ),
                        "sha256": _sha256(
                            staged.get(name, target)
                        ),
                    }
                    for name, target in artifact_paths.items()
                },
            },
        )
        publish_map = {
            staged[name]: artifact_paths[name]
            for name in result_names
        }
        publish_map[staged_manifest] = manifest_path
        _publish_staged_files(publish_map)

    holdout_display = summary[
        summary["scope"] == "locked_holdout"
    ][
        [
            "policy",
            "n",
            "dates",
            "safe_fp_rate",
            "path_dn5_rate",
            "path_up10_rate",
            "sl_first_rate",
            "bracket_net_mean",
            "safe_fp_lift_within_vol",
        ]
    ]
    print(
        f"selected={selected_head}, verdict={gate['verdict']}, "
        f"holdout={holdout_dates[0]}..{holdout_dates[-1]}, "
        f"common_dates={final_common_meta.get('locked_holdout', 0)}"
    )
    print(holdout_display.to_string(index=False))
    print(f"wrote {prefix}_*")


def main() -> None:
    args = parse_args()
    if args.validate_existing:
        _run(args)
        return
    cache_targets = [
        Path(f"{args.output_prefix}_path_panel.csv.gz"),
        Path(f"{args.output_prefix}_path_panel_meta.json"),
    ]
    with _preserve_existing_files_on_failure(cache_targets):
        _run(args)


if __name__ == "__main__":
    main()

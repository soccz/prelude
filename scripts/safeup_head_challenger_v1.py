"""Research-only safe-up head challenger for the manual-trading radar.

The user objective is encoded directly:

    safe_up10(D) = 1[
        high_D / open_D - 1 >= +10%
        and low_D / open_D - 1 > -5%
    ]

This script does not alter the production R1 model, labels, registry, sender,
Telegram format, ledger, or scheduler.  It compares five fixed Top-3 policies:

* ``safeup_head``: rank the direct safe_up10 head.
* ``up10_control``: rank the existing up10 target head.
* ``R1_repaired``: true-inner-OOF calibrated p_up10 / p_dn5.
* ``R1_frozen_pattern``: the historical in-sample/non-monotone bucket pattern
  used as a defect-matched R1 control.
* ``monkey_seed42``: deterministic within-day random Top 3.
* ``safeup_pareto_rank``: within-day pct-rank(raw safe_up10) minus
  pct-rank(raw dn5).  This sixth policy was requested only after the original
  locked holdout had been inspected.  It is explicitly post-hoc,
  promotion-ineligible, and can at most justify a fresh forward-only shadow.

Validation contract
-------------------
* At least 70 completed daily bars must exist before a market/date is eligible.
* The universe is the point-in-time D-1 quote-volume Top 100.
* All 24 model features are shifted to D-1 by ``build_market_features``.
* Discovery uses expanding outer walk-forward with a five-date embargo.
* Calibration uses genuine expanding inner OOF predictions from outer-train
  only; test/holdout outcomes never fit a model or calibrator.
* The standalone diagnostic seals the final 180 PIT D1-eligible dates before
  discovery and unlocks them once.
* A second, schedule-aligned SafeUp/R1 baseline uses the exact
  benchmark-complete train, embargo, and test dates shared by first-passage and
  semivol.  Its historical final-180 slice has already been observed, so it is
  comparison-only evidence rather than virgin/preregistered confirmation.
* The secondary execution lens is the 96-bar path from the first executable
  full bar after a normal 09:10 delivery: [D 09:15, D+1 09:15).  KRW-BTC
  proves collection completeness, target-only gaps are flat-filled, and an
  ambiguous same bar is SL-first.
* A 0.15% round-trip cost is subtracted exactly once.

Outputs are isolated under ``output/safeup_head_challenger_v1*``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.database import list_markets, load_candles  # noqa: E402
from data.market_universe import is_excluded_signal_market  # noqa: E402
from ledger.path_quality import (  # noqa: E402
    BAR_FREQ,
    BENCHMARK_MARKET,
    EXPECTED_BARS,
    assess_15m_window,
)
from ledger.portfolio_metrics import summarize_daily  # noqa: E402
from ops.code_lineage import python_code_lineage  # noqa: E402
from scripts.downside_head_riskreward_v1 import LEAK_COLS  # noqa: E402
from scripts.downside_veto_challenger_v1 import (  # noqa: E402
    BulkPath,
    _safe_up10_before_dn5,
    _valid_bar,
)
from scripts.recommendation_scorer_v1 import PRECURSOR_FEATURES  # noqa: E402
from scripts.recommender_downside_exit_v1 import simulate_path  # noqa: E402
from scripts.regime_split_precursor_v1 import attach_btc_regime  # noqa: E402
from scripts.univariate_precursor_lift_v1 import (  # noqa: E402
    add_cross_sectional,
    build_market_features,
)


D1_DB = ROOT / "data" / "upbit_d1.db"
M15_DB = ROOT / "data" / "upbit_15m.db"
OUT_PREFIX = ROOT / "output" / "safeup_head_challenger_v1"
FP_SCHEDULE_PREFIX = (
    ROOT / "output" / "safeup_head_challenger_v1_fp_schedule"
)

MIN_PRIOR_HISTORY = 70
UNIVERSE_TOP_N = 100
TOP_K = 3
OUTER_FOLDS = 5
INNER_FOLDS = 3
EMBARGO_DATES = 5
LOCKED_HOLDOUT_DATES = 180
ROUND_TRIP_COST = 0.0015
TAKE_PROFIT = 0.05
HARD_STOP = 0.03
MODEL_SEED = 42
RR_EPS = 1e-3
BOOTSTRAP_DRAWS = 5_000
GZIP_COMPRESSION = {"method": "gzip", "mtime": 0}
FP_SCHEDULE_SCHEMA = "challenger_common_benchmark_schedule_v1"
RETURN_TOLERANCE = 1e-8

TARGETS = ("safe_up10", "up10", "dn5")
EX_ANTE_POLICIES = (
    "safeup_head",
    "up10_control",
    "R1_repaired",
    "R1_frozen_pattern",
    "monkey_seed42",
)
POSTHOC_POLICY = "safeup_pareto_rank"
POLICIES = EX_ANTE_POLICIES + (POSTHOC_POLICY,)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("safeup_head_challenger_v1")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _code_lineage() -> dict:
    return python_code_lineage(entrypoint=Path(__file__), root=ROOT)


def _display_path(path: Path) -> str:
    """Use a repo-relative artifact path when possible, absolute otherwise."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


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


def _completed_label_cutoff() -> pd.Timestamp:
    """Return the latest fully closed KST 09:00-to-09:00 trading date."""
    now_kst = pd.Timestamp.now(tz="Asia/Seoul")
    current_session = (now_kst - pd.Timedelta(hours=9)).date()
    return pd.Timestamp(current_session) - pd.Timedelta(days=1)


def _feature_columns(panel: pd.DataFrame) -> list[str]:
    features = [
        column
        for column in PRECURSOR_FEATURES
        if column in panel.columns
        and column not in LEAK_COLS
        and not column.startswith(("next_", "lab_"))
    ]
    if features != list(PRECURSOR_FEATURES):
        missing = sorted(set(PRECURSOR_FEATURES) - set(features))
        raise RuntimeError(
            "24-feature contract changed or leak exclusion fired: "
            f"missing={missing}, found={len(features)}"
        )
    if len(features) != 24:
        raise RuntimeError(f"expected 24 D-1 features, got {len(features)}")
    return features


def _market_frame(
    market: str,
    d1_db: Path | None = None,
) -> pd.DataFrame | None:
    source_db = D1_DB if d1_db is None else d1_db
    raw = load_candles(str(source_db), market)
    if raw is None or len(raw) <= MIN_PRIOR_HISTORY:
        return None
    raw = raw.copy().sort_values("timestamp").reset_index(drop=True)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    raw["market"] = market

    features = build_market_features(raw)
    outcomes = pd.DataFrame(
        {
            "market": market,
            "timestamp": raw["timestamp"],
            "history_prior_bars": np.arange(len(raw), dtype=int),
            "up_high_ret": raw["high"] / raw["open"] - 1.0,
            "down_low_ret": raw["low"] / raw["open"] - 1.0,
            "eod_ret": raw["close"] / raw["open"] - 1.0,
        }
    )
    frame = features.merge(
        outcomes,
        on=["market", "timestamp"],
        how="left",
        validate="one_to_one",
    )
    # Point-in-time eligibility: row D has at least 70 completed bars through
    # D-1.  A market's future lifetime cannot make an early row eligible.
    return frame[frame["history_prior_bars"] >= MIN_PRIOR_HISTORY].copy()


def prepare_panel(
    limit_markets: int | None = None,
    *,
    d1_db: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    source_db = D1_DB if d1_db is None else d1_db
    markets = [
        market
        for market in list_markets(str(source_db))
        if not is_excluded_signal_market(str(market))
    ]
    if limit_markets:
        markets = markets[:limit_markets]
    frames = []
    for index, market in enumerate(markets, start=1):
        frame = _market_frame(str(market), source_db)
        if frame is not None and not frame.empty:
            frames.append(frame)
        if index % 60 == 0:
            log.info("loaded %d/%d daily markets", index, len(markets))
    if not frames:
        raise RuntimeError("no market has the required point-in-time history")

    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["timestamp"]).dt.date
    panel = panel.sort_values(["date", "market"]).reset_index(drop=True)

    # Cross-sectional ranks are calculated only after point-in-time eligibility
    # is enforced.  Thus an eventual long-lived coin cannot enter an old rank.
    panel = add_cross_sectional(panel)
    panel = attach_btc_regime(panel)
    cutoff = _completed_label_cutoff().date()
    panel = panel[panel["date"] <= cutoff].copy()
    panel = panel[panel["f_qv_rank"].notna()].copy()

    panel["up10"] = (panel["up_high_ret"] >= 0.10).astype(int)
    panel["dn5"] = (panel["down_low_ret"] <= -0.05).astype(int)
    panel["safe_up10"] = (
        (panel["up10"] == 1) & (panel["dn5"] == 0)
    ).astype(int)
    panel["vol_band"] = np.minimum(
        np.floor(panel["f_atr_xs_decile"].fillna(0.5).clip(0, 1) * 5),
        4,
    ).astype(int)

    required = [
        "market",
        "date",
        "up_high_ret",
        "down_low_ret",
        "eod_ret",
        "f_qv_rank",
        "f_atr_xs_decile",
    ]
    panel = panel.dropna(subset=required).copy()
    numeric_required = panel[
        [
            "up_high_ret",
            "down_low_ret",
            "eod_ret",
            "f_qv_rank",
            "f_atr_xs_decile",
            "history_prior_bars",
        ]
    ].apply(pd.to_numeric, errors="coerce")
    if (
        numeric_required.isna().any().any()
        or not np.isfinite(numeric_required.to_numpy()).all()
    ):
        raise RuntimeError("panel contains nonfinite required values")
    invalid_ohlc = (
        (panel["up_high_ret"] < -RETURN_TOLERANCE)
        | (panel["down_low_ret"] > RETURN_TOLERANCE)
        | (panel["down_low_ret"] <= -1)
        | (panel["eod_ret"] <= -1)
        | (
            panel["up_high_ret"] + RETURN_TOLERANCE
            < panel["eod_ret"]
        )
        | (
            panel["down_low_ret"] - RETURN_TOLERANCE
            > panel["eod_ret"]
        )
    )
    if invalid_ohlc.any():
        raise RuntimeError("panel daily return columns violate OHLC ordering")
    if not panel["market"].astype("string").str.fullmatch(
        r"KRW-[A-Z0-9]+", na=False
    ).all():
        raise RuntimeError("panel contains invalid/non-canonical markets")
    # Quote-volume ties are resolved deterministically by market.  Re-numbering
    # after the PIT history gate makes the executable candidate frame an exact
    # 1..100 universe rather than a rank<=100 set that can exceed 100 on ties.
    panel = panel.sort_values(
        ["date", "f_qv_rank", "market"],
        ascending=[True, True, True],
    )
    panel["f_qv_rank"] = (
        panel.groupby("date", sort=False).cumcount() + 1
    )
    panel = panel[panel["f_qv_rank"] <= UNIVERSE_TOP_N].copy()
    exact_dates = panel.groupby("date").size()
    exact_dates = set(exact_dates[exact_dates == UNIVERSE_TOP_N].index)
    panel = panel[panel["date"].isin(exact_dates)].copy()
    if panel.empty:
        raise RuntimeError("no exact point-in-time Top100 dates")
    features = _feature_columns(panel)
    bad = [
        feature
        for feature in features
        if feature in LEAK_COLS
        or feature.startswith(("next_", "lab_"))
    ]
    if bad:
        raise RuntimeError(f"leak feature detected: {bad}")
    if int(panel["history_prior_bars"].min()) < MIN_PRIOR_HISTORY:
        raise RuntimeError("point-in-time history contract failed")
    if panel.duplicated(["date", "market"]).any():
        raise RuntimeError("duplicate market/date rows")

    counts = panel.groupby("date").size()
    if not (counts == UNIVERSE_TOP_N).all():
        raise RuntimeError("panel is not exact Top100 on every date")
    meta = {
        "rows": int(len(panel)),
        "markets": int(panel["market"].nunique()),
        "dates": int(panel["date"].nunique()),
        "date_start": str(panel["date"].min()),
        "date_end": str(panel["date"].max()),
        "completed_label_cutoff": str(cutoff),
        "min_history_prior_bars": int(panel["history_prior_bars"].min()),
        "max_candidates_per_date": int(counts.max()),
        "median_candidates_per_date": float(counts.median()),
        "dates_with_exact_100": int((counts == 100).sum()),
        "feature_count": len(features),
        "features": features,
        "base_safe_up10": float(panel["safe_up10"].mean()),
        "base_up10": float(panel["up10"].mean()),
        "base_dn5": float(panel["dn5"].mean()),
    }
    return panel.sort_values(["date", "market"]).reset_index(drop=True), meta


def _expanding_splits(
    dates: Iterable,
    *,
    n_folds: int,
    embargo: int,
    warmup_fraction: float,
    minimum_warmup: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_folds <= 0 or embargo < 0 or minimum_warmup <= 0:
        raise ValueError("invalid expanding-split parameters")
    if not 0 < warmup_fraction < 1:
        raise ValueError("warmup_fraction must be in (0,1)")
    unique = np.sort(np.asarray(list(set(dates)), dtype=object))
    if len(unique) < 2:
        raise ValueError("at least two unique dates are required")
    warmup = max(minimum_warmup, int(len(unique) * warmup_fraction))
    latest_warmup = len(unique) - embargo - n_folds
    if warmup > latest_warmup:
        warmup = min(
            max(30, int(len(unique) * 0.60)),
            latest_warmup,
        )
    if warmup <= 0:
        raise ValueError("not enough dates for expanding splits")
    first_test_start = warmup + embargo
    edges = np.linspace(
        first_test_start,
        len(unique),
        n_folds + 1,
    ).astype(int)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(n_folds):
        test_start = int(edges[fold])
        train_end = test_start - embargo
        test_end = int(edges[fold + 1])
        if test_start >= test_end:
            continue
        train_dates = unique[:train_end]
        test_dates = unique[test_start:test_end]
        if len(train_dates) and len(test_dates):
            splits.append((train_dates, test_dates))
    seen_test_dates: set = set()
    positions = {value: index for index, value in enumerate(unique)}
    for train_dates, test_dates in splits:
        train_set = set(train_dates)
        test_set = set(test_dates)
        if train_set & test_set or seen_test_dates & test_set:
            raise RuntimeError("expanding splits overlap")
        if max(train_dates) >= min(test_dates):
            raise RuntimeError("expanding split chronology is invalid")
        excluded = (
            positions[min(test_dates)] - positions[max(train_dates)] - 1
        )
        if excluded != embargo:
            raise RuntimeError(
                f"expanding split embargo mismatch: {excluded} != {embargo}"
            )
        seen_test_dates.update(test_set)
    test_positions = sorted(
        positions[value] for value in seen_test_dates
    )
    if test_positions and test_positions != list(
        range(test_positions[0], test_positions[-1] + 1)
    ):
        raise RuntimeError("expanding split test coverage has gaps")
    return splits


def _outer_splits(dates: Iterable) -> list[tuple[np.ndarray, np.ndarray]]:
    return _expanding_splits(
        dates,
        n_folds=OUTER_FOLDS,
        embargo=EMBARGO_DATES,
        warmup_fraction=0.35,
        minimum_warmup=180,
    )


def _inner_splits(dates: Iterable) -> list[tuple[np.ndarray, np.ndarray]]:
    return _expanding_splits(
        dates,
        n_folds=INNER_FOLDS,
        embargo=EMBARGO_DATES,
        warmup_fraction=0.45,
        minimum_warmup=120,
    )


def _dates_sha256(dates: Iterable) -> str:
    return hashlib.sha256(
        "\n".join(map(str, dates)).encode("utf-8")
    ).hexdigest()


def build_common_benchmark_schedule(
    benchmark_complete_dates: Iterable,
) -> tuple[
    dict,
    list[tuple[np.ndarray, np.ndarray]],
    np.ndarray,
    np.ndarray,
]:
    """Freeze one path-eligible split schedule shared by all challengers."""
    ordered = np.sort(
        np.asarray(
            list(set(benchmark_complete_dates)),
            dtype=object,
        )
    )
    if len(ordered) <= LOCKED_HOLDOUT_DATES + 60:
        raise RuntimeError(
            "insufficient benchmark-complete dates for shared schedule"
        )
    holdout_dates = ordered[-LOCKED_HOLDOUT_DATES:]
    discovery_dates = ordered[:-LOCKED_HOLDOUT_DATES]
    discovery_splits = _expanding_splits(
        discovery_dates,
        n_folds=OUTER_FOLDS,
        embargo=EMBARGO_DATES,
        warmup_fraction=0.35,
        minimum_warmup=90,
    )
    if len(discovery_splits) != OUTER_FOLDS:
        raise RuntimeError(
            "shared schedule does not have exact outer folds"
        )
    fold_contracts = []
    for fold, (train_dates, test_dates) in enumerate(discovery_splits):
        positions = {
            date: index
            for index, date in enumerate(discovery_dates)
        }
        train_end = positions[max(train_dates)]
        test_start = positions[min(test_dates)]
        embargoed_dates = discovery_dates[
            train_end + 1 : test_start
        ]
        if len(embargoed_dates) != EMBARGO_DATES:
            raise RuntimeError("shared discovery embargo is not exact")
        fold_contracts.append(
            {
                "scope": "discovery_oof",
                "fold": int(fold),
                "train_dates": list(map(str, train_dates)),
                "train_dates_sha256": _dates_sha256(train_dates),
                "train_end": str(max(train_dates)),
                "embargo_dates": list(map(str, embargoed_dates)),
                "test_dates": list(map(str, test_dates)),
                "test_dates_sha256": _dates_sha256(test_dates),
                "test_start": str(min(test_dates)),
                "test_end": str(max(test_dates)),
            }
        )
    holdout_train_dates = discovery_dates[:-EMBARGO_DATES]
    holdout_embargo_dates = discovery_dates[-EMBARGO_DATES:]
    fold_contracts.append(
        {
            "scope": "locked_holdout",
            "fold": -1,
            "train_dates": list(map(str, holdout_train_dates)),
            "train_dates_sha256": _dates_sha256(holdout_train_dates),
            "train_end": str(max(holdout_train_dates)),
            "embargo_dates": list(map(str, holdout_embargo_dates)),
            "test_dates": list(map(str, holdout_dates)),
            "test_dates_sha256": _dates_sha256(holdout_dates),
            "test_start": str(min(holdout_dates)),
            "test_end": str(max(holdout_dates)),
        }
    )
    payload = {
        "schema": FP_SCHEDULE_SCHEMA,
        "eligibility": (
            "PIT D1 exact Top100 dates overlapping nonbenchmark 15m "
            "history with an exact closed KRW-BTC 96-bar window"
        ),
        "outer_folds": OUTER_FOLDS,
        "embargo_eligible_dates": EMBARGO_DATES,
        "eligible_dates": list(map(str, ordered)),
        "eligible_dates_sha256": _dates_sha256(ordered),
        "discovery_dates": list(map(str, discovery_dates)),
        "discovery_dates_sha256": _dates_sha256(discovery_dates),
        "locked_holdout_dates": list(map(str, holdout_dates)),
        "locked_holdout_dates_sha256": _dates_sha256(holdout_dates),
        "historical_holdout_contaminated": True,
        "virgin_or_preregistered": False,
        "maximum_evidence_grade": "historical_comparison_only",
        "folds": fold_contracts,
    }
    payload["split_schedule_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return (
        payload,
        discovery_splits,
        holdout_train_dates,
        holdout_dates,
    )


def _matrix(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    train_x = train[features].replace([np.inf, -np.inf], np.nan)
    medians = train_x.median()
    train_x = train_x.fillna(medians).fillna(0.0)
    test_x = (
        test[features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(medians)
        .fillna(0.0)
    )
    return train_x.to_numpy(dtype=float), test_x.to_numpy(dtype=float)


def _fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    target: str,
    *,
    return_train: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    import xgboost as xgb

    train_ordered = train.sort_values(["date", "market"]).copy()
    x_train, x_test = _matrix(train_ordered, test, features)
    y_train = train_ordered[target].to_numpy(dtype=int)
    positives = int(y_train.sum())
    if positives < 20 or len(np.unique(y_train)) < 2:
        raise RuntimeError(
            f"insufficient target support: target={target} positives={positives}"
        )
    model = xgb.XGBClassifier(
        n_estimators=180,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.5,
        scale_pos_weight=float((len(y_train) - positives) / positives),
        n_jobs=1,
        tree_method="hist",
        eval_metric="logloss",
        random_state=MODEL_SEED,
    )
    model.fit(x_train, y_train, verbose=False)
    test_raw = model.predict_proba(x_test)[:, 1]
    train_raw = model.predict_proba(x_train)[:, 1] if return_train else None
    return np.asarray(test_raw, dtype=float), (
        np.asarray(train_raw, dtype=float) if train_raw is not None else None
    )


def _inner_oof(
    outer_train: pd.DataFrame,
    features: list[str],
    target: str,
) -> tuple[np.ndarray, np.ndarray]:
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    splits = _inner_splits(outer_train["date"])
    if len(splits) != INNER_FOLDS:
        raise RuntimeError(
            f"expected {INNER_FOLDS} inner folds, got {len(splits)}"
        )
    for train_dates, validation_dates in splits:
        train = outer_train[outer_train["date"].isin(set(train_dates))]
        validation = outer_train[
            outer_train["date"].isin(set(validation_dates))
        ]
        if len(train) < 1_000 or len(validation) < 100:
            return np.array([], dtype=float), np.array([], dtype=int)
        if train[target].sum() < 20:
            return np.array([], dtype=float), np.array([], dtype=int)
        raw, _ = _fit_predict(train, validation, features, target)
        scores.append(raw)
        labels.append(validation[target].to_numpy(dtype=int))
    if not scores:
        return np.array([], dtype=float), np.array([], dtype=int)
    return np.concatenate(scores), np.concatenate(labels)


def _fit_isotonic(raw: np.ndarray, labels: np.ndarray):
    if (
        len(raw) < 500
        or int(labels.sum()) < 20
        or len(np.unique(raw)) < 2
        or len(np.unique(labels)) < 2
    ):
        return None
    model = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        increasing=True,
        out_of_bounds="clip",
    )
    model.fit(raw, labels)
    return model


def _apply_isotonic(model, raw: np.ndarray, base_rate: float) -> np.ndarray:
    if model is None:
        return np.full(len(raw), base_rate, dtype=float)
    return np.asarray(model.predict(raw), dtype=float)


def _frozen_bucket_map(
    raw_train: np.ndarray,
    y_train: np.ndarray,
    raw_test: np.ndarray,
) -> np.ndarray:
    """Reproduce the old in-sample, potentially non-monotone mapping pattern."""
    frame = pd.DataFrame({"score": raw_train, "target": y_train}).dropna()
    base = float(frame["target"].mean())
    if len(frame) < 150 or int(frame["target"].sum()) < 5:
        return np.full(len(raw_test), base, dtype=float)
    try:
        frame["bucket"] = pd.qcut(
            frame["score"].rank(method="first"),
            10,
            labels=False,
            duplicates="drop",
        )
    except ValueError:
        return np.full(len(raw_test), base, dtype=float)
    grouped = frame.groupby("bucket").agg(
        upper=("score", "max"),
        hit=("target", "mean"),
    )
    edges = grouped["upper"].to_numpy(dtype=float)
    hits = grouped["hit"].to_dict()
    bucket = np.clip(
        np.searchsorted(edges, raw_test, side="left"),
        0,
        len(edges) - 1,
    )
    return np.asarray(
        [hits.get(int(index), base) for index in bucket],
        dtype=float,
    )


def _predict_split(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    *,
    scope: str,
    fold: int,
) -> tuple[pd.DataFrame, list[dict]]:
    if train.empty or test.empty:
        raise RuntimeError(f"{scope}/{fold}: empty train or test")
    if train["date"].max() >= test["date"].min():
        raise RuntimeError(f"{scope}/{fold}: train/test chronology violated")
    test_counts = test.groupby("date").size()
    if not (test_counts == UNIVERSE_TOP_N).all():
        raise RuntimeError(f"{scope}/{fold}: test is not exact Top100")
    keep = [
        "date",
        "market",
        "regime",
        "history_prior_bars",
        "f_qv_rank",
        "f_atr_pct_14",
        "f_atr_xs_decile",
        "vol_band",
        "up_high_ret",
        "down_low_ret",
        "eod_ret",
        "safe_up10",
        "up10",
        "dn5",
    ]
    result = test[keep].copy()
    result["scope"] = scope
    result["fold"] = fold
    metadata: list[dict] = []

    for target in TARGETS:
        inner_raw, inner_y = _inner_oof(train, features, target)
        calibrator = _fit_isotonic(inner_raw, inner_y)
        raw_test, raw_train = _fit_predict(
            train,
            test,
            features,
            target,
            return_train=True,
        )
        if raw_train is None:
            raise RuntimeError(
                f"{target} requires in-sample scores for frozen calibration"
            )
        base_rate = float(train[target].mean())
        result[f"raw_{target}"] = raw_test
        result[f"p_repaired_{target}"] = _apply_isotonic(
            calibrator,
            raw_test,
            base_rate,
        )
        ordered_y = train.sort_values(["date", "market"])[target].to_numpy(
            dtype=int
        )
        result[f"p_frozen_{target}"] = _frozen_bucket_map(
            raw_train,
            ordered_y,
            raw_test,
        )
        metadata.append(
            {
                "scope": scope,
                "fold": fold,
                "target": target,
                "train_start": str(train["date"].min()),
                "train_end": str(train["date"].max()),
                "test_start": str(test["date"].min()),
                "test_end": str(test["date"].max()),
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "train_dates": int(train["date"].nunique()),
                "test_dates": int(test["date"].nunique()),
                "train_positives": int(train[target].sum()),
                "inner_oof_rows": int(len(inner_raw)),
                "inner_oof_positives": int(inner_y.sum()),
                "isotonic_fitted": calibrator is not None,
                "embargo_dates": EMBARGO_DATES,
            }
        )
    result["score_safeup_head"] = result["raw_safe_up10"]
    result["score_up10_control"] = result["raw_up10"]
    result["score_R1_repaired"] = (
        result["p_repaired_up10"]
        / np.maximum(result["p_repaired_dn5"], RR_EPS)
    )
    result["score_R1_frozen_pattern"] = (
        result["p_frozen_up10"]
        / np.maximum(result["p_frozen_dn5"], RR_EPS)
    )
    result["score_monkey_seed42"] = [
        int.from_bytes(
            hashlib.sha256(
                f"{MODEL_SEED}|{date}|{market}".encode("utf-8")
            ).digest()[:8],
            "big",
        )
        for date, market in zip(result["date"], result["market"])
    ]
    safe_rank = result.groupby("date")["raw_safe_up10"].rank(
        pct=True,
        method="average",
    )
    downside_rank = result.groupby("date")["raw_dn5"].rank(
        pct=True,
        method="average",
    )
    result["score_safeup_pareto_rank"] = safe_rank - downside_rank
    score_columns = [
        column
        for column in result
        if column.startswith(("raw_", "p_repaired_", "p_frozen_", "score_"))
    ]
    if not np.isfinite(result[score_columns].to_numpy(dtype=float)).all():
        raise RuntimeError(f"{scope}/{fold}: model produced nonfinite scores")
    probability_columns = [
        column
        for column in result
        if column.startswith(("p_repaired_", "p_frozen_"))
    ]
    if not result[probability_columns].apply(
        lambda values: values.between(0.0, 1.0, inclusive="both").all()
    ).all():
        raise RuntimeError(f"{scope}/{fold}: probability outside [0,1]")
    return result, metadata


def run_discovery(
    panel: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, list[dict]]:
    predictions = []
    metadata = []
    splits = _outer_splits(panel["date"])
    if len(splits) != OUTER_FOLDS:
        raise RuntimeError(
            f"expected {OUTER_FOLDS} outer folds, got {len(splits)}"
        )
    for fold, (train_dates, test_dates) in enumerate(splits):
        train = panel[panel["date"].isin(set(train_dates))].copy()
        test = panel[panel["date"].isin(set(test_dates))].copy()
        log.info(
            "discovery fold=%d train=%s..%s test=%s..%s",
            fold,
            train["date"].min(),
            train["date"].max(),
            test["date"].min(),
            test["date"].max(),
        )
        predicted, fold_meta = _predict_split(
            train,
            test,
            features,
            scope="discovery_oof",
            fold=fold,
        )
        predictions.append(predicted)
        metadata.extend(fold_meta)
    if not predictions:
        raise RuntimeError("no discovery OOF predictions")
    return pd.concat(predictions, ignore_index=True), metadata


def run_locked_holdout(
    panel: pd.DataFrame,
    holdout_dates: np.ndarray,
    features: list[str],
) -> tuple[pd.DataFrame, list[dict]]:
    ordered_dates = sorted(panel["date"].unique())
    holdout_start = min(holdout_dates)
    start_index = ordered_dates.index(holdout_start)
    train_end = start_index - EMBARGO_DATES
    if train_end <= 0:
        raise RuntimeError("not enough dates before locked holdout embargo")
    train_dates = set(ordered_dates[:train_end])
    embargoed_dates = ordered_dates[train_end:start_index]
    train = panel[panel["date"].isin(train_dates)].copy()
    test = panel[panel["date"].isin(set(holdout_dates))].copy()
    if train.empty or test.empty:
        raise RuntimeError("locked train/test split is empty")
    log.info(
        "unlocking holdout once: train=%s..%s test=%s..%s",
        train["date"].min(),
        train["date"].max(),
        test["date"].min(),
        test["date"].max(),
    )
    result, metadata = _predict_split(
        train,
        test,
        features,
        scope="locked_holdout",
        fold=-1,
    )
    for item in metadata:
        item["embargoed_dates"] = [str(value) for value in embargoed_dates]
    return result, metadata


def benchmark_complete_schedule_dates(
    panel: pd.DataFrame,
    m15_db: Path,
) -> tuple[list, dict]:
    """Return the exact closed BTC path axis overlapping target history."""
    with _connect_readonly(m15_db) as connection:
        first_nonbenchmark = connection.execute(
            """
            SELECT MIN(timestamp) FROM candles
            WHERE market != ?
            """,
            (BENCHMARK_MARKET,),
        ).fetchone()[0]
    if first_nonbenchmark is None:
        raise RuntimeError("15m DB has no non-benchmark history")
    first_path_date = pd.Timestamp(first_nonbenchmark).date()
    candidate_dates = sorted(
        date
        for date in panel["date"].unique()
        if date >= first_path_date
    )
    pairs = pd.DataFrame(
        {
            "market": [BENCHMARK_MARKET] * len(candidate_dates),
            "date": candidate_dates,
        }
    )
    paths, canonical_checked, complete_checked = _bulk_execution_paths(
        pairs,
        m15_db,
    )
    complete_dates = [
        date
        for date in candidate_dates
        if paths[(BENCHMARK_MARKET, date)].complete
    ]
    complete_set = set(complete_dates)
    if len(complete_dates) <= LOCKED_HOLDOUT_DATES + 60:
        raise RuntimeError("insufficient exact benchmark path dates")
    return complete_dates, {
        "first_nonbenchmark_15m_date": str(first_path_date),
        "candidate_dates": len(candidate_dates),
        "benchmark_complete_dates": len(complete_dates),
        "benchmark_incomplete_dates": (
            len(candidate_dates) - len(complete_dates)
        ),
        "benchmark_incomplete_date_values": [
            str(date)
            for date in candidate_dates
            if date not in complete_set
        ],
        "canonical_crosscheck_n": int(canonical_checked),
        "canonical_complete_crosscheck_n": int(complete_checked),
    }


def run_common_schedule_baseline(
    panel: pd.DataFrame,
    features: list[str],
    benchmark_complete_dates: list,
) -> tuple[pd.DataFrame, dict, list[dict]]:
    """Refit SafeUp/R1 on the exact schedule consumed by FP/semivol."""
    (
        schedule,
        discovery_splits,
        holdout_train_dates,
        holdout_dates,
    ) = build_common_benchmark_schedule(benchmark_complete_dates)
    schedule_hash = str(schedule["split_schedule_sha256"])
    predictions = []
    metadata: list[dict] = []
    for fold, (train_dates, test_dates) in enumerate(discovery_splits):
        train_cutoff = max(train_dates)
        train = panel[panel["date"].isin(set(train_dates))].copy()
        test = panel[panel["date"].isin(set(test_dates))].copy()
        predicted, fold_meta = _predict_split(
            train,
            test,
            features,
            scope="discovery_oof",
            fold=fold,
        )
        predicted["split_schedule_sha256"] = schedule_hash
        predictions.append(predicted)
        for item in fold_meta:
            item.update(
                split_schedule_sha256=schedule_hash,
                common_train_end=str(train_cutoff),
                common_train_dates=int(len(train_dates)),
            )
        metadata.extend(fold_meta)
    holdout_train_cutoff = max(holdout_train_dates)
    holdout_train = panel[
        panel["date"].isin(set(holdout_train_dates))
    ].copy()
    holdout_test = panel[
        panel["date"].isin(set(holdout_dates))
    ].copy()
    holdout_prediction, holdout_meta = _predict_split(
        holdout_train,
        holdout_test,
        features,
        scope="locked_holdout",
        fold=-1,
    )
    holdout_prediction["split_schedule_sha256"] = schedule_hash
    predictions.append(holdout_prediction)
    for item in holdout_meta:
        item.update(
            split_schedule_sha256=schedule_hash,
            common_train_end=str(holdout_train_cutoff),
            common_train_dates=int(len(holdout_train_dates)),
            embargoed_dates=schedule["folds"][-1]["embargo_dates"],
        )
    metadata.extend(holdout_meta)
    output = pd.concat(predictions, ignore_index=True)
    identity = [
        "split_schedule_sha256",
        "scope",
        "fold",
        "date",
        "market",
    ]
    if output.duplicated(identity).any():
        raise RuntimeError("shared baseline contains duplicate identities")
    counts = output.groupby(
        ["split_schedule_sha256", "scope", "fold", "date"]
    ).size()
    expected_date_keys = {
        (
            schedule_hash,
            str(record["scope"]),
            int(record["fold"]),
            pd.Timestamp(date).date(),
        )
        for record in schedule["folds"]
        for date in record["test_dates"]
    }
    if (
        set(counts.index) != expected_date_keys
        or not (counts == UNIVERSE_TOP_N).all()
    ):
        raise RuntimeError("shared baseline schedule coverage mismatch")
    if not output["market"].astype("string").str.fullmatch(
        r"KRW-[A-Z0-9]+", na=False
    ).all():
        raise RuntimeError("shared baseline contains invalid markets")
    return output, schedule, metadata


def _safe_auc(labels: pd.Series, score: pd.Series) -> float:
    valid = labels.notna() & score.notna()
    if valid.sum() < 2 or labels[valid].nunique() < 2:
        return np.nan
    return float(roc_auc_score(labels[valid], score[valid]))


def head_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, frame in predictions.groupby("scope", sort=False):
        for target in TARGETS:
            for calibration in ("repaired", "frozen"):
                probability = frame[f"p_{calibration}_{target}"]
                rows.append(
                    {
                        "scope": scope,
                        "target": target,
                        "calibration": calibration,
                        "n_rows": int(len(frame)),
                        "n_dates": int(frame["date"].nunique()),
                        "base_rate": float(frame[target].mean()),
                        "raw_auc": _safe_auc(
                            frame[target],
                            frame[f"raw_{target}"],
                        ),
                        "probability_auc": _safe_auc(
                            frame[target],
                            probability,
                        ),
                        "brier": float(
                            brier_score_loss(frame[target], probability)
                        ),
                        "mean_probability": float(probability.mean()),
                    }
                )
    return pd.DataFrame(rows)


def select_top3(predictions: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for policy in POLICIES:
        score = f"score_{policy}"
        picks = (
            predictions.sort_values(
                ["scope", "date", score, "market"],
                ascending=[True, True, False, True],
            )
            .groupby(["scope", "date"], sort=False)
            .head(TOP_K)
            .copy()
        )
        picks["policy"] = policy
        picks["selection_rank"] = picks.groupby(
            ["scope", "date"],
            sort=False,
        ).cumcount() + 1
        selected.append(picks)
    out = pd.concat(selected, ignore_index=True)
    counts = out.groupby(["scope", "date", "policy"]).size()
    if not (counts == TOP_K).all():
        raise RuntimeError("every policy/date must select exactly Top 3")
    if out.duplicated(["scope", "date", "policy", "market"]).any():
        raise RuntimeError("duplicate selected market")
    return out


def _execution_window(date: object) -> tuple[pd.Timestamp, pd.DatetimeIndex]:
    """First executable full bar after the normal 09:10 delivery."""
    start = (
        pd.Timestamp(date).normalize()
        + pd.Timedelta(hours=9, minutes=15)
    )
    expected = pd.date_range(
        start=start,
        periods=EXPECTED_BARS,
        freq=BAR_FREQ,
    )
    return start, expected


def _bulk_execution_paths(
    pairs: pd.DataFrame,
    db_path: Path,
) -> tuple[dict[tuple[str, object], BulkPath], int, int]:
    """Batch-equivalent of ``assess_15m_window(date 09:15)``.

    A persistent connection avoids repeatedly opening the multi-million-row
    SQLite database.  The first deterministic 20 pairs are then compared
    byte-for-byte on bars and metadata with the canonical helper.
    """
    pairs = pairs[["market", "date"]].drop_duplicates().sort_values(
        ["date", "market"]
    )
    results: dict[tuple[str, object], BulkPath] = {}
    with _connect_readonly(db_path) as connection:
        horizon = connection.execute(
            "SELECT MIN(timestamp), MAX(timestamp) FROM candles WHERE market=?",
            (BENCHMARK_MARKET,),
        ).fetchone()
        benchmark_start = (
            pd.Timestamp(horizon[0]) if horizon and horizon[0] else None
        )
        benchmark_end = (
            pd.Timestamp(horizon[1]) if horizon and horizon[1] else None
        )

        benchmark_cache: dict[object, tuple[bool, str, int]] = {}
        for date in sorted(pairs["date"].unique()):
            start, expected = _execution_window(date)
            end = start + BAR_FREQ * EXPECTED_BARS
            rows = connection.execute(
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
                state = (
                    False,
                    "db_horizon_start_incomplete",
                    len(rows),
                )
            elif benchmark_end < end:
                state = (
                    False,
                    "db_horizon_end_incomplete",
                    len(rows),
                )
            elif any(ts not in expected_set for ts in timestamps):
                state = (False, "benchmark_off_grid", len(rows))
            elif (
                len(timestamps) != EXPECTED_BARS
                or len(set(timestamps)) != len(timestamps)
                or set(timestamps) != expected_set
            ):
                state = (False, "benchmark_gap", len(rows))
            else:
                state = (True, "complete", len(rows))
            benchmark_cache[date] = state

        for pair in pairs.itertuples(index=False):
            market = str(pair.market)
            date = pair.date
            benchmark_ok, benchmark_quality, benchmark_bars = (
                benchmark_cache[date]
            )
            if not benchmark_ok:
                results[(market, date)] = BulkPath(
                    (),
                    False,
                    benchmark_quality,
                    0,
                    0,
                    benchmark_bars,
                )
                continue

            start, expected = _execution_window(date)
            end = start + BAR_FREQ * EXPECTED_BARS
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
            expected_set = set(expected)
            by_timestamp: dict[
                pd.Timestamp, tuple[float, float, float, float]
            ] = {}
            invalid_quality = ""
            for timestamp, open_, high, low, close in rows:
                timestamp = pd.Timestamp(timestamp)
                if (
                    timestamp not in expected_set
                    or timestamp in by_timestamp
                ):
                    invalid_quality = "target_off_grid"
                    break
                bar: tuple[float, float, float, float] = (
                    float(open_),
                    float(high),
                    float(low),
                    float(close),
                )
                if not _valid_bar(bar):
                    invalid_quality = "invalid_target_ohlc"
                    break
                by_timestamp[timestamp] = bar
            if invalid_quality:
                results[(market, date)] = BulkPath(
                    (),
                    False,
                    invalid_quality,
                    len(rows),
                    0,
                    benchmark_bars,
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
                prior = connection.execute(
                    """
                    SELECT close FROM candles
                    WHERE market=? AND timestamp<?
                    ORDER BY timestamp DESC LIMIT 1
                    """,
                    (
                        market,
                        start.strftime("%Y-%m-%d %H:%M:%S"),
                    ),
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

    sample_keys = list(
        pairs.head(10).itertuples(index=False, name=None)
    )
    complete_keys = [
        key for key, path in results.items() if path.complete
    ][:10]
    sample_keys.extend(
        key for key in complete_keys if key not in sample_keys
    )
    checked = 0
    checked_complete = 0
    for market, date in sample_keys:
        market = str(market)
        start, _ = _execution_window(date)
        canonical = assess_15m_window(
            market,
            start,
            db_path=db_path,
        )
        batched = results[(market, date)]
        if (
            canonical.path_complete != batched.complete
            or canonical.path_quality != batched.quality
            or canonical.raw_bars != batched.raw_bars
            or canonical.flat_filled_bars != batched.flat_filled_bars
            or canonical.benchmark_bars != batched.benchmark_bars
        ):
            raise RuntimeError(
                f"bulk/canonical 09:15 metadata mismatch: {(market, date)}"
            )
        if canonical.path_complete and tuple(canonical.bars) != batched.bars:
            raise RuntimeError(
                f"bulk/canonical 09:15 bars mismatch: {(market, date)}"
            )
        checked_complete += int(canonical.path_complete)
        checked += 1
    return results, checked, checked_complete


def attach_paths(
    picks: pd.DataFrame,
    db_path: Path,
) -> tuple[pd.DataFrame, dict]:
    pairs = picks[["market", "date"]].drop_duplicates().copy()
    (
        paths,
        canonical_crosscheck_n,
        canonical_complete_crosscheck_n,
    ) = _bulk_execution_paths(pairs, db_path)
    rows = []
    for pair in pairs.itertuples(index=False):
        path = paths[(str(pair.market), pair.date)]
        row = {
            "market": str(pair.market),
            "date": pair.date,
            "path_complete": bool(path.complete),
            "path_quality": path.quality,
            "path_raw_bars": int(path.raw_bars),
            "path_flat_filled_bars": int(path.flat_filled_bars),
            "path_benchmark_bars": int(path.benchmark_bars),
        }
        if path.complete:
            bars = list(path.bars)
            gross, outcome = simulate_path(
                bars,
                HARD_STOP,
                TAKE_PROFIT,
                None,
            )
            eod_gross, _ = simulate_path(bars, None, None, None)
            entry = float(bars[0][0])
            row.update(
                path_outcome=outcome,
                path_net=float(gross - ROUND_TRIP_COST),
                path_eod_net=float(eod_gross - ROUND_TRIP_COST),
                path_mfe=float(max(bar[1] for bar in bars) / entry - 1.0),
                path_mae=float(min(bar[2] for bar in bars) / entry - 1.0),
                path_up10_before_dn5=bool(
                    _safe_up10_before_dn5(path.bars)
                ),
            )
            row["path_up10"] = bool(row["path_mfe"] >= 0.10)
            row["path_dn5"] = bool(row["path_mae"] <= -0.05)
            row["path_safe_up10"] = bool(
                row["path_up10"] and not row["path_dn5"]
            )
        rows.append(row)
    outcomes = pd.DataFrame(rows)
    merged = picks.merge(
        outcomes,
        on=["market", "date"],
        how="left",
        validate="many_to_one",
    )

    # Path comparisons use only dates for which every fixed policy has three
    # complete paths.  Candidate-dependent missingness cannot favor a policy.
    quality = (
        merged.groupby(["scope", "date", "policy"])
        .agg(n=("market", "size"), complete=("path_complete", "sum"))
        .reset_index()
    )
    good = quality[
        (quality["n"] == TOP_K) & (quality["complete"] == TOP_K)
    ]
    common = (
        good.groupby(["scope", "date"])["policy"]
        .nunique()
        .rename("policy_n")
        .reset_index()
    )
    common_keys = set(
        map(
            tuple,
            common.loc[
                common["policy_n"] == len(POLICIES),
                ["scope", "date"],
            ].itertuples(index=False, name=None),
        )
    )
    merged["path_common_complete_date"] = [
        (scope, date) in common_keys
        for scope, date in zip(merged["scope"], merged["date"])
    ]
    meta = {
        "wanted_unique_pairs": int(len(pairs)),
        "complete_unique_pairs": int(outcomes["path_complete"].sum()),
        "complete_pair_rate": float(outcomes["path_complete"].mean()),
        "canonical_assess_15m_window_crosscheck_n": (
            canonical_crosscheck_n
        ),
        "canonical_complete_path_crosscheck_n": (
            canonical_complete_crosscheck_n
        ),
        "common_complete_dates": {
            scope: int(
                sum(1 for key in common_keys if key[0] == scope)
            )
            for scope in predictions_scopes(merged)
        },
        "rule": (
            "[D 09:15,D+1 09:15), KRW-BTC exact 96 bars and closed next "
            "boundary; target-only gaps flat-filled; same-bar SL first"
        ),
    }
    return merged, meta


def predictions_scopes(frame: pd.DataFrame) -> list[str]:
    return sorted(map(str, frame["scope"].dropna().unique()))


def _matched_atr_values(
    reference: pd.DataFrame,
    picks: pd.DataFrame,
) -> dict[str, float]:
    strata = (
        reference.groupby(["date", "vol_band"])[
            ["safe_up10", "up10", "dn5"]
        ]
        .mean()
        .reset_index()
    )
    matched = picks.merge(
        strata,
        on=["date", "vol_band"],
        how="left",
        suffixes=("", "_matched"),
        validate="many_to_one",
    )
    return {
        target: float(matched[f"{target}_matched"].mean())
        for target in ("safe_up10", "up10", "dn5")
    }


def _portfolio_path_metrics(path: pd.DataFrame) -> dict:
    if path.empty:
        return {
            "path_n": 0,
            "path_dates": 0,
            "path_net_mean": np.nan,
            "path_tp_first": np.nan,
            "path_sl_first": np.nan,
            "path_eod": np.nan,
            "path_hit": np.nan,
            "path_up10_before_dn5": np.nan,
            "path_safe_up10": np.nan,
            "path_up10": np.nan,
            "path_dn5": np.nan,
            "path_cumulative": np.nan,
            "path_max_drawdown": np.nan,
            "path_net_ci95_lo": np.nan,
            "path_net_ci95_hi": np.nan,
            "path_net_p_gt_zero": np.nan,
            "path_sharpe_ann": np.nan,
            "path_sortino_ann": np.nan,
            "path_calmar": np.nan,
        }
    daily = path.groupby("date")["path_net"].mean().sort_index()
    canonical = summarize_daily(daily)
    cumulative = float(canonical["cumulative_return"])
    max_drawdown = float(canonical["max_drawdown"])
    annualized_return = (
        (1.0 + cumulative) ** (365.0 / len(daily)) - 1.0
        if cumulative > -1.0
        else -1.0
    )
    calmar = (
        annualized_return / abs(max_drawdown)
        if max_drawdown < 0
        else np.nan
    )
    daily_values = daily.to_numpy(dtype=float)
    rng = np.random.default_rng(MODEL_SEED)
    indices = rng.integers(
        0,
        len(daily_values),
        size=(BOOTSTRAP_DRAWS, len(daily_values)),
    )
    bootstrap = daily_values[indices].mean(axis=1)
    return {
        "path_n": int(len(path)),
        "path_dates": int(path["date"].nunique()),
        "path_net_mean": float(path["path_net"].mean()),
        "path_tp_first": float((path["path_outcome"] == "tp").mean()),
        "path_sl_first": float((path["path_outcome"] == "sl").mean()),
        "path_eod": float((path["path_outcome"] == "eod").mean()),
        "path_hit": float((path["path_net"] > 0).mean()),
        "path_up10_before_dn5": float(
            path["path_up10_before_dn5"].mean()
        ),
        "path_safe_up10": float(path["path_safe_up10"].mean()),
        "path_up10": float(path["path_up10"].mean()),
        "path_dn5": float(path["path_dn5"].mean()),
        "path_cumulative": cumulative,
        "path_max_drawdown": max_drawdown,
        "path_net_ci95_lo": float(np.percentile(bootstrap, 2.5)),
        "path_net_ci95_hi": float(np.percentile(bootstrap, 97.5)),
        "path_net_p_gt_zero": float((bootstrap > 0).mean()),
        "path_sharpe_ann": float(canonical["sharpe_ann"]),
        "path_sortino_ann": float(canonical["sortino_ann"]),
        "path_calmar": float(calmar),
    }


def policy_metrics(
    predictions: pd.DataFrame,
    picks: pd.DataFrame,
    *,
    by_fold: bool,
) -> pd.DataFrame:
    group_columns = ["scope", "policy"]
    if by_fold:
        group_columns.insert(1, "fold")
    rows = []
    for keys, group in picks.groupby(group_columns, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        identity = dict(zip(group_columns, keys))
        scope = str(identity["scope"])
        fold = identity.get("fold")
        reference = predictions[predictions["scope"] == scope]
        if fold is not None:
            reference = reference[reference["fold"] == fold]
        reference = reference[
            reference["date"].isin(set(group["date"]))
        ].copy()
        matched = _matched_atr_values(reference, group)
        safe_rate = float(group["safe_up10"].mean())
        up_rate = float(group["up10"].mean())
        dn_rate = float(group["dn5"].mean())
        row = {
            **identity,
            "n": int(len(group)),
            "dates": int(group["date"].nunique()),
            "safe_up10_rate": safe_rate,
            "up10_rate": up_rate,
            "dn5_rate": dn_rate,
            "eod_net_mean": float(
                (group["eod_ret"] - ROUND_TRIP_COST).mean()
            ),
            "mfe_mean": float(group["up_high_ret"].mean()),
            "mae_mean": float(group["down_low_ret"].mean()),
            "matched_atr_safe_up10": matched["safe_up10"],
            "safe_up10_lift_within_atr": (
                safe_rate / matched["safe_up10"]
                if matched["safe_up10"] > 0
                else np.nan
            ),
            "matched_atr_up10": matched["up10"],
            "up10_lift_within_atr": (
                up_rate / matched["up10"]
                if matched["up10"] > 0
                else np.nan
            ),
            "matched_atr_dn5": matched["dn5"],
            "dn5_delta_within_atr": dn_rate - matched["dn5"],
            "path_complete_pick_rate": float(
                group["path_complete"].mean()
            ),
        }
        common_path = group[
            group["path_common_complete_date"]
            & group["path_complete"]
        ].copy()
        row.update(_portfolio_path_metrics(common_path))
        rows.append(row)
    return pd.DataFrame(rows)


def _date_metric(
    frame: pd.DataFrame,
    metric: str,
) -> pd.Series:
    if metric == "safe_up10_rate":
        return frame.groupby("date")["safe_up10"].mean()
    if metric == "up10_rate":
        return frame.groupby("date")["up10"].mean()
    if metric == "dn5_rate":
        return frame.groupby("date")["dn5"].mean()
    if metric == "path_net_mean":
        if "path_net" not in frame.columns:
            return pd.Series(dtype=float)
        return (
            frame[
                frame["path_common_complete_date"]
                & frame["path_complete"]
            ]
            .groupby("date")["path_net"]
            .mean()
        )
    if metric == "path_sl_first":
        if "path_outcome" not in frame.columns:
            return pd.Series(dtype=float)
        complete = frame[
            frame["path_common_complete_date"]
            & frame["path_complete"]
        ].copy()
        complete["_sl"] = (complete["path_outcome"] == "sl").astype(float)
        return complete.groupby("date")["_sl"].mean()
    if metric in ("path_safe_up10", "path_up10", "path_dn5"):
        if metric not in frame.columns:
            return pd.Series(dtype=float)
        complete = frame[
            frame["path_common_complete_date"]
            & frame["path_complete"]
        ].copy()
        return complete.groupby("date")[metric].mean()
    raise KeyError(metric)


def paired_bootstrap(
    picks: pd.DataFrame,
    *,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for scope in predictions_scopes(picks):
        scoped = picks[picks["scope"] == scope]
        if (
            "path_common_complete_date" in scoped.columns
            and scoped["path_common_complete_date"].any()
        ):
            scoped = scoped[scoped["path_common_complete_date"]].copy()
        comparisons = {
            "safeup_head": [
                policy
                for policy in POLICIES
                if policy != "safeup_head"
            ],
            POSTHOC_POLICY: list(EX_ANTE_POLICIES),
        }
        for primary_policy, comparators in comparisons.items():
            primary = scoped[scoped["policy"] == primary_policy]
            for comparator in comparators:
                other = scoped[scoped["policy"] == comparator]
                for metric in (
                    "safe_up10_rate",
                    "up10_rate",
                    "dn5_rate",
                    "path_net_mean",
                    "path_sl_first",
                    "path_safe_up10",
                    "path_up10",
                    "path_dn5",
                ):
                    a = _date_metric(primary, metric)
                    b = _date_metric(other, metric)
                    common = a.index.intersection(b.index)
                    if len(common) == 0:
                        continue
                    delta = (
                        a.loc[common] - b.loc[common]
                    ).to_numpy(dtype=float)
                    salt_payload = (
                        f"{seed}|{scope}|{primary_policy}|"
                        f"{comparator}|{metric}"
                    ).encode("utf-8")
                    salt = int.from_bytes(
                        hashlib.sha256(salt_payload).digest()[:8],
                        "big",
                    )
                    rng = np.random.default_rng(salt)
                    indices = rng.integers(
                        0,
                        len(delta),
                        size=(draws, len(delta)),
                    )
                    boot = delta[indices].mean(axis=1)
                    lower_better = metric in (
                        "dn5_rate",
                        "path_sl_first",
                        "path_dn5",
                    )
                    rows.append(
                        {
                            "scope": scope,
                            "cohort": (
                                "common_0915_complete_dates"
                                if scoped[
                                    "path_common_complete_date"
                                ].any()
                                else "all_dates"
                            ),
                            "primary": primary_policy,
                            "comparator": comparator,
                            "metric": metric,
                            "n_paired_dates": int(len(delta)),
                            "delta_primary_minus_comparator": float(
                                delta.mean()
                            ),
                            "ci95_lo": float(
                                np.percentile(boot, 2.5)
                            ),
                            "ci95_hi": float(
                                np.percentile(boot, 97.5)
                            ),
                            "p_delta_gt_zero": float(
                                (boot > 0).mean()
                            ),
                            "p_preferred_direction": float(
                                (boot < 0).mean()
                                if lower_better
                                else (boot > 0).mean()
                            ),
                            "preferred_direction": (
                                "lower" if lower_better else "higher"
                            ),
                            "bootstrap_draws": int(draws),
                            "seed": int(seed),
                            "stream_seed": int(salt),
                        }
                    )
    return pd.DataFrame(rows)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        + "\n",
        encoding="utf-8",
    )


def _artifact_paths(prefix: Path) -> dict[str, Path]:
    return {
        "predictions": Path(f"{prefix}_predictions.csv.gz"),
        "picks": Path(f"{prefix}_picks.csv.gz"),
        "summary": Path(f"{prefix}_summary.csv"),
        "folds": Path(f"{prefix}_folds.csv"),
        "heads": Path(f"{prefix}_heads.csv"),
        "paired": Path(f"{prefix}_paired.csv"),
        "coverage": Path(f"{prefix}_coverage.json"),
    }


def _fp_schedule_artifact_paths(prefix: Path) -> dict[str, Path]:
    return {
        "predictions": Path(f"{prefix}_predictions.csv.gz"),
        "contract": Path(f"{prefix}_contract.json"),
        "coverage": Path(f"{prefix}_coverage.json"),
    }


def _output_prefix_from_predictions(path: Path) -> Path:
    suffix = "_predictions.csv.gz"
    if not path.name.endswith(suffix):
        raise RuntimeError(
            "safeup baseline must be a *_predictions.csv.gz artifact"
        )
    return path.with_name(path.name[: -len(suffix)])


def _verify_schedule_contract(contract: dict) -> str:
    if contract.get("schema") != FP_SCHEDULE_SCHEMA:
        raise RuntimeError("shared schedule schema mismatch")
    recorded_hash = contract.get("split_schedule_sha256")
    if not isinstance(recorded_hash, str):
        raise RuntimeError("shared schedule hash is missing")
    unhashed = dict(contract)
    unhashed.pop("split_schedule_sha256", None)
    current_hash = hashlib.sha256(
        json.dumps(
            unhashed,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if current_hash != recorded_hash:
        raise RuntimeError("shared schedule contract hash mismatch")
    if (
        contract.get("outer_folds") != OUTER_FOLDS
        or contract.get("embargo_eligible_dates") != EMBARGO_DATES
        or contract.get("historical_holdout_contaminated") is not True
        or contract.get("virgin_or_preregistered") is not False
        or contract.get("maximum_evidence_grade")
        != "historical_comparison_only"
    ):
        raise RuntimeError("shared schedule fixed contract is invalid")
    axes: dict[str, list] = {}
    for name in (
        "eligible_dates",
        "discovery_dates",
        "locked_holdout_dates",
    ):
        raw_dates = contract.get(name)
        if not isinstance(raw_dates, list) or not raw_dates:
            raise RuntimeError(f"shared schedule {name} is invalid")
        try:
            parsed = [
                pd.Timestamp(value).date() for value in raw_dates
            ]
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"shared schedule {name} contains invalid dates"
            ) from exc
        if (
            list(map(str, parsed)) != raw_dates
            or parsed != sorted(set(parsed))
        ):
            raise RuntimeError(
                f"shared schedule {name} is not canonical"
            )
        expected_axis_hash = contract.get(f"{name}_sha256")
        if expected_axis_hash != _dates_sha256(parsed):
            raise RuntimeError(
                f"shared schedule {name} hash mismatch"
            )
        axes[name] = parsed
    eligible = axes["eligible_dates"]
    discovery = axes["discovery_dates"]
    holdout = axes["locked_holdout_dates"]
    if (
        len(holdout) != LOCKED_HOLDOUT_DATES
        or discovery + holdout != eligible
        or holdout != eligible[-LOCKED_HOLDOUT_DATES:]
    ):
        raise RuntimeError("shared schedule final-180 partition is invalid")
    folds = contract.get("folds")
    if not isinstance(folds, list) or len(folds) != OUTER_FOLDS + 1:
        raise RuntimeError("shared schedule fold contract is incomplete")
    expected_keys = {
        ("discovery_oof", fold) for fold in range(OUTER_FOLDS)
    } | {("locked_holdout", -1)}
    observed_keys = {
        (str(record.get("scope")), int(record.get("fold", -999)))
        for record in folds
        if isinstance(record, dict)
    }
    if observed_keys != expected_keys:
        raise RuntimeError("shared schedule scope/fold set is invalid")
    expected_discovery_splits = _expanding_splits(
        discovery,
        n_folds=OUTER_FOLDS,
        embargo=EMBARGO_DATES,
        warmup_fraction=0.35,
        minimum_warmup=90,
    )
    records = {
        (str(record["scope"]), int(record["fold"])): record
        for record in folds
    }
    discovery_positions = {
        date: index for index, date in enumerate(discovery)
    }
    for fold, (train_dates, test_dates) in enumerate(
        expected_discovery_splits
    ):
        train = list(train_dates)
        test = list(test_dates)
        train_end_position = discovery_positions[train[-1]]
        test_start_position = discovery_positions[test[0]]
        embargo_dates = discovery[
            train_end_position + 1 : test_start_position
        ]
        expected_record = {
            "scope": "discovery_oof",
            "fold": fold,
            "train_dates": list(map(str, train)),
            "train_dates_sha256": _dates_sha256(train),
            "train_end": str(train[-1]),
            "embargo_dates": list(map(str, embargo_dates)),
            "test_dates": list(map(str, test)),
            "test_dates_sha256": _dates_sha256(test),
            "test_start": str(test[0]),
            "test_end": str(test[-1]),
        }
        if records[("discovery_oof", fold)] != expected_record:
            raise RuntimeError(
                f"shared schedule discovery fold {fold} is invalid"
            )
    holdout_train = discovery[:-EMBARGO_DATES]
    expected_holdout = {
        "scope": "locked_holdout",
        "fold": -1,
        "train_dates": list(map(str, holdout_train)),
        "train_dates_sha256": _dates_sha256(holdout_train),
        "train_end": str(holdout_train[-1]),
        "embargo_dates": list(map(str, discovery[-EMBARGO_DATES:])),
        "test_dates": list(map(str, holdout)),
        "test_dates_sha256": _dates_sha256(holdout),
        "test_start": str(holdout[0]),
        "test_end": str(holdout[-1]),
    }
    if records[("locked_holdout", -1)] != expected_holdout:
        raise RuntimeError("shared schedule locked holdout is invalid")
    return recorded_hash


def validate_fp_schedule_artifacts(
    *,
    output_prefix: Path,
    d1_db: Path,
    m15_db: Path,
) -> dict:
    """Validate the schedule-aligned SafeUp/R1 baseline generation."""
    expected = _fp_schedule_artifact_paths(output_prefix)
    manifest_path = Path(f"{output_prefix}_manifest.json")
    _verify_manifest(
        manifest_path=manifest_path,
        expected=expected,
        schema="safeup_head_challenger_v1_fp_schedule_manifest",
    )
    try:
        contract = json.loads(
            expected["contract"].read_text(encoding="utf-8")
        )
        coverage = json.loads(
            expected["coverage"].read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "shared baseline metadata is unreadable"
        ) from exc
    schedule_hash = _verify_schedule_contract(contract)
    if coverage.get("schema") != (
        "safeup_head_challenger_v1_fp_schedule"
    ):
        raise RuntimeError("shared baseline coverage schema mismatch")
    if coverage.get("split_schedule_sha256") != schedule_hash:
        raise RuntimeError("shared baseline coverage schedule mismatch")
    if (
        coverage.get("historical_holdout_contaminated") is not True
        or coverage.get("virgin_or_preregistered") is not False
        or coverage.get("maximum_evidence_grade")
        != "historical_comparison_only"
    ):
        raise RuntimeError("shared baseline evidence grade is invalid")
    inputs = coverage.get("inputs")
    if not isinstance(inputs, dict):
        raise RuntimeError("shared baseline input lineage is missing")
    if inputs.get("script_sha256") != _sha256(Path(__file__)):
        raise RuntimeError("shared baseline was built by stale code")
    if inputs.get("code_lineage") != _code_lineage():
        raise RuntimeError(
            "shared baseline local code dependencies have changed"
        )
    benchmark_axis = coverage.get("benchmark_axis")
    if (
        not isinstance(benchmark_axis, dict)
        or int(benchmark_axis.get("benchmark_complete_dates", -1))
        != len(contract["eligible_dates"])
        or int(benchmark_axis.get("candidate_dates", -1))
        != int(benchmark_axis.get("benchmark_complete_dates", -1))
        + int(benchmark_axis.get("benchmark_incomplete_dates", -1))
    ):
        raise RuntimeError("shared baseline benchmark-axis audit is invalid")
    current_signatures = {
        "d1_db": _file_signature(d1_db),
        "m15_db": _file_signature(m15_db),
    }
    if inputs.get("source_signatures") != current_signatures:
        raise RuntimeError("shared baseline source inputs have changed")
    model_contract = coverage.get("model_contract")
    if not isinstance(model_contract, dict):
        raise RuntimeError("shared baseline model contract is missing")
    model_metadata = model_contract.get("model_metadata")
    if not isinstance(model_metadata, list):
        raise RuntimeError("shared baseline model metadata is missing")
    features = model_contract.get("features")
    if (
        model_contract.get("targets") != list(TARGETS)
        or not isinstance(features, list)
        or len(features) != 24
        or len(set(map(str, features))) != 24
    ):
        raise RuntimeError("shared baseline model contract is invalid")
    fold_contracts = {
        (str(record["scope"]), int(record["fold"])): record
        for record in contract["folds"]
    }
    metadata_keys = []
    for record in model_metadata:
        if not isinstance(record, dict):
            raise RuntimeError("shared baseline model metadata is malformed")
        key = (str(record.get("scope")), int(record.get("fold", -999)))
        target = str(record.get("target"))
        metadata_keys.append((*key, target))
        fold_contract = fold_contracts.get(key)
        if (
            fold_contract is None
            or record.get("split_schedule_sha256") != schedule_hash
            or record.get("common_train_end")
            != fold_contract["train_end"]
            or int(record.get("common_train_dates", -1))
            != len(fold_contract["train_dates"])
            or int(record.get("train_dates", -1))
            != len(fold_contract["train_dates"])
            or int(record.get("test_dates", -1))
            != len(fold_contract["test_dates"])
            or record.get("train_end") != fold_contract["train_end"]
            or record.get("test_start") != fold_contract["test_start"]
            or record.get("test_end") != fold_contract["test_end"]
        ):
            raise RuntimeError(
                "shared baseline model/schedule provenance mismatch"
            )
    expected_metadata_keys = {
        (scope, fold, target)
        for scope, fold in fold_contracts
        for target in TARGETS
    }
    if set(metadata_keys) != expected_metadata_keys or len(
        metadata_keys
    ) != len(expected_metadata_keys):
        raise RuntimeError(
            "shared baseline model metadata set is incomplete"
        )
    required = [
        "split_schedule_sha256",
        "scope",
        "fold",
        "date",
        "market",
        "score_R1_repaired",
        "score_safeup_head",
    ]
    predictions = pd.read_csv(
        expected["predictions"],
        usecols=required,
        float_precision="round_trip",
    )
    predictions["date"] = pd.to_datetime(
        predictions["date"], errors="raise"
    ).dt.date
    if set(predictions["split_schedule_sha256"].astype(str)) != {
        schedule_hash
    }:
        raise RuntimeError("shared baseline row schedule hash mismatch")
    identity = [
        "split_schedule_sha256",
        "scope",
        "fold",
        "date",
        "market",
    ]
    if predictions.duplicated(identity).any():
        raise RuntimeError("shared baseline has duplicate identities")
    if not predictions["market"].astype("string").str.fullmatch(
        r"KRW-[A-Z0-9]+", na=False
    ).all():
        raise RuntimeError("shared baseline contains invalid markets")
    scores = predictions[
        ["score_R1_repaired", "score_safeup_head"]
    ].apply(pd.to_numeric, errors="coerce")
    if scores.isna().any().any() or not np.isfinite(
        scores.to_numpy()
    ).all():
        raise RuntimeError("shared baseline contains nonfinite scores")
    expected_date_keys = {
        (
            str(record["scope"]),
            int(record["fold"]),
            pd.Timestamp(date).date(),
        )
        for record in contract["folds"]
        for date in record["test_dates"]
    }
    observed_counts = predictions.groupby(
        ["scope", "fold", "date"]
    ).size()
    observed_date_keys = set(observed_counts.index)
    if (
        observed_date_keys != expected_date_keys
        or not (observed_counts == UNIVERSE_TOP_N).all()
    ):
        raise RuntimeError(
            "shared baseline does not exactly cover its schedule"
        )
    return {
        "status": "valid",
        "schema": coverage["schema"],
        "output_prefix": str(output_prefix),
        "split_schedule_sha256": schedule_hash,
        "eligible_dates_sha256": contract[
            "eligible_dates_sha256"
        ],
        "locked_holdout_dates_sha256": contract[
            "locked_holdout_dates_sha256"
        ],
        "locked_holdout_dates": len(
            contract["locked_holdout_dates"]
        ),
        "locked_holdout_start": contract[
            "locked_holdout_dates"
        ][0],
        "locked_holdout_end": contract[
            "locked_holdout_dates"
        ][-1],
        "historical_holdout_contaminated": True,
        "manifest_sha256": _sha256(manifest_path),
    }


def _verify_manifest(
    *,
    manifest_path: Path,
    expected: dict[str, Path],
    schema: str = "safeup_head_challenger_v1_manifest",
) -> dict:
    if not manifest_path.is_file():
        raise RuntimeError(f"artifact manifest is missing: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("safeup artifact manifest is unreadable") from exc
    if payload.get("schema") != schema:
        raise RuntimeError("safeup artifact manifest schema mismatch")
    files = payload.get("files")
    if not isinstance(files, dict) or set(files) != set(expected):
        raise RuntimeError("safeup artifact manifest file set mismatch")
    for name, expected_path in expected.items():
        entry = files.get(name)
        if not isinstance(entry, dict):
            raise RuntimeError(f"safeup manifest entry is invalid: {name}")
        recorded_path = Path(str(entry.get("path", "")))
        if not recorded_path.is_absolute():
            recorded_path = ROOT / recorded_path
        if recorded_path.resolve() != expected_path.resolve():
            raise RuntimeError(f"safeup manifest path mismatch: {name}")
        if not expected_path.is_file():
            raise RuntimeError(f"safeup artifact is missing: {expected_path}")
        if int(entry.get("bytes", -1)) != expected_path.stat().st_size:
            raise RuntimeError(f"safeup artifact size mismatch: {name}")
        if entry.get("sha256") != _sha256(expected_path):
            raise RuntimeError(f"safeup artifact checksum mismatch: {name}")
    return payload


def _prediction_scope_contract(
    predictions: pd.DataFrame,
    locked_holdout: dict,
) -> dict:
    required = {"scope", "date", "market"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise RuntimeError(
            f"safeup prediction identity columns missing: {missing}"
        )
    identity = predictions[list(required)].copy()
    identity["date"] = pd.to_datetime(
        identity["date"], errors="raise"
    ).dt.date
    if identity["date"].isna().any():
        raise RuntimeError("safeup predictions contain missing dates")
    if not identity["market"].astype("string").str.fullmatch(
        r"KRW-[A-Z0-9]+", na=False
    ).all():
        raise RuntimeError("safeup predictions contain invalid markets")
    expected_scopes = {"discovery_oof", "locked_holdout"}
    observed_scopes = set(identity["scope"].astype(str).unique())
    if observed_scopes != expected_scopes:
        raise RuntimeError(
            "safeup prediction scopes are incomplete: "
            f"{sorted(observed_scopes)}"
        )
    if identity.duplicated(["scope", "date", "market"]).any():
        raise RuntimeError("safeup predictions contain duplicate identities")
    scope_per_date = identity.groupby("date")["scope"].nunique()
    if not (scope_per_date == 1).all():
        raise RuntimeError("safeup prediction scope changes within a date")
    counts = identity.groupby(["scope", "date"]).size()
    if not (counts == UNIVERSE_TOP_N).all():
        raise RuntimeError("safeup predictions are not exact Top100")
    discovery_dates = sorted(
        identity.loc[
            identity["scope"] == "discovery_oof", "date"
        ].unique()
    )
    holdout_dates = sorted(
        identity.loc[
            identity["scope"] == "locked_holdout", "date"
        ].unique()
    )
    if not discovery_dates or not holdout_dates:
        raise RuntimeError("safeup prediction date scope is empty")
    if discovery_dates[-1] >= holdout_dates[0]:
        raise RuntimeError("safeup discovery/holdout chronology is invalid")
    prediction_dates = discovery_dates + holdout_dates
    internal_gaps = [
        (previous, current)
        for previous, current in zip(
            prediction_dates, prediction_dates[1:]
        )
        if (current - previous).days != 1
    ]
    if internal_gaps:
        raise RuntimeError(
            "safeup prediction OOF coverage has internal date gaps: "
            f"{internal_gaps[:5]}"
        )
    holdout_hash = hashlib.sha256(
        "\n".join(map(str, holdout_dates)).encode("utf-8")
    ).hexdigest()
    expected_holdout = {
        "n_dates": len(holdout_dates),
        "start": str(holdout_dates[0]),
        "end": str(holdout_dates[-1]),
        "dates_sha256": holdout_hash,
    }
    recorded_holdout = {
        key: locked_holdout.get(key) for key in expected_holdout
    }
    if recorded_holdout != expected_holdout:
        raise RuntimeError(
            "safeup prediction/coverage holdout mismatch: "
            f"{recorded_holdout} != {expected_holdout}"
        )
    return {
        "discovery_oof_dates": len(discovery_dates),
        "discovery_oof_start": str(discovery_dates[0]),
        "discovery_oof_end": str(discovery_dates[-1]),
        "locked_holdout_dates": len(holdout_dates),
        "locked_holdout_start": str(holdout_dates[0]),
        "locked_holdout_end": str(holdout_dates[-1]),
        "prediction_dates_contiguous": True,
        "internal_gap_dates": 0,
    }


def validate_existing_artifacts(
    *,
    output_prefix: Path,
    d1_db: Path,
    m15_db: Path,
) -> dict:
    """Reject stale, mixed-generation, or source-divergent SafeUp outputs."""
    expected = _artifact_paths(output_prefix)
    manifest_path = Path(f"{output_prefix}_manifest.json")
    _verify_manifest(manifest_path=manifest_path, expected=expected)
    try:
        coverage = json.loads(
            expected["coverage"].read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("safeup coverage is unreadable") from exc
    if coverage.get("schema") != "safeup_head_challenger_v1":
        raise RuntimeError("safeup coverage schema mismatch")
    locked_holdout = coverage.get("locked_holdout")
    if not isinstance(locked_holdout, dict):
        raise RuntimeError("safeup locked-holdout metadata is missing")
    predictions = pd.read_csv(
        expected["predictions"],
        usecols=["scope", "date", "market"],
    )
    cohort_audit = _prediction_scope_contract(
        predictions,
        locked_holdout,
    )
    if coverage.get("prediction_scope_contract") != cohort_audit:
        raise RuntimeError(
            "safeup prediction scope audit does not match coverage"
        )
    inputs = coverage.get("inputs")
    if not isinstance(inputs, dict):
        raise RuntimeError("safeup input lineage is missing")
    if inputs.get("script_sha256") != _sha256(Path(__file__)):
        raise RuntimeError("safeup artifacts were built by stale code")
    if inputs.get("code_lineage") != _code_lineage():
        raise RuntimeError("safeup local code dependencies have changed")
    recorded_signatures = inputs.get("source_signatures")
    if not isinstance(recorded_signatures, dict):
        raise RuntimeError("safeup source signatures are missing")
    uses_paths = inputs.get("m15_db_sha256") is not None
    current_paths = [d1_db] + ([m15_db] if uses_paths else [])
    current_signatures = {
        signature["path"]: signature
        for signature in (
            _file_signature(path) for path in current_paths
        )
    }
    normalized_recorded = {}
    for signature in recorded_signatures.values():
        if not isinstance(signature, dict) or not signature.get("path"):
            raise RuntimeError("safeup source signature is malformed")
        normalized_recorded[str(signature["path"])] = signature
    if normalized_recorded != current_signatures:
        raise RuntimeError("safeup source inputs have changed")
    schedule_info = coverage.get("fp_schedule_baseline")
    schedule_audit = None
    if (
        isinstance(schedule_info, dict)
        and schedule_info.get("generated") is True
    ):
        schedule_prefix = Path(str(schedule_info.get("prefix", "")))
        if not schedule_prefix.is_absolute():
            schedule_prefix = ROOT / schedule_prefix
        schedule_audit = validate_fp_schedule_artifacts(
            output_prefix=schedule_prefix,
            d1_db=d1_db,
            m15_db=m15_db,
        )
        if (
            schedule_audit["split_schedule_sha256"]
            != schedule_info.get("split_schedule_sha256")
        ):
            raise RuntimeError(
                "safeup/shared baseline schedule linkage mismatch"
            )
    return {
        "status": "valid",
        "schema": coverage["schema"],
        "output_prefix": str(output_prefix),
        "uses_15m_paths": uses_paths,
        **cohort_audit,
        "fp_schedule_baseline": schedule_audit,
        "manifest_sha256": _sha256(manifest_path),
    }


def _publish_staged_files(
    staged_to_target: dict[Path, Path],
) -> None:
    """Publish all result files together, restoring originals on failure."""
    if not staged_to_target:
        raise ValueError("no staged artifacts to publish")
    parents = {target.parent.resolve() for target in staged_to_target.values()}
    if len(parents) != 1:
        raise ValueError("all generation artifacts must share one directory")
    output_dir = next(iter(parents))
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct safe-up10 challenger with a sealed holdout"
    )
    parser.add_argument("--limit-markets", type=int, default=None)
    parser.add_argument("--d1-db", type=Path, default=D1_DB)
    parser.add_argument("--m15-db", type=Path, default=M15_DB)
    parser.add_argument("--output-prefix", type=Path, default=OUT_PREFIX)
    parser.add_argument(
        "--bootstrap-draws",
        type=int,
        default=BOOTSTRAP_DRAWS,
    )
    parser.add_argument(
        "--no-path",
        action="store_true",
        help="skip the exact 15-minute secondary execution lens",
    )
    parser.add_argument(
        "--validate-existing",
        action="store_true",
        help="validate current artifacts and exit without rebuilding",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_existing:
        audit = validate_existing_artifacts(
            output_prefix=args.output_prefix,
            d1_db=args.d1_db,
            m15_db=args.m15_db,
        )
        print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
        return
    if not 100 <= args.bootstrap_draws <= 100_000:
        raise ValueError("--bootstrap-draws must be in [100, 100000]")
    if args.limit_markets is not None and args.limit_markets < UNIVERSE_TOP_N:
        raise ValueError("--limit-markets must be >= 100")
    if not args.d1_db.is_file():
        raise FileNotFoundError(args.d1_db)
    if not args.no_path and not args.m15_db.is_file():
        raise FileNotFoundError(args.m15_db)
    provenance_paths = [args.d1_db]
    if not args.no_path:
        provenance_paths.append(args.m15_db)
    source_signatures = {
        str(path): _file_signature(path) for path in provenance_paths
    }
    code_lineage = _code_lineage()

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    panel, panel_meta = prepare_panel(
        args.limit_markets,
        d1_db=args.d1_db,
    )
    features = _feature_columns(panel)
    all_dates = np.sort(panel["date"].unique())
    if len(all_dates) <= LOCKED_HOLDOUT_DATES + 365:
        raise RuntimeError("not enough dates for discovery and locked holdout")
    holdout_dates = all_dates[-LOCKED_HOLDOUT_DATES:]
    discovery_dates = all_dates[:-LOCKED_HOLDOUT_DATES]
    holdout_hash = hashlib.sha256(
        "\n".join(map(str, holdout_dates)).encode("utf-8")
    ).hexdigest()
    log.info(
        "SEALED HOLDOUT dates=%d %s..%s sha256=%s",
        len(holdout_dates),
        holdout_dates[0],
        holdout_dates[-1],
        holdout_hash[:16],
    )
    discovery_panel = panel[
        panel["date"].isin(set(discovery_dates))
    ].copy()

    discovery, discovery_meta = run_discovery(discovery_panel, features)
    # No outcome-driven selection occurs among the original five policies.
    # safeup_pareto_rank is intentionally carried as a separately flagged
    # post-hoc diagnostic after the original holdout had already been seen.
    holdout, holdout_meta = run_locked_holdout(
        panel,
        holdout_dates,
        features,
    )
    predictions = pd.concat([discovery, holdout], ignore_index=True)
    evaluation_counts = predictions.groupby(["scope", "date"]).size()
    if not (evaluation_counts == UNIVERSE_TOP_N).all():
        bad = evaluation_counts[evaluation_counts != UNIVERSE_TOP_N]
        raise RuntimeError(
            "evaluation universe is not an identical static Top 100: "
            f"{bad.head().to_dict()}"
        )
    fp_schedule_predictions = None
    fp_schedule_contract = None
    fp_schedule_coverage = None
    if not args.no_path:
        (
            benchmark_schedule_dates,
            benchmark_schedule_audit,
        ) = benchmark_complete_schedule_dates(panel, args.m15_db)
        (
            fp_schedule_predictions,
            fp_schedule_contract,
            fp_schedule_metadata,
        ) = run_common_schedule_baseline(
            panel,
            features,
            benchmark_schedule_dates,
        )
        fp_schedule_coverage = {
            "schema": "safeup_head_challenger_v1_fp_schedule",
            "created_at": pd.Timestamp.now(
                tz="Asia/Seoul"
            ).isoformat(),
            "research_only": True,
            "split_schedule_sha256": fp_schedule_contract[
                "split_schedule_sha256"
            ],
            "inputs": {
                "script_sha256": _sha256(Path(__file__)),
                "code_lineage": code_lineage,
                "source_signatures": {
                    "d1_db": source_signatures[str(args.d1_db)],
                    "m15_db": source_signatures[str(args.m15_db)],
                },
            },
            "benchmark_axis": benchmark_schedule_audit,
            "model_contract": {
                "targets": list(TARGETS),
                "features": features,
                "training_rows": (
                    "exact PIT D1 Top100 rows on each common "
                    "benchmark-complete schedule train-date set"
                ),
                "test_rows": (
                    "exact common benchmark-complete schedule dates"
                ),
                "model_metadata": fp_schedule_metadata,
            },
            "historical_holdout_contaminated": True,
            "virgin_or_preregistered": False,
            "maximum_evidence_grade": "historical_comparison_only",
        }
    picks = select_top3(predictions)

    if args.no_path:
        picks["path_complete"] = False
        picks["path_quality"] = "disabled"
        picks["path_common_complete_date"] = False
        path_meta = {"enabled": False}
    else:
        picks, path_meta = attach_paths(picks, args.m15_db)
        path_meta["enabled"] = True

    summary = policy_metrics(predictions, picks, by_fold=False)
    folds = policy_metrics(predictions, picks, by_fold=True)
    heads = head_metrics(predictions)
    paired = paired_bootstrap(
        picks,
        draws=args.bootstrap_draws,
        seed=MODEL_SEED,
    )
    source_signatures_after = {
        str(path): _file_signature(path) for path in provenance_paths
    }
    if source_signatures_after != source_signatures:
        raise RuntimeError("research input changed while challenger was running")
    if _code_lineage() != code_lineage:
        raise RuntimeError("local code changed while challenger was running")

    prefix = args.output_prefix
    fp_schedule_prefix = prefix.with_name(
        f"{prefix.name}_fp_schedule"
    )
    predictions_path = Path(f"{prefix}_predictions.csv.gz")
    picks_path = Path(f"{prefix}_picks.csv.gz")
    summary_path = Path(f"{prefix}_summary.csv")
    folds_path = Path(f"{prefix}_folds.csv")
    heads_path = Path(f"{prefix}_heads.csv")
    paired_path = Path(f"{prefix}_paired.csv")
    coverage = {
        "schema": "safeup_head_challenger_v1",
        "created_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "research_only": True,
        "production_modified": False,
        "auto_order_code": False,
        "inputs": {
            "d1_db": str(args.d1_db),
            "d1_db_sha256": _sha256(args.d1_db),
            "m15_db": str(args.m15_db) if not args.no_path else None,
            "m15_db_sha256": (
                _sha256(args.m15_db) if not args.no_path else None
            ),
            "script_sha256": _sha256(Path(__file__)),
            "code_lineage": code_lineage,
            "sources_before_and_after_identical": True,
            "source_signatures": source_signatures,
        },
        "target_contract": {
            "primary": (
                "safe_up10 = high_D/open_D-1 >= 0.10 AND "
                "low_D/open_D-1 > -0.05"
            ),
            "control": "up10 = high_D/open_D-1 >= 0.10",
            "downside": "dn5 = low_D/open_D-1 <= -0.05",
        },
        "trial_contract": {
            "model_targets": list(TARGETS),
            "model_fit_count_per_split": len(TARGETS),
            "hyperparameter_sweep": False,
            "policies": list(POLICIES),
            "primary_policy_fixed_before_holdout": "safeup_head",
            "ex_ante_policy_count": len(EX_ANTE_POLICIES),
            "posthoc_policy_count": 1,
            "posthoc_policy": POSTHOC_POLICY,
            "posthoc_reason": (
                "requested after inspection of the original locked holdout; "
                "promotion-ineligible regardless of historical result"
            ),
            "posthoc_max_verdict": "forward_only_shadow_candidate",
        },
        "hygiene": {
            "feature_boundary": "<= D-1",
            "minimum_prior_history_bars": MIN_PRIOR_HISTORY,
            "universe": "point-in-time D-1 quote-volume Top 100",
            "outer": (
                f"expanding {OUTER_FOLDS}-fold discovery WF, "
                f"{EMBARGO_DATES}-date embargo"
            ),
            "calibration": (
                f"expanding {INNER_FOLDS}-fold true inner OOF isotonic "
                "using outer-train only"
            ),
            "round_trip_cost_once": ROUND_TRIP_COST,
            "survivorship": (
                "all D1 DB markets, including inactive retained markets"
            ),
        },
        "locked_holdout": {
            "n_dates": int(len(holdout_dates)),
            "start": str(holdout_dates[0]),
            "end": str(holdout_dates[-1]),
            "dates_sha256": holdout_hash,
            "unlocked_after_discovery_complete": True,
            "selection_from_holdout_ex_ante_policies": False,
            "posthoc_design_informed_by_holdout": True,
            "posthoc_exception": (
                "safeup_pareto_rank was added after this holdout was seen "
                "and cannot use it as confirmatory evidence"
            ),
        },
        "panel": panel_meta,
        "evaluation_universe_audit": {
            "all_scope_dates_exactly_100": True,
            "scope_dates": int(len(evaluation_counts)),
            "min_candidates": int(evaluation_counts.min()),
            "max_candidates": int(evaluation_counts.max()),
            "same_candidate_frame_for_every_policy": True,
            "top_k_per_policy_date": TOP_K,
        },
        "folds": {
            "discovery": discovery_meta,
            "locked_holdout": holdout_meta,
        },
        "path": path_meta,
        "fp_schedule_baseline": (
            {
                "generated": True,
                "prefix": _display_path(fp_schedule_prefix),
                "split_schedule_sha256": fp_schedule_contract[
                    "split_schedule_sha256"
                ],
                "historical_holdout_contaminated": True,
            }
            if fp_schedule_contract is not None
            else {
                "generated": False,
                "reason": "--no-path disables benchmark schedule creation",
            }
        ),
        "artifacts": {
            "predictions": _display_path(predictions_path),
            "picks": _display_path(picks_path),
            "summary": _display_path(summary_path),
            "folds": _display_path(folds_path),
            "heads": _display_path(heads_path),
            "paired": _display_path(paired_path),
        },
    }
    coverage["prediction_scope_contract"] = _prediction_scope_contract(
        predictions,
        coverage["locked_holdout"],
    )
    artifact_paths = _artifact_paths(prefix)
    manifest_path = Path(f"{prefix}_manifest.json")
    schedule_artifact_paths = (
        _fp_schedule_artifact_paths(fp_schedule_prefix)
        if fp_schedule_predictions is not None
        else {}
    )
    schedule_manifest_path = Path(
        f"{fp_schedule_prefix}_manifest.json"
    )
    with tempfile.TemporaryDirectory(
        dir=prefix.parent,
        prefix=f".{prefix.name}.generation.",
    ) as stage_directory:
        stage_root = Path(stage_directory)
        staged = {
            name: stage_root / path.name
            for name, path in artifact_paths.items()
        }
        staged_schedule = {
            name: stage_root / path.name
            for name, path in schedule_artifact_paths.items()
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
        folds.to_csv(staged["folds"], index=False)
        heads.to_csv(staged["heads"], index=False)
        paired.to_csv(staged["paired"], index=False)
        _write_json(staged["coverage"], coverage)
        staged_schedule_manifest = None
        if fp_schedule_predictions is not None:
            if (
                fp_schedule_contract is None
                or fp_schedule_coverage is None
            ):
                raise RuntimeError("shared baseline staging is incomplete")
            fp_schedule_predictions.to_csv(
                staged_schedule["predictions"],
                index=False,
                compression=GZIP_COMPRESSION,
                float_format="%.17g",
            )
            _write_json(
                staged_schedule["contract"],
                fp_schedule_contract,
            )
            _write_json(
                staged_schedule["coverage"],
                fp_schedule_coverage,
            )
            staged_schedule_manifest = (
                stage_root / schedule_manifest_path.name
            )
            _write_json(
                staged_schedule_manifest,
                {
                    "schema": (
                        "safeup_head_challenger_v1_fp_schedule_manifest"
                    ),
                    "gzip_mtime": 0,
                    "files": {
                        name: {
                            "path": str(schedule_artifact_paths[name]),
                            "bytes": int(path.stat().st_size),
                            "sha256": _sha256(path),
                        }
                        for name, path in staged_schedule.items()
                    },
                },
            )
        staged_manifest = stage_root / manifest_path.name
        _write_json(
            staged_manifest,
            {
                "schema": "safeup_head_challenger_v1_manifest",
                "gzip_mtime": 0,
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
        publish_map.update(
            {
                staged_schedule[name]: target
                for name, target in schedule_artifact_paths.items()
            }
        )
        if staged_schedule_manifest is not None:
            publish_map[
                staged_schedule_manifest
            ] = schedule_manifest_path
        publish_map[staged_manifest] = manifest_path
        _publish_staged_files(publish_map)

    display = [
        "scope",
        "policy",
        "dates",
        "safe_up10_rate",
        "up10_rate",
        "dn5_rate",
        "safe_up10_lift_within_atr",
        "dn5_delta_within_atr",
        "path_dates",
        "path_safe_up10",
        "path_up10",
        "path_dn5",
        "path_tp_first",
        "path_sl_first",
        "path_net_mean",
        "path_net_ci95_lo",
        "path_net_ci95_hi",
        "path_sharpe_ann",
        "path_max_drawdown",
    ]
    log.info("RESULTS\n%s", summary[display].to_string(index=False))
    log.info("wrote isolated artifacts at %s*", prefix)


if __name__ == "__main__":
    main()

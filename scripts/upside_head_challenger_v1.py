"""Upper-head challenger research without touching production R1.

Research question
-----------------
Can the existing, user-approved target

    y_up10(D) = 1[high_D / open_D - 1 >= 10%]

be ranked better from information available by D-1?

The live R1 code is deliberately not imported for mutation or deployment.  This
script only reuses the established leak-free feature builder and writes isolated
research artifacts named ``upside_head_challenger_v1*``.

Six pre-declared modelling trials are compared:

1. ``cls_core``: current 24-feature XGB classifier, raw score for ranking.
2. ``cls_bvol``: (1) plus Binance D-1 ``b_vol_surge`` and missingness flag.
3. ``cls_bvol_volbalanced``: (2), but class weights are balanced within D-1
   ATR quintiles to force conditional (not merely high-volatility) separation.
4. ``rank_core``: daily cross-sectional XGB pairwise ranker, core features.
5. ``rank_bvol``: (4) plus Binance D-1 volume-surge features.
6. ``rank_novol_bvol``: (5) without price-volatility level features.

There is no hyperparameter sweep and no best-cell deployment.  A seventh
``legacy_core_in_sample_bucket`` reference is derived from the already-fitted
``cls_core`` model at zero additional model fit.  It intentionally reproduces
the known defective pattern (in-sample, possibly non-monotone bucket mapping)
and is never promotion-eligible.

Validation hygiene
------------------
* Feature row D is built by ``build_market_features`` whose inputs are shifted
  one day: features <= D-1, target = day-D open/high.
* Universe is the production-aligned D-1 quote-volume top 100, calculated at
  each historical date with inactive markets retained in the database.
* Outer expanding walk-forward has a five-day purge/embargo.
* Displayed probability is fitted with monotone isotonic calibration using
  *inner expanding OOF predictions from the outer-train period only*.
* Ranking uses raw score.  Raw discrimination and calibrated probability
  quality are reported separately.
* Binance date D-1 is the UTC day ending exactly at Upbit day-D 09:00 KST.
* Net return subtracts the 0.15% round-trip cost exactly once.
* 15-minute paths use the same BTC-benchmark-complete + target-gap-flat rule as
  ``ledger.path_quality``.  Incomplete benchmark days are deferred, not silently
  treated as losses or removed before candidate ranking.

Usage
-----
    venv/bin/python scripts/upside_head_challenger_v1.py
    venv/bin/python scripts/upside_head_challenger_v1.py --variants cls_core,cls_bvol
    venv/bin/python scripts/upside_head_challenger_v1.py --no-path
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
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ledger.path_quality import (  # noqa: E402
    BAR_FREQ,
    BENCHMARK_MARKET,
    EXPECTED_BARS,
    assess_15m_window,
)
from ops.code_lineage import python_code_lineage  # noqa: E402
from scripts.binance_leadlag_v1 import (  # noqa: E402
    build_binance_features,
    krw_to_binance,
    verify_boundary,
)
from scripts.downside_head_riskreward_v1 import (  # noqa: E402
    LEAK_COLS,
    add_cross_sectional,
    attach_btc_regime,
    build_panel,
)
from scripts.recommendation_scorer_v1 import PRECURSOR_FEATURES  # noqa: E402
from scripts.recommender_downside_exit_v1 import simulate_path  # noqa: E402


UPBIT_D1_DB = ROOT / "data" / "upbit_d1.db"
BINANCE_D1_DB = ROOT / "data" / "binance_d1.db"
UPBIT_15M_DB = ROOT / "data" / "upbit_15m.db"
OUT_PREFIX = ROOT / "output" / "upside_head_challenger_v1"

TARGET_COL = "lab_up_10"
RAW_UP_COL = "up_high_ret"
RAW_DOWN_COL = "down_low_ret"
EOD_COL = "eod_ret"

UNIVERSE_TOP_N = 100
TOP_K = 3
ROUND_TRIP_COST = 0.0015
HARD_SL = 0.03
TAKE_PROFIT = 0.05
OUTER_FOLDS = 5
INNER_FOLDS = 3
EMBARGO_DAYS = 5
LOCKED_HOLDOUT_DAYS = 180
EXECUTION_START_HOUR = 9
EXECUTION_START_MINUTE = 15
PAIRED_BOOTSTRAP_DRAWS = 2000
CALIBRATION_MIN_ROWS = 500
CALIBRATION_MIN_POS = 20
MODEL_SEED = 42
GZIP_COMPRESSION = {"method": "gzip", "mtime": 0}
RETURN_TOLERANCE = 1e-8

PRICE_VOL_FEATURES = {
    "f_atr_pct_14",
    "f_rv_7d",
    "f_rv_21d",
    "f_atr_xs_decile",
}
BINANCE_FEATURES = ["b_vol_surge", "b_vol_missing"]

TRIALS = (
    "cls_core",
    "cls_bvol",
    "cls_bvol_volbalanced",
    "rank_core",
    "rank_bvol",
    "rank_novol_bvol",
)
LEGACY_REFERENCE = "legacy_core_in_sample_bucket"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("upside_head_challenger_v1")


@dataclass(frozen=True)
class TrialSpec:
    name: str
    model_kind: str
    use_binance: bool
    drop_price_vol: bool = False
    vol_balanced: bool = False


TRIAL_SPECS = {
    "cls_core": TrialSpec("cls_core", "classifier", False),
    "cls_bvol": TrialSpec("cls_bvol", "classifier", True),
    "cls_bvol_volbalanced": TrialSpec(
        "cls_bvol_volbalanced", "classifier", True, vol_balanced=True
    ),
    "rank_core": TrialSpec("rank_core", "ranker", False),
    "rank_bvol": TrialSpec("rank_bvol", "ranker", True),
    "rank_novol_bvol": TrialSpec(
        "rank_novol_bvol", "ranker", True, drop_price_vol=True
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _code_lineage() -> dict:
    return python_code_lineage(entrypoint=Path(__file__), root=ROOT)


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
    """Latest fully closed Upbit trading day under KST 09:00 boundaries."""
    now_kst = pd.Timestamp.now(tz="Asia/Seoul")
    current_session = (now_kst - pd.Timedelta(hours=9)).date()
    return pd.Timestamp(current_session) - pd.Timedelta(days=1)


def _base_features(panel: pd.DataFrame) -> list[str]:
    features = [
        c
        for c in PRECURSOR_FEATURES
        if c in panel.columns
        and c not in LEAK_COLS
        and not c.startswith("next_")
        and not c.startswith("lab_")
    ]
    if len(features) != len(PRECURSOR_FEATURES):
        missing = sorted(set(PRECURSOR_FEATURES) - set(features))
        raise RuntimeError(f"core feature contract changed; missing/excluded={missing}")
    return features


def _add_binance_d1(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Join Binance bar D-1 to target row D; never back/forward fill."""
    boundary = verify_boundary(str(UPBIT_D1_DB), str(BINANCE_D1_DB))
    alignment = boundary[boundary["check"] == "ALIGNMENT"]
    if alignment.empty or str(alignment.iloc[0]["status"]) != "PASS":
        raise RuntimeError(
            "Binance/Upbit daily boundary alignment did not pass:\n"
            + boundary.to_string(index=False)
        )

    out = panel.copy()
    out["bn_market"] = out["market"].map(krw_to_binance)
    out["feature_date"] = (
        pd.to_datetime(out["date"]) - pd.Timedelta(days=1)
    ).dt.date
    needed = set(out["bn_market"].dropna().unique())
    bn = build_binance_features(str(BINANCE_D1_DB), needed)
    if bn.empty:
        out["b_vol_surge"] = np.nan
    else:
        out = out.merge(
            bn[["bn_market", "feature_date", "b_vol_surge"]],
            on=["bn_market", "feature_date"],
            how="left",
            validate="many_to_one",
        )
    out["b_vol_missing"] = out["b_vol_surge"].isna().astype(float)
    coverage = {
        "boundary": boundary.to_dict(orient="records"),
        "n_needed_binance_markets": len(needed),
        "row_coverage": float(out["b_vol_surge"].notna().mean()),
        "market_coverage": int(
            out.loc[out["b_vol_surge"].notna(), "market"].nunique()
        ),
        "max_binance_feature_date": (
            str(bn["feature_date"].max()) if not bn.empty else None
        ),
    }
    return out, coverage


def prepare_panel(limit_markets: int | None = None) -> tuple[pd.DataFrame, dict]:
    panel = build_panel(limit_markets)
    panel = panel.sort_values(["market", "timestamp"]).copy()
    panel["history_n"] = panel.groupby("market").cumcount()
    pre_history_rows = len(panel)
    # Production _build_panel requires >=70 bars as of each inference call.
    # Enforce that point-in-time, not using the market's eventual full history.
    panel = panel[panel["history_n"] >= 70].copy()
    post_history_rows = len(panel)
    panel = add_cross_sectional(panel)
    panel = attach_btc_regime(panel)
    cutoff = _completed_label_cutoff()
    panel = panel[pd.to_datetime(panel["date"]) <= cutoff].copy()
    panel = panel[panel["f_qv_rank"].notna()].copy()
    panel = panel.dropna(
        subset=[TARGET_COL, RAW_UP_COL, RAW_DOWN_COL, EOD_COL, "date", "market"]
    )
    panel["y_up10"] = panel[TARGET_COL].astype(int)
    panel["date"] = pd.to_datetime(panel["date"]).dt.date
    required_numeric = panel[
        [
            TARGET_COL,
            RAW_UP_COL,
            RAW_DOWN_COL,
            EOD_COL,
            "f_qv_rank",
            "history_n",
        ]
    ].apply(pd.to_numeric, errors="coerce")
    if (
        required_numeric.isna().any().any()
        or not np.isfinite(required_numeric.to_numpy()).all()
    ):
        raise RuntimeError("panel contains nonfinite required values")
    if not required_numeric[TARGET_COL].isin([0, 1]).all():
        raise RuntimeError("up10 target is not binary")
    invalid_ohlc = (
        (panel[RAW_UP_COL] < -RETURN_TOLERANCE)
        | (panel[RAW_DOWN_COL] > RETURN_TOLERANCE)
        | (panel[RAW_DOWN_COL] <= -1)
        | (panel[EOD_COL] <= -1)
        | (panel[RAW_UP_COL] + RETURN_TOLERANCE < panel[EOD_COL])
        | (panel[RAW_DOWN_COL] - RETURN_TOLERANCE > panel[EOD_COL])
    )
    if invalid_ohlc.any():
        raise RuntimeError("panel daily return columns violate OHLC ordering")
    if not panel["market"].astype("string").str.fullmatch(
        r"KRW-[A-Z0-9]+", na=False
    ).all():
        raise RuntimeError("panel contains invalid/non-canonical markets")
    if panel.duplicated(["date", "market"]).any():
        raise RuntimeError("panel contains duplicate date/market rows")
    panel = panel.sort_values(
        ["date", "f_qv_rank", "market"],
        ascending=[True, True, True],
    )
    panel["f_qv_rank"] = (
        panel.groupby("date", sort=False).cumcount() + 1
    )
    panel = panel[panel["f_qv_rank"] <= UNIVERSE_TOP_N].copy()
    counts = panel.groupby("date").size()
    exact_dates = set(counts[counts == UNIVERSE_TOP_N].index)
    panel = panel[panel["date"].isin(exact_dates)].copy()
    if panel.empty or not (
        panel.groupby("date").size() == UNIVERSE_TOP_N
    ).all():
        raise RuntimeError("panel is not exact point-in-time Top100")
    panel = panel.sort_values(["date", "market"]).reset_index(drop=True)
    panel, binance_meta = _add_binance_d1(panel)

    # Five fixed D-1 volatility strata.  This is diagnostic/sample weighting,
    # never a target redefinition.
    atr_rank = panel["f_atr_xs_decile"].clip(0.0, 1.0)
    panel["vol_band"] = np.minimum(
        np.floor(atr_rank.fillna(0.5).to_numpy() * 5), 4
    ).astype(int)
    panel["mfe_atr_multiple"] = panel[RAW_UP_COL] / np.maximum(
        panel["f_atr_pct_14"].abs(), 1e-6
    )

    leak_features = [
        c
        for c in _base_features(panel) + BINANCE_FEATURES
        if c in LEAK_COLS or c.startswith("next_") or c.startswith("lab_")
    ]
    if leak_features:
        raise RuntimeError(f"leak feature detected: {leak_features}")

    meta = {
        "rows": len(panel),
        "rows_before_point_in_time_history_gate": pre_history_rows,
        "rows_removed_history_lt_70": pre_history_rows - post_history_rows,
        "point_in_time_min_history": int(panel["history_n"].min()),
        "point_in_time_history_gate": "history_n >= 70 before cross-sectional universe ranks",
        "markets": int(panel["market"].nunique()),
        "dates": int(panel["date"].nunique()),
        "date_min": str(panel["date"].min()),
        "date_max": str(panel["date"].max()),
        "label_cutoff": str(cutoff.date()),
        "up10_base_rate": float(panel["y_up10"].mean()),
        "binance": binance_meta,
    }
    return panel, meta


def operational_binance_availability() -> dict:
    """Static audit of whether freshly closed Binance D-1 exists at R1 send."""
    runner = ROOT / "scripts" / "daily_run_distribution.sh"
    lines = runner.read_text(encoding="utf-8").splitlines()
    recommend_line = next(
        (
            i
            for i, line in enumerate(lines, 1)
            if "recommend_send + recommend_today" in line
        ),
        None,
    )
    collector_line = next(
        (
            i
            for i, line in enumerate(lines, 1)
            if "collector_binance_d1 --days 3" in line
        ),
        None,
    )
    available = bool(
        recommend_line is not None
        and collector_line is not None
        and collector_line < recommend_line
    )
    return {
        "runner": str(runner.relative_to(ROOT)),
        "r1_send_step_line": recommend_line,
        "binance_refresh_step_line": collector_line,
        "fresh_d1_available_at_r1_send": available,
        "bar_boundary": (
            "Binance D-1 closes 00:00 UTC = 09:00 KST immediately before R1"
        ),
        "reason": (
            "collector runs before R1 send"
            if available
            else "current runner refreshes Binance only after R1 send; "
            "cls_bvol is not point-in-time deployable at 09:10 without an ops reorder"
        ),
    }


def _outer_splits(
    dates: Iterable,
    n_folds: int = OUTER_FOLDS,
    embargo: int = EMBARGO_DAYS,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_folds <= 0 or embargo < 0:
        raise ValueError("invalid outer-split parameters")
    unique = np.sort(np.asarray(list(set(dates)), dtype=object))
    if len(unique) < 2:
        raise ValueError("at least two unique dates are required")
    warmup = int(len(unique) * 0.35)
    first_test_start = warmup + embargo
    edges = np.linspace(
        first_test_start,
        len(unique),
        n_folds + 1,
    ).astype(int)
    splits = []
    for fold in range(n_folds):
        test_start = edges[fold]
        train_end = test_start - embargo
        test_end = edges[fold + 1]
        if test_start >= test_end:
            continue
        train_dates = unique[:train_end]
        test_dates = unique[test_start:test_end]
        if len(train_dates) and len(test_dates):
            splits.append((train_dates, test_dates))
    _validate_splits(unique, splits, embargo)
    return splits


def _inner_splits(
    dates: Iterable,
    n_folds: int = INNER_FOLDS,
    embargo: int = EMBARGO_DAYS,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_folds <= 0 or embargo < 0:
        raise ValueError("invalid inner-split parameters")
    unique = np.sort(np.asarray(list(set(dates)), dtype=object))
    if len(unique) < 2:
        raise ValueError("at least two unique dates are required")
    warmup = max(120, int(len(unique) * 0.45))
    latest_warmup = len(unique) - embargo - n_folds
    if warmup > latest_warmup:
        warmup = min(
            max(30, int(len(unique) * 0.60)),
            latest_warmup,
        )
    if warmup <= 0:
        raise ValueError("not enough dates for inner splits")
    first_validation_start = warmup + embargo
    edges = np.linspace(
        first_validation_start,
        len(unique),
        n_folds + 1,
    ).astype(int)
    splits = []
    for fold in range(n_folds):
        val_start = edges[fold]
        train_end = val_start - embargo
        val_end = edges[fold + 1]
        if val_start >= val_end:
            continue
        tr = unique[:train_end]
        va = unique[val_start:val_end]
        if len(tr) >= 30 and len(va):
            splits.append((tr, va))
    _validate_splits(unique, splits, embargo)
    return splits


def _validate_splits(
    unique: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    embargo: int,
) -> None:
    positions = {value: index for index, value in enumerate(unique)}
    seen_test: set = set()
    for train_dates, test_dates in splits:
        train_set = set(train_dates)
        test_set = set(test_dates)
        if train_set & test_set or seen_test & test_set:
            raise RuntimeError("walk-forward folds overlap")
        if max(train_dates) >= min(test_dates):
            raise RuntimeError("walk-forward chronology is invalid")
        excluded = (
            positions[min(test_dates)] - positions[max(train_dates)] - 1
        )
        if excluded != embargo:
            raise RuntimeError(
                f"walk-forward embargo mismatch: {excluded} != {embargo}"
            )
        seen_test.update(test_set)
    test_positions = sorted(positions[value] for value in seen_test)
    if test_positions and test_positions != list(
        range(test_positions[0], test_positions[-1] + 1)
    ):
        raise RuntimeError("walk-forward test coverage has gaps")


def _features_for(spec: TrialSpec, panel: pd.DataFrame) -> list[str]:
    features = _base_features(panel)
    if spec.drop_price_vol:
        features = [f for f in features if f not in PRICE_VOL_FEATURES]
    if spec.use_binance:
        features += BINANCE_FEATURES
    return features


def _vol_balanced_weights(frame: pd.DataFrame) -> np.ndarray:
    """Equal positive/negative mass inside each train-only ATR quintile."""
    y = frame["y_up10"].to_numpy(dtype=int)
    bands = frame["vol_band"].to_numpy(dtype=int)
    weights = np.ones(len(frame), dtype=float)
    for band in np.unique(bands):
        in_band = bands == band
        n_band = int(in_band.sum())
        for cls in (0, 1):
            mask = in_band & (y == cls)
            n_cls = int(mask.sum())
            if n_cls:
                weights[mask] = n_band / (2.0 * n_cls)
    weights = np.clip(weights, 0.2, 10.0)
    return weights / weights.mean()


def _matrix(
    train: pd.DataFrame, test: pd.DataFrame, features: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    x_train = train[features].replace([np.inf, -np.inf], np.nan)
    medians = x_train.median()
    x_train = x_train.fillna(medians).fillna(0.0)
    x_test = (
        test[features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(medians)
        .fillna(0.0)
    )
    return x_train.to_numpy(dtype=float), x_test.to_numpy(dtype=float)


def _fit_predict(
    spec: TrialSpec,
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    *,
    return_train_scores: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Fit on train only and return scores in the original test row order."""
    import xgboost as xgb

    train_sorted = train.sort_values(["date", "market"]).copy()
    test_ordered = test.copy()
    x_train, x_test = _matrix(train_sorted, test_ordered, features)
    y_train = train_sorted["y_up10"].to_numpy(dtype=int)

    common = dict(
        n_estimators=180,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.5,
        n_jobs=1,
        tree_method="hist",
        random_state=MODEL_SEED,
    )
    if spec.model_kind == "classifier":
        pos = int(y_train.sum())
        if pos < CALIBRATION_MIN_POS or len(np.unique(y_train)) < 2:
            raise RuntimeError(
                f"insufficient classifier support: positives={pos}"
            )
        scale_pos_weight = (
            1.0
            if spec.vol_balanced
            else float((len(y_train) - pos) / max(pos, 1))
        )
        model = xgb.XGBClassifier(
            **common,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
        )
        sample_weight = (
            _vol_balanced_weights(train_sorted) if spec.vol_balanced else None
        )
        model.fit(x_train, y_train, sample_weight=sample_weight, verbose=False)
        test_score = model.predict_proba(x_test)[:, 1]
        train_score = (
            model.predict_proba(x_train)[:, 1] if return_train_scores else None
        )
    elif spec.model_kind == "ranker":
        if int(y_train.sum()) < CALIBRATION_MIN_POS:
            raise RuntimeError("insufficient ranker positive support")
        qid = pd.factorize(train_sorted["date"], sort=True)[0]
        model = xgb.XGBRanker(
            **common,
            objective="rank:pairwise",
            eval_metric="ndcg@3",
        )
        model.fit(x_train, y_train, qid=qid, verbose=False)
        test_score = model.predict(x_test)
        train_score = model.predict(x_train) if return_train_scores else None
    else:
        raise ValueError(f"unknown model kind: {spec.model_kind}")
    if not np.isfinite(test_score).all():
        raise RuntimeError(f"{spec.name} produced nonfinite test scores")
    if train_score is not None and not np.isfinite(train_score).all():
        raise RuntimeError(f"{spec.name} produced nonfinite train scores")
    return np.asarray(test_score, dtype=float), (
        np.asarray(train_score, dtype=float) if train_score is not None else None
    )


def _inner_oof_scores(
    spec: TrialSpec, train: pd.DataFrame, features: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    splits = _inner_splits(train["date"])
    if len(splits) != INNER_FOLDS:
        raise RuntimeError(
            f"expected {INNER_FOLDS} inner folds, got {len(splits)}"
        )
    for inner_train_dates, inner_val_dates in splits:
        inner_train = train[train["date"].isin(set(inner_train_dates))]
        inner_val = train[train["date"].isin(set(inner_val_dates))]
        if (
            len(inner_train) < 1000
            or len(inner_val) < 100
            or inner_train["y_up10"].sum() < CALIBRATION_MIN_POS
        ):
            return np.array([], dtype=float), np.array([], dtype=int)
        raw, _ = _fit_predict(spec, inner_train, inner_val, features)
        scores.append(raw)
        labels.append(inner_val["y_up10"].to_numpy(dtype=int))
    if not scores:
        return np.array([], dtype=float), np.array([], dtype=int)
    return np.concatenate(scores), np.concatenate(labels)


def _fit_monotone_calibrator(
    raw: np.ndarray, y: np.ndarray, base_rate: float
):
    if (
        len(raw) < CALIBRATION_MIN_ROWS
        or int(y.sum()) < CALIBRATION_MIN_POS
        or len(np.unique(raw)) < 2
        or len(np.unique(y)) < 2
    ):
        return None
    calibrator = IsotonicRegression(
        y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip"
    )
    calibrator.fit(raw, y)
    return calibrator


def _apply_monotone_calibrator(
    calibrator, raw: np.ndarray, base_rate: float
) -> np.ndarray:
    if calibrator is None:
        return np.full(len(raw), base_rate, dtype=float)
    return np.asarray(calibrator.predict(raw), dtype=float)


def _legacy_bucket_map(
    raw_train: np.ndarray, y_train: np.ndarray, raw_test: np.ndarray
) -> np.ndarray:
    """Known-bad in-sample/non-monotone reference, matching current pattern."""
    frame = pd.DataFrame({"score": raw_train, "y": y_train}).dropna()
    if len(frame) < 150 or frame["y"].sum() < 5:
        return np.full(len(raw_test), float(frame["y"].mean()), dtype=float)
    try:
        frame["bucket"] = pd.qcut(
            frame["score"].rank(method="first"),
            10,
            labels=False,
            duplicates="drop",
        )
    except ValueError:
        return np.full(len(raw_test), float(frame["y"].mean()), dtype=float)
    grouped = frame.groupby("bucket").agg(
        upper=("score", "max"), hit=("y", "mean")
    )
    edges = grouped["upper"].to_numpy()
    hit_map = grouped["hit"].to_dict()
    idx = np.clip(np.searchsorted(edges, raw_test, side="left"), 0, len(edges) - 1)
    base = float(frame["y"].mean())
    return np.array([hit_map.get(int(i), base) for i in idx], dtype=float)


def run_walk_forward(
    panel: pd.DataFrame,
    variants: list[str],
    *,
    evaluation_scope: str = "discovery_oof",
) -> tuple[pd.DataFrame, list[dict]]:
    splits = _outer_splits(panel["date"])
    if len(splits) != OUTER_FOLDS:
        raise RuntimeError(
            f"expected {OUTER_FOLDS} outer folds, got {len(splits)}"
        )
    predictions: list[pd.DataFrame] = []
    fold_meta: list[dict] = []
    for variant in variants:
        spec = TRIAL_SPECS[variant]
        features = _features_for(spec, panel)
        log.info(
            "trial=%s model=%s features=%d bvol=%s volbalanced=%s",
            variant,
            spec.model_kind,
            len(features),
            spec.use_binance,
            spec.vol_balanced,
        )
        for fold, (train_dates, test_dates) in enumerate(splits):
            train = panel[panel["date"].isin(set(train_dates))].copy()
            test = panel[panel["date"].isin(set(test_dates))].copy()
            if train["date"].max() >= test["date"].min():
                raise RuntimeError("outer train/test chronology violated")
            if not (
                test.groupby("date").size() == UNIVERSE_TOP_N
            ).all():
                raise RuntimeError("outer test frame is not exact Top100")
            inner_raw, inner_y = _inner_oof_scores(spec, train, features)
            base_rate = float(train["y_up10"].mean())
            calibrator = _fit_monotone_calibrator(inner_raw, inner_y, base_rate)
            raw_test, raw_train = _fit_predict(
                spec,
                train,
                test,
                features,
                return_train_scores=(variant == "cls_core"),
            )
            p_cal = _apply_monotone_calibrator(calibrator, raw_test, base_rate)
            if (
                not np.isfinite(p_cal).all()
                or (p_cal < 0).any()
                or (p_cal > 1).any()
            ):
                raise RuntimeError("calibrator produced invalid probabilities")
            keep = [
                "market",
                "date",
                "regime",
                "y_up10",
                RAW_UP_COL,
                RAW_DOWN_COL,
                EOD_COL,
                "f_atr_pct_14",
                "f_atr_xs_decile",
                "f_qv_rank",
                "vol_band",
                "mfe_atr_multiple",
                "b_vol_surge",
                "b_vol_missing",
            ]
            result = test[keep].copy()
            result["fold"] = fold
            result["variant"] = variant
            result["evaluation_scope"] = evaluation_scope
            result["score_raw"] = raw_test
            result["p_cal"] = p_cal
            result["calibration_source"] = (
                "inner_expanding_oof_isotonic"
                if calibrator is not None
                else "outer_train_base_rate_fallback"
            )
            predictions.append(result)

            if variant == "cls_core":
                if raw_train is None:
                    raise RuntimeError(
                        "cls_core requires in-sample scores for the legacy reference"
                    )
                legacy = result.copy()
                legacy["variant"] = LEGACY_REFERENCE
                legacy["p_cal"] = _legacy_bucket_map(
                    raw_train,
                    train.sort_values(["date", "market"])["y_up10"].to_numpy(),
                    raw_test,
                )
                # The live pattern sorts the mapped bucket probability, not raw.
                legacy["score_raw"] = legacy["p_cal"]
                legacy["calibration_source"] = (
                    "outer_train_in_sample_nonmonotone_reference"
                )
                predictions.append(legacy)

            fold_meta.append(
                {
                    "variant": variant,
                    "fold": fold,
                    "train_start": str(train["date"].min()),
                    "train_end": str(train["date"].max()),
                    "test_start": str(test["date"].min()),
                    "test_end": str(test["date"].max()),
                    "train_n": len(train),
                    "test_n": len(test),
                    "train_dates": int(train["date"].nunique()),
                    "test_dates": int(test["date"].nunique()),
                    "embargo_days": EMBARGO_DAYS,
                    "inner_oof_n": len(inner_raw),
                    "inner_oof_pos": int(inner_y.sum()) if len(inner_y) else 0,
                    "calibrator_fitted": calibrator is not None,
                    "features": features,
                }
            )
            log.info(
                "  fold=%d train=%s..%s test=%s..%s inner_oof=%d cal=%s",
                fold,
                train["date"].min(),
                train["date"].max(),
                test["date"].min(),
                test["date"].max(),
                len(inner_raw),
                calibrator is not None,
            )
    if not predictions:
        raise RuntimeError("no OOF predictions produced")
    return pd.concat(predictions, ignore_index=True), fold_meta


def run_locked_holdout(
    panel: pd.DataFrame,
    selected_variant: str,
    holdout_dates: np.ndarray,
) -> tuple[pd.DataFrame, list[dict]]:
    """Fit only the selected challenger plus cls_core reference, then unlock once."""
    ordered_dates = sorted(panel["date"].unique())
    holdout_start = min(holdout_dates)
    start_index = ordered_dates.index(holdout_start)
    train_end = start_index - EMBARGO_DAYS
    if train_end <= 0:
        raise RuntimeError("not enough dates before holdout embargo")
    train_dates = set(ordered_dates[:train_end])
    embargoed_dates = ordered_dates[train_end:start_index]
    train = panel[panel["date"].isin(train_dates)].copy()
    test = panel[panel["date"].isin(set(holdout_dates))].copy()
    if train.empty or test.empty:
        raise RuntimeError("locked holdout train/test split is empty")
    if train["date"].max() >= test["date"].min():
        raise RuntimeError("locked holdout chronology violated")
    if not (test.groupby("date").size() == UNIVERSE_TOP_N).all():
        raise RuntimeError("locked holdout is not exact Top100")

    variants = [selected_variant]
    if selected_variant != "cls_core":
        variants.append("cls_core")
    predictions: list[pd.DataFrame] = []
    meta: list[dict] = []
    for variant in variants:
        spec = TRIAL_SPECS[variant]
        features = _features_for(spec, panel)
        inner_raw, inner_y = _inner_oof_scores(spec, train, features)
        base_rate = float(train["y_up10"].mean())
        calibrator = _fit_monotone_calibrator(inner_raw, inner_y, base_rate)
        raw_test, raw_train = _fit_predict(
            spec,
            train,
            test,
            features,
            return_train_scores=(variant == "cls_core"),
        )
        keep = [
            "market",
            "date",
            "regime",
            "y_up10",
            RAW_UP_COL,
            RAW_DOWN_COL,
            EOD_COL,
            "f_atr_pct_14",
            "f_atr_xs_decile",
            "f_qv_rank",
            "vol_band",
            "mfe_atr_multiple",
            "b_vol_surge",
            "b_vol_missing",
        ]
        result = test[keep].copy()
        result["fold"] = -1
        result["variant"] = variant
        result["evaluation_scope"] = "locked_holdout"
        result["score_raw"] = raw_test
        result["p_cal"] = _apply_monotone_calibrator(
            calibrator, raw_test, base_rate
        )
        if (
            not np.isfinite(result["p_cal"]).all()
            or not result["p_cal"].between(
                0.0, 1.0, inclusive="both"
            ).all()
        ):
            raise RuntimeError("holdout calibrator output is invalid")
        result["calibration_source"] = (
            "inner_expanding_oof_isotonic"
            if calibrator is not None
            else "outer_train_base_rate_fallback"
        )
        predictions.append(result)

        if variant == "cls_core":
            if raw_train is None:
                raise RuntimeError(
                    "cls_core requires in-sample scores for the legacy reference"
                )
            legacy = result.copy()
            legacy["variant"] = LEGACY_REFERENCE
            legacy["p_cal"] = _legacy_bucket_map(
                raw_train,
                train.sort_values(["date", "market"])["y_up10"].to_numpy(),
                raw_test,
            )
            legacy["score_raw"] = legacy["p_cal"]
            legacy["calibration_source"] = (
                "outer_train_in_sample_nonmonotone_reference"
            )
            predictions.append(legacy)

        meta.append(
            {
                "variant": variant,
                "scope": "locked_holdout",
                "train_start": str(train["date"].min()),
                "train_end": str(train["date"].max()),
                "test_start": str(test["date"].min()),
                "test_end": str(test["date"].max()),
                "train_n": len(train),
                "test_n": len(test),
                "train_dates": int(train["date"].nunique()),
                "test_dates": int(test["date"].nunique()),
                "embargo_days": EMBARGO_DAYS,
                "inner_oof_n": len(inner_raw),
                "inner_oof_pos": int(inner_y.sum()) if len(inner_y) else 0,
                "calibrator_fitted": calibrator is not None,
                "features": features,
                "embargoed_dates": [
                    str(value) for value in embargoed_dates
                ],
            }
        )
        log.info(
            "LOCKED HOLDOUT variant=%s train=%s..%s test=%s..%s n=%d",
            variant,
            train["date"].min(),
            train["date"].max(),
            test["date"].min(),
            test["date"].max(),
            len(test),
        )
    return pd.concat(predictions, ignore_index=True), meta


def _safe_auc(y: pd.Series, score: pd.Series) -> float:
    valid = y.notna() & score.notna()
    if valid.sum() < 2 or y[valid].nunique() < 2:
        return np.nan
    return float(roc_auc_score(y[valid], score[valid]))


def _macro_date_auc(frame: pd.DataFrame, score_col: str) -> tuple[float, int]:
    aucs = []
    for _, group in frame.groupby("date"):
        if group["y_up10"].nunique() < 2:
            continue
        aucs.append(_safe_auc(group["y_up10"], group[score_col]))
    return (float(np.mean(aucs)), len(aucs)) if aucs else (np.nan, 0)


def _within_vol_auc(frame: pd.DataFrame, score_col: str) -> tuple[float, dict]:
    by_band: dict[str, float] = {}
    for band, group in frame.groupby("vol_band"):
        by_band[str(int(band))] = _safe_auc(group["y_up10"], group[score_col])
    valid = [v for v in by_band.values() if np.isfinite(v)]
    return (float(np.mean(valid)) if valid else np.nan), by_band


def _ndcg_at_3(frame: pd.DataFrame, score_col: str) -> float:
    values = []
    discounts = 1.0 / np.log2(np.arange(2, 5))
    for _, group in frame.groupby("date"):
        ranked = group.sort_values(
            [score_col, "market"], ascending=[False, True]
        ).head(3)
        gains = ranked["y_up10"].to_numpy(dtype=float)
        dcg = float((gains * discounts[: len(gains)]).sum())
        ideal_n = min(3, int(group["y_up10"].sum()))
        if ideal_n == 0:
            continue
        idcg = float(discounts[:ideal_n].sum())
        values.append(dcg / idcg)
    return float(np.mean(values)) if values else np.nan


def _top3(frame: pd.DataFrame, score_col: str) -> pd.DataFrame:
    return (
        frame.sort_values(
            ["date", score_col, "market"], ascending=[True, False, True]
        )
        .groupby("date", sort=False)
        .head(3)
        .copy()
    )


def _random_score(date, market: str, seed: int = MODEL_SEED) -> int:
    payload = f"{seed}|{date}|{market}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _baseline_predictions(reference: pd.DataFrame) -> dict[str, pd.DataFrame]:
    base = reference.copy()
    base["random_score"] = [
        _random_score(d, m) for d, m in zip(base["date"], base["market"])
    ]
    return {
        "random_seed42": _top3(base, "random_score"),
        "atr_top3": _top3(base, "f_atr_pct_14"),
        "liquidity_top3": (
            base.sort_values(
                ["date", "f_qv_rank", "market"], ascending=[True, True, True]
            )
            .groupby("date", sort=False)
            .head(3)
            .copy()
        ),
    }


def _matched_vol_base(reference: pd.DataFrame, picks: pd.DataFrame) -> float:
    strata = (
        reference.groupby(["date", "vol_band"])["y_up10"]
        .mean()
        .rename("matched_base")
        .reset_index()
    )
    matched = picks.merge(strata, on=["date", "vol_band"], how="left")
    return float(matched["matched_base"].mean())


def _spearman(a: pd.Series, b: pd.Series) -> float:
    valid = a.notna() & b.notna()
    if valid.sum() < 3:
        return np.nan
    return float(a[valid].rank().corr(b[valid].rank()))


def _prediction_metrics(frame: pd.DataFrame) -> dict:
    raw_auc = _safe_auc(frame["y_up10"], frame["score_raw"])
    cal_auc = _safe_auc(frame["y_up10"], frame["p_cal"])
    day_auc, day_auc_n = _macro_date_auc(frame, "score_raw")
    within_auc, by_band = _within_vol_auc(frame, "score_raw")
    valid_cal = frame.dropna(subset=["p_cal", "y_up10"])
    brier = (
        float(brier_score_loss(valid_cal["y_up10"], valid_cal["p_cal"]))
        if len(valid_cal)
        else np.nan
    )
    top_decile = frame[
        frame["score_raw"]
        >= frame.groupby("date")["score_raw"].transform(
            lambda x: x.quantile(0.9)
        )
    ]
    return {
        "n_rows": len(frame),
        "n_dates": int(frame["date"].nunique()),
        "base_up10": float(frame["y_up10"].mean()),
        "raw_auc": raw_auc,
        "calibrated_auc": cal_auc,
        "macro_within_date_auc": day_auc,
        "macro_within_date_auc_n": day_auc_n,
        "macro_within_vol_auc": within_auc,
        "within_vol_auc_json": json.dumps(by_band, sort_keys=True),
        "calibrated_brier": brier,
        "mean_predicted_p": float(frame["p_cal"].mean()),
        "top_decile_up10": float(top_decile["y_up10"].mean()),
        "top_decile_n": len(top_decile),
        "score_atr_spearman": _spearman(
            frame["score_raw"], frame["f_atr_xs_decile"]
        ),
        "ndcg_at_3": _ndcg_at_3(frame, "score_raw"),
    }


def _pick_metrics(
    reference: pd.DataFrame,
    picks: pd.DataFrame,
    path_cache: dict[tuple[str, object], dict] | None,
    common_path_dates: set | None = None,
) -> dict:
    picks = picks.reset_index(drop=True)
    same_dates = reference[reference["date"].isin(set(picks["date"]))]
    date_base = (
        same_dates.groupby("date")["y_up10"].mean().reindex(picks["date"].unique())
    )
    matched_base = _matched_vol_base(same_dates, picks)
    up10 = float(picks["y_up10"].mean())
    safe_up10 = float(
        (
            (picks["y_up10"] == 1)
            & (picks[RAW_DOWN_COL] > -0.05)
        ).mean()
    )
    out = {
        "top3_n": len(picks),
        "top3_dates": int(picks["date"].nunique()),
        "label_top3_up10_day_open": up10,
        "label_top3_safe_up10_day_open": safe_up10,
        "label_top3_lift_vs_universe": (
            up10 / float(date_base.mean()) if float(date_base.mean()) > 0 else np.nan
        ),
        "label_top3_matched_vol_base_up10": matched_base,
        "top3_within_vol_lift": (
            up10 / matched_base if matched_base > 0 else np.nan
        ),
        "label_top3_mfe_mean_day_open": float(picks[RAW_UP_COL].mean()),
        "label_top3_mfe_atr_multiple": float(
            picks["mfe_atr_multiple"].mean()
        ),
        "label_top3_dn5_day_open": float(
            (picks[RAW_DOWN_COL] <= -0.05).mean()
        ),
        "label_top3_eod_net_mean_day_open": float(
            (picks[EOD_COL] - ROUND_TRIP_COST).mean()
        ),
        "top3_up10": np.nan,
        "top3_safe_up10": np.nan,
        "top3_dn5": np.nan,
    }
    if path_cache is None:
        return out
    path_rows = []
    for _, row in picks.iterrows():
        key = (str(row["market"]), row["date"])
        path_rows.append(path_cache.get(key, {"path_complete": False}))
    path = pd.DataFrame(path_rows)
    path["date"] = picks["date"].to_numpy()
    out["path_complete_n"] = int(path.get("path_complete", False).sum())
    out["path_complete_rate"] = float(path.get("path_complete", False).mean())
    full_dates = (
        path.groupby("date")["path_complete"].agg(["sum", "count"])
    )
    full_dates = set(
        full_dates[(full_dates["sum"] == 3) & (full_dates["count"] == 3)].index
    )
    if common_path_dates is not None:
        full_dates &= common_path_dates
    complete = path[
        path["date"].isin(full_dates) & path.get("path_complete", False)
    ].copy()
    out["path_full_days"] = len(full_dates)
    out["path_full_day_picks"] = len(complete)
    if complete.empty:
        return out
    daily = pd.DataFrame(
        {
            "date": picks.loc[complete.index, "date"].to_numpy(),
            "net": complete["path_net"].to_numpy(),
        }
    ).groupby("date")["net"].mean().sort_index()
    equity = (1.0 + daily).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    out.update(
        top3_up10=float(complete["execution_up10"].mean()),
        top3_safe_up10=float(complete["execution_safe_up10"].mean()),
        top3_dn5=float(complete["execution_dn5"].mean()),
        top3_mfe_mean_execution=float(complete["execution_mfe"].mean()),
        top3_mae_mean_execution=float(complete["execution_mae"].mean()),
        path_net_mean=float(complete["path_net"].mean()),
        path_hit=float((complete["path_net"] > 0).mean()),
        path_pct_sl=float((complete["outcome"] == "sl").mean()),
        path_pct_tp=float((complete["outcome"] == "tp").mean()),
        path_tp5_before_sl3=float((complete["outcome"] == "tp").mean()),
        path_cum=float(equity.iloc[-1] - 1.0),
        path_mdd=float(drawdown.min()),
        path_flat_filled_mean=float(complete["flat_filled_bars"].mean()),
    )
    return out


def _common_complete_path_dates(
    all_picks: dict[str, pd.DataFrame],
    path_cache: dict[tuple[str, object], dict],
) -> set:
    common: set | None = None
    for name, picks in all_picks.items():
        if picks.duplicated(["date", "market"]).any():
            raise RuntimeError(f"{name} contains duplicate picks")
        rows = []
        for pick in picks.itertuples(index=False):
            path = path_cache.get((str(pick.market), pick.date), {})
            rows.append(
                {
                    "date": pick.date,
                    "complete": bool(path.get("path_complete", False)),
                }
            )
        quality = pd.DataFrame(rows).groupby("date")["complete"].agg(
            ["size", "sum"]
        )
        dates = set(
            quality[
                (quality["size"] == TOP_K)
                & (quality["sum"] == TOP_K)
            ].index
        )
        common = dates if common is None else common & dates
    return common or set()


def _expected_grid(date) -> pd.DatetimeIndex:
    start = pd.Timestamp(date).normalize() + pd.Timedelta(
        hours=EXECUTION_START_HOUR,
        minutes=EXECUTION_START_MINUTE,
    )
    return pd.date_range(start=start, periods=EXPECTED_BARS, freq=BAR_FREQ)


def _load_market_rows(
    conn: sqlite3.Connection, market: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
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
    frame = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close"])
    if not frame.empty:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


def _benchmark_complete_dates(
    conn: sqlite3.Connection, dates: list
) -> dict[object, bool]:
    first_grid = _expected_grid(min(dates))
    last_grid = _expected_grid(max(dates))
    start = first_grid[0]
    end = last_grid[0] + BAR_FREQ * EXPECTED_BARS
    btc = _load_market_rows(conn, BENCHMARK_MARKET, start, end)
    horizon = conn.execute(
        "SELECT MIN(timestamp), MAX(timestamp) FROM candles WHERE market=?",
        (BENCHMARK_MARKET,),
    ).fetchone()
    h_start = pd.Timestamp(horizon[0]) if horizon and horizon[0] else None
    h_end = pd.Timestamp(horizon[1]) if horizon and horizon[1] else None
    result = {}
    for date in dates:
        grid = _expected_grid(date)
        end_boundary = grid[0] + BAR_FREQ * EXPECTED_BARS
        window_timestamps = [
            timestamp
            for timestamp in btc["timestamp"]
            if grid[0] <= timestamp < end_boundary
        ]
        result[date] = bool(
            h_start is not None
            and h_end is not None
            and h_start <= grid[0]
            and h_end >= end_boundary
            and len(window_timestamps) == EXPECTED_BARS
            and len(set(window_timestamps)) == EXPECTED_BARS
            and set(window_timestamps) == set(grid)
        )
    return result


def build_path_cache(
    wanted: pd.DataFrame,
) -> tuple[dict[tuple[str, object], dict], dict]:
    """Batch implementation of ``assess_15m_window`` semantics.

    The helper is called on a deterministic sample afterwards and exact
    metadata/path outcomes must agree.  A 09:10 decision can first execute at
    the 09:15 grid, so the window is [D 09:15, D+1 09:15).
    """
    pairs = wanted[["market", "date"]].drop_duplicates().copy()
    dates = sorted(pairs["date"].unique())
    cache: dict[tuple[str, object], dict] = {}
    with _connect_readonly(UPBIT_15M_DB) as conn:
        benchmark_ok = _benchmark_complete_dates(conn, dates)
        for market, group in pairs.groupby("market"):
            wanted_dates = sorted(group["date"].unique())
            q_start = (
                pd.Timestamp(min(wanted_dates)).normalize()
                - pd.Timedelta(days=2)
                + pd.Timedelta(hours=9)
            )
            q_end = (
                pd.Timestamp(max(wanted_dates)).normalize()
                + pd.Timedelta(days=2, hours=9)
            )
            raw = _load_market_rows(conn, str(market), q_start, q_end)
            if raw.empty:
                for date in wanted_dates:
                    cache[(str(market), date)] = {
                        "path_complete": False,
                        "path_quality": "market_missing",
                    }
                continue
            raw = raw.sort_values("timestamp").reset_index(drop=True)
            if raw["timestamp"].duplicated().any():
                raise RuntimeError(
                    f"duplicate 15m timestamps for {market}"
                )
            raw_ts = raw["timestamp"].to_numpy(dtype="datetime64[ns]")
            by_timestamp = {
                pd.Timestamp(row.timestamp): (
                    float(row.open),
                    float(row.high),
                    float(row.low),
                    float(row.close),
                )
                for row in raw.itertuples(index=False)
            }
            for date in wanted_dates:
                key = (str(market), date)
                if not benchmark_ok.get(date, False):
                    cache[key] = {
                        "path_complete": False,
                        "path_quality": "benchmark_incomplete",
                    }
                    continue
                grid = _expected_grid(date)
                window_end = grid[0] + BAR_FREQ * EXPECTED_BARS
                window_raw = raw[
                    (raw["timestamp"] >= grid[0])
                    & (raw["timestamp"] < window_end)
                ]
                if window_raw.empty:
                    cache[key] = {
                        "path_complete": False,
                        "path_quality": "target_no_observations",
                    }
                    continue
                if not window_raw["timestamp"].isin(set(grid)).all():
                    cache[key] = {
                        "path_complete": False,
                        "path_quality": "target_off_grid",
                    }
                    continue
                window_rows = {
                    ts: by_timestamp[ts] for ts in grid if ts in by_timestamp
                }
                start = grid[0]
                previous_close = None
                if start not in window_rows:
                    idx = int(np.searchsorted(raw_ts, np.datetime64(start), side="left")) - 1
                    if idx >= 0:
                        previous_close = float(raw.iloc[idx]["close"])
                    if (
                        previous_close is None
                        or not np.isfinite(previous_close)
                        or previous_close <= 0
                    ):
                        cache[key] = {
                            "path_complete": False,
                            "path_quality": "market_start_horizon_incomplete",
                        }
                        continue
                bars = []
                flat = 0
                invalid = False
                for ts in grid:
                    bar = window_rows.get(ts)
                    if bar is None:
                        if previous_close is None:
                            invalid = True
                            break
                        bar = (
                            previous_close,
                            previous_close,
                            previous_close,
                            previous_close,
                        )
                        flat += 1
                    o, high, low, close = bar
                    if (
                        not all(np.isfinite(v) and v > 0 for v in bar)
                        or high < max(o, close)
                        or low > min(o, close)
                        or high < low
                    ):
                        invalid = True
                        break
                    bars.append(bar)
                    previous_close = close
                if invalid or len(bars) != EXPECTED_BARS:
                    cache[key] = {
                        "path_complete": False,
                        "path_quality": "invalid_target_ohlc",
                    }
                    continue
                gross, outcome = simulate_path(bars, HARD_SL, TAKE_PROFIT, None)
                entry = float(bars[0][0])
                mfe = float(max(bar[1] for bar in bars) / entry - 1.0)
                mae = float(min(bar[2] for bar in bars) / entry - 1.0)
                exec_up10 = bool(mfe >= 0.10)
                exec_dn5 = bool(mae <= -0.05)
                cache[key] = {
                    "path_complete": True,
                    "path_quality": "complete" if flat == 0 else "flat_filled",
                    "flat_filled_bars": flat,
                    "raw_bars": len(window_rows),
                    "path_net": float(gross - ROUND_TRIP_COST),
                    "outcome": outcome,
                    "execution_entry": entry,
                    "execution_mfe": mfe,
                    "execution_mae": mae,
                    "execution_up10": exec_up10,
                    "execution_dn5": exec_dn5,
                    "execution_safe_up10": bool(exec_up10 and not exec_dn5),
                }

    # Contract cross-check against the canonical implementation.
    ordered_pairs = pairs.sort_values(["date", "market"])
    sample_keys = list(
        ordered_pairs.head(12).itertuples(index=False, name=None)
    )
    complete_keys = [
        key
        for key, value in cache.items()
        if value.get("path_complete", False)
    ][:12]
    sample_keys.extend(
        key for key in complete_keys if key not in sample_keys
    )
    checked = 0
    checked_complete = 0
    for market, date in sample_keys:
        key = (str(market), date)
        start_at = pd.Timestamp(date).normalize() + pd.Timedelta(
            hours=EXECUTION_START_HOUR,
            minutes=EXECUTION_START_MINUTE,
        )
        canonical = assess_15m_window(
            str(market), start_at, db_path=UPBIT_15M_DB
        )
        batched = cache[key]
        if canonical.path_complete != batched["path_complete"]:
            raise RuntimeError(f"path batch/canonical completeness mismatch: {key}")
        if canonical.path_complete:
            gross, outcome = simulate_path(
                canonical.bars, HARD_SL, TAKE_PROFIT, None
            )
            if (
                outcome != batched["outcome"]
                or not np.isclose(
                    gross - ROUND_TRIP_COST, batched["path_net"], atol=1e-12
                )
                or canonical.flat_filled_bars != batched["flat_filled_bars"]
            ):
                raise RuntimeError(f"path batch/canonical outcome mismatch: {key}")
            entry = float(canonical.bars[0][0])
            canonical_mfe = max(bar[1] for bar in canonical.bars) / entry - 1.0
            canonical_mae = min(bar[2] for bar in canonical.bars) / entry - 1.0
            if (
                not np.isclose(
                    canonical_mfe, batched["execution_mfe"], atol=1e-12
                )
                or not np.isclose(
                    canonical_mae, batched["execution_mae"], atol=1e-12
                )
            ):
                raise RuntimeError(f"path batch/canonical MFE/MAE mismatch: {key}")
            checked_complete += 1
        checked += 1
    complete_n = sum(bool(v.get("path_complete")) for v in cache.values())
    meta = {
        "wanted_pairs": len(pairs),
        "complete_pairs": complete_n,
        "complete_rate": complete_n / len(pairs) if len(pairs) else np.nan,
        "canonical_crosscheck_n": checked,
        "canonical_complete_crosscheck_n": checked_complete,
        "rule": (
            "BTC 96-grid complete + next boundary observed; target-only gaps "
            "flat-filled from previous close"
        ),
        "execution_start": "09:15 KST (09:10 decision rounded to next 15m grid)",
        "window": "[D 09:15, D+1 09:15)",
    }
    return cache, meta


def _complete_daily_path_metrics(
    picks: pd.DataFrame,
    path_cache: dict[tuple[str, object], dict],
    allowed_dates: set | None = None,
) -> pd.DataFrame:
    rows = []
    for pick in picks.itertuples(index=False):
        path = path_cache.get((str(pick.market), pick.date), {})
        if not path.get("path_complete", False):
            continue
        rows.append(
            {
                "date": pick.date,
                "net": float(path["path_net"]),
                "safe_up10": float(path["execution_safe_up10"]),
                "up10": float(path["execution_up10"]),
                "dn5": float(path["execution_dn5"]),
                "tp5_before_sl3": float(path["outcome"] == "tp"),
            }
        )
    if not rows:
        return pd.DataFrame()
    trades = pd.DataFrame(rows)
    if allowed_dates is not None:
        trades = trades[trades["date"].isin(allowed_dates)]
        if trades.empty:
            return pd.DataFrame()
    counts = trades.groupby("date").size()
    full_dates = set(counts[counts == 3].index)
    trades = trades[trades["date"].isin(full_dates)]
    if trades.empty:
        return pd.DataFrame()
    return trades.groupby("date").agg(
        net=("net", "mean"),
        safe_up10=("safe_up10", "mean"),
        up10=("up10", "mean"),
        dn5=("dn5", "mean"),
        tp5_before_sl3=("tp5_before_sl3", "mean"),
        n_picks=("net", "size"),
    )


def paired_day_bootstrap(
    all_picks: dict[str, pd.DataFrame],
    path_cache: dict[tuple[str, object], dict],
    *,
    candidate: str,
    baselines: tuple[str, ...] = ("cls_core", "random_seed42", "atr_top3"),
    draws: int = PAIRED_BOOTSTRAP_DRAWS,
) -> pd.DataFrame:
    """Locked-holdout paired day bootstrap on dates with 3 complete picks each."""
    if candidate not in all_picks:
        raise RuntimeError(f"paired candidate absent: {candidate}")
    common_dates = _common_complete_path_dates(all_picks, path_cache)
    candidate_daily = _complete_daily_path_metrics(
        all_picks[candidate], path_cache, common_dates
    )
    rows = []
    metrics = ["safe_up10", "up10", "dn5", "tp5_before_sl3", "net"]
    for baseline in baselines:
        if baseline not in all_picks or baseline == candidate:
            continue
        baseline_daily = _complete_daily_path_metrics(
            all_picks[baseline], path_cache, common_dates
        )
        common = candidate_daily.join(
            baseline_daily, how="inner", lsuffix="_candidate", rsuffix="_baseline"
        )
        if common.empty:
            continue
        rng = np.random.default_rng(MODEL_SEED)
        n_days = len(common)
        sample_index = rng.integers(0, n_days, size=(draws, n_days))
        for metric in metrics:
            candidate_values = common[f"{metric}_candidate"].to_numpy(dtype=float)
            baseline_values = common[f"{metric}_baseline"].to_numpy(dtype=float)
            paired_delta = candidate_values - baseline_values
            boot_delta = paired_delta[sample_index].mean(axis=1)
            rows.append(
                {
                    "evaluation_scope": "locked_holdout",
                    "candidate": candidate,
                    "baseline": baseline,
                    "metric": metric,
                    "n_common_days": n_days,
                    "n_picks_each": n_days * 3,
                    "candidate_mean": float(candidate_values.mean()),
                    "baseline_mean": float(baseline_values.mean()),
                    "delta_candidate_minus_baseline": float(
                        paired_delta.mean()
                    ),
                    "ci95_low": float(np.quantile(boot_delta, 0.025)),
                    "ci95_high": float(np.quantile(boot_delta, 0.975)),
                    "bootstrap_draws": draws,
                    "seed": MODEL_SEED,
                }
            )
    return pd.DataFrame(rows)


def calibration_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scope, variant), frame in predictions.groupby(
        ["evaluation_scope", "variant"]
    ):
        ranked = frame["p_cal"].rank(method="first")
        try:
            buckets = pd.qcut(ranked, 10, labels=False, duplicates="drop")
        except ValueError:
            continue
        work = frame.assign(cal_bucket=buckets)
        for bucket, group in work.groupby("cal_bucket"):
            rows.append(
                {
                    "evaluation_scope": scope,
                    "variant": variant,
                    "bucket": int(bucket),
                    "n": len(group),
                    "pred_mean": float(group["p_cal"].mean()),
                    "actual_up10": float(group["y_up10"].mean()),
                    "gap": float(
                        group["p_cal"].mean() - group["y_up10"].mean()
                    ),
                    "raw_score_min": float(group["score_raw"].min()),
                    "raw_score_max": float(group["score_raw"].max()),
                }
            )
    return pd.DataFrame(rows)


def evaluate(
    predictions: pd.DataFrame,
    *,
    include_paths: bool,
    paired_candidate: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    if predictions.empty:
        raise RuntimeError("evaluation predictions are empty")
    if predictions.duplicated(["variant", "date", "market"]).any():
        raise RuntimeError("evaluation predictions contain duplicate identities")
    reference = predictions[predictions["variant"] == "cls_core"].copy()
    if reference.empty:
        raise RuntimeError("cls_core reference predictions are missing")
    reference_keys = set(
        reference[["date", "market"]].itertuples(
            index=False, name=None
        )
    )
    for variant, frame in predictions.groupby("variant"):
        keys = set(
            frame[["date", "market"]].itertuples(
                index=False, name=None
            )
        )
        if keys != reference_keys:
            raise RuntimeError(
                f"{variant} is not aligned to the cls_core cohort"
            )
        scores = frame[["score_raw", "p_cal"]].to_numpy(dtype=float)
        if not np.isfinite(scores).all():
            raise RuntimeError(f"{variant} contains nonfinite scores")
    model_picks = {
        variant: _top3(frame, "score_raw")
        for variant, frame in predictions.groupby("variant")
    }
    baseline_picks = _baseline_predictions(reference)
    all_picks = {**model_picks, **baseline_picks}

    path_cache = None
    common_path_dates: set | None = None
    path_meta: dict = {"enabled": False}
    paired = pd.DataFrame()
    if include_paths:
        wanted = pd.concat(
            [
                frame[["market", "date"]]
                for frame in all_picks.values()
            ],
            ignore_index=True,
        ).drop_duplicates()
        log.info("loading canonical path outcomes for %d unique pairs", len(wanted))
        path_cache, path_meta = build_path_cache(wanted)
        path_meta["enabled"] = True
        common_path_dates = _common_complete_path_dates(
            all_picks, path_cache
        )
        path_meta["all_policy_common_complete_dates"] = int(
            len(common_path_dates)
        )
        if paired_candidate is not None:
            paired = paired_day_bootstrap(
                all_picks,
                path_cache,
                candidate=paired_candidate,
            )
            path_meta["paired_common"] = (
                paired[
                    [
                        "baseline",
                        "n_common_days",
                        "n_picks_each",
                    ]
                ]
                .drop_duplicates()
                .to_dict(orient="records")
                if not paired.empty
                else []
            )

    rows = []
    for variant, frame in predictions.groupby("variant"):
        metrics = _prediction_metrics(frame)
        metrics.update(
            _pick_metrics(
                reference,
                model_picks[variant],
                path_cache,
                common_path_dates,
            )
        )
        metrics.update(
            {
                "variant": variant,
                "evaluation_scope": str(frame["evaluation_scope"].iloc[0]),
                "category": (
                    "legacy_reference"
                    if variant == LEGACY_REFERENCE
                    else "model_trial"
                ),
            }
        )
        rows.append(metrics)
    for name, picks in baseline_picks.items():
        metrics = _pick_metrics(
            reference,
            picks,
            path_cache,
            common_path_dates,
        )
        metrics.update(
            {
                "variant": name,
                "evaluation_scope": str(reference["evaluation_scope"].iloc[0]),
                "category": "baseline",
            }
        )
        rows.append(metrics)
    # Monkey expectation for label/EOD metrics; exact bracket baseline is the
    # deterministic random_seed42 row above.
    universe = {
        "variant": "universe_monkey_expectation",
        "evaluation_scope": str(reference["evaluation_scope"].iloc[0]),
        "category": "baseline",
        "top3_n": len(reference),
        "top3_dates": int(reference["date"].nunique()),
        "label_top3_up10_day_open": float(reference["y_up10"].mean()),
        "label_top3_safe_up10_day_open": float(
            (
                (reference["y_up10"] == 1)
                & (reference[RAW_DOWN_COL] > -0.05)
            ).mean()
        ),
        "label_top3_lift_vs_universe": 1.0,
        "label_top3_matched_vol_base_up10": float(
            reference["y_up10"].mean()
        ),
        "top3_within_vol_lift": 1.0,
        "label_top3_mfe_mean_day_open": float(reference[RAW_UP_COL].mean()),
        "label_top3_mfe_atr_multiple": float(
            reference["mfe_atr_multiple"].mean()
        ),
        "label_top3_dn5_day_open": float(
            (reference[RAW_DOWN_COL] <= -0.05).mean()
        ),
        "label_top3_eod_net_mean_day_open": float(
            (reference[EOD_COL] - ROUND_TRIP_COST).mean()
        ),
        "top3_up10": np.nan,
        "top3_safe_up10": np.nan,
        "top3_dn5": np.nan,
    }
    rows.append(universe)
    return (
        pd.DataFrame(rows),
        calibration_table(predictions),
        path_meta,
        paired,
    )


def select_challenger(discovery_metrics: pd.DataFrame) -> tuple[str, dict]:
    """Predeclared selection using discovery OOF only.

    User utility is encoded directly: maximize a +10% touch that did *not*
    suffer a -5% excursion.  Within-volatility lift is the first tie-break so
    an ATR repackaging cannot win on pooled tail rate alone, followed by lower
    dn5 and then exact 15m net.  Holdout columns do not exist when this runs.
    """
    eligible = discovery_metrics[
        (discovery_metrics["category"] == "model_trial")
        & discovery_metrics["variant"].isin(TRIALS)
    ].copy()
    if eligible.empty:
        raise RuntimeError("no discovery model trial eligible for selection")
    for column, fallback in [
        ("top3_safe_up10", -np.inf),
        ("top3_within_vol_lift", -np.inf),
        ("top3_dn5", np.inf),
        ("path_net_mean", -np.inf),
    ]:
        if column not in eligible:
            eligible[column] = fallback
        eligible[column] = eligible[column].fillna(fallback)
    eligible = eligible.sort_values(
        [
            "top3_safe_up10",
            "top3_within_vol_lift",
            "top3_dn5",
            "path_net_mean",
            "variant",
        ],
        ascending=[False, False, True, False, True],
    )
    selected = str(eligible.iloc[0]["variant"])
    audit = {
        "selected": selected,
        "scope": "discovery_oof_only",
        "primary": "top3_safe_up10 = up10 and not dn5",
        "tie_breaks": [
            "top3_within_vol_lift desc",
            "top3_dn5 asc",
            "path_net_mean desc",
            "variant asc deterministic",
        ],
        "ranked_candidates": eligible[
            [
                "variant",
                "top3_safe_up10",
                "top3_within_vol_lift",
                "top3_up10",
                "top3_dn5",
                "path_net_mean",
            ]
        ].to_dict(orient="records"),
    }
    return selected, audit


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _artifact_paths(prefix: Path) -> dict[str, Path]:
    return {
        "predictions": Path(f"{prefix}_predictions.csv.gz"),
        "metrics": Path(f"{prefix}_metrics.csv"),
        "calibration": Path(f"{prefix}_calibration.csv"),
        "paired_bootstrap": Path(f"{prefix}_paired_bootstrap.csv"),
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
        raise RuntimeError("upside artifact manifest is unreadable") from exc
    if payload.get("schema") != "upside_head_challenger_v1_manifest":
        raise RuntimeError("upside artifact manifest schema mismatch")
    files = payload.get("files")
    if not isinstance(files, dict) or set(files) != set(expected):
        raise RuntimeError("upside artifact manifest file set mismatch")
    for name, expected_path in expected.items():
        entry = files.get(name)
        if not isinstance(entry, dict):
            raise RuntimeError(f"upside manifest entry is invalid: {name}")
        recorded_path = Path(str(entry.get("path", "")))
        if not recorded_path.is_absolute():
            recorded_path = ROOT / recorded_path
        if recorded_path.resolve() != expected_path.resolve():
            raise RuntimeError(f"upside manifest path mismatch: {name}")
        if not expected_path.is_file():
            raise RuntimeError(f"upside artifact is missing: {expected_path}")
        if int(entry.get("bytes", -1)) != expected_path.stat().st_size:
            raise RuntimeError(f"upside artifact size mismatch: {name}")
        if entry.get("sha256") != _sha256(expected_path):
            raise RuntimeError(f"upside artifact checksum mismatch: {name}")
    return payload


def validate_existing_artifacts(
    *,
    output_prefix: Path,
    upbit_d1_db: Path,
    binance_d1_db: Path,
    upbit_15m_db: Path,
) -> dict:
    """Reject stale, tampered, or source-divergent upside results."""
    expected = _artifact_paths(output_prefix)
    manifest_path = Path(f"{output_prefix}_manifest.json")
    _verify_manifest(manifest_path=manifest_path, expected=expected)
    try:
        coverage = json.loads(
            expected["coverage"].read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("upside coverage is unreadable") from exc
    if coverage.get("schema") != "upside_head_challenger_v1":
        raise RuntimeError("upside coverage schema mismatch")
    lineage = coverage.get("input_lineage")
    if not isinstance(lineage, dict):
        raise RuntimeError("upside input lineage is missing")
    if lineage.get("script_sha256") != _sha256(Path(__file__)):
        raise RuntimeError("upside artifacts were built by stale code")
    if lineage.get("code_lineage") != _code_lineage():
        raise RuntimeError("upside local code dependencies have changed")
    recorded_signatures = lineage.get("sources")
    if not isinstance(recorded_signatures, dict):
        raise RuntimeError("upside source signatures are missing")
    normalized_recorded = {}
    for signature in recorded_signatures.values():
        if not isinstance(signature, dict) or not signature.get("path"):
            raise RuntimeError("upside source signature is malformed")
        normalized_recorded[str(signature["path"])] = signature
    m15_resolved = str(upbit_15m_db.resolve())
    uses_paths = m15_resolved in normalized_recorded
    current_paths = [upbit_d1_db, binance_d1_db]
    if uses_paths:
        current_paths.append(upbit_15m_db)
    current_signatures = {
        signature["path"]: signature
        for signature in (
            _file_signature(path) for path in current_paths
        )
    }
    if normalized_recorded != current_signatures:
        raise RuntimeError("upside source inputs have changed")
    return {
        "status": "valid",
        "schema": coverage["schema"],
        "output_prefix": str(output_prefix),
        "uses_15m_paths": uses_paths,
        "verdict": coverage.get("verdict"),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leak-free fixed-label upper-head challenger study"
    )
    parser.add_argument(
        "--variants",
        default=",".join(TRIALS),
        help="comma-separated subset of the six predeclared trials",
    )
    parser.add_argument("--limit-markets", type=int, default=None)
    parser.add_argument(
        "--no-path",
        action="store_true",
        help="skip exact 15m TP5/SL3 path evaluation",
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
            output_prefix=OUT_PREFIX,
            upbit_d1_db=UPBIT_D1_DB,
            binance_d1_db=BINANCE_D1_DB,
            upbit_15m_db=UPBIT_15M_DB,
        )
        print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
        return
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = sorted(set(variants) - set(TRIALS))
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}; allowed={list(TRIALS)}")
    if len(variants) != len(set(variants)):
        raise SystemExit("duplicate variants are not allowed")
    if "cls_core" not in variants:
        raise SystemExit("cls_core reference must be included in every run")
    if args.limit_markets is not None and args.limit_markets < UNIVERSE_TOP_N:
        raise ValueError("--limit-markets must be >= 100")
    required_inputs = [UPBIT_D1_DB, BINANCE_D1_DB]
    if not args.no_path:
        required_inputs.append(UPBIT_15M_DB)
    missing_inputs = [
        str(path) for path in required_inputs if not path.is_file()
    ]
    if missing_inputs:
        raise FileNotFoundError(f"required inputs missing: {missing_inputs}")
    source_signatures = {
        str(path): _file_signature(path) for path in required_inputs
    }
    code_lineage = _code_lineage()

    OUT_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    log.info(
        "SELECTION CONTRACT: %d/%d predeclared trials; no HP sweep; label=up10 fixed",
        len(variants),
        len(TRIALS),
    )
    panel, panel_meta = prepare_panel(args.limit_markets)
    log.info(
        "panel rows=%d markets=%d dates=%d window=%s..%s base=%.4f bvol_cov=%.3f",
        panel_meta["rows"],
        panel_meta["markets"],
        panel_meta["dates"],
        panel_meta["date_min"],
        panel_meta["date_max"],
        panel_meta["up10_base_rate"],
        panel_meta["binance"]["row_coverage"],
    )
    all_dates = np.sort(panel["date"].unique())
    if len(all_dates) <= LOCKED_HOLDOUT_DAYS + 365:
        raise RuntimeError("not enough dates for discovery plus locked holdout")
    holdout_dates = all_dates[-LOCKED_HOLDOUT_DAYS:]
    discovery_dates = all_dates[:-LOCKED_HOLDOUT_DAYS]
    discovery_panel = panel[panel["date"].isin(set(discovery_dates))].copy()
    holdout_hash = hashlib.sha256(
        "\n".join(map(str, holdout_dates)).encode("utf-8")
    ).hexdigest()
    log.info(
        "LOCKED HOLDOUT sealed: %d dates %s..%s hash=%s",
        len(holdout_dates),
        holdout_dates[0],
        holdout_dates[-1],
        holdout_hash[:16],
    )

    discovery_predictions, discovery_fold_meta = run_walk_forward(
        discovery_panel, variants, evaluation_scope="discovery_oof"
    )
    (
        discovery_metrics,
        discovery_calibration,
        discovery_path_meta,
        _,
    ) = evaluate(
        discovery_predictions, include_paths=not args.no_path
    )
    selected_variant, selection_audit = select_challenger(discovery_metrics)
    log.info(
        "DISCOVERY SELECTION frozen before holdout: %s (safe_up10 primary)",
        selected_variant,
    )

    holdout_predictions, holdout_fold_meta = run_locked_holdout(
        panel, selected_variant, holdout_dates
    )
    (
        holdout_metrics,
        holdout_calibration,
        holdout_path_meta,
        paired_bootstrap,
    ) = evaluate(
        holdout_predictions,
        include_paths=not args.no_path,
        paired_candidate=selected_variant,
    )
    predictions = pd.concat(
        [discovery_predictions, holdout_predictions], ignore_index=True
    )
    metrics = pd.concat(
        [discovery_metrics, holdout_metrics], ignore_index=True
    )
    calibration = pd.concat(
        [discovery_calibration, holdout_calibration], ignore_index=True
    )
    source_signatures_after = {
        str(path): _file_signature(path) for path in required_inputs
    }
    if source_signatures_after != source_signatures:
        raise RuntimeError("research input changed while challenger was running")
    if _code_lineage() != code_lineage:
        raise RuntimeError("local code changed while challenger was running")

    predictions_path = Path(f"{OUT_PREFIX}_predictions.csv.gz")
    metrics_path = Path(f"{OUT_PREFIX}_metrics.csv")
    calibration_path = Path(f"{OUT_PREFIX}_calibration.csv")
    paired_path = Path(f"{OUT_PREFIX}_paired_bootstrap.csv")
    coverage_path = Path(f"{OUT_PREFIX}_coverage.json")
    selected_holdout_row = holdout_metrics[
        holdout_metrics["variant"] == selected_variant
    ].iloc[0]
    availability_audit = operational_binance_availability()
    selected_uses_binance = TRIAL_SPECS[selected_variant].use_binance
    paired_core = paired_bootstrap[
        paired_bootstrap["baseline"] == "cls_core"
    ]
    paired_core_safe = paired_core[paired_core["metric"] == "safe_up10"]
    paired_core_net = paired_core[paired_core["metric"] == "net"]
    promotion_gates = {
        "holdout_path_net_positive": bool(
            selected_holdout_row.get("path_net_mean", np.nan) > 0
        ),
        "safe_up10_delta_vs_core_ci_low_positive": bool(
            not paired_core_safe.empty
            and paired_core_safe.iloc[0]["ci95_low"] > 0
        ),
        "net_delta_vs_core_ci_low_positive": bool(
            not paired_core_net.empty
            and paired_core_net.iloc[0]["ci95_low"] > 0
        ),
        "fresh_features_available_at_send": bool(
            not selected_uses_binance
            or availability_audit["fresh_d1_available_at_r1_send"]
        ),
        "point_in_time_history_ge_70": bool(
            panel_meta["point_in_time_min_history"] >= 70
        ),
    }
    verdict = (
        "REJECT"
        if (
            not promotion_gates["holdout_path_net_positive"]
            or not promotion_gates["fresh_features_available_at_send"]
        )
        else "RESEARCH_ONLY"
    )

    coverage = {
        "schema": "upside_head_challenger_v1",
        "created_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "target": "day-D high/open - 1 >= 0.10 (unchanged)",
        "input_lineage": {
            "sources_before_and_after_identical": True,
            "sources": source_signatures,
            "script_sha256": _sha256(Path(__file__)),
            "code_lineage": code_lineage,
        },
        "feature_boundary": "<= D-1",
        "universe": f"D-1 quote-volume top {UNIVERSE_TOP_N}",
        "outer_validation": {
            "kind": "expanding walk-forward",
            "n_folds": OUTER_FOLDS,
            "embargo_days": EMBARGO_DAYS,
        },
        "locked_holdout": {
            "n_dates": LOCKED_HOLDOUT_DAYS,
            "date_start": str(holdout_dates[0]),
            "date_end": str(holdout_dates[-1]),
            "dates_sha256": holdout_hash,
            "unlocked_after_discovery_selection": True,
            "selected_variant": selected_variant,
            "selection_audit": selection_audit,
            "evaluated_model_variants": sorted(
                holdout_predictions["variant"].unique().tolist()
            ),
        },
        "calibration": {
            "kind": "monotone isotonic",
            "source": "inner expanding OOF of outer-train only",
            "inner_folds": INNER_FOLDS,
            "embargo_days": EMBARGO_DAYS,
        },
        "cost_round_trip": ROUND_TRIP_COST,
        "path_policy": (
            "09:10 decision -> [D 09:15,D+1 09:15) exact 15m "
            "SL-3%/TP+5%/EOD, same-bar SL first"
        ),
        "operational_binance_availability": availability_audit,
        "verdict": verdict,
        "promotion_gates": promotion_gates,
        "trial_count_total": len(TRIALS),
        "trial_count_run": len(variants),
        "trials": variants,
        "legacy_reference": LEGACY_REFERENCE if "cls_core" in variants else None,
        "hyperparameter_sweep": False,
        "panel": panel_meta,
        "folds": {
            "discovery": discovery_fold_meta,
            "locked_holdout": holdout_fold_meta,
        },
        "path": {
            "discovery": discovery_path_meta,
            "locked_holdout": holdout_path_meta,
        },
        "artifacts": {
            "predictions": str(predictions_path.relative_to(ROOT)),
            "metrics": str(metrics_path.relative_to(ROOT)),
            "calibration": str(calibration_path.relative_to(ROOT)),
            "paired_bootstrap": str(paired_path.relative_to(ROOT)),
        },
        "production_modified": False,
        "auto_order_code": False,
    }
    artifact_paths = _artifact_paths(OUT_PREFIX)
    manifest_path = Path(f"{OUT_PREFIX}_manifest.json")
    with tempfile.TemporaryDirectory(
        dir=OUT_PREFIX.parent,
        prefix=f".{OUT_PREFIX.name}.generation.",
    ) as stage_directory:
        stage_root = Path(stage_directory)
        staged = {
            name: stage_root / path.name
            for name, path in artifact_paths.items()
        }
        predictions.to_csv(
            staged["predictions"],
            index=False,
            compression=GZIP_COMPRESSION,
            float_format="%.17g",
        )
        metrics.to_csv(staged["metrics"], index=False)
        calibration.to_csv(staged["calibration"], index=False)
        paired_bootstrap.to_csv(staged["paired_bootstrap"], index=False)
        _write_json(staged["coverage"], coverage)
        staged_manifest = stage_root / manifest_path.name
        _write_json(
            staged_manifest,
            {
                "schema": "upside_head_challenger_v1_manifest",
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
        publish_map[staged_manifest] = manifest_path
        _publish_staged_files(publish_map)

    display = [
        "evaluation_scope",
        "variant",
        "category",
        "raw_auc",
        "calibrated_auc",
        "macro_within_date_auc",
        "macro_within_vol_auc",
        "calibrated_brier",
        "score_atr_spearman",
        "top3_up10",
        "top3_safe_up10",
        "label_top3_up10_day_open",
        "label_top3_lift_vs_universe",
        "top3_within_vol_lift",
        "top3_dn5",
        "label_top3_eod_net_mean_day_open",
        "path_net_mean",
        "path_tp5_before_sl3",
        "path_pct_sl",
    ]
    display = [c for c in display if c in metrics.columns]
    log.info("RESULTS\n%s", metrics[display].to_string(index=False))
    log.info(
        "wrote %s, %s, %s, %s, %s",
        metrics_path,
        predictions_path,
        calibration_path,
        paired_path,
        coverage_path,
    )


if __name__ == "__main__":
    main()

"""Fixed downside-semivolatility diagnostic for the 09:15 FP head.

Research-only boundary
----------------------
This script never mutates production recommendation, alert, ledger, or
deployment code.  It performs exactly one pre-specified feature comparison:

* core: the existing 24 D-1 precursor features;
* augmented: the same 24 features plus four close-to-close semivolatility
  features.

The four additions are calculated per market from completed daily closes and
then shifted by one row.  Therefore a candidate for date D can use data only
through D-1.  The windows and formulas are fixed:

``downside_semivol_7``
    sqrt(mean(min(close_return, 0)^2), 7 completed returns)
``downside_semivol_21``
    sqrt(mean(min(close_return, 0)^2), 21 completed returns)
``upside_semivol_21``
    sqrt(mean(max(close_return, 0)^2), 21 completed returns)
``semivol_asym_21``
    upside_semivol_21 - downside_semivol_21

Both variants predict the exact existing ``label_fp_safe10`` target on the
cached [D 09:15, D+1 09:15) 96-bar paths.  They share the same deterministic
single-thread XGBoost model, five outer expanding folds, three true inner OOF
isotonic folds, and five-date embargo.  There is no tuning or variant search.

One benchmark-complete final-180 schedule is shared with the schedule-aligned
SafeUp/R1 baseline and first-passage run.  Identity includes schedule hash,
scope, fold, date, and market.  This is a contaminated historical diagnostic,
not clean preregistered evidence.  Even a passing result can therefore be at
most ``FORWARD_SHADOW_CANDIDATE``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.database import load_candles  # noqa: E402
from ops.code_lineage import python_code_lineage  # noqa: E402
import scripts.first_passage_head_challenger_v1 as fp  # noqa: E402
import scripts.safeup_head_challenger_v1 as safeup  # noqa: E402


D1_DB = ROOT / "data" / "upbit_d1.db"
PATH_PANEL = (
    ROOT / "output" / "first_passage_head_challenger_v1_path_panel.csv.gz"
)
PATH_PANEL_META = (
    ROOT / "output" / "first_passage_head_challenger_v1_path_panel_meta.json"
)
BASELINE_PREDICTIONS = (
    fp.BASELINE_PREDICTIONS
)
CORE_REFERENCE = (
    ROOT / "output" / "first_passage_head_challenger_v1_predictions.csv.gz"
)
OUT_PREFIX = ROOT / "output" / "semivol_joint_challenger_v1"

MIN_PRIOR_HISTORY = 70
UNIVERSE_TOP_N = 100
TOP_K = 3
OUTER_FOLDS = 5
INNER_FOLDS = 3
EMBARGO_DATES = 5
BOOTSTRAP_DRAWS = 5_000
ROUND_TRIP_COST = 0.0015
TARGET = "label_fp_safe10"
CORE_POLICY = "core_fp_head"
AUGMENTED_POLICY = "semivol_joint_head"
GZIP_COMPRESSION = {"method": "gzip", "mtime": 0}

SEMIVOL_FEATURES = [
    "downside_semivol_7",
    "downside_semivol_21",
    "upside_semivol_21",
    "semivol_asym_21",
]
COMPARATORS = (
    CORE_POLICY,
    "R1_repaired",
    "lowest_ATR",
    "liquidity_matched",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _code_lineage() -> dict:
    return python_code_lineage(entrypoint=Path(__file__), root=ROOT)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )


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


def _load_path_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    panel = pd.read_csv(path, float_precision="round_trip")
    panel = fp._validate_labeled_panel(panel)

    required = {
        "market",
        "date",
        "history_prior_bars",
        "f_qv_rank",
        "f_atr_pct_14",
        "f_atr_xs_decile",
        "vol_band",
        "path_complete",
        "benchmark_complete",
        TARGET,
        "path_up10",
        "path_dn5",
        "path_bracket_outcome",
        "path_bracket_net",
    }
    missing = sorted(required - set(panel.columns))
    if missing:
        raise RuntimeError(f"path panel columns missing: {missing}")
    return panel.sort_values(["date", "market"]).reset_index(drop=True)


def _validate_path_panel_lineage(
    *,
    panel_path: Path,
    meta_path: Path,
    d1_db: Path,
) -> dict:
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("path-panel metadata is unreadable") from exc
    signature = metadata.get("signature")
    path_meta = metadata.get("path_meta")
    if not isinstance(signature, dict) or not isinstance(path_meta, dict):
        raise RuntimeError("path-panel metadata is missing required objects")
    if signature.get("schema") != fp.PATH_CACHE_SCHEMA:
        raise RuntimeError(
            "path-panel cache schema is stale or incompatible: "
            f"{signature.get('schema')!r}"
        )
    if metadata.get("cache_sha256") != _sha256(panel_path):
        raise RuntimeError("path-panel bytes do not match cache metadata")
    d1_signature = signature.get("d1_db")
    if not isinstance(d1_signature, dict) or d1_signature != (
        fp._file_signature(d1_db)
    ):
        raise RuntimeError("path-panel D1 source does not match requested DB")
    if signature.get("safeup_script_sha256") != _sha256(
        Path(safeup.__file__)
    ):
        raise RuntimeError("path-panel feature/path producer code has changed")
    if signature.get("code_lineage") != fp._code_lineage():
        raise RuntimeError(
            "path-panel local code dependencies have changed"
        )
    if signature.get("completed_label_cutoff") != str(
        safeup._completed_label_cutoff().date()
    ):
        raise RuntimeError("path-panel completed-label cutoff is stale")
    return metadata


def _market_semivol(market: str, d1_db: Path) -> pd.DataFrame:
    candles = load_candles(str(d1_db), market)
    if candles is None or candles.empty:
        raise RuntimeError(f"no D1 candles for {market}")
    frame = candles[["timestamp", "close"]].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    if frame["timestamp"].duplicated().any():
        raise RuntimeError(f"duplicate D1 timestamps for {market}")
    close = pd.to_numeric(frame["close"], errors="coerce")
    ret = close.pct_change(fill_method=None)
    downside_sq = np.minimum(ret, 0.0).pow(2)
    upside_sq = np.maximum(ret, 0.0).pow(2)

    # The rolling statistic at t is shifted to candidate t+1, so its newest
    # return is the completed close-to-close return ending on D-1.
    downside_7 = np.sqrt(
        downside_sq.rolling(7, min_periods=7).mean()
    ).shift(1)
    downside_21 = np.sqrt(
        downside_sq.rolling(21, min_periods=21).mean()
    ).shift(1)
    upside_21 = np.sqrt(
        upside_sq.rolling(21, min_periods=21).mean()
    ).shift(1)
    source_date = frame["timestamp"].dt.normalize().shift(1)
    result = pd.DataFrame(
        {
            "market": market,
            "date": frame["timestamp"].dt.date,
            "semivol_source_date": source_date.dt.date,
            "downside_semivol_7": downside_7,
            "downside_semivol_21": downside_21,
            "upside_semivol_21": upside_21,
        }
    )
    result["semivol_asym_21"] = (
        result["upside_semivol_21"]
        - result["downside_semivol_21"]
    )
    return result


def attach_semivol_features(
    panel: pd.DataFrame,
    d1_db: Path,
) -> tuple[pd.DataFrame, dict]:
    frames = [
        _market_semivol(str(market), d1_db)
        for market in sorted(panel["market"].unique())
    ]
    feature_panel = pd.concat(frames, ignore_index=True)
    feature_panel = feature_panel[
        feature_panel["date"].isin(set(panel["date"]))
    ].copy()
    if feature_panel.duplicated(["date", "market"]).any():
        raise RuntimeError("semivol feature panel has duplicate keys")
    merged = panel.merge(
        feature_panel,
        on=["date", "market"],
        how="left",
        validate="one_to_one",
    )
    missing_by_feature = {
        column: int(merged[column].isna().sum())
        for column in SEMIVOL_FEATURES
    }
    if any(missing_by_feature.values()):
        raise RuntimeError(
            f"semivol features are missing: {missing_by_feature}"
        )
    target_date = pd.to_datetime(merged["date"])
    source_date = pd.to_datetime(merged["semivol_source_date"])
    source_boundary = target_date - pd.Timedelta(days=1)
    source_violations = int(
        (source_date.isna() | (source_date > source_boundary)).sum()
    )
    if source_violations:
        raise RuntimeError(
            "semivol feature source is later than D-1: "
            f"violations={source_violations}"
        )
    nonfinite = {
        column: int(
            (~np.isfinite(pd.to_numeric(merged[column]))).sum()
        )
        for column in SEMIVOL_FEATURES
    }
    if any(nonfinite.values()):
        raise RuntimeError(
            f"semivol features contain nonfinite values: {nonfinite}"
        )
    if (
        merged[
            [
                "downside_semivol_7",
                "downside_semivol_21",
                "upside_semivol_21",
            ]
        ]
        < 0
    ).any().any():
        raise RuntimeError("semivol magnitude is negative")
    asym_error = (
        merged["semivol_asym_21"]
        - (
            merged["upside_semivol_21"]
            - merged["downside_semivol_21"]
        )
    ).abs()
    if float(asym_error.max()) > 1e-15:
        raise RuntimeError("semivol asymmetry formula mismatch")

    lag_days = (target_date - source_date).dt.days
    audit = {
        "formula": {
            "return": "close_t / close_(t-1) - 1",
            "downside_semivol_7": (
                "shift1(sqrt(rolling7 mean(min(return,0)^2)))"
            ),
            "downside_semivol_21": (
                "shift1(sqrt(rolling21 mean(min(return,0)^2)))"
            ),
            "upside_semivol_21": (
                "shift1(sqrt(rolling21 mean(max(return,0)^2)))"
            ),
            "semivol_asym_21": (
                "upside_semivol_21 - downside_semivol_21"
            ),
            "minimum_periods_equal_window": True,
        },
        "rows": int(len(merged)),
        "markets": int(merged["market"].nunique()),
        "feature_source_boundary": "<= D-1",
        "feature_source_date_violations": source_violations,
        "missing_by_feature": missing_by_feature,
        "nonfinite_by_feature": nonfinite,
        "source_lag_days_min": int(lag_days.min()),
        "source_lag_days_max": int(lag_days.max()),
        "asymmetry_max_abs_error": float(asym_error.max()),
        "descriptive": {
            column: {
                "min": float(merged[column].min()),
                "median": float(merged[column].median()),
                "max": float(merged[column].max()),
            }
            for column in SEMIVOL_FEATURES
        },
    }
    return merged, audit


def _base_columns(frame: pd.DataFrame) -> pd.DataFrame:
    base = fp._base_columns(frame)
    for column in SEMIVOL_FEATURES + ["semivol_source_date"]:
        base[column] = frame[column].to_numpy()
    return base


def _validate_model_metadata(metadata: dict, context: str) -> None:
    inner = metadata.get("inner_folds", [])
    if len(inner) != INNER_FOLDS:
        raise RuntimeError(
            f"{context}: expected {INNER_FOLDS} inner OOF folds, "
            f"got {len(inner)}"
        )
    for fold in inner:
        train_end = pd.Timestamp(fold["train_end"])
        validation_start = pd.Timestamp(fold["validation_start"])
        if (
            pd.isna(train_end)
            or pd.isna(validation_start)
            or validation_start <= train_end
        ):
            raise RuntimeError(f"{context}: inner time order violated")
    all_used = all(
        bool(fold.get("used_for_calibration", False))
        for fold in inner
    )
    if bool(metadata.get("isotonic_fitted")) and not all_used:
        raise RuntimeError(
            f"{context}: partial inner folds fitted a calibrator"
        )


def _predict_both(
    train: pd.DataFrame,
    test: pd.DataFrame,
    core_features: list[str],
    *,
    scope: str,
    fold: int,
) -> tuple[pd.DataFrame, list[dict]]:
    if train.empty or test.empty:
        raise RuntimeError(f"{scope}/{fold}: empty train or test")
    if train["date"].max() >= test["date"].min():
        raise RuntimeError(f"{scope}/{fold}: train/test chronology violated")
    if not (test.groupby("date").size() == UNIVERSE_TOP_N).all():
        raise RuntimeError(f"{scope}/{fold}: test is not exact Top100")
    result = _base_columns(test)
    result["scope"] = scope
    result["fold"] = int(fold)
    metadata = []
    variants = {
        CORE_POLICY: core_features,
        AUGMENTED_POLICY: core_features + SEMIVOL_FEATURES,
    }
    for policy, features in variants.items():
        raw, probability, meta = fp._predict_target(
            train,
            test,
            features,
            TARGET,
        )
        result[f"raw_{policy}"] = raw
        result[f"p_{policy}"] = probability
        meta.update(
            scope=scope,
            outer_fold=int(fold),
            policy=policy,
            feature_count=len(features),
            features=features,
        )
        _validate_model_metadata(meta, f"{scope}/{fold}/{policy}")
        metadata.append(meta)
    score_columns = [
        column
        for column in result
        if column.startswith(("raw_", "p_"))
    ]
    scores = result[score_columns].to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise RuntimeError(f"{scope}/{fold}: nonfinite model output")
    probability_columns = [
        column for column in score_columns if column.startswith("p_")
    ]
    if (
        (result[probability_columns] < 0).any().any()
        or (result[probability_columns] > 1).any().any()
    ):
        raise RuntimeError(f"{scope}/{fold}: probability outside [0,1]")
    return result, metadata


def run_discovery(
    panel: pd.DataFrame,
    discovery_dates: list,
    core_features: list[str],
    split_schedule_sha256: str,
) -> tuple[pd.DataFrame, list[dict]]:
    scoped = panel[panel["date"].isin(set(discovery_dates))]
    predictions = []
    metadata = []
    splits = fp._expanding_splits(
        discovery_dates,
        n_folds=OUTER_FOLDS,
        minimum_warmup=90,
    )
    if len(splits) != OUTER_FOLDS:
        raise RuntimeError(
            f"expected {OUTER_FOLDS} outer folds, got {len(splits)}"
        )
    for fold, (train_dates, test_dates) in enumerate(splits):
        train = scoped[scoped["date"].isin(set(train_dates))].copy()
        test = scoped[scoped["date"].isin(set(test_dates))].copy()
        result, fold_meta = _predict_both(
            train,
            test,
            core_features,
            scope="discovery_oof",
            fold=fold,
        )
        result["split_schedule_sha256"] = split_schedule_sha256
        for item in fold_meta:
            item["split_schedule_sha256"] = split_schedule_sha256
        predictions.append(result)
        metadata.extend(fold_meta)
    return pd.concat(predictions, ignore_index=True), metadata


def run_holdout(
    panel: pd.DataFrame,
    discovery_dates: list,
    holdout_dates: list,
    core_features: list[str],
    split_schedule_sha256: str,
) -> tuple[pd.DataFrame, list[dict]]:
    if len(discovery_dates) <= EMBARGO_DATES:
        raise RuntimeError("not enough discovery dates for holdout embargo")
    train_dates = discovery_dates[:-EMBARGO_DATES]
    train = panel[panel["date"].isin(set(train_dates))].copy()
    test = panel[panel["date"].isin(set(holdout_dates))].copy()
    result, metadata = _predict_both(
        train,
        test,
        core_features,
        scope="locked_holdout",
        fold=-1,
    )
    result["split_schedule_sha256"] = split_schedule_sha256
    for item in metadata:
        item.update(
            split_schedule_sha256=split_schedule_sha256,
            holdout_embargo_dates=EMBARGO_DATES,
            embargoed_discovery_dates=[
                str(value)
                for value in discovery_dates[-EMBARGO_DATES:]
            ],
        )
    return result, metadata


def attach_baselines(
    predictions: pd.DataFrame,
    baseline_path: Path,
) -> tuple[pd.DataFrame, dict]:
    aligned, metadata = fp.attach_reproducible_baselines(
        predictions,
        baseline_path,
    )
    aligned["score_lowest_ATR"] = -pd.to_numeric(
        aligned["f_atr_pct_14"]
    )
    return aligned, metadata


def make_picks(predictions: pd.DataFrame) -> pd.DataFrame:
    policy_scores = {
        AUGMENTED_POLICY: f"raw_{AUGMENTED_POLICY}",
        CORE_POLICY: f"raw_{CORE_POLICY}",
        "R1_repaired": "score_R1_repaired",
        "lowest_ATR": "score_lowest_ATR",
    }
    parts = [
        fp._top3(predictions, score, policy)
        for policy, score in policy_scores.items()
    ]
    parts.append(fp._liquidity_matched(predictions, parts[0]))
    picks = pd.concat(parts, ignore_index=True)
    expected_policies = len(policy_scores) + 1
    counts = picks.groupby(["scope", "date", "policy"]).size()
    if len(counts) == 0 or not (counts == TOP_K).all():
        raise RuntimeError("policy/date picks are not exact Top3")
    policy_counts = picks.groupby(["scope", "date"])[
        "policy"
    ].nunique()
    if not (policy_counts == expected_policies).all():
        raise RuntimeError("policy set is incomplete")
    return picks


def _auc_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope in sorted(predictions["scope"].unique()):
        scoped = predictions[
            (predictions["scope"] == scope)
            & predictions["path_complete"]
            & predictions[TARGET].notna()
        ]
        for policy in (CORE_POLICY, AUGMENTED_POLICY):
            raw_col = f"raw_{policy}"
            probability_col = f"p_{policy}"
            valid = scoped.dropna(
                subset=[TARGET, raw_col, probability_col]
            )
            rows.append(
                {
                    "scope": scope,
                    "policy": policy,
                    "n": int(len(valid)),
                    "dates": int(valid["date"].nunique()),
                    "base_rate": float(valid[TARGET].mean()),
                    "raw_auc": fp._safe_auc(
                        valid[TARGET], valid[raw_col]
                    ),
                    "probability_auc": fp._safe_auc(
                        valid[TARGET], valid[probability_col]
                    ),
                    "daily_macro_auc": fp._daily_macro_auc(
                        valid, TARGET, raw_col
                    ),
                    "within_vol_auc": fp._within_vol_auc(
                        valid, TARGET, raw_col
                    ),
                    "brier": float(
                        brier_score_loss(
                            valid[TARGET],
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


def _paired_metrics(
    common_picks: pd.DataFrame,
) -> pd.DataFrame:
    parts = []
    for scope in ("discovery_oof", "locked_holdout"):
        parts.append(
            fp.paired_bootstrap(
                common_picks,
                primary=AUGMENTED_POLICY,
                comparators=COMPARATORS,
                scope=scope,
                draws=BOOTSTRAP_DRAWS,
            )
        )
    return pd.concat(parts, ignore_index=True)


def _gate(paired: pd.DataFrame) -> dict:
    comparison = paired[
        (paired["scope"] == "locked_holdout")
        & (paired["primary"] == AUGMENTED_POLICY)
        & (paired["comparator"] == "R1_repaired")
    ].set_index("metric")
    downside = bool(
        comparison.loc["path_dn5_rate", "ci95_hi"] <= 0
    )
    safe = bool(comparison.loc["safe_fp_rate", "ci95_lo"] > 0)
    net = bool(comparison.loc["bracket_net", "ci95_lo"] >= 0)
    passed = downside and safe and net
    return {
        "comparison": f"{AUGMENTED_POLICY} versus R1_repaired",
        "rules": {
            "dn5_delta_ci95_upper_le_zero": {
                "passed": downside,
                "observed": float(
                    comparison.loc["path_dn5_rate", "ci95_hi"]
                ),
            },
            "safe_fp_delta_ci95_lower_gt_zero": {
                "passed": safe,
                "observed": float(
                    comparison.loc["safe_fp_rate", "ci95_lo"]
                ),
            },
            "net_delta_ci95_lower_ge_zero": {
                "passed": net,
                "observed": float(
                    comparison.loc["bracket_net", "ci95_lo"]
                ),
            },
        },
        "all_required": passed,
        "verdict": (
            "FORWARD_SHADOW_CANDIDATE" if passed else "REJECT"
        ),
        "maximum_possible_verdict": "FORWARD_SHADOW_CANDIDATE",
        "historical_holdout_contaminated": True,
    }


def _core_reference_audit(
    predictions: pd.DataFrame,
    reference_path: Path,
) -> dict:
    if not reference_path.exists():
        return {
            "available": False,
            "path": str(reference_path),
        }
    reference = pd.read_csv(
        reference_path,
        usecols=[
            "split_schedule_sha256",
            "scope",
            "fold",
            "date",
            "market",
            "raw_fp_fixed_head",
            "p_fp_fixed_head",
        ],
        float_precision="round_trip",
    )
    reference["date"] = pd.to_datetime(reference["date"]).dt.date
    identity = [
        "split_schedule_sha256",
        "scope",
        "fold",
        "date",
        "market",
    ]
    if reference.duplicated(identity).any():
        raise RuntimeError("core reference contains duplicate identity rows")
    reference_scores = reference[
        ["raw_fp_fixed_head", "p_fp_fixed_head"]
    ].apply(pd.to_numeric, errors="coerce")
    if reference_scores.isna().any().any() or not np.isfinite(
        reference_scores.to_numpy()
    ).all():
        raise RuntimeError("core reference contains nonfinite scores")
    reference[["raw_fp_fixed_head", "p_fp_fixed_head"]] = (
        reference_scores
    )
    candidate = predictions[
        [
            "split_schedule_sha256",
            "scope",
            "fold",
            "date",
            "market",
            f"raw_{CORE_POLICY}",
            f"p_{CORE_POLICY}",
        ]
    ]
    merged = candidate.merge(
        reference,
        on=identity,
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(candidate):
        raise RuntimeError(
            "core reference does not cover all aligned predictions: "
            f"candidate={len(candidate)} reference_common={len(merged)}"
        )
    raw_delta = (
        merged[f"raw_{CORE_POLICY}"]
        - merged["raw_fp_fixed_head"]
    ).abs()
    probability_delta = (
        merged[f"p_{CORE_POLICY}"]
        - merged["p_fp_fixed_head"]
    ).abs()
    max_raw = float(raw_delta.max())
    max_probability = float(probability_delta.max())
    if (
        not np.isfinite(max_raw)
        or not np.isfinite(max_probability)
        or max_raw > 1e-15
        or max_probability > 1e-15
    ):
        raise RuntimeError(
            "24-feature core does not reproduce first-passage reference: "
            f"raw={max_raw} probability={max_probability}"
        )
    return {
        "available": True,
        "path": str(reference_path.relative_to(ROOT)),
        "sha256": _sha256(reference_path),
        "rows_compared": int(len(merged)),
        "raw_max_abs_delta": max_raw,
        "probability_max_abs_delta": max_probability,
        "exact_within_1e_15": True,
    }


def _artifacts(prefix: Path) -> dict[str, Path]:
    return {
        "predictions": Path(f"{prefix}_predictions.csv.gz"),
        "picks": Path(f"{prefix}_picks.csv.gz"),
        "summary": Path(f"{prefix}_summary.csv"),
        "paired": Path(f"{prefix}_paired.csv"),
        "auc": Path(f"{prefix}_auc.csv"),
        "folds": Path(f"{prefix}_folds.csv"),
        "coverage": Path(f"{prefix}_coverage.json"),
        "manifest": Path(f"{prefix}_manifest.json"),
    }


def _validate_dependency_generations(
    *,
    d1_db: Path,
    m15_db: Path,
    baseline_predictions: Path,
    core_reference: Path,
) -> dict:
    safeup_audit = fp._validate_safeup_baseline_lineage(
        baseline_predictions=baseline_predictions,
        d1_db=d1_db,
        m15_db=m15_db,
    )
    first_passage_audit = None
    if core_reference.is_file():
        first_passage_audit = fp.validate_existing_artifacts(
            output_prefix=fp._output_prefix_from_predictions(
                core_reference
            ),
            d1_db=d1_db,
            m15_db=m15_db,
            baseline_predictions=baseline_predictions,
        )
    return {
        "safeup": safeup_audit,
        "first_passage": first_passage_audit,
    }


def validate_existing_artifacts(
    *,
    output_prefix: Path,
    d1_db: Path,
    m15_db: Path,
    path_panel: Path,
    path_panel_meta: Path,
    baseline_predictions: Path,
    core_reference: Path,
) -> dict:
    """Reject stale/tampered semivol outputs before report consumption."""
    artifacts = _artifacts(output_prefix)
    expected = {
        name: path
        for name, path in artifacts.items()
        if name != "manifest"
    }
    fp._verify_manifest(
        manifest_path=artifacts["manifest"],
        schema="semivol_joint_challenger_v1_manifest",
        expected=expected,
    )
    try:
        coverage = json.loads(
            artifacts["coverage"].read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("semivol coverage is unreadable") from exc
    if coverage.get("schema") != "semivol_joint_challenger_v1":
        raise RuntimeError("semivol coverage schema mismatch")
    inputs = coverage.get("inputs")
    if not isinstance(inputs, dict):
        raise RuntimeError("semivol input lineage is missing")
    if inputs.get("script_sha256") != _sha256(Path(__file__)):
        raise RuntimeError("semivol artifacts were built by stale code")
    if inputs.get("code_lineage") != _code_lineage():
        raise RuntimeError("semivol local code dependencies have changed")
    expected_hashes = {
        "d1_db_sha256": _sha256(d1_db),
        "path_panel_sha256": _sha256(path_panel),
        "path_panel_meta_sha256": _sha256(path_panel_meta),
        "baseline_predictions_sha256": _sha256(
            baseline_predictions
        ),
    }
    mismatched = {
        key: {"recorded": inputs.get(key), "current": value}
        for key, value in expected_hashes.items()
        if inputs.get(key) != value
    }
    if mismatched:
        raise RuntimeError(f"semivol input checksum mismatch: {mismatched}")
    current_dependency_audit = _validate_dependency_generations(
        d1_db=d1_db,
        m15_db=m15_db,
        baseline_predictions=baseline_predictions,
        core_reference=core_reference,
    )
    if (
        inputs.get("dependency_generation_audit")
        != current_dependency_audit
    ):
        raise RuntimeError("semivol dependency generation has changed")
    _validate_path_panel_lineage(
        panel_path=path_panel,
        meta_path=path_panel_meta,
        d1_db=d1_db,
    )
    panel = _load_path_panel(path_panel)
    benchmark_dates = sorted(
        panel.loc[panel["benchmark_complete"], "date"].unique()
    )
    current_schedule = safeup.build_common_benchmark_schedule(
        benchmark_dates
    )[0]
    if coverage.get("shared_split_schedule") != current_schedule:
        raise RuntimeError("semivol shared schedule is stale")
    if (
        current_schedule["split_schedule_sha256"]
        != current_dependency_audit["safeup"][
            "split_schedule_sha256"
        ]
    ):
        raise RuntimeError("semivol dependency schedule mismatch")
    prediction_identity = pd.read_csv(
        artifacts["predictions"],
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
        raise RuntimeError("semivol predictions have duplicate identities")
    if not prediction_identity["market"].astype("string").str.fullmatch(
        r"KRW-[A-Z0-9]+", na=False
    ).all():
        raise RuntimeError("semivol predictions contain invalid markets")
    if set(
        prediction_identity["split_schedule_sha256"].astype(str)
    ) != {current_schedule["split_schedule_sha256"]}:
        raise RuntimeError("semivol prediction schedule hash mismatch")
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
            "semivol predictions do not exactly cover the schedule"
        )
    reference_audit = coverage.get("core_reference_reproduction", {})
    if core_reference.is_file():
        if (
            not isinstance(reference_audit, dict)
            or not reference_audit.get("available")
            or reference_audit.get("sha256")
            != _sha256(core_reference)
        ):
            raise RuntimeError("semivol core reference lineage mismatch")
    return {
        "status": "valid",
        "schema": coverage["schema"],
        "output_prefix": str(output_prefix),
        "verdict": coverage.get("verdict"),
        "manifest_sha256": _sha256(artifacts["manifest"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "One fixed core24 versus core24+4 semivolatility "
            "first-passage diagnostic"
        )
    )
    parser.add_argument("--d1-db", type=Path, default=D1_DB)
    parser.add_argument("--m15-db", type=Path, default=fp.M15_DB)
    parser.add_argument(
        "--path-panel", type=Path, default=PATH_PANEL
    )
    parser.add_argument(
        "--path-panel-meta", type=Path, default=PATH_PANEL_META
    )
    parser.add_argument(
        "--baseline-predictions",
        type=Path,
        default=BASELINE_PREDICTIONS,
    )
    parser.add_argument(
        "--core-reference",
        type=Path,
        default=CORE_REFERENCE,
    )
    parser.add_argument(
        "--output-prefix", type=Path, default=OUT_PREFIX
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
            path_panel=args.path_panel,
            path_panel_meta=args.path_panel_meta,
            baseline_predictions=args.baseline_predictions,
            core_reference=args.core_reference,
        )
        print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
        return
    required_paths = (
        args.d1_db,
        args.m15_db,
        args.path_panel,
        args.path_panel_meta,
        args.baseline_predictions,
    )
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"required inputs missing: {missing}")
    provenance_paths = list(required_paths)
    if args.core_reference.exists():
        provenance_paths.append(args.core_reference)
    dependency_generation_audit = _validate_dependency_generations(
        d1_db=args.d1_db,
        m15_db=args.m15_db,
        baseline_predictions=args.baseline_predictions,
        core_reference=args.core_reference,
    )
    source_signatures = {
        str(path): fp._file_signature(path)
        for path in provenance_paths
    }
    code_lineage = _code_lineage()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    path_panel_lineage = _validate_path_panel_lineage(
        panel_path=args.path_panel,
        meta_path=args.path_panel_meta,
        d1_db=args.d1_db,
    )
    panel = _load_path_panel(args.path_panel)
    cached_path_meta = path_panel_lineage["path_meta"]
    observed_counts = {
        "path_rows": int(len(panel)),
        "path_dates": int(panel["date"].nunique()),
        "target_complete_rows": int(panel["path_complete"].sum()),
    }
    mismatched_counts = {
        key: {
            "metadata": cached_path_meta.get(key),
            "observed": value,
        }
        for key, value in observed_counts.items()
        if cached_path_meta.get(key) != value
    }
    if mismatched_counts:
        raise RuntimeError(
            f"path-panel metadata counts mismatch: {mismatched_counts}"
        )
    panel, feature_audit = attach_semivol_features(panel, args.d1_db)
    core_features = safeup._feature_columns(panel)
    if len(core_features) != 24:
        raise RuntimeError(
            f"core feature contract is not 24: {len(core_features)}"
        )
    augmented_features = core_features + SEMIVOL_FEATURES
    if len(augmented_features) != 28:
        raise RuntimeError("augmented feature contract is not 28")

    benchmark_dates = sorted(
        panel.loc[panel["benchmark_complete"], "date"].unique()
    )
    if len(benchmark_dates) <= fp.LOCKED_COMMON_DATES + 60:
        raise RuntimeError("insufficient benchmark-complete dates")
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
    safeup_schedule_audit = dependency_generation_audit["safeup"]
    if (
        split_schedule_sha256
        != safeup_schedule_audit["split_schedule_sha256"]
        or shared_schedule["eligible_dates_sha256"]
        != safeup_schedule_audit["eligible_dates_sha256"]
        or shared_schedule["locked_holdout_dates_sha256"]
        != safeup_schedule_audit[
            "locked_holdout_dates_sha256"
        ]
    ):
        raise RuntimeError(
            "semivol/path/SafeUp shared schedules differ"
        )
    discovery_dates = list(
        np.asarray(benchmark_dates, dtype=object)[
            :-fp.LOCKED_COMMON_DATES
        ]
    )
    holdout_dates = list(holdout_date_array)

    discovery, discovery_meta = run_discovery(
        panel,
        discovery_dates,
        core_features,
        split_schedule_sha256,
    )
    holdout, holdout_meta = run_holdout(
        panel,
        discovery_dates,
        holdout_dates,
        core_features,
        split_schedule_sha256,
    )
    predictions = pd.concat(
        [discovery, holdout], ignore_index=True, sort=False
    )
    predictions, baseline_meta = attach_baselines(
        predictions, args.baseline_predictions
    )
    history = pd.to_numeric(
        predictions["history_prior_bars"], errors="coerce"
    )
    if history.isna().any() or int(history.min()) < MIN_PRIOR_HISTORY:
        raise RuntimeError("aligned predictions violate history >= 70")
    counts = predictions.groupby(["scope", "date"]).size()
    if not (counts == UNIVERSE_TOP_N).all():
        raise RuntimeError("aligned predictions are not exact Top100")

    reference_audit = _core_reference_audit(
        predictions, args.core_reference
    )
    picks_all = make_picks(predictions)
    common_picks, common_meta = fp._common_complete(picks_all)
    expected_policies = len(COMPARATORS) + 1
    common_policy_counts = common_picks.groupby(["scope", "date"])[
        "policy"
    ].nunique()
    if not (common_policy_counts == expected_policies).all():
        raise RuntimeError("common-date policy intersection is incomplete")

    summary = fp.policy_metrics(predictions, common_picks)
    paired = _paired_metrics(common_picks)
    auc = _auc_metrics(predictions)
    folds = fp.fold_metrics(common_picks)
    gate = _gate(paired)
    source_signatures_after = {
        str(path): fp._file_signature(path)
        for path in provenance_paths
    }
    if source_signatures_after != source_signatures:
        raise RuntimeError("research input changed while challenger was running")
    if _code_lineage() != code_lineage:
        raise RuntimeError("local code changed while challenger was running")
    if (
        _validate_dependency_generations(
            d1_db=args.d1_db,
            m15_db=args.m15_db,
            baseline_predictions=args.baseline_predictions,
            core_reference=args.core_reference,
        )
        != dependency_generation_audit
    ):
        raise RuntimeError(
            "dependency generation changed while semivol ran"
        )

    predictions = predictions.sort_values(
        ["scope", "fold", "date", "market"]
    ).reset_index(drop=True)
    common_picks = common_picks.sort_values(
        ["scope", "fold", "date", "policy", "selection_rank"]
    ).reset_index(drop=True)
    summary = summary.sort_values(["scope", "policy"]).reset_index(
        drop=True
    )
    paired = paired.sort_values(
        ["scope", "comparator", "metric"]
    ).reset_index(drop=True)
    auc = auc.sort_values(["scope", "policy"]).reset_index(drop=True)
    folds = folds.sort_values(
        ["scope", "fold", "policy"]
    ).reset_index(drop=True)

    artifacts = _artifacts(args.output_prefix)
    coverage = {
        "schema": "semivol_joint_challenger_v1",
        "research_only": True,
        "production_modified": False,
        "auto_order_code": False,
        "authorization_boundary": (
            "user-authorized bounded historical diagnostic; the final "
            "180-date slice is already contaminated by related research"
        ),
        "fixed_trial": {
            "trial_count": 1,
            "target": (
                "[D09:15,D+1 09:15) +10% before -5%; "
                "same-bar downside first"
            ),
            "core": core_features,
            "augmented": augmented_features,
            "new_features": SEMIVOL_FEATURES,
            "windows_or_formulas_tuned": False,
            "model_or_hyperparameters_tuned": False,
        },
        "hygiene": {
            "minimum_prior_history": MIN_PRIOR_HISTORY,
            "observed_minimum_prior_history": int(history.min()),
            "universe": "strict PIT D-1 quote-volume Top100",
            "candidate_counts_all_100": True,
            "feature_boundary": "<= D-1",
            "feature_source_date_violations": feature_audit[
                "feature_source_date_violations"
            ],
            "outer": (
                f"{OUTER_FOLDS}-fold expanding WF, "
                f"{EMBARGO_DATES}-date embargo"
            ),
            "inner": (
                f"{INNER_FOLDS}-fold true expanding OOF isotonic, "
                f"{EMBARGO_DATES}-date embargo"
            ),
            "model": (
                "XGBClassifier n_estimators=180,max_depth=4,lr=.05,"
                "subsample=.8,colsample_bytree=.8,min_child_weight=5,"
                "reg_lambda=1.5,seed=42,n_jobs=1,tree_method=hist"
            ),
            "path": (
                "reused exact [D09:15,D+1 09:15) 96-bar path panel; "
                "same-bar downside first"
            ),
            "round_trip_cost_once": ROUND_TRIP_COST,
        },
        "feature_audit": feature_audit,
        "inputs": {
            "d1_db": str(args.d1_db.relative_to(ROOT)),
            "d1_db_sha256": _sha256(args.d1_db),
            "path_panel": str(args.path_panel.relative_to(ROOT)),
            "path_panel_sha256": _sha256(args.path_panel),
            "path_panel_meta": str(
                args.path_panel_meta.relative_to(ROOT)
            ),
            "path_panel_meta_sha256": _sha256(
                args.path_panel_meta
            ),
            "baseline_predictions": str(
                args.baseline_predictions.relative_to(ROOT)
            ),
            "baseline_predictions_sha256": _sha256(
                args.baseline_predictions
            ),
            "script_sha256": _sha256(Path(__file__)),
            "code_lineage": code_lineage,
            "dependency_generation_audit": dependency_generation_audit,
            "path_panel_cache_schema": path_panel_lineage[
                "signature"
            ]["schema"],
            "path_panel_cache_sha256_verified": True,
            "path_panel_source_lineage_verified": True,
            "sources_before_and_after_identical": True,
            "source_signatures": source_signatures,
        },
        "core_reference_reproduction": reference_audit,
        "dates": {
            "benchmark_complete_total": len(benchmark_dates),
            "discovery": _date_coverage(
                panel[panel["date"].isin(set(discovery_dates))]
            ),
            "locked_holdout": {
                **_date_coverage(
                    panel[panel["date"].isin(set(holdout_dates))]
                ),
                "definition": (
                    "last 180 dates on the exact shared benchmark-complete "
                    "eligibility axis"
                ),
                "contaminated_diagnostic_only": True,
            },
            "aligned_predictions": {
                scope: _date_coverage(
                    predictions[predictions["scope"] == scope]
                )
                for scope in sorted(predictions["scope"].unique())
            },
            "common_complete": {
                scope: _date_coverage(
                    common_picks[common_picks["scope"] == scope]
                )
                for scope in sorted(common_picks["scope"].unique())
            },
        },
        "shared_split_schedule": shared_schedule,
        "common_complete_date_counts": common_meta,
        "model_metadata": {
            "discovery": discovery_meta,
            "locked_holdout": holdout_meta,
        },
        "baseline_alignment": baseline_meta,
        "comparators": list(COMPARATORS),
        "gate": gate,
        "verdict": gate["verdict"],
        "maximum_possible_verdict": "FORWARD_SHADOW_CANDIDATE",
    }
    result_names = (
        "predictions",
        "picks",
        "summary",
        "paired",
        "auc",
        "folds",
        "coverage",
    )
    with tempfile.TemporaryDirectory(
        dir=args.output_prefix.parent,
        prefix=f".{args.output_prefix.name}.generation.",
    ) as stage_directory:
        stage_root = Path(stage_directory)
        staged = {
            name: stage_root / artifacts[name].name
            for name in result_names
        }
        predictions.to_csv(
            staged["predictions"],
            index=False,
            compression=GZIP_COMPRESSION,
            float_format="%.17g",
        )
        common_picks.to_csv(
            staged["picks"],
            index=False,
            compression=GZIP_COMPRESSION,
            float_format="%.17g",
        )
        summary.to_csv(
            staged["summary"], index=False, float_format="%.17g"
        )
        paired.to_csv(
            staged["paired"], index=False, float_format="%.17g"
        )
        auc.to_csv(
            staged["auc"], index=False, float_format="%.17g"
        )
        folds.to_csv(
            staged["folds"], index=False, float_format="%.17g"
        )
        _write_json(staged["coverage"], coverage)
        staged_manifest = stage_root / artifacts["manifest"].name
        _write_json(
            staged_manifest,
            {
                "schema": "semivol_joint_challenger_v1_manifest",
                "gzip_mtime": 0,
                "files": {
                    name: {
                        "path": str(artifacts[name].relative_to(ROOT)),
                        "sha256": _sha256(path),
                        "bytes": int(path.stat().st_size),
                    }
                    for name, path in staged.items()
                },
            },
        )
        publish_map = {
            staged[name]: artifacts[name] for name in result_names
        }
        publish_map[staged_manifest] = artifacts["manifest"]
        fp._publish_staged_files(publish_map)

    holdout_summary = summary[
        (summary["scope"] == "locked_holdout")
        & summary["policy"].isin(
            [AUGMENTED_POLICY, CORE_POLICY, "R1_repaired"]
        )
    ]
    print(holdout_summary.to_string(index=False))
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    print(f"verdict={gate['verdict']}")


if __name__ == "__main__":
    main()

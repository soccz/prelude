"""Read-only independent recalculation for first-passage challenger v1.

The script consumes persisted row artifacts, rebuilds every reported aggregate,
uses an independent 20,000-draw date bootstrap, and diagnoses volatility
confounding.  It writes nothing.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
PREFIX = ROOT / "output/first_passage_head_challenger_v1"
FIXED = "label_fp_safe10"
ATR = "label_fp_atr"
SEED = 20260725
DRAWS = 20_000


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
    arr = numeric.to_numpy(float)
    if not len(arr):
        raise ValueError("bootstrap input is empty")
    rng = np.random.default_rng(SEED + offset)
    index = rng.integers(0, len(arr), size=(DRAWS, len(arr)))
    boot = arr[index].mean(axis=1)
    return {
        "n": int(len(arr)),
        "mean": float(arr.mean()),
        "ci95": [
            float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)),
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
    duplicates = frame.duplicated(columns, keep=False)
    if duplicates.any():
        sample = frame.loc[duplicates, columns].head(3).to_dict("records")
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


def _time_order_audit(
    coverage: dict,
    predictions: pd.DataFrame,
) -> dict:
    metadata = coverage.get("model_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("coverage: model_metadata must be an object")
    discovery = metadata.get("discovery")
    holdout = metadata.get("holdout")
    if not isinstance(discovery, list) or not isinstance(holdout, dict):
        raise ValueError("coverage: malformed discovery/holdout metadata")
    entries = [*discovery, holdout]
    seen: set[tuple[str, int, str]] = set()
    discovery_windows: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for entry in entries:
        scope = str(entry["scope"])
        fold = int(entry["outer_fold"])
        target = str(entry["target"])
        key = (scope, fold, target)
        if key in seen:
            raise ValueError(f"coverage: duplicate model metadata key: {key}")
        seen.add(key)
        train_end = pd.Timestamp(entry["train_end"])
        test_start = pd.Timestamp(entry["test_start"])
        test_end = pd.Timestamp(entry["test_end"])
        if train_end >= test_start or test_start > test_end:
            raise ValueError(f"coverage: train/test overlap or inverted range: {key}")
        if (test_start - train_end).days - 1 < 5:
            raise ValueError(f"coverage: outer embargo shorter than 5 dates: {key}")
        inner = entry.get("inner_folds")
        if not isinstance(inner, list) or len(inner) != 3:
            raise ValueError(f"coverage: expected three inner folds: {key}")
        previous_validation_end: pd.Timestamp | None = None
        for inner_entry in inner:
            inner_train_end = pd.Timestamp(inner_entry["train_end"])
            validation_start = pd.Timestamp(inner_entry["validation_start"])
            validation_end = pd.Timestamp(inner_entry["validation_end"])
            if inner_train_end >= validation_start or validation_start > validation_end:
                raise ValueError(f"coverage: inner train/validation overlap: {key}")
            if (validation_start - inner_train_end).days - 1 < 5:
                raise ValueError(f"coverage: inner embargo shorter than 5 dates: {key}")
            if (
                previous_validation_end is not None
                and validation_start <= previous_validation_end
            ):
                raise ValueError(f"coverage: overlapping inner validation folds: {key}")
            previous_validation_end = validation_end
        scoped_dates = predictions.loc[
            predictions["scope"].eq(scope) & predictions["fold"].eq(fold),
            "date",
        ]
        if scoped_dates.empty:
            raise ValueError(f"predictions: no rows for metadata key: {key}")
        if scoped_dates.min() != test_start or scoped_dates.max() != test_end:
            raise ValueError(f"predictions: test window disagrees with metadata: {key}")
        if scope == "discovery_oof":
            discovery_windows.setdefault(target, []).append((test_start, test_end))
    for target, windows in discovery_windows.items():
        ordered = sorted(windows)
        if len(ordered) != 5:
            raise ValueError(f"coverage: {target} does not have five outer folds")
        if any(current[0] <= previous[1] for previous, current in zip(ordered, ordered[1:])):
            raise ValueError(f"coverage: overlapping outer test folds for {target}")
    embargoed = holdout.get("embargoed_discovery_dates")
    if int(holdout.get("holdout_embargo_dates", -1)) != 5:
        raise ValueError("coverage: locked holdout embargo count is not five")
    if not isinstance(embargoed, list) or len(embargoed) != 5:
        raise ValueError("coverage: locked holdout embargo date list is not five")
    embargo_dates = [pd.Timestamp(value) for value in embargoed]
    expected = list(
        pd.date_range(
            pd.Timestamp(holdout["train_end"]) + pd.Timedelta(days=1),
            periods=5,
            freq="D",
        )
    )
    if embargo_dates != expected:
        raise ValueError("coverage: locked holdout embargo dates are not contiguous")
    return {
        "metadata_entries": len(entries),
        "unique_entries": len(seen),
        "outer_test_overlap_violations": 0,
        "outer_embargo_violations": 0,
        "inner_overlap_or_embargo_violations": 0,
        "locked_holdout_embargo_dates": 5,
    }


def _daily(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["_sl"] = work["path_bracket_outcome"].eq("sl").astype(float)
    work["_tp"] = work["path_bracket_outcome"].eq("tp").astype(float)
    return work.groupby("date", sort=True).agg(
        safe=(FIXED, "mean"),
        atr_label=(ATR, "mean"),
        dn5=("path_dn5", "mean"),
        up10=("path_up10", "mean"),
        sl=("_sl", "mean"),
        tp=("_tp", "mean"),
        net=("path_bracket_net", "mean"),
        n=("market", "size"),
    )


def _table(frame: pd.DataFrame) -> dict:
    output = {}
    for (scope, policy), group in frame.groupby(["scope", "policy"], sort=True):
        daily = _daily(group)
        output[f"{scope}:{policy}"] = {
            "rows": int(len(group)),
            "dates": int(group["date"].nunique()),
            **{
                metric: float(daily[metric].mean())
                for metric in ("safe", "atr_label", "dn5", "up10", "sl", "tp", "net")
            },
            "absolute_net_date_ci": _ci(daily["net"], offset=len(output) + 1),
        }
    return output


def _paired(
    frame: pd.DataFrame,
    *,
    scope: str,
    primary: str,
    comparators: list[str],
    offset: int,
) -> dict:
    scoped = frame[frame["scope"].eq(scope)]
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
                common[f"{metric}_primary"] - common[f"{metric}_baseline"],
                offset=offset + comparator_index * 20 + metric_index,
            )
            for metric_index, metric in enumerate(
                ("safe", "atr_label", "dn5", "up10", "sl", "net")
            )
        }
    return output


def _lowest_atr_diagnostic(
    predictions: pd.DataFrame,
    persisted_picks: pd.DataFrame,
) -> dict:
    """Compare a diagnostic-only lowest-ATR Top3 on a strict common cohort."""
    locked = predictions[
        predictions["scope"].eq("locked_holdout")
    ].copy()
    low_atr = (
        locked.sort_values(
            ["date", "f_atr_pct_14", "market"],
            ascending=[True, True, True],
        )
        .groupby("date", sort=False)
        .head(3)
        .copy()
    )
    low_atr["policy"] = "lowest_ATR_top3_diagnostic"
    low_atr["selection_rank"] = (
        low_atr.groupby("date", sort=False).cumcount() + 1
    )
    low_counts = low_atr.groupby("date").size()
    if len(low_counts) == 0 or not (low_counts == 3).all():
        raise RuntimeError("lowest-ATR diagnostic is not exact Top3")

    existing = persisted_picks[
        persisted_picks["scope"].eq("locked_holdout")
    ].copy()
    existing_dates = set(existing["date"].unique())
    low_complete = (
        low_atr.groupby("date")["path_complete"].all()
    )
    common_dates = existing_dates.intersection(
        set(low_complete[low_complete].index)
    )
    existing_common = existing[
        existing["date"].isin(common_dates)
    ].copy()
    low_common = low_atr[
        low_atr["date"].isin(common_dates)
    ].copy()
    combined = pd.concat(
        [existing_common, low_common],
        ignore_index=True,
        sort=False,
    )
    policy_counts = combined.groupby(["date", "policy"]).size()
    if not (policy_counts == 3).all():
        raise RuntimeError("low-ATR common cohort is not exact Top3")
    if not combined["path_complete"].all():
        raise RuntimeError("low-ATR common cohort contains incomplete paths")

    return {
        "diagnostic_only_not_candidate_or_trial": True,
        "selection": "lowest D-1 ATR14 Top3, market ascending tie-break",
        "common_dates": int(len(common_dates)),
        "rows_per_policy": int(3 * len(common_dates)),
        "policies": sorted(combined["policy"].unique()),
        "table": _table(combined),
        "paired_low_atr_minus": _paired(
            combined,
            scope="locked_holdout",
            primary="lowest_ATR_top3_diagnostic",
            comparators=["R1_repaired", "fp_fixed_head"],
            offset=900,
        ),
    }


def _auc(y: pd.Series, score: pd.Series) -> float | None:
    valid = y.notna() & score.notna()
    if valid.sum() < 2 or y[valid].nunique() < 2:
        return None
    return float(roc_auc_score(y[valid], score[valid]))


def _date_band_auc(
    frame: pd.DataFrame,
    *,
    label: str,
    score: str,
) -> dict:
    values = []
    weights = []
    for _, group in frame.groupby(["date", "vol_band"], sort=True):
        auc = _auc(group[label], group[score])
        if auc is not None:
            values.append(auc)
            weights.append(len(group))
    return {
        "groups": int(len(values)),
        "macro": float(np.mean(values)) if values else None,
        "row_weighted": (
            float(np.average(values, weights=weights)) if values else None
        ),
    }


def _volatility_audit(
    path_panel: pd.DataFrame,
    predictions: pd.DataFrame,
    picks: pd.DataFrame,
    holdout_dates: set[pd.Timestamp],
) -> dict:
    label_rates = {}
    for name, cohort in {
        "discovery": path_panel[~path_panel["date"].isin(holdout_dates)],
        "locked_holdout": path_panel[path_panel["date"].isin(holdout_dates)],
    }.items():
        complete = cohort[cohort["path_complete"]].copy()
        label_rates[name] = {
            str(int(band)): {
                "n": int(len(group)),
                "fixed_safe10": float(group[FIXED].mean()),
                "atr_first_passage": float(group[ATR].mean()),
                "dn5": float(group["path_dn5"].mean()),
                "up10": float(group["path_up10"].mean()),
            }
            for band, group in complete.groupby("vol_band", sort=True)
        }

    score_audits = {}
    for scope, scoped in predictions.groupby("scope", sort=True):
        complete = scoped[scoped["path_complete"]].copy()
        for head, own_label in (
            ("fp_fixed_head", FIXED),
            ("fp_atr_head", ATR),
        ):
            score = f"raw_{head}"
            if score not in complete or not complete[score].notna().any():
                continue
            score_audits[f"{scope}:{head}"] = {
                "score_atr_spearman": float(
                    complete[[score, "f_atr_pct_14"]]
                    .corr(method="spearman")
                    .iloc[0, 1]
                ),
                "pooled_fixed_auc": _auc(complete[FIXED], complete[score]),
                "date_band_fixed_auc": _date_band_auc(
                    complete, label=FIXED, score=score
                ),
                "pooled_own_auc": _auc(complete[own_label], complete[score]),
                "date_band_own_auc": _date_band_auc(
                    complete, label=own_label, score=score
                ),
            }

    pick_bands = {
        f"{scope}:{policy}": {
            str(int(band)): float(value)
            for band, value in (
                group["vol_band"].value_counts(normalize=True).sort_index().items()
            )
        }
        for (scope, policy), group in picks.groupby(["scope", "policy"], sort=True)
    }

    matched = {}
    for (scope, policy), group in picks.groupby(["scope", "policy"], sort=True):
        reference = predictions[
            predictions["scope"].eq(scope)
            & predictions["date"].isin(set(group["date"]))
            & predictions["path_complete"]
        ].copy()
        reference["_sl"] = reference["path_bracket_outcome"].eq("sl").astype(float)
        strata = reference.groupby(["date", "vol_band"], sort=True).agg(
            matched_safe=(FIXED, "mean"),
            matched_dn5=("path_dn5", "mean"),
            matched_up10=("path_up10", "mean"),
            matched_sl=("_sl", "mean"),
            matched_net=("path_bracket_net", "mean"),
        )
        joined = group.merge(
            strata.reset_index(),
            on=["date", "vol_band"],
            how="left",
            validate="many_to_one",
        )
        actual = _daily(group)
        matched[f"{scope}:{policy}"] = {
            "actual": {
                metric: float(actual[metric].mean())
                for metric in ("safe", "dn5", "up10", "sl", "net")
            },
            "same_day_vol_band_expectation": {
                "safe": float(joined["matched_safe"].mean()),
                "dn5": float(joined["matched_dn5"].mean()),
                "up10": float(joined["matched_up10"].mean()),
                "sl": float(joined["matched_sl"].mean()),
                "net": float(joined["matched_net"].mean()),
            },
        }
    expected_score_audits = {
        "discovery_oof:fp_fixed_head",
        "discovery_oof:fp_atr_head",
        "locked_holdout:fp_fixed_head",
    }
    if set(score_audits) != expected_score_audits:
        raise ValueError("volatility audit: expected score populations are missing")
    for key, audit in score_audits.items():
        for metric in ("score_atr_spearman", "pooled_fixed_auc", "pooled_own_auc"):
            value = audit[metric]
            if value is None or not np.isfinite(value):
                raise ValueError(f"volatility audit: invalid {metric} for {key}")
    return {
        "label_rates_by_vol_band": label_rates,
        "score_discrimination": score_audits,
        "pick_vol_band_fraction": pick_bands,
        "same_day_vol_band_matching": matched,
    }


def main() -> None:
    paths = {
        "path_panel": Path(f"{PREFIX}_path_panel.csv.gz"),
        "predictions": Path(f"{PREFIX}_predictions.csv.gz"),
        "picks": Path(f"{PREFIX}_picks.csv.gz"),
        "summary": Path(f"{PREFIX}_summary.csv"),
        "paired": Path(f"{PREFIX}_paired.csv"),
        "coverage": Path(f"{PREFIX}_coverage.json"),
    }
    missing_paths = [str(path) for path in paths.values() if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"missing audit inputs: {missing_paths}")
    panel = pd.read_csv(paths["path_panel"], parse_dates=["date"])
    predictions = pd.read_csv(paths["predictions"], parse_dates=["date"])
    picks = pd.read_csv(paths["picks"], parse_dates=["date"])
    _require_columns(
        panel,
        {
            "market", "date", "path_complete", "benchmark_complete",
            "history_prior_bars", FIXED, ATR, "fp_fixed_outcome",
            "fp_atr_outcome", "path_dn5", "path_up10",
            "path_bracket_outcome", "path_bracket_net", "path_eod_net",
        },
        name="path_panel",
    )
    _require_columns(
        predictions,
        {
            "market", "date", "scope", "fold", "path_complete",
            "history_prior_bars", FIXED, ATR, "path_bracket_net",
            "path_eod_net", "path_bracket_outcome", "path_dn5", "path_up10",
        },
        name="predictions",
    )
    _require_columns(
        picks,
        {
            "market", "date", "scope", "fold", "policy", "path_complete",
            FIXED, ATR, "path_bracket_net", "path_eod_net",
            "path_bracket_outcome", "path_dn5", "path_up10",
        },
        name="picks",
    )
    for name, frame in (
        ("path_panel", panel),
        ("predictions", predictions),
        ("picks", picks),
    ):
        if frame.empty:
            raise ValueError(f"{name}: empty artifact")
        if frame["date"].isna().any():
            raise ValueError(f"{name}: invalid/null dates")
    _require_unique(panel, ["market", "date"], name="path_panel")
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
    for frame in (panel, predictions, picks):
        frame["path_complete"] = _bool(frame["path_complete"])
    panel["benchmark_complete"] = _bool(panel["benchmark_complete"])

    coverage = json.loads(paths["coverage"].read_text())
    if coverage.get("schema") != "first_passage_head_challenger_v1":
        raise ValueError("coverage: unexpected schema")
    completed_cutoff = pd.Timestamp(
        coverage["path_panel"]["panel"]["completed_label_cutoff"]
    )
    if completed_cutoff > _current_completed_cutoff():
        raise ValueError("coverage: completed-label cutoff is in the future")
    for name, frame in (
        ("path_panel", panel),
        ("predictions", predictions),
        ("picks", picks),
    ):
        if frame["date"].max() > completed_cutoff:
            raise ValueError(f"{name}: contains rows after completed-label cutoff")
    if panel["date"].max() != completed_cutoff:
        raise ValueError("path_panel: max date disagrees with completed-label cutoff")
    if len(panel) != int(coverage["path_panel"]["path_rows"]):
        raise ValueError("path_panel: row count disagrees with coverage")
    expected_scopes = {"discovery_oof", "locked_holdout"}
    if set(predictions["scope"]) != expected_scopes:
        raise ValueError("predictions: unexpected/missing scopes")
    expected_policies = {
        "discovery_oof": {"R1_repaired", "fp_atr_head", "fp_fixed_head"},
        "locked_holdout": {
            "ATR_top3", "R1_repaired", "fp_fixed_head",
            "liquidity_matched", "monkey_seed42", "safeup_head",
        },
    }
    for scope, policies in expected_policies.items():
        scoped = picks[picks["scope"].eq(scope)]
        if set(scoped["policy"]) != policies:
            raise ValueError(f"picks: unexpected/missing policies in {scope}")
        counts = scoped.groupby(["date", "policy"]).size()
        if counts.empty or not counts.eq(3).all():
            raise ValueError(f"picks: {scope} is not an exact common Top3 cohort")
        policy_counts = scoped.groupby("date")["policy"].nunique()
        if not policy_counts.eq(len(policies)).all():
            raise ValueError(f"picks: {scope} policy cohort is incomplete")
    if not picks["path_complete"].all():
        raise ValueError("picks: persisted cohort contains incomplete paths")

    complete_panel = panel[panel["path_complete"]]
    _require_finite(
        complete_panel,
        {
            "history_prior_bars", FIXED, "path_dn5", "path_up10",
            "path_bracket_net", "path_eod_net",
        },
        name="path_panel complete rows",
    )
    _require_finite(
        picks,
        {FIXED, "path_dn5", "path_up10", "path_bracket_net", "path_eod_net"},
        name="picks",
    )
    allowed_outcomes = {"tp", "sl", "eod"}
    observed_outcomes = set(complete_panel["path_bracket_outcome"])
    if not observed_outcomes.issubset(allowed_outcomes):
        raise ValueError(f"path_panel: invalid bracket outcomes: {observed_outcomes}")
    benchmark_dates = sorted(
        panel.loc[panel["benchmark_complete"], "date"].unique()
    )
    if len(benchmark_dates) < 180:
        raise ValueError("path_panel: fewer than 180 benchmark-complete dates")
    holdout_dates = set(benchmark_dates[-180:])
    holdout_hash = hashlib.sha256(
        "\n".join(pd.Timestamp(value).strftime("%Y-%m-%d") for value in sorted(holdout_dates)).encode()
    ).hexdigest()

    prediction_counts = (
        predictions.groupby(["scope", "date"]).size().groupby("scope").agg(["min", "max"])
    )
    if not (
        prediction_counts["min"].eq(100).all()
        and prediction_counts["max"].eq(100).all()
    ):
        raise ValueError("predictions: full PIT Top100 population is incomplete")
    for scope, key in (
        ("discovery_oof", "discovery_predictions"),
        ("locked_holdout", "locked_holdout_predictions"),
    ):
        observed_dates = predictions.loc[
            predictions["scope"].eq(scope), "date"
        ].nunique()
        expected_dates = int(coverage["effective_dates"][key]["n"])
        if observed_dates != expected_dates:
            raise ValueError(f"predictions: date count mismatch for {scope}")
    pick_counts = picks.groupby(["scope", "date", "policy"]).size()
    cost_error = panel[panel["path_complete"]].copy()
    expected_net = np.where(
        cost_error["path_bracket_outcome"].eq("tp"),
        0.05 - 0.0015,
        np.where(
            cost_error["path_bracket_outcome"].eq("sl"),
            -0.03 - 0.0015,
            cost_error["path_eod_net"],
        ),
    )
    cost_max_error = float(
        np.max(np.abs(cost_error["path_bracket_net"].to_numpy() - expected_net))
    )
    if cost_max_error > 1e-12:
        raise ValueError(f"path_panel: round-trip cost contract mismatch: {cost_max_error}")
    fixed_label_mismatch = int(
        (
            panel.loc[panel["path_complete"], FIXED].astype(int)
            != panel.loc[panel["path_complete"], "fp_fixed_outcome"].eq("up_first").astype(int)
        ).sum()
    )
    atr_valid = panel["path_complete"] & panel[ATR].notna()
    atr_label_mismatch = int(
        (
            panel.loc[atr_valid, ATR].astype(int)
            != panel.loc[atr_valid, "fp_atr_outcome"].eq("up_first").astype(int)
        ).sum()
    )
    if fixed_label_mismatch or atr_label_mismatch:
        raise ValueError(
            "path_panel: first-passage labels disagree with persisted outcomes"
        )
    if holdout_hash != coverage["sealed_holdout"]["dates_sha256"]:
        raise ValueError("coverage: sealed holdout hash mismatch")
    time_order = _time_order_audit(coverage, predictions)

    discovery_comparators = ["R1_repaired"]
    holdout_comparators = [
        "R1_repaired",
        "safeup_head",
        "monkey_seed42",
        "ATR_top3",
        "liquidity_matched",
    ]
    result = {
        "hashes": {name: _sha(path) for name, path in paths.items()},
        "audit_source_sha256": _sha(Path(__file__)),
        "contracts": {
            "path_panel_rows": int(len(panel)),
            "path_panel_dates": int(panel["date"].nunique()),
            "history_prior_min": int(panel["history_prior_bars"].min()),
            "history_prior_below70": int(
                (panel["history_prior_bars"] < 70).sum()
            ),
            "prediction_universe_minmax": prediction_counts.to_dict(orient="index"),
            "all_pick_policy_dates_exact3": bool((pick_counts == 3).all()),
            "all_persisted_picks_complete": bool(picks["path_complete"].all()),
            "fixed_label_mismatch": fixed_label_mismatch,
            "atr_label_mismatch": atr_label_mismatch,
            "cost_once_max_abs_error": cost_max_error,
            "holdout_dates_n": len(holdout_dates),
            "holdout_dates_sha256": holdout_hash,
            "coverage_holdout_sha256": coverage["sealed_holdout"]["dates_sha256"],
            "holdout_hash_matches": (
                holdout_hash == coverage["sealed_holdout"]["dates_sha256"]
            ),
            "completed_label_cutoff": completed_cutoff.strftime("%Y-%m-%d"),
            "bootstrap_seed": SEED,
            "bootstrap_draws": DRAWS,
            "population_scope": {
                "path_panel": "all persisted PIT Top100 path rows",
                "predictions": "all discovery OOF and locked-holdout candidate rows",
                "picks": "exact common complete Top3 policy cohorts",
            },
            "output_contract": (
                "read-only; no artifact writes; one final JSON document is "
                "emitted only after all checks pass"
            ),
        },
        "time_order": time_order,
        "table": _table(picks),
        "discovery_paired": {
            head: _paired(
                picks,
                scope="discovery_oof",
                primary=head,
                comparators=discovery_comparators,
                offset=100 + index * 100,
            )
            for index, head in enumerate(("fp_fixed_head", "fp_atr_head"))
        },
        "holdout_paired": _paired(
            picks,
            scope="locked_holdout",
            primary="fp_fixed_head",
            comparators=holdout_comparators,
            offset=500,
        ),
        "lowest_atr_diagnostic": _lowest_atr_diagnostic(
            predictions,
            picks,
        ),
        "volatility": _volatility_audit(
            panel, predictions, picks, holdout_dates
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

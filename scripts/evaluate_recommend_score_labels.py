"""전 유니버스 forward score-label 산출물의 순수 감사 평가기.

이 스크립트는 ``output/recommend_score_labels``의 완료 artifact만 읽고 별도 JSON
보고서를 만든다. 활성 추천, 모델 학습, 랭킹, 알림, ledger를 변경하지 않는다.
따라서 신규 가설 탐색이 아니라 이미 동결된 forward score의 측정 인프라이며,
현재 research moratorium을 우회하거나 판정 목표를 소급 변경하지 않는다.

감사 항목:

* p_up10 / p_dn5의 AUC, Brier, calibration과 날짜-cluster bootstrap CI
* 일별 top-N과 같은 날 전체 유니버스의 day-equal 비교
* 기록된 D-1 유동성 feature 기반 deterministic nearest matching
* ATR feature 기반 within-volatility-band lift
* all-score와 실제 delivery 성공 cohort의 분리
* 실제 목표일에 생성된 forward와 예정시각 fallback replay의 출처 분리

수익률 규율:

* 한 channel의 관측된 labeled 행 전체에 ``net_return`` 또는 ``eod_return_net``이 있으면 그대로
  사용하며 비용을 다시 빼지 않는다.
* 그렇지 않으면 ``eod_return``을 gross 진단값 그대로 사용한다. 이 평가기는 알림별 실제
  체결비용을 임의 가정해 차감하지 않으며 보고서에 gross임을 명시한다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from datetime import date as calendar_date
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from signals.recommend_score_labels import (  # noqa: E402
    FORWARD_PROVENANCE_COHORT,
    OFF_SCHEDULE_PROVENANCE_COHORT,
    SCHEDULED_REPLAY_PROVENANCE_COHORT,
    load_label_artifact,
)

DEFAULT_INPUT_ROOT = _ROOT / "output" / "recommend_score_labels"
DEFAULT_OUTPUT = _ROOT / "output" / "recommend_score_label_evaluation.json"

REPORT_SCHEMA = "recommend_score_label_evaluation.v3"
DEFAULT_TOP_NS = (3, 5, 10)
DEFAULT_BOOTSTRAPS = 1000
MAX_BOOTSTRAPS = 100_000
DEFAULT_SEED = 42
MIN_METRIC_ROWS = 30
MIN_CLUSTER_DAYS = 5
MIN_CLASS_COUNT = 5
CALIBRATION_EDGES = np.linspace(0.0, 1.0, 6)

LIQUIDITY_FEATURES = ("f_log_qv", "f_qv_rank_pct", "f_qv_rank")
VOLATILITY_FEATURES = ("f_atr_pct_14", "f_atr_xs_decile")
COMPARISON_METRICS = {
    "safe_up10_rate": "_safe_up10",
    "up10_rate": "up10",
    "dn3_rate": "_dn3",
    "dn5_rate": "dn5",
    "tp_first_rate": "_tp_first",
    "sl_first_rate": "_sl_first",
    "neither_rate": "_neither",
    "first_passage_net_mean": "_first_passage_net",
    "mfe_mean": "mfe",
    "mae_mean": "mae",
    "return_mean": "_audit_return",
}
UNKNOWN_PROVENANCE_COHORT = "unknown"
PROVENANCE_BY_EXECUTION_BASIS = {
    "delivery_sent_at": FORWARD_PROVENANCE_COHORT,
    "snapshot_created_at_delivery_failed": FORWARD_PROVENANCE_COHORT,
    "snapshot_created_at_no_receipt": FORWARD_PROVENANCE_COHORT,
    "delivery_sent_at_off_schedule": OFF_SCHEDULE_PROVENANCE_COHORT,
    (
        "snapshot_created_at_delivery_failed_off_schedule"
    ): OFF_SCHEDULE_PROVENANCE_COHORT,
    (
        "snapshot_created_at_no_receipt_off_schedule"
    ): OFF_SCHEDULE_PROVENANCE_COHORT,
    (
        "scheduled_slot_fallback_snapshot_outside_window"
    ): SCHEDULED_REPLAY_PROVENANCE_COHORT,
}


def _normalise(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _normalise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise(v) for v in value]
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _normalise(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _report_digest(report: dict) -> str:
    payload = {
        k: v for k, v in report.items()
        if k not in {"generated_at", "report_payload_sha256", "output_path"}
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _atomic_write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(
                document,
                f,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                indent=2,
            )
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def _metric_null(reason: str, **counts) -> dict:
    return {"value": None, "ci95": None, "reason": reason, **counts}


def _mean_and_cluster_ci(
    values: pd.Series | np.ndarray | list[float],
    *,
    n_boot: int,
    seed: int,
    min_days: int,
) -> dict:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    n = len(arr)
    if n == 0:
        return _metric_null("no_valid_days", n_days=0)
    point = float(arr.mean())
    if n < min_days:
        return {
            "value": point,
            "ci95": None,
            "reason": f"n_days<{min_days}",
            "n_days": n,
        }
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_boot, n))
    samples = arr[indices].mean(axis=1)
    return {
        "value": point,
        "ci95": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
        "reason": None,
        "n_days": n,
        "n_boot": n_boot,
        "cluster": "date",
    }


def _cluster_bootstrap_binary_metrics(
    frame: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
) -> tuple[list[float], list[float]]:
    groups = {
        date: group[["_prob", "_actual"]].reset_index(drop=True)
        for date, group in frame.groupby("_date", sort=True)
    }
    dates = np.array(list(groups), dtype=object)
    rng = np.random.default_rng(seed)
    aucs: list[float] = []
    briers: list[float] = []
    for _ in range(n_boot):
        sampled = rng.choice(dates, size=len(dates), replace=True)
        boot = pd.concat([groups[d] for d in sampled], ignore_index=True)
        y = boot["_actual"].to_numpy(int)
        p = boot["_prob"].to_numpy(float)
        briers.append(float(brier_score_loss(y, p)))
        if len(np.unique(y)) == 2:
            aucs.append(float(roc_auc_score(y, p)))
    return aucs, briers


def _calibration_report(
    frame: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
    min_days: int,
) -> dict:
    work = frame.copy()
    work["_bin"] = np.clip(
        np.digitize(work["_prob"], CALIBRATION_EDGES[1:-1], right=False),
        0,
        len(CALIBRATION_EDGES) - 2,
    )
    bins = []
    ece_numerator = 0.0
    total_rows = len(work)
    for bin_id in range(len(CALIBRATION_EDGES) - 1):
        sub = work[work["_bin"] == bin_id]
        if sub.empty:
            continue
        daily = sub.groupby("_date", sort=True).agg(
            predicted_mean=("_prob", "mean"),
            actual_rate=("_actual", "mean"),
            rows=("_actual", "size"),
        )
        actual_ci = _mean_and_cluster_ci(
            daily["actual_rate"],
            n_boot=n_boot,
            seed=seed + bin_id,
            min_days=min_days,
        )
        predicted = float(daily["predicted_mean"].mean())
        actual = float(daily["actual_rate"].mean())
        ece_numerator += len(sub) * abs(
            float(sub["_prob"].mean()) - float(sub["_actual"].mean())
        )
        bins.append({
            "bin": bin_id,
            "low": float(CALIBRATION_EDGES[bin_id]),
            "high": float(CALIBRATION_EDGES[bin_id + 1]),
            "n_rows": int(len(sub)),
            "n_days": int(len(daily)),
            "predicted_mean_day_equal": predicted,
            "actual_rate_day_equal": actual,
            "actual_rate_ci95": actual_ci["ci95"],
            "ci_reason": actual_ci["reason"],
        })
    return {
        "method": "fixed_probability_bins_day_equal",
        "edges": [float(v) for v in CALIBRATION_EDGES],
        "ece_row_weighted": (
            float(ece_numerator / total_rows) if total_rows else None
        ),
        "bins": bins,
    }


def _binary_head_report(
    frame: pd.DataFrame,
    probability: str,
    actual: str,
    *,
    n_boot: int,
    seed: int,
    min_rows: int,
    min_days: int,
    min_class_count: int,
) -> dict:
    if probability not in frame or actual not in frame:
        return {
            "status": "unavailable",
            "reason": f"missing_fields:{probability},{actual}",
            "auc": None,
            "brier": None,
            "calibration": None,
        }
    work = pd.DataFrame({
        "_date": frame["_date"],
        "_prob": pd.to_numeric(frame[probability], errors="coerce"),
        "_actual": pd.to_numeric(frame[actual], errors="coerce"),
    }).dropna()
    work = work[
        work["_prob"].between(0.0, 1.0)
        & work["_actual"].isin([0, 1])
    ]
    n_rows = len(work)
    n_days = int(work["_date"].nunique())
    positives = int(work["_actual"].sum())
    negatives = n_rows - positives
    counts = {
        "n_rows": n_rows,
        "n_days": n_days,
        "positives": positives,
        "negatives": negatives,
        "positive_rate": float(positives / n_rows) if n_rows else None,
    }
    reasons = []
    if n_rows < min_rows:
        reasons.append(f"n_rows<{min_rows}")
    if n_days < min_days:
        reasons.append(f"n_days<{min_days}")
    if positives < min_class_count:
        reasons.append(f"positives<{min_class_count}")
    if negatives < min_class_count:
        reasons.append(f"negatives<{min_class_count}")
    if reasons:
        return {
            "status": "insufficient",
            "reason": ";".join(reasons),
            **counts,
            "auc": None,
            "brier": None,
            "calibration": None,
        }

    y = work["_actual"].to_numpy(int)
    p = work["_prob"].to_numpy(float)
    auc = float(roc_auc_score(y, p))
    brier = float(brier_score_loss(y, p))
    auc_boot, brier_boot = _cluster_bootstrap_binary_metrics(
        work, n_boot=n_boot, seed=seed
    )
    min_valid_boot = max(20, int(n_boot * 0.5))
    auc_ci = (
        [float(np.quantile(auc_boot, 0.025)), float(np.quantile(auc_boot, 0.975))]
        if len(auc_boot) >= min_valid_boot
        else None
    )
    brier_ci = [
        float(np.quantile(brier_boot, 0.025)),
        float(np.quantile(brier_boot, 0.975)),
    ]
    return {
        "status": "ok",
        "reason": None,
        **counts,
        "auc": {
            "value": auc,
            "ci95": auc_ci,
            "point_method": "pooled_rows",
            "cluster": "date",
            "valid_bootstraps": len(auc_boot),
            "reason": None if auc_ci is not None else "insufficient_valid_bootstraps",
        },
        "brier": {
            "value": brier,
            "ci95": brier_ci,
            "point_method": "pooled_rows",
            "cluster": "date",
            "valid_bootstraps": len(brier_boot),
            "reason": None,
        },
        "calibration": _calibration_report(
            work, n_boot=n_boot, seed=seed + 100, min_days=min_days
        ),
    }


def _extract_feature(frame: pd.DataFrame, key: str) -> pd.Series:
    return frame["feature_values"].map(
        lambda value: value.get(key) if isinstance(value, dict) else None
    ).pipe(pd.to_numeric, errors="coerce")


def _choose_feature(
    frame: pd.DataFrame,
    candidates: tuple[str, ...],
) -> tuple[str | None, dict]:
    coverage = {}
    for key in candidates:
        values = _extract_feature(frame, key)
        coverage[key] = {
            "n": int(values.notna().sum()),
            "fraction": float(values.notna().mean()) if len(values) else 0.0,
        }
        if values.notna().mean() >= 0.8:
            return key, coverage
    return None, coverage


def _return_basis(frame: pd.DataFrame) -> tuple[pd.Series, dict]:
    net_fields = {}
    for field in ("net_return", "eod_return_net"):
        net_fields[field] = pd.to_numeric(
            frame[field]
            if field in frame
            else pd.Series(np.nan, index=frame.index, dtype=float),
            errors="coerce",
        )
    eod = pd.to_numeric(
        frame["eod_return"]
        if "eod_return" in frame
        else pd.Series(np.nan, index=frame.index, dtype=float),
        errors="coerce",
    )
    n_rows = len(frame)
    for field, values in net_fields.items():
        if n_rows and values.notna().sum() == n_rows:
            return values, {
                "field": field,
                "is_net": True,
                "cost_adjustment": "none_already_net",
                "n_rows": n_rows,
                "reason": None,
            }
    if eod.notna().any():
        return eod, {
            "field": "eod_return",
            "is_net": False,
            "cost_adjustment": "none_gross_diagnostic",
            "n_rows": int(eod.notna().sum()),
            "net_return_available_rows": {
                field: int(values.notna().sum())
                for field, values in net_fields.items()
            },
            "reason": (
                "net_return_not_complete; eod_return kept gross without assumed cost"
            ),
        }
    return pd.Series(np.nan, index=frame.index), {
        "field": None,
        "is_net": None,
        "cost_adjustment": "none",
        "n_rows": 0,
        "reason": "neither_complete_net_return_nor_eod_return_available",
    }


def _row_metric_mean(frame: pd.DataFrame, column: str) -> float | None:
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if len(values) else None


def _with_outcome_audit_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """라벨 row에서 사용자 목적에 직접 대응하는 비교 축을 만든다."""
    out = frame.copy()
    def numeric(name: str) -> pd.Series:
        if name not in out:
            return pd.Series(np.nan, index=out.index, dtype=float)
        return pd.to_numeric(out[name], errors="coerce")

    up10 = numeric("up10")
    dn5 = numeric("dn5")
    mae = numeric("mae")
    passage = out.get(
        "tp5_sl3_first_passage",
        pd.Series(index=out.index, dtype=object),
    ).astype("string")

    valid_safe = up10.notna() & dn5.notna()
    out["_safe_up10"] = np.where(
        valid_safe,
        (up10 > 0) & ~(dn5 > 0),
        np.nan,
    )
    out["_dn3"] = np.where(mae.notna(), mae <= -0.03, np.nan)
    out["_tp_first"] = np.where(
        passage.notna(), passage.eq("tp_first"), np.nan
    )
    out["_sl_first"] = np.where(
        passage.notna(),
        passage.isin(["sl_first", "sl_first_same_bar"]),
        np.nan,
    )
    out["_neither"] = np.where(
        passage.notna(), passage.eq("neither"), np.nan
    )
    out["_first_passage_net"] = numeric("tp5_sl3_return_net")
    return out


def _daily_comparison_report(
    records: list[dict],
    *,
    baseline_name: str,
    n_boot: int,
    seed: int,
    min_days: int,
    extra: dict | None = None,
) -> dict:
    if not records:
        return {
            "status": "unavailable",
            "reason": "no_comparable_days",
            "baseline": baseline_name,
            "n_days": 0,
            "metrics": None,
            **(extra or {}),
        }
    daily = pd.DataFrame(records)
    reports: dict[str, dict[str, Any]] = {}
    for metric in COMPARISON_METRICS:
        selected_col = f"selected_{metric}"
        baseline_col = f"baseline_{metric}"
        valid = daily[[selected_col, baseline_col]].dropna()
        metric_coverage = {
            "comparable_dates": int(len(valid)),
            "unavailable_dates": int(len(daily) - len(valid)),
            "rule": "both_sides_complete_or_neither_side_used",
        }
        if valid.empty:
            reports[metric] = {
                "coverage": metric_coverage,
                "selected_day_equal": None,
                "baseline_day_equal": None,
                "difference_selected_minus_baseline": _metric_null(
                    "no_valid_days", n_days=0
                ),
            }
            continue
        difference = valid[selected_col] - valid[baseline_col]
        reports[metric] = {
            "coverage": metric_coverage,
            "selected_day_equal": float(valid[selected_col].mean()),
            "baseline_day_equal": float(valid[baseline_col].mean()),
            "difference_selected_minus_baseline": _mean_and_cluster_ci(
                difference,
                n_boot=n_boot,
                seed=seed,
                min_days=min_days,
            ),
        }
    n_days = int(daily["_date"].nunique())
    return {
        "status": "ok" if n_days >= min_days else "insufficient",
        "reason": None if n_days >= min_days else f"n_days<{min_days}",
        "baseline": baseline_name,
        "n_days": n_days,
        "metrics": reports,
        **(extra or {}),
    }


def _selection_plan(
    date: str,
    selected: pd.DataFrame,
    baseline_groups: list[list[str]],
    *,
    reason: str | None = None,
    selected_bands: list[str] | None = None,
) -> dict:
    """A predictor-only plan. Never put an outcome or label status here."""
    return {
        "date": str(date),
        "selected": [
            {"coin": str(row.coin), "rank": int(row.rank)}
            for row in selected[["coin", "rank"]].itertuples(index=False)
        ],
        "baseline_groups": baseline_groups,
        "selected_bands": selected_bands or [],
        "plan_status": "unavailable" if reason else "planned",
        "plan_reason": reason,
    }


def _build_comparison_plans(frame: pd.DataFrame, top_ns: tuple[int, ...]) -> dict:
    """Freeze every comparison before making any outcome-dependent decision.

    The projection is deliberate: changing labels, path quality or returns
    cannot alter top-N membership, feature coverage, matching or ATR bands.
    """
    work = frame.reindex(
        columns=["_date", "coin", "rank", "feature_values"]
    ).copy()
    work = work.sort_values(["_date", "rank", "coin"]).reset_index(drop=True)
    liquidity, liquidity_coverage = _choose_feature(work, LIQUIDITY_FEATURES)
    volatility, volatility_coverage = _choose_feature(work, VOLATILITY_FEATURES)
    work["_match_feature"] = (
        _extract_feature(work, liquidity) if liquidity else np.nan
    )
    work["_vol_feature"] = (
        _extract_feature(work, volatility) if volatility else np.nan
    )
    result = {
        "top_n_vs_full_universe": {},
        "liquidity_matched_baseline": {},
        "within_volatility_band_lift": {},
    }
    for top_n in top_ns:
        full_plans, liquidity_plans, volatility_plans = [], [], []
        for date, day in work.groupby("_date", sort=True):
            selected = day.head(top_n)
            full_plans.append(_selection_plan(
                date, selected, [day["coin"].astype(str).tolist()]
            ))

            controls = day.iloc[top_n:].dropna(subset=["_match_feature"])
            reason = None
            matched_coins = []
            if liquidity is None:
                reason = "liquidity_feature_unavailable_or_coverage_below_80pct"
            elif selected["_match_feature"].isna().any():
                reason = "selected_liquidity_feature_missing"
            elif len(controls) < len(selected):
                reason = "insufficient_liquidity_controls"
            else:
                available = set(controls.index)
                for _, pick in selected.iterrows():
                    candidates = controls.loc[sorted(available)].copy()
                    candidates["_distance"] = (
                        candidates["_match_feature"] - pick["_match_feature"]
                    ).abs()
                    candidates = candidates.sort_values(
                        ["_distance", "coin"], kind="mergesort"
                    )
                    chosen = candidates.index[0]
                    matched_coins.append(str(candidates.loc[chosen, "coin"]))
                    available.remove(chosen)
            liquidity_plans.append(_selection_plan(
                date, selected, [matched_coins] if not reason else [],
                reason=reason,
            ))

            valid = day.dropna(subset=["_vol_feature"]).sort_values(
                ["_vol_feature", "coin"], kind="mergesort"
            ).copy()
            reason = None
            groups, selected_bands = [], []
            if volatility is None:
                reason = "atr_feature_unavailable_or_coverage_below_80pct"
            elif selected["_vol_feature"].isna().any():
                reason = "selected_atr_feature_missing"
            elif len(valid) < 6:
                reason = "insufficient_atr_feature_rows"
            else:
                percentile = valid["_vol_feature"].rank(method="first", pct=True)
                valid["_vol_band"] = np.minimum(
                    np.ceil(percentile * 3).astype(int) - 1, 2
                )
                non_top = valid[~valid.index.isin(selected.index)]
                for index in selected.index:
                    band = int(valid.loc[index, "_vol_band"])
                    pool = non_top[non_top["_vol_band"] == band]
                    selected_bands.append(("low", "mid", "high")[band])
                    if pool.empty:
                        reason = "empty_atr_band_controls"
                    groups.append(pool["coin"].astype(str).tolist())
            volatility_plans.append(_selection_plan(
                date, selected, groups, reason=reason,
                selected_bands=selected_bands,
            ))
        result["top_n_vs_full_universe"][str(top_n)] = {
            "plans": full_plans,
            "baseline": "same_day_full_universe_including_top_n",
            "seed": top_n,
            "extra": {"top_n": top_n},
        }
        result["liquidity_matched_baseline"][str(top_n)] = {
            "plans": liquidity_plans,
            "baseline": "deterministic_nearest_liquidity_non_top_controls",
            "seed": top_n + 200,
            "extra": {
                "top_n": top_n,
                "feature": liquidity,
                "feature_coverage": liquidity_coverage,
                "matching": "absolute_distance_without_replacement_tie_by_coin",
            },
        }
        result["within_volatility_band_lift"][str(top_n)] = {
            "plans": volatility_plans,
            "baseline": "same_day_non_top_controls_within_atr_tertile",
            "seed": top_n + 400,
            "extra": {
                "top_n": top_n,
                "feature": volatility,
                "feature_coverage": volatility_coverage,
                "bands": "same-day ATR tertiles",
            },
        }
    return result


def _evaluate_comparison_plan(
    bundle: dict,
    frame: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
    min_days: int,
) -> dict:
    """Join outcomes to frozen identities; missing members are never replaced."""
    records, audits = [], []
    days = {
        str(date): day.set_index("coin", drop=False)
        for date, day in frame.groupby("_date", sort=True)
    }
    for plan in bundle["plans"]:
        audit = {
            **plan,
            "evaluation_status": "unavailable",
            "evaluation_reasons": [],
            "unobserved_selected": [],
            "unobserved_baseline": [],
            "unavailable_metrics": {},
        }
        audits.append(audit)
        if plan["plan_status"] != "planned":
            audit["evaluation_reasons"] = [plan["plan_reason"]]
            continue
        day = days[plan["date"]]
        selected_coins = [item["coin"] for item in plan["selected"]]
        baseline_coins = sorted({
            coin for group in plan["baseline_groups"] for coin in group
        })
        for role, coins in (
            ("selected", selected_coins), ("baseline", baseline_coins)
        ):
            missing = [
                {"coin": coin, "label_status": str(day.loc[coin, "label_status"])}
                for coin in coins if day.loc[coin, "label_status"] != "labeled"
            ]
            audit[f"unobserved_{role}"] = missing
            if missing:
                audit["evaluation_reasons"].append(f"{role}_outcome_unavailable")
        if audit["evaluation_reasons"]:
            continue

        audit["evaluation_status"] = "comparable"
        record = {"_date": plan["date"]}
        for metric, column in COMPARISON_METRICS.items():
            # Check every required member BEFORE either mean. In particular,
            # a legacy missing metric must not shrink one side via dropna().
            values = pd.to_numeric(
                day[column] if column in day
                else pd.Series(np.nan, index=day.index),
                errors="coerce",
            )
            missing = {}
            for role, coins in (
                ("selected", selected_coins), ("baseline", baseline_coins)
            ):
                absent = [
                    coin for coin in coins if not np.isfinite(values.loc[coin])
                ]
                if absent:
                    missing[role] = absent
            if missing:
                audit["unavailable_metrics"][metric] = missing
                record[f"selected_{metric}"] = None
                record[f"baseline_{metric}"] = None
            else:
                record[f"selected_{metric}"] = float(
                    values.loc[selected_coins].mean()
                )
                record[f"baseline_{metric}"] = float(np.mean([
                    values.loc[group].mean() for group in plan["baseline_groups"]
                ]))
        records.append(record)

    reasons: dict[str, int] = {}
    for audit in audits:
        for reason in audit["evaluation_reasons"]:
            reasons[reason] = reasons.get(reason, 0) + 1
    comparable = [
        audit for audit in audits if audit["evaluation_status"] == "comparable"
    ]
    coverage = {
        "recorded_dates": len(audits),
        "planned_dates": sum(p["plan_status"] == "planned" for p in audits),
        "comparable_dates": len(comparable),
        "unavailable_dates": len(audits) - len(comparable),
        "reasons": reasons,
        "rule": "fixed_members_complete_labels_before_any_mean_no_replacement",
    }
    extra = {
        **bundle["extra"], "coverage": coverage, "selection_audit": audits,
    }
    if "matching" in extra:
        extra["matched_pairs"] = sum(len(p["selected"]) for p in comparable)
        extra["skipped_dates"] = coverage["unavailable_dates"]
    if "bands" in extra:
        extra["selected_band_counts"] = {
            band: sum(p["selected_bands"].count(band) for p in comparable)
            for band in ("low", "mid", "high")
        }
        extra["skipped_dates"] = coverage["unavailable_dates"]
    result = _daily_comparison_report(
        records, baseline_name=bundle["baseline"], n_boot=n_boot,
        seed=seed + bundle["seed"], min_days=min_days, extra=extra,
    )
    # Preserve the useful legacy explanation when no feature can be selected.
    if not records and bundle["extra"].get("feature", True) is None:
        result["reason"] = (
            "liquidity_feature_unavailable_or_coverage_below_80pct"
            if "matching" in extra
            else "atr_feature_unavailable_or_coverage_below_80pct"
        )
    return result


def _delivery_cohort(frame: pd.DataFrame) -> tuple[pd.DataFrame | None, dict]:
    for explicit in ("was_delivered", "delivered"):
        if explicit in frame and frame[explicit].notna().any():
            mask = frame[explicit].fillna(False).astype(bool)
            return frame[mask].copy(), {
                "status": "available",
                "selection": f"explicit_row_field:{explicit}",
                "reason": None,
            }
    if "delivery_ok" in frame and frame["delivery_ok"].notna().any():
        delivery = frame["delivery_ok"].fillna(False).astype(bool)
        # Receipt는 snapshot 단위이므로 실제 메시지에 노출된 기존 top-3만 delivered로 본다.
        mask = delivery & (pd.to_numeric(frame["rank"], errors="coerce") <= 3)
        selected = frame[mask].copy()
        return selected, {
            "status": "available" if len(selected) else "empty",
            "selection": "delivery_ok_true_and_rank_le_3",
            "reason": None if len(selected) else "no_successful_delivery_rows",
        }
    return None, {
        "status": "unavailable",
        "selection": None,
        "reason": "delivery_fields_absent_in_label_artifacts",
    }


def _outcome_coverage(frame: pd.DataFrame) -> dict:
    observed = frame[frame["label_status"] == "labeled"]
    daily = []
    for date, day in frame.groupby("_date", sort=True):
        missing = day[day["label_status"] != "labeled"].sort_values(
            ["rank", "coin"]
        )
        daily.append({
            "date": str(date),
            "expected_rows": int(len(day)),
            "observed_rows": int(len(day) - len(missing)),
            "unobserved": [
                {"coin": str(row.coin), "rank": int(row.rank),
                 "label_status": str(row.label_status)}
                for row in missing.itertuples(index=False)
            ],
        })
    return {
        "status": "available",
        "expected_rows": int(len(frame)),
        "observed_rows": int(len(observed)),
        "unobserved_rows": int(len(frame) - len(observed)),
        "unobserved_fraction": (
            float(1 - len(observed) / len(frame)) if len(frame) else None
        ),
        "recorded_dates": int(frame["_date"].nunique()),
        "observed_dates": int(observed["_date"].nunique()),
        "label_status_counts": {
            str(status): int(count)
            for status, count in frame["label_status"].value_counts().items()
        },
        "daily": daily,
    }


def _delivered_outcomes(frame: pd.DataFrame) -> tuple[pd.DataFrame | None, dict]:
    expected, availability = _delivery_cohort(frame)
    if expected is None:
        return None, {
            **availability,
            "outcome_coverage": {
                "status": "unavailable",
                "reason": availability["reason"],
                "expected_rows": None,
                "observed_rows": None,
                "unobserved_rows": None,
                "unobserved_fraction": None,
            },
        }
    observed = expected[expected["label_status"] == "labeled"].copy()
    return observed, {
        **availability,
        "outcome_coverage": {
            **_outcome_coverage(expected),
            "meaning": (
                "missing outcomes among recorded delivered picks; "
                "not a delivery failure rate"
            ),
        },
    }


def _model_version_summary(frame: pd.DataFrame) -> dict:
    fields = ("model_id", "model_version", "rule_version", "score_source_sha256")
    versions: dict[tuple[str, ...], list[dict]] = {}
    for row in frame.to_dict("records"):
        identity = row.get("_model_identity") or {}
        key = tuple(str(identity.get(field) or "unknown") for field in fields)
        versions.setdefault(key, []).append(row)
    groups = []
    for key, rows in sorted(versions.items()):
        dates = sorted({str(row["_date"]) for row in rows})
        groups.append({
            **dict(zip(fields, key)),
            "recorded_rows": len(rows),
            "observed_rows": sum(row["label_status"] == "labeled" for row in rows),
            "n_dates": len(dates), "dates": dates,
        })
    return {
        "groups": groups,
        "mixed_recorded_identities": len(groups) > 1,
        "has_unknown_identity_fields": any("unknown" in key for key in versions),
        "note": (
            "diagnostic only: channel statistics retain mixed recorded versions; "
            "these counts are not a fresh-holdout or promotion eligibility claim"
        ),
    }


def _path_quality_summary(frame: pd.DataFrame) -> dict:
    """flat_filled 비중을 보고서에 상시 노출한다.

    flat_filled 행은 무체결 gap 을 flat 봉으로 재구성한 것이라 MFE/MAE 와
    TP/SL first-passage 가 0 쪽으로 축소 편향된다.  경로 의존 주장은
    complete_path_only 코호트로만 할 것.
    """
    if "path_quality" not in frame or not len(frame):
        return {
            "status": "unavailable",
            "reason": "path_quality_field_absent_or_no_rows",
        }
    quality = frame["path_quality"].fillna("unknown").astype(str)
    flat = frame[quality == "flat_filled"]
    flat_bars = (
        pd.to_numeric(flat["flat_filled_bars"], errors="coerce").dropna()
        if len(flat) and "flat_filled_bars" in flat
        else pd.Series(dtype=float)
    )
    return {
        "status": "available",
        "counts": {
            str(key): int(value)
            for key, value in quality.value_counts().items()
        },
        "flat_filled_share": round(float((quality == "flat_filled").mean()), 6),
        "flat_filled_bars_mean": (
            round(float(flat_bars.mean()), 4) if len(flat_bars) else None
        ),
        "bias_note": (
            "flat_filled rows reconstruct no-trade gaps as flat bars; "
            "MFE/MAE and TP/SL first-passage are biased toward zero on those "
            "rows. cohorts.complete_path_only removes them, but flat-fill is "
            "a liquidity proxy so that cohort systematically drops the "
            "thinnest names and is optimistically biased the other way — "
            "always report both cohorts together, never cite either alone "
            "for path-dependent claims"
        ),
    }


def _path_quality_cohort(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame | None, dict]:
    if "path_quality" not in frame or not frame["path_quality"].notna().any():
        return None, {
            "status": "unavailable",
            "selection": None,
            "reason": "path_quality_field_absent_in_label_artifacts",
        }
    mask = frame["path_quality"].astype(str) == "complete"
    selected = frame[mask].copy()
    return selected, {
        "status": "available" if len(selected) else "empty",
        "selection": "path_quality_complete_only",
        "reason": None if len(selected) else "no_complete_path_rows",
    }


def _cohort_report(
    frame: pd.DataFrame | None,
    *,
    availability: dict,
    n_boot: int,
    seed: int,
    min_rows: int,
    min_days: int,
) -> dict:
    if frame is None:
        return {
            **availability,
            "n_rows": 0,
            "n_days": 0,
            "heads": {"p_up10": None, "p_dn5": None},
        }
    return {
        **availability,
        "n_rows": int(len(frame)),
        "n_days": int(frame["_date"].nunique()) if len(frame) else 0,
        "outcomes": {
            output_name: _row_metric_mean(frame, column)
            for output_name, column in COMPARISON_METRICS.items()
        },
        "heads": {
            "p_up10": _binary_head_report(
                frame,
                "p_up10",
                "up10",
                n_boot=n_boot,
                seed=seed,
                min_rows=min_rows,
                min_days=min_days,
                min_class_count=MIN_CLASS_COUNT,
            ),
            "p_dn5": _binary_head_report(
                frame,
                "p_dn5",
                "dn5",
                n_boot=n_boot,
                seed=seed + 1,
                min_rows=min_rows,
                min_days=min_days,
                min_class_count=MIN_CLASS_COUNT,
            ),
        },
    }


def _resolve_artifact_provenance(document: dict) -> tuple[str, str]:
    """Artifact/row provenance를 교차검증하고 구형 artifact는 basis로 복원한다."""
    rows = document.get("rows") or []
    cohorts = {
        str(value)
        for value in [
            document.get("provenance_cohort"),
            *(row.get("provenance_cohort") for row in rows),
        ]
        if value not in (None, "")
    }
    bases = {
        str(value)
        for value in [
            document.get("execution_time_basis"),
            *(row.get("execution_time_basis") for row in rows),
        ]
        if value not in (None, "")
    }
    eligibility = {
        bool(value)
        for value in [
            document.get("forward_eligible"),
            *(row.get("forward_eligible") for row in rows),
        ]
        if value is not None
    }
    if len(cohorts) > 1:
        raise ValueError("mixed_provenance_cohort")
    if len(bases) > 1:
        raise ValueError("mixed_execution_time_basis")
    if len(eligibility) > 1:
        raise ValueError("mixed_forward_eligible")

    explicit = next(iter(cohorts), None)
    basis = next(iter(bases), None)
    inferred = (
        PROVENANCE_BY_EXECUTION_BASIS.get(basis)
        if basis is not None
        else None
    )
    if explicit is not None and inferred is not None and explicit != inferred:
        raise ValueError("provenance_conflicts_with_execution_time_basis")

    cohort = explicit or inferred or UNKNOWN_PROVENANCE_COHORT
    expected_eligible = (
        True if cohort == FORWARD_PROVENANCE_COHORT
        else False
        if cohort in {
            OFF_SCHEDULE_PROVENANCE_COHORT,
            SCHEDULED_REPLAY_PROVENANCE_COHORT,
        }
        else None
    )
    if (
        eligibility
        and expected_eligible is not None
        and next(iter(eligibility)) != expected_eligible
    ):
        raise ValueError("forward_eligible_conflicts_with_provenance")
    source = (
        "explicit"
        if explicit is not None
        else "inferred_from_execution_time_basis"
        if inferred is not None
        else "unknown"
    )
    return cohort, source


def _load_complete_rows(
    input_root: Path,
    *,
    through_date: calendar_date,
) -> tuple[dict[tuple[str, str], list[dict]], dict]:
    artifacts = sorted(input_root.glob("*/*.json"))
    accepted: dict[tuple[str, str, str], tuple[Path, dict]] = {}
    conflicts: set[tuple[str, str, str]] = set()
    skipped = []
    for path in artifacts:
        if ".limit" in path.name:
            skipped.append({"path": str(path), "reason": "market_limited_snapshot"})
            continue
        try:
            document = load_label_artifact(path)
        except Exception as exc:
            skipped.append({
                "path": str(path),
                "reason": f"{type(exc).__name__}:{exc}",
            })
            continue
        if document.get("artifact_status") != "complete":
            skipped.append({
                "path": str(path),
                "reason": f"artifact_status={document.get('artifact_status')}",
            })
            continue
        asof = str(document.get("asof"))
        try:
            artifact_date = calendar_date.fromisoformat(asof)
        except ValueError:
            skipped.append({"path": str(path), "reason": "invalid_artifact_asof"})
            continue
        if path.parent.name != asof:
            skipped.append({
                "path": str(path),
                "reason": "artifact_path_asof_identity_mismatch",
            })
            continue
        if artifact_date > through_date:
            skipped.append({
                "path": str(path),
                "reason": (
                    f"artifact_after_completed_cutoff={through_date.isoformat()}"
                ),
            })
            continue
        slot = str(document.get("slot", "unknown"))
        ranking = str(document.get("ranking", "unknown"))
        identity = (slot, ranking, asof)
        if identity in accepted:
            old_path, old = accepted[identity]
            same_snapshot = (
                old.get("snapshot_id") == document.get("snapshot_id")
            )
            same_label = (
                old.get("label_payload_sha256")
                == document.get("label_payload_sha256")
            )
            if same_snapshot and same_label:
                skipped.append({"path": str(path), "reason": "duplicate_snapshot_id"})
            else:
                conflicts.add(identity)
                skipped.extend([
                    {"path": str(old_path), "reason": "conflicting_channel_date"},
                    {"path": str(path), "reason": "conflicting_channel_date"},
                ])
            continue
        accepted[identity] = (path, document)
    for identity in conflicts:
        accepted.pop(identity, None)

    channels: dict[tuple[str, str], list[dict]] = {}
    used: list[dict[str, Any]] = []
    for (slot, ranking, asof), (path, document) in sorted(accepted.items()):
        all_rows = document.get("rows") or []
        # Retain structurally unobserved candidates for selection and coverage.
        # A complete artifact may contain ONLY halted rows; it is still evidence.
        if not all_rows or any(
            row.get("label_status") not in {"labeled", "halted_no_observations"}
            for row in all_rows
        ):
            skipped.append({"path": str(path), "reason": "complete_without_all_labeled_rows"})
            continue
        observed_rows = sum(row["label_status"] == "labeled" for row in all_rows)
        try:
            provenance, provenance_source = _resolve_artifact_provenance(document)
        except ValueError as exc:
            skipped.append({
                "path": str(path),
                "reason": f"invalid_provenance:{exc}",
            })
            continue
        model_identity = {
            "model_id": (document.get("snapshot_model") or {}).get("id"),
            "model_version": (document.get("snapshot_model") or {}).get("version"),
            "rule_version": (document.get("snapshot_rule") or {}).get("version"),
            "score_source_sha256": (document.get("snapshot_code") or {}).get(
                "score_source_sha256"
            ),
        }
        for row in all_rows:
            channels.setdefault((slot, ranking), []).append({
                **row,
                "_date": asof,
                "_artifact": str(path),
                "_provenance_cohort": provenance,
                "_model_identity": model_identity,
            })
        used.append({
            "path": str(path),
            "snapshot_id": document.get("snapshot_id"),
            "snapshot_payload_sha256": document.get(
                "snapshot_payload_sha256"
            ),
            "label_payload_sha256": document.get("label_payload_sha256"),
            "model_id": (document.get("snapshot_model") or {}).get("id"),
            "rule_version": (document.get("snapshot_rule") or {}).get(
                "version"
            ),
            "score_source_sha256": (
                document.get("snapshot_code") or {}
            ).get("score_source_sha256"),
            "label_code_sha256": (
                document.get("label_code") or {}
            ).get("sha256"),
            "path_input_sha256": (
                document.get("path_input") or {}
            ).get("sha256"),
            "slot": slot,
            "ranking": ranking,
            "asof": asof,
            "rows": observed_rows,
            "recorded_rows": len(all_rows),
            "unobserved_rows": len(all_rows) - observed_rows,
            "provenance_cohort": provenance,
            "provenance_source": provenance_source,
        })
    provenance_artifacts: dict[str, int] = {}
    provenance_rows: dict[str, int] = {}
    provenance_recorded_rows: dict[str, int] = {}
    for item in used:
        cohort = str(item["provenance_cohort"])
        provenance_artifacts[cohort] = provenance_artifacts.get(cohort, 0) + 1
        provenance_rows[cohort] = provenance_rows.get(cohort, 0) + int(item["rows"])
        provenance_recorded_rows[cohort] = (
            provenance_recorded_rows.get(cohort, 0) + int(item["recorded_rows"])
        )
    return channels, {
        "found": len(artifacts),
        "complete_used": len(used),
        "used": used,
        "skipped": skipped,
        "provenance_artifacts": provenance_artifacts,
        "provenance_rows": provenance_rows,
        "provenance_recorded_rows": provenance_recorded_rows,
    }


def _excluded_provenance_report(
    frame: pd.DataFrame,
    *,
    cohort: str,
    n_boot: int,
    seed: int,
    min_rows: int,
    min_days: int,
) -> dict:
    observed = frame[frame["label_status"] == "labeled"].copy()
    observed = _with_outcome_audit_columns(observed)
    observed["_audit_return"], return_basis = _return_basis(observed)
    delivered, delivery_availability = _delivered_outcomes(frame)
    if delivered is not None:
        delivered = observed.loc[delivered.index].copy()
    status = "available" if len(observed) else "empty"
    return {
        "included_in_default_forward_statistics": False,
        "reason": (
            "scheduled execution fallback is replay, not an observed forward decision"
            if cohort == SCHEDULED_REPLAY_PROVENANCE_COHORT
            else "provenance is not verified as an observed forward decision"
        ),
        "n_rows": int(len(observed)),
        "n_dates": int(observed["_date"].nunique()),
        "recorded_n_rows": int(len(frame)),
        "recorded_n_dates": int(frame["_date"].nunique()),
        "outcome_coverage": _outcome_coverage(frame),
        "model_versions": _model_version_summary(frame),
        "return_basis": return_basis,
        "cohorts": {
            "all_scores": _cohort_report(
                observed,
                availability={
                    "status": status,
                    "selection": f"provenance_cohort={cohort}:all_recorded_scores",
                    "reason": None if len(observed) else "no_rows",
                    "outcome_coverage": _outcome_coverage(frame),
                },
                n_boot=n_boot,
                seed=seed,
                min_rows=min_rows,
                min_days=min_days,
            ),
            "delivered": _cohort_report(
                delivered,
                availability=delivery_availability,
                n_boot=n_boot,
                seed=seed + 20,
                min_rows=min_rows,
                min_days=min_days,
            ),
        },
    }


def _evaluate_channel(
    rows: list[dict],
    *,
    top_ns: tuple[int, ...],
    n_boot: int,
    seed: int,
    min_rows: int,
    min_days: int,
) -> dict:
    raw = pd.DataFrame(rows)
    raw["rank"] = pd.to_numeric(raw.get("rank"), errors="coerce")
    raw = raw.sort_values(["_date", "rank", "coin"]).reset_index(drop=True)
    recorded = raw[
        raw["_provenance_cohort"] == FORWARD_PROVENANCE_COHORT
    ].copy()

    # All three plan families are frozen on raw candidates BEFORE observing
    # label status, returns, delivery outcomes or path-quality subsets.
    plans = _build_comparison_plans(recorded, top_ns)
    recorded = _with_outcome_audit_columns(recorded)
    frame = recorded[recorded["label_status"] == "labeled"].copy()
    frame["_audit_return"], return_basis = _return_basis(frame)
    recorded["_audit_return"] = frame["_audit_return"]
    delivered, delivery_availability = _delivered_outcomes(recorded)
    if delivered is not None:
        delivered = frame.loc[delivered.index].copy()
    complete_path, path_quality_availability = _path_quality_cohort(frame)
    all_availability = {
        "status": "available" if len(frame) else "empty",
        "selection": "all_recorded_universe_scores",
        "reason": None if len(frame) else "no_forward_observed_rows",
        "outcome_coverage": _outcome_coverage(recorded),
    }

    provenance_counts = {}
    for cohort, group in raw.groupby("_provenance_cohort", sort=True, dropna=False):
        observed = group[group["label_status"] == "labeled"]
        provenance_counts[cohort] = {
            "n_rows": int(len(observed)),
            "n_dates": int(observed["_date"].nunique()),
            "recorded_n_rows": int(len(group)),
            "recorded_n_dates": int(group["_date"].nunique()),
            "included_in_default_forward_statistics": (
                cohort == FORWARD_PROVENANCE_COHORT
            ),
        }
    for cohort in (
        FORWARD_PROVENANCE_COHORT,
        OFF_SCHEDULE_PROVENANCE_COHORT,
        SCHEDULED_REPLAY_PROVENANCE_COHORT,
        UNKNOWN_PROVENANCE_COHORT,
    ):
        provenance_counts.setdefault(cohort, {
            "n_rows": 0, "n_dates": 0, "recorded_n_rows": 0, "recorded_n_dates": 0,
            "included_in_default_forward_statistics": (
                cohort == FORWARD_PROVENANCE_COHORT
            ),
        })
    excluded_names = {
        OFF_SCHEDULE_PROVENANCE_COHORT,
        SCHEDULED_REPLAY_PROVENANCE_COHORT,
        UNKNOWN_PROVENANCE_COHORT,
        *(
            str(value)
            for value in raw["_provenance_cohort"].dropna().unique()
            if value != FORWARD_PROVENANCE_COHORT
        ),
    }
    excluded = {}
    for offset, cohort in enumerate(sorted(excluded_names)):
        subset = raw[raw["_provenance_cohort"] == cohort].copy()
        excluded[cohort] = _excluded_provenance_report(
            subset, cohort=cohort, n_boot=n_boot,
            seed=seed + 1000 + offset * 100,
            min_rows=min_rows, min_days=min_days,
        )
    input_observed = raw[raw["label_status"] == "labeled"]
    return {
        "n_rows": int(len(frame)),
        "n_dates": int(frame["_date"].nunique()),
        "input_n_rows": int(len(input_observed)),
        "input_n_dates": int(input_observed["_date"].nunique()),
        "recorded_n_rows": int(len(recorded)),
        "recorded_n_dates": int(recorded["_date"].nunique()),
        "input_recorded_n_rows": int(len(raw)),
        "input_recorded_n_dates": int(raw["_date"].nunique()),
        "outcome_coverage": _outcome_coverage(recorded),
        "model_versions": _model_version_summary(recorded),
        "return_basis": return_basis,
        "path_quality": _path_quality_summary(frame),
        "provenance": {
            "default_forward_cohort": FORWARD_PROVENANCE_COHORT,
            "counts": provenance_counts,
            "excluded_from_default": sorted(excluded),
        },
        "cohorts": {
            "all_scores": _cohort_report(
                frame, availability=all_availability,
                n_boot=n_boot, seed=seed, min_rows=min_rows, min_days=min_days,
            ),
            "delivered": _cohort_report(
                delivered, availability=delivery_availability,
                n_boot=n_boot, seed=seed + 20, min_rows=min_rows, min_days=min_days,
            ),
            "complete_path_only": _cohort_report(
                complete_path, availability=path_quality_availability,
                n_boot=n_boot, seed=seed + 40, min_rows=min_rows, min_days=min_days,
            ),
        },
        "excluded_provenance_cohorts": excluded,
        **{
            family: {
                n: _evaluate_comparison_plan(
                    bundle, recorded, n_boot=n_boot, seed=seed, min_days=min_days
                )
                for n, bundle in bundles.items()
            }
            for family, bundles in plans.items()
        },
    }


def evaluate_label_root(
    input_root: str | Path = DEFAULT_INPUT_ROOT,
    *,
    output_path: str | Path = DEFAULT_OUTPUT,
    top_ns: tuple[int, ...] = DEFAULT_TOP_NS,
    n_boot: int = DEFAULT_BOOTSTRAPS,
    seed: int = DEFAULT_SEED,
    min_rows: int = MIN_METRIC_ROWS,
    min_days: int = MIN_CLUSTER_DAYS,
    through_date: str | calendar_date | None = None,
) -> dict:
    if (
        isinstance(n_boot, bool)
        or not isinstance(n_boot, int)
        or not 1 <= n_boot <= MAX_BOOTSTRAPS
    ):
        raise ValueError(
            f"n_boot must be an integer in [1, {MAX_BOOTSTRAPS}]"
        )
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed < 2**64
    ):
        raise ValueError("seed must be an integer in [0, 2**64)")
    for name, value in (("min_rows", min_rows), ("min_days", min_days)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if (
        not top_ns
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in top_ns
        )
    ):
        raise ValueError("top_ns must contain positive integers")
    top_ns = tuple(sorted(set(top_ns)))
    if through_date is None:
        completed_cutoff = (
            datetime.now(ZoneInfo("Asia/Seoul")).date()
            - timedelta(days=1)
        )
    elif isinstance(through_date, datetime):
        through_datetime = through_date
        if through_datetime.tzinfo is not None:
            through_datetime = through_datetime.astimezone(
                ZoneInfo("Asia/Seoul")
            )
        completed_cutoff = through_datetime.date()
    elif isinstance(through_date, calendar_date):
        completed_cutoff = through_date
    else:
        try:
            completed_cutoff = calendar_date.fromisoformat(str(through_date))
        except ValueError as exc:
            raise ValueError("through_date must be YYYY-MM-DD") from exc

    root = Path(input_root)
    channels, artifact_audit = _load_complete_rows(
        root,
        through_date=completed_cutoff,
    )
    report = {
        "schema": REPORT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "forward measurement only; no active model/ranking/alert mutation and "
            "no moratorium decision-rule change"
        ),
        "input_root": str(root),
        "methodology": {
            "date_aggregation": (
                "baseline lifts and calibration are day-equal; AUC/Brier point "
                "estimates are pooled rows with whole-date cluster bootstrap"
            ),
            "uncertainty": "date_cluster_bootstrap",
            "n_boot": n_boot,
            "seed": seed,
            "min_rows": min_rows,
            "min_days": min_days,
            "top_ns": list(top_ns),
            "completed_through_date_kst": completed_cutoff.isoformat(),
            "return_rule": (
                "use net_return/eod_return_net without extra cost only when complete "
                "for observed labeled channel rows; otherwise use eod_return as gross diagnostic without "
                "assumed cost"
            ),
            "selection_rule": (
                "freeze top-N, liquidity matching and ATR-band controls using all "
                "recorded candidates before reading outcomes; never replace missing "
                "selected or baseline members; comparisons have separate date cohorts"
            ),
            "missing_outcome_rule": (
                "unobserved selected or required control excludes that date-comparison; "
                "a missing metric excludes both sides of that metric-date, not members"
            ),
            "count_rule": (
                "legacy n_rows/n_dates count labeled observations; recorded_* and "
                "outcome_coverage retain unobserved candidates and all-halted dates; "
                "probability and complete_path_only diagnostics remain labeled-only"
            ),
            "model_version_rule": (
                "recorded model/rule/source identities are reported, not filtered; "
                "mixed versions and unknown identities do not establish new forward "
                "evidence or promotion approval"
            ),
            "forward_provenance_rule": (
                "default channel statistics use provenance_cohort=forward_observed "
                "only; scheduled replay and unknown provenance are diagnostic-only"
            ),
        },
        "artifacts": artifact_audit,
        "channels": {
            f"{slot}:{ranking}": _evaluate_channel(
                rows,
                top_ns=top_ns,
                n_boot=n_boot,
                seed=seed,
                min_rows=min_rows,
                min_days=min_days,
            )
            for (slot, ranking), rows in sorted(channels.items())
        },
    }
    report["report_payload_sha256"] = _report_digest(report)
    output = Path(output_path)
    _atomic_write_json(output, _normalise(report))
    result = dict(report)
    result["output_path"] = str(output)
    return result


def _parse_top_ns(value: str) -> tuple[int, ...]:
    try:
        values = tuple(
            sorted({int(v.strip()) for v in value.split(",") if v.strip()})
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "top-N은 양의 정수 comma list여야 합니다"
        ) from exc
    if not values or any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError("top-N은 양의 정수 comma list여야 합니다")
    return values


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("양의 정수여야 합니다") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("양의 정수여야 합니다")
    return parsed


def _bootstrap_count(value: str) -> int:
    parsed = _positive_int(value)
    if parsed > MAX_BOOTSTRAPS:
        raise argparse.ArgumentTypeError(
            f"bootstrap 횟수는 {MAX_BOOTSTRAPS} 이하여야 합니다"
        )
    return parsed


def _seed(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seed는 0 이상의 정수여야 합니다") from exc
    if not 0 <= parsed < 2**64:
        raise argparse.ArgumentTypeError("seed는 [0, 2**64) 정수여야 합니다")
    return parsed


def _date(value: str) -> calendar_date:
    try:
        return calendar_date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "날짜는 YYYY-MM-DD 형식이어야 합니다"
        ) from exc


def main() -> int:
    ap = argparse.ArgumentParser(
        description="전 유니버스 recommend score-label forward 감사 평가"
    )
    ap.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--top-n", type=_parse_top_ns, default=DEFAULT_TOP_NS)
    ap.add_argument("--n-boot", type=_bootstrap_count, default=DEFAULT_BOOTSTRAPS)
    ap.add_argument("--seed", type=_seed, default=DEFAULT_SEED)
    ap.add_argument("--min-rows", type=_positive_int, default=MIN_METRIC_ROWS)
    ap.add_argument("--min-days", type=_positive_int, default=MIN_CLUSTER_DAYS)
    ap.add_argument(
        "--through-date",
        type=_date,
        default=None,
        help="KST 기준 이 날짜까지의 완결 artifact만 평가(default=어제)",
    )
    args = ap.parse_args()
    report = evaluate_label_root(
        args.input_root,
        output_path=args.output,
        top_ns=args.top_n,
        n_boot=args.n_boot,
        seed=args.seed,
        min_rows=args.min_rows,
        min_days=args.min_days,
        through_date=args.through_date,
    )
    print(json.dumps({
        "output": report["output_path"],
        "channels": list(report["channels"]),
        "complete_artifacts": report["artifacts"]["complete_used"],
        "skipped_artifacts": len(report["artifacts"]["skipped"]),
        "report_payload_sha256": report["report_payload_sha256"],
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

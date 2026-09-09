"""Read-only, date-paired decomposition of the two existing R1 alert slots.

P uses preopen picks/preopen outcomes; Q uses those SAME picks/open outcomes;
R uses open picks/open outcomes. There is deliberately no future-open-picks /
preopen-outcomes cell. This arithmetic is not a causal freshness experiment.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, timedelta
from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd

from notifier.delivery_receipt import _LIVE_SEND_WINDOWS, _SERVER_CLOCK_SKEW_SECONDS
from ops.artifact_provenance import canonical_json_bytes, file_set_identity, resolve_identity_path, sha256_bytes
from ops.recommendation_evidence import load_recommendation_evidence
from signals.recommend_experiment_eval import METRICS, _metric_rows, _validate_outcomes

SLOT_CONFIG = {
    "version": "recommend_slot_diagnostics.v1",
    "legs": {"P": "actual_preopen_picks_preopen_window",
             "Q": "same_preopen_picks_open_window", "R": "actual_open_picks_open_window"},
    "components": {"window": "Q-P", "selection": "R-Q", "total": "R-P"},
    "metrics": list(METRICS), "picks_per_leg": 3, "date_weight": "equal",
    "missing_policy": "fix actual picks first; exclude entire date if any leg is unavailable",
    "ratio_floor": 0.001, "n_boot": 1000, "seed": 42, "minimum_ci_dates_initial": 5,
    "bootstrap": ["iid_date", "noncircular_block3_observed_dates"],
    "is_untouched_holdout": False, "deployable": False,
}
PARTS = ("P", "Q", "R", "window", "selection", "total")
_TIMES = ("decision_started_at", "decision_completed_at", "attempted_at", "sent_at", "recorded_at",
          "execution_start_at", "outcome_end_at", "label_created_at", "label_available_at")
_REQUIRED = {"date", "slot", "coin", "snapshot_id", "rank", "was_delivered", "label_available",
             "label_status", "decision_started_at", "outcome_end_at", "label_available_at",
             "p_up10", "p_dn5", "p_dn10", "exp_downside", "rr_ratio", "mfe", "mae",
             "up10", "dn5", "eod_return_net", "tp5_sl3_return_net"}


class SlotDiagnosticError(ValueError):
    """Corrupt evidence or impossible chronology blocks the complete report."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise SlotDiagnosticError(reason)


def _digest(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _same(actual: object, expected: object, reason: str) -> None:
    _require(canonical_json_bytes(actual) == canonical_json_bytes(expected), reason)


def _time(value: object, field: str) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise SlotDiagnosticError(f"invalid {field}") from exc
    _require(not pd.isna(stamp) and stamp.tzinfo is not None, f"{field} must be timezone-aware")
    return stamp.tz_convert("UTC")


def _day(value: object) -> date:
    _require(isinstance(value, str), "invalid ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise SlotDiagnosticError("invalid ISO date") from exc
    _require(parsed.isoformat() == value, "noncanonical ISO date")
    return parsed


def _identities(rows: list[dict]) -> list[dict]:
    return sorted(({"coin": row["coin"], "rank": int(row["rank"])} for row in rows),
                  key=lambda row: row["rank"])


def load_slot_metadata(dataset: dict) -> dict[str, dict]:
    """Read only the exact manifests accepted by ``load_experiment_data``.

    The global DB timestamp and base-training summary are retained as recorded,
    never promoted to per-coin freshness or final-head training statistics.
    """
    provenance = dataset["provenance"]
    root = Path(provenance["root"])
    identities = provenance["files"]
    paths = {key: resolve_identity_path(item["path"], root=root) for key, item in identities.items()}
    _same(file_set_identity(paths, root=root), identities, "frozen inputs changed before slot metadata read")
    cutoff = _time(dataset["as_of"], "as_of").to_pydatetime()
    result = {}
    for manifest in provenance["input_manifests"]:
        files = manifest["files"]
        bound = {key: resolve_identity_path(item["path"], root=root) for key, item in files.items()}
        evidence = load_recommendation_evidence(
            bound["snapshot"], label_root=bound["label"].parent.parent,
            receipt_root=bound["receipt"].parent.parent, now=cutoff,
        )
        _same(evidence["manifest"], manifest, "slot metadata evidence manifest mismatch")
        snapshot, label, receipt = (evidence[key] for key in ("snapshot", "label", "receipt"))
        sid = snapshot["snapshot_id"]
        _require(sid not in result, "duplicate metadata snapshot ID")
        feature_day = _day(snapshot["feature_asof"])
        result[sid] = {
            "snapshot_id": sid, "date": snapshot["asof"], "slot": snapshot["slot"],
            "model_id": snapshot["model"]["id"],
            "model_fit_mode": snapshot["model"]["fit_mode"],
            "score_source_sha256": snapshot["code"]["score_source_sha256"],
            "feature_row_date": feature_day.isoformat(),
            "nominal_shift1_input_bar_date": (feature_day - timedelta(days=1)).isoformat(),
            "actual_latest_input_timestamp": None,
            "actual_input_freshness": "unknown_per_coin_timestamp_not_recorded",
            "db_max_timestamp_raw": snapshot["data"]["max_timestamp"],
            "db_manifest_id": snapshot["data"]["manifest_id"],
            "db_timestamp_scope": "global_database_bar_timestamp_not_per_coin_input",
            "training_cutoff_exclusive": snapshot["training"]["cutoff_exclusive"],
            **{f"training_base_{key}": snapshot["training"][key] for key in ("start", "end", "rows", "dates")},
            "embargo_days": snapshot["training"]["embargo_days"],
            "actual_head_train_end": None, "actual_head_train_rows": None,
            "training_summary_scope": "base train before R1 head drops missing high/low labels",
            "training_code_evidence": {
                "base_train": "signals/recommend.py: cutoff=feature_date-embargo; date<cutoff and in_universe",
                "head_train": "signals/recommend.py: _rr_outcome_labels(train).dropna(high,low); _fit_rr_head(train_lab)",
                "stored_summary": "signals/recommend.py: return.training summarizes train, not train_lab",
                "scope": "current source interpretation; no historical head artifact or exact head sample recovered",
            },
            **{key: snapshot[key] for key in ("decision_started_at", "decision_completed_at")},
            **{key: receipt[key] for key in ("attempted_at", "sent_at", "recorded_at")},
            "sent_at_basis": "last_telegram_server_acceptance_not_user_read_or_fill",
            "execution_start_at": label["execution_start_at"], "outcome_end_at": label["path_window_end"],
            "label_created_at": label["labeled_at"], "label_available_at": manifest["label_available_at"],
            "execution_time_basis": label["execution_time_basis"],
            "reference_price_note": "snapshot entry_open is only a 09:00 reference, not an executable fill",
            "round_trip_cost_fraction": label["round_trip_cost_fraction"],
            "top3": _identities(snapshot["top3"]), "universe_n": len(snapshot["universe"]),
            "universe_identity_sha256": _digest(_identities(snapshot["universe"])),
            "evidence_manifest": deepcopy(manifest),
        }
    _same(file_set_identity(paths, root=root), identities, "frozen inputs changed during slot metadata read")
    _require(set(result) == set(dataset["frame"]["snapshot_id"]), "metadata and frame snapshot IDs differ")
    return result


def _validate_metadata(meta: dict, sid: str) -> dict[str, pd.Timestamp]:
    _require(meta["snapshot_id"] == sid and meta["slot"] in ("preopen", "open"), "metadata identity mismatch")
    day = _day(meta["date"])
    feature_day = day - timedelta(days=meta["slot"] == "preopen")
    _require(_day(meta["feature_row_date"]) == feature_day, "nominal feature row date mismatch")
    _require(_day(meta["nominal_shift1_input_bar_date"]) == feature_day - timedelta(days=1),
             "nominal input bar date mismatch")
    _require(meta["actual_latest_input_timestamp"] is None and meta["actual_head_train_end"] is None
             and meta["actual_head_train_rows"] is None, "unrecorded input/head evidence must stay unknown")
    cutoff = _day(meta["training_cutoff_exclusive"])
    _require(_day(meta["training_base_start"]) <= _day(meta["training_base_end"]) < cutoff
             and (feature_day - cutoff).days == meta["embargo_days"], "base training chronology mismatch")
    stamps = {key: _time(meta[key], key) for key in _TIMES}
    start, completed, attempted, sent, recorded = (stamps[key] for key in _TIMES[:5])
    live_start, live_end = (pd.Timestamp(datetime.combine(day, wall), tz="Asia/Seoul")
                            for wall in _LIVE_SEND_WINDOWS[meta["slot"]])
    _require(live_start <= start <= completed <= attempted < live_end, "decision chronology or slot window mismatch")
    _require(attempted <= recorded and attempted.floor("s") <= sent
             <= recorded + pd.Timedelta(seconds=_SERVER_CLOCK_SKEW_SECONDS)
             and live_start <= sent < live_end, "delivery chronology or slot window mismatch")
    scheduled_open = pd.Timestamp(day, tz="Asia/Seoul") + pd.Timedelta(hours=9)
    expected_entry = max(scheduled_open, sent.floor("15min") + pd.Timedelta(minutes=15))
    _require(stamps["execution_start_at"] == expected_entry, "canonical entry time mismatch")
    _require(stamps["outcome_end_at"] == expected_entry + pd.Timedelta(days=1), "outcome horizon must be 24 hours")
    _require(stamps["label_created_at"] >= stamps["outcome_end_at"]
             and stamps["label_available_at"] == max(stamps["label_created_at"], stamps["outcome_end_at"]),
             "label availability chronology mismatch")
    _require(meta["execution_time_basis"] == "delivery_sent_at", "unconfirmed execution basis")
    _same(meta["round_trip_cost_fraction"], 0.0015, "net cost contract mismatch")
    return stamps


def _validate(frame: pd.DataFrame, metadata: dict) -> tuple[pd.DataFrame, dict]:
    _require(isinstance(frame, pd.DataFrame) and frame.columns.is_unique, "invalid frame or duplicate columns")
    _require(_REQUIRED <= set(frame.columns), f"missing columns: {sorted(_REQUIRED - set(frame.columns))}")
    work = frame.copy(deep=True).sort_values(["date", "slot", "rank"], kind="stable").reset_index(drop=True)
    for name in ("date", "slot", "coin", "snapshot_id"):
        _require(work[name].map(lambda value: isinstance(value, str) and bool(value)).all(), f"invalid {name}")
    _require(work["slot"].isin(["preopen", "open"]).all(), "invalid slot")
    _require(not work.duplicated(["date", "slot", "coin"]).any(), "duplicate date/slot/coin")
    _require(set(metadata) == set(work["snapshot_id"]), "missing or extra snapshot metadata")
    for name in ("was_delivered", "label_available"):
        _require(work[name].map(lambda value: isinstance(value, (bool, np.bool_))).all(), f"{name} must be boolean")
    for name in ("rank", "p_up10", "p_dn5", "p_dn10", "exp_downside", "rr_ratio"):
        _require(work[name].map(lambda value: isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
                               and bool(np.isfinite(value))).all(), f"invalid numeric predictor {name}")
    for name in ("p_up10", "p_dn5", "p_dn10"):
        _require(work[name].between(0, 1).all(), f"probability out of range: {name}")
    _require(work["rr_ratio"].ge(0).all(), "negative ranking ratio")
    stamps = {}
    for (day, slot), group in work.groupby(["date", "slot"], sort=True):
        _day(day)
        _require(group["snapshot_id"].nunique() == 1, "mixed snapshots within date/slot")
        sid = group["snapshot_id"].iloc[0]
        meta = metadata[sid]
        _require(meta["date"] == day and meta["slot"] == slot, "frame metadata date/slot mismatch")
        stamps[sid] = _validate_metadata(meta, sid)
        _require(sorted(group["rank"]) == list(range(1, len(group) + 1)), "candidate ranks must be complete integers")
        identities = _identities(group[["coin", "rank"]].to_dict("records"))
        _require(meta["universe_n"] == len(group) and meta["universe_identity_sha256"] == _digest(identities),
                 "frame universe identity mismatch")
        actual = group.loc[group["was_delivered"]].sort_values("rank")
        _same(_identities(actual[["coin", "rank"]].to_dict("records")), meta["top3"], "actual Top3 metadata mismatch")
        _require(actual["rank"].tolist() == list(range(1, min(3, len(group)) + 1)), "actual picks differ from stored Top3")
        ranked = sorted(group.index, key=lambda i: (
            -work.at[i, "p_up10"] / max(work.at[i, "p_dn5"], SLOT_CONFIG["ratio_floor"]),
            work.at[i, "p_dn10"], -work.at[i, "p_up10"], -work.at[i, "exp_downside"], work.at[i, "rank"],
        ))[:3]
        _require(ranked == actual.index.tolist(), f"identity Top3 parity failed: {day}/{slot}")
        for key in ("decision_started_at", "outcome_end_at", "label_available_at"):
            _require(all(_time(value, key) == stamps[sid][key] for value in group[key]),
                     f"frame metadata {key} mismatch")
    _require(work.groupby("snapshot_id")[["date", "slot"]].nunique().le(1).all().all(), "snapshot reused across slots/dates")
    return work, stamps


def _named(values: np.ndarray) -> dict:
    return {part: {metric: float(values[i, j]) for j, metric in enumerate(METRICS)}
            for i, part in enumerate(PARTS)}


def _parts(legs: np.ndarray) -> np.ndarray:
    p, q, r = (legs[..., i, :] for i in range(3))
    return np.stack((p, q, r, q - p, r - q, r - p), axis=-2)


def _additivity_error(values: np.ndarray) -> float:
    error = float(np.max(np.abs(values[..., 3, :] + values[..., 4, :] - values[..., 5, :]))) if values.size else 0.0
    _require(error <= 1e-12, "window + selection != total")
    return error


def _joint_summary(values: np.ndarray, n_boot: int, seed: int) -> dict:
    n = len(values)
    result = {"dates": n, "means": _named(values.mean(axis=0)) if n else None,
              "daily_additivity_max_error": _additivity_error(values),
              "mean_additivity_max_error": _additivity_error(values.mean(axis=0)) if n else 0.0,
              "ci_status": "available" if n >= SLOT_CONFIG["minimum_ci_dates_initial"] else "insufficient_dates",
              "n_boot": n_boot, "seed": seed, "bootstrap": {}}
    if n < SLOT_CONFIG["minimum_ci_dates_initial"]:
        return result
    rng = np.random.default_rng(seed)
    iid = rng.integers(0, n, size=(n_boot, n))
    starts = rng.integers(0, n - 3 + 1, size=(n_boot, int(np.ceil(n / 3))))
    blocks = (starts[:, :, None] + np.arange(3)).reshape(n_boot, -1)[:, :n]
    for name, indices in zip(SLOT_CONFIG["bootstrap"], (iid, blocks), strict=True):
        # ONE date-index matrix drives all three legs AND every component.
        samples = values[indices].mean(axis=1)
        error = _additivity_error(samples)
        low, high = np.quantile(samples, [0.025, 0.975], axis=0)
        result["bootstrap"][name] = {
            "ci95": {part: {metric: [float(low[i, j]), float(high[i, j])]
                             for j, metric in enumerate(METRICS)} for i, part in enumerate(PARTS)},
            "joint_indices_sha256": _digest(indices.tolist()), "replicates": n_boot,
            "replicate_additivity_max_error": error,
            "interval_note": "percentiles of joint replicates; component CI endpoints are not additive",
        }
    return result


def _rank_changes(pre: pd.DataFrame, opened: pd.DataFrame) -> dict:
    shared = sorted(set(pre["coin"]) & set(opened["coin"]))
    left, right = pre.set_index("coin"), opened.set_index("coin")
    percentile = {slot: {name: rows[name].rank(method="average", pct=True)
                         for name in ("p_up10", "p_dn5")}
                  for slot, rows in (("preopen", left), ("open", right))}
    selected = set(pre.loc[pre["was_delivered"], "coin"]) | set(opened.loc[opened["was_delivered"], "coin"])
    changes = [{"coin": coin, "preopen_rank": int(left.at[coin, "rank"]),
                "open_rank": int(right.at[coin, "rank"]),
                "rank_change": int(right.at[coin, "rank"] - left.at[coin, "rank"]),
                **{f"{name}_percentile_change": float(percentile["open"][name].at[coin]
                                                     - percentile["preopen"][name].at[coin])
                   for name in ("p_up10", "p_dn5")}} for coin in shared]
    return {"common_candidates": len(shared),
            "top3_overlap_coins": sorted(set(pre.loc[pre["was_delivered"], "coin"])
                                         & set(opened.loc[opened["was_delivered"], "coin"])),
            "mean_absolute_rank_change": float(np.mean([abs(row["rank_change"]) for row in changes]))
            if changes else None,
            "mean_absolute_percentile_change": {
                name: float(np.mean([abs(row[f"{name}_percentile_change"]) for row in changes])) if changes else None
                for name in ("p_up10", "p_dn5")},
            "selected_union_common_coin_changes": [row for row in changes if row["coin"] in selected],
            "percentile_basis": "each original full slot universe; higher dn5 percentile means greater predicted downside"}


def analyze_slots(frame: pd.DataFrame, metadata_by_sid: dict, *, n_boot: int = 1000, seed: int = 42) -> dict:
    """Pure three-leg diagnostic; fix actual selections before examining labels."""
    _require(type(n_boot) is int and n_boot > 0 and type(seed) is int and seed >= 0, "invalid bootstrap arguments")
    work, stamps = _validate(frame, metadata_by_sid)
    plans, exclusions, overlap = [], [], []
    for day, rows in work.groupby("date", sort=True):
        groups = {slot: group for slot, group in rows.groupby("slot", sort=True)}
        if set(groups) != {"preopen", "open"}:
            exclusions.append({"date": day, "reason": "missing_slot", "missing_slots": sorted({"preopen", "open"} - set(groups)), "coins": []})
            continue
        pre, opened = groups["preopen"], groups["open"]
        p, r = pre.loc[pre["was_delivered"]].index.tolist(), opened.loc[opened["was_delivered"]].index.tolist()
        overlap.append({"date": day, **_rank_changes(pre, opened)})
        if len(p) != 3 or len(r) != 3:
            exclusions.append({"date": day, "reason": "unsupported_actual_pick_count", "P_count": len(p), "R_count": len(r), "coins": []})
            continue
        p_coins = work.loc[p, "coin"].tolist()
        open_by_coin = dict(zip(opened["coin"], opened.index, strict=True))
        missing = sorted(set(p_coins) - set(open_by_coin))
        if missing:
            exclusions.append({"date": day, "reason": "preopen_pick_missing_from_open_universe", "coins": missing})
            continue
        pre_sid, open_sid = pre["snapshot_id"].iloc[0], opened["snapshot_id"].iloc[0]
        _require(stamps[pre_sid]["sent_at"] < stamps[open_sid]["sent_at"]
                 and stamps[pre_sid]["execution_start_at"] <= stamps[open_sid]["execution_start_at"],
                 "cross-slot time reversal")
        plans.append({"date": day, "P": p, "Q": [open_by_coin[coin] for coin in p_coins], "R": r,
                      "preopen_snapshot_id": pre_sid, "open_snapshot_id": open_sid})
    # Corrupt labels anywhere remain a hard failure; legal unavailable labels
    # are inspected only AFTER all original selections and joins are frozen.
    _validate_outcomes(work)
    _require((work["label_available"] == work["label_status"].eq("labeled")).all(), "label availability/status mismatch")
    included, daily, values = [], [], []
    for plan in plans:
        unavailable = [{"leg": leg, "coin": work.at[i, "coin"], "slot": work.at[i, "slot"],
                        "label_status": work.at[i, "label_status"]}
                       for leg in ("P", "Q", "R") for i in plan[leg] if not work.at[i, "label_available"]]
        if unavailable:
            exclusions.append({"date": plan["date"], "reason": "selected_label_unavailable", "coins": sorted({row["coin"] for row in unavailable}),
                               "unavailable": unavailable})
            continue
        result = _parts(np.stack([_metric_rows(work, plan[leg]).mean(axis=0) for leg in ("P", "Q", "R")]))
        _additivity_error(result)
        values.append(result)
        membership = {key: plan[key] for key in ("date", "preopen_snapshot_id", "open_snapshot_id")}
        membership.update({leg: work.loc[plan[leg], "coin"].tolist() for leg in ("P", "Q", "R")})
        included.append(membership)
        pre_sid, open_sid = plan["preopen_snapshot_id"], plan["open_snapshot_id"]
        pre_meta, open_meta = metadata_by_sid[pre_sid], metadata_by_sid[open_sid]
        daily.append({**membership, "metrics": _named(result),
                      "timing": {"sent_gap_seconds": (stamps[open_sid]["sent_at"] - stamps[pre_sid]["sent_at"]).total_seconds(),
                                 "entry_gap_seconds": (stamps[open_sid]["execution_start_at"] - stamps[pre_sid]["execution_start_at"]).total_seconds(),
                                 "feature_row_gap_days": (_day(open_meta["feature_row_date"]) - _day(pre_meta["feature_row_date"])).days,
                                 "training_base_end_gap_days": (_day(open_meta["training_base_end"]) - _day(pre_meta["training_base_end"])).days}})
    matrix = np.stack(values) if values else np.empty((0, len(PARTS), len(METRICS)))
    cohort_hash = _digest(included)
    result = {
        "schema": SLOT_CONFIG["version"], "config": deepcopy(SLOT_CONFIG), "scope": "offline_diagnostic_only",
        "status": "historical_diagnostic" if included else "no_common_evaluable_dates",
        "deployable": False, "is_untouched_holdout": False,
        "cohort": {"cohort_id": f"slot3leg-{cohort_hash[:20]}", "cohort_sha256": cohort_hash,
                   "dates": len(included), "date_list": [row["date"] for row in included],
                   "picks_per_leg": 3, "pick_rows_per_leg": 3 * len(included), "membership": included},
        "coverage": {"input_dates": int(work["date"].nunique()), "input_rows": len(work),
                     "paired_dates": len(overlap), "included_dates": len(included),
                     "excluded_dates": len(exclusions), "excluded": sorted(exclusions, key=lambda row: row["date"])},
        "summary": _joint_summary(matrix, n_boot, seed), "daily": daily,
        "leave_one_date_out": [{"omitted_date": included[i]["date"], "remaining_dates": len(matrix) - 1,
                                "means": _named(np.delete(matrix, i, axis=0).mean(axis=0))}
                               for i in range(len(matrix))] if len(matrix) > 1 else [],
        "common_candidate_diagnostics": overlap,
        "decision_metadata": [deepcopy(metadata_by_sid[sid]) for sid in sorted(metadata_by_sid)],
        "interpretation": [
            "Window/selection arithmetic, not a causal effect of input freshness or a tradable counterfactual.",
            "Q waits to the observed open execution window; future open picks are never evaluated in the preopen window.",
            "Every leg uses the same retained dates and three equal-weight picks; legitimate missing evidence excludes the whole date.",
            "Exclusions define a conditional complete-case cohort, not unbiased unconditional performance; it can differ from selection diagnostics.",
            "Returns are existing cost-adjusted pick proxies, not actual user fills or portfolio PnL; no extra fee deduction.",
            "Nominal input dates, global DB timestamps, training cutoff, universe, and window all differ; per-coin freshness and exact head samples are unknown.",
            "Bootstrap intervals use observed-date clusters, not independent coins; block3 does not wrap or imply consecutive calendar days.",
        ],
    }
    canonical_json_bytes(result)
    return result

"""Read-only evaluation of the separately launched Top10 flow challenger.

Publication and selection belong to the frozen trial reader. This module only
joins already committed choices to canonical outcomes; it never repairs a
missing prediction, trains a model, or changes a live recommendation.
"""

from __future__ import annotations

from collections import Counter
from datetime import date, time, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from notifier.delivery_receipt import DEFAULT_RECEIPT_ROOT
from ops.artifact_provenance import file_identity
from ops.recommendation_evidence import (
    EvidenceUnavailable,
    load_recommendation_evidence,
)
from signals.recommend_experiment_eval import (
    METRICS,
    _metric_rows,
    _summary,
    _validate_outcomes,
)
from signals.recommend_microstructure_trial import (
    ROOT,
    _cohort_summary,
    _directory,
    _now,
    _path,
    _remember_inputs,
    _require,
    _seal,
    _time,
    write_new_trial_report,
)
from signals.recommend_trade_shortlist_trial import (
    DEFAULT_TRIAL_ROOT,
    GENERATOR_SOURCES as PUBLICATION_SOURCES,
    TRIAL_CONFIG,
    TRIAL_ID,
    read_trade_shortlist_record,
)

EVALUATION_SCHEMA = "recommend_trade_shortlist_evaluation.v1"
GENERATOR_SOURCES = tuple(
    dict.fromkeys(
        (
            *PUBLICATION_SOURCES,
            "signals/recommend_trade_shortlist_eval.py",
            "scripts/evaluate_recommend_trade_shortlist_trial.py",
            "signals/recommend_experiment_eval.py",
            "ops/recommendation_evidence.py",
            "signals/recommend_score_labels.py",
            "notifier/delivery_receipt.py",
            "ledger/path_quality.py",
            "ledger/config.py",
        )
    )
)


def _generator_sources():
    result = [file_identity(ROOT / path, root=ROOT) for path in GENERATOR_SOURCES]
    _require(
        all(item.get("exists") is True for item in result), "evaluation source missing"
    )
    return result


def _days(root):
    """No creation: pin directories, including when the trial root is absent."""
    with _directory(root, create=False) as directory:
        if directory is None:
            return []
        result = []
        for path in sorted(directory[0].iterdir()):
            try:
                day = date.fromisoformat(path.name)
            except ValueError:
                continue
            _require(
                path.name == day.isoformat()
                and not path.is_symlink()
                and path.is_dir(),
                "invalid trial date directory",
            )
            result.append(day.isoformat())
        return result


def _context(plan):
    """Stored pre-outcome context only, never a matching or eligibility rule."""
    rows = {row["coin"]: row for row in plan["candidate_inputs"]}
    result = {"source": "frozen_predecision_context_not_matching_or_risk_control"}
    groups = {
        "control": plan["control_top3"],
        "challenger": plan["challenger_top3"],
        "shortlist": plan["shortlist"],
    }
    fields = (
        "stored_f_atr_pct_14",
        "stored_f_log_qv",
        "observed_trade_count",
        "total_notional",
    )
    for group, coins in groups.items():
        if coins is None:
            result[group] = None
            continue
        summary = {"n_candidates": len(coins)}
        for field in fields:
            values = [rows[coin][field] for coin in coins]
            present = [value for value in values if value is not None]
            summary[field] = {
                "observed": len(present),
                "missing": len(values) - len(present),
                "mean": float(np.mean(present))
                if len(present) == len(values)
                else None,
            }
        summary["saturated_count"] = sum(rows[coin]["saturated"] for coin in coins)
        summary["single_trade_count"] = sum(
            rows[coin]["single_trade"] for coin in coins
        )
        result[group] = summary
    return result


def evaluate_trade_shortlist_trials(
    trial_root=DEFAULT_TRIAL_ROOT,
    *,
    label_root=ROOT / "output/recommend_score_labels",
    receipt_root=DEFAULT_RECEIPT_ROOT,
    now=None,
    n_boot=1000,
    seed=42,
):
    """Evaluate published Top3 pairs; no-op days count, unavailable days do not."""
    now = _time(_now() if now is None else now)
    _require(type(n_boot) is int and 1 <= n_boot <= 100_000, "invalid n_boot")
    _require(type(seed) is int and 0 <= seed < 2**64, "invalid seed")
    sources_before = _generator_sources()
    root = Path(trial_root)
    days = _days(root)
    local_now = now.astimezone(timezone(timedelta(hours=9)))
    launch = date.fromisoformat(TRIAL_CONFIG["prospective_start_asof"])
    _require(
        all(day <= local_now.date().isoformat() for day in days),
        "future trial directory",
    )
    expected = [
        launch + timedelta(days=i)
        for i in range(max(0, (local_now.date() - launch).days + 1))
    ]
    due = [
        day.isoformat()
        for day in expected
        if day < local_now.date()
        or local_now.time()
        >= time.fromisoformat(TRIAL_CONFIG["scheduled_capture_start_kst"])
    ]
    not_due = [day.isoformat() for day in expected if day.isoformat() not in due]
    audits, paired, full, trial_inputs, evidence_inputs = [], [], [], {}, {}
    for day in sorted(set(days) | set(due) | set(not_due)):
        record = read_trade_shortlist_record(root, day, now)
        audit = {
            "date": day,
            "record_status": record["status"],
            "status": record["status"],
            "reason": record["reason"],
        }
        audits.append(audit)
        _remember_inputs(trial_inputs, record.get("trial_artifacts", []))
        if record.get("feature") is not None:
            _remember_inputs(evidence_inputs, record["score"]["source_inputs"])
            _remember_inputs(evidence_inputs, record["feature"]["provenance"]["files"])
        if day in not_due and record["status"] == "missing":
            audit.update(status="not_due", reason="scheduled_capture_has_not_started")
            continue
        if record["status"] != "committed":
            continue
        score, commit, plan = record["score"], record["commit"], record["score"]["plan"]
        audit.update(
            snapshot_id=score["snapshot_id"],
            plan_status=plan["status"],
            control_top3=plan["control_top3"],
            challenger_top3=plan["challenger_top3"],
            changed_picks=plan["changed_picks"],
            is_no_op=plan["is_no_op"],
            feature_coverage=plan["coverage"],
            stored_context=_context(plan),
            score_durable_observed_at=commit["score_durable_observed_at"],
        )
        if day < launch.isoformat():
            audit.update(
                status="prelaunch_replay",
                reason="before_frozen_prospective_launch_date",
            )
            continue
        if plan["status"] != "planned":
            audit.update(status="unavailable", reason=plan["reason"])
            continue
        # Coin membership is frozen before looking at a label or its availability.
        choices = {arm: tuple(plan[f"{arm}_top3"]) for arm in ("control", "challenger")}
        try:
            evidence = load_recommendation_evidence(
                _path(score["source_inputs"][0]["path"]),
                label_root=label_root,
                receipt_root=receipt_root,
                now=now,
            )
        except EvidenceUnavailable as exc:
            reason = str(exc)
            pending = reason in {
                "label_missing",
                "receipt_missing_delivery_unknown",
                "label_not_available_as_of_now",
            } or reason.startswith("label_status=")
            audit.update(status="pending" if pending else "unavailable", reason=reason)
            continue
        _require(
            evidence["snapshot"]["snapshot_id"] == score["snapshot_id"],
            "evaluation snapshot changed",
        )
        _require(
            [row["coin"] for row in evidence["snapshot"]["top3"]]
            == list(choices["control"]),
            "control is not original delivered Top3",
        )
        _remember_inputs(evidence_inputs, evidence["manifest"]["files"].values())
        audit["evidence_manifest"] = evidence["manifest"]
        entry = _time(evidence["label"]["execution_start_at"])
        audit["canonical_execution_start_at"] = entry.isoformat()
        if _time(commit["score_durable_observed_at"]) >= entry:
            audit.update(
                status="late", reason="score_not_durable_before_canonical_entry"
            )
            continue
        frame = (
            pd.DataFrame(evidence["label"]["rows"])
            .sort_values("coin")
            .reset_index(drop=True)
        )
        _require(
            len(frame) == 100
            and frame["coin"].is_unique
            and set(frame["coin"]) == set(plan["control_ranking"]),
            "canonical candidate set changed",
        )
        indices = {coin: i for i, coin in enumerate(frame["coin"])}
        chosen = {
            arm: [indices[coin] for coin in coins] for arm, coins in choices.items()
        }
        _validate_outcomes(frame)
        needed = set(choices["control"]) | set(choices["challenger"])
        missing = sorted(
            coin
            for coin in needed
            if frame.at[indices[coin], "label_status"] != "labeled"
        )
        if missing:
            audit.update(
                status="unavailable",
                reason="selected_outcome_unavailable",
                missing_selected=missing,
            )
            continue
        row = {
            "date": day,
            "changed_picks": plan["changed_picks"],
            **{
                arm: _metric_rows(frame, picks).mean(axis=0).tolist()
                for arm, picks in chosen.items()
            },
        }
        paired.append(row)
        audit.update(status="prospective_comparable", reason=None)
        if frame["label_status"].eq("labeled").all():
            full.append(
                {
                    **row,
                    "baseline": _metric_rows(frame, frame.index.tolist())
                    .mean(axis=0)
                    .tolist(),
                }
            )
            audit["full_universe_baseline_status"] = "available"
        else:
            audit.update(
                full_universe_baseline_status="unavailable",
                full_universe_missing=sorted(
                    frame.loc[frame["label_status"] != "labeled", "coin"]
                ),
            )
    primary = _cohort_summary(paired, n_boot=n_boot, seed=seed)
    baseline = _cohort_summary(full, n_boot=n_boot, seed=seed)
    if full:
        values = np.asarray([row["baseline"] for row in full])
        baseline.update(
            full_universe=_summary(values, n_boot, seed),
            baseline_candidates_per_date=100,
        )
        for arm in ("control", "challenger"):
            baseline[f"{arm}_minus_full_universe"] = _summary(
                np.asarray([row[arm] for row in full]) - values, n_boot, seed
            )
    _require(sources_before == _generator_sources(), "evaluation generator changed")
    _require(days == _days(root), "trial dates changed during evaluation")
    for items, name in (
        (trial_inputs, "trial"),
        (evidence_inputs, "feature or canonical evidence"),
    ):
        _require(
            list(items.values())
            == [
                file_identity(_path(item["path"]), root=ROOT) for item in items.values()
            ],
            f"{name} inputs changed during evaluation",
        )
    missing_dates = [
        row["date"]
        for row in audits
        if row["date"] in due and row["status"] == "missing"
    ]
    return _seal(
        {
            "schema": EVALUATION_SCHEMA,
            "trial_id": TRIAL_ID,
            "config": TRIAL_CONFIG,
            "generated_at": now.isoformat(),
            "generator_sources": sources_before,
            "trial_inputs": list(trial_inputs.values()),
            "evidence_inputs": list(evidence_inputs.values()),
            "inputs_unchanged": True,
            "effect_status": primary["effect_status"],
            "status": (
                "evaluated_descriptively"
                if paired
                else "not_started"
                if local_now.date() < launch
                else "no_comparable_prospective_dates"
            ),
            "deployable": False,
            "automatic_promotion": False,
            "n_boot": n_boot,
            "seed": seed,
            "metrics": list(METRICS),
            "new_hypotheses": 1,
            "microstructure_family_hypotheses": 2,
            "coverage": {
                "recorded_date_directories": len(days),
                "paired_dates": len(paired),
                "prospective_start_asof": launch.isoformat(),
                "calendar_expected_dates": due,
                "calendar_expected_count": len(due),
                "calendar_not_due_dates": not_due,
                "calendar_missing_dates": missing_dates,
                "calendar_missing_count": len(missing_dates),
                "states": dict(
                    sorted(Counter(row["status"] for row in audits).items())
                ),
            },
            "primary_paired": primary,
            "full_universe_common": baseline,
            "dates": audits,
            "methodology": {
                "primary": "same-date frozen original/control and challenger Top3; include no-op dates",
                "selection": "original ranks 1-10; all ten causal features required; never replace missing picks",
                "cost": "copy canonical cost-adjusted pick returns without a second subtraction",
                "uncertainty": "IID date and 3 observed-date blocks; fewer than 5 dates gives no CI",
                "units": "daily equally weighted pick-return proxies, not actual trades or portfolio PnL",
                "baseline": "all 100 original candidates, separate common-complete cohort; not a matched control",
                "confounding": "Stored ATR/liquidity are diagnostic only; Top10 does not protect realized downside or control confounding.",
                "limitations": "A single tiny trade may saturate imbalance; repeated monitoring and two hypotheses forbid an automatic adoption verdict.",
                "clock": "score durability observed strictly before canonical entry, not commit completion; late records are replay only",
                "missing": "Calendar absence is missing as of observation, not no-signal; unavailable is never a zero-return day",
                "provenance": "Full native raw/source verification retained; repeated hashing has unoptimized runtime cost",
                "scope": "No model fitting, label changes, live ranking replacement, Telegram, or automatic promotion",
            },
        }
    )


__all__ = ["evaluate_trade_shortlist_trials", "write_new_trial_report"]

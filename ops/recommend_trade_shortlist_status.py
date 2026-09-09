"""Read-only latest-due Top10 publication check; never joins outcomes or fits.

Unlike the lightweight original capture probe, this checks the single current
trial's native raw/source hashes. Heartbeat bounds it externally at 30 seconds.
It is not the cumulative performance evaluator or proof of pre-entry eligibility.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from ops.recommend_microstructure_status import _ReadSet
from signals import recommend_microstructure_trial as native
from signals import recommend_trade_shortlist_trial as trial

KST = ZoneInfo("Asia/Seoul")
DUE = time(10)  # Initial completion grace; heartbeat actually runs after 10:30.


def inspect_shortlist_status(
    *, now=None, trial_root=trial.DEFAULT_TRIAL_ROOT, _asof=None
):
    observed = native._time(native._now() if now is None else now).astimezone(KST)
    asof = observed.date() if _asof is None else _asof
    launch = date.fromisoformat(trial.TRIAL_CONFIG["prospective_start_asof"])
    report = {
        "schema": "recommend_trade_shortlist_operational_status.v1",
        "asof": asof.isoformat(),
        "checked_at": observed.isoformat(),
        "trial_id": trial.TRIAL_ID,
        "prospective_start_asof": launch.isoformat(),
        "scope": "current_trial_native_graph_and_evaluation_binding_not_outcomes",
        "effect_status": "not_evaluated",
        "prospective_status": "not_checked",
        "deployable": False,
    }
    if asof < launch:
        return {
            **report,
            "status": "not_started",
            "reason": "before new trial launch",
            "attention_required": False,
        }
    if asof == observed.date() and observed.time() < DUE:
        previous = inspect_shortlist_status(
            now=observed, trial_root=trial_root, _asof=asof - timedelta(days=1)
        )
        return {
            **report,
            "status": "waiting",
            "reason": "before completion deadline",
            "previous_due": previous,
            "attention_required": previous["attention_required"],
        }
    try:
        record = trial.read_trade_shortlist_record(trial_root, asof, observed)
        if record["status"] != "committed":
            return {
                **report,
                "status": record["status"],
                "reason": record["reason"],
                "attention_required": True,
            }
        feature_path = native._path(record["source_inputs"][1]["path"])
        reader = _ReadSet()
        evaluation = reader.document(
            feature_path.parent / "trade_shortlist_evaluation.json"
        )
        if evaluation is None:
            return {
                **report,
                "status": "recorded_evaluation_missing",
                "reason": "score committed but session evaluation missing",
                "attention_required": True,
            }
        trial._checked(evaluation, "recommend_trade_shortlist_evaluation.v1")
        from signals.recommend_trade_shortlist_eval import _generator_sources

        evaluation_sources = _generator_sources()
        trial._require(
            evaluation.get("config") == trial.TRIAL_CONFIG
            and evaluation.get("generator_sources") == evaluation_sources
            and evaluation.get("inputs_unchanged") is True
            and evaluation.get("deployable") is False
            and evaluation.get("automatic_promotion") is False,
            "evaluation contract mismatch",
        )
        generated = native._time(evaluation["generated_at"])
        trial._require(
            native._time(record["commit"]["score_durable_observed_at"])
            <= generated
            <= observed
            and generated.astimezone(KST).date() == asof,
            "evaluation clock mismatch",
        )
        for expected in record["trial_artifacts"]:
            matches = [
                item
                for item in evaluation["trial_inputs"]
                if native._path(item["path"]).resolve()
                == native._path(expected["path"]).resolve()
            ]
            trial._require(
                matches == [expected], "evaluation does not bind current trial bytes"
            )
        matches = [
            item for item in evaluation["dates"] if item.get("date") == asof.isoformat()
        ]
        trial._require(
            len(matches) == 1
            and matches[0].get("record_status") == "committed"
            and matches[0].get("snapshot_id") == record["score"]["snapshot_id"]
            and matches[0].get("plan_status") == record["plan"]["status"],
            "evaluation does not include current committed date",
        )
        audit, plan = matches[0], record["plan"]
        expected_audit = {
            key: plan[key]
            for key in ("control_top3", "challenger_top3", "changed_picks", "is_no_op")
        }
        expected_audit["feature_coverage"] = plan["coverage"]
        expected_audit["score_durable_observed_at"] = record["commit"][
            "score_durable_observed_at"
        ]
        trial._require(
            all(
                key in audit and audit[key] == value
                for key, value in expected_audit.items()
            ),
            "evaluation selection or durability metadata mismatch",
        )
        # Small score/commit identities must still be the reader's exact bytes.
        native._verify_identities(record["trial_artifacts"])
        trial._require(
            _generator_sources() == evaluation_sources,
            "evaluation source changed during check",
        )
        reader.unchanged()
        plan = record["plan"]
        state = (
            "complete_unavailable"
            if plan["status"] == "unavailable"
            else ("complete_noop" if plan["is_no_op"] else "complete_changed")
        )
        return {
            **report,
            "status": state,
            "reason": plan["reason"],
            "attention_required": False,
        }
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, IndexError) as exc:
        return {
            **report,
            "status": "evidence_invalid",
            "reason": f"{type(exc).__name__}: {exc}",
            "attention_required": True,
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--now")
    parser.add_argument("--trial-root", type=Path, default=trial.DEFAULT_TRIAL_ROOT)
    parser.add_argument("--format", choices=("json", "text"), default="json")
    args = parser.parse_args(argv)
    try:
        report = inspect_shortlist_status(now=args.now, trial_root=args.trial_root)
    except (OSError, ValueError, TypeError) as exc:
        print(f"shortlist probe error: {type(exc).__name__}: {exc}")
        return 2
    if args.format == "json":
        print(json.dumps(report, allow_nan=False, ensure_ascii=False, sort_keys=True))
    else:
        for item in (
            report,
            *([report["previous_due"]] if "previous_due" in report else []),
        ):
            print(
                f"shortlist {item['asof']}: {item['status']} — {item['reason']} "
                "[publication check only; outcomes/prospective eligibility not checked]"
            )
    return int(report["attention_required"])


if __name__ == "__main__":
    raise SystemExit(main())

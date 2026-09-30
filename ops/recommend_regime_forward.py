"""Write-once pre-entry policy decisions and read-only forward evaluation.

The existing native Top10 evaluator supplies verified, immutable morning
evidence. This module never fits, sends, places orders or changes R1 picks.
Already-observed history can inform a choice but cannot become a forward pick.
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import date, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ledger.path_quality import next_bar_boundary
from notifier.delivery_receipt import DEFAULT_RECEIPT_ROOT, read_delivery_receipt, receipt_path
from ops.artifact_provenance import file_identity, strict_json_object
from signals import recommend_microstructure_trial as native
from signals import recommend_regime_replay as policy
from signals import recommend_trade_shortlist_eval as evaluator
from signals import recommend_trade_shortlist_trial as shortlist
from signals.recommend_experiment_eval import _metric_rows, _summary, _validate_outcomes
from signals.recommend_score_labels import path_window

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "output/recommend_regime_forward"
TRIAL_ID = "r1_regime_forward_v1"
SCORE_SCHEMA = "recommend_regime_forward_score.v1"
COMMIT_SCHEMA = "recommend_regime_forward_commit.v1"
START = date(2026, 10, 1)
KST = ZoneInfo("Asia/Seoul")
DUE = time(9, 20)  # Initial observation grace, never an eligibility deadline.
CONFIG = {
    "trial_id": TRIAL_ID, "slot": "open", "prospective_start_asof": START.isoformat(),
    "policy": copy.deepcopy(policy.CONFIG), "policies": list(policy.POLICIES),
    "automatic_promotion": False, "places_orders": False,
    "deadline": "policy_score_durable_observed_at < canonical_execution_start",
}
GENERATOR_SOURCES = sorted(set(evaluator.GENERATOR_SOURCES) | {
    "ops/recommend_regime_forward.py", "signals/recommend_regime_replay.py",
    "signals/recommend_experiment_eval.py", "scripts/capture_recommend_microstructure.py",
})


def _sources():
    values = [file_identity(ROOT / name, root=ROOT) for name in GENERATOR_SOURCES]
    native._require(all(item["exists"] for item in values), "forward source missing")
    return values


def _checked(document, schema):
    native._require(document.get("schema") == schema and document.get("trial_id") == TRIAL_ID,
                    "forward schema/id mismatch")
    native._require(document == native._seal({k: v for k, v in document.items() if k != "payload_sha256"}),
                    "forward payload checksum mismatch")


def _identities(items):
    unique = {}
    for item in items:
        key = str(native._path(item["path"]).absolute())
        native._require(key not in unique or unique[key] == item, "mixed forward input generations")
        unique[key] = item
    return list(unique.values())


def _verify(items):
    # Native evaluation includes explicitly missing historical score files too.
    for item in _identities(items):
        native._require(file_identity(native._path(item["path"]), root=ROOT) == item,
                        "forward input changed")


def history_records(evaluation, *, before_date):
    """Recompute daily metrics from byte-bound labels, not cached averages."""
    audits = {row["date"]: row for row in evaluation["dates"]}
    native._require(len(audits) == len(evaluation["dates"]), "duplicate native audit date")
    records = []
    for daily in evaluation["primary_paired"].get("daily", []):
        day = daily["date"]
        if day >= before_date:
            continue  # Today's outcomes never enter a decision, even if supplied.
        audit = audits[day]
        native._require(audit["status"] == "prospective_comparable", "noncomparable history")
        files = list(audit["evidence_manifest"]["files"].values())
        _verify(files)
        snapshot = strict_json_object(native._path(audit["evidence_manifest"]["files"]["snapshot"]["path"]))
        label = strict_json_object(native._path(audit["evidence_manifest"]["files"]["label"]["path"]))
        native._require(snapshot["asof"] == label["asof"] == day
                        and snapshot["snapshot_id"] == audit["snapshot_id"] == label["snapshot_id"],
                        "history snapshot identity mismatch")
        native._require([row["coin"] for row in snapshot["top3"]] == audit["control_top3"],
                        "history control changed")
        frame = pd.DataFrame(label["rows"]).sort_values("coin").reset_index(drop=True)
        _validate_outcomes(frame)
        outcomes = {}
        for arm in ("control", "challenger"):
            coins = audit[f"{arm}_top3"]
            indices = frame.index[frame["coin"].isin(coins)].tolist()
            native._require(len(indices) == len(set(coins)) == 3
                            and frame.loc[indices, "label_status"].eq("labeled").all(),
                            "history selected labels unavailable")
            vector = _metric_rows(frame, indices).mean(axis=0)
            native._require(np.allclose(vector, policy._vector(daily[arm]), rtol=0, atol=1e-14),
                            "native history metrics mismatch")
            # Preserve native aggregation ordering after independently checking it.
            outcomes[arm] = daily[arm]
        predictors = [snapshot["created_at"], snapshot["decision_completed_at"],
                      audit["score_durable_observed_at"]]
        records.append({
            "date": day, "version": policy.version_from_snapshot(snapshot),
            "context": policy.context_from_snapshot(snapshot),
            "decision_at": max(predictors, key=policy.aware), "predictors_available_at": predictors,
            "entry_at": audit["canonical_execution_start_at"],
            "label_available_at": max(label["path_window_end"], label["labeled_at"], key=policy.aware),
            "outcomes": outcomes, "changed_picks": daily["changed_picks"],
        })
        _verify(files)
    native._require(len({row["date"] for row in records}) == len(records), "duplicate history date")
    return records


def _evaluation_contract(evaluation, now):
    shortlist._checked(evaluation, evaluator.EVALUATION_SCHEMA)
    generated = policy.aware(evaluation["generated_at"])
    native._require(evaluation["config"] == shortlist.TRIAL_CONFIG
                    and evaluation["generator_sources"] == evaluator._generator_sources()
                    and evaluation["inputs_unchanged"] is True
                    and evaluation["deployable"] is False
                    and evaluation["automatic_promotion"] is False,
                    "native evaluation contract mismatch")
    native._require(generated <= now, "future native evaluation")
    return generated


def _bound_record(evaluation, day, root, now):
    """Check the archived publication chain, NOT raw capture validity.

    The publisher and daily native evaluator perform full raw verification.
    Heartbeats must not repeatedly parse months of raw trades. Every small
    artifact read here must be byte-bound to that immutable morning evaluation.
    """
    bound = _identities([*evaluation["trial_inputs"], *evaluation["evidence_inputs"]])
    with native._directory(root, (day, shortlist.TRIAL_ID), create=False) as directory:
        native._require(directory is not None, "native current record missing")
        artifacts = [file_identity(directory[0] / name, root=ROOT) for name in ("score.json", "commit.json")]
        native._require(all(item in bound for item in artifacts), "unbound native publication")
        score, commit = (native._read_document(directory, name) for name in ("score.json", "commit.json"))
    shortlist._checked(score, shortlist.SCORE_SCHEMA)
    shortlist._checked(commit, shortlist.COMMIT_SCHEMA)
    sources = score["source_inputs"]
    native._require(len(sources) == 2 and all(item in bound for item in sources), "unbound native sources")
    _verify([*artifacts, *sources])
    snapshot = native.load_snapshot(native._path(sources[0]["path"]), asof=day,
                                    slot="open", ranking="R1", model_id="recommend_r1_open")
    native._require(score["config"] == shortlist.TRIAL_CONFIG
                    and score["generator_sources"] == shortlist._generator_sources()
                    and score["asof"] == snapshot["asof"] == commit["asof"] == day
                    and score["snapshot_id"] == snapshot["snapshot_id"] == commit["snapshot_id"]
                    and commit["score_payload_sha256"] == score["payload_sha256"]
                    and commit["clock_contract"] == shortlist.CLOCK_CONTRACT
                    and policy.aware(snapshot["decision_completed_at"]) <= policy.aware(score["planned_at"])
                    <= policy.aware(commit["score_durable_observed_at"]) <= now,
                    "bound native contract mismatch")
    return {"status": "committed", "snapshot": snapshot, "plan": score["plan"],
            "commit": commit, "source_inputs": sources, "trial_artifacts": artifacts}


def _bundle(evaluation_path, day, *, shortlist_root, receipt_root, now, full_raw=True):
    identity = file_identity(evaluation_path, root=ROOT)
    evaluation = strict_json_object(evaluation_path)
    generated = _evaluation_contract(evaluation, now)
    native._require(generated.astimezone(KST).date().isoformat() == day,
                    "native evaluation is not today's available evidence")
    record = (shortlist.read_trade_shortlist_record(shortlist_root, day, now) if full_raw
              else _bound_record(evaluation, day, shortlist_root, now))
    native._require(record["status"] == "committed", "native current score is not committed")
    snapshot = record["snapshot"]
    expected_path = native._path(record["source_inputs"][1]["path"]).parent / "trade_shortlist_evaluation.json"
    native._require(Path(evaluation_path).resolve() == expected_path.resolve(), "noncanonical morning evaluation")
    audits = [row for row in evaluation["dates"] if row["date"] == day]
    native._require(len(audits) == 1, "native evaluation omits current date")
    audit = audits[0]
    native._require(audit["record_status"] == "committed"
                    and audit["snapshot_id"] == snapshot["snapshot_id"]
                    and audit["score_durable_observed_at"] == record["commit"]["score_durable_observed_at"]
                    and all(audit[f"{arm}_top3"] == record["plan"][f"{arm}_top3"]
                            for arm in ("control", "challenger")), "current native audit mismatch")
    inputs = _identities([identity, *evaluation["trial_inputs"], *evaluation["evidence_inputs"]])
    for item in record["trial_artifacts"] + record["source_inputs"]:
        native._require(item in inputs, "native evaluation does not bind current score")
    if full_raw:
        _verify(inputs)
    receipt_file = receipt_path(snapshot, root=receipt_root)
    receipt_identity = file_identity(receipt_file, root=ROOT)
    receipt = read_delivery_receipt(snapshot, root=receipt_root)
    native._require(receipt is not None and receipt["delivery_ok"] is True, "successful R1 receipt required")
    native._require(all(policy.aware(receipt[key]) <= now for key in ("sent_at", "attempted_at", "recorded_at")),
                    "future receipt")
    native._require(policy.aware(snapshot["decision_completed_at"]) <= policy.aware(receipt["attempted_at"]),
                    "receipt precedes decision")
    entry = max(path_window(day)[0], next_bar_boundary(receipt["sent_at"]))
    inputs = _identities([*inputs, receipt_identity])
    history = history_records(evaluation, before_date=day)
    checked = inputs if full_raw else [identity, *record["trial_artifacts"], *record["source_inputs"], receipt_identity]
    _verify(checked)
    return {"evaluation_identity": identity, "record": record, "history": history,
            "entry_at": policy.aware(entry).isoformat(), "inputs": inputs}


def _plan(bundle, planned):
    record, history = bundle["record"], bundle["history"]
    snapshot, original = record["snapshot"], record["plan"]
    base = {"status": "unavailable", "reason": original["reason"], "selection": None,
            "selected_coins": None, "control_top3": original["control_top3"],
            "challenger_top3": original["challenger_top3"]}
    if original["status"] != "planned":
        return base
    if planned >= policy.aware(bundle["entry_at"]):
        return {**base, "status": "late", "reason": "planning_not_before_entry"}
    current = {
        "date": snapshot["asof"], "version": policy.version_from_snapshot(snapshot),
        "context": policy.context_from_snapshot(snapshot), "decision_at": planned.isoformat(),
        "predictors_available_at": [snapshot["created_at"], snapshot["decision_completed_at"],
                                    record["commit"]["score_durable_observed_at"]],
        "entry_at": bundle["entry_at"], "label_available_at": None, "outcomes": None,
        "changed_picks": original["changed_picks"],
    }
    selected = policy.selection_plans([*history, current])[-1]
    native._require(selected["date"] == snapshot["asof"], "current plan is not last")
    return {**base, "status": "planned", "reason": None, "selection": selected,
            "selected_coins": {name: original[f"{choice['arm']}_top3"]
                               for name, choice in selected["choices"].items()}}


def _read(directory, *, now):
    if directory is None:
        return {"status": "missing", "reason": "forward_score_missing", "score": None, "commit": None,
                "artifacts": [], "eligibility": "missing"}
    paths = [directory[0] / f"{name}.json" for name in ("score", "commit")]
    before = [file_identity(path, root=ROOT) for path in paths]
    score, commit = (native._read_document(directory, name) for name in ("score.json", "commit.json"))
    native._require(before == [file_identity(path, root=ROOT) for path in paths], "forward files changed while reading")
    if score is None:
        native._require(commit is None, "orphan forward commit")
        return {**_read(None, now=now), "artifacts": before}
    _checked(score, SCORE_SCHEMA)
    day = directory[0].parent.name
    planned = policy.aware(score["planned_at"])
    native._require(score["config"] == CONFIG and score["generator_sources"] == _sources()
                    and score["deployable"] is False and score["automatic_promotion"] is False,
                    "forward policy/source contract changed")
    native._require(score["asof"] == day and score["slot"] == "open" and date.fromisoformat(day) >= START
                    and planned.astimezone(KST).date().isoformat() == day and planned <= now,
                    "forward planning date/clock mismatch")
    bundle = _bundle(native._path(score["evaluation_identity"]["path"]), day,
                     shortlist_root=score["shortlist_root"], receipt_root=score["receipt_root"], now=planned,
                     full_raw=False)
    native._require(bundle["evaluation_identity"] == score["evaluation_identity"]
                    and bundle["inputs"] == score["inputs"]
                    and bundle["entry_at"] == score["entry_at"]
                    and bundle["record"]["snapshot"]["snapshot_id"] == score["snapshot_id"]
                    and _plan(bundle, planned) == score["plan"], "forward choice/input mismatch")
    result = {"status": "uncertain", "reason": "score_without_durable_commit", "score": score,
              "commit": commit, "artifacts": before, "eligibility": "uncertain"}
    if commit is not None:
        _checked(commit, COMMIT_SCHEMA)
        durable = policy.aware(commit["score_durable_observed_at"])
        native._require(commit["asof"] == day and commit["score_payload_sha256"] == score["payload_sha256"]
                        and commit["clock_contract"] == shortlist.CLOCK_CONTRACT
                        and planned <= durable <= now, "forward commit binding/clock mismatch")
        state = score["plan"]["status"]
        eligibility = ("ready" if state == "planned" and durable < policy.aware(score["entry_at"])
                       else "unavailable" if state == "unavailable" else "late")
        result.update(status="committed", reason=None, eligibility=eligibility)
    _verify(before)
    native._require(score["generator_sources"] == _sources(), "forward sources changed while reading")
    return result


def read_forward(day, *, root=DEFAULT_ROOT, now=None):
    """Verify publication, archived choices and labels; no raw-trade scan.

    Full raw validity is a separate daily native-evaluation requirement, not a
    claim made by this bounded operational check.
    """
    day = shortlist._day(day)
    observed = policy.aware(now or native._now())
    with native._directory(root, (day, TRIAL_ID), create=False) as directory:
        result = _read(directory, now=observed)
        if directory is None:
            result["artifacts"] = [file_identity(Path(root) / day / TRIAL_ID / f"{name}.json", root=ROOT)
                                   for name in ("score", "commit")]
            native._require(not any(item["exists"] for item in result["artifacts"]), "forward record appeared while reading")
        return result


def record_forward(evaluation_path, *, shortlist_root=shortlist.DEFAULT_TRIAL_ROOT,
                   receipt_root=DEFAULT_RECEIPT_ROOT, output_root=DEFAULT_ROOT, now_fn=native._now):
    """No CLI clock override: production always samples the actual wall clock."""
    started = policy.aware(now_fn())
    day = started.astimezone(KST).date()
    native._require(day >= START, "forward trial has not started; no retrospective publication")
    sources = _sources()
    # Inspect read-only first: a crash cannot be healed with a later commit.
    existing = read_forward(day, root=output_root, now=started)
    if existing["status"] != "missing":
        native._require(native._path(existing["score"]["evaluation_identity"]["path"]).resolve()
                        == Path(evaluation_path).resolve(), "existing forward record has other inputs")
        return {"status": existing["status"], "eligibility": existing["eligibility"], "reused": True}
    bundle = _bundle(evaluation_path, day.isoformat(), shortlist_root=shortlist_root,
                     receipt_root=receipt_root, now=started)
    planned = policy.aware(now_fn())
    native._require(started <= planned and planned.astimezone(KST).date() == day, "forward planning clock moved")
    plan = _plan(bundle, planned)
    native._require(sources == _sources(), "forward sources changed while planning")
    score = native._seal({
        "schema": SCORE_SCHEMA, "trial_id": TRIAL_ID, "config": copy.deepcopy(CONFIG),
        "asof": day.isoformat(), "slot": "open", "planned_at": planned.isoformat(),
        "entry_at": bundle["entry_at"], "snapshot_id": bundle["record"]["snapshot"]["snapshot_id"],
        "evaluation_identity": bundle["evaluation_identity"], "inputs": bundle["inputs"],
        "shortlist_root": str(Path(shortlist_root).absolute()), "receipt_root": str(Path(receipt_root).absolute()),
        "generator_sources": sources, "plan": plan, "deployable": False, "automatic_promotion": False,
    })
    with native._directory(output_root, (day.isoformat(), TRIAL_ID), create=True) as directory:
        # No replacement after concurrent creation; the loser cannot commit.
        try:
            native._publish_new(directory, "score.json", score)
        except FileExistsError:
            return {"status": "uncertain", "eligibility": "uncertain", "reused": True}
        durable = policy.aware(now_fn())
        native._require(durable >= planned, "clock moved backwards after forward fsync")
        _verify(bundle["inputs"])
        native._require(sources == _sources(), "forward sources changed during publication")
        commit = native._seal({
            "schema": COMMIT_SCHEMA, "trial_id": TRIAL_ID, "asof": day.isoformat(),
            "score_payload_sha256": score["payload_sha256"],
            "score_durable_observed_at": durable.isoformat(), "clock_contract": shortlist.CLOCK_CONTRACT,
        })
        native._publish_new(directory, "commit.json", commit)
    eligible = "ready" if plan["status"] == "planned" and durable < policy.aware(bundle["entry_at"]) else (
        "unavailable" if plan["status"] == "unavailable" else "late")
    return {"status": "committed", "eligibility": eligible, "reused": False,
            "score_durable_observed_at": durable.isoformat()}


def evaluate_forward(evaluation, *, root=DEFAULT_ROOT, now=None, n_boot=1000, seed=42):
    """Join stored choices to freshly native-verified outcomes, not new choices.

    Raw identities across all saved decisions are verified as ONE deduplicated
    set below, rather than once per day times its entire preceding history.
    """
    observed = policy.aware(now or native._now()).astimezone(KST)
    if type(n_boot) is not int or n_boot < 1:
        raise ValueError("invalid forward bootstrap count")
    _evaluation_contract(evaluation, observed)
    daily = {row["date"]: row for row in evaluation["primary_paired"].get("daily", [])}
    audits = {row["date"]: row for row in evaluation["dates"]}
    native._require(len(daily) == len(evaluation["primary_paired"].get("daily", []))
                    and len(audits) == len(evaluation["dates"]), "duplicate native date")
    if observed.date() >= START:
        history_records(evaluation, before_date=(observed.date() + timedelta(days=1)).isoformat())
    dates, paired = [], []
    inputs = [*evaluation["trial_inputs"], *evaluation["evidence_inputs"]]
    for offset in range(max(0, (observed.date() - START).days + 1)):
        day = (START + timedelta(days=offset)).isoformat()
        record = read_forward(day, root=root, now=observed)
        inputs.extend(record["artifacts"])
        if record["score"] is not None:
            inputs.extend(record["score"]["inputs"])
        row = {"date": day, "record_status": record["status"], "status": record["eligibility"],
               "eligibility": record["eligibility"], "reason": record["reason"]}
        dates.append(row)
        if record["status"] != "committed" or record["eligibility"] != "ready":
            if day == observed.date().isoformat() and observed.time() < DUE and record["status"] == "missing":
                row.update(status="not_due", reason="before_publication_deadline")
            continue
        score = record["score"]
        audit = audits.get(day)
        if day not in daily or audit is None or audit["status"] != "prospective_comparable":
            row.update(status="pending" if day == observed.date().isoformat() else "unavailable",
                       reason="canonical_paired_outcomes_unavailable")
            continue
        native._require(audit["snapshot_id"] == score["snapshot_id"]
                        and policy.aware(audit["canonical_execution_start_at"]) == policy.aware(score["entry_at"])
                        and all(audit[f"{arm}_top3"] == score["plan"][f"{arm}_top3"]
                                for arm in ("control", "challenger")), "forward canonical selection/entry changed")
        selection = score["plan"]["selection"]
        values = {name: daily[day][choice["arm"]] for name, choice in selection["choices"].items()}
        changed = {name: len(set(score["plan"]["selected_coins"][name]) - set(score["plan"]["control_top3"]))
                   for name in policy.POLICIES}
        row.update(status="prospective_comparable", reason=None,
                   choices=selection["choices"], changed_picks=changed)
        paired.append({"date": day, "values": values, "changed_picks": changed})
    summaries = {}
    for name in policy.POLICIES:
        values = np.array([policy._vector(row["values"][name]) for row in paired])
        base = np.array([policy._vector(row["values"]["fixed_r1"]) for row in paired])
        changed = sum(row["changed_picks"][name] for row in paired)
        summaries[name] = {
            "n_dates": len(paired), "changed_picks": changed,
            "changed_dates": sum(row["changed_picks"][name] > 0 for row in paired),
            "metrics": _summary(values, n_boot, seed) if len(values) else None,
            "minus_fixed_r1": _summary(values - base, n_boot, seed) if len(values) else None,
            "effect_status": "no_effect_observations" if not changed else "descriptive_forward_effect",
        }
    inputs = _identities(inputs)
    _verify(inputs)
    return {"schema": "recommend_regime_forward_evaluation.v1", "trial_id": TRIAL_ID,
            "config": copy.deepcopy(CONFIG), "deployable": False, "automatic_promotion": False,
            "scope": "pre_entry_frozen_policies_on_canonical_net_pick_proxies_not_portfolio",
            "n_boot": n_boot, "seed": seed, "dates": dates, "policies": summaries,
            "committed_policy_records": sum(row["record_status"] == "committed" for row in dates),
            "prospective_policy_records": sum(row["eligibility"] == "ready" for row in dates),
            "paired_dates": len(paired), "daily": paired, "input_files": inputs}


def inspect_forward(*, now=None, root=DEFAULT_ROOT):
    observed = policy.aware(now or native._now()).astimezone(KST)
    day = observed.date() if observed.time() >= DUE else observed.date() - timedelta(days=1)
    result = {"trial_id": TRIAL_ID, "checked_at": observed.isoformat(), "due_date": day.isoformat(),
              "deployable": False, "attention_required": False,
              "verification_scope": "publication_and_choice_chain_not_raw_capture"}
    if day < START:
        return {**result, "status": "not_started", "reason": "before_first_publication_due"}
    try:
        record = read_forward(day, root=root, now=observed)
        healthy = record["status"] == "committed" and record["eligibility"] in {"ready", "unavailable"}
        return {**result, "status": record["status"], "eligibility": record["eligibility"],
                "reason": record["reason"], "attention_required": not healthy}
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, IndexError) as exc:
        return {**result, "status": "evidence_invalid", "reason": f"{type(exc).__name__}: {exc}",
                "attention_required": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Read-only pre-entry policy publication check")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--now", help="read-only check time; never publishes or repairs")
    parser.add_argument("--format", choices=("json", "text"), default="json")
    args = parser.parse_args(argv)
    result = inspect_forward(now=args.now, root=args.root)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False) if args.format == "json" else
          f"regime forward: {result['status']} due={result['due_date']} "
          f"eligibility={result.get('eligibility')} reason={result.get('reason')}")
    return int(result["attention_required"])


if __name__ == "__main__":
    raise SystemExit(main())

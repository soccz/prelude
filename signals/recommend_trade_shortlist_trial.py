"""Separate write-once Top10 trial; no fitting, dispatch or outcome selection.

The frozen Phase8 implementation supplies its native evidence loader and generic
filesystem primitives, never its ID/config/record reader. Native loading still
hashes each finalized raw file repeatedly; no unverified cache or speed claim is
introduced. Publication proves score durability only. Prospective eligibility
requires the separate evaluator's strict canonical delivery-entry time check.
"""
from __future__ import annotations

import copy
from datetime import date
from pathlib import Path

from ops.artifact_provenance import file_identity, strict_json_object
from signals import recommend_microstructure_trial as native
from signals.recommend_trade_shortlist import SHORTLIST_CONFIG, plan_trade_shortlist

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRIAL_ROOT = ROOT / "output/recommend_trade_shortlist_trials"
TRIAL_ID = "r1_top10_trade_imbalance_v1"
SCORE_SCHEMA = "recommend_trade_shortlist_trial_score.v1"
COMMIT_SCHEMA = "recommend_trade_shortlist_trial_commit.v1"
CLOCK_CONTRACT = "observed_after_score_file_and_directory_fsync_not_commit_completion"
TRIAL_CONFIG = {
    "trial_id": TRIAL_ID, "slot": "open", "universe_size": 100, "top_k": 3,
    "shortlist_config": copy.deepcopy(SHORTLIST_CONFIG),
    "design_frozen_asof": "2026-09-09", "prospective_start_asof": "2026-09-10",
    "scheduled_capture_start_kst": "08:45:00",
    "microstructure_family_additional_variants": 1,
    "microstructure_family_total_variants": 2,
    "trial_count_scope": "microstructure_family_not_project_total",
    "prospective_deadline": "score_durable_observed_at < canonical_execution_start",
    "no_op_days_in_primary": True, "automatic_promotion": False, "places_orders": False,
}
GENERATOR_SOURCES = (
    "signals/recommend_trade_shortlist_trial.py", "signals/recommend_trade_shortlist.py",
    "signals/recommend_microstructure_trial.py", "signals/recommend_microstructure.py",
    "data/upbit_microstructure.py", "signals/recommend_snapshot.py",
    "ops/artifact_provenance.py",
)


class TradeShortlistTrialError(ValueError):
    """Invalid evidence is a blocking error, not an unavailable observation."""


def _require(condition, reason):
    if not condition:
        raise TradeShortlistTrialError(reason)


def _generator_sources():
    sources = [file_identity(ROOT / name, root=ROOT) for name in GENERATOR_SOURCES]
    _require(all(item.get("exists") is True for item in sources), "trial generator missing")
    return sources


def _checked(document, schema):
    _require(isinstance(document, dict) and document.get("schema") == schema, "unsupported trial schema")
    expected = native._seal({key: value for key, value in document.items() if key != "payload_sha256"})
    _require(document == expected, "trial payload checksum mismatch")
    _require(document.get("trial_id") == TRIAL_ID, "trial id mismatch")


def _day(value):
    text = value.isoformat() if type(value) is date else value
    _require(isinstance(text, str) and date.fromisoformat(text).isoformat() == text,
             "canonical trial date required")
    return text


def _load_inputs(snapshot_path, feature_path):
    snapshot, feature, _old_plan, sources = native._load_inputs(snapshot_path, feature_path)
    return snapshot, feature, plan_trade_shortlist(snapshot, feature), sources


def _check_capture_clock(feature, planned):
    manifest_source = feature["provenance"]["files"][1]
    manifest = strict_json_object(native._path(manifest_source["path"]))
    _require(file_identity(native._path(manifest_source["path"]), root=ROOT) == manifest_source,
             "capture manifest changed while checking clock")
    _require(native._ns(planned.isoformat()) >= manifest["ended_at_ns"], "planning before capture completion")


def _empty_record(root, day):
    folder = Path(root) / day / TRIAL_ID
    paths = [folder / name for name in ("score.json", "commit.json")]
    artifacts = [file_identity(path, root=ROOT) for path in paths]
    _require(not any(item["exists"] for item in artifacts), "trial appeared during missing-record inspection")
    return {
        "status": "missing", "reason": "trial_score_missing", "score": None, "commit": None,
        "snapshot": None, "feature": None, "plan": None, "source_inputs": [],
        "trial_artifacts": artifacts,
        "score_path": str(paths[0]), "commit_path": str(paths[1]),
    }


def _read_record(directory, *, now):
    paths = [directory[0] / name for name in ("score.json", "commit.json")]
    before = [file_identity(path, root=ROOT) for path in paths]
    score = native._read_document(directory, "score.json")
    commit = native._read_document(directory, "commit.json")
    _require(before == [file_identity(path, root=ROOT) for path in paths], "trial artifacts changed while reading")
    if score is None:
        _require(commit is None, "orphan commit without score")
        return _empty_record(directory[0].parent.parent, directory[0].parent.name)
    generators = _generator_sources()
    _checked(score, SCORE_SCHEMA)
    _require(score.get("config") == TRIAL_CONFIG, "trial config mismatch")
    _require(score.get("asof") == directory[0].parent.name and score.get("slot") == "open",
             "trial directory date or slot mismatch")
    _require(score.get("deployable") is False and score.get("effect_status") == "not_evaluated"
             and score.get("prospective_status") == "pending_canonical_delivery_entry_check",
             "unexpected trial promotion or eligibility claim")
    _require(score.get("generator_sources") == generators, "trial generator changed")
    sources = score.get("source_inputs")
    _require(isinstance(sources, list) and len(sources) == 2, "snapshot and feature source identities required")
    native._verify_identities(sources)
    snapshot, feature, plan, loaded_sources = _load_inputs(
        native._path(sources[0]["path"]), native._path(sources[1]["path"])
    )
    _require(loaded_sources == sources and plan == score.get("plan"), "trial plan/source mismatch")
    _require(score.get("snapshot_id") == snapshot["snapshot_id"]
             and score["asof"] == snapshot["asof"], "trial snapshot identity mismatch")
    planned = native._time(score["planned_at"])
    _require(native._time(snapshot["decision_completed_at"]) <= planned <= now, "trial planning clock invalid")
    _check_capture_clock(feature, planned)
    result = {
        "score": score, "commit": commit, "snapshot": snapshot, "feature": feature, "plan": plan,
        "source_inputs": sources, "trial_artifacts": before,
        "score_path": str(paths[0]), "commit_path": str(paths[1]),
        "status": "uncertain", "reason": "score_without_durable_commit_receipt",
    }
    if commit is not None:
        _checked(commit, COMMIT_SCHEMA)
        _require(commit.get("score_payload_sha256") == score["payload_sha256"]
                 and commit.get("snapshot_id") == snapshot["snapshot_id"]
                 and commit.get("asof") == snapshot["asof"], "commit/score binding mismatch")
        _require(commit.get("clock_contract") == CLOCK_CONTRACT, "unsupported publication clock contract")
        _require(planned <= native._time(commit["score_durable_observed_at"]) <= now,
                 "durable score clock invalid")
        result.update(status="committed", reason=None)
    native._verify_identities(sources)
    _require(generators == _generator_sources(), "generator changed while reading trial")
    _require(before == [file_identity(path, root=ROOT) for path in paths], "trial artifacts changed while validating")
    return result


def read_trade_shortlist_record(root, day, now):
    """Read and recompute one record with full native raw/source verification.

    No directory, lock, file or replacement commit is created by inspection.
    A committed unavailable/no-op plan is preserved, not reselected or promoted.
    """
    try:
        day, now = _day(day), native._time(now)
        with native._directory(root, (day, TRIAL_ID), create=False) as directory:
            return _read_record(directory, now=now) if directory else _empty_record(root, day)
    except TradeShortlistTrialError:
        raise
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise TradeShortlistTrialError(str(exc)) from exc


def record_trade_shortlist_trial(snapshot_path, feature_path, *, output_root=DEFAULT_TRIAL_ROOT,
                                now_fn=native._now):
    """Publish once after the old trial, before either cumulative evaluation.

    Score-only crashes stay uncertain even on retry. A later canonical evaluator
    alone decides whether the observed score durability precedes actual entry.
    """
    try:
        generators = _generator_sources()
        snapshot, feature, plan, sources = _load_inputs(snapshot_path, feature_path)
        _require(generators == _generator_sources(), "generator changed during input validation")
        planned = native._time(now_fn())
        _require(native._time(snapshot["decision_completed_at"]) <= planned, "planning before snapshot completion")
        _check_capture_clock(feature, planned)
        score = native._seal({
            "schema": SCORE_SCHEMA, "trial_id": TRIAL_ID, "config": copy.deepcopy(TRIAL_CONFIG),
            "asof": snapshot["asof"], "slot": "open", "snapshot_id": snapshot["snapshot_id"],
            "planned_at": planned.isoformat(), "source_inputs": sources,
            "generator_sources": generators, "plan": plan, "deployable": False,
            "effect_status": "not_evaluated", "prospective_status": "pending_canonical_delivery_entry_check",
        })
        with native._directory(output_root, (snapshot["asof"], TRIAL_ID), create=True) as directory:
            existing = _read_record(directory, now=planned)
            if existing["status"] != "missing":
                _require(existing["source_inputs"] == sources, "existing trial has different feature/snapshot inputs")
                return {
                    "status": existing["status"], "reused": True, "reason": existing["reason"],
                    "score_path": existing["score_path"], "commit_path": existing["commit_path"],
                    "plan_status": existing["plan"]["status"], "changed_picks": existing["plan"]["changed_picks"],
                    "prospective_status": "pending_canonical_delivery_entry_check",
                }
            paths = {name + "_path": str(directory[0] / (name + ".json")) for name in ("score", "commit")}
            try:
                native._publish_new(directory, "score.json", score)
            except FileExistsError:
                return {"status": "uncertain", "reused": True, "reason": "concurrent_score_publication",
                        **paths, "plan_status": plan["status"], "changed_picks": plan["changed_picks"]}
            durable = native._time(now_fn())
            _require(durable >= planned, "clock moved backwards after score durability")
            native._verify_identities(sources)
            _require(generators == _generator_sources(), "generator changed during publication")
            commit = native._seal({
                "schema": COMMIT_SCHEMA, "trial_id": TRIAL_ID, "asof": snapshot["asof"],
                "snapshot_id": snapshot["snapshot_id"], "score_payload_sha256": score["payload_sha256"],
                "score_durable_observed_at": durable.isoformat(), "clock_contract": CLOCK_CONTRACT,
            })
            native._publish_new(directory, "commit.json", commit)
            return {
                "status": "committed", "reused": False, "reason": None, **paths,
                "score_durable_observed_at": durable.isoformat(), "plan_status": plan["status"],
                "changed_picks": plan["changed_picks"],
                "prospective_status": "pending_canonical_delivery_entry_check",
            }
    except TradeShortlistTrialError:
        raise
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise TradeShortlistTrialError(str(exc)) from exc

"""Synthetic evaluator faults; native publication and canonical IO tested separately."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta

import pytest
import requests

import test_recommend_microstructure_trial as native
from ops.artifact_provenance import canonical_json_bytes, file_identity, sha256_bytes
from ops.recommendation_evidence import EvidenceError, EvidenceUnavailable
from scripts.evaluate_recommend_trade_shortlist_trial import main
from signals import recommend_trade_shortlist_eval as evaluate
from signals.recommend_trade_shortlist import plan_trade_shortlist


@pytest.fixture(autouse=True)
def no_live_operations(monkeypatch):
    monkeypatch.setenv("PRELUDE_FORBID_TELEGRAM", "1")

    def forbidden(*args, **kwargs):
        pytest.fail("evaluation attempted network, fitting, or DB collection")

    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    for name in ("_data_metadata", "_git_code_metadata", "_environment_metadata"):
        monkeypatch.setattr(native.snapshots, name, forbidden)


def _record(tmp_path, monkeypatch, *, day="2026-09-10", durable="09:10:00", **kwargs):
    paths = native._files(tmp_path / "inputs" / day, monkeypatch, day=day, **kwargs)
    snapshot, feature = (json.loads(path.read_bytes()) for path in paths)
    plan = plan_trade_shortlist(snapshot, feature)
    score = {
        "snapshot_id": snapshot["snapshot_id"],
        "asof": day,
        "plan": plan,
        "source_inputs": [file_identity(path, root=evaluate.ROOT) for path in paths],
    }
    commit = {"score_durable_observed_at": f"{day}T{durable}+09:00"}
    directory = tmp_path / "trials" / day / evaluate.TRIAL_ID
    directory.mkdir(parents=True)
    artifacts = []
    for name, value in (("score.json", score), ("commit.json", commit)):
        path = directory / name
        path.write_bytes(canonical_json_bytes(value))
        artifacts.append(file_identity(path, root=evaluate.ROOT))
    return {
        "status": "committed",
        "reason": None,
        "score": score,
        "commit": commit,
        "snapshot": snapshot,
        "feature": feature,
        "plan": plan,
        "trial_artifacts": artifacts,
    }


def _evaluate(
    tmp_path, monkeypatch, records=(), *, halt=None, now="2026-09-12T12:00:00+09:00"
):
    by_day = {record["score"]["asof"]: record for record in records}
    monkeypatch.setattr(
        evaluate,
        "read_trade_shortlist_record",
        lambda root, day, now: by_day.get(
            day, {"status": "missing", "reason": "trial_score_missing"}
        ),
    )
    monkeypatch.setattr(
        evaluate,
        "load_recommendation_evidence",
        lambda path, **kwargs: native._evidence(path, halt=halt),
    )
    return evaluate.evaluate_trade_shortlist_trials(
        tmp_path / "trials", now=now, n_boot=100
    )


def test_zero_data_before_launch_is_not_started_no_writes(tmp_path, monkeypatch):
    report = _evaluate(tmp_path, monkeypatch, now="2026-09-09T12:00:00+09:00")
    assert report["status"] == "not_started"
    assert report["effect_status"] == "not_evaluated"
    assert report["coverage"]["calendar_expected_count"] == 0
    assert not (tmp_path / "trials").exists()
    assert (
        report["new_hypotheses"] == 1
        and report["microstructure_family_hypotheses"] == 2
    )
    assert report["deployable"] is report["automatic_promotion"] is False


@pytest.mark.parametrize(
    ("now", "expected", "not_due"),
    [
        ("2026-09-10T08:44:59+09:00", [], ["2026-09-10"]),
        ("2026-09-10T08:45:00+09:00", ["2026-09-10"], []),
        ("2026-09-12T08:00:00+09:00", ["2026-09-10", "2026-09-11"], ["2026-09-12"]),
        ("2026-09-09T23:45:00+00:00", ["2026-09-10"], []),
    ],
)
def test_calendar_includes_unrecorded_dates_and_uses_kst(
    tmp_path, monkeypatch, now, expected, not_due
):
    report = _evaluate(tmp_path, monkeypatch, now=now)
    assert report["coverage"]["calendar_expected_dates"] == expected
    assert report["coverage"]["calendar_missing_dates"] == expected
    assert report["coverage"]["calendar_not_due_dates"] == not_due
    assert report["primary_paired"]["metrics"] is None


def test_frozen_choices_net_cost_and_context_do_not_mutate_inputs(
    tmp_path, monkeypatch
):
    record = _record(tmp_path, monkeypatch)
    before = copy.deepcopy(record)
    report = _evaluate(tmp_path, monkeypatch, [record])
    assert record == before
    daily = report["primary_paired"]["daily"][0]
    assert daily["control"]["tp5_sl3_return_net"] == pytest.approx(0.0005)
    assert daily["challenger"]["tp5_sl3_return_net"] == pytest.approx(
        0.0008333333333333334
    )
    assert daily["challenger_minus_control"]["up10"] == pytest.approx(1 / 3)
    assert daily["challenger_minus_control"]["dn5"] == pytest.approx(-1 / 3)
    assert report["primary_paired"]["picks_per_arm"] == 3
    assert report["full_universe_common"]["baseline_candidates_per_date"] == 100
    assert report["dates"][0]["stored_context"]["shortlist"]["n_candidates"] == 10
    assert report["dates"][0]["challenger_top3"] == record["plan"]["challenger_top3"]
    assert report["effect_status"] == "descriptive_only_not_a_promotion_verdict"
    summary = report["primary_paired"]["metrics"]["challenger_minus_control"]
    assert summary["ci_status"] == "insufficient_dates"
    assert summary["iid_date_ci95"] is summary["observed_date_block3_ci95"] is None


def test_no_op_is_a_real_paired_day_with_exact_zero_delta(tmp_path, monkeypatch):
    record = _record(tmp_path, monkeypatch, changed=False)
    report = _evaluate(tmp_path, monkeypatch, [record])
    assert report["primary_paired"]["n_no_op_dates"] == 1
    assert all(
        value == 0
        for value in report["primary_paired"]["daily"][0][
            "challenger_minus_control"
        ].values()
    )


@pytest.mark.parametrize("halt", [1, 3, 4])
def test_selected_halt_never_replaces_a_pick_or_counts_zero_return(
    tmp_path, monkeypatch, halt
):
    record = _record(tmp_path, monkeypatch)
    report = _evaluate(tmp_path, monkeypatch, [record], halt=halt)
    assert report["primary_paired"]["n_dates"] == 0
    assert report["dates"][0]["reason"] == "selected_outcome_unavailable"
    assert report["dates"][0]["challenger_top3"] == record["plan"]["challenger_top3"]


def test_nonselected_halt_only_removes_separate_full_universe_cohort(
    tmp_path, monkeypatch
):
    record = _record(tmp_path, monkeypatch)
    report = _evaluate(tmp_path, monkeypatch, [record], halt=100)
    assert report["primary_paired"]["n_dates"] == 1
    assert report["full_universe_common"]["n_dates"] == 0
    assert report["dates"][0]["full_universe_baseline_status"] == "unavailable"


def test_missing_shortlist_never_accesses_outcomes(tmp_path, monkeypatch):
    record = _record(tmp_path, monkeypatch, missing=True)
    _evaluate(tmp_path, monkeypatch, [record])

    def forbidden(*args, **kwargs):
        pytest.fail("missing shortlist plan accessed canonical labels")

    monkeypatch.setattr(evaluate, "load_recommendation_evidence", forbidden)
    report = evaluate.evaluate_trade_shortlist_trials(
        tmp_path / "trials", now="2026-09-12T12:00:00+09:00"
    )
    assert report["dates"][0]["status"] == "unavailable"
    assert report["dates"][0]["challenger_top3"] is None


@pytest.mark.parametrize(
    ("durable", "status"),
    [
        ("09:14:59.999999", "prospective_comparable"),
        ("09:15:00", "late"),
        ("09:15:00.000001", "late"),
    ],
)
def test_canonical_entry_requires_strictly_earlier_durability(
    tmp_path, monkeypatch, durable, status
):
    record = _record(tmp_path, monkeypatch, durable=durable)
    report = _evaluate(tmp_path, monkeypatch, [record])
    assert report["dates"][0]["status"] == status


def test_prelaunch_record_is_not_promoted_to_forward(tmp_path, monkeypatch):
    record = _record(tmp_path, monkeypatch, day="2026-09-09")
    report = _evaluate(tmp_path, monkeypatch, [record])
    assert report["dates"][0]["status"] == "prelaunch_replay"
    assert report["coverage"]["paired_dates"] == 0


@pytest.mark.parametrize(
    ("reason", "status"),
    [
        ("label_missing", "pending"),
        ("receipt_missing_delivery_unknown", "pending"),
        ("label_not_available_as_of_now", "pending"),
        ("label_status=OPEN", "pending"),
        ("receipt_not_successful", "unavailable"),
    ],
)
def test_canonical_unavailable_reasons_remain_explicit(
    tmp_path, monkeypatch, reason, status
):
    record = _record(tmp_path, monkeypatch)
    _evaluate(tmp_path, monkeypatch, [record])

    def unavailable(*args, **kwargs):
        raise EvidenceUnavailable(reason)

    monkeypatch.setattr(evaluate, "load_recommendation_evidence", unavailable)
    report = evaluate.evaluate_trade_shortlist_trials(
        tmp_path / "trials", now="2026-09-12T12:00:00+09:00"
    )
    assert report["dates"][0]["status"] == status
    assert report["dates"][0]["reason"] == reason


def test_corruption_blocks_entire_evaluation(tmp_path, monkeypatch):
    record = _record(tmp_path, monkeypatch)
    _evaluate(tmp_path, monkeypatch, [record])

    def corrupt(*args, **kwargs):
        raise EvidenceError("label checksum mismatch")

    monkeypatch.setattr(evaluate, "load_recommendation_evidence", corrupt)
    with pytest.raises(EvidenceError, match="checksum"):
        evaluate.evaluate_trade_shortlist_trials(
            tmp_path / "trials", now="2026-09-12T12:00:00+09:00"
        )


def test_score_without_commit_remains_uncertain_without_label_access(
    tmp_path, monkeypatch
):
    record = _record(tmp_path, monkeypatch)
    record.update(
        status="uncertain", reason="score_without_durable_commit_receipt", commit=None
    )
    report = _evaluate(tmp_path, monkeypatch, [record])
    assert report["dates"][0]["status"] == "uncertain"
    assert report["coverage"]["paired_dates"] == 0


def test_five_days_have_day_cluster_ci_and_leave_one_day_out(tmp_path, monkeypatch):
    records = [
        _record(
            tmp_path,
            monkeypatch,
            day=(datetime(2026, 9, 10) + timedelta(days=i)).date().isoformat(),
        )
        for i in range(5)
    ]
    report = _evaluate(tmp_path, monkeypatch, records, now="2026-09-16T12:00:00+09:00")
    summary = report["primary_paired"]
    assert summary["n_dates"] == 5 and summary["picks_per_arm"] == 15
    assert len(summary["leave_one_date_out"]) == 5
    assert summary["metrics"]["challenger_minus_control"]["ci_status"] == "available"


def test_label_row_shuffle_preserves_metrics(tmp_path, monkeypatch):
    record = _record(tmp_path, monkeypatch)
    before = _evaluate(tmp_path, monkeypatch, [record])

    def shuffled(path, **kwargs):
        document = native._evidence(path)
        document["label"]["rows"].reverse()
        return document

    monkeypatch.setattr(evaluate, "load_recommendation_evidence", shuffled)
    after = evaluate.evaluate_trade_shortlist_trials(
        tmp_path / "trials", now="2026-09-12T12:00:00+09:00", n_boot=100
    )
    assert before["primary_paired"] == after["primary_paired"]


def test_primary_and_full_universe_use_their_own_date_denominators(
    tmp_path, monkeypatch
):
    records = [
        _record(tmp_path, monkeypatch, day=day) for day in ("2026-09-10", "2026-09-11")
    ]
    _evaluate(tmp_path, monkeypatch, records)

    def partial(path, **kwargs):
        day = json.loads(path.read_bytes())["asof"]
        return native._evidence(path, halt=100 if day == "2026-09-11" else None)

    monkeypatch.setattr(evaluate, "load_recommendation_evidence", partial)
    report = evaluate.evaluate_trade_shortlist_trials(
        tmp_path / "trials", now="2026-09-13T12:00:00+09:00", n_boot=100
    )
    assert report["primary_paired"]["n_dates"] == 2
    assert report["primary_paired"]["picks_per_arm"] == 6
    assert report["full_universe_common"]["n_dates"] == 1
    assert [row["date"] for row in report["full_universe_common"]["daily"]] == [
        "2026-09-10"
    ]
    assert report["full_universe_common"]["control_minus_full_universe"]["dates"] == 1


def test_outcome_reversal_does_not_reselect_published_top3(tmp_path, monkeypatch):
    record = _record(tmp_path, monkeypatch)
    before = _evaluate(tmp_path, monkeypatch, [record])

    def reversed_outcomes(path, **kwargs):
        document = native._evidence(path)
        for row in document["label"]["rows"]:
            row.update(
                up10=row["rank"] == 100,
                mfe=0.11 if row["rank"] == 100 else 0.02,
                dn5=row["rank"] == 4,
                mae=-0.06 if row["rank"] == 4 else -0.01,
            )
        return document

    monkeypatch.setattr(evaluate, "load_recommendation_evidence", reversed_outcomes)
    after = evaluate.evaluate_trade_shortlist_trials(
        tmp_path / "trials", now="2026-09-12T12:00:00+09:00", n_boot=100
    )
    for arm in ("control", "challenger"):
        assert after["dates"][0][f"{arm}_top3"] == before["dates"][0][f"{arm}_top3"]
    delta = after["primary_paired"]["daily"][0]["challenger_minus_control"]
    assert delta["dn5"] == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    "fault", ["duplicate", "missing", "nonfinite", "wrong_control"]
)
def test_broken_canonical_contract_is_hard_failure(tmp_path, monkeypatch, fault):
    record = _record(tmp_path, monkeypatch)
    _evaluate(tmp_path, monkeypatch, [record])

    def broken(path, **kwargs):
        document = native._evidence(path)
        if fault == "duplicate":
            document["label"]["rows"][-1] = document["label"]["rows"][0]
        elif fault == "missing":
            document["label"]["rows"].pop()
        elif fault == "nonfinite":
            document["label"]["rows"][0]["eod_return_net"] = float("nan")
        else:
            document["snapshot"]["top3"] = document["snapshot"]["universe"][1:4]
        return document

    monkeypatch.setattr(evaluate, "load_recommendation_evidence", broken)
    with pytest.raises(ValueError):
        evaluate.evaluate_trade_shortlist_trials(
            tmp_path / "trials", now="2026-09-12T12:00:00+09:00"
        )


def test_mutated_evidence_during_read_blocks_whole_report(tmp_path, monkeypatch):
    record = _record(tmp_path, monkeypatch)
    _evaluate(tmp_path, monkeypatch, [record])

    def mutate(path, **kwargs):
        result = native._evidence(path)
        feature_path = evaluate._path(record["score"]["source_inputs"][1]["path"])
        feature_path.write_bytes(feature_path.read_bytes() + b" ")
        return result

    monkeypatch.setattr(evaluate, "load_recommendation_evidence", mutate)
    with pytest.raises(ValueError, match="inputs changed"):
        evaluate.evaluate_trade_shortlist_trials(
            tmp_path / "trials", now="2026-09-12T12:00:00+09:00"
        )


def test_source_change_during_evaluation_blocks_without_modifying_source(
    tmp_path, monkeypatch
):
    original = evaluate._generator_sources
    calls = 0

    def drift():
        nonlocal calls
        calls += 1
        result = original()
        if calls > 1:
            result[0]["sha256"] = "0" * 64
        return result

    monkeypatch.setattr(evaluate, "_generator_sources", drift)
    with pytest.raises(ValueError, match="generator changed"):
        evaluate.evaluate_trade_shortlist_trials(
            tmp_path / "absent", now="2026-09-09T12:00:00+09:00"
        )


def test_symlink_root_and_future_directory_fail_closed(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises((OSError, ValueError)):
        evaluate.evaluate_trade_shortlist_trials(alias, now="2026-09-09T12:00:00+09:00")
    (root / "2026-09-10").mkdir()
    with pytest.raises(ValueError, match="future"):
        evaluate.evaluate_trade_shortlist_trials(root, now="2026-09-09T12:00:00+09:00")


@pytest.mark.parametrize("kwargs", [{"now": "2026-09-09"}, {"n_boot": 0}, {"seed": -1}])
def test_invalid_options_fail_closed(tmp_path, kwargs):
    with pytest.raises(ValueError):
        evaluate.evaluate_trade_shortlist_trials(tmp_path / "absent", **kwargs)


def test_cli_stdout_seal_and_new_only_output(tmp_path, capsys):
    args = [
        "--trial-root",
        str(tmp_path / "absent"),
        "--now",
        "2026-09-09T12:00:00+09:00",
    ]
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    digest = result.pop("payload_sha256")
    assert digest == sha256_bytes(canonical_json_bytes(result))
    assert result["trial_id"] == "r1_top10_trade_imbalance_v1"
    path = tmp_path / "new.json"
    assert main([*args, "--output", str(path)]) == 0
    before = path.read_bytes()
    capsys.readouterr()
    assert main([*args, "--output", str(path)]) == 1
    assert json.loads(capsys.readouterr().err)["status"] == "blocked"
    assert path.read_bytes() == before


def test_native_publication_reader_integration(tmp_path, monkeypatch):
    from signals.recommend_trade_shortlist_trial import record_trade_shortlist_trial

    paths = native._files(tmp_path / "inputs", monkeypatch, day="2026-09-10")
    root = tmp_path / "trials"
    record_trade_shortlist_trial(
        *paths,
        output_root=root,
        now_fn=lambda: datetime.fromisoformat("2026-09-10T09:10:00+09:00"),
    )
    monkeypatch.setattr(
        evaluate,
        "load_recommendation_evidence",
        lambda path, **kwargs: native._evidence(path),
    )
    report = evaluate.evaluate_trade_shortlist_trials(
        root, now="2026-09-12T12:00:00+09:00", n_boot=100
    )
    assert report["coverage"]["paired_dates"] == 1
    assert report["dates"][0]["record_status"] == "committed"

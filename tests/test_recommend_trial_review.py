"""Post-label reports are derived, freshness-checked and never live selectors."""

import copy
import json

import pytest

from ops import recommend_trial_review as review
from ops.artifact_provenance import atomic_write_json, file_identity

NOW = "2026-09-30T02:00:00+00:00"


def _evaluation(module, now, *, state="prospective_comparable"):
    return review.boundary._seal(
        {
            "schema": module.EVALUATION_SCHEMA,
            "trial_id": module.TRIAL_ID,
            "config": module.TRIAL_CONFIG,
            "generated_at": now.isoformat(),
            "generator_sources": module._generator_sources(),
            "trial_inputs": [],
            "evidence_inputs": [],
            "inputs_unchanged": True,
            "deployable": False,
            "automatic_promotion": False,
            "effect_status": "descriptive_only_not_a_promotion_verdict",
            "coverage": {"prospective_start_asof": "2026-09-10"},
            "primary_paired": {"n_dates": 0, "metrics": None},
            "dates": [
                {"date": "2026-09-29", "status": state, "reason": None},
                {"date": "2026-09-30", "status": "pending", "reason": "label_missing"},
            ],
        }
    )


@pytest.fixture
def reports(monkeypatch):
    for module, name in (
        (review.boundary, "evaluate_microstructure_trials"),
        (review.shortlist, "evaluate_trade_shortlist_trials"),
    ):
        monkeypatch.setattr(
            module, name, lambda *, now, n_boot, module=module: _evaluation(module, now)
        )


def _write(tmp_path, document):
    path = tmp_path / "review.json"
    atomic_write_json(path, document)
    return path


def _reseal(document):
    document.pop("payload_sha256", None)
    return review.boundary._seal(document)


def test_today_pending_is_normal_but_yesterday_must_be_evaluated(tmp_path, reports):
    report = review.build_review(now=NOW, n_boot=1)
    assert report["status"] == "evaluated"
    assert report["summary"]["shortlist"]["awaiting_outcomes"] == ["2026-09-30"]
    result = review.inspect_review(_write(tmp_path, report), now=NOW)
    assert result["attention_required"] is False
    assert result["deployable"] is False
    assert result["forward"]["paired_dates"] == 0
    assert "regime forward: ready=0 paired=0" in review.format_review(result)


def test_old_unrepairable_forward_gap_is_audit_not_permanent_daily_alarm():
    report = {"dates": [
        {"date": "2026-10-01", "status": "missing", "record_status": "missing", "reason": "missing"},
        {"date": "2026-10-02", "status": "prospective_comparable", "record_status": "committed", "reason": None},
    ]}
    assert not review._forward_ready(report, "2026-10-01")
    assert review._forward_ready(report, "2026-10-02")
    assert not review._forward_ready(report, "2026-10-03")
    assert review._forward_ready(report, "2026-09-30")


@pytest.mark.parametrize("state", ["pending", "missing", "unavailable", "late"])
def test_mature_missing_or_unusable_evidence_is_not_operational_success(
    tmp_path, reports, monkeypatch, state
):
    monkeypatch.setattr(
        review.shortlist,
        "evaluate_trade_shortlist_trials",
        lambda *, now, n_boot: _evaluation(review.shortlist, now, state=state),
    )
    report = review.build_review(now=NOW, n_boot=1)
    assert report["status"] == "incomplete"
    result = review.inspect_review(_write(tmp_path, report), now=NOW)
    assert result["attention_required"] is True
    assert result["status"] == "incomplete"


@pytest.mark.parametrize(
    "now, attention",
    [
        ("2026-10-01T10:24:59+09:00", False),
        ("2026-10-01T10:25:00+09:00", True),
        ("2026-09-30T01:59:59+00:00", True),
    ],
)
def test_future_and_kst_freshness_boundary(tmp_path, reports, now, attention):
    path = _write(tmp_path, review.build_review(now=NOW, n_boot=1))
    assert review.inspect_review(path, now=now)["attention_required"] is attention


@pytest.mark.parametrize(
    "fault",
    [
        "hash",
        "summary",
        "date",
        "promotion",
        "source",
        "missing_trial",
        "native_config",
    ],
)
def test_report_contract_tampering_fails_closed(tmp_path, reports, fault):
    report = review.build_review(now=NOW, n_boot=1)
    if fault == "hash":
        report["status"] = "modified"
    elif fault == "summary":
        report["summary"]["shortlist"]["paired_dates"] += 1
    elif fault == "date":
        report["through_date"] = "2026-09-30"
    elif fault == "promotion":
        report["automatic_promotion"] = True
    elif fault == "source":
        report["generator_sources"] = []
    elif fault == "missing_trial":
        report["evaluations"].pop("shortlist")
    else:
        inner = copy.deepcopy(report["evaluations"]["shortlist"])
        inner["config"] = {}
        report["evaluations"]["shortlist"] = _reseal(inner)
    if fault != "hash":
        report = _reseal(report)
    assert review.inspect_review(_write(tmp_path, report), now=NOW)[
        "attention_required"
    ]


def test_read_probe_has_no_evaluation_or_writes(tmp_path, reports, monkeypatch):
    path = _write(tmp_path, review.build_review(now=NOW, n_boot=1))
    before = path.read_bytes()

    def forbidden(*args, **kwargs):
        pytest.fail("read probe must not evaluate or write")

    monkeypatch.setattr(review, "build_review", forbidden)
    monkeypatch.setattr(review, "atomic_write_json", forbidden)
    monkeypatch.setattr(review.boundary, "evaluate_microstructure_trials", forbidden)
    monkeypatch.setattr(review.shortlist, "evaluate_trade_shortlist_trials", forbidden)
    assert not review.inspect_review(path, now=NOW)["attention_required"]
    assert path.read_bytes() == before


def test_missing_symlink_and_invalid_json_are_not_healthy(tmp_path):
    missing = tmp_path / "missing.json"
    assert review.inspect_review(missing, now=NOW)["attention_required"]
    missing.write_text("not json")
    assert review.inspect_review(missing, now=NOW)["attention_required"]
    alias = tmp_path / "link.json"
    alias.symlink_to(missing)
    assert review.inspect_review(alias, now=NOW)["attention_required"]


def test_source_changes_are_reported_as_failure_not_success(tmp_path, monkeypatch):
    def changing(*args, **kwargs):
        raise RuntimeError("source changed during read")

    monkeypatch.setattr(review, "file_identity", changing)
    result = review.inspect_review(tmp_path / "review.json", now=NOW)
    assert result["attention_required"]
    assert "source changed" in result["reason"]


def test_failed_refresh_preserves_previous_complete_report(
    tmp_path, reports, monkeypatch, capsys
):
    path = _write(tmp_path, review.build_review(now=NOW, n_boot=1))
    before = path.read_bytes()

    def fail(*args, **kwargs):
        raise ValueError("synthetic evidence failure")

    monkeypatch.setattr(review, "build_review", fail)
    assert review.main(["--refresh", "--output", str(path)]) == 1
    assert path.read_bytes() == before
    assert "synthetic evidence failure" in capsys.readouterr().out


def test_refresh_does_not_publish_cross_generation_evidence(
    tmp_path, reports, monkeypatch
):
    evidence = tmp_path / "evidence.json"
    evidence.write_text("original")

    def first(*, now, n_boot):
        report = _evaluation(review.boundary, now)
        report["evidence_inputs"] = [file_identity(evidence, root=review.ROOT)]
        return _reseal(report)

    def second(*, now, n_boot):
        evidence.write_text("changed")
        return _evaluation(review.shortlist, now)

    monkeypatch.setattr(review.boundary, "evaluate_microstructure_trials", first)
    monkeypatch.setattr(review.shortlist, "evaluate_trade_shortlist_trials", second)
    with pytest.raises(ValueError, match="changed"):
        review.build_review(now=NOW, n_boot=1)


def test_performance_loss_is_not_an_operational_failure(tmp_path, reports, monkeypatch):
    def evaluated(*, now, n_boot):
        doc = _evaluation(review.shortlist, now)
        doc["primary_paired"] = {
            "n_dates": 2,
            "n_changed_dates": 1,
            "n_no_op_dates": 1,
            "changed_picks": 1,
            "picks_per_arm": 6,
            "metrics": {
                arm: {"mean": {"eod_return_net": value, "dn5": 0.5, "up10": 0.0}}
                for arm, value in (("control", 0.01), ("challenger", -0.01))
            },
        }
        return _reseal(doc)

    monkeypatch.setattr(review.shortlist, "evaluate_trade_shortlist_trials", evaluated)
    report = review.build_review(now=NOW, n_boot=1)
    result = review.inspect_review(_write(tmp_path, report), now=NOW)
    assert not result["attention_required"]
    assert "1.0000%->-1.0000%" in review.format_review(result)
    assert "no promotion" in review.format_review(result)
    assert "changed=1 noop=1 changed_picks=1" in review.format_review(result)


def test_cli_refresh_writes_only_designated_report(
    tmp_path, reports, monkeypatch, capsys
):
    built = review.build_review(now=NOW, n_boot=1)
    monkeypatch.setattr(review, "build_review", lambda: built)
    original = review.inspect_review
    monkeypatch.setattr(review, "inspect_review", lambda path: original(path, now=NOW))
    path = tmp_path / "review.json"
    assert review.main(["--refresh", "--output", str(path), "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "evaluated"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["review.json"]


def test_refresh_refuses_to_overwrite_other_artifacts(tmp_path, monkeypatch):
    path = tmp_path / "morning_evaluation.json"
    path.write_text('{"schema":"immutable_trial_evidence"}')
    before = path.read_bytes()

    def forbidden():
        pytest.fail("wrong destination must fail before costly evaluation")

    monkeypatch.setattr(review, "build_review", forbidden)
    assert review.main(["--refresh", "--output", str(path)]) == 1
    assert path.read_bytes() == before


def test_regime_swap_diagnostics_and_actual_label_availability(tmp_path):
    dates, daily = [], []
    means = {"up10": 0.0, "dn5": 0.0, "eod_return_net": 0.0, "mae": 0.0}
    for day, regime in (
        (10, "bull_quiet"),
        (11, "bull_volatile"),
        (12, "bull_volatile"),
    ):
        asof = f"2026-09-{day}"
        snapshot = tmp_path / f"snapshot-{day}.json"
        label = tmp_path / f"label-{day}.json"
        atomic_write_json(
            snapshot,
            {"btc_regime": regime, "decision_started_at": f"{asof}T09:08:00+09:00"},
        )
        rows = [{"coin": coin, **means} for coin in ("A", "B", "C", "D")]
        rows[2].update(up10=1, eod_return_net=0.10)
        rows[3].update(dn5=1, eod_return_net=-0.06, mae=-0.08)
        atomic_write_json(
            label,
            {
                "path_window_end": f"2026-09-{day + 1}T09:15:00+09:00",
                "labeled_at": f"2026-09-{day + 1}T10:12:00+09:00",
                "rows": rows,
            },
        )
        dates.append(
            {
                "date": asof,
                "control_top3": ["A", "B", "C"],
                "challenger_top3": ["A", "B", "D"],
                "evidence_manifest": {
                    "files": {
                        "snapshot": {"path": str(snapshot)},
                        "label": {"path": str(label)},
                    }
                },
            }
        )
        daily.append(
            {
                "date": asof,
                "control": means,
                "challenger": means,
                "challenger_minus_control": means,
            }
        )
    evaluation = {
        "dates": dates,
        "primary_paired": {"daily": daily},
        "metrics": list(means),
    }
    before = copy.deepcopy(evaluation)
    report = review.selection_diagnostics(evaluation)
    assert evaluation == before
    assert report["regime_date_counts"] == {"bull_quiet": 1, "bull_volatile": 2}
    assert report["swaps"]["added"]["n_picks"] == 3
    assert report["swaps"]["added"]["means"]["eod_return_net"] == pytest.approx(-0.06)
    assert report["swaps"]["removed"]["means"]["eod_return_net"] == pytest.approx(0.10)
    assert [
        row["eligible_prior_dates"] for row in report["past_only_availability"]
    ] == [0, 0, 1]
    assert report["past_only_availability"][-1]["latest_eligible_date"] == "2026-09-10"
    assert report["past_only_availability"][-1]["same_regime_prior_dates"] == 0
    assert report["automatic_promotion"] is False

import json
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest
import pandas as pd

from ops import season_shadow as s
from ops import policy_competition as competition
from scripts import label_recommend_snapshots as labels
from signals import recommend_live_snapshot as live
from test_recommend_live_snapshot import _inconsistent


CLOSE = datetime(2026, 11, 24, 1, 30, tzinfo=timezone.utc)
TARGET = "2026-11-25"


def row(i, state=s.STATES[0]):
    entry = datetime(2026, 11, 1, 0, 15, tzinfo=timezone.utc) + timedelta(days=i)
    before = (entry - timedelta(minutes=4)).isoformat()
    control = dict.fromkeys(s.policy.METRICS, 0.0)
    challenger = {
        **control,
        "up10": 1 / 3,
        "whole_path_safe_up10": 1 / 3,
        "eod_return_net": 0.01,
    }
    return dict(
        date=entry.date().isoformat(),
        version="A",
        context={"state": state},
        decision_at=before,
        predictors_available_at=[before],
        entry_at=entry.isoformat(),
        changed_picks=1,
        label_available_at=(entry + timedelta(days=1)).isoformat(),
        outcomes={"control": control, "challenger": challenger},
    )


def prepare(root, history, times=None):
    times = iter(times or [CLOSE, CLOSE + timedelta(seconds=1)])
    return s.prepare(
        root,
        {"generated_at": CLOSE.isoformat(), "payload_sha256": "a" * 64},
        history,
        clock=lambda: next(times),
    )


def test_table_agrees_with_full_frozen_selector_for_every_context():
    history = [row(i, s.STATES[i % 2]) for i in range(21)]
    history += [
        {**row(21), "version": "B"},
        {**row(22), "label_available_at": CLOSE.isoformat()},
    ]
    table = s.choice_table(history, CLOSE, TARGET)
    for version in ("A", "B"):
        for state in s.STATES:
            current = {**row(24, state), "outcomes": None, "version": version}
            expected = s.policy.selection_plans([*history[:-1], current])[-1]["choices"]
            assert table[version][state] == expected
    assert table["A"][s.STATES[0]]["recent"]["arm"] == "challenger"
    assert table["B"][s.STATES[0]]["recent"]["arm"] == "control"


def test_one_ticket_no_backdating_no_repair_or_overwrite(tmp_path):
    assert prepare(tmp_path, []) == TARGET
    ticket = s.read_ticket(tmp_path, TARGET)
    assert ticket["state"] == "committed"
    assert prepare(tmp_path, [row(0)]) == TARGET
    assert s.read_ticket(tmp_path, TARGET) == ticket
    assert s.read_ticket(tmp_path, "2026-11-24") is None
    (tmp_path / TARGET / "commit.json").unlink()
    assert s.read_ticket(tmp_path, TARGET)["state"] == "uncertain"
    with pytest.raises(ValueError, match="never complete"):
        prepare(tmp_path, [])
    with pytest.raises(ValueError, match="stale/future"):
        prepare(tmp_path, [], [CLOSE + timedelta(days=1)])


def test_no_commit_across_kst_midnight_or_for_tampered_plan(tmp_path):
    with pytest.raises(ValueError, match="day/clock"):
        prepare(tmp_path, [], [CLOSE, CLOSE + timedelta(days=1)])
    assert s.read_ticket(tmp_path, TARGET)["state"] == "uncertain"
    path = tmp_path / TARGET / "plan.json"
    plan = json.loads(path.read_text())
    plan["unseen"] = "challenger"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="invalid plan"):
        s.read_ticket(tmp_path, TARGET)


def test_join_native_selections_before_entry_and_never_replay_late(tmp_path):
    history = [row(i) for i in range(6)]
    prepare(tmp_path, history)
    ticket = s.read_ticket(tmp_path, TARGET)
    current = row(24)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            dict(
                delivery_ok=True,
                snapshot_id="x",
                slot="open",
                asof=TARGET,
                recorded_at=current["decision_at"],
            )
        )
    )
    audit = dict(
        evidence_manifest={"files": {"receipt": {"path": str(receipt)}}},
        snapshot_id="x",
        score_durable_observed_at=current["decision_at"],
        control_top3=["A", "B", "C"],
        challenger_top3=["A", "B", "D"],
    )
    result = s.score_day(ticket, audit, current, history)
    assert result["selected"]["recent"] == ["A", "B", "D"]
    assert result["changed"]["recent"] == 1
    other = s.score_day(ticket, audit, {**current, "version": "unseen"}, history)
    assert other["changed"]["recent"] == 0
    stats = s.summarize([result, other, {"status": "unavailable"}])["recent"]
    assert (stats["paired_dates"], stats["changed_dates"]) == (2, 1)
    assert stats["difference_all_dates"]["mean"]["up10"] == pytest.approx(1 / 6)
    late = s.score_day(
        ticket,
        {**audit, "score_durable_observed_at": current["entry_at"]},
        current,
        history,
    )
    assert late == {"status": "late"}
    with pytest.raises(ValueError, match="verified history"):
        s.score_day(ticket, audit, current, [])
    receipt.write_text("{}")
    with pytest.raises((ValueError, KeyError)):
        s.score_day(ticket, audit, current, history)


def test_batch_excludes_only_verified_delivery_only_and_does_not_emit_label(
    tmp_path, monkeypatch, capsys
):
    snap = live.get_or_create_recommend_snapshot(
        "2026-11-05", root=tmp_path, model_id="recommend_r1_open", scorer=_inconsistent
    )
    path = tmp_path / "2026-11-05/open_r1.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["label", "--snapshot", str(path), "--output-root", str(tmp_path / "labels")],
    )
    assert labels.main() == 2
    assert json.loads(capsys.readouterr().out)[0]["status"] == "research_excluded"
    assert not (tmp_path / "labels").exists()
    assert snap["probability_quality"]
    doc = json.loads(path.read_text())
    unknown = {**doc, "delivery_policy_source_sha256": "0" * 64}
    digest = live.native._document_digest(unknown)
    unknown.update(payload_sha256=digest, snapshot_id="recommend-" + digest[:20])
    path.write_text(json.dumps(unknown))
    assert labels.main() == 1
    assert json.loads(capsys.readouterr().out)[0]["status"] == "error"
    doc["probability_quality"]["candidates"].clear()
    path.write_text(json.dumps(doc))
    assert labels.main() == 1
    assert json.loads(capsys.readouterr().out)[0]["status"] == "error"


def test_stale_or_missing_review_rebuilds_only_native_not_broken_review(
    tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        s.forward.evaluator,
        "evaluate_trade_shortlist_trials",
        lambda **kw: calls.append(kw["now"]) or {"native": True},
    )
    path = tmp_path / "review.json"
    assert s.native_evaluation(path, CLOSE) == ({"native": True}, None)
    path.write_text(
        json.dumps(
            s.native._seal(dict(schema=s.review.SCHEMA, through_date="2026-11-21"))
        )
    )
    assert s.native_evaluation(path, CLOSE) == ({"native": True}, None)
    path.write_text("{}")
    with pytest.raises(ValueError):
        s.native_evaluation(path, CLOSE)
    assert calls == [CLOSE, CLOSE]


def test_real_close_interface_next_plan_probe_and_calendar_holes(tmp_path, monkeypatch):
    t = datetime(2026, 12, 3, 1, 30, tzinfo=timezone.utc)
    evaluation = {
        "generated_at": t.isoformat(),
        "payload_sha256": "a" * 64,
        "dates": [],
    }
    monkeypatch.setattr(s, "now", lambda: t)
    monkeypatch.setattr(s, "native_evaluation", lambda *args: (evaluation, None))
    monkeypatch.setattr(s.forward, "_evaluation_contract", lambda *args: t)
    monkeypatch.setattr(s.forward, "history_records", lambda *args, **kw: [])
    original = s.prepare
    monkeypatch.setattr(s, "prepare", lambda *args: original(*args, clock=lambda: t))
    result = s.refresh(root=tmp_path)
    assert result["next_plan"] == "2026-12-04"
    assert result["status_counts"] == {"missing_plan": 8}
    assert result["policies"]["recent"]["paired_dates"] == 0
    assert s.inspect(tmp_path) == result
    assert not (tmp_path / "2026-12-02").exists()
    monkeypatch.setattr(s, "now", lambda: t + timedelta(days=1))
    with pytest.raises(ValueError, match="missing/stale"):
        s.inspect(tmp_path)


def test_validated_delivery_only_date_is_not_reintroduced_from_ledger(tmp_path):
    ledger = tmp_path / "ledger.csv"
    ledger.write_text("date,status\n2026-11-04,closed\n2026-11-05,closed\n")
    spec = replace(
        next(m for m in competition.MODELS if m.id == "recommend_r1_open"),
        ledger_path=str(ledger),
    )
    snapshots = tmp_path / "snaps"
    live.get_or_create_recommend_snapshot(
        "2026-11-05", root=snapshots, model_id=spec.id, scorer=_inconsistent
    )
    excluded = []
    days, _ = competition._source_observed_dates(
        spec, pd.Timestamp("2026-11-24"), snapshot_root=snapshots, excluded=excluded
    )
    assert days == [pd.Timestamp("2026-11-04")]
    assert excluded == ["2026-11-05"]


@pytest.mark.parametrize("rc", [0, 1, 124])
def test_heartbeat_checks_next_plan_without_adding_telegram_incident(tmp_path, rc):
    from ops.operations_notifications import _issue_key
    from test_heartbeat_microstructure import _run

    probe = "-m ops.season_shadow --check"
    result, calls, log, message = _run(tmp_path, probe=probe, probe_rc=rc)
    assert result.returncode == 0 and calls.count(probe) == 1
    warning = "regime forward day-ahead plan/review unavailable"
    assert (warning in log) == bool(rc)
    assert _issue_key(warning) == "research:regime forward"
    if rc:
        assert message.count(warning) == 1
    else:
        assert message is None

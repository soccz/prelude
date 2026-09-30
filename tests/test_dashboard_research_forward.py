"""Derived forward display checks, isolated from live files and raw evaluation."""
import copy
from datetime import timedelta

import numpy as np
import pytest

from ops import dashboard_research as ui
from ops import recommend_regime_forward as forward
from signals.recommend_experiment_eval import METRICS, _summary
from test_dashboard_research import ASOF, NOW, project, source as source_fixture


@pytest.fixture
def source():
    return source_fixture.__wrapped__()


def fixture(n=5, changed=True):
    days, native = [], []
    for i in range(n):
        day = (forward.START + timedelta(days=i)).isoformat()
        base = dict(zip(METRICS, (.2, .3, .1, .002, .003 + i / 1000, -.04), strict=True))
        other = {**base, "eod_return_net": .01 + i / 1000, "up10": .4, "dn5": .2}
        native.append({"date": day, "control": base, "challenger": other, "changed_picks": 2})
        days.append({"date": day, "values": {p: other if changed and p == "recent" else base for p in ui.POLICIES},
                     "changed_picks": {p: 2 if changed and p == "recent" else 0 for p in ui.POLICIES}})
    generated = forward.START + timedelta(days=n)
    audits = [{"date": row["date"], "status": "prospective_comparable", "record_status": "committed",
               "eligibility": "ready", "reason": None, "changed_picks": row["changed_picks"],
               "choices": {p: {"arm": "challenger" if changed and p == "recent" else "control"}
                           for p in ui.POLICIES}} for row in days]
    audits.append({"date": generated.isoformat(), "status": "pending", "record_status": "committed",
                   "eligibility": "ready", "reason": "canonical_paired_outcomes_unavailable"})
    base = np.array([[r["values"]["fixed_r1"][m] for m in METRICS] for r in days])
    policies = {}
    for p in ui.POLICIES:
        values = np.array([[r["values"][p][m] for m in METRICS] for r in days])
        count = 2*n if p == "recent" and changed else 0
        policies[p] = {"n_dates": n, "changed_picks": count, "changed_dates": n if count else 0,
                       "metrics": _summary(values, 100, 42) if n else None,
                       "minus_fixed_r1": _summary(values-base, 100, 42) if n else None,
                       "effect_status": "descriptive_forward_effect" if count else "no_effect_observations"}
    report = {"schema": "recommend_regime_forward_evaluation.v1", "trial_id": forward.TRIAL_ID,
              "config": copy.deepcopy(forward.CONFIG), "deployable": False, "automatic_promotion": False,
              "scope": "pre_entry_frozen_policies_on_canonical_net_pick_proxies_not_portfolio",
              "n_boot": 100, "seed": 42, "dates": audits, "daily": days, "policies": policies,
              "committed_policy_records": n+1, "prospective_policy_records": n+1, "paired_dates": n}
    return report, {"primary_paired": {"daily": native}}, {
        "generated_at": generated.isoformat() + "T10:30:00+09:00",
        "through": (generated - timedelta(days=1)).isoformat(),
    }


@pytest.mark.parametrize("n", [0, 1, 4, 5])
@pytest.mark.parametrize("changed", [False, True])
def test_checked_forward_values_and_no_effect(n, changed):
    report, native, clock = fixture(n, changed)
    result = ui._checked_forward(report, native, **clock)
    item = result["recent"]
    assert item["n_dates"] == n and item["changed_picks"] == (2*n if changed else 0)
    assert (item["metrics"] is None) == (n == 0)
    assert (item["difference_block_ci95"] is None) == (n < 5)
    if n:
        assert item["minus_fixed_r1"]["eod_return_net"] == pytest.approx(.007 if changed else 0)
    if n >= 5 and not changed:
        assert all(v == [0, 0] for v in item["difference_block_ci95"].values())


@pytest.mark.parametrize("fault", ["mean", "ci", "count", "date", "values", "no_op", "fixed", "budget", "future"])
def test_forged_report_cannot_be_displayed(fault):
    report, native, clock = fixture()
    if fault == "mean":
        report["policies"]["recent"]["metrics"]["mean"]["up10"] = .9
    elif fault == "ci":
        report["policies"]["recent"]["minus_fixed_r1"]["observed_date_block3_ci95"]["up10"] = [0, 0]
    elif fault == "count":
        report["paired_dates"] = True
    elif fault == "date":
        report["daily"][0]["date"] = "2026-09-30"
    elif fault == "values":
        report["daily"][0]["values"]["recent"] = {**report["daily"][0]["values"]["recent"], "dn5": .9}
    elif fault == "no_op":
        report["daily"][0]["changed_picks"]["recent"] = 0
    elif fault == "fixed":
        report["dates"][0]["choices"]["fixed_r1"]["arm"] = "challenger"
    elif fault == "budget":
        report["n_boot"] = 10**9
    else:
        clock["through"] = "2026-10-02"
    with pytest.raises((ValueError, KeyError)):
        ui._checked_forward(report, native, **clock)


def test_legacy_assets_continue_to_validate(source):
    payload = project(source)
    payload["schema"] = ui.LEGACY_SCHEMA
    del payload["forward"]["policies"]
    ui.validate_research_progress(payload, asof=ASOF, now=NOW)


@pytest.mark.parametrize("fault", ["read_race", "mixed_summary", "mixed_counter"])
def test_read_binds_inspector_and_derived_report(monkeypatch, source, fault):
    report, native, clock = fixture(0)
    # Use a prelaunch empty report with the source fixture's September clock.
    report.update(dates=[], committed_policy_records=0, prospective_policy_records=0)
    document = {**source, "forward_evaluation": report, "evaluations": {"shortlist": native}}
    del source["forward"]["policies"]
    identities = iter([{"sha256": "before"}, {"sha256": "changed" if fault == "read_race" else "before"}])
    monkeypatch.setattr(ui, "file_identity", lambda *a, **k: next(identities))
    monkeypatch.setattr(ui, "strict_json_object", lambda *a: document)
    from ops import recommend_trial_review
    monkeypatch.setattr(recommend_trial_review, "inspect_review", lambda *a, **k: source)
    if fault == "mixed_summary":
        document["summary"] = {}
    elif fault == "mixed_counter":
        source["forward"]["paired_dates"] = 1
    with pytest.raises(ValueError):
        ui.read_review(now=NOW)


def test_actual_native_forward_chain(tmp_path, monkeypatch):
    from test_recommend_regime_forward import _prepare, _publish, _label, _evaluation, _at
    path, root, _ = _prepare(tmp_path, monkeypatch, history=True)
    _publish(path, root, tmp_path)
    _label(tmp_path, monkeypatch)
    now = _at("12:00:00", "2026-10-02")
    native = _evaluation(tmp_path, now)
    report = forward.evaluate_forward(native, root=root, now=now, n_boot=10)
    result = ui._checked_forward(report, native, generated_at=now.isoformat(), through="2026-10-01")
    assert result["recent"]["changed_picks"] == 1
    assert result["recent"]["minus_fixed_r1"]["eod_return_net"] == pytest.approx(.17/3)


@pytest.mark.parametrize("fault", [None, "extra", "count", "mean", "ci", "no_effect", "nan"])
def test_policy_publication_contract(source, fault):
    from datetime import datetime
    report, native, clock = fixture()
    now = datetime.fromisoformat(clock["generated_at"])
    source.update(checked_at=now.isoformat(), generated_at=now.isoformat(), through_date=clock["through"])
    source["forward"].update(prospective_policy_records=6, paired_dates=5,
        changed_picks={name: row["changed_picks"] for name, row in report["policies"].items()},
        policies=ui._checked_forward(report, native, **clock))
    payload = ui.project_review(source, asof=now.date().isoformat(), now=now)
    recent = payload["forward"]["policies"]["recent"]
    if fault == "extra":
        recent["private_reason"] = "private"
    elif fault == "count":
        recent["picks_per_arm"] = 14
    elif fault == "mean":
        recent["minus_fixed_r1"]["dn5"] = .9
    elif fault == "ci":
        recent["difference_block_ci95"]["up10"] = [1, 0]
    elif fault == "no_effect":
        payload["forward"]["policies"]["fixed_r1"]["difference_block_ci95"]["dn5"] = [-.1, .1]
    elif fault == "nan":
        recent["metrics"]["eod_return_net"] = float("nan")
    if fault:
        with pytest.raises((ValueError, KeyError, TypeError)):
            ui.validate_research_progress(payload, asof=now.date().isoformat(), now=now)
    else:
        ui.validate_research_progress(payload, asof=now.date().isoformat(), now=now)

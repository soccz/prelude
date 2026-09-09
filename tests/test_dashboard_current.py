from __future__ import annotations

import copy
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from ops import dashboard_current as current

NOW = datetime.fromisoformat("2026-09-09T14:00:00+09:00")
ASOF = NOW.date().isoformat()


@pytest.fixture
def sources(monkeypatch):
    live = {
        "asof": ASOF,
        "slots": {
            slot: {
                "state": "delivered_candidates",
                "candidate_count": 3,
                "sent_at": f"{ASOF}T{hour}:00+09:00",
                "decision_completed_at": f"{ASOF}T{hour}:00+09:00",
                "receipt_path": "/private/receipt",
                "message_id": "PRIVATE_TOKEN",
            }
            for slot, hour in (("preopen", "08:53"), ("open", "09:08"))
        },
    }
    probes = {}
    for name, (module, schema, start) in current.PROBES.items():
        probes[module] = {
            "schema": schema,
            "asof": ASOF,
            "checked_at": NOW.isoformat(),
            "status": "evidence_invalid" if name == "microstructure" else "not_started",
            "attention_required": name == "microstructure",
            "effect_status": "not_evaluated",
            "deployable": False,
            "reason": "SECRET private /home/path incident",
        }
    monkeypatch.setattr(current, "build_recommendation_status", lambda *a, **k: live)
    monkeypatch.setattr(current, "_run_probe", lambda module, now: probes[module])
    return live, probes


def test_current_projection_separates_delivery_research_and_history(sources):
    result = current.build_current_system(asof=ASOF, now=NOW)
    current.validate_current_system(result, asof=ASOF, now=NOW)
    assert result["live"]["preopen"]["state"] == "delivered_candidates"
    assert result["research"]["microstructure"]["state"] == "evidence_invalid"
    assert result["research"]["trade_shortlist"]["state"] == "not_started"
    assert result["research"]["trade_shortlist"]["effect_status"] == "not_evaluated"
    assert result["legacy"]["pump_v2"] == "terminal_kill"
    encoded = json.dumps(result)
    assert all(
        term not in encoded
        for term in ("PRIVATE_TOKEN", "SECRET", "/private", "/home", "paired_dates")
    )


def test_missing_delivery_is_unknown_not_zero(sources):
    sources[0]["slots"]["preopen"].update(
        state="missing_decision",
        candidate_count=None,
        sent_at=None,
        decision_completed_at=None,
    )
    result = current.build_current_system(asof=ASOF, now=NOW)
    assert result["live"]["preopen"]["candidate_count"] is None
    assert result["live"]["preopen"]["attention_required"] is True


def test_research_timeout_does_not_hide_live_delivery(sources, monkeypatch):
    def slow(*a):
        raise subprocess.TimeoutExpired("private command", 30)

    monkeypatch.setattr(current, "_run_probe", slow)
    result = current.build_current_system(asof=ASOF, now=NOW)
    assert result["live"]["open"]["state"] == "delivered_candidates"
    assert all(r["state"] == "probe_unavailable" for r in result["research"].values())
    assert "private command" not in json.dumps(result)


def test_waiting_keeps_previous_due_problem(sources):
    source = sources[1][current.PROBES["microstructure"][0]]
    previous = copy.deepcopy(source)
    previous["asof"] = "2026-09-08"
    source.update(status="waiting", previous_due=previous)
    result = current.build_current_system(asof=ASOF, now=NOW)
    assert result["research"]["microstructure"]["previous_due"] == {
        "asof": "2026-09-08",
        "state": "evidence_invalid",
        "attention_required": True,
    }


def test_historical_asof_does_not_relabel_today_as_history(sources, monkeypatch):
    def forbidden(*a):
        pytest.fail("historical build must not run today's probe")

    monkeypatch.setattr(current, "_run_probe", forbidden)
    sources[0]["asof"] = "2026-09-08"
    result = current.build_current_system(asof="2026-09-08", now=NOW)
    assert all(r["state"] == "probe_unavailable" for r in result["live"].values())
    assert all(
        r["state"] == "historical_not_observed" for r in result["research"].values()
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(private_notes="must not publish"),
        lambda p: p.update(automatic_orders=True),
        lambda p: p["live"]["open"].update(message_id=12),
        lambda p: p["live"]["open"].update(candidate_count=True),
        lambda p: p["live"]["open"].update(candidate_count=None),
        lambda p: p["research"]["microstructure"].update(reason="secret"),
        lambda p: p["research"]["trade_shortlist"].update(effect_status="success"),
        lambda p: p["research"]["trade_shortlist"].update(deployable=True),
        lambda p: p["research"]["trade_shortlist"].update(state="<script>"),
        lambda p: p.update(observed_at=(NOW + timedelta(seconds=1)).isoformat()),
        lambda p: p.update(observed_at=(NOW - timedelta(hours=7)).isoformat()),
    ],
)
def test_contract_rejects_private_fields_false_success_and_bad_clock(sources, mutation):
    result = current.build_current_system(asof=ASOF, now=NOW)
    mutation(result)
    with pytest.raises((ValueError, TypeError)):
        current.validate_current_system(result, asof=ASOF, now=NOW)


@pytest.mark.parametrize(
    "change",
    [
        {"status": "unknown"},
        {"asof": "2026-09-08"},
        {"checked_at": "2026-09-10T14:00:00+09:00"},
        {"deployable": True},
        {"effect_status": "proven"},
    ],
)
def test_invalid_research_is_unavailable(sources, change):
    sources[1][current.PROBES["microstructure"][0]].update(change)
    result = current.build_current_system(asof=ASOF, now=NOW)
    assert result["research"]["microstructure"]["state"] == "probe_unavailable"


def test_probe_is_bounded_read_only_and_accepts_attention_exit(monkeypatch):
    def run(command, **kwargs):
        assert command[1:4] == ["-B", "-m", "ops.recommend_microstructure_status"]
        assert kwargs["timeout"] == 30 and kwargs["cwd"] == current.ROOT
        assert kwargs["capture_output"] is True
        return SimpleNamespace(returncode=1, stdout=b'{"attention_required":true}')

    monkeypatch.setattr(current.subprocess, "run", run)
    assert current._run_probe(current.PROBES["microstructure"][0], NOW) == {
        "attention_required": True
    }


def test_builder_never_reads_private_diary_in_main():
    # Keep the offline parser available, but no call to it from publishing.
    import ast
    from scripts import build_dashboard

    tree = ast.parse(Path(build_dashboard.__file__).read_text())
    main = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    calls = {
        n.func.id
        for n in ast.walk(main)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "parse_notes_md" not in calls
    assert "compute_notes_vs_system" not in calls
    assert "build_current_system" in calls


@pytest.mark.parametrize("slot", ["preopen", "open"])
@pytest.mark.parametrize(
    "change",
    [
        {"state": "broken"},
        {"candidate_count": None},
        {"candidate_count": 0},
        {"sent_at": None},
        {"decision_completed_at": None},
        {"sent_at": "2026-09-08T08:53:00+09:00"},
        {"sent_at": "2026-09-09T08:40:00+09:00"},
        {"sent_at": "2026-09-09T15:00:00+09:00"},
    ],
)
def test_invalid_slot_is_unavailable_without_hiding_other_slot(sources, slot, change):
    sources[0]["slots"][slot].update(change)
    original = copy.deepcopy(sources)
    result = current.build_current_system(asof=ASOF, now=NOW)
    other = "open" if slot == "preopen" else "preopen"
    assert result["live"][slot]["state"] == "probe_unavailable"
    assert result["live"][slot]["candidate_count"] is None
    assert result["live"][other]["state"] == "delivered_candidates"
    assert sources == original


def test_missing_first_slot_does_not_hide_valid_second_slot(sources):
    del sources[0]["slots"]["preopen"]
    result = current.build_current_system(asof=ASOF, now=NOW)
    assert result["live"]["preopen"]["state"] == "probe_unavailable"
    assert result["live"]["open"]["state"] == "delivered_candidates"


@pytest.mark.parametrize("seconds", [0, 1])
def test_delivery_clock_uses_native_server_second_precision(sources, seconds):
    sources[0]["slots"]["open"].update(
        sent_at="2026-09-09T09:08:00+09:00",
        decision_completed_at=f"2026-09-09T09:08:0{seconds}.999999+09:00",
    )
    result = current.build_current_system(asof=ASOF, now=NOW)
    assert result["live"]["open"]["state"] == (
        "delivered_candidates" if seconds == 0 else "probe_unavailable"
    )


@pytest.mark.parametrize(
    "change",
    [
        {"status": "evidence_invalid", "attention_required": False},
        {"status": "complete_changed", "attention_required": True},
        {"status": "complete_noop", "attention_required": True},
        {"status": "complete_unavailable", "attention_required": True},
        {"status": "waiting", "attention_required": False},
    ],
)
def test_research_state_attention_contradiction_is_unavailable(sources, change):
    sources[1][current.PROBES["microstructure"][0]].update(change)
    result = current.build_current_system(asof=ASOF, now=NOW)
    assert result["research"]["microstructure"]["state"] == "probe_unavailable"
    assert result["live"]["open"]["state"] == "delivered_candidates"


@pytest.mark.parametrize("previous_attention, outer_attention", [(True, False), (False, True)])
def test_waiting_cannot_hide_or_invent_previous_attention(
    sources, previous_attention, outer_attention
):
    source = sources[1][current.PROBES["microstructure"][0]]
    previous = copy.deepcopy(source)
    previous.update(
        asof="2026-09-08",
        status="evidence_invalid" if previous_attention else "complete_noop",
        attention_required=previous_attention,
    )
    source.update(status="waiting", attention_required=outer_attention, previous_due=previous)
    result = current.build_current_system(asof=ASOF, now=NOW)
    assert result["research"]["microstructure"]["state"] == "probe_unavailable"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p["live"]["open"].update(attention_required=True),
        lambda p: p["live"]["open"].update(sent_at="2026-09-08T09:08:00+09:00"),
        lambda p: p["live"]["open"].update(decision_completed_at="2026-09-09T09:09:00+09:00"),
        lambda p: p["research"]["microstructure"].update(attention_required=False),
        lambda p: p["research"]["microstructure"].update(checked_at=None),
    ],
)
def test_publisher_validator_independently_rejects_semantic_contradictions(sources, mutation):
    result = current.build_current_system(asof=ASOF, now=NOW)
    mutation(result)
    with pytest.raises((ValueError, TypeError)):
        current.validate_current_system(result, asof=ASOF, now=NOW)


@pytest.mark.parametrize("returncode, attention", [(0, True), (1, False), (0, 0), (1, 1), (2, True)])
def test_probe_exit_and_attention_must_agree(monkeypatch, returncode, attention):
    monkeypatch.setattr(
        current.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=returncode,
            stdout=json.dumps({"attention_required": attention}).encode(),
        ),
    )
    with pytest.raises(ValueError):
        current._run_probe(current.PROBES["microstructure"][0], NOW)

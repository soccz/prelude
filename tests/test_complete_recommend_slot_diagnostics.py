"""Frozen-report supplement tests; no database, model, or network execution."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.complete_recommend_slot_diagnostics as cli
from ops.artifact_provenance import file_set_identity, sha256_file


def _outcome(coin, start):
    value = (ord(coin) - 64) * .001 + start.minute * .0001
    return {"actual_entry_open": 100.0 + ord(coin) + start.minute,
            "mfe": .04, "mae": -.02, "eod_return_gross": value, "eod_return_net": value - .0015,
            "up5": False, "up10": False, "up20": False, "dn3": False, "dn5": False, "dn10": False,
            "tp5_before_sl3": None, "tp5_sl3_first_passage": "neither",
            "tp5_sl3_return_gross": value, "tp5_sl3_return_net": value - .0015,
            "first_passage_bar": None, "first_passage_at": None}


def _stamp(day, minutes=0):
    return pd.Timestamp(f"{day}T09:00:00+09:00") + pd.Timedelta(minutes=minutes)


@pytest.fixture
def case():
    evidence, manifests, records = {}, [], []
    for day in ("2026-09-01", "2026-09-02"):
        for slot in ("preopen", "open"):
            sid = f"snapshot-{day}-{slot}"
            start = _stamp(day, 15 if slot == "open" else 0)
            coins = ["B", "C", "D"] if day.endswith("02") and slot == "open" else ["A", "B", "C"]
            candidates = [{"coin": coin, "rank": rank} for rank, coin in enumerate(coins, 1)]
            snapshot = {"snapshot_id": sid, "asof": day, "slot": slot,
                        "top3": deepcopy(candidates), "universe": deepcopy(candidates)}
            label = {"execution_start_at": start.isoformat(), "path_window_end": (start + pd.Timedelta(days=1)).isoformat(),
                     "rows": [{**row, "label_status": "labeled", **_outcome(row["coin"], start)} for row in candidates]}
            manifest = {"snapshot_id": sid, "files": {}, "label_available_at": "2026-09-03T03:00:00+00:00"}
            evidence[sid] = {"snapshot": snapshot, "label": label, "receipt": {}, "manifest": deepcopy(manifest)}
            manifests.append(manifest)
            for candidate in candidates:
                cells = {}
                for minutes in (0, 15, 30):
                    begin = start + pd.Timedelta(minutes=minutes)
                    cells[str(minutes)] = {"start_at": begin.isoformat(), "end_at": (begin + pd.Timedelta(days=1)).isoformat(),
                                           "path_complete": True, "label_status": "labeled", "outcomes": _outcome(candidate["coin"], begin)}
                records.append({"snapshot_id": sid, "asof": day, "slot": slot, **candidate,
                                "delays": cells, "common_eligible": True, "primary_eligible": True, "canonical_parity": "passed"})
    delays = {"schema": "recommend_entry_delay.v1", "status": "ok", "errors": [], "inputs": deepcopy(manifests),
              "created_at": "2026-09-03T13:00:01+09:00", "records": records,
              "contract": {"ranking": "R1", "cohort": "actual_delivered_snapshot_top3", "delays_minutes": [0, 15, 30],
                           "horizon": "each_start_plus_24h", "round_trip_cost_fraction": .0015, "return_unit": "fraction"}}
    # Only day 1 had all preopen picks in the original open universe.
    day = "2026-09-01"
    p = np.array([0., 0., 0., .0005, .0005, -.02])
    q = np.array([0., 0., 0., .002, .002, -.02])
    original = {"date": day, "P": ["A", "B", "C"], "Q": ["A", "B", "C"], "R": ["A", "B", "C"],
                "preopen_snapshot_id": f"snapshot-{day}-preopen", "open_snapshot_id": f"snapshot-{day}-open",
                "metrics": cli.slot._named(np.stack((p, q, q, q - p, q - q, q - p)))}
    diagnostics = {"schema": "recommend_selection_diagnostics.v1",
                   "data_provenance": {"frozen_as_of": "2026-09-03T04:00:00+00:00", "input_manifests": deepcopy(manifests)},
                   "slots": {"daily": [original], "cohort": {"date_list": [day]}}}
    return diagnostics, delays, evidence


def _record(case, *, day="2026-09-02", slot="preopen", coin="A"):
    return next(row for row in case[1]["records"] if row["asof"] == day and row["slot"] == slot and row["coin"] == coin)


def test_repairs_coverage_without_synthetic_candidates_or_mutation(case):
    before = deepcopy(case)
    report = cli.complete_slots(*case)
    assert report["cohort"]["dates"] == 2 and report["cohort"]["pick_rows_per_leg"] == 6
    assert report["coverage"]["original_evaluable_dates"] == 1
    assert report["parity"]["canonical_zero_picks_checked"] == 12
    assert report["parity"]["Q_open_overlap_picks_checked"] == 5
    assert report["parity"]["original_daily_dates_checked"] == 1
    assert report["daily"][1]["P"] == report["daily"][1]["Q"] == ["A", "B", "C"]
    assert report["daily"][1]["R"] == ["B", "C", "D"]
    assert all(row["delay_minutes"] == 15 for day in report["daily"] for row in day["Q_sources"])
    assert report["source_cutoffs"]["diagnostic_frozen_as_of"] != report["source_cutoffs"]["delay_created_at"]
    assert report["config"]["raw_OHLC_reverified"] is False
    assert case == before
    assert len(report["leave_one_date_out"]) == 2
    json.dumps(report, allow_nan=False)


def test_only_required_window_matters_not_all_delay_eligibility(case):
    record = _record(case)
    record["common_eligible"] = record["primary_eligible"] = False
    cell = record["delays"]["30"]
    cell.update(path_complete=False, label_status="path_incomplete")
    cell.pop("outcomes")
    assert cli.complete_slots(*case)["cohort"]["dates"] == 2


@pytest.mark.parametrize("kind", ["missing_Q", "unavailable_Q"])
def test_only_required_missing_Q_excludes_entire_day(case, kind):
    record = _record(case)
    if kind == "missing_Q":
        record["delays"].pop("15")
    else:
        record["delays"]["15"].update(path_complete=False, label_status="path_incomplete")
        record["delays"]["15"].pop("outcomes")
    report = cli.complete_slots(*case)
    assert report["cohort"]["date_list"] == ["2026-09-01"]
    assert report["coverage"]["excluded"][0]["coins"] == ["A"]


@pytest.mark.parametrize("kind", ["duplicate_record", "missing_record", "extra_record", "rank_bool", "duplicate_input", "missing_input"])
def test_identity_mismatches_block_whole_supplement(case, kind):
    _, delays, _ = case
    if kind == "duplicate_record":
        delays["records"].append(deepcopy(delays["records"][0]))
    elif kind == "missing_record":
        delays["records"].pop()
    elif kind == "extra_record":
        extra = deepcopy(delays["records"][0])
        extra["coin"] = "X"
        delays["records"].append(extra)
    elif kind == "rank_bool":
        delays["records"][0]["rank"] = True
    elif kind == "duplicate_input":
        delays["inputs"].append(deepcopy(delays["inputs"][0]))
    else:
        delays["inputs"].pop()
    with pytest.raises(ValueError):
        cli.complete_slots(*case)


@pytest.mark.parametrize("kind", ["start", "end", "unknown_delay", "duplicate_window", "future_window"])
def test_exact_window_contract_prevents_backward_or_future_reuse(case, kind):
    record = _record(case)
    if kind == "start":
        record["delays"]["15"]["start_at"] = _stamp("2026-09-02", 30).isoformat()
    elif kind == "end":
        record["delays"]["15"]["end_at"] = _stamp("2026-09-03", 30).isoformat()
    elif kind == "unknown_delay":
        record["delays"]["-15"] = deepcopy(record["delays"]["0"])
    elif kind == "duplicate_window":
        record["delays"]["30"] = deepcopy(record["delays"]["15"])
    else:
        case[1]["created_at"] = "2026-09-03T09:20:00+09:00"
    with pytest.raises(ValueError):
        cli.complete_slots(*case)


@pytest.mark.parametrize("kind", ["canonical_zero", "Q_overlap", "missing_field", "old_daily", "rebound_manifest"])
def test_parity_is_checked_directly_not_by_saved_pass_flag(case, kind):
    if kind == "canonical_zero":
        _record(case)["delays"]["0"]["outcomes"]["actual_entry_open"] += 1
    elif kind == "Q_overlap":
        _record(case, coin="B")["delays"]["15"]["outcomes"]["actual_entry_open"] += 1
    elif kind == "missing_field":
        _record(case)["delays"]["0"]["outcomes"].pop("first_passage_at")
    elif kind == "old_daily":
        case[0]["slots"]["daily"][0]["metrics"]["Q"]["mae"] = -.99
    else:
        next(iter(case[2].values()))["manifest"]["label_available_at"] = "changed"
    with pytest.raises(ValueError):
        cli.complete_slots(*case)


def test_Q_overlap_checked_even_if_another_pick_already_excludes_day(case):
    _record(case)["delays"].pop("15")
    _record(case, coin="B")["delays"]["15"]["outcomes"]["actual_entry_open"] += 1
    with pytest.raises(ValueError, match="Q open overlap"):
        cli.complete_slots(*case)


def test_record_order_does_not_change_results(case):
    expected = cli.complete_slots(*case)
    case[1]["records"].reverse()
    case[1]["inputs"].reverse()
    actual = cli.complete_slots(case[0], case[1], dict(reversed(list(case[2].items()))))
    assert actual == expected


@pytest.fixture
def stored(case, tmp_path, monkeypatch):
    diagnostics, delays, evidence = case
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "SOURCES", ("supplement.py",))
    monkeypatch.setattr(cli, "SOURCE_FILES", ("delay.py",))
    for name in ("supplement.py", "delay.py", "diagnostics.py"):
        (tmp_path / name).write_text("# frozen source", encoding="utf-8")
    manifests, all_files, records = [], {}, {}
    for sid, item in evidence.items():
        snap = item["snapshot"]
        paths = {role: tmp_path / role / snap["asof"] / f"{snap['slot']}_r1.json" for role in ("snapshot", "label", "receipt")}
        for role, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
            all_files[f"{sid}:{role}"] = path
        item["manifest"]["files"] = file_set_identity(paths, root=tmp_path)
        manifests.append(deepcopy(item["manifest"]))
        records[paths["snapshot"]] = item
    diagnostics["data_provenance"]["input_manifests"] = deepcopy(manifests)
    diagnostics["data_provenance"]["files"] = file_set_identity(all_files, root=tmp_path)
    diagnostics["generator_files"] = file_set_identity({"diagnostics.py": tmp_path / "diagnostics.py"}, root=tmp_path)
    delays["inputs"] = deepcopy(manifests)
    delays["code"] = file_set_identity({"delay.py": tmp_path / "delay.py"}, root=tmp_path)
    paths = {name: tmp_path / f"{name}.json" for name in ("diagnostics", "delays", "design")}

    def save(name, report):
        body = {key: value for key, value in report.items() if key != "report_payload_sha256"}
        report["report_payload_sha256"] = cli.sha256_bytes(cli.canonical_json_bytes(body))
        paths[name].write_text(json.dumps(report), encoding="utf-8")

    save("diagnostics", diagnostics)
    save("delays", delays)
    design = {"schema": cli.DESIGN_SCHEMA, "design_id": cli.DESIGN_ID, "scope": cli.SCOPE,
              "input_sha256": {name: sha256_file(paths[name]) for name in ("diagnostics", "delays")},
              "configuration": deepcopy(cli.SUPPLEMENT_CONFIG)}
    paths["design"].write_text(json.dumps(design), encoding="utf-8")
    calls = []

    def load(path, **kwargs):
        calls.append((path, kwargs))
        return deepcopy(records[path])

    monkeypatch.setattr(cli, "load_recommendation_evidence", load)
    monkeypatch.setattr("requests.sessions.Session.request", lambda *a, **k: pytest.fail("network forbidden"))
    return paths, design, all_files, calls


def _run(stored):
    paths = stored[0]
    return cli.run_supplement(paths["diagnostics"], paths["delays"], paths["design"])


def test_run_rebinds_reports_sources_and_evidence_without_writes(stored):
    paths, _, files, calls = stored
    before = {path: path.read_bytes() for path in (*paths.values(), *files.values())}
    report = _run(stored)
    assert len(calls) == 4 and report["cohort"]["dates"] == 2
    assert all(kwargs["now"] == pd.Timestamp("2026-09-03T04:00:00+00:00") for _, kwargs in calls)
    cli._checksum(report)
    assert before == {path: path.read_bytes() for path in before}


@pytest.mark.parametrize("target", ["diagnostics", "delays", "design", "evidence", "code"])
def test_hash_changes_fail_before_rebinding(stored, target):
    paths, _, evidence, calls = stored
    path = next(iter(evidence.values())) if target == "evidence" else paths["design"].parent / "delay.py" if target == "code" else paths[target]
    path.write_bytes(path.read_bytes() + b" ")
    if target == "design":
        document = json.loads(path.read_text())
        document["configuration"]["seed"] = 11
        path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        _run(stored)
    assert not calls


def test_rehashed_design_cannot_hide_a_broken_report_payload(stored):
    paths, design, _, _ = stored
    report = json.loads(paths["delays"].read_text())
    report["records"].pop()
    paths["delays"].write_text(json.dumps(report))
    design["input_sha256"]["delays"] = sha256_file(paths["delays"])
    paths["design"].write_text(json.dumps(design))
    with pytest.raises(ValueError, match="checksum"):
        _run(stored)


@pytest.mark.parametrize("target", ["evidence", "source", "report"])
def test_changes_during_replay_block_result(stored, monkeypatch, target):
    original = cli.complete_slots
    paths, _, evidence, _ = stored
    path = next(iter(evidence.values())) if target == "evidence" else paths["design"].parent / "supplement.py" if target == "source" else paths["delays"]

    def changed(*args):
        result = original(*args)
        path.write_bytes(path.read_bytes() + b" ")
        return result

    monkeypatch.setattr(cli, "complete_slots", changed)
    with pytest.raises(ValueError, match="during"):
        _run(stored)


def test_cli_uses_exclusive_output_writer_and_protects_both_reports_and_design(stored, monkeypatch, capsys):
    paths = stored[0]
    calls = []
    monkeypatch.setattr(cli, "write_new_report", lambda *a, **k: calls.append((a, k)))
    args = [part for key, path in paths.items() for part in (f"--{key}", str(path))]
    assert cli.main([*args, "--output", "_workspace/new.json"]) == 0
    assert calls[0][0][0] == Path("_workspace/new.json")
    assert calls[0][1]["protected_roots"] == tuple(paths.values())
    assert json.loads(capsys.readouterr().out)["deployable"] is False


def test_cli_failure_never_publishes_output(stored, monkeypatch, capsys):
    paths = stored[0]
    paths["delays"].write_text("changed")
    monkeypatch.setattr(cli, "write_new_report", lambda *a, **k: pytest.fail("must not publish"))
    args = [part for key, path in paths.items() for part in (f"--{key}", str(path))]
    assert cli.main([*args, "--output", "_workspace/new.json"]) == 2
    assert json.loads(capsys.readouterr().err)["status"] == "blocked"

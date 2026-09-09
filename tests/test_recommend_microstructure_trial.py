"""Synthetic-only prospective trial and durable publication fault tests."""
from __future__ import annotations

import copy
import gzip
import json
import os
import stat
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from threading import Barrier

import pytest
import requests

import test_recommend_microstructure as fixtures
import signals.recommend_microstructure as features
import signals.recommend_microstructure_trial as trial
import signals.recommend_snapshot as snapshots
from ops.artifact_provenance import canonical_json_bytes, file_identity, sha256_bytes
from ops.recommendation_evidence import EvidenceError, EvidenceUnavailable
from scripts.evaluate_recommend_microstructure_trial import main


@pytest.fixture(autouse=True)
def no_live_operations(monkeypatch):
    monkeypatch.setenv("PRELUDE_FORBID_TELEGRAM", "1")

    def forbidden(*args, **kwargs):
        pytest.fail("trial attempted network, fitting or DB metadata collection")

    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    for name in ("_data_metadata", "_git_code_metadata", "_environment_metadata"):
        monkeypatch.setattr(snapshots, name, forbidden)


def _sample(monkeypatch, *, day="2026-09-08", opportunity=True, missing=False, changed=True):
    cutoff = features._ns(day + "T09:05:00.123456+09:00")
    for name, value in {"DAY": day, "CUTOFF": cutoff, "START": cutoff-300_000_000_000,
                        "WARMUP": cutoff-900_000_000_000}.items():
        monkeypatch.setattr(fixtures, name, value)
    trades = [fixtures._trade(coin=coin, trade_id=i+1,
                             side="ASK" if changed and i == 2 else "BID")
              for i, coin in enumerate(fixtures.MARKETS) if not (missing and i == 3)]
    snapshot, manifest, records = fixtures._sample(trades)
    for row in snapshot["universe"]:
        row["p_up10"] = .2 if row["rank"] < 3 else .1 if row["rank"] < 5 else .05
        row["rr_ratio"] = row["p_up10"] / row["p_dn5"]
    if not opportunity:
        snapshot["universe"][3]["p_dn10"] += .001
    snapshot["top3"] = copy.deepcopy(snapshot["universe"][:3])
    snapshot["training"]["cutoff_exclusive"] = (datetime.fromisoformat(day)-timedelta(days=5)).date().isoformat()
    snapshot["training"]["end"] = (datetime.fromisoformat(day)-timedelta(days=6)).date().isoformat()
    digest = snapshots._document_digest(snapshot)
    snapshot.update(payload_sha256=digest, snapshot_id=f"recommend-{digest[:20]}")
    manifest["cutoff_source"].update(snapshot_id=snapshot["snapshot_id"], payload_sha256=digest)
    return snapshot, manifest, records


def _pure(monkeypatch, **kwargs):
    sample = _sample(monkeypatch, **kwargs)
    return sample[0], features.compute_trade_imbalance(
        sample[0], sample[1], trade_records=sample[2]["trade"], orderbook_records=sample[2]["orderbook"])


def _files(tmp_path, monkeypatch, **kwargs):
    tmp_path.mkdir(parents=True, exist_ok=True)
    snapshot, manifest, records = _sample(monkeypatch, **kwargs)
    snapshot_path = tmp_path / "open_r1.json"
    snapshot_path.write_bytes(canonical_json_bytes(snapshot))
    manifest["cutoff_source"].update(path=str(snapshot_path), file_sha256=sha256_bytes(snapshot_path.read_bytes()))
    for channel, values in records.items():
        path = tmp_path / f"{channel}.jsonl.gz"
        path.write_bytes(gzip.compress(b"".join(canonical_json_bytes(row)+b"\n" for row in values), mtime=0))
        manifest["streams"][channel]["artifact"].update(
            path=str(path), size_bytes=path.stat().st_size, sha256=sha256_bytes(path.read_bytes()))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    report = features.read_recommend_microstructure(snapshot_path, manifest_path)
    feature_path = tmp_path / "recommend_features.json"
    feature_path.write_bytes(canonical_json_bytes(report))
    return snapshot_path, feature_path


def _clock(day="2026-09-08", minute=10):
    return datetime.fromisoformat(f"{day}T09:{minute:02d}:00+09:00")


def _publish(tmp_path, monkeypatch, *, day="2026-09-08", minute=10, **kwargs):
    paths = _files(tmp_path / "inputs" / day, monkeypatch, day=day, **kwargs)
    output_root = tmp_path / "trials"
    report = trial.record_microstructure_trial(*paths, output_root=output_root, now_fn=lambda: _clock(day, minute))
    return paths, output_root, report


def _evidence(snapshot_path, *, halt=None):
    snapshot = snapshots.load_snapshot(snapshot_path)
    rows = []
    for row in snapshot["universe"]:
        i = row["rank"]
        values = {"up10": i == 4, "dn5": i == 3, "mfe": .11 if i == 4 else .02,
                  "mae": -.06 if i == 3 else -.01, "tp5_sl3_return_net": i / 1000 - .0015,
                  "eod_return_net": i / 2000 - .0015}
        rows.append({**row, **values, "label_status": "labeled"})
        if i == halt:
            rows[-1].update(label_status="halted_no_observations", **dict.fromkeys(values))
    return {"snapshot": snapshot, "label": {"rows": rows,
                "execution_start_at": f"{snapshot['asof']}T09:15:00+09:00"},
            "manifest": {"files": {}, "fixture": "canonical loader separately tested"}}


def _evaluate(root, monkeypatch, *, halt=None, now="2026-09-10T12:00:00+09:00"):
    monkeypatch.setattr(trial, "load_recommendation_evidence",
                        lambda snapshot_path, **kwargs: _evidence(snapshot_path, halt=halt))
    return trial.evaluate_microstructure_trials(root, now=now, n_boot=100)


def test_exact_boundary_only_changes_one_pick_and_keeps_full_universe(monkeypatch):
    snapshot, feature = _pure(monkeypatch)
    before = copy.deepcopy((snapshot, feature))
    result = trial.plan_microstructure_trial(snapshot, feature)
    coins = fixtures.MARKETS
    assert result["boundary_block"] == coins[2:4]
    assert result["control_top3"] == coins[:3]
    assert result["challenger_top3"] == coins[:2] + [coins[3]]
    assert result["challenger_ranking"][:5] == coins[:2] + [coins[3], coins[2], coins[4]]
    assert result["changed_picks"] == 1
    assert result["challenger_ranking"][4:] == coins[4:]
    assert (snapshot, feature) == before


@pytest.mark.parametrize("field", trial.TIE_FIELDS)
def test_no_approximate_tolerance_for_any_of_five_fields(monkeypatch, field):
    snapshot, feature = _pure(monkeypatch)
    snapshot["universe"][3][field] += 1e-12
    digest = snapshots._document_digest(snapshot)
    snapshot.update(payload_sha256=digest, snapshot_id=f"recommend-{digest[:20]}")
    feature["snapshot"].update(payload_sha256=digest, snapshot_id=snapshot["snapshot_id"])
    plan = trial.plan_microstructure_trial(snapshot, feature)
    assert plan["is_no_op"] is True and plan["boundary_opportunity"] is False


def test_missing_required_feature_does_not_reselect_or_impute(monkeypatch):
    plan = trial.plan_microstructure_trial(*_pure(monkeypatch, missing=True))
    assert plan["status"] == "unavailable"
    assert plan["challenger_top3"] is None
    assert plan["required_missing_features"] == [fixtures.MARKETS[3]]


def test_no_op_without_boundary_does_not_require_irrelevant_feature(monkeypatch):
    plan = trial.plan_microstructure_trial(*_pure(monkeypatch, missing=True, opportunity=False))
    assert plan["status"] == "planned" and plan["is_no_op"] is True


def test_equal_imbalance_uses_original_rank_and_shuffled_features_stable(monkeypatch):
    snapshot, feature = _pure(monkeypatch, changed=False)
    expected = trial.plan_microstructure_trial(snapshot, feature)
    feature["rows"].reverse()
    assert trial.plan_microstructure_trial(snapshot, feature) == expected
    assert expected["is_no_op"] is True


@pytest.mark.parametrize("damage", ["quality_reasons", "missing_book", "missing_capture", "available_coins",
                                   "coverage_count", "no_trade_coins", "nan", "duplicate", "rank", "futurewindow"])
def test_feature_contradictions_fail_closed(monkeypatch, damage):
    snapshot, feature = _pure(monkeypatch)
    if damage == "quality_reasons":
        feature["quality_reasons"] = ["capture_not_complete"]
    elif damage in {"missing_book", "missing_capture"}:
        name = "missing_initial_book_coins" if damage == "missing_book" else "missing_capture_coins"
        feature["coverage"][name] = [fixtures.MARKETS[0]]
    elif damage == "available_coins":
        feature["coverage"]["available_coins"] = []
    elif damage == "coverage_count":
        feature["coverage"]["feature_available_rows"] = 0
    elif damage == "no_trade_coins":
        feature["coverage"]["no_observed_trade_coins"] = [fixtures.MARKETS[0]]
    elif damage == "nan":
        feature["rows"][3][features.FEATURE] = float("nan")
    elif damage == "duplicate":
        feature["rows"][1]["coin"] = feature["rows"][0]["coin"]
    elif damage == "rank":
        feature["rows"][0]["rank"] = 2
    else:
        feature["window"]["end_exclusive_at_ns"] += 1
    with pytest.raises(trial.MicrostructureTrialError):
        trial.plan_microstructure_trial(snapshot, feature)


def test_invalid_global_capture_no_op_is_still_unavailable(monkeypatch):
    snapshot, manifest, records = _sample(monkeypatch, opportunity=False)
    manifest["complete"] = False
    feature = features.compute_trade_imbalance(snapshot, manifest, trade_records=records["trade"],
                                               orderbook_records=records["orderbook"])
    assert trial.plan_microstructure_trial(snapshot, feature)["reason"] == "feature_evidence_invalid"


def test_strict_native_input_roundtrip_and_new_publication(tmp_path, monkeypatch):
    paths = _files(tmp_path / "inputs", monkeypatch)
    before = {path: path.read_bytes() for path in paths[0].parent.iterdir()}
    report = trial.record_microstructure_trial(*paths, output_root=tmp_path/"trials", now_fn=_clock)
    score, commit = [json.loads(Path(report[key]).read_bytes()) for key in ("score_path", "commit_path")]
    assert report["status"] == "committed" and report["changed_picks"] == 1
    assert commit["score_payload_sha256"] == score["payload_sha256"]
    assert commit["score_durable_observed_at"] == _clock().astimezone(trial.timezone.utc).isoformat()
    assert score["plan"]["challenger_top3"] == fixtures.MARKETS[:2]+[fixtures.MARKETS[3]]
    assert all(path.read_bytes() == data for path, data in before.items())
    assert sorted(path.name for path in Path(report["score_path"]).parent.iterdir()) == ["commit.json", "score.json"]


@pytest.mark.parametrize("damage", ["omit_manifest_raw", "omit_raw", "manifest_hash", "raw_metadata", "capture_id"])
def test_feature_provenance_graph_cannot_omit_or_change_sources(tmp_path, monkeypatch, damage):
    paths = _files(tmp_path, monkeypatch)
    feature = json.loads(paths[1].read_bytes())
    provenance = feature["provenance"]
    if damage == "omit_manifest_raw":
        provenance["files"] = [provenance["files"][0], *provenance["generator_sources"]]
        provenance.pop("raw_artifacts")
        provenance.pop("capture_manifest_payload_sha256")
    elif damage == "omit_raw":
        provenance["files"] = provenance["files"][:6]
    elif damage == "manifest_hash":
        provenance["capture_manifest_payload_sha256"] = "0"*64
    elif damage == "raw_metadata":
        provenance["raw_artifacts"]["trade"]["size_bytes"] += 1
    else:
        feature["capture_id"] = "different"
    paths[1].write_bytes(canonical_json_bytes(feature))
    with pytest.raises(trial.MicrostructureTrialError):
        trial.record_microstructure_trial(*paths, output_root=tmp_path/"trials", now_fn=_clock)
    assert not (tmp_path/"trials").exists()


def test_retry_preserves_original_score_and_receipt_bytes(tmp_path, monkeypatch):
    paths, root, report = _publish(tmp_path, monkeypatch)
    before = {name: Path(report[name]).read_bytes() for name in ("score_path", "commit_path")}
    again = trial.record_microstructure_trial(*paths, output_root=root, now_fn=lambda: _clock(minute=20))
    assert again["status"] == "committed" and again["reused"] is True
    assert all(Path(again[name]).read_bytes() == content for name, content in before.items())


def test_crash_after_score_never_retroactively_issues_receipt(tmp_path, monkeypatch):
    paths = _files(tmp_path/"inputs", monkeypatch)
    original = trial._publish_new

    def fail_commit(directory, name, document):
        if name == "commit.json":
            raise OSError("injected crash gap")
        original(directory, name, document)

    monkeypatch.setattr(trial, "_publish_new", fail_commit)
    with pytest.raises(OSError, match="crash gap"):
        trial.record_microstructure_trial(*paths, output_root=tmp_path/"trials", now_fn=_clock)
    monkeypatch.setattr(trial, "_publish_new", original)
    again = trial.record_microstructure_trial(*paths, output_root=tmp_path/"trials", now_fn=lambda: _clock(minute=20))
    assert again["status"] == "uncertain"
    assert not Path(again["commit_path"]).exists()
    result = _evaluate(tmp_path/"trials", monkeypatch)
    assert result["dates"][0]["status"] == "uncertain"


def test_observed_clock_sample_is_after_score_file_and_directory_fsync(tmp_path, monkeypatch):
    paths = _files(tmp_path/"inputs", monkeypatch)
    events = []
    original = os.fsync

    def fsync(fd):
        events.append("dir_sync" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file_sync")
        original(fd)

    def clock():
        events.append("clock")
        return _clock()

    monkeypatch.setattr(os, "fsync", fsync)
    trial.record_microstructure_trial(*paths, output_root=tmp_path/"trials", now_fn=clock)
    clocks = [i for i, name in enumerate(events) if name == "clock"]
    assert len(clocks) == 2
    assert events[clocks[1]-1] == "dir_sync"
    assert "file_sync" in events[clocks[0]:clocks[1]]
    assert "file_sync" in events[clocks[1]:]  # Separate commit durability.


def test_backward_clock_preserves_uncertain_score(tmp_path, monkeypatch):
    paths = _files(tmp_path/"inputs", monkeypatch)
    clock = iter([_clock(), _clock()-timedelta(seconds=1)])
    with pytest.raises(trial.MicrostructureTrialError, match="backwards"):
        trial.record_microstructure_trial(*paths, output_root=tmp_path/"trials", now_fn=lambda: next(clock))
    again = trial.record_microstructure_trial(*paths, output_root=tmp_path/"trials", now_fn=_clock)
    assert again["status"] == "uncertain"


def test_plan_clock_cannot_precede_snapshot_completion(tmp_path, monkeypatch):
    paths = _files(tmp_path/"inputs", monkeypatch)
    with pytest.raises(trial.MicrostructureTrialError, match="before snapshot"):
        trial.record_microstructure_trial(*paths, output_root=tmp_path/"trials", now_fn=lambda: _clock(minute=5))
    assert not (tmp_path/"trials").exists()


def test_concurrent_publishers_do_not_overwrite_or_issue_two_receipts(tmp_path, monkeypatch):
    paths = _files(tmp_path/"inputs", monkeypatch)
    original = trial._publish_new
    barrier = Barrier(2)

    def interleave(directory, name, document):
        if name == "score.json":
            barrier.wait(timeout=10)
        original(directory, name, document)

    monkeypatch.setattr(trial, "_publish_new", interleave)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: trial.record_microstructure_trial(
            *paths, output_root=tmp_path/"trials", now_fn=_clock), range(2)))
    assert sorted(row["status"] for row in results) == ["committed", "uncertain"]
    folder = tmp_path/"trials"/"2026-09-08"/trial.TRIAL_ID
    assert sorted(path.name for path in folder.iterdir()) == ["commit.json", "score.json"]


@pytest.mark.parametrize("damage", ["checksum", "plan", "promotion", "commit_binding", "commit_clock", "orphan"])
def test_existing_corrupt_artifacts_block_whole_evaluation(tmp_path, monkeypatch, damage):
    _, root, report = _publish(tmp_path, monkeypatch)
    score_path, commit_path = Path(report["score_path"]), Path(report["commit_path"])
    score, commit = json.loads(score_path.read_bytes()), json.loads(commit_path.read_bytes())
    if damage == "orphan":
        score_path.unlink()
    elif damage in {"checksum", "plan", "promotion"}:
        if damage == "promotion":
            score["deployable"] = True
        else:
            score["plan"]["control_top3"] = fixtures.MARKETS[1:4]
        if damage != "checksum":
            score = trial._seal({key: value for key, value in score.items() if key != "payload_sha256"})
        score_path.write_bytes(canonical_json_bytes(score))
    else:
        if damage == "commit_binding":
            commit["snapshot_id"] = "other"
        else:
            commit["score_durable_observed_at"] = "2026-09-08T09:00:00+09:00"
        commit = trial._seal({key: value for key, value in commit.items() if key != "payload_sha256"})
        commit_path.write_bytes(canonical_json_bytes(commit))
    with pytest.raises(trial.MicrostructureTrialError):
        _evaluate(root, monkeypatch)


def test_changed_feature_file_cannot_reuse_original_commit(tmp_path, monkeypatch):
    paths, root, _ = _publish(tmp_path, monkeypatch)
    paths[1].write_bytes(paths[1].read_bytes()+b"\n")
    with pytest.raises(trial.MicrostructureTrialError, match="source changed"):
        _evaluate(root, monkeypatch)


def test_partial_raw_is_recorded_as_unavailable_not_fabricated_valid(tmp_path, monkeypatch):
    paths = _files(tmp_path/"inputs", monkeypatch)
    manifest_path = paths[0].parent/"manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["complete"] = False
    manifest["streams"]["trade"]["artifact"] = {"exists": False, "writer_state": "failed"}
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    feature = features.read_recommend_microstructure(paths[0], manifest_path)
    paths[1].write_bytes(canonical_json_bytes(feature))
    assert len(feature["provenance"]["files"]) == 7
    recorded = trial.record_microstructure_trial(*paths, output_root=tmp_path/"trials", now_fn=_clock)
    assert recorded["status"] == "committed" and recorded["plan_status"] == "unavailable"
    report = _evaluate(tmp_path/"trials", monkeypatch)
    assert report["dates"][0]["reason"] == "feature_evidence_invalid"
    assert report["primary_paired"]["n_dates"] == 0


@pytest.mark.parametrize("kind", ["root_symlink", "date_symlink", "leaf_symlink", "fifo"])
def test_symlink_and_nonregular_evidence_cannot_escape_or_block(tmp_path, monkeypatch, kind):
    paths = _files(tmp_path/"inputs", monkeypatch)
    root = tmp_path/"trials"
    outside = tmp_path/"outside"
    outside.mkdir()
    if kind == "root_symlink":
        root.symlink_to(outside, target_is_directory=True)
    elif kind == "date_symlink":
        root.mkdir()
        (root/"2026-09-08").symlink_to(outside, target_is_directory=True)
    else:
        folder = root/"2026-09-08"/trial.TRIAL_ID
        folder.mkdir(parents=True)
        if kind == "leaf_symlink":
            (folder/"score.json").symlink_to(outside/"target.json")
        else:
            os.mkfifo(folder/"score.json")
    with pytest.raises((OSError, ValueError, RuntimeError)):
        trial.record_microstructure_trial(*paths, output_root=root, now_fn=_clock)
    assert list(outside.iterdir()) == []


def test_generic_writer_new_only_and_parent_symlink_no_outside_write(tmp_path):
    target = tmp_path/"report.json"
    trial.write_new_trial_report(target, {"plain": "native feature dict"})
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        trial.write_new_trial_report(target, {"changed": True})
    assert target.read_bytes() == before
    outside = tmp_path/"outside"
    outside.mkdir()
    link = tmp_path/"linked"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError):
        trial.write_new_trial_report(link/"forbidden.json", {"write": False})
    assert list(outside.iterdir()) == []


def test_post_link_fsync_failure_preserves_target_and_cleans_own_temporary(tmp_path, monkeypatch):
    original_link, original_sync = os.link, os.fsync
    linked = False

    def link(*args, **kwargs):
        nonlocal linked
        original_link(*args, **kwargs)
        linked = True

    def sync(fd):
        if linked:
            raise OSError("injected directory fsync failure")
        original_sync(fd)

    monkeypatch.setattr(os, "link", link)
    monkeypatch.setattr(os, "fsync", sync)
    with pytest.raises(OSError):
        trial.write_new_trial_report(tmp_path/"saved.json", {"durability": "uncertain"})
    assert sorted(path.name for path in tmp_path.iterdir()) == ["saved.json"]


@pytest.mark.parametrize("minute,status", [(14, "prospective_comparable"), (15, "late"), (16, "late")])
def test_strict_before_canonical_entry_boundary(tmp_path, monkeypatch, minute, status):
    _, root, _ = _publish(tmp_path, monkeypatch, minute=minute)
    report = _evaluate(root, monkeypatch)
    assert report["dates"][0]["status"] == status
    assert report["primary_paired"]["n_dates"] == (1 if minute < 15 else 0)


def test_primary_paired_net_uses_canonical_once_and_all_outcomes_no_reselection(tmp_path, monkeypatch):
    _, root, _ = _publish(tmp_path, monkeypatch)
    report = _evaluate(root, monkeypatch)
    metrics = report["primary_paired"]["metrics"]
    assert metrics["control"]["mean"]["tp5_sl3_return_net"] == pytest.approx(.0005)
    assert metrics["challenger_minus_control"]["mean"]["up10"] == pytest.approx(1/3)
    assert metrics["challenger_minus_control"]["mean"]["dn5"] == pytest.approx(-1/3)
    assert metrics["challenger_minus_control"]["mean"]["tp5_sl3_return_net"] == pytest.approx(.001/3)
    assert metrics["control"]["ci_status"] == "insufficient_dates"
    assert report["full_universe_common"]["n_dates"] == 1
    assert report["deployable"] is False and report["automatic_promotion"] is False


@pytest.mark.parametrize("halt,primary,baseline", [(3, 0, 0), (4, 0, 0), (100, 1, 0)])
def test_halted_fixed_membership_no_substitution_and_separate_baseline_cohort(tmp_path, monkeypatch, halt, primary, baseline):
    _, root, _ = _publish(tmp_path, monkeypatch)
    report = _evaluate(root, monkeypatch, halt=halt)
    assert report["primary_paired"]["n_dates"] == primary
    assert report["full_universe_common"]["n_dates"] == baseline
    assert report["dates"][0]["control_top3"] == fixtures.MARKETS[:3]
    assert report["dates"][0]["challenger_top3"] == fixtures.MARKETS[:2]+[fixtures.MARKETS[3]]


def test_no_op_days_stay_in_primary_and_bootstrap_has_exact_zero_delta(tmp_path, monkeypatch):
    for day in range(8, 13):
        _publish(tmp_path, monkeypatch, day=f"2026-09-{day:02d}", changed=False)
    report = _evaluate(tmp_path/"trials", monkeypatch, now="2026-09-14T12:00:00+09:00")
    cohort = report["primary_paired"]
    assert cohort["n_dates"] == cohort["n_no_op_dates"] == 5
    assert cohort["n_changed_dates"] == 0
    delta = cohort["metrics"]["challenger_minus_control"]
    assert set(delta["mean"].values()) == {0.}
    assert all(ci == [0., 0.] for ci in delta["iid_date_ci95"].values())
    assert all(ci == [0., 0.] for ci in delta["observed_date_block3_ci95"].values())
    assert len(cohort["leave_one_date_out"]) == 5


@pytest.mark.parametrize("reason,status", [("label_missing", "pending"), ("receipt_missing_delivery_unknown", "pending"),
                                         ("label_not_available_as_of_now", "pending"), ("not_forward", "unavailable")])
def test_expected_canonical_unavailability_is_not_corruption(tmp_path, monkeypatch, reason, status):
    _, root, _ = _publish(tmp_path, monkeypatch)

    def unavailable(*args, **kwargs):
        raise EvidenceUnavailable(reason)

    monkeypatch.setattr(trial, "load_recommendation_evidence", unavailable)
    report = trial.evaluate_microstructure_trials(root, now="2026-09-10T12:00:00+09:00", n_boot=10)
    assert report["dates"][0]["status"] == status
    assert report["effect_status"] == "not_evaluated"


def test_hard_canonical_error_blocks_whole_report_and_does_not_alter_commit(tmp_path, monkeypatch):
    _, root, report = _publish(tmp_path, monkeypatch)
    before = Path(report["score_path"]).read_bytes(), Path(report["commit_path"]).read_bytes()

    def corrupted(*args, **kwargs):
        raise EvidenceError("broken canonical receipt")

    monkeypatch.setattr(trial, "load_recommendation_evidence", corrupted)
    with pytest.raises(EvidenceError):
        trial.evaluate_microstructure_trials(root, now="2026-09-10T12:00:00+09:00")
    assert before == (Path(report["score_path"]).read_bytes(), Path(report["commit_path"]).read_bytes())


def test_prelaunch_record_is_replay_never_forward(tmp_path, monkeypatch):
    _, root, _ = _publish(tmp_path, monkeypatch, day="2026-09-07")
    report = _evaluate(root, monkeypatch)
    assert report["dates"][0]["status"] == "prelaunch_replay"
    assert report["primary_paired"]["n_dates"] == 0


@pytest.mark.parametrize("now,expected,missing,not_due", [
    ("2026-09-07T12:00:00+09:00", 0, 0, []),
    ("2026-09-08T08:44:59+09:00", 0, 0, ["2026-09-08"]),
    ("2026-09-08T08:45:00+09:00", 1, 1, []),
    ("2026-09-10T08:00:00+09:00", 2, 2, ["2026-09-10"]),
])
def test_missing_calendar_denominator_without_any_trial_folders(tmp_path, now, expected, missing, not_due):
    root = tmp_path/"nonexistent"
    report = trial.evaluate_microstructure_trials(root, now=now, n_boot=10)
    assert report["coverage"]["calendar_expected_count"] == expected
    assert report["coverage"]["calendar_missing_count"] == missing
    assert report["coverage"]["calendar_not_due_dates"] == not_due
    assert report["effect_status"] == "not_evaluated"
    assert not root.exists()


def test_input_change_during_evaluation_blocks_result(tmp_path, monkeypatch):
    _, root, report = _publish(tmp_path, monkeypatch)
    original = trial.load_recommendation_evidence

    def mutate(snapshot_path, **kwargs):
        Path(report["score_path"]).write_bytes(Path(report["score_path"]).read_bytes()+b"\n")
        return _evidence(snapshot_path)

    monkeypatch.setattr(trial, "load_recommendation_evidence", mutate)
    with pytest.raises(trial.MicrostructureTrialError, match="trial inputs changed"):
        trial.evaluate_microstructure_trials(root, now="2026-09-10T12:00:00+09:00", n_boot=10)
    monkeypatch.setattr(trial, "load_recommendation_evidence", original)


def test_raw_input_change_after_strict_load_blocks_evaluation(tmp_path, monkeypatch):
    paths, root, _ = _publish(tmp_path, monkeypatch)

    def mutate(snapshot_path, **kwargs):
        raw = paths[0].parent/"trade.jsonl.gz"
        raw.write_bytes(raw.read_bytes()+b"\n")
        return _evidence(snapshot_path)

    monkeypatch.setattr(trial, "load_recommendation_evidence", mutate)
    with pytest.raises(trial.MicrostructureTrialError, match="evidence changed"):
        trial.evaluate_microstructure_trials(root, now="2026-09-10T12:00:00+09:00", n_boot=10)


def test_canonical_manifest_inputs_rechecked_after_statistical_summary(tmp_path, monkeypatch):
    paths, root, _ = _publish(tmp_path, monkeypatch)
    label_path = paths[0].parent/"synthetic_label.json"
    label_path.write_text("{}")

    def evidence(snapshot_path, **kwargs):
        result = _evidence(snapshot_path)
        result["manifest"]["files"] = {"label": file_identity(label_path, root=trial.ROOT)}
        return result

    original = trial._cohort_summary

    def change_after_read(*args, **kwargs):
        label_path.write_text('{"changed":true}')
        return original(*args, **kwargs)

    monkeypatch.setattr(trial, "load_recommendation_evidence", evidence)
    monkeypatch.setattr(trial, "_cohort_summary", change_after_read)
    with pytest.raises(trial.MicrostructureTrialError, match="evidence changed"):
        trial.evaluate_microstructure_trials(root, now="2026-09-10T12:00:00+09:00", n_boot=10)


def test_cli_empty_read_only_and_optional_new_only_output(tmp_path, capsys):
    root = tmp_path/"absent"
    args = ["--trial-root", str(root), "--now", "2026-09-07T12:00:00+09:00", "--n-boot", "10"]
    assert main(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["effect_status"] == "not_evaluated"
    assert not root.exists()
    output = tmp_path/"diagnostic.json"
    assert main([*args, "--output", str(output)]) == 0
    before = file_identity(output, root=tmp_path)
    capsys.readouterr()
    assert main([*args, "--output", str(output)]) == 1
    assert json.loads(capsys.readouterr().err)["status"] == "blocked"
    assert file_identity(output, root=tmp_path) == before


def test_real_receipt_root_is_shared_default():
    from notifier.delivery_receipt import DEFAULT_RECEIPT_ROOT
    assert trial.DEFAULT_RECEIPT_ROOT == DEFAULT_RECEIPT_ROOT
    assert trial.evaluate_microstructure_trials.__kwdefaults__["receipt_root"] == DEFAULT_RECEIPT_ROOT

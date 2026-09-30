"""Synthetic native chain: no live data, collector, network, fitting or sends."""

from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import os
import stat
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
import requests

import test_recommend_microstructure_trial as raw
import test_recommend_score_labels as label_fixture
from test_trade_shortlist_independent import _clock
from notifier import delivery_receipt as receipts
from ops import recommend_regime_forward as forward
from ops.artifact_provenance import canonical_json_bytes, file_identity, strict_json_object
from signals import recommend_score_labels as labels

DAY = "2026-10-01"


def _at(wall="09:12:00", day=DAY):
    return datetime.fromisoformat(f"{day}T{wall}+09:00")


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.setenv("PRELUDE_FORBID_TELEGRAM", "1")

    def forbidden(*args, **kwargs):
        pytest.fail("forward test attempted a live operation")

    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    for name in ("_data_metadata", "_git_code_metadata", "_environment_metadata"):
        monkeypatch.setattr(raw.snapshots, name, forbidden)


def _inputs(tmp_path, monkeypatch, day=DAY, *, complete=False, missing=False):
    original = raw._sample

    def with_context(*args, **kwargs):
        snapshot, manifest, records = original(*args, **kwargs)
        for row in snapshot["universe"]:
            row["feature_values"].update(f_ret_1d=.01, f_qv_ma7_vs_ma30=1.2,
                                         f_atr_pct_14=.03, f_pos_in_20d_range=.5)
        snapshot["feature_columns"] = list(snapshot["universe"][0]["feature_values"])
        snapshot["features"]["columns"] = snapshot["feature_columns"]
        snapshot["top3"] = deepcopy(snapshot["universe"][:3])
        digest = raw.snapshots._document_digest(snapshot)
        snapshot.update(payload_sha256=digest, snapshot_id=f"recommend-{digest[:20]}")
        manifest["cutoff_source"].update(snapshot_id=snapshot["snapshot_id"], payload_sha256=digest)
        return snapshot, manifest, records

    with monkeypatch.context() as patch:
        patch.setattr(raw, "_sample", with_context)
        snapshot_path, feature = raw._files(tmp_path / "inputs" / day, monkeypatch, day=day, missing=missing)
    trials, receipt_root, label_root = (tmp_path / name for name in ("trials", "receipts", "labels"))
    forward.shortlist.record_trade_shortlist_trial(snapshot_path, feature, output_root=trials,
                                                  now_fn=lambda: _at("09:10:00", day))
    snapshot = raw.snapshots.load_snapshot(snapshot_path)
    with monkeypatch.context() as patch:
        patch.setattr(receipts, "datetime", _clock(f"{day}T09:08:03+09:00"))
        receipts.write_delivery_receipt(
            snapshot, delivery_ok=True, attempted_at=f"{day}T09:08:01+09:00", sent_at=f"{day}T09:08:02+09:00",
            telegram_result=label_fixture._telegram_result("synthetic", _at("09:08:02", day).astimezone(timezone.utc).isoformat()),
            message="synthetic", root=receipt_root,
        )
    if complete:
        _label(tmp_path, monkeypatch, day)
    return snapshot_path, feature, trials, receipt_root, label_root


def _label(tmp_path, monkeypatch, day=DAY):
    path = tmp_path / "inputs" / day / "open_r1.json"
    snapshot = raw.snapshots.load_snapshot(path)
    ranks = {row["coin"]: row["rank"] for row in snapshot["universe"]}
    following = (date.fromisoformat(day) + timedelta(days=1)).isoformat()

    def assessor(coin, start_at, **kwargs):
        low, high, close = ((94., 100., 94.) if ranks[coin] == 3
                            else (100., 111., 111.) if ranks[coin] == 4
                            else (100., 100., 100.))
        return label_fixture._assessment([(100., high, low, close)] + [(close, close, close, close)] * 95,
                                         start=start_at)

    with monkeypatch.context() as patch:
        patch.setattr(labels, "datetime", _clock(f"{following}T10:00:00+09:00"))
        return labels.label_recommend_snapshot(
            path, output_root=tmp_path / "labels", receipt_root=tmp_path / "receipts",
            db_path=tmp_path / "nonexistent.db", now=_at("12:00:00", following),
            assessor=assessor,
        )


def _evaluation(tmp_path, now):
    return forward.evaluator.evaluate_trade_shortlist_trials(
        tmp_path / "trials", label_root=tmp_path / "labels", receipt_root=tmp_path / "receipts",
        now=now, n_boot=10,
    )


def _prepare(tmp_path, monkeypatch, *, history=False, missing=False):
    if history:
        for day in ("2026-09-24", "2026-09-25", "2026-09-26", "2026-09-27", "2026-09-28", "2026-09-29", "2026-09-30"):
            _inputs(tmp_path, monkeypatch, day, complete=True)
    inputs = _inputs(tmp_path, monkeypatch, missing=missing)
    evaluation = _evaluation(tmp_path, _at("09:11:00"))
    path = inputs[1].parent / "trade_shortlist_evaluation.json"
    forward.native.write_new_trial_report(path, evaluation)
    return path, tmp_path / "forward", inputs


def _publish(path, root, tmp_path, *, clock=None):
    return forward.record_forward(path, output_root=root, shortlist_root=tmp_path / "trials",
                                  receipt_root=tmp_path / "receipts", now_fn=clock or _at)


def _score_path(root):
    return root / DAY / forward.TRIAL_ID / "score.json"


def _reseal(path, mutate):
    value = strict_json_object(path)
    mutate(value)
    value.pop("payload_sha256", None)
    path.write_bytes(canonical_json_bytes(forward.native._seal(value)))


def test_native_end_to_end_records_before_outcomes_and_evaluates_frozen_policy(tmp_path, monkeypatch):
    path, root, _ = _prepare(tmp_path, monkeypatch, history=True)
    inputs = {str(p): file_identity(p, root=tmp_path) for p in tmp_path.rglob("*") if p.is_file()}
    assert not (tmp_path / "labels" / DAY).exists()
    result = _publish(path, root, tmp_path)
    assert result["status"] == "committed" and result["eligibility"] == "ready"
    record = forward.read_forward(DAY, root=root, now=_at("09:13:00"))
    selection = record["score"]["plan"]["selection"]
    assert selection["choices"]["recent"]["arm"] == "challenger"
    assert selection["choices"]["same_context"]["arm"] == "challenger"
    assert all(day < DAY for day in selection["choices"]["recent"]["history_dates"])
    assert "2026-09-30" not in selection["choices"]["recent"]["history_dates"]
    assert all(file_identity(Path(p), root=tmp_path) == ident for p, ident in inputs.items())
    _label(tmp_path, monkeypatch)
    native = _evaluation(tmp_path, _at("12:00:00", "2026-10-02"))
    report = forward.evaluate_forward(native, root=root, now=_at("12:00:00", "2026-10-02"), n_boot=10)
    assert report["paired_dates"] == 1
    assert report["policies"]["recent"]["changed_picks"] == 1
    assert report["policies"]["recent"]["minus_fixed_r1"]["mean"]["eod_return_net"] == pytest.approx(.17 / 3)
    assert report["dates"][1]["status"] == "missing"
    assert report["deployable"] is report["automatic_promotion"] is False
    assert forward.read_forward(DAY, root=root, now=_at("12:00:00", "2026-10-02"))["score"] == record["score"]


def test_warmup_fallback_is_recorded_and_restart_never_rewrites(tmp_path, monkeypatch):
    path, root, _ = _prepare(tmp_path, monkeypatch)
    _publish(path, root, tmp_path)
    before = _score_path(root).read_bytes()
    again = _publish(path, root, tmp_path, clock=lambda: _at("12:00:00"))
    assert again == {"status": "committed", "eligibility": "ready", "reused": True}
    assert before == _score_path(root).read_bytes()
    plan = strict_json_object(_score_path(root))["plan"]
    assert plan["selection"]["choices"]["recent"]["reason"] == "insufficient_completed_history"


def test_prelaunch_rejected_without_creating_output(tmp_path):
    root = tmp_path / "forward"
    with pytest.raises(ValueError, match="not started"):
        _publish(tmp_path / "missing", root, tmp_path, clock=lambda: _at(day="2026-09-30"))
    assert not root.exists()


@pytest.mark.parametrize("durable, expected", [("09:14:59.999999", "ready"), ("09:15:00", "late"), ("09:15:01", "late")])
def test_fsync_durability_not_planning_time_defines_eligibility(tmp_path, monkeypatch, durable, expected):
    path, root, _ = _prepare(tmp_path, monkeypatch)
    stamps = iter([_at(), _at("09:14:00"), _at(durable)])
    result = _publish(path, root, tmp_path, clock=lambda: next(stamps))
    assert result["eligibility"] == expected
    record = forward.read_forward(DAY, root=root, now=_at("10:00:00"))
    assert record["eligibility"] == expected
    if expected == "late":
        assert forward.inspect_forward(root=root, now=_at("10:00:00"))["attention_required"]


def test_late_planning_is_preserved_not_backdated(tmp_path, monkeypatch):
    path, root, _ = _prepare(tmp_path, monkeypatch)
    result = _publish(path, root, tmp_path, clock=lambda: _at("09:16:00"))
    assert result["eligibility"] == "late"
    assert strict_json_object(_score_path(root))["plan"]["selection"] is None


def test_partial_score_cannot_acquire_later_commit(tmp_path, monkeypatch):
    path, root, _ = _prepare(tmp_path, monkeypatch)
    original = forward.native._publish_new

    def interrupted(directory, name, document):
        if name == "commit.json":
            raise OSError("simulated crash")
        original(directory, name, document)

    with monkeypatch.context() as patch:
        patch.setattr(forward.native, "_publish_new", interrupted)
        with pytest.raises(OSError, match="crash"):
            _publish(path, root, tmp_path)
    before = _score_path(root).read_bytes()
    result = _publish(path, root, tmp_path, clock=lambda: _at("12:00:00"))
    assert result["status"] == "uncertain"
    assert _score_path(root).read_bytes() == before
    assert not _score_path(root).with_name("commit.json").exists()


@pytest.mark.parametrize("field", ["plan", "source", "clock", "commit", "evaluation", "receipt"])
def test_resealed_tampering_is_rejected(tmp_path, monkeypatch, field):
    path, root, inputs = _prepare(tmp_path, monkeypatch)
    _publish(path, root, tmp_path)
    score = _score_path(root)
    if field == "plan":
        _reseal(score, lambda value: value["plan"]["selected_coins"].update(recent=["KRW-FAKE"]))
    elif field == "source":
        _reseal(score, lambda value: value.update(generator_sources=[]))
    elif field == "clock":
        _reseal(score, lambda value: value.update(planned_at="2026-09-30T09:12:00+09:00"))
    elif field == "commit":
        _reseal(score.with_name("commit.json"), lambda value: value.update(score_payload_sha256="a" * 64))
    elif field == "evaluation":
        _reseal(path, lambda value: value.update(generated_at="2026-10-02T00:00:00+09:00"))
    else:
        receipt = inputs[3] / DAY / "open_r1.json"
        receipt.write_bytes(receipt.read_bytes() + b" ")
    result = forward.inspect_forward(root=root, now=_at("10:00:00"))
    assert result["attention_required"] and result["status"] == "evidence_invalid"


def test_unavailable_candidate_is_not_fake_switch_or_zero_return(tmp_path, monkeypatch):
    path, root, _ = _prepare(tmp_path, monkeypatch, missing=True)
    result = _publish(path, root, tmp_path)
    assert result["eligibility"] == "unavailable"
    assert not forward.inspect_forward(root=root, now=_at("10:00:00"))["attention_required"]
    report = forward.evaluate_forward(_evaluation(tmp_path, _at("10:00:00")), root=root, now=_at("10:00:00"), n_boot=10)
    assert report["paired_dates"] == 0
    assert report["policies"]["recent"]["metrics"] is None


@pytest.mark.parametrize("instant,attention", [("2026-09-30T12:00:00+09:00", False),
                                             ("2026-10-01T09:19:59+09:00", False),
                                             ("2026-10-01T09:20:00+09:00", True),
                                             ("2026-10-02T08:00:00+09:00", True)])
def test_launch_due_and_yesterday_missing_without_any_writes(tmp_path, instant, attention):
    root = tmp_path / "absent"
    report = forward.inspect_forward(root=root, now=instant)
    assert report["attention_required"] is attention
    assert not root.exists()


def test_input_digest_change_during_publication_leaves_uncertain(tmp_path, monkeypatch):
    path, root, _ = _prepare(tmp_path, monkeypatch)
    original = forward.native._publish_new

    def mutate(directory, name, document):
        original(directory, name, document)
        if name == "score.json":
            path.write_bytes(path.read_bytes() + b" ")

    monkeypatch.setattr(forward.native, "_publish_new", mutate)
    with pytest.raises(ValueError, match="input changed"):
        _publish(path, root, tmp_path)
    assert _score_path(root).exists() and not _score_path(root).with_name("commit.json").exists()


def test_future_cached_result_poisoning_is_rejected_without_rewriting_choices(tmp_path, monkeypatch):
    path, root, _ = _prepare(tmp_path, monkeypatch)
    _publish(path, root, tmp_path)
    _label(tmp_path, monkeypatch)
    native = _evaluation(tmp_path, _at("12:00:00", "2026-10-02"))
    baseline = forward.evaluate_forward(native, root=root, now=_at("12:00:00", "2026-10-02"), n_boot=10)
    before = _score_path(root).read_bytes()
    changed = deepcopy(native)
    changed["primary_paired"]["daily"][0]["challenger"]["eod_return_net"] = 99.
    changed = forward.native._seal({k: v for k, v in changed.items() if k != "payload_sha256"})
    with pytest.raises(ValueError, match="metrics mismatch"):
        forward.evaluate_forward(changed, root=root, now=_at("12:00:00", "2026-10-02"), n_boot=10)
    assert _score_path(root).read_bytes() == before
    assert baseline["dates"][0]["choices"]["recent"]["arm"] == "control"


def test_heartbeat_is_light_but_forward_evaluation_still_checks_raw_bytes(tmp_path, monkeypatch):
    path, root, inputs = _prepare(tmp_path, monkeypatch)
    _publish(path, root, tmp_path)
    _label(tmp_path, monkeypatch)
    evaluation = _evaluation(tmp_path, _at("12:00:00", "2026-10-02"))
    monkeypatch.setattr(forward.shortlist, "read_trade_shortlist_record",
                        lambda *a, **kw: pytest.fail("heartbeat re-parsed raw trades"))
    raw_file = next(forward.native._path(item["path"]) for item in
                    strict_json_object(inputs[1])["provenance"]["files"] if ".jsonl" in item["path"])
    raw_file.write_bytes(raw_file.read_bytes() + b" ")
    probe = forward.inspect_forward(root=root, now=_at("10:00:00"))
    assert not probe["attention_required"]
    assert probe["verification_scope"] == "publication_and_choice_chain_not_raw_capture"
    with pytest.raises(ValueError, match="input changed"):
        forward.evaluate_forward(evaluation, root=root, now=_at("12:00:00", "2026-10-02"), n_boot=10)


def test_forged_history_average_is_rejected_even_with_new_report_checksum(tmp_path, monkeypatch):
    path, root, _ = _prepare(tmp_path, monkeypatch, history=True)
    _reseal(path, lambda value: value["primary_paired"]["daily"][0]["challenger"].update(eod_return_net=99.))
    with pytest.raises(ValueError, match="metrics mismatch"):
        _publish(path, root, tmp_path)
    assert not root.exists()


@pytest.mark.parametrize("phase", ["file", "directory"])
def test_failed_fsync_cannot_create_commit(tmp_path, monkeypatch, phase):
    path, root, _ = _prepare(tmp_path, monkeypatch)
    fsync = os.fsync

    def fail(fd):
        if (phase == "file" and stat.S_ISREG(os.fstat(fd).st_mode)) or (
            phase == "directory" and _score_path(root).exists()
        ):
            raise OSError("fsync failed")
        return fsync(fd)

    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", fail)
        with pytest.raises(OSError, match="fsync"):
            _publish(path, root, tmp_path)
    assert not _score_path(root).with_name("commit.json").exists()


def test_concurrent_publishers_cannot_overwrite_or_double_commit(tmp_path, monkeypatch):
    path, root, _ = _prepare(tmp_path, monkeypatch)
    barrier = Barrier(2)
    original = forward.native._publish_new

    def sync(directory, name, value):
        if name == "score.json":
            barrier.wait(timeout=15)
        return original(directory, name, value)

    monkeypatch.setattr(forward.native, "_publish_new", sync)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: _publish(path, root, tmp_path), range(2)))
    assert sorted(row["status"] for row in results) == ["committed", "uncertain"]
    assert forward.read_forward(DAY, root=root, now=_at("10:00:00"))["eligibility"] == "ready"


def test_clock_reversal_cannot_receive_later_commit(tmp_path, monkeypatch):
    path, root, _ = _prepare(tmp_path, monkeypatch)
    stamps = iter([_at(), _at(), _at("09:11:00")])
    with pytest.raises(ValueError, match="backwards"):
        _publish(path, root, tmp_path, clock=lambda: next(stamps))
    assert _score_path(root).exists() and not _score_path(root).with_name("commit.json").exists()


def test_symlink_output_cannot_escape(tmp_path, monkeypatch):
    path, root, _ = _prepare(tmp_path, monkeypatch)
    outside = tmp_path / "outside"
    outside.mkdir()
    root.symlink_to(outside, target_is_directory=True)
    with pytest.raises((OSError, ValueError)):
        _publish(path, root, tmp_path)
    assert not list(outside.iterdir())


@pytest.mark.parametrize("field", ["slot", "selection"])
def test_resealing_both_score_and_commit_cannot_forge_policy(tmp_path, monkeypatch, field):
    path, root, _ = _prepare(tmp_path, monkeypatch)
    _publish(path, root, tmp_path)
    score = _score_path(root)

    def forge(value):
        if field == "slot":
            value["slot"] = "preopen"
        else:
            value["plan"]["selection"]["choices"]["recent"]["arm"] = "challenger"

    _reseal(score, forge)
    _reseal(score.with_name("commit.json"), lambda value:
            value.update(score_payload_sha256=strict_json_object(score)["payload_sha256"]))
    assert forward.inspect_forward(root=root, now=_at("10:00:00"))["attention_required"]

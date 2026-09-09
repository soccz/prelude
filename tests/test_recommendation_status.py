from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
import requests

import notifier.delivery_attempt as attempts
import notifier.delivery_receipt as receipts
import notifier.telegram as telegram
import ops.recommendation_status as status
import scripts.report_recommendation_status as cli
import signals.recommend as recommend
import signals.recommend_snapshot as snapshots
from ops.artifact_provenance import with_manifest_digest

DAY = "2026-08-20"
NOW = datetime.fromisoformat(f"{DAY}T10:00:00+09:00")


@pytest.fixture(autouse=True)
def forbid_live_operations(monkeypatch):
    monkeypatch.setenv("PRELUDE_FORBID_TELEGRAM", "1")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")

    def forbidden(*_args, **_kwargs):
        pytest.fail("read-only status invoked a live operation")

    for module, names in (
        (recommend, ("score_candidates",)),
        (snapshots, ("get_or_create_recommend_snapshot", "_exclusive_lock", "_data_metadata", "_git_code_metadata")),
        (receipts, ("write_delivery_receipt", "_exclusive_receipt_lock")),
        (telegram, ("send_telegram", "send_telegram_with_receipt")),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)


def _write_snapshot(root: Path, *, slot="open", day=DAY, count=3, rank_basis=None):
    feature_day = date.fromisoformat(day) - timedelta(days=slot == "preopen")
    cutoff = feature_day - timedelta(days=5)
    clock = "08:50" if slot == "preopen" else "09:05"
    start = f"{day}T{clock}:00+09:00"
    end = f"{day}T{clock}:01+09:00"
    candidates = [
        {
            "coin": f"KRW-T{i}", "rank": i, "score": 0.5,
            "pump_prob": 0.02, "pump_prob_pct": "2.0%", "rr_ratio": 1.0,
            "p_up5": 0.3, "p_up10": 0.1, "p_up20": 0.02,
            "p_dn5": 0.1, "p_dn10": 0.03, "exp_downside": -0.02,
            "dump_risk_flag": False, "entry_open": None if slot == "preopen" else 100.0,
            "sl": -0.03, "tp": 0.05, "btc_regime": "neutral",
            "feature_values": {"f_ret_3d": 0.01},
        }
        for i in range(1, count + 1)
    ]
    result = {
        "asof": day, "slot": slot, "feature_date": feature_day.isoformat(),
        "btc_regime": "neutral", "universe_n": count,
        "calibration_source": "test", "rank_basis": rank_basis or status.APPROVED_RANK_BASIS,
        "n_history_dates": 100, "ranking": "R1",
        "score_schema_version": status.APPROVED_SCORE_SCHEMA,
        "rule_version": status.APPROVED_RULE_VERSION,
        "model_random_seed": 42, "feature_columns": ["f_ret_3d"],
        "training": {
            "start": "2025-01-01", "end": (cutoff - timedelta(days=1)).isoformat(),
            "cutoff_exclusive": cutoff.isoformat(), "embargo_days": 5,
            "rows": 1000, "dates": 100,
        },
        "universe": candidates, "top3": candidates[:3],
    }
    data = {
        "path": "data/upbit_d1.db", "exists": True, "sha256": "a" * 64,
        "rows": 1000, "markets": 100,
        # This global DB date must never become the per-coin freshness date.
        "max_timestamp": f"{day} 09:00:00",
    }
    data["manifest_id"] = hashlib.sha256(snapshots._canonical_bytes(data)).hexdigest()
    document = snapshots._build_document(
        result, ranking="R1", limit_markets=None, model_id=None,
        created_at=end, decision_started_at=start, decision_completed_at=end,
        code_metadata={
            "git_commit": "b" * 40, "score_sources_dirty": False,
            "score_source_sha256": "c" * 64,
            "score_source_files": list(snapshots._SCORE_SOURCE_FILES),
        },
        data_metadata=data,
        environment_metadata={"python": "3.12", "implementation": "CPython", "packages": {}},
    )
    path = snapshots.snapshot_path(day, slot, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")
    return snapshots.load_snapshot(path)


def _write_receipt(root: Path, snapshot: dict, *, kind="success", delta=0, message=None):
    completed = datetime.fromisoformat(snapshot["decision_completed_at"])
    attempted = (completed + timedelta(seconds=1 + delta)).astimezone(timezone.utc)
    server = attempted + timedelta(seconds=1)
    recorded = server + timedelta(seconds=1)
    success = kind == "success"
    document = {
        "schema": receipts.RECEIPT_SCHEMA_VERSION,
        "snapshot_id": snapshot["snapshot_id"], "snapshot_path": snapshot["snapshot_path"],
        "asof": snapshot["asof"], "slot": snapshot["slot"],
        "model_id": snapshot["model"]["id"],
        "attempted_at": attempted.isoformat(), "recorded_at": recorded.isoformat(),
        "sent_at": server.isoformat() if success else None,
        "delivery_ok": success,
        "error": None if success else (
            telegram.AMBIGUOUS_DELIVERY_ERROR_PREFIX + "timeout"
            if kind == "ambiguous" else "Telegram rejected request"
        ),
    }
    if date.fromisoformat(snapshot["asof"]) >= receipts.RECEIPT_INTEGRITY_ACTIVATION_DATE:
        messages = [{"message_id": 1, "server_date": server.isoformat(), "text_sha256": "f" * 64}]
        document.update(
            message_sha256="d" * 64 if message is None else hashlib.sha256(message.encode()).hexdigest(),
            chat_id_sha256="e" * 64 if success or kind == "partial" else None,
            chunk_count=2 if kind == "partial" else 1,
            telegram_messages=messages if success or kind == "partial" else [],
        )
        document = with_manifest_digest(document, digest_key="integrity_sha256")
    path = receipts.receipt_path(snapshot, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _report(root: Path, *, now=NOW, day=DAY):
    return status.build_recommendation_status(
        day, now=now, snapshot_root=root / "snapshots", receipt_root=root / "receipts",
    )


@pytest.mark.parametrize(
    "clock,preopen,open_state",
    [
        ("08:44:59", "waiting", "waiting"),
        ("08:45:00", "pending", "waiting"),
        ("08:59:59", "pending", "waiting"),
        ("09:00:00", "missing_decision", "pending"),
        ("09:20:59", "missing_decision", "pending"),
        ("09:21:00", "missing_decision", "missing_decision"),
    ],
)
def test_missing_evidence_boundaries(tmp_path, clock, preopen, open_state):
    report = _report(tmp_path, now=datetime.fromisoformat(f"{DAY}T{clock}+09:00"))
    assert report["slots"]["preopen"]["state"] == preopen
    assert report["slots"]["open"]["state"] == open_state
    assert all(row["candidate_count"] is None for row in report["slots"].values())
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("day", ["20260820", "2026-8-20", "2026-08-20/../x", 1, None])
def test_noncanonical_dates_rejected(tmp_path, day):
    with pytest.raises((ValueError, TypeError)):
        _report(tmp_path, day=day)


def test_naive_clock_rejected_and_utc_converted(tmp_path):
    with pytest.raises(ValueError, match="timezone-aware"):
        _report(tmp_path, now=datetime(2026, 8, 20, 10))
    assert _report(tmp_path, now=NOW.astimezone(timezone.utc)) == _report(tmp_path)


@pytest.mark.parametrize("count,expected", [(3, "delivered_candidates"), (0, "delivered_empty")])
def test_valid_deliveries_and_empty_is_not_quality_rejection(tmp_path, count, expected):
    for slot in status.SLOTS:
        snapshot = _write_snapshot(tmp_path / "snapshots", slot=slot, count=count)
        _write_receipt(tmp_path / "receipts", snapshot)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    report = _report(tmp_path)
    assert report["attention_required"] is False
    for row in report["slots"].values():
        assert row["state"] == expected
        assert row["artifact_state"] == "valid"
        assert row["delivery_state"] == "server_accepted"
        assert row["candidate_count"] == count
        assert row["read_confirmation"] == "unavailable"
        assert row["actual_input_candle_date"] is None
        assert row["actual_input_age_seconds"] is None
    assert report["slots"]["open"]["feature_row_date"] == DAY
    assert report["slots"]["open"]["nominal_input_candle_date"] == "2026-08-19"
    assert report["slots"]["preopen"]["feature_row_date"] == "2026-08-19"
    assert report["slots"]["preopen"]["nominal_input_candle_date"] == "2026-08-18"
    assert any("품질 조건 탈락을 뜻하지" in notice for notice in report["notices"])
    after = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert after == before


@pytest.mark.parametrize("clock,expected", [("09:06:00", "pending"), ("09:21:00", "delivery_uncertain")])
def test_snapshot_without_receipt_is_never_proven_non_delivery(tmp_path, clock, expected):
    _write_snapshot(tmp_path / "snapshots")
    row = _report(tmp_path, now=datetime.fromisoformat(f"{DAY}T{clock}+09:00"))["slots"]["open"]
    assert row["state"] == expected
    assert row["candidate_count"] == 3
    assert row["receipt_state"] == "missing"


@pytest.mark.parametrize("kind,expected", [("rejected", "not_delivered"), ("partial", "delivery_uncertain"), ("ambiguous", "delivery_uncertain")])
def test_valid_failed_receipts_have_distinct_meanings(tmp_path, kind, expected):
    snapshot = _write_snapshot(tmp_path / "snapshots")
    _write_receipt(tmp_path / "receipts", snapshot, kind=kind)
    row = _report(tmp_path)["slots"]["open"]
    assert row["state"] == expected
    assert row["artifact_state"] == "valid"
    assert row["receipt_state"] == "valid"


@pytest.mark.parametrize("old_failure", [False, True])
def test_pending_attempt_overrides_missing_or_old_failed_receipt(tmp_path, old_failure):
    snapshot = _write_snapshot(tmp_path / "snapshots")
    root = tmp_path / "receipts"
    if old_failure:
        _write_receipt(root, snapshot, kind="rejected")
    prior = receipts.read_delivery_receipt(snapshot, root=root)
    now = datetime.fromisoformat(f"{DAY}T09:05:05+09:00")
    with attempts.delivery_attempt_journal(snapshot, receipt_root=root) as journal:
        intent = journal.begin("message", now=now, receipt=prior)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    report = _report(tmp_path, now=now)
    row = report["slots"]["open"]
    assert row["state"] == "delivery_uncertain"
    assert row["reason"] == "unresolved_delivery_attempt"
    assert row["attempt_state"] == "pending"
    assert row["latest_attempt_id"] == intent["attempt_id"]
    assert row["attempt_count"] == 1
    assert report["attention_required"] is True
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("kind,expected", [
    ("success", "delivered_candidates"), ("rejected", "not_delivered"),
    ("ambiguous", "delivery_uncertain"), ("partial", "delivery_uncertain"),
])
def test_matching_receipt_resolves_attempt_without_status_writing_result(tmp_path, kind, expected):
    snapshot = _write_snapshot(tmp_path / "snapshots")
    root = tmp_path / "receipts"
    now = datetime.fromisoformat(f"{DAY}T09:05:02+09:00")
    with attempts.delivery_attempt_journal(snapshot, receipt_root=root) as journal:
        journal.begin("message", now=now, receipt=None)
    _write_receipt(root, snapshot, kind=kind, message="message")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    row = _report(tmp_path)["slots"]["open"]
    assert row["state"] == expected
    assert row["attempt_state"] == "resolved"
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert not list(root.rglob("*.result.json"))


def test_tampered_attempt_is_invalid_without_exposing_content(tmp_path):
    snapshot = _write_snapshot(tmp_path / "snapshots")
    root = tmp_path / "receipts"
    with attempts.delivery_attempt_journal(snapshot, receipt_root=root) as journal:
        journal.begin("message", now=datetime.fromisoformat(f"{DAY}T09:05:02+09:00"), receipt=None)
    path = next(root.rglob("*.intent.json"))
    path.write_text('{"secret": "must-not-appear"}')
    report = _report(tmp_path)
    assert report["slots"]["open"]["state"] == "invalid_evidence"
    assert report["slots"]["open"]["attempt_state"] == "invalid"
    assert "must-not-appear" not in json.dumps(report)


def test_orphan_attempt_journal_cannot_be_reported_as_waiting(tmp_path):
    path = tmp_path / "receipts" / DAY / "open_r1.attempts"
    path.mkdir(parents=True)
    row = _report(tmp_path, now=datetime.fromisoformat(f"{DAY}T08:00:00+09:00"))["slots"]["open"]
    assert row["state"] == "invalid_evidence"
    assert row["reason"] == "orphan_attempt_journal"


@pytest.mark.parametrize("corrupt", [False, True])
def test_pending_published_during_receipt_read_is_not_reported_as_old_failure(tmp_path, monkeypatch, corrupt):
    snapshot = _write_snapshot(tmp_path / "snapshots")
    root = tmp_path / "receipts"
    _write_receipt(root, snapshot, kind="rejected")
    original = status.read_delivery_receipt

    def interleaved_read(*args, **kwargs):
        # The sender captured its clock before a slow durable publication.
        # The journal appears after the report's first inspection, with a
        # timestamp that is still before the report's observed_at.
        previous = original(*args, **kwargs)
        with attempts.delivery_attempt_journal(snapshot, receipt_root=root) as journal:
            journal.begin("message", now=datetime.fromisoformat(f"{DAY}T09:05:05+09:00"), receipt=previous)
        if corrupt:
            next(root.rglob("*.intent.json")).write_text("damaged")
        return previous

    monkeypatch.setattr(status, "read_delivery_receipt", interleaved_read)
    row = _report(tmp_path)["slots"]["open"]
    if corrupt:
        assert row["state"] == "invalid_evidence"
        assert row["reason"] == "attempt_journal_validation_failed"
        assert row["attempt_state"] == "invalid"
    else:
        assert row["state"] == "delivery_uncertain"
        assert row["reason"] == "delivery_evidence_changed_during_read"
        assert row["attempt_state"] == "pending"


def test_orphan_receipt_is_invalid_even_before_slot(tmp_path):
    path = tmp_path / "receipts" / DAY / "open_r1.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    row = _report(tmp_path, now=datetime.fromisoformat(f"{DAY}T08:00:00+09:00"))["slots"]["open"]
    assert row["state"] == "invalid_evidence"
    assert row["reason"] == "orphan_receipt"
    assert row["candidate_count"] is None


@pytest.mark.parametrize("target", ["snapshot", "receipt"])
@pytest.mark.parametrize("damage", ["json", "checksum", "dangling_symlink"])
def test_invalid_evidence_never_becomes_missing_or_success(tmp_path, target, damage):
    snapshot = _write_snapshot(tmp_path / "snapshots")
    receipt = _write_receipt(tmp_path / "receipts", snapshot)
    path = Path(snapshot["snapshot_path"]) if target == "snapshot" else receipt
    if damage == "json":
        path.write_text('{"a":1,"a":2}')
    elif damage == "checksum":
        data = json.loads(path.read_text())
        data["integrity_sha256" if target == "receipt" else "payload_sha256"] = "0" * 64
        path.write_text(json.dumps(data))
    else:
        path.unlink()
        path.symlink_to(tmp_path / "missing")
    row = _report(tmp_path)["slots"]["open"]
    assert row["state"] == "invalid_evidence"
    assert row["artifact_state"] == "invalid"


def test_moved_snapshot_is_not_rebound_to_old_receipt(tmp_path):
    original = tmp_path / "original"
    snapshot = _write_snapshot(original / "snapshots")
    _write_receipt(original / "receipts", snapshot)
    moved = tmp_path / "moved"
    shutil.copytree(original, moved)
    assert _report(moved)["slots"]["open"]["state"] == "invalid_evidence"


def test_parent_directory_alias_preserves_canonical_identity(tmp_path):
    original = tmp_path / "original"
    for slot in status.SLOTS:
        snapshot = _write_snapshot(original / "snapshots", slot=slot)
        _write_receipt(original / "receipts", snapshot)
    alias = tmp_path / "alias"
    alias.symlink_to(original, target_is_directory=True)
    assert _report(alias) == _report(original)
    assert _report(alias)["attention_required"] is False


def test_future_evidence_is_invalid(tmp_path):
    snapshot = _write_snapshot(tmp_path / "snapshots")
    _write_receipt(tmp_path / "receipts", snapshot)
    for clock in ["09:04:59", "09:05:02"]:
        row = _report(tmp_path, now=datetime.fromisoformat(f"{DAY}T{clock}+09:00"))["slots"]["open"]
        assert row["state"] == "invalid_evidence"


def test_attempt_cannot_precede_snapshot_completion(tmp_path):
    snapshot = _write_snapshot(tmp_path / "snapshots")
    _write_receipt(tmp_path / "receipts", snapshot, delta=-2)
    assert _report(tmp_path)["slots"]["open"]["state"] == "invalid_evidence"


def test_legacy_receipt_is_not_server_confirmation(tmp_path):
    day = "2026-07-26"
    snapshot = _write_snapshot(tmp_path / "snapshots", day=day)
    _write_receipt(tmp_path / "receipts", snapshot)
    row = _report(tmp_path, day=day)["slots"]["open"]
    assert row["state"] == "delivery_uncertain"
    assert row["reason"] == "legacy_local_receipt_only"


def test_fallback_ranking_is_diagnostic_only(tmp_path):
    snapshot = _write_snapshot(tmp_path / "snapshots", rank_basis="score_rank(fallback)")
    _write_receipt(tmp_path / "receipts", snapshot)
    row = _report(tmp_path)["slots"]["open"]
    assert row["state"] == "invalid_evidence"
    assert row["reason"] == "unsupported_current_r1_identity"


def test_report_contract_matches_sender_without_invoking_it():
    from scripts import recommend_send

    assert status._LIVE_SEND_WINDOWS == recommend_send.LIVE_SEND_WINDOWS
    assert status.APPROVED_RANK_BASIS == recommend_send.APPROVED_LIVE_R1_RANK_BASIS
    assert status.APPROVED_SCORE_SCHEMA == recommend_send.APPROVED_LIVE_SCORE_SCHEMA
    assert status.APPROVED_RULE_VERSION == recommend_send.APPROVED_LIVE_R1_RULE_VERSION
    assert cli.ROOT == snapshots._ROOT


def _cli_args(root):
    return ["--asof", DAY, "--now", NOW.isoformat(), "--snapshot-root", str(root / "snapshots"), "--receipt-root", str(root / "receipts")]


def test_cli_json_and_text_stdout_are_readonly(tmp_path, capsys):
    assert cli.main(_cli_args(tmp_path)) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["slots"]["open"]["candidate_count"] is None
    assert cli.main([*_cli_args(tmp_path), "--format", "text"]) == 1
    assert "판단·전달 증거 없음" in capsys.readouterr().out
    assert not list(tmp_path.iterdir())


def test_cli_cannot_create_missing_default_label_even_with_custom_inputs(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    parent = tmp_path / "output/recommend_score_labels" / DAY
    parent.mkdir(parents=True)
    with pytest.raises(SystemExit) as error:
        cli.main([*_cli_args(tmp_path), "--output", str(parent / "open_r1.json")])
    assert error.value.code == 2
    assert list(parent.iterdir()) == []


@pytest.mark.parametrize("mode", ["existing", "symlink", "dangling_symlink", "missing_parent", "symlink_parent", "evidence_tree"])
def test_cli_refuses_unsafe_output_without_overwriting(tmp_path, mode, capsys):
    path = tmp_path / "status.json"
    original = tmp_path / "original.json"
    original.write_text("keep me")
    if mode == "existing":
        path.write_text("also keep")
    elif mode in {"symlink", "dangling_symlink"}:
        path.symlink_to(original if mode == "symlink" else tmp_path / "absent")
    elif mode == "missing_parent":
        path = tmp_path / "absent_dir" / "status.json"
    elif mode == "symlink_parent":
        parent = tmp_path / "linked"
        parent.symlink_to(tmp_path, target_is_directory=True)
        path = parent / "status.json"
    else:
        parent = tmp_path / "snapshots" / DAY
        parent.mkdir(parents=True)
        path = parent / "open_r1.json"
    with pytest.raises(SystemExit) as error:
        cli.main([*_cli_args(tmp_path), "--output", str(path)])
    assert error.value.code == 2
    assert original.read_text() == "keep me"
    if mode == "existing":
        assert path.read_text() == "also keep"
    assert not capsys.readouterr().out


def test_cli_explicit_output_is_new_private_file_only(tmp_path, capsys):
    path = tmp_path / "status.json"
    assert cli.main([*_cli_args(tmp_path), "--output", str(path)]) == 1
    assert path.read_text() == capsys.readouterr().out
    assert path.stat().st_mode & 0o777 == 0o600
    assert list(tmp_path.iterdir()) == [path]


def test_cli_subprocess_uses_explicit_fixture_roots(tmp_path):
    result = subprocess.run(
        [sys.executable, str(Path(cli.__file__).resolve()), *_cli_args(tmp_path)],
        cwd=tmp_path, capture_output=True, text=True,
        env={**os.environ, "PRELUDE_FORBID_TELEGRAM": "1", "PYTHONDONTWRITEBYTECODE": "1"},
        check=False,
    )
    assert result.returncode == 1, result.stderr
    assert json.loads(result.stdout)["schema"] == status.REPORT_SCHEMA
    assert not list(tmp_path.iterdir())

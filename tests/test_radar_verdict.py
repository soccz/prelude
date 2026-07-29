from __future__ import annotations

import json
import hashlib
import os
import sys
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import notifier.delivery_receipt as receipt_module
import notifier.telegram as telegram
import ops.policy_competition as policy_competition
import ops.radar_verdict as radar_verdict
import scripts.pump_detector_v2_today as v2_runner
import scripts.recommend_send as recommend_send
import scripts.v2_scoreboard as scoreboard
from notifier.telegram import TelegramSendResult, TelegramServerMessage
from ops.radar_verdict import (
    JUDGMENT_DAY,
    RadarVerdictError,
    assert_radar_send_allowed,
    load_terminal_verdict,
    record_terminal_verdict,
    recover_terminal_verdict,
    terminal_candidate,
    verdict_status_for,
)
from scripts.v2_scoreboard import build_scoreboard, run_scoreboard


@pytest.fixture(autouse=True)
def _allow_mocked_sends(monkeypatch):
    # 이 모듈은 발송 경로 자체를 mock 된 requests 로 검증한다 —
    # 전역 kill-switch(tests/conftest.py)를 in-process 한정 해제.
    # 해제 상태에서도 미래 테스트의 mock 누락이 실 API 호출로 이어지지
    # 않도록 가장 낮은 HTTP 경계를 기본 폭탄으로 봉쇄한다.
    monkeypatch.delenv("PRELUDE_FORBID_TELEGRAM", raising=False)

    def forbid_unmocked_post(*_args, **_kwargs):
        pytest.fail("unmocked Telegram HTTP transport reached")

    monkeypatch.setattr(telegram.requests, "post", forbid_unmocked_post)


def _scorecard(
    *,
    status: str,
    closed_n: int,
    mean: float | None,
    ci95: list[float] | None,
    per_day_t: float | None,
    regimes: list[str],
    early_kill: bool,
) -> dict:
    criteria = {
        "n>=200": closed_n >= 200,
        "mean>0": mean is not None and mean > 0,
        "CI_0_제외": ci95 is not None and ci95[0] > 0,
        "2레짐_or_t>=2": (
            len(regimes) >= 2
            or (per_day_t is not None and per_day_t >= 2)
        ),
    }
    return {
        "status": status,
        "closed_n": closed_n,
        "mean_net_pct": mean,
        "ci95": ci95,
        "per_day_t": per_day_t,
        "regimes": regimes,
        "criteria": criteria,
        "criteria_met": sum(criteria.values()),
        "early_kill_breached": early_kill,
        "terminal_metric_values": {
            "mean_net_pct": mean,
            "ci95": ci95,
            "per_day_t": per_day_t,
        },
    }


def _candidate(
    *,
    verdict: str,
    recorded_at: datetime | None = None,
) -> dict:
    if verdict == "early_kill":
        scorecard = _scorecard(
            status="early_kill",
            closed_n=1,
            mean=-0.5,
            ci95=None,
            per_day_t=None,
            regimes=["bear_quiet"],
            early_kill=True,
        )
        asof = date(2026, 7, 1)
    elif verdict == "judgment_kill":
        scorecard = _scorecard(
            status="judgment_kill",
            closed_n=199,
            mean=0.5,
            ci95=[0.1, 0.9],
            per_day_t=3.0,
            regimes=["bear_quiet", "bull_quiet"],
            early_kill=False,
        )
        asof = JUDGMENT_DAY
    elif verdict == "judgment_go":
        scorecard = _scorecard(
            status="judgment_go",
            closed_n=200,
            mean=0.5,
            ci95=[0.1, 0.9],
            per_day_t=3.0,
            regimes=["bear_quiet", "bull_quiet"],
            early_kill=False,
        )
        asof = JUDGMENT_DAY
    else:
        raise AssertionError(verdict)
    candidate = terminal_candidate(
        scorecard,
        asof=asof,
        recorded_at=recorded_at or datetime(
            asof.year,
            asof.month,
            asof.day,
            tzinfo=timezone.utc,
        ),
    )
    assert candidate is not None
    return candidate


def _closed_rows(n: int, *, net: float) -> list[dict]:
    regimes = ("bull_quiet", "bear_quiet")
    return [
        {
            "date": "2026-08-30",
            "coin": f"KRW-T{index:03d}",
            "status": "closed",
            "realized_pct": str(net),
            "btc_regime": regimes[index % 2],
        }
        for index in range(n)
    ]


def _v2_result() -> dict:
    return {
        "asof": "2026-07-26",
        "model_id": "pump_hunter_v2",
        "rule_version": "pump_detector_v2",
        "rule": v2_runner.PUMP_V2_RULE,
        "feature_date": "2026-07-25T09:00:00",
        "btc_regime": "bear_quiet",
        "universe_n": 100,
        "binance_status": "ok",
        "n_candidates": 1,
        "candidates": [
            {
                "market": "KRW-TEST",
                "rank": 1,
                "score": 0.9,
                "entry_open": 100.0,
                "roc_7d": 0.12,
                "roc_7d_rank": 0.95,
                "atr_pct_14": 0.08,
                "log_return_1d": 0.01,
                "b_vol_surge": 3.2,
                "b_ret_1d": 0.05,
                "liq_rank_daily": 10,
                "btc_regime": "bear_quiet",
                "rule_id": v2_runner.PUMP_V2_RULE_ID,
            }
        ],
        "oos": dict(v2_runner.PUMP_V2_OOS),
    }


def _r1_snapshot(
    tmp_path: Path,
    *,
    asof: str = "2026-07-25",
) -> dict:
    return {
        "asof": asof,
        "slot": "open",
        "snapshot_id": f"recommend-test-{asof}-open",
        "snapshot_path": str(tmp_path / "snapshot.json"),
        "btc_regime": "neutral",
        "universe_n": 0,
        "calibration_source": "test",
        "rank_basis": recommend_send.APPROVED_LIVE_R1_RANK_BASIS,
        "snapshot_schema": recommend_send.SNAPSHOT_SCHEMA_VERSION,
        "score_schema_version": recommend_send.APPROVED_LIVE_SCORE_SCHEMA,
        "rule_version": recommend_send.APPROVED_LIVE_R1_RULE_VERSION,
        "decision_started_at": f"{asof}T09:05:00+09:00",
        "decision_completed_at": f"{asof}T09:05:00+09:00",
        "ranking": "R1",
        "top3": [],
        "request": {
            "asof": asof,
            "slot": "open",
            "ranking": "R1",
            "limit_markets": None,
        },
        "model": {"id": "recommend_r1_open", "ranking": "R1"},
    }


def test_frozen_scoreboard_has_no_undecided_status_at_or_after_deadline():
    before = build_scoreboard(
        _closed_rows(1, net=0.5),
        today=date(2026, 8, 31),
    )
    deadline = build_scoreboard(
        _closed_rows(1, net=0.5),
        today=JUDGMENT_DAY,
    )
    after = build_scoreboard(
        _closed_rows(1, net=0.5),
        today=date(2026, 9, 2),
    )

    assert before["status"] == "insufficient_sample"
    assert before["terminal_verdict"] is None
    assert deadline["status"] == "judgment_kill"
    assert deadline["terminal_verdict"] == "kill"
    assert after["status"] == "judgment_kill"
    assert after["terminal_verdict"] == "kill"


def test_frozen_scoreboard_go_requires_all_four_criteria():
    result = build_scoreboard(
        _closed_rows(200, net=1.0),
        today=JUDGMENT_DAY,
    )

    assert result["criteria_met"] == 4
    assert result["status"] == "judgment_go"
    assert result["terminal_verdict"] == "go"


def test_delayed_deadline_run_excludes_post_cutoff_cohort():
    rows = _closed_rows(199, net=1.0)
    rows.append({
        "date": "2026-09-01",
        "coin": "KRW-POST",
        "status": "closed",
        "realized_pct": "100.0",
        "btc_regime": "bull_quiet",
    })

    result = build_scoreboard(rows, today=date(2026, 9, 2))

    assert result["closed_n"] == 199
    assert result["post_judgment_closed_excluded"] == 1
    assert result["status"] == "judgment_kill"
    assert result["criteria"]["n>=200"] is False


def test_delayed_judgment_state_is_effective_from_frozen_deadline():
    candidate = terminal_candidate(
        _scorecard(
            status="judgment_kill",
            closed_n=199,
            mean=0.5,
            ci95=[0.1, 0.9],
            per_day_t=3.0,
            regimes=["bear_quiet", "bull_quiet"],
            early_kill=False,
        ),
        asof=date(2026, 9, 2),
        recorded_at=datetime(2026, 9, 2, tzinfo=timezone.utc),
    )

    assert candidate is not None
    assert candidate["effective_asof"] == JUDGMENT_DAY.isoformat()
    assert candidate["status"] == "judgment_kill"


def test_predeadline_negative_mean_is_immediate_terminal_kill():
    result = build_scoreboard(
        _closed_rows(1, net=-0.01),
        today=date(2026, 8, 31),
    )

    assert result["status"] == "early_kill"
    assert result["early_kill_breached"] is True
    assert result["terminal_verdict"] == "kill"


def test_deadline_missing_ledger_stays_pending_and_fails_closed(tmp_path):
    terminal_path = tmp_path / "terminal.json"
    code, payload = run_scoreboard(
        ledger=tmp_path / "missing.csv",
        output=tmp_path / "scoreboard.json",
        today=JUDGMENT_DAY,
        decision_root=tmp_path / "decisions",
        receipt_root=tmp_path / "receipts",
        terminal_state=terminal_path,
        verdict_recorded_at=datetime(
            2026,
            9,
            1,
            tzinfo=timezone.utc,
        ),
    )

    state = load_terminal_verdict(terminal_path)
    assert code == 2
    assert payload["status"] == "ledger_missing"
    assert payload["terminal_state"]["status"] == "pending"
    assert payload["terminal_verdict"] is None
    assert state is None


def test_scoreboard_captures_asof_only_after_output_lock(
    tmp_path,
    monkeypatch,
):
    events: list[str] = []

    @contextmanager
    def fake_lock(_path):
        events.append("locked")
        yield

    class FakeDatetime:
        @classmethod
        def now(cls, _tz):
            assert events == ["locked"]
            return datetime(
                2026,
                9,
                1,
                0,
                0,
                tzinfo=scoreboard.KST,
            )

    monkeypatch.setattr(scoreboard, "_exclusive_output_lock", fake_lock)
    monkeypatch.setattr(scoreboard, "datetime", FakeDatetime)
    terminal_path = tmp_path / "terminal.json"

    code, payload = run_scoreboard(
        ledger=tmp_path / "missing.csv",
        output=tmp_path / "scoreboard.json",
        terminal_state=terminal_path,
        verdict_recorded_at=datetime(
            2026,
            9,
            1,
            tzinfo=timezone.utc,
        ),
    )

    assert code == 2
    assert payload["asof"] == JUDGMENT_DAY.isoformat()
    assert payload["status"] == "ledger_missing"
    assert payload["terminal_state"]["status"] == "pending"


def test_terminal_record_is_idempotent_but_cannot_flip(tmp_path):
    path = tmp_path / "terminal.json"
    first = _candidate(
        verdict="early_kill",
        recorded_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
    )
    retry = _candidate(
        verdict="early_kill",
        recorded_at=datetime(2026, 7, 1, 1, tzinfo=timezone.utc),
    )

    recorded = record_terminal_verdict(first, path=path)
    assert record_terminal_verdict(retry, path=path) == recorded
    with pytest.raises(RadarVerdictError, match="immutable"):
        record_terminal_verdict(
            _candidate(verdict="judgment_go"),
            path=path,
        )
    assert load_terminal_verdict(path) == recorded


def test_tampered_terminal_record_fails_closed(tmp_path):
    path = tmp_path / "terminal.json"
    record_terminal_verdict(_candidate(verdict="early_kill"), path=path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["source_status"] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RadarVerdictError, match="checksum"):
        load_terminal_verdict(path)
    with pytest.raises(RadarVerdictError):
        assert_radar_send_allowed(
            path=path,
            now=datetime(2026, 7, 25, 9, 5, tzinfo=recommend_send.KST),
        )


@pytest.mark.parametrize("corruption", ["duplicate_key", "nan"])
def test_terminal_record_uses_strict_json(tmp_path, corruption):
    path = tmp_path / "terminal.json"
    record_terminal_verdict(_candidate(verdict="early_kill"), path=path)
    raw = path.read_text(encoding="utf-8")
    if corruption == "duplicate_key":
        raw = raw.replace(
            '"verdict": "kill"',
            '"verdict": "kill", "verdict": "kill"',
            1,
        )
    else:
        raw = raw.replace('"closed_n": 1', '"closed_n": NaN', 1)
    path.write_text(raw, encoding="utf-8")

    with pytest.raises(RadarVerdictError, match="unreadable"):
        load_terminal_verdict(path)


def test_dangling_terminal_pair_cannot_reopen_send_gate(tmp_path):
    path = tmp_path / "terminal.json"
    anchor = path.with_name(f"{path.name}.anchor")
    record_terminal_verdict(_candidate(verdict="early_kill"), path=path)
    path.unlink()
    anchor.unlink()
    path.symlink_to(tmp_path / "missing-state.json")
    anchor.symlink_to(tmp_path / "missing-anchor.json")

    with pytest.raises(RadarVerdictError, match="unreadable"):
        load_terminal_verdict(path)
    with pytest.raises(RadarVerdictError):
        assert_radar_send_allowed(
            path=path,
            now=datetime(2026, 7, 25, 9, 5, tzinfo=recommend_send.KST),
        )


def test_dangling_terminal_entry_is_not_treated_as_absent_pair(tmp_path):
    path = tmp_path / "terminal.json"
    path.symlink_to(tmp_path / "missing-state.json")

    with pytest.raises(RadarVerdictError, match="incomplete"):
        load_terminal_verdict(path)


def test_recorded_at_tamper_is_covered_by_document_integrity(tmp_path):
    path = tmp_path / "terminal.json"
    record_terminal_verdict(_candidate(verdict="early_kill"), path=path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["recorded_at"] = "2099-01-01T00:00:00+00:00"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RadarVerdictError, match="integrity"):
        load_terminal_verdict(path)


def test_future_recorded_terminal_state_cannot_authorize_earlier_send(tmp_path):
    path = tmp_path / "terminal.json"
    future_candidate = terminal_candidate(
        _scorecard(
            status="judgment_go",
            closed_n=200,
            mean=0.5,
            ci95=[0.1, 0.9],
            per_day_t=3.0,
            regimes=["bear_quiet", "bull_quiet"],
            early_kill=False,
        ),
        asof=date(2099, 1, 1),
        recorded_at=datetime(2099, 1, 1, tzinfo=timezone.utc),
    )
    assert future_candidate is not None
    record_terminal_verdict(future_candidate, path=path)

    with pytest.raises(RadarVerdictError, match="recorded in future"):
        assert_radar_send_allowed(
            path=path,
            now=datetime(2026, 9, 1, 9, 5, tzinfo=recommend_send.KST),
        )


def test_deleted_early_kill_state_cannot_revert_to_predecision(tmp_path):
    path = tmp_path / "terminal.json"
    record_terminal_verdict(_candidate(verdict="early_kill"), path=path)
    path.unlink()

    with pytest.raises(RadarVerdictError, match="pair is incomplete"):
        assert_radar_send_allowed(
            path=path,
            now=datetime(2026, 7, 25, 9, 5, tzinfo=recommend_send.KST),
        )


def test_scoreboard_repairs_deleted_state_from_valid_anchor(tmp_path):
    path = tmp_path / "terminal.json"
    recorded = record_terminal_verdict(
        _candidate(verdict="early_kill"),
        path=path,
    )
    path.unlink()

    code, payload = run_scoreboard(
        ledger=tmp_path / "missing.csv",
        output=tmp_path / "scoreboard.json",
        today=date(2026, 7, 25),
        decision_root=tmp_path / "decisions",
        receipt_root=tmp_path / "receipts",
        terminal_state=path,
    )

    assert code == 21
    assert payload["status"] == "early_kill"
    assert payload["terminal_recovery"]["recovered"] is True
    assert load_terminal_verdict(path) == recorded


def test_recovery_repairs_corrupt_anchor_and_preserves_exact_evidence(
    tmp_path,
):
    path = tmp_path / "terminal.json"
    recorded = record_terminal_verdict(
        _candidate(verdict="early_kill"),
        path=path,
    )
    anchor = path.with_name(f"{path.name}.anchor")
    corrupt = b'{"truncated":'
    anchor.write_bytes(corrupt)

    assert recover_terminal_verdict(path=path) == recorded
    assert load_terminal_verdict(path) == recorded
    quarantined = list(
        tmp_path.glob(f"{anchor.name}.corrupt-*")
    )
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == corrupt


def test_recovery_repairs_corrupt_state_from_anchor_and_preserves_evidence(
    tmp_path,
):
    path = tmp_path / "terminal.json"
    recorded = record_terminal_verdict(
        _candidate(verdict="early_kill"),
        path=path,
    )
    corrupt = b"\xffnot-json"
    path.write_bytes(corrupt)

    assert recover_terminal_verdict(path=path) == recorded
    assert load_terminal_verdict(path) == recorded
    quarantined = list(tmp_path.glob(f"{path.name}.corrupt-*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == corrupt


def test_recovery_forces_kill_when_neither_mirror_is_valid(tmp_path):
    path = tmp_path / "terminal.json"
    anchor = path.with_name(f"{path.name}.anchor")
    path.write_bytes(b"{")
    anchor.write_bytes(b"[]")
    forced_kill = _candidate(verdict="judgment_kill")

    assert recover_terminal_verdict(
        path=path,
        forced_kill=forced_kill,
    ) == forced_kill
    assert load_terminal_verdict(path) == forced_kill
    quarantined = list(tmp_path.glob("*.corrupt-*"))
    assert {item.read_bytes() for item in quarantined} == {b"{", b"[]"}


def test_recovery_rejects_forced_go_without_mutating_corrupt_evidence(
    tmp_path,
):
    path = tmp_path / "terminal.json"
    path.write_bytes(b"{")

    with pytest.raises(RadarVerdictError, match="only force KILL"):
        recover_terminal_verdict(
            path=path,
            forced_kill=_candidate(verdict="judgment_go"),
        )

    assert path.read_bytes() == b"{"
    assert list(tmp_path.glob("*.corrupt-*")) == []


def test_recovery_without_forced_kill_preserves_mismatched_valid_mirrors(
    tmp_path,
):
    path = tmp_path / "terminal.json"
    anchor = path.with_name(f"{path.name}.anchor")
    early_kill = _candidate(verdict="early_kill")
    judgment_go = _candidate(verdict="judgment_go")
    path.write_text(json.dumps(early_kill), encoding="utf-8")
    anchor.write_text(json.dumps(judgment_go), encoding="utf-8")
    state_before = path.read_bytes()
    anchor_before = anchor.read_bytes()

    with pytest.raises(RadarVerdictError, match="irreconcilable"):
        recover_terminal_verdict(path=path)

    assert path.read_bytes() == state_before
    assert anchor.read_bytes() == anchor_before
    assert list(tmp_path.glob("*.corrupt-*")) == []


def test_recovery_forces_kill_and_quarantines_both_mismatched_mirrors(
    tmp_path,
):
    path = tmp_path / "terminal.json"
    anchor = path.with_name(f"{path.name}.anchor")
    early_kill = _candidate(verdict="early_kill")
    judgment_go = _candidate(verdict="judgment_go")
    forced_kill = _candidate(verdict="judgment_kill")
    path.write_text(json.dumps(early_kill), encoding="utf-8")
    anchor.write_text(json.dumps(judgment_go), encoding="utf-8")

    recovered = recover_terminal_verdict(
        path=path,
        forced_kill=forced_kill,
    )

    assert recovered == forced_kill
    assert load_terminal_verdict(path) == forced_kill
    quarantined = list(tmp_path.glob("*.corrupt-*"))
    assert len(quarantined) == 2
    quarantined_ids = {
        json.loads(item.read_text(encoding="utf-8"))["verdict_id"]
        for item in quarantined
    }
    assert quarantined_ids == {
        early_kill["verdict_id"],
        judgment_go["verdict_id"],
    }


def test_forced_mismatch_recovery_failure_cannot_leave_go_recoverable(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "terminal.json"
    anchor = path.with_name(f"{path.name}.anchor")
    early_kill = _candidate(verdict="early_kill")
    judgment_go = _candidate(verdict="judgment_go")
    path.write_text(json.dumps(early_kill), encoding="utf-8")
    anchor.write_text(json.dumps(judgment_go), encoding="utf-8")
    original_quarantine = radar_verdict._quarantine
    calls = 0

    def fail_second_quarantine(target, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected second quarantine failure")
        return original_quarantine(target, **kwargs)

    monkeypatch.setattr(
        radar_verdict,
        "_quarantine",
        fail_second_quarantine,
    )
    with pytest.raises(OSError, match="second quarantine failure"):
        recover_terminal_verdict(
            path=path,
            forced_kill=_candidate(verdict="judgment_kill"),
        )

    monkeypatch.setattr(
        radar_verdict,
        "_quarantine",
        original_quarantine,
    )
    with pytest.raises(RadarVerdictError, match="irreconcilable"):
        recover_terminal_verdict(path=path)


def test_forced_mismatch_partial_publish_stays_irreconcilable(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "terminal.json"
    anchor = path.with_name(f"{path.name}.anchor")
    path.write_text(
        json.dumps(_candidate(verdict="early_kill")),
        encoding="utf-8",
    )
    anchor.write_text(
        json.dumps(_candidate(verdict="judgment_go")),
        encoding="utf-8",
    )
    forced_kill = _candidate(verdict="judgment_kill")
    original_atomic_write = radar_verdict._atomic_write
    replacement_writes = 0

    def fail_second_replacement(target, payload, **kwargs):
        nonlocal replacement_writes
        if kwargs.get("replace_from") is not None:
            replacement_writes += 1
            if replacement_writes == 2:
                raise OSError("injected second replacement failure")
        return original_atomic_write(target, payload, **kwargs)

    monkeypatch.setattr(
        radar_verdict,
        "_atomic_write",
        fail_second_replacement,
    )
    with pytest.raises(OSError, match="second replacement failure"):
        recover_terminal_verdict(
            path=path,
            forced_kill=forced_kill,
        )

    monkeypatch.setattr(
        radar_verdict,
        "_atomic_write",
        original_atomic_write,
    )
    with pytest.raises(RadarVerdictError, match="irreconcilable"):
        recover_terminal_verdict(path=path)
    assert recover_terminal_verdict(
        path=path,
        forced_kill=forced_kill,
    ) == forced_kill
    assert load_terminal_verdict(path) == forced_kill


def test_recovery_quarantines_dangling_mirror_before_repair(tmp_path):
    path = tmp_path / "terminal.json"
    recorded = record_terminal_verdict(
        _candidate(verdict="early_kill"),
        path=path,
    )
    anchor = path.with_name(f"{path.name}.anchor")
    missing = tmp_path / "missing-anchor.json"
    anchor.unlink()
    anchor.symlink_to(missing)

    assert recover_terminal_verdict(path=path) == recorded
    assert load_terminal_verdict(path) == recorded
    quarantined = list(
        tmp_path.glob(f"{anchor.name}.corrupt-*")
    )
    assert len(quarantined) == 1
    assert quarantined[0].is_symlink()
    assert os.readlink(quarantined[0]) == str(missing)


def test_quarantine_does_not_overwrite_dangling_collision(tmp_path):
    path = tmp_path / "terminal.json"
    corrupt = b"{"
    path.write_bytes(corrupt)
    digest = hashlib.sha256(corrupt).hexdigest()[:16]
    collision = tmp_path / f"{path.name}.corrupt-{digest}"
    missing = tmp_path / "preexisting-missing.json"
    collision.symlink_to(missing)

    quarantined = radar_verdict._quarantine(path)

    assert quarantined == Path(f"{collision}-1")
    assert collision.is_symlink()
    assert os.readlink(collision) == str(missing)
    assert quarantined.read_bytes() == corrupt


def test_quarantine_rejects_source_replacement_after_hashing(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "terminal.json"
    path.write_bytes(b"first")
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(b"replacement")
    original_hash_fd = radar_verdict._hash_fd
    raced = False

    def racing_hash_fd(fd):
        nonlocal raced
        digest = original_hash_fd(fd)
        if not raced:
            raced = True
            replacement.replace(path)
        return digest

    monkeypatch.setattr(radar_verdict, "_hash_fd", racing_hash_fd)

    with pytest.raises(RadarVerdictError, match="changed"):
        radar_verdict._quarantine(path)

    assert path.read_bytes() == b"replacement"
    assert list(tmp_path.glob(f"{path.name}.corrupt-*")) == []


def test_recovery_rejects_pair_replacement_between_mirror_reads(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "terminal.json"
    anchor = path.with_name(f"{path.name}.anchor")
    record_terminal_verdict(_candidate(verdict="early_kill"), path=path)
    replacement = _candidate(verdict="judgment_go")
    original_try_read = radar_verdict._try_read_validated

    def racing_try_read(candidate_path):
        result = original_try_read(candidate_path)
        if candidate_path == anchor:
            path.write_text(json.dumps(replacement), encoding="utf-8")
        return result

    monkeypatch.setattr(
        radar_verdict,
        "_try_read_validated",
        racing_try_read,
    )

    with pytest.raises(RadarVerdictError, match="changed during recovery"):
        recover_terminal_verdict(path=path)


def test_atomic_write_rejects_symlink_without_touching_referent(tmp_path):
    referent = tmp_path / "outside.json"
    referent.write_text("keep", encoding="utf-8")
    path = tmp_path / "terminal.json"
    path.symlink_to(referent)

    with pytest.raises(RadarVerdictError, match="unsafe|already exists"):
        radar_verdict._atomic_write(
            path,
            _candidate(verdict="early_kill"),
        )

    assert path.is_symlink()
    assert referent.read_text(encoding="utf-8") == "keep"
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_atomic_write_rejects_target_appearing_during_serialization(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "terminal.json"
    original_dump = json.dump

    def racing_dump(payload, handle, **kwargs):
        original_dump(payload, handle, **kwargs)
        path.write_text("intruder", encoding="utf-8")

    monkeypatch.setattr(radar_verdict.json, "dump", racing_dump)

    with pytest.raises(RadarVerdictError, match="appeared"):
        radar_verdict._atomic_write(
            path,
            _candidate(verdict="early_kill"),
        )

    assert path.read_text(encoding="utf-8") == "intruder"
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_atomic_write_rejects_private_temp_replacement(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "terminal.json"
    original_dump = json.dump

    def racing_dump(payload, handle, **kwargs):
        original_dump(payload, handle, **kwargs)
        [temp_path] = list(tmp_path.glob(f".{path.name}.*.tmp"))
        replacement = tmp_path / "replacement.tmp"
        replacement.write_text(
            json.dumps(_candidate(verdict="judgment_go")),
            encoding="utf-8",
        )
        replacement.replace(temp_path)

    monkeypatch.setattr(radar_verdict.json, "dump", racing_dump)

    with pytest.raises(RadarVerdictError, match="temporary file changed"):
        radar_verdict._atomic_write(
            path,
            _candidate(verdict="early_kill"),
        )

    assert not path.exists()
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_atomic_write_publish_failure_cleans_private_temp(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "terminal.json"

    def fail_link(_source, _target, **_kwargs):
        raise OSError("injected publish failure")

    monkeypatch.setattr(radar_verdict.os, "link", fail_link)

    with pytest.raises(OSError, match="injected publish failure"):
        radar_verdict._atomic_write(
            path,
            _candidate(verdict="early_kill"),
        )

    assert not path.exists()
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_partial_pair_write_fails_closed_and_is_recoverable(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "terminal.json"
    anchor = path.with_name(f"{path.name}.anchor")
    candidate = _candidate(verdict="early_kill")
    original_atomic_write = radar_verdict._atomic_write

    def fail_anchor(target, payload):
        if target == anchor:
            raise OSError("injected anchor write failure")
        original_atomic_write(target, payload)

    monkeypatch.setattr(radar_verdict, "_atomic_write", fail_anchor)
    with pytest.raises(OSError, match="injected anchor write failure"):
        record_terminal_verdict(candidate, path=path)

    assert path.exists()
    assert not anchor.exists()
    with pytest.raises(RadarVerdictError, match="incomplete"):
        load_terminal_verdict(path)

    monkeypatch.setattr(
        radar_verdict,
        "_atomic_write",
        original_atomic_write,
    )
    assert recover_terminal_verdict(path=path) == candidate
    assert load_terminal_verdict(path) == candidate


def test_deadline_corrupt_state_is_invalid_without_synthetic_verdict(tmp_path):
    path = tmp_path / "terminal.json"
    path.write_text("{", encoding="utf-8")

    code, payload = run_scoreboard(
        ledger=tmp_path / "missing.csv",
        output=tmp_path / "scoreboard.json",
        today=JUDGMENT_DAY,
        decision_root=tmp_path / "decisions",
        receipt_root=tmp_path / "receipts",
        terminal_state=path,
        verdict_recorded_at=datetime(
            2026,
            9,
            1,
            tzinfo=timezone.utc,
        ),
    )

    assert code == 2
    assert payload["status"] == "terminal_state_invalid"
    assert payload["terminal_state"]["status"] == "invalid"
    with pytest.raises(RadarVerdictError):
        load_terminal_verdict(path)


def test_send_gate_requires_terminal_state_at_deadline(tmp_path):
    with pytest.raises(RadarVerdictError, match="missing after deadline"):
        assert_radar_send_allowed(
            path=tmp_path / "missing.json",
            now=datetime(2026, 9, 1, 9, 5, tzinfo=recommend_send.KST),
        )
    with pytest.raises(RadarVerdictError, match="missing"):
        verdict_status_for(
            JUDGMENT_DAY,
            path=tmp_path / "missing.json",
        )


def test_go_state_allows_send_gate_and_policy_status(tmp_path):
    path = tmp_path / "terminal.json"
    recorded = record_terminal_verdict(
        _candidate(verdict="judgment_go"),
        path=path,
    )

    assert assert_radar_send_allowed(
        path=path,
        now=datetime(2026, 9, 1, 9, 5, tzinfo=recommend_send.KST),
    ) == recorded
    status = verdict_status_for(JUDGMENT_DAY, path=path)
    assert status["status"] == "judgment_go"
    assert status["verdict"] == "go"
    assert status["effective"] is True


def test_policy_competition_persists_shared_terminal_status(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "terminal.json"
    recorded = record_terminal_verdict(
        _candidate(verdict="judgment_go"),
        path=path,
    )
    monkeypatch.setattr(policy_competition, "MODELS", [])

    payload = policy_competition.run(
        policy_competition.pd.Timestamp(JUDGMENT_DAY),
        output_csv=tmp_path / "policy.csv",
        output_json=tmp_path / "policy.json",
        db_path=None,
        snapshot_root=tmp_path / "snapshots",
        pump_v2_decision_root=tmp_path / "decisions",
        pump_v2_receipt_root=tmp_path / "receipts",
        radar_verdict_path=path,
    )

    assert payload["radar_terminal"]["verdict"] == "go"
    assert payload["radar_terminal"]["verdict_id"] == recorded["verdict_id"]
    persisted = json.loads(
        (tmp_path / "policy.json").read_text(encoding="utf-8")
    )
    assert persisted["radar_terminal"] == payload["radar_terminal"]


def test_r1_kill_gate_blocks_before_scoring(tmp_path, monkeypatch):
    path = tmp_path / "terminal.json"
    record_terminal_verdict(_candidate(verdict="early_kill"), path=path)
    monkeypatch.setattr(
        recommend_send,
        "_now_kst",
        lambda: datetime(
            2026,
            7,
            25,
            9,
            5,
            tzinfo=recommend_send.KST,
        ),
    )
    monkeypatch.setattr(
        recommend_send,
        "resolve_champion",
        lambda _slot: pytest.fail("scoring dispatch must not start"),
    )

    with pytest.raises(RadarVerdictError, match="KILL"):
        recommend_send.send_recommendation(
            "2026-07-25",
            "open",
            radar_verdict_path=path,
        )


@pytest.mark.parametrize(
    ("module", "entrypoint"),
    [
        (recommend_send, recommend_send.main),
        (v2_runner, v2_runner.main),
    ],
)
def test_live_cli_rejects_noncanonical_verdict_path(
    tmp_path,
    monkeypatch,
    module,
    entrypoint,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(module.__name__),
            "--radar-verdict",
            str(tmp_path / "bypass.json"),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        entrypoint()
    assert exc_info.value.code == 2


@pytest.mark.parametrize(
    "flag",
    ["--ledger", "--decision-root", "--receipt-root", "--output"],
)
def test_scoreboard_cli_rejects_noncanonical_evidence_paths(
    tmp_path,
    monkeypatch,
    flag,
):
    monkeypatch.setattr(
        sys,
        "argv",
        ["v2_scoreboard.py", flag, str(tmp_path / "bypass")],
    )

    with pytest.raises(SystemExit) as exc_info:
        scoreboard.main()
    assert exc_info.value.code == 2


def test_r1_kill_gate_blocks_immediately_before_api(tmp_path, monkeypatch):
    path = tmp_path / "terminal.json"
    record_terminal_verdict(_candidate(verdict="early_kill"), path=path)
    snapshot = _r1_snapshot(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        recommend_send,
        "_now_kst",
        lambda: datetime(
            2026,
            7,
            25,
            9,
            5,
            tzinfo=recommend_send.KST,
        ),
    )
    monkeypatch.setattr(
        recommend_send,
        "send_telegram",
        lambda message, **_kwargs: calls.append(message) or True,
    )

    with pytest.raises(RadarVerdictError, match="KILL"):
        recommend_send._send_and_record(
            snapshot,
            "radar",
            slot="open",
            receipt_root=tmp_path / "receipts",
            radar_verdict_path=path,
        )
    assert calls == []


@pytest.mark.parametrize("state_kind", ["missing", "malformed"])
def test_r1_deadline_state_failure_blocks_actual_api_boundary(
    tmp_path,
    monkeypatch,
    state_kind,
):
    path = tmp_path / "terminal.json"
    if state_kind == "malformed":
        path.write_text("{", encoding="utf-8")
    snapshot = _r1_snapshot(
        tmp_path,
        asof=JUDGMENT_DAY.isoformat(),
    )
    calls: list[str] = []
    monkeypatch.setattr(
        recommend_send,
        "_now_kst",
        lambda: datetime(
            2026,
            9,
            1,
            9,
            5,
            tzinfo=recommend_send.KST,
        ),
    )
    monkeypatch.setattr(
        recommend_send,
        "send_telegram",
        lambda message, **_kwargs: calls.append(message) or True,
    )

    with pytest.raises(RadarVerdictError):
        recommend_send._send_and_record(
            snapshot,
            "radar",
            slot="open",
            receipt_root=tmp_path / "receipts",
            radar_verdict_path=path,
        )
    assert calls == []


def test_go_verdict_allows_r1_and_v2_mock_api_boundaries(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "terminal.json"
    record_terminal_verdict(_candidate(verdict="judgment_go"), path=path)
    observed = datetime(
        2026,
        9,
        1,
        9,
        5,
        tzinfo=recommend_send.KST,
    )
    class R1Datetime(datetime):
        @classmethod
        def now(cls, tz=None):
            value = datetime(
                2026,
                9,
                1,
                0,
                5,
                2,
                tzinfo=timezone.utc,
            )
            return value if tz is None else value.astimezone(tz)

    class V2Datetime(datetime):
        @classmethod
        def now(cls, tz=None):
            value = datetime(
                2026,
                9,
                1,
                0,
                15,
                2,
                tzinfo=timezone.utc,
            )
            return value if tz is None else value.astimezone(tz)

    def transport_result(
        message: str,
        server_date: str,
    ) -> TelegramSendResult:
        digest = hashlib.sha256(message.encode()).hexdigest()
        return TelegramSendResult(
            delivery_ok=True,
            message_sha256=digest,
            chunk_count=1,
            chat_id_sha256=hashlib.sha256(b"456").hexdigest(),
            telegram_messages=(
                TelegramServerMessage(
                    message_id=101,
                    server_date=server_date,
                    text_sha256=digest,
                ),
            ),
            error=None,
        )

    monkeypatch.setattr(recommend_send, "_now_kst", lambda: observed)
    monkeypatch.setattr(recommend_send, "datetime", R1Datetime)
    monkeypatch.setattr(receipt_module, "datetime", R1Datetime)
    r1_calls: list[str] = []
    monkeypatch.setattr(
        recommend_send,
        "send_telegram_with_receipt",
        lambda message, **_kwargs: (
            r1_calls.append(message)
            or transport_result(
                message,
                "2026-09-01T00:05:02+00:00",
            )
        ),
    )
    snapshot = _r1_snapshot(
        tmp_path,
        asof=JUDGMENT_DAY.isoformat(),
    )

    assert recommend_send._send_and_record(
        snapshot,
        "r1-radar",
        slot="open",
        receipt_root=tmp_path / "r1-receipts",
        radar_verdict_path=path,
    )

    v2_observed = observed.replace(minute=15)
    monkeypatch.setattr(v2_runner, "_now_kst", lambda: v2_observed)
    monkeypatch.setattr(v2_runner, "datetime", V2Datetime)
    v2_calls: list[str] = []
    monkeypatch.setattr(
        v2_runner,
        "send_telegram_with_receipt",
        lambda message, **_kwargs: (
            v2_calls.append(message)
            or transport_result(
                message,
                "2026-09-01T00:15:02+00:00",
            )
        ),
    )
    result = _v2_result()
    result["asof"] = JUDGMENT_DAY.isoformat()
    result["feature_date"] = "2026-08-31T09:00:00"
    result = v2_runner._with_forward_provenance(result)

    receipt = v2_runner.deliver_once(
        result,
        "v2-radar",
        receipt_root=tmp_path / "v2-receipts",
        live_asof=JUDGMENT_DAY.isoformat(),
        radar_verdict_path=path,
    )

    assert receipt["delivery_ok"] is True
    assert r1_calls == ["r1-radar"]
    assert v2_calls == ["v2-radar"]


def test_r1_dry_run_remains_diagnostic_after_kill(tmp_path, monkeypatch):
    path = tmp_path / "terminal.json"
    record_terminal_verdict(_candidate(verdict="early_kill"), path=path)
    snapshot = _r1_snapshot(tmp_path)
    spec = SimpleNamespace(
        id="recommend_r1_open",
        predict_ref="signals.recommend:score_candidates",
    )
    monkeypatch.setattr(
        recommend_send,
        "resolve_champion",
        lambda _slot, **_kwargs: (spec, False, "test"),
    )
    monkeypatch.setattr(
        recommend_send,
        "call_predict",
        lambda *_args, **_kwargs: snapshot,
    )
    monkeypatch.setattr(
        recommend_send,
        "maybe_notify_champion_change",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        recommend_send,
        "send_telegram",
        lambda _message, *, dry_run=False: dry_run,
    )

    assert recommend_send.send_recommendation(
        snapshot["asof"],
        snapshot["slot"],
        dry_run=True,
        radar_verdict_path=path,
    )


def test_v2_kill_gate_blocks_before_scoring(tmp_path, monkeypatch):
    path = tmp_path / "terminal.json"
    record_terminal_verdict(_candidate(verdict="early_kill"), path=path)
    calls: list[str] = []
    monkeypatch.setattr(
        v2_runner,
        "_now_kst",
        lambda: datetime(
            2026,
            7,
            26,
            9,
            15,
            tzinfo=v2_runner.KST,
        ),
    )
    monkeypatch.setattr(
        v2_runner,
        "score_pump_v2_candidates",
        lambda *_args, **_kwargs: calls.append("score") or _v2_result(),
    )
    monkeypatch.setattr(v2_runner, "RADAR_VERDICT_PATH", path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pump_detector_v2_today.py",
            "--asof",
            "2026-07-26",
            "--ledger",
            str(tmp_path / "ledger.csv"),
        ],
    )

    assert v2_runner.main() == 1
    assert calls == []


def test_v2_kill_gate_blocks_immediately_before_api(tmp_path, monkeypatch):
    path = tmp_path / "terminal.json"
    record_terminal_verdict(_candidate(verdict="early_kill"), path=path)
    calls: list[str] = []
    monkeypatch.setattr(
        v2_runner,
        "_now_kst",
        lambda: datetime(
            2026,
            7,
            26,
            9,
            15,
            tzinfo=v2_runner.KST,
        ),
    )
    monkeypatch.setattr(
        v2_runner,
        "send_telegram",
        lambda message: calls.append(message) or True,
    )

    with pytest.raises(RadarVerdictError, match="KILL"):
        v2_runner.deliver_once(
            _v2_result(),
            "radar",
            receipt_root=tmp_path / "receipts",
            live_asof="2026-07-26",
            radar_verdict_path=path,
        )
    assert calls == []

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

import notifier.delivery_receipt as receipt_module
import ops.artifact_provenance as provenance
import scripts.label_recommend_snapshots as label_cli
import signals.recommend_score_labels as labels_module
from ledger.path_quality import PathAssessment
from notifier.delivery_receipt import DeliveryReceiptError, write_delivery_receipt
from notifier.telegram import TelegramSendResult, TelegramServerMessage
from signals.recommend_score_labels import (
    FORWARD_PROVENANCE_COHORT,
    LABEL_SCHEMA_VERSION,
    OFF_SCHEDULE_PROVENANCE_COHORT,
    SCHEDULED_REPLAY_PROVENANCE_COHORT,
    ScoreLabelError,
    _artifact_digest,
    _execution_metadata,
    _path_input_manifest,
    label_recommend_snapshot,
    load_label_artifact,
    path_window,
)
from signals.recommend_snapshot import get_or_create_recommend_snapshot, load_snapshot


def _telegram_result(message: str, server_date: str) -> TelegramSendResult:
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


def _candidate(rank: int, *, entry_open=None) -> dict:
    return {
        "coin": f"KRW-T{rank}",
        "rank": rank,
        "score": 0.9 - rank / 10,
        "pump_prob": 0.02,
        "pump_prob_pct": "2.0%",
        "rr_ratio": 1.0,
        "p_up5": 0.3,
        "p_up10": 0.1,
        "p_up20": 0.02,
        "p_dn5": 0.1,
        "p_dn10": 0.03,
        "exp_downside": -0.02,
        "dump_risk_flag": False,
        "entry_open": entry_open,
        "sl": -0.03,
        "tp": 0.05,
        "btc_regime": "neutral",
        "feature_values": {"f_ret_3d": 0.01},
    }


def _make_snapshot(root: Path, *, slot: str = "preopen") -> Path:
    def scorer(asof, *, limit_markets, slot, ranking):
        is_preopen = slot == "preopen"
        universe = [
            _candidate(i, entry_open=None if is_preopen else 100.0 + i)
            for i in range(1, 4)
        ]
        result = {
            "asof": asof,
            "slot": slot,
            "feature_date": "2026-07-23" if is_preopen else asof,
            "btc_regime": "neutral",
            "universe_n": 3,
            "calibration_source": "bucket_score_pump20",
            "rank_basis": "R1_riskreward(de-corr head)",
            "n_history_dates": 100,
            "ranking": ranking,
            "score_schema_version": "recommend_score.v1",
            "rule_version": "r1_riskreward_v1",
            "model_random_seed": 42,
            "feature_columns": ["f_ret_3d"],
            "training": {
                "start": "2025-01-01",
                "end": "2026-07-17" if is_preopen else "2026-07-18",
                "cutoff_exclusive": (
                    "2026-07-18" if is_preopen else "2026-07-19"
                ),
                "embargo_days": 5,
                "rows": 1000,
                "dates": 100,
            },
            "universe": universe,
            "top3": universe[:3],
        }
        return result

    result = get_or_create_recommend_snapshot(
        "2026-07-24", slot=slot, root=root, scorer=scorer
    )
    return Path(result["snapshot_path"])


def _timestamps(
    start: str | pd.Timestamp = "2026-07-24 09:00:00",
) -> tuple[pd.Timestamp, ...]:
    return tuple(
        pd.date_range(pd.Timestamp(start).tz_localize(None), periods=96, freq="15min")
    )


def _assessment(
    bars,
    *,
    quality: str = "complete",
    raw_bars: int = 96,
    complete: bool = True,
    start: str | pd.Timestamp = "2026-07-24 09:00:00",
) -> PathAssessment:
    return PathAssessment(
        bars=bars if complete else [],
        timestamps=_timestamps(start),
        path_complete=complete,
        path_quality=quality,
        raw_bars=raw_bars,
        expected_bars=96,
        flat_filled_bars=96 - raw_bars if complete else 0,
        benchmark_bars=96,
    )


def _complete_artifact(tmp_path: Path) -> Path:
    snapshot_file = _make_snapshot(tmp_path / "snapshots", slot="open")
    bars = [(100.0, 101.0, 99.0, 100.0) for _ in range(96)]
    result = label_recommend_snapshot(
        snapshot_file,
        output_root=tmp_path / "labels",
        receipt_root=tmp_path / "receipts",
        db_path=tmp_path / "15m.db",
        now="2026-07-25 09:16:00",
        assessor=lambda _market, start_at, **_kwargs: _assessment(
            bars,
            start=start_at,
        ),
    )
    return Path(result["artifact_path"])


def test_preopen_full_universe_labels_and_complete_artifact_are_idempotent(tmp_path):
    snapshot_file = _make_snapshot(tmp_path / "snapshots", slot="preopen")
    output_root = tmp_path / "labels"
    calls = 0

    def assessor(market, start_at, *, db_path):
        nonlocal calls
        calls += 1
        entry = {"KRW-T1": 100.0, "KRW-T2": 200.0, "KRW-T3": 50.0}[market]
        bars = [(entry, entry, entry, entry) for _ in range(96)]
        if market == "KRW-T1":
            bars[1] = (entry, entry * 1.06, entry * 0.99, entry * 1.02)
        elif market == "KRW-T2":
            bars[2] = (entry, entry * 1.06, entry * 0.96, entry)
        else:
            bars[-1] = (entry, entry * 1.02, entry * 0.99, entry * 1.02)
        return _assessment(bars, start=start_at)

    first = label_recommend_snapshot(
        snapshot_file,
        output_root=output_root,
        db_path=tmp_path / "15m.db",
        receipt_root=tmp_path / "receipts",
        now="2026-07-25 09:16:00",
        assessor=assessor,
    )
    second = label_recommend_snapshot(
        snapshot_file,
        output_root=output_root,
        db_path=tmp_path / "15m.db",
        receipt_root=tmp_path / "receipts",
        now="2026-07-25 10:00:00",
        assessor=lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("complete artifact must be reused")
        ),
    )

    assert calls == 3
    assert first["artifact_status"] == "complete"
    assert first["provenance_cohort"] == SCHEDULED_REPLAY_PROVENANCE_COHORT
    assert first["forward_eligible"] is False
    assert first["summary"] == {
        "snapshot_universe_n": 3,
        "rows": 3,
        "labeled": 3,
        "incomplete": 0,
        "flat_filled": 0,
    }
    assert second["artifact_reused"] is True
    assert second["label_payload_sha256"] == first["label_payload_sha256"]

    t1, t2, t3 = first["rows"]
    assert t1["actual_entry_open"] == 100.0  # preopen None을 09:00 path open으로 해소
    assert t1["up5"] is True
    assert t1["tp5_sl3_first_passage"] == "tp_first"
    assert t1["tp5_before_sl3"] is True
    assert t2["tp5_sl3_first_passage"] == "sl_first_same_bar"
    assert t2["tp5_before_sl3"] is False
    assert t3["eod_return"] == 0.020000000000000018
    assert all(r["snapshot_id"] == first["snapshot_id"] for r in first["rows"])
    assert all(r["snapshot_payload_sha256"] for r in first["rows"])
    assert all(
        r["provenance_cohort"] == SCHEDULED_REPLAY_PROVENANCE_COHORT
        for r in first["rows"]
    )

    raw = Path(first["artifact_path"]).read_text(encoding="utf-8")
    assert "NaN" not in raw
    assert json.loads(raw)["return_unit"] == "fraction"


@pytest.mark.parametrize("corruption", ["duplicate", "nan"])
def test_label_loader_rejects_ambiguous_json_even_when_old_digest_would_match(
    tmp_path,
    corruption,
):
    artifact = _complete_artifact(tmp_path)
    raw = artifact.read_text(encoding="utf-8")
    if corruption == "duplicate":
        marker = f'"schema": "{LABEL_SCHEMA_VERSION}"'
        raw = raw.replace(marker, f'{marker},\n  {marker}', 1)
    else:
        assert '"receipt_path": null' in raw
        raw = raw.replace('"receipt_path": null', '"receipt_path": NaN', 1)
    artifact.write_text(raw, encoding="utf-8")

    with pytest.raises(ScoreLabelError, match="label artifact read failed"):
        load_label_artifact(artifact)


def test_artifact_digest_does_not_normalize_nonfinite_values_to_null(tmp_path):
    artifact = _complete_artifact(tmp_path)
    document = json.loads(artifact.read_text(encoding="utf-8"))
    document["receipt_path"] = float("nan")

    with pytest.raises(ValueError, match="Out of range float values"):
        _artifact_digest(document)


@pytest.mark.parametrize(
    "corruption",
    ["missing_field", "unknown_field", "wrong_type", "summary_mismatch"],
)
def test_label_loader_rejects_checksum_valid_schema_corruption(
    tmp_path,
    corruption,
):
    artifact = _complete_artifact(tmp_path)
    document = json.loads(artifact.read_text(encoding="utf-8"))
    if corruption == "missing_field":
        del document["label_code"]
    elif corruption == "unknown_field":
        document["unbound_runtime_hint"] = "accept"
    elif corruption == "wrong_type":
        document["rows"][0]["score"] = "0.9"
    else:
        document["summary"]["rows"] += 1
    document["label_payload_sha256"] = _artifact_digest(document)
    artifact.write_text(
        json.dumps(document, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(ScoreLabelError, match="invalid label artifact"):
        load_label_artifact(artifact)


def test_label_loader_rejects_symlink(tmp_path):
    artifact = _complete_artifact(tmp_path)
    target = tmp_path / "outside-label.json"
    target.write_bytes(artifact.read_bytes())
    artifact.unlink()
    artifact.symlink_to(target)

    with pytest.raises(ScoreLabelError, match="label artifact read failed"):
        load_label_artifact(artifact)


def test_label_loader_rejects_replacement_during_read(tmp_path, monkeypatch):
    artifact = _complete_artifact(tmp_path)
    original_read_bytes = provenance.Path.read_bytes

    def replace_after_read(path):
        content = original_read_bytes(path)
        if path == artifact:
            artifact.write_bytes(content + b" ")
        return content

    monkeypatch.setattr(provenance.Path, "read_bytes", replace_after_read)

    with pytest.raises(ScoreLabelError, match="label artifact read failed"):
        load_label_artifact(artifact)


def test_not_mature_does_not_assess_or_write(tmp_path):
    snapshot_file = _make_snapshot(tmp_path / "snapshots", slot="open")
    output_root = tmp_path / "labels"

    result = label_recommend_snapshot(
        snapshot_file,
        output_root=output_root,
        db_path=tmp_path / "15m.db",
        receipt_root=tmp_path / "receipts",
        now="2026-07-25 08:59:59",
        assessor=lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("immature path must not be assessed")
        ),
    )

    assert result["artifact_status"] == "not_mature"
    assert result["written"] is False
    assert not Path(result["artifact_path"]).exists()


def test_delivered_open_waits_for_full_96_bars_after_execution_start(tmp_path):
    snapshot_file = _make_snapshot(tmp_path / "snapshots", slot="open")
    snapshot = load_snapshot(snapshot_file)
    receipt_root = tmp_path / "receipts"
    write_delivery_receipt(
        snapshot,
        delivery_ok=True,
        attempted_at="2026-07-24T00:09:30+00:00",
        sent_at="2026-07-24T00:10:00+00:00",
        root=receipt_root,
    )

    result = label_recommend_snapshot(
        snapshot_file,
        output_root=tmp_path / "labels",
        db_path=tmp_path / "15m.db",
        receipt_root=receipt_root,
        now="2026-07-25 09:14:59",
        assessor=lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("09:15+24h 전에는 경로를 평가하면 안 된다")
        ),
    )

    assert result["artifact_status"] == "not_mature"
    assert result["path_window_start"].endswith("09:15:00+09:00")
    assert result["path_window_end"].endswith("09:15:00+09:00")
    assert result["written"] is False


def test_late_same_day_snapshot_keeps_actual_time_but_is_not_forward_eligible(
    tmp_path,
):
    window_start, window_end = path_window("2026-07-24")
    snapshot = {
        "asof": "2026-07-24",
        "slot": "open",
        "created_at": "2026-07-24T09:00:00+00:00",  # 18:00 KST
    }

    execution = _execution_metadata(
        snapshot,
        receipt_root=tmp_path / "receipts",
        window_start=window_start,
        window_end=window_end,
    )

    assert execution["execution_at"].endswith("18:00:00+09:00")
    assert execution["execution_start_at"].endswith("18:15:00+09:00")
    assert execution["execution_time_basis"].endswith("_off_schedule")
    assert execution["provenance_cohort"] == OFF_SCHEDULE_PROVENANCE_COHORT
    assert execution["forward_eligible"] is False


def test_upstream_probe_recognises_only_structural_absence(monkeypatch):
    """404 Code-not-found(상폐)·빈 배열·전부 창밖 봉만 True — 나머지는 전부
    fail-closed False."""
    from signals import recommend_score_labels as module

    class FakeResponse:
        def __init__(self, status, payload=None, text=""):
            self.status_code = status
            self._payload = payload
            self.text = text

        def json(self):
            if isinstance(self._payload, Exception):
                raise self._payload
            return self._payload

    start, _ = module.path_window("2026-08-04")
    in_window = [{"candle_date_time_kst": "2026-08-04T12:00:00"}]
    outside = [{"candle_date_time_kst": "2026-08-01T12:00:00"}]
    cases = [
        (FakeResponse(404, text='{"error":{"name":404,"message":"Code not found"}}'), True),
        (FakeResponse(200, payload=[]), True),
        (FakeResponse(200, payload=outside), True),
        (FakeResponse(200, payload=in_window), False),
        (FakeResponse(200, payload={"weird": 1}), False),
        (FakeResponse(200, payload=ValueError("bad json")), False),
        (FakeResponse(404, text="other 404"), False),
    ]
    import requests as requests_module

    for response, expected in cases:
        monkeypatch.setattr(
            requests_module, "get", lambda *a, _r=response, **k: _r
        )
        assert (
            module._upstream_confirms_no_observations("KRW-TEST", start)
            is expected
        ), (response.status_code, response.text)

    def boom(*_a, **_k):
        raise requests_module.ConnectionError("down")

    monkeypatch.setattr(requests_module, "get", boom)
    monkeypatch.setattr(module.time, "sleep", lambda *_: None)
    assert module._upstream_confirms_no_observations("KRW-TEST", start) is False


def test_upstream_confirmed_halt_is_structural_and_completes_artifact(tmp_path):
    """거래정지 종목(창 전체 무봉 + 업스트림 확인)은 halted 구조적 종결 —
    artifact 를 partial 로 잡아 close·publish 를 무기한 차단하지 않는다
    (2026-08-05 AERGO·AQT 실사고)."""
    snapshot_file = _make_snapshot(tmp_path / "snapshots", slot="open")
    bars = [(100.0, 101.0, 99.0, 100.0) for _ in range(96)]
    probed: list[str] = []

    def assessor(market, start_at, *, db_path):
        if market == "KRW-T2":
            return _assessment(
                [], quality="target_no_observations", raw_bars=0,
                complete=False, start=start_at,
            )
        return _assessment(bars, start=start_at)

    result = label_recommend_snapshot(
        snapshot_file,
        output_root=tmp_path / "labels",
        db_path=tmp_path / "15m.db",
        receipt_root=tmp_path / "receipts",
        now="2026-07-25 10:00:00",
        assessor=assessor,
        halt_prober=lambda market: probed.append(market) or True,
    )

    assert result["artifact_status"] == "complete"
    assert result["summary"]["halted"] == 1
    assert result["summary"]["incomplete"] == 0
    assert probed == ["KRW-T2"]
    halted_rows = [
        r for r in result["rows"]
        if r["label_status"] == "halted_no_observations"
    ]
    assert [r["coin"] for r in halted_rows] == ["KRW-T2"]
    assert halted_rows[0]["path_reason"] == "upstream_confirmed_no_observations"
    assert halted_rows[0]["mfe"] is None
    # 완결 artifact 는 검증기를 통과하고 재실행에서 재사용된다.
    load_label_artifact(Path(result["artifact_path"]))
    again = label_recommend_snapshot(
        snapshot_file,
        output_root=tmp_path / "labels",
        db_path=tmp_path / "15m.db",
        receipt_root=tmp_path / "receipts",
        now="2026-07-25 11:00:00",
        assessor=assessor,
        halt_prober=lambda market: (_ for _ in ()).throw(AssertionError),
    )
    assert again["artifact_reused"] is True


def test_unconfirmed_no_observations_stays_partial_for_retry(tmp_path):
    """업스트림에 봉이 있거나(수집 갭) 확인이 실패하면 halted 로 종결하지
    않고 partial 유지 — fail-closed, 다음 실행에서 재시도."""
    snapshot_file = _make_snapshot(tmp_path / "snapshots", slot="open")
    bars = [(100.0, 101.0, 99.0, 100.0) for _ in range(96)]

    def assessor(market, start_at, *, db_path):
        if market == "KRW-T2":
            return _assessment(
                [], quality="target_no_observations", raw_bars=0,
                complete=False, start=start_at,
            )
        return _assessment(bars, start=start_at)

    for prober in (
        lambda market: False,
        lambda market: (_ for _ in ()).throw(RuntimeError("probe down")),
    ):
        result = label_recommend_snapshot(
            snapshot_file,
            output_root=tmp_path / "labels",
            db_path=tmp_path / "15m.db",
            receipt_root=tmp_path / "receipts",
            now="2026-07-25 10:00:00",
            assessor=assessor,
            halt_prober=prober,
        )
        assert result["artifact_status"] == "partial"
        assert result["summary"]["incomplete"] == 1
        assert "halted" not in result["summary"]


def test_incomplete_is_partial_then_retried_to_complete(tmp_path):
    snapshot_file = _make_snapshot(tmp_path / "snapshots", slot="open")
    output_root = tmp_path / "labels"

    partial = label_recommend_snapshot(
        snapshot_file,
        output_root=output_root,
        db_path=tmp_path / "15m.db",
        receipt_root=tmp_path / "receipts",
        now="2026-07-25 09:16:00",
        assessor=lambda _market, start_at, **_kwargs: _assessment(
            [], quality="benchmark_gap", raw_bars=40, complete=False,
            start=start_at,
        ),
    )
    assert partial["artifact_status"] == "partial"
    assert partial["summary"]["incomplete"] == 3
    assert all(r["mfe"] is None for r in partial["rows"])
    assert all(r["path_reason"] == "benchmark_gap" for r in partial["rows"])

    partial_again = label_recommend_snapshot(
        snapshot_file,
        output_root=output_root,
        db_path=tmp_path / "15m.db",
        receipt_root=tmp_path / "receipts",
        now="2026-07-25 09:30:00",
        assessor=lambda _market, start_at, **_kwargs: _assessment(
            [], quality="benchmark_gap", raw_bars=40, complete=False,
            start=start_at,
        ),
    )
    assert partial_again["artifact_reused"] is True
    assert partial_again["label_payload_sha256"] == partial["label_payload_sha256"]

    bars = [(100.0, 101.0, 99.0, 100.0) for _ in range(96)]
    complete = label_recommend_snapshot(
        snapshot_file,
        output_root=output_root,
        db_path=tmp_path / "15m.db",
        receipt_root=tmp_path / "receipts",
        now="2026-07-25 10:00:00",
        assessor=lambda _market, start_at, **_kwargs: _assessment(
            bars, quality="flat_filled", raw_bars=90, start=start_at
        ),
    )
    assert complete["artifact_status"] == "complete"
    assert complete["summary"]["flat_filled"] == 3
    assert complete["label_payload_sha256"] != partial["label_payload_sha256"]


@pytest.mark.parametrize(
    "bars",
    [
        [(100.0, 101.0, 99.0, 100.0) for _ in range(95)],
        [(100.0, 99.0, 98.0, 100.0) for _ in range(96)],
    ],
)
def test_complete_assessor_contract_violation_is_never_labeled(
    tmp_path,
    bars,
):
    snapshot_file = _make_snapshot(tmp_path / "snapshots", slot="open")

    result = label_recommend_snapshot(
        snapshot_file,
        output_root=tmp_path / "labels",
        receipt_root=tmp_path / "receipts",
        db_path=tmp_path / "15m.db",
        now="2026-07-25 10:00:00",
        assessor=lambda _market, start_at, **_kwargs: _assessment(
            bars,
            start=start_at,
        ),
    )

    assert result["artifact_status"] == "partial"
    assert all(
        row["label_status"] == "invalid_complete_path"
        for row in result["rows"]
    )
    assert all(row["mfe"] is None for row in result["rows"])
    assert load_label_artifact(result["artifact_path"])["artifact_status"] == (
        "partial"
    )


def test_label_lock_rejects_symlink_instead_of_locking_external_inode(
    tmp_path,
):
    snapshot_file = _make_snapshot(tmp_path / "snapshots", slot="open")
    output_root = tmp_path / "labels"
    target_dir = output_root / "2026-07-24"
    target_dir.mkdir(parents=True)
    outside = tmp_path / "outside.lock"
    outside.write_text("", encoding="utf-8")
    (target_dir / ".open_r1.json.lock").symlink_to(outside)
    bars = [(100.0, 101.0, 99.0, 100.0) for _ in range(96)]

    with pytest.raises(ScoreLabelError, match="lock is unsafe"):
        label_recommend_snapshot(
            snapshot_file,
            output_root=output_root,
            receipt_root=tmp_path / "receipts",
            db_path=tmp_path / "15m.db",
            now="2026-07-25 10:00:00",
            assessor=lambda _market, start_at, **_kwargs: _assessment(
                bars,
                start=start_at,
            ),
        )

    assert not (target_dir / "open_r1.json").exists()


def test_labeling_rejects_source_change_during_computation(
    tmp_path,
    monkeypatch,
):
    snapshot_file = _make_snapshot(tmp_path / "snapshots", slot="open")
    manifests = iter(
        [
            {"sha256": "before", "files": []},
            {"sha256": "after", "files": []},
        ]
    )
    monkeypatch.setattr(
        labels_module,
        "_label_code_manifest",
        lambda: next(manifests),
    )
    bars = [(100.0, 101.0, 99.0, 100.0) for _ in range(96)]

    with pytest.raises(ScoreLabelError, match="label source changed"):
        label_recommend_snapshot(
            snapshot_file,
            output_root=tmp_path / "labels",
            receipt_root=tmp_path / "receipts",
            db_path=tmp_path / "15m.db",
            now="2026-07-25 10:00:00",
            assessor=lambda _market, start_at, **_kwargs: _assessment(
                bars,
                start=start_at,
            ),
        )


def test_path_manifest_binds_first_post_window_horizon_witness(tmp_path):
    db_path = tmp_path / "15m.db"
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            "CREATE TABLE candles ("
            "market TEXT, timestamp TEXT, open REAL, high REAL, "
            "low REAL, close REAL)"
        )
        connection.executemany(
            "INSERT INTO candles VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("KRW-BTC", "2026-07-24 09:00:00", 100, 101, 99, 100),
                ("KRW-BTC", "2026-07-25 09:15:00", 100, 101, 99, 100),
            ],
        )
        connection.commit()
    finally:
        connection.close()

    first = _path_input_manifest(
        db_path,
        markets=["KRW-TEST"],
        start_at=pd.Timestamp("2026-07-24 09:00:00", tz="Asia/Seoul"),
    )
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            "UPDATE candles SET close=100.5 "
            "WHERE market='KRW-BTC' AND timestamp='2026-07-25 09:15:00'"
        )
        connection.commit()
    finally:
        connection.close()
    second = _path_input_manifest(
        db_path,
        markets=["KRW-TEST"],
        start_at=pd.Timestamp("2026-07-24 09:00:00", tz="Asia/Seoul"),
    )

    assert first["rows"] == 2
    assert second["rows"] == 2
    assert first["sha256"] != second["sha256"]


def test_concurrent_partial_cannot_overwrite_complete_artifact(tmp_path):
    snapshot_file = _make_snapshot(tmp_path / "snapshots", slot="open")
    output_root = tmp_path / "labels"
    partial_started = threading.Event()
    release_partial = threading.Event()
    errors: list[BaseException] = []
    results: dict[str, dict] = {}
    complete_bars = [(100.0, 101.0, 99.0, 100.0) for _ in range(96)]

    def partial_assessor(_market, start_at, **_kwargs):
        partial_started.set()
        if not release_partial.wait(timeout=5):
            raise TimeoutError("partial assessor was not released")
        return _assessment(
            [],
            quality="benchmark_gap",
            raw_bars=40,
            complete=False,
            start=start_at,
        )

    def run_partial():
        try:
            results["partial"] = label_recommend_snapshot(
                snapshot_file,
                output_root=output_root,
                db_path=tmp_path / "15m.db",
                receipt_root=tmp_path / "receipts",
                now="2026-07-25 10:00:00",
                assessor=partial_assessor,
            )
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run_partial)
    thread.start()
    assert partial_started.wait(timeout=2)

    results["complete"] = label_recommend_snapshot(
        snapshot_file,
        output_root=output_root,
        db_path=tmp_path / "15m.db",
        receipt_root=tmp_path / "receipts",
        now="2026-07-25 10:00:00",
        assessor=lambda _market, start_at, **_kwargs: _assessment(
            complete_bars, start=start_at
        ),
    )
    release_partial.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert not errors
    assert results["complete"]["artifact_status"] == "complete"
    assert results["partial"]["artifact_status"] == "complete"
    persisted = json.loads(
        Path(results["complete"]["artifact_path"]).read_text(encoding="utf-8")
    )
    assert persisted["artifact_status"] == "complete"


def test_successful_delivery_uses_next_executable_bar_and_charges_cost_once(tmp_path):
    snapshot_file = _make_snapshot(tmp_path / "snapshots", slot="open")
    snapshot = load_snapshot(snapshot_file)
    receipt_root = tmp_path / "receipts"
    write_delivery_receipt(
        snapshot,
        delivery_ok=True,
        attempted_at="2026-07-24T00:09:30+00:00",
        sent_at="2026-07-24T00:10:00+00:00",
        root=receipt_root,
    )

    def assessor(_market, start_at, **_kwargs):
        # 09:00 봉의 +20%는 09:10 전달 전에 끝났으므로 라벨에서 제외되어야 한다.
        bars = [(110.0, 110.0, 110.0, 110.0) for _ in range(96)]
        return _assessment(bars, start=start_at)

    result = label_recommend_snapshot(
        snapshot_file,
        output_root=tmp_path / "labels",
        db_path=tmp_path / "15m.db",
        receipt_root=receipt_root,
        now="2026-07-25 09:16:00",
        assessor=assessor,
    )

    assert result["execution_time_basis"] == "delivery_sent_at"
    assert result["provenance_cohort"] == FORWARD_PROVENANCE_COHORT
    assert result["forward_eligible"] is True
    assert result["execution_start_at"].endswith("09:15:00+09:00")
    assert result["delivery_ok"] is True
    assert result["path_window_start"].endswith("09:15:00+09:00")
    assert result["path_window_end"].endswith("09:15:00+09:00")
    assert all(row["path_used_bars"] == 96 for row in result["rows"])
    assert all(row["actual_entry_open"] == 110.0 for row in result["rows"])
    assert all(row["up10"] is False for row in result["rows"])
    assert all(row["eod_return_gross"] == 0.0 for row in result["rows"])
    assert all(row["eod_return_net"] == -0.0015 for row in result["rows"])


def test_label_execution_rejects_postactivation_receipt_outer_tamper(
    tmp_path,
    monkeypatch,
):
    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            value = datetime(
                2026,
                7,
                27,
                0,
                5,
                2,
                tzinfo=timezone.utc,
            )
            return value if tz is None else value.astimezone(tz)

    snapshot = {
        "asof": "2026-07-27",
        "slot": "open",
        "snapshot_id": "recommend-forward-label",
        "snapshot_path": "output/recommend_snapshots/2026-07-27/open_r1.json",
        "created_at": "2026-07-27T00:05:00+00:00",
        "model": {"id": "recommend_r1_open", "ranking": "R1"},
        "request": {"limit_markets": None},
    }
    monkeypatch.setattr(receipt_module, "datetime", FixedDatetime)
    receipt_root = tmp_path / "receipts"
    receipt_path = write_delivery_receipt(
        snapshot,
        delivery_ok=True,
        attempted_at="2026-07-27T00:05:00+00:00",
        sent_at="2026-07-27T00:05:01+00:00",
        telegram_result=_telegram_result(
            "open radar",
            "2026-07-27T00:05:01+00:00",
        ),
        message="open radar",
        root=receipt_root,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["sent_at"] = "2026-07-27T00:05:00.500000+00:00"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    window_start, window_end = path_window("2026-07-27")

    with pytest.raises(DeliveryReceiptError, match="outer integrity mismatch"):
        _execution_metadata(
            snapshot,
            receipt_root=receipt_root,
            window_start=window_start,
            window_end=window_end,
        )


def test_exact_boundary_delivery_starts_at_following_candle(tmp_path):
    snapshot_file = _make_snapshot(tmp_path / "snapshots", slot="open")
    snapshot = load_snapshot(snapshot_file)
    receipt_root = tmp_path / "receipts"
    write_delivery_receipt(
        snapshot,
        delivery_ok=True,
        attempted_at="2026-07-24T00:14:59+00:00",
        sent_at="2026-07-24T00:15:00+00:00",
        root=receipt_root,
    )
    window_start, window_end = path_window("2026-07-24")

    metadata = _execution_metadata(
        snapshot,
        receipt_root=receipt_root,
        window_start=window_start,
        window_end=window_end,
    )

    assert metadata["execution_start_at"].endswith("09:30:00+09:00")


def test_cli_single_snapshot_reports_complete(tmp_path, monkeypatch, capsys):
    snapshot = tmp_path / "open_r1.json"
    snapshot.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        label_cli,
        "label_recommend_snapshot",
        lambda *a, **k: {
            "artifact_status": "complete",
            "artifact_path": "/labels/open_r1.json",
            "written": True,
            "artifact_reused": False,
            "summary": {"rows": 100},
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["label_recommend_snapshots.py", "--snapshot", str(snapshot)],
    )

    assert label_cli.main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output[0]["status"] == "complete"
    assert output[0]["summary"]["rows"] == 100


def test_through_date_includes_old_partial_candidates_only(tmp_path):
    root = tmp_path / "snapshots"
    expected = []
    for day in ("2026-07-22", "2026-07-24", "2026-07-25", "not-a-date"):
        folder = root / day
        folder.mkdir(parents=True)
        path = folder / "open_r1.json"
        path.write_text("{}", encoding="utf-8")
        if day in {"2026-07-22", "2026-07-24"}:
            expected.append(path)
    limited = root / "2026-07-24" / "open_r1.limit10.json"
    limited.write_text("{}", encoding="utf-8")

    result = label_cli._resolve_inputs(
        snapshot=None,
        date=None,
        through_date="2026-07-24",
        snapshot_root=root,
    )

    assert result == expected


def test_batch_labeling_excludes_market_limited_snapshots(tmp_path):
    root = tmp_path / "snapshots"
    folder = root / "2026-07-24"
    folder.mkdir(parents=True)
    production = folder / "open_r1.json"
    limited = folder / "open_r1.limit10.json"
    production.write_text("{}", encoding="utf-8")
    limited.write_text("{}", encoding="utf-8")

    result = label_cli._resolve_inputs(
        snapshot=None,
        date="2026-07-24",
        through_date=None,
        snapshot_root=root,
    )

    assert result == [production]


def test_missing_batch_snapshot_is_operational_failure(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "label_recommend_snapshots.py",
            "--date",
            "2026-07-24",
            "--snapshot-root",
            str(tmp_path),
        ],
    )

    assert label_cli.main() == 3
    assert "matching snapshot JSON 없음" in capsys.readouterr().err


def test_invalid_cli_date_is_rejected():
    with pytest.raises(argparse.ArgumentTypeError):
        label_cli._iso_date("2026-02-30")


def test_nesting_violation_row_is_labeled_complete_with_diagnostic(
    tmp_path,
    caplog,
    monkeypatch,
):
    # 독립 head 의 포함관계 위반(유니버스 꼬리)은 진단 경고일 뿐, 전 유니버스
    # forward 라벨 축적을 죽이지 않는다 (2026-07-27 장애 회귀 방지).
    def scorer(asof, *, limit_markets, slot, ranking):
        universe = [
            _candidate(i, entry_open=100.0 + i) for i in range(1, 5)
        ]
        degraded = universe[3]
        degraded["p_up20"] = 0.2
        degraded["pump_prob"] = 0.2
        degraded["pump_prob_pct"] = "20.0%"
        return {
            "asof": asof,
            "slot": slot,
            "feature_date": asof,
            "btc_regime": "neutral",
            "universe_n": 4,
            "calibration_source": "bucket_score_pump20",
            "rank_basis": "R1_riskreward(de-corr head)",
            "n_history_dates": 100,
            "ranking": ranking,
            "score_schema_version": "recommend_score.v1",
            "rule_version": "r1_riskreward_v1",
            "model_random_seed": 42,
            "feature_columns": ["f_ret_3d"],
            "training": {
                "start": "2025-01-01",
                "end": "2026-07-18",
                "cutoff_exclusive": "2026-07-19",
                "embargo_days": 5,
                "rows": 1000,
                "dates": 100,
            },
            "universe": universe,
            "top3": universe[:3],
        }

    result = get_or_create_recommend_snapshot(
        "2026-07-24",
        slot="open",
        root=tmp_path / "snapshots",
        scorer=scorer,
    )
    bars = [(100.0, 101.0, 99.0, 100.0) for _ in range(96)]

    import signals.recommend_score_labels as label_module

    monkeypatch.setattr(label_module.log, "propagate", True)
    with caplog.at_level(
        "WARNING",
        logger="signals.recommend_score_labels",
    ):
        labeled = label_recommend_snapshot(
            Path(result["snapshot_path"]),
            output_root=tmp_path / "labels",
            receipt_root=tmp_path / "receipts",
            db_path=tmp_path / "15m.db",
            now="2026-07-25 09:16:00",
            assessor=lambda _market, start_at, **_kwargs: _assessment(
                bars,
                start=start_at,
            ),
        )

    assert labeled["artifact_status"] == "complete"
    assert labeled["summary"]["labeled"] == 4
    assert any(
        "upside nesting violated" in record.getMessage()
        for record in caplog.records
    )

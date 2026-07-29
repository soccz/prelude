from __future__ import annotations

import ast
import hashlib
import json
import os
import threading
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from data import collector_upbit_microstructure as collector
from data import upbit_microstructure
from data.upbit_microstructure import (
    RawFrame,
    RawStreamWriter,
    iter_raw_records,
    parse_raw_frame,
)


def trade_payload(
    *,
    event_at_ms: int,
    sequential_id: int,
) -> dict:
    return {
        "type": "trade",
        "code": "KRW-BTC",
        "timestamp": event_at_ms + 1,
        "trade_timestamp": event_at_ms,
        "trade_price": 100_000_000.0,
        "trade_volume": 0.01,
        "ask_bid": "BID",
        "sequential_id": sequential_id,
        "stream_type": "REALTIME",
    }


def orderbook_payload(
    *,
    event_at_ms: int,
    stream_type: str,
) -> dict:
    return {
        "type": "orderbook",
        "code": "KRW-BTC",
        "timestamp": event_at_ms,
        "total_ask_size": 1.0,
        "total_bid_size": 2.0,
        "orderbook_units": [
            {
                "ask_price": 100_001_000.0,
                "ask_size": 1.0,
                "bid_price": 100_000_000.0,
                "bid_size": 2.0,
            }
        ],
        "level": 0,
        "stream_type": stream_type,
    }


def raw_frame(payload: dict, *, ingress_seq: int = 1) -> RawFrame:
    return RawFrame(
        raw=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        received_at_ns=1_800_000_000_000_000_000 + ingress_seq,
        received_monotonic_ns=123_000 + ingress_seq,
        connection_id="connection-1",
        subscription_id="subscription-1",
        ingress_seq=ingress_seq,
    )


class FakeWebSocket:
    def __init__(
        self,
        *,
        messages_per_channel: int = 1,
        first_message_delay: float = 0,
    ):
        self.messages_per_channel = messages_per_channel
        self.first_message_delay = first_message_delay
        self._delayed = False
        self.messages: list[bytes] = []
        self.sent: list[list[dict]] = []

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        return False

    def send(self, raw_request: str):
        request = json.loads(raw_request)
        self.sent.append(request)
        channel = request[1]["type"]
        now_ms = time.time_ns() // 1_000_000
        if channel == "trade":
            for index in range(self.messages_per_channel):
                payload = trade_payload(
                    event_at_ms=now_ms,
                    sequential_id=now_ms * 10_000 + index,
                )
                self.messages.append(
                    json.dumps(payload, separators=(",", ":")).encode("utf-8")
                )
        else:
            for index in range(self.messages_per_channel):
                payload = orderbook_payload(
                    event_at_ms=now_ms,
                    stream_type="SNAPSHOT" if index == 0 else "REALTIME",
                )
                self.messages.append(
                    json.dumps(payload, separators=(",", ":")).encode("utf-8")
                )

    def recv(self, timeout: float):
        if self.messages:
            if self.first_message_delay and not self._delayed:
                self._delayed = True
                time.sleep(self.first_message_delay)
            return self.messages.pop(0)
        time.sleep(min(timeout, 0.005))
        raise TimeoutError


class FakeConnector:
    def __init__(
        self,
        *,
        messages_per_channel: int = 1,
        first_message_delay: float = 0,
    ):
        self.messages_per_channel = messages_per_channel
        self.first_message_delay = first_message_delay
        self.connections: list[FakeWebSocket] = []
        self.lock = threading.Lock()

    def __call__(self, _endpoint: str, **_kwargs):
        connection = FakeWebSocket(
            messages_per_channel=self.messages_per_channel,
            first_message_delay=self.first_message_delay,
        )
        with self.lock:
            self.connections.append(connection)
        return connection


class BlockingWebSocket:
    """recv timeout을 무시하는 비정상 transport를 close로 해제한다."""

    def __init__(self):
        self.subscribed = threading.Event()
        self.released = threading.Event()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        return False

    def send(self, _raw_request: str):
        self.subscribed.set()

    def recv(self, timeout: float):
        del timeout
        self.released.wait(5)
        raise OSError("socket closed")

    def close(self):
        self.released.set()


def _config(tmp_path: Path, *, queue_max: int = 1000):
    return collector.CaptureConfig(
        markets=("KRW-BTC",),
        output_root=tmp_path / "raw",
        duration_seconds=0.15,
        required_warmup_seconds=0,
        max_wait_seconds=2,
        queue_max=queue_max,
        asof=date(2026, 7, 26),
    )


def _valid_snapshot_document() -> dict:
    document = {
        "asof": "2026-07-26",
        "slot": "open",
        "snapshot_schema": "recommend_snapshot.v1",
        "created_at": "2026-07-26T00:09:40+00:00",
        "decision_started_at": "2026-07-26T00:05:10+00:00",
        "decision_completed_at": "2026-07-26T00:09:40+00:00",
    }
    digest_payload = {
        key: value
        for key, value in document.items()
        if key
        not in {"created_at", "snapshot_id", "payload_sha256", "snapshot_path"}
    }
    digest = hashlib.sha256(
        json.dumps(
            digest_payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    document["payload_sha256"] = digest
    document["snapshot_id"] = f"recommend-{digest[:20]}"
    return document


def test_capture_config_rejects_timezone_naive_explicit_cutoff(tmp_path):
    with pytest.raises(ValueError, match="must include timezone"):
        collector.CaptureConfig(
            markets=("KRW-BTC",),
            output_root=tmp_path / "raw",
            feature_cutoff_at=datetime(2026, 7, 26, 9, 5),
        )


def test_snapshot_cutoff_uses_actual_decision_completion_time(tmp_path):
    snapshot_path = tmp_path / "open_r1.json"
    document = _valid_snapshot_document()
    snapshot_path.write_text(json.dumps(document), encoding="utf-8")

    cutoff, metadata = collector.read_snapshot_cutoff(
        snapshot_path,
        expected_asof=date(2026, 7, 26),
        expected_slot="open",
    )
    assert cutoff == datetime(2026, 7, 26, 0, 9, 40, tzinfo=timezone.utc)
    assert metadata["snapshot_id"] == document["snapshot_id"]
    assert metadata["decision_completed_at"] == document["decision_completed_at"]
    assert metadata["file_sha256"] == hashlib.sha256(
        snapshot_path.read_bytes()
    ).hexdigest()


@pytest.mark.parametrize(
    ("raw_document", "message"),
    [
        (
            '{"asof":"2026-07-26","asof":"2026-07-26","slot":"open"}',
            "duplicate JSON object key",
        ),
        (
            '{"asof":"2026-07-26","slot":"open","value":NaN}',
            "non-standard JSON numeric constant",
        ),
    ],
)
def test_snapshot_cutoff_rejects_noncanonical_json(
    tmp_path,
    raw_document,
    message,
):
    snapshot_path = tmp_path / "open_r1.json"
    snapshot_path.write_text(raw_document, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        collector.read_snapshot_cutoff(
            snapshot_path,
            expected_asof=date(2026, 7, 26),
            expected_slot="open",
        )


def test_snapshot_cutoff_rejects_symlink(tmp_path):
    target_path = tmp_path / "real_snapshot.json"
    target_path.write_text(
        json.dumps(_valid_snapshot_document()),
        encoding="utf-8",
    )
    snapshot_path = tmp_path / "open_r1.json"
    snapshot_path.symlink_to(target_path)

    with pytest.raises(ValueError, match="invalid snapshot JSON"):
        collector.read_snapshot_cutoff(
            snapshot_path,
            expected_asof=date(2026, 7, 26),
            expected_slot="open",
        )


def test_snapshot_cutoff_rejects_replacement_after_strict_read(
    tmp_path,
    monkeypatch,
):
    snapshot_path = tmp_path / "open_r1.json"
    document = _valid_snapshot_document()
    snapshot_path.write_text(json.dumps(document), encoding="utf-8")
    replacement_path = tmp_path / "replacement.json"
    replacement_path.write_text(
        json.dumps(document, indent=2),
        encoding="utf-8",
    )
    strict_json_object = collector.strict_json_object

    def replace_after_read(path):
        parsed = strict_json_object(path)
        os.replace(replacement_path, snapshot_path)
        return parsed

    monkeypatch.setattr(
        collector,
        "strict_json_object",
        replace_after_read,
    )

    with pytest.raises(ValueError, match="snapshot changed while reading"):
        collector.read_snapshot_cutoff(
            snapshot_path,
            expected_asof=date(2026, 7, 26),
            expected_slot="open",
        )


def test_snapshot_cutoff_rejects_tampered_payload_and_identity(tmp_path):
    snapshot_path = tmp_path / "open_r1.json"
    document = _valid_snapshot_document()
    document["decision_started_at"] = "2026-07-26T00:05:11+00:00"
    snapshot_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="payload_sha256 mismatch"):
        collector.read_snapshot_cutoff(
            snapshot_path,
            expected_asof=date(2026, 7, 26),
            expected_slot="open",
        )

    document = _valid_snapshot_document()
    document["snapshot_id"] = "recommend-" + "0" * 20
    snapshot_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="snapshot_id"):
        collector.read_snapshot_cutoff(
            snapshot_path,
            expected_asof=date(2026, 7, 26),
            expected_slot="open",
        )


def test_snapshot_cutoff_rejects_degraded_provenance(tmp_path):
    snapshot_path = tmp_path / "open_r1.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "asof": "2026-07-26",
                "slot": "open",
                "created_at": "2026-07-26T00:09:40+00:00",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="snapshot_schema"):
        collector.read_snapshot_cutoff(
            snapshot_path,
            expected_asof=date(2026, 7, 26),
            expected_slot="open",
        )


def test_snapshot_cutoff_rejects_wrong_identity(tmp_path):
    snapshot_path = tmp_path / "open_r1.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "asof": "2026-07-25",
                "slot": "open",
                "created_at": "2026-07-26T00:09:40+00:00",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="asof mismatch"):
        collector.read_snapshot_cutoff(
            snapshot_path,
            expected_asof=date(2026, 7, 26),
            expected_slot="open",
        )


def test_end_to_end_fake_capture_writes_two_raw_streams_and_cutoff_manifest(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        collector,
        "clock_sync_status",
        lambda: {"ntp_synchronized": True},
    )
    connector = FakeConnector(messages_per_channel=3)
    result = collector.run_capture(
        _config(tmp_path),
        connect_fn=connector,
        install_signal_handlers=False,
    )

    assert result.complete
    assert result.manifest["uses_api_key"] is False
    assert result.manifest["places_orders"] is False
    assert result.manifest["stop_reason"] == "feature_cutoff_reached"
    assert result.manifest["quality"]["transport_complete"]
    assert result.manifest["quality"]["causal_window_complete"]
    assert result.manifest["feature_cutoff_at_ns"] is not None
    assert result.manifest["universe"]["markets"] == ["KRW-BTC"]
    assert result.manifest["universe"]["source"] == "explicit"
    assert result.manifest["universe"]["fetch_started_at_ns"] is None
    assert result.manifest["universe"]["observed_at_ns"] <= result.manifest[
        "started_at_ns"
    ]
    source_code = result.manifest["source_code"]
    assert source_code["provenance_complete"]
    assert source_code["unchanged_during_capture"]
    assert source_code["at_start"]["git"]["commit"]
    assert isinstance(source_code["at_start"]["git"]["dirty"], bool)
    assert source_code["at_start"]["runtime"]["python_version"]
    assert source_code["at_start"]["runtime"]["websockets_version"]
    assert result.manifest["streams"]["trade"]["dropped_frames"] == 0
    assert result.manifest["streams"]["orderbook"]["dropped_frames"] == 0

    for channel in ("trade", "orderbook"):
        artifact = result.manifest["streams"][channel]["artifact"]
        assert artifact["event_count"] == 3
        assert Path(artifact["path"]).exists()
        assert len(list(iter_raw_records(artifact["path"]))) == 3
        cutoff_stats = result.manifest["streams"][channel]["at_feature_cutoff"]
        assert cutoff_stats["included_event_count"] == 3
        assert cutoff_stats["max_ingress_seq"] == 3
        if channel == "orderbook":
            assert artifact["requested_orderbook_depth"] == 30
            assert artifact["observed_orderbook_unit_count_min"] == 1
            assert artifact["observed_orderbook_unit_count_max"] == 1
            assert artifact["thin_book_event_count"] == 3
            assert cutoff_stats["observed_orderbook_unit_count_min"] == 1
            records = list(iter_raw_records(artifact["path"]))
            assert {record["orderbook_unit_count"] for record in records} == {1}

    depth = result.manifest["subscription"]["orderbook_depth"]
    assert depth == {
        "requested": 30,
        "observed_unit_count_min": 1,
        "observed_unit_count_max": 1,
        "thin_book_event_count": 3,
        "thin_book_is_schema_error": False,
    }

    trade_requests = [
        connection.sent[0]
        for connection in connector.connections
        if connection.sent and connection.sent[0][1]["type"] == "trade"
    ]
    book_requests = [
        connection.sent[0]
        for connection in connector.connections
        if connection.sent and connection.sent[0][1]["type"] == "orderbook"
    ]
    assert len(trade_requests) == 1
    assert len(book_requests) == 1
    assert trade_requests[0][1]["is_only_realtime"] is True
    assert book_requests[0][1]["codes"] == ["KRW-BTC.30"]


def test_raw_artifact_change_after_watermark_fails_closed(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        collector,
        "clock_sync_status",
        lambda: {"ntp_synchronized": True},
    )
    actual_watermark = collector._cutoff_watermark
    changed = False

    def mutate_after_read(path, cutoff_at_ns):
        nonlocal changed
        result = actual_watermark(path, cutoff_at_ns)
        if not changed:
            changed = True
            Path(path).write_bytes(b"changed after cutoff scan")
        return result

    monkeypatch.setattr(collector, "_cutoff_watermark", mutate_after_read)

    result = collector.run_capture(
        _config(tmp_path),
        connect_fn=FakeConnector(messages_per_channel=2),
        install_signal_handlers=False,
    )

    assert not result.complete
    assert result.manifest["streams"]["trade"]["at_feature_cutoff"] is None
    assert "changed before cutoff manifest" in result.manifest["streams"][
        "trade"
    ]["watermark_error"]


def test_synthetic_high_rate_capture_drains_without_loss(monkeypatch, tmp_path):
    monkeypatch.setattr(
        collector,
        "clock_sync_status",
        lambda: {"ntp_synchronized": True},
    )
    connector = FakeConnector(messages_per_channel=500)
    config = collector.CaptureConfig(
        markets=("KRW-BTC",),
        output_root=tmp_path / "raw",
        duration_seconds=0.4,
        required_warmup_seconds=0,
        max_wait_seconds=2,
        queue_max=2_000,
        asof=date(2026, 7, 26),
    )
    result = collector.run_capture(
        config,
        connect_fn=connector,
        install_signal_handlers=False,
    )

    assert result.complete
    for channel in ("trade", "orderbook"):
        stream = result.manifest["streams"][channel]
        assert stream["dropped_frames"] == 0
        assert stream["artifact"]["event_count"] == 500
        assert stream["at_feature_cutoff"]["included_event_count"] == 500


def test_worker_force_closes_stuck_socket_before_writer_finalization(tmp_path):
    websocket = BlockingWebSocket()
    stop_event = threading.Event()
    worker = collector.ChannelWorker(
        capture_id="capture-1",
        channel="trade",
        config=_config(tmp_path),
        output_path=tmp_path / "trade.jsonl.gz",
        stop_event=stop_event,
        connect_fn=lambda _endpoint, **_kwargs: websocket,
    )
    worker.start()
    assert websocket.subscribed.wait(1)

    stop_event.set()
    worker.join(timeout=0.5)

    assert websocket.released.is_set()
    assert worker.state.receiver_error == "receiver required forced websocket close"
    assert worker.state.artifact
    assert worker.state.artifact["exists"]
    assert not worker._receiver.is_alive()
    assert not worker._writer.is_alive()


def test_queue_overflow_is_explicit_and_capture_fails_closed(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        collector,
        "clock_sync_status",
        lambda: {"ntp_synchronized": True},
    )
    original_write = collector.RawStreamWriter.write

    def slow_write(self, frame):
        time.sleep(0.01)
        return original_write(self, frame)

    monkeypatch.setattr(collector.RawStreamWriter, "write", slow_write)
    connector = FakeConnector(messages_per_channel=100)
    result = collector.run_capture(
        _config(tmp_path, queue_max=1),
        connect_fn=connector,
        install_signal_handlers=False,
    )

    assert not result.complete
    streams = result.manifest["streams"]
    assert any(streams[channel]["dropped_frames"] > 0 for channel in streams)
    assert any(
        any(gap["kind"] == "queue_overflow" for gap in streams[channel]["gaps"])
        for channel in streams
    )
    assert not result.manifest["quality"]["transport_complete"]


def test_post_cutoff_only_frames_cannot_make_capture_complete(
    monkeypatch,
    tmp_path,
):
    """recv polling lag 뒤 첫 frame이 와도 cutoff 이전 evidence 0이면 FAIL.

    타이밍 계약: 수신 스레드가 첫 recv에 진입한 뒤에 cutoff(duration)가
    지나야 지연 frame이 수신될 수 있다. duration이 스레드 기동 지터보다
    짧으면 recv 진입 전에 stop이 걸려 event_count=0으로 flake — duration을
    지터 대비 넉넉히, first_message_delay는 duration보다 크게 유지할 것.
    """
    monkeypatch.setattr(
        collector,
        "clock_sync_status",
        lambda: {"ntp_synchronized": True},
    )
    connector = FakeConnector(
        messages_per_channel=1,
        first_message_delay=1.2,
    )
    config = collector.CaptureConfig(
        markets=("KRW-BTC",),
        output_root=tmp_path / "raw",
        duration_seconds=0.5,
        required_warmup_seconds=0,
        max_wait_seconds=3,
        queue_max=10,
        asof=date(2026, 7, 26),
    )
    result = collector.run_capture(
        config,
        connect_fn=connector,
        install_signal_handlers=False,
    )

    assert not result.complete
    assert not result.manifest["quality"]["transport_complete"]
    for channel in ("trade", "orderbook"):
        stream = result.manifest["streams"][channel]
        assert stream["artifact"]["event_count"] == 1
        assert stream["at_feature_cutoff"]["included_event_count"] == 0
        assert stream["at_feature_cutoff"]["received_after_cutoff_count"] == 1


def test_reconnect_or_gap_cannot_be_marked_transport_complete():
    state = collector.WorkerState(channel="orderbook", reconnect_count=1)
    state.artifact = {
        "event_count": 10,
        "parse_error_count": 0,
        "server_error_count": 0,
        "snapshot_markets": ["KRW-BTC"],
    }
    assert not collector._worker_transport_ok(
        state,
        expected_markets={"KRW-BTC"},
        cutoff_summary={
            "included_event_count": 10,
            "realtime_event_count": 9,
            "snapshot_markets": ["KRW-BTC"],
        },
    )


def test_connection_error_cannot_be_hidden_by_concurrent_stop():
    state = collector.WorkerState(channel="trade")
    state.connections = [
        collector.ConnectionRecord(
            connection_id="connection-1",
            subscription_id="subscription-1",
            attempt=1,
            opened_at_ns=1,
            subscribed_at_ns=2,
            closed_at_ns=3,
            error="ConnectionClosedError",
        )
    ]
    state.artifact = {
        "event_count": 1,
        "parse_error_count": 0,
        "server_error_count": 0,
    }
    assert not collector._worker_transport_ok(
        state,
        expected_markets={"KRW-BTC"},
        cutoff_summary={
            "included_event_count": 1,
            "realtime_event_count": 1,
            "snapshot_markets": [],
        },
    )


def test_unsynchronized_wall_clock_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(
        collector,
        "clock_sync_status",
        lambda: {"ntp_synchronized": False},
    )
    result = collector.run_capture(
        _config(tmp_path),
        connect_fn=FakeConnector(messages_per_channel=2),
        install_signal_handlers=False,
    )
    assert not result.complete
    assert not result.manifest["quality"]["clock_sync_ok"]
    assert not result.manifest["quality"]["transport_complete"]
    assert not result.manifest["quality"]["causal_window_complete"]


def test_warmup_requires_both_subscriptions_and_all_book_snapshots_by_deadline():
    stream_documents = {
        "trade": {
            "connections": [{"subscribed_at_ns": 99}],
            "at_feature_cutoff": {
                "snapshot_watermarks": {},
            },
        },
        "orderbook": {
            "connections": [{"subscribed_at_ns": 99}],
            "at_feature_cutoff": {
                "snapshot_watermarks": {
                    "KRW-BTC": {
                        "event_at_ms": 0,
                        "received_at_ns": 101,
                        "ingress_seq": 1,
                    }
                },
            },
        },
    }
    assert not collector._warmup_ready(
        stream_documents=stream_documents,
        expected_markets={"KRW-BTC"},
        readiness_deadline_ns=100,
    )

    stream_documents["orderbook"]["at_feature_cutoff"]["snapshot_watermarks"][
        "KRW-BTC"
    ] = {
        "event_at_ms": 0,
        "received_at_ns": 99,
        "ingress_seq": 1,
    }
    assert collector._warmup_ready(
        stream_documents=stream_documents,
        expected_markets={"KRW-BTC"},
        readiness_deadline_ns=100,
    )


def test_static_boundary_has_no_private_trading_or_cross_responsibility_imports():
    path = Path(collector.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    assert not any(name.startswith("notifier") for name in imported)
    assert not any(name.startswith("ledger") for name in imported)
    assert not any(name.startswith("signals") for name in imported)
    assert "ccxt" not in imported
    assert "jwt" not in imported
    assert collector.PUBLIC_WEBSOCKET_ENDPOINT.endswith("/websocket/v1")
    assert not collector.PUBLIC_WEBSOCKET_ENDPOINT.endswith("/private")


def test_capture_config_normalises_deduplicates_and_validates_krw_markets(
    tmp_path,
):
    config = collector.CaptureConfig(
        markets=(" krw-eth ", "KRW-BTC", "krw-eth"),
        output_root=tmp_path / "raw",
        duration_seconds=1,
    )

    assert config.markets == ("KRW-BTC", "KRW-ETH")
    assert config.universe_source == "explicit"
    assert config.universe_fetch_started_at_ns is None
    assert config.universe_observed_at_ns is not None

    for invalid in (("BTC-USDT",), ("KRW-",), ("",), ("KRW-BTC.X",)):
        with pytest.raises(ValueError, match="KRW"):
            collector.CaptureConfig(
                markets=invalid,
                output_root=tmp_path / "invalid",
                duration_seconds=1,
            )


def test_live_universe_resolution_records_actual_fetch_window(monkeypatch):
    from data import collector_d1

    observed_times = iter([100, 250])
    monkeypatch.setattr(collector, "utc_now_ns", lambda: next(observed_times))
    monkeypatch.setattr(
        collector_d1,
        "get_krw_markets",
        lambda: ["krw-eth", "KRW-BTC", "KRW-BTC"],
    )

    markets, source, fetch_started, observed = collector._resolve_markets(None)

    assert markets == ("KRW-BTC", "KRW-ETH")
    assert source == "live"
    assert fetch_started == 100
    assert observed == 250


def test_malformed_trade_and_orderbook_numeric_fields_fail_closed():
    trade_cases: list[tuple[dict, str]] = []
    for field, value, fragment in (
        ("trade_price", float("nan"), "trade_price"),
        ("trade_volume", 0, "trade_volume"),
        ("sequential_id", 1.5, "sequential_id"),
    ):
        payload = trade_payload(event_at_ms=1_700_000_000_000, sequential_id=1)
        payload[field] = value
        trade_cases.append((payload, fragment))
    crossed_trade = trade_payload(
        event_at_ms=1_700_000_000_000,
        sequential_id=1,
    )
    crossed_trade.update(
        {
            "best_ask_price": 100.0,
            "best_bid_price": 101.0,
        }
    )
    trade_cases.append((crossed_trade, "best_bid_price"))

    book_cases: list[tuple[dict, str]] = []
    invalid_total = orderbook_payload(
        event_at_ms=1_700_000_000_000,
        stream_type="SNAPSHOT",
    )
    invalid_total["total_bid_size"] = float("inf")
    book_cases.append((invalid_total, "total_bid_size"))
    invalid_size = orderbook_payload(
        event_at_ms=1_700_000_000_000,
        stream_type="SNAPSHOT",
    )
    invalid_size["orderbook_units"][0]["ask_size"] = 0
    book_cases.append((invalid_size, "ask_size"))
    crossed_book = orderbook_payload(
        event_at_ms=1_700_000_000_000,
        stream_type="SNAPSHOT",
    )
    crossed_book["orderbook_units"][0]["bid_price"] = 100_002_000.0
    book_cases.append((crossed_book, "bid_price"))

    for channel, cases in (("trade", trade_cases), ("orderbook", book_cases)):
        for payload, fragment in cases:
            envelope, metadata = parse_raw_frame(
                raw_frame(payload),
                capture_id="capture-invalid",
                channel=channel,
                expected_markets={"KRW-BTC"},
            )
            assert metadata is None
            assert envelope["record_type"] == "parse_error"
            assert fragment in envelope["error"]


def test_raw_writer_close_can_retry_after_replace_failure(
    tmp_path, monkeypatch
):
    path = tmp_path / "trade.jsonl.gz"
    writer = RawStreamWriter(
        path,
        capture_id="capture-retry",
        channel="trade",
        expected_markets={"KRW-BTC"},
    )
    writer.write(
        raw_frame(
            trade_payload(
                event_at_ms=1_700_000_000_000,
                sequential_id=1,
            )
        )
    )
    actual_replace = upbit_microstructure.os.replace
    failed = False

    def fail_once(source, destination):
        nonlocal failed
        if Path(source) == writer.partial_path and not failed:
            failed = True
            raise OSError("injected replace failure")
        return actual_replace(source, destination)

    monkeypatch.setattr(upbit_microstructure.os, "replace", fail_once)

    with pytest.raises(OSError, match="injected"):
        writer.close()
    assert writer.artifact_metadata()["writer_state"] == "close_failed"
    assert writer.partial_path.exists()

    metadata = writer.close()
    assert metadata["writer_state"] == "finalized"
    assert metadata["exists"]
    assert path.exists()
    assert not writer.partial_path.exists()


def test_raw_writer_close_failure_can_abort_and_preserve_evidence(
    tmp_path, monkeypatch
):
    path = tmp_path / "trade.jsonl.gz"
    writer = RawStreamWriter(
        path,
        capture_id="capture-abort",
        channel="trade",
        expected_markets={"KRW-BTC"},
    )
    writer.write(
        raw_frame(
            trade_payload(
                event_at_ms=1_700_000_000_000,
                sequential_id=1,
            )
        )
    )
    actual_replace = upbit_microstructure.os.replace
    failed = False

    def fail_once(source, destination):
        nonlocal failed
        if Path(source) == writer.partial_path and not failed:
            failed = True
            raise OSError("injected replace failure")
        return actual_replace(source, destination)

    monkeypatch.setattr(upbit_microstructure.os, "replace", fail_once)

    with pytest.raises(OSError, match="injected"):
        writer.close()
    metadata = writer.abort(reason="test abort after close failure")

    assert metadata["writer_state"] == "aborted"
    assert not metadata["exists"]
    assert metadata["aborted_path"]
    assert Path(metadata["aborted_path"]).exists()
    assert not writer.partial_path.exists()
    with pytest.raises(RuntimeError, match="aborted"):
        writer.close()


def test_constructor_failure_restores_handlers_cleans_writer_and_writes_manifest(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        collector,
        "clock_sync_status",
        lambda: {"ntp_synchronized": True},
    )
    actual_worker = collector.ChannelWorker
    created: list[collector.ChannelWorker] = []

    def worker_factory(**kwargs):
        if kwargs["channel"] == "orderbook":
            raise RuntimeError("injected constructor failure")
        worker = actual_worker(**kwargs)
        created.append(worker)
        return worker

    previous = {
        collector.signal.SIGINT: object(),
        collector.signal.SIGTERM: object(),
    }
    signal_calls: list[tuple[int, object]] = []
    monkeypatch.setattr(
        collector.signal,
        "getsignal",
        lambda signum: previous[signum],
    )
    monkeypatch.setattr(
        collector.signal,
        "signal",
        lambda signum, handler: signal_calls.append((signum, handler)),
    )
    monkeypatch.setattr(collector, "ChannelWorker", worker_factory)

    with pytest.raises(RuntimeError, match="incomplete manifest"):
        collector.run_capture(
            _config(tmp_path),
            connect_fn=FakeConnector(),
            install_signal_handlers=True,
        )

    manifest_path = next((tmp_path / "raw").rglob("manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert not manifest["complete"]
    assert not manifest["quality"]["lifecycle_ok"]
    assert manifest["lifecycle"]["signal_handlers_restored"]
    assert manifest["lifecycle"]["workers_constructed"] == ["trade"]
    assert any("constructor failure" in item for item in manifest["lifecycle"]["errors"])
    assert created[0].state.artifact["writer_state"] == "aborted"
    assert not list((tmp_path / "raw").rglob("*.partial"))
    assert signal_calls[-2:] == [
        (collector.signal.SIGINT, previous[collector.signal.SIGINT]),
        (collector.signal.SIGTERM, previous[collector.signal.SIGTERM]),
    ]


def test_worker_constructor_internal_failure_aborts_open_writer(
    tmp_path, monkeypatch
):
    def fail_thread_constructor(*args, **kwargs):
        raise RuntimeError("injected thread constructor failure")

    monkeypatch.setattr(collector.threading, "Thread", fail_thread_constructor)

    with pytest.raises(RuntimeError, match="thread constructor"):
        collector.ChannelWorker(
            capture_id="capture-constructor-fault",
            channel="trade",
            config=_config(tmp_path),
            output_path=tmp_path / "trade.jsonl.gz",
            stop_event=threading.Event(),
            connect_fn=FakeConnector(),
        )

    assert not list(tmp_path.rglob("*.partial"))
    aborted = list(tmp_path.rglob("*.aborted-*"))
    assert len(aborted) == 1


def test_partial_start_failure_stops_all_threads_and_writes_manifest(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        collector,
        "clock_sync_status",
        lambda: {"ntp_synchronized": True},
    )
    created: list[collector.ChannelWorker] = []
    actual_init = collector.ChannelWorker.__init__
    actual_start = collector.ChannelWorker.start

    def tracking_init(self, *args, **kwargs):
        actual_init(self, *args, **kwargs)
        created.append(self)

    def fail_second_start(self):
        if self.channel == "orderbook":
            self._writer.start()
            self._writer_started = True
            raise RuntimeError("injected partial start failure")
        return actual_start(self)

    monkeypatch.setattr(collector.ChannelWorker, "__init__", tracking_init)
    monkeypatch.setattr(collector.ChannelWorker, "start", fail_second_start)

    with pytest.raises(RuntimeError, match="incomplete manifest"):
        collector.run_capture(
            _config(tmp_path),
            connect_fn=FakeConnector(),
            install_signal_handlers=False,
        )

    manifest_path = next((tmp_path / "raw").rglob("manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert not manifest["complete"]
    assert manifest["lifecycle"]["workers_constructed"] == [
        "orderbook",
        "trade",
    ]
    assert manifest["lifecycle"]["workers_started"] == ["trade"]
    assert all(not worker._writer.is_alive() for worker in created)
    assert all(not worker._receiver.is_alive() for worker in created)
    assert not list((tmp_path / "raw").rglob("*.partial"))


def test_join_failure_uses_forced_cleanup_and_writes_manifest(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        collector,
        "clock_sync_status",
        lambda: {"ntp_synchronized": True},
    )
    created: list[collector.ChannelWorker] = []
    actual_init = collector.ChannelWorker.__init__
    actual_join = collector.ChannelWorker.join

    def tracking_init(self, *args, **kwargs):
        actual_init(self, *args, **kwargs)
        created.append(self)

    def fail_after_join(self, timeout=30.0):
        actual_join(self, timeout=timeout)
        if self.channel == "trade":
            raise RuntimeError("injected join boundary failure")

    monkeypatch.setattr(collector.ChannelWorker, "__init__", tracking_init)
    monkeypatch.setattr(collector.ChannelWorker, "join", fail_after_join)

    with pytest.raises(RuntimeError, match="incomplete manifest"):
        collector.run_capture(
            _config(tmp_path),
            connect_fn=FakeConnector(messages_per_channel=2),
            install_signal_handlers=False,
        )

    manifest_path = next((tmp_path / "raw").rglob("manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert any("join boundary failure" in item for item in manifest["lifecycle"]["errors"])
    assert all(not worker._writer.is_alive() for worker in created)
    assert all(not worker._receiver.is_alive() for worker in created)
    assert not list((tmp_path / "raw").rglob("*.partial"))


def test_writer_close_failure_is_aborted_and_manifest_is_incomplete(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        collector,
        "clock_sync_status",
        lambda: {"ntp_synchronized": True},
    )
    actual_close = collector.RawStreamWriter.close

    def fail_orderbook_close(self):
        if self.channel == "orderbook":
            raise OSError("injected close failure")
        return actual_close(self)

    monkeypatch.setattr(collector.RawStreamWriter, "close", fail_orderbook_close)

    result = collector.run_capture(
        _config(tmp_path),
        connect_fn=FakeConnector(messages_per_channel=2),
        install_signal_handlers=False,
    )

    assert not result.complete
    stream = result.manifest["streams"]["orderbook"]
    assert "injected close failure" in stream["writer_error"]
    assert stream["artifact"]["writer_state"] == "aborted"
    assert stream["artifact"]["aborted_path"]
    assert Path(stream["artifact"]["aborted_path"]).exists()
    assert not list((tmp_path / "raw").rglob("*.partial"))


def test_stale_partial_is_quarantined_and_marked_before_new_capture(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        collector,
        "clock_sync_status",
        lambda: {"ntp_synchronized": True},
    )
    old_partial = (
        tmp_path
        / "raw"
        / "2026"
        / "07"
        / "25"
        / "old-capture"
        / "trade.jsonl.gz.partial"
    )
    old_partial.parent.mkdir(parents=True)
    old_partial.write_bytes(b"incomplete raw evidence")
    os.utime(old_partial, (1, 1))

    result = collector.run_capture(
        _config(tmp_path),
        connect_fn=FakeConnector(messages_per_channel=2),
        install_signal_handlers=False,
    )

    assert result.complete
    records = result.manifest["orphan_recovery"]["records"]
    assert len(records) == 1
    assert records[0]["status"] == "quarantined"
    assert not old_partial.exists()
    assert Path(records[0]["preserved_path"]).exists()
    marker = json.loads(
        Path(records[0]["marker_path"]).read_text(encoding="utf-8")
    )
    assert marker["status"] == "incomplete_orphan_quarantined"
    assert marker["original_path"] == str(old_partial)


def test_active_capture_ownership_prevents_startup_quarantine(tmp_path):
    output_root = tmp_path / "raw"
    active_partial = (
        output_root
        / "2026"
        / "07"
        / "25"
        / "active-capture"
        / "trade.jsonl.gz.partial"
    )
    active_partial.parent.mkdir(parents=True)
    active_partial.write_bytes(b"active writer evidence")
    os.utime(active_partial, (1, 1))
    connector = FakeConnector(messages_per_channel=2)

    with collector._capture_output_ownership(output_root):
        with pytest.raises(
            collector.CaptureOwnershipError,
            match="already owns output root",
        ):
            collector.run_capture(
                _config(tmp_path),
                connect_fn=connector,
                install_signal_handlers=False,
            )

    assert active_partial.read_bytes() == b"active writer evidence"
    assert not list(output_root.rglob("*.orphan-*"))
    assert connector.connections == []


def test_capture_ownership_rejects_symlink_lock(tmp_path):
    output_root = tmp_path / "raw"
    output_root.mkdir()
    target_path = tmp_path / "outside-lock-target"
    target_path.write_bytes(b"must remain untouched")
    lock_path = output_root / collector._CAPTURE_LOCK_NAME
    lock_path.symlink_to(target_path)

    with pytest.raises(
        collector.CaptureOwnershipError,
        match="cannot be opened safely",
    ):
        with collector._capture_output_ownership(output_root):
            pytest.fail("symlink lock must never be acquired")

    assert target_path.read_bytes() == b"must remain untouched"


def test_source_change_during_capture_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(
        collector,
        "clock_sync_status",
        lambda: {"ntp_synchronized": True},
    )

    def provenance(digest: str) -> dict:
        return {
            "files": [
                {
                    "path": "data/collector_upbit_microstructure.py",
                    "sha256": digest,
                    "error": None,
                },
                {
                    "path": "data/upbit_microstructure.py",
                    "sha256": "stable",
                    "error": None,
                },
            ],
            "git": {
                "commit": "abc123",
                "dirty": True,
                "commit_error": None,
                "status_error": None,
            },
            "runtime": {
                "python_version": "3.10.0",
                "python_implementation": "CPython",
                "websockets_version": "15.0",
            },
        }

    observations = iter([provenance("start"), provenance("changed")])
    monkeypatch.setattr(collector, "_source_metadata", lambda: next(observations))

    result = collector.run_capture(
        _config(tmp_path),
        connect_fn=FakeConnector(messages_per_channel=2),
        install_signal_handlers=False,
    )

    assert not result.complete
    assert result.manifest["source_code"]["provenance_complete"]
    assert not result.manifest["source_code"]["unchanged_during_capture"]
    assert not result.manifest["quality"]["source_unchanged_during_capture"]

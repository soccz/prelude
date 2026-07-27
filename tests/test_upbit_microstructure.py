from __future__ import annotations

import base64
import hashlib
import json

import pytest

from data.upbit_microstructure import (
    PUBLIC_WEBSOCKET_ENDPOINT,
    RawFrame,
    RawStreamWriter,
    build_subscription,
    event_available_at_cutoff,
    iter_raw_records,
    parse_raw_frame,
)


def trade_payload(
    *,
    market: str = "KRW-BTC",
    event_at_ms: int = 1_730_336_862_047,
    stream_type: str = "REALTIME",
    sequential_id: int = 17_303_368_620_470_000,
) -> dict:
    return {
        "type": "trade",
        "code": market,
        "timestamp": event_at_ms + 35,
        "trade_date": "2024-10-31",
        "trade_time": "01:07:42",
        "trade_timestamp": event_at_ms,
        "trade_price": 100_473_000.0,
        "trade_volume": 0.00014208,
        "ask_bid": "BID",
        "prev_closing_price": 100_571_000.0,
        "change": "FALL",
        "change_price": 98_000.0,
        "sequential_id": sequential_id,
        "best_ask_price": 100_473_000.0,
        "best_ask_size": 0.43139478,
        "best_bid_price": 100_465_000.0,
        "best_bid_size": 0.01990656,
        "stream_type": stream_type,
    }


def orderbook_payload(
    *,
    market: str = "KRW-BTC",
    event_at_ms: int = 1_746_601_573_804,
    stream_type: str = "SNAPSHOT",
) -> dict:
    return {
        "type": "orderbook",
        "code": market,
        "timestamp": event_at_ms,
        "total_ask_size": 4.79158413,
        "total_bid_size": 2.65609625,
        "orderbook_units": [
            {
                "ask_price": 137_002_000.0,
                "ask_size": 0.10623869,
                "bid_price": 137_001_000.0,
                "bid_size": 0.03656812,
            },
            {
                "ask_price": 137_023_000.0,
                "ask_size": 0.06144079,
                "bid_price": 137_000_000.0,
                "bid_size": 0.33543284,
            },
        ],
        "level": 0,
        "stream_type": stream_type,
    }


def raw_frame(
    payload: dict,
    *,
    received_at_ns: int,
    ingress_seq: int = 1,
) -> RawFrame:
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return RawFrame(
        raw=raw,
        received_at_ns=received_at_ns,
        received_monotonic_ns=123_000 + ingress_seq,
        connection_id="connection-1",
        subscription_id="subscription-1",
        ingress_seq=ingress_seq,
    )


def test_public_subscriptions_keep_trade_and_orderbook_contract_separate():
    markets = ["KRW-ETH", "KRW-BTC"]
    trade = build_subscription("trade", markets, ticket="trade-ticket")
    orderbook = build_subscription(
        "orderbook", markets, ticket="book-ticket", orderbook_depth=30
    )

    assert PUBLIC_WEBSOCKET_ENDPOINT == "wss://api.upbit.com/websocket/v1"
    assert trade == [
        {"ticket": "trade-ticket"},
        {
            "type": "trade",
            "codes": ["KRW-BTC", "KRW-ETH"],
            "is_only_realtime": True,
        },
        {"format": "DEFAULT"},
    ]
    assert orderbook == [
        {"ticket": "book-ticket"},
        {
            "type": "orderbook",
            "codes": ["KRW-BTC.30", "KRW-ETH.30"],
            "level": 0,
        },
        {"format": "DEFAULT"},
    ]
    assert "private" not in PUBLIC_WEBSOCKET_ENDPOINT


@pytest.mark.parametrize("depth", [0, 2, 10, 31])
def test_subscription_rejects_unsupported_orderbook_depth(depth):
    with pytest.raises(ValueError, match="orderbook_depth"):
        build_subscription(
            "orderbook", ["KRW-BTC"], ticket="ticket", orderbook_depth=depth
        )


def test_trade_envelope_preserves_exact_raw_payload_and_dual_timestamps(tmp_path):
    payload = trade_payload()
    frame = raw_frame(
        payload,
        received_at_ns=payload["trade_timestamp"] * 1_000_000 + 50_000_000,
    )
    path = tmp_path / "trade.jsonl.gz"
    writer = RawStreamWriter(
        path,
        capture_id="capture-1",
        channel="trade",
        expected_markets={"KRW-BTC"},
    )
    writer.write(frame)
    metadata = writer.close()

    records = list(iter_raw_records(path))
    assert len(records) == 1
    record = records[0]
    assert record["record_type"] == "event"
    assert record["event_at_ms"] == payload["trade_timestamp"]
    assert record["source_timestamp_ms"] == payload["timestamp"]
    assert record["received_at_ns"] == frame.received_at_ns
    assert record["received_monotonic_ns"] == frame.received_monotonic_ns
    assert record["raw_payload"] == frame.raw.decode("utf-8")
    assert record["payload_sha256"] == hashlib.sha256(frame.raw).hexdigest()
    assert json.loads(record["raw_payload"]) == payload
    assert metadata["event_count"] == 1
    assert metadata["realtime_markets"] == ["KRW-BTC"]
    assert not writer.partial_path.exists()


def test_invalid_utf8_preserves_exact_binary_frame():
    frame = RawFrame(
        raw=b"\xff\xfe\x00",
        received_at_ns=1_800_000_000_000_000_000,
        received_monotonic_ns=123,
        connection_id="connection-1",
        subscription_id="subscription-1",
        ingress_seq=1,
        frame_kind="binary",
    )
    envelope, metadata = parse_raw_frame(
        frame,
        capture_id="capture-1",
        channel="trade",
        expected_markets={"KRW-BTC"},
        persisted_at_ns=1_800_000_000_000_000_010,
    )

    assert metadata is None
    assert envelope["record_type"] == "parse_error"
    assert envelope["frame_kind"] == "binary"
    assert base64.b64decode(envelope["raw_payload_base64"]) == frame.raw
    assert envelope["payload_sha256"] == hashlib.sha256(frame.raw).hexdigest()


def test_full_orderbook_levels_and_same_exchange_timestamp_are_both_retained(
    tmp_path,
):
    payload = orderbook_payload()
    path = tmp_path / "orderbook.jsonl.gz"
    writer = RawStreamWriter(
        path,
        capture_id="capture-1",
        channel="orderbook",
        expected_markets={"KRW-BTC"},
    )
    writer.write(
        raw_frame(
            payload,
            received_at_ns=payload["timestamp"] * 1_000_000 + 1,
            ingress_seq=1,
        )
    )
    changed = json.loads(json.dumps(payload))
    changed["stream_type"] = "REALTIME"
    changed["orderbook_units"][0]["bid_size"] = 9.5
    writer.write(
        raw_frame(
            changed,
            received_at_ns=payload["timestamp"] * 1_000_000 + 2,
            ingress_seq=2,
        )
    )
    metadata = writer.close()

    records = list(iter_raw_records(path))
    assert len(records) == 2
    assert records[0]["event_at_ms"] == records[1]["event_at_ms"]
    assert records[0]["ingress_seq"] == 1
    assert records[1]["ingress_seq"] == 2
    assert (
        json.loads(records[1]["raw_payload"])["orderbook_units"][0]["bid_size"]
        == 9.5
    )
    assert metadata["snapshot_markets"] == ["KRW-BTC"]
    assert metadata["realtime_markets"] == ["KRW-BTC"]


def test_duplicate_trade_id_is_preserved_in_raw_evidence(tmp_path):
    """Raw lake는 중복을 숨기지 않고, downstream feature가 ID로 dedupe한다."""
    payload = trade_payload(sequential_id=999)
    path = tmp_path / "trade.jsonl.gz"
    writer = RawStreamWriter(
        path,
        capture_id="capture-1",
        channel="trade",
        expected_markets={"KRW-BTC"},
    )
    for seq in (1, 2):
        writer.write(
            raw_frame(
                payload,
                received_at_ns=payload["trade_timestamp"] * 1_000_000 + seq,
                ingress_seq=seq,
            )
        )
    writer.close()

    records = list(iter_raw_records(path))
    assert len(records) == 2
    assert {
        json.loads(record["raw_payload"])["sequential_id"] for record in records
    } == {999}


def test_late_arrival_cannot_be_used_even_when_exchange_event_is_old():
    cutoff_ns = 10_000_000_000
    record = {
        "record_type": "event",
        "event_at_ms": 9_000,
        "received_at_ns": cutoff_ns + 1,
    }
    assert not event_available_at_cutoff(record, cutoff_ns)

    record["received_at_ns"] = cutoff_ns
    assert event_available_at_cutoff(record, cutoff_ns)

    record["event_at_ms"] = 10_001
    assert not event_available_at_cutoff(record, cutoff_ns)


@pytest.mark.parametrize(
    ("payload", "error_fragment"),
    [
        ({"error": {"name": "NO_CODES", "message": "codes required"}}, "NO_CODES"),
        ({"type": "ticker", "code": "KRW-BTC"}, "expected channel=trade"),
        (trade_payload(market="KRW-NOT-SUBSCRIBED"), "unexpected market"),
        (trade_payload(stream_type="SNAPSHOT"), "realtime-only"),
    ],
)
def test_invalid_or_server_frames_are_preserved_and_fail_closed(
    payload,
    error_fragment,
):
    frame = raw_frame(payload, received_at_ns=1_800_000_000_000_000_000)
    envelope, meta = parse_raw_frame(
        frame,
        capture_id="capture-1",
        channel="trade",
        expected_markets={"KRW-BTC"},
        persisted_at_ns=1_800_000_000_000_000_010,
    )
    assert meta is None
    assert envelope["record_type"] in {"parse_error", "server_error"}
    assert error_fragment in envelope["error"]
    assert envelope["raw_payload"] == frame.raw.decode("utf-8")


def test_raw_writer_marks_schema_error_fatal(tmp_path):
    frame = raw_frame(
        {"type": "trade", "code": "KRW-BTC"},
        received_at_ns=1_800_000_000_000_000_000,
    )
    writer = RawStreamWriter(
        tmp_path / "trade.jsonl.gz",
        capture_id="capture-1",
        channel="trade",
        expected_markets={"KRW-BTC"},
    )
    writer.write(frame)
    metadata = writer.close()
    assert metadata["parse_error_count"] == 1
    assert metadata["fatal_error"]

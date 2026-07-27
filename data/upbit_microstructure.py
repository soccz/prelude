"""업비트 공개 체결·호가 원시 스트림의 영속화 계약.

이 모듈은 모델 피처를 미리 정하지 않는다. WebSocket payload를 그대로 보존하고
거래소 event 시각과 로컬 수신 시각을 함께 남겨, 나중에 동일 raw data로 체결
불균형·호가 충격·흡수·재충전 proxy를 다시 계산할 수 있게 한다.

공개 Quotation endpoint만 사용하며 API key나 주문 기능은 포함하지 않는다.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import math
import os
import re
import stat
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal


RAW_SCHEMA_VERSION = "upbit_microstructure.raw.v1"
MANIFEST_SCHEMA_VERSION = "upbit_microstructure.capture.v1"
PUBLIC_WEBSOCKET_ENDPOINT = "wss://api.upbit.com/websocket/v1"
SUPPORTED_CHANNELS = ("trade", "orderbook")
SUPPORTED_ORDERBOOK_DEPTHS = (1, 5, 15, 30)

Channel = Literal["trade", "orderbook"]

_TRADE_REQUIRED = {
    "trade_price",
    "trade_volume",
    "ask_bid",
    "trade_timestamp",
    "timestamp",
    "sequential_id",
    "stream_type",
}
_ORDERBOOK_REQUIRED = {
    "total_ask_size",
    "total_bid_size",
    "orderbook_units",
    "timestamp",
    "stream_type",
}
_ORDERBOOK_UNIT_REQUIRED = {
    "ask_price",
    "ask_size",
    "bid_price",
    "bid_size",
}


class RawFrameError(ValueError):
    """수신 frame이 공식 trade/orderbook 계약과 맞지 않는다."""


@dataclass(frozen=True)
class RawFrame:
    """네트워크 수신 직후 찍은 시각과 원문 frame."""

    raw: bytes
    received_at_ns: int
    received_monotonic_ns: int
    connection_id: str
    subscription_id: str
    ingress_seq: int
    frame_kind: Literal["binary", "text"] = "binary"


@dataclass
class StreamStats:
    """한 채널 raw artifact의 품질·커버리지 통계."""

    channel: Channel
    record_count: int = 0
    event_count: int = 0
    parse_error_count: int = 0
    server_error_count: int = 0
    unknown_market_count: int = 0
    wrong_channel_count: int = 0
    bytes_received: int = 0
    first_received_at_ns: int | None = None
    last_received_at_ns: int | None = None
    first_event_at_ms: int | None = None
    last_event_at_ms: int | None = None
    snapshot_markets: set[str] = field(default_factory=set)
    realtime_markets: set[str] = field(default_factory=set)
    markets_seen: set[str] = field(default_factory=set)
    requested_orderbook_depth: int | None = None
    orderbook_unit_counts: dict[int, int] = field(default_factory=dict)
    fatal_error: str | None = None

    def note_time(self, *, received_at_ns: int, event_at_ms: int | None) -> None:
        if self.first_received_at_ns is None:
            self.first_received_at_ns = received_at_ns
        self.last_received_at_ns = received_at_ns
        if event_at_ms is not None:
            if self.first_event_at_ms is None:
                self.first_event_at_ms = event_at_ms
            self.last_event_at_ms = event_at_ms

    def note_orderbook_depth(self, unit_count: int) -> None:
        self.orderbook_unit_counts[unit_count] = (
            self.orderbook_unit_counts.get(unit_count, 0) + 1
        )

    def as_dict(self, expected_markets: set[str]) -> dict[str, Any]:
        observed_counts = sorted(self.orderbook_unit_counts)
        requested_depth = self.requested_orderbook_depth
        return {
            "channel": self.channel,
            "record_count": self.record_count,
            "event_count": self.event_count,
            "parse_error_count": self.parse_error_count,
            "server_error_count": self.server_error_count,
            "unknown_market_count": self.unknown_market_count,
            "wrong_channel_count": self.wrong_channel_count,
            "bytes_received": self.bytes_received,
            "first_received_at_ns": self.first_received_at_ns,
            "first_received_at": ns_to_iso(self.first_received_at_ns),
            "last_received_at_ns": self.last_received_at_ns,
            "last_received_at": ns_to_iso(self.last_received_at_ns),
            "first_event_at_ms": self.first_event_at_ms,
            "first_event_at": ms_to_iso(self.first_event_at_ms),
            "last_event_at_ms": self.last_event_at_ms,
            "last_event_at": ms_to_iso(self.last_event_at_ms),
            "snapshot_markets": sorted(self.snapshot_markets),
            "snapshot_market_count": len(self.snapshot_markets),
            "realtime_markets": sorted(self.realtime_markets),
            "realtime_market_count": len(self.realtime_markets),
            "markets_seen": sorted(self.markets_seen),
            "market_count": len(self.markets_seen),
            "missing_markets": sorted(expected_markets - self.markets_seen),
            "requested_orderbook_depth": requested_depth,
            "observed_orderbook_unit_count_min": (
                observed_counts[0] if observed_counts else None
            ),
            "observed_orderbook_unit_count_max": (
                observed_counts[-1] if observed_counts else None
            ),
            "observed_orderbook_unit_count_histogram": {
                str(count): self.orderbook_unit_counts[count]
                for count in observed_counts
            },
            "thin_book_event_count": (
                sum(
                    frequency
                    for count, frequency in self.orderbook_unit_counts.items()
                    if requested_depth is not None and count < requested_depth
                )
            ),
            "fatal_error": self.fatal_error,
        }


def utc_now_ns() -> int:
    return time.time_ns()


def ns_to_iso(value: int | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value / 1_000_000_000, timezone.utc).isoformat()


def ms_to_iso(value: int | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value / 1_000, timezone.utc).isoformat()


def parse_utc_datetime(value: str) -> datetime:
    """timezone이 명시된 ISO-8601 문자열만 UTC로 정규화한다."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include timezone")
    return parsed.astimezone(timezone.utc)


def datetime_to_ns(value: datetime) -> int:
    if value.tzinfo is None:
        raise ValueError("datetime must include timezone")
    return int(value.astimezone(timezone.utc).timestamp() * 1_000_000_000)


def normalise_markets(markets: list[str] | tuple[str, ...]) -> list[str]:
    """KRW market 목록을 trim/uppercase/dedupe/sort하고 형식을 검증한다."""
    if any(not isinstance(market, str) for market in markets):
        raise ValueError("markets must contain strings only")
    normalised = sorted({market.strip().upper() for market in markets})
    if not normalised:
        raise ValueError("markets must not be empty")
    invalid = [
        market
        for market in normalised
        if re.fullmatch(r"KRW-[A-Z0-9]+", market) is None
    ]
    if invalid:
        raise ValueError(f"only KRW markets are supported: {invalid[:3]}")
    return normalised


def build_subscription(
    channel: Channel,
    markets: list[str] | tuple[str, ...],
    *,
    ticket: str,
    orderbook_depth: int = 30,
) -> list[dict[str, Any]]:
    """공식 DEFAULT format 구독 payload.

    Trade는 flow에 과거 snapshot 체결이 섞이지 않도록 REALTIME만 받는다.
    Orderbook은 최초 SNAPSHOT으로 상태를 세운 뒤 REALTIME을 받는다.
    """
    if channel not in SUPPORTED_CHANNELS:
        raise ValueError(f"unsupported channel: {channel}")
    if not ticket:
        raise ValueError("ticket must not be empty")
    codes = normalise_markets(markets)

    if channel == "trade":
        data_request: dict[str, Any] = {
            "type": "trade",
            "codes": codes,
            "is_only_realtime": True,
        }
    else:
        if orderbook_depth not in SUPPORTED_ORDERBOOK_DEPTHS:
            raise ValueError(
                f"orderbook_depth must be one of {SUPPORTED_ORDERBOOK_DEPTHS}"
            )
        data_request = {
            "type": "orderbook",
            "codes": [f"{market}.{orderbook_depth}" for market in codes],
            "level": 0,
        }
    return [{"ticket": ticket}, data_request, {"format": "DEFAULT"}]


def universe_sha256(markets: list[str] | tuple[str, ...]) -> str:
    payload = "\n".join(normalise_markets(markets)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Hash one stable private regular file without following a final symlink."""
    source = Path(path)
    try:
        path_before = source.lstat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"regular file missing: {source}") from exc
    if not stat.S_ISREG(path_before.st_mode) or path_before.st_nlink != 1:
        raise RawFrameError(
            f"raw artifact must be a private regular file: {source}"
        )

    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    fd = os.open(source, flags)
    digest = hashlib.sha256()
    with os.fdopen(fd, "rb") as handle:
        fd_before = os.fstat(handle.fileno())
        if (
            not stat.S_ISREG(fd_before.st_mode)
            or fd_before.st_nlink != 1
            or (path_before.st_dev, path_before.st_ino)
            != (fd_before.st_dev, fd_before.st_ino)
        ):
            raise RawFrameError(
                f"raw artifact changed before hashing: {source}"
            )
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        fd_after = os.fstat(handle.fileno())
    path_after = source.lstat()
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(
        getattr(fd_before, field_name) != getattr(fd_after, field_name)
        or getattr(fd_after, field_name) != getattr(path_after, field_name)
        for field_name in stable_fields
    ):
        raise RawFrameError(f"raw artifact changed while hashing: {source}")
    return digest.hexdigest()


def _payload_market(payload: dict[str, Any]) -> str:
    value = payload.get("code", payload.get("market"))
    if not isinstance(value, str) or not value:
        raise RawFrameError("payload has no code/market")
    return value.upper()


def _positive_epoch_ms(value: Any, field_name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not float(value).is_integer()
    ):
        raise RawFrameError(f"{field_name} must be epoch milliseconds")
    result = int(value)
    if result <= 0:
        raise RawFrameError(f"{field_name} must be positive")
    return result


def _positive_number(value: Any, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise RawFrameError(f"{field_name} must be finite and positive")
    return float(value)


def _positive_integer(value: Any, field_name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not float(value).is_integer()
        or int(value) <= 0
    ):
        raise RawFrameError(f"{field_name} must be a positive integer")
    return int(value)


def _validate_payload(
    payload: dict[str, Any],
    channel: Channel,
) -> tuple[str, int, dict[str, Any]]:
    payload_channel = payload.get("type")
    if payload_channel != channel:
        raise RawFrameError(
            f"expected channel={channel}, received type={payload_channel!r}"
        )
    market = _payload_market(payload)
    required = _TRADE_REQUIRED if channel == "trade" else _ORDERBOOK_REQUIRED
    missing = sorted(field for field in required if field not in payload)
    if missing:
        raise RawFrameError(f"missing fields: {missing}")
    stream_type = payload.get("stream_type")
    if stream_type not in {"SNAPSHOT", "REALTIME"}:
        raise RawFrameError("stream_type must be SNAPSHOT or REALTIME")

    if channel == "trade":
        if stream_type != "REALTIME":
            raise RawFrameError(
                "trade subscription is realtime-only but received non-REALTIME"
            )
        if payload.get("ask_bid") not in {"ASK", "BID"}:
            raise RawFrameError("trade ask_bid must be ASK or BID")
        _positive_number(payload.get("trade_price"), "trade_price")
        _positive_number(payload.get("trade_volume"), "trade_volume")
        _positive_integer(payload.get("sequential_id"), "sequential_id")
        best_ask = payload.get("best_ask_price")
        best_bid = payload.get("best_bid_price")
        if best_ask is not None:
            best_ask = _positive_number(best_ask, "best_ask_price")
        if best_bid is not None:
            best_bid = _positive_number(best_bid, "best_bid_price")
        if best_ask is not None and best_bid is not None and best_bid > best_ask:
            raise RawFrameError("best_bid_price must be <= best_ask_price")
        for field_name in ("best_ask_size", "best_bid_size"):
            if payload.get(field_name) is not None:
                _positive_number(payload[field_name], field_name)
        event_at_ms = _positive_epoch_ms(
            payload.get("trade_timestamp"), "trade_timestamp"
        )
        metadata: dict[str, Any] = {}
    else:
        _positive_number(payload.get("total_ask_size"), "total_ask_size")
        _positive_number(payload.get("total_bid_size"), "total_bid_size")
        units = payload.get("orderbook_units")
        if not isinstance(units, list) or not units:
            raise RawFrameError("orderbook_units must be a non-empty list")
        ask_prices: list[float] = []
        bid_prices: list[float] = []
        for index, unit in enumerate(units):
            if not isinstance(unit, dict):
                raise RawFrameError(f"orderbook_units[{index}] must be an object")
            unit_missing = sorted(_ORDERBOOK_UNIT_REQUIRED - set(unit))
            if unit_missing:
                raise RawFrameError(
                    f"orderbook_units[{index}] missing fields: {unit_missing}"
                )
            ask_price = _positive_number(
                unit["ask_price"], f"orderbook_units[{index}].ask_price"
            )
            bid_price = _positive_number(
                unit["bid_price"], f"orderbook_units[{index}].bid_price"
            )
            _positive_number(
                unit["ask_size"], f"orderbook_units[{index}].ask_size"
            )
            _positive_number(
                unit["bid_size"], f"orderbook_units[{index}].bid_size"
            )
            if bid_price > ask_price:
                raise RawFrameError(
                    f"orderbook_units[{index}] bid_price must be <= ask_price"
                )
            ask_prices.append(ask_price)
            bid_prices.append(bid_price)
        if max(bid_prices) > min(ask_prices):
            raise RawFrameError("orderbook best bid must be <= best ask")
        event_at_ms = _positive_epoch_ms(payload.get("timestamp"), "timestamp")
        metadata = {"orderbook_unit_count": len(units)}
    return market, event_at_ms, metadata


def parse_raw_frame(
    frame: RawFrame,
    *,
    capture_id: str,
    channel: Channel,
    expected_markets: set[str],
    persisted_at_ns: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """원문을 감사 envelope로 감싼다.

    반환 두 번째 값은 정상 event의 최소 메타데이터다. parse/server 오류도 원문과
    checksum을 잃지 않고 record_type을 달리해 보존한다.
    """
    persisted_ns = utc_now_ns() if persisted_at_ns is None else persisted_at_ns
    raw_sha = hashlib.sha256(frame.raw).hexdigest()
    try:
        raw_text = frame.raw.decode("utf-8")
    except UnicodeDecodeError:
        raw_text = frame.raw.decode("utf-8", errors="replace")
        error = "payload is not valid UTF-8"
        envelope = _error_envelope(
            frame,
            capture_id=capture_id,
            channel=channel,
            persisted_at_ns=persisted_ns,
            raw_payload=raw_text,
            raw_sha256=raw_sha,
            record_type="parse_error",
            error=error,
            raw_payload_base64=base64.b64encode(frame.raw).decode("ascii"),
        )
        return envelope, None

    try:
        payload = json.loads(raw_text)
        if not isinstance(payload, dict):
            raise RawFrameError("DEFAULT stream frame must be a JSON object")
        if "error" in payload:
            envelope = _error_envelope(
                frame,
                capture_id=capture_id,
                channel=channel,
                persisted_at_ns=persisted_ns,
                raw_payload=raw_text,
                raw_sha256=raw_sha,
                record_type="server_error",
                error=json.dumps(payload["error"], ensure_ascii=False),
            )
            return envelope, None
        market, event_at_ms, validation_metadata = _validate_payload(payload, channel)
        if market not in expected_markets:
            raise RawFrameError(f"unexpected market: {market}")
        source_timestamp_ms = _positive_epoch_ms(
            payload.get("timestamp"), "timestamp"
        )
    except (json.JSONDecodeError, RawFrameError) as exc:
        envelope = _error_envelope(
            frame,
            capture_id=capture_id,
            channel=channel,
            persisted_at_ns=persisted_ns,
            raw_payload=raw_text,
            raw_sha256=raw_sha,
            record_type="parse_error",
            error=str(exc),
        )
        return envelope, None

    envelope = {
        "schema_version": RAW_SCHEMA_VERSION,
        "record_type": "event",
        "source": "upbit",
        "channel": channel,
        "market": market,
        "capture_id": capture_id,
        "connection_id": frame.connection_id,
        "subscription_id": frame.subscription_id,
        "ingress_seq": frame.ingress_seq,
        "frame_kind": frame.frame_kind,
        "stream_type": payload.get("stream_type"),
        "event_at_ms": event_at_ms,
        "event_at": ms_to_iso(event_at_ms),
        "source_timestamp_ms": source_timestamp_ms,
        "source_timestamp": ms_to_iso(source_timestamp_ms),
        "received_at_ns": frame.received_at_ns,
        "received_at": ns_to_iso(frame.received_at_ns),
        "received_monotonic_ns": frame.received_monotonic_ns,
        "persisted_at_ns": persisted_ns,
        "persisted_at": ns_to_iso(persisted_ns),
        "payload_sha256": raw_sha,
        "raw_payload": raw_text,
        **validation_metadata,
    }
    meta = {
        "market": market,
        "event_at_ms": event_at_ms,
        "stream_type": payload.get("stream_type"),
        **validation_metadata,
    }
    return envelope, meta


def _error_envelope(
    frame: RawFrame,
    *,
    capture_id: str,
    channel: Channel,
    persisted_at_ns: int,
    raw_payload: str,
    raw_sha256: str,
    record_type: str,
    error: str,
    raw_payload_base64: str | None = None,
) -> dict[str, Any]:
    envelope = {
        "schema_version": RAW_SCHEMA_VERSION,
        "record_type": record_type,
        "source": "upbit",
        "channel": channel,
        "capture_id": capture_id,
        "connection_id": frame.connection_id,
        "subscription_id": frame.subscription_id,
        "ingress_seq": frame.ingress_seq,
        "frame_kind": frame.frame_kind,
        "received_at_ns": frame.received_at_ns,
        "received_at": ns_to_iso(frame.received_at_ns),
        "received_monotonic_ns": frame.received_monotonic_ns,
        "persisted_at_ns": persisted_at_ns,
        "persisted_at": ns_to_iso(persisted_at_ns),
        "payload_sha256": raw_sha256,
        "raw_payload": raw_payload,
        "error": error,
    }
    if raw_payload_base64 is not None:
        envelope["raw_payload_base64"] = raw_payload_base64
    return envelope


class RawStreamWriter:
    """한 채널을 append-only ``jsonl.gz``로 원자 finalize한다.

    ``close()``는 단계별 진행 상태를 보존해 replace/fsync 같은 후반 I/O 실패 뒤
    재호출할 수 있다. 복구를 포기할 때는 ``abort()``가 partial을 증거 파일로
    격리한다. 실패한 close를 성공으로 가장하지 않는다.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        capture_id: str,
        channel: Channel,
        expected_markets: set[str],
        requested_orderbook_depth: int | None = None,
        compresslevel: int = 1,
    ):
        self.final_path = Path(path)
        self.partial_path = self.final_path.with_suffix(
            self.final_path.suffix + ".partial"
        )
        self.capture_id = capture_id
        self.channel = channel
        self.expected_markets = set(expected_markets)
        self.stats = StreamStats(
            channel=channel,
            requested_orderbook_depth=(
                requested_orderbook_depth if channel == "orderbook" else None
            ),
        )
        self._state = "open"
        self._text_sealed = False
        self._raw_synced = False
        self._raw_closed = False
        self._renamed = False
        self._directory_synced = False
        self._aborted_path: Path | None = None
        self._close_error: str | None = None
        self._abort_errors: list[str] = []

        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        if self.final_path.exists():
            raise FileExistsError(f"raw final artifact already exists: {self.final_path}")
        if self.partial_path.exists():
            raise FileExistsError(
                f"orphan raw partial must be recovered first: {self.partial_path}"
            )
        try:
            self._raw_handle = self.partial_path.open("xb")
            self._gzip_handle = gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=compresslevel,
                fileobj=self._raw_handle,
                mtime=0,
            )
            self._text_handle = io.TextIOWrapper(
                self._gzip_handle, encoding="utf-8", newline="\n"
            )
        except Exception:
            for handle_name in ("_text_handle", "_gzip_handle", "_raw_handle"):
                handle = getattr(self, handle_name, None)
                try:
                    if handle is not None:
                        handle.close()
                except Exception as close_exc:
                    self._abort_errors.append(
                        f"{handle_name} init cleanup: "
                        f"{type(close_exc).__name__}: {close_exc}"
                    )
            if self.partial_path.exists():
                failed_path = self.partial_path.with_name(
                    f"{self.partial_path.name}.aborted-init-{time.time_ns()}"
                )
                try:
                    os.replace(self.partial_path, failed_path)
                except OSError as rename_exc:
                    self._abort_errors.append(
                        "partial init quarantine: "
                        f"{type(rename_exc).__name__}: {rename_exc}"
                    )
            raise

    def write(self, frame: RawFrame) -> None:
        if self._state != "open":
            raise RuntimeError(f"writer does not accept writes in state={self._state}")
        envelope, meta = parse_raw_frame(
            frame,
            capture_id=self.capture_id,
            channel=self.channel,
            expected_markets=self.expected_markets,
        )
        self._text_handle.write(
            json.dumps(
                envelope,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        self._text_handle.write("\n")

        self.stats.record_count += 1
        self.stats.bytes_received += len(frame.raw)
        record_type = envelope["record_type"]
        if record_type == "parse_error":
            self.stats.parse_error_count += 1
            error = str(envelope.get("error", "parse error"))
            if error.startswith("unexpected market:"):
                self.stats.unknown_market_count += 1
            if error.startswith("expected channel="):
                self.stats.wrong_channel_count += 1
            self.stats.fatal_error = self.stats.fatal_error or error
        elif record_type == "server_error":
            self.stats.server_error_count += 1
            self.stats.fatal_error = self.stats.fatal_error or str(
                envelope.get("error", "server error")
            )
        else:
            if meta is None:
                raise RawFrameError(
                    "validated market event is missing parsed metadata"
                )
            self.stats.event_count += 1
            market = str(meta["market"])
            self.stats.markets_seen.add(market)
            stream_type = str(meta.get("stream_type", ""))
            if stream_type == "SNAPSHOT":
                self.stats.snapshot_markets.add(market)
            elif stream_type == "REALTIME":
                self.stats.realtime_markets.add(market)
            self.stats.note_time(
                received_at_ns=frame.received_at_ns,
                event_at_ms=int(meta["event_at_ms"]),
            )
            unit_count = meta.get("orderbook_unit_count")
            if unit_count is not None:
                self.stats.note_orderbook_depth(int(unit_count))

    def close(self) -> dict[str, Any]:
        if self._state == "finalized":
            return self.artifact_metadata()
        if self._state in {"aborting", "aborted"}:
            raise RuntimeError("aborted writer cannot be finalized")
        self._state = "closing"
        try:
            if not self._text_sealed:
                self._text_handle.flush()
                self._text_handle.close()
                self._text_sealed = True
            if not self._raw_synced:
                self._raw_handle.flush()
                os.fsync(self._raw_handle.fileno())
                self._raw_synced = True
            if not self._raw_closed:
                self._raw_handle.close()
                self._raw_closed = True
            if not self._renamed:
                os.replace(self.partial_path, self.final_path)
                self._renamed = True
            if not self._directory_synced:
                self._fsync_parent()
                self._directory_synced = True
        except Exception as exc:
            self._text_sealed = self._text_sealed or self._text_handle.closed
            self._raw_closed = self._raw_closed or self._raw_handle.closed
            self._state = "close_failed"
            self._close_error = f"{type(exc).__name__}: {exc}"
            raise
        self._state = "finalized"
        self._close_error = None
        return self.artifact_metadata()

    def abort(self, *, reason: str | None = None) -> dict[str, Any]:
        """열린/close-failed writer를 닫고 partial/final을 aborted 증거로 격리한다."""
        if self._state == "finalized":
            return self.artifact_metadata()
        if self._state == "aborted":
            return self.artifact_metadata()
        self._state = "aborting"
        if reason:
            self._close_error = self._close_error or reason

        for handle_name in ("_text_handle", "_gzip_handle", "_raw_handle"):
            handle = getattr(self, handle_name, None)
            if handle is None or getattr(handle, "closed", False):
                continue
            try:
                handle.close()
            except Exception as exc:
                self._abort_errors.append(
                    f"{handle_name} {type(exc).__name__}: {exc}"
                )

        evidence_source: Path | None = None
        if self.partial_path.exists():
            evidence_source = self.partial_path
        elif self.final_path.exists():
            evidence_source = self.final_path
        if evidence_source is not None:
            aborted_path = self.final_path.with_name(
                f"{self.final_path.name}.aborted-{time.time_ns()}"
            )
            try:
                os.replace(evidence_source, aborted_path)
                self._aborted_path = aborted_path
                self._fsync_parent()
            except Exception as exc:
                self._abort_errors.append(
                    f"preserve {type(exc).__name__}: {exc}"
                )

        self._state = "aborted"
        return self.artifact_metadata()

    def _fsync_parent(self) -> None:
        dir_fd = os.open(self.final_path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

    def artifact_metadata(self) -> dict[str, Any]:
        finalized = self._state == "finalized" and self.final_path.exists()
        evidence_path = self._aborted_path
        if evidence_path is None and self.partial_path.exists():
            evidence_path = self.partial_path
        if (
            evidence_path is None
            and self._state != "finalized"
            and self.final_path.exists()
        ):
            evidence_path = self.final_path
        return {
            "path": str(self.final_path),
            "exists": finalized,
            "size_bytes": self.final_path.stat().st_size if finalized else None,
            "sha256": file_sha256(self.final_path) if finalized else None,
            "writer_state": self._state,
            "partial_path": (
                str(self.partial_path) if self.partial_path.exists() else None
            ),
            "aborted_path": (
                str(self._aborted_path) if self._aborted_path is not None else None
            ),
            "preserved_evidence_path": (
                str(evidence_path) if evidence_path is not None else None
            ),
            "close_error": self._close_error,
            "abort_errors": list(self._abort_errors),
            **self.stats.as_dict(self.expected_markets),
        }


def iter_raw_records(path: str | Path) -> Iterator[dict[str, Any]]:
    with gzip.open(Path(path), "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RawFrameError("raw artifact line must be a JSON object")
                yield value


def event_available_at_cutoff(record: dict[str, Any], cutoff_at_ns: int) -> bool:
    """거래소 시각과 실제 수신 시각이 모두 cutoff 이전인 event만 허용한다."""
    if record.get("record_type") != "event":
        return False
    try:
        event_at_ns = int(record["event_at_ms"]) * 1_000_000
        received_at_ns = int(record["received_at_ns"])
    except (KeyError, TypeError, ValueError):
        return False
    return event_at_ns <= cutoff_at_ns and received_at_ns <= cutoff_at_ns


def atomic_write_json(path: str | Path, document: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                document,
                handle,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                indent=2,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, destination)
        dir_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def clock_sync_status() -> dict[str, Any]:
    """로컬 wall clock의 NTP 동기화 상태를 best-effort로 기록한다."""
    checked_at_ns = utc_now_ns()
    result: dict[str, Any] = {
        "checked_at_ns": checked_at_ns,
        "checked_at": ns_to_iso(checked_at_ns),
        "ntp_synchronized": None,
        "source": "timedatectl",
    }
    try:
        completed = subprocess.run(
            [
                "/usr/bin/timedatectl",
                "show",
                "--property=NTPSynchronized",
                "--value",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        result["error"] = type(exc).__name__
        return result
    value = completed.stdout.strip().lower()
    if completed.returncode == 0 and value in {"yes", "no"}:
        result["ntp_synchronized"] = value == "yes"
    else:
        result["error"] = completed.stderr.strip() or f"returncode={completed.returncode}"
    return result

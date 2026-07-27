"""업비트 KRW 체결·호가 record-only WebSocket 수집기.

목적:
  - 활성 추천·텔레그램·ledger를 건드리지 않고 추천 직전 microstructure 원시자료 축적
  - trade와 orderbook을 별도 공개 WebSocket 연결로 받아 상호 backpressure 격리
  - 거래소 event_at과 callback 직후 received_at을 함께 저장
  - 실제 R1 snapshot created_at을 feature cutoff로 묶은 감사 manifest 생성

자동 주문, private endpoint, API key는 사용하지 않는다.

Canary:
    python -m data.collector_upbit_microstructure \
        --markets KRW-BTC --duration-seconds 10 \
        --required-warmup-seconds 0

일일 shadow capture 후보(스케줄 연결은 별도 운영 승인 후):
    python -m data.collector_upbit_microstructure \
        --cutoff-slot open --asof 2026-07-27
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import queue
import random
import signal
import stat
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, time as datetime_time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, ContextManager, Iterator, Literal, cast
from zoneinfo import ZoneInfo

from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect as websocket_connect

# 프로젝트 root를 path에 추가(직접 실행 시).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.upbit_microstructure import (  # noqa: E402
    MANIFEST_SCHEMA_VERSION,
    PUBLIC_WEBSOCKET_ENDPOINT,
    SUPPORTED_ORDERBOOK_DEPTHS,
    Channel,
    RawFrame,
    RawStreamWriter,
    atomic_write_json,
    build_subscription,
    clock_sync_status,
    datetime_to_ns,
    event_available_at_cutoff,
    file_sha256,
    iter_raw_records,
    normalise_markets,
    ns_to_iso,
    parse_utc_datetime,
    universe_sha256,
    utc_now_ns,
)
from ops.artifact_provenance import (  # noqa: E402
    ArtifactSourceChangedError,
    ArtifactValidationError,
    file_identity,
    strict_json_object,
)


KST = ZoneInfo("Asia/Seoul")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "microstructure" / "upbit"
DEFAULT_SNAPSHOT_ROOT = PROJECT_ROOT / "output" / "recommend_snapshots"
DEFAULT_QUEUE_MAX = 50_000
DEFAULT_MAX_WAIT_SECONDS = 30 * 60
DEFAULT_REQUIRED_WARMUP_SECONDS = 10 * 60
_CAPTURE_LOCK_NAME = ".capture-owner.lock"
_SENTINEL = object()

logger = logging.getLogger("collector_upbit_microstructure")

ConnectFn = Callable[..., ContextManager[Any]]


class CaptureOwnershipError(RuntimeError):
    """동일 output root의 수집기 소유권을 안전하게 얻지 못한 경우."""


@dataclass(frozen=True)
class CaptureConfig:
    markets: tuple[str, ...]
    output_root: Path = DEFAULT_OUTPUT_ROOT
    endpoint: str = PUBLIC_WEBSOCKET_ENDPOINT
    orderbook_depth: int = 30
    queue_max: int = DEFAULT_QUEUE_MAX
    max_wait_seconds: float = DEFAULT_MAX_WAIT_SECONDS
    required_warmup_seconds: float = DEFAULT_REQUIRED_WARMUP_SECONDS
    feature_cutoff_at: datetime | None = None
    duration_seconds: float | None = None
    cutoff_slot: str | None = None
    asof: date | None = None
    snapshot_root: Path = DEFAULT_SNAPSHOT_ROOT
    universe_source: str = "explicit"
    universe_fetch_started_at_ns: int | None = None
    universe_observed_at_ns: int | None = None

    def __post_init__(self) -> None:
        normalised_markets = tuple(normalise_markets(self.markets))
        object.__setattr__(self, "markets", normalised_markets)
        modes = sum(
            value is not None
            for value in (
                self.feature_cutoff_at,
                self.duration_seconds,
                self.cutoff_slot,
            )
        )
        if modes != 1:
            raise ValueError(
                "exactly one cutoff mode is required: feature_cutoff_at, "
                "duration_seconds, or cutoff_slot"
            )
        if self.cutoff_slot not in {None, "open"}:
            raise ValueError("cutoff_slot currently supports only 'open'")
        if (
            self.feature_cutoff_at is not None
            and self.feature_cutoff_at.tzinfo is None
        ):
            raise ValueError("feature_cutoff_at must include timezone")
        if self.universe_source not in {"explicit", "live"}:
            raise ValueError("universe_source must be explicit or live")
        observed_at_ns = (
            utc_now_ns()
            if self.universe_observed_at_ns is None
            else self.universe_observed_at_ns
        )
        if (
            isinstance(observed_at_ns, bool)
            or not isinstance(observed_at_ns, int)
            or observed_at_ns <= 0
        ):
            raise ValueError("universe_observed_at_ns must be positive")
        object.__setattr__(self, "universe_observed_at_ns", observed_at_ns)
        fetch_started_at_ns = self.universe_fetch_started_at_ns
        if self.universe_source == "live" and fetch_started_at_ns is None:
            raise ValueError("live universe requires universe_fetch_started_at_ns")
        if fetch_started_at_ns is not None:
            if (
                isinstance(fetch_started_at_ns, bool)
                or not isinstance(fetch_started_at_ns, int)
                or fetch_started_at_ns <= 0
            ):
                raise ValueError("universe_fetch_started_at_ns must be positive")
            if fetch_started_at_ns > observed_at_ns:
                raise ValueError("universe fetch start must not follow observation")
        if self.orderbook_depth not in SUPPORTED_ORDERBOOK_DEPTHS:
            raise ValueError(
                f"orderbook_depth must be one of {SUPPORTED_ORDERBOOK_DEPTHS}"
            )
        if self.queue_max <= 0:
            raise ValueError("queue_max must be positive")
        if self.duration_seconds is not None and self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if self.max_wait_seconds <= 0:
            raise ValueError("max_wait_seconds must be positive")
        if self.required_warmup_seconds < 0:
            raise ValueError("required_warmup_seconds must be non-negative")


@dataclass
class ConnectionRecord:
    connection_id: str
    subscription_id: str
    attempt: int
    opened_at_ns: int
    subscribed_at_ns: int | None = None
    closed_at_ns: int | None = None
    first_ingress_seq: int | None = None
    last_ingress_seq: int | None = None
    clean_stop: bool = False
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "connection_id": self.connection_id,
            "subscription_id": self.subscription_id,
            "attempt": self.attempt,
            "opened_at_ns": self.opened_at_ns,
            "opened_at": ns_to_iso(self.opened_at_ns),
            "subscribed_at_ns": self.subscribed_at_ns,
            "subscribed_at": ns_to_iso(self.subscribed_at_ns),
            "closed_at_ns": self.closed_at_ns,
            "closed_at": ns_to_iso(self.closed_at_ns),
            "first_ingress_seq": self.first_ingress_seq,
            "last_ingress_seq": self.last_ingress_seq,
            "clean_stop": self.clean_stop,
            "error": self.error,
        }


@dataclass
class WorkerState:
    channel: Channel
    dropped_frames: int = 0
    reconnect_count: int = 0
    writer_error: str | None = None
    receiver_error: str | None = None
    artifact: dict[str, Any] | None = None
    connections: list[ConnectionRecord] = field(default_factory=list)
    gaps: list[dict[str, Any]] = field(default_factory=list)

    @property
    def fatal_error(self) -> str | None:
        return self.writer_error or self.receiver_error

    def as_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "dropped_frames": self.dropped_frames,
            "reconnect_count": self.reconnect_count,
            "writer_error": self.writer_error,
            "receiver_error": self.receiver_error,
            "connections": [item.as_dict() for item in self.connections],
            "gaps": list(self.gaps),
            "artifact": self.artifact,
        }


@dataclass(frozen=True)
class CaptureResult:
    manifest_path: Path
    manifest: dict[str, Any]

    @property
    def complete(self) -> bool:
        return bool(self.manifest.get("complete"))


class ChannelWorker:
    """한 channel의 receiver와 gzip writer를 분리해 queue drop을 감사한다."""

    def __init__(
        self,
        *,
        capture_id: str,
        channel: Channel,
        config: CaptureConfig,
        output_path: Path,
        stop_event: threading.Event,
        connect_fn: ConnectFn = websocket_connect,
        rng: random.Random | None = None,
    ):
        self.capture_id = capture_id
        self.channel = channel
        self.config = config
        self.stop_event = stop_event
        self.connect_fn = connect_fn
        # Non-security reconnect jitter; no token/key material is generated.
        self.rng = rng or random.Random()  # noqa: S311
        self.state = WorkerState(channel=channel)
        self.frames: queue.Queue[RawFrame | object] = queue.Queue(
            maxsize=config.queue_max
        )
        self._writer_started = False
        self._receiver_started = False
        self._active_socket_lock = threading.Lock()
        self._active_websocket: Any | None = None
        self._receiver_done = threading.Event()
        try:
            self.writer = RawStreamWriter(
                output_path,
                capture_id=capture_id,
                channel=channel,
                expected_markets=set(config.markets),
                requested_orderbook_depth=config.orderbook_depth,
            )
            self._receiver = threading.Thread(
                target=self._receiver_loop,
                name=f"upbit-{channel}-receiver",
                daemon=True,
            )
            self._writer = threading.Thread(
                target=self._writer_loop,
                name=f"upbit-{channel}-writer",
                daemon=True,
            )
        except Exception:
            writer = getattr(self, "writer", None)
            if writer is not None:
                writer.abort(reason="ChannelWorker constructor failed")
            raise

    def start(self) -> None:
        if self._writer_started or self._receiver_started:
            raise RuntimeError(f"{self.channel} worker already started")
        try:
            self._writer.start()
            self._writer_started = True
            self._receiver.start()
            self._receiver_started = True
        except Exception:
            self._writer_started = (
                self._writer_started or self._writer.ident is not None
            )
            self._receiver_started = (
                self._receiver_started or self._receiver.ident is not None
            )
            self.stop_event.set()
            raise

    def join(self, timeout: float = 30.0) -> None:
        if timeout <= 0:
            raise ValueError("join timeout must be positive")
        deadline = time.monotonic() + timeout
        self.stop_event.set()
        if self._receiver_started:
            self._receiver.join(min(6.0, timeout / 2))
            if self._receiver.is_alive():
                forced_close = self._close_active_websocket()
                if forced_close:
                    self.state.receiver_error = (
                        "receiver required forced websocket close"
                    )
                self._receiver.join(max(0.0, deadline - time.monotonic()))
            if self._receiver.is_alive():
                self.state.receiver_error = "receiver did not stop before timeout"
                raise RuntimeError(f"{self.channel} receiver could not be stopped")
            if not self._receiver_done.is_set():
                raise RuntimeError(
                    f"{self.channel} receiver exited without finalizing"
                )
        elif self._writer_started:
            self._signal_writer_stop(deadline)

        if self._writer_started:
            self._writer.join(max(0.0, deadline - time.monotonic()))
            if self._writer.is_alive():
                self.state.writer_error = "writer did not drain before timeout"
                raise RuntimeError(f"{self.channel} writer could not be stopped")
        else:
            self.state.artifact = self.writer.abort(
                reason=f"{self.channel} worker never started"
            )

    def abort(self, *, reason: str) -> dict[str, Any]:
        """멈춘/unstarted worker의 raw writer를 incomplete 증거로 격리한다."""
        self.stop_event.set()
        self._close_active_websocket()
        writer_alive = self._writer_started and self._writer.is_alive()
        receiver_alive = self._receiver_started and self._receiver.is_alive()
        if writer_alive or receiver_alive:
            artifact = self.writer.artifact_metadata()
            artifact["abort_deferred_threads_alive"] = {
                "writer": writer_alive,
                "receiver": receiver_alive,
            }
            self.state.artifact = artifact
            return artifact
        artifact = self.writer.abort(reason=reason)
        self.state.artifact = artifact
        return artifact

    def force_shutdown(self, *, reason: str, timeout: float = 5.0) -> list[str]:
        """join 경계 실패 뒤에도 socket/thread/writer 정리를 재시도한다."""
        errors: list[str] = []
        deadline = time.monotonic() + max(timeout, 0.1)
        self.stop_event.set()
        self._close_active_websocket()
        if self._receiver_started and self._receiver.is_alive():
            self._receiver.join(max(0.0, deadline - time.monotonic()))
        if self._receiver_started and self._receiver.is_alive():
            errors.append("receiver still alive after forced shutdown")
        if self._writer_started and self._writer.is_alive():
            self._signal_writer_stop(deadline)
            self._writer.join(max(0.0, deadline - time.monotonic()))
        if self._writer_started and self._writer.is_alive():
            errors.append("writer still alive after forced shutdown")
        if not errors:
            self.abort(reason=reason)
        else:
            self.state.artifact = self.writer.artifact_metadata()
        return errors

    def _signal_writer_stop(self, deadline: float) -> None:
        while self._writer.is_alive() and time.monotonic() < deadline:
            try:
                self.frames.put(_SENTINEL, timeout=0.1)
                return
            except queue.Full:
                continue

    def _close_active_websocket(self) -> bool:
        with self._active_socket_lock:
            websocket = self._active_websocket
        close = getattr(websocket, "close", None)
        if not callable(close):
            return False
        try:
            close()
        except Exception as exc:  # pragma: no cover - defensive shutdown boundary
            self.state.receiver_error = (
                self.state.receiver_error
                or f"websocket close {type(exc).__name__}: {exc}"
            )
        return True

    def _writer_loop(self) -> None:
        try:
            while True:
                item = self.frames.get()
                try:
                    if item is _SENTINEL:
                        break
                    if not isinstance(item, RawFrame):
                        raise TypeError(
                            "microstructure writer queue contained "
                            f"{type(item).__name__}, expected RawFrame"
                        )
                    self.writer.write(item)
                    if self.writer.stats.fatal_error:
                        self.state.writer_error = self.writer.stats.fatal_error
                        self.stop_event.set()
                finally:
                    self.frames.task_done()
        except Exception as exc:  # pragma: no cover - defensive I/O boundary
            self.state.writer_error = f"{type(exc).__name__}: {exc}"
            self.stop_event.set()
        finally:
            try:
                self.state.artifact = self.writer.close()
            except Exception as exc:  # pragma: no cover - disk/fs boundary
                error = (
                    self.state.writer_error
                    or f"writer close {type(exc).__name__}: {exc}"
                )
                self.state.writer_error = error
                self.state.artifact = self.writer.abort(reason=error)
                self.stop_event.set()

    def _receiver_loop(self) -> None:
        ingress_seq = 0
        attempt = 0
        try:
            while not self.stop_event.is_set():
                attempt += 1
                connection_id = str(uuid.uuid4())
                subscription_id = str(uuid.uuid4())
                connection_record = ConnectionRecord(
                    connection_id=connection_id,
                    subscription_id=subscription_id,
                    attempt=attempt,
                    opened_at_ns=utc_now_ns(),
                )
                self.state.connections.append(connection_record)
                websocket: Any | None = None
                try:
                    with self.connect_fn(
                        self.config.endpoint,
                        open_timeout=10,
                        ping_interval=30,
                        ping_timeout=10,
                        close_timeout=5,
                        max_size=2 * 1024 * 1024,
                        max_queue=4096,
                        compression="deflate",
                        proxy=None,
                    ) as websocket:
                        with self._active_socket_lock:
                            self._active_websocket = websocket
                        try:
                            subscription = build_subscription(
                                self.channel,
                                self.config.markets,
                                ticket=subscription_id,
                                orderbook_depth=self.config.orderbook_depth,
                            )
                            websocket.send(
                                json.dumps(
                                    subscription,
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                )
                            )
                            connection_record.subscribed_at_ns = utc_now_ns()
                            logger.info(
                                "%s connected attempt=%d markets=%d",
                                self.channel,
                                attempt,
                                len(self.config.markets),
                            )

                            while not self.stop_event.is_set():
                                try:
                                    message = websocket.recv(timeout=0.5)
                                except TimeoutError:
                                    continue
                                received_at_ns = utc_now_ns()
                                received_monotonic_ns = time.monotonic_ns()
                                frame_kind: Literal["binary", "text"] = (
                                    "binary"
                                    if isinstance(message, bytes)
                                    else "text"
                                )
                                raw = (
                                    message
                                    if isinstance(message, bytes)
                                    else str(message).encode("utf-8")
                                )
                                ingress_seq += 1
                                if connection_record.first_ingress_seq is None:
                                    connection_record.first_ingress_seq = ingress_seq
                                connection_record.last_ingress_seq = ingress_seq
                                frame = RawFrame(
                                    raw=raw,
                                    received_at_ns=received_at_ns,
                                    received_monotonic_ns=received_monotonic_ns,
                                    connection_id=connection_id,
                                    subscription_id=subscription_id,
                                    ingress_seq=ingress_seq,
                                    frame_kind=frame_kind,
                                )
                                try:
                                    self.frames.put_nowait(frame)
                                except queue.Full:
                                    self.state.dropped_frames += 1
                                    self.state.receiver_error = (
                                        f"{self.channel} queue overflow at "
                                        f"ingress_seq={ingress_seq}"
                                    )
                                    self.state.gaps.append(
                                        {
                                            "kind": "queue_overflow",
                                            "at_ns": received_at_ns,
                                            "at": ns_to_iso(received_at_ns),
                                            "ingress_seq": ingress_seq,
                                        }
                                    )
                                    self.stop_event.set()
                                    break
                        finally:
                            connection_record.clean_stop = self.stop_event.is_set()
                except ConnectionClosed as exc:
                    connection_record.error = f"{type(exc).__name__}: {exc}"
                except Exception as exc:
                    connection_record.error = f"{type(exc).__name__}: {exc}"
                finally:
                    with self._active_socket_lock:
                        if self._active_websocket is websocket:
                            self._active_websocket = None
                    connection_record.closed_at_ns = utc_now_ns()

                if connection_record.error:
                    self.state.gaps.append(
                        {
                            "kind": "connection_gap",
                            "started_at_ns": connection_record.closed_at_ns,
                            "started_at": ns_to_iso(connection_record.closed_at_ns),
                            "after_connection_id": connection_id,
                            "reason": connection_record.error,
                            "retry_scheduled": not self.stop_event.is_set(),
                        }
                    )
                if self.stop_event.is_set():
                    break

                self.state.reconnect_count += 1
                if not connection_record.error:
                    self.state.gaps.append(
                        {
                            "kind": "connection_gap",
                            "started_at_ns": connection_record.closed_at_ns,
                            "started_at": ns_to_iso(
                                connection_record.closed_at_ns
                            ),
                            "after_connection_id": connection_id,
                            "reason": "connection closed",
                            "retry_scheduled": True,
                        }
                    )
                delay = min(30.0, 2.0 ** min(attempt - 1, 5))
                delay += self.rng.uniform(0, min(1.0, delay * 0.2))
                logger.warning(
                    "%s disconnected (%s); reconnect in %.2fs",
                    self.channel,
                    connection_record.error,
                    delay,
                )
                self.stop_event.wait(delay)
        except Exception as exc:  # pragma: no cover - defensive thread boundary
            self.state.receiver_error = f"{type(exc).__name__}: {exc}"
            self.stop_event.set()
        finally:
            # Queue가 가득 차도 writer가 살아 있으면 drain을 기다린다. 반대로 disk
            # error 등으로 writer가 이미 죽었으면 무한 block하지 않는다.
            while self._writer.is_alive():
                try:
                    self.frames.put(_SENTINEL, timeout=0.1)
                    break
                except queue.Full:
                    continue
            self._receiver_done.set()


def snapshot_path_for(
    *,
    snapshot_root: Path,
    asof: date,
    slot: str,
) -> Path:
    return snapshot_root / asof.isoformat() / f"{slot}_r1.json"


def read_snapshot_cutoff(
    path: str | Path,
    *,
    expected_asof: date,
    expected_slot: str,
) -> tuple[datetime, dict[str, Any]]:
    """atomic R1 snapshot의 실제 decision 완료시각을 cutoff로 읽는다."""
    snapshot_path = Path(path)
    try:
        identity_before = file_identity(
            snapshot_path,
            root=snapshot_path.parent,
        )
        document = strict_json_object(snapshot_path)
        identity_after = file_identity(
            snapshot_path,
            root=snapshot_path.parent,
        )
    except (
        ArtifactSourceChangedError,
        ArtifactValidationError,
        OSError,
    ) as exc:
        raise ValueError(f"invalid snapshot JSON: {exc}") from exc
    if (
        not identity_before.get("exists")
        or identity_before != identity_after
        or not isinstance(identity_after.get("sha256"), str)
    ):
        raise ValueError(
            f"snapshot changed while reading: {snapshot_path}"
        )

    request = document.get("request")
    request = request if isinstance(request, dict) else {}
    asof_value = str(document.get("asof") or request.get("asof"))
    slot_value = str(document.get("slot") or request.get("slot"))
    if asof_value != expected_asof.isoformat():
        raise ValueError(
            f"snapshot asof mismatch: {asof_value!r} != {expected_asof.isoformat()!r}"
        )
    if slot_value != expected_slot:
        raise ValueError(
            f"snapshot slot mismatch: {slot_value!r} != {expected_slot!r}"
        )
    if document.get("snapshot_schema") not in {
        "recommend_snapshot.v1",
        "recommend_snapshot.v2",
    }:
        raise ValueError("snapshot_schema must be a supported recommend snapshot")
    snapshot_id = document.get("snapshot_id")
    payload_sha256 = document.get("payload_sha256")
    if (
        not isinstance(payload_sha256, str)
        or len(payload_sha256) != 64
        or any(char not in "0123456789abcdef" for char in payload_sha256.lower())
    ):
        raise ValueError("snapshot has no valid payload_sha256")
    digest_payload = {
        key: value
        for key, value in document.items()
        if key
        not in {"created_at", "snapshot_id", "payload_sha256", "snapshot_path"}
    }
    canonical_payload = json.dumps(
        digest_payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    expected_payload_sha256 = hashlib.sha256(canonical_payload).hexdigest()
    if payload_sha256 != expected_payload_sha256:
        raise ValueError("snapshot payload_sha256 mismatch")
    expected_snapshot_id = f"recommend-{expected_payload_sha256[:20]}"
    if snapshot_id != expected_snapshot_id:
        raise ValueError("snapshot_id does not match payload_sha256")
    started_value = document.get("decision_started_at")
    completed_value = document.get("decision_completed_at")
    created_value = document.get("created_at")
    if (
        not isinstance(started_value, str)
        or not isinstance(completed_value, str)
        or not isinstance(created_value, str)
    ):
        raise ValueError(
            "snapshot requires decision_started_at, decision_completed_at, created_at"
        )
    decision_started_at = parse_utc_datetime(started_value)
    cutoff = parse_utc_datetime(completed_value)
    created_at = parse_utc_datetime(created_value)
    if decision_started_at > cutoff:
        raise ValueError("snapshot decision_started_at is after decision_completed_at")
    if created_at != cutoff:
        raise ValueError("snapshot created_at must equal decision_completed_at")
    return cutoff, {
        "path": str(snapshot_path),
        "file_sha256": identity_after["sha256"],
        "snapshot_id": snapshot_id,
        "payload_sha256": payload_sha256,
        "asof": asof_value,
        "slot": slot_value,
        "created_at": document.get("created_at"),
        "decision_started_at": document.get("decision_started_at"),
        "decision_completed_at": document.get("decision_completed_at"),
    }


def _cutoff_watermark(
    artifact_path: str | Path,
    cutoff_at_ns: int,
) -> dict[str, Any]:
    included = 0
    received_after_cutoff = 0
    exchange_after_cutoff = 0
    max_ingress_seq: int | None = None
    market_watermarks: dict[str, dict[str, int]] = {}
    included_markets: set[str] = set()
    snapshot_markets: set[str] = set()
    realtime_markets: set[str] = set()
    snapshot_watermarks: dict[str, dict[str, int]] = {}
    snapshot_event_count = 0
    realtime_event_count = 0
    orderbook_unit_counts: dict[int, int] = {}

    for record in iter_raw_records(artifact_path):
        if record.get("record_type") != "event":
            continue
        try:
            event_at_ms = int(record["event_at_ms"])
            received_at_ns = int(record["received_at_ns"])
            ingress_seq = int(record["ingress_seq"])
        except (KeyError, TypeError, ValueError):
            continue
        if received_at_ns > cutoff_at_ns:
            received_after_cutoff += 1
        if event_at_ms * 1_000_000 > cutoff_at_ns:
            exchange_after_cutoff += 1
        if not event_available_at_cutoff(record, cutoff_at_ns):
            continue
        included += 1
        stream_type = str(record.get("stream_type", ""))
        market = str(record["market"])
        unit_count = record.get("orderbook_unit_count")
        if unit_count is not None:
            try:
                unit_count_int = int(unit_count)
            except (TypeError, ValueError):
                unit_count_int = 0
            if unit_count_int > 0:
                orderbook_unit_counts[unit_count_int] = (
                    orderbook_unit_counts.get(unit_count_int, 0) + 1
                )
        included_markets.add(market)
        if stream_type == "SNAPSHOT":
            snapshot_event_count += 1
            snapshot_markets.add(market)
            candidate_snapshot = {
                "event_at_ms": event_at_ms,
                "received_at_ns": received_at_ns,
                "ingress_seq": ingress_seq,
            }
            if market not in snapshot_watermarks:
                snapshot_watermarks[market] = candidate_snapshot
        elif stream_type == "REALTIME":
            realtime_event_count += 1
            realtime_markets.add(market)
        max_ingress_seq = (
            ingress_seq
            if max_ingress_seq is None
            else max(max_ingress_seq, ingress_seq)
        )
        previous = market_watermarks.get(market)
        candidate = {
            "event_at_ms": event_at_ms,
            "received_at_ns": received_at_ns,
            "ingress_seq": ingress_seq,
        }
        if previous is None or ingress_seq > previous["ingress_seq"]:
            market_watermarks[market] = candidate

    observed_counts = sorted(orderbook_unit_counts)
    return {
        "included_event_count": included,
        "received_after_cutoff_count": received_after_cutoff,
        "exchange_after_cutoff_count": exchange_after_cutoff,
        "max_ingress_seq": max_ingress_seq,
        "included_markets": sorted(included_markets),
        "included_market_count": len(included_markets),
        "snapshot_event_count": snapshot_event_count,
        "snapshot_markets": sorted(snapshot_markets),
        "snapshot_market_count": len(snapshot_markets),
        "snapshot_watermarks": snapshot_watermarks,
        "realtime_event_count": realtime_event_count,
        "realtime_markets": sorted(realtime_markets),
        "realtime_market_count": len(realtime_markets),
        "market_watermarks": market_watermarks,
        "observed_orderbook_unit_count_min": (
            observed_counts[0] if observed_counts else None
        ),
        "observed_orderbook_unit_count_max": (
            observed_counts[-1] if observed_counts else None
        ),
        "observed_orderbook_unit_count_histogram": {
            str(count): orderbook_unit_counts[count] for count in observed_counts
        },
    }


def _verify_raw_artifact_metadata(artifact: dict[str, Any]) -> None:
    """Bind cutoff statistics to the exact finalized compressed bytes."""
    path_value = artifact.get("path")
    expected_sha256 = artifact.get("sha256")
    expected_size = artifact.get("size_bytes")
    if (
        not isinstance(path_value, str)
        or not isinstance(expected_sha256, str)
        or isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size < 0
    ):
        raise ValueError("raw artifact metadata is incomplete")
    path = Path(path_value)
    actual_sha256 = file_sha256(path)
    actual_size = path.lstat().st_size
    if actual_sha256 != expected_sha256 or actual_size != expected_size:
        raise ValueError(
            f"raw artifact changed before cutoff manifest: {path}"
        )


def _git_output(*args: str) -> tuple[str | None, str | None]:
    try:
        # Fixed git binary and internal call-site arguments only.
        result = subprocess.run(  # noqa: S603
            ["/usr/bin/git", "-C", str(PROJECT_ROOT), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if result.returncode != 0:
        error = result.stderr.strip() or f"git exit={result.returncode}"
        return None, error
    return result.stdout.strip(), None


def _source_metadata() -> dict[str, Any]:
    """현재 source/runtime provenance를 한 시점에 관측한다."""
    files = (
        PROJECT_ROOT / "data" / "collector_upbit_microstructure.py",
        PROJECT_ROOT / "data" / "upbit_microstructure.py",
    )
    file_documents: list[dict[str, Any]] = []
    for path in files:
        try:
            digest = file_sha256(path)
            error = None
        except (OSError, ValueError) as exc:
            digest = None
            error = f"{type(exc).__name__}: {exc}"
        file_documents.append(
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": digest,
                "error": error,
            }
        )
    commit, commit_error = _git_output("rev-parse", "HEAD")
    status, status_error = _git_output(
        "status", "--porcelain", "--untracked-files=normal"
    )
    try:
        websockets_version = importlib.metadata.version("websockets")
    except importlib.metadata.PackageNotFoundError:
        websockets_version = None
    captured_at_ns = utc_now_ns()
    return {
        "captured_at_ns": captured_at_ns,
        "captured_at": ns_to_iso(captured_at_ns),
        "files": file_documents,
        "git": {
            "commit": commit,
            "dirty": bool(status) if status is not None else None,
            "commit_error": commit_error,
            "status_error": status_error,
        },
        "runtime": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "websockets_version": websockets_version,
        },
    }


def _source_provenance_complete(document: dict[str, Any]) -> bool:
    git = document.get("git") or {}
    runtime = document.get("runtime") or {}
    files = document.get("files") or []
    return bool(
        files
        and all(item.get("sha256") and not item.get("error") for item in files)
        and git.get("commit")
        and isinstance(git.get("dirty"), bool)
        and not git.get("commit_error")
        and not git.get("status_error")
        and runtime.get("python_version")
        and runtime.get("python_implementation")
        and runtime.get("websockets_version")
    )


def _source_unchanged(
    at_start: dict[str, Any],
    at_end: dict[str, Any],
) -> bool:
    def file_hashes(document: dict[str, Any]) -> dict[str, str | None]:
        return {
            str(item.get("path")): item.get("sha256")
            for item in document.get("files") or []
        }

    return bool(
        _source_provenance_complete(at_start)
        and _source_provenance_complete(at_end)
        and file_hashes(at_start) == file_hashes(at_end)
        and (at_start.get("git") or {}).get("commit")
        == (at_end.get("git") or {}).get("commit")
        and (at_start.get("git") or {}).get("dirty")
        == (at_end.get("git") or {}).get("dirty")
        and at_start.get("runtime") == at_end.get("runtime")
    )


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


@contextmanager
def _capture_output_ownership(output_root: Path) -> Iterator[None]:
    """cleanup부터 manifest 확정까지 output root의 단일 writer를 보장한다."""
    output_root = Path(output_root)
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        root_before = output_root.lstat()
    except OSError as exc:
        raise CaptureOwnershipError(
            f"capture output root is unavailable: {output_root}"
        ) from exc
    if not stat.S_ISDIR(root_before.st_mode):
        raise CaptureOwnershipError(
            f"capture output root is not a real directory: {output_root}"
        )

    lock_path = output_root / _CAPTURE_LOCK_NAME
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        lock_fd = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise CaptureOwnershipError(
            f"capture ownership lock cannot be opened safely: {lock_path}"
        ) from exc

    acquired = False
    try:
        try:
            root_after_open = output_root.lstat()
            lock_fd_stat = os.fstat(lock_fd)
            lock_path_stat = lock_path.lstat()
        except OSError as exc:
            raise CaptureOwnershipError(
                f"capture ownership changed while opening: {lock_path}"
            ) from exc
        if (
            not stat.S_ISDIR(root_after_open.st_mode)
            or not _same_inode(root_before, root_after_open)
        ):
            raise CaptureOwnershipError(
                f"capture output root changed while locking: {output_root}"
            )
        if (
            not stat.S_ISREG(lock_fd_stat.st_mode)
            or lock_fd_stat.st_nlink != 1
            or lock_fd_stat.st_uid != os.geteuid()
            or not _same_inode(lock_fd_stat, lock_path_stat)
        ):
            raise CaptureOwnershipError(
                f"capture ownership lock is not a private regular file: {lock_path}"
            )

        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CaptureOwnershipError(
                f"another capture already owns output root: {output_root}"
            ) from exc
        except OSError as exc:
            raise CaptureOwnershipError(
                f"capture ownership lock failed: {lock_path}"
            ) from exc
        acquired = True

        try:
            root_after_lock = output_root.lstat()
            lock_after = lock_path.lstat()
            lock_fd_after = os.fstat(lock_fd)
        except OSError as exc:
            raise CaptureOwnershipError(
                f"capture ownership changed after locking: {lock_path}"
            ) from exc
        if (
            not stat.S_ISDIR(root_after_lock.st_mode)
            or not _same_inode(root_before, root_after_lock)
            or not stat.S_ISREG(lock_after.st_mode)
            or lock_after.st_nlink != 1
            or lock_after.st_uid != os.geteuid()
            or not _same_inode(lock_fd_after, lock_after)
        ):
            raise CaptureOwnershipError(
                f"capture ownership is unstable: {lock_path}"
            )
        yield
    finally:
        try:
            if acquired:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)


def _quarantine_orphan_partials(
    output_root: Path,
    *,
    observed_at_ns: int,
    stale_after_seconds: float,
) -> dict[str, Any]:
    """이전 crash의 오래된 partial을 보존명으로 옮기고 marker를 남긴다."""
    result: dict[str, Any] = {
        "scanned_at_ns": observed_at_ns,
        "scanned_at": ns_to_iso(observed_at_ns),
        "stale_after_seconds": stale_after_seconds,
        "records": [],
        "scan_error": None,
    }
    if not output_root.exists():
        return result
    try:
        partials = sorted(output_root.rglob("*.partial"))
    except OSError as exc:
        result["scan_error"] = f"{type(exc).__name__}: {exc}"
        return result

    for partial in partials:
        record: dict[str, Any] = {
            "original_path": str(partial),
            "status": "observed",
        }
        try:
            age_seconds = max(
                0.0,
                observed_at_ns / 1_000_000_000 - partial.stat().st_mtime,
            )
            record["age_seconds"] = age_seconds
            if age_seconds < stale_after_seconds:
                record["status"] = "recent_partial_preserved"
                result["records"].append(record)
                continue
            orphan_path = partial.with_name(
                f"{partial.name}.orphan-{observed_at_ns}-{uuid.uuid4().hex[:8]}"
            )
            os.replace(partial, orphan_path)
            marker_path = orphan_path.with_name(f"{orphan_path.name}.json")
            atomic_write_json(
                marker_path,
                {
                    "schema_version": MANIFEST_SCHEMA_VERSION,
                    "status": "incomplete_orphan_quarantined",
                    "reason": "stale .partial found at collector startup",
                    "original_path": str(partial),
                    "preserved_path": str(orphan_path),
                    "observed_at_ns": observed_at_ns,
                    "observed_at": ns_to_iso(observed_at_ns),
                    "age_seconds": age_seconds,
                },
            )
            record.update(
                {
                    "status": "quarantined",
                    "preserved_path": str(orphan_path),
                    "marker_path": str(marker_path),
                }
            )
        except Exception as exc:
            record.update(
                {
                    "status": "quarantine_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        result["records"].append(record)
    return result


def _capture_directory(
    output_root: Path,
    *,
    asof: date,
    capture_id: str,
) -> Path:
    return (
        output_root
        / f"{asof.year:04d}"
        / f"{asof.month:02d}"
        / f"{asof.day:02d}"
        / capture_id
    )


def _snapshot_feature_window_start(asof: date) -> datetime:
    return datetime.combine(asof, datetime_time(9, 0), tzinfo=KST).astimezone(
        timezone.utc
    )


def _worker_transport_ok(
    state: WorkerState,
    *,
    expected_markets: set[str],
    cutoff_summary: dict[str, Any] | None,
) -> bool:
    artifact = state.artifact or {}
    if state.fatal_error or state.dropped_frames or state.reconnect_count:
        return False
    if any(connection.error for connection in state.connections):
        return False
    if artifact.get("parse_error_count") or artifact.get("server_error_count"):
        return False
    if not cutoff_summary or not cutoff_summary.get("included_event_count"):
        return False
    if not cutoff_summary.get("realtime_event_count"):
        return False
    if state.channel == "orderbook":
        snapshot_markets = set(cutoff_summary.get("snapshot_markets") or [])
        if snapshot_markets != expected_markets:
            return False
    return True


def _clock_sync_ok(*statuses: dict[str, Any]) -> bool:
    """received_at과 exchange time을 함께 쓰므로 wall clock은 fail-closed."""
    return all(status.get("ntp_synchronized") is True for status in statuses)


def _warmup_ready(
    *,
    stream_documents: dict[str, Any],
    expected_markets: set[str],
    readiness_deadline_ns: int,
) -> bool:
    """두 구독과 전 종목 최초 book snapshot이 분석 시작 전에 준비됐는지."""
    for channel in ("trade", "orderbook"):
        connections = stream_documents[channel]["connections"]
        subscribed = [
            item.get("subscribed_at_ns")
            for item in connections
            if item.get("subscribed_at_ns") is not None
        ]
        if not subscribed or min(subscribed) > readiness_deadline_ns:
            return False

    cutoff_summary = stream_documents["orderbook"].get("at_feature_cutoff") or {}
    snapshot_watermarks = cutoff_summary.get("snapshot_watermarks") or {}
    if set(snapshot_watermarks) != expected_markets:
        return False
    for watermark in snapshot_watermarks.values():
        try:
            event_at_ns = int(watermark["event_at_ms"]) * 1_000_000
            received_at_ns = int(watermark["received_at_ns"])
        except (KeyError, TypeError, ValueError):
            return False
        if (
            event_at_ns > readiness_deadline_ns
            or received_at_ns > readiness_deadline_ns
        ):
            return False
    return True


def _run_capture_owned(
    config: CaptureConfig,
    *,
    connect_fn: ConnectFn = websocket_connect,
    install_signal_handlers: bool = True,
) -> CaptureResult:
    """소유권 확보 후 두 공개 stream과 causal cutoff manifest를 남긴다."""
    started_at_ns = utc_now_ns()
    started_at = datetime.fromtimestamp(
        started_at_ns / 1_000_000_000, timezone.utc
    )
    source_at_start = _source_metadata()
    asof = config.asof or started_at.astimezone(KST).date()
    capture_id = (
        f"upbit-{asof.strftime('%Y%m%d')}-"
        f"{started_at.strftime('%H%M%S')}-{uuid.uuid4().hex[:8]}"
    )
    orphan_recovery = _quarantine_orphan_partials(
        config.output_root,
        observed_at_ns=started_at_ns,
        stale_after_seconds=max(60.0, config.max_wait_seconds + 60.0),
    )
    capture_dir = _capture_directory(
        config.output_root, asof=asof, capture_id=capture_id
    )
    capture_dir.mkdir(parents=True, exist_ok=False)
    expected_markets = set(config.markets)
    try:
        clock_at_start = clock_sync_status()
    except Exception as exc:  # pragma: no cover - environment boundary
        clock_at_start = {
            "ntp_synchronized": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    stop_event = threading.Event()
    stop_reason = "unknown"
    signal_received: int | None = None
    previous_handlers: dict[signal.Signals, Any] = {}
    handlers_restored = True
    workers: dict[str, ChannelWorker] = {}
    started_channels: list[str] = []
    lifecycle_errors: list[str] = []
    primary_error: BaseException | None = None
    lifecycle_phase = "initialize"

    def handle_signal(signum: int, _frame: Any) -> None:
        nonlocal signal_received, stop_reason
        signal_received = signum
        stop_reason = f"signal_{signum}"
        stop_event.set()

    cutoff_at: datetime | None = None
    cutoff_source: dict[str, Any] = {}
    feature_window_start_at = started_at
    if config.duration_seconds is not None:
        cutoff_at = started_at + timedelta(seconds=config.duration_seconds)
        cutoff_source = {
            "kind": "duration_canary",
            "duration_seconds": config.duration_seconds,
        }
    elif config.feature_cutoff_at is not None:
        cutoff_at = config.feature_cutoff_at.astimezone(timezone.utc)
        cutoff_source = {"kind": "explicit"}
    else:
        if config.cutoff_slot is None:
            raise RuntimeError(
                "validated collector config has no cutoff source"
            )
        feature_window_start_at = _snapshot_feature_window_start(asof)
        cutoff_source = {
            "kind": "recommend_snapshot",
            "slot": config.cutoff_slot,
            "path": str(
                snapshot_path_for(
                    snapshot_root=config.snapshot_root,
                    asof=asof,
                    slot=config.cutoff_slot,
                )
            ),
        }

    deadline_monotonic = time.monotonic() + config.max_wait_seconds
    snapshot_error: str | None = None
    try:
        lifecycle_phase = "install_signal_handlers"
        if (
            install_signal_handlers
            and threading.current_thread() is threading.main_thread()
        ):
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, handle_signal)

        lifecycle_phase = "construct_workers"
        for channel in ("trade", "orderbook"):
            workers[channel] = ChannelWorker(
                capture_id=capture_id,
                channel=channel,
                config=config,
                output_path=capture_dir / f"{channel}.jsonl.gz",
                stop_event=stop_event,
                connect_fn=connect_fn,
            )

        lifecycle_phase = "start_workers"
        for channel in ("trade", "orderbook"):
            workers[channel].start()
            started_channels.append(channel)

        lifecycle_phase = "capture"
        while not stop_event.is_set():
            now_ns = utc_now_ns()
            if cutoff_at is not None and now_ns >= datetime_to_ns(cutoff_at):
                stop_reason = "feature_cutoff_reached"
                stop_event.set()
                break
            if config.cutoff_slot is not None:
                path = Path(cutoff_source["path"])
                if path.exists():
                    try:
                        cutoff_at, snapshot_meta = read_snapshot_cutoff(
                            path,
                            expected_asof=asof,
                            expected_slot=config.cutoff_slot,
                        )
                    except (OSError, ValueError, json.JSONDecodeError) as exc:
                        snapshot_error = f"{type(exc).__name__}: {exc}"
                    else:
                        cutoff_source.update(snapshot_meta)
                        if datetime_to_ns(cutoff_at) <= utc_now_ns():
                            stop_reason = "recommend_snapshot_cutoff_observed"
                            stop_event.set()
                            break
            if time.monotonic() >= deadline_monotonic:
                stop_reason = "cutoff_wait_timeout"
                stop_event.set()
                break
            time.sleep(0.1)
    except BaseException as exc:
        primary_error = exc
        if stop_reason == "unknown":
            stop_reason = f"{lifecycle_phase}_failed"
        lifecycle_errors.append(
            f"{lifecycle_phase}: {type(exc).__name__}: {exc}"
        )
    finally:
        stop_event.set()
        lifecycle_phase = "shutdown"
        for channel, worker in workers.items():
            try:
                worker.join()
            except Exception as exc:  # pragma: no cover - fatal shutdown boundary
                error = f"{channel} join: {type(exc).__name__}: {exc}"
                lifecycle_errors.append(error)
                fallback_errors = worker.force_shutdown(reason=error)
                lifecycle_errors.extend(
                    f"{channel} forced shutdown: {item}"
                    for item in fallback_errors
                )
        for signum, handler in previous_handlers.items():
            try:
                signal.signal(signum, handler)
            except Exception as exc:  # pragma: no cover - process boundary
                handlers_restored = False
                lifecycle_errors.append(
                    f"restore signal {signum}: {type(exc).__name__}: {exc}"
                )

    ended_at_ns = utc_now_ns()
    try:
        clock_at_end = clock_sync_status()
    except Exception as exc:  # pragma: no cover - environment boundary
        clock_at_end = {
            "ntp_synchronized": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    source_at_end = _source_metadata()

    cutoff_at_ns = datetime_to_ns(cutoff_at) if cutoff_at is not None else None
    stream_documents: dict[str, Any] = {}
    for channel in ("trade", "orderbook"):
        channel_worker = workers.get(channel)
        if channel_worker is None:
            state_document = WorkerState(
                channel=cast(Channel, channel)
            ).as_dict()
            state_document["lifecycle_status"] = "constructor_not_completed"
            stream_documents[channel] = state_document
            continue
        state_document = channel_worker.state.as_dict()
        state_document["lifecycle_status"] = (
            "started" if channel in started_channels else "constructed_not_started"
        )
        artifact = channel_worker.state.artifact
        if cutoff_at_ns is not None and artifact and artifact.get("exists"):
            try:
                state_document["at_feature_cutoff"] = _cutoff_watermark(
                    artifact["path"], cutoff_at_ns
                )
                _verify_raw_artifact_metadata(artifact)
            except Exception as exc:
                state_document["at_feature_cutoff"] = None
                state_document["watermark_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
        else:
            state_document["at_feature_cutoff"] = None
        stream_documents[channel] = state_document

    source_provenance_complete = bool(
        _source_provenance_complete(source_at_start)
        and _source_provenance_complete(source_at_end)
    )
    source_unchanged = _source_unchanged(source_at_start, source_at_end)
    universe_timing_valid = bool(
        config.universe_observed_at_ns is not None
        and config.universe_observed_at_ns <= started_at_ns
        and (
            config.universe_source == "explicit"
            or (
                config.universe_fetch_started_at_ns is not None
                and config.universe_fetch_started_at_ns
                <= config.universe_observed_at_ns
            )
        )
    )
    orphan_recovery_ok = bool(
        orphan_recovery.get("scan_error") is None
        and all(
            item.get("status") != "quarantine_failed"
            for item in orphan_recovery.get("records") or []
        )
    )
    lifecycle_ok = bool(
        not lifecycle_errors
        and handlers_restored
        and set(workers) == {"trade", "orderbook"}
        and set(started_channels) == {"trade", "orderbook"}
    )
    clock_sync_ok = _clock_sync_ok(clock_at_start, clock_at_end)
    transport_complete = (
        lifecycle_ok
        and cutoff_at is not None
        and stop_reason
        in {"feature_cutoff_reached", "recommend_snapshot_cutoff_observed"}
        and clock_sync_ok
        and source_provenance_complete
        and source_unchanged
        and universe_timing_valid
        and orphan_recovery_ok
        and all(
            _worker_transport_ok(
                worker.state,
                expected_markets=expected_markets,
                cutoff_summary=stream_documents[channel][
                    "at_feature_cutoff"
                ],
            )
            for channel, worker in workers.items()
        )
    )
    required_start_at = feature_window_start_at - timedelta(
        seconds=config.required_warmup_seconds
    )
    required_start_at_ns = datetime_to_ns(required_start_at)
    # duration/explicit canary with zero warmup proves the wire contract only;
    # daily research capture requires readiness by the pre-window warmup boundary.
    readiness_deadline_ns = (
        required_start_at_ns
        if config.required_warmup_seconds > 0
        else cutoff_at_ns
    )
    warmup_ready = bool(
        cutoff_at_ns is not None
        and readiness_deadline_ns is not None
        and _warmup_ready(
            stream_documents=stream_documents,
            expected_markets=expected_markets,
            readiness_deadline_ns=readiness_deadline_ns,
        )
    )
    causal_window_complete = (
        cutoff_at is not None
        and started_at <= required_start_at
        and cutoff_at >= feature_window_start_at
        and warmup_ready
        and clock_sync_ok
        and source_provenance_complete
        and source_unchanged
        and universe_timing_valid
    )
    complete = (
        transport_complete
        and causal_window_complete
        and lifecycle_ok
        and orphan_recovery_ok
    )
    orderbook_artifact = (
        stream_documents.get("orderbook", {}).get("artifact") or {}
    )

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "capture_id": capture_id,
        "source": "upbit",
        "endpoint": config.endpoint,
        "public_quotation_only": True,
        "uses_api_key": False,
        "places_orders": False,
        "asof": asof.isoformat(),
        "started_at_ns": started_at_ns,
        "started_at": ns_to_iso(started_at_ns),
        "ended_at_ns": ended_at_ns,
        "ended_at": ns_to_iso(ended_at_ns),
        "stop_reason": stop_reason,
        "signal_received": signal_received,
        "feature_window_start_at": feature_window_start_at.isoformat(),
        "required_warmup_seconds": config.required_warmup_seconds,
        "required_capture_start_at": required_start_at.isoformat(),
        "feature_cutoff_at_ns": cutoff_at_ns,
        "feature_cutoff_at": cutoff_at.isoformat() if cutoff_at else None,
        "cutoff_source": cutoff_source,
        "snapshot_error": snapshot_error,
        "universe": {
            "source": config.universe_source,
            "fetch_started_at_ns": config.universe_fetch_started_at_ns,
            "fetch_started_at": ns_to_iso(config.universe_fetch_started_at_ns),
            "observed_at_ns": config.universe_observed_at_ns,
            "observed_at": ns_to_iso(config.universe_observed_at_ns),
            "capture_started_after_observation": universe_timing_valid,
            "markets": list(config.markets),
            "count": len(config.markets),
            "sha256": universe_sha256(config.markets),
            "frozen_for_capture": True,
        },
        "subscription": {
            "trade": build_subscription(
                "trade",
                config.markets,
                ticket="<per-connection-uuid>",
                orderbook_depth=config.orderbook_depth,
            )[1:],
            "orderbook": build_subscription(
                "orderbook",
                config.markets,
                ticket="<per-connection-uuid>",
                orderbook_depth=config.orderbook_depth,
            )[1:],
            "orderbook_depth": {
                "requested": config.orderbook_depth,
                "observed_unit_count_min": orderbook_artifact.get(
                    "observed_orderbook_unit_count_min"
                ),
                "observed_unit_count_max": orderbook_artifact.get(
                    "observed_orderbook_unit_count_max"
                ),
                "thin_book_event_count": orderbook_artifact.get(
                    "thin_book_event_count", 0
                ),
                "thin_book_is_schema_error": False,
            },
            "orderbook_level": 0,
            "format": "DEFAULT",
        },
        "streams": stream_documents,
        "lifecycle": {
            "phase": lifecycle_phase,
            "errors": lifecycle_errors,
            "primary_error": (
                f"{type(primary_error).__name__}: {primary_error}"
                if primary_error is not None
                else None
            ),
            "signal_handlers_installed": sorted(previous_handlers),
            "signal_handlers_restored": handlers_restored,
            "workers_constructed": sorted(workers),
            "workers_started": sorted(started_channels),
            "ok": lifecycle_ok,
        },
        "orphan_recovery": orphan_recovery,
        "clock": {
            "at_start": clock_at_start,
            "at_end": clock_at_end,
        },
        "source_code": {
            "at_start": source_at_start,
            "at_end": source_at_end,
            "provenance_complete": source_provenance_complete,
            "unchanged_during_capture": source_unchanged,
        },
        "quality": {
            "transport_complete": transport_complete,
            "causal_window_complete": causal_window_complete,
            "clock_sync_ok": clock_sync_ok,
            "warmup_ready": warmup_ready,
            "lifecycle_ok": lifecycle_ok,
            "universe_timing_valid": universe_timing_valid,
            "source_provenance_complete": source_provenance_complete,
            "source_unchanged_during_capture": source_unchanged,
            "orphan_recovery_ok": orphan_recovery_ok,
            "complete": complete,
            "late_event_rule": (
                "include only when event_at_ms*1e6 <= feature_cutoff_at_ns "
                "AND received_at_ns <= feature_cutoff_at_ns"
            ),
            "reconnect_rule": (
                "any reconnect makes this capture ineligible because public "
                "orderbook history cannot be replayed"
            ),
        },
        "complete": complete,
    }
    manifest_path = capture_dir / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    if primary_error is not None or lifecycle_errors:
        detail = "; ".join(lifecycle_errors) or "capture lifecycle failed"
        raise RuntimeError(
            f"{detail}; incomplete manifest={manifest_path}"
        ) from primary_error
    return CaptureResult(manifest_path=manifest_path, manifest=manifest)


def run_capture(
    config: CaptureConfig,
    *,
    connect_fn: ConnectFn = websocket_connect,
    install_signal_handlers: bool = True,
) -> CaptureResult:
    """output root를 독점해 cleanup과 active writer 사이의 경합을 막는다."""
    with _capture_output_ownership(config.output_root):
        return _run_capture_owned(
            config,
            connect_fn=connect_fn,
            install_signal_handlers=install_signal_handlers,
        )


def _resolve_markets(
    value: str | None,
) -> tuple[tuple[str, ...], str, int | None, int]:
    if value is not None:
        markets = normalise_markets(
            [item for item in value.split(",") if item.strip()]
        )
        observed_at_ns = utc_now_ns()
        return tuple(markets), "explicit", None, observed_at_ns

    fetch_started_at_ns = utc_now_ns()
    try:
        # Live-universe discovery is the only path that needs pyupbit/pandas.
        # Keep explicit-market canaries and pure collector imports lightweight.
        from data.collector_d1 import get_krw_markets

        markets = get_krw_markets()
    finally:
        observed_at_ns = utc_now_ns()
    normalised = normalise_markets(markets)
    return (
        tuple(normalised),
        "live",
        fetch_started_at_ns,
        observed_at_ns,
    )


def _parse_markets(value: str | None) -> tuple[str, ...]:
    """호환용 market-only parser. main은 timing/source를 보존하는 resolver를 쓴다."""
    markets, _source, _fetch_started, _observed = _resolve_markets(value)
    return markets


def _parse_asof(value: str | None) -> date:
    if value is None:
        return datetime.now(KST).date()
    return date.fromisoformat(value)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upbit KRW trade/orderbook record-only collector"
    )
    parser.add_argument(
        "--markets",
        help="comma-separated KRW markets (default: current live KRW universe)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--endpoint",
        default=PUBLIC_WEBSOCKET_ENDPOINT,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--orderbook-depth",
        type=int,
        choices=SUPPORTED_ORDERBOOK_DEPTHS,
        default=30,
        help="initial placeholder; raw wall/impact research uses full depth by default",
    )
    parser.add_argument("--queue-max", type=int, default=DEFAULT_QUEUE_MAX)
    parser.add_argument(
        "--max-wait-seconds",
        type=float,
        default=DEFAULT_MAX_WAIT_SECONDS,
    )
    parser.add_argument(
        "--required-warmup-seconds",
        type=float,
        default=DEFAULT_REQUIRED_WARMUP_SECONDS,
        help="initial placeholder; 600s means capture must start by 08:50 for 09:00 flow",
    )
    parser.add_argument("--asof", help="KST decision date YYYY-MM-DD")
    parser.add_argument(
        "--snapshot-root",
        type=Path,
        default=DEFAULT_SNAPSHOT_ROOT,
    )
    cutoff_group = parser.add_mutually_exclusive_group(required=True)
    cutoff_group.add_argument(
        "--duration-seconds",
        type=float,
        help="bounded canary; end of duration is recorded as feature cutoff",
    )
    cutoff_group.add_argument(
        "--feature-cutoff-at",
        help="explicit timezone-aware ISO-8601 cutoff",
    )
    cutoff_group.add_argument(
        "--cutoff-slot",
        choices=("open",),
        help="wait for the actual R1 snapshot and use decision_completed_at",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    try:
        (
            markets,
            universe_source,
            universe_fetch_started_at_ns,
            universe_observed_at_ns,
        ) = _resolve_markets(args.markets)
        explicit_cutoff = (
            parse_utc_datetime(args.feature_cutoff_at)
            if args.feature_cutoff_at
            else None
        )
        config = CaptureConfig(
            markets=markets,
            output_root=args.output_root,
            endpoint=args.endpoint,
            orderbook_depth=args.orderbook_depth,
            queue_max=args.queue_max,
            max_wait_seconds=args.max_wait_seconds,
            required_warmup_seconds=args.required_warmup_seconds,
            feature_cutoff_at=explicit_cutoff,
            duration_seconds=args.duration_seconds,
            cutoff_slot=args.cutoff_slot,
            asof=_parse_asof(args.asof),
            snapshot_root=args.snapshot_root,
            universe_source=universe_source,
            universe_fetch_started_at_ns=universe_fetch_started_at_ns,
            universe_observed_at_ns=universe_observed_at_ns,
        )
        result = run_capture(config)
    except (OSError, ValueError, RuntimeError) as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        return 2

    print(f"manifest: {result.manifest_path}")
    print(
        "complete={complete} transport={transport} causal_window={causal} "
        "stop={stop}".format(
            complete=result.complete,
            transport=result.manifest["quality"]["transport_complete"],
            causal=result.manifest["quality"]["causal_window_complete"],
            stop=result.manifest["stop_reason"],
        )
    )
    for channel in ("trade", "orderbook"):
        artifact = result.manifest["streams"][channel]["artifact"] or {}
        print(
            f"{channel}: events={artifact.get('event_count', 0)} "
            f"markets={artifact.get('market_count', 0)} "
            f"reconnects={result.manifest['streams'][channel]['reconnect_count']} "
            f"drops={result.manifest['streams'][channel]['dropped_frames']}"
        )
    return 0 if result.complete else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Offline, outcome-free evidence for one pre-decision trade-flow feature.

This sidecar is generated AFTER the decision. Causal event availability is not
proof that the live recommendation consumed the feature. No fitting, ranking,
labels, orders, network access, or artifact writers belong in this module.
"""
from __future__ import annotations

import gzip
import copy
import hashlib
import json
import math
import re
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

from data.upbit_microstructure import (
    MANIFEST_SCHEMA_VERSION, RAW_SCHEMA_VERSION, RawFrame, build_subscription, parse_raw_frame,
    universe_sha256,
)
from ops.artifact_provenance import canonical_json_bytes, file_identity, strict_json_object
from signals.recommend_snapshot import load_snapshot

MICROSTRUCTURE_CONFIG = {
    "lookback_seconds": 300,
    "window": "[decision_started_at-300s,decision_started_at)",
    "feature_columns": ["trade_notional_imbalance_300s"],
    "zero_trade": "null_not_zero",
    "scope": "offline_record_only_no_fit_rank_or_label_join",
    "supported_slot": "open",
    "universe_size": 100,
}
SCHEMA = "recommend_microstructure_features.v1"
FEATURE = "trade_notional_imbalance_300s"
ROOT = Path(__file__).resolve().parent.parent
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


class MicrostructureEvidenceError(ValueError):
    """Corrupted, inconsistent or changing evidence: fail the entire build."""


def _require(condition, reason):
    if not condition:
        raise MicrostructureEvidenceError(reason)


def _integer(value, name, *, minimum=0):
    _require(type(value) is int and value >= minimum, f"invalid {name}")
    return value


def _ns(value):
    _require(isinstance(value, str), "timestamp must be an aware ISO string")
    _require(re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})", value)
             is not None, "timestamp format/precision unsupported")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    _require(parsed.tzinfo is not None and parsed.utcoffset() is not None, "naive timestamp")
    delta = parsed.astimezone(timezone.utc) - _EPOCH
    return ((delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds) * 1000


def _iso(value):
    return (_EPOCH + timedelta(microseconds=value // 1000)).isoformat()


def _digest(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _unique(items):
    result = {}
    for key, value in items:
        _require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _json(value):
    def bad_constant(_value):
        raise MicrostructureEvidenceError("nonfinite JSON constant")
    return json.loads(value, object_pairs_hook=_unique, parse_constant=bad_constant)


def _snapshot(snapshot):
    _require(snapshot.get("snapshot_schema") == "recommend_snapshot.v2", "modern snapshot required")
    payload = {k: v for k, v in snapshot.items()
               if k not in {"created_at", "snapshot_id", "payload_sha256", "snapshot_path"}}
    digest = _digest(payload)
    _require(snapshot.get("payload_sha256") == digest, "snapshot payload checksum mismatch")
    _require(snapshot.get("snapshot_id") == f"recommend-{digest[:20]}", "snapshot id mismatch")
    request, model = snapshot["request"], snapshot["model"]
    _require(snapshot["slot"] == request["slot"] == "open", "only open snapshots supported")
    _require(snapshot["ranking"] == request["ranking"] == model["ranking"] == "R1", "R1 required")
    _require(model["id"] == "recommend_r1_open" and request.get("limit_markets") is None,
             "full-universe R1 identity required")
    _require(snapshot["asof"] == request["asof"], "snapshot date mismatch")
    cutoff = _ns(snapshot["decision_started_at"])
    completed = _ns(snapshot["decision_completed_at"])
    _require(cutoff <= completed == _ns(snapshot["created_at"]), "snapshot chronology mismatch")
    day_start = _ns(snapshot["asof"] + "T09:00:00+09:00")
    _require(day_start + 300_000_000_000 <= cutoff < day_start + 1260_000_000_000,
             "decision outside open slot or insufficient post-open feature window")
    rows = snapshot["universe"]
    _require(isinstance(rows, list) and len(rows) == 100, "exactly 100 original candidates required")
    _require(all(type(row["rank"]) is int for row in rows), "invalid rank")
    _require(sorted(row["rank"] for row in rows) == list(range(1, 101)), "rank sequence mismatch")
    _require(len({row["coin"] for row in rows}) == 100, "duplicate candidate")
    ordered = sorted(rows, key=lambda row: row["rank"])
    _require(snapshot["top3"] == ordered[:3], "original top3 identity mismatch")
    return ordered, cutoff, cutoff - 300_000_000_000


def _source_quality(manifest):
    source = manifest.get("source_code") or {}
    left, right = source.get("at_start") or {}, source.get("at_end") or {}
    for document in (left, right):
        files = document.get("files") or []
        expected_sources = {"data/collector_upbit_microstructure.py", "data/upbit_microstructure.py"}
        if (len(files) != 2 or {item.get("path") for item in files} != expected_sources
                or any(not _sha(item.get("sha256")) or item.get("error") for item in files)):
            return False
        git, runtime = document.get("git") or {}, document.get("runtime") or {}
        if not git.get("commit") or type(git.get("dirty")) is not bool:
            return False
        if git.get("commit_error") or git.get("status_error"):
            return False
        if not all(runtime.get(key) for key in ("python_version", "python_implementation", "websockets_version")):
            return False
    return (left["files"] == right["files"] and left["runtime"] == right["runtime"]
            and left["git"] == right["git"]
            and source.get("provenance_complete") is True
            and source.get("unchanged_during_capture") is True)


def _capture_quality(snapshot, manifest, candidates, cutoff, start):
    _require(manifest.get("schema_version") == MANIFEST_SCHEMA_VERSION, "unsupported capture schema")
    _require(manifest.get("source") == "upbit", "unexpected source")
    _require(isinstance(manifest.get("capture_id"), str) and manifest["capture_id"], "capture id missing")
    _require(manifest.get("asof") == snapshot["asof"], "capture date mismatch")
    universe = manifest["universe"]
    markets = universe["markets"]
    _require(isinstance(markets, list) and len(markets) == len(set(markets)), "invalid capture universe")
    _require(universe["sha256"] == universe_sha256(markets), "capture universe hash mismatch")
    _require(universe["count"] == len(markets), "capture universe count mismatch")
    subscription = manifest["subscription"]
    depth = subscription["orderbook_depth"]["requested"]
    _integer(depth, "orderbook depth", minimum=1)
    for channel in ("trade", "orderbook"):
        expected_subscription = build_subscription(channel, markets, ticket="unused", orderbook_depth=depth)[1:]
        _require(subscription[channel] == expected_subscription, f"{channel} subscription universe mismatch")
    _require(subscription.get("format") == "DEFAULT" and type(subscription.get("orderbook_level")) is int
             and subscription["orderbook_level"] == 0, "unsupported subscription format/level")
    reasons = []
    required = {row["coin"] for row in candidates}
    missing = sorted(required - set(markets))
    if missing:
        reasons.append("snapshot_markets_missing_from_capture")
    source = manifest.get("cutoff_source") or {}
    if source.get("kind") != "recommend_snapshot":
        reasons.append("canary_or_explicit_capture_not_research_evidence")
    else:
        for key in ("snapshot_id", "payload_sha256", "asof", "slot", "decision_started_at", "decision_completed_at"):
            _require(source.get(key) == snapshot.get(key), f"capture snapshot binding mismatch: {key}")
        _require(manifest["feature_cutoff_at_ns"] == _ns(snapshot["decision_completed_at"]),
                 "capture cutoff is not original decision completion")
    for key in ("public_quotation_only", "complete"):
        if manifest.get(key) is not True:
            reasons.append(f"capture_{key}_not_true")
    if manifest.get("uses_api_key") is not False or manifest.get("places_orders") is not False:
        reasons.append("capture_not_public_record_only")
    quality = manifest.get("quality") or {}
    for key in ("transport_complete", "causal_window_complete", "clock_sync_ok", "warmup_ready",
                "lifecycle_ok", "universe_timing_valid", "source_provenance_complete",
                "source_unchanged_during_capture", "orphan_recovery_ok", "complete"):
        if quality.get(key) is not True:
            reasons.append(f"capture_quality_{key}_not_true")
    if not _source_quality(manifest):
        reasons.append("capture_source_provenance_unavailable")
    clock = manifest.get("clock") or {}
    if any((clock.get(key) or {}).get("ntp_synchronized") is not True for key in ("at_start", "at_end")):
        reasons.append("clock_synchronization_unproven")
    warmup = manifest["required_warmup_seconds"]
    _require(type(warmup) in (int, float) and math.isfinite(warmup) and warmup >= 0,
             "invalid required warmup")
    deadline = start - int(warmup * 1_000_000_000)
    began = _integer(manifest["started_at_ns"], "capture start", minimum=1)
    ended = _integer(manifest["ended_at_ns"], "capture end", minimum=1)
    _require(began <= ended, "capture chronology reversed")
    if began > deadline or ended < cutoff:
        reasons.append("capture_does_not_cover_window_and_warmup")
    observed = _integer(universe["observed_at_ns"], "universe observed", minimum=1)
    if observed > began or universe.get("frozen_for_capture") is not True:
        reasons.append("universe_not_frozen_before_capture")
    for channel in ("trade", "orderbook"):
        stream = manifest["streams"][channel]
        if any(stream.get(key) for key in ("dropped_frames", "reconnect_count", "writer_error", "receiver_error", "gaps")):
            reasons.append(f"{channel}_transport_discontinuity")
        connections = stream.get("connections") or []
        for connection in connections:
            for key in ("opened_at_ns", "subscribed_at_ns", "closed_at_ns"):
                if connection.get(key) is not None:
                    _integer(connection[key], f"{channel}.{key}", minimum=1)
        if len(connections) != 1:
            reasons.append(f"{channel}_single_connection_unproven")
        elif (connections[0].get("error") or connections[0].get("clean_stop") is not True
              or connections[0].get("subscribed_at_ns") is None
              or connections[0].get("opened_at_ns") is None
              or connections[0]["opened_at_ns"] > connections[0]["subscribed_at_ns"]
              or connections[0]["subscribed_at_ns"] > deadline
              or connections[0].get("closed_at_ns") is None
              or connections[0]["closed_at_ns"] < cutoff):
            reasons.append(f"{channel}_subscription_or_continuity_unproven")
        artifact = stream.get("artifact") or {}
        if artifact.get("exists") is not True or artifact.get("writer_state") != "finalized":
            reasons.append(f"{channel}_raw_not_finalized")
    return reasons, set(markets), missing, deadline


def _validated_records(records, channel, manifest, markets):
    """Validate one pass; final counters are checked only after exhaustion."""
    stream = manifest["streams"][channel]
    artifact = stream.get("artifact") or {}
    connections = {item["connection_id"]: item for item in stream.get("connections") or []}
    previous_mono, previous_wall = -1, -1
    record_count, event_count = 0, 0
    for sequence, record in enumerate(records, 1):
        record_count += 1
        _require(record["schema_version"] == RAW_SCHEMA_VERSION, "raw schema mismatch")
        _require(record["capture_id"] == manifest["capture_id"] and record["channel"] == channel,
                 "raw capture/channel mismatch")
        _require(_integer(record["ingress_seq"], "ingress", minimum=1) == sequence, "raw ingress gap/order mismatch")
        wall = _integer(record["received_at_ns"], "received ns", minimum=1)
        mono = _integer(record["received_monotonic_ns"], "monotonic ns", minimum=1)
        _require(mono > previous_mono and wall >= previous_wall, "raw receive clock/order reversed")
        previous_mono, previous_wall = mono, wall
        _require(manifest["started_at_ns"] <= wall <= manifest["ended_at_ns"], "raw receive outside capture")
        persisted = _integer(record["persisted_at_ns"], "persisted ns", minimum=1)
        _require(persisted >= wall, "persisted clock precedes receipt")
        connection = connections.get(record["connection_id"])
        _require(connection is not None and connection["subscription_id"] == record["subscription_id"],
                 "raw connection/subscription mismatch")
        first = _integer(connection["first_ingress_seq"], "connection first ingress", minimum=1)
        last = _integer(connection["last_ingress_seq"], "connection last ingress", minimum=1)
        _require(first <= sequence <= last,
                 "raw ingress outside connection")
        _require(connection["subscribed_at_ns"] <= wall <= connection["closed_at_ns"],
                 "raw receive outside connection subscription lifetime")
        raw = record["raw_payload"].encode("utf-8")
        # A valid market event can never be encoded using replacement UTF-8.
        if record.get("raw_payload_base64") is not None:
            import base64
            raw = base64.b64decode(record["raw_payload_base64"], validate=True)
        _require(hashlib.sha256(raw).hexdigest() == record["payload_sha256"], "raw payload checksum mismatch")
        frame = RawFrame(raw, wall, mono, record["connection_id"], record["subscription_id"],
                         sequence, record["frame_kind"])
        regenerated, _ = parse_raw_frame(frame, capture_id=manifest["capture_id"], channel=channel,
                                         expected_markets=markets, persisted_at_ns=record["persisted_at_ns"])
        _require(regenerated == record, "raw envelope/payload inconsistency")
        payload = _json(raw) if record["record_type"] == "event" else None
        event_count += record["record_type"] == "event"
        yield record, payload
    if artifact.get("exists") is True:
        expected_count = _integer(artifact.get("record_count"), "raw record_count")
        expected_events = _integer(artifact.get("event_count"), "raw event_count")
        _require(record_count == expected_count, f"{channel} raw count mismatch")
        _require(event_count == expected_events, f"{channel} event count mismatch")


def _add_notional(partials, value):
    """Compact compensated sum: keep rounding residuals, not every trade."""
    index = 0
    for other in partials:
        if abs(value) < abs(other):
            value, other = other, value
        high = value + other
        _require(math.isfinite(high), "nonfinite summed trade notional")
        low = other - (high - value)
        if low:
            partials[index] = low
            index += 1
        value = high
    partials[index:] = [value]


def _compute(snapshot, manifest, trade_records, orderbook_records):
    candidates, cutoff, start = _snapshot(snapshot)
    reasons, markets, missing, warmup_deadline = _capture_quality(snapshot, manifest, candidates, cutoff, start)
    trades = _validated_records(trade_records, "trade", manifest, markets)
    books = _validated_records(orderbook_records, "orderbook", manifest, markets)
    early_books, last_books = {}, {}
    for record, payload in books:
        if payload is None:
            reasons.append("raw_parse_or_server_errors")
            continue
        event_ns, received = record["event_at_ms"] * 1_000_000, record["received_at_ns"]
        coin = record["market"]
        if record["stream_type"] == "SNAPSHOT" and event_ns <= warmup_deadline and received <= warmup_deadline:
            early_books[coin] = True
        if event_ns < cutoff and received < cutoff:
            # Orderbook timestamps are NOT trade identifiers: preserve updates.
            last_books[coin] = (record, payload)
    missing_books = sorted({row["coin"] for row in candidates} - set(early_books))
    if missing_books:
        reasons.append("initial_book_snapshot_missing_before_warmup")
    counts = {"before_window": 0, "event_at_or_after_cutoff": 0,
              "received_at_or_after_cutoff": 0, "duplicates": 0}
    grouped, seen = {}, {}
    for record, payload in trades:
        if payload is None:
            reasons.append("raw_parse_or_server_errors")
            continue
        event_ns, received = record["event_at_ms"] * 1_000_000, record["received_at_ns"]
        # Trim BEFORE sequential-id deduplication: future data cannot replace
        # an admissible trade or poison its duplicate-conflict decision.
        if event_ns >= cutoff or received >= cutoff:
            counts["event_at_or_after_cutoff"] += event_ns >= cutoff
            counts["received_at_or_after_cutoff"] += received >= cutoff
            continue
        if event_ns < start or received < start:
            counts["before_window"] += 1
            continue
        _require(record["stream_type"] == "REALTIME", "trade must be REALTIME")
        key = (record["market"], payload["sequential_id"])
        core = (payload["trade_timestamp"], payload["trade_price"], payload["trade_volume"], payload["ask_bid"])
        if key in seen:
            _require(seen[key] == core, "conflicting admissible trade sequential_id")
            counts["duplicates"] += 1
            continue
        seen[key] = core
        amount = float(payload["trade_price"]) * float(payload["trade_volume"])
        _require(math.isfinite(amount) and amount > 0, "invalid trade notional")
        totals = grouped.setdefault(record["market"], {"BID": [], "ASK": [], "count": 0})
        _add_notional(totals[payload["ask_bid"]], amount)
        totals["count"] += 1
    reasons = sorted(set(reasons))
    rows = []
    for candidate in candidates:
        coin = candidate["coin"]
        totals = grouped.get(coin, {"BID": [], "ASK": [], "count": 0})
        bid, ask = math.fsum(totals["BID"]), math.fsum(totals["ASK"])
        total = bid + ask
        _require(math.isfinite(total), "nonfinite summed trade notional")
        status = "unavailable" if reasons else "available" if total else "no_observed_trades"
        spread, age = None, None
        if coin in last_books:
            record, book = last_books[coin]
            best_ask = min(unit["ask_price"] for unit in book["orderbook_units"])
            best_bid = max(unit["bid_price"] for unit in book["orderbook_units"])
            spread = (best_ask - best_bid) / ((best_ask + best_bid) / 2)
            age = (cutoff - record["event_at_ms"] * 1_000_000) / 1_000_000_000
        rows.append({"coin": coin, "rank": candidate["rank"],
                     FEATURE: (bid - ask) / total if status == "available" else None,
                     "feature_status": status, "observed_trade_count": totals["count"],
                     "bid_notional": bid, "ask_notional": ask,
                     "quality_diagnostics": {"spread_fraction": spread, "book_event_age_seconds": age}})
    available = [row["coin"] for row in rows if row[FEATURE] is not None]
    return {
        "schema": SCHEMA, "config": json.loads(json.dumps(MICROSTRUCTURE_CONFIG)),
        "snapshot": {key: snapshot[key] for key in ("snapshot_id", "payload_sha256", "asof", "slot", "decision_started_at")},
        "capture_id": manifest["capture_id"],
        "window": {"start_at_ns": start, "end_exclusive_at_ns": cutoff,
                   "start_at": _iso(start), "end_exclusive_at": _iso(cutoff),
                   "rule": "both event_at and received_at inside half-open window"},
        "feature_evidence_valid": not reasons,
        "feature_available_rows": len(available), "feature_coverage": len(available) / 100,
        "experiment_readiness": "not_assessed", "effect_status": "not_evaluated",
        "quality_reasons": reasons, "rows": rows,
        "coverage": {"recorded_rows": 100, "feature_available_rows": len(available),
                     "available_coins": available, "missing_capture_coins": missing,
                     "missing_initial_book_coins": missing_books,
                     "no_observed_trade_coins": [row["coin"] for row in rows if not row["observed_trade_count"]]},
        "trade_filter_audit": counts,
        "provenance": {"capture_manifest_payload_sha256": _digest(manifest),
                       "raw_artifacts": {channel: copy.deepcopy(manifest["streams"][channel].get("artifact"))
                                         for channel in ("trade", "orderbook")}},
        "notes": ["Post-decision sidecar: causal feature evidence, not an actual decision-time prediction.",
                  "No labels, fitting, ranking, promotion, or predictive-effect evaluation.",
                  "Whole-capture failures conservatively invalidate this sidecar even if after its cutoff.",
                  "No observed trades is not proof of market-wide absence; public-feed completeness is not guaranteed.",
                  "BID-minus-ASK executed notional is not net capital inflow; book fields are quality diagnostics only."],
    }


def compute_trade_imbalance(snapshot, manifest, *, trade_records, orderbook_records):
    """Pure input-preserving calculation; dictionaries are never mutated."""
    try:
        return _compute(snapshot, manifest, trade_records, orderbook_records)
    except MicrostructureEvidenceError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError, AttributeError) as exc:
        raise MicrostructureEvidenceError(f"invalid microstructure evidence: {type(exc).__name__}") from exc


def read_recommend_microstructure(snapshot_path, manifest_path):
    """Read stable finalized sources; never create a lock, directory or file."""
    snapshot_path, manifest_path = Path(snapshot_path), Path(manifest_path)
    paths = [snapshot_path, manifest_path, Path(__file__), ROOT / "data/upbit_microstructure.py",
             ROOT / "signals/recommend_snapshot.py", ROOT / "ops/artifact_provenance.py"]
    try:
        before = [file_identity(path, root=ROOT) for path in paths]
        _require(all(item["exists"] for item in before), "snapshot/manifest/generator file missing")
        snapshot = load_snapshot(snapshot_path, slot="open", ranking="R1", model_id="recommend_r1_open")
        manifest = strict_json_object(manifest_path)
        source = manifest.get("cutoff_source") or {}
        if source.get("kind") == "recommend_snapshot":
            _require(source.get("file_sha256") == before[0]["sha256"], "bound snapshot file checksum mismatch")
            _require(Path(source["path"]).absolute().resolve() == snapshot_path.absolute().resolve(),
                     "bound snapshot file location mismatch")
        raw_inputs = {}
        for channel in ("trade", "orderbook"):
            artifact = manifest["streams"][channel].get("artifact") or {}
            if artifact.get("exists") is not True or artifact.get("writer_state") != "finalized":
                raw_inputs[channel] = []
                continue
            path = Path(artifact["path"])
            _require(path.resolve() == (manifest_path.parent / f"{channel}.jsonl.gz").resolve(),
                     "raw artifact must be its capture's canonical channel file")
            identity = file_identity(path, root=ROOT)
            _require(identity["exists"] and identity["sha256"] == artifact["sha256"]
                     and identity["size"] == artifact["size_bytes"], "raw artifact checksum/size mismatch")
            paths.append(path)
            before.append(identity)
            raw_inputs[channel] = _gzip_records(path)
        try:
            report = compute_trade_imbalance(snapshot, manifest, trade_records=raw_inputs["trade"],
                                             orderbook_records=raw_inputs["orderbook"])
        finally:
            for records in raw_inputs.values():
                close = getattr(records, "close", None)
                if close is not None:
                    close()
        after = [file_identity(path, root=ROOT) for path in paths]
        _require(before == after, "input or generator changed during read")
        report["provenance"]["files"] = before
        report["provenance"]["generator_sources"] = before[2:6]
        report["provenance"]["inputs_unchanged"] = True
        return report
    except MicrostructureEvidenceError:
        raise
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise MicrostructureEvidenceError(f"microstructure read failed: {type(exc).__name__}") from exc


def _gzip_records(path):
    """Own and close each compressed stream while reading exactly once."""
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield _json(line)
    except (OSError, EOFError, zlib.error) as exc:
        raise MicrostructureEvidenceError("invalid or truncated compressed raw stream") from exc

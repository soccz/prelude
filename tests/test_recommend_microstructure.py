"""Synthetic-only sidecar tests: no real captures, DB, models or HTTP."""
from __future__ import annotations

import copy
import gzip
import hashlib
import json

import pytest
import requests

from data.upbit_microstructure import RawFrame, build_subscription, parse_raw_frame, universe_sha256
import signals.recommend_microstructure as feature
import signals.recommend_snapshot as snapshots

DAY = "2026-09-07"
CUTOFF = feature._ns(DAY + "T09:05:00.123456+09:00")
START = CUTOFF - 300_000_000_000
WARMUP = START - 600_000_000_000
MARKETS = [f"KRW-T{i:03d}" for i in range(100)]


@pytest.fixture(autouse=True)
def no_live_operations(monkeypatch):
    monkeypatch.setenv("PRELUDE_FORBID_TELEGRAM", "1")

    def forbidden(*_args, **_kwargs):
        pytest.fail("sidecar attempted a live operation")

    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    monkeypatch.setattr(snapshots, "_data_metadata", forbidden)
    monkeypatch.setattr(snapshots, "_git_code_metadata", forbidden)
    monkeypatch.setattr(snapshots, "_environment_metadata", forbidden)


def _snapshot():
    rows = [{"coin": coin, "rank": index + 1, "score": 0.5,
             "pump_prob": 0.02, "pump_prob_pct": "2.0%", "rr_ratio": 1.,
             "p_up5": 0.3, "p_up10": 0.1, "p_up20": 0.02, "p_dn5": 0.1,
             "p_dn10": 0.03, "exp_downside": -0.02, "dump_risk_flag": False,
             "entry_open": 100., "sl": -0.03, "tp": 0.05, "btc_regime": "neutral",
             "feature_values": {"f_ret_3d": .01}} for index, coin in enumerate(MARKETS)]
    result = {"asof": DAY, "slot": "open", "feature_date": DAY, "btc_regime": "neutral",
              "universe_n": 100, "calibration_source": "test", "rank_basis": "R1_riskreward(de-corr head)",
              "n_history_dates": 100, "ranking": "R1", "score_schema_version": "recommend_score.v2",
              "rule_version": "r1_riskreward_v1", "model_random_seed": 42,
              "feature_columns": ["f_ret_3d"],
              "training": {"start": "2025-01-01", "end": "2026-09-01", "cutoff_exclusive": "2026-09-02",
                           "embargo_days": 5, "rows": 1000, "dates": 100},
              "universe": rows, "top3": rows[:3]}
    data = {"path": "data/upbit_d1.db", "exists": True, "sha256": "a"*64,
            "rows": 1000, "markets": 100, "max_timestamp": DAY + " 09:00:00"}
    data["manifest_id"] = hashlib.sha256(snapshots._canonical_bytes(data)).hexdigest()
    return snapshots._build_document(
        result, ranking="R1", limit_markets=None, model_id=None,
        created_at=feature._iso(CUTOFF + 1_000_000_000),
        decision_started_at=feature._iso(CUTOFF),
        decision_completed_at=feature._iso(CUTOFF + 1_000_000_000),
        code_metadata={"git_commit": "b"*40, "score_sources_dirty": False,
                       "score_source_sha256": "c"*64,
                       "score_source_files": list(snapshots._SCORE_SOURCE_FILES)},
        data_metadata=data,
        environment_metadata={"python": "3.12", "implementation": "CPython", "packages": {}},
    )


def _trade(*, coin=MARKETS[0], event=None, received=None, side="BID", price=10., volume=2., trade_id=1):
    event = START + 1_000_000_000 if event is None else event
    received = event + 1_000_000 if received is None else received
    return {"type": "trade", "code": coin, "trade_timestamp": event//1_000_000,
            "timestamp": event//1_000_000, "ask_bid": side, "trade_price": price,
            "trade_volume": volume, "sequential_id": trade_id, "stream_type": "REALTIME"}, received


def _book(coin, *, at=None, stream="SNAPSHOT", ask=11., bid=10.):
    at = WARMUP - 1_000_000_000 if at is None else at
    return {"type": "orderbook", "code": coin, "timestamp": at//1_000_000,
            "total_ask_size": 10., "total_bid_size": 10., "stream_type": stream,
            "orderbook_units": [{"ask_price": ask, "bid_price": bid, "ask_size": 10., "bid_size": 10.}]}, at


def _records(events, channel):
    result = []
    for i, (payload, received) in enumerate(events, 1):
        frame = RawFrame(json.dumps(payload).encode(), received, 10_000+i,
                         channel+"-connection", channel+"-subscription", i)
        record, _ = parse_raw_frame(frame, capture_id="synthetic-capture", channel=channel,
                                    expected_markets=set(MARKETS), persisted_at_ns=received+20_000_000_000)
        result.append(record)
    return result


def _sample(trades=None, books=None):
    snapshot = _snapshot()
    trades = [_trade()] if trades is None else trades
    books = [_book(coin) for coin in MARKETS] if books is None else books
    records = {"trade": _records(trades, "trade"), "orderbook": _records(books, "orderbook")}
    source = {"files": [{"path": name, "sha256": "a"*64, "error": None} for name in
                        ("data/collector_upbit_microstructure.py", "data/upbit_microstructure.py")],
              "git": {"commit": "b"*40, "dirty": False, "commit_error": None, "status_error": None},
              "runtime": {"python_version": "3.12", "python_implementation": "CPython", "websockets_version": "15"}}
    streams = {}
    for channel, values in records.items():
        streams[channel] = {"dropped_frames": 0, "reconnect_count": 0, "writer_error": None,
                            "receiver_error": None, "gaps": [],
                            "connections": [{"connection_id": channel+"-connection",
                                             "subscription_id": channel+"-subscription",
                                             "opened_at_ns": WARMUP-2_100_000_000,
                                             "subscribed_at_ns": WARMUP-2_000_000_000,
                                             "closed_at_ns": CUTOFF+30_000_000_000,
                                             "first_ingress_seq": 1 if values else None,
                                             "last_ingress_seq": len(values) if values else None,
                                             "clean_stop": True, "error": None}],
                            "artifact": {"exists": True, "writer_state": "finalized",
                                         "record_count": len(values),
                                         "event_count": sum(r["record_type"] == "event" for r in values)}}
    manifest = {"schema_version": "upbit_microstructure.capture.v1", "capture_id": "synthetic-capture",
                "source": "upbit", "asof": DAY, "public_quotation_only": True,
                "uses_api_key": False, "places_orders": False, "complete": True,
                "started_at_ns": WARMUP-3_000_000_000, "ended_at_ns": CUTOFF+30_000_000_000,
                "required_warmup_seconds": 600, "feature_cutoff_at_ns": CUTOFF+1_000_000_000,
                "cutoff_source": {"kind": "recommend_snapshot", **{k:snapshot[k] for k in
                    ("snapshot_id", "payload_sha256", "asof", "slot", "decision_started_at", "decision_completed_at")}},
                "universe": {"markets": MARKETS.copy(), "sha256": universe_sha256(MARKETS), "count": 100,
                             "observed_at_ns": WARMUP-4_000_000_000, "frozen_for_capture": True},
                "subscription": {"trade": build_subscription("trade", MARKETS, ticket="unused",orderbook_depth=1)[1:],
                                 "orderbook": build_subscription("orderbook",MARKETS,ticket="unused",orderbook_depth=1)[1:],
                                 "orderbook_depth": {"requested": 1}, "orderbook_level": 0, "format": "DEFAULT"},
                "source_code": {"at_start": copy.deepcopy(source), "at_end": copy.deepcopy(source),
                                "provenance_complete": True, "unchanged_during_capture": True},
                "quality": {k: True for k in ("transport_complete", "causal_window_complete", "clock_sync_ok",
                    "warmup_ready", "lifecycle_ok", "universe_timing_valid", "source_provenance_complete",
                    "source_unchanged_during_capture", "orphan_recovery_ok", "complete")},
                "clock": {"at_start": {"ntp_synchronized": True}, "at_end": {"ntp_synchronized": True}},
                "streams": streams}
    return snapshot, manifest, records


def _compute(sample):
    snapshot, manifest, records = sample
    return feature.compute_trade_imbalance(snapshot, manifest, trade_records=records["trade"],
                                            orderbook_records=records["orderbook"])


def test_one_feature_exact_notional_no_trade_null_and_original_universe_preserved():
    sample = _sample([_trade(side="BID", price=10, volume=3), _trade(side="ASK", price=10, volume=1, trade_id=2)])
    before = copy.deepcopy(sample)
    result = _compute(sample)
    assert result["feature_evidence_valid"] is True
    assert result["rows"][0][feature.FEATURE] == .5
    assert result["rows"][1][feature.FEATURE] is None
    assert result["rows"][1]["feature_status"] == "no_observed_trades"
    assert result["feature_coverage"] == .01
    assert result["experiment_readiness"] == "not_assessed"
    assert result["effect_status"] == "not_evaluated"
    assert [r["rank"] for r in result["rows"]] == list(range(1,101))
    assert sample == before


def test_all_no_trade_capture_is_valid_but_has_no_feature():
    result = _compute(_sample([]))
    assert result["feature_evidence_valid"] is True
    assert result["feature_available_rows"] == 0
    assert all(r[feature.FEATURE] is None and r["feature_status"] == "no_observed_trades" for r in result["rows"])


@pytest.mark.parametrize("side,value", [("ASK", -1.), ("BID", 1.)])
def test_one_sided_notional(side, value):
    assert _compute(_sample([_trade(side=side)]))["rows"][0][feature.FEATURE] == value


@pytest.mark.parametrize("case", ["late_receive", "equal_receive", "future_event", "before_start"])
def test_cutoff_trims_before_dedup_even_conflicting_future_trade(case):
    existing = _trade()
    event, received = START+2_000_000_000, CUTOFF
    if case == "late_receive":
        received += 1
    elif case == "future_event":
        event = CUTOFF+1_000_000
        received = CUTOFF-1000
    elif case == "before_start":
        event = START-1_000_000
        received = START+2_000_000_000
    extra = _trade(event=event, received=received, side="ASK", price=999999., trade_id=1)
    result = _compute(_sample([existing, extra]))
    assert result["rows"][0][feature.FEATURE] == 1.
    assert result["rows"][0]["observed_trade_count"] == 1


def test_ns_conversion_does_not_round_via_float_timestamp():
    assert feature._ns(DAY+"T09:05:00.123456+09:00") % 1_000_000_000 == 123456000


@pytest.mark.parametrize("value", [True, 123, "2026-09-07", "2026-09-07T09:05:00",
                                   "2026-09-07T09:05:00.123456789+09:00"])
def test_ns_conversion_rejects_invalid_or_silently_truncated_times(value):
    with pytest.raises(feature.MicrostructureEvidenceError):
        feature._ns(value)


def test_lower_bound_is_inclusive_and_upper_bound_exclusive():
    sample = _sample()
    snap, manifest, _ = sample
    old = snap["decision_started_at"]
    cutoff = feature._ns(old) // 1_000_000 * 1_000_000
    start = cutoff-300_000_000_000
    snap["decision_started_at"] = feature._iso(cutoff)
    payload = {k:v for k,v in snap.items() if k not in {"created_at","snapshot_id","payload_sha256","snapshot_path"}}
    snap["payload_sha256"] = feature._digest(payload)
    snap["snapshot_id"] = "recommend-"+snap["payload_sha256"][:20]
    for key in ("snapshot_id","payload_sha256","decision_started_at"):
        manifest["cutoff_source"][key] = snap[key]
    events = [_trade(event=start, received=start), _trade(event=cutoff, received=cutoff, side="ASK", trade_id=2)]
    sample[2]["trade"] = _records(events,"trade")
    manifest["streams"]["trade"]["artifact"].update(record_count=2,event_count=2)
    manifest["streams"]["trade"]["connections"][0]["last_ingress_seq"] = 2
    result = _compute(sample)
    assert result["rows"][0][feature.FEATURE] == 1.


def test_admissible_same_id_duplicate_once_but_conflict_invalid():
    duplicate = _trade()
    result = _compute(_sample([duplicate, duplicate]))
    assert result["trade_filter_audit"]["duplicates"] == 1
    assert result["rows"][0]["observed_trade_count"] == 1
    with pytest.raises(feature.MicrostructureEvidenceError, match="conflicting"):
        _compute(_sample([duplicate, _trade(side="ASK")]))


def test_orderbook_same_timestamp_updates_not_deduplicated():
    books = [_book(coin) for coin in MARKETS]
    books.extend([_book(MARKETS[0],at=START,stream="REALTIME",ask=12),
                  _book(MARKETS[0],at=START,stream="REALTIME",ask=13)])
    result = _compute(_sample(books=books))
    assert result["rows"][0]["quality_diagnostics"]["spread_fraction"] == pytest.approx(3/11.5)


@pytest.mark.parametrize("damage", ["complete", "clock", "warmup", "book_missing", "drop", "reconnect", "canary", "partial", "missing_market"])
def test_unavailable_evidence_retains_all_candidates(damage):
    sample = _sample()
    _, manifest, records = sample
    if damage == "complete":
        manifest["complete"] = False
    elif damage == "clock":
        manifest["clock"]["at_end"]["ntp_synchronized"] = None
    elif damage == "warmup":
        manifest["streams"]["trade"]["connections"][0]["subscribed_at_ns"] = START
    elif damage == "book_missing":
        records["orderbook"] = _records([_book(coin) for coin in MARKETS[1:]],"orderbook")
        manifest["streams"]["orderbook"]["artifact"].update(record_count=99,event_count=99)
    elif damage == "drop":
        manifest["streams"]["trade"]["dropped_frames"] = 1
    elif damage == "reconnect":
        manifest["streams"]["trade"]["reconnect_count"] = 1
    elif damage == "canary":
        manifest["cutoff_source"] = {"kind": "duration_canary"}
    elif damage == "partial":
        manifest["streams"]["trade"]["artifact"]["exists"] = False
        records["trade"] = []
    elif damage == "missing_market":
        manifest["universe"].update(markets=MARKETS[:-1],count=99,sha256=universe_sha256(MARKETS[:-1]))
        for channel in ("trade", "orderbook"):
            manifest["subscription"][channel] = build_subscription(channel,MARKETS[:-1],ticket="unused",orderbook_depth=1)[1:]
        records["orderbook"] = _records([_book(coin) for coin in MARKETS[:-1]],"orderbook")
        manifest["streams"]["orderbook"]["artifact"].update(record_count=99,event_count=99)
    result = _compute(sample)
    assert result["feature_evidence_valid"] is False
    assert result["quality_reasons"]
    assert len(result["rows"]) == 100
    assert all(row[feature.FEATURE] is None for row in result["rows"])


@pytest.mark.parametrize("damage", ["ingress", "bool_ns", "naive", "hash", "envelope", "clock_backwards", "snapshot", "source_identity", "rank"])
def test_corruption_is_hard_failure(damage):
    sample = _sample([_trade(),_trade(trade_id=2)])
    snap, manifest, records = sample
    if damage == "ingress":
        records["trade"].reverse()
    elif damage == "bool_ns":
        records["trade"][0]["received_at_ns"] = True
    elif damage == "naive":
        snap["decision_started_at"] = "2026-09-07T09:05:00"
    elif damage == "hash":
        records["trade"][0]["payload_sha256"] = "f"*64
    elif damage == "envelope":
        records["trade"][0]["event_at_ms"] += 1
    elif damage == "clock_backwards":
        records["trade"][1]["received_at_ns"] -= 1000
    elif damage == "snapshot":
        snap["snapshot_id"] = "other"
    elif damage == "source_identity":
        manifest["cutoff_source"]["snapshot_id"] = "other"
    elif damage == "rank":
        snap["universe"][1]["rank"] = 1
    with pytest.raises(feature.MicrostructureEvidenceError):
        _compute(sample)


def _write_fixture(tmp_path):
    snapshot, manifest, records = _sample()
    snapshot_path = tmp_path / "open_r1.json"
    snapshot_path.write_text(json.dumps(snapshot))
    manifest["cutoff_source"].update(path=str(snapshot_path),file_sha256=hashlib.sha256(snapshot_path.read_bytes()).hexdigest())
    for channel, values in records.items():
        path = tmp_path / f"{channel}.jsonl.gz"
        path.write_bytes(gzip.compress(b"".join((json.dumps(r)+"\n").encode() for r in values),mtime=0))
        manifest["streams"][channel]["artifact"].update(path=str(path),size_bytes=path.stat().st_size,
                                                        sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return snapshot_path, manifest_path


def test_read_wrapper_real_snapshot_validation_and_no_file_changes(tmp_path):
    paths = _write_fixture(tmp_path)
    before = {p:p.read_bytes() for p in tmp_path.iterdir()}
    result = feature.read_recommend_microstructure(*paths)
    assert result["feature_evidence_valid"] is True
    assert result["provenance"]["inputs_unchanged"] is True
    assert len(result["provenance"]["generator_sources"]) == 4
    assert {p:p.read_bytes() for p in tmp_path.iterdir()} == before


@pytest.mark.parametrize("damage", ["gzip", "sha", "symlink", "source_changed"])
def test_read_wrapper_integrity_and_change_guards(tmp_path, monkeypatch, damage):
    paths = _write_fixture(tmp_path)
    raw = tmp_path / "trade.jsonl.gz"
    if damage in {"gzip", "sha"}:
        raw.write_bytes(b"corrupt")
        if damage == "gzip":
            doc = json.loads(paths[1].read_bytes())
            doc["streams"]["trade"]["artifact"].update(sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),size_bytes=7)
            paths[1].write_text(json.dumps(doc))
    elif damage == "symlink":
        target = tmp_path / "other.gz"
        raw.rename(target)
        raw.symlink_to(target)
    else:
        original = feature.compute_trade_imbalance
        def changed(*args, **kwargs):
            result = original(*args, **kwargs)
            paths[1].write_text(paths[1].read_text()+"\n")
            return result
        monkeypatch.setattr(feature,"compute_trade_imbalance",changed)
    with pytest.raises(feature.MicrostructureEvidenceError):
        feature.read_recommend_microstructure(*paths)


def test_public_api_does_not_mutate_returned_input_metadata():
    sample = _sample()
    result = _compute(sample)
    # Returned provenance must not expose aliases into caller-owned objects.
    result["provenance"]["raw_artifacts"]["trade"]["record_count"] = 999
    assert sample[1]["streams"]["trade"]["artifact"]["record_count"] == 1


def test_subscription_cannot_claim_a_coin_that_was_not_subscribed():
    sample = _sample()
    sample[1]["subscription"]["trade"][0]["codes"].pop()
    with pytest.raises(feature.MicrostructureEvidenceError, match="subscription universe"):
        _compute(sample)


def test_same_trade_identifier_in_different_markets_is_not_a_duplicate():
    sample = _sample([_trade(),_trade(coin=MARKETS[1],side="ASK")])
    result = _compute(sample)
    assert result["rows"][0][feature.FEATURE] == 1
    assert result["rows"][1][feature.FEATURE] == -1
    assert result["feature_available_rows"] == 2


def test_selected_universe_row_shuffle_does_not_change_original_rank_mapping():
    sample = _sample()
    expected = _compute(sample)
    sample[0]["universe"].reverse()
    snapshot = sample[0]
    payload = {k:v for k,v in snapshot.items() if k not in {"created_at","snapshot_id","payload_sha256","snapshot_path"}}
    snapshot["payload_sha256"] = feature._digest(payload)
    snapshot["snapshot_id"] = "recommend-"+snapshot["payload_sha256"][:20]
    for key in ("snapshot_id", "payload_sha256"):
        sample[1]["cutoff_source"][key] = snapshot[key]
    assert _compute(sample)["rows"] == expected["rows"]


def test_read_wrapper_accepts_parent_alias_but_not_relocated_raw(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    paths = _write_fixture(original)
    alias = tmp_path / "alias"
    alias.symlink_to(original, target_is_directory=True)
    result = feature.read_recommend_microstructure(alias / paths[0].name, alias / paths[1].name)
    assert result["feature_evidence_valid"] is True


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -1., 0., "10"])
def test_invalid_price_is_not_used_as_a_feature(value):
    sample = _sample([_trade(price=value)])
    result = _compute(sample)
    assert result["feature_evidence_valid"] is False
    assert "raw_parse_or_server_errors" in result["quality_reasons"]
    assert result["rows"][0][feature.FEATURE] is None


@pytest.mark.parametrize("field,value", [("subscribed_at_ns",True),("closed_at_ns","123"),
                                         ("first_ingress_seq",True),("last_ingress_seq",1.0)])
def test_manifest_time_and_ingress_types_are_not_coerced(field,value):
    sample = _sample()
    sample[1]["streams"]["trade"]["connections"][0][field] = value
    with pytest.raises(feature.MicrostructureEvidenceError):
        _compute(sample)


def test_missing_collector_source_evidence_is_unavailable():
    sample = _sample()
    sample[1]["source_code"]["at_start"]["files"].pop()
    result = _compute(sample)
    assert result["feature_evidence_valid"] is False
    assert "capture_source_provenance_unavailable" in result["quality_reasons"]


def test_one_shot_inputs_are_consumed_exactly_once_and_fully():
    sample = _sample()
    expected = _compute(sample)

    class OneShot:
        def __init__(self, values):
            self.values = values
            self.iterations = 0
            self.yielded = 0

        def __iter__(self):
            self.iterations += 1
            assert self.iterations == 1
            for value in self.values:
                self.yielded += 1
                yield value

    inputs = {key: OneShot(values) for key, values in sample[2].items()}
    actual = feature.compute_trade_imbalance(sample[0],sample[1],
                                             trade_records=inputs["trade"],orderbook_records=inputs["orderbook"])
    assert actual == expected
    assert inputs["trade"].yielded == 1 and inputs["orderbook"].yielded == 100


@pytest.mark.parametrize("channel,field", [("trade","record_count"),("trade","event_count"),
                                           ("orderbook","record_count"),("orderbook","event_count")])
def test_counter_mismatch_is_checked_after_stream_exhaustion(channel,field):
    sample = _sample()
    sample[1]["streams"][channel]["artifact"][field] += 1
    sample[2]["trade"] = iter(sample[2]["trade"])
    sample[2]["orderbook"] = iter(sample[2]["orderbook"])
    with pytest.raises(feature.MicrostructureEvidenceError,match="count mismatch"):
        _compute(sample)


def test_validation_generator_does_not_buffer_future_records():
    sample = _sample()
    seen = []

    def stream():
        for record in sample[2]["orderbook"]:
            seen.append(record["ingress_seq"])
            yield record

    validated = feature._validated_records(stream(),"orderbook",sample[1],set(MARKETS))
    next(validated)
    assert seen == [1]
    for _record in validated:
        pass
    assert len(seen) == 100


def test_compact_notional_sum_matches_fsum_without_storing_all_trades():
    import math
    values = [1e16, 1., 1., 10., .001] * 1000
    partials = []
    for value in values:
        feature._add_notional(partials,value)
    assert math.fsum(partials) == math.fsum(values)
    assert len(partials) < 10


def test_read_wrapper_does_not_materialize_raw_inputs_before_compute(tmp_path,monkeypatch):
    paths = _write_fixture(tmp_path)
    original = feature.compute_trade_imbalance

    def inspect(*args,**kwargs):
        for key in ("trade_records","orderbook_records"):
            assert iter(kwargs[key]) is kwargs[key]
            assert not isinstance(kwargs[key],list)
        return original(*args,**kwargs)

    monkeypatch.setattr(feature,"compute_trade_imbalance",inspect)
    assert feature.read_recommend_microstructure(*paths)["feature_evidence_valid"] is True


def test_truncated_gzip_fails_after_valid_records_without_publishing_report(tmp_path):
    paths = _write_fixture(tmp_path)
    path = tmp_path / "trade.jsonl.gz"
    path.write_bytes(path.read_bytes()[:-2])
    manifest = json.loads(paths[1].read_bytes())
    manifest["streams"]["trade"]["artifact"].update(size_bytes=path.stat().st_size,
                                                     sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    paths[1].write_text(json.dumps(manifest))
    with pytest.raises(feature.MicrostructureEvidenceError,match="truncated"):
        feature.read_recommend_microstructure(*paths)


def test_partially_consumed_gzip_iterator_closed_on_validation_failure(tmp_path,monkeypatch):
    paths = _write_fixture(tmp_path)
    original = feature._gzip_records
    closed = []

    def tracked(path):
        try:
            yield from original(path)
        finally:
            closed.append(path.name)

    def fail(*args,**kwargs):
        next(kwargs["orderbook_records"])
        raise feature.MicrostructureEvidenceError("injected after first read")

    monkeypatch.setattr(feature,"_gzip_records",tracked)
    monkeypatch.setattr(feature,"compute_trade_imbalance",fail)
    with pytest.raises(feature.MicrostructureEvidenceError,match="injected"):
        feature.read_recommend_microstructure(*paths)
    assert closed == ["orderbook.jsonl.gz"]

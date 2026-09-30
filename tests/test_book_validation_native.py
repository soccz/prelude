"""Native synthetic raw capture → canonical receipt/label → L1 comparison."""

from copy import deepcopy
import json

import pytest

import test_recommend_microstructure as micro
import test_recommend_microstructure_trial as fixtures
from ops import recommend_book_validation as io
from test_recommend_regime_forward import _at, _inputs, isolated as isolated  # noqa: F401


def inputs(tmp_path, monkeypatch):
    sample, book = fixtures._sample, micro._book

    def with_liquidity(*args, **kwargs):
        snapshot, manifest, records = sample(*args, **kwargs)
        for row in snapshot["universe"]:
            row["feature_values"]["f_log_qv"] = 20.0
        snapshot["feature_columns"] = list(snapshot["universe"][0]["feature_values"])
        snapshot["features"]["columns"] = snapshot["feature_columns"]
        snapshot["top3"] = deepcopy(snapshot["universe"][:3])
        digest = fixtures.snapshots._document_digest(snapshot)
        snapshot.update(payload_sha256=digest, snapshot_id=f"recommend-{digest[:20]}")
        manifest["cutoff_source"].update(
            snapshot_id=snapshot["snapshot_id"], payload_sha256=digest
        )
        return snapshot, manifest, records

    def with_pressure(coin, **kwargs):
        payload, clock = book(coin, **kwargs)
        if coin == micro.MARKETS[3]:
            payload["orderbook_units"][0]["bid_size"] = 20.0
            payload["total_bid_size"] = 20.0
        return payload, clock

    monkeypatch.setattr(fixtures, "_sample", with_liquidity)
    monkeypatch.setattr(micro, "_book", with_pressure)
    return _inputs(tmp_path, monkeypatch, complete=True)


def extract(paths, **kwargs):
    snapshot, _, trial_root, receipt_root, label_root = paths
    reference = fixtures.snapshots.load_snapshot(snapshot)
    return io.extract_day(
        "2026-10-01",
        _at("12:00:00", "2026-10-02"),
        expected_version=io.version_from_snapshot(reference),
        shortlist_root=trial_root,
        receipt_root=receipt_root,
        label_root=label_root,
        **kwargs,
    )


def test_actual_native_raw_receipt_and_label_chain(tmp_path, monkeypatch):
    paths = inputs(tmp_path, monkeypatch)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    result = extract(paths)
    row = result["result"]
    assert row["selection"]["control"] == list(micro.MARKETS[:3])
    assert row["selection"]["challenger"] == [micro.MARKETS[i] for i in (0, 1, 3)]
    assert row["metrics"]["challenger_minus_control"][
        "eod_return_net"
    ] == pytest.approx(0.17 / 3)
    assert row["metrics"]["challenger_minus_control"]["dn5"] == pytest.approx(-1 / 3)
    assert row["metrics"]["challenger_minus_control"]["up10"] == pytest.approx(1 / 3)
    assert row["entry_at"] == "2026-10-01T09:15:00+09:00"
    assert {
        str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()
    } == before


def test_selection_is_fixed_before_canonical_label_loader(tmp_path, monkeypatch):
    paths = inputs(tmp_path, monkeypatch)
    select, load = io.policy.fixed.select_top3, io.load_recommendation_evidence
    events = []

    def selected(*args, **kwargs):
        events.append("select")
        return select(*args, **kwargs)

    def labeled(*args, **kwargs):
        assert events == ["select"]
        events.append("labels")
        return load(*args, **kwargs)

    monkeypatch.setattr(io.policy.fixed, "select_top3", selected)
    monkeypatch.setattr(io, "load_recommendation_evidence", labeled)
    extract(paths)
    assert events == ["select", "labels", "select"]


@pytest.mark.parametrize("fault", ["version", "raw", "receipt"])
def test_native_corruption_and_mixed_model_are_blocked(tmp_path, monkeypatch, fault):
    paths = inputs(tmp_path, monkeypatch)
    if fault == "version":
        monkeypatch.setattr(io, "version_from_snapshot", lambda s: "other")
        snapshot, _, trial_root, receipt_root, label_root = paths
        with pytest.raises(ValueError, match="version"):
            io.extract_day(
                "2026-10-01",
                _at("12:00:00", "2026-10-02"),
                expected_version="fixed",
                shortlist_root=trial_root,
                receipt_root=receipt_root,
                label_root=label_root,
            )
        return
    if fault == "raw":
        path = paths[1].parent / "orderbook.jsonl.gz"
    else:
        path = paths[3] / "2026-10-01/open_r1.json"
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError):
        extract(paths)


@pytest.mark.parametrize(
    "fault", ["receipt_missing", "label_incomplete", "label_late", "not_forward"]
)
def test_unavailable_metadata_never_consumes_raw_budget(tmp_path, monkeypatch, fault):
    paths = inputs(tmp_path, monkeypatch)
    if fault == "receipt_missing":
        (paths[3] / "2026-10-01/open_r1.json").unlink()
    else:
        path = paths[4] / "2026-10-01/open_r1.json"
        document = json.loads(path.read_text())
        if fault == "label_incomplete":
            document["artifact_status"] = "incomplete"
        elif fault == "label_late":
            document["labeled_at"] = "2026-10-03T12:00:00+09:00"
        else:
            document["forward_eligible"] = False
        document["label_payload_sha256"] = io._artifact_digest(document)
        path.write_text(json.dumps(document))
    with pytest.raises(io.EvidenceUnavailable):
        extract(
            paths, before_raw=lambda: pytest.fail("unavailable day consumed raw budget")
        )

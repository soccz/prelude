from datetime import datetime, timezone

import pytest

from ops import dashboard_day_ahead as d


def sample(day, changed):
    a, b = ["A", "B", "C"], ["A", "B", "D"] if changed else ["A", "B", "C"]
    return {"date": day, "status": "comparable", "changed": {"recent": changed, "same_context": 0},
            "selected": {"recent": b, "same_context": a, "fixed_r1": a},
            "values": {p: {"whole_path_safe_up10": x} for p, x in
                       (("fixed_r1", 1/3), ("recent", 0), ("same_context", 1/3))}}


def test_paired_is_not_send_ready_or_improvement():
    rows = [sample(f"2026-11-{i}", int(i < 28)) for i in range(25, 29)]
    report = {"asof": "2026-11-30", "days": [*rows, {"date": "2026-11-29", "status": "late"}],
              "status_counts": {"comparable": 4, "late": 1}, "next_plan": "2026-12-01",
              "policies": {p: {"changed_dates": n, "paired_dates": 4}
                           for p, n in (("same_context", 0), ("recent", 3))}}
    clocks = iter([True, False, None, True])
    data = d._project(report, lambda _: next(clocks))
    payload = {"schema": d.SCHEMA, "status": "observed", "asof": report["asof"], "data": data}
    d.validate_day_ahead(payload, asof=report["asof"])
    assert (data["before_send"], data["not_before_send"], data["clock_unknown"]) == (2, 1, 1)
    assert data["policies"]["recent"] == {"changed": 3, "send_ready_changed": 1,
                                          "safe_up_delta_pp": pytest.approx(-100/3)}
    assert data["policies"]["same_context"]["safe_up_delta_pp"] is None
    data["clock_unknown"] = 0
    with pytest.raises(ValueError):
        d.validate_day_ahead(payload, asof=report["asof"])


@pytest.mark.parametrize("second,expected", [(9, True), (10, False), (11, False), (13, False)])
def test_score_must_predate_send_attempt_not_receipt_recording(monkeypatch, second, expected):
    def at(i):
        return f"2026-11-25T00:09:{i:02d}+00:00"
    snapshot = {"snapshot_id": "s", "created_at": at(1), "decision_completed_at": at(3)}
    record = {"snapshot": snapshot, "commit": {"score_durable_observed_at": at(second)},
              "plan": {"control_top3": ["a", "b", "c"]}}
    monkeypatch.setattr(d.season.forward, "_bound_record", lambda *a: record)
    monkeypatch.setattr(d.season.forward, "receipt_path", lambda *a, **k: "receipt")
    monkeypatch.setattr(d, "file_identity", lambda *a, **k: "identity")
    audit = {"date": "2026-11-25", "snapshot_id": "s", "evidence_manifest": {"files": {"receipt": "identity"}}}
    monkeypatch.setattr(d.season.forward, "read_delivery_receipt", lambda *a, **k: {
        "snapshot_id": "s", "delivery_ok": True, "sent_at": at(12), "attempted_at": at(10), "recorded_at": at(14)})
    ticket = {"state": "committed", "plan": {"payload_sha256": "p"},
              "durable": datetime.fromisoformat(at(0))}
    row = {"snapshot_id": "s", "known_before_entry_at": at(14), "plan_sha256": "p",
           "arms": dict.fromkeys(("fixed_r1", *d.POLICIES), "control"),
           "selected": dict.fromkeys(("fixed_r1", *d.POLICIES), ["a", "b", "c"])}
    assert d._send_clock("2026-11-25", ticket, row, {}, audit, datetime.fromisoformat(at(30))) is expected
    row["snapshot_id"] = "other"
    with pytest.raises(ValueError):
        d._send_clock("2026-11-25", ticket, row, {}, audit, datetime.fromisoformat(at(30)))


def test_absent_review_is_unknown_not_zero(monkeypatch):
    def missing():
        raise FileNotFoundError()
    monkeypatch.setattr(d.season, "inspect", missing)
    payload = d.build_day_ahead(asof="2026-11-24", now=datetime(2026, 11, 24, tzinfo=timezone.utc))
    assert payload["status"] == "unavailable" and payload["data"] is None
    d.validate_day_ahead(payload, asof="2026-11-24")


def test_today_card_must_match_delivered_snapshot_and_rank(monkeypatch):
    import notifier.delivery_receipt as receipts
    import signals.recommend_live_snapshot as snaps
    from scripts.validate_dashboard_assets import DashboardAssetError, _validate_delivered_radar
    day, sid, at = "2026-11-24", "recommend-" + "a" * 20, "2026-11-24T00:09:00+00:00"
    snapshot = {"snapshot_id": sid, "top3": [{"rank": 1, "coin": "KRW-X"}]}
    monkeypatch.setattr(snaps, "load_snapshot", lambda *args, **kwargs: snapshot)
    monkeypatch.setattr(receipts, "read_delivery_receipt", lambda *_: {
        "delivery_ok": True, "snapshot_id": sid, "sent_at": at})
    r = {"latest_radar_date": day, "latest_radar_snapshot_id": sid,
         "latest_radar": [{"rank": 1, "coin": "X", "date": day}]}
    s = {"channels": {"recommend": r}, "current_system": {"live": {"open": {
        "sent_at": at, "state": "delivered_candidates", "candidate_count": 1}}}}
    _validate_delivered_radar(s, day)
    for field, value in (("latest_radar_date", "2026-11-23"), ("latest_radar_snapshot_id", "other"),
                         ("latest_radar", [{"rank": 2, "coin": "X", "date": day}])):
        with pytest.raises(DashboardAssetError):
            _validate_delivered_radar({**s, "channels": {"recommend": {**r, field: value}}}, day)

"""Read-only day-ahead evidence for the PIN dashboard. No policy selection."""
from collections import Counter
from datetime import date, datetime, timedelta
import math

from ops import season_shadow as season
from ops.artifact_provenance import file_identity, strict_json_object
from ops.dashboard_current import KST, _require

SCHEMA = "prelude_dashboard_day_ahead.v1"
POLICIES = ("recent", "same_context")
EXCLUDED = {"uncertain", "missing_plan", "late", "invalid_evidence", "unavailable"}
ERRORS = (ValueError, KeyError, TypeError, OSError, RuntimeError, OverflowError)


def _send_clock(day, ticket, row, evaluation, audit, now):
    """Receipt recording is evidence, not a predictor. Never infer send from entry."""
    f = season.forward
    record = f._bound_record(evaluation, day, f.shortlist.DEFAULT_TRIAL_ROOT, now)
    snapshot = record["snapshot"]
    receipt_path = f.receipt_path(snapshot, root=f.DEFAULT_RECEIPT_ROOT)
    receipt_id = audit["evidence_manifest"]["files"]["receipt"]
    _require(audit["date"] == day and audit["snapshot_id"] == row["snapshot_id"] and
             file_identity(receipt_path, root=season.ROOT) == receipt_id)
    receipt = f.read_delivery_receipt(snapshot, root=f.DEFAULT_RECEIPT_ROOT)
    _require(receipt and receipt["delivery_ok"] is True and
             receipt["snapshot_id"] == snapshot["snapshot_id"] == row["snapshot_id"]
             and file_identity(receipt_path, root=season.ROOT) == receipt_id)
    _require(ticket["state"] == "committed" and ticket["plan"]["payload_sha256"] == row["plan_sha256"])
    for name in ("fixed_r1", *POLICIES):
        _require(record["plan"][row["arms"][name] + "_top3"] == row["selected"][name])
    aware = season.policy.aware
    ready = max(ticket["durable"], aware(snapshot["created_at"]), aware(snapshot["decision_completed_at"]),
                aware(record["commit"]["score_durable_observed_at"]))
    attempt, sent, recorded = (aware(receipt[key]) for key in ("attempted_at", "sent_at", "recorded_at"))
    _require(attempt <= sent <= recorded <= now and
             max(ready, recorded) == aware(row["known_before_entry_at"]))
    # Equal timestamps or a score becoming durable during the API call cannot
    # prove it could have been included in the already composed message.
    return ready < attempt


def _project(report, clock):
    end = date.fromisoformat(report["asof"])
    rows = report["days"]
    _require([r["date"] for r in rows] == [
        (season.START + timedelta(days=i)).isoformat() for i in range((end - season.START).days)
    ])
    counts = Counter(r["status"] for r in rows)
    _require(set(counts) <= EXCLUDED | {"comparable"} and dict(counts) == report["status_counts"])
    paired = [r for r in rows if r["status"] == "comparable"]
    clocks = [clock(r) for r in paired]
    verified = [r for r, ok in zip(paired, clocks, strict=True) if ok is True]
    stats = {}
    for name in POLICIES:
        changed = []
        for r in paired:
            n = r["changed"][name]
            _require(type(n) is int and 0 <= n <= 3 and
                     n == len(set(r["selected"][name]) - set(r["selected"]["fixed_r1"])))
            if n:
                changed.append(r)
        _require(report["policies"][name]["changed_dates"] == len(changed)
                 and report["policies"][name]["paired_dates"] == len(paired))
        delta = []
        for r in verified:
            if r["changed"][name]:
                a, b = (r["values"][arm]["whole_path_safe_up10"] for arm in (name, "fixed_r1"))
                _require(all(type(x) in (float, int) and 0 <= x <= 1 for x in (a, b)))
                delta.append(a - b)
        stats[name] = {"changed": len(changed), "send_ready_changed": len(delta),
                       "safe_up_delta_pp": 100 * sum(delta) / len(delta) if delta else None}
    return {"next_plan": report["next_plan"], "paired": len(paired),
            "last_paired": paired[-1]["date"] if paired else None,
            "excluded": {k: v for k, v in counts.items() if k != "comparable"},
            "before_send": len(verified), "not_before_send": clocks.count(False),
            "clock_unknown": clocks.count(None), "policies": stats}


def build_day_ahead(*, asof, now=None):
    now = (now or datetime.now(KST)).astimezone(KST)
    payload = {"schema": SCHEMA, "asof": asof, "status": "unavailable", "data": None}
    if asof != now.date().isoformat():
        return payload
    try:
        report = season.inspect()
        _require(report["asof"] == asof)
        evaluation = None
        audits = {}
        if any(r["status"] == "comparable" for r in report["days"]):
            # No replay and no fresh timestamps. A fallback evaluation with
            # no identical archived generation is not proof of a send clock.
            path = season.review.DEFAULT_OUTPUT
            if report["source"] and report["source"] == file_identity(path, root=season.ROOT):
                native = strict_json_object(path)
                season.review._check_document(native, season.review.SCHEMA)
                candidate = native["evaluations"]["shortlist"]
                if (candidate["payload_sha256"] == report["native_evaluation_sha256"]
                        and report["source"] == file_identity(path, root=season.ROOT)):
                    evaluation = candidate
                    audits = {r["date"]: r for r in candidate["dates"]}
                    _require(len(audits) == len(candidate["dates"]))

        def clock(row):
            if evaluation is not None:
                try:
                    day = row["date"]
                    return _send_clock(day, season.read_ticket(season.STORE, day), row, evaluation, audits[day], now)
                except ERRORS:
                    pass
            return None

        result = {**payload, "status": "observed", "data": _project(report, clock)}
        validate_day_ahead(result, asof=asof)
        return result
    except ERRORS:
        return payload


def validate_day_ahead(payload, *, asof):
    _require(set(payload) == {"schema", "asof", "status", "data"}
             and payload["schema"] == SCHEMA and payload["asof"] == asof)
    data = payload["data"]
    if payload["status"] == "unavailable":
        _require(data is None)
        return
    _require(payload["status"] == "observed" and set(data) == {
        "next_plan", "paired", "last_paired", "excluded", "before_send",
        "not_before_send", "clock_unknown", "policies"})
    today = date.fromisoformat(asof)
    _require(data["next_plan"] == (today + timedelta(days=1)).isoformat())
    n = data["paired"]
    counts = [data[k] for k in ("before_send", "not_before_send", "clock_unknown")]
    _require(type(n) is int and all(type(x) is int and x >= 0 for x in counts) and sum(counts) == n)
    _require(set(data["excluded"]) <= EXCLUDED
             and all(type(x) is int and x > 0 for x in data["excluded"].values())
             and n + sum(data["excluded"].values()) == max(0, (today - season.START).days))
    _require((n == 0 and data["last_paired"] is None) or
             (n > 0 and season.START <= date.fromisoformat(data["last_paired"]) < today))
    _require(set(data["policies"]) == set(POLICIES))
    for item in data["policies"].values():
        _require(set(item) == {"changed", "send_ready_changed", "safe_up_delta_pp"})
        changed, ready, delta = item["changed"], item["send_ready_changed"], item["safe_up_delta_pp"]
        _require(type(changed) is int and type(ready) is int and
                 0 <= ready <= changed <= n and ready <= data["before_send"])
        _require((delta is None and ready == 0) or (ready > 0 and type(delta) in (int, float)
                 and math.isfinite(delta) and -100 <= delta <= 100))

from copy import deepcopy
from datetime import datetime, timedelta
import json
import subprocess
from types import SimpleNamespace

import pytest

from ops import dashboard_book_forward as dashboard
from ops import recommend_book_forward as forward
from test_recommend_book_forward import review_days

NOW = datetime.fromisoformat("2026-09-30T22:00:00+09:00")


def payload(*, observed=False):
    now = NOW + timedelta(days=31) if observed else NOW
    days = review_days() if observed else []
    calendar = [{"date": r["date"], "status": "evaluated"} for r in days]
    report = forward._report({"payload_sha256": "a" * 64}, calendar, days, now)
    native = {
        "report": report,
        "today_record": "ready" if observed else "not_due",
        "attention_required": False,
    }
    return dashboard.project(native, now=now), now


def test_waiting_is_null_not_zero_and_observed_contains_no_private_fields():
    waiting, now = payload()
    assert waiting["paired_dates"] == 0 and waiting["difference"] is None
    assert waiting["verdict"] == "waiting"
    dashboard.validate_book_forward(waiting, asof=now.date().isoformat(), now=now)
    observed, now = payload(observed=True)
    assert observed["paired_dates"] == 30 and observed["verdict"] == "review_candidate"
    assert observed["automatic_promotion"] is False
    text = json.dumps(observed)
    assert all(
        s not in text for s in ("KRW-", "snapshot_path", "input_files", "receipt_root")
    )


@pytest.mark.parametrize(
    "fault",
    [
        "extra",
        "count",
        "infinite",
        "boolean",
        "promotion",
        "unknown",
        "stale",
        "false_review",
        "today",
    ],
)
def test_invalid_projection_rejected(fault):
    value, now = payload(observed=True)
    if fault == "extra":
        value["secret"] = "never publish"
    elif fault == "count":
        value["changed_dates"] = 31
    elif fault == "infinite":
        value["difference"]["eod_return_net"] = float("inf")
    elif fault == "boolean":
        value["paired_dates"] = True
    elif fault == "promotion":
        value["automatic_promotion"] = True
    elif fault == "unknown":
        value["checks"]["<script>"] = True
    elif fault == "stale":
        value["generated_at"] = (now - timedelta(hours=7)).isoformat()
    elif fault == "false_review":
        value["checks"]["joint_down_up_safe_net"] = False
    else:
        value["today_record"] = "missing"
    with pytest.raises(ValueError):
        dashboard.validate_book_forward(value, asof=now.date().isoformat(), now=now)


@pytest.mark.parametrize("fault", ["exit", "timeout", "bad_json", "large", "invalid"])
def test_bounded_child_failure_is_unavailable_not_success(monkeypatch, fault):
    value, _ = payload()
    if fault == "invalid":
        value["automatic_promotion"] = True

    def run(args, **kwargs):
        assert (
            kwargs["timeout"] == 30
            and "--refresh" not in args
            and "--record" not in args
        )
        if fault == "timeout":
            raise subprocess.TimeoutExpired(args, 30)
        return SimpleNamespace(
            returncode=2 if fault == "exit" else 0,
            stdout=b"x" * 32769
            if fault == "large"
            else b"bad"
            if fault == "bad_json"
            else json.dumps(value).encode(),
        )

    monkeypatch.setattr(dashboard.subprocess, "run", run)
    result = dashboard.build_book_forward(asof="2026-09-30", now=NOW)
    assert result["status"] == "unavailable" and result["paired_dates"] is None


def test_good_child_and_historical_request(monkeypatch):
    value, _ = payload()
    monkeypatch.setattr(
        dashboard.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=0, stdout=json.dumps(value).encode()
        ),
    )
    assert dashboard.build_book_forward(asof="2026-09-30", now=NOW) == value
    monkeypatch.setattr(
        dashboard.subprocess, "run", lambda *a, **k: pytest.fail("historical probe")
    )
    assert (
        dashboard.build_book_forward(asof="2026-09-29", now=NOW)["status"]
        == "historical_not_observed"
    )


def test_projection_cannot_mutate_native_review():
    value, _ = payload()
    before = deepcopy(value)
    dashboard.validate_book_forward(value, asof="2026-09-30", now=NOW)
    assert value == before

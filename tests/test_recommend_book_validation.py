"""Synthetic future calendar/cache checks; no live input mutation or training."""

import copy
import fcntl
import json
from datetime import datetime, timedelta

import pytest
import requests

from ops import recommend_book_validation as io
from ops.artifact_provenance import canonical_json_bytes, file_identity
from signals import recommend_book_validation as model
from test_recommend_book_pressure import rows


def at(day="2026-09-30", hour=12):
    return datetime.fromisoformat(f"{day}T{hour:02d}:00:00+09:00")


def bundle(day, identity, *, halt=False):
    values = rows()
    predictors = [
        {"coin": r["coin"], **{k: r[k] for k in io.book.PREDICTORS}} for r in values
    ]
    outcomes = [
        {
            "coin": r["coin"],
            "label_status": r["label_status"],
            **{k: r[k] for k in io.book.OUTCOMES},
        }
        for r in values
    ]
    if halt:
        outcomes[-1].update(
            label_status="halted_no_observations", **dict.fromkeys(io.book.OUTCOMES)
        )
    end = (datetime.fromisoformat(day) + timedelta(days=1)).date().isoformat()
    result = model.evaluate_day(
        day,
        predictors,
        outcomes,
        entry_at=day + "T09:15:00+09:00",
        end_at=end + "T09:15:00+09:00",
    )
    return {
        "date": day,
        "predictors": predictors,
        "outcomes": outcomes,
        "result": result,
        "input_files": [identity],
        "r1_version": "version",
    }


@pytest.fixture(autouse=True)
def no_live(monkeypatch):
    monkeypatch.setenv("PRELUDE_FORBID_TELEGRAM", "1")
    monkeypatch.setattr(
        requests.sessions.Session,
        "request",
        lambda *a, **k: pytest.fail("network access"),
    )


@pytest.fixture
def ready(tmp_path, monkeypatch):
    anchor = tmp_path / "reference.json"
    anchor.write_text("{}")
    monkeypatch.setattr(io, "ROOT", tmp_path)
    monkeypatch.setattr(io, "_sources", lambda: [file_identity(anchor, root=tmp_path)])
    monkeypatch.setattr(
        io, "load_snapshot", lambda *a, **k: {"created_at": at().isoformat()}
    )
    monkeypatch.setattr(io, "version_from_snapshot", lambda d: "version")
    root = tmp_path / "_workspace/book_validation"
    io.initialize(anchor, root=root, now=at())
    calls = []

    def extract(day, now, *, before_raw, **kwargs):
        before_raw()
        calls.append(day)
        return bundle(day, file_identity(anchor, root=tmp_path))

    monkeypatch.setattr(io, "extract_day", extract)
    return root, anchor, calls


def labels_exist(tmp_path, *days):
    for day in days:
        path = tmp_path / "output/recommend_score_labels" / day / "open_r1.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")


def reseal(path, change):
    doc = json.loads(path.read_text())
    change(doc)
    doc.pop("payload_sha256", None)
    path.write_bytes(canonical_json_bytes(io.native._seal(doc)))


@pytest.mark.parametrize(
    "clock,n",
    [
        ("2026-09-30T23:59:59+09:00", 0),
        ("2026-10-01T23:59:59+09:00", 0),
        ("2026-10-01T15:00:00+00:00", 1),
        ("2026-10-31T12:00:00+09:00", 30),
        ("2026-12-01T12:00:00+09:00", 30),
    ],
)
def test_calendar_never_reuses_development_dates_and_caps_frozen_window(clock, n):
    actual = model.due_dates(datetime.fromisoformat(clock))
    assert len(actual) == n
    if actual:
        assert actual[0] == "2026-10-01"
    with pytest.raises(ValueError):
        model.due_dates(datetime(2026, 10, 2))


def test_late_initialization_and_wrong_namespace_rejected(ready, tmp_path):
    root, anchor, _ = ready
    with pytest.raises(ValueError, match="too late"):
        io.initialize(anchor, root=root, now=at("2026-10-01", 0))
    with pytest.raises(ValueError, match="namespace"):
        io.initialize(anchor, root=tmp_path / "output/recommend_snapshots", now=at())
    original = (root / "design.json").read_bytes()
    with pytest.raises(FileExistsError):
        io.initialize(anchor, root=root, now=at())
    assert (root / "design.json").read_bytes() == original


def test_before_first_outcome_null_is_not_zero_and_no_extraction(ready):
    root, _, calls = ready
    r = io.refresh(root=root, now=at())
    assert not calls and r["status"] == "waiting_for_first_outcome"
    assert r["summary"]["metrics"] is None and r["summary"]["paired_dates"] == 0
    assert (
        not r["attention_required"]
        and not r["automatic_promotion"]
        and r["prospective_pick_records"] == 0
    )
    assert io.inspect(root=root, now=at()) == r
    assert json.loads(json.dumps(r, allow_nan=False)) == r


def test_budget_resume_reuses_immutable_days_and_recomputes_statistics(ready, tmp_path):
    root, _, calls = ready
    labels_exist(tmp_path, "2026-10-01", "2026-10-02", "2026-10-03")
    r = io.refresh(root=root, now=at("2026-10-04"))
    assert (
        r["summary"]["paired_dates"] == 2 and r["calendar"][-1]["status"] == "deferred"
    )
    assert r["attention_required"]
    saved = (root / "2026-10-01.json").read_bytes()
    again = io.refresh(root=root, now=at("2026-10-04", 13))
    assert calls == ["2026-10-01", "2026-10-02", "2026-10-03"]
    assert (root / "2026-10-01.json").read_bytes() == saved
    assert again["summary"]["paired_dates"] == 3 and not again["attention_required"]
    assert again["summary"]["metrics"]["challenger_minus_control"]["mean"][
        "eod_return_net"
    ] == pytest.approx(-0.05)
    assert (
        again["summary"]["metrics"]["challenger"]["observed_date_block3_ci95"] is None
    )
    assert io.inspect(root=root, now=at("2026-10-04", 13)) == again


def test_missing_labels_do_not_spend_budget_or_turn_into_zero_results(ready, tmp_path):
    root, _, calls = ready
    labels_exist(tmp_path, "2026-10-02", "2026-10-03")
    r = io.refresh(root=root, now=at("2026-10-04"))
    assert calls == ["2026-10-02", "2026-10-03"]
    assert (
        r["calendar"][0]["reason"] == "label_missing"
        and r["summary"]["paired_dates"] == 2
    )
    assert r["attention_required"]


def test_unavailable_native_record_does_not_starve_later_dates(
    ready, tmp_path, monkeypatch
):
    root, anchor, calls = ready
    labels_exist(tmp_path, "2026-10-01", "2026-10-02", "2026-10-03")

    def extract(day, now, *, before_raw, **kwargs):
        if day == "2026-10-01":
            raise io.EvidenceUnavailable("native_record_missing")
        before_raw()
        calls.append(day)
        return bundle(day, file_identity(anchor, root=tmp_path))

    monkeypatch.setattr(io, "extract_day", extract)
    r = io.refresh(root=root, now=at("2026-10-04"))
    assert calls == ["2026-10-02", "2026-10-03"] and r["summary"]["paired_dates"] == 2


def test_halted_candidate_not_replaced_and_no_effect_counted(
    ready, tmp_path, monkeypatch
):
    root, anchor, _ = ready
    labels_exist(tmp_path, "2026-10-01")

    def extract(day, now, **kwargs):
        return bundle(day, file_identity(anchor, root=tmp_path), halt=True)

    monkeypatch.setattr(io, "extract_day", extract)
    r = io.refresh(root=root, now=at("2026-10-02"))
    assert (
        r["calendar"][0]["status"] == "excluded" and r["summary"]["paired_dates"] == 0
    )
    assert "COIN099" in r["daily"][0]["selection"]["challenger"]
    assert io.inspect(root=root, now=at("2026-10-02")) == r


@pytest.mark.parametrize(
    "fault", ["mean", "ci", "count", "promotion", "prospective", "calendar", "extra"]
)
def test_forged_derived_report_is_blocked_even_if_resealed(ready, tmp_path, fault):
    root, _, _ = ready
    labels_exist(tmp_path, "2026-10-01")
    io.refresh(root=root, now=at("2026-10-02"))

    def change(d):
        if fault == "mean":
            d["summary"]["metrics"]["challenger"]["mean"]["up10"] = 0.9
        elif fault == "ci":
            d["summary"]["metrics"]["challenger"]["observed_date_block3_ci95"] = {}
        elif fault == "count":
            d["summary"]["paired_dates"] = True
        elif fault == "promotion":
            d["automatic_promotion"] = True
        elif fault == "prospective":
            d["prospective_pick_records"] = 1
        elif fault == "calendar":
            d["calendar"][0]["date"] = "2026-09-30"
        else:
            d["unrecognized"] = "unexpected"

    reseal(root / "report.json", change)
    with pytest.raises(ValueError):
        io.inspect(root=root, now=at("2026-10-02"))


@pytest.mark.parametrize("fault", ["plan", "mean", "future", "source"])
def test_changed_cache_or_bound_input_is_never_silently_repaired(
    ready, tmp_path, fault
):
    root, anchor, calls = ready
    labels_exist(tmp_path, "2026-10-01")
    io.refresh(root=root, now=at("2026-10-02"))
    if fault == "source":
        anchor.write_text("changed")
    else:

        def change(d):
            if fault == "plan":
                d["result"]["selection"]["challenger"] = d["result"]["selection"][
                    "control"
                ]
            elif fault == "mean":
                d["result"]["metrics"]["challenger"]["up10"] = 0.9
            else:
                d["generated_at"] = "2026-12-01T12:00:00+09:00"

        reseal(root / "2026-10-01.json", change)
    old_report = (root / "report.json").read_bytes()
    with pytest.raises(ValueError):
        io.refresh(root=root, now=at("2026-10-02", 13))
    assert (root / "report.json").read_bytes() == old_report and len(calls) == 1


@pytest.mark.parametrize(
    "wall", [at("2026-10-02", 11), at("2026-10-02", 19), at("2026-10-03", 12)]
)
def test_future_stale_and_overdue_reports_rejected(ready, tmp_path, wall):
    root, _, _ = ready
    labels_exist(tmp_path, "2026-10-01")
    io.refresh(root=root, now=at("2026-10-02"))
    with pytest.raises(ValueError):
        io.inspect(root=root, now=wall)


def test_publication_failure_preserves_day_and_retry_does_not_reparse(
    ready, tmp_path, monkeypatch
):
    root, _, calls = ready
    labels_exist(tmp_path, "2026-10-01")
    publish = io._replace_report
    monkeypatch.setattr(
        io, "_replace_report", lambda *a: (_ for _ in ()).throw(OSError("disk"))
    )
    with pytest.raises(OSError):
        io.refresh(root=root, now=at("2026-10-02"))
    assert (root / "2026-10-01.json").is_file() and not (root / "report.json").exists()
    monkeypatch.setattr(io, "_replace_report", publish)
    assert (
        io.refresh(root=root, now=at("2026-10-02", 13))["summary"]["paired_dates"] == 1
    )
    assert len(calls) == 1


def test_symlink_report_and_root_cannot_overwrite_other_files(ready, tmp_path):
    root, anchor, _ = ready
    target = tmp_path / "untouched.json"
    target.write_text("untouched")
    (root / "report.json").symlink_to(target)
    with pytest.raises(OSError):
        io.refresh(root=root, now=at())
    alias = tmp_path / "_workspace/alias"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError):
        io.initialize(anchor, root=alias, now=at())
    assert target.read_text() == "untouched"


def test_selection_and_prior_day_immune_to_future_outcome_changes(tmp_path):
    path = tmp_path / "bound"
    path.write_text("source")
    one = bundle("2026-10-01", file_identity(path, root=tmp_path))
    before = copy.deepcopy(one)
    two = bundle("2026-10-02", file_identity(path, root=tmp_path))
    for r in two["outcomes"]:
        r["eod_return_net"] *= 10
    new = model.evaluate_day(
        two["date"],
        two["predictors"],
        two["outcomes"],
        entry_at=two["result"]["entry_at"],
        end_at=two["result"]["end_at"],
    )
    assert new["selection"] == two["result"]["selection"] and one == before


def test_concurrent_refresh_rejected_without_mutation_and_inspection_read_only(ready):
    root, _, calls = ready
    with (root / ".refresh.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = {p.name: p.read_bytes() for p in root.iterdir()}
        with pytest.raises(BlockingIOError):
            io.refresh(root=root, now=at())
        assert {p.name: p.read_bytes() for p in root.iterdir()} == before and not calls
    io.refresh(root=root, now=at())
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    io.inspect(root=root, now=at())
    assert {p.name: p.read_bytes() for p in root.iterdir()} == before


def test_gap_keeps_observed_means_but_does_not_fabricate_basket_path(tmp_path):
    path = tmp_path / "bound"
    path.write_text("source")
    identity = file_identity(path, root=tmp_path)
    days = [bundle(day, identity)["result"] for day in ("2026-10-01", "2026-10-03")]
    result = model.summarize(days)
    assert result["paired_dates"] == 2 and result["metrics"] is not None
    assert result["basket_proxy"]["control"] == {
        "status": "gaps_or_overlaps",
        "metrics": None,
    }


def test_wrong_date_or_short_interval_rejected(tmp_path):
    path = tmp_path / "bound"
    path.write_text("source")
    row = bundle("2026-10-01", file_identity(path, root=tmp_path))
    for day, end in (
        ("2026-09-30", row["result"]["end_at"]),
        (row["date"], row["result"]["entry_at"]),
    ):
        with pytest.raises(ValueError, match="24h interval"):
            model.evaluate_day(
                day,
                row["predictors"],
                row["outcomes"],
                entry_at=row["result"]["entry_at"],
                end_at=end,
            )

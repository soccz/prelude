"""Synthetic pre-entry publication → labels → review; never real operations."""

from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path
import sqlite3

import pytest

from ops import recommend_book_forward as io
from ops.artifact_provenance import canonical_json_bytes
from signals import recommend_book_readiness as policy
from test_book_validation_native import inputs
from test_recommend_regime_forward import _at, _inputs, isolated as isolated  # noqa: F401


@pytest.fixture
def ready(tmp_path, monkeypatch):
    paths = inputs(tmp_path, monkeypatch)
    reference = _inputs(tmp_path / "reference", monkeypatch, day="2026-09-30")[0]
    root = tmp_path / "_workspace/forward"
    original_root = io._root
    monkeypatch.setattr(
        io, "_root", lambda value: root if Path(value) == root else original_root(value)
    )
    io.initialize(reference, root=root, now=_at("20:00:00", "2026-09-30"))
    return root, paths


def publish(ready, *, clock=None):
    root, paths = ready
    return io.record(
        root=root, shortlist_root=paths[2], receipt_root=paths[3], now_fn=clock or _at
    )


def score_path(root):
    return root / "scores/2026-10-01/score.json"


def reseal(path, mutate):
    value = json.loads(path.read_text())
    mutate(value)
    value.pop("payload_sha256", None)
    path.write_bytes(canonical_json_bytes(io.native._seal(value)))


def test_no_labels_are_read_to_record_and_retry_cannot_change_score(ready, monkeypatch):
    root, paths = ready
    labels = paths[4] / "2026-10-01/open_r1.json"
    labels.unlink()
    monkeypatch.setattr(
        io, "load_recommendation_evidence", lambda *a, **k: pytest.fail("outcomes read")
    )
    before = paths[0].read_bytes()
    assert publish(ready) == {"status": "ready", "reused": False}
    frozen = score_path(root).read_bytes()
    record = io.read_score("2026-10-01", root=root, now=_at("09:14:00"))
    assert record["score"]["selection"]["changed_picks"] == 1
    assert record["score"]["selection"]["selected_original_ranks"] == [1, 2, 4]
    assert not labels.exists()
    assert publish(ready, clock=lambda: _at("15:00:00"))["reused"]
    assert score_path(root).read_bytes() == frozen and paths[0].read_bytes() == before


@pytest.mark.parametrize(
    "wall,status",
    [("09:14:59.999999", "ready"), ("09:15:00", "late"), ("09:15:01", "late")],
)
def test_fsync_observation_not_start_time_is_deadline(ready, wall, status):
    root, _ = ready
    stamps = iter([_at(), _at("09:14:00"), _at(wall)])
    assert publish(ready, clock=lambda: next(stamps))["status"] == status
    assert (
        io.read_score("2026-10-01", root=root, now=_at("10:00:00"))["status"] == status
    )


def test_never_backfill_a_missing_pre_entry_record(ready):
    with pytest.raises(io.EvidenceUnavailable, match="deadline"):
        publish(ready, clock=lambda: _at("09:16:00"))
    assert not score_path(ready[0]).exists()


def test_partial_publication_is_never_healed_with_later_commit(ready, monkeypatch):
    root, _ = ready
    original = io.native._publish_new

    def interrupted(directory, name, doc):
        if name == "commit.json":
            raise OSError("synthetic interruption")
        original(directory, name, doc)

    with monkeypatch.context() as patch:
        patch.setattr(io.native, "_publish_new", interrupted)
        with pytest.raises(OSError):
            publish(ready)
    before = score_path(root).read_bytes()
    assert publish(ready, clock=lambda: _at("16:00:00"))["status"] == "uncertain"
    assert before == score_path(root).read_bytes()
    assert not score_path(root).with_name("commit.json").exists()


@pytest.mark.parametrize(
    "fault", ["selection", "design", "clock", "commit", "entry", "context", "source"]
)
def test_resealed_corruption_rejected(ready, fault, monkeypatch):
    root, _ = ready
    publish(ready)
    if fault == "source":
        monkeypatch.setattr(io, "_sources", lambda: [])
    elif fault == "commit":
        reseal(
            score_path(root).with_name("commit.json"),
            lambda d: d.update(score_payload_sha256="0" * 64),
        )
    else:
        mutations = {
            "selection": lambda d: d["selection"].update(challenger=["KRW-FAKE"] * 3),
            "design": lambda d: d.update(design_sha256="0" * 64),
            "clock": lambda d: d.update(planned_at="2026-10-02T09:00:00+09:00"),
            "entry": lambda d: d.update(entry_at="2026-10-01T09:30:00+09:00"),
            "context": lambda d: d["context"].update(state="invented"),
        }
        reseal(score_path(root), mutations[fault])
    with pytest.raises(ValueError):
        io.read_score("2026-10-01", root=root, now=_at("10:00:00"))


def test_prestart_waiting_readonly_missing_and_stale_status(ready):
    root, _ = ready
    now = _at("20:00:00", "2026-09-30")
    report = io.refresh(root=root, now=now)
    assert report["review"]["verdict"] == "waiting"
    assert report["review"]["summary"]["metrics"] is None
    assert report["review"]["execution"]["delays"] is None
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert not io.inspect(root=root, now=now)["attention_required"]
    assert before == {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="stale"):
        io.inspect(root=root, now=now + timedelta(hours=7))
    missing = io.refresh(root=root, now=_at("12:00:00", "2026-10-02"))
    assert missing["calendar"] == [{"date": "2026-10-01", "status": "missing"}]
    assert (
        missing["attention_required"]
        and missing["review"]["summary"]["paired_dates"] == 0
    )


def test_future_initialize_and_wrong_namespace_are_blocked(ready, tmp_path):
    root, paths = ready
    with pytest.raises(ValueError, match="too late"):
        io.initialize(paths[0], root=root, now=_at())
    with pytest.raises(ValueError, match="namespace"):
        io._root(tmp_path / "output/recommend_snapshots")


def synthetic_diagnostic(score, evidence, **kwargs):
    """Used only by cache wiring tests; real SQLite tests live separately."""
    from test_recommend_book_execution import paths_from_canonical

    paths = paths_from_canonical(score, evidence)
    return {
        "status": "evaluated",
        "paths": paths,
        "metrics": io.execution.aggregate(paths, score["selection"]),
    }


def test_end_to_end_cached_evaluation_and_resealed_report_detection(ready, monkeypatch):
    root, paths = ready
    publish(ready)
    original = io.evaluate_day
    monkeypatch.setattr(io.execution, "evaluate", synthetic_diagnostic)
    monkeypatch.setattr(
        io, "evaluate_day", lambda r, **k: original(r, label_root=paths[4], **k)
    )
    now = _at("12:00:00", "2026-10-02")
    report = io.refresh(root=root, now=now)
    assert report["calendar"] == [{"date": "2026-10-01", "status": "evaluated"}]
    result = report["review"]
    assert result["summary"]["paired_dates"] == result["execution"]["paired_dates"] == 1
    assert result["verdict"] == "continue_observing" and result["deployable"] is False
    assert result["summary"]["metrics"]["challenger_minus_control"]["mean"][
        "eod_return_net"
    ] == pytest.approx(0.17 / 3)
    before = (root / "2026-10-01.json").read_bytes()
    monkeypatch.setattr(
        io, "evaluate_day", lambda *a, **k: pytest.fail("cached date re-evaluated")
    )
    io.refresh(root=root, now=now)
    assert (root / "2026-10-01.json").read_bytes() == before
    # Today's missing score is operationally separate from yesterday's results.
    assert io.inspect(root=root, now=now)["attention_required"]
    reseal(
        root / "report.json", lambda d: d["review"].update(verdict="review_candidate")
    )
    with pytest.raises(ValueError, match="arithmetic"):
        io.inspect(root=root, now=now)


def test_rule_continues_after_first_30_day_review_window():
    assert len(policy.due_dates(_at("12:00:00", "2026-11-03"))) == 33


def test_native_record_to_real_sqlite_to_cache_and_review_without_execution_mock(
    ready, monkeypatch, tmp_path
):
    root, paths = ready
    publish(ready)
    saved = io.read_score("2026-10-01", root=root, now=_at("09:14:00"))["score"]
    selected = set(saved["selection"]["control"]) | set(
        saved["selection"]["challenger"]
    )
    ranks = {r["coin"]: r["rank"] for r in saved["predictors"]}
    db = tmp_path / "execution.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE candles (market TEXT,timestamp TEXT,open REAL,high REAL,low REAL,close REAL)"
        )
        for coin in selected | {"KRW-BTC"}:
            rank = ranks.get(coin, 0)
            low, high, closing = (
                (94.0, 100.0, 94.0)
                if rank == 3
                else (100.0, 111.0, 111.0)
                if rank == 4
                else (100.0, 100.0, 100.0)
            )
            for index in range(-1, 100):
                at = (_at("09:15:00") + timedelta(minutes=index * 15)).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                bar = (
                    (100.0, 100.0, 100.0, 100.0)
                    if index < 0
                    else (100.0, high, low, closing)
                    if index == 0
                    else (closing, closing, closing, closing)
                )
                conn.execute(
                    "INSERT INTO candles VALUES (?,?,?,?,?,?)", (coin, at, *bar)
                )
    original = io.evaluate_day
    monkeypatch.setattr(
        io,
        "evaluate_day",
        lambda r, **k: original(r, label_root=paths[4], db_path=db, **k),
    )
    now = _at("12:00:00", "2026-10-02")
    report = io.refresh(root=root, now=now)
    assert (
        report["review"]["summary"]["paired_dates"]
        == report["review"]["execution"]["paired_dates"]
        == 1
    )
    assert report["review"]["execution"]["delays"]["0"]["challenger_minus_control"][
        "mean"
    ]["eod_return_net"] == pytest.approx(0.17 / 3)
    io.inspect(root=root, now=now)
    reseal(
        root / "2026-10-01.json", lambda d: d["outcomes"][0].update(eod_return_net=0.99)
    )
    with pytest.raises(ValueError, match="canonical"):
        io.inspect(root=root, now=now)


def test_namespace_symlink_and_refresh_lock_are_rejected(ready):
    import fcntl

    root, _ = ready
    # The locked file belongs only to this synthetic test, never the live lock.
    with (root / ".refresh.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            io.refresh(root=root, now=_at("20:00:00", "2026-09-30"))
    score_path(root).parent.mkdir(parents=True)
    target = root / "target.json"
    target.write_text("{}")
    score_path(root).symlink_to(target)
    with pytest.raises((OSError, ValueError)):
        publish(ready)


def test_atomic_report_failure_keeps_daily_evidence_and_retry_recovers(
    ready, monkeypatch
):
    root, paths = ready
    publish(ready)
    original = io.evaluate_day
    monkeypatch.setattr(io.execution, "evaluate", synthetic_diagnostic)
    monkeypatch.setattr(
        io, "evaluate_day", lambda r, **k: original(r, label_root=paths[4], **k)
    )
    now = _at("12:00:00", "2026-10-02")
    with monkeypatch.context() as patch:

        def fail(*a, **k):
            raise OSError("synthetic report failure")

        patch.setattr(io.replay, "_replace_report", fail)
        with pytest.raises(OSError):
            io.refresh(root=root, now=now)
    assert (root / "2026-10-01.json").exists()
    monkeypatch.setattr(
        io, "evaluate_day", lambda *a, **k: pytest.fail("recompute cached day")
    )
    assert io.refresh(root=root, now=now)["review"]["summary"]["paired_dates"] == 1


def review_days(n=30, *, upside=1.0, downside=-1.0, gain=0.02):
    from test_recommend_book_validation import bundle

    # Pure-review fixture: controlled metrics, not claimed canonical outcomes.
    rows = []
    for index in range(n):
        day = (policy.START + timedelta(days=index)).isoformat()
        result = bundle(day, {})["result"]
        m = result["metrics"]
        delta = {key: 0.0 for key in io.execution.METRICS}
        delta.update(
            up10=upside, dn5=downside, whole_path_safe_up10=1.0, eod_return_net=gain
        )
        m["challenger_minus_control"] = delta
        m["challenger"]["eod_return_net"] = 0.03
        m["control"]["eod_return_net"] = 0.01
        m["challenger_minus_matched_random_expectation"]["eod_return_net"] = 0.02
        result["selection"]["changed_picks"] = 1
        result["selection"]["selected_outside_top10"] = 0
        rows.append(
            {
                "date": day,
                "result": result,
                "context": {"state": "broad" if index % 2 else "narrow"},
                "execution": {
                    "status": "evaluated",
                    "metrics": {
                        str(d): deepcopy(
                            {
                                a: m[a]
                                for a in (
                                    "control",
                                    "challenger",
                                    "challenger_minus_control",
                                )
                            }
                        )
                        for d in (0, 15, 30)
                    },
                },
            }
        )
    return rows


@pytest.mark.parametrize(
    "n,up,dn,gain,verdict",
    [
        (0, 1.0, -1.0, 0.02, "waiting"),
        (29, 1.0, -1.0, 0.02, "continue_observing"),
        (30, 1.0, -1.0, 0.02, "review_candidate"),
        (30, -1.0, -1.0, 0.02, "do_not_adopt"),
        (30, 1.0, 1.0, 0.02, "do_not_adopt"),
        (30, 1.0, -1.0, -0.01, "do_not_adopt"),
        (30, 1.0, -1.0, 0.001, "continue_observing"),
    ],
)
def test_joint_goal_and_cost_not_just_thirty_days(n, up, dn, gain, verdict):
    result = policy.review(review_days(n, upside=up, downside=dn, gain=gain))
    assert result["verdict"] == verdict
    assert result["deployable"] is result["automatic_promotion"] is False


@pytest.mark.parametrize(
    "fault", ["no_changes", "one_context", "delay_failure", "unknown_context"]
)
def test_robustness_and_coverage_cannot_be_assumed(fault):
    rows = review_days()
    for row in rows:
        if fault == "no_changes":
            row["result"]["selection"]["changed_picks"] = 0
        if fault == "one_context":
            row["context"]["state"] = "one"
        if fault == "unknown_context":
            row["context"]["state"] = None
        if fault == "delay_failure":
            row["execution"]["metrics"]["30"]["challenger_minus_control"][
                "eod_return_net"
            ] = -0.01
    assert policy.review(rows)["verdict"] == "continue_observing"

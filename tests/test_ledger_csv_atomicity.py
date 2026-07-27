from __future__ import annotations

import json
import logging
import multiprocessing as mp
import time
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ledger import csv_store


def _shadow_candidate(coin: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "market": coin,
                "decision": "WATCH_ONLY",
                "primary_setups": ["S02"],
            }
        ]
    )


def _shadow_append_worker(
    ledger_path: str,
    date: str,
    coin: str,
    start: Any,
    results: Any,
) -> None:
    from ledger.shadow import append_shadow_ledger

    start.wait()
    try:
        result = append_shadow_ledger(
            _shadow_candidate(coin),
            pd.Timestamp(f"{date} 09:05:00"),
            ledger_path,
            "distribution",
        )
        results.put(("ok", result))
    except Exception as exc:  # pragma: no cover - surfaced in parent assertion
        results.put(("error", repr(exc)))


def _distribution_alert(coin: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "market": coin,
                "primary_setups": ["S02"],
                "btc_regime": "bull_quiet",
            }
        ]
    )


def _distribution_append_worker(
    ledger_path: str,
    date: str,
    coin: str,
    start: Any,
    results: Any,
) -> None:
    from scripts.predict_today_distribution import append_to_paper_ledger

    start.wait()
    try:
        result = append_to_paper_ledger(
            _distribution_alert(coin),
            pd.Timestamp(f"{date} 09:05:00"),
            ledger_path,
        )
        results.put(("append", result))
    except Exception as exc:  # pragma: no cover - surfaced in parent assertion
        results.put(("error", repr(exc)))


def _paper_close_worker(
    ledger_path: str,
    start: Any,
    results: Any,
) -> None:
    from scripts import close_paper_ledger

    close_paper_ledger.compute_realized = lambda *args, **kwargs: {
        "status": "closed",
        "next_open": 100.0,
        "next_high": 106.0,
        "next_low": 98.0,
        "next_close": 103.0,
        "next_max_return_pct": 6.0,
        "next_min_return_pct": -2.0,
        "next_close_return_pct": 3.0,
        "hit_h2": 1,
        "hit_h6": 1,
        "hit_h5": 0,
    }
    start.wait()
    try:
        close_paper_ledger.close_ledger_file(
            ledger_path,
            "unused.db",
            pd.Timestamp("2026-05-08 10:00:00"),
            False,
            logging.getLogger("atomic-close-worker"),
        )
        results.put(("close", 1))
    except Exception as exc:  # pragma: no cover - surfaced in parent assertion
        results.put(("error", repr(exc)))


def _v2_result(date: str, coin: str) -> dict:
    from scripts.pump_detector_v2_today import PUMP_V2_RULE

    return {
        "asof": date,
        "model_id": "pump_hunter_v2",
        "rule_version": "pump_detector_v2",
        "rule": PUMP_V2_RULE,
        "feature_date": f"{pd.Timestamp(date) - pd.Timedelta(days=1):%Y-%m-%d} 09:00:00",
        "btc_regime": "bear_quiet",
        "universe_n": 100,
        "binance_status": "ok",
        "n_candidates": 1,
        "candidates": [
            {
                "market": coin,
                "rank": 1,
                "score": 0.9,
                "entry_open": 100.0,
                "roc_7d": 12.0,
                "roc_7d_rank": 0.95,
                "atr_pct_14": 0.08,
                "log_return_1d": 0.01,
                "b_vol_surge": 3.2,
                "b_ret_1d": 0.05,
                "liq_rank_daily": 10,
                "btc_regime": "bear_quiet",
                "rule_id": "roc7_rank+bn_volsurge",
            }
        ],
        "oos": {
            "hit_pct": 8.1,
            "baseline_hit_pct": 5.6,
            "base_rate_pct": 1.4,
            "net_tp5sl3_pct": -0.36,
        },
    }


def _v2_append_worker(
    ledger_path: str,
    started: Any,
    finished: Any,
    results: Any,
) -> None:
    from scripts.pump_detector_v2_today import append_ledger

    started.set()
    try:
        append_ledger(
            _v2_result("2026-07-26", "KRW-NEW"),
            ledger_path,
            False,
            delivery_ok=True,
            sent_at="2026-07-26T00:05:00+00:00",
            receipt_path="decision-new.json",
        )
        results.put(("v2-append", 1))
    except Exception as exc:  # pragma: no cover - surfaced in parent assertion
        results.put(("error", repr(exc)))
    finally:
        finished.set()


def _recommend_close_worker(
    ledger_path: str,
    entered_close: Any,
    release_close: Any,
    results: Any,
) -> None:
    from ledger.path_quality import PathAssessment
    from scripts import close_recommend_ledger

    timestamps = tuple(pd.date_range("2026-07-24 09:15:00", periods=96, freq="15min"))
    assessment = PathAssessment(
        bars=[(100.0, 100.0, 100.0, 100.0)] * 96,
        timestamps=timestamps,
        path_complete=True,
        path_quality="complete",
        raw_bars=96,
        expected_bars=96,
        benchmark_bars=96,
    )

    def blocked_assessment(*args, **kwargs):
        entered_close.set()
        if not release_close.wait(timeout=20):
            raise TimeoutError("test did not release close transaction")
        return assessment

    close_recommend_ledger._load_15m_path = blocked_assessment
    close_recommend_ledger._daily_pump20 = lambda *args, **kwargs: {
        "status": "ok",
        "pump20_hit": 0,
    }
    close_recommend_ledger.evaluate_exit_variants = lambda *args, **kwargs: None
    try:
        close_recommend_ledger.close_recommend_ledger(
            ledger_path,
            pd.Timestamp("2026-07-26 10:00:00"),
            False,
            logging.getLogger("atomic-v2-close-worker"),
            decision_date=pd.Timestamp("2026-07-24"),
        )
        results.put(("v2-close", 1))
    except Exception as exc:  # pragma: no cover - surfaced in parent assertion
        results.put(("error", repr(exc)))


def _atomic_writer_worker(
    ledger_path: str,
    start: Any,
    generations: int,
    rows_per_generation: int,
) -> None:
    start.wait()
    for generation in range(1, generations + 1):
        frame = pd.DataFrame(
            {
                "generation": [generation] * rows_per_generation,
                "value": list(range(rows_per_generation)),
            }
        )
        with csv_store.ledger_lock(ledger_path):
            csv_store.atomic_write_csv(frame, ledger_path)


def _join_clean(processes: list[mp.Process]) -> None:
    for process in processes:
        process.join(timeout=30)
        assert not process.is_alive(), f"worker hung: pid={process.pid}"
        assert process.exitcode == 0


def test_concurrent_shadow_appends_do_not_lose_distinct_snapshots(tmp_path):
    ctx = mp.get_context("spawn")
    path = tmp_path / "shadow.csv"
    start = ctx.Event()
    results = ctx.Queue()
    processes = [
        ctx.Process(
            target=_shadow_append_worker,
            args=(
                str(path),
                f"2026-05-{day:02d}",
                f"KRW-{day:03d}",
                start,
                results,
            ),
        )
        for day in range(1, 7)
    ]
    for process in processes:
        process.start()
    start.set()
    _join_clean(processes)

    outcomes = [results.get(timeout=5) for _ in processes]
    assert outcomes.count(("ok", 1)) == len(processes)
    ledger = pd.read_csv(path)
    assert len(ledger) == len(processes)
    assert ledger["date"].nunique() == len(processes)


def test_concurrent_same_snapshot_rechecks_idempotency_inside_lock(tmp_path):
    ctx = mp.get_context("spawn")
    path = tmp_path / "shadow.csv"
    start = ctx.Event()
    results = ctx.Queue()
    processes = [
        ctx.Process(
            target=_shadow_append_worker,
            args=(str(path), "2026-05-25", f"KRW-{index:03d}", start, results),
        )
        for index in range(6)
    ]
    for process in processes:
        process.start()
    start.set()
    _join_clean(processes)

    outcomes = [results.get(timeout=5) for _ in processes]
    assert sum(value for status, value in outcomes if status == "ok") == 1
    assert sum(status == "error" for status, _ in outcomes) == len(processes) - 1
    assert all(
        "shadow snapshot identity conflict" in value
        for status, value in outcomes
        if status == "error"
    )
    ledger = pd.read_csv(path)
    assert len(ledger) == 1
    assert ledger.loc[0, "date"] == "2026-05-25"


def test_concurrent_append_and_close_preserve_both_updates(tmp_path):
    from scripts.predict_today_distribution import append_to_paper_ledger

    path = tmp_path / "paper.csv"
    assert (
        append_to_paper_ledger(
            _distribution_alert("KRW-OLD"),
            pd.Timestamp("2026-05-05 09:05:00"),
            str(path),
        )
        == 1
    )

    ctx = mp.get_context("spawn")
    start = ctx.Event()
    results = ctx.Queue()
    processes = [
        ctx.Process(
            target=_distribution_append_worker,
            args=(str(path), "2026-05-08", "KRW-NEW", start, results),
        ),
        ctx.Process(
            target=_paper_close_worker,
            args=(str(path), start, results),
        ),
    ]
    for process in processes:
        process.start()
    start.set()
    _join_clean(processes)
    outcomes = [results.get(timeout=5) for _ in processes]
    assert sorted(outcomes) == [("append", 1), ("close", 1)]

    ledger = pd.read_csv(path).set_index("coin")
    assert set(ledger.index) == {"KRW-OLD", "KRW-NEW"}
    assert ledger.loc["KRW-OLD", "status"] == "closed"
    assert ledger.loc["KRW-NEW", "status"] == "entered"


def test_v2_append_and_recommend_close_share_one_transaction_lock(tmp_path):
    from scripts.pump_detector_v2_today import append_ledger

    path = tmp_path / "pump-v2.csv"
    append_ledger(
        _v2_result("2026-07-24", "KRW-OLD"),
        str(path),
        False,
        delivery_ok=True,
        sent_at="2026-07-24T00:05:00+00:00",
        receipt_path="decision-old.json",
    )

    ctx = mp.get_context("spawn")
    entered_close = ctx.Event()
    release_close = ctx.Event()
    append_started = ctx.Event()
    append_finished = ctx.Event()
    results = ctx.Queue()
    close_process = ctx.Process(
        target=_recommend_close_worker,
        args=(str(path), entered_close, release_close, results),
    )
    append_process = ctx.Process(
        target=_v2_append_worker,
        args=(str(path), append_started, append_finished, results),
    )
    processes = [close_process, append_process]
    close_process.start()
    assert entered_close.wait(timeout=10), "closer never entered locked assessment"
    append_process.start()
    assert append_started.wait(timeout=10), "v2 appender never started"
    try:
        # The closer deliberately pauses after reading while still holding the
        # ledger lock. A writer using another/no lock would finish here and its
        # row would then be overwritten by the closer's stale DataFrame.
        time.sleep(0.25)
        assert not append_finished.is_set()
    finally:
        release_close.set()
    _join_clean(processes)
    outcomes = [results.get(timeout=5) for _ in processes]
    assert sorted(outcomes) == [("v2-append", 1), ("v2-close", 1)]

    ledger = pd.read_csv(path).set_index("coin")
    assert set(ledger.index) == {"KRW-OLD", "KRW-NEW"}
    assert ledger.loc["KRW-OLD", "status"] == "closed"
    assert ledger.loc["KRW-OLD", "realized_pct"] == pytest.approx(-0.15)
    assert ledger.loc["KRW-NEW", "status"] == "open"
    assert bool(ledger.loc["KRW-NEW", "delivery_ok"]) is True


def test_preopen_append_rechecks_duplicate_key_inside_transaction(tmp_path):
    from scripts.predict_preopen_trigger import (
        PAPER_LEDGER_COLS,
        append_to_paper_ledger,
    )

    path = tmp_path / "preopen.csv"
    first = {column: pd.NA for column in PAPER_LEDGER_COLS}
    first.update(
        {
            "date": "2026-05-05",
            "coin": "KRW-AAA",
            "status": "entered",
            "notes": "",
        }
    )
    duplicate = dict(first)
    duplicate["status"] = "closed"
    different_coin = dict(first)
    different_coin["coin"] = "KRW-BBB"

    assert append_to_paper_ledger([first], str(path)) == 1
    assert append_to_paper_ledger([duplicate], str(path)) == 0
    assert append_to_paper_ledger([different_coin], str(path)) == 0
    ledger = pd.read_csv(path)
    assert len(ledger) == 1
    assert ledger.loc[0, "status"] == "entered"


def test_atomic_replace_never_exposes_partial_csv_to_unlocked_reader(tmp_path):
    ctx = mp.get_context("spawn")
    path = tmp_path / "ledger.csv"
    rows_per_generation = 200
    pd.DataFrame(
        {
            "generation": [0] * rows_per_generation,
            "value": list(range(rows_per_generation)),
        }
    ).to_csv(path, index=False)

    start = ctx.Event()
    writer = ctx.Process(
        target=_atomic_writer_worker,
        args=(str(path), start, 30, rows_per_generation),
    )
    writer.start()
    start.set()
    reads = 0
    while writer.is_alive():
        observed = pd.read_csv(path)
        assert len(observed) == rows_per_generation
        assert observed["generation"].nunique() == 1
        assert observed["value"].tolist() == list(range(rows_per_generation))
        reads += 1
        time.sleep(0.002)
    _join_clean([writer])
    assert reads > 0


def test_replace_failure_preserves_existing_csv_and_cleans_temp(tmp_path, monkeypatch):
    path = tmp_path / "ledger.csv"
    original = b"date,coin\n2026-05-01,KRW-OLD\n"
    path.write_bytes(original)

    def fail_replace(source, target, **_kwargs):
        raise OSError("injected replace failure")

    monkeypatch.setattr(csv_store.os, "replace", fail_replace)
    with csv_store.ledger_lock(path):
        with pytest.raises(OSError, match="injected replace failure"):
            csv_store.atomic_write_csv(
                pd.DataFrame([{"date": "2026-05-02", "coin": "KRW-NEW"}]),
                path,
            )

    assert path.read_bytes() == original
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_replace_failure_preserves_existing_diagnostic_json(tmp_path, monkeypatch):
    path = tmp_path / "distribution_log_20260501.json"
    original = {"state": "complete", "generation": 1}
    path.write_text(json.dumps(original), encoding="utf-8")

    def fail_replace(source, target, **_kwargs):
        raise OSError("injected JSON replace failure")

    monkeypatch.setattr(csv_store.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected JSON replace failure"):
        csv_store.atomic_write_json({"state": "partial"}, path)

    assert json.loads(path.read_text(encoding="utf-8")) == original
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_atomic_json_normalizes_nonfinite_values_to_strict_null(tmp_path):
    path = tmp_path / "diagnostic.json"
    csv_store.atomic_write_json(
        {
            "nan": float("nan"),
            "positive_inf": float("inf"),
            "negative_inf": float("-inf"),
            "numpy_nan": np.float64("nan"),
            "missing": pd.NA,
            "nested": [1.0, np.float64(2.0)],
        },
        path,
    )

    raw = path.read_text(encoding="utf-8")
    assert "NaN" not in raw
    assert "Infinity" not in raw
    assert json.loads(raw) == {
        "nan": None,
        "positive_inf": None,
        "negative_inf": None,
        "numpy_nan": None,
        "missing": None,
        "nested": [1.0, 2.0],
    }


def test_atomic_write_rejects_symlink_target_without_touching_referent(
    tmp_path,
):
    referent = tmp_path / "outside.csv"
    referent.write_text("keep\n", encoding="utf-8")
    path = tmp_path / "ledger.csv"
    path.symlink_to(referent)

    with pytest.raises(OSError, match="regular file"):
        csv_store.atomic_write_csv(pd.DataFrame([{"value": 1}]), path)

    assert path.is_symlink()
    assert referent.read_text(encoding="utf-8") == "keep\n"


def test_idempotent_shadow_append_rejects_symlink_ledger(tmp_path):
    from ledger.shadow import append_shadow_ledger

    referent = tmp_path / "real-shadow.csv"
    candidate = _shadow_candidate("KRW-AAA")
    asof = pd.Timestamp("2026-05-25 09:05:00")
    assert append_shadow_ledger(
        candidate,
        asof,
        str(referent),
        "distribution",
    ) == 1
    alias = tmp_path / "shadow.csv"
    alias.symlink_to(referent)

    with pytest.raises(OSError, match="regular file"):
        append_shadow_ledger(
            candidate,
            asof,
            str(alias),
            "distribution",
        )

    assert alias.is_symlink()
    assert len(pd.read_csv(referent)) == 1


def test_atomic_write_rejects_target_replacement_during_serialization(
    tmp_path,
):
    path = tmp_path / "ledger.csv"
    path.write_text("old\n", encoding="utf-8")

    def replace_target(handle):
        handle.write("writer\n")
        intruder = tmp_path / "intruder.csv"
        intruder.write_text("intruder\n", encoding="utf-8")
        intruder.replace(path)

    with pytest.raises(OSError, match="changed during write"):
        csv_store._atomic_write_text(path, replace_target)

    assert path.read_text(encoding="utf-8") == "intruder\n"
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_atomic_write_rejects_parent_replacement_during_serialization(
    tmp_path,
):
    parent = tmp_path / "ledger-dir"
    parent.mkdir()
    path = parent / "ledger.csv"
    path.write_text("old\n", encoding="utf-8")
    moved_parent = tmp_path / "moved-ledger-dir"

    def replace_parent(handle):
        handle.write("writer\n")
        parent.rename(moved_parent)
        parent.mkdir()
        (parent / path.name).write_text("intruder\n", encoding="utf-8")

    with pytest.raises(OSError, match="parent changed during write"):
        csv_store._atomic_write_text(path, replace_parent)

    assert path.read_text(encoding="utf-8") == "intruder\n"
    assert (moved_parent / path.name).read_text(encoding="utf-8") == "old\n"
    assert list(moved_parent.glob(f".{path.name}.*.tmp")) == []

"""Bounded spawn startup and operation phases for concurrency regression tests."""

from __future__ import annotations

import os
import sqlite3
import time
from threading import BrokenBarrierError


def isolate_append_child():
    """Spawn does not inherit pytest monkeypatches; CSV appends need no DB."""
    os.environ["PRELUDE_FORBID_TELEGRAM"] = "1"

    def forbid_database(*args, **kwargs):
        raise AssertionError("isolated append worker must not access SQLite")

    sqlite3.connect = forbid_database


def stop_processes(processes):
    """Reap every started child, including after partial startup or assertions."""
    started = [process for process in processes if process.pid is not None]
    for process in started:
        if process.is_alive():
            process.terminate()
    deadline = time.monotonic() + 3
    for process in started:
        process.join(timeout=max(0, deadline - time.monotonic()))
    for process in started:
        if process.is_alive():
            process.kill()
    deadline = time.monotonic() + 3
    for process in started:
        process.join(timeout=max(0, deadline - time.monotonic()))
    assert not any(process.is_alive() for process in started), "worker cleanup failed"


def join_processes(processes, *, timeout=30):
    """One group-wide operation deadline, not a separate allowance per child."""
    deadline = time.monotonic() + timeout
    for process in processes:
        process.join(timeout=max(0, deadline - time.monotonic()))
        assert not process.is_alive(), f"worker operation timed out: pid={process.pid}"
        assert (
            process.exitcode == 0
        ), f"worker failed: pid={process.pid}, exit={process.exitcode}"


def run_concurrent(processes, start, *, startup_timeout=90, operation_timeout=30):
    """Release workers together after imports; retain the original work budget.

    CPUQuota=50% is shared by all spawn children. Their interpreter/import
    startup is not ledger lock/write latency. A separate finite 90s initial
    startup budget covers that phase; hangs still fail, without retries.
    Each target must wait on ``start`` after its imports and test guards.
    """
    deadline = time.monotonic() + startup_timeout
    try:
        for process in processes:
            process.start()
        try:
            start.wait(timeout=max(0, deadline - time.monotonic()))
        except BrokenBarrierError as exc:
            raise AssertionError(
                "worker startup timed out or readiness barrier broke"
            ) from exc
        join_processes(processes, timeout=operation_timeout)
    finally:
        stop_processes(processes)

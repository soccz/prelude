"""Deterministic spawn latency/failure injection; no real services or data."""

from threading import BrokenBarrierError

import pytest

import process_helpers as harness


class Clock:
    now = 0.0

    def monotonic(self):
        return self.now


class Process:
    def __init__(self, clock, *, work=1, crash=0, start_error=False, ignore_term=False):
        self.clock = clock
        self.work = work
        self.crash = crash
        self.start_error = start_error
        self.ignore_term = ignore_term
        self.pid = None
        self.exitcode = None
        self.alive = False
        self.terminated = self.killed = False
        self.joins = []

    def start(self):
        if self.start_error:
            raise OSError("injected spawn failure")
        self.pid = id(self)
        self.alive = True

    def is_alive(self):
        return self.alive

    def join(self, timeout):
        self.joins.append(timeout)
        if self.alive:
            spent = min(self.work, timeout)
            self.clock.now += spent
            self.work -= spent
            if self.work <= 0:
                self.alive = False
                self.exitcode = self.crash

    def terminate(self):
        self.terminated = True
        if not self.ignore_term:
            self.alive = False
            self.exitcode = -15

    def kill(self):
        self.killed = True
        self.alive = False
        self.exitcode = -9


class Barrier:
    def __init__(self, clock, delay):
        self.clock, self.delay = clock, delay
        self.waits = []

    def wait(self, timeout):
        self.waits.append(timeout)
        self.clock.now += min(timeout, self.delay)
        if self.delay > timeout:
            raise BrokenBarrierError


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(harness.time, "monotonic", clock.monotonic)
    return clock


def test_slow_startup_does_not_consume_original_operation_budget(clock):
    # The former v2 barrier(10s) fails on imports alone, without ledger work.
    with pytest.raises(BrokenBarrierError):
        Barrier(clock, 40).wait(timeout=10)
    processes = [Process(clock) for _ in range(4)]
    barrier = Barrier(clock, 40)
    harness.run_concurrent(processes, barrier, operation_timeout=10)
    assert barrier.waits == [90]
    assert processes[0].joins[0] == 10
    assert processes[-1].joins[0] == 7
    assert all(process.exitcode == 0 for process in processes)


def test_startup_still_has_a_finite_group_deadline(clock):
    processes = [Process(clock) for _ in range(6)]
    with pytest.raises(AssertionError, match="startup"):
        harness.run_concurrent(processes, Barrier(clock, 91))
    assert clock.now == 90
    assert all(process.terminated and not process.is_alive() for process in processes)


def test_operation_hang_is_not_retried_or_given_startup_budget(clock):
    processes = [Process(clock, work=100), Process(clock)]
    with pytest.raises(AssertionError, match="operation timed out"):
        harness.run_concurrent(processes, Barrier(clock, 40), operation_timeout=10)
    assert clock.now == 50
    assert all(process.terminated and not process.is_alive() for process in processes)


def test_operation_deadline_is_shared_by_all_children(clock):
    processes = [Process(clock, work=6), Process(clock, work=6)]
    with pytest.raises(AssertionError, match="operation timed out"):
        harness.run_concurrent(processes, Barrier(clock, 0), operation_timeout=10)
    assert [process.joins[0] for process in processes] == [10, 4]


def test_crashed_worker_fails_and_other_children_are_reaped(clock):
    processes = [Process(clock, crash=2), Process(clock, work=100)]
    with pytest.raises(AssertionError, match="exit=2"):
        harness.run_concurrent(processes, Barrier(clock, 0))
    assert processes[1].terminated
    assert not any(process.is_alive() for process in processes)


def test_partial_spawn_failure_cleans_only_started_children(clock):
    processes = [Process(clock), Process(clock, start_error=True), Process(clock)]
    with pytest.raises(OSError, match="spawn failure"):
        harness.run_concurrent(processes, Barrier(clock, 0))
    assert processes[0].terminated
    assert processes[1].pid is processes[2].pid is None
    assert not processes[1].joins and not processes[2].joins


def test_cleanup_escalates_to_kill_with_bounded_group_wait(clock):
    processes = [Process(clock, work=100, ignore_term=True) for _ in range(6)]
    with pytest.raises(AssertionError, match="startup"):
        harness.run_concurrent(processes, Barrier(clock, 91))
    assert clock.now == 93
    assert all(process.terminated and process.killed for process in processes)
    assert not any(process.is_alive() for process in processes)


def test_append_child_restores_notification_and_database_guards(monkeypatch):
    monkeypatch.setenv("PRELUDE_FORBID_TELEGRAM", "")
    # Restore the session's original guard after exercising the child reset.
    monkeypatch.setattr(harness.sqlite3, "connect", harness.sqlite3.connect)
    harness.isolate_append_child()
    assert harness.os.environ["PRELUDE_FORBID_TELEGRAM"] == "1"
    with pytest.raises(AssertionError, match="must not access SQLite"):
        harness.sqlite3.connect(":memory:")

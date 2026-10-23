# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Reaping the worker process when a replica stops.

The parent owns the only handle on its worker: the HTTP endpoint closes when
the parent exits, so a worker that outlives it is invisible from outside while
its device memory is still allocated. These check that stopping escalates once
per stage, never twice, and that a worker which survives everything is reported
rather than passed off as a clean stop.

Every stage of a shutdown is bounded, including the one in front of all of
them -- uvicorn waiting out the requests already accepted.
"""

from __future__ import annotations

import asyncio

import pytest

from pypto_serving.serving.engine import async_engine
from pypto_serving.serving.engine.async_engine import (
    AsyncLLMEngine,
    EngineConfig,
    ReplicaEngineCore,
)
from pypto_serving.serving.utils.env import graceful_shutdown_timeout_seconds


class _FakeProcess:
    """A worker that dies after a given number of reap attempts, or never."""

    def __init__(self, *, dies_after: int | None = 0) -> None:
        self.pid = 4242
        self.signals: list[str] = []
        self.joins: list[float] = []
        self._dies_after = dies_after
        self._reaps = 0

    def join(self, timeout=None):
        self.joins.append(timeout)
        self._reaps += 1

    def is_alive(self) -> bool:
        if self._dies_after is None:
            return True
        return self._reaps <= self._dies_after

    def terminate(self) -> None:
        self.signals.append("SIGTERM")

    def kill(self) -> None:
        self.signals.append("SIGKILL")


class _FakeQueue:
    def __init__(self) -> None:
        self.sent: list[object] = []

    def put(self, item) -> None:
        self.sent.append(item)


def _core(process, queue=None) -> ReplicaEngineCore:
    """A core with only the attributes the shutdown path touches.

    Built without __init__ on purpose: constructing a real one needs a model,
    a device and a scheduler, none of which this behaviour depends on.
    """
    core = ReplicaEngineCore.__new__(ReplicaEngineCore)
    core.config = EngineConfig(device_ids=(4, 5))
    core._input_queue = queue
    core._worker_process = process
    core._output_queue = None
    core._profile_output_queue = None
    core._worker_stranded = False
    core._running = True
    core._loop_task = None
    return core


def test_a_worker_that_stops_when_asked_is_not_signalled():
    process = _FakeProcess(dies_after=0)
    queue = _FakeQueue()

    assert _core(process, queue)._shutdown_worker(timeout=0.01) is True
    assert len(queue.sent) == 1, "the cooperative shutdown command goes first"
    assert process.signals == [], "a worker that stopped must not be signalled"


def test_a_worker_that_ignores_the_command_is_terminated_then_killed():
    """One SIGTERM, then one SIGKILL. Never the same signal twice.

    A signal already pending is not queued again, and a worker blocked in an
    uninterruptible device call is no more reachable by the second than by the
    first -- so the escalation is what repeats, not the signal.
    """
    process = _FakeProcess(dies_after=2)
    core = _core(process, _FakeQueue())

    assert core._shutdown_worker(timeout=0.01) is True
    assert process.signals == ["SIGTERM", "SIGKILL"]


def test_a_worker_that_survives_sigkill_is_reported_not_forgotten():
    """The stop is not clean, and saying so is the whole point.

    Returning True here is what strands device memory: the endpoint closes with
    the parent, so the replica looks stopped from outside while its cards are
    still held, and the next launch onto them fails for want of memory.
    """
    process = _FakeProcess(dies_after=None)
    core = _core(process, _FakeQueue())

    assert core._shutdown_worker(timeout=0.01) is False
    assert process.signals == ["SIGTERM", "SIGKILL"]


def test_every_stage_is_bounded():
    """A stop must not hang on a worker that never exits."""
    process = _FakeProcess(dies_after=None)
    _core(process, _FakeQueue())._shutdown_worker(timeout=0.01)

    assert len(process.joins) == 3, "one bounded wait per stage"
    assert all(timeout is not None and timeout > 0 for timeout in process.joins)


def test_the_handles_are_dropped_even_when_the_worker_survives():
    """Whatever happened, this core must not keep talking to that worker."""
    core = _core(_FakeProcess(dies_after=None), _FakeQueue())
    core._shutdown_worker(timeout=0.01)

    assert core._worker_process is None
    assert core._input_queue is None
    assert core._output_queue is None
    assert core._profile_output_queue is None


def test_stopping_a_core_that_never_started_is_clean():
    core = _core(None)
    assert core._shutdown_worker(timeout=0.01) is True
    assert core._shutdown_worker(timeout=0.01) is True


def test_a_worker_that_outlived_a_stop_is_still_reported_by_the_next_one():
    """The handle is gone by then, so without memory the second stop looks clean.

    This is a real sequence, not a hypothetical: a core whose initialization
    fails shuts its worker down on the way out, and the engine then calls stop()
    on every core -- which must not report reclaimed devices that the first call
    could not reclaim.
    """
    core = _core(_FakeProcess(dies_after=None), _FakeQueue())

    assert core._shutdown_worker(timeout=0.01) is False
    assert core._worker_process is None, "the handle is dropped either way"
    assert core._shutdown_worker(timeout=0.01) is False


# --- the wait in front of the stop -------------------------------------------


def test_the_graceful_shutdown_wait_is_bounded_by_default(monkeypatch):
    """uvicorn's default is no timeout at all.

    That wait runs before the shutdown hook, so one request that never finishes
    means the worker is never even asked to stop and keeps its devices for the
    life of the process.
    """
    monkeypatch.delenv("PYPTO_GRACEFUL_SHUTDOWN_TIMEOUT", raising=False)
    timeout = graceful_shutdown_timeout_seconds()

    assert timeout is not None and 0 < timeout <= 600


def test_the_graceful_shutdown_wait_can_be_tuned(monkeypatch):
    monkeypatch.setenv("PYPTO_GRACEFUL_SHUTDOWN_TIMEOUT", "45")
    assert graceful_shutdown_timeout_seconds() == 45.0


@pytest.mark.parametrize("value", ["0", "-1", "forever"])
def test_an_unusable_graceful_shutdown_wait_is_refused(monkeypatch, value):
    """Rejected at startup rather than silently restoring the unbounded wait."""
    monkeypatch.setenv("PYPTO_GRACEFUL_SHUTDOWN_TIMEOUT", value)
    with pytest.raises(ValueError, match="positive number of seconds"):
        graceful_shutdown_timeout_seconds()


# --- the loop that runs in front of the stop ---------------------------------


def test_a_failed_engine_loop_still_gets_its_worker_stopped():
    """The loop dying is the case where stopping matters most.

    Awaiting the loop task re-raised its exception here, so the worker was
    never sent a command, never signalled, and kept its devices while the
    parent exited and the endpoint closed behind it.
    """
    async def scenario():
        async def loop_that_dies():
            raise RuntimeError("the scheduler raised")

        process, queue = _FakeProcess(dies_after=0), _FakeQueue()
        core = _core(process, queue)
        core._loop_task = asyncio.create_task(loop_that_dies())
        await asyncio.sleep(0)

        reaped = await core.stop()
        return reaped, len(queue.sent), core._worker_process

    reaped, sent, handle = asyncio.run(scenario())
    assert reaped is True, "a dead loop must not stop the worker from being reaped"
    assert sent == 1, "the shutdown command still went out"
    assert handle is None


def test_an_engine_loop_that_will_not_finish_does_not_hold_up_the_stop(monkeypatch):
    """Clearing _running does not interrupt a read already in flight.

    The loop usually sits in a worker read bounded by the worker step timeout,
    twenty minutes by default, which is not the shutdown's budget: waiting it
    out would put every bounded stage behind an unbounded one. The read ends
    when the worker does, so the worker is stopped first.
    """
    monkeypatch.setattr(async_engine, "LOOP_DRAIN_GRACE_SECONDS", 0.05)

    async def scenario():
        async def loop_that_never_returns():
            await asyncio.sleep(3600)

        process = _FakeProcess(dies_after=0)
        core = _core(process, _FakeQueue())
        core._loop_task = asyncio.create_task(loop_that_never_returns())

        started = asyncio.get_running_loop().time()
        reaped = await core.stop()
        return reaped, asyncio.get_running_loop().time() - started

    reaped, elapsed = asyncio.run(scenario())
    assert reaped is True
    assert elapsed < 1.0, "the stop waited on the loop instead of stopping the worker"


def test_one_core_failing_does_not_leave_its_siblings_unstopped():
    """gather() without return_exceptions abandons the cores it has not reached.

    With data parallelism that is one raised exception against several device
    groups: the siblings never receive a command or a signal, and the tally
    that would have reported them never runs either.
    """
    class _Core:
        def __init__(self, outcome) -> None:
            self.outcome = outcome
            self.stopped = False

        async def stop(self):
            await asyncio.sleep(0)
            self.stopped = True
            if isinstance(self.outcome, BaseException):
                raise self.outcome
            return self.outcome

    async def scenario():
        cores = [_Core(RuntimeError("core 0 is wedged")), _Core(True), _Core(True)]
        engine = AsyncLLMEngine.__new__(AsyncLLMEngine)
        engine._cores = cores
        return await engine.stop(), [core.stopped for core in cores]

    reaped, stopped = asyncio.run(scenario())
    assert stopped == [True, True, True], "a sibling was left holding its devices"
    assert reaped is False, "a core that could not say what became of its worker counts"

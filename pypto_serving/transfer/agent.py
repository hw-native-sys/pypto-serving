# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded admission and observable completion for worker-local transfer."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future
import math
import threading
import time
from typing import Callable, Protocol

from .errors import ErrorCode, TransferError, TransferFailure
from .events import StatusIndex, TransferEvent
from .provider import TransferProvider
from .types import CompletionCertainty as Certainty, OwnerRef, ProviderTransferTask, Stage, integer


class ReadyFence(Protocol):
    def wait(self, timeout: float) -> bool:
        """Return true when source computation completed, false when timed out."""
        ...


class AlreadyCompletedFence:
    def wait(self, timeout: float) -> bool:
        return True


class TransferAgent:
    """One progress thread per owner; process supervision bounds blocked providers."""

    def __init__(self, owner: OwnerRef, provider: TransferProvider, *,
                 poison: Callable[[TransferEvent], None], max_tasks: int = 64,
                 max_bytes: int | None = None, status_capacity: int = 1024):
        integer(max_tasks, "task limit", 1)
        if max_bytes is not None:
            integer(max_bytes, "byte limit", 1)
        self.owner, self.provider, self.poison = owner, provider, poison
        self.status = StatusIndex(owner, status_capacity)
        self.max_tasks, self.max_bytes = max_tasks, max_bytes
        self._cv = threading.Condition(threading.RLock())
        self._queue = deque()
        self._bytes = 0
        self._count = 0
        self._closing = False
        self._poisoned = False
        self._thread = threading.Thread(target=self._run, name="transfer-agent", daemon=False)
        self._thread.start()

    def submit(self, task: ProviderTransferTask, fence: ReadyFence, *, timeout: float = 30) -> Future:
        if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("deadline must be positive and finite")
        task.validate(self.provider.capabilities)
        with self._cv:
            if self._closing or self._poisoned:
                raise TransferFailure(TransferError(ErrorCode.POISONED, Certainty.NOT_SUBMITTED))
            if self._count >= self.max_tasks or (
                self.max_bytes is not None
                and self._bytes + task.nbytes > self.max_bytes
            ):
                raise TransferFailure(TransferError(ErrorCode.BACKPRESSURE, Certainty.NOT_SUBMITTED, True))
            self.status.create(task.attempt)
            self.status.advance(task.attempt, Stage.VALIDATED)
            self.status.advance(task.attempt, Stage.QUEUED)
            future = Future()
            # Caller-side Future.cancel is not a native cancellation interface.
            future.set_running_or_notify_cancel()
            deadline = time.monotonic() + timeout
            queue_timer = threading.Timer(timeout, self._expire_queued, args=(future,))
            self._queue.append((task, fence, deadline, future, queue_timer))
            self._bytes += task.nbytes
            self._count += 1
            queue_timer.start()
            self._cv.notify()
            return future

    def _expire_queued(self, future):
        with self._cv:
            entry = next((item for item in self._queue if item[3] is future), None)
            if entry is None:
                return
            self._queue.remove(entry)
            task = entry[0]
            event = self.status.advance(task.attempt, Stage.FAILED,
                                        TransferError(ErrorCode.DEADLINE, Certainty.NOT_SUBMITTED))
            self._bytes -= task.nbytes
            self._count -= 1
            self._finish(future, event)
            self._cv.notify_all()

    def _run(self):
        while True:
            with self._cv:
                self._cv.wait_for(lambda: self._queue or self._closing)
                if not self._queue:
                    return
                task, fence, deadline, future, queue_timer = self._queue.popleft()
                queue_timer.cancel()
            try:
                event = self._execute(task, fence, deadline, future)
                if event is not None:
                    self._finish(future, event)
            except BaseException as exc:
                # A failed supervisor callback remains visible to the caller.
                with self._cv:
                    self._poisoned = True
                self._finish(future, error=exc)
            finally:
                with self._cv:
                    self._bytes -= task.nbytes
                    self._count -= 1
                    self._cv.notify_all()

    def _finish(self, future, event=None, error=None):
        with self._cv:
            if not future.done():
                if error is None:
                    future.set_result(event)
                else:
                    future.set_exception(error)

    def _execute(self, task, fence, deadline, future):
        attempt = task.attempt
        submitted = False
        expired = False
        completion_lock = threading.RLock()

        def expire():
            nonlocal expired
            with completion_lock:
                current = self.status.query(attempt).event
                if current.stage.terminal:
                    return
                error = TransferError(ErrorCode.DEADLINE,
                                      Certainty.UNKNOWN if submitted else Certainty.NOT_SUBMITTED)
                event = self.status.advance(attempt, Stage.FAILED, error)
                expired = True
                if submitted:
                    with self._cv:
                        self._poisoned = True
            try:
                if submitted:
                    self.poison(event)
                self._finish(future, event)
            except BaseException as exc:
                self._finish(future, error=exc)

        timer = threading.Timer(max(0, deadline - time.monotonic()), expire)
        timer.start()
        try:
            if self._poisoned:
                raise TransferFailure(TransferError(ErrorCode.POISONED, Certainty.NOT_SUBMITTED))
            with completion_lock:
                current = self.status.query(attempt).event
                if current.stage.terminal:
                    return None if expired else current
                self.status.advance(attempt, Stage.WAITING_FENCE)
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not fence.wait(remaining):
                raise TransferFailure(TransferError(ErrorCode.DEADLINE, Certainty.NOT_SUBMITTED))
            if time.monotonic() >= deadline:
                raise TransferFailure(TransferError(ErrorCode.DEADLINE, Certainty.NOT_SUBMITTED))
            with completion_lock:
                current = self.status.query(attempt).event
                if current.stage.terminal:
                    return None if expired else current
                self.status.advance(attempt, Stage.READY)

            def on_submitted():
                nonlocal submitted
                with completion_lock:
                    if submitted:
                        raise ValueError("provider submitted twice")
                    if self.status.query(attempt).event.stage.terminal:
                        raise TransferFailure(TransferError(ErrorCode.DEADLINE, Certainty.NOT_SUBMITTED))
                    self.status.advance(attempt, Stage.SUBMITTED)
                    submitted = True

            self.provider.write(task, on_submitted)
            if not submitted:
                raise RuntimeError("provider returned without submission observation")
            with completion_lock:
                current = self.status.query(attempt).event
                if current.stage.terminal:
                    return None if expired else current
                return self.status.advance(attempt, Stage.COMPLETED)
        except BaseException as exc:
            if isinstance(exc, TransferFailure):
                error = exc.error
            else:
                error = TransferError(ErrorCode.BACKEND_FAILURE if submitted else ErrorCode.FENCE_FAILED,
                                      Certainty.UNKNOWN if submitted else Certainty.NOT_SUBMITTED)
            if submitted and error.certainty == Certainty.NOT_SUBMITTED:
                error = TransferError(error.code, Certainty.UNKNOWN, backend_code=error.backend_code)
            with completion_lock:
                current = self.status.query(attempt).event
                if current.stage.terminal:
                    return None if expired else current
                if error.certainty == Certainty.UNKNOWN and not submitted:
                    self.status.advance(attempt, Stage.SUBMITTED)
                event = self.status.advance(attempt, Stage.FAILED, error)
                if error.certainty == Certainty.UNKNOWN:
                    with self._cv:
                        self._poisoned = True
            if error.certainty == Certainty.UNKNOWN:
                self.poison(event)
            return event
        finally:
            timer.cancel()

    def close(self, timeout: float = 5) -> None:
        with self._cv:
            self._closing = True
            self._cv.notify_all()
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise TimeoutError("transfer progress still alive; owner supervision must terminate native work")

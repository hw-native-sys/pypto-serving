# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Local crash-only recovery facts; peer coordination belongs to the caller."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import os
import select
import signal
import socket
from dataclasses import asdict
from typing import Callable, Protocol

from .events import TransferEvent
from .types import CompletionCertainty, OwnerRef


class OwnedProcess(Protocol):
    def terminate(self) -> None: ...
    def kill(self) -> None: ...
    def wait(self, timeout: float) -> None: ...


class OwnerProcessHandle:
    """Linux pidfd pins identity and observes exit without reaping Simpler's child."""

    def __init__(self, pid: int):
        if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
            raise RuntimeError("owner supervision requires Linux pidfd support")
        os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        self.pid = pid
        self.fd = os.pidfd_open(pid)

    def terminate(self):
        self._signal(signal.SIGTERM)

    def kill(self):
        self._signal(signal.SIGKILL)

    def _signal(self, sig):
        try:
            signal.pidfd_send_signal(self.fd, sig)
        except ProcessLookupError:
            pass

    def wait(self, timeout: float):
        if not select.select([self.fd], [], [], timeout)[0]:
            raise TimeoutError("owner has not exited")

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


class OwnerHealthMonitor:
    """Observe idle owner exit too; a private wakeup socket makes normal close bounded."""

    def __init__(self, process: OwnerProcessHandle, on_exit: Callable[[], None]):
        self.process, self.on_exit = process, on_exit
        self._reader, self._writer = socket.socketpair()
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="owner-health", daemon=False)
        self._thread.start()

    def _run(self):
        readable = select.select([self.process.fd, self._reader], [], [])[0]
        if self._reader not in readable:
            self.on_exit()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._writer.send(b"x")
        if self._thread is not threading.current_thread():
            self._thread.join(5)
            if self._thread.is_alive():
                raise TimeoutError("owner health callback still running")
        self._reader.close()
        self._writer.close()


@dataclass(frozen=True)
class RecoveryEvent:
    stage: str
    owner: OwnerRef
    parent_event_id: str
    recovery_set: tuple[OwnerRef, ...]


class OwnerSupervisor:
    """Only supplied local process handles may be terminated; never selects a new route."""

    def __init__(self, owner: OwnerRef, processes: tuple[OwnedProcess, ...], *,
                 emit: Callable[[RecoveryEvent], None], graceful_timeout: float = 1,
                 kill_timeout: float = 5, local_owners: tuple[OwnerRef, ...] | None = None,
                 peer_owners: tuple[OwnerRef, ...] | None = None):
        if not processes or graceful_timeout <= 0 or kill_timeout <= 0:
            raise ValueError("owned recovery processes and positive deadlines required")
        self.owner, self.processes, self.emit = owner, processes, emit
        self.local_owners = local_owners or (owner,)
        self.peer_owners = peer_owners
        if owner not in self.local_owners or len(set(self.local_owners)) != len(self.local_owners):
            raise ValueError("local recovery set must contain the initiating owner exactly once")
        if any(ref.run_id != owner.run_id or ref.worker_id != owner.worker_id for ref in self.local_owners):
            raise ValueError("local recovery set must belong to one worker in this run")
        if peer_owners is not None and (not peer_owners or len(set(peer_owners)) != len(peer_owners)
                                       or any(ref.run_id != owner.run_id for ref in peer_owners)
                                       or set(peer_owners) & set(self.local_owners)):
            raise ValueError("peer recovery set must be distinct, nonempty and in the same run")
        self.graceful_timeout, self.kill_timeout = graceful_timeout, kill_timeout
        self._lock = threading.Lock()
        self._poisoned = False
        self.dead = False
        self.peer_dead = False
        self._peer = None
        self._confirmed_peers = set()
        self.events = []
        self.snapshot = None

    def poison(self, event: TransferEvent):
        if self.owner not in (event.attempt.source, event.attempt.destination) or (
            event.certainty != CompletionCertainty.UNKNOWN
        ):
            raise ValueError("poison requires a local uncertain attempt")
        with self._lock:
            if self._poisoned:
                if not self.dead:
                    raise RuntimeError("owner recovery is in progress or incomplete")
                return
            self._poisoned = True
            self._peer = (event.attempt.destination if event.attempt.source == self.owner
                          else event.attempt.source)
        if self.peer_owners is None:
            self.peer_owners = (self._peer,)
        if self._peer not in self.peer_owners:
            raise ValueError("attempt peer is absent from recovery set")
        recovery_set = self.local_owners + self.peer_owners
        self.snapshot = {
            "cause": event.to_dict(),
            "processes": [getattr(process, "pid", None) for process in self.processes],
        }
        callback_errors = []

        def publish(stage):
            fact = RecoveryEvent(stage, self.owner, event.event_id, recovery_set)
            self.events.append(asdict(fact))
            try:
                self.emit(fact)
            except Exception as exc:
                callback_errors.append(type(exc).__name__)

        publish("OwnerPoisoned")
        # The peer requirement is published even if local termination subsequently fails.
        publish("PeerPoisonRequired")
        termination_errors = []
        for process in self.processes:
            try:
                process.terminate()
            except Exception as exc:
                termination_errors.append(type(exc).__name__)
        all_dead = True
        for process in self.processes:
            try:
                process.wait(self.graceful_timeout)
            except TimeoutError:
                try:
                    process.kill()
                    process.wait(self.kill_timeout)
                except Exception as exc:
                    all_dead = False
                    termination_errors.append(type(exc).__name__)
            except Exception as exc:
                all_dead = False
                termination_errors.append(type(exc).__name__)
        self.snapshot["termination_errors"] = termination_errors
        if not all_dead:
            publish("OwnerTerminationFailed")
            raise RuntimeError("some recovery processes could not be confirmed dead")
        self.dead = True
        publish("OwnerDead")
        if callback_errors:
            raise RuntimeError("recovery event delivery failed after local termination")

    def confirm_peer_dead(self, peer: OwnerRef):
        if self._peer is None or peer not in self.peer_owners:
            raise ValueError("stale peer death")
        self._confirmed_peers.add(peer)
        self.peer_dead = self._confirmed_peers == set(self.peer_owners)

    def permit_rebuild(self, new_owner: OwnerRef):
        if not self.dead or not self.peer_dead:
            raise RuntimeError("both local and peer owner death must be confirmed")
        if (new_owner.run_id != self.owner.run_id or new_owner.worker_id != self.owner.worker_id
                or new_owner.rank_id != self.owner.rank_id
                or new_owner.generation <= self.owner.generation
                or new_owner.endpoint_generation <= self.owner.endpoint_generation):
            raise ValueError("rebuild requires a fresh owner and endpoint generation")

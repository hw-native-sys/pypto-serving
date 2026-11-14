# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent transfer submission and completion, shared by all peers."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import logging
import select
import socket
import struct
import threading
import time

import msgspec

from pypto_serving.tools.observe import record, span
from .protocol import Completion, MAX_FRAME, Plan, Topology, decode, digest
from .transfer import Owner, Region, TransferError, owner_factories

logger = logging.getLogger(__name__)


def _remaining(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("independent transfer response deadline")
    return remaining


def _send_frame(channel, payload, deadline):
    if not 0 < len(payload) <= MAX_FRAME:
        raise ValueError("invalid independent transfer frame size")
    channel.settimeout(_remaining(deadline))
    channel.sendall(struct.pack("!I", len(payload)) + payload)


def _receive_frame(channel, deadline):
    def exact(size):
        parts = bytearray()
        while len(parts) < size:
            channel.settimeout(_remaining(deadline))
            part = channel.recv(size - len(parts))
            if not part:
                raise EOFError("independent transfer channel closed")
            parts.extend(part)
        return bytes(parts)
    size, = struct.unpack("!I", exact(4))
    if not 0 < size <= MAX_FRAME:
        raise ValueError("invalid independent transfer frame size")
    return exact(size)


def _shutdown(channel):
    try:
        channel.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass


class WorkerRuntime:
    def __init__(self, settings, model, progress_channel=None):
        self.settings, self.model = settings, model
        self.owners = tuple(Owner(settings.boot, i, settings.hostname, settings.provider)
                            for i in range(model.ranks))
        self.peers, self.peer_fingerprints, self.jobs = {}, {}, {}
        self.registry = None
        self.registry_digest = None
        self.local_regions, self.peer_regions = {}, {}
        self.uncertain = False
        self.accepting = True
        self.pool = ThreadPoolExecutor(max_workers=model.ranks, thread_name_prefix="pd-write")
        self.lock = threading.Lock()
        self.progress_channel, self.progress = progress_channel, None

    def start_progress(self):
        # Start only after L3 owners have forked; do not fork a progress thread.
        if self.progress_channel is not None:
            self.progress = ProgressServer(self.progress_channel, self.handle_progress, self.settings.handoff_timeout)

    def worker_options(self):
        return {"chip_service_factories": owner_factories(self.owners)}

    def worker_ready(self):
        for owner in self.owners:
            owner.ready()

    def prepare(self):
        if self.registry is None:
            geometry = self.model.geometry()
            regions = tuple(owner.register(self.model.worker(), tensor.shards[rank], name)
                            for rank, owner in enumerate(self.owners)
                            for name, tensor in self.model.regions().items())
            self.registry = dict(geometry=geometry, regions=regions,
                                 topology=Topology.for_tp(self.model.ranks, self.model.tp))
            self.registry_digest = digest(self.registry)
            # Region geometry is fixed for this boot; request page IDs are not cached here.
            self.local_regions = {(r.rank, r.name): r for r in regions}
        return self.registry

    def install(self, value):
        with self.lock:
            self._check_accepting()
            self._install(value)

    def _install(self, value):
        regions = tuple(decode(r, Region) for r in value["regions"])
        topology = decode(value["topology"], Topology)
        boots = {r.boot for r in regions}
        if len(boots) != 1:
            raise ValueError("registry must belong to one node boot")
        peer = next(iter(boots))
        fingerprint = digest(value)
        if peer in self.peers:
            if self.peer_fingerprints[peer] != fingerprint:
                raise ValueError("peer registry changed within boot")
            return
        if len(self.peers) >= self.settings.peer_capacity:
            raise ValueError("peer registry capacity exhausted")
        names = set(self.model.regions())
        if ({(r.rank, r.name) for r in regions} != {(i, n) for i in range(topology.ranks) for n in names}
                or len(regions) != len(names) * topology.ranks):
            raise ValueError("peer registry has missing or duplicate rank/region")
        for owner in self.owners:
            owner.call("install", regions=regions)
        self.peers[peer] = regions
        self.peer_fingerprints[peer] = fingerprint
        self.peer_regions[peer] = {(r.rank, r.name): r for r in regions}

    def _lower(self, plan):
        if (plan.binding.prefill[1] != self.settings.boot
                or plan.binding.prefill[2] != self.registry_digest):
            raise ValueError("source boot or registry mismatch")
        peer = plan.binding.decode[1]
        if peer not in self.peers:
            raise ValueError("destination registry not installed")
        if self.peer_fingerprints[peer] != plan.binding.decode[2]:
            raise ValueError("destination registry fingerprint mismatch")
        local, remote = self.local_regions, self.peer_regions[peer]
        batches = {}
        occupied = {}
        for copy in plan.copies:
            if (copy.source_rank, copy.destination_rank) not in plan.mapping:
                raise ValueError("copy does not belong to request rank mapping")
            src = local[copy.source_rank, copy.component]
            dst = remote[copy.destination_rank, copy.component]
            if (min(copy.source_offset, copy.destination_offset) < 0 or copy.length <= 0
                    or copy.source_offset + copy.length > src.extent
                    or copy.destination_offset + copy.length > dst.extent):
                raise ValueError("copy outside source/destination registration")
            key = copy.destination_rank, copy.component
            occupied.setdefault(key, []).append((copy.destination_offset, copy.destination_offset + copy.length))
            batches.setdefault((copy.source_rank, copy.destination_rank), []).append(copy)
        for intervals in occupied.values():
            ordered = sorted(intervals)
            if any(a[1] > b[0] for a, b in zip(ordered, ordered[1:])):
                raise ValueError("overlapping destination writes")
        job = digest(plan)
        observation = dict(request=plan.binding.request, plan_digest=job,
                           prefill=plan.binding.prefill, decode=plan.binding.decode)
        return peer, batches, job, observation

    def submit(self, plan):
        with self.lock:
            self._check_accepting()
            with span(logger, "pd.transfer.lower", log=False, thread_cpu=True,
                      request=plan.binding.request, chunk=plan.chunk) as timing:
                peer, batches, job, observation = self._lower(plan)
            if len(self.jobs) >= self.settings.capacity or job in self.jobs:
                raise ValueError("transfer capacity exceeded or duplicate submission")
            # Every rank was validated before the first native task is enqueued.
            try:
                futures = tuple(self.pool.submit(self.owners[rank].call, "write", peer=peer,
                    copies=tuple(copies), observation={**observation, "source_rank": rank,
                                                     "destination_rank": destination})
                    for (rank, destination), copies in batches.items())
            except BaseException as exc:
                self.uncertain = True
                raise TransferError("partial rank submission", uncertain=True) from exc
            self.jobs[job] = (plan, futures)
        record(logger, "transfer.submit", **observation, bytes=sum(c.length for c in plan.copies),
               rank_batches=len(batches), lower_ms=timing["duration_ms"],
               lower_thread_cpu_ms=timing["thread_cpu_ms"])
        return job

    def _check_accepting(self):
        if self.uncertain or not self.accepting:
            raise TransferError("transfer worker is gated or closing", uncertain=self.uncertain)

    def poll(self, jobs):
        """Take terminal results once; source pins outlive this bookkeeping.

        A lost reply remains UNKNOWN at the caller. Neither taking a result
        nor losing it releases cache snapshots or registered owner allocations.
        """
        with self.lock:
            result = []
            for job in jobs:
                plan, futures = self.jobs[job]
                if not all(f.done() for f in futures):
                    result.append(None)
                    continue
                errors = [f.exception() for f in futures if f.exception() is not None]
                if errors:
                    uncertain = any(not isinstance(e, TransferError) or e.uncertain for e in errors)
                    self.uncertain |= uncertain
                    result.append(dict(error="rank transfer failed", uncertain=uncertain))
                else:
                    result.append(dict(completion=Completion(job, plan.units)))
                del self.jobs[job]
            return result

    def handle(self, operation, payload):
        with span(logger, "pd.worker.decode_message", log=False, thread_cpu=True, operation=operation):
            value = msgspec.msgpack.decode(payload) if payload else None
        if operation == "registry":
            result = self.prepare()
        elif operation == "install":
            result = self.install(value)
        elif operation == "release":
            with self.lock:
                self.accepting = False
                if self.uncertain or self.jobs:
                    raise TransferError("cannot unregister an unsafe or active owner", uncertain=True)
                for owner in self.owners:
                    owner.release()
            result = None
        else:
            result = self.model.handle(operation, value)
        return msgspec.msgpack.encode(result)

    def handle_progress(self, operation, payload):
        """Only these operations may run beside the compute command lane."""
        try:
            with span(logger, "pd.worker.decode_message", log=False, thread_cpu=True, operation=operation):
                value = msgspec.msgpack.decode(payload)
            if operation == "submit":
                with span(logger, "pd.worker.decode_plan", log=False, thread_cpu=True):
                    plan = decode(value, Plan)
                result = dict(job=self.submit(plan))
            elif operation == "poll":
                result = self.poll(value)
            elif operation == "source_state":
                with self.lock:
                    self._check_accepting()
                    result = self.model.source_state(value)
            else:
                raise ValueError("unknown independent transfer operation")
        except Exception as exc:
            uncertain = (exc.uncertain if isinstance(exc, TransferError) else operation == "poll"
                         or (operation == "submit" and not isinstance(exc, (ValueError, KeyError, msgspec.DecodeError))))
            if uncertain:
                with self.lock:
                    self.uncertain = True
            result = dict(error=str(exc), uncertain=uncertain)
        return msgspec.msgpack.encode(result)

    def close(self):
        with self.lock:
            self.accepting = False
            if self.uncertain or self.jobs or any(o.retained for o in self.owners):
                raise RuntimeError("transfer must be safely drained before model shutdown")
        self.pool.shutdown(wait=True)
        if self.progress is not None:
            self.progress.close()
        for owner in self.owners:
            owner.close()


@dataclass(frozen=True)
class WorkerSettings:
    boot: str
    hostname: str
    provider: str
    capacity: int
    peer_capacity: int
    handoff_timeout: float = 300.0


class Services:
    def __init__(self, extensions, request_lookup):
        if len(extensions) != 1:
            raise ValueError("PD requires one model extension")
        self.extension, = extensions
        self.lookup = request_lookup
        self.extension.transfer.start_progress()

    def wrap_batch(self, build):
        def batch(cmd, *args, **kwargs):
            from dataclasses import replace
            value = build(cmd, *args, **kwargs)
            if value is None:
                return value
            initial = tuple(r for r in value.request_ids if self.lookup(r).initialize_from_cache)
            return replace(value, initial_request_ids=initial)
        return batch

    def handle_command(self, operation, payload):
        return self.extension.transfer.handle(operation, payload)

    def set_profile_active(self, active):
        # Owner transfer spans use the same on-demand profile lifecycle.
        self.extension.set_profile_active(active)


class ProgressServer:
    def __init__(self, channel, handle, timeout=300.0):
        self.channel, self.handle, self.timeout = channel, handle, timeout
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="pd-completion", daemon=False)
        self.thread.start()

    def _run(self):
        try:
            while not self.stop.is_set():
                if select.select((self.channel,), (), (), .02)[0]:
                    deadline = time.monotonic() + self.timeout
                    sequence, operation, payload = msgspec.msgpack.decode(
                        _receive_frame(self.channel, deadline), type=tuple[int, str, bytes])
                    with span(logger, "pd.progress.handle", log=False,
                              command_id=sequence, operation=operation):
                        result = self.handle(operation, payload)
                    _send_frame(self.channel, msgspec.msgpack.encode((sequence, result)), deadline)
        except (EOFError, OSError, ValueError, msgspec.DecodeError):
            pass
        finally:
            self.channel.close()

    def close(self):
        self.stop.set()
        _shutdown(self.channel)
        self.thread.join(2)
        if self.thread.is_alive():
            raise RuntimeError("completion observer did not stop")
        self.channel.close()


class Progress:
    def __init__(self, channel, timeout, capacity):
        self.channel, self.timeout, self.capacity = channel, timeout, capacity
        self.pending = {}
        self.task = None
        self.lock = asyncio.Lock()
        self.sequence = self.calls = 0
        self.accepting = True
        self.closed = False
        self.error = None

    def _exchange(self, sequence, packet, deadline):
        _send_frame(self.channel, packet, deadline)
        reply, result = msgspec.msgpack.decode(_receive_frame(self.channel, deadline), type=tuple[int, bytes])
        if reply != sequence:
            raise ValueError("independent transfer reply correlation mismatch")
        return result

    def _fail_channel(self, error):
        self.error = TransferError(str(error), uncertain=True)
        _shutdown(self.channel)
        self.channel.close()

    async def call(self, operation, payload):
        if self.closed or (not self.accepting and operation != "poll"):
            raise TransferError("transfer observer is closing", uncertain=False)
        if operation != "poll" and self.calls >= self.capacity:
            raise TransferError("transfer command capacity exceeded", uncertain=False)
        counted = operation != "poll"
        self.calls += counted
        self.sequence += 1
        sequence = self.sequence
        deadline = time.monotonic() + self.timeout
        try:
            with span(logger, "pd.progress.wait_channel", log=False, operation=operation, command_id=sequence):
                await asyncio.wait_for(self.lock.acquire(), _remaining(deadline))
            try:
                if self.error is not None:
                    raise self.error
                if self.closed or (not self.accepting and operation != "poll"):
                    raise TransferError("transfer observer is closing", uncertain=False)
                packet = msgspec.msgpack.encode((sequence, operation, payload))
                if len(packet) > MAX_FRAME:
                    raise TransferError("transfer command exceeds frame limit", uncertain=False)
                _remaining(deadline)  # Expiry before starting I/O is a known rejection.
                with span(logger, "pd.progress.roundtrip", log=False, operation=operation, command_id=sequence):
                    response = asyncio.create_task(asyncio.to_thread(self._exchange, sequence, packet, deadline))
                    try:
                        result = await asyncio.shield(response)
                    except asyncio.CancelledError:
                        # Keep the sole reply reader until it finishes, including
                        # a late submit ack. Cancellation is not native completion.
                        while not response.done():
                            try:
                                await asyncio.shield(response)
                            except asyncio.CancelledError:
                                continue
                            except Exception:
                                break
                        if not response.cancelled():
                            error = response.exception()
                            if error is not None:
                                self._fail_channel(error)
                            elif operation == "submit" and not self.closed:
                                reply = msgspec.msgpack.decode(response.result())
                                if "job" in reply:
                                    self._track(reply["job"])
                        raise
                    except Exception as exc:
                        self._fail_channel(exc)
                        raise self.error from exc
                return result
            finally:
                self.lock.release()
        except (asyncio.TimeoutError, TimeoutError) as exc:
            raise TransferError("transfer command waited for channel", uncertain=False) from exc
        finally:
            self.calls -= counted

    def _track(self, job):
        if self.closed or job in self.pending or len(self.pending) >= self.capacity:
            raise TransferError("completion observer closed, full or duplicate waiter", uncertain=True)
        future = asyncio.get_running_loop().create_future()
        self.pending[job] = future
        def consume(done):
            self.pending.pop(job, None)
            if not done.cancelled():
                done.exception()
        future.add_done_callback(consume)
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._observe())
        return future

    async def wait(self, job):
        future = self._track(job)
        try:
            return await asyncio.wait_for(asyncio.shield(future), self.timeout)
        except asyncio.TimeoutError as exc:
            raise TransferError("native completion deadline", uncertain=True) from exc

    async def _observe(self):
        try:
            while self.pending:
                jobs = tuple(self.pending)
                values = msgspec.msgpack.decode(await self.call("poll", msgspec.msgpack.encode(jobs)))
                if isinstance(values, dict) or len(values) != len(jobs):
                    raise TransferError("completion reply correlation mismatch", uncertain=True)
                for job, value in zip(jobs, values):
                    if value is not None:
                        future = self.pending[job]
                        if "error" in value:
                            future.set_exception(TransferError(value["error"], uncertain=value["uncertain"]))
                        else:
                            future.set_result(decode(value["completion"], Completion))
                await asyncio.sleep(.002)
        except Exception as exc:
            for future in tuple(self.pending.values()):
                if not future.done():
                    future.set_exception(TransferError(str(exc), uncertain=True))

    async def close(self):
        self.accepting = False
        async def drain():
            async with self.lock:
                pass  # Finish accepted command replies, including late submit acks.
            if self.task is not None:
                await asyncio.shield(self.task)
        try:
            await asyncio.wait_for(drain(), self.timeout)
        except asyncio.TimeoutError as exc:
            self._fail_channel("native completion drain deadline")
            for future in tuple(self.pending.values()):
                if not future.done():
                    future.set_exception(self.error)
            raise self.error from exc
        finally:
            self.closed = True
            _shutdown(self.channel)
            self.channel.close()

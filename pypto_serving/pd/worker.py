# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Ordered submission with independent completion observation, shared by all peers."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import threading

import msgspec

from .protocol import Completion, MAX_FRAME, Plan, decode, digest
from .transfer import Owner, Region, TransferError, owner_factories


class WorkerRuntime:
    def __init__(self, settings, model, progress_channel=None):
        self.settings, self.model = settings, model
        self.owners = tuple(Owner(settings.boot, i, settings.hostname, settings.provider)
                            for i in range(model.ranks))
        self.peers, self.peer_fingerprints, self.jobs = {}, {}, {}
        self.registry = None
        self.uncertain = False
        self.pool = ThreadPoolExecutor(max_workers=model.ranks, thread_name_prefix="pd-write")
        self.lock = threading.Lock()
        self.progress_channel, self.progress = progress_channel, None

    def start_progress(self):
        # Start only after L3 owners have forked; do not fork a progress thread.
        if self.progress_channel is not None:
            self.progress = ProgressServer(self.progress_channel, self.poll)

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
            self.registry = dict(geometry=geometry, regions=regions, topology=(self.model.ranks, self.model.tp))
        return self.registry

    def install(self, value):
        regions = tuple(decode(r, Region) for r in value["regions"])
        boots = {r.boot for r in regions}
        if len(boots) != 1:
            raise ValueError("registry must belong to one node boot")
        peer = next(iter(boots))
        if peer in self.peers:
            if self.peers[peer] != regions:
                raise ValueError("peer registry changed within boot")
            return
        if len(self.peers) >= self.settings.peer_capacity:
            raise ValueError("peer registry capacity exhausted")
        names = set(self.model.regions())
        if ({(r.rank, r.name) for r in regions} != {(i, n) for i in range(self.model.ranks) for n in names}
                or len(regions) != len(names) * self.model.ranks):
            raise ValueError("peer registry has missing or duplicate rank/region")
        for owner in self.owners:
            owner.call("install", regions=regions)
        self.peers[peer] = regions
        self.peer_fingerprints[peer] = digest(value)

    def submit(self, plan):
        if (self.uncertain or plan.binding.prefill[1] != self.settings.boot
                or plan.binding.prefill[2] != digest(self.registry)):
            raise TransferError("worker gated or source boot mismatch", uncertain=self.uncertain)
        peer = plan.binding.decode[1]
        if peer not in self.peers:
            raise ValueError("destination registry not installed")
        if self.peer_fingerprints[peer] != plan.binding.decode[2]:
            raise ValueError("destination registry fingerprint mismatch")
        local = {(r.rank, r.name): r for r in self.registry["regions"]}
        remote = {(r.rank, r.name): r for r in self.peers[peer]}
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
            batches.setdefault(copy.source_rank, []).append(copy)
        for intervals in occupied.values():
            ordered = sorted(intervals)
            if any(a[1] > b[0] for a, b in zip(ordered, ordered[1:])):
                raise ValueError("overlapping destination writes")
        job = digest(plan)
        with self.lock:
            if len(self.jobs) >= self.settings.capacity or job in self.jobs:
                raise ValueError("transfer capacity exceeded or duplicate submission")
            # Every rank was validated before the first native task is enqueued.
            try:
                futures = tuple(self.pool.submit(self.owners[rank].call, "write", peer=peer, copies=tuple(copies))
                                for rank, copies in batches.items())
            except BaseException as exc:
                self.uncertain = True
                raise TransferError("partial rank submission", uncertain=True) from exc
            self.jobs[job] = (plan, futures)
        return job

    def poll(self, jobs):
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
            return result

    def handle(self, operation, payload):
        value = msgspec.msgpack.decode(payload) if payload else None
        if operation == "registry":
            result = self.prepare()
        elif operation == "install":
            result = self.install(value)
        elif operation == "submit":
            try:
                result = dict(job=self.submit(decode(value, Plan)))
            except (TransferError, ValueError) as exc:
                result = dict(error=str(exc), uncertain=isinstance(exc, TransferError) and exc.uncertain)
        elif operation == "forget":
            with self.lock:
                _, futures = self.jobs[value]
                if not all(f.done() for f in futures):
                    raise RuntimeError("cannot forget an in-flight native job")
                del self.jobs[value]
            result = None
        elif operation == "release":
            if self.uncertain or self.jobs:
                raise TransferError("cannot unregister an unsafe or active owner", uncertain=True)
            for owner in self.owners:
                owner.release()
            result = None
        else:
            result = self.model.handle(operation, value)
        return msgspec.msgpack.encode(result)

    def close(self):
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
    def __init__(self, channel, poll):
        self.channel, self.poll = channel, poll
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="pd-completion", daemon=False)
        self.thread.start()

    def _run(self):
        try:
            while not self.stop.is_set():
                if self.channel.poll(.02):
                    jobs = msgspec.msgpack.decode(self.channel.recv_bytes(MAX_FRAME), type=tuple[str, ...])
                    self.channel.send_bytes(msgspec.msgpack.encode(self.poll(jobs)))
        except (EOFError, OSError):
            pass

    def close(self):
        self.stop.set()
        self.thread.join(2)
        if self.thread.is_alive():
            raise RuntimeError("completion observer did not stop")
        self.channel.close()


class Progress:
    def __init__(self, channel, timeout):
        self.channel, self.timeout = channel, timeout
        self.pending = {}
        self.task = None

    def _poll(self, jobs):
        self.channel.send_bytes(msgspec.msgpack.encode(jobs))
        if not self.channel.poll(self.timeout):
            raise TransferError("completion observation timed out", uncertain=True)
        values = msgspec.msgpack.decode(self.channel.recv_bytes(MAX_FRAME))
        if len(values) != len(jobs):
            raise TransferError("completion reply correlation mismatch", uncertain=True)
        return values

    async def wait(self, job):
        if job in self.pending:
            raise ValueError("duplicate completion waiter")
        future = asyncio.get_running_loop().create_future()
        self.pending[job] = future
        def consume(done):
            self.pending.pop(job, None)
            if not done.cancelled():
                done.exception()
        future.add_done_callback(consume)
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._observe())
        try:
            return await asyncio.wait_for(asyncio.shield(future), self.timeout)
        except asyncio.TimeoutError as exc:
            raise TransferError("native completion deadline", uncertain=True) from exc

    async def _observe(self):
        try:
            while self.pending:
                jobs = tuple(self.pending)
                values = await asyncio.to_thread(self._poll, jobs)
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
        if self.task is not None:
            await self.task
        self.channel.close()

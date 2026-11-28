# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise Serving's PD handoff over HIXL RoCE with one small device allocation per rank."""

import argparse
import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
import ctypes as ct
from dataclasses import replace
import math
import multiprocessing
import socket
import threading
from types import SimpleNamespace
import uuid

from pypto_serving.config.types import KVCacheGroupSpec, KVCacheSpec
from pypto_serving.pd.config import Config, Endpoint
from pypto_serving.pd.planner import Geometry, plan_chunk, rank_mapping
from pypto_serving.pd.protocol import (Binding, Component, Completion, Continuation, Contract, Node,
                                      Placement, Prefix, decode, digest, receive, send)
from pypto_serving.pd.service import Service
from pypto_serving.pd.transfer import TransferError, host_copy, provider_factory
from pypto_serving.pd.worker import Progress, WorkerRuntime, WorkerSettings
from pypto_serving.serving.memory.kv_cache import KvCacheManager


PAGE_BYTES = 4096
SOURCE_PAGES = (3, 5)
TOKENS = (100, 101)
GROUP = KVCacheGroupSpec("kv", (0,), KVCacheSpec(1, PAGE_BYTES), 4, num_blocks=16)
GEOMETRY = (Geometry("kv", (0,), GROUP.num_blocks, 1, PAGE_BYTES, "uint8"),)
EXTENT = GEOMETRY[0].extent
GUARDS = bytes([0xD4]) * EXTENT


def source(rank, seed):
    return bytes((17 * index + index // 251 + 33 + rank + 73 * seed) % 256 for index in range(EXTENT))


class Hbm:
    def __init__(self, acl, device, contents):
        self.acl = acl
        self.address = ct.c_void_p()
        self.nbytes = len(contents)
        self.acl.aclrtSetDevice.argtypes = [ct.c_int32]
        self.acl.aclrtMalloc.argtypes = [ct.POINTER(ct.c_void_p), ct.c_size_t, ct.c_int]
        self.acl.aclrtMemcpy.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_void_p, ct.c_size_t, ct.c_int]
        self.acl.aclrtFree.argtypes = [ct.c_void_p]
        # The driver deploys the HIXL AICPU kernel when this process sets its device.
        self.check("aclrtSetDevice", device)
        self.check("aclrtMalloc", ct.byref(self.address), self.nbytes, 0)  # ACL_MEM_MALLOC_HUGE_FIRST
        memory = ct.create_string_buffer(contents)
        self.check("aclrtMemcpy", self.address, self.nbytes, memory, self.nbytes, 1)  # HOST_TO_DEVICE

    def check(self, name, *args):
        result = getattr(self.acl, name)(*args)
        if result:
            raise RuntimeError(f"{name} returned {result}")

    def read(self):
        memory = ct.create_string_buffer(self.nbytes)
        self.check("aclrtMemcpy", memory, self.nbytes, self.address, self.nbytes, 2)  # DEVICE_TO_HOST
        return memory.raw

    def close(self):
        self.check("aclrtFree", self.address)


class ProbeOwner:
    """One device per process, matching the isolation of Serving's chip owners."""

    def __init__(self, settings, device, rank, seed):
        ctx = multiprocessing.get_context("fork")
        self.channel, child = ctx.Pipe()
        self.process = ctx.Process(target=run_owner, args=(child, self.channel, settings, device, rank, seed),
                                   daemon=True)
        self.rank = rank
        self.lock = threading.Lock()
        self.timeout = settings.handoff_timeout
        self.hbm = SimpleNamespace(nbytes=EXTENT)
        self.process.start()
        child.close()
        self.retained = []
        self.released = False

    def register(self, runtime, tensor, name):
        region = self.call("register", name=name)
        if region.extent != tensor.nbytes:
            raise ValueError("probe region extent mismatch")
        self.retained.append(region)
        return region

    def call(self, operation, **payload):
        with self.lock:
            try:
                self.channel.send((operation, payload))
                if not self.channel.poll(self.timeout):
                    raise TimeoutError("owner did not reply")
                ok, result = self.channel.recv()
                if ok:
                    return result
                raise TransferError(result, uncertain=True)
            except (EOFError, OSError) as exc:
                raise TransferError(f"probe owner rank {self.rank}: {exc}", uncertain=True) from exc

    def read(self):
        return self.call("read")

    def release(self):
        self.call("release")
        self.retained.clear()
        self.released = True

    def close(self):
        if not self.released:
            raise RuntimeError("HBM cannot be freed before the HIXL owner retires")
        self.call("close")
        self.channel.close()
        self.process.join(5)
        if self.process.exitcode != 0:
            raise RuntimeError(f"probe owner rank {self.rank} did not retire cleanly")


def run_owner(channel, parent, settings, device, rank, seed):
    parent.close()
    acl = ct.CDLL("libascendcl.so")
    acl.aclInit.argtypes = [ct.c_char_p]
    try:
        result = acl.aclInit(None)
        if result:
            raise RuntimeError(f"aclInit returned {result}")
        buffer = Hbm(acl, device, source(rank, seed) if settings.role == "prefill" else GUARDS)
        provider = provider_factory(settings.provider, settings.hostname, role=settings.role, rank=rank)
        methods = {"prepare_link": provider.prepare, "install": provider.install, "release": provider.close,
                   "connect_ranks": provider.connect, "close_peer": provider.close_peer,
                   "read": buffer.read}
        while True:
            operation, payload = channel.recv()
            try:
                result = None
                if operation == "register":
                    result = provider.register(settings.boot, rank, payload["name"], buffer.address.value,
                                               buffer.nbytes)
                elif operation == "write":
                    provider.put_sync_host(payload["peer"], tuple(host_copy(copy) for copy in payload["copies"]))
                elif operation == "close":
                    buffer.close()
                    buffer.check("aclFinalize")
                else:
                    result = methods[operation](**payload)
                channel.send((True, result))
                if operation == "close":
                    return
            except Exception as exc:
                channel.send((False, f"rank {rank} {operation}: {exc}"))
    except EOFError:
        pass
    except Exception as exc:
        channel.send((False, f"rank {rank} initialization: {exc}"))
    finally:
        channel.close()


class ProbeModel:
    geometry = staticmethod(lambda: GEOMETRY)
    state_ranges = staticmethod(lambda _groups, _end: {})
    prefix_spec = staticmethod(lambda tokens, _cache: Prefix(len(tokens), 1, {}))
    validate = staticmethod(lambda _config: None)

    def __init__(self, owners, tp):
        self.owners, self.ranks, self.tp = owners, len(owners), tp

    def worker(self):
        return None  # ProbeOwner owns the HBM directly.

    def regions(self):
        return {"kv": SimpleNamespace(shards=tuple(owner.hbm for owner in self.owners))}

    def contract(self, model, geometry):
        return Contract("hixl-spike", model, digest(geometry), (Component("kv", "kv", "kv"),))

    def validate_mapping(self, source, target, reservation, mapping):
        if mapping != rank_mapping(source, target, 0, reservation.partition):
            raise ValueError("unexpected spike rank mapping")

    def handle(self, operation, value):
        if operation == "reserve_state":
            return {}
        if operation == "discard_state":
            return None
        raise ValueError(f"unsupported probe worker operation {operation}")


class CheckedCache(KvCacheManager):
    def __init__(self, owners, tp):
        super().__init__(enable_prefix_cache=False)
        self.owners, self.tp = owners, tp
        self.commits = 0
        self.seeds = {}
        self.source_ranks = {}
        self.expected = [bytearray(GUARDS) for _ in owners]
        self.written_pages = set()
        self.init_groups((replace(GROUP, num_partitions=len(owners) // tp),), max_batch_size=2)
        for partition in range(len(owners) // tp):
            self.ensure_group_blocks(f"guard-page-{partition}", 1, partition=partition)

    def check_before_dma(self, request):
        lease = self.group_cache_reservation(request)
        if (self.commits or lease.state.value == "READY"
                or any(owner.read() != GUARDS for owner in self.owners)):
            raise RuntimeError("invalid completion modified or committed Decode HBM")
        print(f"BAD_COMPLETION_REJECTED commits={self.commits} all_hbm_bytes_untouched={EXTENT * len(self.owners)}",
              flush=True)

    def commit_group_cache(self, reservation_id, *, valid_from=None):
        lease = self.group_cache_reservation(reservation_id)
        destination = lease.block_ids_by_group["kv"][:len(SOURCE_PAGES)]
        pages = {(lease.partition, page) for page in destination}
        if len(pages) != len(SOURCE_PAGES) or min(destination) <= 0 or pages & self.written_pages:
            raise RuntimeError("expected disjoint guarded Decode pages")
        for owner in self.owners:
            expected = self.expected[owner.rank]
            if owner.rank // self.tp == lease.partition:
                contents = source(self.source_ranks[reservation_id][owner.rank], self.seeds[reservation_id])
                for source_page, dest_page in zip(SOURCE_PAGES, destination):
                    src, dst = source_page * PAGE_BYTES, dest_page * PAGE_BYTES
                    expected[dst:dst + PAGE_BYTES] = contents[src:src + PAGE_BYTES]
            observed = owner.read()
            if observed != expected:
                index = next(i for i, (a, b) in enumerate(zip(observed, expected)) if a != b)
                raise RuntimeError(f"Decode rank {owner.rank} HBM differs at byte {index}")
        print(f"DECODE_HBM_VERIFIED bytes={EXTENT * len(self.owners)} destination_pages={destination} "
              f"partition={lease.partition} before_cache_commit=true", flush=True)
        result = super().commit_group_cache(reservation_id, valid_from=valid_from)
        self.written_pages.update(pages)
        self.commits += 1
        print(f"DECODE_COMMITTED commits={self.commits} state={result.state.value}", flush=True)
        return result


async def decode_run(service, timeout):
    finished = asyncio.get_running_loop().create_future()
    seeds = {endpoint.node_id: index for index, endpoint in enumerate(service.config.prefill)}
    linked = set()
    linked_all = asyncio.Event()
    write_lock = asyncio.Lock()
    entries = []

    async def harness_control(reader, writer):
        # Bootstrap stands in for Router discovery and the generation POST only.
        try:
            await send(writer, service.node)
            binding = decode(await receive(reader), Binding)
            entry = service.register_http(binding, {})
            sender = binding.prefill[0]
            service.cache.seeds[binding.request] = seeds[sender]
            await send(writer, {"admitted": True})
            if await receive(reader) != "check_before_dma":
                raise ValueError("expected pre-DMA cache check")
            service.cache.source_ranks[binding.request] = {target: source for source, target in entry.mapping}
            service.cache.check_before_dma(binding.request)
            await send(writer, {"commits": service.cache.commits})
            if await receive(reader) != "linked" or sender in linked:
                raise ValueError("expected one native link from each Prefill")
            linked.add(sender)
            if linked == seeds.keys():
                linked_all.set()
                print(f"DECODE_NATIVE_LINKS_READY senders={len(linked)}", flush=True)
            await linked_all.wait()
            # Verify the entire shared allocation before each commit. Keep all
            # links and leases live while giving one sender permission to write.
            async with write_lock:
                await send(writer, "write_ready")
                if await receive(reader) != "write_verified":
                    raise ValueError("expected successful native write")
            if await receive(reader) != "prefill_retired":
                raise ValueError("expected Prefill retirement")
            if not entry.source_done.is_set() or not entry.transferred.is_set() or entry.error is not None:
                raise RuntimeError("Decode did not observe a successful handoff and Prefill stop")
            entries.append(entry)
            if len(entries) == len(seeds):
                finished.set_result(None)
        except Exception as exc:
            if not finished.done():
                finished.set_exception(exc)
        finally:
            writer.close()
            await writer.wait_closed()

    listener = await asyncio.start_server(harness_control, service.endpoint.host, service.endpoint.port)
    print(f"DECODE_READY control_port={service.endpoint.control_port} bootstrap_port={service.endpoint.port}", flush=True)
    async with listener:
        await asyncio.wait_for(finished, timeout)
        deadline = asyncio.get_running_loop().time() + timeout
        while service.connections:
            if asyncio.get_running_loop().time() > deadline:
                raise TimeoutError("Prefill control session did not close after retirement")
            await asyncio.sleep(0.01)
    for entry in entries:
        await service.abort(entry, stopped=True)
        service.finish(entry)
    await service.close()


@asynccontextmanager
async def probe_session(remote, timeout):
    reader, writer = await asyncio.wait_for(asyncio.open_connection(remote.host, remote.port), timeout)
    try:
        yield reader, writer
    finally:
        writer.close()
        await writer.wait_closed()


async def prefill_handoff(service, reader, writer, timeout):
    peer = decode(await asyncio.wait_for(receive(reader), timeout), Node)
    binding = Binding.create(service.node, peer, {})
    await send(writer, binding)
    if await asyncio.wait_for(receive(reader), timeout) != {"admitted": True}:
        raise RuntimeError("Decode admission failed")

    async with service.peers.use(peer.key, lambda: service.connect(peer)) as session:
        async def control(operation, **payload):
            return await asyncio.wait_for(session.call(operation, dict(binding=binding, **payload)), timeout)

        continuation = Continuation(TOKENS, {"max_new_tokens": 1}, None, None)
        placement = decode(await control("begin", continuation=continuation), Placement)
        chunk = SimpleNamespace(request=binding.request, chunk=0, end=len(TOKENS), final=True,
                                first_token=102, blocks={"kv": SOURCE_PAGES})
        mapping = rank_mapping(service.node.topology, peer.topology, 0, placement.partition)
        plan = plan_chunk(binding, chunk, placement, service.node.contract, service.cache.group_specs,
                          service.geometry, session.geometry, 0, mapping, digest(continuation), {})
        plan_digest = await control("plan", plan=plan, p_hit=0)
        if plan_digest != digest(plan) or len(plan.copies) != len(SOURCE_PAGES) * len(mapping):
            raise RuntimeError("unexpected write plan authorization")
        try:
            await control("complete", completion=Completion(plan_digest, ()))
        except RuntimeError as exc:
            if str(exc) != "incomplete or mismatched transfer completion":
                raise
        else:
            raise RuntimeError("Decode accepted completion before DMA")
        await send(writer, "check_before_dma")
        if await asyncio.wait_for(receive(reader), timeout) != {"commits": 0}:
            raise RuntimeError("Decode did not verify zero commits before DMA")

        await service.connect_transfer_plan(session, plan, plan_digest)
        await send(writer, "linked")
        if await asyncio.wait_for(receive(reader), timeout) != "write_ready":
            raise RuntimeError("Decode did not release the native write barrier")
        completion = await service.put_transfer_plan(session, plan, plan_digest)
        completion.check(plan_digest, plan.units)
        print(f"PREFILL_NATIVE_COMPLETE ranges={len(plan.copies)} bytes={sum(c.length for c in plan.copies)}", flush=True)
        if await control("complete", completion=completion) != {}:
            raise RuntimeError("Decode rejected valid completion")
        await control("source_done")
        await send(writer, "write_verified")


async def prefill_run(service, timeout):
    async with AsyncExitStack() as stack:
        writers = []
        for remote in service.config.decode:
            reader, writer = await stack.enter_async_context(probe_session(remote, timeout))
            writers.append(writer)
            await prefill_handoff(service, reader, writer, timeout)
        await service.close()
        print("PREFILL_TRANSFER_RETIRED", flush=True)
        for writer in writers:
            await send(writer, "prefill_retired")


async def run(args):
    boot = uuid.uuid4().hex
    if args.deployment:
        config = Config.load(args.deployment)
        local = config.endpoint(args.role, args.node_id)
        if config.provider != "hixl" or config.capacity < len(config.prefill):
            raise ValueError("the probe requires HIXL and one reservation per Prefill")
    else:
        local = Endpoint(args.host, args.port, args.role)
        remote_role = "decode" if args.role == "prefill" else "prefill"
        remote = Endpoint(args.peer_host, args.peer_port, remote_role)
        prefill, decode_endpoint = (local, remote) if args.role == "prefill" else (remote, local)
        config = Config((prefill,), (decode_endpoint,), provider="hixl", capacity=2, log_enabled=False,
                        handoff_timeout=args.timeout, request_timeout=args.timeout)
    settings = WorkerSettings(boot=boot, hostname=local.transfer_hostname or local.host, provider=config.provider,
                              capacity=config.capacity, peer_capacity=len(config.decode),
                              handoff_timeout=args.timeout, role=args.role)
    seed = config.prefill.index(local) if args.role == "prefill" else 0
    owners = tuple(ProbeOwner(settings, device, rank, seed) for rank, device in enumerate(args.devices))
    model = ProbeModel(owners, args.tp)
    progress = parent = child = None
    if args.role == "prefill":
        parent, child = socket.socketpair()
        progress = Progress(parent, args.timeout, config.capacity)
    runtime = WorkerRuntime(settings, model, child)
    for unused in runtime.owners:
        unused.parent.close()
        unused.child.close()
    runtime.owners = owners
    runtime.start_progress()

    async def handle(operation, payload):
        return await asyncio.to_thread(runtime.handle, operation, payload)

    core = SimpleNamespace(kv_cache_manager=CheckedCache(owners, args.tp), scheduler=SimpleNamespace(requests={}),
                           config=SimpleNamespace(max_num_running_reqs=config.capacity,
                                                  runtime_config=SimpleNamespace(max_seq_len=4)),
                           is_ready=lambda: True, call_worker=handle)
    service = Service(config, local, args.role, model, core, progress)
    try:
        await service.start(boot, "hixl-roce-spike")
        if args.role == "prefill":
            await prefill_run(service, args.timeout)
            await progress.close()
        else:
            await decode_run(service, args.timeout)
    except BaseException:
        # Unblock the observer on probe failure without releasing HBM or HIXL registrations.
        if parent is not None:
            parent.close()
        raise
    runtime.close()
    print(f"HIXL_ROCE_SPIKE_PASSED role={args.role} devices={args.devices}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=("prefill", "decode"))
    devices = parser.add_mutually_exclusive_group(required=True)
    devices.add_argument("--device", type=int, help="Logical ACL device ID")
    devices.add_argument("--devices", help="Comma-separated logical ACL IDs, in rank order")
    parser.add_argument("--tp", type=int, help="Ranks per cache partition. Defaults to the device count.")
    parser.add_argument("--deployment", help="Shared replica JSON with provider hixl")
    parser.add_argument("--node-id", help="Select a replica in the shared deployment")
    parser.add_argument("--host", help="Local Serving control-plane IP address for a two-node probe")
    parser.add_argument("--port", type=int, help="Local bootstrap port, PD control port is +10000")
    parser.add_argument("--peer-host", help="Peer Serving control-plane IP address")
    parser.add_argument("--peer-port", type=int, help="Peer bootstrap port, PD control port is +10000")
    parser.add_argument("--timeout", type=float, default=90, help="Control and transfer wait limit in seconds")
    args = parser.parse_args(argv)
    try:
        args.devices = (args.device,) if args.device is not None else tuple(map(int, args.devices.split(",")))
    except ValueError:
        parser.error("device IDs must be integers")
    if args.tp is None:
        args.tp = len(args.devices)
    if (min(args.devices) < 0 or len(set(args.devices)) != len(args.devices)
            or args.tp <= 0 or len(args.devices) % args.tp or not 0 < args.timeout < math.inf):
        parser.error("requires distinct nonnegative device IDs, valid TP, and a positive finite timeout")
    peer_args = (args.host, args.peer_host, args.port, args.peer_port)
    if args.deployment:
        if any(value is not None for value in peer_args):
            parser.error("use --deployment with --node-id in place of explicit peer addresses")
    elif (args.node_id is not None or not all(peer_args)
          or any(not 0 < port < 55536 for port in (args.port, args.peer_port))):
        parser.error("the two-node probe requires local and peer addresses and ports in 1..55535")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

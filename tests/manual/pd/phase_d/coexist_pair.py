# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Weight-free compute-window / Mooncake coexistence test in actual chip owners.

Host invocation overlap is reported separately from device-stream overlap.
Use a device profiler to establish the latter; two Python threads are not proof.
"""
import argparse
import os
from pathlib import Path
import socket
import time
import traceback
import json

from pypto_serving.transfer.agent import AlreadyCompletedFence, TransferAgent
from pypto_serving.transfer.owner import OwnerBridge, OwnerBridgeGroup, _send, _receive
from pypto_serving.transfer.supervisor import OwnerSupervisor
from pypto_serving.transfer.types import (
    CompletionCertainty, OwnerRef, ProviderTransferTask, RegionLease, Segment, Stage, TransferAttemptRef,
)

SIZE = 1024
GUARD_BYTES = 64


def log(kind, **fields):
    print(json.dumps(dict(kind=kind, monotonic_ns=time.monotonic_ns(), **fields)), flush=True)


def intersection_ns(left, right):
    """Compare intervals only on the same host's monotonic clock."""
    return max(0, min(left[1], right[1]) - max(left[0], right[0]))


class ObserveFactory:
    """Test-only native-call timestamps without adding delays or changing payloads."""
    def __init__(self, factory, channel, repeat):
        self.factory, self.channel, self.repeat = factory, channel, repeat

    def __call__(self, context):
        import pypto_serving.transfer.owner as module
        original = module.MooncakeTransferProvider
        channel = self.channel
        repeat = self.repeat

        class EngineObserver:
            def __init__(self, engine):
                self.engine = engine
                self.calls = 0

            def __getattr__(self, name):
                return getattr(self.engine, name)

            def batch_transfer_sync_write(self, endpoint, sources, destinations, lengths):
                # Repetition stays inside this test wrapper, not the bounded production IPC envelope.
                count = 1 if self.calls == 0 else repeat
                self.calls += 1
                sources, destinations, lengths = sources * count, destinations * count, lengths * count
                _send(channel, dict(kind="native_enter", ns=time.monotonic_ns(),
                                    bytes=sum(lengths), descriptors=len(lengths), pid=os.getpid()))
                try:
                    return self.engine.batch_transfer_sync_write(endpoint, sources, destinations, lengths)
                finally:
                    _send(channel, dict(kind="native_return", ns=time.monotonic_ns(), pid=os.getpid()))

        class ProviderObserver(original):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._engine = EngineObserver(self._engine)

        module.MooncakeTransferProvider = ProviderObserver
        return self.factory(context)


def compile_collective(devices, output_dir):
    # Same explicit program surface as the frozen Phase C compiler baseline.
    import pypto.language as pl
    import pypto.language.distributed as pld
    from pypto import ir
    from pypto.ir import DistributedConfig
    nr = pl.dynamic("NR")

    @pl.program
    class ComputeCollective:
        @pl.function(type=pl.FunctionType.InCore)
        def compute_reduce(self, x: pl.Tensor[[1, SIZE], pl.FP32],
                           y: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
                           data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
                           signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]]
                           ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            ctx = pld.get_comm_ctx(data)
            rank = pld.rank(ctx)
            ranks = pld.nranks(ctx)
            value = pl.add(pl.load(x, [0, 0], [1, SIZE]), 1.0)
            data = pl.store(value, [0, 0], data)
            for peer in pl.range(ranks):
                if peer != rank:
                    pld.system.notify(signal, peer=peer, offsets=[rank, 0], value=1,
                                      op=pld.NotifyOp.AtomicAdd)
            for peer in pl.range(ranks):
                if peer != rank:
                    pld.system.wait(signal, offsets=[peer, 0], expected=1, cmp=pld.WaitCmp.Ge)
            acc = pl.load(data, [0, 0], [1, SIZE])
            for peer in pl.range(ranks):
                if peer != rank:
                    other = pld.tile.remote_load(data, peer=peer, offsets=[0, 0], shape=[1, SIZE])
                    acc = pl.add(acc, other)
            return pl.store(acc, [0, 0], y)

        @pl.function(type=pl.FunctionType.Orchestration)
        def child(self, x: pl.Tensor[[1, SIZE], pl.FP32],
                  y: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
                  data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
                  signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]]
                  ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            return self.compute_reduce(x, y, data, signal)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host(self, x: pl.Tensor[[nr, 1, SIZE], pl.FP32],
                 y: pl.Out[pl.Tensor[[nr, 1, SIZE], pl.FP32]]):
            data_buf = pld.alloc_window_buffer([1, SIZE], dtype=pl.FP32)
            signal_buf = pld.alloc_window_buffer(pld.world_size() * pl.INT32.get_byte())
            for rank in pl.range(pld.world_size()):
                data = pld.window(data_buf, [1, SIZE], dtype=pl.FP32)
                signal = pld.window(signal_buf, [pld.world_size(), 1], dtype=pl.INT32)
                self.child(x[rank], y[rank], data, signal, device=rank)

    return ir.compile(ComputeCollective, platform="a2a3", output_dir=str(output_dir),
                      distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0))


def make_task(run_id, rank, iteration, source, destination, repeats):
    if repeats < 1:
        raise ValueError("repeats must be positive")
    ref = TransferAttemptRef(run_id, "coexist", "handoff", f"transfer-{iteration}-{rank}",
                             1, 1, iteration, source.owner, destination.owner, "coexist-v1", iteration + 1)
    segment = Segment("probe", source, destination, GUARD_BYTES, GUARD_BYTES,
                      source.extent - 2 * GUARD_BYTES)
    # Identical writes widen the observation interval; not a throughput benchmark.
    return ProviderTransferTask(ref, (segment,) * repeats)


def run(args):
    import torch
    torch.set_num_threads(1)
    devices = [int(value) for value in args.devices.split(",")]
    if len(devices) < 2 or len(set(devices)) != len(devices):
        raise ValueError("at least two distinct local devices are required")
    ranks = len(devices)
    sender = args.role == "sender"
    owners = [OwnerRef(args.run_id, rank, 1, 1, args.role) for rank in range(ranks)]
    bridges = [OwnerBridge(owner, args.local_host) for owner in owners]
    factories = OwnerBridgeGroup(tuple(bridges)).factories
    observations = []
    if sender:
        for rank in range(ranks):
            read, write = socket.socketpair()
            read.settimeout(90)
            observations.append(read)
            factories[rank] = ObserveFactory(factories[rank], write, args.repeat)
    compiled = compile_collective(devices, args.output_dir)
    if args.compile_only:
        log("compile_result", status="ok")
        return
    host = torch.stack([torch.arange(SIZE, dtype=torch.float32).reshape(1, SIZE) + rank * 100
                        for rank in range(ranks)]).share_memory_()
    rt = compiled.prepare(chip_service_factories=factories, inherited_host_tensors=[host])
    agents, conn, listener, clean = [], None, None, False
    try:
        for bridge in bridges:
            bridge.ready()
        log("owners_ready", role=args.role, pids=[bridge.owner_pid for bridge in bridges])
        payload = rt.alloc_stacked_tensor(torch.full_like(host, -99))
        scratch = rt.alloc_stacked_tensor(torch.zeros_like(host))
        after = rt.alloc_stacked_tensor(torch.zeros_like(host))
        leases = [RegionLease(owner, "probe", 1, SIZE * 4) for owner in owners]
        envelopes = [bridge.register_tensor(rt, payload.shards[rank], leases[rank])
                     for rank, bridge in enumerate(bridges)]

        def compute(inp, output, label):
            start = time.monotonic_ns()
            log("compute_enter", label=label, role=args.role)
            rt(inp, output)
            end = time.monotonic_ns()
            log("compute_return", label=label, role=args.role, start_ns=start, end_ns=end)
            return start, end

        def validate(arena, expected, label):
            actual = torch.empty_like(host)
            rt.copy_stacked_from(arena, actual)
            if not torch.equal(actual, expected):
                raise AssertionError(f"{label}: max error {float((actual - expected).abs().max())}")
            log("validated", label=label, role=args.role)

        golden = (host + 1).sum(dim=0, keepdim=True).expand_as(host)
        compute(host, scratch, "warmup")
        validate(scratch, golden, "warmup")
        if sender:
            conn = socket.create_connection((args.receiver_host, args.port), timeout=120)
        else:
            listener = socket.socket()
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((args.local_host, args.port))
            listener.listen(1)
            listener.settimeout(180)
            log("receiver_ready", run_id=args.run_id)
            conn, _ = listener.accept()
        conn.settimeout(120)

        def send(kind, **fields):
            _send(conn, dict(kind=kind, run_id=args.run_id, ranks=ranks, **fields))

        def receive(kind):
            msg = _receive(conn)
            if (msg.get("kind"), msg.get("run_id"), msg.get("ranks")) != (kind, args.run_id, ranks):
                raise RuntimeError("control identity mismatch")
            return msg

        if sender:
            remote = receive("arenas")
            destinations = [RegionLease(**dict(e["lease"], owner=OwnerRef(**e["lease"]["owner"])))
                            for e in remote["envelopes"]]
            for rank, bridge in enumerate(bridges):
                bridge.install_destination(destinations[rank], remote["envelopes"][rank])
                supervisor = OwnerSupervisor(owners[rank], tuple(b.process_handle for b in bridges),
                                             emit=lambda fact: log("recovery", fact=str(fact)),
                                             local_owners=tuple(owners),
                                             peer_owners=tuple(lease.owner for lease in destinations))
                agents.append(TransferAgent(owners[rank], bridge, poison=supervisor.poison))
        else:
            send("arenas", envelopes=envelopes)

        for iteration in range(args.rounds):
            overlap = iteration > 0
            # Different source data catches stale DMA, while the overlap computation uses separate memory.
            host.add_(10)
            golden = (host + 1).sum(dim=0, keepdim=True).expand_as(host)
            compute(host, payload if sender else scratch, f"produce-{iteration}")
            validate(payload if sender else scratch, golden, f"produce-{iteration}")
            if sender:
                send("begin", iteration=iteration)
                receive("ready")
                futures = [agent.submit(make_task(args.run_id, rank, iteration, leases[rank],
                                                  destinations[rank], 1),
                                         AlreadyCompletedFence(), timeout=60)
                           for rank, agent in enumerate(agents)]
                enters = [_receive(channel) for channel in observations]
                if any(item["kind"] != "native_enter" for item in enters):
                    raise AssertionError("missing native entry")
                intervals = []
                if overlap:
                    for step in range(args.compute_rounds):
                        intervals.append(compute(host, scratch, f"overlap-{iteration}-{step}"))
                    validate(scratch, golden, "independent_compute")
                for future in futures:
                    event = future.result(90)
                    if event.stage != Stage.COMPLETED or event.certainty != CompletionCertainty.COMPLETED:
                        raise AssertionError(f"uncertain transfer: {event.to_dict()}")
                exits = [_receive(channel) for channel in observations]
                for rank, (enter, leave) in enumerate(zip(enters, exits)):
                    if leave["kind"] != "native_return":
                        raise AssertionError("missing native return")
                    intersections = [intersection_ns((enter["ns"], leave["ns"]), interval)
                                     for interval in intervals]
                    log("native_interval", rank=rank, start_ns=enter["ns"], end_ns=leave["ns"],
                        host_compute_overlap_ns=sum(intersections), iteration=iteration)
                    if overlap and not any(intersections):
                        raise AssertionError("no observed host invocation overlap; not an overlap pass")
                send("complete")
                receive("validated")
                validate(payload, golden, "source_unchanged")
                compute(payload, after, f"after-transfer-{iteration}")
                validate(after, (golden + 1).sum(dim=0, keepdim=True).expand_as(host), "sender_after")
            else:
                msg = receive("begin")
                if msg["iteration"] != iteration:
                    raise AssertionError("iteration mismatch")
                send("ready")
                if overlap:
                    for step in range(args.compute_rounds):
                        compute(host, scratch, f"overlap-{iteration}-{step}")
                    validate(scratch, golden, "independent_compute")
                receive("complete")
                expected = golden.clone()
                guard = GUARD_BYTES // 4
                expected[:, :, :guard] = -99
                expected[:, :, -guard:] = -99
                validate(payload, expected, "received_payload_and_guards")
                compute(payload, after, f"after-transfer-{iteration}")
                validate(after, (expected + 1).sum(dim=0, keepdim=True).expand_as(host), "receiver_after")
                send("validated")

        if sender:
            for agent in agents:
                agent.close()
            for bridge in bridges:
                bridge.release()
                bridge.close()
            send("sender_stopped")
            receive("receiver_stopped")
        else:
            receive("sender_stopped")
            for bridge in bridges:
                bridge.release()
                bridge.close()
            send("receiver_stopped")
        rt.close()
        clean = True
        log("coexist_result", run_id=args.run_id, role=args.role, ranks=ranks, status="ok",
            rounds=args.rounds, host_overlap=args.rounds > 1, device_stream_overlap="requires_profiler")
    finally:
        if not clean:
            traceback.print_exc()
            # Never free possibly in-flight registered memory after a test failure.
            for bridge in bridges:
                if bridge.process_handle is not None and bridge.process_handle.fd is not None:
                    bridge.process_handle.kill()
                    bridge.process_handle.wait(10)
            log("coexist_failed_closed", run_id=args.run_id)
            os._exit(2)
        for channel in observations:
            channel.close()
        if conn:
            conn.close()
        if listener:
            listener.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("sender", "receiver"), required=True)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--local-host", required=True)
    parser.add_argument("--receiver-host", required=True)
    parser.add_argument("--port", type=int, default=29821)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=16384)
    parser.add_argument("--compute-rounds", type=int, default=4)
    parser.add_argument("--compile-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.rounds < 1 or arguments.compute_rounds < 1 or arguments.repeat < 1:
        parser.error("round counts and repeat must be positive")
    run(arguments)

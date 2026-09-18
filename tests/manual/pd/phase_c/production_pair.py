# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Production-path Phase C pair smoke test (reduced arenas, not model inference).

Control envelopes travel only on the dedicated private socket and are never logged.
No Phase B hooks are imported. Run only on explicitly reserved, idle devices.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import socket
import traceback
import threading
import time

from faults import DeadPeerFence, InflightFactory, LostReplyFactory, confirm_group, decode_event, previous_generation_task

from pypto_serving.model.deepseek.transfer_layout import COMPONENTS, BlockCopy, ComponentLayout, DSV4Registry
from pypto_serving.transfer.agent import AlreadyCompletedFence, TransferAgent
from pypto_serving.transfer.errors import TransferFailure
from pypto_serving.transfer.owner import OwnerBridge, OwnerBridgeGroup, _receive, _send
from pypto_serving.transfer.supervisor import OwnerSupervisor
from pypto_serving.transfer.types import CompletionCertainty, OwnerRef, RegionLease, Stage, TransferAttemptRef


def compile_source(devices, output_dir):
    import pypto.language as pl
    import pypto.language.distributed as pld
    from pypto import ir
    from pypto.ir import DistributedConfig

    @pl.program
    class SourceKernel:
        @pl.function(type=pl.FunctionType.InCore)
        def add_one(self, x: pl.Tensor[[128, 128], pl.FP32],
                    y: pl.Out[pl.Tensor[[128, 128], pl.FP32]]) -> pl.Tensor[[128, 128], pl.FP32]:
            a = pl.load(x, [0, 0], [128, 128])
            return pl.store(pl.add(a, 1.0), [0, 0], y)

        @pl.function(type=pl.FunctionType.Orchestration)
        def child(self, x: pl.Tensor[[128, 128], pl.FP32],
                  y: pl.Out[pl.Tensor[[128, 128], pl.FP32]]) -> pl.Tensor[[128, 128], pl.FP32]:
            return self.add_one(x, y)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host(self, x: pl.Tensor[[pl.dynamic("NR"), 128, 128], pl.FP32],
                 y: pl.Out[pl.Tensor[[pl.dynamic("NR"), 128, 128], pl.FP32]]):
            for rank in pl.range(pld.world_size()):
                self.child(x[rank], y[rank], device=rank)

    return ir.compile(SourceKernel, platform="a2a3", output_dir=str(output_dir),
                      distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0))


def log(kind, **fields):
    print(json.dumps(dict(kind=kind, **fields)), flush=True)


def run(args):
    import torch
    torch.set_num_threads(1)
    devices = [int(v) for v in args.devices.split(",")]
    ranks = len(devices)
    compiled = compile_source(devices, args.output_dir)
    sender = args.role == "sender"
    gen = args.generation
    owners = tuple(OwnerRef(args.run_id, r, gen, gen, args.role) for r in range(ranks))
    peers = tuple(OwnerRef(args.run_id, r, gen, gen, "receiver" if sender else "sender") for r in range(ranks))
    bridges = tuple(OwnerBridge(owner, args.local_host) for owner in owners)
    group = OwnerBridgeGroup(bridges)
    factories = group.factories
    fault = args.lost_reply or args.peer_loss_inflight
    observation = None
    if sender and args.lost_reply:
        factories[0] = LostReplyFactory(factories[0])
    if sender and args.peer_loss_inflight:
        observation, child_observation = socket.socketpair()
        observation.settimeout(45)
        factories[0] = InflightFactory(factories[0], child_observation)
    # All factories and all shared dispatch inputs exist before the single packed worker forks.
    host_input = torch.arange(ranks * 128 * 128, dtype=torch.float32).reshape(ranks, 128, 128)
    host_input.share_memory_()
    rt = compiled.prepare(chip_service_factories=factories, inherited_host_tensors=[host_input])
    agents = []
    conn = listener = None
    clean = False
    try:
        for bridge in bridges:
            bridge.ready()
        log("owners_ready", run_id=args.run_id, parent_pid=os.getpid(),
            owner_pids=[b.owner_pid for b in bridges], ranks=ranks)
        registry = DSV4Registry("phase-c-reduced-dsv4", (ranks,), tuple(
            ComponentLayout(name, "torch.float32", 4, (0,), 2, 64,
                            4 if name == "idx_scale" else 512) for name in COMPONENTS))
        arenas, initial = {}, {}
        for index, name in enumerate(COMPONENTS):
            shape = (ranks, 128, 1 if name == "idx_scale" else 128)
            host = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape)
            host.add_(index * 100000)
            initial[name] = host.clone()
            arenas[name] = rt.alloc_stacked_tensor(host if sender else torch.full(shape, -99.0))
        if sender:
            rt(host_input, arenas["ori"])
            initial["ori"] = host_input + 1.0
        leases = [{name: RegionLease(owners[r], name, 1, entry.extent)
                   for name in COMPONENTS for entry in (registry.entry(name),)} for r in range(ranks)]
        envelopes = [{name: bridges[r].register_tensor(rt, arenas[name].shards[r], leases[r][name])
                      for name in COMPONENTS} for r in range(ranks)]
        # Parent-side free must reject registered allocations before reaching native free.
        for r in range(ranks):
            try:
                rt.free_tensor(arenas["ori"].shards[r], worker_id=r)
            except RuntimeError:
                pass
            else:
                raise AssertionError("registered arena was freed")

        if not sender:
            listener = socket.socket()
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.settimeout(180)
            listener.bind((args.local_host, args.port))
            listener.listen(1)
            log("receiver_ready", run_id=args.run_id)
            conn, _ = listener.accept()
        else:
            connect_deadline = time.monotonic() + 120
            while True:
                try:
                    conn = socket.create_connection((args.receiver_host, args.port), timeout=10)
                    break
                except ConnectionRefusedError:
                    if time.monotonic() >= connect_deadline:
                        raise
                    threading.Event().wait(0.1)
        conn.settimeout(90)

        def send(kind, **fields):
            _send(conn, dict(kind=kind, run_id=args.run_id, **fields))

        def receive(kind):
            message = _receive(conn)
            if message.get("kind") != kind or message.get("run_id") != args.run_id:
                raise RuntimeError("peer control identity mismatch")
            return message

        def retire_failed_generation(supervisor):
            if not supervisor.dead or not supervisor.peer_dead:
                raise RuntimeError("paired owner death is required before worker retirement")
            for agent in agents:
                agent.close()
            for bridge in bridges:
                # Every native owner is already dead; retain pins and never call release/unregister.
                if any(item["operation"] in ("release", "stop") for item in bridge.commands):
                    raise AssertionError("failed generation entered normal release")
                bridge.close_after_owner_death()
            try:
                rt.close()
            except RuntimeError as exc:
                # An abnormal child exit is a memoized close outcome, not a retryable fence.
                # Do not infer reclamation or permit same-process reuse from owner death alone.
                log("retired_runtime_close_error", error_type=type(exc).__name__)
            # No in-process reuse: the external controller must also reap this parent.
            log("worker_retirement_certificate", run_id=args.run_id, generation=gen,
                role=args.role, parent_pid=os.getpid(), owners=[asdict(o) for o in owners],
                peers=[asdict(o) for o in peers])
            os._exit(75)

        copies = tuple(BlockCopy(name, 0, 0, 1, 1) for name in COMPONENTS)
        if sender:
            peer = receive("arenas")
            if (
                peer["fingerprint"] != registry.fingerprint
                or peer["layout_fingerprint"] != registry.layout_fingerprint
                or len(peer["envelopes"]) != ranks
            ):
                raise ValueError("peer layout/rank mismatch")
            if args.pre_submit_peer_loss:
                dead = receive("pre_submit_peer_dead")
                assert dead["owners"] == [asdict(o) for o in peers]
            futures = []
            for r, bridge in enumerate(bridges):
                if fault and r != 0:
                    break
                destination = {name: RegionLease(peers[r], name, 1, registry.entry(name).extent)
                               for name in COMPONENTS}
                for name in COMPONENTS:
                    bridge.install_destination(destination[name], peer["envelopes"][r][name])
                ref = TransferAttemptRef("request", "plan", "handoff", "attempt", gen, gen, 0,
                                         owners[r], peers[r], registry.manifest_hash(copies, True))
                task = registry.lower(ref, copies, final=True,
                                      destination_layout_fingerprint=peer["layout_fingerprint"],
                                      source_leases=leases[r], destination_leases=destination)
                supervisor = OwnerSupervisor(owners[r], tuple(b.process_handle for b in bridges),
                                             emit=lambda event: log("recovery", **asdict(event)),
                                             local_owners=owners, peer_owners=peers)
                agent = TransferAgent(owners[r], bridge, poison=supervisor.poison)
                agents.append(agent)
                if r == 0 and args.reject_stale:
                    stale = previous_generation_task(task)
                    assert agent.status.query(stale.attempt).kind == "stale_owner"
                    before = tuple(bridge.commands)
                    try:
                        bridge.write(stale)
                    except TransferFailure as exc:
                        assert exc.error.certainty == CompletionCertainty.NOT_SUBMITTED
                    else:
                        raise AssertionError("stale source owner accepted")
                    assert tuple(bridge.commands) == before
                    log("stale_generation_rejected", generation=gen)
                fence = DeadPeerFence() if args.pre_submit_peer_loss else AlreadyCompletedFence()
                futures.append(agent.submit(task, fence, timeout=40))
            if args.peer_loss_inflight:
                entered = _receive(observation)
                assert entered["kind"] == "native_enter"
                # Scheduling grace is not evidence: the return/death timestamps below are the gate.
                threading.Event().wait(0.05)
                send("kill_peer")
                receive("peer_killed")
                death_observed_ns = time.monotonic_ns()
                returned = _receive(observation)
                assert returned["kind"] == "native_return"
                assert entered["monotonic_ns"] < death_observed_ns < returned["monotonic_ns"], (
                    "native invocation did not overlap peer death; cannot count this fault")
                log("native_inflight_peer_loss_proof", entered_ns=entered["monotonic_ns"],
                    peer_death_observed_ns=death_observed_ns, returned_ns=returned["monotonic_ns"],
                    descriptors=entered["descriptors"], bytes=entered["bytes"])
            for future in futures:
                event = future.result(50)
                log("transfer_event", **event.to_dict())
                if args.pre_submit_peer_loss:
                    assert event.stage == Stage.FAILED and event.certainty == CompletionCertainty.NOT_SUBMITTED
                    continue
                if fault:
                    assert event.stage == Stage.FAILED and event.certainty == CompletionCertainty.UNKNOWN
                    assert supervisor.dead and not supervisor.peer_dead
                    send("uncertain", event=event.to_dict())
                    confirm_group(supervisor, receive("peer_dead"), peers)
                    send("local_dead", owners=[asdict(owner) for owner in owners])
                    retire_failed_generation(supervisor)
                    clean = True
                    return
                if event.stage != Stage.COMPLETED:
                    raise RuntimeError("transfer did not complete")
            if args.pre_submit_peer_loss:
                assert not supervisor.events
                assert not any(c["operation"] == "write" for b in bridges for c in b.commands)
                # The healthy source still executes compute in exactly the same runtime.
                rt(host_input, arenas["ori"])
                for agent in agents:
                    agent.close()
                for bridge in bridges:
                    bridge.release()
                    bridge.close()
                rt.close()
                send("source_closed_without_poison")
                receive("peer_retired")
                log("pre_submit_loss_result", run_id=args.run_id, role=args.role,
                    ranks=ranks, status="ok", native_writes=0, source_reused=True)
                clean = True
                return
            send("complete")
            receive("validated")
            for agent in agents:
                agent.close()
            for bridge in bridges:
                bridge.release()
                bridge.close()
            send("sender_stopped")
            receive("receiver_stopped")
        else:
            send(
                "arenas",
                fingerprint=registry.fingerprint,
                layout_fingerprint=registry.layout_fingerprint,
                envelopes=envelopes,
            )
            if args.pre_submit_peer_loss:
                for bridge in bridges:
                    bridge.process_handle.kill()
                for bridge in bridges:
                    bridge.process_handle.wait(5)
                send("pre_submit_peer_dead", owners=[asdict(o) for o in owners])
                receive("source_closed_without_poison")
                for bridge in bridges:
                    bridge.close_after_owner_death()
                try:
                    rt.close()
                except RuntimeError as exc:
                    log("retired_runtime_close_error", error_type=type(exc).__name__)
                send("peer_retired")
                log("pre_submit_loss_result", run_id=args.run_id, role=args.role, ranks=ranks, status="ok")
                # Terminated destination runtime is never reused; source closed normally.
                os._exit(0)
            if fault:
                if args.peer_loss_inflight:
                    receive("kill_peer")
                    bridges[0].process_handle.kill()
                    bridges[0].process_handle.wait(5)
                    send("peer_killed")
                event = decode_event(receive("uncertain")["event"])
                supervisor = OwnerSupervisor(owners[0], tuple(b.process_handle for b in bridges),
                                             emit=lambda fact: log("recovery", **asdict(fact)),
                                             local_owners=owners, peer_owners=peers)
                supervisor.poison(event)
                send("peer_dead", owners=[asdict(owner) for owner in owners])
                confirm_group(supervisor, receive("local_dead"), peers)
                retire_failed_generation(supervisor)
                clean = True
                return
            receive("complete")
            for name in COMPONENTS:
                readback = torch.empty_like(initial[name])
                rt.copy_stacked_from(arenas[name], readback)
                expected = torch.full_like(readback, -99.0)
                source = host_input + 1.0 if name == "ori" else initial[name]
                expected[:, 64:128] = source[:, :64]
                if not torch.equal(expected, readback):
                    raise AssertionError("payload or untouched-page mismatch: " + name)
            send("validated")
            receive("sender_stopped")
            for bridge in bridges:
                bridge.release()
                bridge.close()
            send("receiver_stopped")
        rt.close()
        clean = True
        log("phase_c_result", run_id=args.run_id, role=args.role, ranks=ranks,
            components=len(COMPONENTS), status="ok", payload_validated=True,
            source_kernel=True, generation=gen)
    finally:
        if conn is not None:
            conn.close()
        if listener is not None:
            listener.close()
        if not clean:
            traceback.print_exc()
            # A harness failure cannot enter normal allocator cleanup with potentially live DMA.
            # Terminate ONLY the chip children captured by this invocation's pidfds.
            for bridge in bridges:
                if bridge.process_handle is not None and bridge.process_handle.fd is not None:
                    bridge.process_handle.kill()
                    bridge.process_handle.wait(10)
            log("harness_failed_closed", run_id=args.run_id)
            os._exit(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("sender", "receiver"), required=True)
    parser.add_argument("--devices", default="0")
    parser.add_argument("--local-host", required=True)
    parser.add_argument("--receiver-host", required=True)
    parser.add_argument("--port", type=int, default=29761)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lost-reply", action="store_true")
    parser.add_argument("--peer-loss-inflight", action="store_true")
    parser.add_argument("--pre-submit-peer-loss", action="store_true")
    parser.add_argument("--generation", type=int, default=1)
    parser.add_argument("--worker-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--reject-stale", action="store_true")
    args = parser.parse_args()
    if args.generation < 1 or (args.reject_stale and args.generation < 2):
        parser.error("invalid generation")
    if sum((args.lost_reply, args.peer_loss_inflight, args.pre_submit_peer_loss)) > 1:
        parser.error("choose one fault")
    if (args.lost_reply or args.peer_loss_inflight) and not args.worker_child:
        from recovery_pair import supervise_pair
        supervise_pair(args)
    else:
        run(args)

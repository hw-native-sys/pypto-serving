# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Transfer control must progress independently without weakening buffer lifetime."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import socket
import threading
import time
from types import SimpleNamespace

import msgspec
import pytest

from pypto_serving.pd.service import Service
from pypto_serving.pd.transfer import TransferError
from pypto_serving.pd.worker import Progress, ProgressServer, _receive_frame, _send_frame
from pypto_serving.pd.adapter import DeepSeekDSpark
from pypto_serving.pd.config import Config, Endpoint
from pypto_serving.pd.integration import Chunk
from pypto_serving.pd.protocol import Completion, MAX_FRAME, Placement, digest
from tests.unit.pd.test_service import handoff
from tests.unit.test_pd_metadata import finish, peer, plan, runtime as runtime


@asynccontextmanager
async def connected(runtime, *, handler=None, timeout=2, capacity=2):
    parent, child = socket.socketpair()
    server = ProgressServer(child, handler or runtime.handle_progress, timeout)
    progress = Progress(parent, timeout, capacity)
    try:
        yield progress
    finally:
        await progress.close()
        server.close()
        assert not server.thread.is_alive() and not progress.calls and not progress.lock.locked()


async def call(progress, operation, value):
    result = msgspec.msgpack.decode(await progress.call(operation, msgspec.msgpack.encode(value)))
    if isinstance(result, dict) and "error" in result:
        raise TransferError(result["error"], uncertain=result["uncertain"])
    return result


def test_submit_and_native_completion_do_not_wait_for_unrelated_compute(runtime):
    started, release = threading.Event(), threading.Event()
    def compute():
        started.set()
        assert release.wait(5)
    thread = threading.Thread(target=compute)
    thread.start()
    try:
        assert started.wait(2)
        remote = peer(runtime)
        runtime.install(remote)
        async def run():
            async with connected(runtime) as progress:
                value = plan(runtime, remote)
                job = (await call(progress, "submit", value))["job"]
                completion = await progress.wait(job)
                completion.check(job, value.units)
                assert not release.is_set() and thread.is_alive()
                assert not runtime.jobs and all(owner.writes == 1 for owner in runtime.owners)
        asyncio.run(run())
    finally:
        release.set()
        thread.join(2)


def test_service_uses_independent_channel_only_for_submit_and_ready_state(runtime):
    remote = peer(runtime)
    runtime.install(remote)
    runtime.model.source_state = lambda request: {"bootstrap": 7}
    ordered = []
    async def ordered_call(operation, payload):
        ordered.append(operation)
        return msgspec.msgpack.encode(None)
    async def run():
        async with connected(runtime) as progress:
            service = object.__new__(Service)
            service.progress = progress
            service.core = type("Core", (), {"call_worker": staticmethod(ordered_call)})()
            assert await service.worker("source_state", "request") == {"bootstrap": 7}
            job = await service.worker("submit", plan(runtime, remote))
            await progress.wait(job)
            for operation in ("registry", "install", "reserve_state", "discard_state", "release"):
                await service.worker(operation)
            assert ordered == ["registry", "install", "reserve_state", "discard_state", "release"]
    asyncio.run(run())


def test_poll_is_not_starved_by_queries_and_multiple_peers_remain_correlated(runtime):
    remotes = peer(runtime), peer(runtime, "other-boot")
    for remote in remotes:
        runtime.install(remote)
    runtime.model.source_state = lambda request: {"bootstrap": int(request)}
    operations = []
    def handle(operation, payload):
        operations.append(operation)
        return runtime.handle_progress(operation, payload)
    async def run():
        async with connected(runtime, handler=handle) as progress:
            values = [plan(runtime, remote, str(i)) for i, remote in enumerate(remotes)]
            replies = await asyncio.gather(*(call(progress, "submit", value) for value in values))
            waits = [asyncio.create_task(progress.wait(reply["job"])) for reply in replies]
            for number in range(8):
                assert await call(progress, "source_state", str(number)) == {"bootstrap": number}
            completions = await asyncio.gather(*waits)
            for reply, value, completion in zip(replies, values, completions):
                completion.check(reply["job"], value.units)
            assert "poll" in operations and not runtime.jobs
            with pytest.raises(KeyError):
                runtime.poll((replies[0]["job"],))
    asyncio.run(run())


def test_bad_plan_is_rejected_before_any_rank_writes_and_whitelist_is_narrow(runtime):
    remote = peer(runtime)
    runtime.install(remote)
    value = plan(runtime, remote)
    bad = msgspec.structs.replace(value, copies=(value.copies[0],
        msgspec.structs.replace(value.copies[1], destination_offset=4096)))
    async def run():
        async with connected(runtime) as progress:
            for operation, payload in (("submit", bad), ("release", None), ("reserve_state", {})):
                with pytest.raises(TransferError) as error:
                    await call(progress, operation, payload)
                assert not error.value.uncertain
            assert not runtime.jobs and not any(owner.writes for owner in runtime.owners)
    asyncio.run(run())


def test_capacity_is_bounded_and_completion_poll_does_not_spend_command_capacity(runtime):
    remote = peer(runtime)
    runtime.install(remote)
    async def run():
        async with connected(runtime, capacity=1) as progress:
            await progress.lock.acquire()
            queued = asyncio.create_task(call(progress, "submit", plan(runtime, remote)))
            await asyncio.sleep(0)  # Let the caller reach the held channel lock.
            with pytest.raises(TransferError, match="capacity") as error:
                await call(progress, "submit", plan(runtime, remote, "overflow"))
            assert not error.value.uncertain and progress.calls == 1 and not runtime.jobs
            progress.lock.release()
            job = (await queued)["job"]
            await progress.wait(job)
            assert not runtime.jobs
    asyncio.run(run())


def test_job_capacity_rejects_before_enqueue_and_drains_normally(runtime):
    remote = peer(runtime)
    runtime.install(remote)
    async def run():
        async with connected(runtime) as progress:
            jobs = [(await call(progress, "submit", plan(runtime, remote, str(i))))["job"] for i in range(2)]
            with pytest.raises(TransferError) as error:
                await call(progress, "submit", plan(runtime, remote, "overflow"))
            assert not error.value.uncertain and len(runtime.jobs) == 2
            await asyncio.gather(*(progress.wait(job) for job in jobs))
            assert not runtime.jobs and all(owner.writes == 2 for owner in runtime.owners)
    asyncio.run(run())


def test_frame_limit_and_queued_deadline_are_known_failures_without_channel_poisoning(runtime):
    runtime.model.source_state = lambda request: {"bootstrap": 1}
    async def run():
        async with connected(runtime, timeout=.1) as progress:
            with pytest.raises(TransferError, match="frame limit") as error:
                await progress.call("submit", b"x" * MAX_FRAME)
            assert not error.value.uncertain and progress.error is None
            await progress.lock.acquire()
            try:
                with pytest.raises(TransferError, match="waited for channel") as error:
                    await call(progress, "source_state", "request")
                assert not error.value.uncertain and not progress.calls and progress.error is None
            finally:
                progress.lock.release()
            assert await call(progress, "source_state", "request") == {"bootstrap": 1}
    asyncio.run(run())


def test_partial_rank_enqueue_failure_is_unknown_and_prevents_unregister(runtime, monkeypatch):
    remote = peer(runtime)
    runtime.install(remote)
    original = runtime.pool.submit
    calls = []
    def partial(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            raise RuntimeError("cannot enqueue second rank")
        return original(*args, **kwargs)
    monkeypatch.setattr(runtime.pool, "submit", partial)
    async def run():
        async with connected(runtime) as progress:
            with pytest.raises(TransferError) as error:
                await call(progress, "submit", plan(runtime, remote))
            assert error.value.uncertain and runtime.uncertain
            with pytest.raises(TransferError):
                runtime.handle("release", b"")
    asyncio.run(run())


def test_unexpected_submit_exception_cannot_be_reported_as_no_write(runtime, monkeypatch):
    remote = peer(runtime)
    runtime.install(remote)
    def unexpected(value):
        raise RuntimeError("unexpected submit failure")
    monkeypatch.setattr(runtime, "submit", unexpected)
    async def run():
        async with connected(runtime) as progress:
            with pytest.raises(TransferError) as error:
                await call(progress, "submit", plan(runtime, remote))
            assert error.value.uncertain and runtime.uncertain
    asyncio.run(run())


def test_cancel_before_channel_ownership_never_submits(runtime):
    remote = peer(runtime)
    runtime.install(remote)
    async def run():
        async with connected(runtime) as progress:
            await progress.lock.acquire()
            task = asyncio.create_task(call(progress, "submit", plan(runtime, remote)))
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            progress.lock.release()
            assert not runtime.jobs and not any(owner.writes for owner in runtime.owners)
    asyncio.run(run())


def test_cancel_after_acceptance_drains_late_ack_before_next_reader(runtime):
    remote = peer(runtime)
    runtime.install(remote)
    accepted, ack = threading.Event(), threading.Event()
    runtime.model.source_state = lambda request: {"bootstrap": 9}
    def handle(operation, payload):
        result = runtime.handle_progress(operation, payload)
        if operation == "submit":
            accepted.set()
            assert ack.wait(5)
        return result
    async def run():
        async with connected(runtime, handler=handle) as progress:
            task = asyncio.create_task(call(progress, "submit", plan(runtime, remote)))
            try:
                assert await asyncio.to_thread(accepted.wait, 2)
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done() and progress.lock.locked()
            finally:
                ack.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert await call(progress, "source_state", "next") == {"bootstrap": 9}
            assert len(runtime.jobs) == 1  # A drained ack did not prove native completion to its cancelled caller.
            finish(runtime, next(iter(runtime.jobs)))
    asyncio.run(run())


@pytest.mark.parametrize("failure", ["eof", "wrong_sequence", "timeout"])
def test_lost_submit_reply_is_unknown_and_channel_is_not_reused(runtime, failure):
    remote = peer(runtime)
    runtime.install(remote)
    parent, child = socket.socketpair()
    release = threading.Event()
    def respond():
        try:
            sequence, operation, payload = msgspec.msgpack.decode(_receive_frame(child, time.monotonic() + 2))
            result = runtime.handle_progress(operation, payload)
            if failure == "wrong_sequence":
                _send_frame(child, msgspec.msgpack.encode((sequence + 1, result)), time.monotonic() + 2)
            elif failure == "timeout":
                release.wait(5)
        finally:
            child.close()
    thread = threading.Thread(target=respond)
    thread.start()
    async def run():
        progress = Progress(parent, .1, 2)
        try:
            with pytest.raises(TransferError) as error:
                await call(progress, "submit", plan(runtime, remote))
            assert error.value.uncertain and runtime.jobs
            with pytest.raises(TransferError) as repeated:
                await call(progress, "source_state", "request")
            assert repeated.value.uncertain
            with pytest.raises(TransferError, match="unsafe or active"):
                runtime.handle("release", b"")
        finally:
            release.set()
            await progress.close()
    try:
        asyncio.run(run())
    finally:
        release.set()
        thread.join(2)
        parent.close()
    assert not thread.is_alive()


@pytest.mark.parametrize("uncertain", [False, True])
def test_native_failure_keeps_its_existing_certainty(runtime, uncertain):
    remote = peer(runtime)
    runtime.install(remote)
    def fail(*args, **kwargs):
        raise TransferError("native failed", uncertain=uncertain)
    runtime.owners[1].call = fail
    async def run():
        async with connected(runtime) as progress:
            job = (await call(progress, "submit", plan(runtime, remote)))["job"]
            with pytest.raises(TransferError) as error:
                await progress.wait(job)
            assert error.value.uncertain == uncertain and runtime.uncertain == uncertain
            assert not runtime.jobs
    asyncio.run(run())


def test_submit_racing_close_cannot_write_after_unregister(runtime, monkeypatch):
    remote = peer(runtime)
    runtime.install(remote)
    entered, release = threading.Event(), threading.Event()
    lower = runtime._lower
    def blocked(plan):
        entered.set()
        assert release.wait(5)
        return lower(plan)
    monkeypatch.setattr(runtime, "_lower", blocked)
    with ThreadPoolExecutor(2) as pool:
        submitted = pool.submit(runtime.submit, plan(runtime, remote))
        try:
            assert entered.wait(2)
            closed = pool.submit(runtime.close)
        finally:
            release.set()
        job = submitted.result(timeout=2)
        with pytest.raises(RuntimeError, match="safely drained"):
            closed.result(timeout=2)
    assert not runtime.accepting and job in runtime.jobs
    finish(runtime, job)
    runtime.handle("release", b"")
    with pytest.raises(TransferError) as error:
        runtime.submit(plan(runtime, remote, "late"))
    assert not error.value.uncertain
    runtime.close()


@pytest.mark.parametrize("failure", ["none", "not_ready", "plan", "known_submit", "lost_submit", "cancel_submit"])
def test_sender_preserves_authorization_pins_and_completion_boundary(failure):
    async def run():
        case = handoff(128)
        binding, request = case.entry.binding, case.entry.binding.request
        finished, confirmed, authorized, proof = (asyncio.Event() for _ in range(4))
        consumers, sent, stops = [], [], []
        cfg = Config((Endpoint("p", 8111, "p"),), (Endpoint("d", 8112, "d"),),
                     log_enabled=False, handoff_timeout=2, request_timeout=3)
        async def consume(*args, **kwargs):
            consumers.append(asyncio.current_task())
            blocks = case.p.ensure_group_blocks(case.source_request, 128, partition=0)
            confirmed.set()
            source.chunks[request].put_nowait(Chunk(request, 0, 0, 128, True, 7,
                {k: tuple(v) for k, v in blocks.items()}, 0))
            await finished.wait()
            yield None
        async def abort(request):
            finished.set()
        core = SimpleNamespace(kv_cache_manager=case.p, add_request=consume, abort_request=abort,
            finish_prefilled_request=lambda _: finished.set(), is_ready=lambda: True)
        source = Service(cfg, cfg.prefill[0], "prefill", DeepSeekDSpark(), core, None)
        source.node, source.geometry = case.nodes[0], case.source_geometry
        observed = []
        source.observe = lambda name, entry, **fields: observed.append((name, fields))
        entry = source.register_http(binding, {})
        reservation = case.entry.reservation
        async def control(operation, payload):
            if operation == "begin":
                return Placement(request, 0, reservation.partition, reservation.block_ids_by_group,
                                 {"bootstrap": 1})
            if operation == "plan":
                assert request in source.pins.active and confirmed.is_set()
                if failure == "plan":
                    raise ValueError("destination rejected plan")
                authorized.set()
                return digest(payload["plan"])
            if operation == "complete":
                assert proof.is_set() and request not in source.pins.active
                return {}
            if operation == "abort":
                stops.append(payload["stopped"])
                return {}
            assert operation == "source_done"
            return {}
        @asynccontextmanager
        async def session(*args):
            yield SimpleNamespace(call=control, geometry=case.destination.geometry)
        source.peers = SimpleNamespace(use=session)
        async def independent(operation, payload):
            assert confirmed.is_set()
            if operation == "source_state":
                if failure == "not_ready":
                    return msgspec.msgpack.encode(dict(error="source not ready", uncertain=False))
                return msgspec.msgpack.encode({"bootstrap": 0})
            assert operation == "submit" and authorized.is_set() and request in source.pins.active
            value = msgspec.msgpack.decode(payload)
            sent.append(value)
            assert not proof.is_set()
            if failure == "known_submit":
                return msgspec.msgpack.encode(dict(error="capacity exceeded", uncertain=False))
            if failure == "lost_submit":
                raise TransferError("reply lost", uncertain=True)
            if failure == "cancel_submit":
                raise asyncio.CancelledError
            return msgspec.msgpack.encode({"job": entry.pending_digest})
        async def wait(job):
            assert request in source.pins.active and authorized.is_set()
            proof.set()
            return Completion(job, entry.pending.units)
        source.progress = SimpleNamespace(call=independent, wait=wait)
        try:
            if failure == "none":
                await source.prefill(entry, case.entry.continuation, case.nodes[1])
                assert proof.is_set() and entry.chunk == 1 and entry.end == 128
                timing = next(fields for name, fields in observed if name == "completion")
                assert timing["plan_ms"] >= timing["plan_construct_ms"] + timing["plan_digest_ms"] >= 0
                assert timing["plan_thread_cpu_ms"] >= timing["plan_construct_thread_cpu_ms"] + timing["plan_digest_thread_cpu_ms"] >= 0
            else:
                with pytest.raises(asyncio.CancelledError if failure == "cancel_submit" else Exception):
                    await source.prefill(entry, case.entry.continuation, case.nodes[1])
            unknown = failure in ("lost_submit", "cancel_submit")
            assert (request in source.pins.active) == unknown
            assert (request in source.requests) == unknown
            if failure != "none":
                assert stops == [not unknown] and not proof.is_set()
            if failure in ("not_ready", "plan"):
                assert not sent
        finally:
            finished.set()
            await asyncio.gather(*consumers)
    asyncio.run(run())

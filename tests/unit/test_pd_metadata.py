# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Boot-scoped metadata reuse must not weaken transfer validation or grow per request."""

from types import SimpleNamespace

import msgspec
import pytest

from pypto_serving.pd.planner import Geometry
from pypto_serving.pd.protocol import Binding, Completion, Copy, Plan, Topology, digest
from pypto_serving.pd.transfer import Region, TransferError
from pypto_serving.pd.worker import WorkerRuntime, WorkerSettings


@pytest.fixture
def runtime(monkeypatch):
    class Owner:
        def __init__(self, boot, rank, *args):
            self.boot, self.rank, self.writes, self.installs = boot, rank, 0, 0
        def register(self, worker, tensor, name):
            return Region(self.boot, self.rank, name, tensor.nbytes,
                          {"endpoint": "local", "address": 4096 * (self.rank + 1)})
        def call(self, operation, **payload):
            if operation == "install":
                self.installs += 1
            else:
                assert operation == "write"
                self.writes += 1

    monkeypatch.setattr("pypto_serving.pd.worker.Owner", Owner)
    geometry = (Geometry("kv", (0,), 4, 16, 64, "uint8"),)
    model = SimpleNamespace(ranks=2, tp=1, geometry=lambda: geometry, worker=lambda: None,
                            regions=lambda: {"kv": SimpleNamespace(shards=(SimpleNamespace(nbytes=4096),) * 2)})
    worker = WorkerRuntime(WorkerSettings("p-boot", "host", "mooncake", 2, 2), model)
    worker.prepare()
    try:
        yield worker
    finally:
        worker.pool.shutdown(wait=True)


def peer(runtime, boot="d-boot"):
    return dict(geometry=runtime.registry["geometry"], topology=Topology.for_tp(2, 1),
                regions=tuple(Region(boot, rank, "kv", 4096,
                    {"endpoint": f"destination-{rank}", "address": 16384 + 4096 * rank}) for rank in range(2)))


def plan(runtime, remote, request="request"):
    boot = remote["regions"][0].boot
    return Plan(Binding(request, ("p", "p-boot", runtime.registry_digest),
                        ("d", boot, digest(remote)), "input"),
                "continuation", 0, 0, 128, False, ((0, 0), (1, 1)),
                tuple(Copy("kv", rank, rank, 0, 0, 64) for rank in range(2)), ((0, "kv"), (1, "kv")))


def finish(runtime, job):
    for future in runtime.jobs[job][1]:
        future.result(timeout=2)
    result, = runtime.poll((job,))
    return result["completion"]


def test_registry_indices_are_reused_and_jobs_do_not_accumulate(runtime, monkeypatch):
    remote = peer(runtime)
    runtime.install(remote)
    local, destination = runtime.local_regions, runtime.peer_regions["d-boot"]
    def only_plan(value):
        assert isinstance(value, Plan), "immutable registry was hashed again during submission"
        return digest(value)
    monkeypatch.setattr("pypto_serving.pd.worker.digest", only_plan)
    for number in range(500):
        value = plan(runtime, remote, str(number))
        job = runtime.submit(value)
        finish(runtime, job).check(job, value.units)
        assert not runtime.jobs
        assert runtime.local_regions is local and runtime.peer_regions["d-boot"] is destination
        assert len(runtime.peers) == len(runtime.peer_fingerprints) == len(runtime.peer_regions) == 1
    assert runtime.prepare() is runtime.registry
    assert len(local) == len(destination) == 2


def test_peer_reinstall_and_restart_keep_metadata_bounded(runtime):
    first = peer(runtime)
    runtime.install(first)
    index = runtime.peer_regions["d-boot"]
    runtime.install(first)
    assert runtime.peer_regions["d-boot"] is index
    with pytest.raises(ValueError, match="changed within boot"):
        runtime.install({**first, "topology": Topology.for_tp(2, 2)})
    runtime.install(peer(runtime, "next-boot"))
    with pytest.raises(ValueError, match="capacity exhausted"):
        runtime.install(peer(runtime, "third-boot"))
    assert set(runtime.peers) == set(runtime.peer_fingerprints) == set(runtime.peer_regions) == {
        "d-boot", "next-boot"}
    assert all(owner.installs == 2 for owner in runtime.owners)


@pytest.mark.parametrize("bad", ["source_boot", "source_registry", "destination_registry", "bounds", "overlap"])
def test_cached_indices_still_reject_invalid_work_before_any_rank_writes(runtime, bad):
    remote = peer(runtime)
    runtime.install(remote)
    value = plan(runtime, remote)
    if bad in ("source_boot", "source_registry"):
        key = list(value.binding.prefill)
        key[1 if bad == "source_boot" else 2] = "stale"
        value = msgspec.structs.replace(value, binding=msgspec.structs.replace(value.binding, prefill=tuple(key)))
    elif bad == "destination_registry":
        value = msgspec.structs.replace(value, binding=msgspec.structs.replace(
            value.binding, decode=("d", "d-boot", "stale")))
    elif bad == "bounds":
        value = msgspec.structs.replace(value, copies=(value.copies[0],
            msgspec.structs.replace(value.copies[1], destination_offset=4096)))
    else:
        value = msgspec.structs.replace(value, copies=value.copies + (value.copies[0],))
    with pytest.raises((TransferError, ValueError)):
        runtime.submit(value)
    assert not runtime.jobs and not any(owner.writes for owner in runtime.owners)


def test_registry_wire_is_a_copy_not_a_mutable_view_of_cached_metadata(runtime):
    external = msgspec.msgpack.decode(runtime.handle("registry", b""))
    external["regions"][0]["handle"]["address"] = 0
    assert runtime.local_regions[0, "kv"].handle["address"] == 4096
    assert runtime.registry_digest == digest(runtime.registry)


def test_completion_uses_accepted_identity_and_still_validates_atomic_units(monkeypatch):
    units = ((0, "idx_k"), (0, "idx_scale"))
    monkeypatch.setattr("pypto_serving.pd.protocol.digest", lambda _: pytest.fail("unexpected rehash"))
    Completion("accepted-plan", units).check("accepted-plan", units)
    for value in (Completion("other-plan", units), Completion("accepted-plan", units[:1]),
                  Completion("accepted-plan", units + units[:1])):
        with pytest.raises(ValueError, match="completion"):
            value.check("accepted-plan", units)

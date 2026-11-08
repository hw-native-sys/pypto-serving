# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DSpark cache byte geometry and composition with optional existing owner services."""

from functools import partial
from math import prod

from pypto_serving.serving.memory.tiering.simpler import CopyService, descriptor_wire


COMPONENTS = {
    "ori": ("kv_cache",), "cmp_c128": ("hca_cmp_kv",), "cmp_c4": ("csa_cmp_kv",),
    "idx": ("idx_kv_cache", "idx_kv_scale"), "hca_state": ("hca_compress_state",),
    "csa_state": ("csa_compress_state",), "csa_inner_state": ("csa_inner_compress_state",),
}


class OwnerServices:
    def __init__(self, services):
        self.services = services

    def close(self):
        for service in reversed(self.services):
            service.close()


def owner_factory(endpoints, rank, previous, context):
    for index, endpoint in enumerate(endpoints):
        endpoint.parent.close()
        if index != rank:
            endpoint.child.close()
    services = []
    if previous is not None:
        services.append(previous(context))
    services.append(CopyService(context, endpoints[rank].child))
    return OwnerServices(services)


class CacheOffloadExtension:
    def __init__(self, runner, endpoints, metadata, previous=None):
        self.runner, self.endpoints, self.metadata = runner, endpoints, metadata
        self.previous = previous(runner=runner) if previous is not None else None
        self.tokens = []

    def __getattr__(self, name):
        return getattr(self.previous, name)

    def wrap_prepare(self, build):
        return self.previous.wrap_prepare(build) if self.previous is not None else build

    def wrap_dispatch(self, dispatch):
        return self.previous.wrap_dispatch(dispatch) if self.previous is not None else dispatch

    def wrap_capture(self, capture):
        return self.previous.wrap_capture(capture) if self.previous is not None else capture

    def worker_options(self):
        options = self.previous.worker_options() if self.previous is not None else {}
        previous = options.get("chip_service_factories", {})
        options["chip_service_factories"] = {
            rank: partial(owner_factory, self.endpoints, rank, previous.get(rank))
            for rank in range(len(self.endpoints))}
        return options

    def worker_ready(self):
        if self.previous is not None:
            self.previous.worker_ready()

    def model_ready(self):
        worker = self.runner._shared_l3_worker()
        tensors = self.runner._materialize_decode_device_cache()
        plan = self.runner._compiled.layer_plan
        layers = {
            "ori": len(plan),
            "cmp_c128": sum(layer.compress_ratio == 128 for layer in plan),
            "cmp_c4": sum(layer.compress_ratio == 4 for layer in plan),
        }
        layers.update(idx=layers["cmp_c4"], hca_state=layers["cmp_c128"],
                      csa_state=layers["cmp_c4"], csa_inner_state=layers["cmp_c4"])
        groups = {}
        for group, names in COMPONENTS.items():
            regions = []
            for name in names:
                tensor = tensors[name]
                shard = tensor.shards[0]
                shape = tuple(shard.shape)
                count = layers[group]
                if not count or shape[0] % count:
                    raise ValueError("invalid DSpark offload region geometry")
                blocks = shape[0] // count
                page_bytes = shard.nbytes // (count * blocks)
                if prod(shape) <= 0 or page_bytes * count * blocks != shard.nbytes:
                    raise ValueError("invalid DSpark offload extent")
                regions.append((name, count, blocks, page_bytes))
                for rank, endpoint in enumerate(self.endpoints):
                    token, descriptor = worker.retain_service_tensor(tensor.shards[rank], worker_id=rank)
                    self.tokens.append(token)
                    endpoint.call("register", (name, token, descriptor_wire(descriptor)))
            groups[group] = tuple(regions)
        self.metadata.put(dict(groups=groups, ranks=len(self.endpoints),
                               tp=self.runner._compiled.layout.tp_size))

    def close(self):
        for endpoint in self.endpoints:
            if not endpoint.stopped.value:
                # Startup failures can leave registered regions but no DDR jobs.
                # The owner still rejects stop if any allocation is in use.
                endpoint.call("stop")
        worker = self.runner._shared_l3_worker()
        for token in self.tokens:
            worker.release_service_tensor(token)
        self.tokens.clear()
        if self.previous is not None:
            self.previous.close()

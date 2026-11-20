# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DSpark cache geometry and native Buffer leases for independent copy processes."""

from dataclasses import asdict
import multiprocessing
import logging
from math import prod

from pypto_serving.serving.memory.tiering.simpler import serve_copy

logger = logging.getLogger(__name__)


COMPONENTS = {
    "ori": ("kv_cache",), "cmp_c128": ("hca_cmp_kv",), "cmp_c4": ("csa_cmp_kv",),
    "idx": ("idx_kv_cache", "idx_kv_scale"), "hca_state": ("hca_compress_state",),
    "csa_state": ("csa_compress_state",), "csa_inner_state": ("csa_inner_compress_state",),
}


class CacheOffloadExtension:
    def __init__(self, runner, endpoints, metadata, pool, previous=None):
        self.runner, self.endpoints, self.metadata = runner, endpoints, metadata
        self.pool = pool
        self.previous = previous(runner=runner) if previous is not None else None
        self.exports = []
        self.processes = []

    def __getattr__(self, name):
        return getattr(self.previous, name)

    def wrap_prepare(self, build):
        return self.previous.wrap_prepare(build) if self.previous is not None else build

    def wrap_dispatch(self, dispatch):
        return self.previous.wrap_dispatch(dispatch) if self.previous is not None else dispatch

    def wrap_capture(self, capture):
        return self.previous.wrap_capture(capture) if self.previous is not None else capture

    def allocation_options(self, scope, name=""):
        options = self.previous.allocation_options(scope, name) if self.previous is not None else {}
        return {**options, "exportable": True} if scope == "cache" else options

    def worker_ready(self):
        if self.previous is not None:
            self.previous.worker_ready()
        context = multiprocessing.get_context("spawn")
        devices = self.runner._shared_l3_worker().dc.device_ids
        for rank, (endpoint, device) in enumerate(zip(self.endpoints, devices, strict=True)):
            process = context.Process(target=serve_copy, name=f"cache-copy-{rank}",
                                      args=(device, endpoint.child, self.pool, endpoint.stopped))
            process.start()
            self.processes.append(process)
            endpoint.child.close()
            endpoint.call("ready")

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
                    from simpler.buffer_export import BufferExportError
                    try:
                        exported = worker.export_buffer(tensor.shards[rank], worker_id=rank)
                    except BufferExportError as error:
                        self.exports.append(error.exported)
                        raise
                    self.exports.append(exported)
                    endpoint.call("register", (name, asdict(exported)))
            groups[group] = tuple(regions)
        self.metadata.put(dict(groups=groups, ranks=len(self.endpoints),
                               tp=self.runner._compiled.layout.tp_size))

    def close(self):
        for endpoint in self.endpoints[:len(self.processes)]:
            if not endpoint.stopped.value:
                endpoint.call("stop")
        for process in self.processes:
            process.join(10)
            if process.is_alive() or process.exitcode != 0:
                raise RuntimeError("cache copy process did not retire normally")
        worker = self.runner._shared_l3_worker()
        while self.exports:
            worker.release_export(self.exports[0])
            self.exports.pop(0)
        if self.previous is not None:
            self.previous.close()

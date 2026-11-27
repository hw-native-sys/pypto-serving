# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Packed routed-expert weights shared by the prefill and decode graphs."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import ExitStack, contextmanager

import torch


ROUTED_FIELDS = tuple(f"routed_{projection}{suffix}"
                      for projection in ("w1", "w2", "w3") for suffix in ("", "_scale"))
EXPERT_METADATA = ("expert_state", "expert_descriptors")


def expert_slot_capacity(budget_bytes: int, expert_bytes: int, layers: int, local_experts: int) -> int:
    """Return the number of complete rank-local packed experts that fit the bank."""
    count = min(budget_bytes // expert_bytes, layers * local_experts)
    if count < 1:
        raise ValueError(f"expert bank needs at least {expert_bytes} bytes per rank for one slot")
    return count


class ExpertBacking:
    """One CPU view per field and rank, shaped [layers, local experts, *packed slot]."""

    def __init__(self, by_rank: Sequence[Mapping[str, torch.Tensor]], budget_bytes: int):
        if not by_rank or not by_rank[0]:
            raise ValueError("expert backing requires rank-local packed fields")
        prototype = by_rank[0]
        first = next(iter(prototype.values()))
        if first.ndim < 3:
            raise ValueError("expected packed fields shaped [layer, local expert, ...]")
        self.layers, self.local_experts = first.shape[:2]
        if self.layers == 0 or self.local_experts == 0:
            raise ValueError("expert backing requires nonempty layer and expert axes")
        self.by_rank = by_rank
        self.strides = {}
        for rank_fields in by_rank:
            if tuple(rank_fields) != tuple(prototype):
                raise ValueError("ranks must agree on expert fields and their descriptor order")
        for field, tensor in prototype.items():
            for rank_fields in by_rank:
                source = rank_fields[field]
                if (source.device.type != "cpu" or source.ndim < 3 or source.shape != tensor.shape
                        or source.shape[:2] != first.shape[:2] or source.dtype != tensor.dtype
                        or not source[0].is_contiguous()):
                    raise ValueError(f"incompatible packed layout for expert field {field}")
            stride = tensor[0, 0].nbytes
            if not stride or stride % 32:
                raise ValueError(f"expert field {field} must contain whole 32-byte AIV transfers")
            self.strides[field] = stride
        self.expert_bytes = sum(self.strides.values())
        self.capacity = expert_slot_capacity(budget_bytes, self.expert_bytes, self.layers, self.local_experts)

    def initialize(self, rank: int, window: memoryview) -> None:
        """Copy the checkpoint's physical rows once into rank-local Host DDR."""
        offset = 0
        dest = torch.frombuffer(window, dtype=torch.uint8)
        for name, tensor in self.by_rank[rank].items():
            size = self.local_experts * self.strides[name]
            for rows in tensor.unbind():
                dest.narrow(0, offset, size).copy_(rows.view(torch.uint8).view(-1))
                offset += size

    def descriptors(self, bases: Sequence[int]) -> torch.Tensor:
        """[mapped expert zero, source stride, copy bytes, destination stride] per field."""
        catalog = torch.empty((len(bases), self.layers, len(self.strides), 4), dtype=torch.int64)
        for rank, base in enumerate(bases):
            for i, stride in enumerate(self.strides.values()):
                layer_bytes = self.local_experts * stride
                catalog[rank, :, i, 0] = torch.arange(self.layers, dtype=torch.int64) * layer_bytes + base
                catalog[rank, :, i, 1:] = stride
                base += self.layers * layer_bytes
        return catalog

    def bind(self, worker) -> ExpertPool:
        return ExpertPool(worker, self)


class ExpertPool:
    """Device allocations owned by one worker, used by all of its MoE graphs."""

    def __init__(self, worker, backing: ExpertBacking):
        from pypto_serving.model.common.runner.buffer_set import alloc_device_buffer

        self.worker = worker
        self.expert_bytes = backing.expert_bytes
        self.capacity = backing.capacity
        self.banks = {}
        world, capacity = len(backing.by_rank), backing.capacity
        self._resources = ExitStack()
        try:
            bases = self._resources.enter_context(_mapped_expert_backing(worker, backing))
            self.descriptors = worker.alloc_stacked_tensor(backing.descriptors(bases))
            self._resources.callback(worker.free_stacked_tensor, self.descriptors)
            from models.common.expert_cache import initial_expert_cache

            initial = initial_expert_cache(capacity, backing.layers, backing.local_experts)
            initial = initial.unsqueeze(0).repeat(world, 1, 1)
            self.state = worker.alloc_stacked_tensor(initial)
            self._resources.callback(worker.free_stacked_tensor, self.state)
            for name, source in backing.by_rank[0].items():
                bank = alloc_device_buffer(worker, (world, capacity, *source.shape[2:]), source.dtype, stacked=True)
                self.banks[name] = bank
                self._resources.callback(worker.free_stacked_tensor, bank)
        except BaseException:
            self.close()
            raise

    def arguments(self):
        return {**self.banks, "expert_state": self.state, "expert_descriptors": self.descriptors}

    def stats(self) -> dict[str, int]:
        """Read device counters after outstanding graphs have completed."""
        snapshot = torch.empty(self.state.full_shape, dtype=torch.int32)
        self.worker.copy_stacked_from(self.state, snapshot)
        # Device tallies are unsigned counters carried in int32 tensors.
        hits, misses = (snapshot[:, 1:self.capacity + 1, 4:6].long() & 0xFFFFFFFF).sum(dim=(0, 1)).tolist()
        return {"hits": hits, "misses": misses, "mapped_bytes": misses * self.expert_bytes}

    def close(self):
        self._resources.close()


@contextmanager
def _mapped_expert_backing(worker, backing):
    """Retain one Host window per rank across the prefill and decode programs."""
    from simpler.buffer import AddressSpace
    from simpler.task_interface import CommBufferSpec

    ranks = tuple(range(len(backing.by_rank)))
    nbytes = backing.expert_bytes * backing.layers * backing.local_experts
    aligned = (nbytes + 63) // 64 * 64
    bases = []
    handle = None

    def initialize(domain):
        nonlocal backing
        for rank in ranks:
            buffer = domain[rank].buffers["data"]
            with buffer.shm.buf[:nbytes] as view:
                backing.initialize(rank, view)
            bases.append(int(buffer.base))

    def allocate(orch, _args, _config):
        nonlocal handle
        handle = orch.allocate_domain(
            name="routed_experts", members=ranks, window_size=aligned,
            buffers=[CommBufferSpec("data", "uint8", nbytes, aligned)],
            memory_source=AddressSpace.HOST, retain_after_run=True, initializer=initialize,
        )

    try:
        # Keep the native-worker bridge here. Graphs receive addresses through
        # descriptors, so these windows need no PyPTO DeviceTensor registration.
        worker._w.run(allocate)
        del backing
        yield tuple(bases)
    finally:
        if handle is not None:
            handle.release()

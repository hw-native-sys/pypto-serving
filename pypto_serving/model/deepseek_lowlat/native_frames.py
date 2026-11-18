# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded native output leases with whole-parent publication before capture."""

from dataclasses import dataclass

import torch
from pypto.runtime import StackedDeviceTensor

from .profiling import operation
from .bindings import DTYPES
from .speculation import validate_packets


HOST_SHAPES = {"s_control": (8, 2, 16), "s_emitted": (8, 2, 6)}


def allocate_native_hosts(frame_count=1):
    if type(frame_count) is not int or frame_count not in (1, 2):
        raise ValueError("one or two native frames required")
    return tuple({name: torch.full(shape, -777, dtype=torch.int64).share_memory_()
                  for name, shape in HOST_SHAPES.items()} for _ in range(frame_count))


@dataclass
class NativeFrame:
    arguments: tuple
    hosts: dict
    handle: object = None
    generation: int = 0
    cycles: int = 0
    leased: bool = False
    published: bool = False
    packet: object = None
    packet_state: object = None
    consumed: bool = False


class NativeFrames:
    def __init__(self, worker, programs, metadata, common, outputs, hosts, config):
        if (not programs or set(programs) not in ({1}, {1, 2}) or set(programs) != set(metadata)
                or len(outputs) != len(hosts) or len(hosts) not in (1, 2)):
            raise ValueError("actual native programs and matching independent frames required")
        rows = metadata[1]["params"]
        if len(rows) not in (153, 155) or any(rows != m["params"] for m in metadata.values()):
            raise ValueError("native one/two ABI must be identical public153 or155")
        names = [row["name"].split("__ssa", 1)[0] for row in rows]
        if len(set(names)) != len(names) or not HOST_SHAPES.keys() <= set(names):
            raise ValueError("native packet parameters are missing or duplicated")
        self.worker, self.programs, self.config = worker, programs, config
        self.poisoned = False
        self.frames = []
        out_owners, input_owners = set(), set()
        out_ranges, input_ranges = [], []
        for index in range(len(hosts)):
            arguments = []
            for name, row in zip(names, rows, strict=True):
                if name in HOST_SHAPES:
                    value = hosts[index][name]
                    if (row["direction"] != "Out" or tuple(row["shape"]) != HOST_SHAPES[name]
                            or row["dtype"] != "int64" or not isinstance(value, torch.Tensor)
                            or value.device.type != "cpu" or value.dtype != torch.int64
                            or tuple(value.shape) != HOST_SHAPES[name]
                            or not value.is_shared() or not value.is_contiguous()):
                        raise ValueError("native HOST packets require exact prefork shared roots")
                    start, end = value.data_ptr(), value.data_ptr()+value.numel()*value.element_size()
                    if not any(lo <= start and end <= hi for (lo, hi), _ in worker._inherited_host_spans):
                        raise ValueError("native HOST root was not inherited before Worker preparation")
                else:
                    value = outputs[index][name] if row["direction"] == "Out" else common[name]
                    shape = row["shape"]
                    if (not isinstance(value, StackedDeviceTensor) or shape is None
                            or len(shape) != len(value.shape)
                            or any(want != -1 and want != got for want, got in zip(shape, value.shape))
                            or value.dtype != DTYPES[row["dtype"]]
                            or value.worker_ids != tuple(range(shape[0]))):
                        raise ValueError("native device argument differs from generated metadata: " + name)
                    for rank, shard in zip(value.worker_ids, value.shards, strict=True):
                        owner = shard.buffer
                        if owner is None:
                            raise ValueError("native arguments must retain owning Buffers")
                        if row["direction"] == "Out":
                            if shard.byte_offset or id(owner) in out_owners:
                                raise ValueError("native output frames require independent owning Buffers")
                            out_owners.add(id(owner))
                            out_ranges.append((rank, shard.data_ptr, shard.data_ptr+shard.nbytes))
                        else:
                            input_owners.add(id(owner))
                            input_ranges.append((rank, shard.data_ptr, shard.data_ptr+shard.nbytes))
                arguments.append(value)
            for program in programs.values():
                infos = worker._states[program]["param_infos"]
                if len(infos) != len(arguments):
                    raise ValueError("registered native arity differs from actual metadata")
                for info, value in zip(infos, arguments, strict=True):
                    worker._validate_prepared_dispatch_arg(info, value)
            self.frames.append(NativeFrame(tuple(arguments), hosts[index]))
        if out_owners & input_owners:
            raise ValueError("native Out owner aliases shared persistent input")
        for index, (rank, start, end) in enumerate(out_ranges):
            if any(rank == r and start < hi and lo < end
                   for r, lo, hi in [*out_ranges[index+1:], *input_ranges]):
                raise ValueError("native device output ranges overlap another frame or persistent input")
        host_spans = [(value.data_ptr(), value.data_ptr()+value.numel()*value.element_size())
                      for frame in self.frames for value in frame.hosts.values()]
        if any(a < d and c < b for i, (a, b) in enumerate(host_spans) for c, d in host_spans[i+1:]):
            raise ValueError("native HOST output frames overlap")

    def _frame(self, index):
        if type(index) is not int or not 0 <= index < len(self.frames):
            raise ValueError("invalid native frame index")
        return self.frames[index]

    def reserve(self, index, *, generation, cycles):
        frame = self._frame(index)
        if (self.poisoned or frame.leased or type(generation) is not int or generation <= 0
                or type(cycles) is not int or cycles not in self.programs):
            raise RuntimeError("native frame is unavailable or submission context is invalid")
        frame.leased, frame.generation, frame.cycles = True, generation, cycles

    @operation("lowlat.native.submit", category="kernel")
    def submit_reserved(self, index, *, generation, cycles):
        frame = self._frame(index)
        if (self.poisoned or not frame.leased or frame.handle is not None
                or frame.generation != generation or frame.cycles != cycles):
            raise RuntimeError("native frame has no matching submission reservation")
        try:
            frame.handle = self.worker.submit(self.programs[cycles], *frame.arguments, config=self.config)
        except Exception:
            self.poisoned = True
            raise
        return frame.handle

    def submit(self, index, *, generation, cycles):
        self.reserve(index, generation=generation, cycles=cycles)
        return self.submit_reserved(index, generation=generation, cycles=cycles)

    @operation("lowlat.native.wait", category="lowlat")
    def complete(self, index):
        frame = self._frame(index)
        if not frame.leased or frame.handle is None or self.poisoned:
            raise RuntimeError("native frame has no completable parent")
        if not frame.published:
            try:
                frame.handle.result()
            except Exception:
                self.poisoned = True
                raise
            frame.published = True

    @operation("lowlat.native.capture", category="lowlat")
    def capture(self, index, *, generation, state):
        frame = self._frame(index)
        if (self.poisoned or type(generation) is not int or generation != frame.generation
                or not frame.published):
            raise RuntimeError("capture requires this generation's successful whole-parent publication")
        if frame.packet is not None and state != frame.packet_state:
            raise RuntimeError("captured native packet belongs to a different predecessor state")
        if frame.packet is None:
            try:
                frame.packet = validate_packets(frame.hosts["s_control"], frame.hosts["s_emitted"],
                                                state=state, cycles=frame.cycles)
                frame.packet_state = state
            except Exception:
                self.poisoned = True
                raise
        return frame.packet

    def consume(self, index):
        frame = self._frame(index)
        if self.poisoned or frame.packet is None:
            raise RuntimeError("validate before delivering or discarding native output")
        frame.consumed = True

    @operation("lowlat.native.release", category="lowlat")
    def release(self, index):
        frame = self._frame(index)
        if self.poisoned or not frame.leased or not frame.published or not frame.consumed:
            raise RuntimeError("native frame remains owned through capture and delivery/discard")
        frame.handle = frame.packet = frame.packet_state = None
        frame.leased = frame.published = frame.consumed = False
        frame.cycles = frame.generation = 0

    def idle(self):
        return not self.poisoned and all(not frame.leased for frame in self.frames)

# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate generated ABIs and manage owning tensors separately from views."""

from math import ceil, prod

import torch

from pypto.runtime import StackedDeviceTensor


DTYPES = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
}
HOST_PARAMETERS = frozenset({"num_tokens_per_owner"})


def frame_shape(name, shape, token_capacity):
    """Resolve request, checkpoint, CP-local and full-token axes separately."""
    result = list(shape)
    for axis, size in enumerate(result):
        if size != -1:
            continue
        if name == "query_start_loc":
            result[axis] = 2
        elif name == "state_checkpoint_ends":
            result[axis] = 1 if axis == 1 else ceil(token_capacity / 512)
        elif name.endswith("block_table"):
            result[axis] = 1
        elif name in ("position_ids_local", "local_request_ids", "dspark_target_hidden"):
            result[axis] = token_capacity // 8
        elif name in ("x_hc", "attn_stage", "x_out", "position_ids_full", "input_ids") or name.endswith(
            ("_freqs_cos", "_freqs_sin", "_slot_mapping_full")
        ):
            result[axis] = token_capacity
        else:
            raise ValueError(f"unknown dynamic lowlat frame axis: {name}/{axis}")
    return tuple(result)


def parameter_rows(metadata):
    rows = metadata.get("params", [])
    if not rows:
        raise ValueError("generated lowlat parameter metadata is required")
    result = {}
    for row in rows:
        name = row["name"].split("__ssa", 1)[0]
        if name in result or not {"dtype", "direction", "shape"} <= row.keys():
            raise ValueError(f"invalid or duplicate ABI parameter: {name}")
        if row["shape"] is not None and row.get("layout", "ND") != "ND":
            raise ValueError(f"lowlat M0 requires ND parameters: {name}")
        result[name] = row
    return result


def ordered_arguments(metadata, bindings):
    arguments = []
    for name, row in parameter_rows(metadata).items():
        if name not in bindings:
            raise ValueError(f"missing lowlat binding: {name}")
        value = bindings[name]
        expected = row["shape"]
        if expected is None:
            if type(value) is not int or row["dtype"] != "int32" or not -(2**31) <= value < 2**31:
                raise ValueError(f"{name} must be a scalar INT32")
        else:
            if name in HOST_PARAMETERS:
                if (
                    not isinstance(value, torch.Tensor)
                    or value.device.type != "cpu"
                    or not value.is_shared()
                    or not value.is_contiguous()
                    or list(value.shape) != expected
                    or value.dtype != DTYPES[row["dtype"]]
                ):
                    raise ValueError(f"{name} requires the pre-fork shared host control slot")
                arguments.append(value)
                continue
            if not isinstance(value, StackedDeviceTensor):
                raise TypeError(f"{name} must be a Worker-owned rank-stacked tensor")
            if (
                len(value.shape) != len(expected)
                or any(want >= 0 and got != want for got, want in zip(value.shape, expected))
                or value.dtype != DTYPES[row["dtype"]]
                or value.worker_ids != tuple(range(8))
            ):
                raise ValueError(
                    f"lowlat ABI mismatch for {name}: {value.shape}/{value.dtype}, expected {row}"
                )
        arguments.append(value)
    return tuple(arguments)


def borrow_view(worker, owner, shape):
    if shape[0] != owner.shape[0] or prod(shape) * torch.empty((), dtype=owner.dtype).element_size() > sum(
        s.nbytes for s in owner.shards
    ):
        raise ValueError("borrowed shape exceeds its owner")
    shards = [
        worker.view_tensor(shard, tuple(shape[1:]), worker_id=rank) for rank, shard in enumerate(owner.shards)
    ]
    return StackedDeviceTensor(shards, tuple(shape), owner.worker_ids)


class TensorArena:
    """Explicit owners; borrowed views are never freed independently."""

    def __init__(self, worker):
        self.worker = worker
        self.owners = []

    def allocate(self, shape, dtype):
        if shape[0] not in (4, 8) or any(size <= 0 for size in shape):
            raise ValueError(f"invalid rank-stacked allocation: {shape}")
        shards = []
        try:
            for rank in range(shape[0]):
                shards.append(self.worker.alloc_tensor(tuple(shape[1:]), dtype, worker_id=rank))
        except Exception:
            for rank, shard in enumerate(shards):
                self.worker.free_tensor(shard, worker_id=rank)
            raise
        owner = StackedDeviceTensor(shards, tuple(shape), tuple(range(shape[0])))
        self.owners.append(owner)
        return owner

    def upload(self, host):
        owner = self.allocate(tuple(host.shape), host.dtype)
        self.copy(owner, host)
        return owner

    def copy(self, owner, host):
        if tuple(host.shape) != tuple(owner.shape) or host.dtype != owner.dtype:
            raise ValueError("upload shape/dtype differs from its destination")
        for rank, shard in enumerate(owner.shards):
            source = host[rank].contiguous()
            self.worker.copy_to(
                shard.data_ptr, source.data_ptr(), source.numel() * source.element_size(), worker_id=rank
            )

    def close(self):
        first = None
        for owner in reversed(self.owners):
            try:
                self.worker.free_stacked_tensor(owner)
            except Exception as exc:
                first = first or exc
        self.owners.clear()
        if first is not None:
            raise first

# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compiled GLM entry points and ordered arguments for persistent L3 programs."""

from dataclasses import dataclass

from pypto_serving.model.common.runner.buffer_set import Placement, Slot
from pypto_serving.model.common.runner.task_args import TaskArgs


@dataclass
class Glm53CompiledKernels:
    api: object
    programs: dict
    checkpoint: object
    weights: object


def input_task_args(inputs):
    """Allocate all mutable host metadata before the L3 worker starts."""
    args = TaskArgs(stacked=True)
    for name, value in inputs.items():
        shape = tuple(value.shape)
        args.add_slot(Slot(name, Placement.HOST_SHARED, value.dtype, lambda _, shape=shape: shape))
    return args


def forward_task_args(api, phase, tensors):
    """Bind one full forward; its runtime epoch is appended at dispatch."""
    args = TaskArgs(stacked=True)
    for name in api.forward_parameter_names(phase):
        args.add_arg(name, tensors[name])
    return args


def forward_input_task_args(api, phase, inputs):
    """Own per-phase rank-stacked host inputs before the persistent worker forks."""
    args = TaskArgs(stacked=True)
    for name, (shape, dtype) in api.forward_tensor_layouts(phase).items():
        if name not in inputs or name in api.CONSTANT_STEP_INPUTS or name in api._PARAM_TO_STACK:
            continue
        args.add_slot(Slot(name, Placement.HOST_SHARED, dtype,
                           lambda _, shape=shape: (api.RANKS, *shape)))
    return args


def resident_input_names(api, inputs, *, enable_mtp=False):
    """Select inputs also consumed by synchronous cache or MTP auxiliary entries."""
    if enable_mtp:
        return set(inputs)
    required = set(api.CONSTANT_STEP_INPUTS) | set(api._PARAM_TO_STACK)
    return set(inputs) & required

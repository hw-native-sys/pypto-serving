# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""V4.1 worker initialization with a host-backed routed-expert bank."""

import importlib
import sys

from pypto_serving.model.common.moe_offload import ROUTED_FIELDS
from .expert_offload import mapped_weight_arguments, routed_expert_backing, v41_expert_slots
from .npu_runner import (
    V41ForwardPrograms, V41ResidentResources, compile_prefill_fwd, load_segment_modules,
)
from .routed_weight_staging import stage_routed_weights
from .weight_loader import ResidentFullDecodeWeights


class V41OffloadResources(V41ResidentResources):
    def __init__(self, *args, expert_slots, expert_cache_bytes, **kwargs):
        self.expert_pool = None
        self._expert_slots = expert_slots
        self._expert_cache_bytes = expert_cache_bytes
        super().__init__(*args, **kwargs)

    def expert_cache_stats(self):
        return self.expert_pool.stats() if self.expert_pool is not None else {}

    def _initialize_weights(self, programs, run_config, worker_factory, inherited,
                            weight_bank, weight_param_names):
        with stage_routed_weights(weight_bank) as restore:
            upload = tuple(tensor for name, tensor in weight_bank.tensors.items() if name not in ROUTED_FIELDS)
            self.worker = worker_factory(programs, run_config, upload + inherited)
            restore()
            backing = routed_expert_backing(weight_bank, self._expert_cache_bytes)
            if backing.capacity != self._expert_slots:
                raise ValueError("packed V4.1 experts disagree with the compiled slot geometry")
            self.expert_pool = self._own(backing.bind(self.worker))
            borrowed = mapped_weight_arguments(self.expert_pool)
        self.weights = OffloadDecodeWeights(
            self.worker, weight_bank, weight_param_names, borrowed=borrowed)


class OffloadDecodeWeights(ResidentFullDecodeWeights):
    """Resident attention weights combined with borrowed expert-pool bindings."""

    def __init__(self, worker, bank, parameter_names, *, borrowed):
        super().__init__(worker, bank, parameter_names)
        self._borrowed = borrowed
        self.layout_names = (*self.layout_names, *borrowed)

    def full(self, name):
        if name in self._borrowed:
            return self._borrowed[name]
        return super().full(name)


class V41OffloadExecution:
    supported_dp_sizes = (1, 2, 4, 8)

    def default_max_batch_size(self, dp_size):
        return min(16, 12 * dp_size)

    def prepare(self, root, topology, runtime, options, config, layers):
        if runtime.num_speculative_tokens:
            raise ValueError("V4.1 mapped experts currently support non-speculative generation")
        self.expert_cache_bytes = options.expert_cache_bytes
        self.expert_slots = v41_expert_slots(
            self.expert_cache_bytes, config, world=topology.world, layers=layers)
        load_segment_modules(root, topology)
        old_argv = sys.argv[:]
        try:
            sys.argv = [old_argv[0], "--expert-slots", str(self.expert_slots)]
            context = importlib.import_module("expert_context")
        finally:
            sys.argv = old_argv
        if context.EXPERT_SLOTS != self.expert_slots:
            raise ValueError("V4.1 expert slot geometry changed; initialize a fresh model worker")

    def compile(self, compiler, root, topology, prefill_topology, *, num_speculative_tokens=0):
        config = importlib.import_module("config")
        previous_argv = sys.argv[:]
        try:
            sys.argv = [*previous_argv, "--experts-per-rank", str(config.FLASH.n_routed_experts // topology.world)]
            entry = importlib.import_module("models.deepseek_v4_1_flash.decode_generate").l3_decode_generate
        finally:
            sys.argv = previous_argv
        decode = compiler.compile("v41_decode_generate", entry)
        prefill, prefill_names = compile_prefill_fwd(
            compiler, root, prefill_topology, entry_module="offload_prefill_fwd")
        abi = importlib.import_module("models.deepseek_v4_1_flash.decode_fwd")
        return V41ForwardPrograms(
            decode=decode, decode_names=tuple(entry.param_names), decode_counts_are_tensor=True,
            prefill=prefill, prefill_names=prefill_names,
            decode_head=None, decode_head_names=(), abi=abi,
        )

    def create_resources(self, programs, run_config, **kwargs):
        return V41OffloadResources(
            programs, run_config, expert_slots=self.expert_slots,
            expert_cache_bytes=self.expert_cache_bytes, **kwargs)

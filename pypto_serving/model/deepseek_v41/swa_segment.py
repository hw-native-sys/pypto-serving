# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded device-resident SWA Attention -> FP4 MoE layer composition.

This is an execution segment, not the complete V41 CLI backend. The owner
allocates/uploads weights, caches and scratch before worker creation, retains
all buffers until completion, and closes the worker on a failed dispatch.
"""
from dataclasses import dataclass
import ctypes
import importlib
from pathlib import Path
import sys

from .composite import LayerState

ATTENTION_ARGS = (
    "x_hc incoming_pre_mix hc_attn_fn hc_attn_scale hc_attn_base attn_norm_weight "
    "wq_a wq_a_scale q_norm_weight wq_b wq_b_scale wkv wkv_scale kv_norm_weight "
    "attn_sink wo_a wo_b wo_b_scale rope_cos rope_sin window_slots window_indices "
    "window_cache window_cache_scale output next_pre_mix hidden attn_out num_tokens attention_epoch"
).split()
MOE_ARGS = (
    "x_hc pre_mix hc_ffn_fn hc_ffn_scale hc_ffn_base norm_weight gate_weight correction_bias "
    "routed_w1 routed_w1_scale routed_w2 routed_w2_scale routed_w3 routed_w3_scale mxfp4_pair_lut "
    "shared_w1 shared_w1_scale shared_w2 shared_w2_scale shared_w3 shared_w3_scale "
    "next_pre_mix x_mixed x_next num_tokens moe_epoch"
).split()


@dataclass(frozen=True)
class SegmentTopology:
    tp: int = 4
    dp: int = 2
    local_capacity: int = 16

    @classmethod
    def for_decode(cls, *, tp: int = 4, dp: int = 2):
        """Match lib's padded local MoE extent for a full decode batch."""
        if type(tp) is not int or tp not in (1, 2, 4, 8):
            raise ValueError("unsupported decode TP size")
        decode_tokens = 32 * (5 + 1)
        row_tile = 16
        local_rows = (decode_tokens + tp - 1) // tp
        return cls(tp=tp, dp=dp, local_capacity=(local_rows + row_tile - 1) // row_tile * row_tile)

    def __post_init__(self):
        if self.tp not in (1, 2, 4, 8) or type(self.tp) is not int:
            raise ValueError("unsupported TP size")
        if type(self.dp) is not int or self.dp <= 0 or self.world not in (2, 4, 8):
            raise ValueError("TP * DP must be a supported EP size")
        if type(self.local_capacity) is not int or self.local_capacity <= 0:
            raise ValueError("local capacity must be positive")

    @property
    def world(self):
        return self.tp * self.dp

    @property
    def capacity(self):
        return self.tp * self.local_capacity

    def counts(self, group_counts):
        """Contiguous SP slabs; padded/empty ranks still participate in collectives."""
        if len(group_counts) != self.dp or any(
            type(n) is not int or not 0 <= n <= self.capacity for n in group_counts
        ):
            raise ValueError("one bounded active-token count is required per DP group")
        attention = tuple(n for n in group_counts for _ in range(self.tp))
        local = tuple(max(0, min(self.local_capacity, n - r * self.local_capacity))
                      for n in group_counts for r in range(self.tp))
        return attention, local


def load_segment_modules(lib_root, topology):
    """Use V4's explicit lib path and import-time topology, in a model worker.

    Never reload an already imported model with another topology/revision.
    Lib currently captures dimensions from argv when modules are imported.
    """
    root = Path(lib_root).resolve()
    package = "models.deepseek_v4_1_flash"
    existing = sys.modules.get(package + ".config")
    expected = root / "models/deepseek_v4_1_flash/config.py"
    if existing is not None and Path(existing.__file__).resolve() != expected:
        raise RuntimeError("V41 lib from another checkout is already imported; use a fresh worker")
    old_argv, old_path = sys.argv[:], sys.path[:]
    try:
        sys.path.insert(0, str(root))
        sys.argv = [old_argv[0], "--tp", str(topology.tp), "--ep", str(topology.world)]
        config = importlib.import_module(package + ".config")
        if Path(config.__file__).resolve() != expected:
            raise RuntimeError("import resolved to a different lib checkout")
        if (config.TP_SIZE, config.EP_SIZE, config.MOE_TOKENS) != (
            topology.tp, topology.world, topology.local_capacity
        ):
            raise ValueError("lib topology/MOE_TOKENS disagrees with the segment capacity")
        attention = importlib.import_module(package + ".prefill_swa")
        moe = importlib.import_module(package + ".moe")
        if moe.SKIP_SHARED_TEST or moe.SKIP_TRANSPORT_TEST:
            raise ValueError("cannot run serving with lib test-only MoE bypasses")
        return attention, moe
    finally:
        sys.argv, sys.path = old_argv, old_path


def compile_segment(compiler, lib_root, topology):
    """Compile the two existing lib composites with the shared V4 compiler."""
    import pypto.language as pl

    attention, moe = load_segment_modules(lib_root, topology)
    swa = compiler.compile(
        "v41_swa_segment", attention.make_hc_program(topology.capacity, topology.world, epochs=1),
        attention_epoch=pl.RUNTIME,
    )
    ffn = compiler.compile("v41_moe_segment", moe.l3_moe, moe_epoch=pl.RUNTIME)
    return swa, ffn


def make_segment_worker(programs, run_config, inherited_host_tensors=()):
    """Retain both programs' communication windows as in the V4 runner.

    Attention waits for (epoch - 1) * 2 before publishing the next phase.
    The runtime's default reset-on-reuse policy would erase that completion
    signal and stall the second layer. Each compiled program owns its windows.
    """
    from pypto.runtime import DistributedWorker

    return DistributedWorker(
        [program.compiled for program in programs], config=run_config,
        persistent=True, reset_persistent_windows=False,
        inherited_host_tensors=inherited_host_tensors,
    )


class SwaSegment:
    """Synchronous half-layer dispatcher on one persistent DistributedWorker.

    Construct the worker with make_segment_worker(). Metadata must be shared
    CPU tensors created before worker startup:
    attention_counts [world, 1] and moe_counts [world], both int32.
    Per-layer argument maps contain all ABI tensors except input state, counts
    and epoch. They own separate output/scratch/cache buffers. No host copies
    occur between half-layers or between consecutive calls to run_layer().
    """

    def __init__(self, worker, programs, topology, attention_counts, moe_counts, run_config):
        import torch

        for value, shape in ((attention_counts, (topology.world, 1)),
                             (moe_counts, (topology.world,))):
            if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                    or tuple(value.shape) != shape or value.dtype != torch.int32 or not value.is_shared()):
                raise ValueError("counts must be shared CPU int32 metadata allocated before worker startup")
        self.worker, self.programs, self.topology = worker, programs, topology
        self.attention_counts, self.moe_counts = attention_counts, moe_counts
        self.run_config = run_config
        self._epoch = 0
        self._attention_epochs = {}
        self._failed = False

    def _check_state(self, state):
        import torch
        from pypto.runtime import StackedDeviceTensor

        if state.layout != "tp_local_token":
            raise ValueError("SWA segment requires contiguous TP-local token state")
        prefix = (self.topology.world, self.topology.local_capacity)
        for value, shape in ((state.residual, (*prefix, 4, 5120)), (state.pre_mix, (*prefix, 4))):
            if (not isinstance(value, StackedDeviceTensor) or tuple(value.shape) != shape
                    or value.dtype != torch.float32
                    or tuple(value.worker_ids) != tuple(range(self.topology.world))):
                raise ValueError("state must be FP32 device shards with canonical rank placement and capacity")

    def run_layer(self, state, attention, moe, *, group_counts):
        """Return MoE outputs directly for the following layer's Attention input.

        Request packing, RoPE, page mapping and cache initialization are owned
        by the caller and must use the same contiguous group/slab ordering.
        """
        return self._run_layer(state, attention, moe, group_counts=group_counts,
                               attention_program=self.programs[0], argument_names=ATTENTION_ARGS,
                               input_mix="incoming_pre_mix", output_residual="output")

    def _run_layer(self, state, attention, moe, *, group_counts, attention_program,
                   argument_names, input_mix, output_residual):
        """Dispatch an audited TP-local Attention ABI and the shared MoE entry."""
        if self._failed:
            raise RuntimeError("segment dispatch failed; close this worker before attempting recovery")
        self._check_state(state)
        group, local = self.topology.counts(group_counts)
        a = dict(attention, x_hc=state.residual, num_tokens=self.attention_counts)
        a[input_mix] = state.pre_mix
        attention_state = LayerState(a[output_residual], a["next_pre_mix"], "tp_local_token")
        m = dict(moe, x_hc=attention_state.residual, pre_mix=attention_state.pre_mix,
                 num_tokens=self.moe_counts)
        result = LayerState(m["x_next"], m["next_pre_mix"], "tp_local_token")
        self._check_state(attention_state)
        self._check_state(result)
        # Reject aliases before launching either kernel, including distinct
        # handles pointing at the same allocation. Reuse across layers is fine.
        buffers = (state.residual, state.pre_mix, attention_state.residual,
                   attention_state.pre_mix, result.residual, result.pre_mix)
        seen = set()
        for buffer in buffers:
            for rank, shard in zip(buffer.worker_ids, buffer.shards):
                key = (rank, shard.data_ptr)
                if key in seen:
                    raise ValueError("input and half-layer outputs must not alias")
                seen.add(key)
        epoch = self._epoch + 1
        program_key = id(attention_program.compiled)
        attention_epoch = self._attention_epochs.get(program_key, 0) + 1
        if max(epoch, attention_epoch) > (2**31 - 1) // 2:
            raise OverflowError("communication epoch exhausted; recreate worker")
        # Each compiled Attention owns separate retained signal windows. A
        # mode's first call must use epoch 1 even after another mode ran.
        a["attention_epoch"] = ctypes.c_int32(attention_epoch)
        m["moe_epoch"] = ctypes.c_int32(epoch)
        # Resolve all arguments before the first collective: a missing MoE
        # weight must not leave an already-mutated Attention cache behind.
        a_args = [a[name] for name in argument_names]
        m_args = [m[name] for name in MOE_ARGS]
        self._check_weights(m)
        for rank, (global_n, local_n) in enumerate(zip(group, local)):
            self.attention_counts[rank, 0] = global_n
            self.moe_counts[rank] = local_n
        self._epoch = epoch
        self._attention_epochs[program_key] = attention_epoch
        try:
            self.worker.run(attention_program.compiled, *a_args, config=self.run_config)
            self.worker.run(self.programs[1].compiled, *m_args, config=self.run_config)
        except BaseException:
            self._failed = True
            raise
        return result

    @staticmethod
    def _check_weights(moe):
        import torch
        from pypto.runtime import StackedDeviceTensor

        for name in ("routed_w1", "routed_w2", "routed_w3", "mxfp4_pair_lut"):
            value = moe[name]
            expected = torch.int16 if name == "mxfp4_pair_lut" else torch.uint8
            if not isinstance(value, StackedDeviceTensor) or value.dtype != expected:
                raise ValueError(f"{name} must be a resident packed-FP4 ABI tensor ({expected})")

# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Explicit prefill half-layer dispatch for the lib's sequence-parallel entries.

Signatures were audited at lib fbe92bfc. This module does not allocate caches,
invent metadata, or enable the full CLI backend. The caller must supply the
entire audited ABI and retain every program's communication windows.
"""
import importlib

from .swa_segment import ATTENTION_ARGS, SwaSegment, load_segment_modules
from .execution_plan import LayerPlan


_WEIGHTS = (
    "x_hc pre_mix hc_attn_fn hc_attn_scale hc_attn_base attn_norm_weight "
    "wq_a wq_a_scale q_norm_weight wq_b wq_b_scale wkv wkv_scale kv_norm_weight "
    "attn_sink wo_a wo_b wo_b_scale "
)
_WINDOW = "window_slots window_indices window_cache window_cache_scale "
_OUTPUTS = "attn_input attn_output next_pre_mix x_hc_out num_tokens attention_epoch"
C2A_FULL_ARGS = (_WEIGHTS + "freqs_cos freqs_sin " + _WINDOW +
    "compressed_cache compressed_cache_scale token_to_req_indices compressed_lens "
    "index_cache index_cache_scale index_block_table position_ids compressed_freqs_cos compressed_freqs_sin "
    "compressed_rope_positions compressor_wkv compressor_wgate query_start_loc state_block_table state_cache "
    "compressor_norm_weight compressed_slots index_wk index_norm_weight index_wq_b index_wq_b_scale "
    "index_weights_proj topk_indices " + _OUTPUTS).split()
C2A_REUSE_ARGS = (_WEIGHTS + "rope_cos rope_sin " + _WINDOW +
    "compressed_cache compressed_cache_scale compressed_indices " + _OUTPUTS).split()
C1A_ARGS = (_WEIGHTS + "rope_cos rope_sin " + _WINDOW +
    "compressed_cache compressed_cache_scale request_ids compressed_lens index_cache index_cache_scale "
    "index_block_table compressed_rope_cos compressed_rope_sin compressor_wkv compressor_norm_weight "
    "compressed_slots index_wk index_norm_weight index_wq_b index_wq_b_scale index_weights_proj "
    "topk_indices candidate_mask compressed_indices " + _OUTPUTS).split()
PREFILL_ARGUMENTS = {
    "swa": ATTENTION_ARGS, "c2a_full": C2A_FULL_ARGS, "c2a_reuse": C2A_REUSE_ARGS,
    "c1a_full": C1A_ARGS, "c1a_reindex": C1A_ARGS, "c1a_reuse": C1A_ARGS,
}


def bind_prefill_producers(plans, arguments):
    """Bind one ordered forward chain to its actual producer allocations.

    Like V4's worker-owned cache arguments, handles stay resident for the whole
    dispatch. V4.1 additionally shares compressed KV/index pools and physical
    Top-K rows according to LayerPlan. This performs no tensor computation.
    Every producer must execute earlier in this chain, so a previous request's
    transient Top-K/candidate rows cannot satisfy a missing dependency.

    C1A's common ABI retains unused Full weights in Reindex/Reuse. Bind those
    slots to their real producer weights; never allocate placeholder weights.
    Keep unused index slots in separate allocations. The common C1A ABI marks
    them writable even when a selected mode does not use them; aliasing those
    arguments makes the runtime reject overlapping write ranges (lib fbe92bfc).
    """
    plans = tuple(plans)
    if not plans or any(not isinstance(p, LayerPlan) for p in plans):
        raise ValueError("prefill chain requires LayerPlan entries")
    if any(b.layer_id != a.layer_id + 1 for a, b in zip(plans, plans[1:])):
        raise ValueError("prefill chain must execute consecutive layers in order")
    result, seen = {}, {}
    for plan in plans:
        if plan.mode not in PREFILL_ARGUMENTS:
            raise ValueError("unsupported prefill mode")
        current = dict(arguments[plan.layer_id])
        if plan.mode != "swa":
            family, mode = plan.mode.split("_")

            def producer(source, kind):
                if source == plan.layer_id:
                    if mode != "full" and not (kind == "index" and mode == "reindex"):
                        raise ValueError(f"{plan.mode} cannot produce its own {kind}")
                    return current
                if source not in seen or not seen[source].mode.startswith(family + "_"):
                    raise ValueError(f"missing preceding {kind} producer for layer {plan.layer_id}")
                source_mode = seen[source].mode.split("_")[1]
                if source_mode != "full" and not (kind == "index" and source_mode == "reindex"):
                    raise ValueError(f"invalid {kind} producer mode")
                return result[source]

            if mode == "full" and (plan.kv_source != plan.layer_id or plan.index_source != plan.layer_id):
                raise ValueError("Full must own its KV and index outputs")
            if mode == "reindex" and (family != "c1a" or plan.index_source != plan.layer_id):
                raise ValueError("Reindex must own its C1A index output")
            kv = producer(plan.kv_source, "KV")
            index = producer(plan.index_source, "index")
            if plan.index_source != plan.layer_id and seen[plan.index_source].kv_source != plan.kv_source:
                raise ValueError("index selection must address the same KV producer")
            for name in ("compressed_cache", "compressed_cache_scale"):
                current[name] = kv[name]
            if family == "c1a":
                candidate = producer(plan.candidate_source, "candidate")
                if plan.candidate_source != plan.kv_source:
                    raise ValueError("C1A candidates must use the same KV producer")
                for name in ("index_cache", "index_cache_scale", "compressor_wkv", "compressor_norm_weight",
                             "index_wk", "index_norm_weight"):
                    current[name] = kv[name]
                current["candidate_mask"] = candidate["candidate_mask"]
                if mode == "reuse":
                    for name in ("index_wq_b", "index_wq_b_scale", "index_weights_proj"):
                        current[name] = index[name]
                    current["compressed_indices"] = index["topk_indices"]
                if current["compressed_indices"] is current["topk_indices"]:
                    raise ValueError("C1A index ABI arguments require separate allocations")
            elif mode == "reuse":
                current["compressed_indices"] = index["topk_indices"]
        elif any(source is not None for source in (plan.kv_source, plan.index_source, plan.candidate_source)):
            raise ValueError("SWA must not declare compressed producers")
        result[plan.layer_id], seen[plan.layer_id] = current, plan
    return result


def compile_prefill_segments(compiler, lib_root, topology, modes):
    """Compile existing composite entries, sharing one packed-FP4 MoE program.

    C1A must use prefill_c1a_sp, not the older replicated-residual wrappers.
    No model operators are composed by this serving module.
    """
    import pypto.language as pl

    modes = tuple(dict.fromkeys(modes))
    if not modes or any(mode not in PREFILL_ARGUMENTS for mode in modes):
        raise ValueError("unsupported or empty prefill mode selection")
    swa, moe = load_segment_modules(lib_root, topology)
    attention = {}
    for mode in modes:
        if mode == "swa":
            entry = swa.make_hc_program(topology.capacity, topology.world, epochs=1)
        elif mode.startswith("c2a_"):
            module = importlib.import_module("models.deepseek_v4_1_flash.prefill_" + mode)
            entry = module.make_hc_program(topology.capacity, topology.world, epochs=1)
        else:
            module = importlib.import_module("models.deepseek_v4_1_flash.prefill_c1a_sp")
            entry = module.make_program(mode.removeprefix("c1a_"), topology.world, epochs=1)
        attention[mode] = compiler.compile("v41_prefill_" + mode, entry, attention_epoch=pl.RUNTIME)
    ffn = compiler.compile("v41_moe_segment", moe.l3_moe, moe_epoch=pl.RUNTIME)
    return attention, ffn


class PrefillSegment(SwaSegment):
    """Carry device residual/pre_mix through a selected prefill mode and MoE.

    The worker must own all supplied programs via make_segment_worker(). Each
    Attention program advances its own epoch; the shared MoE advances on every
    layer. Cache ownership, cross-layer producer bindings and padded metadata
    remain caller responsibilities. Any runtime failure poisons the whole worker.
    """

    def __init__(self, worker, attention_programs, moe_program, topology,
                 attention_counts, moe_counts, run_config):
        if not attention_programs or any(mode not in PREFILL_ARGUMENTS for mode in attention_programs):
            raise ValueError("unsupported or empty prefill program set")
        self.attention_programs = dict(attention_programs)
        super().__init__(worker, (next(iter(attention_programs.values())), moe_program),
                         topology, attention_counts, moe_counts, run_config)

    def run_layer(self, state, attention, moe, *, group_counts, mode):
        if mode not in self.attention_programs:
            raise ValueError(f"prefill mode was not compiled: {mode}")
        return self._run_layer(
            state, attention, moe, group_counts=group_counts,
            attention_program=self.attention_programs[mode], argument_names=PREFILL_ARGUMENTS[mode],
            input_mix="incoming_pre_mix" if mode == "swa" else "pre_mix",
            output_residual="output" if mode == "swa" else "x_hc_out",
        )

    def run_chain(self, state, plans, attention, moe, *, group_counts):
        """Execute one request step through its producers and consumers in order.

        The resource owner must prepare every layer for the SAME packed step
        and retain all allocations until this synchronous call returns. This
        is still a bounded prefill segment, not a complete serving backend.
        """
        from .swa_segment import MOE_ARGS

        plans = tuple(plans)
        bound = bind_prefill_producers(plans, attention)
        self.topology.counts(group_counts)
        # Resolve the whole chain before any cache mutation. Runtime state,
        # active counts and per-program epochs are provided by run_layer.
        dynamic = {"x_hc", "pre_mix", "incoming_pre_mix", "num_tokens", "attention_epoch", "moe_epoch"}
        for plan in plans:
            if plan.mode not in self.attention_programs:
                raise ValueError(f"prefill mode was not compiled: {plan.mode}")
            for name in PREFILL_ARGUMENTS[plan.mode]:
                if name not in dynamic:
                    bound[plan.layer_id][name]
            for name in MOE_ARGS:
                if name not in dynamic:
                    moe[plan.layer_id][name]
            self._check_weights(moe[plan.layer_id])
        for plan in plans:
            state = self.run_layer(state, bound[plan.layer_id], moe[plan.layer_id],
                                   group_counts=group_counts, mode=plan.mode)
        return state

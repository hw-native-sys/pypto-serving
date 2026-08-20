# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Differential parity for Qwen3 weight staging: the rule table against the executor's own loop.
#
# The Qwen kernels read one stacked slab per weight kind, so the staged output is a contract in
# the same way DeepSeekV4's is — and the two families disagree on two axes that a shape check
# would not catch on its own: Qwen stacks on axis 0 (no rank axis) and stores its projections
# transposed. This reproduces `_stage_stacked_decode_weights` and compares.
from __future__ import annotations

import torch

from pypto_serving.model.common.weights.packer import pack_globals, pack_layer
from pypto_serving.model.common.weights.spec import LayerContext
from pypto_serving.model.common.weights.stacker import stack_layers
from pypto_serving.model.qwen.weight_spec import (
    QWEN_GLOBAL_RULES,
    QWEN_LAYER_RULES,
    QWEN_NORM_WEIGHT_NAMES,
    qwen_layer_prefix,
    qwen_policy,
    qwen_stack_groups,
)

_LAYERS = 3
_HIDDEN = 8
_HEADS = 2
_HEAD_DIM = 4
_INTERMEDIATE = 16
_VOCAB = 10

# (kernel name, checkpoint suffix, kind) exactly as the executor's `fields` tuple lists them.
_FIELDS = (
    ("decode_input_rms_weight", "input_layernorm.weight", "norm"),
    ("decode_wq", "self_attn.q_proj.weight", "proj"),
    ("decode_wk", "self_attn.k_proj.weight", "proj"),
    ("decode_wv", "self_attn.v_proj.weight", "proj"),
    ("decode_q_norm_weight", "self_attn.q_norm.weight", "norm"),
    ("decode_k_norm_weight", "self_attn.k_norm.weight", "norm"),
    ("decode_wo", "self_attn.o_proj.weight", "proj"),
    ("decode_post_rms_weight", "post_attention_layernorm.weight", "norm"),
    ("decode_w_gate", "mlp.gate_proj.weight", "proj"),
    ("decode_w_up", "mlp.up_proj.weight", "proj"),
    ("decode_w_down", "mlp.down_proj.weight", "proj"),
)

_SHAPES = {
    "input_layernorm.weight": (_HIDDEN,),
    "self_attn.q_proj.weight": (_HEADS * _HEAD_DIM, _HIDDEN),
    "self_attn.k_proj.weight": (_HEADS * _HEAD_DIM, _HIDDEN),
    "self_attn.v_proj.weight": (_HEADS * _HEAD_DIM, _HIDDEN),
    "self_attn.q_norm.weight": (_HEAD_DIM,),
    "self_attn.k_norm.weight": (_HEAD_DIM,),
    "self_attn.o_proj.weight": (_HIDDEN, _HEADS * _HEAD_DIM),
    "post_attention_layernorm.weight": (_HIDDEN,),
    "mlp.gate_proj.weight": (_INTERMEDIATE, _HIDDEN),
    "mlp.up_proj.weight": (_INTERMEDIATE, _HIDDEN),
    "mlp.down_proj.weight": (_HIDDEN, _INTERMEDIATE),
}


def _raw_layers(*, with_qk_norm=True):
    """Per-layer checkpoint tensors, distinct per layer so a misplaced layer changes the bytes."""
    raw = {}
    for layer_id in range(_LAYERS):
        prefix = qwen_layer_prefix(layer_id)
        for index, (suffix, shape) in enumerate(_SHAPES.items()):
            if not with_qk_norm and suffix in {"self_attn.q_norm.weight", "self_attn.k_norm.weight"}:
                continue
            count = 1
            for dim in shape:
                count *= dim
            values = torch.arange(count, dtype=torch.float32) + (layer_id * 1000 + index * 10)
            raw[f"{prefix}.{suffix}"] = values.reshape(shape).to(torch.bfloat16)
    return raw


def _legacy_stage(raw, *, with_qk_norm=True):
    """Reproduce `_stage_stacked_decode_weights`: pre-allocate, then copy per-layer views in."""

    def ready_view(layer_id, suffix, kind):
        name = f"{qwen_layer_prefix(layer_id)}.{suffix}"
        tensor = raw.get(name)
        if tensor is None:
            tensor = torch.ones(_HEAD_DIM, dtype=torch.bfloat16)
        tensor = tensor.cpu()
        return tensor.transpose(0, 1) if kind == "proj" else tensor.reshape(1, -1)

    stacked = {}
    rows_by_key = {}
    for key, suffix, kind in _FIELDS:
        first = ready_view(0, suffix, kind)
        if kind == "proj":
            rows = int(first.shape[0])
            shape = (_LAYERS * rows, int(first.shape[1]))
            dtype = torch.bfloat16
        else:
            rows = 1
            shape = (_LAYERS, int(first.shape[1]))
            dtype = torch.float32
        rows_by_key[key] = rows
        stacked[key] = torch.empty(shape, dtype=dtype)
    for layer_id in range(_LAYERS):
        for key, suffix, kind in _FIELDS:
            rows = rows_by_key[key]
            stacked[key][layer_id * rows : (layer_id + 1) * rows].copy_(
                ready_view(layer_id, suffix, kind)
            )
    return stacked


def _spec_stage(raw):
    """Stage through the rule table and the generic stacker, on axis 0 with no rank axis."""

    def context(layer_id):
        return LayerContext(
            layer_id=layer_id,
            prefix=qwen_layer_prefix(layer_id),
            ranks=1,
            dims={"head_dim": _HEAD_DIM},
        )

    template = pack_layer(QWEN_LAYER_RULES, raw, context(0), policy=qwen_policy())

    def pack_into(layer_id, destinations):
        pack_layer(
            QWEN_LAYER_RULES, raw, context(layer_id), policy=qwen_policy(), destinations=destinations
        )

    return stack_layers(
        qwen_stack_groups(_LAYERS),
        template,
        layer_ids=range(_LAYERS),
        pack_into=pack_into,
        template_layer_id=0,
        stack_axis=0,
    )


def test_the_rule_table_stages_what_the_executor_stages(fingerprint_tensors):
    """All eleven slabs, byte for byte, including the transposes and the [1, dim] gammas."""
    raw = _raw_layers()

    legacy = _legacy_stage(raw)
    spec = _spec_stage(raw)

    assert list(legacy) == list(spec), "slab order is the layout the kernels read"
    assert fingerprint_tensors(legacy) == fingerprint_tensors(spec)


def test_absent_qk_norms_default_to_ones(fingerprint_tensors):
    """A Qwen3 checkpoint without QK norms is a variant, and ones is the neutral gamma.

    Zeros would annihilate the activations the gamma scales rather than leaving them unscaled —
    a wrong default here changes the model's output without changing a single shape.
    """
    raw = _raw_layers(with_qk_norm=False)

    legacy = _legacy_stage(raw, with_qk_norm=False)
    spec = _spec_stage(raw)

    assert fingerprint_tensors(legacy) == fingerprint_tensors(spec)
    assert torch.all(spec["decode_q_norm_weight"] == 1.0)
    assert torch.all(spec["decode_k_norm_weight"] == 1.0)


def test_projections_are_stacked_transposed_and_norms_are_rows():
    """The two family-specific shapes, asserted rather than assumed."""
    spec = _spec_stage(_raw_layers())

    # wq is [heads*head_dim, hidden] in the checkpoint; stacked it is [layers*hidden, heads*head_dim].
    assert spec["decode_wq"].shape == (_LAYERS * _HIDDEN, _HEADS * _HEAD_DIM)
    assert spec["decode_wq"].dtype == torch.bfloat16
    # a gamma is [hidden]; stacked it is [layers, hidden], one row per layer.
    assert spec["decode_input_rms_weight"].shape == (_LAYERS, _HIDDEN)
    assert spec["decode_input_rms_weight"].dtype == torch.float32
    for name in QWEN_NORM_WEIGHT_NAMES:
        assert spec[name].shape[0] == _LAYERS, f"{name} must stack one row per layer"


def test_no_slab_carries_a_rank_axis():
    """Qwen kernels take one copy of each weight; a leading 1-axis would be a DeepSeek habit."""
    spec = _spec_stage(_raw_layers())

    for name, tensor in spec.items():
        assert tensor.ndim == 2, f"{name} has rank {tensor.ndim}, expected 2"


def test_a_reordered_layer_changes_the_slabs(fingerprint_tensors):
    """The sensitivity check: placement is what a shape comparison cannot see."""
    raw = _raw_layers()
    swapped = dict(raw)
    for suffix in _SHAPES:
        a = f"{qwen_layer_prefix(0)}.{suffix}"
        b = f"{qwen_layer_prefix(1)}.{suffix}"
        swapped[a], swapped[b] = raw[b], raw[a]

    assert fingerprint_tensors(_spec_stage(raw)) != fingerprint_tensors(_spec_stage(swapped))


class TestQwenGlobals:
    """Embedding, LM head and final norm — where the padding rules differ."""

    def _globals(self, *, tied: bool):
        embed = torch.arange(_VOCAB * _HIDDEN, dtype=torch.float32).reshape(_VOCAB, _HIDDEN)
        available = {
            "model.embed_tokens.weight": embed,
            "model.norm.weight": torch.ones(_HIDDEN, dtype=torch.float32),
        }
        if not tied:
            available["lm_head.weight"] = embed + 100.0
        return available

    def test_the_embedding_pads_with_zeros_and_the_head_replicates_row_zero(self):
        packed = pack_globals(QWEN_GLOBAL_RULES, self._globals(tied=False), padded_rows={
            "embed_weight": 16, "lm_head_weight": 16
        })

        assert packed["embed_weight"].shape == (16, _HIDDEN)
        assert torch.all(packed["embed_weight"][_VOCAB:] == 0.0)
        assert packed["lm_head_weight"].shape == (16, _HIDDEN)
        head = self._globals(tied=False)["lm_head.weight"].to(torch.bfloat16)
        for row in range(_VOCAB, 16):
            assert torch.equal(packed["lm_head_weight"][row], head[0])

    def test_a_tied_checkpoint_uses_the_embedding_as_its_head(self):
        packed = pack_globals(QWEN_GLOBAL_RULES, self._globals(tied=True))

        assert torch.equal(
            packed["lm_head_weight"][:_VOCAB], packed["embed_weight"][:_VOCAB]
        ), "a tied head must be the embedding, not zeros"

    def test_the_final_norm_becomes_a_float32_row(self):
        packed = pack_globals(QWEN_GLOBAL_RULES, self._globals(tied=False))

        assert packed["final_norm_weight"].shape == (1, _HIDDEN)
        assert packed["final_norm_weight"].dtype == torch.float32

    def test_the_pad_multiple_matches_the_kernels_vocab_chunk(self):
        """512 must stay a multiple of the fused LM head's VOCAB_CHUNK (64)."""
        from pypto_serving.model.qwen.weight_spec import QWEN_VOCAB_PAD_MULTIPLE

        assert QWEN_VOCAB_PAD_MULTIPLE % 64 == 0

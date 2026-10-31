# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host-side DSpark state and workspace contracts for DeepSeek V4.1.

The lib owns the device programs and their stage order.  This module owns only
the bounded serving resources needed to call those programs.  It intentionally
does not load or invent draft weights: a production provider must supply a
verified payload builder for the selected lib revision.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable

import torch

from .dspark_runtime import (
    DSPARK_ROWS_PER_REQUEST,
    DSPARK_SPECULATIVE_TOKENS,
    DSparkDecodeDispatch,
    DSparkLoweredInputs,
    DSparkPrefillDispatch,
)


DSPARK_MAX_BATCH_PER_DP = 12
DSPARK_PADDED_ROWS = 80
DSPARK_TARGET_ROWS = DSPARK_MAX_BATCH_PER_DP * DSPARK_ROWS_PER_REQUEST
DSPARK_MAIN_HIDDEN_SIZE = 15360
DSPARK_HIDDEN_SIZE = 5120
DSPARK_HC_MULT = 4
DSPARK_WINDOW = 128
# The drafter's RoPE ABI is qk_rope_head_dim // 2 (64 // 2), not the
# compressed-attention RoPE width used by the target backbone.
DSPARK_ROPE_HALF = 32
DSPARK_DRAFT_LAYERS = 3
DSPARK_PREFILL_SAMPLE_ROWS = 16
DSPARK_SAMPLED_STRIDE = 8

_STATE_TOKEN_WIDTH = DSPARK_ROWS_PER_REQUEST
_STATE_META_WIDTH = 6
_STATE_VALID = 0
_STATE_GENERATION = 1

# These are the non-workspace arguments consumed by the current lib composites.
# The target backbone has additional per-layer arguments and is checked by its
# compiled ABI; the drafter/Markov heads are fixed by the V4.1 DSpark entry.
_DRAFT_MODEL_KEYS = frozenset({
    "embedding_weight", "main_proj_weight", "main_norm_weight",
    "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_weight",
    "wq_a", "wq_a_scale", "q_norm_weight", "wq_b", "wq_b_scale",
    "wkv", "wkv_scale", "kv_norm_weight", "attn_sink", "wo_a", "wo_a_scale",
    "wo_b", "wo_b_scale", "window_cache", "window_cache_scale",
    "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "ffn_norm_weight",
    "gate_weight", "correction_bias", "routed_w1", "routed_w1_scale",
    "routed_w2", "routed_w2_scale", "routed_w3", "routed_w3_scale",
    "mxfp4_pair_lut", "shared_w1", "shared_w1_scale", "shared_w2",
    "shared_w2_scale", "shared_w3", "shared_w3_scale",
})
_MARKOV_MODEL_KEYS = frozenset({
    "final_norm_weight", "lm_head_weight", "markov_w1", "markov_w2", "confidence_weight",
})
_CACHE_TABLE_KEYS = frozenset({"target_window", "target_c2a", "target_c1a", "draft"})
_ROPE_TABLE_KEYS = frozenset({"draft_cos", "draft_sin"})
_TARGET_DYNAMIC_KEYS = frozenset({
    "embedding_weight", "x_hc", "pre_mix", "dspark_target_hidden", "position_ids",
    "token_to_req_indices", "window_slots", "window_indices", "c2a_compressed_slots",
    "c1a_compressed_slots", "query_start_loc", "logit_row_indices", "logits", "sampled_ids",
    "attention_num_tokens", "moe_num_tokens",
})
_DRAFT_DYNAMIC_KEYS = frozenset({
    "target_hidden", "num_sampled", "last_sampled", "next_prefill_tokens", "anchor_ids",
    "first_query_positions", "block_tables", "context_slots", "context_rope_cos",
    "context_rope_sin", "query_rope_cos", "query_rope_sin", "head_hidden",
    "attention_epoch", "moe_epoch",
})
_MARKOV_DYNAMIC_KEYS = frozenset({
    "head_hidden", "num_sampled", "last_sampled", "next_prefill_tokens", "anchor_ids",
    "draft_ids", "confidences", "logits_epoch",
})


def _missing(message: str) -> NotImplementedError:
    """Construct the runner's public missing-composite exception lazily."""
    # Importing npu_runner at module import time would create a cycle: the
    # runner imports this module when production bindings are assembled.
    from .npu_runner import MissingCompositeInterface

    return MissingCompositeInterface(message)


@dataclass(frozen=True)
class DSparkResourceLayout:
    """Fixed dimensions shared by the V4.1 DSpark host ABI."""

    dp_size: int
    max_seq_len: int
    hidden_size: int = DSPARK_HIDDEN_SIZE
    main_hidden_size: int = DSPARK_MAIN_HIDDEN_SIZE
    vocab_size: int = 0
    max_prefill_tokens: int = 8192

    def __post_init__(self) -> None:
        if self.dp_size not in (4, 8):
            raise ValueError("V4.1 DSpark resources require DP/EP size 4 or 8")
        if type(self.max_seq_len) is not int or self.max_seq_len <= 0:
            raise ValueError("DSpark max_seq_len must be a positive integer")
        if self.hidden_size != DSPARK_HIDDEN_SIZE:
            raise ValueError("V4.1 DSpark drafter hidden size must be 5120")
        if self.main_hidden_size != DSPARK_MAIN_HIDDEN_SIZE:
            raise ValueError("V4.1 DSpark target hidden size must be 15360")
        if type(self.vocab_size) is not int or self.vocab_size <= 0:
            raise ValueError("DSpark vocab_size must be a positive integer")
        if type(self.max_prefill_tokens) is not int or self.max_prefill_tokens <= 0:
            raise ValueError("DSpark max_prefill_tokens must be a positive integer")
        if self.max_prefill_tokens > 8192:
            raise ValueError("V4.1 DSpark prefill supports at most 8192 tokens per DP rank")

    @property
    def state_shapes(self) -> Mapping[str, tuple[int, ...]]:
        return {
            "slot_ids": (self.dp_size, DSPARK_MAX_BATCH_PER_DP),
            "generations": (self.dp_size, DSPARK_MAX_BATCH_PER_DP),
            "state_tokens": (self.dp_size, DSPARK_MAX_BATCH_PER_DP, _STATE_TOKEN_WIDTH),
            "state_meta": (self.dp_size, DSPARK_MAX_BATCH_PER_DP, _STATE_META_WIDTH),
        }

    @property
    def state_dtypes(self) -> Mapping[str, torch.dtype]:
        return {
            "slot_ids": torch.int32,
            "generations": torch.int32,
            "state_tokens": torch.int64,
            "state_meta": torch.int32,
        }

    @property
    def decode_shapes(self) -> Mapping[str, tuple[int, ...]]:
        world, batch, rows = self.dp_size, DSPARK_MAX_BATCH_PER_DP, DSPARK_ROWS_PER_REQUEST
        return {
            "input_ids": (world, batch, rows),
            "target_positions": (world, batch, rows),
            "active_widths": (world, batch),
            "x_hc": (world, DSPARK_PADDED_ROWS, DSPARK_HC_MULT, self.hidden_size),
            "pre_mix": (world, DSPARK_PADDED_ROWS, DSPARK_HC_MULT),
            "position_ids": (world, DSPARK_PADDED_ROWS),
            "request_ids": (world, DSPARK_PADDED_ROWS),
            "window_slots": (world, DSPARK_PADDED_ROWS),
            "window_indices": (world, DSPARK_PADDED_ROWS, DSPARK_WINDOW),
            "c2a_slots": (world, DSPARK_PADDED_ROWS),
            "c1a_slots": (world, DSPARK_PADDED_ROWS),
            "query_starts": (world, batch + 1),
            "logit_row_indices": (world, DSPARK_PREFILL_SAMPLE_ROWS),
            "target_hidden_padded": (world, DSPARK_PADDED_ROWS, self.main_hidden_size),
            "target_hidden": (world, DSPARK_TARGET_ROWS, self.main_hidden_size),
            **{
                f"chunk_{chunk}": (world, 16, DSPARK_SAMPLED_STRIDE)
                for chunk in range(DSPARK_SPECULATIVE_TOKENS)
            },
            "target_sampled_ids": (world, DSPARK_TARGET_ROWS, DSPARK_SAMPLED_STRIDE),
            "accepted_ids": (world, batch, rows),
            "accepted_counts": (world, batch),
            "context_valid": (world, batch, rows),
            "context_positions": (world, batch, rows),
            "num_sampled": (world, batch),
            "last_sampled": (world, batch),
            "anchor_ids": (world, batch),
            "first_query_positions": (world, batch),
            "context_slots": (world, DSPARK_DRAFT_LAYERS, DSPARK_TARGET_ROWS),
            "context_rope_cos": (world, DSPARK_TARGET_ROWS, DSPARK_ROPE_HALF),
            "context_rope_sin": (world, DSPARK_TARGET_ROWS, DSPARK_ROPE_HALF),
            "query_rope_cos": (world, batch * DSPARK_SPECULATIVE_TOKENS, DSPARK_ROPE_HALF),
            "query_rope_sin": (world, batch * DSPARK_SPECULATIVE_TOKENS, DSPARK_ROPE_HALF),
            "draft_head_hidden": (world, batch, DSPARK_SPECULATIVE_TOKENS, self.hidden_size),
            "draft_ids": (world, batch, DSPARK_SPECULATIVE_TOKENS),
            "confidences": (world, batch, DSPARK_SPECULATIVE_TOKENS),
            "committed": (world, batch),
            "zero_prefill_tokens": (world, batch),
        }

    @property
    def decode_dtypes(self) -> Mapping[str, torch.dtype]:
        result = {name: torch.int32 for name in (
            "target_positions", "active_widths", "position_ids", "request_ids",
            "window_indices", "query_starts", "logit_row_indices", "accepted_ids", "accepted_counts",
            "context_valid", "context_positions", "num_sampled", "first_query_positions",
            "draft_ids", "committed", "target_sampled_ids", "chunk_0", "chunk_1",
            "chunk_2", "chunk_3", "chunk_4",
        )}
        result.update({name: torch.int64 for name in (
            "input_ids", "window_slots", "c2a_slots", "c1a_slots", "context_slots",
            "last_sampled", "anchor_ids", "zero_prefill_tokens",
        )})
        result.update({name: torch.float32 for name in (
            "x_hc", "pre_mix", "context_rope_cos", "context_rope_sin",
            "query_rope_cos", "query_rope_sin", "confidences",
        )})
        result.update({name: torch.bfloat16 for name in ("target_hidden_padded", "target_hidden", "draft_head_hidden")})
        expected = set(self.decode_shapes)
        if set(result) != expected:
            raise AssertionError(f"internal DSpark dtype table mismatch: {sorted(expected ^ set(result))}")
        return result

    @property
    def prefill_shapes(self) -> Mapping[str, tuple[int, ...]]:
        world, batch = self.dp_size, DSPARK_MAX_BATCH_PER_DP
        return {
            "prompt_lengths": (world, batch),
            "position_limits": (world, batch),
            "sampled_rows": (world, batch),
            "prefill_sampled_ids": (world, DSPARK_PREFILL_SAMPLE_ROWS, DSPARK_SAMPLED_STRIDE),
            "next_prefill_tokens": (world, batch),
            "prefill_positions": (world, self.max_prefill_tokens),
            "prefill_request_ids": (world, self.max_prefill_tokens),
            "prefill_counts": (world, 1),
            "prefill_target_hidden": (world, self.max_prefill_tokens, self.main_hidden_size),
            "zero_sampled": (world, batch),
            "zero_last_sampled": (world, batch),
        }

    @property
    def prefill_dtypes(self) -> Mapping[str, torch.dtype]:
        result = {name: torch.int32 for name in (
            "prompt_lengths", "position_limits", "sampled_rows", "prefill_sampled_ids",
            "prefill_positions", "prefill_request_ids", "prefill_counts",
        )}
        result["zero_sampled"] = torch.int32
        result.update({name: torch.int64 for name in ("next_prefill_tokens", "zero_last_sampled")})
        result["prefill_target_hidden"] = torch.bfloat16
        return result


@dataclass
class DSparkHostResources:
    """Shared host tensors retained until the worker and provider are closed."""

    layout: DSparkResourceLayout
    state: "DSparkHostState"
    decode_workspace: dict[str, torch.Tensor]
    prefill_workspace: dict[str, torch.Tensor]
    cache_tables: dict[str, torch.Tensor]
    rope_tables: dict[str, torch.Tensor]

    @property
    def workspace(self) -> Mapping[str, torch.Tensor]:
        return self.decode_workspace

    def for_phase(self, phase: str) -> dict[str, torch.Tensor]:
        if phase == "decode":
            return self.decode_workspace
        if phase == "prefill":
            return self.prefill_workspace
        raise ValueError("DSpark phase must be 'prefill' or 'decode'")


@dataclass
class DSparkHostState:
    """Host-backed object with the attribute ABI used by lib DSparkState."""

    slot_ids: torch.Tensor
    generations: torch.Tensor
    tokens: torch.Tensor
    meta: torch.Tensor

    def __getitem__(self, name: str) -> torch.Tensor:
        name = {"state_tokens": "tokens", "state_meta": "meta"}.get(name, name)
        try:
            return getattr(self, name)
        except AttributeError as exc:
            raise KeyError(name) from exc

    def values(self):
        return (self.slot_ids, self.generations, self.tokens, self.meta)

    def bind_request(self, rank: int, row: int, *, slot: int, generation: int) -> None:
        """Bind one batch row to its persistent local state slot."""
        if any(type(value) is not int for value in (rank, row, slot, generation)):
            raise TypeError("DSpark state indices and generation must be integers")
        if not 0 <= rank < self.slot_ids.shape[0]:
            raise ValueError("DSpark state rank is outside the EP topology")
        if not 0 <= row < self.slot_ids.shape[1]:
            raise ValueError("DSpark state request row is outside the batch capacity")
        if not 0 <= slot < self.meta.shape[1] or generation < 0:
            raise ValueError("DSpark state slot or generation is invalid")
        self.slot_ids[rank, row] = slot
        self.generations[rank, row] = generation

    def reserve_slot(self, rank: int, slot: int, *, generation: int, position_limit: int) -> None:
        """Reset a persistent slot and publish its generation before bootstrap."""
        if any(type(value) is not int for value in (rank, slot, generation, position_limit)):
            raise TypeError("DSpark state reservation values must be integers")
        if not 0 <= rank < self.meta.shape[0]:
            raise ValueError("DSpark state rank is outside the EP topology")
        if not 0 <= slot < self.meta.shape[1] or generation < 0 or position_limit <= 0:
            raise ValueError("DSpark state reservation is invalid")
        self.tokens[rank, slot].zero_()
        self.meta[rank, slot].zero_()
        self.meta[rank, slot, _STATE_GENERATION] = generation
        self.meta[rank, slot, 5] = position_limit


def _shared(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    value = torch.empty(shape, dtype=dtype).share_memory_()
    value.zero_()
    return value


def allocate_dspark_host_resources(layout: DSparkResourceLayout) -> DSparkHostResources:
    """Allocate only bounded, zeroed host state/workspace buffers.

    Weight tensors, device handles and cache page contents deliberately do not
    belong here.  A provider must bind those from a verified checkpoint/runtime
    contract before returning a dispatch.
    """
    state = DSparkHostState(
        slot_ids=_shared(layout.state_shapes["slot_ids"], layout.state_dtypes["slot_ids"]),
        generations=_shared(layout.state_shapes["generations"], layout.state_dtypes["generations"]),
        tokens=_shared(layout.state_shapes["state_tokens"], layout.state_dtypes["state_tokens"]),
        meta=_shared(layout.state_shapes["state_meta"], layout.state_dtypes["state_meta"]),
    )
    state.slot_ids.fill_(-1)
    # The bootstrap composite accepts only reset rows with STATE_VALID == 0.
    state.meta.zero_()
    state.tokens.zero_()
    decode = {name: _shared(shape, layout.decode_dtypes[name]) for name, shape in layout.decode_shapes.items()}
    for name in ("position_ids", "request_ids", "window_slots", "window_indices", "c2a_slots", "c1a_slots",
                 "context_valid", "context_positions"):
        decode[name].fill_(-1)
    prefill = dict(decode)
    # Prefill geometry consumes the complete prompt-token axis, whereas decode
    # only consumes the fixed six-row target axis.  Keep separate buffers for
    # the dynamic axes so a long prompt cannot overwrite decode scratch.
    prefill["context_slots"] = _shared(
        (layout.dp_size, DSPARK_DRAFT_LAYERS, layout.max_prefill_tokens), torch.int64)
    prefill["context_rope_cos"] = _shared(
        (layout.dp_size, layout.max_prefill_tokens, DSPARK_ROPE_HALF), torch.float32)
    prefill["context_rope_sin"] = _shared(
        (layout.dp_size, layout.max_prefill_tokens, DSPARK_ROPE_HALF), torch.float32)
    prefill["prefill_target_hidden"] = _shared(
        (layout.dp_size, layout.max_prefill_tokens, layout.main_hidden_size), torch.bfloat16)
    prefill.update({name: _shared(shape, layout.prefill_dtypes[name])
                    for name, shape in layout.prefill_shapes.items()})
    for name in ("sampled_rows", "prefill_positions", "prefill_request_ids"):
        prefill[name].fill_(-1)
    table_pages = (layout.max_seq_len + DSPARK_WINDOW - 1) // DSPARK_WINDOW
    compressed_pages = (layout.max_seq_len + 255) // 256
    cache_tables = {
        "target_window": _shared(
            (layout.dp_size, DSPARK_MAX_BATCH_PER_DP, table_pages), torch.int32),
        "target_c2a": _shared(
            (layout.dp_size, DSPARK_MAX_BATCH_PER_DP, compressed_pages), torch.int32),
        "target_c1a": _shared(
            (layout.dp_size, DSPARK_MAX_BATCH_PER_DP, table_pages), torch.int32),
        "draft": _shared(
            (layout.dp_size, DSPARK_DRAFT_LAYERS, DSPARK_MAX_BATCH_PER_DP, table_pages),
            torch.int32),
    }
    for value in cache_tables.values():
        value.fill_(-1)
    # Geometry uses the same full precision RoPE convention as the target
    # metadata resources.  The provider fills these from decode_host's tables.
    rope_tables = {
        "draft_cos": _shared(
            (layout.dp_size, layout.max_seq_len, DSPARK_ROPE_HALF), torch.float32),
        "draft_sin": _shared(
            (layout.dp_size, layout.max_seq_len, DSPARK_ROPE_HALF), torch.float32),
    }
    return DSparkHostResources(layout, state, decode, prefill, cache_tables, rope_tables)


def validate_resource_mapping(resources: DSparkHostResources, phase: str) -> None:
    """Check shape/dtype/shared-memory invariants before any lib dispatch."""
    layout = resources.layout
    for group, shapes, dtypes in (
        ("state", layout.state_shapes, layout.state_dtypes),
        ("workspace", layout.decode_shapes, layout.decode_dtypes),
    ):
        values = ({
            "slot_ids": resources.state.slot_ids,
            "generations": resources.state.generations,
            "state_tokens": resources.state.tokens,
            "state_meta": resources.state.meta,
        } if group == "state" else resources.decode_workspace)
        for name, shape in shapes.items():
            value = values.get(name)
            if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                    or value.dtype != dtypes[name] or value.device.type != "cpu"
                    or not value.is_shared() or not value.is_contiguous()):
                raise ValueError(f"DSpark {group} tensor {name!r} violates ABI shape/dtype/shared contract")
    if phase == "prefill":
        for name, shape in layout.prefill_shapes.items():
            value = resources.prefill_workspace.get(name)
            if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                    or value.dtype != layout.prefill_dtypes[name] or value.device.type != "cpu"
                    or not value.is_shared() or not value.is_contiguous()):
                raise ValueError(f"DSpark prefill tensor {name!r} violates ABI shape/dtype/shared contract")
    elif phase != "decode":
        raise ValueError("DSpark phase must be 'prefill' or 'decode'")


def _validate_complete_lowered(lowered: DSparkLoweredInputs, resources: DSparkHostResources, phase: str) -> None:
    validate_resource_mapping(resources, phase)
    if lowered.state is not resources.state:
        raise ValueError("DSpark lowered state must be owned by the resident resource allocator")
    workspace = resources.for_phase(phase)
    if lowered.workspace is not workspace:
        raise ValueError("DSpark lowered workspace must be owned by the resident resource allocator")
    required = ("draft_model", "markov_model", "cache_tables", "rope_tables")
    missing = [name for name in required if not getattr(lowered, name)]
    if missing:
        raise _missing("V4.1 DSpark payload is missing: " + ", ".join(missing))
    for name, keys, mapping in (
        ("draft_model", _DRAFT_MODEL_KEYS, lowered.draft_model),
        ("markov_model", _MARKOV_MODEL_KEYS, lowered.markov_model),
        ("cache_tables", _CACHE_TABLE_KEYS, lowered.cache_tables),
        ("rope_tables", _ROPE_TABLE_KEYS, lowered.rope_tables),
    ):
        absent = sorted(keys - set(mapping))
        if absent:
            raise _missing(f"V4.1 DSpark {name} is missing ABI fields: {', '.join(absent)}")
    if phase == "decode" and not lowered.target_model:
        raise _missing("V4.1 DSpark decode payload is missing target_model")
    if phase == "decode":
        absent = sorted({"embedding_weight", "final_norm_weight", "lm_head_weight", "logit_row_indices"}
                        - set(lowered.target_model))
        if absent:
            raise _missing("V4.1 DSpark target_model is missing ABI fields: " + ", ".join(absent))


class V41DSparkLoweringProvider:
    """Validate complete payloads supplied by a real serving binder.

    ``payload_builder`` is intentionally mandatory for production use.  It is
    where checkpoint-backed draft/Markov weights, cache page tables and the
    target hidden handoff are bound.  This class owns the resource lifetime and
    rejects a partial or synthetic payload before the lib orchestrator runs.
    """

    def __init__(
        self,
        resources: DSparkHostResources,
        *,
        payload_builder: Callable[..., DSparkLoweredInputs] | None = None,
        state_resolver: Callable[[str], tuple[int, int]] | None = None,
        target_param_names: Sequence[str] = (),
        draft_param_names: Sequence[str] = (),
        markov_param_names: Sequence[str] = (),
    ) -> None:
        if not isinstance(resources, DSparkHostResources):
            raise TypeError("DSpark provider requires DSparkHostResources")
        validate_resource_mapping(resources, "decode")
        self.resources = resources
        self.payload_builder = payload_builder
        self.state_resolver = state_resolver
        self._target_param_names = frozenset(target_param_names)
        self._draft_param_names = frozenset(draft_param_names)
        self._markov_param_names = frozenset(markov_param_names)

    def _build(self, *, model: Any, batch: Any, phase: str, **extra: Any) -> DSparkLoweredInputs:
        if self.payload_builder is None:
            raise _missing(
                f"V4.1 DSpark {phase} lowering requires a checkpoint-backed payload builder; "
                "draft/Markov weights and cache bindings are not available"
            )
        value = self.payload_builder(
            model=model, batch=batch, phase=phase, resources=self.resources, **extra,
        )
        if not isinstance(value, DSparkLoweredInputs):
            raise TypeError("DSpark payload builder must return DSparkLoweredInputs")
        _validate_complete_lowered(value, self.resources, phase)
        if phase == "decode":
            self._validate_compiled_abis(value)
        return value

    def _validate_compiled_abis(self, lowered: DSparkLoweredInputs) -> None:
        """Cross-check payload mappings against the compiled lib parameter lists."""
        checks = (
            ("target_model", self._target_param_names - _TARGET_DYNAMIC_KEYS, lowered.target_model),
            ("draft_model", self._draft_param_names - _DRAFT_DYNAMIC_KEYS, lowered.draft_model),
            ("markov_model", self._markov_param_names - _MARKOV_DYNAMIC_KEYS, lowered.markov_model),
        )
        for name, required, mapping in checks:
            absent = sorted(required - set(mapping))
            if absent:
                raise _missing(f"V4.1 DSpark {name} is missing compiled ABI fields: {', '.join(absent)}")

    def _resolve_state(self, request_id: str, rank_hint: int) -> tuple[int, int, int]:
        """Resolve ``(rank, slot, generation)`` with a legacy two-field hook."""
        if self.state_resolver is None:
            raise _missing("V4.1 DSpark lowering requires a request state resolver")
        value = tuple(self.state_resolver(request_id))
        if len(value) == 2:
            slot, generation = value
            rank = rank_hint
        elif len(value) == 3:
            rank, slot, generation = value
            if int(rank) != rank_hint:
                raise ValueError("DSpark state resolver rank disagrees with the cache partition")
        else:
            raise ValueError("DSpark state resolver must return (slot, generation) or (rank, slot, generation)")
        rank, slot, generation = int(rank), int(slot), int(generation)
        if not 0 <= rank < self.resources.layout.dp_size:
            raise ValueError("DSpark state rank is outside the selected DP topology")
        if not 0 <= slot < DSPARK_MAX_BATCH_PER_DP or generation < 0:
            raise ValueError("DSpark state slot or generation is invalid")
        return rank, slot, generation

    def lower_prefill(self, *, model: Any, batch: Any, request_ids: Sequence[str],
                      sampled_token_ids: Sequence[int]) -> DSparkPrefillDispatch:
        if not request_ids or len(request_ids) != len(sampled_token_ids):
            raise ValueError("DSpark prefill request and sampled-token rows must match")
        if self.state_resolver is None:
            raise _missing("V4.1 DSpark prefill lowering requires a request state resolver")
        partitions = getattr(batch, "cache_partitions", ())
        rank = 0 if not partitions or partitions[0] is None else int(partitions[0])
        rank, slot, generation = self._resolve_state(request_ids[0], rank)
        request_rows = tuple(getattr(batch, "request_ids", ()))
        row = request_rows.index(request_ids[0]) if request_ids[0] in request_rows else 0
        # The lib bootstrap composite verifies the generation before publishing
        # the sampled token.  Check the row before lowering, but only mutate it
        # after the complete payload builder succeeds.
        state_meta = self.resources.state.meta[rank, int(slot)]
        if int(state_meta[_STATE_VALID]) != 0:
            raise ValueError("DSpark prefill state slot is already active")
        prompt_lengths = getattr(batch, "prompt_lens", ())
        if not prompt_lengths:
            raise ValueError("DSpark prefill batch is missing prompt_lens")
        prompt_length = int(prompt_lengths[row])
        lowered = self._build(
            model=model, batch=batch, phase="prefill",
            request_ids=tuple(request_ids), sampled_token_ids=tuple(int(v) for v in sampled_token_ids),
        )
        self.resources.state.slot_ids.fill_(-1)
        self.resources.state.generations.zero_()
        self.resources.state.bind_request(rank, row, slot=slot, generation=generation)
        self.resources.state.reserve_slot(
            rank, slot, generation=generation, position_limit=self.resources.layout.max_seq_len,
        )
        return DSparkPrefillDispatch(
            request_id=str(request_ids[0]), slot=int(slot), generation=int(generation),
            prompt_length=prompt_length, anchor=prompt_length, stages=(), lowered=lowered,
        )

    def lower_decode(self, *, model: Any, batch: Any) -> DSparkDecodeDispatch:
        request_ids = tuple(getattr(batch, "request_ids", ()))
        if not request_ids:
            raise ValueError("DSpark decode batch must contain request IDs")
        partitions = tuple(getattr(batch, "cache_partitions", ()))
        if len(partitions) != len(request_ids):
            raise ValueError("DSpark decode cache partitions must match request order")
        self.resources.state.slot_ids.fill_(-1)
        self.resources.state.generations.zero_()
        request_ranks = []
        next_rows: dict[int, int] = {}
        for request_id, partition in zip(request_ids, partitions, strict=True):
            rank_hint = 0 if partition is None else int(partition)
            rank, slot, generation = self._resolve_state(request_id, rank_hint)
            row = next_rows.get(rank, 0)
            next_rows[rank] = row + 1
            self.resources.state.bind_request(rank, row, slot=slot, generation=generation)
            request_ranks.append(rank)
        lowered = self._build(model=model, batch=batch, phase="decode")
        ranks = lowered.request_ranks
        if ranks is None:
            ranks = tuple(request_ranks)
        if len(ranks) != len(request_ids):
            raise ValueError("DSpark request ranks must match decode request order")
        return DSparkDecodeDispatch(
            request_ids=request_ids, stages=(), lowered=lowered, request_ranks=tuple(ranks),
        )


__all__ = [
    "DSparkHostState",
    "DSparkHostResources",
    "DSparkResourceLayout",
    "V41DSparkLoweringProvider",
    "allocate_dspark_host_resources",
    "validate_resource_mapping",
]

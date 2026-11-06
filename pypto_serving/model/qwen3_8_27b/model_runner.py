# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""One-card Qwen3.8 request adapter over the prefill and decoder stacks."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import torch
from pypto.runtime import DeviceTensor

from pypto_serving.config.types import (
    DecodeBatch,
    DecodeResult,
    ModelConfig,
    ModelRecord,
    PrefillBatch,
    PrefillResult,
    RuntimeConfig,
    RuntimeModel,
)
from pypto_serving.model.common.runner.buffer_set import StaticDeviceTensor, resolve_l3_arg
from pypto_serving.model.common.runner.l3_dispatch import L3DispatchMixin
from pypto_serving.model.common.runner.model_runner import ModelRunner
from pypto_serving.model.qwen3_8_27b.cache import (
    QWEN38_ATTENTION_LAYERS,
    QWEN38_GDN_LAYERS,
)
from pypto_serving.model.qwen3_8_27b.npu_runner import (
    Qwen38CompiledKernels,
    Qwen38Layout,
    QWEN38_HIDDEN_SIZE,
    QWEN38_VOCAB_SIZE,
    resident_view,
)
from pypto_serving.model.qwen3_8_27b.task_args import (
    DECODE_ARGS,
    PREFILL_ARGS,
    decode_task_args,
    embedding_task_args,
    final_head_task_args,
    prefill_task_args,
)
from pypto_serving.model.qwen3_8_27b.weight_spec import (
    QWEN38_COMMON_RULES,
    QWEN38_GDN_RULES,
)


logger = logging.getLogger(__name__)
_GDN_COUNT = len(QWEN38_GDN_LAYERS)
_COMMON_NAMES = frozenset(rule.name for rule in QWEN38_COMMON_RULES)
_GDN_NAMES = frozenset(rule.name for rule in QWEN38_GDN_RULES)
_TABLE_NAMES = frozenset(("tril", "mask_strict", "eye", "m_diag", "m_low", "mask_bias", "tail_mask"))


@dataclass
class _HybridCache:
    k_pages: DeviceTensor
    v_pages: DeviceTensor
    conv_state: DeviceTensor
    final_state: DeviceTensor


def _rotary_tables(positions: torch.Tensor, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Match the text-only partial RoPE table in pypto-lib's reference."""
    exponent = torch.arange(0, 64, 2, dtype=torch.float64) / 64
    inv_freq = 1.0 / (theta ** exponent)
    angles = positions.to(torch.float64)[:, None] * inv_freq[None, :]
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos().float(), angles.sin().float()


def _static_tables() -> dict[str, torch.Tensor]:
    """Build the small masks accepted by the tested PyPTO kernel fixtures."""
    rows = torch.arange(128)
    block = rows // 16
    bias_rows = torch.arange(64)[:, None]
    bias_cols = torch.arange(448)[None, :]
    tail_cols = rows[None, :]
    tail_valid = torch.arange(129)[:, None]
    return {
        "tril": torch.tril(torch.ones(128, 128, dtype=torch.float32)),
        "mask_strict": (rows[:, None] > rows[None, :]).float(),
        "eye": torch.eye(128, dtype=torch.float16),
        "m_diag": -(block[:, None] == block[None, :]).to(torch.float16),
        "m_low": -(block[:, None] > block[None, :]).to(torch.float16),
        "mask_bias": torch.where(bias_cols <= bias_rows + 192, 0.0, -1.0e30).float(),
        "tail_mask": torch.where(tail_cols < tail_valid, 0.0, -1.0e30).float(),
    }


class Qwen38ModelRunner(L3DispatchMixin, ModelRunner):
    """Own shared weights, physical hybrid caches and request-level dispatch."""

    def __init__(self, *, model_id: str, compiled: Qwen38CompiledKernels, device_id: int = 0) -> None:
        super().__init__()
        self._init_l3_dispatch(stacked=False)
        self.model_id = model_id
        self._compiled = compiled
        self._layout = compiled.layout
        self._device_id = device_id
        self._cache: _HybridCache | None = None
        self._device: dict[str, DeviceTensor] = {}
        self._decode_weight_views: dict[str, Any] = {}
        self._scratch: dict[str, DeviceTensor] = {}
        self._active_values: dict[str, Any] = {}
        self._run_config: Any | None = None
        self._static_host = {**compiled.weights, **compiled.globals, **_static_tables()}
        self._static_host["zero_prefill_residual"] = torch.zeros(
            self._layout.prefill_tokens, QWEN38_HIDDEN_SIZE, dtype=torch.float32
        )
        self._static_host["zero_decode_residual"] = torch.zeros(
            self._layout.decode_batch_pad, QWEN38_HIDDEN_SIZE, dtype=torch.float32
        )
        zero_rows = max(_GDN_COUNT, self._layout.decode_batch_pad)
        self._zero_conv_rows = torch.zeros(zero_rows, 3, 10240, dtype=torch.bfloat16).share_memory_()
        self._zero_state_rows = torch.zeros(
            zero_rows, 48, 128, 128, dtype=torch.float32
        ).share_memory_()
        for name, tensor in self._static_host.items():
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            self._static_host[name] = tensor if tensor.is_shared() else tensor.share_memory_()

        source = lambda name: self._active_values[name]
        self._prefill_tasks = {
            rows: prefill_task_args(self._layout, source, rows)
            for rows in compiled.prefill_by_rows
        }
        self._tasks = {
            "decode": decode_task_args(self._layout, source),
            "embed": embedding_task_args(self._layout, source),
            "head": final_head_task_args(self._layout, source),
        }
        for task in (*self._prefill_tasks.values(), *self._tasks.values()):
            task.allocate_host_shared(self._layout)

    def _shared_l3_worker(self) -> Any:
        worker = self._l3_worker
        if worker is None:
            from pypto.runtime import DistributedWorker  # noqa: PLC0415

            kernels = self._compiled
            inherited = [*self._static_host.values(), self._zero_conv_rows, self._zero_state_rows]
            inherited.extend(tensor for task in (*self._prefill_tasks.values(), *self._tasks.values())
                             for tensor in task.tensors.values())
            worker = DistributedWorker(
                [*(kernel.compiled for kernel in kernels.prefill_by_rows.values()),
                 *(kernel.compiled for kernel in kernels.decode_fwd_by_rows.values()),
                 kernels.decode_embedding.compiled,
                 kernels.final_head.compiled],
                persistent=True,
                reset_persistent_windows=False,
                inherited_host_tensors=inherited,
                config=self._run_config,
            )
            self._l3_worker = worker
        return worker

    def _alloc_kv_cache_tensor(self, shape: tuple[int, ...], dtype: torch.dtype) -> DeviceTensor:
        return self._shared_l3_worker().alloc_tensor(shape, dtype)

    def _free_kv_cache_tensor(self, tensor: DeviceTensor) -> None:
        if self._l3_worker is not None:
            self._l3_worker.free_tensor(tensor)

    def _upload_static(self) -> None:
        worker = self._shared_l3_worker()
        for name, tensor in self._static_host.items():
            self._device[name] = resolve_l3_arg(
                worker, StaticDeviceTensor(tensor), self._l3_static_tensors, stacked=False
            )

    def _alloc_scratch(self) -> None:
        worker = self._shared_l3_worker()
        t = self._layout.prefill_tokens
        b = self._layout.decode_batch_pad
        shapes = {
            "prefill_residual": ((t, QWEN38_HIDDEN_SIZE), torch.float32),
            "prefill_delta": ((t, QWEN38_HIDDEN_SIZE), torch.bfloat16),
            "decode_delta_a": ((b, QWEN38_HIDDEN_SIZE), torch.bfloat16),
            "decode_delta_b": ((b, QWEN38_HIDDEN_SIZE), torch.bfloat16),
            "decode_residual_b": ((b, QWEN38_HIDDEN_SIZE), torch.float32),
        }
        for name, (shape, dtype) in shapes.items():
            self._scratch[name] = worker.alloc_tensor(shape, dtype)

    def init_kv_cache(self, model_id: str, config: ModelConfig, runtime: RuntimeConfig) -> int:
        """Allocate fixed-shape K/V and GDN pools after resident weights and rings."""
        if model_id != self.model_id or config.num_hidden_layers != 64:
            raise ValueError("Qwen3.8 runner received another model")
        if self._cache is not None:
            return self._layout.num_kv_pages
        if runtime.total_kv_pages is not None and runtime.total_kv_pages != self._layout.num_kv_pages:
            raise ValueError("Qwen3.8 compiled pool requires total_kv_pages to match its fixed layout")
        from pypto.runtime import RunConfig  # noqa: PLC0415

        self._run_config = RunConfig(
            ring_heap=runtime.ring_heap or list(self._compiled.prefill_ring_heap),
            ring_task_window=runtime.ring_task_window,
            ring_dep_pool=runtime.ring_dep_pool,
        )
        self._l3_run_config = self._run_config
        worker = self._shared_l3_worker()
        try:
            self._upload_static()
            self._alloc_scratch()
            free_bytes, total_bytes = worker.device_memory_info(0)
            layout = self._layout
            needed = (
                2 * math.prod(layout.kv_pool_shape) * torch.bfloat16.itemsize
                + math.prod(layout.conv_pool_shape) * torch.bfloat16.itemsize
                + math.prod(layout.recurrent_pool_shape) * torch.float32.itemsize
            )
            budget = int(total_bytes * runtime.npu_memory_utilization) - (total_bytes - free_bytes)
            if needed > budget:
                raise RuntimeError(
                    f"Qwen3.8 fixed hybrid cache needs {needed} bytes, but only {budget} bytes "
                    "remain inside npu_memory_utilization; reduce max_num_seqs or max_model_len"
                )
            allocated: list[DeviceTensor] = []
            try:
                for shape, dtype in (
                    (layout.kv_pool_shape, torch.bfloat16),
                    (layout.kv_pool_shape, torch.bfloat16),
                    (layout.conv_pool_shape, torch.bfloat16),
                    (layout.recurrent_pool_shape, torch.float32),
                ):
                    allocated.append(worker.alloc_tensor(shape, dtype))
            except Exception:
                for tensor in allocated:
                    worker.free_tensor(tensor)
                raise
            self._cache = _HybridCache(*allocated)
            self._zero_state_range(layout.prefill_scratch_state_base, _GDN_COUNT)
            self._zero_state_range(layout.decode_scratch_state_row(0), layout.decode_batch_pad)
            logger.info("Qwen3.8 cache: %d attention pages, %d sequence slots",
                        layout.num_kv_pages, layout.capacity_slots)
            return layout.num_kv_pages
        except Exception:
            self.close()
            raise

    def _zero_state_range(self, first: int, rows: int) -> None:
        cache = self._require_cache()
        if first < 0 or rows <= 0 or any(
            rows > zero.shape[0] or first + rows > tensor.shape[0]
            for tensor, zero in ((cache.conv_state, self._zero_conv_rows),
                                 (cache.final_state, self._zero_state_rows))
        ):
            raise ValueError("State reset exceeds the cache or host zero buffer")
        worker = self._shared_l3_worker()
        for tensor, zero in ((cache.conv_state, self._zero_conv_rows),
                             (cache.final_state, self._zero_state_rows)):
            row_bytes = math.prod(tensor.shape[1:]) * tensor.dtype.itemsize
            worker.copy_to(tensor.data_ptr, zero.data_ptr(), rows * row_bytes,
                           dst_offset=first * row_bytes)

    def _require_cache(self) -> _HybridCache:
        if self._cache is None:
            raise RuntimeError("Qwen3.8 hybrid cache is not initialized")
        return self._cache

    def preflight(self, record: ModelRecord) -> None:
        """Make the ready event mean both prefill and decode have dispatched."""
        try:
            self.warmup(record.runtime_model)
        except Exception:
            self.close()
            raise

    def warmup(self, model: RuntimeModel) -> None:
        """Exercise a full prefill chunk and all decode stages on scratch rows."""
        layout = self._layout
        self._zero_state_range(layout.prefill_scratch_state_base, _GDN_COUNT)
        tokens = torch.zeros(layout.prefill_tokens, dtype=torch.int32)
        pages = list(range(layout.prefill_tokens // 128))
        self._run_prefill_chunk(model, [tokens], [0], [pages], [layout.prefill_scratch_state_base])
        self._zero_state_range(layout.decode_scratch_state_row(0), layout.decode_batch_pad)
        self._run_decode_tokens(model, torch.zeros(1, dtype=torch.int32), [1],
                                [[layout.decode_scratch_kv_page(0)]], [0], scratch=True)

    def _full_weight(self, name: str) -> Any:
        slab = self._compiled.weights[name]
        handle = self._device[name]
        count = (64 if name in _COMMON_NAMES else
                 _GDN_COUNT if name in _GDN_NAMES else len(QWEN38_ATTENTION_LAYERS))
        if slab.shape[0] == count:
            return resident_view(handle, 0, (slab.numel(),))
        return handle


    def _run_prefill_chunk(
        self, model: RuntimeModel, tokens: list[torch.Tensor], starts: list[int],
        page_ids: list[list[int]], state_bases: list[int],
    ) -> torch.Tensor:
        """Pack aligned request spans and return one final-token logit row per owner."""
        layout = self._layout
        rows = len(tokens)
        lengths = [span.numel() for span in tokens]
        if (rows not in self._prefill_tasks or len(starts) != rows or len(page_ids) != rows
                or len(state_bases) != rows or sum(lengths) != layout.prefill_tokens
                or any(length <= 0 or length % 256 for length in lengths)
                or any(start < 0 or start % 256 for start in starts)):
            raise ValueError("Qwen3.8 text prefill requires aligned spans filling one complete chunk")
        cache = self._require_cache()
        task = self._prefill_tasks[rows]
        buffers = task.tensors
        buffers["input_ids"].copy_(torch.cat([span.reshape(-1) for span in tokens]).to(torch.int32))
        positions = torch.cat([torch.arange(start, start + length)
                               for start, length in zip(starts, lengths, strict=True)])
        cos, sin = _rotary_tables(positions, model.config.rope_theta)
        buffers["rope_cos"].copy_(cos)
        buffers["rope_sin"].copy_(sin)
        buffers["block_table"].fill_(0)
        base = 0
        for row, (start, length, pages) in enumerate(zip(starts, lengths, page_ids, strict=True)):
            end = base + length
            buffers["row_base"][base:end] = base
            buffers["row_owner"][base:end] = row
            buffers["row_ends"][row] = end
            buffers["page_bases"][row] = start // 128
            buffers["visible_pages"][row] = (start + length) // 128
            buffers["block_table"][row, :len(pages)] = torch.tensor(pages, dtype=torch.int32)
            base = end
        # Kernel ABI is layer-major, while the physical state pool is owner-major.
        buffers["state_rows"].copy_((
            torch.arange(_GDN_COUNT, dtype=torch.int32)[:, None]
            + torch.tensor(state_bases, dtype=torch.int32)[None, :]
        ).reshape(-1))
        buffers["logit_ids"].fill_(layout.prefill_tokens - 1)
        buffers["logit_ids"][:rows].copy_(buffers["row_ends"] - 1)
        values: dict[str, Any] = dict(self._device)
        values.update({
            name: self._full_weight(name) for name in PREFILL_ARGS if name in self._compiled.weights
        })
        values.update({
            "residual": self._device["zero_prefill_residual"],
            "conv_state": cache.conv_state,
            "final_state": cache.final_state,
            "k_pages": cache.k_pages,
            "v_pages": cache.v_pages,
            "residual_out": self._scratch["prefill_residual"],
            "mlp_out": self._scratch["prefill_delta"],
            "valid_rows": rows,
        })
        self._active_values = values
        self._run_l3(self._compiled.prefill_by_rows[rows], *task.build())
        return buffers["logits"][:rows].clone()

    @staticmethod
    def _group_ids(row: dict[str, list[int]], layout: Qwen38Layout, seq_len: int) -> tuple[list[int], int]:
        try:
            attention = row["attn_kv"]
            state = row["gdn_state"]
        except KeyError as exc:
            raise ValueError("Qwen3.8 batch requires attn_kv and gdn_state block IDs") from exc
        if len(state) != 1 or not 0 <= state[0] < layout.capacity_slots:
            raise ValueError("Qwen3.8 request requires exactly one valid GDN state block")
        pages = math.ceil(seq_len / 128)
        if (len(attention) < pages or len(attention) > layout.max_blocks_per_seq
                or any(not 0 <= page < layout.num_kv_pages for page in attention)):
            raise ValueError("Qwen3.8 request lacks valid attention pages for its sequence length")
        return attention, state[0]

    def _run_decode_tokens(
        self, model: RuntimeModel, token_ids: torch.Tensor, seq_lens: list[int],
        attention_pages: list[list[int]], state_blocks: list[int], *, scratch: bool = False,
        emit_logits: bool = True,
    ) -> torch.Tensor | None:
        layout = self._layout
        cache = self._require_cache()
        batch_size = len(seq_lens)
        if not 0 < batch_size <= layout.max_batch_size or token_ids.numel() != batch_size:
            raise ValueError("Qwen3.8 decode batch size or token count is invalid")
        if len(attention_pages) != batch_size or len(state_blocks) != batch_size:
            raise ValueError("Qwen3.8 decode cache metadata does not match its batch")
        pad = layout.decode_batch_pad
        embed = self._tasks["embed"]
        embed.tensors["input_ids"].zero_()
        embed.tensors["input_ids"][:batch_size] = token_ids.to(torch.int32).reshape(-1)
        self._active_values = {
            "embed_weight": self._device["embed_weight"],
            "out": self._scratch["decode_delta_a"],
        }
        self._run_l3(self._compiled.decode_embedding, *embed.build())

        decode = self._tasks["decode"]
        attn = decode.tensors
        request_pages = torch.zeros((pad, layout.max_blocks_per_seq), dtype=torch.int32)
        attn["seq_lens"].fill_(1)
        positions = torch.zeros(pad, dtype=torch.int64)
        for row in range(pad):
            if row < batch_size:
                length = int(seq_lens[row])
                if not 0 < length <= layout.max_seq_len:
                    raise ValueError("Qwen3.8 decode sequence length is outside configured context")
                pages = attention_pages[row]
                if len(pages) < math.ceil(length / 128) or len(pages) > layout.max_blocks_per_seq:
                    raise ValueError("Qwen3.8 decode is missing attention pages")
                request_pages[row, :len(pages)] = torch.tensor(pages, dtype=torch.int32)
                attn["seq_lens"][row] = length
                positions[row] = length - 1
            else:
                request_pages[row, 0] = layout.decode_scratch_kv_page(row)
        cos, sin = _rotary_tables(positions, model.config.rope_theta)
        attn["rope_cos"].copy_(cos)
        attn["rope_sin"].copy_(sin)

        state_rows = attn["state_rows"].view(_GDN_COUNT, pad)
        for local, layer in enumerate(QWEN38_GDN_LAYERS):
            for row in range(pad):
                state_rows[local, row] = (
                    layout.decode_scratch_state_row(row)
                    if scratch or row >= batch_size
                    else layout.state_row(state_blocks[row], layer)
                )
        tables = attn["block_tables"].view(len(QWEN38_ATTENTION_LAYERS), pad, -1)
        for local, layer in enumerate(QWEN38_ATTENTION_LAYERS):
            torch.add(request_pages, layout.attention_layer_base(layer) // 128, out=tables[local])

        if not self._decode_weight_views:
            self._decode_weight_views = {
                name: self._full_weight(name) for name in DECODE_ARGS if name in self._compiled.weights
            }
        residual = self._scratch["decode_residual_b"]
        delta = self._scratch["decode_delta_b"]
        self._active_values = {
            **self._decode_weight_views,
            "residual": self._device["zero_decode_residual"],
            "delta": self._scratch["decode_delta_a"],
            "residual_out": residual, "mlp_out": delta, "group_start": 0,
            "tail_mask": self._device["tail_mask"],
            "conv_state": cache.conv_state, "final_state": cache.final_state,
            "k_pages": cache.k_pages, "v_pages": cache.v_pages,
        }
        kernels = self._compiled.decode_fwd_by_rows
        self._run_l3(kernels.get(batch_size, kernels[pad]), *decode.build())

        # Intermediate prompt tokens only need their recurrent and KV updates.
        # The next input token is already known, so their logits are discarded.
        if not emit_logits:
            return None

        head = self._tasks["head"]
        self._active_values = {
            "residual": residual, "delta": delta,
            "norm_w": self._device["final_norm_w"], "lm_w": self._device["lm_w"],
            "valid_rows": batch_size,
        }
        self._run_l3(self._compiled.final_head, *head.build())
        return head.tensors["logits"][:batch_size, :QWEN38_VOCAB_SIZE].clone()

    def run_prefill(self, model: RuntimeModel, batch: PrefillBatch) -> PrefillResult:
        """Fill fused chunks from aligned spans; decode exact unaligned tokens and tails."""
        count = len(batch.request_ids)
        if not 0 < count <= self._layout.max_batch_size or len(batch.block_ids_by_group) != count:
            raise ValueError("Qwen3.8 prefill needs a nonempty batch with grouped cache IDs")
        if any(len(values) != count for values in (
            batch.chunk_starts, batch.chunk_lens, batch.seq_lens, batch.chunk_offsets,
        )):
            raise ValueError("Qwen3.8 prefill metadata does not match its request count")
        token_ids = batch.token_ids.reshape(-1)
        starts: list[int] = []
        lengths: list[int] = []
        offsets: list[int] = []
        page_ids: list[list[int]] = []
        state_blocks: list[int] = []
        for row in range(count):
            start = batch.chunk_starts[row]
            length = batch.chunk_lens[row]
            seq_len = batch.seq_lens[row]
            offset = batch.chunk_offsets[row]
            if length <= 0 or start < 0 or seq_len != start + length or seq_len > self._layout.max_seq_len:
                raise ValueError("Qwen3.8 prefill chunk positions are inconsistent")
            if offset < 0 or offset + length > token_ids.numel():
                raise ValueError("Qwen3.8 prefill token offset exceeds packed input")
            pages, state_block = self._group_ids(batch.block_ids_by_group[row], self._layout, seq_len)
            starts.append(start)
            lengths.append(length)
            offsets.append(offset)
            page_ids.append(pages)
            state_blocks.append(state_block)

        if len(set(state_blocks)) != count:
            raise ValueError("Qwen3.8 prefill requests must own distinct GDN state blocks")
        page_owners: dict[int, bool] = {}
        for start, length, pages in zip(starts, lengths, page_ids, strict=True):
            visible = pages[:math.ceil((start + length) / 128)]
            if len(set(visible)) != len(visible):
                raise ValueError("Qwen3.8 prefill logical pages must map to distinct physical pages")
            for index, page in enumerate(visible):
                writable = index >= start // 128
                if page in page_owners and (writable or page_owners[page]):
                    raise ValueError("Qwen3.8 prefill writes overlap another request's attention pages")
                page_owners[page] = writable

        for row, start in enumerate(starts):
            if start == 0:
                self._zero_state_range(state_blocks[row] * _GDN_COUNT, _GDN_COUNT)

        consumed = [0] * count
        logits: list[torch.Tensor | None] = [None] * count
        while any(consumed[row] < lengths[row] for row in range(count)):
            packed_rows: list[int] = []
            packed_lengths: list[int] = []
            budget = self._layout.prefill_tokens
            for row in range(count):
                position = starts[row] + consumed[row]
                remaining = lengths[row] - consumed[row]
                length = min(budget, remaining // 256 * 256) if position % 256 == 0 else 0
                if length:
                    packed_rows.append(row)
                    packed_lengths.append(length)
                    budget -= length
                if budget == 0:
                    break
            if budget == 0:
                spans = [token_ids[offsets[row] + consumed[row]:offsets[row] + consumed[row] + length]
                         for row, length in zip(packed_rows, packed_lengths, strict=True)]
                packed_logits = self._run_prefill_chunk(
                    model, spans, [starts[row] + consumed[row] for row in packed_rows],
                    [page_ids[row] for row in packed_rows],
                    [state_blocks[row] * _GDN_COUNT for row in packed_rows],
                )
                for index, (row, length) in enumerate(zip(packed_rows, packed_lengths, strict=True)):
                    logits[row] = packed_logits[index]
                    consumed[row] += length
                continue

            active = [row for row in range(count) if consumed[row] < lengths[row]]
            # Keep aligned spans available while other owners advance to a tile boundary.
            decode_rows = [row for row in active if (starts[row] + consumed[row]) % 256] or active
            if decode_rows:
                tokens = torch.stack([
                    token_ids[offsets[row] + consumed[row]] for row in decode_rows
                ])
                positions = [starts[row] + consumed[row] + 1 for row in decode_rows]
                needs_logits = any(consumed[row] + 1 == lengths[row] for row in decode_rows)
                decoded = self._run_decode_tokens(
                    model, tokens, positions,
                    [page_ids[row] for row in decode_rows],
                    [state_blocks[row] for row in decode_rows],
                    emit_logits=needs_logits,
                )
                for index, row in enumerate(decode_rows):
                    consumed[row] += 1
                    if consumed[row] == lengths[row]:
                        if decoded is None:
                            raise RuntimeError("Qwen3.8 terminal prefill token produced no logits")
                        logits[row] = decoded[index]

        if any(value is None for value in logits):
            raise RuntimeError("Qwen3.8 prefill produced no logits for a request")
        for row, seq_len in enumerate(batch.seq_lens):
            if row < len(batch.kv_allocations):
                batch.kv_allocations[row].tokens_used = max(
                    batch.kv_allocations[row].tokens_used, seq_len
                )
        return PrefillResult(last_hidden=None,
                             logits=torch.stack([value for value in logits if value is not None]))

    def run_decode(self, model: RuntimeModel, batch: DecodeBatch) -> DecodeResult:
        """Embed one token per request, run 64 layers, and score the last stream."""
        count = len(batch.request_ids)
        if not 0 < count <= self._layout.max_batch_size or len(batch.block_ids_by_group) != count:
            raise ValueError("Qwen3.8 decode needs a nonempty batch with grouped cache IDs")
        lengths = [int(value) for value in batch.seq_lens.reshape(-1).tolist()]
        if len(lengths) != count:
            raise ValueError("Qwen3.8 decode token and sequence rows disagree")
        pages, states = zip(*(
            self._group_ids(row, self._layout, length)
            for row, length in zip(batch.block_ids_by_group, lengths, strict=True)
        ), strict=True)
        logits = self._run_decode_tokens(model, batch.token_ids.reshape(-1), lengths,
                                         list(pages), list(states))
        for row, allocation in enumerate(batch.kv_allocations):
            allocation.tokens_used = max(allocation.tokens_used, lengths[row])
        return DecodeResult(hidden_states=None, logits=logits)

    def close(self) -> None:
        """Release physical caches, scratch and static uploads before closing the worker."""
        self._decode_weight_views.clear()
        worker = self._l3_worker
        if worker is None:
            return
        cache = self._cache
        self._cache = None
        if cache is not None:
            for tensor in (cache.k_pages, cache.v_pages, cache.conv_state, cache.final_state):
                worker.free_tensor(tensor)
        for tensor in self._scratch.values():
            worker.free_tensor(tensor)
        self._scratch.clear()
        for tensor in self._device.values():
            worker.free_tensor(tensor)
        self._device.clear()
        self._reset_l3_dispatch()
        worker.close()
        self._l3_worker = None

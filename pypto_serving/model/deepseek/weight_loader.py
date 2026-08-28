# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import ctypes
import hashlib
import json
import logging
import mmap
import os
import struct
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ContextManager

import numpy as np
import torch

from pypto_serving.model.common.weights.store import LazySafetensorsStore, SafeOpenFn, SafeTensorReader

logger = logging.getLogger(__name__)


# The reader/opener shapes are family-neutral and now live with the shared store; the
# private aliases stay so this module's annotations and its callers read unchanged.
_SafeTensorReader = SafeTensorReader
_SafeOpenFn = SafeOpenFn


_GLOBAL_WEIGHT_NAMES = (
    "embed.weight",
    "norm.weight",
    "head.weight",
    "hc_head_fn",
    "hc_head_scale",
    "hc_head_base",
)
_LM_HEAD_VOCAB_CHUNK = 512
_LAYER_COMMON_SUFFIXES = (
    "attn.attn_sink",
    "attn.kv_norm.weight",
    "attn.q_norm.weight",
    "attn.wkv.weight",
    "attn.wo_a.weight",
    "attn.wo_b.weight",
    "attn.wo_b.scale",
    "attn.wq_a.weight",
    "attn.wq_b.weight",
    "attn.wq_b.scale",
    "attn_norm.weight",
    "ffn.gate.weight",
    "ffn.shared_experts.w1.weight",
    "ffn.shared_experts.w1.scale",
    "ffn.shared_experts.w2.weight",
    "ffn.shared_experts.w2.scale",
    "ffn.shared_experts.w3.weight",
    "ffn.shared_experts.w3.scale",
    "ffn_norm.weight",
    "hc_attn_base",
    "hc_attn_fn",
    "hc_attn_scale",
    "hc_ffn_base",
    "hc_ffn_fn",
    "hc_ffn_scale",
)
_LAYER_COMPRESSOR_SUFFIXES = (
    "attn.compressor.ape",
    "attn.compressor.norm.weight",
    "attn.compressor.wgate.weight",
    "attn.compressor.wkv.weight",
)
_LAYER_INDEXER_SUFFIXES = (
    "attn.indexer.compressor.ape",
    "attn.indexer.compressor.norm.weight",
    "attn.indexer.compressor.wgate.weight",
    "attn.indexer.compressor.wkv.weight",
    "attn.indexer.weights_proj.weight",
    "attn.indexer.wq_b.weight",
    "attn.indexer.wq_b.scale",
)
_EXPERT_SUFFIXES = ("w1.weight", "w1.scale", "w2.weight", "w2.scale", "w3.weight", "w3.scale")
_DEEPSEEK_V4_O_GROUPS = 8
_DEEPSEEK_V4_HADAMARD_IDX_DIM = 128
_DEEPSEEK_V4_HCA_COMPRESS_RATIO = 128
_DEEPSEEK_V4_CSA_COMPRESS_RATIO = 4
_DEEPSEEK_V4_HCA_MAIN_OUT_DIM = 512
_DEEPSEEK_V4_CSA_MAIN_OUT_DIM = 1024
_DEEPSEEK_V4_CSA_INNER_OUT_DIM = 256
_DEEPSEEK_V4_HIDDEN_SIZE = 4096
_DEEPSEEK_V4_Q_LORA = 1024
_DEEPSEEK_V4_HEAD_DIM = 512
_DEEPSEEK_V4_ATTENTION_OUT = 64 * 512
_DEEPSEEK_V4_N_ROUTED_EXPERTS = 256
_DEEPSEEK_V4_TOPK = 6
_DEEPSEEK_V4_VOCAB_SIZE = 129280
DEEPSEEK_V4_PACKED_FORMAT = "pypto-deepseek-v4-stacked-v1"
_PREPACKED_CACHE_SAMPLE_WINDOWS = 64
_PREPACKED_CACHE_SAMPLE_BYTES = 4 * 1024 * 1024
_PREPACKED_MIN_CACHE_RESIDENCY = 0.95


_PACKED_NP_DTYPE = {
    "F32": np.float32,
    "I32": np.int32,
    "I8": np.int8,
    "U8": np.uint8,
    # No numpy dtype matches these, so they are read as same-width integers and
    # reinterpreted with torch.Tensor.view below.
    "BF16": np.int16,
    "F16": np.int16,
}
_PACKED_VIEW_DTYPE = {"BF16": torch.bfloat16, "F16": torch.float16}


def _map_shared_prepacked_tensors(fd: int, names: Iterable[str]) -> dict[str, torch.Tensor]:
    """Map the prepacked sidecar shared/read-only and return zero-copy tensors.

    Takes an already-open descriptor rather than a path: the caller validates
    metadata, fingerprint and residency against that same descriptor, and the
    sidecar is published with an atomic ``os.replace``, so re-opening the name
    here could map a different inode than the one that was validated. The
    mapping outlives the descriptor — closing an fd does not unmap it.

    ``safetensors`` has to hand PyTorch a *writable* buffer, so it maps the file
    ``MAP_PRIVATE`` (``rw-p``) — copy-on-write. The resident upload then reads
    those pages from the forked chip children, and the driver must break the
    copy-on-write of every page it pins for the H2D DMA: measured at 0.6 GB/s
    against 7.3 GB/s for the same bytes behind a shared mapping, which made the
    upload 90% of a warm start. A read-only shared mapping has no copy-on-write
    to break, and the file is opened ``O_RDONLY`` so it still cannot be written
    through. The returned tensors keep the mapping alive through the numpy base
    chain, so the caller does not have to hold it.
    """
    header_len = struct.unpack("<Q", os.pread(fd, 8, 0))[0]
    header = json.loads(os.pread(fd, header_len, 8))
    mapping = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)  # MAP_SHARED | PROT_READ

    data_start = 8 + header_len
    tensors: dict[str, torch.Tensor] = {}
    for name in names:
        spec = header[name]
        dtype = spec["dtype"]
        if dtype not in _PACKED_NP_DTYPE:
            raise ValueError(f"DeepSeekV4 packed weight {name} has unsupported dtype {dtype}")
        shape = tuple(int(dim) for dim in spec["shape"])
        count = 1
        for dim in shape:
            count *= dim
        begin, end = spec["data_offsets"]
        array = np.frombuffer(mapping, dtype=_PACKED_NP_DTYPE[dtype], count=count, offset=data_start + begin)
        if array.nbytes != end - begin:
            raise ValueError(
                f"DeepSeekV4 packed weight {name} declares {end - begin} bytes but {dtype}{shape} needs "
                f"{array.nbytes}"
            )
        with warnings.catch_warnings():
            # The mapping is read-only on purpose, so the tensor is non-writable.
            warnings.simplefilter("ignore")
            tensor = torch.from_numpy(array)
        if dtype in _PACKED_VIEW_DTYPE:
            tensor = tensor.view(_PACKED_VIEW_DTYPE[dtype])
        tensors[name] = tensor.reshape(shape)
    return tensors


def _default_safe_open(path: Path, device: str) -> ContextManager[_SafeTensorReader]:
    """Open a safetensors shard without loading unrelated tensors."""
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("safetensors is required to read DeepSeekV4 W8A8 weights.") from exc

    return safe_open(str(path), framework="pt", device=device)


def deepseek_v4_global_weight_names() -> tuple[str, ...]:
    """Return global DeepSeekV4 tensor names needed outside the layer stack."""
    return _GLOBAL_WEIGHT_NAMES


@dataclass(frozen=True)
class DeepSeekV4LmHeadLayout:
    """8-way tensor-parallel LM-head shard layout."""

    ranks: int
    vocab_size: int
    hidden_size: int
    vocab_per_rank: int
    padded_vocab_per_rank: int


@dataclass(frozen=True)
class DeepSeekV4GlobalWeights:
    """Global DeepSeekV4 weights packed for serving kernels."""

    embed_weight: torch.Tensor
    final_norm_weight: torch.Tensor
    lm_head_weight: torch.Tensor
    lm_head_layout: DeepSeekV4LmHeadLayout
    hc_head_fn: torch.Tensor
    hc_head_scale: torch.Tensor
    hc_head_base: torch.Tensor


@dataclass(frozen=True)
class DeepSeekV4PackedLayerWeights:
    """One DeepSeekV4 layer's tensors packed in pypto-lib host argument names."""

    layer_id: int
    tensors: Mapping[str, torch.Tensor]

    def args(self, names: Sequence[str]) -> tuple[torch.Tensor, ...]:
        """Return tensors in a kernel host order."""
        missing = [name for name in names if name not in self.tensors]
        if missing:
            raise KeyError(f"Packed DeepSeekV4 layer is missing tensors: {', '.join(missing)}")
        return tuple(self.tensors[name] for name in names)


# Layer-stacking groups for the packed all-layer ``l3_decode_fwd`` kernel. These
# mirror the name groups in pypto-lib decode_fwd.py, but only cover *loaded*
# weights -- the per-layer work-cache/state tensors (kv_cache, cmp_kv,
# idx_kv_cache, *_compress_state) are owned by the runner work cache and are not
# emitted by the weight loader.
DEEPSEEK_V4_CSA_STACKED_WEIGHT_NAMES = (
    "csa_cmp_wkv",
    "csa_cmp_wgate",
    "csa_cmp_ape",
    "csa_cmp_norm_w",
    "csa_idx_wq_b",
    "csa_idx_wq_b_scale",
    "csa_weights_proj",
    "csa_hadamard_idx",
    "csa_inner_wkv",
    "csa_inner_wgate",
    "csa_inner_ape",
    "csa_inner_norm_w",
)
DEEPSEEK_V4_HCA_STACKED_WEIGHT_NAMES = (
    "hca_cmp_wkv",
    "hca_cmp_wgate",
    "hca_cmp_ape",
    "hca_cmp_norm_w",
)
_DEEPSEEK_V4_RANK_REPLICATED_WEIGHT_NAMES = frozenset(
    {
        "hc_attn_fn",
        "hc_attn_scale",
        "hc_attn_base",
        "attn_norm_w",
        "wq_a",
        "wq_b",
        "wq_b_scale",
        "wkv",
        "gamma_cq",
        "gamma_ckv",
        "attn_sink",
        "wo_a",
        "wo_b",
        "wo_b_scale",
        "hc_ffn_fn",
        "hc_ffn_scale",
        "hc_ffn_base",
        "norm_w",
        "gate_w",
        "shared_w1",
        "shared_w1_scale",
        "shared_w3",
        "shared_w3_scale",
        "shared_w2",
        "shared_w2_scale",
        "gate_bias",
        "tid2eid",
        *DEEPSEEK_V4_CSA_STACKED_WEIGHT_NAMES,
        *DEEPSEEK_V4_HCA_STACKED_WEIGHT_NAMES,
    }
)
_DEEPSEEK_V4_EP_SHARDED_WEIGHT_NAMES = frozenset(
    {
        "routed_w1",
        "routed_w1_scale",
        "routed_w2",
        "routed_w2_scale",
        "routed_w3",
        "routed_w3_scale",
    }
)
_DEEPSEEK_V4_PACKED_WEIGHT_NAMES = (
    _DEEPSEEK_V4_RANK_REPLICATED_WEIGHT_NAMES | _DEEPSEEK_V4_EP_SHARDED_WEIGHT_NAMES
)


def deepseek_v4_packed_weights_path(model_dir: str | Path, *, ranks: int) -> Path:
    """Return the default sidecar path for prepacked hidden-layer weights."""
    return Path(model_dir) / f"pypto-deepseek-v4-stacked-r{int(ranks)}.safetensors"


def _sample_file_page_cache_residency(fd: int, path: Path) -> float | None:
    """Estimate Linux page-cache residency without reading the sampled pages.

    Samples the descriptor the caller will go on to validate and map, so the
    answer cannot describe a sidecar that a concurrent publish has since
    replaced. *path* is only used to name the file in diagnostics.
    """
    mapping: mmap.mmap | None = None
    anchor: ctypes.c_char | None = None
    result: float | None = None
    try:
        size = os.fstat(fd).st_size
        if size <= 0:
            result = 0.0
        else:
            mapping = mmap.mmap(fd, size, access=mmap.ACCESS_COPY)
            anchor = ctypes.c_char.from_buffer(mapping)
            base_address = ctypes.addressof(anchor)
            page_size = mmap.PAGESIZE
            sample_bytes = min(size, _PREPACKED_CACHE_SAMPLE_BYTES)
            if size <= _PREPACKED_CACHE_SAMPLE_WINDOWS * sample_bytes:
                offsets = (0,)
                sample_bytes = size
            else:
                last_offset = size - sample_bytes
                offsets = tuple(
                    ((index * last_offset // (_PREPACKED_CACHE_SAMPLE_WINDOWS - 1)) // page_size)
                    * page_size
                    for index in range(_PREPACKED_CACHE_SAMPLE_WINDOWS)
                )

            mincore = ctypes.CDLL(None, use_errno=True).mincore
            mincore.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte))
            mincore.restype = ctypes.c_int
            resident_pages = 0
            sampled_pages = 0
            for offset in offsets:
                length = min(sample_bytes, size - offset)
                page_count = (length + page_size - 1) // page_size
                residency = (ctypes.c_ubyte * page_count)()
                if mincore(
                    ctypes.c_void_p(base_address + offset),
                    ctypes.c_size_t(length),
                    residency,
                ):
                    error = ctypes.get_errno()
                    raise OSError(error, os.strerror(error), path)
                resident_pages += sum(value & 1 for value in residency)
                sampled_pages += page_count
            result = resident_pages / sampled_pages
    except (AttributeError, BufferError, OSError, ValueError):
        logger.warning("Could not inspect page-cache residency for %s", path, exc_info=True)
    finally:
        anchor = None
        if mapping is not None:
            try:
                mapping.close()
            except (BufferError, OSError):
                logger.warning("Could not close page-cache residency mapping for %s", path, exc_info=True)
                result = None
    return result


@dataclass(frozen=True)
class DeepSeekV4StackedLayerWeights:
    """All hidden-layer weights stacked on the layer axis for ``l3_decode_fwd``.

    Each tensor fuses its layer axis into dim 1: ``[ranks, layer_count*d1, ...]``.
    FWD weights stack across all 43 hidden layers; CSA-group weights stack across
    the 21 compress_ratio==4 layers in order; HCA-group weights stack across the
    20 compress_ratio==128 layers in order.
    """

    tensors: Mapping[str, torch.Tensor]

    def args(self, names: Sequence[str]) -> tuple[torch.Tensor, ...]:
        """Return stacked tensors in a kernel host order."""
        missing = [name for name in names if name not in self.tensors]
        if missing:
            raise KeyError(f"Stacked DeepSeekV4 weights are missing tensors: {', '.join(missing)}")
        return tuple(self.tensors[name] for name in names)


@dataclass(frozen=True)
class DeepSeekV4MtpWeights:
    """Rank-stacked weights consumed by ``l3_mtp_decode_layer``."""

    tensors: Mapping[str, torch.Tensor]

    def args(self, names: Sequence[str]) -> tuple[torch.Tensor, ...]:
        """Return MTP tensors in kernel host order."""
        missing = [name for name in names if name not in self.tensors]
        if missing:
            raise KeyError(f"Packed DeepSeekV4 MTP weights are missing tensors: {', '.join(missing)}")
        return tuple(self.tensors[name] for name in names)


@dataclass(frozen=True)
class DeepSeekV4DSparkWeights:
    """Rank-stacked weights consumed by the DSpark drafter and heads."""

    tensors: Mapping[str, torch.Tensor]

    def args(self, names: Sequence[str]) -> tuple[torch.Tensor, ...]:
        """Return DSpark tensors in kernel host order."""
        missing = [name for name in names if name not in self.tensors]
        if missing:
            raise KeyError(f"Packed DeepSeekV4 DSpark weights are missing tensors: {', '.join(missing)}")
        return tuple(self.tensors[name] for name in names)


def deepseek_v4_lm_head_layout(
    *,
    vocab_size: int,
    hidden_size: int,
    ranks: int,
    vocab_chunk: int = _LM_HEAD_VOCAB_CHUNK,
) -> DeepSeekV4LmHeadLayout:
    """Return the LM-head shard shape expected by ``lm_head.py``."""
    if ranks <= 0:
        raise ValueError("ranks must be positive")
    if vocab_chunk <= 0:
        raise ValueError("vocab_chunk must be positive")
    if vocab_size % ranks != 0:
        raise ValueError(f"vocab_size={vocab_size} must divide evenly across ranks={ranks}")
    vocab_per_rank = vocab_size // ranks
    padded_vocab_per_rank = ((vocab_per_rank + vocab_chunk - 1) // vocab_chunk) * vocab_chunk
    return DeepSeekV4LmHeadLayout(
        ranks=ranks,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        vocab_per_rank=vocab_per_rank,
        padded_vocab_per_rank=padded_vocab_per_rank,
    )


def pack_deepseek_v4_lm_head_weight(
    weight: torch.Tensor,
    *,
    ranks: int,
    vocab_chunk: int = _LM_HEAD_VOCAB_CHUNK,
) -> tuple[torch.Tensor, DeepSeekV4LmHeadLayout]:
    """Pack flat ``head.weight`` into contiguous TP vocab shards."""
    if weight.ndim != 2:
        raise ValueError(f"lm_head weight must be rank-2, got shape={tuple(weight.shape)}")
    vocab_size, hidden_size = (int(dim) for dim in weight.shape)
    layout = deepseek_v4_lm_head_layout(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        ranks=ranks,
        vocab_chunk=vocab_chunk,
    )
    packed = torch.zeros(
        (layout.ranks, layout.padded_vocab_per_rank, layout.hidden_size),
        dtype=weight.dtype,
        device=weight.device,
    )
    for rank in range(layout.ranks):
        start = rank * layout.vocab_per_rank
        end = start + layout.vocab_per_rank
        packed[rank, : layout.vocab_per_rank].copy_(weight[start:end])
    return packed.contiguous(), layout


def _attention_suffixes_for_compress_ratio(compress_ratio: int) -> tuple[str, ...]:
    """Return attention parameter suffixes required by one layer attention mode."""
    if compress_ratio == 0:
        return ()
    if compress_ratio == 128:
        return _LAYER_COMPRESSOR_SUFFIXES
    if compress_ratio == 4:
        return (*_LAYER_COMPRESSOR_SUFFIXES, *_LAYER_INDEXER_SUFFIXES)
    raise ValueError(f"unsupported DeepSeekV4 attention compress ratio: {compress_ratio}")


def deepseek_v4_layer_core_weight_names(
    layer_id: int,
    *,
    compress_ratio: int = 0,
    include_tid2eid: bool = False,
    include_gate_bias: bool = False,
) -> tuple[str, ...]:
    """Return non-routed-expert tensor names for one DeepSeekV4 layer."""
    prefix = f"layers.{int(layer_id)}"
    suffixes = [*_LAYER_COMMON_SUFFIXES, *_attention_suffixes_for_compress_ratio(compress_ratio)]
    if include_tid2eid:
        suffixes.append("ffn.gate.tid2eid")
    if include_gate_bias:
        suffixes.append("ffn.gate.bias")
    return tuple(f"{prefix}.{suffix}" for suffix in suffixes)


def deepseek_v4_routed_expert_weight_names(layer_id: int, expert_ids: Iterable[int]) -> tuple[str, ...]:
    """Return routed expert tensor names for one DeepSeekV4 layer."""
    names: list[str] = []
    for expert_id in expert_ids:
        prefix = f"layers.{int(layer_id)}.ffn.experts.{int(expert_id)}"
        names.extend(f"{prefix}.{suffix}" for suffix in _EXPERT_SUFFIXES)
    return tuple(names)


def deepseek_v4_local_expert_ids(*, rank: int, ranks: int, n_routed_experts: int) -> tuple[int, ...]:
    """Return the contiguous routed-expert ids owned by one EP rank."""
    if ranks <= 0:
        raise ValueError("ranks must be positive")
    if not 0 <= rank < ranks:
        raise ValueError(f"rank must be in [0, {ranks - 1}], got {rank}")
    if n_routed_experts <= 0:
        raise ValueError("n_routed_experts must be positive")
    if n_routed_experts % ranks != 0:
        raise ValueError(f"n_routed_experts={n_routed_experts} must divide evenly across ranks={ranks}")
    local_count = n_routed_experts // ranks
    start = rank * local_count
    return tuple(range(start, start + local_count))


def deepseek_v4_hadamard_idx(dim: int = _DEEPSEEK_V4_HADAMARD_IDX_DIM) -> torch.Tensor:
    """Return the normalized Hadamard matrix used by the CSA indexer."""
    if dim <= 0 or dim & (dim - 1) != 0:
        raise ValueError("Hadamard dimension must be a positive power of two")
    h = torch.ones((1, 1), dtype=torch.bfloat16)
    while h.shape[0] < dim:
        h = torch.cat(
            [torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)],
            dim=0,
        )
    return (h * (dim**-0.5)).contiguous()


def deepseek_v4_layer_weight_names(
    layer_id: int,
    *,
    n_routed_experts: int,
    compress_ratio: int = 0,
    include_tid2eid: bool = False,
    include_gate_bias: bool = False,
    expert_ids: Iterable[int] | None = None,
) -> tuple[str, ...]:
    """Return all tensor names needed to execute one DeepSeekV4 layer."""
    if n_routed_experts <= 0:
        raise ValueError("n_routed_experts must be positive")
    expert_ids = range(n_routed_experts) if expert_ids is None else tuple(expert_ids)
    return (
        *deepseek_v4_layer_core_weight_names(
            layer_id,
            compress_ratio=compress_ratio,
            include_tid2eid=include_tid2eid,
            include_gate_bias=include_gate_bias,
        ),
        *deepseek_v4_routed_expert_weight_names(layer_id, expert_ids),
    )


def deepseek_v4_startup_weight_names(
    num_hidden_layers: int,
    *,
    n_routed_experts: int,
    compress_ratios: Sequence[int] | None = None,
    num_hash_layers: int = 3,
) -> tuple[str, ...]:
    """Return tensor names used for metadata-only checkpoint contract validation.

    Startup checks every layer's core tensors plus the first and last routed
    expert in each layer. Full expert materialization remains an explicit
    per-layer load so serving startup does not read shard payloads.
    """
    if num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be positive")
    if n_routed_experts <= 0:
        raise ValueError("n_routed_experts must be positive")
    if compress_ratios is None:
        compress_ratios = (0,) * num_hidden_layers
    if len(compress_ratios) < num_hidden_layers:
        raise ValueError("compress_ratios must include at least one entry per hidden layer")

    edge_experts = tuple(dict.fromkeys((0, n_routed_experts - 1)))
    names = list(_GLOBAL_WEIGHT_NAMES)
    for layer_id in range(num_hidden_layers):
        names.extend(
            deepseek_v4_layer_core_weight_names(
                layer_id,
                compress_ratio=int(compress_ratios[layer_id]),
                include_tid2eid=layer_id < num_hash_layers,
                include_gate_bias=layer_id >= num_hash_layers,
            )
        )
        names.extend(deepseek_v4_routed_expert_weight_names(layer_id, edge_experts))
    return tuple(dict.fromkeys(names))


def deepseek_v4_mtp_startup_weight_names(n_routed_experts: int) -> tuple[str, ...]:
    """Return the lightweight startup contract for the single MTP draft layer."""
    edge_experts = tuple(dict.fromkeys((0, n_routed_experts - 1)))
    layer_names = [
        name.replace("layers.0", "mtp.0", 1)
        for name in deepseek_v4_layer_weight_names(
            0,
            n_routed_experts=n_routed_experts,
            compress_ratio=0,
            include_gate_bias=True,
            expert_ids=edge_experts,
        )
    ]
    projection_and_head = (
        "mtp.0.enorm.weight",
        "mtp.0.hnorm.weight",
        "mtp.0.e_proj.weight",
        "mtp.0.e_proj.scale",
        "mtp.0.h_proj.weight",
        "mtp.0.h_proj.scale",
        "mtp.0.hc_head_fn",
        "mtp.0.hc_head_scale",
        "mtp.0.hc_head_base",
        "mtp.0.norm.weight",
    )
    return tuple(dict.fromkeys((*layer_names, *projection_and_head)))


def deepseek_v4_dspark_startup_weight_names(
    n_routed_experts: int,
    *,
    stages: int = 3,
) -> tuple[str, ...]:
    """Return the metadata contract for the three-layer DSpark model."""
    if stages <= 0:
        raise ValueError("DSpark stages must be positive")
    edge_experts = tuple(dict.fromkeys((0, n_routed_experts - 1)))
    names: list[str] = []
    for stage in range(stages):
        names.extend(
            name.replace("layers.0", f"mtp.{stage}", 1)
            for name in deepseek_v4_layer_weight_names(
                0,
                n_routed_experts=n_routed_experts,
                compress_ratio=0,
                include_gate_bias=True,
                expert_ids=edge_experts,
            )
        )
    final_prefix = f"mtp.{stages - 1}"
    names.extend(
        (
            "mtp.0.main_proj.weight",
            "mtp.0.main_norm.weight",
            f"{final_prefix}.norm.weight",
            f"{final_prefix}.hc_head_fn",
            f"{final_prefix}.hc_head_scale",
            f"{final_prefix}.hc_head_base",
            f"{final_prefix}.markov_head.markov_w1.weight",
            f"{final_prefix}.markov_head.markov_w2.weight",
            f"{final_prefix}.confidence_head.weight",
        )
    )
    return tuple(dict.fromkeys(names))


class DeepSeekV4WeightStore(LazySafetensorsStore):
    """Lazy name-based safetensors access for DeepSeekV4 W8A8 checkpoints.

    The index handling and the grouped reads come from
    :class:`~pypto_serving.model.common.weights.store.LazySafetensorsStore`; what stays
    here is the DeepSeekV4 contract — which names a checkpoint must expose, and how the
    global and per-layer tensors are packed for the fused kernels.
    """

    missing_name_error = "Missing DeepSeekV4 weight tensor in index: {name}"
    missing_names_error = "DeepSeekV4 W8A8 checkpoint is missing required tensors: {names}"
    missing_shard_error = "Missing safetensors shard for DeepSeekV4 weight load: {path}"

    def _default_open_fn(self) -> SafeOpenFn:
        # Resolved through this module's global on purpose: the opener carries the
        # DeepSeekV4-specific import diagnostic, and tests patch it by name here.
        return _default_safe_open

    def validate_startup_contract(
        self,
        *,
        num_hidden_layers: int,
        n_routed_experts: int,
        compress_ratios: Sequence[int] | None = None,
        num_hash_layers: int = 3,
    ) -> None:
        """Validate the startup-visible checkpoint contract without opening shards."""
        self.require(
            deepseek_v4_startup_weight_names(
                num_hidden_layers,
                n_routed_experts=n_routed_experts,
                compress_ratios=compress_ratios,
                num_hash_layers=num_hash_layers,
            )
        )

    def validate_mtp_startup_contract(self, *, n_routed_experts: int) -> None:
        """Validate MTP metadata without opening checkpoint shards."""
        self.require(deepseek_v4_mtp_startup_weight_names(n_routed_experts))

    def validate_dspark_startup_contract(
        self,
        *,
        n_routed_experts: int,
        stages: int = 3,
    ) -> None:
        """Validate DSpark metadata without opening checkpoint shards."""
        self.require(
            deepseek_v4_dspark_startup_weight_names(
                n_routed_experts,
                stages=stages,
            )
        )

    def load_global_weights(self) -> dict[str, torch.Tensor]:
        """Load embedding, final norm, and LM head tensors."""
        return self.load_many(deepseek_v4_global_weight_names())

    def load_packed_global_weights(self, *, ranks: int) -> DeepSeekV4GlobalWeights:
        """Load and pack global tensors for the DeepSeekV4 serving kernels."""
        weights = self.load_global_weights()
        packed_lm_head, layout = pack_deepseek_v4_lm_head_weight(weights["head.weight"], ranks=ranks)
        if weights["embed.weight"].ndim != 2:
            raise ValueError(f"embed.weight must be rank-2, got shape={tuple(weights['embed.weight'].shape)}")
        if weights["norm.weight"].ndim != 1:
            raise ValueError(f"norm.weight must be rank-1, got shape={tuple(weights['norm.weight'].shape)}")
        if tuple(weights["embed.weight"].shape) != (layout.vocab_size, layout.hidden_size):
            raise ValueError(
                "embed.weight shape must match head.weight shape, "
                f"got embed={tuple(weights['embed.weight'].shape)}, head={tuple(weights['head.weight'].shape)}"
            )
        if int(weights["norm.weight"].shape[0]) != layout.hidden_size:
            raise ValueError(
                f"norm.weight hidden size must be {layout.hidden_size}, "
                f"got {int(weights['norm.weight'].shape[0])}"
            )
        if tuple(weights["hc_head_fn"].shape) != (4, layout.hidden_size * 4):
            raise ValueError(f"hc_head_fn has unsupported shape {tuple(weights['hc_head_fn'].shape)}")
        if tuple(weights["hc_head_scale"].shape) != (1,):
            raise ValueError(f"hc_head_scale has unsupported shape {tuple(weights['hc_head_scale'].shape)}")
        if tuple(weights["hc_head_base"].shape) != (4,):
            raise ValueError(f"hc_head_base has unsupported shape {tuple(weights['hc_head_base'].shape)}")
        return DeepSeekV4GlobalWeights(
            embed_weight=weights["embed.weight"],
            final_norm_weight=weights["norm.weight"],
            lm_head_weight=packed_lm_head,
            lm_head_layout=layout,
            hc_head_fn=weights["hc_head_fn"].to(torch.float32).contiguous().cpu(),
            hc_head_scale=weights["hc_head_scale"].to(torch.float32).contiguous().cpu(),
            hc_head_base=weights["hc_head_base"].to(torch.float32).contiguous().cpu(),
        )

    def load_layer_weights(
        self,
        layer_id: int,
        *,
        n_routed_experts: int,
        compress_ratio: int = 0,
        include_tid2eid: bool = False,
        include_gate_bias: bool = False,
        expert_ids: Iterable[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Load all tensors needed for one DeepSeekV4 layer."""
        return self.load_many(
            deepseek_v4_layer_weight_names(
                layer_id,
                n_routed_experts=n_routed_experts,
                compress_ratio=compress_ratio,
                include_tid2eid=include_tid2eid,
                include_gate_bias=include_gate_bias,
                expert_ids=expert_ids,
            )
        )

    def load_rank_layer_weights(
        self,
        layer_id: int,
        *,
        rank: int,
        ranks: int,
        n_routed_experts: int,
        compress_ratio: int = 0,
        include_tid2eid: bool = False,
        include_gate_bias: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Load common layer tensors plus the routed experts owned by one rank."""
        local_experts = deepseek_v4_local_expert_ids(
            rank=rank,
            ranks=ranks,
            n_routed_experts=n_routed_experts,
        )
        return self.load_layer_weights(
            layer_id,
            n_routed_experts=n_routed_experts,
            compress_ratio=compress_ratio,
            include_tid2eid=include_tid2eid,
            include_gate_bias=include_gate_bias,
            expert_ids=local_experts,
        )

    def load_packed_layer_weights(
        self,
        layer_id: int,
        *,
        ranks: int,
        n_routed_experts: int,
        compress_ratio: int = 0,
        include_tid2eid: bool = False,
        include_gate_bias: bool = False,
        destinations: Mapping[str, torch.Tensor] | None = None,
        o_projection_tp_size: int | None = None,
    ) -> DeepSeekV4PackedLayerWeights:
        """Load and pack one layer into the tensor names expected by pypto-lib kernels.

        When ``destinations`` is provided, packing writes directly into those
        final-layout tensor views instead of allocating rank-expanded outputs.
        """
        all_experts = range(n_routed_experts)
        raw = self.load_layer_weights(
            layer_id,
            n_routed_experts=n_routed_experts,
            compress_ratio=compress_ratio,
            include_tid2eid=include_tid2eid,
            include_gate_bias=include_gate_bias,
            expert_ids=all_experts,
        )
        return pack_deepseek_v4_layer_weights(
            layer_id,
            raw,
            ranks=ranks,
            n_routed_experts=n_routed_experts,
            compress_ratio=compress_ratio,
            include_tid2eid=include_tid2eid,
            include_gate_bias=include_gate_bias,
            destinations=destinations,
            o_projection_tp_size=o_projection_tp_size,
        )

    def packed_stacked_layer_weights_fingerprint(
        self,
        *,
        ranks: int,
        n_routed_experts: int,
        compress_ratios: Sequence[int],
        num_hash_layers: int,
    ) -> str:
        """Return the source/deployment fingerprint for a packed-weight sidecar."""
        source_files = []
        for filename in sorted(set(self.weight_map.values())):
            stat = (self.model_dir / filename).stat()
            source_files.append((filename, stat.st_size, stat.st_mtime_ns))
        payload = {
            "format": DEEPSEEK_V4_PACKED_FORMAT,
            "ranks": int(ranks),
            "n_routed_experts": int(n_routed_experts),
            "compress_ratios": [int(value) for value in compress_ratios],
            "num_hash_layers": int(num_hash_layers),
            "weight_map": sorted(self.weight_map.items()),
            "source_files": source_files,
        }
        encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()

    def load_prepacked_stacked_layer_weights(
        self,
        *,
        ranks: int,
        n_routed_experts: int,
        compress_ratios: Sequence[int],
        num_hash_layers: int,
        path: str | Path | None = None,
    ) -> DeepSeekV4StackedLayerWeights | None:
        """Map a valid prepacked sidecar, or return ``None`` when none is usable."""
        packed_path = (
            deepseek_v4_packed_weights_path(self.model_dir, ranks=ranks)
            if path is None
            else Path(path)
        )
        if not packed_path.is_file():
            return None
        # One descriptor for residency, validation and mapping. The sidecar is
        # published with an atomic os.replace, so re-opening the name between
        # those steps could validate one inode and then map another; /proc/self/fd
        # keeps every step on the instance opened here. The mapping outlives the
        # descriptor, so it is closed as soon as the tensors exist.
        fd = os.open(packed_path, os.O_RDONLY)
        try:
            cache_residency = _sample_file_page_cache_residency(fd, packed_path)
            if cache_residency is None or cache_residency < _PREPACKED_MIN_CACHE_RESIDENCY:
                logger.info(
                    "Skipping cold DeepSeekV4 packed weights sidecar %s "
                    "(sampled page-cache residency %.1f%%; requires %.1f%%)",
                    packed_path,
                    0.0 if cache_residency is None else 100.0 * cache_residency,
                    100.0 * _PREPACKED_MIN_CACHE_RESIDENCY,
                )
                return None
            try:
                from safetensors import safe_open
            except ImportError as exc:
                raise RuntimeError("safetensors is required to read DeepSeekV4 W8A8 weights.") from exc

            expected_fingerprint = self.packed_stacked_layer_weights_fingerprint(
                ranks=ranks,
                n_routed_experts=n_routed_experts,
                compress_ratios=compress_ratios,
                num_hash_layers=num_hash_layers,
            )
            with safe_open(f"/proc/self/fd/{fd}", framework="pt", device="cpu") as reader:
                metadata = reader.metadata() or {}
                if (
                    metadata.get("format") != DEEPSEEK_V4_PACKED_FORMAT
                    or metadata.get("source_fingerprint") != expected_fingerprint
                ):
                    logger.warning(
                        "Ignoring stale DeepSeekV4 packed weights sidecar: %s",
                        packed_path,
                    )
                    return None
                names = frozenset(reader.keys())
                if names != _DEEPSEEK_V4_PACKED_WEIGHT_NAMES:
                    missing = sorted(_DEEPSEEK_V4_PACKED_WEIGHT_NAMES - names)
                    extra = sorted(names - _DEEPSEEK_V4_PACKED_WEIGHT_NAMES)
                    logger.warning(
                        "Ignoring DeepSeekV4 packed weights sidecar %s with invalid tensor names; "
                        "missing=%s, extra=%s",
                        packed_path,
                        missing,
                        extra,
                    )
                    return None
            # Read the payload through our own shared mapping rather than the
            # safetensors reader; see _map_shared_prepacked_tensors for why.
            tensors = _map_shared_prepacked_tensors(fd, names)

            # Deliberately strict from here on, unlike the `return None` paths above: a
            # missing, cold, stale or wrong-named sidecar is an expected miss that falls
            # back to packing from the shards, but one whose format and fingerprint match
            # while its payload does not describe [ranks, ...] tensors is corrupt (or was
            # written by a buggy packer). Failing loudly beats silently taking the slow
            # path and leaving the bad artifact in place for every later start.
            for name, tensor in tensors.items():
                if tensor.device.type != "cpu" or not tensor.is_contiguous():
                    raise ValueError(
                        f"DeepSeekV4 packed weight {name} must be a contiguous CPU tensor, "
                        f"got device={tensor.device} shape={tuple(tensor.shape)}"
                    )
                if tensor.ndim < 2 or int(tensor.shape[0]) != int(ranks):
                    raise ValueError(
                        f"DeepSeekV4 packed weight {name} must have leading rank dimension {ranks}, "
                        f"got shape={tuple(tensor.shape)}"
                    )
            logger.info("Mapped prepacked DeepSeekV4 layer weights from %s", packed_path)
            return DeepSeekV4StackedLayerWeights(tensors=tensors)
        finally:
            os.close(fd)

    def load_stacked_layer_weights(
        self,
        *,
        ranks: int,
        n_routed_experts: int,
        compress_ratios: Sequence[int],
        num_hash_layers: int,
        use_prepacked: bool = True,
        o_projection_tp_size: int | None = None,
    ) -> DeepSeekV4StackedLayerWeights:
        """Load every hidden layer once and stack weights on the layer axis.

        FWD weights are concatenated across all hidden layers in order; CSA-group
        weights across the compress_ratio==4 layers in order; HCA-group weights
        across the compress_ratio==128 layers in order. Each per-layer tensor is
        ``[ranks, d1, ...]`` and stacking concatenates on dim 1.

        Layers are packed serially. A thread pool was tried but regressed: pack
        is a mixed IO+CPU workload and per-layer packing allocates ~8 GB of
        intermediate tensors (256 routed experts each ``torch.stack``-ed and
        rank-replicated), so N parallel layers multiply peak allocation and
        contend on CPU memory bandwidth; the GIL-switch cost also exceeded the
        parallel gain when workers <= layer count. Serial packing keeps the
        working set to one layer at a time and lets the disk prefetcher run.
        """
        num_hidden_layers = len(compress_ratios)
        if num_hidden_layers <= 0:
            raise ValueError("compress_ratios must include at least one entry per hidden layer")
        if use_prepacked and o_projection_tp_size is None:
            prepacked = self.load_prepacked_stacked_layer_weights(
                ranks=ranks,
                n_routed_experts=n_routed_experts,
                compress_ratios=compress_ratios,
                num_hash_layers=num_hash_layers,
            )
            if prepacked is not None:
                return prepacked
        first = self.load_packed_layer_weights(
            0,
            ranks=ranks,
            n_routed_experts=n_routed_experts,
            compress_ratio=int(compress_ratios[0]),
            include_tid2eid=num_hash_layers > 0,
            include_gate_bias=num_hash_layers <= 0,
            o_projection_tp_size=o_projection_tp_size,
        )
        from pypto_serving.model.common.weights.stacker import stack_layers  # noqa: PLC0415

        from .weight_spec import (  # noqa: PLC0415
            DEEPSEEK_V4_RANK_ERROR,
            DEEPSEEK_V4_STACK_MISMATCH_ERROR,
            DEEPSEEK_V4_STAGING_POLICY,
            deepseek_v4_stack_groups,
        )


        def pack_into(layer_id: int, destinations: Mapping[str, torch.Tensor]) -> None:
            self.load_packed_layer_weights(
                layer_id,
                ranks=ranks,
                n_routed_experts=n_routed_experts,
                compress_ratio=int(compress_ratios[layer_id]),
                include_tid2eid=layer_id < num_hash_layers,
                include_gate_bias=layer_id >= num_hash_layers,
                destinations=destinations,
                o_projection_tp_size=o_projection_tp_size,
            )

        def log_progress(layer_id: int) -> None:
            # Every fifth layer plus the last, and layer 0 included: it takes the template path
            # rather than pack_into, so logging there would drop the first line of progress.
            if layer_id % 5 == 0 or layer_id == num_hidden_layers - 1:
                logger.info(
                    "DeepSeekV4 weight load progress: layer %d/%d",
                    layer_id + 1,
                    num_hidden_layers,
                )

        stacked = stack_layers(
            deepseek_v4_stack_groups(compress_ratios),
            first.tensors,
            layer_ids=range(num_hidden_layers),
            pack_into=pack_into,
            template_layer_id=0,
            on_layer_done=log_progress,
            policy=DEEPSEEK_V4_STAGING_POLICY,
            rank_error=DEEPSEEK_V4_RANK_ERROR,
            mismatch_error=DEEPSEEK_V4_STACK_MISMATCH_ERROR,
        )
        return DeepSeekV4StackedLayerWeights(tensors=stacked)

    def load_mtp_weights(
        self,
        *,
        ranks: int,
        n_routed_experts: int,
    ) -> DeepSeekV4MtpWeights:
        """Load and pack the checkpoint's single ``mtp.0`` draft layer."""
        prefix = "mtp.0"
        layer_names = [
            name.replace("layers.0", prefix, 1)
            for name in deepseek_v4_layer_weight_names(
                0,
                n_routed_experts=n_routed_experts,
                compress_ratio=0,
                include_gate_bias=True,
            )
        ]
        extra_names = (
            f"{prefix}.enorm.weight",
            f"{prefix}.hnorm.weight",
            f"{prefix}.e_proj.weight",
            f"{prefix}.e_proj.scale",
            f"{prefix}.h_proj.weight",
            f"{prefix}.h_proj.scale",
            f"{prefix}.hc_head_fn",
            f"{prefix}.hc_head_scale",
            f"{prefix}.hc_head_base",
            f"{prefix}.norm.weight",
        )
        raw = self.load_many((*layer_names, *extra_names))
        packed_layer = pack_deepseek_v4_layer_weights(
            0,
            raw,
            ranks=ranks,
            n_routed_experts=n_routed_experts,
            compress_ratio=0,
            include_tid2eid=False,
            include_gate_bias=True,
            prefix=prefix,
        )

        from pypto_serving.model.common.weights.packer import pack_layer  # noqa: PLC0415
        from pypto_serving.model.common.weights.spec import LayerContext  # noqa: PLC0415

        from .weight_spec import (  # noqa: PLC0415
            DEEPSEEK_V4_MTP_EXTRA_RULES,
            DEEPSEEK_V4_SOURCE_MISSING_ERROR,
            deepseek_v4_factories,
            deepseek_v4_replicate,
        )

        # The draft layer's own weights, through the same evaluator: `LayerContext` already
        # carries the prefix, so `mtp.0` needs no special case beyond this table.
        extras = pack_layer(
            DEEPSEEK_V4_MTP_EXTRA_RULES,
            raw,
            LayerContext(layer_id=0, prefix=prefix, ranks=int(ranks)),
            policy=deepseek_v4_replicate(int(ranks)),
            factories=deepseek_v4_factories(),
            missing_source_error=DEEPSEEK_V4_SOURCE_MISSING_ERROR,
        )
        return DeepSeekV4MtpWeights(tensors={**packed_layer.tensors, **extras})

    def load_dspark_weights(
        self,
        *,
        ranks: int,
        tp_size: int,
        n_routed_experts: int,
        stages: int = 3,
    ) -> DeepSeekV4DSparkWeights:
        """Load and pack the three DSpark draft layers and their shared heads."""
        if stages <= 0:
            raise ValueError("DSpark stages must be positive")

        from pypto_serving.model.common.weights.packer import pack_layer  # noqa: PLC0415
        from pypto_serving.model.common.weights.pipeline import StagingPolicy  # noqa: PLC0415
        from pypto_serving.model.common.weights.spec import LayerContext  # noqa: PLC0415
        from pypto_serving.model.common.weights.stacker import StackGroup, stack_layers  # noqa: PLC0415

        from .weight_spec import (  # noqa: PLC0415
            DEEPSEEK_V4_DSPARK_LAYER_RULES,
            DEEPSEEK_V4_EXPERT_MISSING_ERROR,
            DEEPSEEK_V4_SOURCE_MISSING_ERROR,
            deepseek_v4_expert_parallel,
            deepseek_v4_o_projection_tp_policies,
            deepseek_v4_replicate,
        )

        replicate = deepseek_v4_replicate(int(ranks))
        expert_policy = deepseek_v4_expert_parallel(int(ranks), int(n_routed_experts))
        tp_policies = deepseek_v4_o_projection_tp_policies(int(ranks), int(tp_size))

        def load_stage(
            stage: int,
            destinations: Mapping[str, torch.Tensor] | None = None,
        ) -> Mapping[str, torch.Tensor]:
            prefix = f"mtp.{stage}"
            names = tuple(
                name.replace("layers.0", prefix, 1)
                for name in deepseek_v4_layer_weight_names(
                    0,
                    n_routed_experts=n_routed_experts,
                    compress_ratio=0,
                    include_gate_bias=True,
                )
            )
            raw = self.load_many(names)
            return pack_layer(
                DEEPSEEK_V4_DSPARK_LAYER_RULES,
                raw,
                LayerContext(
                    layer_id=stage,
                    prefix=prefix,
                    ranks=int(ranks),
                    n_routed_experts=int(n_routed_experts),
                    include_gate_bias=True,
                ),
                policy=replicate,
                expert_policy=expert_policy,
                policies=tp_policies,
                destinations=destinations,
                missing_source_error=DEEPSEEK_V4_SOURCE_MISSING_ERROR,
                missing_expert_error=DEEPSEEK_V4_EXPERT_MISSING_ERROR,
            )

        first = load_stage(0)
        stacked = stack_layers(
            (StackGroup(id="dspark", members=None, layer_ids=tuple(range(stages))),),
            first,
            layer_ids=range(stages),
            pack_into=lambda stage, destinations: load_stage(stage, destinations),
            template_layer_id=0,
            policy=StagingPolicy(workers=1),
        )
        stacked["ffn_norm_w"] = stacked.pop("norm_w")

        final_prefix = f"mtp.{stages - 1}"
        head_sources = {
            "main_proj_weight": ("mtp.0.main_proj.weight", torch.bfloat16),
            "main_norm_weight": ("mtp.0.main_norm.weight", torch.bfloat16),
            "final_norm_weight": (f"{final_prefix}.norm.weight", torch.bfloat16),
            "hc_head_fn": (f"{final_prefix}.hc_head_fn", torch.float32),
            "hc_head_scale": (f"{final_prefix}.hc_head_scale", torch.float32),
            "hc_head_base": (f"{final_prefix}.hc_head_base", torch.float32),
            "markov_w1": (
                f"{final_prefix}.markov_head.markov_w1.weight",
                torch.bfloat16,
            ),
            "markov_w2": (
                f"{final_prefix}.markov_head.markov_w2.weight",
                torch.bfloat16,
            ),
            "confidence_head_weight": (
                f"{final_prefix}.confidence_head.weight",
                torch.float32,
            ),
        }
        raw_heads = self.load_many(source for source, _ in head_sources.values())
        heads = {
            output: replicate.apply(
                output,
                raw_heads[source],
                dtype=dtype,
                destination=None,
            )
            for output, (source, dtype) in head_sources.items()
        }
        return DeepSeekV4DSparkWeights(tensors={**stacked, **heads})


def pack_deepseek_v4_layer_weights(
    layer_id: int,
    raw: Mapping[str, torch.Tensor],
    *,
    ranks: int,
    n_routed_experts: int,
    compress_ratio: int,
    include_tid2eid: bool,
    include_gate_bias: bool,
    destinations: Mapping[str, torch.Tensor] | None = None,
    prefix: str | None = None,
    o_projection_tp_size: int | None = None,
) -> DeepSeekV4PackedLayerWeights:
    """Pack raw checkpoint tensors into new buffers or final-layout destinations.

    This is DeepSeekV4's binding to the generic evaluator, not a pass-through: it turns the
    family's arguments into a :class:`LayerContext`, selects the rank and expert-placement
    policies, supplies the synthetic-weight factories, and carries the two error templates its
    callers' diagnostics are matched against. Both production callers
    (:meth:`DeepSeekV4WeightStore.load_packed_layer_weights` and
    :meth:`~DeepSeekV4WeightStore.load_mtp_weights`) would otherwise repeat that wiring, and
    the MTP path would have to know to pass ``prefix="mtp.0"`` through it.

    The rules themselves live in ``weight_spec.py``; what is here is the argument surface the
    rest of the DeepSeekV4 code already calls.
    """
    from pypto_serving.model.common.weights.packer import pack_layer  # noqa: PLC0415
    from pypto_serving.model.common.weights.spec import LayerContext  # noqa: PLC0415

    from .weight_spec import (  # noqa: PLC0415
        DEEPSEEK_V4_EXPERT_MISSING_ERROR,
        DEEPSEEK_V4_LAYER_RULES,
        DEEPSEEK_V4_SOURCE_MISSING_ERROR,
        deepseek_v4_expert_parallel,
        deepseek_v4_factories,
        deepseek_v4_o_projection_tp_policies,
        deepseek_v4_replicate,
    )

    context = LayerContext(
        layer_id=int(layer_id),
        prefix=f"layers.{int(layer_id)}" if prefix is None else prefix,
        ranks=int(ranks),
        compress_ratio=int(compress_ratio),
        n_routed_experts=int(n_routed_experts),
        include_tid2eid=bool(include_tid2eid),
        include_gate_bias=bool(include_gate_bias),
    )
    tensors = pack_layer(
        DEEPSEEK_V4_LAYER_RULES,
        raw,
        context,
        policy=deepseek_v4_replicate(int(ranks)),
        policies=(
            deepseek_v4_o_projection_tp_policies(int(ranks), int(o_projection_tp_size))
            if o_projection_tp_size is not None
            else None
        ),
        expert_policy=deepseek_v4_expert_parallel(int(ranks), int(n_routed_experts)),
        factories=deepseek_v4_factories(),
        destinations=destinations,
        missing_source_error=DEEPSEEK_V4_SOURCE_MISSING_ERROR,
        missing_expert_error=DEEPSEEK_V4_EXPERT_MISSING_ERROR,
    )
    return DeepSeekV4PackedLayerWeights(layer_id=layer_id, tensors=tensors)

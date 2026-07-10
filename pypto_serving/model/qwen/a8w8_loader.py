# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

from pypto_serving.config.types import LoadedModel, RuntimeConfig, RuntimeModel
from pypto_serving.model.model_loader import (
    ModelLoadRequest,
    _build_layer_specs,
    _build_model_config,
)
from pypto_serving.model.tokenizer import TransformersTokenizerAdapter


_DTYPE_MAP = {
    "I8": torch.int8,
    "F32": torch.float32,
    "BF16": torch.bfloat16,
}


@dataclass(frozen=True)
class _TensorRecord:
    shard: Path
    dtype: str
    shape: tuple[int, ...]
    data_start: int
    offsets: tuple[int, int]


class _SafeTensorIndex:
    """Minimal safetensors reader for Qwen compressed-tensors checkpoints."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.records: dict[str, _TensorRecord] = {}
        for shard in _safetensor_shards(root):
            with shard.open("rb") as f:
                header_len = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(header_len))
            data_start = 8 + header_len
            for key, meta in header.items():
                if key == "__metadata__":
                    continue
                begin, end = meta["data_offsets"]
                self.records[key] = _TensorRecord(
                    shard=shard,
                    dtype=meta["dtype"],
                    shape=tuple(meta["shape"]),
                    data_start=data_start,
                    offsets=(begin, end),
                )

    def load(self, key: str) -> torch.Tensor:
        rec = self.records[key]
        dtype = _DTYPE_MAP[rec.dtype]
        begin, end = rec.offsets
        with rec.shard.open("rb") as f:
            f.seek(rec.data_start + begin)
            raw = bytearray(f.read(end - begin))
        return torch.frombuffer(raw, dtype=dtype).clone().reshape(rec.shape)


def _safetensor_shards(root: Path) -> list[Path]:
    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        index_data = json.loads(index_path.read_text())
        return [root / filename for filename in sorted(set(index_data["weight_map"].values()))]
    return sorted(root.glob("*.safetensors"))


def _hf_linear_to_kernel_i8(index: _SafeTensorIndex, key: str) -> torch.Tensor:
    return index.load(key).t().contiguous()


def _hf_scale_to_kernel(index: _SafeTensorIndex, key: str) -> torch.Tensor:
    return index.load(key).reshape(1, -1).float().contiguous()


def _hf_linear_to_bf16(index: _SafeTensorIndex, weight_key: str, scale_key: str) -> torch.Tensor:
    weight = index.load(weight_key).float()
    scale = index.load(scale_key).float().reshape(-1, 1)
    return (weight * scale).t().contiguous().to(torch.bfloat16)


def _layer_prefix(layer_idx: int) -> str:
    return f"model.layers.{layer_idx}"


class Qwen3A8W8DirectoryLoader:
    """Loader for local Qwen3-14B compressed-tensors W8A8 checkpoints."""

    format_names = ("qwen3-a8w8", "qwen3-w8a8", "a8w8")

    def supports_format(self, model_format: str) -> bool:
        return model_format.lower() in self.format_names

    def can_load(self, model_path: Path) -> bool:
        if not (model_path / "config.json").exists():
            return False
        return bool(_safetensor_shards(model_path))

    def load(self, request: ModelLoadRequest) -> LoadedModel:
        model_path = Path(request.model_dir)
        config_data = json.loads((model_path / "config.json").read_text())
        trust_remote_code = bool(request.loader_options.get("trust_remote_code", False))
        tokenizer = TransformersTokenizerAdapter.from_pretrained(
            str(model_path),
            trust_remote_code=trust_remote_code,
        )
        config = _build_model_config(request.model_id, config_data, tokenizer)
        runtime = request.runtime_config or RuntimeConfig(max_seq_len=config.max_position_embeddings)
        decode_backend = str(request.loader_options.get("decode_backend", "a8w8")).lower()
        if decode_backend != "a8w8":
            raise ValueError(f"unsupported qwen3-a8w8 decode_backend: {decode_backend!r}")
        layer_specs = _build_layer_specs(config)
        index = _SafeTensorIndex(model_path)

        embed_tokens = index.load("model.embed_tokens.weight").to(torch.bfloat16).contiguous()
        final_norm_weight = index.load("model.norm.weight").float().contiguous()
        lm_head = index.load("lm_head.weight").to(torch.bfloat16).contiguous()

        layers = []
        for spec in layer_specs:
            prefix = _layer_prefix(spec.layer_idx)
            layer = SimpleNamespace(
                quantization="a8w8",
                input_rms_weight=index.load(f"{prefix}.input_layernorm.weight").reshape(1, -1).float(),
                wq=_hf_linear_to_kernel_i8(index, f"{prefix}.self_attn.q_proj.weight"),
                wk=_hf_linear_to_kernel_i8(index, f"{prefix}.self_attn.k_proj.weight"),
                wv=_hf_linear_to_kernel_i8(index, f"{prefix}.self_attn.v_proj.weight"),
                wq_scale=_hf_scale_to_kernel(index, f"{prefix}.self_attn.q_proj.weight_scale"),
                wk_scale=_hf_scale_to_kernel(index, f"{prefix}.self_attn.k_proj.weight_scale"),
                wv_scale=_hf_scale_to_kernel(index, f"{prefix}.self_attn.v_proj.weight_scale"),
                q_norm_weight=index.load(f"{prefix}.self_attn.q_norm.weight").reshape(1, -1).float(),
                k_norm_weight=index.load(f"{prefix}.self_attn.k_norm.weight").reshape(1, -1).float(),
                wo=_hf_linear_to_kernel_i8(index, f"{prefix}.self_attn.o_proj.weight"),
                wo_scale=_hf_scale_to_kernel(index, f"{prefix}.self_attn.o_proj.weight_scale"),
                post_rms_weight=index.load(f"{prefix}.post_attention_layernorm.weight").reshape(1, -1).float(),
                w_gate=_hf_linear_to_bf16(
                    index,
                    f"{prefix}.mlp.gate_proj.weight",
                    f"{prefix}.mlp.gate_proj.weight_scale",
                ),
                w_up=_hf_linear_to_bf16(
                    index,
                    f"{prefix}.mlp.up_proj.weight",
                    f"{prefix}.mlp.up_proj.weight_scale",
                ),
                w_down=_hf_linear_to_bf16(
                    index,
                    f"{prefix}.mlp.down_proj.weight",
                    f"{prefix}.mlp.down_proj.weight_scale",
                ),
            )
            layers.append(layer)

        runtime_model = RuntimeModel(
            config=config,
            runtime=runtime,
            embed_tokens=embed_tokens,
            final_norm_weight=final_norm_weight,
            lm_head=lm_head,
            layers=layers,
        )
        return LoadedModel(
            model_id=request.model_id,
            model_dir=str(model_path),
            config=config,
            tokenizer=tokenizer,
            layer_specs=layer_specs,
            runtime_model=runtime_model,
        )

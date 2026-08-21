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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import torch

from .tokenizer import TokenizerAdapter, load_tokenizer
from pypto_serving.config.types import (
    LayerSpec,
    LoadedModel,
    ModelConfig,
    RuntimeConfig,
    RuntimeModel,
)


def _torch_dtype_from_name(name: str) -> torch.dtype:
    """Convert a config dtype string into a torch dtype."""
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    lowered = name.lower()
    if lowered not in mapping:
        raise ValueError(f"Unsupported dtype name: {name}")
    return mapping[lowered]


@dataclass(frozen=True)
class ModelLoadRequest:
    """Normalized request passed from ``ModelLoader`` to a format loader."""

    model_id: str
    model_dir: str
    runtime_config: RuntimeConfig | None = None
    model_format: str | None = None
    loader_options: dict[str, object] = field(default_factory=dict)


class ModelFormatLoader(Protocol):
    """Protocol implemented by model-format-specific loaders."""

    format_names: tuple[str, ...]

    def supports_format(self, model_format: str) -> bool:
        """Return whether this loader handles the explicit format name."""
        raise NotImplementedError

    def can_load(self, model_path: Path) -> bool:
        """Return whether this loader can infer support for a model path."""
        raise NotImplementedError

    def load(self, request: ModelLoadRequest) -> LoadedModel:
        """Load tensors, tokenizer, and metadata for one model request."""
        raise NotImplementedError


def _load_safetensors_weight_map(model_dir: Path) -> dict[str, str]:
    """Return the ``{tensor_name: shard_filename}`` map from a safetensors index.

    Reads ``model.safetensors.index.json`` when present; otherwise synthesizes a
    flat map from every ``*.safetensors`` shard in the directory (each shard
    owning all of its tensors, which is only correct for the single-shard case).
    Shared by every format loader that needs checkpoint layout without reading
    tensor payloads.
    """
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index_data = json.loads(index_path.read_text())
        return dict(index_data.get("weight_map", {}))
    shards = sorted(path.name for path in model_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")
    if len(shards) == 1:
        try:
            from safetensors import safe_open

            with safe_open(str(model_dir / shards[0]), framework="pt", device="cpu") as reader:
                return {name: shards[0] for name in reader.keys()}
        except ImportError:
            raise RuntimeError("safetensors is required to read weight names from a single-shard checkpoint.")
    raise FileNotFoundError(
        f"{model_dir} has multiple safetensors shards but no model.safetensors.index.json; "
        "the index is required to map tensor names to shards."
    )


def _safetensors_shard_filenames(weight_map: dict[str, str]) -> list[str]:
    """Return the sorted unique shard filenames referenced by a weight map."""
    return sorted(set(weight_map.values()))


def _load_safetensors_subset(
    model_dir: Path,
    weight_map: dict[str, str],
    names: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    """Load only *names*, opening each shard once and skipping absent optional weights.

    Reads through `safe_open` rather than `load_file` so a shard contributes only the tensors
    asked for: the point of this path is that the checkpoint is never fully resident, and
    `load_file` would materialize a whole shard to reach one tensor in it.
    """
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("safetensors is required to load weights from a local model directory.") from exc

    wanted: dict[str, list[str]] = {}
    for name in names:
        filename = weight_map.get(name)
        if filename is None:
            # Optional weights (a tied `lm_head`, an alternative final-norm name) are absent by
            # design; the caller decides what a missing one means.
            continue
        wanted.setdefault(filename, []).append(name)

    state_dict: dict[str, torch.Tensor] = {}
    for filename, shard_names in wanted.items():
        with safe_open(str(model_dir / filename), framework="pt", device="cpu") as reader:
            for name in shard_names:
                state_dict[name] = reader.get_tensor(name)
    return state_dict


def _require_tensor(state_dict: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    """Return a required tensor or raise a key error with its weight name."""
    if name not in state_dict:
        raise KeyError(f"Missing weight tensor: {name}")
    return state_dict[name]


def _optional_tensor(state_dict: dict[str, torch.Tensor], names: list[str]) -> torch.Tensor | None:
    """Return the first available tensor from a list of candidate names."""
    for name in names:
        if name in state_dict:
            return state_dict[name]
    return None


def _build_model_config(model_id: str, config_data: dict, tokenizer: TokenizerAdapter) -> ModelConfig:
    """Build internal model metadata from Hugging Face config JSON."""
    hidden_size = int(config_data["hidden_size"])
    num_heads = int(config_data["num_attention_heads"])
    num_kv_heads = int(config_data.get("num_key_value_heads", num_heads))
    head_dim = hidden_size // num_heads
    return ModelConfig(
        model_id=model_id,
        architecture=str(config_data.get("architectures", [config_data.get("model_type", "unknown")])[0]),
        vocab_size=int(config_data["vocab_size"]),
        hidden_size=hidden_size,
        intermediate_size=int(config_data["intermediate_size"]),
        num_hidden_layers=int(config_data["num_hidden_layers"]),
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        max_position_embeddings=int(config_data.get("max_position_embeddings", config_data.get("seq_length", 4096))),
        rms_norm_eps=float(config_data.get("rms_norm_eps", config_data.get("layer_norm_epsilon", 1e-6))),
        rope_theta=float(config_data.get("rope_theta", 10000.0)),
        bos_token_id=config_data.get("bos_token_id", tokenizer.bos_token_id),
        eos_token_id=config_data.get("eos_token_id", tokenizer.eos_token_id),
        pad_token_id=config_data.get("pad_token_id", tokenizer.pad_token_id),
        torch_dtype=str(config_data.get("torch_dtype", "float16")),
    )


def _build_layer_specs(config: ModelConfig) -> list[LayerSpec]:
    """Build per-layer shape specs from model metadata."""
    return [
        LayerSpec(
            layer_idx=layer_idx,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
        )
        for layer_idx in range(config.num_hidden_layers)
    ]


def _cast_weight(weight: torch.Tensor, runtime: RuntimeConfig) -> torch.Tensor:
    """Move a weight tensor to the configured runtime device and dtype."""
    dtype = _torch_dtype_from_name(runtime.weight_dtype)
    return weight.to(device=runtime.device, dtype=dtype)


class HuggingFaceDirectoryLoader:
    """Loader for local Hugging Face-style decoder model directories."""

    format_names = ("huggingface", "hf")

    def supports_format(self, model_format: str) -> bool:
        """Return whether ``model_format`` names the Hugging Face loader."""
        return model_format.lower() in self.format_names

    def can_load(self, model_path: Path) -> bool:
        """Detect a local directory with config and safetensors weights."""
        config_path = model_path / "config.json"
        if not config_path.exists():
            return False
        if (model_path / "model.safetensors.index.json").exists():
            return True
        if any(model_path.glob("*.safetensors")):
            return True
        return False

    def load(self, request: ModelLoadRequest) -> LoadedModel:
        """Load a supported Hugging Face directory into runtime tensors."""
        model_path = Path(request.model_dir)
        config_path = model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config.json in {model_path}")

        trust_remote_code = bool(request.loader_options.get("trust_remote_code", False))
        tokenizer = load_tokenizer(model_path, trust_remote_code=trust_remote_code)
        config_data = json.loads(config_path.read_text())
        config = _build_model_config(request.model_id, config_data, tokenizer)
        runtime = request.runtime_config or RuntimeConfig(max_seq_len=config.max_position_embeddings)
        layer_specs = _build_layer_specs(config)
        weight_map = _load_safetensors_weight_map(model_path)
        # Metadata plus the globals only. `embed_tokens` has to stay resident because
        # `Executor.lookup_embeddings` reads it at request time; the other two are one tensor
        # each. Every per-layer tensor is read later, while it is staged.
        state_dict = _load_safetensors_subset(
            model_path,
            weight_map,
            (
                "model.embed_tokens.weight",
                "model.norm.weight",
                "model.final_layernorm.weight",
                "lm_head.weight",
            ),
        )

        if config.architecture.lower() not in {"qwen2forcausallm", "qwen3forcausallm", "qwen2model", "qwen3model"}:
            raise ValueError(
                f"Unsupported architecture {config.architecture}. "
                "Current Hugging Face adapter supports Qwen2/Qwen3-style decoder-only models."
            )

        embed_tokens = _cast_weight(_require_tensor(state_dict, "model.embed_tokens.weight"), runtime)
        final_norm = _optional_tensor(state_dict, ["model.norm.weight", "model.final_layernorm.weight"])
        if final_norm is None:
            final_norm = _require_tensor(state_dict, "model.norm.weight")
        final_norm_weight = _cast_weight(final_norm, runtime)
        lm_head = _optional_tensor(state_dict, ["lm_head.weight"])
        if lm_head is None:
            lm_head = embed_tokens
        else:
            lm_head = _cast_weight(lm_head, runtime)

        runtime_model = RuntimeModel(
            config=config,
            runtime=runtime,
            embed_tokens=embed_tokens,
            final_norm_weight=final_norm_weight,
            lm_head=lm_head,
            # `layers` is left at its default: the executor stages each layer as it reads it,
            # through the metadata below. Populating it eagerly cost a second copy of the model
            # at the staging peak.
            extra={"model_dir": str(model_path), "weight_map": weight_map},
        )

        return LoadedModel(
            model_id=request.model_id,
            model_dir=str(model_path),
            config=config,
            tokenizer=tokenizer,
            layer_specs=layer_specs,
            runtime_model=runtime_model,
        )


class DeepSeekV4W8A8DirectoryLoader:
    """Lazy loader for the local DeepSeekV4 Flash W8A8 checkpoint."""

    format_names = ("deepseek_v4_w8a8", "deepseek-v4-w8a8", "dsv4-w8a8")

    def supports_format(self, model_format: str) -> bool:
        """Return whether ``model_format`` names the DeepSeekV4 W8A8 loader."""
        return model_format.lower() in self.format_names

    def can_load(self, model_path: Path) -> bool:
        """Detect a DeepSeekV4 compressed-tensors checkpoint directory."""
        config_path = model_path / "config.json"
        index_path = model_path / "model.safetensors.index.json"
        if not config_path.exists() or not index_path.exists():
            return False
        try:
            config_data = json.loads(config_path.read_text())
        except json.JSONDecodeError:
            return False
        return _is_deepseek_v4_config(config_data)

    def load(self, request: ModelLoadRequest) -> LoadedModel:
        """Load tokenizer and metadata without materializing all quantized weights."""
        model_path = Path(request.model_dir)
        config_path = model_path / "config.json"
        index_path = model_path / "model.safetensors.index.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config.json in {model_path}")
        if not index_path.exists():
            raise FileNotFoundError(f"Missing model.safetensors.index.json in {model_path}")

        config_data = json.loads(config_path.read_text())
        if not _is_deepseek_v4_config(config_data):
            raise ValueError(f"{model_path} is not a DeepSeekV4 checkpoint")
        quantization = config_data.get("quantization_config", {})
        if quantization.get("quant_method") != "compressed-tensors":
            raise ValueError(
                "DeepSeekV4 serving requires the W8A8 compressed-tensors checkpoint; "
                f"got quant_method={quantization.get('quant_method')!r}"
            )

        trust_remote_code = bool(request.loader_options.get("trust_remote_code", False))
        tokenizer = load_tokenizer(model_path, trust_remote_code=trust_remote_code)
        config = _build_deepseek_v4_model_config(request.model_id, config_data, tokenizer)
        runtime = request.runtime_config or RuntimeConfig(max_seq_len=min(config.max_position_embeddings, 8192))
        layer_specs = _build_layer_specs(config)
        weight_map = _load_safetensors_weight_map(model_path)
        _validate_deepseek_v4_weight_index(weight_map, config_data)

        placeholder = torch.empty(0, config.hidden_size, dtype=torch.bfloat16)
        runtime_model = RuntimeModel(
            config=config,
            runtime=runtime,
            embed_tokens=placeholder,
            final_norm_weight=torch.empty(0, dtype=torch.bfloat16),
            lm_head=placeholder,
            extra={
                "family": "deepseek_v4",
                "checkpoint_format": "w8a8-compressed-tensors",
                "config_data": config_data,
                "quantization_config": quantization,
                "weight_map": weight_map,
                "model_dir": str(model_path),
                "compress_ratios": tuple(int(ratio) for ratio in config_data["compress_ratios"]),
            },
        )

        return LoadedModel(
            model_id=request.model_id,
            model_dir=str(model_path),
            config=config,
            tokenizer=tokenizer,
            layer_specs=layer_specs,
            runtime_model=runtime_model,
        )


def _is_deepseek_v4_config(config_data: dict) -> bool:
    """Return whether config metadata names DeepSeekV4."""
    model_type = str(config_data.get("model_type", "")).lower()
    architectures = {str(item).lower() for item in config_data.get("architectures", [])}
    return model_type == "deepseek_v4" or "deepseekv4forcausallm" in architectures


def _build_deepseek_v4_model_config(
    model_id: str,
    config_data: dict,
    tokenizer: TokenizerAdapter,
) -> ModelConfig:
    """Build internal metadata for DeepSeekV4 Flash."""
    return ModelConfig(
        model_id=model_id,
        architecture=str(config_data.get("architectures", ["DeepseekV4ForCausalLM"])[0]),
        vocab_size=int(config_data["vocab_size"]),
        hidden_size=int(config_data["hidden_size"]),
        intermediate_size=int(config_data["moe_intermediate_size"]),
        num_hidden_layers=int(config_data["num_hidden_layers"]),
        num_attention_heads=int(config_data["num_attention_heads"]),
        num_key_value_heads=int(config_data.get("num_key_value_heads", 1)),
        head_dim=int(config_data["head_dim"]),
        max_position_embeddings=int(config_data["max_position_embeddings"]),
        rms_norm_eps=float(config_data["rms_norm_eps"]),
        rope_theta=float(config_data["rope_theta"]),
        bos_token_id=config_data.get("bos_token_id", tokenizer.bos_token_id),
        eos_token_id=config_data.get("eos_token_id", tokenizer.eos_token_id),
        pad_token_id=config_data.get("pad_token_id", tokenizer.pad_token_id),
        torch_dtype=str(config_data.get("torch_dtype", "bfloat16")),
    )


def _validate_deepseek_v4_weight_index(weight_map: dict[str, str], config_data: dict) -> None:
    """Fail early if the W8A8 checkpoint does not expose required tensor names."""
    required = [
        "embed.weight",
        "norm.weight",
        "head.weight",
        "layers.0.attn.wq_b.weight",
        "layers.0.attn.wq_b.scale",
        "layers.0.attn.wo_b.weight",
        "layers.0.attn.wo_b.scale",
        "layers.0.ffn.experts.0.w1.weight",
        "layers.0.ffn.experts.0.w1.scale",
    ]
    missing = [name for name in required if name not in weight_map]
    if missing:
        raise KeyError(f"DeepSeekV4 W8A8 checkpoint is missing required tensors: {', '.join(missing)}")
    ratios = config_data.get("compress_ratios")
    if not isinstance(ratios, list) or len(ratios) != int(config_data["num_hidden_layers"]) + 1:
        raise ValueError(
            "DeepSeekV4 config compress_ratios must include one entry per hidden layer plus MTP/final entry"
        )


class ModelLoader:
    """Registry that selects a model-format loader and loads models."""

    def __init__(self, format_loaders: list[ModelFormatLoader] | None = None) -> None:
        """Create a loader registry with optional custom format loaders."""
        self._format_loaders = format_loaders or [DeepSeekV4W8A8DirectoryLoader(), HuggingFaceDirectoryLoader()]

    def register(self, format_loader: ModelFormatLoader) -> None:
        """Register an additional model format loader."""
        self._format_loaders.append(format_loader)

    def load(
        self,
        model_id: str,
        model_dir: str,
        runtime_config: RuntimeConfig | None = None,
        model_format: str | None = None,
        **loader_options: object,
    ) -> LoadedModel:
        """Load a model directory using an explicit or inferred format."""
        request = ModelLoadRequest(
            model_id=model_id,
            model_dir=model_dir,
            runtime_config=runtime_config,
            model_format=model_format,
            loader_options=loader_options,
        )
        loader = self._select_loader(request)
        return loader.load(request)

    def _select_loader(self, request: ModelLoadRequest) -> ModelFormatLoader:
        """Select the first registered loader that can handle a request."""
        model_path = Path(request.model_dir)
        if request.model_format is not None:
            for loader in self._format_loaders:
                if loader.supports_format(request.model_format):
                    return loader
            supported = sorted({name for loader in self._format_loaders for name in loader.format_names})
            raise ValueError(
                f"Unsupported model_format={request.model_format!r}. "
                f"Registered formats: {', '.join(supported)}"
            )

        for loader in self._format_loaders:
            if loader.can_load(model_path):
                return loader
        raise ValueError(
            f"Could not detect a supported model format in {model_path}. "
            "Pass model_format explicitly or register another format loader."
        )

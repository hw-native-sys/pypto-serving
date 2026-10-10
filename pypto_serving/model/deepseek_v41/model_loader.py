# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Metadata-only registration of the original V4.1 text checkpoint.

The executor owns device readiness. Loading metadata does not assert that the
selected lib revision has the composite entries needed to execute the model.
"""

import json
from pathlib import Path

import torch

from pypto_serving.config.types import LoadedModel, ModelConfig, RuntimeConfig, RuntimeModel
from pypto_serving.model.model_family import is_deepseek_v41_config, read_model_config
from pypto_serving.model.model_loader import SafetensorsDirectoryLoader, _build_layer_specs
from pypto_serving.model.tokenizer import load_tokenizer
from .weight_loader import V41WeightLoader


class DeepSeekV41DirectoryLoader(SafetensorsDirectoryLoader):
    """Validate text metadata and the index without opening weight payloads."""

    format_names = ("deepseek_v41", "deepseek-v41", "dsv41")

    def _recognises(self, model_path: Path) -> bool:
        return is_deepseek_v41_config(read_model_config(model_path))

    def load(self, request) -> LoadedModel:
        model_path = Path(request.model_dir).resolve()
        weights = V41WeightLoader(model_path)
        text = weights.config
        raw = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
        values = raw["text_config"]
        tokenizer = load_tokenizer(
            model_path, trust_remote_code=bool(request.loader_options.get("trust_remote_code", False)),
        )
        config = ModelConfig(
            model_id=request.model_id,
            architecture="DeepseekV41ForCausalLM",
            vocab_size=text.vocab_size,
            hidden_size=text.hidden_size,
            intermediate_size=int(values["moe_intermediate_size"]),
            num_hidden_layers=text.num_hidden_layers,
            num_attention_heads=text.num_attention_heads,
            num_key_value_heads=int(values.get("num_key_value_heads", 1)),
            head_dim=text.head_dim,
            max_position_embeddings=text.max_position_embeddings,
            rms_norm_eps=text.rms_norm_eps,
            rope_theta=float(values["rope_theta"]),
            bos_token_id=text.bos_token_id,
            eos_token_id=text.eos_token_id,
            pad_token_id=text.pad_token_id,
            torch_dtype="bfloat16",
        )
        runtime = request.runtime_config or RuntimeConfig(
            page_size=128, max_seq_len=min(text.max_position_embeddings, 8192 + 128),
        )
        if runtime.device != "cpu":
            raise ValueError("V4.1 metadata and checkpoint preparation require runtime.device='cpu'")
        if not 0 < runtime.max_seq_len <= text.max_position_embeddings:
            raise ValueError("V4.1 max_seq_len must fit the checkpoint position capacity")
        placeholder = torch.empty((0, text.hidden_size), dtype=torch.bfloat16)
        model = RuntimeModel(
            config=config,
            runtime=runtime,
            embed_tokens=placeholder,
            final_norm_weight=torch.empty(0, dtype=torch.bfloat16),
            lm_head=placeholder,
            extra={
                "family": "deepseek_v41",
                "checkpoint_format": "fp8-ue8m0-packed-fp4",
                "config_data": raw,
                "model_dir": str(model_path),
                "compress_ratios": text.compress_ratios,
            },
        )
        return LoadedModel(
            model_id=request.model_id, model_dir=str(model_path), config=config,
            tokenizer=tokenizer, layer_specs=_build_layer_specs(config), runtime_model=model,
        )

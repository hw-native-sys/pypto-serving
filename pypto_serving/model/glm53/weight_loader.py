# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""GLM metadata loading through the kernel-owned W8A8 checkpoint reader."""

from __future__ import annotations

from pathlib import Path

import torch

from pypto_serving.config.types import LoadedModel, ModelConfig, RuntimeConfig, RuntimeModel
from pypto_serving.model.glm53.kernel_api import import_kernel, packed_capacity
from pypto_serving.model.model_family import is_glm53_config, read_model_config
from pypto_serving.model.tokenizer import load_tokenizer


def load_checkpoint(model_id: str, model_dir: str, runtime: RuntimeConfig | None = None) -> LoadedModel:
    """Validate deployment metadata without materializing the weight payloads."""
    root = Path(model_dir)
    raw = read_model_config(root)
    if not is_glm53_config(raw):
        raise ValueError("The checkpoint does not identify GLM-5.3-Flash")
    runtime = runtime or RuntimeConfig(max_batch_size=1, max_seq_len=512)
    capacity = packed_capacity(runtime.max_seq_len)
    api = import_kernel("weights", capacity=capacity)
    checkpoint = api.Checkpoint(root)
    text = checkpoint.config
    api.ResidentPrefillWeights.load(checkpoint, tp_size=16, ep_size=16)
    tokenizer = load_tokenizer(root)
    config = ModelConfig(
        model_id=model_id, architecture="Glm5NextForConditionalGeneration",
        vocab_size=int(text["vocab_size"]), hidden_size=int(text["hidden_size"]),
        intermediate_size=int(text["intermediate_size"]), num_hidden_layers=int(text["num_hidden_layers"]),
        num_attention_heads=int(text["num_attention_heads"]), num_key_value_heads=1,
        head_dim=int(text["kv_lora_rank"]), max_position_embeddings=int(text.get("max_position_embeddings", 32768)),
        rms_norm_eps=float(text["rms_norm_eps"]), rope_theta=float(text.get("rope_theta", 10000)),
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id, torch_dtype="bfloat16",
    )
    placeholder = torch.empty(0, config.hidden_size, dtype=torch.bfloat16)
    model = RuntimeModel(
        config=config, runtime=runtime, embed_tokens=placeholder,
        final_norm_weight=torch.empty(0), lm_head=placeholder,
        extra={"family": "glm53", "model_dir": str(root.resolve()), "capacity": capacity},
    )
    return LoadedModel(
        model_id=model_id, model_dir=str(root.resolve()), config=config,
        tokenizer=tokenizer, layer_specs=[], runtime_model=model,
    )

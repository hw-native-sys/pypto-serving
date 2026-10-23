# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek V4.1 npu executor."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import torch

from pypto_serving.config.types import LoadedModel, ModelConfig, RuntimeConfig, RuntimeModel
from pypto_serving.model.common.executor.executor import ModelExecutor
from pypto_serving.model.model_family import is_deepseek_v41_config, read_model_config
from pypto_serving.model.model_loader import SafetensorsDirectoryLoader, _build_layer_specs
from pypto_serving.model.tokenizer import load_tokenizer

from .npu_runner import (
    BuildOptions,
    MissingCompositeInterface,
    SegmentTopology,
    V41ModelRunner,
    load_composite_bindings,
)
from .task_args import build_cache_groups, validate_cache_groups
from .weight_loader import RankPlacement, V41ExecutionPlan, V41WeightLoader, plan_layers


# model_loader
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
        if runtime.page_size != 128:
            raise ValueError("V4.1 cache pages require 128 physical rows")
        layers = plan_layers(raw)
        if runtime.kv_cache_groups:
            validate_cache_groups(layers, runtime.kv_cache_groups, runtime.max_seq_len,
                                  dp_size=runtime.kv_cache_groups[0].num_partitions)
        elif len(layers) == 40:
            runtime = replace(runtime, kv_cache_groups=build_cache_groups(layers, runtime.max_seq_len))
        if len(layers) == 40:
            capacity = SegmentTopology.for_decode().capacity
            runtime = replace(
                runtime,
                max_num_batched_tokens=min(runtime.max_num_batched_tokens, capacity),
                max_prefill_tokens_per_request=min(
                    runtime.max_prefill_tokens_per_request or capacity, capacity),
                requires_homogeneous_prefill_decode=True,
            )
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


# npu_executor
class DeepSeekV41PyptoExecutor(ModelExecutor):
    """Shared worker interface with a synchronous, collective V4.1 runner."""
    def __init__(self, kv_cache_manager=None, *, platform="a5", device_ids=tuple(range(8)),
                 pypto_build_dir=None, bindings=None):
        super().__init__(kv_cache_manager)
        if platform != "a5":
            raise ValueError("V4.1 M0 currently requires the A5 platform")
        self.device_ids = tuple(device_ids)
        self.bindings = bindings
        self.runners = {}
        self._failed_runners = []
        self.platform = platform
        self.pypto_build_dir = pypto_build_dir

    def register_model(self, model_id, record):
        if model_id in self.runners:
            raise ValueError("model already registered")
        bindings = self.bindings or load_composite_bindings(
            record.runtime_model.extra["model_dir"], record.runtime.max_seq_len,
            dp_size=len(self.device_ids))
        if not bindings.cache_groups:
            raise MissingCompositeInterface("V4.1 requires explicit grouped cache layouts and capacities")
        if record.runtime.kv_cache_groups != bindings.cache_groups:
            raise ValueError("scheduler and composite cache group contracts differ")
        plan = V41ExecutionPlan(
            record.runtime_model.extra["model_dir"],
            RankPlacement(0, tp_size=1, dp_size=len(self.device_ids), ep_size=len(self.device_ids)),
        )
        runner = V41ModelRunner(
            plan, bindings, device_ids=self.device_ids, runtime=record.runtime,
            build_options=BuildOptions(self.platform, self.pypto_build_dir),
        )
        try:
            pages = runner.preflight()
        except Exception:
            if not runner.closed:
                self._failed_runners.append(runner)
            raise
        self.runners[model_id] = runner
        return pages

    def lookup_embeddings(self, model, token_ids):
        return self.runners[model.config.model_id].lookup_embeddings(token_ids)

    def run_prefill(self, model, batch):
        return self.runners[model.config.model_id].run_prefill(model, batch)

    def run_decode(self, model, batch):
        return self.runners[model.config.model_id].run_decode(model, batch)

    def release_finished_requests(self, request_ids):
        for runner in self.runners.values():
            runner.release_finished_requests(request_ids)

    def close(self):
        for runner in (*self.runners.values(), *self._failed_runners):
            runner.close()
        self.runners.clear()
        self._failed_runners.clear()

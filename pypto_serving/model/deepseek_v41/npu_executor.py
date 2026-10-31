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
import os
from dataclasses import replace
from pathlib import Path

import torch

from pypto_serving.config.types import (
    DecodeBatch,
    DecodeResult,
    LoadedModel,
    ModelConfig,
    PrefillBatch,
    RuntimeConfig,
    RuntimeModel,
)
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
from .dspark_runtime import DSparkLoweringProvider
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
            dp_size = runtime.kv_cache_groups[0].num_partitions
            validate_cache_groups(
                layers, runtime.kv_cache_groups, runtime.max_seq_len, dp_size=dp_size,
            )
        elif len(layers) == 40:
            # Direct executor users may omit grouped cache specs. Keep that
            # path consistent with the CLI-selected DP topology instead of
            # silently constructing an eight-partition runtime for DP4.
            configured_dp = os.environ.get("PYPTO_DSV41_DP_SIZE", "8")
            try:
                dp_size = int(configured_dp)
            except ValueError as exc:
                raise ValueError("PYPTO_DSV41_DP_SIZE must be 4 or 8") from exc
            if dp_size not in (4, 8):
                raise ValueError("PYPTO_DSV41_DP_SIZE must be 4 or 8")
            runtime = replace(
                runtime,
                kv_cache_groups=build_cache_groups(
                    layers, runtime.max_seq_len, dp_size=dp_size,
                ),
            )
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
                 pypto_build_dir=None, use_compile_cache=False, bindings=None,
                 dspark_lowering_provider: DSparkLoweringProvider | None = None):
        super().__init__(kv_cache_manager)
        if platform != "a5":
            raise ValueError("V4.1 M0 currently requires the A5 platform")
        self.device_ids = tuple(device_ids)
        self.bindings = bindings
        self.runners = {}
        self._failed_runners = []
        self.platform = platform
        self.pypto_build_dir = pypto_build_dir
        self.use_compile_cache = use_compile_cache
        self.dspark_lowering_provider = dspark_lowering_provider

    def register_model(self, model_id, record):
        if model_id in self.runners:
            raise ValueError("model already registered")
        dp_size = (record.runtime.kv_cache_groups[0].num_partitions
                   if record.runtime.kv_cache_groups else None)
        bindings = self.bindings or load_composite_bindings(
            record.runtime_model.extra["model_dir"], record.runtime.max_seq_len,
            tokenizer=record.tokenizer, dp_size=dp_size,
            dspark_lowering_provider=self.dspark_lowering_provider)
        if not bindings.cache_groups:
            raise MissingCompositeInterface("V4.1 requires explicit grouped cache layouts and capacities")
        if record.runtime.kv_cache_groups != bindings.cache_groups:
            raise ValueError("scheduler and composite cache group contracts differ")
        rank_count = len(self.device_ids)
        if rank_count not in (4, 8):
            raise ValueError("V4.1 DSpark serving requires 4 or 8 devices")
        if bindings.segment_topology.dp != rank_count:
            raise ValueError(
                "V4.1 DSpark topology mismatch: set PYPTO_DSV41_DP_SIZE "
                f"to {rank_count} before creating bindings"
            )
        plan = V41ExecutionPlan(
            record.runtime_model.extra["model_dir"],
            RankPlacement(0, tp_size=1, dp_size=rank_count, ep_size=rank_count),
        )
        runner = V41ModelRunner(
            plan, bindings, device_ids=self.device_ids, runtime=record.runtime,
            build_options=BuildOptions(self.platform, self.pypto_build_dir, self.use_compile_cache),
        )
        try:
            pages = runner.preflight()
        except Exception:
            if not runner.closed:
                self._failed_runners.append(runner)
            raise
        if self.dspark_lowering_provider is not None:
            # Custom bindings may have been supplied by a test or deployment;
            # attach the same provider after resident resources are created.
            runner.resources.dspark_lowering_provider = self.dspark_lowering_provider
        self.runners[model_id] = runner
        return pages

    def lookup_embeddings(self, model, token_ids):
        return self.runners[model.config.model_id].lookup_embeddings(token_ids)

    def run_prefill(self, model, batch):
        return self.runners[model.config.model_id].run_prefill(model, batch)

    def run_decode(self, model, batch):
        return self.runners[model.config.model_id].run_decode(model, batch)

    def run_dspark_prefill_batch(self, batch: PrefillBatch):
        """Dispatch serving-owned, already-lowered DSpark bootstrap stages."""
        if not self.runners:
            raise RuntimeError("V4.1 DSpark runner is not registered")
        if len(self.runners) != 1:
            raise RuntimeError("V4.1 DSpark batch dispatch requires one registered model")
        return next(iter(self.runners.values())).run_dspark_prefill_batch(batch)

    def lower_dspark_prefill_batch(
        self, model: RuntimeModel, batch: PrefillBatch, *,
        request_ids, sampled_token_ids,
    ):
        """Create a complete DSpark prefill payload through the resident provider."""
        if not self.runners:
            raise MissingCompositeInterface("V4.1 DSpark runner is not registered")
        if len(self.runners) != 1:
            raise MissingCompositeInterface(
                "V4.1 DSpark lowering requires one registered model"
            )
        return next(iter(self.runners.values())).lower_dspark_prefill_batch(
            model, batch, request_ids=request_ids, sampled_token_ids=sampled_token_ids,
        )

    def run_dspark_decode_batch(self, batch: DecodeBatch) -> DecodeResult:
        """Dispatch one serving-owned DSpark verification round."""
        if not self.runners:
            raise RuntimeError("V4.1 DSpark runner is not registered")
        if len(self.runners) != 1:
            raise RuntimeError("V4.1 DSpark batch dispatch requires one registered model")
        result = next(iter(self.runners.values())).run_dspark_decode_batch(batch)
        if not isinstance(result, DecodeResult):
            raise TypeError("V4.1 DSpark runner must return DecodeResult")
        return result

    def lower_dspark_decode_batch(self, model: RuntimeModel, batch: DecodeBatch):
        """Create a complete DSpark decode payload through the resident provider."""
        if not self.runners:
            raise MissingCompositeInterface("V4.1 DSpark runner is not registered")
        if len(self.runners) != 1:
            raise MissingCompositeInterface(
                "V4.1 DSpark lowering requires one registered model"
            )
        return next(iter(self.runners.values())).lower_dspark_decode_batch(model, batch)

    def release_finished_requests(self, request_ids):
        for runner in self.runners.values():
            runner.release_finished_requests(request_ids)

    def close(self):
        for runner in (*self.runners.values(), *self._failed_runners):
            runner.close()
        self.runners.clear()
        self._failed_runners.clear()

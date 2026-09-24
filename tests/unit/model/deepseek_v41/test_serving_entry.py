# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Serving registration and lifecycle contracts without device execution."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pypto_serving.cli.main import build_parser, build_serving_engine_config
from pypto_serving.config.types import KVCacheGroupSpec, KVCacheSpec, RuntimeConfig
from pypto_serving.model.deepseek_v41.composite import CompositeBindings, MissingCompositeInterface
from pypto_serving.model.deepseek_v41.execution_plan import plan_layers
from pypto_serving.model.deepseek_v41.weight_spec import backbone_weight_specs
from pypto_serving.model.model_loader import ModelLoader


FIXTURE = Path(__file__).resolve().parents[4] / "tests/fixtures/deepseek_v41/config.json"


@pytest.fixture
def metadata_checkpoint(tmp_path, monkeypatch):
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    raw["text_config"].update(
        hidden_size=256, vocab_size=256, num_hidden_layers=1, num_attention_heads=8,
        head_dim=64, q_lora_rank=256, o_lora_rank=128, o_groups=4, n_routed_experts=8,
        moe_intermediate_size=256, index_n_heads=8, index_head_dim=32,
        kv_source_layer_ids=[0], index_source_layer_ids=[0], compress_ratios=[2],
    )
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    # There is deliberately no shard file: metadata loading must never open one.
    index = {name: "missing-payload.safetensors" for name in backbone_weight_specs(raw)}
    index["vision.weight"] = "also-missing.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": index}), encoding="utf-8",
    )
    monkeypatch.setattr(
        "pypto_serving.model.deepseek_v41.model_loader.load_tokenizer",
        lambda *args, **kwargs: SimpleNamespace(bos_token_id=0, eos_token_id=1, pad_token_id=2),
    )
    return tmp_path, raw


@pytest.mark.parametrize("model_format", [None, "deepseek_v41", "deepseek-v41", "dsv41"])
def test_metadata_loader_registers_without_weight_payload(metadata_checkpoint, model_format):
    path, raw = metadata_checkpoint
    runtime = RuntimeConfig(page_size=128, max_seq_len=8320)
    loaded = ModelLoader().load("text", str(path), runtime, model_format=model_format)
    assert loaded.config.architecture == "DeepseekV41ForCausalLM"
    assert loaded.config.head_dim == 64  # Not hidden_size / num_attention_heads.
    assert loaded.runtime_model.extra["family"] == "deepseek_v41"
    assert loaded.runtime_model.extra["config_data"] == raw
    assert loaded.runtime_model.runtime is runtime
    assert loaded.runtime_model.embed_tokens.numel() == 0
    assert loaded.runtime_model.lm_head.numel() == 0
    assert len(loaded.layer_specs) == 1


def test_loader_keeps_unsupported_quantization_closed(metadata_checkpoint):
    path, raw = metadata_checkpoint
    raw["quantization_config"]["quant_method"] = "compressed-tensors"
    (path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="FP8 32x32"):
        ModelLoader().load("text", str(path))


def _args(path, *extra):
    return build_parser().parse_args([
        "--model", str(path), "--platform", "a5", "--tp", "4", "--dp", "2", "--ep", "8",
        "--devices", "0,1,2,3,4,5,6,7", "--max-model-len", "8320", *extra,
    ])


@pytest.fixture
def binding_metadata(metadata_checkpoint, monkeypatch):
    _, raw = metadata_checkpoint

    def unreachable(*args, **kwargs):
        raise AssertionError("configuration must not dispatch or allocate")

    groups = (KVCacheGroupSpec(
        "test_cache", (0,), KVCacheSpec(block_size=128, page_size_bytes=128),
        max_blocks_per_seq=65, num_partitions=2,
    ),)
    bindings = CompositeBindings(
        revision="test-contract-only",
        entries={(phase, layer.mode): unreachable for phase in ("prefill", "decode")
                 for layer in plan_layers(raw)},
        initialize=unreachable, output=unreachable, allocate=unreachable,
        prepare_weights=unreachable, reset_request=unreachable, wait=unreachable,
        close=unreachable, cache_groups=groups,
    )
    monkeypatch.setattr(
        "pypto_serving.model.deepseek_v41.composite.load_composite_bindings", lambda: bindings,
    )
    return bindings


def test_cli_routes_one_ep_worker_with_two_cache_partitions(metadata_checkpoint, binding_metadata):
    path, _ = metadata_checkpoint
    config = build_serving_engine_config(_args(path))
    assert config.executor_cls == "PyptoDeepSeekV41Executor"
    assert config.parallel_config.num_replicas == 1
    assert config.worker_device_ids() == tuple(range(8))
    assert config.parallel_config.tensor_parallel_size == 1
    assert config.parallel_config.data_parallel_size == 1
    assert config.parallel_config.expert_parallel_size == 8
    assert config.runtime_config.kv_cache_groups == binding_metadata.cache_groups
    assert config.runtime_config.requires_homogeneous_prefill_decode
    assert not config.enable_prefix_cache
    assert not config.resolve_async_scheduling()


@pytest.mark.parametrize("extra,message", [
    (("--platform", "a2a3"), "platform a5"),
    (("--tp", "2"), "--tp 4 --dp 2 --ep 8"),
    (("--dp", "1"), "--tp 4 --dp 2 --ep 8"),
    (("--devices", "0,1,2,3"), "exactly eight"),
    (("--block-size", "64"), "block-size 128"),
    (("--num-speculative-tokens", "1"), "DSpark or MTP"),
    (("--speculative-config", '{"method":"dspark"}'), "DSpark or MTP"),
])
def test_cli_rejects_wrong_topology_before_binding(metadata_checkpoint, monkeypatch, extra, message):
    path, _ = metadata_checkpoint

    def forbidden():
        raise AssertionError("invalid settings must fail before binding resolution")

    monkeypatch.setattr("pypto_serving.model.deepseek_v41.composite.load_composite_bindings", forbidden)
    with pytest.raises(ValueError, match=message):
        build_serving_engine_config(_args(path, *extra))


def test_cli_cannot_fall_back_to_generic_kv(metadata_checkpoint, binding_metadata, monkeypatch):
    path, _ = metadata_checkpoint
    monkeypatch.setattr(
        "pypto_serving.model.deepseek_v41.composite.load_composite_bindings",
        lambda: replace(binding_metadata, cache_groups=()),
    )
    with pytest.raises(MissingCompositeInterface, match="grouped cache"):
        build_serving_engine_config(_args(path))


def test_worker_resolves_v41_executor_and_releases_request_state():
    from pypto_serving.model.deepseek_v41.npu_executor import DeepSeekV41PyptoExecutor
    from pypto_serving.serving.server.serving_worker import WorkerProcess

    worker = WorkerProcess(SimpleNamespace(executor_cls="PyptoDeepSeekV41Executor"), None, None)
    assert worker._resolve_executor_cls() is DeepSeekV41PyptoExecutor
    released = []
    worker.executor = SimpleNamespace(release_finished_requests=lambda ids: released.extend(ids))
    worker._req_cache = {"finished": object(), "live": object()}
    worker._last_tokens = {"finished": [4], "live": [8]}
    worker._release_finished_request_state(["finished"])
    assert released == ["finished"]
    assert set(worker._req_cache) == {"live"}
    assert worker._last_tokens == {"live": [8]}


def test_executor_passes_worker_build_options(metadata_checkpoint, binding_metadata, monkeypatch):
    from pypto_serving.model.deepseek_v41.npu_executor import DeepSeekV41PyptoExecutor

    path, _ = metadata_checkpoint
    received = []
    monkeypatch.setattr(
        "pypto_serving.model.deepseek_v41.npu_runner.V41ModelRunner.preflight",
        lambda runner: received.append(runner.build_options) or 4,
    )
    record = SimpleNamespace(
        runtime=RuntimeConfig(kv_cache_groups=binding_metadata.cache_groups),
        runtime_model=SimpleNamespace(extra={"model_dir": str(path)}),
    )
    executor = DeepSeekV41PyptoExecutor(
        bindings=binding_metadata, pypto_build_dir="worker-7-build", use_compile_cache=True,
    )
    assert executor.register_model("text", record) == 4
    assert received[0].platform == "a5"
    assert received[0].pypto_build_dir == "worker-7-build"
    assert received[0].use_compile_cache

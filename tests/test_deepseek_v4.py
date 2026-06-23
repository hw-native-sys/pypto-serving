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
import json
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import python.cli.main as cli
from examples.model.deepseek_v4.runner import npu_executor, weight_loader
from examples.model.deepseek_v4.runner.npu_runner import (
    DeepSeekV4CacheLayout,
    DeepSeekV4CacheManager,
    DeepSeekV4CompiledKernels,
    DeepSeekV4InputBuilder,
    DeepSeekV4L3Callable,
    DeepSeekV4LayerCache,
    DeepSeekV4LayerCacheSnapshot,
    DeepSeekV4LayerPlan,
    DeepSeekV4ModelRunner,
    build_deepseek_v4_layer_plan,
)
from examples.model.deepseek_v4.runner.weight_loader import (
    DeepSeekV4WeightStore,
    deepseek_v4_layer_core_weight_names,
    deepseek_v4_hadamard_idx,
    deepseek_v4_local_expert_ids,
    deepseek_v4_routed_expert_weight_names,
    deepseek_v4_startup_weight_names,
    pack_deepseek_v4_lm_head_weight,
    pack_deepseek_v4_layer_weights,
)
from python.core import model_loader
from python.core.model_loader import ModelLoader
from python.core.types import DecodeBatch, PrefillBatch, RuntimeConfig


def test_cli_selects_deepseek_executor_and_forces_prefix_cache_off(tmp_path):
    model_dir = _write_deepseek_model_dir(tmp_path)
    args = cli.build_parser().parse_args(
        [
            "--model", str(model_dir),
            "--devices", "0,1,2,3,4,5,6,7",
            "--dp", "1",
            "--tp", "8",
            "--block-size", "128",
            "--max-model-len", "260",
            "--dtype", "int8",
        ]
    )

    config = cli.build_serving_engine_config(args)

    assert config.executor_cls == "PyptoDeepSeekV4Executor"
    assert config.device_ids == (0, 1, 2, 3, 4, 5, 6, 7)
    assert config.parallel_config.replica_device_groups == ((0, 1, 2, 3, 4, 5, 6, 7),)
    assert config.runtime_config.page_size == 128
    assert config.runtime_config.weight_dtype == "int8"
    assert config.enable_prefix_cache is False


def test_cli_rejects_deepseek_non_w8a8_checkpoint(tmp_path):
    model_dir = _write_deepseek_model_dir(tmp_path, quant_method="fp8")
    args = cli.build_parser().parse_args(
        [
            "--model", str(model_dir),
            "--devices", "0,1,2,3,4,5,6,7",
            "--dp", "1",
            "--tp", "8",
            "--block-size", "128",
        ]
    )

    with pytest.raises(ValueError, match="compressed-tensors"):
        cli.build_serving_engine_config(args)


def test_cli_rejects_deepseek_non_8_way_topology(tmp_path):
    model_dir = _write_deepseek_model_dir(tmp_path)
    args = cli.build_parser().parse_args(
        [
            "--model", str(model_dir),
            "--devices", "0,1,2,3",
            "--dp", "1",
            "--tp", "4",
            "--block-size", "128",
        ]
    )

    with pytest.raises(ValueError, match="--dp 1 --tp 8"):
        cli.build_serving_engine_config(args)


def test_cli_rejects_deepseek_context_beyond_decode_state_capacity(tmp_path):
    model_dir = _write_deepseek_model_dir(tmp_path)
    args = cli.build_parser().parse_args(
        [
            "--model", str(model_dir),
            "--devices", "0,1,2,3,4,5,6,7",
            "--dp", "1",
            "--tp", "8",
            "--block-size", "128",
            "--max-model-len", "512",
        ]
    )

    with pytest.raises(ValueError, match="--max-model-len 260"):
        cli.build_serving_engine_config(args)


def test_deepseek_loader_keeps_w8a8_weights_lazy(tmp_path, monkeypatch):
    model_dir = _write_deepseek_model_dir(tmp_path)
    monkeypatch.setattr(
        model_loader.TransformersTokenizerAdapter,
        "from_pretrained",
        lambda *args, **kwargs: _Tokenizer(),
    )

    loaded = ModelLoader().load(
        model_id="dsv4",
        model_dir=str(model_dir),
        runtime_config=RuntimeConfig(page_size=128, max_batch_size=16, max_seq_len=256, weight_dtype="int8"),
    )

    assert loaded.config.architecture == "DeepseekV4ForCausalLM"
    assert loaded.config.head_dim == 512
    assert loaded.runtime_model.layers == []
    assert loaded.runtime_model.embed_tokens.numel() == 0
    assert loaded.runtime_model.extra["family"] == "deepseek_v4"
    assert loaded.runtime_model.extra["checkpoint_format"] == "w8a8-compressed-tensors"
    assert "layers.0.attn.wq_b.scale" in loaded.runtime_model.extra["weight_map"]
    assert "layers.2.attn.indexer.wq_b.scale" in loaded.runtime_model.extra["weight_map"]
    assert "layers.3.attn.compressor.wkv.weight" in loaded.runtime_model.extra["weight_map"]
    assert "layers.3.ffn.gate.bias" in loaded.runtime_model.extra["weight_map"]


def test_deepseek_compile_attaches_lazy_weight_store_without_opening_shards(tmp_path, monkeypatch):
    model_dir = _write_deepseek_model_dir(tmp_path)
    kernel_dir = _write_deepseek_kernel_dir(tmp_path, lm_head_tp_size=8)
    monkeypatch.setattr(
        model_loader.TransformersTokenizerAdapter,
        "from_pretrained",
        lambda *args, **kwargs: _Tokenizer(),
    )
    opened: list[Path] = []

    def _fail_open(path: Path, device: str):
        opened.append(path)
        raise AssertionError(f"unexpected safetensors open on {device}: {path}")

    monkeypatch.setattr(weight_loader, "_default_safe_open", _fail_open)
    monkeypatch.setattr(npu_executor, "_find_pypto_lib_deepseek_v4_dir", lambda *args, **kwargs: kernel_dir)
    loaded = ModelLoader().load(
        model_id="dsv4",
        model_dir=str(model_dir),
        runtime_config=RuntimeConfig(page_size=128, max_batch_size=16, max_seq_len=256, weight_dtype="int8"),
    )
    executor = npu_executor.DeepSeekV4PyptoExecutor(platform="a2a3sim", device_ids=tuple(range(8)))

    compiled = executor._compile_model(loaded.runtime_model)

    assert opened == []
    assert isinstance(compiled.weight_store, DeepSeekV4WeightStore)
    assert compiled.weight_store.filename_for("head.weight") == "model-00001-of-00001.safetensors"
    assert compiled.weight_store.device == "cpu"
    assert compiled.layer_plan[0].attention_kind == "swa"
    assert compiled.layer_plan[2].attention_kind == "csa"
    assert compiled.layer_plan[2].include_tid2eid is True
    assert compiled.layer_plan[3].attention_kind == "hca"
    assert compiled.layer_plan[3].include_gate_bias is True


def test_deepseek_compile_builds_one_runtime_scalar_layer_callable(tmp_path, monkeypatch):
    model_dir = _write_deepseek_model_dir(tmp_path)
    kernel_dir = _write_deepseek_kernel_dir(tmp_path, lm_head_tp_size=8)
    monkeypatch.setattr(
        model_loader.TransformersTokenizerAdapter,
        "from_pretrained",
        lambda *args, **kwargs: _Tokenizer(),
    )
    monkeypatch.setattr(npu_executor, "_find_pypto_lib_deepseek_v4_dir", lambda *args, **kwargs: kernel_dir)
    loaded = ModelLoader().load(
        model_id="dsv4",
        model_dir=str(model_dir),
        runtime_config=RuntimeConfig(page_size=128, max_batch_size=16, max_seq_len=256, weight_dtype="int8"),
    )
    compiled_args: dict[str, tuple[object, ...]] = {}

    class _PrefillModule:
        l3_prefill_layer = object()

    class _DecodeModule:
        l3_decode_layer = object()

    class _LmHeadModule:
        l3_lm_head = object()

    class _FlashConfig:
        hidden_size = 4096
        num_attention_heads = 64
        head_dim = 512
        qk_rope_head_dim = 64
        q_lora_rank = 1024
        o_lora_rank = 1024
        o_groups = 8
        mix_hc = 24
        hc_dim = 16384
        max_position_embeddings = 8192
        moe_intermediate_size = 2048
        n_routed_experts = 256
        num_experts_per_tok = 6
        index_n_heads = 64
        index_head_dim = 128

    class _ConfigModule:
        FLASH = _FlashConfig

    compiled_names: list[str] = []

    def _fake_compile(self, name, jit_fn, dummy_args):
        compiled_names.append(name)
        compiled_args[name] = tuple(dummy_args)
        return DeepSeekV4L3Callable(compiled=object(), name=name)

    monkeypatch.setattr(
        npu_executor.DeepSeekV4PyptoExecutor,
        "_load_kernel_modules",
        lambda self: {
            "config": _ConfigModule,
            "prefill_layer": _PrefillModule,
            "decode_layer": _DecodeModule,
            "lm_head": _LmHeadModule,
            "rope_tables": object(),
        },
    )
    monkeypatch.setattr(npu_executor.DeepSeekV4PyptoExecutor, "_compile_l3_callable", _fake_compile)
    monkeypatch.setattr(
        npu_executor.DeepSeekV4PyptoExecutor,
        "_build_rope_tables",
        lambda self, rope_tables_module, config_module: (torch.empty(1), torch.empty(1)),
    )
    executor = npu_executor.DeepSeekV4PyptoExecutor(
        platform="a2a3sim",
        device_ids=tuple(range(8)),
        compile_kernels=True,
    )

    executor._compile_model(loaded.runtime_model)

    assert compiled_names == ["deepseek_v4_prefill", "deepseek_v4_decode", "deepseek_v4_lm_head"]
    assert len(compiled_args["deepseek_v4_prefill"]) == 78
    assert len(compiled_args["deepseek_v4_decode"]) == 77
    assert len(compiled_args["deepseek_v4_lm_head"]) == 3
    assert isinstance(compiled_args["deepseek_v4_prefill"][-2], ctypes.c_int32)
    assert isinstance(compiled_args["deepseek_v4_prefill"][-1], ctypes.c_int32)
    assert isinstance(compiled_args["deepseek_v4_decode"][-2], ctypes.c_int32)
    assert isinstance(compiled_args["deepseek_v4_decode"][-1], ctypes.c_int32)
    assert compiled_args["deepseek_v4_prefill"][0].shape == (8, 128, 4, 4096)
    assert compiled_args["deepseek_v4_decode"][0].shape == (8, 64, 4, 4096)
    decode_order = npu_executor._DECODE_TENSOR_ORDER
    assert compiled_args["deepseek_v4_decode"][decode_order.index("hca_compress_state")].shape == (8, 2048, 8, 1024)
    assert compiled_args["deepseek_v4_decode"][decode_order.index("csa_compress_state")].shape == (8, 2080, 4, 2048)
    assert compiled_args["deepseek_v4_decode"][decode_order.index("csa_inner_compress_state")].shape == (
        8,
        2080,
        4,
        512,
    )
    assert compiled_args["deepseek_v4_lm_head"][1].shape == (8, 16384, 4096)


def test_deepseek_layer_plan_tracks_attention_and_router_metadata():
    plan = build_deepseek_v4_layer_plan(
        compress_ratios=_deepseek_flash_compress_ratios(),
        num_hidden_layers=43,
        num_hash_layers=3,
    )

    assert [(layer.attention_kind, layer.include_tid2eid) for layer in plan[:5]] == [
        ("swa", True),
        ("swa", True),
        ("csa", True),
        ("hca", False),
        ("csa", False),
    ]


def test_deepseek_compile_rejects_non_8_way_lm_head_kernel(tmp_path, monkeypatch):
    model_dir = _write_deepseek_model_dir(tmp_path)
    kernel_dir = _write_deepseek_kernel_dir(tmp_path, lm_head_tp_size=2)
    monkeypatch.setattr(
        model_loader.TransformersTokenizerAdapter,
        "from_pretrained",
        lambda *args, **kwargs: _Tokenizer(),
    )
    monkeypatch.setattr(npu_executor, "_find_pypto_lib_deepseek_v4_dir", lambda *args, **kwargs: kernel_dir)
    loaded = ModelLoader().load(
        model_id="dsv4",
        model_dir=str(model_dir),
        runtime_config=RuntimeConfig(page_size=128, max_batch_size=16, max_seq_len=256, weight_dtype="int8"),
    )
    executor = npu_executor.DeepSeekV4PyptoExecutor(platform="a2a3sim", device_ids=tuple(range(8)))

    with pytest.raises(ValueError, match="LM_HEAD_TP_SIZE=2 expected 8"):
        executor._compile_model(loaded.runtime_model)


def test_deepseek_kernel_contract_accepts_config_named_tp_size(tmp_path):
    kernel_dir = _write_deepseek_kernel_dir(tmp_path, lm_head_tp_size=8, use_config_constant=True)
    executor = npu_executor.DeepSeekV4PyptoExecutor.__new__(npu_executor.DeepSeekV4PyptoExecutor)
    executor._kernel_dir = kernel_dir

    executor._validate_kernel_contract(DeepSeekV4CacheLayout())


def test_deepseek_kernel_contract_rejects_config_dimension_mismatch(tmp_path):
    kernel_dir = _write_deepseek_kernel_dir(tmp_path, lm_head_tp_size=8, block_size=64)
    executor = npu_executor.DeepSeekV4PyptoExecutor.__new__(npu_executor.DeepSeekV4PyptoExecutor)
    executor._kernel_dir = kernel_dir

    with pytest.raises(ValueError, match="BLOCK_SIZE=64 expected 128"):
        executor._validate_kernel_contract(DeepSeekV4CacheLayout())


def test_deepseek_hc_input_builder_shapes_prefill_and_decode():
    builder = DeepSeekV4InputBuilder(layout=DeepSeekV4CacheLayout(), hidden_size=4)

    prefill = builder.prefill_x_hc(torch.arange(12, dtype=torch.bfloat16).reshape(3, 4), actual_tokens=3)
    decode = builder.decode_x_hc(torch.arange(8, dtype=torch.bfloat16).reshape(2, 4), actual_batch=2)

    assert prefill.shape == (8, 128, 4, 4)
    assert prefill[0, 0, 0].tolist() == [0, 1, 2, 3]
    assert prefill[7, 2, 3].tolist() == [8, 9, 10, 11]
    assert torch.count_nonzero(prefill[:, 3:]) == 0
    assert decode.shape == (8, 64, 4, 4)
    assert decode[0, 0, 0].tolist() == [0, 1, 2, 3]
    assert decode[0, 1, 3].tolist() == [0, 1, 2, 3]
    assert decode[7, 2, 0].tolist() == [4, 5, 6, 7]
    assert decode[7, 3, 3].tolist() == [4, 5, 6, 7]
    assert decode[0, 4, 0].tolist() == [0, 1, 2, 3]
    assert decode[7, 5, 3].tolist() == [0, 1, 2, 3]
    assert torch.equal(decode[:, 4:], decode[:, 0:2].repeat(1, 30, 1, 1))


def test_deepseek_layout_rejects_context_beyond_decode_state_capacity():
    model = _runtime_model_for_embeddings()

    with pytest.raises(ValueError, match="max_seq_len=260"):
        DeepSeekV4CacheLayout().validate_runtime(
            model.config,
            RuntimeConfig(page_size=128, max_batch_size=1, max_seq_len=261, weight_dtype="int8"),
            tuple(range(8)),
        )


def test_deepseek_layer_plan_tracks_attention_and_gate_modes():
    plan = build_deepseek_v4_layer_plan(
        compress_ratios=[0, 0, 4, 128, 4],
        num_hidden_layers=5,
        num_hash_layers=3,
    )

    assert [layer.attention_kind for layer in plan] == ["swa", "swa", "csa", "hca", "csa"]
    assert [layer.include_tid2eid for layer in plan] == [True, True, True, False, False]
    assert [layer.include_gate_bias for layer in plan] == [False, False, False, True, True]


def test_deepseek_weight_store_groups_requested_reads_by_shard(tmp_path):
    weight_map = {
        "a": "one.safetensors",
        "b": "one.safetensors",
        "c": "two.safetensors",
    }
    for filename in set(weight_map.values()):
        (tmp_path / filename).touch()
    opened: list[tuple[str, str]] = []
    reads: list[tuple[str, str]] = []
    tensors = {
        "a": torch.tensor([1]),
        "c": torch.tensor([3]),
    }

    class _Reader:
        def __init__(self, filename: str) -> None:
            self.filename = filename

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get_tensor(self, name: str) -> torch.Tensor:
            reads.append((self.filename, name))
            return tensors[name]

    def _open(path: Path, device: str):
        opened.append((path.name, device))
        return _Reader(path.name)

    store = DeepSeekV4WeightStore(model_dir=tmp_path, weight_map=weight_map, safe_open_fn=_open)

    loaded = store.load_many(["c", "a"])

    assert list(loaded) == ["c", "a"]
    assert loaded["c"].item() == 3
    assert loaded["a"].item() == 1
    assert opened == [("two.safetensors", "cpu"), ("one.safetensors", "cpu")]
    assert reads == [("two.safetensors", "c"), ("one.safetensors", "a")]


def test_deepseek_weight_store_reads_real_safetensors_by_name(tmp_path):
    from safetensors.torch import save_file

    save_file(
        {
            "embed.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
            "head.weight": torch.ones(2, 2),
        },
        str(tmp_path / "global.safetensors"),
    )
    store = DeepSeekV4WeightStore(
        model_dir=tmp_path,
        weight_map={
            "embed.weight": "global.safetensors",
            "head.weight": "global.safetensors",
        },
    )

    loaded = store.load_tensor("embed.weight")

    assert loaded.tolist() == [[0.0, 1.0], [2.0, 3.0]]


def test_deepseek_executor_lazily_loads_and_caches_embeddings(tmp_path):
    from safetensors.torch import save_file

    save_file(
        {"embed.weight": torch.arange(24, dtype=torch.float32).reshape(6, 4)},
        str(tmp_path / "embed.safetensors"),
    )
    open_count = 0
    store = DeepSeekV4WeightStore(
        model_dir=tmp_path,
        weight_map={"embed.weight": "embed.safetensors"},
    )
    original_open = store._safe_open_fn

    def _counting_open(path: Path, device: str):
        nonlocal open_count
        open_count += 1
        return original_open(path, device)

    store._safe_open_fn = _counting_open
    executor = npu_executor.DeepSeekV4PyptoExecutor.__new__(npu_executor.DeepSeekV4PyptoExecutor)
    executor._compiled = {
        "dsv4": DeepSeekV4CompiledKernels(
            layout=DeepSeekV4CacheLayout(),
            model_dir=str(tmp_path),
            weight_map=store.weight_map,
            weight_store=store,
            compress_ratios=tuple([0] * 44),
            layer_plan=build_deepseek_v4_layer_plan(
                compress_ratios=tuple([0] * 44),
                num_hidden_layers=43,
                num_hash_layers=3,
            ),
            kernel_dir=str(tmp_path),
        )
    }
    executor._embedding_cache = {}
    model = _runtime_model_for_embeddings()

    first = executor.lookup_embeddings(model, torch.tensor([1, 3], dtype=torch.long))
    second = executor.lookup_embeddings(model, torch.tensor([[2, 4]], dtype=torch.long))

    assert first.tolist() == [[4.0, 5.0, 6.0, 7.0], [12.0, 13.0, 14.0, 15.0]]
    assert second.shape == (1, 2, 4)
    assert second[0, 1].tolist() == [16.0, 17.0, 18.0, 19.0]
    assert open_count == 1


def test_deepseek_weight_store_loads_rank_local_experts(tmp_path):
    core_names = deepseek_v4_layer_core_weight_names(0, include_tid2eid=True)
    local_experts = deepseek_v4_local_expert_ids(rank=1, ranks=4, n_routed_experts=8)
    expert_names = deepseek_v4_routed_expert_weight_names(0, local_experts)
    weight_map = {name: "layer.safetensors" for name in (*core_names, *expert_names)}
    (tmp_path / "layer.safetensors").touch()
    reads: list[str] = []

    class _Reader:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get_tensor(self, name: str) -> torch.Tensor:
            reads.append(name)
            return torch.tensor([len(reads)])

    store = DeepSeekV4WeightStore(model_dir=tmp_path, weight_map=weight_map, safe_open_fn=lambda path, device: _Reader())

    loaded = store.load_rank_layer_weights(
        0,
        rank=1,
        ranks=4,
        n_routed_experts=8,
        include_tid2eid=True,
    )

    assert local_experts == (2, 3)
    assert set(loaded) == set(weight_map)
    assert all(".experts.2." in name or ".experts.3." in name for name in expert_names)
    assert not any(".experts.0." in name or ".experts.1." in name for name in loaded)


def test_deepseek_weight_store_packs_lm_head_into_8_tp_shards(tmp_path):
    from safetensors.torch import save_file

    save_file(
        {
            "embed.weight": torch.arange(64, dtype=torch.float32).reshape(16, 4),
            "norm.weight": torch.arange(4, dtype=torch.float32),
            "head.weight": torch.arange(64, dtype=torch.float32).reshape(16, 4) + 100,
            "hc_head_fn": torch.zeros((4, 16), dtype=torch.float32),
            "hc_head_scale": torch.ones((1,), dtype=torch.float32),
            "hc_head_base": torch.zeros((4,), dtype=torch.float32),
        },
        str(tmp_path / "global.safetensors"),
    )
    store = DeepSeekV4WeightStore(
        model_dir=tmp_path,
        weight_map={
            "embed.weight": "global.safetensors",
            "norm.weight": "global.safetensors",
            "head.weight": "global.safetensors",
            "hc_head_fn": "global.safetensors",
            "hc_head_scale": "global.safetensors",
            "hc_head_base": "global.safetensors",
        },
    )

    global_weights = store.load_packed_global_weights(ranks=8)

    assert global_weights.lm_head_layout.vocab_per_rank == 2
    assert global_weights.lm_head_layout.padded_vocab_per_rank == 512
    assert global_weights.lm_head_weight.shape == (8, 512, 4)
    assert global_weights.lm_head_weight[0, :2].tolist() == [[100.0, 101.0, 102.0, 103.0], [104.0, 105.0, 106.0, 107.0]]
    assert global_weights.lm_head_weight[1, :2].tolist() == [[108.0, 109.0, 110.0, 111.0], [112.0, 113.0, 114.0, 115.0]]
    assert torch.count_nonzero(global_weights.lm_head_weight[:, 2:]) == 0


def test_deepseek_lm_head_packer_rejects_uneven_vocab():
    with pytest.raises(ValueError, match="divide evenly"):
        pack_deepseek_v4_lm_head_weight(torch.zeros((17, 4)), ranks=8)


def test_deepseek_layer_packer_transposes_and_stacks_rank_local_experts():
    raw = _synthetic_layer_raw(layer_id=0, n_experts=4)

    packed = pack_deepseek_v4_layer_weights(
        0,
        raw,
        ranks=2,
        n_routed_experts=4,
        compress_ratio=4,
        include_tid2eid=False,
        include_gate_bias=True,
    )

    assert packed.tensors["wq_a"].shape == (2, 4, 2)
    assert packed.tensors["wq_a"][0].tolist() == raw["layers.0.attn.wq_a.weight"].t().tolist()
    assert packed.tensors["wo_a"].shape == (2, 8, 2, 4)
    assert packed.tensors["csa_cmp_wkv"].shape == (2, 4, 2)
    assert packed.tensors["hca_cmp_wkv"].shape == (2, 4096, 512)
    assert torch.count_nonzero(packed.tensors["hca_cmp_wkv"]) == 0
    assert packed.tensors["gate_bias"].shape == (2, 4)
    assert packed.tensors["tid2eid"].shape == (2, 129280, 6)
    assert packed.tensors["routed_w1"].shape == (2, 2, 2, 4)
    assert packed.tensors["routed_w1"][0, 0].tolist() == raw["layers.0.ffn.experts.0.w1.weight"].tolist()
    assert packed.tensors["routed_w1"][1, 0].tolist() == raw["layers.0.ffn.experts.2.w1.weight"].tolist()
    assert torch.equal(packed.tensors["csa_hadamard_idx"][0], deepseek_v4_hadamard_idx())


def test_deepseek_cache_slots_tables_and_mappings():
    manager = DeepSeekV4CacheManager(layout=DeepSeekV4CacheLayout())

    assert manager.allocate("req-a") == 0
    assert manager.allocate("req-b") == 1
    assert manager.allocate("req-a") == 0

    table = manager.block_table([1], max_blocks=64)
    assert table.shape == (1, 64)
    assert table[0, 0].item() == 64
    assert table[0, 63].item() == 127

    cmp_mapping = manager.slot_mapping([1], [[0, 4, 256]], max_blocks=64, compress_ratio=4)
    base = 1 * 64 * 128
    assert cmp_mapping.tolist() == [[base, base + 1, base + 64]]

    hca_state_mapping = manager.slot_mapping(
        [1],
        [[0, 128, 256]],
        max_blocks=64,
        block_size=8,
        compress_ratio=128,
    )
    assert hca_state_mapping.tolist() == [[1 * 64 * 8, 1 * 64 * 8 + 1, 1 * 64 * 8 + 2]]

    manager.release(["req-a"])
    assert manager.allocate("req-c") == 0


def test_deepseek_prepare_prefill_inputs_maps_chunk_and_sparse_tables():
    runner, model = _runner_for_prepared_inputs()
    embeddings = torch.arange(12, dtype=torch.bfloat16).reshape(1, 3, 4)

    prepared = runner.prepare_prefill_inputs(
        model,
        PrefillBatch(
            request_ids=["req-a"],
            token_ids=torch.tensor([[10, 11, 12]], dtype=torch.long),
            input_embeddings=embeddings,
            seq_lens=torch.tensor([129], dtype=torch.int32),
            positions=torch.tensor([[126, 127, 128]], dtype=torch.long),
        ),
    )

    assert prepared.request_id == "req-a"
    assert prepared.slot == 0
    assert prepared.actual_tokens == 3
    assert prepared.x_hc.shape == (8, 128, 4, 4)
    assert prepared.cmp_block_table.shape == (8, 2)
    assert prepared.idx_block_table.shape == (8, 2)
    assert prepared.position_ids.shape == (8, 128)
    assert prepared.position_ids[0, :4].tolist() == [126, 127, 128, 3]
    assert prepared.input_ids[0, :4].tolist() == [10, 11, 12, 10]
    assert prepared.ori_slot_mapping.shape == (8, 128)
    assert prepared.ori_slot_mapping[0, :3].tolist() == [126, 127, 0]
    assert prepared.ori_slot_mapping[0, 3].item() == -1
    assert prepared.hca_cmp_slot_mapping.shape == (8, 128)
    assert prepared.hca_cmp_slot_mapping[0, :3].tolist() == [-1, 0, -1]
    assert prepared.hca_cmp_slot_mapping[0, 3].item() == -1
    assert prepared.csa_cmp_slot_mapping.shape == (8, 128)
    assert prepared.csa_cmp_slot_mapping[0, :3].tolist() == [-1, 31, -1]
    assert prepared.csa_cmp_slot_mapping[0, 3].item() == -1
    assert prepared.csa_idx_slot_mapping.shape == (8, 128)
    assert prepared.csa_idx_slot_mapping[0, :3].tolist() == [-1, 31, -1]
    assert prepared.csa_idx_slot_mapping[0, 3].item() == -1
    assert prepared.hca_state_slot_mapping.shape == (8, 128)
    assert prepared.hca_state_slot_mapping[0, :3].tolist() == [126, 127, 128]
    assert prepared.hca_state_slot_mapping[0, 3].item() == -1
    assert prepared.csa_state_slot_mapping.shape == (8, 128)
    assert prepared.csa_state_slot_mapping[0, :3].tolist() == [126, 127, 128]
    assert prepared.csa_state_slot_mapping[0, 3].item() == -1
    assert prepared.csa_inner_state_slot_mapping.shape == (8, 128)
    assert prepared.csa_inner_state_slot_mapping[0, :3].tolist() == [126, 127, 128]
    assert prepared.csa_inner_state_slot_mapping[0, 3].item() == -1
    sparse4, lens4 = prepared.sparse_inputs_for_ratio(4)
    sparse0, lens0 = prepared.sparse_inputs_for_ratio(0)
    assert sparse4.shape == (8, 128, 640)
    assert lens4[0, 1].item() > lens0[0, 1].item()
    assert sparse4[0, 1, 0].item() == 0
    assert sparse4[0, 1, 127].item() == 129
    assert sparse4[0, 1, 128].item() == 256


def test_deepseek_prepare_decode_inputs_uses_scratch_slots_for_fixed_rows():
    runner, model = _runner_for_prepared_inputs()

    prepared = runner.prepare_decode_inputs(
        model,
        DecodeBatch(
            request_ids=["req-a", "req-b"],
            token_ids=torch.tensor([[5], [9]], dtype=torch.long),
            hidden_states=torch.arange(8, dtype=torch.bfloat16).reshape(2, 4),
            seq_lens=torch.tensor([128, 5], dtype=torch.int32),
        ),
    )

    assert prepared.actual_batch == 2
    assert prepared.slots == (0, 1)
    assert prepared.kernel_slots[:4] == (0, 1, 2, 3)
    assert len(prepared.kernel_slots) == 32
    assert prepared.x_hc.shape == (8, 64, 4, 4)
    assert prepared.x_hc[0, 0, 0].tolist() == [0, 1, 2, 3]
    assert prepared.x_hc[0, 1, 0].tolist() == [0, 1, 2, 3]
    assert prepared.x_hc[0, 2, 0].tolist() == [4, 5, 6, 7]
    assert prepared.x_hc[0, 3, 0].tolist() == [4, 5, 6, 7]
    assert prepared.x_hc[0, 4, 0].tolist() == [0, 1, 2, 3]
    assert prepared.input_ids[0, :6].tolist() == [5, 5, 9, 9, 4, 5]
    assert prepared.position_ids[0, :6].tolist() == [127, 128, 4, 5, 127, 128]
    assert prepared.kv_seq_lens[0, :4].tolist() == [129, 6, 129, 129]
    assert prepared.block_table.shape == (8, 32, 1)
    assert prepared.cmp_block_table.shape == (8, 32, 64)
    assert prepared.ori_slot_mapping[0, :6].tolist() == [127, 0, 132, 133, 383, 256]
    assert prepared.hca_cmp_slot_mapping[0, :6].tolist() == [0, -1, -1, -1, 16384, -1]
    assert prepared.csa_cmp_slot_mapping[0, :6].tolist() == [31, -1, -1, -1, 16415, -1]
    assert prepared.csa_idx_slot_mapping[0, :6].tolist() == [31, -1, -1, -1, 16415, -1]
    assert prepared.csa_state_slot_mapping[0, :6].tolist() == [127, 128, 264, 265, 647, 648]


def test_deepseek_stage_decode_inputs_uses_shared_buffers():
    runner, model = _runner_for_prepared_inputs()
    prepared = runner.prepare_decode_inputs(
        model,
        DecodeBatch(
            request_ids=["req-a"],
            token_ids=torch.tensor([[5]], dtype=torch.long),
            hidden_states=torch.arange(4, dtype=torch.bfloat16).reshape(1, 4),
            seq_lens=torch.tensor([128], dtype=torch.int32),
        ),
    )

    staged = runner._stage_decode_inputs(prepared)

    assert staged.x_hc.is_shared()
    assert runner._decode_buffers is not None
    assert runner._decode_buffers.x_hc_b.is_shared()
    for name in (
        "input_ids",
        "position_ids",
        "kv_seq_lens",
        "block_table",
        "ori_slot_mapping",
        "cmp_block_table",
        "idx_block_table",
        "hca_compress_state_block_table",
        "csa_compress_state_block_table",
        "csa_inner_compress_state_block_table",
        "hca_cmp_slot_mapping",
        "hca_state_slot_mapping",
        "csa_cmp_slot_mapping",
        "csa_idx_slot_mapping",
        "csa_state_slot_mapping",
        "csa_inner_state_slot_mapping",
    ):
        assert getattr(staged, name).is_shared()


def test_deepseek_l3_dispatch_rejects_non_shared_tensor_before_worker_start():
    runner, _model = _runner_for_prepared_inputs()

    with pytest.raises(TypeError, match="before the L3 worker starts"):
        runner._run_l3(DeepSeekV4L3Callable(compiled=object(), name="fake"), torch.zeros(1))


def test_deepseek_l3_scalars_are_runtime_python_ints():
    runner, _model = _runner_for_prepared_inputs()

    value = runner._int32_scalar(7)

    assert isinstance(value, int)
    assert value == 7


def test_deepseek_cache_replicates_decode_padding_rows():
    active = torch.tensor([[10, 11], [20, 21]], dtype=torch.int32)

    padded = DeepSeekV4CacheManager.replicate_first_row(active, actual_rows=2, kernel_rows=4)

    assert padded.tolist() == [[10, 11], [20, 21], [10, 11], [10, 11]]


def test_deepseek_decode_work_cache_loads_snapshot_into_kernel_slots():
    layout = DeepSeekV4CacheLayout(
        ranks=1,
        decode_batch=3,
        decode_seq=2,
        decode_tokens=6,
        ori_max_blocks=1,
        cmp_max_blocks=1,
    )
    layer = DeepSeekV4LayerPlan(
        layer_id=0,
        compress_ratio=0,
        attention_kind="swa",
        include_tid2eid=True,
        include_gate_bias=False,
    )
    compiled = DeepSeekV4CompiledKernels(
        layout=layout,
        model_dir="",
        weight_map={},
        weight_store=None,
        compress_ratios=(),
        layer_plan=(layer,),
        kernel_dir="",
    )
    runner = DeepSeekV4ModelRunner(compiled=compiled)
    runner._decode_work_cache = DeepSeekV4LayerCache(
        kv_cache=torch.full((1, 3, 1, 1, 1), -1, dtype=torch.bfloat16),
        cmp_kv=torch.full((1, 3, 1, 1, 1), -2, dtype=torch.bfloat16),
        idx_kv_cache=torch.empty(0),
        hca_compress_state=torch.empty(0),
        csa_compress_state=torch.empty(0),
        csa_inner_compress_state=torch.empty(0),
    )
    runner._prefill_cache_snapshots[layer.layer_id] = DeepSeekV4LayerCacheSnapshot(
        {
            "kv_cache": torch.tensor([[[[[7.0]]]]], dtype=torch.bfloat16),
            "cmp_kv": torch.tensor([[[[[8.0]]]]], dtype=torch.bfloat16),
        }
    )

    runner._load_decode_work_cache(layer, slot=0, kernel_slots=(0, 2))

    assert runner._decode_work_cache.kv_cache.flatten().tolist() == [7.0, -1.0, 7.0]
    assert runner._decode_work_cache.cmp_kv.flatten().tolist() == [8.0, -2.0, 8.0]


def _write_deepseek_model_dir(tmp_path: Path, *, quant_method: str = "compressed-tensors") -> Path:
    model_dir = tmp_path / "dsv4-flash-w8a8"
    model_dir.mkdir()
    compress_ratios = _deepseek_flash_compress_ratios()
    config = {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "vocab_size": 129280,
        "hidden_size": 4096,
        "moe_intermediate_size": 2048,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "num_hidden_layers": 43,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "max_position_embeddings": 1048576,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "torch_dtype": "bfloat16",
        "compress_ratios": compress_ratios,
        "quantization_config": {
            "quant_method": quant_method,
            "format": "int-quantized",
            "quantization_status": "compressed",
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config))
    weight_names = deepseek_v4_startup_weight_names(
        43,
        n_routed_experts=256,
        compress_ratios=compress_ratios,
        num_hash_layers=3,
    )
    index = {"weight_map": {name: "model-00001-of-00001.safetensors" for name in weight_names}}
    (model_dir / "model.safetensors.index.json").write_text(json.dumps(index))
    return model_dir


def _deepseek_flash_compress_ratios() -> list[int]:
    return [0, 0, *(4 if layer_id % 2 == 0 else 128 for layer_id in range(2, 43)), 0]


def _write_deepseek_kernel_dir(
    tmp_path: Path,
    *,
    lm_head_tp_size: int,
    use_config_constant: bool = False,
    block_size: int = 128,
) -> Path:
    kernel_dir = tmp_path / f"deepseek-v4-kernels-tp{lm_head_tp_size}"
    kernel_dir.mkdir()
    (kernel_dir / "prefill_layer.py").write_text("")
    (kernel_dir / "decode_layer.py").write_text("")
    (kernel_dir / "config.py").write_text(
        "\n".join(
            [
                f"BLOCK_SIZE = {block_size}",
                "DECODE_BATCH = 32",
                "DECODE_SEQ = 2",
                "DECODE_TOKENS = DECODE_BATCH * DECODE_SEQ",
                "PREFILL_BATCH = 1",
                "PREFILL_SEQ = 128",
                "EP_WORLD_SIZE = 8",
                f"LM_HEAD_TP_SIZE = {lm_head_tp_size}",
                "",
            ]
        )
    )
    if use_config_constant:
        (kernel_dir / "lm_head.py").write_text("TP_SIZE = LM_HEAD_TP_SIZE\n")
    else:
        (kernel_dir / "lm_head.py").write_text(f"TP_SIZE = {lm_head_tp_size}\n")
    return kernel_dir


def _synthetic_layer_raw(*, layer_id: int, n_experts: int) -> dict[str, torch.Tensor]:
    prefix = f"layers.{layer_id}"
    raw = {
        f"{prefix}.hc_attn_fn": torch.arange(4, dtype=torch.float32).reshape(1, 4),
        f"{prefix}.hc_attn_scale": torch.arange(3, dtype=torch.float32),
        f"{prefix}.hc_attn_base": torch.arange(1, dtype=torch.float32),
        f"{prefix}.attn_norm.weight": torch.arange(4, dtype=torch.bfloat16),
        f"{prefix}.attn.wq_a.weight": torch.arange(8, dtype=torch.bfloat16).reshape(2, 4),
        f"{prefix}.attn.wq_b.weight": torch.arange(12, dtype=torch.int8).reshape(6, 2),
        f"{prefix}.attn.wq_b.scale": torch.arange(6, dtype=torch.float32),
        f"{prefix}.attn.wkv.weight": torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
        f"{prefix}.attn.q_norm.weight": torch.arange(2, dtype=torch.bfloat16),
        f"{prefix}.attn.kv_norm.weight": torch.arange(3, dtype=torch.bfloat16),
        f"{prefix}.attn.attn_sink": torch.arange(2, dtype=torch.float32),
        f"{prefix}.attn.wo_a.weight": torch.arange(64, dtype=torch.bfloat16).reshape(16, 4),
        f"{prefix}.attn.wo_b.weight": torch.arange(64, dtype=torch.int8).reshape(4, 16),
        f"{prefix}.attn.wo_b.scale": torch.arange(4, dtype=torch.float32),
        f"{prefix}.hc_ffn_fn": torch.arange(4, dtype=torch.float32).reshape(1, 4),
        f"{prefix}.hc_ffn_scale": torch.arange(3, dtype=torch.float32),
        f"{prefix}.hc_ffn_base": torch.arange(1, dtype=torch.float32),
        f"{prefix}.ffn_norm.weight": torch.arange(4, dtype=torch.bfloat16),
        f"{prefix}.ffn.gate.weight": torch.arange(16, dtype=torch.bfloat16).reshape(4, 4),
        f"{prefix}.ffn.gate.bias": torch.arange(4, dtype=torch.float32),
        f"{prefix}.ffn.shared_experts.w1.weight": torch.arange(8, dtype=torch.int8).reshape(2, 4),
        f"{prefix}.ffn.shared_experts.w1.scale": torch.arange(2, dtype=torch.float32),
        f"{prefix}.ffn.shared_experts.w2.weight": torch.arange(8, dtype=torch.int8).reshape(4, 2),
        f"{prefix}.ffn.shared_experts.w2.scale": torch.arange(4, dtype=torch.float32),
        f"{prefix}.ffn.shared_experts.w3.weight": torch.arange(8, dtype=torch.int8).reshape(2, 4),
        f"{prefix}.ffn.shared_experts.w3.scale": torch.arange(2, dtype=torch.float32),
        f"{prefix}.attn.compressor.wkv.weight": torch.arange(8, dtype=torch.bfloat16).reshape(2, 4),
        f"{prefix}.attn.compressor.wgate.weight": torch.arange(8, dtype=torch.bfloat16).reshape(2, 4),
        f"{prefix}.attn.compressor.ape": torch.arange(8, dtype=torch.float32).reshape(4, 2),
        f"{prefix}.attn.compressor.norm.weight": torch.arange(3, dtype=torch.bfloat16),
        f"{prefix}.attn.indexer.wq_b.weight": torch.arange(12, dtype=torch.int8).reshape(6, 2),
        f"{prefix}.attn.indexer.wq_b.scale": torch.arange(6, dtype=torch.float32),
        f"{prefix}.attn.indexer.weights_proj.weight": torch.arange(8, dtype=torch.bfloat16).reshape(2, 4),
        f"{prefix}.attn.indexer.compressor.wkv.weight": torch.arange(8, dtype=torch.bfloat16).reshape(2, 4),
        f"{prefix}.attn.indexer.compressor.wgate.weight": torch.arange(8, dtype=torch.bfloat16).reshape(2, 4),
        f"{prefix}.attn.indexer.compressor.ape": torch.arange(8, dtype=torch.float32).reshape(4, 2),
        f"{prefix}.attn.indexer.compressor.norm.weight": torch.arange(2, dtype=torch.bfloat16),
    }
    for expert_id in range(n_experts):
        base = expert_id * 10
        raw.update(
            {
                f"{prefix}.ffn.experts.{expert_id}.w1.weight": torch.full((2, 4), base, dtype=torch.int8),
                f"{prefix}.ffn.experts.{expert_id}.w1.scale": torch.full((2,), base + 1, dtype=torch.float32),
                f"{prefix}.ffn.experts.{expert_id}.w2.weight": torch.full((4, 2), base + 2, dtype=torch.int8),
                f"{prefix}.ffn.experts.{expert_id}.w2.scale": torch.full((4,), base + 3, dtype=torch.float32),
                f"{prefix}.ffn.experts.{expert_id}.w3.weight": torch.full((2, 4), base + 4, dtype=torch.int8),
                f"{prefix}.ffn.experts.{expert_id}.w3.scale": torch.full((2,), base + 5, dtype=torch.float32),
            }
        )
    return raw


class _Tokenizer:
    bos_token_id = 0
    eos_token_id = 1
    pad_token_id = None

    def encode(self, text: str) -> list[int]:
        return [1]

    def decode(self, token_ids: list[int]) -> str:
        return ""


def _runtime_model_for_embeddings():
    from python.core.types import ModelConfig, RuntimeModel

    config = ModelConfig(
        model_id="dsv4",
        architecture="DeepseekV4ForCausalLM",
        vocab_size=6,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=43,
        num_attention_heads=64,
        num_key_value_heads=1,
        head_dim=512,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=1,
        torch_dtype="bfloat16",
    )
    runtime = RuntimeConfig(page_size=128, max_batch_size=1, max_seq_len=260, weight_dtype="int8")
    placeholder = torch.empty(0, config.hidden_size)
    return RuntimeModel(
        config=config,
        runtime=runtime,
        embed_tokens=placeholder,
        final_norm_weight=torch.empty(0),
        lm_head=placeholder,
        layers=[],
    )


def _runner_for_prepared_inputs() -> tuple[DeepSeekV4ModelRunner, object]:
    model = _runtime_model_for_embeddings()
    compiled = DeepSeekV4CompiledKernels(
        layout=DeepSeekV4CacheLayout(),
        model_dir="",
        weight_map={},
        weight_store=None,
        compress_ratios=tuple([0] * 44),
        layer_plan=build_deepseek_v4_layer_plan(
            compress_ratios=tuple([0] * 44),
            num_hidden_layers=43,
            num_hash_layers=3,
        ),
        kernel_dir="",
    )
    runner = DeepSeekV4ModelRunner(compiled=compiled)
    runner.init_kv_cache("dsv4", model.config, model.runtime)
    return runner, model


def test_deepseek_lm_head_packs_requested_rows_into_kernel_capacity():
    layout = DeepSeekV4CacheLayout(ranks=2, decode_batch=2, decode_seq=2, decode_tokens=4)
    compiled = DeepSeekV4CompiledKernels(
        layout=layout,
        model_dir="",
        weight_map={},
        weight_store=None,
        compress_ratios=(),
        layer_plan=(),
        kernel_dir="",
        lm_head=DeepSeekV4L3Callable(compiled=object(), name="lm_head"),
    )
    runner = DeepSeekV4ModelRunner(compiled=compiled)
    runner._global_weights = weight_loader.DeepSeekV4GlobalWeights(
        embed_weight=torch.empty(0),
        final_norm_weight=torch.empty(0),
        lm_head_weight=torch.zeros((layout.ranks, 4, 3), dtype=torch.bfloat16).share_memory_(),
        lm_head_layout=weight_loader.DeepSeekV4LmHeadLayout(
            ranks=layout.ranks,
            vocab_size=5,
            hidden_size=3,
            vocab_per_rank=3,
            padded_vocab_per_rank=4,
        ),
        hc_head_fn=torch.empty(0),
        hc_head_scale=torch.empty(0),
        hc_head_base=torch.empty(0),
    )
    runner._static_lm_head_weight = runner._global_weights.lm_head_weight
    hidden = torch.arange(layout.ranks * 6 * 3, dtype=torch.float32).reshape(layout.ranks, 6, 3).to(torch.bfloat16)
    captured = {}

    runner._final_hidden = lambda _: hidden

    def fake_run_l3(callable_spec, hidden_states, lm_head_weight, logits):
        del callable_spec, lm_head_weight
        captured["hidden_states"] = hidden_states.clone()
        logits.zero_()
        logits[0, 0].copy_(torch.arange(logits.shape[-1], dtype=torch.float32))
        logits[0, 1].copy_(torch.arange(logits.shape[-1], dtype=torch.float32) + 10)

    runner._run_l3 = fake_run_l3

    logits = runner._logits_for_hidden(torch.empty(0), active_rows=(5, 2))

    assert torch.equal(captured["hidden_states"][:, :2], hidden[:, [5, 2]])
    assert torch.count_nonzero(captured["hidden_states"][:, 2:]) == 0
    assert logits.shape == (2, 5)
    assert logits[0].tolist() == [0, 1, 2, 3, 4]
    assert logits[1].tolist() == [10, 11, 12, 13, 14]


def test_deepseek_final_hidden_normalizes_before_hc_head_projection_overflows():
    compiled = DeepSeekV4CompiledKernels(
        layout=DeepSeekV4CacheLayout(),
        model_dir="",
        weight_map={},
        weight_store=None,
        compress_ratios=(),
        layer_plan=(),
        kernel_dir="",
    )
    runner = DeepSeekV4ModelRunner(compiled=compiled)
    hidden_size = 3
    runner._global_weights = weight_loader.DeepSeekV4GlobalWeights(
        embed_weight=torch.empty(0),
        final_norm_weight=torch.ones(hidden_size),
        lm_head_weight=torch.empty(0),
        lm_head_layout=weight_loader.DeepSeekV4LmHeadLayout(
            ranks=1,
            vocab_size=1,
            hidden_size=hidden_size,
            vocab_per_rank=1,
            padded_vocab_per_rank=1,
        ),
        hc_head_fn=torch.ones((4, hidden_size * 4), dtype=torch.float32),
        hc_head_scale=torch.ones((1,), dtype=torch.float32),
        hc_head_base=torch.zeros((4,), dtype=torch.float32),
    )
    x_hc = torch.full(
        (1, 2, 4, hidden_size),
        torch.finfo(torch.bfloat16).max,
        dtype=torch.bfloat16,
    )

    flat = x_hc.flatten(2).float()
    inv_rms = torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + 1e-6)
    unstable_mixes = torch.matmul(flat, runner._global_weights.hc_head_fn.t()) * inv_rms
    assert not torch.isfinite(unstable_mixes).all()

    hidden = runner._final_hidden(x_hc)

    assert hidden.shape == (1, 2, hidden_size)
    assert torch.isfinite(hidden.float()).all()

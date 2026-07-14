# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import torch

from examples.model.qwen3_14b.runner.npu_runner import (
    _CompiledKernels,
    _L3Callable,
    Qwen314BModelRunner,
)
from python.core._profiling import StageTimer
from python.core.model_runner import ModelRunner
from python.core.pypto_executor import PyptoExecutor as CorePyptoExecutor
from python.core.types import RuntimeModel
from python.core.utils import rope_tables


_QWEN14B_BLOCK_DIM = 24


def _find_pypto_lib_dir() -> Path:
    """Find the pypto-lib submodule root."""
    start_dir = Path(__file__).resolve().parent
    for directory in (start_dir, *start_dir.parents):
        pypto_lib_dir = directory / "pypto-lib"
        if pypto_lib_dir.is_dir():
            return pypto_lib_dir
    raise FileNotFoundError(
        "Cannot locate the pypto-lib submodule from npu_executor.py. "
        "Run from a pypto-serving checkout with `git submodule update --init --recursive`."
    )


_PYPTO_LIB_DIR = _find_pypto_lib_dir()


def _get_pypto_lib_qwen3_contract(model: RuntimeModel) -> object:
    """Select the Qwen3 contract through pypto-lib's registry."""
    sys.path.insert(0, str(_PYPTO_LIB_DIR))
    try:
        from contract.registry import find_contract_for_model_config  # noqa: PLC0415

        return find_contract_for_model_config(model.config)
    finally:
        try:
            sys.path.remove(str(_PYPTO_LIB_DIR))
        except ValueError:
            pass


def _contract_validation_model(model: RuntimeModel, contract: object) -> SimpleNamespace:
    """Provide contract validators with serving runtime fields plus contract limits."""
    runtime_fields = dict(vars(model.runtime))
    runtime_fields.setdefault("vocab_pad_multiple", int(contract.limits["vocab_pad_multiple"]))
    return SimpleNamespace(
        config=model.config,
        runtime=SimpleNamespace(**runtime_fields),
    )


class Qwen314BPyptoExecutor(CorePyptoExecutor):
    """PyPTO executor that compiles and registers the Qwen3-14B kernels."""

    def __init__(
        self,
        kv_cache_manager=None,
        *,
        platform: str = "a2a3sim",
        device_ids: Sequence[int] = (0,),
        save_kernels_dir: str | None = None,
        l3_trace: bool = False,
    ) -> None:
        super().__init__(
            kv_cache_manager,
            platform=platform,
            device_ids=device_ids,
            save_kernels_dir=save_kernels_dir,
        )
        self._l3_trace = l3_trace

    @property
    def profile_verbose(self) -> bool:
        """Return whether compile and L3 execution timing logs are enabled."""
        return self._l3_trace

    @property
    def supports_device_sampling(self) -> bool:
        """Qwen3 NPU runner can return greedy sampled token ids."""
        return True

    @property
    def supports_device_embedding(self) -> bool:
        """Qwen3 NPU decode embeds greedy token ids inside the device kernel."""
        return True

    def _create_runner(self, model_id: str, compiled: object) -> ModelRunner:
        """Create the Qwen3-14B runtime runner for compiled kernels."""
        if not isinstance(compiled, _CompiledKernels):
            raise TypeError("Qwen314BPyptoExecutor requires Qwen3-14B compiled kernels.")
        return Qwen314BModelRunner(
            compiled=compiled,
            device_id=self._device_ids[0],
        )

    def _compile_model(self, model: RuntimeModel) -> _CompiledKernels:
        """Compile Qwen3-14B PyPTO kernels and pack runtime artifacts."""
        timer = StageTimer(
            enabled=self._l3_trace,
            prefix="compile-breakdown",
            title="_compile_model stage timings",
        )

        def _mark(label: str) -> None:
            timer.mark(label)

        contract = _get_pypto_lib_qwen3_contract(model)
        loaded_kernels = contract.load_kernels()
        contract.validate_kernels(contract, loaded_kernels, _contract_validation_model(model, contract))
        contract.kernel_binder(**loaded_kernels.functions)
        _mark("contract")

        kernel_batch = int(contract.limits["batch"])
        padded_vocab = int(contract.limits["vocab"])
        sampled_ids_width = int(contract.limits["sampled_ids_pad"])
        page_size = model.runtime.page_size
        max_blocks_per_seq = (model.runtime.max_seq_len + page_size - 1) // page_size
        prefill = self._compile_contract_stage(contract.kernels["prefill"], model)
        _mark("compile_prefill")
        decode = self._compile_contract_stage(contract.kernels["decode"], model)
        _mark("compile_decode")
        greedy_sample = self._compile_contract_stage(contract.kernels["greedy_sample"], model)
        _mark("compile_greedy_sample")

        rope_cos_raw, rope_sin_raw = rope_tables(
            model.runtime.max_seq_len,
            model.config.head_dim,
            model.config.rope_theta,
        )
        rope_cos = self._shared_tensor(rope_cos_raw)
        rope_sin = self._shared_tensor(rope_sin_raw)

        _mark("rope_tables")

        prepared_weights = contract.prepare_weights(
            model,
            self._shared_tensor,
            padded_vocab=padded_vocab,
            release_layers=True,
        )
        _mark("prepare_weights")
        prefill_hidden_buffer = torch.empty(
            (kernel_batch * model.runtime.max_seq_len, model.config.hidden_size),
            dtype=torch.bfloat16,
        ).share_memory_()
        prefill_seq_lens_buffer = torch.empty((kernel_batch,), dtype=torch.int32).share_memory_()
        prefill_chunk_lens_buffer = torch.empty((kernel_batch,), dtype=torch.int32).share_memory_()
        prefill_chunk_offsets_buffer = torch.empty((kernel_batch,), dtype=torch.int32).share_memory_()
        prefill_block_table_buffer = torch.empty(
            (kernel_batch * max_blocks_per_seq,),
            dtype=torch.int32,
        ).share_memory_()
        prefill_slot_mapping_buffer = torch.empty(
            (kernel_batch * model.runtime.max_seq_len,),
            dtype=torch.int32,
        ).share_memory_()
        prefill_logits_buffer = torch.empty(
            (kernel_batch, padded_vocab),
            dtype=torch.float32,
        ).share_memory_()
        prefill_sampled_ids_buffer = torch.empty(
            (kernel_batch, sampled_ids_width),
            dtype=torch.int32,
        ).share_memory_()
        prefill_next_hidden_buffer = torch.empty(
            (kernel_batch, model.config.hidden_size),
            dtype=torch.bfloat16,
        ).share_memory_()
        _mark("prefill_buffers")
        decode_logits_buffer = torch.empty(
            (kernel_batch, padded_vocab),
            dtype=torch.float32,
        ).share_memory_()
        decode_seq_lens_buffer = torch.empty((kernel_batch,), dtype=torch.int32).share_memory_()
        decode_block_table_buffer = torch.empty(
            (kernel_batch * max_blocks_per_seq,),
            dtype=torch.int32,
        ).share_memory_()
        decode_slot_mapping_buffer = torch.empty((kernel_batch,), dtype=torch.int32).share_memory_()
        decode_token_ids_buffer = torch.empty(
            (kernel_batch, sampled_ids_width),
            dtype=torch.int32,
        ).share_memory_()
        decode_sampled_ids_buffer = torch.empty(
            (kernel_batch, sampled_ids_width),
            dtype=torch.int32,
        ).share_memory_()
        decode_next_hidden_buffer = torch.empty(
            (kernel_batch, model.config.hidden_size),
            dtype=torch.bfloat16,
        ).share_memory_()
        _mark("decode_logits_buffer")

        timer.report()

        return _CompiledKernels(
            contract=contract,
            prefill=prefill,
            decode=decode,
            greedy_sample=greedy_sample,
            final_norm_weight=prepared_weights.final_norm_weight,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            padded_vocab=padded_vocab,
            padded_lm_head_weight=prepared_weights.padded_lm_head_weight,
            padded_embed_weight=prepared_weights.padded_embed_weight,
            decode_weights=prepared_weights.decode_weights,
            prefill_hidden_buffer=prefill_hidden_buffer,
            prefill_seq_lens_buffer=prefill_seq_lens_buffer,
            prefill_chunk_lens_buffer=prefill_chunk_lens_buffer,
            prefill_chunk_offsets_buffer=prefill_chunk_offsets_buffer,
            prefill_block_table_buffer=prefill_block_table_buffer,
            prefill_slot_mapping_buffer=prefill_slot_mapping_buffer,
            prefill_logits_buffer=prefill_logits_buffer,
            prefill_sampled_ids_buffer=prefill_sampled_ids_buffer,
            prefill_next_hidden_buffer=prefill_next_hidden_buffer,
            decode_seq_lens_buffer=decode_seq_lens_buffer,
            decode_block_table_buffer=decode_block_table_buffer,
            decode_slot_mapping_buffer=decode_slot_mapping_buffer,
            decode_logits_buffer=decode_logits_buffer,
            decode_token_ids_buffer=decode_token_ids_buffer,
            decode_sampled_ids_buffer=decode_sampled_ids_buffer,
            decode_next_hidden_buffer=decode_next_hidden_buffer,
        )

    def _compile_contract_stage(self, stage: object, model: RuntimeModel) -> _L3Callable:
        """Compile one contract-owned HOST wrapper."""
        dummy_args = stage.compile_args_builder(model.config, model.runtime)
        return self._compile_jit_fwd_callable(stage.name, stage.host_jit_fn, dummy_args)

    def _compile_jit_fwd_callable(
        self,
        name: str,
        jit_fn: object,
        dummy_args: list[torch.Tensor],
    ) -> _L3Callable:
        """Compile a HOST wrapper into a PyPTO DistributedCompiledProgram."""
        from pypto.ir.distributed_compiled_program import DistributedCompiledProgram  # noqa: PLC0415
        from pypto.ir.distributed_compiled_program import DistributedConfig  # noqa: PLC0415
        from pypto.runtime import RunConfig  # noqa: PLC0415

        config = self._run_config(codegen_only=True)
        distributed_config = DistributedConfig(
            device_ids=list(self._device_ids),
            num_sub_workers=0,
            block_dim=_QWEN14B_BLOCK_DIM,
            aicpu_thread_num=4,
        )
        run_config = RunConfig(
            platform=config.platform,
            device_id=config.device_id,
            backend_type=config.backend_type,
            strategy=config.strategy,
            dump_passes=config.dump_passes,
            save_kernels=config.save_kernels,
            save_kernels_dir=config.save_kernels_dir,
            codegen_only=True,
            pto_isa_commit=config.pto_isa_commit,
            diagnostic_phase=config.diagnostic_phase,
            disabled_diagnostics=config.disabled_diagnostics,
            compile_profiling=config.compile_profiling,
            distributed_config=distributed_config,
        )
        compiled = jit_fn.compile(*dummy_args, config=run_config)
        if not isinstance(compiled, DistributedCompiledProgram):
            raise TypeError(
                f"{name} did not compile to DistributedCompiledProgram; got {type(compiled).__name__}"
            )
        return _L3Callable(
            compiled=compiled,
            name=name,
            block_dim=_QWEN14B_BLOCK_DIM,
            aicpu_thread_num=4,
        )

    @staticmethod
    def _shared_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Move a CPU tensor into shared memory if needed."""
        if tensor.device.type == "cpu" and not tensor.is_shared():
            return tensor.share_memory_()
        return tensor

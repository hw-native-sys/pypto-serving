# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Import the pinned Lib and compare its real ABI without compiling or using devices."""

import inspect
import sys
from dataclasses import replace
from types import SimpleNamespace

import pypto.language as pl
import pytest
import torch

from pypto_serving.model.deepseek_dspark import task_args
from pypto_serving.model.deepseek_dspark.npu_executor import DeepSeekV4DSparkPyptoExecutor
from pypto_serving.model.deepseek_dspark.npu_runner import (
    DSparkCacheLayout, DSparkCompiledKernels, DSparkModelRunner,
)


def _check_shapes(program, specs):
    params = inspect.signature(program._func).parameters
    dtypes = {torch.int32: pl.INT32, torch.int64: pl.INT64,
              torch.float32: pl.FP32, torch.bfloat16: pl.BF16, torch.int16: pl.INT16}
    for name, (dtype, shape) in specs.items():
        if name not in params:
            continue  # Fused programs keep some preparation tensors internal.
        annotation = params[name].annotation
        assert annotation.dtype == dtypes[dtype], name
        assert len(annotation.shape) == len(shape), name
        for expected, actual in zip(annotation.shape, shape, strict=True):
            if isinstance(expected, int):
                assert actual == expected, (name, shape, annotation.shape)


@pytest.mark.parametrize("ranks,tp", [(16, 1), (16, 4), (8, 1), (8, 4), (16, 1)])
def test_real_lib_abi_keeps_one_owner_tile(ranks, tp, monkeypatch):
    layout = DSparkCacheLayout.for_ranks(ranks, tp_size=tp)
    executor = DeepSeekV4DSparkPyptoExecutor(device_ids=range(ranks), cache_ranks=ranks, tp_size=tp)
    argv, config = list(sys.argv), sys.modules.get("config")
    modules = executor._load_kernel_modules(layout, speculative=True)
    assert sys.argv == argv and sys.modules.get("config") is config
    assert (layout.decode_batch, layout.decode_tokens) == (16 * tp, 128 * tp)
    assert modules["decode_fwd_dspark"].LOCAL_DECODE_TOKENS == layout.decode_local_tokens == 128
    assert modules["dspark_drafter"].DSPARK_MAX_BATCH == layout.decode_local_batch == 16
    scratch = {name: (dtype, shape) for name, (shape, dtype) in task_args.decode_scratch_specs(ranks).items()}
    decode = task_args._decode_slots(layout) | scratch
    for key, entry in (("decode_fwd", "l3_decode_fwd"), ("decode_fwd_dspark", "l3_decode_fwd_dspark")):
        _check_shapes(getattr(modules[key], entry), decode)
    draft = task_args._drafter_slots(layout) | {
        name: (dtype, shape) for name, (shape, dtype) in task_args.drafter_scratch_specs(ranks).items()}
    _check_shapes(modules["dspark_drafter"].l3_dspark_drafter, draft)
    _check_shapes(modules["dspark_markov"].l3_distributed_markov_sample,
                  task_args._markov_slots(layout) | {"head_hidden": draft["head_hidden"]})
    bootstrap = draft | {name: (dtype, shape) for name, (shape, dtype) in
                         task_args.bootstrap_metadata_specs(layout).items()}
    _check_shapes(modules["prefill_bootstrap_one_l2"].l3_bootstrap_one_l2, bootstrap)
    _check_shapes(modules["prefill_bootstrap_metadata"].l3_prepare_bootstrap_metadata, bootstrap)
    _check_shapes(modules["prefill_fwd"].l3_prefill_fwd, task_args._prefill_slots(layout))
    # The same buffers feed fused Decode; no separate enlarged drafter allocation.
    fused_draft = {"draft_" + name: spec for name, spec in draft.items()}
    _check_shapes(modules["decode_fwd_dspark"].l3_decode_fwd_dspark, fused_draft)
    runner = DSparkModelRunner(compiled=DSparkCompiledKernels(layout=layout, model_dir="unused",
        weight_map={}, weight_store=None, compress_ratios=(), layer_plan=(), kernel_dir="unused"))
    monkeypatch.setattr(runner, "_alloc_empty_stacked_tensor",
                        lambda shape, dtype: SimpleNamespace(full_shape=tuple(shape), dtype=dtype))
    tokens = runner._materialize_dspark_device_state_tokens()
    meta = runner._materialize_dspark_device_state_meta()
    _check_shapes(modules["decode_fwd_dspark"].l3_decode_fwd_dspark,
                  {"state_tokens": (tokens.dtype, tokens.full_shape),
                   "state_meta": (meta.dtype, meta.full_shape)})
    assert runner._materialize_dspark_device_state_tokens() is tokens
    assert runner._materialize_dspark_device_state_meta() is meta
    for group in (0, layout.partitions - 1):
        leases = [runner._reserve_drafter_state(f"{group}-{i}", group=group, prompt_len=128).lease
                  for i in range(layout.decode_batch)]
        assert sorted(leases) == list(range(tokens.full_shape[1]))
        with pytest.raises(RuntimeError, match="leases exhausted"):
            runner._reserve_drafter_state(f"{group}-extra", group=group, prompt_len=128)
    for state in ("HCA", "CSA", "CSA_INNER"):
        assert getattr(modules["config"], state + "_STATE_PHYSICAL_BLOCKS") == (
            layout.decode_batch * getattr(modules["config"], state + "_STATE_BLOCKS_PER_REQUEST"))
    plain = executor._load_kernel_modules(layout, speculative=False)
    assert plain["decode_fwd"].MAX_LOGIT_ROWS == modules["decode_fwd"].MAX_LOGIT_ROWS == 128
    assert plain["decode_fwd"].LM_HEAD_TP_SIZE == 4
    with pytest.raises(ValueError, match="LM-head rows"):
        executor._load_kernel_modules(replace(layout, max_logit_rows=512), speculative=True)
    assert sys.argv == argv and sys.modules.get("config") is config

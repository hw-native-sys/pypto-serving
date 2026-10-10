# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Serving dispatch boundary for lib's complete V4.1 decode backbone."""

import ctypes
import importlib

from .composite import LayerState
from .swa_segment import load_segment_modules


def compile_decode_backbone(compiler, lib_root, topology):
    """Compile the existing 40-layer lib entry without rebuilding model operators."""
    import pypto.language as pl

    load_segment_modules(lib_root, topology)
    module = importlib.import_module("models.deepseek_v4_1_flash.decode_fwd")
    config = importlib.import_module("models.deepseek_v4_1_flash.config")
    if module.N_LAYERS != config.FLASH.num_hidden_layers or module.N_LAYERS != 40:
        raise ValueError("decode backbone and checkpoint layer schedules disagree")
    program = compiler.compile(
        "v41_decode_fwd", module.l3_decode_fwd, attention_num_tokens=pl.RUNTIME,
    )
    return program, tuple(module.l3_decode_fwd.param_names)


class DecodeBackbone:
    """Run one device-resident decode step with explicit lib-owned state/cache ABI."""

    def __init__(self, worker, program, param_names, run_config):
        self.worker = worker
        self.program = program
        self.param_names = tuple(param_names)
        self.run_config = run_config
        self.failed = False
        if len(set(self.param_names)) != len(self.param_names):
            raise ValueError("duplicate decode ABI parameter")
        if not {"x_hc", "pre_mix", "attention_num_tokens"} <= set(self.param_names):
            raise ValueError("decode ABI must expose residual, delayed mix and active count")

    def run(self, state: LayerState, arguments, *, active_tokens: int) -> LayerState:
        if self.failed:
            raise RuntimeError("decode dispatch failed; close the worker before retrying")
        if state.layout != "tp_local_token":
            raise ValueError("decode backbone requires TP-local residual and pre_mix")
        if type(active_tokens) is not int or not 0 < active_tokens <= 192:
            raise ValueError("decode requires a compact active batch of 1..192 tokens")
        reserved = {"x_hc", "pre_mix", "attention_num_tokens"}
        if reserved & arguments.keys():
            raise ValueError("decode state/count must come from this dispatch, not argument buffers")
        missing = set(self.param_names) - reserved - arguments.keys()
        if missing:
            raise ValueError("missing decode ABI arguments: " + ", ".join(sorted(missing)))
        bound = dict(arguments, x_hc=state.residual, pre_mix=state.pre_mix,
                     attention_num_tokens=ctypes.c_int32(active_tokens))
        try:
            self.worker.run(self.program.compiled, *(bound[name] for name in self.param_names),
                            config=self.run_config)
        except BaseException:
            self.failed = True
            raise
        return state

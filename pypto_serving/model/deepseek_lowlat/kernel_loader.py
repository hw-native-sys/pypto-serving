# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Freeze TP8/T1 profiles and compile without persistent JIT caching."""

import contextlib
from dataclasses import dataclass, field, replace
import importlib
import json
import os
from pathlib import Path
import sys
import threading

from pypto import CacheConfig
from pypto_serving.model.common.compiler.compiler import KernelCompiler

from .bindings import parameter_rows


_LOCK = threading.RLock()
_PREFIX = "models.deepseek_v4_flash_lowlat"


@contextlib.contextmanager
def isolated_profile(modules=None, *, context_capacity=16384):
    """Retain a private import closure, including lazy relative helper imports."""
    with _LOCK:
        old_argv, old_path = sys.argv, list(sys.path)
        before = {
            name: module
            for name, module in sys.modules.items()
            if name == _PREFIX or name.startswith(_PREFIX + ".")
        }
        for name in before:
            sys.modules.pop(name, None)
        sys.argv = [
            "lowlat-serving",
            "--tp",
            "8",
            "--spec-tokens",
            "0",
            "--moe-banks",
            "43",
            "--max-seq-len",
            str(context_capacity),
        ]
        lib = Path(os.environ.get("PYPTO_LIB_ROOT", Path(__file__).resolve().parents[3] / "pypto-lib"))
        sys.path.insert(0, str(lib))
        parent = importlib.import_module("models")
        missing = object()
        previous_attribute = getattr(parent, "deepseek_v4_flash_lowlat", missing)
        if modules:
            sys.modules.update(modules)
            setattr(parent, "deepseek_v4_flash_lowlat", modules[_PREFIX])
        captured = {}
        try:
            yield captured
        finally:
            captured.update(
                {
                    name: module
                    for name, module in sys.modules.items()
                    if name == _PREFIX or name.startswith(_PREFIX + ".")
                }
            )
            for name in captured:
                sys.modules.pop(name, None)
            sys.modules.update(before)
            if previous_attribute is missing:
                if hasattr(parent, "deepseek_v4_flash_lowlat"):
                    delattr(parent, "deepseek_v4_flash_lowlat")
            else:
                setattr(parent, "deepseek_v4_flash_lowlat", previous_attribute)
            sys.argv, sys.path[:] = old_argv, old_path


@dataclass
class LowlatPrograms:
    single: object
    prefill: object
    single_metadata: dict
    prefill_metadata: dict
    modules: dict
    output_dir: Path
    model: object
    run_config: object
    native: dict = field(default_factory=dict)
    native_metadata: dict = field(default_factory=dict)
    native_modules: dict = field(default_factory=dict)
    draft_weights: object = None
    context_capacity: int = 16384
    prefix_namespace: str = ""

    @contextlib.contextmanager
    def context(self):
        captured = {}
        try:
            with isolated_profile(self.modules, context_capacity=self.context_capacity) as captured:
                yield
        finally:
            self.modules.update(captured)


def compile_programs(model, run_config):
    compiler = KernelCompiler(run_config=replace(run_config, cache_config=CacheConfig(enabled=False)))
    from .capacity import LowlatCapacityPlan

    runtime = model.runtime
    plan = LowlatCapacityPlan(runtime.max_seq_len, runtime.max_prefill_tokens_per_request or 128,
                             "rolling" if any(g.sliding_window for g in runtime.kv_cache_groups) else "full")
    with isolated_profile(context_capacity=plan.compiled_context) as captured:
        single_module = importlib.import_module(_PREFIX + ".native_single")
        prefill_module = importlib.import_module(_PREFIX + ".prefill_fwd")
        single = compiler.compile("lowlat_single", single_module.l3_decode_fwd)
        prefill = compiler.compile("lowlat_prefill", prefill_module.load_prefill_programs()["chunk"])
        metadata = []
        for program, expected in ((single, 81), (prefill, 100)):
            rows = json.loads((Path(program.compiled.output_dir) / "distributed_meta.json").read_text())
            if len(parameter_rows(rows)) != expected:
                raise ValueError(f"lowlat ABI drift: {program.name}")
            metadata.append(rows)
    directory = Path(single.compiled.output_dir).parent
    (directory / "lowlat-abi.json").write_text(
        json.dumps(
            {
                "single": metadata[0],
                "prefill": metadata[1],
                "compile_cache": False,
            },
            indent=2,
        )
        + "\n"
    )
    return LowlatPrograms(single, prefill, *metadata, captured, directory, model, run_config,
                          context_capacity=plan.compiled_context)


def validate_shared_rope(target_config, draft_config, context_capacity):
    fields = ("qk_rope_head_dim", "rope_theta", "original_max_position_embeddings",
              "beta_fast", "beta_slow", "rope_factor")
    if any(getattr(target_config,name)!=getattr(draft_config,name) for name in fields):
        raise ValueError("target and drafter cannot share the uncompressed RoPE table")
    # Drafter page32 tables deliberately retain their full model depth. Its
    # capacity may exceed the admitted target profile without changing RoPE.
    if (target_config.max_position_embeddings!=context_capacity
            or draft_config.max_position_embeddings<context_capacity):
        raise ValueError("native target/drafter capacity does not cover the compiled profile")


def compile_native_programs(programs, *, cycles=(1,)):
    if not cycles or any(type(c) is not int or c not in (1, 2) for c in cycles):
        raise ValueError("native programs require one or two actual cycles")
    compiler = KernelCompiler(run_config=replace(programs.run_config,
                                                 cache_config=CacheConfig(enabled=False)))
    with isolated_profile(context_capacity=programs.context_capacity) as captured:
        sys.argv = ["lowlat-native-serving", "--tp", "8", "--dp", "1", "--spec-tokens", "5",
                    "--moe-banks", "43", "--max-seq-len", str(programs.context_capacity)]
        module = importlib.import_module(_PREFIX+".native_pipeline")
        validate_shared_rope(programs.modules[_PREFIX+".config"].FLASH,
                             importlib.import_module(_PREFIX+".drafter.config").FLASH,
                             programs.context_capacity)
        for cycle in cycles:
            compiled = compiler.compile("lowlat_native_"+str(cycle), module.native_cycles,
                                        cycle_count=cycle, native_w2_fingerprint="generated")
            metadata = json.loads((Path(compiled.compiled.output_dir)/"distributed_meta.json").read_text())
            if len(parameter_rows(metadata)) != 153:
                raise ValueError("native compact ABI must have153parameters")
            programs.native[cycle], programs.native_metadata[cycle] = compiled, metadata
    if any(m["params"] != programs.native_metadata[1]["params"]
           for m in programs.native_metadata.values()):
        raise ValueError("native one/two parameter metadata differ")
    programs.native_modules = captured
    (programs.output_dir/"lowlat-native-abi.json").write_text(json.dumps(
        dict(w2_implementation="standard_generated", compile_cache=False,
             metadata=programs.native_metadata), indent=2)+"\n")
    return programs

# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Replay one C1A Reuse Attention composite from a saved same-input chain.

This is a diagnostic readback/replay, not a production serving path or an
accumulated accuracy result. No routed expert weights are loaded. Establish
bitwise equivalence to the original capture before interpreting tagged cuts.
"""
import argparse
import ctypes
import json
from pathlib import Path
from types import SimpleNamespace

from validate_v41_c2a_chain import prepare, select_plans, state_names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lib-root", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--layer", type=int, default=22)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--dump-tagged", action="store_true")
    parser.add_argument("--candidate", action="store_true",
                        help="Gate a changed kernel against the unchanged original reference")
    args = parser.parse_args()

    import torch
    import pypto.language as pl
    from pypto.ir import DistributedConfig
    from pypto.runtime import RunConfig
    from pypto_serving.model.common.compiler.compiler import KernelCompiler
    from pypto_serving.model.deepseek_v41.prefill_segment import C1A_ARGS, bind_prefill_producers
    from pypto_serving.model.deepseek_v41.swa_segment import (
        SegmentTopology, load_segment_modules, make_segment_worker,
    )
    from pypto_serving.model.deepseek_v41.swa_weights import load_prefill_attention_weights

    torch.set_num_threads(4)
    devices = tuple(int(d) for d in args.devices.split(","))
    if args.tp <= 0 or len(set(devices)) != len(devices) or len(devices) % args.tp:
        raise ValueError("devices must form complete TP groups")
    topology = SegmentTopology(tp=args.tp, dp=len(devices) // args.tp)
    load_segment_modules(args.lib_root, topology)
    from models.deepseek_v4_1_flash import prefill_c1a_sp as module, prefill_c2a_full as common

    artifact = Path(args.artifact_dir)
    artifact.mkdir(parents=True, exist_ok=True)
    if (artifact / "comparison.pt").exists():
        raise FileExistsError("refusing to overwrite a previous replay")
    saved = torch.load(args.comparison, map_location="cpu", weights_only=True)
    if saved["family"] != "c1a" or len(saved["chunks"]) != 1:
        raise ValueError("replay currently requires a single-chunk C1A capture")
    records = {(r["layer"], r["stage"]): r for r in saved["stages"]}
    raw = json.loads((Path(args.model_dir) / "config.json").read_text())
    plans = select_plans(raw, "c1a", args.layer)
    plan = plans[-1]
    if plan.mode != "c1a_reuse":
        raise ValueError("isolated replay currently requires a Reuse consumer")
    # Reuse the original metadata builder and identical Attention weight loader.
    # Empty MoE weight maps are unused by this Attention-only diagnostic.
    weights = {p.layer_id: (load_prefill_attention_weights(args.model_dir, p.layer_id, topology), {})
               for p in plans}
    options = SimpleNamespace(model_dir=args.model_dir, input_state=saved["input_state"],
        family="c1a", last_layer=args.layer, group_counts=saved["group_counts"])
    _, attention, _, _, _ = prepare(options, topology, module, weights)
    values = bind_prefill_producers(plans, attention)[args.layer]
    previous = records[args.layer - 1, "moe"]["actual"]
    values.update(x_hc=previous["x_next"], pre_mix=previous["next_pre_mix"])
    for name in ("compressed_cache", "compressed_cache_scale", "index_cache", "index_cache_scale"):
        values[name] = records[plan.kv_source, "attention"]["actual"][name]
    values["candidate_mask"] = records[plan.candidate_source, "attention"]["actual"]["candidate_mask"]
    values["compressed_indices"] = records[plan.index_source, "attention"]["actual"]["topk_indices"]
    names = ("attn_input", "attn_output", "next_pre_mix", "x_hc_out") + state_names(module, "reuse")
    initial = {n: values[n].clone() for n in state_names(module, "reuse")}
    captured = {n: torch.empty_like(values[n]).share_memory_() for n in names}
    counts = values["num_tokens"].clone().share_memory_()
    config = RunConfig(platform="a5", distributed_config=DistributedConfig(device_ids=list(devices)),
        ring_heap=4096 << 20, ring_task_window=131072, ring_dep_pool=131072,
        enable_dump_args=int(args.dump_tagged))
    compiler = KernelCompiler(run_config=config, cache_dir=args.build_dir)
    program = compiler.compile("v41_replay_c1a_reuse",
        module.make_program("reuse", topology.world, epochs=1), attention_epoch=pl.RUNTIME)
    with make_segment_worker([program], config, list(values.values())) as worker:
        handles = {n: worker.alloc_stacked_tensor(values[n]) for n in C1A_ARGS
                   if n not in ("num_tokens", "attention_epoch")}
        arguments = {**handles, "num_tokens": counts, "attention_epoch": ctypes.c_int32(1)}
        worker.run(program.compiled, *(arguments[n] for n in C1A_ARGS), config=config)
        for name, destination in captured.items():
            worker.copy_stacked_from(handles[name], destination)
        for handle in reversed(list(handles.values())):
            worker.free_stacked_tensor(handle)
    original = records[args.layer, "attention"]
    equivalent = {}
    for name, value in captured.items():
        equivalent[name] = torch.equal(value.view(torch.uint8), original["actual"][name].view(torch.uint8))
        print("ORIGINAL REPLAY", name, equivalent[name], flush=True)
    expected = {**values, **{n: values[n].clone() for n in captured}}
    common.make_golden("reuse", 1, attention_reference=module.reference_attention,
                       state_names=module.STATE_NAMES["reuse"])(expected)
    results = {}
    for name, check in module.make_compare("reuse", 1, initial).items():
        results[name] = check(captured[name], expected[name], inputs=expected,
            actual_outputs=captured, expected_outputs=expected, rtol=1e-3, atol=1e-3)
        print("NATIVE REPLAY", name, results[name], flush=True)
    expected_equal = {n: torch.equal(expected[n].view(torch.uint8), v.view(torch.uint8))
                      for n, v in original["expected"].items()}
    print("ORIGINAL EXPECTED", expected_equal, flush=True)
    torch.save(dict(actual=captured, expected={n: expected[n] for n in captured},
        equivalent=equivalent, expected_equal=expected_equal, results=results,
        comparison=args.comparison, layer=args.layer), artifact / "comparison.pt")
    if not all(expected_equal.values()):
        raise AssertionError("canonical expected values changed; candidate comparison is invalid")
    if args.candidate:
        if not all(ok for ok, _ in results.values()):
            raise AssertionError("candidate fails original native precision checks")
        print("CANDIDATE NATIVE REPLAY PASS; not accumulated or M0 acceptance", flush=True)
    elif not all(equivalent.values()):
        raise AssertionError("isolated replay differs from original; do not interpret cuts yet")
    else:
        print("ISOLATED REPLAY EQUIVALENCE PASS; native accuracy results remain separate", flush=True)


if __name__ == "__main__":
    main()

# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A5 two-layer SWA/MoE diagnostic with optional checkpoint weights/reference.

Activations and request metadata are controlled fixtures, including when real
checkpoint weights are selected. This is not text-generation acceptance.
"""
import argparse
from pathlib import Path
from types import SimpleNamespace
import sys

ATTENTION_OUTPUTS = ("output", "next_pre_mix", "hidden", "attn_out", "window_cache", "window_cache_scale")
MOE_OUTPUTS = ("x_next", "next_pre_mix", "x_mixed")
REQUEST_INPUTS = ("rope_cos", "rope_sin", "window_slots", "window_indices", "window_cache", "window_cache_scale")


def prepare_checkpoint_metadata(tensors, model_dir, topology, token_ids):
    """First-chunk request metadata with checkpoint RoPE and private empty pages."""
    import json
    import torch
    from models.deepseek_v4_1_flash.rope_tables import precompute_rope_tables
    from pypto_serving.model.deepseek_v41.request_state import ForwardStep, RequestSlice
    from pypto_serving.model.deepseek_v41.swa_metadata import gather_swa_rope_rows, prepare_swa_window_metadata

    text = json.loads((Path(model_dir) / "config.json").read_text())["text_config"]
    pages = (topology.capacity + 127) // 128
    step = ForwardStep("prefill", tuple(
        RequestSlice(str(group), group, 0, 0, tuple(row.tolist()), topology.capacity,
                     {"window": tuple(range(pages))})
        for group, row in enumerate(token_ids)
    ), 1)
    # Keep fixture allocation shapes to isolate addressing/positions from compilation.
    metadata = prepare_swa_window_metadata(step, topology, cache_pages=tensors["window_cache"].shape[1])
    rope_config = SimpleNamespace(qk_rope_head_dim=int(text["qk_rope_head_dim"]),
                                  rope_theta=float(text["rope_theta"]))
    tables = precompute_rope_tables(topology.capacity, False, config=rope_config)
    tensors["rope_cos"], tensors["rope_sin"] = gather_swa_rope_rows(metadata, tables)
    tensors["window_slots"], tensors["window_indices"] = metadata.window_slots, metadata.window_indices
    tensors["window_cache"].view(torch.uint8).zero_()
    # UE8M0 code 127 represents scale 1 for unpublished zero cache payloads.
    tensors["window_cache_scale"].view(torch.uint8).fill_(127)
    return {name: tensors[name].clone() for name in REQUEST_INPUTS}


def prepare_checkpoint_inputs(tensors, model_dir, topology, token_ids=None):
    """Prepare an embedding-broadcast control; retain the random stress fixture separately."""
    import torch
    from pypto_serving.model.deepseek_v41.input_preparation import lookup_token_embeddings
    from pypto_serving.model.deepseek_v41.request_state import ForwardStep, RequestSlice
    from pypto_serving.model.deepseek_v41.segment_inputs import prepare_segment_inputs
    from pypto_serving.model.deepseek_v41.swa_segment import SegmentTopology
    from pypto_serving.model.deepseek_v41.weight_loader import V41WeightLoader

    if token_ids is None:
        ids = torch.arange(1, topology.dp * topology.capacity + 1, dtype=torch.int64).reshape(
            topology.dp, topology.capacity)
    else:
        ids = torch.as_tensor(token_ids)
        if ids.dtype not in (torch.int32, torch.int64) or ids.shape != (topology.dp, topology.capacity):
            raise ValueError("token_ids must contain exactly one full integer token row per DP group")
    loaders = [V41WeightLoader(model_dir, tp_size=topology.tp, tp_rank=rank,
                              ep_size=topology.world, ep_rank=rank, max_load_bytes=512 << 20)
               for rank in range(topology.tp)]
    embeddings = lookup_token_embeddings(loaders, ids)
    step = ForwardStep("prefill", tuple(
        RequestSlice(str(group), group, 0, 0, tuple(row.tolist()), topology.capacity, {})
        for group, row in enumerate(ids)
    ), 1)
    prepared = prepare_segment_inputs(
        embeddings.flatten(0, 1), step,
        SegmentTopology(tp=topology.tp, dp=topology.dp, local_capacity=topology.local_capacity),
    )
    tensors["x_hc"] = prepared.residual
    tensors["incoming_pre_mix"] = prepared.pre_mix
    return ids


def compare_stages(layers, captured, initial, swa, moe, topology):
    """Localize errors against each half-layer's actual input, after execution."""
    from golden.validation import ratio_allclose

    residual, mix = initial
    records = []
    for layer_id, ((la, lm), (actual_a, actual_m)) in enumerate(zip(layers, captured)):
        ra = dict(la, x_hc=residual, incoming_pre_mix=mix)
        for name in ATTENTION_OUTPUTS:
            ra[name] = la[name].clone()
        swa.golden_prefill_swa_case(ra)
        rm = dict(lm, x_hc=actual_a["output"], pre_mix=actual_a["next_pre_mix"])
        for name in MOE_OUTPUTS:
            rm[name] = lm[name].clone()
        moe.golden_moe(rm)
        checks = (
            ("attention", actual_a, ra, swa.make_staged_compare()),
            ("moe", actual_m, rm, {
                "next_pre_mix": ratio_allclose(atol=2.5e-5, rtol=5e-3),
                "x_mixed": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
                "x_next": moe._local_mhc_compare([topology.local_capacity] * topology.world),
            }),
        )
        for label, actual, reference, comparators in checks:
            results = {}
            for name, compare in comparators.items():
                ok, message = compare(actual[name], reference[name], actual_outputs=actual,
                                      expected_outputs=reference, inputs=reference, rtol=1e-3, atol=1e-3)
                print(f"STAGE layer={layer_id} {label}.{name}: {ok} {message}", flush=True)
                results[name] = (bool(ok), message)
            records.append({"layer": layer_id, "stage": label, "results": results,
                            "actual": actual, "expected": {name: reference[name] for name in actual}})
        residual, mix = actual_m["x_next"], actual_m["next_pre_mix"]
    return records


def compare_saved(data, moe, topology, residual_profile="dsv4-layer"):
    """Apply the selected residual budget, retaining pre-mix and local-stage gates."""
    import torch

    actual = data["actual_residual"]
    expected = data["expected_residual"]
    if actual.shape != expected.shape or actual.shape[0] != topology.world:
        raise ValueError("residual shapes must match and contain every logical rank")
    for name in ("residual", "pre_mix"):
        a, e = data["actual_" + name].double(), data["expected_" + name].double()
        error = a - e
        print(f"{name}: rel_l2={(error.norm() / e.norm().clamp_min(1e-12)).item():.8g} "
              f"max_abs={error.abs().max().item():.8g}", flush=True)
    counts = [topology.local_capacity] * topology.world
    kwargs = dict(actual_outputs={"x_next": actual}, expected_outputs={"x_next": expected},
                  inputs={"num_tokens": torch.tensor(counts, dtype=torch.int32)}, rtol=1e-5, atol=1e-5)
    if residual_profile == "v41-local":
        ok, message = moe._local_mhc_compare(counts)(actual, expected, **kwargs)
    elif residual_profile == "dsv4-layer":
        from golden.validation import ratio_reldiff

        compare = ratio_reldiff(diff_thd=0.01, pct_thd=0.05)
        results = []
        for rank, count in enumerate(counts):
            passed, detail = compare(actual[rank, :count], expected[rank, :count], **kwargs)
            print(f"Final residual rank={rank} profile={residual_profile}: {passed} {detail}", flush=True)
            results.append((passed, detail))
        ok = all(passed for passed, _ in results)
        message = "\n".join(detail for passed, detail in results if not passed)
    else:
        raise ValueError(f"Unknown residual profile: {residual_profile}")
    print(f"Final residual reference check ({residual_profile}):", ok, message, flush=True)
    mix_error = None
    try:
        # User-approved provisional budget for this two-layer accumulated output only.
        torch.testing.assert_close(data["actual_pre_mix"], data["expected_pre_mix"], rtol=1e-2, atol=1e-3)
    except AssertionError as exc:
        mix_error = str(exc)
    print("Final pre_mix reference check:", mix_error is None, mix_error or "", flush=True)
    stages_ok = all(result[0] for stage in data.get("stages", []) for result in stage["results"].values())
    assert ok and mix_error is None and stages_ok, message + (mix_error or "") + (
        "Half-layer stage comparison failed" if not stages_ok else ""
    )
    print("TWO-LAYER TORCH REFERENCE PASS", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lib-root", required=True)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--build-dir", default="build_output/v41-swa-segment")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--input-source", choices=("stress", "embeddings"), default="stress",
                        help="Keep the random stress input, or load real embedding rows and broadcast HC streams")
    parser.add_argument("--token-ids", help="JSON array [DP, capacity] of embedding control IDs; no padding")
    parser.add_argument("--request-metadata", action="store_true",
                        help="Use fresh private request pages and checkpoint RoPE with embedding inputs")
    parser.add_argument("--model-dir", help="Use actual checkpoint weights for layers 0 and 1")
    parser.add_argument("--reference", action="store_true", help="Compare against composed Torch references")
    parser.add_argument("--stage-reference", action="store_true",
                        help="Also localize errors on each half-layer's device input, after the full run")
    parser.add_argument("--compare-only", help="Recheck a saved comparison.pt on CPU without compilation/device use")
    parser.add_argument("--residual-profile", choices=("dsv4-layer", "v41-local"), default="dsv4-layer",
                        help="DSV4 layer residual budget (0.01/5%%), or the historical V4.1 local MoE budget")
    parser.add_argument("--dump-tagged", action="store_true",
                        help="Enable selective runtime dumps from a separately instrumented lib checkout")
    parser.add_argument("--artifact-dir", default=".validation-artifacts/swa-segment")
    parser.add_argument("--ring-heap-mib", type=int, default=1024,
                        help="Per-ring temporary heap; lib MoE validation uses 1024 MiB")
    args = parser.parse_args()
    if args.request_metadata and (args.input_source != "embeddings" or not args.model_dir):
        parser.error("--request-metadata requires --input-source embeddings and --model-dir")
    if args.input_source == "embeddings" and not args.model_dir:
        parser.error("--input-source embeddings requires --model-dir")
    if args.token_ids and args.input_source != "embeddings":
        parser.error("--token-ids requires --input-source embeddings")
    if args.stage_reference:
        args.reference = True
    sys.path.insert(0, str(Path(args.lib_root).resolve()))
    import torch
    from pypto.ir import DistributedConfig
    from pypto.runtime import RunConfig
    from golden.spec import TensorSpec
    from pypto_serving.model.common.compiler.compiler import KernelCompiler
    from pypto_serving.model.deepseek_v41.composite import LayerState
    from pypto_serving.model.deepseek_v41.swa_segment import (
        SegmentTopology, SwaSegment, compile_segment, load_segment_modules, make_segment_worker,
    )

    torch.set_num_threads(4)
    devices = [int(v) for v in args.devices.split(",")]
    if len(devices) % args.tp or len(set(devices)) != len(devices):
        raise ValueError("unique devices must form complete TP groups")
    topology = SegmentTopology(tp=args.tp, dp=len(devices) // args.tp)
    if args.compare_only:
        _, moe = load_segment_modules(args.lib_root, topology)
        compare_saved(torch.load(args.compare_only, map_location="cpu", weights_only=True), moe, topology,
                      args.residual_profile)
        return
    config = RunConfig(platform="a5", distributed_config=DistributedConfig(device_ids=devices),
                       ring_heap=args.ring_heap_mib << 20, ring_task_window=131072, ring_dep_pool=131072,
                       enable_dump_args=1 if args.dump_tagged else 0)
    compiler = KernelCompiler(run_config=config, cache_dir=args.build_dir)
    programs = compile_segment(compiler, args.lib_root, topology)
    print("COMPILE PASS", flush=True)
    if args.compile_only:
        return
    swa, moe = load_segment_modules(args.lib_root, topology)
    fixture = SimpleNamespace(tokens=topology.capacity, requests=1, dp=topology.dp,
                              seed=11, case="normal", fixture="checkpoint", dp_tokens=None,
                              epochs=1, bench=False)

    def materialize(specs):
        return {s.name: s.create_tensor().contiguous() for s in specs if isinstance(s, TensorSpec)}

    a = materialize(swa.build_hc_specs(fixture))
    token_ids = None
    if args.input_source == "embeddings":
        import json

        supplied_ids = json.loads(Path(args.token_ids).read_text(encoding="utf-8")) if args.token_ids else None
        token_ids = prepare_checkpoint_inputs(a, args.model_dir, topology, supplied_ids)
    request_inputs = prepare_checkpoint_metadata(a, args.model_dir, topology, token_ids) if (
        args.request_metadata) else None
    initial_state = {"residual": a["x_hc"].clone(), "pre_mix": a["incoming_pre_mix"].clone()}
    print(f"Attention fixture ready: input_source={args.input_source}", flush=True)
    if args.model_dir:
        from pypto_serving.model.deepseek_v41.swa_weights import load_swa_layer_weights

        layers = []
        for layer_id in (0, 1):
            print(f"Loading real checkpoint layer {layer_id}", flush=True)
            aw, mw = load_swa_layer_weights(args.model_dir, layer_id, topology)
            layer_a = dict(a, **aw)
            layer_m = dict(mw, next_pre_mix=torch.zeros_like(a["next_pre_mix"]),
                           x_mixed=torch.zeros_like(a["attn_out"]), x_next=torch.zeros_like(a["output"]),
                           num_tokens=torch.full((topology.world,), 16, dtype=torch.int32))
            layers.append((layer_a, layer_m))
    else:
        print("Preparing synthetic packed-FP4 experts", flush=True)
        m = materialize(moe.build_tensor_specs([16] * topology.world))
        layers = [(a, m), (dict(a), dict(m))]
    # Independent output/scratch/cache per layer, even with tied fixture weights.
    for index, (la, lm) in enumerate(layers):
        if index:
            for name in ("window_cache", "window_cache_scale", "output", "next_pre_mix", "hidden", "attn_out"):
                la[name] = la[name].clone()
            for name in ("next_pre_mix", "x_mixed", "x_next"):
                lm[name] = lm[name].clone()
    ac = torch.zeros(topology.world, 1, dtype=torch.int32).share_memory_()
    mc = torch.zeros(topology.world, dtype=torch.int32).share_memory_()
    readback = torch.empty_like(layers[-1][1]["x_next"]).share_memory_()
    mix_readback = torch.empty_like(layers[-1][1]["next_pre_mix"]).share_memory_()
    captured = [({name: torch.empty_like(la[name]).share_memory_() for name in ATTENTION_OUTPUTS},
                 {name: torch.empty_like(lm[name]).share_memory_() for name in MOE_OUTPUTS})
                for la, lm in layers] if args.stage_reference else []
    sources = [v for pair in layers for mapping in pair for v in mapping.values()]
    print("Weights ready; executing two-layer device segment", flush=True)
    with make_segment_worker(programs, config, sources) as worker:
        allocations, uploaded = [], {}

        def upload(values):
            result = {}
            for name, value in values.items():
                if name == "num_tokens":
                    continue
                if id(value) not in uploaded:
                    device = worker.alloc_stacked_tensor(value)
                    allocations.append(device)
                    uploaded[id(value)] = device
                result[name] = uploaded[id(value)]
            return result

        device_layers = [(upload(la), upload(lm)) for la, lm in layers]
        da = device_layers[0][0]
        state = LayerState(da["x_hc"], da["incoming_pre_mix"], "tp_local_token")
        runner = SwaSegment(worker, programs, topology, ac, mc, config)
        for layer_id, (da, dm) in enumerate(device_layers):
            state = runner.run_layer(state, da, dm, group_counts=[topology.capacity] * topology.dp)
            print(f"Device layer {layer_id} complete", flush=True)
            if args.dump_tagged:
                # Preserve completed runtime dumps before the next dispatch reuses its path.
                import shutil

                destination = Path(args.artifact_dir) / f"layer-{layer_id}-dumps"
                for manifest in list(Path(args.build_dir).rglob("args_dump.json")):
                    relative = manifest.parent.relative_to(Path(args.build_dir))
                    target = destination / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists():
                        raise FileExistsError(f"Refusing to overwrite diagnostic dumps: {target}")
                    shutil.move(str(manifest.parent), str(target))
        worker.copy_stacked_from(state.residual, readback)
        worker.copy_stacked_from(state.pre_mix, mix_readback)
        # Read only after both layers finish; never feed diagnostic state back.
        for (da, dm), (ca, cm) in zip(device_layers, captured):
            for device, cpu in ((da, ca), (dm, cm)):
                for name, destination in cpu.items():
                    worker.copy_stacked_from(device[name], destination)
        assert torch.isfinite(readback).all() and torch.isfinite(mix_readback).all()
        assert readback.abs().max() > 0 and mix_readback.abs().max() > 0
        assert not torch.equal(readback, a["x_hc"]), "residual was not updated"
        assert not torch.equal(mix_readback, a["incoming_pre_mix"]), "pre_mix was not updated"
        for value in reversed(allocations):
            worker.free_stacked_tensor(value)
    print("DEVICE TWO-LAYER SMOKE PASS", flush=True)
    if args.reference:
        # References run after worker shutdown and never feed device execution.
        # All weight buffers are read-only; clone only mutable state/scratch.
        residual, mix = a["x_hc"], a["incoming_pre_mix"]
        for layer_id, (la, lm) in enumerate(layers):
            ra = dict(la, x_hc=residual, incoming_pre_mix=mix)
            for name in ("window_cache", "window_cache_scale", "output", "next_pre_mix", "hidden", "attn_out"):
                ra[name] = la[name].clone()
            swa.golden_prefill_swa_case(ra)
            rm = dict(lm, x_hc=ra["output"], pre_mix=ra["next_pre_mix"])
            for name in ("next_pre_mix", "x_mixed", "x_next"):
                rm[name] = lm[name].clone()
            moe.golden_moe(rm)
            residual, mix = rm["x_next"], rm["next_pre_mix"]
            print(f"Torch reference layer {layer_id} complete", flush=True)
        artifact = Path(args.artifact_dir)
        artifact.mkdir(parents=True, exist_ok=True)
        data = {"input_source": args.input_source, "token_ids": token_ids, "initial_state": initial_state,
                "request_inputs": request_inputs,
                "actual_residual": readback, "expected_residual": residual,
                "actual_pre_mix": mix_readback, "expected_pre_mix": mix}
        torch.save(data, artifact / "comparison.pt")
        if captured:
            data["stages"] = compare_stages(layers, captured, (a["x_hc"], a["incoming_pre_mix"]),
                                            swa, moe, topology)
            torch.save(data, artifact / "comparison.pt")
        compare_saved(data, moe, topology, args.residual_profile)


if __name__ == "__main__":
    main()

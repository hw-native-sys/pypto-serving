# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded real-weight compressed Attention -> MoE device diagnostic.

Start from the saved output of the two SWA layers. This does not claim that
that input passed its accumulated precision gate, or run a full model. Only
existing native same-input stage comparators gate this diagnostic.
With --family c1a the saved state is an explicitly injected diagnostic input,
not a claim that layers 2 through 19 have executed.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace


def select_plans(raw, family, last_layer=None):
    """Select a consecutive chain beginning with the family's real Full producer."""
    from pypto_serving.model.deepseek_v41.execution_plan import plan_layers

    if family not in ("c1a", "c2a"):
        raise ValueError("diagnostic family must be c1a or c2a")
    layers = plan_layers(raw)
    first = next(p.layer_id for p in layers if p.mode == family + "_full")
    last = first + 1 if last_layer is None else last_layer
    if type(last) is not int or not first <= last < len(layers):
        raise ValueError("last layer must belong to the selected compressed family")
    selected = layers[first:last + 1]
    if any(not p.mode.startswith(family + "_") for p in selected):
        raise ValueError("diagnostic cannot cross compressed families")
    seen = set()
    for plan in selected:
        seen.add(plan.layer_id)
        if any(source is not None and source not in seen
               for source in (plan.kv_source, plan.index_source, plan.candidate_source)):
            raise ValueError("diagnostic is missing a preceding state producer")
    return selected


def state_names(module, mode):
    """Capture both mutable and read-only state checked by the native contract."""
    if hasattr(module, "STATE_NAMES"):
        return tuple(dict.fromkeys((*module.STATE_NAMES[mode],
            "compressed_cache", "compressed_cache_scale", "index_cache", "index_cache_scale",
            "topk_indices", "candidate_mask")))
    return module.ATTENTION_STATE[mode]


def select_active_state(saved, topology, group_counts, starts=None):
    """Select causal prefixes for a ragged diagnostic, preserving the source artifact."""
    import torch

    topology.counts(group_counts)
    starts = [0] * topology.dp if starts is None else starts
    if len(starts) != topology.dp or any(type(start) is not int or start < 0 or
            start + count > topology.capacity for start, count in zip(starts, group_counts)):
        raise ValueError("diagnostic chunks must fit within the saved causal source")
    residual, mix = saved["actual_residual"], saved["actual_pre_mix"]
    if residual.shape != (topology.world, topology.local_capacity, 4, 5120) or mix.shape != residual.shape[:-1]:
        raise ValueError("saved SWA state has incompatible topology or shape")
    if residual.dtype != torch.float32 or mix.dtype != torch.float32:
        raise ValueError("saved SWA state must preserve its FP32 storage")
    if not torch.isfinite(residual).all() or not torch.isfinite(mix).all():
        raise ValueError("saved SWA state must be finite")
    selected, selected_mix = torch.zeros_like(residual), torch.zeros_like(mix)
    for group, (start, count) in enumerate(zip(starts, group_counts)):
        ranks = slice(group * topology.tp, (group + 1) * topology.tp)
        selected[ranks].flatten(0, 1)[:count] = residual[ranks].flatten(0, 1)[start:start + count]
        selected_mix[ranks].flatten(0, 1)[:count] = mix[ranks].flatten(0, 1)[start:start + count]
    return selected, selected_mix


def diagnostic_chunks(topology, counts, continue_to_capacity=False, repeat_chunks=1):
    """Separate absolute positions from offsets into the bounded injected source."""
    topology.counts(counts)
    if type(repeat_chunks) is not int or not 1 <= repeat_chunks <= 16:
        raise ValueError("repeat chunks must be between 1 and 16")
    if repeat_chunks > 1:
        if continue_to_capacity or not any(counts) or any(n not in (0, topology.capacity) for n in counts):
            raise ValueError("repeated input requires full chunks without partial continuation")
        return [(list(counts), [i * count for count in counts], [0] * topology.dp)
                for i in range(repeat_chunks)]
    chunks = [(list(counts), [0] * topology.dp, [0] * topology.dp)]
    if continue_to_capacity:
        remaining = [topology.capacity - n if n else 0 for n in counts]
        if not any(remaining):
            raise ValueError("continuation requires an unfinished nonempty request")
        chunks.append((remaining, list(counts), list(counts)))
    return chunks


def diagnostic_step(ids, topology, counts, starts, source_starts, context_tokens, family):
    """Build private full-history pages; repeated IDs remain diagnostic input only."""
    from pypto_serving.model.deepseek_v41.request_state import ForwardStep, RequestSlice

    ratio = 1 if family == "c1a" else 2
    pages = {"window": tuple(range((context_tokens + 127) // 128)),
             "cmp": tuple(range((context_tokens // ratio + 127) // 128))}
    return ForwardStep("prefill", tuple(
        RequestSlice(str(g), g, 0, starts[g],
                     tuple(row[source_starts[g]:source_starts[g] + counts[g]].tolist()),
                     context_tokens, pages)
        for g, row in enumerate(ids) if counts[g]), 1)


def size_diagnostic_caches(values, context_tokens, family):
    """Extend only dynamic context axes, preserving lib payload layouts and dtypes."""
    import torch

    window_pages = (context_tokens + 127) // 128
    compressed_pages = (context_tokens // (1 if family == "c1a" else 2) + 127) // 128
    for name in ("window_cache", "window_cache_scale", "compressed_cache",
                 "compressed_cache_scale", "index_cache", "index_cache_scale"):
        if name not in values:
            continue
        old = values[name]
        pages = window_pages if name.startswith("window") else compressed_pages
        if pages > old.shape[1]:
            values[name] = torch.empty((old.shape[0], pages, *old.shape[2:]), dtype=old.dtype)
    if family == "c1a":
        old = values["candidate_mask"]
        columns = compressed_pages * 128
        if columns > old.shape[-1]:
            values["candidate_mask"] = torch.empty((*old.shape[:-1], columns), dtype=old.dtype)


def prepare(args, topology, module, weight_bundles=None):
    import torch
    from golden.spec import TensorSpec
    from models.deepseek_v4_1_flash.config import FLASH
    from models.deepseek_v4_1_flash.rope_tables import precompute_rope_tables
    from pypto_serving.model.deepseek_v41.compressed_metadata import prepare_compressed_metadata
    from pypto_serving.model.deepseek_v41.swa_metadata import gather_swa_rope_rows, prepare_swa_window_metadata
    from pypto_serving.model.deepseek_v41.swa_weights import load_prefill_layer_weights

    saved = torch.load(args.input_state, map_location="cpu", weights_only=True)
    if saved.get("request_inputs") is None or saved.get("input_source") != "embeddings":
        raise ValueError("input must come from the explicit fresh-request embedding diagnostic")
    ids = saved["token_ids"]
    if ids is None or tuple(ids.shape) != (topology.dp, topology.capacity):
        raise ValueError("saved SWA diagnostic must contain exactly the same packed token capacity")
    group_counts = args.group_counts
    starts = getattr(args, "starts", [0] * topology.dp)
    source_starts = getattr(args, "source_starts", starts)
    context_tokens = getattr(args, "context_tokens", topology.capacity)
    weight_bundles = {} if weight_bundles is None else weight_bundles
    global_counts, local_counts = topology.counts(group_counts)
    residual, mix = select_active_state(saved, topology, group_counts, source_starts)
    raw = json.loads((Path(args.model_dir) / "config.json").read_text())
    family = getattr(args, "family", "c2a")
    plans = select_plans(raw, family, getattr(args, "last_layer", None))
    text = raw["text_config"]
    for key in ("qk_rope_head_dim", "rope_theta", "compress_rope_theta"):
        if text[key] != getattr(FLASH, key):
            raise ValueError(f"lib/checkpoint RoPE mismatch: {key}")
    for source, target in (("factor", "rope_factor"), ("beta_fast", "beta_fast"),
                           ("beta_slow", "beta_slow"), ("original_max_position_embeddings",
                                                      "original_max_position_embeddings")):
        if text["rope_scaling"][source] != getattr(FLASH, target):
            raise ValueError(f"lib/checkpoint compressed RoPE mismatch: {source}")
    # Private physical pages per DP group, shared only through declared producers.
    step = diagnostic_step(ids, topology, group_counts, starts, source_starts, context_tokens, family)
    tables = precompute_rope_tables(context_tokens, False)
    compressed_tables = precompute_rope_tables(context_tokens, True)
    fixture = SimpleNamespace(tokens=topology.capacity, requests=1, dp=topology.dp,
                              seed=11, case="mixed", dp_tokens=None, epochs=1, bench=False)
    attention, moe = {}, {}
    for plan in plans:
        mode = plan.mode.split("_")[1]
        if family == "c1a":
            fixture.case, fixture.dp_tokens = "causal", list(group_counts)
            specs = module.build_specs(fixture, {})
        else:
            specs = module.build_specs(fixture, mode, {})
        values = {s.name: s.create_tensor().contiguous() for s in specs
                  if isinstance(s, TensorSpec)}
        size_diagnostic_caches(values, context_tokens, family)
        if plan.layer_id not in weight_bundles:
            print(f"Loading real checkpoint layer {plan.layer_id}", flush=True)
            weight_bundles[plan.layer_id] = load_prefill_layer_weights(args.model_dir, plan.layer_id, topology)
        aw, mw = weight_bundles[plan.layer_id]
        values.update(aw, x_hc=residual, pre_mix=mix)
        values["num_tokens"] = torch.tensor(global_counts, dtype=torch.int32).reshape(-1, 1)
        window = prepare_swa_window_metadata(step, topology, cache_pages=values["window_cache"].shape[1])
        values.update(window_slots=window.window_slots, window_indices=window.window_indices)
        values["window_cache"].view(torch.uint8).zero_()
        values["window_cache_scale"].view(torch.uint8).fill_(127)
        if family == "c1a":
            cm = prepare_compressed_metadata(step, topology, ratio=1, compressed_group="cmp",
                cache_pages=values["compressed_cache"].shape[1], max_requests=1, state_blocks=1)
            values.update(request_ids=cm.request_ids, compressed_lens=cm.compressed_lens,
                          compressed_slots=cm.compressed_slots, index_block_table=cm.index_block_table)
            values["rope_cos"], values["rope_sin"] = gather_swa_rope_rows(window, tables)
            values["compressed_rope_cos"], values["compressed_rope_sin"] = gather_swa_rope_rows(
                window, compressed_tables)
            for name in ("compressed_cache", "index_cache"):
                values[name].view(torch.uint8).zero_()
            values["compressed_cache_scale"].view(torch.uint8).fill_(0x38)
            values["index_cache_scale"].view(torch.uint8).fill_(127)
            values["topk_indices"].fill_(-1)
            values["compressed_indices"].fill_(-1)
            values["candidate_mask"].zero_()
        elif mode == "full":
            cm = prepare_compressed_metadata(step, topology, ratio=2, compressed_group="cmp",
                cache_pages=values["compressed_cache"].shape[1], max_requests=1,
                state_blocks=values["state_cache"].shape[1])
            values.update(token_to_req_indices=cm.request_ids, compressed_lens=cm.compressed_lens,
                          compressed_slots=cm.compressed_slots, index_block_table=cm.index_block_table,
                          position_ids=cm.position_ids, query_start_loc=cm.query_start_loc,
                          state_block_table=cm.state_block_table,
                          compressed_rope_positions=cm.compressed_rope_positions)
            for name, table in zip(("freqs_cos", "freqs_sin", "compressed_freqs_cos", "compressed_freqs_sin"),
                                   (*tables, *compressed_tables)):
                values[name] = table.unsqueeze(0).repeat(topology.world, 1, 1)
            for name in ("compressed_cache", "index_cache"):
                values[name].view(torch.uint8).zero_()
            values["compressed_cache_scale"].view(torch.uint8).fill_(0x38)  # E4M3 scale 1
            values["index_cache_scale"].view(torch.uint8).fill_(127)  # E8M0 scale 1
            values["state_cache"].zero_()
            values["topk_indices"].fill_(-1)
        else:
            values["rope_cos"], values["rope_sin"] = gather_swa_rope_rows(window, tables)
        attention[plan.layer_id] = values
        moe[plan.layer_id] = dict(mw, next_pre_mix=torch.zeros_like(mix),
            x_mixed=torch.zeros(topology.world, topology.local_capacity, 5120, dtype=torch.bfloat16),
            x_next=torch.zeros_like(residual), num_tokens=torch.tensor(local_counts, dtype=torch.int32))
    return plans, attention, moe, residual, mix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lib-root", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--input-state", required=True)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--family", choices=("c2a", "c1a"), default="c2a")
    parser.add_argument("--last-layer", type=int, help="Inclusive last layer; starts at the real Full producer")
    parser.add_argument("--group-counts", help="Comma-separated causal prefix lengths per DP group")
    parser.add_argument("--continue-to-capacity", action="store_true",
                        help="Run a second chunk of each nonempty request using its resident caches")
    parser.add_argument("--repeat-input-chunks", type=int, default=1,
                        help="Diagnostic only: repeat the saved full input 2-16 times at advancing positions")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dump-tagged", action="store_true",
                        help="Preserve lib-tagged kernel arguments after each diagnostic layer")
    parser.add_argument("--build-dir")
    parser.add_argument("--artifact-dir")
    parser.add_argument("--ring-heap-mib", type=int, default=4096)
    args = parser.parse_args()
    args.build_dir = args.build_dir or f"build_output/v41-{args.family}-chain"
    args.artifact_dir = args.artifact_dir or f".validation-artifacts/{args.family}-chain"
    import torch
    from pypto_serving.model.deepseek_v41.swa_segment import SegmentTopology, load_segment_modules
    from pypto_serving.model.deepseek_v41.prefill_segment import (
        PrefillSegment, bind_prefill_producers, compile_prefill_segments,
    )
    devices = tuple(map(int, args.devices.split(",")))
    if args.tp <= 0 or len(set(devices)) != len(devices) or len(devices) % args.tp:
        raise ValueError("unique devices must form complete TP groups")
    topology = SegmentTopology(tp=args.tp, dp=len(devices) // args.tp)
    args.group_counts = ([topology.capacity] * topology.dp if args.group_counts is None
                         else [int(n) for n in args.group_counts.split(",")])
    topology.counts(args.group_counts)
    torch.set_num_threads(4)
    _, moe_module = load_segment_modules(args.lib_root, topology)
    from models.deepseek_v4_1_flash import prefill_c2a_full as common
    if args.family == "c1a":
        from models.deepseek_v4_1_flash import prefill_c1a_sp as module
    else:
        module = common
    chunk_plan = diagnostic_chunks(topology, args.group_counts, args.continue_to_capacity,
                                   args.repeat_input_chunks)
    chunks = [(counts, starts) for counts, starts, _ in chunk_plan]
    context_tokens = topology.capacity * args.repeat_input_chunks
    prepared, weights = [], {}
    for counts, starts, source_starts in chunk_plan:
        options = SimpleNamespace(**{**vars(args), "group_counts": counts, "starts": starts,
                                     "source_starts": source_starts, "context_tokens": context_tokens})
        plans, attention, moe, residual, mix = prepare(options, topology, module, weights)
        if prepared:
            for plan in plans:
                for name in state_names(module, plan.mode.split("_")[1]):
                    if "cache" in name:
                        attention[plan.layer_id][name] = prepared[0][0][plan.layer_id][name]
        prepared.append((bind_prefill_producers(plans, attention), moe, residual, mix, counts))
    from pypto_serving.model.deepseek_v41.prefill_segment import PREFILL_ARGUMENTS
    dynamic = {"attention_epoch", "num_tokens"}
    for bound, *_ in prepared:
        for plan in plans:
            for name in PREFILL_ARGUMENTS[plan.mode]:
                if name not in dynamic:
                    assert isinstance(bound[plan.layer_id][name], torch.Tensor), name
    family_label = args.family.upper()
    print(f"REAL CHECKPOINT {family_label} PREPARATION PASS; layers {[p.layer_id for p in plans]}", flush=True)
    if args.prepare_only:
        return
    from pypto.ir import DistributedConfig
    from pypto.runtime import RunConfig
    from pypto_serving.model.common.compiler.compiler import KernelCompiler
    from pypto_serving.model.deepseek_v41.composite import LayerState
    from pypto_serving.model.deepseek_v41.swa_segment import make_segment_worker
    from golden.validation import ratio_allclose
    artifact = Path(args.artifact_dir)
    artifact.mkdir(parents=True, exist_ok=True)
    if (artifact / "comparison.pt").exists():
        raise FileExistsError("refusing to overwrite a previous comparison")
    config = RunConfig(platform="a5", distributed_config=DistributedConfig(device_ids=list(devices)),
                       ring_heap=args.ring_heap_mib << 20, ring_task_window=131072, ring_dep_pool=131072,
                       enable_dump_args=1 if args.dump_tagged else 0)
    compiler = KernelCompiler(run_config=config, cache_dir=args.build_dir)
    programs, ffn = compile_prefill_segments(compiler, args.lib_root, topology, [p.mode for p in plans])
    ac = torch.zeros(topology.world, 1, dtype=torch.int32).share_memory_()
    mc = torch.zeros(topology.world, dtype=torch.int32).share_memory_()
    captures = []
    for bound, moe, *_ in prepared:
        captured = {}
        for plan in plans:
            layer = plan.layer_id
            names = ("attn_input", "attn_output", "next_pre_mix", "x_hc_out") + state_names(
                module, plan.mode.split("_")[1])
            captured[layer] = ({n: torch.empty_like(bound[layer][n]).share_memory_() for n in names},
                              {n: torch.empty_like(moe[layer][n]).share_memory_()
                               for n in ("x_next", "next_pre_mix", "x_mixed")})
        captures.append(captured)
    sources = [v for bound, moe, *_ in prepared for maps in (bound, moe)
               for values in maps.values() for v in values.values()]
    with make_segment_worker([*programs.values(), ffn], config, sources) as worker:
        uploaded = {}
        def upload(values):
            result = {}
            for name, value in values.items():
                if name == "num_tokens":
                    continue
                if id(value) not in uploaded:
                    uploaded[id(value)] = worker.alloc_stacked_tensor(value)
                result[name] = uploaded[id(value)]
            return result
        device_steps = [({layer: upload(values) for layer, values in bound.items()},
                         {layer: upload(values) for layer, values in moe.items()})
                        for bound, moe, *_ in prepared]
        runner = PrefillSegment(worker, programs, ffn, topology, ac, mc, config)
        for step_id, ((da, dm), captured) in enumerate(zip(device_steps, captures)):
            first = plans[0].layer_id
            state = LayerState(da[first]["x_hc"], da[first]["pre_mix"], "tp_local_token")
            if args.dump_tagged:
                import shutil

                # Same composite calls and epochs; only completed diagnostic files
                # are moved before a repeated program can reuse the dump path.
                for plan in plans:
                    state = runner.run_layer(state, da[plan.layer_id], dm[plan.layer_id],
                                             group_counts=prepared[step_id][4], mode=plan.mode)
                    destination = artifact / f"chunk-{step_id}-layer-{plan.layer_id}-dumps"
                    for manifest in list(Path(args.build_dir).rglob("args_dump.json")):
                        target = destination / manifest.parent.relative_to(Path(args.build_dir))
                        if target.exists():
                            raise FileExistsError(f"Refusing to overwrite diagnostic dumps: {target}")
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(manifest.parent), str(target))
            else:
                runner.run_chain(state, plans, da, dm, group_counts=prepared[step_id][4])
            for layer, (ca, cm) in captured.items():
                for device, host in ((da[layer], ca), (dm[layer], cm)):
                    for name, destination in host.items():
                        worker.copy_stacked_from(device[name], destination)
            print(f"{family_label} DEVICE CHUNK {step_id} COMPLETE", flush=True)
        for handle in reversed(list(uploaded.values())):
            worker.free_stacked_tensor(handle)
    # CPU same-input references run only after worker shutdown. Device state
    # never depends on these diagnostic readbacks or reference results.
    records, passed = [], True
    for step_id, ((bound, moe, residual, mix, counts), captured) in enumerate(zip(prepared, captures)):
        for plan in plans:
            layer, mode = plan.layer_id, plan.mode.split("_")[1]
            actual_a, actual_m = captured[layer]
            inputs = dict(bound[layer], x_hc=residual, pre_mix=mix)
            if step_id:
                for name, value in captures[step_id - 1][layer][0].items():
                    if "cache" in name:
                        inputs[name] = value
            if mode != "full":
                for name in ("compressed_cache", "compressed_cache_scale"):
                    inputs[name] = captured[plan.kv_source][0][name]
                if args.family == "c1a":
                    for name in ("index_cache", "index_cache_scale"):
                        inputs[name] = captured[plan.kv_source][0][name]
                    inputs["candidate_mask"] = captured[plan.candidate_source][0]["candidate_mask"]
                if mode == "reuse":
                    inputs["compressed_indices"] = captured[plan.index_source][0]["topk_indices"]
            initial_cache = {n: inputs[n].clone() for n in state_names(module, mode)}
            expected_a = dict(inputs)
            for name in actual_a:
                expected_a[name] = inputs[name].clone()
            golden = (common.make_golden(mode, 1, attention_reference=module.reference_attention,
                       state_names=module.STATE_NAMES[mode]) if args.family == "c1a"
                      else common.make_golden(mode, 1))
            golden(expected_a)
            expected_m = dict(moe[layer], x_hc=actual_a["x_hc_out"], pre_mix=actual_a["next_pre_mix"])
            for name in actual_m:
                expected_m[name] = moe[layer][name].clone()
            moe_module.golden_moe(expected_m)
            mc_check = {
                "next_pre_mix": ratio_allclose(atol=2.5e-5, rtol=5e-3),
                "x_mixed": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
                "x_next": moe_module._local_mhc_compare(list(topology.counts(counts)[1])),
            }
            for label, actual, expected, checks in (("attention", actual_a, expected_a,
                    module.make_compare(mode, 1, initial_cache)), ("moe", actual_m, expected_m, mc_check)):
                results = {}
                for name, check in checks.items():
                    ok, detail = check(actual[name], expected[name], inputs=expected,
                        actual_outputs=actual, expected_outputs=expected, rtol=1e-3, atol=1e-3)
                    print(f"NATIVE STAGE chunk={step_id} layer={layer} {label}.{name}: {ok} {detail}", flush=True)
                    results[name] = (bool(ok), detail)
                    passed &= bool(ok)
                records.append(dict(chunk=step_id, layer=layer, stage=label, results=results, actual=actual,
                                    expected={n: expected[n] for n in actual}))
            residual, mix = actual_m["x_next"], actual_m["next_pre_mix"]
    torch.save({"input_state": str(args.input_state), "family": args.family,
                "layer_ids": [p.layer_id for p in plans], "injected_boundary_input": True,
                "group_counts": args.group_counts, "chunks": chunks, "stages": records,
                "repeat_input_chunks": args.repeat_input_chunks, "context_tokens": context_tokens,
                "source_starts": [source for _, _, source in chunk_plan],
                "actual_residual": residual, "actual_pre_mix": mix}, artifact / "comparison.pt")
    if not passed:
        raise AssertionError(f"{family_label} chain native stage check failed; see comparison.pt")
    print(f"{family_label} CHAIN NATIVE STAGES PASS; accumulated full-model acceptance remains pending", flush=True)


if __name__ == "__main__":
    main()

# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Standalone GLM checkpoint validation using serving-owned execution resources."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch

from pypto_serving.model.glm53.kernel_api import import_kernel
from pypto_serving.model.glm53.npu_executor import prepare_kernels
from pypto_serving.model.glm53.npu_runner import Glm53ModelRunner

DEFAULT_CHAT_TEMPLATE = (
    "{% for message in messages %}{{ '<|user|>\n' + message['content'] + '<|assistant|>\n' }}{% endfor %}"
)
DEFAULT_PROMPTS = (
    "What is 2 + 3? Reply with just the number.",
    "What is the capital of France? Reply with just the city name.",
    "Translate 'hello' into Chinese. Reply with only the translation.",
    "Continue the sequence with three numbers: 1, 2, 3,",
    "Name the largest planet in the solar system. One word.",
    "What is 7 times 8? Reply with just the number.",
    "Which continent is Egypt in? One word.",
    "Spell the word 'computer'. Reply with only the spelling.",
    "What color do you get mixing blue and yellow? One word.",
    "How many days are in a leap year? Reply with just the number.",
    "Name the ocean between Europe and America. One word.",
    "What is the square root of 81? Reply with just the number.",
    "In what language is 'bonjour' a greeting? One word.",
    "Count from one to three in Spanish. Only the numbers.",
    "What is the chemical symbol for gold? Reply with only the symbol.",
    "Finish the proverb: 'An apple a day...'",
)


def _resolve_eos_ids(checkpoint, tokenizer, tokenizer_config: dict) -> set[int]:
    """Every generation-stop token id the checkpoint declares.

    ``generation_config.json``'s ``eos_token_id`` wins (the GLM-5.3-Flash snapshot
    ships the list ``[154820, 154827, 154829]``); a checkpoint without one falls
    back to the ``tokenizer_config.json`` ``eos_token`` spelling mapped through
    the tokenizer.
    """
    generation_path = checkpoint.root / "generation_config.json"
    value = None
    if generation_path.exists():
        value = json.loads(generation_path.read_text()).get("eos_token_id")
    ids = [int(token_id) for token_id in (value if isinstance(value, list) else [value] if value is not None else [])]
    if not ids:
        eos = tokenizer_config.get("eos_token")
        token_id = None if eos is None else tokenizer.token_to_id(eos["content"] if isinstance(eos, dict) else eos)
        if token_id is None:
            raise ValueError("checkpoint EOS token is missing from its tokenizer")
        ids = [int(token_id)]
    return set(ids)


def _render_prompt(prompt: str, chat_template: str | None) -> str:
    """Wrap one user prompt for encoding: the checkpoint template, else ChatML.

    Rendering follows the transformers convention: ``add_generation_prompt=True``
    lets the template append its assistant header (the GLM-5.3-Flash template
    gates ``<|assistant|>`` on that flag). Any template error falls back to the
    verbatim ChatML wrap.
    """
    template = chat_template if chat_template else DEFAULT_CHAT_TEMPLATE
    try:
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        # loopcontrols: the GLM-5.3-Flash template breaks out of loops; without
        # the extension jinja rejects the template and the verbatim fallback
        # would silently mask it.
        environment = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"],
        )
        return environment.from_string(template).render(
            messages=[{"role": "user", "content": prompt}], add_generation_prompt=True,
        )
    except Exception as error:  # noqa: BLE001 -- fall back verbatim on any template error
        print("GLM_CHAT_TEMPLATE_FALLBACK", repr(error), flush=True)
        return f"<|user|>\n{prompt}<|assistant|>\n"


class _VisionEncoder:
    """Execute the checkpoint's BF16 vision tower on one allocated device."""

    def __init__(self, api, checkpoint, device_id, *, platform="a2a3", compiled=None):
        from pypto.ir import DistributedConfig
        from pypto.runtime import RunConfig

        self._api = api
        self.checkpoint = checkpoint if isinstance(checkpoint, api.Checkpoint) else api.Checkpoint(checkpoint)
        self.weights = self.checkpoint.vision_weights()
        self.config = RunConfig(
            platform=platform,
            distributed_config=DistributedConfig(device_ids=[device_id], num_sub_workers=0),
        )
        self.program = api.vision_forward.compile(config=self.config) if compiled is None else compiled
        config = self.checkpoint._json("config.json")["vision_config"]
        rope = config.get("rope_parameters") or {"rope_type": "axial", "rope_theta": 10000.0}
        if rope.get("rope_type") != "axial" or float(rope.get("rope_theta", 0)) <= 0:
            raise ValueError("vision RoPE must use an axial positive-frequency base")
        powers = torch.arange(api.VISION_HEAD_DIM // 4).float() * 4 / api.VISION_HEAD_DIM
        self.inv_freq = float(rope["rope_theta"]) ** (-powers)

    def __call__(self, pixel_values, grid_thw):
        api = self._api
        positions, starts = api.vision_metadata(grid_thw)
        if tuple(pixel_values.shape) != (positions.shape[0], api.VISION_PATCH_IN):
            raise ValueError("pixel_values rows must match grid_thw and the flattened patch width")
        if not torch.isfinite(pixel_values).all():
            raise ValueError("pixel_values must be finite")
        patches = pixel_values.cpu().bfloat16().contiguous()
        features = torch.empty((patches.shape[0] // api.VISION_MERGE**2, api.D), dtype=torch.bfloat16)
        rows = patches.shape[0]
        workspace_a = torch.zeros((1, rows, api.VISION_HIDDEN), dtype=torch.bfloat16)
        workspace_b = torch.zeros_like(workspace_a)
        cosine = torch.empty((1, rows, api.VISION_HEAD_DIM), dtype=torch.float32)
        sine = torch.empty_like(cosine)
        cosine_table, sine_table = api.vision_rope_tables(positions, self.inv_freq)
        arguments = [
            patches.unsqueeze(0),
            *(self.weights[name].unsqueeze(0) for name in api.vision_weight_layouts()),
            positions.unsqueeze(0),
            starts.unsqueeze(0),
            cosine_table.unsqueeze(0),
            sine_table.unsqueeze(0),
            features.unsqueeze(0),
            workspace_a,
            workspace_b,
            cosine,
            sine,
            torch.tensor(api.VISION_DEPTH, dtype=torch.int32),
        ]
        self.program(*arguments, config=self.config)
        if not torch.isfinite(features).all():
            raise FloatingPointError("vision encoder produced nonfinite features")
        return features


def generate(
    api,
    checkpoint,
    programs: dict,
    device_ids: list[int],
    prompts: list[str],
    max_new_tokens: int,
    *,
    raw_prompts: bool = False,
    platform: str = "a2a3",
    expect_tokens: dict[int, list[int]] | None = None,
    dump_path: Path | None = None,
    visual_inputs: dict[int, list] | None = None,
    spec_tokens: int = 0,
):
    """Generate incrementally with resident caches and optional MTP verification.

    Report legacy ``seconds`` from runtime preparation and step ``seconds`` from
    dispatch, plus separate checkpoint/input setup and end-to-end wall times.
    """
    BATCH, CAPACITY, MAX_SEQ_LEN = api.BATCH, api.CAPACITY, api.MAX_SEQ_LEN
    CHUNK_TILE, MTP_SPEC_TOKENS, FLASH = api.CHUNK_TILE, api.MTP_SPEC_TOKENS, api.FLASH
    IMAGE_TOKEN_ID, NUM_LAYERS = api.IMAGE_TOKEN_ID, api.NUM_LAYERS
    TP_SIZE, EP_SIZE, DP_SIZE = api.TP_SIZE, api.EP_SIZE, api.DP_SIZE
    CACHE_TOKENS = api.CACHE_TOKENS
    vision_api = None
    if visual_inputs:
        vision_api = import_kernel(
            "vision", capacity=CAPACITY, max_seq_len=MAX_SEQ_LEN, tp_size=api.TP_SIZE,
            ep_size=api.EP_SIZE, cache_tokens=api.CACHE_TOKENS, context_limit=None,
        )
        VisualInput = vision_api.VisualInput
    end_to_end_start = time.monotonic()
    from tokenizers import Tokenizer

    if type(spec_tokens) is not int or spec_tokens not in range(MTP_SPEC_TOKENS + 1):
        raise ValueError(f"spec_tokens must be in 0..{MTP_SPEC_TOKENS}")
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    if not 1 <= len(prompts) <= BATCH:
        raise ValueError(f"provide between 1 and {BATCH} prompts")

    tokenizer = Tokenizer.from_file(str(checkpoint.root / "tokenizer.json"))
    tokenizer_config_path = checkpoint.root / "tokenizer_config.json"
    tokenizer_config = json.loads(tokenizer_config_path.read_text()) if tokenizer_config_path.exists() else {}
    # The transformers convention: an inline tokenizer_config chat_template wins,
    # else the checkpoint's own chat_template.jinja beside the weights (the
    # GLM-5.3-Flash snapshot ships the file and no inline field).
    chat_template = tokenizer_config.get("chat_template")
    if not chat_template:
        template_path = checkpoint.root / "chat_template.jinja"
        if template_path.exists():
            chat_template = template_path.read_text()
    eos_ids = _resolve_eos_ids(checkpoint, tokenizer, tokenizer_config)

    visual_features = {}
    if visual_inputs:

        prompts = list(prompts)
        packed_visuals = []
        for request, visuals in visual_inputs.items():
            if type(request) is not int or request not in range(len(prompts)) or not visuals:
                raise ValueError("visual inputs need a nonempty list for a valid prompt index")
            for visual in visuals:
                if not isinstance(visual, VisualInput):
                    raise ValueError("visual inputs must be produced by VisionProcessor")
                packed_visuals.append((request, visual))
            remainder = prompts[request]
            expanded = []
            if "<|image|>" not in remainder and "<|video|>" not in remainder:
                prompts[request] = "\n".join(visual.prompt_fragment for visual in visuals) + "\n" + remainder
            else:
                for visual in visuals:
                    candidates = [
                        (remainder.find(marker), marker)
                        for marker in ("<|image|>", "<|video|>")
                        if marker in remainder
                    ]
                    if not candidates or min(candidates)[1] != visual.marker:
                        raise ValueError("prompt visual markers must match the supplied visual input order")
                    before, remainder = remainder.split(visual.marker, 1)
                    expanded.extend((before, visual.prompt_fragment))
                if "<|image|>" in remainder or "<|video|>" in remainder:
                    raise ValueError("each prompt visual marker needs one visual input")
                prompts[request] = "".join(expanded) + remainder
        pixels = torch.cat([visual.pixel_values for _, visual in packed_visuals])
        grid = torch.cat([visual.grid_thw for _, visual in packed_visuals])
        encoder = _VisionEncoder(vision_api, checkpoint, device_ids[0], platform=platform, compiled=programs.get("_vision"))
        encoded = encoder(pixels, grid)
        offset = 0
        for request, visual in packed_visuals:
            count = int(visual.grid_thw.prod(dim=-1).sum()) // 4
            part = encoded[offset : offset + count]
            visual_features[request] = (
                torch.cat((visual_features[request], part)) if request in visual_features else part
            )
            offset += count
        del encoder

    sequences = [
        tokenizer.encode(
            prompt if raw_prompts else _render_prompt(prompt, chat_template),
            add_special_tokens=False,
        ).ids
        for prompt in prompts
    ]
    if any(not sequence for sequence in sequences):
        raise ValueError("prompts must contain at least one token")
    if any(len(sequence) + max_new_tokens > MAX_SEQ_LEN for sequence in sequences):
        raise ValueError(f"prompt and generation exceed the compiled context limit {MAX_SEQ_LEN}")

    generated: list[list[int]] = [[] for _ in prompts]
    finished = [False] * len(prompts)
    step_log: list[dict] = []
    runner = Glm53ModelRunner(
        api, programs, checkpoint, device_ids, platform, enable_mtp=bool(spec_tokens),
    )
    try:
        runner.open()
        start = runner.runtime_started_at
        logits = torch.empty(len(prompts), FLASH.vocab_size, dtype=torch.float32)
        remaining = [list(sequence) for sequence in sequences]
        feature_offsets = [0] * len(prompts)
        draft_queues = [[] for _ in prompts]
        for step in range(max_new_tokens):
            tick = time.monotonic()
            dispatch_seconds = input_setup_seconds = 0.0
            while any(remaining):
                budget, slots, chunks, features = CAPACITY, [], [], {}
                for request, tokens in enumerate(remaining):
                    if not tokens or budget == 0:
                        continue
                    count = min(len(tokens), budget)
                    if count < len(tokens):
                        count -= count % CHUNK_TILE
                    if count == 0:
                        continue
                    chunk, remaining[request] = tokens[:count], tokens[count:]
                    slots.append(request)
                    chunks.append(chunk)
                    if request in visual_features:

                        count_visual = chunk.count(IMAGE_TOKEN_ID)
                        begin = feature_offsets[request]
                        if count_visual:
                            features[request] = visual_features[request][begin : begin + count_visual]
                            feature_offsets[request] += count_visual
                    budget -= count
                dispatch_start = time.monotonic()
                partial = runner.forward_chunks(
                    chunks, slots, visual_features=features,
                    phase="prefill" if step == 0 else "decode",
                )
                dispatch_seconds += time.monotonic() - runner.last_dispatch_started_at
                input_setup_seconds += runner.last_dispatch_started_at - dispatch_start
                logits[slots] = partial
                if spec_tokens:
                    next_tokens = [
                        remaining[slot][0] if remaining[slot] else int(partial[index].argmax())
                        for index, slot in enumerate(slots)
                    ]
                    runner.update_mtp(next_tokens)
            next_ids = logits.argmax(dim=-1).tolist()
            for request, token in enumerate(next_ids):
                if not finished[request]:
                    if draft_queues[request]:
                        candidate = draft_queues[request].pop(0)
                        if candidate == int(token):
                            runner.mtp_stats["accepted"] += 1
                        else:
                            runner.mtp_stats["rejected"] += 1
                            draft_queues[request].clear()
                    generated[request].append(int(token))
                    sequences[request].append(int(token))
                    finished[request] = int(token) in eos_ids
                    if not finished[request]:
                        remaining[request] = [int(token)]
            if spec_tokens and step + 1 < max_new_tokens:
                draft_slots = [
                    request
                    for request in range(len(prompts))
                    if not finished[request] and not draft_queues[request]
                ]
                if draft_slots:
                    count = min(spec_tokens, max_new_tokens - step - 1)
                    for slot, proposed in runner.propose_mtp(draft_slots, count).items():
                        draft_queues[slot] = proposed
            step_end = time.monotonic()
            step_log.append(
                {
                    "step": step,
                    "seconds": dispatch_seconds,
                    "input_setup_seconds": input_setup_seconds,
                    "end_to_end_seconds": step_end - tick,
                    "token_ids": next_ids,
                    "text": [tokenizer.decode(tokens, skip_special_tokens=False) for tokens in generated],
                },
            )
            print("GLM_GENERATION_STEP", json.dumps(step_log[-1], ensure_ascii=False), flush=True)
            if all(finished):
                break
        resident_bytes_per_rank = runner.weights.bytes_per_rank
        guard_events = runner.guard_events
    finally:
        runner.close()

    if dump_path is not None:
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(json.dumps(generated, indent=2) + "\n")
    if expect_tokens:
        for request, expected in expect_tokens.items():
            produced = generated[request][: len(expected)]
            if produced != list(expected):
                raise AssertionError(f"prompt {request}: expected top-1 prefix {list(expected)}, got {produced}")

    def _truncate_at_eos(tokens: list[int]) -> list[int]:
        stops = [index for index, token in enumerate(tokens) if token in eos_ids]
        return tokens[: stops[0]] if stops else tokens

    return {
        "platform": "a2a3",
        "tp": TP_SIZE,
        "ep": EP_SIZE,
        "dp": DP_SIZE,
        "precision": "official_bf16_w8a8",
        "checkpoint": str(checkpoint.root),
        "capacity": CAPACITY,
        "max_seq_len": MAX_SEQ_LEN,
        "cache_tokens": CACHE_TOKENS,
        "full_prefix_recomputation": False,
        "mtp": {"tokens": spec_tokens, "verification": "sequential_greedy", **runner.mtp_stats},
        "layers_per_forward": NUM_LAYERS,
        "launch_mode": "rt",
        "device_launches_per_rank_per_forward": 1,
        "main_forward_dispatches": runner.main_dispatches,
        "resident_bytes_per_rank": resident_bytes_per_rank,
        "seconds": time.monotonic() - start,
        "steps": step_log,
        "guard_events": guard_events,
        "checkpoint_setup_seconds": runner.checkpoint_setup_seconds,
        "end_to_end_seconds": time.monotonic() - end_to_end_start,
        "cases": [
            {
                "prompt": prompt,
                "token_ids": _truncate_at_eos(tokens),
                "text": tokenizer.decode(_truncate_at_eos(tokens), skip_special_tokens=False),
                "eos_reached": done,
            }
            for prompt, tokens, done in zip(prompts, generated, finished)
        ],
    }


def _load_programs(api, runtime_dir, device_ids, platform):
    """Load artifacts with the same explicit layout contract used at compilation."""
    from pypto.ir import DistributedConfig
    from pypto.ir.distributed_compiled_program import DistributedCompiledProgram
    from pypto_serving.model.common.compiler.l3_callable import L3Callable

    manifest = json.loads((runtime_dir / "manifest.json").read_text())
    geometry = {phase: api.forward_geometry(phase) for phase in api.FORWARD_PROGRAMS}
    if manifest.get("_mode") != "rt-packed" or manifest.get("_geometry") != geometry:
        raise ValueError("runtime artifact layout differs; recompile with the selected GLM geometry")
    if manifest.get("_head_backend") != "npu":
        raise ValueError("runtime artifact must include the NPU LM head; recompile the GLM programs")
    names = {name for name in manifest if not name.startswith("_")}
    if names != set(api.FORWARD_PROGRAMS):
        raise ValueError("runtime manifest must contain every GLM forward program")
    config = DistributedConfig(device_ids=device_ids, num_sub_workers=0)

    def load(name, path):
        compiled = DistributedCompiledProgram.from_dir(path, platform=platform, distributed_config=config)
        return L3Callable(compiled=compiled, name=f"glm53_{name}")

    programs = {name: load(name, manifest[name]) for name in api.FORWARD_PROGRAMS}
    programs["_auxiliary"] = {
        name: load(name, path) for name, path in manifest.get("_auxiliary", {}).items()
    }
    return programs


def _compile_programs(api, runtime_dir, device_ids, platform, *, enable_mtp=False):
    """Compile with the serving executor and persist only artifact locations."""
    programs = prepare_kernels(
        api, device_ids, platform=platform, pypto_build_dir=str(runtime_dir), enable_mtp=enable_mtp,
    )

    def artifact(callable_spec):
        compiled = callable_spec.compiled
        path = getattr(compiled, "output_dir", None) or getattr(compiled, "work_dir", None)
        if path is None:
            raise RuntimeError(f"compiled {callable_spec.name} has no artifact directory")
        return str(path)

    manifest = {
        "_mode": "rt-packed",
        "_geometry": {phase: api.forward_geometry(phase) for phase in api.FORWARD_PROGRAMS},
        "_head_backend": "npu",
    }
    manifest.update({name: artifact(programs[name]) for name in api.FORWARD_PROGRAMS})
    manifest["_auxiliary"] = {name: artifact(value) for name, value in programs["_auxiliary"].items()}
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return programs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=["a2a3"], default="a2a3")
    parser.add_argument("--checkpoint", type=Path, help="W8A8 checkpoint root")
    parser.add_argument("--runtime-dir", type=Path, required=True, help="compiled artifact manifest directory")
    parser.add_argument("-d", "--device", "--devices", dest="devices", default=os.environ.get("TASK_DEVICE"))
    parser.add_argument("--tp", type=int, choices=[8, 16], default=16)
    parser.add_argument("--ep", type=int, choices=[8, 16])
    parser.add_argument("--capacity", type=int)
    parser.add_argument("--max-seq-len", type=int)
    parser.add_argument("--cache-tokens", type=int)
    parser.add_argument("--tokens", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--sim", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--prompt", action="append")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--spec-tokens", type=int, choices=range(4), default=0)
    parser.add_argument("--image", action="append", type=Path)
    parser.add_argument("--video", action="append", type=Path)
    parser.add_argument("--vision-runtime-dir", type=Path)
    parser.add_argument("--video-frame", action="append", type=Path)
    parser.add_argument("--video-frame-fps", type=float, default=2.0)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--raw", action="store_true", help="encode prompts without a chat template")
    parser.add_argument("--expect-tokens", type=Path)
    parser.add_argument("--dump-tokens", type=Path)
    options = parser.parse_args(argv)
    if options.sim:
        parser.error("generation requires a2a3 devices")
    if not options.compile_only and options.checkpoint is None:
        parser.error("generation requires --checkpoint")
    ep_size = options.ep or options.tp
    api = import_kernel(
        "prefill_fwd", capacity=options.capacity, max_seq_len=options.max_seq_len,
        tp_size=options.tp, ep_size=ep_size, cache_tokens=options.cache_tokens, context_limit=None,
    )
    if options.tokens is not None and options.tokens != api.T_MOE:
        parser.error(f"--tokens must match the compiled per-rank capacity {api.T_MOE}")
    device_ids = [int(value) for value in (options.devices or "").split(",") if value]
    if options.compile_only:
        _compile_programs(api, options.runtime_dir, device_ids or list(range(api.RANKS)), options.platform,
                          enable_mtp=bool(options.spec_tokens))
        return
    if len(device_ids) != api.RANKS or len(set(device_ids)) != api.RANKS:
        parser.error(f"provide {api.RANKS} distinct task-allocated a2a3 devices")
    prompts = list(options.prompt or DEFAULT_PROMPTS)
    if not 1 <= len(prompts) <= api.BATCH or options.max_new_tokens <= 0:
        parser.error(f"provide 1..{api.BATCH} prompts and a positive --max-new-tokens")
    checkpoint = api.Checkpoint(options.checkpoint)
    torch.set_num_threads(8)
    required_auxiliary = {"reset"}
    if options.spec_tokens:
        required_auxiliary.update(("mtp_prepare", "mtp_head", "dsa_attn", "moe_ffn", "head"))
    programs = None
    if (options.runtime_dir / "manifest.json").exists():
        programs = _load_programs(api, options.runtime_dir, device_ids, options.platform)
        if not required_auxiliary <= programs["_auxiliary"].keys():
            programs = None
    if programs is None:
        programs = _compile_programs(api, options.runtime_dir, device_ids, options.platform,
                                     enable_mtp=bool(options.spec_tokens))
    visual_inputs = None
    if options.image or options.video or options.video_frame:
        vision_api = import_kernel(
            "vision", capacity=api.CAPACITY, max_seq_len=api.MAX_SEQ_LEN,
            tp_size=api.TP_SIZE, ep_size=api.EP_SIZE, cache_tokens=api.CACHE_TOKENS, context_limit=None,
        )
        processor = vision_api.VisionProcessor(checkpoint)
        visuals = [processor.image(image) for image in options.image or []]
        visuals.extend(processor.video_file(video) for video in options.video or [])
        if options.video_frame:
            if not math.isfinite(options.video_frame_fps) or options.video_frame_fps <= 0:
                parser.error("--video-frame-fps must be positive and finite")
            times = [index / options.video_frame_fps for index in range(len(options.video_frame))]
            visuals.append(processor.video(options.video_frame, times))
        visual_inputs = {0: visuals}
    if options.vision_runtime_dir is not None:
        from pypto.ir import DistributedConfig
        from pypto.ir.distributed_compiled_program import DistributedCompiledProgram

        programs["_vision"] = DistributedCompiledProgram.from_dir(
            options.vision_runtime_dir, platform=options.platform,
            distributed_config=DistributedConfig(device_ids=[device_ids[0]], num_sub_workers=0),
        )
    expected = None
    if options.expect_tokens is not None:
        raw = json.loads(options.expect_tokens.read_text())
        expected = {int(key): value for key, value in raw.items()} if isinstance(raw, dict) else dict(enumerate(raw))
    report = generate(
        api, checkpoint, programs, device_ids, prompts, options.max_new_tokens, raw_prompts=options.raw,
        platform=options.platform, expect_tokens=expected, dump_path=options.dump_tokens,
        visual_inputs=visual_inputs, spec_tokens=options.spec_tokens,
    )
    if options.report:
        options.report.parent.mkdir(parents=True, exist_ok=True)
        options.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("GLM_GENERATION_COMPLETE", json.dumps(report, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    main()

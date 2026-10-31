# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from types import SimpleNamespace
import ctypes
import json
import sys

import pytest
import torch

from pypto_serving.cli.main import build_parser, build_serving_engine_config
from pypto_serving.config.types import DecodeBatch, RuntimeConfig
from pypto_serving.serving.memory.kv_cache import KvCacheManager
from pypto_serving.model.glm53 import kernel_api, npu_executor, weight_loader
from pypto_serving.model.glm53.task_args import (
    forward_input_task_args, forward_task_args,
)
from pypto_serving.model.glm53.npu_runner import Glm53ModelRunner
from pypto_serving.model.model_family import detect_model_family
from pypto_serving.model.model_loader import ModelLoader
from pypto_serving.serving.engine.async_engine import AsyncLLMEngine, EngineConfig, ReplicaEngineCore
from pypto_serving.serving.sched.scheduler import Request
from pypto_serving.serving.server.ipc import NewRequestData, PrefillRequest
from pypto_serving.serving.server.serving_worker import WorkerProcess
from pypto_serving.serving.utils.prefill import pack_prefill_batch
from pypto_serving.model.common.executor.sampler import Sampler
from pypto_serving.model.common.compiler.l3_callable import L3Callable


@pytest.mark.parametrize("ranks", [8, 16])
def test_glm_family_topology_and_runtime_options(tmp_path, ranks):
    (tmp_path / "config.json").write_text('{"model_type":"glm5_next"}')
    assert detect_model_family({"architectures": ["Glm5NextForConditionalGeneration"]}) == "glm53"
    options = ["--model", str(tmp_path), "--devices", ",".join(map(str, range(ranks))),
               "--tp", str(ranks), "--ep", str(ranks), "--max-num-seqs", "4",
               "--max-model-len", "512", "--max-num-batched-tokens", "128",
               "--long-prefill-token-threshold", "32"]
    config = build_serving_engine_config(build_parser().parse_args(options))
    assert config.executor_cls == "PyptoGlm53Executor"
    assert (config.parallel_config.worker_group_size, config.parallel_config.num_replicas) == (ranks, 1)
    assert config.runtime_config.max_prefill_tokens_per_request == 32
    assert config.runtime_config.prefill_chunk_alignment == 16
    assert config.runtime_config.requires_homogeneous_prefill_decode
    assert config.enable_prefix_cache
    disabled = build_serving_engine_config(build_parser().parse_args(options + ["--no-enable-prefix-caching"]))
    assert not disabled.enable_prefix_cache
    assert tuple(group.name for group in config.runtime_config.kv_cache_groups) == ("attention", "kda_state")
    with pytest.raises(ValueError, match="max-num-seqs must be 1..16"):
        build_serving_engine_config(build_parser().parse_args(options + ["--max-num-seqs", "17"]))


def test_programmatic_engine_uses_overlapped_topology_and_compiled_limits():
    cores = []
    runtime = RuntimeConfig(max_batch_size=3, max_seq_len=512, max_num_batched_tokens=16)
    engine = AsyncLLMEngine(
        EngineConfig(model_id="glm", executor_cls="PyptoGlm53Executor", device_ids=tuple(range(8)),
                     runtime_config=runtime, max_num_running_reqs=32, max_num_scheduled_tokens=4096),
        tokenizer=SimpleNamespace(eos_token_id=2, bos_token_id=1),
        core_factory=lambda **kwargs: cores.append(kwargs["config"]) or SimpleNamespace(),
    )
    parallel = engine.parallel_config
    assert (parallel.tensor_parallel_size, parallel.expert_parallel_size,
            parallel.data_parallel_size, parallel.placement_mode) == (8, 8, 1, "overlapped")
    assert len(cores) == 1
    assert (cores[0].max_num_running_reqs, cores[0].max_num_scheduled_tokens) == (3, 16)


@pytest.mark.parametrize("context", [512, 16384])
def test_compiler_selects_each_phase_from_its_own_module(monkeypatch, context):
    prefill, decode, reset, run_config = (object() for _ in range(4))
    compiled, imports = [], []

    def compile_entry(name, entry):
        compiled.append((name, entry))
        return entry

    api = SimpleNamespace(
        CAPACITY=256, MAX_SEQ_LEN=context, CACHE_TOKENS=16 * context, TP_SIZE=8, EP_SIZE=8,
        l3_prefill_fwd=prefill, l3_reset_request_caches=reset,
    )

    def import_phase(name, **geometry):
        imports.append((name, geometry))
        return SimpleNamespace(l3_decode_fwd=decode)

    monkeypatch.setattr(npu_executor, "build_pypto_run_config", lambda **options: run_config)
    monkeypatch.setattr(npu_executor, "KernelCompiler", lambda **options: SimpleNamespace(compile=compile_entry))
    monkeypatch.setattr(npu_executor, "import_kernel", import_phase)
    programs = npu_executor.prepare_kernels(
        api, list(range(8)),
    )
    assert imports == [("decode_fwd", {
        "capacity": 256, "max_seq_len": context, "cache_tokens": 16 * context,
        "tp_size": 8, "ep_size": 8, "context_limit": None,
    })]
    assert compiled == [("glm53_prefill", prefill), ("glm53_decode", decode),
                        ("glm53_reset", reset)]
    assert programs == {"prefill": prefill, "decode": decode, "_auxiliary": {"reset": reset}}


@pytest.mark.parametrize("nested", [False, True])
def test_checkpoint_loader_reads_metadata_without_importing_kernels(tmp_path, monkeypatch, nested):
    text_config = {
        "vocab_size": 32, "hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 45,
        "num_attention_heads": 8, "kv_lora_rank": 4, "rms_norm_eps": 1e-6,
    }
    raw = {"model_type": "glm5_next", **({"text_config": text_config} if nested else text_config)}
    (tmp_path / "config.json").write_text(json.dumps(raw))
    (tmp_path / "quant_model_weights.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"model.norm.weight": "weights.safetensors"},
    }))
    (tmp_path / "quant_model_description.json").write_text("{}")
    monkeypatch.setenv("PYPTO_LIB_ROOT", str(tmp_path / "no-kernel-checkout"))
    monkeypatch.setattr(kernel_api, "import_kernel", lambda *args, **kwargs: pytest.fail(
        "the metadata loader must not select a kernel topology",
    ))
    monkeypatch.setattr(weight_loader, "load_tokenizer", lambda _root: SimpleNamespace(
        bos_token_id=None, eos_token_id=2, pad_token_id=0,
    ))
    original_argv = list(sys.argv)
    loaded = ModelLoader().load("glm", str(tmp_path))
    assert loaded.config.hidden_size == 8 and loaded.config.num_hidden_layers == 45
    assert loaded.runtime_model.embed_tokens.shape == (0, 8)
    assert loaded.runtime_model.runtime.max_seq_len == 512
    assert loaded.runtime_model.extra == {"family": "glm53", "model_dir": str(tmp_path.resolve())}
    runtime = RuntimeConfig(max_batch_size=4, max_seq_len=2048, max_num_batched_tokens=128)
    loaded = ModelLoader().load("glm", str(tmp_path), runtime_config=runtime)
    assert loaded.runtime_model.runtime is runtime
    assert sys.argv == original_argv


@pytest.mark.parametrize("ranks", [8, 16])
def test_executor_checks_weights_before_compilation_and_preserves_devices(tmp_path, monkeypatch, ranks):
    devices = tuple(range(16 - ranks, 16))
    checkpoint, weights, programs = object(), object(), {}
    calls = []

    def load_weights(actual_checkpoint, **options):
        calls.append("weights")
        assert actual_checkpoint is checkpoint
        assert (options["tp_size"], options["ep_size"], options["mla_mode"]) == (ranks, ranks, "native")
        return weights

    api = SimpleNamespace(Checkpoint=lambda _root: checkpoint, _prefill_model=lambda: object(),
                          ResidentPrefillWeights=SimpleNamespace(load=load_weights))
    monkeypatch.setattr(npu_executor, "import_kernel", lambda *args, **kwargs: api)

    def compile_programs(actual_api, actual_devices, **options):
        calls.append("compile")
        assert actual_api is api and tuple(actual_devices) == devices
        return programs

    monkeypatch.setattr(npu_executor, "prepare_kernels", compile_programs)
    executor = npu_executor.Glm53PyptoExecutor(device_ids=devices)
    model = SimpleNamespace(extra={"family": "glm53", "model_dir": str(tmp_path)}, runtime=RuntimeConfig())
    compiled = executor._compile_model(model)
    assert calls == ["weights", "compile"]
    assert compiled.checkpoint is checkpoint and compiled.weights is weights
    calls.clear()

    def reject_weights(*args, **kwargs):
        calls.append("weights")
        raise ValueError("invalid checkpoint")

    api.ResidentPrefillWeights.load = reject_weights
    with pytest.raises(ValueError, match="invalid checkpoint"):
        executor._compile_model(model)
    assert calls == ["weights"]


def _runner(ranks=8, *, requests=3, context=32, budget=8):
    """Run real staging/dispatch/reclaim against a small in-memory device."""
    runtime = RuntimeConfig(page_size=128, max_batch_size=requests,
                            max_seq_len=context, max_num_batched_tokens=budget,
                            kv_cache_groups=kernel_api.build_glm53_cache_group_specs(context, budget, tp_size=ranks))
    device = SimpleNamespace(runtime=runtime, inputs=[], resets=[], tokens={}, cached_states={},
                             failure=None, events=[], prepared_inputs=[], submissions=[], copy_contexts=[])

    def prepare_inputs(weights, sequences, *, kv_seq_lens, **options):
        if device.failure == "forward":
            raise RuntimeError("device failure")
        device.prepared_inputs.append([list(tokens) for tokens in sequences])
        tokens = torch.zeros(16, max(context, budget), dtype=torch.int32)
        sizes = torch.tensor([len(chunk) for chunk in sequences], dtype=torch.int32)
        for slot, chunk in enumerate(sequences):
            tokens[slot, :len(chunk)] = torch.tensor(chunk, dtype=torch.int32)
        return {"tokens": tokens, "sizes": sizes, "prior": kv_seq_lens.clone(),
                "state_rows": options["kda_state_rows"].clone()}

    class Worker:
        def __init__(self):
            self.inflight = 0
            self.in_callback = False

        def _check_control(self):
            if self.inflight and not self.in_callback:
                raise RuntimeError("device control requires the ordered run callback while a run is in flight")

        def copy_to(self, destination, source, size, *, worker_id):
            self._check_control()
            device.copy_contexts.append(self.in_callback)
            device.events.append("upload")
            ctypes.memmove(destination, source, size)

        def submit(self, compiled, token_rows, sizes, prior, state_rows, head, logits, *scalars, config):
            phase = program_names[compiled]
            device.events.append(("submit", phase))
            device.submissions.append((phase, token_rows, tuple(scalars)))
            self.inflight += 1
            # Public shared-tensor arguments are staged by the runtime's
            # generated HOST callback; the runner never opens this context.
            self.in_callback = True
            try:
                staged = []
                for value in (token_rows, sizes, prior, state_rows):
                    assert isinstance(value, torch.Tensor) and value.is_shared() and value.is_contiguous()
                    destination = torch.empty_like(value)
                    self.copy_to(destination.data_ptr(), value.data_ptr(), value.numel() * value.element_size(),
                                 worker_id=0)
                    staged.append(destination[0])
                token_values, size_values, prior_values, state_values = staged
                slots = size_values.nonzero().reshape(-1).tolist()
                chunks = [token_values[slot, :int(size_values[slot])].tolist() for slot in slots]
                priors = [int(prior_values[slot]) for slot in slots]
                device.inputs.append((chunks, slots, priors))
                head_output, logits_output = torch.ones_like(head), torch.zeros_like(logits)
                for slot, chunk, old_length in zip(slots, chunks, priors, strict=True):
                    source = int(state_values[slot, 0])
                    if source != slot:
                        device.tokens[slot] = list(device.cached_states[source])
                    assert len(device.tokens.get(slot, [])) == old_length
                    history = list(device.tokens.get(slot, []))
                    for index, destination in enumerate(state_values[slot, 2:].tolist()):
                        if destination >= 0:
                            device.cached_states[destination] = history + chunk[:min((index + 1) * 16, len(chunk))]
                    device.tokens.setdefault(slot, []).extend(chunk)
                    logits_output[0, slot, sum(device.tokens[slot]) % 32] = 10
                assert head.is_shared() and logits.is_shared()
                self.copy_stacked_from(SimpleNamespace(tensor=head_output), head)
                self.copy_stacked_from(SimpleNamespace(tensor=logits_output), logits)
            finally:
                self.in_callback = False

            def complete():
                device.events.append(("wait", phase))
                self.inflight -= 1
                if device.failure == "wait":
                    device.failure = None
                    raise RuntimeError("device completion failure")

            return SimpleNamespace(result=complete)

        def copy_stacked_from(self, source, destination):
            self._check_control()
            device.copy_contexts.append(self.in_callback)
            device.events.append("readback")
            destination.copy_(source.tensor)

        def close(self):
            self.inflight = 0

    api = SimpleNamespace(
        BATCH=16, RANKS=1, LOCAL_VOCAB=32, CONSTANT_STEP_INPUTS=set(), _PARAM_TO_STACK={},
        BLOCK_SIZE=128, CHUNK_TILE=16, KDA_STATE_COLUMNS=2 + (kernel_api.packed_capacity(budget) + 15) // 16,
        KDA_STATE_BLOCKS=runtime.kv_cache_groups[1].max_blocks_per_seq,
        prepare_step_inputs=prepare_inputs, forward_epoch_guard=lambda index: index,
        forward_query_tensor_names=lambda: (), forward_geometry=lambda phase: {"query_capacity": 16},
        forward_parameter_names=lambda phase, **options: ("tokens", "sizes", "prior", "state_rows", "head_hidden", "logits"),
        forward_tensor_layouts=lambda phase, **options: {
            **{name: (tuple(value.shape), value.dtype) for name, value in sample.items()},
            "head_hidden": ((16, 1), torch.float32), "logits": ((16, 32), torch.float32),
        },
    )
    programs = {phase: L3Callable(object(), phase) for phase in ("prefill", "decode")}
    program_names = {program.compiled: phase for phase, program in programs.items()}
    runner = Glm53ModelRunner(api, programs, SimpleNamespace(config={"vocab_size": 32}),
                              list(range(ranks)), "a2a3")
    pages = runner.init_kv_cache("glm", None, runtime)
    assert pages == runtime.kv_cache_groups[0].num_blocks
    runner.programs = programs
    runner.config, runner.weights, runner._dispatch_error = None, object(), None
    runner._seq_lens, runner._invalid_slots = torch.zeros(16, dtype=torch.int32), set()
    runner.cache_pages = SimpleNamespace(table=torch.full((16, (context + 127) // 128), -1, dtype=torch.int32))
    runner._local_cache = KvCacheManager(enable_prefix_cache=False)
    runner._local_cache.init_groups(runtime.kv_cache_groups, max_batch_size=requests)
    runner.calls, runner.main_dispatches, runner.guard_events = 0, 0, []
    runner._next_forward_epoch = {"prefill": 0, "decode": 0}
    runner._head_rows_by_slot = [torch.empty(1, 16, 1).share_memory_() for _ in range(2)]
    runner._logits_by_slot = [torch.empty(1, 16, 32).share_memory_() for _ in range(2)]
    runner._slot_per_rank, runner._phase_tensors = [], []
    sample = {"tokens": torch.zeros(16, max(context, budget), dtype=torch.int32),
              "sizes": torch.zeros(16, dtype=torch.int32), "prior": torch.zeros(16, dtype=torch.int32),
              "state_rows": torch.full((16, api.KDA_STATE_COLUMNS), -1, dtype=torch.int32)}
    for buffer_slot in range(2):
        phases = {}
        for phase in programs:
            args = forward_input_task_args(api, phase, sample)
            args.allocate_host_shared(None)
            runner._forward_input_args[buffer_slot, phase] = args
            tensors = {**args.tensors, "head_hidden": runner._head_rows_by_slot[buffer_slot],
                       "logits": runner._logits_by_slot[buffer_slot]}
            phases[phase] = tensors
            runner._forward_args[buffer_slot, phase] = forward_task_args(api, phase, tensors)
        runner._phase_tensors.append(phases)
    runner.runtime = runner._l3_worker = Worker()

    def reset(slots):
        device.resets.append(list(slots))
        for slot in slots:
            device.tokens.pop(slot, None)
        runner._seq_lens[slots] = 0

    runner.reset_requests = reset
    return runner, device


def _groups(request_id):
    owner = sum(request_id.encode()) % 16
    return {"attention": [owner], "kda_state": [owner]}


def _prefill(ids, chunks, starts, prompt_lengths):
    batch = pack_prefill_batch(request_ids=ids, token_chunks=chunks, chunk_starts=starts,
                              seq_lens=[start + len(chunk) for start, chunk in zip(starts, chunks, strict=True)],
                              prompt_lens=prompt_lengths, device="cpu")
    batch.block_ids_by_group = [_groups(request) for request in ids]
    return batch


def _decode(ids, tokens, lengths):
    return DecodeBatch(request_ids=ids, token_ids=torch.tensor(tokens).reshape(-1, 1), hidden_states=None,
                       seq_lens=torch.tensor(lengths, dtype=torch.int32),
                       block_ids_by_group=[_groups(request) for request in ids])


@pytest.mark.parametrize("ranks", [8, 16])
def test_runner_chunks_reorder_decode_and_reuse_without_recomputing_prefixes(ranks):
    runner, device = _runner(ranks)
    result = runner.run_prefill(None, _prefill(["a", "b"], [[20, 21], [5]], [0, 0], [4, 2]))
    assert result.logits.shape == (2, 32)
    assert device.inputs[-1] == ([[20, 21], [5]], [0, 1], [0, 0])
    runner.run_prefill(None, _prefill(["b", "a"], [[6], [22, 23]], [1, 2], [2, 4]))
    assert device.inputs[-1] == ([[22, 23], [6]], [0, 1], [2, 1])
    result = runner.run_decode(None, _decode(["b", "a"], [7, 24], [3, 5]))
    assert device.inputs[-1] == ([[24], [7]], [0, 1], [4, 2])
    assert result.logits.argmax(dim=-1).tolist() == [18, 14]
    runner.release_finished_requests(["a", "a"])
    assert device.resets[-1] == [0]
    assert device.tokens[1] == [5, 6, 7]
    runner.run_prefill(None, _prefill(["c"], [[9]], [0], [3]))
    assert device.inputs[-1] == ([[9]], [0], [0])
    runner.run_decode(None, _decode(["b"], [8], [4]))
    assert device.tokens == {0: [9], 1: [5, 6, 7, 8]}


def test_runner_rejects_unfinished_decode_and_does_not_commit_failed_forward():
    runner, device = _runner(context=4, budget=2)
    with pytest.raises(ValueError, match="max_num_batched_tokens"):
        runner.run_prefill(None, _prefill(["a", "b"], [[1, 2], [3]], [0, 0], [2, 1]))
    assert not device.inputs
    runner.run_prefill(None, _prefill(["a"], [[1, 2]], [0], [4]))
    with pytest.raises(ValueError, match="completed prefill"):
        runner.run_decode(None, _decode(["a"], [3], [3]))
    device.failure = "forward"
    with pytest.raises(RuntimeError, match="device failure"):
        runner.run_prefill(None, _prefill(["a"], [[3, 4]], [2], [4]))
    assert runner._requests["a"].committed_length == 2
    device.failure = None
    runner.run_prefill(None, _prefill(["a"], [[3, 4]], [2], [4]))
    assert runner._requests["a"].committed_length == 4


def test_worker_samples_only_terminal_glm_chunks():
    runner, device = _runner()
    worker = WorkerProcess(EngineConfig(executor_cls="PyptoGlm53Executor", device_ids=tuple(range(8))), None, None)
    worker.sampler = Sampler()
    worker.executor = npu_executor.Glm53PyptoExecutor(device_ids=range(8))
    worker.executor._runners["glm"] = runner
    worker.executor.lookup_embeddings = lambda model, tokens: torch.zeros(len(tokens), 8)
    model = SimpleNamespace(config=SimpleNamespace(model_id="glm"), runtime=RuntimeConfig())
    for request_id, tokens in [("a", [1, 2, 3]), ("b", [4, 5])]:
        worker._req_cache[request_id] = NewRequestData(
            request_id=request_id, prompt_token_ids=tokens, temperature=0, top_p=1, top_k=None,
        )
    output = {}
    worker._batch_prefill([
        PrefillRequest(request_id="a", chunk_tokens=[1], num_computed_tokens=0, block_ids=[0], block_ids_by_group=_groups("a")),
        PrefillRequest(request_id="b", chunk_tokens=[4, 5], num_computed_tokens=0, block_ids=[1], block_ids_by_group=_groups("b")),
    ], model, output)
    assert output == {"b": [9]}
    worker._batch_prefill([
        PrefillRequest(request_id="a", chunk_tokens=[2, 3], num_computed_tokens=1, block_ids=[0], block_ids_by_group=_groups("a")),
    ], model, output)
    assert output == {"b": [9], "a": [6]}
    assert device.tokens == {0: [1, 2, 3], 1: [4, 5]}


@pytest.mark.parametrize("ranks", [8, 16])
@pytest.mark.parametrize("prompt_length", [128, 160])
def test_cli_prefix_cache_leaves_a_prompt_suffix_to_compute(tmp_path, ranks, prompt_length):
    (tmp_path / "config.json").write_text('{"model_type":"glm5_next"}')
    config = build_serving_engine_config(build_parser().parse_args([
        "--model", str(tmp_path), "--devices", ",".join(map(str, range(ranks))),
        "--tp", str(ranks), "--ep", str(ranks), "--max-num-seqs", "1",
        "--max-model-len", "512", "--max-num-batched-tokens", "128",
        "--long-prefill-token-threshold", "128", "--enable-prefix-caching",
    ]))
    core = ReplicaEngineCore(config, SimpleNamespace(eos_token_id=None))
    manager, scheduler = core.kv_cache_manager, core.scheduler
    manager.initialize(config.runtime_config, num_blocks=config.runtime_config.kv_cache_groups[0].num_blocks,
                       group_num_blocks={group.name: group.num_blocks for group in config.runtime_config.kv_cache_groups})
    scheduler.add_request(Request("producer", [1] * 128, max_new_tokens=1))
    produced = scheduler.schedule()
    command = core._build_step_command(produced, [], 1)
    assert command.prefill_requests[0].block_ids_by_group
    scheduler.advance_after_schedule(produced)
    scheduler.update_from_output(produced, {"producer": [3]})

    suffix = [2] * (prompt_length - 128)
    scheduler.add_request(Request("consumer", [1] * 128 + suffix, max_new_tokens=1))
    restored = scheduler.schedule()
    command = core._build_step_command(restored, ["producer"], 2)
    request = command.prefill_requests[0]
    if suffix:
        assert request.num_computed_tokens == 128
        assert request.chunk_tokens == suffix
    else:
        assert request.num_computed_tokens == 0
        assert request.chunk_tokens == [1] * 128


def test_runner_prepares_metadata_then_submits_one_complete_forward_per_phase():
    runner, device = _runner()
    prepared = runner.prepare_decode(None, _decode(["a"], [0], [3]), buffer_slot=1)
    assert not device.events and not device.prepared_inputs and not runner._requests
    runner.run_prefill(None, _prefill(["a"], [[1, 2]], [0], [2]))
    assert (runner.main_dispatches, runner.calls) == (1, 1)

    device.events.clear()
    pending = runner.dispatch_prepared_decode(None, _decode(["a"], [11], [3]), prepared)
    assert device.inputs[-1] == ([[11]], [0], [2])
    assert device.events.count(("submit", "decode")) == 1 and ("wait", "decode") not in device.events
    assert all(device.copy_contexts)
    assert runner._seq_lens[0] == runner._requests["a"].committed_length == 2
    assert (runner.main_dispatches, runner.calls) == (2, 1)
    result = runner.reclaim_prepared_decode(pending)
    assert result.logits.argmax(-1).tolist() == [14]
    assert runner._seq_lens[0] == runner._requests["a"].committed_length == 3
    assert (runner.main_dispatches, runner.calls) == (2, 2)
    assert torch.equal(runner.reclaim_prepared_decode(pending).logits, result.logits)
    assert runner._requests["a"].committed_length == 3 and runner.calls == 2

    runner.run_prefill(None, _prefill(["b"], [[4]], [0], [1]))
    runner.run_decode(None, _decode(["b"], [12], [2]))
    assert [phase for phase, _, _ in device.submissions] == ["prefill", "decode", "prefill", "decode"]
    for phase in ("prefill", "decode"):
        tickets = [scalars for submitted, _, scalars in device.submissions if submitted == phase]
        assert len(set(tickets)) == 2 and tickets == sorted(tickets)
    assert (runner.main_dispatches, runner.calls) == (4, 4)


def test_runner_keeps_inflight_input_and_output_slots_independent():
    runner, device = _runner()
    runner.run_prefill(None, _prefill(["a", "b"], [[1], [5]], [0, 0], [1, 1]))
    batch_a, batch_b = _decode(["a"], [7], [2]), _decode(["b"], [9], [2])
    prepared_a = runner.prepare_decode(None, batch_a, buffer_slot=0)
    pending_a = runner.dispatch_prepared_decode(None, batch_a, prepared_a)
    submitted_a = device.submissions[-1][1].clone()
    source, destination = torch.ones(1), torch.zeros(1)
    with pytest.raises(RuntimeError, match="ordered run callback"):
        runner.runtime.copy_to(destination.data_ptr(), source.data_ptr(), source.element_size(), worker_id=0)
    waits_before = device.events.count(("wait", "decode"))
    prepared_b = runner.prepare_decode(None, batch_b, buffer_slot=1)
    pending_b = runner.dispatch_prepared_decode(None, batch_b, prepared_b)
    assert torch.equal(device.submissions[-2][1], submitted_a)
    assert runner.runtime.inflight == 2 and device.events.count(("wait", "decode")) == waits_before
    assert all(device.copy_contexts)
    assert runner._requests["a"].committed_length == runner._requests["b"].committed_length == 1
    result_b = runner.reclaim_prepared_decode(pending_b)
    assert runner.runtime.inflight == 1 and runner._requests["a"].committed_length == 1
    result_a = runner.reclaim_prepared_decode(pending_a)
    assert result_a.logits.argmax(-1).tolist() == [8]
    assert result_b.logits.argmax(-1).tolist() == [14]
    result_b.logits.fill_(-999)
    assert result_a.logits.argmax(-1).tolist() == [8]
    assert runner._seq_lens[:2].tolist() == [2, 2]


def test_runner_failed_reclaim_preserves_committed_lengths_and_blocks_reuse():
    runner, device = _runner()
    runner.run_prefill(None, _prefill(["a", "b"], [[1], [5]], [0, 0], [1, 1]))
    batch = _decode(["a"], [7], [2])
    prepared = runner.prepare_decode(None, batch, buffer_slot=0)
    pending = runner.dispatch_prepared_decode(None, batch, prepared)
    other = _decode(["b"], [9], [2])
    other_pending = runner.dispatch_prepared_decode(
        None, other, runner.prepare_decode(None, other, buffer_slot=1),
    )
    device.failure = "wait"
    with pytest.raises(RuntimeError, match="completion failure"):
        runner.reclaim_prepared_decode(pending)
    assert runner._seq_lens[:2].tolist() == [1, 1]
    assert runner._requests["a"].committed_length == runner._requests["b"].committed_length == 1
    assert runner.calls == 1 and runner.main_dispatches == 3
    assert device.events.count(("wait", "decode")) == 2
    with pytest.raises(RuntimeError, match="completion failure"):
        runner.reclaim_prepared_decode(pending)
    with pytest.raises(RuntimeError):
        runner.reclaim_prepared_decode(other_pending)
    with pytest.raises(RuntimeError):
        runner.dispatch_prepared_decode(None, batch, prepared)
    assert len(device.submissions) == 3 and runner._seq_lens[:2].tolist() == [1, 1]
    runner.close()


def test_failed_worker_close_keeps_arguments_alive_until_retry_succeeds():
    runner, device = _runner()
    runner.run_prefill(None, _prefill(["active"], [[1]], [0], [1]))
    batch = _decode(["active"], [2], [2])
    pending = runner.dispatch_prepared_decode(None, batch, runner.prepare_decode(None, batch, buffer_slot=0))
    worker = runner.runtime
    inputs, forward_args = dict(runner._forward_input_args), dict(runner._forward_args)
    device.events.clear()

    native_close = worker.close

    def close():
        device.events.append("worker_close")
        if device.events.count("worker_close") == 1:
            raise RuntimeError("incomplete cleanup")
        native_close()

    worker.close = close
    with pytest.raises(RuntimeError, match="incomplete cleanup"):
        runner.close()
    assert runner.runtime is worker and runner._l3_worker is worker
    assert runner._forward_input_args == inputs and all(args.tensors for args in inputs.values())
    assert runner._forward_args == forward_args
    assert runner._pending_forwards[0] is pending
    assert runner._requests and runner._free_slots
    with pytest.raises(RuntimeError, match="cleanup failed"):
        runner.open()
    with pytest.raises(RuntimeError, match="cleanup failed"):
        runner.forward_chunks([[1]], [0], [0])
    assert device.events == ["worker_close"]
    assert runner._requests["active"].committed_length == 1
    runner.close()
    assert device.events == ["worker_close", "worker_close"]
    assert all(not args.tensors for args in inputs.values())
    assert runner.runtime is None and runner._l3_worker is None and runner._input_args is None
    assert not runner._forward_args and not runner._requests
    assert not runner._input_args_by_slot and not runner._pending_forwards
    assert not runner._forward_input_args
    assert not runner._free_slots
    runner.close()
    assert device.events.count("worker_close") == 2


@pytest.mark.parametrize("ranks", [8, 16])
def test_grouped_prefix_shares_history_and_preserves_kda_seed(ranks):
    runner, device = _runner(ranks, context=1024, budget=256)
    cache = KvCacheManager(enable_prefix_cache=True)
    cache.init_groups(device.runtime.kv_cache_groups, max_batch_size=4)
    tokens = list(range(256))
    hashes = cache.compute_group_block_hashes(tokens)
    groups = cache.ensure_group_blocks("producer", 256)
    cache.retain_group_block_snapshot(groups, 0)
    batch = _prefill(["producer"], [tokens], [0], [257])
    batch.block_ids_by_group = [groups]
    runner.run_prefill(None, batch)
    # Device completion alone does not publish pages before the scheduler ACK.
    assert cache.acquire_group_prefix_blocks("early", hashes, max_cache_hit_tokens=128)[1] == 0
    cache.cache_group_blocks_from_snapshot(hashes, 256, {}, groups, 0, request_id="producer")
    cache.release_group_block_snapshot(groups, 0)

    restored, hit, partition = cache.acquire_group_prefix_blocks(
        "consumer", cache.compute_group_block_hashes(tokens[:128] + [500]), max_cache_hit_tokens=128,
    )
    assert hit == 128 and partition == 0
    assert restored["attention"] == groups["attention"][:1]
    state_row = runner._api.BATCH + restored["kda_state"][0]
    assert device.cached_states[state_row] == tokens[:128]
    saved = list(device.cached_states[state_row])
    consumer_groups = cache.ensure_group_blocks("consumer", 129)
    batch = _prefill(["consumer"], [[500]], [128], [129])
    batch.block_ids_by_group = [consumer_groups]
    result = runner.run_prefill(None, batch)
    assert device.inputs[-1][2] == [128]
    assert result.logits.argmax(-1).item() == (sum(saved) + 500) % 32
    assert device.cached_states[state_row] == saved

    decode = _decode(["consumer"], [501], [130])
    decode.block_ids_by_group = [cache.ensure_group_blocks("consumer", 130)]
    runner.run_decode(None, decode)
    assert device.cached_states[state_row] == saved
    assert device.tokens[0] == tokens  # Producer's working state remains independent.
    runner.close()


def test_decode_preparation_freezes_group_tables():
    runner, _ = _runner()
    runner.run_prefill(None, _prefill(["a"], [[1]], [0], [1]))
    batch = _decode(["a"], [2], [2])
    prepared = runner.prepare_decode(None, batch, buffer_slot=0)
    batch.block_ids_by_group[0]["attention"].clear()
    assert prepared.cache_groups[0]["attention"]
    result = runner.run_prepared_decode(None, batch, prepared)
    assert result.logits.argmax(-1).item() == 3
    runner.close()


def test_group_pools_report_independent_physical_capacities():
    runner, device = _runner()
    executor = npu_executor.Glm53PyptoExecutor(device_ids=range(8))
    executor._runners["glm"] = runner
    sizes = executor.cache_group_sizes("glm")
    assert sizes == {group.name: group.num_blocks for group in device.runtime.kv_cache_groups}
    cache = KvCacheManager()
    cache.initialize(device.runtime, num_blocks=sizes["attention"], group_num_blocks=sizes)
    assert cache.group_num_blocks("kda_state") == sizes["kda_state"]
    runner.close()

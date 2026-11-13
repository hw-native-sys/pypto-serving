# Qwen3-14B

PyPTO Serving supports Qwen3-14B through the bundled Qwen model loader, NPU executor, and PyPTO kernels. Use this path for single-device validation, offline tensor-parallel runs, and HTTP serving with one or more data-parallel replicas.

## Checkpoint

Use a local Hugging Face style Qwen3-14B checkpoint directory. The directory must contain `config.json`, tokenizer files, and model weight shards readable by the active Python environment.

## Offline Generation

```bash
pypto-serving \
  --model /path/to/Qwen3-14B \
  --prompt 'Huawei is' \
  --platform a2a3 \
  --device 0 \
  --max-model-len 512 \
  --generate-config '{"max_new_tokens": 32}'
```

For one tensor-parallel worker group, provide `--devices` and `--tp`:

```bash
pypto-serving \
  --model /path/to/Qwen3-14B \
  --prompt 'Huawei is' \
  --platform a2a3 \
  --devices 0,1 \
  --tp 2 \
  --max-model-len 512 \
  --generate-config '{"max_new_tokens": 32}'
```

Offline generate mode shares the serving engine. `--dp` creates independent replica engines, and repeated `--prompt` values are routed across them by the `least_pending_tokens` policy.

## HTTP Serving

```bash
pypto-serving \
  --model /path/to/Qwen3-14B \
  --backend npu \
  --platform a2a3 \
  --device 0 \
  --max-model-len 512 \
  --port 8899
```

Send a request after startup:

```bash
curl --noproxy "*" http://127.0.0.1:8899/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Huawei is","max_tokens":32,"temperature":0.0}'
```

## DP=2 Serving

Data parallel serving creates independent replicas and routes requests by the `least_pending_tokens` policy:

```bash
pypto-serving \
  --model /path/to/Qwen3-14B \
  --backend npu \
  --platform a2a3 \
  --devices 0,1 \
  --dp 2 \
  --tp 1 \
  --max-model-len 512 \
  --port 8899
```

## Weight Staging

The loader stages per-layer weights without keeping a second full model copy. Its host slabs use shared memory because upload reads them from a forked child. A checkpoint without QK norms is supported; the missing gamma weights default to ones. See [Weight Staging](../developer-guide/weight-staging.md) for the shared loading flow.

## Runtime Notes

Qwen uses the standard `pypto-serving` runtime capacity, generation, prefix caching, chunked prefill, and compile-cache controls. See [pypto-serving](../cli-reference/pypto-serving.md) for the command-line reference.

Prefix caching and chunked prefill are enabled by default for Qwen serving.

## Program Request Acceptance

`examples/qwen3_program_acceptance.py` checks complete requests through
`AsyncLLMEngine`, the scheduler and the model worker. It defaults to two rounds
of 16 concurrent requests, each with a frozen 3338-token prompt and 128 output
tokens. Python 3.10 or newer is required, matching Serving's package boundary.

Use the checkout's Lib sources and a source-built PyPTO with its matching
Simpler runtime pin. Submit the run through `task-submit` on an A3:

```bash
task-submit --device auto --max-time 1800 --timeout 1200 --run \
  'python examples/qwen3_program_acceptance.py \
    --model /path/to/Qwen3-14B --device "$TASK_DEVICE" \
    --runtime tensormap_and_ringbuffer \
    --prompt /path/to/prompt.txt --reference /path/to/reference.json \
    --reference-sha256 PINNED_SHA256 --output /path/to/new-result'
```

Select `--runtime host_build_graph` for HBG. `PYPTO_LIB_ROOT` can select an
explicit Lib checkout as in the HBG CI guard. The entry uses BF16 weights and
KV, 128-token pages, a 4096-token context, chunked prefill and synchronous
scheduling. TMR uses a 4096-token scheduling budget; HBG uses 80 with the
Serving worker's single active prefill row. `--requests 1 --rounds 1` provides
a smaller request cohort; `--max-tokens` accepts 2 through 128. The frozen
reference always describes the complete 16-row, 128-output token chain.

### Frozen Reference

Obtain the reference from an independent oracle or an authenticated existing
golden. Freeze the model and reference files for the duration of the run, and
pin the SHA-256 of the reference JSON separately from that file. The schema is
`qwen3-14b-program-tokens-v1`:

| Field | Required content |
| --- | --- |
| `schema` | `qwen3-14b-program-tokens-v1` |
| `versions` | Reference producer source, model and toolchain identities |
| `checkpoint_sha256` | Relative path to SHA-256 mapping for every visible checkpoint file, including tokenizer auxiliaries, the index and every indexed shard |
| `prompt_sha256` | SHA-256 of the exact UTF-8 prompt file bytes |
| `prompt_token_ids` | The 3338 IDs shared by the requests; the active tokenizer must reproduce these IDs |
| `first_generated_token_ids` | 16 tokens sampled after prefill |
| `decode_input_token_ids` | 127 rows of 16 nonnegative integer IDs |
| `decode_output_token_ids` | 127 rows of 16 nonnegative integer IDs; the inputs must equal the first-token row followed by the preceding output rows |

The entry authenticates inputs before constructing the engine. It requires the
real Qwen3 model configuration with 40 layers and hidden size 5120, rejects
unverified checkpoint assets and unsafe indexed shard paths, and verifies the
selected Program runtime. This reference is independent of the standalone KV
fixture format used by Simpler's runtime examples.

### Results and Failure Handling

The output directory must be new. `result.json` records source and package
identities, reference hashes, generated artifact hashes, complete or partial
outputs, and host-observed TTFT, TPOT and request latency. These timings include
scheduler work. They do not constitute a Clean performance benchmark.

Every streamed token, its counts and the final cumulative history must match
the reference. Request failure cancels and drains its cohort before checked
engine shutdown. Startup, request and artifact failures remain failures, and a
secondary shutdown error is retained without replacing the primary error.
`--complete-on-mismatch` drains every round for diagnosis and still fails the
gate. The build-directory environment override is restored after success or
failure.

`token_gate_passed` covers the observed request chain and confirmed shutdown.
Tensor/KV numerics, per-step private-address audits, child-loaded runtime
qualification and the Kernel eager/capture paths retain their separate
acceptance evidence. Six-cell acceptance remains explicitly unqualified by
this entry.

### Migration from Simpler #2948

The full-request consumer proposed in
[Simpler #2948](https://github.com/hw-native-sys/simpler/pull/2948) belongs to
Serving. Its original 42 passing CPU checks remain historical CPU evidence.
The replacement keeps the request contract and authenticates the inputs it
actually consumes using the pinned JSON reference. Standalone artifact and KV
fixture validation remains with the existing Simpler fixture producer and
loader; Serving builds its own Program artifacts and allocates its own KV.

| Contract | Serving replacement and permanent coverage |
| --- | --- |
| Authenticated checkpoint, tokenizer, prompt and complete token feedback | `load_reference()` / `prepare()`; `test_pinned_reference_authenticates_model_and_raw_prompt`, `test_prepare_rejects_unverified_inputs`, `test_reference_rejects_broken_feedback` |
| 3338-token input and selected runtime before engine construction | `test_reference_rejects_short_prompt`, `test_prepare_rejects_wrong_runtime_before_startup`, `test_prepare_selects_program_runtime_after_input_authentication` |
| Prefill token, every decode token, counts and terminal history | `collect_request()`; parameterized request failures and output-after-terminal tests |
| Two cohorts, independent request IDs and worker release | `test_actual_engine_runs_prefill_decode_and_acknowledges_worker_frees` drives the real engine/scheduler/IPC with a CPU transport stand-in |
| Startup/request failure, sibling cancellation and explicit abort | Request lifecycle regressions plus real engine decode-error and owner-routed abort tests |
| Checked shutdown and primary error preservation | Stop refusal/exception and primary-plus-cleanup regressions |
| Provenance, output protection and environment restoration | CLI success/error/artifact regressions and the installed-file provenance regression |
| Real model use of the shared token checker | Existing TMR/HBG `tests/test_qwen3_serving.py` guards retain their batching, chunking and prefix assertions; short-oracle requests also call `collect_request()` |

The CPU migration regressions live in
`tests/unit/serving/engine/test_program_acceptance.py` and run in the existing
Serving CPU CI suite. The standalone entry supports long-request qualification
with an externally frozen reference. Existing short hardware guards and a
long-request run are recorded as separate validation scopes.

# GLM-5.3-Flash

The GLM integration serves the 45-layer W8A8 text backbone on eight or sixteen
logical Ascend NPU devices. Use TP8/EP8/DP1 with eight devices, or TP16/EP16/DP1
with sixteen devices. Attention and dense FFNs use TP; MoE uses EP with complete
shared experts. These axes overlap on one worker group. Each rank owns 36 routed
experts with EP8, or 18 routed experts with EP16, out of 288 total experts.
Count logical device IDs, rather than physical accelerator boards: a dual-chip
board exposes two logical devices.

Set `PYPTO_LIB_ROOT` to an updated pypto-lib checkout with the consolidated
GLM kernels and the chosen TP/EP topology. The checkpoint reader is `models.glm5_3_flash.utils.Checkpoint`. The library exports the
L3 compute entries, tensor layouts and CPU checkpoint plan. Serving owns the
worker, device buffers, weight upload, cache lifecycle and program dispatch.
GLM requires the matching grouped-cache ABI from pypto-lib; recompile older
runtime artifacts before using the updated interface.
The model loader reads the published `quant_model_weights.safetensors.index.json`,
nested `text_config`, tokenizer, chat template, and generation EOS IDs, and checks
that the quantization description is present. As with the other model integrations,
kernel import and weight preparation belong to the executor and runner. The GLM
executor derives TP/EP from its worker device group and uses the kernel-owned
checkpoint reader to validate the quantization, model geometry and all planned
weight headers before compilation. The planner uses native W8A8 MLA projections,
matching the resident compute programs. Runner preflight then stages and uploads the
weights. Serving does not reimplement quantization or sharding transformations.

## Chunked prefill and request batching

Use `--max-num-seqs` between 1 and 16. The scheduler batches active requests
and admits waiting requests when earlier ones finish. Requests share the same
TP8 or TP16 worker group. Each request owns a stable cache/state slot, while
its row in a dispatched batch may change as other requests arrive or finish.
Weights and the compiled prefill/decode programs remain resident across requests.
The library owns the complete 45-layer orchestration loop. Each main forward
submits one L3 program; the NPU-head backend includes final normalization and
vocabulary projection in that program. Cache reset and the standalone MTP path retain their own auxiliary programs.

Chunked prefill is enabled by default. Each prompt chunk sends only its new
tokens, with the number of already committed tokens as its cache position.
MLA, Indexer and KDA state persist across chunks. Sampling runs only when the
last prompt chunk completes. Decode appends one new token per participating
request. Completed, cancelled and preempted requests release their ownership; private
working state is cleared before slot reuse. Prefill and decode use separate scheduler phases.

Prefill uses the compiled query capacity. Decode has a separate entry with
sixteen physical query rows, one per participating request; its MoE owner buffers
retain the capacity selected for the model. Both entries reuse the same backbone
computation and persistent cache/state slabs.

Decode follows the shared prepare/dispatch/reclaim interface. Preparation copies
request metadata without accessing the device or looking up embeddings. Dispatch
binds the latest sampled token IDs, performs CPU embedding lookup, stages one of
two input/output buffer sets, and submits without waiting for the result. Reclaim
waits for that submission, reads its own output buffer and commits request lengths
after success. A request's dispatch waits for its previous sample before binding
tokens; metadata preparation can overlap the previous device step. Buffer sets
remain owned until their submissions have been reclaimed.

Nonfinal prompt chunks end on 16-token boundaries so KDA keeps the same BF16
recurrence boundaries across dispatches. The scheduler rounds an unfinished
chunk down to a multiple of 16 and defers it when the remaining step budget is
smaller than 16. A final prompt tail can have any length; decode still appends
one token. Arbitrary splits such as 17 + 1 + 14 can change KDA's intermediate
rounding and are not used for serving.

Prefix caching is enabled by default through the shared grouped-cache allocator.
Use `--no-enable-prefix-caching` to disable reuse. GLM declares two groups: `attention` jointly
owns MLA latent pages, raw/pooled Indexer pages and scales; `kda_state` owns
rolling convolution and recurrent-state checkpoints. A hit requires every
group at the same 128-token boundary. Completed pages become reusable only
after the matching successful worker result; scheduled tables retain references
until their in-flight steps settle.

Attention pages are shared directly. KDA reads the matched immutable checkpoint,
updates a private working row, and writes new aligned checkpoints into its
assigned state pages. Continuing one request does not overwrite another
request's prefix state. The engine, scheduler and IPC use the same grouped
allocation, lookup and release paths as DeepSeek.

Only endpoints strictly shorter than the prompt are reused. Every admitted
request computes a nonempty suffix through prefill and the NPU LM head before
its first sample. No logits are cached. Generated-output replay after preemption
and grammar masking remain unsupported.

Sixteen private KDA working rows are separate from the grouped checkpoint pool.
The state ring covers the maximum prefill chunk plus in-flight boundaries, with
additional pool space for retention and queued steps. Its capacity is bounded by
the compiled query and context sizes; it does not reserve a full KV context per
saved prefix. The shared allocator controls page ownership and eviction, so an
older boundary is reusable only while all its required pages remain resident.

Embedding lookup remains on the CPU. The GLM runner executes NPU
final RMSNorm and full-vocabulary projection for its 16 returned rows. BF16
hidden rows and final norm weights are widened to FP32; normalization,
projection weights and logits remain FP32. Each rank stores one vocabulary
shard with zero padding to a multiple of 64 rows. The additional device weight allocation
is 317,718,528 bytes (303 MiB) per rank at TP8 or 159,383,552 bytes (152 MiB)
at TP16, excluding workspace and runtime overhead. The host strips each
shard's padding before concatenating the full logits.

The CPU LM-head reference remains in pypto-lib for numerical validation.
Serving always runs the NPU head and retains the full 32-row native hidden
readback and all-rank finiteness guard. Device weight size is logged as
`GLM_DEVICE_LM_HEAD_BYTES_PER_RANK`.
The serving engine owns sampling, termination and streaming. Speculation,
vision inputs and model-specific reasoning/tool parsing are not provided.

The new-token budget and the context bound are separate settings:

- `--max-model-len` sets each request's persistent context capacity, up to 8192.
- `--max-num-batched-tokens` bounds the sum of new tokens in one scheduler step.
  The compiled query capacity rounds this budget up to a multiple of 64 rows.
  The budget must be between 16 and 8177 when the context bound is at least 16;
  smaller contexts require enough budget
  for the complete context.
- `--long-prefill-token-threshold` further limits each request's prompt chunk
  when positive. The final chunk may be smaller. Disabling chunked prefill
  requires each complete prompt to fit in one dispatch.
  Positive thresholds and per-request limits must fit at least 16 tokens, or
  the complete context when its bound is smaller. Smaller limits are rejected
  before loading weights because they cannot advance an unfinished KDA chunk.

For a 128-token step budget and 512-token context bound, query capacity is 128.
MoE owner capacity is 16 rows with either TP8 or TP16. Cache storage is
sized independently for the request slots and their context bound. Scheduler
page counts limit admission; the GLM runner owns the physical cache slabs.

## Offline generation

Run through the machine's NPU allocator with all eight or sixteen selected
devices reserved. The following command assumes `TASK_DEVICE` contains eight
allocated logical device IDs:

```bash
pypto-serving \
  --model "$GLM_CHECKPOINT" --platform a2a3 --devices "$TASK_DEVICE" \
  --tp 8 --ep 8 --dp 1 --max-num-seqs 4 \
  --max-model-len 512 --max-num-batched-tokens 128 \
  --long-prefill-token-threshold 32 \
  --prompt 'The capital of France is' \
  --prompt 'The capital of Germany is' \
  --generate-config '{"max_new_tokens": 16}'
```

For sixteen devices, use `--tp 16 --ep 16` and provide sixteen IDs instead.
Each topology compiles its own programs and uses a separate worker process;
kernel TP/EP, query-capacity and context dimensions are fixed when imported. Use separate build directories
when retaining compiled artifacts for both topologies. EP8 places roughly twice
as many routed-expert weights on each device as EP16, so reserve room for
weights, activation and communication buffers, and the configured scratch
capacity on every selected device.

Offline prompts are raw completions. The chat endpoint applies the checkpoint's
official chat template, whose generation prompt opens a reasoning segment.
Short generation limits may therefore stop before the final answer.

## HTTP and streaming

Start the same command without `--prompt`, adding `--port 8899`. Model weights
finish loading before the worker reports ready. The initial load can take
several minutes.

```bash
curl --noproxy '*' http://127.0.0.1:8899/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"The capital of France is","max_tokens":16,"temperature":0,"stream":true}'
```

The existing `/v1/chat/completions` endpoint also accepts text messages. Chat
stops on any published EOS ID unless `ignore_eos` is enabled. The raw
`/v1/completions` route ignores EOS, matching the existing server behavior.
Completed or cancelled requests release their runner-owned cache and state slots.

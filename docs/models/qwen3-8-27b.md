# Qwen3.8-27B W8A8

Qwen3.8-27B runs its 64-layer hybrid Gated DeltaNet and full-attention text
model on one Ascend A2/A3 NPU per replica. Tensor and expert parallelism must
both be one. Vision and speculative decoding are not supported by this path.

## Checkpoint and kernels

Use the Eco-Tech Qwen3.8-27B W8A8 checkpoint layout: `config.json`, tokenizer
files, `quant_model_weights.safetensors.index.json`, and the referenced shards.
The loader checks the fixed text dimensions and required weight names before
staging the shared, GDN and attention weights in their respective layer orders.
See [Weight Staging](../developer-guide/weight-staging.md) for the shared pipeline.

The pypto-lib checkout must provide the Qwen3.8 serving capability manifest
and the `build_l3_kernel` entries in `prefill_text_fwd.py`, `decode_fwd.py`,
`embedding.py` and `rms_lm_head.py`. These entries own the HOST signatures and
compose the same operators used by standalone validation. Serving owns weight
staging, request metadata, cache allocation and dispatch.

Initialize the bundled submodule as described in
[Installation](../get-started/installation.md). To select another compatible
checkout, set `PYPTO_LIB_ROOT=/path/to/pypto-lib` before starting the process.
After changing kernel sources or dispatch shapes, use a new build directory
or leave `--use-compile-cache` disabled; the compile cache does not fingerprint
sources.

## Offline generation

The following configuration uses the tested two-request capacity, 4096-token
context and 2048-token prefill extent:

```bash
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

pypto-serving \
  --model /path/to/Qwen3.8-27B-w8a8 \
  --platform a2a3 \
  --device 0 \
  --tp 1 --ep 1 \
  --max-model-len 4096 \
  --max-num-seqs 2 \
  --max-num-batched-tokens 2048 \
  --long-prefill-token-threshold 2048 \
  --block-size 128 \
  --no-enable-prefix-caching \
  --prompt 'The capital of France is' \
  --generate-config '{"max_new_tokens": 8, "temperature": 0.0}'
```

Set thread limits before launch: workers are spawned processes and do not
inherit a parent Python client's `torch.set_num_threads` setting. Sixteen
threads is the setting used in the validated workload; tune it for other
machines and workloads.

## HTTP serving

Use the same model and capacity options, omit `--prompt` and
`--generate-config`, and add `--host 127.0.0.1 --port 8000`. After startup:

```bash
curl --noproxy '*' http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"The capital of France is","max_tokens":8,"temperature":0.0}'
```

See [Online Serving](../user-guide/online-serving.md) and the
[CLI reference](../cli-reference/pypto-serving.md) for request formats and
runtime controls. Allocate a device through your site's scheduler before
running either command.

## Shape and cache constraints

The kernel capability manifest supplies the prefill limits and scheduling
requirements. A fused prefill extent is a multiple of 2048 tokens, up to
32768; 2048 is the extent validated through the complete serving integration.
Larger extents need their own correctness and memory checks. Decode pads the
batch to a multiple of 16 and supports at most 128 rows; wider serving batches
also need device validation beyond the tested two-request configuration.

Attention uses 128-token BF16 pages. GDN uses separate BF16 convolution and
FP32 recurrent-state pools, with one state block per live request. Both cache
groups are allocated for the configured concurrency and context limit; reduce
those limits if allocation exceeds the device memory budget. Mutable GDN state
must not be shared between requests as a cached prefix.

Prefill and decode run in separate scheduling steps. Full prefill chunks are
preserved when another request leaves only a partial token budget. A fused
launch consists of complete 256-token request spans whose starting positions
also align to 256 tokens. Each owner keeps its own RoPE positions, attention
pages and GDN state. Entries are compiled for each possible owner count, up to
the smaller of the batch limit and the prefill extent divided by 256.

Unaligned tails, or spans that cannot fill a fused launch, use exact one-token
decode steps. Synthetic padding must not advance live recurrent state.
Intermediate prompt tokens skip vocabulary logits when they would be discarded.
A generated token uses three NPU programs: embedding, the complete decoder,
and final normalization with vocabulary projection. Model arithmetic and cache
updates run on the NPU; the CPU prepares metadata and samples tokens.

## Validation scope

The real-checkpoint vendor comparison covers raw prompt lengths 5, 5, 2048
and 2050, eight greedy output tokens per request, concurrency two, a 4096-token
context, and disabled prefix caching. Generated token IDs match vLLM-ascend
v0.23.0 on that fixture. This is not a full-logit comparison with vLLM or a
language-quality evaluation.

Additional PyPTO checks compare full logits and cache contents across fresh
and resumed prefill, packed owners, reordered requests, changing batch sizes,
state-slot reuse and untouched storage. Boundary prompts cover lengths 1, 2,
15, 16, 17, 255, 256, 257, 2049 and 2051. The composed operators also pass their
random-weight and checkpoint-weight numerical reference gates. These checks
do not extend the vendor-equivalence claim to other workloads.

Measure throughput and time to first token separately using the same
checkpoint, device, request settings and CPU thread configuration for both
engines.

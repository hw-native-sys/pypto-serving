# GLM-5.3-Flash

The GLM integration serves the 45-layer W8A8 text backbone on eight or sixteen
logical Ascend NPU devices. Use the published W8A8 checkpoint and set
`PYPTO_LIB_ROOT` to the matching pypto-lib checkout.

Direct Python callers prepare the model runtime with
`build_glm53_runtime_config()` before passing it to `EngineConfig`.

## Supported configuration

- Eight devices use overlapped TP8/EP8/DP1; sixteen use TP16/EP16/DP1.
- `--max-num-seqs` supports 1 to 32 concurrent requests.
- `--max-model-len` supports a context bound up to 8192 tokens.
- The checkpoint path requires `--block-size 640` and `--dtype bfloat16`,
  which are selected by default for GLM.
- Serving supports text generation. Speculative decoding, vision inputs and
  model-specific reasoning/tool parsing are not provided.

Count logical device IDs: a dual-chip board exposes two logical devices.

## Runtime settings

`--max-num-batched-tokens` limits the total new tokens processed in one scheduler
step, up to 8177. Positive `--long-prefill-token-threshold` values further limit
one request's prompt chunk. Both limits must allow at least 16 tokens, or the
complete context when `--max-model-len` is smaller than 16.

Chunked prefill is enabled by default. Nonfinal prompt chunks are rounded down
to a multiple of 16; the final chunk can have any length. Disabling chunked
prefill requires the complete prompt to fit in one dispatch.

Prefix caching is enabled by default. Reuse requires a matching cached prefix
at a 640-token boundary that is shorter than the prompt. Use
`--no-enable-prefix-caching` to disable it.

## Offline generation

Reserve all eight or sixteen devices through the machine's NPU allocator.
The following command assumes `TASK_DEVICE` contains eight allocated logical
device IDs and `GLM_CHECKPOINT` points to the published W8A8 checkpoint:

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

For sixteen devices, use `--tp 16 --ep 16` and provide sixteen device IDs.

Offline prompts are raw completions. The chat endpoint applies the checkpoint's
chat template, whose generation prompt opens a reasoning segment. Short
output limits may therefore stop before the final answer.

## HTTP and streaming

Start the same command without `--prompt`, adding `--port 8899`. Wait for
model weights to finish loading and the worker to report ready.

```bash
curl --noproxy '*' http://127.0.0.1:8899/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"The capital of France is","max_tokens":16,"temperature":0,"stream":true}'
```

The `/v1/chat/completions` endpoint also accepts text messages. Chat stops on
any published EOS ID unless `ignore_eos` is enabled. The raw `/v1/completions`
route ignores EOS, matching the existing server behavior.

The complete-forward ABI uses 32 request slots, at least 192 query rows,
640-token cache pages, packed INT8 MLA latents and BF16 pooled index keys.
The vocabulary projection retains BF16 weights and BF16 result rounding.

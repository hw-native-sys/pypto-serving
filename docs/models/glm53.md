# GLM-5.3-Flash

The GLM integration serves the 45-layer W8A8 text backbone on 16 Ascend NPUs.
Attention and dense FFNs use TP16; MoE uses EP16 with complete shared experts.
These axes overlap on one worker group, with DP1.

Set `PYPTO_LIB_ROOT` to a pypto-lib checkout exposing
`models.glm5_3_flash.prefill_fwd.ResidentPrefillSession`. The loader reads the
published `quant_model_weights.safetensors.index.json`, nested `text_config`,
quantization description, tokenizer, chat template, and generation EOS IDs.
The kernel-owned checkpoint reader validates the geometry and stages weights;
Serving does not reimplement its quantization or sharding transformations.
GLM uses this external kernel checkout; the repository's pinned `pypto-lib`
revision continues to serve the other model integrations.

## Initial execution mode

This version supports one active request (`--max-num-seqs 1`). Waiting requests
are queued. The runner fills the kernel's other 15 batch slots with one valid
token each and returns only the real request's logits. Weights and the six
compiled programs remain resident across requests.

Each generation step recomputes the full prefix with freshly cleared MLA,
Indexer, and KDA scratch caches. Embedding lookup, final RMSNorm, and LM-head
remain on the CPU. The serving engine owns sampling, termination, and streaming.
Prefix caching and chunked prefill are disabled for this mode; speculation,
vision inputs, and model-specific reasoning/tool parsing are not provided.
Scheduler page counts represent request admission capacity, not retained device
KV pages. The physical scratch allocation is fixed by the compiled geometry.

The packed kernel capacity is the sequence limit plus 15 inactive tokens,
rounded up to 64 rows, and cannot exceed 8192. Set the scheduler token budget at
least as high as `--max-model-len`.

## Offline generation

Run through the machine's NPU allocator with all 16 devices reserved. The
following command assumes `TASK_DEVICE` contains the allocated device IDs:

```bash
pypto-serving \
  --model "$GLM_CHECKPOINT" --platform a2a3 --devices "$TASK_DEVICE" \
  --tp 16 --ep 16 --dp 1 --max-num-seqs 1 \
  --max-model-len 512 --max-num-batched-tokens 512 \
  --prompt 'The capital of France is' \
  --generate-config '{"max_new_tokens": 16}'
```

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
Completed or cancelled requests release their runner-owned token histories.
